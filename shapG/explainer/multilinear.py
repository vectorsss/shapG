"""
Multilinear extension-based Shapley value computation using Owen (1972) theory.

This module provides an explainer that uses multilinear extensions for
computing Shapley values through sampled partial derivative integration.

Key insight: Instead of enumerating all 2^n coalitions, we:
1. Sample partial derivatives at quadrature points along the diagonal
2. Integrate numerically to get Shapley values

Complexity: O(n × n_quadrature × n_samples) instead of O(2^n)

Reference:
    Owen, G. (1972). Multilinear Extensions of Games.
    Management Science, 18(5-Part-2), 64-79.
"""

from typing import Dict, Union, Optional, Set, List, Tuple, Callable
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import time

from scipy.integrate import simpson

from .base import GraphExplainer, CharacteristicFunction


class MultilinearExplainer(GraphExplainer):
    """
    Shapley value computation using Owen's multilinear extension.

    This explainer uses the multilinear extension framework from Owen (1972)
    to compute Shapley values via sampled partial derivative integration.

    The multilinear extension f(x) of a game v is defined as:
        f(x) = sum_{S in N} [prod_{j in S} x_j * prod_{j not in S} (1-x_j)] * v(S)

    The Shapley value is obtained by integrating partial derivatives along
    the main diagonal (Owen's Theorem 5):
        phi_i = integral_0^1 (df/dx_i)(t, t, ..., t) dt

    Algorithm:
        For each feature i:
            1. Sample f_i(t, t, ..., t) at quadrature points t in [0, 1]
            2. Integrate: phi_i = integral_0^1 f_i(t, ..., t) dt

    Complexity: O(n × n_quadrature × n_samples) characteristic function calls

    The explainer can work in two modes:
    1. Local: Compute Shapley values for local neighborhoods (like ShapG)
    2. Global: Compute for all features at once
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        depth: int = 1,
        max_exact_size: int = 10,
        n_quadrature: int = 21,
        n_samples: int = 100,
        verbose: bool = False
    ):
        """
        Initialize the multilinear explainer.

        Args:
            characteristic_function: Function to compute coalition values
            depth: Depth for local neighborhood computation (0 = node only, 1 = neighbors, etc.)
            max_exact_size: Maximum coalition size for exact computation
            n_quadrature: Number of quadrature points for integration (odd number preferred)
            n_samples: Number of samples per quadrature point
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function, verbose)

        if depth < 0:
            raise ValueError(f"Depth must be non-negative, got {depth}")
        if max_exact_size <= 0:
            raise ValueError(f"Max exact size must be positive, got {max_exact_size}")
        if n_quadrature < 3:
            raise ValueError(f"n_quadrature must be >= 3, got {n_quadrature}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        self.depth = depth
        self.max_exact_size = max_exact_size
        self.n_quadrature = n_quadrature
        self.n_samples = n_samples

        # Results storage
        self.shapley_values = {}
        self.computation_times = {}
        self.method_used = {}
        self.cache_stats = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'MultilinearExplainer':
        """
        Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            from ..utils.graph_construction import GraphBuilder
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True
        self.n_nodes = self.graph.number_of_nodes()

        if self.verbose:
            print(f"MultilinearExplainer fitted to graph with {self.n_nodes} nodes")

        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        mode: str = 'local',
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute Shapley values using multilinear extension.

        Args:
            X: Optional input data (uses fitted data if None)
            mode: 'local' for neighborhood-based, 'global' for all nodes at once
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping node/feature indices to Shapley values
        """
        if not self._fitted:
            raise ValueError("Explainer must be fitted before calling explain()")

        if X is not None:
            self.fit(X, **kwargs)

        if mode == 'local':
            return self._explain_local()
        elif mode == 'global':
            return self._explain_global()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'local' or 'global'")

    def _explain_local(self) -> Dict[int, float]:
        """
        Compute Shapley values using local neighborhoods.

        For each node, we compute Shapley value within its local neighborhood
        using the multilinear extension approach.
        """
        shapley_values = {}

        if self.verbose:
            print(f"\nComputing Shapley values using multilinear extension (local mode)")
            print(f"Depth: {self.depth}, n_quadrature: {self.n_quadrature}, n_samples: {self.n_samples}")

        total_char_func_calls = 0

        for node in self.graph.nodes():
            start_time = time.time()

            # Get local neighborhood
            neighborhood = self._get_neighborhood(node)
            neighborhood_size = len(neighborhood)

            # Create local coalition mapping
            local_to_global = {i: n for i, n in enumerate(neighborhood)}
            global_to_local = {n: i for i, n in enumerate(neighborhood)}

            # Create coalition value cache for this neighborhood
            value_cache = {}

            def local_char_func(coalition: Set[int]) -> float:
                """Characteristic function for local coalition."""
                coalition_tuple = tuple(sorted(coalition))
                if coalition_tuple not in value_cache:
                    if not coalition:
                        value_cache[coalition_tuple] = 0.0
                    else:
                        global_coalition = {local_to_global[i] for i in coalition}
                        value_cache[coalition_tuple] = self.characteristic_function(
                            global_coalition, self.graph
                        )
                return value_cache[coalition_tuple]

            # Choose method based on neighborhood size
            if neighborhood_size <= self.max_exact_size:
                local_shapley = self._compute_shapley_exact(
                    local_char_func, neighborhood_size, self.n_quadrature
                )
                self.method_used[node] = 'exact'
            else:
                local_shapley = self._compute_shapley_sampled(
                    local_char_func, neighborhood_size,
                    self.n_quadrature, self.n_samples
                )
                self.method_used[node] = 'sampled_integration'

            # Extract Shapley value for the target node
            target_local_idx = global_to_local[node]
            shapley_values[node] = local_shapley[target_local_idx]

            # Track statistics
            self.computation_times[node] = time.time() - start_time
            self.cache_stats[node] = len(value_cache)
            total_char_func_calls += len(value_cache)

            # Progress reporting
            if self.verbose:
                nodes_processed = len(self.computation_times)
                print(f"  Processed {nodes_processed}/{self.n_nodes}: "
                      f"node {node}, size={neighborhood_size}, "
                      f"method={self.method_used[node]}, "
                      f"time={self.computation_times[node]:.3f}s")

        self.shapley_values = shapley_values

        if self.verbose:
            total_time = sum(self.computation_times.values())
            method_counts = {}
            for method in self.method_used.values():
                method_counts[method] = method_counts.get(method, 0) + 1

            print(f"\nMultilinear computation complete:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average time per node: {total_time/self.n_nodes:.3f}s")
            print(f"  Total char. function calls: {total_char_func_calls}")
            print(f"  Methods used: {method_counts}")

        return shapley_values

    def _explain_global(self) -> Dict[int, float]:
        """
        Compute Shapley values for all nodes at once.

        Uses sampled integration for the entire graph.
        """
        if self.verbose:
            print(f"\nComputing global Shapley values using multilinear extension")
            print(f"Graph size: {self.n_nodes} nodes")
            print(f"n_quadrature: {self.n_quadrature}, n_samples: {self.n_samples}")

        start_time = time.time()

        # Create characteristic function wrapper
        nodes = sorted(self.graph.nodes())
        idx_to_node = {i: n for i, n in enumerate(nodes)}
        value_cache = {}

        def char_func_wrapper(coalition: Set[int]) -> float:
            coalition_tuple = tuple(sorted(coalition))
            if coalition_tuple not in value_cache:
                if not coalition:
                    value_cache[coalition_tuple] = 0.0
                else:
                    global_coalition = {idx_to_node[i] for i in coalition}
                    value_cache[coalition_tuple] = self.characteristic_function(
                        global_coalition, self.graph
                    )
            return value_cache[coalition_tuple]

        # Compute Shapley values
        if self.n_nodes <= self.max_exact_size:
            shapley_array = self._compute_shapley_exact(
                char_func_wrapper, self.n_nodes, self.n_quadrature
            )
            method = 'exact'
        else:
            shapley_array = self._compute_shapley_sampled(
                char_func_wrapper, self.n_nodes,
                self.n_quadrature, self.n_samples
            )
            method = 'sampled_integration'

        # Convert to dictionary
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        shapley_values = {node: shapley_array[node_to_idx[node]] for node in nodes}

        total_time = time.time() - start_time

        # Store results
        self.shapley_values = shapley_values
        for node in nodes:
            self.computation_times[node] = total_time / self.n_nodes
            self.method_used[node] = method

        if self.verbose:
            print(f"  Method: {method}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Char. function calls: {len(value_cache)}")

        return shapley_values

    # =========================================================================
    # Core Multilinear Extension Methods
    # =========================================================================

    def _compute_shapley_exact(
        self,
        char_func: Callable[[Set[int]], float],
        n: int,
        n_quadrature: int
    ) -> np.ndarray:
        """
        Compute exact Shapley values using Owen's diagonal integration.

        Uses exact partial derivative computation with numerical integration.
        Complexity: O(n × n_quadrature × 2^(n-1))

        Args:
            char_func: Characteristic function
            n: Number of players
            n_quadrature: Number of quadrature points

        Returns:
            Array of Shapley values
        """
        shapley_values = np.zeros(n)

        # Use odd number for Simpson's rule
        if n_quadrature % 2 == 0:
            n_quadrature += 1
        t_values = np.linspace(0, 1, n_quadrature)

        for i in range(n):
            f_values = np.zeros(n_quadrature)
            for k, t in enumerate(t_values):
                f_values[k] = self._partial_derivative_exact(char_func, i, t, n)
            shapley_values[i] = simpson(f_values, x=t_values)

        return shapley_values

    def _partial_derivative_exact(
        self,
        char_func: Callable[[Set[int]], float],
        i: int,
        t: float,
        n: int
    ) -> float:
        """
        Exact computation of partial derivative f_i(t, t, ..., t).

        From Owen's equation (8):
        f_i(x) = sum_{S not containing i} [prod_{j in S} x_j * prod_{j not in S,i} (1-x_j)]
                 * [v(S union {i}) - v(S)]

        Complexity: O(2^(n-1))

        Args:
            char_func: Characteristic function
            i: Player index
            t: Diagonal parameter
            n: Number of players

        Returns:
            Partial derivative value
        """
        total = 0.0
        other_players = [j for j in range(n) if j != i]

        for size in range(len(other_players) + 1):
            for coalition in combinations(other_players, size):
                coalition_set = set(coalition)
                coalition_with_i = coalition_set | {i}

                # Compute probability: t^|S| * (1-t)^(n-1-|S|)
                prob = (t ** size) * ((1 - t) ** (len(other_players) - size))

                # Marginal contribution
                marginal = char_func(coalition_with_i) - char_func(coalition_set)
                total += prob * marginal

        return total

    def _compute_shapley_sampled(
        self,
        char_func: Callable[[Set[int]], float],
        n: int,
        n_quadrature: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Compute Shapley values via sampled partial derivative integration.

        This implements Owen's Theorem 5 using:
        1. Numerical quadrature along the diagonal t in [0, 1]
        2. Monte Carlo sampling to estimate f_i(t, ..., t) at each point

        Complexity: O(n × n_quadrature × n_samples)

        Args:
            char_func: Characteristic function
            n: Number of players
            n_quadrature: Number of quadrature points
            n_samples: Number of samples per quadrature point

        Returns:
            Array of Shapley values
        """
        shapley_values = np.zeros(n)

        # Use odd number for Simpson's rule
        if n_quadrature % 2 == 0:
            n_quadrature += 1
        t_values = np.linspace(0, 1, n_quadrature)

        for i in range(n):
            f_values = np.zeros(n_quadrature)
            for k, t in enumerate(t_values):
                f_values[k] = self._sample_partial_derivative(char_func, i, t, n, n_samples)
            shapley_values[i] = simpson(f_values, x=t_values)

        return shapley_values

    def _sample_partial_derivative(
        self,
        char_func: Callable[[Set[int]], float],
        i: int,
        t: float,
        n: int,
        n_samples: int
    ) -> float:
        """
        Estimate partial derivative f_i(t, t, ..., t) using sampling.

        At point (t, t, ..., t) on the diagonal, each player j (j != i) joins
        the coalition independently with probability t. We sample coalitions
        from this distribution and compute the expected marginal contribution.

        This directly implements Owen's equation (8) via Monte Carlo:
        f_i(t,...,t) = E_{S ~ Bernoulli(t)} [v(S union {i}) - v(S)]

        Args:
            char_func: Characteristic function
            i: Target player index
            t: Diagonal parameter (probability each other player joins)
            n: Number of players
            n_samples: Number of coalition samples

        Returns:
            Estimated partial derivative at (t, t, ..., t)
        """
        contributions = []
        other_players = [j for j in range(n) if j != i]

        for _ in range(n_samples):
            # Each player j joins with probability t (independent Bernoulli)
            S = set(j for j in other_players if np.random.random() < t)
            S_with_i = S | {i}

            # Marginal contribution of player i to coalition S
            marginal = char_func(S_with_i) - char_func(S)
            contributions.append(marginal)

        return np.mean(contributions)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_neighborhood(self, node) -> List:
        """Get sorted list of nodes in the neighborhood of given node."""
        if self.depth == 0:
            return [node]

        neighborhood = {node}
        current_layer = {node}

        for _ in range(self.depth):
            next_layer = set()
            for n in current_layer:
                next_layer.update(self.graph.neighbors(n))
            neighborhood.update(next_layer)
            current_layer = next_layer - neighborhood

        return sorted(list(neighborhood))

    def get_computation_stats(self) -> Dict[str, any]:
        """
        Get statistics about the computation.

        Returns:
            Dictionary with computation statistics
        """
        if not self.shapley_values:
            return {}

        method_counts = {}
        for method in self.method_used.values():
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'n_nodes': self.n_nodes,
            'depth': self.depth,
            'n_quadrature': self.n_quadrature,
            'n_samples': self.n_samples,
            'max_exact_size': self.max_exact_size,
            'total_time': sum(self.computation_times.values()),
            'avg_time_per_node': np.mean(list(self.computation_times.values())),
            'total_cache_entries': sum(self.cache_stats.values()) if self.cache_stats else 0,
            'method_counts': method_counts,
            'methods_used': list(set(self.method_used.values()))
        }
