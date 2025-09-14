"""
Concrete implementations of explainers for Shapley value computation.
"""

from typing import Dict, Set, List, Union, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from math import factorial
from tqdm import tqdm

from .base import GraphExplainer, Explainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree, CombinedImputation
from ..utils.graph_construction import GraphBuilder, CoalitionManager


class ExactExplainer(GraphExplainer):
    """Exact Shapley value computation using all possible coalitions."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False
    ):
        """Initialize exact explainer.

        Args:
            characteristic_function: Function to compute coalition values
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'ExactExplainer':
        """Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            # Build graph from data
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute exact Shapley values.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        nodes = list(self.graph.nodes())
        n = len(nodes)
        shapley_values = {node: 0.0 for node in nodes}

        # Precompute factorials
        fact = [factorial(i) for i in range(n + 1)]

        # Progress bar
        iterator = tqdm(nodes, desc="Computing Shapley values") if self.verbose else nodes

        for node in iterator:
            other_nodes = [n for n in nodes if n != node]

            # Consider all possible coalitions
            for size in range(len(other_nodes) + 1):
                weight = fact[size] * fact[n - size - 1] / fact[n]

                for coalition_tuple in combinations(other_nodes, size):
                    coalition = set(coalition_tuple)

                    # Marginal contribution
                    v_with = self.characteristic_function(
                        coalition | {node},
                        self.graph
                    )
                    v_without = self.characteristic_function(
                        coalition,
                        self.graph
                    )

                    shapley_values[node] += weight * (v_with - v_without)

        return shapley_values


class ShapGExplainer(GraphExplainer):
    """Fast approximate Shapley value computation using local search and sampling.

    This implementation matches the original ShapG algorithm exactly.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        depth: int = 1,
        n_samples: int = 15,
        sampling_strategy: str = 'uniform',  # Kept for API compatibility
        approximate_by_ratio: bool = True,
        scale: bool = True,
        verbose: bool = False
    ):
        """Initialize ShapG explainer.

        Args:
            characteristic_function: Function to compute coalition values
            depth: Depth for neighbor coalition
            n_samples: Number of Monte Carlo samples (m in original algorithm)
            sampling_strategy: Strategy for sampling (kept for compatibility, not used)
            approximate_by_ratio: Use ratio-based approximation
            scale: Scale final values
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)
        self.depth = depth
        self.m = n_samples  # Use 'm' internally to match original algorithm
        self.approximate_by_ratio = approximate_by_ratio
        self.scale = scale

    @property
    def n_samples(self) -> int:
        """Number of Monte Carlo samples (backward compatibility)."""
        return self.m

    @n_samples.setter
    def n_samples(self, value: int):
        """Set number of Monte Carlo samples (backward compatibility)."""
        self.m = value

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'ShapGExplainer':
        """Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute approximate Shapley values using original ShapG algorithm.

        This implements the EXACT algorithm from the original shapG function.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        import random
        import itertools
        from math import log2, ceil
        from functools import lru_cache

        G = self.graph
        shapley_values = {node: 0 for node in G.nodes()}

        # Precompute full coalition value if needed for scaling
        full_coalition_value = None
        if self.approximate_by_ratio:
            full_coalition = set(G.nodes())
            full_coalition_value = self.characteristic_function(full_coalition, G)

        # Progress tracking
        node_iterator = tqdm(G.nodes(), desc="Computing ShapG values") if self.verbose else G.nodes()

        # Cache for function evaluations
        @lru_cache(maxsize=2**15)
        def cached_f(coalition_tuple):
            return self.characteristic_function(set(coalition_tuple), G)

        # Use the standardized get_reachable_nodes_at_depth function
        from ..core.shapley import get_reachable_nodes_at_depth

        for node in node_iterator:
            # Collect all reachable nodes within specified depth
            reachable_nodes_at_depth = set()
            for d in range(1, self.depth + 1):
                reachable_nodes_at_depth.update(get_reachable_nodes_at_depth(G, node, d))

            # Determine if we need sampling or can process the full neighborhood
            if len(reachable_nodes_at_depth) < self.m:
                # Small enough neighborhood - process ALL subsets (exact computation)
                reachable_nodes_at_depth.add(node)  # Add the node itself

                coeff = 1 / 2 ** (len(reachable_nodes_at_depth) - 1)

                # Note: Original shapG does NOT apply scaling for small neighborhoods
                # Only apply scaling for sampling branch, not exact computation branch

                for S_size in range(len(reachable_nodes_at_depth)):
                    for S in itertools.combinations(reachable_nodes_at_depth - {node}, S_size):
                        S_tuple = tuple(sorted(S))
                        S_with_node_tuple = tuple(sorted(S + (node,)))

                        marginal_contribution = (
                            cached_f(S_with_node_tuple) -
                            cached_f(S_tuple)
                        )
                        shapley_values[node] += marginal_contribution

                # Apply scaling factor
                shapley_values[node] *= coeff
            else:
                # Large neighborhood - use sampling
                # Calculate number of samples using original formula
                sample_nums = ceil(len(reachable_nodes_at_depth) / self.m *
                                 (log2(len(reachable_nodes_at_depth)) + 0.5772156649))

                # Precompute coefficient
                coeff = 1 / 2 ** self.m / sample_nums
                if self.scale:
                    # Scale proportionally to ratio of full neighborhood to sample size
                    coeff *= ((len(reachable_nodes_at_depth) + 1) / (self.m + 1))

                reachable_nodes_list = list(reachable_nodes_at_depth)

                for _ in range(sample_nums):
                    # Sample a subset of reachable nodes
                    reachable_nodes_sampled = set(random.sample(
                        reachable_nodes_list,
                        min(self.m, len(reachable_nodes_list))
                    ))
                    reachable_nodes_sampled.add(node)  # Add the node itself

                    # Compute ALL coalitions of the sampled nodes
                    for S_size in range(len(reachable_nodes_sampled)):
                        for S in itertools.combinations(reachable_nodes_sampled - {node}, S_size):
                            S_tuple = tuple(sorted(S))
                            S_with_node_tuple = tuple(sorted(S + (node,)))

                            marginal_contribution = (
                                cached_f(S_with_node_tuple) -
                                cached_f(S_tuple)
                            )
                            shapley_values[node] += marginal_contribution

                # Apply scaling factors
                shapley_values[node] *= coeff

        # Optional: scale all values to match the full coalition value
        if self.approximate_by_ratio and full_coalition_value is not None:
            approximated_sum = sum(shapley_values.values())
            if approximated_sum > 0:
                scale_factor = full_coalition_value / approximated_sum
                shapley_values = {node: val * scale_factor
                                for node, val in shapley_values.items()}

        return shapley_values


class CISExplainer(GraphExplainer):
    """CIS (Combined Imputation Score) explainer that matches the original cis function.

    The CIS value for each node/feature is its individual contribution plus
    an equal share of the surplus (grand coalition value minus sum of individual values).
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False,
        n_samples: Optional[int] = None
    ):
        """Initialize CIS explainer.

        Args:
            model: Prediction model for CIS computation
            characteristic_function: Function to compute coalition values
            verbose: Whether to print progress
            n_samples: Number of samples for approximation (ignored, for API compatibility)
        """
        self.model = model
        self.n_samples = n_samples
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'CISExplainer':
        """Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            # If model is provided, use CombinedImputation characteristic function
            if self.model is not None:
                self.characteristic_function = CombinedImputation(
                    data=X if isinstance(X, np.ndarray) else X.values,
                    model=self.model
                )
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute CIS values for all nodes.

        The CIS value for each node is its individual contribution plus
        an equal share of the surplus (grand coalition value minus sum of individual values).
        This matches the original cis() function exactly.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            CIS values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)

        # Compute grand coalition value
        grand_coalition = set(nodes)
        grand_coalition_value = self.characteristic_function(grand_coalition, self.graph)

        # Compute individual values
        individual_values = {}
        for node in nodes:
            individual_values[node] = self.characteristic_function({node}, self.graph)

        total_individual_value = sum(individual_values.values())

        # Compute surplus and equal share
        surplus = grand_coalition_value - total_individual_value
        equal_share = surplus / n_nodes if n_nodes > 0 else 0

        # CIS value = individual value + equal share of surplus
        cis_values = {
            node: individual_values[node] + equal_share
            for node in nodes
        }

        return cis_values


class CSExplainer(GraphExplainer):
    """Coalition Structure explainer that uses predefined coalitions."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        coalition_structure: Optional[Dict[int, Set[int]]] = None,
        n_samples: int = 50,
        max_coalition_size: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            coalition_structure: Predefined coalition structure
            n_samples: Number of samples per coalition
            max_coalition_size: Maximum size of coalitions to consider
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)
        self.coalition_structure = coalition_structure
        self.n_samples = n_samples
        self.max_coalition_size = max_coalition_size

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'CSExplainer':
        """Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments including coalition_structure

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        # Override coalition structure if provided
        if 'coalition_structure' in kwargs:
            self.coalition_structure = kwargs['coalition_structure']

        # Build default coalition structure if not provided
        if self.coalition_structure is None:
            manager = CoalitionManager(self.graph)
            self.coalition_structure = {}
            for node in self.graph.nodes():
                self.coalition_structure[node] = manager.get_neighbors_coalition(
                    node, depth=2, include_self=False
                )

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute Shapley values using coalition structure.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        shapley_values = {}
        manager = CoalitionManager(self.graph)

        # Progress bar
        nodes = list(self.coalition_structure.keys())
        iterator = tqdm(nodes, desc="Computing CS values") if self.verbose else nodes

        for node in iterator:
            coalition_set = self.coalition_structure[node]

            # Sample from the coalition structure
            sampled_coalitions = manager.sample_coalitions(
                node,
                self.n_samples,
                'stratified',
                coalition_set
            )

            # Compute marginal contributions
            marginal_sum = 0.0

            for coalition in sampled_coalitions:
                v_with = self.characteristic_function(
                    coalition | {node},
                    self.graph
                )
                v_without = self.characteristic_function(
                    coalition,
                    self.graph
                )

                marginal_sum += (v_with - v_without)

            shapley_values[node] = marginal_sum / len(sampled_coalitions)

        return shapley_values