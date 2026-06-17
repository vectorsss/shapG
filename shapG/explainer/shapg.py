"""
Fast approximate Shapley value computation using local search and sampling.
"""

from typing import Dict, Union, Optional, Set, List, Tuple, Callable
import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools
from math import log2, ceil
from functools import lru_cache
from tqdm import tqdm

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder
from ..utils.graph_helpers import get_reachable_nodes_at_depth

# Named constant for Euler-Mascheroni constant
EULER_MASCHERONI_CONSTANT = 0.5772156649


class ShapGExplainer(GraphExplainer):
    """Fast approximate Shapley value computation using local search and sampling.

    This implementation matches the original ShapG algorithm exactly.

    When the characteristic function returns per-sample values (via
    ``batch_compute``), Shapley values are computed per-sample and the
    result is ``Dict[int, np.ndarray]``.  Otherwise ``Dict[int, float]``.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        depth: int = 1,
        n_samples: int = 15,
        sampling_strategy: str = "uniform",
        approximate_by_ratio: bool = True,
        scale: bool = True,
        verbose: bool = False,
        cache_size: int = 2**15,
    ):
        """Initialize ShapG explainer.

        Args:
            characteristic_function: Function to compute coalition values
            depth: Depth for neighbor coalition (must be positive)
            n_samples: Number of Monte Carlo samples (m in original algorithm, must be positive)
            sampling_strategy: Strategy for sampling (kept for compatibility, not used)
            approximate_by_ratio: Use ratio-based approximation
            scale: Scale final values
            verbose: Whether to print progress
            cache_size: Size of LRU cache for coalition values
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

        # Validate inputs
        if depth <= 0:
            raise ValueError(f"Depth must be positive, got {depth}")
        if n_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {n_samples}")
        if cache_size <= 0:
            raise ValueError(f"Cache size must be positive, got {cache_size}")

        self.depth = depth
        self.m = n_samples
        self.approximate_by_ratio = approximate_by_ratio
        self.scale = scale
        self.cache_size = cache_size

    @property
    def n_samples(self) -> int:
        """Number of Monte Carlo samples (backward compatibility)."""
        return self.m

    @n_samples.setter
    def n_samples(self, value: int):
        """Set number of Monte Carlo samples (backward compatibility)."""
        if value <= 0:
            raise ValueError(f"Number of samples must be positive, got {value}")
        self.m = value

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> "ShapGExplainer":
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

    def _create_cached_characteristic_function(
        self,
    ) -> Callable[[Tuple[int, ...]], float]:
        """Create a cached version of the characteristic function.

        Returns:
            Cached characteristic function
        """

        @lru_cache(maxsize=self.cache_size)
        def cached_f(coalition_tuple):
            return self.characteristic_function(set(coalition_tuple), self.graph)

        return cached_f

    def _get_reachable_nodes(self, node: int) -> Set[int]:
        """Get all reachable nodes from a given node up to specified depth.

        Args:
            node: Starting node

        Returns:
            Set of reachable nodes
        """
        reachable_nodes = set()
        for d in range(1, self.depth + 1):
            reachable_nodes.update(get_reachable_nodes_at_depth(self.graph, node, d))
        return reachable_nodes

    def _compute_exact_shapley_for_small_coalition(
        self,
        node: int,
        reachable_nodes: Set[int],
        cached_f: Callable[[Tuple[int, ...]], float],
    ) -> float:
        """Compute exact Shapley value for small coalitions.

        Args:
            node: Node to compute Shapley value for
            reachable_nodes: Set of reachable nodes including the node itself
            cached_f: Cached characteristic function

        Returns:
            Shapley value for the node
        """
        shapley_value = 0.0
        coeff = 1 / 2 ** (len(reachable_nodes) - 1)

        for S_size in range(len(reachable_nodes)):
            for S in itertools.combinations(reachable_nodes - {node}, S_size):
                S_tuple = tuple(sorted(S))
                S_with_node_tuple = tuple(sorted(S + (node,)))

                marginal_contribution = cached_f(S_with_node_tuple) - cached_f(S_tuple)
                shapley_value += marginal_contribution

        return shapley_value * coeff

    def _calculate_sample_nums(self, n_reachable: int) -> int:
        """Calculate the number of samples needed for approximation.

        Args:
            n_reachable: Number of reachable nodes

        Returns:
            Number of samples to take
        """
        return ceil(
            n_reachable / self.m * (log2(n_reachable) + EULER_MASCHERONI_CONSTANT)
        )

    def _calculate_sampling_coefficient(
        self, n_reachable: int, sample_nums: int
    ) -> float:
        """Calculate the coefficient for sampling-based approximation.

        Args:
            n_reachable: Number of reachable nodes
            sample_nums: Number of samples

        Returns:
            Coefficient for scaling
        """
        coeff = 1 / 2**self.m / sample_nums
        if self.scale:
            coeff *= (n_reachable + 1) / (self.m + 1)
        return coeff

    def _compute_sampled_marginal_contributions(
        self,
        node: int,
        reachable_nodes_list: List[int],
        sample_nums: int,
        cached_f: Callable[[Tuple[int, ...]], float],
    ) -> float:
        """Compute marginal contributions using sampling.

        Args:
            node: Node to compute contributions for
            reachable_nodes_list: List of reachable nodes
            sample_nums: Number of samples to take
            cached_f: Cached characteristic function

        Returns:
            Sum of marginal contributions
        """
        total_contribution = 0.0

        for _ in range(sample_nums):
            reachable_nodes_sampled = set(
                random.sample(
                    reachable_nodes_list, min(self.m, len(reachable_nodes_list))
                )
            )
            reachable_nodes_sampled.add(node)

            for S_size in range(len(reachable_nodes_sampled)):
                for S in itertools.combinations(
                    reachable_nodes_sampled - {node}, S_size
                ):
                    S_tuple = tuple(sorted(S))
                    S_with_node_tuple = tuple(sorted(S + (node,)))

                    marginal_contribution = cached_f(S_with_node_tuple) - cached_f(
                        S_tuple
                    )
                    total_contribution += marginal_contribution

        return total_contribution

    def _compute_approximate_shapley_for_large_coalition(
        self,
        node: int,
        reachable_nodes: Set[int],
        cached_f: Callable[[Tuple[int, ...]], float],
    ) -> float:
        """Compute approximate Shapley value for large coalitions using sampling.

        Args:
            node: Node to compute Shapley value for
            reachable_nodes: Set of reachable nodes
            cached_f: Cached characteristic function

        Returns:
            Shapley value for the node
        """
        reachable_nodes_list = sorted(reachable_nodes)
        sample_nums = self._calculate_sample_nums(len(reachable_nodes))
        coeff = self._calculate_sampling_coefficient(len(reachable_nodes), sample_nums)

        total_contribution = self._compute_sampled_marginal_contributions(
            node, reachable_nodes_list, sample_nums, cached_f
        )

        return total_contribution * coeff

    def _compute_node_shapley_value(
        self, node: int, cached_f: Callable[[Tuple[int, ...]], float]
    ) -> float:
        """Compute Shapley value for a single node.

        Args:
            node: Node to compute Shapley value for
            cached_f: Cached characteristic function

        Returns:
            Shapley value for the node
        """
        reachable_nodes = self._get_reachable_nodes(node)

        if len(reachable_nodes) < self.m:
            # Small coalition: compute exactly
            reachable_nodes.add(node)
            return self._compute_exact_shapley_for_small_coalition(
                node, reachable_nodes, cached_f
            )
        else:
            # Large coalition: use sampling
            return self._compute_approximate_shapley_for_large_coalition(
                node, reachable_nodes, cached_f
            )

    def _apply_ratio_approximation(
        self, shapley_values: Dict, full_coalition_value
    ) -> Dict:
        """Apply ratio-based approximation to scale Shapley values.

        Supports both scalar and per-sample (array) values.

        Args:
            shapley_values: Computed Shapley values
            full_coalition_value: Value of the full coalition

        Returns:
            Scaled Shapley values
        """
        approximated_sum = sum(shapley_values.values())
        if np.isscalar(approximated_sum):
            if approximated_sum > 0:
                scale_factor = full_coalition_value / approximated_sum
                return {
                    node: val * scale_factor for node, val in shapley_values.items()
                }
        else:
            # Per-sample: element-wise ratio, avoiding division by zero
            nonzero = approximated_sum != 0
            safe_sum = np.where(nonzero, approximated_sum, 1.0)
            scale_factor = np.where(nonzero, full_coalition_value / safe_sum, 0.0)
            return {node: val * scale_factor for node, val in shapley_values.items()}
        return shapley_values

    # ------------------------------------------------------------------
    # Per-sample path: collect all coalitions, batch-evaluate, accumulate
    # ------------------------------------------------------------------

    def _collect_node_coalitions(self, node):
        """Collect all (S, S∪{node}, coeff) entries for one node.

        Returns:
            list of (S_tuple, S_with_node_tuple, coeff) triples
        """
        reachable_nodes = self._get_reachable_nodes(node)
        entries = []

        if len(reachable_nodes) < self.m:
            # Small coalition: exact enumeration
            reachable_nodes.add(node)
            coeff = 1 / 2 ** (len(reachable_nodes) - 1)
            for S_size in range(len(reachable_nodes)):
                for S in itertools.combinations(reachable_nodes - {node}, S_size):
                    S_tuple = tuple(sorted(S))
                    S_with = tuple(sorted(S + (node,)))
                    entries.append((S_tuple, S_with, coeff))
        else:
            # Large coalition: sampling
            reachable_nodes_list = sorted(reachable_nodes)
            sample_nums = self._calculate_sample_nums(len(reachable_nodes))
            coeff = self._calculate_sampling_coefficient(
                len(reachable_nodes), sample_nums
            )
            for _ in range(sample_nums):
                sampled = set(
                    random.sample(
                        reachable_nodes_list,
                        min(self.m, len(reachable_nodes_list)),
                    )
                )
                sampled.add(node)
                for S_size in range(len(sampled)):
                    for S in itertools.combinations(sampled - {node}, S_size):
                        S_tuple = tuple(sorted(S))
                        S_with = tuple(sorted(S + (node,)))
                        entries.append((S_tuple, S_with, coeff))

        return entries

    def _explain_per_sample(self) -> Dict[int, np.ndarray]:
        """Compute per-sample Shapley values using batch_compute.

        Collects all unique coalitions across all nodes, evaluates them
        in a single ``batch_compute()`` call, then accumulates weighted
        marginal contributions per node.

        Returns:
            Dict mapping node id to per-sample Shapley value array.
        """
        nodes = list(self.graph.nodes())

        # Phase 1: collect all (S, S∪{i}, coeff) triples per node
        all_coalitions_set: Set[Tuple[int, ...]] = set()
        node_entries: Dict[int, List] = {}

        for node in nodes:
            entries = self._collect_node_coalitions(node)
            node_entries[node] = entries
            for s_off, s_on, _ in entries:
                all_coalitions_set.add(s_off)
                all_coalitions_set.add(s_on)

        # Ensure full coalition is included (needed for ratio approximation)
        full_coalition_tuple = tuple(sorted(nodes))
        all_coalitions_set.add(full_coalition_tuple)

        # Phase 2: batch-evaluate all unique coalitions
        unique_coalitions = list(all_coalitions_set)
        coalition_index = {c: i for i, c in enumerate(unique_coalitions)}
        coalition_sets = [set(c) for c in unique_coalitions]

        all_values = self.characteristic_function.batch_compute(
            coalition_sets, self.graph
        )
        # shape: (n_coalitions, n_data_samples)

        n_data_samples = all_values.shape[1]

        # Phase 3: accumulate per-node Shapley values
        shapley_values: Dict[int, np.ndarray] = {
            node: np.zeros(n_data_samples) for node in nodes
        }

        node_iterator = (
            tqdm(nodes, desc="Computing ShapG values") if self.verbose else nodes
        )
        for node in node_iterator:
            for s_off, s_on, coeff in node_entries[node]:
                v_off = all_values[coalition_index[s_off]]
                v_on = all_values[coalition_index[s_on]]
                shapley_values[node] += coeff * (v_on - v_off)

        # Phase 4: ratio approximation
        if self.approximate_by_ratio:
            full_coalition = tuple(sorted(nodes))
            full_value = all_values[coalition_index[full_coalition]]
            shapley_values = self._apply_ratio_approximation(shapley_values, full_value)

        return shapley_values

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def explain(
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
    ) -> Dict:
        """Compute approximate Shapley values using original ShapG algorithm.

        When the characteristic function's ``batch_compute`` returns 2-D
        output (per-sample), returns ``Dict[int, np.ndarray]``.  Otherwise
        returns ``Dict[int, float]``.

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

        # Detect per-sample mode by probing batch_compute
        per_sample = (
            hasattr(self.characteristic_function, "batch_compute")
            and self.characteristic_function.batch_compute([set()], self.graph).ndim
            == 2
        )

        if per_sample:
            return self._explain_per_sample()

        # --- Scalar path (original algorithm, unchanged) ---

        shapley_values = {node: 0.0 for node in self.graph.nodes()}

        # Compute full coalition value if using ratio approximation
        full_coalition_value = None
        if self.approximate_by_ratio:
            full_coalition = set(self.graph.nodes())
            full_coalition_value = self.characteristic_function(
                full_coalition, self.graph
            )

        # Create cached characteristic function
        cached_f = self._create_cached_characteristic_function()

        # Setup progress iterator
        node_iterator = (
            tqdm(self.graph.nodes(), desc="Computing ShapG values")
            if self.verbose
            else self.graph.nodes()
        )

        # Compute Shapley values for each node
        for node in node_iterator:
            shapley_values[node] = self._compute_node_shapley_value(node, cached_f)

        # Apply ratio approximation if requested
        if self.approximate_by_ratio and full_coalition_value is not None:
            shapley_values = self._apply_ratio_approximation(
                shapley_values, full_coalition_value
            )

        return shapley_values
