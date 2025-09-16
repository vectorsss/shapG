"""
Fast approximate Shapley value computation using local search and sampling.
"""

from typing import Dict, Union, Optional
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


class ShapGExplainer(GraphExplainer):
    """Fast approximate Shapley value computation using local search and sampling.

    This implementation matches the original ShapG algorithm exactly.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        depth: int = 1,
        n_samples: int = 15,
        sampling_strategy: str = 'uniform',
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
        self.m = n_samples
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

        G = self.graph
        shapley_values = {node: 0 for node in G.nodes()}

        full_coalition_value = None
        if self.approximate_by_ratio:
            full_coalition = set(G.nodes())
            full_coalition_value = self.characteristic_function(full_coalition, G)

        node_iterator = tqdm(G.nodes(), desc="Computing ShapG values") if self.verbose else G.nodes()

        @lru_cache(maxsize=2**15)
        def cached_f(coalition_tuple):
            return self.characteristic_function(set(coalition_tuple), G)

        for node in node_iterator:
            reachable_nodes_at_depth = set()
            for d in range(1, self.depth + 1):
                reachable_nodes_at_depth.update(get_reachable_nodes_at_depth(G, node, d))

            if len(reachable_nodes_at_depth) < self.m:
                reachable_nodes_at_depth.add(node)

                coeff = 1 / 2 ** (len(reachable_nodes_at_depth) - 1)

                for S_size in range(len(reachable_nodes_at_depth)):
                    for S in itertools.combinations(reachable_nodes_at_depth - {node}, S_size):
                        S_tuple = tuple(sorted(S))
                        S_with_node_tuple = tuple(sorted(S + (node,)))

                        marginal_contribution = (
                            cached_f(S_with_node_tuple) -
                            cached_f(S_tuple)
                        )
                        shapley_values[node] += marginal_contribution

                shapley_values[node] *= coeff
            else:
                sample_nums = ceil(len(reachable_nodes_at_depth) / self.m *
                                 (log2(len(reachable_nodes_at_depth)) + 0.5772156649))

                coeff = 1 / 2 ** self.m / sample_nums
                if self.scale:
                    coeff *= ((len(reachable_nodes_at_depth) + 1) / (self.m + 1))

                reachable_nodes_list = list(reachable_nodes_at_depth)

                for _ in range(sample_nums):
                    reachable_nodes_sampled = set(random.sample(
                        reachable_nodes_list,
                        min(self.m, len(reachable_nodes_list))
                    ))
                    reachable_nodes_sampled.add(node)
                    for S_size in range(len(reachable_nodes_sampled)):
                        for S in itertools.combinations(reachable_nodes_sampled - {node}, S_size):
                            S_tuple = tuple(sorted(S))
                            S_with_node_tuple = tuple(sorted(S + (node,)))

                            marginal_contribution = (
                                cached_f(S_with_node_tuple) -
                                cached_f(S_tuple)
                            )
                            shapley_values[node] += marginal_contribution

                shapley_values[node] *= coeff

        if self.approximate_by_ratio and full_coalition_value is not None:
            approximated_sum = sum(shapley_values.values())
            if approximated_sum > 0:
                scale_factor = full_coalition_value / approximated_sum
                shapley_values = {node: val * scale_factor
                                for node, val in shapley_values.items()}

        return shapley_values