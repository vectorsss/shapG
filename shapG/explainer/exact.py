"""
Exact Shapley value computation using all possible coalitions.
"""

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from math import factorial
from tqdm import tqdm

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


# Memory warning threshold
LARGE_GRAPH_WARNING_THRESHOLD = 20  # Warn if n > 20 nodes (2^20 = 1M coalitions)


class ExactExplainer(GraphExplainer):
    """Exact Shapley value computation using all possible coalitions."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False,
    ):
        """Initialize exact explainer.

        Args:
            characteristic_function: Function to compute coalition values
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> "ExactExplainer":
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
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
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

        # Warn about memory requirements for large graphs
        if n > LARGE_GRAPH_WARNING_THRESHOLD:
            import warnings

            warnings.warn(
                f"Computing exact Shapley values for {n} nodes requires evaluating "
                f"2^{n} = {2**n:,} coalitions. This may consume significant memory and time. "
                f"Consider using an approximate method (ShapGExplainer, QRCSExplainer) for n > {LARGE_GRAPH_WARNING_THRESHOLD}.",
                ResourceWarning,
            )

        fact = [factorial(i) for i in range(n + 1)]

        iterator = (
            tqdm(nodes, desc="Computing Shapley values") if self.verbose else nodes
        )

        for node in iterator:
            other_nodes = [n for n in nodes if n != node]

            for size in range(len(other_nodes) + 1):
                weight = fact[size] * fact[n - size - 1] / fact[n]

                for coalition_tuple in combinations(other_nodes, size):
                    coalition = set(coalition_tuple)

                    v_with = self.characteristic_function(
                        coalition | {node}, self.graph
                    )
                    v_without = self.characteristic_function(coalition, self.graph)

                    shapley_values[node] += weight * (v_with - v_without)

        return shapley_values
