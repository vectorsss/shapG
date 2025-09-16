"""
Coalition Structure explainer that uses predefined coalitions.
"""

from typing import Dict, Set, Union, Optional
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder, CoalitionManager


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

        if 'coalition_structure' in kwargs:
            self.coalition_structure = kwargs['coalition_structure']

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

        nodes = list(self.coalition_structure.keys())
        iterator = tqdm(nodes, desc="Computing CS values") if self.verbose else nodes

        for node in iterator:
            coalition_set = self.coalition_structure[node]

            sampled_coalitions = manager.sample_coalitions(
                node,
                self.n_samples,
                'stratified',
                coalition_set
            )

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