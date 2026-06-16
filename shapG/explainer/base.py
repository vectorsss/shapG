"""
Base classes for ShapG explainers and characteristic functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd


class CharacteristicFunction(ABC):
    """Abstract base class for characteristic/value functions.

    This defines how to compute the value of a coalition of features.
    Can be used for Shapley values, game theory, or other mathematical tasks.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the characteristic function.

        Args:
            name: Optional name for the function
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """Compute the value of a coalition.

        Args:
            coalition: Set of feature indices in the coalition
            context: Optional context (e.g., graph, data) for computation

        Returns:
            The characteristic value of the coalition
        """
        raise NotImplementedError

    def batch_compute(
        self, coalitions: List[Set[int]], context: Optional[Any] = None
    ) -> np.ndarray:
        """Compute values for multiple coalitions.

        Subclasses may override this to return either:
        - shape ``(n_coalitions,)`` — one scalar value per coalition (existing behaviour)
        - shape ``(n_coalitions, n_samples)`` — one value per (coalition, sample) pair,
          used by batched neural-network characteristic functions where all samples are
          scored in a single ``model.predict()`` call.

        ``ExactExplainer`` inspects ``ndim`` of the returned array to decide the return
        type of ``explain()``: ``Dict[int, float]`` for 1-D output and
        ``Dict[int, np.ndarray]`` for 2-D output.

        Args:
            coalitions: List of coalitions
            context: Optional context for computation

        Returns:
            Array of characteristic values, shape ``(n_coalitions,)`` or
            ``(n_coalitions, n_samples)``
        """
        return np.array([self(coalition, context) for coalition in coalitions])


class Explainer(ABC):
    """Abstract base class for all explainers."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False,
    ):
        """Initialize the explainer.

        Args:
            characteristic_function: Function to compute coalition values
            verbose: Whether to print progress information
        """
        self.characteristic_function = characteristic_function
        self.verbose = verbose
        self._fitted = False
        self._feature_names = None

    @abstractmethod
    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> "Explainer":
        """Fit the explainer to data.

        Args:
            X: Input data (array, dataframe, or graph)
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        raise NotImplementedError

    @abstractmethod
    def explain(
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
    ) -> Dict:
        """Compute feature importance scores.

        Args:
            X: Optional input data (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping feature indices to importance scores.
            Values are ``float`` when the characteristic function returns
            scalars, or ``np.ndarray`` when it returns per-sample values.
        """
        raise NotImplementedError

    def fit_explain(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> Dict:
        """Fit and explain in one step.

        Args:
            X: Input data
            **kwargs: Additional arguments

        Returns:
            Feature importance scores
        """
        return self.fit(X, **kwargs).explain()

    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names if available.

        Returns:
            List of feature names or None
        """
        return self._feature_names

    def set_feature_names(self, names: List[str]) -> "Explainer":
        """Set feature names.

        Args:
            names: List of feature names

        Returns:
            Self for method chaining
        """
        self._feature_names = names
        return self

    def _parse_input(
        self,
        X: Union[int, np.ndarray, pd.DataFrame, nx.Graph, List],
        context: Optional[Any] = None,
    ) -> Tuple[int, List, Any]:
        """Parse input to extract player count, identifiers, and context.

        Args:
            X: Input that defines players. Can be:
               - int: number of players (ids will be 0..n-1)
               - nx.Graph: n = number_of_nodes(), ids = nodes()
               - pd.DataFrame: n = number of columns, ids = column names
               - np.ndarray: n = number of columns (2D) or length (1D)
               - List: n = length, ids = elements or indices
            context: Optional context for characteristic function.
                     If None and X is Graph/DataFrame/array, X is used as context.

        Returns:
            Tuple of (n_players, player_ids, context)
        """
        if isinstance(X, int):
            n = X
            player_ids = list(range(n))
            ctx = context
        elif isinstance(X, nx.Graph):
            n = X.number_of_nodes()
            player_ids = list(X.nodes())
            ctx = context if context is not None else X
        elif isinstance(X, pd.DataFrame):
            n = X.shape[1]
            player_ids = list(X.columns)
            ctx = context if context is not None else X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                n = len(X)
            else:
                n = X.shape[1]
            player_ids = list(range(n))
            ctx = context if context is not None else X
        elif isinstance(X, list):
            n = len(X)
            try:
                if len(set(X)) == len(X):
                    player_ids = list(X)
                else:
                    player_ids = list(range(n))
            except TypeError:
                player_ids = list(range(n))
            ctx = context
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        return n, player_ids, ctx


class GraphExplainer(Explainer):
    """Base class for graph-based explainers."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False,
    ):
        """Initialize the graph explainer."""
        super().__init__(characteristic_function, verbose)
        self.graph = None
        self.coalitions = None

    def set_graph(self, G: nx.Graph) -> "GraphExplainer":
        """Set the graph for explanation.

        Args:
            G: NetworkX graph

        Returns:
            Self for method chaining
        """
        self.graph = G
        return self

    def set_coalitions(self, coalitions: Dict[int, Set[int]]) -> "GraphExplainer":
        """Set pre-computed coalitions for each node.

        Args:
            coalitions: Dictionary mapping node to its coalition set

        Returns:
            Self for method chaining
        """
        self.coalitions = coalitions
        return self
