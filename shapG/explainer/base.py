"""
Base classes for ShapG explainers and characteristic functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Union, Optional, Any, Callable
import numpy as np
import pandas as pd
import networkx as nx


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
        pass

    def batch_compute(self, coalitions: List[Set[int]], context: Optional[Any] = None) -> np.ndarray:
        """Compute values for multiple coalitions.

        Args:
            coalitions: List of coalitions
            context: Optional context for computation

        Returns:
            Array of characteristic values
        """
        return np.array([self(coalition, context) for coalition in coalitions])


class Explainer(ABC):
    """Abstract base class for all explainers."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False
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
    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'Explainer':
        """Fit the explainer to data.

        Args:
            X: Input data (array, dataframe, or graph)
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute feature importance scores.

        Args:
            X: Optional input data (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        pass

    def fit_explain(
        self,
        X: Union[np.ndarray, pd.DataFrame, nx.Graph],
        **kwargs
    ) -> Dict[int, float]:
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

    def set_feature_names(self, names: List[str]) -> 'Explainer':
        """Set feature names.

        Args:
            names: List of feature names

        Returns:
            Self for method chaining
        """
        self._feature_names = names
        return self


class GraphExplainer(Explainer):
    """Base class for graph-based explainers."""

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        verbose: bool = False
    ):
        """Initialize the graph explainer."""
        super().__init__(characteristic_function, verbose)
        self.graph = None
        self.coalitions = None

    def set_graph(self, G: nx.Graph) -> 'GraphExplainer':
        """Set the graph for explanation.

        Args:
            G: NetworkX graph

        Returns:
            Self for method chaining
        """
        self.graph = G
        return self

    def set_coalitions(self, coalitions: Dict[int, Set[int]]) -> 'GraphExplainer':
        """Set pre-computed coalitions for each node.

        Args:
            coalitions: Dictionary mapping node to its coalition set

        Returns:
            Self for method chaining
        """
        self.coalitions = coalitions
        return self