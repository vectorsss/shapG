"""
Implementations of various characteristic functions for Shapley value computation.
"""

from typing import Set, Optional, Any, Callable, Union
import numpy as np
import networkx as nx
from ..explainer.base import CharacteristicFunction


class CoalitionDegree(CharacteristicFunction):
    """Characteristic function based on coalition degree in a graph."""

    def __call__(self, coalition: Set[int], context: Optional[nx.Graph] = None) -> float:
        """Compute coalition degree based on the coalition's behavior.

        Returns the sum of weighted degrees in the induced subgraph divided by 2
        (to avoid double-counting edges). Always uses weighted degrees, treating
        unweighted edges as having weight=1.

        Args:
            coalition: Set of node indices
            context: NetworkX graph

        Returns:
            Coalition degree value
        """
        if not coalition:
            return 0.0

        if context is None:
            raise AttributeError("Graph context is required for CoalitionDegree computation")

        G = context

        # Create subgraph of the coalition
        subgraph = G.subgraph(coalition)

        # Always use weighted degrees (NetworkX treats missing weights as 1)
        return float(sum(dict(subgraph.degree(weight='weight')).values()) / 2)


class NodeCount(CharacteristicFunction):
    """Simple characteristic function that counts the number of nodes."""

    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """Count the number of nodes in the coalition.

        Args:
            coalition: Set of node indices
            context: Unused

        Returns:
            Number of nodes in coalition
        """
        return float(len(coalition))


class WeightedSum(CharacteristicFunction):
    """Characteristic function based on weighted sum of features."""

    def __init__(self, weights: Optional[Union[dict, np.ndarray]] = None, name: Optional[str] = None):
        """Initialize with optional weights.

        Args:
            weights: Feature weights as a dict {index: weight} or numpy array
            name: Optional function name
        """
        super().__init__(name)
        if weights is None:
            self.weights = {}
        elif isinstance(weights, np.ndarray):
            # Convert numpy array to dict {index: weight}
            self.weights = {i: float(w) for i, w in enumerate(weights)}
        else:
            self.weights = weights

    def __call__(self, coalition: Set[int], context: Optional[np.ndarray] = None) -> float:
        """Compute weighted sum of features in coalition.

        Args:
            coalition: Set of feature indices
            context: Feature values array

        Returns:
            Weighted sum of coalition features
        """
        if not coalition:
            return 0.0

        if context is None:
            # If no context, just sum the weights from dict
            if self.weights:  # Non-empty dict
                return sum(self.weights.get(i, 0.0) for i in coalition)
            return 0.0  # Empty weights dict should return 0

        # Sum feature values, optionally weighted
        coalition_indices = list(coalition)
        values = context[coalition_indices] if context.ndim == 1 else context[:, coalition_indices].sum(axis=0)

        if self.weights:  # Non-empty dict
            weights = np.array([self.weights.get(i, 0.0) for i in coalition_indices])
            return float(np.dot(values, weights))

        return float(np.sum(values))


class CustomFunction(CharacteristicFunction):
    """Wrapper for custom characteristic functions."""

    def __init__(self, func: Callable, name: Optional[str] = None):
        """Initialize with a custom function.

        Args:
            func: Custom function (coalition, context) -> float
            name: Optional function name
        """
        super().__init__(name or "CustomFunction")
        self.func = func

    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """Apply the custom function.

        Args:
            coalition: Set of feature indices
            context: Context for computation

        Returns:
            Custom function value
        """
        return self.func(coalition, context)


class CombinedImputation(CharacteristicFunction):
    """Characteristic function for Combined Imputation Score (CIS)."""

    def __init__(self, data: np.ndarray, model: Any, baseline: Optional[np.ndarray] = None):
        """Initialize CIS characteristic function.

        Args:
            data: Input data
            model: Prediction model
            baseline: Baseline values for imputation
        """
        super().__init__("CombinedImputation")
        self.data = data
        self.model = model
        self.baseline = baseline if baseline is not None else np.zeros(data.shape[1])

    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """Compute model prediction with features in coalition.

        Args:
            coalition: Set of feature indices to keep
            context: Optional sample index

        Returns:
            Model prediction score
        """
        if context is None or not isinstance(context, int):
            # Use mean over all samples (ignore non-integer context like graphs)
            imputed_data = self.data.copy()
            mask = np.ones(self.data.shape[1], dtype=bool)
            # Filter coalition to only valid indices
            valid_coalition = [i for i in coalition if 0 <= i < self.data.shape[1]]
            if valid_coalition:
                mask[valid_coalition] = False
            if np.any(mask):  # Only impute if there are features to impute
                imputed_data[:, mask] = self.baseline[mask]
            return float(np.mean(self.model.predict(imputed_data)))

        # Single sample
        sample = self.data[context].copy()
        mask = np.ones(len(sample), dtype=bool)
        # Filter coalition to only valid indices
        valid_coalition = [i for i in coalition if 0 <= i < len(sample)]
        if valid_coalition:
            mask[valid_coalition] = False
        if np.any(mask):  # Only impute if there are features to impute
            sample[mask] = self.baseline[mask]

        return float(self.model.predict(sample.reshape(1, -1))[0])