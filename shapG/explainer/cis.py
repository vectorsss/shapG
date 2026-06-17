"""
CIS (Combined Imputation Score) explainer.
"""

from typing import Dict, Union, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import (
    CoalitionDegree,
    CenterOfImputationSet,
)
from ..utils.graph_construction import GraphBuilder


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
        n_samples: Optional[int] = None,
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

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> "CISExplainer":
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
            if self.model is not None:
                self.characteristic_function = CenterOfImputationSet(
                    data=X if isinstance(X, np.ndarray) else X.values, model=self.model
                )
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True
        return self

    def explain(
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
    ) -> Dict:
        """Compute CIS values for all nodes.

        The CIS value for each node is its individual contribution plus
        an equal share of the surplus (grand coalition value minus sum of individual values).

        When the characteristic function returns per-sample values (via
        ``batch_compute``), the result is ``Dict[int, np.ndarray]`` with
        per-sample CIS values.  Otherwise ``Dict[int, float]``.

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

        # Batch-evaluate grand coalition + all singletons in one call
        coalitions = [set(nodes)] + [{node} for node in nodes]
        values = self.characteristic_function.batch_compute(coalitions, self.graph)
        per_sample = values.ndim == 2

        grand_coalition_value = values[0]
        individual_values = {nodes[i]: values[i + 1] for i in range(n_nodes)}

        total_individual_value = sum(individual_values.values())
        surplus = grand_coalition_value - total_individual_value
        equal_share = (
            surplus / n_nodes
            if n_nodes > 0
            else (np.zeros_like(surplus) if per_sample else 0)
        )

        cis_values = {node: individual_values[node] + equal_share for node in nodes}

        # Convert back to float when scalar
        if not per_sample:
            cis_values = {node: float(v) for node, v in cis_values.items()}

        return cis_values
