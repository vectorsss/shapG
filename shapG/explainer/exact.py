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
    ) -> Dict:
        """Compute exact Shapley values using a streaming accumulation.

        Processes coalitions in chunks.  For each coalition C evaluated, its
        value v(C) is immediately accumulated into the Shapley totals using
        the streaming identity:

            C acts as "S"      → subtract  w(|C|)   * v(C)  from every i ∉ C
            C acts as "S∪{j}"  → add       w(|C|-1) * v(C)  to   every j ∈ C

        where w(s) = s! (n-s-1)! / n!.  No ``all_values`` matrix is stored;
        peak memory is proportional to one chunk of coalition values.

        Supports any metric shape: scalar ``Dict[int, float]``, per-sample
        ``Dict[int, np.ndarray]``, or higher-dimensional arrays — the
        accumulators are promoted automatically from 0.0 on first addition.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Shapley values for each node.
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        nodes = list(self.graph.nodes())
        n = len(nodes)

        if n > LARGE_GRAPH_WARNING_THRESHOLD:
            import warnings

            warnings.warn(
                f"Computing exact Shapley values for {n} nodes requires evaluating "
                f"2^{n} = {2**n:,} coalitions. This may consume significant memory and time. "
                f"Consider using an approximate method (ShapGExplainer, QRCSExplainer) for n > {LARGE_GRAPH_WARNING_THRESHOLD}.",
                ResourceWarning,
            )

        all_coalitions = [
            set(combo) for r in range(n + 1) for combo in combinations(nodes, r)
        ]

        fact = [factorial(i) for i in range(n + 1)]

        # Accumulators start as 0.0; numpy promotes to array automatically when
        # v_C is an array (e.g. per-sample or per-element metric output).
        shapley_values: Dict = {node: 0.0 for node in nodes}

        # Honour chunk_size from BatchedModelCharacteristic (or similar) so that
        # each model.predict call stays within the same memory budget as before.
        # Fall back to processing all coalitions at once only for characteristic
        # functions that don't do batched model inference (e.g. CoalitionDegree).
        # For BatchedModelCharacteristic without chunk_size set, default to 32 to
        # avoid building (n_coalitions * n_samples, ...) tensors in one shot.
        chunk_size = getattr(self.characteristic_function, "chunk_size", None)
        if not chunk_size:
            has_model = hasattr(self.characteristic_function, "model")
            chunk_size = 32 if has_model else len(all_coalitions)

        iterator = (
            tqdm(
                range(0, len(all_coalitions), chunk_size),
                desc="Computing Shapley values",
                total=(len(all_coalitions) + chunk_size - 1) // chunk_size,
            )
            if self.verbose
            else range(0, len(all_coalitions), chunk_size)
        )

        for start in iterator:
            chunk = all_coalitions[start : start + chunk_size]
            chunk_values = self.characteristic_function.batch_compute(chunk, self.graph)
            # chunk_values[j] is the value for chunk[j]: scalar or ndarray

            for C, v_C in zip(chunk, chunk_values):
                s = len(C)

                # C acts as "S" (feature i not in C): subtract contribution
                if s < n:
                    w_without = fact[s] * fact[n - s - 1] / fact[n]
                    for node in nodes:
                        if node not in C:
                            shapley_values[node] = (
                                shapley_values[node] - w_without * v_C
                            )

                # C acts as "S ∪ {j}" (feature j in C): add contribution
                if s > 0:
                    w_with = fact[s - 1] * fact[n - s] / fact[n]
                    for node in C:
                        shapley_values[node] = shapley_values[node] + w_with * v_C

        return shapley_values
