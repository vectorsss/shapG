"""
Model-based characteristic functions with feature masking (no retraining).

These classes provide efficient characteristic functions for pre-trained models,
avoiding the computational cost of retraining for each coalition evaluation.
"""

from typing import Set, List, Optional, Any, Callable, Union
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import r2_score, accuracy_score
from .characteristic_functions import CharacteristicFunction


class ModelBasedCharacteristic(CharacteristicFunction):
    """
    Characteristic function using a pre-trained model with feature masking.

    This class evaluates model performance on coalitions by masking (imputing)
    features NOT in the coalition, without retraining the model.

    Performance benefits:
    - ~100x faster than retraining for each coalition
    - Deterministic results (no random train/test splits)
    - Memory efficient (single model instance)

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.model_selection import train_test_split
        >>>
        >>> # Train model once
        >>> X, y = load_data()
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> model = RandomForestRegressor()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Create efficient characteristic function
        >>> char_func = ModelBasedCharacteristic(
        ...     model=model,
        ...     X=X_test,
        ...     y=y_test,
        ...     masking_strategy='mean'
        ... )
        >>>
        >>> # Evaluate coalitions (no retraining!)
        >>> score = char_func({0, 2, 5})  # Fast!
    """

    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        masking_strategy: str = "mean",
        metric_fn: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize model-based characteristic function.

        Args:
            model: Pre-trained model with .predict() method
            X: Evaluation data (typically test set)
            y: True labels for X
            masking_strategy: How to impute masked features
                - 'zero': Replace with zeros
                - 'mean': Replace with column means (default)
                - 'median': Replace with column medians
                - 'noise': Replace with random noise (std-scaled)
            metric_fn: Custom metric function(y_true, y_pred) -> score
                       If None, auto-detects (R² for regression, accuracy for classification)
            name: Optional name for the function
        """
        super().__init__(name or "ModelBased")
        self.model = model

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.X = X.values
        else:
            self.feature_names = None
            self.X = X

        self.y = y
        self.metric_fn = metric_fn or self._auto_detect_metric()
        self.masking_strategy = masking_strategy

        # Pre-compute baseline values for masking
        self.baseline = self._compute_baseline(masking_strategy)

    def _auto_detect_metric(self) -> Callable:
        """Auto-detect appropriate metric based on target distribution."""
        n_unique = len(np.unique(self.y))

        if n_unique <= 10:  # Classification
            return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        else:  # Regression
            return lambda y_true, y_pred: r2_score(y_true, y_pred)

    def _compute_baseline(self, strategy: str) -> np.ndarray:
        """Compute baseline values for feature masking."""
        if strategy == "zero":
            return np.zeros(self.X.shape[1])
        elif strategy == "mean":
            return np.mean(self.X, axis=0)
        elif strategy == "median":
            return np.median(self.X, axis=0)
        elif strategy == "noise":
            # Random noise scaled by feature std
            return np.random.randn(self.X.shape[1]) * np.std(self.X, axis=0)
        else:
            raise ValueError(
                f"Unknown masking strategy: {strategy}. "
                f"Choose from: 'zero', 'mean', 'median', 'noise'"
            )

    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """
        Evaluate model performance with only coalition features.

        Args:
            coalition: Set of feature indices to KEEP (include)
            context: Unused (for API compatibility with graph-based methods)

        Returns:
            Model performance score (higher = better)
        """
        if len(coalition) == 0:
            return 0.0

        # Validate coalition indices
        valid_coalition = [i for i in coalition if 0 <= i < self.X.shape[1]]
        if not valid_coalition:
            return 0.0

        # Create masked data: keep coalition features, impute others
        X_masked = self.X.copy()
        mask = np.ones(self.X.shape[1], dtype=bool)
        mask[valid_coalition] = False  # False = keep, True = mask

        if np.any(mask):
            X_masked[:, mask] = self.baseline[mask]

        # Predict with masked data (NO RETRAINING!)
        y_pred = self.model.predict(X_masked)

        # Compute metric
        try:
            score = self.metric_fn(self.y, y_pred)
            return float(score)
        except Exception as e:
            # Handle edge cases (e.g., constant predictions)
            print(f"Warning: Metric computation failed for coalition {coalition}: {e}")
            return 0.0


class GraphModelCharacteristic(CharacteristicFunction):
    """
    Model-based characteristic function for graph node coalitions.

    This class is designed for graph-based Shapley value computation where:
    - Graph nodes represent features (columns)
    - Coalitions are sets of feature names (strings) or indices (ints)
    - Model is pre-trained and not retrained

    Example:
        >>> import pandas as pd
        >>> from shapG import GraphBuilder, ShapGExplainer
        >>>
        >>> # Train model once
        >>> X = pd.DataFrame(...)  # Column names = feature names
        >>> y = ...
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> model = lgb.LGBMRegressor()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Create graph and characteristic function
        >>> builder = GraphBuilder()
        >>> G = builder.from_correlation(X)
        >>>
        >>> char_func = GraphModelCharacteristic(
        ...     model=model,
        ...     X=X_test,
        ...     y=y_test,
        ...     masking_strategy='mean'
        ... )
        >>>
        >>> # Compute Shapley values (fast!)
        >>> explainer = ShapGExplainer(characteristic_function=char_func)
        >>> shapley_values = explainer.fit_explain(G)
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        masking_strategy: str = "mean",
        metric_fn: Optional[Callable] = None,
        name: Optional[str] = None,
        baseline: Optional[np.ndarray] = None,
    ):
        """
        Initialize graph-aware model characteristic function.

        Args:
            model: Pre-trained model with .predict() method
            X: DataFrame where columns are features/graph nodes (typically test set)
            y: True labels for X
            masking_strategy: Feature imputation strategy ('mean', 'median', 'zero', 'noise')
            metric_fn: Custom metric function (defaults to R² for regression)
            name: Optional name
            baseline: Optional pre-computed baseline values (e.g., from training set).
                     If None, will compute from X (not recommended if X is test set).
        """
        super().__init__(name or "GraphModel")
        self.model = model
        self.X = X
        self.y = y
        self.metric_fn = metric_fn or (lambda yt, yp: r2_score(yt, yp))
        self.feature_names = X.columns.tolist()
        self.masking_strategy = masking_strategy

        # Use provided baseline or compute from X
        if baseline is not None:
            self.baseline = baseline
        else:
            # Compute baseline values from X (WARNING: if X is test set, this may not be ideal)
            if masking_strategy == "mean":
                self.baseline = X.mean().values
            elif masking_strategy == "median":
                self.baseline = X.median().values
            elif masking_strategy == "zero":
                self.baseline = np.zeros(len(self.feature_names))
            elif masking_strategy == "permutation":
                # For permutation, we'll shuffle on-the-fly in __call__
                self.baseline = None
            elif masking_strategy == "noise":
                self.baseline = (
                    np.random.randn(len(self.feature_names)) * X.std().values
                )
            else:
                raise ValueError(f"Unknown masking strategy: {masking_strategy}")

        # For permutation, store random seed for reproducibility
        self.rng = (
            np.random.default_rng(42) if masking_strategy == "permutation" else None
        )

    def __call__(
        self, coalition: Set[Union[int, str]], context: Optional[nx.Graph] = None
    ) -> float:
        """
        Evaluate model with coalition of features/nodes.

        Args:
            coalition: Set of feature names (strings) or indices (ints)
            context: Optional graph (unused, for API compatibility)

        Returns:
            Model performance score
        """
        if len(coalition) == 0:
            return 0.0

        # Convert coalition to column names
        coalition_cols = []
        for item in coalition:
            if isinstance(item, int):
                if 0 <= item < len(self.feature_names):
                    coalition_cols.append(self.feature_names[item])
            elif isinstance(item, str):
                if item in self.feature_names:
                    coalition_cols.append(item)

        if not coalition_cols:
            return 0.0

        # Create masked dataframe
        X_masked = self.X.copy()
        masked_cols = [col for col in self.feature_names if col not in coalition_cols]

        # Impute masked features
        if self.masking_strategy == "permutation":
            # Randomly permute values for masked columns
            for col in masked_cols:
                X_masked[col] = self.rng.permutation(X_masked[col].values)
        else:
            # Use fixed baseline values
            for col in masked_cols:
                col_idx = self.feature_names.index(col)
                X_masked[col] = self.baseline[col_idx]

        # Predict (no retraining!)
        y_pred = self.model.predict(X_masked)

        try:
            score = self.metric_fn(self.y, y_pred)
            return float(score)
        except Exception as e:
            print(f"Warning: Metric computation failed: {e}")
            return 0.0


class EnsembleMaskingCharacteristic(CharacteristicFunction):
    """
    Characteristic function that ensembles multiple masking strategies.

    This reduces variance by averaging scores across different imputation methods.

    Example:
        >>> char_func = EnsembleMaskingCharacteristic(
        ...     model=model,
        ...     X=X_test,
        ...     y=y_test,
        ...     strategies=['zero', 'mean', 'median']
        ... )
        >>> score = char_func({0, 1, 2})  # Average over 3 strategies
    """

    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        strategies: list = None,
        metric_fn: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize ensemble masking characteristic function.

        Args:
            model: Pre-trained model
            X: Evaluation data
            y: True labels
            strategies: List of masking strategies to ensemble (default: ['zero', 'mean', 'median'])
            metric_fn: Metric function
            name: Optional name
        """
        super().__init__(name or "EnsembleMasking")
        self.strategies = strategies or ["zero", "mean", "median"]

        # Create a characteristic function for each strategy
        self.char_funcs = [
            ModelBasedCharacteristic(model, X, y, strategy, metric_fn)
            for strategy in self.strategies
        ]

    def __call__(self, coalition: Set[int], context: Optional[Any] = None) -> float:
        """
        Evaluate using ensemble of masking strategies.

        Args:
            coalition: Feature indices to keep
            context: Optional context

        Returns:
            Average score across all strategies
        """
        scores = [cf(coalition, context) for cf in self.char_funcs]
        return float(np.mean(scores))


class BatchedModelCharacteristic(CharacteristicFunction):
    """
    Batched characteristic function for pre-trained models — 2D and 3D data.

    Unlike ``ModelBasedCharacteristic``, which calls ``model.predict()`` once per
    coalition, this class stacks the masked data for **all** coalitions into a
    single array and issues a single ``model.predict()`` call.  This makes it
    suitable for neural-network models (e.g. TensorFlow/Keras) where prediction
    overhead per call is significant.

    Supports:

    * 2-D data ``(n_samples, n_features)`` — standard tabular features.
    * 3-D data ``(n_samples, timesteps, n_features)`` — time-series features;
      the feature mask is broadcast across the time axis.

    ``batch_compute()`` returns:

    * ``shape (n_coalitions,)`` when ``metric_fn`` returns a scalar.
    * ``shape (n_coalitions, n_samples)`` when ``metric_fn`` returns an array —
      e.g. per-sample log-loss.  ``ExactExplainer`` then returns
      ``Dict[int, np.ndarray]`` (per-sample Shapley values).

    Example::

        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from shapG.characteristic import BatchedModelCharacteristic
        >>> from shapG import ExactExplainer
        >>> import networkx as nx
        >>>
        >>> X = np.random.randn(50, 4)
        >>> y = X[:, 0] + X[:, 1]
        >>> model = LinearRegression().fit(X, y)
        >>>
        >>> char_fn = BatchedModelCharacteristic(
        ...     model=model, X=X, y=y,
        ...     metric_fn=lambda yt, yp: float(np.mean((yt - yp) ** 2))
        ... )
        >>>
        >>> G = nx.path_graph(4)
        >>> explainer = ExactExplainer(characteristic_function=char_fn)
        >>> shap_vals = explainer.fit_explain(G)
    """

    def __init__(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable,
        mask_value: Optional[Union[np.ndarray, float]] = None,
        name: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize the batched model characteristic function.

        Args:
            model: Pre-trained model with a ``.predict()`` method.
            X: Data array of shape ``(n_samples, n_features)`` or
               ``(n_samples, timesteps, n_features)``.
            y: True labels/values for *X*, shape ``(n_samples,)`` or
               ``(n_samples, ...)``.
            metric_fn: ``Callable(y_true, y_pred) -> scalar or array``.
                       If the callable returns a Python/numpy scalar the return
                       shape of ``batch_compute`` is ``(n_coalitions,)``.  If it
                       returns an array the return shape is
                       ``(n_coalitions, *array.shape)``.
            mask_value: Baseline used for features *not* in a coalition.
                        ``None`` → per-feature means (mean over samples and,
                        for 3-D data, also over the time axis).
                        ``float`` → constant fill value.
                        ``np.ndarray`` of shape ``(n_features,)`` → per-feature
                        baseline supplied by the caller.
            name: Optional display name.
            chunk_size: Default chunk size used by ``batch_compute()`` when no
                        per-call override is supplied.  ``None`` means process
                        all coalitions in a single ``model.predict()`` call.
        """
        super().__init__(name or "BatchedModel")
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.metric_fn = metric_fn
        self.chunk_size = chunk_size

        if self.X.ndim == 2:
            self.n_samples, self.n_features = self.X.shape
            self._is_3d = False
        elif self.X.ndim == 3:
            self.n_samples, self.timesteps, self.n_features = self.X.shape
            self._is_3d = True
        else:
            raise ValueError(f"X must be 2-D or 3-D, got shape {self.X.shape}")

        if mask_value is None:
            if self._is_3d:
                self.baseline = self.X.mean(axis=(0, 1))  # (n_features,)
            else:
                self.baseline = self.X.mean(axis=0)  # (n_features,)
        elif isinstance(mask_value, (int, float)):
            self.baseline = np.full(self.n_features, float(mask_value))
        else:
            self.baseline = np.asarray(mask_value, dtype=float)
            if self.baseline.shape != (self.n_features,):
                raise ValueError(
                    f"mask_value shape {self.baseline.shape} must be ({self.n_features},)"
                )

    def __call__(self, coalition: Set[int], context: Optional[Any] = None):
        """Evaluate a single coalition.

        Delegates to ``batch_compute()``.  Returns a scalar when
        ``metric_fn`` returns a scalar, or an array when it returns
        per-sample values.

        Args:
            coalition: Set of feature indices to keep.
            context: Unused (for API compatibility).

        Returns:
            Scalar or array characteristic value, depending on ``metric_fn``.
        """
        result = self.batch_compute([coalition], context)[0]
        if np.ndim(result) == 0:
            return float(result)
        return result

    def _compute_chunk(self, coalitions: List[Set[int]]) -> np.ndarray:
        """Evaluate a list of coalitions with a single ``model.predict()`` call.

        Args:
            coalitions: List of coalitions (each a set of integer feature indices).

        Returns:
            ``np.ndarray`` of shape ``(n_coalitions,)`` or ``(n_coalitions, ...)``.
        """
        n_coalitions = len(coalitions)

        if self._is_3d:
            X_batched = np.empty(
                (n_coalitions * self.n_samples, self.timesteps, self.n_features),
                dtype=self.X.dtype,
            )
        else:
            X_batched = np.empty(
                (n_coalitions * self.n_samples, self.n_features),
                dtype=self.X.dtype,
            )

        for ci, coalition in enumerate(coalitions):
            # Binary mask: 1 = keep feature, 0 = replace with baseline
            mask = np.zeros(self.n_features, dtype=float)
            for f in coalition:
                if 0 <= f < self.n_features:
                    mask[f] = 1.0

            start = ci * self.n_samples
            end = (ci + 1) * self.n_samples
            # numpy broadcasts mask (n_features,) over (n_samples [, timesteps], n_features)
            X_batched[start:end] = self.X * mask + self.baseline * (1.0 - mask)

        # Single model call for all coalitions in this chunk
        y_pred_all = self.model.predict(X_batched)

        # Align y shape with predictions to avoid broadcasting cross-product
        # (e.g., model wrapper may flatten 3D→2D while self.y stays 3D)
        y_ref = self.y
        if y_pred_all.ndim != y_ref.ndim:
            y_ref = y_ref.reshape(y_ref.shape[0], -1)

        results = []
        for ci in range(n_coalitions):
            start = ci * self.n_samples
            end = (ci + 1) * self.n_samples
            score = self.metric_fn(y_ref, y_pred_all[start:end])
            results.append(score)

        return np.array(results)

    def batch_compute(
        self,
        coalitions: List[Set[int]],
        context: Optional[Any] = None,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate all coalitions, optionally splitting into memory-safe chunks.

        Args:
            coalitions: List of coalitions (each a set of integer feature indices).
            context: Unused (for API compatibility).
            chunk_size: If set, process at most this many coalitions per
                ``model.predict()`` call.  Useful when ``len(coalitions)`` is
                large and a single batched call would exhaust GPU/CPU memory.
                ``None`` falls back to the ``chunk_size`` set in ``__init__``;
                if that is also ``None``, all coalitions are processed in one
                call.

        Returns:
            ``np.ndarray`` of shape ``(n_coalitions,)`` when ``metric_fn``
            returns a scalar, or ``(n_coalitions, ...)`` when it returns an
            array.
        """
        effective_chunk = chunk_size if chunk_size is not None else self.chunk_size
        if effective_chunk is None or len(coalitions) <= effective_chunk:
            return self._compute_chunk(coalitions)

        parts = []
        for i in range(0, len(coalitions), effective_chunk):
            parts.append(self._compute_chunk(coalitions[i : i + effective_chunk]))
        return np.concatenate(parts)
