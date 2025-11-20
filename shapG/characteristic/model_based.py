"""
Model-based characteristic functions with feature masking (no retraining).

These classes provide efficient characteristic functions for pre-trained models,
avoiding the computational cost of retraining for each coalition evaluation.
"""

from typing import Set, Optional, Any, Callable, Union
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
        masking_strategy: str = 'mean',
        metric_fn: Optional[Callable] = None,
        name: Optional[str] = None
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
        if strategy == 'zero':
            return np.zeros(self.X.shape[1])
        elif strategy == 'mean':
            return np.mean(self.X, axis=0)
        elif strategy == 'median':
            return np.median(self.X, axis=0)
        elif strategy == 'noise':
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
        masking_strategy: str = 'mean',
        metric_fn: Optional[Callable] = None,
        name: Optional[str] = None,
        baseline: Optional[np.ndarray] = None
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
            if masking_strategy == 'mean':
                self.baseline = X.mean().values
            elif masking_strategy == 'median':
                self.baseline = X.median().values
            elif masking_strategy == 'zero':
                self.baseline = np.zeros(len(self.feature_names))
            elif masking_strategy == 'permutation':
                # For permutation, we'll shuffle on-the-fly in __call__
                self.baseline = None
            elif masking_strategy == 'noise':
                self.baseline = np.random.randn(len(self.feature_names)) * X.std().values
            else:
                raise ValueError(f"Unknown masking strategy: {masking_strategy}")

        # For permutation, store random seed for reproducibility
        self.rng = np.random.default_rng(42) if masking_strategy == 'permutation' else None

    def __call__(self, coalition: Set[Union[int, str]], context: Optional[nx.Graph] = None) -> float:
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
        if self.masking_strategy == 'permutation':
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
        name: Optional[str] = None
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
        self.strategies = strategies or ['zero', 'mean', 'median']

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
