"""
Permutation feature importance explainer.
"""

from typing import Callable, List, Optional
import numpy as np
from tqdm import tqdm


class PermutationExplainer:
    """Permutation feature importance for 2D (tabular) and 3D (time series) data.

    Permutes each feature across the batch dimension and measures the change in
    model performance to determine feature importance.
    """

    def __init__(
        self,
        model: Callable,
        metric: Callable,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        is_global_metric: bool = False,
    ):
        self.model = model
        self.metric = metric
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.is_global_metric = is_global_metric
        if random_state is not None:
            np.random.seed(random_state)

    def _align_pred_shape(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Align y_pred shape to match y."""
        if y.shape == y_pred.shape:
            return y_pred
        if (
            y_pred.ndim == 3
            and y_pred.shape[2] == 1
            and y.ndim == 3
            and y_pred.shape[1] == y.shape[1] * y.shape[2]
        ):
            y_pred = np.squeeze(y_pred, axis=-1).reshape(
                y_pred.shape[0], y.shape[1], y.shape[2]
            )
        elif y_pred.ndim == y.ndim - 1:
            y_pred = np.expand_dims(y_pred, axis=-1)
        elif y_pred.ndim == y.ndim + 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred

    def _normalize_score(self, score, y, is_timeseries, batch_size, n_timesteps):
        """Reduce metric output to a consistent shape for aggregation."""
        if self.is_global_metric:
            return float(score) if not np.isscalar(score) else score
        if not is_timeseries and isinstance(score, np.ndarray):
            if score.ndim == 2 and score.shape[1] > 1:
                return np.mean(score, axis=1)
            if score.ndim == 2 and score.shape[1] == 1:
                return score.squeeze(axis=1)
            return score
        if is_timeseries and isinstance(score, np.ndarray):
            if y.ndim == 3 and y.shape[2] > 1:
                total = batch_size * n_timesteps * y.shape[2]
                if score.ndim == 2 and score.shape[0] == total:
                    score = score.reshape(batch_size, n_timesteps, y.shape[2])
                if score.ndim == 3:
                    return np.mean(score, axis=(1, 2))
            else:
                if score.ndim == 2 and score.shape[0] == batch_size * n_timesteps:
                    score = score.reshape(batch_size, n_timesteps)
                elif score.ndim == 3 and score.shape[2] == 1:
                    score = np.squeeze(score, axis=2)
                if score.ndim == 2 and score.shape[0] == batch_size:
                    return np.mean(score, axis=1)
        return score

    def _permute_feature(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Permute a feature across the batch dimension."""
        X_perm = X.copy()
        perm = np.random.permutation(X.shape[0])
        if X.ndim == 2:
            X_perm[:, feature_idx] = X_perm[perm, feature_idx]
        elif X.ndim == 3:
            X_perm[:, :, feature_idx] = X_perm[perm, :, feature_idx]
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
        return X_perm

    def _format_output(self, shapley_values: np.ndarray):
        """Return dict keyed by feature names, or squeezed array."""
        axes = [i for i, s in enumerate(shapley_values.shape[:-1]) if s == 1]
        if axes:
            shapley_values = np.squeeze(shapley_values, axis=tuple(axes))
        if self.feature_names is not None:
            return {
                feat: shapley_values[..., i]
                for i, feat in enumerate(self.feature_names)
            }
        return shapley_values

    def explain(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Calculate permutation feature importance.

        Args:
            X: Input data (batch_size, features) or (batch_size, time_steps, features)
            y: True labels
            verbose: Whether to show progress

        Returns:
            Dict mapping feature names to importance scores, or array if feature_names is None
        """
        is_timeseries = X.ndim == 3
        batch_size = X.shape[0]
        n_timesteps = y.shape[1] if (is_timeseries and y.ndim >= 2) else 1
        n_features = len(self.feature_names) if self.feature_names else X.shape[-1]

        y_pred_base = self._align_pred_shape(y, self.model.predict(X))
        baseline = self._normalize_score(
            self.metric(y, y_pred_base), y, is_timeseries, batch_size, n_timesteps
        )

        if verbose:
            val = np.mean(baseline) if isinstance(baseline, np.ndarray) else baseline
            print(f"Baseline score: {val:.4f}")

        iterator = (
            tqdm(range(n_features), desc="Computing permutation importance")
            if verbose
            else range(n_features)
        )
        feature_importance = {}

        for feature_idx in iterator:
            scores = []
            for _ in range(self.n_repeats):
                X_perm = self._permute_feature(X, feature_idx)
                y_pred_perm = self._align_pred_shape(y, self.model.predict(X_perm))
                perm_score = self._normalize_score(
                    self.metric(y, y_pred_perm),
                    y,
                    is_timeseries,
                    batch_size,
                    n_timesteps,
                )
                scores.append(baseline - perm_score)

            arr = np.array(scores)
            feat_key = (
                self.feature_names[feature_idx] if self.feature_names else feature_idx
            )
            feature_importance[feat_key] = (
                np.mean(arr) if arr.ndim == 1 else np.mean(arr, axis=0)
            )

        # Build array for _format_output
        feat_keys = (
            self.feature_names if self.feature_names else list(range(n_features))
        )
        first = feature_importance[feat_keys[0]]
        is_scalar = not isinstance(first, np.ndarray) or first.ndim == 0

        if is_timeseries and not is_scalar:
            shapley_values = np.zeros(first.shape + (n_features,))
            for i, k in enumerate(feat_keys):
                shapley_values[..., i] = feature_importance[k]
        elif is_scalar:
            # Scalar importance (e.g. global metric) — (1, n_features) gets squeezed by _format_output
            shapley_values = np.zeros((1, n_features))
            for i, k in enumerate(feat_keys):
                shapley_values[0, i] = float(feature_importance[k])
        else:
            n_out = first.shape[0]
            shapley_values = np.zeros((n_out, n_features))
            for i, k in enumerate(feat_keys):
                imp = feature_importance[k]
                if isinstance(imp, np.ndarray) and imp.shape[0] == n_out:
                    shapley_values[:, i] = imp.flatten()
                else:
                    shapley_values[:, i] = imp

        return self._format_output(shapley_values)
