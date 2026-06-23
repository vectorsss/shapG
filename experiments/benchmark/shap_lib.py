"""Official `shap` library baselines (KernelSHAP, SamplingSHAP).

These wrappers run the *official* shap-package implementations with their
default parameters on a fixed trained model (mask/impute semantics), in the
style of the icomp-2025 reference wrappers. They intentionally do NOT use the
retrain-based characteristic function: the comparison narrative is that our
methods are more stable and faster than the off-the-shelf library baselines.

Standard usage choices (required by the library, logged at runtime):
- background: random sample of the training split (`shap.sample`)
- explained set: random subsample of the test split; global importance is
  mean(|SHAP|) over the explained instances
- classifiers are explained through `predict_proba` (positive class),
  regressors through `predict`

`nsamples` / `l1_reg` are left at the library defaults ("auto").
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without shap installed
    SHAP_AVAILABLE = False
    logger.warning("shap library not available. Install with: pip install shap")


class _SHAPLibBaseline:
    """Base wrapper: fit the model on the train split, explain a test subsample.

    Parameters
    ----------
    model : sklearn-compatible estimator (unfitted template; cloned internally)
    X, y : full dataset (the wrapper re-creates the benchmark's train/test split)
    split_cfg : dict with ``test_size`` and ``random_state`` (same as the
        benchmark model config, so the fixed model sees the same split as the
        retrain characteristic function)
    n_background : int, background sample size for masking (default 100)
    n_explain : int, number of test instances to explain (default 100)
    seed : int, random seed for background/instance subsampling
    verbose : bool
    """

    explainer_kind = None  # "kernel" | "sampling", set by subclasses
    requires_model_data = (
        True  # benchmark flag: build with (model, X, y), not char_func
    )

    def __init__(
        self,
        model,
        X,
        y,
        split_cfg,
        n_background=100,
        n_explain=100,
        seed=42,
        verbose=False,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError(
                "shap library is required. Install with: pip install shap"
            )
        self.model = clone(model)
        self.X = X
        self.y = y
        self.split_cfg = split_cfg
        self.n_background = n_background
        self.n_explain = n_explain
        self.seed = seed
        self.verbose = verbose
        self.n_explained_ = None
        self._shap_explainer = None

    def _model_fn(self):
        """Prediction function handed to shap (probabilities for classifiers)."""
        use_proba = is_classifier(self.model)

        def fn(X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            if use_proba:
                return self.model.predict_proba(X)[:, 1]
            return self.model.predict(X)

        return fn

    def _build_explainer(self, model_fn, background):
        raise NotImplementedError

    def fit_explain(self, _G=None):
        """Run the library explainer; returns {feature_name: mean |SHAP|}."""
        X_train, X_test, y_train, _y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.split_cfg["test_size"],
            random_state=self.split_cfg["random_state"],
        )
        self.model.fit(X_train, y_train)

        rng = np.random.default_rng(self.seed)
        background = shap.sample(
            X_train, min(self.n_background, len(X_train)), random_state=self.seed
        )
        n_explain = min(self.n_explain, len(X_test))
        explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)
        X_explain = X_test.iloc[explain_idx]
        self.n_explained_ = n_explain

        if self.verbose:
            logger.info(
                f"shap.{self.explainer_kind}: background={len(background)}, "
                f"explaining {n_explain}/{len(X_test)} test instances, "
                f"nsamples=auto (library default)"
            )

        self._shap_explainer = self._build_explainer(self._model_fn(), background)
        shap_values = self._shap_explainer.shap_values(
            X_explain.values, silent=not self.verbose
        )

        # Multi-output safety (predict_proba[:, 1] keeps us single-output,
        # but mirror the icomp-2025 handling just in case)
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = np.mean(np.abs(shap_values), axis=2)

        # Global importance: mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.X.columns, importance.astype(float)))


class SHAPLibKernelExplainer(_SHAPLibBaseline):
    """Official shap.KernelExplainer with library-default parameters."""

    explainer_kind = "KernelExplainer"

    def _build_explainer(self, model_fn, background):
        return shap.KernelExplainer(model_fn, background)


class SHAPLibSamplingExplainer(_SHAPLibBaseline):
    """Official shap.SamplingExplainer with library-default parameters."""

    explainer_kind = "SamplingExplainer"

    def _build_explainer(self, model_fn, background):
        return shap.SamplingExplainer(model_fn, background)
