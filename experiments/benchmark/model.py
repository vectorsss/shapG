"""Model construction and characteristic function building."""

import lightgbm as lgb
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from shapG.characteristic import GraphModelCharacteristic
from shapG.characteristic.characteristic_functions import (
    CoalitionDegree,
    CustomFunction,
)

MODEL_CLASSES = {
    "LGBMRegressor": lgb.LGBMRegressor,
    "LGBMClassifier": lgb.LGBMClassifier,
}


def build_model(cfg):
    mcfg = cfg["model"]
    cls = MODEL_CLASSES[mcfg["type"]]
    return cls(**mcfg.get("params", {}))


def get_metric_fn(model, cfg=None):
    """Return (metric_fn, display_name).

    All returned metric functions share the signature
    ``metric_fn(y_true, y_pred, n_features=None)`` so that call sites
    can optionally pass the current number of features.  Only the
    adjusted-R² variant actually uses *n_features*.
    """
    use_adjusted = (cfg or {}).get("evaluation", {}).get("adjusted_r2", False)

    if isinstance(model, lgb.LGBMRegressor):
        if use_adjusted:

            def metric_fn(y_true, y_pred, n_features=None):
                r2 = r2_score(y_true, y_pred)
                n = len(y_true)
                if n_features is None or n_features == 0 or n <= n_features + 1:
                    return r2
                return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

            return metric_fn, "$R^2_{adj}$"
        else:

            def metric_fn(y_true, y_pred, n_features=None):
                return r2_score(y_true, y_pred)

            return metric_fn, "$R^2$"

    def metric_fn(y_true, y_pred, n_features=None):
        return accuracy_score(y_true, y_pred)

    return metric_fn, "Accuracy"


def build_char_func(cfg, model, X_train, X_test, y_train, y_test, X, y):
    cc = cfg["characteristic_function"]
    if cc["type"] == "coalition_degree":
        return CoalitionDegree()

    params = cc.get("params", {})
    no_retrain = params.get("no_retrain", True)
    strategy = params.get("imputation_strategy", "mean")

    if no_retrain:
        # Masking mode: train once, mask features for each coalition
        model.fit(X_train, y_train)
        baseline = X_train.mean().values if strategy == "mean" else None
        return GraphModelCharacteristic(
            model=model,
            X=X_test,
            y=y_test.values if hasattr(y_test, "values") else y_test,
            masking_strategy=strategy,
            baseline=baseline,
        )
    else:
        # Retraining mode: retrain a fresh model for each coalition
        metric_fn, _ = get_metric_fn(model, cfg)
        mcfg = cfg["model"]
        model_cls = MODEL_CLASSES[mcfg["type"]]
        model_params = dict(mcfg.get("params", {}))
        split_cfg = mcfg["train_test_split"]

        def retrain_char_func(coalition, context):
            if len(coalition) == 0:
                return 0.0
            cols = list(coalition)
            X_sub = X[cols]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub,
                y,
                test_size=split_cfg["test_size"],
                random_state=split_cfg["random_state"],
            )
            m = model_cls(**model_params)
            m.fit(X_tr, y_tr)
            return float(metric_fn(y_te, m.predict(X_te), n_features=len(coalition)))

        return CustomFunction(retrain_char_func, name="Retrain")
