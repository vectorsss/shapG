"""Backward and forward KPI evaluation."""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .model import MODEL_CLASSES, get_metric_fn


def _build_evaluator(cfg, X, y):
    """Build a feature-set evaluator (train once, mask to evaluate subsets)."""
    mcfg = cfg["model"]
    model = MODEL_CLASSES[mcfg["type"]](**mcfg.get("params", {}))
    metric_fn, metric_name = get_metric_fn(model, cfg)
    split_cfg = mcfg["train_test_split"]
    ev = cfg["evaluation"]
    imputation = ev.get("imputation_strategy", "mean")
    retrain = ev.get("retrain_kpi", False)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    model.fit(x_train, y_train)
    all_features = list(X.columns)
    col_means = x_train.mean(numeric_only=True)
    rng = np.random.default_rng(cfg.get("seed", 42))

    def evaluate(keep_features):
        n_feat = len(keep_features) if keep_features else 0
        if retrain:
            if not keep_features:
                return 0.0
            reduced = X[keep_features]
            xtr, xte, ytr, yte = train_test_split(
                reduced,
                y,
                test_size=split_cfg["test_size"],
                random_state=split_cfg["random_state"],
            )
            model.fit(xtr, ytr)
            return metric_fn(yte, model.predict(xte), n_features=n_feat)
        else:
            X_m = x_test.copy()
            for c in [col for col in all_features if col not in keep_features]:
                if imputation == "zero":
                    X_m[c] = 0
                elif imputation == "permutation":
                    X_m[c] = rng.permutation(X_m[c].values)
                else:
                    if pd.api.types.is_numeric_dtype(X_m[c]):
                        X_m[c] = col_means.get(c, X_m[c].mean())
            return metric_fn(y_test, model.predict(X_m), n_features=n_feat)

    full_metric = evaluate(all_features)
    return evaluate, all_features, full_metric, metric_name


def compute_kpi(cfg, X, y, feature_rankings):
    ev = cfg["evaluation"]
    if not ev.get("enabled", True):
        return {}, ""

    mcfg = cfg["model"]
    model = MODEL_CLASSES[mcfg["type"]](**mcfg.get("params", {}))
    metric_fn, metric_name = get_metric_fn(model, cfg)
    split_cfg = mcfg["train_test_split"]
    limit = ev.get("limit", 10)
    beta = ev.get("weighted_slope_beta", 0.8)
    retrain = ev.get("retrain_kpi", False)
    imputation = ev.get("imputation_strategy", "mean")

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    n_all_features = len(X.columns)
    initial_metric = metric_fn(y_test, y_pred, n_features=n_all_features)

    col_means = x_train.mean(numeric_only=True)
    rng = np.random.default_rng(cfg.get("seed", 42))

    def mask_features(X_df, keep_cols):
        X_m = X_df.copy()
        for c in [col for col in X_df.columns if col not in keep_cols]:
            if imputation == "zero":
                X_m[c] = 0
            elif imputation == "permutation":
                X_m[c] = rng.permutation(X_m[c].values)
            else:
                if pd.api.types.is_numeric_dtype(X_m[c]):
                    X_m[c] = col_means.get(c, X_m[c].mean())
        return X_m

    kpi_results = {}
    for method, feature_order in feature_rankings.items():
        order = feature_order[:limit]
        metrics = [initial_metric]
        deltas = []

        for i in range(1, len(order) + 1):
            to_drop = order[:i]
            missing = [c for c in to_drop if c not in X.columns]
            if missing:
                continue

            if retrain:
                reduced = X.drop(columns=to_drop)
                xtr, xte, ytr, yte = train_test_split(
                    reduced,
                    y,
                    test_size=split_cfg["test_size"],
                    random_state=split_cfg["random_state"],
                )
                model.fit(xtr, ytr)
                new_metric = metric_fn(
                    yte, model.predict(xte), n_features=len(reduced.columns)
                )
            else:
                keep = [c for c in x_test.columns if c not in to_drop]
                x_masked = mask_features(x_test, set(keep))
                new_metric = metric_fn(
                    y_test, model.predict(x_masked), n_features=len(keep)
                )

            deltas.append(metrics[-1] - new_metric)
            metrics.append(new_metric)

        weights = [beta**i for i in range(len(deltas))]
        kpi_results[method] = {
            "metrics": metrics,
            "slope": float(np.dot(deltas, weights)) if deltas else 0.0,
            "features_dropped": order,
        }

    return kpi_results, metric_name


def compute_forward_kpi(cfg, X, y, feature_rankings, skip_references=False):
    ev = cfg["evaluation"]
    if not ev.get("enabled", True):
        return {}, None, ""

    evaluate, all_features, full_metric, metric_name = _build_evaluator(cfg, X, y)
    limit = ev.get("limit", 10)
    beta = ev.get("weighted_slope_beta", 0.8)
    n_steps = min(limit, len(all_features))
    baseline = evaluate([])

    results = {}

    # Method curves
    for method, feature_order in feature_rankings.items():
        order = feature_order[:limit]

        metrics = [baseline]
        included = []
        deltas = []

        for feat in order:
            if feat not in all_features:
                continue
            included.append(feat)
            m = evaluate(included)
            deltas.append(m - metrics[-1])
            metrics.append(m)

        weights = [beta**i for i in range(len(deltas))]
        results[method] = {
            "metrics": metrics,
            "slope": float(np.dot(deltas, weights)) if deltas else 0.0,
        }

    # Greedy/Random reference curves are off by default (expensive: each is an
    # O(n^2) greedy search, very slow under retrain). Enable with
    # evaluation.reference_curves: true.
    if skip_references or not ev.get("reference_curves", False):
        return results, full_metric, metric_name

    # Greedy best (add most impactful first)
    print("  Computing forward greedy best...")
    t0 = time.time()
    metrics_best = [baseline]
    included, remaining = [], list(all_features)
    for _ in range(n_steps):
        best_m, best_f = -np.inf, None
        for f in remaining:
            m = evaluate(included + [f])
            if m > best_m:
                best_m, best_f = m, f
        remaining.remove(best_f)
        included.append(best_f)
        metrics_best.append(best_m)
    t_best = time.time() - t0
    deltas = [
        metrics_best[i + 1] - metrics_best[i] for i in range(len(metrics_best) - 1)
    ]
    weights = [beta**i for i in range(len(deltas))]
    results["Greedy Best"] = {
        "metrics": metrics_best,
        "slope": float(np.dot(deltas, weights)),
        "time": t_best,
    }
    print(f"    Greedy Best: {t_best:.2f}s")

    # Greedy worst (add least impactful first)
    print("  Computing forward greedy worst...")
    t0 = time.time()
    metrics_worst = [baseline]
    included, remaining = [], list(all_features)
    for _ in range(n_steps):
        worst_m, worst_f = np.inf, None
        for f in remaining:
            m = evaluate(included + [f])
            if m < worst_m:
                worst_m, worst_f = m, f
        remaining.remove(worst_f)
        included.append(worst_f)
        metrics_worst.append(worst_m)
    t_worst = time.time() - t0
    deltas = [
        metrics_worst[i + 1] - metrics_worst[i] for i in range(len(metrics_worst) - 1)
    ]
    weights = [beta**i for i in range(len(deltas))]
    results["Greedy Worst"] = {
        "metrics": metrics_worst,
        "slope": float(np.dot(deltas, weights)),
        "time": t_worst,
    }
    print(f"    Greedy Worst: {t_worst:.2f}s")

    # Random baseline (average over trials)
    n_trials = 10
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)
    print(f"  Computing forward random curve ({n_trials} trials)...")
    t0 = time.time()
    all_trial_metrics = []
    for _ in range(n_trials):
        order = rng.permutation(all_features).tolist()
        trial_metrics = [baseline]
        inc = []
        for step in range(n_steps):
            inc.append(order[step])
            trial_metrics.append(evaluate(inc))
        all_trial_metrics.append(trial_metrics)
    t_random = time.time() - t0
    avg = np.mean(all_trial_metrics, axis=0).tolist()
    deltas = [avg[i + 1] - avg[i] for i in range(len(avg) - 1)]
    weights = [beta**i for i in range(len(deltas))]
    results["Random"] = {
        "metrics": avg,
        "slope": float(np.dot(deltas, weights)),
        "time": t_random,
    }
    print(f"    Random ({n_trials} trials): {t_random:.2f}s")

    return results, full_metric, metric_name
