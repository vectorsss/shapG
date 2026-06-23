"""Explainer execution and feature ranking.

The registry below covers the graph-based Shapley estimators compared by the
benchmark.  Every class accepts ``characteristic_function=`` and ``verbose=``
and exposes ``fit_explain(G) -> {node: value}`` (inherited from
``Explainer`` / ``GraphExplainer``), so they all run through the same code path.
"""

import time

from shapG import (
    BlockQRCSExplainer,
    CISExplainer,
    ExactExplainer,
    ImprovedBlockQRCSExplainer,
    ImprovedQRCSExplainer,
    LeverageScoreExplainer,
    MultilinearExplainer,
    QRCSExplainer,
    RandomCSExplainer,
    ShapGExplainer,
    StratifiedShapleyExplainer,
)

from .shap_lib import SHAPLibKernelExplainer, SHAPLibSamplingExplainer

EXPLAINER_CLASSES = {
    # Headline graph algorithm and the cheap CIS baseline
    "ShapGExplainer": ShapGExplainer,
    "CISExplainer": CISExplainer,
    # Compressed-sensing estimators
    "RandomCSExplainer": RandomCSExplainer,
    "QRCSExplainer": QRCSExplainer,
    "BlockQRCSExplainer": BlockQRCSExplainer,
    "ImprovedQRCSExplainer": ImprovedQRCSExplainer,
    "ImprovedBlockQRCSExplainer": ImprovedBlockQRCSExplainer,
    # Sampling / leverage / multilinear estimators
    "StratifiedShapleyExplainer": StratifiedShapleyExplainer,
    "LeverageScoreExplainer": LeverageScoreExplainer,
    "MultilinearExplainer": MultilinearExplainer,
    # Exact enumeration baseline (small problems only)
    "ExactExplainer": ExactExplainer,
    # Official shap-library baselines (fixed model + masking, library defaults)
    "SHAPLibKernelExplainer": SHAPLibKernelExplainer,
    "SHAPLibSamplingExplainer": SHAPLibSamplingExplainer,
}


def run_explainers(cfg, char_func, graphs, X, y, model, only=None):
    """Run every (explainer x graph) combination.

    Explainers with ``graph_dependent: false`` run once on the first graph.
    If ``only`` is set, only run explainers whose names are in that list.
    """
    results = {}
    first_graph_name = next(iter(graphs))
    first_graph = graphs[first_graph_name]

    for exp_name, ecfg in cfg["explainers"].items():
        if only and exp_name not in only:
            continue

        cls = EXPLAINER_CLASSES[ecfg["class"]]
        params = dict(ecfg.get("params", {}))
        graph_dependent = ecfg.get("graph_dependent", True)
        verbose = params.pop("verbose", False)

        # shap-library baselines: fixed model + data, no characteristic function
        if getattr(cls, "requires_model_data", False):
            print(f"  Running {exp_name} ...")
            explainer = cls(
                model=model,
                X=X,
                y=y,
                split_cfg=cfg["model"]["train_test_split"],
                verbose=verbose,
                **params,
            )
            t0 = time.time()
            values = explainer.fit_explain()
            elapsed = time.time() - t0
            results[exp_name] = {
                "values": values,
                "time": elapsed,
                "explainer": explainer,
            }
            print(f"    done in {elapsed:.2f}s")
            continue

        if graph_dependent:
            run_graphs = graphs.items()
        else:
            run_graphs = [(first_graph_name, first_graph)]

        for graph_name, G in run_graphs:
            label = f"{exp_name} | {graph_name}" if graph_dependent else exp_name
            print(f"  Running {label} ...")
            explainer = cls(
                characteristic_function=char_func, verbose=verbose, **params
            )

            t0 = time.time()
            values = explainer.fit_explain(G)
            elapsed = time.time() - t0

            results[label] = {"values": values, "time": elapsed, "explainer": explainer}
            print(f"    done in {elapsed:.2f}s")
    return results


def node_to_feature(node, columns):
    try:
        idx = int(node)
        if 0 <= idx < len(columns):
            return columns[idx]
    except (ValueError, TypeError):
        if node in columns:
            return str(node)
    return None


def to_feature_ranking(values, columns):
    sorted_vals = sorted(values.items(), key=lambda x: x[1], reverse=True)
    return [
        node_to_feature(n, columns)
        for n, _ in sorted_vals
        if node_to_feature(n, columns) is not None
    ]
