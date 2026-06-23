"""Graph construction."""

from collections import Counter

from shapG import GraphBuilder

GRAPH_METHODS_NEED_Y = {"from_rank_deletion"}


def detect_feature_types(X):
    """Classify each column by cardinality into a per-index type dict.

    Features are interleaved in all four benchmark datasets, so the mixed
    similarity needs per-index types (not contiguous feature_ranges):
    nunique == 2 -> "binary", 3..10 -> "categorical", > 10 -> "numerical".
    """
    types = {}
    for i, col in enumerate(X.columns):
        nunique = X[col].nunique(dropna=False)
        if nunique == 2:
            types[i] = "binary"
        elif nunique <= 10:
            types[i] = "categorical"
        else:
            types[i] = "numerical"
    return types


def build_graphs(cfg, X, y, graph_names=None):
    """Construct the configured graphs.

    If *graph_names* is given, only those graphs are built (used to skip
    unused graphs — e.g. the rank_deletion graphs are only needed when a
    graph-dependent explainer such as ShapG actually runs).
    """
    builder = GraphBuilder()
    graphs = {}
    for name, gcfg in cfg["graphs"].items():
        if graph_names is not None and name not in graph_names:
            continue
        method = gcfg["method"]
        params = dict(gcfg.get("params", {}))
        # Mixed similarity needs per-index feature types (interleaved columns)
        if params.get("use_mixed_similarity") and "feature_types" not in params:
            ftypes = detect_feature_types(X)
            params["feature_types"] = ftypes
            counts = Counter(ftypes.values())
            print(
                f"  Feature types for '{name}': "
                f"{counts.get('numerical', 0)} numerical, "
                f"{counts.get('categorical', 0)} categorical, "
                f"{counts.get('binary', 0)} binary"
            )
        fn = getattr(builder, method)
        if method in GRAPH_METHODS_NEED_Y:
            G = fn(X, y, **params)
        else:
            G = fn(X, **params)
        graphs[name] = G
        print(
            f"  Graph '{name}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
    return graphs
