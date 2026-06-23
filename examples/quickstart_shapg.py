"""Minimal quickstart: compute Shapley values with ShapG on the housing data.

    python examples/quickstart_shapg.py

Builds a feature-similarity graph, then runs the headline ShapG algorithm with
the default graph characteristic function (CoalitionDegree). For the full
config-driven benchmark harness see ../experiments/.
"""

from pathlib import Path

import pandas as pd

from shapG import GraphBuilder, ShapGExplainer

DATA = Path(__file__).resolve().parent.parent / "datasets" / "housing_price.csv"


def main():
    X = pd.read_csv(DATA).drop(columns=["MEDV"])

    # Nodes = features, edges = feature similarity (Kendall-tau minimal edge).
    G = GraphBuilder().from_kendalltau_minimal_edge(X, reverse=True, version="v3")
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # depth=1 restricts each node's coalitions to its 1-hop neighborhood.
    explainer = ShapGExplainer(depth=1, n_samples=3)
    values = explainer.fit_explain(G)

    print("\nShapG feature importance (descending):")
    for node, val in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        name = X.columns[int(node)] if str(node).isdigit() else node
        print(f"  {str(name):12s} {val:+.4f}")


if __name__ == "__main__":
    main()
