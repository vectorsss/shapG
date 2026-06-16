"""Quickstart: approximate Shapley values on a feature-correlation graph.

Builds a graph where nodes are features and edges connect correlated
features, then uses ShapGExplainer to estimate each feature's importance.

Run:
    python examples/01_quickstart.py
"""

import os
import sys
from pathlib import Path

# Use the shapG source in this repository, not a pip-installed copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from shapG import ShapGExplainer, GraphBuilder, FeatureImportanceVisualizer

DATA = Path(__file__).resolve().parent.parent / "datasets" / "housing_price.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    features = df.drop(columns=["MEDV"])  # MEDV is the regression target

    # Nodes = features; edges connect features with |correlation| >= threshold.
    graph = GraphBuilder().from_correlation(features, threshold=0.5)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Approximate Shapley values via local-neighborhood sampling.
    explainer = ShapGExplainer(depth=1, n_samples=10)
    shapley_values = explainer.fit_explain(graph)

    print("Structural feature importance (CoalitionDegree):")
    for feature, value in sorted(
        shapley_values.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {feature:>8}: {value:+.4f}")

    FeatureImportanceVisualizer().plot_importance(
        shapley_values,
        title="ShapG feature importance (housing)",
        filename="examples/quickstart_importance.png",
        show_plot=False,
    )
    print("Saved plot to examples/quickstart_importance.png")


if __name__ == "__main__":
    main()
