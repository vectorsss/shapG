"""Compare exact and approximate Shapley values on a real feature graph.

ExactExplainer enumerates all 2^n coalitions, so it is only feasible for
small graphs but gives a ground truth to check ShapGExplainer against.
The housing dataset has 13 features (2^13 coalitions), which is still exact.

Run:
    python examples/02_exact_vs_approx.py
"""

import os
import sys
from pathlib import Path

# Use the shapG source in this repository, not a pip-installed copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from shapG import ExactExplainer, ShapGExplainer, GraphBuilder

DATA = Path(__file__).resolve().parent.parent / "datasets" / "housing_price.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    features = df.drop(columns=["MEDV"])

    # Same graph for both explainers; default value function is CoalitionDegree.
    graph = GraphBuilder().from_correlation(features, threshold=0.3)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

    exact = ExactExplainer().fit_explain(graph)
    approx = ShapGExplainer(depth=2, n_samples=8).fit_explain(graph)

    print(f"{'feature':>8} {'exact':>10} {'approx':>10} {'abs err':>10}")
    total_err = 0.0
    for node in graph.nodes():
        err = abs(exact[node] - approx[node])
        total_err += err
        print(f"{node:>8} {exact[node]:>10.4f} {approx[node]:>10.4f} {err:>10.4f}")

    print(f"\nmean absolute error: {total_err / graph.number_of_nodes():.4f}")


if __name__ == "__main__":
    main()
