"""Use a custom characteristic function.

Any function ``(coalition, context) -> float`` can drive the Shapley
computation. Here the value of a coalition is the density of the subgraph
it induces, scaled by its size.

Run:
    python examples/03_custom_characteristic.py
"""

import os
import sys

# Use the shapG source in this repository, not a pip-installed copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx

from shapG import ShapGExplainer, CustomFunction


def coalition_density(coalition, graph):
    """Density of the subgraph induced by the coalition, scaled by size."""
    if len(coalition) < 2:
        return 0.0
    subgraph = graph.subgraph(coalition)
    return nx.density(subgraph) * len(coalition)


def main() -> None:
    graph = nx.karate_club_graph()

    char_func = CustomFunction(coalition_density, name="scaled_density")
    explainer = ShapGExplainer(characteristic_function=char_func, depth=1, n_samples=12)
    shapley_values = explainer.fit_explain(graph)

    top = sorted(shapley_values.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("Top 5 nodes by custom characteristic function:")
    for node, value in top:
        print(f"  node {node:>2}: {value:+.4f}")


if __name__ == "__main__":
    main()
