"""
Backward compatibility layer for legacy API.
"""

from .compat import (
    shapley_value,
    shapG,
    coalition_degree,
    cis,
    graph_generator,
    get_reachable_nodes_at_depth,
    plot,
)

__all__ = [
    "shapley_value",
    "shapG",
    "coalition_degree",
    "cis",
    "graph_generator",
    "get_reachable_nodes_at_depth",
    "plot",
]
