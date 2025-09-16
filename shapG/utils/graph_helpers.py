"""
Graph helper functions for ShapG library.

This module contains utility functions for graph generation and analysis.
"""

from typing import Union, Optional, Tuple
import networkx as nx
import itertools
import random


def graph_generator(n_nodes: int, density: float, weight_range: Optional[Tuple[int, int]] = (1, 10), seed: int = 2333) -> nx.Graph:
    """Generate a random graph based on the density.

    Args:
        n_nodes: Number of nodes.
        density: Density of the graph (0-1).
        weight_range: Range of edge weights. Defaults to (1, 10).
        seed: Random seed for reproducibility. Defaults to 2333.

    Returns:
        Generated graph.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer")
    if not isinstance(density, (int, float)) or density < 0 or density > 1:
        raise ValueError("density must be between 0 and 1")
    if weight_range is not None:
        if not (
            isinstance(weight_range, (tuple, list))
            and len(weight_range) == 2
        ):
            raise ValueError("weight_range must be a tuple or list of length 2")
        low, high = weight_range
        if not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError("weight_range bounds must be integers")
        if low > high:
            raise ValueError("weight_range lower bound cannot exceed upper bound")

    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    max_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = int(max_edges * density)

    all_possible_edges = list(itertools.combinations(range(n_nodes), 2))
    rng.shuffle(all_possible_edges)
    selected_edges = all_possible_edges[:n_edges]

    for u, v in selected_edges:
        if weight_range is None:
            G.add_edge(u, v)
        else:
            weight = rng.randint(*weight_range)
            G.add_edge(u, v, weight=weight)

    return G


def get_reachable_nodes_at_depth(G: nx.Graph, node: Union[int, str], depth: int) -> set:
    """Get all nodes at exactly k-hop distance from a given node.

    Args:
        G: Graph.
        node: The source node.
        depth: Hop distance (k).

    Returns:
        Nodes that are exactly 'depth' hops away from the source node.
    """
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=depth)
    return {n for n, d in path_lengths.items() if d == depth}


def coalition_degree(G: nx.Graph, S: Union[set, list]) -> float:
    """Calculate the characteristic function of a coalition in a graph.

    This function computes the sum of weighted degrees for nodes in coalition S.

    Args:
        G: The graph.
        S: Set or list of nodes forming the coalition.

    Returns:
        The characteristic value of the coalition.
    """
    if not S:
        return 0

    subgraph = G.subgraph(S)
    return sum(dict(subgraph.degree(weight='weight')).values()) / 2