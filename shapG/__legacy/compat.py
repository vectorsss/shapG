"""
Backward compatibility layer for old ShapG API.

This module provides wrapper functions that maintain the old API while using the new modular architecture.
"""

from typing import Optional, Callable, Union, Dict, Any
import warnings
import networkx as nx
import numpy as np
import pandas as pd

from ..explainer.explainers import ExactExplainer, ShapGExplainer, CISExplainer
from ..characteristic.characteristic_functions import CoalitionDegree, CustomFunction
from ..utils.graph_construction import GraphBuilder
from ..visualization.visualization import plot_shapley_values as _plot_shapley_values


def shapley_value(G: nx.Graph, f: Optional[Callable] = None, verbose: bool = False) -> Dict[int, float]:
    """Calculate exact Shapley values for all nodes (backward compatible).

    Args:
        G: NetworkX graph
        f: Characteristic function (default: coalition_degree)
        verbose: Whether to show progress bar

    Returns:
        Dictionary of Shapley values for each node
    """
    # Handle characteristic function
    if f is None:
        char_func = CoalitionDegree()
    else:
        # Wrap the old-style function
        char_func = CustomFunction(lambda coalition, context: f(context, coalition))

    # Use ExactExplainer
    explainer = ExactExplainer(characteristic_function=char_func, verbose=verbose)
    return explainer.fit_explain(G)


def shapG(
    G: nx.Graph,
    f: Optional[Callable] = None,
    depth: int = 1,
    m: int = 15,
    approximate_by_ratio: bool = True,
    scale: bool = True,
    verbose: bool = False
) -> Dict[int, float]:
    """Approximate Shapley values using ShapG algorithm (backward compatible).

    Args:
        G: NetworkX graph
        f: Characteristic function (default: coalition_degree)
        depth: Maximum neighborhood depth
        m: Maximum number of nodes to sample
        approximate_by_ratio: Whether to use ratio-based approximation
        scale: Whether to apply scaling
        verbose: Whether to show progress

    Returns:
        Dictionary of approximated Shapley values for each node
    """
    # Handle characteristic function
    if f is None:
        char_func = CoalitionDegree()
    else:
        # Wrap the old-style function
        char_func = CustomFunction(lambda coalition, context: f(context, coalition))

    # Use ShapGExplainer
    explainer = ShapGExplainer(
        characteristic_function=char_func,
        depth=depth,
        n_samples=m,
        approximate_by_ratio=approximate_by_ratio,
        scale=scale,
        verbose=verbose
    )
    return explainer.fit_explain(G)


def coalition_degree(G: nx.Graph, S: Union[set, list]) -> float:
    """Calculate the characteristic function of a coalition (backward compatible).

    Args:
        G: NetworkX graph
        S: Set or list of nodes forming the coalition

    Returns:
        The characteristic value of the coalition
    """
    char_func = CoalitionDegree()
    return char_func(set(S) if not isinstance(S, set) else S, G)


def cis(G: nx.Graph, f: Optional[Callable] = None) -> Dict[int, float]:
    """Calculate CIS-values for all nodes (backward compatible).

    Args:
        G: NetworkX graph
        f: Characteristic function (default: coalition_degree)

    Returns:
        Dictionary of CIS-values for each node
    """
    # This is a simplified CIS implementation
    # For full CIS with model predictions, use CISExplainer directly

    if f is None:
        f = coalition_degree

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    grand_coalition_value = f(G, set(nodes))
    individual_values = {node: f(G, {node}) for node in nodes}
    total_individual_value = sum(individual_values.values())

    surplus = grand_coalition_value - total_individual_value
    equal_share = surplus / n_nodes if n_nodes > 0 else 0

    cis_values = {
        node: individual_values[node] + equal_share
        for node in nodes
    }
    return cis_values


def graph_generator(
    n_nodes: int,
    density: float,
    weight_range: Optional[tuple] = (1, 10),
    seed: int = 2333
) -> nx.Graph:
    """Generate a random graph (backward compatible).

    Args:
        n_nodes: Number of nodes
        density: Density of the graph (0-1)
        weight_range: Range of edge weights
        seed: Random seed for reproducibility

    Returns:
        Generated NetworkX graph
    """
    builder = GraphBuilder()
    return builder.random_graph(n_nodes, density, weight_range, seed)


def get_reachable_nodes_at_depth(G: nx.Graph, node: int, depth: int) -> set:
    """Get all nodes at exactly k-hop distance from a given node (backward compatible).

    Args:
        G: NetworkX graph
        node: The source node
        depth: Hop distance (k)

    Returns:
        Set of nodes that are exactly 'depth' hops away from the source node
    """
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=depth)
    return {n for n, d in path_lengths.items() if d == depth}


# Re-export plot function with backward compatibility
def plot(
    shapley_values: Union[Dict[int, float], pd.Series],
    feature_names: Optional[list] = None,
    top_n: int = 10,
    style: str = 'seaborn-v0_8',
    figsize: tuple = (10, 6),
    color: str = '#1f77b4',
    show_values: bool = True,
    show_plot: bool = True
) -> Optional[tuple]:
    """Plot Shapley values (backward compatible).

    Args:
        shapley_values: Shapley values to plot
        feature_names: Optional feature names
        top_n: Number of top features to show
        style: Matplotlib style
        figsize: Figure size
        color: Bar color
        show_values: Whether to show values on bars
        show_plot: Whether to display the plot

    Returns:
        Figure and axes if show_plot is False, None otherwise
    """
    return _plot_shapley_values(
        shapley_values,
        feature_names=feature_names,
        top_n=top_n,
        style=style,
        figsize=figsize,
        color=color,
        show_values=show_values,
        show_plot=show_plot
    )


# Deprecation warning decorator
def deprecated(message: str):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# Optional: Add deprecation warnings for direct imports
def _warn_old_api():
    """Issue warning about using old API."""
    warnings.warn(
        "You are using the old ShapG API. Consider migrating to the new modular API "
        "with Explainer classes for better flexibility and performance. "
        "See the migration guide for details.",
        FutureWarning,
        stacklevel=3
    )