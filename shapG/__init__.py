"""
ShapG - Scalable Shapley Value Computation for Graph Data
=========================================================

ShapG provides efficient algorithms for computing Shapley values 
on graph data structures, with both exact and approximate methods.

Main Components:
---------------
* shapley - Core Shapley value computation algorithms
* utils - Supporting functions for data processing and graph operations
* plot - Visualization tools for Shapley values

Example Usage:
-------------
```python
import networkx as nx
from shapG.shapley import shapG, graph_generator
from shapG.plot import plot

# Generate a random graph
G = graph_generator(n_nodes=10, density=0.5)

# Compute approximate Shapley values
shapley_values = shapG(G, depth=1, m=15)

# Visualize the results
plot(shapley_values, top_n=10)
```
"""

from .shapley import shapley_value, shapG, coalition_degree, graph_generator, cis
from .plot import plot
from .utils import (
    corr_generator, 
    matrix_generator, 
    kl, 
    kl_mi_matrix, 
    create_minimal_edge_graph
)

__version__ = '0.13.3'
__all__ = [
    'shapley_value',
    'shapG',
    'cis',
    'coalition_degree',
    'graph_generator',
    'plot',
    'corr_generator',
    'matrix_generator',
    'kl',
    'kl_mi_matrix',
    'create_minimal_edge_graph'
]