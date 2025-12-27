"""
ShapG - Scalable Shapley Value Computation for Graph Data
=========================================================

ShapG provides efficient algorithms for computing Shapley values
on graph data structures, with both exact and approximate methods.

New Modular API (Recommended):
------------------------------
```python
from shapG import ShapGExplainer, GraphBuilder

# Build graph from data
builder = GraphBuilder()
G = builder.from_correlation(data, threshold=0.3)

# Create explainer and compute Shapley values
explainer = ShapGExplainer(depth=2, n_samples=20)
shapley_values = explainer.fit_explain(G)

# Visualize
from shapG import FeatureImportanceVisualizer
viz = FeatureImportanceVisualizer()
viz.plot_importance(shapley_values)
```

Legacy API (Backward Compatible):
---------------------------------
```python
from shapG import shapG, graph_generator, plot

# Generate a random graph
G = graph_generator(n_nodes=10, density=0.5)

# Compute approximate Shapley values
shapley_values = shapG(G, depth=1, m=15)

# Visualize the results
plot(shapley_values, top_n=10)
```
"""

__version__ = '0.13.8'

# New modular API imports
from .explainer import (
    CharacteristicFunction,
    Explainer,
    GraphExplainer,
    ExactExplainer,
    ShapGExplainer,
    CISExplainer,
    RandomCSExplainer,
    QRCSExplainer,
    BlockQRCSExplainer,
    ImprovedQRCSExplainer,
    ImprovedBlockQRCSExplainer,
    LeverageScoreExplainer,
    MultilinearExplainer,
    StratifiedShapleyExplainer
)

from .characteristic import (
    CoalitionDegree,
    NodeCount,
    WeightedSum,
    CustomFunction,
    CenterOfImputationSet
)

from .utils import (
    GraphBuilder,
    CoalitionManager,
    corr_generator,
    matrix_generator,
    kl,
    kl_mi_matrix,
    create_minimal_edge_graph
)

from .visualization import (
    FeatureImportanceVisualizer,
    plot_shapley_values,
    plot
)

# Backward compatibility imports from old API
from .__legacy import (
    shapley_value,
    shapG,
    coalition_degree,
    cis,
    graph_generator,
    get_reachable_nodes_at_depth
)

# All exported symbols
__all__ = [
    # Version
    '__version__',

    # New API - Base classes
    'CharacteristicFunction',
    'Explainer',
    'GraphExplainer',

    # New API - Explainers
    'ExactExplainer',
    'ShapGExplainer',
    'CISExplainer',
    'RandomCSExplainer',
    'QRCSExplainer',
    'BlockQRCSExplainer',
    'ImprovedQRCSExplainer',
    'ImprovedBlockQRCSExplainer',
    'LeverageScoreExplainer',
    'MultilinearExplainer',
    'StratifiedShapleyExplainer',

    # New API - Characteristic functions
    'CoalitionDegree',
    'NodeCount',
    'WeightedSum',
    'CustomFunction',
    'CenterOfImputationSet',

    # New API - Graph construction
    'GraphBuilder',
    'CoalitionManager',

    # New API - Visualization
    'FeatureImportanceVisualizer',
    'plot_shapley_values',

    # Old API - Backward compatibility
    'shapley_value',
    'shapG',
    'cis',
    'coalition_degree',
    'graph_generator',
    'get_reachable_nodes_at_depth',
    'plot',

    # Utils
    'corr_generator',
    'matrix_generator',
    'kl',
    'kl_mi_matrix',
    'create_minimal_edge_graph',
]
