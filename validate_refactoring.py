#!/usr/bin/env python
"""
Validation script to ensure the refactored ShapG works correctly.
Tests both old and new APIs.
"""

import numpy as np
import pandas as pd
import networkx as nx
import sys

def test_old_api():
    """Test backward compatibility with old API."""
    print("Testing OLD API...")

    from shapG import (
        shapley_value,
        shapG,
        coalition_degree,
        cis,
        graph_generator,
        get_reachable_nodes_at_depth
    )

    # Generate graph
    G = graph_generator(15, 0.4, seed=42)
    assert G.number_of_nodes() == 15, "Graph generation failed"

    # Test coalition degree
    value = coalition_degree(G, {0, 1, 2})
    assert isinstance(value, float), "Coalition degree failed"

    # Test ShapG
    shapg_values = shapG(G, depth=1, m=10)
    assert len(shapg_values) == 15, "ShapG computation failed"
    assert all(isinstance(v, float) for v in shapg_values.values()), "ShapG values invalid"

    # Test exact Shapley on small graph
    small_G = nx.cycle_graph(5)
    exact_values = shapley_value(small_G)
    assert len(exact_values) == 5, "Exact Shapley failed"

    # Test CIS
    cis_values = cis(small_G)
    assert len(cis_values) == 5, "CIS computation failed"

    # Test reachable nodes
    path_G = nx.path_graph(5)
    nodes = get_reachable_nodes_at_depth(path_G, 2, 1)
    assert nodes == {1, 3}, f"Reachable nodes failed: got {nodes}"

    print("✓ Old API tests PASSED")
    return True


def test_new_api():
    """Test new modular API."""
    print("\nTesting NEW API...")

    from shapG import (
        ExactExplainer,
        ShapGExplainer,
        CISExplainer,
        CSExplainer,
        CoalitionDegree,
        NodeCount,
        CustomFunction,
        GraphBuilder,
        CoalitionManager,
        FeatureImportanceVisualizer
    )

    # Test graph builder
    builder = GraphBuilder()
    G = builder.random_graph(15, 0.4, seed=42)
    assert G.number_of_nodes() == 15, "GraphBuilder failed"

    # Test from correlation
    data = pd.DataFrame(np.random.randn(100, 10))
    G_corr = builder.from_correlation(data, threshold=0.3)
    assert G_corr.number_of_nodes() == 10, "Correlation graph failed"

    # Test characteristic functions
    char_degree = CoalitionDegree()
    value = char_degree({0, 1}, G)
    assert isinstance(value, float), "CoalitionDegree failed"

    char_count = NodeCount()
    value = char_count({0, 1, 2}, None)
    assert value == 3.0, "NodeCount failed"

    # Test ShapGExplainer
    explainer = ShapGExplainer(depth=1, n_samples=10)
    shapg_values = explainer.fit_explain(G)
    assert len(shapg_values) == 15, "ShapGExplainer failed"

    # Test ExactExplainer
    small_G = nx.cycle_graph(5)
    exact_explainer = ExactExplainer()
    exact_values = exact_explainer.fit_explain(small_G)
    assert len(exact_values) == 5, "ExactExplainer failed"

    # Test coalition manager
    manager = CoalitionManager(G)
    coalition = manager.get_neighbors_coalition(0, depth=1)
    assert isinstance(coalition, set), "CoalitionManager failed"

    samples = manager.sample_coalitions(0, n_samples=10, strategy='uniform')
    assert len(samples) == 10, "Coalition sampling failed"

    # Test CSExplainer
    cs_explainer = CSExplainer(n_samples=10)
    cs_values = cs_explainer.fit_explain(small_G)
    assert len(cs_values) == 5, "CSExplainer failed"

    # Test visualizer
    viz = FeatureImportanceVisualizer()
    assert viz is not None, "Visualizer creation failed"

    print("✓ New API tests PASSED")
    return True


def test_consistency():
    """Test that old and new APIs produce consistent results."""
    print("\nTesting CONSISTENCY between APIs...")

    import shapG

    # Create smaller test graph for exact computation
    np.random.seed(42)
    G = nx.cycle_graph(8)  # Small graph for exact computation

    # Old API
    old_exact = shapG.shapley_value(G)
    old_shapg = shapG.shapG(G, depth=2, m=20)

    # New API
    exact_explainer = shapG.ExactExplainer()
    new_exact = exact_explainer.fit_explain(G)

    shapg_explainer = shapG.ShapGExplainer(depth=2, n_samples=20)
    np.random.seed(42)  # Reset seed for consistency
    new_shapg = shapg_explainer.fit_explain(G)

    # Compare exact values
    for node in G.nodes():
        diff = abs(old_exact[node] - new_exact[node])
        assert diff < 1e-10, f"Exact values differ for node {node}: {diff}"

    # Check that approximate values are in same ballpark
    for node in G.nodes():
        # Allow more tolerance for approximate methods
        ratio = old_shapg[node] / new_shapg[node] if new_shapg[node] != 0 else 1
        assert 0.1 < ratio < 10, f"ShapG values too different for node {node}"

    print("✓ Consistency tests PASSED")
    return True


def test_performance():
    """Basic performance test."""
    print("\nTesting PERFORMANCE...")

    import time
    from shapG import ShapGExplainer, GraphBuilder

    # Create larger graph
    builder = GraphBuilder()
    G = builder.random_graph(100, 0.15, seed=42)

    # Time ShapG computation
    explainer = ShapGExplainer(depth=1, n_samples=3)
    start = time.time()
    values = explainer.fit_explain(G)
    elapsed = time.time() - start

    assert len(values) == 100, "Performance test failed"
    print(f"✓ Computed Shapley values for 100 nodes in {elapsed:.2f}s")

    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("ShapG Refactoring Validation")
    print("=" * 60)

    tests = [
        ("Old API", test_old_api),
        ("New API", test_new_api),
        ("Consistency", test_consistency),
        ("Performance", test_performance)
    ]

    failed = []
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {name} test FAILED: {e}")
            failed.append(name)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if not failed:
        print("✓ ALL TESTS PASSED!")
        print("The refactoring is successful and maintains backward compatibility.")
        return 0
    else:
        print(f"✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
