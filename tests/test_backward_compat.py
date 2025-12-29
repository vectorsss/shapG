"""
Tests for backward compatibility with old API.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
import warnings

# Import old API functions
from shapG import (
    shapley_value,
    shapG,
    coalition_degree,
    cis,
    graph_generator,
    get_reachable_nodes_at_depth,
    plot,
)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that old API still works correctly."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test graph
        self.G = graph_generator(n_nodes=10, density=0.3, seed=42)

        # Small graph for exact computation
        self.small_G = nx.Graph()
        self.small_G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_graph_generator(self):
        """Test graph generation with old API."""
        G = graph_generator(20, 0.5, weight_range=(1, 5), seed=123)

        self.assertEqual(G.number_of_nodes(), 20)
        self.assertIsInstance(G, nx.Graph)

        # Check edges have weights
        for u, v, data in G.edges(data=True):
            if "weight" in data:
                self.assertTrue(1 <= data["weight"] <= 5)

    def test_coalition_degree(self):
        """Test coalition degree function."""
        # Empty coalition
        value = coalition_degree(self.small_G, set())
        self.assertEqual(value, 0)

        # Single node
        value = coalition_degree(self.small_G, {0})
        self.assertEqual(value, 0)

        # Two connected nodes
        value = coalition_degree(self.small_G, {0, 1})
        self.assertEqual(value, 1.0)

        # All nodes
        value = coalition_degree(self.small_G, {0, 1, 2})
        self.assertEqual(value, 3.0)

        # Test with list input
        value = coalition_degree(self.small_G, [0, 1])
        self.assertEqual(value, 1.0)

    def test_shapley_value_exact(self):
        """Test exact Shapley value computation."""
        values = shapley_value(self.small_G)

        # Check all nodes have values
        self.assertEqual(len(values), 3)
        for node in self.small_G.nodes():
            self.assertIn(node, values)

        # Check efficiency property
        total_value = coalition_degree(self.small_G, set(self.small_G.nodes()))
        self.assertAlmostEqual(sum(values.values()), total_value, places=5)

    def test_shapley_value_with_custom_function(self):
        """Test Shapley value with custom characteristic function."""

        def custom_f(G, S):
            return len(S) * 2

        values = shapley_value(self.small_G, f=custom_f)

        self.assertEqual(len(values), 3)
        # All nodes should have equal value for this symmetric function
        unique_values = set(values.values())
        self.assertEqual(len(unique_values), 1)

    def test_shapley_value_verbose(self):
        """Test verbose mode."""
        # Should not raise error
        values = shapley_value(self.small_G, verbose=True)
        self.assertEqual(len(values), 3)

    def test_shapG_approximate(self):
        """Test approximate Shapley computation with ShapG."""
        values = shapG(self.G, depth=1, m=10)

        # Check all nodes have values
        self.assertEqual(len(values), 10)
        for node in self.G.nodes():
            self.assertIn(node, values)

    def test_shapG_parameters(self):
        """Test ShapG with different parameters."""
        # Test different depths
        values1 = shapG(self.G, depth=1, m=10)
        values2 = shapG(self.G, depth=2, m=10)

        self.assertEqual(len(values1), len(values2))

        # Test with/without ratio approximation
        values3 = shapG(self.G, approximate_by_ratio=True)
        values4 = shapG(self.G, approximate_by_ratio=False)

        self.assertEqual(len(values3), len(values4))

        # Test with/without scaling
        values5 = shapG(self.G, scale=True)
        values6 = shapG(self.G, scale=False)

        self.assertEqual(len(values5), len(values6))

    def test_shapG_with_custom_function(self):
        """Test ShapG with custom characteristic function."""

        def custom_f(G, S):
            return float(len(S))

        values = shapG(self.G, f=custom_f, depth=1, m=5)

        self.assertEqual(len(values), 10)

    def test_cis_values(self):
        """Test CIS value computation."""
        values = cis(self.small_G)

        # Check all nodes have values
        self.assertEqual(len(values), 3)

        # Check efficiency property
        total_value = coalition_degree(self.small_G, set(self.small_G.nodes()))
        self.assertAlmostEqual(sum(values.values()), total_value, places=5)

    def test_cis_with_custom_function(self):
        """Test CIS with custom function."""

        def custom_f(G, S):
            return len(S) ** 2

        values = cis(self.small_G, f=custom_f)
        self.assertEqual(len(values), 3)

    def test_get_reachable_nodes(self):
        """Test getting reachable nodes at depth."""
        # Create a path graph for clear testing
        path_G = nx.path_graph(5)  # 0-1-2-3-4

        # Nodes at depth 1 from node 2
        nodes_d1 = get_reachable_nodes_at_depth(path_G, 2, depth=1)
        self.assertEqual(nodes_d1, {1, 3})

        # Nodes at depth 2 from node 2
        nodes_d2 = get_reachable_nodes_at_depth(path_G, 2, depth=2)
        self.assertEqual(nodes_d2, {0, 4})

        # Nodes at depth 0 (the node itself)
        nodes_d0 = get_reachable_nodes_at_depth(path_G, 2, depth=0)
        self.assertEqual(nodes_d0, {2})

    def test_plot_function(self):
        """Test plot function (without actually showing)."""
        values = {0: 0.5, 1: 0.3, 2: 0.2}

        # Test without showing
        result = plot(values, show_plot=False)
        self.assertIsNotNone(result)
        fig, ax = result
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Test with feature names
        result = plot(values, feature_names=["A", "B", "C"], top_n=2, show_plot=False)
        self.assertIsNotNone(result)

    def test_consistency_between_apis(self):
        """Test that old and new APIs produce consistent results."""
        # Import new API
        from shapG import ExactExplainer, ShapGExplainer

        # Exact computation
        old_exact = shapley_value(self.small_G)
        new_explainer = ExactExplainer()
        new_exact = new_explainer.fit_explain(self.small_G)

        # Should produce same values (within numerical precision)
        for node in self.small_G.nodes():
            self.assertAlmostEqual(old_exact[node], new_exact[node], places=5)

        # Approximate computation with same seed
        np.random.seed(42)
        old_approx = shapG(self.G, depth=1, m=10)

        np.random.seed(42)
        new_explainer = ShapGExplainer(depth=1, n_samples=10)
        new_approx = new_explainer.fit_explain(self.G)

        # Check same nodes
        self.assertEqual(set(old_approx.keys()), set(new_approx.keys()))


class TestDeprecationWarnings(unittest.TestCase):
    """Test that deprecation warnings work correctly."""

    def test_no_immediate_warnings(self):
        """Test that using old API doesn't immediately trigger warnings."""
        # The old API should work without warnings by default
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            G = graph_generator(5, 0.5)
            values = shapG(G)

            # No deprecation warnings should be raised yet
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, (DeprecationWarning, FutureWarning))
            ]
            # With deprecation warnings now implemented, expect warnings
            self.assertGreaterEqual(len(deprecation_warnings), 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for backward compatibility."""

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()

        values = shapley_value(G)
        self.assertEqual(len(values), 0)

        values = shapG(G)
        self.assertEqual(len(values), 0)

    def test_single_node_graph(self):
        """Test with single node."""
        G = nx.Graph()
        G.add_node(0)

        values = shapley_value(G)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 0)

        values = shapG(G)
        self.assertEqual(len(values), 1)

    def test_disconnected_graph(self):
        """Test with disconnected components."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components

        values = shapley_value(G)
        self.assertEqual(len(values), 4)

        values = shapG(G)
        self.assertEqual(len(values), 4)


if __name__ == "__main__":
    unittest.main()
