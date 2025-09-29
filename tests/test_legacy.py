"""
Comprehensive tests for the legacy module (shapG/__legacy/).
Tests backward compatibility and wrapper functions.
"""

import unittest
import numpy as np
import networkx as nx
import warnings
from unittest.mock import patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shapG import (
    shapley_value,
    shapG,
    coalition_degree,
    cis,
    graph_generator,
    get_reachable_nodes_at_depth,
    plot
)


class TestLegacyShapleyValue(unittest.TestCase):
    """Test legacy shapley_value function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_basic_computation(self):
        """Test basic Shapley value computation."""
        values = shapley_value(self.G, verbose=False)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check that values are numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_custom_characteristic_function(self):
        """Test with custom characteristic function."""
        def custom_f(G, S):
            return len(S)

        values = shapley_value(self.G, f=custom_f, verbose=False)

        # Check efficiency property
        total_value = sum(values.values())
        grand_coalition_value = custom_f(self.G, set(self.G.nodes()))
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)

    def test_verbose_mode(self):
        """Test verbose mode (should not crash)."""
        values = shapley_value(self.G, verbose=True)
        self.assertIsInstance(values, dict)

    def test_compatibility_with_new_api(self):
        """Test that legacy function produces same results as new API."""
        from shapG import ExactExplainer, CoalitionDegree

        # Legacy approach
        legacy_values = shapley_value(self.G, verbose=False)

        # New API approach
        explainer = ExactExplainer(CoalitionDegree())
        new_values = explainer.fit_explain(self.G)

        # Should produce identical results
        self.assertEqual(set(legacy_values.keys()), set(new_values.keys()))

        for node in self.G.nodes():
            self.assertAlmostEqual(
                legacy_values[node],
                new_values[node],
                places=6
            )

    def test_edge_cases(self):
        """Test edge cases."""
        # Single node
        G_single = nx.Graph()
        G_single.add_node(0)

        values = shapley_value(G_single, verbose=False)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 0.0)

        # Empty graph
        G_empty = nx.Graph()
        values = shapley_value(G_empty, verbose=False)
        self.assertEqual(len(values), 0)


class TestLegacyShapG(unittest.TestCase):
    """Test legacy shapG function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 1)])

    def test_basic_computation(self):
        """Test basic shapG computation."""
        values = shapG(self.G, depth=1, m=5, verbose=False)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # Check that values are numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_parameter_variations(self):
        """Test different parameter combinations."""
        # Test depth
        values1 = shapG(self.G, depth=1, m=5, verbose=False)
        values2 = shapG(self.G, depth=2, m=5, verbose=False)

        self.assertEqual(set(values1.keys()), set(values2.keys()))

        # Test m (samples)
        values3 = shapG(self.G, depth=1, m=10, verbose=False)
        self.assertEqual(set(values1.keys()), set(values3.keys()))

        # Test approximate_by_ratio
        values4 = shapG(self.G, depth=1, m=5, approximate_by_ratio=True, verbose=False)
        values5 = shapG(self.G, depth=1, m=5, approximate_by_ratio=False, verbose=False)

        self.assertEqual(set(values4.keys()), set(values5.keys()))

        # Test scale
        values6 = shapG(self.G, depth=1, m=5, scale=True, verbose=False)
        values7 = shapG(self.G, depth=1, m=5, scale=False, verbose=False)

        self.assertEqual(set(values6.keys()), set(values7.keys()))

    def test_custom_characteristic_function(self):
        """Test with custom characteristic function."""
        def custom_f(G, S):
            return len(S) ** 2

        values = shapG(self.G, f=custom_f, depth=1, m=5, verbose=False)

        # Check basic properties
        self.assertEqual(set(values.keys()), set(self.G.nodes()))
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_compatibility_with_new_api(self):
        """Test compatibility with new API."""
        from shapG import ShapGExplainer, CoalitionDegree

        # Legacy approach
        legacy_values = shapG(self.G, depth=1, m=15, verbose=False)

        # New API approach
        explainer = ShapGExplainer(
            characteristic_function=CoalitionDegree(),
            depth=1,
            n_samples=15
        )
        new_values = explainer.fit_explain(self.G)

        # Should produce similar results (allowing for randomness)
        self.assertEqual(set(legacy_values.keys()), set(new_values.keys()))

        # Results should be reasonably close (within tolerance for approximate method)
        for node in self.G.nodes():
            if abs(legacy_values[node]) > 1e-6 and abs(new_values[node]) > 1e-6:
                relative_error = abs(legacy_values[node] - new_values[node]) / max(
                    abs(legacy_values[node]), abs(new_values[node])
                )
                self.assertLess(relative_error, 0.5)  # Within 50% (approximate method)


class TestLegacyCoalitionDegree(unittest.TestCase):
    """Test legacy coalition_degree function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_basic_computation(self):
        """Test basic coalition degree computation."""
        # Single node
        degree = coalition_degree(self.G, {0})
        self.assertEqual(degree, 0.0)  # Single node subgraph has no edges

        # Multiple nodes
        degree = coalition_degree(self.G, {0, 1})
        self.assertEqual(degree, 1.0)  # Edge (0,1) contributes degree 2, divided by 2

        # Full coalition
        degree = coalition_degree(self.G, set(self.G.nodes()))
        expected = sum(dict(self.G.degree()).values()) / 2  # Divided by 2 as per characteristic function
        self.assertEqual(degree, expected)

    def test_compatibility_with_new_api(self):
        """Test compatibility with new API."""
        from shapG.characteristic.characteristic_functions import CoalitionDegree

        char_func = CoalitionDegree()

        # Test various coalitions
        coalitions = [
            set(),
            {0},
            {1},
            {0, 1},
            {0, 1, 2},
            set(self.G.nodes())
        ]

        for coalition in coalitions:
            legacy_value = coalition_degree(self.G, coalition)
            new_value = char_func(coalition, self.G)

            self.assertAlmostEqual(legacy_value, new_value, places=6)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty coalition
        degree = coalition_degree(self.G, set())
        self.assertEqual(degree, 0.0)

        # Nonexistent nodes
        degree = coalition_degree(self.G, {10})
        self.assertEqual(degree, 0.0)

        # Mixed existing and nonexistent nodes
        degree = coalition_degree(self.G, {0, 10})
        self.assertEqual(degree, 0.0)  # Only isolated node 0


class TestLegacyCIS(unittest.TestCase):
    """Test legacy cis function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_basic_computation(self):
        """Test basic CIS computation."""
        values = cis(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check efficiency property
        total_value = sum(values.values())
        grand_coalition_value = coalition_degree(self.G, set(self.G.nodes()))
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)

    def test_custom_characteristic_function(self):
        """Test CIS with custom characteristic function."""
        def custom_f(G, S):
            return len(S) * 2

        values = cis(self.G, f=custom_f)

        # Check efficiency
        total_value = sum(values.values())
        grand_coalition_value = custom_f(self.G, set(self.G.nodes()))
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)

    def test_compatibility_with_new_api(self):
        """Test compatibility with new API."""
        from shapG import CISExplainer, CoalitionDegree

        # Legacy approach
        legacy_values = cis(self.G)

        # New API approach
        explainer = CISExplainer(CoalitionDegree())
        new_values = explainer.fit_explain(self.G)

        # Should produce identical results
        self.assertEqual(set(legacy_values.keys()), set(new_values.keys()))

        for node in self.G.nodes():
            self.assertAlmostEqual(
                legacy_values[node],
                new_values[node],
                places=6
            )


class TestLegacyGraphGenerator(unittest.TestCase):
    """Test legacy graph_generator function."""

    def test_basic_generation(self):
        """Test basic graph generation."""
        G = graph_generator(n_nodes=10, density=0.3, seed=42)

        self.assertEqual(G.number_of_nodes(), 10)
        self.assertIsInstance(G, nx.Graph)

    def test_reproducibility(self):
        """Test reproducibility with seed."""
        G1 = graph_generator(n_nodes=8, density=0.4, seed=123)
        G2 = graph_generator(n_nodes=8, density=0.4, seed=123)

        self.assertEqual(G1.number_of_nodes(), G2.number_of_nodes())
        self.assertEqual(G1.number_of_edges(), G2.number_of_edges())
        self.assertEqual(set(G1.edges()), set(G2.edges()))

    def test_compatibility_with_new_api(self):
        """Test compatibility with new API."""
        from shapG.utils.graph_construction import GraphBuilder

        # Legacy approach
        legacy_G = graph_generator(n_nodes=6, density=0.5, seed=42)

        # New API approach
        builder = GraphBuilder()
        new_G = builder.random_graph(n_nodes=6, density=0.5, seed=42)

        # Should produce similar structure (same number of nodes, similar density)
        self.assertEqual(legacy_G.number_of_nodes(), new_G.number_of_nodes())
        # Allow some variation in number of edges due to different random generation algorithms
        edge_diff = abs(legacy_G.number_of_edges() - new_G.number_of_edges())
        max_edges = 6 * 5 // 2  # n_nodes * (n_nodes - 1) // 2
        self.assertLessEqual(edge_diff / max_edges, 0.3)  # Within 30% variation

    def test_edge_cases(self):
        """Test edge cases."""
        # Single node
        G = graph_generator(n_nodes=1, density=0.5, seed=42)
        self.assertEqual(G.number_of_nodes(), 1)
        self.assertEqual(G.number_of_edges(), 0)

        # Zero density
        G = graph_generator(n_nodes=5, density=0.0, seed=42)
        self.assertEqual(G.number_of_edges(), 0)


class TestLegacyGetReachableNodes(unittest.TestCase):
    """Test legacy get_reachable_nodes_at_depth function."""

    def setUp(self):
        """Set up test fixtures."""
        # Path graph: 0-1-2-3-4
        self.G = nx.path_graph(5)

    def test_basic_functionality(self):
        """Test basic reachable nodes computation."""
        # Depth 1 from node 2
        reachable = get_reachable_nodes_at_depth(self.G, 2, 1)
        self.assertEqual(reachable, {1, 3})

        # Depth 2 from node 2
        reachable = get_reachable_nodes_at_depth(self.G, 2, 2)
        self.assertEqual(reachable, {0, 4})

    def test_depth_zero(self):
        """Test depth 0."""
        reachable = get_reachable_nodes_at_depth(self.G, 2, 0)
        self.assertEqual(reachable, {2})

    def test_compatibility_with_new_api(self):
        """Test compatibility with new API."""
        from shapG.utils.graph_helpers import get_reachable_nodes_at_depth as utils_func

        test_cases = [
            (0, 1), (0, 2), (2, 1), (2, 2), (4, 1)
        ]

        for node, depth in test_cases:
            legacy_result = get_reachable_nodes_at_depth(self.G, node, depth)
            utils_result = utils_func(self.G, node, depth)

            self.assertEqual(legacy_result, utils_result)

    def test_edge_cases(self):
        """Test edge cases."""
        # Isolated node
        G = nx.Graph()
        G.add_node(0)

        reachable = get_reachable_nodes_at_depth(G, 0, 1)
        self.assertEqual(reachable, set())

        # Large depth - should return empty set since no nodes exist at depth 10
        reachable = get_reachable_nodes_at_depth(self.G, 2, 10)
        self.assertEqual(reachable, set())


class TestLegacyPlot(unittest.TestCase):
    """Test legacy plot function."""

    def setUp(self):
        """Set up test fixtures."""
        self.shapley_values = {
            'Feature_A': 0.5,
            'Feature_B': -0.3,
            'Feature_C': 0.8
        }

    def test_basic_plotting(self):
        """Test basic plotting functionality."""
        # Import matplotlib and set non-interactive backend
        import matplotlib
        matplotlib.use('Agg')

        result = plot(self.shapley_values, show_plot=False)

        # Should return figure and axes
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        import matplotlib.pyplot as plt
        fig, ax = result
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        plt.close(fig)

    def test_compatibility_with_new_api(self):
        """Test compatibility with new visualization API."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from shapG.visualization.plot import plot as new_plot

        # Legacy approach
        legacy_result = plot(self.shapley_values, show_plot=False)

        # New API approach
        new_result = new_plot(self.shapley_values, show_plot=False)

        # Both should return figure and axes
        self.assertEqual(len(legacy_result), 2)
        self.assertEqual(len(new_result), 2)

        # Clean up
        plt.close(legacy_result[0])
        plt.close(new_result[0])

    def test_with_feature_names(self):
        """Test with feature names parameter."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        values = {0: 0.5, 1: -0.3, 2: 0.8}
        feature_names = ['Custom_A', 'Custom_B', 'Custom_C']

        result = plot(values, feature_names=feature_names, show_plot=False)
        self.assertIsInstance(result, tuple)

        plt.close(result[0])


class TestLegacyBackwardCompatibility(unittest.TestCase):
    """Test overall backward compatibility."""

    def test_import_compatibility(self):
        """Test that all legacy functions can be imported."""
        # Test direct imports
        from shapG import (
            shapley_value,
            shapG,
            coalition_degree,
            cis,
            graph_generator,
            get_reachable_nodes_at_depth,
            plot
        )

        # All should be callable
        self.assertTrue(callable(shapley_value))
        self.assertTrue(callable(shapG))
        self.assertTrue(callable(coalition_degree))
        self.assertTrue(callable(cis))
        self.assertTrue(callable(graph_generator))
        self.assertTrue(callable(get_reachable_nodes_at_depth))
        self.assertTrue(callable(plot))

    def test_complete_legacy_workflow(self):
        """Test complete workflow using only legacy functions."""
        # Generate graph
        G = graph_generator(n_nodes=6, density=0.4, seed=42)

        # Compute exact Shapley values
        exact_values = shapley_value(G, verbose=False)

        # Compute approximate Shapley values
        approx_values = shapG(G, depth=1, m=10, verbose=False)

        # Compute CIS values
        cis_values = cis(G)

        # All should produce valid results
        self.assertIsInstance(exact_values, dict)
        self.assertIsInstance(approx_values, dict)
        self.assertIsInstance(cis_values, dict)

        # All should have same keys
        self.assertEqual(set(exact_values.keys()), set(G.nodes()))
        self.assertEqual(set(approx_values.keys()), set(G.nodes()))
        self.assertEqual(set(cis_values.keys()), set(G.nodes()))

        # Test utility functions
        coalition = {0, 1}
        degree_value = coalition_degree(G, coalition)
        self.assertIsInstance(degree_value, (int, float))

        reachable = get_reachable_nodes_at_depth(G, 0, 1)
        self.assertIsInstance(reachable, set)

    def test_consistency_with_new_api_complete(self):
        """Test that legacy API produces same results as new API for full workflow."""
        from shapG import ExactExplainer, ShapGExplainer, CISExplainer, CoalitionDegree

        # Create test graph
        G = graph_generator(n_nodes=5, density=0.6, seed=42)

        # Legacy computations
        legacy_exact = shapley_value(G, verbose=False)
        legacy_cis = cis(G)
        legacy_approx = shapG(G, depth=2, m=20, verbose=False)

        # New API computations
        char_func = CoalitionDegree()

        exact_explainer = ExactExplainer(char_func)
        new_exact = exact_explainer.fit_explain(G)

        cis_explainer = CISExplainer(char_func)
        new_cis = cis_explainer.fit_explain(G)

        shapg_explainer = ShapGExplainer(char_func, depth=2, n_samples=20)
        new_approx = shapg_explainer.fit_explain(G)

        # Exact methods should be identical
        for node in G.nodes():
            self.assertAlmostEqual(legacy_exact[node], new_exact[node], places=6)
            self.assertAlmostEqual(legacy_cis[node], new_cis[node], places=6)

        # Approximate methods should be reasonably close
        for node in G.nodes():
            if abs(legacy_approx[node]) > 1e-6 and abs(new_approx[node]) > 1e-6:
                relative_error = abs(legacy_approx[node] - new_approx[node]) / max(
                    abs(legacy_approx[node]), abs(new_approx[node])
                )
                self.assertLess(relative_error, 0.3)  # Within 30%

    def test_deprecation_warnings(self):
        """Test that deprecation warnings ARE triggered for legacy functions."""
        # Legacy functions should trigger deprecation warnings since version 0.14.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            G = graph_generator(n_nodes=3, density=0.5, seed=42)
            values = shapley_value(G, verbose=False)

            # Deprecation warnings should be triggered for legacy API usage
            deprecation_warnings = [warning for warning in w
                                    if issubclass(warning.category, DeprecationWarning)]
            # We should have 2 warnings: one for graph_generator and one for shapley_value
            self.assertEqual(len(deprecation_warnings), 2)

            # Check the warning messages contain the correct information
            warning_messages = [str(w.message) for w in deprecation_warnings]
            self.assertTrue(any("graph_generator" in msg for msg in warning_messages))
            self.assertTrue(any("shapley_value" in msg for msg in warning_messages))
            self.assertTrue(any("0.15.0" in msg for msg in warning_messages))


if __name__ == '__main__':
    unittest.main()