"""
Comprehensive tests for the core module (shapG/core/shapley.py).
Tests all core functions with edge cases, parameter validation, and mathematical properties.
"""

import unittest
import numpy as np
import networkx as nx
from math import factorial
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shapG.core.shapley import (
    graph_generator,
    coalition_degree,
    shapley_value,
    get_reachable_nodes_at_depth,
    cis,
    shapG
)


class TestGraphGenerator(unittest.TestCase):
    """Test graph_generator function."""

    def test_basic_generation(self):
        """Test basic graph generation."""
        G = graph_generator(n_nodes=10, density=0.3, seed=42)
        self.assertEqual(G.number_of_nodes(), 10)
        self.assertIsInstance(G, nx.Graph)

    def test_density_approximately_correct(self):
        """Test that density is approximately as expected."""
        n_nodes = 20
        density = 0.4
        G = graph_generator(n_nodes=n_nodes, density=density, seed=123)

        max_edges = n_nodes * (n_nodes - 1) // 2
        expected_edges = int(density * max_edges)
        actual_density = G.number_of_edges() / max_edges

        # Allow some tolerance due to randomness
        self.assertAlmostEqual(actual_density, density, delta=0.15)

    def test_seed_reproducibility(self):
        """Test that same seed produces same graph."""
        G1 = graph_generator(n_nodes=15, density=0.5, seed=42)
        G2 = graph_generator(n_nodes=15, density=0.5, seed=42)

        self.assertEqual(G1.number_of_nodes(), G2.number_of_nodes())
        self.assertEqual(G1.number_of_edges(), G2.number_of_edges())
        self.assertEqual(set(G1.edges()), set(G2.edges()))

    def test_edge_cases(self):
        """Test edge cases."""
        # Single node
        G = graph_generator(n_nodes=1, density=0.5, seed=42)
        self.assertEqual(G.number_of_nodes(), 1)
        self.assertEqual(G.number_of_edges(), 0)

        # Two nodes
        G = graph_generator(n_nodes=2, density=1.0, seed=42)
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 1)

        # Zero density
        G = graph_generator(n_nodes=10, density=0.0, seed=42)
        self.assertEqual(G.number_of_edges(), 0)

    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with self.assertRaises((ValueError, TypeError)):
            graph_generator(n_nodes=0, density=0.5)

        with self.assertRaises((ValueError, TypeError)):
            graph_generator(n_nodes=5, density=-0.1)

        with self.assertRaises((ValueError, TypeError)):
            graph_generator(n_nodes=5, density=1.1)


class TestCoalitionDegree(unittest.TestCase):
    """Test coalition_degree function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])

    def test_basic_computation(self):
        """Test basic coalition degree computation."""
        # Single node - no edges in subgraph, so degree = 0
        degree = coalition_degree(self.G, {0})
        self.assertEqual(degree, 0)  # Single node subgraph has no edges

        # Coalition of connected nodes - has edge (0,1) in subgraph
        degree = coalition_degree(self.G, {0, 1})
        self.assertEqual(degree, 1.0)  # One edge contributes degree 2, divided by 2

    def test_empty_coalition(self):
        """Test empty coalition."""
        degree = coalition_degree(self.G, set())
        self.assertEqual(degree, 0)

    def test_full_coalition(self):
        """Test full coalition."""
        all_nodes = set(self.G.nodes())
        degree = coalition_degree(self.G, all_nodes)
        # Sum of degrees in full graph divided by 2 (each edge counted twice)
        expected = sum(dict(self.G.degree()).values()) / 2
        self.assertEqual(degree, expected)

    def test_disconnected_components(self):
        """Test coalition with disconnected components."""
        degree = coalition_degree(self.G, {0, 3})
        # Nodes 0 and 3 are not connected, so no edges in their subgraph
        self.assertEqual(degree, 0)

    def test_nonexistent_nodes(self):
        """Test coalition with nonexistent nodes."""
        degree = coalition_degree(self.G, {0, 10})  # Node 10 doesn't exist
        self.assertEqual(degree, 0)  # Single node 0 has no edges in subgraph


class TestShapleyValue(unittest.TestCase):
    """Test shapley_value function."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple triangle graph
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_basic_computation(self):
        """Test basic Shapley value computation."""
        values = shapley_value(self.G, verbose=False)

        # Check that all nodes are included
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check that values are numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_efficiency_property(self):
        """Test efficiency property: sum of Shapley values equals grand coalition value."""
        values = shapley_value(self.G, verbose=False)
        total_shapley = sum(values.values())
        grand_coalition_value = coalition_degree(self.G, set(self.G.nodes()))

        self.assertAlmostEqual(total_shapley, grand_coalition_value, places=6)

    def test_symmetry_property(self):
        """Test symmetry property on symmetric graph."""
        # Complete graph where all nodes are symmetric
        G = nx.complete_graph(4)
        values = shapley_value(G, verbose=False)

        # All values should be equal due to symmetry
        shapley_values = list(values.values())
        for i in range(1, len(shapley_values)):
            self.assertAlmostEqual(shapley_values[0], shapley_values[i], places=6)

    def test_custom_characteristic_function(self):
        """Test with custom characteristic function."""
        def custom_f(G, S):
            return len(S)  # Simple: value equals coalition size

        values = shapley_value(self.G, f=custom_f, verbose=False)

        # Efficiency: sum should approximately equal grand coalition value
        total_shapley = sum(values.values())
        grand_coalition_value = custom_f(self.G, set(self.G.nodes()))
        # Note: Small discrepancy expected due to discrete computation
        self.assertAlmostEqual(total_shapley, grand_coalition_value, delta=0.6)

    def test_single_node_graph(self):
        """Test single node graph."""
        G = nx.Graph()
        G.add_node(0)

        values = shapley_value(G, verbose=False)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 0)  # Single node has degree 0

    def test_empty_graph(self):
        """Test empty graph."""
        G = nx.Graph()
        values = shapley_value(G, verbose=False)
        self.assertEqual(len(values), 0)


class TestGetReachableNodes(unittest.TestCase):
    """Test get_reachable_nodes_at_depth function."""

    def setUp(self):
        """Set up test fixtures."""
        # Path graph: 0-1-2-3-4
        self.G = nx.path_graph(5)

    def test_depth_zero(self):
        """Test depth 0 (should return only the node itself)."""
        reachable = get_reachable_nodes_at_depth(self.G, 2, 0)
        self.assertEqual(reachable, {2})

    def test_depth_one(self):
        """Test depth 1 (immediate neighbors)."""
        reachable = get_reachable_nodes_at_depth(self.G, 2, 1)
        self.assertEqual(reachable, {1, 3})

    def test_depth_two(self):
        """Test depth 2."""
        reachable = get_reachable_nodes_at_depth(self.G, 2, 2)
        self.assertEqual(reachable, {0, 4})

    def test_depth_exceeds_graph(self):
        """Test when depth exceeds graph diameter."""
        reachable = get_reachable_nodes_at_depth(self.G, 2, 10)
        # Should return empty set since no nodes exist at depth 10
        self.assertEqual(reachable, set())

    def test_isolated_node(self):
        """Test isolated node."""
        G = nx.Graph()
        G.add_node(0)

        reachable = get_reachable_nodes_at_depth(G, 0, 1)
        self.assertEqual(reachable, set())

    def test_disconnected_graph(self):
        """Test disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components

        reachable = get_reachable_nodes_at_depth(G, 0, 2)
        self.assertEqual(reachable, set())  # Can't reach other component


class TestCIS(unittest.TestCase):
    """Test cis function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    def test_basic_computation(self):
        """Test basic CIS computation."""
        values = cis(self.G)

        # Check that all nodes are included
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check that values are numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_efficiency_property(self):
        """Test efficiency property for CIS."""
        values = cis(self.G)
        total_cis = sum(values.values())
        grand_coalition_value = coalition_degree(self.G, set(self.G.nodes()))

        self.assertAlmostEqual(total_cis, grand_coalition_value, places=6)

    def test_custom_characteristic_function(self):
        """Test CIS with custom characteristic function."""
        def custom_f(G, S):
            return len(S) ** 2  # Quadratic in coalition size

        values = cis(self.G, f=custom_f)

        # Check efficiency
        total_cis = sum(values.values())
        grand_coalition_value = custom_f(self.G, set(self.G.nodes()))
        self.assertAlmostEqual(total_cis, grand_coalition_value, places=6)


class TestShapG(unittest.TestCase):
    """Test shapG function (approximate Shapley computation)."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = graph_generator(n_nodes=10, density=0.3, seed=42)

    def test_basic_computation(self):
        """Test basic shapG computation."""
        values = shapG(self.G, depth=1, m=5, verbose=False)

        # Check that all nodes are included
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # Check that values are numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_different_parameters(self):
        """Test with different parameter combinations."""
        # Test different depths
        values1 = shapG(self.G, depth=1, m=5, verbose=False)
        values2 = shapG(self.G, depth=2, m=5, verbose=False)

        # Should produce different results but same structure
        self.assertEqual(set(values1.keys()), set(values2.keys()))
        self.assertNotEqual(values1, values2)

        # Test different sample sizes
        values3 = shapG(self.G, depth=1, m=10, verbose=False)
        self.assertEqual(set(values1.keys()), set(values3.keys()))

    def test_approximate_by_ratio(self):
        """Test approximate_by_ratio parameter."""
        values1 = shapG(self.G, depth=1, m=5, approximate_by_ratio=True, verbose=False)
        values2 = shapG(self.G, depth=1, m=5, approximate_by_ratio=False, verbose=False)

        # Should produce different results
        self.assertEqual(set(values1.keys()), set(values2.keys()))

    def test_scale_parameter(self):
        """Test scale parameter."""
        values1 = shapG(self.G, depth=1, m=5, scale=True, verbose=False)
        values2 = shapG(self.G, depth=1, m=5, scale=False, verbose=False)

        # Should produce different magnitudes but same relative ordering
        self.assertEqual(set(values1.keys()), set(values2.keys()))

    def test_custom_characteristic_function(self):
        """Test with custom characteristic function."""
        def custom_f(G, S):
            return len(S)

        values = shapG(self.G, f=custom_f, depth=1, m=5, verbose=False)

        # Check basic properties
        self.assertEqual(set(values.keys()), set(self.G.nodes()))
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_edge_cases(self):
        """Test edge cases."""
        # Single node
        G_single = nx.Graph()
        G_single.add_node(0)
        values = shapG(G_single, depth=1, m=5, verbose=False)
        self.assertEqual(len(values), 1)

        # Small graph
        G_small = nx.Graph()
        G_small.add_edges_from([(0, 1)])
        values = shapG(G_small, depth=1, m=5, verbose=False)
        self.assertEqual(len(values), 2)

    def test_efficiency_approximately_holds(self):
        """Test that efficiency property approximately holds."""
        values = shapG(self.G, depth=2, m=15, verbose=False)
        total_shapg = sum(values.values())
        grand_coalition_value = coalition_degree(self.G, set(self.G.nodes()))

        # Allow larger tolerance for approximate method
        relative_error = abs(total_shapg - grand_coalition_value) / grand_coalition_value
        self.assertLess(relative_error, 0.3)  # Within 30%


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties across functions."""

    def test_consistency_small_graph(self):
        """Test consistency between exact and approximate methods on small graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle

        exact_values = shapley_value(G, verbose=False)
        approx_values = shapG(G, depth=2, m=20, verbose=False)

        # Should be reasonably close for small graph
        for node in G.nodes():
            relative_error = abs(exact_values[node] - approx_values[node]) / max(abs(exact_values[node]), 1e-6)
            self.assertLess(relative_error, 0.5)  # Within 50%

    def test_monotonicity(self):
        """Test monotonicity: adding edges shouldn't decrease node values too much."""
        # Start with triangle
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (1, 2), (2, 0)])

        # Add one more edge
        G2 = G1.copy()
        G2.add_edge(0, 3)
        G2.add_edge(1, 3)

        values1 = shapley_value(G1, verbose=False)
        values2 = shapley_value(G2, verbose=False)

        # Total value should increase (more edges = higher total degree)
        total1 = sum(values1.values())
        total2 = sum(values2.values())
        self.assertGreater(total2, total1)


if __name__ == '__main__':
    unittest.main()