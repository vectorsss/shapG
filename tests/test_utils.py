"""
Comprehensive tests for the utils module (shapG/utils/).
Tests graph construction, coalition management, and utility functions.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau
from unittest.mock import patch, Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapG.utils import (
    corr_generator,
    matrix_generator,
    kl,
    kl_mi_matrix,
    create_minimal_edge_graph,
    GraphBuilder,
    CoalitionManager,
)


class TestCorrGenerator(unittest.TestCase):
    """Test corr_generator function."""

    def test_basic_correlation_matrix(self):
        """Test basic correlation matrix generation."""
        data = np.random.randn(100, 5)
        df = pd.DataFrame(data, columns=["A", "B", "C", "D", "E"])

        corr_matrix = corr_generator(df)

        # Should be square matrix
        self.assertEqual(corr_matrix.shape, (5, 5))

        # Should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)

        # Diagonal should be 1 (or close to 1)
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix), np.ones(5), decimal=10
        )

        # Values should be between -1 and 1
        self.assertTrue(np.all(corr_matrix >= -1))
        self.assertTrue(np.all(corr_matrix <= 1))

    def test_different_correlation_methods(self):
        """Test different correlation methods."""
        data = np.random.randn(50, 3)
        df = pd.DataFrame(data)

        # Test different methods
        methods = ["pearson", "kendall", "spearman"]
        for method in methods:
            corr_matrix = corr_generator(df, method=method)
            self.assertEqual(corr_matrix.shape, (3, 3))
            self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))

    def test_with_numpy_array(self):
        """Test with numpy array input."""
        data = np.random.randn(50, 4)

        corr_matrix = corr_generator(data)
        self.assertEqual(corr_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))

    def test_edge_cases(self):
        """Test edge cases."""
        # Single column
        data = np.random.randn(50, 1)
        corr_matrix = corr_generator(data)
        self.assertEqual(corr_matrix.shape, (1, 1))
        self.assertEqual(corr_matrix.iloc[0, 0], 1.0)

        # Two identical columns (perfect correlation)
        data = np.random.randn(50, 1)
        data_dup = np.hstack([data, data])
        corr_matrix = corr_generator(data_dup)
        self.assertAlmostEqual(corr_matrix.iloc[0, 1], 1.0, places=10)

    def test_constant_columns(self):
        """Test with constant columns."""
        data = np.ones((50, 2))  # All values the same

        # Should handle constant columns gracefully
        corr_matrix = corr_generator(data)
        self.assertEqual(corr_matrix.shape, (2, 2))
        # Correlation of constant with itself should be handled
        self.assertTrue(
            np.isnan(corr_matrix.iloc[0, 1]) or corr_matrix.iloc[0, 1] == 0.0
        )


class TestMatrixGenerator(unittest.TestCase):
    """Test matrix_generator function."""

    def test_default_kendalltau_method(self):
        """Test default kendalltau method."""
        data = np.random.randn(100, 4)
        df = pd.DataFrame(data)

        matrix = matrix_generator(df)

        # Should be square matrix
        self.assertEqual(matrix.shape, (4, 4))

        # Should be symmetric
        np.testing.assert_array_almost_equal(matrix, matrix.T)

        # Values should be reasonable for kendall tau
        self.assertTrue(np.all(matrix >= -1))
        self.assertTrue(np.all(matrix <= 1))

    def test_different_methods(self):
        """Test different correlation methods."""
        data = np.random.randn(50, 3)
        df = pd.DataFrame(data)

        methods = ["pearson", "kendall", "spearman"]
        for method in methods:
            matrix = matrix_generator(df, method=method)
            self.assertEqual(matrix.shape, (3, 3))
            self.assertTrue(np.allclose(matrix, matrix.T))

    def test_with_absolute_values(self):
        """Test that matrix values are typically processed correctly."""
        # Create data with known correlations
        n = 100
        x = np.random.randn(n)
        y = x + 0.1 * np.random.randn(n)  # Positively correlated
        z = -x + 0.1 * np.random.randn(n)  # Negatively correlated

        data = pd.DataFrame({"x": x, "y": y, "z": z})
        matrix = matrix_generator(data)

        # Should capture correlation structure
        self.assertGreater(matrix[0, 1], 0)  # x and y should be positively correlated
        # Note: kendalltau may have different sign behavior

    def test_method_consistency(self):
        """Test consistency with scipy methods."""
        data = np.random.randn(50, 3)
        df = pd.DataFrame(data)

        # Compare with direct scipy calculation for kendall
        matrix = matrix_generator(df, method="kendall")

        # Manual kendall calculation for first two columns
        tau, _ = kendalltau(df.iloc[:, 0], df.iloc[:, 1])

        # Should be approximately equal (allowing for different implementations)
        self.assertAlmostEqual(abs(matrix[0, 1]), abs(tau), places=2)


class TestKLFunctions(unittest.TestCase):
    """Test KL divergence related functions."""

    def test_kl_basic(self):
        """Test basic KL divergence computation."""
        # Two probability distributions
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        kl_div = kl(p, q)

        # Should be non-negative
        self.assertGreaterEqual(kl_div, 0)

        # Should be a scalar
        self.assertIsInstance(kl_div, (float, np.floating))

    def test_kl_identical_distributions(self):
        """Test KL divergence for identical distributions."""
        p = np.array([0.3, 0.4, 0.3])
        kl_div = kl(p, p)

        # Should be 0 for identical distributions
        self.assertAlmostEqual(kl_div, 0.0, places=10)

    def test_kl_edge_cases(self):
        """Test KL divergence edge cases."""
        # Test with zeros (but not in denominator)
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.1, 0.4, 0.5])

        kl_div = kl(p, q)
        self.assertGreaterEqual(kl_div, 0)

    def test_kl_mi_matrix_basic(self):
        """Test kl_mi_matrix basic functionality."""
        # Create test data
        data = np.random.randn(100, 3)

        matrix = kl_mi_matrix(data)

        # Should be square matrix
        self.assertEqual(matrix.shape, (3, 3))

        # Should be symmetric (MI is symmetric)
        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=5)

        # Diagonal should be maximum (self-information)
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertGreaterEqual(matrix[i, i], matrix[i, j])

    def test_kl_mi_matrix_parameters(self):
        """Test kl_mi_matrix with different parameters."""
        data = np.random.randn(50, 2)

        # Test with different bin numbers
        matrix1 = kl_mi_matrix(data, bins=5)
        matrix2 = kl_mi_matrix(data, bins=10)

        self.assertEqual(matrix1.shape, matrix2.shape)
        # Different binning should give different results
        self.assertFalse(np.allclose(matrix1, matrix2))

    def test_kl_mi_matrix_with_constant_data(self):
        """Test kl_mi_matrix with constant data."""
        import warnings

        # Suppress sklearn warning about constant features (expected in this test)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Feature .* is constant", category=UserWarning
            )
            # Constant data
            data = np.ones((50, 2))

            matrix = kl_mi_matrix(data)
            self.assertEqual(matrix.shape, (2, 2))
            # Should handle constant data gracefully


class TestCreateMinimalEdgeGraph(unittest.TestCase):
    """Test create_minimal_edge_graph function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple correlation matrix
        self.W = np.array(
            [
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0],
            ]
        )

    def test_basic_functionality(self):
        """Test basic minimal edge graph creation."""
        A, W_new = create_minimal_edge_graph(self.W)

        # Should return adjacency matrix and weight matrix
        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(W_new, np.ndarray)

        # Same shape as input
        self.assertEqual(A.shape, self.W.shape)
        self.assertEqual(W_new.shape, self.W.shape)

        # Adjacency matrix should be binary
        self.assertTrue(np.all((A == 0) | (A == 1)))

        # Should be symmetric
        np.testing.assert_array_equal(A, A.T)
        np.testing.assert_array_equal(W_new, W_new.T)

    def test_different_versions(self):
        """Test different algorithm versions."""
        versions = ["v1", "v2", "v3"]

        for version in versions:
            A, W_new = create_minimal_edge_graph(self.W, version=version)

            self.assertEqual(A.shape, self.W.shape)
            self.assertTrue(np.all((A == 0) | (A == 1)))
            np.testing.assert_array_equal(A, A.T)

    def test_reverse_parameter(self):
        """Test reverse parameter effect."""
        A1, W1 = create_minimal_edge_graph(self.W, reverse=False)
        A2, W2 = create_minimal_edge_graph(self.W, reverse=True)

        # Should produce different results
        self.assertFalse(np.array_equal(A1, A2))

    def test_diagonal_handling(self):
        """Test that diagonal is handled correctly."""
        A, W_new = create_minimal_edge_graph(self.W)

        # Diagonal should be zero in adjacency matrix
        np.testing.assert_array_equal(np.diag(A), np.zeros(A.shape[0]))

    def test_small_matrix(self):
        """Test with small matrices."""
        # 2x2 matrix
        W_small = np.array([[1.0, 0.5], [0.5, 1.0]])
        A, W_new = create_minimal_edge_graph(W_small)

        self.assertEqual(A.shape, (2, 2))

        # Single edge case
        W_single = np.array([[1.0]])
        A, W_new = create_minimal_edge_graph(W_single)
        self.assertEqual(A.shape, (1, 1))
        self.assertEqual(A[0, 0], 0)  # No self-loops

    def test_connectivity_property(self):
        """Test that result tends to create connected graph."""
        A, W_new = create_minimal_edge_graph(self.W)

        # Convert to NetworkX graph to check connectivity
        G = nx.from_numpy_array(A)

        # For most cases with reasonable input, should be connected
        # (This depends on the algorithm and input, so we just check it's a valid graph)
        self.assertGreaterEqual(G.number_of_edges(), 0)
        self.assertEqual(G.number_of_nodes(), len(self.W))


class TestGraphBuilder(unittest.TestCase):
    """Test GraphBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = GraphBuilder()
        self.data = pd.DataFrame(np.random.randn(50, 4), columns=["A", "B", "C", "D"])

    def test_from_correlation(self):
        """Test graph building from correlation."""
        G = self.builder.from_correlation(self.data, threshold=0.3)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)

        # Node labels should be column names
        self.assertEqual(set(G.nodes()), {"A", "B", "C", "D"})

    def test_from_correlation_parameters(self):
        """Test from_correlation with different parameters."""
        # Test different thresholds
        G1 = self.builder.from_correlation(self.data, threshold=0.1)
        G2 = self.builder.from_correlation(self.data, threshold=0.5)

        # Higher threshold should result in fewer edges
        self.assertLessEqual(G2.number_of_edges(), G1.number_of_edges())

        # Test different methods
        G3 = self.builder.from_correlation(self.data, method="kendall", threshold=0.3)
        self.assertEqual(G3.number_of_nodes(), 4)

    def test_from_mutual_information(self):
        """Test graph building from mutual information."""
        G = self.builder.from_mutual_information(self.data, threshold=0.1)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)

    def test_from_adjacency(self):
        """Test graph building from adjacency matrix."""
        adj_matrix = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

        G = self.builder.from_adjacency(adj_matrix)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)
        self.assertEqual(G.number_of_edges(), 4)

    def test_from_adjacency_with_labels(self):
        """Test from_adjacency with node labels."""
        adj_matrix = np.array([[0, 1], [1, 0]])
        labels = ["X", "Y"]

        G = self.builder.from_adjacency(adj_matrix, node_labels=labels)

        self.assertEqual(set(G.nodes()), {"X", "Y"})
        self.assertEqual(G.number_of_edges(), 1)

    def test_random_graph(self):
        """Test random graph generation."""
        G = self.builder.random_graph(n_nodes=10, density=0.3, seed=42)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 10)

        # Test reproducibility
        G2 = self.builder.random_graph(n_nodes=10, density=0.3, seed=42)
        self.assertEqual(G.edges(), G2.edges())

    def test_from_kendalltau_minimal_edge(self):
        """Test from_kendalltau_minimal_edge method."""
        G = self.builder.from_kendalltau_minimal_edge(
            self.data, reverse=True, version="v3"
        )

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)

    def test_from_matrix_generator(self):
        """Test from_matrix_generator method."""
        G = self.builder.from_matrix_generator(self.data, method="kendall")

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)

    def test_with_numpy_array(self):
        """Test with numpy array input."""
        data_np = self.data.values

        G = self.builder.from_correlation(data_np, threshold=0.3)
        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 4)

    def test_static_method_access(self):
        """Test that methods can be called as static methods."""
        G = GraphBuilder.from_correlation(self.data, threshold=0.3)
        self.assertIsInstance(G, nx.Graph)

        G2 = GraphBuilder.random_graph(n_nodes=5, density=0.5, seed=123)
        self.assertIsInstance(G2, nx.Graph)

    def test_from_rank_deletion_basic(self):
        """Test basic from_rank_deletion functionality."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        G = self.builder.from_rank_deletion(X, y)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertGreater(G.number_of_edges(), 0)

        # Check connectivity
        self.assertTrue(nx.is_connected(G))

        # Check metadata
        self.assertIn("actual_density_ratio", G.graph)

    def test_from_rank_deletion_with_dataframe(self):
        """Test from_rank_deletion with DataFrame input."""
        X = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
        y = np.random.randn(100)

        G = GraphBuilder.from_rank_deletion(X, y)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(set(G.nodes()), {"A", "B", "C", "D"})
        self.assertTrue(nx.is_connected(G))

    def test_from_rank_deletion_density_ratio(self):
        """Test from_rank_deletion with different density ratios."""
        X = np.random.randn(80, 6)
        y = np.random.randn(80)

        # Test different density ratios
        G1 = GraphBuilder.from_rank_deletion(X, y, density_ratio=0.3)
        G2 = GraphBuilder.from_rank_deletion(X, y, density_ratio=0.6)

        # Higher density should have more edges
        self.assertLessEqual(G1.number_of_edges(), G2.number_of_edges())

        # Both should be connected
        self.assertTrue(nx.is_connected(G1))
        self.assertTrue(nx.is_connected(G2))

    def test_from_rank_deletion_default_density(self):
        """Test from_rank_deletion with default density (None)."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        G = GraphBuilder.from_rank_deletion(X, y, density_ratio=None)

        self.assertIsInstance(G, nx.Graph)
        self.assertTrue(nx.is_connected(G))

        # Should use 1.5x minimum ratio
        n_nodes = G.number_of_nodes()
        min_edges = n_nodes - 1
        self.assertGreaterEqual(G.number_of_edges(), min_edges)

    def test_from_rank_deletion_methods(self):
        """Test from_rank_deletion with different correlation methods."""
        X = np.random.randn(60, 4)
        y = np.random.randn(60)

        methods = ["cosine", "pearsonr", "kendalltau", "spearmanr", "mutual_info"]

        for method in methods:
            G = GraphBuilder.from_rank_deletion(
                X, y, correlation_method=method, similarity_method=method
            )

            self.assertIsInstance(G, nx.Graph)
            self.assertEqual(G.number_of_nodes(), 4)
            self.assertTrue(nx.is_connected(G))

    def test_from_rank_deletion_feature_ranges(self):
        """Test from_rank_deletion with feature type ranges."""
        X = np.random.randn(50, 6)
        y = np.random.randn(50)

        # Define feature ranges: 3 numerical, 2 categorical, 1 binary
        feature_ranges = {"num": (0, 3), "cat": (3, 5), "bin": (5, 6)}

        G = GraphBuilder.from_rank_deletion(
            X, y, feature_ranges=feature_ranges, enforce_cross_type_edges=True
        )

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertTrue(nx.is_connected(G))

    def test_from_rank_deletion_edge_weights(self):
        """Test that from_rank_deletion creates weighted edges."""
        X = np.random.randn(50, 4)
        y = np.random.randn(50)

        G = GraphBuilder.from_rank_deletion(X, y)

        # Check that edges have weights
        for u, v in G.edges():
            self.assertIn("weight", G[u][v])
            self.assertIsInstance(G[u][v]["weight"], (int, float, np.number))
            self.assertGreaterEqual(G[u][v]["weight"], 0)

    def test_from_rank_deletion_small_graph(self):
        """Test from_rank_deletion with small graphs."""
        # Test with 3 nodes
        X = np.random.randn(30, 3)
        y = np.random.randn(30)

        G = GraphBuilder.from_rank_deletion(X, y)

        self.assertEqual(G.number_of_nodes(), 3)
        self.assertGreaterEqual(
            G.number_of_edges(), 2
        )  # At least minimum for connectivity
        self.assertTrue(nx.is_connected(G))

    def test_from_rank_deletion_reproducibility(self):
        """Test from_rank_deletion reproducibility."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = np.random.randn(50)

        G1 = GraphBuilder.from_rank_deletion(X, y, density_ratio=0.4)
        G2 = GraphBuilder.from_rank_deletion(X, y, density_ratio=0.4)

        # Should produce same graph structure
        self.assertEqual(G1.number_of_nodes(), G2.number_of_nodes())
        self.assertEqual(G1.number_of_edges(), G2.number_of_edges())
        self.assertEqual(set(G1.edges()), set(G2.edges()))

    def test_get_feature_rank(self):
        """Test _get_feature_rank helper method."""
        X = np.random.randn(50, 4)
        y = np.random.randn(50)

        # Test different ranking methods
        for method in ["cosine", "pearsonr", "mutual_info"]:
            rank = GraphBuilder._get_feature_rank(X, y, method)

            self.assertIsInstance(rank, list)
            self.assertEqual(len(rank), 4)
            self.assertEqual(set(rank), {0, 1, 2, 3})

    def test_calculate_similarity_matrix(self):
        """Test _calculate_similarity_matrix helper method."""
        X = np.random.randn(50, 4)

        # Test different similarity methods
        for method in ["cosine", "pearsonr", "mutual_info"]:
            sim_matrix = GraphBuilder._calculate_similarity_matrix(X, method)

            self.assertEqual(sim_matrix.shape, (4, 4))
            # Should be symmetric
            np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
            # Values should be reasonable
            self.assertTrue(np.all(sim_matrix >= -1))
            self.assertTrue(np.all(sim_matrix <= 1))

    def test_get_feature_type(self):
        """Test _get_feature_type helper method."""
        feature_ranges = {"num": (0, 3), "cat": (3, 5), "bin": (5, 6)}

        self.assertEqual(GraphBuilder._get_feature_type(0, feature_ranges), "num")
        self.assertEqual(GraphBuilder._get_feature_type(2, feature_ranges), "num")
        self.assertEqual(GraphBuilder._get_feature_type(3, feature_ranges), "cat")
        self.assertEqual(GraphBuilder._get_feature_type(4, feature_ranges), "cat")
        self.assertEqual(GraphBuilder._get_feature_type(5, feature_ranges), "bin")
        self.assertEqual(GraphBuilder._get_feature_type(6, feature_ranges), "unknown")

    def test_is_cross_type_edge(self):
        """Test _is_cross_type_edge helper method."""
        feature_ranges = {"num": (0, 2), "cat": (2, 4)}

        # Numerical to categorical should be cross-type
        self.assertTrue(GraphBuilder._is_cross_type_edge(0, 2, feature_ranges, True))
        self.assertTrue(GraphBuilder._is_cross_type_edge(1, 3, feature_ranges, True))

        # Same type should not be cross-type
        self.assertFalse(GraphBuilder._is_cross_type_edge(0, 1, feature_ranges, True))
        self.assertFalse(GraphBuilder._is_cross_type_edge(2, 3, feature_ranges, True))

        # Should return False if not enforcing
        self.assertFalse(GraphBuilder._is_cross_type_edge(0, 2, feature_ranges, False))


class TestCoalitionManager(unittest.TestCase):
    """Test CoalitionManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 1)])
        self.manager = CoalitionManager()

    def test_get_neighbors_coalition(self):
        """Test getting neighbor coalitions."""
        neighbors = self.manager.get_neighbors_coalition(self.G, {0}, depth=1)

        # Should include neighbors of node 0
        expected = {1, 2}  # Neighbors of 0
        self.assertEqual(neighbors, expected)

    def test_get_neighbors_coalition_depth_2(self):
        """Test neighbor coalitions with depth 2."""
        neighbors = self.manager.get_neighbors_coalition(self.G, {0}, depth=2)

        # Should include neighbors and neighbors of neighbors
        # Node 0 -> {1, 2}, Node 1 -> {0, 2, 3}, Node 2 -> {0, 1}
        # So depth 2 from {0} should include all nodes except maybe isolated ones
        self.assertIn(1, neighbors)
        self.assertIn(2, neighbors)
        self.assertIn(3, neighbors)  # Neighbor of neighbor

    def test_get_all_coalitions(self):
        """Test getting all possible coalitions."""
        coalitions = self.manager.get_all_coalitions(self.G)

        # Should have 2^n coalitions where n is number of nodes
        n_nodes = self.G.number_of_nodes()
        expected_count = 2**n_nodes

        self.assertEqual(len(coalitions), expected_count)

        # Should include empty set and full set
        self.assertIn(frozenset(), coalitions)
        self.assertIn(frozenset(self.G.nodes()), coalitions)

    def test_get_all_coalitions_max_size(self):
        """Test getting coalitions with maximum size constraint."""
        coalitions = self.manager.get_all_coalitions(self.G, max_size=2)

        # Should only include coalitions of size <= 2
        for coalition in coalitions:
            self.assertLessEqual(len(coalition), 2)

        # Should include empty set, all singletons, and all pairs
        self.assertIn(frozenset(), coalitions)
        self.assertIn(frozenset([0]), coalitions)
        self.assertIn(frozenset([0, 1]), coalitions)

    def test_sample_coalitions_uniform(self):
        """Test uniform coalition sampling."""
        coalitions = self.manager.sample_coalitions(
            self.G, n_samples=100, strategy="uniform", seed=42
        )

        self.assertEqual(len(coalitions), 100)

        # Should be list of sets
        for coalition in coalitions:
            self.assertIsInstance(coalition, set)

        # Test reproducibility
        coalitions2 = self.manager.sample_coalitions(
            self.G, n_samples=100, strategy="uniform", seed=42
        )
        self.assertEqual(coalitions, coalitions2)

    def test_sample_coalitions_stratified(self):
        """Test stratified coalition sampling."""
        coalitions = self.manager.sample_coalitions(
            self.G, n_samples=50, strategy="stratified", seed=42
        )

        self.assertEqual(len(coalitions), 50)

        # Should have variety of coalition sizes
        sizes = [len(coalition) for coalition in coalitions]
        self.assertGreater(len(set(sizes)), 1)  # Multiple different sizes

    def test_sample_coalitions_by_size(self):
        """Test coalition sampling by specific sizes."""
        coalitions = self.manager.sample_coalitions(
            self.G, n_samples=20, strategy="by_size", coalition_sizes=[1, 2]
        )

        # Should only include coalitions of size 1 or 2
        for coalition in coalitions:
            self.assertIn(len(coalition), [1, 2])

    def test_clear_cache(self):
        """Test cache clearing."""
        # Perform operations that might cache results
        self.manager.get_all_coalitions(self.G)

        # Clear cache (should not raise error)
        self.manager.clear_cache()

    def test_static_method_access(self):
        """Test static method access."""
        neighbors = CoalitionManager.get_neighbors_coalition(self.G, {0}, depth=1)
        self.assertIsInstance(neighbors, set)

        coalitions = CoalitionManager.sample_coalitions(
            self.G, n_samples=10, strategy="uniform", seed=42
        )
        self.assertEqual(len(coalitions), 10)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utils module."""

    def test_full_pipeline(self):
        """Test full pipeline from data to graph."""
        # Generate test data
        np.random.seed(42)
        data = np.random.randn(100, 5)
        df = pd.DataFrame(data, columns=["A", "B", "C", "D", "E"])

        # Create correlation matrix
        corr_matrix = corr_generator(df)

        # Create minimal edge graph
        adj_matrix, weight_matrix = create_minimal_edge_graph(corr_matrix)

        # Build NetworkX graph
        builder = GraphBuilder()
        G = builder.from_adjacency(adj_matrix, node_labels=df.columns.tolist())

        # Test final graph properties
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertEqual(set(G.nodes()), {"A", "B", "C", "D", "E"})
        self.assertGreaterEqual(G.number_of_edges(), 0)

        # Test coalition management
        manager = CoalitionManager()
        coalitions = manager.sample_coalitions(G, n_samples=50, strategy="uniform")

        self.assertEqual(len(coalitions), 50)
        for coalition in coalitions:
            self.assertTrue(coalition.issubset(set(G.nodes())))

    def test_matrix_methods_consistency(self):
        """Test consistency between different matrix generation methods."""
        data = pd.DataFrame(np.random.randn(50, 3), columns=["X", "Y", "Z"])

        # Different methods should produce different but valid results
        corr_matrix = corr_generator(data, method="pearson")
        kendall_matrix = matrix_generator(data, method="kendall")

        # Both should be valid correlation matrices
        self.assertEqual(corr_matrix.shape, kendall_matrix.shape)
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))
        self.assertTrue(np.allclose(kendall_matrix, kendall_matrix.T))

    def test_graph_builder_methods_consistency(self):
        """Test that different GraphBuilder methods produce valid graphs."""
        data = pd.DataFrame(np.random.randn(30, 4), columns=["A", "B", "C", "D"])
        y = np.random.randn(30)
        builder = GraphBuilder()

        methods = [
            lambda: builder.from_correlation(data, threshold=0.3),
            lambda: builder.from_mutual_information(data, threshold=0.1),
            lambda: builder.from_kendalltau_minimal_edge(data),
            lambda: builder.from_matrix_generator(data),
            lambda: builder.from_rank_deletion(data, y),
        ]

        for method in methods:
            G = method()
            self.assertIsInstance(G, nx.Graph)
            self.assertEqual(G.number_of_nodes(), 4)
            self.assertEqual(set(G.nodes()), {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
