"""
Comprehensive tests for the new modular API.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import Mock

# Import new API components
from shapG import (
    CharacteristicFunction,
    Explainer,
    GraphExplainer,
    ExactExplainer,
    ShapGExplainer,
    CISExplainer,
    RandomCSExplainer,
    QRCSExplainer,
    CoalitionDegree,
    NodeCount,
    WeightedSum,
    CustomFunction,
    GraphBuilder,
    CoalitionManager,
    FeatureImportanceVisualizer,
)
from shapG.explainer import BlockQRCSExplainer


class TestCharacteristicFunctions(unittest.TestCase):
    """Test characteristic function implementations."""

    def setUp(self):
        """Set up test graph."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_coalition_degree(self):
        """Test coalition degree function."""
        func = CoalitionDegree()

        # Empty coalition
        self.assertEqual(func(set(), self.G), 0.0)

        # Single node
        self.assertEqual(func({0}, self.G), 0.0)

        # Two connected nodes
        self.assertEqual(func({0, 1}, self.G), 1.0)

        # All nodes
        self.assertEqual(func({0, 1, 2, 3}, self.G), 4.0)

    def test_node_count(self):
        """Test node count function."""
        func = NodeCount()

        self.assertEqual(func(set(), None), 0.0)
        self.assertEqual(func({1, 2, 3}, None), 3.0)

    def test_weighted_sum(self):
        """Test weighted sum function."""
        weights = np.array([1.0, 2.0, 3.0])
        func = WeightedSum(weights)

        self.assertEqual(func({0}, None), 1.0)
        self.assertEqual(func({0, 2}, None), 4.0)

        # With context
        context = np.array([10, 20, 30])
        self.assertEqual(func({1}, context), 40.0)  # 20 * 2.0

    def test_custom_function(self):
        """Test custom function wrapper."""

        def my_func(coalition, context):
            return len(coalition) * 2

        func = CustomFunction(my_func)
        self.assertEqual(func({1, 2}, None), 4)


class TestGraphBuilder(unittest.TestCase):
    """Test graph construction utilities."""

    def test_from_correlation(self):
        """Test building graph from correlation."""
        data = pd.DataFrame(np.random.randn(100, 5))
        builder = GraphBuilder()

        G = builder.from_correlation(data, threshold=0.3)

        self.assertEqual(G.number_of_nodes(), 5)
        self.assertIsInstance(G, nx.Graph)

    def test_from_adjacency(self):
        """Test building graph from adjacency matrix."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        builder = GraphBuilder()

        G = builder.from_adjacency(adj)

        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 2)

    def test_random_graph(self):
        """Test random graph generation."""
        builder = GraphBuilder()

        G = builder.random_graph(10, density=0.5, seed=42)

        self.assertEqual(G.number_of_nodes(), 10)
        # Approximate edge count
        max_edges = 10 * 9 // 2
        expected_edges = int(0.5 * max_edges)
        self.assertAlmostEqual(G.number_of_edges(), expected_edges, delta=5)


class TestCoalitionManager(unittest.TestCase):
    """Test coalition management utilities."""

    def setUp(self):
        """Set up test graph."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        self.manager = CoalitionManager(self.G)

    def test_get_neighbors_coalition(self):
        """Test getting neighbor coalitions."""
        # Depth 1
        coalition = self.manager.get_neighbors_coalition(2, depth=1, include_self=False)
        self.assertEqual(coalition, {1, 3})

        # Depth 2
        coalition = self.manager.get_neighbors_coalition(2, depth=2, include_self=True)
        self.assertEqual(coalition, {0, 1, 2, 3, 4})

    def test_sample_coalitions_uniform(self):
        """Test uniform coalition sampling."""
        samples = self.manager.sample_coalitions(2, n_samples=10, strategy="uniform")

        self.assertEqual(len(samples), 10)
        for sample in samples:
            self.assertNotIn(2, sample)  # Target node should not be in samples

    def test_sample_coalitions_stratified(self):
        """Test stratified coalition sampling."""
        samples = self.manager.sample_coalitions(2, n_samples=10, strategy="stratified")

        self.assertEqual(len(samples), 10)
        # Check that we have various coalition sizes
        sizes = [len(s) for s in samples]
        self.assertTrue(len(set(sizes)) > 1)


class TestExplainers(unittest.TestCase):
    """Test explainer implementations."""

    def setUp(self):
        """Set up test data and graph."""
        # Create a simple test graph
        self.G = nx.Graph()
        self.G.add_edges_from(
            [(0, 1, {"weight": 1}), (1, 2, {"weight": 2}), (2, 0, {"weight": 3})]
        )

        # Create test data
        np.random.seed(42)
        self.data = pd.DataFrame(np.random.randn(100, 5))

    def test_exact_explainer(self):
        """Test exact Shapley value computation."""
        explainer = ExactExplainer()
        shapley_values = explainer.fit_explain(self.G)

        # Check basic properties
        self.assertEqual(len(shapley_values), 3)
        self.assertIn(0, shapley_values)
        self.assertIn(1, shapley_values)
        self.assertIn(2, shapley_values)

        # Values should sum to total coalition value
        char_func = CoalitionDegree()
        total_value = char_func({0, 1, 2}, self.G)
        self.assertAlmostEqual(sum(shapley_values.values()), total_value, places=5)

    def test_shapg_explainer(self):
        """Test ShapG approximate computation."""
        explainer = ShapGExplainer(depth=1, n_samples=10)
        shapley_values = explainer.fit_explain(self.G)

        # Check basic properties
        self.assertEqual(len(shapley_values), 3)
        for node in self.G.nodes():
            self.assertIn(node, shapley_values)

    def test_shapg_with_coalitions(self):
        """Test ShapG with predefined coalitions."""
        explainer = ShapGExplainer()
        explainer.fit(self.G)

        # Set custom coalitions
        coalitions = {0: {1}, 1: {0, 2}, 2: {1}}
        explainer.set_coalitions(coalitions)

        shapley_values = explainer.explain()
        self.assertEqual(len(shapley_values), 3)

    def test_cis_explainer(self):
        """Test CIS explainer."""
        # Create mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5]))

        data = np.random.randn(10, 3)
        explainer = CISExplainer(model, n_samples=5)
        importance = explainer.fit_explain(data)

        self.assertEqual(len(importance), 3)
        model.predict.assert_called()

    def test_random_cs_explainer(self):
        """Test Random Compressed Sensing explainer."""
        explainer = RandomCSExplainer(m=50, t=30, verbose=False, seed=42)
        shapley_values = explainer.fit_explain(self.G)

        self.assertEqual(len(shapley_values), 3)
        for node in self.G.nodes():
            self.assertIn(node, shapley_values)
            self.assertIsInstance(shapley_values[node], (int, float))

    def test_qrcs_explainer(self):
        """Test QR-based Compressed Sensing explainer."""
        explainer = QRCSExplainer(n_measurements=10, use_fast_fallback=False)
        shapley_values = explainer.fit_explain(self.G)

        self.assertEqual(len(shapley_values), 3)
        for node in self.G.nodes():
            self.assertIn(node, shapley_values)
            self.assertIsInstance(shapley_values[node], (int, float))

    def test_block_qrcs_explainer(self):
        """Test Block QR-CS explainer."""
        explainer = BlockQRCSExplainer(
            n_blocks=2, parallel=False, use_fast_fallback=False
        )
        shapley_values = explainer.fit_explain(self.G)

        self.assertEqual(len(shapley_values), 3)
        for node in self.G.nodes():
            self.assertIn(node, shapley_values)
            self.assertIsInstance(shapley_values[node], (int, float))

    def test_explainer_from_data(self):
        """Test building graph from data in explainer."""
        explainer = ShapGExplainer()
        shapley_values = explainer.fit_explain(self.data, threshold=0.3)

        self.assertEqual(len(shapley_values), 5)  # 5 features

    def test_feature_names(self):
        """Test setting and getting feature names."""
        explainer = ShapGExplainer()

        names = ["feature_1", "feature_2", "feature_3"]
        explainer.set_feature_names(names)

        self.assertEqual(explainer.get_feature_names(), names)


class TestVisualization(unittest.TestCase):
    """Test visualization components."""

    def test_visualizer_creation(self):
        """Test creating visualizer."""
        viz = FeatureImportanceVisualizer()
        self.assertIsNotNone(viz)

    def test_plot_importance(self):
        """Test plotting importance scores."""
        viz = FeatureImportanceVisualizer()

        scores = {0: 0.5, 1: 0.3, 2: 0.2}
        fig, ax = viz.plot_importance(scores, show_values=True, show_plot=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_comparison(self):
        """Test plotting comparison of methods."""
        viz = FeatureImportanceVisualizer()

        importance_dict = {
            "Method1": {0: 0.5, 1: 0.3, 2: 0.2},
            "Method2": {0: 0.4, 1: 0.4, 2: 0.2},
        }

        fig, ax = viz.plot_comparison(importance_dict, show_plot=False)

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self):
        """Test complete analysis workflow."""
        # Generate data
        np.random.seed(42)
        data = pd.DataFrame(np.random.randn(100, 10))

        # Build graph
        builder = GraphBuilder()
        G = builder.from_correlation(data, threshold=0.3)

        # Compute Shapley values with different methods
        exact_explainer = ExactExplainer()
        shapg_explainer = ShapGExplainer(depth=2, n_samples=20)

        # Only test on small subgraph for exact computation
        subgraph = G.subgraph(list(G.nodes())[:5])

        exact_values = exact_explainer.fit_explain(subgraph)
        approx_values = shapg_explainer.fit_explain(subgraph)

        # Check consistency
        self.assertEqual(set(exact_values.keys()), set(approx_values.keys()))

        # Visualize (without showing)
        viz = FeatureImportanceVisualizer()
        fig, ax = viz.plot_comparison(
            {"Exact": exact_values, "Approximate": approx_values}, show_plot=False
        )

        self.assertIsNotNone(fig)

    def test_custom_characteristic_function(self):
        """Test using custom characteristic function."""

        # Define custom function
        def my_characteristic(coalition, graph):
            if not coalition:
                return 0
            # Custom logic: sum of node indices
            return sum(coalition)

        custom_func = CustomFunction(my_characteristic)

        # Use with explainer
        G = nx.complete_graph(4)
        explainer = ShapGExplainer(characteristic_function=custom_func)
        values = explainer.fit_explain(G)

        self.assertEqual(len(values), 4)


if __name__ == "__main__":
    unittest.main()
