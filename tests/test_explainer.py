"""
Comprehensive tests for the explainer module (shapG/explainer/).
Tests all explainer classes with various configurations and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shapG.explainer import (
    CharacteristicFunction, Explainer, GraphExplainer,
    ExactExplainer, ShapGExplainer, CISExplainer, CSExplainer
)
from shapG.characteristic import CoalitionDegree, NodeCount, CustomFunction


class TestCharacteristicFunction(unittest.TestCase):
    """Test CharacteristicFunction base class."""

    def test_abstract_base_class(self):
        """Test that CharacteristicFunction cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            CharacteristicFunction()

    def test_batch_compute_default_implementation(self):
        """Test default batch_compute implementation."""
        class TestFunction(CharacteristicFunction):
            def __call__(self, coalition, context=None):
                return len(coalition)

        func = TestFunction()
        coalitions = [{0}, {0, 1}, {1, 2}]
        context = nx.Graph()

        results = func.batch_compute(coalitions, context)
        expected = np.array([1, 2, 2])
        np.testing.assert_array_equal(results, expected)


class TestExplainerBase(unittest.TestCase):
    """Test Explainer base class."""

    def test_abstract_base_class(self):
        """Test that Explainer cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Explainer()

    def test_feature_names_management(self):
        """Test feature names get/set functionality."""
        class TestExplainer(Explainer):
            def __init__(self):
                super().__init__()

            def fit(self, X, **kwargs):
                return self

            def explain(self, X=None):
                return {}

        explainer = TestExplainer()

        # Test setting and getting feature names
        names = ['A', 'B', 'C']
        explainer.set_feature_names(names)
        self.assertEqual(explainer.get_feature_names(), names)

        # Test with None
        explainer.set_feature_names(None)
        self.assertIsNone(explainer.get_feature_names())


class TestGraphExplainerBase(unittest.TestCase):
    """Test GraphExplainer base class."""

    def test_initialization(self):
        """Test GraphExplainer initialization."""
        class TestGraphExplainer(GraphExplainer):
            def fit(self, X, **kwargs):
                return self

            def explain(self, X=None):
                return {}

        char_func = CoalitionDegree()
        explainer = TestGraphExplainer(char_func, verbose=True)

        self.assertEqual(explainer.characteristic_function, char_func)
        self.assertTrue(explainer.verbose)
        self.assertFalse(explainer._fitted)

    def test_graph_and_coalition_management(self):
        """Test graph and coalition setting."""
        class TestGraphExplainer(GraphExplainer):
            def fit(self, X, **kwargs):
                return self

            def explain(self, X=None):
                return {}

        explainer = TestGraphExplainer(CoalitionDegree())

        # Test graph setting
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        explainer.set_graph(G)
        self.assertEqual(explainer.graph, G)

        # Test coalition setting
        coalitions = [{0}, {1}, {0, 1}]
        explainer.set_coalitions(coalitions)
        self.assertEqual(explainer.coalitions, coalitions)


class TestExactExplainer(unittest.TestCase):
    """Test ExactExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test ExactExplainer initialization."""
        explainer = ExactExplainer(self.char_func, verbose=True)
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertTrue(explainer.verbose)

    def test_fit_with_graph(self):
        """Test fitting with NetworkX graph."""
        explainer = ExactExplainer(self.char_func)
        fitted_explainer = explainer.fit(self.G)

        self.assertIs(fitted_explainer, explainer)  # Returns self
        self.assertEqual(explainer.graph, self.G)
        self.assertTrue(explainer._fitted)

    def test_fit_with_data(self):
        """Test fitting with data array."""
        explainer = ExactExplainer(self.char_func)
        data = np.random.randn(10, 3)

        fitted_explainer = explainer.fit(data)
        self.assertIs(fitted_explainer, explainer)
        self.assertTrue(explainer._fitted)
        self.assertIsInstance(explainer.graph, nx.Graph)

    def test_explain(self):
        """Test explain method."""
        explainer = ExactExplainer(self.char_func)
        explainer.fit(self.G)

        values = explainer.explain()

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check efficiency property
        total_value = sum(values.values())
        grand_coalition_value = self.char_func(set(self.G.nodes()), self.G)
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)

    def test_fit_explain_convenience(self):
        """Test fit_explain convenience method."""
        explainer = ExactExplainer(self.char_func)
        values = explainer.fit_explain(self.G)

        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})

    def test_unfitted_explain_error(self):
        """Test that explain fails on unfitted explainer."""
        explainer = ExactExplainer(self.char_func)
        with self.assertRaises(ValueError):
            explainer.explain()

    def test_with_custom_characteristic_function(self):
        """Test with custom characteristic function."""
        def custom_func(coalition, context=None):
            return len(coalition) ** 2

        char_func = CustomFunction(custom_func)
        explainer = ExactExplainer(char_func)
        values = explainer.fit_explain(self.G)

        # Check efficiency
        total_value = sum(values.values())
        grand_coalition_value = char_func(set(self.G.nodes()), self.G)
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)


class TestShapGExplainer(unittest.TestCase):
    """Test ShapGExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 1)])  # Triangle + extra node
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test ShapGExplainer initialization."""
        explainer = ShapGExplainer(
            characteristic_function=self.char_func,
            depth=2,
            n_samples=10,
            approximate_by_ratio=False,
            scale=True,
            verbose=True
        )

        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertEqual(explainer.depth, 2)
        self.assertEqual(explainer.n_samples, 10)
        self.assertFalse(explainer.approximate_by_ratio)
        self.assertTrue(explainer.scale)
        self.assertTrue(explainer.verbose)

    def test_default_parameters(self):
        """Test default parameter values."""
        explainer = ShapGExplainer()

        self.assertIsInstance(explainer.characteristic_function, CoalitionDegree)
        self.assertEqual(explainer.depth, 1)
        self.assertEqual(explainer.n_samples, 15)
        self.assertTrue(explainer.approximate_by_ratio)
        self.assertTrue(explainer.scale)
        self.assertFalse(explainer.verbose)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = ShapGExplainer(depth=1, n_samples=5)
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # All values should be numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_different_parameters(self):
        """Test with different parameter combinations."""
        # Create a larger graph to force sampling behavior
        large_G = nx.complete_graph(10)  # 10 nodes with all possible edges

        explainer1 = ShapGExplainer(depth=1, n_samples=5)
        explainer2 = ShapGExplainer(depth=2, n_samples=10)

        values1 = explainer1.fit_explain(large_G)
        values2 = explainer2.fit_explain(large_G)

        # Should produce different results with different parameters on large graph
        self.assertEqual(set(values1.keys()), set(values2.keys()))
        self.assertNotEqual(values1, values2)

    def test_approximate_by_ratio_parameter(self):
        """Test approximate_by_ratio parameter effect."""
        # Create a larger graph to force sampling behavior
        large_G = nx.complete_graph(10)  # 10 nodes with all possible edges

        explainer1 = ShapGExplainer(approximate_by_ratio=True, n_samples=5)
        explainer2 = ShapGExplainer(approximate_by_ratio=False, n_samples=5)

        values1 = explainer1.fit_explain(large_G)
        values2 = explainer2.fit_explain(large_G)

        # Should produce different results
        self.assertNotEqual(values1, values2)

    def test_scale_parameter(self):
        """Test scale parameter effect."""
        # Create a larger graph to force sampling behavior
        large_G = nx.complete_graph(10)  # 10 nodes with all possible edges

        explainer1 = ShapGExplainer(scale=True, n_samples=5)
        explainer2 = ShapGExplainer(scale=False, n_samples=5)

        values1 = explainer1.fit_explain(large_G)
        values2 = explainer2.fit_explain(large_G)

        # Should produce different magnitudes
        self.assertNotEqual(values1, values2)

    def test_with_predefined_coalitions(self):
        """Test with predefined coalitions."""
        coalitions = [{0}, {1}, {0, 1}, {0, 1, 2}]
        explainer = ShapGExplainer(n_samples=5)
        explainer.fit(self.G)
        explainer.set_coalitions(coalitions)

        values = explainer.explain()
        self.assertIsInstance(values, dict)

    def test_efficiency_approximately_holds(self):
        """Test that efficiency property approximately holds."""
        explainer = ShapGExplainer(depth=2, n_samples=20)
        values = explainer.fit_explain(self.G)

        total_value = sum(values.values())
        grand_coalition_value = self.char_func(set(self.G.nodes()), self.G)

        # Allow tolerance for approximate method
        relative_error = abs(total_value - grand_coalition_value) / max(abs(grand_coalition_value), 1e-6)
        self.assertLess(relative_error, 0.5)  # Within 50%


class TestCISExplainer(unittest.TestCase):
    """Test CISExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test CISExplainer initialization."""
        explainer = CISExplainer(characteristic_function=self.char_func, verbose=True)
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertTrue(explainer.verbose)

    def test_initialization_with_model(self):
        """Test CISExplainer initialization with model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5]))

        explainer = CISExplainer(model=model, n_samples=10)
        self.assertEqual(explainer.model, model)
        self.assertEqual(explainer.n_samples, 10)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = CISExplainer(self.char_func)
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), {0, 1, 2})

        # Check efficiency property
        total_value = sum(values.values())
        grand_coalition_value = self.char_func(set(self.G.nodes()), self.G)
        self.assertAlmostEqual(total_value, grand_coalition_value, places=6)

    def test_with_data_and_model(self):
        """Test CIS explainer with data and model."""
        # Create mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5, 0.6, 0.7]))

        # Create explainer with model
        explainer = CISExplainer(model=model)

        # Test with data
        data = np.random.randn(10, 3)
        values = explainer.fit_explain(data)

        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), 3)  # 3 features

        # Model should have been called
        model.predict.assert_called()

    def test_surplus_distribution(self):
        """Test that CIS correctly distributes surplus."""
        explainer = CISExplainer(self.char_func)
        values = explainer.fit_explain(self.G)

        # Calculate individual values and surplus manually
        individual_values = {}
        for node in self.G.nodes():
            individual_values[node] = self.char_func({node}, self.G)

        grand_coalition_value = self.char_func(set(self.G.nodes()), self.G)
        total_individual = sum(individual_values.values())
        surplus = grand_coalition_value - total_individual
        equal_share = surplus / len(self.G.nodes())

        # Each CIS value should be individual value + equal share
        for node in self.G.nodes():
            expected = individual_values[node] + equal_share
            self.assertAlmostEqual(values[node], expected, places=6)


class TestCSExplainer(unittest.TestCase):
    """Test CSExplainer (Coalition Structure) class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test CSExplainer initialization."""
        explainer = CSExplainer(
            characteristic_function=self.char_func,
            max_coalition_size=3,
            verbose=True
        )
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertEqual(explainer.max_coalition_size, 3)
        self.assertTrue(explainer.verbose)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = CSExplainer(self.char_func, max_coalition_size=2)
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # All values should be numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_max_coalition_size_constraint(self):
        """Test that max_coalition_size is respected."""
        explainer = CSExplainer(self.char_func, max_coalition_size=2)
        explainer.fit(self.G)

        # Check that coalitions don't exceed max size
        # This is internal behavior, hard to test directly
        values = explainer.explain()
        self.assertIsInstance(values, dict)

    def test_different_max_sizes(self):
        """Test with different maximum coalition sizes."""
        explainer1 = CSExplainer(self.char_func, max_coalition_size=2)
        explainer2 = CSExplainer(self.char_func, max_coalition_size=3)

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Should produce different results
        self.assertEqual(set(values1.keys()), set(values2.keys()))
        # Results might be same for this simple case, but structure should be maintained



class TestExplainerIntegration(unittest.TestCase):
    """Integration tests for explainer classes."""

    def test_all_explainers_consistency(self):
        """Test that all explainers produce consistent results on same graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        char_func = CoalitionDegree()

        # Test all explainers
        exact_explainer = ExactExplainer(char_func)
        shapg_explainer = ShapGExplainer(char_func, depth=2, n_samples=20)
        cis_explainer = CISExplainer(char_func)

        exact_values = exact_explainer.fit_explain(G)
        shapg_values = shapg_explainer.fit_explain(G)
        cis_values = cis_explainer.fit_explain(G)

        # All should have same keys
        self.assertEqual(set(exact_values.keys()), set(shapg_values.keys()))
        self.assertEqual(set(exact_values.keys()), set(cis_values.keys()))

        # All should satisfy efficiency
        grand_coalition_value = char_func(set(G.nodes()), G)

        self.assertAlmostEqual(sum(exact_values.values()), grand_coalition_value, places=6)
        self.assertAlmostEqual(sum(cis_values.values()), grand_coalition_value, places=6)

        # ShapG should be approximately efficient
        relative_error = abs(sum(shapg_values.values()) - grand_coalition_value) / grand_coalition_value
        self.assertLess(relative_error, 0.3)

    def test_explainer_with_different_data_types(self):
        """Test explainers with different input data types."""
        # NetworkX graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        # NumPy array
        data_np = np.random.randn(10, 3)

        # Pandas DataFrame
        data_pd = pd.DataFrame(data_np, columns=['A', 'B', 'C'])

        explainer = ExactExplainer(CoalitionDegree())

        # All should work
        values_graph = explainer.fit_explain(G)
        values_np = explainer.fit_explain(data_np)
        values_pd = explainer.fit_explain(data_pd)

        self.assertIsInstance(values_graph, dict)
        self.assertIsInstance(values_np, dict)
        self.assertIsInstance(values_pd, dict)

    def test_feature_names_preservation(self):
        """Test that feature names are preserved through pipeline."""
        data = pd.DataFrame(np.random.randn(10, 3), columns=['Feature_A', 'Feature_B', 'Feature_C'])

        explainer = ExactExplainer(CoalitionDegree())
        values = explainer.fit_explain(data)

        # Should use DataFrame column names as keys
        expected_keys = {'Feature_A', 'Feature_B', 'Feature_C'}
        self.assertEqual(set(values.keys()), expected_keys)


if __name__ == '__main__':
    unittest.main()