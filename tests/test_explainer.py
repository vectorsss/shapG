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
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapG.explainer import (
    CharacteristicFunction,
    Explainer,
    GraphExplainer,
    ExactExplainer,
    ShapGExplainer,
    CISExplainer,
    RandomCSExplainer,
    QRCSExplainer,
    BlockQRCSExplainer,
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
        names = ["A", "B", "C"]
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
            verbose=True,
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
        relative_error = abs(total_value - grand_coalition_value) / max(
            abs(grand_coalition_value), 1e-6
        )
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


class TestRandomCSExplainer(unittest.TestCase):
    """Test RandomCSExplainer (Random Compressed Sensing) class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test RandomCSExplainer initialization."""
        explainer = RandomCSExplainer(
            characteristic_function=self.char_func,
            m=100,
            t=50,
            tolerance=1e-3,
            verbose=True,
            seed=42,
        )
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertEqual(explainer.m, 100)
        self.assertEqual(explainer.t, 50)
        self.assertEqual(explainer.tolerance, 1e-3)
        self.assertTrue(explainer.verbose)
        self.assertEqual(explainer.seed, 42)

    def test_default_parameters(self):
        """Test default parameter values."""
        explainer = RandomCSExplainer()
        self.assertIsInstance(explainer.characteristic_function, CoalitionDegree)
        self.assertEqual(explainer.m, 100)
        self.assertEqual(explainer.t, 50)
        self.assertEqual(explainer.tolerance, 1e-3)
        self.assertFalse(explainer.verbose)
        self.assertIsNone(explainer.seed)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = RandomCSExplainer(m=50, t=20, seed=42)
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # All values should be numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        # Create two separate instances and run them independently
        # Reset random state between runs to ensure reproducibility

        # First run
        np.random.seed(100)
        random.seed(100)
        explainer1 = RandomCSExplainer(m=50, t=20, seed=42)
        values1 = explainer1.fit_explain(self.G)

        # Second run with same setup
        np.random.seed(100)
        random.seed(100)
        explainer2 = RandomCSExplainer(m=50, t=20, seed=42)
        values2 = explainer2.fit_explain(self.G)

        # Should produce identical results with same seed
        for node in self.G.nodes():
            self.assertAlmostEqual(values1[node], values2[node], places=5)

    def test_different_parameters_produce_different_results(self):
        """Test that different parameters produce different results."""
        explainer1 = RandomCSExplainer(m=30, t=10, seed=42)
        explainer2 = RandomCSExplainer(m=100, t=50, seed=42)

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Should produce different results with different parameters
        self.assertNotEqual(values1, values2)

    def test_batch_explain(self):
        """Test batch explanation with multiple runs."""
        explainer = RandomCSExplainer(m=50, t=20, seed=42)
        explainer.fit(self.G)

        batch_results = explainer.explain_batch(n_runs=5)

        # Should return mean and std for each node
        self.assertIsInstance(batch_results, dict)
        for node in self.G.nodes():
            self.assertIn(node, batch_results)
            self.assertIsInstance(batch_results[node], tuple)
            self.assertEqual(len(batch_results[node]), 2)  # (mean, std)


class TestQRCSExplainer(unittest.TestCase):
    """Test QRCSExplainer (QR-based Compressed Sensing) class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test QRCSExplainer initialization."""
        explainer = QRCSExplainer(
            characteristic_function=self.char_func,
            n_measurements=50,
            tolerance=5e-5,
            use_fast_fallback=False,
            verbose=True,
        )
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertEqual(explainer.n_measurements, 50)
        self.assertEqual(explainer.tolerance, 5e-5)
        self.assertFalse(explainer.use_fast_fallback)
        self.assertTrue(explainer.verbose)

    def test_default_parameters(self):
        """Test default parameter values."""
        explainer = QRCSExplainer()
        self.assertIsInstance(explainer.characteristic_function, CoalitionDegree)
        self.assertIsNone(explainer.n_measurements)  # Auto-determined
        self.assertEqual(explainer.tolerance, 5e-5)
        self.assertFalse(explainer.use_fast_fallback)
        self.assertFalse(explainer.verbose)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = QRCSExplainer(n_measurements=10, use_fast_fallback=False)
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # All values should be numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_fast_fallback_mode(self):
        """Test fast fallback mode."""
        explainer1 = QRCSExplainer(use_fast_fallback=True)
        explainer2 = QRCSExplainer(use_fast_fallback=False)

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Both should produce valid results
        self.assertEqual(set(values1.keys()), set(values2.keys()))

        # Results might differ due to different computation methods
        # Fast fallback uses mean approximation

    def test_automatic_measurement_determination(self):
        """Test automatic determination of n_measurements."""
        explainer = QRCSExplainer()  # n_measurements=None
        explainer.fit(self.G)

        # Should automatically determine n_measurements
        self.assertIsNotNone(explainer.l)
        self.assertGreater(explainer.l, 0)

        values = explainer.explain()
        self.assertIsInstance(values, dict)

    def test_deterministic_results(self):
        """Test that QR-CS produces deterministic results."""
        explainer1 = QRCSExplainer(n_measurements=10, use_fast_fallback=False)
        explainer2 = QRCSExplainer(n_measurements=10, use_fast_fallback=False)

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Should produce identical results (deterministic method)
        for node in self.G.nodes():
            self.assertAlmostEqual(values1[node], values2[node], places=6)


class TestBlockQRCSExplainer(unittest.TestCase):
    """Test BlockQRCSExplainer (Block QR-based Compressed Sensing) class."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)])
        self.char_func = CoalitionDegree()

    def test_initialization(self):
        """Test BlockQRCSExplainer initialization."""
        explainer = BlockQRCSExplainer(
            characteristic_function=self.char_func,
            n_blocks=4,
            block_sizes=[10, 10, 10, 10],
            n_measurements_per_block=[5, 5, 5, 5],
            tolerance=5e-5,
            use_fast_fallback=False,
            parallel=True,
            max_workers=2,
            verbose=True,
        )
        self.assertEqual(explainer.characteristic_function, self.char_func)
        self.assertEqual(explainer.n_blocks, 4)
        self.assertEqual(explainer.block_sizes, [10, 10, 10, 10])
        self.assertEqual(explainer.n_measurements_per_block, [5, 5, 5, 5])
        self.assertEqual(explainer.tolerance, 5e-5)
        self.assertFalse(explainer.use_fast_fallback)
        self.assertTrue(explainer.parallel)
        self.assertEqual(explainer.max_workers, 2)
        self.assertTrue(explainer.verbose)

    def test_default_parameters(self):
        """Test default parameter values."""
        explainer = BlockQRCSExplainer()
        self.assertIsInstance(explainer.characteristic_function, CoalitionDegree)
        self.assertEqual(explainer.n_blocks, 4)
        self.assertIsNone(explainer.block_sizes)  # Auto-determined
        self.assertIsNone(explainer.n_measurements_per_block)  # Auto-determined
        self.assertEqual(explainer.tolerance, 5e-5)
        self.assertFalse(explainer.use_fast_fallback)
        self.assertTrue(explainer.parallel)
        self.assertIsNone(explainer.max_workers)
        self.assertFalse(explainer.verbose)

    def test_fit_and_explain(self):
        """Test fit and explain methods."""
        explainer = BlockQRCSExplainer(
            n_blocks=2, parallel=False, use_fast_fallback=False
        )
        values = explainer.fit_explain(self.G)

        # Check output structure
        self.assertIsInstance(values, dict)
        self.assertEqual(set(values.keys()), set(self.G.nodes()))

        # All values should be numeric
        for v in values.values():
            self.assertIsInstance(v, (int, float))

    def test_parallel_vs_sequential(self):
        """Test parallel vs sequential computation."""
        explainer_parallel = BlockQRCSExplainer(n_blocks=2, parallel=True)
        explainer_sequential = BlockQRCSExplainer(n_blocks=2, parallel=False)

        values_parallel = explainer_parallel.fit_explain(self.G)
        values_sequential = explainer_sequential.fit_explain(self.G)

        # Should produce similar results
        for node in self.G.nodes():
            self.assertAlmostEqual(
                values_parallel[node], values_sequential[node], places=5
            )

    def test_different_block_sizes(self):
        """Test with different numbers of blocks."""
        explainer1 = BlockQRCSExplainer(
            n_blocks=2, parallel=False, use_fast_fallback=True
        )
        explainer2 = BlockQRCSExplainer(
            n_blocks=4, parallel=False, use_fast_fallback=True
        )

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Both should produce valid results
        self.assertEqual(set(values1.keys()), set(values2.keys()))

        # Results might vary due to different block decomposition
        # Just check that all values are reasonable (not NaN or infinite)
        for node in self.G.nodes():
            self.assertIsInstance(values1[node], (int, float))
            self.assertIsInstance(values2[node], (int, float))
            self.assertFalse(np.isnan(values1[node]))
            self.assertFalse(np.isnan(values2[node]))
            self.assertFalse(np.isinf(values1[node]))
            self.assertFalse(np.isinf(values2[node]))

    def test_automatic_block_setup(self):
        """Test automatic block size determination."""
        explainer = BlockQRCSExplainer(n_blocks=3)
        explainer.fit(self.G)

        # Should automatically determine block sizes
        self.assertIsNotNone(explainer.block_sizes)
        self.assertEqual(len(explainer.block_sizes), 3)
        self.assertEqual(sum(explainer.block_sizes), explainer.m_total)

        # Should automatically determine measurements per block
        self.assertIsNotNone(explainer.n_measurements_per_block)
        self.assertEqual(len(explainer.n_measurements_per_block), 3)

    def test_deterministic_results(self):
        """Test that Block QR-CS produces deterministic results."""
        explainer1 = BlockQRCSExplainer(
            n_blocks=2, parallel=False, use_fast_fallback=False
        )
        explainer2 = BlockQRCSExplainer(
            n_blocks=2, parallel=False, use_fast_fallback=False
        )

        values1 = explainer1.fit_explain(self.G)
        values2 = explainer2.fit_explain(self.G)

        # Should produce identical results (deterministic method)
        for node in self.G.nodes():
            self.assertAlmostEqual(values1[node], values2[node], places=6)


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

        self.assertAlmostEqual(
            sum(exact_values.values()), grand_coalition_value, places=6
        )
        self.assertAlmostEqual(
            sum(cis_values.values()), grand_coalition_value, places=6
        )

        # ShapG should be approximately efficient
        relative_error = (
            abs(sum(shapg_values.values()) - grand_coalition_value)
            / grand_coalition_value
        )
        self.assertLess(relative_error, 0.3)

    def test_explainer_with_different_data_types(self):
        """Test explainers with different input data types."""
        # NetworkX graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        # NumPy array
        data_np = np.random.randn(10, 3)

        # Pandas DataFrame
        data_pd = pd.DataFrame(data_np, columns=["A", "B", "C"])

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
        data = pd.DataFrame(
            np.random.randn(10, 3), columns=["Feature_A", "Feature_B", "Feature_C"]
        )

        explainer = ExactExplainer(CoalitionDegree())
        values = explainer.fit_explain(data)

        # Should use DataFrame column names as keys
        expected_keys = {"Feature_A", "Feature_B", "Feature_C"}
        self.assertEqual(set(values.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
