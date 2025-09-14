"""
Comprehensive tests for the characteristic module (shapG/characteristic/).
Tests all characteristic function classes with various coalition types and contexts.
"""

import unittest
import numpy as np
import networkx as nx
from unittest.mock import Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shapG.characteristic import (
    CoalitionDegree,
    NodeCount,
    WeightedSum,
    CustomFunction,
    CombinedImputation
)


class TestCoalitionDegree(unittest.TestCase):
    """Test CoalitionDegree characteristic function."""

    def setUp(self):
        """Set up test fixtures."""
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 1)])  # Triangle + extra node

    def test_initialization(self):
        """Test CoalitionDegree initialization."""
        func = CoalitionDegree()
        self.assertEqual(func.name, "CoalitionDegree")

    def test_single_node_coalition(self):
        """Test with single node coalition."""
        func = CoalitionDegree()

        # Single node subgraphs have no edges, so degree = 0
        value = func({0}, self.G)
        self.assertEqual(value, 0.0)

        value = func({1}, self.G)
        self.assertEqual(value, 0.0)

        value = func({3}, self.G)
        self.assertEqual(value, 0.0)

    def test_multiple_node_coalition(self):
        """Test with multiple node coalition."""
        func = CoalitionDegree()

        # Coalition {0, 1}: subgraph has edge (0,1), so total degree = 2, divided by 2 = 1.0
        value = func({0, 1}, self.G)
        self.assertEqual(value, 1.0)

        # Coalition {0, 1, 2}: triangle subgraph, each node has degree 2, total = 6, divided by 2 = 3.0
        value = func({0, 1, 2}, self.G)
        self.assertEqual(value, 3.0)

    def test_empty_coalition(self):
        """Test with empty coalition."""
        func = CoalitionDegree()
        value = func(set(), self.G)
        self.assertEqual(value, 0.0)

    def test_full_coalition(self):
        """Test with full coalition."""
        func = CoalitionDegree()
        all_nodes = set(self.G.nodes())
        value = func(all_nodes, self.G)

        # Should equal total degree of original graph divided by 2 (each edge counted once)
        expected = sum(dict(self.G.degree()).values()) / 2
        self.assertEqual(value, expected)

    def test_disconnected_coalition(self):
        """Test coalition with disconnected nodes."""
        func = CoalitionDegree()

        # Nodes 0 and 3 are not connected
        value = func({0, 3}, self.G)
        # Subgraph has no edges, so total degree = 0
        self.assertEqual(value, 0.0)

    def test_nonexistent_nodes(self):
        """Test coalition with nonexistent nodes."""
        func = CoalitionDegree()

        # Node 10 doesn't exist in the graph
        value = func({0, 10}, self.G)
        # Should only consider existing nodes
        self.assertEqual(value, 0.0)  # Only node 0, no edges in subgraph

    def test_weighted_graph(self):
        """Test with weighted graph."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 0, 1.5)])

        func = CoalitionDegree()

        # Single node: no edges in subgraph
        value = func({0}, G)
        self.assertEqual(value, 0.0)

        # Two connected nodes: should sum edge weights in subgraph / 2
        value = func({0, 1}, G)
        # Edge (0,1) has weight 2.0, each endpoint contributes 2.0, total = 4.0, divided by 2 = 2.0
        self.assertEqual(value, 2.0)

    def test_no_context_provided(self):
        """Test behavior when no context (graph) is provided."""
        func = CoalitionDegree()

        # Should handle gracefully, return 0 or raise appropriate error
        with self.assertRaises((AttributeError, TypeError)):
            func({0, 1}, None)


class TestNodeCount(unittest.TestCase):
    """Test NodeCount characteristic function."""

    def test_initialization(self):
        """Test NodeCount initialization."""
        func = NodeCount()
        self.assertEqual(func.name, "NodeCount")

    def test_coalition_counting(self):
        """Test that it correctly counts coalition size."""
        func = NodeCount()

        # Empty coalition
        self.assertEqual(func(set()), 0.0)

        # Single node
        self.assertEqual(func({0}), 1.0)

        # Multiple nodes
        self.assertEqual(func({0, 1, 2}), 3.0)

        # Large coalition
        self.assertEqual(func(set(range(100))), 100.0)

    def test_context_independence(self):
        """Test that context doesn't affect result."""
        func = NodeCount()
        coalition = {0, 1, 2}

        # Should give same result regardless of context
        result1 = func(coalition, None)
        result2 = func(coalition, nx.Graph())
        result3 = func(coalition, "some string")

        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
        self.assertEqual(result1, 3.0)


class TestWeightedSum(unittest.TestCase):
    """Test WeightedSum characteristic function."""

    def test_initialization_with_weights(self):
        """Test WeightedSum initialization with weights."""
        weights = {0: 1.0, 1: 2.0, 2: 3.0}
        func = WeightedSum(weights)
        self.assertEqual(func.weights, weights)
        self.assertEqual(func.name, "WeightedSum")

    def test_initialization_without_weights(self):
        """Test WeightedSum initialization without weights."""
        func = WeightedSum()
        self.assertEqual(func.weights, {})

    def test_weighted_sum_computation(self):
        """Test weighted sum computation."""
        weights = {0: 1.5, 1: 2.0, 2: 0.5, 3: 3.0}
        func = WeightedSum(weights)

        # Single node
        self.assertEqual(func({0}), 1.5)
        self.assertEqual(func({3}), 3.0)

        # Multiple nodes
        self.assertEqual(func({0, 1}), 1.5 + 2.0)
        self.assertEqual(func({1, 2, 3}), 2.0 + 0.5 + 3.0)

    def test_missing_weights(self):
        """Test behavior with missing weights."""
        weights = {0: 1.0, 1: 2.0}
        func = WeightedSum(weights)

        # Node 2 not in weights, should default to 0
        self.assertEqual(func({0, 2}), 1.0)
        self.assertEqual(func({2}), 0.0)

    def test_empty_coalition(self):
        """Test with empty coalition."""
        weights = {0: 1.0, 1: 2.0}
        func = WeightedSum(weights)
        self.assertEqual(func(set()), 0.0)

    def test_weight_updating(self):
        """Test updating weights after initialization."""
        func = WeightedSum()

        # Initially empty weights
        self.assertEqual(func({0, 1}), 0.0)

        # Update weights
        func.weights = {0: 5.0, 1: 10.0}
        self.assertEqual(func({0, 1}), 15.0)

    def test_negative_weights(self):
        """Test with negative weights."""
        weights = {0: 1.0, 1: -2.0, 2: 3.0}
        func = WeightedSum(weights)

        self.assertEqual(func({0, 1}), 1.0 + (-2.0))
        self.assertEqual(func({1, 2}), -2.0 + 3.0)


class TestCustomFunction(unittest.TestCase):
    """Test CustomFunction wrapper."""

    def test_initialization(self):
        """Test CustomFunction initialization."""
        def test_func(coalition, context=None):
            return len(coalition)

        func = CustomFunction(test_func, name="TestFunction")
        self.assertEqual(func.name, "TestFunction")
        self.assertEqual(func.func, test_func)

    def test_initialization_without_name(self):
        """Test initialization without explicit name."""
        def test_func(coalition, context=None):
            return len(coalition)

        func = CustomFunction(test_func)
        self.assertEqual(func.name, "CustomFunction")

    def test_function_calling(self):
        """Test that wrapped function is called correctly."""
        def test_func(coalition, context=None):
            return len(coalition) * 2

        func = CustomFunction(test_func)

        self.assertEqual(func({0}), 2)
        self.assertEqual(func({0, 1, 2}), 6)
        self.assertEqual(func(set()), 0)

    def test_context_passing(self):
        """Test that context is passed to wrapped function."""
        def test_func(coalition, context=None):
            if context is None:
                return len(coalition)
            else:
                return len(coalition) + context

        func = CustomFunction(test_func)

        self.assertEqual(func({0, 1}), 2)  # No context
        self.assertEqual(func({0, 1}, 10), 12)  # With context

    def test_complex_function(self):
        """Test with more complex wrapped function."""
        def graph_based_func(coalition, context=None):
            if context is None or not hasattr(context, 'nodes'):
                return len(coalition) ** 2

            # Count edges within coalition
            subgraph = context.subgraph(coalition)
            return subgraph.number_of_edges()

        func = CustomFunction(graph_based_func, "GraphEdgeCount")

        # Test without graph context
        self.assertEqual(func({0, 1, 2}), 9)  # 3^2

        # Test with graph context
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.assertEqual(func({0, 1, 2}, G), 3)  # Triangle has 3 edges

    def test_function_with_kwargs(self):
        """Test function that uses keyword arguments."""
        def test_func(coalition, context=None, multiplier=1):
            return len(coalition) * multiplier

        # Note: CustomFunction doesn't support kwargs directly,
        # but we can test with lambda
        func = CustomFunction(lambda c, ctx: test_func(c, ctx, multiplier=5))

        self.assertEqual(func({0, 1}), 10)  # 2 * 5


class TestCombinedImputation(unittest.TestCase):
    """Test CombinedImputation characteristic function."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        # Create mock model
        self.model = Mock()
        self.model.predict = Mock(return_value=np.array([0.5, 0.6, 0.7]))

    def test_initialization(self):
        """Test CombinedImputation initialization."""
        func = CombinedImputation(self.data, self.model)

        self.assertEqual(func.name, "CombinedImputation")
        np.testing.assert_array_equal(func.data, self.data)
        self.assertEqual(func.model, self.model)
        np.testing.assert_array_equal(func.baseline, np.zeros(3))

    def test_initialization_with_baseline(self):
        """Test initialization with custom baseline."""
        baseline = np.array([1.0, 2.0, 3.0])
        func = CombinedImputation(self.data, self.model, baseline)

        np.testing.assert_array_equal(func.baseline, baseline)

    def test_full_coalition_no_context(self):
        """Test with full coalition (no imputation needed)."""
        func = CombinedImputation(self.data, self.model)

        # Full coalition: all features present
        result = func({0, 1, 2}, context=None)

        # Should use original data
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]
        np.testing.assert_array_equal(predicted_data, self.data)

        # Should return mean of predictions
        self.assertEqual(result, 0.6)  # Mean of [0.5, 0.6, 0.7]

    def test_partial_coalition_no_context(self):
        """Test with partial coalition (imputation needed)."""
        func = CombinedImputation(self.data, self.model)

        # Partial coalition: only features 0 and 2
        result = func({0, 2}, context=None)

        # Should impute feature 1 with baseline (0)
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]

        expected_data = self.data.copy()
        expected_data[:, 1] = 0.0  # Feature 1 imputed with baseline

        np.testing.assert_array_equal(predicted_data, expected_data)

    def test_empty_coalition_no_context(self):
        """Test with empty coalition."""
        func = CombinedImputation(self.data, self.model)

        result = func(set(), context=None)

        # Should impute all features with baseline
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]

        expected_data = np.zeros_like(self.data)
        np.testing.assert_array_equal(predicted_data, expected_data)

    def test_single_sample_context(self):
        """Test with single sample context."""
        func = CombinedImputation(self.data, self.model)
        self.model.predict.return_value = np.array([0.8])

        # Use sample index 1 as context
        result = func({0, 2}, context=1)

        # Should use only row 1 of data
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]

        # Should be single row with feature 1 imputed
        expected_sample = np.array([[4.0, 0.0, 6.0]])  # Row 1 with feature 1 = 0
        np.testing.assert_array_equal(predicted_data, expected_sample)

        self.assertEqual(result, 0.8)

    def test_non_integer_context(self):
        """Test with non-integer context (should use default behavior)."""
        func = CombinedImputation(self.data, self.model)

        # Pass string context - should use default (no context) behavior
        result = func({0, 1}, context="some_string")

        # Should behave like context=None
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]

        expected_data = self.data.copy()
        expected_data[:, 2] = 0.0  # Feature 2 imputed

        np.testing.assert_array_equal(predicted_data, expected_data)

    def test_custom_baseline(self):
        """Test with custom baseline values."""
        baseline = np.array([10.0, 20.0, 30.0])
        func = CombinedImputation(self.data, self.model, baseline)

        result = func({0}, context=None)

        # Should impute features 1 and 2 with baseline values
        self.model.predict.assert_called_once()
        predicted_data = self.model.predict.call_args[0][0]

        expected_data = self.data.copy()
        expected_data[:, 1] = 20.0  # Feature 1 imputed with baseline
        expected_data[:, 2] = 30.0  # Feature 2 imputed with baseline

        np.testing.assert_array_equal(predicted_data, expected_data)

    def test_edge_case_coalitions(self):
        """Test edge cases in coalition handling."""
        func = CombinedImputation(self.data, self.model)

        # Coalition with indices beyond data dimensions
        result = func({0, 5}, context=None)  # Index 5 doesn't exist

        # Should handle gracefully - only existing features considered
        self.model.predict.assert_called_once()

    def test_model_prediction_error_handling(self):
        """Test handling of model prediction errors."""
        func = CombinedImputation(self.data, self.model)

        # Make model raise an exception
        self.model.predict.side_effect = Exception("Model error")

        with self.assertRaises(Exception):
            func({0, 1}, context=None)


class TestCharacteristicFunctionIntegration(unittest.TestCase):
    """Integration tests for characteristic functions."""

    def test_all_functions_basic_properties(self):
        """Test that all functions satisfy basic properties."""
        # Set up test data
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        weights = {0: 1.0, 1: 2.0, 2: 3.0}
        data = np.random.randn(5, 3)
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5]))

        def custom_func(coalition, context=None):
            return len(coalition) * 2

        # Create all function types
        functions = [
            CoalitionDegree(),
            NodeCount(),
            WeightedSum(weights),
            CustomFunction(custom_func),
            CombinedImputation(data, model)
        ]

        # Test basic properties
        for func in functions:
            # Some functions need context
            context = G if isinstance(func, CoalitionDegree) else None

            # Empty coalition should return numeric value
            result = func(set(), context)
            self.assertIsInstance(result, (int, float, np.number))

            # Single node coalition should return numeric value
            result = func({0}, context)
            self.assertIsInstance(result, (int, float, np.number))

            # Multiple node coalition should return numeric value
            result = func({0, 1}, context)
            self.assertIsInstance(result, (int, float, np.number))

    def test_monotonicity_property(self):
        """Test monotonicity where applicable."""
        # NodeCount should be monotonic
        func = NodeCount()

        self.assertLessEqual(func(set()), func({0}))
        self.assertLessEqual(func({0}), func({0, 1}))
        self.assertLessEqual(func({0, 1}), func({0, 1, 2}))

    def test_efficiency_property_components(self):
        """Test components needed for efficiency property."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        func = CoalitionDegree()

        # Individual values
        individual_sum = sum(func({node}, G) for node in G.nodes())

        # Grand coalition value
        grand_value = func(set(G.nodes()), G)

        # For coalition degree, these should be related
        # (not necessarily equal due to how edges are counted)
        self.assertIsInstance(individual_sum, (int, float))
        self.assertIsInstance(grand_value, (int, float))

    def test_batch_computation_consistency(self):
        """Test that batch computation gives same results as individual calls."""
        func = NodeCount()
        coalitions = [set(), {0}, {0, 1}, {0, 1, 2}]

        # Individual computations
        individual_results = [func(coalition) for coalition in coalitions]

        # Batch computation
        batch_results = func.batch_compute(coalitions)

        np.testing.assert_array_equal(individual_results, batch_results)


if __name__ == '__main__':
    unittest.main()