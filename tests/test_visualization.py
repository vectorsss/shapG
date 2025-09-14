"""
Comprehensive tests for the visualization module (shapG/visualization/).
Tests plotting functions and visualization classes without actually displaying plots.

The test classes inherit from VisualizationTestBase which provides:
- Temporary directory creation and cleanup
- Helper methods for generating temp file paths
- File existence validation for saved plots

This approach saves plots to temporary files during testing instead of just
suppressing display, which allows verification that plots are actually created
and could enable future tests that validate plot content.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock
import tempfile
import os
import shutil

# Use non-interactive backend for testing
matplotlib.use('Agg')

from shapG.visualization.plot import plot
from shapG.visualization.visualization import FeatureImportanceVisualizer, plot_shapley_values


class VisualizationTestBase(unittest.TestCase):
    """Base class for visualization tests with plot file saving.

    This class provides infrastructure for testing visualization functions
    by saving plots to a dedicated test_outputs directory under the project root.

    Benefits:
    - Verifies that plots are actually created successfully
    - Saves plots for manual inspection and validation
    - Better testing practice than just using show_plot=False
    - Plots are preserved for review after test completion
    """

    def setUp(self):
        """Set up test output directory."""
        super().setUp()
        # Create test_outputs directory under project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_output_dir = os.path.join(project_root, 'test_outputs')

        # Create directory if it doesn't exist
        os.makedirs(self.test_output_dir, exist_ok=True)

    def get_test_filename(self, test_name=None, suffix='.png'):
        """Get a filename for saving test plots.

        Args:
            test_name: Name of the test (auto-detected if None)
            suffix: File extension (default: '.png')

        Returns:
            Path to save the test plot
        """
        if test_name is None:
            # Auto-detect test name from the calling method
            import inspect
            test_name = inspect.stack()[1].function

        # Create a descriptive filename
        class_name = self.__class__.__name__
        filename = f'{class_name}_{test_name}{suffix}'
        return os.path.join(self.test_output_dir, filename)


class TestPlotFunction(VisualizationTestBase):
    """Test the legacy plot function."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.shapley_values = {
            'Feature_A': 0.5,
            'Feature_B': -0.3,
            'Feature_C': 0.8,
            'Feature_D': 0.1,
            'Feature_E': -0.2
        }

    def test_basic_plot_creation(self):
        """Test basic plot creation and save to test_outputs directory."""
        test_file = self.get_test_filename()
        result = plot(self.shapley_values, file_name=test_file, show_plot=False)

        # Should return figure and axes
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        fig, ax = result
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        # Verify file was created
        self.assertTrue(os.path.exists(test_file), f"Plot file should be created at {test_file}")
        print(f"✓ Plot saved to: {test_file}")

        plt.close(fig)

    def test_top_n_parameter(self):
        """Test top_n parameter functionality."""
        # Test with top 3 and save to test_outputs directory
        test_file = self.get_test_filename()
        fig, ax = plot(self.shapley_values, top_n=3, file_name=test_file, show_plot=False)

        # Should show only top 3 features
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        visible_labels = [label for label in y_labels if label]  # Non-empty labels

        self.assertLessEqual(len(visible_labels), 3)

        # Verify file was created
        self.assertTrue(os.path.exists(test_file), f"Plot file should be created at {test_file}")
        print(f"✓ Top-3 plot saved to: {test_file}")

        plt.close(fig)

    def test_style_parameter(self):
        """Test different style parameters."""
        styles = ['default', 'seaborn-v0_8']

        for style in styles:
            try:
                fig, ax = plot(self.shapley_values, style=style, show_plot=False)
                self.assertIsInstance(fig, plt.Figure)
                plt.close(fig)
            except OSError:
                # Style might not be available, skip
                pass

    def test_feature_names_as_list(self):
        """Test feature_names parameter as list."""
        values = {0: 0.5, 1: -0.3, 2: 0.8}
        feature_names = ['Custom_A', 'Custom_B', 'Custom_C']

        fig, ax = plot(values, feature_names=feature_names, show_plot=False)

        # Check that custom names are used
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        # At least some of the custom names should appear
        self.assertTrue(any(name in y_labels for name in feature_names))

        plt.close(fig)

    def test_feature_names_as_dict(self):
        """Test feature_names parameter as dictionary."""
        values = {0: 0.5, 1: -0.3, 2: 0.8}
        feature_names = {0: 'Dict_A', 1: 'Dict_B', 2: 'Dict_C'}

        fig, ax = plot(values, feature_names=feature_names, show_plot=False)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_customization_parameters(self):
        """Test plot customization parameters."""
        fig, ax = plot(
            self.shapley_values,
            title='Custom Title',
            figsize=(10, 8),
            color='red',
            show_values=True,
            value_format='{:.3f}',
            show_plot=False
        )

        # Check title
        self.assertEqual(ax.get_title(), 'Custom Title')

        # Check figure size
        self.assertEqual(fig.get_figwidth(), 10)
        self.assertEqual(fig.get_figheight(), 8)

        plt.close(fig)

    def test_file_saving(self):
        """Test saving plot to file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name

        try:
            fig, ax = plot(
                self.shapley_values,
                file_name=temp_filename,
                show_plot=False
            )

            # File should be created
            self.assertTrue(os.path.exists(temp_filename))

            plt.close(fig)

        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_empty_values(self):
        """Test with empty shapley values."""
        empty_values = {}

        fig, ax = plot(empty_values, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)

        plt.close(fig)

    def test_single_value(self):
        """Test with single value."""
        single_value = {'OnlyFeature': 0.7}

        fig, ax = plot(single_value, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)

        plt.close(fig)

    def test_zero_values(self):
        """Test with all zero values."""
        zero_values = {'A': 0.0, 'B': 0.0, 'C': 0.0}

        fig, ax = plot(zero_values, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)

        plt.close(fig)

    def test_negative_values(self):
        """Test with negative values."""
        negative_values = {'A': -0.5, 'B': -0.3, 'C': -0.8}

        fig, ax = plot(negative_values, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)

        plt.close(fig)


class TestFeatureImportanceVisualizer(VisualizationTestBase):
    """Test FeatureImportanceVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.visualizer = FeatureImportanceVisualizer()
        self.shapley_values = {
            'Feature_A': 0.5,
            'Feature_B': -0.3,
            'Feature_C': 0.8,
            'Feature_D': 0.1,
            'Feature_E': -0.2
        }

    def test_initialization(self):
        """Test visualizer initialization."""
        viz = FeatureImportanceVisualizer()
        self.assertIsInstance(viz, FeatureImportanceVisualizer)

        # Test with custom parameters
        viz_custom = FeatureImportanceVisualizer(
            figsize=(12, 8),
            style='default'
        )
        self.assertEqual(viz_custom.figsize, (12, 8))
        self.assertEqual(viz_custom.style, 'default')

    def test_plot_importance(self):
        """Test plot_importance method with file saving."""
        test_file = self.get_test_filename()
        fig, ax = self.visualizer.plot_importance(
            self.shapley_values,
            title='Test Importance',
            filename=test_file,
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), 'Test Importance')

        # Verify file was created
        self.assertTrue(os.path.exists(test_file), f"Plot file should be created at {test_file}")
        print(f"✓ Importance plot saved to: {test_file}")

        plt.close(fig)

    def test_plot_comparison(self):
        """Test plot_comparison method."""
        methods_data = {
            'ShapG': self.shapley_values,
            'CIS': {
                'Feature_A': 0.4,
                'Feature_B': -0.2,
                'Feature_C': 0.7,
                'Feature_D': 0.15,
                'Feature_E': -0.25
            }
        }

        fig, ax = self.visualizer.plot_comparison(
            methods_data,
            title='Methods Comparison',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        plt.close(fig)

    def test_plot_heatmap(self):
        """Test plot_heatmap method."""
        # Create matrix data
        methods = ['ShapG', 'CIS', 'Exact']
        features = ['A', 'B', 'C', 'D']
        matrix_data = np.random.randn(len(features), len(methods))

        fig, ax = self.visualizer.plot_heatmap(
            matrix_data,
            row_labels=features,
            col_labels=methods,
            title='Heatmap Test',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        plt.close(fig)

    def test_plot_network(self):
        """Test plot_network method."""
        # Create test graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 1)])

        # Add node values
        node_values = {0: 0.5, 1: -0.3, 2: 0.8, 3: 0.1}

        fig, ax = self.visualizer.plot_network(
            G,
            node_values=node_values,
            title='Network Test',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

        plt.close(fig)

    def test_plot_network_without_values(self):
        """Test plot_network without node values."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        fig, ax = self.visualizer.plot_network(
            G,
            title='Network Without Values',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_parameters(self):
        """Test visualizer with different parameters."""
        viz = FeatureImportanceVisualizer(
            figsize=(15, 10),
            style='default'
        )

        fig, ax = viz.plot_importance(
            self.shapley_values,
            color='green',
            show_plot=False
        )

        self.assertEqual(fig.get_figwidth(), 15)
        self.assertEqual(fig.get_figheight(), 10)

        plt.close(fig)

    def test_file_saving_methods(self):
        """Test saving plots to files."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name

        try:
            fig, ax = self.visualizer.plot_importance(
                self.shapley_values,
                filename=temp_filename,
                show_plot=False
            )

            # File should be created
            self.assertTrue(os.path.exists(temp_filename))

            plt.close(fig)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty data
        fig, ax = self.visualizer.plot_importance({}, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Single value
        fig, ax = self.visualizer.plot_importance(
            {'Single': 0.5},
            show_plot=False
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotShapleyValues(unittest.TestCase):
    """Test plot_shapley_values function."""

    def setUp(self):
        """Set up test fixtures."""
        self.shapley_values = {
            'Feature_A': 0.5,
            'Feature_B': -0.3,
            'Feature_C': 0.8,
            'Feature_D': 0.1
        }

    def test_basic_functionality(self):
        """Test basic functionality."""
        fig, ax = plot_shapley_values(
            self.shapley_values,
            title='Test Plot',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_title(), 'Test Plot')

        plt.close(fig)

    def test_with_graph_context(self):
        """Test with graph context."""
        G = nx.Graph()
        G.add_edges_from([('Feature_A', 'Feature_B'), ('Feature_B', 'Feature_C')])

        fig, ax = plot_shapley_values(
            self.shapley_values,
            graph=G,
            layout='spring',
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_layouts(self):
        """Test different graph layouts."""
        G = nx.Graph()
        G.add_edges_from([('Feature_A', 'Feature_B'), ('Feature_B', 'Feature_C')])

        layouts = ['spring', 'circular', 'random']

        for layout in layouts:
            fig, ax = plot_shapley_values(
                self.shapley_values,
                graph=G,
                layout=layout,
                show_plot=False
            )

            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

    def test_customization_options(self):
        """Test customization options."""
        fig, ax = plot_shapley_values(
            self.shapley_values,
            figsize=(12, 8),
            node_size=1000,
            font_size=12,
            show_plot=False
        )

        self.assertEqual(fig.get_figwidth(), 12)
        self.assertEqual(fig.get_figheight(), 8)

        plt.close(fig)

    def test_without_graph(self):
        """Test plotting without graph (should create bar plot)."""
        fig, ax = plot_shapley_values(
            self.shapley_values,
            show_plot=False
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for visualization module."""

    def test_full_visualization_pipeline(self):
        """Test full visualization pipeline with real data."""
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(50, 4),
            columns=['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D']
        )

        # Simulate Shapley values
        shapley_values = {
            'Feature_A': 0.3,
            'Feature_B': -0.1,
            'Feature_C': 0.5,
            'Feature_D': 0.2
        }

        # Create graph
        G = nx.Graph()
        G.add_edges_from([
            ('Feature_A', 'Feature_B'),
            ('Feature_B', 'Feature_C'),
            ('Feature_C', 'Feature_D')
        ])

        # Test multiple visualization methods
        visualizer = FeatureImportanceVisualizer()

        # Bar plot
        fig1, ax1 = visualizer.plot_importance(shapley_values, show_plot=False)
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)

        # Network plot
        fig2, ax2 = visualizer.plot_network(
            G,
            node_values=shapley_values,
            show_plot=False
        )
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)

        # Legacy plot function
        fig3, ax3 = plot(shapley_values, show_plot=False)
        self.assertIsInstance(fig3, plt.Figure)
        plt.close(fig3)

        # Standalone function
        fig4, ax4 = plot_shapley_values(shapley_values, show_plot=False)
        self.assertIsInstance(fig4, plt.Figure)
        plt.close(fig4)

    def test_consistency_between_methods(self):
        """Test consistency between different plotting methods."""
        values = {'A': 0.5, 'B': -0.3, 'C': 0.8}

        # Legacy plot function
        fig1, ax1 = plot(values, show_plot=False)

        # New visualizer
        visualizer = FeatureImportanceVisualizer()
        fig2, ax2 = visualizer.plot_importance(values, show_plot=False)

        # Standalone function
        fig3, ax3 = plot_shapley_values(values, show_plot=False)

        # All should produce valid plots
        for fig in [fig1, fig2, fig3]:
            self.assertIsInstance(fig, plt.Figure)

        # Clean up
        for fig in [fig1, fig2, fig3]:
            plt.close(fig)

    def test_error_handling(self):
        """Test error handling in visualization functions."""
        # Test with invalid data types
        with self.assertRaises((TypeError, ValueError)):
            plot("invalid_data", show_plot=False)

        # Test with invalid graph
        visualizer = FeatureImportanceVisualizer()
        with self.assertRaises((TypeError, AttributeError)):
            visualizer.plot_network("not_a_graph", show_plot=False)

    def test_large_data_handling(self):
        """Test handling of large datasets."""
        # Create large shapley values dictionary
        large_values = {f'Feature_{i}': np.random.randn() for i in range(100)}

        # Should handle large data without errors
        fig, ax = plot(large_values, top_n=20, show_plot=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        visualizer = FeatureImportanceVisualizer()
        fig2, ax2 = visualizer.plot_importance(large_values, show_plot=False)
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_memory_management(self):
        """Test that plots don't cause memory leaks."""
        # Create and close many plots
        for i in range(10):
            values = {f'F_{j}': np.random.randn() for j in range(5)}
            fig, ax = plot(values, show_plot=False)
            plt.close(fig)

        # Should complete without memory issues
        self.assertTrue(True)

    @patch('matplotlib.pyplot.show')
    def test_show_plot_parameter(self, mock_show):
        """Test that show_plot parameter controls display."""
        values = {'A': 0.5, 'B': -0.3}

        # With show_plot=False, plt.show() should not be called
        plot(values, show_plot=False)
        mock_show.assert_not_called()

        # With show_plot=True, plt.show() should be called
        fig, ax = plot(values, show_plot=True)
        mock_show.assert_called_once()

        plt.close(fig)


if __name__ == '__main__':
    unittest.main()