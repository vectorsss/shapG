import unittest
import time
import networkx as nx
import sys
import os
import numpy as np
from tabulate import tabulate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shapG.shapley import shapG, shapley_value, graph_generator

class TestShapleyValue(unittest.TestCase):
    """Test case for ShapG module functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test graph once before all tests."""
        print("Setting up test graph...")
        cls.G = graph_generator(10, 0.5, seed=42)  # Fixed seed for reproducibility
        cls.shapley_values = {}
        cls.shapley_values_G = {}
    
    def test_01_graph_generator(self):
        """Test graph generation."""
        G = graph_generator(20, 0.3, seed=123)
        self.assertEqual(G.number_of_nodes(), 20)
        # Check if density is approximately as expected
        max_edges = 20 * 19 // 2
        expected_edges = int(0.3 * max_edges)
        self.assertAlmostEqual(G.number_of_edges() / max_edges, 0.3, delta=0.1)

    def test_02_shapley_value(self):
        """Test exact Shapley value calculation."""
        print("\nExact Shapley value calculation starting...")
        start = time.time()
        TestShapleyValue.shapley_values = shapley_value(TestShapleyValue.G)
        end = time.time()
        
        print(f"Exact Shapley value calculation time: {end - start:.3f} seconds")
        
        # Verify results structure
        self.assertIsInstance(TestShapleyValue.shapley_values, dict)
        self.assertEqual(len(TestShapleyValue.shapley_values), TestShapleyValue.G.number_of_nodes())
        
        # Verify efficiency property (sum of values equals characteristic function of entire graph)
        from shapG.shapley import coalition_degree
        total_value = sum(TestShapleyValue.shapley_values.values())
        full_coalition_value = coalition_degree(TestShapleyValue.G, set(TestShapleyValue.G.nodes()))
        self.assertAlmostEqual(total_value, full_coalition_value, places=6)

    def test_03_shapG(self):
        """Test approximate Shapley value calculation."""
        print("\nShapG approximate calculation starting...")
        start = time.time()
        TestShapleyValue.shapley_values_G = shapG(
            TestShapleyValue.G, 
            depth=1,
            approximate_by_ratio=True,
            verbose=True
        )
        end = time.time()
        
        print(f"ShapG approximate calculation time: {end - start:.3f} seconds")
        
        # Verify results structure
        self.assertIsInstance(TestShapleyValue.shapley_values_G, dict)
        self.assertEqual(len(TestShapleyValue.shapley_values_G), TestShapleyValue.G.number_of_nodes())

    def test_04_compare_results(self):
        """Compare exact and approximate results."""
        if not TestShapleyValue.shapley_values or not TestShapleyValue.shapley_values_G:
            self.skipTest("Previous tests didn't produce Shapley values")
        
        exact = TestShapleyValue.shapley_values
        approx = TestShapleyValue.shapley_values_G
        
        # Calculate correlation between exact and approximate results
        exact_values = np.array([exact[node] for node in sorted(exact.keys())])
        approx_values = np.array([approx[node] for node in sorted(approx.keys())])
        correlation = np.corrcoef(exact_values, approx_values)[0, 1]
        
        # Print comparison table
        print("\nComparing Exact vs. Approximate Shapley values:")
        table_data = []
        for node in sorted(exact.keys()):
            sv = exact[node]
            asv = approx.get(node, 'N/A')
            diff = asv - sv if asv != 'N/A' else 'N/A'
            rel_diff = (diff / sv * 100) if asv != 'N/A' and sv != 0 else 'N/A'
            if rel_diff != 'N/A':
                rel_diff = f"{rel_diff:.2f}%"
            table_data.append([node, f"{sv:.4f}", f"{asv:.4f}", f"{diff:.2e}", rel_diff])
        
        headers = ["Node", "Exact SV", "Approx SV", "Difference", "Rel. Diff"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nCorrelation between exact and approximate values: {correlation:.4f}")
        
        # The approximation should be reasonably close to exact values
        self.assertGreater(correlation, 0.8, "Correlation between exact and approximate values is too low")

if __name__ == '__main__':
    unittest.main()