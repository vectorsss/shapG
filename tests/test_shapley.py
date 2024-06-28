import unittest
import time
import networkx as nx
from shapG.shapley import shapG, shapley_value, graph_generator

class TestShapleyValue(unittest.TestCase):
    shapley_values = {}
    shapley_values_G = {}
    @classmethod
    def setUp(cls):
        cls.G = graph_generator(10, 0.5)

    def test_shapley_value_degree_local(self):
        print("Shapley value start running...")
        start = time.time()
        TestShapleyValue.shapley_values = shapley_value(TestShapleyValue.G)
        end = time.time()
        print("Shapley value time: ", end - start)
        self.assertIsInstance(TestShapleyValue.shapley_values, dict)

    def test_shapG(self):
        print("ShapG start running...")
        start = time.time()
        TestShapleyValue.shapley_values_G = shapG(TestShapleyValue.G, approximate_by_ratio=False)
        end = time.time()
        print("ShapG running time: ", end - start)
        self.assertIsInstance(TestShapleyValue.shapley_values_G, dict)
    
    @classmethod
    def tearDown(cls):
        if cls.shapley_values and cls.shapley_values_G:
            print("\nComparing Shapley values...")
            print(f"{'Node':>5} | {'SV':>10} | {'ASV':>10} | {'diff':>10}")
            print("-" * 45)
            for node in cls.shapley_values:
                sv = cls.shapley_values[node]
                asv = cls.shapley_values_G.get(node, 'N/A')
                diff = 'N/A' if asv == 'N/A' else asv - sv
                print(f"{node:>5} | {sv:>10.4f} | {asv:>10.4f} | {diff:.2e}")



if __name__ == '__main__':
    unittest.main()
