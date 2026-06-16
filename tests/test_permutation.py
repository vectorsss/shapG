"""Tests for PermutationExplainer."""

import unittest
import numpy as np
from shapG import PermutationExplainer


class DummyModel:
    """Returns mean of unmasked features as prediction."""

    def predict(self, X):
        if X.ndim == 3:
            return X.mean(axis=(1, 2))
        return X.mean(axis=1)


def mse(y_true, y_pred):
    return -((y_true - y_pred) ** 2)


class TestPermutationExplainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.model = DummyModel()
        self.X = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 0.0],
            ]
        )
        self.y = self.X[:, 0]  # first feature drives target
        self.feature_names = ["f0", "f1", "f2"]

    def test_returns_dict_with_feature_names(self):
        exp = PermutationExplainer(
            self.model,
            mse,
            feature_names=self.feature_names,
            n_repeats=5,
            random_state=42,
        )
        result = exp.explain(self.X, self.y)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.feature_names))

    def test_returns_array_without_feature_names(self):
        exp = PermutationExplainer(self.model, mse, n_repeats=5, random_state=42)
        result = exp.explain(self.X, self.y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[-1], 3)

    def test_first_feature_most_important(self):
        exp = PermutationExplainer(
            self.model,
            mse,
            feature_names=self.feature_names,
            n_repeats=20,
            random_state=42,
        )
        result = exp.explain(self.X, self.y)
        importances = {k: np.mean(v) for k, v in result.items()}
        most_important = max(importances, key=importances.get)
        self.assertEqual(most_important, "f0")

    def test_3d_input(self):
        X3d = self.X[:, np.newaxis, :]  # (5, 1, 3)
        y3d = self.y
        exp = PermutationExplainer(
            self.model,
            mse,
            feature_names=self.feature_names,
            n_repeats=5,
            random_state=42,
        )
        result = exp.explain(X3d, y3d)
        self.assertIsInstance(result, dict)

    def test_invalid_ndim_raises(self):
        exp = PermutationExplainer(self.model, mse, n_repeats=2)
        with self.assertRaises(ValueError):
            exp.explain(np.ones((5, 3, 2, 1)), self.y)

    def test_global_metric(self):
        global_metric = lambda yt, yp: float(np.mean(-((yt - yp) ** 2)))
        exp = PermutationExplainer(
            self.model,
            global_metric,
            feature_names=self.feature_names,
            n_repeats=5,
            random_state=42,
            is_global_metric=True,
        )
        result = exp.explain(self.X, self.y)
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(float(v), float)


if __name__ == "__main__":
    unittest.main()
