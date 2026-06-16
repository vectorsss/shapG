"""Explain a trained model's predictions with graph-based Shapley values.

ModelBasedCharacteristic scores a coalition by keeping its features and
imputing the rest (no retraining), so any fitted scikit-learn-style model
works. Combined with ShapGExplainer, importance is attributed using only
each feature's local neighborhood in the correlation graph.

Run:
    python examples/04_model_feature_importance.py
"""

import os
import sys
from pathlib import Path

# Use the shapG source in this repository, not a pip-installed copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapG import ShapGExplainer, GraphBuilder
from shapG.characteristic import ModelBasedCharacteristic

DATA = Path(__file__).resolve().parent.parent / "datasets" / "housing_price.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    feature_names = [c for c in df.columns if c != "MEDV"]
    X = df[feature_names].to_numpy()
    y = df["MEDV"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Coalition value = model R^2 when only the coalition's features are kept.
    # Node ids are integer column indices, matching ModelBasedCharacteristic.
    char_func = ModelBasedCharacteristic(
        model=model, X=X_test, y=y_test, masking_strategy="mean"
    )

    # Build the graph from feature correlations -> integer node labels 0..n-1.
    graph = GraphBuilder().from_correlation(X_train, threshold=0.3)
    explainer = ShapGExplainer(characteristic_function=char_func, depth=2, n_samples=8)
    shapley_values = explainer.fit_explain(graph)

    print("Feature importance (Shapley attribution of R^2):")
    for idx, value in sorted(
        shapley_values.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {feature_names[idx]:>8}: {value:+.4f}")


if __name__ == "__main__":
    main()
