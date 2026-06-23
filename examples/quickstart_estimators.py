"""Minimal quickstart: compare ShapG against a compressed-sensing estimator.

    python examples/quickstart_estimators.py

Both explainers score the same coalition (characteristic) function over the
same feature graph, so their per-feature values are directly comparable. For
the full estimator roster and KPI evaluation see ../experiments/.
"""

from pathlib import Path

import pandas as pd

from shapG import GraphBuilder, QRCSExplainer, ShapGExplainer
from shapG.characteristic import GraphModelCharacteristic

import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "datasets" / "housing_price.csv"


def main():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=["MEDV"]), df["MEDV"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Score a coalition by masking out-of-coalition features on a fixed model.
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.3, verbosity=-1)
    model.fit(X_tr, y_tr)
    char = GraphModelCharacteristic(
        model=model,
        X=X_te,
        y=y_te.values,
        masking_strategy="mean",
        baseline=X_tr.mean().values,
    )

    G = GraphBuilder().from_kendalltau_minimal_edge(X, reverse=True, version="v3")
    shapg = ShapGExplainer(
        characteristic_function=char, depth=1, n_samples=3
    ).fit_explain(G)
    qrcs = QRCSExplainer(characteristic_function=char).fit_explain(G)

    print(f"{'feature':12s}{'ShapG':>10s}{'QR-CS':>10s}")
    for node in sorted(shapg, key=lambda n: shapg[n], reverse=True):
        name = X.columns[int(node)] if str(node).isdigit() else node
        print(f"  {str(name):10s}{shapg[node]:>10.4f}{qrcs.get(node, 0.0):>10.4f}")


if __name__ == "__main__":
    main()
