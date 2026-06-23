"""Dataset loaders."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


def _load_housing():
    data = pd.read_csv(DATA_DIR / "housing_price.csv")
    X = data.drop(["MEDV"], axis=1)
    y = data["MEDV"]
    return X, y


def _load_h1n1():
    data = pd.read_csv(DATA_DIR / "process_data.csv")
    X = data.drop(["h1n1_vaccine", "respondent_id", "seasonal_vaccine"], axis=1)
    y = data["h1n1_vaccine"]
    return X, y


def _load_crimedata():
    data = pd.read_csv(DATA_DIR / "crimedata_processing.csv")
    X = data.drop(["ViolentCrimesPerPop", "nonViolPerPop"], axis=1)
    y = data["ViolentCrimesPerPop"]
    return X, y


def _load_mic():
    data = pd.read_csv(DATA_DIR / "MIC_processing.csv")
    X = data.drop(["LET_IS"], axis=1)
    y = data["LET_IS"]
    return X, y


DATASETS = {
    "housing": _load_housing,
    "h1n1": _load_h1n1,
    "crimedata": _load_crimedata,
    "mic": _load_mic,
}


def load_dataset(cfg):
    name = cfg["dataset"]
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )
    return DATASETS[name]()
