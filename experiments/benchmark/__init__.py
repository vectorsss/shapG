"""Benchmark modules for Shapley value experiments."""

import sys
from pathlib import Path

# Ensure we import shapG from this repository, not from pip
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .data import load_dataset
from .model import build_model, build_char_func, get_metric_fn
from .graph import build_graphs
from .explainers import (
    run_explainers,
    to_feature_ranking,
    node_to_feature,
)
from .kpi import compute_kpi, compute_forward_kpi
from .io import load_config, load_previous_results, save_results
from .plotting import (
    plot_timing,
    plot_backward_kpi,
    plot_forward_kpi,
    plot_shapley_values,
)
