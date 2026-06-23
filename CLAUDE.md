# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ShapG is a Python library for scalable Shapley-value computation on **graph-structured feature interactions**. The core idea: build a graph where nodes are features and edges encode feature similarity/correlation, then compute Shapley values by only considering each node's local neighborhood (reachable nodes up to `depth`), making large-feature problems tractable. The repo also contains a research benchmark harness (`experiments/`) used to produce the accompanying paper's results.

## Common commands

```bash
# Install for development (editable)
pip install -e ".[dev]"

# Run the full test suite (config in pyproject.toml: testpaths=["tests"])
pytest

# Run one test file / one test
pytest tests/test_shapley.py
pytest tests/test_shapley.py::TestClassName::test_method

# Skip slow tests
pytest -m "not slow"

# Coverage
pytest --cov

# Lint / format / type-check (dev deps)
black .          # line-length 100 (pyproject); NOTE pre-commit uses 88 — see caveat below
isort .          # profile=black, line-length 100
flake8 .
mypy shapG

# Build distribution
python -m build   # or: python setup.py sdist bdist_wheel
```

**Formatting caveat:** `pyproject.toml` sets black `line-length = 100`, but `.pre-commit-config.yaml` runs black with `--line-length=88`. These disagree — match whichever gate you're targeting (pre-commit is what runs on commit).

## Package architecture (`shapG/`)

The public API is re-exported from `shapG/__init__.py`. Four subpackages plus a legacy shim:

- **`explainer/`** — the algorithms. All inherit from `Explainer` / `GraphExplainer` in `base.py`, which defines the `fit()` / `explain()` / `fit_explain()` contract and `_parse_input()` (accepts `int`, `nx.Graph`, `pd.DataFrame`, `np.ndarray`, or `list` and normalizes to `(n_players, player_ids, context)`). Key explainers:
  - `ExactExplainer` — O(2^n) enumeration; only for small problems.
  - `ShapGExplainer` — the headline algorithm (`shapg.py`). Per node, gathers `depth`-reachable neighbors; if the neighborhood is smaller than `n_samples` (`m`) it enumerates exactly, otherwise it Monte-Carlo samples. Has a scalar path and a `_explain_per_sample()` batch path (triggered when the characteristic function's `batch_compute` returns a 2-D array — one value per (coalition, data-sample)).
  - `MarkovianExplainer` — greedy coalition-formation process (Faigle & Grabisch 2012); O(n²) evaluations, supports `allow_leave` to escape local optima.
  - Sampling/estimator variants: `CISExplainer`, `RandomCSExplainer`, `QRCSExplainer`, `BlockQRCSExplainer`, `ImprovedQRCSExplainer`, `ImprovedBlockQRCSExplainer`, `LeverageScoreExplainer`, `MultilinearExplainer`, `StratifiedShapleyExplainer`, `PermutationExplainer`.
- **`characteristic/`** — the value/coalition functions. `CharacteristicFunction` (in `explainer/base.py`) is the abstract base; subclasses implement `__call__(coalition, context)` and optionally `batch_compute()`. `characteristic_functions.py` has graph-only functions (`CoalitionDegree` — the default, `NodeCount`, `WeightedSum`, `CustomFunction`, `CenterOfImputationSet`). `model_based.py` has ML-model functions that score a coalition by masking out-of-coalition features on a *pre-trained* model without retraining (`ModelBasedCharacteristic`, `GraphModelCharacteristic`, `EnsembleMaskingCharacteristic`, `BatchedModelCharacteristic`).
- **`utils/`** — `GraphBuilder` (in `graph_construction.py`) constructs feature graphs via `from_correlation`, `from_mutual_information`, `from_kendalltau_minimal_edge`, `from_matrix_generator`, `from_rank_deletion`, etc.; `CoalitionManager` handles neighbor-coalition sampling/caching. `graph_helpers.py` has `get_reachable_nodes_at_depth` (the depth-limited BFS at the heart of ShapG). `utils.py` has matrix builders (`corr_generator`, `kl_mi_matrix`, `create_minimal_edge_graph`). `feature_similarity.py` handles mixed categorical/continuous similarity (`cramers_v`).
- **`visualization/`** — `FeatureImportanceVisualizer` and `plot_shapley_values` / `plot`.
- **`__legacy/compat.py`** — backward-compatible procedural functions (`shapG`, `shapley_value`, `cis`, `graph_generator`, `plot`). **Deprecated, slated for removal in v0.15.0**; they emit `DeprecationWarning` and delegate to the modular API. Don't build new functionality here.

### Conventions when adding an explainer
Subclass `GraphExplainer` (or `Explainer` for non-graph methods), implement `fit()` and `explain()`, set `self._fitted = True` in `fit()`, default the characteristic function to `CoalitionDegree()`, and register the class in both `shapG/explainer/__init__.py` and the top-level `shapG/__init__.py` `__all__`.

## Experiments harness (`experiments/`)

Config-driven research benchmark, **separate from the library** and not part of the installed package. `experiments/benchmark/__init__.py` deliberately inserts the repo root on `sys.path` so it imports the local `shapG` rather than a pip-installed one.

```bash
cd experiments
python run_benchmark.py configs/comparison_housing.yaml
python run_benchmark.py configs/<cfg>.yaml --only shapg cis            # run a subset
python run_benchmark.py configs/<cfg>.yaml --resume results/<prev>/    # reuse prior results
python run_benchmark.py configs/<cfg>.yaml --resume results/<prev>/ --tidy  # drop stale labels
python compare_runs.py results/<run_A>/ results/<run_B>/              # diff two runs (e.g. mask vs retrain)
bash run_all.sh                                                       # batch over configs
```

Each YAML config declares a `dataset`, a `model` (e.g. `LGBMRegressor`), one or more `graphs` (GraphBuilder method + params), and `explainers` (class + params). Datasets are loaded by `experiments/benchmark/data.py` from CSVs in the repo-root `datasets/` directory — supported keys: `housing`, `h1n1`, `crimedata`, `mic`. The `benchmark/` submodules split responsibilities: `data` / `model` / `graph` / `explainers` (estimator registry `EXPLAINER_CLASSES`) / `kpi` (forward & backward KPI) / `io` (config + result persistence) / `plotting` / `labels` (plot labels) / `shap_lib` (optional official-`shap` baselines, guarded import). There is one config family, `comparison_*` (compare every estimator on a dataset), with `_retrain` variants (retrain model per coalition vs. masking) and an optional `evaluation.adjusted_r2: true` toggle. **Markovian-process benchmarking is intentionally out of scope here** — that work lives in a separate repository, so the harness registers no `MarkovianExplainer` and carries no Markovian-specific plots.

## Notes

- `numba`/`ray`/`joblib` (the `performance` extra) and `lightgbm`/`seaborn` (the `examples` extra) are optional — guard imports accordingly.
- Supports Python 3.8+; avoid syntax/stdlib newer than 3.8.
