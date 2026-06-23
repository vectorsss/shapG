# Experiments harness

Config-driven benchmark for the Shapley-value estimators in `shapG`. Each YAML
config under `configs/` fully declares one experiment (dataset, model, graph,
explainers, KPI settings); running it writes a self-contained, timestamped
folder under `results/` so a config never needs to be re-run to inspect its
output.

This harness is **separate from the installed package** —
`benchmark/__init__.py` puts the repo root on `sys.path` so it imports the
local `shapG`, not a pip-installed one.

> Markovian-process benchmarking is intentionally **not** part of this harness;
> that work lives in a separate repository.

## Run

```bash
cd experiments

# Run one config (writes results/<config>_<timestamp>/)
python run_benchmark.py configs/comparison_housing.yaml

# Run a subset of explainers
python run_benchmark.py configs/comparison_housing.yaml --only shapg cis

# Reuse a previous run: only recompute the --only methods, merge the rest
python run_benchmark.py configs/comparison_housing.yaml --resume results/<prev>/ --only qrcs
python run_benchmark.py configs/comparison_housing.yaml --resume results/<prev>/ --only   # replot only
python run_benchmark.py configs/comparison_housing.yaml --resume results/<prev>/ --tidy   # drop stale methods

# Batch over every config, logging each to results/logs/
bash run_all.sh

# Diff two runs (ranking overlap, KPI slope delta, speedup) — e.g. mask vs retrain
python compare_runs.py results/<run_A>/ results/<run_B>/ --top-k 5
```

## Configs

The `comparison_*` family compares the full estimator roster on one dataset:
ShapG, CIS, RandomCS, QR-CS / BlockQR-CS / Improved variants, the
StratifiedShapley sweep (`{shapley_weighted, leverage, leverage_bernoulli}` ×
`{direct, CS}`), LeverageSHAP, four Multilinear variants, and (housing/masking
only) Exact. Official `shap` KernelSHAP/SamplingSHAP baselines are available but
off by default — add an explainer entry and `pip install shap` to enable.

| Config | Dataset | v(S) mode |
|---|---|---|
| `comparison_housing.yaml` | housing (R²) | masking |
| `comparison_housing_retrain.yaml` | housing (R²) | retrain |
| `comparison_h1n1.yaml` | h1n1 (Accuracy) | masking |
| `comparison_h1n1_retrain.yaml` | h1n1 (Accuracy) | retrain |

Set `evaluation.adjusted_r2: true` for an adjusted-R² variant (regression only).
Set `evaluation.reference_curves: true` to also compute the forward-KPI
Greedy-Best / Greedy-Worst / Random oracle curves (off by default — each is an
O(n²) greedy search and very slow under retrain). Datasets `crimedata` and `mic`
are also wired in `benchmark/data.py`.

**Graph construction is pruned automatically:** only ShapG-style
`graph_dependent` explainers use the extra `rank_deletion` graphs; if no
graph-dependent explainer is in the run (e.g. `--only cis qrcs`), only the first
graph is built.

## Output layout (`results/<config>_<timestamp>/`)

| File | Contents |
|---|---|
| `config.yaml` | copy of the config that produced this run |
| `shapley_values.csv` | `method, node, shapley_value` (one row per method × feature) |
| `timing.json` | method → wall-clock seconds |
| `kpi_backward.json` / `.pdf` | feature-dropping KPI curve + weighted slope |
| `kpi_forward.json` / `.pdf` | feature-addition KPI curve (+ greedy/random reference curves) |
| `shapley_values.pdf`, `timing.pdf` | comparison plots |

## Adding an explainer

Register the class in `benchmark/explainers.py` (`EXPLAINER_CLASSES`), give it a
display label in `benchmark/labels.py`, then add an entry under `explainers:` in
a config. Graph-based methods use `graph_dependent: true` (run once per graph);
full-set estimators use `graph_dependent: false` (run once).
