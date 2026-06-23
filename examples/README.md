# Examples

Small, self-contained scripts that demonstrate the `shapG` API. Run them from
the repository root:

```bash
python examples/quickstart_shapg.py        # ShapG on a feature-similarity graph
python examples/quickstart_estimators.py   # ShapG vs. a QR-CS estimator (model-based v(S))
```

## Data

The raw CSVs used by both the examples and the benchmark harness live in the
repo-root [`../datasets/`](../datasets/) directory:

| File | Dataset | Target |
|---|---|---|
| `housing_price.csv` | Boston housing (regression) | `MEDV` |
| `process_data.csv` | H1N1 vaccine (classification) | `h1n1_vaccine` |
| `crimedata_processing.csv` | Communities & crime (regression) | `ViolentCrimesPerPop` |
| `MIC_processing.csv` | MIC (classification) | `LET_IS` |

## Full benchmarks

The heavyweight, config-driven comparisons (every estimator, KPI evaluation,
plots, resumable runs) live in [`../experiments/`](../experiments/README.md),
not here.
