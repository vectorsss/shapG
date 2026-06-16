# Examples

Short, self-contained examples for the ShapG modular API. Run any of them
from the repository root:

```bash
python examples/01_quickstart.py
```

| Script | What it shows |
| --- | --- |
| `01_quickstart.py` | Build a feature-correlation graph and estimate Shapley values with `ShapGExplainer`, then plot them. |
| `02_exact_vs_approx.py` | Compare `ShapGExplainer` against the exact `ExactExplainer` on the same graph. |
| `03_custom_characteristic.py` | Drive the computation with a custom `(coalition, context) -> float` value function via `CustomFunction`. |
| `04_model_feature_importance.py` | Explain a trained scikit-learn model with `ModelBasedCharacteristic` (feature masking, no retraining). |

Examples 01, 02 and 04 use `datasets/housing_price.csv`; example 03 uses a
built-in NetworkX graph. The `datasets/` folder at the repository root also
contains larger datasets (`process_data.csv`, `crimedata_processing.csv`,
`MIC_processing.csv`) you can swap in.

All examples depend only on the core install (`numpy`, `pandas`,
`networkx`, `matplotlib`, `scikit-learn`).

Each script starts with

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

so that `import shapG` resolves to the source in this repository rather
than a pip-installed copy. This lets you run the examples against your
local changes without reinstalling the package.
