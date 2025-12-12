## SHAP Force Plot Example

Generate per-sample Shapley values and SHAP visualizations using ShapG's graph construction.

### Prerequisites

```bash
pip install lightgbm shap
```

Install ShapG (if not installed):

```bash
pip install -i https://test.pypi.org/simple/ shapG==0.13.8
```

Clone the `strategy_inputs` repo (contains `SamplerExplainer` for per-sample computation) into the current directory:

```bash
git clone https://github.com/vectorsss/strategy_inputs.git
```

### Quick Start

```bash
# Classification (H1N1 Vaccine) with improved graph
python shap_forceplot_persample.py --dataset h1n1 --metric model_output --use_improved_graph

# Regression (Boston Housing) with improved graph
python shap_forceplot_persample.py --dataset housing --metric model_output --use_improved_graph
```

### Options

| Flag | Values | Description |
|------|--------|-------------|
| `--dataset` | `housing`, `h1n1` | Dataset to use |
| `--metric` | `model_output`, `default` | `model_output` for force plots, `default` for error-based importance |
| `--use_improved_graph` | - | Use improved graph construction (faster & more accurate) |

### Output

- `shapg_{dataset}_{metric}_plots.png` - Bar plot + Beeswarm plot
- `shapg_{dataset}_{metric}_forceplot.html` - Interactive force plot (open in browser)
