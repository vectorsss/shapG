"""
Per-sample Shapley values using ShapG graph + SamplerExplainer

Usage:
    python shap_forceplot_persample.py --dataset h1n1 --metric model_output --use_improved_graph 
    python shap_forceplot_persample.py --dataset housing --metric model_output --use_improved_graph 
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Suppress sklearn warning about feature names (harmless)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import shap

# Add paths for local imports
# if you have installed the ShapG package, you can skip the next line
# if no, please consider installing it via:
# pip install -i https://test.pypi.org/simple/ shapG==0.13.8
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))
from shapG import GraphBuilder

# please clone the strategy_inputs repo into the current directory
# because the currently ShapG package does not include the per sample explainer like SHAP library
# we have implemented the SamplerExplainer in the strategy_inputs repo
# git clone https://github.com/vectorsss/strategy_inputs.git
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), 'strategy_inputs', 'paper')))

from explainers.sampler import SamplerExplainer


# =============================================================================
# Metric Functions
# =============================================================================

def model_output_metric(y_true, y_pred):
    """Use model predictions directly (for force plots)."""
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] > 1:
        return y_pred.mean(axis=1, keepdims=True)
    return y_pred


def default_regression_metric(y_true, y_pred):
    """Negative MSE (per-sample). Higher is better."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return -((y_true - y_pred) ** 2)


def default_classification_metric(y_true, y_pred):
    """Correctness: 1 for correct, -1 for incorrect."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.min() >= 0 and y_pred.max() <= 1 and not np.all(np.isin(y_pred, [0, 1])):
        y_pred_class = (y_pred > 0.5).astype(float) if y_pred.shape[1] == 1 else np.argmax(y_pred, axis=1).reshape(-1, 1)
    else:
        y_pred_class = np.round(y_pred)
    correct = (y_true == y_pred_class).astype(float)
    return 2 * correct - 1


def main(dataset="housing", metric="model_output", use_improved_graph=False):
    """
    Main function.

    Args:
        dataset: "housing" (regression) or "h1n1" (classification)
        metric: "model_output" or "default"
        use_improved_graph: Use improved graph construction method
    """
    is_classification = (dataset == "h1n1")

    print("=" * 80)
    print(f"Dataset: {dataset} | Metric: {metric} | Improved Graph: {use_improved_graph}")
    print("=" * 80)

    # Load data
    if dataset == "housing":
        df = pd.read_csv("data/housing_price.csv")
        X = df.drop(columns=["MEDV"])
        y = df["MEDV"].values
    else:  # h1n1
        df = pd.read_csv("data/process_data.csv")
        X = df.drop(columns=["respondent_id", "h1n1_vaccine", "seasonal_vaccine"])
        y = df["h1n1_vaccine"].values
        X = X.iloc[:500]
        y = y[:500]

    print(f"\nData shape: {X.shape}")

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_classification:
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        print(f"Accuracy: {model.score(X_test, y_test):.3f}")
    else:
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        print(f"R² score: {model.score(X_test, y_test):.3f}")

    # Build graph
    builder = GraphBuilder()
    # this is the improved graph construction method compare to the original ShapG paper
    # in the most case, it's faster and more accurate
    # Complete-to-Sparse: A Novel Graph Construction Strategy for Efficient ShapG
    if use_improved_graph:
        G = builder.from_rank_deletion(
            X, y,
            density_ratio=None,
            correlation_method='cosine',
            similarity_method='cosine'
        )
        print(f"Graph (improved): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        # original ShapG graph construction method
        G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
        print(f"Graph (original): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Select metric
    if metric == "model_output":
        metric_func = model_output_metric
    else:
        metric_func = default_classification_metric if is_classification else default_regression_metric

    # Compute Shapley values
    collaborator_map = SamplerExplainer.graph_to_collaborator_map(G)
    explainer = SamplerExplainer(
        model=model,
        metric=metric_func,
        collaborator_map=collaborator_map,
        feature_names=X.columns.tolist(),
        mask_value=X_train.mean().values,
        is_global_metric=False,
        max_collaborator_subset=5,
    )
    shapley_dict = explainer.explain(X_test, y_test, verbose=True)

    # Convert to matrix
    shapley_matrix = np.zeros((len(X_test), len(X.columns)))
    for feat_name, values in shapley_dict.items():
        feat_idx = X.columns.get_loc(feat_name)
        shapley_matrix[:, feat_idx] = values.flatten()

    # Global importance
    if metric == "model_output":
        global_importance = np.mean(np.abs(shapley_matrix), axis=0)
        print("\nFeature Importance (mean |SHAP|):")
    else:
        global_importance = np.mean(shapley_matrix, axis=0)
        print("\nFeature Importance (mean SHAP):")

    for idx in np.argsort(np.abs(global_importance))[::-1][:10]:
        print(f"  {X.columns[idx]:25s}: {global_importance[idx]:.4f}")

    # Visualize
    base_value = model.predict(X_train).mean()
    explanation = shap.Explanation(
        values=shapley_matrix,
        base_values=np.full(len(X_test), base_value),
        data=X_test.values,
        feature_names=X.columns.tolist()
    )

    # Plots
    graph_suffix = "_improved" if use_improved_graph else ""
    output_prefix = f"shapg_{dataset}_{metric}{graph_suffix}"

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plt.sca(axes[0])
    shap.plots.bar(explanation, show=False)
    axes[0].set_title(f"{dataset} - Feature Importance ({metric})", fontsize=14, fontweight='bold')
    plt.sca(axes[1])
    shap.plots.beeswarm(explanation, show=False)
    axes[1].set_title(f"{dataset} - Beeswarm Plot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plots.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_prefix}_plots.png")
    #plt.show()

    # Force plot
    force_plot = shap.plots.force(base_value, shapley_matrix, feature_names=X.columns.tolist(), matplotlib=False)
    shap.save_html(f"{output_prefix}_forceplot.html", force_plot)
    print(f"✓ Saved: {output_prefix}_forceplot.html")

    return shapley_matrix, explanation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="housing", choices=["housing", "h1n1"])
    parser.add_argument("--metric", type=str, default="model_output", choices=["model_output", "default"])
    parser.add_argument("--use_improved_graph", action="store_true")
    args = parser.parse_args()

    main(dataset=args.dataset, metric=args.metric, use_improved_graph=args.use_improved_graph)
