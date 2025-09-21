"""
Benchmark script comparing multiple Shapley value computation methods using the NEW modular API.
Migrated from benchmark_qr.py to use the new ShapG architecture.

Methods compared:
- ShapGExplainer: Fast approximate computation using local search and sampling
- CISExplainer: Combined Imputation Score computation
- RandomCSExplainer: Random compressed sensing approach using Bernoulli matrices
- QRCSExplainer: QR-based compressed sensing for Shapley values
- BlockQRCSExplainer: Block QR-CS for high-dimensional problems with parallel computation
"""

import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import kendalltau, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb
import warnings
from typing import Callable, Optional, Tuple, Dict, Set
from scipy.optimize import minimize
from scipy.linalg import qr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))

# Import NEW API
from shapG import (
    ShapGExplainer,
    CISExplainer,
    RandomCSExplainer,
    QRCSExplainer,
    CustomFunction,
    GraphBuilder
)
from shapG.explainer import BlockQRCSExplainer
try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    print("gnuplot_style not found, using default style")

# ==============================================================================
# QR-CS Shapley Implementation with NEW API Integration
# ==============================================================================
# QRCSExplainer is now imported from shapG.explainer - no longer defined locally


# ==============================================================================
# Data readers (unchanged)
# ==============================================================================

def housing_data_reader(filename='./data/housing_price.csv'):
    data = pd.read_csv(filename)
    X = data.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return X, y

def h1n1_data_reader(filename='./data/process_data.csv'):
    data = pd.read_csv(filename)
    X = data.drop(['h1n1_vaccine', 'respondent_id', 'seasonal_vaccine'], axis=1)
    y = data['h1n1_vaccine']
    return X, y

# ==============================================================================
# Benchmark functions using NEW API
# ==============================================================================

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=10):
    """
    Plot comparison using new API structure.
    Same functionality as before but cleaner implementation.
    """
    # Define model specific parameters
    random_states = {
        lgb.LGBMClassifier: [10, 10],
        lgb.LGBMRegressor: [42, 42]
    }
    test_sizes = {
        lgb.LGBMClassifier: [0.2, 0.2],
        lgb.LGBMRegressor: [0.2, 0.3]
    }
    random_state = random_states.get(type(model), [42, 42])
    test_size = test_sizes.get(type(model), [0.2, 0.2])

    # Generate results file name
    model_name = type(model).__name__
    results_file = f"{model_name}_new_api_results.pkl"

    # Load or calculate results
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results for {model_name} (new API) from disk.")
    else:
        X, y = reader()
        results = {}

        # Calculate initial metric
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        initial_metric = r2_score(y_test, y_pred) if isinstance(model, lgb.LGBMRegressor) else accuracy_score(y_test, y_pred)

        # Process each ranking method
        for method, feature_order in feature_rankings.items():
            feature_order = [feat if isinstance(feat, str) else feat[0] for feat in feature_order]

            if limit:
                feature_order = feature_order[:limit]

            metrics = [initial_metric]
            features = [[]]
            deltas = []

            for i in range(1, len(feature_order) + 1):
                features_to_drop = feature_order[:i]
                missing_cols = [col for col in features_to_drop if col not in X.columns]
                if missing_cols:
                    print(f"Warning: Columns {missing_cols} not found in dataset. Skipping.")
                    continue

                reduced_X = X.drop(columns=features_to_drop)
                x_train, x_test, y_train, y_test = train_test_split(
                    reduced_X, y, test_size=test_size[1], random_state=random_state[1]
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                new_metric = r2_score(y_test, y_pred) if isinstance(model, lgb.LGBMRegressor) else accuracy_score(y_test, y_pred)
                deltas.append(metrics[-1] - new_metric)
                metrics.append(new_metric)
                features.append(features_to_drop)

            # Calculate weighted slope
            beta = 0.8
            weight = [beta**i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weight) if deltas else 0
            }

        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results for {model_name} (new API) to disk.")

    # Create plot (matching original benchmark)
    plt.figure(figsize=(12, 8))
    metric_name = "$R^2$" if isinstance(model, lgb.LGBMRegressor) else "Accuracy"

    for method, data in results.items():
        label = f'{method} $S$={data["Slope"]:.4f}'
        plt.plot(
            range(len(data['Metrics'])),
            data['Metrics'],
            label=label,
            alpha=0.6
        )

    plt.xlabel('Number of Features Dropped')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} after dropping features (New API - {model_name})')
    plt.legend()
    plt.grid()

    if filename:
        # Update filename to indicate new API
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_new_api{ext}"
        plt.savefig(new_filename, dpi=300)
        print(f"Saved plot to {new_filename}")

    return results


def benchmark_feature_importance(reader, model, filename=None, limit=10):
    """
    Benchmark feature importance using NEW modular API.

    This function demonstrates:
    - Using GraphBuilder for graph construction
    - Using different Explainer classes (ShapG, CIS, RandomCS, QR-CS, Block QR-CS)
    - Using CustomFunction for characteristic functions
    - Using the new visualization API

    Returns:
        tuple: (shapley_values, cis_values, random_cs_values, qrcs_values, block_qrcs_values, results)
    """
    X, y = reader()

    # Build graph to match original benchmark
    print("Building graph to match original benchmark...")
    builder = GraphBuilder()

    # Use the new API method that exactly matches the original
    G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
    print("Using original graph construction (kendalltau + minimal edge graph)")

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Define custom characteristic function for classification KPI
    def classification_kpi_wrapper(coalition: Set[int], context: nx.Graph) -> float:
        """Wrapper to work with new API signature."""
        if len(coalition) == 0:
            return 0

        # Coalition contains column names (strings) from graph nodes
        cols = list(coalition)

        # Select columns by name directly since graph nodes are column names
        X_subset = X[cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42
        )
        lgb_model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1, device='cpu')
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        return r2_score(y_test, y_pred)

    # Create custom characteristic function
    custom_char_func = CustomFunction(classification_kpi_wrapper, name="ClassificationKPI")

    # NEW API: Use ShapGExplainer
    print("\nComputing Shapley values using ShapGExplainer...")
    shapg_explainer = ShapGExplainer(
        characteristic_function=custom_char_func,
        depth=1,
        n_samples=3,  # equivalent to m=3 in old API
        approximate_by_ratio=False,
        scale=False,
        verbose=True
    )
    shapley_values = shapg_explainer.fit_explain(G)

    # Compute CIS values using NEW API
    print("\nComputing CIS values...")
    cis_explainer = CISExplainer(
        characteristic_function=custom_char_func,
        verbose=False
    )
    cis_values = cis_explainer.fit_explain(G)

    # Compute Random CS Shapley values using NEW API
    # RandomCSExplainer uses random Bernoulli matrices and iterative sampling
    # to approximate Shapley values using compressed sensing
    print("\nComputing Random CS Shapley values...")
    random_cs_explainer = RandomCSExplainer(
        characteristic_function=custom_char_func,
        m=50,  # Number of measurements per iteration
        t=30,  # Number of iterations
        verbose=True,
        seed=42  # For reproducibility
    )
    random_cs_values = random_cs_explainer.fit_explain(G)

    # Compute QR-CS Shapley values
    print("\nComputing QR-CS Shapley values...")
    print(f"Using CVXPY for L1 minimization (much faster than scipy)")
    n_measurements = min(100, 2**(len(X.columns)-1) // 4)
    qrcs_explainer = QRCSExplainer(
        characteristic_function=custom_char_func,
        n_measurements=n_measurements,
        use_fast_fallback=False,  # Now this should be fast with CVXPY
        verbose=True
    )
    qrcs_values = qrcs_explainer.fit_explain(G)

    # Compute Block QR-CS Shapley values
    print("\nComputing Block QR-CS Shapley values...")
    print(f"Using parallel block computation for scalability")
    # Use fewer blocks for small problems, more for larger ones
    n_blocks = min(3, max(2, len(X.columns) // 5))
    block_qrcs_explainer = BlockQRCSExplainer(
        characteristic_function=custom_char_func,
        n_blocks=n_blocks,
        parallel=True,  # Enable parallel computation
        use_fast_fallback=True,  # Use fast mode for speed
        verbose=True
    )
    block_qrcs_values = block_qrcs_explainer.fit_explain(G)

    # Convert to sorted feature lists for plotting
    feature_rankings = {}

    # Helper function to convert node indices to feature names
    def node_to_feature_name(node, columns):
        try:
            idx = int(node)
            if 0 <= idx < len(columns):
                return columns[idx]
        except (ValueError, TypeError):
            if node in columns:
                return node
        return None

    # Add ShapG values
    sorted_shapley = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['ShapG'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_shapley
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add CIS values
    sorted_cis = sorted(cis_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['CIS'] = [
        X.columns[node] if isinstance(node, int) and 0 <= node < len(X.columns) else str(node)
        for node, _ in sorted_cis
    ]

    # Add Random CS values
    sorted_random_cs = sorted(random_cs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['RandomCS'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_random_cs
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add QR-CS values
    sorted_qrcs = sorted(qrcs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['QR-CS'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_qrcs
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add Block QR-CS values
    sorted_block_qrcs = sorted(block_qrcs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['BlockQRCS'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_block_qrcs
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add model feature importances (if model is provided and trained)
    if model and hasattr(model, 'feature_importances_'):
        # Train model if not already trained
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[::-1]
        feature_rankings['Model'] = [X.columns[i] for i in feature_indices]

    # Plot comparison using the original plotting function
    results = plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename, limit)

    # Return results including all CS variants
    return shapley_values, cis_values, random_cs_values, qrcs_values, block_qrcs_values, results


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK WITH NEW MODULAR API")
    print("=" * 60)

    # Example usage with new API
    model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)

    print("\nRunning benchmark with housing data...")
    shapley_values, cis_values, random_cs_values, qrcs_values, block_qrcs_values, results = benchmark_feature_importance(
        housing_data_reader,
        model,
        filename='housing_benchmark_with_qr.png',
        limit=10
    )

    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)
    print("Random CS values:", random_cs_values)
    print("QR-CS values:", qrcs_values)
    print("Block QR-CS values:", block_qrcs_values)
