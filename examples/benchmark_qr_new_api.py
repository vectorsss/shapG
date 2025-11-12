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
import time
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
from shapG.explainer import (
    BlockQRCSExplainer,
    ImprovedQRCSExplainer,
    ImprovedBlockQRCSExplainer,
    LeverageScoreExplainer
)
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

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=10, cached_kpi_results=None):
    """
    Plot comparison using new API structure.
    Same functionality as before but cleaner implementation.

    Parameters:
    - cached_kpi_results: Pre-computed KPI results to avoid recalculation
    """
    # Get model name (needed for plot title regardless of cache status)
    model_name = type(model).__name__

    # If cached results provided, use them directly
    if cached_kpi_results is not None:
        results = cached_kpi_results
        print(f"Using cached KPI results.")
    else:
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

        print(f"Computing KPI results (no cache)...")
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


def plot_time_comparison(time_results, filename=None):
    """
    Plot time comparison for all algorithms.

    Parameters:
    - time_results: Dictionary mapping algorithm names to execution times in seconds
    - filename: Output filename for the plot
    """
    # Sort by time for better visualization
    sorted_items = sorted(time_results.items(), key=lambda x: x[1])
    algorithms = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]

    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(algorithms)))
    bars = plt.barh(algorithms, times, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        plt.text(time_val, i, f' {time_val:.2f}s',
                va='center', ha='left', fontweight='bold', fontsize=10)

    plt.xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title('Algorithm Execution Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if filename:
        base, ext = os.path.splitext(filename)
        time_filename = f"{base}_time_comparison{ext}"
        plt.savefig(time_filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved time comparison plot to {time_filename}")

    return sorted_items


def benchmark_feature_importance(reader, model, filename=None, limit=10, test_improved_graphs=True, use_cache=True):
    """
    Benchmark feature importance using NEW modular API.

    This function demonstrates:
    - Using GraphBuilder for graph construction
    - Using different Explainer classes (ShapG, CIS, RandomCS, QR-CS, Block QR-CS, Improved QR-CS)
    - Using CustomFunction for characteristic functions
    - Using the new visualization API

    Parameters:
    - reader: Function to read the dataset
    - model: Machine learning model
    - filename: Output filename for plot
    - limit: Number of top features to consider
    - test_improved_graphs: If True, also test improved graph construction methods
    - use_cache: If True, load cached results if available and save new results

    Returns:
        tuple: (shapley_values, cis_values, random_cs_values, qrcs_values, block_qrcs_values,
                improved_qrcs_values, leverage_shap_values, time_results, results)
    """
    X, y = reader()

    # Generate cache filename based on dataset
    dataset_name = reader.__name__.replace('_data_reader', '')
    cache_file = f"{dataset_name}_shapley_cache.pkl"

    # Try to load cached results
    if use_cache and os.path.exists(cache_file):
        print(f"\n" + "=" * 60)
        print(f"Loading all cached results from {cache_file}...")
        print("=" * 60)
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        shapley_values = cached_data['shapley_values']
        cis_values = cached_data['cis_values']
        random_cs_values = cached_data['random_cs_values']
        qrcs_values = cached_data['qrcs_values']
        block_qrcs_values = cached_data['block_qrcs_values']
        improved_qrcs_values = cached_data['improved_qrcs_values']
        improved_block_qrcs_values = cached_data['improved_block_qrcs_values']
        leverage_shap_values = cached_data.get('leverage_shap_values', {})
        improved_shapley_values = cached_data.get('improved_shapley_values', {})
        time_results = cached_data['time_results']
        cached_kpi_results = cached_data.get('kpi_results', None)

        print(f"Successfully loaded cached results!")
        print(f"  - Shapley methods: {len(time_results)}")
        print(f"  - KPI results: {'Yes' if cached_kpi_results else 'No'}")
        print(f"  - Example time (ShapG): {time_results.get('ShapG', 0):.2f}s")
    else:
        # Dictionary to store execution times
        time_results = {}
        leverage_shap_values = {}
        improved_shapley_values = {}
        cached_kpi_results = None

    # Build graph to match original benchmark (always needed for feature rankings)
    print("\nBuilding graph to match original benchmark...")
    builder = GraphBuilder()

    # Use the new API method that exactly matches the original
    G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
    print("Using original graph construction (kendalltau + minimal edge graph)")

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Only compute Shapley values if not cached
    if not (use_cache and os.path.exists(cache_file)):
        print("\n" + "=" * 60)
        print("Computing all Shapley values (no cache found)...")
        print("=" * 60)

        # Define custom characteristic function for classification KPI
        def characteristic_function_wrapper(coalition: Set[int], context: nx.Graph) -> float:
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
        custom_char_func = CustomFunction(characteristic_function_wrapper, name="Regression R2")

        # NEW API: Use ShapGExplainer
        print("\nComputing Shapley values using ShapGExplainer...")
        start_time = time.time()
        shapg_explainer = ShapGExplainer(
            characteristic_function=custom_char_func,
            depth=1,
            n_samples=3,  # equivalent to m=3 in old API
            approximate_by_ratio=False,
            scale=False,
            verbose=True
        )
        shapley_values = shapg_explainer.fit_explain(G)
        time_results['ShapG'] = time.time() - start_time
        print(f"  Time: {time_results['ShapG']:.2f}s")

        # Compute CIS values using NEW API
        print("\nComputing CIS values...")
        start_time = time.time()
        cis_explainer = CISExplainer(
            characteristic_function=custom_char_func,
            verbose=False
        )
        cis_values = cis_explainer.fit_explain(G)
        time_results['CIS'] = time.time() - start_time
        print(f"  Time: {time_results['CIS']:.2f}s")

        # Compute Random CS Shapley values using NEW API
        # RandomCSExplainer uses random Bernoulli matrices and iterative sampling
        # to approximate Shapley values using compressed sensing
        print("\nComputing Random CS Shapley values...")
        start_time = time.time()
        random_cs_explainer = RandomCSExplainer(
            characteristic_function=custom_char_func,
            m=50,  # Number of measurements per iteration
            t=30,  # Number of iterations
            verbose=True,
            seed=42  # For reproducibility
        )
        random_cs_values = random_cs_explainer.fit_explain(G)
        time_results['RandomCS'] = time.time() - start_time
        print(f"  Time: {time_results['RandomCS']:.2f}s")

        # Compute QR-CS Shapley values
        print("\nComputing QR-CS Shapley values...")
        print(f"Using CVXPY for L1 minimization (much faster than scipy)")
        start_time = time.time()
        n_measurements = min(100, 2**(len(X.columns)-1) // 4)
        qrcs_explainer = QRCSExplainer(
            characteristic_function=custom_char_func,
            n_measurements=n_measurements,
            use_fast_fallback=False,  # Now this should be fast with CVXPY
            verbose=True
        )
        qrcs_values = qrcs_explainer.fit_explain(G)
        time_results['QR-CS'] = time.time() - start_time
        print(f"  Time: {time_results['QR-CS']:.2f}s")

        # Compute Block QR-CS Shapley values
        print("\nComputing Block QR-CS Shapley values...")
        print(f"Using parallel block computation for scalability")
        start_time = time.time()
        # Use fewer blocks for small problems, more for larger ones
        n_blocks = min(3, max(2, len(X.columns) // 5))
        block_qrcs_explainer = BlockQRCSExplainer(
            characteristic_function=custom_char_func,
            n_blocks=n_blocks,
            parallel=False,  # Enable parallel computation
            use_fast_fallback=False,  # Use fast mode for speed
            verbose=True
        )
        block_qrcs_values = block_qrcs_explainer.fit_explain(G)
        time_results['BlockQRCS'] = time.time() - start_time
        print(f"  Time: {time_results['BlockQRCS']:.2f}s")

        # Compute Improved QR-CS Shapley values (with adaptive sparsity detection)
        print("\nComputing Improved QR-CS Shapley values (adaptive)...")
        print(f"Automatically detecting sparsity and adapting method")
        start_time = time.time()
        improved_qrcs_explainer = ImprovedQRCSExplainer(
            characteristic_function=custom_char_func,
            n_measurements=n_measurements,
            sparsity_threshold=0.8,  # Switch to fast mode if <80% sparse
            auto_adapt=True,  # Automatically detect and adapt
            verbose=True
        )
        improved_qrcs_values = improved_qrcs_explainer.fit_explain(G)
        time_results['ImprovedQRCS'] = time.time() - start_time

        # Print sparsity detection results
        sparsity_info = improved_qrcs_explainer.get_sparsity_info()
        print(f"  Detected sparsity: {sparsity_info['detected_sparsity']*100:.1f}%")
        print(f"  Using method: {sparsity_info['method']}")
        print(f"  Time: {time_results['ImprovedQRCS']:.2f}s")

        # Compute Improved Block QR-CS Shapley values (with per-block adaptive detection)
        print("\nComputing Improved Block QR-CS Shapley values (per-block adaptive)...")
        print(f"Each block independently detects sparsity and adapts method")
        start_time = time.time()
        improved_block_qrcs_explainer = ImprovedBlockQRCSExplainer(
            characteristic_function=custom_char_func,
            n_blocks=n_blocks,
            sparsity_threshold=0.8,
            auto_adapt=True,
            parallel=True,
            verbose=True
        )
        improved_block_qrcs_values = improved_block_qrcs_explainer.fit_explain(G)
        time_results['ImprovedBlockQRCS'] = time.time() - start_time

        # Print block sparsity statistics
        block_info = improved_block_qrcs_explainer.get_block_sparsity_info()
        if 'avg_sparsity' in block_info:
            print(f"\nBlock sparsity statistics:")
            print(f"  Average sparsity: {block_info['avg_sparsity']*100:.1f}%")
            print(f"  Blocks using CS: {block_info['blocks_using_cs']}/{block_info['n_blocks']}")
            print(f"  Blocks using fast: {block_info['blocks_using_fast']}/{block_info['n_blocks']}")
        print(f"  Time: {time_results['ImprovedBlockQRCS']:.2f}s")

        # Compute Leverage SHAP values (ICLR 2025 - Musco & Witter)
        print("\nComputing Leverage SHAP values (ICLR 2025)...")
        print(f"Using leverage score sampling with provable O(n log n) guarantees")
        start_time = time.time()
        leverage_explainer = LeverageScoreExplainer(
            characteristic_function=custom_char_func,
            n_samples=None,  # Uses 5*n by default for good accuracy
            paired_sampling=True,  # Use paired sampling for better accuracy
            use_bernoulli=True,  # Use Bernoulli sampling without replacement
            random_state=42,
            verbose=True
        )
        leverage_shap_values = leverage_explainer.fit_explain(G)
        time_results['LeverageSHAP'] = time.time() - start_time
        print(f"  Time: {time_results['LeverageSHAP']:.2f}s")
        print(f"  Achieved ~50% error reduction compared to Kernel SHAP (based on paper)")

        # Note: KPI results will be saved after they are computed below

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

    # Add Improved QR-CS values
    sorted_improved_qrcs = sorted(improved_qrcs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['ImprovedQRCS'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_improved_qrcs
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add Improved Block QR-CS values
    sorted_improved_block_qrcs = sorted(improved_block_qrcs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['ImprovedBlockQRCS'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_improved_block_qrcs
        if node_to_feature_name(node, X.columns) is not None
    ]

    # Add Leverage SHAP values
    sorted_leverage_shap = sorted(leverage_shap_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['LeverageSHAP'] = [
        node_to_feature_name(node, X.columns)
        for node, _ in sorted_leverage_shap
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

    # Test improved graph construction methods if requested
    improved_shapley_values = {}
    if test_improved_graphs:
        print("\n" + "="*60)
        print("Testing improved graph construction methods...")
        print("="*60)

        # Test different density ratios for from_rank_deletion
        density_ratios = [None]  # None uses default 1.5x minimum

        for density in density_ratios:
            density_label = f"{density}" if density is not None else "auto"
            print(f"\nBuilding improved graph with density={density_label}...")

            for rank_method in ['cosine', 'kendalltau', 'mutual_info']:
                G_improved = builder.from_rank_deletion(
                    X, y,
                    density_ratio=density,
                    correlation_method=rank_method,
                    similarity_method=rank_method
                )
                print(f"Improved graph has {G_improved.number_of_nodes()} nodes and {G_improved.number_of_edges()} edges")

                # Compute ShapG with improved graph
                print(f"Computing ShapG with improved graph (density={density_label})...")
                start_time = time.time()
                shapg_improved = ShapGExplainer(
                    characteristic_function=custom_char_func,
                    depth=1,
                    n_samples=3,
                    approximate_by_ratio=False,
                    scale=False,
                    verbose=False
                )
                improved_values = shapg_improved.fit_explain(G_improved)
                elapsed_time = time.time() - start_time

                # Add to feature rankings
                sorted_improved = sorted(improved_values.items(), key=lambda x: x[1], reverse=True)
                method_name = f'Improved ShapG-{rank_method}-{density_label}'
                feature_rankings[method_name] = [
                    node_to_feature_name(node, X.columns)
                    for node, _ in sorted_improved
                    if node_to_feature_name(node, X.columns) is not None
                ]
                improved_shapley_values[method_name] = improved_values
                time_results[method_name] = elapsed_time

                print(f"  Completed improved graph benchmark (density={density_label}, time={elapsed_time:.2f}s)")

    # Plot comparison using the original plotting function (pass cached KPI results if available)
    results = plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename, limit, cached_kpi_results)

    # Save all results to unified cache if we computed anything new
    if use_cache and (cached_kpi_results is None or not os.path.exists(cache_file)):
        print("\n" + "=" * 60)
        print("Saving all results to unified cache...")
        print("=" * 60)
        cache_data = {
            'shapley_values': shapley_values,
            'cis_values': cis_values,
            'random_cs_values': random_cs_values,
            'qrcs_values': qrcs_values,
            'block_qrcs_values': block_qrcs_values,
            'improved_qrcs_values': improved_qrcs_values,
            'improved_block_qrcs_values': improved_block_qrcs_values,
            'leverage_shap_values': leverage_shap_values,
            'improved_shapley_values': improved_shapley_values,
            'time_results': time_results,
            'kpi_results': results  # Add KPI results to cache
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved all results (Shapley values, timing, and KPI) to {cache_file}")

    # Print time summary
    print("\n" + "=" * 60)
    print("Execution Time Summary:")
    print("=" * 60)
    for method, exec_time in sorted(time_results.items(), key=lambda x: x[1]):
        print(f"  {method}: {exec_time:.2f}s")

    # Return results including all CS variants, Leverage SHAP, improved graphs, and timing information
    return (shapley_values, cis_values, random_cs_values, qrcs_values,
            block_qrcs_values, improved_qrcs_values, improved_block_qrcs_values,
            leverage_shap_values, improved_shapley_values, time_results, results)


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK WITH NEW MODULAR API")
    print("=" * 60)

    # Example usage with new API
    model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)

    print("\nRunning benchmark with housing data...")
    (shapley_values, cis_values, random_cs_values, qrcs_values,
     block_qrcs_values, improved_qrcs_values, improved_block_qrcs_values,
     leverage_shap_values, improved_shapley_values, time_results, results) = benchmark_feature_importance(
        housing_data_reader,
        model,
        filename='housing_benchmark_with_qr.png',
        limit=10,
        test_improved_graphs=True
    )

    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)
    print("Random CS values:", random_cs_values)
    print("QR-CS values:", qrcs_values)
    print("Block QR-CS values:", block_qrcs_values)
    print("Improved QR-CS values:", improved_qrcs_values)
    print("Improved Block QR-CS values:", improved_block_qrcs_values)
    print("Leverage SHAP values:", leverage_shap_values)
    if improved_shapley_values:
        print("\nImproved graph Shapley values:")
        for method, values in improved_shapley_values.items():
            print(f"  {method}:", values)

    # Generate time comparison plot
    print("\n" + "=" * 60)
    print("Generating time comparison visualization...")
    print("=" * 60)
    plot_time_comparison(time_results, filename='housing_benchmark_with_qr.png')
    print("\nBenchmark complete!")
