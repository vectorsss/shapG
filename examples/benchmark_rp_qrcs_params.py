"""Benchmark script comparing different parameter configurations of StratifiedShapleyExplainer.

This script tests various combinations of:
- allocation_strategy: 'shapley_weighted', 'uniform', 'leverage', 'leverage_bernoulli'
- use_direct_estimation: True (direct weighted estimation) vs False (CS reconstruction)
- n_samples: Different sample budgets
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import pickle
import time
import argparse
from pathlib import Path
from typing import Set
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))

from shapG import GraphBuilder
from shapG.explainer import StratifiedShapleyExplainer, ExactExplainer
from shapG.characteristic import GraphModelCharacteristic

# Optional plotting style (consistent with benchmark_qr_new_api.py)
try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    print("gnuplot_style not found, using default style")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OUTPUT_DIR = Path("output")
CACHE_ENABLED = True

# Configurations to test
# ALLOCATION_STRATEGIES = ['leverage_bernoulli']
ALLOCATION_STRATEGIES = ['shapley_weighted', 'leverage', 'leverage_bernoulli']
# USE_DIRECT_ESTIMATIONS = [False]
USE_DIRECT_ESTIMATIONS = [True, False]
N_SAMPLES_OPTIONS = [50]

# Fixed parameters
SEED = 42
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 300
PLOT_ALPHA = 0.6
WEIGHTED_SLOPE_BETA = 0.8

# ==============================================================================
# DATA READERS
# ==============================================================================

def housing_data_reader(filename='./data/housing_price.csv'):
    """Load Boston housing dataset."""
    data = pd.read_csv(filename)
    X = data.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return X, y


def h1n1_data_reader(filename='./data/process_data.csv'):
    """Load H1N1 vaccine dataset."""
    data = pd.read_csv(filename)
    X = data.drop(['h1n1_vaccine', 'respondent_id', 'seasonal_vaccine'], axis=1)
    y = data['h1n1_vaccine']
    return X, y


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Model hyperparameters (for KPI computation)
MODEL_RANDOM_STATES = {
    lgb.LGBMClassifier: [10, 10],
    lgb.LGBMRegressor: [42, 42]
}
MODEL_TEST_SIZES = {
    lgb.LGBMClassifier: [0.2, 0.2],
    lgb.LGBMRegressor: [0.2, 0.3]
}


def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _get_cache_filename(dataset, imputation_strategy='mean', retrain_kpi=False):
    """
    Generate cache filename based on dataset name and configuration.

    Returns:
        Full path to cache file in output directory
    """
    output_dir = _ensure_output_dir()
    kpi_mode = "retrain" if retrain_kpi else "mask"
    cache_filename = f"stratified_{dataset}_cache_{imputation_strategy}_{kpi_mode}.pkl"
    return str(output_dir / cache_filename)


def _load_cache(cache_file):
    """Load cached results from file."""
    if not os.path.exists(cache_file):
        return None

    print(f"Loading cached results from {cache_file}...")

    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        print(f"Successfully loaded cached results!")
        print(f"  - Configurations: {len(cached_data.get('names', []))}")
        return cached_data
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


def _save_cache(cache_file, results):
    """Save results to cache file."""
    _ensure_output_dir()
    print(f"\nSaving results to cache: {cache_file}")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Successfully saved cache!")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def _get_model_params(model):
    """Get model-specific random states and test sizes."""
    model_type = type(model)
    random_state = MODEL_RANDOM_STATES.get(model_type, [42, 42])
    test_size = MODEL_TEST_SIZES.get(model_type, [0.2, 0.2])
    return random_state, test_size


def _get_metric_score(model, y_test, y_pred):
    """Calculate appropriate metric score based on model type."""
    if isinstance(model, lgb.LGBMRegressor):
        return r2_score(y_test, y_pred)
    return accuracy_score(y_test, y_pred)


def _get_metric_name(model):
    """Get metric name for plotting."""
    return "$R^2$" if isinstance(model, lgb.LGBMRegressor) else "Accuracy"


def _train_model_once(X, y, model):
    """Train model once for use in characteristic function."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    is_classifier = isinstance(model, lgb.LGBMClassifier)
    task_type = "Classification" if is_classifier else "Regression"
    metric_name = "Accuracy" if is_classifier else "R²"

    print(f"Training model once ({task_type})...")
    model.fit(X_train, y_train)
    print(f"  Train {metric_name}: {model.score(X_train, y_train):.4f}")
    print(f"  Test {metric_name}: {model.score(X_test, y_test):.4f}")

    return model, X_train, X_test, y_test


def _create_characteristic_function(model, X_train, X_test, y_test, imputation_strategy='mean'):
    """Create characteristic function using masking approach."""
    is_classifier = isinstance(model, lgb.LGBMClassifier)
    metric_fn = accuracy_score if is_classifier else r2_score
    task_type = "Classification" if is_classifier else "Regression"
    metric_name = "Accuracy" if is_classifier else "R2"

    baseline = X_train.mean().values

    return GraphModelCharacteristic(
        model=model,
        X=X_test,
        y=y_test,
        masking_strategy=imputation_strategy,
        metric_fn=metric_fn,
        baseline=baseline,
        name=f"{task_type} {metric_name} (Masking-{imputation_strategy})"
    )


def node_to_feature_name(node, columns):
    """Convert graph node (index or name) to feature name."""
    try:
        idx = int(node)
        if 0 <= idx < len(columns):
            return columns[idx]
    except (ValueError, TypeError):
        if node in columns:
            return node
    return None


def _compute_exact_shapley(G, custom_char_func):
    """Compute exact Shapley values as ground truth."""
    print("\nComputing exact Shapley values (ground truth)...")
    start_time = time.time()

    exact_explainer = ExactExplainer(
        characteristic_function=custom_char_func,
        verbose=True
    )
    exact_values = exact_explainer.fit_explain(G)
    elapsed = time.time() - start_time

    print(f"  Exact computation time: {elapsed:.2f}s")
    return exact_values


def _compute_error(values, exact_values):
    """Compute L2 error between estimated and exact Shapley values."""
    if not exact_values:
        return None

    error_sum = 0.0
    count = 0

    for node in exact_values:
        if node in values:
            error_sum += (values[node] - exact_values[node]) ** 2
            count += 1

    if count == 0:
        return None

    return np.sqrt(error_sum / count)


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def _generate_config_name(config):
    """Generate readable name for configuration."""
    strategy = config['allocation_strategy']
    estimation = 'direct' if config['use_direct_estimation'] else 'cs'
    n_samples = config['n_samples']
    return f"{strategy}_{estimation}_{n_samples}s"


def benchmark_stratified_configurations(reader, model, dataset, compute_exact=True,
                                     imputation_strategy='mean', use_cache=True,
                                     retrain_kpi=False):
    """
    Benchmark different Stratified Shapley configurations.

    Args:
        reader: Dataset reader function
        model: ML model to use
        dataset: Dataset name
        compute_exact: Whether to compute exact Shapley values for comparison
        imputation_strategy: Masking strategy
        use_cache: Whether to use caching
        retrain_kpi: Whether to use retraining for KPI computation (for cache key)

    Returns:
        Dictionary with results for all configurations
    """
    # Check cache first
    cache_file = _get_cache_filename(dataset, imputation_strategy, retrain_kpi)
    if use_cache:
        cached_results = _load_cache(cache_file)
        if cached_results is not None:
            print("\nUsing cached results. Run with --no-cache to recompute.")
            return cached_results

    # Load data
    X, y = reader()

    # Train model once
    print("\n" + "=" * 80)
    print("STEP 1: Train model once")
    print("=" * 80)
    trained_model, X_train, X_test, y_test = _train_model_once(X, y, model)

    # Build graph
    print("\n" + "=" * 80)
    print("STEP 2: Build graph")
    print("=" * 80)
    builder = GraphBuilder()
    G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Create characteristic function
    print("\n" + "=" * 80)
    print("STEP 3: Create characteristic function")
    print("=" * 80)
    custom_char_func = _create_characteristic_function(
        model=trained_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        imputation_strategy=imputation_strategy
    )

    # Compute exact Shapley values if requested
    exact_values = None
    if compute_exact and G.number_of_nodes() <= 15:
        exact_values = _compute_exact_shapley(G, custom_char_func)
    elif compute_exact:
        print(f"\nSkipping exact computation (n={G.number_of_nodes()} > 15 nodes)")

    # Generate all configurations to test
    print("\n" + "=" * 80)
    print("STEP 4: Generate configurations")
    print("=" * 80)

    configs = []
    for strategy, direct, n_samples in product(
        ALLOCATION_STRATEGIES,
        USE_DIRECT_ESTIMATIONS,
        N_SAMPLES_OPTIONS
    ):
        config = {
            'allocation_strategy': strategy,
            'use_direct_estimation': direct,
            'n_samples': n_samples,
            'seed': SEED
        }
        configs.append(config)

    print(f"Total configurations to test: {len(configs)}")

    # Test all configurations
    print("\n" + "=" * 80)
    print("STEP 5: Test all configurations")
    print("=" * 80)

    results = {
        'configs': [],
        'names': [],
        'shapley_values': [],
        'times': [],
        'errors': [],
        'sampling_stats': []
    }

    for i, config in enumerate(configs, 1):
        config_name = _generate_config_name(config)
        print(f"\n[{i}/{len(configs)}] Testing: {config_name}")
        print(f"  Config: {config}")

        # Run Stratified Shapley with this configuration
        start_time = time.time()

        try:
            explainer = StratifiedShapleyExplainer(
                characteristic_function=custom_char_func,
                verbose=True,
                **config
            )

            values = explainer.fit_explain(G)
            elapsed = time.time() - start_time

            # Compute error if exact values available
            error = _compute_error(values, exact_values) if exact_values else None

            # Get sampling statistics
            stats = explainer.get_sampling_stats()

            # Store results
            results['configs'].append(config)
            results['names'].append(config_name)
            results['shapley_values'].append(values)
            results['times'].append(elapsed)
            results['errors'].append(error)
            results['sampling_stats'].append(stats)

            print(f"  Time: {elapsed:.2f}s")
            if error is not None:
                print(f"  L2 Error: {error:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results['configs'].append(config)
            results['names'].append(config_name)
            results['shapley_values'].append({})
            results['times'].append(None)
            results['errors'].append(None)
            results['sampling_stats'].append(None)

    # Add exact values to results
    results['exact_values'] = exact_values
    results['dataset'] = dataset
    results['n_features'] = X.shape[1]
    results['feature_names'] = X.columns.tolist()

    # Save cache
    if use_cache:
        _save_cache(cache_file, results)

    return results


# ==============================================================================
# VISUALIZATION AND EXPORT
# ==============================================================================

def export_results_csv(results, output_dir=None):
    """Export benchmark results to CSV."""
    if output_dir is None:
        output_dir = _ensure_output_dir()

    dataset = results['dataset']

    # Create summary DataFrame
    data = {
        'Config': results['names'],
        'Allocation Strategy': [c['allocation_strategy'] for c in results['configs']],
        'Estimation Method': ['Direct' if c['use_direct_estimation'] else 'CS'
                             for c in results['configs']],
        'N Samples': [c['n_samples'] for c in results['configs']],
        'Time (s)': results['times'],
        'L2 Error': results['errors']
    }

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = output_dir / f"stratified_benchmark_{dataset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to: {csv_path}")

    return csv_path


def plot_error_comparison(results, output_dir=None):
    """Plot L2 error comparison across configurations."""
    if output_dir is None:
        output_dir = _ensure_output_dir()

    dataset = results['dataset']

    # Filter out None errors
    valid_indices = [i for i, e in enumerate(results['errors']) if e is not None]
    if not valid_indices:
        print("No error data available for plotting")
        return

    names = [results['names'][i] for i in valid_indices]
    errors = [results['errors'][i] for i in valid_indices]
    strategies = [results['configs'][i]['allocation_strategy'] for i in valid_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # Color by strategy
    unique_strategies = sorted(set(strategies))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
    color_map = {s: colors[i] for i, s in enumerate(unique_strategies)}

    bar_colors = [color_map[s] for s in strategies]

    # Plot bars
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, errors, color=bar_colors, alpha=0.7, edgecolor='black')

    # Customize plot
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Error', fontsize=12, fontweight='bold')
    ax.set_title(f'Stratified Shapley Parameter Comparison - L2 Error ({dataset})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=90, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[s], label=s)
                      for s in unique_strategies]
    ax.legend(handles=legend_elements, title='Allocation Strategy', loc='best')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"stratified_error_comparison_{dataset}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_DPI)
    print(f"Saved error comparison plot to: {plot_path}")

    plt.close()
    return plot_path


def plot_time_comparison(results, output_dir=None):
    """
    Plot horizontal bar chart comparing algorithm execution times.
    Matches the style from benchmark_qr_new_api.py.
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    dataset = results['dataset']

    # Filter out None times
    valid_indices = [i for i, t in enumerate(results['times']) if t is not None]
    if not valid_indices:
        print("No timing data available for plotting")
        return

    # Build time results dict and sort by time
    time_results = {}
    for i in valid_indices:
        time_results[results['names'][i]] = results['times'][i]

    # Sort by time for better visualization (fastest first)
    sorted_items = sorted(time_results.items(), key=lambda x: x[1])
    algorithms = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]

    # Create horizontal bar chart (same style as benchmark_qr_new_api.py)
    plt.figure(figsize=PLOT_FIGSIZE)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(algorithms)))
    bars = plt.barh(algorithms, times, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        plt.text(time_val, i, f' {time_val:.2f}s',
                va='center', ha='left', fontweight='bold', fontsize=10)

    plt.xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12, fontweight='bold')
    plt.title(f'Stratified Shapley Execution Time Comparison ({dataset})', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"stratified_time_comparison_{dataset}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_DPI)
    print(f"Saved time comparison plot to: {plot_path}")

    plt.close()
    return plot_path


def _compute_kpi_results(reader, feature_rankings, model, limit=10,
                         retrain_kpi=False, imputation_strategy='mean'):
    """
    Compute KPI results by progressively dropping features.

    Two modes (consistent with benchmark_qr_new_api.py):
    1. Masking mode (retrain_kpi=False): Train once, mask excluded features
    2. Retraining mode (retrain_kpi=True): Retrain model for each feature subset

    Parameters:
        reader: Dataset reader function
        feature_rankings: Dict mapping config names to ranked feature lists
        model: ML model to evaluate
        limit: Number of top features to evaluate
        retrain_kpi: If True, retrain for each drop. If False, use masking.
        imputation_strategy: For masking mode ('mean', 'zero', 'permutation')

    Returns:
        Dictionary mapping method names to metrics, features, and weighted slopes
    """
    random_state, test_size = _get_model_params(model)
    X, y = reader()
    results = {}

    if retrain_kpi:
        # RETRAINING MODE: Retrain model for each feature subset
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        initial_metric = _get_metric_score(model, y_test, y_pred)

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
                    print(f"Warning: Columns {missing_cols} not found. Skipping.")
                    continue

                # Drop columns and retrain
                reduced_X = X.drop(columns=features_to_drop)
                x_train, x_test, y_train, y_test = train_test_split(
                    reduced_X, y, test_size=test_size[1], random_state=random_state[1]
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                new_metric = _get_metric_score(model, y_test, y_pred)

                deltas.append(metrics[-1] - new_metric)
                metrics.append(new_metric)
                features.append(features_to_drop)

            weights = [WEIGHTED_SLOPE_BETA**i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weights) if deltas else 0
            }
    else:
        # MASKING MODE: Train once, mask excluded features
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)

        # Compute baseline statistics from training set
        col_means = x_train.mean(numeric_only=True)
        col_modes = {
            c: (x_train[c].mode(dropna=True).iloc[0]
                if not pd.api.types.is_numeric_dtype(x_train[c]) and not x_train[c].mode().empty
                else None)
            for c in x_train.columns
        }

        # Random number generator for permutation
        rng = np.random.default_rng(42) if imputation_strategy == 'permutation' else None

        def mask_features(X_df, keep_cols):
            """Mask features not in keep_cols using specified strategy."""
            X_masked = X_df.copy()
            drop_cols = [c for c in X_df.columns if c not in keep_cols]

            for c in drop_cols:
                if imputation_strategy == 'zero':
                    X_masked[c] = 0
                elif imputation_strategy == 'permutation':
                    X_masked[c] = rng.permutation(X_masked[c].values)
                else:  # 'mean' (default)
                    if pd.api.types.is_numeric_dtype(X_masked[c]):
                        X_masked[c] = col_means.get(c, X_masked[c].mean())
                    else:
                        mode_val = col_modes.get(c, None)
                        if mode_val is not None:
                            X_masked[c] = mode_val
            return X_masked

        # Calculate initial metric (all features)
        y_pred = model.predict(x_test)
        initial_metric = _get_metric_score(model, y_test, y_pred)

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
                    print(f"Warning: Columns {missing_cols} not found. Skipping.")
                    continue

                # Mask dropped features and evaluate (NO RETRAINING!)
                keep_cols = [c for c in x_test.columns if c not in features_to_drop]
                x_test_masked = mask_features(x_test, set(keep_cols))
                y_pred = model.predict(x_test_masked)
                new_metric = _get_metric_score(model, y_test, y_pred)

                deltas.append(metrics[-1] - new_metric)
                metrics.append(new_metric)
                features.append(features_to_drop)

            weights = [WEIGHTED_SLOPE_BETA**i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weights) if deltas else 0
            }

    return results


def plot_kpi_comparison(reader, results, model, dataset, limit=10,
                       retrain_kpi=False, imputation_strategy='mean', output_dir=None):
    """
    Plot KPI comparison for different Stratified Shapley configurations.

    Shows how model performance degrades as top features are progressively dropped.

    Parameters:
        reader: Dataset reader function
        results: Benchmark results dictionary
        model: ML model
        dataset: Dataset name
        limit: Number of top features to evaluate
        retrain_kpi: If True, retrain for each drop. If False, use masking.
        imputation_strategy: Masking strategy
        output_dir: Output directory

    Returns:
        Tuple of (kpi_results, plot_path)
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    # Convert Shapley values to feature rankings
    print(f"\nConverting Shapley values to feature rankings (top {limit})...")

    X, y = reader()
    feature_rankings = {}

    for i, config_name in enumerate(results['names']):
        values = results['shapley_values'][i]
        if not values:
            continue

        # Sort by Shapley value
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

        # Convert to feature names
        feature_ranking = []
        for node, _ in sorted_values:
            try:
                idx = int(node)
                if 0 <= idx < len(X.columns):
                    feature_ranking.append(X.columns[idx])
            except (ValueError, TypeError):
                if node in X.columns:
                    feature_ranking.append(node)

        feature_rankings[config_name] = feature_ranking

    if not feature_rankings:
        print("No valid feature rankings found. Skipping KPI plot.")
        return None, None

    # Compute KPI results
    mode_desc = "retraining" if retrain_kpi else f"{imputation_strategy} masking"
    print(f"\nComputing KPI results using {mode_desc}...")
    kpi_results = _compute_kpi_results(
        reader,
        feature_rankings,
        model,
        limit,
        retrain_kpi=retrain_kpi,
        imputation_strategy=imputation_strategy
    )

    # Create plot
    plt.figure(figsize=PLOT_FIGSIZE)
    metric_name = _get_metric_name(model)
    model_name = type(model).__name__

    # Plot each configuration (same style as benchmark_qr_new_api.py)
    for method, data in kpi_results.items():
        label = f'{method} $S$={data["Slope"]:.4f}'
        plt.plot(
            range(len(data['Metrics'])),
            data['Metrics'],
            label=label,
            alpha=PLOT_ALPHA
        )

    plt.xlabel('Number of Features Dropped')
    plt.ylabel(metric_name)
    plt.title(f'Stratified Shapley Parameter Comparison - {metric_name} after dropping features ({model_name})')
    plt.legend()
    plt.grid()

    # Save plot
    plot_path = output_dir / f"stratified_kpi_{dataset}_{imputation_strategy}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=PLOT_DPI)
    print(f"Saved KPI comparison plot to: {plot_path}")

    plt.close()

    return kpi_results, plot_path


def print_summary_table(results):
    """Print summary table of results."""
    print("\n" + "=" * 130)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 130)

    # Print header
    header = f"{'Config':<40} {'Strategy':<20} {'Method':<8} {'Samples':<8} {'Time(s)':<12} {'Relative':<10} {'Error':<12}"
    print(header)
    print("-" * 130)

    # Get valid times for relative speed calculation
    valid_times = [t for t in results['times'] if t is not None]
    fastest_time = min(valid_times) if valid_times else 1.0

    # Print rows sorted by error (if available)
    indices = list(range(len(results['names'])))
    if any(e is not None for e in results['errors']):
        indices = sorted(indices, key=lambda i: results['errors'][i] if results['errors'][i] is not None else float('inf'))

    for i in indices:
        name = results['names'][i]
        config = results['configs'][i]
        strategy = config['allocation_strategy']
        method = 'Direct' if config['use_direct_estimation'] else 'CS'
        n_samples = config['n_samples']

        if results['times'][i] is not None:
            time_val = f"{results['times'][i]:.2f}"
            relative_speed = results['times'][i] / fastest_time
            relative_val = f"{relative_speed:.1f}x"
        else:
            time_val = "N/A"
            relative_val = "N/A"

        error_val = f"{results['errors'][i]:.6f}" if results['errors'][i] is not None else "N/A"

        print(f"{name:<40} {strategy:<20} {method:<8} {n_samples:<8} {time_val:<12} {relative_val:<10} {error_val:<12}")

    print("=" * 130)

    # Print execution time comparison (sorted by time)
    _print_time_comparison_table(results)


def _print_time_comparison_table(results):
    """
    Print execution time comparison table to terminal.
    Matches the style from benchmark_qr_new_api.py.
    """
    print("\n" + "=" * 80)
    print("EXECUTION TIME COMPARISON")
    print("=" * 80)

    # Build time results dict (filter out None values)
    time_results = {}
    for i, name in enumerate(results['names']):
        if results['times'][i] is not None:
            time_results[name] = results['times'][i]

    if not time_results:
        print("No valid time results to display.")
        print("=" * 80)
        return

    # Sort by time
    sorted_times = sorted(time_results.items(), key=lambda x: x[1])

    # Print header
    print(f"{'Method':<50} {'Time (seconds)':<15} {'Relative Speed':<15}")
    print("-" * 80)

    # Get fastest time for relative comparison
    fastest_time = sorted_times[0][1] if sorted_times else 1.0

    # Print rows
    for method, exec_time in sorted_times:
        relative = exec_time / fastest_time
        print(f"{method:<50} {exec_time:>14.2f}s {relative:>14.1f}x")

    print("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark different Stratified Shapley parameter configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        choices=['housing', 'h1n1'],
        default='housing',
        help='Dataset to use (default: housing)'
    )

    parser.add_argument(
        '--no-exact',
        action='store_true',
        help='Skip exact Shapley computation (faster for large graphs)'
    )

    parser.add_argument(
        '--imputation',
        choices=['mean', 'zero', 'permutation'],
        default='mean',
        help='Masking strategy (default: mean)'
    )

    parser.add_argument(
        '--kpi-limit',
        type=int,
        default=10,
        help='Number of top features to evaluate in KPI plot (default: 10)'
    )

    parser.add_argument(
        '--retrain-kpi',
        action='store_true',
        help='Use retraining (not masking) for KPI plots. '
             'Slower but measures true feature importance for model quality. '
             'Default is to use masking (consistent with Shapley computation).'
    )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Use retraining approach (slow, retrains model for each coalition). '
             'Default is to use efficient masking (~100x faster).'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of results'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STRATIFIED SHAPLEY PARAMETER BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Compute exact: {not args.no_exact}")
    print(f"  Imputation: {args.imputation}")
    print(f"  KPI limit: {args.kpi_limit} features")
    print(f"  Retrain KPI: {args.retrain_kpi}")
    print(f"  Retrain Shapley: {args.retrain}")
    print(f"  Use cache: {not args.no_cache}")
    print(f"\nParameter ranges:")
    print(f"  Allocation strategies: {ALLOCATION_STRATEGIES}")
    print(f"  Estimation methods: Direct vs CS")
    print(f"  Sample budgets: {N_SAMPLES_OPTIONS}")

    # Select dataset and model
    if args.dataset == 'housing':
        reader = housing_data_reader
        model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
        print(f"  Task: Regression (R² metric)")
    else:
        reader = h1n1_data_reader
        model = lgb.LGBMClassifier(learning_rate=0.3, verbosity=-1)
        print(f"  Task: Classification (Accuracy metric)")

    # Run benchmark
    results = benchmark_stratified_configurations(
        reader=reader,
        model=model,
        dataset=args.dataset,
        compute_exact=not args.no_exact,
        imputation_strategy=args.imputation,
        use_cache=not args.no_cache,
        retrain_kpi=args.retrain_kpi
    )

    # Print summary
    print_summary_table(results)

    # Export results
    print("\n" + "=" * 80)
    print("Exporting results...")
    print("=" * 80)
    export_results_csv(results)

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    plot_error_comparison(results)
    plot_time_comparison(results)

    # Generate KPI comparison plot
    print("\n" + "=" * 80)
    print("Generating KPI comparison plot...")
    print("=" * 80)
    kpi_results, kpi_plot_path = plot_kpi_comparison(
        reader=reader,
        results=results,
        model=model,
        dataset=args.dataset,
        limit=args.kpi_limit,
        retrain_kpi=args.retrain_kpi,
        imputation_strategy=args.imputation
    )

    if kpi_results:
        # Export KPI metrics to CSV
        kpi_csv_data = {
            'Config': [],
            'Weighted Slope (S)': []
        }

        # Determine max number of metrics
        max_metrics = max(len(data['Metrics']) for data in kpi_results.values())
        for i in range(max_metrics):
            kpi_csv_data[f'Metric_{i}'] = []

        # Populate data
        for config_name in sorted(kpi_results.keys()):
            data = kpi_results[config_name]
            kpi_csv_data['Config'].append(config_name)
            kpi_csv_data['Weighted Slope (S)'].append(data['Slope'])

            metrics = data['Metrics']
            for i in range(max_metrics):
                if i < len(metrics):
                    kpi_csv_data[f'Metric_{i}'].append(metrics[i])
                else:
                    kpi_csv_data[f'Metric_{i}'].append('')

        kpi_df = pd.DataFrame(kpi_csv_data)
        kpi_csv_path = OUTPUT_DIR / f"stratified_kpi_metrics_{args.dataset}_{args.imputation}.csv"
        kpi_df.to_csv(kpi_csv_path, index=False)
        print(f"Saved KPI metrics to: {kpi_csv_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
