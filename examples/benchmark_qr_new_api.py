"""
Benchmark script comparing multiple Shapley value computation methods using the NEW modular API.
Migrated from benchmark_qr.py to use the new ShapG architecture.

Methods compared:
- ShapGExplainer: Fast approximate computation using local search and sampling
- CISExplainer: Combined Imputation Score computation
- RandomCSExplainer: Random compressed sensing approach using Bernoulli matrices
- QRCSExplainer: QR-based compressed sensing for Shapley values
- BlockQRCSExplainer: Block QR-CS for high-dimensional problems with parallel computation
- ImprovedQRCSExplainer: Adaptive QR-CS with automatic sparsity detection
- ImprovedBlockQRCSExplainer: Per-block adaptive detection
- StratifiedShapleyExplainer: Stratified coalition sampling with flexible allocation strategies
- LeverageScoreExplainer: Leverage score sampling (ICLR 2025)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Standard library
import os
import sys
import pickle
import time
import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Set

# Scientific computing
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau, pearsonr
from scipy.optimize import minimize
from scipy.linalg import qr

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))

# ShapG API
from shapG import (
    ShapGExplainer,
    CISExplainer,
    RandomCSExplainer,
    QRCSExplainer,
    MultilinearExplainer,
    CustomFunction,
    GraphBuilder
)
from shapG.explainer import (
    BlockQRCSExplainer,
    ImprovedQRCSExplainer,
    ImprovedBlockQRCSExplainer,
    StratifiedShapleyExplainer,
    LeverageScoreExplainer
)
from shapG.characteristic import GraphModelCharacteristic

# Optional plotting style
try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    print("gnuplot_style not found, using default style")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Output directory
OUTPUT_DIR = Path("output")

# Cache settings
CACHE_ENABLED = True

# Model hyperparameters
MODEL_RANDOM_STATES = {
    lgb.LGBMClassifier: [10, 10],
    lgb.LGBMRegressor: [42, 42]
}
MODEL_TEST_SIZES = {
    lgb.LGBMClassifier: [0.2, 0.2],
    lgb.LGBMRegressor: [0.2, 0.3]
}

# Explainer configurations
SHAPG_CONFIG = {
    'depth': 1,
    'n_samples': 3,
    'approximate_by_ratio': False,
    'scale': False,
}

RANDOM_CS_CONFIG = {
    'm': 50,          # Measurements per iteration
    't': 30,          # Number of iterations
    'seed': 42,
}

LEVERAGE_SHAP_CONFIG = {
    'n_samples': 500,      # Uses 5*n by default
    'paired_sampling': True,
    'use_bernoulli': True,
    'random_state': 42,
}

IMPROVED_QRCS_CONFIG = {
    'sparsity_threshold': 0.8,
    'auto_adapt': True,
}

RP_QRCS_CONFIG = {
    'n_samples': 50,                      # Total coalition samples
    'allocation_strategy': 'leverage_bernoulli',  # 'shapley_weighted', 'uniform', or 'leverage'
    'use_direct_estimation': True,          # Use direct weighted estimation (True) or CS reconstruction (False)
    'seed': 42,
}

MULTILINEAR_CONFIG = {
    'max_exact_size': 10,
    'n_quadrature': 21,      # Quadrature points for integration (odd number preferred)
    'n_samples': 100,        # Samples per quadrature point for partial derivative estimation
    'use_leverage': True,    # Use leverage-stratified sampling (LEM)
    'compute_error_bounds': True,  # Compute rigorous error bounds
    'confidence': 0.95,      # Confidence level for error bounds
    'seed': 42,              # Random seed for reproducibility
}

# Plotting configuration
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


def _compute_kpi_results(reader, feature_rankings, model, limit=10,
                        retrain_kpi=False, imputation_strategy='mean'):
    """
    Compute KPI results by progressively dropping features.

    Two modes:
    1. Masking mode (retrain_kpi=False): Train once, mask excluded features
    2. Retraining mode (retrain_kpi=True): Retrain model for each feature subset

    Parameters:
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
        # Calculate initial metric
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)
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

            # Progressively drop features and RETRAIN
            for i in range(1, len(feature_order) + 1):
                features_to_drop = feature_order[:i]

                missing_cols = [col for col in features_to_drop if col not in X.columns]
                if missing_cols:
                    print(f"Warning: Columns {missing_cols} not found. Skipping.")
                    continue

                # Actually drop columns and retrain
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
            # Normalize feature order
            feature_order = [feat if isinstance(feat, str) else feat[0] for feat in feature_order]
            if limit:
                feature_order = feature_order[:limit]

            metrics = [initial_metric]
            features = [[]]
            deltas = []

            # Progressively MASK features (not drop/retrain)
            for i in range(1, len(feature_order) + 1):
                features_to_drop = feature_order[:i]

                # Validate features exist
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

            # Calculate weighted slope
            weights = [WEIGHTED_SLOPE_BETA**i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weights) if deltas else 0
            }

    return results


def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _generate_filename(plot_type, dataset, imputation_strategy, no_retrain, retrain_kpi):
    """
    Generate filename based on configuration.

    Args:
        plot_type: 'KPI' or 'time_comparison'
        dataset: 'housing', 'h1n1', etc.
        imputation_strategy: 'mean', 'zero', 'permutation'
        no_retrain: True if Shapley uses masking, False if retraining
        retrain_kpi: True if KPI uses retraining, False if masking

    Returns:
        Base filename (without extension)
    """
    shapley_mode = 'mask' if no_retrain else 'retrain'
    kpi_mode = 'retrain' if retrain_kpi else 'mask'
    return f"{plot_type}_{dataset}_{imputation_strategy}_shapley-{shapley_mode}_kpi-{kpi_mode}"


def _save_plot(base_filename, output_dir=None):
    """
    Save current plot in both PNG and PDF formats.

    Args:
        base_filename: Filename without extension
        output_dir: Output directory (defaults to OUTPUT_DIR)

    Returns:
        Dict with paths to saved files
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    saved_files = {}

    # Save PDF
    pdf_path = output_dir / f"{base_filename}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    saved_files['pdf'] = str(pdf_path)

    print(f"Saved plots:")
    print(f"  PDF: {pdf_path}")

    return saved_files


def _export_feature_rankings_csv(feature_rankings, dataset, imputation_strategy, no_retrain, retrain_kpi, output_dir=None):
    """
    Export feature rankings to CSV file.

    CSV format:
        Top-N, Method1, Method2, Method3, ...
        1,     RM,      RM,      LSTAT, ...
        2,     LSTAT,   LSTAT,   RM,    ...
        ...

    Args:
        feature_rankings: Dict mapping method names to ranked feature lists
        dataset: Dataset name
        imputation_strategy: Imputation strategy used
        no_retrain: Whether Shapley used masking (True) or retraining (False)
        retrain_kpi: Whether KPI used retraining
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    # Determine maximum number of features across all methods
    max_features = max(len(ranking) for ranking in feature_rankings.values())

    # Create DataFrame
    data = {'Top-N': list(range(1, max_features + 1))}

    for method, ranking in sorted(feature_rankings.items()):
        # Pad with empty strings if needed
        padded_ranking = ranking + [''] * (max_features - len(ranking))
        data[method] = padded_ranking

    df = pd.DataFrame(data)

    # Generate filename with both Shapley and KPI modes
    shapley_mode = 'mask' if no_retrain else 'retrain'
    kpi_mode = 'retrain' if retrain_kpi else 'mask'
    csv_filename = f"rankings_{dataset}_{imputation_strategy}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    csv_path = output_dir / csv_filename

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved feature rankings to: {csv_path}")

    return csv_path


def _print_feature_ranking_table(feature_rankings, limit=10):
    """
    Print feature ranking table to terminal.

    Args:
        feature_rankings: Dict mapping method names to ranked feature lists
        limit: Number of top features to display
    """
    print("\n" + "=" * 100)
    print("FEATURE RANKING TABLE")
    print("=" * 100)

    # Get methods
    methods = sorted(feature_rankings.keys())
    if not methods:
        print("No rankings to display")
        return

    # Determine column width
    max_method_len = max(len(m) for m in methods)
    col_width = max(max_method_len, 15)

    # Print header
    header = f"{'Top-N':<8}"
    for method in methods:
        header += f" {method:<{col_width}}"
    print(header)
    print("-" * len(header))

    # Print rows
    max_features = min(limit, max(len(ranking) for ranking in feature_rankings.values()))
    for i in range(max_features):
        row = f"{i+1:<8}"
        for method in methods:
            ranking = feature_rankings[method]
            feature = ranking[i] if i < len(ranking) else ''
            row += f" {feature:<{col_width}}"
        print(row)

    print("=" * 100)


def _export_time_comparison_csv(time_results, dataset, imputation_strategy, no_retrain, retrain_kpi, output_dir=None):
    """
    Export execution time comparison to CSV file.

    CSV format:
        Method, Time (seconds), Relative Speed
        ShapG, 1.23, 1.0x
        QRCS, 2.45, 2.0x
        ...

    Args:
        time_results: Dict mapping method names to execution times (seconds)
        dataset: Dataset name
        imputation_strategy: Imputation strategy used
        no_retrain: Whether Shapley used masking (True) or retraining (False)
        retrain_kpi: Whether KPI used retraining
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    # Sort by time
    sorted_times = sorted(time_results.items(), key=lambda x: x[1])

    # Get fastest time for relative comparison
    fastest_time = sorted_times[0][1] if sorted_times else 1.0

    # Create DataFrame
    data = {
        'Method': [method for method, _ in sorted_times],
        'Time (seconds)': [time for _, time in sorted_times],
        'Relative Speed': [f"{time/fastest_time:.2f}x" for _, time in sorted_times]
    }
    df = pd.DataFrame(data)

    # Generate filename
    shapley_mode = 'mask' if no_retrain else 'retrain'
    kpi_mode = 'retrain' if retrain_kpi else 'mask'
    csv_filename = f"time_{dataset}_{imputation_strategy}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    csv_path = output_dir / csv_filename

    # Save CSV
    df.to_csv(csv_path, index=False)

    print(f"\nSaved time comparison CSV: {csv_path}")
    return str(csv_path)


def _export_kpi_metrics_csv(kpi_results, dataset, imputation_strategy, no_retrain, retrain_kpi, output_dir=None):
    """
    Export KPI metrics (S values and metric sequences) to CSV file.

    CSV format:
        Method, Weighted Slope (S), Metric_0, Metric_1, Metric_2, ...
        ShapG, 0.123, 0.85, 0.82, 0.78, ...
        QRCS, 0.145, 0.85, 0.81, 0.76, ...
        ...

    Args:
        kpi_results: Dict mapping method names to KPI result dictionaries
        dataset: Dataset name
        imputation_strategy: Imputation strategy used
        no_retrain: Whether Shapley used masking (True) or retraining (False)
        retrain_kpi: Whether KPI used retraining
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()

    # Determine maximum number of metrics across all methods
    max_metrics = max(len(result['Metrics']) for result in kpi_results.values())

    # Create DataFrame
    data = {'Method': [], 'Weighted Slope (S)': []}

    # Add metric columns
    for i in range(max_metrics):
        data[f'Metric_{i}'] = []

    # Populate data
    for method in sorted(kpi_results.keys()):
        result = kpi_results[method]
        data['Method'].append(method)
        data['Weighted Slope (S)'].append(result['Slope'])

        # Add metrics, padding with empty strings if needed
        metrics = result['Metrics']
        for i in range(max_metrics):
            if i < len(metrics):
                data[f'Metric_{i}'].append(metrics[i])
            else:
                data[f'Metric_{i}'].append('')

    df = pd.DataFrame(data)

    # Generate filename
    shapley_mode = 'mask' if no_retrain else 'retrain'
    kpi_mode = 'retrain' if retrain_kpi else 'mask'
    csv_filename = f"kpi_metrics_{dataset}_{imputation_strategy}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    csv_path = output_dir / csv_filename

    # Save CSV
    df.to_csv(csv_path, index=False)

    print(f"Saved KPI metrics CSV: {csv_path}")
    return str(csv_path)


def _print_time_comparison_table(time_results):
    """
    Print execution time comparison table to terminal.

    Args:
        time_results: Dict mapping method names to execution times (seconds)
    """
    print("\n" + "=" * 80)
    print("EXECUTION TIME COMPARISON")
    print("=" * 80)

    # Sort by time
    sorted_times = sorted(time_results.items(), key=lambda x: x[1])

    # Print header
    print(f"{'Method':<30} {'Time (seconds)':<15} {'Relative Speed':<15}")
    print("-" * 80)

    # Get fastest time for relative comparison
    fastest_time = sorted_times[0][1] if sorted_times else 1.0

    # Print rows
    for method, exec_time in sorted_times:
        relative = exec_time / fastest_time
        print(f"{method:<30} {exec_time:>14.2f}s {relative:>14.1f}x")

    print("=" * 80)


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


# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================

def _get_cache_filename(reader, imputation_strategy='mean', no_retrain=True, retrain_kpi=False):
    """
    Generate cache filename based on dataset name and configuration.

    Args:
        reader: Dataset reader function
        imputation_strategy: Imputation strategy used
        no_retrain: If True, Shapley uses masking. If False, uses retraining.
        retrain_kpi: Whether KPI uses retraining

    Returns:
        Full path to cache file in output directory
    """
    dataset_name = reader.__name__.replace('_data_reader', '')

    # Shapley mode: mask or retrain
    shapley_mode = 'mask' if no_retrain else 'retrain'

    # KPI mode: mask or retrain
    kpi_mode = 'retrain' if retrain_kpi else 'mask'

    # Save to output directory
    output_dir = _ensure_output_dir()
    cache_filename = f"{dataset_name}_shapley_cache_{imputation_strategy}_{shapley_mode}_{kpi_mode}.pkl"

    return str(output_dir / cache_filename)


def _load_cache(cache_file):
    """Load cached results from file."""
    if not os.path.exists(cache_file):
        return None

    print(f"\n" + "=" * 60)
    print(f"Loading all cached results from {cache_file}...")
    print("=" * 60)

    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)

    print(f"Successfully loaded cached results!")
    print(f"  - Shapley methods: {len(cached_data.get('time_results', {}))}")
    print(f"  - KPI results: {'Yes' if cached_data.get('kpi_results') else 'No'}")
    print(f"  - Example time (ShapG): {cached_data.get('time_results', {}).get('ShapG', 0):.2f}s")

    return cached_data


def _save_cache(cache_file, **data):
    """Save results to cache file."""
    print("\n" + "=" * 60)
    print("Saving all results to unified cache...")
    print("=" * 60)

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved all results (Shapley values, timing, and KPI) to {cache_file}")


# ==============================================================================
# CHARACTERISTIC FUNCTION
# ==============================================================================

def _train_model_once(X, y, model):
    """
    Train model once for use in characteristic function.

    Args:
        X: Feature dataframe
        y: Target values
        model: Pre-configured model (LGBMRegressor or LGBMClassifier)

    Returns:
        Tuple of (model, X_train, X_test, y_test)
    """
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


def _create_characteristic_function(model, X_train, X_test, y_test, X_full, y_full,
                                   use_masking=True, imputation_strategy='mean'):
    """
    Create characteristic function for regression or classification task.

    Args:
        model: Pre-trained model (TRAINED ONCE!)
        X_train: Training features (for computing baseline statistics)
        X_test: Test features for evaluation
        y_test: Test labels for evaluation
        X_full: Full dataset (for retraining mode only)
        y_full: Full labels (for retraining mode only)
        use_masking: If True, use efficient masking approach (recommended).
                     If False, use retraining approach (slower, for comparison).
        imputation_strategy: Strategy for masking ('mean', 'zero', 'permutation')

    Returns:
        CharacteristicFunction instance
    """
    # Determine if this is a classification or regression task
    is_classifier = isinstance(model, lgb.LGBMClassifier)
    metric_fn = accuracy_score if is_classifier else r2_score
    task_type = "Classification" if is_classifier else "Regression"
    metric_name = "Accuracy" if is_classifier else "R2"

    if use_masking:
        # FAST: Use pre-trained model with masking for coalitions
        print(f"Using efficient masking approach with '{imputation_strategy}' imputation")

        # ✓ CRITICAL: Compute baseline from TRAINING set, not test set!
        baseline = X_train.mean().values

        return GraphModelCharacteristic(
            model=model,  # Pre-trained model passed as parameter
            X=X_test,     # Evaluate on test set
            y=y_test,
            masking_strategy=imputation_strategy,  # Use specified strategy
            metric_fn=metric_fn,
            baseline=baseline,  # ✓ Use training set statistics!
            name=f"{task_type} {metric_name} (Masking-{imputation_strategy})"
        )
    else:
        # SLOW: Retrain model for each coalition (original approach)
        print("Using retraining approach (slower, for comparison)")

        def characteristic_function_wrapper(coalition: Set[int], context: nx.Graph) -> float:
            """Compute metric score for coalition of features."""
            if len(coalition) == 0:
                return 0

            # Coalition contains column names (strings) from graph nodes
            cols = list(coalition)
            X_subset = X_full[cols]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_subset, y_full, test_size=0.2, random_state=42
            )

            if is_classifier:
                lgb_model = lgb.LGBMClassifier(learning_rate=0.3, verbosity=-1, device='mps')
            else:
                lgb_model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1, device='mps')

            lgb_model.fit(X_tr, y_tr)
            y_pred = lgb_model.predict(X_te)
            return metric_fn(y_te, y_pred)

        return CustomFunction(characteristic_function_wrapper, name=f"{task_type} {metric_name} (Retraining)")


# ==============================================================================
# EXPLAINER COMPUTATION
# ==============================================================================

def _compute_all_explainers(G, custom_char_func, X):
    """
    Compute Shapley values using all available explainers.

    Returns:
        Tuple of (all_values_dict, time_results_dict)
    """
    time_results = {}
    all_values = {}

    # 1. ShapG
    print("\nComputing Shapley values using ShapGExplainer...")
    start_time = time.time()
    shapg_explainer = ShapGExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **SHAPG_CONFIG
    )
    all_values['shapley'] = shapg_explainer.fit_explain(G)
    time_results['ShapG'] = time.time() - start_time
    print(f"  Time: {time_results['ShapG']:.2f}s")

    # 2. CIS
    print("\nComputing CIS values...")
    start_time = time.time()
    cis_explainer = CISExplainer(
        characteristic_function=custom_char_func,
        verbose=False
    )
    all_values['cis'] = cis_explainer.fit_explain(G)
    time_results['CIS'] = time.time() - start_time
    print(f"  Time: {time_results['CIS']:.2f}s")

    # 3. Random CS
    print("\nComputing Random CS Shapley values...")
    start_time = time.time()
    random_cs_explainer = RandomCSExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **RANDOM_CS_CONFIG
    )
    all_values['random_cs'] = random_cs_explainer.fit_explain(G)
    time_results['RandomCS'] = time.time() - start_time
    print(f"  Time: {time_results['RandomCS']:.2f}s")

    # 4. QR-CS
    print("\nComputing QR-CS Shapley values...")
    print("Using CVXPY for L1 minimization (much faster than scipy)")
    start_time = time.time()
    n_measurements = min(100, 2**(len(X.columns)-1) // 4)
    qrcs_explainer = QRCSExplainer(
        characteristic_function=custom_char_func,
        n_measurements=n_measurements,
        use_fast_fallback=False,
        verbose=True
    )
    all_values['qrcs'] = qrcs_explainer.fit_explain(G)
    time_results['QR-CS'] = time.time() - start_time
    print(f"  Time: {time_results['QR-CS']:.2f}s")

    # 5. Block QR-CS
    print("\nComputing Block QR-CS Shapley values...")
    print("Using parallel block computation for scalability")
    start_time = time.time()
    n_blocks = min(3, max(2, len(X.columns) // 5))
    block_qrcs_explainer = BlockQRCSExplainer(
        characteristic_function=custom_char_func,
        n_blocks=n_blocks,
        parallel=False,
        use_fast_fallback=False,
        verbose=True
    )
    all_values['block_qrcs'] = block_qrcs_explainer.fit_explain(G)
    time_results['BlockQRCS'] = time.time() - start_time
    print(f"  Time: {time_results['BlockQRCS']:.2f}s")

    # 6. Improved QR-CS
    print("\nComputing Improved QR-CS Shapley values (adaptive)...")
    print("Automatically detecting sparsity and adapting method")
    start_time = time.time()
    improved_qrcs_explainer = ImprovedQRCSExplainer(
        characteristic_function=custom_char_func,
        n_measurements=n_measurements,
        verbose=True,
        **IMPROVED_QRCS_CONFIG
    )
    all_values['improved_qrcs'] = improved_qrcs_explainer.fit_explain(G)
    time_results['ImprovedQRCS'] = time.time() - start_time

    sparsity_info = improved_qrcs_explainer.get_sparsity_info()
    print(f"  Detected sparsity: {sparsity_info['detected_sparsity']*100:.1f}%")
    print(f"  Using method: {sparsity_info['method']}")
    print(f"  Time: {time_results['ImprovedQRCS']:.2f}s")

    # 7. Improved Block QR-CS
    print("\nComputing Improved Block QR-CS Shapley values (per-block adaptive)...")
    print("Each block independently detects sparsity and adapts method")
    start_time = time.time()
    improved_block_qrcs_explainer = ImprovedBlockQRCSExplainer(
        characteristic_function=custom_char_func,
        n_blocks=n_blocks,
        parallel=True,
        verbose=True,
        **IMPROVED_QRCS_CONFIG
    )
    all_values['improved_block_qrcs'] = improved_block_qrcs_explainer.fit_explain(G)
    time_results['ImprovedBlockQRCS'] = time.time() - start_time

    block_info = improved_block_qrcs_explainer.get_block_sparsity_info()
    if 'avg_sparsity' in block_info:
        print(f"\nBlock sparsity statistics:")
        print(f"  Average sparsity: {block_info['avg_sparsity']*100:.1f}%")
        print(f"  Blocks using CS: {block_info['blocks_using_cs']}/{block_info['n_blocks']}")
        print(f"  Blocks using fast: {block_info['blocks_using_fast']}/{block_info['n_blocks']}")
    print(f"  Time: {time_results['ImprovedBlockQRCS']:.2f}s")

    # 8. RP-QRCS (Random Projection QRCS with Stratified Sampling)
    print("\nComputing RP-QRCS Shapley values (stratified sampling)...")
    print("Using stratified Shapley-weighted sampling + random projections")
    print("This fixes the coalition size bias in original QRCS")
    start_time = time.time()
    stratified_explainer = StratifiedShapleyExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **RP_QRCS_CONFIG
    )
    all_values['stratified'] = stratified_explainer.fit_explain(G)
    time_results['RP-QRCS'] = time.time() - start_time

    # Print sampling statistics
    rp_stats = stratified_explainer.get_sampling_stats()
    if rp_stats:
        print(f"\nRP-QRCS Sampling Statistics:")
        print(f"  Total samples: {rp_stats.total_budget}")
        nonzero_strata = sum(1 for a in rp_stats.allocations_by_size if a > 0)
        print(f"  Strata with samples: {nonzero_strata}/{rp_stats.n_players}")
    print(f"  Time: {time_results['RP-QRCS']:.2f}s")

    # 9. Leverage SHAP
    print("\nComputing Leverage SHAP values (ICLR 2025)...")
    print("Using leverage score sampling with provable O(n log n) guarantees")
    start_time = time.time()
    leverage_explainer = LeverageScoreExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **LEVERAGE_SHAP_CONFIG
    )
    all_values['leverage_shap'] = leverage_explainer.fit_explain(G)
    time_results['LeverageSHAP'] = time.time() - start_time
    print(f"  Time: {time_results['LeverageSHAP']:.2f}s")
    print("  Achieved ~50% error reduction compared to Kernel SHAP (based on paper)")

    # 10. Multilinear Extension with Leverage Sampling (Owen 1972 + Musco & Witter 2025)
    print("\nComputing Multilinear Extension Shapley values (with leverage sampling)...")
    print("Using Owen's multilinear extension + leverage-stratified sampling (LEM)")
    start_time = time.time()
    multilinear_explainer = MultilinearExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **MULTILINEAR_CONFIG
    )
    all_values['multilinear'] = multilinear_explainer.fit_explain(G)
    time_results['Multilinear-LEM'] = time.time() - start_time

    # Get computation statistics
    ml_stats = multilinear_explainer.get_computation_stats()
    print(f"  Method used: {ml_stats.get('method_used', 'N/A')}")
    if ml_stats.get('use_leverage'):
        print(f"  Leverage efficiency: {ml_stats.get('leverage_efficiency', 1.0):.2f}x")
    error_bounds = multilinear_explainer.get_error_bounds()
    if error_bounds:
        print(f"  Error bound: {error_bounds.total_error:.2e} (conf={error_bounds.confidence_level:.0%})")
    print(f"  Time: {time_results['Multilinear-LEM']:.2f}s")

    # 11. Multilinear Extension with Naive Sampling (Owen 1972 only)
    print("\nComputing Multilinear Extension Shapley values (naive sampling)...")
    print("Using Owen's multilinear extension + naive Bernoulli sampling")
    start_time = time.time()
    multilinear_naive_config = {k: v for k, v in MULTILINEAR_CONFIG.items()}
    multilinear_naive_config['use_leverage'] = False
    multilinear_naive_config['compute_error_bounds'] = False
    multilinear_naive_explainer = MultilinearExplainer(
        characteristic_function=custom_char_func,
        verbose=True,
        **multilinear_naive_config
    )
    all_values['multilinear_naive'] = multilinear_naive_explainer.fit_explain(G)
    time_results['Multilinear-Naive'] = time.time() - start_time

    ml_naive_stats = multilinear_naive_explainer.get_computation_stats()
    print(f"  Method used: {ml_naive_stats.get('method_used', 'N/A')}")
    print(f"  Time: {time_results['Multilinear-Naive']:.2f}s")

    return all_values, time_results


def _convert_to_feature_rankings(all_values, X, model, y):
    """
    Convert Shapley value dictionaries to ranked feature lists.

    Returns:
        Dictionary mapping method names to ranked feature lists
    """
    feature_rankings = {}

    # Standard methods
    methods_mapping = {
        'ShapG': 'shapley',
        'CIS': 'cis',
        'RandomCS': 'random_cs',
        'QR-CS': 'qrcs',
        'BlockQRCS': 'block_qrcs',
        'ImprovedQRCS': 'improved_qrcs',
        'ImprovedBlockQRCS': 'improved_block_qrcs',
        'RP-QRCS': 'stratified',
        'LeverageSHAP': 'leverage_shap',
        'Multilinear-LEM': 'multilinear',
        'Multilinear-Naive': 'multilinear_naive',
    }

    for display_name, key in methods_mapping.items():
        if key in all_values:
            sorted_values = sorted(all_values[key].items(), key=lambda x: x[1], reverse=True)
            feature_rankings[display_name] = [
                node_to_feature_name(node, X.columns)
                for node, _ in sorted_values
                if node_to_feature_name(node, X.columns) is not None
            ]

    # Add model feature importances if available
    if model and hasattr(model, 'feature_importances_'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[::-1]
        feature_rankings['Model'] = [X.columns[i] for i in feature_indices]

    return feature_rankings


def _test_improved_graphs(X, y, custom_char_func, feature_rankings, time_results):
    """Test improved graph construction methods."""
    print("\n" + "="*60)
    print("Testing improved graph construction methods...")
    print("="*60)

    builder = GraphBuilder()
    improved_shapley_values = {}

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
                verbose=False,
                **SHAPG_CONFIG
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

    return improved_shapley_values


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, dataset,
                                limit=10, cached_kpi_results=None,
                                no_retrain=True, retrain_kpi=False, imputation_strategy='mean'):
    """
    Plot comparison of feature importance methods using KPI metrics.

    Parameters:
        reader: Function to load dataset
        feature_rankings: Dict mapping method names to ranked feature lists
        model: ML model to evaluate
        dataset: Dataset name (e.g., 'housing', 'h1n1')
        limit: Number of top features to evaluate
        cached_kpi_results: Pre-computed results to avoid recalculation
        retrain_kpi: If True, retrain model for each feature drop (slow but accurate).
                     If False, use masking (fast, consistent with Shapley).
        imputation_strategy: Strategy for masking ('mean', 'zero', 'permutation')

    Returns:
        Tuple of (results_dict, saved_files_dict)
    """
    model_name = type(model).__name__

    # Use cached results if available
    if cached_kpi_results is not None:
        print("Using cached KPI results.")
        results = cached_kpi_results
    else:
        mode_desc = "retraining" if retrain_kpi else f"masking ({imputation_strategy})"
        print(f"Computing KPI results (no cache) using {mode_desc}...")
        results = _compute_kpi_results(reader, feature_rankings, model, limit,
                                      retrain_kpi=retrain_kpi,
                                      imputation_strategy=imputation_strategy)

    # Create plot
    plt.figure(figsize=PLOT_FIGSIZE)
    metric_name = _get_metric_name(model)

    for method, data in results.items():
        label = f'{method} $S$={data["Slope"]:.4f}'
        plt.plot(
            range(len(data['Metrics'])),
            data['Metrics'],
            label=label,
            alpha=PLOT_ALPHA
        )

    plt.xlabel('Number of Features Dropped')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} after dropping features (New API - {model_name})')
    plt.legend()
    plt.grid()

    # Generate filename and save
    base_filename = _generate_filename('KPI', dataset, imputation_strategy, no_retrain, retrain_kpi)
    saved_files = _save_plot(base_filename)

    return results, saved_files


def plot_time_comparison(time_results, dataset, imputation_strategy, no_retrain, retrain_kpi):
    """
    Plot horizontal bar chart comparing algorithm execution times.

    Parameters:
        time_results: Dict mapping algorithm names to execution times (seconds)
        dataset: Dataset name (e.g., 'housing', 'h1n1')
        imputation_strategy: Imputation strategy used
        retrain_kpi: Whether KPI used retraining

    Returns:
        Tuple of (sorted_items, saved_files_dict)
    """
    # Sort by time for better visualization
    sorted_items = sorted(time_results.items(), key=lambda x: x[1])
    algorithms = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]

    # Create horizontal bar chart
    plt.figure(figsize=PLOT_FIGSIZE)
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

    # Generate filename and save
    base_filename = _generate_filename('time_comparison', dataset, imputation_strategy, no_retrain, retrain_kpi)
    saved_files = _save_plot(base_filename)

    return sorted_items, saved_files


# ==============================================================================
# MAIN BENCHMARK FUNCTION
# ==============================================================================

def benchmark_feature_importance(reader, model, dataset, limit=10,
                                 test_improved_graphs=True, use_cache=True,
                                 no_retrain=True, retrain_kpi=False,
                                 imputation_strategy='mean'):
    """
    Benchmark feature importance using NEW modular API.

    This function demonstrates:
    - Using GraphBuilder for graph construction
    - Using different Explainer classes
    - Using CustomFunction for characteristic functions
    - Caching and visualization

    Parameters:
        reader: Function to read the dataset
        model: Machine learning model
        filename: Output filename for plot
        limit: Number of top features to consider
        test_improved_graphs: If True, test improved graph construction
        use_cache: If True, load/save cached results
        no_retrain: If True, use efficient masking (recommended, ~100x faster).
                    If False, retrain model for each coalition (slower).
        retrain_kpi: If True, use retraining for KPI plots (slow but measures true
                     feature importance). If False, use masking (faster, consistent).
        imputation_strategy: Strategy for handling masked features ('mean', 'zero', 'permutation').

    Returns:
        Tuple of (shapley_values, cis_values, random_cs_values, qrcs_values,
                 block_qrcs_values, improved_qrcs_values, improved_block_qrcs_values,
                 stratified_values, leverage_shap_values, improved_shapley_values,
                 time_results, kpi_results)
    """
    # Load data
    X, y = reader()

    # Check cache (use configuration-specific cache)
    cache_file = _get_cache_filename(reader, imputation_strategy, no_retrain, retrain_kpi)
    cached_data = _load_cache(cache_file) if use_cache else None

    if cached_data:
        # Unpack cached results
        all_values = {
            'shapley': cached_data['shapley_values'],
            'cis': cached_data['cis_values'],
            'random_cs': cached_data['random_cs_values'],
            'qrcs': cached_data['qrcs_values'],
            'block_qrcs': cached_data['block_qrcs_values'],
            'improved_qrcs': cached_data['improved_qrcs_values'],
            'improved_block_qrcs': cached_data['improved_block_qrcs_values'],
            'stratified': cached_data.get('stratified_values', {}),
            'leverage_shap': cached_data.get('leverage_shap_values', {}),
        }
        time_results = cached_data['time_results']
        improved_shapley_values = cached_data.get('improved_shapley_values', {})
        cached_kpi_results = cached_data.get('kpi_results', None)
    else:
        # Train model ONCE (outside characteristic function)
        print("\n" + "=" * 60)
        print("STEP 1: Train model once")
        print("=" * 60)
        trained_model, X_train, X_test, y_test = _train_model_once(X, y, model)

        # Build graph
        print("\n" + "=" * 60)
        print("STEP 2: Build graph")
        print("=" * 60)
        builder = GraphBuilder()
        G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
        print(f"Using original graph construction (kendalltau + minimal edge graph)")
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Create characteristic function (using pre-trained model)
        print("\n" + "=" * 60)
        print("STEP 3: Create characteristic function")
        print("=" * 60)
        custom_char_func = _create_characteristic_function(
            model=trained_model,  # Pre-trained model
            X_train=X_train,      # For baseline statistics
            X_test=X_test,
            y_test=y_test,
            X_full=X,
            y_full=y,
            use_masking=no_retrain,
            imputation_strategy=imputation_strategy
        )

        # Compute all explainers
        print("\n" + "=" * 60)
        print("STEP 4: Compute Shapley values using all explainers")
        print("=" * 60)
        all_values, time_results = _compute_all_explainers(G, custom_char_func, X)
        improved_shapley_values = {}
        cached_kpi_results = None

    # Convert to feature rankings
    feature_rankings = _convert_to_feature_rankings(all_values, X, model, y)

    # Test improved graphs if requested
    # NOTE: Always compute fresh (don't use cached) to ensure correct baseline
    if test_improved_graphs:
        # Train model if we only loaded from cache
        if cached_data:
            trained_model, X_train, X_test, y_test = _train_model_once(X, y, model)

        # Create characteristic function with training set baseline
        custom_char_func = _create_characteristic_function(
            model=trained_model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            X_full=X,
            y_full=y,
            use_masking=no_retrain,
            imputation_strategy=imputation_strategy
        )
        improved_shapley_values = _test_improved_graphs(
            X, y, custom_char_func, feature_rankings, time_results
        )

    # Export results to CSV
    print("\n" + "=" * 60)
    print("Exporting results...")
    print("=" * 60)
    _export_feature_rankings_csv(feature_rankings, dataset, imputation_strategy, no_retrain, retrain_kpi)
    _export_time_comparison_csv(time_results, dataset, imputation_strategy, no_retrain, retrain_kpi)

    # Print feature ranking table
    _print_feature_ranking_table(feature_rankings, limit=limit)

    # Print time comparison table
    _print_time_comparison_table(time_results)

    # Plot KPI comparison
    print("\n" + "=" * 60)
    print("Generating KPI comparison plot...")
    print("=" * 60)
    kpi_results, kpi_saved_files = plot_KPI_comparison_by_dict(
        reader, feature_rankings, model, dataset, limit, cached_kpi_results,
        retrain_kpi=retrain_kpi, imputation_strategy=imputation_strategy, no_retrain=no_retrain
    )

    # Export KPI metrics to CSV
    _export_kpi_metrics_csv(kpi_results, dataset, imputation_strategy, no_retrain, retrain_kpi)

    # Save cache if needed
    if use_cache and not cached_data:
        _save_cache(
            cache_file,
            shapley_values=all_values['shapley'],
            cis_values=all_values['cis'],
            random_cs_values=all_values['random_cs'],
            qrcs_values=all_values['qrcs'],
            block_qrcs_values=all_values['block_qrcs'],
            improved_qrcs_values=all_values['improved_qrcs'],
            improved_block_qrcs_values=all_values['improved_block_qrcs'],
            stratified_values=all_values['stratified'],
            leverage_shap_values=all_values['leverage_shap'],
            improved_shapley_values=improved_shapley_values,
            time_results=time_results,
            kpi_results=kpi_results
        )

    # Print time summary
    print("\n" + "=" * 60)
    print("Execution Time Summary:")
    print("=" * 60)
    for method, exec_time in sorted(time_results.items(), key=lambda x: x[1]):
        print(f"  {method}: {exec_time:.2f}s")

    # Return all results
    return (
        all_values['shapley'],
        all_values['cis'],
        all_values['random_cs'],
        all_values['qrcs'],
        all_values['block_qrcs'],
        all_values['improved_qrcs'],
        all_values['improved_block_qrcs'],
        all_values['stratified'],
        all_values['leverage_shap'],
        improved_shapley_values,
        time_results,
        kpi_results
    )


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Benchmark Shapley value computation methods with new modular API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use efficient masking (default, recommended - ~100x faster)
  python benchmark_qr_new_api.py --no-retrain

  # Use retraining approach (slower, for comparison)
  python benchmark_qr_new_api.py --retrain

  # Disable cache
  python benchmark_qr_new_api.py --no-cache

  # Disable improved graph testing
  python benchmark_qr_new_api.py --no-improved-graphs
        """
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
    parser.add_argument(
        '--no-improved-graphs',
        action='store_true',
        help='Skip improved graph construction tests'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of top features to evaluate (default: 10)'
    )
    parser.add_argument(
        '--dataset',
        choices=['housing', 'h1n1'],
        default='housing',
        help='Dataset to use (default: housing)'
    )
    parser.add_argument(
        '--retrain-kpi',
        action='store_true',
        help='Use retraining (not masking) for KPI plots. '
             'Slower but measures true feature importance for model quality. '
             'Default is to use masking (consistent with Shapley computation).'
    )
    parser.add_argument(
        '--imputation',
        choices=['mean', 'zero', 'permutation'],
        default='mean',
        help='Strategy for handling masked features. '
             'mean: replace with training set mean (default). '
             'zero: replace with 0. '
             'permutation: randomly permute values.'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BENCHMARK WITH NEW MODULAR API")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Shapley mode: {'Retraining (slow)' if args.retrain else 'Masking (fast, recommended)'}")
    print(f"  KPI mode: {'Retraining' if args.retrain_kpi else 'Masking (consistent with Shapley)'}")
    print(f"  Imputation strategy: {args.imputation}")
    print(f"  Cache: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"  Improved graphs: {'Disabled' if args.no_improved_graphs else 'Enabled'}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Top features: {args.limit}")

    # Select dataset reader and appropriate model
    if args.dataset == 'housing':
        reader = housing_data_reader
        model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
        print(f"  Task type: Regression (using LGBMRegressor, metric: R²)")
    else:  # h1n1
        reader = h1n1_data_reader
        model = lgb.LGBMClassifier(learning_rate=0.3, verbosity=-1)
        print(f"  Task type: Classification (using LGBMClassifier, metric: Accuracy)")

    print("\nRunning benchmark...")
    (shapley_values, cis_values, random_cs_values, qrcs_values,
     block_qrcs_values, improved_qrcs_values, improved_block_qrcs_values,
     stratified_values, leverage_shap_values, improved_shapley_values,
     time_results, results) = benchmark_feature_importance(
        reader,
        model,
        dataset=args.dataset,
        limit=args.limit,
        test_improved_graphs=not args.no_improved_graphs,
        use_cache=not args.no_cache,
        no_retrain=not args.retrain,  # Default is True (use masking)
        retrain_kpi=args.retrain_kpi,
        imputation_strategy=args.imputation
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
    print("RP-QRCS values:", stratified_values)
    print("Leverage SHAP values:", leverage_shap_values)
    if improved_shapley_values:
        print("\nImproved graph Shapley values:")
        for method, values in improved_shapley_values.items():
            print(f"  {method}:", values)

    # Generate time comparison plot
    print("\n" + "=" * 60)
    print("Generating time comparison visualization...")
    print("=" * 60)
    sorted_times, time_saved_files = plot_time_comparison(
        time_results, args.dataset, args.imputation, not args.retrain, args.retrain_kpi
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    if not args.retrain:
        print("\n✓ Used efficient masking approach (~100x faster than retraining)")
        print("  To compare with retraining, run: python benchmark_qr_new_api.py --retrain")
    else:
        print("\n⚠ Used retraining approach (slow)")
        print("  For faster execution, run: python benchmark_qr_new_api.py (default)")
    print()
