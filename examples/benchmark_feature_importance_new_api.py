"""
Benchmark for feature importance using the NEW modular ShapG API.
This exactly matches benchmark_feature_importance.py:
- Compares ShapG, CIS, and Model
- Uses the same classification_kpi characteristic function
- Produces the same graph with S values
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import NEW API components
from shapG import GraphBuilder, ShapGExplainer, CISExplainer, CustomFunction


# ==============================================================================
# Data readers (unchanged from original)
# ==============================================================================

def housing_data_reader(filename='./data/housing_price.csv'):
    """Read housing price dataset."""
    data = pd.read_csv(filename)
    X = data.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return X, y


def h1n1_data_reader(filename='./data/process_data.csv'):
    """Read H1N1 vaccine dataset."""
    data = pd.read_csv(filename)
    X = data.drop(['h1n1_vaccine', 'respondent_id', 'seasonal_vaccine'], axis=1)
    y = data['h1n1_vaccine']
    return X, y


# ==============================================================================
# Characteristic function (EXACTLY from original)
# ==============================================================================

def classification_kpi(X, y, S):
    """Original characteristic function from benchmark_feature_importance.py"""
    cols = list(S)
    if len(cols) == 0:
        return 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)


# ==============================================================================
# Plotting function (from original)
# ==============================================================================

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=7):
    """
    Plot the comparison of KPIs for different feature selection methods.
    Exactly from original benchmark_feature_importance.py.
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
    results_file = f"{model_name}_dict_results_new_api.pkl"

    # Load or calculate results
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results for {model_name} from disk.")
    else:
        X, y = reader()
        results = {}

        # Calculate initial metric (without dropping features)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        initial_metric = r2_score(y_test, y_pred) if isinstance(model, lgb.LGBMRegressor) else accuracy_score(y_test, y_pred)

        # Process each ranking method
        for method, feature_order in feature_rankings.items():
            # Ensure feature_order is a list of column names
            if isinstance(feature_order[0], tuple):
                feature_order = [feat[0] for feat in feature_order]

            if limit:
                feature_order = feature_order[:limit]

            metrics = [initial_metric]
            features = [[]]
            deltas = []

            for i in range(1, len(feature_order) + 1):
                features_to_drop = feature_order[:i]
                # Check if all features exist in dataframe
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

            # Calculate weighted slope for comparison
            beta = 0.8
            weight = [beta**i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weight) if deltas else 0
            }

        # Save results to disk
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results for {model_name} to disk.")

    # Create the plot
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
    plt.title(f'Comparison of {metric_name} after dropping features based on different XAI methods ({model_name})')
    plt.legend()
    plt.grid()

    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

    return results


# ==============================================================================
# Main benchmark function using NEW API
# ==============================================================================

def benchmark_feature_importance(reader, model, filename=None, limit=7):
    """
    Benchmark feature importance using different methods.
    Uses NEW API but produces exact same results as original.

    Parameters:
    - reader: Function to read the dataset.
    - model: The machine learning model to use (LGBM).
    - filename: File name to save the plot.
    - limit: Maximum number of features to consider.
    """
    X, y = reader()

    # Build graph using NEW API
    builder = GraphBuilder()
    G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')

    # Create characteristic function wrapper for new API
    def char_func_wrapper(coalition, context=None):
        return classification_kpi(X, y, coalition)

    char_func = CustomFunction(char_func_wrapper, name="classification_kpi")

    # Compute Shapley values using NEW API
    shapg_explainer = ShapGExplainer(
        characteristic_function=char_func,
        depth=1,
        n_samples=3,  # m=3 in original
        approximate_by_ratio=False,
        scale=False,
        verbose=False
    )
    shapley_values = shapg_explainer.fit_explain(G)

    # Compute CIS values using NEW API
    cis_explainer = CISExplainer(
        characteristic_function=char_func,
        verbose=False
    )
    cis_values = cis_explainer.fit_explain(G)

    # Convert to sorted feature lists for plot_KPI_comparison_by_dict
    feature_rankings = {}

    # Add shapG values
    sorted_shapley = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['shapG'] = []
    for node, value in sorted_shapley:
        # Nodes are column names already
        feature_rankings['shapG'].append(node)

    # Add CIS values
    sorted_cis = sorted(cis_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['CIS'] = []
    for node, value in sorted_cis:
        feature_rankings['CIS'].append(node)

    # Add model feature importances
    if hasattr(model, 'feature_importances_'):
        model.fit(X, y)
        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[::-1]
        feature_rankings['Model'] = [X.columns[i] for i in feature_indices]

    # Plot the comparison
    results = plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename, limit)

    return shapley_values, cis_values, results


# ==============================================================================
# Main execution
# ==============================================================================

if __name__ == "__main__":
    # Example usage (EXACTLY like original)
    model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
    shapley_values, cis_values, results = benchmark_feature_importance(
        housing_data_reader,
        model,
        filename='housing_benchmark_new_api.png'
    )
    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)