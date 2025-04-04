import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import kendalltau, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))

from shapG.shapley import shapG, cis
import shapG.plot as shapGplot
from shapG.utils import corr_generator, create_minimal_edge_graph, matrix_generator, kl, kl_mi_matrix

# Data readers
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

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=7):
    """
    Plot the comparison of KPIs for different feature selection methods.

    Parameters:
    - reader: Function to read the dataset.
    - feature_rankings: Dictionary where keys are method names and values are lists of features in order of importance.
    - model: The machine learning model to use (LGBM or MLP).
    - filename: File name to save the plot.
    - limit: Maximum number of features to consider.

    Returns:
    - Dictionary containing results for each method.
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
    results_file = f"{model_name}_dict_results.pkl"

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
            # Make sure feature_order contains only column names as strings
            feature_order = [feat if isinstance(feat, str) else feat[0] for feat in feature_order]
            
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
    # plt.show()
    
    return results

def benchmark_feature_importance(reader, model, filename=None, limit=7):
    """
    Benchmark feature importance using different methods.

    Parameters:
    - reader: Function to read the dataset.
    - model: The machine learning model to use (LGBM or MLP).
    - filename: File name to save the plot.
    - limit: Maximum number of features to consider.
    """
    X, y = reader()
    W = matrix_generator(X)
    A, W_new = create_minimal_edge_graph(W, reverse=True, version='v3')
    G = nx.Graph(A)

    # Compute Shapley values
    shapley_values = shapG(G, m=3, f=lambda G, S: classification_kpi(X, y, S), approximate_by_ratio=False, scale=False)
    cis_values = cis(G, f=lambda G, S: classification_kpi(X, y, S))
    
    # Convert to sorted feature lists for plot_KPI_comparison_by_dict
    feature_rankings = {}
    
    # Add shapG values - ensure we map node IDs to actual column names
    sorted_shapley = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['shapG'] = []
    for node, value in sorted_shapley:
        # Convert node ID to integer index
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                feature_rankings['shapG'].append(X.columns[idx])
        except (ValueError, TypeError):
            # If node isn't a valid integer, use it directly if it's a column name
            if node in X.columns:
                feature_rankings['shapG'].append(node)
    
    # Add CIS values
    sorted_cis = sorted(cis_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['CIS'] = []
    for node, value in sorted_cis:
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                feature_rankings['CIS'].append(X.columns[idx])
        except (ValueError, TypeError):
            if node in X.columns:
                feature_rankings['CIS'].append(node)
    
    # Add model feature importances if available
    if hasattr(model, 'feature_importances_'):
        # Train the model to get feature importances
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        feature_indices = np.argsort(importances)[::-1]
        feature_rankings['Model'] = [X.columns[i] for i in feature_indices]
    
    # Plot the comparison
    results = plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename, limit)
    
    return shapley_values, cis_values, results
# Classification KPI
def classification_kpi(X, y, S):
    cols = list(S)
    if len(cols) == 0:
        return 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)

if __name__ == "__main__":
    # Example usage
    model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
    shapley_values, cis_values, results = benchmark_feature_importance(housing_data_reader, model, filename='housing_benchmark.png')
    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)
