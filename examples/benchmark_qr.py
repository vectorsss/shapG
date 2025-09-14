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
from typing import Callable, Optional, Tuple
from scipy.optimize import minimize
from scipy.linalg import qr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))

from shapG import shapG, cis
from shapG.visualization import plot as shapGplot
from shapG import corr_generator, create_minimal_edge_graph, matrix_generator, kl, kl_mi_matrix

# ==============================================================================
# QR-CS Shapley Implementation (NEW)
# ==============================================================================

class QRCompressedSensingShapley:
    """
    QR-CS Shapley value approximation based on:
    "A novel sparsity-based deterministic method for Shapley value approximation"
    """
    
    def __init__(self, n_features: int, n_measurements: Optional[int] = None, 
                 tolerance: float = 5e-5):
        self.n = n_features
        self.m = min(2**(n_features - 1), 5000)  # Cap for memory
        
        if n_measurements is None:
            self.l = min(int(2 * n_features * np.log(max(n_features, 2))), self.m // 2, 500)
        else:
            self.l = min(n_measurements, self.m)
            
        self.tolerance = tolerance
        
        # Pre-compute components
        self.weights = self._compute_shapley_weights()
        self.Psi = self._get_dct_basis()
        self.B, self.selected_coalitions = self._compute_measurement_matrix()
        
    def _compute_shapley_weights(self) -> np.ndarray:
        """Compute Shapley weights"""
        weights = []
        for s in range(min(self.n, 20)):
            weight = math.factorial(s) * math.factorial(self.n - s - 1) / math.factorial(self.n)
            n_coalitions = self._comb(self.n - 1, s)
            weights.extend([weight] * min(n_coalitions, self.m - len(weights)))
            if len(weights) >= self.m:
                break
        return np.array(weights[:self.m])
    
    def _comb(self, n: int, k: int) -> int:
        """Binomial coefficient"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c
    
    def _get_dct_basis(self) -> np.ndarray:
        """DCT-II basis"""
        if self.m > 1000:
            return np.eye(self.m)
            
        Psi = np.zeros((self.m, self.m))
        for k in range(self.m):
            for n in range(self.m):
                if k == 0:
                    Psi[n, k] = np.sqrt(1/self.m)
                else:
                    Psi[n, k] = np.sqrt(2/self.m) * np.cos(np.pi * k * (n + 0.5) / self.m)
        return Psi
    
    def _compute_measurement_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """QR decomposition for measurement matrix"""
        V = self.Psi.T
        Q, R, P = qr(V, pivoting=True, mode='economic' if self.m > 1000 else 'full')
        
        selected_indices = P[:self.l]
        
        B = np.zeros((self.l, self.m))
        for i, idx in enumerate(selected_indices):
            B[i, idx] = 1
            
        return B, selected_indices
    
    def _index_to_coalition(self, idx: int, player: int) -> set:
        """Convert index to coalition"""
        coalition = set()
        available_players = [i for i in range(self.n) if i != player]
        
        if idx >= 2**(self.n - 1):
            idx = idx % 2**(self.n - 1)
        
        for i, p in enumerate(available_players):
            if idx & (1 << i):
                coalition.add(p)
        
        return coalition
    
    def compute_shapley(self, utility_func: Callable) -> dict:
        """Compute Shapley values"""
        shapley_values = {}
        
        for player in range(self.n):
            # Measure marginal contributions
            y = np.zeros(self.l)
            
            for i, coal_idx in enumerate(self.selected_coalitions):
                coalition = self._index_to_coalition(coal_idx, player)
                
                v_with = utility_func(coalition | {player})
                v_without = utility_func(coalition) if coalition else 0
                y[i] = v_with - v_without
            
            # Compressed sensing reconstruction
            try:
                Q, _, _ = qr(self.Psi, mode='economic' if self.m > 1000 else 'full')
                Theta = self.B @ self.Psi @ Q
                
                # L1 minimization
                s_hat = self._l1_minimization(Theta, y)
                u_hat = self.Psi @ Q @ s_hat
                
                shapley_values[player] = np.dot(self.weights[:len(u_hat)], u_hat)
            except:
                shapley_values[player] = np.mean(y)
        
        return shapley_values
    
    def _l1_minimization(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L1 minimization"""
        m, n = A.shape
        
        try:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            x0 = np.zeros(n)
        
        def objective(x):
            return np.sum(np.abs(x))
        
        def constraint(x):
            return self.tolerance - np.linalg.norm(A @ x - b)
        
        constraints = {'type': 'ineq', 'fun': constraint}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
        
        return result.x if result.success else x0


def qrcs_shapley(G, f, n_measurements=None):
    """
    Compute QR-CS Shapley values for a graph
    
    Parameters:
    - G: NetworkX graph
    - f: utility function that takes (G, S) where S is a set of nodes
    - n_measurements: number of measurements (optional)
    
    Returns:
    - Dictionary with node: shapley_value pairs
    """
    n_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Create utility wrapper that works with node indices
    def utility_wrapper(S):
        node_set = {nodes[i] for i in S}
        return f(G, node_set)
    
    # Compute QR-CS Shapley values
    qrcs = QRCompressedSensingShapley(n_nodes, n_measurements)
    shapley_indices = qrcs.compute_shapley(utility_wrapper)
    
    # Map back to node names
    shapley_values = {nodes[i]: value for i, value in shapley_indices.items()}
    
    return shapley_values

# ==============================================================================
# Original functions (unchanged)
# ==============================================================================

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

def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=10):
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

def benchmark_feature_importance(reader, model, filename=None, limit=10):
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
    
    # Compute QR-CS Shapley values (NEW)
    print("Computing QR-CS Shapley values...")
    n_measurements = min(100, 2**(len(X.columns)-1) // 4)
    qrcs_values = qrcs_shapley(G, lambda G, S: classification_kpi(X, y, S), n_measurements=n_measurements)
    
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
    
    # Add QR-CS values (NEW)
    sorted_qrcs = sorted(qrcs_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['QR-CS'] = []
    for node, value in sorted_qrcs:
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                feature_rankings['QR-CS'].append(X.columns[idx])
        except (ValueError, TypeError):
            if node in X.columns:
                feature_rankings['QR-CS'].append(node)
    
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
    
    return shapley_values, cis_values, qrcs_values, results

# Classification KPI
def classification_kpi(X, y, S):
    cols = list(S)
    if len(cols) == 0:
        return 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1, device='cpu')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)

if __name__ == "__main__":
    # Example usage
    model = lgb.LGBMRegressor(learning_rate=0.3, verbosity=-1)
    shapley_values, cis_values, qrcs_values, results = benchmark_feature_importance(housing_data_reader, model, filename='housing_benchmark.png')
    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)
    print("QR-CS values:", qrcs_values)
