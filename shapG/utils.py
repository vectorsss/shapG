import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from typing import Callable, Union, Optional

def corr_generator(df: pd.DataFrame, method: Callable = kendalltau) -> pd.DataFrame:
    """Generate a correlation matrix of a dataframe using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (Callable, optional): Function to calculate correlation coefficients.
            Options: pearsonr, kendalltau, spearmanr. Defaults to kendalltau.

    Returns:
        pd.DataFrame: Correlation matrix.
        
    Raises:
        ValueError: If method is not one of the supported correlation methods.
    """
    if method not in [pearsonr, kendalltau, spearmanr]:
        raise ValueError("method should be pearsonr, kendalltau, or spearmanr")
    
    # Create empty correlation matrix
    corr_df = pd.DataFrame(
        np.zeros((df.shape[1], df.shape[1])), 
        columns=df.columns, 
        index=df.columns
    )
    
    # Calculate correlations for all column pairs
    for i, col1 in enumerate(df.columns):
        # Only need to calculate upper triangle due to symmetry
        for col2 in df.columns[i+1:]:
            corr, _ = method(df[col1], df[col2])
            corr_df.loc[col1, col2] = corr
            corr_df.loc[col2, col1] = corr  # Symmetry
            
    return corr_df

def matrix_generator(df: pd.DataFrame, method: Callable = kendalltau) -> pd.DataFrame:
    """Generate a similarity/distance matrix for a dataframe using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (Callable, optional): Function to calculate similarity or distance.
            Options: pearsonr, kendalltau, spearmanr, mutual_info_score, mutual_info_regression, kl.
            Defaults to kendalltau.

    Returns:
        pd.DataFrame: Similarity/distance matrix.
    """
    # Handle standard correlation methods
    if method in [pearsonr, kendalltau, spearmanr]:
        return corr_generator(df, method)
    
    # Handle mutual information for categorical variables
    elif method == mutual_info_score:
        # Check if columns appear to be categorical
        if df.apply(lambda x: len(x.unique())).max() > 10:
            raise ValueError("mutual_info_score is best suited for categorical data (columns with ≤10 unique values)")
        
        # Initialize matrix
        matrix_df = pd.DataFrame(
            np.zeros((df.shape[1], df.shape[1])), 
            columns=df.columns, 
            index=df.columns
        )
        
        # Calculate mutual information for all column pairs
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                mi = method(df[col1], df[col2])
                matrix_df.loc[col1, col2] = mi
                matrix_df.loc[col2, col1] = mi  # Symmetry
    
    # Handle mutual information regression
    elif method == mutual_info_regression:
        matrix_df = pd.DataFrame(
            np.zeros((df.shape[1], df.shape[1])), 
            columns=df.columns, 
            index=df.columns
        )
        
        for col1 in matrix_df.columns:
            for col2 in matrix_df.columns:
                if col1 != col2:
                    measures = method(df[[col1]], df[col2])
                    matrix_df.loc[col1, col2] = measures[0]
    
    # Handle other methods (including kl divergence)
    else:
        matrix_df = pd.DataFrame(
            np.zeros((df.shape[1], df.shape[1])), 
            columns=df.columns, 
            index=df.columns
        )
        
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if col1 != col2:
                    measure = method(df[col1], df[col2])
                    matrix_df.loc[col1, col2] = measure
                    # For symmetric measures, also set the opposite direction
                    if method != kl:  # KL divergence is not symmetric
                        matrix_df.loc[col2, col1] = measure
                    else:
                        # For KL, calculate the reverse direction separately
                        matrix_df.loc[col2, col1] = method(df[col2], df[col1])
    
    return matrix_df

def kl(P: np.ndarray, Q: np.ndarray) -> float:
    """Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        P (np.ndarray): First distribution.
        Q (np.ndarray): Second distribution.
        
    Returns:
        float: KL divergence from Q to P.
    """
    epsilon = 1e-10
    
    # Add epsilon to avoid log(0) and ensure proper normalization
    P = P + epsilon
    Q = Q + epsilon
    
    # Normalize to probability distributions
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Calculate KL divergence: sum(P(i) * log(P(i)/Q(i)))
    divergence = np.sum(P * np.log(P / Q))
    return divergence

def kl_mi_matrix(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Create a combined matrix of KL divergence and mutual information.
    
    This function creates a weighted adjacency matrix that combines pairwise
    KL divergences between features with mutual information between features and target.
    
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (np.ndarray): Target variable.
        
    Returns:
        pd.DataFrame: Combined KL and MI matrix.
    """
    # Generate KL divergence matrix
    W = matrix_generator(X, kl)
    
    # Calculate mutual information between features and target
    mi = mutual_info_regression(X, y)
    
    # Normalize matrices for consistent weighting
    W = W / np.sum(np.array(W))
    mi = mi / np.sum(mi)
    
    # Create a copy to avoid modifying the original
    W2 = W.copy()
    
    # Combine KL divergence and mutual information
    for i in range(len(mi)):
        # Add normalized MI to each row and column except diagonal
        mi_contribution = mi[i] / (len(mi) - 1)
        W2.iloc[i, :] += mi_contribution
        W2.iloc[:, i] += mi_contribution
        # Reset diagonal to zero (no self-connections)
        W2.iloc[i, i] = 0
        
    return W2

def create_minimal_edge_graph(
    W: pd.DataFrame, 
    version: str = 'v3', 
    reverse: bool = True, 
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a weight matrix to a minimal adjacency matrix that preserves connectivity.
    
    Args:
        W (pd.DataFrame): Weight matrix.
        version (str, optional): Algorithm version.
            - 'v1': Stop when all nodes are in the graph.
            - 'v2': Continue until the graph is connected.
            - 'v3': Ensure strong connectivity. Defaults to 'v3'.
        reverse (bool, optional): Sort order (True=descending, False=ascending). Defaults to True.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        
    Returns:
        tuple: (adjacency_matrix, reduced_weight_matrix)
    """
    columns = W.columns.tolist()
    
    # Create list of all edges with weights
    edges = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            edges.append((columns[i], columns[j], abs(W.iloc[i, j])))

    # Sort edges by weight
    edges.sort(key=lambda x: x[2], reverse=reverse)

    # Initialize output matrices and tracking set
    connected_nodes = set()
    adjacency_matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=np.int8)
    reduced_df = pd.DataFrame(0, index=columns, columns=columns, dtype=np.float64)
    
    # Helper function to check if graph is connected
    def is_graph_connected():
        G = nx.Graph(adjacency_matrix)
        return nx.is_connected(G)
    
    # Add edges according to selected algorithm version
    for edge in edges:
        node1, node2, weight = edge
        add_edge = False
        
        if version == 'v1':
            # V1: Add edge if either node is not yet in the graph
            if node1 not in connected_nodes or node2 not in connected_nodes:
                add_edge = True
                # If all nodes are in the graph after adding this edge, we're done
                if len(connected_nodes.union({node1, node2})) == len(columns):
                    if verbose:
                        print(f"v1 terminating at weight: {weight}")
                    add_edge = True
                    # Final edge to add
                    adjacency_matrix.loc[node1, node2] = adjacency_matrix.loc[node2, node1] = 1
                    reduced_df.loc[node1, node2] = reduced_df.loc[node2, node1] = weight
                    break
        
        elif version == 'v2':
            # V2: Add edge if either node is not yet in the graph
            if node1 not in connected_nodes or node2 not in connected_nodes:
                add_edge = True
            # If all nodes are in graph, add edges until connected
            elif len(connected_nodes) == len(columns) and not is_graph_connected():
                add_edge = True
            # If graph is connected with all nodes, we're done
            elif len(connected_nodes) == len(columns) and is_graph_connected():
                if verbose:
                    print(f"v2 terminating at weight: {weight}")
                break
        
        elif version == 'v3':
            # V3: Add all edges until the graph is connected with all nodes
            if not (len(connected_nodes) == len(columns) and is_graph_connected()):
                add_edge = True
            else:
                if verbose:
                    print(f"v3 terminating at weight: {weight}")
                break
        
        # Add the edge if needed
        if add_edge:
            adjacency_matrix.loc[node1, node2] = adjacency_matrix.loc[node2, node1] = 1
            reduced_df.loc[node1, node2] = reduced_df.loc[node2, node1] = weight
            connected_nodes.update([node1, node2])

    return adjacency_matrix, reduced_df