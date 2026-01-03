import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from typing import Callable, Union, Optional


class MatrixDataFrame(pd.DataFrame):
    """DataFrame that supports tuple indexing for matrix access."""

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self.iloc[key[0], key[1]]
        return super().__getitem__(key)


def corr_generator(
    df: Union[pd.DataFrame, np.ndarray], method: Union[Callable, str] = kendalltau
) -> pd.DataFrame:
    """Generate a correlation matrix of a dataframe using the specified method.

    Uses vectorized pandas methods when possible for better performance.

    Args:
        df (pd.DataFrame or np.ndarray): Input DataFrame or array.
        method (Callable or str, optional): Function to calculate correlation coefficients
            or string name. Options: pearsonr/'pearson', kendalltau/'kendall',
            spearmanr/'spearman'. Defaults to kendalltau.

    Returns:
        pd.DataFrame: Correlation matrix.

    Raises:
        ValueError: If method is not one of the supported correlation methods.
    """
    # Convert numpy array to DataFrame if needed
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=[f"col_{i}" for i in range(df.shape[1])])

    # Map functions to pandas method names for vectorized computation
    func_to_pandas_method = {
        pearsonr: "pearson",
        kendalltau: "kendall",
        spearmanr: "spearman",
    }

    # Handle string method names
    if isinstance(method, str):
        pandas_method_map = {
            "pearson": "pearson",
            "kendall": "kendall",
            "spearman": "spearman",
        }
        if method not in pandas_method_map:
            raise ValueError(
                "method should be 'pearson', 'kendall', 'spearman', or the corresponding functions"
            )
        pandas_method = pandas_method_map[method]
    elif method in func_to_pandas_method:
        pandas_method = func_to_pandas_method[method]
    else:
        raise ValueError("method should be pearsonr, kendalltau, or spearmanr")

    # Use vectorized pandas correlation (much faster for large datasets)
    corr_df = df.corr(method=pandas_method)

    return corr_df


def matrix_generator(
    df: pd.DataFrame, method: Union[Callable, str] = kendalltau
) -> MatrixDataFrame:
    """Generate a similarity/distance matrix for a dataframe using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (Callable or str, optional): Function to calculate similarity or distance
            or string name. Options: pearsonr/'pearson', kendalltau/'kendall',
            spearmanr/'spearman', mutual_info_score, mutual_info_regression, kl.
            Defaults to kendalltau.

    Returns:
        MatrixDataFrame: Similarity/distance matrix with tuple indexing support.
    """
    # Handle string method names
    if isinstance(method, str):
        method_map = {
            "pearson": pearsonr,
            "kendall": kendalltau,
            "spearman": spearmanr,
            "mutual_info_score": mutual_info_score,
            "mutual_info_regression": mutual_info_regression,
        }
        if method not in method_map:
            raise ValueError(f"Unknown method string: {method}")
        method = method_map[method]
    # Handle standard correlation methods
    if method in [pearsonr, kendalltau, spearmanr]:
        corr_df = corr_generator(df, method)
        return MatrixDataFrame(
            corr_df.values, index=corr_df.index, columns=corr_df.columns
        )

    # Handle mutual information for categorical variables
    elif method == mutual_info_score:
        # Check if columns appear to be categorical
        if df.apply(lambda x: len(x.unique())).max() > 10:
            raise ValueError(
                "mutual_info_score is best suited for categorical data (columns with ≤10 unique values)"
            )

        # Initialize matrix
        matrix_df = MatrixDataFrame(
            np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
        )

        # Calculate mutual information for all column pairs
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i + 1 :]:
                mi = method(df[col1], df[col2])
                matrix_df.loc[col1, col2] = mi
                matrix_df.loc[col2, col1] = mi  # Symmetry

    # Handle mutual information regression
    elif method == mutual_info_regression:
        matrix_df = MatrixDataFrame(
            np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
        )

        for col1 in matrix_df.columns:
            for col2 in matrix_df.columns:
                if col1 != col2:
                    measures = method(df[[col1]], df[col2])
                    matrix_df.loc[col1, col2] = measures[0]

    # Handle other methods (including kl divergence)
    else:
        matrix_df = MatrixDataFrame(
            np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
        )

        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i + 1 :]:
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


def kl_mi_matrix(data: Union[pd.DataFrame, np.ndarray], bins: int = 10) -> pd.DataFrame:
    """Create a matrix of mutual information between features.

    This function creates a symmetric matrix of mutual information values
    between all pairs of features in the dataset.

    Args:
        data (pd.DataFrame or np.ndarray): Input data.
        bins (int): Number of bins for discretization. Defaults to 10.

    Returns:
        pd.DataFrame: Mutual information matrix.
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(
            data, columns=[f"feature_{i}" for i in range(data.shape[1])]
        )

    # Initialize matrix
    n_features = data.shape[1]
    mi_matrix = MatrixDataFrame(
        np.zeros((n_features, n_features)), index=data.columns, columns=data.columns
    )

    # Fill diagonal with 1s (self-information)
    for i in range(n_features):
        mi_matrix.iloc[i, i] = 1.0

    # Calculate pairwise mutual information
    from sklearn.preprocessing import KBinsDiscretizer

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Discretize features for mutual information calculation
            discretizer = KBinsDiscretizer(
                n_bins=bins, encode="ordinal", strategy="uniform"
            )
            col_i_discrete = discretizer.fit_transform(data.iloc[:, [i]]).ravel()
            col_j_discrete = discretizer.fit_transform(data.iloc[:, [j]]).ravel()

            # Calculate mutual information (symmetric)
            mi_val = mutual_info_score(col_i_discrete, col_j_discrete)
            mi_matrix.iloc[i, j] = mi_val
            mi_matrix.iloc[j, i] = mi_val  # Symmetry

    return mi_matrix


def create_minimal_edge_graph(
    W: Union[pd.DataFrame, np.ndarray],
    version: str = "v3",
    reverse: bool = True,
    verbose: bool = False,
) -> tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
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
    # Remember input type for output conversion
    input_was_numpy = isinstance(W, np.ndarray)

    # Convert numpy array to DataFrame for internal processing
    if input_was_numpy:
        W = pd.DataFrame(
            W,
            columns=[f"col_{i}" for i in range(W.shape[1])],
            index=[f"col_{i}" for i in range(W.shape[0])],
        )

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

        if version == "v1":
            # V1: Add edge if either node is not yet in the graph
            if node1 not in connected_nodes or node2 not in connected_nodes:
                add_edge = True
                # If all nodes are in the graph after adding this edge, we're done
                if len(connected_nodes.union({node1, node2})) == len(columns):
                    if verbose:
                        print(f"v1 terminating at weight: {weight}")
                    add_edge = True
                    # Final edge to add
                    adjacency_matrix.loc[node1, node2] = adjacency_matrix.loc[
                        node2, node1
                    ] = 1
                    reduced_df.loc[node1, node2] = reduced_df.loc[node2, node1] = weight
                    break

        elif version == "v2":
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

        elif version == "v3":
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

    # Convert back to numpy arrays if input was numpy
    if input_was_numpy:
        return adjacency_matrix.values, reduced_df.values
    else:
        return adjacency_matrix, reduced_df
