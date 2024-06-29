import pandas as pd
import numpy as np
from scipy.stats import pearsonr,kendalltau,spearmanr

def corr_generator(df, method=kendalltau):
    """generate the correlation matrix of a dataframe

    Args:
        df (pd.DataFrame): DataFrame
        method (function, optional): the function to calculate the correlation coefficient. Defaults to pearsonr.
                                     it canbe pearsonr, kendalltau, spearmanr
    Returns:
        _type_: pd.DataFrame: correlation matrix
    """
    
    # Initialize dataframes for storing correlation values
    corr_df = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    # Iterate over all pairs of columns
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:  # Ensuring not to correlate a column with itself
                corr, _ = method(df[col1], df[col2])
                corr_df.loc[col1, col2] = corr
    return corr_df

def create_minimal_edge_graph(W):
    """weight matrix to reduced adjacency matrix

    Args:
        W (pd.DataFrame): DataFrame

    Returns:
        adjacency_matrix: reduced adjacency matrix
        reduced_df: reduced weight matrix
    """
    edges = []
    columns = W.columns.tolist()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            edges.append((columns[i], columns[j], abs(W.iloc[i, j])))

    # sorting
    edges.sort(key=lambda x: x[2], reverse=True)

    connected_nodes = set()
    adjacency_matrix = pd.DataFrame(0, index=columns, columns=columns)
    reduced_df = pd.DataFrame(0, index=columns, columns=columns)
    # add edges to the graph until all nodes are connected
    for edge in edges:
        node1, node2, weight = edge
        if node1 not in connected_nodes or node2 not in connected_nodes:
            adjacency_matrix.loc[node1, node2] = 1
            adjacency_matrix.loc[node2, node1] = 1
            reduced_df.loc[node1, node2] = weight
            reduced_df.loc[node2, node1] = weight
            connected_nodes.update([node1, node2])
        # stop when all nodes are connected
        if len(connected_nodes) == len(columns):
            # print(weight)
            break
    return adjacency_matrix, reduced_df