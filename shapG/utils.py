import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr,kendalltau,spearmanr
from sklearn.metrics import mutual_info_score

import logging
def corr_generator(df, method=kendalltau):
    """generate the correlation matrix of a dataframe

    Args:
        df (pd.DataFrame): DataFrame
        method (function, optional): the function to calculate the correlation coefficient. Defaults to pearsonr.
                                     it canbe pearsonr, kendalltau, spearmanr
    Returns:
        _type_: pd.DataFrame: correlation matrix
    """
    if method not in [pearsonr, kendalltau, spearmanr]:
        raise ValueError("method should be pearsonr, kendalltau, spearmanr")
    # Initialize dataframes for storing correlation values
    corr_df = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    # Iterate over all pairs of columns
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:  # Ensuring not to correlate a column with itself
                corr, _ = method(df[col1], df[col2])
                corr_df.loc[col1, col2] = corr
    return corr_df

def matrix_generator(df, method=kendalltau):
    """generate the correlation matrix of a dataframe

    Args:
        df (pd.DataFrame): DataFrame
        method (function, optional): the function to calculate the correlation coefficient. Defaults to pearsonr.
                                     it canbe pearsonr, kendalltau, spearmanr, mutual information (mutual_info_score), K-L divergence(kl)
    Returns:
        _type_: pd.DataFrame: correlation matrix
    """
    if method in [pearsonr, kendalltau, spearmanr]:
        return corr_generator(df, method)
    else:
        # Initialize dataframes for storing correlation values
        matrix_df = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
        # Iterate over all pairs of columns
        for col1 in matrix_df.columns:
            for col2 in matrix_df.columns:
                if col1 != col2:  # Ensuring not to correlate a column with itself
                    measures = method(df[col1], df[col2])
                    matrix_df.loc[col1, col2] = measures
    return matrix_df

def kl(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 1e-10
    P = P+epsilon
    Q = Q+epsilon
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    divergence = np.sum(P*np.log(P/Q))
    return divergence

def create_minimal_edge_graph(W, version='v3', reverse=True, verbose=False):
    """weight matrix to reduced adjacency matrix

    Args:
        W (pd.DataFrame): DataFrame
        version (str): version of the algorithm
            v1: stop when all nodes are added to the graph
            v2: when all nodes are added to the graph, then check the connectivity, 
                continue add edges until the graph is connected
            v3: strong connectivity check, more edges than v2
            example: 
            links: [1,2] [1,3] [2,3] [4,5] [4,6] [5,6] [1,4]
            v1: [1,2] [1,3] [4,5] [4,6]
            v2: [1,2] [1,3] [4,5] [4,6] [1,4]
            v3: [1,2] [1,3] [2,3] [4,5] [4,6] [5,6] [1,4]
        reverse (bool): sorting order (True: descending, False: ascending)

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
    edges.sort(key=lambda x: x[2], reverse=reverse)

    connected_nodes = set()
    adjacency_matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=np.int8)
    reduced_df = pd.DataFrame(0, index=columns, columns=columns, dtype=np.float64)
    # add edges to the graph until all nodes are connected
    for edge in edges:
        node1, node2, weight = edge
        if node1 not in connected_nodes or node2 not in connected_nodes:
            adjacency_matrix.loc[node1, node2] = 1
            adjacency_matrix.loc[node2, node1] = 1
            reduced_df.loc[node1, node2] = weight
            reduced_df.loc[node2, node1] = weight
            connected_nodes.update([node1, node2])
        if version == 'v1':
            if len(connected_nodes) == len(columns):
                if verbose:
                    print(weight)
                break
        elif version == 'v2':
            if len(connected_nodes) == len(columns):
                G = nx.Graph(adjacency_matrix)
                if nx.is_connected(G):
                    if verbose:
                        print(weight)
                    break
                else:
                    adjacency_matrix.loc[node1, node2] = 1
                    adjacency_matrix.loc[node2, node1] = 1
                    reduced_df.loc[node1, node2] = weight
                    reduced_df.loc[node2, node1] = weight
        elif version == 'v3':
            G = nx.Graph(adjacency_matrix)
            if nx.is_connected(G) and len(connected_nodes) == len(columns):
                if verbose:
                    print(weight)
                break
            else:
                adjacency_matrix.loc[node1, node2] = 1
                adjacency_matrix.loc[node2, node1] = 1
                reduced_df.loc[node1, node2] = weight
                reduced_df.loc[node2, node1] = weight
                connected_nodes.update([node1, node2])

    return adjacency_matrix, reduced_df