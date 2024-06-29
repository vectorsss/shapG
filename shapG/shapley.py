import networkx as nx
import itertools
import random
from math import factorial, log2, ceil
from tqdm import tqdm

def graph_generator(n_nodes,density, weight_range=(1, 10),seed=2333):
    """generate a random graph based on the density

    Args:
        n_nodes (int): number of nodes
        density (float): the density of the graph
        weight_range (tuple, optional): range of weight. Defaults to (1, 10).
        seed (int, optional): random seed. Defaults to 2333.

    Returns:
        nx.Graph: graph
    """
    random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))  
    max_edges = n_nodes * (n_nodes - 1) / 2
    # calculate the expected number of edges
    n_edges = int(max_edges * density)

    while G.number_of_edges() < n_edges:
        u, v = random.sample(list(G.nodes()), 2)
        if not G.has_edge(u, v):
            if weight_range is None:
                G.add_edge(u, v)
            else:
                weight = random.randint(*weight_range)
                G.add_edge(u, v, weight=weight)

    return G

def coalition_degree(G, S):
    """characteristic function of a coalition in a graph

    Args:
        G (nx.Graph): the graph
        S (list/set): list/set of nodes to generate teh coalition

    Returns:
        int: the marginal contribution of the coalition
    """
    subgraph = G.subgraph(S)
    return sum(dict(subgraph.degree(weight='weight')).values()) / 2

def shapley_value(G: nx.Graph,f=coalition_degree):
    """shapley value without sampling

    Args:
        G (nx.Graph): graph
        f (function, optional): the characteristic function based on graph. Defaults to coalition_degree.

    Returns:
        dict: dictionary of shapley value
    """
    n_nodes = len(G.nodes())
    shapley_values = {node: 0 for node in G.nodes()}
    
    # Precompute factorials to improve efficiency
    fact = [factorial(i) for i in range(n_nodes + 1)]
    
    for node in tqdm(G.nodes()):
        for r in range(n_nodes + 1):
            for subset in itertools.combinations(G.nodes(), r):
                if node not in subset:
                    S = list(subset)
                    S_with_node = S + [node]
                    coeff = (fact[len(S)] * fact[n_nodes - len(S) - 1]) / fact[n_nodes]
                    marginal_contribution = f(G, S_with_node) - f(G, S)
                    shapley_values[node] += coeff * marginal_contribution
    return shapley_values

def get_neighbors_at_depth(G, node, depth):
    """Get k-hop neighbors for a given node.

    Args:
        G (nx.Graph): Graph.
        node (nx.Node): The given node.
        depth (int): Depth (k).

    Returns:
        list: k-hop neighbors.
    """
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=depth)
    neighbors = {n for n, d in path_lengths.items() if d == depth}
    return neighbors

def shapG(G: nx.Graph, f=coalition_degree, depth=1, m=15, approximate_by_ratio=True, scale=True, verbose=False):
    """shapG algorithm with local search

    Args:
        G (nx.Graph): graph
        f (function, optional): characteristic function. Defaults to coalition_degree.
        depth (int, optional): the maximal depth. Defaults to 1.
        m (int, optional): the maximal neighbors. Defaults to 15.
        approximate_by_ratio (bool, optional): whether to approximate by ratio. Defaults to False (depends on the characteristic function).
        scale (bool, optional): the scaling factor. Defaults to True.

    Returns:
        dict: dictionary of shapley value
    """
    shapley_values = {node: 0 for node in G.nodes()}
    for node in tqdm(G.nodes()):
        neighbors_at_depth = set()
        for d in range(1, depth + 1):
            neighbors_at_depth |= get_neighbors_at_depth(G, node, d)
        
        if len(neighbors_at_depth) < m:
            neighbors_at_depth.add(node)
            for S_size in range(len(neighbors_at_depth) + 1):
                for S in itertools.combinations(neighbors_at_depth, S_size):
                    if node not in S:
                        S_with_node = list(S) + [node]
                        marginal_contribution = f(G, S_with_node) - f(G, S)
                        shapley_values[node] += marginal_contribution
            coeff = 1 / 2 ** (len(neighbors_at_depth) - 1)
            shapley_values[node] *= coeff
        else:
            # Eine Wahrscheinlichkeitsaufgabe in der Kundenwerbung Equation 18
            sample_nums = ceil(len(neighbors_at_depth) / m * (log2(len(neighbors_at_depth)) + 0.5772156649))  # Average sampling times
            for i in range(sample_nums):
                neighbors_at_depth_sampled = set(random.sample(list(neighbors_at_depth), m))
                neighbors_at_depth_sampled.add(node)
                iteration = 0
                for S_size in range(len(neighbors_at_depth_sampled) + 1):
                    for S in itertools.combinations(neighbors_at_depth_sampled, S_size):
                        if verbose and iteration % 100000 == 0:
                            print("{}: {}/{}".format(node, iteration, 1 << len(neighbors_at_depth_sampled)))
                        if node not in S:
                            S_with_node = list(S) + [node]
                            marginal_contribution = f(G, S_with_node) - f(G, S)
                            shapley_values[node] += marginal_contribution
                        iteration += 1
            coeff = 1 / 2 ** (len(neighbors_at_depth_sampled) - 1) / sample_nums
            if scale:
                coeff *= ((len(neighbors_at_depth) + 1) / len(neighbors_at_depth_sampled))
            shapley_values[node] *= coeff
    if approximate_by_ratio:
        biggest_coalition_contribution = f(G, G.nodes())
        approximate_biggest_coalition_contribution = sum(shapley_values.values())
        for k,v in shapley_values.items():
            corrected_v = v / approximate_biggest_coalition_contribution * biggest_coalition_contribution
            shapley_values[k] = corrected_v
    return shapley_values
