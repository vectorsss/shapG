"""
Graph construction and coalition management utilities.
"""

from typing import Dict, Set, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import squareform, pdist, cosine
from scipy.stats import pearsonr, kendalltau, spearmanr
from loguru import logger


class GraphBuilder:
    """Utility class for constructing graphs from data."""

    @staticmethod
    def from_kendalltau_minimal_edge(
        data: Union[np.ndarray, pd.DataFrame], reverse: bool = True, version: str = "v3"
    ) -> nx.Graph:
        """Build graph using the original ShapG method: kendalltau + minimal edge graph.

        This matches the original benchmark construction method exactly.

        Args:
            data: Input data
            reverse: Whether to reverse the edge selection order
            version: Version of the minimal edge algorithm ('v1', 'v2', 'v3')

        Returns:
            NetworkX graph
        """
        from .utils import matrix_generator, create_minimal_edge_graph

        # Step 1: Generate weight matrix using kendalltau (default)
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        W = matrix_generator(data)  # Uses kendalltau by default

        # Step 2: Create minimal edge graph
        A, W_new = create_minimal_edge_graph(W, reverse=reverse, version=version)

        # Step 3: Convert to NetworkX graph
        G = nx.Graph(A)

        return G

    @staticmethod
    def from_matrix_generator(
        data: Union[np.ndarray, pd.DataFrame],
        method: Optional[Callable] = None,
        create_graph_method: str = "minimal_edge",
        **kwargs,
    ) -> nx.Graph:
        """Build graph using matrix_generator with custom method.

        Args:
            data: Input data
            method: Method for matrix generation (kendalltau, pearsonr, spearmanr, kl, etc.)
                   If None, uses kendalltau (default)
            create_graph_method: How to create graph from matrix
                                'minimal_edge': Use create_minimal_edge_graph
                                'threshold': Use simple thresholding
            **kwargs: Additional arguments for graph creation

        Returns:
            NetworkX graph
        """
        from .utils import matrix_generator, create_minimal_edge_graph
        from scipy.stats import kendalltau

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Generate weight matrix
        if method is None:
            W = matrix_generator(data)  # Uses kendalltau by default
        else:
            W = matrix_generator(data, method)

        # Create graph based on method
        if create_graph_method == "minimal_edge":
            reverse = kwargs.get("reverse", True)
            version = kwargs.get("version", "v3")
            A, _ = create_minimal_edge_graph(W, reverse=reverse, version=version)
            G = nx.Graph(A)
        elif create_graph_method == "threshold":
            threshold = kwargs.get("threshold", 0.3)
            A = (W.abs() >= threshold).astype(int)
            np.fill_diagonal(A.values if hasattr(A, "values") else A, 0)
            G = (
                nx.from_pandas_adjacency(A)
                if isinstance(A, pd.DataFrame)
                else nx.from_numpy_array(A)
            )
        else:
            raise ValueError(f"Unknown graph creation method: {create_graph_method}")

        return G

    @staticmethod
    def from_correlation(
        data: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.3,
        method: str = "pearson",
    ) -> nx.Graph:
        """Build graph from correlation matrix.

        Args:
            data: Input data
            threshold: Correlation threshold for edge creation
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            NetworkX graph
        """
        if isinstance(data, pd.DataFrame):
            corr_matrix = data.corr(method=method)
            node_labels = data.columns.tolist()
        else:
            corr_df = pd.DataFrame(data)
            corr_matrix = corr_df.corr(method=method)
            node_labels = corr_df.columns.tolist()

        G = nx.Graph()
        n_features = len(corr_matrix)
        G.add_nodes_from(node_labels)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                correlation = abs(corr_matrix.iloc[i, j])
                if correlation >= threshold:
                    G.add_edge(node_labels[i], node_labels[j], weight=correlation)

        return G

    @staticmethod
    def from_mutual_information(
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        n_neighbors: int = 3,
    ) -> nx.Graph:
        """Build graph from mutual information.

        Args:
            X: Feature matrix
            y: Optional target variable
            threshold: MI threshold for edge creation
            n_neighbors: Number of neighbors for MI estimation

        Returns:
            NetworkX graph
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            node_labels = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            node_labels = list(range(X.shape[1]))

        n_features = X_array.shape[1]
        G = nx.Graph()
        G.add_nodes_from(node_labels)

        if y is not None:
            # Compute MI with target
            mi_scores = mutual_info_regression(X_array, y, n_neighbors=n_neighbors)

            # Add edges based on MI similarity
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    mi_similarity = min(mi_scores[i], mi_scores[j])
                    if mi_similarity >= threshold:
                        G.add_edge(node_labels[i], node_labels[j], weight=mi_similarity)
        else:
            # Compute pairwise MI between features
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    mi = mutual_info_regression(
                        X_array[:, i].reshape(-1, 1),
                        X_array[:, j],
                        n_neighbors=n_neighbors,
                    )[0]
                    if mi >= threshold:
                        G.add_edge(node_labels[i], node_labels[j], weight=mi)

        return G

    @staticmethod
    def from_adjacency(
        adj_matrix: Union[np.ndarray, pd.DataFrame],
        weighted: bool = True,
        node_labels: Optional[List[str]] = None,
    ) -> nx.Graph:
        """Build graph from adjacency matrix.

        Args:
            adj_matrix: Adjacency matrix
            weighted: Whether to use weights
            node_labels: Optional list of node labels

        Returns:
            NetworkX graph
        """
        if isinstance(adj_matrix, pd.DataFrame):
            adj_matrix = adj_matrix.values

        if weighted:
            G = nx.from_numpy_array(adj_matrix)
        else:
            G = nx.from_numpy_array((adj_matrix > 0).astype(int))

        # Apply node labels if provided
        if node_labels is not None:
            if len(node_labels) != G.number_of_nodes():
                raise ValueError(
                    f"Number of labels ({len(node_labels)}) must match number of nodes ({G.number_of_nodes()})"
                )

            # Create mapping from integers to labels
            mapping = {i: label for i, label in enumerate(node_labels)}
            G = nx.relabel_nodes(G, mapping)

        return G

    @staticmethod
    def random_graph(
        n_nodes: int,
        density: float = 0.3,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        seed: Optional[int] = None,
    ) -> nx.Graph:
        """Generate random graph.

        Args:
            n_nodes: Number of nodes
            density: Edge density (0 to 1)
            weight_range: Range for edge weights
            seed: Random seed

        Returns:
            Random NetworkX graph
        """
        if seed is not None:
            np.random.seed(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))

        n_possible_edges = n_nodes * (n_nodes - 1) // 2
        n_edges = int(density * n_possible_edges)

        # Generate random edges
        possible_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        selected_edges = np.random.choice(len(possible_edges), n_edges, replace=False)

        for idx in selected_edges:
            i, j = possible_edges[idx]
            weight = np.random.uniform(weight_range[0], weight_range[1])
            G.add_edge(i, j, weight=weight)

        return G

    @staticmethod
    def from_rank_deletion(
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        density_ratio: Optional[float] = None,
        correlation_method: str = "cosine",
        similarity_method: str = "cosine",
        feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        use_mixed_similarity: bool = False,
        use_target_aware: bool = False,
        enforce_cross_type_edges: bool = False,
    ) -> nx.Graph:
        """Construct graph by selectively removing edges based on feature ranking.

        This method creates a complete graph, ranks features by their correlation with
        the target, and removes edges following the ranking while maintaining connectivity.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values
            density_ratio: Target edge density ratio (0 to 1). If None, uses 1.5x minimum ratio
            correlation_method: Method for feature ranking ('pearsonr', 'kendalltau',
                              'spearmanr', 'cosine', 'mutual_info')
            similarity_method: Method for edge weight calculation (same options as correlation_method)
            feature_ranges: Dict with 'num', 'cat', 'bin' keys for feature type ranges
                          Example: {'num': (0, 5), 'cat': (5, 8), 'bin': (8, 10)}
            use_mixed_similarity: If True, use appropriate similarity for each feature type
            use_target_aware: If True, use target-aware similarity for binary classification
            enforce_cross_type_edges: If True, always keep edges between different feature types

        Returns:
            NetworkX graph with weighted edges
        """
        # Handle input types
        if isinstance(X, pd.DataFrame):
            features = X.columns.tolist()
            X_array = X.to_numpy()
        else:
            n_features = X.shape[1]
            features = list(range(n_features))
            X_array = X

        n_features = len(features)

        # Use target-aware similarity for binary classification
        if use_target_aware and len(np.unique(y)) == 2:
            from sklearn.feature_selection import mutual_info_classif

            # Get feature importance using mutual information with target
            mi_scores = mutual_info_classif(X_array, y.ravel() if y.ndim > 1 else y)
            feature_rank_indices = np.argsort(mi_scores)[::-1]
            feature_rank = [features[i] for i in feature_rank_indices]

            # Calculate similarity based on how similarly features predict the target
            similarity_matrix = np.zeros((n_features, n_features))

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    sim = 0
                    for class_val in np.unique(y):
                        mask = (
                            (y == class_val).ravel() if y.ndim > 1 else (y == class_val)
                        )
                        if np.sum(mask) > 1:
                            X_class = X_array[mask]
                            try:
                                if feature_ranges and use_mixed_similarity:
                                    # Determine feature types
                                    type_i = GraphBuilder._get_feature_type(
                                        i, feature_ranges
                                    )
                                    type_j = GraphBuilder._get_feature_type(
                                        j, feature_ranges
                                    )

                                    if type_i == "num" and type_j == "num":
                                        corr, _ = pearsonr(X_class[:, i], X_class[:, j])
                                        sim += abs(corr) * (np.sum(mask) / len(y))
                                    else:
                                        # For categorical/binary, use normalized mutual information
                                        from sklearn.metrics import (
                                            normalized_mutual_info_score,
                                        )

                                        sim += normalized_mutual_info_score(
                                            X_class[:, i], X_class[:, j]
                                        ) * (np.sum(mask) / len(y))
                                else:
                                    corr, _ = pearsonr(X_class[:, i], X_class[:, j])
                                    sim += abs(corr) * (np.sum(mask) / len(y))
                            except:
                                sim += 0

                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        # Use mixed similarity if requested
        elif use_mixed_similarity:
            from .feature_similarity import (
                calculate_mixed_similarity_matrix,
                get_feature_ranking_mixed,
            )

            # Get feature ranking using mixed methods
            feature_rank_indices = get_feature_ranking_mixed(
                X, y, feature_ranges=feature_ranges
            )
            feature_rank = [features[i] for i in feature_rank_indices]

            # Calculate similarity matrix using mixed methods
            similarity_matrix, _ = calculate_mixed_similarity_matrix(
                X, feature_ranges=feature_ranges
            )
        else:
            # Get feature ranking
            feature_rank_indices = GraphBuilder._get_feature_rank(
                X, y, correlation_method
            )
            feature_rank = [features[i] for i in feature_rank_indices]

            # Calculate similarity matrix for edge weights
            similarity_matrix = GraphBuilder._calculate_similarity_matrix(
                X_array, similarity_method
            )

        # Create initial complete graph
        G = nx.Graph()
        edges = []
        cross_type_edges = set()

        for i in range(n_features):
            for j in range(i + 1, n_features):
                weight = similarity_matrix[i, j]
                edges.append((features[i], features[j], weight))
                # Track cross-type edges
                if GraphBuilder._is_cross_type_edge(
                    i, j, feature_ranges, enforce_cross_type_edges
                ):
                    cross_type_edges.add((features[i], features[j]))
                    cross_type_edges.add((features[j], features[i]))

        # Sort edges by weight (descending)
        edges.sort(key=lambda x: x[2], reverse=True)
        G.add_weighted_edges_from(edges)

        # Check initial connectivity
        if not nx.is_connected(G):
            raise ValueError("Initial graph is not connected")

        # Calculate target number of edges
        max_edges = (n_features * (n_features - 1)) // 2

        # If density_ratio not specified, use 1.5x minimum ratio as default
        if density_ratio is None:
            min_ratio = (n_features - 1) / max_edges if max_edges > 0 else 0.3
            density_ratio = min(1.5 * min_ratio, 0.3)
            logger.info(
                f"No density ratio specified. Using {density_ratio:.3f} (1.5x minimum ratio)"
            )

        target_edges = int(density_ratio * max_edges)

        # If enforcing cross-type edges, count them separately
        num_cross_type_edges = (
            len(cross_type_edges) // 2 if enforce_cross_type_edges else 0
        )
        if num_cross_type_edges > 0:
            logger.info(
                f"Enforcing {num_cross_type_edges} cross-type edges between numerical and categorical/binary features"
            )
            target_edges = max(
                target_edges,
                num_cross_type_edges + min(n_features - 1, target_edges // 2),
            )

        # Minimum edges for connectivity
        min_edges = n_features - 1
        min_ratio = min_edges / max_edges if max_edges > 0 else 1.0

        # Auto-adjust if density ratio is too small
        actual_density_ratio = density_ratio
        if target_edges < min_edges:
            actual_density_ratio = min(1.5 * min_ratio, 1.0)
            target_edges = int(actual_density_ratio * max_edges)
            target_edges = max(target_edges, min_edges)
            logger.info(
                f"Density ratio {density_ratio} is too small (minimum: {min_ratio:.3f}). "
                f"Auto-adjusting to {actual_density_ratio:.3f}"
            )

        # Remove edges following feature rank while maintaining connectivity
        for node in feature_rank:
            if G.number_of_edges() <= target_edges:
                break

            # Get edges of current node sorted by weight (descending)
            node_edges = [(u, v) for u, v in G.edges(node)]
            node_edges.sort(key=lambda x: G[x[0]][x[1]]["weight"], reverse=True)

            for edge in node_edges:
                if G.number_of_edges() <= target_edges:
                    break

                # Skip cross-type edges if enforcing
                if enforce_cross_type_edges and (
                    edge in cross_type_edges or (edge[1], edge[0]) in cross_type_edges
                ):
                    continue

                # Try removing edge
                weight = G[edge[0]][edge[1]]["weight"]
                G.remove_edge(*edge)

                # Check connectivity
                if nx.is_connected(G):
                    continue
                else:
                    # Restore edge to maintain connectivity
                    G.add_edge(*edge, weight=weight)

        # Add metadata about the actual density used
        G.graph["actual_density_ratio"] = actual_density_ratio

        return G

    @staticmethod
    def _get_feature_type(
        idx: int, feature_ranges: Optional[Dict[str, Tuple[int, int]]]
    ) -> str:
        """Determine feature type from feature ranges."""
        if feature_ranges:
            if (
                "num" in feature_ranges
                and feature_ranges["num"][0] <= idx < feature_ranges["num"][1]
            ):
                return "num"
            elif (
                "cat" in feature_ranges
                and feature_ranges["cat"][0] <= idx < feature_ranges["cat"][1]
            ):
                return "cat"
            elif (
                "bin" in feature_ranges
                and feature_ranges["bin"][0] <= idx < feature_ranges["bin"][1]
            ):
                return "bin"
        return "unknown"

    @staticmethod
    def _is_cross_type_edge(
        i: int,
        j: int,
        feature_ranges: Optional[Dict[str, Tuple[int, int]]],
        enforce_cross_type_edges: bool,
    ) -> bool:
        """Check if edge is cross-type (between numerical and categorical/binary)."""
        if not enforce_cross_type_edges or not feature_ranges:
            return False
        type_i = GraphBuilder._get_feature_type(i, feature_ranges)
        type_j = GraphBuilder._get_feature_type(j, feature_ranges)
        return (type_i == "num" and type_j in ["cat", "bin"]) or (
            type_j == "num" and type_i in ["cat", "bin"]
        )

    @staticmethod
    def _get_feature_rank(
        X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, method: str = "cosine"
    ) -> List[int]:
        """Get feature ranking based on correlation with target."""
        method_map = {
            "pearsonr": pearsonr,
            "kendalltau": kendalltau,
            "spearmanr": spearmanr,
            "cosine": lambda x, y: 1 - cosine(x, y),
            "mutual_info": None,
        }

        if method not in method_map:
            raise ValueError(f"Method {method} not supported")

        # Handle mutual information separately
        if method == "mutual_info":
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )

            is_classification = len(np.unique(y)) < 10

            if isinstance(X, pd.DataFrame):
                X_array = X.to_numpy()
            else:
                X_array = X

            if is_classification:
                mi_scores = mutual_info_classif(X_array, y.ravel() if y.ndim > 1 else y)
            else:
                mi_scores = mutual_info_regression(
                    X_array, y.ravel() if y.ndim > 1 else y
                )

            return np.argsort(mi_scores)[::-1].tolist()

        correlation_func = method_map[method]
        use_pvalue = method in ["pearsonr", "kendalltau", "spearmanr"]

        values = {}
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if use_pvalue:
                    values[col] = correlation_func(X[col], y)[1]
                else:
                    result = correlation_func(X[col], y)
                    values[col] = result if np.isscalar(result) else result[0]
            values_rank = sorted(values, key=values.get, reverse=use_pvalue)
            return [X.columns.get_loc(col) for col in values_rank]
        else:
            n_features = X.shape[1]
            for i in range(n_features):
                if use_pvalue:
                    values[i] = correlation_func(X[:, i], y)[1]
                else:
                    result = correlation_func(X[:, i], y)
                    values[i] = result if np.isscalar(result) else result[0]
            return sorted(values, key=values.get, reverse=use_pvalue)

    @staticmethod
    def _calculate_similarity_matrix(
        X_array: np.ndarray, method: str = "cosine"
    ) -> np.ndarray:
        """Calculate similarity matrix between features."""
        n_features = X_array.shape[1]
        similarity_matrix = np.zeros((n_features, n_features))

        if method == "mutual_info":
            from sklearn.metrics import normalized_mutual_info_score

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    nmi = normalized_mutual_info_score(X_array[:, i], X_array[:, j])
                    similarity_matrix[i, j] = nmi
                    similarity_matrix[j, i] = nmi
        else:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if method in ["pearsonr", "kendalltau", "spearmanr"]:
                        corr, _ = eval(method)(X_array[:, i], X_array[:, j])
                        similarity_matrix[i, j] = abs(corr)
                        similarity_matrix[j, i] = abs(corr)
                    elif method == "cosine":
                        sim = 1 - cosine(X_array[:, i], X_array[:, j])
                        similarity_matrix[i, j] = abs(sim)
                        similarity_matrix[j, i] = abs(sim)

        return similarity_matrix


class CoalitionManager:
    """Manages coalition generation and sampling for Shapley value computation."""

    def __init__(self, graph: Optional[nx.Graph] = None):
        """Initialize coalition manager.

        Args:
            graph: Optional graph for coalition structure
        """
        self.graph = graph
        self._coalitions_cache = {}

    def get_neighbors_coalition(
        self,
        graph_or_nodes: Union[nx.Graph, Set[int], int],
        nodes: Optional[Union[Set[int], int]] = None,
        depth: int = 1,
        include_self: bool = False,
    ) -> Set[int]:
        """Get coalition of neighbors up to specified depth.

        Args:
            graph_or_nodes: Either a graph (if used as static method) or nodes (if instance method)
            nodes: Node or set of nodes (when graph provided as first arg)
            depth: Maximum distance from center node
            include_self: Whether to include the center nodes

        Returns:
            Set of nodes in the coalition
        """
        # Handle flexible calling patterns
        if isinstance(graph_or_nodes, nx.Graph):
            # Called as: get_neighbors_coalition(graph, nodes, depth=1)
            graph = graph_or_nodes
            target_nodes = nodes
        else:
            # Called as: get_neighbors_coalition(nodes, depth=1) - use instance graph
            if hasattr(self, "graph") and self.graph is not None:
                graph = self.graph
                target_nodes = graph_or_nodes
            else:
                raise ValueError(
                    "Graph not available - either pass graph as argument or set instance graph"
                )

        if graph is None:
            raise ValueError("Graph not available")

        # Ensure target_nodes is a set
        if isinstance(target_nodes, int):
            target_nodes = {target_nodes}
        elif isinstance(target_nodes, (list, tuple)):
            target_nodes = set(target_nodes)

        coalition = set()
        if include_self:
            coalition.update(target_nodes)

        # BFS to find neighbors up to depth
        current_level = set(target_nodes)
        visited = set(target_nodes)  # Track visited nodes to avoid cycles

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in graph:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)

            if not next_level:  # No more nodes to expand
                break

            coalition.update(next_level)
            visited.update(next_level)
            current_level = next_level

        return coalition

    # Add static method support by monkey patching after class definition

    def get_all_coalitions(
        self, nodes: Optional[List[int]] = None, max_size: Optional[int] = None
    ) -> List[Set[int]]:
        """Generate all possible coalitions.

        Args:
            nodes: List of nodes (uses all graph nodes if None)
            max_size: Maximum coalition size

        Returns:
            List of all coalitions
        """
        if nodes is None:
            if self.graph is None:
                raise ValueError("Either nodes or graph must be provided")
            nodes = list(self.graph.nodes())

        from itertools import combinations

        n = len(nodes)
        max_size = max_size or n
        coalitions = []

        for size in range(min(max_size + 1, n + 1)):
            for coalition in combinations(nodes, size):
                coalitions.append(set(coalition))

        return coalitions

    def sample_coalitions(
        self,
        graph_or_node: Union[nx.Graph, int],
        n_samples: int = 100,
        strategy: str = "uniform",
        coalition_set: Optional[Set[int]] = None,
        seed: Optional[int] = None,
        coalition_sizes: Optional[List[int]] = None,
    ) -> List[Set[int]]:
        """Sample coalitions.

        Args:
            graph_or_node: Either a graph (for global sampling) or node (for node-specific sampling)
            n_samples: Number of samples
            strategy: Sampling strategy ('uniform', 'weighted', 'stratified')
            coalition_set: Optional set of candidate nodes
            seed: Random seed for reproducibility
            coalition_sizes: Specific sizes for sampling

        Returns:
            List of sampled coalitions
        """
        # Handle flexible calling patterns
        if isinstance(graph_or_node, nx.Graph):
            # Called with graph: sample from all nodes in graph
            graph = graph_or_node
            node = None
            if coalition_set is None:
                coalition_set = set(graph.nodes())
        else:
            # Called with node: sample excluding that node
            node = graph_or_node
            graph = self.graph
            if coalition_set is None:
                if graph is None:
                    raise ValueError("Either coalition_set or graph must be provided")
                coalition_set = set(graph.nodes()) - {node}

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        coalition_list = list(coalition_set)
        n_features = len(coalition_list)
        samples = []

        if strategy == "uniform":
            # Uniform random sampling
            if coalition_sizes is not None:
                # Sample specific sizes
                for size in coalition_sizes:
                    if size > n_features:
                        size = n_features
                    if size > 0:
                        selected = np.random.choice(coalition_list, size, replace=False)
                        samples.append(set(selected))
                    else:
                        samples.append(set())
            else:
                # Sample random sizes
                for _ in range(n_samples):
                    size = np.random.randint(0, n_features + 1)
                    if size > 0:
                        selected = np.random.choice(coalition_list, size, replace=False)
                        samples.append(set(selected))
                    else:
                        samples.append(set())

        elif strategy == "weighted":
            # Weight by distance from node
            if self.graph is None:
                raise ValueError("Graph required for weighted sampling")

            # Compute distances
            try:
                distances = nx.single_source_shortest_path_length(self.graph, node)
            except:
                distances = {n: 1 for n in coalition_list}

            weights = np.array(
                [1.0 / (distances.get(n, n_features) + 1) for n in coalition_list]
            )
            weights /= weights.sum()

            for _ in range(n_samples):
                size = np.random.randint(0, n_features + 1)
                if size > 0:
                    selected = np.random.choice(
                        coalition_list, size, replace=False, p=weights
                    )
                    samples.append(set(selected))
                else:
                    samples.append(set())

        elif strategy == "stratified":
            # Stratified by coalition size
            sizes = list(range(n_features + 1))
            samples_per_size = n_samples // len(sizes)
            remainder = n_samples % len(sizes)

            for i, size in enumerate(sizes):
                n_size_samples = samples_per_size + (1 if i < remainder else 0)
                for _ in range(n_size_samples):
                    if size > 0 and size <= n_features:
                        selected = np.random.choice(coalition_list, size, replace=False)
                        samples.append(set(selected))
                    elif size == 0:
                        samples.append(set())

        return samples[:n_samples]  # Ensure exact number of samples

    def clear_cache(self):
        """Clear the coalition cache."""
        self._coalitions_cache.clear()


# Class methods for backward compatibility with static method access
# These allow both CoalitionManager.method(graph, ...) and manager.method(graph, ...)
_original_get_neighbors_coalition = CoalitionManager.get_neighbors_coalition
_original_sample_coalitions = CoalitionManager.sample_coalitions


def _wrapped_get_neighbors_coalition(self_or_graph, *args, **kwargs):
    """Wrapper to support both instance and static method calls."""
    if isinstance(self_or_graph, CoalitionManager):
        # Instance method call: self.get_neighbors_coalition(graph_or_nodes, ...)
        return _original_get_neighbors_coalition(self_or_graph, *args, **kwargs)
    else:
        # Static method call: CoalitionManager.get_neighbors_coalition(graph, nodes, ...)
        # Create temporary instance and call the method
        temp_manager = CoalitionManager()
        return _original_get_neighbors_coalition(
            temp_manager, self_or_graph, *args, **kwargs
        )


def _wrapped_sample_coalitions(self_or_graph, *args, **kwargs):
    """Wrapper to support both instance and static method calls."""
    if isinstance(self_or_graph, CoalitionManager):
        # Instance method call: self.sample_coalitions(graph_or_node, ...)
        return _original_sample_coalitions(self_or_graph, *args, **kwargs)
    else:
        # Static method call: CoalitionManager.sample_coalitions(graph, ...)
        # Create temporary instance and call the method
        temp_manager = CoalitionManager()
        return _original_sample_coalitions(temp_manager, self_or_graph, *args, **kwargs)


# Apply wrappers to support both calling patterns
CoalitionManager.get_neighbors_coalition = _wrapped_get_neighbors_coalition
CoalitionManager.sample_coalitions = _wrapped_sample_coalitions
