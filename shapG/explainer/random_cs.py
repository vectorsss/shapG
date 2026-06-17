"""
Random Compressed Sensing Shapley value computation.

Based on the random sampling approach from the paper:
"A novel sparsity-based deterministic method for Shapley value approximation"
This implements the baseline random CS method for comparison.
"""

from typing import Dict, Set, Union, Optional, Callable
import numpy as np
import pandas as pd
import networkx as nx
import random
import warnings

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


class RandomCSExplainer(GraphExplainer):
    """Random Compressed Sensing Shapley value approximation.

    This explainer uses random sampling and compressed sensing to approximate
    Shapley values. Unlike QRCSExplainer which uses deterministic QR decomposition,
    this method uses random Bernoulli measurement matrices and iterative sampling.

    The method combines:
    - Random Bernoulli measurement matrix
    - Multiple random coalition sampling iterations
    - L1 minimization for sparse signal reconstruction
    - Averaging across iterations for final Shapley values
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        m: int = 100,  # Number of measurements per iteration
        t: int = 50,  # Number of iterations
        tolerance: float = 1e-3,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize Random CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            m: Number of measurements per iteration (default: 100, must be > 0)
            t: Number of iterations (default: 50, must be > 0)
            tolerance: L1 optimization tolerance
            verbose: Whether to print progress
            seed: Random seed for reproducibility
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

        # Validate inputs
        if m <= 0:
            raise ValueError(f"Number of measurements (m) must be positive, got {m}")
        if t <= 0:
            raise ValueError(f"Number of iterations (t) must be positive, got {t}")

        self.m = m
        self.t = t
        self.tolerance = tolerance
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs
    ) -> "RandomCSExplainer":
        """Fit the explainer to data.

        Args:
            X: Input data or graph
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if isinstance(X, nx.Graph):
            self.graph = X
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            builder = GraphBuilder()
            self.graph = builder.from_correlation(X, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self.n = self.graph.number_of_nodes()
        self.nodes = list(self.graph.nodes())

        # Ensure m doesn't exceed reasonable bounds
        self.m = min(self.m, self.n * 10, 500)

        self._fitted = True
        return self

    def explain(
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
    ) -> Dict[int, float]:
        """Compute Random CS Shapley values.

        Args:
            X: Optional input (uses fitted data if None)
            **kwargs: Additional arguments

        Returns:
            Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        def utility_wrapper(S: Set[int]) -> float:
            node_set = {self.nodes[i] for i in S}
            return self.characteristic_function(node_set, self.graph)

        shapley_indices = self._compute_shapley(utility_wrapper)

        # Map back to node names
        shapley_values = {self.nodes[i]: value for i, value in shapley_indices.items()}

        return shapley_values

    def _bernoulli_matrix(self, m: int, n: int) -> np.ndarray:
        """Generate Bernoulli measurement matrix.

        Following the paper's implementation:
        Creates a random matrix with entries +1/sqrt(m) or -1/sqrt(m)
        """
        arr = np.random.random([m, n])
        sp = np.random.binomial(1, arr).astype(np.float64)
        sp[sp == 1] = 1.0 / np.sqrt(m)
        sp[sp == 0] = -1.0 / np.sqrt(m)
        return sp

    def _random_coalition(
        self, available_players: list, size: Optional[int] = None
    ) -> set:
        """Generate a random coalition of players.

        Args:
            available_players: List of player indices to sample from
            size: Coalition size (random if None)

        Returns:
            Set of player indices forming the coalition
        """
        if size is None:
            size = np.random.randint(0, len(available_players) + 1)

        if size == 0:
            return set()
        elif size >= len(available_players):
            return set(available_players)
        else:
            return set(random.sample(available_players, size))

    def _compute_shapley(self, utility_func: Callable) -> Dict[int, float]:
        """Compute Shapley values using Random CS method.

        Algorithm from paper:
        1. Generate random Bernoulli measurement matrix A
        2. For t iterations:
           - Sample random coalitions
           - Compute marginal contributions
           - Apply measurement matrix
        3. Average measurements across iterations
        4. Solve L1 minimization to recover Shapley values
        """
        # Initialize measurement matrix
        A = self._bernoulli_matrix(self.m, self.n)

        # Storage for measurements across iterations (matching paper)
        y = {}

        # Requests tracking (for debugging/statistics)
        requests = []

        for iteration in range(self.t):
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Random CS iteration {iteration + 1}/{self.t}")

            # Sample a random coalition (matching paper's approach)
            coalition_size = np.random.randint(
                1, self.n + 1
            )  # Note: starts from 1, not 0
            row_idx = self._random_coalition(list(range(self.n)), coalition_size)

            # Compute marginal contributions for all players
            phi_arr = np.zeros(self.n)

            for player in range(self.n):
                if player in row_idx:
                    # Player is in the coalition
                    coalition_with = row_idx.copy()
                    coalition_without = row_idx - {player}
                else:
                    # Player is not in the coalition
                    coalition_without = row_idx.copy()
                    coalition_with = row_idx | {player}

                # Compute marginal contribution
                v_with = utility_func(coalition_with)
                v_without = utility_func(coalition_without) if coalition_without else 0
                phi_arr[player] = v_with - v_without

                # Track requests
                requests.append(coalition_with)
                if coalition_without:
                    requests.append(coalition_without)

            # Apply measurement matrix (matching paper)
            y_m = A @ phi_arr
            y[iteration] = y_m

        # Average the measurements (matching paper's approach)
        y_bar = np.sum(np.array(list(y.values())).T, axis=1) * (1 / self.t)

        # Compute s_bar (matching paper)
        s_bar = utility_func(set(range(self.n))) / self.n

        # L1 minimization reconstruction (matching paper)
        if CVXPY_AVAILABLE:
            try:
                # Create variable (matching paper: shape=(n,1))
                x_l1 = cp.Variable(shape=(self.n, 1))

                # Create constraint (matching paper)
                constraints = [
                    cp.norm(A @ (s_bar + x_l1) - y_bar[:, np.newaxis]) <= self.tolerance
                ]

                # Form objective (matching paper)
                obj = cp.Minimize(cp.norm(x_l1, 1))

                # Form and solve problem
                prob = cp.Problem(obj, constraints)
                prob.solve()

                if prob.status in ["optimal", "optimal_inaccurate"]:
                    # Compute final result (matching paper: CS_res = s_bar + x_l1.value)
                    CS_res = s_bar + x_l1.value.flatten()
                    shapley_values = {i: CS_res[i] for i in range(self.n)}
                else:
                    if self.verbose:
                        print(f"L1 minimization status: {prob.status}, using fallback")
                    # Fallback to simple averaging
                    shapley_values = {i: np.mean(y_bar) for i in range(self.n)}
            except Exception as e:
                if self.verbose:
                    print(f"L1 minimization failed: {e}, using fallback")
                # Fallback to simple averaging
                shapley_values = {i: np.mean(y_bar) for i in range(self.n)}
        else:
            # Without CVXPY, use simple averaging as fallback
            if self.verbose:
                print("CVXPY not available, using simple averaging")
            shapley_values = {i: np.mean(y_bar) for i in range(self.n)}

        # Report statistics if verbose
        if self.verbose:
            unique_requests = len(set(map(tuple, map(sorted, requests))))
            total_possible = 2**self.n - 1
            print(f"Used coalitions: {unique_requests}")
            print(f"Total possible: {total_possible}")
            print(f"Coverage: {100 * unique_requests / total_possible:.2f}%")

        return shapley_values

    def explain_batch(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        n_runs: int = 10,
        **kwargs,
    ) -> Dict[int, tuple]:
        """Run multiple Random CS explanations and return statistics.

        Args:
            X: Optional input (uses fitted data if None)
            n_runs: Number of independent runs
            **kwargs: Additional arguments

        Returns:
            Dictionary with mean and std of Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        all_values = []

        for run in range(n_runs):
            if self.verbose:
                print(f"Batch run {run + 1}/{n_runs}")
            values = self.explain()
            all_values.append(list(values.values()))

        all_values = np.array(all_values)

        results = {}
        for i, node in enumerate(self.nodes):
            results[node] = (
                np.mean(all_values[:, i]),  # mean
                np.std(all_values[:, i]),  # standard deviation
            )

        return results
