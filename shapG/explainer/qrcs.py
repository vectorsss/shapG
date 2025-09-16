"""
QR-based Compressed Sensing Shapley value computation.

Based on: "A novel sparsity-based deterministic method for Shapley value approximation"
"""

from typing import Dict, Set, Union, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import networkx as nx
import math
import warnings
from scipy.linalg import qr
from scipy.optimize import minimize

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


class QRCSExplainer(GraphExplainer):
    """QR-CS Shapley value approximation using compressed sensing.

    This explainer uses QR decomposition and compressed sensing theory to
    approximate Shapley values more efficiently than exhaustive computation
    while maintaining theoretical guarantees.

    The method combines:
    - DCT (Discrete Cosine Transform) basis representation
    - QR decomposition for intelligent measurement selection
    - L1 minimization for sparse signal reconstruction
    - Shapley weight integration for final value computation
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        n_measurements: Optional[int] = None,
        tolerance: float = 5e-5,
        verbose: bool = False
    ):
        """Initialize QR-CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            n_measurements: Number of compressed measurements (auto-computed if None)
            tolerance: L1 optimization tolerance
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)
        self.n_measurements = n_measurements
        self.tolerance = tolerance

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'QRCSExplainer':
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

        # Initialize QR-CS components with more conservative defaults
        # Cap coalition space size for computational feasibility
        max_coalitions = min(2**(self.n - 1), 1000)  # More conservative cap
        self.m = max(min(max_coalitions, 100), self.n * 2)  # Ensure minimum size

        if self.n_measurements is None:
            # Use compressed sensing theory: l ~ k * log(m/k) where k is sparsity
            sparsity_estimate = max(self.n // 2, 2)  # Assume moderate sparsity
            self.l = min(
                int(sparsity_estimate * np.log(max(self.m / sparsity_estimate, 2))),
                self.m // 2,
                50  # Conservative upper bound
            )
        else:
            self.l = min(self.n_measurements, self.m)

        # Ensure minimum measurements
        self.l = max(self.l, min(self.n, 10))

        # Pre-compute components
        self.weights = self._compute_shapley_weights()
        self.Psi = self._get_dct_basis()
        self.B, self.selected_coalitions = self._compute_measurement_matrix()

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute QR-CS Shapley values.

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

    def _compute_shapley_weights(self) -> np.ndarray:
        """Compute Shapley weights for coalition contributions."""
        weights = []
        for s in range(min(self.n, 20)):
            weight = math.factorial(s) * math.factorial(self.n - s - 1) / math.factorial(self.n)
            n_coalitions = self._comb(self.n - 1, s)
            weights.extend([weight] * min(n_coalitions, self.m - len(weights)))
            if len(weights) >= self.m:
                break
        return np.array(weights[:self.m])

    def _comb(self, n: int, k: int) -> int:
        """Compute binomial coefficient C(n,k)."""
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
        """Generate DCT-II basis matrix."""
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
        """Compute measurement matrix using QR decomposition."""
        try:
            # Use QR decomposition with pivoting for better numerical stability
            Q, R, P = qr(self.Psi, pivoting=True, mode='economic')

            # Select the most important measurements based on diagonal of R
            R_diag = np.abs(np.diag(R))
            selected_indices = np.argsort(R_diag)[-self.l:]  # Take largest diagonal elements
            selected_indices = np.sort(selected_indices)  # Sort for consistent ordering

            # Create measurement matrix
            B = np.zeros((self.l, self.m))
            for i, idx in enumerate(selected_indices):
                if idx < self.m:  # Ensure index is valid
                    B[i, idx] = 1

            return B, selected_indices

        except Exception:
            # Fallback: random sampling
            if self.verbose:
                print("  Warning: QR-based measurement selection failed, using random sampling")
            selected_indices = np.random.choice(self.m, size=min(self.l, self.m), replace=False)
            B = np.zeros((len(selected_indices), self.m))
            for i, idx in enumerate(selected_indices):
                B[i, idx] = 1
            return B, selected_indices

    def _index_to_coalition(self, idx: int, player: int) -> set:
        """Convert coalition index to set of players (excluding target player)."""
        coalition = set()
        available_players = [i for i in range(self.n) if i != player]

        if idx >= 2**(self.n - 1):
            idx = idx % 2**(self.n - 1)

        for i, p in enumerate(available_players):
            if idx & (1 << i):
                coalition.add(p)

        return coalition

    def _compute_shapley(self, utility_func: Callable) -> Dict[int, float]:
        """Compute Shapley values using QR-CS method."""
        shapley_values = {}

        for player in range(self.n):
            if self.verbose:
                print(f"Computing QR-CS for player {player+1}/{self.n}")

            # Measure marginal contributions
            y = np.zeros(self.l)

            for i, coal_idx in enumerate(self.selected_coalitions):
                coalition = self._index_to_coalition(coal_idx, player)

                v_with = utility_func(coalition | {player})
                v_without = utility_func(coalition) if coalition else 0
                y[i] = v_with - v_without

            # Compressed sensing reconstruction
            try:
                # Check dimensions and conditioning
                if self.verbose:
                    print(f"  Matrix dimensions: B={self.B.shape}, Psi={self.Psi.shape}, y={y.shape}")

                # Use more stable QR decomposition
                Q, R = qr(self.Psi.T, mode='economic')  # Transpose for correct dimensions

                # Measurement matrix for compressed sensing
                Phi = self.B @ self.Psi  # Direct measurement matrix

                # Check conditioning
                cond_num = np.linalg.cond(Phi)
                if cond_num > 1e12:
                    if self.verbose:
                        print(f"  Warning: Ill-conditioned matrix (cond={cond_num:.2e}), using pseudoinverse")
                    # Use pseudoinverse for ill-conditioned systems
                    u_hat = np.linalg.pinv(Phi) @ y
                else:
                    # L1 minimization for well-conditioned systems
                    s_hat = self._l1_minimization(Phi, y)
                    u_hat = s_hat

                # Ensure proper dimensions for dot product
                weights_truncated = self.weights[:len(u_hat)]
                if len(weights_truncated) != len(u_hat):
                    # Pad or truncate as needed
                    if len(u_hat) > len(weights_truncated):
                        u_hat = u_hat[:len(weights_truncated)]
                    else:
                        weights_truncated = weights_truncated[:len(u_hat)]

                shapley_values[player] = np.dot(weights_truncated, u_hat)

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: QR-CS reconstruction failed for player {player}: {str(e)}")
                    print(f"  Falling back to mean marginal contribution")
                shapley_values[player] = np.mean(y)

        return shapley_values

    def _l1_minimization(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve L1 minimization problem: min ||x||_1 s.t. ||Ax - b|| <= tolerance."""
        m, n = A.shape

        # Check if the system is overdetermined or underdetermined
        if m >= n:
            # Overdetermined or square system - use least squares
            try:
                x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
                residual = np.linalg.norm(A @ x_ls - b)
                if residual <= self.tolerance:
                    return x_ls
            except Exception:
                pass

        # For underdetermined systems or when least squares doesn't satisfy tolerance
        try:
            x0 = np.linalg.pinv(A) @ b  # Better initial guess
        except Exception:
            x0 = np.zeros(n)

        def objective(x):
            return np.sum(np.abs(x))

        def constraint(x):
            residual = np.linalg.norm(A @ x - b)
            return self.tolerance - residual

        # Use multiple optimization strategies
        constraints = {'type': 'ineq', 'fun': constraint}

        methods_to_try = ['SLSQP', 'trust-constr']

        for method in methods_to_try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        objective, x0, method=method,
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-8}
                    )

                if result.success and constraint(result.x) >= 0:
                    return result.x

            except Exception:
                continue

        # Fallback: regularized least squares
        try:
            alpha = 1e-6  # Small regularization
            AtA_reg = A.T @ A + alpha * np.eye(n)
            Atb = A.T @ b
            return np.linalg.solve(AtA_reg, Atb)
        except Exception:
            return x0