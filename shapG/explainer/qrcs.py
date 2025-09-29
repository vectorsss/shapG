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
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


# Constants for QR-CS algorithm
DEFAULT_MAX_COALITIONS = 5000  # Maximum coalition size for memory safety
DEFAULT_MAX_MEASUREMENTS = 500  # Maximum number of measurements


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
        use_fast_fallback: bool = False,
        verbose: bool = False
    ):
        """Initialize QR-CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            n_measurements: Number of compressed measurements (auto-computed if None)
            tolerance: L1 optimization tolerance
            use_fast_fallback: If True, use fast mean approximation (like original benchmark)
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)
        self.n_measurements = n_measurements
        self.tolerance = tolerance
        self.use_fast_fallback = use_fast_fallback

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

        # Initialize QR-CS components (matching original implementation)
        self.m = min(2**(self.n - 1), DEFAULT_MAX_COALITIONS)

        # Warn about memory for large graphs
        if self.n > 15 and self.m >= 1000:
            import warnings
            warnings.warn(
                f"QR-CS for {self.n} nodes will use {self.m} coalitions. "
                f"This may consume significant memory. Consider using BlockQRCSExplainer "
                f"for better memory efficiency on large graphs.",
                ResourceWarning
            )

        if self.n_measurements is None:
            self.l = min(int(2 * self.n * np.log(max(self.n, 2))), self.m // 2, DEFAULT_MAX_MEASUREMENTS)
        else:
            self.l = min(self.n_measurements, self.m)

        # Pre-compute components
        self.weights = self._compute_weights_per_index()
        self.Psi = self._get_dct_basis()
        self.B, self.selected_coalitions, self.Q_from_V = self._compute_measurement_matrix()

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

    def _compute_weights_per_index(self) -> np.ndarray:
        """Compute Shapley weights aligned per coalition index (bit-count based)."""
        weights = np.zeros(self.m)
        fact_cache = {k: math.factorial(k) for k in range(self.n + 1)}
        for j in range(self.m):
            # Coalition size is the number of set bits in j (players other than the target)
            s = int(bin(j).count("1"))
            # w = s!(n-s-1)! / n!
            w = (fact_cache[s] * fact_cache[self.n - s - 1]) / fact_cache[self.n]
            weights[j] = w
        return weights

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

    def _compute_measurement_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute measurement matrix using QR decomposition.

        Per the paper: We perform QR decomposition with column pivoting on Ψ^T
        to select the most linearly independent rows (coalitions).
        """
        # QR decomposition with pivoting on Ψ^T (DCT basis transpose)
        V = self.Psi.T
        Q, R, P = qr(V, pivoting=True, mode='economic' if self.m > 1000 else 'full')

        # Select first l pivoted indices (most important coalitions)
        selected_indices = P[:self.l]

        # Create binary measurement matrix B
        B = np.zeros((self.l, self.m))
        for i, idx in enumerate(selected_indices):
            B[i, idx] = 1

        # Return also Q from the QR on V so that Θ = B @ Ψ @ Q
        return B, selected_indices, Q

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
        """Compute Shapley values using QR-CS method.

        Algorithm from paper:
        1. For each player i, measure marginal contributions for selected coalitions
        2. Reconstruct full marginal contribution vector via CS
        3. Compute Shapley value as weighted sum
        """
        shapley_values = {}

        # Use Q from the QR decomposition on V = Ψ^T (per paper eq. 18)
        Theta = self.B @ self.Psi @ self.Q_from_V

        for player in range(self.n):
            if self.verbose:
                print(f"Computing QR-CS for player {player+1}/{self.n}")

            # Step 1: Measure marginal contributions for selected coalitions
            y = np.zeros(self.l)

            for i, coal_idx in enumerate(self.selected_coalitions):
                coalition = self._index_to_coalition(coal_idx, player)

                v_with = utility_func(coalition | {player})
                v_without = utility_func(coalition) if coalition else 0
                y[i] = v_with - v_without

            # Step 2: Compressed sensing reconstruction
            if self.use_fast_fallback:
                # Fast approximation: use mean of sampled contributions
                # This matches the original benchmark's behavior (due to its bug)
                shapley_values[player] = np.mean(y) if len(y) > 0 else 0
            else:
                try:
                    # Full L1 minimization to recover sparse representation
                    s_hat = self._l1_minimization(Theta, y)

                    # Transform back to original domain
                    u_hat = self.Psi @ self.Q_from_V @ s_hat

                    # Step 3: Compute Shapley value as weighted sum
                    shapley_values[player] = float(np.dot(self.weights[:len(u_hat)], u_hat))
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"QR-CS L1 reconstruction failed for player {player}: {str(e)}. "
                        f"Falling back to mean of measured contributions. "
                        f"Consider installing cvxpy for better results: pip install cvxpy",
                        RuntimeWarning
                    )
                    if self.verbose:
                        print(f"Warning: QR-CS reconstruction failed for player {player}, using fallback: {str(e)}")
                    # Fallback: use mean of measured contributions
                    shapley_values[player] = np.mean(y) if len(y) > 0 else 0

        return shapley_values

    def _l1_minimization(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L1 minimization using CVXPY (like the paper) or scipy fallback."""
        m, n = A.shape

        if CVXPY_AVAILABLE:
            # Use CVXPY for much faster L1 minimization (matching paper's implementation)
            try:
                x = cp.Variable(n)
                objective = cp.Minimize(cp.norm(x, 1))
                constraints = [cp.norm(A @ x - b) <= self.tolerance]
                prob = cp.Problem(objective, constraints)
                prob.solve()

                if prob.status in ['optimal', 'optimal_inaccurate']:
                    return x.value
                else:
                    import warnings
                    warnings.warn(
                        f"CVXPY L1 minimization failed with status: {prob.status}. "
                        f"Falling back to least squares approximation.",
                        RuntimeWarning
                    )
                    # Fall back to least squares if CVXPY fails
                    return np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception as e:
                import warnings
                warnings.warn(
                    f"CVXPY solver failed: {e}. Falling back to least squares approximation.",
                    RuntimeWarning
                )
                if self.verbose:
                    print(f"CVXPY solver failed: {e}, using least squares fallback")
                try:
                    return np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    return np.zeros(n)
        else:
            # Original scipy implementation as fallback
            # Get initial guess (improved from original)
            try:
                x0 = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                try:
                    x0 = np.linalg.pinv(A) @ b
                except Exception:
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
