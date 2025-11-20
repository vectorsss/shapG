"""
Block QR-based Compressed Sensing Shapley value computation.

Based on Section 6.3 of: "A novel sparsity-based deterministic method for Shapley value approximation"
This implements the block version for high-dimensional problems with parallel/distributed computation support.
"""

from typing import Dict, Set, Union, Optional, Tuple, Callable, List
import numpy as np
import pandas as pd
import networkx as nx
import math
import warnings
from scipy.linalg import qr
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


class BlockQRCSExplainer(GraphExplainer):
    """Block QR-CS Shapley value approximation using compressed sensing.

    This explainer extends QRCSExplainer to handle high-dimensional problems
    by dividing the computation into blocks that can be processed in parallel.

    From the paper (Section 6.3):
    - Divides marginal contributions into k non-overlapping blocks
    - Each block has its own sparsifying basis and measurement matrix
    - Blocks can be computed in parallel or distributed fashion
    - Reduces memory requirements from O(m^2) to O(max_j(m_j^2))

    The method is particularly suitable for:
    - Large-scale problems where m (number of coalitions) is very large
    - Distributed computing environments
    - Memory-constrained systems
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        n_blocks: int = 4,
        block_sizes: Optional[List[int]] = None,
        n_measurements_per_block: Optional[List[int]] = None,
        tolerance: float = 5e-5,
        use_fast_fallback: bool = False,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize Block QR-CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            n_blocks: Number of blocks to divide the problem into (must be > 0)
            block_sizes: Size of each block (auto-computed if None)
            n_measurements_per_block: Measurements per block (auto-computed if None)
            tolerance: L1 optimization tolerance
            use_fast_fallback: If True, use fast mean approximation
            parallel: If True, compute blocks in parallel
            max_workers: Maximum number of parallel workers (None for default)
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

        # Validate inputs
        if n_blocks <= 0:
            raise ValueError(f"Number of blocks must be positive, got {n_blocks}")

        self.n_blocks = n_blocks
        self.block_sizes = block_sizes
        self.n_measurements_per_block = n_measurements_per_block
        self.tolerance = tolerance
        self.use_fast_fallback = use_fast_fallback
        self.parallel = parallel
        self.max_workers = max_workers

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'BlockQRCSExplainer':
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

        # Compute total coalition space size
        self.m_total = min(2**(self.n - 1), 5000)  # Cap for memory

        # Precompute global per-index weights aligned with coalition indexing
        self.weights_full = self._compute_global_weights()

        # Divide into blocks
        self._setup_blocks()

        self._fitted = True
        return self

    def _setup_blocks(self):
        """Setup block decomposition following paper's approach."""
        # Compute block sizes if not provided
        if self.block_sizes is None:
            base_size = self.m_total // self.n_blocks
            remainder = self.m_total % self.n_blocks
            self.block_sizes = [base_size + (1 if i < remainder else 0)
                               for i in range(self.n_blocks)]

        # Compute measurements per block if not provided
        if self.n_measurements_per_block is None:
            self.n_measurements_per_block = [
                min(int(2 * math.log(max(size, 2))), size // 2, 100)
                for size in self.block_sizes
            ]

        # Create block ranges
        self.block_ranges = []
        start = 0
        for size in self.block_sizes:
            self.block_ranges.append((start, start + size))
            start += size

        if self.verbose:
            print(f"Block setup: {self.n_blocks} blocks")
            for i, (size, n_meas, (start, end)) in enumerate(
                zip(self.block_sizes, self.n_measurements_per_block, self.block_ranges)
            ):
                print(f"  Block {i}: size={size}, measurements={n_meas}, range=[{start}:{end})")

    def _compute_global_weights(self) -> np.ndarray:
        """Compute Shapley weights per global coalition index via popcount."""
        weights = np.zeros(self.m_total)
        # factorial cache
        fact_cache = {k: math.factorial(k) for k in range(self.n + 1)}
        for j in range(self.m_total):
            s = int(bin(j).count("1"))
            w = (fact_cache[s] * fact_cache[self.n - s - 1]) / fact_cache[self.n]
            weights[j] = w
        return weights

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """Compute Block QR-CS Shapley values.

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

        # Compute Shapley values using block decomposition
        shapley_indices = self._compute_shapley_blocks(utility_wrapper)

        # Map back to node names
        shapley_values = {self.nodes[i]: value for i, value in shapley_indices.items()}

        return shapley_values

    def _compute_shapley_blocks(self, utility_func: Callable) -> Dict[int, float]:
        """Compute Shapley values using block decomposition.

        Following paper's Section 6.3:
        - Each block processes a subset of coalitions
        - Blocks can be computed in parallel
        - Results are combined at the end
        """
        if self.parallel and self.n_blocks > 1:
            return self._compute_blocks_parallel(utility_func)
        else:
            return self._compute_blocks_sequential(utility_func)

    def _compute_blocks_parallel(self, utility_func: Callable) -> Dict[int, float]:
        """Compute blocks in parallel using thread pool."""
        if self.verbose:
            print(f"Computing {self.n_blocks} blocks in parallel...")

        # Aggregate results from all blocks (sum contributions across blocks)
        shapley_sum = {i: 0.0 for i in range(self.n)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all block computations
            future_to_block = {
                executor.submit(
                    self._compute_single_block,
                    block_idx,
                    utility_func
                ): block_idx
                for block_idx in range(self.n_blocks)
            }

            # Collect results as they complete
            for future in as_completed(future_to_block):
                block_idx = future_to_block[future]
                try:
                    block_shapley = future.result()

                    # Aggregate block contributions by summation
                    for player, value in block_shapley.items():
                        shapley_sum[player] += float(value)

                    if self.verbose:
                        print(f"  Block {block_idx} completed")

                except Exception as e:
                    if self.verbose:
                        print(f"  Block {block_idx} failed: {e}")
                    # On failure, skip (equivalent to adding zero)

        # Combine block results (sum across blocks to reconstruct full dot product)
        shapley_values = {player: shapley_sum[player] for player in range(self.n)}

        return shapley_values

    def _compute_blocks_sequential(self, utility_func: Callable) -> Dict[int, float]:
        """Compute blocks sequentially."""
        if self.verbose:
            print(f"Computing {self.n_blocks} blocks sequentially...")

        # Aggregate results from all blocks (sum contributions across blocks)
        shapley_sum = {i: 0.0 for i in range(self.n)}

        for block_idx in range(self.n_blocks):
            block_shapley = self._compute_single_block(block_idx, utility_func)

            # Aggregate block contributions by summation
            for player, value in block_shapley.items():
                shapley_sum[player] += float(value)

            if self.verbose:
                print(f"  Block {block_idx} completed")

        # Combine block results (sum across blocks)
        shapley_values = {player: shapley_sum[player] for player in range(self.n)}

        return shapley_values

    def _compute_single_block(
        self,
        block_idx: int,
        utility_func: Callable,
        use_fast_override: Optional[bool] = None
    ) -> Dict[int, float]:
        """Compute Shapley values for a single block.

        This implements the core block QR-CS algorithm for one block,
        following the paper's approach but adapted to a subset of coalitions.
        """
        start, end = self.block_ranges[block_idx]
        block_size = self.block_sizes[block_idx]
        n_measurements = self.n_measurements_per_block[block_idx]

        # Generate block-specific weights by slicing global weights
        weights = self.weights_full[start:end]

        # Generate block-specific DCT basis
        Psi = self._get_block_dct_basis(block_size)

        # Compute block measurement matrix using QR and get Q from QR(V)
        B, selected_indices, Q_from_V = self._compute_block_measurement_matrix(
            Psi, n_measurements, block_size
        )

        # Pre-compute sensing matrix Θ = B @ Ψ @ Q (Q from QR on V = Ψ^T)
        Theta = B @ Psi @ Q_from_V

        shapley_values = {}

        # Allow callers (e.g. subclasses) to override the fast-fallback choice per block
        use_fast = self.use_fast_fallback if use_fast_override is None else use_fast_override

        for player in range(self.n):
            # Measure marginal contributions for selected coalitions in this block
            y = np.zeros(n_measurements)

            for i, local_idx in enumerate(selected_indices):
                # Map local block index to global coalition index
                global_idx = start + local_idx
                coalition = self._index_to_coalition(global_idx, player)

                v_with = utility_func(coalition | {player})
                v_without = utility_func(coalition) if coalition else 0
                y[i] = v_with - v_without

            # Reconstruct using compressed sensing
            if use_fast:
                shapley_values[player] = np.mean(y) if len(y) > 0 else 0
            else:
                try:
                    # L1 minimization to recover sparse representation
                    s_hat = self._l1_minimization_block(Theta, y)

                    # Transform back to original domain
                    u_hat = Psi @ Q_from_V @ s_hat

                    # Compute Shapley value as weighted sum for this block
                    shapley_values[player] = np.dot(weights[:len(u_hat)], u_hat)

                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Block QR-CS reconstruction failed for block {block_idx}, player {player}: {str(e)}. "
                        f"Falling back to mean of measured contributions. "
                        f"Consider installing cvxpy for better results: pip install cvxpy",
                        RuntimeWarning
                    )
                    if self.verbose:
                        print(f"    Block {block_idx}, player {player} reconstruction failed: {e}")
                    shapley_values[player] = np.mean(y) if len(y) > 0 else 0

        return shapley_values

    def _compute_block_weights(self, start: int, end: int) -> np.ndarray:
        """Deprecated: kept for backward-compat; now we slice global weights."""
        return self.weights_full[start:end]

    def _get_block_dct_basis(self, block_size: int) -> np.ndarray:
        """Generate DCT-II basis matrix for this block."""
        if block_size > 100:
            return np.eye(block_size)

        Psi = np.zeros((block_size, block_size))
        for k in range(block_size):
            for n in range(block_size):
                if k == 0:
                    Psi[n, k] = np.sqrt(1/block_size)
                else:
                    Psi[n, k] = np.sqrt(2/block_size) * np.cos(np.pi * k * (n + 0.5) / block_size)
        return Psi

    def _compute_block_measurement_matrix(
        self,
        Psi: np.ndarray,
        n_measurements: int,
        block_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute measurement matrix for this block using QR decomposition and return Q from QR(V)."""
        V = Psi.T
        Q, R, P = qr(V, pivoting=True, mode='economic' if block_size > 100 else 'full')

        # Select first n_measurements pivoted indices
        selected_indices = P[:n_measurements]

        # Create binary measurement matrix B
        B = np.zeros((n_measurements, block_size))
        for i, idx in enumerate(selected_indices):
            B[i, idx] = 1

        return B, selected_indices, Q

    def _index_to_coalition(self, idx: int, player: int) -> set:
        """Convert coalition index to set of players (excluding target player)."""
        coalition = set()
        available_players = [i for i in range(self.n) if i != player]

        # Convert idx to Python int to use arbitrary precision arithmetic
        # This avoids overflow for large coalition indices (> 2^31)
        idx = int(idx)

        if idx >= 2**(self.n - 1):
            idx = idx % 2**(self.n - 1)

        for i, p in enumerate(available_players):
            # Bit shift with Python int (arbitrary precision)
            if idx & (1 << int(i)):
                coalition.add(p)

        return coalition

    def _l1_minimization_block(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L1 minimization for block using CVXPY or scipy fallback."""
        m, n = A.shape

        if CVXPY_AVAILABLE:
            try:
                x = cp.Variable(n)
                objective = cp.Minimize(cp.norm(x, 1))
                constraints = [cp.norm(A @ x - b) <= self.tolerance]
                prob = cp.Problem(objective, constraints)
                prob.solve()

                if prob.status in ['optimal', 'optimal_inaccurate']:
                    return x.value
                else:
                    warnings.warn(
                        f"CVXPY L1 minimization failed with status: {prob.status}. "
                        f"Falling back to least squares approximation.",
                        RuntimeWarning
                    )
                    return np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception as e:
                warnings.warn(
                    f"CVXPY solver failed: {e}. Falling back to least squares approximation.",
                    RuntimeWarning
                )
                try:
                    return np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    return np.zeros(n)
        else:
            # Scipy fallback
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
                    options={'maxiter': 100, 'ftol': 1e-6}
                )

            return result.x if result.success else x0
