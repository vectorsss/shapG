"""
Random Projection QRCS (RP-QRCS) Shapley value computation.

This module implements an improved QRCS algorithm that addresses the coalition
sampling bias problem in the original QRCS implementation.

Key Improvements:
- Stratified sampling by coalition size (no index truncation bias)
- Random projection matrices instead of DCT (works with non-contiguous samples)
- Shapley-weighted allocation across coalition size strata
- Maintains compressed sensing framework for algorithm comparison

References:
    [1] Original QRCS: "A novel sparsity-based deterministic method for
        Shapley value approximation"
    [2] Compressed Sensing: Candès, E. J., & Wakin, M. B. (2008).
        "An Introduction to Compressive Sampling"
    [3] Leverage Score Sampling: Musco, C., & Witter, R. T. (2025).
        "Provably Accurate Shapley Value Estimation via Leverage Score Sampling"
"""

from typing import Dict, Set, Union, Optional, Tuple, List, Callable
from dataclasses import dataclass
from math import factorial, sqrt, log
import numpy as np
import pandas as pd
import networkx as nx
import warnings

from scipy.special import comb
from scipy.optimize import minimize

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .base import GraphExplainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree
from ..utils.graph_construction import GraphBuilder


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SamplingStats:
    """Statistics about the stratified sampling process."""
    n_players: int
    total_budget: int
    allocations_by_size: np.ndarray
    actual_samples_by_size: np.ndarray
    coverage_by_size: np.ndarray  # fraction of coalitions sampled per stratum

    def __str__(self) -> str:
        lines = [f"SamplingStats(n={self.n_players}, budget={self.total_budget})"]
        for s in range(len(self.allocations_by_size)):
            if self.allocations_by_size[s] > 0:
                lines.append(
                    f"  size {s}: {int(self.actual_samples_by_size[s])} samples "
                    f"({self.coverage_by_size[s]*100:.1f}% coverage)"
                )
        return "\n".join(lines)


# =============================================================================
# Stratified Coalition Sampler
# =============================================================================

class StratifiedCoalitionSampler:
    """
    Stratified coalition sampler with Shapley-weighted allocation.

    This sampler addresses the bias problem in original QRCS by:
    1. Stratifying coalitions by size (0 to n-1)
    2. Allocating samples proportionally to Shapley weight importance
    3. Sampling coalitions directly (not via biased index enumeration)

    Parameters
    ----------
    n : int
        Number of players
    budget : int
        Total sampling budget (number of coalitions to sample)
    allocation_strategy : str
        How to allocate budget across strata:
        - 'shapley_weighted': proportional to Shapley weight × stratum size
        - 'uniform': equal allocation per stratum
        - 'leverage': based on leverage scores (like LeverageScoreExplainer)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        n: int,
        budget: int,
        allocation_strategy: str = 'shapley_weighted',
        seed: Optional[int] = None
    ):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        self.n = n
        self.budget = budget
        self.allocation_strategy = allocation_strategy
        self.rng = np.random.default_rng(seed)

        # Precompute Shapley weights and coalition counts
        self._precompute()

        # Compute allocation
        self.allocations = self._compute_allocations()

    def _precompute(self):
        """Precompute Shapley weights and coalition counts per size."""
        n = self.n

        # Factorial cache
        self._fact = {k: factorial(k) for k in range(n + 1)}

        # Shapley weights by coalition size: w_s = s!(n-s-1)!/n!
        self.shapley_weights = np.zeros(n)
        for s in range(n):
            self.shapley_weights[s] = (
                self._fact[s] * self._fact[n - s - 1] / self._fact[n]
            )

        # Number of coalitions of each size (for a single player, n-1 others)
        self.n_coalitions_by_size = np.array([
            comb(n - 1, s, exact=True) for s in range(n)
        ])

        # Total coalitions
        self.total_coalitions = 2 ** (n - 1)

    def _compute_allocations(self) -> np.ndarray:
        """Compute sample allocation across strata."""
        n = self.n

        if self.allocation_strategy == 'shapley_weighted':
            # Importance = Shapley weight × number of coalitions of that size
            importance = self.shapley_weights * self.n_coalitions_by_size
            importance_sum = importance.sum()
            if importance_sum > 0:
                importance /= importance_sum
            else:
                importance = np.ones(n) / n

        elif self.allocation_strategy == 'uniform':
            # Equal allocation per stratum
            importance = np.ones(n) / n

        elif self.allocation_strategy == 'leverage':
            # Leverage score based allocation (inverse of coalition count)
            leverage = np.zeros(n)
            for s in range(n):
                if self.n_coalitions_by_size[s] > 0:
                    leverage[s] = 1.0 / self.n_coalitions_by_size[s]
            leverage_sum = leverage.sum()
            if leverage_sum > 0:
                importance = leverage / leverage_sum
            else:
                importance = np.ones(n) / n

        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")

        # Allocate budget proportionally
        raw_allocations = importance * self.budget

        # Round down and distribute remainder
        allocations = np.floor(raw_allocations).astype(int)

        # Cap at actual number of coalitions per stratum
        allocations = np.minimum(allocations, self.n_coalitions_by_size)

        # Distribute remaining budget to highest importance strata
        remaining = self.budget - allocations.sum()
        if remaining > 0:
            # Sort by importance descending
            indices = np.argsort(-importance)
            for idx in indices:
                if remaining <= 0:
                    break
                # How many more can we add to this stratum?
                can_add = min(
                    remaining,
                    self.n_coalitions_by_size[idx] - allocations[idx]
                )
                allocations[idx] += can_add
                remaining -= can_add

        return allocations

    def sample_all_strata(self, player: int) -> Tuple[List[Set[int]], np.ndarray, np.ndarray]:
        """
        Sample coalitions from all strata for a given player.

        Parameters
        ----------
        player : int
            The player index (0 to n-1) for whom we're sampling coalitions

        Returns
        -------
        coalitions : List[Set[int]]
            List of sampled coalitions (sets of player indices, excluding target player)
        sizes : np.ndarray
            Coalition size for each sampled coalition
        weights : np.ndarray
            Importance weight for each sample (for unbiased estimation)
        """
        other_players = [j for j in range(self.n) if j != player]
        n_others = len(other_players)

        coalitions = []
        sizes = []
        weights = []

        for size in range(self.n):
            n_samples = self.allocations[size]
            if n_samples == 0:
                continue

            # Actual number of coalitions of this size
            n_total = int(comb(n_others, size, exact=True))

            if n_samples >= n_total:
                # Exact enumeration for this stratum
                from itertools import combinations
                for combo in combinations(other_players, size):
                    coalitions.append(set(combo))
                    sizes.append(size)
                    # Weight = 1 for exact enumeration
                    weights.append(1.0)
            else:
                # Random sampling without replacement
                sampled = self._sample_coalitions_of_size(other_players, size, n_samples)
                for coalition in sampled:
                    coalitions.append(coalition)
                    sizes.append(size)
                    # Weight = (total coalitions of this size) / (samples of this size)
                    # This ensures unbiased estimation
                    weights.append(n_total / n_samples)

        return coalitions, np.array(sizes), np.array(weights)

    def _sample_coalitions_of_size(
        self,
        players: List[int],
        size: int,
        n_samples: int
    ) -> List[Set[int]]:
        """Sample n_samples coalitions of given size from players."""
        if size == 0:
            return [set() for _ in range(n_samples)]

        if size >= len(players):
            return [set(players) for _ in range(n_samples)]

        # Use rejection sampling for uniqueness
        sampled = []
        seen = set()
        max_attempts = n_samples * 10

        for _ in range(max_attempts):
            if len(sampled) >= n_samples:
                break

            # Sample a random coalition of given size
            indices = self.rng.choice(len(players), size=size, replace=False)
            coalition = frozenset(players[i] for i in indices)

            if coalition not in seen:
                seen.add(coalition)
                sampled.append(set(coalition))

        # If we couldn't get enough unique samples, just return what we have
        if len(sampled) < n_samples:
            warnings.warn(
                f"Could only sample {len(sampled)} unique coalitions of size {size}, "
                f"requested {n_samples}",
                RuntimeWarning
            )

        return sampled

    def get_stats(self) -> SamplingStats:
        """Get statistics about the sampling configuration."""
        coverage = np.zeros(self.n)
        for s in range(self.n):
            if self.n_coalitions_by_size[s] > 0:
                coverage[s] = self.allocations[s] / self.n_coalitions_by_size[s]

        return SamplingStats(
            n_players=self.n,
            total_budget=self.budget,
            allocations_by_size=self.allocations,
            actual_samples_by_size=self.allocations.copy(),
            coverage_by_size=coverage
        )


# =============================================================================
# Random Projection Matrix
# =============================================================================

class RandomProjectionMatrix:
    """
    Random projection matrix for compressed sensing.

    Supports multiple projection types that satisfy RIP (Restricted Isometry Property):
    - Gaussian: entries ~ N(0, 1/l)
    - Bernoulli: entries ~ {-1/sqrt(l), +1/sqrt(l)}
    - Sparse: sparse random projections

    Parameters
    ----------
    n_measurements : int
        Number of measurements (rows)
    n_signals : int
        Signal dimension (columns)
    projection_type : str
        Type of random projection: 'gaussian', 'bernoulli', or 'sparse'
    sparsity : float
        For sparse projections, fraction of non-zero entries per column
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_measurements: int,
        n_signals: int,
        projection_type: str = 'gaussian',
        sparsity: float = 0.1,
        seed: Optional[int] = None
    ):
        self.l = n_measurements
        self.m = n_signals
        self.projection_type = projection_type
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)

        self.matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """Build the random projection matrix."""
        l, m = self.l, self.m

        if self.projection_type == 'gaussian':
            # Gaussian random matrix with proper scaling
            Phi = self.rng.standard_normal((l, m))
            Phi /= sqrt(l)

        elif self.projection_type == 'bernoulli':
            # Bernoulli ±1 random matrix
            Phi = self.rng.choice([-1.0, 1.0], size=(l, m))
            Phi /= sqrt(l)

        elif self.projection_type == 'sparse':
            # Sparse random projection (Achlioptas, 2003)
            Phi = np.zeros((l, m))
            n_nonzero = max(1, int(l * self.sparsity))

            for j in range(m):
                # Random positions for non-zero entries
                positions = self.rng.choice(l, size=n_nonzero, replace=False)
                # Random signs
                signs = self.rng.choice([-1.0, 1.0], size=n_nonzero)
                Phi[positions, j] = signs / sqrt(n_nonzero)

        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

        return Phi

    def project(self, signal: np.ndarray) -> np.ndarray:
        """Project signal to lower dimension."""
        return self.matrix @ signal

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix multiplication operator."""
        return self.matrix @ other


# =============================================================================
# L1 Minimization Solver
# =============================================================================

class L1Solver:
    """
    L1 minimization solver for compressed sensing reconstruction.

    Solves: min ||x||_1 subject to ||Ax - b||_2 <= tolerance

    Uses CVXPY if available, otherwise falls back to scipy.
    """

    def __init__(self, tolerance: float = 1e-4, max_iter: int = 500):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the L1 minimization problem.

        Parameters
        ----------
        A : np.ndarray
            Measurement matrix (l x m)
        b : np.ndarray
            Measurements (l,)

        Returns
        -------
        x : np.ndarray
            Reconstructed signal (m,)
        """
        l, m = A.shape

        if CVXPY_AVAILABLE:
            return self._solve_cvxpy(A, b)
        else:
            return self._solve_scipy(A, b)

    def _solve_cvxpy(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using CVXPY (preferred method)."""
        m = A.shape[1]

        try:
            x = cp.Variable(m)
            objective = cp.Minimize(cp.norm(x, 1))
            constraints = [cp.norm(A @ x - b, 2) <= self.tolerance]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, max_iters=self.max_iter)

            if prob.status in ['optimal', 'optimal_inaccurate']:
                return x.value
            else:
                # Fall back to least squares
                return np.linalg.lstsq(A, b, rcond=None)[0]

        except Exception as e:
            warnings.warn(
                f"CVXPY solver failed: {e}. Falling back to least squares.",
                RuntimeWarning
            )
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _solve_scipy(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using scipy optimization (fallback)."""
        m = A.shape[1]

        # Initial guess via least squares
        try:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            x0 = np.zeros(m)

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
                options={'maxiter': self.max_iter, 'ftol': 1e-6}
            )

        return result.x if result.success else x0


# =============================================================================
# Main Explainer Class
# =============================================================================

class RPQRCSExplainer(GraphExplainer):
    """
    Random Projection QRCS Shapley value explainer.

    This explainer improves upon the original QRCS by:
    1. Using stratified sampling by coalition size (eliminates bias)
    2. Using random projections instead of DCT (works with sampled coalitions)
    3. Maintaining the compressed sensing framework for fair comparison

    The algorithm:
    1. For each player i:
       a. Sample coalitions using stratified Shapley-weighted sampling
       b. Compute marginal contributions for sampled coalitions
       c. Use random projection for dimensionality reduction
       d. Reconstruct full marginal contribution vector via L1 minimization
       e. Compute Shapley value as weighted sum

    Parameters
    ----------
    characteristic_function : CharacteristicFunction, optional
        Function to compute coalition values
    n_samples : int, default=5000
        Total number of coalitions to sample
    n_measurements : int, optional
        Number of compressed measurements (auto-computed if None)
    measurement_ratio : float, default=0.2
        Ratio of measurements to samples (used if n_measurements is None)
    projection_type : str, default='gaussian'
        Type of random projection: 'gaussian', 'bernoulli', or 'sparse'
    allocation_strategy : str, default='shapley_weighted'
        How to allocate samples across coalition sizes
    tolerance : float, default=1e-4
        L1 optimization tolerance
    use_importance_weighting : bool, default=True
        Whether to use importance weights for unbiased estimation
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Examples
    --------
    >>> from shapG import RPQRCSExplainer, CoalitionDegree
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(20, 0.3)
    >>> explainer = RPQRCSExplainer(
    ...     characteristic_function=CoalitionDegree(),
    ...     n_samples=5000,
    ...     verbose=True
    ... )
    >>> values = explainer.fit_explain(G)
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        n_samples: int = 5000,
        n_measurements: Optional[int] = None,
        measurement_ratio: float = 0.2,
        projection_type: str = 'gaussian',
        allocation_strategy: str = 'shapley_weighted',
        tolerance: float = 1e-4,
        use_importance_weighting: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

        self.n_samples = n_samples
        self.n_measurements = n_measurements
        self.measurement_ratio = measurement_ratio
        self.projection_type = projection_type
        self.allocation_strategy = allocation_strategy
        self.tolerance = tolerance
        self.use_importance_weighting = use_importance_weighting
        self.seed = seed

        # Will be set during fit
        self._sampler = None
        self._rng = np.random.default_rng(seed)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'RPQRCSExplainer':
        """
        Fit the explainer to data.

        Parameters
        ----------
        X : array-like or Graph
            Input data or graph

        Returns
        -------
        self
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

        # Initialize stratified sampler
        self._sampler = StratifiedCoalitionSampler(
            n=self.n,
            budget=self.n_samples,
            allocation_strategy=self.allocation_strategy,
            seed=self.seed
        )

        if self.verbose:
            print(f"RP-QRCS Explainer fitted:")
            print(f"  n_players: {self.n}")
            print(f"  n_samples: {self.n_samples}")
            print(f"  projection_type: {self.projection_type}")
            print(f"  allocation_strategy: {self.allocation_strategy}")
            print(f"\nSampling allocation:")
            stats = self._sampler.get_stats()
            for s in range(self.n):
                if self._sampler.allocations[s] > 0:
                    print(f"    size {s}: {self._sampler.allocations[s]} samples")

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute Shapley values using RP-QRCS.

        Parameters
        ----------
        X : array-like or Graph, optional
            Input data (uses fitted data if None)

        Returns
        -------
        shapley_values : Dict[int, float]
            Shapley values for each node
        """
        if X is not None:
            self.fit(X, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        # Create utility wrapper
        def utility_func(S: Set[int]) -> float:
            node_set = {self.nodes[i] for i in S}
            return self.characteristic_function(node_set, self.graph)

        # Compute Shapley values
        shapley_indices = self._compute_shapley(utility_func)

        # Map back to node names
        shapley_values = {self.nodes[i]: value for i, value in shapley_indices.items()}

        return shapley_values

    def _compute_shapley(self, utility_func: Callable[[Set[int]], float]) -> Dict[int, float]:
        """
        Compute Shapley values using RP-QRCS algorithm.

        Parameters
        ----------
        utility_func : callable
            Function that computes coalition value

        Returns
        -------
        shapley_values : Dict[int, float]
            Shapley values indexed by player (0 to n-1)
        """
        shapley_values = {}
        l1_solver = L1Solver(tolerance=self.tolerance)

        for player in range(self.n):
            if self.verbose:
                print(f"Computing Shapley value for player {player + 1}/{self.n}")

            # Step 1: Sample coalitions using stratified sampling
            coalitions, sizes, weights = self._sampler.sample_all_strata(player)
            m = len(coalitions)

            if m == 0:
                shapley_values[player] = 0.0
                continue

            # Step 2: Compute marginal contributions
            marginals = np.zeros(m)
            for j, S in enumerate(coalitions):
                v_with = utility_func(S | {player})
                v_without = utility_func(S) if S else 0.0
                marginals[j] = v_with - v_without

            # Step 3: Compute Shapley value
            if self.use_importance_weighting:
                # Weighted average (unbiased estimator)
                # Group by size and compute weighted contribution
                shapley_value = 0.0
                for size in range(self.n):
                    mask = (sizes == size)
                    if not np.any(mask):
                        continue

                    size_marginals = marginals[mask]
                    size_weights = weights[mask]

                    # Shapley weight for this size
                    shapley_weight = self._sampler.shapley_weights[size]

                    # Number of coalitions of this size
                    n_coalitions = self._sampler.n_coalitions_by_size[size]

                    # Weighted mean of marginals for this size
                    # weight already accounts for sampling probability
                    weighted_sum = np.sum(size_marginals * size_weights)
                    weighted_count = np.sum(size_weights)

                    if weighted_count > 0:
                        size_mean = weighted_sum / weighted_count
                        # Contribution = Shapley_weight × n_coalitions × mean_marginal
                        # But since Shapley formula already has the weight, we just need:
                        shapley_value += shapley_weight * n_coalitions * size_mean

                shapley_values[player] = shapley_value

            else:
                # Use compressed sensing reconstruction
                shapley_value = self._compute_with_cs(
                    marginals, weights, sizes, l1_solver
                )
                shapley_values[player] = shapley_value

        return shapley_values

    def _compute_with_cs(
        self,
        marginals: np.ndarray,
        weights: np.ndarray,
        sizes: np.ndarray,
        l1_solver: L1Solver
    ) -> float:
        """
        Compute Shapley value using compressed sensing reconstruction.

        This maintains the CS framework from original QRCS but with
        random projections instead of DCT.
        """
        m = len(marginals)

        # Determine number of measurements
        if self.n_measurements is not None:
            l = min(self.n_measurements, m)
        else:
            l = max(1, int(m * self.measurement_ratio))

        if l >= m:
            # No compression needed, use weighted average
            weighted_sum = np.sum(marginals * weights)
            weighted_count = np.sum(weights)
            if weighted_count > 0:
                return weighted_sum / weighted_count
            return 0.0

        # Build random projection matrix
        projection = RandomProjectionMatrix(
            n_measurements=l,
            n_signals=m,
            projection_type=self.projection_type,
            seed=self._rng.integers(0, 2**31)
        )

        # Project marginal contributions
        y = projection @ marginals

        # Reconstruct via L1 minimization
        try:
            u_hat = l1_solver.solve(projection.matrix, y)

            # Compute Shapley value as weighted sum
            # Weight by importance weights and Shapley weights
            shapley_value = 0.0
            for size in range(self.n):
                mask = (sizes == size)
                if not np.any(mask):
                    continue

                size_contributions = u_hat[mask]
                size_weights = weights[mask]
                shapley_weight = self._sampler.shapley_weights[size]
                n_coalitions = self._sampler.n_coalitions_by_size[size]

                weighted_sum = np.sum(size_contributions * size_weights)
                weighted_count = np.sum(size_weights)

                if weighted_count > 0:
                    size_mean = weighted_sum / weighted_count
                    shapley_value += shapley_weight * n_coalitions * size_mean

            return shapley_value

        except Exception as e:
            warnings.warn(
                f"CS reconstruction failed: {e}. Falling back to weighted average.",
                RuntimeWarning
            )
            weighted_sum = np.sum(marginals * weights)
            weighted_count = np.sum(weights)
            if weighted_count > 0:
                return weighted_sum / weighted_count
            return 0.0

    def get_sampling_stats(self) -> Optional[SamplingStats]:
        """Get statistics about the sampling configuration."""
        if self._sampler is not None:
            return self._sampler.get_stats()
        return None
