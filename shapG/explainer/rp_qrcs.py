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

from typing import Dict, Set, Union, Optional, Tuple, List, Callable, Any
from dataclasses import dataclass
from math import factorial, sqrt, log
import numpy as np
import pandas as pd
import networkx as nx
import warnings

from scipy.special import comb
from scipy.optimize import minimize
from scipy.fft import dct, idct

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .base import Explainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree


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
        - 'leverage': based on leverage scores (inverse coalition count, deterministic)
        - 'leverage_bernoulli': Bernoulli sampling with leverage scores (adapted from Musco & Witter 2025)
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

        # Compute allocation (skip for leverage_bernoulli, handled during sampling)
        if allocation_strategy != 'leverage_bernoulli':
            self.allocations = self._compute_allocations()
        else:
            self.allocations = None  # Will be computed dynamically

        # Track actual samples for leverage_bernoulli (updated during sampling)
        self._actual_samples_by_size = None

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

        elif self.allocation_strategy == 'leverage_bernoulli':
            # Bernoulli sampling - allocations computed dynamically
            # This is a placeholder, actual sampling happens in sample_all_strata_bernoulli
            return np.zeros(n, dtype=int)

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

    def _compute_leverage_score(self, size: int, n_others: int) -> float:
        """
        Compute leverage score for coalition size.

        Adapted from Lemma 3.2: ℓ_s = C(n-1, s)^{-1}

        Parameters
        ----------
        size : int
            Coalition size
        n_others : int
            Number of other players (n-1)

        Returns
        -------
        float
            Leverage score
        """
        if size < 0 or size > n_others:
            return 0.0
        n_coalitions = comb(n_others, size, exact=False)
        if n_coalitions == 0:
            return 0.0
        return 1.0 / n_coalitions

    def _find_c_binary_search(self, n_others: int, target_budget: int) -> float:
        """
        Find oversampling parameter c via binary search.

        Solves: target_budget ≈ Σ_{s=0}^{n-1} min(N_s, 2c * ℓ_s * N_s)
        where N_s = C(n_others, s), ℓ_s = 1/N_s

        Note: With paired sampling, actual samples ≈ 2 * target_budget

        Parameters
        ----------
        n_others : int
            Number of other players (n-1)
        target_budget : int
            Target number of coalition pairs

        Returns
        -------
        float
            Oversampling parameter c
        """
        def expected_pairs(c):
            """Expected number of coalition pairs sampled."""
            total = 0
            for s in range(n_others + 1):
                n_coalitions = comb(n_others, s, exact=False)
                if n_coalitions == 0:
                    continue
                leverage_score = self._compute_leverage_score(s, n_others)
                prob = min(1.0, 2 * c * leverage_score)
                # Each coalition sampled contributes 1 pair (S, complement)
                expected_sampled = prob * n_coalitions
                total += expected_sampled
            # Paired sampling: if we sample S, we also get complement
            # But we need to avoid double counting for s and n_others-s
            # For simplicity, just count expected samples
            return total / 2  # Each pair counted once

        # Binary search for c
        c_low, c_high = 0.01, 100.0

        for _ in range(50):  # Max iterations
            c_mid = (c_low + c_high) / 2
            pairs = expected_pairs(c_mid)

            if abs(pairs - target_budget) < 1:
                return c_mid

            if pairs < target_budget:
                c_low = c_mid
            else:
                c_high = c_mid

        return c_mid

    def sample_all_strata_bernoulli(
        self,
        player: int
    ) -> Tuple[List[Set[int]], np.ndarray, np.ndarray]:
        """
        Sample coalitions using Bernoulli sampling with leverage scores.

        Adapted from LeverageScoreExplainer (Musco & Witter 2025):
        - Uses leverage scores ℓ_s = C(n-1, s)^{-1}
        - Bernoulli sampling with probability p_s = min(1, 2c·ℓ_s)
        - Paired sampling (S, complement) for variance reduction
        - Oversampling parameter c tuned via binary search

        Parameters
        ----------
        player : int
            The player index (0 to n-1) for whom we're sampling coalitions

        Returns
        -------
        coalitions : List[Set[int]]
            List of sampled coalitions (excluding target player)
        sizes : np.ndarray
            Coalition size for each sampled coalition
        weights : np.ndarray
            Importance weight for each sample (1 / sampling probability).
            This provides the sampling correction factor for unbiased estimation.
            Each coalition uses the weight corresponding to its own size.
        """
        other_players = [j for j in range(self.n) if j != player]
        n_others = len(other_players)

        # Find oversampling parameter c
        # Target: budget/2 pairs (since paired sampling doubles samples)
        target_pairs = self.budget // 2
        c = self._find_c_binary_search(n_others, target_pairs)

        coalitions = []
        sizes = []
        weights = []

        # Track actual samples by size for statistics
        actual_samples = np.zeros(self.n, dtype=int)

        # Sample coalitions by size (only up to halfway to avoid duplicates from pairing)
        # For sizes s and (n-s), we only sample from size s and add complements
        for s in range(n_others // 2 + 1):
            # Leverage score
            leverage_score = self._compute_leverage_score(s, n_others)
            if leverage_score == 0:
                continue

            # Sampling probability
            prob = min(1.0, 2 * c * leverage_score)
            if prob == 0:
                continue

            # Total coalitions of this size
            n_total = int(comb(n_others, s, exact=True))
            if n_total == 0:
                continue

            # Check if this is the middle size (when s == n_others - s)
            is_middle_size = (s == n_others - s)

            # Bernoulli sampling: random number of coalitions
            n_samples = self.rng.binomial(n_total, prob)

            if n_samples == 0:
                continue

            # Sample specific coalitions uniformly without replacement
            sampled = self._sample_coalitions_of_size(other_players, s, n_samples)

            # Importance weight = 1 / sampling probability
            # (Shapley weight is applied separately in the direct estimation formula)
            importance_weight = 1.0 / prob

            if is_middle_size:
                # Middle size: sample from full space but don't use paired sampling
                # Each coalition is added once (complements are in same stratum, may or may not be sampled)
                for coalition in sampled:
                    coalitions.append(coalition)
                    sizes.append(s)
                    weights.append(importance_weight)
                    actual_samples[s] += 1
            else:
                # Non-middle size: use paired sampling for variance reduction
                # Compute weight for complement coalitions
                complement_size = n_others - s
                leverage_score_complement = self._compute_leverage_score(complement_size, n_others)
                prob_complement = min(1.0, 2 * c * leverage_score_complement)
                if prob_complement > 0:
                    importance_weight_complement = 1.0 / prob_complement
                else:
                    importance_weight_complement = 0.0

                for coalition in sampled:
                    complement = set(other_players) - coalition

                    coalitions.append(coalition)
                    coalitions.append(complement)

                    sizes.append(s)
                    sizes.append(complement_size)

                    # Each gets its own importance weight
                    weights.append(importance_weight)
                    weights.append(importance_weight_complement)

                    # Track samples for both sizes
                    actual_samples[s] += 1
                    actual_samples[complement_size] += 1

        # Accumulate actual samples tracking (across all players)
        if self._actual_samples_by_size is None:
            self._actual_samples_by_size = actual_samples
        else:
            self._actual_samples_by_size = self._actual_samples_by_size + actual_samples

        return coalitions, np.array(sizes), np.array(weights)

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

        # Track actual samples for statistics
        actual_samples = np.zeros(self.n, dtype=int)

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
                    actual_samples[size] += 1
            else:
                # Random sampling without replacement
                sampled = self._sample_coalitions_of_size(other_players, size, n_samples)
                for coalition in sampled:
                    coalitions.append(coalition)
                    sizes.append(size)
                    # Weight = (total coalitions of this size) / (samples of this size)
                    # This ensures unbiased estimation
                    weights.append(n_total / n_samples)
                    actual_samples[size] += 1

        # Accumulate actual samples tracking (across all players)
        if self._actual_samples_by_size is None:
            self._actual_samples_by_size = actual_samples
        else:
            self._actual_samples_by_size = self._actual_samples_by_size + actual_samples

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
        # For leverage_bernoulli, use tracked actual samples
        if self.allocations is None:
            actual_samples = self._actual_samples_by_size
            if actual_samples is None:
                actual_samples = np.zeros(self.n, dtype=int)

            # Compute coverage from actual samples
            coverage = np.zeros(self.n)
            for s in range(self.n):
                if self.n_coalitions_by_size[s] > 0:
                    coverage[s] = actual_samples[s] / self.n_coalitions_by_size[s]

            return SamplingStats(
                n_players=self.n,
                total_budget=self.budget,
                allocations_by_size=actual_samples,  # Use actual samples as "allocation"
                actual_samples_by_size=actual_samples,
                coverage_by_size=coverage
            )

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
# Implicit DCT Operator for Large Spaces
# =============================================================================

class ImplicitDCTOperator:
    """
    Implicit operator for A @ s = B @ IDCT(s) without storing full matrix.

    This enables compressed sensing reconstruction for large coalition spaces
    where storing a full 2^(n-1) × 2^(n-1) DCT matrix is infeasible.

    Parameters
    ----------
    sampled_indices : np.ndarray
        Indices of sampled coalitions in the full space [0, 2^(n-1) - 1]
    full_size : int
        Full signal size: 2^(n-1)
    """

    def __init__(self, sampled_indices: np.ndarray, full_size: int):
        self.sampled_indices = sampled_indices
        self.full_size = full_size
        self.shape = (len(sampled_indices), full_size)

    def matvec(self, s: np.ndarray) -> np.ndarray:
        """
        Compute A @ s = B @ IDCT(s) implicitly.

        Parameters
        ----------
        s : np.ndarray
            DCT coefficients (full_size,)

        Returns
        -------
        y : np.ndarray
            Measurements at sampled indices (len(sampled_indices),)
        """
        # Fast IDCT: O(N log N) instead of O(N^2)
        u_full = idct(s, norm='ortho')
        # Select sampled indices
        return u_full[self.sampled_indices]

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """
        Compute A^T @ y implicitly (for iterative solvers).

        Parameters
        ----------
        y : np.ndarray
            Measurements (len(sampled_indices),)

        Returns
        -------
        g : np.ndarray
            Gradient in DCT coefficient space (full_size,)
        """
        # Scatter y to full space
        u_full = np.zeros(self.full_size)
        u_full[self.sampled_indices] = y
        # Fast DCT: O(N log N)
        return dct(u_full, norm='ortho')

    def __matmul__(self, s: np.ndarray) -> np.ndarray:
        """Matrix multiplication operator."""
        return self.matvec(s)


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

    def solve(self, A, b: np.ndarray) -> np.ndarray:
        """
        Solve the L1 minimization problem.

        Parameters
        ----------
        A : np.ndarray or ImplicitDCTOperator
            Measurement matrix/operator (l x m)
        b : np.ndarray
            Measurements (l,)

        Returns
        -------
        x : np.ndarray
            Reconstructed signal (m,)
        """
        l, m = A.shape

        # Check if A is an implicit operator
        if isinstance(A, ImplicitDCTOperator):
            # Use method that works with implicit operators
            return self._solve_implicit(A, b)
        else:
            # A is a numpy array, use standard methods
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

    def _solve_implicit(self, A: 'ImplicitDCTOperator', b: np.ndarray) -> np.ndarray:
        """
        Solve L1 minimization for implicit DCT operator.

        For implicit operators, we cannot use CVXPY because it requires
        materializing the full matrix. Instead, we use ISTA (Iterative Soft
        Thresholding Algorithm) which only requires matrix-vector products.
        """
        # Use ISTA for implicit operators (can't materialize matrix for cvxpy)
        return self._ista(A, b)

    def _ista(self, A: 'ImplicitDCTOperator', b: np.ndarray, lam: float = None) -> np.ndarray:
        """
        Fast Iterative Soft Thresholding Algorithm (FISTA) for L1 minimization.

        Solves: min λ||x||_1 + (1/2)||Ax - b||_2^2

        Uses Nesterov acceleration for faster convergence.
        """
        m = A.shape[1]

        # Auto-tune lambda if not specified
        if lam is None:
            # Use noise-based estimate: λ ≈ ||A^T @ noise||_∞
            # For normalized measurements, a typical value is 0.01-0.1
            lam = 0.05

        # Better initialization: least squares solution
        try:
            # x0 = argmin ||Ax - b||² via gradient descent (few iterations)
            x = np.zeros(m)
            for _ in range(10):
                residual = A.matvec(x) - b
                grad = A.rmatvec(residual)
                x = x - 0.5 * grad
        except:
            x = np.zeros(m)

        # FISTA with Nesterov acceleration
        # For operator A = B @ IDCT:
        # - IDCT is orthonormal: ||IDCT|| = 1
        # - B is selection operator: ||B|| = 1
        # - Lipschitz constant L ≤ ||A||² ≤ 1
        # - Safe step size: α = 1/L ≈ 1.0
        step_size = 0.9  # Slightly conservative

        y = x.copy()
        t = 1.0

        for iteration in range(self.max_iter):
            # Gradient step on y
            residual = A.matvec(y) - b
            grad = A.rmatvec(residual)
            x_new = y - step_size * grad

            # Soft thresholding (proximal operator)
            x_new = self._soft_threshold(x_new, step_size * lam)

            # Nesterov acceleration
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)

            # Check convergence
            rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
            if rel_change < 1e-5:
                if self.tolerance < 1e-4:  # Only print for verbose mode
                    print(f"  [ISTA] Converged in {iteration+1} iterations")
                break

            x = x_new
            t = t_new

        return x

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


# =============================================================================
# Main Explainer Class
# =============================================================================

class RPQRCSExplainer(Explainer):
    """
    Random Projection QRCS Shapley value explainer.

    This explainer improves upon the original QRCS by:
    1. Using stratified sampling by coalition size (eliminates bias)
    2. Using random projections instead of DCT (works with sampled coalitions)
    3. Maintaining the compressed sensing framework for fair comparison

    The algorithm:
    1. For each player i:
       a. Sample coalitions using configurable allocation strategy:
          - 'shapley_weighted': Deterministic allocation proportional to Shapley weights
          - 'uniform': Equal allocation across all coalition sizes
          - 'leverage': Inverse coalition count (deterministic, heuristic)
          - 'leverage_bernoulli': Bernoulli sampling with leverage scores (random, theory-based)
       b. Compute marginal contributions for sampled coalitions
       c. Compute Shapley value via:
          - Direct weighted estimation (use_direct_estimation=True, recommended)
          - OR compressed sensing with L1 minimization (use_direct_estimation=False)

    Note:
        This explainer is decoupled from graph structure. It accepts various input
        types to determine the number of players and their identifiers. The actual
        use of any structure (graph, data, etc.) is delegated to the characteristic
        function via the context parameter.

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
        How to allocate samples across coalition sizes:
        - 'shapley_weighted': Proportional to Shapley weight × coalition count (recommended)
        - 'uniform': Equal allocation per stratum
        - 'leverage': Inverse coalition count (heuristic, may have poor coverage)
        - 'leverage_bernoulli': Bernoulli sampling with leverage scores (experimental, theory-based)
    tolerance : float, default=1e-4
        L1 optimization tolerance
    use_direct_estimation : bool, default=True
        If True, use direct weighted estimation (faster, recommended).
        If False, use compressed sensing with L1 minimization.
        Note: Both methods use importance weights for unbiased sampling.
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
        use_direct_estimation: bool = True,
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
        self.use_direct_estimation = use_direct_estimation
        self.seed = seed

        # Will be set during fit
        self.n = None  # Number of players
        self.player_ids = None  # Player identifiers
        self._context = None  # Context passed to characteristic function
        self._sampler = None
        self._rng = np.random.default_rng(seed)

    def _parse_input(
        self,
        X: Union[int, np.ndarray, pd.DataFrame, nx.Graph, List],
        context: Optional[Any] = None
    ) -> Tuple[int, List, Any]:
        """Parse input to extract player count, identifiers, and context.

        Args:
            X: Input that defines players. Can be:
               - int: number of players (ids will be 0..n-1)
               - nx.Graph: n = number_of_nodes(), ids = nodes()
               - pd.DataFrame: n = number of columns, ids = column names
               - np.ndarray: n = number of columns (2D) or length (1D)
               - List: n = length, ids = elements or indices
            context: Optional context for characteristic function.
                     If None and X is Graph/DataFrame/array, X is used as context.

        Returns:
            Tuple of (n_players, player_ids, context)
        """
        if isinstance(X, int):
            n = X
            player_ids = list(range(n))
            ctx = context
        elif isinstance(X, nx.Graph):
            n = X.number_of_nodes()
            player_ids = list(X.nodes())
            ctx = context if context is not None else X
        elif isinstance(X, pd.DataFrame):
            n = X.shape[1]
            player_ids = list(X.columns)
            ctx = context if context is not None else X
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                n = len(X)
            else:
                n = X.shape[1]
            player_ids = list(range(n))
            ctx = context if context is not None else X
        elif isinstance(X, list):
            n = len(X)
            try:
                if len(set(X)) == len(X):
                    player_ids = list(X)
                else:
                    player_ids = list(range(n))
            except TypeError:
                player_ids = list(range(n))
            ctx = context
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        return n, player_ids, ctx

    def fit(
        self,
        X: Union[int, np.ndarray, pd.DataFrame, nx.Graph, List],
        context: Optional[Any] = None,
        **kwargs
    ) -> 'RPQRCSExplainer':
        """
        Fit the explainer to data.

        Parameters
        ----------
        X : int, array-like, DataFrame, Graph, or List
            Input that defines players. Can be:
            - int: number of players directly
            - nx.Graph: uses nodes as players
            - pd.DataFrame: uses columns as players
            - np.ndarray: uses columns (2D) or elements (1D) as players
            - List: uses elements as player identifiers
        context : Any, optional
            Context passed to characteristic function.
            If None and X is Graph/DataFrame/array, X is used as context.

        Returns
        -------
        self
        """
        self.n, self.player_ids, self._context = self._parse_input(X, context)

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
            if self._sampler.allocations is not None:
                for s in range(self.n):
                    if self._sampler.allocations[s] > 0:
                        print(f"    size {s}: {self._sampler.allocations[s]} samples")
            else:
                # For leverage_bernoulli, allocations are determined dynamically
                print(f"    Dynamic allocation (leverage_bernoulli strategy)")

        self._fitted = True
        return self

    def explain(
        self,
        X: Optional[Union[int, np.ndarray, pd.DataFrame, nx.Graph, List]] = None,
        context: Optional[Any] = None,
        **kwargs
    ) -> Dict:
        """
        Compute Shapley values using RP-QRCS.

        Parameters
        ----------
        X : int, array-like, DataFrame, Graph, List, optional
            Input data (uses fitted data if None)
        context : Any, optional
            Context for characteristic function

        Returns
        -------
        shapley_values : Dict
            Shapley values for each player (keyed by player_id)
        """
        if X is not None:
            self.fit(X, context=context, **kwargs)
        elif not self._fitted:
            raise ValueError("Explainer not fitted. Call fit() first or provide X.")

        # Create utility wrapper
        def utility_func(S: Set[int]) -> float:
            # Map internal indices to player identifiers
            player_set = {self.player_ids[i] for i in S}
            return self.characteristic_function(player_set, self._context)

        # Compute Shapley values
        shapley_indices = self._compute_shapley(utility_func)

        # Map back to player identifiers
        shapley_values = {self.player_ids[i]: value for i, value in shapley_indices.items()}

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
            if self.allocation_strategy == 'leverage_bernoulli':
                coalitions, sizes, weights = self._sampler.sample_all_strata_bernoulli(player)
            else:
                coalitions, sizes, weights = self._sampler.sample_all_strata(player)
            m = len(coalitions)

            if m == 0:
                shapley_values[player] = 0.0
                continue

            # Step 2 & 3: Compute Shapley value
            if self.use_direct_estimation:
                # Direct estimation: compute marginals for sampled coalitions only
                marginals = np.zeros(m)
                for j, S in enumerate(coalitions):
                    v_with = utility_func(S | {player})
                    v_without = utility_func(S) if S else 0.0
                    marginals[j] = v_with - v_without

                # Direct estimation using stratified sampling
                # Within each size stratum, all coalitions have the same importance weight
                # Therefore, the weighted mean simplifies to the simple mean
                shapley_value = 0.0
                for size in range(self.n):
                    mask = (sizes == size)
                    if not np.any(mask):
                        continue

                    size_marginals = marginals[mask]

                    # Shapley weight for this size
                    shapley_weight = self._sampler.shapley_weights[size]

                    # Number of coalitions of this size
                    n_coalitions = self._sampler.n_coalitions_by_size[size]

                    # Simple mean (weights cancel out within each size stratum)
                    size_mean = np.mean(size_marginals)

                    # Shapley value contribution from this size stratum
                    shapley_value += shapley_weight * n_coalitions * size_mean

                shapley_values[player] = shapley_value

            else:
                # CS reconstruction: reconstruct full 2^(n-1) marginals from samples
                shapley_value = self._compute_with_cs_implicit(
                    coalitions, player, utility_func, l1_solver
                )
                shapley_values[player] = shapley_value

        return shapley_values

    def _coalition_to_index(self, coalition: Set[int], player: int) -> int:
        """
        Convert coalition to index in full signal space [0, 2^(n-1) - 1].

        Parameters
        ----------
        coalition : Set[int]
            Coalition excluding target player
        player : int
            Target player index

        Returns
        -------
        index : int
            Index in [0, 2^(n-1) - 1]
        """
        other_players = [i for i in range(self.n) if i != player]
        index = 0
        for i, p in enumerate(other_players):
            if p in coalition:
                index |= (1 << i)
        return index

    def _compute_with_cs_implicit(
        self,
        coalitions: List[Set[int]],
        player: int,
        utility_func: Callable[[Set[int]], float],
        l1_solver: L1Solver
    ) -> float:
        """
        Compute Shapley value using compressed sensing with implicit DCT.

        This properly implements CS by:
        1. Measuring only m sampled coalitions
        2. Using implicit DCT operator (no full matrix storage)
        3. Reconstructing full 2^(n-1) marginal vector
        4. Computing Shapley value from full marginals

        Parameters
        ----------
        coalitions : List[Set[int]]
            Sampled coalitions (m coalitions)
        player : int
            Target player index
        utility_func : callable
            Function to compute coalition value
        l1_solver : L1Solver
            L1 minimization solver

        Returns
        -------
        float
            Shapley value for the player
        """
        m = len(coalitions)
        full_size = 2 ** (self.n - 1)

        if self.verbose:
            print(f"  [CS] Full space: 2^{self.n-1} = {full_size}")
            print(f"  [CS] Sampled: {m} coalitions ({100*m/full_size:.2f}%)")

        # Step 1: Map coalitions to indices in [0, 2^(n-1) - 1]
        sampled_indices = np.array([
            self._coalition_to_index(S, player) for S in coalitions
        ])

        # Step 2: Measure marginal contributions for sampled coalitions
        y = np.zeros(m)
        for j, S in enumerate(coalitions):
            v_with = utility_func(S | {player})
            v_without = utility_func(S) if S else 0.0
            y[j] = v_with - v_without

        if self.verbose:
            zero_threshold = 1e-10
            sparsity_y = np.sum(np.abs(y) < zero_threshold) / len(y)
            print(f"  [CS] Measurement sparsity: {sparsity_y*100:.1f}%")

        # Step 3: Create implicit DCT operator
        # A @ s = B @ IDCT(s) where B selects sampled indices
        A = ImplicitDCTOperator(sampled_indices, full_size)

        # Step 4: Solve min ||s||_1 s.t. ||A @ s - y|| <= epsilon
        # s are DCT coefficients (hopefully sparse!)
        try:
            s_hat = l1_solver.solve(A, y)

            # Check sparsity of recovered coefficients
            if self.verbose:
                sparsity_s = np.sum(np.abs(s_hat) < zero_threshold) / len(s_hat)
                print(f"  [CS] DCT coefficient sparsity: {sparsity_s*100:.1f}%")

            # Step 5: Transform back to marginal domain
            u_hat = idct(s_hat, norm='ortho')  # Full 2^(n-1) marginals

            # Step 6: Compute Shapley value as weighted sum over full space
            shapley_value = 0.0
            for idx in range(full_size):
                s = bin(idx).count('1')  # Coalition size
                shapley_weight = self._sampler.shapley_weights[s]
                shapley_value += shapley_weight * u_hat[idx]

            return shapley_value

        except Exception as e:
            warnings.warn(
                f"CS reconstruction failed: {e}. Falling back to mean of samples.",
                RuntimeWarning
            )
            return np.mean(y) if len(y) > 0 else 0.0

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

        Note: weights parameter is kept for API compatibility but not used
        since weights are constant within each size stratum.
        """
        m = len(marginals)

        # Determine number of measurements
        if self.n_measurements is not None:
            l = min(self.n_measurements, m)
        else:
            l = max(1, int(m * self.measurement_ratio))

        if l >= m:
            # No compression needed, use stratified mean
            shapley_value = 0.0
            for size in range(self.n):
                mask = (sizes == size)
                if not np.any(mask):
                    continue

                size_marginals = marginals[mask]
                shapley_weight = self._sampler.shapley_weights[size]
                n_coalitions = self._sampler.n_coalitions_by_size[size]

                size_mean = np.mean(size_marginals)
                shapley_value += shapley_weight * n_coalitions * size_mean

            return shapley_value

        # Build random projection matrix
        projection = RandomProjectionMatrix(
            n_measurements=l,
            n_signals=m,
            projection_type=self.projection_type,
            seed=self._rng.integers(0, 2**31)
        )

        # Project marginal contributions
        y = projection @ marginals

        # Check sparsity before reconstruction
        if self.verbose:
            zero_threshold = 1e-3
            n_zeros = np.sum(np.abs(marginals) < zero_threshold)
            sparsity = n_zeros / len(marginals)
            print(f"  [CS] Signal sparsity: {sparsity*100:.1f}% ({n_zeros}/{len(marginals)} zeros)")
            print(f"  [CS] WARNING: Marginals are NOT sparse - CS may not help!")

        # Reconstruct via L1 minimization
        try:
            u_hat = l1_solver.solve(projection.matrix, y)

            # Compute Shapley value using stratified mean
            # Weights are constant within each size stratum so simple mean suffices
            shapley_value = 0.0
            for size in range(self.n):
                mask = (sizes == size)
                if not np.any(mask):
                    continue

                size_contributions = u_hat[mask]
                shapley_weight = self._sampler.shapley_weights[size]
                n_coalitions = self._sampler.n_coalitions_by_size[size]

                size_mean = np.mean(size_contributions)
                shapley_value += shapley_weight * n_coalitions * size_mean

            return shapley_value

        except Exception as e:
            warnings.warn(
                f"CS reconstruction failed: {e}. Falling back to stratified mean.",
                RuntimeWarning
            )
            # Fallback: use stratified mean on original marginals
            shapley_value = 0.0
            for size in range(self.n):
                mask = (sizes == size)
                if not np.any(mask):
                    continue

                size_marginals = marginals[mask]
                shapley_weight = self._sampler.shapley_weights[size]
                n_coalitions = self._sampler.n_coalitions_by_size[size]

                size_mean = np.mean(size_marginals)
                shapley_value += shapley_weight * n_coalitions * size_mean

            return shapley_value

    def get_sampling_stats(self) -> Optional[SamplingStats]:
        """Get statistics about the sampling configuration."""
        if self._sampler is not None:
            return self._sampler.get_stats()
        return None
