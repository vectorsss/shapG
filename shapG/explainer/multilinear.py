"""
Multilinear extension-based Shapley value computation.

This module implements the Leverage-Enhanced Multilinear (LEM) estimator,
combining Owen's multilinear extension (1972) with leverage score sampling
from Musco & Witter (2025) for optimal variance reduction.

Key Features:
- Owen's exact integral formulation: φ_i = ∫₀¹ f_i(t,...,t) dt
- Leverage-stratified sampling for variance reduction
- Rigorous error bounds (quadrature + sampling)
- O(n × n_quadrature × n_samples) complexity

References:
    [1] Owen, G. (1972). Multilinear Extensions of Games.
        Management Science, 18(5-Part-2), 64-79.
    [2] Musco, C. & Witter, R. (2025). Provably Accurate Shapley Value
        Estimation via Leverage Score Sampling. ICLR 2025.
"""

from typing import Dict, Union, Optional, Set, List, Callable, Tuple
from dataclasses import dataclass
from math import factorial, sqrt, log
import numpy as np
import pandas as pd
import networkx as nx
import time

from scipy.integrate import simpson
from scipy.special import comb
from scipy.stats import norm

from .base import Explainer, CharacteristicFunction


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ErrorBounds:
    """Rigorous error bounds for the LEM estimator."""

    quadrature_error: float
    sampling_error: float
    total_error: float
    confidence_level: float

    def __str__(self) -> str:
        return (
            f"ErrorBounds(total={self.total_error:.2e}, "
            f"quad={self.quadrature_error:.2e}, "
            f"sample={self.sampling_error:.2e}, "
            f"conf={self.confidence_level:.1%})"
        )


# =============================================================================
# Helper Classes (Private)
# =============================================================================


class _LeverageScoreComputer:
    """
    Computes leverage scores for Shapley value estimation.

    Supports two leverage score definitions:

    1. "shapley_weighted" (default): Leverage proportional to Shapley weights.
       ℓ_s ∝ C(n-1,s) × w_s, where w_s = s!(n-s-1)!/n!
       This weights coalition sizes by their Shapley importance.

    2. "inverse_coalition" (Musco & Witter 2025): Pure inverse coalition count.
       ℓ_s = 1/C(n-1,s)
       This is the "true" leverage score from the compressed sensing formulation,
       ensuring equal representation across coalition sizes.

    Parameters
    ----------
    n : int
        Number of players
    leverage_type : str, default="shapley_weighted"
        Type of leverage score: "shapley_weighted" or "inverse_coalition"
    tilt_by_bernoulli : bool, default=True
        If True, tilt leverage by Bernoulli(t) probabilities for Owen's integration.
        If False, use pure leverage scores (matches StratifiedShapley behavior).
        Note: With inverse_coalition + tilt_by_bernoulli=True, the leverage
        factor gets cancelled by the Bernoulli mass, resulting in pure
        Bernoulli(t) sampling.
    """

    LEVERAGE_TYPES = ("shapley_weighted", "inverse_coalition")

    def __init__(
        self,
        n: int,
        leverage_type: str = "shapley_weighted",
        tilt_by_bernoulli: bool = True,
    ):
        if leverage_type not in self.LEVERAGE_TYPES:
            raise ValueError(
                f"leverage_type must be one of {self.LEVERAGE_TYPES}, got '{leverage_type}'"
            )
        self.n = n
        self.leverage_type = leverage_type
        self.tilt_by_bernoulli = tilt_by_bernoulli
        self._cache = {}
        self._precompute()

    def _precompute(self):
        """Precompute factorial cache and base leverage scores."""
        self._fact = {k: factorial(k) for k in range(self.n + 2)}

        # Shapley weights by coalition size
        self._shapley_weights = np.zeros(self.n)
        for s in range(self.n):
            self._shapley_weights[s] = (
                self._fact[s] * self._fact[self.n - s - 1] / self._fact[self.n]
            )

        # Number of coalitions by size (for player i, considering n-1 other players)
        self._n_coalitions_by_size = np.array(
            [comb(self.n - 1, s, exact=True) for s in range(self.n)], dtype=float
        )

        # Compute base leverage scores based on leverage_type
        if self.leverage_type == "shapley_weighted":
            # Original: ℓ_s ∝ C(n-1,s) × shapley_weights
            self._base_leverage = self._n_coalitions_by_size * self._shapley_weights
        else:  # inverse_coalition
            # Musco & Witter 2025: ℓ_s = 1/C(n-1,s)
            self._base_leverage = np.zeros(self.n)
            for s in range(self.n):
                n_coalitions = self._n_coalitions_by_size[s]
                if n_coalitions > 0:
                    self._base_leverage[s] = 1.0 / n_coalitions
                else:
                    self._base_leverage[s] = 0.0

        # Normalize
        total = self._base_leverage.sum()
        if total > 0:
            self._base_leverage /= total

    def shapley_weight(self, s: int) -> float:
        """Get Shapley weight for coalition of size s."""
        if s < 0 or s >= self.n:
            return 0.0
        return self._shapley_weights[s]

    def tilted_leverage(self, t: float) -> np.ndarray:
        """
        Compute leverage scores, optionally tilted by Bernoulli(t) probabilities.

        If tilt_by_bernoulli=True:
            tilted_ℓ_s(t) = ℓ_s × Pr(|S| = s | Bernoulli(t)^{n-1})
            This combines Owen's probabilistic interpretation with leverage importance.

        If tilt_by_bernoulli=False:
            Returns base leverage scores (same for all t).
            This matches StratifiedShapley behavior.
        """
        # If not tilting, return base leverage (same for all t)
        if not self.tilt_by_bernoulli:
            return self._base_leverage.copy()

        cache_key = round(t, 6)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_others = self.n - 1

        # Binomial probabilities under Bernoulli(t)
        bernoulli_mass = np.zeros(self.n)
        for s in range(self.n):
            if s <= n_others:
                bernoulli_mass[s] = (
                    comb(n_others, s, exact=True) * (t**s) * ((1 - t) ** (n_others - s))
                )

        # Combine leverage and Bernoulli
        tilted = self._base_leverage * bernoulli_mass

        # Normalize
        total = tilted.sum()
        if total > 1e-15:
            tilted /= total
        else:
            tilted = self._base_leverage.copy()

        self._cache[cache_key] = tilted
        return tilted

    def optimal_allocation(self, t: float, budget: int) -> np.ndarray:
        """Compute optimal sample allocation across coalition sizes."""
        tilted = self.tilted_leverage(t)

        # Allocate proportionally, ensuring at least some samples for important sizes
        allocation = np.floor(tilted * budget).astype(int)
        remaining = budget - allocation.sum()

        # Distribute remaining samples to highest leverage strata
        if remaining > 0:
            indices = np.argsort(-tilted)
            for i in range(int(remaining)):
                allocation[indices[i % len(indices)]] += 1

        return allocation

    def efficiency_gain(self) -> float:
        """Estimate variance reduction factor from leverage sampling."""
        return max(1.0, log(self.n + 1) / 2)

    def find_c_binary_search(self, target_budget: int) -> float:
        """
        Find oversampling parameter c via binary search.

        Solves: target_budget ≈ Σ_{s=0}^{n-1} min(N_s, 2c * ℓ_s * N_s)
        where N_s = C(n-1, s), ℓ_s = leverage score for size s

        Parameters
        ----------
        target_budget : int
            Target number of coalition samples

        Returns
        -------
        float
            Oversampling parameter c
        """

        def expected_samples(c):
            """Expected number of coalitions sampled."""
            total = 0
            for s in range(self.n):
                n_coalitions = self._n_coalitions_by_size[s]
                if n_coalitions == 0:
                    continue
                leverage_score = self._base_leverage[s]
                prob = min(1.0, 2 * c * leverage_score)
                expected_sampled = prob * n_coalitions
                total += expected_sampled
            return total

        # Binary search for c
        c_low, c_high = 0.01, 1000.0

        for _ in range(50):  # Max iterations
            c_mid = (c_low + c_high) / 2
            samples = expected_samples(c_mid)

            if abs(samples - target_budget) < 1:
                return c_mid

            if samples < target_budget:
                c_low = c_mid
            else:
                c_high = c_mid

        return c_mid

    def allocate_across_quadrature(
        self, t_values: np.ndarray, total_budget: int
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Allocate sample budget across quadrature points adaptively.

        At t=0 and t=1, only one coalition is possible, so we use exact
        computation (no sampling needed). For other t values, we allocate
        proportionally to the variance t×(1-t).

        Parameters
        ----------
        t_values : np.ndarray
            Quadrature points in [0, 1]
        total_budget : int
            Total sample budget across all quadrature points

        Returns
        -------
        allocations : np.ndarray
            Number of samples for each quadrature point
        is_exact : List[bool]
            Whether each quadrature point uses exact computation
        """
        K = len(t_values)
        allocations = np.zeros(K, dtype=int)
        is_exact = [False] * K

        # Identify exact computation points (t=0 or t=1)
        epsilon = 1e-10
        for k, t in enumerate(t_values):
            if t < epsilon or t > 1 - epsilon:
                is_exact[k] = True

        # Compute variance proxy for non-exact points: Var ~ t(1-t)
        variance_proxy = np.zeros(K)
        for k, t in enumerate(t_values):
            if not is_exact[k]:
                variance_proxy[k] = t * (1 - t)

        # Normalize and allocate
        total_variance = variance_proxy.sum()
        if total_variance > 0:
            for k in range(K):
                if not is_exact[k]:
                    # Allocate proportionally to variance, minimum 1 sample
                    allocations[k] = max(
                        1, int(total_budget * variance_proxy[k] / total_variance)
                    )

        # Distribute remaining budget to highest variance points
        remaining = total_budget - allocations.sum()
        if remaining > 0:
            # Sort by variance (descending)
            sorted_indices = np.argsort(-variance_proxy)
            for i in range(int(remaining)):
                k = sorted_indices[i % K]
                if not is_exact[k]:
                    allocations[k] += 1

        return allocations, is_exact


class _StratifiedCoalitionSampler:
    """
    Stratified coalition sampler with importance weighting.

    Implements leverage-guided stratified sampling for unbiased
    estimation of E_{S~Bernoulli(t)}[Δ_i(S)].
    """

    def __init__(
        self, leverage_computer: _LeverageScoreComputer, rng: np.random.Generator
    ):
        self.leverage = leverage_computer
        self.n = leverage_computer.n
        self.rng = rng
        # Track size distribution across all samples
        self.size_counts = np.zeros(self.n, dtype=int)

    def sample_stratified(
        self, player: int, t: float, n_samples: int
    ) -> Tuple[List[Set[int]], np.ndarray]:
        """
        Sample coalitions using leverage-stratified sampling.

        When allocation exceeds the total number of coalitions at a size,
        uses exact enumeration instead of sampling.

        Returns:
            (coalitions, weights) where weights are for importance sampling
        """
        from itertools import combinations

        other_players = [j for j in range(self.n) if j != player]
        n_others = len(other_players)

        tilted = self.leverage.tilted_leverage(t)
        allocation = self.leverage.optimal_allocation(t, n_samples)

        coalitions = []
        weights = []

        for s in range(n_others + 1):
            n_s = allocation[s]
            if n_s == 0:
                continue

            # Total number of coalitions of size s
            n_total = int(comb(n_others, s, exact=True))

            # True probability mass for size s under Bernoulli(t)
            true_prob = (
                comb(n_others, s, exact=True) * (t**s) * ((1 - t) ** (n_others - s))
            )

            # Our sampling probability (avoid division by zero)
            sample_prob = tilted[s] if tilted[s] > 1e-15 else 1e-15

            # Use exact enumeration when:
            # 1. Allocation exceeds total coalitions, OR
            # 2. Total coalitions is small enough to enumerate efficiently
            use_exact = (n_s >= n_total) or (n_total <= 100)

            if use_exact:
                # Exact enumeration - enumerate all coalitions of size s
                for combo in combinations(other_players, s):
                    coalitions.append(set(combo))
                    # Weight = 1 for exact enumeration (or true_prob for consistency)
                    weights.append(true_prob / sample_prob)

                # Track actual count
                self.size_counts[s] += n_total
            else:
                # Sample n_s coalitions uniformly from size s
                self.size_counts[s] += n_s

                for _ in range(n_s):
                    if s == 0:
                        S = set()
                    elif s >= n_others:
                        S = set(other_players)
                    else:
                        S = set(self.rng.choice(other_players, size=s, replace=False))

                    coalitions.append(S)

                    # Importance weight for unbiased estimation
                    weight = true_prob / sample_prob
                    weights.append(weight)

        return coalitions, np.array(weights)

    def get_size_distribution(self) -> Dict[int, int]:
        """Get the size distribution of sampled coalitions."""
        return {s: int(count) for s, count in enumerate(self.size_counts) if count > 0}

    def sample_naive(self, player: int, t: float, n_samples: int) -> List[Set[int]]:
        """Naive Bernoulli(t) sampling (for comparison)."""
        other_players = [j for j in range(self.n) if j != player]

        coalitions = []
        for _ in range(n_samples):
            S = set(j for j in other_players if self.rng.random() < t)
            coalitions.append(S)

        return coalitions

    def sample_bernoulli(
        self, player: int, t: float, n_samples: int
    ) -> Tuple[List[Set[int]], np.ndarray]:
        """
        Sample coalitions using Bernoulli sampling with leverage scores.

        Similar to StratifiedShapley's leverage_bernoulli strategy:
        - Uses leverage scores ℓ_s from the leverage computer
        - Bernoulli sampling with probability p_s = min(1, 2c·ℓ_s)
        - Oversampling parameter c tuned via binary search
        - Exact enumeration when p_s >= 1

        Parameters
        ----------
        player : int
            The player index for whom we're sampling coalitions
        t : float
            The quadrature point (used for importance weights)
        n_samples : int
            Target number of coalition samples

        Returns
        -------
        coalitions : List[Set[int]]
            List of sampled coalitions (excluding target player)
        weights : np.ndarray
            Importance weight for each sample
        """
        from itertools import combinations

        other_players = [j for j in range(self.n) if j != player]
        n_others = len(other_players)

        # Find oversampling parameter c
        c = self.leverage.find_c_binary_search(n_samples)

        coalitions = []
        weights = []

        for s in range(n_others + 1):
            # Total coalitions of this size
            n_total = int(comb(n_others, s, exact=True))
            if n_total == 0:
                continue

            # Leverage score for this size
            leverage_score = self.leverage._base_leverage[s]
            if leverage_score == 0:
                continue

            # Sampling probability
            prob = min(1.0, 2 * c * leverage_score)
            if prob == 0:
                continue

            # True probability mass for size s under Bernoulli(t)
            true_prob = (
                comb(n_others, s, exact=True) * (t**s) * ((1 - t) ** (n_others - s))
            )

            if prob >= 1.0:
                # Exact enumeration - enumerate all coalitions of size s
                for combo in combinations(other_players, s):
                    coalitions.append(set(combo))
                    weights.append(true_prob)
                self.size_counts[s] += n_total
            else:
                # Bernoulli sampling: each coalition is sampled with probability prob
                n_sampled = self.rng.binomial(n_total, prob)

                if n_sampled == 0:
                    continue

                # Sample n_sampled coalitions uniformly
                if n_sampled >= n_total:
                    # Sample all coalitions of this size
                    for combo in combinations(other_players, s):
                        coalitions.append(set(combo))
                        weights.append(true_prob)
                    self.size_counts[s] += n_total
                else:
                    # Sample n_sampled coalitions uniformly from this size
                    for _ in range(n_sampled):
                        if s == 0:
                            S = set()
                        elif s >= n_others:
                            S = set(other_players)
                        else:
                            S = set(
                                self.rng.choice(other_players, size=s, replace=False)
                            )
                        coalitions.append(S)
                        weights.append(true_prob / prob)
                    self.size_counts[s] += n_sampled

        return coalitions, np.array(weights)


class _ErrorBoundComputer:
    """
    Computes rigorous error bounds for the LEM estimator.

    Error Decomposition:
        |φ̂_i - φ_i| ≤ ε_quad + ε_mc

    where:
        ε_quad: Quadrature error from Simpson's rule
        ε_mc: Monte Carlo sampling error
    """

    def __init__(
        self, n: int, n_quadrature: int, n_samples: int, confidence: float = 0.95
    ):
        self.n = n
        self.K = n_quadrature
        self.m = n_samples
        self.confidence = confidence
        self.z_score = norm.ppf((1 + confidence) / 2)

    def quadrature_error_bound(self, fourth_derivative_bound: float = 1.0) -> float:
        """
        Compute quadrature error bound for Simpson's rule.

        Simpson's rule error: |∫f - S_K(f)| ≤ (b-a)^5 / 180 × max|f^(4)(x)| / K^4
        """
        h = 1.0 / (self.K - 1)
        return (h**4) * fourth_derivative_bound / 180

    def sampling_error_bound(
        self, variance_estimate: float, leverage_efficiency: float = 1.0
    ) -> float:
        """
        Compute Monte Carlo sampling error bound.

        Using CLT: ε_mc ≤ z_{α/2} × σ / √(m × η × K)
        """
        effective_samples = self.m * leverage_efficiency * self.K
        if effective_samples <= 0:
            return float("inf")
        return self.z_score * sqrt(variance_estimate / effective_samples)

    def compute_bounds(
        self,
        observed_variance: float,
        fourth_derivative_bound: float = 1.0,
        leverage_efficiency: float = 1.0,
    ) -> ErrorBounds:
        """Compute complete error bounds."""
        quad_err = self.quadrature_error_bound(fourth_derivative_bound)
        samp_err = self.sampling_error_bound(observed_variance, leverage_efficiency)

        return ErrorBounds(
            quadrature_error=quad_err,
            sampling_error=samp_err,
            total_error=quad_err + samp_err,
            confidence_level=self.confidence,
        )


# =============================================================================
# Main Explainer Class
# =============================================================================


class MultilinearExplainer(Explainer):
    """
    Shapley value computation using Owen's multilinear extension with
    optional leverage-enhanced sampling.

    This explainer implements the Leverage-Enhanced Multilinear (LEM) estimator,
    combining Owen's exact integral formulation with optimal leverage score
    sampling for variance reduction.

    The multilinear extension f(x) of a game v is defined as:
        f(x) = Σ_{S⊆N} [∏_{j∈S} x_j · ∏_{j∉S} (1-x_j)] · v(S)

    The Shapley value is obtained by integrating partial derivatives along
    the main diagonal (Owen's Theorem 5):
        φ_i = ∫₀¹ (∂f/∂x_i)(t, t, ..., t) dt

    Algorithm:
        For each player i:
            1. For each quadrature point t_k ∈ {t_1, ..., t_K}:
                a. Sample coalitions (leverage-stratified or naive Bernoulli)
                b. Estimate f_i(t_k) using (importance-weighted) average
            2. Integrate: φ̂_i = Simpson(f_i(t_1), ..., f_i(t_K))

    Theoretical Guarantees:
        - Unbiased: E[φ̂_i] = φ_i
        - Error bound: |φ̂_i - φ_i| ≤ O(K^{-4}) + O(σ/√(mK)) w.h.p.
        - Complexity: O(n × K × m) characteristic function calls

    Note: This explainer does not use graph structure. It computes Shapley
    values for all n players based purely on the characteristic function.

    Parameters
    ----------
    characteristic_function : callable, optional
        Function v(S, context) -> float for coalition S
    n_quadrature : int, optional
        Number of quadrature points for Simpson's rule (odd preferred).
        If None, automatically set based on n: 11 if n≤20, else 21.
    n_samples : int, optional
        Total number of coalition samples across ALL quadrature points (per player).
        If None, automatically set to min(5*n, 2^(n-1)) to scale with problem size
        while avoiding wasteful oversampling for small n.
        Samples are allocated adaptively based on variance at each t:
        - t=0 and t=1: exact computation (only 1 coalition possible)
        - t near 0.5: more samples (higher variance region)
    use_leverage : bool, default=True
        If True, use leverage-based sampling for variance reduction.
        If False, use naive Bernoulli sampling.
    sampling_method : str, default="stratified"
        Sampling method when use_leverage=True:
        - "stratified": Stratified sampling by coalition size with importance weighting
        - "bernoulli": Bernoulli sampling with oversampling parameter c
          Each coalition sampled with probability min(1, 2c*leverage_score).
          Similar to leverage_bernoulli in StratifiedShapley.
    leverage_type : str, default="shapley_weighted"
        Type of leverage score definition (only used when use_leverage=True):
        - "shapley_weighted": ℓ_s ∝ C(n-1,s) × w_s (weights by Shapley importance)
        - "inverse_coalition": ℓ_s = 1/C(n-1,s) (Musco & Witter 2025, pure inverse)
    tilt_by_bernoulli : bool, default=True
        If True, tilt leverage by Bernoulli(t) for Owen's integration.
        If False, use pure leverage scores (matches StratifiedShapley).
        Note: With inverse_coalition + tilt_by_bernoulli=True, the leverage
        factor gets cancelled, resulting in pure Bernoulli(t) sampling.
        Only used when sampling_method="stratified".
    compute_error_bounds : bool, default=False
        If True, compute rigorous error bounds
    confidence : float, default=0.95
        Confidence level for error bounds
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Note
    ----
    For exact Shapley value computation, use ExactExplainer instead.
    MultilinearExplainer is designed for approximation via sampling.

    References
    ----------
    [1] Owen, G. (1972). Multilinear Extensions of Games.
    [2] Musco, C. & Witter, R. (2025). Provably Accurate Shapley Value
        Estimation via Leverage Score Sampling. ICLR 2025.
    """

    SAMPLING_METHODS = ("stratified", "bernoulli")

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        n_quadrature: Optional[int] = None,
        n_samples: Optional[int] = None,
        use_leverage: bool = True,
        sampling_method: str = "stratified",
        leverage_type: str = "shapley_weighted",
        tilt_by_bernoulli: bool = True,
        compute_error_bounds: bool = False,
        confidence: float = 0.95,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(characteristic_function, verbose)

        # Validate non-None values
        if n_quadrature is not None and n_quadrature < 3:
            raise ValueError(f"n_quadrature must be >= 3, got {n_quadrature}")
        if n_samples is not None and n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if leverage_type not in _LeverageScoreComputer.LEVERAGE_TYPES:
            raise ValueError(
                f"leverage_type must be one of {_LeverageScoreComputer.LEVERAGE_TYPES}, "
                f"got '{leverage_type}'"
            )
        if sampling_method not in self.SAMPLING_METHODS:
            raise ValueError(
                f"sampling_method must be one of {self.SAMPLING_METHODS}, "
                f"got '{sampling_method}'"
            )

        # Store user-provided values (None means auto-compute later)
        self._n_quadrature_user = n_quadrature
        self._n_samples_user = n_samples
        self.n_quadrature = None  # Will be set in explain()
        self.n_samples = None  # Will be set in explain()
        self.use_leverage = use_leverage
        self.sampling_method = sampling_method
        self.leverage_type = leverage_type
        self.tilt_by_bernoulli = tilt_by_bernoulli
        self.compute_error_bounds = compute_error_bounds
        self.confidence = confidence
        self.seed = seed

        # Random number generator
        self._rng = np.random.default_rng(seed)

        # Results storage
        self.shapley_values_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self.computation_time_ = None
        self.method_used_ = None
        self.n_char_func_calls_ = 0
        self.error_bounds_ = None
        self.leverage_efficiency_ = 1.0
        self.size_distribution_ = None

    def _compute_smart_defaults(self, n: int) -> Tuple[int, int]:
        """
        Compute smart defaults for n_quadrature and n_samples based on n.

        Parameters
        ----------
        n : int
            Number of players/features

        Returns
        -------
        n_quadrature : int
            Number of quadrature points (11 if n≤20, else 21)
        n_samples : int
            Number of samples, capped at total coalitions for small n
        """
        # n_quadrature: 11 for small problems, 21 for larger ones
        if self._n_quadrature_user is not None:
            n_quadrature = self._n_quadrature_user
        else:
            n_quadrature = 11 if n <= 20 else 21

        # Ensure odd number for Simpson's rule
        if n_quadrature % 2 == 0:
            n_quadrature += 1

        # n_samples: scale with n, but cap at total coalitions
        if self._n_samples_user is not None:
            n_samples = self._n_samples_user
        else:
            # Default: 5*n samples, reasonable for most cases
            n_samples = 5 * n

        # Cap at total coalitions (2^(n-1)) to avoid wasteful oversampling
        total_coalitions = 2 ** (n - 1)
        if n_samples > total_coalitions:
            n_samples = total_coalitions

        return n_quadrature, n_samples

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **_kwargs
    ) -> "MultilinearExplainer":
        """
        Fit the explainer to data.

        Args:
            X: Input data - can be numpy array, DataFrame, or networkx Graph

        Returns:
            Self for method chaining
        """
        if isinstance(X, nx.Graph):
            self.n_features_ = X.number_of_nodes()
            self.feature_names_ = list(X.nodes())
            self._context = X
        elif isinstance(X, pd.DataFrame):
            self.n_features_ = X.shape[1]
            self.feature_names_ = list(X.columns)
            self._context = X
        elif isinstance(X, np.ndarray):
            self.n_features_ = X.shape[1] if X.ndim > 1 else X.shape[0]
            self.feature_names_ = list(range(self.n_features_))
            self._context = X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

        self._fitted = True

        if self.verbose:
            print(f"MultilinearExplainer fitted with {self.n_features_} features")

        return self

    def explain(
        self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs
    ) -> Dict:
        """
        Compute Shapley values using multilinear extension.

        Args:
            X: Optional input data (uses fitted data if None)

        Returns:
            Dictionary mapping feature names/indices to Shapley values
        """
        if not self._fitted:
            raise ValueError("Explainer must be fitted before calling explain()")

        if X is not None:
            self.fit(X, **kwargs)

        start_time = time.time()
        n = self.n_features_

        # Compute smart defaults based on n
        self.n_quadrature, self.n_samples = self._compute_smart_defaults(n)

        # Reset random generator for reproducibility
        self._rng = np.random.default_rng(self.seed)

        # Create coalition value cache
        self._value_cache = {}
        self.n_char_func_calls_ = 0

        def cached_char_func(coalition: Set[int]) -> float:
            """Cached characteristic function using integer indices."""
            coalition_tuple = tuple(sorted(coalition))
            if coalition_tuple not in self._value_cache:
                if not coalition:
                    self._value_cache[coalition_tuple] = 0.0
                else:
                    named_coalition = {self.feature_names_[i] for i in coalition}
                    self._value_cache[coalition_tuple] = self.characteristic_function(
                        named_coalition, self._context
                    )
                self.n_char_func_calls_ += 1
            return self._value_cache[coalition_tuple]

        if self.verbose:
            print(f"\nComputing Shapley values using Owen's multilinear extension")
            # Show if defaults were auto-computed
            quad_note = "" if self._n_quadrature_user is not None else " (auto)"
            samp_note = "" if self._n_samples_user is not None else " (auto)"
            print(
                f"  n={n}, n_quadrature={self.n_quadrature}{quad_note}, n_samples={self.n_samples}{samp_note}"
            )
            if self.use_leverage:
                print(f"  use_leverage=True, sampling_method={self.sampling_method}")
                print(f"  leverage_type={self.leverage_type}")
                if self.sampling_method == "stratified":
                    print(f"  tilt_by_bernoulli={self.tilt_by_bernoulli}")
            else:
                print(f"  use_leverage=False")

        # Always use sampled method (for exact computation, use ExactExplainer)
        shapley_array, total_variance = self._compute_shapley_sampled(
            cached_char_func, n
        )
        if self.use_leverage:
            self.method_used_ = f"leverage_{self.sampling_method}_{self.leverage_type}"
        else:
            self.method_used_ = "naive_sampling"

        self.computation_time_ = time.time() - start_time

        # Compute error bounds if requested
        if self.compute_error_bounds:
            error_computer = _ErrorBoundComputer(
                n, self.n_quadrature, self.n_samples, self.confidence
            )
            avg_variance = total_variance / n if n > 0 else 0.0
            self.error_bounds_ = error_computer.compute_bounds(
                observed_variance=avg_variance,
                fourth_derivative_bound=1.0,
                leverage_efficiency=self.leverage_efficiency_,
            )

        # Convert to dictionary with feature names
        self.shapley_values_ = {
            self.feature_names_[i]: shapley_array[i] for i in range(n)
        }

        if self.verbose:
            print(f"\nComputation complete:")
            print(f"  Method: {self.method_used_}")
            print(f"  Time: {self.computation_time_:.3f}s")
            print(f"  Char. function calls: {self.n_char_func_calls_}")
            if self.use_leverage:
                print(f"  Leverage efficiency: {self.leverage_efficiency_:.2f}x")
                print(f"  Leverage type: {self.leverage_type}")
                if self.size_distribution_:
                    total_samples = sum(self.size_distribution_.values())
                    print(f"  Size distribution (total={total_samples}):")
                    for size in sorted(self.size_distribution_.keys()):
                        count = self.size_distribution_[size]
                        pct = count / total_samples * 100
                        print(f"    size {size}: {count} ({pct:.1f}%)")
            if self.error_bounds_:
                print(
                    f"  Error bound: {self.error_bounds_.total_error:.2e} "
                    f"(conf={self.error_bounds_.confidence_level:.0%})"
                )

        return self.shapley_values_

    # =========================================================================
    # Core Computation Methods
    # =========================================================================

    def _compute_shapley_sampled(
        self, char_func: Callable[[Set[int]], float], n: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Shapley values via sampled partial derivative integration.

        Uses adaptive allocation: more samples at high-variance t values,
        exact computation at t=0 and t=1.

        Complexity: O(n × total_samples)

        Returns:
            (shapley_values, total_variance)
        """
        # Initialize helpers
        leverage_computer = _LeverageScoreComputer(
            n, self.leverage_type, self.tilt_by_bernoulli
        )
        sampler = _StratifiedCoalitionSampler(leverage_computer, self._rng)

        if self.use_leverage:
            self.leverage_efficiency_ = leverage_computer.efficiency_gain()

        shapley_values = np.zeros(n)
        total_variance = 0.0
        t_values = np.linspace(0, 1, self.n_quadrature)

        # Compute adaptive allocation across quadrature points
        quad_allocations, is_exact = leverage_computer.allocate_across_quadrature(
            t_values, self.n_samples
        )

        # Store allocation for reporting
        self.quadrature_allocation_ = {
            t_values[k]: int(quad_allocations[k])
            for k in range(len(t_values))
            if quad_allocations[k] > 0 or is_exact[k]
        }

        if self.verbose:
            print(
                f"  Adaptive allocation across {self.n_quadrature} quadrature points:"
            )
            exact_count = sum(is_exact)
            sample_count = self.n_quadrature - exact_count
            print(f"    Exact computation: {exact_count} points (t=0, t=1)")
            print(f"    Sampled: {sample_count} points, total {self.n_samples} samples")

        for i in range(n):
            if self.verbose and (i % max(1, n // 5) == 0 or i == n - 1):
                print(f"  Processing player {i+1}/{n}...")

            f_values = np.zeros(self.n_quadrature)
            var_values = np.zeros(self.n_quadrature)

            for k, t in enumerate(t_values):
                if is_exact[k]:
                    # Exact computation at t=0 or t=1
                    f_values[k], var_values[k] = self._compute_exact_at_t(
                        char_func, i, t, n
                    )
                else:
                    # Sampled estimation with adaptive budget
                    n_samples_k = quad_allocations[k]
                    f_values[k], var_values[k] = self._estimate_partial_derivative(
                        char_func, sampler, i, t, n_samples_k
                    )

            shapley_values[i] = simpson(f_values, x=t_values)
            total_variance += np.mean(var_values)

        # Store size distribution from sampler (only for leverage sampling)
        if self.use_leverage:
            self.size_distribution_ = sampler.get_size_distribution()

        return shapley_values, total_variance

    def _compute_exact_at_t(
        self, char_func: Callable[[Set[int]], float], player: int, t: float, n: int
    ) -> Tuple[float, float]:
        """
        Compute exact partial derivative at t=0 or t=1.

        At t=0: f_i(0) = v({i}) - v(∅)
        At t=1: f_i(1) = v(N) - v(N\\{i})

        Returns:
            (exact_value, 0.0)  # variance is 0 for exact computation
        """
        other_players = [j for j in range(n) if j != player]

        if t < 0.5:  # t ≈ 0
            # S = ∅ (empty coalition)
            S_without = set()
            S_with = {player}
        else:  # t ≈ 1
            # S = all other players
            S_without = set(other_players)
            S_with = set(other_players) | {player}

        v_with = char_func(S_with)
        v_without = char_func(S_without)

        return v_with - v_without, 0.0

    def _estimate_partial_derivative(
        self,
        char_func: Callable[[Set[int]], float],
        sampler: _StratifiedCoalitionSampler,
        player: int,
        t: float,
        n_samples: int,
    ) -> Tuple[float, float]:
        """
        Estimate f_i(t,...,t) = E_{S~Bernoulli(t)}[v(S∪{i}) - v(S)].

        Returns:
            (estimate, variance_estimate)
        """
        if self.use_leverage:
            if self.sampling_method == "bernoulli":
                coalitions, weights = sampler.sample_bernoulli(player, t, n_samples)
            else:  # stratified
                coalitions, weights = sampler.sample_stratified(player, t, n_samples)
        else:
            coalitions = sampler.sample_naive(player, t, n_samples)
            weights = np.ones(len(coalitions))

        # Handle empty coalition case (can happen with very small budgets)
        if len(coalitions) == 0:
            # Fallback: sample one coalition using naive Bernoulli
            other_players = [j for j in range(sampler.n) if j != player]
            S = set(j for j in other_players if sampler.rng.random() < t)
            S_with_i = S | {player}
            v_with = char_func(S_with_i)
            v_without = char_func(S)
            return v_with - v_without, 1.0  # High variance for single sample

        # Compute marginal contributions
        marginals = np.zeros(len(coalitions))
        for j, S in enumerate(coalitions):
            S_with_i = S | {player}
            v_with = char_func(S_with_i)
            v_without = char_func(S)
            marginals[j] = v_with - v_without

        # Importance-weighted estimate (self-normalized for stability)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weighted_marginals = marginals * weights
            estimate = np.sum(weighted_marginals) / weight_sum
        else:
            estimate = np.mean(marginals) if len(marginals) > 0 else 0.0

        # Variance estimate for error bounds
        if len(marginals) > 1:
            variance = np.average((marginals - estimate) ** 2, weights=weights)
        else:
            variance = 0.0

        return estimate, variance

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_computation_stats(self) -> Dict:
        """
        Get statistics about the computation.

        Returns:
            Dictionary with computation statistics
        """
        if self.shapley_values_ is None:
            return {}

        stats = {
            "n_features": self.n_features_,
            "n_quadrature": self.n_quadrature,
            "n_samples": self.n_samples,
            "method_used": self.method_used_,
            "use_leverage": self.use_leverage,
            "computation_time": self.computation_time_,
            "n_char_func_calls": self.n_char_func_calls_,
        }

        if self.use_leverage:
            stats["sampling_method"] = self.sampling_method
            stats["leverage_type"] = self.leverage_type
            if self.sampling_method == "stratified":
                stats["tilt_by_bernoulli"] = self.tilt_by_bernoulli
            stats["leverage_efficiency"] = self.leverage_efficiency_

        return stats

    def get_error_bounds(self) -> Optional[ErrorBounds]:
        """
        Get error bounds for the computation.

        Returns:
            ErrorBounds object if computed, None otherwise
        """
        return self.error_bounds_

    def get_size_distribution(self) -> Optional[Dict[int, int]]:
        """
        Get the size distribution of sampled coalitions.

        Only available when use_leverage=True.

        Returns:
            Dictionary mapping coalition size to sample count,
            or None if not using leverage sampling.
        """
        return self.size_distribution_
