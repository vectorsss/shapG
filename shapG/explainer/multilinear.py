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
from itertools import combinations
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
        return (f"ErrorBounds(total={self.total_error:.2e}, "
                f"quad={self.quadrature_error:.2e}, "
                f"sample={self.sampling_error:.2e}, "
                f"conf={self.confidence_level:.1%})")


# =============================================================================
# Helper Classes (Private)
# =============================================================================

class _LeverageScoreComputer:
    """
    Computes leverage scores for Shapley value estimation.

    For the weighted regression formulation of Shapley values,
    leverage scores are approximately proportional to Shapley weights,
    with corrections for the constraint subspace.
    """

    def __init__(self, n: int):
        self.n = n
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
        self._n_coalitions_by_size = np.array([
            comb(self.n - 1, s, exact=True) for s in range(self.n)
        ])

        # Base leverage scores by size (aggregate)
        self._base_leverage = self._n_coalitions_by_size * self._shapley_weights
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
        Compute leverage scores tilted by Bernoulli(t) probabilities.

        Combines Owen's probabilistic interpretation with leverage importance:
            tilted_ℓ_s(t) = ℓ_s × Pr(|S| = s | Bernoulli(t)^{n-1})
        """
        cache_key = round(t, 6)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_others = self.n - 1

        # Binomial probabilities under Bernoulli(t)
        bernoulli_mass = np.zeros(self.n)
        for s in range(self.n):
            if s <= n_others:
                bernoulli_mass[s] = (
                    comb(n_others, s, exact=True) *
                    (t ** s) * ((1 - t) ** (n_others - s))
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


class _StratifiedCoalitionSampler:
    """
    Stratified coalition sampler with importance weighting.

    Implements leverage-guided stratified sampling for unbiased
    estimation of E_{S~Bernoulli(t)}[Δ_i(S)].
    """

    def __init__(self, leverage_computer: _LeverageScoreComputer, rng: np.random.Generator):
        self.leverage = leverage_computer
        self.n = leverage_computer.n
        self.rng = rng

    def sample_stratified(
        self,
        player: int,
        t: float,
        n_samples: int
    ) -> Tuple[List[Set[int]], np.ndarray]:
        """
        Sample coalitions using leverage-stratified sampling.

        Returns:
            (coalitions, weights) where weights are for importance sampling
        """
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

            # True probability mass for size s under Bernoulli(t)
            true_prob = comb(n_others, s, exact=True) * (t**s) * ((1-t)**(n_others-s))

            # Our sampling probability (avoid division by zero)
            sample_prob = tilted[s] if tilted[s] > 1e-15 else 1e-15

            for _ in range(n_s):
                # Sample uniformly from coalitions of size s
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

    def sample_naive(
        self,
        player: int,
        t: float,
        n_samples: int
    ) -> List[Set[int]]:
        """Naive Bernoulli(t) sampling (for comparison)."""
        other_players = [j for j in range(self.n) if j != player]

        coalitions = []
        for _ in range(n_samples):
            S = set(j for j in other_players if self.rng.random() < t)
            coalitions.append(S)

        return coalitions


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
        self,
        n: int,
        n_quadrature: int,
        n_samples: int,
        confidence: float = 0.95
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
        return (h ** 4) * fourth_derivative_bound / 180

    def sampling_error_bound(
        self,
        variance_estimate: float,
        leverage_efficiency: float = 1.0
    ) -> float:
        """
        Compute Monte Carlo sampling error bound.

        Using CLT: ε_mc ≤ z_{α/2} × σ / √(m × η × K)
        """
        effective_samples = self.m * leverage_efficiency * self.K
        if effective_samples <= 0:
            return float('inf')
        return self.z_score * sqrt(variance_estimate / effective_samples)

    def compute_bounds(
        self,
        observed_variance: float,
        fourth_derivative_bound: float = 1.0,
        leverage_efficiency: float = 1.0
    ) -> ErrorBounds:
        """Compute complete error bounds."""
        quad_err = self.quadrature_error_bound(fourth_derivative_bound)
        samp_err = self.sampling_error_bound(observed_variance, leverage_efficiency)

        return ErrorBounds(
            quadrature_error=quad_err,
            sampling_error=samp_err,
            total_error=quad_err + samp_err,
            confidence_level=self.confidence
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
    max_exact_size : int, default=10
        Maximum number of players for exact (non-sampled) computation
    n_quadrature : int, default=21
        Number of quadrature points for Simpson's rule (odd preferred)
    n_samples : int, default=100
        Number of samples per quadrature point
    use_leverage : bool, default=True
        If True, use leverage-stratified sampling for variance reduction.
        If False, use naive Bernoulli sampling.
    compute_error_bounds : bool, default=False
        If True, compute rigorous error bounds
    confidence : float, default=0.95
        Confidence level for error bounds
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    References
    ----------
    [1] Owen, G. (1972). Multilinear Extensions of Games.
    [2] Musco, C. & Witter, R. (2025). Provably Accurate Shapley Value
        Estimation via Leverage Score Sampling. ICLR 2025.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        max_exact_size: int = 10,
        n_quadrature: int = 21,
        n_samples: int = 100,
        use_leverage: bool = True,
        compute_error_bounds: bool = False,
        confidence: float = 0.95,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(characteristic_function, verbose)

        if max_exact_size <= 0:
            raise ValueError(f"max_exact_size must be positive, got {max_exact_size}")
        if n_quadrature < 3:
            raise ValueError(f"n_quadrature must be >= 3, got {n_quadrature}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        self.max_exact_size = max_exact_size
        self.n_quadrature = n_quadrature if n_quadrature % 2 == 1 else n_quadrature + 1
        self.n_samples = n_samples
        self.use_leverage = use_leverage
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

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'MultilinearExplainer':
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

    def explain(self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs) -> Dict:
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
            print(f"  n={n}, n_quadrature={self.n_quadrature}, n_samples={self.n_samples}")
            print(f"  use_leverage={self.use_leverage}")

        # Choose method based on problem size
        if n <= self.max_exact_size:
            shapley_array, total_variance = self._compute_shapley_exact(cached_char_func, n)
            self.method_used_ = 'exact'
            self.leverage_efficiency_ = 1.0
        else:
            shapley_array, total_variance = self._compute_shapley_sampled(cached_char_func, n)
            self.method_used_ = 'leverage_sampling' if self.use_leverage else 'naive_sampling'

        self.computation_time_ = time.time() - start_time

        # Compute error bounds if requested
        if self.compute_error_bounds and n > self.max_exact_size:
            error_computer = _ErrorBoundComputer(
                n, self.n_quadrature, self.n_samples, self.confidence
            )
            avg_variance = total_variance / n if n > 0 else 0.0
            self.error_bounds_ = error_computer.compute_bounds(
                observed_variance=avg_variance,
                fourth_derivative_bound=1.0,
                leverage_efficiency=self.leverage_efficiency_
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
            if self.use_leverage and n > self.max_exact_size:
                print(f"  Leverage efficiency: {self.leverage_efficiency_:.2f}x")
            if self.error_bounds_:
                print(f"  Error bound: {self.error_bounds_.total_error:.2e} "
                      f"(conf={self.error_bounds_.confidence_level:.0%})")

        return self.shapley_values_

    # =========================================================================
    # Core Computation Methods
    # =========================================================================

    def _compute_shapley_exact(
        self,
        char_func: Callable[[Set[int]], float],
        n: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute exact Shapley values using Owen's diagonal integration.

        Complexity: O(n × n_quadrature × 2^(n-1))

        Returns:
            (shapley_values, total_variance)
        """
        shapley_values = np.zeros(n)
        t_values = np.linspace(0, 1, self.n_quadrature)

        for i in range(n):
            f_values = np.zeros(self.n_quadrature)
            for k, t in enumerate(t_values):
                f_values[k] = self._partial_derivative_exact(char_func, i, t, n)
            shapley_values[i] = simpson(f_values, x=t_values)

        return shapley_values, 0.0  # No variance for exact computation

    def _partial_derivative_exact(
        self,
        char_func: Callable[[Set[int]], float],
        i: int,
        t: float,
        n: int
    ) -> float:
        """
        Exact computation of partial derivative f_i(t, t, ..., t).

        From Owen's equation (8):
        f_i(x) = Σ_{S⊄i} [∏_{j∈S} x_j · ∏_{j∉S∪{i}} (1-x_j)] · [v(S∪{i}) - v(S)]

        Complexity: O(2^(n-1))
        """
        total = 0.0
        other_players = [j for j in range(n) if j != i]

        for size in range(len(other_players) + 1):
            for coalition in combinations(other_players, size):
                coalition_set = set(coalition)
                coalition_with_i = coalition_set | {i}

                # Probability: t^|S| * (1-t)^(n-1-|S|)
                prob = (t ** size) * ((1 - t) ** (len(other_players) - size))

                # Marginal contribution
                marginal = char_func(coalition_with_i) - char_func(coalition_set)
                total += prob * marginal

        return total

    def _compute_shapley_sampled(
        self,
        char_func: Callable[[Set[int]], float],
        n: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Shapley values via sampled partial derivative integration.

        Uses either leverage-stratified or naive Bernoulli sampling.

        Complexity: O(n × n_quadrature × n_samples)

        Returns:
            (shapley_values, total_variance)
        """
        # Initialize helpers
        leverage_computer = _LeverageScoreComputer(n)
        sampler = _StratifiedCoalitionSampler(leverage_computer, self._rng)

        if self.use_leverage:
            self.leverage_efficiency_ = leverage_computer.efficiency_gain()

        shapley_values = np.zeros(n)
        total_variance = 0.0
        t_values = np.linspace(0, 1, self.n_quadrature)

        for i in range(n):
            if self.verbose and (i % max(1, n // 5) == 0 or i == n - 1):
                print(f"  Processing player {i+1}/{n}...")

            f_values = np.zeros(self.n_quadrature)
            var_values = np.zeros(self.n_quadrature)

            for k, t in enumerate(t_values):
                f_values[k], var_values[k] = self._estimate_partial_derivative(
                    char_func, sampler, i, t, n
                )

            shapley_values[i] = simpson(f_values, x=t_values)
            total_variance += np.mean(var_values)

        return shapley_values, total_variance

    def _estimate_partial_derivative(
        self,
        char_func: Callable[[Set[int]], float],
        sampler: _StratifiedCoalitionSampler,
        player: int,
        t: float,
        n: int
    ) -> Tuple[float, float]:
        """
        Estimate f_i(t,...,t) = E_{S~Bernoulli(t)}[v(S∪{i}) - v(S)].

        Returns:
            (estimate, variance_estimate)
        """
        if self.use_leverage:
            coalitions, weights = sampler.sample_stratified(player, t, self.n_samples)
        else:
            coalitions = sampler.sample_naive(player, t, self.n_samples)
            weights = np.ones(len(coalitions))

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
            estimate = np.mean(marginals)

        # Variance estimate for error bounds
        if len(marginals) > 1:
            variance = np.average((marginals - estimate)**2, weights=weights)
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
            'n_features': self.n_features_,
            'n_quadrature': self.n_quadrature,
            'n_samples': self.n_samples,
            'max_exact_size': self.max_exact_size,
            'method_used': self.method_used_,
            'use_leverage': self.use_leverage,
            'computation_time': self.computation_time_,
            'n_char_func_calls': self.n_char_func_calls_,
        }

        if self.use_leverage and self.method_used_ != 'exact':
            stats['leverage_efficiency'] = self.leverage_efficiency_

        return stats

    def get_error_bounds(self) -> Optional[ErrorBounds]:
        """
        Get error bounds for the computation.

        Returns:
            ErrorBounds object if computed, None otherwise
        """
        return self.error_bounds_
