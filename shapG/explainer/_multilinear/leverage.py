"""Leverage score computation for Shapley value estimation."""

from typing import Tuple, List
from math import factorial, log
import numpy as np
from scipy.special import comb


class LeverageScoreComputer:
    """
    Computes leverage scores for Shapley value estimation.

    Supports two leverage score definitions:

    1. "shapley_weighted" (default): Leverage proportional to Shapley weights.
       l_s proportional to C(n-1,s) * w_s, where w_s = s!(n-s-1)!/n!
       This weights coalition sizes by their Shapley importance.

    2. "inverse_coalition" (Musco & Witter 2025): Pure inverse coalition count.
       l_s = 1/C(n-1,s)
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
            # Original: l_s proportional to C(n-1,s) * shapley_weights
            self._base_leverage = self._n_coalitions_by_size * self._shapley_weights
        else:  # inverse_coalition
            # Musco & Witter 2025: l_s = 1/C(n-1,s)
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
            tilted_l_s(t) = l_s * Pr(|S| = s | Bernoulli(t)^{n-1})
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

        Solves: target_budget ~ sum_{s=0}^{n-1} min(N_s, 2c * l_s * N_s)
        where N_s = C(n-1, s), l_s = leverage score for size s

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
        proportionally to the variance t*(1-t).

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
