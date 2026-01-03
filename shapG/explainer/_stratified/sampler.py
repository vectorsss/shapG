"""Stratified coalition sampler for Shapley value computation."""

from typing import Optional, Tuple, List, Set
from math import factorial
import numpy as np
import warnings

from scipy.special import comb

from .stats import SamplingStats


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
        - 'shapley_weighted': proportional to Shapley weight x stratum size
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
        allocation_strategy: str = "shapley_weighted",
        seed: Optional[int] = None,
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
        if allocation_strategy != "leverage_bernoulli":
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
        self.n_coalitions_by_size = np.array(
            [comb(n - 1, s, exact=True) for s in range(n)]
        )

        # Total coalitions
        self.total_coalitions = 2 ** (n - 1)

    def _compute_allocations(self) -> np.ndarray:
        """Compute sample allocation across strata."""
        n = self.n

        if self.allocation_strategy == "shapley_weighted":
            # Importance = Shapley weight x number of coalitions of that size
            importance = self.shapley_weights * self.n_coalitions_by_size
            importance_sum = importance.sum()
            if importance_sum > 0:
                importance /= importance_sum
            else:
                importance = np.ones(n) / n

        elif self.allocation_strategy == "uniform":
            # Equal allocation per stratum
            importance = np.ones(n) / n

        elif self.allocation_strategy == "leverage":
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

        elif self.allocation_strategy == "leverage_bernoulli":
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
                    remaining, self.n_coalitions_by_size[idx] - allocations[idx]
                )
                allocations[idx] += can_add
                remaining -= can_add

        return allocations

    def _compute_leverage_score(self, size: int, n_others: int) -> float:
        """
        Compute leverage score for coalition size.

        Adapted from Lemma 3.2: l_s = C(n-1, s)^{-1}

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

        Solves: target_budget ~ sum_{s=0}^{n-1} min(N_s, 2c * l_s * N_s)
        where N_s = C(n_others, s), l_s = 1/N_s

        Note: With paired sampling, actual samples ~ 2 * target_budget

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
        self, player: int
    ) -> Tuple[List[Set[int]], np.ndarray, np.ndarray]:
        """
        Sample coalitions using Bernoulli sampling with leverage scores.

        Adapted from LeverageScoreExplainer (Musco & Witter 2025):
        - Uses leverage scores l_s = C(n-1, s)^{-1}
        - Bernoulli sampling with probability p_s = min(1, 2c*l_s)
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
            is_middle_size = s == n_others - s

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
                leverage_score_complement = self._compute_leverage_score(
                    complement_size, n_others
                )
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

    def sample_all_strata(
        self, player: int
    ) -> Tuple[List[Set[int]], np.ndarray, np.ndarray]:
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
                sampled = self._sample_coalitions_of_size(
                    other_players, size, n_samples
                )
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
        self, players: List[int], size: int, n_samples: int
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
                RuntimeWarning,
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
                coverage_by_size=coverage,
            )

        # Use accumulated actual samples if available, otherwise use allocations
        actual_samples = self._actual_samples_by_size
        if actual_samples is None:
            actual_samples = self.allocations.copy()

        coverage = np.zeros(self.n)
        for s in range(self.n):
            if self.n_coalitions_by_size[s] > 0:
                coverage[s] = actual_samples[s] / self.n_coalitions_by_size[s]

        return SamplingStats(
            n_players=self.n,
            total_budget=self.budget,
            allocations_by_size=self.allocations,
            actual_samples_by_size=actual_samples,
            coverage_by_size=coverage,
        )
