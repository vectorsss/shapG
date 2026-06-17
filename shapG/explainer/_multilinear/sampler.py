"""Stratified coalition sampler for multilinear extension."""

from typing import Tuple, List, Set, Dict
from itertools import combinations
import numpy as np
from scipy.special import comb

from .leverage import LeverageScoreComputer


class StratifiedCoalitionSampler:
    """
    Stratified coalition sampler with importance weighting.

    Implements leverage-guided stratified sampling for unbiased
    estimation of E_{S~Bernoulli(t)}[Delta_i(S)].
    """

    def __init__(
        self, leverage_computer: LeverageScoreComputer, rng: np.random.Generator
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
        - Uses leverage scores l_s from the leverage computer
        - Bernoulli sampling with probability p_s = min(1, 2c*l_s)
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
