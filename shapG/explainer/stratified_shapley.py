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
    [2] Compressed Sensing: Candes, E. J., & Wakin, M. B. (2008).
        "An Introduction to Compressive Sampling"
    [3] Leverage Score Sampling: Musco, C., & Witter, R. T. (2025).
        "Provably Accurate Shapley Value Estimation via Leverage Score Sampling"
"""

from typing import Dict, Set, Union, Optional, Tuple, List, Callable, Any
import numpy as np
import pandas as pd
import networkx as nx
import warnings

from scipy.fft import idct

from .base import Explainer, CharacteristicFunction
from ..characteristic.characteristic_functions import CoalitionDegree

# Import from submodules
from ._stratified import (
    SamplingStats,
    StratifiedCoalitionSampler,
    ImplicitDCTOperator,
    L1Solver,
)

# Re-export for backward compatibility
__all__ = [
    "SamplingStats",
    "StratifiedCoalitionSampler",
    "ImplicitDCTOperator",
    "L1Solver",
    "StratifiedShapleyExplainer",
]


class StratifiedShapleyExplainer(Explainer):
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
    allocation_strategy : str, default='shapley_weighted'
        How to allocate samples across coalition sizes:
        - 'shapley_weighted': Proportional to Shapley weight x coalition count (recommended)
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
    >>> from shapG import StratifiedShapleyExplainer, CoalitionDegree
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(20, 0.3)
    >>> explainer = StratifiedShapleyExplainer(
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
        allocation_strategy: str = "shapley_weighted",
        tolerance: float = 1e-4,
        use_direct_estimation: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(characteristic_function or CoalitionDegree(), verbose)

        self.n_samples = n_samples
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
        context: Optional[Any] = None,
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
        **kwargs,
    ) -> "StratifiedShapleyExplainer":
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
            seed=self.seed,
        )

        if self.verbose:
            print(f"RP-QRCS Explainer fitted:")
            print(f"  n_players: {self.n}")
            print(f"  n_samples: {self.n_samples}")
            print(f"  allocation_strategy: {self.allocation_strategy}")
            print(f"\nSampling allocation:")
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
        **kwargs,
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
        shapley_values = {
            self.player_ids[i]: value for i, value in shapley_indices.items()
        }

        return shapley_values

    def _compute_shapley(
        self, utility_func: Callable[[Set[int]], float]
    ) -> Dict[int, float]:
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
            if self.allocation_strategy == "leverage_bernoulli":
                coalitions, sizes, weights = self._sampler.sample_all_strata_bernoulli(
                    player
                )

                # Print actual allocation after first player's sampling
                if (
                    player == 0
                    and self.verbose
                    and self._sampler._actual_samples_by_size is not None
                ):
                    print(f"\nActual sampling allocation (leverage_bernoulli):")
                    for s in range(self.n):
                        if self._sampler._actual_samples_by_size[s] > 0:
                            print(
                                f"    size {s}: {int(self._sampler._actual_samples_by_size[s])} samples"
                            )
                    print()
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
                    mask = sizes == size
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

    def _compute_with_cs_implicit(
        self,
        coalitions: List[Set[int]],
        player: int,
        utility_func: Callable[[Set[int]], float],
        l1_solver: L1Solver,
    ) -> float:
        """
        Compute Shapley value using compressed sensing with implicit DCT.

        Uses budget-based stratified CS which:
        1. Measures only m sampled coalitions
        2. Uses implicit DCT operator in O(m) budget space (not O(2^n))
        3. Reconstructs via L1 minimization
        4. Computes Shapley value using stratified structure

        Falls back to direct estimation if CS fails.

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

        if self.verbose:
            n_bits = self.n - 1
            full_size = 2**n_bits
            print(f"  [CS] Full space: 2^{n_bits} = {full_size}")
            print(f"  [CS] Sampled: {m} coalitions ({100*m/full_size:.4f}%)")
            print(f"  [CS] Using budget-based stratified CS (O(m) space)")

        try:
            return self._compute_with_cs_stratified_budget(
                coalitions, player, utility_func, l1_solver
            )
        except Exception as e:
            warnings.warn(
                f"CS reconstruction failed: {e}. Falling back to direct estimation.",
                RuntimeWarning,
            )
            return self._compute_direct_from_coalitions(
                coalitions, player, utility_func
            )

    def _compute_cs_oversampling_factors(
        self, allocations: np.ndarray, base_oversampling: int = 4
    ) -> np.ndarray:
        """
        Compute per-stratum oversampling factors for budget-based CS.

        This method handles all allocation strategies uniformly by computing
        factors based on actual coverage ratio:
        - Exhaustive strata (n_samples >= n_total): factor = 1
        - Partial strata: factor = min(base_oversampling, ceil(n_total / n_samples))

        The coverage-based formula works for all strategies:
        - shapley_weighted: deterministic allocation
        - uniform: equal allocation
        - leverage: inverse coalition count
        - leverage_bernoulli: Bernoulli sampling with probability ~ c * leverage_score

        Capping at ceil(n_total / n_samples) prevents over-allocation beyond
        what's useful for reconstruction.

        Parameters
        ----------
        allocations : np.ndarray
            Number of samples per stratum (size n)
        base_oversampling : int
            Maximum oversampling factor for partial strata (default 4)

        Returns
        -------
        np.ndarray
            Oversampling factor for each stratum
        """
        factors = np.ones(self.n, dtype=int)

        for s in range(self.n):
            n_total = self._sampler.n_coalitions_by_size[s]
            n_samples = allocations[s]

            if n_samples == 0:
                # No samples in this stratum, factor doesn't matter
                factors[s] = 1
            elif n_samples >= n_total:
                # Exhaustive sampling - no oversampling needed
                factors[s] = 1
            else:
                # Partial sampling - compute effective oversampling
                # Cap at ceil(n_total / n_samples) to avoid wasting space
                max_useful_factor = int(np.ceil(n_total / n_samples))
                factors[s] = min(base_oversampling, max_useful_factor)

        return factors

    def _compute_with_cs_stratified_budget(
        self,
        coalitions: List[Set[int]],
        player: int,
        utility_func: Callable[[Set[int]], float],
        l1_solver: L1Solver,
        base_oversampling: int = 4,
    ) -> float:
        """
        Budget-based CS with stratified one-to-one mapping.

        Key innovations:
        1. Reconstruction space scales with budget, not 2^(n-1)
        2. Stratified structure preserved from allocation strategy
        3. Smart per-stratum oversampling based on coverage ratio:
           - Exhaustive strata (n_samples >= n_total): factor = 1
           - Partial strata: factor = min(base, ceil(n_total / n_samples))
        4. One-to-one mapping (no collisions)
        5. Spread indices within each stratum for CS incoherence

        Parameters
        ----------
        coalitions : List[Set[int]]
            Sampled coalitions from allocation strategy
        player : int
            Target player index
        utility_func : callable
            Function to compute coalition value
        l1_solver : L1Solver
            L1 minimization solver
        base_oversampling : int
            Oversampling factor for partial strata (default 4)

        Returns
        -------
        float
            Shapley value for the player
        """
        m = len(coalitions)

        # Step 1: Compute marginals and track sizes
        y = np.zeros(m)
        sizes = np.zeros(m, dtype=int)
        for j, S in enumerate(coalitions):
            v_with = utility_func(S | {player})
            v_without = utility_func(S) if S else 0.0
            y[j] = v_with - v_without
            sizes[j] = len(S)

        # Step 2: Count samples per size stratum
        allocations = np.zeros(self.n, dtype=int)
        for s in range(self.n):
            allocations[s] = np.sum(sizes == s)

        # Step 3: Compute per-stratum oversampling factors
        # Uses coverage-based formula that works for all allocation strategies
        oversampling_factors = self._compute_cs_oversampling_factors(
            allocations, base_oversampling
        )
        n_exhaustive = np.sum(oversampling_factors == 1)
        n_partial = self.n - n_exhaustive

        # Step 4: Compute stratum offsets in reconstruction space
        recon_offsets = np.zeros(self.n + 1, dtype=int)
        for s in range(self.n):
            recon_offsets[s + 1] = (
                recon_offsets[s] + oversampling_factors[s] * allocations[s]
            )

        recon_size = recon_offsets[-1]

        # Round up to next power of 2 for efficient DCT
        recon_size_padded = 1
        while recon_size_padded < recon_size:
            recon_size_padded *= 2

        if self.verbose:
            print(f"  [CS-Stratified] m={m}, recon_size={recon_size_padded}")
            print(
                f"  [CS-Stratified] Exhaustive strata: {n_exhaustive}, "
                f"Partial strata: {n_partial}"
            )
            # Show range of oversampling factors used
            unique_factors = np.unique(oversampling_factors[oversampling_factors > 1])
            if len(unique_factors) > 0:
                print(
                    f"  [CS-Stratified] Oversampling factors: {unique_factors.tolist()}"
                )

        # Step 5: Create one-to-one mapping and pre-compute Shapley weights
        # Within each stratum: position j -> offset + j * stride
        # Weight for sample j: w_s * N_s / a_s (importance sampling correction)
        sampled_indices = np.zeros(m, dtype=int)
        recon_weights = np.zeros(m)
        within_stratum_counter = np.zeros(self.n, dtype=int)

        for j in range(m):
            s = sizes[j]
            stride = oversampling_factors[s]
            # Position in recon space = stratum_offset + within_stratum_idx * stride
            sampled_indices[j] = recon_offsets[s] + within_stratum_counter[s] * stride
            within_stratum_counter[s] += 1
            # Pre-compute weight: w_s * N_s / a_s
            recon_weights[j] = (
                self._sampler.shapley_weights[s]
                * self._sampler.n_coalitions_by_size[s]
                / allocations[s]
            )

        # Step 6: Create implicit DCT operator in budget-based space
        A = ImplicitDCTOperator(sampled_indices, recon_size_padded)

        # Step 7: L1 solve
        try:
            s_hat = l1_solver.solve(A, y)

            if self.verbose:
                zero_threshold = 1e-10
                sparsity = np.sum(np.abs(s_hat) < zero_threshold) / len(s_hat)
                print(f"  [CS-Stratified] DCT sparsity: {sparsity*100:.1f}%")

            # Step 8: Reconstruct signal
            u_hat = idct(s_hat, norm="ortho")

            # Step 9: Compute Shapley value via dot product with pre-computed weights
            # phi = sum_j recon_weights[j] * u_hat[sampled_indices[j]]
            shapley_value = float(np.dot(recon_weights, u_hat[sampled_indices]))

            return shapley_value

        except Exception as e:
            warnings.warn(
                f"Stratified CS failed: {e}. Falling back to direct estimation.",
                RuntimeWarning,
            )
            return self._compute_direct_from_coalitions(
                coalitions, player, utility_func
            )

    def _compute_direct_from_coalitions(
        self,
        coalitions: List[Set[int]],
        player: int,
        utility_func: Callable[[Set[int]], float],
    ) -> float:
        """
        Compute Shapley value using direct estimation from sampled coalitions.

        This is a fallback for when CS reconstruction is infeasible due to
        memory constraints.

        Parameters
        ----------
        coalitions : List[Set[int]]
            Sampled coalitions
        player : int
            Target player index
        utility_func : callable
            Function to compute coalition value

        Returns
        -------
        float
            Shapley value estimate
        """
        if not coalitions:
            return 0.0

        # Compute marginals and sizes
        marginals = []
        sizes = []
        for S in coalitions:
            v_with = utility_func(S | {player})
            v_without = utility_func(S) if S else 0.0
            marginals.append(v_with - v_without)
            sizes.append(len(S))

        marginals = np.array(marginals)
        sizes = np.array(sizes)

        # Direct estimation using stratified sampling formula
        shapley_value = 0.0
        for size in range(self.n):
            mask = sizes == size
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
