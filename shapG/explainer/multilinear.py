"""
Multilinear extension-based Shapley value computation using Owen (1972) theory.

This module computes Shapley values through sampled partial derivative integration
along the main diagonal of the multilinear extension.

Key insight: Instead of enumerating all 2^n coalitions, we:
1. Sample partial derivatives at quadrature points along the diagonal
2. Integrate numerically to get Shapley values

Complexity: O(n × n_quadrature × n_samples) instead of O(2^n)

Note: This explainer does NOT use graph structure. It computes Shapley values
for all n players based purely on the characteristic function, following
Owen's original formulation.

Reference:
    Owen, G. (1972). Multilinear Extensions of Games.
    Management Science, 18(5-Part-2), 64-79.
"""

from typing import Dict, Union, Optional, Set, Callable
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import time

from scipy.integrate import simpson

from .base import Explainer, CharacteristicFunction


class MultilinearExplainer(Explainer):
    """
    Shapley value computation using Owen's multilinear extension.

    This explainer uses the multilinear extension framework from Owen (1972)
    to compute Shapley values via sampled partial derivative integration.

    The multilinear extension f(x) of a game v is defined as:
        f(x) = sum_{S in N} [prod_{j in S} x_j * prod_{j not in S} (1-x_j)] * v(S)

    The Shapley value is obtained by integrating partial derivatives along
    the main diagonal (Owen's Theorem 5):
        phi_i = integral_0^1 (df/dx_i)(t, t, ..., t) dt

    Algorithm:
        For each feature i:
            1. Sample f_i(t, t, ..., t) at quadrature points t in [0, 1]
            2. Integrate: phi_i = integral_0^1 f_i(t, ..., t) dt

    Complexity: O(n × n_quadrature × n_samples) characteristic function calls

    Note: This explainer does not use graph structure. It computes Shapley values
    for all n players based purely on the characteristic function.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        max_exact_size: int = 10,
        n_quadrature: int = 21,
        n_samples: int = 100,
        verbose: bool = False
    ):
        """
        Initialize the multilinear explainer.

        Args:
            characteristic_function: Function v(S, context) -> float for coalition S
            max_exact_size: Maximum number of players for exact computation
            n_quadrature: Number of quadrature points for integration (odd preferred)
            n_samples: Number of samples per quadrature point (for large n)
            verbose: Whether to print progress
        """
        super().__init__(characteristic_function, verbose)

        if max_exact_size <= 0:
            raise ValueError(f"max_exact_size must be positive, got {max_exact_size}")
        if n_quadrature < 3:
            raise ValueError(f"n_quadrature must be >= 3, got {n_quadrature}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        self.max_exact_size = max_exact_size
        self.n_quadrature = n_quadrature
        self.n_samples = n_samples

        # Results storage
        self.shapley_values_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self.computation_time_ = None
        self.method_used_ = None
        self.n_char_func_calls_ = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'MultilinearExplainer':
        """
        Fit the explainer to data.

        Args:
            X: Input data - can be:
               - numpy array or DataFrame: uses number of columns as n_features
               - networkx Graph: uses number of nodes as n_features

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
                    # Convert indices to feature names for the characteristic function
                    named_coalition = {self.feature_names_[i] for i in coalition}
                    self._value_cache[coalition_tuple] = self.characteristic_function(
                        named_coalition, self._context
                    )
                self.n_char_func_calls_ += 1
            return self._value_cache[coalition_tuple]

        if self.verbose:
            print(f"\nComputing Shapley values using Owen's multilinear extension")
            print(f"n={n}, n_quadrature={self.n_quadrature}, n_samples={self.n_samples}")

        # Choose method based on problem size
        if n <= self.max_exact_size:
            shapley_array = self._compute_shapley_exact(cached_char_func, n)
            self.method_used_ = 'exact'
        else:
            shapley_array = self._compute_shapley_sampled(cached_char_func, n)
            self.method_used_ = 'sampled_integration'

        self.computation_time_ = time.time() - start_time

        # Convert to dictionary with feature names
        self.shapley_values_ = {
            self.feature_names_[i]: shapley_array[i] for i in range(n)
        }

        if self.verbose:
            print(f"\nComputation complete:")
            print(f"  Method: {self.method_used_}")
            print(f"  Time: {self.computation_time_:.3f}s")
            print(f"  Char. function calls: {self.n_char_func_calls_}")

        return self.shapley_values_

    # =========================================================================
    # Core Multilinear Extension Methods
    # =========================================================================

    def _compute_shapley_exact(
        self,
        char_func: Callable[[Set[int]], float],
        n: int
    ) -> np.ndarray:
        """
        Compute exact Shapley values using Owen's diagonal integration.

        Uses exact partial derivative computation with numerical integration.
        Complexity: O(n × n_quadrature × 2^(n-1))

        Args:
            char_func: Characteristic function (uses integer indices)
            n: Number of players

        Returns:
            Array of Shapley values
        """
        shapley_values = np.zeros(n)
        n_quad = self.n_quadrature

        # Use odd number for Simpson's rule
        if n_quad % 2 == 0:
            n_quad += 1
        t_values = np.linspace(0, 1, n_quad)

        for i in range(n):
            f_values = np.zeros(n_quad)
            for k, t in enumerate(t_values):
                f_values[k] = self._partial_derivative_exact(char_func, i, t, n)
            shapley_values[i] = simpson(f_values, x=t_values)

        return shapley_values

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
        f_i(x) = sum_{S not containing i} [prod_{j in S} x_j * prod_{j not in S,i} (1-x_j)]
                 * [v(S union {i}) - v(S)]

        Complexity: O(2^(n-1))

        Args:
            char_func: Characteristic function
            i: Player index
            t: Diagonal parameter
            n: Number of players

        Returns:
            Partial derivative value
        """
        total = 0.0
        other_players = [j for j in range(n) if j != i]

        for size in range(len(other_players) + 1):
            for coalition in combinations(other_players, size):
                coalition_set = set(coalition)
                coalition_with_i = coalition_set | {i}

                # Compute probability: t^|S| * (1-t)^(n-1-|S|)
                prob = (t ** size) * ((1 - t) ** (len(other_players) - size))

                # Marginal contribution
                marginal = char_func(coalition_with_i) - char_func(coalition_set)
                total += prob * marginal

        return total

    def _compute_shapley_sampled(
        self,
        char_func: Callable[[Set[int]], float],
        n: int
    ) -> np.ndarray:
        """
        Compute Shapley values via sampled partial derivative integration.

        This implements Owen's Theorem 5 using:
        1. Numerical quadrature along the diagonal t in [0, 1]
        2. Monte Carlo sampling to estimate f_i(t, ..., t) at each point

        Complexity: O(n × n_quadrature × n_samples)

        Args:
            char_func: Characteristic function
            n: Number of players

        Returns:
            Array of Shapley values
        """
        shapley_values = np.zeros(n)
        n_quad = self.n_quadrature

        # Use odd number for Simpson's rule
        if n_quad % 2 == 0:
            n_quad += 1
        t_values = np.linspace(0, 1, n_quad)

        for i in range(n):
            f_values = np.zeros(n_quad)
            for k, t in enumerate(t_values):
                f_values[k] = self._sample_partial_derivative(char_func, i, t, n)
            shapley_values[i] = simpson(f_values, x=t_values)

        return shapley_values

    def _sample_partial_derivative(
        self,
        char_func: Callable[[Set[int]], float],
        i: int,
        t: float,
        n: int
    ) -> float:
        """
        Estimate partial derivative f_i(t, t, ..., t) using sampling.

        At point (t, t, ..., t) on the diagonal, each player j (j != i) joins
        the coalition independently with probability t. We sample coalitions
        from this distribution and compute the expected marginal contribution.

        This directly implements Owen's equation (8) via Monte Carlo:
        f_i(t,...,t) = E_{S ~ Bernoulli(t)} [v(S union {i}) - v(S)]

        Args:
            char_func: Characteristic function
            i: Target player index
            t: Diagonal parameter (probability each other player joins)
            n: Number of players

        Returns:
            Estimated partial derivative at (t, t, ..., t)
        """
        contributions = []
        other_players = [j for j in range(n) if j != i]

        for _ in range(self.n_samples):
            # Each player j joins with probability t (independent Bernoulli)
            S = set(j for j in other_players if np.random.random() < t)
            S_with_i = S | {i}

            # Marginal contribution of player i to coalition S
            marginal = char_func(S_with_i) - char_func(S)
            contributions.append(marginal)

        return np.mean(contributions)

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

        return {
            'n_features': self.n_features_,
            'n_quadrature': self.n_quadrature,
            'n_samples': self.n_samples,
            'max_exact_size': self.max_exact_size,
            'method_used': self.method_used_,
            'computation_time': self.computation_time_,
            'n_char_func_calls': self.n_char_func_calls_,
        }
