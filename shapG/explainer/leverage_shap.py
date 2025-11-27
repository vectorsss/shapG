"""
Leverage SHAP: Provably Accurate Shapley Value Estimation via Leverage Score Sampling

Implementation based on:
"Provably Accurate Shapley Value Estimation via Leverage Score Sampling"
by Christopher Musco and R. Teal Witter (ICLR 2025)

Key features:
- O(n log n) model evaluations with provable accuracy guarantees
- Leverage score sampling for efficient subset selection
- Paired sampling for balanced feature representation
- Bernoulli sampling without replacement

Note: This explainer does NOT use graph structure. It computes Shapley values
for all n players based purely on the characteristic function.
"""

from typing import Dict, Set, Union, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import comb
import warnings

from .base import Explainer, CharacteristicFunction


class LeverageScoreExplainer(Explainer):
    """
    Leverage SHAP explainer for computing Shapley values with theoretical guarantees.

    This implementation provides provably accurate Shapley value estimates with just
    O(n log n) model evaluations, achieving ~50% reduction in error compared to
    Kernel SHAP on average.

    Note: This explainer does not use graph structure. It computes Shapley values
    for all n players based purely on the characteristic function.

    Parameters
    ----------
    characteristic_function : CharacteristicFunction, optional
        Function to compute coalition values
    n_samples : int, optional (default=None)
        Target number of function evaluations. If None, uses 5*n_features.
        Algorithm uses m = O(n log(n/δ) + n/(εδ)) for theoretical guarantees.
    paired_sampling : bool, optional (default=True)
        Whether to use paired sampling (sample S and its complement simultaneously).
        This balances samples and improves accuracy.
    use_bernoulli : bool, optional (default=True)
        Whether to use Bernoulli sampling without replacement.
        Improves accuracy especially for small n.
    random_state : int or None, optional (default=None)
        Random seed for reproducibility.
    verbose : bool, optional (default=False)
        Whether to print progress information.

    References
    ----------
    Musco, C., & Witter, R. T. (2025). Provably Accurate Shapley Value Estimation
    via Leverage Score Sampling. ICLR 2025.
    """

    def __init__(
        self,
        characteristic_function: Optional[CharacteristicFunction] = None,
        n_samples: Optional[int] = None,
        paired_sampling: bool = True,
        use_bernoulli: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__(characteristic_function, verbose)
        self.n_samples = n_samples
        self.paired_sampling = paired_sampling
        self.use_bernoulli = use_bernoulli
        self.random_state = random_state

        # To be set during fitting
        self.shapley_values_ = None
        self.n_features_ = None
        self.feature_names_ = None
        self._v_empty = None
        self._v_full = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, nx.Graph], **kwargs) -> 'LeverageScoreExplainer':
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

        n = self.n_features_

        # Determine number of samples
        m = self.n_samples if self.n_samples is not None else 5 * n
        m = min(m, 2**n)  # Cap at total possible subsets

        if self.verbose:
            print(f"Computing Shapley values for n={n} features with m={m} samples")

        # Compute Shapley values using leverage score sampling
        self.shapley_values_ = self._compute_shapley_values(m)
        self._fitted = True

        return self

    def explain(self, X: Optional[Union[np.ndarray, pd.DataFrame, nx.Graph]] = None, **kwargs) -> Dict:
        """
        Compute Shapley values.

        Args:
            X: Optional input data (uses fitted data if None)

        Returns:
            Dictionary mapping feature names/indices to Shapley values
        """
        if not self._fitted:
            raise ValueError("Must call fit() before explain()")

        # Return Shapley values as dictionary mapping feature names to values
        return dict(zip(self.feature_names_, self.shapley_values_))

    def _compute_shapley_values(self, m: int) -> np.ndarray:
        """
        Compute Shapley values using the Leverage SHAP algorithm.

        Parameters
        ----------
        m : int
            Target number of function evaluations

        Returns
        -------
        np.ndarray
            Shapley values for each feature
        """
        n = self.n_features_

        # Step 1: Compute boundary values v(∅) and v([n])
        empty_coalition = set()
        full_coalition = set(self.feature_names_)

        self._v_empty = self.characteristic_function(empty_coalition, self._context)
        self._v_full = self.characteristic_function(full_coalition, self._context)

        if self.verbose:
            print(f"v(∅) = {self._v_empty:.4f}, v([n]) = {self._v_full:.4f}")

        # Step 2: Sample coalitions using leverage score sampling
        if self.use_bernoulli:
            Z_prime, weights = self._bernoulli_sample(n, m)
        else:
            Z_prime, weights = self._simple_leverage_sample(n, m)

        n_sampled = len(Z_prime)
        if self.verbose:
            print(f"Sampled {n_sampled} coalitions")

        # Step 3: Evaluate value function on sampled coalitions
        y_prime = self._evaluate_coalitions(Z_prime) - self._v_empty

        # Step 4: Solve weighted constrained regression
        shapley_values = self._solve_regression(Z_prime, y_prime, weights, n)

        return shapley_values

    def _evaluate_coalitions(self, Z: np.ndarray) -> np.ndarray:
        """
        Evaluate characteristic function on binary coalition vectors.

        Parameters
        ----------
        Z : np.ndarray
            Binary matrix where each row represents a coalition

        Returns
        -------
        np.ndarray
            Coalition values
        """
        values = np.zeros(len(Z))
        for i, coalition_vec in enumerate(Z):
            # Convert binary vector to set of feature names
            coalition = {self.feature_names_[j] for j in range(len(coalition_vec)) if coalition_vec[j] == 1}
            values[i] = self.characteristic_function(coalition, self._context)
        return values

    def _compute_leverage_score(self, subset_size: int, n: int) -> float:
        """
        Compute leverage score for a subset of given size.

        For Leverage SHAP, leverage score is: ℓ_z = (n choose |S|)^(-1)

        Parameters
        ----------
        subset_size : int
            Size of the subset |S|
        n : int
            Total number of features

        Returns
        -------
        float
            Leverage score
        """
        if subset_size == 0 or subset_size == n:
            return 0.0
        return 1.0 / comb(n, subset_size, exact=False)

    def _compute_kernel_weight(self, subset_size: int, n: int) -> float:
        """
        Compute Kernel SHAP weight for regression.

        w(s) = (n-1) / (n choose s * s * (n-s))

        Parameters
        ----------
        subset_size : int
            Size of the subset |S|
        n : int
            Total number of features

        Returns
        -------
        float
            Kernel weight
        """
        s = subset_size
        if s == 0 or s == n:
            return 1e10  # Large weight for boundary

        # w(s) = (s-1)!(n-s-1)! / n! = (n-1) / (C(n,s) * s * (n-s))
        return (n - 1) / (comb(n, s, exact=False) * s * (n - s))

    def _bernoulli_sample(
        self,
        n: int,
        m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Bernoulli sampling without replacement (Algorithm 2 from paper).

        Parameters
        ----------
        n : int
            Number of features
        m : int
            Target number of samples

        Returns
        -------
        Z_prime : np.ndarray
            Sampled coalitions, shape (n_samples, n)
        weights : np.ndarray
            Regression weights for each sample
        """
        # Find oversampling parameter c via binary search
        c = self._find_c_binary_search(n, m)

        if self.verbose:
            print(f"Using oversampling parameter c={c:.4f}")

        Z_prime = []
        weights = []

        # Sample for each subset size
        for s in range(1, (n // 2) + 1):
            # Compute sampling probability
            leverage_score = self._compute_leverage_score(s, n)
            prob = min(1.0, 2 * c * leverage_score)

            # Number of subsets of this size
            n_subsets = int(comb(n, s, exact=True))

            # Special handling for middle size when n is even
            is_middle = (n % 2 == 0) and (s == n // 2)
            if is_middle:
                n_subsets = n_subsets // 2

            # Sample number of coalitions to draw (Binomial)
            n_samples_size = np.random.binomial(n_subsets, prob)

            if n_samples_size == 0:
                continue

            # Sample specific coalitions uniformly
            sampled_indices = np.random.choice(
                n_subsets, size=n_samples_size, replace=False
            )

            for idx in sampled_indices:
                # Generate coalition from index
                coalition = self._index_to_coalition(n, s, idx, is_middle)
                complement = 1 - coalition

                # Add pair if using paired sampling
                Z_prime.append(coalition)
                Z_prime.append(complement)

                # Compute weight: w(s) / P(sample)
                w = self._compute_kernel_weight(s, n) / prob
                weights.extend([w, w])

        return np.array(Z_prime), np.array(weights)

    def _simple_leverage_sample(
        self,
        n: int,
        m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple leverage score sampling with replacement.

        Parameters
        ----------
        n : int
            Number of features
        m : int
            Target number of samples

        Returns
        -------
        Z_prime : np.ndarray
            Sampled coalitions
        weights : np.ndarray
            Regression weights
        """
        n_pairs = (m - 2) // 2 if self.paired_sampling else m - 2

        Z_prime = []
        weights = []

        for _ in range(n_pairs):
            # Sample subset size uniformly from {1, ..., n-1}
            s = np.random.randint(1, n)

            # Sample coalition of size s uniformly
            coalition = np.zeros(n)
            indices = np.random.choice(n, size=s, replace=False)
            coalition[indices] = 1

            if self.paired_sampling:
                complement = 1 - coalition
                Z_prime.extend([coalition, complement])

                # Weight by leverage score
                w = self._compute_kernel_weight(s, n) * (n - 1)
                weights.extend([w, w])
            else:
                Z_prime.append(coalition)
                w = self._compute_kernel_weight(s, n) * (n - 1)
                weights.append(w)

        return np.array(Z_prime), np.array(weights)

    def _find_c_binary_search(self, n: int, m: int) -> float:
        """
        Find oversampling parameter c via binary search.

        Solves: m - 2 = Σ_{s=1}^{⌊n/2⌋} min(C(n,s), 2c * ℓ_s)

        Parameters
        ----------
        n : int
            Number of features
        m : int
            Target number of samples

        Returns
        -------
        float
            Oversampling parameter c
        """
        def expected_samples(c):
            total = 0
            for s in range(1, (n // 2) + 1):
                n_subsets = comb(n, s, exact=False)
                leverage_score = self._compute_leverage_score(s, n)
                total += min(n_subsets, 2 * c * leverage_score * n_subsets)
            return total

        # Binary search for c
        c_low, c_high = 0.01, 100.0
        target = m - 2

        for _ in range(50):  # Max iterations
            c_mid = (c_low + c_high) / 2
            samples = expected_samples(c_mid)

            if abs(samples - target) < 1:
                return c_mid

            if samples < target:
                c_low = c_mid
            else:
                c_high = c_mid

        return c_mid

    def _index_to_coalition(
        self,
        n: int,
        s: int,
        index: int,
        is_middle: bool = False
    ) -> np.ndarray:
        """
        Convert index to coalition (Algorithm 3: Combo from paper).

        Generates the index-th combination of n items with size s
        in lexicographic order.

        Parameters
        ----------
        n : int
            Total number of items
        s : int
            Subset size
        index : int
            Index of the combination
        is_middle : bool
            Special handling for middle size

        Returns
        -------
        np.ndarray
            Binary coalition vector
        """
        if is_middle:
            # Special case: partition middle size by setting last element to 1
            coalition = self._index_to_coalition(n - 1, s - 1, index, False)
            return np.append(coalition, 1)

        coalition = np.zeros(n, dtype=int)
        k = s
        i = index
        start = 0

        for _ in range(s):
            for j in range(start, n):
                # Count combinations possible if we add j
                count = int(comb(n - j - 1, k - 1, exact=True))

                if i < count:
                    coalition[j] = 1
                    k -= 1
                    start = j + 1
                    break
                else:
                    i -= count

        return coalition

    def _solve_regression(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        n: int
    ) -> np.ndarray:
        """
        Solve weighted constrained least squares regression.

        Minimizes: ||W^(1/2) Z x - W^(1/2) b||²
        Subject to: <x, 1> = v([n]) - v(∅)

        Parameters
        ----------
        Z : np.ndarray
            Coalition matrix, shape (m, n)
        y : np.ndarray
            Target values (already centered)
        weights : np.ndarray
            Sample weights
        n : int
            Number of features

        Returns
        -------
        np.ndarray
            Shapley values
        """
        # Adjust target: b = y - (v([n]) - v(∅))/n * Z @ 1
        v_diff = self._v_full - self._v_empty
        b = y - (v_diff / n) * Z.sum(axis=1)

        # Weight matrix (diagonal)
        W_sqrt = np.sqrt(weights)

        # Weighted regression: W^(1/2) Z x = W^(1/2) b
        Z_weighted = W_sqrt[:, None] * Z
        b_weighted = W_sqrt * b

        # Project onto constraint subspace: <x, 1> = 0
        # Using QR decomposition for numerical stability
        ones = np.ones(n) / np.sqrt(n)
        Q, _ = np.linalg.qr(ones.reshape(-1, 1))
        P = np.eye(n) - Q @ Q.T  # Projection matrix

        # Solve: W^(1/2) Z P x_perp = W^(1/2) b
        Z_proj = Z_weighted @ P

        try:
            # Use least squares solver
            x_perp, _, _, _ = np.linalg.lstsq(Z_proj, b_weighted, rcond=None)
            phi_perp = P @ x_perp
        except np.linalg.LinAlgError:
            warnings.warn("Regression failed, using pseudo-inverse")
            phi_perp = P @ np.linalg.pinv(Z_proj) @ b_weighted

        # Add back the constraint: phi = phi_perp + (v([n]) - v(∅))/n * 1
        phi = phi_perp + (v_diff / n) * np.ones(n)

        return phi

    def get_feature_importance(self) -> np.ndarray:
        """
        Get absolute Shapley values as feature importance scores.

        Returns
        -------
        np.ndarray
            Absolute Shapley values
        """
        if self.shapley_values_ is None:
            raise ValueError("Must call fit() before getting importance")

        return np.abs(self.shapley_values_)
