"""Improved QR-CS explainer with adaptive sparsity detection."""

import numpy as np
from typing import Optional, Callable, Dict
from .qrcs import QRCSExplainer


# Constants for adaptive sparsity detection
DEFAULT_SPARSITY_SAMPLE_SIZE = 50  # Number of samples to check sparsity
DEFAULT_SPARSITY_THRESHOLD = 0.01  # Threshold for considering coefficients as zero


class ImprovedQRCSExplainer(QRCSExplainer):
    """QR-CS explainer with adaptive sparsity detection.

    Automatically detects if the signal is sparse enough for compressed sensing
    and falls back to robust estimation if not.
    """

    def __init__(
        self,
        characteristic_function=None,
        n_measurements: Optional[int] = None,
        tolerance: float = 5e-5,
        sparsity_threshold: float = 0.8,
        auto_adapt: bool = True,
        verbose: bool = False
    ):
        """Initialize improved QR-CS explainer.

        Args:
            characteristic_function: Function to compute coalition values
            n_measurements: Number of compressed measurements
            tolerance: L1 optimization tolerance
            sparsity_threshold: Minimum sparsity ratio to use CS (default 0.8 = 80% sparse)
            auto_adapt: Automatically switch to fast fallback if not sparse
            verbose: Whether to print progress
        """
        super().__init__(
            characteristic_function=characteristic_function,
            n_measurements=n_measurements,
            tolerance=tolerance,
            use_fast_fallback=False,  # Start with full CS
            verbose=verbose
        )
        self.sparsity_threshold = sparsity_threshold
        self.auto_adapt = auto_adapt
        self._sparsity_detected = None

    def _check_sparsity(self, utility_func: Callable, sample_size: int = DEFAULT_SPARSITY_SAMPLE_SIZE) -> float:
        """Check if marginal contributions are sparse in DCT domain.

        Args:
            utility_func: Utility function for coalition values
            sample_size: Number of coalitions to sample

        Returns:
            Sparsity ratio (fraction of near-zero coefficients)
        """
        # Sample marginal contributions for first player
        player = 0
        sample_size = min(sample_size, self.m)
        sample_indices = np.random.choice(self.m, sample_size, replace=False)

        marginal_contribs = []
        for idx in sample_indices:
            coalition = self._index_to_coalition(idx, player)
            v_with = utility_func(coalition | {player})
            v_without = utility_func(coalition) if coalition else 0
            marginal_contribs.append(v_with - v_without)

        u_sample = np.array(marginal_contribs)

        # Create small DCT basis for sample
        Psi_sample = self._get_dct_basis_for_size(sample_size)

        # Transform to DCT domain
        s_sample = Psi_sample.T @ u_sample

        # Count near-zero coefficients
        threshold = DEFAULT_SPARSITY_THRESHOLD * np.max(np.abs(s_sample)) if len(s_sample) > 0 else DEFAULT_SPARSITY_THRESHOLD
        sparsity = np.sum(np.abs(s_sample) < threshold) / len(s_sample)

        return sparsity

    def _get_dct_basis_for_size(self, size: int) -> np.ndarray:
        """Generate DCT basis for given size."""
        if size > 1000:
            return np.eye(size)

        Psi = np.zeros((size, size))
        for k in range(size):
            for n in range(size):
                if k == 0:
                    Psi[n, k] = np.sqrt(1/size)
                else:
                    Psi[n, k] = np.sqrt(2/size) * np.cos(np.pi * k * (n + 0.5) / size)
        return Psi

    def _compute_shapley(self, utility_func: Callable) -> Dict[int, float]:
        """Compute Shapley values with adaptive method selection.

        Checks sparsity first and adapts method accordingly.
        """
        if self.auto_adapt and self._sparsity_detected is None:
            # Check sparsity on first run
            sparsity = self._check_sparsity(utility_func)
            self._sparsity_detected = sparsity

            if self.verbose:
                print(f"Detected sparsity: {100*sparsity:.1f}%")

            if sparsity < self.sparsity_threshold:
                if self.verbose:
                    print(f"Signal not sparse enough ({100*sparsity:.1f}% < {100*self.sparsity_threshold:.1f}%)")
                    print("Switching to robust mean estimation (fast_fallback=True)")
                self.use_fast_fallback = True
            else:
                if self.verbose:
                    print(f"Signal is sparse ({100*sparsity:.1f}%), using full CS reconstruction")

        # Call parent implementation with adapted settings
        return super()._compute_shapley(utility_func)

    def get_sparsity_info(self) -> Dict[str, float]:
        """Get information about detected sparsity.

        Returns:
            Dict with sparsity information
        """
        return {
            'detected_sparsity': self._sparsity_detected,
            'threshold': self.sparsity_threshold,
            'using_cs': not self.use_fast_fallback,
            'method': 'Compressed Sensing' if not self.use_fast_fallback else 'Robust Mean'
        }