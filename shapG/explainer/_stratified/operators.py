"""Implicit DCT operator for compressed sensing reconstruction."""

import numpy as np
from scipy.fft import dct, idct


class ImplicitDCTOperator:
    """
    Implicit operator for A @ s = B @ IDCT(s) without storing full matrix.

    This enables compressed sensing reconstruction for large coalition spaces
    where storing a full 2^(n-1) x 2^(n-1) DCT matrix is infeasible.

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
        u_full = idct(s, norm="ortho")
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
        return dct(u_full, norm="ortho")

    def __matmul__(self, s: np.ndarray) -> np.ndarray:
        """Matrix multiplication operator."""
        return self.matvec(s)
