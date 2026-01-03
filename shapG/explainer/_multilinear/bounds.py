"""Error bounds computation for multilinear extension estimator."""

from dataclasses import dataclass
from math import sqrt
from scipy.stats import norm


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


class ErrorBoundComputer:
    """
    Computes rigorous error bounds for the LEM estimator.

    Error Decomposition:
        |phi_hat_i - phi_i| <= eps_quad + eps_mc

    where:
        eps_quad: Quadrature error from Simpson's rule
        eps_mc: Monte Carlo sampling error
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

        Simpson's rule error: |integral(f) - S_K(f)| <= (b-a)^5 / 180 * max|f^(4)(x)| / K^4
        """
        h = 1.0 / (self.K - 1)
        return (h**4) * fourth_derivative_bound / 180

    def sampling_error_bound(
        self, variance_estimate: float, leverage_efficiency: float = 1.0
    ) -> float:
        """
        Compute Monte Carlo sampling error bound.

        Using CLT: eps_mc <= z_{alpha/2} * sigma / sqrt(m * eta * K)
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
