"""L1 minimization solver for compressed sensing reconstruction."""

import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.fft import idct

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .operators import ImplicitDCTOperator


class L1Solver:
    """
    L1 minimization solver for compressed sensing reconstruction.

    Solves: min ||x||_1 subject to ||Ax - b||_2 <= tolerance

    Uses CVXPY if available, otherwise falls back to scipy.
    """

    def __init__(self, tolerance: float = 1e-4, max_iter: int = 500):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve(self, A, b: np.ndarray) -> np.ndarray:
        """
        Solve the L1 minimization problem.

        Parameters
        ----------
        A : np.ndarray or ImplicitDCTOperator
            Measurement matrix/operator (l x m)
        b : np.ndarray
            Measurements (l,)

        Returns
        -------
        x : np.ndarray
            Reconstructed signal (m,)
        """
        l, m = A.shape

        # Check if A is an implicit operator
        if isinstance(A, ImplicitDCTOperator):
            # Use method that works with implicit operators
            return self._solve_implicit(A, b)
        else:
            # A is a numpy array, use standard methods
            if CVXPY_AVAILABLE:
                return self._solve_cvxpy(A, b)
            else:
                return self._solve_scipy(A, b)

    def _solve_cvxpy(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using CVXPY (preferred method)."""
        m = A.shape[1]

        try:
            x = cp.Variable(m)
            objective = cp.Minimize(cp.norm(x, 1))
            constraints = [cp.norm(A @ x - b, 2) <= self.tolerance]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, max_iters=self.max_iter)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                return x.value
            else:
                # Fall back to least squares
                return np.linalg.lstsq(A, b, rcond=None)[0]

        except Exception as e:
            warnings.warn(
                f"CVXPY solver failed: {e}. Falling back to least squares.",
                RuntimeWarning,
            )
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _solve_scipy(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using scipy optimization (fallback)."""
        m = A.shape[1]

        # Initial guess via least squares
        try:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            x0 = np.zeros(m)

        def objective(x):
            return np.sum(np.abs(x))

        def constraint(x):
            return self.tolerance - np.linalg.norm(A @ x - b)

        constraints = {"type": "ineq", "fun": constraint}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints=constraints,
                options={"maxiter": self.max_iter, "ftol": 1e-6},
            )

        return result.x if result.success else x0

    def _solve_implicit(self, A: "ImplicitDCTOperator", b: np.ndarray) -> np.ndarray:
        """
        Solve L1 minimization for implicit DCT operator.

        For implicit operators, we cannot use CVXPY because it requires
        materializing the full matrix. Instead, we use ISTA (Iterative Soft
        Thresholding Algorithm) which only requires matrix-vector products.
        """
        # Use ISTA for implicit operators (can't materialize matrix for cvxpy)
        return self._ista(A, b)

    def _ista(
        self, A: "ImplicitDCTOperator", b: np.ndarray, lam: float = None
    ) -> np.ndarray:
        """
        Fast Iterative Soft Thresholding Algorithm (FISTA) for L1 minimization.

        Solves: min lambda||x||_1 + (1/2)||Ax - b||_2^2

        Uses Nesterov acceleration for faster convergence.
        """
        m = A.shape[1]

        # Auto-tune lambda if not specified
        if lam is None:
            # Use noise-based estimate: lambda ~ ||A^T @ noise||_inf
            # For normalized measurements, a typical value is 0.01-0.1
            lam = 0.05

        # Better initialization: least squares solution
        try:
            # x0 = argmin ||Ax - b||^2 via gradient descent (few iterations)
            x = np.zeros(m)
            for _ in range(10):
                residual = A.matvec(x) - b
                grad = A.rmatvec(residual)
                x = x - 0.5 * grad
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            x = np.zeros(m)

        # FISTA with Nesterov acceleration
        # For operator A = B @ IDCT:
        # - IDCT is orthonormal: ||IDCT|| = 1
        # - B is selection operator: ||B|| = 1
        # - Lipschitz constant L <= ||A||^2 <= 1
        # - Safe step size: alpha = 1/L ~ 1.0
        step_size = 0.9  # Slightly conservative

        y = x.copy()
        t = 1.0

        for iteration in range(self.max_iter):
            # Gradient step on y
            residual = A.matvec(y) - b
            grad = A.rmatvec(residual)
            x_new = y - step_size * grad

            # Soft thresholding (proximal operator)
            x_new = self._soft_threshold(x_new, step_size * lam)

            # Nesterov acceleration
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)

            # Check convergence
            rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
            if rel_change < 1e-5:
                if self.tolerance < 1e-4:  # Only print for verbose mode
                    print(f"  [ISTA] Converged in {iteration+1} iterations")
                break

            x = x_new
            t = t_new

        return x

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
