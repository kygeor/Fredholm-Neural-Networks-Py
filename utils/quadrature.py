"""
Numerical quadrature helpers for the Laplace PDE solver.

These routines compute weakly-singular integrals that arise in the
double-layer potential representation used by the Laplace BIE solver.
SciPy's ``dblquad`` / ``quad`` are used because they achieve ~1e-15
accuracy at singularities, far better than uniform tensor-product rules.

The integrals are embarrassingly parallel over output grid points and are
evaluated with :class:`concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import quad

_TOL = 1e-8  # proximity threshold for the weak singularity treatment


# ---------------------------------------------------------------------------
# Scalar integrand building blocks (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _fundamental_scalar(r2: float, theta: float, r_out: float, theta_out: float) -> float:
    """
    Fundamental solution of the 2-D Laplace equation (Green's function),
    scaled by the area element r_2 for integration in polar coordinates.

        Φ(x, y) = (1/2π) · ln|x − y|

    Evaluated as the integrand contribution at integration point
    (r_2, θ) for output point (r_out, θ_out).
    """
    dx = r_out * np.cos(theta_out) - r2 * np.cos(theta)
    dy = r_out * np.sin(theta_out) - r2 * np.sin(theta)
    return (1.0 / (2.0 * np.pi)) * 0.5 * np.log(dx**2 + dy**2) * r2


def _source_scalar(r2: float, theta: float) -> float:
    """
    Source term f(r, θ) for the Poisson equation.  Zero for the pure
    Laplace case (Δu = 0).
    """
    return 0.0


def _integrand_scalar(theta: float, r2: float, r_out: float, theta_out: float) -> float:
    return _source_scalar(r2, theta) * _fundamental_scalar(r2, theta, r_out, theta_out)


def _inner_integral(r2: float, r_out: float, theta_out: float) -> float:
    """Integrate over θ for fixed r_2, handling the weak singularity."""
    f = lambda th: _integrand_scalar(th, r2, r_out, theta_out)
    if abs(r2 - r_out) < _TOL:
        result, _ = quad(f, 0.0, 2.0 * np.pi, points=[theta_out])
    else:
        result, _ = quad(f, 0.0, 2.0 * np.pi)
    return result


def _full_integral(r_out: float, theta_out: float) -> float:
    """Full double integral over (r_2, θ) for a single output point."""
    f_r2 = lambda r2: _inner_integral(r2, r_out, theta_out)
    result, _ = quad(f_r2, 0.0, 1.0, points=[r_out])
    return result


def _integrate_single(args: tuple) -> float:
    r_out, theta_out = args
    return _full_integral(r_out, theta_out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parallel_integrate_meshgrid(
    r_out_values: np.ndarray,
    theta_out_values: np.ndarray,
    max_workers: int | None = None,
) -> np.ndarray:
    """
    Compute the fundamental-solution / source-term integral on a meshgrid.

    For each output point (r_out_i, θ_out_j) evaluates

        ∬ f(r, θ) Φ(x_out, (r, θ)) r dr dθ

    over the unit disc in polar coordinates.  The integrals are independent
    and are evaluated in parallel.

    Parameters
    ----------
    r_out_values : np.ndarray, shape (Nr,)
        Radial output coordinates.
    theta_out_values : np.ndarray, shape (Nθ,)
        Angular output coordinates (radians).
    max_workers : int or None, optional
        Number of worker processes.  ``None`` uses ``os.cpu_count()``.

    Returns
    -------
    np.ndarray, shape (Nr, Nθ)
        Integral values on the meshgrid.
    """
    R_out, Theta_out = np.meshgrid(r_out_values, theta_out_values, indexing="ij")
    pairs = list(zip(R_out.ravel(), Theta_out.ravel()))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_integrate_single, pairs))

    return np.array(results, dtype=np.float64).reshape(R_out.shape)
