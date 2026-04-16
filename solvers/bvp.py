"""
Solver for second-order boundary value problem (BVP) ODEs via reduction to
a Fredholm integral equation of the second kind.

Given the BVP

    y''(x) + p(x) y(x) = q(x),   x ∈ [0, 1],
    y(0) = α,   y(1) = β,

the substitution u(x) = y''(x) yields the linear FIE

    u(x) = f(x) + ∫_0^1 K(x, t) u(t) dt,

with

    f(x)    = q(x) − α p(x) − (β − α) x p(x),

    K(x, t) = p(x) · G(x, t),   where G is the Green's function

              ⎧ t(1 − x),   0 ≤ t ≤ x
    G(x, t) = ⎨
              ⎩ x(1 − t),   x < t ≤ 1.

Once u(x) is found the solution y(x) is recovered via

    y(x) = (q(x) − u(x)) / p(x).

Reference
---------
Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
Fredholm neural networks. SIAM Journal on Scientific Computing, 47(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from fredholm_nn.solvers.fie import solve_linear_fie, FIESolution


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BVPSolution:
    """
    Container for the output of :func:`solve_bvp_ode`.

    Attributes
    ----------
    x : np.ndarray
        Query points.
    y : np.ndarray
        Predicted ODE solution y(x).
    u : np.ndarray
        Auxiliary variable u(x) = y''(x) from the intermediate FIE solve.
    fie_solution : FIESolution
        The underlying FIE solution object (for diagnostics).
    """

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    fie_solution: FIESolution = field(repr=False)

    def error(self, exact: Callable) -> np.ndarray:
        """Pointwise absolute error |y(x) − y_exact(x)|."""
        return np.abs(self.y - np.asarray(exact(self.x), dtype=np.float64))

    def mse(self, exact: Callable) -> float:
        """Mean-squared error against an exact solution."""
        return float(np.mean(self.error(exact) ** 2))


# ---------------------------------------------------------------------------
# BVP solver
# ---------------------------------------------------------------------------

def solve_bvp_ode(
    p_func: Callable,
    q_func: Callable,
    alpha: float,
    beta: float,
    *,
    domain: tuple[float, float] = (0.0, 1.0),
    n_grid: int = 1000,
    n_iterations: int = 15,
    predict_at: np.ndarray | None = None,
    km_constant: float = 1.0,
) -> BVPSolution:
    """
    Solve a second-order BVP ODE via reduction to a Fredholm IE.

    The ODE must be of the form

        y''(x) + p(x) y(x) = q(x),   x ∈ [a, b],
        y(a) = α,   y(b) = β.

    Parameters
    ----------
    p_func : callable
        p(x) — coefficient of y(x).  Must accept a NumPy array.
    q_func : callable
        q(x) — right-hand side.  Must accept a NumPy array.
    alpha, beta : float
        Boundary values y(a) = α, y(b) = β.
    domain : (a, b), optional
        Spatial domain (default ``(0.0, 1.0)``).
    n_grid : int, optional
        Number of quadrature points (default 1000).
    n_iterations : int, optional
        FNN hidden layers K (default 15).
    predict_at : np.ndarray, optional
        Points at which to evaluate y(x).  Defaults to the integration grid.
    km_constant : float, optional
        KM relaxation parameter (default 1.0).

    Returns
    -------
    BVPSolution
        Container with ``.x``, ``.y``, ``.u``, and ``.fie_solution``.

    Examples
    --------
    Solve  y''(x) + (3p/(p+x²)²) y(x) = 0  on [0,1] with y(0)=0, y(1)=1/√(p+1):

    >>> import numpy as np
    >>> p = 3.2
    >>> sol = solve_bvp_ode(
    ...     p_func=lambda x: 3*p / (p + x**2)**2,
    ...     q_func=lambda x: np.zeros_like(x),
    ...     alpha=0.0,
    ...     beta=1.0 / np.sqrt(p + 1.0),
    ...     n_grid=1000,
    ...     n_iterations=10,
    ...     predict_at=np.linspace(0, 1, 200),
    ... )
    >>> sol.y.shape
    (200,)
    """
    a, b = domain

    # ------------------------------------------------------------------
    # Build the FIE kernel K(x, t) and free term f(x)
    # ------------------------------------------------------------------

    def _kernel(x, t):
        """
        K(x, t) = p(x) · G(x, t).

        Accepts broadcastable (x, t) pairs (e.g. shapes (N,1) and (1,M)).
        """
        # Green's function for the second-order BVP on [a, b]
        # Generalised for arbitrary [a, b] domain:
        #   G(x, t) = (t−a)(b−x)/(b−a)  for t ≤ x
        #           = (x−a)(b−t)/(b−a)  for t > x
        L = b - a
        green = np.where(
            t <= x,
            (t - a) * (b - x) / L,
            (x - a) * (b - t) / L,
        )
        return p_func(x) * green

    def _additive(x):
        """
        f(x) = q(x) − α p(x) − (β − α)·(x − a)/(b − a)·p(x).

        Handles both np.ndarray and torch.Tensor inputs.
        """
        is_tensor = isinstance(x, torch.Tensor)
        x_np = x.numpy() if is_tensor else np.asarray(x, dtype=np.float64)

        scale = (beta - alpha) / (b - a)
        result = (
            q_func(x_np)
            - alpha * p_func(x_np)
            - scale * (x_np - a) * p_func(x_np)
        ).astype(np.float32)

        return torch.tensor(result, dtype=torch.float32) if is_tensor else result

    # ------------------------------------------------------------------
    # Solve the FIE to obtain u(x) = y''(x)
    # ------------------------------------------------------------------

    if predict_at is None:
        predict_at = np.linspace(a, b, n_grid)

    fie_sol = solve_linear_fie(
        kernel=_kernel,
        additive=_additive,
        domain=domain,
        n_grid=n_grid,
        n_iterations=n_iterations,
        predict_at=predict_at,
        km_constant=km_constant,
    )

    u = fie_sol.f  # u(x) = y''(x)
    x = fie_sol.x

    # ------------------------------------------------------------------
    # Recover y(x) = (q(x) − u(x)) / p(x)
    # ------------------------------------------------------------------
    p_vals = np.asarray(p_func(x), dtype=np.float64)
    q_vals = np.asarray(q_func(x), dtype=np.float64)

    # Guard against division by zero when p(x) vanishes
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(p_vals != 0.0, (q_vals - u) / p_vals, np.nan)

    return BVPSolution(x=x, y=y, u=u, fie_solution=fie_sol)
