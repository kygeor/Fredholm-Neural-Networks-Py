"""
High-level solver for the 2-D Laplace PDE on the unit disc.

Solves

    Δu(x) = 0,    x ∈ Ω  (unit disc),
    u(x)  = f(x), x ∈ ∂Ω (unit circle),

via the Fredholm Neural Network framework: the problem is reformulated as a
Boundary Integral Equation (BIE) and solved with :class:`LaplaceFredholmNN`.

The solution is evaluated on a polar meshgrid (r_values, theta_values) inside
the disc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from fredholm_nn.models.linear import FredholmNN
from fredholm_nn.models.pde import LaplaceFredholmNN, diff_potentials_limit, potential_boundary
from fredholm_nn.utils.grid import make_uniform_grid, make_grid_dictionary


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LaplaceSolution:
    """
    Container for the output of :func:`solve_laplace`.

    Attributes
    ----------
    r : np.ndarray, shape (Nr,)
        Radial grid.
    theta : np.ndarray, shape (Nθ,)
        Angular grid (radians).
    u : np.ndarray, shape (Nr, Nθ)
        Predicted solution u(r, θ) on the polar meshgrid.
    model : LaplaceFredholmNN
        Underlying model (retained for inspection).
    """

    r: np.ndarray
    theta: np.ndarray
    u: np.ndarray
    model: LaplaceFredholmNN = field(repr=False)

    def error(self, exact: Callable) -> np.ndarray:
        """
        Pointwise absolute error |u(r,θ) − u_exact(r,θ)|.

        Parameters
        ----------
        exact : callable
            Signature ``exact(r, theta) -> np.ndarray`` with r, theta being
            1-D arrays; the function is called with 2-D meshgrid arrays.
        """
        R, Th = np.meshgrid(self.r, self.theta, indexing="ij")
        return np.abs(self.u - np.asarray(exact(R, Th), dtype=np.float64))

    def mse(self, exact: Callable) -> float:
        """Mean-squared error against an exact solution."""
        return float(np.mean(self.error(exact) ** 2))


# ---------------------------------------------------------------------------
# Laplace solver
# ---------------------------------------------------------------------------

def solve_laplace(
    boundary_fn: Callable,
    *,
    n_boundary: int = 500,
    n_iterations: int = 30,
    km_constant: float = 0.3,
    r_values: np.ndarray | None = None,
    theta_values: np.ndarray | None = None,
    diff_potentials_fn: Callable = diff_potentials_limit,
    potential_boundary_fn: Callable = potential_boundary,
) -> LaplaceSolution:
    """
    Solve the 2-D Laplace equation on the unit disc.

    Parameters
    ----------
    boundary_fn : callable
        Dirichlet boundary condition f(θ) on the unit circle.
        Must accept a NumPy array of angles (radians) and return values.
    n_boundary : int, optional
        Number of quadrature points on the boundary ∂Ω (default 500).
    n_iterations : int, optional
        FNN hidden layers K for the inner BIE solve (default 30).
    km_constant : float, optional
        KM relaxation parameter κ ∈ (0, 1].  Non-expansive BIE kernels
        typically require κ < 1 (default 0.3).
    r_values : np.ndarray, optional
        Radial query points in [0, 1].  Defaults to
        ``np.linspace(0, 1, 100)``.
    theta_values : np.ndarray, optional
        Angular query points in [0, 2π).  Defaults to the boundary grid.
    diff_potentials_fn : callable, optional
        Override for the difference-potential kernel.
    potential_boundary_fn : callable, optional
        Override for the boundary-potential kernel.

    Returns
    -------
    LaplaceSolution

    Examples
    --------
    Solve Δu = 0 on the unit disc with u = 1 + cos(2θ) on ∂Ω:

    >>> import numpy as np
    >>> sol = solve_laplace(
    ...     boundary_fn=lambda theta: 1.0 + np.cos(2 * theta),
    ...     n_boundary=500,
    ...     n_iterations=30,
    ...     km_constant=0.3,
    ... )
    >>> sol.u.shape
    (100, 500)
    """
    # --- Angular integration grid on ∂Ω ---
    phi_0, phi_n = 0.0, 2.0 * np.pi
    dphi = (phi_n - phi_0) / n_boundary
    phi_grid = np.arange(phi_0, phi_n, dphi)   # half-open [0, 2π)

    grid_dict = make_grid_dictionary(phi_grid, n_iterations)

    # BIE kernel: K(φ_in, φ_out) = −2 · (1/4π) on the unit circle
    # (derived from the double-layer potential with constant weight 1/2π)
    def _bie_kernel(phi_in, phi_out):
        weight = (1.0 / (4.0 * np.pi)) * np.ones_like(phi_in - phi_out)
        return -2.0 * weight

    # Additive (free) term: 2 f(φ)
    def _bie_additive(phi):
        is_tensor = isinstance(phi, torch.Tensor)
        phi_np = phi.numpy() if is_tensor else np.asarray(phi)
        result = (2.0 * boundary_fn(phi_np)).astype(np.float32)
        return torch.tensor(result, dtype=torch.float32) if is_tensor else result

    # --- Build the inner FredholmNN for the BIE ---
    bie_model = FredholmNN(
        grid_dictionary=grid_dict,
        kernel=_bie_kernel,
        additive=_bie_additive,
        grid_step=dphi,
        n_iterations=n_iterations,
        km_constant=km_constant,
    )

    # --- Full Laplace model ---
    laplace_model = LaplaceFredholmNN(
        fredholm_model=bie_model,
        diff_potentials_fn=diff_potentials_fn,
        potential_boundary_fn=potential_boundary_fn,
    )

    # --- Output grids ---
    if r_values is None:
        r_values = np.linspace(0.0, 1.0, 100)
    if theta_values is None:
        theta_values = phi_grid

    phi_input = torch.tensor(phi_grid, dtype=torch.float32)

    with torch.no_grad():
        output = laplace_model(
            phi_input=phi_input,
            r_out=r_values,
            theta_out=theta_values,
            phi_grid=phi_grid,
            grid_step=dphi,
        )

    # output shape: (Nr, Nθ, 1) → (Nr, Nθ)
    u = output.numpy()[:, :, 0].astype(np.float64)

    return LaplaceSolution(r=r_values, theta=theta_values, u=u, model=laplace_model)
