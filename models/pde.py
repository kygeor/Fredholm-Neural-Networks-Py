"""
Fredholm Neural Network architecture for the 2-D Laplace PDE on the unit disc.

The Laplace equation

    Δu(x) = 0,   x ∈ Ω,
    u(x)  = f(x), x ∈ ∂Ω,

is reformulated as a Boundary Integral Equation (BIE) via the double-layer
potential.  The density β on the boundary satisfies

    β(x*) = 2f(x*) − 2 ∫_{∂Ω} β(y) ∂Φ/∂n_y (x*, y) dσ_y,

which is a linear Fredholm IE of the second kind on ∂Ω.

A two-component FNN is used:
  1.  ``FredholmNN`` solves the BIE on a discretised angular grid to obtain β.
  2.  ``PotentialFredholmNN`` maps β into the interior via the double-layer
      potential representation.

Reference
---------
Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
Fredholm neural networks for forward and inverse problems in elliptic PDEs.
arXiv:2507.06038.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable

from fredholm_nn.models.linear import FredholmNN


# ---------------------------------------------------------------------------
# Potential kernels
# ---------------------------------------------------------------------------

def diff_potentials_limit(
    phi_grid: np.ndarray | torch.Tensor,
    r_out: np.ndarray | torch.Tensor,
    theta_out: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """
    Difference potential kernel for the interior evaluation.

    Computes

        D_Φ(x, y) = ∂Φ/∂n_y(x, y) − ∂Φ/∂n_y(x*, y)

    where x = (r_out, θ_out) is an interior point and x* = (1, θ_out)
    is the corresponding boundary point.  At x* itself (r_out = 1) the
    value is set to 0.5 (the boundary limit).

    Parameters
    ----------
    phi_grid : array-like, shape (N,)
        Angular integration variable φ on the boundary ∂Ω.
    r_out : array-like, shape (Nr,)
        Radial output coordinates.
    theta_out : array-like, shape (Nθ,)
        Angular output coordinates.

    Returns
    -------
    torch.Tensor, shape (Nr, Nθ, N)
    """
    phi = torch.tensor(phi_grid, dtype=torch.float32) if not isinstance(phi_grid, torch.Tensor) else phi_grid
    r   = torch.atleast_1d(torch.tensor(r_out,   dtype=torch.float32))
    th  = torch.atleast_1d(torch.tensor(theta_out, dtype=torch.float32))

    # Create meshgrid: axes → (r, θ, φ)
    r_g, th_g, ph_g = torch.meshgrid(r, th, phi, indexing="ij")

    boundary_mask = (r_g == 1.0)

    # ∂Φ/∂n_y(x, y) = (n_y · (y − x)) / (2π |y − x|²)
    # On the unit circle, n_y = y = (cos φ, sin φ), so:
    num = (torch.cos(ph_g) * (torch.cos(ph_g) - r_g * torch.cos(th_g))
           + torch.sin(ph_g) * (torch.sin(ph_g) - r_g * torch.sin(th_g)))
    den = ((torch.cos(ph_g) - r_g * torch.cos(th_g)) ** 2
           + (torch.sin(ph_g) - r_g * torch.sin(th_g)) ** 2)

    kernel = num / den
    kernel[boundary_mask] = 0.5  # boundary limit

    # Difference potential: D_Φ = (1/2π)(k − 0.5)
    return (1.0 / (2.0 * np.pi)) * (kernel - 0.5)


def potential_boundary(
    phi_grid: np.ndarray | torch.Tensor,
    r_out: np.ndarray | torch.Tensor,
    theta_out: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """
    Constant boundary potential kernel (1/4π) used in the bias term.

    Returns
    -------
    torch.Tensor, shape (Nr, Nθ, N)
        Uniform tensor filled with 1/(4π).
    """
    phi = torch.tensor(phi_grid, dtype=torch.float32) if not isinstance(phi_grid, torch.Tensor) else phi_grid
    r   = torch.atleast_1d(torch.tensor(r_out,   dtype=torch.float32))
    th  = torch.atleast_1d(torch.tensor(theta_out, dtype=torch.float32))

    r_g, th_g, ph_g = torch.meshgrid(r, th, phi, indexing="ij")
    return torch.full(r_g.shape, 1.0 / (4.0 * np.pi), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Laplace FNN model
# ---------------------------------------------------------------------------

class LaplaceFredholmNN(nn.Module):
    """
    Two-component Fredholm Neural Network for the 2-D Laplace PDE.

    Component 1: ``FredholmNN`` solves the BIE on the boundary (∂Ω = unit
    circle) and returns the density β(φ).

    Component 2: maps β into the interior via the double-layer potential,
    adding the difference-potential correction and the boundary-potential
    bias to obtain u(r, θ) at arbitrary interior points.

    Parameters
    ----------
    fredholm_model : FredholmNN
        Pre-constructed FNN for the boundary BIE.
    diff_potentials_fn : callable, optional
        Kernel for the difference potential (default :func:`diff_potentials_limit`).
    potential_boundary_fn : callable, optional
        Kernel for the boundary potential bias (default :func:`potential_boundary`).
    """

    def __init__(
        self,
        fredholm_model: FredholmNN,
        diff_potentials_fn: Callable = diff_potentials_limit,
        potential_boundary_fn: Callable = potential_boundary,
    ):
        super().__init__()
        self.fredholm_model = fredholm_model
        self.diff_potentials_fn = diff_potentials_fn
        self.potential_boundary_fn = potential_boundary_fn

    def forward(
        self,
        phi_input: torch.Tensor,
        r_out: np.ndarray | torch.Tensor,
        theta_out: np.ndarray | torch.Tensor,
        phi_grid: np.ndarray | torch.Tensor,
        grid_step: float,
    ) -> torch.Tensor:
        """
        Evaluate u(r, θ) at all (r_out_i, θ_out_j) pairs.

        Parameters
        ----------
        phi_input : torch.Tensor, shape (N,)
            Angular integration points on ∂Ω (passed to the inner FNN).
        r_out : array-like, shape (Nr,)
            Radial query coordinates in the interior.
        theta_out : array-like, shape (Nθ,)
            Angular query coordinates.
        phi_grid : array-like, shape (N,)
            Same angular grid used by the FNN (= phi_input).
        grid_step : float
            Angular quadrature step Δφ.

        Returns
        -------
        torch.Tensor, shape (Nr, Nθ, 1)
            Predicted solution u(r_out_i, θ_out_j).
        """
        phi_grid_t = (torch.tensor(phi_grid, dtype=torch.float32)
                      if not isinstance(phi_grid, torch.Tensor) else phi_grid)
        theta_out_t = (torch.tensor(theta_out, dtype=torch.float32)
                       if not isinstance(theta_out, np.ndarray) else
                       torch.tensor(theta_out, dtype=torch.float32))

        # --- Step 1: Solve the BIE to get β(φ) on the boundary ---
        beta = self.fredholm_model(phi_input).to(torch.float32)  # (N,)

        # --- Step 2: Additional hidden layer (identity weights) ---
        # For each output angle θ_j, subtract the corresponding β(x*)
        theta_indices = torch.argmin(
            torch.abs(phi_grid_t.unsqueeze(0) - theta_out_t.unsqueeze(-1)), dim=-1
        )  # (Nθ,)

        beta_star = beta[theta_indices]                                   # (Nθ,)
        beta_star_exp = beta_star.unsqueeze(0).repeat(len(r_out), 1)      # (Nr, Nθ)
        # hidden output: β(φ) − β(x*), broadcast over (Nr, Nθ, N)
        hidden_out = beta.unsqueeze(0).unsqueeze(0) - beta_star_exp.unsqueeze(-1)
        # shape: (Nr, Nθ, N)

        # --- Step 3: Output layer via difference potential ---
        W_out = torch.tensor(
            self.diff_potentials_fn(phi_grid, r_out, theta_out) * grid_step,
            dtype=torch.float32,
        )  # (Nr, Nθ, N)

        # Boundary potential bias: ∫ β(φ) · k_boundary(φ) dφ + 0.5 β(x*)
        k_boundary = self.potential_boundary_fn(phi_grid, r_out, theta_out)  # (Nr, Nθ, N)
        bias_out = (
            torch.sum(beta.unsqueeze(0).unsqueeze(0) * k_boundary, dim=-1) * grid_step
            + 0.5 * beta_star_exp
        )  # (Nr, Nθ)

        # Final dot product: Σ_φ hidden_out · W_out + bias
        W_out_col = W_out.unsqueeze(-1)                             # (Nr, Nθ, N, 1)
        hidden_col = hidden_out.unsqueeze(-2)                       # (Nr, Nθ, 1, N)
        output = torch.matmul(hidden_col, W_out_col).squeeze(-2)    # (Nr, Nθ, 1)
        output = output + bias_out.unsqueeze(-1)                    # (Nr, Nθ, 1)

        return output
