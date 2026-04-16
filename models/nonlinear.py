"""
Non-linear Fredholm Neural Network model.

Solves non-linear Fredholm integral equations of the second kind:

    f(x) = g(x) + ∫_Ω K(x, z) G(f(z)) dz

via iterative linearisation.  At each outer step n the algorithm solves
the linear FIE

    f̃_n(x) = L̃_n(x) + ∫_Ω K(x, z) f̃_n(z) dz

with modified free term

    L̃_n(x) = g(x) + ∫_Ω K(x, z) [G(f̃_{n-1}(z)) − f̃_{n-1}(z)] dz.

The inner linear FIE is solved in closed-form by :class:`FredholmNN`.

Reference
---------
Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
Fredholm neural networks. SIAM Journal on Scientific Computing, 47(4).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable

from fredholm_nn.models.linear import FredholmNN
from fredholm_nn.utils.grid import make_grid_dictionary


class NonlinearFredholmNN(nn.Module):
    """
    Iterative-linearisation solver for non-linear FIEs.

    At each outer iteration the correction

        c_{n-1}(z) = G(f̃_{n-1}(z)) − f̃_{n-1}(z)

    is computed on the integration grid, and a new linear FIE is formed
    with the modified additive

        L̃_n(x) = g(x) + Σ_j K(x, z_j) · c_{n-1}(z_j) · Δz.

    This modified additive is passed to :class:`FredholmNN` which solves
    the resulting linear system in one forward pass.

    Parameters
    ----------
    kernel : callable
        K(x, z) — same broadcasting convention as :class:`FredholmNN`.
    additive : callable
        g(x) — free term of the *original* non-linear FIE.  Must handle
        both ``np.ndarray`` and ``torch.Tensor`` inputs.
    nonlinearity : callable
        G : ℝ → ℝ applied pointwise.  Must accept and return array-like.
    grid : np.ndarray
        Integration grid z.
    grid_step : float
        Quadrature step Δz.
    n_iterations : int
        Number of FNN hidden layers K used in each inner linear solve.
    km_constant : float, optional
        KM relaxation parameter (default 1.0).
    """

    def __init__(
        self,
        kernel: Callable,
        additive: Callable,
        nonlinearity: Callable,
        grid: np.ndarray,
        grid_step: float,
        n_iterations: int,
        km_constant: float = 1.0,
    ):
        super().__init__()

        self.kernel = kernel
        self.additive = additive
        self.nonlinearity = nonlinearity
        self.grid = grid
        self.grid_step = grid_step
        self.n_iterations = n_iterations
        self.km_constant = km_constant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _modified_additive(self, f_prev: np.ndarray) -> Callable:
        """
        Build a callable L̃_n(x) for the next inner linear solve.

        L̃_n(x) = g(x) + Σ_j K(x, z_j) · [G(f_{n-1}(z_j)) − f_{n-1}(z_j)] · Δz

        Parameters
        ----------
        f_prev : np.ndarray, shape (n_grid,)
            Solution estimate from the previous outer iteration, evaluated
            on the integration grid.

        Returns
        -------
        callable
            A function of x (array-like) returning L̃_n(x).
        """
        # Correction term evaluated on the grid — shape (n_grid,)
        G_f = np.asarray(self.nonlinearity(f_prev), dtype=np.float64)
        correction = G_f - f_prev  # c_{n-1}(z)

        kernel = self.kernel
        grid = self.grid
        dz = self.grid_step
        additive = self.additive

        def _additive_n(x):
            """L̃_n(x) for query points x."""
            is_tensor = isinstance(x, torch.Tensor)
            x_np = x.numpy() if is_tensor else np.asarray(x, dtype=np.float64)

            # Broadcast K(z_j, x_i): shape (n_grid, n_query)
            K_mat = kernel(
                grid[:, np.newaxis],       # (n_grid, 1)
                x_np[np.newaxis, :],       # (1, n_query)
            ).astype(np.float64)

            # Riemann sum over z: (n_query,)
            integral_correction = K_mat.T @ correction * dz

            # Original free term g(x)
            g_x = np.asarray(additive(x_np), dtype=np.float64)

            result = g_x + integral_correction
            return torch.tensor(result, dtype=torch.float32) if is_tensor else result

        return _additive_n

    def _solve_inner(self, modified_additive: Callable) -> np.ndarray:
        """
        Solve one inner linear FIE using FredholmNN.

        Returns the solution evaluated on ``self.grid``.
        """
        grid_dict = make_grid_dictionary(self.grid, self.n_iterations)
        inner_model = FredholmNN(
            grid_dictionary=grid_dict,
            kernel=self.kernel,
            additive=modified_additive,
            grid_step=self.grid_step,
            n_iterations=self.n_iterations,
            km_constant=self.km_constant,
        )
        with torch.no_grad():
            out = inner_model(self.grid)
        return out.numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        predict_array: torch.Tensor,
        grid_tensor: torch.Tensor,
        n_outer: int = 10,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Run the outer linearisation loop and return predictions.

        Parameters
        ----------
        predict_array : torch.Tensor
            Query points x at which to evaluate f̂(x).
        grid_tensor : torch.Tensor
            Integration grid (must match ``self.grid``).
        n_outer : int
            Maximum number of outer iterations.
        tol : float
            Convergence tolerance on ||f_n − f_{n-1}||_∞.

        Returns
        -------
        torch.Tensor, shape (len(predict_array),)
        """
        grid_np = self.grid

        # Initialise f̃_0 = g(z) on the integration grid
        f_prev = np.asarray(self.additive(grid_np), dtype=np.float64)

        for _ in range(n_outer):
            modified_add = self._modified_additive(f_prev)
            f_curr = self._solve_inner(modified_add)

            # Convergence check
            if np.max(np.abs(f_curr - f_prev)) < tol:
                f_prev = f_curr
                break
            f_prev = f_curr

        # Final evaluation at the requested query points using the converged
        # modified additive of the last iteration.
        final_additive = self._modified_additive(f_prev)
        grid_dict = make_grid_dictionary(grid_np, self.n_iterations)
        final_model = FredholmNN(
            grid_dictionary=grid_dict,
            kernel=self.kernel,
            additive=final_additive,
            grid_step=self.grid_step,
            n_iterations=self.n_iterations,
            km_constant=self.km_constant,
        )
        with torch.no_grad():
            output = final_model(predict_array.numpy())

        return output
