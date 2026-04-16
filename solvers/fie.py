"""
High-level solvers for Fredholm integral equations of the second kind.

Linear FIE
----------
    f(x) = g(x) + ∫_Ω K(x, z) f(z) dz

Non-linear FIE
--------------
    f(x) = g(x) + ∫_Ω K(x, z) G(f(z)) dz

    Solved via iterative linearisation: at each outer iteration n, f_n is
    taken as the solution of the linear FIE

        f̃_n(x) = L̃(f̃_{n-1})(x) + ∫_Ω K(x, z) f̃_n(z) dz

    where the modified free term is

        L̃(f̃_{n-1})(x) = g(x) + ∫_Ω K(x, z) [G(f̃_{n-1}(z)) − f̃_{n-1}(z)] dz

    This process converges to f* when the nonlinear operator is non-expansive.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Callable

from fredholm_nn.models.linear import FredholmNN
from fredholm_nn.models.nonlinear import NonlinearFredholmNN
from fredholm_nn.utils.grid import make_uniform_grid, make_grid_dictionary


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FIESolution:
    """
    Container for the output of ``solve_linear_fie`` and ``solve_nonlinear_fie``.

    Attributes
    ----------
    x : np.ndarray
        Query points at which the solution is evaluated.
    f : np.ndarray
        Predicted solution values f̂(x).
    model : FredholmNN or NonlinearFredholmNN
        The underlying model (retained for inspection or re-use).
    """

    x: np.ndarray
    f: np.ndarray
    model: object = field(repr=False)

    def error(self, exact: Callable) -> np.ndarray:
        """
        Pointwise absolute error |f̂(x) − f_exact(x)|.

        Parameters
        ----------
        exact : callable
            The exact solution f(x) as a function of x.

        Returns
        -------
        np.ndarray
        """
        return np.abs(self.f - np.asarray(exact(self.x), dtype=np.float64))

    def mse(self, exact: Callable) -> float:
        """Mean-squared error against an exact solution."""
        return float(np.mean(self.error(exact) ** 2))


# ---------------------------------------------------------------------------
# Linear FIE solver
# ---------------------------------------------------------------------------

def solve_linear_fie(
    kernel: Callable,
    additive: Callable,
    domain: tuple[float, float],
    *,
    n_grid: int = 500,
    n_iterations: int = 50,
    predict_at: np.ndarray | None = None,
    km_constant: float = 1.0,
) -> FIESolution:
    """
    Solve a linear Fredholm integral equation of the second kind.

    The equation is

        f(x) = g(x) + ∫_{a}^{b} K(x, z) f(z) dz

    using the Fredholm Neural Network (successive approximations / KM
    iteration).

    Parameters
    ----------
    kernel : callable
        K(x, z) — kernel function.  Must support NumPy broadcasting:
        when called with arrays of shapes (N, 1) and (1, M) it should
        return an (N, M) array.
    additive : callable
        g(x) — free term.  Must accept a 1-D ``np.ndarray`` *or*
        ``torch.Tensor`` and return an array/tensor of the same shape.
    domain : (a, b)
        Integration interval.
    n_grid : int, optional
        Number of uniform grid points for numerical quadrature (default 500).
    n_iterations : int, optional
        Number of Picard / KM iterations K (= number of hidden layers,
        default 50).
    predict_at : np.ndarray, optional
        Points at which to evaluate f̂(x).  Defaults to the integration
        grid.
    km_constant : float, optional
        Relaxation parameter κ ∈ (0, 1].  Use ``1.0`` (default) for the
        contractive case; use a value < 1 for non-expansive operators.

    Returns
    -------
    FIESolution
        Solution container with attributes ``.x``, ``.f``, and ``.model``.

    Examples
    --------
    Solve  f(x) = sin(x) + ∫_0^{π/2} sin(x)cos(z) f(z) dz :

    >>> import numpy as np
    >>> sol = solve_linear_fie(
    ...     kernel=lambda x, z: np.sin(z) * np.cos(x),
    ...     additive=lambda x: np.sin(x),
    ...     domain=(0.0, np.pi / 2),
    ...     n_grid=500,
    ...     n_iterations=15,
    ...     predict_at=np.linspace(0, 2 * np.pi, 200),
    ... )
    >>> sol.f.shape
    (200,)
    """
    a, b = domain
    grid, dz = make_uniform_grid(a, b, n_grid)
    grid_dict = make_grid_dictionary(grid, n_iterations)

    if predict_at is None:
        predict_at = grid

    model = FredholmNN(
        grid_dictionary=grid_dict,
        kernel=kernel,
        additive=additive,
        grid_step=dz,
        n_iterations=n_iterations,
        km_constant=km_constant,
    )

    with torch.no_grad():
        output = model(predict_at)

    return FIESolution(
        x=np.asarray(predict_at, dtype=np.float64),
        f=output.numpy().astype(np.float64),
        model=model,
    )


# ---------------------------------------------------------------------------
# Non-linear FIE solver
# ---------------------------------------------------------------------------

def solve_nonlinear_fie(
    kernel: Callable,
    additive: Callable,
    nonlinearity: Callable,
    domain: tuple[float, float],
    *,
    n_grid: int = 500,
    n_iterations: int = 50,
    n_outer: int = 10,
    predict_at: np.ndarray | None = None,
    km_constant: float = 1.0,
    tol: float = 1e-6,
) -> FIESolution:
    """
    Solve a non-linear Fredholm integral equation of the second kind.

    The equation is

        f(x) = g(x) + ∫_{a}^{b} K(x, z) G(f(z)) dz

    via iterative linearisation.  At each outer step n the method solves
    the linear FIE

        f̃_n(x) = L̃_n(x) + ∫ K(x, z) f̃_n(z) dz

    with modified free term

        L̃_n(x) = g(x) + ∫ K(x, z) [G(f̃_{n-1}(z)) − f̃_{n-1}(z)] dz.

    Parameters
    ----------
    kernel : callable
        K(x, z) — same broadcasting convention as :func:`solve_linear_fie`.
    additive : callable
        g(x) — free term of the original non-linear FIE.
    nonlinearity : callable
        G : ℝ → ℝ — applied pointwise to f.  Must accept and return
        array-like values.
    domain : (a, b)
        Integration interval.
    n_grid : int, optional
        Number of quadrature points (default 500).
    n_iterations : int, optional
        FNN hidden layers K per outer solve (default 50).
    n_outer : int, optional
        Maximum number of outer linearisation iterations (default 10).
    predict_at : np.ndarray, optional
        Query points.  Defaults to the integration grid.
    km_constant : float, optional
        KM relaxation parameter (default 1.0).
    tol : float, optional
        Convergence tolerance on the max pointwise change between successive
        outer iterates.  Iteration stops early if ||f_n − f_{n-1}||_∞ < tol.

    Returns
    -------
    FIESolution
    """
    a, b = domain
    grid, dz = make_uniform_grid(a, b, n_grid)
    grid_dict = make_grid_dictionary(grid, n_iterations)

    if predict_at is None:
        predict_at = grid

    predict_tensor = torch.tensor(predict_at, dtype=torch.float32)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    model = NonlinearFredholmNN(
        kernel=kernel,
        additive=additive,
        nonlinearity=nonlinearity,
        grid=grid,
        grid_step=dz,
        n_iterations=n_iterations,
        km_constant=km_constant,
    )

    with torch.no_grad():
        output = model(predict_tensor, grid_tensor, n_outer=n_outer, tol=tol)

    return FIESolution(
        x=np.asarray(predict_at, dtype=np.float64),
        f=output.numpy().astype(np.float64),
        model=model,
    )
