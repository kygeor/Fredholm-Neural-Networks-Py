"""
fredholm_nn — Fredholm Neural Networks for solving Fredholm integral equations
of the second kind, BVP ODEs, and elliptic PDEs.

Quick-start
-----------
>>> from fredholm_nn import solve_linear_fie, solve_nonlinear_fie
>>> from fredholm_nn import solve_bvp_ode
>>> from fredholm_nn import solve_laplace
"""

from fredholm_nn.models.linear import FredholmNN
from fredholm_nn.models.nonlinear import NonlinearFredholmNN
from fredholm_nn.solvers.fie import solve_linear_fie, solve_nonlinear_fie
from fredholm_nn.solvers.bvp import solve_bvp_ode
from fredholm_nn.solvers.pde import solve_laplace

__all__ = [
    "FredholmNN",
    "NonlinearFredholmNN",
    "solve_linear_fie",
    "solve_nonlinear_fie",
    "solve_bvp_ode",
    "solve_laplace",
]
