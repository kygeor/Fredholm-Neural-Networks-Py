from __future__ import annotations

"""
Grid construction utilities for Fredholm Neural Networks.

The FNN architecture requires:
* a 1-D integration grid z = [z_0, …, z_N] covering the domain [a, b]
* a ``grid_dictionary`` mapping ``'layer_i'`` → grid array for i = 0 … K,
  where K is the number of iterations (hidden layers).

In the standard formulation every layer uses the same grid, but the
dictionary format allows per-layer refinement if needed.
"""

import numpy as np


def make_uniform_grid(a: float, b: float, n_points: int) -> tuple[np.ndarray, float]:
    """
    Return a uniform grid on [a, b] and its step size.

    Parameters
    ----------
    a, b : float
        Domain endpoints.
    n_points : int
        Number of grid points (including both endpoints).

    Returns
    -------
    grid : np.ndarray, shape (n_points,)
        Uniformly spaced points from a to b inclusive.
    step : float
        Grid spacing Δz = (b − a) / (n_points − 1).

    Examples
    --------
    >>> z, dz = make_uniform_grid(0.0, 1.0, 501)
    >>> z.shape
    (501,)
    >>> dz
    0.002
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    grid = np.linspace(a, b, n_points)
    step = (b - a) / (n_points - 1)
    return grid, step


def make_grid_dictionary(
    grid: np.ndarray,
    n_iterations: int,
    per_layer_grids: dict | None = None,
) -> dict:
    """
    Build the ``grid_dictionary`` expected by :class:`~fredholm_nn.models.linear.FredholmNN`.

    By default every layer shares the same ``grid``.  Pass ``per_layer_grids``
    to override individual layers (useful for adaptive refinement).

    Parameters
    ----------
    grid : np.ndarray
        Default grid array used for all layers.
    n_iterations : int
        Number of FNN iterations K.  The dictionary will have keys
        ``'layer_0'`` through ``'layer_K'``.
    per_layer_grids : dict, optional
        Mapping ``layer_index (int)`` → ``np.ndarray`` to override
        specific layers.  E.g. ``{0: fine_grid, 5: coarser_grid}``.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``'layer_0'``, …, ``'layer_{n_iterations}'``.

    Examples
    --------
    >>> z, dz = make_uniform_grid(0.0, 1.0, 501)
    >>> gd = make_grid_dictionary(z, n_iterations=20)
    >>> list(gd.keys())[:3]
    ['layer_0', 'layer_1', 'layer_2']
    """
    overrides = per_layer_grids or {}
    return {
        f"layer_{i}": overrides.get(i, grid)
        for i in range(n_iterations + 1)
    }
