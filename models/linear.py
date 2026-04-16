from __future__ import annotations

"""
Linear Fredholm Neural Network model.

Implements the FNN architecture for linear Fredholm integral equations of the
second kind:

    f(x) = g(x) + ∫_Ω K(x, z) f(z) dz

Two convergence modes are supported via the ``km_constant`` parameter:

* **Contractive** (``km_constant=1.0``): direct successive approximations,
  converges when the integral operator is a contraction.
* **Krasnoselskii-Mann** (``0 < km_constant < 1``): relaxed iteration,
  converges for non-expansive operators provided the sequence of relaxation
  parameters satisfies the divergence condition Σ κ(1−κ) = ∞.

Reference
---------
Georgiou, K., Siettos, C., & Yannacopoulos, A. N. (2025).
Fredholm neural networks. SIAM Journal on Scientific Computing, 47(4).
"""

import numpy as np
import torch
import torch.nn as nn


class FredholmNN(nn.Module):
    """
    Fredholm Neural Network for linear FIEs of the second kind.

    The network has ``K + 1`` hidden layers whose weights and biases are
    determined analytically from the kernel and the additive (source) term,
    discretised on a provided grid.

    Parameters
    ----------
    grid_dictionary : dict[str, np.ndarray]
        Mapping ``'layer_i'`` → 1-D grid array for layer *i*,
        for ``i = 0, ..., K``.  Typically all entries are the same grid.
    kernel : callable
        K(x, z) — accepts two broadcastable array-like arguments and returns
        an array of the same broadcast shape.
    additive : callable
        g(x) — the free term.  Must accept a 1-D ``np.ndarray`` *or* a
        1-D ``torch.Tensor`` and return values of the same type/shape.
    grid_step : float
        Quadrature step size Δz used in the Riemann sum approximation of
        the integral.
    n_iterations : int
        Number of hidden layers K (= number of Picard/KM iterations).
    km_constant : float, optional
        Relaxation parameter κ ∈ (0, 1].  Use ``1.0`` (default) for the
        standard contractive case; use a value < 1 for non-expansive
        operators (Krasnoselskii-Mann).

    Notes
    -----
    *  The weight matrix for layer 0 is the identity scaled by g(z), so
       the first hidden-layer activations equal g(z).
    *  For layers 1 … K the weight matrix encodes κ · K(z_i, z_j) · Δz
       plus the (1 − κ) · I correction required by the KM scheme.
    *  The output (prediction) layer applies the final integral mapping to
       an arbitrary array of query points, which need not coincide with the
       integration grid.
    """

    def __init__(
        self,
        grid_dictionary: dict,
        kernel,
        additive,
        grid_step: float,
        n_iterations: int,
        km_constant: float = 1.0,
    ):
        super().__init__()

        if not (0.0 < km_constant <= 1.0):
            raise ValueError("km_constant must be in (0, 1].")

        self.grid_dictionary = grid_dictionary
        self.kernel = kernel
        self.additive = additive
        self.grid_step = grid_step
        self.K = n_iterations
        self.km_constant = km_constant

        # Layer widths: input + (K+1) hidden layers
        self._layer_sizes = [len(grid_dictionary["layer_0"])] + [
            len(grid_dictionary[f"layer_{i}"]) for i in range(self.K + 1)
        ]

    # ------------------------------------------------------------------
    # Weight / bias construction
    # ------------------------------------------------------------------

    def _build_weights_and_biases(self):
        """Return lists of weight tensors and bias tensors for all layers."""
        weights, biases = [], []
        kappa = self.km_constant

        for i in range(self.K + 1):
            grid_i = self.grid_dictionary[f"layer_{i}"]

            if i == 0:
                # First layer: diagonal matrix with g(z) on the diagonal.
                g_vals = np.asarray(self.additive(grid_i), dtype=np.float32)
                mat = np.diag(g_vals)
                weights.append(torch.tensor(mat, dtype=torch.float32))
                biases.append(torch.zeros(self._layer_sizes[i], dtype=torch.float32))
            else:
                grid_prev = self.grid_dictionary[f"layer_{i - 1}"]

                # Broadcast: rows = grid_prev, cols = grid_i
                z_row = grid_prev[:, np.newaxis]  # (N, 1)
                z_col = grid_i[np.newaxis, :]     # (1, M)

                # Core kernel matrix scaled by κ·Δz
                W = np.asarray(self.kernel(z_row, z_col), dtype=np.float32) * self.grid_step * kappa

                # KM correction: add (1 − κ) on the diagonal (when grids match)
                if W.shape[0] == W.shape[1]:
                    np.fill_diagonal(W, W.diagonal() + (1.0 - kappa))

                weights.append(torch.tensor(W, dtype=torch.float32))

                # Bias: κ · g(z_i)
                b = np.asarray(self.additive(grid_i), dtype=np.float32) * kappa
                biases.append(torch.tensor(b, dtype=torch.float32))

        return weights, biases

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, predict_array):
        """
        Run the FNN and return solution estimates at ``predict_array``.

        Parameters
        ----------
        predict_array : array-like or torch.Tensor
            Query points x at which to evaluate the solution f(x).

        Returns
        -------
        torch.Tensor, shape (len(predict_array),)
            Predicted solution values f̂(x).
        """
        weights, biases = self._build_weights_and_biases()

        # Propagate through hidden layers starting from the all-ones vector.
        x = torch.ones(self._layer_sizes[0], dtype=torch.float32)
        for W, b in zip(weights, biases):
            x = W.T @ x + b

        # x is now the last hidden-layer output (shape: N_grid,)
        last_grid = self.grid_dictionary[f"layer_{self.K}"]
        grid_tensor = torch.tensor(last_grid, dtype=torch.float32)

        if not isinstance(predict_array, torch.Tensor):
            predict_array = torch.tensor(predict_array, dtype=torch.float32)
        predict_array = predict_array.to(torch.float32)

        # Final output layer: for each query point xq compute
        #   f̂(xq) = Σ_j K(z_j, xq) · Δz · x_j  +  κ · g(xq)
        # Vectorised over all query points at once.
        z_col = grid_tensor.unsqueeze(0)            # (1, N_grid)
        xq_row = predict_array.unsqueeze(1)         # (n_query, 1)

        # kernel expects numpy-broadcastable inputs; use numpy then convert
        K_out = torch.tensor(
            self.kernel(
                grid_tensor.numpy()[:, np.newaxis],   # (N_grid, 1)
                predict_array.numpy()[np.newaxis, :], # (1, n_query)
            ).astype(np.float32),
            dtype=torch.float32,
        )  # shape: (N_grid, n_query)

        integral_term = (K_out * self.grid_step).T @ x  # (n_query,)

        raw = self.additive(predict_array)
        if isinstance(raw, torch.Tensor):
            additive_term = raw.to(torch.float32)
        else:
            additive_term = torch.tensor(np.asarray(raw, dtype=np.float32), dtype=torch.float32)

        return (integral_term + additive_term).squeeze()
