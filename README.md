# Fredholm Neural Networks (Python) 

A Python package for solving **Fredholm integral equations of the second kind** using the Fredholm Neural Network (FNN) framework.

The FNN approach encodes the method of successive approximations (Picard / Krasnoselskii-Mann iterations) directly into the weights and biases of a deep network with linear activations. No training is required — the network is constructed analytically from the kernel and free term.

**Supported problem types:**

| Problem | Solver |
|---|---|
| Linear FIE (contractive operator) | `solve_linear_fie` |
| Linear FIE (non-expansive operator, KM iteration) | `solve_linear_fie(..., km_constant=κ)` |
| Non-linear FIE | `solve_nonlinear_fie` |
| Second-order BVP ODE | `solve_bvp_ode` |
| 2-D Laplace PDE (unit disc) | `solve_laplace` |

---

## Installation

**From source (recommended while the package is local):**

```bash
git clone <repo-url>
cd Fredholm_Neural_Networks
pip install -e .
```

**Requirements** (installed automatically): `torch >= 2.0`, `numpy >= 1.24`, `scipy >= 1.10`.

---

## Quick start

```python
import numpy as np
from fredholm_nn import solve_linear_fie

sol = solve_linear_fie(
    kernel=lambda x, z: np.sin(z) * np.cos(x),
    additive=lambda x: np.sin(x),
    domain=(0.0, np.pi / 2),
    n_grid=500,
    n_iterations=15,
    predict_at=np.linspace(0, 2 * np.pi, 200),
)

print(sol.f)                                     # predicted solution values
print(sol.mse(lambda x: 2 * np.sin(x)))          # MSE against exact solution
```

---

## Writing `kernel` and `additive`

Every solver accepts these two callables:

| Parameter | Contract |
|---|---|
| `kernel(x, z)` | Accepts NumPy arrays of shape `(N, 1)` and `(1, M)`, returns an `(N, M)` array. Standard NumPy broadcasting works out of the box. |
| `additive(x)` | Accepts a 1-D NumPy array **or** a `torch.Tensor`; returns values of the same type and shape. NumPy ufuncs (`np.sin`, `np.exp`, …) satisfy this automatically. |

---

## Solvers

### 1. Linear FIE — `solve_linear_fie`

Solves

$$f(x) = g(x) + \int_a^b K(x, z)\, f(z)\, dz$$

```python
from fredholm_nn import solve_linear_fie

sol = solve_linear_fie(
    kernel   = lambda x, z: 0.3 * (np.cos(25*(z-x)) + np.cos(7*(z-x))),
    additive = lambda x: np.sin(25*x) + np.sin(7*x),
    domain   = (0.0, 1.0),
    n_grid        = 500,   # quadrature points on the integration grid
    n_iterations  = 50,    # Picard iterations (= hidden layers)
    predict_at    = np.linspace(0, 1, 200),  # points to evaluate f(x) at
)
```

**For non-expansive operators** (Krasnoselskii-Mann iteration), set `km_constant` to a value in `(0, 1)`:

```python
sol = solve_linear_fie(..., km_constant=0.5)
```

**Return value — `FIESolution`:**

| Attribute | Description |
|---|---|
| `sol.x` | Query points (NumPy array) |
| `sol.f` | Predicted solution values |
| `sol.model` | Underlying `FredholmNN` model |
| `sol.error(exact)` | Pointwise absolute error given an exact solution callable |
| `sol.mse(exact)` | Mean-squared error |

---

### 2. Non-linear FIE — `solve_nonlinear_fie`

Solves

$$f(x) = g(x) + \int_a^b K(x, z)\, G(f(z))\, dz$$

via iterative linearisation. At each outer step a modified linear FIE is solved with the free term

$$\tilde{g}_n(x) = g(x) + \int K(x, z)\bigl[G(\tilde{f}_{n-1}(z)) - \tilde{f}_{n-1}(z)\bigr]\, dz.$$

```python
from fredholm_nn import solve_nonlinear_fie

sol = solve_nonlinear_fie(
    kernel       = lambda x, z: 0.1 * np.cos(x - z),
    additive     = lambda x: np.exp(x),
    nonlinearity = lambda f: np.sin(f),   # G(f)
    domain       = (0.0, 1.0),
    n_grid        = 500,
    n_iterations  = 30,   # FNN layers per inner linear solve
    n_outer       = 10,   # outer linearisation steps
    tol           = 1e-7, # convergence tolerance (max pointwise change)
    predict_at    = np.linspace(0, 1, 200),
)
```

Returns a `FIESolution` with the same attributes as above.

---

### 3. BVP ODE — `solve_bvp_ode`

Solves second-order BVPs of the form

$$y''(x) + p(x)\,y(x) = q(x), \quad x \in [a, b], \quad y(a) = \alpha,\ y(b) = \beta$$

by reducing them to a Fredholm IE via the Green's function of the interval, solving for $u(x) = y''(x)$, then recovering $y(x) = (q(x) - u(x))\,/\,p(x)$.

```python
from fredholm_nn import solve_bvp_ode

p = 3.2
sol = solve_bvp_ode(
    p_func = lambda x: 3*p / (p + x**2)**2,
    q_func = lambda x: np.zeros_like(x),
    alpha  = 0.0,
    beta   = 1.0 / np.sqrt(p + 1.0),
    domain = (0.0, 1.0),
    n_grid       = 1000,
    n_iterations = 10,
    predict_at   = np.linspace(0, 1, 200),
)
```

**Return value — `BVPSolution`:**

| Attribute | Description |
|---|---|
| `sol.x` | Query points |
| `sol.y` | ODE solution $y(x)$ |
| `sol.u` | Auxiliary variable $u(x) = y''(x)$ |
| `sol.fie_solution` | The underlying `FIESolution` |
| `sol.error(exact)` | Pointwise absolute error |
| `sol.mse(exact)` | Mean-squared error |

---

### 4. Laplace PDE — `solve_laplace`

Solves the 2-D Laplace equation on the unit disc

$$\Delta u = 0\ \text{ in } \Omega, \qquad u = f\ \text{ on } \partial\Omega$$

by reformulating it as a boundary integral equation (BIE) and solving the BIE with an FNN. The interior solution is then recovered via the double-layer potential.

```python
from fredholm_nn import solve_laplace

sol = solve_laplace(
    boundary_fn = lambda theta: 1.0 + np.cos(2 * theta),  # f(θ) on ∂Ω
    n_boundary   = 500,   # quadrature points on the unit circle
    n_iterations = 30,    # FNN layers for the BIE
    km_constant  = 0.3,   # BIE kernel is non-expansive; use KM iteration
    r_values     = np.linspace(0, 1, 100),
    theta_values = np.linspace(0, 2*np.pi, 200),
)

print(sol.u.shape)   # (100, 200) — u(r, θ) on the polar meshgrid
```

**Return value — `LaplaceSolution`:**

| Attribute | Description |
|---|---|
| `sol.r`, `sol.theta` | Radial and angular grids |
| `sol.u` | Solution array, shape `(Nr, Nθ)` |
| `sol.model` | Underlying `LaplaceFredholmNN` model |
| `sol.error(exact)` | Pointwise absolute error; `exact(R, Theta)` receives 2-D meshgrid arrays |
| `sol.mse(exact)` | Mean-squared error |

---

## Package structure

```
fredholm_nn/
├── models/
│   ├── linear.py       FredholmNN            — core FNN for linear FIEs
│   ├── nonlinear.py    NonlinearFredholmNN   — iterative linearisation
│   └── pde.py          LaplaceFredholmNN     — double-layer BIE for Laplace
├── solvers/
│   ├── fie.py          solve_linear_fie, solve_nonlinear_fie
│   ├── bvp.py          solve_bvp_ode
│   └── pde.py          solve_laplace
└── utils/
    ├── grid.py         make_uniform_grid, make_grid_dictionary
    └── quadrature.py   parallel_integrate_meshgrid
```

---

## Parameter reference

| Parameter | Used in | Description |
|---|---|---|
| `kernel` | all FIE/BVP solvers | $K(x,z)$ — see broadcasting note above |
| `additive` | all FIE/BVP solvers | $g(x)$ — free term |
| `domain` | all FIE/BVP solvers | Integration interval $(a, b)$ |
| `n_grid` | all FIE/BVP solvers | Number of quadrature points (default 500) |
| `n_iterations` | all FIE/BVP solvers | Picard/KM iterations $K$ (default 50) |
| `predict_at` | all FIE/BVP solvers | Query points for $\hat{f}(x)$; defaults to the integration grid |
| `km_constant` | linear, nonlinear, Laplace | KM relaxation $\kappa \in (0,1]$; use $< 1$ for non-expansive operators |
| `nonlinearity` | `solve_nonlinear_fie` | $G: \mathbb{R} \to \mathbb{R}$ applied pointwise to $f$ |
| `n_outer` | `solve_nonlinear_fie` | Outer linearisation iterations (default 10) |
| `tol` | `solve_nonlinear_fie` | Convergence tolerance $\|\tilde{f}_n - \tilde{f}_{n-1}\|_\infty$ (default 1e-6) |
| `p_func`, `q_func` | `solve_bvp_ode` | Coefficient and RHS of the ODE |
| `alpha`, `beta` | `solve_bvp_ode` | Boundary values $y(a)$, $y(b)$ |
| `boundary_fn` | `solve_laplace` | Dirichlet BC $f(\theta)$ on the unit circle |
| `n_boundary` | `solve_laplace` | Quadrature points on $\partial\Omega$ (default 500) |

---

## Citation

If you use this package please cite:

```bibtex
@article{georgiou2025fredholm,
  title   = {Fredholm neural networks},
  author  = {Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {47},
  number  = {4},
  pages   = {C1006--C1031},
  year    = {2025},
  publisher = {SIAM}
}
```

and/or

```bibtex
@article{georgiou2025fredholm_pde,
  title   = {Fredholm Neural Networks for forward and inverse problems in elliptic PDEs},
  author  = {Georgiou, Kyriakos and Siettos, Constantinos and Yannacopoulos, Athanasios N},
  journal = {arXiv preprint arXiv:2507.06038},
  year    = {2025}
}
```
