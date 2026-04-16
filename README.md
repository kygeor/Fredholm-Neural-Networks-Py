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


## Fredholm Neural Networks Theory 

### Background
The basis of FNNs is the method of successive approximations (fixed point iterations) to approximate the fixed-point solution to Fredholm Integral Equations (FIEs). Specifically, the framework is built upon linear FIEs of the second kind, which are of the form:

$$f(x) = g(x) + \int_{\Omega}K(x,z) f(z)dz, $$

as well as the non-linear counterpart,

$$f(x) = g(x) + \int_{\Omega}K(x,z) G(f(z))dz,$$

for some function $G: \mathbb{R} \rightarrow\mathbb{R}$ considered to be a Lipschitz function. 

We consider the cases where the integral operators are either contractive or non-expansive. This allows linear FIE defined by a non-expansive operator $\mathcal{T}$, and a sequence $\{\kappa_n\}, \kappa_n \in (0,1]$ such that $\sum_n \kappa_n(1-\kappa_n) = \infty$. Then, the iterative scheme:

$$f_{n+1}(x) = f_n(x) + \kappa_n(\mathcal{T}f_n(x) -f_n(x)) = (1-\kappa_n)f_n(x) + \kappa_n \mathcal{T} f_n(x),$$

with $f_0(x) = g(x)$, converges to the fixed point solution of the FIE, $f^{*}(x)$.

When $\mathcal{T}$ is a contraction, we can obtain the iterative process:
$$f_n(x)= g(x) +  \int_{\Omega}f_{n-1})(x), \,\,\ n \geq 1,$$
which converges to the fixed point solution. This is often referred to as the method of successive approximations.

### FNN construction for forward FIEs 

Fredholm Neural Networks are based on the observation that the FIE approximation $f_K(x)$ can be implemented as a deep neural network with a one-dimensional input $x$, $M$ hidden layers, a linear activation function and a single output node corresponding to the estimated solution $f(x)$. The weights and biases are:

$$
W_1 =
\begin{bmatrix}
\kappa g(z_1) \\
\vdots \\
\kappa g(z_{N})
\end{bmatrix},
\qquad
b_1 =
\begin{bmatrix}
0 \\
\vdots \\
0
\end{bmatrix}.
$$

for the first hidden layer,

$$
W_m =
\begin{bmatrix}
K_D(z_1) & K(z_1,z_2)\,\Delta z & \cdots & K(z_1,z_N)\,\Delta z \\
K(z_2,z_1)\,\Delta z & K_D(z_2) & \cdots & K(z_2,z_N)\,\Delta z \\
\vdots & \vdots & \ddots & \vdots \\
K(z_N,z_1)\,\Delta z & K(z_N,z_2)\,\Delta z & \cdots & K_D(z_N)
\end{bmatrix},
$$

and

$$
b_m =
\begin{bmatrix}
\kappa g(z_1) \\
\vdots \\
\kappa g(z_N)
\end{bmatrix},
\qquad m=2,\dots,M-1,
$$

where $K_D(z) := K(z,z)\,\Delta z + (1-\kappa_m)$. Finally,

$$
W_M =
\begin{bmatrix}
K(z_1,x)\,\Delta z \\
\vdots \\
K(z_{i-1},x)\,\Delta z \\
K_D(x) \\
K(z_{i+1},x)\,\Delta z \\
\vdots \\
K(z_N,x)\,\Delta z
\end{bmatrix},
\qquad
b_M = \kappa g(x),
$$

assuming $z_i = x$.


<img width="324" height="290" alt="Screenshot 2025-10-08 at 11 45 05 AM" src="https://github.com/user-attachments/assets/2cdfd98b-7c52-4119-999d-b1bc40732a6b" /> 
<img width="575" height="248" alt="Screenshot 2025-10-08 at 11 45 33 AM" src="https://github.com/user-attachments/assets/bbda1e93-36b5-4c83-afa3-8b86d9459996" />  

*Figure 1: Architecture of the Fredholm Neural Network (FNN). Outputs can be considered across the entire (or a subset of the) input grid, or for an arbitrary output vector as shown in the second graph, by applying the integral mapping one last time.*


### Application to non-linear FIEs 

We can create an iterative process that "linearizes" the integral equation and allows us to solve a linear FIE at each step. To this end, consider the non-linear, non-expansive integral operator:

$$(\mathcal{T}f)(x) := g(x) + \int_{\Omega}K(x,z) G(f(z))dz.$$

Then, the iterative scheme $f_n(x) = \tilde{f}_n(x)$, where $\tilde{f}_n(x)$ is the solution to the linear FIE:

$$\tilde{f}_{n}(x) = ({L}\tilde{f}_{n-1})(x) + \int_{\Omega}K(x,z) \tilde{f}_{n}(z))dz,$$
    
where: 

$$(\mathcal{L}\tilde{f}_{n-1})(x) := g(x) + \int_{\Omega} K(x,y)\big( G(\tilde{f}_{n-1}(y)) - \tilde{f}_{n-1}(y)\big)dy,$$ 

for $n \geq 1$, converges to the fixed point $f^*$  which is a solution of the non-linear FIE.

<img width="636" height="207" alt="Screenshot 2025-10-08 at 1 35 31 PM" src="https://github.com/user-attachments/assets/f692d52e-21a0-4f2a-b668-f8a938527a3f" />

*Figure 2: Iterative process to solve the non-linear FIE using the Fredholm NN architecture.*


### Application to BVP ODEs

Consider a BVP of the form:

$$y''(x) + g(x)y(x) = h(x), 0<x<1,$$ 
    
with $y(0) = \alpha, y(1) = \beta$. Then we can solve the BVP by obtaining the following FIE:

$$u(x) = f(x) + \int_{0}^{1} K(x,t) u(t)dt,$$

where $u(x) = y''(x), f(x) = h(x) - \alpha g(x) - (\beta - \alpha) x g(x)$, and the kernel is given by:

$$ K(x,t) = 
    \begin{cases}
        t(1-x)g(x), \,\,\, 0 \leq t \leq x \\
        x(1-t)g(x), \,\,\, x\leq t \leq 1.
    \end{cases}$$
    
Finally, by definition of $u(x)$, we can obtain the solution to the BVP by:

$$y(x) = \frac{h(x) - u(x)}{g(x)}.$$


### Fredholm Neural Networks for the Laplace PDE

Here we briefly provide the background in Potential Theory and how it is applied in the context of FNNs, resulting in the Fredholm Neural Network to solve the PDE.

Consider the two-dimensional Laplace equation for $u(x)$:

$$\begin{cases}\Delta u(x) = 0, & \text { for } x \in \Omega \\
u(x)= f(x) & \text { for } {x} \in \partial \Omega. \end{cases}$$

Its solution can be written via the double layer boundary integral given by:

$$u(x) =  \int_{\partial \Omega} \beta(y) \frac{\partial \Phi}{\partial n_{y}}(x, y) d \sigma_{y} ,  x \in \Omega,$$

where $\Phi(x,y)$ is the fundamental solution of the Laplace equation, $n_y$ is the outward pointing normal vector to $y$, $\sigma_y$ is the surface element at point $y\in \partial \Omega$, and $\frac{\partial \Phi}{\partial n_{y}} = n_y \cdot \nabla_{ y}{\Phi}$. It can be shown that the following limit holds, as we approach the boundary: 

$$\lim _{\substack{x \rightarrow x^{\star} \\ x \in \Omega}}   \int_{\partial \Omega} \beta(y) \frac{\partial \Phi}{\partial n_y}(x, y) d \sigma_{y} =u\left({x}^{\star}\right) - \frac{1}{2} \beta(x^{\star}), \quad x^{\star} \in \partial \Omega.$$

Hence, the function $\beta({x}^{\star})$, defined on the boundary, must satisfy the Boundary Integral Equation (BIE):

$$\beta({x}^{\star}) = 2 f(x^{\star}) - 2 \int_{\partial \Omega} \beta(y) \frac{\partial \Phi}{\partial n_{y}}(x^{\star}, y) d \sigma_{y},  x^{\star} \in \partial \Omega.$$

<img width="548" height="376" alt="Screenshot 2025-10-08 at 4 58 58 PM" src="https://github.com/user-attachments/assets/f9edb609-f257-4c06-b96e-7ee4095c34bd" />

*Figure 4: Custom FNN construction. The first component is a Fredholm Neural Network and the second encapsulates the representation of the double layer potential, decomposed into a the final hidden layer.*


####  
The Laplace PDE 

$$
\begin{cases}
 \Delta u(x)  = 0, \quad x \in \Omega \\ 
u(x) = f(x), \quad x \in \partial \Omega.   
\end{cases}
$$

can be solved using a Fredholm NN, with M+1 hidden layers, where the weights and biases of the M hidden layers are used iteratively solve the BIE on a discretized grid of the boundary, $y_1, \dots, y_N$, 
for which the final and output weights $W_{M+1} \in \mathbb{R}^{N \times N}, W_O \in \mathbb{R}^N$ are given by:

$$
W_{M+1}= I_{N \times N}, 
W_{O}= \left(\begin{array}{cccc}
\mathcal{D} \Phi(x, y_1)\Delta \sigma_y, & \mathcal{D} \Phi(x, y_2)\Delta\sigma_y, & \dots, & \mathcal{D} \Phi(x, y_N) \Delta \sigma_y
\end{array}\right)^{\top},
$$

where we define the simple operator $\mathcal{D} \Phi({x}, {y}):= \Big(\frac{\partial \Phi}{\partial n_y}(x, y)- \frac{\partial \Phi}{\partial n_y}(x^{\star}, y)\Big)$. The corresponding biases $b_{M+1} \in \mathbb{R}^{N}$ and $b_O \in \mathbb{R}$ are given by:

$$ b_{M+1} = \left(\begin{array}{ccc}
-\beta(x^{\star}), \dots, - \beta(x^{\star})
\end{array}\right)^{\top}, b_O= \frac{1}{2} \beta(x^{\star}) + \int_{\partial \Omega} \beta(y) \frac{\partial \Phi(x^*, y)}{\partial n_y} d\sigma_y,
$$

where $x^*:= (1, \phi) \in \partial \Omega$ is the unique point on the boundary corresponding to $x:= (r, \phi) \in \Omega$.  

