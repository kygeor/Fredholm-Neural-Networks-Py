"""
run_examples.py
---------------
Runs one example of every solver, prints the MSE against the known exact
solution, and saves two figures per example:

  <n>_<name>_solution.png  — predicted vs. exact overlay
  <n>_<name>_error.png     — pointwise absolute error

For the non-linear FIE (no closed-form exact), only the solution plot is
produced.  For the Laplace PDE the plots are 2-D contour maps.

Run from the project root:
    python run_examples.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

PASS_ICON  = "[PASS]"
FAIL_ICON  = "[FAIL]"
MSE_TOL    = 1e-4
OUTPUT_DIR = "example_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report(name, mse, tol=MSE_TOL) -> bool:
    passed = mse < tol
    icon = PASS_ICON if passed else FAIL_ICON
    print(f"  {icon}  MSE = {mse:.2e}  (tol {tol:.0e})")
    return passed


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_1d(tag, x, f_pred, f_exact=None,
            xlabel="x", ylabel="f(x)",
            title_pred="", title_err=""):
    """
    Produce and save solution + error plots for a 1-D result.

    Parameters
    ----------
    tag : str
        Filename prefix (e.g. '1_linear_fie').
    x : np.ndarray
        Query points.
    f_pred : np.ndarray
        Predicted solution.
    f_exact : np.ndarray or None
        Exact solution values on the same x.  If None, only the solution
        plot is produced (no error plot).
    """
    # --- Solution plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f_pred, label="FNN prediction", linewidth=1.8)
    if f_exact is not None:
        ax.plot(x, f_exact, "--", label="Exact solution", linewidth=1.4)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_pred or "Predicted vs. exact solution")
    ax.grid(True, alpha=0.3)
    _save(fig, f"{tag}_solution.png")

    # --- Error plot ---
    if f_exact is not None:
        error = np.abs(f_pred - f_exact)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, error, color="C2", linewidth=1.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\hat{f}(x) - f(x)|$")
        ax.set_title(title_err or "Pointwise absolute error")
        ax.grid(True, alpha=0.3)
        _save(fig, f"{tag}_error.png")


def plot_2d(tag, r, theta, u_pred, u_exact=None,
            title_pred="", title_err=""):
    """
    Produce and save contour solution + error plots for a 2-D polar result.
    Axes are (φ, r) directly — matching the paper/notebook style.

    Parameters
    ----------
    tag : str
        Filename prefix.
    r, theta : np.ndarray, 1-D
        Radial and angular grids.
    u_pred : np.ndarray, shape (Nr, Ntheta)
        Predicted solution on the meshgrid (rows = r, cols = theta).
    u_exact : np.ndarray or None
        Exact solution on the same meshgrid.
    """
    # Meshgrid for contourf: x-axis = phi, y-axis = r
    Th, R = np.meshgrid(theta, r, indexing="xy")  # shape (Nr, Ntheta)

    # --- Solution plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    cf = ax.contourf(Th, R, u_pred, levels=50, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\tilde{u}(r,\phi)$", fontsize=12)
    ax.set_xlabel(r"$\phi$", fontsize=13)
    ax.set_ylabel(r"$r$", fontsize=13)
    ax.set_title(title_pred or "Predicted solution")
    _save(fig, f"{tag}_solution.png")

    # --- Error plot ---
    if u_exact is not None:
        error = np.abs(u_pred - u_exact)
        fig, ax = plt.subplots(figsize=(6, 4))
        cf = ax.contourf(Th, R, error, levels=50, cmap="viridis")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"$|\tilde{u}(r,\phi) - u(r,\phi)|$", fontsize=12)
        ax.set_xlabel(r"$\phi$", fontsize=13)
        ax.set_ylabel(r"$r$", fontsize=13)
        ax.set_title(title_err or "Pointwise absolute error")
        _save(fig, f"{tag}_error.png")


# ---------------------------------------------------------------------------
# 1. Linear FIE — contractive operator
# ---------------------------------------------------------------------------
def example_linear_fie():
    print("\n[1] Linear FIE  (contractive, standard Picard iteration)")
    print("    f(x) = sin(x) + ∫_0^{π/2} sin(x)cos(z) f(z) dz")
    print("    Exact solution: f(x) = 2 sin(x)\n")

    from fredholm_nn import solve_linear_fie

    sol = solve_linear_fie(
        kernel      = lambda x, z: np.sin(z) * np.cos(x),
        additive    = lambda x: np.sin(x),
        domain      = (0.0, np.pi / 2),
        n_grid      = 500,
        n_iterations= 15,
        predict_at  = np.linspace(0, 2 * np.pi, 200),
    )

    exact = lambda x: 2 * np.sin(x)
    f_exact = exact(sol.x)

    plot_1d(
        tag="1_linear_fie",
        x=sol.x, f_pred=sol.f, f_exact=f_exact,
        xlabel=r"$x$", ylabel=r"$\hat{f}(x)$",
        title_pred=r"Linear FIE — $f(x) = \sin(x) + \int_0^{\pi/2} \sin(x)\cos(z)f(z)\,dz$",
        title_err="Linear FIE — absolute error",
    )

    return report("linear FIE", sol.mse(exact))


# ---------------------------------------------------------------------------
# 2. Linear FIE — non-expansive operator (Krasnoselskii-Mann)
# ---------------------------------------------------------------------------
def example_linear_fie_km():
    print("\n[2] Linear FIE  (non-expansive, Krasnoselskii-Mann  κ = 0.5)")
    print("    f(x) = sin(25x)+sin(7x) + 0.3∫_0^1 [cos(25(z-x))+cos(7(z-x))] f(z) dz")
    print("    Exact solution: f(x) = (2/(2−0.3)) (sin(25x)+sin(7x))\n")

    from fredholm_nn import solve_linear_fie

    lamb = 0.3
    sol = solve_linear_fie(
        kernel      = lambda x, z: lamb * (np.cos(25*(z-x)) + np.cos(7*(z-x))),
        additive    = lambda x: np.sin(25*x) + np.sin(7*x),
        domain      = (0.0, 1.0),
        n_grid      = 5000,
        n_iterations= 200,
        km_constant = 0.5,
        predict_at  = np.linspace(0, 1, 200),
    )

    exact = lambda x: (2 / (2 - lamb)) * (np.sin(25*x) + np.sin(7*x))
    f_exact = exact(sol.x)

    plot_1d(
        tag="2_linear_fie_km",
        x=sol.x, f_pred=sol.f, f_exact=f_exact,
        xlabel=r"$x$", ylabel=r"$\hat{f}(x)$",
        title_pred=r"Linear FIE (KM, $\kappa=0.5$)",
        title_err=r"Linear FIE (KM) — absolute error",
    )

    return report("linear FIE (KM)", sol.mse(exact), tol=1e-3)


# ---------------------------------------------------------------------------
# 3. Non-linear FIE
# ---------------------------------------------------------------------------
def example_nonlinear_fie():
    print("\n[3] Non-linear FIE  (iterative linearisation)")
    print("    f(x) = cos(x) + 0.1 ∫_0^1 x·z · sin(f(z)) dz")
    print("    No closed-form exact solution; showing predicted solution only.\n")

    from fredholm_nn import solve_nonlinear_fie

    sol = solve_nonlinear_fie(
        kernel       = lambda x, z: 0.1 * x * z,
        additive     = lambda x: np.cos(x),
        nonlinearity = lambda f: np.sin(f),
        domain       = (0.0, 1.0),
        n_grid       = 300,
        n_iterations = 20,
        n_outer      = 8,
        tol          = 1e-6,
        predict_at   = np.linspace(0, 1, 100),
    )

    # No exact solution — plot prediction only
    plot_1d(
        tag="3_nonlinear_fie",
        x=sol.x, f_pred=sol.f, f_exact=None,
        xlabel=r"$x$", ylabel=r"$\hat{f}(x)$",
        title_pred=r"Non-linear FIE — $f(x) = \cos(x) + 0.1\int_0^1 xz\sin(f(z))\,dz$",
    )

    # Self-consistency residual for the pass/fail check
    x = sol.x
    z = np.linspace(0.0, 1.0, 300)
    dz = z[1] - z[0]
    f_interp = np.interp(z, x, sol.f)
    residuals = np.abs(
        sol.f
        - np.cos(x)
        - np.array([np.sum(0.1 * xi * z * np.sin(f_interp)) * dz for xi in x])
    )
    mse_residual = float(np.mean(residuals**2))
    return report("nonlinear FIE (self-consistency)", mse_residual)


# ---------------------------------------------------------------------------
# 4. BVP ODE
# ---------------------------------------------------------------------------
def example_bvp_ode():
    print("\n[4] BVP ODE  (reduced to Fredholm IE via Green's function)")
    print("    y''(x) + 3p/(p+x²)² · y = 0,  y(0)=0,  y(1)=1/√(p+1)")
    print("    Exact solution: y(x) = x / √(p + x²)\n")

    from fredholm_nn import solve_bvp_ode

    p = 3.2
    sol = solve_bvp_ode(
        p_func  = lambda x: 3*p / (p + x**2)**2,
        q_func  = lambda x: np.zeros_like(x),
        alpha   = 0.0,
        beta    = 1.0 / np.sqrt(p + 1.0),
        domain  = (0.0, 1.0),
        n_grid      = 1000,
        n_iterations= 10,
        predict_at  = np.linspace(0, 1, 200),
    )

    exact = lambda x: x / np.sqrt(p + x**2)
    f_exact = exact(sol.x)

    plot_1d(
        tag="4_bvp_ode",
        x=sol.x, f_pred=sol.y, f_exact=f_exact,
        xlabel=r"$x$", ylabel=r"$\tilde{y}(x)$",
        title_pred=r"BVP ODE — $y'' + \frac{3p}{(p+x^2)^2}y = 0$",
        title_err="BVP ODE — absolute error",
    )

    return report("BVP ODE", sol.mse(exact), tol=1e-3)


# ---------------------------------------------------------------------------
# 5. Laplace PDE
# ---------------------------------------------------------------------------
def example_laplace_pde():
    print("\n[5] Laplace PDE  (2-D, unit disc, boundary integral equation)")
    print("    Δu = 0 in Ω,  u = 1 + cos(2θ) on ∂Ω")
    print("    Exact solution: u(r, θ) = 1 + r² cos(2θ)\n")

    from fredholm_nn import solve_laplace

    n_boundary = 500
    # Use the same angular grid as the boundary discretisation (half-open [0, 2π))
    dphi   = 2.0 * np.pi / n_boundary
    phi    = np.arange(0.0, 2.0 * np.pi, dphi)   # shape (500,)
    r_vals = np.linspace(0.0, 1.0, 100)

    sol = solve_laplace(
        boundary_fn  = lambda theta: 1.0 + np.cos(2 * theta),
        n_boundary   = n_boundary,
        n_iterations = 30,
        km_constant  = 0.3,
        r_values     = r_vals,
        theta_values = phi,
    )

    def exact_fn(R, Theta):
        return 1.0 + R**2 * np.cos(2 * Theta)

    R, Th = np.meshgrid(r_vals, phi, indexing="ij")
    u_exact = exact_fn(R, Th)

    plot_2d(
        tag="5_laplace_pde",
        r=r_vals, theta=phi,
        u_pred=sol.u, u_exact=u_exact,
        title_pred=r"Laplace PDE — $\tilde{u}(r,\phi)$",
        title_err=r"Laplace PDE — $|\tilde{u}(r,\phi) - u(r,\phi)|$",
    )

    return report("Laplace PDE", sol.mse(exact_fn), tol=1e-4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  fredholm_nn — example run")
    print(f"  Plots saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)

    examples = [
        example_linear_fie,
        example_linear_fie_km,
        example_nonlinear_fie,
        example_bvp_ode,
        example_laplace_pde,
    ]

    results = []
    for fn in examples:
        try:
            passed = fn()
            results.append(bool(passed))
        except Exception as exc:
            print(f"  [ERROR]  {exc}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"  {passed} / {len(results)} examples passed")
    print(f"  Plots saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)
