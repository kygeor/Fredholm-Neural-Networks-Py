"""
run_examples.py
---------------
Runs one example of every solver, prints the MSE against the known exact
solution, and displays two figures per example:

  - predicted vs. exact solution overlay
  - pointwise absolute error

For the non-linear FIE (no closed-form exact), only the solution plot is
produced.  For the Laplace PDE the plots are 2-D contour maps.

Run from any directory:
    python run_examples.py
"""

import numpy as np
import matplotlib.pyplot as plt

PASS_ICON = "[PASS]"
FAIL_ICON = "[FAIL]"
MSE_TOL   = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report(name, mse, tol=MSE_TOL) -> bool:
    passed = mse < tol
    icon = PASS_ICON if passed else FAIL_ICON
    print(f"  {icon}  MSE = {mse:.2e}  (tol {tol:.0e})")
    return passed


def plot_1d(x, f_pred, f_exact=None,
            xlabel="x", ylabel="f(x)",
            title_pred="", title_err=""):
    """Display solution + error plots for a 1-D result."""
    # --- Solution plot ---
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f_pred, label="FNN prediction", linewidth=1.8)
    if f_exact is not None:
        ax.plot(x, f_exact, "--", label="Exact solution", linewidth=1.4)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_pred or "Predicted vs. exact solution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Error plot ---
    if f_exact is not None:
        error = np.abs(f_pred - f_exact)
        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, error, color="C2", linewidth=1.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\hat{f}(x) - f(x)|$")
        ax.set_title(title_err or "Pointwise absolute error")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_2d(r, theta, u_pred, u_exact=None,
            title_pred="", title_err=""):
    """Display contour solution + error plots for a 2-D polar result."""
    Th, R = np.meshgrid(theta, r, indexing="xy")  # shape (Nr, Ntheta)

    # --- Solution plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    cf = ax.contourf(Th, R, u_pred, levels=50, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\tilde{u}(r,\phi)$", fontsize=12)
    ax.set_xlabel(r"$\phi$", fontsize=13)
    ax.set_ylabel(r"$r$", fontsize=13)
    ax.set_title(title_pred or "Predicted solution")
    plt.tight_layout()
    plt.show()

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
        plt.tight_layout()
        plt.show()


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
        n_grid      = 2000,
        n_iterations= 200,
        km_constant = 0.5,
        predict_at  = np.linspace(0, 1, 200),
    )

    exact = lambda x: (2 / (2 - lamb)) * (np.sin(25*x) + np.sin(7*x))
    f_exact = exact(sol.x)

    plot_1d(
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

    plot_1d(
        x=sol.x, f_pred=sol.f, f_exact=None,
        xlabel=r"$x$", ylabel=r"$\hat{f}(x)$",
        title_pred=r"Non-linear FIE — $f(x) = \cos(x) + 0.1\int_0^1 xz\sin(f(z))\,dz$",
    )

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
        n_grid      = 5000,
        n_iterations= 20,
        predict_at  = np.linspace(0, 1, 200),
    )

    exact = lambda x: x / np.sqrt(p + x**2)
    f_exact = exact(sol.x)

    plot_1d(
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
    dphi   = 2.0 * np.pi / n_boundary
    phi    = np.arange(0.0, 2.0 * np.pi, dphi)
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
    print("=" * 60)
