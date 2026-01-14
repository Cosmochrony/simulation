#!/usr/bin/env python3
"""
Toy-model Cosmochrony D.3 pipeline (1D):
- Build a discrete kink (phi^4) soliton by energy relaxation
- Build the stability operator L_sol = Delta + diag(V''(chi_sol))
- Extract the lowest eigenmodes with sparse eigensolver (ARPACK via scipy.sparse.linalg.eigsh)
- Diagnose localization (IPR) and plot the soliton + first modes

Dependencies:
  pip install numpy scipy matplotlib
"""

import argparse
import sys
import numpy as np

# Matplotlib: do NOT set styles or colors (per your preference & tool guidelines)
import matplotlib.pyplot as plt

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh


def build_kink_initial(x: np.ndarray, v: float, alpha: float) -> np.ndarray:
    """Analytic kink-like initial guess."""
    return v * np.tanh(alpha * x)


def energy_and_grad_phi4_1d(
    chi: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    pinned_index: int | None = None,
    kappa_pin: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    Effective coarse-grained energy (toy-model):
      E = sum_i [ K0/(2a^2) (chi_{i+1}-chi_i)^2 + mu/4 (chi_i^2 - v^2)^2 ] + pin

    Returns:
      (E, grad dE/dchi)
    """
    N = chi.size

    # Gradient term
    d = chi[1:] - chi[:-1]  # length N-1
    E_grad = (K0 / (2.0 * a * a)) * np.sum(d * d)

    # Potential term
    E_pot = (mu / 4.0) * np.sum((chi * chi - v * v) ** 2)

    # Pinning term (optional)
    E_pin = 0.0
    if pinned_index is not None and kappa_pin > 0.0:
        E_pin = 0.5 * kappa_pin * (chi[pinned_index] - 0.0) ** 2

    E = float(E_grad + E_pot + E_pin)

    # Compute gradient dE/dchi
    grad = np.zeros_like(chi)

    # Gradient contribution: discrete Laplacian from springs
    # For i=1..N-2: grad_i += K0/a^2 (2 chi_i - chi_{i-1} - chi_{i+1})
    grad[1:-1] += (K0 / (a * a)) * (2.0 * chi[1:-1] - chi[:-2] - chi[2:])

    # Potential contribution: d/dchi [mu/4 (chi^2 - v^2)^2] = mu * chi * (chi^2 - v^2)
    grad += mu * chi * (chi * chi - v * v)

    # Pinning contribution
    if pinned_index is not None and kappa_pin > 0.0:
        grad[pinned_index] += kappa_pin * (chi[pinned_index] - 0.0)

    return E, grad


def relax_to_soliton(
    chi0: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    n_steps: int,
    lr: float,
    bc: str,
    pinned_index: int | None = None,
    kappa_pin: float = 0.0,
    tol: float = 1e-10,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Gradient descent relaxation to a stationary configuration.

    Boundary conditions:
      - "dirichlet-kink": chi[0]=-v, chi[-1]=+v held fixed (recommended for kink)
      - "periodic": periodic chain (no fixed boundaries)
    """
    chi = chi0.copy()
    N = chi.size
    E_hist: list[float] = []

    def apply_bc_in_place(arr: np.ndarray) -> None:
        if bc == "dirichlet-kink":
            arr[0] = -v
            arr[-1] = +v
        elif bc == "periodic":
            pass
        else:
            raise ValueError(f"Unknown bc: {bc}")

    apply_bc_in_place(chi)

    for it in range(n_steps):
        E, grad = energy_and_grad_phi4_1d(
            chi, K0=K0, a=a, mu=mu, v=v, pinned_index=pinned_index, kappa_pin=kappa_pin
        )
        E_hist.append(E)

        # Stop criterion: small gradient (excluding fixed boundaries)
        if bc == "dirichlet-kink":
            grad_eff = grad[1:-1]
        else:
            grad_eff = grad

        gnorm = float(np.linalg.norm(grad_eff))
        if verbose and (it % max(1, n_steps // 10) == 0):
            print(f"[relax] it={it:6d}  E={E:.6e}  ||grad||={gnorm:.3e}")

        if gnorm < tol:
            if verbose:
                print(f"[relax] Converged at it={it} with ||grad||={gnorm:.3e}")
            break

        # Gradient descent update (simple, robust for toy-model)
        chi = chi - lr * grad
        apply_bc_in_place(chi)

    return chi, E_hist


def build_stability_operator(
    chi_sol: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    bc: str,
    pinned_index: int | None = None,
    kappa_pin: float = 0.0,
) -> csr_matrix:
    """
    Stability operator (Hessian) around chi_sol for the toy-model:
      L_sol = Delta + diag(V''(chi_sol)) (+ pin)
    where:
      Delta on 1D chain with nearest neighbors (springs),
      V''(chi) = mu * (3 chi^2 - v^2)
    """
    N = chi_sol.size
    Vpp = mu * (3.0 * chi_sol * chi_sol - v * v)

    # Discrete Laplacian part from gradient energy:
    # For interior points, Delta has:
    #   main:  2K0/a^2
    #   off:  -K0/a^2
    # Periodic adjusts corners.
    main = (2.0 * K0 / (a * a)) + Vpp
    off = (-K0 / (a * a)) * np.ones(N - 1, dtype=float)

    # Pinning adds to diagonal
    if pinned_index is not None and kappa_pin > 0.0:
        main = main.copy()
        main[pinned_index] += kappa_pin

    if bc == "dirichlet-kink":
        # We'll keep the operator on all nodes, but note:
        # boundary nodes are fixed in the relaxation; for spectral analysis,
        # we can still include them, but the lowest modes can be dominated by boundaries.
        # To avoid confusion, users may later restrict to interior nodes (see below).
        L = diags([off, main, off], offsets=[-1, 0, 1], format="csr")
        return L

    if bc == "periodic":
        # Add wrap-around couplings
        L = diags([off, main, off], offsets=[-1, 0, 1], format="lil")
        L[0, N - 1] = -K0 / (a * a)
        L[N - 1, 0] = -K0 / (a * a)
        return L.tocsr()

    raise ValueError(f"Unknown bc: {bc}")


def restrict_to_interior(L: csr_matrix, start: int = 1, end: int = -1) -> csr_matrix:
    """Restrict an operator to interior indices [start, end) to avoid boundary artifacts."""
    N = L.shape[0]
    if end < 0:
        end = N + end
    idx = np.arange(start, end, dtype=int)
    # Efficient submatrix extraction
    return L[idx, :][:, idx].tocsr()


def ipr(vecs: np.ndarray) -> np.ndarray:
    """
    Inverse Participation Ratio for each eigenvector column:
      IPR = sum_i |psi_i|^4  (assuming normalized vectors)
    Larger IPR => more localized.
    """
    return np.sum(np.abs(vecs) ** 4, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="1D kink toy-model: soliton + stability spectrum")
    ap.add_argument("--N", type=int, default=2000, help="Number of lattice sites")
    ap.add_argument("--a", type=float, default=1.0, help="Lattice spacing")
    ap.add_argument("--K0", type=float, default=1.0, help="Baseline coupling (stiffness)")
    ap.add_argument("--mu", type=float, default=1.0, help="phi^4 strength")
    ap.add_argument("--v", type=float, default=1.0, help="vacuum value (kink from -v to +v)")
    ap.add_argument("--alpha", type=float, default=0.03, help="Initial kink sharpness")
    ap.add_argument("--steps", type=int, default=4000, help="Relaxation steps")
    ap.add_argument("--lr", type=float, default=0.05, help="Gradient descent learning rate")
    ap.add_argument("--bc", choices=["dirichlet-kink", "periodic"], default="dirichlet-kink")
    ap.add_argument("--pin", action="store_true", help="Enable pinning to suppress translation mode")
    ap.add_argument("--kappa_pin", type=float, default=0.1, help="Pinning strength (if --pin)")
    ap.add_argument("--k_eigs", type=int, default=10, help="Number of lowest eigenpairs to compute")
    ap.add_argument("--interior", action="store_true", help="Restrict stability operator to interior nodes")
    ap.add_argument("--no_plots", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    N = args.N
    a = args.a
    K0 = args.K0
    mu = args.mu
    v = args.v

    x = (np.arange(N) - N / 2.0) * a

    chi0 = build_kink_initial(x, v=v, alpha=args.alpha)

    pinned_index = None
    kappa_pin = 0.0
    if args.pin:
        pinned_index = N // 2  # pin at the center
        kappa_pin = args.kappa_pin

    chi_sol, E_hist = relax_to_soliton(
        chi0=chi0,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        n_steps=args.steps,
        lr=args.lr,
        bc=args.bc,
        pinned_index=pinned_index,
        kappa_pin=kappa_pin,
        tol=1e-10,
        verbose=True,
    )

    L = build_stability_operator(
        chi_sol=chi_sol,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        bc=args.bc,
        pinned_index=pinned_index,
        kappa_pin=kappa_pin,
    )

    if args.interior and args.bc == "dirichlet-kink":
        # Remove boundary nodes which are fixed and can introduce artifacts in modes
        L_eff = restrict_to_interior(L, start=1, end=-1)
        x_eff = x[1:-1]
        chi_eff = chi_sol[1:-1]
        print("[spectrum] Using interior restriction (Dirichlet boundaries removed).")
    else:
        L_eff = L
        x_eff = x
        chi_eff = chi_sol

    # Compute lowest eigenpairs
    k = min(args.k_eigs, L_eff.shape[0] - 2)  # eigsh needs k < N
    if k < 2:
        print("N too small for requested eigenpairs.", file=sys.stderr)
        return 2

    # For a positive semi-definite operator, smallest algebraic eigenvalues are of interest.
    # Use 'SA' (smallest algebraic). If convergence is slow, you can switch to shift-invert.
    try:
        vals, vecs = eigsh(L_eff, k=k, which="SA")
    except Exception as e:
        print(f"[spectrum] eigsh failed with {e}", file=sys.stderr)
        print("[spectrum] Try reducing k, increasing N, or using --pin and/or --interior.", file=sys.stderr)
        return 3

    # Sort ascending
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    ipr_vals = ipr(vecs)

    print("\nLowest eigenvalues (lambda_n) and IPR (higher=more localized):")
    for i in range(k):
        print(f"  n={i:2d}  lambda={vals[i]: .6e}   IPR={ipr_vals[i]: .6e}")

    # Optional plots
    if not args.no_plots:
        # Plot soliton profile
        plt.figure()
        plt.plot(x_eff, chi_eff)
        plt.title("Soliton configuration (chi_sol)")
        plt.xlabel("x")
        plt.ylabel("chi")

        # Plot energy history
        plt.figure()
        plt.plot(np.arange(len(E_hist)), E_hist)
        plt.title("Relaxation energy history")
        plt.xlabel("iteration")
        plt.ylabel("E")

        # Plot first few eigenmodes (magnitude)
        n_plot = min(5, k)
        for i in range(n_plot):
            plt.figure()
            plt.plot(x_eff, vecs[:, i])
            plt.title(f"Eigenmode n={i}  lambda={vals[i]:.3e}  IPR={ipr_vals[i]:.3e}")
            plt.xlabel("x")
            plt.ylabel("psi_n(x)")

        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
