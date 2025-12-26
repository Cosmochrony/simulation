#!/usr/bin/env python3
"""
Toy-model Cosmochrony D.3 pipeline (1D, OPTION A):
Periodic lattice + kink–antikink pair (net topological charge 0)

What it does:
1) Build a kink–antikink initial profile on a periodic ring
2) Relax it (gradient descent) to a stationary configuration chi_sol
3) Build the stability operator (Hessian) L_sol = Delta_periodic + diag(V''(chi_sol)) (+ optional pinning)
4) Extract the lowest eigenmodes and diagnose localization via IPR
5) Plot chi_sol, relaxation energy, and first eigenmodes

Dependencies:
  pip install numpy scipy matplotlib

Notes:
- With periodic BC, there is a (near-)zero translation mode. Small negative lambda can occur (discretization).
  Use --pin to lift it (recommended for clean spectra).
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def kink_antikink_periodic_initial(x: np.ndarray, v: float, alpha: float, x1: float, x2: float) -> np.ndarray:
    """
    A standard kink–antikink ansatz on an effectively infinite line:
      chi = v * (tanh(alpha(x-x1)) - tanh(alpha(x-x2)) - 1)

    It goes:
      -v (far left) -> +v between x1 and x2 -> -v (far right),
    which is compatible with periodic BC if the domain is large enough and x1,x2 are well separated from edges.

    On a periodic ring, we simply evaluate this function on the ring coordinates.
    """
    return v * (np.tanh(alpha * (x - x1)) - np.tanh(alpha * (x - x2)) - 1.0)


def energy_and_grad_phi4_periodic(
    chi: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    pinned_indices: list[int] | None = None,
    kappa_pin: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    Energy:
      E = sum_i [ K0/(2a^2) (chi_{i+1}-chi_i)^2 + mu/4 (chi_i^2 - v^2)^2 ]  (periodic i+1 wrap)
        + sum_{pinned i} 0.5*kappa*(chi_i - 0)^2

    Gradient (dE/dchi):
      spring term => K0/a^2 (2 chi_i - chi_{i-1} - chi_{i+1}) with periodic wrap
      potential term => mu * chi_i (chi_i^2 - v^2)
      pin term => kappa * chi_i at pinned sites
    """
    N = chi.size

    # periodic forward differences
    chi_next = np.roll(chi, -1)
    d = chi_next - chi
    E_grad = (K0 / (2.0 * a * a)) * float(np.sum(d * d))

    E_pot = (mu / 4.0) * float(np.sum((chi * chi - v * v) ** 2))

    E_pin = 0.0
    if pinned_indices and kappa_pin > 0.0:
        for idx in pinned_indices:
            E_pin += 0.5 * kappa_pin * float(chi[idx] ** 2)

    E = E_grad + E_pot + E_pin

    # gradient
    chi_prev = np.roll(chi, 1)
    grad = (K0 / (a * a)) * (2.0 * chi - chi_prev - chi_next)
    grad += mu * chi * (chi * chi - v * v)

    if pinned_indices and kappa_pin > 0.0:
        for idx in pinned_indices:
            grad[idx] += kappa_pin * chi[idx]

    return E, grad


def relax_periodic(
    chi0: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    n_steps: int,
    lr: float,
    pinned_indices: list[int] | None = None,
    kappa_pin: float = 0.0,
    tol: float = 1e-10,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Plain gradient descent. For this toy-model, it's robust if lr is small (e.g. 0.01–0.05).
    """
    chi = chi0.copy()
    E_hist: list[float] = []

    for it in range(n_steps):
        E, grad = energy_and_grad_phi4_periodic(
            chi, K0=K0, a=a, mu=mu, v=v, pinned_indices=pinned_indices, kappa_pin=kappa_pin
        )
        E_hist.append(E)

        gnorm = float(np.linalg.norm(grad))
        if verbose and (it % max(1, n_steps // 10) == 0):
            print(f"[relax] it={it:6d}  E={E:.6e}  ||grad||={gnorm:.3e}")

        if gnorm < tol:
            if verbose:
                print(f"[relax] Converged at it={it} with ||grad||={gnorm:.3e}")
            break

        chi -= lr * grad

    return chi, E_hist


def build_stability_operator_periodic(
    chi_sol: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    pinned_indices: list[int] | None = None,
    kappa_pin: float = 0.0,
):
    """
    Hessian (stability operator) around chi_sol:
      L = Delta_periodic + diag(V''(chi_sol)) (+ pin on diagonal)
    where:
      Delta_periodic: main 2K0/a^2, off -K0/a^2 including wrap edges
      V''(chi) = mu*(3 chi^2 - v^2)
    """
    N = chi_sol.size
    Vpp = mu * (3.0 * chi_sol * chi_sol - v * v)

    main = (2.0 * K0 / (a * a)) + Vpp
    off = (-K0 / (a * a)) * np.ones(N - 1, dtype=float)

    if pinned_indices and kappa_pin > 0.0:
        main = main.copy()
        for idx in pinned_indices:
            main[idx] += kappa_pin

    # tri-diagonal plus periodic wrap
    L = diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    wrap = -K0 / (a * a)
    L[0, N - 1] = wrap
    L[N - 1, 0] = wrap
    return L.tocsr()


def ipr(vecs: np.ndarray) -> np.ndarray:
    """Inverse participation ratio for each eigenvector column (assuming normalized)."""
    return np.sum(np.abs(vecs) ** 4, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="1D periodic kink–antikink toy-model + stability spectrum")
    ap.add_argument("--N", type=int, default=2000, help="Number of lattice sites")
    ap.add_argument("--a", type=float, default=1.0, help="Lattice spacing")
    ap.add_argument("--K0", type=float, default=1.0, help="Baseline coupling (stiffness)")
    ap.add_argument("--mu", type=float, default=1.0, help="phi^4 strength")
    ap.add_argument("--v", type=float, default=1.0, help="vacuum value (|chi| in minima)")
    ap.add_argument("--alpha", type=float, default=0.03, help="Initial wall sharpness")
    ap.add_argument("--sep", type=float, default=400.0, help="Kink–antikink separation in x-units")
    ap.add_argument("--steps", type=int, default=8000, help="Relaxation steps")
    ap.add_argument("--lr", type=float, default=0.02, help="Gradient descent learning rate")
    ap.add_argument("--k_eigs", type=int, default=16, help="Number of lowest eigenpairs to compute")
    ap.add_argument("--pin", action="store_true", help="Pin the pair (recommended) to lift translation modes")
    ap.add_argument("--kappa_pin", type=float, default=0.2, help="Pinning strength if --pin")
    ap.add_argument("--no_plots", action="store_true", help="Disable plots")
    args = ap.parse_args()

    N = args.N
    a = args.a
    K0 = args.K0
    mu = args.mu
    v = args.v

    # periodic coordinate (centered)
    L = N * a
    x = (np.arange(N) - N / 2.0) * a

    # place kink and antikink symmetrically around 0
    d = args.sep
    x1 = -0.5 * d
    x2 = +0.5 * d

    chi0 = kink_antikink_periodic_initial(x, v=v, alpha=args.alpha, x1=x1, x2=x2)

    pinned_indices: list[int] | None = None
    if args.pin:
        # Pin near the centers: enforce chi=0 at x ~ x1 and x ~ x2 (wall cores)
        i1 = int(np.argmin(np.abs(x - x1)))
        i2 = int(np.argmin(np.abs(x - x2)))
        pinned_indices = [i1, i2]
        print(f"[pin] pinning indices: i1={i1}, x1~{x[i1]:.3f}; i2={i2}, x2~{x[i2]:.3f}")

    chi_sol, E_hist = relax_periodic(
        chi0=chi0,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        n_steps=args.steps,
        lr=args.lr,
        pinned_indices=pinned_indices,
        kappa_pin=args.kappa_pin if args.pin else 0.0,
        tol=1e-10,
        verbose=True,
    )

    Lsol = build_stability_operator_periodic(
        chi_sol=chi_sol,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        pinned_indices=pinned_indices,
        kappa_pin=args.kappa_pin if args.pin else 0.0,
    )

    k = min(args.k_eigs, N - 2)
    if k < 2:
        print("N too small for requested eigenpairs.", file=sys.stderr)
        return 2

    try:
        vals, vecs = eigsh(Lsol, k=k, which="SA")
    except Exception as e:
        print(f"[spectrum] eigsh failed: {e}", file=sys.stderr)
        print("[spectrum] Try reducing k, increasing N, or enabling --pin.", file=sys.stderr)
        return 3

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    ipr_vals = ipr(vecs)

    print("\nLowest eigenvalues (lambda_n) and IPR (higher=more localized):")
    for i in range(k):
        print(f"  n={i:2d}  lambda={vals[i]: .6e}   IPR={ipr_vals[i]: .6e}")

    if not args.no_plots:
        # soliton
        plt.figure()
        plt.plot(x, chi_sol)
        plt.title("Periodic kink–antikink configuration (chi_sol)")
        plt.xlabel("x")
        plt.ylabel("chi")

        # energy
        plt.figure()
        plt.plot(np.arange(len(E_hist)), E_hist)
        plt.title("Relaxation energy history")
        plt.xlabel("iteration")
        plt.ylabel("E")

        # first few eigenmodes
        n_plot = min(6, k)
        for i in range(n_plot):
            plt.figure()
            plt.plot(x, vecs[:, i])
            plt.title(f"Eigenmode n={i}  lambda={vals[i]:.3e}  IPR={ipr_vals[i]:.3e}")
            plt.xlabel("x")
            plt.ylabel("psi_n(x)")

        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
