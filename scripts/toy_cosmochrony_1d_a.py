#!/usr/bin/env python3
"""
Cosmochrony toy-model (1D, Option A): periodic lattice + kink–antikink pair.

Pipeline:
1) Build kink–antikink initial profile on a periodic ring
2) Relax it via gradient descent to a stationary configuration chi_sol
3) Build stability operator (Hessian): L_sol = Delta_periodic + diag(V''(chi_sol)) (+ optional pinning)
4) Compute lowest eigenpairs with sparse eigensolver (ARPACK via scipy.sparse.linalg.eigsh)
5) Diagnose localization via IPR and (optionally) plot chi_sol, energy history, and eigenmodes

Dependencies:
  pip install numpy scipy matplotlib

Tips:
- Perfect degeneracies (doublets) are expected when the two solitons are equivalent and symmetry is strong.
  Use --pin_mode one or --pin_mode asym to break symmetry and reveal splittings.
- If splittings are extremely small, use --sigma (shift-invert) around the target eigenvalue (e.g. 1.48).
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def kink_antikink_initial(x: np.ndarray, v: float, alpha: float, x1: float, x2: float) -> np.ndarray:
    """
    Kink–antikink ansatz (phi^4-style) on an effectively infinite line:
      chi = v * (tanh(alpha(x-x1)) - tanh(alpha(x-x2)) - 1)

    It goes:
      -v far left -> +v between x1 and x2 -> -v far right.

    On a periodic ring, this is acceptable if the domain is large enough and the walls are not too close.
    """
    return v * (np.tanh(alpha * (x - x1)) - np.tanh(alpha * (x - x2)) - 1.0)


def energy_and_grad_phi4_periodic(
    chi: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    pinned_kappa: dict[int, float] | None = None,
) -> tuple[float, np.ndarray]:
    """
    Energy (toy effective functional):
      E = sum_i [ K0/(2a^2) (chi_{i+1}-chi_i)^2 + mu/4 (chi_i^2 - v^2)^2 ]  with periodic wrap
        + sum_{i in pinned} 0.5*kappa_i * chi_i^2

    Gradient:
      dE/dchi_i = K0/a^2 (2 chi_i - chi_{i-1} - chi_{i+1}) + mu*chi_i(chi_i^2 - v^2) + kappa_i*chi_i
      with periodic indices.
    """
    # Periodic neighbor values
    chi_next = np.roll(chi, -1)
    chi_prev = np.roll(chi, 1)

    # Spring (gradient) energy
    d = chi_next - chi
    E_grad = (K0 / (2.0 * a * a)) * float(np.sum(d * d))

    # Potential energy
    E_pot = (mu / 4.0) * float(np.sum((chi * chi - v * v) ** 2))

    # Pinning energy
    E_pin = 0.0
    if pinned_kappa:
        for idx, kap in pinned_kappa.items():
            if kap > 0.0:
                E_pin += 0.5 * kap * float(chi[idx] ** 2)

    E = E_grad + E_pot + E_pin

    # Gradient of the energy
    grad = (K0 / (a * a)) * (2.0 * chi - chi_prev - chi_next)
    grad += mu * chi * (chi * chi - v * v)

    if pinned_kappa:
        for idx, kap in pinned_kappa.items():
            if kap > 0.0:
                grad[idx] += kap * chi[idx]

    return E, grad


def relax_periodic(
    chi0: np.ndarray,
    K0: float,
    a: float,
    mu: float,
    v: float,
    n_steps: int,
    lr: float,
    pinned_kappa: dict[int, float] | None = None,
    tol: float = 1e-10,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Plain gradient descent relaxation on a periodic lattice.
    """
    chi = chi0.copy()
    E_hist: list[float] = []

    for it in range(n_steps):
        E, grad = energy_and_grad_phi4_periodic(
            chi, K0=K0, a=a, mu=mu, v=v, pinned_kappa=pinned_kappa
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
    pinned_kappa: dict[int, float] | None = None,
):
    """
    Stability operator (Hessian) around chi_sol:
      L = Delta_periodic + diag(V''(chi_sol)) (+ pin on diagonal)

    For the toy phi^4 potential:
      V(chi) = mu/4 (chi^2 - v^2)^2
      => V''(chi) = mu (3 chi^2 - v^2)

    Delta_periodic on 1D ring with nearest neighbors:
      main += 2K0/a^2
      off  += -K0/a^2 including wrap-around (0<->N-1)
    """
    N = chi_sol.size
    Vpp = mu * (3.0 * chi_sol * chi_sol - v * v)

    main = (2.0 * K0 / (a * a)) + Vpp
    off = (-K0 / (a * a)) * np.ones(N - 1, dtype=float)

    if pinned_kappa:
        main = main.copy()
        for idx, kap in pinned_kappa.items():
            if kap > 0.0:
                main[idx] += kap

    # Tridiagonal + periodic wrap
    L = diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    wrap = -K0 / (a * a)
    L[0, N - 1] = wrap
    L[N - 1, 0] = wrap

    return L.tocsr()


def ipr(vecs: np.ndarray) -> np.ndarray:
    """
    Inverse participation ratio for each eigenvector column (assuming normalized):
      IPR = sum_i |psi_i|^4
    Larger => more localized.
    """
    return np.sum(np.abs(vecs) ** 4, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="1D periodic kink–antikink toy-model + stability spectrum")
    ap.add_argument("--N", type=int, default=2000, help="Number of lattice sites")
    ap.add_argument("--a", type=float, default=1.0, help="Lattice spacing")
    ap.add_argument("--K0", type=float, default=1.0, help="Baseline coupling (stiffness)")
    ap.add_argument("--mu", type=float, default=1.0, help="phi^4 strength")
    ap.add_argument("--v", type=float, default=1.0, help="Vacuum value (minima at +-v)")
    ap.add_argument("--alpha", type=float, default=0.03, help="Initial wall sharpness")
    ap.add_argument("--sep", type=float, default=400.0, help="Kink–antikink separation in x-units")
    ap.add_argument("--steps", type=int, default=8000, help="Relaxation steps")
    ap.add_argument("--lr", type=float, default=0.02, help="Gradient descent learning rate")
    ap.add_argument("--k_eigs", type=int, default=16, help="Number of eigenpairs to compute")

    ap.add_argument("--pin", action="store_true", help="Enable pinning constraints (recommended)")
    ap.add_argument(
        "--pin_mode",
        choices=["both", "one", "asym"],
        default="both",
        help="Pin mode: both centers (default), one center (break symmetry), or asym (two different kappas).",
    )
    ap.add_argument("--kappa_pin", type=float, default=0.2, help="Primary pinning strength (kappa1)")
    ap.add_argument("--kappa_pin2", type=float, default=0.18, help="Secondary pinning strength (kappa2, for pin_mode=asym)")

    ap.add_argument("--sigma", type=float, default=None, help="Shift-invert target sigma for eigsh (e.g. 1.48)")
    ap.add_argument("--no_plots", action="store_true", help="Disable plots")
    args = ap.parse_args()

    N = args.N
    a = args.a
    K0 = args.K0
    mu = args.mu
    v = args.v

    # Periodic coordinate (centered)
    x = (np.arange(N) - N / 2.0) * a

    # Place kink and antikink symmetrically around 0
    d = float(args.sep)
    x1 = -0.5 * d
    x2 = +0.5 * d

    chi0 = kink_antikink_initial(x, v=v, alpha=args.alpha, x1=x1, x2=x2)

    pinned_kappa: dict[int, float] | None = None
    if args.pin:
        i1 = int(np.argmin(np.abs(x - x1)))
        i2 = int(np.argmin(np.abs(x - x2)))

        if args.pin_mode == "both":
            pinned_kappa = {i1: args.kappa_pin, i2: args.kappa_pin}
        elif args.pin_mode == "one":
            pinned_kappa = {i1: args.kappa_pin}
        elif args.pin_mode == "asym":
            pinned_kappa = {i1: args.kappa_pin, i2: args.kappa_pin2}
        else:
            raise ValueError(f"Unknown pin_mode: {args.pin_mode}")

        pin_str = ", ".join([f"{idx}:{kap:g}" for idx, kap in pinned_kappa.items()])
        print(f"[pin] indices and kappas: {pin_str}  (i1={i1}, x~{x[i1]:.3f}; i2={i2}, x~{x[i2]:.3f})")

    chi_sol, E_hist = relax_periodic(
        chi0=chi0,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        n_steps=args.steps,
        lr=args.lr,
        pinned_kappa=pinned_kappa,
        tol=1e-10,
        verbose=True,
    )

    Lsol = build_stability_operator_periodic(
        chi_sol=chi_sol,
        K0=K0,
        a=a,
        mu=mu,
        v=v,
        pinned_kappa=pinned_kappa,
    )

    k = min(int(args.k_eigs), N - 2)
    if k < 2:
        print("N too small for requested eigenpairs.", file=sys.stderr)
        return 2

    try:
        if args.sigma is None:
            vals, vecs = eigsh(Lsol, k=k, which="SA")
        else:
            # shift-invert around sigma (good for micro-splittings)
            vals, vecs = eigsh(Lsol, k=k, sigma=float(args.sigma), which="LM")
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
        # Soliton profile
        plt.figure()
        plt.plot(x, chi_sol)
        plt.title("Periodic kink–antikink configuration (chi_sol)")
        plt.xlabel("x")
        plt.ylabel("chi")

        # Energy history
        plt.figure()
        plt.plot(np.arange(len(E_hist)), E_hist)
        plt.title("Relaxation energy history")
        plt.xlabel("iteration")
        plt.ylabel("E")

        # First few eigenmodes
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
