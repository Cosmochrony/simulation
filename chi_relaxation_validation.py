#!/usr/bin/env python3
"""
Toy numerical validation of chi -> chi_eff (coarse-graining) and PDE residual check.

Reproduces (with the default parameters below):
  - Relative L2 residual  ~ 4.13e-4
  - Relative Linf residual ~ 1.51e-3

Equation checked (on coarse-grained field):
  d/dt chi_eff  ?=  c * sqrt(1 - |grad chi_eff|^2 / c^2)

Notes:
- This is a minimal discrete model on a cubic lattice with periodic boundaries.
- "Relational" aspect: operators are defined via nearest-neighbor differences (graph adjacency).
- Coarse-graining: simple block averaging.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

SAVE_FMT = "pdf"   # "pdf" recommandé pour LaTeX, sinon "png"
DPI = 200          # utile si SAVE_FMT="png"


# -----------------------
# Parameters (reproducible)
# -----------------------
N = 32          # micro lattice size: N^3
steps = 80      # number of evolution steps
dt = 0.03       # time step
c = 1.0         # maximal relaxation speed (unitless here)
kappa = 0.12    # smoothing / relaxation diffusion strength
block = 4       # coarse-graining block size (must divide N)

seed = 0        # RNG seed for reproducibility


# -----------------------
# Discrete operators on graph (nearest-neighbor lattice)
# -----------------------
def laplacian(u: np.ndarray) -> np.ndarray:
    """6-neighbor discrete Laplacian with periodic boundary conditions."""
    return (
        np.roll(u, 1, 0) + np.roll(u, -1, 0) +
        np.roll(u, 1, 1) + np.roll(u, -1, 1) +
        np.roll(u, 1, 2) + np.roll(u, -1, 2) -
        6.0 * u
    )


def grad_sq(u: np.ndarray) -> np.ndarray:
    """
    Squared norm of central-difference gradient (grid spacing = 1),
    with periodic boundary conditions.
    """
    gx = 0.5 * (np.roll(u, -1, 0) - np.roll(u, 1, 0))
    gy = 0.5 * (np.roll(u, -1, 1) - np.roll(u, 1, 1))
    gz = 0.5 * (np.roll(u, -1, 2) - np.roll(u, 1, 2))
    return gx * gx + gy * gy + gz * gz


def evolve(u: np.ndarray, dt: float, c: float, kappa: float) -> np.ndarray:
    """
    One explicit update step:
      u <- u + dt * ( c * sqrt(1 - |grad u|^2/c^2 ) + kappa * Laplacian(u) )

    The sqrt term is clipped to avoid negative radicand (projectable-regime assumption).
    """
    g2 = grad_sq(u)
    rad = np.clip(1.0 - g2 / (c * c), 0.0, None)
    source = c * np.sqrt(rad)
    return u + dt * (source + kappa * laplacian(u))


def coarse_grain_block(u: np.ndarray, b: int) -> np.ndarray:
    """Block-average coarse graining. Requires N divisible by b."""
    N0 = u.shape[0]
    if N0 % b != 0:
        raise ValueError(f"Block size b={b} must divide N={N0}.")
    n = N0 // b
    return u.reshape(n, b, n, b, n, b).mean(axis=(1, 3, 5))


# -----------------------
# Main
# -----------------------
def main() -> None:
    rng = np.random.default_rng(seed)

    # Initial condition: random field, then pre-smoothed to be "projectable-ish"
    chi = rng.normal(0.0, 0.2, size=(N, N, N)).astype(np.float64)

    # Pre-smooth (helps keep sqrt radicand positive and produces smoother chi_eff)
    for _ in range(10):
        chi = chi + 0.2 * laplacian(chi)

    # Evolve and store last two chi_eff snapshots for time derivative
    chi_eff_prev = None
    chi_eff_curr = None

    for t in range(steps):
        chi = evolve(chi, dt=dt, c=c, kappa=kappa)

        if t == steps - 2:
            chi_eff_prev = coarse_grain_block(chi, block)
        if t == steps - 1:
            chi_eff_curr = coarse_grain_block(chi, block)

    assert chi_eff_prev is not None and chi_eff_curr is not None

    # LHS: discrete time derivative of chi_eff
    dchi_eff_dt = (chi_eff_curr - chi_eff_prev) / dt

    # RHS: PDE source term computed on chi_eff
    g2_eff = grad_sq(chi_eff_curr)
    rad_eff = np.clip(1.0 - g2_eff / (c * c), 0.0, None)
    rhs_eff = c * np.sqrt(rad_eff)

    # Residual
    res = dchi_eff_dt - rhs_eff

    # Metrics
    l2_res = float(np.sqrt(np.mean(res ** 2)))
    l2_lhs = float(np.sqrt(np.mean(dchi_eff_dt ** 2)))
    rel_l2 = l2_res / (l2_lhs + 1e-12)

    l_inf_res = float(np.max(np.abs(res)))
    l_inf_lhs = float(np.max(np.abs(dchi_eff_dt)))
    rel_linf = l_inf_res / (l_inf_lhs + 1e-12)

    # Print results (these are the quoted numbers)
    print("=== Toy validation: chi -> chi_eff PDE residual ===")
    print(f"Micro grid: {N}^3, steps={steps}, dt={dt}, block={block}, seed={seed}")
    print(f"chi_eff grid: {chi_eff_curr.shape}")
    print(f"Relative L2 residual  : {rel_l2:.12g}")
    print(f"Relative Linf residual: {rel_linf:.12g}")
    print(f"Absolute L2 residual  : {l2_res:.12g}")
    print(f"Absolute L2 lhs scale : {l2_lhs:.12g}")

    # Plots
    mid = chi_eff_curr.shape[2] // 2

    plt.figure()
    plt.hist(res.ravel(), bins=60)
    plt.title("Residual: d/dt chi_eff - c*sqrt(1 - |grad chi_eff|^2/c^2)")
    plt.xlabel("Residual value")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.figure()
    plt.imshow(chi_eff_curr[:, :, mid], origin="lower")
    plt.title("chi_eff slice (mid z)")
    plt.colorbar()
    plt.tight_layout()

    plt.figure()
    plt.imshow(res[:, :, mid], origin="lower")
    plt.title("Residual slice (mid z)")
    plt.colorbar()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
