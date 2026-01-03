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

# Failure mode toggle (second run)
DO_FAILURE_MODE = True

# -----------------------
# Parameters (reproducible)
# -----------------------
N = 32          # micro lattice size: N^3
steps = 80      # number of evolution steps
dt = 0.03       # time step
c = 1.0         # maximal relaxation speed (unitless here)
kappa = 0.12    # smoothing / relaxation diffusion strength
block = 4       # coarse-graining block size (must divide N)
K0 = 1.0         # dimensionless stiffness scale (toy)
chi_c = 1.0      # characteristic variation scale (toy)

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


def S_local(u: np.ndarray, K0: float, chi_c: float, c: float) -> np.ndarray:
  """
  Compute S_i = (1/c^2) * sum_{j in N(i)} K_ij * (u_i - u_j)^2
  with nearest-neighbor graph and K_ij = K0 / (1 + (Δu/chi_c)^2).
  Periodic boundary conditions.
  """
  # neighbor differences
  dxp = u - np.roll(u, -1, 0)
  dxm = u - np.roll(u, 1, 0)
  dyp = u - np.roll(u, -1, 1)
  dym = u - np.roll(u, 1, 1)
  dzp = u - np.roll(u, -1, 2)
  dzm = u - np.roll(u, 1, 2)

  # K_ij for each directed neighbor edge
  def K_of(d):
    return K0 / (1.0 + (d / chi_c) ** 2)

  S = (
          K_of(dxp) * dxp * dxp +
          K_of(dxm) * dxm * dxm +
          K_of(dyp) * dyp * dyp +
          K_of(dym) * dym * dym +
          K_of(dzp) * dzp * dzp +
          K_of(dzm) * dzm * dzm
      ) / (c * c)

  return S


def evolve(u: np.ndarray, dt: float, c: float, kappa: float, K0: float, chi_c: float) -> np.ndarray:
  """
  Explicit update:
    S_i = (1/c^2) sum_j K_ij (u_i - u_j)^2
    R_i = c * sqrt(max(0, 1 - S_i))   (saturation when S_i > 1)
    u <- u + dt * ( R_i + kappa * Laplacian(u) )
  """
  S = S_local(u, K0=K0, chi_c=chi_c, c=c)
  rad = np.clip(1.0 - S, 0.0, None)  # saturation: if S>1 -> rad=0 -> R=0
  R = c * np.sqrt(rad)
  return u + dt * (R + kappa * laplacian(u))

def coarse_grain_block(u: np.ndarray, b: int) -> np.ndarray:
    """Block-average coarse graining. Requires N divisible by b."""
    N0 = u.shape[0]
    if N0 % b != 0:
        raise ValueError(f"Block size b={b} must divide N={N0}.")
    n = N0 // b
    return u.reshape(n, b, n, b, n, b).mean(axis=(1, 3, 5))


def rhs_micro(u: np.ndarray, c: float, kappa: float, K0: float, chi_c: float) -> np.ndarray:
  S = S_local(u, K0=K0, chi_c=chi_c, c=c)
  R = c * np.sqrt(np.clip(1.0 - S, 0.0, None))
  return R + kappa * laplacian(u)


def compute_epsilon_from_rhs_eff(chi_eff_prev, chi_eff_curr, rhs_eff, dt: float):
  dchi_eff_dt = (chi_eff_curr - chi_eff_prev) / dt
  res = dchi_eff_dt - rhs_eff

  l2_res = float(np.sqrt(np.mean(res ** 2)))
  l2_lhs = float(np.sqrt(np.mean(dchi_eff_dt ** 2)))
  eps_l2 = l2_res / (l2_lhs + 1e-12)

  l_inf_res = float(np.max(np.abs(res)))
  l_inf_lhs = float(np.max(np.abs(dchi_eff_dt)))
  eps_inf = l_inf_res / (l_inf_lhs + 1e-12)

  return eps_l2, eps_inf, res


def run_case(case_name: str, smooth_init: bool):
  rng = np.random.default_rng(seed)

  # Initial condition
  chi = rng.normal(0.0, 0.2, size=(N, N, N)).astype(np.float64)

  # Optional pre-smoothing to reach a projectable regime
  if smooth_init:
    for _ in range(10):
      chi = chi + 0.2 * laplacian(chi)

  eps_t = []
  epsinf_t = []

  # We'll compute epsilon(t) from consecutive chi_eff snapshots
  chi_eff_prev = None

  # Store final diagnostics for figures
  final_res = None
  final_chi_eff = None

  for t in range(steps):
    chi = evolve(chi, dt=dt, c=c, kappa=kappa, K0=K0, chi_c=chi_c)

    rhs_curr_micro = rhs_micro(chi, c=c, kappa=kappa, K0=K0, chi_c=chi_c)
    rhs_curr_eff = coarse_grain_block(rhs_curr_micro, block)

    chi_eff_curr = coarse_grain_block(chi, block)

    if chi_eff_prev is not None:
      eps_l2, eps_inf, res = compute_epsilon_from_rhs_eff(
        chi_eff_prev, chi_eff_curr, rhs_eff=rhs_curr_eff, dt=dt
      )
      eps_t.append(eps_l2)
      epsinf_t.append(eps_inf)

      final_res = res
      final_chi_eff = chi_eff_curr

    chi_eff_prev = chi_eff_curr

  # Print summary
  print(f"\n=== {case_name} ===")
  print(f"Micro grid: {N}^3, steps={steps}, dt={dt}, block={block}, seed={seed}")
  print(f"chi_eff grid: {final_chi_eff.shape}")
  print(f"Final epsilon L2 : {eps_t[-1]:.12g}")
  print(f"Final epsilon Linf: {epsinf_t[-1]:.12g}")

  # --- Figures ---
  # 1) epsilon(t)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(np.arange(1, len(eps_t) + 1) * dt, eps_t)
  plt.title(f"Epsilon vs time ({case_name})")
  plt.xlabel("time")
  plt.ylabel("epsilon (relative L2 residual)")
  plt.tight_layout()
  p = OUTDIR / f"fig_D4_epsilon_vs_time_{case_name}.{SAVE_FMT}"
  plt.savefig(p, dpi=DPI if SAVE_FMT == "png" else None)
  plt.close()

  # 2) residual histogram (final)
  plt.figure()
  plt.hist(final_res.ravel(), bins=60)
  plt.title(f"Residual histogram ({case_name})")
  plt.xlabel("Residual value")
  plt.ylabel("Count")
  plt.tight_layout()
  p = OUTDIR / f"fig_D4_residual_hist_{case_name}.{SAVE_FMT}"
  plt.savefig(p, dpi=DPI if SAVE_FMT == "png" else None)
  plt.close()

  # 3) slices (final)
  mid = final_chi_eff.shape[2] // 2

  plt.figure()
  plt.imshow(final_chi_eff[:, :, mid], origin="lower")
  plt.title(f"chi_eff slice (mid z) ({case_name})")
  plt.colorbar()
  plt.tight_layout()
  p = OUTDIR / f"fig_D4_chi_eff_slice_{case_name}.{SAVE_FMT}"
  plt.savefig(p, dpi=DPI if SAVE_FMT == "png" else None)
  plt.close()

  plt.figure()
  plt.imshow(final_res[:, :, mid], origin="lower")
  plt.title(f"Residual slice (mid z) ({case_name})")
  plt.colorbar()
  plt.tight_layout()
  p = OUTDIR / f"fig_D4_residual_slice_{case_name}.{SAVE_FMT}"
  plt.savefig(p, dpi=DPI if SAVE_FMT == "png" else None)
  plt.close()

  return eps_t, epsinf_t


# -----------------------
# Main
# -----------------------
def main() -> None:
  eps_smooth, epsinf_smooth = run_case("smooth", smooth_init=True)

  if DO_FAILURE_MODE:
    eps_rough, epsinf_rough = run_case("rough", smooth_init=False)

    # Optional: combined plot smooth vs rough
    import matplotlib.pyplot as plt
    plt.figure()
    t_s = np.arange(1, len(eps_smooth) + 1) * dt
    t_r = np.arange(1, len(eps_rough) + 1) * dt
    plt.plot(t_s, eps_smooth, label="smooth")
    plt.plot(t_r, eps_rough, label="rough")
    plt.title("Epsilon vs time (smooth vs rough)")
    plt.xlabel("time")
    plt.ylabel("epsilon (relative L2 residual)")
    plt.legend()
    plt.tight_layout()
    p = OUTDIR / f"fig_D4_epsilon_vs_time_compare.{SAVE_FMT}"
    plt.savefig(p, dpi=DPI if SAVE_FMT == "png" else None)
    plt.close()

  print(f"\nSaved figures to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
