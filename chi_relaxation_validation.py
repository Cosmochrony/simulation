#!/usr/bin/env python3
"""
chi_relaxation_validation.py

Numerical validation of the chi -> chi_eff transition (Appendix D.4).

Key idea:
- Evolve a micro field chi on a 3D cubic lattice with a relational local rule:
    S_i = (1/c^2) sum_{j in N(i)} K_ij (chi_i - chi_j)^2
    K_ij = K0 / (1 + (Δchi/chi_c)^2)
    R_i = c * sqrt(max(0, 1 - S_i))        (saturation if S_i > 1)
    chi <- chi + dt * ( R_i + kappa * Laplacian(chi) )

- Define chi_eff by block coarse-graining (block averaging).
- Validate the *coarse-grained micro-dynamics*:
    d/dt chi_eff  ≈  CG( R_micro(chi) + kappa Laplacian(chi) )

This avoids the non-commutation trap:
    CG(R(chi)) != R(CG(chi)) for nonlinear/saturating dynamics.

Outputs:
- fig_D4_epsilon_vs_time_compare.pdf
- fig_D4_residual_hist_smooth.pdf
- fig_D4_chi_eff_slice_smooth.pdf
- fig_D4_residual_slice_smooth.pdf
(and analogous files for rough + nonprojectable)

Failure mode recipe:
- Use large initial amplitude sigma (e.g., 1.5–2.5),
- disable pre-smoothing,
- reduce diffusion kappa (0 or tiny),
- optionally decrease chi_c to make S_i>1 common.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Discrete operators (periodic boundary conditions)
# -----------------------
def laplacian(u: np.ndarray) -> np.ndarray:
    return (
        np.roll(u, 1, 0) + np.roll(u, -1, 0) +
        np.roll(u, 1, 1) + np.roll(u, -1, 1) +
        np.roll(u, 1, 2) + np.roll(u, -1, 2) -
        6.0 * u
    )


def S_local(u: np.ndarray, K0: float, chi_c: float, c: float) -> np.ndarray:
    """
    S_i = (1/c^2) sum_{j in N(i)} K_ij (u_i - u_j)^2
    with 6 nearest neighbors and K_ij = K0 / (1 + (Δu/chi_c)^2).
    """
    dxp = u - np.roll(u, -1, 0)
    dxm = u - np.roll(u,  1, 0)
    dyp = u - np.roll(u, -1, 1)
    dym = u - np.roll(u,  1, 1)
    dzp = u - np.roll(u, -1, 2)
    dzm = u - np.roll(u,  1, 2)

    def K_of(d: np.ndarray) -> np.ndarray:
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


def rhs_micro(u: np.ndarray, c: float, kappa: float, K0: float, chi_c: float) -> np.ndarray:
    """RHS of micro evolution (without dt): R(u) + kappa Laplacian(u)."""
    S = S_local(u, K0=K0, chi_c=chi_c, c=c)
    R = c * np.sqrt(np.clip(1.0 - S, 0.0, None))  # saturation if S>1
    return R + kappa * laplacian(u)


def evolve(u: np.ndarray, dt: float, c: float, kappa: float, K0: float, chi_c: float) -> np.ndarray:
    """Explicit Euler step."""
    return u + dt * rhs_micro(u, c=c, kappa=kappa, K0=K0, chi_c=chi_c)


def coarse_grain_block(u: np.ndarray, b: int) -> np.ndarray:
    """Block-average coarse graining. Requires N divisible by b."""
    N0 = u.shape[0]
    if N0 % b != 0:
        raise ValueError(f"Block size b={b} must divide N={N0}.")
    n = N0 // b
    return u.reshape(n, b, n, b, n, b).mean(axis=(1, 3, 5))


def compute_eps_from_rhs_eff(chi_eff_prev: np.ndarray,
                             chi_eff_curr: np.ndarray,
                             rhs_eff_curr: np.ndarray,
                             dt: float) -> tuple[float, float, np.ndarray]:
    """
    epsilon(t) = || d/dt chi_eff - rhs_eff || / || d/dt chi_eff ||
    using L2 and Linf variants; returns (eps_l2, eps_linf, residual_field).
    """
    dchi_eff_dt = (chi_eff_curr - chi_eff_prev) / dt
    res = dchi_eff_dt - rhs_eff_curr

    l2_res = float(np.sqrt(np.mean(res ** 2)))
    l2_lhs = float(np.sqrt(np.mean(dchi_eff_dt ** 2)))
    eps_l2 = l2_res / (l2_lhs + 1e-12)

    l_inf_res = float(np.max(np.abs(res)))
    l_inf_lhs = float(np.max(np.abs(dchi_eff_dt)))
    eps_linf = l_inf_res / (l_inf_lhs + 1e-12)

    return eps_l2, eps_linf, res


# -----------------------
# Experiment config
# -----------------------
@dataclass
class CaseConfig:
    name: str
    smooth_init: bool
    sigma: float
    kappa: float
    chi_c: float


# -----------------------
# Runner
# -----------------------
def run_case(cfg: CaseConfig,
             N: int,
             steps: int,
             dt: float,
             block: int,
             seed: int,
             c: float,
             K0: float,
             pre_iters: int,
             pre_alpha: float,
             outdir: Path,
             save_fmt: str,
             dpi: int) -> dict:
    rng = np.random.default_rng(seed)

    # Initial condition
    chi = rng.normal(0.0, cfg.sigma, size=(N, N, N)).astype(np.float64)

    # Optional pre-smoothing to reach a projectable-ish regime
    if cfg.smooth_init:
        for _ in range(pre_iters):
            chi = chi + pre_alpha * laplacian(chi)

    eps_t: list[float] = []
    epsinf_t: list[float] = []

    chi_eff_prev = None

    final_res = None
    final_chi_eff = None

    # Track saturation fraction for diagnostics (optional but useful)
    sat_frac_t: list[float] = []

    for t in range(steps):
      # Evolve micro
      chi = evolve(chi, dt=dt, c=c, kappa=cfg.kappa, K0=K0, chi_c=cfg.chi_c)

      # Compute RHS micro
      rhs_curr_micro = rhs_micro(chi, c=c, kappa=cfg.kappa, K0=K0, chi_c=cfg.chi_c)

      # >>> FIX: coarse-grain RHS micro <<<
      rhs_eff_curr = coarse_grain_block(rhs_curr_micro, block)

      # Saturation diagnostics
      S = S_local(chi, K0=K0, chi_c=cfg.chi_c, c=c)
      sat_frac_t.append(float(np.mean(S > 1.0)))

      # Coarse-grain field
      chi_eff_curr = coarse_grain_block(chi, block)

      # Compute epsilon
      if chi_eff_prev is not None:
        eps_l2, eps_linf, res = compute_eps_from_rhs_eff(
          chi_eff_prev,
          chi_eff_curr,
          rhs_eff_curr,
          dt=dt
        )
        eps_t.append(eps_l2)
        epsinf_t.append(eps_linf)

        final_res = res
        final_chi_eff = chi_eff_curr

      chi_eff_prev = chi_eff_curr

    assert final_res is not None and final_chi_eff is not None

    # Print summary
    print(f"\n=== {cfg.name} ===")
    print(f"Micro grid: {N}^3, steps={steps}, dt={dt}, block={block}, seed={seed}")
    print(f"chi_eff grid: {final_chi_eff.shape}")
    print(f"Final epsilon L2 : {eps_t[-1]:.12g}")
    print(f"Final epsilon Linf: {epsinf_t[-1]:.12g}")
    print(f"Final saturation fraction (S>1): {sat_frac_t[-1]:.4f}")

    # --- Figures ---
    t_axis = np.arange(1, len(eps_t) + 1) * dt

    # epsilon(t)
    plt.figure()
    plt.plot(t_axis, eps_t)
    plt.title(f"Epsilon vs time ({cfg.name})")
    plt.xlabel("time")
    plt.ylabel("epsilon (relative L2 residual)")
    plt.tight_layout()
    p = outdir / f"fig_D4_epsilon_vs_time_{cfg.name}.{save_fmt}"
    plt.savefig(p, dpi=dpi if save_fmt == "png" else None)
    plt.close()

    # saturation fraction vs time (useful for failure mode)
    plt.figure()
    plt.plot(np.arange(len(sat_frac_t)) * dt, sat_frac_t)
    plt.title(f"Saturation fraction vs time ({cfg.name})")
    plt.xlabel("time")
    plt.ylabel("fraction of nodes with S>1")
    plt.tight_layout()
    p = outdir / f"fig_D4_saturation_fraction_{cfg.name}.{save_fmt}"
    plt.savefig(p, dpi=dpi if save_fmt == "png" else None)
    plt.close()

    # residual histogram (final)
    plt.figure()
    plt.hist(final_res.ravel(), bins=60)
    plt.title(f"Residual histogram ({cfg.name})")
    plt.xlabel("Residual value")
    plt.ylabel("Count")
    plt.tight_layout()
    p = outdir / f"fig_D4_residual_hist_{cfg.name}.{save_fmt}"
    plt.savefig(p, dpi=dpi if save_fmt == "png" else None)
    plt.close()

    # slices (final)
    mid = final_chi_eff.shape[2] // 2

    plt.figure()
    plt.imshow(final_chi_eff[:, :, mid], origin="lower")
    plt.title(f"chi_eff slice (mid z) ({cfg.name})")
    plt.colorbar()
    plt.tight_layout()
    p = outdir / f"fig_D4_chi_eff_slice_{cfg.name}.{save_fmt}"
    plt.savefig(p, dpi=dpi if save_fmt == "png" else None)
    plt.close()

    plt.figure()
    plt.imshow(final_res[:, :, mid], origin="lower")
    plt.title(f"Residual slice (mid z) ({cfg.name})")
    plt.colorbar()
    plt.tight_layout()
    p = outdir / f"fig_D4_residual_slice_{cfg.name}.{save_fmt}"
    plt.savefig(p, dpi=dpi if save_fmt == "png" else None)
    plt.close()

    return {
        "name": cfg.name,
        "eps_t": np.array(eps_t),
        "epsinf_t": np.array(epsinf_t),
        "sat_frac_t": np.array(sat_frac_t),
        "final_eps_l2": eps_t[-1],
        "final_eps_linf": epsinf_t[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--K0", type=float, default=1.0)

    ap.add_argument("--pre_iters", type=int, default=10)
    ap.add_argument("--pre_alpha", type=float, default=0.2)

    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--save_fmt", type=str, default="pdf", choices=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=200)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Cases ---
    # Projectable-ish
    smooth = CaseConfig(
        name="smooth",
        smooth_init=True,
        sigma=0.2,
        kappa=0.12,
        chi_c=1.0,
    )

    # Less prepared: no pre-smoothing
    rough = CaseConfig(
        name="rough",
        smooth_init=False,
        sigma=0.2,
        kappa=0.12,
        chi_c=1.0,
    )

    # FAILURE MODE recipe:
    # - larger amplitude (sigma)
    # - no pre-smoothing
    # - reduce diffusion (kappa)
    # - smaller chi_c => slopes more likely to saturate (S>1) persistently
    nonprojectable = CaseConfig(
      name="nonprojectable",
      smooth_init=False,
      sigma=2.0,
      kappa=0.0,
      chi_c=10.0,  # empêche K de s'effondrer
    )

    results = []
    for cfg in (smooth, rough, nonprojectable):
        results.append(
            run_case(
                cfg=cfg,
                N=args.N,
                steps=args.steps,
                dt=args.dt,
                block=args.block,
                seed=args.seed,
                c=args.c,
                K0=args.K0,
                pre_iters=args.pre_iters,
                pre_alpha=args.pre_alpha,
                outdir=outdir,
                save_fmt=args.save_fmt,
                dpi=args.dpi,
            )
        )

    # --- Combined comparison plot: epsilon(t) ---
    plt.figure()
    for r in results:
        t_axis = np.arange(1, len(r["eps_t"]) + 1) * args.dt
        plt.plot(t_axis, r["eps_t"], label=r["name"])
    plt.title("Epsilon vs time (comparison)")
    plt.xlabel("time")
    plt.ylabel("epsilon (relative L2 residual)")
    plt.legend()
    plt.tight_layout()
    p = outdir / f"fig_D4_epsilon_vs_time_compare.{args.save_fmt}"
    plt.savefig(p, dpi=args.dpi if args.save_fmt == "png" else None)
    plt.close()

    print(f"\nSaved figures to: {outdir.resolve()}")
    print(f" - {p}")


if __name__ == "__main__":
    main()
