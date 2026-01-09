#!/usr/bin/env python3
"""
compare_mc_vs_weighted_laplacian_hopf_fiberbase_split.py

Monte-Carlo vs Spectral response on S^3 (Hopf fibration S^1 -> S^3 -> S^2),
with explicit fiber/base separation.

- We represent points on S^3 as unit quaternions q = (w, x, y, z).
- Hopf map h: S^3 -> S^2 is computed via q * i * q^{-1} where i=(0,1,0,0).
  The resulting pure imaginary quaternion gives a unit vector in R^3 (S^2).
- Fiber angle is the phase along the S^1 fiber in the standard complex coords:
    z1 = w + i x,  z2 = y + i z
  fiber phase ~ arg(z1) + arg(z2)  (gauge-fixed relative phase).

We define:
- d_base^2 : squared chordal distance on S^2 between Hopf images
- d_fiber^2: squared circular distance on S^1 between fiber phases

Anisotropic kernel (shared by MC and Spectral):
    K_α(i,j) = exp( - ( d_base^2 + a(α)*d_fiber^2 ) / (2 σ^2) )
where a(α) = exp(-max(α,0))  (fiber "relaxes" less when α>0)
This choice produces a plateau for α<=0 and growth for α>0.

Monte-Carlo:
- sample pairs (i,j) from the biased distribution proportional to K_α(i,j)
  using acceptance-rejection over uniformly sampled pairs
- estimate E_base = E[d_base^2], E_fiber = E[d_fiber^2], ratio = E_fiber / E_base

Spectral:
- build kNN graph on S^3 (Euclidean in R^4) for locality
- edge weights w_ij = K_α(i,j)
- define weighted "energies" via Dirichlet form:
    E_base_spec = (sum_{i<j} w_ij * d_base^2) / (sum_{i<j} w_ij)
    E_fiber_spec = (sum_{i<j} w_ij * d_fiber^2) / (sum_{i<j} w_ij)
  ratio_spec = E_fiber_spec / E_base_spec
This is not "eigenvalue ratio" but a Laplacian-consistent kernel-weighted
energy ratio (Dirichlet-form style), which is what matches the MC kernel.

Outputs:
- ratio curves with 95% CI (over repeats)
- optional fiber/base energies curves

Author: (you)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Geometry: S^3, Hopf map, fiber phase
# -----------------------------

def sample_s3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on S^3 via normalized Gaussian."""
    x = rng.normal(size=(n, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product. a,b shape (...,4) with (w,x,y,z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return np.stack([w, x, y, z], axis=-1)

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def hopf_map(q: np.ndarray) -> np.ndarray:
    """
    Hopf map h(q) = q * i * q^{-1} with i=(0,1,0,0).
    Returns points on S^2 in R^3.
    """
    i = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    qi = quat_mul(q, i[None, :])
    qinv = quat_conj(q)  # for unit quaternions
    v = quat_mul(qi, qinv)
    # v is pure imaginary: (0, vx, vy, vz)
    return v[:, 1:4]

def fiber_phase(q: np.ndarray) -> np.ndarray:
    """
    A simple gauge-fixed fiber coordinate on S^1.
    Using complex coords: z1 = w + i x, z2 = y + i z.
    Take phase = arg(z1) + arg(z2), then wrap to (-pi,pi].
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi1 = np.arctan2(x, w)
    phi2 = np.arctan2(z, y)
    phi = phi1 + phi2
    # wrap to (-pi, pi]
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    return phi

def s1_circular_dist2(phi_i: np.ndarray, phi_j: np.ndarray) -> np.ndarray:
    """Squared circular distance on S^1: d = min(|Δ|, 2π-|Δ|)."""
    d = np.abs(phi_i - phi_j)
    d = np.minimum(d, 2*np.pi - d)
    return d**2

def s2_chordal_dist2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Squared chordal distance on S^2 embedded in R^3."""
    d = u - v
    return np.sum(d*d, axis=-1)


# -----------------------------
# Kernel & bias
# -----------------------------

def fiber_scale(alpha: float) -> float:
    """
    a(alpha) = exp(-max(alpha,0)).
    - alpha <= 0  -> a=1 (plateau)
    - alpha > 0   -> a decreases -> fiber distances penalized less -> ratio grows
    """
    return math.exp(-max(alpha, 0.0))

def kernel_from_dists(d_base2: np.ndarray, d_fiber2: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    a = fiber_scale(alpha)
    return np.exp(-(d_base2 + a * d_fiber2) / (2.0 * sigma * sigma))


# -----------------------------
# Monte-Carlo estimator (biased by kernel)
# -----------------------------

@dataclass
class MCResult:
    e_base: float
    e_fiber: float
    ratio: float

def mc_biased_pairs(
    q: np.ndarray,
    hopf: np.ndarray,
    phi: np.ndarray,
    alpha: float,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
    max_tries_factor: int = 80,
) -> MCResult:
    """
    Acceptance-rejection over uniformly proposed pairs (i,j).
    Target distribution proportional to K_α(i,j).
    Since K <= 1, accept with prob K.
    """
    n = q.shape[0]
    e_base_acc = []
    e_fiber_acc = []

    max_tries = max_tries_factor * n_samples
    accepted = 0
    tries = 0

    while accepted < n_samples and tries < max_tries:
        tries += 1
        i = rng.integers(0, n)
        j = rng.integers(0, n-1)
        if j >= i:
            j += 1

        d_base2 = float(s2_chordal_dist2(hopf[i:i+1], hopf[j:j+1])[0])
        d_fiber2 = float(s1_circular_dist2(phi[i:i+1], phi[j:j+1])[0])

        k = float(kernel_from_dists(np.array([d_base2]), np.array([d_fiber2]), alpha, sigma)[0])
        if rng.random() < k:
            e_base_acc.append(d_base2)
            e_fiber_acc.append(d_fiber2)
            accepted += 1

    if accepted < n_samples:
        raise RuntimeError(
            f"MC acceptance-rejection failed: accepted={accepted}/{n_samples}. "
            f"Try increasing sigma, reducing n_samples, or increasing max_tries_factor."
        )

    e_base = float(np.mean(e_base_acc))
    e_fiber = float(np.mean(e_fiber_acc))
    ratio = e_fiber / max(e_base, 1e-12)
    return MCResult(e_base=e_base, e_fiber=e_fiber, ratio=ratio)


# -----------------------------
# Spectral / Laplacian-side estimator (kernel-weighted Dirichlet form)
# -----------------------------

@dataclass
class SpecResult:
    e_base: float
    e_fiber: float
    ratio: float

def build_knn_edges(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build directed kNN edges on points x in R^d using brute-force distances.
    Returns arrays (src, dst) of length n*k.
    """
    n = x.shape[0]
    # brute force pairwise dist^2
    # For n ~ 2000 this is OK. For larger, switch to sklearn NearestNeighbors.
    d2 = np.sum((x[:, None, :] - x[None, :, :])**2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, kth=k, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = nn.reshape(-1)
    return src, dst

def spectral_kernel_energy(
    q: np.ndarray,
    hopf: np.ndarray,
    phi: np.ndarray,
    alpha: float,
    sigma: float,
    k_nn: int,
) -> SpecResult:
    """
    Compute kernel-weighted energies over a kNN graph:
        E = sum w_ij * d^2 / sum w_ij
    """
    src, dst = build_knn_edges(q, k_nn)

    d_base2 = s2_chordal_dist2(hopf[src], hopf[dst])
    d_fiber2 = s1_circular_dist2(phi[src], phi[dst])

    w = kernel_from_dists(d_base2, d_fiber2, alpha, sigma)

    wsum = float(np.sum(w))
    if wsum <= 0:
        raise RuntimeError("Sum of weights is zero; increase sigma or check distances.")

    e_base = float(np.sum(w * d_base2) / wsum)
    e_fiber = float(np.sum(w * d_fiber2) / wsum)
    ratio = e_fiber / max(e_base, 1e-12)
    return SpecResult(e_base=e_base, e_fiber=e_fiber, ratio=ratio)


# -----------------------------
# Statistics helpers
# -----------------------------

def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    """Return mean and half-width of 95% CI using normal approx."""
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(max(x.size, 1))
    return m, ci


# -----------------------------
# Main experiment
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_points", type=int, default=2500, help="Number of S^3 points for the graph / MC proposals")
    ap.add_argument("--k_nn", type=int, default=12, help="k for kNN graph (spectral branch)")
    ap.add_argument("--sigma", type=float, default=0.7, help="Kernel bandwidth sigma")
    ap.add_argument("--mc_samples", type=int, default=20000, help="Accepted MC pairs per repeat")
    ap.add_argument("--repeats", type=int, default=6, help="Repeats (independent seeds) for CI")
    ap.add_argument("--seed", type=int, default=1, help="Base random seed")
    ap.add_argument("--alpha_min", type=float, default=-2.0)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=17)
    ap.add_argument("--plot_energies", action="store_true", help="Also plot fiber/base energies separately")
    ap.add_argument("--save", type=str, default="", help="If set, save figure to this filename (png/pdf)")
    args = ap.parse_args()

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)

    # Storage
    mc_ratio_mean, mc_ratio_ci = [], []
    sp_ratio_mean, sp_ratio_ci = [], []

    mc_eb_mean, mc_eb_ci = [], []
    mc_ef_mean, mc_ef_ci = [], []
    sp_eb_mean, sp_eb_ci = [], []
    sp_ef_mean, sp_ef_ci = [], []

    print("Monte-Carlo vs Spectral response on S^3 (Hopf fibration S^1->S^3->S^2)")
    print(f"n_points={args.n_points}  k_nn={args.k_nn}  sigma={args.sigma}  mc_samples={args.mc_samples}  repeats={args.repeats}")
    print()

    for alpha in alphas:
        mc_ratios, sp_ratios = [], []
        mc_eb, mc_ef = [], []
        sp_eb, sp_ef = [], []

        for r in range(args.repeats):
            rng = np.random.default_rng(args.seed + 1000 * r + int((alpha - args.alpha_min) * 100))

            q = sample_s3(args.n_points, rng)
            hopf = hopf_map(q)
            phi = fiber_phase(q)

            # --- MC ---
            mcr = mc_biased_pairs(
                q=q, hopf=hopf, phi=phi,
                alpha=float(alpha), sigma=float(args.sigma),
                n_samples=args.mc_samples,
                rng=rng,
            )
            mc_eb.append(mcr.e_base)
            mc_ef.append(mcr.e_fiber)
            mc_ratios.append(mcr.ratio)

            # --- Spectral / Laplacian-style kernel energy ---
            spr = spectral_kernel_energy(
                q=q, hopf=hopf, phi=phi,
                alpha=float(alpha), sigma=float(args.sigma),
                k_nn=args.k_nn,
            )
            sp_eb.append(spr.e_base)
            sp_ef.append(spr.e_fiber)
            sp_ratios.append(spr.ratio)

        # summarize
        m_mc, ci_mc = mean_ci95(np.array(mc_ratios))
        m_sp, ci_sp = mean_ci95(np.array(sp_ratios))

        mc_ratio_mean.append(m_mc); mc_ratio_ci.append(ci_mc)
        sp_ratio_mean.append(m_sp); sp_ratio_ci.append(ci_sp)

        m_mc_eb, ci_mc_eb = mean_ci95(np.array(mc_eb))
        m_mc_ef, ci_mc_ef = mean_ci95(np.array(mc_ef))
        m_sp_eb, ci_sp_eb = mean_ci95(np.array(sp_eb))
        m_sp_ef, ci_sp_ef = mean_ci95(np.array(sp_ef))

        mc_eb_mean.append(m_mc_eb); mc_eb_ci.append(ci_mc_eb)
        mc_ef_mean.append(m_mc_ef); mc_ef_ci.append(ci_mc_ef)
        sp_eb_mean.append(m_sp_eb); sp_eb_ci.append(ci_sp_eb)
        sp_ef_mean.append(m_sp_ef); sp_ef_ci.append(ci_sp_ef)

        # debug: fiber scale and a quick sense of typical distances
        a = fiber_scale(float(alpha))
        print(f"alpha={alpha:+.2f}  a(exp(-max(alpha,0)))={a:.4f} | "
              f"MC ratio={m_mc:.6f}  95%CI=±{ci_mc:.6f} | "
              f"Spec ratio={m_sp:.6f}  95%CI=±{ci_sp:.6f}")

    # -----------------------------
    # Plot
    # -----------------------------
    alphas = np.array(alphas)
    target = 8.0 / 3.0

    if args.plot_energies:
        plt.figure(figsize=(11.5, 8.0))
        # Ratio on top, energies bottom
        ax1 = plt.subplot(2, 1, 1)
        ax1.axhline(y=target, linestyle="--", label="Theoretical target (8/3)")
        ax1.errorbar(alphas, mc_ratio_mean, yerr=mc_ratio_ci, marker="o",
                     capsize=3, label="Monte-Carlo (anisotropic kernel) — Hopf fiber/base")
        ax1.errorbar(alphas, sp_ratio_mean, yerr=sp_ratio_ci, marker="o",
                     capsize=3, label=f"Spectral (weighted Laplacian kernel) — Hopf fiber/base  [k={args.k_nn}, sigma={args.sigma}]")
        ax1.set_title("Monte-Carlo vs Spectral response on S³ (Hopf fibration S¹→S³→S²)")
        ax1.set_xlabel("Relaxation bias α")
        ax1.set_ylabel("Fiber/Base ratio  R(α) = E_fiber / E_base")
        ax1.grid(True, alpha=0.35)
        ax1.legend(loc="best")

        ax2 = plt.subplot(2, 1, 2)
        ax2.errorbar(alphas, mc_ef_mean, yerr=mc_ef_ci, marker="o",
                     capsize=3, label="MC  E_fiber")
        ax2.errorbar(alphas, mc_eb_mean, yerr=mc_eb_ci, marker="o",
                     capsize=3, label="MC  E_base")
        ax2.errorbar(alphas, sp_ef_mean, yerr=sp_ef_ci, marker="o",
                     capsize=3, label="Spec  E_fiber")
        ax2.errorbar(alphas, sp_eb_mean, yerr=sp_eb_ci, marker="o",
                     capsize=3, label="Spec  E_base")
        ax2.set_xlabel("Relaxation bias α")
        ax2.set_ylabel("Kernel-weighted energies")
        ax2.grid(True, alpha=0.35)
        ax2.legend(loc="best")
        plt.tight_layout()
    else:
        plt.figure(figsize=(11.5, 6.5))
        plt.axhline(y=target, linestyle="--", label="Theoretical target (8/3)")
        plt.errorbar(alphas, mc_ratio_mean, yerr=mc_ratio_ci, marker="o",
                     capsize=3, label="Monte-Carlo (anisotropic kernel) — Hopf fiber/base")
        plt.errorbar(alphas, sp_ratio_mean, yerr=sp_ratio_ci, marker="o",
                     capsize=3, label=f"Spectral (weighted Laplacian kernel) — Hopf fiber/base  [k={args.k_nn}, sigma={args.sigma}]")
        plt.title("Monte-Carlo vs Spectral response on S³ (Hopf fibration S¹→S³→S²)")
        plt.xlabel("Relaxation bias α")
        plt.ylabel("Fiber/Base ratio  R(α) = E_fiber / E_base")
        plt.grid(True, alpha=0.35)
        plt.legend(loc="best")
        plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=160)
        print(f"\nSaved figure to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
