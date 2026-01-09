#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_mc_vs_weighted_laplacian_hopf_fiberbase_split.py

Monte-Carlo vs "spectral" (kernel/Laplacian-style) response on S^3
for the Hopf fibration S^1 -> S^3 -> S^2, explicitly separating fiber/base.

CRITICAL FIX:
  The Monte-Carlo branch is evaluated on the SAME kNN EDGE SUPPORT as the
  weighted-kernel ("spectral") branch. This removes the structural mismatch
  that occurs when MC uses global pairs while the spectral branch uses local kNN edges.

Output:
  - Curve R(alpha) = E_fiber(alpha) / E_base(alpha)
  - Optional plot of energies E_fiber and E_base for both branches.

Dependencies:
  numpy, scipy, matplotlib

Example:
  python compare_mc_vs_weighted_laplacian_hopf_fiberbase_split.py --plot_energies
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree


# -----------------------------
# Geometry: S^3 sampling + Hopf map
# -----------------------------

def sample_s3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on S^3 by normalizing 4D Gaussians."""
    x = rng.normal(size=(n, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def hopf_map_s3_to_s2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hopf map using complex coords:
      z1 = x0 + i x1
      z2 = x2 + i x3

    Base point on S^2:
      X = 2 Re(z1 * conj(z2))
      Y = 2 Im(z1 * conj(z2))
      Z = |z1|^2 - |z2|^2

    Fiber angle (a convenient choice):
      phi = arg(z1) - arg(z2)  (wrapped to [-pi, pi])
    """
    x0, x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    z1 = x0 + 1j * x1
    z2 = x2 + 1j * x3

    base_x = 2.0 * np.real(z1 * np.conj(z2))
    base_y = 2.0 * np.imag(z1 * np.conj(z2))
    base_z = (np.abs(z1) ** 2) - (np.abs(z2) ** 2)

    base = np.stack([base_x, base_y, base_z], axis=1)
    # Numerical safety (should already be on S^2)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    arg1 = np.angle(z1)
    arg2 = np.angle(z2)
    phi = wrap_angle(arg1 - arg2)

    return base, phi


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def s2_chordal_dist2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Chordal squared distance on S^2 embedded in R^3:
      d^2 = ||u - v||^2 = 2 - 2 <u, v>
    """
    dot = np.sum(u * v, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 - 2.0 * dot


def s1_circular_dist2(phi_i: np.ndarray, phi_j: np.ndarray) -> np.ndarray:
    """
    Squared circular distance on S^1 from angle difference:
      d = min(|Δ|, 2π-|Δ|)
      return d^2
    """
    d = np.abs(wrap_angle(phi_i - phi_j))
    d = np.minimum(d, 2.0 * np.pi - d)
    return d * d


# -----------------------------
# Graph / edges
# -----------------------------

def knn_edges(points: np.ndarray, k: int) -> np.ndarray:
    """
    Build directed kNN edges in R^4 using KDTree.
    Return array of shape (n*k, 2): (i, j) directed.
    """
    n = points.shape[0]
    tree = cKDTree(points)
    # query k+1 because the nearest neighbor is itself at distance 0
    dists, idxs = tree.query(points, k=k + 1)
    idxs = idxs[:, 1:]  # drop self

    i = np.repeat(np.arange(n), k)
    j = idxs.reshape(-1)
    return np.stack([i, j], axis=1)


def make_undirected_unique_edges(edges_ij: np.ndarray) -> np.ndarray:
    """
    Convert directed edges to undirected unique edges (i<j).
    """
    i = edges_ij[:, 0]
    j = edges_ij[:, 1]
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    und = np.stack([a, b], axis=1)
    und = np.unique(und, axis=0)
    # Remove self-loops if any (should not happen)
    und = und[und[:, 0] != und[:, 1]]
    return und


# -----------------------------
# Kernel + energies
# -----------------------------

def a_from_alpha(alpha: float) -> float:
    """
    Coupling factor a(alpha) used in your previous logs:
      a(alpha) = exp(-max(alpha, 0))
    So:
      alpha <= 0 -> a = 1
      alpha > 0  -> a decreases
    """
    return float(math.exp(-max(alpha, 0.0)))


def anisotropic_kernel(d_base2: np.ndarray, d_fiber2: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """
    Kernel K_alpha(i,j) = exp(-(d_base^2 + a(alpha) * d_fiber^2) / (2 sigma^2))
    """
    a = a_from_alpha(alpha)
    denom = 2.0 * (sigma ** 2)
    return np.exp(-(d_base2 + a * d_fiber2) / denom)


def spectral_energy_on_edges(
    base: np.ndarray,
    phi: np.ndarray,
    edges_und: np.ndarray,
    alpha: float,
    sigma: float,
) -> Tuple[float, float, float]:
    """
    "Spectral" branch energies: mean distances over the SAME edge set,
    weighted by the kernel (this matches the quantity your Laplacian-kernel
    construction implicitly encodes).

      E_base = sum_e w_e * d_base^2 / sum_e w_e
      E_fiber = sum_e w_e * d_fiber^2 / sum_e w_e
      R = E_fiber / E_base
    """
    i = edges_und[:, 0]
    j = edges_und[:, 1]
    d_base2 = s2_chordal_dist2(base[i], base[j])
    d_fiber2 = s1_circular_dist2(phi[i], phi[j])
    w = anisotropic_kernel(d_base2, d_fiber2, alpha, sigma)

    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise RuntimeError("Sum of kernel weights is zero. Increase sigma or check kernel.")
    e_base = float(np.sum(w * d_base2) / wsum)
    e_fiber = float(np.sum(w * d_fiber2) / wsum)
    r = e_fiber / max(e_base, 1e-15)
    return e_fiber, e_base, r


def mc_energy_on_same_edges(
    base: np.ndarray,
    phi: np.ndarray,
    edges_und: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
    n_edge_samples: int,
) -> Tuple[float, float, float]:
    """
    Monte-Carlo branch on the SAME edge support.

    We sample edges uniformly from the undirected edge list, and use
    importance weighting with the same kernel weights w_e:

      E_base = sum_s w_s * d_base^2 / sum_s w_s
      E_fiber = sum_s w_s * d_fiber^2 / sum_s w_s

    This is the "fixed" MC that is now directly comparable to the spectral branch.
    """
    m = edges_und.shape[0]
    idx = rng.integers(0, m, size=n_edge_samples)
    i = edges_und[idx, 0]
    j = edges_und[idx, 1]

    d_base2 = s2_chordal_dist2(base[i], base[j])
    d_fiber2 = s1_circular_dist2(phi[i], phi[j])
    w = anisotropic_kernel(d_base2, d_fiber2, alpha, sigma)

    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise RuntimeError("Sum of MC weights is zero. Increase sigma or check kernel.")
    e_base = float(np.sum(w * d_base2) / wsum)
    e_fiber = float(np.sum(w * d_fiber2) / wsum)
    r = e_fiber / max(e_base, 1e-15)
    return e_fiber, e_base, r


# -----------------------------
# Stats helpers
# -----------------------------

def mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    """
    Mean and approximate 95% CI (normal approx) for repeated estimates.
    """
    v = np.asarray(values, dtype=float)
    mu = float(np.mean(v))
    if len(v) <= 1:
        return mu, 0.0
    s = float(np.std(v, ddof=1))
    ci = 1.96 * s / math.sqrt(len(v))
    return mu, ci


@dataclass
class CurvePoint:
    alpha: float
    a: float
    mc_r: float
    mc_ci: float
    spec_r: float
    spec_ci: float
    # energies (optional)
    mc_e_fiber: float
    mc_e_fiber_ci: float
    mc_e_base: float
    mc_e_base_ci: float
    spec_e_fiber: float
    spec_e_fiber_ci: float
    spec_e_base: float
    spec_e_base_ci: float


# -----------------------------
# Main experiment
# -----------------------------

def run_one_repeat(
    n_points: int,
    k_nn: int,
    sigma: float,
    alpha: float,
    rng: np.random.Generator,
    mc_edge_samples: int,
) -> Dict[str, float]:
    """
    One repeat:
      - sample S^3 points
      - build Hopf base + fiber angle
      - build kNN edges and undirected unique set
      - compute MC-on-edges energies & ratio
      - compute spectral energies & ratio on full edge set
    """
    s3 = sample_s3(n_points, rng)
    base, phi = hopf_map_s3_to_s2(s3)

    edges_dir = knn_edges(s3, k=k_nn)
    edges_und = make_undirected_unique_edges(edges_dir)

    mc_e_f, mc_e_b, mc_r = mc_energy_on_same_edges(
        base, phi, edges_und, alpha, sigma, rng, n_edge_samples=mc_edge_samples
    )
    sp_e_f, sp_e_b, sp_r = spectral_energy_on_edges(
        base, phi, edges_und, alpha, sigma
    )

    return {
        "mc_r": mc_r,
        "spec_r": sp_r,
        "mc_e_f": mc_e_f,
        "mc_e_b": mc_e_b,
        "spec_e_f": sp_e_f,
        "spec_e_b": sp_e_b,
        "m_edges": float(edges_und.shape[0]),
    }


def run_curve(
    alphas: np.ndarray,
    n_points: int,
    k_nn: int,
    sigma: float,
    mc_edge_samples: int,
    repeats: int,
    seed: int,
) -> List[CurvePoint]:
    rng_master = np.random.default_rng(seed)
    curve: List[CurvePoint] = []

    print("Monte-Carlo vs Spectral response on S^3 (Hopf fibration S^1->S^3->S^2)")
    print(f"n_points={n_points}  k_nn={k_nn}  sigma={sigma}  mc_edge_samples={mc_edge_samples}  repeats={repeats}")
    print()

    for alpha in alphas:
        # collect repeats
        mc_rs, sp_rs = [], []
        mc_ef, mc_eb = [], []
        sp_ef, sp_eb = [], []

        # independent RNG stream per alpha for reproducibility
        alpha_seed = int(rng_master.integers(0, 2**31 - 1))
        rng_alpha = np.random.default_rng(alpha_seed)

        m_edges_list = []

        for _ in range(repeats):
            rep_seed = int(rng_alpha.integers(0, 2**31 - 1))
            rng = np.random.default_rng(rep_seed)
            out = run_one_repeat(
                n_points=n_points,
                k_nn=k_nn,
                sigma=sigma,
                alpha=float(alpha),
                rng=rng,
                mc_edge_samples=mc_edge_samples,
            )
            mc_rs.append(out["mc_r"])
            sp_rs.append(out["spec_r"])
            mc_ef.append(out["mc_e_f"])
            mc_eb.append(out["mc_e_b"])
            sp_ef.append(out["spec_e_f"])
            sp_eb.append(out["spec_e_b"])
            m_edges_list.append(out["m_edges"])

        mc_r_mu, mc_r_ci = mean_ci95(np.array(mc_rs))
        sp_r_mu, sp_r_ci = mean_ci95(np.array(sp_rs))

        mc_ef_mu, mc_ef_ci = mean_ci95(np.array(mc_ef))
        mc_eb_mu, mc_eb_ci = mean_ci95(np.array(mc_eb))
        sp_ef_mu, sp_ef_ci = mean_ci95(np.array(sp_ef))
        sp_eb_mu, sp_eb_ci = mean_ci95(np.array(sp_eb))

        a = a_from_alpha(float(alpha))

        print(
            f"alpha={alpha:+.2f}  a(exp(-max(alpha,0)))={a:.4f} | "
            f"MC ratio={mc_r_mu:.6f}  95%CI=±{mc_r_ci:.6f} | "
            f"Spec ratio={sp_r_mu:.6f}  95%CI=±{sp_r_ci:.6f} | "
            f"edges≈{np.mean(m_edges_list):.0f}"
        )

        curve.append(
            CurvePoint(
                alpha=float(alpha),
                a=a,
                mc_r=mc_r_mu,
                mc_ci=mc_r_ci,
                spec_r=sp_r_mu,
                spec_ci=sp_r_ci,
                mc_e_fiber=mc_ef_mu,
                mc_e_fiber_ci=mc_ef_ci,
                mc_e_base=mc_eb_mu,
                mc_e_base_ci=mc_eb_ci,
                spec_e_fiber=sp_ef_mu,
                spec_e_fiber_ci=sp_ef_ci,
                spec_e_base=sp_eb_mu,
                spec_e_base_ci=sp_eb_ci,
            )
        )

    return curve


# -----------------------------
# Plotting
# -----------------------------

def plot_curve(curve: List[CurvePoint], sigma: float, k_nn: int, show_energies: bool):
  alphas = np.array([p.alpha for p in curve])

  # Ratio plot
  mc = np.array([p.mc_r for p in curve])
  mc_ci = np.array([p.mc_ci for p in curve])
  sp = np.array([p.spec_r for p in curve])
  sp_ci = np.array([p.spec_ci for p in curve])

  if show_energies:
    fig = plt.figure(figsize=(11.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
  else:
    fig, ax1 = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    ax2 = None

  # Reference at alpha=0 (empirical baseline)
  r0 = sp[np.argmin(np.abs(alphas - 0.0))]
  ax1.axhline(r0, linestyle="--", label=f"Reference at α=0  (R≈{r0:.3f})")

  ax1.errorbar(
    alphas, mc, yerr=mc_ci, marker="o", capsize=4,
    label="Monte-Carlo (on kNN edges) — Hopf fiber/base"
  )
  ax1.errorbar(
    alphas, sp, yerr=sp_ci, marker="o", capsize=4,
    label=f"Spectral (kernel on kNN edges) — Hopf fiber/base  [k={k_nn}, sigma={sigma}]"
  )

  ax1.set_title("Monte-Carlo vs Spectral response on S³ (Hopf fibration S¹→S³→S²)")
  ax1.set_xlabel("Relaxation bias α")
  ax1.set_ylabel("Fiber/Base ratio  R(α) = E_fiber / E_base")
  ax1.grid(True, alpha=0.3)
  ax1.legend(loc="best")

  if show_energies and ax2 is not None:
    mc_ef = np.array([p.mc_e_fiber for p in curve])
    mc_ef_ci = np.array([p.mc_e_fiber_ci for p in curve])
    mc_eb = np.array([p.mc_e_base for p in curve])
    mc_eb_ci = np.array([p.mc_e_base_ci for p in curve])

    sp_ef = np.array([p.spec_e_fiber for p in curve])
    sp_ef_ci = np.array([p.spec_e_fiber_ci for p in curve])
    sp_eb = np.array([p.spec_e_base for p in curve])
    sp_eb_ci = np.array([p.spec_e_base_ci for p in curve])

    ax2.errorbar(alphas, mc_ef, yerr=mc_ef_ci, marker="o", capsize=4, label="MC  E_fiber")
    ax2.errorbar(alphas, mc_eb, yerr=mc_eb_ci, marker="o", capsize=4, label="MC  E_base")
    ax2.errorbar(alphas, sp_ef, yerr=sp_ef_ci, marker="o", capsize=4, label="Spec  E_fiber")
    ax2.errorbar(alphas, sp_eb, yerr=sp_eb_ci, marker="o", capsize=4, label="Spec  E_base")

    ax2.set_xlabel("Relaxation bias α")
    ax2.set_ylabel("Kernel-weighted energies")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

  plt.show()

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Monte-Carlo vs spectral (kernel/Laplacian-style) response on S^3 for Hopf fibration, fiber/base split."
    )
    p.add_argument("--n_points", type=int, default=2500, help="Number of S^3 points per repeat")
    p.add_argument("--k_nn", type=int, default=12, help="k for kNN graph")
    p.add_argument("--sigma", type=float, default=0.7, help="Kernel sigma")
    p.add_argument("--mc_edge_samples", type=int, default=200000, help="Number of MC edge samples per repeat")
    p.add_argument("--repeats", type=int, default=6, help="Number of independent repeats per alpha")
    p.add_argument("--seed", type=int, default=12345, help="Master RNG seed")
    p.add_argument("--alpha_min", type=float, default=-2.0)
    p.add_argument("--alpha_max", type=float, default=2.0)
    p.add_argument("--alpha_step", type=float, default=0.25)
    p.add_argument("--plot_energies", action="store_true", help="Also plot E_fiber and E_base curves")
    return p.parse_args()


def main():
    args = parse_args()
    # alpha grid
    n_steps = int(round((args.alpha_max - args.alpha_min) / args.alpha_step)) + 1
    alphas = np.linspace(args.alpha_min, args.alpha_max, n_steps)

    curve = run_curve(
        alphas=alphas,
        n_points=args.n_points,
        k_nn=args.k_nn,
        sigma=args.sigma,
        mc_edge_samples=args.mc_edge_samples,
        repeats=args.repeats,
        seed=args.seed,
    )

    plot_curve(curve, sigma=args.sigma, k_nn=args.k_nn, show_energies=args.plot_energies)


if __name__ == "__main__":
    main()
