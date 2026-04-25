"""
o28_reff_figures.py
===================
Generate o28_reff_measurement.pdf from o28_reff_summary.npz
(produced by o28_reff_extract.py).

Usage:
    python o28_reff_figures.py --summary o28_reff_summary.npz
    python o28_reff_figures.py --summary o28_reff_summary.npz --out-dir .
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameters ───────────────────────────────────────────────────────────────
HEFF_DIM  = 3
D_RHO_SQ  = 4          # spin-1/2 prediction: d_rho^2
PR_TARGET = 8.0 / 3.0  # r_eff_PR for eigenvalue structure [1, 1/2, 1/2]

COLOR_VALID   = "#2c7bb6"
COLOR_INVALID = "#cccccc"
COLOR_TARGET  = "#d7191c"
COLOR_HEFF    = "#31a354"
COLOR_FILL    = "#abd9e9"


def load_summary(path):
    d = np.load(path, allow_pickle=True)
    qs = list(map(int, d['qs']))
    data = {}
    for q in qs:
        p = f'q{q}_'
        data[q] = dict(
            r_pr  = d[p + 'r_pr_all'],
            r_thr = d[p + 'r_thr_all'],
            sv2   = d[p + 'sv2_all'],
            n0    = int(d[p + 'n0']),
            n1    = int(d[p + 'n1']),
        )
    return qs, data


def make_figure(qs, data, outpath):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9,  "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Panel (a): r_eff_thr per pair, all q ─────────────────────────────────
    ax = axes[0]
    offsets = np.linspace(-0.3, 0.3, len(qs))
    for i, q in enumerate(qs):
        r_thr = data[q]['r_thr']
        valid = r_thr > 0
        xs_v = np.full(valid.sum(),  q) + offsets[i] + np.random.default_rng(q).uniform(
            -0.06, 0.06, valid.sum())
        xs_i = np.full((~valid).sum(), q) + offsets[i]
        if valid.sum() > 0:
            ax.scatter(xs_v, r_thr[valid], color=COLOR_VALID, s=12, alpha=0.6,
                       zorder=3, label="valid" if i == 0 else None)
        if (~valid).sum() > 0:
            ax.scatter(xs_i, np.zeros((~valid).sum()), color=COLOR_INVALID,
                       s=12, alpha=0.4, marker='x', zorder=2,
                       label="empty $\\pi_c$" if i == 0 else None)
    ax.axhline(D_RHO_SQ,  color=COLOR_TARGET, linestyle="--", lw=1.4,
               label=f"$d_\\rho^2 = {D_RHO_SQ}$ (spin-1/2 target)")
    ax.axhline(HEFF_DIM,  color=COLOR_HEFF,   linestyle=":",  lw=1.4,
               label=f"$\\mathrm{{HEFF\\_DIM}} = {HEFF_DIM}$")
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$r_{\\mathrm{eff}}^{\\mathrm{thr}}$  per pair")
    ax.set_title("(a) Threshold rank per conjugate pair")
    ax.set_xticks(qs)
    ax.set_ylim(-0.5, D_RHO_SQ + 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")

    # ── Panel (b): sv2/sv1 distribution per q ────────────────────────────────
    ax = axes[1]
    for i, q in enumerate(qs):
        sv2   = data[q]['sv2']
        valid = ~np.isnan(sv2)
        if valid.sum() == 0:
            continue
        xs = np.full(valid.sum(), q) + np.random.default_rng(q + 1).uniform(
            -0.15, 0.15, valid.sum())
        ax.scatter(xs, sv2[valid], color=COLOR_VALID, s=12, alpha=0.6, zorder=3)
    ax.axhline(0.5, color=COLOR_TARGET, linestyle="--", lw=1.4,
               label="exact $\\sigma_2/\\sigma_1 = 1/2$")
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$\\sigma_2/\\sigma_1$  (normalised eigenvalue ratio)")
    ax.set_title("(b) Eigenvalue ratio $\\sigma_2/\\sigma_1$ per conjugate pair")
    ax.set_xticks(qs)
    ax.set_ylim(0.45, 0.85)
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")

    # ── Panel (c): eigenvalue structure at each q (mean ± std) ───────────────
    ax = axes[2]
    # From r_eff_thr=3 and sv2/sv1=0.5, structure is [sv1, sv1/2, sv1/2, 0...].
    # Normalised: [1, 0.5, 0.5, 0, ...].  Compute from actual sv2_mean.
    bar_width = 0.18
    offsets2  = np.linspace(-(len(qs)-1)/2, (len(qs)-1)/2, len(qs)) * bar_width
    colors_q  = plt.cm.Blues(np.linspace(0.4, 0.85, len(qs)))

    for i, q in enumerate(qs):
        sv2   = data[q]['sv2']
        valid = ~np.isnan(sv2)
        if valid.sum() == 0:
            continue
        sv2_mean = float(np.nanmean(sv2[valid]))
        # Normalised eigenvalue vector: [1, sv2_mean, sv2_mean, 0, ...]
        eig_norm = np.array([1.0, sv2_mean, sv2_mean])
        xs = np.arange(1, HEFF_DIM + 1) + offsets2[i]
        ax.bar(xs, eig_norm, width=bar_width * 0.85, color=colors_q[i],
               label=f"$q={q}$", zorder=3)

    ax.axhline(0.5, color=COLOR_TARGET, linestyle="--", lw=1.0, alpha=0.7,
               label="exact $= 1/2$")
    ax.set_xlabel("eigenvalue index $i$  of $\\mathcal{C}_c$ in $\\mathrm{End}(H_{\\mathrm{eff}})$")
    ax.set_ylabel("$\\lambda_i / \\lambda_1$")
    ax.set_title(f"(c) Normalised eigenvalue structure of $\\mathcal{{C}}_c$\n"
                 f"(rank 3 fills $\\mathrm{{End}}(H_{{\\mathrm{{eff}}}})$,"
                 f" $\\mathrm{{HEFF\\_DIM}}={HEFF_DIM}$)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["$\\lambda_1$", "$\\lambda_2$", "$\\lambda_3$"])
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.tick_params(direction="in")

    fig.suptitle(
        r"O28 --- Part B: Effective dimension of $\mathcal{C}_c$ in"
        r" $\mathrm{End}(H_{\mathrm{eff}})$  (O26 Criterion 5.4, formal computation)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate O28 Part B figure from o28_reff_summary.npz")
    parser.add_argument("--summary", default="o28_reff_summary.npz")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    qs, data = load_summary(args.summary)
    outpath  = os.path.join(args.out_dir, "o28_reff_measurement.pdf")
    make_figure(qs, data, outpath)


if __name__ == "__main__":
    main()