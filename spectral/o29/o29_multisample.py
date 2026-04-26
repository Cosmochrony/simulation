"""
o29_multisample.py
==================
Diagnose the q=101 anomaly from o29_rank_computation.py by replaying M Monte
Carlo samples per pair (instead of the single m=0 stored in the checkpoint).

Runs sequentially -- no multiprocessing overhead.  Shows per-sample progress
and a running ETA.

Question answered: is reff=3 robust across M samples, or was the m=0 result
an outlier?

Requires: spectral_O12.py and o25_paired_pipeline.py in --pipeline-dir.

Usage:
    python o29_multisample.py \\
        --prime 101 \\
        --checkpoint-dir ../../gravity/q5a-o5/o25_outputs \\
        --pipeline-dir   ../../gravity/q5a-o5

    # Faster diagnostic: 5 samples on the 10 most anomalous pairs
    python o29_multisample.py ... --n-samples 5 --pairs 1 3 4 5 6 7 12

Options:
    --n-samples N  Samples per pair (default: M from checkpoint).
    --pairs i j …  Pair indices to process (0-based). Default: all.
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEFF_DIM           = 3
CHECKPOINT_PATTERN = "q{q}_o25.npz"


# =============================================================================
# Pipeline import
# =============================================================================
def import_pipeline(pipeline_dir):
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    try:
        from spectral_O12 import (compute_block_capacity_with_residuals,
                                   build_generators, bfs_shells)
        from o25_paired_pipeline import sample_block_with_c1
        return (compute_block_capacity_with_residuals, sample_block_with_c1,
                build_generators, bfs_shells)
    except ImportError as e:
        print(f"[ERROR] Cannot import pipeline functions from '{pipeline_dir}': {e}")
        sys.exit(1)


# =============================================================================
# Rank helpers
# =============================================================================
def covariance_rank(w_c, w_qc, threshold_rel):
    N   = min(len(w_c), len(w_qc))
    wc  = w_c[:N]
    wqc = w_qc[:N]
    d   = wc.shape[1]
    V   = (wc[:, :, None] * np.conj(wqc[:, None, :])).reshape(N, d * d)
    C   = (V.conj().T @ V) / N
    ev  = np.maximum(np.linalg.eigvalsh(C).real, 0.0)
    ev  = np.sort(ev)[::-1]
    if ev[0] < 1e-30:
        return 0
    return int(np.sum(ev > threshold_rel * ev[0]))


def reff_from_pi(pi_c_shells, pi_qmc_shells, threshold_rel):
    rows_c, rows_qmc = [], []
    for cell_c, cell_qmc in zip(pi_c_shells, pi_qmc_shells):
        c   = np.asarray(cell_c,   dtype=complex)
        qmc = np.asarray(cell_qmc, dtype=complex)
        if c.ndim   == 1: c   = c[np.newaxis]
        if qmc.ndim == 1: qmc = qmc[np.newaxis]
        if c.shape[0]   == 0 or c.shape[-1]   != HEFF_DIM: continue
        if qmc.shape[0] == 0 or qmc.shape[-1] != HEFF_DIM: continue
        rows_c.append(c)
        rows_qmc.append(qmc)
    if not rows_c:
        return 0
    return covariance_rank(
        np.concatenate(rows_c,   axis=0),
        np.concatenate(rows_qmc, axis=0),
        threshold_rel
    )


# =============================================================================
# Main computation
# =============================================================================
def run(q, pairs, shells, gens, seed, M, n0, n1, n_max_block,
        threshold_rel, compute_residuals, sample_block):
    n_pairs   = len(pairs)
    t_start   = time.perf_counter()
    done_jobs = 0
    total     = n_pairs * M
    results   = []

    for p_idx, (c, qmc) in enumerate(pairs):
        c, qmc = int(c), int(qmc)
        # per-pair seed: deterministic, independent of other pairs
        pair_seed = int(seed) ^ (p_idx * 1_000_003) ^ (c * 997) ^ (qmc * 991)
        rng = np.random.default_rng(pair_seed)

        reff_samples = np.zeros(M, dtype=int)
        t_pair = time.perf_counter()

        for m in range(M):
            cb_c   = sample_block(c,   q, rng)
            cb_qmc = sample_block(qmc, q, rng)

            _, _, _, _, _, _, pi_c,   _ = compute_residuals(
                shells, cb_c,   q, gens, n_max=n_max_block, n0=n0, n1=n1)
            _, _, _, _, _, _, pi_qmc, _ = compute_residuals(
                shells, cb_qmc, q, gens, n_max=n_max_block, n0=n0, n1=n1)

            reff = reff_from_pi(pi_c, pi_qmc, threshold_rel)
            reff_samples[m] = reff
            done_jobs += 1

            elapsed = time.perf_counter() - t_start
            rate    = done_jobs / elapsed if elapsed > 0 else 1e-9
            eta_s   = (total - done_jobs) / rate
            eta_str = (f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m"
                       if eta_s >= 3600
                       else f"{int(eta_s//60)}m{int(eta_s%60):02d}s")
            print(f"\r  pair {p_idx+1:3d}/{n_pairs} ({c:3d},{qmc:3d})"
                  f"  sample {m+1:2d}/{M}"
                  f"  reff={reff}"
                  f"  elapsed {elapsed:.0f}s  ETA {eta_str}   ",
                  end="", flush=True)

        dist = dict(zip(*np.unique(reff_samples, return_counts=True)))
        modal = int(np.bincount(reff_samples).argmax())
        t_pair_elapsed = time.perf_counter() - t_pair
        print(f"\n  => ({c:3d},{qmc:3d})"
              f"  modal={modal}"
              f"  mean={reff_samples.mean():.2f}"
              f"  dist={dist}"
              f"  ({t_pair_elapsed:.1f}s/pair)",
              flush=True)

        results.append({
            "c": c, "qmc": qmc,
            "reff_per_sample": reff_samples,
            "reff_mean":   float(reff_samples.mean()),
            "reff_median": float(np.median(reff_samples)),
            "reff_modal":  modal,
        })

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total: {elapsed:.1f}s  ({1000*elapsed/total:.0f} ms/job)", flush=True)
    return results


# =============================================================================
# Plot
# =============================================================================
def make_plot(pair_results, q, M, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    reff_all = np.stack([r["reff_per_sample"] for r in pair_results])
    n_pairs  = len(pair_results)
    x        = np.arange(1, n_pairs + 1)

    ax = axes[0]
    ax.errorbar(x, reff_all.mean(axis=1), yerr=reff_all.std(axis=1),
                fmt="o", color="#2166ac", markersize=4, linewidth=0.8,
                alpha=0.7, label=f"mean $\\pm$ std, $M={M}$ samples")
    ax.scatter(x, [r["reff_modal"] for r in pair_results],
               marker="s", color="#d6604d", s=18, zorder=3, label="modal reff")
    ax.axhline(3, color="forestgreen", linewidth=1.5, linestyle="--",
               label=r"$r_{\rm eff}=3$ (spin-1/2)")
    ax.set_xlabel("Pair index $p$", fontsize=11)
    ax.set_ylabel(r"$r_{\rm eff}$ in ${\rm End}(H_{\rm eff})$", fontsize=11)
    ax.set_title(rf"(a) reff over $M={M}$ samples — $q={q}$", fontsize=11)
    ax.set_ylim(0, max(reff_all.max() + 1, 5))
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    vals = reff_all.ravel()
    bins = np.arange(vals.min(), vals.max() + 2) - 0.5
    ax2.hist(vals, bins=bins, color="#2166ac", edgecolor="white", linewidth=0.4)
    ax2.axvline(3, color="forestgreen", linewidth=1.5, linestyle="--",
                label=r"$r_{\rm eff}=3$")
    ax2.set_xlabel(r"$r_{\rm eff}$", fontsize=11)
    ax2.set_ylabel("Count (pair x sample)", fontsize=11)
    ax2.set_title(rf"(b) reff histogram — $q={q}$, $M={M}$, {n_pairs} pairs",
                  fontsize=11)
    ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.25)

    pct3 = 100 * (vals == 3).mean()
    fig.suptitle(
        rf"O29 multi-sample: $q={q}$, $M={M}$ — {pct3:.1f}% of cells have"
        r" $r_{\rm eff}=3$",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Plot saved: {path}")


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="O29 multi-sample rank computation")
    p.add_argument("--prime",          type=int,   default=101)
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--pipeline-dir",   default=".",
                   help="Directory containing spectral_O12.py and o25_paired_pipeline.py")
    p.add_argument("--threshold",  type=float, default=0.01)
    p.add_argument("--pattern",    default=CHECKPOINT_PATTERN)
    p.add_argument("--n-samples",  type=int,   default=None,
                   help="Samples per pair (default: M from checkpoint).")
    p.add_argument("--pairs", nargs="+", type=int, default=None,
                   help="Pair indices to process (0-based). Default: all.")
    return p.parse_args()


def main():
    args = parse_args()
    q    = args.prime

    compute_residuals, sample_block, build_generators, bfs_shells = \
        import_pipeline(args.pipeline_dir)

    ckpt_path = os.path.join(args.checkpoint_dir, args.pattern.format(q=q))
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    data     = dict(np.load(ckpt_path, allow_pickle=True))
    seed     = int(data["seed"])
    M        = args.n_samples or int(data["M_per_pair"])
    n0       = int(data["n0"])
    n1       = int(data["n1"])
    n_max    = int(data.get("n_max_block", 200))
    bfs_frac = float(data.get("bfs_frac", 0.99))
    pairs    = np.asarray(data["pairs"])

    print(f"Loaded: q={q}  seed={seed}  M_ckpt={int(data['M_per_pair'])}"
          f"  n0={n0}  n1={n1}  n_max={n_max}  bfs_frac={bfs_frac}")
    print(f"Using M={M} samples per pair.")
    print(f"Building BFS shells for q={q} (bfs_frac={bfs_frac})...", flush=True)

    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    print(f"  {len(shells)} shells, |G| = {sum(len(s) for s in shells)}")

    if args.pairs is not None:
        pairs = pairs[args.pairs]
    print(f"Processing {len(pairs)} pairs.\n")

    results = run(q, pairs, shells, gens, seed, M, n0, n1, n_max,
                  args.threshold, compute_residuals, sample_block)

    # Summary
    modals   = np.array([r["reff_modal"] for r in results])
    means    = np.array([r["reff_mean"]  for r in results])
    reff_all = np.stack([r["reff_per_sample"] for r in results]).ravel()
    pct3_m   = 100 * (modals == 3).mean()
    pct3_all = 100 * (reff_all == 3).mean()

    print(f"\n{'='*60}")
    print(f"SUMMARY  q={q}, M={M} samples, {len(pairs)} pairs")
    print(f"{'='*60}")
    print(f"  Pairs with modal reff=3     : {(modals==3).sum()}/{len(modals)}"
          f"  ({pct3_m:.1f}%)")
    print(f"  (pair, sample) cells reff=3 : {pct3_all:.1f}%")
    print(f"  Mean reff (pair average)    : {means.mean():.3f} +/- {means.std():.3f}")
    print()
    if pct3_all > 90:
        print("  => artefact mono-echantillon confirme : reff converge vers 3")
    elif pct3_all > 50:
        print("  => majorite reff=3 ; quelques paires consistamment anomales")
    else:
        print("  => reff=3 non dominant : signal structurel possible")

    out_dir  = args.checkpoint_dir
    npz_path = os.path.join(out_dir, f"o29_multisample_q{q}.npz")
    np.savez(npz_path, q=q, M=M, n0=n0, n1=n1, seed=seed, pairs=pairs,
             reff_per_sample=np.stack([r["reff_per_sample"] for r in results]),
             reff_modal=modals, reff_mean=means)
    print(f"Results saved: {npz_path}")

    make_plot(results, q, M,
              os.path.join(out_dir, f"o29_multisample_q{q}.pdf"))


if __name__ == "__main__":
    main()