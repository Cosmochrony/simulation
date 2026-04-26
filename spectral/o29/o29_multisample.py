"""
o29_multisample.py
==================
Resolve the q=101 anomaly observed in o29_rank_computation.py by replaying all
M=50 Monte Carlo samples (not just m=0) and computing reff_A for each sample.

If reff_A converges to 3 when averaged over all samples -> single-sample artefact.
If reff_A remains > 3 on most samples -> structural signal requiring investigation.

Requires:
  - A Q5a-O5 checkpoint  q<prime>_o25.npz  in --checkpoint-dir
  - spectral_O12.py and o25_paired_pipeline.py in --pipeline-dir (or on PYTHONPATH)
  - o29_rank_computation.py  in the same directory (for compute_A_heff etc.)

Usage:
    python o29_multisample.py --prime 101 --checkpoint-dir ../../gravity/q5a-o5/o25_outputs
    python o29_multisample.py --prime 101 --checkpoint-dir ... --pipeline-dir ../..
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEFF_DIM      = 3
MIN_TRAJ_LEN  = 4
DEFAULT_PRIME = 101
CHECKPOINT_PATTERN = "q{q}_o25.npz"


# =============================================================================
# Locate and import pipeline functions
# =============================================================================
def import_pipeline(pipeline_dir):
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    try:
        from spectral_O12 import compute_block_capacity_with_residuals
        from o25_paired_pipeline import sample_block_with_c1
        from spectral_O12 import build_generators, bfs_shells
        return (compute_block_capacity_with_residuals, sample_block_with_c1,
                build_generators, bfs_shells)
    except ImportError as e:
        print(f"[ERROR] Cannot import pipeline functions from '{pipeline_dir}': {e}")
        print("  Set --pipeline-dir to the directory containing spectral_O12.py "
              "and o25_paired_pipeline.py")
        sys.exit(1)


# =============================================================================
# Rank computation (duplicated from o29_rank_computation to keep standalone)
# =============================================================================
def covariance_rank(w_c, w_qc, threshold_rel):
    N = min(len(w_c), len(w_qc))
    wc, wqc = w_c[:N], w_qc[:N]
    d = wc.shape[1]
    M_vecs = (wc[:, :, None] * np.conj(wqc[:, None, :])).reshape(N, d * d)
    C = (M_vecs.conj().T @ M_vecs) / N
    eigvals = np.maximum(np.linalg.eigvalsh(C).real, 0.0)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals[0] < 1e-30:
        return eigvals, 0
    reff = int(np.sum(eigvals > threshold_rel * eigvals[0]))
    return eigvals, reff


def reff_from_pi(pi_c_shells, pi_qmc_shells, threshold_rel):
    """Stack all vectors from all shells and compute reff."""
    all_c, all_qmc = [], []
    for cell_c, cell_qmc in zip(pi_c_shells, pi_qmc_shells):
        c   = np.asarray(cell_c,   dtype=complex)
        qmc = np.asarray(cell_qmc, dtype=complex)
        if c.ndim == 1:
            c, qmc = c[np.newaxis], qmc[np.newaxis]
        if c.shape[0] == 0 or c.shape[-1] != HEFF_DIM:
            continue
        if qmc.shape[0] == 0 or qmc.shape[-1] != HEFF_DIM:
            continue
        all_c.append(c)
        all_qmc.append(qmc)
    if not all_c:
        return None, 0
    w_c   = np.concatenate(all_c,   axis=0)
    w_qmc = np.concatenate(all_qmc, axis=0)
    return covariance_rank(w_c, w_qmc, threshold_rel)


# =============================================================================
# Reconstruct the RNG sequence and replay all M samples
# =============================================================================
def replay_samples(q, pairs, shells, gens, seed, M, n0, n1, n_max_block,
                   compute_residuals, sample_block, threshold_rel):
    """
    Replay the o25 pipeline for all M samples, computing pi_c via
    compute_block_capacity_with_residuals for every sample (not just m=0).

    Returns: list of dicts per pair, each with
      "c", "qmc",
      "reff_per_sample": int array (M,)
      "reff_mean": float (reff averaged over samples)
      "reff_median": float
    """
    import time
    rng = np.random.default_rng(int(seed))

    pair_results = []
    n_pairs      = len(pairs)
    total_jobs   = n_pairs * M
    jobs_done    = 0
    t_start      = time.perf_counter()

    for p_idx, (c, qmc) in enumerate(pairs):
        c, qmc = int(c), int(qmc)
        reff_samples = np.zeros(M, dtype=int)
        t_pair = time.perf_counter()

        for m in range(M):
            cb_c   = sample_block(c,   q, rng)
            cb_qmc = sample_block(qmc, q, rng)

            _, _, _, _, _, _, pi_c,   _ = compute_residuals(
                shells, cb_c,   q, gens, n_max=n_max_block, n0=n0, n1=n1)
            _, _, _, _, _, _, pi_qmc, _ = compute_residuals(
                shells, cb_qmc, q, gens, n_max=n_max_block, n0=n0, n1=n1)

            _, reff = reff_from_pi(pi_c, pi_qmc, threshold_rel)
            reff_samples[m] = reff
            jobs_done += 1

            # progress within a pair every 5 samples
            if (m + 1) % 5 == 0 or m == M - 1:
                elapsed  = time.perf_counter() - t_start
                rate     = jobs_done / elapsed if elapsed > 0 else 1e-9
                eta_s    = (total_jobs - jobs_done) / rate
                eta_str  = (f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
                            if eta_s < 3600
                            else f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m")
                print(f"  pair {p_idx+1:3d}/{n_pairs} ({c:3d},{qmc:3d})"
                      f"  sample {m+1:2d}/{M}"
                      f"  reff={reff}"
                      f"  elapsed {elapsed:.0f}s  ETA {eta_str}",
                      flush=True)

        pair_results.append({
            "c": c, "qmc": qmc,
            "reff_per_sample": reff_samples,
            "reff_mean":   float(np.mean(reff_samples)),
            "reff_median": float(np.median(reff_samples)),
            "reff_modal":  int(np.bincount(reff_samples).argmax()),
        })
        dist = dict(zip(*np.unique(reff_samples, return_counts=True)))
        print(f"  => pair {p_idx+1:3d}/{n_pairs} ({c:3d},{qmc:3d}) done"
              f"  modal={pair_results[-1]['reff_modal']}"
              f"  mean={pair_results[-1]['reff_mean']:.2f}"
              f"  dist={dist}",
              flush=True)

    return pair_results


# =============================================================================
# Plot
# =============================================================================
def make_plot(pair_results, q, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    reff_all = np.stack([r["reff_per_sample"] for r in pair_results])  # (n_pairs, M)
    n_pairs, M = reff_all.shape

    # panel (a): distribution of reff over samples, per pair
    ax = axes[0]
    means   = reff_all.mean(axis=1)
    stds    = reff_all.std(axis=1)
    modals  = np.array([r["reff_modal"] for r in pair_results])
    x = np.arange(1, n_pairs + 1)
    ax.errorbar(x, means, yerr=stds, fmt="o", color="#2166ac",
                markersize=4, linewidth=0.8, alpha=0.7, label=f"mean $\\pm$ std over $M={M}$ samples")
    ax.scatter(x, modals, marker="s", color="#d6604d", s=18, zorder=3,
               label="modal reff")
    ax.axhline(3, color="forestgreen", linewidth=1.5, linestyle="--",
               label=r"$r_{\rm eff}=3$ (spin-1/2 prediction)")
    ax.set_xlabel("Pair index $p$", fontsize=11)
    ax.set_ylabel(r"$r_{\rm eff}$ in ${\rm End}(H_{\rm eff})$", fontsize=11)
    ax.set_title(rf"(a) reff distribution over $M={M}$ samples, $q={q}$", fontsize=11)
    ax.set_ylim(0, max(reff_all.max() + 1, 5))
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)

    # panel (b): histogram of reff values across all (pair, sample) cells
    ax2 = axes[1]
    vals = reff_all.ravel()
    bins = np.arange(vals.min(), vals.max() + 2) - 0.5
    ax2.hist(vals, bins=bins, color="#2166ac", edgecolor="white", linewidth=0.4)
    ax2.axvline(3, color="forestgreen", linewidth=1.5, linestyle="--",
                label=r"$r_{\rm eff}=3$")
    ax2.set_xlabel(r"$r_{\rm eff}$", fontsize=11)
    ax2.set_ylabel("Count (pair $\\times$ sample)", fontsize=11)
    ax2.set_title(rf"(b) reff histogram — $q={q}$, $M={M}$ samples, {n_pairs} pairs",
                  fontsize=11)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    pct3 = 100 * (vals == 3).mean()
    fig.suptitle(
        rf"O29 multi-sample: $q={q}$, $M={M}$ — "
        rf"{pct3:.1f}\% of (pair, sample) cells have $r_{{\rm eff}}=3$",
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
    p.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--pipeline-dir", default=".",
                   help="Directory containing spectral_O12.py and o25_paired_pipeline.py")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--pattern", default=CHECKPOINT_PATTERN)
    p.add_argument("--pairs", nargs="+", type=int, default=None,
                   help="Restrict to specific pair indices (0-based). Default: all pairs.")
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

    data = dict(np.load(ckpt_path, allow_pickle=True))
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Keys: {list(data.keys())}")

    seed     = int(data["seed"])
    M        = int(data["M_per_pair"])
    n0       = int(data["n0"])
    n1       = int(data["n1"])
    n_max    = int(data.get("n_max_block", 200))
    bfs_frac = float(data.get("bfs_frac", 0.99))
    pairs    = np.asarray(data["pairs"])

    print(f"  seed={seed}, M={M}, n0={n0}, n1={n1}, "
          f"n_max={n_max}, bfs_frac={bfs_frac}")
    print(f"  Building Cayley graph shells for q={q} (bfs_frac={bfs_frac})...")
    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    print(f"  {len(shells)} shells, |G_q_partial| = {sum(len(s) for s in shells)}")

    if args.pairs is not None:
        pairs = pairs[args.pairs]

    print(f"\nPrime q={q}, seed={seed}, M={M}, n0={n0}, n1={n1}, "
          f"n_max={n_max}, pairs to test={len(pairs)}")
    print(f"Threshold: {args.threshold}")
    print()

    pair_results = replay_samples(
        q, pairs, shells, gens, seed, M, n0, n1, n_max,
        compute_residuals, sample_block, args.threshold
    )

    # Summary
    modals  = np.array([r["reff_modal"]  for r in pair_results])
    means   = np.array([r["reff_mean"]   for r in pair_results])
    pct3_m  = 100 * (modals == 3).mean()
    reff_all = np.stack([r["reff_per_sample"] for r in pair_results]).ravel()
    pct3_all = 100 * (reff_all == 3).mean()

    print(f"\n{'='*60}\nSUMMARY  q={q}, M={M} samples\n{'='*60}")
    print(f"  Pairs with modal reff=3  : {(modals==3).sum()}/{len(modals)} ({pct3_m:.1f}%)")
    print(f"  (pair,sample) cells reff=3: {pct3_all:.1f}%")
    print(f"  Mean reff over all pairs  : {means.mean():.3f} +/- {means.std():.3f}")
    print()
    if pct3_all > 90:
        print("  => single-sample artefact confirmed: reff converges to 3 over M samples")
    elif pct3_all > 50:
        print("  => mostly reff=3; some pairs consistently anomalous (investigate)")
    else:
        print("  => reff=3 not the dominant value: possible structural signal")

    # Save
    out_dir = args.checkpoint_dir
    npz_path = os.path.join(out_dir, f"o29_multisample_q{q}.npz")
    np.savez(npz_path,
             q=q, M=M, n0=n0, n1=n1, seed=seed,
             pairs=pairs,
             reff_per_sample=np.stack([r["reff_per_sample"] for r in pair_results]),
             reff_modal=modals,
             reff_mean=means)
    print(f"Results saved: {npz_path}")

    plot_path = os.path.join(out_dir, f"o29_multisample_q{q}.pdf")
    make_plot(pair_results, q, plot_path)


if __name__ == "__main__":
    main()