"""
o28_reff_extract.py
===================
O28 Part B -- formal r_eff computation (O26 Criterion 5.4) from Q5a-O5 checkpoints.

For each prime q and each conjugate pair (c, q-c), computes the covariance operator
    C_c = (1/N) sum_{n in [n0,n1]} sum_v  vec(M_n^v) vec(M_n^v)†
where
    M_n^v = pi_c[p, s, v] (x) pi_qmc[p, s, v]*  in End(C^3)
    vec(M) in C^9 (for HEFF_DIM=3)

and estimates r_eff = numerical rank of C_c.

The spin-1/2 prediction (O26 Criterion 5.4) is r_eff = d_rho^2 = 4.

Usage (run locally where the large npz files live):
    python o28_reff_extract.py --npz-dir o25_outputs --primes 29 61 101 151
    python o28_reff_extract.py --files q29_o25.npz q61_o25.npz

Output: o28_reff_summary.npz  (compact, uploadable)
"""

import argparse
import glob
import os
import numpy as np
from pathlib import Path

# Singular value threshold for numerical rank
RANK_THRESHOLD = 0.01   # sv > threshold * sv[0]
HEFF_DIM = 3            # dimension of H_eff subspace (= HEFF_DIM in spectral_O12)


def compute_reff_one_pair(pi_c_pair, pi_qmc_pair):
    """
    Compute r_eff for one conjugate pair.

    Parameters
    ----------
    pi_c_pair   : list of n_win arrays, each (N_s, HEFF_DIM) complex
    pi_qmc_pair : list of n_win arrays, each (N_s, HEFF_DIM) complex

    Returns
    -------
    r_eff_pr    : float  participation-ratio rank
    r_eff_thr   : int    threshold rank (sv > RANK_THRESHOLD * sv[0])
    sv_ratio    : float  sigma_2 / sigma_1
    sv_norm     : ndarray  normalised singular values
    n_vecs      : int    total number of outer products used
    """
    outer_products = []

    for pc, pqc in zip(pi_c_pair, pi_qmc_pair):
        if pc is None or pqc is None:
            continue
        pc  = np.asarray(pc,  dtype=complex)
        pqc = np.asarray(pqc, dtype=complex)
        if pc.ndim != 2 or pqc.ndim != 2:
            continue
        n_vecs = min(pc.shape[0], pqc.shape[0])
        if n_vecs == 0:
            continue
        pc  = pc[:n_vecs]
        pqc = pqc[:n_vecs]

        # M_v = pc[v] (x) pqc[v]^*  -- outer product in End(C^3)
        # vec(M_v) in C^{HEFF_DIM^2}
        # Efficient: np.einsum('vi,vj->vij', pc, pqc.conj()).reshape(n_vecs, -1)
        M_vecs = np.einsum('vi,vj->vij', pc, pqc.conj()).reshape(n_vecs, HEFF_DIM * HEFF_DIM)
        outer_products.append(M_vecs)

    if not outer_products:
        return np.nan, 0, np.nan, np.array([]), 0

    all_vecs = np.concatenate(outer_products, axis=0)   # (N_total, HEFF_DIM^2)
    N = all_vecs.shape[0]

    # Covariance C_c = (1/N) X† X  where X rows are vec(M_v)
    # SVD of X / sqrt(N) gives singular values of C_c^{1/2}
    sv = np.linalg.svd(all_vecs / np.sqrt(N), compute_uv=False)
    sv_sq = sv**2   # eigenvalues of C_c

    sv_max = sv_sq[0] if sv_sq[0] > 0 else 1.0
    sv_norm = sv_sq / sv_max

    # Participation ratio
    p2 = sv_norm / sv_norm.sum() if sv_norm.sum() > 0 else sv_norm
    r_eff_pr = float(1.0 / np.sum(p2**2)) if np.sum(p2**2) > 0 else 0.0

    # Threshold rank
    r_eff_thr = int(np.sum(sv_norm > RANK_THRESHOLD))

    sv_ratio = float(sv_norm[1]) if len(sv_norm) >= 2 else 0.0

    return r_eff_pr, r_eff_thr, sv_ratio, sv_norm, N


def process_file(path):
    """Load checkpoint and compute r_eff for all pairs."""
    d = np.load(path, allow_pickle=True)
    q     = int(d['q'])
    n0    = int(d['n0'])
    n1    = int(d['n1'])
    pairs = d['pairs']

    if 'pi_c' not in d or 'pi_qmc' not in d:
        print(f"  [q={q}] pi_c / pi_qmc not found -- re-run with --store-vectors")
        return None

    pi_c_all   = d['pi_c']    # (n_pairs, n_win) object array
    pi_qmc_all = d['pi_qmc']

    n_pairs, n_win = pi_c_all.shape
    print(f"\n  q={q}  n_pairs={n_pairs}  window=[{n0},{n1}]  HEFF_DIM={HEFF_DIM}")
    print(f"  {'pair':>12}  {'r_eff_PR':>10}  {'r_eff_thr':>10}  {'sv2/sv1':>9}  {'n_vecs':>8}")

    r_pr_all  = np.full(n_pairs, np.nan)
    r_thr_all = np.zeros(n_pairs, dtype=int)
    sv2_all   = np.full(n_pairs, np.nan)

    for i, (c, qc) in enumerate(pairs):
        pi_c_pair   = [pi_c_all[i, k]   for k in range(n_win)]
        pi_qmc_pair = [pi_qmc_all[i, k] for k in range(n_win)]

        r_pr, r_thr, sv_ratio, sv_norm, n_vecs = compute_reff_one_pair(
            pi_c_pair, pi_qmc_pair)

        r_pr_all[i]  = r_pr
        r_thr_all[i] = r_thr
        sv2_all[i]   = sv_ratio

        print(f"  ({c:3d},{qc:3d})  {r_pr:>10.3f}  {r_thr:>10d}  {sv_ratio:>9.4f}  {n_vecs:>8d}")

    print(f"\n  Summary q={q}:")
    print(f"    r_eff_PR   mean={np.nanmean(r_pr_all):.3f}  std={np.nanstd(r_pr_all):.3f}"
          f"  median={np.nanmedian(r_pr_all):.3f}")
    print(f"    r_eff_thr  mean={np.nanmean(r_thr_all):.2f}  "
          f"fraction={{'>=4': {(r_thr_all>=4).mean():.2f}, "
          f"'>=2': {(r_thr_all>=2).mean():.2f}}}")
    print(f"    Spin-1/2 prediction (O26): r_eff = d_rho^2 = 4")

    return dict(q=q, n0=n0, n1=n1, n_pairs=n_pairs,
                r_pr_all=r_pr_all, r_thr_all=r_thr_all, sv2_all=sv2_all,
                r_pr_mean=float(np.nanmean(r_pr_all)),
                r_pr_std=float(np.nanstd(r_pr_all)),
                r_thr_mean=float(np.nanmean(r_thr_all)),
                r_thr_frac4=float((r_thr_all >= 4).mean()),
                sv2_mean=float(np.nanmean(sv2_all)))


def main():
    parser = argparse.ArgumentParser(
        description="O28 Part B: formal r_eff (O26 Criterion 5.4) from Q5a-O5 checkpoints")
    parser.add_argument("--npz-dir", default=None,
                        help="Directory containing q*_o25.npz files")
    parser.add_argument("--files", nargs="+", default=[],
                        help="Explicit npz file paths")
    parser.add_argument("--primes", type=int, nargs="+", default=None,
                        help="Restrict to these primes")
    parser.add_argument("--out", default="o28_reff_summary.npz",
                        help="Output summary file (default: o28_reff_summary.npz)")
    args = parser.parse_args()

    files = list(args.files)
    if args.npz_dir:
        files += sorted(f for f in glob.glob(os.path.join(args.npz_dir, "q*_o25.npz"))
                        if ".v1." not in f)
    if not files:
        parser.error("Provide --files or --npz-dir")

    if args.primes:
        files = [f for f in files
                 if any(Path(f).name.startswith(f"q{q}_") for q in args.primes)]

    print(f"\n=== O28 Part B: r_eff extraction (O26 Criterion 5.4) ===")
    print(f"  Files: {[Path(f).name for f in files]}")
    print(f"  HEFF_DIM={HEFF_DIM}, rank threshold={RANK_THRESHOLD}")
    print(f"  Spin-1/2 prediction: r_eff_thr = d_rho^2 = 4")

    results = {}
    for path in files:
        r = process_file(path)
        if r is not None:
            results[r['q']] = r

    if not results:
        print("\nNo results.")
        return

    # Cross-q summary
    qs = sorted(results.keys())
    print(f"\n{'='*65}")
    print(f"CROSS-q SUMMARY  (spin-1/2 target: r_eff_PR~4, r_eff_thr>=4)")
    print(f"{'='*65}")
    print(f"{'q':>5}  {'r_PR mean':>10}  {'r_PR std':>9}  {'r_thr mean':>11}  "
          f"{'frac>=4':>9}  {'sv2/sv1':>9}")
    for q in qs:
        r = results[q]
        print(f"{q:5d}  {r['r_pr_mean']:>10.3f}  {r['r_pr_std']:>9.3f}  "
              f"{r['r_thr_mean']:>11.2f}  {r['r_thr_frac4']:>9.2f}  "
              f"{r['sv2_mean']:>9.4f}")

    # Save compact summary
    payload = {"qs": np.array(qs)}
    for q in qs:
        r = results[q]
        prefix = f"q{q}_"
        for key in ("r_pr_all", "r_thr_all", "sv2_all",
                    "r_pr_mean", "r_pr_std", "r_thr_mean",
                    "r_thr_frac4", "sv2_mean",
                    "n_pairs", "n0", "n1"):
            payload[prefix + key] = r[key]

    np.savez(args.out, **payload)
    print(f"\nSummary saved to {args.out}")
    print(f"(Upload this file to complete O28 Part B.)")


if __name__ == "__main__":
    main()