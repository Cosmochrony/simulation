"""
build_hsuff_input.py

Adapter: convert existing O25 pipeline outputs (q<q>_o25.npz, produced by
o25_paired_pipeline.py) into the per-prime input format consumed by
fibre_erasure_hsuff_test.py.

Each input q<q>_o25.npz file contains, among other fields:
    pairs          : int   (n_pairs, 2)            -- pairs[i] = (c, q-c)
    sigma_c_mean   : float (n_pairs, n_shells)     -- M-averaged sigma_c(n)
    sigma_qmc_mean : float (n_pairs, n_shells)     -- M-averaged sigma_{q-c}(n)
    ns             : int   (n_shells,)             -- shell depths
    M_per_pair     : scalar                        -- block samples per pair
    n0, n1         : scalars                       -- fitting window

The adapter stacks sigma_c_mean and sigma_qmc_mean into a single
(n_samples, n_shells) array with n_samples = q - 1.
Each row corresponds to a unique c in {1, ..., q-1}, with its M-block average
BFS capacity profile.

Output: q_<q>.npz with
    c          : int   (n_samples,)             -- fibre label per sample
    sigma      : float (n_samples, n_shells)    -- mean BFS capacity profile
    ns         : int   (n_shells,)              -- shell depths
    q          : int                            -- prime
    M_per_pair : int                            -- preserved for reference
    n0, n1     : int                            -- fitting window preserved

Usage:
    python build_hsuff_input.py
    python build_hsuff_input.py --in-dir o25_outputs --out-dir hsuff_input
    python build_hsuff_input.py --q-list 61 151 211 307

LIMITATION
----------
The O25 pipeline stores only the per-pair MEAN over M block samples;
individual block trajectories sigma_c^(m)(n) are not preserved.  Consequently
the adapter produces one row per c value (not M rows).

The downstream test fibre_erasure_hsuff_test.py auto-detects this and uses
a permutation-based Spearman correlation test (plus a triplet-eta-squared
test when q ≡ 1 mod 3) rather than the categorical-eta-squared test that
the synthetic mode uses.

To enable the categorical-eta-squared test on real data, the pipeline must
be extended to save the M individual trajectories per pair (a small
modification of _compute_one_pair in o25_paired_pipeline.py: add
sv_c_samples and sv_qc_samples accumulators of shape (M, n_shells) and
include them in the saved payload).
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np


_QFILE_RE = re.compile(r"^q(\d+)_o25\.npz$")


def adapt_one(in_path: pathlib.Path, out_path: pathlib.Path) -> dict:
    """Convert a single q<q>_o25.npz to q_<q>.npz; return a summary dict."""
    z = np.load(in_path, allow_pickle=True)
    q = int(z["q"])
    pairs = np.asarray(z["pairs"])                       # (n_pairs, 2)
    sigma_c = np.asarray(z["sigma_c_mean"])              # (n_pairs, n_shells)
    sigma_qmc = np.asarray(z["sigma_qmc_mean"])          # (n_pairs, n_shells)
    ns = np.asarray(z["ns"])
    n_pairs = pairs.shape[0]
    n_shells = sigma_c.shape[1]

    # Stack: 2 * n_pairs samples = q - 1.
    # First half: c values from pairs[:, 0]; second half: q - c from pairs[:, 1].
    c = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    sigma = np.concatenate([sigma_c, sigma_qmc], axis=0).astype(np.float64)

    # Sanity: every c in {1, ..., q-1} should appear exactly once.
    if len(np.unique(c)) != len(c):
        raise ValueError(f"Duplicate c values after stacking for q={q}")
    if c.min() < 1 or c.max() > q - 1:
        raise ValueError(f"c out of range [1, q-1] for q={q}")

    payload = dict(
        c=c,
        sigma=sigma,
        ns=ns.astype(np.int64),
        q=np.int64(q),
        M_per_pair=np.int64(z["M_per_pair"]) if "M_per_pair" in z.files else np.int64(0),
        n0=np.int64(z["n0"]) if "n0" in z.files else np.int64(0),
        n1=np.int64(z["n1"]) if "n1" in z.files else np.int64(0),
    )
    np.savez(out_path, **payload)
    return {
        "q": q,
        "n_samples": len(c),
        "n_shells": n_shells,
        "n_pairs": n_pairs,
        "M_per_pair": int(payload["M_per_pair"]),
        "n0": int(payload["n0"]),
        "n1": int(payload["n1"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=pathlib.Path,
                        default=pathlib.Path("o25_outputs"),
                        help="Directory containing q<q>_o25.npz files.")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("hsuff_input"),
                        help="Directory where q_<q>.npz files will be written.")
    parser.add_argument("--q-list", type=int, nargs="+", default=None,
                        help="Restrict to specific primes.  Default: all q<q>_o25.npz "
                             "files found in --in-dir.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.q_list:
        in_paths = [args.in_dir / f"q{q}_o25.npz" for q in args.q_list]
    else:
        in_paths = sorted(args.in_dir.glob("q*_o25.npz"))

    if not in_paths:
        print(f"[ERROR] No matching files in {args.in_dir}")
        return 1

    print(f"[INFO] adapting {len(in_paths)} files from {args.in_dir} to {args.out_dir}")
    print()
    print(f"  {'q':>6}  {'n_samples':>10}  {'n_shells':>9}  {'M_per_pair':>11}  "
          f"{'n0,n1':>8}")
    print("  " + "-" * 56)

    for in_path in in_paths:
        if not in_path.exists():
            print(f"  [WARN] {in_path} not found, skipping")
            continue
        m = _QFILE_RE.match(in_path.name)
        if not m:
            print(f"  [WARN] {in_path.name} does not match expected pattern, skipping")
            continue
        q_str = m.group(1)
        out_path = args.out_dir / f"q_{q_str}.npz"
        summary = adapt_one(in_path, out_path)
        print(f"  {summary['q']:>6}  {summary['n_samples']:>10}  "
              f"{summary['n_shells']:>9}  {summary['M_per_pair']:>11}  "
              f"[{summary['n0']},{summary['n1']}]")

    print()
    print(f"[DONE] Output in {args.out_dir}")
    print()
    print("Next:")
    print("  python fibre_erasure_hsuff_test.py "
          f"--data-dir {args.out_dir} --out hsuff_results")


if __name__ == "__main__":
    raise SystemExit(main() or 0)