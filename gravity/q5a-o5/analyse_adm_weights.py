"""
analyse_adm_weights.py
======================
Post-run analysis: compute A_hw = a_q(s) * q^2 for ALL conjugate pairs
using the correct formula (projected Weil fingerprint vectors).

Usage:
    python analyse_adm_weights.py --dir o25_outputs
    python analyse_adm_weights.py --dir o25_outputs --primes 29 61 101 151
"""

import numpy as np
import argparse, glob, os, sys, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import build_generators, bfs_shells, compute_block_capacity_with_residuals
from admissibility_weight import adm_weight_shell


def analyse_prime(path, seed=42, n_jobs=1):
    d        = np.load(path, allow_pickle=True)
    q        = int(d['q'])
    n0, n1   = int(d['n0']), int(d['n1'])
    pairs    = d['pairs']
    bfs_frac = float(d['bfs_frac'])
    n_max    = int(d['n_max_block'])

    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)

    A_hw_all = []
    iso_all  = []   # |a_X - a_Y|

    print(f"\n{'─'*60}")
    print(f"q={q}  n_pairs={len(pairs)}  window=[{n0},{n1}]  bfs_frac={bfs_frac}")
    print(f"{'pair':>12}  {'A_Eucl':>10}  {'A_hw':>8}  {'a_X':>10}  {'a_Y':>10}  {'|aX-aY|':>10}")

    for i, (c, qc) in enumerate(pairs):
        rng = np.random.default_rng(seed + i * 997 + int(c) * 7)
        c2  = int(rng.integers(1, q))
        c3  = int(rng.integers(1, q))
        while (int(c) + c2 + c3) % q == 0:
            c3 = int(rng.integers(1, q))
        cb = np.array([c, c2, c3], dtype=np.int64)

        out = compute_block_capacity_with_residuals(
            shells, cb, q, gens, n_max=n_max, n0=n0, n1=n1)
        if len(out) < 8:
            print(f"  ({c:3d},{qc:3d})  spectral_O12 not patched (needs 8-tuple)")
            continue
        *_, basis, _, shell_vecs = out

        # Average a_q(s) over all shells in window
        a_per_s = {'X': [], 'Xinv': [], 'Y': [], 'Yinv': []}
        for sv in shell_vecs:
            w = adm_weight_shell(sv, basis, q)
            for s in a_per_s:
                if not np.isnan(w.get(s, np.nan)):
                    a_per_s[s].append(w[s])

        a_q = {s: float(np.mean(v)) if v else np.nan for s, v in a_per_s.items()}
        A   = float(np.nanmean(list(a_q.values())))
        A_hw = A * q**2

        if not np.isnan(A):
            A_hw_all.append(A_hw)
            iso_all.append(abs(a_q['X'] - a_q['Y']) * q**2)
            flag = ''
        else:
            flag = '  [NaN — rank=0 in window]'

        print(f"  ({c:3d},{qc:3d})  {A:>10.5f}  {A_hw:>8.3f}  "
              f"{a_q['X']:>10.5f}  {a_q['Y']:>10.5f}  "
              f"{abs(a_q.get('X',0)-a_q.get('Y',0))*q**2:>10.4f}{flag}")

    if A_hw_all:
        print(f"\n  Summary q={q}:")
        print(f"    valid pairs  : {len(A_hw_all)} / {len(pairs)}")
        print(f"    A_hw mean    : {np.mean(A_hw_all):.4f} ± {np.std(A_hw_all):.4f}")
        print(f"    isotropy     : mean |a_X - a_Y|*q² = {np.mean(iso_all):.4f}")
        print(f"    H-w status   : {'✓ CONFIRMED (A_hw ≈ 2)' if abs(np.mean(A_hw_all)-2)<0.1 else '? CHECK'}")
    return {'q': q, 'A_hw_mean': float(np.mean(A_hw_all)) if A_hw_all else np.nan,
            'A_hw_std': float(np.std(A_hw_all)) if A_hw_all else np.nan,
            'n_valid': len(A_hw_all), 'n_pairs': len(pairs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',    default='o25_outputs')
    parser.add_argument('--primes', type=int, nargs='+', default=None)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    pattern = os.path.join(args.dir, 'q*_o25.npz')
    files = sorted(f for f in glob.glob(pattern) if '.v1.' not in f)
    if args.primes:
        files = [f for f in files
                 if any(f'q{q}_' in Path(f).name for q in args.primes)]
    if not files:
        print(f"No files found in {args.dir}")
        return

    results = []
    for path in files:
        r = analyse_prime(path, seed=args.seed)
        results.append(r)

    print(f"\n{'='*60}")
    print("CROSS-q SUMMARY")
    print(f"{'='*60}")
    print(f"{'q':>6}  {'A_hw':>8}  {'std':>8}  {'valid':>8}")
    for r in results:
        print(f"  {r['q']:>4}  {r['A_hw_mean']:>8.4f}  "
              f"{r['A_hw_std']:>8.4f}  {r['n_valid']:>3}/{r['n_pairs']}")

    with open('adm_weights_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → adm_weights_analysis.json")


if __name__ == '__main__':
    main()
