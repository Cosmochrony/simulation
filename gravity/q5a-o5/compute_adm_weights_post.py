"""
compute_adm_weights_post.py
===========================
Post-processing: compute admissibility weights a_q(s) from existing
O25/O26 npz files that already contain 'basis_c' and 'basis_qmc'
(produced with --store-vectors).

Usage:
    python compute_adm_weights_post.py --dir ../../spectral/o25/o25_outputs
    python compute_adm_weights_post.py --files q29_o25.npz q61_o25.npz

Output: one JSON per file  q{q}_adm_weights.json
"""

import numpy as np
import argparse, glob, json, os
from pathlib import Path
from admissibility_weight import admissibility_weight_shell

GEN_NAMES = ['X', 'Xinv', 'Y', 'Yinv']


def process_one(path):
    d = np.load(path, allow_pickle=True)
    q = int(d['q'])
    keys = list(d.keys())

    if 'basis_c' not in keys:
        print(f"  [q={q}] No basis_c in {path} — re-run with --store-vectors")
        return None

    pairs       = d['pairs']          # (n_pairs, 2)
    basis_c_all = d['basis_c']        # object array (n_pairs,), each (rank, q)
    basis_qmc_all = d['basis_qmc']
    n_pairs = len(pairs)

    results = {s: {'c': [], 'qmc': []} for s in GEN_NAMES}

    for i in range(n_pairs):
        bc  = basis_c_all[i]
        bqc = basis_qmc_all[i]
        if bc is None or bc.shape[0] == 0:
            for s in GEN_NAMES:
                results[s]['c'].append(np.nan)
                results[s]['qmc'].append(np.nan)
            continue
        wc  = admissibility_weight_shell(bc,  q)
        wqc = admissibility_weight_shell(bqc, q)
        for s in GEN_NAMES:
            results[s]['c'].append(wc.get(s, np.nan))
            results[s]['qmc'].append(wqc.get(s, np.nan))

    # Summary
    print(f"\n--- q={q}  n_pairs={n_pairs} ---")
    print(f"  {'gen':6s}  {'mean_c':>10}  {'std_c':>10}  {'mean_qmc':>10}  {'std_qmc':>10}")
    for s in GEN_NAMES:
        vc  = np.array(results[s]['c'])
        vqc = np.array(results[s]['qmc'])
        print(f"  {s:6s}  {np.nanmean(vc):>10.4f}  {np.nanstd(vc):>10.4f}"
              f"  {np.nanmean(vqc):>10.4f}  {np.nanstd(vqc):>10.4f}")

    A_est = np.nanmean([np.nanmean(results[s]['c']) for s in GEN_NAMES])
    print(f"  A_estimate (mean over gens & pairs): {A_est:.4f}")
    print(f"  Isotropy check X vs Y: "
          f"a_X={np.nanmean(results['X']['c']):.4f}  "
          f"a_Y={np.nanmean(results['Y']['c']):.4f}")

    out = {
        'q': q,
        'A_estimate': float(A_est),
        'per_generator': {
            s: {
                'mean_c':   float(np.nanmean(results[s]['c'])),
                'std_c':    float(np.nanstd(results[s]['c'])),
                'mean_qmc': float(np.nanmean(results[s]['qmc'])),
                'std_qmc':  float(np.nanstd(results[s]['qmc'])),
            } for s in GEN_NAMES
        }
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--dir', default=None)
    args = parser.parse_args()

    files = list(args.files or [])
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, 'q*_o25.npz')))
        files = [f for f in files if '.v1.' not in f]
    if not files:
        parser.error("Provide --files or --dir")

    all_results = {}
    for path in files:
        r = process_one(path)
        if r is not None:
            all_results[r['q']] = r

    if all_results:
        with open('adm_weights_all.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results → adm_weights_all.json")

        qs = sorted(all_results.keys())
        A_vals = [all_results[q]['A_estimate'] for q in qs]
        print(f"\nCross-q stability of A:")
        for q, A in zip(qs, A_vals):
            print(f"  q={q}:  A={A:.4f}")
        print(f"Grand mean A = {np.mean(A_vals):.4f} ± {np.std(A_vals):.4f}")
        print(f"Relative spread = {np.std(A_vals)/np.mean(A_vals)*100:.1f}%")


if __name__ == '__main__':
    main()
