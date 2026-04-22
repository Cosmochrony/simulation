"""
Q5a-O5 : extraction du coefficient A de la forme de Mosco limite.

Dans le pipeline O25, sigma_pair_mean[pair, shell] = sigma_pair(n)
~ C * n^{-delta_pair}.  L'hypothèse H-w de Q5a dit que le profil
NORMALISÉ sigma_pair(n)/sigma_pair(0) est approximativement plat
(= A_norm) indépendamment de n, q et du pair.

A_norm est l'estimateur de A dans les unités du pipeline.
Pour obtenir A en unités physiques il faudra identifier c_BI (Q5a-O5).

Usage:
    python extract_A.py --dir ../../spectral/o25/o25_outputs
    python extract_A.py --files q29_o25.npz q61_o25.npz q101_o25.npz
"""

import numpy as np
import argparse
import os
import glob
import json
from pathlib import Path


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def extract(data, q):
    """
    Returns (ns, shell_sizes, sigma_pair_mean_over_pairs, n0, n1)
    sigma_pair_mean_over_pairs[n] = mean over pairs of sigma_pair_mean[pair, n]
    """
    ns          = np.asarray(data['ns'], dtype=int)          # (n_shells,)
    shells      = np.asarray(data['shell_sizes'], dtype=float) # (n_shells,)
    sp          = np.asarray(data['sigma_pair_mean'],
                             dtype=float)                     # (n_pairs, n_shells)
    n0          = int(data['n0'])
    n1          = int(data['n1'])

    # mean over pairs at each shell
    sp_mean = np.mean(sp, axis=0)                             # (n_shells,)

    # keep only shells with positive values
    mask = (ns > 0) & (sp_mean > 0) & np.isfinite(sp_mean)
    return ns[mask], shells[mask], sp_mean[mask], n0, n1


def fit_plateau(ns, rho_norm, n0, n1):
    """
    Estimate plateau in the fitting window [n0, n1] (the pre-saturation regime).
    Returns (mean, std) of rho_norm in that window.
    """
    mask = (ns >= n0) & (ns <= n1)
    if mask.sum() < 2:
        # fallback: last 40% of shells
        mask = ns >= ns[-1] * 0.6
    vals = rho_norm[mask]
    return float(np.mean(vals)), float(np.std(vals))


def run(files, window_frac=0.4, plot=True):
    results = {}

    for path in sorted(files):
        stem = Path(path).stem
        q = None
        for part in stem.split('_'):
            if part.startswith('q') and part[1:].isdigit():
                q = int(part[1:])
                break
        if q is None:
            print(f"Cannot infer q from {path}, skipping.")
            continue

        print(f"\n--- q={q}  ({path}) ---")
        try:
            data = load(path)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        ns, shells, sp_mean, n0, n1 = extract(data, q)

        # Normalised profile: rho_norm(n) = sigma_pair(n) / sigma_pair(n=1)
        # (n=0 shell is often size 1 and noisy; use first non-zero shell)
        ref = sp_mean[0]
        rho_norm = sp_mean / ref

        A_norm, A_std = fit_plateau(ns, rho_norm, n0, n1)
        # Raw sigma_pair plateau (in pipeline units)
        A_raw,  A_raw_std = fit_plateau(ns, sp_mean, n0, n1)

        results[q] = {
            'A_norm':     A_norm,      # plateau of rho_norm (dimensionless)
            'A_norm_std': A_std,
            'A_raw':      A_raw,       # plateau of sigma_pair (pipeline units)
            'A_raw_std':  A_raw_std,
            'sigma0':     float(ref),  # sigma_pair at n=1 (= C in C*n^{-delta})
            'ns':         ns,
            'rho_norm':   rho_norm,
            'sp_mean':    sp_mean,
            'n0': n0, 'n1': n1,
        }

        print(f"  sigma_pair(n=1) = {ref:.4g}  (amplitude C)")
        print(f"  A_norm  = {A_norm:.4f} ± {A_std:.4f}   "
              f"(plateau of sigma_pair(n)/sigma_pair(1), window n=[{n0},{n1}])")
        print(f"  A_raw   = {A_raw:.4g} ± {A_raw_std:.4g}   "
              f"(plateau of sigma_pair(n), pipeline units)")

    if not results:
        print("\nNo results.")
        return

    qs     = sorted(results.keys())
    A_norms = np.array([results[q]['A_norm'] for q in qs])
    A_raws  = np.array([results[q]['A_raw']  for q in qs])

    print("\n=== Cross-q stability of A_norm ===")
    print(f"{'q':>6}  {'A_norm':>10}  {'std':>8}  {'A_raw':>12}")
    for q in qs:
        r = results[q]
        print(f"{q:>6}  {r['A_norm']:>10.4f}  {r['A_norm_std']:>8.4f}  "
              f"{r['A_raw']:>12.4g}")
    print(f"\nGrand mean A_norm = {np.mean(A_norms):.4f} ± {np.std(A_norms):.4f}")
    print(f"Relative spread   = {np.std(A_norms)/np.mean(A_norms)*100:.2f}%")
    print(f"\nInterpretation:")
    print(f"  A_norm ≈ 1 means sigma_pair is flat in [n0,n1] → H-w confirmed")
    print(f"  A_norm < 1 means sigma_pair decays within window (expected for power law)")
    print(f"  Stability of A_norm across q is the key test")

    out = {str(q): {k: float(v) if not isinstance(v, np.ndarray) else list(v)
                    for k, v in results[q].items()
                    if not isinstance(v, np.ndarray)}
           for q in qs}
    out['grand_mean_A_norm'] = float(np.mean(A_norms))
    out['grand_std_A_norm']  = float(np.std(A_norms))
    with open('A_extraction_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nResults → A_extraction_results.json")

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            # Left: normalised sigma_pair profiles (H-w test)
            ax = axes[0]
            for q in qs:
                r = results[q]
                ns, rho = r['ns'], r['rho_norm']
                ax.semilogy(ns, rho, marker='.', label=f'q={q}')
                ax.axvspan(r['n0'], r['n1'], alpha=0.08, color='grey')
            ax.axhline(1.0, color='k', lw=0.8, ls='--', label='flat = 1')
            ax.set_xlabel('BFS shell n')
            ax.set_ylabel('σ_pair(n) / σ_pair(1)  [log scale]')
            ax.set_title('Normalised admissibility profile (H-w test)\n'
                         'Grey band = fitting window [n0, n1]')
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.3)

            # Right: A_norm vs q (stability test)
            ax = axes[1]
            A_stds = np.array([results[q]['A_norm_std'] for q in qs])
            ax.errorbar(qs, A_norms, yerr=A_stds, fmt='o-', capsize=4)
            gm = np.mean(A_norms)
            ax.axhline(gm, color='r', ls='--',
                       label=f'grand mean = {gm:.4f}')
            ax.set_xlabel('Prime q')
            ax.set_ylabel('A_norm')
            ax.set_title('Stability of A_norm across primes\n'
                         '(flat → H-w holds, A well-defined)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('A_extraction.pdf', bbox_inches='tight')
            print("Plot → A_extraction.pdf")
        except ImportError:
            print("matplotlib not available.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract A from O25 checkpoints.')
    parser.add_argument('--files', nargs='+',
                        help='npz files (e.g. q29_o25.npz ...)')
    parser.add_argument('--dir', default=None,
                        help='Directory containing q*_o25.npz files')
    parser.add_argument('--window', type=float, default=0.4,
                        help='Fallback tail fraction for plateau (default 0.4)')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    files = list(args.files or [])
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, 'q*_o25.npz')))
        files = [f for f in files if '.v1.' not in f]
    if not files:
        parser.error("Provide --files or --dir")

    run(files, window_frac=args.window, plot=not args.no_plot)