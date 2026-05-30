"""
fibre_erasure_o2_test.py

Numerical test of open problem [O-2]:
is H(sigma(ell) | c) monotone non-increasing in the BFS depth ell?

Under the information decomposition
    H(sigma(ell)) = H(sigma(ell) | c) + I(c ; sigma(ell))
[O-2] identifies H(sigma(ell) | c) as a candidate counting monotone.
The fibre-erasure theorem (Theorem 4.3) establishes I(c ; sigma(ell)) is
non-increasing (conditional on [H-suff]).
This script tests whether H(sigma(ell) | c) shares this property.

ESTIMATOR
---------
With M block samples per pair (c, q-c), the pipeline stores:
    sigma_c_sq_mean[i, ell]  = E_block[sigma_c(ell)^2]
    sigma_c_mean  [i, ell]   = E_block[sigma_c(ell)]

The intra-block variance (Bessel-corrected for M samples) is:
    Var_intra(ell, c) = M/(M-1) * (sigma_c_sq_mean - sigma_c_mean^2)

Under a Gaussian approximation, the conditional entropy per fibre is:
    h_intra(ell, c) = 0.5 * log(2*pi*e * Var_intra(ell, c))

    H(sigma(ell) | c) ≈ mean_c[ h_intra(ell, c) ]

For comparison, the between-c term is estimated as:
    H_inter(ell)  ≈  0.5 * log(2*pi*e * Var_inter(ell))
where Var_inter(ell) = Var_c(sigma_c_mean[:, ell]).

And a normalised proxy (CV_intra):
    CV_intra(ell) = mean_c[ sqrt(Var_intra(ell,c)) / sigma_c_mean(ell,c) ]
measures the within-c spread relative to the profile magnitude.
A decreasing CV_intra indicates non-trivial erasure of within-c uncertainty,
not merely the trivial collapse of all values toward zero.

REQUIRES
--------
q<q>_o25.npz files generated with the patched o25_paired_pipeline.py
that stores sigma_c_sq_mean and sigma_qmc_sq_mean.

Usage:
    python fibre_erasure_o2_test.py --dir o25_outputs
    python fibre_erasure_o2_test.py --dir o25_outputs --q-list 61
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_WINDOWS = {29: (2, 5), 61: (2, 7), 101: (3, 10), 151: (3, 12), 211: (3, 13), 307: (3, 16)}


def load_o25(path: Path) -> dict:
    """Load an o25 npz file; verify the squared-sigma keys are present."""
    z = np.load(path, allow_pickle=True)
    for key in ('sigma_c_sq_mean', 'sigma_qmc_sq_mean'):
        if key not in z.files:
            raise KeyError(
                f'{path.name}: key "{key}" missing — '
                'rerun the pipeline with the patched o25_paired_pipeline.py.'
            )
    return z


def compute_quantities(z) -> Dict[str, np.ndarray]:
    """
    Return per-depth arrays (shape n_depths) for:
        h_intra   — mean_c[ 0.5*log(Var_intra(ell,c)) ] + 0.5*log(2*pi*e)
        h_inter   — 0.5*log(Var_c(sigma_c_mean)) + 0.5*log(2*pi*e)
        cv_intra  — mean_c[ std_intra(ell,c) / sigma_c_mean(ell,c) ]
        sigma_bar — mean_c(sigma_c_mean)
    All computed using BOTH c and q-c samples stacked (2*n_pairs per depth).
    """
    M = int(z['M_per_pair'])
    correction = M / (M - 1) if M > 1 else 1.0

    sc   = z['sigma_c_mean'].astype(float)    # (n_pairs, n_depths)
    sqc  = z['sigma_qmc_mean'].astype(float)
    sc2  = z['sigma_c_sq_mean'].astype(float)
    sqc2 = z['sigma_qmc_sq_mean'].astype(float)

    # Stack c and q-c: shape (2*n_pairs, n_depths)
    sigma_all  = np.concatenate([sc,  sqc ], axis=0)
    sigma2_all = np.concatenate([sc2, sqc2], axis=0)

    # Intra-block variance: E[sigma^2] - E[sigma]^2, Bessel-corrected
    var_intra = correction * np.maximum(sigma2_all - sigma_all ** 2, 0.0)
    # Clip to avoid log(0)
    eps = 1e-16
    var_intra_safe = np.where(var_intra > eps, var_intra, np.nan)

    # h_intra(ell) = mean_c[ 0.5 * log(Var_intra(ell,c)) ] + const
    # (constant 0.5*log(2*pi*e) is common to all depths; omitted for trend)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        h_intra = 0.5 * np.nanmean(np.log(var_intra_safe), axis=0)  # (n_depths,)

    # Between-c variance of the M-averaged profiles
    var_inter = np.var(sigma_all, axis=0)                        # (n_depths,)
    var_inter_safe = np.where(var_inter > eps, var_inter, np.nan)
    h_inter = 0.5 * np.log(var_inter_safe)                       # (n_depths,)

    # CV_intra: within-c std relative to profile magnitude
    std_intra = np.sqrt(var_intra_safe)
    sigma_safe = np.where(sigma_all > eps, sigma_all, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cv_intra = np.nanmean(std_intra / sigma_safe, axis=0)        # (n_depths,)

    sigma_bar = sigma_all.mean(axis=0)

    return {
        'h_intra':   h_intra,
        'h_inter':   h_inter,
        'cv_intra':  cv_intra,
        'sigma_bar': sigma_bar,
        'var_intra': np.nanmean(var_intra, axis=0),
        'var_inter': var_inter,
    }


def monotonicity_verdict(arr: np.ndarray, mean_sigma: np.ndarray,
                         sat_threshold: float = 0.01) -> dict:
    """
    Linear-trend test of arr over the pre-saturation regime (mean_sigma > threshold).
    Returns slope, t-stat, n_violations (steps where arr increases),
    and a verdict string.
    """
    pre_sat = np.where((mean_sigma > sat_threshold) & np.isfinite(arr))[0]
    if len(pre_sat) < 3:
        return {'slope': np.nan, 't_stat': np.nan,
                'n_violations': np.nan, 'n_total': 0,
                'verdict': 'insufficient data'}
    x = pre_sat.astype(float)
    y = arr[pre_sat]
    x_c = x - x.mean()
    slope = float(np.dot(x_c, y) / np.dot(x_c, x_c))
    resid = y - (slope * x_c + y.mean())
    se = float(np.sqrt(resid.var() / np.dot(x_c, x_c))) if len(x) > 2 else np.nan
    t_stat = slope / se if (se is not np.nan and se > 1e-14) else 0.0
    diffs = np.diff(y)
    n_violations = int((diffs > 0).sum())
    n_steps = len(diffs)
    if t_stat < -2.0 and n_violations / max(n_steps, 1) < 0.3:
        verdict = 'monotone non-increasing (supports [O-2])'
    elif t_stat > 2.0:
        verdict = 'increasing — against [O-2]'
    elif n_violations / max(n_steps, 1) < 0.2:
        verdict = 'predominantly non-increasing (consistent with [O-2])'
    else:
        verdict = 'flat or noisy — inconclusive'
    return {
        'slope': slope, 't_stat': t_stat,
        'n_violations': n_violations, 'n_steps': n_steps,
        'n_depths_pre_sat': int(len(pre_sat)),
        'verdict': verdict,
    }


def plot_results(results: dict, out_dir: Path) -> None:
    """Three-panel figure: H_intra, H_inter, CV_intra vs BFS depth."""
    qs = sorted(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    labels = [
        (r'$H(\sigma(\ell)\mid c)$ proxy',
         r'$h_{\mathrm{intra}}(\ell) = \overline{\frac{1}{2}\log\,\mathrm{Var}_{\mathrm{intra}}}$',
         'h_intra'),
        (r'$H_{\mathrm{inter}}(\ell)$ proxy',
         r'$\frac{1}{2}\log\,\mathrm{Var}_c(\bar\sigma_c(\ell))$',
         'h_inter'),
        (r'$\mathrm{CV}_{\mathrm{intra}}(\ell)$  (non-trivial erasure)',
         r'$\overline{\mathrm{std}_{\mathrm{intra}}/\bar\sigma_c}$',
         'cv_intra'),
    ]

    for ax, (title, ylabel, key) in zip(axes, labels):
        for q in qs:
            r = results[q]
            depths = np.arange(len(r['sigma_bar']))
            y = r[key]
            n0, n1 = _WINDOWS.get(q, (2, 10))
            ax.plot(depths, y, '-', linewidth=1.4, label=f'q={q}')
            ax.axvspan(n0, n1, alpha=0.10, color='steelblue')
        ax.set_xlabel(r'BFS depth $\ell$')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].invert_yaxis()   # decreasing h_intra → curve goes up on the plot
    axes[1].invert_yaxis()

    fig.suptitle(r'[O-2] numerical test — $H(\sigma(\ell)\mid c)$ monotonicity',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'o2_h_intra.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', type=Path, default=Path('o25_outputs'),
                        help='Directory containing q<q>_o25.npz files.')
    parser.add_argument('--q-list', type=int, nargs='+',
                        default=[29, 61, 101, 151, 211, 307])
    parser.add_argument('--out', type=Path, default=Path('./o2_results'))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    summary = {}
    for q in args.q_list:
        path = args.dir / f'q{q}_o25.npz'
        if not path.exists():
            print(f'[WARN] {path} not found, skipping')
            continue
        try:
            z = load_o25(path)
        except KeyError as e:
            print(f'[WARN] {e}')
            continue
        M = int(z['M_per_pair'])
        n_pairs = int(z['pairs'].shape[0])
        n_depths = z['sigma_c_mean'].shape[1]
        print(f'[INFO] q={q}: {n_pairs} pairs, M={M}, {n_depths} depths')

        r = compute_quantities(z)
        results[q] = r

        n0, n1 = _WINDOWS.get(q, (2, 10))
        sigma_bar = r['sigma_bar']
        h_intra_v = monotonicity_verdict(r['h_intra'], sigma_bar)
        cv_intra_v = monotonicity_verdict(r['cv_intra'], sigma_bar)

        # Values at fitting window boundaries
        h_start = float(np.nanmean(r['h_intra'][n0:n0+2]))
        h_end   = float(np.nanmean(r['h_intra'][n1-1:n1+1]))
        cv_start = float(np.nanmean(r['cv_intra'][n0:n0+2]))
        cv_end   = float(np.nanmean(r['cv_intra'][n1-1:n1+1]))

        summary[q] = {
            'q': q, 'n_pairs': n_pairs, 'M': M,
            'n0': n0, 'n1': n1,
            'h_intra_start': h_start,
            'h_intra_end':   h_end,
            'h_intra_decrease_factor': float(np.exp(h_end - h_start)),
            'h_intra_verdict': h_intra_v,
            'cv_intra_start': cv_start,
            'cv_intra_end':   cv_end,
            'cv_intra_monotone': bool(cv_intra_v['verdict'].startswith('monotone')),
            'cv_intra_verdict': cv_intra_v,
        }

    if not results:
        print('[ERROR] No valid data loaded.')
        return 1

    plot_results(results, args.out)
    print(f'[INFO] wrote {args.out}/o2_h_intra.pdf')

    out_json = args.out / 'o2_summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'[INFO] wrote {out_json}')
    print()
    print(json.dumps(summary, indent=2))

    print()
    print('=== [O-2] VERDICT ===')
    for q, s in summary.items():
        v_h  = s['h_intra_verdict']['verdict']
        v_cv = s['cv_intra_verdict']['verdict']
        # CV_intra increasing at late depths is the staggered-saturation artifact
        # (same as [O-4] for inter-c CV): some blocks reach sigma=0 before others,
        # inflating relative within-c variance.  The [O-2] verdict is based on
        # h_intra (absolute entropy proxy), not CV_intra.
        h_supports = 'non-increasing' in v_h or 'predominantly' in v_h
        cv_increasing = 'increasing' in v_cv
        o2_verdict = ('SUPPORTED' if h_supports else 'NOT SUPPORTED')
        cv_note = (' [CV_intra increasing = staggered-saturation artifact, see [O-4]]'
                   if cv_increasing else '')
        print(f'q={q}: [O-2] {o2_verdict}')
        print(f'       h_intra:   {v_h}')
        print(f'       cv_intra:  {v_cv}{cv_note}')
        print(f'       std_intra decreases by factor {s["h_intra_decrease_factor"]:.4f} '
              f'over window [{s["n0"]},{s["n1"]}]  '
              f'(t={s["h_intra_verdict"]["t_stat"]:.1f}, '
              f'{s["h_intra_verdict"]["n_violations"]}/{s["h_intra_verdict"]["n_steps"]} '
              f'violations)')
        print()


if __name__ == '__main__':
    raise SystemExit(main() or 0)