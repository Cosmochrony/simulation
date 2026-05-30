"""
fibre_erasure_o4_axis.py

Numerical investigation of open problem [O-4]: axis disambiguation.

Determines whether the fibre-erasure fixed point sigma* is approached as a
limit in the coarse-graining depth ell at fixed q (a Wilsonian fixed point
of the semigroup), or only in the thermodynamic limit q -> inf.

The discriminating diagnostic is the coefficient of variation

    CV(ell) = std_c(sigma_c(ell)) / mean_c(sigma_c(ell))

evaluated as a function of BFS depth ell for each prime q.

  If sigma_c(ell) = A_c * sigma*(ell) for a c-dependent amplitude A_c
  and a c-independent profile sigma*(ell), then CV(ell) = std(A_c)/mean(A_c)
  is CONSTANT across ell: the profiles have equal exponents but different
  amplitudes.  No convergence to a common profile at fixed q.

  If CV(ell) -> 0 before BFS saturation (i.e. before mean(sigma) -> 0),
  then the profiles converge to a common sigma*(ell): genuine Wilsonian
  fixed point at fixed q.

A secondary diagnostic is the cross-q mean profile comparison:
after normalising by their value at a reference depth, do the mean profiles
for different q collapse onto the same curve?  Collapse = universal profile;
no collapse = profiles differ at finite q and only agree as q -> inf.

Expected data format (same as fibre_erasure_hsuff_test.py):
    q_<q>.npz
        c     : int   array, shape (n_samples,)           -- fibre label
        sigma : float array, shape (n_samples, n_depths)  -- BFS capacity profile
        q     : int                                        -- prime
        n0, n1 : int                                       -- fitting window

Usage:
    python fibre_erasure_o4_axis.py --data-dir ./hsuff_input --out ./o4_results
    python fibre_erasure_o4_axis.py --synthetic            # for testing

Dependencies: numpy, pandas, matplotlib, scipy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_FITTING_WINDOWS = {61: (2, 7), 151: (3, 12), 211: (3, 13), 307: (3, 16)}


def load_npz_prime(path: Path) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Load (c, sigma, q, n0, n1) from a per-prime .npz file."""
    z = np.load(path)
    c = z['c']
    sigma = z['sigma']
    q = int(z['q']) if 'q' in z.files else int(
        path.stem.replace('q_', '').replace('q', ''))
    n0 = int(z['n0']) if 'n0' in z.files else _FITTING_WINDOWS.get(q, (2, 10))[0]
    n1 = int(z['n1']) if 'n1' in z.files else _FITTING_WINDOWS.get(q, (2, 10))[1]
    return c, sigma, q, n0, n1


def make_synthetic(q_list, n_depths=80, seed=42,
                   mode='amplitude') -> Dict[int, Tuple]:
    """
    Synthetic data for testing.

    mode='amplitude': sigma_c(l) = A_c * f(l), where A_c differs across c
                      and f(l) is the same for all c.
                      -> CV constant, no Wilsonian fixed point.
    mode='converge':  sigma_c(l) -> common sigma*(l) for large l.
                      -> CV -> 0, Wilsonian fixed point present.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for q in q_list:
        n_pairs = (q - 1) // 2
        # c values from 1 to q-1 (both sides of pair)
        c_vals = list(range(1, n_pairs + 1)) + list(range(n_pairs + 1, q))
        c = np.array(c_vals, dtype=np.int64)
        n_c = len(c)
        sigma = np.ones((n_c, n_depths))
        # c-dependent amplitudes (O(q^{-1/2}) variation)
        amp = 1.0 + 0.15 / np.sqrt(q) * rng.standard_normal(n_c)
        for l in range(n_depths - 1):
            base = np.exp(-0.04 + 0.008 * rng.standard_normal(n_c))
            if mode == 'amplitude':
                sigma[:, l + 1] = np.clip(sigma[:, l] * base, 0, 1)
            else:  # converge
                conv_rate = 0.05 * l / n_depths  # convergence increases with depth
                sigma[:, l + 1] = np.clip(
                    sigma[:, l] * base * (1 - conv_rate) + conv_rate * amp.mean(),
                    0, 1)
        sigma = sigma * amp[:, None]
        sigma = np.clip(sigma, 0, 1)
        n0, n1 = _FITTING_WINDOWS.get(q, (2, 10))
        data[q] = (c, sigma, q, n0, n1)
    return data


def compute_cv_profile(sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std, and CV across fibre labels c at each depth.

    Returns (mean, std, cv) arrays of shape (n_depths,).
    CV is set to NaN where mean < 1e-6 (saturation regime).
    """
    mean = sigma.mean(axis=0)
    std = sigma.std(axis=0)
    cv = np.where(mean > 1e-6, std / mean, np.nan)
    return mean, std, cv


def cv_trend(cv: np.ndarray, mean: np.ndarray,
             sat_threshold: float = 0.01) -> dict:
    """
    Assess whether CV is flat, decreasing, or increasing in the pre-saturation regime.

    The pre-saturation regime is defined by mean_sigma > sat_threshold.
    A flat CV confirms amplitude-only variation (universal exponent, c-dependent
    amplitudes); the fixed point sigma* requires q -> inf.
    A significantly decreasing CV suggests convergence at fixed q.
    """
    pre_sat = (mean > sat_threshold) & ~np.isnan(cv)
    idx = np.where(pre_sat)[0]
    if len(idx) < 5:
        return {'slope': np.nan, 'slope_se': np.nan, 'norm_slope': np.nan,
                'cv_start': np.nan, 'cv_end': np.nan,
                'verdict': 'insufficient data'}
    x = idx.astype(float)
    y = cv[idx]
    x_c = x - x.mean()
    slope = float(np.dot(x_c, y) / np.dot(x_c, x_c))
    resid = y - (slope * x_c + y.mean())
    se = float(np.sqrt(resid.var() / np.dot(x_c, x_c))) if len(x) > 2 else np.nan
    norm_slope = slope / max(np.nanmedian(y), 1e-8)
    t_stat = slope / se if se > 1e-12 else 0.0
    cv_start = float(np.nanmean(cv[idx[:3]]))
    cv_end = float(np.nanmean(cv[idx[-3:]]))
    rel_change = (cv_end - cv_start) / max(cv_start, 1e-8)
    if t_stat < -2.0 and rel_change < -0.10:
        verdict = 'decreasing — candidate Wilsonian fixed point at fixed q'
    elif t_stat > 2.0 and rel_change > 0.10:
        verdict = 'increasing — profiles diverging (likely noise-dominated at late depths)'
    else:
        verdict = 'flat — amplitude-only variation; fixed point likely requires q->inf'
    return {
        'slope': slope, 'slope_se': se, 't_stat': t_stat,
        'norm_slope': norm_slope,
        'cv_start': cv_start, 'cv_end': cv_end,
        'rel_change': rel_change,
        'n_depths_pre_sat': int(len(idx)),
        'verdict': verdict,
    }


def amplitude_ratio_test(sigma: np.ndarray, mean: np.ndarray,
                         sat_threshold: float = 0.01, n_pairs: int = 20) -> dict:
    """
    Test whether profiles differ only in amplitude (same shape) or in exponent.

    For a random sample of (c, c') pairs, compute the ratio r(ell) = sigma_c(ell) /
    sigma_{c'}(ell) in the pre-saturation regime.
    If all ratios are constant across ell (std_ell(r) / mean_ell(r) is small),
    profiles differ only by amplitude and share the same shape/exponent.
    If ratios drift with ell, exponents differ.

    Returns mean and std of the relative ell-variation of the ratio.
    """
    pre_sat = mean > sat_threshold
    n_c = sigma.shape[0]
    if n_c < 2 or pre_sat.sum() < 3:
        return {'ratio_cv_mean': np.nan, 'ratio_cv_std': np.nan,
                'verdict': 'insufficient data'}
    rng = np.random.default_rng(0)
    idx_c = rng.choice(n_c, size=min(n_pairs * 2, n_c), replace=False)
    pairs = [(idx_c[i], idx_c[i + 1]) for i in range(0, len(idx_c) - 1, 2)]
    ratio_cvs = []
    for i, j in pairs:
        denom = sigma[j, pre_sat]
        valid = denom > 1e-8
        if valid.sum() < 3:
            continue
        ratio = sigma[i, pre_sat][valid] / denom[valid]
        cv_r = ratio.std() / (ratio.mean() + 1e-12)
        ratio_cvs.append(cv_r)
    if not ratio_cvs:
        return {'ratio_cv_mean': np.nan, 'ratio_cv_std': np.nan,
                'verdict': 'insufficient data'}
    rcv_mean = float(np.mean(ratio_cvs))
    rcv_std = float(np.std(ratio_cvs))
    if rcv_mean < 0.02:
        verdict = 'ratio CV very small: same shape, amplitude-only variation confirmed'
    elif rcv_mean < 0.10:
        verdict = 'ratio CV moderate: predominantly same shape, minor exponent differences'
    else:
        verdict = 'ratio CV large: significant shape differences between c profiles'
    return {
        'ratio_cv_mean': rcv_mean, 'ratio_cv_std': rcv_std,
        'n_pairs_tested': len(ratio_cvs),
        'verdict': verdict,
    }


def cross_q_comparison(data: dict) -> pd.DataFrame:
    """
    Compare normalised mean profiles across q values.

    For each q, normalise sigma_bar(ell) by its value at n0 (start of window).
    If profiles collapse onto the same curve, sigma* is q-independent.
    Returns a DataFrame with columns: q, ell, sigma_bar_norm.
    """
    rows = []
    for q, (c, sigma, q_val, n0, n1) in data.items():
        mean = sigma.mean(axis=0)
        ref = mean[n0] if mean[n0] > 1e-8 else 1.0
        for ell in range(len(mean)):
            rows.append({'q': q_val, 'ell': ell, 'ell_rel': ell - n0,
                         'sigma_bar': mean[ell],
                         'sigma_bar_norm': mean[ell] / ref,
                         'n0': n0, 'n1': n1})
    return pd.DataFrame(rows)


def plot_results(data: dict, df_cross: pd.DataFrame, out_dir: Path) -> None:
    """Generate two figures: CV profiles and cross-q normalised mean profiles."""
    qs = sorted(data.keys())
    n_q = len(qs)

    fig, axes = plt.subplots(2, n_q, figsize=(4.5 * n_q, 7), squeeze=False)

    for col, q in enumerate(qs):
        c, sigma, q_val, n0, n1 = data[q]
        mean, std, cv = compute_cv_profile(sigma)
        depths = np.arange(len(mean))

        ax = axes[0, col]
        ax.plot(depths, cv, 'k-', linewidth=1.4)
        ax.axvspan(n0, n1, alpha=0.12, color='blue', label=f'window [{n0},{n1}]')
        ax.axhline(np.nanmedian(cv[n0:n1 + 1]), color='r', linestyle='--',
                   linewidth=0.8, label='median in window')
        ax.set_title(fr'$q={q_val}$ — $\mathrm{{CV}}(\ell)$')
        ax.set_xlabel(r'BFS depth $\ell$')
        ax.set_ylabel(r'$\mathrm{CV}(\ell) = \sigma_c / \bar{\sigma}$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Individual profiles (random sample of 8 c values)
        ax2 = axes[1, col]
        rng = np.random.default_rng(q)
        sample_idx = rng.choice(len(sigma), size=min(8, len(sigma)), replace=False)
        for idx in sample_idx:
            ax2.semilogy(depths, np.clip(sigma[idx], 1e-10, 1),
                         alpha=0.5, linewidth=0.9)
        ax2.semilogy(depths, np.clip(mean, 1e-10, 1),
                     'k-', linewidth=2.0, label=r'$\bar{\sigma}(\ell)$')
        ax2.axvspan(n0, n1, alpha=0.12, color='blue')
        ax2.set_xlabel(r'BFS depth $\ell$')
        ax2.set_ylabel(r'$\sigma_c(\ell)$ (log scale)')
        ax2.set_title(f'q={q_val} — sample profiles')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'o4_cv_profiles.pdf')
    plt.close(fig)

    # Cross-q normalised mean profiles
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        sub = df_cross[(df_cross['q'] == q) & (df_cross['ell_rel'] >= 0)]
        n1_rel = int(sub['n1'].iloc[0]) - int(sub['n0'].iloc[0])
        window_sub = sub[sub['ell_rel'] <= n1_rel + 5]
        ax.semilogy(window_sub['ell_rel'], window_sub['sigma_bar_norm'],
                    '-', linewidth=1.5, label=f'q={q}')
    ax.axhline(1.0, color='k', linewidth=0.5)
    ax.set_xlabel(r'$\ell - \ell_0$ (depth relative to fitting window start)')
    ax.set_ylabel(r'$\bar{\sigma}(\ell) / \bar{\sigma}(\ell_0)$ (normalised)')
    ax.set_title(r'Cross-$q$ mean profile comparison (collapse = $q$-universal $\sigma^*$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'o4_cross_q.pdf')
    plt.close(fig2)


def make_summary(data: dict) -> dict:
    """Per-q summary of CV trend, amplitude-ratio test, and [O-4] verdict."""
    summary = {}
    for q, (c, sigma, q_val, n0, n1) in data.items():
        mean, std, cv = compute_cv_profile(sigma)
        trend = cv_trend(cv, mean)
        amp = amplitude_ratio_test(sigma, mean)
        summary[int(q_val)] = {
            'n0': n0, 'n1': n1,
            'cv_median': float(np.nanmedian(cv[~np.isnan(cv)])),
            **{f'cv_{k}': v for k, v in trend.items()},
            **{f'amp_{k}': v for k, v in amp.items()},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Directory containing q_<q>.npz files.')
    parser.add_argument('--out', type=Path, default=Path('./o4_results'),
                        help='Output directory.')
    parser.add_argument('--q-list', type=int, nargs='+',
                        default=[61, 151, 211, 307])
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data.')
    parser.add_argument('--synthetic-mode', choices=['amplitude', 'converge'],
                        default='amplitude',
                        help='amplitude=CV flat (expected), converge=CV decreasing.')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print(f'[INFO] generating synthetic data (mode={args.synthetic_mode})')
        data = make_synthetic(args.q_list, mode=args.synthetic_mode)
    elif args.data_dir is not None:
        data = {}
        for q in args.q_list:
            for candidate in [args.data_dir / f'q_{q}.npz',
                               args.data_dir / f'q{q}.npz']:
                if candidate.exists():
                    c, sigma, q_real, n0, n1 = load_npz_prime(candidate)
                    data[q] = (c, sigma, q_real, n0, n1)
                    print(f'[INFO] q={q_real}: {len(c)} samples, '
                          f'{sigma.shape[1]} depths, window=[{n0},{n1}]')
                    break
            else:
                print(f'[WARN] q_{q}.npz not found in {args.data_dir}, skipping')
        if not data:
            parser.error('No data loaded.')
    else:
        parser.error('Either --data-dir or --synthetic must be specified.')

    df_cross = cross_q_comparison(data)
    plot_results(data, df_cross, args.out)
    print(f'[INFO] wrote {args.out}/o4_cv_profiles.pdf')
    print(f'[INFO] wrote {args.out}/o4_cross_q.pdf')

    summary = make_summary(data)
    out_json = args.out / 'o4_summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'[INFO] wrote {out_json}')
    print()
    print(json.dumps(summary, indent=2))

    print()
    print('=== [O-4] VERDICT ===')
    for q, v in summary.items():
        cv_v = v.get('cv_verdict', 'N/A')
        amp_v = v.get('amp_verdict', 'N/A')
        print(f'q={q}: CV trend: {cv_v}')
        print(f'       Amplitude ratio: {amp_v}')
        print()


if __name__ == '__main__':
    main()