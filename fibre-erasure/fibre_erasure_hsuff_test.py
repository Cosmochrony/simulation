"""
fibre_erasure_hsuff_test.py

Numerical test of hypothesis [H-suff] for the Fibre Erasure programme
(Paper FibreErasure_1_0, Open problem [O-1]).

Under [H-suff], the BFS depth-increment kernel
    p(sigma(l+1) | sigma(l), c) = p(sigma(l+1) | sigma(l))
is conditionally independent of the fibre label c given sigma(l).

This script tests that condition for each depth step l -> l+1 via three
complementary statistics on the k-NN regression residuals
    e_i = sigma_i(l+1) - E_{j ~ NN(i)}[ sigma_j(l+1) ]:

  (A) eta_squared(e | c)           -- variance fraction explained by c as
                                       a categorical label.  Requires at
                                       least 2 samples per c value (used in
                                       --synthetic mode).
  (B) |Spearman corr(e, c)|        -- monotone dependence of residuals on c.
                                       Works with any number of samples per c.
  (C) eta_squared(e_loto | triplet)  -- variance fraction explained by SU(3)
                                       colour-triplet membership, using
                                       leave-triplet-out (LOTO) k-NN residuals.
                                       Computed when q ≡ 1 (mod 3).
                                       LOTO prevents the k-NN artifact that
                                       arises when [H-color] holds: if triplet
                                       members have nearly equal sigma(l) they
                                       are close neighbours, making standard
                                       residuals spuriously correlated within
                                       triplets.  LOTO excludes triplet peers
                                       from the k-NN pool for each sample.

Each test has its own permutation-based p-value (shuffle c labels, recompute).
Under [H-suff], all applicable statistics should be statistically
indistinguishable from their permutation null distributions.

Expected data format (one .npz per prime, e.g., produced by build_hsuff_input.py):
    q_<q>.npz
        c     : int   array, shape (n_samples,)           -- fibre label
        sigma : float array, shape (n_samples, n_depths)  -- BFS capacity profile
        q     : int                                        -- prime (optional;
                                                              falls back to filename)

Usage:
    python fibre_erasure_hsuff_test.py --data-dir ./hsuff_input --out ./results
    python fibre_erasure_hsuff_test.py --synthetic                 # Markov sanity check
    python fibre_erasure_hsuff_test.py --synthetic --violation     # sensitivity check

Dependencies: numpy, pandas, matplotlib, scikit-learn, scipy, joblib.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


_QFILE_RE = re.compile(r"^q_?(\d+)\.npz$")


def find_omega(q: int) -> int | None:
    """Return a primitive cube root of unity modulo q, or None if q != 1 (mod 3)."""
    if q % 3 != 1:
        return None
    for g in range(2, q):
        omega = pow(g, (q - 1) // 3, q)
        if omega != 1 and pow(omega, 3, q) == 1:
            return omega
    return None


def triplet_class(c_array: np.ndarray, omega: int, q: int) -> np.ndarray:
    """Map each c to a canonical triplet representative: min({c, omega*c, omega^2*c} mod q)."""
    t1 = c_array.astype(np.int64) % q
    t2 = (c_array.astype(np.int64) * omega) % q
    t3 = (c_array.astype(np.int64) * omega * omega) % q
    return np.minimum(np.minimum(t1, t2), t3)


def eta_squared(residuals: np.ndarray, group: np.ndarray) -> float:
    """
    Fraction of variance of residuals explained by a discrete group label.

    eta^2 = 1 - sum_k (n_k * Var_k) / (n * Var_total)
          = (between-group variance) / (total variance)
    """
    var_total = float(residuals.var())
    if var_total < 1e-20:
        return 0.0
    _, idx, counts = np.unique(group, return_inverse=True, return_counts=True)
    within = 0.0
    for k in range(len(counts)):
        mask = (idx == k)
        if mask.sum() > 1:
            within += mask.sum() * float(residuals[mask].var())
    within /= len(residuals)
    return max(0.0, 1.0 - within / var_total)


def _permutation_pvalue(stat_obs: float, stat_null: np.ndarray) -> Tuple[float, float, float]:
    """Right-tail p-value, null mean, z-score for an upper-tailed test statistic."""
    null_mean = float(stat_null.mean())
    null_std = float(stat_null.std())
    p_value = float((stat_null >= stat_obs).mean())
    z_score = (stat_obs - null_mean) / null_std if null_std > 1e-12 else 0.0
    return p_value, null_mean, z_score


def hsuff_test_depth(sigma_l: np.ndarray, sigma_lp1: np.ndarray,
                     c: np.ndarray, q: int,
                     n_permutations: int = 500,
                     k_neighbors: int | None = None,
                     seed: int = 0) -> dict:
    """
    Run the three [H-suff] tests at a single depth step.

    Returns a dict with all three statistics, their permutation null means
    and p-values, plus z-scores.  Tests that do not apply at this data
    configuration are set to NaN (with explicit `*_applicable` flags).
    """
    n = len(sigma_l)
    if k_neighbors is None:
        k_neighbors = max(10, int(np.sqrt(n)))
    k_neighbors = min(k_neighbors, n - 1)

    tree = NearestNeighbors(n_neighbors=k_neighbors + 1)
    tree.fit(sigma_l.reshape(-1, 1))
    _, idx = tree.kneighbors(sigma_l.reshape(-1, 1))
    idx_no_self = idx[:, 1:]
    e_hat = sigma_lp1[idx_no_self].mean(axis=1)
    residuals = sigma_lp1 - e_hat

    out: dict = {}

    _, counts = np.unique(c, return_counts=True)
    cat_applicable = bool(counts.min() >= 2)
    out['cat_applicable'] = cat_applicable
    if cat_applicable:
        rng_a = np.random.default_rng(seed)
        eta_obs = eta_squared(residuals, c)
        eta_null = np.array([eta_squared(residuals, rng_a.permutation(c))
                             for _ in range(n_permutations)])
        p, mu, z = _permutation_pvalue(eta_obs, eta_null)
        out['eta2_c_obs'] = eta_obs
        out['eta2_c_null_mean'] = mu
        out['eta2_c_z'] = z
        out['eta2_c_p'] = p
    else:
        out['eta2_c_obs'] = out['eta2_c_null_mean'] = out['eta2_c_z'] = out['eta2_c_p'] = np.nan

    rng_b = np.random.default_rng(seed + 1)
    rho_obs, _ = spearmanr(residuals, c)
    rho_obs = abs(float(rho_obs)) if not np.isnan(rho_obs) else 0.0
    rho_null = np.empty(n_permutations)
    for i in range(n_permutations):
        r, _ = spearmanr(residuals, rng_b.permutation(c))
        rho_null[i] = abs(r) if not np.isnan(r) else 0.0
    p, mu, z = _permutation_pvalue(rho_obs, rho_null)
    out['spearman_abs_rho_obs'] = rho_obs
    out['spearman_null_mean'] = mu
    out['spearman_z'] = z
    out['spearman_p'] = p

    omega = find_omega(int(q))
    tri_applicable = omega is not None
    out['tri_applicable'] = tri_applicable
    if tri_applicable:
        triplet = triplet_class(c, omega, int(q))
        # Use leave-triplet-out residuals to prevent k-NN leakage between
        # triplet peers when [H-color] makes their sigma(l) values nearly equal.
        loto_res = _loto_residuals(sigma_l, sigma_lp1, triplet, k_neighbors)
        rng_c = np.random.default_rng(seed + 2)
        eta_t_obs = eta_squared(loto_res, triplet)
        eta_t_null = np.array([eta_squared(loto_res, rng_c.permutation(triplet))
                               for _ in range(n_permutations)])
        p, mu, z = _permutation_pvalue(eta_t_obs, eta_t_null)
        out['eta2_triplet_obs'] = eta_t_obs
        out['eta2_triplet_null_mean'] = mu
        out['eta2_triplet_z'] = z
        out['eta2_triplet_p'] = p
    else:
        out['eta2_triplet_obs'] = out['eta2_triplet_null_mean'] = np.nan
        out['eta2_triplet_z'] = out['eta2_triplet_p'] = np.nan

    return out


def _loto_residuals(sigma_l: np.ndarray, sigma_lp1: np.ndarray,
                    group: np.ndarray, k_neighbors: int) -> np.ndarray:
    """
    Leave-group-out k-NN regression residuals.

    For each sample i, predict sigma_lp1[i] using the k nearest neighbours
    of sigma_l[i] drawn exclusively from samples j with group[j] != group[i].
    This prevents prediction leakage between group members that would
    otherwise create spurious within-group residual correlation when group
    members share similar sigma(l) values (e.g., colour triplets under [H-color]).
    """
    n = len(sigma_l)
    e_hat = np.empty(n)
    unique_groups = np.unique(group)
    for i in range(n):
        out_mask = (group != group[i])
        sigma_l_out = sigma_l[out_mask]
        sigma_lp1_out = sigma_lp1[out_mask]
        k = min(k_neighbors, int(out_mask.sum()) - 1)
        if k < 1:
            e_hat[i] = float(sigma_lp1_out.mean()) if len(sigma_lp1_out) else sigma_lp1[i]
            continue
        dists = np.abs(sigma_l_out - sigma_l[i])
        nn_idx = np.argpartition(dists, k)[:k]
        e_hat[i] = sigma_lp1_out[nn_idx].mean()
    return sigma_lp1 - e_hat


def make_synthetic(q_list, n_depths: int = 80, M: int = 50,
                   seed: int = 42, violation: bool = False,
                   violation_strength: float = 0.05
                   ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic BFS-like data. See main docstring of test_hsuff for behaviour."""
    rng = np.random.default_rng(seed)
    data = {}
    for q in q_list:
        c_values = np.arange(1, (q - 1) // 2 + 1)
        c = np.repeat(c_values, M)
        n_samples = len(c)
        sigma = np.ones((n_samples, n_depths))
        if violation:
            c_norm = (c_values - c_values.mean()) / max(c_values.std(), 1.0)
            drift = violation_strength * np.repeat(c_norm, M)
        for l in range(n_depths - 1):
            log_sigma = np.log(np.clip(sigma[:, l], 1e-10, 1.0))
            noise = 0.01 * rng.standard_normal(n_samples)
            new_log_sigma = log_sigma - 0.04 + noise
            if violation:
                new_log_sigma -= drift
            sigma[:, l + 1] = np.clip(np.exp(new_log_sigma), 0.0, 1.0)
        data[q] = (c, sigma)
    return data


def load_npz_prime(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load (c, sigma, q) from a per-prime .npz file."""
    z = np.load(path)
    c = z['c']
    sigma = z['sigma']
    if 'q' in z.files:
        q = int(z['q'])
    else:
        m = _QFILE_RE.match(path.name)
        if m is None:
            raise ValueError(f"Cannot infer q from filename {path.name}")
        q = int(m.group(1))
    return c, sigma, q


def run_depth(q: int, l: int, c: np.ndarray, sigma: np.ndarray,
              n_permutations: int) -> dict:
    """Run [H-suff] tests for one depth step.  Skip degenerate cases."""
    sigma_l = sigma[:, l]
    sigma_lp1 = sigma[:, l + 1]
    if sigma_l.std() < 1e-10 or sigma_lp1.std() < 1e-10:
        return None
    out = hsuff_test_depth(sigma_l, sigma_lp1, c, q,
                           n_permutations=n_permutations, seed=l)
    out.update({
        'q': q,
        'l': l,
        'n_samples': int(len(c)),
        'sigma_l_mean': float(sigma_l.mean()),
        'sigma_l_std': float(sigma_l.std()),
    })
    return out


def run_q(q: int, c: np.ndarray, sigma: np.ndarray,
          depth_stride: int = 1, n_permutations: int = 500,
          n_jobs: int = -1) -> pd.DataFrame:
    """Run [H-suff] tests over all depth steps for a single prime q."""
    depths = list(range(0, sigma.shape[1] - 1, depth_stride))
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(run_depth)(q, l, c, sigma, n_permutations) for l in depths
    )
    return pd.DataFrame([r for r in results if r is not None])


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    """Plot the three test statistics (z-scores) and their p-values across depth."""
    has_cat = df['cat_applicable'].any()
    has_tri = df['tri_applicable'].any()
    n_panels_top = 1 + int(has_cat) + int(has_tri)
    fig, axes = plt.subplots(2, n_panels_top, figsize=(4.5 * n_panels_top, 7),
                             sharex='col')
    if n_panels_top == 1:
        axes = axes.reshape(2, 1)

    col = 0

    if has_cat:
        ax = axes[0, col]
        for q in sorted(df['q'].unique()):
            sub = df[(df['q'] == q) & df['cat_applicable']].sort_values('l')
            if len(sub):
                ax.plot(sub['l'], sub['eta2_c_z'], '-', linewidth=1.4, label=f'q={q}')
        ax.axhline(0.0, color='k', linewidth=0.5)
        ax.axhline(2.0, color='r', linestyle=':', linewidth=0.8)
        ax.set_ylabel(r'$z$ score')
        ax.set_title(r'(A) $\eta^2$ by $c$ (categorical)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_p = axes[1, col]
        for q in sorted(df['q'].unique()):
            sub = df[(df['q'] == q) & df['cat_applicable']].sort_values('l')
            if len(sub):
                ax_p.plot(sub['l'], sub['eta2_c_p'], '-', linewidth=1.4, label=f'q={q}')
        ax_p.axhline(0.05, color='k', linestyle=':', linewidth=0.8)
        ax_p.set_xlabel(r'BFS depth $\ell$')
        ax_p.set_ylabel(r'permutation $p$-value')
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.grid(True, alpha=0.3)
        col += 1

    ax = axes[0, col]
    for q in sorted(df['q'].unique()):
        sub = df[df['q'] == q].sort_values('l')
        ax.plot(sub['l'], sub['spearman_z'], '-', linewidth=1.4, label=f'q={q}')
    ax.axhline(0.0, color='k', linewidth=0.5)
    ax.axhline(2.0, color='r', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'$z$ score')
    ax.set_title(r'(B) Spearman $|\rho|(e, c)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax_p = axes[1, col]
    for q in sorted(df['q'].unique()):
        sub = df[df['q'] == q].sort_values('l')
        ax_p.plot(sub['l'], sub['spearman_p'], '-', linewidth=1.4, label=f'q={q}')
    ax_p.axhline(0.05, color='k', linestyle=':', linewidth=0.8)
    ax_p.set_xlabel(r'BFS depth $\ell$')
    ax_p.set_ylabel(r'permutation $p$-value')
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.grid(True, alpha=0.3)
    col += 1

    if has_tri:
        ax = axes[0, col]
        for q in sorted(df['q'].unique()):
            sub = df[(df['q'] == q) & df['tri_applicable']].sort_values('l')
            if len(sub):
                ax.plot(sub['l'], sub['eta2_triplet_z'], '-', linewidth=1.4, label=f'q={q}')
        ax.axhline(0.0, color='k', linewidth=0.5)
        ax.axhline(2.0, color='r', linestyle=':', linewidth=0.8)
        ax.set_ylabel(r'$z$ score')
        ax.set_title(r'(C) $\eta^2$ by triplet — [H-color] co-admissibility'
                     '\n(high $z$ supports [H-color], not a [H-suff] violation)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_p = axes[1, col]
        for q in sorted(df['q'].unique()):
            sub = df[(df['q'] == q) & df['tri_applicable']].sort_values('l')
            if len(sub):
                ax_p.plot(sub['l'], sub['eta2_triplet_p'], '-', linewidth=1.4, label=f'q={q}')
        ax_p.axhline(0.05, color='k', linestyle=':', linewidth=0.8)
        ax_p.set_xlabel(r'BFS depth $\ell$')
        ax_p.set_ylabel(r'permutation $p$-value')
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _verdict(n_signif: int, n_total: int) -> Tuple[float, bool]:
    """Standardised excess of significant depths over null expectation; verdict."""
    expected = 0.05 * n_total
    excess = n_signif - expected
    excess_sigma = excess / max(np.sqrt(expected * 0.95), 1.0)
    return float(excess_sigma), bool(excess_sigma < 3.0)


def make_summary(df: pd.DataFrame) -> dict:
    """Per-q summary; verdict is the joint result over the applicable tests."""
    summary = {}
    for q in sorted(df['q'].unique()):
        sub = df[df['q'] == q]
        n_total = int(len(sub))
        n_cat = int(sub['cat_applicable'].any())
        n_tri = int(sub['tri_applicable'].any())

        entry: dict = {'q': int(q), 'n_depth_steps': n_total}

        if sub['cat_applicable'].any():
            cat_sub = sub[sub['cat_applicable']]
            n_signif = int((cat_sub['eta2_c_p'] < 0.05).sum())
            excess, ok = _verdict(n_signif, len(cat_sub))
            entry['cat_eta2_max'] = float(cat_sub['eta2_c_obs'].max())
            entry['cat_z_max'] = float(cat_sub['eta2_c_z'].max())
            entry['cat_n_signif'] = n_signif
            entry['cat_excess_sigma'] = excess
            entry['cat_consistent'] = ok

        sp_signif = int((sub['spearman_p'] < 0.05).sum())
        sp_excess, sp_ok = _verdict(sp_signif, n_total)
        entry['spearman_abs_rho_max'] = float(sub['spearman_abs_rho_obs'].max())
        entry['spearman_z_max'] = float(sub['spearman_z'].max())
        entry['spearman_n_signif'] = sp_signif
        entry['spearman_excess_sigma'] = sp_excess
        entry['spearman_consistent'] = sp_ok

        if sub['tri_applicable'].any():
            tri_sub = sub[sub['tri_applicable']]
            n_signif = int((tri_sub['eta2_triplet_p'] < 0.05).sum())
            excess, ok = _verdict(n_signif, len(tri_sub))
            entry['tri_eta2_max'] = float(tri_sub['eta2_triplet_obs'].max())
            entry['tri_z_max'] = float(tri_sub['eta2_triplet_z'].max())
            entry['tri_n_signif'] = n_signif
            entry['tri_excess_sigma'] = excess
            entry['tri_consistent'] = ok

        consistencies = []
        if 'cat_consistent' in entry:
            consistencies.append(entry['cat_consistent'])
        consistencies.append(entry['spearman_consistent'])
        entry['consistent_with_hsuff'] = bool(all(consistencies))

        # Triplet test (C) measures [H-color] co-admissibility quality, NOT [H-suff].
        # When [H-color] holds, triplet members share equal sigma profiles, so LOTO
        # residuals within triplets are nearly equal and eta^2 exceeds the permutation
        # null.  A high tri_z therefore SUPPORTS [H-color] rather than indicating a
        # [H-suff] violation.  The field below records the triplet result separately.
        if 'tri_consistent' in entry:
            entry['hcolor_coadmissibility_signal'] = not entry['tri_consistent']
            if entry['hcolor_coadmissibility_signal']:
                entry['hcolor_note'] = (
                    'tri_consistent=False: strong within-triplet residual '
                    'correlation detected, consistent with [H-color] co-admissibility. '
                    'This does NOT indicate a [H-suff] violation.'
                )

        summary[int(q)] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Directory containing q_<q>.npz files (output of build_hsuff_input.py).')
    parser.add_argument('--out', type=Path, default=Path('./results'),
                        help='Output directory (created if missing).')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of loading from disk.')
    parser.add_argument('--violation', action='store_true',
                        help='With --synthetic, inject a c-dependent drift.')
    parser.add_argument('--violation-strength', type=float, default=0.05,
                        help='Magnitude of synthetic violation (default 0.05).')
    parser.add_argument('--q-list', type=int, nargs='+',
                        default=[61, 151, 211, 307],
                        help='List of primes to test.')
    parser.add_argument('--n-permutations', type=int, default=500,
                        help='Number of permutations for the null (default 500).')
    parser.add_argument('--depth-stride', type=int, default=1,
                        help='Stride over depths (1 = every depth step).')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs (-1 = all cores).')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: depth_stride=5, n_permutations=100.')
    args = parser.parse_args()

    if args.quick:
        args.depth_stride = 5
        args.n_permutations = 100

    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        mode = f'violation (strength={args.violation_strength})' if args.violation else 'Markov'
        print(f'[INFO] generating synthetic data ({mode})')
        raw = make_synthetic(args.q_list, violation=args.violation,
                             violation_strength=args.violation_strength)
        data = {q: (c, sigma, q) for q, (c, sigma) in raw.items()}
    elif args.data_dir is not None:
        data = {}
        for q in args.q_list:
            for candidate in [args.data_dir / f'q_{q}.npz', args.data_dir / f'q{q}.npz']:
                if candidate.exists():
                    c, sigma, q_loaded = load_npz_prime(candidate)
                    data[q] = (c, sigma, q_loaded)
                    break
            else:
                print(f'[WARN] q_{q}.npz not found in {args.data_dir}, skipping')
        if not data:
            parser.error('No data loaded.')
    else:
        parser.error('Either --data-dir or --synthetic must be specified.')

    all_results = []
    for q, (c, sigma, q_real) in data.items():
        print(f'[INFO] q={q_real}: {len(c)} samples, {sigma.shape[1]} depths, '
              f'unique c = {len(np.unique(c))}, '
              f'min count/c = {int(np.unique(c, return_counts=True)[1].min())}, '
              f'triplet={"yes" if q_real % 3 == 1 else "no"}')
        df_q = run_q(q_real, c, sigma,
                     depth_stride=args.depth_stride,
                     n_permutations=args.n_permutations,
                     n_jobs=args.n_jobs)
        all_results.append(df_q)

    df = pd.concat(all_results, ignore_index=True)
    csv_path = args.out / 'hsuff_per_depth.csv'
    df.to_csv(csv_path, index=False)
    print(f'[INFO] wrote {csv_path}')

    plot_path = args.out / 'hsuff_vs_depth.pdf'
    plot_results(df, plot_path)
    print(f'[INFO] wrote {plot_path}')

    summary = make_summary(df)
    summary_path = args.out / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'[INFO] wrote {summary_path}')
    print()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()