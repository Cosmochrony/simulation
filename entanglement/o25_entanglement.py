"""
o25_entanglement.py
===================
Stage-1 post-processor for the conjugate-pair entanglement note.

Reconstructs the DOMINANT entanglement observables from an existing
q<q>_o25.npz produced by o25_paired_pipeline.py, WITHOUT re-running the
campaign.  Everything here uses only quantities already stored in the npz:

    sigma_c_mean, sigma_qmc_mean : (n_pairs, n_shells)  M-averaged sigma_c(n)
    sigma_pair_mean              : (n_pairs, n_shells)  M-averaged sigma_pair(n)
    shell_sizes                  : (n_shells,)          |S_n|, deterministic
    ns, n0, n1                   : shell indices and fit window
    delta_pair_mean              : canonical exponent (mean over pairs)

Derived observables (per pair, then aggregated):

    delta_r_mean(n) = sigma_c_mean(n) * |S_n|           mean new GS directions
    R(n)            = cumsum(delta_r_mean)              cumulative span (grows)
    final_rank      = R(last)                           saturated span
    r_pair(n)       = min over the two factors of (final_rank - R(n)), >= 1
                                                        RESIDUAL rank (decreases)
    log r_pair(n)   ~ dominant term of S_ent(n)
    Delta_I(n)      = sigma_pair(n) - sigma_pair(n+1)   stabilization rate

This is the rank-dominated trajectory.  The admissibility-weight deficit
eps_adm(n) is NOT computed here: it requires the direction-level residual
capture (stage 2, engine extension), since the stored adm_w_* are per-generator
(4 scalars/pair) and cannot populate a Schmidt spectrum on the residual support.

NOTE ON STATUS
    r_pair here is the M-averaged (float) residual rank.  When eps_adm is later
    folded in, log r_pair and eps_adm must come from the SAME support: either
    compute both per-block then average, or report eps_adm as a first-block
    estimate.  Do not subtract a single-block eps_adm from this mean-float rank
    silently.

Usage
    python o25_entanglement.py --file q61_o25.npz
    python o25_entanglement.py --dir o25_outputs --q-list 61 151 211 307
    python o25_entanglement.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


EPS_RANK = 1.0  # floor for the residual rank (rank 1 = separable, S_ent = 0)


def per_factor_residual_rank(sigma_factor_mean_row: np.ndarray,
                             shell_sizes: np.ndarray) -> tuple[np.ndarray, float]:
    """Residual GS rank for one factor (c or q-c) of a conjugate pair.

    sigma_factor_mean_row : (n_shells,)  M-averaged sigma_c(n) = mean(delta_r_n)/|S_n|
    shell_sizes           : (n_shells,)  |S_n|

    Returns (r_res, final_rank) where r_res(n) = final_rank - cumsum(delta_r_mean),
    floored at EPS_RANK.  r_res is non-increasing because delta_r_mean >= 0.
    """
    delta_r_mean = np.asarray(sigma_factor_mean_row, float) * np.asarray(shell_sizes, float)
    rank_cum = np.cumsum(delta_r_mean)
    final_rank = float(rank_cum[-1])
    r_res = np.maximum(final_rank - rank_cum, EPS_RANK)
    return r_res, final_rank


def pair_residual_rank(sigma_c_row: np.ndarray,
                       sigma_qmc_row: np.ndarray,
                       shell_sizes: np.ndarray) -> np.ndarray:
    """Pair residual rank = elementwise min of the two factor residual ranks.

    The Schmidt rank of a bipartite state is bounded by the smaller factor
    dimension, so min is the correct pair-level rank.
    """
    r_c, _ = per_factor_residual_rank(sigma_c_row, shell_sizes)
    r_qmc, _ = per_factor_residual_rank(sigma_qmc_row, shell_sizes)
    return np.minimum(r_c, r_qmc)


def fit_loglog_exponent(values: np.ndarray, ns: np.ndarray,
                        n0: int, n1: int, shift: int = 1) -> tuple[float, float]:
    """OLS fit  log(values) = -exponent * log(n + shift) + const  over [n0, n1].

    shift = 1 reproduces the O16 log(n+1) convention used for delta_pair.
    Returns (exponent, r2).  Returns (nan, nan) if the window is degenerate.
    """
    ns = np.asarray(ns, float)
    values = np.asarray(values, float)
    mask = (ns >= n0) & (ns <= n1) & (values > 0)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    x = np.log(ns[mask] + shift)
    y = np.log(values[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return -float(slope), r2


def dominant_observables(npz: dict) -> dict:
    """Compute per-pair and pair-averaged dominant entanglement observables.

    npz is a mapping with keys sigma_c_mean, sigma_qmc_mean, sigma_pair_mean,
    shell_sizes, ns, n0, n1 (and optionally delta_pair_mean for comparison).
    """
    sigma_c = np.atleast_2d(np.asarray(npz['sigma_c_mean'], float))
    sigma_qmc = np.atleast_2d(np.asarray(npz['sigma_qmc_mean'], float))
    sigma_pair = np.atleast_2d(np.asarray(npz['sigma_pair_mean'], float))
    shell_sizes = np.asarray(npz['shell_sizes'], float)
    ns = np.asarray(npz['ns'], float)
    n0 = int(npz['n0'])
    n1 = int(npz['n1'])

    n_pairs = sigma_c.shape[0]
    r_pair = np.empty_like(sigma_c)
    delta_I = np.empty((n_pairs, sigma_pair.shape[1] - 1))
    exp_rank = np.full(n_pairs, np.nan)
    r2_rank = np.full(n_pairs, np.nan)
    exp_sig = np.full(n_pairs, np.nan)
    r2_sig = np.full(n_pairs, np.nan)

    for i in range(n_pairs):
        r_pair[i] = pair_residual_rank(sigma_c[i], sigma_qmc[i], shell_sizes)
        delta_I[i] = sigma_pair[i, :-1] - sigma_pair[i, 1:]
        exp_rank[i], r2_rank[i] = fit_loglog_exponent(r_pair[i], ns, n0, n1)
        exp_sig[i], r2_sig[i] = fit_loglog_exponent(sigma_pair[i], ns, n0, n1)

    log_r_pair = np.log(r_pair)

    return {
        'ns': ns,
        'n0': n0, 'n1': n1,
        'r_pair': r_pair,                       # (n_pairs, n_shells)
        'log_r_pair': log_r_pair,
        'delta_I': delta_I,
        'r_pair_mean': r_pair.mean(axis=0),     # pair-averaged trajectory
        'log_r_pair_mean': log_r_pair.mean(axis=0),
        'delta_I_mean': delta_I.mean(axis=0),
        'exponent_rank_per_pair': exp_rank,     # contraction exponent of r_pair
        'r2_rank_per_pair': r2_rank,
        'exponent_rank_mean': float(np.nanmean(exp_rank)),
        'exponent_sigma_per_pair': exp_sig,     # exponent of sigma_pair (= delta_pair)
        'exponent_sigma_mean': float(np.nanmean(exp_sig)),
        'delta_pair_stored': (float(npz['delta_pair_mean'])
                              if 'delta_pair_mean' in npz else float('nan')),
    }


def summarize(obs: dict, q: int | str = '?') -> None:
    print(f'  q={q}   window=[{obs["n0"]},{obs["n1"]}]')
    print(f'    r_pair(n0)   = {obs["r_pair_mean"][obs["n0"]]:.2f}   '
          f'(early, ~max)')
    print(f'    r_pair(n1)   = {obs["r_pair_mean"][obs["n1"]]:.2f}   '
          f'(later, contracted)')
    monotone = np.all(np.diff(obs['r_pair_mean']) <= 1e-9)
    print(f'    r_pair monotone non-increasing: {monotone}')
    print(f'    Delta_I >= 0 everywhere:        '
          f'{bool(np.all(obs["delta_I_mean"] >= -1e-12))}')
    print(f'    contraction exponent of r_pair: '
          f'{obs["exponent_rank_mean"]:.3f}   (mean over pairs)')
    print(f'    exponent of sigma_pair (refit): '
          f'{obs["exponent_sigma_mean"]:.3f}')
    print(f'    delta_pair stored in npz:       '
          f'{obs["delta_pair_stored"]:.3f}')
    print()


def load_o25_npz(path: pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=True)
    keys = ('sigma_c_mean', 'sigma_qmc_mean', 'sigma_pair_mean',
            'shell_sizes', 'ns', 'n0', 'n1')
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise KeyError(f'{path.name} missing keys: {missing}')
    out = {k: z[k] for k in keys}
    if 'delta_pair_mean' in z.files:
        out['delta_pair_mean'] = z['delta_pair_mean']
    if 'q' in z.files:
        out['q'] = int(z['q'])
    return out


def self_test() -> None:
    """Synthetic check: build a plausible saturating cascade and verify signs."""
    rng = np.random.default_rng(0)
    n_shells = 16
    ns = np.arange(n_shells)
    # Shells grow then plateau (typical BFS on a Cayley graph near bfs_frac).
    shell_sizes = np.array([1, 4, 16, 64, 256, 1024, 4096, 9000,
                            9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000],
                           dtype=float)
    # Build the cascade from delta_r (new GS directions per shell): a positive
    # decreasing profile that sums to ~final_rank ~ q, spread over the window.
    # Then sigma_c(n) = delta_r(n) / |S_n| follows by construction.
    n_pairs = 5
    sigma_c = np.zeros((n_pairs, n_shells))
    sigma_qmc = np.zeros((n_pairs, n_shells))
    for i in range(n_pairs):
        expo = 1.3 + 0.15 * rng.standard_normal()
        amp = 14.0 + 2.0 * rng.standard_normal()
        delta_r_c = np.clip(amp * (ns + 1.0) ** (-expo), 0, None)
        delta_r_qmc = np.clip((amp * 1.05) * (ns + 1.0) ** (-expo), 0, None)
        sigma_c[i] = delta_r_c / shell_sizes
        sigma_qmc[i] = delta_r_qmc / shell_sizes
    sigma_pair = sigma_c * sigma_qmc

    npz = {
        'sigma_c_mean': sigma_c, 'sigma_qmc_mean': sigma_qmc,
        'sigma_pair_mean': sigma_pair, 'shell_sizes': shell_sizes,
        'ns': ns, 'n0': 2, 'n1': 7, 'delta_pair_mean': np.float64(9.9),
    }
    obs = dominant_observables(npz)

    assert np.all(np.diff(obs['r_pair_mean']) <= 1e-9), 'r_pair must be non-increasing'
    assert obs['r_pair_mean'][2] > obs['r_pair_mean'][7], 'early rank must exceed later'
    assert np.all(obs['delta_I_mean'] >= -1e-12), 'Delta_I must be >= 0 (sigma_pair decays)'
    assert np.isfinite(obs['exponent_rank_mean']), 'rank exponent must be finite'

    print('  self-test PASSED')
    print()
    summarize(obs, q='synthetic')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', type=pathlib.Path, default=None)
    p.add_argument('--dir', type=pathlib.Path, default=None)
    p.add_argument('--q-list', type=int, nargs='+', default=None)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if args.file:
        paths = [args.file]
    elif args.dir and args.q_list:
        paths = [args.dir / f'q{q}_o25.npz' for q in args.q_list]
    elif args.dir:
        paths = sorted(args.dir.glob('q*_o25.npz'))
    else:
        p.error('Provide --file, or --dir (optionally with --q-list), or --self-test.')

    for path in paths:
        if not path.exists():
            print(f'[WARN] {path} not found, skipping')
            continue
        npz = load_o25_npz(path)
        obs = dominant_observables(npz)
        summarize(obs, q=npz.get('q', '?'))


if __name__ == '__main__':
    main()