"""
o32_triplet_pipeline.py  —  O32 Spectral Admissibility Sub-programme
Numerical test of Hypothesis [H-color]: triplet co-admissibility on Heis_3(Z/qZ).

GROUNDING IN THE O-SERIES
--------------------------
Direct triplet analogue of o25_paired_pipeline.py.
Observable: sigma_c(n) = delta_r_n / |S_n| via compute_block_capacity() in spectral_O12.py.
Block: (c, c2_rand, c3_rand) with c+c2+c3 ≢ 0 (mod q)  — same as O25.
Triplet observable: sigma_triplet(n) = sigma_c1(n) * sigma_c2(n) * sigma_c3(n).
Fitting: log(sigma) = -delta * log(n+1) + const  (O25/O16 convention, +1 shift).
Window: WINDOW_O12 table from O25, or auto-calibrated with --auto-window.

Protocol (O31 §8):
  Test primes    q ≡ 1 (mod 3): {61, 151, 211, 307}
  Control primes q ≡ 2 (mod 3): {29, 101}

Falsification (O31 §8.2):
  [H-color] PASSES  Var_i(sigma_ci(n)) ≈ 0 in pre-sat window (test primes)
  [H-color] FAILS   Var_i(sigma_ci(n)) = O(1) comparable to controls

REQUIRES:  spectral_O12.py in the same directory (or on PYTHONPATH).

USAGE:
  python o32_triplet_pipeline.py                     # defaults
  python o32_triplet_pipeline.py --primes 61 151 29
  python o32_triplet_pipeline.py --primes 61 --M 50 --auto-window
  python o32_triplet_pipeline.py --quick             # smoke test
"""

import argparse, os, sys, time
import numpy as np
from joblib import Parallel, delayed

for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from spectral_O12 import (build_generators, bfs_shells,
                               compute_block_capacity, find_fitting_window)
except Exception as exc:
    print(f"ERROR importing spectral_O12: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parameters  (mirror O25)
# ---------------------------------------------------------------------------
DEFAULT_PRIMES = [61, 151, 29, 101]
M_DEFAULT      = 20          # blocks per character; use 50 for paper runs
N_TRIP_DEFAULT = 4           # triplets (or pairs) per prime
DEFAULT_SEED   = 42
EPS            = 1e-15

WINDOW_O12 = {29:(2,5), 61:(2,7), 101:(3,10), 151:(3,12), 211:(3,13)}
BFS_FRAC   = {29:0.99, 61:0.99, 101:0.99, 151:0.29, 211:0.11, 307:0.08}
N_MAX_BLK  = {29:8, 61:10, 101:20, 151:37, 211:50, 307:70}

# ---------------------------------------------------------------------------
# 1.  Colour-triplet arithmetic
# ---------------------------------------------------------------------------

def primitive_cube_root(q):
    """Primitive cube root of unity omega in (Z/qZ)* for q ≡ 1 mod 3, else None."""
    if q % 3 != 1:
        return None
    for x in range(2, q):
        if pow(x, 3, q) == 1:
            return x
    return None


def find_colour_triplets(q, omega, n_max):
    """
    Return up to n_max valid colour triplets {c, omega*c, omega^2*c} in (Z/qZ)*.
    Validity (O31 Def. 4.1): all nonzero, sum ≡ 0 (automatic), pairwise non-conjugate.
    """
    w2 = (omega * omega) % q
    seen, out = set(), []
    for c in range(1, q):
        t = (c, (omega*c) % q, (w2*c) % q)
        key = frozenset(t)
        if key in seen or len(key) < 3:
            continue
        seen.add(key)
        elems = list(t)
        conjs = [(q-x) % q for x in elems]
        if any(elems[i] == conjs[j] for i in range(3) for j in range(3) if i != j):
            continue
        out.append(t)
        if len(out) >= n_max:
            break
    return out


def find_conjugate_pairs(q, n_max):
    """Standard conjugate pairs {c, q-c} for control primes."""
    seen, out = set(), []
    for c in range(1, q//2 + 1):
        conj = (q-c) % q
        key  = frozenset([c, conj])
        if conj != c and key not in seen:
            seen.add(key)
            out.append((c, conj))
        if len(out) >= n_max:
            break
    return out

# ---------------------------------------------------------------------------
# 2.  Block sampling  (same as O25 sample_block_with_c1)
# ---------------------------------------------------------------------------

def sample_block(c1, q, rng, max_attempts=2000):
    for _ in range(max_attempts):
        c2 = int(rng.integers(1, q))
        c3 = int(rng.integers(1, q))
        if (c1 + c2 + c3) % q != 0:
            return np.array([c1, c2, c3], dtype=np.int64)
    raise RuntimeError(f"Cannot sample block c1={c1}, q={q}")

# ---------------------------------------------------------------------------
# 3.  Fitting  (O25/O16 convention: log(n+1))
# ---------------------------------------------------------------------------

def fit_delta(sigma, ns, n0, n1):
    n_arr = ns[n0:n1+1].astype(float)
    s_arr = sigma[n0:min(n1+1, len(sigma))]
    if len(s_arr) < len(n_arr):
        s_arr = np.concatenate([s_arr, np.zeros(len(n_arr)-len(s_arr))])
    mask = (n_arr > 0) & (s_arr > EPS)
    if mask.sum() < 2:
        return np.nan, np.nan
    log_n = np.log(n_arr[mask] + 1)
    log_s = np.log(s_arr[mask])
    coef  = np.polyfit(log_n, log_s, 1)
    resid = log_s - np.polyval(coef, log_n)
    ss_tot = np.sum((log_s - log_s.mean())**2)
    r2 = 1.0 - np.sum(resid**2)/ss_tot if ss_tot > 1e-15 else np.nan
    return float(-coef[0]), float(r2)

# ---------------------------------------------------------------------------
# 4.  Per-character worker  (parallel)
# ---------------------------------------------------------------------------

def _worker_character(c, q, shells, gens, n_max, n0, n1, M, base_seed, cidx):
    """Compute M sigma_c(n) samples for character c."""
    rng = np.random.default_rng(base_seed + cidx*997 + c*7)
    ns  = np.arange(len(shells), dtype=np.int64)
    sigma_list, delta_arr, r2_arr = [], np.full(M, np.nan), np.full(M, np.nan)
    for m in range(M):
        cb = sample_block(c, q, rng)
        sv, _, _, _ = compute_block_capacity(shells, cb, q, gens, n_max=n_max)
        sigma_list.append(sv)
        d, r2 = fit_delta(sv, ns, n0, min(n1, len(sv)-1))
        delta_arr[m], r2_arr[m] = d, r2
    # Pad short profiles with zeros (sigma=0 after saturation is physically correct).
    # Using min-length would let a single outlier block corrupt the mean for the
    # entire character; zero-padding retains the full pre-saturation window.
    n_max_len = max(len(s) for s in sigma_list)
    padded = []
    for s in sigma_list:
        if len(s) < n_max_len:
            s = np.concatenate([s, np.zeros(n_max_len - len(s))])
        padded.append(s)
    stack = np.stack(padded)
    return c, stack.mean(0), stack.std(0), delta_arr, r2_arr

# ---------------------------------------------------------------------------
# 5.  Triplet observables
# ---------------------------------------------------------------------------

def triplet_obs(sm, triplet, ns, n0, n1):
    """sigma_triplet, variance profile, var_ratio, delta_triplet."""
    s = [sm[c] for c in triplet]
    n_c = min(len(x) for x in s)
    stk = np.stack([x[:n_c] for x in s])
    var = np.var(stk, axis=0)
    sq_mean = np.mean(stk**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        vr = np.where(sq_mean > EPS, var / sq_mean, np.nan)
    st = np.prod(stk, axis=0)
    # fit only if window lies within available shells
    fit_n1 = min(n1, n_c - 1)
    dt, r2t = (fit_delta(st, ns[:n_c], n0, fit_n1)
               if fit_n1 >= n0 else (np.nan, np.nan))
    return dict(triplet=triplet, sigma_trip=st, var=var, vr=vr,
                delta=dt, r2=r2t, n=n_c)


def colour_cov(sm, triplet, n0, n1):
    """
    3x3 covariance of (sigma_c1, sigma_c2, sigma_c3) over window [n0, n1].
    Returns NaN matrix if window has fewer than 2 points.
    """
    n_c   = min(len(sm[c]) for c in triplet)
    n_win = min(n1 + 1, n_c)
    if n_win - n0 < 2:
        return np.full((3, 3), np.nan)
    mat = np.stack([sm[c][n0:n_win] for c in triplet])
    C   = np.cov(mat)
    return C if not np.any(np.isnan(C)) else np.full((3, 3), np.nan)

# ---------------------------------------------------------------------------
# 6.  Main pipeline for one prime
# ---------------------------------------------------------------------------

def run_prime(q, M, bfs_frac, n_max, n0, n1, auto_window,
              n_triplets, n_jobs, seed, verbose, filter_anomalous=False):
    q_class = q % 3
    omega   = primitive_cube_root(q)
    is_test = (q_class == 1 and omega is not None)

    if verbose:
        label = "TEST" if is_test else "CONTROL"
        print(f"\n{'='*58}")
        print(f"q={q}  [{q}≡{q_class} mod 3, {label}]  omega={omega}")
        print(f"{'='*58}")

    groups = (find_colour_triplets(q, omega, n_triplets) if is_test
              else find_conjugate_pairs(q, n_triplets))
    if not groups:
        print(f"  WARNING: no valid groups for q={q}")
        return None
    characters = list({c for g in groups for c in g})
    if verbose:
        print(f"Groups     : {groups}")
        print(f"Characters : {sorted(characters)}")

    # BFS
    if verbose:
        print(f"BFS (frac={bfs_frac}, n_max={n_max})...")
    t0 = time.perf_counter()
    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    ns     = np.arange(len(shells), dtype=np.int64)
    sizes  = np.array([len(s) for s in shells], dtype=np.int64)
    if verbose:
        print(f"  {len(shells)} shells, {sizes.sum()} nodes — {time.perf_counter()-t0:.1f}s")

    # auto-window calibration
    if auto_window:
        prng = np.random.default_rng(seed + 999999)
        probe = []
        for _ in range(5):
            cp = int(prng.integers(1, q))
            cb = sample_block(cp, q, prng)
            sv, _, _, _ = compute_block_capacity(shells, cb, q, gens, n_max=n_max)
            pad = len(shells) - len(sv)
            probe.append(np.concatenate([sv, np.zeros(pad)]) if pad else sv)
        sb = np.mean(probe, axis=0)
        n0, n1 = find_fitting_window(ns[1:], sb[1:], q)
        n0, n1 = max(n0, 1), min(n1, len(shells)-1)
        if verbose:
            print(f"  Auto-window: [{n0}, {n1}]")
    elif verbose:
        print(f"  Window: [{n0}, {n1}] (WINDOW_O12 table)")

    # parallel character computation
    if verbose:
        print(f"Computing {len(characters)} chars × M={M} blocks ({n_jobs} jobs)...")
    results_c = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_worker_character)(c, q, shells, gens, n_max, n0, n1, M, seed, i)
        for i, c in enumerate(characters)
    )
    sm, ss, di = {}, {}, {}
    for c, mean, std, darr, r2arr in results_c:
        sm[c], ss[c] = mean, std
        di[c] = (float(np.nanmean(darr)), float(np.nanstd(darr)),
                 float(np.nanmean(r2arr)))
        if verbose:
            d, sd, r2 = di[c]
            print(f"  c={c}: delta={d:.3f}±{sd:.3f}  R2={r2:.3f}")

    result = dict(q=q, q_class=q_class, omega=omega, is_test=is_test,
                  groups=groups, characters=characters,
                  sigma_means=sm, sigma_stds=ss, delta_ind=di,
                  ns=ns, shell_sizes=sizes, n0=n0, n1=n1)

    if is_test:
        obs_list, C_list = [], []
        for trip in groups:
            obs = triplet_obs(sm, trip, ns, n0, n1)
            obs_list.append(obs)
            C_list.append(colour_cov(sm, trip, n0, n1))
            if verbose:
                win = obs['vr'][n0:max(n0+1, obs['n']//2)]
                vr_win = np.nanmean(win)
                status = f"var_ratio={vr_win:.3e}"
                print(f"  trip {trip}: delta_tri={obs['delta']:.3f} "
                      f"R2={obs['r2']:.3f}  var_ratio={vr_win:.3e} {status}")
                C = C_list[-1]
                if not np.any(np.isnan(C)):
                    try:
                        ev = np.linalg.eigvalsh(C)
                        print(f"    C_color eig = {np.round(ev,6)}")
                    except np.linalg.LinAlgError:
                        print(f"    C_color eig = [eigvalsh did not converge]")
                else:
                    print(f"    C_color = NaN (window too short)")
        n_min_all = min(obs['n'] for obs in obs_list)
        # pad shorter vr arrays with NaN before averaging
        vr_padded = []
        for obs in obs_list:
            v = obs['vr']
            if len(v) < n_min_all:
                v = np.concatenate([v, np.full(n_min_all - len(v), np.nan)])
            vr_padded.append(v[:n_min_all])
        vr_mean = np.nanmean(vr_padded, axis=0)
        result.update(obs_list=obs_list, C_list=C_list, vr_mean=vr_mean)

        if verbose:
            dc_vals = [di[c][0] for c in characters if not np.isnan(di[c][0])]
            dt_vals = [o['delta'] for o in obs_list if not np.isnan(o['delta'])]
            dc_m, dt_m = np.mean(dc_vals), np.mean(dt_vals)
            print(f"\n  delta_c (mean)    = {dc_m:.3f} ± {np.std(dc_vals):.3f}")
            print(f"  delta_triplet     = {dt_m:.3f} ± {np.std(dt_vals):.3f}")
            print(f"  3 × delta_c       = {3*dc_m:.3f}  "
                  f"(expected: delta_triplet ≈ 3 × delta_c)")
            win_vr = np.nanmean(vr_mean[n0:max(n0+1, n_min_all//2)])
            print(f"  var_ratio (window) = {win_vr:.4e}  "
                  f"(verdict requires control prime — see summary)")
    else:
        # --- optional anomaly filter (--filter-anomalous) ---
        filter_info = None
        pairs_used  = groups
        pairs_removed = []
        if filter_anomalous:
            pairs_used, pairs_removed, filter_info = filter_anomalous_pairs(
                groups, di)
            if verbose:
                print(f"  Anomaly filter (IQR×{filter_info['iqr_factor']:.1f}): "
                      f"delta_c ∈ [{filter_info['lo']:.3f}, {filter_info['hi']:.3f}]")
                print(f"  Pairs KEPT    ({len(pairs_used)}): {pairs_used}")
                print(f"  Pairs REMOVED ({len(pairs_removed)}): {pairs_removed}")
                if not pairs_removed:
                    print(f"  No anomalous pairs detected.")
        # --- compute variance over kept pairs ---
        pv_list, pv_list_all = [], []
        for c1, c2 in groups:
            n_c = min(len(sm[c1]), len(sm[c2]))
            stk = np.stack([sm[c1][:n_c], sm[c2][:n_c]])
            v   = np.var(stk, axis=0)
            pv_list_all.append(v)
            if (c1, c2) in pairs_used or (c2, c1) in pairs_used:
                pv_list.append(v)
        n_min_all = min(len(v) for v in pv_list_all)
        n_min     = min(len(v) for v in pv_list) if pv_list else n_min_all
        pv_mean_all = np.nanmean([v[:n_min_all] for v in pv_list_all], axis=0)
        pv_mean     = (np.nanmean([v[:n_min] for v in pv_list], axis=0)
                       if pv_list else pv_mean_all)
        result['pair_var_mean']     = pv_mean      # used for ctrl_ref
        result['pair_var_mean_all'] = pv_mean_all  # unfiltered (for paper)
        result['pairs_removed']     = pairs_removed
        result['filter_info']       = filter_info
        if verbose:
            _slc = pv_mean[n0:] if len(pv_mean) > n0 else pv_mean
            pv_win = np.nanmean(_slc) if len(_slc) > 0 else np.nan
            label = "(filtered)" if pairs_removed else "(all pairs)"
            print(f"  Pair var_ratio reference (raw) {label}: {pv_win:.4e}")
            if pairs_removed:
                _slc2 = pv_mean_all[n0:] if len(pv_mean_all) > n0 else pv_mean_all
                pv_win2 = np.nanmean(_slc2) if len(_slc2) > 0 else np.nan
                print(f"  Pair var_ratio reference (raw) (unfiltered): {pv_win2:.4e}")

    return result

# ---------------------------------------------------------------------------
# 7.  Summary + save
# ---------------------------------------------------------------------------

def print_summary(results):
    """
    Summary table with relative verdict: compare triplet var_ratio against
    the mean control pair_var_mean (conjugate pairs, known co-admissible).
    Criterion (O31 §8.2): [H-color] passes iff
        var_ratio(triplet) ≈ var_ratio(pair_ctrl)  [noise floor]
    Absolute threshold 1e-6 is inappropriate for small M; the ratio
        R = var_ratio(triplet) / var_ratio(ctrl_pair)
    is the correct observable.  R ≈ 1 → PASSES;  R >> 1 → FAILS.
    """
    # compute control reference (mean pair var_ratio over all control primes)
    ctrl_vr_vals = []
    for res in results:
        if res is None or res['is_test']:
            continue
        n0 = res['n0']
        pv = res['pair_var_mean']
        # normalise by mean sigma^2 over window to get dimensionless ratio
        chars = res['characters']
        sm_list = [res['sigma_means'][c] for c in chars]
        n_win = min(len(s) for s in sm_list)
        sq_mean = np.mean(np.stack([s[:n_win] for s in sm_list])**2, axis=0)
        n_end = max(n0 + 1, n_win // 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            pv_r = np.where(sq_mean[:len(pv)] > EPS,
                            pv[:n_win] / sq_mean[:len(pv)], np.nan)
        # use all available shells if window slice is empty
        slc = pv_r[n0:n_end] if (len(pv_r) > n0 and n_end > n0) else pv_r
        ctrl_vr_vals.append(np.nanmean(slc) if len(slc) > 0 else np.nan)
    ctrl_ref = np.nanmean(ctrl_vr_vals) if ctrl_vr_vals else np.nan

    print("\n" + "="*80)
    print("O32 SUMMARY — [H-color] test")
    if not np.isnan(ctrl_ref):
        print(f"Control pair var_ratio reference (noise floor): {ctrl_ref:.4e}")
        print("Verdict criterion: R = var_ratio(triplet)/ctrl_ref")
        print("  R ≤ 3  →  ✓ PASSES (triplet variance at noise floor)")
        print("  R ≤ 10 →  ~ MARGINAL  |  R > 10  →  ✗ FAILS")
    else:
        print("(No control prime available — run q≡2 mod 3 for calibrated verdict)")
    print("="*80)
    print(f"{'q':>5}  {'class':>9}  {'delta_c':>8}  {'delta_tri':>10}"
          f"  {'3×delta_c':>10}  {'var_ratio':>12}  {'R':>6}")
    print("-"*80)
    for res in results:
        if res is None:
            continue
        q = res['q']
        cls = f"≡{res['q_class']} mod 3"
        dc_vals = [res['delta_ind'][c][0] for c in res['characters']
                   if not np.isnan(res['delta_ind'][c][0])]
        dc = np.mean(dc_vals) if dc_vals else np.nan
        if res['is_test']:
            dt_vals = [o['delta'] for o in res['obs_list']
                       if not np.isnan(o['delta'])]
            dt  = np.mean(dt_vals) if dt_vals else np.nan
            vr  = np.nanmean(res['vr_mean'][
                res['n0']:max(res['n0']+1, len(res['vr_mean'])//2)])
            if not np.isnan(ctrl_ref) and ctrl_ref > EPS:
                R = vr / ctrl_ref
                tag = ("✓ PASSES" if R <= 3 else
                       "~ MARGINAL" if R <= 10 else "✗ FAILS")
                r_str = f"{R:>6.1f}"
            else:
                tag, r_str = "? (no ctrl)", "   n/a"
            print(f"{q:>5}  {cls:>9}  {dc:>8.3f}  {dt:>10.3f}"
                  f"  {3*dc:>10.3f}  {vr:>12.4e}  {r_str}  {tag}")
        else:
            _pv_arr = res['pair_var_mean']
            # normalise raw pair variance by mean sigma^2 (same as test metric)
            _sm_list = [res['sigma_means'][c] for c in res['characters']]
            _n_w = min(len(s) for s in _sm_list)
            _sq  = np.mean(np.stack([s[:_n_w] for s in _sm_list])**2, axis=0)
            _n0  = res['n0']
            _pv_full = _pv_arr[:_n_w]
            with np.errstate(divide='ignore', invalid='ignore'):
                _pvr = np.where(_sq[:len(_pv_full)] > EPS,
                                _pv_full / _sq[:len(_pv_full)], np.nan)
            _slc = _pvr[_n0:] if len(_pvr) > _n0 else _pvr
            pv = np.nanmean(_slc) if len(_slc) > 0 else np.nan
            print(f"{q:>5}  {cls:>9}  {dc:>8.3f}  {'n/a':>10}"
                  f"  {'n/a':>10}  {pv:>12.4e}  {'  ctrl':>6}  [reference]")
    print("="*80)


def save_results(results, out_path):
    d = {}
    for res in results:
        if res is None: continue
        p = f"q{res['q']}"
        d[f"{p}_class"] = np.int64(res['q_class'])
        d[f"{p}_ns"], d[f"{p}_shells"] = res['ns'], res['shell_sizes']
        d[f"{p}_n0"], d[f"{p}_n1"] = np.int64(res['n0']), np.int64(res['n1'])
        for c in res['characters']:
            d[f"{p}_sm_c{c}"] = res['sigma_means'][c]
            d[f"{p}_ss_c{c}"] = res['sigma_stds'][c]
        if res['is_test']:
            d[f"{p}_vr_mean"] = res['vr_mean']
            for k, (obs, C) in enumerate(zip(res['obs_list'], res['C_list'])):
                d[f"{p}_t{k}_delta"] = obs['delta']
                d[f"{p}_t{k}_vr"]    = obs['vr']
                d[f"{p}_t{k}_C"]     = C
        else:
            d[f"{p}_pair_var"]     = res['pair_var_mean']
            d[f"{p}_pair_var_all"] = res.get('pair_var_mean_all',
                                              res['pair_var_mean'])
            n_rm = len(res.get('pairs_removed', []))
            d[f"{p}_n_pairs_removed"] = np.int64(n_rm)
    np.savez(out_path, **d)
    print(f"Saved → {out_path}")

# ---------------------------------------------------------------------------
# 8.  CLI
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(description="O32 colour-triplet pipeline")
    pa.add_argument("--primes",      nargs="+", type=int, default=DEFAULT_PRIMES)
    pa.add_argument("--M",           type=int,  default=M_DEFAULT)
    pa.add_argument("--n-triplets",  type=int,  default=N_TRIP_DEFAULT)
    pa.add_argument("--bfs-frac",    type=float, default=None)
    pa.add_argument("--n-max",       type=int,  default=None)
    pa.add_argument("--auto-window", action="store_true")
    pa.add_argument("--n-jobs",      type=int,  default=-1)
    pa.add_argument("--seed",        type=int,  default=DEFAULT_SEED)
    pa.add_argument("--out",         type=str,  default="results_o32.npz")
    pa.add_argument("--filter-anomalous", action="store_true", dest="filter_anomalous",
                    help="Exclude outlier pairs from control reference "
                         "(delta_c outside median ± 2×IQR). "
                         "Both filtered and unfiltered results are reported.")
    pa.add_argument("--quick",       action="store_true",
                    help="Smoke test: q=61 only, M=5, 2 triplets")
    args = pa.parse_args()

    primes     = [args.primes[0]] if args.quick else args.primes
    M          = 5                if args.quick else args.M
    n_triplets = 2                if args.quick else args.n_triplets
    if args.quick:
        print(f"[QUICK] q={primes[0]}, M={M}, n_triplets={n_triplets}")

    results = []
    for q in primes:
        bfs_frac = args.bfs_frac or BFS_FRAC.get(q, 0.05)
        n_max    = args.n_max    or N_MAX_BLK.get(q, 60)
        if not args.auto_window and q in WINDOW_O12:
            n0, n1 = WINDOW_O12[q]
        else:
            n0, n1 = 2, 10
        results.append(run_prime(q=q, M=M, bfs_frac=bfs_frac, n_max=n_max,
                                 n0=n0, n1=n1, auto_window=args.auto_window,
                                 n_triplets=n_triplets, n_jobs=args.n_jobs,
                                 seed=args.seed, verbose=True,
                                 filter_anomalous=args.filter_anomalous))
    print_summary(results)
    save_results(results, args.out)

if __name__ == "__main__":
    main()