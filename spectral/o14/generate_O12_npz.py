"""
generate_O12_npz.py
===================
Generates .npz checkpoint files from the O12 pipeline for use by
SpectralO14_simulation.py (MODE B / pipeline validation).

REQUIRES: spectral_O12.py in the same directory (or on PYTHONPATH).

USAGE
-----
# All O13 primes, paper-compatible parameters:
python generate_O12_npz.py

# Only small primes (fast, ~5 min total):
python generate_O12_npz.py --primes 29 61

# One prime with custom parameters:
python generate_O12_npz.py --primes 101 --m-block 20 --n-max 20

# Output directory:
python generate_O12_npz.py --out-dir /path/to/npz_outputs

OUTPUT FILES
------------
One file per prime: <out_dir>/q<q>_o12.npz

Each .npz contains the following arrays (all are numpy arrays):

  q             : scalar int     — prime
  seed          : scalar int     — RNG seed used
  m_block       : scalar int     — number of blocks
  ns            : (n_shells,)    — BFS depth indices 0, 1, 2, ...
  shell_sizes   : (n_shells,)    — |S_n| per shell
  all_sigma     : (m_block, n_shells)  — Sigma_n^(c) per block and shell
  sigma_bar     : (n_shells,)    — mean over blocks
  sigma_var     : (n_shells,)    — variance over blocks
  V_n           : (n_shells,)    — normalised inter-block variance
  ell_gam       : (n_shells,)    — coherence length ell_gamma(n)
  n0            : scalar int     — fitting window lower bound
  n1            : scalar int     — fitting window upper bound
  delta_hat     : scalar float   — extracted delta_exact
  C_hat         : scalar float   — amplitude coefficient
  r2            : scalar float   — R^2 of OLS fit
  v_max_win     : scalar float   — max V_n in fitting window
  e1_ok         : scalar bool
  e2_ok         : scalar bool
  e3_ok         : scalar bool
  blocks        : (m_block, 3)   — block indices (c1, c2, c3)

WHAT SpectralO14_simulation.py uses from each file
---------------------------------------------------
  delta_hat     -> hat_delta_exact (replaces HAT_DELTA_EXACT table)
  ell_gam       -> C(q) = mean(ell_gam[n0:n1+1])
                   Var(theta)(q) = -2 * log(C(q))
  V_n           -> epsilon(q) correction via v_max_win
  n0, n1        -> actual fitting window for norm_corr = eta*log(q)/log(n1)
  sigma_bar     -> sanity check on delta_hat
"""

import argparse
import pathlib
import sys
import time

import numpy as np

try:
    from spectral_O12 import run_one_prime, coherence_length
except ImportError:
    print("ERROR: spectral_O12.py not found. Place it in the same directory.")
    sys.exit(1)


# O13-compatible parameters (from o13_benchmark_report.txt)
# These reproduce the delta_hat values used throughout O13/O14.
O13_PARAMS = {
    29:  dict(m_block=50,  n_max=8,   bfs_frac=0.99,  seed=0),
    61:  dict(m_block=15,  n_max=10,  bfs_frac=0.99,  seed=0),
    101: dict(m_block=20,  n_max=20,  bfs_frac=0.99,  seed=0),
    151: dict(m_block=10,  n_max=37,  bfs_frac=0.29,  seed=0),
    211: dict(m_block=10,  n_max=50,  bfs_frac=0.106, seed=0),
}

# Quick parameters for testing (reduced m_block, runs in ~2 min total for q<=61)
QUICK_PARAMS = {
    29:  dict(m_block=20,  n_max=8,   bfs_frac=0.99,  seed=0),
    61:  dict(m_block=8,   n_max=10,  bfs_frac=0.99,  seed=0),
    101: dict(m_block=5,   n_max=15,  bfs_frac=0.99,  seed=0),
}


def save_npz(res, out_path):
    """
    Save the output of run_one_prime() as a .npz file.
    Excludes non-array fields (shells list, bool flags stored as int).
    """
    np.savez(
        out_path,
        q           = np.int64(res["q"]),
        seed        = np.int64(res["seed"]),
        m_block     = np.int64(res["m_block"]),
        ns          = res["ns"].astype(np.int64),
        shell_sizes = res["shell_sizes"].astype(np.int64),
        all_sigma   = res["all_sigma"].astype(np.float64),
        sigma_bar   = res["sigma_bar"].astype(np.float64),
        sigma_var   = res["sigma_var"].astype(np.float64),
        V_n         = res["V_n"].astype(np.float64),
        ell_gam     = res["ell_gam"].astype(np.float64),
        n0          = np.int64(res["n0"]),
        n1          = np.int64(res["n1"]),
        delta_hat   = np.float64(res["delta_hat"]),
        C_hat       = np.float64(res["C_hat"]),
        r2          = np.float64(res["r2"]),
        v_max_win   = np.float64(res["v_max_win"]),
        e1_ok       = np.int8(res["e1_ok"]),
        e2_ok       = np.int8(res["e2_ok"]),
        e3_ok       = np.int8(res["e3_ok"]),
        blocks      = res["blocks"].astype(np.int64),
    )
    print(f"  Saved: {out_path}")


def verify_npz(path):
    """
    Quick sanity check: load the .npz and print key fields.
    """
    z = np.load(path)
    q         = int(z["q"])
    m_block   = int(z["m_block"])
    n0        = int(z["n0"])
    n1        = int(z["n1"])
    delta_hat = float(z["delta_hat"])
    r2        = float(z["r2"])
    v_max     = float(z["v_max_win"])
    ell_win   = z["ell_gam"][n0:n1 + 1]
    C_q       = float(np.mean(ell_win))
    C_q       = max(C_q, 1e-12)
    var_theta = -2.0 * np.log(C_q)
    print(
        f"  [verify] q={q}  m_block={m_block}  window=[{n0},{n1}]"
        f"  delta_hat={delta_hat:.4f}  R2={r2:.4f}  Vmax={v_max:.4f}"
        f"  C(q)={C_q:.4f}  Var(theta)={var_theta:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate O12 .npz checkpoint files for O14 pipeline validation"
    )
    parser.add_argument(
        "--primes", type=int, nargs="+",
        default=list(O13_PARAMS.keys()),
        help="Primes to compute (default: 29 61 101 151 211)"
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path("o14_pipeline"),
        help="Output directory for .npz files (default: ./o14_pipeline/)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced parameters for fast testing (q<=101 only)"
    )
    parser.add_argument(
        "--m-block", type=int, default=None,
        help="Override m_block for all primes"
    )
    parser.add_argument(
        "--n-max", type=int, default=None,
        help="Override n_max for all primes"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (default: 0)"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params_table = QUICK_PARAMS if args.quick else O13_PARAMS

    requested = sorted(set(args.primes))
    print("generate_O12_npz.py")
    print("===================")
    print(f"Primes     : {requested}")
    print(f"Mode       : {'QUICK' if args.quick else 'O13-compatible'}")
    print(f"Output dir : {args.out_dir}")
    print()

    total_t0 = time.perf_counter()

    for q in requested:
        if q not in params_table:
            print(
                f"WARNING: q={q} not in params table. "
                f"Using q=211 defaults (m_block=10, n_max=50, bfs_frac=0.106)."
            )
            p = dict(m_block=10, n_max=50,
                     bfs_frac=max(0.10, 1e6 / q**3), seed=args.seed)
        else:
            p = dict(params_table[q])
            p["seed"] = args.seed

        if args.m_block is not None:
            p["m_block"] = args.m_block
        if args.n_max is not None:
            p["n_max"] = args.n_max

        out_path = args.out_dir / f"q{q}_o12.npz"

        if out_path.exists():
            print(f"q={q}: {out_path} already exists — skipping.")
            print(f"  (delete it to recompute)")
            verify_npz(out_path)
            continue

        print(
            f"q={q}: m_block={p['m_block']}, n_max={p['n_max']}, "
            f"bfs_frac={p['bfs_frac']}, seed={p['seed']}"
        )
        t0 = time.perf_counter()

        res = run_one_prime(
            q        = q,
            m_block  = p["m_block"],
            n_max    = p["n_max"],
            bfs_frac = p["bfs_frac"],
            seed     = p["seed"],
        )

        dt = time.perf_counter() - t0
        print(f"  Computation: {dt:.1f}s")

        save_npz(res, out_path)
        verify_npz(out_path)
        print()

    total_dt = time.perf_counter() - total_t0
    print(f"Total time: {total_dt:.1f}s")
    print()
    print("Next step:")
    print(f"  Set PIPELINE_DATA_PATH = '{args.out_dir}' in SpectralO14_simulation.py")
    print(f"  Set USE_PIPELINE = True")
    print(f"  Run: python SpectralO14_simulation.py")


if __name__ == "__main__":
    main()
