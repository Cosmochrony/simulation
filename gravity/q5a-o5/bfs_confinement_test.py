"""
BFS Spectral Confinement Test  --  Q5a-O2 paper, §4
=====================================================
Computes M_q(R) = ||P_{>R} Pi_q||_op^2 for the admissible projection Pi_q.

Confirmed structure of O25 npz files (from pipeline source + inspection):

  basis_c[p] : (q, q) unitary complex matrix.
               ROWS are the Gram-Schmidt orthonormal basis of C^q for pair p.
               The first rank_adm_p rows are the ADMISSIBLE vectors in C^q.

  pi_c[p, s] : (N_s, rank_adm_p) complex matrix.
               H_eff projection of N_s fingerprint vectors at BFS shell n0+s
               onto the rank_adm_p-dimensional admissible subspace.
               rank_adm_p = pi_c[p, 0].shape[1]  (e.g. 3 for q=29, pair 0).
               Empty (shape (0,)) for pairs with no admissible content.

  vecs_c[p, s] : (N_s,) float64 -- residual norms after GS projection.
                 NOT q-dimensional vectors.

  adm_w_c_* : (n_pairs,) float64 -- admissibility weights; NaN for empty pairs.

Strategy for M_q(R):
  For each non-empty pair p  (pi_c[p,0].shape = (N_s, rank_adm_p)):
    V_p = basis_c[p][:rank_adm_p, :]   # (rank_adm_p, q) -- admissible rows
  Stack all V_p -> V_orth (total_rank, q) [already orthonormal]
  M_q(R) = sigma_max( F_{>R}  V_orth^H )^2

Usage:
    python bfs_confinement_test.py --diagnose --npz-dir ./o25_outputs --primes 29
    python bfs_confinement_test.py --npz-dir ./o25_outputs --primes 29 61 101 151 --R-max 15
    python bfs_confinement_test.py --npz-dir ./o25_outputs --primes 29 61 101 151 \
        --R-values 1 2 3 5 8 12
"""

import argparse
import glob
import os
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_npz(prime, npz_dir="."):
    for pat in [
        os.path.join(npz_dir, f"q{prime}_o25.npz"),
        os.path.join(npz_dir, f"*q{prime}*.npz"),
        os.path.join(npz_dir, f"*{prime}*.npz"),
    ]:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]
    return None


def dft_matrix(q):
    k = np.arange(q)
    xi = np.arange(q)
    return np.exp(-2j * np.pi * np.outer(xi, k) / q) / np.sqrt(q)


def high_freq_mask(q, R):
    """True for |xi| > R (symmetric DFT convention)."""
    mask = np.ones(q, dtype=bool)
    for r in range(R + 1):
        mask[r] = False
    for r in range(1, R + 1):
        mask[q - r] = False
    return mask


# ---------------------------------------------------------------------------
# Admissible rank for a pair: read from pi_c[p, first_non_empty_shell]
# ---------------------------------------------------------------------------

def admissible_rank_from_pi(pi_c_row):
    """
    pi_c_row : 1-D object array of length n_window_shells.
    Returns (rank_adm, shell_idx) where pi_c_row[shell_idx] is the first
    non-empty shell, or (0, -1) if all shells are empty.
    """
    for s, raw in enumerate(pi_c_row):
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            return arr.shape[1], s
    return 0, -1


# ---------------------------------------------------------------------------
# Admissible basis extraction
# ---------------------------------------------------------------------------

def build_admissible_basis(data, q, verbose=True):
    """
    For each pair p with non-empty admissible content:
      rank_adm_p = pi_c[p, s_first].shape[1]
      V_p        = basis_c[p][:rank_adm_p, :]   # (rank_adm_p, q)
    Stack -> V_orth (total_rank, q).
    Already orthonormal (rows of unitary basis_c); re-orthogonalise globally
    in case different pairs share basis directions.
    """
    basis_c = data['basis_c']
    pi_c    = data['pi_c']
    n_pairs = basis_c.shape[0]

    vecs_list = []
    for p in range(n_pairs):
        rank_adm, s_first = admissible_rank_from_pi(pi_c[p])
        if rank_adm == 0:
            continue

        B = np.asarray(basis_c[p]).astype(complex)   # (q, q) unitary
        if B.shape != (q, q):
            if verbose:
                print(f"  pair {p}: unexpected basis_c shape {B.shape}, skipping")
            continue

        # Rows of B are orthonormal; first rank_adm are the admissible ones
        V_p = B[:rank_adm, :]                          # (rank_adm, q)
        vecs_list.append(V_p)
        if verbose:
            print(f"  pair {p}: rank_adm={rank_adm}  "
                  f"(pi_c[{p},{s_first}] shape={np.asarray(pi_c[p, s_first]).shape})")

    if not vecs_list:
        if verbose:
            print("  No non-empty pairs found.")
        return None

    V = np.vstack(vecs_list)   # (total_raw_rank, q)

    # Global re-orthogonalisation (pairs may share leading directions)
    _, sv, Vt = np.linalg.svd(V, full_matrices=False)
    tol = sv[0] * q * np.finfo(float).eps * 100
    rank = int(np.sum(sv > tol))
    V_orth = Vt[:rank, :]      # (rank, q)

    if verbose:
        print(f"\n  [q={q}] raw stacked rank={V.shape[0]}, global rank after SVD={rank}")
        print(f"  leading singular values: {sv[:min(6, len(sv))].round(6)}")

    return V_orth


# ---------------------------------------------------------------------------
# M_q(R) computation
# ---------------------------------------------------------------------------

def compute_Mq(V_orth, q, R_values):
    """
    M_q(R) = sigma_max( F_{>R} V^H )^2
    where V_orth is (rank, q) with orthonormal rows.
    """
    F = dft_matrix(q)
    FVH = F @ V_orth.conj().T    # (q, rank)
    results = {}
    for R in R_values:
        mask = high_freq_mask(q, R)
        A = FVH[mask, :]          # (n_high_freq, rank)
        sv = np.linalg.svd(A, compute_uv=False)
        results[R] = float(sv[0] ** 2) if len(sv) > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Admissibility weights  (for [H-E1'] bound c = pi^2 * a_min / q^4)
# ---------------------------------------------------------------------------

def weight_summary(data):
    """
    Return (a_min, a_max) from adm_w_c_* arrays, ignoring NaN (empty pairs).
    Returns (None, None) if no valid weights found.
    """
    keys = ['adm_w_c_X', 'adm_w_c_Xinv', 'adm_w_c_Y', 'adm_w_c_Yinv',
            'adm_w_qmc_X', 'adm_w_qmc_Xinv', 'adm_w_qmc_Y', 'adm_w_qmc_Yinv']
    all_w = []
    for k in keys:
        if k in data:
            all_w.append(np.asarray(data[k]).ravel())
    if not all_w:
        return None, None
    w = np.concatenate(all_w)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return None, None
    return float(w.min()), float(w.max())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(all_results, all_weight_bounds):
    primes = sorted(all_results.keys())
    all_R   = sorted({R for r in all_results.values() for R in r})
    width   = 10 + len(all_R) * 12
    sep     = "=" * width

    print(f"\n{sep}")
    print("M_q(R) = ||P_{{>R}} Pi_q||_op^2  --  BFS spectral confinement")
    print(sep)
    print(f"{'q':>6}  " + "".join(f"  R={R:<7}" for R in all_R))
    print("-" * width)
    for q in primes:
        print(f"q={q:<5}" +
              "".join(f"  {all_results[q].get(R, float('nan')):8.2e}" for R in all_R))
    print(sep)

    print("\n[H-E1'] bound: lambda_1(E_q) >= c/q^4,  c = pi^2 * a_min")
    for q in primes:
        a_min, a_max = all_weight_bounds.get(q, (None, None))
        if a_min is not None:
            c = np.pi ** 2 * a_min
            print(f"  q={q:<5}  a_min={a_min:.4e}  a_max={a_max:.4e}  "
                  f"c={c:.4e}  lambda_1 >= {c / q**4:.3e}")
        else:
            print(f"  q={q:<5}  (weights unavailable -- check adm_w_c_* keys)")

    print("\n% --- LaTeX table snippet ---")
    print("% \\begin{tabular}{r" + "r" * len(all_R) + "}")
    print("% $q$ & " + " & ".join(f"$R={R}$" for R in all_R) + " \\\\")
    print("% \\midrule")
    for q in primes:
        vals = " & ".join(
            f"${all_results[q].get(R, float('nan')):.2e}$" for R in all_R)
        print(f"% {q} & {vals} \\\\")
    print("% \\end{tabular}")


def plot_results(all_results, out_path="bfs_confinement.pdf"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return

    primes  = sorted(all_results.keys())
    all_R   = sorted({R for r in all_results.values() for R in r})
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(primes)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for q, col in zip(primes, colors):
        R_vals = sorted(all_results[q].keys())
        M_vals = [all_results[q][R] for R in R_vals]
        ax.semilogy(R_vals, M_vals, 'o-', color=col, label=f"$q={q}$",
                    linewidth=1.8, markersize=5)
    ax.set_xlabel("Frequency cutoff $R$", fontsize=12)
    ax.set_ylabel(r"$M_q(R)=\|\mathrm{P}_{>R}\Pi_q\|_{\mathrm{op}}^2$", fontsize=11)
    ax.set_title("Spectral confinement of admissible sector", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    R_fixed = all_R[0] if all_R else 1
    q_arr   = np.array(primes, dtype=float)
    M_fixed = [all_results[q].get(R_fixed, np.nan) for q in primes]
    valid   = [(q, m) for q, m in zip(q_arr, M_fixed) if np.isfinite(m)]
    if valid:
        q0, m0 = valid[0]
        ref = m0 * (q0 / q_arr) ** 2
        ax2.semilogy(q_arr, ref, 'r--', linewidth=1, alpha=0.6,
                     label=r"$\propto q^{-2}$ (reference)")
    ax2.semilogy(q_arr, M_fixed, 'ks-', linewidth=1.8, markersize=6,
                 label=f"$M_q(R={R_fixed})$")
    ax2.set_xlabel("Prime $q$", fontsize=12)
    ax2.set_ylabel(f"$M_q(R={R_fixed})$", fontsize=12)
    ax2.set_title(f"Confinement decay vs $q$ at $R={R_fixed}$", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Diagnose mode
# ---------------------------------------------------------------------------

def run_diagnose(data, q):
    print(f"\n=== Diagnostics for q={q} ===")
    basis_c = data['basis_c']
    pi_c    = data['pi_c']
    vecs_c  = data['vecs_c']
    n_pairs, n_shells = pi_c.shape
    print(f"n_pairs={n_pairs}, n_window_shells={n_shells}, q={q}")

    print("\nPer-pair summary (rank_adm, basis_c shape, pi_c[p,0] shape):")
    for p in range(n_pairs):
        rank_adm, s_first = admissible_rank_from_pi(pi_c[p])
        B = np.asarray(basis_c[p])
        pi_shape = np.asarray(pi_c[p, s_first]).shape if s_first >= 0 else "N/A"
        vc_shape = np.asarray(vecs_c[p, 0]).shape
        if rank_adm > 0:
            print(f"  pair {p:2d}: rank_adm={rank_adm}  "
                  f"basis_c={B.shape}  pi_c[{p},{s_first}]={pi_shape}  "
                  f"vecs_c[{p},0]={vc_shape}")

    # Verify unitarity on first non-empty pair
    B0 = np.asarray(basis_c[0]).astype(complex)
    print(f"\nbasis_c[0] unitarity: ||B B^H - I|| = "
          f"{np.linalg.norm(B0 @ B0.conj().T - np.eye(q)):.2e}")

    # Fourier profile for every admissible vector across all non-empty pairs
    print("\nFourier profile of all admissible vectors (basis_c[p][:rank_adm, :]):")
    print(f"  {'pair':>4}  {'row':>3}  {'xi_peak':>7}  "
          f"{'P(xi=0)':>10}  {'P(xi=1)':>10}  {'P(xi=2)':>10}  "
          f"{'R_90%':>6}  {'R_99%':>6}")
    print("  " + "-" * 70)
    for p in range(n_pairs):
        rank_adm, _ = admissible_rank_from_pi(pi_c[p])
        if rank_adm == 0:
            continue
        B = np.asarray(basis_c[p]).astype(complex)
        for r in range(rank_adm):
            v = B[r, :]
            fv = np.abs(np.fft.fft(v) / np.sqrt(q)) ** 2
            xi_peak = int(np.argmax(fv))
            if xi_peak > q // 2:
                xi_peak = q - xi_peak   # fold to positive frequencies
            fv_sorted = np.sort(fv)[::-1]
            cum = np.cumsum(fv_sorted) / fv.sum()
            R_90 = int(np.searchsorted(cum, 0.90)) + 1
            R_99 = int(np.searchsorted(cum, 0.99)) + 1
            print(f"  {p:>4}  {r:>3}  {xi_peak:>7d}  "
                  f"{fv[0]:>10.3e}  {fv[1]:>10.3e}  {fv[2]:>10.3e}  "
                  f"{R_90:>6}  {R_99:>6}")

    # Summary: admissible frequency set and max frequency (critical for [H-BFS] scaling)
    adm_freqs = set()
    for p in range(n_pairs):
        rank_adm, _ = admissible_rank_from_pi(pi_c[p])
        if rank_adm == 0:
            continue
        B = np.asarray(basis_c[p]).astype(complex)
        for r in range(rank_adm):
            v = B[r, :]
            fv = np.abs(np.fft.fft(v) / np.sqrt(q)) ** 2
            xi_peak = int(np.argmax(fv))
            if xi_peak > q // 2:
                xi_peak = q - xi_peak
            adm_freqs.add(xi_peak)

    xi_max = max(adm_freqs)
    print(f"\n[H-BFS scaling summary]")
    print(f"  Admissible frequency set: xi in {sorted(adm_freqs)}")
    print(f"  xi_max = {xi_max}   q = {q}   xi_max/q = {xi_max/q:.4f}")
    print(f"  xi_max/sqrt(q) = {xi_max/q**0.5:.4f}")
    print(f"  --> M_q(R) = 0 for R >= {xi_max}  (all admissible mass confined)")

    # Admissibility weights (may be unavailable if _HAS_ADM=False at generation)
    a_min, a_max = weight_summary(data)
    if a_min is not None:
        print(f"\nAdmissibility weights: a_min={a_min:.4e}, a_max={a_max:.4e}")
        print(f"[H-E1'] c = pi^2*a_min = {np.pi**2*a_min:.4e}  "
              f"--> lambda_1(E_q) >= {np.pi**2*a_min/q**4:.3e}")
    else:
        print("\nAdmissibility weights: not stored (_HAS_ADM=False at generation)")
        print("  [H-E1'] bound requires re-running pipeline with admissibility_weight module.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BFS spectral confinement test M_q(R) for Q5a-O2 paper")
    parser.add_argument('--primes',    nargs='+', type=int, default=[29, 61, 101, 151])
    parser.add_argument('--npz-dir',   default='.')
    parser.add_argument('--R-max',     type=int, default=15)
    parser.add_argument('--R-values',  nargs='+', type=int, default=None)
    parser.add_argument('--diagnose',  action='store_true',
                        help="Detailed inspection of admissible structure and exit")
    parser.add_argument('--out-plot',  default='bfs_confinement.pdf')
    args = parser.parse_args()

    R_values = args.R_values if args.R_values else list(range(1, args.R_max + 1))

    all_results       = {}
    all_weight_bounds = {}

    for q in args.primes:
        path = find_npz(q, args.npz_dir)
        if path is None:
            print(f"[q={q}] No npz found in '{args.npz_dir}' -- skipping")
            continue
        print(f"\n[q={q}] Loading {os.path.basename(path)}")
        data    = np.load(path, allow_pickle=True)
        q_actual = int(data['q'])

        if args.diagnose:
            run_diagnose(data, q_actual)
            continue

        V_orth = build_admissible_basis(data, q_actual)
        if V_orth is None:
            print(f"  Failed -- run with --diagnose to inspect")
            continue

        results = compute_Mq(V_orth, q_actual, R_values)
        all_results[q_actual] = results

        a_min, a_max = weight_summary(data)
        all_weight_bounds[q_actual] = (a_min, a_max)

        for R in [1, 3, 5]:
            if R in results:
                print(f"  R={R}: M_q = {results[R]:.3e}")
        if a_min is not None:
            print(f"  [H-E1'] lambda_1(E_q) >= {np.pi**2 * a_min / q_actual**4:.3e}")

    if all_results:
        print_table(all_results, all_weight_bounds)
        plot_results(all_results, out_path=args.out_plot)


if __name__ == '__main__':
    main()