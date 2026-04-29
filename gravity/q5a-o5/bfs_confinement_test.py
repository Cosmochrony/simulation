"""
BFS Spectral Confinement Test  --  Q5a-O2 paper, §4
=====================================================
Computes M_q(R) = ||P_{>R} Pi_q||_op^2 for the admissible projection Pi_q,
where P_{>R} is the spectral projection onto Fourier modes |xi| > R.

Structure of O25 npz files (from inspection):
  basis_c : (n_pairs,) object array
            basis_c[p] = orthonormal basis for conjugate pair p,
            shape (rank_p, q) -- each ROW is one admissible vector
  pi_c    : (n_pairs, n_window_shells) object array
            projection matrices per pair, per shell in fitting window
  pairs   : (n_pairs, 2) int64 -- character indices (c, q-c)
  q       : scalar prime

Computation of M_q(R):
  1. Assemble V = vstack(basis_c[p] for all p)  -- shape (total_rank, q)
  2. Compute F_{>R} = rows of DFT matrix F for |xi| > R
  3. A = F_{>R} @ V.conj().T  -- shape (n_high, total_rank)
  4. M_q(R) = sigma_max(A)^2

Usage:
    # Peek at internal structure of basis_c:
    python bfs_confinement_test.py --peek --npz-dir ./o25_outputs --primes 29

    # Full confinement test, R = 1..15:
    python bfs_confinement_test.py --npz-dir ./o25_outputs --primes 29 61 101 151 --R-max 15

    # Quick run, fewer R values:
    python bfs_confinement_test.py --npz-dir ./o25_outputs --primes 29 61 101 151 \
        --R-values 1 2 3 5 8 12
"""

import argparse
import glob
import os
import numpy as np


def find_npz(prime, npz_dir="."):
    patterns = [
        os.path.join(npz_dir, f"q{prime}_o25.npz"),
        os.path.join(npz_dir, f"*q{prime}*.npz"),
        os.path.join(npz_dir, f"*{prime}*.npz"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]
    return None


def dft_matrix(q):
    k = np.arange(q)
    xi = np.arange(q)
    return np.exp(-2j * np.pi * np.outer(xi, k) / q) / np.sqrt(q)


def high_freq_mask(q, R):
    mask = np.ones(q, dtype=bool)
    for r in range(R + 1):
        mask[r] = False
    for r in range(1, R + 1):
        mask[q - r] = False
    return mask


def coerce_basis(raw, q):
    if raw is None:
        return None
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        if raw.shape[1] == q:
            return raw.astype(complex)
        if raw.shape[0] == q:
            return raw.T.astype(complex)
    if isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.shape[0] == q:
        return raw.astype(complex).reshape(1, q)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        vecs = []
        for item in raw.flat:
            if item is not None:
                arr = np.asarray(item)
                if arr.size == q:
                    vecs.append(arr.ravel().astype(complex))
        if vecs:
            return np.vstack(vecs)
    if isinstance(raw, (list, tuple)):
        vecs = []
        for item in raw:
            arr = np.asarray(item)
            if arr.size == q:
                vecs.append(arr.ravel().astype(complex))
        if vecs:
            return np.vstack(vecs)
    return None


def build_admissible_basis(data, q):
    basis_c = data['basis_c']
    vecs_list = []
    for p in range(basis_c.shape[0]):
        V_p = coerce_basis(basis_c[p], q)
        if V_p is not None:
            vecs_list.append(V_p)
        else:
            print(f"  pair {p}: could not parse basis_c[{p}], type={type(basis_c[p])}")
    if not vecs_list:
        return None
    V = np.vstack(vecs_list)
    _, sv, Vt = np.linalg.svd(V, full_matrices=False)
    tol = sv[0] * max(V.shape) * np.finfo(float).eps * 10
    rank = int(np.sum(sv > tol))
    V_orth = Vt[:rank, :]
    print(f"  [q={q}] vectors stacked: {V.shape[0]}, numerical rank: {rank}")
    return V_orth


def compute_Mq(V_orth, q, R_values):
    F = dft_matrix(q)
    FVH = F @ V_orth.conj().T
    results = {}
    for R in R_values:
        mask = high_freq_mask(q, R)
        A = FVH[mask, :]
        sv = np.linalg.svd(A, compute_uv=False)
        results[R] = float(sv[0] ** 2) if len(sv) > 0 else 0.0
    return results


def weight_summary(data, q):
    keys = ['adm_w_c_X', 'adm_w_c_Xinv', 'adm_w_c_Y', 'adm_w_c_Yinv']
    all_w = []
    for k in keys:
        if k in data:
            all_w.append(data[k])
    if not all_w:
        return None, None
    w = np.concatenate(all_w)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.min()), float(w.max())


def peek_basis(data, q):
    basis_c = data['basis_c']
    pi_c = data['pi_c']
    print(f"\n-- Peek inside basis_c and pi_c for q={q} --")
    for p in range(min(3, len(basis_c))):
        raw = basis_c[p]
        print(f"  basis_c[{p}]: type={type(raw).__name__}", end="")
        if isinstance(raw, np.ndarray):
            print(f", shape={raw.shape}, dtype={raw.dtype}")
            if raw.dtype == object and raw.size > 0:
                inner = raw.flat[0]
                print(f"    -> inner[0]: type={type(inner).__name__}", end="")
                inner_arr = np.asarray(inner) if inner is not None else None
                if inner_arr is not None:
                    print(f", shape={inner_arr.shape}")
                else:
                    print()
        else:
            print(f", value={raw}")
    print()
    for p in range(min(2, pi_c.shape[0])):
        for s in range(min(2, pi_c.shape[1])):
            raw = pi_c[p, s]
            print(f"  pi_c[{p},{s}]: type={type(raw).__name__}", end="")
            if isinstance(raw, np.ndarray):
                print(f", shape={raw.shape}, dtype={raw.dtype}")
            else:
                print()


def print_table(all_results, all_weight_bounds):
    primes = sorted(all_results.keys())
    all_R = sorted({R for res in all_results.values() for R in res})
    width = 10 + len(all_R) * 12
    print("\n" + "=" * width)
    print("M_q(R) = ||P_{>R} Pi_q||_op^2  --  BFS spectral confinement")
    print("=" * width)
    print(f"{'q':>6}  " + "".join(f"  R={R:<7}" for R in all_R))
    print("-" * width)
    for q in primes:
        print(f"q={q:<5}" + "".join(
            f"  {all_results[q].get(R, float('nan')):8.2e}" for R in all_R))
    print("=" * width)
    print("\nAdmissibility weight bounds (for [H-E1'] coercivity c = pi^2 * a_min / q^4):")
    for q in primes:
        a_min, a_max = all_weight_bounds.get(q, (None, None))
        if a_min is not None:
            print(f"  q={q:<5}  a_min={a_min:.4e}  a_max={a_max:.4e}"
                  f"  c(q)={np.pi**2 * a_min / q**4:.3e}")
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
    primes = sorted(all_results.keys())
    all_R = sorted({R for res in all_results.values() for R in res})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(primes)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for q, col in zip(primes, colors):
        R_vals = sorted(all_results[q].keys())
        M_vals = [all_results[q][R] for R in R_vals]
        ax.semilogy(R_vals, M_vals, 'o-', color=col, label=f"$q={q}$",
                    linewidth=1.8, markersize=5)
    ax.set_xlabel("Frequency cutoff $R$", fontsize=12)
    ax.set_ylabel(r"$M_q(R) = \|\mathrm{P}_{>R}\Pi_q\|_{\mathrm{op}}^2$", fontsize=11)
    ax.set_title("Spectral confinement of admissible sector", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    R_fixed = all_R[0] if all_R else 1
    q_arr = np.array(primes, dtype=float)
    M_fixed = [all_results[q].get(R_fixed, np.nan) for q in primes]
    ref = M_fixed[0] * (q_arr[0] / q_arr) ** 2
    ax2.semilogy(primes, M_fixed, 'ks-', linewidth=1.8, markersize=6,
                 label=f"$M_q(R={R_fixed})$")
    ax2.semilogy(q_arr, ref, 'r--', linewidth=1, alpha=0.6, label=r"$\propto q^{-2}$")
    ax2.set_xlabel("Prime $q$", fontsize=12)
    ax2.set_ylabel(f"$M_q(R={R_fixed})$", fontsize=12)
    ax2.set_title(f"Confinement decay with $q$ at $R={R_fixed}$", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primes', nargs='+', type=int, default=[29, 61, 101, 151])
    parser.add_argument('--npz-dir', default='.')
    parser.add_argument('--R-max', type=int, default=15)
    parser.add_argument('--R-values', nargs='+', type=int, default=None)
    parser.add_argument('--peek', action='store_true',
                        help="Inspect basis_c/pi_c internal structure and exit")
    parser.add_argument('--out-plot', default='bfs_confinement.pdf')
    args = parser.parse_args()

    R_values = args.R_values if args.R_values else list(range(1, args.R_max + 1))

    all_results = {}
    all_weight_bounds = {}

    for q in args.primes:
        path = find_npz(q, args.npz_dir)
        if path is None:
            print(f"[q={q}] No npz found in '{args.npz_dir}' -- skipping")
            continue
        print(f"\n[q={q}] Loading {os.path.basename(path)}")
        data = np.load(path, allow_pickle=True)

        if args.peek:
            peek_basis(data, q)
            continue

        V_orth = build_admissible_basis(data, q)
        if V_orth is None:
            print(f"  Failed -- run with --peek to diagnose basis_c layout")
            continue

        results = compute_Mq(V_orth, q, R_values)
        all_results[q] = results

        a_min, a_max = weight_summary(data, q)
        all_weight_bounds[q] = (a_min, a_max)

        for R in [1, 3, 5]:
            if R in results:
                print(f"  R={R}: M_q = {results[R]:.3e}")
        if a_min:
            print(f"  [H-E1'] c(q) = pi^2*a_min/q^4 = {np.pi**2*a_min/q**4:.3e}")

    if all_results:
        print_table(all_results, all_weight_bounds)
        plot_results(all_results, out_path=args.out_plot)


if __name__ == '__main__':
    main()