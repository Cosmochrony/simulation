"""
BFS Spectral Confinement Test
==============================
Computes M_q(R) = ||P_{>R} Pi_q||_op^2 for the admissible projection,
where P_{>R} is the projection onto Fourier modes with |xi| > R.

This is the direct numerical test of Hypothesis [H-BFS] (Admissible BFS
spectral confinement) required for §4 of the Q5a-O2 resolution paper.

M_q(R) is the maximum fraction of spectral mass that any admissible vector
can place above frequency R.  If M_q(R) -> 0 uniformly as R grows (with
R fixed and q -> inf), spectral confinement holds.

Usage:
    # Inspect npz structure first:
    python bfs_confinement_test.py --inspect --primes 29 61 101 151

    # Run full confinement test:
    python bfs_confinement_test.py --primes 29 61 101 151 --R-max 20

The script auto-detects the npz key layout (pi_c matrix vs sigma_c scalars).
"""

import argparse
import glob
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Fourier helpers
# ---------------------------------------------------------------------------

def dft_matrix(q):
    """Return the q x q unitary DFT matrix F with F[xi,k] = q^{-1/2} e^{-2pi i k xi/q}."""
    k = np.arange(q)
    xi = np.arange(q)
    return np.exp(-2j * np.pi * np.outer(xi, k) / q) / np.sqrt(q)


def high_freq_projector(q, R):
    """Return the diagonal 0/1 mask for |xi| > R (indices 1..R and q-R..q-1 kept low)."""
    mask = np.ones(q, dtype=bool)
    mask[0] = False             # xi = 0
    for r in range(1, R + 1):
        mask[r] = False         # xi = r
        mask[q - r] = False     # xi = q-r (negative frequency)
    return mask


# ---------------------------------------------------------------------------
# NPZ inspection
# ---------------------------------------------------------------------------

def find_npz(prime, npz_dir="."):
    patterns = [
        os.path.join(npz_dir, f"*q{prime}*.npz"),
        os.path.join(npz_dir, f"*{prime}*.npz"),
        os.path.join(npz_dir, f"q{prime}.npz"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def inspect_npz(path):
    data = np.load(path, allow_pickle=True)
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print(f"Keys: {list(data.keys())}")
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {key}: {type(arr)} = {arr}")
    return data


# ---------------------------------------------------------------------------
# Extract Pi_q from npz
# ---------------------------------------------------------------------------

def extract_pi_q(data, q):
    """
    Try to reconstruct Pi_q (q x q projection matrix) from stored npz data.

    Expected layouts (in order of preference):
      1. data['pi_c']          -- direct (q x q) projection matrix
      2. data['pi_matrix']     -- idem, alternate key
      3. data['fingerprints']  -- (n_vecs x q) matrix of orthonormal admissible
                                  vectors; Pi_q = V V^H
      4. data['basis']         -- idem
    Returns Pi_q as (q, q) complex array, or None if not found.
    """
    for key in ('pi_c', 'pi_matrix', 'Pi_q', 'Pi'):
        if key in data:
            arr = data[key]
            if arr.shape == (q, q):
                return arr.astype(complex)
            # might be averaged over pairs -- still usable
            if arr.ndim == 2 and arr.shape[1] == q:
                return arr.astype(complex)

    for key in ('fingerprints', 'basis', 'vecs', 'admissible_basis'):
        if key in data:
            V = data[key].astype(complex)   # shape (n_vecs, q)
            if V.shape[1] == q:
                return V.conj().T @ V       # Pi_q = V^H V  (note: rows are vectors)

    return None


# ---------------------------------------------------------------------------
# M_q(R) computation
# ---------------------------------------------------------------------------

def compute_Mq(Pi_q, q, R_values):
    """
    Compute M_q(R) = ||P_{>R} Pi_q||_op^2 for each R in R_values.

    P_{>R} Pi_q is the restriction of Pi_q to high-frequency output modes.
    Its operator norm squared = largest singular value squared of the
    (high-freq rows) x (all cols) submatrix of (F Pi_q).

    Equivalently: M_q(R) = max eigenvalue of  Pi_q F^H P_{>R} F Pi_q.
    """
    F = dft_matrix(q)
    # FPi: (q, q) -- Fourier transform of the projected subspace basis
    FPi = F @ Pi_q          # rows indexed by frequency xi

    results = {}
    for R in R_values:
        mask = high_freq_projector(q, R)
        # high-frequency rows of F Pi_q
        HFPi = FPi[mask, :]     # shape (n_high, q)
        # M_q(R) = max singular value^2 of HFPi
        sv = np.linalg.svd(HFPi, compute_uv=False)
        results[R] = float(sv[0] ** 2) if len(sv) > 0 else 0.0
    return results


def compute_Mq_from_basis(V, q, R_values):
    """
    If we have an orthonormal basis V (n_vecs x q), Pi_q = V^H V.
    More efficient: work directly with (F V^H) restricted to high-freq rows.
    F V^H is (q x n_vecs); restrict rows.
    """
    F = dft_matrix(q)
    FVH = F @ V.conj().T    # (q, n_vecs)
    results = {}
    for R in R_values:
        mask = high_freq_projector(q, R)
        HFVh = FVH[mask, :]
        sv = np.linalg.svd(HFVh, compute_uv=False)
        results[R] = float(sv[0] ** 2) if len(sv) > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Summary table + plot
# ---------------------------------------------------------------------------

def print_table(all_results):
    """Print M_q(R) results as a LaTeX-ready table."""
    primes = sorted(all_results.keys())
    all_R = sorted({R for res in all_results.values() for R in res})

    header = "q \\ R  " + "  ".join(f"{R:>8}" for R in all_R)
    print("\n" + "="*len(header))
    print("M_q(R) -- spectral confinement criterion")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for q in primes:
        row = f"q={q:<5}" + "  ".join(
            f"{all_results[q].get(R, float('nan')):8.2e}" for R in all_R
        )
        print(row)
    print("="*len(header))

    # LaTeX table snippet
    print("\n% LaTeX table snippet:")
    print("% \\begin{tabular}{l" + "r"*len(all_R) + "}")
    print("% $q$ & " + " & ".join(f"$R={R}$" for R in all_R) + " \\\\")
    print("% \\midrule")
    for q in primes:
        vals = " & ".join(
            f"${all_results[q].get(R, float('nan')):.2e}$" for R in all_R
        )
        print(f"% {q} & {vals} \\\\")
    print("% \\end{tabular}")


def plot_results(all_results, out_path="bfs_confinement.pdf"):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        primes = sorted(all_results.keys())
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(primes)))

        # Left: M_q(R) vs R for each q
        ax = axes[0]
        for q, col in zip(primes, colors):
            R_vals = sorted(all_results[q].keys())
            M_vals = [all_results[q][R] for R in R_vals]
            ax.semilogy(R_vals, M_vals, 'o-', color=col, label=f"$q={q}$", linewidth=1.5)
        ax.set_xlabel("Frequency cutoff $R$")
        ax.set_ylabel("$M_q(R) = \\|P_{>R}\\Pi_q\\|_{\\mathrm{op}}^2$")
        ax.set_title("Spectral confinement: $M_q(R)$ vs $R$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: M_q(R=1) vs q  (decay with q at fixed R)
        ax2 = axes[1]
        R_fixed = sorted(all_results[primes[0]].keys())[0]
        M_at_R1 = [all_results[q].get(R_fixed, np.nan) for q in primes]
        ax2.semilogy(primes, M_at_R1, 'ks-', linewidth=1.5, label=f"$R={R_fixed}$")
        ax2.set_xlabel("Prime $q$")
        ax2.set_ylabel(f"$M_q(R={R_fixed})$")
        ax2.set_title(f"Confinement at $R={R_fixed}$ vs $q$")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("matplotlib not available -- skipping plot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BFS spectral confinement test M_q(R)")
    parser.add_argument('--primes', nargs='+', type=int, default=[29, 61, 101, 151],
                        help="List of primes to test")
    parser.add_argument('--npz-dir', default='.',
                        help="Directory containing the O25 npz checkpoint files")
    parser.add_argument('--R-max', type=int, default=15,
                        help="Maximum frequency cutoff R to test")
    parser.add_argument('--R-values', nargs='+', type=int, default=None,
                        help="Explicit list of R values (overrides --R-max)")
    parser.add_argument('--inspect', action='store_true',
                        help="Inspect npz structure only, do not compute M_q(R)")
    parser.add_argument('--out-plot', default='bfs_confinement.pdf',
                        help="Output path for the confinement plot")
    args = parser.parse_args()

    R_values = args.R_values if args.R_values else list(range(1, args.R_max + 1))

    all_results = {}

    for q in args.primes:
        path = find_npz(q, args.npz_dir)
        if path is None:
            print(f"[q={q}] No npz found in {args.npz_dir} -- skipping")
            continue

        data = inspect_npz(path)

        if args.inspect:
            continue

        Pi_q = extract_pi_q(data, q)
        if Pi_q is None:
            print(f"[q={q}] Could not extract Pi_q from {path}")
            print(f"  Available keys: {list(data.keys())}")
            print("  --> Check key names above and update extract_pi_q() if needed.")
            continue

        print(f"[q={q}] Pi_q extracted: shape={Pi_q.shape}, "
              f"rank~{int(np.round(np.real(np.trace(Pi_q))))} (trace)")

        results = compute_Mq(Pi_q, q, R_values)
        all_results[q] = results

        # Quick summary for this prime
        print(f"  R=1:  M_q = {results.get(1, float('nan')):.3e}")
        print(f"  R=3:  M_q = {results.get(3, float('nan')):.3e}")
        print(f"  R=10: M_q = {results.get(10, float('nan')):.3e}")

    if all_results:
        print_table(all_results)
        plot_results(all_results, out_path=args.out_plot)


if __name__ == '__main__':
    main()