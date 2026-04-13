"""
dump_orbits.py
Extract orbit vectors X_c and X_qc for a given conjugate pair (c, q-c)
at a given BFS depth, and compute the GS separator R_n^(c).

Uses the same BFS and Weil machinery as spectral_O12.py.

Usage:
    python dump_orbits.py --q 61 --c 1 --n 5
    python dump_orbits.py --q 61 --c 1 --n 5 --save orbits_q61_c1_n5.npz
    python dump_orbits.py --q 61 --c 1 --n 5 --all-depths

The script prints the SeparatorResult for the given depth, or a table
of R_n values across all depths if --all-depths is specified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# import machinery from spectral_O12
sys.path.insert(0, str(Path(__file__).parent))
from spectral_O12 import (
    build_generators,
    bfs_shells,
    fingerprint_vectors_batch,
    coherence_length,
    EPS_GS,
)
from separator import separator_scores, orthonormal_basis, SeparatorResult


def collect_orbit_vectors(shells, c_block, gens, q, depth: int) -> np.ndarray:
    """
    Collect all fingerprint vectors produced at BFS depth `depth` for block c_block.
    Returns array of shape (m, q) where m = |S_depth| * 4^3.
    """
    gens_arr = np.array(gens, dtype=np.int64)
    c_block = np.asarray(c_block, dtype=np.int64)
    if depth >= len(shells):
        raise ValueError(f"Depth {depth} exceeds available shells ({len(shells)})")
    shell = shells[depth]
    if len(shell) == 0:
        return np.zeros((0, q), dtype=np.complex128)
    shell_arr = np.array(shell, dtype=np.int64)
    return fingerprint_vectors_batch(shell_arr, c_block, gens_arr, q)


def collect_orbit_vectors_cumulative(shells, c_block, gens, q,
                                     depth: int) -> np.ndarray:
    """
    Collect all fingerprint vectors from depth 0 to `depth` (inclusive).
    This gives the full orbit span V_n^(c) used in the GS basis.
    Returns array of shape (m_total, q).
    """
    gens_arr = np.array(gens, dtype=np.int64)
    c_block = np.asarray(c_block, dtype=np.int64)
    all_vecs = []
    for d in range(min(depth + 1, len(shells))):
        shell = shells[d]
        if len(shell) == 0:
            continue
        shell_arr = np.array(shell, dtype=np.int64)
        vecs = fingerprint_vectors_batch(shell_arr, c_block, gens_arr, q)
        all_vecs.append(vecs)
    if not all_vecs:
        return np.zeros((0, q), dtype=np.complex128)
    return np.concatenate(all_vecs, axis=0)


def compute_separator_at_depth(shells, c, q, gens, depth: int,
                                cumulative: bool = True) -> dict:
    """
    Compute the GS separator for conjugate pair (c, q-c) at given depth.
    Returns dict with separator metrics and coherence.
    """
    q_minus_c = (q - c) % q

    if cumulative:
        x_c = collect_orbit_vectors_cumulative(shells, [c, 0, 0], gens, q, depth)
        x_qc = collect_orbit_vectors_cumulative(
            shells, [q_minus_c, 0, 0], gens, q, depth)
    else:
        x_c = collect_orbit_vectors(shells, [c, 0, 0], gens, q, depth)
        x_qc = collect_orbit_vectors(shells, [q_minus_c, 0, 0], gens, q, depth)

    if x_c.shape[0] == 0 or x_qc.shape[0] == 0:
        return dict(depth=depth, rank_vc=0, rank_vqc=0, rank_w=0,
                    energy_c=0.0, energy_qc=0.0, r_trace=None,
                    ell_gamma=float(coherence_length(shells, q)[depth]))

    # Transpose: separator expects (d, m) not (m, d)
    result = separator_scores(x_c.T, x_qc.T, tol=EPS_GS)
    ell = coherence_length(shells, q)

    return dict(
        depth=depth,
        rank_vc=result.rank_vc,
        rank_vqc=result.rank_vqc,
        rank_w=result.rank_w,
        energy_c=result.energy_c,
        energy_qc=result.energy_qc,
        r_trace=result.r_trace,
        ell_gamma=float(ell[depth]) if depth < len(ell) else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract orbit vectors and compute GS separator R_n^(c)."
    )
    parser.add_argument("--q", type=int, required=True,
                        help="Prime q")
    parser.add_argument("--c", type=int, required=True,
                        help="Character index c (block c, conjugate q-c)")
    parser.add_argument("--n", type=int, default=None,
                        help="BFS depth (required unless --all-depths)")
    parser.add_argument("--all-depths", action="store_true",
                        help="Compute separator at all available depths")
    parser.add_argument("--bfs-frac", type=float, default=0.99,
                        help="BFS fraction of q^3 nodes to explore")
    parser.add_argument("--cumulative", action="store_true", default=True,
                        help="Use cumulative orbit vectors up to depth n (default)")
    parser.add_argument("--shell-only", action="store_false", dest="cumulative",
                        help="Use only shell n vectors (not cumulative)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save orbit vectors to this .npz file")
    args = parser.parse_args()

    q = args.q
    c = args.c % q
    q_minus_c = (q - c) % q

    if args.n is None and not args.all_depths:
        parser.error("Specify --n DEPTH or --all-depths")

    print(f"q={q}, pair=({c},{q_minus_c}), bfs_frac={args.bfs_frac}")
    gens = build_generators(q)
    shells = bfs_shells(None, None, gens, q, args.bfs_frac)
    n_shells = len(shells)
    print(f"BFS: {n_shells} shells")

    ell = coherence_length(shells, q)

    if args.all_depths:
        print(f"\n{'depth':>6} {'ell_gamma':>10} {'rank_W':>8} "
              f"{'energy_c':>12} {'energy_qc':>12}")
        print("-" * 55)
        for depth in range(1, n_shells):
            res = compute_separator_at_depth(
                shells, c, q, gens, depth, cumulative=args.cumulative)
            print(f"{depth:>6} {res['ell_gamma']:>10.4f} {res['rank_w']:>8} "
                  f"{res['energy_c']:>12.6f} {res['energy_qc']:>12.6f}")
        return

    depth = args.n
    res = compute_separator_at_depth(
        shells, c, q, gens, depth, cumulative=args.cumulative)
    print(json.dumps(res, indent=2))

    if args.save is not None:
        if args.cumulative:
            x_c = collect_orbit_vectors_cumulative(
                shells, [c, 0, 0], gens, q, depth)
            x_qc = collect_orbit_vectors_cumulative(
                shells, [q_minus_c, 0, 0], gens, q, depth)
        else:
            x_c = collect_orbit_vectors(shells, [c, 0, 0], gens, q, depth)
            x_qc = collect_orbit_vectors(shells, [q_minus_c, 0, 0], gens, q, depth)
        np.savez(args.save, X_c=x_c.T, X_qc=x_qc.T)
        print(f"Saved orbit vectors to {args.save}")
        print(f"  X_c shape: {x_c.T.shape}, X_qc shape: {x_qc.T.shape}")


if __name__ == "__main__":
    main()