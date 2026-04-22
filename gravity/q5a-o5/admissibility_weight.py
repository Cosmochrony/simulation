"""
admissibility_weight.py
=======================
Q5a-O5: correct computation of a_q(s) from raw Weil fingerprint vectors.

FORMULA (Q5a Definition 2.3)
-----------------------------
    a_q(s, n, c) = E_v[ ||(rho_q(s) - I) Pi_q v||^2 ]

where:
  - v are Weil fingerprint vectors at BFS shell n, character c
  - Pi_q v = B (B† v) is the admissible projection (B = GS basis up to n1)
  - rho_q(s) is applied analytically (X: modulation, Y: shift)
  - E_v = average over all fingerprint vectors in the shell

a_q(s) = E_{n,c}[ a_q(s, n, c) ]  averaged over shells in [n0, n1]

KEY INSIGHT on why basis vectors gave A=2:
  For any unit vector e with uniform magnitude |e_k|^2 ~ 1/q (which holds
  for GS basis vectors in the Weil representation):
    Re<e, rho_q(X) e> = sum_k |e_k|^2 exp(2pi i k/q) ~ 0
  => ||(rho_q(X)-I)e||^2 = 2 - 2*Re<...> = 2  (always, regardless of subspace)

  For PROJECTED fingerprint vectors Pi_q v = B B† v:
  - norm: ||Pi_q v||^2 = 1 - ||residual||^2 = 1 - sigma_hat^2
  - the magnitude distribution of (Pi_q v)_k depends on the BFS geometry
  - this CAN give Re<Pi_q v, rho_q(s) Pi_q v> != 0, yielding A != 2

INTEGRATION
-----------
Requires spectral_O12.compute_block_capacity_with_residuals to return
shell_all_vecs as 8th element (patched spectral_O12.py).

In _compute_one_pair (store_vectors=True, m==0):
    # After the existing block:
    adm_c, adm_qmc = compute_adm_weights_from_vecs(
        _sv_c, _sv_qmc, _basis_c, _basis_qmc, q, n0, n1)
"""
import os

import numpy as np


# ── Weil generator actions on batches of vectors ─────────────────────────────

def rho_X(vecs, q):
    """rho_q(X): modulation.  vecs shape (N, q) -> (N, q)."""
    k = np.arange(q, dtype=float)
    phase = np.exp(2j * np.pi * k / q)   # (q,)
    return vecs * phase                    # broadcast: (N, q) * (q,)

def rho_Xinv(vecs, q):
    k = np.arange(q, dtype=float)
    phase = np.exp(-2j * np.pi * k / q)
    return vecs * phase

def rho_Y(vecs, q):
    """rho_q(Y): cyclic shift left by 1.  vecs shape (N, q) -> (N, q)."""
    return np.roll(vecs, -1, axis=-1)

def rho_Yinv(vecs, q):
    return np.roll(vecs, +1, axis=-1)


GENERATORS = {
    'X':    rho_X,
    'Xinv': rho_Xinv,
    'Y':    rho_Y,
    'Yinv': rho_Yinv,
}


# ── Per-shell admissibility weight ────────────────────────────────────────────

def adm_weight_shell(shell_vecs, basis, q):
    """
    Compute a_q(s, n, c) for one shell.

    Parameters
    ----------
    shell_vecs : ndarray (N, q), complex
        Raw Weil fingerprint vectors for this shell (unit norm).
    basis      : ndarray (rank, q), complex
        GS orthonormal admissible basis (from compute_block_capacity_with_residuals).
    q          : int

    Returns
    -------
    weights : dict {gen_name: float}
        a_q(s, n, c) averaged over the N vectors in this shell.
    """
    if shell_vecs is None or shell_vecs.shape[0] == 0 or basis.shape[0] == 0:
        return {s: np.nan for s in GENERATORS}

    # Project onto admissible subspace: Pi_q v = B† (B v†)
    # basis: (rank, q),  shell_vecs: (N, q)
    coeffs    = basis.conj() @ shell_vecs.T          # (rank, N)
    proj_vecs = (basis.T @ coeffs).T                 # (N, q)  = Pi_q * v_i

    # Skip shells where all projections are near-zero (early shells, nothing yet)
    proj_norms = np.linalg.norm(proj_vecs, axis=1)   # (N,)
    valid = proj_norms > 1e-12
    if valid.sum() == 0:
        return {s: np.nan for s in GENERATORS}
    proj_vecs = proj_vecs[valid]                      # (N_valid, q)

    weights = {}
    for name, rho_s in GENERATORS.items():
        rho_pv  = rho_s(proj_vecs, q)                # (N_valid, q)
        diff    = rho_pv - proj_vecs                  # (rho-I) Pi_q v
        w       = np.sum(np.abs(diff)**2, axis=1)     # (N_valid,)
        weights[name] = float(np.mean(w))
    return weights


# ── Full a_q(s) averaged over window [n0, n1] ────────────────────────────────

def compute_adm_weights_from_vecs(shell_all_vecs, basis, q, n0, n1, ns_window):
    """
    Compute a_q(s, n) for each shell n in [n0, n1], then a_q(s).

    Parameters
    ----------
    shell_all_vecs : list of ndarrays (N_s, q), one per shell in [n0, n1]
        From compute_block_capacity_with_residuals 8th return value.
    basis          : ndarray (rank, q) — basis_mat_window
    q              : int
    n0, n1         : fitting window
    ns_window      : array of shell indices for [n0, n1]

    Returns
    -------
    a_per_shell : dict {gen_name: ndarray (n_window,)}
    a_q         : dict {gen_name: float}
    A_estimate  : float
    """
    n_window    = len(shell_all_vecs)
    gen_names   = list(GENERATORS.keys())
    a_per_shell = {s: np.full(n_window, np.nan) for s in gen_names}

    for k, shell_vecs in enumerate(shell_all_vecs):
        w = adm_weight_shell(shell_vecs, basis, q)
        for s in gen_names:
            a_per_shell[s][k] = w[s]

    # Average over valid shells in window
    a_q = {}
    for s in gen_names:
        vals = a_per_shell[s]
        vals = vals[np.isfinite(vals)]
        a_q[s] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    A_estimate = float(np.nanmean(list(a_q.values())))
    return a_per_shell, a_q, A_estimate


# ── Post-processing on existing npz files ────────────────────────────────────

def process_npz(path, pipeline_dir='.'):
    """
    Recompute a_q(s) by re-running a minimal BFS for the first pair.
    Uses data already in the npz: q, n0, n1, pairs, bfs_frac, n_max_block.

    This avoids modifying the pipeline: it runs a fresh single-block
    compute_block_capacity_with_residuals call and extracts shell_all_vecs.
    """
    import sys, os
    sys.path.insert(0, pipeline_dir)
    from spectral_O12 import (build_generators, bfs_shells,
                               compute_block_capacity_with_residuals)

    d = np.load(path, allow_pickle=True)
    q        = int(d['q'])
    n0       = int(d['n0'])
    n1       = int(d['n1'])
    pairs    = d['pairs']               # (n_pairs, 2)
    bfs_frac = float(d['bfs_frac'])
    n_max    = int(d['n_max_block'])
    ns       = np.asarray(d['ns'], dtype=int)

    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    ns_w   = np.arange(n0, n1 + 1)

    results = {s: [] for s in GENERATORS}

    for i, (c, qc) in enumerate(pairs[:5]):  # first 5 pairs as sample
        # Minimal block: c1=c, c2=1, c3=1 (generic enough for most q)
        # Use the exact seed from the pipeline (seed=42, idx=0)
        rng = np.random.default_rng(42 + i * 997 + int(c) * 7)
        c2  = int(rng.integers(1, q))
        c3  = int(rng.integers(1, q))
        while (int(c) + c2 + c3) % q == 0:
            c3 = int(rng.integers(1, q))
        cb = np.array([c, c2, c3], dtype=np.int64)

        out = compute_block_capacity_with_residuals(
            shells, cb, q, gens, n_max=n_max, n0=n0, n1=n1)
        # 8-tuple: sigma_vals, delta_r, shell_sizes, rank, residuals, basis, pi, shell_vecs
        if len(out) < 8:
            print(f"  [q={q}] spectral_O12 not patched — need 8-tuple return")
            return None
        *_, basis, _, shell_vecs = out

        _, a_q, A = compute_adm_weights_from_vecs(
            shell_vecs, basis, q, n0, n1, ns_w)
        for s in GENERATORS:
            results[s].append(a_q[s])
        A_hw = A * q**2
        print(f"  q={q} pair ({c},{qc}): A_Eucl={A:.5f}  A_hw=A*q^2={A_hw:.3f}  "
              + "  ".join(f"a_{s}={a_q[s]:.5f}" for s in ['X','Y']))

    print(f"\n  q={q}  Grand mean A = {np.nanmean([v for vl in results.values() for v in vl]):.4f}")
    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob, sys

    paths = sorted(glob.glob('o25_outputs/q*_o25.npz'))
    if not paths:
        paths = sorted(glob.glob('../../spectral/o25/o25_outputs/q*_o25.npz'))
    if not paths:
        print("No npz files found.")
        sys.exit(1)

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))

    for p in paths:
        if '.v1.' in p:
            continue
        print(f"\n=== {p} ===")
        process_npz(p, pipeline_dir=pipeline_dir)

