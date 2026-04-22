"""
test_reff_phi_o27.py
====================
O26 Test 4 (Criterion 5.4) with the canonical Phi_{q,rho} embedding from O27.

Construction: Phi_{q,rho} = rho o iota o pi
  pi   : C^q -> H_eff  (admissible quotient, dim=3 by O23)
         pi(v) = basis_mat @ v    where basis_mat is the GS basis at end of window
  iota : H_eff -> su(2)           (canonical quaternionic identification)
         H_eff = span{e1, e2, e3} maps to su(2) basis {i*sigma_x, i*sigma_y, i*sigma_z}
  rho  : su(2) -> End(V_rho)      (irreducible SU(2) representation of spin j)
         j=1/2 -> d_rho=2,  j=1 -> d_rho=3,  j=3/2 -> d_rho=4

Test 4 protocol (O26 Criterion 5.4):
  For each pair {c, q-c}, for each shell n in [n0,n1]:
    u_n = Phi(v_c^(n))  in C^{d_rho}          (O26 Eq.17-18)
    M_n = u_n x u_n^*   in End(V_rho)          (rank-1 matrix)
    C_c = (1/N) sum_n vec(M_n) vec(M_n)^dag    (covariance, shape d_rho^4)
    r_eff = rank(C_c)

Prediction: r_eff = d_rho^2 = 4 (spin-1/2) or 16 (spin-3/2)

USAGE
-----
  python test_reff_phi_o27.py q61_o25.npz
  python test_reff_phi_o27.py q29_o25.npz --spin 0.5 --pairs 0
"""

import argparse, sys
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# su(2) representation matrices
# ─────────────────────────────────────────────

def su2_generators(j):
    """
    Return (Jx, Jy, Jz) as d x d complex matrices for spin j.
    d = 2j+1,  m runs from -j to +j.
    Conventions: standard physics (Condon-Shortley).
    """
    d = int(round(2*j + 1))
    m = np.arange(-j, j+1, 1.0)   # -j, -j+1, ..., +j
    Jz = np.diag(m).astype(complex)
    # J+ = sum_m sqrt(j(j+1)-m(m+1)) |m+1><m|
    Jp = np.zeros((d, d), dtype=complex)
    for i, mi in enumerate(m[:-1]):
        Jp[i+1, i] = np.sqrt(j*(j+1) - mi*(mi+1))
    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    return Jx, Jy, Jz


def rho_matrix(X, j):
    """
    Apply the spin-j representation rho: su(2) -> End(V_rho).
    X = (x1, x2, x3) in R^3 (coordinates in su(2) basis {iJx, iJy, iJz}).
    Returns x1*(iJx) + x2*(iJy) + x3*(iJz) as a d x d matrix.
    """
    Jx, Jy, Jz = su2_generators(j)
    return X[0] * (1j*Jx) + X[1] * (1j*Jy) + X[2] * (1j*Jz)


# ─────────────────────────────────────────────
# pi: projection onto H_eff via basis_mat
# ─────────────────────────────────────────────

def pi_project(v, basis_mat):
    """
    Project v in C^q onto H_eff = span(basis_mat).
    Returns coordinates in basis_mat: shape (rank,).
    pi(v) = basis_mat @ v   (basis_mat has orthonormal rows)
    """
    if basis_mat is None or basis_mat.shape[0] == 0:
        return np.zeros(0, dtype=complex)
    return basis_mat @ v


# ─────────────────────────────────────────────
# iota: H_eff -> su(2) identification
# ─────────────────────────────────────────────

def iota_identify(h_eff_coords):
    """
    Map H_eff coordinates (dim 3) to R^3 su(2) coordinates.
    iota takes the real part: the su(2) basis {iJx, iJy, iJz} is anti-Hermitian,
    and H_eff carries a real structure via the BI involution.
    We use: x_k = Re(h_eff_coords[k]) after normalisation.
    """
    c = h_eff_coords
    if len(c) == 0:
        return np.zeros(3, dtype=float)
    # Take real part of first 3 coordinates; pad or truncate to dim 3
    real_c = np.real(c[:3]) if len(c) >= 3 else np.pad(np.real(c), (0, 3-len(c)))
    return real_c


# ─────────────────────────────────────────────
# Full Phi = rho o iota o pi
# ─────────────────────────────────────────────

def Phi(v, basis_mat, j):
    """
    Apply Phi_{q,rho} = rho o iota o pi to v in C^q.
    Returns a d_rho-vector in C^{d_rho} (the image in V_rho).
    Here we take the first column of rho(iota(pi(v))) as the representative vector.
    """
    h = pi_project(v, basis_mat)         # C^{rank}
    x = iota_identify(h)                 # R^3
    # rho(x) is a d x d matrix; as a vector in V_rho we take it as a flattened matrix
    # For the outer product M_n = Phi(v_c) x Phi(v_{qc})^dagger, the natural choice
    # is to treat rho(x) in End(V_rho) directly as M_n.
    M = rho_matrix(x, j)                 # d x d matrix
    return M


# ─────────────────────────────────────────────
# Reconstruct v_c^(n) from stored residuals + basis
# ─────────────────────────────────────────────

def reconstruct_vc(residuals_k, shells_k, basis_up_to_k, c_block, gens, q):
    """
    Reconstruct the full vector v_c^(n) in C^q from the normalised residual
    norms stored for shell n.

    Since we only stored ||r_s^(n)|| / ||v_s|| and not the directions,
    we cannot fully reconstruct v_c^(n) in C^q.

    Instead we use pi(v_c^(n)) directly: the projection onto H_eff is
    equivalent to the coordinates in basis_mat, which ARE the residuals
    for the basis vectors.

    For the embedding, we use the AGGREGATE vector:
      pi_coords[k] = sum_s (residual_k[s] * basis_row_k)
    which gives the centroid of the projected shell in H_eff.
    """
    if residuals_k is None or len(residuals_k) == 0:
        return np.zeros(0, dtype=complex)
    if basis_up_to_k is None or basis_up_to_k.shape[0] == 0:
        return np.zeros(0, dtype=complex)
    rank = basis_up_to_k.shape[0]
    # The residuals r[s] = ||w_s||/||v_s|| are normalised scalars.
    # The projection of v_s^(n) onto basis direction k is:
    #   <e_k, v_s> / ||v_s||  (before subtraction)
    # We approximate pi(v_c^(n)) as the mean squared residual per basis direction.
    # Since we don't have directional info, use the aggregate:
    #   coord_k ~ sqrt(sigma_c(n)) * delta_k,admissible
    # Best available: use ||v_c^(n)||^2 = sum_s r[s]^2 as the norm,
    # and distribute equally across the admissible rank dimensions.
    norm_sq = float(np.sum(residuals_k**2))
    if norm_sq < 1e-30:
        return np.zeros(rank, dtype=complex)
    # Distribute norm_sq evenly across admissible directions
    # This is the maximally uncertain prior given only the aggregate norm
    coords = np.ones(rank, dtype=complex) * np.sqrt(norm_sq / rank)
    return coords


# ─────────────────────────────────────────────
# Main Test 4 computation
# ─────────────────────────────────────────────

def compute_reff_phi(pi_c_list, pi_qmc_list, j, threshold=1e-10):
    """
    Compute r_eff for one pair using the canonical Phi embedding (O27).

    pi_c_list[k]   = (N_s, 3) complex matrix of per-vector H_eff projections
                     for shell n0+k.  Row s = pi(v_s^c) = basis_heff @ v_s.
    pi_qmc_list[k] = same for q-c.

    Phi pipeline per shell k (O26 Eq.17-18):
      For each vector index s:
        x_c[s]  = iota(pi_c[s]) in R^3
        x_qc[s] = iota(pi_qmc[s]) in R^3
        rho_c[s]  = rho(x_c[s])  in End(V_rho), d x d
        rho_qc[s] = rho(x_qc[s]) in End(V_rho), d x d
      M_n = (1/N_s) sum_s rho_c[s] @ rho_qc[s]^dag   (shell average)

    C_c   = (1/N_window) sum_k vec(M_n(k)) vec(M_n(k))^dag
    r_eff = rank(C_c)

    Prediction: r_eff = d_rho^2 = 4 (spin-1/2) or 16 (spin-3/2).
    """
    d = int(round(2*j + 1))
    Mn_list = []

    for pc, pqc in zip(pi_c_list, pi_qmc_list):
        pc  = np.asarray(pc  if pc  is not None else np.zeros((0,3)), dtype=complex)
        pqc = np.asarray(pqc if pqc is not None else np.zeros((0,3)), dtype=complex)
        # pc shape: (N_s, 3)
        if pc.ndim == 1:   pc  = pc[np.newaxis, :]   # backward compat
        if pqc.ndim == 1:  pqc = pqc[np.newaxis, :]
        if pc.shape[0] == 0 or pqc.shape[0] == 0:
            continue

        # Per-vector construction (O26 Eq.17-18, O27):
        #   u_s = rho(iota(pi_c[s])) |e_0>  in V_rho  (d-vector)
        #   v_s = rho(iota(pi_qmc[s])) |e_0> in V_rho
        #   M_s = u_s x v_s^dag  in End(V_rho)         (rank-1, d x d)
        # Covariance over all s across all shells -> r_eff = rank(C_c).
        # Prediction: r_eff = d_rho^2 if (u_s, v_s) span End(V_rho).
        N_s = min(pc.shape[0], pqc.shape[0])
        x_c  = np.real(pc[:N_s])    # (N_s, 3)
        x_qc = np.real(pqc[:N_s])   # (N_s, 3)
        norms_c  = np.linalg.norm(x_c,  axis=1)
        norms_qc = np.linalg.norm(x_qc, axis=1)
        valid = (norms_c > 1e-15) & (norms_qc > 1e-15)
        if not np.any(valid):
            continue
        x_c  = x_c[valid]
        x_qc = x_qc[valid]
        Jx, Jy, Jz = su2_generators(j)
        iJx = 1j * Jx; iJy = 1j * Jy; iJz = 1j * Jz
        e0 = np.zeros(d, dtype=complex); e0[0] = 1.0
        # rho(x): (K, d, d)
        rho_c  = (x_c[:,0,None,None]*iJx + x_c[:,1,None,None]*iJy
                + x_c[:,2,None,None]*iJz)
        rho_qc = (x_qc[:,0,None,None]*iJx + x_qc[:,1,None,None]*iJy
                + x_qc[:,2,None,None]*iJz)
        # u_s = rho_c[s] @ e0, v_s = rho_qc[s] @ e0: (K, d)
        u = np.einsum('kab,b->ka', rho_c,  e0)   # (K, d)
        v = np.einsum('kab,b->ka', rho_qc, e0)   # (K, d)
        # M_s = outer(u_s, v_s^*): (K, d, d) -> (K, d^2)
        Mk = np.einsum('ka,kb->kab', u, v.conj()).reshape(len(u), d*d)
        Mn_list.extend(Mk)

    if len(Mn_list) < 2:
        return 0, np.array([])

    V = np.stack(Mn_list, axis=0)          # (N_window, d^2)
    N = len(Mn_list)
    G = (V @ V.conj().T) / N              # (N_window, N_window) Gram
    svals = np.linalg.svd(G, compute_uv=False)
    thresh = threshold * svals[0] if svals[0] > 0 else 0.0
    r_eff = int(np.sum(svals > thresh))
    return r_eff, svals


def run_test4_phi(npz_path, spin=0.5, n_pairs_show=10, threshold=1e-10):
    d = np.load(npz_path, allow_pickle=True)

    if 'vecs_c' not in d or 'vecs_qmc' not in d:
        print("ERROR: run o25_paired_pipeline.py --store-vectors --force first.")
        sys.exit(1)

    q     = int(d['q'])
    n0    = int(d['n0'])
    n1    = int(d['n1'])
    pairs = d['pairs']
    P     = len(pairs)
    pi_c_arr   = d['pi_c']   if 'pi_c'   in d else None
    pi_qmc_arr = d['pi_qmc'] if 'pi_qmc' in d else None
    has_pi = pi_c_arr is not None
    j     = spin
    d_rho = int(round(2*j + 1))
    limit = P if n_pairs_show == 0 else min(P, n_pairs_show)

    print("\n" + "="*62)
    print(f"O26 Test 4 with Phi_{'{q,rho}'}  q={q}  spin={j}  d_rho={d_rho}")
    print(f"  window=[{n0},{n1}]  {P} pairs")
    print("="*62)
    print(f"  Prediction: r_eff = d_rho^2 = {d_rho**2}")
    if has_pi:
        rank0 = len(pi_c_arr[0, 0]) if pi_c_arr[0, 0] is not None else 0
        print(f"  pi_coords stored: rank={rank0}  (exact Phi via O27)")
    else:
        print("  pi_coords not found — re-run with --store-vectors --force")

    print(f"\n  {'pair':>8}  {'r_eff':>6}  {'sigma_1':>12}  {'sigma_2':>12}")

    reff_all = []
    for i in range(P):
        c, qc = int(pairs[i, 0]), int(pairs[i, 1])
        pc_list  = list(pi_c_arr[i])   if has_pi else []
        pqc_list = list(pi_qmc_arr[i]) if has_pi else []

        r_eff, svals = compute_reff_phi(pc_list, pqc_list, j=j,
                                        threshold=threshold)
        reff_all.append(r_eff)

        if i < limit:
            sv = list(svals[:2]) + [0.0] * max(0, 2 - len(svals))
            mark = " *" if r_eff == d_rho**2 else "  "
            print(f"  ({c:2d},{qc:2d}){mark}  {r_eff:>6d}  "
                  + "  ".join(f"{s:12.4e}" for s in sv))

    if P > limit:
        print(f"  ... {P - limit} more pairs")

    reff_arr = np.array(reff_all)
    nz = reff_arr[reff_arr > 0]
    print("\n" + "="*62)
    print("SUMMARY")
    if len(nz):
        unique, counts = np.unique(nz, return_counts=True)
        for u, cnt in zip(unique, counts):
            t = {d_rho**2: f"  <- PREDICTED (spin={j}) confirmed"}.get(int(u), "")
            print(f"  r_eff = {int(u):2d} : {cnt}/{P} pairs{t}")
    else:
        print("  All r_eff = 0 — check that --store-vectors was used")
    print("="*62 + "\n")
    return reff_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--spin", type=float, default=0.5,
                   help="SU(2) spin: 0.5 (d=2), 1.0 (d=3), 1.5 (d=4)")
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--threshold", type=float, default=1e-10)
    args = p.parse_args()
    run_test4_phi(args.npz, spin=args.spin,
                  n_pairs_show=args.pairs, threshold=args.threshold)

if __name__ == "__main__":
    main()