"""
test_reff_spectral.py
=====================
O26 Test 4 with spectral H_eff basis: eigenvectors of the Cayley graph
Laplacian at ADE values lambda in {5/6, 1, 5/4} (O23).

Construction of pi_spectral:
  1. Build sparse Laplacian L of Cay(G_q, S_q) on all BFS nodes.
  2. Extract 3 eigenvectors near ADE values via shift-invert ARPACK.
  3. For each fingerprint vector v_s in C^q, embed into ell^2(G_q):
       f_s(g) = v_s[g[2] mod q]  (the gamma-component of g)
     (or use the full Weil coefficient map -- see below).
  4. Project f_s onto the 3 spectral directions.

NOTE on the embedding C^q -> ell^2(G_q):
The Weil representation rho_c acts on C^q via:
  (rho_c(a,b,gamma) psi)(x) = exp(2pi i (c * gamma + b*x) / q) * psi(x+a)
The fingerprint vector v_s is the Weil transform applied along a BFS path.
The simplest embedding: v_s -> f_s in ell^2(G_q) where
  f_s((a,b,gamma)) = v_s[a]    (read off the 'a' coordinate)
This is an approximation; the exact map requires the full Weil coefficient.

USAGE
-----
  python test_reff_spectral.py q29_o25.npz
  python test_reff_spectral.py q29_o25.npz --spin 0.5 --pairs 0
"""

import argparse, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pathlib import Path


# ─── su(2) generators ────────────────────────────────────────────────────────

def su2_generators(j):
    d = int(round(2*j + 1))
    m = np.arange(-j, j+1, 1.0)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((d, d), dtype=complex)
    for i, mi in enumerate(m[:-1]):
        Jp[i+1, i] = np.sqrt(j*(j+1) - mi*(mi+1))
    Jm = Jp.conj().T
    return 0.5*(Jp+Jm), -0.5j*(Jp-Jm), Jz


# ─── Build Cayley graph Laplacian ────────────────────────────────────────────

def build_laplacian(q, nodes, gens):
    """
    Sparse normalised Laplacian of Cay(G_q, S_q).
    L = I - (1/deg) * A,  eigenvalues in [0, 2].
    """
    node_idx = {tuple(int(x) for x in g): i for i, g in enumerate(nodes)}

    def heis_mul(a, b):
        return ((a[0]+b[0])%q, (a[1]+b[1])%q, (a[2]+b[2]+a[0]*b[1])%q)

    rows, cols = [], []
    for i, g in enumerate(nodes):
        gi = tuple(int(x) for x in g)
        for s in gens:
            for sign in [1, -1]:
                si = (int(s[0])*sign%q, int(s[1])*sign%q, int(s[2])*sign%q)
                j_idx = node_idx.get(heis_mul(gi, si))
                if j_idx is not None:
                    rows.append(i); cols.append(j_idx)

    N = len(nodes)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    deg = float(A.sum(axis=1).max())
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A) / deg, deg


# ─── Extract spectral H_eff basis ────────────────────────────────────────────

def get_spectral_basis(L, deg, n_per_level=1):
    """
    Extract eigenvectors at ADE levels lambda in {5/6, 1, 5/4}.
    Returns B: (N, 3*n_per_level) orthonormal complex matrix.
    """
    ADE = [5/6, 1.0, 5/4]
    basis_cols = []
    for lam in ADE:
        sigma = lam    # normalised Laplacian
        vals, vecs = spla.eigsh(L, k=max(3, n_per_level+2),
                                sigma=sigma, which='LM', tol=1e-8)
        close = np.abs(vals - sigma) < 0.1
        n_found = close.sum()
        if n_found == 0:
            print(f"  WARNING: no eigenvectors found near lambda={lam:.4f}")
            continue
        use = min(n_per_level, n_found)
        basis_cols.append(vecs[:, close][:, :use])
        print(f"  ADE lambda={lam:.4f}: {n_found} found, using {use}. "
              f"Actual vals: {vals[close][:3].round(5)}")

    if not basis_cols:
        return None
    B = np.hstack(basis_cols)   # (N, <=3)
    # Orthonormalise
    B, _ = np.linalg.qr(B)
    return B.astype(complex)    # (N, 3)


# ─── Embed fingerprint vector into ell^2(G_q) ────────────────────────────────

def embed_fingerprint(v, q, nodes, node_idx):
    """
    Embed fingerprint vector v in C^q into ell^2(G_q) of dim N.
    Map: f(a,b,gamma) = v[a]   (coordinate 'a' of the Heisenberg element).
    This uses the 'a'-component embedding -- an approximation of the
    full Weil coefficient map. Shape: (N,) complex.
    """
    f = np.zeros(len(nodes), dtype=complex)
    for i, g in enumerate(nodes):
        a = int(g[0]) % q
        f[i] = v[a]
    return f


# ─── r_eff computation ───────────────────────────────────────────────────────

def compute_reff_spectral(pi_c_mats, pi_qmc_mats, j, threshold=1e-10):
    """
    pi_c_mats[k]:   (N_s, 3) spectral H_eff projections for shell k
    pi_qmc_mats[k]: same for q-c
    """
    d = int(round(2*j + 1))
    Mn_list = []
    Jx, Jy, Jz = su2_generators(j)
    iJx = 1j*Jx; iJy = 1j*Jy; iJz = 1j*Jz

    for pc, pqc in zip(pi_c_mats, pi_qmc_mats):
        pc  = np.asarray(pc,  dtype=complex)
        pqc = np.asarray(pqc, dtype=complex)
        if pc.shape[0] == 0 or pqc.shape[0] == 0:
            continue

        N_s = min(pc.shape[0], pqc.shape[0])
        x_c  = np.real(pc[:N_s])
        x_qc = np.real(pqc[:N_s])
        norms_c  = np.linalg.norm(x_c,  axis=1)
        norms_qc = np.linalg.norm(x_qc, axis=1)
        valid = (norms_c > 1e-12) & (norms_qc > 1e-12)
        if not np.any(valid):
            continue
        x_c  = x_c[valid]
        x_qc = x_qc[valid]

        rho_c  = (x_c[:,0,None,None]*iJx[None]
                + x_c[:,1,None,None]*iJy[None]
                + x_c[:,2,None,None]*iJz[None])   # (K, d, d)
        rho_qc = (x_qc[:,0,None,None]*iJx[None]
                + x_qc[:,1,None,None]*iJy[None]
                + x_qc[:,2,None,None]*iJz[None])

        # Per-vector M_n[s] = rho_c[s] @ rho_qc[s]^dag
        Mk = np.einsum('kab,kcb->kac', rho_c, rho_qc.conj())  # (K, d, d)
        for m in Mk:
            Mn_list.append(m.ravel())

    if len(Mn_list) < 2:
        return 0, np.array([])

    V = np.stack(Mn_list, axis=0)
    G = (V @ V.conj().T) / len(Mn_list)
    sv = np.linalg.svd(G, compute_uv=False)
    thresh = threshold * sv[0] if sv[0] > 0 else 0.
    return int(np.sum(sv > thresh)), sv


# ─── Main ────────────────────────────────────────────────────────────────────

def run(npz_path, spin=0.5, n_pairs_show=10, threshold=1e-10):
    import importlib, sys
    sys.path.insert(0, str(npz_path.parent.parent / 'spectral'))
    sys.path.insert(0, str(Path(__file__).parent))
    from spectral_O12 import (build_generators, bfs_shells,
                               fingerprint_vectors_batch)

    d_load = np.load(npz_path, allow_pickle=True)
    q    = int(d_load['q'])
    n0   = int(d_load['n0'])
    n1   = int(d_load['n1'])
    pairs = d_load['pairs']
    P    = len(pairs)
    j    = spin
    d_rho = int(round(2*j+1))
    limit = P if n_pairs_show == 0 else min(P, n_pairs_show)

    print(f"\n{'='*62}")
    print(f"O26 Test 4 (spectral H_eff)  q={q}  spin={j}  d_rho={d_rho}")
    print(f"  window=[{n0},{n1}]  {P} pairs")
    print(f"  Prediction: r_eff = {d_rho**2}")
    print(f"{'='*62}")

    # Build spectral basis
    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, 0.99)
    nodes  = [g for shell in shells for g in shell]
    N      = len(nodes)
    node_idx = {tuple(int(x) for x in g): i for i, g in enumerate(nodes)}
    print(f"\nBuilding Laplacian (q={q}, N={N})...")
    t0 = time.perf_counter()
    L, deg = build_laplacian(q, nodes, gens)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    print("Extracting ADE spectral basis...")
    t0 = time.perf_counter()
    B = get_spectral_basis(L, deg)  # (N, 3) complex orthonormal
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    if B is None:
        print("ERROR: could not build spectral basis"); return

    # For each pair: compute pi_spectral per window shell
    # by projecting fingerprint vectors onto B
    print(f"\nComputing r_eff for {limit} pairs...")
    print(f"  {'pair':>8}  {'r_eff':>6}  {'sigma_1':>12}  {'sigma_2':>12}")

    reff_all = []
    for i in range(P):
        c, qc_val = int(pairs[i, 0]), int(pairs[i, 1])
        cb_c  = np.array([c,     2, 3], dtype=np.int64)
        cb_qc = np.array([qc_val, 2, 3], dtype=np.int64)

        pi_c_mats   = []
        pi_qmc_mats = []

        for n, shell in enumerate(shells):
            if n < n0 or n > n1:
                continue
            if len(shell) == 0:
                break
            shell_arr = np.array(shell, dtype=np.int64)
            vecs_c  = fingerprint_vectors_batch(shell_arr, cb_c,
                                                np.array(gens), q)
            vecs_qc = fingerprint_vectors_batch(shell_arr, cb_qc,
                                                np.array(gens), q)
            # vecs_c: (|S_n|*64, q) complex
            # Embed each fingerprint vector into ell^2(G_q) and project onto B
            # Approximation: use 'a'-coordinate embedding
            # f_s(node_j) = vecs_c[s, nodes[node_j][0] % q]
            # pi(v_s) = B^dag @ f_s  -> (3,)

            # Vectorised: shape (N_fp, N) <- too large; use chunk approach
            # vecs_c[s, :] is in C^q; B is (N, 3)
            # f_s[j] = vecs_c[s, nodes[j][0] % q]
            # pi(v_s) = sum_j B[j,:].conj() * f_s[j]
            #         = sum_j B[j,:].conj() * vecs_c[s, a_j]
            # where a_j = nodes[j][0] % q

            a_coords = np.array([int(g[0]) % q for g in nodes], dtype=np.int64)
            # Shape: (N,), index into v

            # vecs_c: (K, q), B: (N, 3)
            # pi[s, :] = sum_j B[j,:].conj() * vecs_c[s, a_j]
            #          = (B.conj().T @ vecs_c[:, a_j].T).T  -- but need a_j indexing
            # = vecs_c[:, a_coords] @ B.conj()  <- (K, N) @ (N, 3) = (K, 3)
            # vecs_c[:, a_coords]: shape (K, N) -- too large for N=24k, K=~50k

            # Use smaller sample: take first 200 fingerprint vectors per shell
            K = min(200, len(vecs_c))
            vc_sub  = vecs_c[:K]   # (K, q)
            vqc_sub = vecs_qc[:K]

            # (K, N) @ (N, 3): each row is pi(v_s)
            # vc_sub[:, a_coords] has shape (K, N)
            pi_c_k  = vc_sub[:, a_coords]  @ B.conj()   # (K, 3)
            pi_qc_k = vqc_sub[:, a_coords] @ B.conj()

            pi_c_mats.append(pi_c_k)
            pi_qmc_mats.append(pi_qc_k)

        r_eff, sv = compute_reff_spectral(pi_c_mats, pi_qmc_mats, j, threshold)
        reff_all.append(r_eff)

        if i < limit:
            s1s2 = list(sv[:2]) + [0.]*max(0, 2-len(sv))
            mark = " *" if r_eff == d_rho**2 else "  "
            print(f"  ({c:2d},{qc_val:2d}){mark}  {r_eff:>6d}  "
                  + "  ".join(f"{s:12.4e}" for s in s1s2))

    reff_arr = np.array(reff_all)
    print(f"\n{'='*62}")
    print("SUMMARY")
    unique, counts = np.unique(reff_arr, return_counts=True)
    for u, cnt in zip(unique, counts):
        t = {d_rho**2: f"  <- PREDICTED"}.get(int(u), "")
        print(f"  r_eff = {int(u):2d} : {cnt}/{P} pairs{t}")
    print(f"{'='*62}\n")
    return reff_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--spin", type=float, default=0.5)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--threshold", type=float, default=1e-10)
    args = p.parse_args()
    run(args.npz, spin=args.spin, n_pairs_show=args.pairs,
        threshold=args.threshold)

if __name__ == "__main__":
    main()