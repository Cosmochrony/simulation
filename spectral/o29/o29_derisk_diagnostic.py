"""
o29_derisk_diagnostic.py
========================
De-risking battery for O29 / O26 Test 4 (Criterion 5.4).

Runs three diagnostics on the existing O25/Q5a checkpoints (pi_c / pi_qmc),
WITHOUT any new BFS:

  D1  Full eigenvalue spectra of C_c in End(H_eff)=C^9 and End(V_rho)=C^4,
      per pair. Settles whether reff=3 is clean (gap after lambda_3) and
      whether reff in End(V_rho) is 2 or 3 (O29 eq. 3.18 claims 3).

  D2  Parity defect asym(q) = median |M_j - M_j^T|_F/|M_j|_F.
      Tests whether the anti-linear parity (hence M_j=w w^T symmetric)
      is exact-in-the-limit (asym->0) or only approximate (plateau).

  D3  Mechanism isolation. For each pair, reff in End(V_rho) under
        (i)   real conjugate data         M_j = w_c_j (x) conj(w_qc_j)
        (ii)  exact parity imposed        M_j = w_c_j (x) w_c_j   (= w w^T)
        (iii) per-vector locking broken   M_j = w_c_j (x) conj(w_qc_sigma(j))
              (sigma = random permutation; same distribution, independent pairing)
      If (i),(ii) give <=3 and (iii) gives 4, the d_rho^2=4 target is
      inaccessible from conjugate-pair data BY CONSTRUCTION (O29 Def. 4.4).

Usage:
  python o29_derisk_diagnostic.py --dir <checkpoint_dir> --primes 29 61 151
"""
import argparse, os, sys
import numpy as np

HEFF = 3
RNG  = np.random.default_rng(0)


def load_pair_trajectories(path, window_only=True):
    """Return list of dicts {c, qmc, Wc (N,3), Wqc (N,3)} from a checkpoint."""
    d = np.load(path, allow_pickle=True)
    if "pi_c" not in d:
        return None, None
    pi_c, pi_qmc = d["pi_c"], d["pi_qmc"]
    pairs = d["pairs"]
    q = int(d["q"]); n0 = int(d["n0"]); n1 = int(d["n1"]); ns = d["ns"]
    # window shell indices present as columns of pi_c (already restricted to window)
    nwin = pi_c.shape[1]
    out = []
    for p in range(pi_c.shape[0]):
        allc, allq = [], []
        for k in range(nwin):
            cc = pi_c[p, k]; qq = pi_qmc[p, k]
            if cc is None or qq is None:
                continue
            cc = np.asarray(cc, complex); qq = np.asarray(qq, complex)
            if cc.ndim == 1: cc = cc[None, :]
            if qq.ndim == 1: qq = qq[None, :]
            if cc.shape[-1] != HEFF or cc.shape[0] == 0:
                continue
            allc.append(cc); allq.append(qq)
        if not allc:
            continue
        Wc = np.concatenate(allc, 0); Wqc = np.concatenate(allq, 0)
        if np.allclose(Wc, 0):
            continue
        out.append(dict(c=int(pairs[p, 0]), qmc=int(pairs[p, 1]), Wc=Wc, Wqc=Wqc))
    return q, out


def cov_eigs(U, V):
    """Eigenvalues (desc) of C = (1/N) sum vec(M_j) vec(M_j)^dag,
    M_j = U_j (x) conj(V_j).  U,V shape (N,d)."""
    N = min(len(U), len(V)); U = U[:N]; V = V[:N]; d = U.shape[1]
    Mv = (U[:, :, None] * np.conj(V[:, None, :])).reshape(N, d * d)
    C = (Mv.conj().T @ Mv) / N
    ev = np.sort(np.maximum(np.linalg.eigvalsh(C).real, 0))[::-1]
    return ev


def reff(ev, rel=1e-2):
    return int(np.sum(ev > rel * ev[0])) if ev[0] > 0 else 0


def asym_defect(Wc, Wqc):
    N = min(len(Wc), len(Wqc)); wc = Wc[:N]; wq = Wqc[:N]
    M = wc[:, :, None] * np.conj(wq[:, None, :])
    a = np.linalg.norm((M - M.transpose(0, 2, 1)).reshape(N, -1), axis=1)
    n = np.linalg.norm(M.reshape(N, -1), axis=1) + 1e-30
    return float(np.median(a / n))


def vrho_project(Wc, Wqc, dim=2):
    """Top-`dim` right singular directions of Wc; project both."""
    N = min(len(Wc), len(Wqc))
    Uc, s, _ = np.linalg.svd(Wc[:N].T, full_matrices=False)  # Wc.T : (3,N)
    P = Uc[:, :dim]
    return Wc[:N] @ P.conj(), Wqc[:N] @ P.conj(), s


def run(path):
    q, traj = load_pair_trajectories(path)
    if not traj:
        return None
    rows = []
    for tr in traj:
        Wc, Wqc = tr["Wc"], tr["Wqc"]
        # --- D1: full spectra
        evH = cov_eigs(Wc, Wqc)                          # End(H_eff) C^9
        wc2, wq2, sv = vrho_project(Wc, Wqc, 2)
        evV = cov_eigs(wc2, wq2)                         # End(V_rho) C^4
        # --- D2
        asy = asym_defect(Wc, Wqc)
        # --- D3 variants in End(V_rho)
        ev_real = evV
        ev_par  = cov_eigs(wc2, wc2.conj())              # exact parity w (x) w
        sig = RNG.permutation(len(wq2))
        ev_brk  = cov_eigs(wc2, wq2[sig])                # locking broken
        rows.append(dict(
            c=tr["c"], qmc=tr["qmc"],
            evH=evH, evV=evV, asy=asy, sv=sv,
            rH=reff(evH), rV=reff(evV),
            r_real=reff(ev_real), r_par=reff(ev_par), r_brk=reff(ev_brk),
            ev_brk=ev_brk,
        ))
    return q, rows


def summarize(q, rows):
    rH = np.array([r["rH"] for r in rows])
    rV = np.array([r["rV"] for r in rows])
    asy = np.array([r["asy"] for r in rows])
    sv = np.array([r["sv"] for r in rows])            # (P,3) trajectory singular values
    # median normalised spectra
    def med_norm(key, m):
        A = np.zeros((len(rows), m))
        for i, r in enumerate(rows):
            ev = r[key]; n = min(m, len(ev)); A[i, :n] = ev[:n] / (ev[0] + 1e-30)
        return np.median(A, 0)
    sH = med_norm("evH", 9)
    sV = med_norm("evV", 4)
    sv_ratio = np.median(sv[:, 2] / sv[:, 0])         # sigma3/sigma1 of trajectory
    sv_ratio12 = np.median(sv[:, 1] / sv[:, 0])
    print(f"\n{'='*70}\n q = {q}   ({len(rows)} pairs)\n{'='*70}")
    print(f" D1  reff(H_eff): modes {dict(zip(*np.unique(rH, return_counts=True)))}")
    print(f" D1  reff(V_rho): modes {dict(zip(*np.unique(rV, return_counts=True)))}")
    print(f" D1  median norm spectrum End(H_eff): " +
          " ".join(f"{x:.3f}" for x in sH[:5]))
    print(f" D1  median norm spectrum End(V_rho): " +
          " ".join(f"{x:.3f}" for x in sV))
    print(f" D1  trajectory singular ratios  sigma2/sigma1={sv_ratio12:.3f}  "
          f"sigma3/sigma1={sv_ratio:.2e}   (dim_C V_rho test)")
    print(f" D2  parity defect asym  median={np.median(asy):.3e}  "
          f"mean={np.mean(asy):.3e}")
    r_real = np.array([r["r_real"] for r in rows])
    r_par  = np.array([r["r_par"]  for r in rows])
    r_brk  = np.array([r["r_brk"]  for r in rows])
    print(f" D3  reff(V_rho)  real-conjugate : {dict(zip(*np.unique(r_real,return_counts=True)))}")
    print(f" D3  reff(V_rho)  exact-parity   : {dict(zip(*np.unique(r_par, return_counts=True)))}")
    print(f" D3  reff(V_rho)  locking-broken : {dict(zip(*np.unique(r_brk, return_counts=True)))}")
    return dict(q=q, asym_med=float(np.median(asy)),
                rH=rH.tolist(), rV=rV.tolist(),
                sH=sH.tolist(), sV=sV.tolist(),
                sv3_1=float(sv_ratio), sv2_1=float(sv_ratio12),
                r_real=r_real.tolist(), r_par=r_par.tolist(), r_brk=r_brk.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--primes", nargs="+", type=int, default=[29, 61])
    a = ap.parse_args()
    summ = []
    for q in a.primes:
        path = os.path.join(a.dir, f"q{q}_o25.npz")
        if not os.path.isfile(path):
            path = os.path.join(a.dir, f"q{q}_o25_partial.npz")
        if not os.path.isfile(path):
            print(f"[skip] no checkpoint for q={q}"); continue
        res = run(path)
        if res is None:
            print(f"[skip] no pi_c in q={q}"); continue
        q_, rows = res
        summ.append(summarize(q_, rows))
    # D2 scaling table
    if len(summ) >= 2:
        print(f"\n{'='*70}\n D2  PARITY-DEFECT SCALING\n{'='*70}")
        print(f"   {'q':>6}  {'asym_median':>12}")
        for s in summ:
            print(f"   {s['q']:>6}  {s['asym_med']:>12.3e}")
    np.savez(os.path.join(a.dir, "o29_derisk_summary.npz"),
             summary=np.array(summ, dtype=object))


if __name__ == "__main__":
    main()
