"""
variety_characterization.py
===========================
Characterise the variety on which the H_eff projections w_j in C^3 live,
i.e. explain the 6 -> 3 suppression of the symmetric squares span{w (x) w}
that produces reff = 3 in End(H_eff).

reff = 3 <=> dim span{w (x) w} = 3 <=> there exist exactly 6-3 = 3 complex
symmetric forms B_k (3x3) with w_j^T B_k w_j = 0 for all j: the w_j lie on the
intersection of 3 conics in P^2. This script extracts and identifies them.

Diagnostics (per pair and pooled), q in {29, 61}:
  (1a) full singular spectrum of the monomial matrix Phi (N x 6); hard rank?
  (1b) the 3 null forms B_k: ranks, pairwise commutators, simultaneous structure;
       common base locus / common kernel direction.
  (1c) reality structure rho_j = |w_j^T w_j| / (w_j^dag w_j)  (1 => phase x real,
       0 => isotropic w^Tw=0); projective cloud dimension via PCA on normalised pts.
"""
import numpy as np, sys, os

HEFF = 3

def traj(path, pmax=None):
    d = np.load(path, allow_pickle=True)
    pi_c, pairs = d["pi_c"], d["pairs"]
    nwin = pi_c.shape[1]
    out = []
    P = pi_c.shape[0] if pmax is None else min(pmax, pi_c.shape[0])
    for p in range(P):
        acc = []
        for k in range(nwin):
            c = pi_c[p, k]
            if c is None: continue
            c = np.asarray(c, complex)
            if c.ndim == 1: c = c[None, :]
            if c.shape[-1] == HEFF and c.shape[0] > 0: acc.append(c)
        if acc:
            W = np.concatenate(acc, 0)
            if not np.allclose(W, 0):
                out.append((int(pairs[p,0]), int(pairs[p,1]), W))
    return int(d["q"]), out

def monomials(W):
    """N x 6 : [w0^2, w1^2, w2^2, w0w1, w0w2, w1w2]."""
    w0, w1, w2 = W[:,0], W[:,1], W[:,2]
    return np.stack([w0*w0, w1*w1, w2*w2, w0*w1, w0*w2, w1*w2], axis=1)

def b_to_B(b):
    """Recover symmetric 3x3 from monomial-coeff vector (w^T B w = b . m)."""
    B = np.zeros((3,3), complex)
    B[0,0], B[1,1], B[2,2] = b[0], b[1], b[2]
    B[0,1]=B[1,0]=b[3]/2; B[0,2]=B[2,0]=b[4]/2; B[1,2]=B[2,1]=b[5]/2
    return B

def analyze(W, label):
    N = len(W)
    # linear rank
    sL = np.linalg.svd(W.T, compute_uv=False)
    rankL = int(np.sum(sL > 1e-2*sL[0]))
    # quadratic: monomial matrix
    Phi = monomials(W)
    # normalise columns by overall scale via per-row norm to avoid magnitude bias
    sQ = np.linalg.svd(Phi, compute_uv=False)
    sQn = sQ / sQ[0]
    rankQ = int(np.sum(sQn > 1e-2))
    # null space (right singular vectors of Phi with smallest sv)
    _,_,Vt = np.linalg.svd(Phi, full_matrices=False)   # Vt: (6,6)
    null = Vt[rankQ:].conj()      # (6-rankQ, 6) rows = b vectors
    Bs = [b_to_B(b) for b in null]
    # reality structure
    wTw = np.einsum('ni,ni->n', W, W)          # bilinear w^T w
    wdw = np.einsum('ni,ni->n', np.conj(W), W) # w^dag w
    rho = np.abs(wTw)/(wdw+1e-30)
    # projective cloud dimension: normalise (unit norm + fix phase of dominant comp), PCA real
    idx = np.argmax(np.abs(W).mean(0))
    ph = np.exp(-1j*np.angle(W[:,idx]+1e-30))
    Wn = (W*ph[:,None]); Wn = Wn/ (np.linalg.norm(Wn,axis=1,keepdims=True)+1e-30)
    real_emb = np.concatenate([Wn.real, Wn.imag], axis=1)  # N x 6
    real_emb = real_emb - real_emb.mean(0)
    sP = np.linalg.svd(real_emb, compute_uv=False)
    sPn = sP/sP[0]
    proj_dim = int(np.sum(sPn > 1e-2))
    print(f"\n----- {label}  (N={N}) -----")
    print(f"  linear rank(W)={rankL}   singvals {np.round(sL/sL[0],3)}")
    print(f"  quad rank(Phi)={rankQ}   monomial singvals {np.round(sQn,4)}")
    print(f"  reality rho=|wTw|/|w|^2 : median={np.median(rho):.3f} mean={np.mean(rho):.3f} "
          f"(1=phase x real, 0=isotropic)")
    print(f"  proj-cloud real PCA dim={proj_dim}  singvals {np.round(sPn,3)}")
    for i,B in enumerate(Bs):
        ev = np.linalg.eigvalsh((B.conj().T@B))  # singular^2
        sv = np.sqrt(np.maximum(ev,0))[::-1]
        rk = int(np.sum(sv>1e-6*sv[0]))
        # eigen-structure of symmetric B (complex symmetric: use Takagi-ish via |.|)
        evB = np.linalg.eigvals(B)
        print(f"    B_{i}: rank={rk}  |sv|={np.round(sv/ (sv[0]+1e-30),3)}  "
              f"eig(B)={np.round(evB,3)}")
    # common kernel of the B_k (shared base direction)?
    if Bs:
        stacked = np.vstack([B for B in Bs])      # (3*nnull, 3)
        sk = np.linalg.svd(stacked, compute_uv=False)
        print(f"  common-kernel test: singvals of stacked B = {np.round(sk/sk[0],3)} "
              f"(small last => shared kernel direction)")
    return dict(rankL=rankL, rankQ=rankQ, rho_med=float(np.median(rho)),
                proj_dim=proj_dim, Bs=Bs, rho=rho)

def main():
    base = sys.argv[1] if len(sys.argv)>1 else "../o26/o25_outputs"
    for q in [29, 61]:
        path = os.path.join(base, f"q{q}_o25.npz")
        if not os.path.isfile(path):
            print(f"[skip] {path}"); continue
        qv, tr = traj(path)
        print(f"\n{'='*72}\nq = {qv}   {len(tr)} pairs\n{'='*72}")
        # a few pairs
        for (c,qc,W) in tr[:3]:
            analyze(W, f"q{qv} pair ({c},{qc})")
        # pooled
        Wall = np.concatenate([W for (_,_,W) in tr], 0)
        analyze(Wall, f"q{qv} POOLED all pairs")

if __name__ == "__main__":
    main()
