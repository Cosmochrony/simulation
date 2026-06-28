"""
variety_summary.py
==================
Consolidated characterisation of the variety carrying the 6 -> 3 symmetric-square
suppression behind reff = 3 (O29 / O26 Test 4). Reproducible, emits a figure and
a summary npz. No new BFS; reads existing O25/Q5a checkpoints (pi_c).

Pipeline per prime (pooled over pairs):
  - monomial matrix Phi (N x 6), hard rank of the squaring (Veronese) image;
  - the 3 constraint forms B_k (null space): trace, rank, spectrum;
  - commutators L_k = [B_i, B_j]: antisymmetry and approximate so(3) closure;
  - reality/polarisation: rho = |w^T w| / |w|^2 (fixed: real-valued median).
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HEFF = 3

def pooled(path, pmax=12):
    d = np.load(path, allow_pickle=True); pi = d["pi_c"]; nwin = pi.shape[1]; out = []
    for p in range(min(pmax, pi.shape[0])):
        acc = []
        for k in range(nwin):
            c = pi[p, k]
            if c is None: continue
            c = np.asarray(c, complex)
            if c.ndim == 1: c = c[None, :]
            if c.shape[-1] == HEFF and c.shape[0] > 0: acc.append(c)
        if acc: out.append(np.concatenate(acc, 0))
    return int(d["q"]), np.concatenate(out, 0)

def b2B(b):
    B = np.zeros((3, 3), complex); B[0,0],B[1,1],B[2,2] = b[0],b[1],b[2]
    B[0,1]=B[1,0]=b[3]/2; B[0,2]=B[2,0]=b[4]/2; B[1,2]=B[2,1]=b[5]/2; return B

def analyse(W):
    w0,w1,w2 = W[:,0],W[:,1],W[:,2]
    Phi = np.stack([w0*w0,w1*w1,w2*w2,w0*w1,w0*w2,w1*w2], 1)
    _, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    sn = s/s[0]
    Bs = [b2B(b) for b in Vt[3:].conj()]
    traces = [complex(np.trace(B)) for B in Bs]
    Ls = [Bs[i]@Bs[j]-Bs[j]@Bs[i] for (i,j) in [(1,2),(2,0),(0,1)]]
    antisym = [float(np.linalg.norm(L+L.T)/(np.linalg.norm(L)+1e-30)) for L in Ls]
    Ln = [L/np.linalg.norm(L) for L in Ls]
    closure = []
    for (i,j,k) in [(0,1,2),(1,2,0),(2,0,1)]:
        C = Ln[i]@Ln[j]-Ln[j]@Ln[i]
        coef = np.vdot(Ln[k],C)/np.vdot(Ln[k],Ln[k])
        closure.append(float(np.linalg.norm(C-coef*Ln[k])/(np.linalg.norm(C)+1e-30)))
    wTw = np.einsum('ni,ni->n', W, W); wdw = np.einsum('ni,ni->n', W.conj(), W).real
    rho = np.abs(wTw)/wdw
    return dict(mono=sn, traces=traces, antisym=antisym, closure=closure,
                rho_med=float(np.median(rho)), N=len(W))

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "../o26/o25_outputs"
    res = {}
    for q in [29, 61]:
        p = os.path.join(base, f"q{q}_o25.npz")
        if not os.path.isfile(p): continue
        qv, W = pooled(p); res[qv] = analyse(W)
        r = res[qv]
        print(f"\nq={qv} N={r['N']}")
        print(f"  monomial singvals (Veronese 6->3): {np.round(r['mono'],4)}")
        print(f"  constraint forms traceless: |trace|={[round(abs(t),3) for t in r['traces']]}")
        print(f"  commutator antisymmetry resid: {[f'{a:.1e}' for a in r['antisym']]}  (0=>so(3)-valued)")
        print(f"  so(3) closure off-axis resid: {[round(c,3) for c in r['closure']]}  (approx, not exact)")
        print(f"  median |w^Tw|/|w|^2 = {r['rho_med']:.3f}")
    # figure: monomial spectrum (hard 6->3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for q in res:
        ax[0].plot(range(1,7), res[q]['mono'], 'o-', label=f"q={q}")
    ax[0].axhline(1e-12, ls=':', c='r'); ax[0].set_yscale('symlog', linthresh=1e-12)
    ax[0].set_xlabel("monomial index"); ax[0].set_ylabel(r"$\sigma_i/\sigma_1$")
    ax[0].set_title("(a) Veronese squaring image: hard 6$\\to$3 collapse")
    ax[0].legend(); ax[0].grid(alpha=.3)
    qs = sorted(res); x = np.arange(len(qs))
    clo = [np.mean(res[q]['closure']) for q in qs]
    ant = [np.mean(res[q]['antisym']) for q in qs]
    ax[1].bar(x-0.2, ant, 0.4, label="commutator antisym resid")
    ax[1].bar(x+0.2, clo, 0.4, label="so(3) closure off-axis resid")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"q={q}" for q in qs])
    ax[1].set_title("(b) constraint forms: so(3)-valued but not exactly closing")
    ax[1].legend(); ax[1].grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig("variety_summary.pdf", dpi=160, bbox_inches="tight")
    print("\nsaved variety_summary.pdf")
    np.savez("variety_summary.npz", summary=np.array([res], dtype=object))

if __name__ == "__main__":
    main()
