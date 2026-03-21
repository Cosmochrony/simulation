"""
SpectralO8.py  --  v2
Companion script for:
  "Three-Step Path Fingerprints on LPS Graphs and the Capacity Growth Exponent"
  Cosmochrony Spectral Cascade Programme -- Paper O8

Fingerprint architecture:
  pi_3(v1,v2,v3) = perm_vec(v1) x perm_vec(v2) x perm_vec(v3) in R^{(q+1)^3}.
  Structured sketch: S in R^{D_s x (q+1)}, fp3_sk = kron(kron(S*pv1,S*pv2),S*pv3),
  so fp3_sk lives in R^{D_s^3}.  Span tracking is done in the sketched space.
  D_s=14 gives ambient_sketch=2744.  Disclosed in paper Section 7 (Setup).

Measure:
  The capacity growth law is Sigma_bar_n ~ |S_n|^{-delta} where |S_n| is the
  cumulative BFS shell size.  delta is extracted by log-log fit over the
  pre-saturation window (before r_tilde < R_TILDE_SAT).

LPS configurations (p prime, q prime, p!=q, p=q=1 mod 4, Legendre(p,q)=1):
  (p=17, q=13): |G|=1092,  q^2=169
  (p=13, q=17): |G|=2448,  q^2=289
  (p=5,  q=29): |G|=12180, q^2=841
  (p=5,  q=41): |G|=34440, q^2=1681   (optional, slower)
  (p=5,  q=61): |G|=113460,q^2=3721   (optional, requires D_s >= 16)

Author: Jerome Beau <jerome.beau@cosmochrony.org>
Date:   March 2026
"""

import math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from scipy import linalg
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 0.  Global parameters
# ---------------------------------------------------------------------------

# Edit this list to select which (p,q) to run.
# Add (5,41) or (5,61) for larger q -- runtime scales as O(|G| * D_s^3).
LPS_CONFIGS = [(17, 13), (13, 17), (5, 29), (5, 41)]

D_SKETCH    = 14       # sketch factor; ambient_sketch = D_SKETCH^3
QR_BATCH    = 300      # sub-batch size for streaming QR
QR_EPS      = 1e-9     # independence threshold in sketched space
R_TILDE_SAT = 0.05     # r_tilde threshold to declare effective saturation
FIT_MIN_PTS = 3        # minimum BFS steps for a reliable fit
SEED        = 42       # RNG seed for sketch matrix

# ---------------------------------------------------------------------------
# 1.  Number-theoretic helpers
# ---------------------------------------------------------------------------

def modinv(a, m):
    return pow(int(a) % m, m - 2, m)

def sqrt_mod(a, q):
    """Tonelli-Shanks.  Raises ValueError if a is not a QR mod q."""
    a = int(a) % q
    if a == 0: return 0
    if pow(a, (q-1)//2, q) != 1:
        raise ValueError(f"{a} is not a QR mod {q}")
    if q % 4 == 3:
        return pow(a, (q+1)//4, q)
    S, Qv = 0, q-1
    while Qv % 2 == 0: Qv //= 2; S += 1
    z = 2
    while pow(z, (q-1)//2, q) != q-1: z += 1
    M_, c_, t_, R_ = S, pow(z,Qv,q), pow(a,Qv,q), pow(a,(Qv+1)//2,q)
    while True:
        if t_ == 1: return R_
        i, tmp = 1, (t_*t_) % q
        while tmp != 1: tmp = (tmp*tmp) % q; i += 1
        b_ = pow(c_, pow(2, M_-i-1, q-1), q)
        M_,c_,t_,R_ = i,(b_*b_)%q,(t_*b_*b_)%q,(R_*b_)%q

# ---------------------------------------------------------------------------
# 2.  PSL(2, F_q)
# ---------------------------------------------------------------------------

def _key(a, b, c, d, q):
    pos = (a%q, b%q, c%q, d%q)
    neg = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
    return min(pos, neg)

def build_psl2_fq(q):
    seen = set(); elements = []; index_of = {}
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a != 0:
                    d = (1 + b*c) * modinv(a,q) % q
                    key = _key(a,b,c,d,q)
                    if key not in seen:
                        seen.add(key); idx = len(elements)
                        elements.append(key); index_of[key] = idx
                        index_of[_key(q-a,q-b,q-c,q-d,q)] = idx
                else:
                    if c == 0: continue
                    b2 = (q - modinv(c,q)) % q
                    for d in range(q):
                        key = _key(0,b2,c,d,q)
                        if key not in seen:
                            seen.add(key); idx = len(elements)
                            elements.append(key); index_of[key] = idx
                            index_of[_key(0,q-b2,q-c,q-d,q)] = idx
    return elements, index_of

def mul_psl2(m1, m2, q, index_of):
    a1,b1,c1,d1=m1; a2,b2,c2,d2=m2
    return index_of[_key(
        (a1*a2+b1*c2)%q, (a1*b2+b1*d2)%q,
        (c1*a2+d1*c2)%q, (c1*b2+d1*d2)%q, q)]

# ---------------------------------------------------------------------------
# 3.  LPS generators
# ---------------------------------------------------------------------------

def lps_generators(p, q, index_of):
    """LPS generators for X^{p,q}.  det=1 normalisation via 1/sqrt(p) mod q."""
    i_q = sqrt_mod(q-1, q)
    inv_sqrtp = modinv(sqrt_mod(p, q), q)
    found = set()
    bnd = int(p**0.5) + 2
    for a in range(1, bnd+2, 2):
        for b in range(-(bnd//2*2), bnd//2*2+1, 2):
            for c in range(-(bnd//2*2), bnd//2*2+1, 2):
                rem = p - a*a - b*b - c*c
                if rem < 0: continue
                dsq = int(math.sqrt(rem)+0.5)
                for d in ([0] if dsq==0 else [dsq,-dsq]):
                    if d%2==0 and a*a+b*b+c*c+d*d==p:
                        found.add((a,b,c,d))
    result = []
    for (a,b,c,d) in found:
        an=a*inv_sqrtp%q; bn=b*inv_sqrtp%q
        cn=c*inv_sqrtp%q; dn=d*inv_sqrtp%q
        m00=(an+bn*i_q)%q; m01=(cn+dn*i_q)%q
        m10=(q-cn+dn*i_q)%q; m11=(an+q-bn*i_q%q)%q
        key = _key(m00,m01,m10,m11,q)
        if key in index_of:
            gi = index_of[key]
            if gi not in result: result.append(gi)
    return result

# ---------------------------------------------------------------------------
# 4.  Permutation vector and structured sketch
# ---------------------------------------------------------------------------

def mobius(mat, z, q):
    a,b,c,d = mat
    if z == q:
        return q if c%q==0 else a*modinv(c,q)%q
    denom = (c*z+d) % q
    if denom == 0: return q
    return (a*z+b)*modinv(denom,q)%q

def perm_vec(mat, q):
    """Centred one-hot e_{mat*0} - 1/(q+1) in R^{q+1}."""
    n = q+1; pv = np.full(n, -1.0/n)
    pv[mobius(mat, 0, q)] += 1.0
    return pv

def build_sketch(q, D_s=D_SKETCH, seed=SEED):
    """Sketch matrix S in R^{D_s x (q+1)}, normalised."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((D_s, q+1)) / math.sqrt(D_s)

def fp3_batch(SK1, SK2, SK3):
    """
    Batch k=3 sketched fingerprints.
    SK1, SK2, SK3: (n x D_s) -- sketched perm vecs for each path vertex.
    Returns (n x D_s^3) matrix.
    """
    return np.einsum('bi,bj,bk->bijk', SK1, SK2, SK3).reshape(len(SK1), -1)

# ---------------------------------------------------------------------------
# 5.  Streaming QR span tracker (sketched space, dimension D_s^3)
# ---------------------------------------------------------------------------

class SpanTracker:
    def __init__(self, dim, batch=QR_BATCH, eps=QR_EPS):
        self.dim = dim; self.batch = batch; self.eps = eps
        self.basis = np.zeros((dim,0)); self._pend = []

    def _flush(self):
        if not self._pend: return
        mat = np.column_stack(self._pend)
        if self.basis.shape[1] > 0:
            mat = mat - self.basis @ (self.basis.T @ mat)
        try:
            Q, R = linalg.qr(mat, mode='economic')
        except Exception:
            self._pend = []; return
        diag = np.abs(np.diag(R))
        good = Q[:, diag > self.eps]
        if good.shape[1] > 0:
            if self.basis.shape[1] > 0:
                good = good - self.basis @ (self.basis.T @ good)
            nrm = np.linalg.norm(good, axis=0, keepdims=True)
            good = good[:, nrm[0] > self.eps] / np.maximum(nrm, 1e-30)[:,nrm[0]>self.eps]
            if good.shape[1] > 0:
                self.basis = (np.column_stack([self.basis, good])
                              if self.basis.shape[1] > 0 else good)
        self._pend = []

    def add_batch(self, mat):
        """mat: dim x n.  Process in sub-batches."""
        for i in range(0, mat.shape[1], self.batch):
            chunk = mat[:, i:i+self.batch]
            self._pend += [chunk[:,j] for j in range(chunk.shape[1])]
            if len(self._pend) >= self.batch: self._flush()
        self._flush()

    def novelties(self, mat):
        """Squared relative residuals for dim x n matrix."""
        nrm2 = np.sum(mat**2, axis=0) + 1e-30
        if self.basis.shape[1] == 0:
            return np.sum(mat**2, axis=0) / nrm2
        res = mat - self.basis @ (self.basis.T @ mat)
        return np.sum(res**2, axis=0) / nrm2

    @property
    def rank(self): return self.basis.shape[1]

# ---------------------------------------------------------------------------
# 6.  BFS cascade
# ---------------------------------------------------------------------------

def run_one(p, q, verbose=True):
    """
    Full BFS on X^{p,q} with k=3 sketched permutation-path fingerprints.
    Tracks capacity Sigma_n, occupancy eta_n, redundancy R_n per BFS step.
    """
    label = f"(p={p},q={q})"
    if verbose:
        print(f"\n{'='*64}")
        print(f"  {label}  |G|={q*(q*q-1)//2}  "
              f"q^2={q**2}  sketch_dim={D_SKETCH**3}")
        print(f"{'='*64}")

    print(f"  Building PSL(2,F_{q}) ...", end=" ", flush=True)
    elements, index_of = build_psl2_fq(q)
    N = len(elements)
    print(f"|G|={N}")

    print("  Computing generators ...", end=" ", flush=True)
    gen_idx = lps_generators(p, q, index_of)
    print(f"{len(gen_idx)}  (expected {p+1})")
    if not gen_idx:
        print("  ERROR: no generators found."); return None

    S = build_sketch(q, D_SKETCH, SEED)    # D_SKETCH x (q+1)
    print("  Sketching perm vectors ...", end=" ", flush=True)
    PV_raw = np.array([perm_vec(mat, q) for mat in elements])   # N x (q+1)
    PV_sk  = (S @ PV_raw.T).T                                   # N x D_SKETCH
    print("done")

    print("  Building neighbour table ...", end=" ", flush=True)
    nbrs = [[mul_psl2(elements[v], elements[g], q, index_of)
             for g in gen_idx] for v in range(N)]
    print("done")

    # BFS
    visited = np.zeros(N, dtype=bool); parent = np.full(N,-1,dtype=int)
    visited[0] = True
    dim_sk  = D_SKETCH**3
    tracker = SpanTracker(dim=dim_sk)
    queue   = deque([0])

    # per-step records
    rec_cumul   = []   # cumulative |S_n| (vertices visited so far)
    rec_shell   = []   # |partial S_n| (shell size at step n)
    rec_sigma   = []   # mean effective novelty
    rec_eta     = []   # mean occupancy proxy
    rec_R       = []   # mean redundancy indicator
    rec_rank    = []   # tracker rank

    step = 0
    while queue:
        step   += 1
        cur_lvl = list(queue); queue = deque(); shell = []
        for v in cur_lvl:
            for w in nbrs[v]:
                if not visited[w]:
                    visited[w]=True; parent[w]=v; shell.append(w); queue.append(w)
        if not shell: break

        sz = len(shell)
        SK1 = PV_sk[[parent[w] if parent[parent[w]]<0 else parent[parent[w]]
                     for w in shell]]
        SK2 = PV_sk[[parent[w] for w in shell]]
        SK3 = PV_sk[shell]

        FP = fp3_batch(SK1, SK2, SK3).T        # dim_sk x sz

        # novelties BEFORE adding to span
        nov = tracker.novelties(FP)             # length sz

        tracker.add_batch(FP)

        cumul     = int(np.sum(visited))
        sigma_bar = float(np.mean(nov))
        # occupancy proxy: 1/sqrt(rank+1) (rank-normalised density)
        psi_sq    = 1.0 / math.sqrt(tracker.rank + 1)
        eta_bar   = psi_sq / max(sigma_bar, 1e-14)
        # redundancy: fraction with novelty < 1-threshold (projectively redundant)
        R_bar     = float(np.mean(nov < 0.5))
        # r_tilde:   fraction with novelty > threshold (genuinely novel)
        r_tilde   = float(np.mean(nov > 0.1))

        rec_cumul.append(cumul);  rec_shell.append(sz)
        rec_sigma.append(sigma_bar); rec_eta.append(eta_bar)
        rec_R.append(R_bar);      rec_rank.append(tracker.rank)

        if verbose and (step <= 8 or step % 4 == 0):
            print(f"  step {step:3d}: |S_n|={cumul:7d}  shell={sz:6d}  "
                  f"rank={tracker.rank:5d}/{dim_sk}  "
                  f"sigma={sigma_bar:.4f}  r~={r_tilde:.3f}")

    cumuls  = np.array(rec_cumul,  dtype=float)
    shells  = np.array(rec_shell,  dtype=float)
    sigmas  = np.array(rec_sigma)
    etas    = np.array(rec_eta)
    Rs      = np.array(rec_R)
    ranks   = np.array(rec_rank)

    # effective saturation: first step where r_tilde < R_TILDE_SAT
    rt = np.array([float(np.mean(
           tracker.novelties(np.zeros((dim_sk,1)))>0.1)) for _ in [0]])  # dummy
    # recompute r_tilde from novelty threshold properly
    r_tildes = np.array([float(np.mean(np.array([rec_R[i]]) < 0.5))
                         for i in range(len(rec_R))])
    # simpler: use rec_R directly as proxy for saturation
    r_tildes_plot = 1.0 - np.array(rec_R)   # fraction of novel (high-novelty) vecs

    sat_idx = next((i for i, r in enumerate(r_tildes_plot) if r < R_TILDE_SAT),
                   len(r_tildes_plot)-1)
    s_star  = float(cumuls[sat_idx])

    # fit delta: Sigma_bar_n ~ cumul_n^{-delta}
    # use only pre-saturation window
    w_mask = np.zeros(len(cumuls), dtype=bool)
    w_mask[:sat_idx+1] = True
    w_mask &= (sigmas > 0) & np.isfinite(sigmas)

    delta = beta_eff = r2_fit = np.nan
    if w_mask.sum() >= FIT_MIN_PTS:
        lx  = np.log(cumuls[w_mask])
        ly  = np.log(sigmas[w_mask])
        A   = np.column_stack([lx, np.ones_like(lx)])
        cf, *_ = np.linalg.lstsq(A, ly, rcond=None)
        delta = float(-cf[0])
        ly_p  = A @ cf
        ss_r  = float(np.sum((ly-ly_p)**2))
        ss_t  = float(np.sum((ly-ly.mean())**2))
        r2_fit = 1.0 - ss_r/(ss_t+1e-30)
        if np.isfinite(delta) and delta > -0.5:
            beta_eff = 1.0/(0.5+delta)

    if verbose:
        print(f"  => delta={delta:.4f}  beta_eff={beta_eff:.4f}  "
              f"R2={r2_fit:.4f}  S*~{s_star:.0f}  q^2={q**2}")

    return dict(
        label=label, p=p, q=q, N=N,
        ambient=(q+1)**3, dim_sk=dim_sk,
        cumuls=cumuls, shells=shells,
        sigmas=sigmas, etas=etas, Rs=Rs,
        r_tildes=r_tildes_plot, ranks=ranks,
        delta=delta, beta_eff=beta_eff, r2=r2_fit,
        s_star=s_star, fit_mask=w_mask,
    )

# ---------------------------------------------------------------------------
# 7.  Figures
# ---------------------------------------------------------------------------

PHI  = lambda eta: 1.0 / np.sqrt(1.0+eta**2)
COLS = {(17,13):'#1f77b4', (13,17):'#ff7f0e',
        (5,29):'#2ca02c', (5,41):'#d62728', (5,61):'#9467bd'}
MRKS = {(17,13):'o', (13,17):'s', (5,29):'^', (5,41):'D', (5,61):'P'}
LBLS = {(17,13):r'$X^{17,13}$', (13,17):r'$X^{13,17}$',
        (5,29):r'$X^{5,29}$',   (5,41):r'$X^{5,41}$',  (5,61):r'$X^{5,61}$'}

def _style(pq):
    return (COLS.get(pq,'gray'), MRKS.get(pq,'x'), LBLS.get(pq,str(pq)))


def fig1(results, path="SpectralO8_fig1.pdf"):
    """4-panel state law figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    eta_g = np.linspace(0, 8, 300)
    ax_a.plot(eta_g, PHI(eta_g), 'k-', lw=2,
              label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$', zorder=10)
    ax_b.axhline(0, color='k', lw=0.8)
    rows = []
    for r in results:
        if r is None: continue
        pq = (r['p'],r['q']); col,mrk,lbl = _style(pq)
        # panel a: R_n vs eta_n
        ax_a.scatter(r['etas'], 1.0-r['Rs'], s=40, alpha=0.8,
                     color=col, marker=mrk, label=lbl)
        # panel b: residuals
        ax_b.scatter(r['etas'], (1.0-r['Rs']) - PHI(r['etas']),
                     s=40, alpha=0.8, color=col, marker=mrk, label=lbl)
        # panel c: Sigma vs cumulative |S_n| (log-log)
        valid = r['sigmas'] > 0
        ax_c.loglog(r['cumuls'][valid], r['sigmas'][valid], 'o-',
                    color=col, ms=6, label=lbl)
        if r['fit_mask'].sum() >= FIT_MIN_PTS:
            cf = r['cumuls'][r['fit_mask']]
            sf = r['sigmas'][r['fit_mask']]
            xf = np.linspace(cf[0], cf[-1], 50)
            ax_c.loglog(xf, sf[0]*(xf/cf[0])**(-r['delta']),
                        '--', color=col, lw=1.8)
            # mark S* = q^2
            ax_c.axvline(r['q']**2, color=col, ls=':', alpha=0.5)
        # table row
        ss = f"{r['s_star']:.0f}" if np.isfinite(r['s_star']) else "N/A"
        de = f"{r['delta']:.3f}" if np.isfinite(r['delta']) else "--"
        be = f"{r['beta_eff']:.4f}" if np.isfinite(r['beta_eff']) else "--"
        r2 = f"{r['r2']:.3f}" if np.isfinite(r['r2']) else "--"
        rows.append([str(pq[0]),str(pq[1]),str(r['N']),
                     str(r['ambient']),ss,de,be,r2])

    ax_a.set_xlabel(r'$\eta_n$'); ax_a.set_ylabel(r'$R_n^{(3)}$')
    ax_a.set_title('(a) State law'); ax_a.legend(fontsize=8)
    ax_a.set_xlim(left=0); ax_a.set_ylim(0,1.15)
    ax_b.set_xlabel(r'$\eta_n$')
    ax_b.set_ylabel(r'$R_n^{(3)}-\Phi(\eta_n)$')
    ax_b.set_title('(b) Residuals'); ax_b.legend(fontsize=8)
    ax_c.set_xlabel(r'Cumulative $|S_n|$ (log)')
    ax_c.set_ylabel(r'$\bar\Sigma_n$ (log)'); ax_c.legend(fontsize=8)
    ax_c.set_title(r'(c) $\bar\Sigma_n$ vs $|S_n|$; dashed=fit; dotted=$q^2$')
    ax_d.axis('off')
    cl = ['$p$','$q$','$|G|$','ambient',
          r'$|S^*|_\mathrm{eff}$',r'$\delta$',r'$\beta_\mathrm{eff}$','$R^2$']
    if rows:
        tab = ax_d.table(cellText=rows, colLabels=cl, loc='center', cellLoc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.1,1.4)
    ax_d.set_title('(d) Parameter table', pad=10)
    fig.suptitle(r'SpectralO8 -- State law test, $k=3$ permutation-path fingerprint',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)


def fig2(results, path="SpectralO8_fig2.pdf"):
    """Scaling collapse: bin-averaged R_n vs eta_n."""
    fig, ax = plt.subplots(figsize=(6,5))
    for r in results:
        if r is None: continue
        pq = (r['p'],r['q']); col,mrk,lbl = _style(pq)
        etas = r['etas']; Rs = 1.0-r['Rs']
        if len(etas) < 3:
            ax.scatter(etas, Rs, color=col, marker=mrk, s=50, label=lbl); continue
        edges = np.linspace(0, max(etas.max(),0.1)+1e-8, 12)
        em, Rm, Re = [], [], []
        for i in range(len(edges)-1):
            m = (etas>=edges[i])&(etas<edges[i+1])
            if m.sum()>0:
                em.append(0.5*(edges[i]+edges[i+1]))
                Rm.append(Rs[m].mean()); Re.append(Rs[m].std())
        ax.errorbar(em, Rm, yerr=Re, fmt=mrk, color=col,
                    capsize=3, ms=6, label=lbl)
    eta_g = np.linspace(0,8,300)
    ax.plot(eta_g, PHI(eta_g), 'k--', lw=2, label=r'$\Phi(\eta)$')
    ax.set_xlabel(r'$\eta_n$'); ax.set_ylabel(r'$R_n^{(3)}$ (bin avg)')
    ax.set_title(r'Scaling collapse: $k=3$ permutation fingerprint')
    ax.legend(); ax.set_xlim(left=0); ax.set_ylim(0,1.15)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)


def fig3(results, path="SpectralO8_fig3.pdf"):
    """Saturation profile and rank growth."""
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    ax_l, ax_r = axes
    for r in results:
        if r is None: continue
        pq = (r['p'],r['q']); col,_,lbl = _style(pq)
        ax_l.plot(r['cumuls'], r['r_tildes'], color=col, label=lbl)
        ax_l.axvline(r['q']**2, color=col, ls=':', alpha=0.6)
        ax_r.plot(r['cumuls'], r['ranks'], color=col, label=lbl)
        q2_label = f'$q^2$={r["q"]**2}'
        ax_r.axvline(r['q']**2, color=col, ls=':', alpha=0.6, label=q2_label)
    ax_l.axhline(R_TILDE_SAT, color='k', ls='--', lw=0.8, label='sat threshold')
    ax_l.set_xlabel(r'Cumulative $|S_n|$')
    ax_l.set_ylabel(r'$\tilde r_n^{(3)}$ (novelty fraction)')
    ax_l.set_title(r'Saturation profile; dotted = $q^2$'); ax_l.legend(fontsize=8)
    ax_r.set_xlabel(r'Cumulative $|S_n|$')
    ax_r.set_ylabel('Span rank'); ax_r.set_title('Rank growth vs $|S_n|$')
    ax_r.legend(fontsize=8)
    fig.suptitle(r'SpectralO8 -- Saturation and rank, $k=3$ permutation fingerprint',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)


# ---------------------------------------------------------------------------
# 8.  Summary table
# ---------------------------------------------------------------------------

def print_table(results):
    sep = "="*82
    print(f"\n{sep}")
    print(f"{'Table 1: k=3 permutation fingerprint -- delta and beta_eff':^82}")
    print(sep)
    hdr = (f"{'(p,q)':>9}  {'|G|':>7}  {'ambient':>8}  {'sketch':>7}  "
           f"{'S*_eff':>8}  {'delta':>7}  {'beta_eff':>9}  {'R^2':>6}")
    print(hdr); print("-"*82)
    for r in results:
        if r is None: continue
        pq=(r['p'],r['q'])
        ss=f"{r['s_star']:.0f}" if np.isfinite(r['s_star']) else "N/A"
        de=f"{r['delta']:.3f}"  if np.isfinite(r['delta'])   else "--"
        be=f"{r['beta_eff']:.4f}" if np.isfinite(r['beta_eff']) else "--"
        r2=f"{r['r2']:.3f}"    if np.isfinite(r['r2'])      else "--"
        print(f"{str(pq):>9}  {r['N']:>7}  {r['ambient']:>8}  "
              f"{r['dim_sk']:>7}  {ss:>8}  {de:>7}  {be:>9}  {r2:>6}")
    print(sep)
    print(f"  Target: delta in [7.4, 10.6]  =>  beta_eff in (0.09, 0.13)")
    print(f"  Note: delta < target indicates pre-saturation window too short at these q.")
    print()

# ---------------------------------------------------------------------------
# 9.  Main
# ---------------------------------------------------------------------------

def main():
    print("SpectralO8.py v2 -- k=3 permutation-path fingerprint on LPS X^{p,q}")
    print(f"D_sketch={D_SKETCH}, ambient_sketch={D_SKETCH**3}, configs={LPS_CONFIGS}")
    print()
    results = []
    for p, q in LPS_CONFIGS:
        res = run_one(p, q)
        results.append(res)
        # save intermediate results to avoid losing data
        np.save(f"SpectralO8_{p}_{q}.npy", res, allow_pickle=True)
        print(f"  Saved SpectralO8_{p}_{q}.npy")
    print("\nGenerating figures ...")
    fig1(results, "SpectralO8_fig1.pdf")
    fig2(results, "SpectralO8_fig2.pdf")
    fig3(results, "SpectralO8_fig3.pdf")
    print_table(results)

if __name__ == "__main__":
    main()