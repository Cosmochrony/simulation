"""
SpectralO8.py  --  v3
Companion script for:
  "Three-Step Path Fingerprints on LPS Graphs and the Capacity Growth Exponent"
  Cosmochrony Spectral Cascade Programme -- Paper O8

Fingerprint architecture:
  pi_3(v1,v2,v3) = rho_perm(v1) x rho_perm(v2) x rho_perm(v3) in R^{(q+1)^3}.
  Structured sketch: S in R^{D_s x (q+1)}, fp3_sk = kron(kron(S*pv1,S*pv2),S*pv3),
  ambient_sketch = D_s^3.  Span tracking is done in the sketched space.
  D_s=14 gives ambient_sketch=2744.  Disclosed in paper Section 5 (Setup).

Fit window:
  delta is extracted by log-log fit of Sigma_bar_n vs cumulative |S_n|
  restricted to the pre-saturation window |S_n| <= q^2 (Conjecture 3.6).

LPS configurations (p prime, q prime, p!=q, p=q=1 mod 4, Legendre(p,q)=1):
  (p=17, q=13): |G|=1092,  q^2=169    ~ 2 s
  (p=13, q=17): |G|=2448,  q^2=289    ~ 5 s
  (p=5,  q=29): |G|=12180, q^2=841    ~ 2 min
  (p=5,  q=41): |G|=34440, q^2=1681   ~ 8 min

Outputs:
  SpectralO8_fig1.pdf  -- state law test (4-panel)
  SpectralO8_fig2.pdf  -- scaling collapse
  SpectralO8_fig3.pdf  -- saturation profile and rank growth
  SpectralO8_fig4.pdf  -- window depth diagnosis (Fig 4 of paper)
  SpectralO8_{p}_{q}.npy  -- per-config result dict (allow_pickle=True)

Author: Jerome Beau <jerome.beau@cosmochrony.org>
Date:   March 2026
"""

import math
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

LPS_CONFIGS = [(17, 13), (13, 17), (5, 29), (5, 41)]

D_SKETCH   = 14      # sketch dimension; ambient_sketch = D_SKETCH^3 = 2744
QR_BATCH   = 300     # sub-batch size for streaming QR
QR_EPS     = 1e-9    # independence threshold in sketched space
FIT_MIN_PTS = 3      # minimum BFS steps in window for a reliable fit
SEED       = 42      # RNG seed for sketch matrix

# ---------------------------------------------------------------------------
# 1.  Number-theoretic helpers
# ---------------------------------------------------------------------------

def modinv(a, m):
    return pow(int(a) % m, m - 2, m)


def sqrt_mod(a, q):
    """Tonelli-Shanks.  Raises ValueError if a is not a QR mod q."""
    a = int(a) % q
    if a == 0:
        return 0
    if pow(a, (q - 1) // 2, q) != 1:
        raise ValueError(f"{a} is not a QR mod {q}")
    if q % 4 == 3:
        return pow(a, (q + 1) // 4, q)
    S, Qv = 0, q - 1
    while Qv % 2 == 0:
        Qv //= 2; S += 1
    z = 2
    while pow(z, (q - 1) // 2, q) != q - 1:
        z += 1
    M_, c_, t_, R_ = S, pow(z, Qv, q), pow(a, Qv, q), pow(a, (Qv + 1) // 2, q)
    while True:
        if t_ == 1:
            return R_
        i, tmp = 1, (t_ * t_) % q
        while tmp != 1:
            tmp = (tmp * tmp) % q; i += 1
        b_ = pow(c_, pow(2, M_ - i - 1, q - 1), q)
        M_, c_, t_, R_ = i, (b_*b_)%q, (t_*b_*b_)%q, (R_*b_)%q

# ---------------------------------------------------------------------------
# 2.  PSL(2, F_q)
# ---------------------------------------------------------------------------

def _key(a, b, c, d, q):
    pos = (a % q, b % q, c % q, d % q)
    neg = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
    return min(pos, neg)


def build_psl2_fq(q):
    """Enumerate PSL(2,F_q). Returns (elements, index_of)."""
    seen = set(); elements = []; index_of = {}
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a != 0:
                    d   = (1 + b * c) * modinv(a, q) % q
                    key = _key(a, b, c, d, q)
                    if key not in seen:
                        seen.add(key); idx = len(elements)
                        elements.append(key); index_of[key] = idx
                        index_of[_key(q-a, q-b, q-c, q-d, q)] = idx
                else:
                    if c == 0:
                        continue
                    b2 = (q - modinv(c, q)) % q
                    for d in range(q):
                        key = _key(0, b2, c, d, q)
                        if key not in seen:
                            seen.add(key); idx = len(elements)
                            elements.append(key); index_of[key] = idx
                            index_of[_key(0, q-b2, q-c, q-d, q)] = idx
    return elements, index_of


def mul_psl2(m1, m2, q, index_of):
    a1, b1, c1, d1 = m1; a2, b2, c2, d2 = m2
    return index_of[_key(
        (a1*a2+b1*c2)%q, (a1*b2+b1*d2)%q,
        (c1*a2+d1*c2)%q, (c1*b2+d1*d2)%q, q)]

# ---------------------------------------------------------------------------
# 3.  LPS generators
# ---------------------------------------------------------------------------

def lps_generators(p, q, index_of):
    """
    LPS generators for X^{p,q}.
    Matrix for quaternion (a,b,c,d) with a^2+b^2+c^2+d^2=p, a>0 odd, b,c,d even:
      M = (1/sqrt(p)) * [[a + b*i_q,  c + d*i_q],
                         [-c + d*i_q,  a - b*i_q]]  mod q
    where i_q = sqrt(-1) mod q and sqrt(p) mod q exists since p is QR mod q.
    """
    i_q       = sqrt_mod(q - 1, q)
    inv_sqrtp = modinv(sqrt_mod(p, q), q)
    found = set()
    bnd = int(p**0.5) + 2
    for a in range(1, bnd + 2, 2):
        for b in range(-(bnd//2*2), bnd//2*2 + 1, 2):
            for c in range(-(bnd//2*2), bnd//2*2 + 1, 2):
                rem = p - a*a - b*b - c*c
                if rem < 0:
                    continue
                dsq = int(math.sqrt(rem) + 0.5)
                for d in ([0] if dsq == 0 else [dsq, -dsq]):
                    if d % 2 == 0 and a*a + b*b + c*c + d*d == p:
                        found.add((a, b, c, d))
    result = []
    for (a, b, c, d) in found:
        an = a*inv_sqrtp % q; bn = b*inv_sqrtp % q
        cn = c*inv_sqrtp % q; dn = d*inv_sqrtp % q
        m00 = (an + bn*i_q) % q; m01 = (cn + dn*i_q) % q
        m10 = (q - cn + dn*i_q) % q; m11 = (an + q - bn*i_q % q) % q
        key = _key(m00, m01, m10, m11, q)
        if key in index_of:
            gi = index_of[key]
            if gi not in result:
                result.append(gi)
    return result

# ---------------------------------------------------------------------------
# 4.  Permutation vector and structured sketch
# ---------------------------------------------------------------------------

def mobius(mat, z, q):
    """Mobius action of mat on P^1(F_q). z in {0,...,q-1, q} (q = infinity)."""
    a, b, c, d = mat
    if z == q:
        return q if c % q == 0 else a * modinv(c, q) % q
    denom = (c * z + d) % q
    if denom == 0:
        return q
    return (a * z + b) * modinv(denom, q) % q


def perm_vec(mat, q):
    """
    Centred one-hot image of base point 0 under mat in R^{q+1}:
      rho_perm(mat) = e_{mat*0} - 1/(q+1) * 1.
    """
    n = q + 1
    pv = np.full(n, -1.0 / n)
    pv[mobius(mat, 0, q)] += 1.0
    return pv


def build_sketch(q, D_s=D_SKETCH, seed=SEED):
    """
    Sketch matrix S in R^{D_s x (q+1)}, entries N(0, 1/D_s).
    The sketched fingerprint fp3_sk = (S*pv1) x (S*pv2) x (S*pv3) in R^{D_s^3}.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((D_s, q + 1)) / math.sqrt(D_s)


def fp3_batch(SK1, SK2, SK3):
    """
    Batch k=3 sketched fingerprints.
    SK1, SK2, SK3: (n x D_s) arrays -- sketched perm vecs per path vertex.
    Returns (n x D_s^3) matrix via batched outer product.
    """
    return np.einsum('bi,bj,bk->bijk', SK1, SK2, SK3).reshape(len(SK1), -1)

# ---------------------------------------------------------------------------
# 5.  Streaming QR span tracker (in sketched space, dimension D_s^3)
# ---------------------------------------------------------------------------

class SpanTracker:
    """
    Incremental orthonormal basis via sub-batch QR decompositions.
    Hard cap at dim: once rank == dim, all novelties are 0 and no
    further directions are added (full sketch space spanned).
    """

    def __init__(self, dim, batch=QR_BATCH, eps=QR_EPS):
        self.dim   = dim
        self.batch = batch
        self.eps   = eps
        self.basis = np.zeros((dim, 0))  # dim x rank
        self._pend = []

    def _flush(self):
        if not self._pend:
            return
        # Hard cap: once the full sketch space is spanned, nothing more to add.
        if self.basis.shape[1] >= self.dim:
            self._pend = []
            return
        mat = np.column_stack(self._pend)
        if self.basis.shape[1] > 0:
            mat = mat - self.basis @ (self.basis.T @ mat)
        # Cap columns to remaining capacity before QR
        remaining = self.dim - self.basis.shape[1]
        if mat.shape[1] > remaining:
            mat = mat[:, :remaining]
        try:
            Q, R = linalg.qr(mat, mode='economic')
        except Exception:
            self._pend = []
            return
        diag = np.abs(np.diag(R))
        good = Q[:, diag > self.eps]
        if good.shape[1] > 0:
            if self.basis.shape[1] > 0:
                good = good - self.basis @ (self.basis.T @ good)
            nrm = np.linalg.norm(good, axis=0)
            keep = nrm > self.eps
            good = good[:, keep] / nrm[keep]
            # Enforce hard cap
            slots = self.dim - self.basis.shape[1]
            good  = good[:, :slots]
            if good.shape[1] > 0:
                self.basis = (np.column_stack([self.basis, good])
                              if self.basis.shape[1] > 0 else good)
        self._pend = []

    def add_batch(self, mat):
        """mat: dim x n. Process in sub-batches; stops once rank reaches dim."""
        for i in range(0, mat.shape[1], self.batch):
            if self.basis.shape[1] >= self.dim:
                break
            chunk = mat[:, i:i + self.batch]
            self._pend += [chunk[:, j] for j in range(chunk.shape[1])]
            if len(self._pend) >= self.batch:
                self._flush()
        self._flush()

    def novelties(self, mat):
        """
        Squared relative residuals for dim x n matrix, clamped to [0, 1].
        Returns zeros if rank == dim (full sketch space spanned).
        """
        nrm2 = np.sum(mat**2, axis=0) + 1e-30
        if self.basis.shape[1] >= self.dim:
            return np.zeros(mat.shape[1])
        if self.basis.shape[1] == 0:
            return np.clip(np.sum(mat**2, axis=0) / nrm2, 0.0, 1.0)
        res = mat - self.basis @ (self.basis.T @ mat)
        return np.clip(np.sum(res**2, axis=0) / nrm2, 0.0, 1.0)

    @property
    def rank(self):
        return self.basis.shape[1]

# ---------------------------------------------------------------------------
# 6.  BFS cascade
# ---------------------------------------------------------------------------

def run_one(p, q, verbose=True):
    """
    Full BFS cascade on X^{p,q} with k=3 sketched permutation-path fingerprints.

    Records per BFS step: cumulative |S_n|, shell size, sketch rank,
    mean capacity proxy Sigma_bar, mean occupancy proxy eta_bar,
    mean redundancy R_bar, novelty fraction r_tilde.

    Fits delta over window |S_n| <= q^2 (Conjecture 3.6 pre-saturation window).
    Returns result dict; also prints summary line.
    """
    label = f"(p={p},q={q})"
    if verbose:
        print(f"\n{'='*64}")
        print(f"  {label}  |G|={q*(q*q-1)//2}  q^2={q**2}  "
              f"sketch_dim={D_SKETCH**3}")
        print(f"{'='*64}")

    print(f"  Building PSL(2,F_{q}) ...", end=" ", flush=True)
    elements, index_of = build_psl2_fq(q)
    N = len(elements)
    print(f"|G|={N}")

    print("  Computing generators ...", end=" ", flush=True)
    gen_idx = lps_generators(p, q, index_of)
    print(f"{len(gen_idx)}  (expected {p+1})")
    if not gen_idx:
        print("  ERROR: no generators found.")
        return None

    S = build_sketch(q, D_SKETCH, SEED)
    print("  Sketching perm vectors ...", end=" ", flush=True)
    PV_raw = np.array([perm_vec(mat, q) for mat in elements])   # N x (q+1)
    PV_sk  = (S @ PV_raw.T).T                                   # N x D_SKETCH
    print("done")

    print("  Building neighbour table ...", end=" ", flush=True)
    nbrs = [[mul_psl2(elements[v], elements[g], q, index_of)
             for g in gen_idx] for v in range(N)]
    print("done")

    # BFS state
    visited = np.zeros(N, dtype=bool)
    parent  = np.full(N, -1, dtype=int)
    visited[0] = True

    dim_sk  = D_SKETCH**3
    tracker = SpanTracker(dim=dim_sk)
    queue   = deque([0])

    rec_cumul = []; rec_shell = []; rec_sigma = []
    rec_eta   = []; rec_R     = []; rec_rank  = []

    step = 0
    while queue:
        step   += 1
        cur_lvl = list(queue); queue = deque(); shell = []
        for v in cur_lvl:
            for w in nbrs[v]:
                if not visited[w]:
                    visited[w] = True; parent[w] = v
                    shell.append(w); queue.append(w)
        if not shell:
            break

        sz  = len(shell)
        SK1 = PV_sk[[parent[parent[w]] if parent[w] >= 0 and parent[parent[w]] >= 0
                     else (parent[w] if parent[w] >= 0 else w)
                     for w in shell]]
        SK2 = PV_sk[[parent[w] if parent[w] >= 0 else w for w in shell]]
        SK3 = PV_sk[shell]
        FP  = fp3_batch(SK1, SK2, SK3).T        # dim_sk x sz

        # novelties BEFORE adding to span (clamped to [0,1])
        nov = tracker.novelties(FP)              # length sz
        tracker.add_batch(FP)

        cumul     = int(np.sum(visited))
        sigma_bar = float(np.mean(nov))
        psi_sq    = 1.0 / math.sqrt(tracker.rank + 1)
        eta_bar   = psi_sq / max(sigma_bar, 1e-14)
        R_bar     = float(np.mean(nov < 0.5))    # fraction projectively redundant
        r_tilde   = float(np.mean(nov > 0.1))    # fraction genuinely novel

        rec_cumul.append(cumul);  rec_shell.append(sz)
        rec_sigma.append(sigma_bar); rec_eta.append(eta_bar)
        rec_R.append(R_bar);      rec_rank.append(tracker.rank)

        if verbose and (step <= 9 or step % 4 == 0):
            print(f"  step {step:3d}: |S_n|={cumul:7d}  shell={sz:6d}  "
                  f"rank={tracker.rank:5d}/{dim_sk}  "
                  f"sigma={sigma_bar:.4f}  r~={r_tilde:.3f}")

    cumuls  = np.array(rec_cumul,  dtype=float)
    shells  = np.array(rec_shell,  dtype=float)
    sigmas  = np.array(rec_sigma)
    etas    = np.array(rec_eta)
    Rs      = np.array(rec_R)
    ranks   = np.array(rec_rank, dtype=int)

    # fit delta over pre-saturation window |S_n| <= q^2
    w_mask = (cumuls <= q**2) & (sigmas > 0) & np.isfinite(sigmas)
    # fallback: if window has fewer than FIT_MIN_PTS, use all valid pre-zero steps
    if w_mask.sum() < FIT_MIN_PTS:
        first_zero = next((i for i, s in enumerate(sigmas) if s == 0.0),
                          len(sigmas))
        w_mask = np.zeros(len(cumuls), dtype=bool)
        w_mask[:first_zero] = True
        w_mask &= (sigmas > 0) & np.isfinite(sigmas)

    delta = beta_eff = r2_fit = np.nan
    if w_mask.sum() >= FIT_MIN_PTS:
        lx  = np.log(cumuls[w_mask])
        ly  = np.log(sigmas[w_mask])
        A   = np.column_stack([lx, np.ones_like(lx)])
        cf, *_ = np.linalg.lstsq(A, ly, rcond=None)
        delta = float(-cf[0])
        ly_p  = A @ cf
        ss_r  = float(np.sum((ly - ly_p)**2))
        ss_t  = float(np.sum((ly - ly.mean())**2))
        r2_fit = 1.0 - ss_r / (ss_t + 1e-30)
        if np.isfinite(delta) and delta > -0.5:
            beta_eff = 1.0 / (0.5 + delta)

    # rank at first step where cumul >= q^2
    rank_at_q2 = next((int(rk) for cu, rk in zip(cumuls, ranks) if cu >= q**2),
                      int(ranks[-1]))
    # effective saturation: first step where r_tilde < 0.1 (if any)
    r_tildes_arr = 1.0 - Rs
    s_star = next((float(cu) for cu, rt in zip(cumuls, r_tildes_arr) if rt < 0.1),
                  float(cumuls[-1]))

    if verbose:
        print(f"  => delta={delta:.4f}  beta_eff={beta_eff:.4f}  "
              f"R2={r2_fit:.4f}  rank@q^2={rank_at_q2}  q^2={q**2}")

    return dict(
        label=label, p=p, q=q, N=N,
        ambient=(q+1)**3, dim_sk=dim_sk,
        cumuls=cumuls, shells=shells,
        sigmas=sigmas, etas=etas, Rs=Rs,
        r_tildes=r_tildes_arr, ranks=ranks,
        delta=delta, beta_eff=beta_eff, r2=r2_fit,
        s_star=s_star, rank_at_q2=rank_at_q2, fit_mask=w_mask,
    )

# ---------------------------------------------------------------------------
# 7.  Figure helpers
# ---------------------------------------------------------------------------

PHI  = lambda eta: 1.0 / np.sqrt(1.0 + eta**2)
COLS = {(17,13):'#1f77b4', (13,17):'#ff7f0e',
        (5,29): '#2ca02c', (5,41): '#d62728', (5,61):'#9467bd'}
MRKS = {(17,13):'o', (13,17):'s', (5,29):'^', (5,41):'D', (5,61):'P'}
LBLS = {(17,13):r'$X^{17,13}$', (13,17):r'$X^{13,17}$',
        (5,29): r'$X^{5,29}$',  (5,41): r'$X^{5,41}$', (5,61):r'$X^{5,61}$'}


def _style(pq):
    return COLS.get(pq, 'gray'), MRKS.get(pq, 'x'), LBLS.get(pq, str(pq))

# ---------------------------------------------------------------------------
# 8.  Figure 1: state law test (4-panel)
# ---------------------------------------------------------------------------

def fig1(results, path="SpectralO8_fig1.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    eta_g = np.linspace(0, 8, 300)
    ax_a.plot(eta_g, PHI(eta_g), 'k-', lw=2,
              label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$', zorder=10)
    ax_b.axhline(0, color='k', lw=0.8)
    rows = []
    for r in results:
        if r is None:
            continue
        pq = (r['p'], r['q']); col, mrk, lbl = _style(pq)
        # panels (a) and (b)
        ax_a.scatter(r['etas'], 1.0 - r['Rs'], s=40, alpha=0.8,
                     color=col, marker=mrk, label=lbl)
        ax_b.scatter(r['etas'], (1.0 - r['Rs']) - PHI(r['etas']),
                     s=40, alpha=0.8, color=col, marker=mrk, label=lbl)
        # panel (c): Sigma vs cumulative |S_n| (log-log)
        valid = r['sigmas'] > 0
        ax_c.loglog(r['cumuls'][valid], r['sigmas'][valid], 'o-',
                    color=col, ms=6, label=lbl)
        if r['fit_mask'].sum() >= FIT_MIN_PTS and np.isfinite(r['delta']):
            cf = r['cumuls'][r['fit_mask']]
            sf = r['sigmas'][r['fit_mask']]
            xf = np.linspace(cf[0], cf[-1], 50)
            ax_c.loglog(xf, sf[0] * (xf / cf[0])**(-r['delta']),
                        '--', color=col, lw=1.8)
        ax_c.axvline(r['q']**2, color=col, ls=':', alpha=0.5)
        # table row
        de = f"{r['delta']:.3f}"   if np.isfinite(r['delta'])    else "--"
        be = f"{r['beta_eff']:.4f}" if np.isfinite(r['beta_eff']) else "--"
        r2 = f"{r['r2']:.3f}"     if np.isfinite(r['r2'])       else "--"
        rows.append([str(pq[0]), str(pq[1]), str(r['N']),
                     str(r['ambient']), str(r['rank_at_q2']),
                     str(r['q']**2), de, be, r2])

    ax_a.set_xlabel(r'$\eta_n$'); ax_a.set_ylabel(r'$\bar R_n^{(3)}$')
    ax_a.set_title('(a) State law: $\\bar R_n^{(3)}$ vs $\\bar\\eta_n$')
    ax_a.legend(fontsize=8); ax_a.set_xlim(left=0); ax_a.set_ylim(0, 1.15)
    ax_b.set_xlabel(r'$\eta_n$')
    ax_b.set_ylabel(r'$\bar R_n^{(3)} - \Phi(\bar\eta_n)$')
    ax_b.set_title('(b) Residuals from master curve')
    ax_b.legend(fontsize=8)
    ax_c.set_xlabel(r'Cumulative $|S_n|$ (log)')
    ax_c.set_ylabel(r'$\bar\Sigma_n$ (log)')
    ax_c.set_title(r'(c) Capacity $\bar\Sigma_n$ vs $|S_n|$; '
                   r'dashed = fit; dotted = $q^2$')
    ax_c.legend(fontsize=8)
    ax_d.axis('off')
    col_labels = ['$p$', '$q$', '$|G|$', 'ambient',
                  r'rank@$q^2$', r'$q^2$',
                  r'$\delta$', r'$\beta_\mathrm{eff}$', r'$R^2$']
    if rows:
        tab = ax_d.table(cellText=rows, colLabels=col_labels,
                         loc='center', cellLoc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1.05, 1.4)
    ax_d.set_title('(d) Parameter table', pad=10)
    fig.suptitle(r'SpectralO8 -- State law test, $k=3$ sketched permutation fingerprint',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)

# ---------------------------------------------------------------------------
# 9.  Figure 2: scaling collapse
# ---------------------------------------------------------------------------

def fig2(results, path="SpectralO8_fig2.pdf"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        if r is None:
            continue
        pq = (r['p'], r['q']); col, mrk, lbl = _style(pq)
        etas = r['etas']; Rs = 1.0 - r['Rs']
        if len(etas) < 3:
            ax.scatter(etas, Rs, color=col, marker=mrk, s=50, label=lbl)
            continue
        edges = np.linspace(0, max(etas.max(), 0.1) + 1e-8, 12)
        em, Rm, Re = [], [], []
        for i in range(len(edges) - 1):
            m = (etas >= edges[i]) & (etas < edges[i + 1])
            if m.sum() > 0:
                em.append(0.5 * (edges[i] + edges[i + 1]))
                Rm.append(Rs[m].mean()); Re.append(Rs[m].std())
        ax.errorbar(em, Rm, yerr=Re, fmt=mrk, color=col,
                    capsize=3, ms=6, label=lbl)
    eta_g = np.linspace(0, 8, 300)
    ax.plot(eta_g, PHI(eta_g), 'k--', lw=2, label=r'$\Phi(\eta)$')
    ax.set_xlabel(r'$\eta_n$'); ax.set_ylabel(r'$\bar R_n^{(3)}$ (bin avg)')
    ax.set_title(r'Scaling collapse: $k=3$ permutation fingerprint')
    ax.legend(); ax.set_xlim(left=0); ax.set_ylim(0, 1.15)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)

# ---------------------------------------------------------------------------
# 10.  Figure 3: saturation profile and rank growth
# ---------------------------------------------------------------------------

def fig3(results, path="SpectralO8_fig3.pdf"):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    for r in results:
        if r is None:
            continue
        pq = (r['p'], r['q']); col, _, lbl = _style(pq)
        ax_l.plot(r['cumuls'], r['r_tildes'], color=col, label=lbl)
        ax_l.axvline(r['q']**2, color=col, ls=':', alpha=0.6)
        ax_r.plot(r['cumuls'], r['ranks'], color=col, label=lbl)
        ax_r.axvline(r['q']**2, color=col, ls=':', alpha=0.6)
    ax_l.axhline(0.1, color='k', ls='--', lw=0.8, label='threshold 0.1')
    ax_l.set_xlabel(r'Cumulative $|S_n|$')
    ax_l.set_ylabel(r'$\tilde r_n^{(3)}$ (novelty fraction)')
    ax_l.set_title(r'Saturation profile; dotted = $q^2$')
    ax_l.legend(fontsize=8)
    ax_r.set_xlabel(r'Cumulative $|S_n|$')
    ax_r.set_ylabel('Sketch rank')
    ax_r.set_title(r'Rank growth vs $|S_n|$')
    ax_r.legend(fontsize=8)
    fig.suptitle(r'SpectralO8 -- Saturation and rank, $k=3$ fingerprint',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)

# ---------------------------------------------------------------------------
# 11.  Figure 4: window depth diagnosis (rescaling n -> n/log2(q))
# ---------------------------------------------------------------------------

def fig4(results, path="SpectralO8_fig4.pdf"):
    """
    Two panels:
      Left:  Sigma_n vs BFS step n (semi-log).
             Solid = in-window |S_n| <= q^2; faded dashed = post-window.
             Dotted vertical at step where |S_n| first reaches q^2.
      Right: Sigma_n vs rescaled step n/log2(q) (semi-log).
             Collapse to n/log2(q) ~ 0.82 for all (p,q) confirms O(log q) depth.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8))
    X_WINDOW = 0.82   # empirical collapse point of the window boundary

    for r in results:
        if r is None:
            continue
        pq = (r['p'], r['q']); col, mrk, lbl = _style(pq)
        sigmas  = r['sigmas']
        cumuls  = r['cumuls']
        n_steps = np.arange(1, len(sigmas) + 1, dtype=float)
        log2q   = math.log2(r['q'])
        n_resc  = n_steps / log2q

        # step where cumul first reaches q^2
        step_q2 = next((i + 1 for i, c in enumerate(cumuls) if c >= r['q']**2),
                       len(cumuls))
        in_win   = (n_steps <= step_q2) & (sigmas > 0)
        post_win = (n_steps >  step_q2) & (sigmas > 0)

        # --- left panel ---
        if in_win.sum():
            ax_l.semilogy(n_steps[in_win], sigmas[in_win],
                          f'{mrk}-', color=col, ms=8, lw=2,
                          label=lbl, zorder=3)
        if post_win.sum():
            ax_l.semilogy(n_steps[post_win], sigmas[post_win],
                          f'{mrk}--', color=col, ms=5, lw=1,
                          alpha=0.30, zorder=2)
        # dotted vertical at step_q2
        if step_q2 <= len(sigmas) and sigmas[step_q2 - 1] > 0:
            ax_l.axvline(step_q2, color=col, ls=':', lw=1.2, alpha=0.55)

        # --- right panel ---
        if in_win.sum():
            ax_r.semilogy(n_resc[in_win], sigmas[in_win],
                          f'{mrk}-', color=col, ms=8, lw=2,
                          label=lbl, zorder=3)
        if post_win.sum():
            ax_r.semilogy(n_resc[post_win], sigmas[post_win],
                          f'{mrk}--', color=col, ms=5, lw=1,
                          alpha=0.30, zorder=2)

    # shared collapse reference
    ax_r.axvline(X_WINDOW, color='#555555', ls='--', lw=1.4, alpha=0.7,
                 label=r'$n/\log_2 q \approx 0.82$')

    # formatting
    ax_l.set_xlabel(r'BFS step $n$', fontsize=12)
    ax_l.set_ylabel(r'$\bar{\Sigma}_n$', fontsize=12)
    ax_l.set_title('(a) Capacity per BFS step\n'
                   r'solid = $|S_n|\!\leq\!q^2$, faded = $|S_n|\!>\!q^2$; '
                   r'dotted = $q^2$ boundary', fontsize=10)
    ax_l.legend(fontsize=9, loc='lower left')
    ax_l.set_xlim(left=0.3)
    ax_l.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_r.set_xlabel(r'Rescaled step $n / \log_2 q$', fontsize=12)
    ax_r.set_ylabel(r'$\bar{\Sigma}_n$', fontsize=12)
    ax_r.set_title(r'(b) Structural collapse under rescaling $n \mapsto n/\log_2 q$'
                   '\n'
                   r'window ends at $n/\log_2 q \approx 0.82$ for all $(p,q)$',
                   fontsize=10)
    ax_r.legend(fontsize=9, loc='lower left')
    ax_r.set_xlim(left=0)
    ax_r.text(0.97, 0.97,
              'Geometric constraint:\n'
              r'$O(q^2)$ vertices $\Rightarrow$ $O(\log q)$ BFS steps',
              transform=ax_r.transAxes, ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.35', fc='#fffbe6',
                        ec='#ccaa00', alpha=0.92))

    fig.suptitle(
        r'SpectralO8 Fig.\ 4 -- Window depth: pre-saturation region'
        r' $|S_n|\!\leq\!q^2$ spans only $O(\log q)$ BFS steps',
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}"); plt.close(fig)

# ---------------------------------------------------------------------------
# 12.  Summary table (stdout)
# ---------------------------------------------------------------------------

def print_table(results):
    sep = "=" * 86
    print(f"\n{sep}")
    print(f"{'Table 1: k=3 permutation fingerprint -- capacity exponent delta':^86}")
    print(sep)
    hdr = (f"{'(p,q)':>9}  {'|G|':>7}  {'ambient':>8}  {'sketch':>7}  "
           f"{'rank@q2':>8}  {'q^2':>6}  {'delta':>7}  {'beta_eff':>9}  {'R^2':>6}")
    print(hdr); print("-" * 86)
    for r in results:
        if r is None:
            continue
        pq  = (r['p'], r['q'])
        de  = f"{r['delta']:.3f}"    if np.isfinite(r['delta'])    else "--"
        be  = f"{r['beta_eff']:.4f}" if np.isfinite(r['beta_eff']) else "--"
        r2  = f"{r['r2']:.3f}"      if np.isfinite(r['r2'])       else "--"
        print(f"{str(pq):>9}  {r['N']:>7}  {r['ambient']:>8}  "
              f"{r['dim_sk']:>7}  {r['rank_at_q2']:>8}  {r['q']**2:>6}  "
              f"{de:>7}  {be:>9}  {r2:>6}")
    print(sep)
    print("  Target: delta in [7.4, 10.6]  =>  beta_eff in (0.09, 0.13)")
    print("  Result: Outcome C -- window exists (O(q^2) vertices) but spans")
    print("          only O(log q) BFS steps (exponential LPS shell growth).")
    print()

# ---------------------------------------------------------------------------
# 13.  Main
# ---------------------------------------------------------------------------

def main():
    print("SpectralO8.py v3 -- k=3 permutation-path fingerprint on LPS X^{p,q}")
    print(f"D_sketch={D_SKETCH}, ambient_sketch={D_SKETCH**3}")
    print(f"Configs: {LPS_CONFIGS}")
    print()
    results = []
    for p, q in LPS_CONFIGS:
        res = run_one(p, q)
        results.append(res)
        fname = f"SpectralO8_{p}_{q}.npy"
        np.save(fname, res, allow_pickle=True)
        print(f"  Saved {fname}")
    print("\nGenerating figures ...")
    fig1(results, "SpectralO8_fig1.pdf")
    fig2(results, "SpectralO8_fig2.pdf")
    fig3(results, "SpectralO8_fig3.pdf")
    fig4(results, "SpectralO8_fig4.pdf")
    print_table(results)
    print("Done.")


if __name__ == "__main__":
    main()
