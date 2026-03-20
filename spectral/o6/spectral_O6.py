"""
spectral_O6_fast.py
===================
Optimised Steinberg-matrix fingerprint cascade on X_{5,q}.
Uses batched QR decomposition instead of element-by-element Gram-Schmidt
to handle large span dimensions efficiently.

q in {29, 41, 61} — all satisfy p=5 QR mod q, q=1 mod 4.
"""

import numpy as np
from numpy.linalg import qr, matrix_rank
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import linregress
import time, warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# Arithmetic
# -----------------------------------------------------------------------

def mod_sqrt_minus1(q):
    for i in range(1, q):
        if (i*i + 1) % q == 0:
            return i
    raise ValueError(f"sqrt(-1) mod {q} not found")

def mod_sqrt(a, q):
    a = a % q
    if a == 0: return 0
    if pow(a, (q-1)//2, q) != 1: return None
    if q % 4 == 3: return pow(a, (q+1)//4, q)
    s, n = 0, q - 1
    while n % 2 == 0: n //= 2; s += 1
    z = 2
    while pow(z, (q-1)//2, q) != q-1: z += 1
    m, c, t, r = s, pow(z,n,q), pow(a,n,q), pow(a,(n+1)//2,q)
    while True:
        if t == 0: return 0
        if t == 1: return r
        i, tmp = 1, t*t % q
        while tmp != 1: tmp = tmp*tmp % q; i += 1
        b = pow(c, 1 << (m-i-1), q)
        m, c, t, r = i, b*b%q, t*b*b%q, r*b%q

def _even_range(lim):
    start = -lim if (-lim)%2==0 else -lim+1
    return range(start, lim+1, 2)

def sum4sq(p):
    lim = int(p**0.5) + 2
    res = []
    for a in range(1, lim+1, 2):
        for b in _even_range(lim):
            for c in _even_range(lim):
                d2 = p - a*a - b*b - c*c
                if d2 < 0: continue
                dv = int(d2**0.5 + 0.5)
                if dv*dv == d2 and dv%2 == 0:
                    res.append((a, b, c, dv))
                    if dv != 0: res.append((a, b, c, -dv))
    return res

# -----------------------------------------------------------------------
# LPS generators  (SL(2,F_q), det=1)
# -----------------------------------------------------------------------

def lps_generators(p, q):
    i_q = mod_sqrt_minus1(q)
    r   = mod_sqrt(p % q, q)
    if r is None:
        raise ValueError(f"p={p} is NQR mod q={q}")
    r_inv = pow(int(r), q-2, q)
    sols  = sum4sq(p)

    gens, seen = [], set()
    for (a, b, c, d) in sols:
        m00 = (a + i_q*b)*r_inv % q
        m01 = (c + i_q*d)*r_inv % q
        m10 = ((-c) + i_q*d)*r_inv % q
        m11 = (a - i_q*b)*r_inv % q
        mat = np.array([[m00, m01], [m10, m11]], dtype=np.int64)
        if int((mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]) % q) != 1:
            continue
        k  = tuple(mat.ravel().tolist())
        nk = tuple(((q-mat)%q).ravel().tolist())
        if k not in seen and nk not in seen:
            seen.add(k); seen.add(nk); gens.append(mat)
    if len(gens) < p+1:
        raise RuntimeError(f"Got {len(gens)} generators for p={p},q={q}")
    return gens[:p+1]

def mat_key(M, q):
    k  = tuple(M.ravel().tolist())
    nk = tuple(((q-M)%q).ravel().tolist())
    return k if k <= nk else nk

def mmul(A, B, q):
    return np.mod(A @ B, q)

# -----------------------------------------------------------------------
# PSL BFS
# -----------------------------------------------------------------------

def build_psl(p, q, gens):
    id_ = np.array([[1,0],[0,1]], dtype=np.int64)
    ik  = mat_key(id_, q)
    vis = {ik: id_}
    Q   = deque([ik])
    while Q:
        uk = Q.popleft()
        u  = vis[uk]
        for g in gens:
            vk = mat_key(mmul(u, g, q), q)
            if vk not in vis:
                vis[vk] = mmul(u, g, q)
                Q.append(vk)
    return vis

# -----------------------------------------------------------------------
# Steinberg permutation matrix  (flattened, dim = (q+1)^2)
# -----------------------------------------------------------------------

def proj1(mat, pt, q):
    if pt < q:
        num = (int(mat[0,0])*pt + int(mat[0,1])) % q
        den = (int(mat[1,0])*pt + int(mat[1,1])) % q
    else:
        num = int(mat[0,0]) % q
        den = int(mat[1,0]) % q
    if den == 0: return q
    return num * pow(den, q-2, q) % q

_ones_cache = {}
def steinberg_vec(mat, q):
    if q not in _ones_cache:
        _ones_cache[q] = np.ones((q+1, q+1)) / (q+1)
    P = np.zeros((q+1, q+1))
    for pt in range(q+1):
        P[proj1(mat, pt, q), pt] = 1.0
    return (P - _ones_cache[q]).ravel()   # shape: (q+1)^2

def build_steinberg(vis, q):
    return {k: steinberg_vec(m, q) for k, m in vis.items()}

# -----------------------------------------------------------------------
# Batched rank-tracking cascade
# -----------------------------------------------------------------------

def _rank_update_chunk(basis, cur_rank, chunk_fps, dim):
    """
    Update orthonormal basis with a chunk of new fingerprints.
    Returns (basis, cur_rank, eff).
    chunk_fps: list of float32 vectors of length dim.
    """
    FP = np.column_stack(chunk_fps).astype(np.float64)
    if cur_rank > 0:
        B = basis.astype(np.float64)
        FP = FP - B @ (B.T @ FP)
    Q_new, R_new = qr(FP, mode='reduced')
    dr  = np.abs(np.diag(R_new))
    tol = max(1e-6, dr.max() * 1e-6) if dr.size > 0 else 1e-6
    nd  = dr > tol
    max_new   = dim - cur_rank
    eff_chunk = min(int(nd.sum()), max_new)
    if eff_chunk > 0:
        idx       = np.where(nd)[0][:eff_chunk]
        basis     = np.hstack([basis, Q_new[:, idx].astype(np.float32)])
        cur_rank += eff_chunk
    return basis, cur_rank, eff_chunk


def run_cascade_batch(vis, gens, q, sub_batch=400):
    """
    BFS cascade with streaming sub-batch QR rank updates.
    Fingerprint of v: steinberg_vec(v) ∈ R^{(q+1)^2}, computed on the fly.
    Large BFS steps are split into sub-batches to control peak memory.
    """
    id_ = np.array([[1,0],[0,1]], dtype=np.int64)
    ik  = mat_key(id_, q)
    gks = [mat_key(g, q) for g in gens]

    vis_set  = {ik}
    frontier = [ik]

    dim      = (q+1)**2
    basis    = np.zeros((dim, 0), dtype=np.float32)
    cur_rank = 0

    Sn_l, rn_l, pe_l = [], [], []
    pe = 0.0

    while frontier:
        new_front = []
        new_fps   = []

        for uk in frontier:
            u = vis[uk]
            for gk, g in zip(gks, gens):
                v  = mmul(u, g, q)
                vk = mat_key(v, q)
                if vk not in vis_set:
                    vis_set.add(vk)
                    new_front.append(vk)
                    if cur_rank < dim:
                        new_fps.append(steinberg_vec(v, q).astype(np.float32))

        if not new_front:
            break

        raw = len(new_front)
        eff = 0

        # Streaming sub-batch QR
        i = 0
        while i < len(new_fps) and cur_rank < dim:
            chunk = new_fps[i: i + sub_batch]
            basis, cur_rank, eff_c = _rank_update_chunk(basis, cur_rank, chunk, dim)
            eff += eff_c
            i   += sub_batch

        Sn_l.append(len(vis_set))
        rn = eff / max(raw, 1)
        rn_l.append(rn)
        pe += eff
        pe_l.append(pe)
        frontier = new_front

    return {
        'Sn':    np.array(Sn_l),
        'rn':    np.array(rn_l),
        'p_eff': np.array(pe_l),
        'final_rank':  cur_rank,
        'ambient_dim': dim,
    }

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

def sat_threshold(rn, Sn, tol=1e-8):
    for i, r in enumerate(rn):
        if r < tol:
            return int(Sn[i])
    return int(Sn[-1])

def fit_loglog(x, y, frac=0.5):
    n = max(3, int(len(x)*frac))
    m = (x[:n] > 0) & (y[:n] > 0)
    if m.sum() < 3: return np.nan, np.nan
    sl, ic, *_ = linregress(np.log(x[:n][m]), np.log(y[:n][m]))
    return sl, ic

def fit_alpha(rn, p_eff):
    m = (rn > 1e-9) & (p_eff > 0)
    if m.sum() < 3: return np.nan
    sl, *_ = linregress(np.log(p_eff[m]), np.log(rn[m]))
    return -sl

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

P        = 5
Q_VALUES = [29, 41, 61]
results  = {}

print("=" * 70)
print(f"SpectralO6 — Steinberg-matrix fingerprint on X_{{{P},q}}")
print("=" * 70)

for q in Q_VALUES:
    t0 = time.time()
    G_exp = q*(q**2-1)//2
    print(f"\nq = {q}  (|PSL| = {G_exp})")

    gens = lps_generators(P, q)
    print(f"  {len(gens)} generators (det=1 verified)")

    vis = build_psl(P, q, gens)
    print(f"  |G| = {len(vis)}  [{time.time()-t0:.1f}s]")

    print("  Running cascade (batched QR) ...")
    res = run_cascade_batch(vis, gens, q)
    print(f"  Done [{time.time()-t0:.1f}s]")

    Sn, rn, pe = res['Sn'], res['rn'], res['p_eff']
    print(f"  Ambient dim = {res['ambient_dim']},  final span rank = {res['final_rank']}")

    S_star  = sat_threshold(rn, Sn)
    ratio   = S_star / len(vis)
    rq18    = S_star / q**1.8
    sat_idx = int(np.searchsorted(Sn, S_star))
    beta_eff, _ = fit_loglog(Sn[:sat_idx], pe[:sat_idx])
    alpha       = fit_alpha(rn, pe)

    results[q] = dict(G=len(vis), S_star=S_star, ratio=ratio, rq18=rq18,
                      beta_eff=beta_eff, alpha=alpha,
                      Sn=Sn, rn=rn, pe=pe, dim=res['ambient_dim'],
                      rank=res['final_rank'])

    bv = f"{beta_eff:.4f}" if not np.isnan(beta_eff) else "N/A"
    av = f"{alpha:.2f}"    if not np.isnan(alpha)    else "N/A"
    print(f"  |S*|={S_star}, ratio={ratio:.5f}, rq18={rq18:.3f}, "
          f"beta_eff={bv}, alpha={av}")

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

print("\n" + "=" * 70)
print("Table 1 — SpectralO6")
print("=" * 70)
hdr = f"{'q':>4}  {'|G|':>8}  {'dim':>5}  {'rank':>5}  {'|S*|':>6}  "
hdr += f"{'S*/G':>7}  {'S*/q^1.8':>9}  {'beta_eff':>9}  {'alpha':>7}"
print(hdr); print("-"*70)
for q in Q_VALUES:
    if q not in results: print(f"{q:>4}  (failed)"); continue
    r = results[q]
    bv = f"{r['beta_eff']:.4f}" if not np.isnan(r['beta_eff']) else "N/A"
    av = f"{r['alpha']:.2f}"    if not np.isnan(r['alpha'])    else "N/A"
    print(f"{q:>4}  {r['G']:>8}  {r['dim']:>5}  {r['rank']:>5}  "
          f"{r['S_star']:>6}  {r['ratio']:>7.5f}  {r['rq18']:>9.3f}  "
          f"{bv:>9}  {av:>7}")

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = ['tab:blue', 'tab:orange', 'tab:green']
ax_lin, ax_slog, ax_ll, ax_tab = axes.ravel()

for idx, q in enumerate(Q_VALUES):
    if q not in results: continue
    r  = results[q]
    c  = colors[idx]
    lb = f"$q={q}$"
    Sn, rn, pe = r['Sn'], r['rn'], r['pe']
    S_star = r['S_star']

    ax_lin.plot(Sn, rn, color=c, lw=1.5, label=lb)
    ax_lin.axvline(S_star, color=c, ls='--', lw=0.7, alpha=0.45)

    mask = rn > 1e-10
    if mask.any():
        ax_slog.semilogy(Sn[mask], rn[mask], color=c, lw=1.5, label=lb)
        ax_slog.axvline(S_star, color=c, ls='--', lw=0.7, alpha=0.45)

    m2 = (pe > 0) & (Sn > 1)
    if m2.sum() > 3:
        ax_ll.loglog(Sn[m2], pe[m2], color=c, lw=1.5, label=lb)
        si = int(np.searchsorted(Sn, S_star))
        sl, ic = fit_loglog(Sn[:si], pe[:si])
        if not (np.isnan(sl) or np.isnan(ic)):
            xf = np.array([Sn[m2][0], Sn[min(si-1, len(Sn)-1)]])
            ax_ll.loglog(xf, np.exp(ic)*xf**sl, color=c, ls='--', lw=1.1,
                         label=f"$\\beta_{{\\rm eff}}={sl:.3f}$")

for ax, ti, xl, yl in [
    (ax_lin,  "Admissible frontier fraction (linear)",
     r"$|S_n|$",    r"$\tilde r_n^{B,\mathrm{mat}}$"),
    (ax_slog, "Frontier fraction (semi-log)",
     r"$|S_n|$",    r"$\tilde r_n^{B,\mathrm{mat}}$ (log)"),
    (ax_ll,   "Cumulative admissible front (log-log)",
     r"$|S_n|$ (log)", r"$p_n^{\rm eff}$ (log)"),
]:
    ax.set_title(ti, fontsize=10)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(fontsize=8)
ax_lin.set_ylim(-0.05, 1.05)

# Parameter table
ax_tab.axis('off')
col_labs = [r'$q$', r'$|G|$', r'$d_{\rm St}$', r'$|S^*|$',
            r'$|S^*|/|G|$', r'$|S^*|/q^{1.8}$',
            r'$\beta_{\rm eff}$', r'$\alpha$']
tdata = []
for q in Q_VALUES:
    if q not in results: continue
    r = results[q]
    bv = f"{r['beta_eff']:.4f}" if not np.isnan(r['beta_eff']) else "–"
    av = f"{r['alpha']:.1f}"    if not np.isnan(r['alpha'])    else "–"
    tdata.append([str(q), str(r['G']), str(r['dim']),
                  str(r['S_star']), f"{r['ratio']:.5f}",
                  f"{r['rq18']:.3f}", bv, av])

if tdata:
    tbl = ax_tab.table(cellText=tdata, colLabels=col_labs,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.65)
ax_tab.set_title("Parameter stability", pad=12, fontsize=10)

fig.suptitle(
    r"Steinberg-matrix fingerprint on $X_{5,q}$, "
    r"$q\in\{29,41,61\}$ — SpectralO6",
    fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])

out_pdf = "fig_O6_main.pdf"
out_png = "fig_O6_main.png"
fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
fig.savefig(out_png, bbox_inches='tight', dpi=150)
print(f"\nFigure: {out_pdf}")
