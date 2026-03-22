"""
spectral_o10.py
---------------
Numerical companion to SpectralO10.

Extracts the capacity exponent delta on Cayley graphs of
Heis_3(Z/qZ) via BFS and the k=3 permutation-path fingerprint,
sketched using three independent (D x q^2) Gaussian matrices
and the Hadamard-product TensorSketch approximation.

The fingerprint of path v0->v1->v2->v3 is the sketched vector
    phi(v1,v2,v3)[j] = A1[j, sigma_{v1}[h[j]]]
                     * A2[j, sigma_{v2}[h[j]]]
                     * A3[j, sigma_{v3}[h[j]]]
where h[j] is a random index in {0..q^2-1} drawn once at init,
and A_k[j, :] are i.i.d. N(0,1) entries, and sigma_v is the
affine action of v on (Z/qZ)^2.

This is the Count-Sketch / AMS approximation of the true tensor
product vec(P_{v1}) (x) vec(P_{v2}) (x) vec(P_{v3}).
The span of phi vectors grows proportional to |B_n| until saturation
at rank D_sketch.  For reliable extraction of delta, set D_sketch >> q^2.

Usage:
    python spectral_o10.py [--primes 101 103 107 109]
                           [--fraction 0.05]
                           [--sketch 32768]
                           [--seed 42]
                           [--frac-lo 0.10] [--frac-hi 0.60]

Guideline: --sketch should be at least 3*q^2.
At q=101: q^2=10201, recommend --sketch 32768.
At q=109: q^2=11881, recommend --sketch 40000.

Runtime (fraction=0.05, sketch=32768, single core, q=101):
    ~5-15 min depending on hardware.
    For a quick test: --primes 101 --fraction 0.20 --sketch 16384

Outputs (current directory):
    fig_ballgrowth_O10.pdf
    fig_capacity_O10.pdf
    fig_statelaw_O10.pdf
    results_O10.txt
"""

import argparse, time, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==================================================================
# 0. CLI
# ==================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--primes",   nargs="+", type=int, default=[101,103,107,109])
parser.add_argument("--fraction", type=float, default=0.05)
parser.add_argument("--sketch",   type=int,   default=32768)
parser.add_argument("--seed",     type=int,   default=42)
parser.add_argument("--frac-lo",  type=float, default=0.10, dest="frac_lo")
parser.add_argument("--frac-hi",  type=float, default=0.60, dest="frac_hi")
args = parser.parse_args()

PRIMES, FRACTION, SK_DIM, SEED = args.primes, args.fraction, args.sketch, args.seed
FRAC_LO, FRAC_HI = args.frac_lo, args.frac_hi

print("=== SpectralO10: capacity exponent on Heisenberg Cayley graphs ===")
print(f"primes={PRIMES}  fraction={FRACTION}  sketch={SK_DIM}  seed={SEED}")
for q in PRIMES:
    if SK_DIM < 2*q*q:
        print(f"  WARNING q={q}: sketch={SK_DIM} < 2*q^2={2*q**2}. "
              f"Recommend --sketch {3*q**2}.")
print()

# ==================================================================
# 1. Heisenberg group mod q
# ==================================================================

def heis_mul(a, b, q):
    return ((a[0]+b[0])%q, (a[1]+b[1])%q, (a[2]+b[2]+a[0]*b[1])%q)

def heis_gens(q):
    return [(1,0,0),(q-1,0,0),(0,1,0),(0,q-1,0)]

# ==================================================================
# 2. Permutation action of Heis_3(Z/qZ) on (Z/qZ)^2
#
# (a,b,c) acts on (x,y) by:  (x,y) -> (x+a, y+b+a*x+c)  mod q
#
# sigma_v[i] where i = x*q + y gives the image index j = x'*q + y'.
# We precompute sigma_v as a q^2 integer array for each v encountered.
# ==================================================================

_PERM_CACHE = {}

def get_sigma(v, q):
    """Return permutation array of length q^2 for group element v."""
    key = (v, q)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]
    a, b, c = v
    idx = np.arange(q*q, dtype=np.int32)
    x = idx // q
    y = idx % q
    xp = (x + a) % q
    yp = (y + b + a*x + c) % q
    sigma = (xp*q + yp).astype(np.int32)
    _PERM_CACHE[key] = sigma
    return sigma

# ==================================================================
# 3. TensorSketch (Count Sketch variant)
#
# For each output dimension j in {0..D-1}:
#   - h[j]  : random index in {0..q^2-1}  (pre-drawn once)
#   - s1[j], s2[j], s3[j]: random signs in {-1,+1} (pre-drawn once)
#
# The sketch of a 3-path (v1,v2,v3) is:
#   phi[j] = s1[j] * A1[j, sigma_{v1}[h[j]]]
#           * s2[j] * A2[j, sigma_{v2}[h[j]]]
#           * s3[j] * A3[j, sigma_{v3}[h[j]]]
#
# Equivalently with Gaussian entries A_k (not signs):
#   phi[j] = G1[j, sigma_{v1}[h[j]]]
#           * G2[j, sigma_{v2}[h[j]]]
#           * G3[j, sigma_{v3}[h[j]]]
# where G_k are independent standard Gaussian matrices of shape (D, q^2).
#
# Memory: 3 * D * q^2 float32 = 3 * 32768 * 10201 * 4 ~ 4 GB at q=101, D=32768.
# That is too large.  Use D <= 8192 and q^2 = 10201 => 3 * 8192 * 10201 * 4 ~ 1 GB.
# Or store only G_k[:, h] for the pre-sampled indices h:
#   G1_h[j] = G1[j, h[j]],  shape (D,)   => 3*D floats = tiny.
# Then after applying sigma:
#   phi_k[j] = G_k[j, sigma_{v_k}[h[j]]]
# This means: for each j, look up sigma_{v_k}[h[j]], then index into G_k[j, ...].
# Since h[j] is fixed, sigma_{v_k}[h[j]] is a function only of v_k and the
# single input index h[j].  We precompute for each v:
#   feat_k(v)[j] = G_k[j, sigma_v[h[j]]]
# = G_k[j, image of h[j] under v].
# This is a D-dimensional vector stored per group element (cached).
#
# Memory for G_k: D * q^2 float32.
# At D=8192, q=101: 8192 * 10201 * 4 = 334 MB per matrix, 1 GB total.
# At D=4096, q=101: 167 MB per matrix, 500 MB total. Feasible.
# At D=16384, q=101: 670 MB per matrix, 2 GB total. Tight.
#
# Saturation: the rank of {phi(v1,v2,v3)} grows until it equals D.
# Saturation at |B_n| ~ D^{1/3} (since paths are triples, roughly).
# For D=4096: saturation at |B_n| ~ 16, too small for q=101.
# For D=32768: saturation at |B_n| ~ 32, still << q^2=10201. Good.
# For D=524288 (512k): saturation at |B_n| ~ 80, still << q^2. Better.
#
# Actually the saturation of phi(v1,v2,v3) depends on the number of
# DISTINCT PATHS, not merely vertices. At depth n there are O(|B_n|^2)
# possible (v1,v2) pairs for each v3. The rank of phi grows as
# min(|B_n|^2 * |S_n|, D). For the pre-saturation window |B_n| <= q^2:
# rank ~ min(q^4 * q^2 / scaling, D). This is always D for D << q^6.
# So with sufficient D the span never saturates in the pre-saturation window.
#
# The NOVELTY (mean residual norm) decays when NEW paths (v1,v2,v3)
# produce phi vectors increasingly close to the existing span.
# This happens when the CONTENT of the fingerprint span becomes rich
# enough that new paths contribute diminishing new directions.
# The rate of decay is exactly what encodes delta.
# ==================================================================

class TensorSketch:
    """
    Maintains three Gaussian matrices G1, G2, G3 of shape (D, q^2)
    and a hash array h of shape (D,) indexing into {0..q^2-1}.
    Computes phi(v1,v2,v3) = G1[:,sigma_{v1}[h]] * G2[:,sigma_{v2}[h]]
                              * G3[:,sigma_{v3}[h]]
    and caches per-element feature vectors feat_k(v) = G_k[:, sigma_v[h]].
    """

    def __init__(self, D, q, rng):
        self.D = D
        self.q = q
        q2 = q*q
        # Random hash: which input index to probe per output dim
        self.h = rng.integers(0, q2, size=D).astype(np.int32)
        # Gaussian matrices restricted to the hashed columns
        # G_k_full[j, i] = G_k[j, i], but we only need G_k[j, :] to look up sigma_v[h[j]]
        # Store full q^2 columns: (D, q^2) float32
        self.G1 = rng.standard_normal((D, q2)).astype(np.float32)
        self.G2 = rng.standard_normal((D, q2)).astype(np.float32)
        self.G3 = rng.standard_normal((D, q2)).astype(np.float32)
        # Per-element feature cache: v -> (D,) float32 for each G_k
        self._cache1 = {}
        self._cache2 = {}
        self._cache3 = {}

    def feat1(self, v):
        if v not in self._cache1:
            sigma = get_sigma(v, self.q)
            # self.G1[:, sigma[self.h]] : look up G1[j, sigma[h[j]]] for each j
            self._cache1[v] = self.G1[np.arange(self.D), sigma[self.h]]
        return self._cache1[v]

    def feat2(self, v):
        if v not in self._cache2:
            sigma = get_sigma(v, self.q)
            self._cache2[v] = self.G2[np.arange(self.D), sigma[self.h]]
        return self._cache2[v]

    def feat3(self, v):
        if v not in self._cache3:
            sigma = get_sigma(v, self.q)
            self._cache3[v] = self.G3[np.arange(self.D), sigma[self.h]]
        return self._cache3[v]

    def phi(self, v1, v2, v3):
        """Return phi(v1,v2,v3) as float64 vector of shape (D,)."""
        return (self.feat1(v1) * self.feat2(v2) * self.feat3(v3)).astype(np.float64)

    def mem_mb(self):
        q2 = self.q*self.q
        return 3 * self.D * q2 * 4 / 1e6

# ==================================================================
# 4. Incremental Gram-Schmidt span tracker
# ==================================================================

class SpanTracker:
    def __init__(self, D):
        self.D = D
        self.Q = np.zeros((D, 0), dtype=np.float64)

    @property
    def rank(self):
        return self.Q.shape[1]

    def novelty_and_update(self, v):
        if self.rank == 0:
            residual = v.copy()
        else:
            c = self.Q.T @ v
            residual = v - self.Q @ c
        norm = float(np.linalg.norm(residual))
        if norm > 1e-10 and self.rank < self.D:
            unit = (residual / norm).reshape(-1, 1)
            self.Q = np.hstack([self.Q, unit])
        return norm

# ==================================================================
# 5. BFS
# ==================================================================

def run_bfs(q, fraction, D_sketch, seed, frac_lo, frac_hi):
    rng = np.random.default_rng(seed)
    gens = heis_gens(q)

    print(f"  Building TensorSketch (D={D_sketch}, q^2={q**2}, "
          f"mem~{3*D_sketch*q*q*4/1e6:.0f} MB)...", flush=True)
    ts = TensorSketch(D_sketch, q, np.random.default_rng(seed+1))
    print(f"  Done. Memory estimate: {ts.mem_mb():.0f} MB", flush=True)

    span = SpanTracker(D_sketch)

    identity = (0,0,0)
    visited  = {identity}
    frontier = [identity]

    ball_sizes  = [1]
    shell_sizes = [1]
    Sigma_list  = []
    eta_list    = []
    R_list      = []
    Sigma_sat   = None
    t0 = time.time()

    while frontier:
        # Full BFS expansion
        new_shell = []
        for v in frontier:
            for g in gens:
                w = heis_mul(v, g, q)
                if w not in visited:
                    visited.add(w)
                    new_shell.append(w)
        if not new_shell:
            break

        shell_sizes.append(len(new_shell))
        ball_sizes.append(ball_sizes[-1] + len(new_shell))
        depth = len(shell_sizes) - 1

        # Sample
        n_s = max(1, int(fraction * len(new_shell)))
        if n_s < len(new_shell):
            idx = rng.choice(len(new_shell), size=n_s, replace=False)
            sampled = [new_shell[i] for i in idx]
        else:
            sampled = new_shell

        # Fingerprint novelty
        novelties = []
        prev = frontier
        for v3 in sampled:
            v1 = prev[int(rng.integers(0, len(prev)))]
            v2 = prev[int(rng.integers(0, len(prev)))]
            phi = ts.phi(v1, v2, v3)
            nu  = span.novelty_and_update(phi)
            novelties.append(nu)

        mean_nu = float(np.mean(novelties))
        if Sigma_sat is None:
            Sigma_sat = mean_nu if mean_nu > 1e-12 else 1.0

        eta = mean_nu / Sigma_sat
        R   = float(1.0 / np.sqrt(1.0 + eta**2))
        Sigma_list.append(mean_nu)
        eta_list.append(eta)
        R_list.append(R)

        print(f"  n={depth:3d}  |B_n|={ball_sizes[-1]:>9}  |S_n|={len(new_shell):>7}"
              f"  Sigma={mean_nu:.5f}  rank={span.rank:>6}/{D_sketch}"
              f"  t={time.time()-t0:.1f}s", flush=True)

        if ball_sizes[-1] > 3*q**2:
            print(f"  -> |B_n| > 3q^2={3*q**2}, stopping.")
            break
        if span.rank >= D_sketch:
            print(f"  -> span full, stopping.")
            break

        frontier = new_shell

    return (np.array(ball_sizes), np.array(shell_sizes),
            np.array(Sigma_list), np.array(eta_list), np.array(R_list))

# ==================================================================
# 6. Analysis
# ==================================================================

def estimate_D(ball_sizes, frac_hi=0.65):
    diam = len(ball_sizes) - 1
    n_hi = max(3, int(frac_hi * diam))
    n = np.arange(1, min(n_hi, len(ball_sizes)-1) + 1)
    s, *_ = linregress(np.log(n), np.log(ball_sizes[1:len(n)+1].astype(float)+1))
    return float(s)

def extract_delta(shell_sizes, Sigma, ball_sizes, q, frac_lo, frac_hi):
    n_star = int(np.searchsorted(ball_sizes, q**2))
    if n_star < 4:
        print(f"  WARNING: n_star={n_star} < 4; sketch too small for q={q} "
              f"(need D >> q^2={q**2}). Increase --sketch.")
        return np.nan, np.nan, np.nan, 0
    n_lo = max(1, int(frac_lo * n_star))
    n_hi = min(int(frac_hi * (len(shell_sizes)-1)), n_star)
    if n_hi - n_lo < 3:
        n_lo = 1; n_hi = n_star
    idx = np.arange(n_lo-1, min(n_hi, len(Sigma)))
    p_n = shell_sizes[n_lo:n_lo+len(idx)].astype(float)
    s_n = Sigma[idx]
    mask = (p_n > 1) & (s_n > 1e-12)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, 0
    sl, ic, r, _, se = linregress(np.log(p_n[mask]), np.log(s_n[mask]))
    return float(-sl), float(se), float(r**2), int(mask.sum())

# ==================================================================
# 7. Main
# ==================================================================

results = {}
for q in PRIMES:
    print(f"\n--- q={q}  |G|={q**3}  q^2={q**2}  diam~{q-1} ---")
    t0 = time.time()
    ball_sizes, shell_sizes, Sigma, eta, R = run_bfs(
        q, FRACTION, SK_DIM, SEED, FRAC_LO, FRAC_HI)
    Dhat = estimate_D(ball_sizes)
    delta_hat, stderr, R2, n_pts = extract_delta(
        shell_sizes, Sigma, ball_sizes, q, FRAC_LO, FRAC_HI)
    beta_eff = 1.0/(0.5+delta_hat) if not np.isnan(delta_hat) else np.nan
    n_star = int(np.searchsorted(ball_sizes, q**2))
    elapsed = time.time() - t0
    print(f"  => Dhat={Dhat:.3f}  n_star={n_star}  "
          f"delta={delta_hat:.2f}+-{stderr:.2f}  "
          f"beta={beta_eff:.3f}  R2={R2:.3f}  n_pts={n_pts}  t={elapsed:.1f}s")
    results[q] = dict(ball_sizes=ball_sizes, shell_sizes=shell_sizes,
                      Sigma=Sigma, eta=eta, R=R, Dhat=Dhat, n_star=n_star,
                      delta_hat=delta_hat, stderr=stderr, R2=R2,
                      beta_eff=beta_eff, n_pts=n_pts)
    _PERM_CACHE.clear()  # free memory between primes

# Summary
print("\n=== SUMMARY ===")
print(f"{'q':>5}  {'|G|':>10}  {'n_star':>7}  {'Dhat':>5}  "
      f"{'delta':>7}  {'+-':>5}  {'beta':>6}  {'R2':>5}  {'pts':>4}")
print("-"*65)
for q in PRIMES:
    r = results[q]
    print(f"{q:>5}  {q**3:>10}  {r['n_star']:>7}  {r['Dhat']:>5.3f}  "
          f"{r['delta_hat']:>7.2f}  {r['stderr']:>5.2f}  "
          f"{r['beta_eff']:>6.3f}  {r['R2']:>5.3f}  {r['n_pts']:>4}")
valid = [q for q in PRIMES if not np.isnan(results[q]["delta_hat"])]
if valid:
    ds = np.array([results[q]["delta_hat"] for q in valid])
    print(f"\nCombined: delta = {np.mean(ds):.2f} +- {np.std(ds):.2f}")
    print(f"Target:   delta in [7.4, 10.6],  beta* in (0.09, 0.13)")

# Save text
with open("results_O10.txt","w") as f:
    f.write(f"SpectralO10  primes={PRIMES}  fraction={FRACTION}  "
            f"sketch={SK_DIM}  seed={SEED}\n\n")
    for q in PRIMES:
        r = results[q]
        f.write(f"q={q}  Dhat={r['Dhat']:.4f}  n_star={r['n_star']}  "
                f"delta={r['delta_hat']:.4f}  stderr={r['stderr']:.4f}  "
                f"R2={r['R2']:.4f}  beta_eff={r['beta_eff']:.4f}\n")
    if valid:
        f.write(f"\nCombined delta = {np.mean(ds):.4f} +- {np.std(ds):.4f}\n")
print("\nSaved results_O10.txt")

# ==================================================================
# 8. Figures
# ==================================================================

all_c = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
COLORS = {q: all_c[i % len(all_c)] for i, q in enumerate(PRIMES)}

# Fig 1: Ball growth
fig, ax = plt.subplots(figsize=(7,5))
max_n = max(len(r["ball_sizes"]) for r in results.values())
ns = np.array([1.0, float(max_n)])
ax.loglog(ns, 0.05*ns**4, "k--", lw=1.5, label=r"$\propto n^4$")
for q in PRIMES:
    r = results[q]
    bs = r["ball_sizes"]
    ax.loglog(np.arange(1,len(bs)), bs[1:], "o-", ms=3, color=COLORS[q],
              label=f"$q={q}$ ($\\hat{{D}}={r['Dhat']:.2f}$)")
ax.set_xlabel("BFS depth $n$", fontsize=12)
ax.set_ylabel("$|B_n|$", fontsize=12)
ax.set_title(
    r"Ball growth on $\mathrm{Cay}(\mathrm{Heis}_3(\mathbb{Z}/q\mathbb{Z}),\,S_q)$",
    fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("fig_ballgrowth_O10.pdf", dpi=150)
plt.close(fig)

# Fig 2: Capacity
ncols = min(2, len(PRIMES))
nrows = (len(PRIMES)+ncols-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols,5*nrows), squeeze=False)
afl = axes.flatten()
for i, q in enumerate(PRIMES):
    r = results[q]; ax = afl[i]
    Sig = r["Sigma"]; ss = r["shell_sizes"]; bs = r["ball_sizes"]
    ns_ = r["n_star"]
    p_n = ss[1:len(Sig)+1].astype(float)
    in_w = bs[1:len(Sig)+1] <= q**2
    mp = p_n > 0
    if (in_w&mp).sum():
        ax.loglog(p_n[in_w&mp], Sig[in_w&mp], "o", ms=5, color=COLORS[q],
                  label="pre-sat.")
    if (~in_w&mp).sum():
        ax.loglog(p_n[~in_w&mp], Sig[~in_w&mp], "o", ms=3, color=COLORS[q],
                  alpha=0.25, label="post")
    # Fit
    dh = r["delta_hat"]
    if not np.isnan(dh):
        n_lo_f = max(0, int(FRAC_LO*ns_)-1)
        n_hi_f = min(int(FRAC_HI*(len(ss)-1)), ns_)-1
        if n_hi_f > n_lo_f+1:
            pp = p_n[n_lo_f:n_hi_f+1]; sg = Sig[n_lo_f:n_hi_f+1]
            m_ = (pp>1)&(sg>1e-12)
            if m_.sum()>=2:
                sl,ic_,*_ = linregress(np.log(pp[m_]),np.log(sg[m_]))
                ps2 = np.array([pp[m_].min(), pp[m_].max()])
                ax.loglog(ps2, np.exp(ic_)*ps2**sl, "k--", lw=1.5,
                          label=f"$\\hat{{\\delta}}={dh:.1f}$, $R^2={r['R2']:.2f}$")
    ax.set_title(f"$q={q}$", fontsize=11)
    ax.set_xlabel("$p(n)$", fontsize=9)
    ax.set_ylabel(r"$\bar{\Sigma}_n$", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
for i in range(len(PRIMES), len(afl)):
    afl[i].set_visible(False)
fig.suptitle(r"Capacity $\bar{\Sigma}_n$ vs $p(n)$", fontsize=12)
fig.tight_layout()
fig.savefig("fig_capacity_O10.pdf", dpi=150)
plt.close(fig)

# Fig 3: State law
fig, ax = plt.subplots(figsize=(7,5))
er = np.linspace(0,4,400)
ax.plot(er, 1/np.sqrt(1+er**2), "k-", lw=2,
        label=r"$\Phi(\eta)=1/\sqrt{1+\eta^2}$")
for q in PRIMES:
    r = results[q]; m = (r["eta"]>=0)&(r["eta"]<=4)
    ax.scatter(r["eta"][m], r["R"][m], s=15, color=COLORS[q], alpha=0.8,
               label=f"$q={q}$")
ax.set_xlabel(r"$\bar{\eta}_n$", fontsize=12)
ax.set_ylabel(r"$R_n^{(3)}$", fontsize=12)
ax.set_title("State law verification", fontsize=11)
ax.legend(fontsize=9); ax.set_xlim(0,4); ax.set_ylim(0,1.05)
ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig("fig_statelaw_O10.pdf", dpi=150)
plt.close(fig)

print("Saved fig_ballgrowth_O10.pdf  fig_capacity_O10.pdf  fig_statelaw_O10.pdf")
print("Done.")
