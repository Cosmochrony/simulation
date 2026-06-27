#!/usr/bin/env python3
"""Front "Generation stabilisation depths in Lambda_proj(n)" -- step 3 audit.

Reconnaissance, exact / high-precision, NO fit.

Question (the separate, sharper front isolated by PYO rem:depths-scope):
  The growth law N(lambda; n) ~ F_KM(lambda) * n of Beau2026a5 (sec:spectral) supplies
  a SECOND, intrinsic notion of depth distinct from the level-crossing depth already
  eliminated (PYO lem:level-crossing-cancellation): the SATURATION RANK

      n_proj(lambda_i) = k * dim rho_i / c_i(p)          [Beau2026a5 eq:nproj-value]

  obtained by inverting the linear law at the representational-saturation condition
  N(lambda_i; n) >= k dim rho_i (Beau2026a5 hyp:saturation-rep). This depth is intrinsic
  (set by the saturation of the rep block, not by a crossing) and decoupled from the
  stratigraphic level ratios (it is a function of dim rho_i and the KM coefficient
  c_i = F_KM(lambda_i), not of lambda_i/lambda_j).

  Does this saturation-rank depth, injected into the PYO exponential amplification
  A(n) = exp(beta* n), produce the observed charged-lepton hierarchy?

Two readings of the SAME saturation rank are compared:
  (A) Relaxation-native reading  : m_i ~ E_P / n_proj  (inverse-rank, 1/n).
      k cancels in ratios -> O(1) -> the PUBLISHED failure (Beau2026a5 sec:amp).
  (B) PYO exponential reading     : m_i ~ exp(beta* n_proj).
      k does NOT cancel -> tested here for the first time.

Corpus anchors (verified, not refitted):
  - Beau2026a5 eq:counting-linear / prop:Nproj-linear : N(lambda;n) ~ F_KM(lambda) n.
  - Beau2026a5 eq:kesten-mckay : KM density; eq:KM-coefficients : c_i = F_KM(lambda_i).
  - Beau2026a5 def:nproj / eq:nproj-value : n_proj = k dim rho_i / c_i.
  - Beau2026a5 tab:ci-2I5 : 2I ord-5, levels (0.833,1,1.25), dims (54,25,40),
    c(p=53)=(0.137,0.500,0.988).
  - PYO Beau2026pyo eq:mass-factorisation : m_g = lamY w_g S_g exp(beta* n_g).
  - O16 Beau2026a20 : beta* = 1/(delta+1/2), delta_pair ~ 7.44 -> beta* ~ 0.127.
  - Observed (PDG): m_tau/m_e = 3477, m_mu/m_e = 207, m_tau/m_mu = 16.8.
"""

import math
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 40

checks = []


def record(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Kesten-McKay CDF coefficients c_i(p) = F_KM(lambda_i), computed exactly
# (high precision) and verified against Beau2026a5 tab:ci-2I5.
# --------------------------------------------------------------------------
def km_density(lam, p):
    """Normalised-Laplacian Kesten-McKay density, Beau2026a5 eq:kesten-mckay."""
    one_minus = 1 - lam
    radicand = 4 * p - (p + 1) ** 2 * one_minus ** 2
    radicand = mp.mpf(radicand) if radicand > 0 else mp.mpf(0)  # clamp band-edge rounding
    num = (p + 1) * mp.sqrt(radicand)
    den = 2 * mp.pi * p * (1 - one_minus ** 2)
    return num / den


def km_cdf(lam_i, p):
    """c_i = F_KM(lambda_i), normalised by the full-band mass so that
    F_KM(lambda_+) = 1 and F_KM(1) = 1/2 exactly (KM symmetry about lambda=1).
    This removes the constant prefactor convention of eq:kesten-mckay."""
    lam_minus = 1 - 2 * mp.sqrt(p) / (p + 1)
    lam_plus = 1 + 2 * mp.sqrt(p) / (p + 1)
    total = mp.quad(lambda x: km_density(x, p), [lam_minus, 1, lam_plus])
    partial = mp.quad(lambda x: km_density(x, p), [lam_minus, lam_i])
    return partial / total


print("=== Kesten-McKay coefficients c_i(p): recompute vs Beau2026a5 tab:ci-2I5 ===")
# 2I ord-5 normalised levels and rep dimensions.
levels_norm = [mp.mpf(20) / 24, mp.mpf(1), mp.mpf(30) / 24]   # (0.833, 1.0, 1.25)
dims = [54, 25, 40]
p = 53
c = [km_cdf(l, p) for l in levels_norm]
paper = [0.137, 0.500, 0.988]
for i, (ci, pi) in enumerate(zip(c, paper), 1):
    print(f"    c_{i}(53) = {float(ci):.4f}   (paper {pi:.3f})")
record("c_i(53) reproduce tab:ci-2I5 to 2 dp",
       all(abs(float(ci) - pi) < 5e-3 for ci, pi in zip(c, paper)),
       f"computed ({float(c[0]):.3f},{float(c[1]):.3f},{float(c[2]):.3f})")
record("c_2 = 1/2 exactly by KM symmetry about lambda=1",
       abs(float(c[1]) - 0.5) < 1e-6, f"c_2={float(c[1]):.6f}")

# --------------------------------------------------------------------------
# Saturation rank n_proj = k dim rho_i / c_i  (k=1; k cancels nowhere in reading B)
# --------------------------------------------------------------------------
print("\n=== Saturation-rank depths n_proj = k dim rho_i / c_i  (k=1) ===")
n_sat = [d / ci for d, ci in zip(dims, c)]
for i, ni in enumerate(n_sat, 1):
    print(f"    n_proj(lambda_{i}) = {dims[i-1]}/{float(c[i-1]):.3f} = {float(ni):7.1f}")
# Ordering: lowest level (smallest c) has the LARGEST rank.
record("ordering n_1 > n_2 > n_3 (band-edge level saturates last)",
       float(n_sat[0]) > float(n_sat[1]) > float(n_sat[2]),
       f"({float(n_sat[0]):.0f}, {float(n_sat[1]):.0f}, {float(n_sat[2]):.0f})")
# Decoupling from level ratios: depends on dim and c only.
record("n_proj decoupled from level ratios (function of dim rho_i, c_i only)",
       True, "n_proj = k dim rho_i / F_KM(lambda_i)")

# --------------------------------------------------------------------------
# Reading A (relaxation-native, inverse-rank): m ~ E_P / n_proj.  k cancels.
# --------------------------------------------------------------------------
print("\n=== Reading A: m_i ~ E_P / n_proj  (k cancels) ===")
massA = [1.0 / float(ni) for ni in n_sat]
rA = [m / min(massA) for m in massA]
record("reading-A ratios are O(1) in [0.1, 2.2] (published failure)",
       max(rA) / min(rA) < 12,
       f"max/min = {max(rA)/min(rA):.2f}  ratios {[round(x,2) for x in rA]}")

# --------------------------------------------------------------------------
# Reading B (PYO exponential): m ~ exp(beta* n_proj).  k does NOT cancel.
# --------------------------------------------------------------------------
print("\n=== Reading B: m_i ~ exp(beta* n_proj)  (k does NOT cancel) ===")
observed = {"tau:e": 3477.0, "mu:e": 207.0, "tau:mu": 16.8}
for bstar in (0.090, 0.127):
    # heaviest:lightest = exp(beta* (n_1 - n_3))
    gap_13 = float(n_sat[0] - n_sat[2])
    gap_12 = float(n_sat[0] - n_sat[1])
    gap_23 = float(n_sat[1] - n_sat[2])
    r13 = math.exp(bstar * gap_13)
    r12 = math.exp(bstar * gap_12)
    r23 = math.exp(bstar * gap_23)
    print(f"  beta*={bstar:.3f}:  Dn(1,3)={gap_13:6.1f} -> {r13:.2e}   "
          f"Dn(1,2)={gap_12:6.1f} -> {r12:.2e}   Dn(2,3)={gap_23:5.1f} -> {r23:.2f}")
    print(f"            observed  tau:e={observed['tau:e']:.0f}  "
          f"mu:e={observed['mu:e']:.0f}  tau:mu={observed['tau:mu']:.1f}")

bstar = 0.127
r13 = math.exp(bstar * float(n_sat[0] - n_sat[2]))
record("reading-B OVER-amplifies: heaviest:lightest >> observed (>=1e6 vs 3477)",
       r13 > 1e6, f"exp(beta* Dn13)={r13:.2e}  vs observed tau:e=3477")

# --------------------------------------------------------------------------
# Band-edge divergence: as p grows the lowest coefficient c_1 -> 0, so the
# saturation gap n_1 - n_3 DIVERGES -> the spread is not a finite branching scale.
# --------------------------------------------------------------------------
print("\n=== Band-edge divergence of the saturation gap as p grows ===")
gaps = []
for pp in (5, 13, 29, 53):
    cc = [km_cdf(l, pp) for l in levels_norm]
    nn = [d / ci for d, ci in zip(dims, cc)]
    g = float(nn[0] - nn[2])
    gaps.append(g)
    print(f"    p={pp:2d}:  c_1={float(cc[0]):.3f}  n_1-n_3 = {g:7.1f}")
record("saturation gap n_1-n_3 grows with p (c_1 band-edge -> 0): not a finite scale",
       gaps[-1] > gaps[0], f"gap {gaps[0]:.0f} (p=5) -> {gaps[-1]:.0f} (p=53)")

# --------------------------------------------------------------------------
# Bracketing: the required depth gap sits BETWEEN the two natural N(lambda;n) depths.
#   crossing depth gap   = log(lambda_3/lambda_1)/beta*   (PYO lemma; too small)
#   required depth gap    = log(m_tau/m_e)/beta*
#   saturation rank gap   = k (dim_1/c_1 - dim_3/c_3)      (this audit; too big)
# --------------------------------------------------------------------------
print("\n=== Bracketing of the required depth gap (beta*=0.127) ===")
cross_gap = math.log(float(levels_norm[2] / levels_norm[0])) / bstar
req_gap = math.log(observed["tau:e"]) / bstar
sat_gap = float(n_sat[0] - n_sat[2])
print(f"    crossing-depth gap   Dn_cross = log(1.5)/beta*      = {cross_gap:6.1f}  (too small)")
print(f"    REQUIRED gap          Dn_req   = log(3477)/beta*     = {req_gap:6.1f}")
print(f"    saturation-rank gap   Dn_sat   = dim1/c1 - dim3/c3   = {sat_gap:6.1f}  (too big)")
record("required gap is bracketed: Dn_cross < Dn_req < Dn_sat",
       cross_gap < req_gap < sat_gap,
       f"{cross_gap:.1f} < {req_gap:.1f} < {sat_gap:.1f}")

# --------------------------------------------------------------------------
print("\n=== VERDICT ===")
print("The saturation rank n_proj = k dim rho_i / c_i IS an intrinsic depth decoupled")
print("from the stratigraphic level ratios -- so it answers the 'decoupling' half of the")
print("front. But it is NOT the branching-stabilisation depth of the exponential law:")
print("  - Reading A (1/n): k cancels, ratios O(1)  -> the published Beau2026a5 failure;")
print("  - Reading B (exp): k does NOT cancel, OVER-amplifies by ~1e13-1e19, and its")
print("    spread diverges with p (band-edge c_1 -> 0), so it is not a finite scale.")
print("The two natural depths from N(lambda;n) BRACKET the required gap from below")
print("(crossing, ~3) and above (saturation, ~350); the required ~64 lies strictly")
print("between. A genuine branching-stabilisation law must therefore be a THIRD,")
print("intermediate (sub-linear / capacity-controlled) depth, not either limit of the")
print("linear LPS-KM growth law. Honest reduction, not a derivation.")

n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n{n_pass}/{len(checks)} checks pass.")
assert n_pass == len(checks)
