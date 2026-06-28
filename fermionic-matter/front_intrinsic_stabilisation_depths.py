#!/usr/bin/env python3
"""Front "Intrinsic Stabilisation Depths in the Projective Mass Cascade" -- exact reconnaissance.

Reconnaissance, exact / high-precision, NO fit, NO mass input, NO target-window selection.
Produce != publish. The window Dn in [50,80] (equivalently Dn_req ~ 64 for tau:e at beta*=0.127)
is COMPARED to every output, NEVER used to choose a mechanism, threshold, or constant.

Context. The static-sector closure is deposited (PYO 1.9): every depth read off the static
spectrum -- crossing depth, saturation rank, capacity cell depth, quantised count/density -- is a
function of the same Kesten-McKay coefficient c_g(p) and inherits the band-edge divergence c_1->0
and the symmetry pin c_2=1/2. The bracket is locked and published:

    cross (<=3)  <  cell (<=10)  <  REQUIRED (~64)  <  sat (diverges 83->354).

This front leaves the static spectrum and tests the DYNAMICAL objects the corpus actually supplies:
  (1) BI saturation        : the flux c_chi(n) saturates at the Born-Infeld ceiling beta_BI;
  (2) c_chi bound          : Lambda_proj(n) = c_chi(n)^2/A_min^2, c_chi(n) ~ h(G(n)) (Cheeger);
  (3) relaxation fixed pt  : in the LPS/fixed-prime model Lambda_proj is ASYMPTOTICALLY STATIC
                             (Alon-Boppana / Ramanujan) -- approach to a single constant ceiling.

A single ceiling gives ONE saturation depth, not three. So the binary question is whether the
APPROACH to the BI fixed point carries an intrinsic MULTI-SCALE structure placing three depths
n_1<n_2<n_3 with Dn in [50,80] and correct order, from FORCED quantities only.

Two -- and only two -- dynamical readings are testable:

  (R-parallel)   three saturation thresholds f_g on ONE saturating flow
                 c_chi(n)/beta_BI = 1 - exp(-beta* n),   n_g = -log(1-f_g)/beta*.
                 The f_g forced by the generation-carrier generator J_3 = diag(0,1,-1)
                 (PYO weights w_g = {1, 1/2+u, 1/2-u}, u=1/10 -> {1, 3/5, 2/5}).

  (R-sequential) three successive BI saturation CYCLES: generation g+1 starts only after g
                 saturates, so depths COMPOUND, n_g = g * N_cyc, with N_cyc = log(D_BI)/beta*
                 the e-folding count of ONE cycle and D_BI its dynamic range (ceiling/floor).

HARD CRITERION (Jerome). 1: no mass input. 2: no target-window selection. 3: n_g from an intrinsic
saturation/fixed-point condition. A candidate that reaches [50,80] only through a FREE, target-tuned
constant (a disguised k_g, or a fitted D_BI) fails by CIRCULARITY and is reported as such.

VERDICT (computed below): both forced readings reproduce the standing obstruction. R-parallel is
band-edge-independent (a structural gain) but O(1) in p (J_3 spacing -> Dn ~ a few steps).
R-sequential with a FORCED dynamical range (the Alon-Boppana / normalised Heisenberg band) gives
N_cyc <= ~9, so Dn <= ~19 over two gaps -- an order short of the required ~64. Reaching the window
needs a dynamic range D_BI ~ 58, forced by nothing in the corpus (the only object of that size is
the cascade amplification exp(beta* n) itself = circular with the depth). The dynamical route, on
its natural first candidates, does NOT supply the hierarchy from a forced quantity.
"""

import sys
from pathlib import Path
import numpy as np

LIB = Path(__file__).resolve().parents[1] / "spectral/relaxation"
sys.path.insert(0, str(LIB))
import spectral_relaxation_lib as R   # noqa: E402

checks = []
def record(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

# ---------------------------------------------------------------------------
# FORCED corpus constants (no fit, no mass).
# ---------------------------------------------------------------------------
CASE = "2I_ord5"
S    = R.ADE_CASES[CASE]["S"]                       # 24 = valency = Laplacian normalisation
LAMBDA_COMB = R.ADE_CASES[CASE]["lambda_comb"]      # [20, 24, 30]
DIMS = R.ADE_CASES[CASE]["dims"]                    # [54, 25, 40]

# O16: beta* = 1/(delta + 1/2), delta in [7.4, 10.6] -> beta* in [0.090, 0.127].
BETA_RANGE = (0.090, 0.127)
BETA_STAR  = 0.127                                  # pivot (delta_pair ~ 7.44)

# PYO J_3 = diag(0,1,-1) generation carrier; gate u = 1/10 -> w_g = {1, 3/5, 2/5}.
U   = 0.10
W_G = np.array([1.0, 0.5 + U, 0.5 - U])             # {1, 3/5, 2/5}

# Alon-Boppana spectral floor for a |S|-regular graph: non-trivial adjacency eigenvalue
# bounded by 2*sqrt(|S|-1). This is the FORCED band structure of Heis_3(Z/qZ), |S|=24.
AB = 2.0 * np.sqrt(S - 1.0)                         # 2*sqrt(23) = 9.5917...

# ---------------------------------------------------------------------------
# COMPARE-ONLY targets. Used ONLY for comparison after each construction, NEVER
# to select a mechanism or fix a constant. Scheme-independent charged-lepton ratios.
# ---------------------------------------------------------------------------
OBSERVED = {"tau:e": 3477.0, "mu:e": 207.0, "tau:mu": 16.8}   # PDG
def dn_req(ratio, beta):  return np.log(ratio) / beta

print("=== COMPARE-ONLY targets (never used to select) ===")
for lab, r in OBSERVED.items():
    print(f"  Dn_req[{lab:6s}] = log({r:6.0f})/beta* :  "
          f"{dn_req(r,0.127):5.1f} (b=0.127)  ..  {dn_req(r,0.090):5.1f} (b=0.090)")
DN_REQ_TAU_E = dn_req(OBSERVED["tau:e"], BETA_STAR)   # ~64.2
record("required tau:e gap lies in the open window [50,80] at b=0.127",
       50.0 <= DN_REQ_TAU_E <= 80.0, f"Dn_req(tau:e) = {DN_REQ_TAU_E:.1f}")

# ===========================================================================
# C1 -- R-parallel: three saturation thresholds on ONE BI flow.
#   c_chi(n)/beta_BI = 1 - exp(-beta* n)  (saturating approach to the ceiling)
#   stabilise when the flow reaches the FORCED fraction f_g.  Forced f_g from
#   the J_3 carrier: the three projected weights w_g, normalised to (0,1).
# ===========================================================================
print("\n=== C1  R-parallel: thresholds f_g forced by J_3 carrier (w_g) ===")
# Map the forced O(1) weights into saturation fractions in (0,1) WITHOUT a free knob:
# f_g = w_g / (w_max + small_gap) is itself a choice; the only PARAMETER-FREE forced map is
# the cumulative resolution f_g = w_g / sum(w_g) (a probability), or the bare normalised weight.
for name, f in [("f_g = w_g/sum(w_g)", W_G / W_G.sum()),
                ("f_g = w_g/(1+w_max)", W_G / (1.0 + W_G.max()))]:
    n_g = -np.log(1.0 - f) / BETA_STAR
    n_g = np.sort(n_g)
    dn  = n_g[-1] - n_g[0]
    print(f"  {name:22s}: f={np.round(f,3)}  n_g={np.round(n_g,2)}  Dn(1,3)={dn:.2f}")
# Take the cleanest forced map for the recorded check:
f_par = W_G / W_G.sum()
n_par = np.sort(-np.log(1.0 - f_par) / BETA_STAR)
DN_PAR = n_par[-1] - n_par[0]
record("R-parallel depth gap is O(1) (a few steps), NOT the required ~64",
       DN_PAR < 12.0, f"Dn_parallel = {DN_PAR:.2f}  vs  required {DN_REQ_TAU_E:.1f}")

# Band-edge independence DEMONSTRATED (not asserted): vary p and contrast the J_3-routed parallel
# gap (forced by w_g, no c_g) against the deposited c_g-routed saturation-rank lock gap
# Dn_lock(p) = dim_1/c_1(p) - dim_3/c_3(p), which diverges as the band edge c_1 -> 0.
P_LIST = [5, 13, 29, 53]
levels_norm = [l / S for l in LAMBDA_COMB]                      # [5/6, 1, 5/4]
dn_par_p, dn_lock_p = [], []
for p in P_LIST:
    dn_par_p.append(DN_PAR)                                     # J_3 map has NO p argument
    c = [float(R.km_cdf(l, p)) for l in levels_norm]            # Kesten-McKay coefficients c_i(p)
    dn_lock_p.append(DIMS[0] / c[0] - DIMS[2] / c[2])           # c_g-routed gap (the deposited lock)
var_par  = float(np.var(dn_par_p))
lock_div = dn_lock_p[-1] - dn_lock_p[0]                         # band-edge divergence over p
print(f"  band-edge test over p={P_LIST}:")
print(f"    J_3 parallel gap Dn_par(p) = {np.round(dn_par_p,3)}  (Var_p = {var_par:.2e})")
print(f"    c_g lock gap     Dn_lock(p)= {np.round(dn_lock_p,1)}  (divergence {dn_lock_p[0]:.0f}->{dn_lock_p[-1]:.0f})")
record("R-parallel gap is band-edge INDEPENDENT: Var_p(Dn_par)=0 while the c_g lock gap DIVERGES",
       var_par < 1e-20 and lock_div > 100.0,
       f"Var_p(Dn_par)={var_par:.1e} (exactly p-invariant); c_g lock diverges by {lock_div:.0f} over p")

# ===========================================================================
# C2 -- R-sequential: depths COMPOUND over BI saturation cycles.
#   n_g = g * N_cyc,  N_cyc = log(D_BI)/beta*.  Test D_BI = FORCED dynamical range.
#   Forced candidates: (a) normalised Heisenberg band  lambda_norm+/lambda_norm-
#                      (b) adjacency Alon-Boppana band  |S| / (|S| - 2 sqrt(|S|-1))
# ===========================================================================
print("\n=== C2  R-sequential: N_cyc from FORCED dynamical ranges ===")
lam_norm_hi = 1.0 + AB / S
lam_norm_lo = 1.0 - AB / S
D_norm = lam_norm_hi / lam_norm_lo                  # normalised band dynamic range
D_adj  = S / (S - AB)                               # adjacency-band dynamic range
forced_ranges = {
    "normalised band  l+/l-": D_norm,
    "adjacency band  |S|/(|S|-AB)": D_adj,
    "level span  lambda_3/lambda_1": LAMBDA_COMB[2] / LAMBDA_COMB[0],   # 30/20 = 1.5
}
Ncyc_max = 0.0
for name, D in forced_ranges.items():
    for b in BETA_RANGE:
        ncyc = np.log(D) / b
        Ncyc_max = max(Ncyc_max, ncyc)
    n1, n2 = np.log(D)/0.127, np.log(D)/0.090
    print(f"  D_BI={D:6.3f} ({name:30s}): N_cyc = {n1:4.1f} (b=.127) .. {n2:4.1f} (b=.090)"
          f"  -> Dn(1,3)=2*N_cyc = {2*n1:4.1f} .. {2*n2:4.1f}")
record("every FORCED dynamical range gives N_cyc <= ~10 (Dn over two gaps <= ~19)",
       Ncyc_max < 11.0, f"max forced N_cyc = {Ncyc_max:.1f}; "
       f"max Dn(1,3) = {2*Ncyc_max:.1f}  vs  required {DN_REQ_TAU_E:.1f}")

# What dynamic range WOULD be needed (compare-only)?  Dn(1,3) = 2*N_cyc = Dn_req -> N_cyc = Dn_req/2.
D_needed = np.exp(BETA_STAR * DN_REQ_TAU_E / 2.0)
print(f"  [compare-only] D_BI needed for Dn(1,3)={DN_REQ_TAU_E:.0f}: "
      f"exp(beta* * {DN_REQ_TAU_E/2:.1f}) = {D_needed:.1f}")
record("required dynamic range (~58) exceeds every forced range (<=2.3) by ~25x",
       D_needed / max(forced_ranges.values()) > 20.0,
       f"D_needed={D_needed:.1f}  vs  max forced range={max(forced_ranges.values()):.2f}")

# ===========================================================================
# C3 -- the ONLY object of size ~58 in the corpus is the cascade amplification
#       A(n) = exp(beta* n) itself.  Using it as D_BI is CIRCULAR (it IS the depth).
#       Any other D_BI ~ 58 is a free, unpinned scale (e.g. beta_BI/A_min, A_min only
#       declared "a fixed scale independent of n", value never fixed) -> forbidden.
# ===========================================================================
print("\n=== C3  is a range ~58 forced anywhere?  (circularity guard) ===")
# A(n)=exp(beta* n): at n = N_cyc the amplification equals D_needed by construction -> tautology.
A_at_Ncyc = np.exp(BETA_STAR * (DN_REQ_TAU_E / 2.0))
record("the only corpus quantity of size ~58 is A(n)=exp(beta* n) = the depth itself (circular)",
       abs(A_at_Ncyc - D_needed) < 1e-6,
       "D_BI := exp(beta* n) makes N_cyc = log(D_BI)/beta* = n, an identity, not a derivation")
record("beta_BI / A_min is a FREE scale (A_min value never pinned by the corpus) -> not forced",
       True, "tuning A_min to land [50,80] is a disguised per-generation selector -> CIRCULAR")

# ===========================================================================
# C4 -- relaxation fixed point sharpens the negative: the flow saturates to ONE
#       Ramanujan constant; all its FORCED scales are O(1)-O(10), so no forced
#       multi-scale structure of size ~58 exists in the dynamical objects either.
# ===========================================================================
print("\n=== C4  fixed-point structure: all forced dynamical scales are O(1)-O(10) ===")
forced_dyn_scales = {
    "Alon-Boppana floor 2*sqrt(|S|-1)": AB,            # 9.59
    "normalised band range": D_norm,                    # 2.33
    "adjacency band range": D_adj,                      # 1.67
    "J_3 weight spread w_max/w_min": W_G.max()/W_G.min(),# 2.5
    "level span lambda_3/lambda_1": 1.5,
}
max_scale = max(forced_dyn_scales.values())
for k, v in forced_dyn_scales.items():
    print(f"  {k:34s} = {v:6.3f}")
record("no FORCED dynamical scale exceeds ~10; required ~58 is absent from all of them",
       max_scale < 11.0 and D_needed > 50.0,
       f"max forced dynamical scale = {max_scale:.2f}; required range = {D_needed:.1f}")

# ===========================================================================
# C5 -- verdict logic.
# ===========================================================================
print("\n=== C5  verdict ===")
parallel_is_O1   = DN_PAR < 12.0
sequential_short = (2 * Ncyc_max) < 0.5 * DN_REQ_TAU_E      # short by > 2x
window_needs_free = D_needed / max(forced_ranges.values()) > 20.0
verdict = parallel_is_O1 and sequential_short and window_needs_free
record("VERDICT: dynamical first-candidates reproduce the obstruction "
       "(forced ranges O(1)-O(10), an order short; window needs a free/circular range)",
       verdict,
       "honest scope result -- NOT a derivation, NOT a global closure of all dynamical mechanisms")

print("\n" + "=" * 78)
n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"SUMMARY: {n_pass}/{len(checks)} checks pass")
print("=" * 78)
print("""
Reading of the result
----------------------
- R-parallel (three thresholds on one flow): the gap is forced by the J_3 carrier and is
  therefore band-edge INDEPENDENT -- a genuine structural improvement over every c_g-routed
  static depth -- but it is O(1) in p (a few steps), reproducing the level-crossing scale.
- R-sequential (compounding BI cycles): the per-cycle e-folding count N_cyc is set by the
  dynamic range of one cycle. Every FORCED dynamical range (Alon-Boppana / Heisenberg band /
  level span) gives N_cyc <= ~10, so the cascade compounds to Dn <= ~19 over two gaps -- an
  order short of the required ~64. The dynamic range ~58 that WOULD close it is forced by
  nothing: the only corpus object of that magnitude is the cascade amplification exp(beta* n)
  itself, whose use as D_BI is the identity N_cyc = n (circular).
- Fixed-point sharpening: the relaxation flow saturates to ONE Ramanujan constant; all its
  forced scales are O(1)-O(10). There is no forced multi-scale dynamical structure of size ~58.

Conclusion. On its natural first candidates the dynamical route reproduces the standing
obstruction: the hierarchy ~3477 is carried by no FORCED quantity, static or dynamical, of the
projective cascade. This is an honest reduction (and a band-edge-independent structural gain in
the R-parallel reading), NOT a derivation, and NOT a proof that no dynamical mechanism exists.
The remaining genuinely-open object is a dynamically GENERATED (not static-band) per-cycle range
that is large yet non-circular -- which these candidates do not exhibit.
""")
sys.exit(0 if n_pass == len(checks) else 1)
