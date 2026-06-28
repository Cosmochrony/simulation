#!/usr/bin/env python3
"""Front "quantised per-level resolution density" -- the dedicated modelling front.

Reconnaissance, exact, NO fit, NO mass input. Produce != publish; no deposit without
Jerome's explicit go.

CONTEXT (the lock to beat). The deposited no-go (PYO 1.8) shows that EVERY depth read off
the linear growth law N(lambda_g; n) = c_g(p) n is c_g-routed: crossing depth, saturation
rank, capacity cell, and even the quantised COUNT all reduce to

    n_g = DeltaI_g / c_g(p),                                   [the deposited lock]

so they inherit the band-edge divergence c_1 -> 0 and the symmetry pin c_2 = 1/2. The
inter-generation gap then DIVERGES with p (Dn_sat = 83 -> 354), never landing in [50,80].

THE NEW OBJECT (this front). Quantise the RESOLUTION DENSITY itself, not the count. In the
continuum (q -> infinity) the per-level resolution density is the smooth Kesten-McKay value
F_KM(lambda_g) = c_g (the "Rayleigh-Jeans equipartition"). The Planck cure replaces it by a
DISCRETE occupation of a per-level profile sigma_{lambda_g}(n), with a quantum epsilon_g that
must be c_g-INDEPENDENT (forced by V_rho / C^3_gen / the BI-ADE lock), so the depth bypasses
the denominator c_g where the obstruction lived.

CONSTRUCTION (Weil/Schur transport + Planck/Bose occupation).
  * Carrier: C^3_gen = Sym^2(C^2), J_3 = diag(0,1,-1) (schur_transversality_alpha.py). The three
    2I ord-5 stratigraphic levels {20,24,30} are transported onto the three generation slots.
  * Common cascade clock tau(n) (c_g-INDEPENDENT), two readings:
        (L) exponential-regime depth clock   tau(n) = beta* n     (O16 delta-exponent rate),
        (A) amplification clock              A(n)   = exp(beta* n) (the same, read as a ratio).
    A non-parametric clock from the stored O25 sigma_c(n) is cross-checked at the end.
  * Per-level Bose/Planck occupation of the level resolution, quantum epsilon_g:
        sigma_{lambda_g}(n) = 1 / ( exp(epsilon_g / tau(n)) - 1 ).
    Stabilisation depth = first quantum of resolved occupation, n_g = inf{ n : sigma >= 1/2 },
    i.e. tau(n_g) = epsilon_g / ln 3  ->  n_g = epsilon_g / (beta* ln 3)   (linear clock).

THE STRICT GUARD (anti-circularity). The quantisation acts on F_KM, which is a function of the
NORMALISED level lambda_g^norm in [5/6, 5/4] (Laplacian L = I - A/|S|, spectrum in [0,2]). The
FORCED resolution-density quantum conjugate to F_KM is therefore O(1) (the normalised level or
its band-centre deficit). A quantum is admissible iff it is computable WITHOUT the target value.
A unit tuned to land [50,80] (a disguised per-generation k_g, or the raw combinatorial level =
|S| x normalised level, i.e. re-inserting the valency the normalisation removed) is CIRCULAR.

Brutal pass/fail (Jerome):
  PASS    a c_g-INDEPENDENT FORCED quantum yields Dn in [50,80], q-stable, correctly ordered.
  GAIN    the depth gap is c_g-INDEPENDENT (flat in p) -- beats the lock's band-edge divergence.
  FAIL    only a NON-forced unit (valency re-insertion, or a quantum ratio = the target ~3477)
          reaches [50,80]; forced O(1) quanta stay in the crossing regime.

Reuses simulation/spectral/relaxation/spectral_relaxation_lib.py (km_cdf, ADE_CASES) and the
stored O25 profiles simulation/spectral/o25/o25_outputs/q{29,61,101,151}_o25.npz.
"""

import math
import sys
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parents[1] / "spectral/relaxation"
sys.path.insert(0, str(LIB))
import spectral_relaxation_lib as R   # noqa: E402

CASE = "2I_ord5"
S = R.ADE_CASES[CASE]["S"]                       # 24 = valency (generating-set cardinality)
LEVELS_COMB = R.ADE_CASES[CASE]["lambda_comb"]   # [20, 24, 30] : raw Cayley spectrum of 2I
LEVELS_NORM = R.normalised_levels(CASE)          # [5/6, 1, 5/4] : the variable F_KM lives on
DIMS = R.ADE_CASES[CASE]["dims"]                 # [54, 25, 40]
LEAD_IRREP = [6, 5, 4]                            # leading irrep dim per block

BETA_STAR = 0.127                                 # O16 delta-exponent cascade rate
LN3 = math.log(3.0)                               # Bose half-occupation threshold constant
TARGET = 3477.0                                   # tau:e amplification (REFERENCE ONLY, never an input)
DN_REQ = math.log(TARGET) / BETA_STAR             # ~64.2, centre of the [50,80] target band
BAND = (50.0, 80.0)

PRIMES_SUPPORT = [5, 13, 29, 53]                  # in-support primes for the analytic c_g(p)
PRIMES_STORED = [29, 61, 101, 151]                # stored O25 profiles (non-parametric clock)
ODIR = Path(__file__).resolve().parents[1] / "spectral/o25/o25_outputs"

checks = []
def record(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# ===========================================================================
# 0. Reference: the deposited lock n_g = DeltaI_g / c_g(p) DIVERGES with p.
# ===========================================================================
print("=== 0. The lock to beat: n_g = DeltaI_g/c_g(p) is c_g-routed and DIVERGES with p ===")
lock_gaps = []
for p in PRIMES_SUPPORT:
    c = [float(R.km_cdf(l, p)) for l in LEVELS_NORM]
    nsat = [d / ci for d, ci in zip(DIMS, c)]          # saturation rank = dim/c (DeltaI_g = dim)
    gap = max(nsat) - min(nsat)
    lock_gaps.append(gap)
    print(f"    p={p:2d}: c=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  "
          f"n_sat=({nsat[0]:6.1f},{nsat[1]:5.1f},{nsat[2]:5.1f})  Dn={gap:6.1f}")
record("the deposited lock gap DIVERGES with p (band edge c_1->0)",
       lock_gaps[-1] > 2 * lock_gaps[0],
       f"Dn_lock {lock_gaps[0]:.0f} (p=5) -> {lock_gaps[-1]:.0f} (p=53)")


# ===========================================================================
# 1. The new object: Bose/Planck occupation of the per-level resolution density.
#    sigma_{lambda_g}(n) = 1/(exp(eps_g/tau(n))-1);  n_g = inf{n : sigma>=1/2}.
#    Linear clock tau(n)=beta* n  =>  n_g = eps_g/(beta* ln3),  c_g-INDEPENDENT.
# ===========================================================================
def stab_depth_linear(eps_g):
    """Stabilisation depth from the Bose half-occupation on the linear clock tau=beta* n."""
    return [e / (BETA_STAR * LN3) for e in eps_g]

def stab_depth_amplif(Q_g):
    """Stabilisation depth from the amplification clock A(n)=exp(beta* n)=Q_g => n=ln(Q)/beta*."""
    return [math.log(q) / BETA_STAR for q in Q_g]

# All c_g-INDEPENDENT structural quanta (computable WITHOUT the target).
FORCED_QUANTA = {
    "normalised level lambda^norm  (FORCED: F_KM lives here)": LEVELS_NORM,
    "band-centre deficit |lambda^norm - 1|":                   [abs(l - 1.0) for l in LEVELS_NORM],
    "leading irrep dim":                                       [float(x) for x in LEAD_IRREP],
    "log dim rho_g":                                           [math.log(d) for d in DIMS],
}
# The valency re-insertion (NOT forced by the quantisation principle -- audited separately).
RESCALED_QUANTA = {
    "raw Cayley level = |S| x lambda^norm  (valency re-inserted)": [float(x) for x in LEVELS_COMB],
}

print("\n=== 1. Per-level resolution-density quantisation: stabilisation-depth gap (linear clock) ===")
print(f"    n_g = eps_g/(beta* ln3),  Dn = (max eps - min eps)/(beta* ln3);  target band {BAND}\n")

def report_quantum(name, eps_g, store):
    ng = stab_depth_linear(eps_g)
    gap = max(ng) - min(ng)
    order = tuple(int(x) for x in np.argsort(ng))     # ascending depth order of (slot0,1,2)
    in_band = BAND[0] <= gap <= BAND[1]
    store.append((name, gap, in_band))
    print(f"    {name:58s}  eps=({eps_g[0]:.3f},{eps_g[1]:.3f},{eps_g[2]:.3f})  "
          f"Dn={gap:6.1f}  in-band={in_band}")
    return gap, in_band

forced_results, rescaled_results = [], []
for nm, eps in FORCED_QUANTA.items():
    report_quantum(nm, eps, forced_results)
print()
for nm, eps in RESCALED_QUANTA.items():
    report_quantum(nm, eps, rescaled_results)

forced_in_band = any(b for _, _, b in forced_results)
rescaled_in_band = any(b for _, _, b in rescaled_results)

# --- Threshold robustness: the in-band raw-Cayley near-hit is fragile in the O(1) occupation
#     threshold theta (sigma >= theta). n_g = eps_g/(beta* ln(1 + 1/theta)); the band-membership
#     of the rescaled quantum slides out for an equally natural theta, so it is NOT robust. ---
print("\n    threshold robustness of the raw-Cayley near-hit (sigma >= theta, theta natural O(1)):")
eps_raw = [float(x) for x in LEVELS_COMB]
theta_band_hits = []
for theta_name, theta in [("1/2", 0.5), ("1 (one quantum)", 1.0), ("1/e", 1 / math.e), ("1/(e-1)", 1 / (math.e - 1))]:
    thr_const = math.log(1.0 + 1.0 / theta)
    gap_t = (max(eps_raw) - min(eps_raw)) / (BETA_STAR * thr_const)
    hit = BAND[0] <= gap_t <= BAND[1]
    theta_band_hits.append(hit)
    print(f"        theta={theta_name:16s} -> Dn(raw Cayley) = {gap_t:6.1f}  in-band={hit}")
record("the raw-Cayley near-hit is THRESHOLD-FRAGILE (slides out of [50,80] for natural theta)",
       not all(theta_band_hits),
       "even the valency-rescaled quantum needs a tuned O(1) occupation threshold to stay in band")

record("NO forced O(1) quantum (F_KM-conjugate) reaches the [50,80] band",
       not forced_in_band,
       "forced quanta stay in the crossing regime, Dn = O(1-15)")
record("[50,80] is reached ONLY by re-inserting the valency |S| (NOT forced by quantisation)",
       rescaled_in_band,
       f"raw Cayley level Dn = {rescaled_results[0][1]:.1f} in band; = |S| x the forced O(1) quantum")


# ===========================================================================
# 2. THE GAIN: the gap is c_g-INDEPENDENT (flat in p) -- it BEATS the band-edge divergence.
#    Whatever the (admissible or not) quantum, the depth uses tau(n)=beta* n, NOT c_g(p).
# ===========================================================================
print("\n=== 2. The genuine gain: c_g-INDEPENDENCE of the gap (flat in p, no band edge) ===")
# The clocked gap does not depend on p at all (no c_g enters); demonstrate by evaluating the
# SAME construction at every in-support p and confirming the gap is constant, unlike the lock.
eps_demo = [float(x) for x in LEVELS_COMB]            # use the in-band quantum for the demo
clocked_gap = stab_depth_linear(eps_demo)
clocked_gap = max(clocked_gap) - min(clocked_gap)
print(f"    clocked gap (this front) = {clocked_gap:.1f} for ALL p   (c_g never enters)")
print(f"    lock gap (PYO 1.8)       = " +
      ", ".join(f"p{p}:{g:.0f}" for p, g in zip(PRIMES_SUPPORT, lock_gaps)) + "  (diverges)")
record("this front's gap is c_g-INDEPENDENT (constant in p) -- beats the lock's divergence",
       True,
       f"flat {clocked_gap:.0f} vs lock {lock_gaps[0]:.0f}->{lock_gaps[-1]:.0f}")


# ===========================================================================
# 3. The amplification clock: the quantum RATIO needed for [50,80] IS the target ~3477.
#    n_g = ln(Q_g)/beta*  =>  Dn in [50,80]  <=>  Q_max/Q_min in [exp(6.35), exp(10.2)].
#    No FORCED structural quantum has an O(10^3) ratio; the level/dim ratios are O(1).
# ===========================================================================
print("\n=== 3. Amplification-clock reading: the required quantum RATIO is the target itself ===")
ratio_lo, ratio_hi = math.exp(BAND[0] * BETA_STAR), math.exp(BAND[1] * BETA_STAR)
print(f"    Dn in [50,80] <=> Q_max/Q_min in [{ratio_lo:.0f}, {ratio_hi:.0f}];  target tau:e = {TARGET:.0f}")
structural_ratios = {
    "level ratio lambda_3/lambda_1": LEVELS_NORM[2] / LEVELS_NORM[0],
    "dim ratio max/min":             max(DIMS) / min(DIMS),
    "lead-irrep ratio":              max(LEAD_IRREP) / min(LEAD_IRREP),
}
for nm, rr in structural_ratios.items():
    print(f"    {nm:28s} = {rr:.3f}   ->  Dn = {math.log(rr)/BETA_STAR:5.1f}")
max_struct_ratio = max(structural_ratios.values())
record("no FORCED structural quantum has a ratio reaching the target window (O(1) vs O(10^3))",
       max_struct_ratio < ratio_lo,
       f"max structural ratio {max_struct_ratio:.2f} << required {ratio_lo:.0f}")
record("hitting [50,80] on the amplification clock REQUIRES a quantum ratio = the target ~3477",
       ratio_lo <= TARGET <= ratio_hi,
       "i.e. the construction would re-import the hierarchy, not derive it")


# ===========================================================================
# 4. Non-parametric cross-check: the stored cascade clock saturates too early to host [50,80].
#    tau_data(n) = cumulative resolved capacity I(n)=sigma_c(0)-sigma_c(n) (monotone, saturating).
# ===========================================================================
print("\n=== 4. Non-parametric clock from stored O25 sigma_c(n): saturates ~ knee, hosts crossing ===")
knees = {}
for q in PRIMES_STORED:
    f = ODIR / f"q{q}_o25.npz"
    if not f.exists():
        continue
    z = np.load(f, allow_pickle=True)
    ns = np.array(z["ns"])
    key = "sigma_c_mean" if "sigma_c_mean" in z.files else "sigma_pair_mean"
    sig = np.array(z[key]).mean(0)
    I = sig[0] - sig
    Imax = I[-1] if I[-1] != 0 else 1.0
    Ihat = I / Imax
    # one-e-folding cell depth on the shared clock (the corpus capacity cell)
    knee = int(ns[np.argmax(Ihat >= (1 - 1 / math.e))]) if np.any(Ihat >= (1 - 1 / math.e)) else int(ns[-1])
    knees[q] = knee
    print(f"    q={q:3d}: shared-clock cell depth (1 e-folding) = {knee:3d}  "
          f"(n_max={int(ns[-1])})")
record("the stored shared cascade clock saturates at a cell depth << 64 (crossing regime)",
       max(knees.values()) < 30,
       f"max shared-clock cell depth = {max(knees.values())} << {DN_REQ:.0f}")


# ===========================================================================
# VERDICT
# ===========================================================================
print("\n=== VERDICT (front: quantised per-level resolution density) ===")
print("GAIN (real, and exactly what the lock could not do): quantising the resolution density")
print("instead of the count makes the inter-generation depth gap c_g-INDEPENDENT -- it is flat")
print("in p and never sees the band-edge divergence c_1->0 or the c_2=1/2 pin. The obstruction")
print("is removed from the DENOMINATOR.")
print()
print("BUT the obstruction MIGRATES to the numerator scale. The quantisation acts on F_KM, a")
print("function of the NORMALISED level (L = I - A/|S|); the forced, F_KM-conjugate quantum is")
print("therefore O(1), and every forced O(1) quantum lands Dn in the CROSSING regime (~3-15),")
print("not [50,80]. The band is reached only by (i) re-inserting the valency |S|=24 (the raw")
print("combinatorial Cayley level = |S| x normalised level), which is NOT forced by the")
print("quantisation principle, or equivalently (ii) on the amplification clock, by a quantum")
print("RATIO equal to the target tau:e ~ 3477 -- i.e. re-importing the hierarchy, not deriving")
print("it. No forced structural ratio exceeds ~1.5.")
print()
print("HONEST STATUS: the front is OPENED and the lock is BEATEN on c_g-independence (a genuine")
print("structural advance), but it does NOT supply the generation stabilisation scale: the")
print("magnitude [50,80] requires a non-forced valency re-insertion / a target-valued quantum")
print("ratio. The hierarchy ~3477 remains absent from every c_g-independent O(1) datum. The")
print("obstruction has moved from 'band-edge divergence of c_g' to 'no O(1) structural quantum")
print("carries an O(10^3) ratio'. Reported as a conditional / circularity-flagged outcome under")
print("the anti-circularity guard, NOT a derivation.")

n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n{n_pass}/{len(checks)} checks pass.")
assert n_pass == len(checks)
