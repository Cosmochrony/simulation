#!/usr/bin/env python3
"""Front "Generation stabilisation depths" -- Go (2): inter-LEVEL capacity cell depth.

Reconnaissance, exact (KM growth law), NO fit, NO mass input.

Jerome's Go (2): the generation split should live in a capacity profile resolved by the
three stratigraphic LEVELS lambda_g in {20,24,30} of 2I (ord-5), not by the color blocks
(O32 co-admissibility kills the inter-block split, but the level partition is different).

Construction (the only one faithful to the corpus growth law, no new modelling input):
  the level-resolved cumulative count is the linear growth law of Beau2026a5,
      N(lambda_g; n) = c_g(p) * n,   c_g(p) = F_KM(lambda_g),     [eq:counting-linear]
  and level g saturates its rep block at N = k dim rho_g (hyp:saturation-rep), i.e.
      n_sat,g(p) = k dim rho_g / c_g(p).                          [eq:nproj-value]
  A cell-ENTRY (before full saturation) at cumulative fraction f of the block is then
      n_g(f, p) = f * n_sat,g(p),     so   Dn(f,p) = f * (n_sat,1 - n_sat,3).

Brutal pass/fail (Jerome):
  PASS  Dn_lambda in [50,80], coherent generational order, q-stable.
  FAIL  Dn_lambda = O(1-10), OR Dn_lambda unstable in q.

Reuses simulation/spectral/relaxation/spectral_relaxation_lib.py (km_cdf, ADE_CASES).
"""

import math
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "spectral/relaxation"
sys.path.insert(0, str(LIB))
import spectral_relaxation_lib as R   # noqa: E402

CASE = "2I_ord5"
DIMS = R.ADE_CASES[CASE]["dims"]               # [54, 25, 40]
LEVELS = R.normalised_levels(CASE)             # [20/24, 1, 30/24]
PRIMES_IN_SUPPORT = [5, 13, 29, 53]            # 2I ord-5 support: lambda_+ > lambda_3
BETA_STAR = 0.127
DN_REQ = math.log(3477.0) / BETA_STAR          # ~64.2, the target inter-gen gap

checks = []
def record(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


print("=== Level-resolved saturation ranks n_sat,g(p) = dim rho_g / c_g(p)  (k=1) ===")
print(f"    levels (norm) = {[round(float(l),4) for l in LEVELS]}, dims = {DIMS}\n")

nsat = {}
for p in PRIMES_IN_SUPPORT:
    c = [float(R.km_cdf(l, p)) for l in LEVELS]
    ns = [d / ci for d, ci in zip(DIMS, c)]
    nsat[p] = ns
    print(f"  p={p:2d}: c=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  "
          f"n_sat=({ns[0]:6.1f},{ns[1]:5.1f},{ns[2]:5.1f})  "
          f"Dn_sat=n1-n3={ns[0]-ns[2]:6.1f}")

# --------------------------------------------------------------------------
# (a) generational ordering: n_2 = dim_2/c_2 = 25/0.5 = 50 is FIXED (c_2=1/2 by KM
#     symmetry), while n_3 = 40/c_3(p) crosses 50 -> the n_2,n_3 order SWAPS with p.
# --------------------------------------------------------------------------
print("\n=== (a) Generational ordering (q-stability of the order itself) ===")
signs = [1 if nsat[p][1] > nsat[p][2] else -1 for p in PRIMES_IN_SUPPORT]  # n_2 vs n_3
print(f"    n_2 fixed = {nsat[PRIMES_IN_SUPPORT[0]][1]:.1f} (c_2=1/2);  "
      f"n_3(p) = " + ", ".join(f"p{p}:{nsat[p][2]:.0f}" for p in PRIMES_IN_SUPPORT))
record("generational ORDER is q-unstable: n_2,n_3 swap across in-support p",
       len(set(signs)) > 1,
       f"sign(n2-n3) = {signs}  (n_3 crosses the fixed n_2=50)")

# --------------------------------------------------------------------------
# (b) a UNIFORM sub-saturation fraction f can hit Dn=64 -- but only at one p
# --------------------------------------------------------------------------
print("\n=== (b) Cell fraction f needed to hit Dn_req=64.2 at each p ===")
f_needed = {p: DN_REQ / (nsat[p][0] - nsat[p][2]) for p in PRIMES_IN_SUPPORT}
for p in PRIMES_IN_SUPPORT:
    f = f_needed[p]
    cells = [f * x for x in nsat[p]]
    print(f"  p={p:2d}: f_needed = {f:.3f}   ->  cells=({cells[0]:.1f},{cells[1]:.1f},{cells[2]:.1f})  "
          f"Dn={cells[0]-cells[2]:.1f}")
fvals = list(f_needed.values())
record("the cell fraction f is NOT a stable constant across p (band-edge spread)",
       max(fvals) / min(fvals) > 2.0,
       f"f ranges {min(fvals):.3f}..{max(fvals):.3f}  (x{max(fvals)/min(fvals):.1f})")

# --------------------------------------------------------------------------
# (c) q-stability of the GAP at a FIXED structural fraction f
#     test a few structural constants for f
# --------------------------------------------------------------------------
print("\n=== (c) q-stability of Dn at fixed structural fractions f ===")
struct_f = {"1/e": 1/math.e, "1/e^2": 1/math.e**2, "1-1/e": 1-1/math.e, "1/(2pi)": 1/(2*math.pi)}
for name, f in struct_f.items():
    dns = [f * (nsat[p][0] - nsat[p][2]) for p in PRIMES_IN_SUPPORT]
    in_band = all(50 <= d <= 80 for d in dns)
    print(f"  f={name:7s}={f:.3f}:  Dn(p)=({dns[0]:.0f},{dns[1]:.0f},{dns[2]:.0f},{dns[3]:.0f})  "
          f"-> in[50,80] for all p? {in_band}")
record("no structural fraction f puts Dn in [50,80] for ALL in-support p",
       not any(all(50 <= f*(nsat[p][0]-nsat[p][2]) <= 80 for p in PRIMES_IN_SUPPORT)
               for f in struct_f.values()),
       "Dn = f*Dn_sat(p) tracks the band-edge divergence, not a finite scale")

# --------------------------------------------------------------------------
# (d) the gap divergence is the band edge: Dn_sat(p) grows monotonically with p
# --------------------------------------------------------------------------
print("\n=== (d) Band-edge origin of the q-instability ===")
gaps = [nsat[p][0] - nsat[p][2] for p in PRIMES_IN_SUPPORT]
print(f"    Dn_sat(p) = " + ", ".join(f"p{p}:{g:.0f}" for p, g in zip(PRIMES_IN_SUPPORT, gaps)))
record("Dn_sat(p) diverges with p (c_1 -> 0 at the band edge): not a finite scale",
       gaps[-1] > 2 * gaps[0], f"{gaps[0]:.0f} (p=5) -> {gaps[-1]:.0f} (p=53)")

print("\n=== VERDICT (Go 2) ===")
print("The inter-LEVEL cell depth, built faithfully on the KM growth law, is the saturation")
print("rank read at a sub-saturation fraction f: n_g = f * dim rho_g / c_g(p).")
print("  - ORDERING: q-UNSTABLE. n_2 = 25/(1/2) = 50 is pinned by KM symmetry while")
print("    n_3 = 40/c_3(p) crosses it, so the n_2,n_3 generational order SWAPS between p.")
print("  - MAGNITUDE: Dn=64 IS reachable -- but only at a tuned fraction f~0.18 and one p.")
print("  - q-STABILITY: FAILS. Dn = f*Dn_sat(p) tracks the band-edge divergence of c_1->0,")
print("    so the gap is not q-stable for any fixed structural f; the needed f swings x>2")
print("    across in-support p. This is Jerome's explicit FAILURE condition (Dn unstable in q).")
print("Conclusion: resolving the LINEAR growth law by level does not escape the saturation")
print("rank -- it only rescales it. The level partition changes the ORDERING carrier (and is")
print("NOT killed by O32 co-admissibility), but the MAGNITUDE/stability obstruction is the")
print("same band-edge divergence. A genuinely q-stable inter-level carrier would need a")
print("DECAYING (capacity-controlled, sub-linear) per-level profile, which the linear LPS-KM")
print("law does not provide and the corpus does not yet contain. Honest reduction, not a")
print("derivation -> the guard-rail bracket stands; the middle stays open.")

n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n{n_pass}/{len(checks)} checks pass.")
assert n_pass == len(checks)
