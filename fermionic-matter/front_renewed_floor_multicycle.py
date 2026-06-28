#!/usr/bin/env python3
"""Renewed-floor multi-cycle: FALSIFICATION test with a hard kill-switch (Jerome's go (b)).

Reconnaissance, exact, NO fit, NO mass input, NO target-window selection. Produce != publish.

The last untested dynamical reading. R-sequential (front_intrinsic_stabilisation_depths.py) compounds
BI saturation cycles, n_g = sum_h N_cyc,h, N_cyc,h = log(D_cyc,h)/beta*. It failed because every FORCED
single-cycle dynamic range D_cyc is O(1)-O(10), so the cascade reaches Dn <= ~19, an order short of the
required ~64; and the range D_BI ~ 59 that would close it is forced by nothing but the circular
A(n) = exp(beta* n).

The one escape not yet tested: a RENEWED FLOOR. Let each generation be a full BI saturation cycle whose
floor is not reset to a fixed A_min but RENEWED from the previous ceiling by a transfer law

        floor_{g+1} = R( ceiling_g ),

so the dynamic range can compound across cycles. The cumulative depth is then
        n_g = sum_{h<=g} log( ceiling_h / floor_h ) / beta*.

JEROME'S HARD KILL-SWITCH (the decisive criterion).
    R must be a transfer relation ALREADY PRESENT in the corpus -- forced by BI / ADE / Heisenberg /
    Schur structure -- NOT chosen to manufacture D_BI ~ 59. If no forced R yields a LARGE
    (cumulative Dn in [50,80]) and NON-CIRCULAR range, the dynamical front is CLOSED honestly.
    A renewal tuned to the window is a free normalisation under another name = the disguised k_g,
    exactly the circularity the static |S| hard-exit already rejected.

This script enumerates EVERY forced renewal candidate the corpus supplies, computes its cumulative
depth, classifies it forced/circular/O(1), and fires the kill-switch.

VERDICT (computed below): no forced renewal supplies a large non-circular range. The only corpus
transfer relations are (Schur) scale-free O(1), (ADE) O(1) level ladder, (band) O(1) fixed normalised
band, (Heisenberg) the polynomial shell growth |S_n|~n^3 that IS the cascade amplification (circular),
and (BI) the ceiling/floor beta_BI/A_min with A_min an unpinned free scale (circular). Kill-switch
FIRES -> dynamical front closed; land (c).
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

# --- forced corpus constants -------------------------------------------------
CASE = "2I_ord5"
S    = R.ADE_CASES[CASE]["S"]                       # 24
LAMBDA = R.ADE_CASES[CASE]["lambda_comb"]           # [20, 24, 30]
AB   = 2.0 * np.sqrt(S - 1.0)                       # Alon-Boppana floor 2*sqrt(23) = 9.5917
BETA_STAR = 0.127                                   # O16 pivot
U = 0.10
W_G = np.array([1.0, 0.5 + U, 0.5 - U])            # {1, 3/5, 2/5} -- PYO J_3 weights

# compare-only target (never used to select)
DN_REQ = np.log(3477.0) / BETA_STAR                 # ~64.2
WINDOW = (50.0, 80.0)
print(f"=== compare-only: required cumulative gap Dn_req(tau:e) = {DN_REQ:.1f}, window {WINDOW} ===\n")

def cumulative_depth(ceilings, floors):
    """n_g = sum_{h<=g} log(ceiling_h/floor_h)/beta* ; return the total gap n_3 - 0 = n_3."""
    per_cyc = np.log(np.asarray(ceilings) / np.asarray(floors)) / BETA_STAR
    return float(np.sum(per_cyc)), per_cyc

# ============================================================================
# Enumerate FORCED renewal candidates.  Each: (name, source, ceilings, floors,
# is_forced, is_circular, note).  floors[g] = R(ceiling[g-1]) for g>=1.
# ============================================================================
print("=== Forced renewal candidates floor_{g+1} = R(ceiling_g) ===\n")
results = []

# (1) SCHUR transport reset.  R = Sym^2 Schur-residue lift: PROVEN scale-free (homogeneous deg 1,
#     front_valency_forced_hardexit.py), so floor_{g+1} = w-ratio * ceiling_g with an O(1) factor.
#     Take the forced residue ratio r_schur = w_2/w_1 etc. (O(1)); floors renewed by it.
r_schur = float(W_G[1] / W_G[0])                    # 3/5 -> contraction 0.6  (O(1), forced, scale-free)
ceil1 = [1.0, 1.0, 1.0]                              # scale-free: only ratios matter
floor1 = [r_schur, r_schur, r_schur]
tot1, pc1 = cumulative_depth(ceil1, floor1)
results.append(("Schur Sym^2 reset", "front_valency_forced_hardexit (scale-free deg 1)",
                tot1, True, False, f"per-cycle range = 1/r_schur = {1/r_schur:.2f} (O(1), forced)"))

# (2) ADE level ladder.  ceiling_g = lambda_g, floor_{g+1} = lambda_g -> floor_g = lambda_{g-1}.
#     The renewal is the inter-level step, O(1).
ceil2  = LAMBDA                                      # [20,24,30]
floor2 = [LAMBDA[0]] + LAMBDA[:-1]                   # [20,20,24] : floor_{g}=lambda_{g-1}
tot2, pc2 = cumulative_depth(ceil2, floor2)
results.append(("ADE level ladder", "stratigraphy {20,24,30}",
                tot2, True, False, f"per-cycle ranges = {np.round(np.array(ceil2)/np.array(floor2),3)} (O(1))"))

# (3) Band-edge renewal.  floor_{g+1} = Alon-Boppana floor of the shell; normalised band is FIXED.
D_band = (1.0 + AB / S) / (1.0 - AB / S)             # 2.33, q-independent
ceil3, floor3 = [D_band]*3, [1.0]*3
tot3, pc3 = cumulative_depth(ceil3, floor3)
results.append(("Band-edge (Alon-Boppana)", "normalised Heisenberg band l+/l-",
                tot3, True, False, f"per-cycle range = {D_band:.2f} fixed (O(1))"))

# (4) HEISENBERG shell growth.  |S_n| ~ n^3, D_hom=4 (Bass-Guivarc'h, relaxation l.1041).  Using the
#     polynomial shell range AS the per-cycle range is the cascade amplification itself: the capacity
#     accumulates exp(beta* n) over a cycle -> N_cyc = log(exp(beta* n))/beta* = n  (TAUTOLOGY).
ceil4, floor4 = [np.exp(BETA_STAR * (DN_REQ/3))]*3, [1.0]*3   # any large range == exp(beta* n)
tot4, pc4 = cumulative_depth(ceil4, floor4)
results.append(("Heisenberg shell / cascade", "|S_n|~n^3 feeds A(n)=exp(beta* n)",
                tot4, True, True, "per-cycle range = exp(beta* n) -> N_cyc = n (CIRCULAR tautology)"))

# (5) BI ceiling/floor.  D_cyc = beta_BI / A_min; A_min only 'a fixed scale', value NEVER pinned
#     (relaxation l.345) -> free scale, tunable to 59 -> CIRCULAR (disguised k_g).
A_min_free = 59.0 ** (-1.0)                          # whatever lands the window; UNFORCED
ceil5, floor5 = [1.0]*3, [A_min_free**(1/3)]*3       # tuned -> hits window by construction
tot5, pc5 = cumulative_depth(ceil5, floor5)
results.append(("BI ceiling/floor beta_BI/A_min", "relaxation l.345: A_min unpinned free scale",
                tot5, False, True, "lands window ONLY by tuning the free A_min = disguised k_g (CIRCULAR)"))

print(f"{'candidate':30s} {'forced':>6s} {'circular':>8s} {'cum.Dn':>8s}   note")
print("-" * 110)
for name, src, tot, forced, circ, note in results:
    print(f"{name:30s} {str(forced):>6s} {str(circ):>8s} {tot:8.1f}   {note}")
    print(f"{'':30s} src: {src}")

# ============================================================================
# KILL-SWITCH.  Survivor = forced AND non-circular AND cumulative Dn in window.
# ============================================================================
print("\n=== KILL-SWITCH ===")
survivors = [(name, tot) for name, src, tot, forced, circ, note in results
             if forced and (not circ) and (WINDOW[0] <= tot <= WINDOW[1])]
forced_noncirc_max = max((tot for name, src, tot, forced, circ, note in results
                          if forced and not circ), default=0.0)

record("every FORCED non-circular renewal gives cumulative Dn an order below the window",
       forced_noncirc_max < 0.5 * WINDOW[0],
       f"max forced non-circular cumulative Dn = {forced_noncirc_max:.1f}  vs  window [{WINDOW[0]:.0f},{WINDOW[1]:.0f}]")
record("the only renewals reaching the window are CIRCULAR (cascade tautology or free A_min)",
       all(circ for name, src, tot, forced, circ, note in results
           if WINDOW[0] <= tot <= WINDOW[1]),
       "Heisenberg-shell = exp(beta* n); BI ceiling/floor = unpinned A_min")
record("NO forced corpus transfer law floor_{g+1}=R(ceiling_g) supplies a large non-circular range",
       len(survivors) == 0,
       f"survivors = {survivors}")

KILL_SWITCH_FIRES = (len(survivors) == 0)
record("KILL-SWITCH FIRES -> renewed-floor route CLOSED -> land (c), not (a)",
       KILL_SWITCH_FIRES,
       "renewal not derived from a corpus structural relation -> close the dynamical front honestly")

print("\n" + "=" * 78)
n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"SUMMARY: {n_pass}/{len(checks)} checks pass")
print("=" * 78)
print("""
Verdict (renewed-floor falsification, go (b))
---------------------------------------------
The corpus supplies exactly five transfer relations that could renew the cycle floor, and NONE is
a large non-circular range:
  - Schur Sym^2 reset       : scale-free (homogeneous deg 1) -> O(1) residue ratio  [forced, O(1)]
  - ADE level ladder        : {20,24,30} successive ratios ~1.2-1.25               [forced, O(1)]
  - band-edge (Alon-Boppana): fixed normalised band 2.33                           [forced, O(1)]
  - Heisenberg shell |S_n|~n^3 : polynomial growth = the cascade amplification      [forced, CIRCULAR]
                              exp(beta* n) -> N_cyc = n, a tautology, not a derivation
  - BI ceiling/floor beta_BI/A_min : A_min an unpinned 'fixed scale' -> free knob    [unforced, CIRCULAR]

The kill-switch fires: the renewed-floor mechanism reaches the window ONLY through the circular
cascade amplification or a free A_min -- a disguised per-generation k_g, exactly the circularity the
static |S| hard-exit already rejected. The dynamical route is CLOSED on its renewed-floor escape.

LANDING (c). The projective cascade fixes generation STRUCTURE (three stratigraphic levels), the
order-one SPLITTING (PYO {1,3/5,2/5}), and -- new this front -- a band-edge-INDEPENDENT O(1) depth
gap; it does NOT fix the charged-fermion HIERARCHY ~3477, neither statically nor dynamically at first
order:
    generation structure + order-one splitting + band-edge-independent depth gap  !=  hierarchy ~3477.
The R-parallel band-edge independence is retained as a SECONDARY note for PYO, to be folded ONLY after
this closure, in the sober form: "the dynamical parallel reading removes the band-edge obstruction but
remains an order-one depth mechanism."
""")
sys.exit(0 if n_pass == len(checks) else 1)
