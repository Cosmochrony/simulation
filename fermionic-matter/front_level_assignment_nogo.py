"""Front level-assignment: no direct mass assignment from the squared Yukawa levels.

Bias-independent, exact symbolic (SymPy / exact rationals), no fit. Reconnaissance audit for the next mass-sector
front, opened after Front 3 (arithmetic gate eps <-> u).

Context (PYO Beau2026pyo, prop:squared-yukawa): the generation-level observable of the Schur-residue sector is the
positive hermitian square H_Pi = Y_Pi^dag Y_Pi, with

    Spec(H_Pi)|gen = lambda_Y^2 * diag(1, 1/2 + u, 1/2 - u),

a free overall positive scale lambda_Y times an internal split set by u. Front 3 established that u and the angular
datum eps are the same functional of the level ratio R_Pi = (1/2+u)/(1/2-u), with eps(R_Pi) = u, and that the value
u = eps = 1/10 holds iff the arithmetic dictionary gate R_Pi = R_ADE = 3/2 (order-five Cayley ratio) is adopted.

This audit assumes the dictionary gate (u = 1/10) and tests the NAIVE level-to-generation map
"squared level k = squared mass of generation k". The verdict is a quantitative no-go: the squared levels are O(1)
ratios, whereas the observed charged-fermion mass spectrum is strongly hierarchical. Hence the diagonal datum u is a
generation-LEVEL WEIGHT inside H_Pi, not a direct mass-ratio predictor; the hierarchy must be carried by a separate
amplification channel (spectral cascade exponent, sector-dependent lambda_Y, or a downstream operator).

No floating-point fitting: the level structure is exact; observed masses enter only as reference data for the
order-of-magnitude gap, not as a target to be matched.
"""

from fractions import Fraction as F
import math


def main():
    checks = {}

    # ---- 1. squared Yukawa levels at the dictionary gate u = 1/10 -------------------------------
    u = F(1, 10)
    levels = [F(1), F(1, 2) + u, F(1, 2) - u]                 # diag(1, 1/2+u, 1/2-u), up to lambda_Y^2
    checks["1_levels_are_1_3o5_2o5"] = levels == [F(1), F(3, 5), F(2, 5)]

    # squared-level ratios, normalised to the smallest level (2/5)
    sq_ratios = [l / levels[2] for l in levels]               # 5/2 : 3/2 : 1
    checks["1_squared_ratios_5o2_3o2_1"] = sq_ratios == [F(5, 2), F(3, 2), F(1)]

    # the internal level ratio R_Pi = (1/2+u)/(1/2-u) = 3/2 (Front 3 dictionary gate)
    R_Pi = (F(1, 2) + u) / (F(1, 2) - u)
    checks["1_internal_ratio_R_Pi_is_3o2"] = R_Pi == F(3, 2)

    # ---- 2. root (mass-level) ratios: still O(1) ----------------------------------------------
    roots = [math.sqrt(float(l)) for l in levels]             # 1 : sqrt(3/5) : sqrt(2/5)
    root_ratios = [r / roots[2] for r in roots]               # normalise to smallest
    max_root_ratio = max(root_ratios)
    checks["2_max_root_ratio_is_order_one"] = max_root_ratio < 2.0   # ~1.581, O(1)
    # exact statement: largest/smallest mass-level ratio is sqrt(5/2)
    checks["2_max_root_ratio_equals_sqrt_5o2"] = abs(max_root_ratio - math.sqrt(5 / 2)) < 1e-12

    # ---- 3. observed charged-lepton hierarchy (reference data, NOT a fit target) ----------------
    # PDG charged-lepton masses (MeV): e, mu, tau
    m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
    obs_mass_ratio = m_tau / m_e                              # ~3477
    obs_mass2_ratio = (m_tau / m_e) ** 2                     # ~1.2e7
    # the smallest non-trivial observed step already dwarfs the whole predicted O(1) spread
    obs_min_step = m_mu / m_e                                 # ~207

    # ---- 4. the quantitative no-go ------------------------------------------------------------
    # predicted spread of mass levels (largest/smallest) vs the smallest observed inter-generation step
    predicted_spread = max_root_ratio                         # ~1.581
    gap_factor = obs_min_step / predicted_spread              # how far off the direct map is
    checks["4_direct_map_excluded_gap_over_100x"] = gap_factor > 100.0

    # ---- report -------------------------------------------------------------------------------
    print("Front level-assignment: no direct mass assignment from the squared Yukawa levels")
    print("=" * 78)
    print(f"  dictionary gate:        u = {u}  (R_Pi = R_ADE = {R_Pi})")
    print(f"  squared Yukawa levels:  lambda_Y^2 * {{1, {levels[1]}, {levels[2]}}}  = {{1, 3/5, 2/5}}")
    print(f"  squared-level ratios:   {sq_ratios[0]} : {sq_ratios[1]} : {sq_ratios[2]}")
    print(f"  root (mass) ratios:     1 : sqrt(3/5) : sqrt(2/5)  =  "
          f"{root_ratios[0]:.4f} : {root_ratios[1]:.4f} : {root_ratios[2]:.4f}")
    print(f"  predicted O(1) spread:  {predicted_spread:.4f}  (= sqrt(5/2))")
    print()
    print(f"  observed lepton mass ratio  tau:e   = {obs_mass_ratio:.0f}")
    print(f"  observed lepton mass^2 ratio tau:e  = {obs_mass2_ratio:.3e}")
    print(f"  smallest observed step      mu:e    = {obs_min_step:.0f}")
    print(f"  gap factor (mu:e / predicted spread) = {gap_factor:.0f}x  -> direct map excluded")
    print()
    print("  VERDICT: the squared Yukawa levels fix an O(1) internal generation-level split, NOT the")
    print("  charged-fermion mass hierarchy. The diagonal datum u is a level weight inside H_Pi; the")
    print("  hierarchy must be carried by a separate amplification channel (cascade exponent, sector-")
    print("  dependent lambda_Y, or a downstream operator). Mixing/CP stay out of scope ([U_Pi] != [I]).")
    print("=" * 78)

    failed = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if failed:
        raise SystemExit(f"\n{len(failed)} check(s) FAILED: {failed}")
    print(f"\nAll {len(checks)} checks passed (exact, no fit).")


if __name__ == "__main__":
    main()
