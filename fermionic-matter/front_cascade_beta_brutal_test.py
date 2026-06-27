"""Front cascade-beta: brutal test of m_g ~ lambda_Y * w_g^{1/2} * A_g(beta).

Reconnaissance for the hierarchy-amplification channel (Jerome's choice after the level-assignment no-go), with the
methodological prohibition: lambda_Y may NOT absorb the hierarchy. The amplification factor A_g(beta) is the ONLY
authorised carrier of large generation ratios. The order-one PYO split w_g in {1, 3/5, 2/5} is NOT refitted.

Corpus anchors (no fit):
  * PYO Beau2026pyo: squared Yukawa levels H_Pi = lambda_Y^2 diag(1, 1/2+u, 1/2-u); at the gate u=1/10, w_g={1,3/5,2/5}.
  * SpectralRelaxation Beau2026relax: the single-level ADE mechanism (Kesten-McKay x dim rho) gives mass ratios of
    ORDER UNITY and inter-level separations of only 0.08-0.18 log-decades; the ADE spectral ratios lie in [1.5, 2.0]
    (R=3/2 sits there). Amplification to 10^5-10^17 requires a rapidly growing growth law Lambda_proj(n) along the
    cascade depth n, controlled by the flux bound c_chi(n); scalar power-law calibrations are ruled out (no-go).
  * O16 Beau2026o16: cascade exponent beta* = 1/(delta + 1/2); with delta in [7.4, 10.6], beta* in [0.090, 0.127].

Test logic: stack the only two corpus-fixed O(1) factors (the PYO split and the single-level cascade factor) and
show the product stays O(1); then quantify, in the exponential growth-law regime A(n)=exp(beta* * n), the per-
generation cascade-depth separation Delta n required to reach the observed charged-lepton hierarchy. The depth n_g is
NOT fixed by w_g, by beta* alone, or by lambda_Y; it is the structural datum the hierarchy actually needs.
"""

from fractions import Fraction as F
import math


def main():
    checks = {}

    # ---- 1. the two corpus-fixed O(1) factors --------------------------------------------------
    u = F(1, 10)
    w = [F(1), F(1, 2) + u, F(1, 2) - u]                 # PYO split weights {1, 3/5, 2/5}
    w_root = [math.sqrt(float(x)) for x in w]            # mass-level weights w_g^{1/2}
    pyo_spread = max(w_root) / min(w_root)               # = sqrt(5/2) ~ 1.581
    checks["1_pyo_split_spread_is_sqrt_5o2"] = abs(pyo_spread - math.sqrt(5 / 2)) < 1e-12

    # single-level ADE cascade factor: corpus reports inter-level separations 0.08-0.18 log-decades,
    # i.e. an amplification spread <= 10^0.18 across the three levels (Kesten-McKay x dim rho, O(1)).
    cascade_single_level_max_logdecade = 0.18
    cascade_single_level_spread = 10 ** cascade_single_level_max_logdecade   # ~1.51
    checks["1_single_level_cascade_is_order_one"] = cascade_single_level_spread < 2.0

    # ---- 2. stacking both O(1) factors stays O(1) (cannot make the hierarchy) -------------------
    stacked_spread = pyo_spread * cascade_single_level_spread                # ~2.4
    checks["2_stacked_spread_still_order_one"] = stacked_spread < 3.0

    # ---- 3. observed charged-lepton hierarchy (reference data, NOT a fit target) ----------------
    m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
    obs_spread = m_tau / m_e                             # ~3477
    obs_log10 = math.log10(obs_spread)                  # ~3.54
    checks["3_observed_spread_three_plus_decades"] = obs_log10 > 3.0

    # the gap the amplification must supply, on top of the stacked O(1) factors
    residual_gap = obs_spread / stacked_spread
    checks["3_residual_gap_over_1000x"] = residual_gap > 1e3

    # ---- 4. exponential growth-law regime: required per-generation cascade depth ----------------
    # A(n) = exp(beta* * n)  =>  ratio between generations = exp(beta* * Delta n)
    # solve Delta n = ln(obs_spread) / beta* for the corpus range of beta*
    required_depth = {}
    for label, beta_star in (("beta*=0.090 (delta=10.6)", 0.090),
                             ("beta*=0.127 (delta=7.44)", 0.127),
                             ("beta*=0.213 (older est.)", 0.213)):
        dn = math.log(obs_spread) / beta_star
        required_depth[label] = dn
    # all required depths are large integers (tens of cascade steps), i.e. a NEW structural datum
    checks["4_required_depth_is_tens_of_steps"] = all(v > 30 for v in required_depth.values())

    # ---- report -------------------------------------------------------------------------------
    print("Front cascade-beta: brutal test  m_g ~ lambda_Y * w_g^{1/2} * A_g(beta)")
    print("=" * 78)
    print(f"  PYO split weights w_g        = {{1, 3/5, 2/5}};  root spread = {pyo_spread:.3f} (= sqrt(5/2))")
    print(f"  single-level cascade spread  <= 10^{cascade_single_level_max_logdecade} = "
          f"{cascade_single_level_spread:.3f}  (corpus: O(1), ratios in [1.5,2.0])")
    print(f"  stacked O(1)x O(1) spread     = {stacked_spread:.3f}   <-- cannot exceed O(1)")
    print()
    print(f"  observed lepton spread tau:e = {obs_spread:.0f}  (10^{obs_log10:.2f})")
    print(f"  residual gap to be supplied  = {residual_gap:.0f}x  by the growth law alone")
    print()
    print("  exponential regime A(n)=exp(beta* n): required cascade-depth separation tau<->e")
    for label, dn in required_depth.items():
        print(f"    {label}:  Delta n = ln(3477)/beta* = {dn:.0f} steps")
    print()
    print("  VERDICT: neither the order-one PYO split nor the single-level cascade factor (both O(1),")
    print("  corpus-fixed) can carry the hierarchy, and lambda_Y is forbidden to. In the exponential")
    print("  growth-law regime the hierarchy is reachable, but ONLY through a per-generation cascade")
    print("  DEPTH n_g (tens of steps) that is a NEW structural datum -- not fixed by w_g, by beta*")
    print("  alone, or by lambda_Y. So the hierarchy does not live in beta; it lives in the cascade")
    print("  depths (the growth law Lambda_proj(n) evaluated at the three generation stabilisations).")
    print("=" * 78)

    failed = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if failed:
        raise SystemExit(f"\n{len(failed)} check(s) FAILED: {failed}")
    print(f"\nAll {len(checks)} checks passed (exact split; corpus-anchored amplification bounds; no fit).")


if __name__ == "__main__":
    main()
