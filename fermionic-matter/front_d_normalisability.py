"""Front D normalisability test: does the BI/A4 saturation functional define s_* coordinate-independently?

Bias-independent, exact symbolic verification (no sampling) of the single Front D fence question
(fermionic-matter/front-d-normalisation-fence.md): the chiral defect rate dDelta_chi/ds|_0 is covariant
under odd reparametrisation of the modulus, so the physical magnitude is carried by u(s_*), the split
evaluated at the A4-saturated configuration s_*. The test asks whether s_* is a coordinate-independent
physical point.

Grounding in the A4-note (Beau2026bim):
  - saturation is the radicand contact: D_chi(s) >= 0 with saturation D_chi(s) -> 0; for an electric
    (timelike) chiral polarisation D_chi reaches 0 at finite |s|, where A4 locks a non-trivial pair +-s_*
    (A4-note Sec. "The Born-Infeld saturation margin");
  - the projector modulus s_proj (Front C) and the BI saturation modulus s_BI are related by a
    non-degenerate odd reparametrisation s_BI = phi(s_proj), phi(0)=0, phi'(0)!=0 (Lemma "Tangential
    coincidence"), so they label the SAME physical point;
  - sign(mu_chi^2) is reparametrisation-invariant (Corollary "Reparametrisation-invariance of the genus").

This script verifies four facts that together answer the test:
  (a) the first-order defect rate is covariant: d/ds_proj ↦ phi'(0) · d/ds_BI  (so the rate alone is not
      physical, exactly the fence's lambda^{-1} covariance);
  (b) when s_* is the radicand contact D_chi(s_*)=0, the physical value u(s_*) is the SAME computed in the
      s_proj chart or the s_BI chart — the contact is coordinate-independent (ADMISSIBLE s_*);
  (c) the genus sign mu_chi^2 = B''(0) is reparametrisation-invariant (re-derives the A4-note corollary);
  (d) by contrast, the extremum of a FINITE polynomial truncation of B is chart-dependent: its physical
      location differs between the two charts (INADMISSIBLE s_*), confirming the fence success condition
      that a truncated-polynomial minimum is not an admissible s_*.

Conclusion (printed): on the saturation-contact (boundary) branch s_* is coordinate-independent, so
|u| = |u(s_*)| is well-posed; the truncated-interior alternative is not, matching the A4-note's open
boundary-vs-interior selection. No figures. Code and comments in English.
"""

import sympy as sp


def main():
    s, sigma = sp.symbols("s sigma", real=True)
    checks = {}

    # Non-degenerate ODD reparametrisation s_BI = phi(s_proj), phi(0)=0, phi'(0)=a1!=0 (odd preserves
    # the J_Pi-odd parity of the modulus). Use sigma = s_proj, s = s_BI.
    a1, a3, a5 = sp.Rational(3, 2), sp.Rational(1, 5), sp.Rational(-1, 40)
    phi = a1 * sigma + a3 * sigma**3 + a5 * sigma**5     # s_BI = phi(s_proj)

    # Odd split function u(s_BI) with u(0)=0 (Lemma u-odd: u(-s) = -u(s)); generic odd cubic jet.
    b1, b3 = sp.Rational(7, 4), sp.Rational(-2, 3)
    u = lambda x: b1 * x + b3 * x**3

    # Even saturation functional B(s_BI) = B0 + (mu2/2) s^2 + g4 s^4 + g6 s^6, electric genus mu2 < 0.
    B0, mu2, g4, g6 = sp.Integer(5), sp.Rational(-1, 1), sp.Rational(1, 8), sp.Rational(-1, 200)
    B = lambda x: B0 + (mu2 / 2) * x**2 + g4 * x**4 + g6 * x**6

    # Radicand D_chi(s_BI): electric -> decreasing, hits the bound D=0 at finite |s|. Pick a rational root.
    # D(x) = D0 - c x^2, D0=c*4 -> contact at x_* = 2 (the BI saturation contact, a geometric condition).
    c = sp.Integer(3)
    D0 = c * 4
    D = lambda x: D0 - c * x**2
    x_star_BI = sp.sqrt(D0 / c)                          # = 2, contact in the s_BI chart
    assert sp.simplify(x_star_BI - 2) == 0

    # (a) Rate covariance: d/ds_proj (u o phi)|_0 = u'(0) * phi'(0); ratio to the s_BI rate is phi'(0).
    rate_BI = sp.diff(u(s), s).subs(s, 0)
    rate_proj = sp.diff(u(phi), sigma).subs(sigma, 0)
    checks["a_rate_covariant_phiprime"] = sp.simplify(rate_proj - sp.diff(phi, sigma).subs(sigma, 0) * rate_BI) == 0
    checks["a_rate_not_invariant"] = sp.simplify(rate_proj - rate_BI) != 0   # covariant, not invariant

    # (b) u at the radicand contact is coordinate-independent.
    # s_BI chart: contact x_* from D(x)=0; physical value u(x_*).
    u_phys_BI = sp.nsimplify(u(x_star_BI))
    # s_proj chart: the SAME geometric condition is D(phi(sigma))=0; its positive root sigma_* gives the
    # physical point phi(sigma_*); evaluate u there and compare.
    roots = sp.solve(sp.Eq(D(phi), 0), sigma)
    sigma_star = [r for r in roots if r.is_real and r > 0]
    sigma_star = min(sigma_star, key=lambda r: sp.Abs(r))   # first positive contact
    u_phys_proj = sp.nsimplify(u(phi.subs(sigma, sigma_star)))
    checks["b_contact_point_same"] = sp.simplify(phi.subs(sigma, sigma_star) - x_star_BI) == 0
    checks["b_u_at_contact_invariant"] = sp.simplify(u_phys_proj - u_phys_BI) == 0

    # (c) Genus sign reparametrisation-invariance: d^2/dsigma^2 (B o phi)|_0 = phi'(0)^2 * B''(0).
    B_proj_2nd = sp.diff(B(phi), sigma, 2).subs(sigma, 0)
    B_BI_2nd = sp.diff(B(s), s, 2).subs(s, 0)
    phip0 = sp.diff(phi, sigma).subs(sigma, 0)
    checks["c_genus_chainrule"] = sp.simplify(B_proj_2nd - phip0**2 * B_BI_2nd) == 0
    checks["c_genus_sign_preserved"] = sp.sign(B_proj_2nd) == sp.sign(B_BI_2nd)

    # (d) Truncated interior extremum is chart-dependent (INADMISSIBLE s_*).
    # s_BI chart: extremum of B truncated at quartic order, B4(x) = B0 + (mu2/2)x^2 + g4 x^4.
    B4_BI = B0 + (mu2 / 2) * s**2 + g4 * s**4
    ext_BI = [r for r in sp.solve(sp.diff(B4_BI, s), s) if r.is_real and r > 0]
    ext_BI = min(ext_BI, key=lambda r: sp.Abs(r))
    # s_proj chart: compose then truncate to quartic in sigma, extremise, map back through phi.
    B_comp = sp.expand(B(phi))
    B4_proj = sum(t for t in B_comp.as_ordered_terms() if sp.degree(t, sigma) <= 4)
    ext_proj_sigma = [r for r in sp.solve(sp.diff(B4_proj, sigma), sigma) if r.is_real and r > 0]
    ext_proj_sigma = min(ext_proj_sigma, key=lambda r: sp.Abs(r))
    ext_proj_mapped = phi.subs(sigma, ext_proj_sigma)       # same physical chart (s_BI) for comparison
    checks["d_truncated_extremum_chart_dependent"] = sp.simplify(ext_proj_mapped - ext_BI) != 0

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("rate (s_BI)   =", rate_BI, "   rate (s_proj) =", sp.nsimplify(rate_proj),
          "  -> covariant by phi'(0) =", phip0, "(rate alone NOT physical)")
    print("contact s_* (geometric, D_chi=0): u(s_*) =", u_phys_BI,
          "= same in both charts -> ADMISSIBLE, |u| = |u(s_*)| well-posed")
    print("truncated interior extremum: s_BI-chart =", sp.nsimplify(ext_BI),
          ", s_proj-chart mapped back =", sp.nsimplify(ext_proj_mapped),
          "-> DIFFERENT physical points -> INADMISSIBLE")
    print()
    print("ANSWER: yes, on the saturation-contact (boundary) branch s_* is coordinate-independent;")
    print("the truncated-interior alternative is not (A4-note's open boundary-vs-interior selection).")
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
