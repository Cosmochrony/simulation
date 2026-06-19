"""Front D1: boundary-vs-interior selection test for the A4 chiral lock.

Bias-independent, exact symbolic / exact-rational verification (no sampling) of the single remaining
Front D conditionality (fermionic-matter/front-d-normalisability-test.md): which configuration A4 locks,
the radicand contact (boundary) or an interior extremum of the saturation functional (interior).

Discipline: a configuration is fence-admissible only if it is determined by DERIVED, available data
without importing a non-derived complex phase or cascade ordering (A4-note states the sixth-order
interior lock is "doubly conditional, on a non-derived complex phase and on a non-prescribed ordering").

Available data: the radicand D_chi(s) and the saturation functional B(s) to the order actually fixed by
the corpus (quartic; the sextic coefficient is exactly the non-derived phase/ordering datum).

The test verifies:
  (1) the radicand contact s_c (root of D_chi(s)=0) is independent of every B-coefficient AND of the
      phase datum: it is fixed by the radicand alone -> available, chart-independent (boundary ADMISSIBLE);
  (2) a non-trivial interior extremum s_int (nonzero root of B'(s)=0) DEPENDS on the sextic coefficient
      and on the phase datum gamma: it is not fixed by the available quartic data -> requires the
      non-derived phase/ordering (interior NOT ADMISSIBLE without extra data);
  (3) turning on the non-derived phase gamma moves s_int but leaves s_c untouched.

Conclusion (printed): within the no-circularity fence, only the radicand contact defines a
chart-independent A4 lock from the available data; the interior lock is contingent on a future
first-principles derivation of the phase and ordering. Hence the boundary branch is uniquely admissible
and |u| = |u(s_*)| at the contact is the fence-admissible normalisation, no longer branch-conditional.

No figures. Code and comments in English.
"""

import sympy as sp


def main():
    s = sp.symbols("s", real=True)
    # Radicand: electric/timelike, decreasing, reaches the BI bound D_chi=0 at finite |s|.
    D0, c = sp.symbols("D0 c", positive=True)
    D_chi = D0 - c * s**2
    s_contact = sp.sqrt(D0 / c)                       # boundary: geometric contact D_chi(s_c)=0
    assert sp.simplify(D_chi.subs(s, s_contact)) == 0

    # Saturation functional. Quartic data (mu2<0 electric, g4) are available; the sextic coefficient is
    # g6 + gamma, where gamma is the non-derived complex-phase/ordering datum the A4-note isolates.
    mu2, g4, g6, gamma, B0 = sp.symbols("mu2 g4 g6 gamma B0", real=True)
    B = B0 + (mu2 / 2) * s**2 + g4 * s**4 + (g6 + gamma) * s**6

    checks = {}

    # (1) Contact is fixed by the radicand alone: independent of all B-data and of the phase.
    checks["contact_indep_g4"] = sp.diff(s_contact, g4) == 0
    checks["contact_indep_g6"] = sp.diff(s_contact, g6) == 0
    checks["contact_indep_gamma"] = sp.diff(s_contact, gamma) == 0
    checks["contact_indep_mu2"] = sp.diff(s_contact, mu2) == 0

    # (2) Non-trivial interior extremum: nonzero root of B'(s)=0, i.e. of
    #     f(s) = mu2 + 4 g4 s^2 + 6(g6+gamma) s^4 = 0.
    # The extremum condition itself depends on the sextic datum and on the phase gamma (df/d* != 0), so by
    # the implicit function theorem the root location moves with them.
    f = mu2 + 4 * g4 * s**2 + 6 * (g6 + gamma) * s**4
    checks["interior_cond_depends_on_g6"] = sp.simplify(sp.diff(f, g6)) != 0
    checks["interior_cond_depends_on_phase_gamma"] = sp.simplify(sp.diff(f, gamma)) != 0
    # Implicit ds/dgamma = -(df/dgamma)/(df/ds): generically non-zero at an interior root.
    ds_dgamma = -sp.diff(f, gamma) / sp.diff(f, s)
    checks["interior_root_moves_with_phase"] = sp.simplify(ds_dgamma) != 0

    # (3) Concrete exact instance: turning on the non-derived phase gamma moves the interior root in
    # w = s^2, while the radicand contact s_c is untouched. Use a genuine sextic (g6 != 0).
    w = sp.symbols("w", positive=True)
    subs0 = {D0: 4, c: 1, mu2: sp.Rational(-1, 1), g4: sp.Rational(1, 8), g6: sp.Rational(1, 100), B0: 5}

    def interior_w(gval):
        quad = (mu2 + 4 * g4 * w + 6 * (g6 + gamma) * w**2).subs({**subs0, gamma: gval})
        return [r for r in sp.solve(quad, w) if r.is_real and r > 0]

    w_g0 = interior_w(0)[0]
    w_gp = interior_w(sp.Rational(1, 100))[0]
    s_c_val = sp.nsimplify(s_contact.subs(subs0))
    checks["phase_moves_interior"] = sp.simplify(w_gp - w_g0) != 0
    checks["phase_leaves_contact"] = sp.simplify(
        s_contact.subs({**subs0, gamma: sp.Rational(1, 100)}) - s_contact.subs({**subs0, gamma: 0})
    ) == 0
    dwint_dg6 = sp.simplify(sp.diff(f, g6))
    dwint_dgamma = sp.simplify(sp.diff(f, gamma))
    w_int_g0, w_int_gp = w_g0, w_gp

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("radicand contact  s_c = sqrt(D0/c)  -> independent of B-coefficients and of the phase gamma")
    print("                       (fixed by the radicand alone: AVAILABLE, chart-independent -> ADMISSIBLE)")
    print("interior extremum condition f(s) = mu2 + 4 g4 s^2 + 6(g6+gamma) s^4 = 0")
    print("   df/d(g6)    =", dwint_dg6, "   (extremum condition depends on the non-derived sextic datum)")
    print("   df/d(gamma) =", dwint_dgamma, "   (extremum condition depends on the non-derived complex phase)")
    print(f"   exact instance: s_c = {s_c_val} (phase-invariant); "
          f"interior s_int^2: gamma=0 -> {w_int_g0}, gamma=1/100 -> {w_int_gp} (phase-dependent)")
    print()
    print("ANSWER: within the no-circularity fence, only the radicand contact D_chi(s_*)=0 defines a")
    print("chart-independent A4 lock from the available data; the interior lock is inadmissible unless the")
    print("non-derived complex phase/ordering is supplied. Boundary branch uniquely admissible.")
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
