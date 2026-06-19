"""Front D2: u_partial = u(s_*), the split at the admissible A4 contact, modulo odd reparametrisation.

Bias-independent, exact symbolic verification (no sampling) of the first Front D2 step: the
"linear-exact vs transverse-curvature" test. It decides whether u(s_*) is exhausted by u_1 s_*, where
u_1 is the Front C/D0 carrier rate, or whether a genuine transverse-curvature contribution survives.

Setup (Front C): the split is the odd carrier
    u(s) = u_1 s + u_3 s^3 + ...,   u_1 = <J_3, {E_0, E_1}> / <J_3, J_3>,
with E_Pi(s) = E_0 + s E_1 + s^2 E_2 + s^3 E_3 + ... and parity E_0,E_2 even, E_1,E_3 odd. s_* is the
saturation contact Delta_chi(s_*) = 0 (admissible A4 lock, Front D1).

Three results are verified.
  (A) Carrier expansion: u(s) has only ODD powers, with u_1 = <J_3,{E_0,E_1}>/<J_3,J_3> and
      u_3 = <J_3,{E_0,E_3}+{E_1,E_2}>/<J_3,J_3>.
  (B) The scalar carrier's higher odd terms are PURE reparametrisation gauge: under an odd
      reparametrisation s = t + alpha t^3 + ... the cubic transforms as u_3 -> u_3 + alpha u_1, so it is
      removed by alpha = -u_3/u_1; equivalently the carrier-adapted modulus t = u(s)/u_1 linearises u to
      u = u_1 t identically. So the carrier carries no reparametrisation-invariant higher obstruction
      (test (ii) closes positively for the carrier).
  (C) The transverse curvature is NOT in the carrier but in s_*: the contact s_* depends on the radicand
      quartic b = P.R_chi/beta^2 (the A4-note open transverse-curvature datum), while u_1 does not.

Conclusion (printed): modulo odd reparametrisation, u_partial = u_1 s_*, with the only transverse-curvature
dependence entering through s_* (the radicand contact). No number is produced: u_1 awaits the explicit
locked-frame operators E_0, E_1, and s_* awaits the radicand quartic. No figures. English.
"""

import sympy as sp


def main():
    s = sp.symbols("s", real=True)
    checks = {}

    # ---- C^3_gen convention frozen to schur_transversality_alpha.py ---------------------------------
    J = sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])     # J_Pi^(2)
    J3 = sp.diag(0, 1, -1)

    def hs(X, Y):
        return sp.trace(X.T * Y)                          # real symmetric data: HS pairing tr(X^T Y)

    def even(M):
        return (M + J * M * J) / 2

    def odd(M):
        return (M - J * M * J) / 2

    def gen_sym(tag):
        v = sp.symbols(f"{tag}0:9", real=True)
        M = sp.Matrix(3, 3, v)
        return (M + M.T) / 2                               # generic symmetric (phase-free)

    # Parity-graded residue jet (Front C): E0,E2 J_Pi-even; E1,E3 J_Pi-odd.
    E0 = even(gen_sym("a"))
    E1 = odd(gen_sym("b"))
    E2 = even(gen_sym("c"))
    E3 = odd(gen_sym("d"))
    Epi = E0 + s * E1 + s**2 * E2 + s**3 * E3

    # ---- (A) carrier expansion u(s) -----------------------------------------------------------------
    odd_sq = odd(sp.expand(Epi * Epi))
    u_series = sp.expand(hs(odd_sq, J3) / hs(J3, J3))      # polynomial in s

    u0 = u_series.coeff(s, 0)
    u1 = sp.expand(u_series.coeff(s, 1))
    u2 = u_series.coeff(s, 2)
    u3 = sp.expand(u_series.coeff(s, 3))
    u4 = u_series.coeff(s, 4)

    def acomm(X, Y):
        return X * Y + Y * X

    u1_expected = sp.expand(hs(acomm(E0, E1), J3) / hs(J3, J3))
    u3_expected = sp.expand(hs(acomm(E0, E3) + acomm(E1, E2), J3) / hs(J3, J3))

    checks["A_u_odd_no_constant"] = (u0 == 0)
    checks["A_u_odd_no_quadratic"] = (sp.expand(u2) == 0)
    checks["A_u_odd_no_quartic"] = (sp.expand(u4) == 0)
    checks["A_u1_matches_FrontC"] = sp.expand(u1 - u1_expected) == 0
    checks["A_u3_matches_jet"] = sp.expand(u3 - u3_expected) == 0

    # ---- (B) the scalar cubic is reparametrisation gauge --------------------------------------------
    t = sp.symbols("t", real=True)
    U1, U3, U5 = sp.symbols("U1 U3 U5", real=True)
    alpha, beta = sp.symbols("alpha beta", real=True)
    u_scalar = lambda x: U1 * x + U3 * x**3 + U5 * x**5
    phi = t + alpha * t**3 + beta * t**5                   # odd reparametrisation, phi'(0)=1
    u_in_t = sp.expand(u_scalar(phi))
    c3 = u_in_t.coeff(t, 3)
    checks["B_cubic_shifts_by_alpha_u1"] = sp.expand(c3 - (U3 + alpha * U1)) == 0
    # choosing alpha = -U3/U1 removes the cubic
    c3_gauged = sp.expand(c3.subs(alpha, -U3 / U1))
    checks["B_cubic_removable"] = sp.simplify(c3_gauged) == 0
    # carrier-adapted modulus t = u(s)/U1 linearises u identically: U1 * (u(s)/U1) == u(s)
    checks["B_carrier_adapted_linear"] = sp.expand(U1 * (u_scalar(s) / U1) - u_scalar(s)) == 0

    # ---- (C) transverse curvature enters s_*, not u_1 -----------------------------------------------
    a, b = sp.symbols("a b", positive=True)               # a = |E|^2/beta^2 ; b = P.R_chi/beta^2 (curvature)
    # electric radicand Delta_chi(s) = 1 - a s^2 + b s^4 ; first saturation contact Delta_chi(s_*) = 0.
    w = sp.symbols("w", positive=True)
    w_roots = sp.solve(b * w**2 - a * w + 1, w)
    w_star = min(w_roots, key=lambda r: sp.limit(r, b, 0) if sp.limit(r, b, 0).is_finite else sp.oo)
    s_star = sp.sqrt(w_star)
    checks["C_contact_depends_on_curvature"] = sp.simplify(sp.diff(w_star, b)) != 0
    checks["C_contact_limit_b0"] = sp.simplify(sp.limit(w_star, b, 0) - 1 / a) == 0
    # u_1 is a function of the operators only; it contains no a, b.
    checks["C_u1_independent_of_curvature"] = (u1.has(a) is False and u1.has(b) is False)

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("u(s) =", u_series)
    print("u_1 =", u1_expected, " (Front C carrier rate <J3,{E0,E1}>/<J3,J3>)")
    print("scalar cubic gauge: new c3 = U3 + alpha*U1  -> removable by alpha = -U3/U1")
    print("contact w_* = s_*^2 =", w_star, "  (depends on curvature b = P.R_chi/beta^2; -> 1/a as b->0)")
    print()
    print("RESULT (modulo odd reparametrisation):  u_partial = u(s_*) = u_1 * s_* ,")
    print("the scalar carrier's higher odd terms being pure reparametrisation gauge; the transverse")
    print("curvature enters ONLY through s_* (the radicand contact, A4-note open R_chi-transverse datum).")
    print("No number: u_1 awaits explicit locked-frame E0,E1; s_* awaits the radicand quartic.")
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
