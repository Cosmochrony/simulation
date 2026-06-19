"""Front D: evaluation of <J_3, E_1>, the J_3-projection of the D^pm-transported chiral defect rate.

Bias-independent, exact symbolic verification (no sampling). The carrier rate reduced (front_d3_u1_reduction)
to u_1 = sqrt(2) sigma <J_3, E_1>/<J_3, J_3>, concentrating all non-trivial dependence in <J_3, E_1>. This
script evaluates that projection via the Sym^2 lift (the locked spin/chiral frame, modulo symbol
compatibility Q14: the principal symbol equals the Clifford symbol, so the D^pm transport acts as the
Sym^2 representation), answering the three tests.

Convention frozen to schur_transversality_alpha.py: V = C^2 with sl_2 generators E,F,H ([X,Y]=H=2J_3);
Sym^2(V) = C^3_gen = span(e_0,e_+,e_-); J_Pi^(2): e_0->-e_0, e_+ <-> e_-; J_3 = diag(0,1,-1);
R_mix = [[0,0,0],[0,0,1],[0,-1,0]].

Tests.
  (1) Reduction to defect: <J_3, E_1> depends only on the J_Pi-odd defect, not on the J_Pi-even chiral sum.
  (2) Generation projection: <J_3, E_1> = T_{J_3}(defect) = the Cartan (H/J_3) component of the Sym^2-lifted
      transported defect -- a CLOSED linear functional (the E,F off-diagonal content drops under the J_3
      projection). So the projection closes as a map, independent of the explicit Pi_S, D^pm beyond
      symbol-compatibility.
  (3) Norm vs sign: T_{J_3} evaluated on the oriented cascade is the oriented symplectic AREA
      alpha = ts r/sinh r, ORIENTATION-ODD (T_{J_3}(g) = -T_{J_3}(g^{-1}-ordered)), hence the spontaneous
      V-A sign -- NOT a positive norm. This is the AAR angular observable, dictionary-bound to N_A.

Conclusion (printed): closed factorisation u_partial = sqrt(2) sigma s_* T_{J_3}(d_s Delta_chi)/<J_3,J_3>,
with T_{J_3} the angular oriented area (= N_A, dictionary-bound) and s_* the radial saturation contact
(A4-note open R_chi curvature). The only open pieces are the already-known radial s_* and angular N_A;
no new selector. No figures. English.
"""

import sympy as sp


def fundamental():
    E = sp.Matrix([[0, 1], [0, 0]])
    F = sp.Matrix([[0, 0], [1, 0]])
    H = sp.Matrix([[1, 0], [0, -1]])
    return E, F, H


def sym2_lift(M):
    """Derived Sym^2 representation of a 2x2 sl_2 element, basis (e_0, e_+, e_-), e_0 = sqrt2 v+ v-."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    s2 = sp.sqrt(2)
    return sp.Matrix([
        [a + d,  s2 * c, s2 * b],
        [s2 * b, 2 * a,  0],
        [s2 * c, 0,      2 * d],
    ])


J = sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])   # J_Pi^(2)
J3 = sp.diag(0, 1, -1)
Rmix = sp.Matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]])


def hs(A, B):
    return sp.trace(A.conjugate().T * B)


def jpi_odd(A):
    return (A - J * A * J.inv()) / 2


def T_J3(M):
    """The J_3 projection of the J_Pi-odd part of the Sym^2 lift of an sl_2 defect M."""
    return sp.simplify(hs(jpi_odd(sym2_lift(M)), J3) / hs(J3, J3))


def main():
    E, F, H = fundamental()
    a, b, c, t, s = sp.symbols("a b c t s", real=True)
    checks = {}

    # ---- (1)+(2) generic defect M = aE + bF + cH ; T_J3 picks ONLY the Cartan coefficient c ----------
    M = a * E + b * F + c * H
    TJ3_generic = T_J3(M)
    checks["closed_cartan_only"] = sp.simplify(TJ3_generic - 2 * c) == 0          # = 2c, independent of a,b
    checks["independent_of_E"] = sp.diff(TJ3_generic, a) == 0
    checks["independent_of_F"] = sp.diff(TJ3_generic, b) == 0
    # The J_Pi-even chiral sum is the (E+F) direction; it contributes nothing to T_J3.
    checks["even_sum_drops"] = T_J3(E + F) == 0
    # off-diagonal E,F content also drops from J_3 (only the Cartan H reaches J_3).
    checks["EminusF_not_on_J3"] = T_J3(E - F) == 0
    # With real (phase-free) data the R_mix (mixing) channel is empty, consistent with mu == 0 of the
    # Schur audit: a complex metaplectic phase would be required to populate R_mix.
    checks["Rmix_empty_phasefree"] = sp.simplify(hs(jpi_odd(sym2_lift(a * E + b * F + c * H)), Rmix)) == 0

    # ---- (3) oriented cascade: T_J3 is the oriented symplectic area, orientation-ODD (V-A sign) ------
    r = sp.symbols("r", positive=True)
    cosh_r = 1 + t * s / 2
    g1 = sp.exp(t * E) * sp.exp(s * F)                                            # = [[1+ts, t],[s, 1]]
    g2 = sp.exp(s * F) * sp.exp(t * E)                                            # opposite ordering
    g1 = sp.simplify(g1)
    g2 = sp.simplify(g2)
    logg1 = (r / sp.sinh(r)) * (g1 - cosh_r * sp.eye(2))
    logg2 = (r / sp.sinh(r)) * (g2 - cosh_r * sp.eye(2))
    # Cartan coefficient is the (0,0) entry (coeff of H = diag(1,-1)); T_J3 = 2 * Cartan coeff.
    cartan1 = logg1[0, 0]
    cartan2 = logg2[0, 0]
    TJ3_g1 = sp.simplify(2 * cartan1)
    TJ3_g2 = sp.simplify(2 * cartan2)
    pref = r / sp.sinh(r)
    checks["cascade_area_value"] = sp.simplify(TJ3_g1 - t * s * pref) == 0        # = ts r/sinh r (alpha)
    checks["orientation_odd_VminusA"] = sp.simplify(TJ3_g1 + TJ3_g2) == 0          # T(g1) = -T(g2): not a norm
    # leading term is the oriented symplectic area ts (Carnot-degree-2 increment), sign-carrying.
    u = sp.symbols("u", positive=True)
    area_lead = sp.series(u * sp.acosh(1 + u / 2) / sp.sinh(sp.acosh(1 + u / 2)), u, 0, 2).removeO()
    checks["area_leading_is_ts"] = sp.simplify(area_lead - u) == 0

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("(1)+(2)  <J3,E1> = T_J3(defect) = 2 * (Cartan/J3 component of the Sym^2-lifted transported defect)")
    print("         -> CLOSED linear functional: depends only on the J_Pi-odd Cartan part of d_s Delta_chi,")
    print("            independent of the E,F off-diagonal content and of the J_Pi-even chiral sum.")
    print("(3)      on the oriented cascade T_J3 = ts*r/sinh r (oriented symplectic area), and")
    print("         T_J3(g1) = -T_J3(g2): ORIENTATION-ODD -> spontaneous V-A sign, NOT a positive norm.")
    print("         This is the AAR angular observable, dictionary-bound to N_A.")
    print()
    print("CLOSURE:  u_partial = sqrt(2) sigma s_* T_J3(d_s Delta_chi) / <J3,J3>,")
    print("  factors: sqrt(2) sigma (Born-Infeld even block, V-A) x s_* (radial saturation, A4 R_chi open)")
    print("           x T_J3 (angular oriented area = N_A, dictionary-bound). No new selector.")
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
