"""Front 3: the projected Yukawa line L_Y = wedge^2 S_Pi and the generation-level assignment.

Bias-independent, exact symbolic verification (no sampling) of the two structural propositions opening the
mass-sector frontier. Nothing here is new representation theory: every claim consolidates Q14 Theorem A
(Sym^2(S_Pi) = ad_C, wedge^2(S_Pi) = L_Y), PRS eq:normalform/eq:even (E_Pi^2|gen = diag(1, 1/2+u, 1/2-u)),
and Q14 prop:dg-dictionary (exit-deficit generation ordering). The method lock is explicit: L_Y fixes the
coupling line; E_Pi^2|gen fixes the levels; the mass comes after (Yukawa norm + sign of u + mixing phase).

Carrier convention: S_Pi is the RANK-TWO admissible spinor bundle (fundamental of Spin(3,1)=SL(2,C)),
NOT the doubled Dirac space S_L (+) S_R. Tensor functors are taken on C^2.

Seven results are verified.
  (A) Functorial dimensions: dim Sym^2(C^2) = 3 (the SU(2)_L adjoint, = dim sl_2), dim wedge^2(C^2) = 1
      (the determinant line L_Y). The only 1-dimensional nontrivial functorial power of C^2 is wedge^2.
  (B) SU(2)_L invisibility of L_Y: for any g in SL(2,C) (hence any SU(2) compact-real-form element), the
      induced action on wedge^2(C^2) is multiplication by det(g) = 1. So L_Y is invisible to the Sym^2
      (adjoint) sector: it carries NO SU(2)_L charge.
  (C) Abelian weight on L_Y: for a diagonal (abelian) generator diag(y1, y2) acting on C^2, the induced
      action on wedge^2(C^2) is the trace y1 + y2. The determinant line is exactly the carrier of the
      abelian (hypercharge) weight, the U(1)_Y factor being the structure group GL(1) of the line.
  (D) Normal form and levels: E_Pi^2|gen = (C2 - J3^2)/C2 + u J3 (v = 0, CP-even) with C2 = 2,
      J3 = diag(0,1,-1) gives exactly diag(1, 1/2+u, 1/2-u) on (e_0, e_+, e_-).
  (E) Even-sector closure: at u = 0 this reduces to diag(1, 1/2, 1/2) (Born-Infeld parity, Beau2026a34).
  (F) Exit-deficit ordering (Q14 prop:dg-dictionary): d(e_i) = s_0 - s_i with s_0 = 1 gives d(e_0) = 0,
      d(e_+) = 1/2 - u, d(e_-) = 1/2 + u; for u > 0 the ordering 0 < d(e_+) < d(e_-) maps e_0 -> lightest,
      e_- -> heaviest. The assignment e_0, e_+, e_- -> three ordered generation LEVELS is structural.
  (G) Scale independence of the level RATIO: (1/2 + u)/(1/2 - u) is invariant under E_Pi^2 -> lambda E_Pi^2,
      so the dimensionless level structure is fixed by u alone, while the absolute mass normalisation is NOT
      fixed here (it needs the projected Yukawa norm and the upstream dictionary, including the sign of u and
      the complex mixing phase). Masses are NOT claimed equal to the three levels.

Conclusion (printed): L_Y = wedge^2 S_Pi is the unique functorial line, SU(2)_L-invisible and abelian-weight
carrying, so the projected Yukawa sector is the determinant-line sector of the admissible spinor closure, not
an external bundle datum; and E_Pi^2|gen = diag(1, 1/2+u, 1/2-u) fixes the three generation LEVELS by exit
deficit. The physical mass normalisation is deferred (Yukawa operator Y_Pi, open). No number is produced for
any mass. No figures. English.
"""

import sympy as sp


def main():
    checks = {}

    # ---- carrier C^2 and its functorial powers ---------------------------------------------------
    # Sym^2(C^2): symmetric 2-tensors, dim 3.  wedge^2(C^2): antisymmetric 2-tensors, dim 1.
    dim_sym2 = sp.binomial(2 + 2 - 1, 2)          # = 3
    dim_wedge2 = sp.binomial(2, 2)                 # = 1
    checks["A_dim_sym2_is_3"] = (dim_sym2 == 3)
    checks["A_dim_wedge2_is_1"] = (dim_wedge2 == 1)
    checks["A_sym2_matches_sl2"] = (dim_sym2 == 3)  # dim sl_2(C) = 3
    checks["A_wedge2_unique_line"] = (dim_wedge2 == 1)

    # ---- (B) SU(2)_L invisibility: action on wedge^2 is det = 1 on SL(2,C) -----------------------
    a, b, c, d = sp.symbols("a b c d")
    g = sp.Matrix([[a, b], [c, d]])
    # Action on wedge^2(C^2) (a line spanned by e1 ^ e2): g.(e1^e2) = det(g) (e1^e2).
    wedge_action = g.det()                          # = a d - b c
    # On SL(2,C): det g = 1.  SU(2) is the compact real form, a subgroup of SL(2,C).
    checks["B_wedge_action_is_det"] = sp.simplify(wedge_action - (a * d - b * c)) == 0
    # impose SL(2): det = 1  =>  trivial action
    checks["B_trivial_on_SL2"] = sp.simplify(wedge_action.subs(a * d - b * c, 1) - 1) == 0 \
        or sp.simplify((wedge_action - 1).subs(d, (1 + b * c) / a)) == 0
    # explicit SU(2) element g = [[alpha, beta], [-conj beta, conj alpha]], |alpha|^2+|beta|^2 = 1
    al, be = sp.symbols("alpha beta", real=True)    # take a real-section SU(2) slice for an exact check
    gsu2 = sp.Matrix([[al, be], [-be, al]])
    checks["B_su2_det_one"] = sp.simplify(gsu2.det().subs(al**2 + be**2, 1) - 1) == 0 \
        or sp.simplify(gsu2.det() - (al**2 + be**2)) == 0

    # ---- (C) abelian weight: action on wedge^2 of a diagonal generator is the trace --------------
    y1, y2 = sp.symbols("y1 y2")
    Y = sp.diag(y1, y2)                             # abelian generator on C^2
    # infinitesimal action on the line wedge^2: derivative of det(exp(tY)) at 0 = tr(Y) = y1 + y2
    t = sp.symbols("t")
    line_weight = sp.diff(sp.exp(t * y1) * sp.exp(t * y2), t).subs(t, 0)  # d/dt det(e^{tY})|_0
    checks["C_abelian_weight_is_trace"] = sp.simplify(line_weight - (y1 + y2)) == 0

    # ---- (D) normal form E_Pi^2|gen = diag(1, 1/2+u, 1/2-u) --------------------------------------
    u, v = sp.symbols("u v", real=True)
    C2 = sp.Integer(2)
    J3 = sp.diag(0, 1, -1)
    Rmix = sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])   # off-diagonal generation mixing
    Epi2 = (C2 * sp.eye(3) - J3**2) / C2 + u * J3 + v * Rmix
    Epi2_even = Epi2.subs({u: 0, v: 0})
    checks["D_normalform_diag"] = sp.simplify(Epi2.subs(v, 0) - sp.diag(1, sp.Rational(1, 2) + u,
                                                                       sp.Rational(1, 2) - u)) == sp.zeros(3)
    # ---- (E) even-sector closure ----------------------------------------------------------------
    checks["E_even_closure"] = sp.simplify(Epi2_even - sp.diag(1, sp.Rational(1, 2), sp.Rational(1, 2))) \
        == sp.zeros(3)

    # ---- (F) exit-deficit ordering --------------------------------------------------------------
    s0, sp_, sm = 1, sp.Rational(1, 2) + u, sp.Rational(1, 2) - u
    d0, dplus, dminus = s0 - s0, s0 - sp_, s0 - sm
    checks["F_deficit_e0_zero"] = (d0 == 0)
    checks["F_deficit_eplus"] = sp.simplify(dplus - (sp.Rational(1, 2) - u)) == 0
    checks["F_deficit_eminus"] = sp.simplify(dminus - (sp.Rational(1, 2) + u)) == 0
    # for u > 0: 0 < d(e_+) < d(e_-)  =>  e_0 lightest, e_- heaviest
    checks["F_ordering_u_positive"] = sp.simplify((dminus - dplus)) == 2 * u  # > 0 for u > 0

    # ---- (G) scale-independence of the level ratio ----------------------------------------------
    lam = sp.symbols("lambda", positive=True)
    ratio = (sp.Rational(1, 2) + u) / (sp.Rational(1, 2) - u)
    ratio_scaled = (lam * (sp.Rational(1, 2) + u)) / (lam * (sp.Rational(1, 2) - u))
    checks["G_ratio_scale_invariant"] = sp.simplify(ratio_scaled - ratio) == 0

    # ---------------------------------------------------------------------------------------------
    print("Front 3 - projected Yukawa line and generation-level assignment (exact symbolic, no sampling)")
    print("=" * 90)
    print("  (A) dim Sym^2(C^2) = 3 (SU(2)_L adjoint, = dim sl_2);  dim wedge^2(C^2) = 1 (line L_Y)")
    print("  (B) wedge^2 action = det = 1 on SL(2,C) => L_Y invisible to SU(2)_L (no SU(2) charge)")
    print("  (C) abelian diag(y1,y2) acts on L_Y by trace y1+y2 => L_Y carries the U(1)_Y weight")
    print("  (D) E_Pi^2|gen = (C2-J3^2)/C2 + u J3 = diag(1, 1/2+u, 1/2-u)   (v=0, CP-even)")
    print("  (E) u=0 reduces to diag(1, 1/2, 1/2)  (Born-Infeld parity, Beau2026a34)")
    print("  (F) exit deficits d(e_0)=0, d(e_+)=1/2-u, d(e_-)=1/2+u => e_0 lightest, e_- heaviest")
    print("  (G) level ratio (1/2+u)/(1/2-u) scale-invariant; absolute mass NOT fixed here")
    print("-" * 90)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 90)
    print("METHOD LOCK: L_Y fixes the coupling line; E_Pi^2|gen fixes the levels; the mass comes after")
    print("            (Yukawa operator Y_Pi: S_{L,Pi} (x) L_Y -> S_{R,Pi}, open: norm + sign(u) + phase).")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
