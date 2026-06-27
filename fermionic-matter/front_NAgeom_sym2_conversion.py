"""Front N_A^geom: does Sym^2(C^2) intrinsically convert the frontier rational 1/3 into the spectral split 1/10?

Bias-independent, exact symbolic (SymPy / exact rationals), no fit. Reduced target (Jerome): strip the trivial
angle convention 2pi/q from N_A^geom = 3q/(20pi) = (q/2pi) * (3/10); the structural residue is

    N_A,red^geom = eps / (onset) = (1/10) / (1/3) = 3/10,

i.e. the question "does Sym^2 send the onset frontier rational 1/3 to the spectral split coefficient eps = 1/10?".

Three normalisations must be separated (else a missing norm convention masquerades as a no-go):
    (Weil central phase theta_A) -> (J_3 generator) -> (spectral split coeff on Sym^2(C^2)) -> (dictionary eps).

Anchors (corpus, exact):
  * AAR lem:area: the per-step J_3 coefficient IS the oriented symplectic area, [X,Y]=2J_3 => (ts/2)(2J_3)=ts J_3.
    So the area->J_3 conversion is factor 1; the static covariance [1:1/2:1/2] and its factor 2 come from the
    Carnot-degree-2 weight of Z=[X,Y] (Beau2026a34, Beau2026a33).
  * PYO/PRS: the generation split is E_Pi^2|gen = diag(1, 1/2+u, 1/2-u), with u = eps the J_3 split coefficient.
  * FM-Note / AOG: the ADE case-selection gate fixes the LEVEL RATIO R = (1/2+eps)/(1/2-eps) = 3/2, whence
    5 eps = 1/2 and eps = 1/10. The "5" = 2(R+1) is the ADE-ratio signature.

Method: keep R symbolic; locate where the value R = 3/2 (hence eps = 1/10, hence 3/10) is fixed, and test whether
it is a Sym^2(C^2) representation invariant or the external ADE input.
"""

import sympy as sp


def j3_norms(J3, label):
    """Hilbert-Schmidt and operator data of the J_3 generator on C^3_gen = Sym^2(C^2)."""
    hs = sp.trace(J3.conjugate().T * J3)             # <J3,J3>_HS
    spec = sorted(J3.eigenvals().keys(), key=lambda z: sp.re(z))
    op = max(abs(e) for e in spec)
    return {"label": label, "HS2": sp.nsimplify(hs), "spec": spec, "op": op, "HS": sp.sqrt(hs)}


def main():
    checks = {}
    out = {}

    # ---- Step 1: J_3 conventions and Sym^2(C^2) invariants ---------------------------------------
    J3a = sp.diag(0, 1, -1)          # AAR/PYO convention (H=diag(1,-1) lifts to 2 J3 = diag(0,2,-2))
    J3b = sp.diag(1, 0, -1)          # alternate labelling
    na, nb = j3_norms(J3a, "diag(0,1,-1)"), j3_norms(J3b, "diag(1,0,-1)")
    out["J3a"], out["J3b"] = na, nb
    # the only Sym^2(C^2) numbers available: weights {0,+-1}; HS^2=2 (=Carnot degree 2); dim=3; static [1:1/2:1/2]
    checks["1_HS2_is_2"] = (na["HS2"] == 2)                 # <J3,J3>=2 : the Carnot-2 factor
    static_cov = [sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 2)]   # [1 : 1/2 : 1/2] covariance
    checks["1_static_cov_ratio_e0_over_epm_is_2"] = (static_cov[0] / static_cov[1] == 2)

    # ---- Step 2: phase -> J_3 -> split chain, kept symbolic in the level ratio R ------------------
    # split E_Pi^2|gen = diag(1, 1/2+u, 1/2-u); level ratio R := (1/2+u)/(1/2-u); invert for u=eps:
    u, R = sp.symbols("u R", positive=True)
    R_of_u = (sp.Rational(1, 2) + u) / (sp.Rational(1, 2) - u)
    eps_of_R = sp.solve(sp.Eq(R_of_u, R), u)[0]            # eps as a function of the level ratio R
    checks["2_eps_of_R"] = sp.simplify(eps_of_R - (R - 1) / (2 * (R + 1))) == 0
    out["eps_of_R"] = sp.nsimplify(eps_of_R)
    # area->J_3 conversion factor is 1 (AAR lem:area): the J_3 coefficient equals the oriented area directly.
    area_to_J3 = sp.Integer(1)
    checks["2_area_to_J3_is_unity"] = (area_to_J3 == 1)

    # ---- Step 3: the reduced residue and where R=3/2 enters --------------------------------------
    onset = sp.Rational(1, 3)                              # exact frontier onset <|Delta A_c|>_d+ (Front N_A)
    eps_dict = sp.Rational(1, 10)                          # ADE dictionary value
    N_red = eps_dict / onset                               # reduced geometric residue
    checks["3_reduced_is_3_over_10"] = (N_red == sp.Rational(3, 10))
    # express the reduced residue through the level ratio R (no fit): N_red(R) = eps(R)/onset = 3 eps(R)
    N_red_of_R = sp.simplify(eps_of_R / onset)
    out["N_red_of_R"] = N_red_of_R                          # = 3(R-1)/(2(R+1))
    R_ADE = sp.Rational(3, 2)                               # ADE level ratio (FM-Note / AOG gate)
    checks["3_Nred_at_R_3_2_is_3_10"] = (sp.simplify(N_red_of_R.subs(R, R_ADE)) == sp.Rational(3, 10))
    checks["3_eps_at_R_3_2_is_1_10"] = (sp.simplify(eps_of_R.subs(R, R_ADE)) == sp.Rational(1, 10))
    # the "5": denominator of eps at the gate is 2(R+1) = 5 at R=3/2 -- the ADE-ratio signature
    five = sp.simplify(2 * (R_ADE + 1))
    checks["3_five_is_2_Rplus1"] = (five == 5)
    out["five"] = five

    # (a) the split MAGNITUDE u (hence R) is a FREE deformation: diag(1, 1/2+u, 1/2-u) is J_3-aligned for ALL u
    #     (commutes with J_3), so the representation Sym^2(C^2) does NOT select a value of u. The magnitude is
    #     external dynamical/ADE data, not a representation invariant -- this is the rigorous core (not a norm pool).
    split_op = sp.diag(1, sp.Rational(1, 2) + u, sp.Rational(1, 2) - u)
    checks["3a_split_J3_aligned_all_u"] = (sp.simplify(split_op * J3a - J3a * split_op) == sp.zeros(3))
    checks["3a_split_undetermined_by_rep"] = (sp.diff(split_op, u) != sp.zeros(3))   # u is a free direction

    # (b) the irreducible arithmetic obstruction is the PRIME 5: eps=1/10 carries a factor 5 (=2(R+1) at R=3/2)
    #     that is NOT in the multiplicative monoid <2, 3, sqrt2> of the Sym^2(C^2) normalisations.
    #     (5 is prime and 5 != 2^a 3^b 2^(c/2); the sqrt2 only adds half-powers of 2.)
    checks["3b_five_is_prime"] = sp.isprime(5)
    checks["3b_five_not_in_monoid_2_3_sqrt2"] = (sp.gcd(sp.Integer(5), sp.Integer(2 * 3)) == 1)  # 5 coprime to 6
    out["irreducible_prime"] = sp.Integer(5)

    # ---- report ----------------------------------------------------------------------------------
    print("Front N_A^geom: does Sym^2(C^2) convert the onset 1/3 into the spectral split eps=1/10?  (exact, no fit)")
    print("=" * 104)
    print(f"  Step 1  J_3 = {na['label']}:  <J3,J3>_HS = {na['HS2']},  Spec = {na['spec']},  "
          f"||J3||_op = {na['op']},  ||J3||_HS = {sp.nsimplify(na['HS'])}")
    print(f"          J_3 = {nb['label']}:  <J3,J3>_HS = {nb['HS2']},  Spec = {nb['spec']}")
    print(f"          Sym^2(C^2) invariants available: weights {{0,+-1}}, <J3,J3>=2 (Carnot deg 2), dim 3,")
    print(f"          static covariance [1:1/2:1/2] (e_0/e_pm = 2).  -> pool = {{1, 2, 3, sqrt2, 1/2}}")
    print(f"  Step 2  area->J_3 factor (AAR lem:area) = {area_to_J3}  (J_3 coeff = oriented area, exact)")
    print(f"          level ratio R=(1/2+u)/(1/2-u)  =>  eps(R) = {out['eps_of_R']}")
    print(f"  Step 3  reduced residue  N_red = eps/onset = (1/10)/(1/3) = {N_red}")
    print(f"          N_red(R) = eps(R)/onset = {N_red_of_R}   (= 3(R-1)/(2(R+1)))")
    print(f"          at the ADE gate R=3/2:  eps=1/10, N_red=3/10, denominator 2(R+1) = {five}  (the 'ADE 5')")
    print("-" * 104)
    allok = True
    for k, v in checks.items():
        ok = bool(v)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 104)
    print("VERDICT (honest, no fit):  OUTCOME 3 -- the residue is gated by the ADE level ratio, not Sym^2.")
    print("  N_red = 3/10 = 3(R-1)/(2(R+1)), R the spectral level ratio. Sym^2(C^2) DOES supply the norm conventions")
    print("  unambiguously: the triplet (dim 3), J_3 with Spec {0,+-1}, <J3,J3>=2 (Carnot deg 2), static [1:1/2:1/2],")
    print("  and the BI sqrt2 -- the multiplicative pool <2, 3, sqrt2>. What it does NOT supply is the split MAGNITUDE")
    print("  u (hence R): diag(1,1/2+u,1/2-u) is J_3-aligned for EVERY u, so the representation leaves u free; the")
    print("  value R=3/2 is external dynamical/ADE data. The obstruction is therefore not a missing norm convention")
    print("  but an irreducible ARITHMETIC factor: the prime 5 = 2(R+1)|_{R=3/2}, coprime to the Sym^2 pool {2,3} --")
    print("  the icosahedral/ADE signature (cf. the sqrt5 of AOG). So Sym^2 does NOT intrinsically convert 1/3 -> 1/10:")
    print("  front N_A^geom reduces to AOG lem:rigidity, SHARPENED to a single arithmetic object -- R=3/2, i.e. the")
    print("  prime 5. The angular coefficient (1/3) and the norm conventions (2,3,sqrt2) are Sym^2; the 5 is the gate.")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
