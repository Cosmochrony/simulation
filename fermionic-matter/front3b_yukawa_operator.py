"""Front 3b: the squared Yukawa observable H_Pi = Y_Pi^dag Y_Pi and the chiral polar ambiguity.

Bias-independent, exact symbolic verification (no sampling) of the partial no-go that opens Front 3b: the
squared projective residue E_Pi^2|gen determines the HERMITIAN SQUARE of the Yukawa morphism, not the
morphism itself. The undetermined object is the chiral polar factor U_Pi, which carries the norm-branch,
the sign of u, and the mixing phase.

Setup. The projected Yukawa morphism is a chiral map Y_Pi : S_{L,Pi} (x) L_Y -> S_{R,Pi} (PYL, Beau2026pyl).
Its left hermitian square H_Pi := Y_Pi^dag Y_Pi acts on the LEFT generation carrier. The Schur-residue
sector fixes, on the gauge-singlet triplet C^3_gen, the dimensionless spectral part
    H_Pi|gen = lambda_Y^2 diag(1, 1/2+u, 1/2-u)        (~ E_Pi^2|gen, PRS Beau2026prs / PYL Beau2026pyl),
with lambda_Y the overall (undetermined) Yukawa norm. PRS gives E_Pi = -M^dag M (negative semi-definite),
so the positive square root is -E_Pi|gen.

The polar no-go is proved for a GENERIC positive H with singular values (a, b, c) > 0 (so it does not depend
on the particular level values), and the level identification a^2 : b^2 : c^2 = 1 : (1/2+u) : (1/2-u) is
checked separately.

Results (all exact symbolic).
  (A) Level identification: with singular values (a,b,c) = (lambda, lambda sqrt(1/2+u), lambda sqrt(1/2-u)),
      H = diag(a^2,b^2,c^2) = lambda^2 diag(1, 1/2+u, 1/2-u), so Spec(Y^dag Y) = lambda^2 {1,1/2+u,1/2-u}.
  (B) PRS consistency: E_Pi = -M^dag M negative s.d. and (-E_Pi|gen)^2 = E_Pi^2|gen = diag(1,1/2+u,1/2-u),
      so the positive root H^{1/2}/lambda = -E_Pi|gen.
  (C) Polar non-uniqueness (NO-GO): for a generic positive H = diag(a^2,b^2,c^2) and ANY unitary U,
      Y = U H^{1/2} satisfies Y^dag Y = H. Distinct U give distinct Y with identical H, so H does not fix Y.
  (D) The free factor is the R-side / chiral polar unitary: under Y -> U Y, (U Y)^dag (U Y) = Y^dag Y, so
      H is blind to U. U_Pi is exactly this undetermined chiral polar map.
  (E) CP-real diagonal branch: U = I gives Y = H^{1/2}, real, diagonal, NO mixing (the v = 0 CP-even branch
      of PRS); a non-trivial U (rotation or phase) injects off-diagonal / complex structure absent from H.
  (F) Mixing invisibility: the rotation family U(theta) leaves H invariant for all theta while Y(theta)
      sweeps a one-parameter family -- the mixing lives entirely in U_Pi.

Conclusion (printed): E_Pi^2|gen fixes the squared Yukawa levels, but the chiral polar factor U_Pi -- with
it the norm-branch, sign(u), and the mixing phase -- is free. Front 3b can close H_Pi = Y_Pi^dag Y_Pi; it
cannot close Y_Pi unless U_Pi is fixed (Front 3c). No mass value is produced. No figures. English.
"""

import sympy as sp


def _zero(M):
    """Robust matrix-zero test: complex- and trig-aware simplification entrywise."""
    return all(sp.simplify(sp.trigsimp(sp.expand_complex(e))) == 0 for e in M)


def dag(M):
    return M.conjugate().T


def main():
    checks = {}

    a, b, c, th, ph = sp.symbols("a b c theta phi", positive=True)   # generic singular values > 0
    u = sp.symbols("u", real=True)
    lam = sp.symbols("lambda", positive=True)
    I3 = sp.eye(3)

    # generic positive H with singular values (a,b,c)
    Hhalf = sp.diag(a, b, c)
    H = Hhalf * Hhalf                                                 # = diag(a^2,b^2,c^2), positive

    # unitary right-carrier maps (chiral polar factor candidates)
    U_rot = sp.Matrix([[1, 0, 0],
                       [0, sp.cos(th), -sp.sin(th)],
                       [0, sp.sin(th), sp.cos(th)]])                  # real generation rotation (2,3)
    U_ph = sp.diag(1, sp.exp(sp.I * ph), 1)                           # diagonal complex phase

    # ---- (A) level identification: a^2:b^2:c^2 = 1:(1/2+u):(1/2-u) -------------------------------
    sub = {a: lam, b: lam * sp.sqrt(sp.Rational(1, 2) + u), c: lam * sp.sqrt(sp.Rational(1, 2) - u)}
    H_levels = H.subs(sub)
    target = lam**2 * sp.diag(1, sp.Rational(1, 2) + u, sp.Rational(1, 2) - u)
    # compare on the assumption 0 < u < 1/2 so that the sqrt-squares resolve
    asm = sp.Q.positive(sp.Rational(1, 2) + u) & sp.Q.positive(sp.Rational(1, 2) - u)
    checks["A_levels"] = all(sp.refine(sp.simplify(H_levels[i, i] - target[i, i]), asm) == 0
                             for i in range(3))
    checks["A_spectrum_distinct"] = sp.simplify((sp.Rational(1, 2) + u) - (sp.Rational(1, 2) - u)) == 2 * u

    # ---- (B) PRS consistency: (-E_Pi|gen)^2 = E_Pi^2|gen ----------------------------------------
    Epi_gen = -sp.diag(1, sp.sqrt(sp.Rational(1, 2) + u), sp.sqrt(sp.Rational(1, 2) - u))  # negative s.d.
    checks["B_Epi_sq"] = all(sp.refine(sp.simplify((Epi_gen * Epi_gen)[i, i]
                             - sp.diag(1, sp.Rational(1, 2) + u, sp.Rational(1, 2) - u)[i, i]), asm) == 0
                             for i in range(3))
    checks["B_posroot"] = _zero((-Epi_gen) - sp.diag(1, sp.sqrt(sp.Rational(1, 2) + u),
                                                     sp.sqrt(sp.Rational(1, 2) - u)))

    # ---- (C) polar non-uniqueness (the NO-GO) ---------------------------------------------------
    Y0, Y_rot, Y_ph = Hhalf, U_rot * Hhalf, U_ph * Hhalf
    checks["C_rot_same_H"] = _zero(dag(Y_rot) * Y_rot - H)
    checks["C_ph_same_H"] = _zero(dag(Y_ph) * Y_ph - H)
    checks["C_Y_differs_rot"] = not _zero(Y_rot - Y0)
    checks["C_Y_differs_ph"] = not _zero(Y_ph - Y0)

    # ---- (D) H blind to the left/R-side unitary -------------------------------------------------
    checks["D_rot_unitary"] = _zero(dag(U_rot) * U_rot - I3)
    checks["D_ph_unitary"] = _zero(dag(U_ph) * U_ph - I3)
    checks["D_H_blind"] = _zero(dag(U_rot * Y0) * (U_rot * Y0) - dag(Y0) * Y0)

    # ---- (E) CP-real diagonal branch vs mixing in U ---------------------------------------------
    Y_cp = I3 * Hhalf
    checks["E_cp_no_mixing"] = _zero(Y_cp - sp.diag(a, b, c))
    checks["E_mixing_in_U"] = sp.simplify(Y_rot[2, 1] - sp.sin(th) * b) == 0 and sp.sin(th) * b != 0

    # ---- (F) mixing invisibility ---------------------------------------------------------------
    checks["F_H_theta_invariant"] = _zero(dag(Y_rot) * Y_rot - H)
    checks["F_Y_theta_varies"] = not _zero(sp.diff(Y_rot, th))

    # ---------------------------------------------------------------------------------------------
    print("Front 3b - squared Yukawa observable and chiral polar ambiguity (exact symbolic, no sampling)")
    print("=" * 92)
    print("  Y_Pi : S_{L,Pi} (x) L_Y -> S_{R,Pi};   H_Pi := Y_Pi^dag Y_Pi  on the left carrier")
    print("  (A) levels a^2:b^2:c^2 = 1:(1/2+u):(1/2-u),  Spec(Y^dag Y) = lambda^2 {1, 1/2+u, 1/2-u}")
    print("  (B) PRS: E_Pi = -M^dag M negative s.d.,  (-E_Pi|gen)^2 = E_Pi^2|gen,  pos. root = H^{1/2}")
    print("  (C) NO-GO: Y = U H^{1/2} gives Y^dag Y = H for ANY unitary U => H does not fix Y")
    print("  (D) H is blind to the R-side / chiral polar unitary U_Pi: (U Y)^dag (U Y) = Y^dag Y")
    print("  (E) CP-real branch U=I: Y real diagonal, NO mixing; mixing/phase live only in U_Pi != I")
    print("  (F) H invariant under the rotation family U(theta) while Y(theta) sweeps a 1-param family")
    print("-" * 92)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 92)
    print("RESULT: E_Pi^2|gen fixes the squared Yukawa LEVELS, not the morphism. The chiral polar factor")
    print("        U_Pi (norm-branch, sign(u), mixing phase) is free. Front 3b closes H_Pi = Y_Pi^dag Y_Pi;")
    print("        it cannot close Y_Pi unless U_Pi is fixed (Front 3c).")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
