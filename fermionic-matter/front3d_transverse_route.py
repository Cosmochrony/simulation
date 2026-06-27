"""Front 3c -> Front 2 bridge: transverse metaplectic route audit.

Bias-independent, exact symbolic verification (no sampling). Front 3c reduced the non-triviality of the chiral
polar class to one question: does the metaplectic generator of U_Pi have a non-zero TRANSVERSE (off-diagonal)
component in T_[I](U(1)^3_R \\ U(3) / U(1)^3_L)? On the derived real cascade the answer is no. The bridge to
Front 2 asks whether the FULL-TOWER complex metaplectic phase can produce a transverse component, or whether the
spin-stratum structure confines the metaplectic datum to the diagonal (Cartan / orientation / N_A) channel.

Correct U_Pi generator. U_Pi is the CHIRAL (J_Pi swaps L<->R) and UNITARY polar factor, so its generator is the
J_Pi-ODD and ANTI-HERMITIAN part of the lifted cascade generator:
    A := antiherm( J_Pi-odd( lift(M) ) ),     M in sl_2 (metaplectic step, coefficients possibly complex).
(The J_Pi-odd part that is HERMITIAN/symmetric is the split-generating, longitudinal piece -- it builds the
diagonal level operator E_Pi^2 = diag(1, 1/2+u, 1/2-u) and is rotated away in the generation eigenbasis; it is
NOT a U_Pi direction. This is why a real cascade, whose J_Pi-odd part is real symmetric, gives U_Pi = I.)

Two transverse blocks on C^3_gen = Sym^2(C^2), basis (e_0, e_+, e_-):
    INNER block  (e_0 <-> e_+, e_0 <-> e_-): central generation mixing with the outer pair;
    OUTER block  (e_+ <-> e_-) = R_mix:      mixing of the two outer generations.

Corpus anchoring. AAR (Beau2026aar): N_A is the J_3 / Cartan oriented symplectic area alpha = ts -- the DIAGONAL
channel, with R_mix "a distinct direction". Schur (PRS / schur_transversality_alpha.py): real cascade gives
alpha != 0 (J_3) and mu = 0 (R_mix). EBJ (Beau2026ebj): mixing needs a complex metaplectic phase. AOG
(Beau2026aog, prop:spinrigidity, lem:rigidity): the spin-stratum type-rigidity fixes sqrt(5) and closes
[H-orient] -- it constrains the ARITHMETIC TYPE / orientation, not the transverse polar direction.

Results (all exact symbolic).
  (A) Real-cascade collapse: for a real sl_2 element the generator A = antiherm(J_Pi-odd(lift)) is identically
      zero (both blocks), so [U_Pi] = [I] -- no mixing (recovers the Front 3c corollary and PRS v = 0).
  (B) Complex phase sources the INNER block: for a complex metaplectic element the inner component
      A_{e_0,e_+} = sqrt(2)\,i\,(Im p + Im q)/2 != 0 (carried by the imaginary part = the complex phase), while
      the OUTER component A_{e_+,e_-} (R_mix) = 0 for ANY complex sl_2 element. So a complex metaplectic phase
      DOES produce transverse mixing, but only in the inner (e_0 <-> e_+/-) block.
  (C) Inner mixing vs N_A -- same carrier, independent components: the SAME metaplectic step M carries both the
      N_A oriented area (alpha = J_3 projection, sourced by the REAL part) and the inner mixing (sourced by the
      IMAGINARY part); alpha depends only on the real product, the inner mixing only on the imaginary phase, so
      fixing N_A does NOT fix the inner mixing. The two SEPARATE as data (N_A real area vs the complex phase).
  (D) OUTER block needs a new stratum: the Sym^2 lift has identically zero outer (e_+, e_-) entries for ANY
      complex M, and R_mix is linearly independent of span{lift(E), lift(F), lift(H), I_3} (rank 4 -> 5). The
      e_+ <-> e_- mixing therefore requires a generator outside sl_2 (+) centre -- a genuinely new tower stratum.
  (E) Central metaplectic phase is even and projectively trivial (scalar c I_3): no alpha, no mixing, removed by
      the common-phase quotient. So the CHO central phase zeta_q^{Delta A_c} alone produces no mixing.
  (F) Orientation / sqrt(5) data give no transverse: any diagonal generator diag(d_0, d_+, d_-) (complex), the
      home of the orientation / sqrt(5) arithmetic that AOG fixes, has zero off-diagonal generator -- the AOG
      rigidity channel cannot produce mixing in either block.

Bridge verdict (printed). NOT outcome 1 (the transverse is not killed): a genuinely complex metaplectic phase
sources INNER-block mixing at the present stratum. Closest to outcome 3 (the transverse datum is INDEPENDENT of
N_A): N_A is the real oriented area, the inner mixing is the independent imaginary phase of the SAME cascade
carrier, and the OUTER block needs a separate new-stratum datum; N_A / epsilon can be fixed without predicting
the mixing. NOT outcome 2 (N_A's real area does not by itself feed the mixing). Whether the derived full-tower
phase is genuinely complex (-> inner mixing) or real (-> the derived-cascade collapse, no mixing), and whether a
new stratum opens the outer block, are the open questions -- gated by the same AOG lem:rigidity as epsilon = 1/10.
No mass and no mixing value is produced. No figures. English.
"""

import sympy as sp


def fundamental_generators():
    E = sp.Matrix([[0, 1], [0, 0]])
    F = sp.Matrix([[0, 0], [1, 0]])
    H = sp.Matrix([[1, 0], [0, -1]])
    return E, F, H


def sym2_lift(M):
    """Derived Sym^2 representation of a 2x2 traceless M, basis (e_0, e_+, e_-), e_0 = sqrt(2) v_+ v_-."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    s2 = sp.sqrt(2)
    return sp.Matrix([
        [a + d,  s2 * c,  s2 * b],
        [s2 * b, 2 * a,   0],
        [s2 * c, 0,       2 * d],
    ])


def hs_inner(A, B):
    return sp.trace(A.conjugate().T * B)


def J_invol():
    return sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])   # e_0 -> -e_0, e_+ <-> e_-


def jpi_odd_part(A):
    J = J_invol()
    return (A - J * A * J.inv()) / 2


def antiherm(A):
    return (A - A.conjugate().T) / 2


def Upi_generator(M):
    """The chiral, unitary U_Pi generator: J_Pi-odd AND anti-hermitian part of the Sym^2 lift of M."""
    return antiherm(jpi_odd_part(sym2_lift(M)))


def vec(M):
    return sp.Matrix([M[i, j] for i in range(3) for j in range(3)])


def main():
    checks = {}
    E, F, H = fundamental_generators()
    J3 = sp.diag(0, 1, -1)
    Rmix = sp.Matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    I3 = sp.eye(3)
    p, q, r = sp.symbols("p q r")                       # generic COMPLEX sl_2 coefficients

    def offdiag_zero(A):
        return all(sp.simplify(sp.expand_complex(A[i, j])) == 0
                   for i in range(3) for j in range(3) if i != j)

    # ---- (A) real-cascade collapse: A = 0 (both blocks) => [U_Pi] = [I] --------------------------
    pr, qr, rr = sp.symbols("pr qr rr", real=True)
    A_real = Upi_generator(pr * E + qr * F + rr * H)
    checks["A_real_generator_zero"] = all(sp.simplify(sp.expand_complex(e)) == 0 for e in A_real)

    # ---- (B) complex phase sources the INNER block, never the OUTER block ------------------------
    # split each coefficient into explicit real + imaginary parts
    pR, pI, qR, qI, rR, rI = sp.symbols("pR pI qR qI rR rI", real=True)
    rep = {p: pR + sp.I * pI, q: qR + sp.I * qI, r: rR + sp.I * rI}
    A_cpx = Upi_generator(p * E + q * F + r * H)
    inner = sp.simplify(sp.expand_complex(A_cpx[0, 1].subs(rep)))     # (e_0, e_+)
    outer = sp.simplify(sp.expand_complex(A_cpx[1, 2].subs(rep)))     # (e_+, e_-) = R_mix
    checks["B_inner_is_imaginary_phase"] = sp.simplify(inner - sp.sqrt(2) * sp.I * (pI + qI) / 2) == 0
    checks["B_inner_nonzero_for_phase"] = inner.subs({pI: 1, qI: 1, pR: 0, qR: 0, rR: 0, rI: 0}) != 0
    checks["B_outer_zero_any_M"] = outer == 0

    # ---- (C) inner mixing vs N_A: same carrier, independent (Re vs Im) components ----------------
    # N_A channel = J_3 projection of the J_Pi-odd part (the oriented area alpha), real cascade:
    t, s = sp.symbols("t s", real=True)
    odd_real = jpi_odd_part(sym2_lift(t * E + s * F + (t * s / 2) * H))
    alpha = sp.simplify(hs_inner(odd_real, J3) / hs_inner(J3, J3))   # the N_A / oriented-area datum
    checks["C_NA_is_real_area"] = sp.simplify(alpha - t * s) == 0    # alpha = ts, real product (AAR)
    # inner mixing depends ONLY on the imaginary parts; it is invariant under the real parts pR, qR:
    checks["C_inner_indep_of_NA"] = (sp.simplify(sp.diff(inner, pR)) == 0
                                     and sp.simplify(sp.diff(inner, qR)) == 0
                                     and sp.simplify(inner.subs({pI: 0, qI: 0})) == 0)
    checks["C_NA_real_no_inner"] = sp.simplify(sp.expand_complex(A_real[0, 1])) == 0  # real carrier => no mixing

    # ---- (D) OUTER block needs a new stratum -----------------------------------------------------
    LE, LF, LH = sym2_lift(E), sym2_lift(F), sym2_lift(H)
    checks["D_lift_outer_zero"] = sp.simplify(sym2_lift(p * E + q * F + r * H)[1, 2]) == 0 \
        and sp.simplify(sym2_lift(p * E + q * F + r * H)[2, 1]) == 0
    basis4 = sp.Matrix.hstack(vec(LE), vec(LF), vec(LH), vec(I3))
    basis5 = sp.Matrix.hstack(basis4, vec(Rmix))
    checks["D_Rmix_outside_image"] = basis4.rank() == 4 and basis5.rank() == 5
    checks["D_J3_perp_Rmix"] = sp.simplify(hs_inner(J3, Rmix)) == 0

    # ---- (E) central metaplectic phase: even, projectively trivial -------------------------------
    c = sp.symbols("c")
    checks["E_central_generator_zero"] = all(sp.simplify(sp.expand_complex(e)) == 0
                                             for e in Upi_generator_safe_central(c, I3))

    # ---- (F) orientation / sqrt(5) diagonal data: no transverse ----------------------------------
    d0, dp, dm = sp.symbols("d0 dp dm")
    A_diag = antiherm(jpi_odd_part(sp.diag(d0, dp, dm)))
    checks["F_diag_no_transverse"] = offdiag_zero(A_diag)

    # ---------------------------------------------------------------------------------------------
    print("Front 3c -> Front 2 bridge: transverse metaplectic route audit (exact symbolic, no sampling)")
    print("=" * 100)
    print("  U_Pi generator A = antiherm(J_Pi-odd(lift(M)));  transverse blocks: INNER (e_0<->e_+/-), OUTER R_mix")
    print("  (A) real cascade: A = 0 (both blocks) => [U_Pi] = [I], no mixing  (Front 3c / PRS v=0)")
    print("  (B) complex phase: INNER A_{e0,e+} = sqrt(2) i (Im p + Im q)/2 != 0;  OUTER R_mix = 0 for ANY M")
    print("  (C) N_A = real oriented area alpha=ts;  inner mixing = imaginary phase => independent data")
    print("  (D) OUTER R_mix not in sl_2(+)centre (lift outer block = 0; rank 4->5) => needs a NEW stratum")
    print("  (E) central metaplectic phase c I_3: even, projectively trivial => no mixing")
    print("  (F) diagonal orientation/sqrt(5) data: zero off-diagonal generator => no mixing")
    print("-" * 100)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 100)
    print("BRIDGE VERDICT: NOT outcome 1 -- a genuinely complex metaplectic phase sources INNER-block mixing at")
    print("  the present stratum. Closest to outcome 3 -- the transverse datum is INDEPENDENT of N_A: N_A is the")
    print("  real oriented area (J_3), the inner mixing is the independent IMAGINARY phase of the same cascade")
    print("  carrier, and the OUTER (e_+<->e_-) block needs a separate NEW-stratum datum; so N_A/epsilon can be")
    print("  fixed without predicting the mixing. NOT outcome 2 -- N_A's real area does not by itself feed it.")
    print("  Open (gated by the same AOG lem:rigidity as epsilon=1/10): is the derived full-tower phase genuinely")
    print("  complex (-> inner mixing) or real (-> collapse, no mixing), and does a new stratum open the outer block?")
    print("  No mass and no mixing value is produced.")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


def Upi_generator_safe_central(c, I3):
    # central scalar generator c*I_3: J_Pi-odd part is zero (I_3 is J_Pi-even), so the U_Pi generator vanishes.
    return antiherm(jpi_odd_part(c * I3))


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
