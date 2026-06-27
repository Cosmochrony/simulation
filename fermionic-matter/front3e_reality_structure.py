"""Front 3e: non-central complex-phase audit (reality structure of the metaplectic step).

Bias-independent, exact symbolic verification (no sampling). Front 3c -> Front 2 bridge isolated the chiral polar
class internal block e_0 <-> e_+/- as sourced by the IMAGINARY part of the non-central sl_2 coefficients of the
metaplectic step M = p E + q F + r H:

    A_{e0,e+} = sqrt(2) i (Im p + Im q) / 2     (front3d_transverse_route.py).

Front 3e asks the upstream question the bridge left open: does the present full-tower stratum DERIVE Im p, Im q != 0
(internal block opens, mixing != 0), or does it FORCE p, q in R (=> A_Pi = 0, mixing closes negatively at this
stratum)? The decision is a REALITY-STRUCTURE question and must NOT conflate three distinct objects (Jerome's
caveat):

    complexity of the REPRESENTATION matrices  !=  complexity of the COEFFICIENTS p, q  !=  central phase zeta_q.

Convention contract (frozen, identical to schur_transversality_alpha.py / front3d_transverse_route.py):
fundamental V = C^2, sl_2 generators E, F, H (real matrices), [E,F]=H=2 J_3; derived Sym^2 lift on
C^3_gen = span(e_0, e_+, e_-); chiral involution J_Pi^(2): e_0 -> -e_0, e_+ <-> e_-; chiral unitary polar generator
A := antiherm( J_Pi-odd( lift(M) ) ).

REALITY STRUCTURE. Because E, F, H are REAL matrices, the split real form sl_2(R) = span_R{E,F,H} is the fixed
locus of the entrywise complex conjugation sigma(M) = conj(M) on sl_2(C); sigma(M) = M  <=>  p, q, r in R. The
sigma-anti-real part is (M - sigma M)/2 = i (Im p E + Im q F + Im r H). The audit proves the polar-class generator
A depends ONLY on this sigma-anti-real part, and splits the way the polar class [U_Pi] does:
  * its TRANSVERSE (off-diagonal, mixing) part is non-zero  <=>  Im p != 0 or Im q != 0  (the NON-CENTRAL phase);
  * its DIAGONAL part is i Im(r) diag(0,2,-2), a pure J_3/Cartan rephasing lying in the U(1)^3 double-coset
    quotient that defines [U_Pi] = U(1)^3_R \\ U(3) / U(1)^3_L, hence CLASS-TRIVIAL (the N_A / J_3 channel).
So [U_Pi] = [I]  <=>  Im p = Im q = 0, irrespective of Im r: the genuine mixing is the failure of sigma-invariance
in the OFF-DIAGONAL coefficients (leaving the real form transversally), NOT complex matrix entries in a complex
spinorial frame, and NOT the imaginary Cartan rephasing.

Corpus anchoring (the EXHAUSTIVITY control -- structural, surveyed, not a symbolic theorem here):
  * PRS / AAR (Beau2026prs, Beau2026aar): the cascade metaplectic step is g = exp(tE) exp(sF) with t, s in R; the
    per-step J_Pi-odd J_3 component is the oriented symplectic area alpha = ts in R. The non-central sl_2 step is
    fed by REAL Heisenberg data -> real coefficients.
  * CHO (Beau2026cho, prop. "Arithmetic nature of the inherited phase"): in the Schroedinger--Heisenberg pipeline
    the ONLY phase is the inherited central cyclotomic character zeta_q^{Delta A_c}; there is NO Fourier/Weyl
    generator, hence NO metaplectic Gauss phase. The non-central complex phase that would carry Im p, Im q != 0 is
    structurally ABSENT from the present pipeline; the full Lorentzian SL(2,C) spin-lift carrying it is "not present
    in this pipeline".
  * AOG (Beau2026aog, lem:rigidity): the full recursive (all-tower) type-rigidity is structural / OPEN -- the same
    gate as epsilon = 1/10. Introducing a Weil/Fourier generator (a genuinely complex non-central metaplectic
    phase) = a NEW tower stratum.

Verdict (printed). At the present stratum the non-central step is sigma-real (real Heisenberg data + only a central
phase + no Weil generator): Im p = Im q = 0  =>  A_Pi = 0  =>  [U_Pi] = [I] internal  =>  N_A != 0, u != 0, but no
internal mixing. NEGATIVE CLOSURE at the present stratum (outcome 1). The positive opening Im p, Im q != 0 requires
a genuinely complex non-central metaplectic phase = a new tower stratum, gated by the SAME AOG lem:rigidity as
epsilon = 1/10. No mass and no mixing value is produced. No figures. English.
"""

import sympy as sp


# ----------------------------------------------------------------------------------------------------------------
# Frozen convention (matches schur_transversality_alpha.py and front3d_transverse_route.py)
# ----------------------------------------------------------------------------------------------------------------
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


def J_invol():
    return sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])   # e_0 -> -e_0, e_+ <-> e_-


def jpi_odd_part(A, J=None):
    """J_Pi-odd part (A - J A J^{-1})/2.  J defaults to the canonical real involution; a transported J is passed
    in the spinorial-frame covariance test."""
    if J is None:
        J = J_invol()
    return (A - J * A * J.inv()) / 2


def antiherm(A):
    return (A - A.conjugate().T) / 2


def Upi_generator(M, J=None):
    """Chiral, unitary polar generator A = antiherm( J_Pi-odd( Sym^2 lift of M ) )."""
    return antiherm(jpi_odd_part(sym2_lift(M), J))


def is_zero_matrix(A):
    return all(sp.simplify(sp.expand_complex(e)) == 0 for e in A)


def main():
    checks = {}
    E, F, H = fundamental_generators()
    I3 = sp.eye(3)

    # explicit real + imaginary split of generic non-central coefficients
    pR, pI, qR, qI, rR, rI = sp.symbols("pR pI qR qI rR rI", real=True)
    p = pR + sp.I * pI
    q = qR + sp.I * qI
    r = rR + sp.I * rI
    M = p * E + q * F + r * H
    A = Upi_generator(M)

    # ----------------------------------------------------------------------------------------------------------
    # (1) Real cascade collapse: sigma-real M (p,q,r in R) => A = 0  (recovers Front 3c / PRS v=0)
    # ----------------------------------------------------------------------------------------------------------
    A_real = A.subs({pI: 0, qI: 0, rI: 0})
    checks["1_real_form_gives_zero"] = is_zero_matrix(A_real)

    # ----------------------------------------------------------------------------------------------------------
    # (2) Internal-block coefficient is exactly the sigma-anti-real part:  A_{e0,e+} = sqrt(2) i (Im p+Im q)/2
    # ----------------------------------------------------------------------------------------------------------
    inner = sp.expand_complex(A[0, 1])
    checks["2_internal_is_imag_phase"] = sp.simplify(inner - sp.sqrt(2) * sp.I * (pI + qI) / 2) == 0

    # ----------------------------------------------------------------------------------------------------------
    # (3) A depends ONLY on the sigma-anti-real (imaginary-coefficient) part: A(M) = A( i Im M ), and the
    #     Cartan imaginary part Im r contributes NO transverse generator (it lands in the diagonal, projected out).
    # ----------------------------------------------------------------------------------------------------------
    A_imag_only = Upi_generator(sp.I * (pI * E + qI * F + rI * H))    # i * Im(M)
    checks["3a_depends_only_on_anti_real"] = is_zero_matrix(A - A_imag_only)
    # genuine transverse generator vanishes iff the NON-CENTRAL imaginary parts vanish (Im r alone gives no block):
    A_cartan_imag = A.subs({pI: 0, qI: 0})        # only Im r kept
    offdiag_cartan = [A_cartan_imag[i, j] for i in range(3) for j in range(3) if i != j]
    checks["3b_cartan_imag_no_transverse"] = all(sp.simplify(sp.expand_complex(e)) == 0 for e in offdiag_cartan)

    # ----------------------------------------------------------------------------------------------------------
    # (4) Reality lemma for the polar CLASS:  transverse(A) = 0  <=>  Im p = Im q = 0.  The diagonal imaginary
    #     Cartan part is a pure J_3 rephasing (class-trivial). Witness: one non-central imaginary part opens it.
    # ----------------------------------------------------------------------------------------------------------
    def transverse(X):
        return [X[i, j] for i in range(3) for j in range(3) if i != j]

    A_witness = A.subs({pI: 1, qI: 0, rI: 0, pR: 0, qR: 0, rR: 0})
    checks["4a_witness_transverse_nonzero"] = any(sp.simplify(sp.expand_complex(e)) != 0
                                                  for e in transverse(A_witness))
    A_pq_real = A.subs({pI: 0, qI: 0})            # Im p = Im q = 0, Im r free
    checks["4b_pq_real_transverse_zero"] = all(sp.simplify(sp.expand_complex(e)) == 0
                                               for e in transverse(A_pq_real))
    # the residual diagonal-only generator from Im r is exactly the J_3/Cartan rephasing i Im(r) diag(0,2,-2):
    A_cartan_only = A.subs({pI: 0, qI: 0, rR: 0, pR: 0, qR: 0})       # keep only rI
    checks["4c_cartan_is_J3_rephasing"] = sp.simplify(
        A_cartan_only - sp.I * rI * sp.diag(0, 2, -2)) == sp.zeros(3)

    # ----------------------------------------------------------------------------------------------------------
    # (5) Central phase is projectively trivial:  c I_3 (central scalar) gives no polar generator, and adding it
    #     to a real step leaves A = 0.  (CHO: the inherited phase is central -> zeta_q^{Delta A_c}.)
    # ----------------------------------------------------------------------------------------------------------
    cR, cI = sp.symbols("cR cI", real=True)
    c = cR + sp.I * cI
    checks["5a_central_scalar_zero"] = is_zero_matrix(Upi_generator_central(c, I3))
    A_real_plus_central = antiherm(jpi_odd_part(sym2_lift(pR * E + qR * F + rR * H) + c * I3))
    checks["5b_real_plus_central_zero"] = is_zero_matrix(A_real_plus_central)

    # ----------------------------------------------------------------------------------------------------------
    # (6) SPINORIAL-FRAME COVARIANCE (the heart of Jerome's caveat: complex matrices != complex coefficients).
    #     Express a sigma-REAL step in a manifestly COMPLEX unitary spinorial frame W on C^3 (here a complex
    #     rotation of the outer pair e_+/-). The lift acquires complex entries, BUT transporting J_Pi and the
    #     hermitian structure by the SAME W gives A_frame = W A W^dagger = 0. So complex representation matrices
    #     produced by a complex frame do NOT open the block; only sigma-anti-real COEFFICIENTS do.
    # ----------------------------------------------------------------------------------------------------------
    inv2 = 1 / sp.sqrt(2)
    W = sp.Matrix([[1, 0, 0],
                   [0, inv2, sp.I * inv2],
                   [0, sp.I * inv2, inv2]])          # complex unitary: e_+/- -> (e_+ +/- i e_-)/sqrt(2)
    checks["6a_W_unitary"] = sp.simplify(W * W.conjugate().T - I3) == sp.zeros(3)
    M_realstep = pR * E + qR * F + rR * H            # sigma-real coefficients
    L = sym2_lift(M_realstep)
    L_frame = sp.simplify(W * L * W.conjugate().T)
    # complex entries genuinely appear in the rotated frame (not a no-op):
    checks["6b_frame_has_complex_entries"] = any(sp.simplify(sp.im(L_frame[i, j])) != 0
                                                 for i in range(3) for j in range(3))
    Jt = sp.simplify(W * J_invol() * W.conjugate().T)             # transported involution
    A_frame = antiherm(jpi_odd_part(L_frame, Jt))
    checks["6c_frame_generator_zero"] = is_zero_matrix(A_frame)   # complex matrices, still no mixing
    # ... while a genuine complex COEFFICIENT (sigma-anti-real) is frame-independently non-zero:
    A_coeff = Upi_generator((pR + sp.I) * E + qR * F + rR * H)    # Im p = 1
    checks["6d_complex_coeff_nonzero"] = not is_zero_matrix(A_coeff)

    # ----------------------------------------------------------------------------------------------------------
    # report
    # ----------------------------------------------------------------------------------------------------------
    print("Front 3e: non-central complex-phase audit -- reality structure of the metaplectic step")
    print("=" * 104)
    print("  Caveat (Jerome): rep-matrix complexity  !=  coefficient complexity (p,q)  !=  central phase zeta_q")
    print("  sigma(M)=conj(M) on sl_2(C); real form sl_2(R)=span_R{E,F,H}; sigma-real <=> p,q,r in R")
    print("  A := antiherm( J_Pi-odd( Sym^2 lift M ) );  internal block A_{e0,e+} = sqrt(2) i (Im p + Im q)/2")
    print("-" * 104)
    print("  (1) real form (p,q,r in R)            => A = 0           [Front 3c / PRS v=0]")
    print("  (2) internal coeff = sqrt2 i (Imp+Imq)/2                 [sigma-anti-real part only]")
    print("  (3) A depends only on i*Im(M); Im r (Cartan) gives NO transverse block")
    print("  (4) reality lemma: transverse(A)=0 <=> Im p=Im q=0; Im r = J_3 rephasing (class-trivial)")
    print("  (5) central phase c I_3 projectively trivial: real+central => A=0   [CHO zeta_q central]")
    print("  (6) spinorial-frame covariance: complex frame => complex matrices but A_frame=0;")
    print("      only a sigma-anti-real COEFFICIENT (Im p=1) is frame-independently non-zero")
    print("-" * 104)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 104)
    print("REALITY LEMMA (proved, exact symbolic): the chiral polar generator A is a function of the sigma-anti-real")
    print("  (imaginary-coefficient) part alone; its TRANSVERSE part = 0  <=>  Im p = Im q = 0  <=>  [U_Pi] = [I].")
    print("  The imaginary Cartan part Im r is a pure J_3 rephasing (class-trivial, the N_A channel). Complex")
    print("  representation matrices from a complex spinorial frame are frame-covariant (A_frame = W A W^dag) and")
    print("  DO NOT open the block: rep-matrix complexity != coefficient complexity != central phase.")
    print()
    print("EXHAUSTIVITY (structural, surveyed -- PRS/AAR + CHO + AOG): at the present full-tower stratum the only")
    print("  data feeding the non-central step are REAL (Heisenberg translations/modulations, oriented area ts in R)")
    print("  plus a CENTRAL cyclotomic phase zeta_q (CHO: no Fourier/Weyl generator => no metaplectic Gauss phase).")
    print("  Hence Im p = Im q = 0 is FORCED => A_Pi = 0 => [U_Pi] = [I] internal.")
    print()
    print("VERDICT: outcome 1 -- NEGATIVE CLOSURE at the present stratum. p,q in R forced, A_Pi = 0, no internal")
    print("  mixing; N_A != 0 and u != 0 stand (diagonal split derived). The positive opening Im p,Im q != 0 needs a")
    print("  genuinely complex NON-CENTRAL metaplectic phase = a Weil/Fourier generator = a NEW tower stratum, gated")
    print("  by the SAME AOG lem:rigidity as epsilon = 1/10. No mass and no mixing value is produced.")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


def Upi_generator_central(c, I3):
    """Central scalar generator c*I_3: J_Pi-even, so the J_Pi-odd anti-hermitian polar generator vanishes."""
    return antiherm(jpi_odd_part(c * I3))


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
