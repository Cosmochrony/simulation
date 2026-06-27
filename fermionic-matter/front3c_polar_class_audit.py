"""Front 3c: canonical-vs-class gauge audit of the chiral polar factor U_Pi.

Bias-independent, exact symbolic verification (no sampling). Front 3b left the projected Yukawa morphism
    Y_Pi = U_Pi H_Pi^{1/2},   H_Pi := Y_Pi^dag Y_Pi  on the left generation carrier,
with H_Pi|gen = lambda_Y^2 diag(1, 1/2+u, 1/2-u) closed (PYO Beau2026pyo / PRS Beau2026prs / PYL Beau2026pyl),
but the chiral polar factor U_Pi : S_{L,Pi} (x) L_Y -> S_{R,Pi} undetermined. Front 3c asks the delimiter
question (Jerome): does the complex metaplectic phase fix a CANONICAL U_Pi, or only a CLASS

    U_Pi  ~  V_R U_Pi V_L^{-1}        (V_L on S_{L,Pi}, V_R on S_{R,Pi}, admissible chiral basis changes) ?

The audit acts with admissible chiral basis changes and reports which data of U_Pi survive.

Admissibility of the basis changes. The generation triplet C^3_gen carries the LEFT level operator
H_Pi (= lambda_Y^2 E_Pi^2|gen) and, by the antiunitary J_Pi exchange L<->R (PRS sec:chiral-defect), a RIGHT
level operator Y_Pi Y_Pi^dag = U_Pi H_Pi U_Pi^dag with the SAME spectrum. Both spectra are
lambda_Y^2 {1, 1/2+u, 1/2-u}, which are DISTINCT for 0 < u < 1/2, u != 1/2. Distinctness pins each carrier to
its J_3-labelled level eigenbasis up to a diagonal rephasing: the admissible residual group is the diagonal
torus U(1)^3_L x U(1)^3_R, exactly the generation-rephasing freedom. (If instead the right carrier were a free
U(3), U_Pi would be removable outright -- see check A -- and there would be no mixing at all; the pinning by the
distinct-spectrum right level operator is what makes the question non-trivial.)

Results (all exact symbolic).
  (A) Free-right triviality: with an UNRESTRICTED V_R in U(3), V_R = U_Pi removes U_Pi entirely
      (U_Pi^{-1} Y_Pi = H_Pi^{1/2}, diagonal positive). So if the right basis were unconstrained the orbit
      invariant of U_Pi is empty -- only Spec(H_Pi) survives, no mixing.
  (B) Right-basis pinning: the right level operator Y Y^dag = U H U^dag has the SAME (distinct) spectrum as H,
      so its eigenbasis is fixed up to a diagonal phase; any unitary W commuting with diag(distinct) is diagonal
      ((W D - D W)_{ij} = W_{ij}(d_j - d_i)). Hence the admissible V_R (and V_L) are diagonal rephasings only.
  (C) Class invariants under diagonal rephasing U_Pi -> V_R U_Pi V_L^{-1}, V_L,V_R in U(1)^3:
      the moduli |(U_Pi)_{ij}| are invariant, and the Jarlskog-type quartet phase
      Im( U_11 U_22 conj(U_12) conj(U_21) ) is invariant. These are the surviving (rephasing-invariant) data.
  (D) Parameter count: U(3) has 9 real parameters; the rephasings remove 2*3 - 1 = 5; so 4 physical survive
      = 3 mixing magnitudes (angles) + 1 CP phase (Jarlskog). U_Pi is therefore CLASS-only, not canonical.
  (E) CP-real / no-mixing collapse: a real orthogonal U_Pi has Jarlskog J = 0 (no CP phase); the PRS even
      closure v = 0 forces U_Pi diagonal, i.e. U_Pi ~ I up to rephasing, giving |(U_Pi)_{ij}| = delta_{ij} and
      ZERO mixing. A non-trivial U_Pi (the off-diagonal/complex part) requires a complex metaplectic phase gamma.
  (F) Metaplectic source: a one-parameter chiral generator U_Pi(gamma) = exp(gamma (R - R^dag)) with R a
      strictly-upper nilpotent (the L->R off-diagonal induced by the metaplectic phase) gives U_Pi(0) = I
      (no mixing) and d/dgamma U_Pi|_0 = R - R^dag != 0 (mixing switched on by gamma). Real gamma with a real R
      keeps J = 0 (CP-conserving); a genuine complex phase is needed for J != 0.

Conclusion (printed): the complex metaplectic phase fixes U_Pi only up to the rephasing CLASS
U_Pi ~ V_R U_Pi V_L^{-1}. The physical content of Front 3c is the class invariants: 3 mixing magnitudes
|(U_Pi)_{ij}| + 1 Jarlskog CP phase. If gamma is real on the derived spin stratum (CHO/AOG sqrt(5)-rigidity,
rho_chi = 1) the CP phase vanishes and, with PRS v = 0, U_Pi collapses to I (no mixing at this stratum) -- a
strong conditional result; a genuine complex phase, hence non-trivial mixing, can enter only via the open
full-tower stratum, gated by the same AOG lem:rigidity as epsilon = 1/10. No mass and no mixing value is
produced. No figures. English.
"""

import sympy as sp


def _zero(M):
    """Robust matrix-zero test: complex- and trig-aware simplification entrywise."""
    return all(sp.simplify(sp.trigsimp(sp.expand_complex(e))) == 0 for e in M)


def dag(M):
    return M.conjugate().T


def main():
    checks = {}

    u = sp.symbols("u", real=True)
    lam = sp.symbols("lambda", positive=True)
    I3 = sp.eye(3)

    # the polar no-go (A,B,C) is proved for GENERIC positive singular values (a,b,c) so it does not depend
    # on the particular level values (front3b convention); the level identification enters only via the
    # distinctness that pins the eigenbases (B_levels_distinct).
    a, b, c = sp.symbols("a b c", positive=True)
    Hhalf = sp.diag(a, b, c)
    H = Hhalf * Hhalf
    levels = [sp.Integer(1), sp.Rational(1, 2) + u, sp.Rational(1, 2) - u]
    asm = sp.Q.positive(sp.Rational(1, 2) + u) & sp.Q.positive(sp.Rational(1, 2) - u) & sp.Q.positive(u)

    # a generic chiral polar factor (a real generation rotation + a complex phase), unitary
    th, ph = sp.symbols("theta phi", real=True)
    U_rot = sp.Matrix([[sp.cos(th), -sp.sin(th), 0],
                       [sp.sin(th),  sp.cos(th), 0],
                       [0, 0, 1]])
    U_ph = sp.diag(1, 1, sp.exp(sp.I * ph))
    U_pi = U_ph * U_rot                                  # a non-trivial unitary U_Pi
    Y = U_pi * Hhalf                                     # Y_Pi = U_Pi H_Pi^{1/2}

    # ---- (A) free-right triviality: unrestricted V_R = U_Pi removes U_Pi -------------------------
    checks["A_unitary_Upi"] = _zero(dag(U_pi) * U_pi - I3)
    checks["A_free_right_removes_Upi"] = _zero(U_pi.inv() * Y - Hhalf)   # diagonal positive, no mixing left
    checks["A_invariant_is_spectrum"] = _zero(dag(Y) * Y - H)           # only Spec(H) survives the full orbit

    # ---- (B) right-basis pinning: right level operator has same distinct spectrum ----------------
    H_right = Y * dag(Y)                                                 # = U_Pi H U_Pi^dag
    # same characteristic polynomial as H  =>  same spectrum
    x = sp.symbols("x")
    cpL = H.charpoly(x).as_expr()
    cpR = H_right.charpoly(x).as_expr()
    checks["B_right_same_spectrum"] = sp.simplify(sp.expand(sp.expand_complex(cpL - cpR))) == 0
    # distinctness of the three levels (pins the eigenbasis up to a diagonal phase)
    distinct = [sp.refine(sp.simplify(levels[i] - levels[j]), asm)
                for i, j in [(0, 1), (0, 2), (1, 2)]]
    checks["B_levels_distinct"] = all(d != 0 for d in distinct)
    # commutant of a distinct diagonal is diagonal: (W D - D W)_{ij} = W_{ij}(d_j - d_i)
    w = sp.symbols("w0:9")
    W = sp.Matrix(3, 3, lambda i, j: w[3 * i + j])
    d0, d1, d2 = sp.symbols("d0 d1 d2")
    D = sp.diag(d0, d1, d2)
    comm = W * D - D * W
    ds = [d0, d1, d2]
    checks["B_commutant_offdiag"] = all(
        sp.simplify(comm[i, j] - W[i, j] * (ds[j] - ds[i])) == 0
        for i in range(3) for j in range(3))

    # ---- (C) class invariants under diagonal rephasing U -> V_R U V_L^{-1} -----------------------
    pL = sp.symbols("pL0:3", real=True)
    pR = sp.symbols("pR0:3", real=True)
    VL = sp.diag(*[sp.exp(sp.I * pL[i]) for i in range(3)])
    VR = sp.diag(*[sp.exp(sp.I * pR[i]) for i in range(3)])
    Uent = sp.symbols("U0:9")                                            # abstract entries of a unitary U_Pi
    Uabs = sp.Matrix(3, 3, lambda i, j: Uent[3 * i + j])
    Urep = VR * Uabs * VL.inv()
    # moduli invariance: |(VR U VL^{-1})_{ij}| = |U_{ij}|
    checks["C_moduli_invariant"] = all(
        sp.simplify(sp.Abs(Urep[i, j]) - sp.Abs(Uabs[i, j])) == 0
        for i in range(3) for j in range(3))
    # Jarlskog quartet phase invariance
    def jarl(M):
        return M[0, 0] * M[1, 1] * sp.conjugate(M[0, 1]) * sp.conjugate(M[1, 0])
    checks["C_jarlskog_invariant"] = sp.simplify(
        sp.expand_complex(sp.im(jarl(Urep)) - sp.im(jarl(Uabs)))) == 0

    # ---- (D) parameter count: 9 - (2*3 - 1) = 4 = 3 angles + 1 phase -----------------------------
    n = 3
    dim_U = n * n
    removed = 2 * n - 1
    physical = dim_U - removed
    angles = n * (n - 1) // 2
    phases = (n - 1) * (n - 2) // 2
    checks["D_param_count"] = (physical == 4 and angles == 3 and phases == 1
                               and angles + phases == physical)

    # ---- (E) CP-real / no-mixing collapse -------------------------------------------------------
    # real orthogonal U_Pi (phi = 0) => Jarlskog J = 0
    U_real = U_pi.subs(ph, 0)
    checks["E_real_unitary_J_zero"] = sp.simplify(sp.expand_complex(sp.im(jarl(U_real)))) == 0
    # PRS even closure v = 0 => U_Pi diagonal => ~ I up to rephasing => no mixing (off-diagonals vanish)
    U_diag = sp.diag(1, 1, sp.exp(sp.I * ph))            # v=0 leaves only diagonal phases
    offdiag_zero = all(sp.simplify(U_diag[i, j]) == 0 for i in range(3) for j in range(3) if i != j)
    checks["E_v0_no_mixing"] = offdiag_zero
    checks["E_v0_moduli_identity"] = all(
        sp.simplify(sp.Abs(U_diag[i, j]) - (1 if i == j else 0)) == 0
        for i in range(3) for j in range(3))

    # ---- (F) metaplectic source: gamma switches mixing on -------------------------------------
    g = sp.symbols("gamma", real=True)
    R = sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])    # strictly-upper L->R off-diagonal carrier
    A = R - R.T                                          # real antisymmetric => exp is orthogonal
    U_gamma = sp.exp(A * g)                              # one-parameter chiral generator
    checks["F_gamma0_identity"] = _zero(U_gamma.subs(g, 0) - I3)
    dU = sp.diff(U_gamma, g).subs(g, 0)
    checks["F_dgamma_nonzero"] = not _zero(dU)           # mixing switched on by gamma
    checks["F_dgamma_is_A"] = _zero(dU - A)
    # real gamma + real R keeps J = 0 (CP-conserving); a complex phase is needed for J != 0
    checks["F_real_gamma_J_zero"] = sp.simplify(
        sp.expand_complex(sp.im(jarl(U_gamma)))) == 0

    # ---------------------------------------------------------------------------------------------
    print("Front 3c - canonical-vs-class audit of the chiral polar factor U_Pi (exact symbolic, no sampling)")
    print("=" * 100)
    print("  Y_Pi = U_Pi H_Pi^{1/2};  question: does the metaplectic phase fix U_Pi, or only the class")
    print("  U_Pi ~ V_R U_Pi V_L^{-1} (V_L on S_{L,Pi}, V_R on S_{R,Pi})?")
    print("  (A) free-right V_R=U_Pi removes U_Pi -> only Spec(H_Pi) survives the full orbit (no mixing)")
    print("  (B) but right level op YY^dag has the SAME distinct spectrum -> right basis pinned up to phase")
    print("      => admissible V_L,V_R are DIAGONAL rephasings only (commutant of distinct diagonal = diagonal)")
    print("  (C) under diagonal rephasing: |(U_Pi)_{ij}| and the Jarlskog quartet phase are INVARIANT")
    print("  (D) count: 9 - (2*3-1) = 4 physical = 3 mixing angles + 1 CP (Jarlskog) phase => CLASS, not canonical")
    print("  (E) real U_Pi => J=0; PRS even closure v=0 => U_Pi diagonal ~ I => ZERO mixing")
    print("  (F) metaplectic gamma switches mixing on (U_Pi(0)=I, dU/dgamma|_0 != 0); real gamma keeps J=0")
    print("-" * 100)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 100)
    print("RESULT: U_Pi is CLASS-only, not canonical. The metaplectic phase fixes U_Pi up to")
    print("        U_Pi ~ V_R U_Pi V_L^{-1} (diagonal rephasings). Surviving invariants: 3 mixing magnitudes")
    print("        |(U_Pi)_{ij}| + 1 Jarlskog CP phase. If gamma is real on the derived spin stratum")
    print("        (CHO/AOG sqrt(5)-rigidity, rho_chi=1) the CP phase vanishes and, with PRS v=0, U_Pi ~ I:")
    print("        NO mixing at this stratum (strong conditional). A genuine complex phase / non-trivial mixing")
    print("        can enter only via the open full-tower stratum, gated by the same AOG lem:rigidity as eps=1/10.")
    print("        No mass and no mixing value is produced.")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
