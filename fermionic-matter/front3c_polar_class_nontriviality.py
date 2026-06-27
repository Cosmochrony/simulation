"""Front 3c step 3: polar-class non-triviality audit of the complex metaplectic phase.

Bias-independent, exact symbolic verification (no sampling). The canonical-vs-class audit
(front3c_polar_class_audit.py) established that the chiral polar factor U_Pi of the projected Yukawa
Y_Pi = U_Pi H_Pi^{1/2} is not a canonical observable but a rephasing CLASS

    [U_Pi]  in  U(3) / (U(1)^3_R x U(1)^3_L),     U_Pi ~ V_R U_Pi V_L^{-1}  (V_L,V_R diagonal).

The real observable is therefore not the metaplectic phase gamma but the class [U_Pi]. This audit decides the
genuine next lock (Jerome's reframing):

    Does the complex metaplectic phase generate a NON-TRIVIAL polar class  [U_Pi(gamma)] != [I],
    or only a representative diagonal-rephasing-equivalent to the identity?

Geometric criterion (proved below). The diagonal-rephasing orbit through I has tangent space exactly the
DIAGONAL anti-hermitian matrices (i d, d real). A unitary generator A := dU_Pi/dgamma|_0 is anti-hermitian;
its class moves off [I] at first order iff A has a component TRANSVERSE to that orbit tangent, i.e. iff its
OFF-DIAGONAL part is non-zero. The rephasing-invariant detectors are:
  * the first variation of the moduli  d|(U_Pi)_{ij}|  (i != j), which is |A_{ij}|;
  * the Jarlskog-type invariant  J_Pi = Im( U_11 U_22 conj(U_12) conj(U_21) ), whose leading non-vanishing
    order in gamma is gamma^3 and is proportional to Im(A_12 A_23 A_31) -- the genuine 3-generation CP datum.

Two precautions (Jerome, firm).
  (i)  AOG/CHO closed the ORIENTATION caveat [H-orient] and the spin-Galois sqrt(5) factor; they do NOT close
       the non-triviality of [U_Pi]. The chiral lift carries no action on Q(sqrt 5), so no spin-Galois factor
       orthogonal to zeta_q survives -- but that settles orientation, not the polar class.
  (ii) u != 0 (the diagonal generation split, A4/PRS) is NOT mixing. u lives in the DIAGONAL channel A_diag
       (pure level split / rephasing) and contributes nothing to the off-diagonal moduli or to J_Pi. Mixing
       requires a NON-DIAGONAL polar class, sourced by the complex metaplectic channel v (EBJ: v != 0 needs the
       complex metaplectic phase; A4-note: on the derived real cascade the transverse channel vanishes at
       gamma = 0, the complex interior remaining conditional on a non-derived phase and an unprescribed order).

Results (all exact symbolic).
  (A) Orbit tangent at I is diagonal anti-hermitian: d/ds [V_R(s) V_L(s)^{-1}]|_0 = i(D_R - D_L), diagonal,
      anti-hermitian; off-diagonal part identically zero; and every diagonal anti-hermitian is reached.
  (B) Transversality detector: for U(gamma) = exp(gamma A), A anti-hermitian, the second-order coefficient of
      |(U)_{ij}|^2 (i != j) is |A_{ij}|^2; so d|U_{ij}| != 0 iff A_{ij} != 0, i.e. iff A is transverse.
  (C) Jarlskog order: J_Pi vanishes at orders gamma^0, gamma^1, gamma^2; its leading gamma^3 coefficient is
      proportional to Im(A_12 A_23 A_31), independent of the diagonal phases d_k (rephasing-invariant).
  (D) Real-cascade collapse: with v = 0 the generator is purely diagonal (A_off = 0), so U_Pi(gamma) stays
      diagonal => [U_Pi] = [I]: |(U_Pi)_{ij}| = delta_{ij} and J_Pi = 0 to all orders. NO physical mixing at
      this stratum.
  (E) Complex channel non-triviality: a generator with A_off != 0 gives off-diagonal moduli != 0 ([U_Pi] != [I],
      CP-conserving mixing if A_off real); a genuinely complex A_off with Im(A_12 A_23 A_31) != 0 gives
      J_Pi != 0 at order gamma^3 (a genuine polar mixing class with CP violation).
  (F) u-channel orthogonality: the diagonal split channel A_diag (carrier of u) contributes zero to every
      off-diagonal modulus first variation and zero to J_Pi -- u != 0 does not produce mixing.

Conclusion (printed). The non-triviality of the polar class is controlled EXACTLY by the off-diagonal
(transverse) part of the metaplectic generator, i.e. by the complex metaplectic channel v. On the derived real
cascade v = 0 => [U_Pi] = [I] => no physical mixing at this stratum (a strong conditional result). A non-trivial
[U_Pi] != [I], hence CKM/PMNS-type mixing, requires a genuine complex metaplectic phase -- not derived in the
present construction, conditional on the same open data (an unprescribed complex phase + ordering, A4-note), and
observable only relative to a second fermionic sector sharing one generation carrier. No mass and no mixing value
is produced. No figures. English.
"""

import sympy as sp


def anti_hermitian(d, off):
    """Build a 3x3 anti-hermitian matrix: diagonal i*d_k (d real); off[(i,j)] = A_{ij}, A_{ji} = -conj(A_{ij})."""
    A = sp.zeros(3, 3)
    for k in range(3):
        A[k, k] = sp.I * d[k]
    for (i, j), val in off.items():
        A[i, j] = val
        A[j, i] = -sp.conjugate(val)
    return A


def jarl(M):
    return sp.im(M[0, 0] * M[1, 1] * sp.conjugate(M[0, 1]) * sp.conjugate(M[1, 0]))


def expm_series(A, g, order):
    """Truncated matrix exponential exp(g A) up to g^order (exact, symbolic)."""
    U = sp.zeros(3, 3)
    term = sp.eye(3)
    for k in range(order + 1):
        U += term * (g**k) / sp.factorial(k)
        term = sp.simplify(term * A)
    return U


def series_coeff(expr, g, n):
    return sp.expand_complex(sp.expand(expr)).coeff(g, n)


def main():
    checks = {}
    g = sp.symbols("gamma", real=True)

    # real diagonal phases and generic complex off-diagonal couplings (anti-hermitian generator)
    d = sp.symbols("d0:3", real=True)
    a12, b12, a13, b13, a23, b23 = sp.symbols("a12 b12 a13 b13 a23 b23", real=True)
    off_full = {(0, 1): a12 + sp.I * b12, (0, 2): a13 + sp.I * b13, (1, 2): a23 + sp.I * b23}
    A = anti_hermitian(list(d), off_full)

    # ---- (A) orbit tangent at I = diagonal anti-hermitian ---------------------------------------
    s = sp.symbols("s", real=True)
    dl = sp.symbols("dl0:3", real=True)
    dr = sp.symbols("dr0:3", real=True)
    VL = sp.diag(*[sp.exp(sp.I * s * dl[k]) for k in range(3)])
    VR = sp.diag(*[sp.exp(sp.I * s * dr[k]) for k in range(3)])
    curve = VR * VL.inv()
    tangent = sp.diff(curve, s).subs(s, 0)
    checks["A_tangent_offdiag_zero"] = all(sp.simplify(tangent[i, j]) == 0
                                           for i in range(3) for j in range(3) if i != j)
    checks["A_tangent_diag_antiherm"] = all(
        sp.simplify(tangent[k, k] - sp.I * (dr[k] - dl[k])) == 0 for k in range(3))
    # every diagonal anti-hermitian is reached (set dl = 0): i*dr arbitrary
    checks["A_tangent_surjective_diag"] = all(
        sp.simplify(tangent.subs({dl[k]: 0 for k in range(3)})[k, k] - sp.I * dr[k]) == 0
        for k in range(3))

    # ---- (B) modulus first variation detects the transverse (off-diagonal) part -----------------
    U2 = expm_series(A, g, 2)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        mod2 = sp.expand_complex(sp.expand(U2[i, j] * sp.conjugate(U2[i, j])))
        coeff_g2 = mod2.coeff(g, 2)
        target = sp.Abs(off_full[(i, j)])**2          # |A_{ij}|^2
        checks[f"B_modvar_{i}{j}"] = sp.simplify(coeff_g2 - target) == 0

    # ---- (C) Jarlskog order: 0 up to g^2, leading g^3 ~ Im(A_12 A_23 A_31) -----------------------
    U3 = expm_series(A, g, 3)
    J = sp.expand_complex(sp.expand(jarl(U3)))
    checks["C_J_order0"] = series_coeff(J, g, 0) == 0
    checks["C_J_order1"] = series_coeff(J, g, 1) == 0
    checks["C_J_order2"] = series_coeff(J, g, 2) == 0
    J3 = sp.simplify(series_coeff(J, g, 3))
    # A_31 = -conj(A_13); the genuine 3-gen CP datum is Im(A_12 A_23 A_31)
    A31 = -sp.conjugate(off_full[(0, 2)])
    cp_datum = sp.im(off_full[(0, 1)] * off_full[(1, 2)] * A31)
    ratio = sp.simplify(J3 / cp_datum)
    checks["C_J3_prop_cpdatum"] = ratio != 0 and sp.simplify(sp.diff(ratio, a12)) == 0 \
        and sp.simplify(sp.diff(ratio, b23)) == 0
    checks["C_J3_indep_diag"] = all(sp.simplify(sp.diff(J3, d[k])) == 0 for k in range(3))

    # ---- (D) real-cascade collapse: v = 0 => A_off = 0 => diagonal => [U_Pi] = [I] --------------
    # diagonal channel: the EXACT exponential of a diagonal anti-hermitian is diag(exp(i g d_k)), |.| = 1
    U_diag = sp.diag(*[sp.exp(sp.I * g * d[k]) for k in range(3)])
    checks["D_diag_offdiag_zero"] = all(sp.simplify(U_diag[i, j]) == 0
                                        for i in range(3) for j in range(3) if i != j)
    checks["D_diag_moduli_identity"] = all(
        sp.simplify(sp.Abs(U_diag[i, j]) - (1 if i == j else 0)) == 0
        for i in range(3) for j in range(3))
    checks["D_diag_J_zero"] = sp.simplify(sp.expand_complex(jarl(U_diag))) == 0

    # ---- (E) complex-channel non-triviality vs CP-conserving real channel -----------------------
    # real off-diagonal channel (A_off real): [U_Pi] != [I] but J = 0 (CP conserving)
    A_real_off = anti_hermitian([0, 0, 0], {(0, 1): a12, (0, 2): a13, (1, 2): a23})
    U_re = expm_series(A_real_off, g, 2)
    mod_re = sp.expand_complex(sp.expand(U_re[0, 1] * sp.conjugate(U_re[0, 1])))
    checks["E_real_off_mixing"] = sp.simplify(mod_re.coeff(g, 2) - a12**2) == 0
    U_re3 = expm_series(A_real_off, g, 3)
    checks["E_real_off_J_zero"] = sp.simplify(series_coeff(jarl(U_re3), g, 3)) == 0
    # genuinely complex channel with Im(A_12 A_23 A_31) != 0 => J != 0 at g^3
    sub_cp = {a12: 1, b12: 0, a23: 1, b23: 0, a13: 0, b13: 1, d[0]: 0, d[1]: 0, d[2]: 0}
    cp_val = sp.simplify(cp_datum.subs(sub_cp))
    checks["E_complex_off_J_nonzero"] = cp_val != 0 and sp.simplify(J3.subs(sub_cp)) != 0

    # ---- (F) u-channel orthogonality: diagonal split (carrier of u) gives no mixing -------------
    # u enters E_Pi^2|gen = diag(1,1/2+u,1/2-u) -> a DIAGONAL level operator; its generator is in A_diag.
    checks["F_u_no_offdiag"] = all(sp.simplify(U_diag[i, j]) == 0
                                   for i in range(3) for j in range(3) if i != j)
    checks["F_u_no_J"] = sp.simplify(sp.expand_complex(jarl(U_diag))) == 0

    # ---------------------------------------------------------------------------------------------
    print("Front 3c step 3 - polar-class non-triviality of the complex metaplectic phase (exact symbolic)")
    print("=" * 100)
    print("  Question: does the metaplectic phase give [U_Pi(gamma)] != [I] in U(3)/(U(1)^3_R x U(1)^3_L)?")
    print("  Criterion: [U_Pi] moves off [I] at first order  <=>  the generator A = dU_Pi/dgamma|_0 has a")
    print("             non-zero OFF-DIAGONAL (transverse) part (the orbit tangent at I is diagonal anti-herm).")
    print("  (A) orbit tangent at I = diagonal anti-hermitian (off-diagonal part identically zero)")
    print("  (B) modulus first variation: coeff of g^2 in |U_{ij}|^2 is |A_{ij}|^2  => detects A_off")
    print("  (C) Jarlskog: 0 up to g^2; leading g^3 term ~ Im(A_12 A_23 A_31), independent of diagonal phases")
    print("  (D) real cascade v=0 => A_off=0 => U_Pi diagonal => [U_Pi]=[I]: |U_{ij}|=delta_{ij}, J=0 (no mixing)")
    print("  (E) complex channel A_off != 0 => [U_Pi] != [I]; Im(A_12 A_23 A_31) != 0 => J_Pi != 0 (CP mixing)")
    print("  (F) u-channel is DIAGONAL (A_diag): zero off-diagonal moduli, zero J  => u != 0 is NOT mixing")
    print("-" * 100)
    allok = True
    for k, val in checks.items():
        ok = bool(val)
        allok = allok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {k}")
    print("=" * 100)
    print("RESULT: polar-class non-triviality is controlled EXACTLY by the off-diagonal (transverse) part of the")
    print("        metaplectic generator = the complex metaplectic channel v. On the derived real cascade v=0 =>")
    print("        [U_Pi]=[I] => NO physical mixing at this stratum (strong conditional). A non-trivial")
    print("        [U_Pi] != [I] / CKM-PMNS-type mixing requires a genuine complex metaplectic phase -- NOT")
    print("        derived in the present construction (conditional on an unprescribed phase + ordering), and")
    print("        observable only relative to a second fermionic sector. AOG/CHO closed [H-orient], NOT this.")
    print("        No mass and no mixing value is produced.")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    return allok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
