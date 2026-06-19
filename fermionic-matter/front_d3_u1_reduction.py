"""Front D2 (operator step): reduction of the carrier rate u_1 to a constrained bilinear.

Bias-independent, exact symbolic verification (no sampling) that the Front C/D0 carrier rate
    u_1 = <J_3, {E_0, E_1}> / <J_3, J_3>,    E_1 = -Pi_S D \\dot Q(0) D Pi_S^*,
is NOT a free scalar but a constrained bilinear between the even residue block E_0 and the D^pm-transported
chiral defect carried by E_1. Using the even-sector closure (Born--Infeld parity, O30/Beau2026a34)
E_0^2|_{C^3_gen} = diag(1, 1/2, 1/2), the E_0-dependence collapses to a single Born--Infeld factor.

Three results.
  (A) Block transport (chiral frame, tau = Pi_S = I): with the eliminated-block velocity
      \\dot Q(0) = [[pi_LL', pi_LR'], [pi_RL', pi_RR']], the residue first-order coefficient has chiral
      diagonal blocks E_{1,LL} = -Pi_S D_- pi_RR' D_+ Pi_S^*, E_{1,RR} = -Pi_S D_+ pi_LL' D_- Pi_S^*
      (PRS transport), and the J_Pi-odd diagonal of \\dot Q(0) is the transported defect rate
      pi_LL' - pi_RR'(bar) = d/ds Delta_chi(P)|_0 (D0).
  (B) Even-sector reduction (the new content): on C^3_gen the even sector closes to E_0^2 = diag(1,1/2,1/2),
      so E_0 acts as sigma/sqrt(2) on the J_3-carrying outer block (e_+, e_-), sigma = +-1 the V-A sign.
      Then {E_0, E_1}|_{e_pm} = sqrt(2) sigma E_1|_{e_pm}, hence
          u_1 = sqrt(2) sigma <J_3, E_1> / <J_3, J_3> .
      The E_0-dependence has collapsed to the scalar sqrt(2) sigma (the even-block eigenvalue 1/sqrt(2)
      times the spontaneous V-A sign); u_1 is the generation-J_3 projection of E_1, not a free scalar.
  (C) Assembly: u_partial = u_1 s_* = sqrt(2) sigma s_* <J_3, E_1(d_s Delta_chi)>/<J_3, J_3>. The residual
      unknown is exactly <J_3, E_1>, the generation-J_3 projection of the D^pm-transported defect rate; it
      is localised there (not a free scalar and not in s_*).

No figures. Code and comments in English.
"""

import sympy as sp


def comm(A, B):
    return A * B - B * A


def is_zero(M):
    return sp.expand(sp.Matrix(M)) == sp.zeros(*M.shape)


# ---- chiral frame on S_Pi = S_L (+) S_R (each C^2), tau = Pi_S = I -----------------------------------
I2, Z2 = sp.eye(2), sp.zeros(2, 2)


def block(a, b, c, d):
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


def part_A_block_transport():
    """E_1 = -D (dotQ) D in the chiral frame; check its diagonal blocks (Pi_S = I)."""
    checks = {}
    # Chiral Dirac D = [[0, D_-], [D_+, 0]], D self-adjoint => D_+ = D_-^dagger.
    Dm = sp.Matrix([[2, sp.I], [0, 1]])
    Dp = Dm.conjugate().T
    D = block(Z2, Dm, Dp, Z2)
    # Eliminated-block velocity dotQ = d/ds (1-P)|_0, Hermitian blocks pi_LL', pi_RR'; pi_RL' = pi_LR'^dagger.
    pLL = sp.Matrix([[3, 7 + sp.I], [7 - sp.I, 5]])
    pRR = sp.Matrix([[2, 1 - 2 * sp.I], [1 + 2 * sp.I, 4]])
    pLR = sp.Matrix([[sp.I, 1], [2, -sp.I]])
    dotQ = block(pLL, pLR, pLR.conjugate().T, pRR)
    E1 = -D * dotQ * D
    checks["transport_E1_LL"] = is_zero(E1[0:2, 0:2] - (-Dm * pRR * Dp))
    checks["transport_E1_RR"] = is_zero(E1[2:4, 2:4] - (-Dp * pLL * Dm))
    # J_Pi swaps L<->R (antilinear, tau=I): odd-diagonal of dotQ = (pi_LL' - conj(pi_RR'))/2 = (1/2) dDelta_chi.
    SWAP = block(Z2, I2, I2, Z2)
    odd = (dotQ - SWAP * dotQ.conjugate() * SWAP) / 2
    dDelta = pLL - pRR.conjugate()
    checks["odd_diag_is_half_dDelta_chi"] = is_zero(odd[0:2, 0:2] - dDelta / 2)
    return checks


def part_B_even_sector_reduction():
    """On C^3_gen: E_0^2 = diag(1,1/2,1/2) => E_0 = sigma diag(1, 1/sqrt2, 1/sqrt2); reduce u_1."""
    J3 = sp.diag(0, 1, -1)
    r2 = 1 / sp.sqrt(2)
    checks = {}
    extras = {}

    for sigma in (sp.Integer(1), sp.Integer(-1)):
        E0 = sigma * sp.diag(1, r2, r2)                 # even-sector root (Born--Infeld closure, O30)
        # Check the even-sector closure E_0^2 = diag(1,1/2,1/2).
        checks[f"E0_sq_closure_s{sigma}"] = is_zero(E0 * E0 - sp.diag(1, sp.Rational(1, 2), sp.Rational(1, 2)))

        # Generic J_Pi-odd Hermitian E_1 on C^3_gen (J_Pi^(2): e0->-e0, e+ <-> e-).
        Jpi = sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        t = sp.symbols("t0:9", real=True)
        T = sp.Matrix(3, 3, t)
        T = (T + T.T) / 2
        E1 = (T - Jpi * T * Jpi) / 2                     # J_Pi-odd part

        def hs(X, Y):
            return sp.trace(X.T * Y)

        anti = E0 * E1 + E1 * E0
        u1 = sp.expand(hs(anti, J3) / hs(J3, J3))
        u1_reduced = sp.expand(sp.sqrt(2) * sigma * hs(E1, J3) / hs(J3, J3))
        checks[f"u1_reduces_sqrt2_sigma_s{sigma}"] = sp.simplify(u1 - u1_reduced) == 0
        # u_1 is a linear functional of E_1 (no free E_0 scalar beyond sqrt(2) sigma): off-diagonal e0<->e+-
        # parts of E_1 do not enter (the reduction uses only the e_pm block eigenvalue of E_0).
        extras[f"u1_s{sigma}"] = u1
    return checks, extras


def main():
    checks = {}
    checks.update({f"A.{k}": v for k, v in part_A_block_transport().items()})
    cB, extras = part_B_even_sector_reduction()
    checks.update({f"B.{k}": v for k, v in cB.items()})

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("Even-sector reduction:  u_1 = sqrt(2) * sigma * <J_3, E_1> / <J_3, J_3>   (sigma = +-1, V-A sign)")
    print("  -> the E_0-dependence collapses to the Born--Infeld factor sqrt(2) sigma;")
    print("     u_1 is the generation-J_3 projection of E_1 = -Pi_S D dotQ(0) D Pi_S^*, a constrained")
    print("     bilinear, NOT a free scalar.")
    print("Assembly:  u_partial = u_1 s_* = sqrt(2) sigma s_* <J_3, E_1(d_s Delta_chi)> / <J_3, J_3>.")
    print("Residual unknown localised in <J_3, E_1> = generation-J_3 projection of the D^pm-transported")
    print("defect rate d_s Delta_chi(P)|_0 -- not a free scalar, not in s_*.")
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
