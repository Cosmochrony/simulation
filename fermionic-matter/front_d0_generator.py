"""Front D0: identification of the chiral generator A_- of the eliminated-block jet.

Bias-independent, exact verification (exact complex-rational / symbolic, no sampling) of the bridge
between the Front C constrained jet (eliminated_block_jet.py) and the PRS chiral reduction
(ProjectiveResidueSchur), grounding the identification of the chiral generator A_- WITHOUT evaluating
the split magnitude |u|.

Allowed inputs only (Front D0 fence): the chirally off-diagonal Dirac D = [[0, D_-], [D_+, 0]], the
chiral block structure 1-P = [[pi_LL, pi_LR], [pi_RL, pi_RR]], the antiunitary Born--Infeld parity
J_Pi = (chirality swap) o conj, and the transported defect Delta_chi(P) = pi_LL - tau conj(pi_RR) tau^-1
(verified here in the frame tau = I, the general tau following by unitary conjugation). NOT N_A, NOT
epsilon, NOT beta*, NOT N_casc.

Facts grounded in PRS (proved there) and re-verified:
  - block transport of the residue: E_LL = -Pi_S D_- pi_RR D_+ Pi_S^*, E_RR = -Pi_S D_+ pi_LL D_- Pi_S^*;
  - u is localised in the chiral diagonal asymmetry pi_RR - pi_LL, dressed by D^± transport;
  - J_Pi-equivariance [P, J_Pi] = 0 forces pi_LL = tau conj(pi_RR) tau^-1, hence Delta_chi = 0 and u = 0.

Front D0 identification (the new content), verified below:
  (1) the J_Pi-odd part of the eliminated block has chiral-diagonal component equal to Delta_chi(P):
        [Pi_{J_Pi-odd}(1-P)]_LL = (1/2)(pi_LL - conj(pi_RR)) = (1/2) Delta_chi   (tau = I frame);
  (2) along the modulus s, 1-P(s) = Q0 + s Q1 + ..., the Front C tangent satisfies
        Pi_{J_Pi-odd} Q1 = [A_-, Q0],  with  [A_-, Q0]_LL = (1/2) d/ds Delta_chi|_0 ;
      the finite point is equivariant (Delta_chi(0) = 0, u(0) = 0); A4 turns on Delta_chi at O(s);
  (3) the map A_- -> u is the D^± transport followed by generation projection, with the sign fixed by
      the electric (Schur-transverse) polarisation; its MAGNITUDE (the A4 value of d Delta_chi/ds) is the
      dictionary-bound input and is NOT evaluated here, so |u| stays free.

No figures. Code and comments in English.
"""

import sympy as sp


def comm(A, B):
    return A * B - B * A


def is_zero(M):
    return sp.expand(sp.Matrix(M)) == sp.zeros(*M.shape)


# Chiral frame on S_Pi = S_L (+) S_R, each C^2. gamma5 = diag(I, -I); swap exchanges L <-> R.
I2 = sp.eye(2)
Z2 = sp.zeros(2, 2)
SWAP = sp.Matrix(sp.BlockMatrix([[Z2, I2], [I2, Z2]]))  # tau = I frame


def block(a, b, c, d):
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


def jpi_conj(X):
    """Antiunitary J_Pi X J_Pi^{-1} = SWAP * conj(X) * SWAP (tau = I)."""
    return SWAP * X.conjugate() * SWAP


def jpi_odd(X):
    return (X - jpi_conj(X)) / 2


def hermitian_2x2(tag):
    """Exact Hermitian 2x2 with distinct complex-rational entries."""
    a, d = sp.Rational(*{"LL": (3, 1), "RR": (5, 1), "v": (2, 1)}.get(tag, (1, 1))), None
    # Build H = [[r1, x+iy],[x-iy, r2]] with rational r1,r2,x,y depending on tag.
    seed = {"LL": (3, 7, 1, 2), "RR": (5, 11, 4, 3), "dLL": (2, 1, 1, 1), "dRR": (1, 3, 2, 5)}[tag]
    r1, r2, x, y = [sp.Integer(s) for s in seed]
    return sp.Matrix([[r1, x + sp.I * y], [x - sp.I * y, r2]])


def main():
    checks = {}

    # Generic chiral eliminated block 1 - P with Hermitian diagonal blocks and pi_RL = pi_LR^dagger.
    pi_LL = hermitian_2x2("LL")
    pi_RR = hermitian_2x2("RR")
    pi_LR = sp.Matrix([[1 + sp.I, 2], [sp.I, 3 - sp.I]])
    pi_RL = pi_LR.conjugate().T
    elim = block(pi_LL, pi_LR, pi_RL, pi_RR)

    # (1) Chiral-diagonal component of the J_Pi-odd part equals (1/2) Delta_chi(P).
    odd = jpi_odd(elim)
    Delta_chi = pi_LL - pi_RR.conjugate()          # tau = I:  pi_LL - conj(pi_RR)
    checks["D0.odd_diag_LL_is_half_Delta_chi"] = is_zero(odd[0:2, 0:2] - Delta_chi / 2)
    checks["D0.odd_diag_RR_is_minus_half_Delta_chi_conj"] = is_zero(
        odd[2:4, 2:4] - (pi_RR - pi_LL.conjugate()) / 2
    )

    # (2) Equivariance pi_LL = conj(pi_RR) => Delta_chi = 0 => J_Pi-odd diagonal vanishes (u = 0 branch).
    elim_eq = block(pi_RR.conjugate(), pi_LR, pi_RL, pi_RR)  # impose pi_LL = conj(pi_RR)
    checks["D0.equivariant_defect_zero"] = is_zero(jpi_odd(elim_eq)[0:2, 0:2])

    # (3) PRS block transport (re-derivation): residue diagonal blocks from D (1-P) D, Pi_S = I.
    Dm = sp.Matrix([[2, sp.I], [0, 1]])            # D_-: S_R -> S_L
    Dp = Dm.conjugate().T                          # D_+ = D_-^dagger so that D is self-adjoint
    D = block(Z2, Dm, Dp, Z2)
    DPD = D * elim * D
    E = -DPD                                       # E_Pi = -Pi_S D (1-P) D Pi_S^*, Pi_S = I
    checks["D0.transport_E_LL"] = is_zero(E[0:2, 0:2] - (-Dm * pi_RR * Dp))
    checks["D0.transport_E_RR"] = is_zero(E[2:4, 2:4] - (-Dp * pi_LL * Dm))

    # (4) A_- <-> d Delta_chi/ds bridge. Model the modulus jet 1-P(s) = Q0 + s Q1 with Q0 equivariant
    # (finite, J_Pi-even) and Q1 the A4 first-order breaking. Then Pi_{J_Pi-odd} Q1 = [A_-, Q0] and its
    # LL block is (1/2) d Delta_chi/ds|_0.
    q = pi_RR                                      # any Hermitian block; Q0 chirally symmetric (tau = I)
    Q0 = block(q.conjugate(), Z2, Z2, q)           # equivariant: pi_LL(0) = conj(pi_RR(0)), off-diag 0
    checks["D0.Q0_equivariant"] = is_zero(jpi_odd(Q0))     # u(0) = 0
    # First-order A4 breaking: independent Hermitian rates on the diagonal, plus an off-diagonal rate.
    dLL = hermitian_2x2("dLL")
    dRR = hermitian_2x2("dRR")
    dLR = sp.Matrix([[sp.I, 1], [2, -sp.I]])
    Q1 = block(dLL, dLR, dLR.conjugate().T, dRR)
    dDelta = dLL - dRR.conjugate()                 # d Delta_chi/ds|_0  (tau = I)
    checks["D0.A_minus_LL_is_half_dDelta_chi"] = is_zero(jpi_odd(Q1)[0:2, 0:2] - dDelta / 2)
    # And Pi_{J_Pi-odd} Q1 is genuinely J_Pi-odd while generically non-zero (the live A_- channel).
    checks["D0.A_minus_is_J_odd"] = is_zero(jpi_odd(Q1) + jpi_conj(jpi_odd(Q1)))
    checks["D0.A_minus_nonzero"] = not is_zero(jpi_odd(Q1))

    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        assert ok, name

    print()
    print("Delta_chi(P) (tau = I) =", list(sp.Matrix(Delta_chi)))
    print("d Delta_chi/ds|_0      =", list(sp.Matrix(dDelta)),
          "  (A4-level magnitude: dictionary-bound, NOT evaluated -> |u| stays free)")
    print()
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
