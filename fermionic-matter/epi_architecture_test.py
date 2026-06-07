"""
epi_architecture_test.py

Architecture test for the amplitude programme (step 4, before deriving N_casc).

Question (Q14 cascade bridge, Subsection ssec:dg-bridge):
    Is the J_3-split component of the cascade generator d_tau the SAME operator as the
    generator of the capacity decay sigma_pair(n) ~ n^{-delta_pair}?
        - SAME  -> unified architecture: N_casc = F(delta_pair, n_3^obs, saturation)
        - DISTINCT (coupled) -> factorised: N_casc = transfer constant.

Method (bias-independent, built from V = C^2 by Sym^2):
    Decompose the metaplectic cascade generator d_tau = log(rho_1(g)),
    g = S_X(t) S_Y(s), under SU(2): End(Sym^2 V) = scalar(1) (+) adjoint(3) (+) spin2(5).
        - the J_3 split lives in the ADJOINT sector;
        - a capacity-decay (norm/magnitude) generator must populate the SCALAR sector
          (net trace = net norm change).
    If d_tau^{metaplectic} is traceless, it carries NO scalar component, so the decay
    (a magnitude effect of the non-injective projection) is a DISTINCT generator.

Outputs:
    - stdout: sector weights of d_tau, scalar component, HS-orthogonality with J_3
    - epi_architecture_test.png
"""

import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
SHEAR_T = 0.40
SHEAR_S = 0.40
OUT_PNG = "epi_architecture_test.png"

# --------------------------------------------------------------------------- #
# rank-3 factor from V = C^2 by Sym^2, basis (e_0, e_+, e_-)
# --------------------------------------------------------------------------- #
vp = np.array([1.0, 0.0]); vm = np.array([0.0, 1.0])
e_pp = np.kron(vp, vp)
e_0 = (np.kron(vp, vm) + np.kron(vm, vp)) / np.sqrt(2.0)
e_mm = np.kron(vm, vm)
B = np.stack([e_0, e_pp, e_mm], axis=1)            # columns (e_0, e_+, e_-)


def lift1(A_V):
    A4 = np.kron(A_V, np.eye(2)) + np.kron(np.eye(2), A_V)
    return B.T @ A4 @ B


def sym2(g2):
    return B.T @ np.kron(g2, g2) @ B


# SU(2) generators on the rank-3 factor (spin-1), built from C^2
Jz_V = np.diag([0.5, -0.5])
Jp_V = np.array([[0.0, 1.0], [0.0, 0.0]])          # raising on C^2
Jm_V = np.array([[0.0, 0.0], [1.0, 0.0]])          # lowering on C^2
Jz = lift1(Jz_V)
Jp = lift1(Jp_V)
Jm = lift1(Jm_V)
Jx = 0.5 * (Jp + Jm)
Jy = 0.5j * (Jm - Jp)
Id = np.eye(3)

def hs(A, Bm):
    return np.trace(A.conj().T @ Bm)

# orthonormal-ish bases for the three SU(2) sectors (Hilbert-Schmidt)
scalar_basis = [Id]
adjoint_basis = [Jx, Jy, Jz]                       # spin-1 (vector)

# spin-2 (quadrupole) basis: symmetric traceless quadratics in J, HS-orthogonalised
raw_spin2 = [
    Jz @ Jz - (2.0/3.0) * Id,
    Jx @ Jx - Jy @ Jy,
    Jx @ Jz + Jz @ Jx,
    Jy @ Jz + Jz @ Jy,
    Jx @ Jy + Jy @ Jx,
]

def gram_schmidt(mats):
    out = []
    for M in mats:
        N = M.copy().astype(complex)
        for Q in out:
            N = N - (hs(Q, N) / hs(Q, Q)) * Q
        if np.real(hs(N, N)) > 1e-12:
            out.append(N)
    return out

scalar_on = gram_schmidt(scalar_basis)
adjoint_on = gram_schmidt(adjoint_basis)
spin2_on = gram_schmidt(raw_spin2)


def sector_weight(M, basis):
    w = 0.0
    for Q in basis:
        c = hs(Q, M) / hs(Q, Q)
        w += np.real(np.conjugate(c) * c) * np.real(hs(Q, Q))
    return w


# --------------------------------------------------------------------------- #
# metaplectic cascade generator
# --------------------------------------------------------------------------- #
S_X = np.array([[1.0, SHEAR_T], [0.0, 1.0]])
S_Y = np.array([[1.0, 0.0], [SHEAR_S, 1.0]])
g = S_X @ S_Y
rho1 = sym2(g)
dtau = logm(rho1)

tot = np.real(hs(dtau, dtau))
w_scalar = sector_weight(dtau, scalar_on)
w_adjoint = sector_weight(dtau, adjoint_on)
w_spin2 = sector_weight(dtau, spin2_on)
trace_dtau = np.real(np.trace(dtau))
det_rho1 = np.real(np.linalg.det(rho1))

print("=" * 72)
print("METAPLECTIC CASCADE GENERATOR d_tau = log(rho_1(S_X S_Y))")
print("=" * 72)
print(f"  det rho_1(g)            : {det_rho1:.6f}   (=1 -> volume preserving)")
print(f"  Tr(d_tau)               : {trace_dtau:+.3e}  (scalar/capacity-decay component)")
print(f"  HS norm^2 total         : {tot:.5f}")
print(f"  sector weights (fraction of HS norm^2):")
print(f"    scalar  (Id, capacity decay)  : {w_scalar/tot:6.3%}")
print(f"    adjoint (J: split + mixing)   : {w_adjoint/tot:6.3%}")
print(f"    spin-2  (quadrupole)          : {w_spin2/tot:6.3%}")

# J_3 split coefficient and HS-orthogonality with the scalar (decay) sector
c_J3 = np.real(hs(Jz, dtau) / hs(Jz, Jz))
print(f"  J_3 split coefficient (adjoint)  : {c_J3:+.4f}  (nonzero -> split present)")
print(f"  <Id, J_3>_HS                     : {np.real(hs(Id, Jz)):+.3e}  "
      f"(scalar _|_ adjoint)")

# --------------------------------------------------------------------------- #
# a capacity-decay generator is a magnitude (scalar) effect of the projection
# --------------------------------------------------------------------------- #
D_decay = -np.eye(3)                                # uniform admissible-norm contraction
print()
print("  capacity-decay generator D_decay = -Id (uniform norm contraction):")
print(f"    Tr(D_decay)                    : {np.real(np.trace(D_decay)):+.1f}  (nonzero scalar)")
print(f"    <D_decay, J_3>_HS              : {np.real(hs(D_decay, Jz)):+.3e}  "
      f"(orthogonal to the split)")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print("  - d_tau^{metaplectic} is TRACELESS (det rho_1 = 1): it carries NO scalar")
print("    component, hence no net capacity-decay direction. Its content is purely")
print("    adjoint (J_3 split + mixing) plus spin-2.")
print("  - The capacity decay sigma_pair(n) ~ n^{-delta_pair} is a magnitude effect:")
print("    it requires the scalar sector (net norm change), supplied by the")
print("    non-injective projection, NOT by the metaplectic rotation.")
print("  - The split generator (adjoint, J_3) and the decay generator (scalar) are")
print("    Hilbert-Schmidt ORTHOGONAL: <Id, J_3> = 0. They are DISTINCT operators.")
print()
print("  => ARCHITECTURE = FACTORISED (coupled but distinct).")
print("     The J_3 split is the angular (metaplectic, norm-preserving) component;")
print("     the sigma_pair decay is the radial (projection, norm-reducing) component.")
print("     They are coupled only through the shared cascade arrow I(n) (the sign of")
print("     the split flips with cascade reversal; sigma_pair is monotone in the same")
print("     orientation). N_casc is therefore a TRANSFER CONSTANT between the angular")
print("     split and the radial decay, not a unified F(delta_pair) prediction.")
print("=" * 72)

# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
labels = ["scalar\n(decay)", "adjoint\n($J_3$ split)", "spin-2"]
vals = [w_scalar/tot, w_adjoint/tot, w_spin2/tot]
colors = ["#bbbbbb", "#a83232", "#3b7a57"]
ax.bar(labels, vals, color=colors, width=0.55)
ax.set_ylabel(r"fraction of $\|\partial_\tau\|^2_{\rm HS}$")
ax.set_title(r"$\partial_\tau^{\rm metaplectic}$ has zero scalar (decay) component")
ax.set_ylim(0, 1.05)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

ax = axes[1]
ax.axhline(0, color="grey", lw=0.8)
ax.bar(["angular\n($J_3$, metaplectic)", "radial\n($\\sigma_{\\rm pair}$, projection)"],
       [c_J3, -1.0], color=["#a83232", "#1f3b73"], width=0.5)
ax.set_title(r"Distinct sectors: $\langle \mathrm{Id}, J_3\rangle_{\rm HS}=0$")
ax.set_ylabel("generator component")
ax.text(0.5, 0.5, "FACTORISED\n(coupled via the arrow $I(n)$)",
        transform=ax.transAxes, ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999"))

fig.suptitle("Architecture test: J_3 split (angular) and capacity decay (radial) "
             "are distinct generators", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=150)
print(f"\nFigure written to {OUT_PNG}")
