"""
epi_oriented_lift_test.py

Second domino for the explicit E_Pi programme.

The static diagnostic (epi_generation_diagnostic.py) established:
    E_Pi^2 |_{C^3_gen}  (proxy C_c) has spectrum [1 : 1/2 : 1/2] with the outer pair
    {e_+, e_-} DEGENERATE, protected by the parity c <-> q-c, i.e. by J_Pi = HK,
    whereas the ADE hierarchy requires a SPLIT outer pair (gaps 1/4 : 1/6 = 3:2).

This script tests the oriented-lift hypothesis:
    E_{Pi,dyn}^2 = C_c + eps * T_tau,
with T_tau the cascade lift restricted to the generation factor, required to be
    (i)  T_tau e_0 = 0                  (invisible on the central, parity-fixed level)
    (ii) T_tau({e_+, e_-}) subset {e_+, e_-}   (preserves the outer doublet)
    (iii) J_Pi T_tau J_Pi^{-1} = - T_tau        (odd sector: lifts via orientation,
                                                 J_Pi stays the reality structure)

Goal: identify which natural Heisenberg-graded generator carries this signature, and
the eps that reproduces the ADE outer ratio 3:2.

Construction is bias-independent: the rank-3 generators are built from V = C^2 by the
Sym^2 tensor (Leibniz) construction, not postulated as 3x3 matrices.

Outputs:
    - stdout: parity classification + calibration
    - epi_oriented_lift_test.png
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
STATIC_SPECTRUM = (1.0, 0.5, 0.5)        # C_c on (e_0, e_+, e_-), from diagnostic 1
ADE_OUTER_RATIO = 3.0 / 2.0              # target: (1/4)/(1/6) for 2I ord-5
EPS_GRID = np.linspace(0.0, 0.25, 501)
OUT_PNG = "epi_oriented_lift_test.png"

# --------------------------------------------------------------------------- #
# 1. Build V = C^2 and the Sym^2 generation factor by tensor construction
#    weight basis of V: v_+ (J3 = +1/2), v_- (J3 = -1/2)
#    Sym^2 basis: e_++ = v_+^2 (wt +1), e_0 = sqrt2 v_+v_- (wt 0), e_-- = v_-^2 (wt -1)
#    Working order on Sym^2 is (e_0, e_+, e_-) == (e_0, e_++, e_--)
# --------------------------------------------------------------------------- #
vp = np.array([1.0, 0.0])
vm = np.array([0.0, 1.0])


def kron_sym(a, b):
    return np.kron(a, b)


# symmetric basis vectors in C^2 (x) C^2 (dim 4, index = 2i+j)
e_pp = kron_sym(vp, vp)
e_0 = (kron_sym(vp, vm) + kron_sym(vm, vp)) / np.sqrt(2.0)
e_mm = kron_sym(vm, vm)
B = np.stack([e_0, e_pp, e_mm], axis=1)        # 4x3, columns = (e_0, e_+, e_-)


def lift_single_factor(A_V):
    """Promote a 1-factor operator A on V to Sym^2 by Leibniz A(x)I + I(x)A,
    then express it as a 3x3 matrix in the (e_0, e_+, e_-) basis."""
    A4 = np.kron(A_V, np.eye(2)) + np.kron(np.eye(2), A_V)
    # project onto the symmetric basis: M = B^+ A4 B  (B has orthonormal columns)
    return B.T @ A4 @ B


def conjugate_factor(P_V):
    """Promote a multiplicative parity P on V (P(x)P) to Sym^2 in (e_0,e_+,e_-)."""
    P4 = np.kron(P_V, P_V)
    return B.T @ P4 @ B


# weight generator J3 on V, parity swap on V
J3_V = np.diag([0.5, -0.5])
P_V = np.array([[0.0, 1.0], [1.0, 0.0]])        # v_+ <-> v_-

J3_S = lift_single_factor(J3_V)                 # expect diag(0, 1, -1)
P_S = conjugate_factor(P_V)                     # expect swap e_+ <-> e_-, fix e_0

print("=" * 72)
print("REP-THEORY CHECK (built from V = C^2, not postulated)")
print("=" * 72)
print("  J3 on Sym^2 in (e_0,e_+,e_-):")
print(np.round(J3_S, 6))
print("  parity P on Sym^2 in (e_0,e_+,e_-):")
print(np.round(P_S, 6))

# --------------------------------------------------------------------------- #
# 2. J_Pi parity test.  J_Pi = P . (complex conjugation).
#    For an operator T:  J_Pi T J_Pi^{-1} = P conj(T) P.
#    even: = +T ;  odd: = -T.
# --------------------------------------------------------------------------- #
def jpi_image(T):
    return P_S @ np.conjugate(T) @ P_S


def parity(T, tol=1e-9):
    img = jpi_image(T)
    if np.allclose(img, T, atol=tol):
        return "even"
    if np.allclose(img, -T, atol=tol):
        return "odd"
    return "mixed"


def annihilates_e0(T, tol=1e-9):
    # e_0 is column/index 0 in our ordering
    return np.allclose(T[:, 0], 0.0, atol=tol) and np.allclose(T[0, :], 0.0, atol=tol)


def preserves_doublet(T, tol=1e-9):
    # doublet = indices {1,2}; T must not couple e_0 to the doublet (already by above)
    return np.allclose(T[0, 1:], 0.0, atol=tol) and np.allclose(T[1:, 0], 0.0, atol=tol)


# candidate generators on (e_0, e_+, e_-)
D_carnot = np.diag([2.0, 1.0, 1.0])             # Carnot dilation (degree 2 / degree 1)
T_weight = J3_S                                 # weight / horizontal-rotation generator
T_mix = np.array([[0, 0, 0],                    # off-diagonal antisym on the doublet
                  [0, 0, 1.0],
                  [0, -1.0, 0]])

candidates = {
    "Carnot dilation D = diag(2,1,1)": D_carnot,
    "weight J3 = diag(0,1,-1)": T_weight,
    "doublet mixing (antisym)": T_mix,
}

print()
print("=" * 72)
print("CANDIDATE GENERATOR CLASSIFICATION")
print("=" * 72)
print(f"  {'generator':35s} {'kills e_0':>10s} {'keeps doublet':>14s} {'J_Pi parity':>12s}")
for name, T in candidates.items():
    print(f"  {name:35s} {str(annihilates_e0(T)):>10s} "
          f"{str(preserves_doublet(T)):>14s} {parity(T):>12s}")

# --------------------------------------------------------------------------- #
# 3. Admissible J_Pi-odd lift space (annihilating e_0, preserving doublet)
#    parametrise the doublet block M = [[a, b],[c, d]] and impose the three conditions
# --------------------------------------------------------------------------- #
def admissible_dim():
    # basis of 2x2 real matrices; keep those that are J_Pi-odd on the doublet.
    # J_Pi on doublet = swap(e+,e-) . conj ; for real M: P2 M P2 = -M, P2=[[0,1],[1,0]]
    P2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    basis = [np.array([[1.0, 0], [0, 0]]), np.array([[0, 1.0], [0, 0]]),
             np.array([[0, 0], [1.0, 0]]), np.array([[0, 0], [0, 1.0]])]
    odd = []
    for M in basis:
        odd.append((P2 @ M @ P2 + M).flatten())   # even part must vanish for odd
    A = np.stack(odd, axis=1)
    # odd solutions: M with P2 M P2 = -M  => (P2 M P2 + M) = 0
    ns = A.shape[1] - np.linalg.matrix_rank(A)
    return ns


dim_odd = admissible_dim()
print()
print(f"  dim of J_Pi-odd admissible lift space on the doublet = {dim_odd}")
print("    -> 1 real-split direction (weight J3: real eigenvalues, mass split)")
print("    -> 1 imaginary-mixing direction (antisym: imaginary eigenvalues, mixing)")

# --------------------------------------------------------------------------- #
# 4. Calibration with T_tau = weight generator J3
# --------------------------------------------------------------------------- #
c0, cp, cm = STATIC_SPECTRUM
def outer_ratio(eps):
    lp = cp + eps
    lm = cm - eps
    return lp / lm


ratios = np.array([outer_ratio(e) for e in EPS_GRID])
idx = int(np.argmin(np.abs(ratios - ADE_OUTER_RATIO)))
eps_star = EPS_GRID[idx]
# exact solve: (1/2+eps)/(1/2-eps) = 3/2  ->  eps = 1/10
eps_exact = 0.5 * (ADE_OUTER_RATIO - 1.0) / (ADE_OUTER_RATIO + 1.0)

print()
print("=" * 72)
print("CALIBRATION  E_dyn^2 = C_c + eps * J3")
print("=" * 72)
print(f"  target outer ratio (ADE)         : {ADE_OUTER_RATIO:.4f}")
print(f"  eps reproducing it (grid)        : {eps_star:.4f}")
print(f"  eps reproducing it (exact)       : {eps_exact:.4f}  (= 1/10)")
print(f"  resulting outer eigenvalues      : "
      f"({cp + eps_exact:.3f}, {cm - eps_exact:.3f})  ratio {outer_ratio(eps_exact):.4f}")
print(f"  central level e_0 (unchanged)    : {c0:.3f}")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print("  - Carnot dilation is J_Pi-EVEN: cannot lift the degeneracy.")
print("  - The weight generator J3 is the UNIQUE real-split J_Pi-odd direction that")
print("    kills e_0 and preserves the doublet -> it carries the correct SIGNATURE.")
print("  - SIGNATURE confirmed (qualitative): real split, central intact, doublet kept,")
print("    J_Pi-odd. The cascade lift must be the J3-projection of d_tau.")
print("  - AMPLITUDE eps = 1/10 reproduces ADE 3:2 but is a CALIBRATION, not a")
print("    prediction: relating eps to the cascade normalisation (c_chi -> delta_pair")
print("    -> beta*) is the next step. (Note the proximity eps=0.10 vs beta*~0.126 is")
print("    NOT established and likely dimensionally distinct -- to be ruled out, not in.)")
print("=" * 72)

# --------------------------------------------------------------------------- #
# 5. Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
lp = cp + EPS_GRID
lm = cm - EPS_GRID
ax.plot(EPS_GRID, np.full_like(EPS_GRID, c0), color="#1f3b73", lw=2,
        label=r"$e_0$ central (fixed)")
ax.plot(EPS_GRID, lp, color="#a83232", lw=2, label=r"$e_+$ (lifted up)")
ax.plot(EPS_GRID, lm, color="#d98a32", lw=2, label=r"$e_-$ (lifted down)")
ax.axvline(eps_exact, ls="--", color="grey", lw=1)
ax.scatter([eps_exact, eps_exact], [cp + eps_exact, cm - eps_exact],
           color="k", zorder=5, s=30)
ax.annotate(r"$\varepsilon=\frac{1}{10}\Rightarrow 3:2$", xy=(eps_exact, cp + eps_exact),
            xytext=(eps_exact + 0.02, 0.72), fontsize=10)
ax.set_xlabel(r"lift amplitude $\varepsilon$")
ax.set_ylabel(r"$E_{\Pi,\rm dyn}^2$ eigenvalue")
ax.set_title(r"Oriented lift $C_c + \varepsilon J_3$: degeneracy splits, $e_0$ untouched")
ax.set_ylim(0, 1.1)
ax.legend(loc="center right", fontsize=9)

ax = axes[1]
names = ["Carnot\ndilation", "weight\n$J_3$", "doublet\nmixing"]
par = [parity(D_carnot), parity(T_weight), parity(T_mix)]
lifts = ["no (even)", "REAL split", "imag. (mixing)"]
colors = ["#bbbbbb" if p == "even" else "#a83232" for p in par]
ax.bar(names, [0.4, 1.0, 0.6], color=colors, width=0.55)
for i, (p, l) in enumerate(zip(par, lifts)):
    ax.text(i, 0.05, f"{p}\n{l}", ha="center", va="bottom", fontsize=9, color="white"
            if colors[i] != "#bbbbbb" else "black")
ax.set_title(r"Generator classification under $J_\Pi$")
ax.set_ylabel("lifts the degeneracy?")
ax.set_yticks([])
ax.set_ylim(0, 1.2)

fig.suptitle("Oriented-lift test: J3 is the unique real-split J_Pi-odd direction; "
             "eps=1/10 hits ADE 3:2", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=150)
print(f"\nFigure written to {OUT_PNG}")
