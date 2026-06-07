"""
epi_generation_diagnostic.py

First-domino diagnostic for the explicit E_Pi programme.

Question (Q14 / FermionicMatterNote open deliverable 1):
    Does the STATIC spectrum of E_Pi^2 restricted to the generation factor C^3_gen
    reproduce the ADE / Kesten-McKay gap structure that the O-series uses to order
    the charged-lepton hierarchy?

        GapSpec( E_Pi^2 |_{C^3_gen} )  ~?~  { |lambda_i^ADE - 1| }

Canonical-model proxy (STRUCTURAL IDENTIFICATION -- explicitly NOT a derivation from
Def. 3.1 of Q14):
    E_Pi^2 |_{C^3_gen}  :=  C_c,
    the rank-3 covariance endomorphism of the Heisenberg-graded generation factor
    Sym^2(V_rho) -> {e_{++}, e_0, e_{--}}, established in O29/O30. We RE-derive its
    spectrum here from the abstract equatorial-constraint ensemble, independently of
    the O29 BFS pipeline, as a bias-independent cross-check of the O30 result.

Outputs:
    - stdout: numerical verdict
    - epi_generation_diagnostic.png: comparison figure
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Parameters (all at the top)
# --------------------------------------------------------------------------- #
N_SAMPLES = 4_000_000            # Monte-Carlo ensemble size for the static spectrum
SEED = 20260607                  # reproducibility
R_DISTRIBUTION = "rayleigh"      # amplitude law for r = |w| ; ratios are r-law-independent
RAYLEIGH_SCALE = 1.0

# ADE three-level data: 2I (binary icosahedral), order-5 generating set
# (SpectralRelaxation / O1). lambda_2 = 1 is the parity-fixed central (midpoint) level.
ADE_LEVELS = {"lambda_1": 5.0 / 6.0, "lambda_2": 1.0, "lambda_3": 5.0 / 4.0}

OUT_PNG = "epi_generation_diagnostic.png"

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# 1. Static generation spectrum from the equatorial-constraint ensemble
#    w = (alpha, beta), |alpha| = |beta| = r / sqrt(2), independent uniform phases.
#    M = w w^T in Sym^2(C^2) basis {e_++, e_0, e_--}:
#        e_++ component = alpha^2
#        e_0  component = sqrt(2) alpha beta     (central, parity-fixed)
#        e_-- component = beta^2
#    C_c = Cov(vec(M)). This reconstructs O30 Theorem 5.1 without the BFS pipeline.
# --------------------------------------------------------------------------- #
def sample_amplitudes(n):
    if R_DISTRIBUTION == "rayleigh":
        r = rng.rayleigh(RAYLEIGH_SCALE, size=n)
    elif R_DISTRIBUTION == "uniform":
        r = rng.uniform(0.0, 1.0, size=n)
    else:
        raise ValueError("unknown R_DISTRIBUTION")
    return r


def static_generation_covariance(n):
    r = sample_amplitudes(n)
    a_mag = r / np.sqrt(2.0)            # equatorial constraint |alpha| = |beta|
    b_mag = r / np.sqrt(2.0)
    pa = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pb = rng.uniform(0.0, 2.0 * np.pi, size=n)
    alpha = a_mag * np.exp(1j * pa)
    beta = b_mag * np.exp(1j * pb)

    v_pp = alpha ** 2
    v_0 = np.sqrt(2.0) * alpha * beta
    v_mm = beta ** 2
    V = np.stack([v_pp, v_0, v_mm], axis=1)   # vec(M) in {e_++, e_0, e_--}

    # zero-mean (random phases) Hermitian covariance
    C = (V.conj().T @ V) / n
    return C


C_c = static_generation_covariance(N_SAMPLES)
eig = np.sort(np.real(np.linalg.eigvalsh(C_c)))[::-1]   # descending
eig_norm = eig / eig[0]                                  # normalise to leading

# identify which eigenvector is the parity-fixed central direction e_0
# (the e_0 basis vector is index 1 in {e_++, e_0, e_--})
diag = np.real(np.diag(C_c))
central_is_max = (np.argmax(diag) == 1)

print("=" * 72)
print("STATIC GENERATION SPECTRUM  (E_Pi^2 |_{C^3_gen} := C_c, canonical proxy)")
print("=" * 72)
print(f"  raw eigenvalues (desc)   : {eig}")
print(f"  normalised [lambda_i/l1] : {np.round(eig_norm, 4)}")
print(f"  O30 prediction           : [1, 0.5, 0.5]   (ratio lambda_1/lambda_2 = 2)")
print(f"  central direction e_0 is the MAX eigenvalue : {central_is_max}")
print(f"  structural ratio (max / pair)               : {eig_norm[0] / eig_norm[1]:.4f}")

# the static outer pair (e_+, e_-) -- gaps of the two non-central levels relative
# to the central e_0 level. Here the two outer levels are DEGENERATE by construction
# (the equatorial / parity symmetry c <-> q-c, i.e. J_Pi = HK).
static_central = eig_norm[0]            # e_0
static_outer = eig_norm[1:]             # e_+, e_-  (degenerate)
static_gaps_from_central = np.abs(static_outer - static_central)


# --------------------------------------------------------------------------- #
# 2. ADE Kesten-McKay gap structure (exact)
#    distance of each level from the support midpoint lambda = 1
# --------------------------------------------------------------------------- #
l1, l2, l3 = ADE_LEVELS["lambda_1"], ADE_LEVELS["lambda_2"], ADE_LEVELS["lambda_3"]
ade_central = l2                               # parity-fixed midpoint, never exits
ade_outer = np.array([l3, l1])                 # exit first (heaviest) then second
ade_gaps_from_central = np.abs(ade_outer - ade_central)   # {1/4, 1/6}
ade_ratio = ade_gaps_from_central[0] / ade_gaps_from_central[1]

print()
print("=" * 72)
print("ADE KESTEN-MCKAY GAP STRUCTURE  (2I, ord-5)")
print("=" * 72)
print(f"  levels {{l1,l2,l3}}        : {l1:.4f}, {l2:.4f}, {l3:.4f}")
print(f"  central (parity-fixed)   : lambda_2 = {ade_central} (gap 0, never exits)")
print(f"  outer gaps |l - 1|       : l3 -> {ade_gaps_from_central[0]:.4f}, "
      f"l1 -> {ade_gaps_from_central[1]:.4f}")
print(f"  outer-gap ratio          : {ade_ratio:.4f}   (= 3/2)")


# --------------------------------------------------------------------------- #
# 3. Verdict
# --------------------------------------------------------------------------- #
static_outer_ratio = (static_gaps_from_central[0] / static_gaps_from_central[1]
                      if static_gaps_from_central[1] > 1e-12 else np.inf)
outer_degenerate = np.isclose(static_outer[0], static_outer[1], atol=5e-3)

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  rank-3 structure                 : MATCH (both have 3 levels)")
print(f"  distinguished central direction  : "
      f"{'MATCH' if central_is_max else 'CHECK'} "
      f"(e_0 <-> lambda_2, both parity-fixed)")
print(f"  outer-pair degeneracy            : "
      f"static {'DEGENERATE (1:1)' if outer_degenerate else 'split'} "
      f"vs ADE SPLIT (3:2)")
print(f"  outer-gap ratio  static / ADE    : {static_outer_ratio:.3f}  /  {ade_ratio:.3f}")
print()
print("  CONCLUSION: the static E_Pi^2 generation spectrum carries the correct")
print("  rank-3 + distinguished-central structure but its outer pair is")
print("  J_Pi-DEGENERATE (protected by the parity c <-> q-c), whereas the ADE")
print("  hierarchy requires a SPLIT outer pair. The degeneracy lifting (3:2)")
print("  is therefore NOT in the static operator: it is dynamical (cascade) content.")
print("  => Q14<->O-series bridge must read: O-series = J_Pi-breaking dynamical")
print("     degeneracy-lifting of the static E_Pi^2 spectrum, NOT a renormalisation")
print("     of an already-matching spectrum.")
print("=" * 72)


# --------------------------------------------------------------------------- #
# 4. Figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# left: static spectrum
ax = axes[0]
labels = [r"$e_0$ (central)", r"$e_+$", r"$e_-$"]
colors = ["#1f3b73", "#3b7a57", "#3b7a57"]
ax.bar(labels, eig_norm, color=colors, width=0.55)
ax.axhline(0.5, ls="--", lw=1, color="grey")
ax.set_title(r"Static $E_\Pi^2|_{\mathbb{C}^3_{\rm gen}}$ "
             r"(proxy $C_c$): $[1:\frac{1}{2}:\frac{1}{2}]$")
ax.set_ylabel(r"normalised eigenvalue")
ax.set_ylim(0, 1.15)
ax.text(0.5, 0.52, r"$e_+,e_-$ degenerate ($J_\Pi$-protected)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#3b7a57")

# right: ADE gap structure
ax = axes[1]
ade_lab = [r"$\lambda_2{=}1$" + "\n(central)",
           r"$\lambda_3{=}\frac{5}{4}$" + "\nexits 1st",
           r"$\lambda_1{=}\frac{5}{6}$" + "\nexits 2nd"]
ade_vals = [0.0, ade_gaps_from_central[0], ade_gaps_from_central[1]]
ax.bar(ade_lab, ade_vals, color=["#1f3b73", "#a83232", "#d98a32"], width=0.55)
ax.set_title(r"ADE gaps $|\lambda_i-1|$ (2I, ord-5): split $3:2$")
ax.set_ylabel(r"$|\lambda_i - 1|$  (support-exit distance)")
ax.set_ylim(0, 0.30)
ax.text(0.5, 0.78, r"outer pair SPLIT $(\frac{1}{4}:\frac{1}{6})=3:2$",
        transform=ax.transAxes, ha="center", fontsize=9, color="#a83232")

fig.suptitle("Diagnostic: static generation spectrum vs ADE hierarchy "
             "-- degeneracy mismatch is the dynamical content",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=150)
print(f"\nFigure written to {OUT_PNG}")
