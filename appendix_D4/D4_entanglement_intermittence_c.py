import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# ============================================================
# Cosmochrony-inspired toy simulation:
# - Relational graph from random geometric sampling
# - Born–Infeld-like saturation of couplings controlled by C in [0,1]
# - Chiral (CP) bias as asymmetric relaxation efficiency wrt Q=±1
# - Spectral analysis of weighted Laplacian
# - "Spectral crowding" proxies DeltaPi:
#     (A) pairwise Gaussian kernel over eigenvalues  -> can show peaks
#     (B) gap-based local crowding proxy             -> typically smoother
# - Effective entanglement observable E_eff(C)=DeltaPi(C)*(1-C^nu)
# ============================================================

# -----------------------------
# Global configuration
# -----------------------------
SEED = 42
np.random.seed(SEED)

# Graph / sampling
N = 250          # number of basis coefficients / nodes in representation
DIM = 2          # embedding dimension for sampling (auxiliary)
R_CUT = 0.35     # cutoff radius for couplings
SIGMA = 0.15     # distance scale for raw kernel

# Saturation
K_MAX = 1.0      # Born–Infeld saturation scale in this toy model

# Spectrum
K_EIGS = 40      # number of low-lying eigenpairs
EIG_WHICH = "SM" # smallest magnitude

# Degeneracy proxy (pairwise)
EPS_DEG = 5e-2   # dimensionless tuning; 1e-2..1e-1 typical
                 # higher => more tolerant => smoother/higher DeltaPi

# Gap-based proxy tuning (relative to median gap)
EPS_GAP = 0.55   # 0.3..0.8 typical; smaller => more "spiky", larger => smoother

# Effective entanglement
NU = 1.5         # exponent in (1 - C^nu) mobility suppression

# Sweep
Cs = np.linspace(0.05, 1.0, 30)

# Chiral bias strength (CP-like admissibility bias)
CHI_BIAS_STRENGTH = 0.15  # keep modest; too big can destabilize weights

# Output files
OUT_PDF = "D4_entanglement_intermittence_c.pdf"
OUT_PNG = "D4_entanglement_intermittence_c.png"


# -----------------------------
# Helper functions
# -----------------------------
def born_infeld_saturation(K0: np.ndarray, C: float, Kmax: float) -> np.ndarray:
    """
    Smooth saturation (Born–Infeld-like) of couplings.
    """
    return Kmax * np.tanh((C * K0) / Kmax)


def build_weighted_laplacian(points: np.ndarray, Q: np.ndarray, C: float) -> csr_matrix:
    """
    Build weighted Laplacian L = D - W from an auxiliary sampling.
    Q is a signed invariant (chirality-like) producing a slight asymmetry in coupling efficiency.
    """
    dist = squareform(pdist(points))

    # Raw kernel (auxiliary representation)
    K0 = np.exp(-(dist**2) / (2.0 * SIGMA**2))
    K0[dist > R_CUT] = 0.0
    np.fill_diagonal(K0, 0.0)

    # CP-like bias: asymmetric efficiency depending on Q difference.
    bias = 1.0 + CHI_BIAS_STRENGTH * (Q[:, None] - Q[None, :])
    K0 = K0 * bias
    K0 = np.clip(K0, 0.0, None)  # ensure nonnegative couplings

    # Born–Infeld saturation controlled by C
    W = born_infeld_saturation(K0, C, K_MAX)

    # Laplacian
    d = W.sum(axis=1)
    L = diags(d) - csr_matrix(W)
    return L


def degeneracy_proxy_pairwise(evals: np.ndarray, eps: float = 5e-2) -> float:
    """
    Pairwise spectral crowding proxy (Gaussian kernel similarity across eigenvalues).

    Returns DeltaPi in [0,1] (approximately):
      - higher when many eigenvalues are close (crowded spectrum)
      - lower when eigenvalues are well-separated

    eps controls the kernel width relative to a robust spectral scale (median gap).
    """
    evals = np.sort(np.asarray(evals, dtype=float))
    if evals.size < 3:
        return 0.0

    gaps = np.diff(evals)
    med = np.median(gaps)

    # Robust fallback if median gap is degenerate
    if not np.isfinite(med) or med <= 0:
        pos = gaps[gaps > 0]
        med = float(np.mean(pos)) if pos.size else 1.0

    sigma = max(eps * med, 1e-12)

    diff = np.abs(evals[:, None] - evals[None, :])
    K = np.exp(-(diff**2) / (2.0 * sigma**2))
    np.fill_diagonal(K, 0.0)

    # Normalize: average pairwise similarity (excluding diagonal)
    n = evals.size
    Delta = float(K.sum() / (n * (n - 1)))
    return Delta


def degeneracy_proxy_gaps(evals: np.ndarray, eps: float = 0.55) -> float:
    """
    Gap-based crowding proxy:
    - compute gaps between consecutive sorted eigenvalues
    - define a scale eps * median_gap
    - return mean exp(-(gap/scale)^2)

    Typically smoother / less sensitive than pairwise-all-pairs.
    """
    evals = np.sort(np.asarray(evals, dtype=float))
    if evals.size < 3:
        return 0.0

    gaps = np.diff(evals)
    med = np.median(gaps)

    if not np.isfinite(med) or med <= 0:
        pos = gaps[gaps > 0]
        med = float(np.mean(pos)) if pos.size else 1.0

    scale = max(eps * med, 1e-12)
    return float(np.mean(np.exp(-(gaps / scale) ** 2)))


def effective_chiral_bias(evecs: np.ndarray, Q: np.ndarray) -> float:
    """
    Measures how much the low-lying eigenmodes are preferentially supported
    on one chiral sector. This is just a diagnostic.

    evecs shape: (N, K_EIGS)
    """
    weights = np.sum((evecs**2) * Q[:, None], axis=0)
    return float(np.mean(weights))


# -----------------------------
# Build a single auxiliary configuration (fixed across C sweep)
# -----------------------------
points = np.random.rand(N, DIM)
Q = np.random.choice([-1, 1], size=N)

# -----------------------------
# Sweep and compute diagnostics
# -----------------------------
DeltaPi_pairwise_vals = []
DeltaPi_gaps_vals = []
Eeff_pairwise_vals = []
Eeff_gaps_vals = []
Chiral_vals = []

for C in Cs:
    L = build_weighted_laplacian(points, Q, C)

    # Low-lying spectrum (L is PSD)
    evals, evecs = eigsh(L, k=K_EIGS, which=EIG_WHICH, return_eigenvectors=True)

    DeltaPi_pair = degeneracy_proxy_pairwise(evals, eps=EPS_DEG)
    DeltaPi_gap = degeneracy_proxy_gaps(evals, eps=EPS_GAP)

    DeltaPi_pairwise_vals.append(DeltaPi_pair)
    DeltaPi_gaps_vals.append(DeltaPi_gap)

    # Effective entanglement observable (compute both for comparison)
    Eeff_pairwise_vals.append(DeltaPi_pair * (1.0 - C**NU))
    Eeff_gaps_vals.append(DeltaPi_gap * (1.0 - C**NU))

    # CP/chiral diagnostic
    Chiral_vals.append(effective_chiral_bias(evecs, Q))

DeltaPi_pairwise_vals = np.array(DeltaPi_pairwise_vals, dtype=float)
DeltaPi_gaps_vals = np.array(DeltaPi_gaps_vals, dtype=float)
Eeff_pairwise_vals = np.array(Eeff_pairwise_vals, dtype=float)
Eeff_gaps_vals = np.array(Eeff_gaps_vals, dtype=float)
Chiral_vals = np.array(Chiral_vals, dtype=float)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# (0) ΔΠ proxies
ax[0].plot(Cs, DeltaPi_pairwise_vals, marker='o', linewidth=2, label=r'$\Delta_\Pi$ (pairwise)')
ax[0].plot(Cs, DeltaPi_gaps_vals, marker='s', linewidth=2, linestyle='--', label=r'$\Delta_\Pi$ (gap-based)')
ax[0].set_ylabel(r'$\Delta_\Pi$')
ax[0].set_title('Spectral crowding diagnostics vs Born–Infeld compression')
ax[0].legend()

# (1) E_eff from both proxies (same (1-C^nu) mobility factor)
ax[1].plot(Cs, Eeff_pairwise_vals, marker='o', linewidth=2, label=r'$E_{\mathrm{eff}}$ (pairwise)')
ax[1].plot(Cs, Eeff_gaps_vals, marker='s', linewidth=2, linestyle='--', label=r'$E_{\mathrm{eff}}$ (gap-based)')
ax[1].set_ylabel(r'$E_{\mathrm{eff}}(\mathcal{C})$')
ax[1].set_title(r'Effective entanglement observable: $\Delta_\Pi(\mathcal{C})\,(1-\mathcal{C}^\nu)$')
ax[1].legend()

# (2) Chiral diagnostic
ax[2].plot(Cs, Chiral_vals, marker='s', linewidth=2)
ax[2].set_xlabel(r'Compression ratio $\mathcal{C}$')
ax[2].set_ylabel('Effective chiral bias')
ax[2].set_title('CP-like chiral selection diagnostic')

plt.tight_layout()

# Save PDF/PNG (vector PDF)
plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
plt.savefig(OUT_PNG, format="png", dpi=200, bbox_inches="tight")

plt.show()

# -----------------------------
# Console summary (optional)
# -----------------------------
imax_pair = int(np.argmax(Eeff_pairwise_vals))
imax_gap = int(np.argmax(Eeff_gaps_vals))

print("Summary (pairwise proxy):")
print(f"  max E_eff at C = {Cs[imax_pair]:.3f}")
print(f"  DeltaPi(C*) = {DeltaPi_pairwise_vals[imax_pair]:.6f}")
print(f"  Eeff(C*)    = {Eeff_pairwise_vals[imax_pair]:.6f}")
print(f"  Chiral(C*)  = {Chiral_vals[imax_pair]:.6f}")
print()
print("Summary (gap-based proxy):")
print(f"  max E_eff at C = {Cs[imax_gap]:.3f}")
print(f"  DeltaPi(C*) = {DeltaPi_gaps_vals[imax_gap]:.6f}")
print(f"  Eeff(C*)    = {Eeff_gaps_vals[imax_gap]:.6f}")
print(f"  Chiral(C*)  = {Chiral_vals[imax_gap]:.6f}")
print()
print("Tuning tips:")
print("  - EPS_DEG (pairwise): increase => smoother/higher DeltaPi_pairwise.")
print("  - EPS_GAP (gap-based): increase => smoother/higher DeltaPi_gaps.")
print("  - NU controls suppression near C→1.")
print("  - R_CUT and SIGMA control graph density; too sparse => noisy spectrum, too dense => washed structure.")
print(f"\nSaved: {OUT_PDF} and {OUT_PNG}")
