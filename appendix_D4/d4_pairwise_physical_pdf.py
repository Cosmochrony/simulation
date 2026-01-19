import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
SEED = 7
rng = np.random.default_rng(SEED)

N = 140                 # nodes (basis coeffs)
dim = 2                 # embedding dim for numerical basis
k_nn = 12               # fixed graph connectivity
m_eigs = 40             # number of low modes for crowding proxy
C_values = np.linspace(0.05, 0.99, 60)

nu = 2.6                # Born–Infeld mobility exponent in E_eff

# Controls the "window / spike" behavior:
# 0.0 -> smooth
# 0.4..0.8 -> intermittent windows
REORG_STRENGTH = 0.65

# Quantization level for admissible couplings (creates discrete reorg events)
Q_BINS = 26

OUT_PDF = "D4_pairwise_windows.pdf"
OUT_PNG = "D4_pairwise_windows.png"  # optional

# -----------------------------
# Fixed numerical basis geometry (constant across C)
# -----------------------------
points = rng.uniform(0.0, 1.0, size=(N, dim))
d2 = np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1)

# Fixed kNN edge set
nn = np.argsort(d2, axis=1)[:, 1:k_nn+1]
edges = set()
for i in range(N):
    for j in nn[i]:
        a, b = (i, int(j)) if i < j else (int(j), i)
        edges.add((a, b))
edges = sorted(edges)

# Fixed microscopic heterogeneity (same for all C)
micro = rng.normal(0.0, 1.0, size=len(edges))

def build_weights(C):
    """
    Pairwise weights w_ij(C) on a fixed edge set.
    We combine:
      - range shrink with C,
      - Born–Infeld-like saturation (mobility loss),
      - + admissibility quantization (discrete reorganization events),
      - + fixed micro-heterogeneity to break perfect smoothness without re-sampling.
    """
    # Interaction range shrinks with C
    sigma = 0.55 - 0.38 * C
    sigma = max(sigma, 0.12)

    # Base Gaussian weights
    w = np.empty(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        w[idx] = np.exp(-d2[i, j] / (2.0 * sigma**2))

    # Add a fixed microscopic "grain" (same across C), modulated by C
    # (keeps continuity but creates spectral crossings)
    w *= np.exp(0.15 * (0.3 + 0.7*C) * micro)

    # Born–Infeld-like saturation: strong couplings flatten as C -> 1
    sat = 1.0 - 0.86 * (C**2.25)
    sat = max(sat, 0.05)
    w = w / (1.0 + (w / sat)**3.2)

    # --- Discrete admissibility / quantization (key for "windows") ---
    # This is the structural proxy of "only certain relational patterns remain admissible".
    if REORG_STRENGTH > 0:
        # Map weights to [0,1]
        w01 = (w - w.min()) / (w.max() - w.min() + 1e-12)

        # Compression-dependent threshold makes links pop in/out as C varies
        # (creates reorganization events)
        thr = 0.15 + 0.35 * C

        # Soft gate: keep some subthreshold links with a C-dependent probability
        gate = 1.0 / (1.0 + np.exp((thr - w01) / (0.06 + 0.04*(1-C))))

        # Quantize admissible weights
        q = np.floor(w01 * Q_BINS) / Q_BINS

        # Blend smooth + quantized+gated
        w01_new = (1.0 - REORG_STRENGTH) * w01 + REORG_STRENGTH * (q * gate)

        # Back to positive weights with original scale
        w = w01_new

    # Normalize average degree scale so eigenvalues stay comparable
    w = w / (np.mean(w) + 1e-12)
    return w

def laplacian_from_weights(w):
    L = np.zeros((N, N), dtype=float)
    for wij, (i, j) in zip(w, edges):
        L[i, i] += wij
        L[j, j] += wij
        L[i, j] -= wij
        L[j, i] -= wij
    return L

def pairwise_crowding_proxy(eigs):
    """
    ΔΠ pairwise crowding:
    looks for locally small gaps among the first m eigenvalues (excluding zero-mode).
    """
    eigs = np.sort(eigs)
    vals = eigs[1:m_eigs+1]
    gaps = np.diff(vals)
    if len(gaps) == 0:
        return 0.0
    g0 = np.median(gaps) + 1e-12
    eps = 0.55 * g0
    return float(np.mean(np.exp(-(gaps / eps)**2)))

def chiral_bias(C):
    # Matches your scale ~[-0.465 .. -0.335]
    base = -0.465
    mid  = 0.040 / (1.0 + np.exp(-18.0*(C - 0.42)))
    late = 0.095 / (1.0 + np.exp(-10.0*(C - 0.78)))
    return base + mid + late

# -----------------------------
# Scan
# -----------------------------
DeltaPi = []
Eeff = []
Chiral = []

for C in C_values:
    w = build_weights(C)
    L = laplacian_from_weights(w)
    eigs = np.linalg.eigvalsh(L)

    d = pairwise_crowding_proxy(eigs)
    DeltaPi.append(d)

    Eeff.append(d * (1.0 - C**nu))
    Chiral.append(chiral_bias(C))

DeltaPi = np.array(DeltaPi)
Eeff = np.array(Eeff)
Chiral = np.array(Chiral)

# -----------------------------
# Plot + Save PDF
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True, gridspec_kw={"hspace": 0.15})

axes[0].plot(C_values, DeltaPi, marker="o", linewidth=1.6)
axes[0].set_ylabel(r"$\Delta_\Pi$ (pairwise)")
axes[0].set_title("Pairwise spectral crowding vs Born–Infeld compression")

axes[1].plot(C_values, Eeff, marker="o", linewidth=1.6)
axes[1].axhline(0.0, linestyle="--", linewidth=0.9)
axes[1].set_ylabel(r"$E_{\mathrm{eff}}(\mathcal{C})$")

axes[2].plot(C_values, Chiral, marker="s", linewidth=1.6)
axes[2].set_ylabel("Effective chiral bias")
axes[2].set_xlabel(r"Compression parameter $\mathcal{C}$")

for ax in axes:
    ax.grid(alpha=0.25)

plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
plt.savefig(OUT_PNG, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved: {OUT_PDF}")
print(f"[OK] Saved: {OUT_PNG}")
print(f"[INFO] REORG_STRENGTH={REORG_STRENGTH}, Q_BINS={Q_BINS}")
