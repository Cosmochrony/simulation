import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)

N = 250                 # number of relational nodes
dim = 2                 # embedding dimension
k_eigs = 40             # low-lying spectrum
r_cut = 0.35            # relational cutoff
sigma = 0.15            # interaction scale
K_max = 1.0             # Born–Infeld saturation
eps_deg = 5e-2  # ou 1e-2 à 1e-1 selon la courbe voulue

Cs = np.linspace(0.05, 1.0, 30)  # compression / saturation sweep

# -----------------------------
# Relational configuration
# -----------------------------
points = np.random.rand(N, dim)

# Assign a chiral invariant Q = ±1
Q = np.random.choice([-1, 1], size=N)

# -----------------------------
# Utility functions
# -----------------------------
def born_infeld_saturation(K0, C, Kmax):
    return Kmax * np.tanh((C * K0) / Kmax)

def build_laplacian(points, Q, C):
    dist = squareform(pdist(points))
    K0 = np.exp(-(dist**2) / (2 * sigma**2))
    K0[dist > r_cut] = 0.0
    np.fill_diagonal(K0, 0.0)

    # Chiral bias: asymmetric relaxation efficiency
    chiral_bias = 1.0 + 0.15 * (Q[:, None] - Q[None, :])
    K0 *= chiral_bias

    W = born_infeld_saturation(K0, C, K_max)

    D = np.sum(W, axis=1)
    L = diags(D) - csr_matrix(W)
    return L


def degeneracy_proxy(evals, eps=1e-3):
  """
  Continuous 'spectral crowding' proxy (drop-in replacement).

  Returns:
    Delta: in [0,1] (higher = more quasi-degeneracy / stronger compression)
    groups: diagnostic dict (kept for API compatibility)
  """
  evals = np.sort(np.asarray(evals, dtype=float))
  gaps = np.diff(evals)

  # Robust scale for "what counts as small gap"
  # If all gaps ~0 (rare), avoid division by zero.
  med = np.median(gaps)
  if not np.isfinite(med) or med <= 0:
    med = np.mean(gaps[gaps > 0]) if np.any(gaps > 0) else 1.0

  # Use eps as a dimensionless tuning knob:
  # eps ~ 1e-2 to 1e-1 is typical for smooth curves.
  sigma_gap = max(eps * med, 1e-12)

  # Crowd score: small gaps contribute ~1, large gaps contribute ~0
  crowd = np.exp(-gaps / sigma_gap)

  Delta = float(np.mean(crowd))  # normalized to [0,1]
  groups = {"median_gap": float(med), "sigma_gap": float(sigma_gap)}
  return Delta, groups

def chiral_asymmetry(evecs, Q):
    # project eigenmodes onto chirality
    weights = np.sum((evecs**2) * Q[:, None], axis=0)
    return np.mean(weights)

# -----------------------------
# Main sweep
# -----------------------------
Delta_vals = []
Chiral_vals = []

for C in Cs:
    L = build_laplacian(points, Q, C)

    evals, evecs = eigsh(
        L, k=k_eigs, which='SM', return_eigenvectors=True
    )

    Delta, groups = degeneracy_proxy(evals, eps=eps_deg)
    Delta_vals.append(Delta)

    chiral = chiral_asymmetry(evecs, Q)
    Chiral_vals.append(chiral)

Delta_vals = np.array(Delta_vals)
Chiral_vals = np.array(Chiral_vals)

# -----------------------------
# Plots
# -----------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax[0].plot(Cs, Delta_vals, 'o-', lw=2)
ax[0].set_ylabel(r'$\Delta_\Pi$ (normalized)')
ax[0].set_title('Spectral degeneracy vs Born–Infeld compression')

ax[1].plot(Cs, Chiral_vals, 's-', lw=2, color='darkred')
ax[1].set_xlabel(r'Compression ratio $\mathcal{C}$')
ax[1].set_ylabel('Effective chiral bias')

plt.tight_layout()
plt.show()
