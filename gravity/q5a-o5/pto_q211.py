"""Compute PTO Table 1 row for q=211 from existing npz."""
import numpy as np, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'o25_outputs/q211_o25.npz'
d = np.load(path, allow_pickle=True)

q     = int(d['q'])
ns    = np.asarray(d['ns'], dtype=int)
sigma = np.asarray(d['sigma_pair_mean'], dtype=float)  # (n_pairs, n_shells)
n_pairs = sigma.shape[0]

# Burn-in: skip shell 0 (I(0)=0 by definition)
# Compute I(n) = sigma_pair(0) - sigma_pair(n)
sigma0 = sigma[:, 0:1]  # (n_pairs, 1)
I = sigma0 - sigma      # (n_pairs, n_shells), I[p,0]=0

# Strict monotonicity: I(n+1) >= I(n) ↔ sigma(n+1) <= sigma(n)
# Threshold: relative decrease > 1e-6 of plateau
sigma_plateau = sigma[:, -1]  # minimum (plateau) value
regressions = 0
reg_pairs = []
for p in range(n_pairs):
    plateau = sigma_plateau[p]
    threshold = 1e-6 * max(sigma[p, 0], 1e-30)
    for n in range(1, len(ns)):
        decrease = sigma[p, n-1] - sigma[p, n]
        if decrease < -threshold:
            regressions += 1
            reg_pairs.append(p)
            break

n_reg_pairs = len(reg_pairs)
mono_frac = (n_pairs - n_reg_pairs) / n_pairs

# Count post-burn-in steps (all shells after n=0)
# Table 1 uses total post-burn-in shell steps summed over pairs
# From paper: q=151 has 75 pairs × 138/75 ≈ 1.84 steps/pair...
# Actually it's just the number of shells in the BFS minus burn-in (n=0)
steps = len(ns) - 1  # shells 1..N

# Mean monotonicity
mean_mono = "YES" if n_reg_pairs == 0 else "NO"

print(f"q={q}  n_pairs={n_pairs}  n_shells={len(ns)}  steps(post-burnin)={steps}")
print(f"Regressions: {n_reg_pairs} pairs")
print(f"Mono fraction: {mono_frac:.4f}")
print(f"Mean mono: {mean_mono}")
print()
print("Table 1 row:")
print(f"  {q} & {n_pairs} & {steps} & {mean_mono} & {n_reg_pairs} & {mono_frac:.4f} \\\\")