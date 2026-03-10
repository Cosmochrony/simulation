"""
Spectral capacity functional - valence-normalized comparison.

C(G, S) = sum_{lambda>0} (dim rho)^2 / sqrt(lambda)

Key issue: raw C is valence-dependent because lambda = d - mu.
We therefore report BOTH:
  - C_norm = C(G,S) / |G|
  - C_scaled = C(G,S) / (|G| * sqrt(d))   [removes sqrt(d) scaling]

For a fair comparison, we also show results at fixed d=6 where possible,
and flag groups computed at other valences.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RepData:
    label: str
    dim: int
    mu: float
    lam: float
    multiplicity: int
    A_max: float


def rep(label, dim, mu, d):
    lam = d - mu
    return RepData(label, dim, mu, lam, dim*dim,
                   1.0/np.sqrt(lam) if lam > 1e-12 else np.inf)


def capacity(reps, order, d):
    nz = [r for r in reps if r.lam > 1e-12]
    C = sum(r.multiplicity / np.sqrt(r.lam) for r in nz)
    C_norm   = C / order
    C_scaled = C / (order * np.sqrt(d))  # removes sqrt(d) dimension
    lam_min  = min(r.lam for r in nz)
    lam_max  = max(r.lam for r in nz)
    R        = np.sqrt(lam_max / lam_min)
    return C, C_norm, C_scaled, lam_min, lam_max, R


# ==========================================================================
# d = 6 groups  (direct comparison)
# ==========================================================================
d6_groups = {}

# Z_8 with S={+/-1, +/-2 mod 8} ... d=4 minimum for abelian of order 8.
# For d=6: use Z_6 with all non-identity, d=5... not clean.
# Best abelian comparison at d=6: Z_2 x Z_2 x ... not order 8.
# Use Z_8 with S = {1,2,3,5,6,7} mod 8 (all non-trivial), d=6:
#   mu_k = sum_{s in S} cos(2pi k s / 8), k=0..7
# S = {1,2,3,5,6,7} -> these are all elements except 0 and 4.
S_Z8 = [1, 2, 3, 5, 6, 7]
d = 6
Z8_reps = []
for k in range(8):
    mu_k = sum(np.cos(2*np.pi*k*s/8) for s in S_Z8)
    Z8_reps.append(rep(f"chi_{k}", 1, float(np.real(mu_k)), d))
d6_groups["Z_8"] = (8, d, Z8_reps)

# D_4 at d=6: S = {r,r^{-1}, s, rs, r^2s, r^3s} (2 rotations + 4 reflections)
# mu values computed above: rho_0->6, rho_1->2, rho_2->-2, rho_3->-2, rho_4->0
d6_groups["D_4"] = (8, 6, [
    rep("rho_0", 1,  6.0, 6),
    rep("rho_1", 1,  2.0, 6),
    rep("rho_2", 1, -2.0, 6),
    rep("rho_3", 1, -2.0, 6),
    rep("rho_4", 2,  0.0, 6),
])

# Q_8 at d=6 (from paper)
d6_groups["Q_8"] = (8, 6, [
    rep("rho_0", 1,  6.0, 6),
    rep("rho_1", 1, -2.0, 6),
    rep("rho_2", 1, -2.0, 6),
    rep("rho_3", 1, -2.0, 6),
    rep("rho_4", 2,  0.0, 6),
])

# 2T at d=6: S = all 6 order-4 elements (one conjugacy class)
# mu: dim-1 reps: 6, 6*Re(omega), 6*Re(omega^2) = 6, -3, -3
# dim-2 reps: 0 (Pauli tracelessness)
# dim-3 rep: (6*(-1))/3 = -2  -> lam=8
omega = np.exp(2j*np.pi/3)
d6_groups["2T"] = (24, 6, [
    rep("rho_0", 1,  6.0,                  6),
    rep("rho_1", 1,  float(np.real(6*omega)),   6),   # mu=-3, lam=9
    rep("rho_2", 1,  float(np.real(6*omega**2)),6),   # mu=-3, lam=9
    rep("rho_3", 2,  0.0,                  6),        # lam=6
    rep("rho_4", 2,  0.0,                  6),
    rep("rho_5", 2,  0.0,                  6),
    rep("rho_6", 3, -2.0,                  6),        # lam=8
])

# ==========================================================================
# Natural generating sets (not d=6)
# ==========================================================================
other_groups = {}

# A_4 at d=11 (all non-identity)
omega = np.exp(2j*np.pi/3)
mu1 = float(np.real(3*1 + 4*omega + 4*omega**2))   # = -1
mu2 = float(np.real(3*1 + 4*omega**2 + 4*omega))   # = -1
other_groups["A_4"] = (12, 11, [
    rep("rho_0", 1, 11.0, 11),
    rep("rho_1", 1, mu1,  11),
    rep("rho_2", 1, mu2,  11),
    rep("rho_3", 3, -1.0, 11),
])

# S_4 at d=12 (transpositions + 4-cycles)
other_groups["S_4"] = (24, 12, [
    rep("rho_1", 1,  12.0, 12),
    rep("rho_2", 1, -12.0, 12),
    rep("rho_3", 2,   0.0, 12),
    rep("rho_4", 3,   0.0, 12),
    rep("rho_5", 3,   0.0, 12),
])

# 2I at d=24 (from paper)
other_groups["2I"] = (120, 24, [
    rep("chi_1", 1,  24.0, 24),
    rep("chi_2", 2,   6.0, 24),
    rep("chi_3", 3,   4.0, 24),
    rep("chi_4", 4,   6.0, 24),
    rep("chi_5", 5,   0.0, 24),
    rep("chi_6", 6,  -4.0, 24),
])

# ==========================================================================
# Print tables
# ==========================================================================
def print_group_table(label, groups_dict):
    print(f"\n{'='*92}")
    print(f"  {label}")
    print(f"{'='*92}")
    hdr = (f"  {'Group':<8} {'|G|':>5} {'d':>4}  "
           f"{'C(G)':>8} {'C_norm':>9} {'C_scaled':>10}  "
           f"{'lam_min':>8} {'lam_max':>8}  {'R':>7}")
    print(hdr)
    print(f"  {'-'*88}")
    rows = []
    for name, (order, d, reps) in groups_dict.items():
        C, C_norm, C_sc, lm, lM, R = capacity(reps, order, d)
        rows.append((C_sc, name, order, d, C, C_norm, C_sc, lm, lM, R))
    rows.sort(key=lambda x: -x[0])
    for row in rows:
        _, name, order, d, C, C_norm, C_sc, lm, lM, R = row
        print(f"  {name:<8} {order:>5} {d:>4}  "
              f"{C:>8.3f} {C_norm:>9.5f} {C_sc:>10.6f}  "
              f"{lm:>8.4f} {lM:>8.4f}  {R:>7.4f}")
    print(f"  {'='*88}")


print_group_table("Fixed valence d=6  (direct fair comparison)", d6_groups)
print_group_table("Natural generating sets  (for completeness)", other_groups)

# ==========================================================================
# Sector comparison within order-8 groups (d=6)
# ==========================================================================
print("\n\nSECTOR DETAIL: order-8 groups at d=6")
print("="*60)
for name in ["Z_8", "D_4", "Q_8"]:
    order, d, reps = d6_groups[name]
    print(f"\n{name}:")
    print(f"  {'Sector':<12} {'dim':>4} {'mu':>7} {'lam':>7} "
          f"{'mult':>6} {'A_max':>8}")
    for r in reps:
        if r.lam < 1e-12:
            tag = "  (zero mode)"
        else:
            tag = ""
        A_str = f"{r.A_max:.4f}" if r.A_max < 1e6 else "inf"
        print(f"  {r.label:<12} {r.dim:>4} {r.mu:>7.3f} {r.lam:>7.3f} "
              f"{r.multiplicity:>6} {A_str:>8}{tag}")

# ==========================================================================
# Sector comparison within order-24 groups
# ==========================================================================
print("\n\nSECTOR DETAIL: order-24 groups")
print("="*60)
for name in ["2T", "S_4"]:
    if name in d6_groups:
        order, d, reps = d6_groups[name]
    else:
        order, d, reps = other_groups[name]
    print(f"\n{name} (d={d}):")
    print(f"  {'Sector':<12} {'dim':>4} {'mu':>7} {'lam':>7} "
          f"{'mult':>6} {'A_max':>8}")
    for r in reps:
        if r.lam < 1e-12:
            continue
        print(f"  {r.label:<12} {r.dim:>4} {r.mu:>7.3f} {r.lam:>7.3f} "
              f"{r.multiplicity:>6} {r.A_max:>8.4f}")

print("\n\nKEY INSIGHT: C_scaled comparison at d=6")
print("="*60)
print("A higher C_scaled means more total admissible relational capacity.")
print("Groups ranked by C_scaled (valence-independent):\n")
all_d6 = []
for name, (order, d, reps) in d6_groups.items():
    C, C_norm, C_sc, lm, lM, R = capacity(reps, order, d)
    all_d6.append((C_sc, name, order))
all_d6.sort(key=lambda x: -x[0])
for rank, (C_sc, name, order) in enumerate(all_d6, 1):
    print(f"  {rank}. {name:<6}  |G|={order:>3}   C_scaled={C_sc:.6f}")
