"""
O3 — Projective Dynamics and Mass Hierarchy
Numerical computations: ADE eigenvalues, exit ranks, mass ratios, beta windows.

All results reported in the paper are reproduced here.
Sections of the paper are indicated in comments.
"""

import math

# ============================================================
# 1. ADE substrate data (Section 4.2)
# ============================================================
# Raw Cayley graph eigenvalues from SpectralStratigraphy [4].
# Normalised convention: lambda_norm = lambda_raw / |S|
# where |S| = degree of Cayley graph = size of generating set.
# This matches the normalised Laplacian L = I - A/d used for LPS graphs
# in SpectralRelaxation [5] (confirmed by Tables 1-3 therein).

substrates = {
    "2I_ord4": {
        "label":    r"2I, ord-4 generators",
        "S_size":   30,        # |S|
        "raw":      [24, 30, 40],
        "fracs":    ["4/5", "1", "4/3"],
        "dim_rho":  [25, 76, 18],   # multiplicities from SpectralStratigraphy
    },
    "2I_ord5": {
        "label":    r"2I, ord-5 generators",
        "S_size":   24,
        "raw":      [20, 24, 30],
        "fracs":    ["5/6", "1", "5/4"],
        "dim_rho":  [54, 25, 40],
    },
}

print("=" * 65)
print("1. ADE NORMALISED EIGENVALUES AND SPECTRAL DISTANCES")
print("=" * 65)

for key, sub in substrates.items():
    d = sub["S_size"]
    lam_norm = [r / d for r in sub["raw"]]
    dist = [abs(l - 1) for l in lam_norm]
    sub["lam_norm"] = lam_norm
    sub["dist"] = dist

    print(f"\n{sub['label']}  (|S| = {d})")
    print(f"  Raw eigenvalues : {sub['raw']}")
    print(f"  Normalised      : {[f'{l:.6f}' for l in lam_norm]}")
    print(f"  Fractions       : {sub['fracs']}")
    print(f"  d_i = |lambda_i - 1| : {[f'{x:.6f}' for x in dist]}")
    print(f"  d1/d3 = {dist[0]:.6f} / {dist[2]:.6f} = {dist[0]/dist[2]:.6f}",
          f"= {int(round(dist[0]*6))}/{int(round(dist[2]*6))}"
          if key == "2I_ord5" else f"= 3/5")
    print(f"  Asymmetry check |d3| > |d1|: {dist[2] > dist[0]}")

# ============================================================
# 2. Exit-rank formula (Section 3, Remark 3.3 and Eq. 3.9)
# ============================================================
# Under p(n) ~ A * n^beta (power-law growth), the exit rank is:
#   n_exit(lambda_i) ~ (4 / (A * (lambda_i - 1)^2))^{1/beta}
# The prefactor A cancels in all ratios.
# The mass ratio is (Eq. 4.7 / Eq. 3.10):
#   M_i / M_j = (|lambda_i - 1| / |lambda_j - 1|)^{2/beta}

print("\n" + "=" * 65)
print("2. MASS RATIO FORMULA — ANALYTICAL CHECK")
print("=" * 65)

for key, sub in substrates.items():
    d1, d3 = sub["dist"][0], sub["dist"][2]
    ratio = d1 / d3
    print(f"\n{sub['label']}")
    print(f"  d1/d3 = {ratio:.6f}")
    print(f"  M1/M3 = (d1/d3)^(2/beta) = ({ratio:.4f})^(2/beta)")
    print(f"  log10(M1/M3) = (2/beta) * log10({ratio:.4f})"
          f" = (2/beta) * {math.log10(ratio):.6f}")

# ============================================================
# 3. Predicted mass ratios for selected beta values (Table 2)
# ============================================================

print("\n" + "=" * 65)
print("3. PREDICTED log10(M1/M3) FOR SELECTED BETA (Table 2)")
print("=" * 65)

betas_table = [0.05, 0.07, 0.09, 0.10, 0.12, 0.13, 0.20, 0.30, 0.50, 1.00]
window = {0.09, 0.10, 0.12, 0.13}

print(f"\n{'beta':>6}  {'ord-4':>12}  {'ord-5':>12}  {'window?':>8}")
print("-" * 45)
for b in betas_table:
    vals = {}
    for key, sub in substrates.items():
        d1, d3 = sub["dist"][0], sub["dist"][2]
        vals[key] = (2 / b) * math.log10(d1 / d3)
    mark = "  <--" if b in window else ""
    print(f"{b:6.2f}  {vals['2I_ord4']:12.3f}  {vals['2I_ord5']:12.3f}{mark}")

# ============================================================
# 4. Observed SM lepton ratios (PDG 2024) — Section 4.4
# ============================================================

print("\n" + "=" * 65)
print("4. OBSERVED SM LEPTON RATIOS (PDG 2024)")
print("=" * 65)

me, mmu, mtau = 0.51099895, 105.6583755, 1776.86   # MeV
sm = {
    "me/mtau":  me / mtau,
    "me/mmu":   me / mmu,
    "mmu/mtau": mmu / mtau,
}
for label, val in sm.items():
    print(f"  {label:12s} = {val:.4e}   log10 = {math.log10(val):.4f}")

# ============================================================
# 5. Beta* compatibility windows (Eq. 4.10 / Section 4.4)
# ============================================================
# Solve: (d1/d3)^{2/beta} = target
# => beta = 2 * log(d1/d3) / log(target)

print("\n" + "=" * 65)
print("5. BETA* COMPATIBILITY WINDOWS (Eq. 4.10)")
print("=" * 65)

for key, sub in substrates.items():
    d1, d3 = sub["dist"][0], sub["dist"][2]
    print(f"\n{sub['label']}  (d1/d3 = {d1/d3:.6f})")
    for label, target in sm.items():
        beta_star = 2 * math.log(d1 / d3) / math.log(target)
        predicted = (2 / beta_star) * math.log10(d1 / d3)
        print(f"  beta* for {label:12s}: {beta_star:.4f}  "
              f"(check: log10(M1/M3) = {predicted:.4f} vs target {math.log10(target):.4f})")

# ============================================================
# 6. Precise check at beta* = 0.125 for 2I ord-4 (Section 4.4)
# ============================================================

print("\n" + "=" * 65)
print("6. PRECISION CHECK AT beta* = 0.125 (2I ord-4)")
print("=" * 65)

beta_check = 0.125
d1_4 = substrates["2I_ord4"]["dist"][0]
d3_4 = substrates["2I_ord4"]["dist"][2]
pred = (2 / beta_check) * math.log10(d1_4 / d3_4)
obs  = math.log10(me / mtau)
print(f"  (2/0.125) * log10(3/5) = 16 * log10(3/5)")
print(f"  = 16 * {math.log10(3/5):.6f} = {pred:.4f}")
print(f"  Observed log10(me/mtau) = {obs:.4f}")
print(f"  Discrepancy: {abs(pred - obs):.4f}  ({100*abs(pred-obs)/abs(obs):.2f}% in log)")

# Alternative: exact fraction
pred_exact = 16 * math.log10(3 / 5)
print(f"  Exact: 16 * log10(3/5) = {pred_exact:.6f}")

# ============================================================
# 7. Quark sector analysis (Section 7.2)
# ============================================================

print("\n" + "=" * 65)
print("7. QUARK SECTOR — BETA* AND RATIO beta*_quark/beta*_lepton")
print("=" * 65)

# PDG 2024 MSbar masses at 2 GeV
mu_q, mc_q, mt_q = 2.2e-3, 1.27, 172.7    # GeV
md_q, ms_q, mb_q = 4.7e-3, 0.093, 4.18    # GeV

quark_ratios = {
    "mu/mt (up-type)":   mu_q / mt_q,
    "mu/mc":             mu_q / mc_q,
    "mc/mt":             mc_q / mt_q,
    "md/mb (down-type)": md_q / mb_q,
}

print("\n  Quark mass ratios (PDG 2024):")
for label, val in quark_ratios.items():
    print(f"  {label:22s} = {val:.3e}   log10 = {math.log10(val):.4f}")

print()
lepton_beta4 = 2 * math.log(substrates["2I_ord4"]["dist"][0]
                             / substrates["2I_ord4"]["dist"][2]) \
               / math.log(me / mtau)
lepton_beta5 = 2 * math.log(substrates["2I_ord5"]["dist"][0]
                             / substrates["2I_ord5"]["dist"][2]) \
               / math.log(me / mtau)

for key, sub in substrates.items():
    d1, d3 = sub["dist"][0], sub["dist"][2]
    lepton_b = lepton_beta4 if key == "2I_ord4" else lepton_beta5
    print(f"  {sub['label']}")
    for label, target in [("mu/mt", mu_q / mt_q)]:
        quark_b = 2 * math.log(d1 / d3) / math.log(target)
        ratio_b = quark_b / lepton_b
        print(f"    beta*(leptons) = {lepton_b:.4f}")
        print(f"    beta*(mu/mt)   = {quark_b:.4f}")
        print(f"    ratio beta*_quark/beta*_lepton = {ratio_b:.4f}")
    print()

print("  Note: ratio ~ 0.72 is substrate-independent (same for ord-4 and ord-5).")

# ============================================================
# 8. Regime boundaries (Section 6.3)
# ============================================================

print("\n" + "=" * 65)
print("8. REGIME BOUNDARIES (Section 6.3)")
print("=" * 65)

regimes = [
    ("Strong amplification (quarks)", 0.07, "< 1e-5"),
    ("Lepton window",                  0.09, "[1e-5, 1e-3]"),
    ("Lepton window upper",            0.13, ""),
    ("Weak amplification",             0.30, "> 1e-1"),
]
print()
for key, sub in substrates.items():
    d1, d3 = sub["dist"][0], sub["dist"][2]
    print(f"  {sub['label']}")
    for label, b, note in regimes:
        val = 10 ** ((2 / b) * math.log10(d1 / d3))
        print(f"    beta = {b:.2f}:  M1/M3 = {val:.3e}  ({label})")
    print()
