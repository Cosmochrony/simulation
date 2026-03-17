"""
Figure 1 for paper O1:
  Kesten-McKay support contraction and ADE level exit order.

The Kesten-McKay support edges for the normalised Laplacian of an LPS graph
X^{p,q} are

    lambda_pm(p) = 1 +/- 2*sqrt(p) / (p+1).

The three ADE eigenvalue levels for the 2I group (order-5 generating set) are

    lambda_1^ADE = 5/6,  lambda_2^ADE = 1,  lambda_3^ADE = 5/4.

Critical primes at which each outer level exits the support:

    p_3* = (4 + sqrt(15))^2 ~  62   (lambda_3^ADE exits upper edge)
    p_1* = (6 + sqrt(35))^2 ~ 142   (lambda_1^ADE exits lower edge)

Output: O1_fig1_KM_contraction.pdf  (and .png for draft use)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

p_vals = np.linspace(2, 220, 2000)

def lambda_plus(p):
    return 1.0 + 2.0 * np.sqrt(p) / (p + 1.0)

def lambda_minus(p):
    return 1.0 - 2.0 * np.sqrt(p) / (p + 1.0)

# ADE levels
L1_ADE = 5.0 / 6.0
L2_ADE = 1.0
L3_ADE = 5.0 / 4.0

# Critical primes
p3_star = (4.0 + np.sqrt(15.0))**2   # ~62
p1_star = (6.0 + np.sqrt(35.0))**2   # ~142

lp = lambda_plus(p_vals)
lm = lambda_minus(p_vals)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

# -- KM support band ---------------------------------------------------------
ax.fill_between(p_vals, lm, lp,
                color="#d0e8f5", alpha=0.7,
                label=r"Kesten--McKay support $[\lambda_-(p),\,\lambda_+(p)]$")
ax.plot(p_vals, lp, color="#1f77b4", lw=1.8)
ax.plot(p_vals, lm, color="#1f77b4", lw=1.8)

# -- ADE levels --------------------------------------------------------------
ax.axhline(L3_ADE, color="#d62728", lw=1.5, ls="--",
           label=r"$\lambda_3^{\mathrm{ADE}} = 5/4$")
ax.axhline(L2_ADE, color="#2ca02c", lw=1.5, ls="-.",
           label=r"$\lambda_2^{\mathrm{ADE}} = 1$  (never exits)")
ax.axhline(L1_ADE, color="#ff7f0e", lw=1.5, ls=":",
           label=r"$\lambda_1^{\mathrm{ADE}} = 5/6$")

# -- Critical prime markers --------------------------------------------------
ax.axvline(p3_star, color="#d62728", lw=1.0, alpha=0.55)
ax.axvline(p1_star, color="#ff7f0e", lw=1.0, alpha=0.55)

# Annotations
y_ann = 0.385
ax.annotate(r"$p_3^* \approx 62$",
            xy=(p3_star, y_ann), xytext=(p3_star + 6, y_ann - 0.025),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))
ax.annotate(r"$p_1^* \approx 142$",
            xy=(p1_star, y_ann), xytext=(p1_star + 6, y_ann - 0.025),
            fontsize=9, color="#ff7f0e",
            arrowprops=dict(arrowstyle="-", color="#ff7f0e", lw=0.8))

# Exit markers on support edges
ax.plot(p3_star, L3_ADE, "v", color="#d62728", ms=7, zorder=5)
ax.plot(p1_star, L1_ADE, "^", color="#ff7f0e", ms=7, zorder=5)

# -- Axes and labels ---------------------------------------------------------
ax.set_xlim(2, 220)
ax.set_ylim(0.35, 1.65)
ax.set_xlabel(r"LPS prime parameter $p$", fontsize=12)
ax.set_ylabel(r"Normalised Laplacian eigenvalue $\lambda$", fontsize=12)
ax.set_title(
    r"Kesten--McKay support contraction and ADE level exit order ($2I$, ord-5, $|S|=24$)",
    fontsize=11)

ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

# Secondary x-ticks at critical primes
ax.set_xticks(list(ax.get_xticks()) + [p3_star, p1_star])
ax.set_xticklabels(
    [str(int(t)) if t not in [p3_star, p1_star] else ""
     for t in ax.get_xticks()])

fig.tight_layout()
fig.savefig("O1_fig1_KM_contraction.pdf", dpi=300, bbox_inches="tight")
fig.savefig("O1_fig1_KM_contraction.png", dpi=200, bbox_inches="tight")
print("Figure 1 saved: O1_fig1_KM_contraction.pdf / .png")
