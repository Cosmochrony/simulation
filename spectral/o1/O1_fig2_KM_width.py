"""
Figure 2 for paper O1:
  Kesten-McKay window width  width_KM(p) = 4*sqrt(p)/(p+1)
  and its asymptotic rate 4/sqrt(p).

This is the direct visualisation of Proposition 1 (KM contraction):
the support width goes to zero at rate 4/sqrt(p) as p -> infinity.

Output: O1_fig2_KM_width.pdf  (and .png for draft use)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Use actual prime values for p for exactness, supplemented by a dense grid
# for the smooth curves.
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes_1mod4 = [p for p in range(5, 500) if is_prime(p) and p % 4 == 1]

p_dense = np.linspace(5, 500, 5000)

def width_KM(p):
    return 4.0 * np.sqrt(p) / (p + 1.0)

def width_asymp(p):
    return 4.0 / np.sqrt(p)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))

# Exact curve
ax.plot(p_dense, width_KM(p_dense),
        color="#1f77b4", lw=2.0,
        label=r"$\mathrm{width}_{\mathrm{KM}}(p) = \dfrac{4\sqrt{p}}{p+1}$")

# Asymptotic rate
ax.plot(p_dense, width_asymp(p_dense),
        color="#d62728", lw=1.4, ls="--",
        label=r"Asymptotic rate $\dfrac{4}{\sqrt{p}}$")

# Scatter at actual p = 1 mod 4 primes
ax.scatter(primes_1mod4, [width_KM(p) for p in primes_1mod4],
           s=18, color="#1f77b4", zorder=4, alpha=0.7,
           label=r"Admissible primes $p \equiv 1\ (\mathrm{mod}\ 4)$")

# Critical prime markers
p3_star = (4.0 + np.sqrt(15.0))**2
p1_star = (6.0 + np.sqrt(35.0))**2

for pstar, label, color in [
        (p3_star, r"$p_3^* \approx 62$", "#e377c2"),
        (p1_star, r"$p_1^* \approx 142$", "#8c564b")]:
    ax.axvline(pstar, color=color, lw=1.0, ls=":", alpha=0.8)
    ax.text(pstar + 4, 0.28, label, color=color, fontsize=8.5, va="center")

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

ax.set_xlim(5, 500)
ax.set_ylim(0, 0.72)
ax.set_xlabel(r"LPS prime parameter $p$", fontsize=12)
ax.set_ylabel(r"$\mathrm{width}_{\mathrm{KM}}(p)$", fontsize=12)
ax.set_title(r"Kesten--McKay window width and asymptotic rate (Proposition~1)",
             fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

fig.tight_layout()
fig.savefig("O1_fig2_KM_width.pdf", dpi=300, bbox_inches="tight")
fig.savefig("O1_fig2_KM_width.png", dpi=200, bbox_inches="tight")
print("Figure 2 saved: O1_fig2_KM_width.pdf / .png")
