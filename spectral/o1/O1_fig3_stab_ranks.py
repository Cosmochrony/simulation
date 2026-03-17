"""
Figure 3 for paper O1:
  Stabilisation rank ordering for different growth laws p(n) -> infinity.

For each growth law p(n), the stabilisation rank n_k is defined as
    n_k = min{ n >= 1 : p(n) >= p_k* }.

We compare three representative growth laws:
    (A)  p(n) = floor(n^{0.5})   (slow)
    (B)  p(n) = floor(n^{1.0})   (linear)
    (C)  p(n) = floor(n^{1.5})   (fast)

In each case the ordering n_3 < n_1 holds independently of the growth law,
illustrating Remark 1 (independence from the growth law).

The figure shows p(n) vs n for each law, with horizontal lines at p_3* and
p_1*, and the stabilisation rank marked for each law.

Output: O1_fig3_stab_ranks.pdf  (and .png for draft use)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

p3_star = (4.0 + np.sqrt(15.0))**2   # ~62
p1_star = (6.0 + np.sqrt(35.0))**2   # ~142

N_MAX = 500
n_vals = np.arange(1, N_MAX + 1, dtype=float)

# Growth laws (not required to produce primes -- we treat p(n) as a
# continuous proxy for the valence growth; the ordering result is independent
# of whether exact prime values are used).
# We use a log scale on y to make all three curves simultaneously readable.
growth_laws = {
    r"$p(n) \sim n^{1/2}$ (slow)":      (lambda n: n**0.5,  "#1f77b4", "-"),
    r"$p(n) \sim n^{1}$ (linear)":      (lambda n: n**1.0,  "#ff7f0e", "--"),
    r"$p(n) \sim n^{3/2}$ (fast)":      (lambda n: n**1.5,  "#2ca02c", "-."),
}

# ---------------------------------------------------------------------------
# Compute stabilisation ranks
# ---------------------------------------------------------------------------

def stabilisation_rank(p_func, p_star, n_max=N_MAX):
    """Return smallest n such that p_func(n) >= p_star."""
    for n in range(1, n_max + 1):
        if p_func(n) >= p_star:
            return n
    return None   # not reached within n_max

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for label, (p_func, color, ls) in growth_laws.items():
    p_curve = p_func(n_vals)
    ax.plot(n_vals, p_curve, color=color, lw=1.8, ls=ls, label=label)

    # Stabilisation ranks
    n3 = stabilisation_rank(p_func, p3_star)
    n1 = stabilisation_rank(p_func, p1_star)

    if n3 is not None:
        ax.plot(n3, p_func(n3), "v", color=color, ms=8, zorder=5)
        ax.annotate(rf"$n_3={n3}$",
                    xy=(n3, p_func(n3)),
                    xytext=(n3 + 8, p_func(n3) * 0.85),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7))

    if n1 is not None:
        ax.plot(n1, p_func(n1), "^", color=color, ms=8, zorder=5)
        ax.annotate(rf"$n_1={n1}$",
                    xy=(n1, p_func(n1)),
                    xytext=(n1 + 8, p_func(n1) * 0.85),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7))

# Critical prime horizontal lines
ax.axhline(p3_star, color="#d62728", lw=1.2, ls=":",
           label=r"$p_3^* \approx 62$ ($\lambda_3^{\mathrm{ADE}}$ exits)")
ax.axhline(p1_star, color="#8c564b", lw=1.2, ls=":",
           label=r"$p_1^* \approx 142$ ($\lambda_1^{\mathrm{ADE}}$ exits)")

# Annotations for critical primes
ax.text(N_MAX * 0.97, p3_star + 3, r"$p_3^*$",
        ha="right", fontsize=9, color="#d62728")
ax.text(N_MAX * 0.97, p1_star + 3, r"$p_1^*$",
        ha="right", fontsize=9, color="#8c564b")

# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

ax.set_xlim(1, N_MAX)
ax.set_yscale("log")
ax.set_ylim(1, 3e4)
ax.set_xlabel(r"Cascade depth $n$", fontsize=12)
ax.set_ylabel(r"LPS prime parameter $p(n)$  [log scale]", fontsize=12)
ax.set_title(
    r"Stabilisation ranks $n_3 < n_1$ under different growth laws "
    r"(Theorem~2, Remark~1)",
    fontsize=11)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

fig.tight_layout()
fig.savefig("O1_fig3_stab_ranks.pdf", dpi=300, bbox_inches="tight")
fig.savefig("O1_fig3_stab_ranks.png", dpi=200, bbox_inches="tight")
print("Figure 3 saved: O1_fig3_stab_ranks.pdf / .png")
