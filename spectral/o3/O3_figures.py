"""
O3 — Projective Dynamics and Mass Hierarchy
Figure generation: Fig. 1 (full range) and Fig. 2 (zoom + beta* annotation).

Requires: matplotlib, numpy (standard scientific Python stack).
Output: fig1_full.pdf, fig2_zoom.pdf  (also .png for preview)

Run from the same directory as O3_with_figures.tex so that pdflatex
finds the figures automatically.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Spectral data (consistent with O3_computations.py)
# -----------------------------------------------------------------------
r_ord4 = 3 / 5      # d1/d3 for 2I ord-4
r_ord5 = 2 / 3      # d1/d3 for 2I ord-5

# SM lepton ratios — PDG 2024 (MeV)
me, mmu, mtau = 0.51099895, 105.6583755, 1776.86
sm_etau  = math.log10(me / mtau)     # -3.541
sm_emu   = math.log10(me / mmu)      # -2.315
sm_mutau = math.log10(mmu / mtau)    # -1.225

# beta* values matching me/mtau exactly
beta4_etau = 2 * math.log10(r_ord4) / sm_etau   # 0.1253
beta5_etau = 2 * math.log10(r_ord5) / sm_etau   # 0.0995

# -----------------------------------------------------------------------
# Colour palette (consistent between both figures)
# -----------------------------------------------------------------------
COL4   = "#2c3e50"     # ord-4 curve (dark slate)
COL5   = "#2980b9"     # ord-5 curve (steel blue)
COL_13 = "#c0392b"     # me/mtau line (red dashed)
COL_12 = "#e67e22"     # me/mmu  line (orange dotted)
COL_23 = "#f39c12"     # mmu/mtau line (amber dotted)

# -----------------------------------------------------------------------
# Figure 1 — Full range beta in (0.04, 1.0)
# -----------------------------------------------------------------------
betas1 = np.linspace(0.04, 1.0, 600)
y4_1   = (2 / betas1) * np.log10(r_ord4)
y5_1   = (2 / betas1) * np.log10(r_ord5)

fig1, ax1 = plt.subplots(figsize=(7.5, 5.2))

# Regime shading (subtle, behind everything)
ymin1 = -5.5
ax1.axhspan(ymin1,          sm_etau - 0.05, alpha=0.06, color="steelblue", zorder=0)
ax1.axhspan(sm_etau + 0.05, sm_mutau - 0.05, alpha=0.06, color="seagreen",  zorder=0)
ax1.axhspan(sm_mutau + 0.05, 0.2,            alpha=0.06, color="goldenrod", zorder=0)

# SM horizontal reference lines
ax1.axhline(sm_etau,  color=COL_13, lw=1.3, ls="--", zorder=2)
ax1.axhline(sm_emu,   color=COL_12, lw=1.0, ls=":",  zorder=2)
ax1.axhline(sm_mutau, color=COL_23, lw=1.0, ls=":",  zorder=2)

# Theory curves
ax1.plot(betas1, y4_1, color=COL4, lw=2.0,
         label=r"$2I$, ord-4 ($d_1/d_3 = 3/5$)", zorder=3)
ax1.plot(betas1, y5_1, color=COL5, lw=2.0, ls=(0, (5, 2)),
         label=r"$2I$, ord-5 ($d_1/d_3 = 2/3$)", zorder=3)

# SM line labels (right of axes)
offset = 1.012   # axes-fraction x coordinate
for y_val, label, col in [
    (sm_etau,  r"$m_e/m_\tau$",   COL_13),
    (sm_emu,   r"$m_e/m_\mu$",    COL_12),
    (sm_mutau, r"$m_\mu/m_\tau$", COL_23),
]:
    ax1.annotate(label, xy=(offset, y_val),
                 xycoords=("axes fraction", "data"),
                 fontsize=8.5, color=col, va="center")

# Regime zone labels (right side, inside axes)
zone_x = 0.97   # axes-fraction
ax1.text(zone_x, (ymin1 + sm_etau) / 2 - 0.2,
         "quarks\n(strong ampl.)",
         fontsize=7.5, color="steelblue", alpha=0.85,
         ha="right", transform=ax1.get_yaxis_transform())
ax1.text(zone_x, (sm_etau + sm_mutau) / 2,
         "lepton\nwindow",
         fontsize=7.5, color="seagreen", alpha=0.85,
         ha="right", transform=ax1.get_yaxis_transform())
ax1.text(zone_x, (sm_mutau + 0.15) / 2 + 0.15,
         "static LPS\nrange",
         fontsize=7.5, color="goldenrod", alpha=0.85,
         ha="right", transform=ax1.get_yaxis_transform())

ax1.set_xlabel(r"Growth exponent $\beta$", fontsize=11)
ax1.set_ylabel(r"$\log_{10}(M_1/M_3)$", fontsize=11)
ax1.set_title(
    r"Mass ratio $M_1/M_3$ as a function of the growth exponent $\beta$",
    fontsize=10.5, pad=8)
ax1.set_xlim(0.04, 1.0)
ax1.set_ylim(ymin1, 0.2)
ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
ax1.grid(True, alpha=0.25, lw=0.7)
ax1.tick_params(labelsize=9)

fig1.tight_layout()
fig1.savefig("fig1_full.pdf", bbox_inches="tight")
fig1.savefig("fig1_full.png", dpi=180, bbox_inches="tight")
plt.close(fig1)
print("Saved: fig1_full.pdf / fig1_full.png")

# -----------------------------------------------------------------------
# Figure 2 — Zoom beta in (0.05, 0.20) with beta* annotation
# -----------------------------------------------------------------------
betas2 = np.linspace(0.05, 0.20, 600)
y4_2   = (2 / betas2) * np.log10(r_ord4)
y5_2   = (2 / betas2) * np.log10(r_ord5)

fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))

# Compatibility window shading
ax2.axvspan(0.09, 0.13, alpha=0.12, color="seagreen", zorder=0,
            label=r"Compatibility window $\beta \in [0.09,\,0.13]$")

# SM reference lines
ax2.axhline(sm_etau, color=COL_13, lw=1.4, ls="--", zorder=2,
            label=r"$m_e/m_\tau \approx 2.88\times10^{-4}$")
ax2.axhline(sm_emu,  color=COL_12, lw=1.0, ls=":",  zorder=2,
            label=r"$m_e/m_\mu \approx 4.84\times10^{-3}$")

# Theory curves
ax2.plot(betas2, y4_2, color=COL4, lw=2.2,
         label=r"$2I$, ord-4 ($d_1/d_3 = 3/5$)", zorder=3)
ax2.plot(betas2, y5_2, color=COL5, lw=2.2, ls=(0, (5, 2)),
         label=r"$2I$, ord-5 ($d_1/d_3 = 2/3$)", zorder=3)

# Vertical guides to beta* markers
for b, col in [(beta4_etau, COL4), (beta5_etau, COL5)]:
    ax2.axvline(b, color=col, lw=0.9, ls=":", alpha=0.65, zorder=2)

# beta* intersection markers
ax2.scatter([beta4_etau], [sm_etau], s=55, color=COL4, zorder=5)
ax2.scatter([beta5_etau], [sm_etau], s=55, color=COL5, zorder=5)

# Annotations with arrows
ax2.annotate(
    r"$\beta^* \approx 0.125$" + "\n(ord-4)",
    xy=(beta4_etau, sm_etau),
    xytext=(beta4_etau + 0.013, sm_etau - 0.38),
    fontsize=8.2, color=COL4,
    arrowprops=dict(arrowstyle="->", color=COL4, lw=0.8))

ax2.annotate(
    r"$\beta^* \approx 0.100$" + "\n(ord-5)",
    xy=(beta5_etau, sm_etau),
    xytext=(beta5_etau - 0.036, sm_etau - 0.58),
    fontsize=8.2, color=COL5,
    arrowprops=dict(arrowstyle="->", color=COL5, lw=0.8))

# SM labels (right of axes)
for y_val, label, col in [
    (sm_etau, r"$m_e/m_\tau$", COL_13),
    (sm_emu,  r"$m_e/m_\mu$",  COL_12),
]:
    ax2.annotate(label, xy=(0.206, y_val),
                 xycoords=("data", "data"),
                 fontsize=8.5, color=col, va="center")

ax2.set_xlabel(r"Growth exponent $\beta$", fontsize=11)
ax2.set_ylabel(r"$\log_{10}(M_1/M_3)$", fontsize=11)
ax2.set_title(
    r"Zoom: compatibility window $\beta \in [0.09,\,0.13]$",
    fontsize=10.5, pad=8)
ax2.set_xlim(0.05, 0.20)
ax2.set_ylim(-5.5, -1.5)
ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax2.grid(True, alpha=0.25, lw=0.7)
ax2.tick_params(labelsize=9)

fig2.tight_layout()
fig2.savefig("fig2_zoom.pdf", bbox_inches="tight")
fig2.savefig("fig2_zoom.png", dpi=180, bbox_inches="tight")
plt.close(fig2)
print("Saved: fig2_zoom.pdf / fig2_zoom.png")
