import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- Kesten-McKay support edges ---
# Support: [lambda_-(p), lambda_+(p)] = [1 - 2/sqrt(p), 1 + 2/sqrt(p)]
# Normalised Laplacian eigenvalue lambda in [0, 2]

def lam_minus(p):
    return 1.0 - 2.0 / np.sqrt(p)

def lam_plus(p):
    return 1.0 + 2.0 / np.sqrt(p)

# ADE eigenvalue levels from SpectralStratigraphy / O3
# lambda_1 = 5/6, lambda_2 = 1, lambda_3 = 5/4
ADE_levels = {
    r'$\lambda_1 = 5/6$': 5/6,
    r'$\lambda_2 = 1$': 1.0,
    r'$\lambda_3 = 5/4$': 5/4,
}

# Three representative p values
p_values = [5, 30, 120]
p_labels = [r'$p = 5$  (small valence)', r'$p = 30$', r'$p = 120$  (large valence)']
colors = ['#2166ac', '#4dac26', '#d6604d']
alphas = [0.25, 0.20, 0.15]

fig, ax = plt.subplots(figsize=(8.5, 4.2))

# Draw KM supports as horizontal bands
y_positions = [2.5, 1.5, 0.5]
height = 0.55

for i, (p, label, col, y, alph) in enumerate(
        zip(p_values, p_labels, colors, y_positions, alphas)):
    lo = lam_minus(p)
    hi = lam_plus(p)
    rect = mpatches.FancyBboxPatch(
        (lo, y - height/2), hi - lo, height,
        boxstyle='round,pad=0.01',
        facecolor=col, alpha=0.35, edgecolor=col, linewidth=1.5
    )
    ax.add_patch(rect)
    # Label the support interval
    ax.text(0.5*(lo + hi), y + height/2 + 0.08, label,
            ha='center', va='bottom', fontsize=9, color=col, fontweight='bold')
    # Annotate edges
    ax.annotate('', xy=(lo, y), xytext=(lo - 0.015, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
    ax.annotate('', xy=(hi, y), xytext=(hi + 0.015, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

# Draw ADE levels as vertical dashed lines
ade_colors = {'$\\lambda_1 = 5/6$': '#762a83',
              '$\\lambda_2 = 1$': '#1a1a1a',
              '$\\lambda_3 = 5/4$': '#bf812d'}
ade_styles = ['--', ':', '-.']

for (label, lval), sty in zip(ADE_levels.items(), ade_styles):
    col = ade_colors.get(label, 'black')
    ax.axvline(lval, color=col, linestyle=sty, linewidth=1.8, zorder=5)
    ax.text(lval, 3.3, label, ha='center', va='bottom',
            fontsize=9.5, color=col,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=col, alpha=0.85))

# Exit markers: lambda_3 exits first (small p), lambda_1 exits second
# Mark exit events with arrows
ax.annotate(r'$\lambda_3$ exits at $p^*_3 \approx 62$',
            xy=(5/4, 1.5), xytext=(1.45, 1.5),
            fontsize=8, ha='left', color='#bf812d',
            arrowprops=dict(arrowstyle='->', color='#bf812d', lw=1.0))
ax.annotate(r'$\lambda_1$ exits at $p^*_1 \approx 142$',
            xy=(5/6, 0.5), xytext=(0.45, 0.5),
            fontsize=8, ha='right', color='#762a83',
            arrowprops=dict(arrowstyle='->', color='#762a83', lw=1.0))
ax.annotate(r'$\lambda_2$ never exits',
            xy=(1.0, 0.5), xytext=(1.18, 0.18),
            fontsize=8, ha='left', color='#1a1a1a',
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

# Axis formatting
ax.set_xlim(0.4, 1.75)
ax.set_ylim(-0.2, 3.8)
ax.set_xlabel(r'Normalised Laplacian eigenvalue $\lambda$', fontsize=11)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.axhline(0, color='black', lw=0.5)

# Arrow indicating contraction direction
ax.annotate('', xy=(1.0, -0.05), xytext=(0.55, -0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.annotate('', xy=(1.45, -0.05), xytext=(1.0, -0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(1.0, -0.15, r'Support contracts as $p(n)\to\infty$',
        ha='center', va='top', fontsize=8.5, color='gray', style='italic')

ax.set_title(
    r'Kesten--McKay support contraction and ADE level exits'
    '\n'
    r'as effective valence $p(n)$ grows along the cascade',
    fontsize=10.5, pad=10
)

plt.tight_layout()
plt.savefig('fig1_km_contraction.pdf', bbox_inches='tight', dpi=200)
plt.savefig('fig1_km_contraction.png', bbox_inches='tight', dpi=200)
print("Figure 1 saved.")
