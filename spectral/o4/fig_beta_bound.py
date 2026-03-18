import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = np.linspace(1, 300, 2000)
n0 = 10.0

def plaw(beta):
    return (n / n0) ** beta * n0

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Excluded region: above beta = 1
ax.fill_between(n, plaw(1.0), 220,
                color='#d73027', alpha=0.09)
ax.text(255, 130, r'Excluded', color='#d73027', fontsize=9, style='italic',
        ha='center')
ax.text(255, 95, r'$(\beta > 1)$', color='#d73027', fontsize=9, style='italic',
        ha='center')

# Phenomenological window band
ax.fill_between(n, plaw(0.09), plaw(0.13),
                color='#1b7837', alpha=0.20)

# Curves: excluded
excl_curves = [
    (3.0,   '#b2182b', '-',  r'$p(n)\sim n^{3}$'),
    (2.0,   '#d6604d', '--', r'$p(n)\sim n^{2}$'),
    (1.5,   '#f4a582', ':',  r'$p(n)\sim n^{3/2}$'),
]
for beta, col, ls, lab in excl_curves:
    ax.plot(n, plaw(beta), color=col, linestyle=ls, linewidth=1.3,
            label=lab, zorder=3)

# Structural bound beta = 1
ax.plot(n, plaw(1.0), color='#333333', linestyle='-', linewidth=2.4,
        label=r'$p(n)\sim n$  (bound, $\beta=1$)', zorder=5)

# Admitted curves
adm_curves = [
    (0.5,   '#4393c3', '--', r'$p(n)\sim n^{1/2}$'),
    (0.13,  '#74add1', ':',  r'$p(n)\sim n^{0.13}$  (marginal)'),
    (0.125, '#1b7837', '-',  r'$p(n)\sim n^{0.125}$'),
    (0.10,  '#00441b', '--', r'$p(n)\sim n^{0.10}$'),
]
for beta, col, ls, lab in adm_curves:
    ax.plot(n, plaw(beta), color=col, linestyle=ls, linewidth=1.8,
            label=lab, zorder=4)

# Structural bound annotation — placed to the right, outside legend area
ax.annotate(r'Structural bound $\beta\leq 1$',
            xy=(280, plaw(1.0)[np.argmin(np.abs(n - 280))]),
            xytext=(190, 42),
            fontsize=9, color='#333333',
            arrowprops=dict(arrowstyle='->', color='#555', lw=0.9),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#aaa', alpha=0.9))

# Phenomenological window annotation — bottom right, clear area
ax.text(295, 1.42, r'$\beta^*\in(0.09,0.13)$',
        color='#1b7837', fontsize=9, style='italic', ha='right', va='bottom')

# Axes
ax.set_xlabel(r'Cascade rank $n$', fontsize=11)
ax.set_ylabel(r'Effective valence $p(n)$ (normalised)', fontsize=11)
ax.set_xlim(1, 300)
ax.set_ylim(0.7, 220)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])

ax.set_title(
    r'Valence growth laws: structural bound and phenomenological window',
    fontsize=11, pad=10
)

# Legend — top left, two columns
ax.legend(fontsize=8.0, loc='upper left', ncol=2,
          framealpha=0.95, edgecolor='#cccccc',
          handlelength=2.0, labelspacing=0.4)

plt.tight_layout()
plt.savefig('fig2_beta_bound.pdf', bbox_inches='tight', dpi=200)
plt.savefig('fig2_beta_bound.png', bbox_inches='tight', dpi=200)
print("Figure 2 saved.")
