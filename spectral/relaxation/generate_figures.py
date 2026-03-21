"""
generate_figures.py
===================
Generates all figures for SpectralRelaxation 1.0.

Figures produced:
  fig1_km_density.pdf       -- Kesten-McKay density for selected p values
  fig2_km_cdf.pdf           -- Kesten-McKay CDF with ADE levels marked
  fig3_reading_A_ratios.pdf -- Reading A mass ratios vs p (all three cases)
  fig4_reading_B_logM.pdf   -- Reading B log10(M_i) values
  fig5_O1_support.pdf       -- O1: support narrowing and ADE level exit order
                               (main figure used in the paper)

Usage:
  python generate_figures.py

All PDFs are saved in the current directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectral.relaxation.spectral_relaxation_lib import (
  lam_support, km_density, km_cdf, normalised_levels,
  reading_A, reading_B, ADE_CASES, SM_RATIOS
)

# Global style
plt.rcParams.update({
    'font.size':      10,
    'axes.titlesize': 10.5,
    'axes.labelsize': 10.5,
    'legend.fontsize': 8.5,
    'figure.dpi':     150,
})

COLORS = {
    'l1': '#2166ac',   # blue
    'l2': '#4dac26',   # green
    'l3': '#d6604d',   # red
    'supp': 'steelblue',
}

P_CASES = [5, 13, 29, 53]   # for multi-p plots


# ===========================================================
# Figure 1: KM density for several p values
# ===========================================================

def fig1_km_density():
    fig, ax = plt.subplots(figsize=(7, 4))
    p_list  = [5, 13, 29, 53]
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(p_list)))

    lam_plot = np.linspace(0.0, 2.0, 2000)

    for p, col in zip(p_list, palette):
        lm, lp = lam_support(p)
        rho = np.array([km_density(lv, p) for lv in lam_plot])
        ax.plot(lam_plot, rho, color=col, lw=1.8,
                label=f'$p={p}$  ($d={p+1}$)')
        ax.axvline(lm, color=col, lw=0.7, ls=':', alpha=0.6)
        ax.axvline(lp, color=col, lw=0.7, ls=':', alpha=0.6)

    ax.axvline(1.0, color='grey', lw=0.8, ls='--', alpha=0.5, label='$\\lambda=1$')
    ax.set_xlabel(r'Normalised Laplacian eigenvalue $\lambda$')
    ax.set_ylabel(r'Kesten--McKay density $\rho_{\mathrm{KM}}(\lambda;p)$')
    ax.set_title('Kesten--McKay spectral density for various $p$')
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, None)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('fig1_km_density.pdf', bbox_inches='tight')
    plt.close()
    print("fig1_km_density.pdf saved.")


# ===========================================================
# Figure 2: KM CDF with ADE levels marked
# ===========================================================

def fig2_km_cdf():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cases_to_show = [("2I_ord4", 29), ("2I_ord5", 53)]

    for ax, (case_key, p) in zip(axes, cases_to_show):
        case   = ADE_CASES[case_key]
        norms  = normalised_levels(case_key)
        dims   = case["dims"]
        lm, lp = lam_support(p)

        lam_plot = np.linspace(max(0.0, lm - 0.02), min(2.0, lp + 0.02), 800)
        cdf_plot = np.array([km_cdf(lv, p) for lv in lam_plot])

        ax.plot(lam_plot, cdf_plot, color='steelblue', lw=2.0,
                label=r'$F_{\mathrm{KM}}(\lambda)$')
        ax.axvline(lm, color='steelblue', lw=0.8, ls='--', alpha=0.5)
        ax.axvline(lp, color='steelblue', lw=0.8, ls='--', alpha=0.5)

        colors_lev = [COLORS['l1'], COLORS['l2'], COLORS['l3']]
        for i, (lv, col) in enumerate(zip(norms, colors_lev)):
            ci = km_cdf(lv, p)
            ax.axvline(lv, color=col, lw=1.5, ls='-.',
                       label=fr'$\lambda_{i+1}={lv:.3f}$,  $c_{i+1}={ci:.3f}$')
            ax.axhline(ci, color=col, lw=0.8, ls=':', alpha=0.7)
            ax.plot(lv, ci, 'o', color=col, ms=5)

        ax.axhline(0.5, color='grey', lw=0.7, ls='--', alpha=0.5)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$F_{\mathrm{KM}}(\lambda;p)$')
        ax.set_title(f'{case["label"]},  $p={p}$')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig('fig2_km_cdf.pdf', bbox_inches='tight')
    plt.close()
    print("fig2_km_cdf.pdf saved.")


# ===========================================================
# Figure 3: Reading A mass ratios vs p
# ===========================================================

def fig3_reading_A():
    case_keys = ["2T_ord3", "2I_ord4", "2I_ord5"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    p_fine = list(range(3, 210, 2))   # dense p grid (primes not required for plot)

    for ax, case_key in zip(axes, case_keys):
        case = ADE_CASES[case_key]
        r12_list, r23_list, r13_list, p_valid = [], [], [], []

        for p in p_fine:
            res = reading_A(case_key, p)
            if res is not None:
                r12_list.append(res["r12"])
                r23_list.append(res["r23"])
                r13_list.append(res["r13"])
                p_valid.append(p)

        ax.semilogy(p_valid, r12_list, color=COLORS['l1'], lw=1.8,
                    label=r'$\mathcal{M}_1/\mathcal{M}_2$')
        ax.semilogy(p_valid, r23_list, color=COLORS['l2'], lw=1.8,
                    label=r'$\mathcal{M}_2/\mathcal{M}_3$')
        ax.semilogy(p_valid, r13_list, color=COLORS['l3'], lw=1.8,
                    label=r'$\mathcal{M}_1/\mathcal{M}_3$')
        ax.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.5)

        # SM reference lines
        sm = SM_RATIOS['charged_leptons']
        ax.axhline(sm['r12'], color='grey', lw=0.6, ls=':', alpha=0.6)
        ax.axhline(sm['r13'], color='grey', lw=0.6, ls=':', alpha=0.6)

        ax.set_xlabel('$p$')
        ax.set_ylabel('Mass ratio (Reading A)')
        ax.set_title(case['label'])
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.2)
        ax.set_xlim(3, max(p_valid) + 5)
        ax.set_ylim(1e-2, 10)

    plt.tight_layout()
    plt.savefig('fig3_reading_A_ratios.pdf', bbox_inches='tight')
    plt.close()
    print("fig3_reading_A_ratios.pdf saved.")


# ===========================================================
# Figure 4: Reading B log10(M_i) vs p
# ===========================================================

def fig4_reading_B():
    case_keys = ["2T_ord3", "2I_ord4", "2I_ord5"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    p_fine = list(range(3, 210, 2))

    for ax, case_key in zip(axes, case_keys):
        case = ADE_CASES[case_key]
        lm1, lm2, lm3, p_valid = [], [], [], []

        for p in p_fine:
            res = reading_B(case_key, p)
            if res is not None:
                lm = res["log10_M"]
                lm1.append(lm[0])
                lm2.append(lm[1])
                lm3.append(lm[2])
                p_valid.append(p)

        ax.plot(p_valid, lm1, color=COLORS['l1'], lw=1.8,
                label=r'$\log_{10}\mathcal{M}_1$')
        ax.plot(p_valid, lm2, color=COLORS['l2'], lw=1.8,
                label=r'$\log_{10}\mathcal{M}_2$')
        ax.plot(p_valid, lm3, color=COLORS['l3'], lw=1.8,
                label=r'$\log_{10}\mathcal{M}_3$')

        # SM reference
        lep = SM_RATIOS['charged_leptons']
        # Show observed log10(m_e/m_tau) ~ -3.5 as a span
        ax.axhspan(-3.6, 0.0, alpha=0.05, color='orange',
                   label='SM range (leptons)')

        ax.set_xlabel('$p$')
        ax.set_ylabel(r'$\log_{10}(\mathcal{M}_i)$  (Reading B)')
        ax.set_title(case['label'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(3, max(p_valid) + 5)

    plt.tight_layout()
    plt.savefig('fig4_reading_B_logM.pdf', bbox_inches='tight')
    plt.close()
    print("fig4_reading_B_logM.pdf saved.")


# ===========================================================
# Figure 5: O1 support narrowing with ADE level exit order
# (main paper figure)
# ===========================================================

def fig5_O1_support():
    """
    Main figure for the O1 discussion.
    Shows the Kesten-McKay support [lambda_-, lambda_+] as p varies,
    with the three 2I ord-5 ADE levels as horizontal lines.
    Exact exit-p values are annotated.
    """
    p_cont   = np.linspace(3.0, 200.0, 600)
    lam_minus = 1.0 - 2.0 * np.sqrt(p_cont) / (p_cont + 1.0)
    lam_plus  = 1.0 + 2.0 * np.sqrt(p_cont) / (p_cont + 1.0)

    # ADE levels for 2I ord-5
    levels = [20/24, 24/24, 30/24]
    labels_lev = [
        r'$\lambda_1 = 5/6 \approx 0.833$',
        r'$\lambda_2 = 1$',
        r'$\lambda_3 = 5/4 = 1.250$',
    ]
    colors_lev = [COLORS['l1'], COLORS['l2'], COLORS['l3']]

    # Exact exit-p values
    p3_exit = (4.0 + np.sqrt(15.0))**2   # ~62
    p1_exit = (6.0 + np.sqrt(35.0))**2   # ~142

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Support band
    ax.fill_between(p_cont, lam_minus, lam_plus,
                    alpha=0.13, color=COLORS['supp'])
    ax.plot(p_cont, lam_minus, color=COLORS['supp'], lw=1.5, ls='--')
    ax.plot(p_cont, lam_plus,  color=COLORS['supp'], lw=1.5, ls='--',
            label=r'$\lambda_\pm = 1 \pm 2\sqrt{p}/(p+1)$')
    ax.text(5.0, 1.80,
            r'KM support $[\lambda_-, \lambda_+]$',
            fontsize=8.5, color=COLORS['supp'], alpha=0.85)

    # ADE levels
    for lv, lb, col in zip(levels, labels_lev, colors_lev):
        ax.axhline(lv, color=col, lw=1.8, ls='-.', label=lb)

    # Exit verticals
    ax.axvline(p3_exit, color=colors_lev[2], lw=1.0, ls=':', alpha=0.8)
    ax.axvline(p1_exit, color=colors_lev[0], lw=1.0, ls=':', alpha=0.8)

    # Annotations with exact values
    ax.annotate(
        r'$\lambda_3$ exits' '\n'
        r'$p = (4+\sqrt{15})^2 \approx 62$',
        xy=(p3_exit, levels[2]),
        xytext=(p3_exit + 7, 1.46),
        fontsize=7.5, color=colors_lev[2],
        arrowprops=dict(arrowstyle='->', color=colors_lev[2], lw=0.9)
    )
    ax.annotate(
        r'$\lambda_1$ exits' '\n'
        r'$p = (6+\sqrt{35})^2 \approx 142$',
        xy=(p1_exit, levels[0]),
        xytext=(p1_exit - 70, 0.58),
        fontsize=7.5, color=colors_lev[0],
        arrowprops=dict(arrowstyle='->', color=colors_lev[0], lw=0.9)
    )

    # Exit order summary box
    ax.text(
        138, 0.20,
        'Exit order:\n'
        r'$\lambda_3$ first $\Rightarrow$ heaviest' + '\n'
        r'$\lambda_1$ last  $\Rightarrow$ lightest',
        fontsize=7.5, color='#333333',
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#aaaaaa', lw=0.8)
    )

    ax.set_xlabel(r'Prime $p$', fontsize=11)
    ax.set_ylabel(r'Normalised eigenvalue $\lambda$', fontsize=11)
    ax.set_title(
        r'Support narrowing and ADE level exit order '
        r'($2I$, ord-$5$, $|S|=24$)',
        fontsize=10.5
    )
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_xlim(3, 200)
    ax.set_ylim(0.08, 1.93)
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    plt.savefig('fig5_O1_support.pdf', bbox_inches='tight')
    plt.close()
    print("fig5_O1_support.pdf saved.")


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    print("Generating figures for SpectralRelaxation 1.0...")
    fig1_km_density()
    fig2_km_cdf()
    fig3_reading_A()
    fig4_reading_B()
    fig5_O1_support()
    print("\nAll figures saved.")
