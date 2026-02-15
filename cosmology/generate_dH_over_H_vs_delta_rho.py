#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Density contrast range
    delta_rho = np.linspace(0.0, 0.5, 500)

    # Representative bounded values for f_max (no fitting)
    fmax_values = [0.05, 0.10]

    # Representative late-time fractional offset (orientation only)
    tension_low = 0.07
    tension_high = 0.11

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200)

    # Subtle neutral benchmark band (background layer)
    ax.axhspan(
        tension_low,
        tension_high,
        color="0.85",        # light grey
        alpha=0.2,          # very subtle
        zorder=0,
        label="Representative late-time fractional offset (orientation only)"
    )

    # Deep-saturation relation: ΔH/H = δρ f_max
    for fmax in fmax_values:
        dH_over_H = delta_rho * fmax
        ax.plot(
            delta_rho,
            dH_over_H,
            linewidth=1.8,
            label=f"$f_{{\\max}}={fmax:.2f}$"
        )

    ax.set_xlabel(r"Void density contrast $\delta_\rho$")
    ax.set_ylabel(r"Fractional shift $\Delta H/H$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.14)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig("dH_over_H_vs_delta_rho.pdf")

if __name__ == "__main__":
    main()
