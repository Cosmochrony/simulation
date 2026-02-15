#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Density contrast range
    delta_rho = np.linspace(0.0, 0.5, 500)

    # Representative bounded values for f_max (no fitting)
    fmax_values = [0.05, 0.10]

    # Optional benchmark band (visual reference only)
    tension_low = 0.07
    tension_high = 0.11

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200)

    # Shaded benchmark band
    ax.axhspan(tension_low, tension_high, alpha=0.15,
               label="Observed tension band (benchmark)")

    # Deep-saturation relation: ΔH/H = δρ f_max
    for fmax in fmax_values:
        dH_over_H = delta_rho * fmax
        ax.plot(delta_rho, dH_over_H,
                label=f"$f_{{max}}={fmax:.2f}$")

    ax.set_xlabel(r"Void density contrast $\delta_\rho$")
    ax.set_ylabel(r"Fractional shift $\Delta H/H$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.14)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig("dH_over_H_vs_delta_rho.pdf")

if __name__ == "__main__":
    main()
