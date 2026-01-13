#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# I/O: robust SPARC rotmod loader
# -----------------------------
def load_rotmod(filename):
    """
    Expected columns (SPARC rotmod):
    Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    Some files may omit SB columns; we keep robust parsing.
    Returns: r[kpc], vobs[km/s], ev[km/s], vgas[km/s], vdisk[km/s], vbul[km/s]
    """
    data = np.loadtxt(filename, comments="#")
    ncol = data.shape[1]
    if ncol < 6:
        raise ValueError(f"{filename}: expected at least 6 columns, got {ncol}")

    r     = data[:, 0]
    vobs  = data[:, 1]
    ev    = data[:, 2]
    vgas  = data[:, 3]
    vdisk = data[:, 4]
    vbul  = data[:, 5]
    return r, vobs, ev, vgas, vdisk, vbul


# -----------------------------
# Cosmochrony effective law
# -----------------------------
def vcosmo_from_components(
    r_kpc,
    vgas,
    vdisk,
    vbul,
    ups_disk=0.5,
    ups_bul=0.5,
    eta=0.15,
    H0_km_s_Mpc=70.0,
):
    """
    Effective Cosmochrony / saturation-projective law, MOND-like interpolation:
        g_eff = sqrt(gN^2 + a0*gN),   a0 = eta * c * H0
    with gN = vbar^2 / r.

    Units:
        r in kpc
        velocities in km/s
        H0 in km/s/Mpc -> convert to km/s/kpc
        a0 ends up in (km/s)^2 / kpc (same as v^2/r)

    vbar^2 = vgas^2 + ups_disk*vdisk^2 + ups_bul*vbul^2
    """
    # Avoid division at r=0
    r = np.clip(r_kpc, 1e-6, None)

    vbar2 = vgas**2 + ups_disk * vdisk**2 + ups_bul * vbul**2
    gN = vbar2 / r  # (km/s)^2 / kpc

    # H0 in km/s/kpc
    H0_km_s_kpc = H0_km_s_Mpc / 1000.0
    c_km_s = 299792.458

    a0 = eta * c_km_s * H0_km_s_kpc  # (km/s)^2 / kpc

    geff = np.sqrt(gN * gN + a0 * gN)
    v_model = np.sqrt(r * geff)
    return v_model


# -----------------------------
# Simple 1D fit for ups_disk (optional)
# -----------------------------
def fit_ups_disk(r, vobs, ev, vgas, vdisk, vbul, ups_bul=0.5, eta=0.15, H0=70.0,
                 ups_min=0.05, ups_max=1.5, ngrid=800):
    """
    Grid-search fit for ups_disk only. Robust and deterministic.
    Returns best ups_disk and reduced chi2.
    """
    # Ignore points with non-positive errors (rare, but safe)
    mask = (ev > 0) & np.isfinite(ev) & np.isfinite(vobs) & np.isfinite(r)
    r2, vobs2, ev2 = r[mask], vobs[mask], ev[mask]

    grid = np.linspace(ups_min, ups_max, ngrid)
    chi2_vals = np.empty_like(grid)

    for i, ups in enumerate(grid):
        vm = vcosmo_from_components(r2, vgas[mask], vdisk[mask], vbul[mask],
                                    ups_disk=ups, ups_bul=ups_bul, eta=eta, H0_km_s_Mpc=H0)
        chi2 = np.sum(((vobs2 - vm) / ev2) ** 2)
        chi2_vals[i] = chi2

    best_i = int(np.argmin(chi2_vals))
    best_ups = float(grid[best_i])

    dof = max(len(vobs2) - 1, 1)
    red_chi2 = float(chi2_vals[best_i] / dof)
    return best_ups, red_chi2


# -----------------------------
# Plot 3-panel figure
# -----------------------------
def plot_3panel(
    galaxies,
    data_dir=".",
    eta=0.15,
    H0=70.0,
    ups_bul=0.5,
    fit_ups=True,
    outfile="cosmochrony_rotcurves_3panel.png",
):
    """
    galaxies: list of dicts with keys:
        - name: display label
        - file: rotmod filename (relative to data_dir)
        - ups_disk: optional fixed value if fit_ups=False
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, g in zip(axes, galaxies):
        path = f"{data_dir.rstrip('/')}/{g['file']}"
        r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)

        if fit_ups:
            ups_disk, redchi2 = fit_ups_disk(
                r, vobs, ev, vgas, vdisk, vbul,
                ups_bul=ups_bul, eta=eta, H0=H0
            )
        else:
            ups_disk = float(g.get("ups_disk", 0.5))
            # Compute reduced chi2 for reporting
            vm = vcosmo_from_components(r, vgas, vdisk, vbul,
                                        ups_disk=ups_disk, ups_bul=ups_bul, eta=eta, H0_km_s_Mpc=H0)
            mask = (ev > 0)
            dof = max(np.sum(mask) - 1, 1)
            redchi2 = float(np.sum(((vobs[mask] - vm[mask]) / ev[mask]) ** 2) / dof)

        vmodel = vcosmo_from_components(
            r, vgas, vdisk, vbul,
            ups_disk=ups_disk, ups_bul=ups_bul,
            eta=eta, H0_km_s_Mpc=H0,
        )

        # Plot
        ax.plot(r, vmodel, label="Cosmochrony")
        ax.errorbar(r, vobs, yerr=ev, fmt="o", ms=4.5, capsize=2.0, label="Observed")

        ax.set_title(f"{g['name']}\n$\\Upsilon_\\star={ups_disk:.2f}$, $\\chi^2_\\nu={redchi2:.2f}$")
        ax.set_xlabel(r"$r$ [kpc]")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$v$ [km/s]")

    # Shared legend: put it once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.08))

    fig.suptitle(rf"Effective Cosmochrony Rotation Curves ($a_0=\eta cH_0$, $\eta={eta}$, $H_0={H0}$ km/s/Mpc)",
                 y=1.16)
    fig.tight_layout()

    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    # === EDIT HERE: choose your 3 galaxies ===
    # Put the correct rotmod filenames you have locally.
    galaxies = [
        {"name": "NGC 3198", "file": "NGC3198_rotmod.dat"},
        {"name": "NGC 2403", "file": "NGC2403_rotmod.dat"},
        {"name": "NGC 5055", "file": "NGC5055_rotmod.dat"},
    ]

    plot_3panel(
        galaxies=galaxies,
        data_dir="data/Rotmod_LTG/",     # change if your rotmod files are in another folder
        eta=0.15,
        H0=70.0,
        ups_bul=0.5,
        fit_ups=True,     # set False if you prefer fixed values
        outfile="cosmochrony_rotcurves_3panel.pdf",
    )
