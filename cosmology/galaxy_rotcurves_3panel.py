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
        a_star=3700.0,
):
    """
    Effective saturation law (MOND-like interpolation, but parameterized directly by a_star):
        g_eff = sqrt(gN^2 + a_star*gN)
    with gN = vbar^2 / r.

    Units:
        r in kpc
        velocities in km/s
        gN in (km/s)^2/kpc
        a_star in (km/s)^2/kpc

    vbar^2 = vgas^2 + ups_disk*vdisk^2 + ups_bul*vbul^2
    """
    r = np.clip(r_kpc, 1e-6, None)

    vbar2 = vgas**2 + ups_disk * vdisk**2 + ups_bul * vbul**2
    gN = vbar2 / r  # (km/s)^2 / kpc

    geff = np.sqrt(gN * gN + a_star * gN)
    v_model = np.sqrt(r * geff)
    return v_model


# -----------------------------
# Simple 1D fit for ups_disk (optional)
# -----------------------------
def fit_ups_disk(
    r,
    vobs,
    ev,
    vgas,
    vdisk,
    vbul,
    ups_bul=0.5,
    a_star=3700.0,
    ups_min=0.05,
    ups_max=1.5,
    ngrid=800,
):
  """
  Grid-search fit for ups_disk only (a_star fixed).
  Returns best ups_disk and reduced chi2.
  """
  mask = (ev > 0) & np.isfinite(ev) & np.isfinite(vobs) & np.isfinite(r)
  r2, vobs2, ev2 = r[mask], vobs[mask], ev[mask]

  grid = np.linspace(ups_min, ups_max, ngrid)
  chi2_vals = np.empty_like(grid)

  for i, ups in enumerate(grid):
    vm = vcosmo_from_components(
      r2,
      vgas[mask],
      vdisk[mask],
      vbul[mask],
      ups_disk=ups,
      ups_bul=ups_bul,
      a_star=a_star,
    )
    chi2_vals[i] = np.sum(((vobs2 - vm) / ev2) ** 2)

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
        a_star=3700.0,
        ups_bul=0.5,
        fit_ups=True,
        outfile="cosmochrony_rotcurves_3panel.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, g in zip(axes, galaxies):
        path = f"{data_dir.rstrip('/')}/{g['file']}"
        r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)

        if fit_ups:
            ups_disk, redchi2 = fit_ups_disk(
                r,
                vobs,
                ev,
                vgas,
                vdisk,
                vbul,
                ups_bul=ups_bul,
                a_star=a_star,
            )
        else:
            ups_disk = float(g.get("ups_disk", 0.5))
            vm = vcosmo_from_components(
                r,
                vgas,
                vdisk,
                vbul,
                ups_disk=ups_disk,
                ups_bul=ups_bul,
                a_star=a_star,
            )
            mask = (ev > 0)
            dof = max(np.sum(mask) - 1, 1)
            redchi2 = float(np.sum(((vobs[mask] - vm[mask]) / ev[mask]) ** 2) / dof)

        vmodel = vcosmo_from_components(
            r,
            vgas,
            vdisk,
            vbul,
            ups_disk=ups_disk,
            ups_bul=ups_bul,
            a_star=a_star,
        )

        ax.plot(r, vmodel, label="Cosmochrony")
        ax.errorbar(r, vobs, yerr=ev, fmt="o", ms=4.5, capsize=2.0, label="Observed")

        ax.set_title(f"{g['name']}\n$\\Upsilon_\\star={ups_disk:.2f}$, $\\chi^2_\\nu={redchi2:.2f}$")
        ax.set_xlabel(r"$r$ [kpc]")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$v$ [km/s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.suptitle(
        rf"Effective Rotation Curves (global fit: $a_\star={a_star:.0f}$ $(\mathrm{{km/s}})^2/\mathrm{{kpc}}$)",
        y=1.16,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {outfile}")


def galaxy_redchi2_for_astar(
    r, vobs, ev, vgas, vdisk, vbul, a_star, ups_bul=0.5, ups_min=0.05, ups_max=1.5
):
  ups_disk, redchi2 = fit_ups_disk(
    r, vobs, ev, vgas, vdisk, vbul, ups_bul=ups_bul, a_star=a_star, ups_min=ups_min, ups_max=ups_max
  )
  return redchi2, ups_disk


def fit_global_a_star(
    galaxies, data_dir=".", ups_bul=0.5, a_min=1500.0, a_max=8000.0, ngrid=120
):
  a_grid = np.linspace(a_min, a_max, ngrid)
  score = np.empty_like(a_grid)

  for i, a_star in enumerate(a_grid):
    total = 0.0
    count = 0
    for g in galaxies:
      path = f"{data_dir.rstrip('/')}/{g['file']}"
      r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)
      redchi2, _ = galaxy_redchi2_for_astar(
        r, vobs, ev, vgas, vdisk, vbul, a_star=a_star, ups_bul=ups_bul
      )
      total += redchi2
      count += 1
    score[i] = total / max(count, 1)

  best_i = int(np.argmin(score))
  return float(a_grid[best_i]), float(score[best_i])


def a_ms2_to_km2s2_per_kpc(a_ms2):
  kpc_m = 3.085677581491367e19
  return a_ms2 * (kpc_m / 1e6)

if __name__ == "__main__":
    galaxies = [
        {"name": "NGC 3198", "file": "NGC3198_rotmod.dat"},
        {"name": "NGC 2403", "file": "NGC2403_rotmod.dat"},
        {"name": "NGC 5055", "file": "NGC5055_rotmod.dat"},
    ]

    best_a, best_score = fit_global_a_star(
        galaxies=galaxies,
        data_dir="../data/Rotmod_LTG/",
        ups_bul=0.5,
        a_min=1500.0,
        a_max=8000.0,
        ngrid=140,
    )
    print(f"Best global a_star: {best_a:.1f}  (mean red chi2: {best_score:.2f})")

    plot_3panel(
        galaxies=galaxies,
        data_dir="../data/Rotmod_LTG/",
        a_star=best_a,
        ups_bul=0.5,
        fit_ups=True,
        outfile="galaxy_rotcurves_3panel.pdf",
    )
