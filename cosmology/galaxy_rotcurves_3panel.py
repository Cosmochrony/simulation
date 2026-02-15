#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


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
    if data.ndim != 2:
        raise ValueError(f"{filename}: expected 2D array, got shape {data.shape}")
    ncol = data.shape[1]
    if ncol < 6:
        raise ValueError(f"{filename}: expected at least 6 columns, got {ncol}")

    r = data[:, 0]
    vobs = data[:, 1]
    ev = data[:, 2]
    vgas = data[:, 3]
    vdisk = data[:, 4]
    vbul = data[:, 5]
    return r, vobs, ev, vgas, vdisk, vbul


# -----------------------------
# File listing
# -----------------------------
def list_rotmod_files(data_dir, limit=None, seed=0):
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No rotmod files found in: {data_dir}")

    if limit is None or limit >= len(files):
        return files

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=limit, replace=False)
    return [files[i] for i in sorted(idx)]


# -----------------------------
# Effective law (parameterized by a_star)
# -----------------------------
def geff_from_gN(gN, a_star, law="simple_nu"):
    """
    gN, a_star in (km/s)^2 / kpc.
    """
    if law == "simple_sqrt":
        return np.sqrt(gN * gN + a_star * gN)

    if law == "simple_nu":
        y = np.clip(gN / a_star, 1e-12, None)
        nu = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y)
        return nu * gN

    raise ValueError(f"Unknown law: {law}")


def vcosmo_from_components(
    r_kpc,
    vgas,
    vdisk,
    vbul,
    ups_disk=0.5,
    ups_bul=0.5,
    a_star=3700.0,
    law="simple_nu",
):
    """
    Units:
        r in kpc
        velocities in km/s
        gN in (km/s)^2/kpc
        a_star in (km/s)^2/kpc

    vbar^2 = vgas^2 + ups_disk*vdisk^2 + ups_bul*vbul^2
    """
    r = np.clip(r_kpc, 1e-6, None)

    vbar2 = vgas**2 + ups_disk * vdisk**2 + ups_bul * vbul**2
    gN = vbar2 / r
    geff = geff_from_gN(gN, a_star=a_star, law=law)
    return np.sqrt(r * geff)


# -----------------------------
# 1D fit for ups_disk (a_star fixed)
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
    law="simple_nu",
    ups_min=0.05,
    ups_max=1.5,
    ngrid=600,
):
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
            law=law,
        )
        chi2_vals[i] = np.sum(((vobs2 - vm) / ev2) ** 2)

    best_i = int(np.argmin(chi2_vals))
    best_ups = float(grid[best_i])

    dof = max(len(vobs2) - 1, 1)
    red_chi2 = float(chi2_vals[best_i] / dof)
    return best_ups, red_chi2


# -----------------------------
# Per-file evaluation
# -----------------------------
def redchi2_for_file(path, a_star, ups_bul=0.5, law="simple_nu"):
    r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)
    ups, red = fit_ups_disk(
        r,
        vobs,
        ev,
        vgas,
        vdisk,
        vbul,
        ups_bul=ups_bul,
        a_star=a_star,
        law=law,
    )
    filename = os.path.basename(path)
    name = filename.replace("_rotmod.dat", "")
    return name, ups, red, filename


# -----------------------------
# Robust global fit for a_star
# -----------------------------
def fit_global_a_star_robust(
    files,
    a_min=1500.0,
    a_max=8000.0,
    ngrid=120,
    ups_bul=0.5,
    law="simple_nu",
    score="median",
):
    a_grid = np.linspace(a_min, a_max, ngrid)
    scores = np.empty_like(a_grid)

    for i, a_star in enumerate(a_grid):
        vals = []
        for path in files:
            try:
                _, _, red, _ = redchi2_for_file(path, a_star=a_star, ups_bul=ups_bul, law=law)
            except Exception:
                continue
            if np.isfinite(red):
                vals.append(red)

        if not vals:
            scores[i] = np.inf
            continue

        v = np.array(vals, dtype=float)

        if score == "median":
            scores[i] = float(np.median(v))
        elif score == "trimmed_mean":
            v2 = np.sort(v)
            k = max(int(0.1 * len(v2)), 0)
            v2 = v2[k:len(v2) - k] if len(v2) - 2 * k >= 5 else v2
            scores[i] = float(np.mean(v2))
        else:
            raise ValueError(f"Unknown score: {score}")

    best_i = int(np.argmin(scores))
    return float(a_grid[best_i]), float(scores[best_i])


# -----------------------------
# Summary + boundary saturation stats
# -----------------------------
def summarize_sample(
    files,
    a_star,
    ups_bul=0.5,
    law="simple_nu",
    topk=10,
    ups_min=0.05,
    ups_max=1.5,
):
    rows = []
    skipped = 0

    for path in files:
        try:
            name, ups, red, filename = redchi2_for_file(path, a_star=a_star, ups_bul=ups_bul, law=law)
            rows.append((name, filename, ups, red))
        except Exception:
            skipped += 1

    rows = [r for r in rows if np.isfinite(r[3])]
    reds = np.array([r[3] for r in rows], dtype=float)
    upss = np.array([r[2] for r in rows], dtype=float)

    if len(rows) == 0:
        print("No valid galaxies in sample.")
        return rows

    med = float(np.median(reds))
    p16 = float(np.percentile(reds, 16))
    p84 = float(np.percentile(reds, 84))
    frac5 = float(np.mean(reds < 5.0))
    frac10 = float(np.mean(reds < 10.0))

    eps = 1e-9
    frac_ups_min = float(np.mean(upss <= (ups_min + eps)))
    frac_ups_max = float(np.mean(upss >= (ups_max - eps)))

    best = sorted(rows, key=lambda x: x[3])[:topk]
    worst = sorted(rows, key=lambda x: x[3], reverse=True)[:topk]

    print(f"Law: {law}")
    print(f"a_star = {a_star:.1f} (km/s)^2/kpc")
    print(f"N = {len(rows)} galaxies  (skipped={skipped})")
    print(f"red chi2 median = {med:.2f}   (P16={p16:.2f}, P84={p84:.2f})")
    print(f"fraction red chi2 < 5:  {frac5:.2%}")
    print(f"fraction red chi2 < 10: {frac10:.2%}")
    print(f"fraction ups at min ({ups_min}): {frac_ups_min:.2%}")
    print(f"fraction ups at max ({ups_max}): {frac_ups_max:.2%}")

    print("\nBest (lowest red chi2):")
    for name, filename, ups, red in best:
        print(f"  {name:18s}  ups={ups:5.2f}  redchi2={red:7.2f}  file={filename}")

    print("\nWorst (highest red chi2):")
    for name, filename, ups, red in worst:
        print(f"  {name:18s}  ups={ups:5.2f}  redchi2={red:7.2f}  file={filename}")

    return rows


# -----------------------------
# Plot (optional, uses a_star already fitted)
# -----------------------------
def plot_3panel(
    galaxies,
    data_dir=".",
    a_star=3700.0,
    ups_bul=0.5,
    law="simple_nu",
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
                law=law,
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
                law=law,
            )
            mask = ev > 0
            dof = max(int(np.sum(mask)) - 1, 1)
            redchi2 = float(np.sum(((vobs[mask] - vm[mask]) / ev[mask]) ** 2) / dof)

        vmodel = vcosmo_from_components(
            r,
            vgas,
            vdisk,
            vbul,
            ups_disk=ups_disk,
            ups_bul=ups_bul,
            a_star=a_star,
            law=law,
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


def run_robust_fit_suite(
    data_dir,
    limit=None,
    seed=1,
    a_min=1500.0,
    a_max=8000.0,
    ngrid=120,
    ups_bul=0.5,
    law="simple_nu",
    score="median",
    ups_ranges=(0.01, 2.0),
):
  files = list_rotmod_files(data_dir, limit=limit, seed=seed)

  for ups_min, ups_max in ups_ranges:
    # Local helper: same as summarize_sample but with custom ups bounds
    def redchi2_for_file_bounds(path, a_star):
      r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)
      ups, red = fit_ups_disk(
        r,
        vobs,
        ev,
        vgas,
        vdisk,
        vbul,
        ups_bul=ups_bul,
        a_star=a_star,
        law=law,
        ups_min=ups_min,
        ups_max=ups_max,
        ngrid=600,
      )
      filename = os.path.basename(path)
      name = filename.replace("_rotmod.dat", "")
      return name, filename, ups, red

    # Robust fit for a_star with bounds-aware chi2 evaluation
    a_grid = np.linspace(a_min, a_max, ngrid)
    scores = np.empty_like(a_grid)

    for i, a_star in enumerate(a_grid):
      vals = []
      for path in files:
        try:
          _, _, _, red = redchi2_for_file_bounds(path, a_star=a_star)
        except Exception:
          continue
        if np.isfinite(red):
          vals.append(red)

      if not vals:
        scores[i] = np.inf
        continue

      v = np.array(vals, dtype=float)
      if score == "median":
        scores[i] = float(np.median(v))
      elif score == "trimmed_mean":
        v2 = np.sort(v)
        k = max(int(0.1 * len(v2)), 0)
        v2 = v2[k:len(v2) - k] if len(v2) - 2 * k >= 5 else v2
        scores[i] = float(np.mean(v2))
      else:
        raise ValueError(f"Unknown score: {score}")

    best_i = int(np.argmin(scores))
    best_a = float(a_grid[best_i])
    best_score = float(scores[best_i])

    rows = []
    skipped = 0
    for path in files:
      try:
        name, filename, ups, red = redchi2_for_file_bounds(path, a_star=best_a)
        rows.append((name, filename, ups, red))
      except Exception:
        skipped += 1

    rows = [r for r in rows if np.isfinite(r[3])]
    reds = np.array([r[3] for r in rows], dtype=float)
    upss = np.array([r[2] for r in rows], dtype=float)

    med = float(np.median(reds))
    p16 = float(np.percentile(reds, 16))
    p84 = float(np.percentile(reds, 84))
    frac5 = float(np.mean(reds < 5.0))
    frac10 = float(np.mean(reds < 10.0))

    eps = 1e-9
    frac_ups_min = float(np.mean(upss <= (ups_min + eps)))
    frac_ups_max = float(np.mean(upss >= (ups_max - eps)))

    print("")
    print("=" * 72)
    print(f"Law: {law}   score: {score}")
    print(f"ups range: [{ups_min}, {ups_max}]   ups_bul={ups_bul}")
    print(f"Best global a_star (robust): {best_a:.1f}   score={best_score:.2f}")
    print(f"N = {len(rows)} galaxies  (skipped={skipped})")
    print(f"red chi2 median = {med:.2f}   (P16={p16:.2f}, P84={p84:.2f})")
    print(f"fraction red chi2 < 5:  {frac5:.2%}")
    print(f"fraction red chi2 < 10: {frac10:.2%}")
    print(f"fraction ups at min ({ups_min}): {frac_ups_min:.2%}")
    print(f"fraction ups at max ({ups_max}): {frac_ups_max:.2%}")
    print("=" * 72)

  return files

if __name__ == "__main__":
    data_dir = "../data/Rotmod_LTG/"

    run_robust_fit_suite(
        data_dir=data_dir,
        limit=80,
        seed=4,
        a_min=1500.0,
        a_max=8000.0,
        ngrid=120,
        ups_bul=0.5,
        law="simple_nu",
        score="trimmed_mean",
        ups_ranges=((0.05, 1.5), (0.01, 2.0)),
    )
