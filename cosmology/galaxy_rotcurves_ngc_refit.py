#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-fit of the classic bright spirals NGC 3198, NGC 2403, NGC 5055 with the
CURRENT global saturation scale (showcase methodology), replacing the legacy
white-paper-era calibration a0 = eta*c*H0 of the old galaxy_rotcurves_3panel
figure (which gave red-chi2 ~ 4.8 / 47.8 / 20.7).

Methodology aligned with the published showcase figure (Cosmology 2.0, Fig 1):
  * a_star fixed globally (default 2429 (km/s)^2/kpc, the full-sample robust
    value used in the showcase); no per-galaxy adjustment of a_star;
  * per-galaxy stellar mass-to-light ratio ups_disk fitted in [0.01, 2.0],
    ups_bul = 0.7 fixed;
  * law = simple_nu.

Usage:
  python galaxy_rotcurves_ngc_refit.py                    # a_star = 2429
  python galaxy_rotcurves_ngc_refit.py --a-star 3000
  python galaxy_rotcurves_ngc_refit.py --refit-global     # recompute a_star
                                                          # from the full
                                                          # sample first

Output: galaxy_rotcurves_ngc_refit.pdf + printed chi2 table.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from galaxy_rotcurves_3panel import (
  fit_global_a_star_robust,
  fit_ups_disk,
  list_rotmod_files,
  load_rotmod,
  vcosmo_from_components,
)

TARGETS = ["NGC3198", "NGC2403", "NGC5055"]
A_STAR_SHOWCASE = 2429.0   # (km/s)^2/kpc, full-sample robust value (Fig 1)
UPS_BUL = 0.7
LAW = "simple_nu"


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
  p.add_argument("--a-star", type=float, default=A_STAR_SHOWCASE)
  p.add_argument("--data-dir", default="../data/Rotmod_LTG/")
  p.add_argument("--refit-global", action="store_true",
                 help="recompute the global a_star from the full sample")
  p.add_argument("--out", default="galaxy_rotcurves_ngc_refit.pdf")
  args = p.parse_args()

  a_star = args.a_star
  if args.refit_global:
    files = list_rotmod_files(args.data_dir, limit=None)
    a_star, score = fit_global_a_star_robust(
      files=files, a_min=1500.0, a_max=8000.0, ngrid=120,
      ups_bul=UPS_BUL, law=LAW, score="median",
    )
    print(f"Global refit: a_star = {a_star:.1f} (km/s)^2/kpc"
          f"  (median score {score:.2f})")

  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
  print(f"\na_star = {a_star:.1f} (km/s)^2/kpc   ups_bul = {UPS_BUL}"
        f"   law = {LAW}")
  print(f"{'galaxy':>10} | {'ups_disk':>8} | {'red chi2':>8}")
  for ax, name in zip(axes, TARGETS):
    path = os.path.join(args.data_dir, f"{name}_rotmod.dat")
    r, vobs, ev, vgas, vdisk, vbul = load_rotmod(path)
    ups, red = fit_ups_disk(
      r, vobs, ev, vgas, vdisk, vbul,
      ups_bul=UPS_BUL, a_star=a_star, law=LAW,
      ups_min=0.01, ups_max=2.0,
    )
    vm = vcosmo_from_components(
      r, vgas, vdisk, vbul,
      ups_disk=ups, ups_bul=UPS_BUL, a_star=a_star, law=LAW,
    )
    print(f"{name:>10} | {ups:8.2f} | {red:8.2f}")
    ax.errorbar(r, vobs, yerr=ev, fmt="o", ms=3.5, lw=1, color="C1",
                label="Observed", zorder=2)
    ax.plot(r, vm, "-", color="C0", lw=1.6, label="Cosmochrony", zorder=3)
    ax.set_title(f"{name}\n"
                 rf"$\Upsilon_\star={ups:.2f}$, $\chi^2_\nu={red:.2f}$",
                 fontsize=10)
    ax.set_xlabel(r"$r$ [kpc]")
  axes[0].set_ylabel(r"$v$ [km/s]")
  axes[0].legend(fontsize=8, loc="lower right")
  fig.suptitle(
    rf"Classic spirals re-fit (global $a_\star={a_star:.0f}$ (km/s)$^2$/kpc,"
    " showcase methodology)", y=1.04, fontsize=11)
  fig.tight_layout()
  fig.savefig(args.out, dpi=200, bbox_inches="tight")
  print(f"Saved: {args.out}")


if __name__ == "__main__":
  main()
