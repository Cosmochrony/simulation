#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-fit 2x2 grid test of the distinguishable-history capacity envelope
against Planck R3 low-ell spectra (TT, TE, EE).

Mechanism (low-ell capacity note): the largest angular modes probe the
first admissibility ranks, which contain very few distinguishable
projective histories (canonical Pell count, trajectory-branching note):

    N_hist(n) = ((1+sqrt(2))^(n+1) + (1-sqrt(2))^(n+1)) / 2
              = 1, 3, 7, 17, 41, 99, ...

Grid, parameter-free after the quadrupole anchoring n=1 <-> ell=2:

  Envelopes:     S_Bessel(N) = (N-1)/N        (finite-sample variance loss)
                 S_cap(N)    = N/(N+1)        (one-cell capacity completion)
  Dictionaries:  n_H(ell)   = 1 + ln(ell/2)              (H-dict, n = ln a)
                 n_rho(ell) = 1 + log_{1+sqrt2}(ell/2)   (Pell-rate)

Integer rank by nearest-integer projection (floor reported as robustness).
Comparison: D_ell^pred = D_ell^LCDM * S(ell) against Planck R3 data.
NO renormalisation, NO fitted constant, and the Planck points are never
multiplied by anything.

chi2 is computed on ell in [2, ELL_MAX] with symmetrised Planck errors;
the LCDM baseline (S=1) is reported alongside every cell.

Output: lowl_capacity_grid.pdf (TT panel + envelope panel),
        lowl_capacity_grid_pol.pdf (TE, EE panels),
        printed chi2 tables.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "Planck", "release_3",
                    "ancillary-data", "cosmoparams")
F_THEORY = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
F_TT = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
F_TE = "COM_PowerSpect_CMB-TE-full_R3.01.txt"
F_EE = "COM_PowerSpect_CMB-EE-full_R3.01.txt"

ELL_MIN, ELL_MAX = 2, 30
RHO = 1.0 + np.sqrt(2.0)


def n_hist(n):
    """Canonical Pell count of distinguishable histories at rank n."""
    n = np.asarray(n, dtype=float)
    return ((1 + np.sqrt(2)) ** (n + 1) + (1 - np.sqrt(2)) ** (n + 1)) / 2.0


def rank(ell, dictionary, mode="round"):
    """Integer admissibility rank probed by multipole ell (anchor n=1 <-> ell=2)."""
    ell = np.asarray(ell, dtype=float)
    if dictionary == "H":
        n = 1.0 + np.log(ell / 2.0)
    elif dictionary == "rho":
        n = 1.0 + np.log(ell / 2.0) / np.log(RHO)
    else:
        raise ValueError(dictionary)
    n = np.round(n) if mode == "round" else np.floor(n)
    return np.maximum(n, 0.0)


def envelope(N, form):
    N = np.asarray(N, dtype=float)
    if form == "Bessel":
        return (N - 1.0) / N
    if form == "cap":
        return N / (N + 1.0)
    raise ValueError(form)


def s_hist(ell, form, dictionary, mode="round"):
    return envelope(n_hist(rank(ell, dictionary, mode)), form)


def load_theory():
    d = np.loadtxt(os.path.join(DATA, F_THEORY))
    ell = d[:, 0].astype(int)
    return ell, {"TT": d[:, 1], "TE": d[:, 2], "EE": d[:, 3]}


def load_data(fname):
    d = np.loadtxt(os.path.join(DATA, fname))
    ell = d[:, 0].astype(int)
    dl = d[:, 1]
    err = 0.5 * (d[:, 2] + d[:, 3])   # symmetrised
    return ell, dl, err


def chi2(ell, dl, err, pred_ell, pred):
    m = (ell >= ELL_MIN) & (ell <= ELL_MAX)
    p = np.interp(ell[m], pred_ell, pred)
    return float(np.sum(((dl[m] - p) / err[m]) ** 2)), int(m.sum())


def main():
    th_ell, th = load_theory()
    channels = {"TT": F_TT, "TE": F_TE, "EE": F_EE}
    cells = [(f, d) for f in ("Bessel", "cap") for d in ("H", "rho")]

    print(f"ell range: [{ELL_MIN}, {ELL_MAX}]   anchoring n=1 <-> ell=2   "
          "integer rank: round (floor as robustness)")
    results = {}
    for ch, fname in channels.items():
        ell, dl, err = load_data(fname)
        base, npts = chi2(ell, dl, err, th_ell, th[ch])
        print(f"\n--- {ch}  (N={npts} points) ---")
        print(f"{'cell':>16} | {'chi2':>9} | {'chi2/N':>7} | {'round/floor':>11}")
        print(f"{'LCDM (S=1)':>16} | {base:9.2f} | {base/npts:7.2f} |")
        results[ch] = {"base": base, "npts": npts}
        for form, dic in cells:
            pred = th[ch] * s_hist(th_ell, form, dic, "round")
            c, _ = chi2(ell, dl, err, th_ell, pred)
            pred_f = th[ch] * s_hist(th_ell, form, dic, "floor")
            cf, _ = chi2(ell, dl, err, th_ell, pred_f)
            name = f"{form}+n_{dic}"
            results[ch][name] = c
            print(f"{name:>16} | {c:9.2f} | {c/npts:7.2f} | {cf:11.2f}")

    # TT figure
    ell, dl, err = load_data(F_TT)
    m = (ell >= ELL_MIN) & (ell <= ELL_MAX)
    mt = (th_ell >= ELL_MIN) & (th_ell <= ELL_MAX)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    ax.errorbar(ell[m], dl[m], yerr=err[m], fmt="o", ms=3.5, color="k",
                label="Planck 2018 TT", zorder=3)
    ax.plot(th_ell[mt], th["TT"][mt], "-",
            color="gray", lw=1.4, label=r"$\Lambda$CDM")
    styles = {("Bessel", "H"): ("C0", "-"), ("Bessel", "rho"): ("C0", "--"),
              ("cap", "H"): ("C3", "-"), ("cap", "rho"): ("C3", "--")}
    for (form, dic), (c, ls) in styles.items():
        pred = th["TT"] * s_hist(th_ell, form, dic, "round")
        ax.plot(th_ell[mt], pred[mt], ls, color=c, lw=1.2,
                label=rf"$\Lambda$CDM$\times S_{{\rm {form}}}(n_{{\rm {dic}}})$")
        ax2.plot(th_ell[mt], s_hist(th_ell, form, dic, "round")[mt], ls,
                 color=c, lw=1.2)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
    ax.legend(fontsize=7, ncol=2)
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$S_{\rm hist}(\ell)$")
    ax2.set_ylim(0.6, 1.02)
    ax2.axhline(1.0, color="gray", lw=0.6, ls=":")
    fig.suptitle("Distinguishable-history capacity envelopes vs Planck TT "
                 "(no fit, quadrupole-anchored)", fontsize=10)
    fig.tight_layout()
    fig.savefig("lowl_capacity_grid.pdf", bbox_inches="tight")
    print("\nSaved: lowl_capacity_grid.pdf")

    # Polarisation figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, ch in zip(axes, ("TE", "EE")):
        ell, dl, err = load_data(channels[ch])
        m = (ell >= ELL_MIN) & (ell <= ELL_MAX)
        ax.errorbar(ell[m], dl[m], yerr=err[m], fmt="o", ms=3, color="k",
                    label=f"Planck {ch}", zorder=3)
        ax.plot(th_ell[mt], th[ch][mt], "-", color="gray", lw=1.3,
                label=r"$\Lambda$CDM")
        for (form, dic), (c, ls) in styles.items():
            pred = th[ch] * s_hist(th_ell, form, dic, "round")
            ax.plot(th_ell[mt], pred[mt], ls, color=c, lw=1.0)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(rf"$D_\ell^{{{ch}}}$ [$\mu K^2$]")
        ax.legend(fontsize=7)
    fig.suptitle("Same no-fit envelopes propagated to TE and EE", fontsize=10)
    fig.tight_layout()
    fig.savefig("lowl_capacity_grid_pol.pdf", bbox_inches="tight")
    print("Saved: lowl_capacity_grid_pol.pdf")


if __name__ == "__main__":
    main()
