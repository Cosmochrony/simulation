#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
DATA_DIR = "../data/Planck/release_3/ancillary-data/cosmoparams/"
PLANCK_DATA_FILE = DATA_DIR + "COM_PowerSpect_CMB-TT-full_R3.01.txt"
LCDM_THEORY_FILE = DATA_DIR + "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

ELL_MIN = 2
ELL_MAX = 30
ELL_NORM_MIN = 25  # normalize curves using ℓ >= this value (within plotted range)

# Cosmochrony suppression parameters (shape only; amplitude fixed by normalization)
ELL_C = 10.0
ALPHA = 4.0

# Choose the "unsmoothed pattern" that Cosmochrony attenuates:
# - "planck": uses Planck multipole-by-multipole pattern (illustrative; preserves up/down trend)
# - "lcdm": uses ΛCDM theory multipole pattern (methodologically cleaner, but mostly smooth)
COSMO_PATTERN = "planck"  # <-- change to "lcdm" if you prefer


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------
def load_planck_txt(path: str) -> np.ndarray:
    """Loads whitespace-separated numeric rows from a Planck .txt file, skipping comments/headers."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No numeric data found in file: {path}")
    return np.asarray(rows, dtype=float)


def parse_planck_tt(data: np.ndarray):
    """
    Accepts:
      - 2 columns: ell, Dl
      - 3 columns: ell, Dl, sigma (symmetric)
      - >=4 columns: ell, Dl, err_low, err_high
    """
    ell = data[:, 0]
    Dl = data[:, 1]
    err_lo = err_hi = None

    if data.shape[1] == 3:
        err_lo = err_hi = np.abs(data[:, 2])
    elif data.shape[1] >= 4:
        err_lo = np.abs(data[:, 2])
        err_hi = np.abs(data[:, 3])

    return ell, Dl, err_lo, err_hi


def parse_lcdm_theory(data: np.ndarray):
    """
    For Planck *-minimum-theory*.txt, column 0 is ell, column 1 is typically Dl_TT.
    Extra columns may contain TE/EE/etc. We only use TT (2nd column).
    """
    ell = data[:, 0]
    Dl_tt = data[:, 1]
    return ell, Dl_tt


# ---------------------------------------------------------------------
# Cosmochrony suppression (regularized)
# ---------------------------------------------------------------------
def cosmochrony_suppression(ell, ell_c=ELL_C, alpha=ALPHA):
    ell = np.asarray(ell, dtype=float)
    # Smooth regularization at the very lowest multipoles
    return 1.0 / (1.0 + (ell_c / (ell + 1.0)) ** alpha)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def restrict_range(ell, *arrs, ell_min=ELL_MIN, ell_max=ELL_MAX):
    mask = (ell >= ell_min) & (ell <= ell_max)
    out = [ell[mask]]
    for a in arrs:
        out.append(a[mask] if a is not None else None)
    return out


def norm_to_data_high_ell(ell_data, Dl_data, ell_curve, Dl_curve, ell_norm_min=ELL_NORM_MIN):
    """Rescales Dl_curve so its mean over ell>=ell_norm_min matches data mean over same ell-range."""
    m_data = np.mean(Dl_data[ell_data >= ell_norm_min])
    m_curve = np.mean(Dl_curve[ell_curve >= ell_norm_min])
    return Dl_curve / m_curve * m_data


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # --- Load Planck data
    planck = load_planck_txt(PLANCK_DATA_FILE)
    ell_p, Dl_p, err_lo, err_hi = parse_planck_tt(planck)
    ell_p, Dl_p, err_lo, err_hi = restrict_range(ell_p, Dl_p, err_lo, err_hi)

    # --- Load ΛCDM theory
    lcdm = load_planck_txt(LCDM_THEORY_FILE)
    ell_l, Dl_l = parse_lcdm_theory(lcdm)
    ell_l, Dl_l = restrict_range(ell_l, Dl_l)

    # Normalize ΛCDM to Planck (high end of plotted range)
    Dl_lcdm = norm_to_data_high_ell(ell_p, Dl_p, ell_l, Dl_l, ELL_NORM_MIN)

    # --- Build Cosmochrony unsmoothed curve
    # Attenuation factor S(ell)
    S_p = cosmochrony_suppression(ell_p)

    if COSMO_PATTERN.lower() == "planck":
        # Illustrative "unsmoothed" Cosmochrony curve:
        # preserves multipole-by-multipole ups/downs by attenuating the observed Planck pattern.
        Dl_cosmo_raw = Dl_p * S_p

        # Normalize to the same high-ℓ reference as others (so amplitude is comparable)
        Dl_cosmo = norm_to_data_high_ell(ell_p, Dl_p, ell_p, Dl_cosmo_raw, ELL_NORM_MIN)

        cosmo_label = "Cosmochrony (unsmoothed; Planck-pattern attenuated)"
    elif COSMO_PATTERN.lower() == "lcdm":
        # Cleaner alternative:
        # attenuate the ΛCDM theory curve (still deterministic, but mostly smooth).
        # Interpolate ΛCDM onto Planck ell grid for a fair point-by-point product.
        Dl_lcdm_interp = np.interp(ell_p, ell_l, Dl_lcdm)
        Dl_cosmo_raw = Dl_lcdm_interp * S_p
        Dl_cosmo = norm_to_data_high_ell(ell_p, Dl_p, ell_p, Dl_cosmo_raw, ELL_NORM_MIN)

        cosmo_label = "Cosmochrony (unsmoothed; ΛCDM pattern attenuated)"
    else:
        raise ValueError("COSMO_PATTERN must be 'planck' or 'lcdm'")

    # --- Plot
    plt.figure(figsize=(7.5, 5))

    # Planck points
    if err_lo is not None:
        plt.errorbar(
            ell_p, Dl_p,
            yerr=[err_lo, err_hi],
            fmt="o",
            label="Planck 2018 TT",
            zorder=3
        )
    else:
        plt.plot(ell_p, Dl_p, "o", label="Planck 2018 TT", zorder=3)

    # ΛCDM theory (normalized)
    plt.plot(
        ell_l, Dl_lcdm,
        "--",
        linewidth=2,
        label=r"$\Lambda$CDM (Planck 2018 best-fit)",
        zorder=1
    )

    # Cosmochrony unsmoothed
    plt.plot(
        ell_p, Dl_cosmo,
        linewidth=2,
        label=cosmo_label,
        zorder=2
    )

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{TT}\;[\mu{\rm K}^2]$")
    plt.title(r"Low-$\ell$ CMB TT spectrum (Option A: Planck + $\Lambda$CDM + unsmoothed Cosmochrony)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cmb_lowell_optionA_planck_lcdm_cosmo_unsmoothed.pdf")
    plt.show()


if __name__ == "__main__":
    main()
