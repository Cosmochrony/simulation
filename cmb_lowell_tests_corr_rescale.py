#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Inputs (place these two files next to this script)
# ---------------------------------------------------------------------
DATA_DIR = "data/Planck/release_3/ancillary-data/cosmoparams/"
PLANCK_DATA_FILE = DATA_DIR + "COM_PowerSpect_CMB-TT-full_R3.01.txt"
LCDM_THEORY_FILE = DATA_DIR + "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

ELL_MIN = 2
ELL_MAX = 30
ELL_NORM_MIN = 25  # normalization range within plotted range

# Cosmochrony attenuation parameters
ELL_C = 10.0
ALPHA = 4.0

# Tests configuration
N_PERM = 50_000          # permutation count for p-values (increase if you want)
GAMMA_GRID = np.linspace(0.6, 1.4, 161)  # γ for ℓ' = ℓ^γ

# Output files
FIG_RESCALED = "cmb_lowell_rescaled_gamma_fit.pdf"


# ---------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------
def load_planck_txt(path: str) -> np.ndarray:
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
    Planck TT full typically: ell, Dl, err_low, err_high (or symmetric)
    Accepts:
      - 2 cols: ell, Dl
      - 3 cols: ell, Dl, sigma
      - >=4 cols: ell, Dl, err_low, err_high
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
    In Planck *-minimum-theory*.txt, col0=ell, col1=Dl_TT usually.
    Extra cols may exist; we only use TT (col1).
    """
    return data[:, 0], data[:, 1]


def restrict_range(ell, *arrs, ell_min=ELL_MIN, ell_max=ELL_MAX):
    m = (ell >= ell_min) & (ell <= ell_max)
    out = [ell[m]]
    for a in arrs:
        out.append(a[m] if a is not None else None)
    return out


def norm_to_data_high_ell(ell_data, Dl_data, ell_curve, Dl_curve, ell_norm_min=ELL_NORM_MIN):
    m_data = float(np.mean(Dl_data[ell_data >= ell_norm_min]))
    m_curve = float(np.mean(Dl_curve[ell_curve >= ell_norm_min]))
    return Dl_curve / m_curve * m_data


# ---------------------------------------------------------------------
# Cosmochrony attenuation (regularized)
# ---------------------------------------------------------------------
def cosmochrony_suppression(ell, ell_c=ELL_C, alpha=ALPHA):
    ell = np.asarray(ell, dtype=float)
    return 1.0 / (1.0 + (ell_c / (ell + 1.0)) ** alpha)


# ---------------------------------------------------------------------
# Stats helpers (no scipy)
# ---------------------------------------------------------------------
def pearson_r(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def ranks(a):
    """Average ranks for ties (simple, good enough for small arrays)."""
    a = np.asarray(a)
    order = np.argsort(a)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            r[order[i:j + 1]] = avg
        i = j + 1
    return r


def spearman_rho(x, y):
    rx = ranks(x)
    ry = ranks(y)
    return pearson_r(rx, ry)


def perm_pvalue_corr(x, y, corr_fn, n_perm=N_PERM, seed=0):
    """
    Two-sided permutation test for correlation.
    Null: x and y are unrelated; permute y.
    """
    rng = np.random.default_rng(seed)
    obs = corr_fn(x, y)
    if not np.isfinite(obs):
        return obs, np.nan
    count = 0
    y = np.asarray(y)
    for _ in range(n_perm):
        yp = rng.permutation(y)
        c = corr_fn(x, yp)
        if np.abs(c) >= np.abs(obs):
            count += 1
    # add-one smoothing
    p = (count + 1) / (n_perm + 1)
    return obs, p


def dlog(D):
    D = np.asarray(D, float)
    return np.diff(np.log(np.maximum(D, 1e-12)))


# ---------------------------------------------------------------------
# Rescaling test ℓ' = ℓ^γ + best amplitude
# ---------------------------------------------------------------------
def best_amp(y, yref):
    # Minimise ||y - A*yref||^2 => A = (y·yref)/(yref·yref)
    denom = float(np.dot(yref, yref))
    if denom == 0:
        return np.nan
    return float(np.dot(y, yref) / denom)


def rmse(y, yfit):
    y = np.asarray(y, float)
    yfit = np.asarray(yfit, float)
    return float(np.sqrt(np.mean((y - yfit) ** 2)))


def find_best_gamma(ell, y_target, ell_ref, y_ref, gamma_grid):
    """
    Fit y_target ≈ A * y_ref( ell^γ ) by interpolation.
    Returns best {gamma, A, rmse}.
    """
    best = None
    for gamma in gamma_grid:
        ell_prime = ell ** gamma
        # interpolate reference curve at ell'
        y_interp = np.interp(ell_prime, ell_ref, y_ref)
        A = best_amp(y_target, y_interp)
        if not np.isfinite(A):
            continue
        y_fit = A * y_interp
        e = rmse(y_target, y_fit)
        if best is None or e < best["rmse"]:
            best = {"gamma": float(gamma), "A": float(A), "rmse": float(e), "y_fit": y_fit, "ell_prime": ell_prime}
    return best


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # --- Load Planck (data)
    p = load_planck_txt(PLANCK_DATA_FILE)
    ell_p, Dl_p, err_lo, err_hi = parse_planck_tt(p)
    ell_p, Dl_p, err_lo, err_hi = restrict_range(ell_p, Dl_p, err_lo, err_hi)

    # --- Load ΛCDM theory
    t = load_planck_txt(LCDM_THEORY_FILE)
    ell_l, Dl_l = parse_lcdm_theory(t)
    ell_l, Dl_l = restrict_range(ell_l, Dl_l)

    # --- Normalize ΛCDM to Planck at high-ℓ end of plotted range
    Dl_lcdm = norm_to_data_high_ell(ell_p, Dl_p, ell_l, Dl_l, ELL_NORM_MIN)

    # --- Build Cosmo (pattern = lcdm): Dl_cosmo = A * S(ell) * Dl_lcdm(ell)
    # Interpolate lcdm onto Planck ell grid, apply suppression, then normalize to Planck high-ℓ range
    Dl_lcdm_interp = np.interp(ell_p, ell_l, Dl_lcdm)
    S = cosmochrony_suppression(ell_p)
    Dl_cosmo_raw = Dl_lcdm_interp * S
    Dl_cosmo = norm_to_data_high_ell(ell_p, Dl_p, ell_p, Dl_cosmo_raw, ELL_NORM_MIN)

    # -----------------------------------------------------------------
    # 1) Correlation test on derivatives (Δ log Dℓ)
    # -----------------------------------------------------------------
    x = dlog(Dl_p)       # Planck
    y = dlog(Dl_cosmo)   # Cosmo (lcdm-pattern attenuated)

    rho_s, p_s = perm_pvalue_corr(x, y, spearman_rho, n_perm=N_PERM, seed=1)
    rho_p, p_p = perm_pvalue_corr(x, y, pearson_r, n_perm=N_PERM, seed=2)

    print("=== Correlation on Δ log Dℓ (ℓ={}..{}) ===".format(ELL_MIN, ELL_MAX))
    print("Spearman rho =", rho_s, "perm p =", p_s)
    print("Pearson  r   =", rho_p, "perm p =", p_p)
    print("(Note: using COSMO_PATTERN='lcdm'; if you use 'planck', this becomes trivial.)")
    print()

    # -----------------------------------------------------------------
    # 2) Rescale ℓ for Cosmo to match Planck: Planck(ℓ') ≈ A^{-1} Cosmo(ℓ), ℓ' = ℓ^γ
    # Here we fit Cosmo as target, Planck as reference (or vice versa).
    # We'll fit: Dl_cosmo(ℓ) ≈ A * Dl_planck(ℓ^γ)
    # -----------------------------------------------------------------
    best = find_best_gamma(
      ell_p,  # ℓ grid
      Dl_p,  # target = Planck
      ell_p,  # reference ℓ for Cosmo
      Dl_cosmo,  # reference curve = Cosmo
      GAMMA_GRID
    )
    # Baseline (no rescale): fit amplitude only
    A0 = best_amp(Dl_cosmo, Dl_p)
    fit0 = A0 * Dl_p
    rmse0 = rmse(Dl_cosmo, fit0)

    print("=== ℓ-rescaling fit: Dl_planck(ℓ) ≈ A * Dl_cosmo(ℓ^γ) ===")
    print("Baseline (γ=1): A =", A0, "RMSE =", rmse0)
    if best is None:
        print("No valid γ found.")
        return
    print("Best γ =", best["gamma"], "A =", best["A"], "RMSE =", best["rmse"])
    improvement = (rmse0 - best["rmse"]) / rmse0 if rmse0 > 0 else np.nan
    print("RMSE improvement vs γ=1:", improvement)
    print()

    # -----------------------------------------------------------------
    # Plot rescaled overlay (annex figure)
    # -----------------------------------------------------------------
    plt.figure(figsize=(7.5, 5))

    # Planck points
    if err_lo is not None:
        plt.errorbar(
            ell_p, Dl_p, yerr=[err_lo, err_hi],
            fmt="o", label="Planck 2018 TT", zorder=3
        )
    else:
        plt.plot(ell_p, Dl_p, "o", label="Planck 2018 TT", zorder=3)

    # Cosmo curve (lcdm pattern attenuated)
    plt.plot(ell_p, Dl_cosmo, linewidth=2, label="Cosmochrony (lcdm-pattern attenuated)", zorder=2)

    # Best rescaled Planck curve (A * Planck(ℓ^γ)) evaluated on ell_p grid
    Dl_cosmo_rescaled = np.interp(best["ell_prime"], ell_p, Dl_cosmo)
    Dl_cosmo_rescaled_fit = best["A"] * Dl_cosmo_rescaled

    plt.plot(
      ell_p, Dl_cosmo_rescaled_fit,
      "--", linewidth=2,
      label=rf"Rescaled Cosmo: $A\,D_{{\ell^{{\gamma}}}}$ (γ={best['gamma']:.2f})",
      zorder=1
    )

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{TT}\;[\mu{\rm K}^2]$")
    plt.title(r"Low-$\ell$ rescaling test: $D_\ell^{\rm Cosmo}\approx A\,D_{\ell^\gamma}^{\rm Planck}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_RESCALED)
    plt.show()

    print(f"Saved rescaling figure: {FIG_RESCALED}")


if __name__ == "__main__":
    main()
