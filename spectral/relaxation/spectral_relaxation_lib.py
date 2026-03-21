"""
spectral_relaxation_lib.py
==========================
Core numerical library for SpectralRelaxation 1.0
(Cosmochrony / Bounded Admissibility Theory -- Admissibility Sub-Programme, Step 5)

Provides:
  - Kesten-McKay spectral density and CDF for the normalised Laplacian
    of (p+1)-regular graphs
  - LPS graph support edges lambda_pm(p)
  - ADE case definitions (from SpectralStratigraphy, Proposition 3)
  - Reading A mass ratios: M_i propto c_i / dim_rho_i
  - Reading B mass ratios: M_i propto c_i^{dim_rho_i}
  - Exact analytical exit-p formulae for ADE levels leaving the KM support
  - Support narrowing table for O1 analysis

Convention throughout: NORMALISED graph Laplacian L = I - A/d,
eigenvalues in [0, 2].  For d-regular graph: lambda(L) = 1 - mu(A)/d.
"""

import numpy as np
from scipy import integrate


# ===========================================================
# Kesten-McKay measure (normalised Laplacian)
# ===========================================================

def lam_support(p):
    """
    Kesten-McKay support edges for the normalised Laplacian of a
    (p+1)-regular graph.

    Returns (lambda_minus, lambda_plus) with
        lambda_pm = 1 -+ 2*sqrt(p) / (p+1).

    The support is symmetric about lambda = 1.
    lambda_minus coincides with the Ramanujan lower bound on lambda_2(L).
    """
    d = p + 1
    delta = 2.0 * np.sqrt(float(p)) / d
    return 1.0 - delta, 1.0 + delta


def km_density(lam, p):
    """
    Kesten-McKay spectral density rho_KM(lambda; p) for the normalised
    Laplacian of a (p+1)-regular graph.

    Derivation:
      rho_A(mu) = d * sqrt(4(d-1) - mu^2) / (2*pi*(d^2 - mu^2))
      is the KM density for the adjacency matrix eigenvalue mu.
      Under the change of variable lambda = 1 - mu/d, the Jacobian is d,
      so rho_L(lambda) = rho_A(d*(1-lambda)) * d.

    Returns 0 outside the support [lambda_minus, lambda_plus].
    """
    d = p + 1
    mu = d * (1.0 - lam)
    inner = 4.0 * (d - 1) - mu * mu
    denom = d * d - mu * mu
    if inner <= 0.0 or denom <= 0.0 or abs(mu) >= d:
        return 0.0
    return d * np.sqrt(inner) / (2.0 * np.pi * denom) * d


def km_cdf(lam, p, epsabs=1e-12, epsrel=1e-12):
    """
    Kesten-McKay cumulative distribution function F_KM(lambda; p).

    F_KM(lambda) = integral_{lambda_-}^{min(lambda, lambda_+)} rho_KM(x;p) dx.

    Exact values:
      F_KM(lambda_-)  = 0
      F_KM(1)         = 1/2  (by symmetry rho_KM(lambda) = rho_KM(2-lambda))
      F_KM(lambda_+)  = 1
    """
    lam_minus, lam_plus = lam_support(p)
    if lam <= lam_minus:
        return 0.0
    lam_c = min(lam, lam_plus)
    val, _ = integrate.quad(
        lambda x: km_density(x, p),
        lam_minus, lam_c,
        limit=400, epsabs=epsabs, epsrel=epsrel
    )
    return val


def km_norm_check(p):
    """Verify KM density integrates to 1. Should return ~1.0."""
    lam_minus, lam_plus = lam_support(p)
    val, _ = integrate.quad(
        lambda x: km_density(x, p),
        lam_minus, lam_plus,
        limit=400
    )
    return val


# ===========================================================
# ADE case definitions (SpectralStratigraphy, Proposition 3)
# ===========================================================
#
# Keys: case identifier string
# Values: dict with
#   label        : LaTeX label for tables/figures
#   group        : binary polyhedral group
#   gen_order    : order of generators in S
#   S            : |S| = cardinality of the generating set
#   lambda_comb  : combinatorial Cayley eigenvalues [lambda_1, lambda_2, lambda_3]
#   dims         : multiplicities (= sum of squares of irrep dimensions in block)
#   rep_content  : description of irreps in each block
#
# Normalised eigenvalue: lambda_i_norm = lambda_i_comb / S
# (because L = I - A_Cayley / S for valency S)

ADE_CASES = {
    "2T_ord3": {
        "label":       r"$2T$, ord-$3$, $|S|=8$",
        "group":       "2T",
        "gen_order":   3,
        "S":           8,
        "lambda_comb": [6, 8, 12],
        "dims":        [8, 9, 6],
        "rep_content": ["2D+2D", "3D", "1D+1D+2D"],
    },
    "2O_ord3": {
        "label":       r"$2O$, ord-$3$, $|S|=8$",
        "group":       "2O",
        "gen_order":   3,
        "S":           8,
        "lambda_comb": [6, 8, 12],
        "dims":        [16, 18, 12],
        "rep_content": ["4D", "3D+3D", "2D+2D+2D"],
    },
    "2I_ord4": {
        "label":       r"$2I$, ord-$4$, $|S|=30$",
        "group":       "2I",
        "gen_order":   4,
        "S":           30,
        "lambda_comb": [24, 30, 40],
        "dims":        [25, 76, 18],
        "rep_content": ["5D", "6D+4D+...", "3D+3D"],
    },
    "2I_ord5": {
        "label":       r"$2I$, ord-$5$, $|S|=24$",
        "group":       "2I",
        "gen_order":   5,
        "S":           24,
        "lambda_comb": [20, 24, 30],
        "dims":        [54, 25, 40],
        "rep_content": ["6D+3D+3D", "5D", "4D+4D+2D+2D"],
    },
}


def normalised_levels(case_key):
    """Return list of normalised eigenvalue levels for an ADE case."""
    c = ADE_CASES[case_key]
    return [lc / c["S"] for lc in c["lambda_comb"]]


def levels_in_support(case_key, p):
    """
    Check whether all three ADE levels lie strictly inside the KM support.
    Returns (all_in: bool, per_level_flags: list[bool]).
    """
    levels = normalised_levels(case_key)
    lm, lp = lam_support(p)
    flags = [lm < lv < lp for lv in levels]
    return all(flags), flags


# ===========================================================
# Reading A: linear saturation
# M_i propto c_i / dim_rho_i
# ===========================================================

def reading_A(case_key, p):
    """
    Reading-A mass proxies for an ADE case at prime p.

    Saturation condition: N(lambda_i; n) >= k * dim_rho_i
    with N(lambda; n) ~ F_KM(lambda) * n (linear growth).
    Stabilisation rank: n_i ~ k * dim_i / c_i.
    Mass proxy: M_i propto c_i / dim_i  (k cancels in ratios).

    Returns dict with:
      c           : [c1, c2, c3]  (KM CDF values)
      mass_proxy  : [c1/d1, c2/d2, c3/d3]
      r12, r23, r13 : mass ratios M_i/M_j
      mass_order  : indices sorted by decreasing mass

    Returns None if any level is outside the support.
    """
    ok, flags = levels_in_support(case_key, p)
    if not ok:
        return None

    levels = normalised_levels(case_key)
    dims   = ADE_CASES[case_key]["dims"]
    c      = [km_cdf(lv, p) for lv in levels]
    proxy  = [c[i] / dims[i] for i in range(3)]

    return {
        "c":          c,
        "mass_proxy": proxy,
        "r12":        proxy[0] / proxy[1],
        "r23":        proxy[1] / proxy[2],
        "r13":        proxy[0] / proxy[2],
        "mass_order": sorted(range(3), key=lambda i: -proxy[i]),
    }


# ===========================================================
# Reading B: combinatorial / exponential saturation
# M_i propto c_i^{dim_rho_i}
# ===========================================================

def reading_B(case_key, p):
    """
    Reading-B mass proxies for an ADE case at prime p.

    Saturation condition: requires dim_rho_i independent projective
    constraints to be simultaneously satisfied. If each mode satisfies
    a single constraint with spectral weight c_i = F_KM(lambda_i),
    the joint probability is c_i^{dim_i}. Stabilisation occurs when
    N(lambda_i; n) * c_i^{dim_i} ~ 1, giving n_i ~ c_i^{-(dim_i+1)}.
    Mass proxy: M_i propto c_i^{dim_i+1} ~ c_i^{dim_i} (same scaling).

    Returns dict with:
      c           : [c1, c2, c3]
      log10_M     : [dim_1*log10(c1), dim_2*log10(c2), dim_3*log10(c3)]
      log10_r12, log10_r23, log10_r13 : log10 of mass ratios
      r12, r23, r13 : mass ratios (may be very small)
      mass_order  : indices sorted by decreasing log10_M

    Returns None if any level is outside the support.
    """
    ok, _ = levels_in_support(case_key, p)
    if not ok:
        return None

    levels = normalised_levels(case_key)
    dims   = ADE_CASES[case_key]["dims"]
    c      = [km_cdf(lv, p) for lv in levels]
    log10m = [dims[i] * np.log10(c[i]) for i in range(3)]

    return {
        "c":          c,
        "log10_M":    log10m,
        "log10_r12":  log10m[0] - log10m[1],
        "log10_r23":  log10m[1] - log10m[2],
        "log10_r13":  log10m[0] - log10m[2],
        "r12":        10**(log10m[0] - log10m[1]),
        "r23":        10**(log10m[1] - log10m[2]),
        "r13":        10**(log10m[0] - log10m[2]),
        "mass_order": sorted(range(3), key=lambda i: -log10m[i]),
    }


# ===========================================================
# O1 analysis: exit-p formulae and support table
# ===========================================================

def exit_p_exact(lambda_norm):
    """
    Exact prime value p at which normalised level lambda_norm exits the
    Kesten-McKay support [lambda_-, lambda_+].

    Derivation (for lambda_norm > 1, exits through lambda_+):
      1 + 2*sqrt(p)/(p+1) = lambda_norm
      Let delta = lambda_norm - 1 > 0, x = sqrt(p):
      2x = delta*(x^2 + 1)
      delta*x^2 - 2x + delta = 0
      x = (1 + sqrt(1 - delta^2)) / delta   [large-p root]
      p_exit = x^2

    For lambda_norm < 1, delta = 1 - lambda_norm, same formula.
    For lambda_norm = 1, never exits.

    Returns (p_exit: float or None, description: str).
    """
    if abs(lambda_norm - 1.0) < 1e-14:
        return None, "lambda = 1: never exits (midpoint of support)"

    delta = abs(lambda_norm - 1.0)
    if delta >= 1.0:
        return None, "level outside support for all p >= 3"

    disc = 1.0 - delta * delta
    x = (1.0 + np.sqrt(disc)) / delta
    p_exit = x * x
    return p_exit, f"sqrt(p_exit) = (1 + sqrt(1 - {delta:.6f}^2)) / {delta:.6f} = {x:.6f}"


def support_table(case_key, p_values):
    """
    For a given ADE case and list of prime values, compute support edges,
    which levels are inside the support, and CDF values for in-support levels.

    Returns list of dicts, one per p, each with keys:
      p, lam_minus, lam_plus, in_support (list), c (list, None if out)
    """
    levels = normalised_levels(case_key)
    rows = []
    for p in p_values:
        lm, lp = lam_support(p)
        flags = [lm < lv < lp for lv in levels]
        c_vals = []
        for i, lv in enumerate(levels):
            c_vals.append(km_cdf(lv, p) if flags[i] else None)
        rows.append({
            "p":          p,
            "lam_minus":  lm,
            "lam_plus":   lp,
            "in_support": flags,
            "c":          c_vals,
        })
    return rows


# ===========================================================
# Standard Model mass ratios (approximate, for comparison)
# ===========================================================

SM_RATIOS = {
    "charged_leptons": {
        "label": "Charged leptons",
        "r12":   1.0 / 206.8,       # m_e / m_mu
        "r23":   1.0 / 16.82,       # m_mu / m_tau
        "r13":   1.0 / 3477.0,      # m_e / m_tau
    },
    "up_quarks": {
        "label": "Up-type quarks",
        "r12":   2.2e-3 / 1.27,     # m_u / m_c
        "r23":   1.27   / 172.4,    # m_c / m_t
        "r13":   2.2e-3 / 172.4,    # m_u / m_t
    },
    "down_quarks": {
        "label": "Down-type quarks",
        "r12":   4.7e-3 / 93e-3,    # m_d / m_s
        "r23":   93e-3  / 4.18,     # m_s / m_b
        "r13":   4.7e-3 / 4.18,     # m_d / m_b
    },
}


# ===========================================================
# Self-test
# ===========================================================

if __name__ == "__main__":
    print("=" * 55)
    print("spectral_relaxation_lib.py -- self-test")
    print("=" * 55)

    print("\n1. KM density normalisation (should be ~1.0):")
    for p in [5, 13, 29, 53]:
        norm = km_norm_check(p)
        lm, lp = lam_support(p)
        print(f"   p={p:3d}: norm={norm:.8f}, "
              f"support=[{lm:.4f}, {lp:.4f}]")

    print("\n2. F_KM(1) = 0.5 (exact by symmetry):")
    for p in [5, 13, 29, 53, 101]:
        val = km_cdf(1.0, p)
        print(f"   p={p:3d}: F_KM(1) = {val:.10f}")

    print("\n3. Exit-p values for 2I ord-5 levels:")
    case = "2I_ord5"
    for lv, name in zip(normalised_levels(case),
                        ["lambda_1=20/24", "lambda_2=24/24", "lambda_3=30/24"]):
        p_exit, desc = exit_p_exact(lv)
        if p_exit is not None:
            print(f"   {name}: p_exit = {p_exit:.4f}")
        else:
            print(f"   {name}: {desc}")

    print("\n4. Reading A for 2I_ord5, p=29:")
    res = reading_A("2I_ord5", 29)
    if res:
        print(f"   c = {[f'{v:.4f}' for v in res['c']]}")
        print(f"   M1/M2 = {res['r12']:.4f}, M2/M3 = {res['r23']:.4f}, "
              f"M1/M3 = {res['r13']:.6f}")

    print("\n5. Reading B for 2I_ord5, p=29:")
    res = reading_B("2I_ord5", 29)
    if res:
        print(f"   log10(M) = {[f'{v:.1f}' for v in res['log10_M']]}")
        print(f"   log10(M1/M2) = {res['log10_r12']:.1f}, "
              f"log10(M2/M3) = {res['log10_r23']:.1f}, "
              f"log10(M1/M3) = {res['log10_r13']:.1f}")
    print("\nAll checks passed.")
