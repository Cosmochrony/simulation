"""
O28 — Asymptotic Calibration of the BFS Window and Effective Dimension
       of the Admissible Representation Sector

Two measurements from existing O25 checkpoints q{q}_o25.npz:
  (A) n1(q)/q calibration: extract the fitting-window upper bound n1 for each
      prime q, fit n1 = alpha*q + beta, test convergence of n1/q.
  (B) r_eff measurement: estimate the effective dimension of the trajectory
      {M_f(n)} in End(V_rho) by computing the rank of the Gram matrix of
      vectorised sigma_c and sigma_{q-c} vectors across shells, testing
      against the spin-1/2 prediction r_eff = d_rho^2 = 4.

Usage:
    python o28_analysis.py --npz-dir .
    python o28_analysis.py --npz-dir /path/to/checkpoints --primes 29 61 101 151 211

Output:
    o28_n1_calibration.pdf
    o28_reff_measurement.pdf
    o28_results.txt
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DEFAULT_PRIMES = [29, 61, 101, 151, 211]
ETA = 0.5          # O14 normalisation exponent
ADMISSIBLE_LO = 7.4
ADMISSIBLE_HI = 10.6


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_o25_npz(npz_dir, q):
    """Load q{q}_o25.npz; return the data dict or None."""
    path = os.path.join(npz_dir, f"q{q}_o25.npz")
    if not os.path.exists(path):
        print(f"  [warn] {path} not found, skipping q={q}")
        return None
    data = dict(np.load(path, allow_pickle=True))
    return data


def inspect_keys(data, q):
    """Print available keys for diagnostic."""
    print(f"  q={q} keys: {list(data.keys())}")


# ---------------------------------------------------------------------------
# Part A: n1(q)/q calibration
# ---------------------------------------------------------------------------

def extract_n1_from_npz(data, q):
    """
    Extract the fitting-window upper bound n1 from the npz checkpoint.

    The O25 pipeline stores per-pair results.  We look for:
      - 'n1' or 'window_n1' as a scalar or per-pair array
      - 'windows': array of shape (n_pairs, 2) with [n0, n1] per pair
      - fallback: infer from sigma_bar decay end

    Returns the mean n1 (float) or None if not recoverable.
    """
    # Direct scalar key
    for key in ("n1", "window_n1", "n1_mean"):
        if key in data:
            val = float(np.asarray(data[key]).flat[0])
            print(f"  q={q}: n1 from key '{key}' = {val:.1f}")
            return val

    # Array of windows shape (n_pairs, 2)
    for key in ("windows", "fitting_windows", "window"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                n1_vals = arr[:, 1]
                n1_mean = float(np.mean(n1_vals))
                print(f"  q={q}: n1 from '{key}' array, mean={n1_mean:.2f} "
                      f"(min={n1_vals.min():.0f}, max={n1_vals.max():.0f})")
                return n1_mean
            elif arr.ndim == 1 and len(arr) == 2:
                n1 = float(arr[1])
                print(f"  q={q}: n1 from '{key}' 1-D = {n1:.1f}")
                return n1

    # Per-pair delta list with stored window info
    for key in ("delta_pairs", "per_pair"):
        if key in data:
            arr = np.asarray(data[key], dtype=object)
            print(f"  q={q}: '{key}' shape {arr.shape}, dtype {arr.dtype}")

    # sigma_bar fallback: find last shell before saturation
    for key in ("sigma_bar", "sigma_mean", "sigma_c_mean"):
        if key in data:
            sigma = np.asarray(data[key]).ravel()
            if len(sigma) >= 3:
                # saturation: sigma drops below 1% of its maximum
                threshold = 0.01 * sigma.max()
                idx = np.where(sigma > threshold)[0]
                n1_est = int(idx[-1]) if len(idx) > 0 else len(sigma) - 1
                print(f"  q={q}: n1 estimated from '{key}' threshold = {n1_est}")
                return float(n1_est)

    print(f"  q={q}: could not extract n1 from checkpoint keys {list(data.keys())}")
    return None


def calibrate_n1_over_q(npz_dir, primes):
    """
    For each prime q, extract n1 and compute n1/q.
    Fit n1 = alpha*q + beta by OLS.  Return results dict.
    """
    results = {}
    for q in primes:
        data = load_o25_npz(npz_dir, q)
        if data is None:
            continue
        n1 = extract_n1_from_npz(data, q)
        if n1 is not None:
            results[q] = {"n1": n1, "ratio": n1 / q}

    if len(results) < 2:
        print("[warn] fewer than 2 primes with n1 data; calibration skipped")
        return results, None, None

    qs = np.array(sorted(results.keys()), dtype=float)
    n1s = np.array([results[int(q)]["n1"] for q in qs])

    slope, intercept, r, p, se = stats.linregress(qs, n1s)
    print(f"\n  OLS fit n1 = {slope:.4f}*q + {intercept:.2f}  "
          f"(R^2={r**2:.4f}, p={p:.4f})")
    return results, slope, intercept


def plot_n1_calibration(results, slope, intercept, outdir, delta_rows=None):
    qs  = np.array(sorted(results.keys()), dtype=float)
    n1s = np.array([results[int(q)]["n1"] for q in qs])
    ratios = np.array([results[int(q)]["ratio"] for q in qs])

    n_panels = 3 if delta_rows else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))

    COLOR_OBS  = "#2c7bb6"
    COLOR_FIT  = "#333333"
    COLOR_CORR = "#d7191c"
    COLOR_RAW  = "#bdbdbd"
    COLOR_WIN  = "#31a354"

    # Panel (a): n1 vs q with OLS fit and 95%% CI band
    ax = axes[0]
    if slope is not None:
        n = len(qs)
        _, _, _, _, se = stats.linregress(qs, n1s)
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ci = t_crit * se
        q_fit = np.linspace(qs.min() * 0.88, qs.max() * 1.08, 300)
        ax.plot(q_fit, slope * q_fit + intercept, "--", color=COLOR_FIT, lw=1.4,
                label=rf"OLS: $n_1 = {slope:.4f}\,q + {intercept:.2f}$")
        ax.fill_between(q_fit,
                        (slope - ci) * q_fit + intercept,
                        (slope + ci) * q_fit + intercept,
                        alpha=0.15, color=COLOR_FIT, label="95\%% CI on slope")
    ax.scatter(qs, n1s, color=COLOR_OBS, zorder=5, s=50, label="$n_1(q)$ (O25)")
    for q, n1 in zip(qs, n1s):
        ax.annotate(f"$q={int(q)}$", (q, n1), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$n_1(q)$")
    ax.set_title("(a) BFS window depth $n_1(q)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")

    # Panel (b): n1/q convergence with labels
    ax = axes[1]
    if slope is not None:
        ax.axhline(slope, color=COLOR_FIT, linestyle="--", lw=1.2,
                   label=rf"OLS slope $\hat{{\alpha}} = {slope:.4f}$")
        ax.axhspan(slope - ci, slope + ci, alpha=0.12, color=COLOR_FIT)
    ax.scatter(qs, ratios, color=COLOR_OBS, zorder=5, s=50, label="$n_1(q)/q$")
    for q, r in zip(qs, ratios):
        ax.annotate(f"{r:.3f}", (q, r), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$n_1(q)/q$")
    ax.set_title("(b) Convergence of $n_1(q)/q$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")

    # Panel (c): raw vs O14-corrected delta_pair
    if delta_rows:
        ax = axes[2]
        dqs   = np.array([r["q"]              for r in delta_rows], dtype=float)
        dbars = np.array([r["delta_pair_mean"] for r in delta_rows])
        dstds = np.array([r["delta_pair_std"]  for r in delta_rows])
        dcorr = np.array([r["delta_corr"]      for r in delta_rows])
        ax.axhspan(ADMISSIBLE_LO, ADMISSIBLE_HI, alpha=0.12, color=COLOR_WIN,
                   label=f"admissible $[{ADMISSIBLE_LO},{ADMISSIBLE_HI}]$")
        ax.errorbar(dqs, dbars, yerr=dstds, fmt="o", color=COLOR_RAW,
                    capsize=4, lw=1.2, label=r"$\bar{\delta}_{\rm pair}$ (raw)")
        ax.plot(dqs, dcorr, "s--", color=COLOR_CORR, markersize=7, lw=1.4,
                label=r"$\delta_{\rm corr}$ (O14-corrected)")
        for q, dc in zip(dqs, dcorr):
            ax.annotate(f"{dc:.2f}", (q, dc), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color=COLOR_CORR)
        ax.set_xlabel("prime $q$")
        ax.set_ylabel(r"$\delta_{\rm pair}$")
        ax.set_title(r"(c) Raw vs corrected $\delta_{\rm pair}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in")

    fig.suptitle("O28 \u2014 Part A: BFS window depth calibration", fontsize=11)
    fig.tight_layout()
    outpath = os.path.join(outdir, "o28_n1_calibration.pdf")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")
    return outpath



# ---------------------------------------------------------------------------
# Part B: r_eff measurement
# ---------------------------------------------------------------------------

def extract_sigma_vectors(data, q):
    """
    Extract per-pair mean sigma_c and sigma_{q-c} vectors from the checkpoint.

    The patched O25 pipeline stores:
      - 'sigma_c_mean':   array (n_pairs, n_shells) — mean sigma_c per shell
      - 'sigma_qmc_mean': array (n_pairs, n_shells) — mean sigma_{q-c} per shell

    Returns (sigma_c, sigma_qmc) each of shape (n_pairs, n_shells), or (None, None).
    """
    if "sigma_c_mean" in data and "sigma_qmc_mean" in data:
        sc   = np.asarray(data["sigma_c_mean"],   dtype=float)
        sqmc = np.asarray(data["sigma_qmc_mean"], dtype=float)
        print(f"  q={q}: sigma_c_mean {sc.shape}, sigma_qmc_mean {sqmc.shape}")
        return sc, sqmc

    # Legacy key names from earlier script versions
    for kc, kqmc in [("sigma_c_shells", "sigma_qmc_shells"),
                     ("sc", "sqmc"),
                     ("sigma_c", "sigma_qmc"),
                     ("s_c_mean", "s_qmc_mean")]:
        if kc in data and kqmc in data:
            sc   = np.asarray(data[kc],   dtype=float)
            sqmc = np.asarray(data[kqmc], dtype=float)
            print(f"  q={q}: sigma vectors from ('{kc}', '{kqmc}')")
            return sc, sqmc

    print(f"  q={q}: sigma_c_mean / sigma_qmc_mean not found in checkpoint.")
    print(f"         Re-run o25_paired_pipeline.py --force to regenerate "
          f"with the patched pipeline.")
    return None, None


def measure_reff_from_vectors(sigma_c, sigma_qmc, q, n0=None, n1=None):
    """
    O28 proxy test for the effective dimension of pair trajectories.

    IMPORTANT — scope and limitations
    -----------------------------------
    O26 Criterion 5.4 (Test 4) defines r_eff as the numerical rank of the
    per-pair covariance operator
        C_c = (1/N) sum_{n=n0}^{n1} vec(M^f_n) vec(M^f_n)^dagger
    where M^f_n = v^{(n)}_c ⊗ v^{(n)}_{q-c} is the OUTER PRODUCT of the
    individual Weil fingerprint vectors v^{(n)}_c ∈ C^q.
    This requires storing the full per-block trajectories (n_pairs, M, n_shells,
    q), which the current O25 pipeline does not save.

    The present function implements a PROXY test using only the per-pair mean
    vectors sigma_c_mean[p, n] and sigma_qmc_mean[p, n] (scalars, not full
    Weil vectors).  It measures how many independent directions the (n_pairs)
    mean trajectories span after removing the common power-law decay.
    This proxy is sensitive to inter-pair geometric diversity but cannot
    directly measure the rank of C_c in End(V_rho); it is a necessary
    condition for r_eff > 1, not a sufficient one.

    The exact O26 Test 4 requires a pipeline extension to save per-block
    sigma_c vectors of shape (n_pairs, M, n_shells) — this is noted in O28
    as a direction for subsequent work.

    Method (proxy)
    --------------
    1. Restrict to the pre-saturation fitting window [n0, n1].
    2. Form the pair-product matrix M[p, n] = sigma_c[p, n] * sigma_qmc[p, n].
       Shape: (n_pairs, n_win).
    3. Row-normalise (removes amplitude variation between pairs).
    4. Centre columns (removes the common power-law decay shared by all pairs).
    5. SVD of the centred, row-normalised matrix.  The effective rank measures
       the number of independent decay-shape directions across pairs.

    Returns
    -------
    dict with SVD results, r_eff estimates, and diagnostic quantities.
    """
    if sigma_c.ndim == 1:
        sigma_c   = sigma_c[np.newaxis, :]
    if sigma_qmc.ndim == 1:
        sigma_qmc = sigma_qmc[np.newaxis, :]

    n_pairs  = sigma_c.shape[0]
    n_shells = min(sigma_c.shape[1], sigma_qmc.shape[1])
    sigma_c   = sigma_c[:, :n_shells]
    sigma_qmc = sigma_qmc[:, :n_shells]

    # Step 1: restrict to fitting window
    if n0 is None:
        n0 = 0
    if n1 is None:
        n1 = n_shells - 1
    n0 = max(0, n0)
    n1 = min(n_shells - 1, n1)
    n_win = n1 - n0 + 1

    sc_win   = sigma_c[:, n0:n1 + 1]    # (n_pairs, n_win)
    sqmc_win = sigma_qmc[:, n0:n1 + 1]

    if n_win < 2:
        print(f"  q={q}: window [{n0},{n1}] too short for r_eff — skipping.")
        return None

    # Step 2: pair-product matrix
    Mprod = sc_win * sqmc_win           # (n_pairs, n_win)

    # Step 3: row-normalise (remove amplitude variation between pairs)
    row_norms = np.linalg.norm(Mprod, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-15, 1.0, row_norms)
    Mnorm = Mprod / row_norms           # (n_pairs, n_win), each row unit-norm

    # Step 4: centre columns (remove common decay shape)
    Mcent = Mnorm - Mnorm.mean(axis=0, keepdims=True)

    # Step 5: SVD of centred matrix
    U, sv, Vt = np.linalg.svd(Mcent, full_matrices=False)
    sv_norm = sv / sv[0] if sv[0] > 1e-15 else sv

    # Effective rank: participation ratio
    p2 = sv_norm ** 2
    p2 = p2 / p2.sum()
    r_eff_pr = 1.0 / np.sum(p2 ** 2)

    # Hard threshold at 10% of leading SV (centred matrix; 5% was too loose)
    r_eff_thr = int(np.sum(sv_norm > 0.10))

    # Fraction of variance explained by top-4 components
    var_top4 = float(np.sum(sv[:4] ** 2) / np.sum(sv ** 2))

    print(f"  q={q}: window=[{n0},{n1}]  n_pairs={n_pairs}  n_win={n_win}")
    print(f"  q={q}: singular values (centred, normalised) = "
          + ", ".join(f"{s:.4f}" for s in sv_norm[:8]))
    print(f"  q={q}: r_eff (participation ratio) = {r_eff_pr:.2f}  "
          f"r_eff (10% threshold) = {r_eff_thr}  "
          f"var(top 4) = {100*var_top4:.1f}%")
    print(f"  q={q}: spin-1/2 prediction r_eff = d_rho^2 = 4")
    if n_win < 10:
        print(f"  q={q}: [diagnostic] n_win={n_win} < 10 — window too short "
              f"to discriminate r_eff; result non-informative.")

    return {"sv": sv, "sv_norm": sv_norm,
            "r_eff_pr": r_eff_pr, "r_eff_thr": r_eff_thr,
            "var_top4": var_top4, "informative": n_win >= 10,
            "n_pairs": n_pairs, "n_win": n_win, "n0": n0, "n1": n1}


def plot_reff(reff_results, outdir):
    if not reff_results:
        print("  No r_eff results to plot.")
        return None

    n_panels = len(reff_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (q, res) in zip(axes, sorted(reff_results.items())):
        sv = res["sv_norm"]
        k = min(len(sv), 12)
        ax.bar(range(1, k + 1), sv[:k], color="steelblue", alpha=0.8)
        ax.axhline(0.05, color="red", linestyle="--", linewidth=0.8,
                   label="5\\% threshold")
        ax.axvline(4.5, color="darkorange", linestyle=":", linewidth=1.0,
                   label=r"$d_\rho^2 = 4$ (spin-1/2)")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("normalised $\\sigma_i / \\sigma_1$")
        r_pr  = res["r_eff_pr"]
        r_thr = res["r_eff_thr"]
        v4    = res.get("var_top4", float("nan"))
        n_win = res.get("n_win", "?")
        ax.set_title(f"$q={q}$,  window $n_{{\\rm win}}={n_win}$\n"
                     f"$r_\\mathrm{{eff}}^\\mathrm{{PR}}={r_pr:.2f}$, "
                     f"$r_\\mathrm{{eff}}^{{10\\%}}={r_thr}$, "
                     f"var(top 4)$={100*v4:.0f}\\%$")
        ax.legend(fontsize=7)
        ax.set_xlim(0.5, k + 0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(direction="in")

    fig.suptitle(r"O28 — Part B: Effective dimension $r_\mathrm{eff}$ "
                 r"of trajectory in $\mathrm{End}(V_\rho)$", fontsize=11)
    fig.tight_layout()
    outpath = os.path.join(outdir, "o28_reff_measurement.pdf")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Delta_corr summary
# ---------------------------------------------------------------------------

def compute_delta_corr(npz_dir, primes, n1_results):
    """
    Recompute delta_corr(q) = delta_pair_mean - eta * log(q) / log(n1(q))
    using the measured n1 values.
    Also extracts delta_pair_std for error bars.
    """
    rows = []
    for q in sorted(primes):
        data = load_o25_npz(npz_dir, q)
        if data is None:
            continue
        # Extract mean delta_pair
        delta_mean = None
        for key in ("delta_pair_mean", "delta_mean", "delta_pairs_mean",
                    "mean_delta"):
            if key in data:
                arr = np.asarray(data[key], dtype=float)
                # delta_pair_mean is shape (n_pairs,) — take the mean over pairs
                delta_mean = float(np.nanmean(arr))
                break
        if delta_mean is None and "delta_pairs" in data:
            arr = np.asarray(data["delta_pairs"], dtype=float).ravel()
            delta_mean = float(np.nanmean(arr))
        if delta_mean is None:
            print(f"  q={q}: delta_pair_mean not found in checkpoint")
            continue

        delta_std = None
        for key in ("delta_pair_mean", "delta_mean", "delta_pairs_mean", "mean_delta"):
            if key in data:
                delta_std = float(np.nanstd(np.asarray(data[key], dtype=float)))
                break
        if delta_std is None and "delta_pairs" in data:
            delta_std = float(np.nanstd(np.asarray(data["delta_pairs"], dtype=float).ravel()))

        n1 = n1_results.get(q, {}).get("n1")
        if n1 is None or n1 <= 1:
            continue

        corr_factor = ETA * np.log(q) / np.log(n1)
        delta_corr = delta_mean - corr_factor
        in_window = ADMISSIBLE_LO <= delta_corr <= ADMISSIBLE_HI
        rows.append({
            "q": q,
            "delta_pair_mean": delta_mean,
            "delta_pair_std": delta_std if delta_std is not None else 0.0,
            "n1": n1,
            "n1_over_q": n1 / q,
            "corr_factor": corr_factor,
            "delta_corr": delta_corr,
            "in_window": in_window,
        })
        print(f"  q={q}: delta_pair={delta_mean:.4f}, n1={n1:.1f}, "
              f"n1/q={n1/q:.4f}, factor={corr_factor:.4f}, "
              f"delta_corr={delta_corr:.4f}  "
              + ("[OK]" if in_window else "[OUT]"))
    return rows


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(n1_results, slope, intercept, reff_results, delta_rows, outdir):
    lines = []
    lines.append("O28 — Analysis Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Part A: BFS window depth n1(q)/q")
    lines.append("-" * 40)
    lines.append(f"{'q':>6}  {'n1':>8}  {'n1/q':>8}")
    for q in sorted(n1_results.keys()):
        r = n1_results[q]
        lines.append(f"{q:>6}  {r['n1']:>8.2f}  {r['ratio']:>8.4f}")
    if slope is not None:
        lines.append("")
        lines.append(f"OLS fit: n1 = {slope:.4f} * q + {intercept:.2f}")
        lines.append(f"Estimated asymptotic constant: alpha_hat = {slope:.4f}")
    lines.append("")
    lines.append("Part A: delta_corr with measured n1")
    lines.append("-" * 40)
    lines.append(f"{'q':>6}  {'delta_pair':>12}  {'n1/q':>8}  "
                 f"{'factor':>8}  {'delta_corr':>12}  {'in [7.4,10.6]':>14}")
    for row in delta_rows:
        lines.append(
            f"{row['q']:>6}  {row['delta_pair_mean']:>12.4f}  "
            f"{row['n1_over_q']:>8.4f}  {row['corr_factor']:>8.4f}  "
            f"{row['delta_corr']:>12.4f}  {'YES' if row['in_window'] else 'NO':>14}")
    lines.append("")
    lines.append("Part B: Inter-pair diversity (proxy for O26 Test 4)")
    lines.append("-" * 40)
    lines.append("Note: O26 Test 4 (Criterion 5.4) requires per-block Weil")
    lines.append("      vector trajectories (shape n_pairs x M x n_shells x q),")
    lines.append("      not available in current O25 checkpoints.")
    lines.append("      The values below measure inter-pair trajectory diversity")
    lines.append("      (a necessary but not sufficient proxy for r_eff).")
    lines.append(f"Spin-1/2 prediction (O26): r_eff = d_rho^2 = 4")
    lines.append("")
    if reff_results:
        lines.append(f"{'q':>6}  {'n_win':>6}  {'r_eff (PR)':>12}  {'r_eff (10%)':>12}  "
                     f"{'var top4 %':>12}  {'informative':>12}  {'consistent with 4':>18}")
        for q, res in sorted(reff_results.items()):
            consistent = abs(res["r_eff_pr"] - 4.0) < 1.5
            v4 = res.get("var_top4", float("nan"))
            inf_flag = "YES" if res.get("informative", False) else "NO (n_win<10)"
            lines.append(
                f"{q:>6}  {res['n_win']:>6}  {res['r_eff_pr']:>12.2f}  "
                f"{res['r_eff_thr']:>12}  {100*v4:>12.1f}  "
                f"{inf_flag:>12}  {'YES' if consistent else 'NO':>18}")
    else:
        lines.append("  (sigma_c / sigma_{q-c} shell vectors not found in checkpoints)")
        lines.append("  Re-run o25_paired_pipeline.py with --save-sigma-shells to enable Part B.")
    lines.append("")

    report = "\n".join(lines)
    outpath = os.path.join(outdir, "o28_results.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved {outpath}")
    print()
    print(report)
    return outpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="O28 analysis: n1(q)/q calibration and r_eff measurement")
    parser.add_argument("--npz-dir", default=".",
                        help="Directory containing q{q}_o25.npz files")
    parser.add_argument("--primes", type=int, nargs="+", default=DEFAULT_PRIMES,
                        metavar="Q",
                        help="Primes to analyse (default: 29 61 101 151)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for figures and report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== O28 — Part A: BFS window depth calibration ===\n")
    n1_results, slope, intercept = calibrate_n1_over_q(args.npz_dir, args.primes)

    print("\n=== O28 — Part A (bis): delta_corr with measured n1 ===\n")
    delta_rows = compute_delta_corr(args.npz_dir, args.primes, n1_results)

    outpath_n1 = None
    if n1_results:
        outpath_n1 = plot_n1_calibration(n1_results, slope, intercept, args.out_dir,
                                         delta_rows=delta_rows)

    print("\n=== O28 — Part B: Inter-pair diversity (proxy for O26 Test 4) ===\n")
    print("  [scope] O26 Criterion 5.4 requires per-block Weil vector trajectories")
    print("  [scope] not stored in current O25 checkpoints.  Results below are a")
    print("  [scope] proxy (inter-pair mean trajectory diversity after centering).\n")
    reff_results = {}
    for q in args.primes:
        data = load_o25_npz(args.npz_dir, q)
        if data is None:
            continue
        sc, sqmc = extract_sigma_vectors(data, q)
        if sc is not None and sqmc is not None:
            n0_q = int(np.asarray(data.get("n0", 0)).flat[0])
            n1_q = int(np.asarray(data.get("n1", sc.shape[1] - 1)).flat[0])
            res = measure_reff_from_vectors(sc, sqmc, q, n0=n0_q, n1=n1_q)
            if res is not None:
                reff_results[q] = res

    outpath_reff = plot_reff(reff_results, args.out_dir)

    print("\n=== O28 — Report ===\n")
    write_report(n1_results, slope, intercept, reff_results, delta_rows, args.out_dir)

    # Diagnostic: if checkpoints not found, explain what to add to pipeline
    missing = [q for q in args.primes
               if q not in n1_results and q not in reff_results]
    if missing:
        print(f"\n[info] Primes with no data: {missing}")
        print("       Ensure q{q}_o25.npz exist in --npz-dir or regenerate with")
        print("       python o25_paired_pipeline.py --primes "
              + " ".join(str(q) for q in missing)
              + " --M 50 --bfs-frac 0.99 --auto-window")

    # If r_eff could not be measured, explain the pipeline patch needed
    if not reff_results:
        print()
        print("[info] Part B requires sigma_c_mean and sigma_qmc_mean arrays.")
        print("       Re-run the patched o25_paired_pipeline.py with --force:")
        print("       python o25_paired_pipeline.py --primes "
              + " ".join(str(q) for q in args.primes)
              + " --M 50 --bfs-frac 0.99 --auto-window --force")


if __name__ == "__main__":
    main()