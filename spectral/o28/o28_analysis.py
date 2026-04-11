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
    o28_n1_calibration.png
    o28_reff_measurement.png
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
DEFAULT_PRIMES = [29, 61, 101, 151]
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


def plot_n1_calibration(results, slope, intercept, outdir):
    qs = np.array(sorted(results.keys()), dtype=float)
    ratios = np.array([results[int(q)]["ratio"] for q in qs])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: n1 vs q with linear fit
    ax = axes[0]
    ax.scatter(qs, [results[int(q)]["n1"] for q in qs], color="steelblue",
               zorder=5, label="observed $n_1(q)$")
    if slope is not None:
        q_fit = np.linspace(qs.min() * 0.9, qs.max() * 1.1, 200)
        ax.plot(q_fit, slope * q_fit + intercept, "k--",
                label=rf"fit: $n_1 = {slope:.3f}\,q + {intercept:.1f}$")
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$n_1(q)$")
    ax.set_title("BFS window depth $n_1(q)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: n1/q ratio
    ax = axes[1]
    ax.scatter(qs, ratios, color="darkorange", zorder=5)
    if slope is not None:
        ax.axhline(slope, color="k", linestyle="--",
                   label=rf"OLS slope $\hat\alpha = {slope:.4f}$")
    ax.set_xlabel("prime $q$")
    ax.set_ylabel("$n_1(q)/q$")
    ax.set_title("Convergence of $n_1(q)/q$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(direction="in")

    fig.suptitle("O28 — Part A: BFS window depth calibration", fontsize=11)
    fig.tight_layout()
    outpath = os.path.join(outdir, "o28_n1_calibration.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Part B: r_eff measurement
# ---------------------------------------------------------------------------

def extract_sigma_vectors(data, q):
    """
    Extract per-shell sigma_c and sigma_{q-c} vectors from the checkpoint.

    The O25 pipeline stores sigma values per pair and per shell.  We look for:
      - 'sigma_c_shells': array (n_pairs, n_shells) for c-blocks
      - 'sigma_qmc_shells': array (n_pairs, n_shells) for (q-c)-blocks
      - or combined 'sigma_pair_shells': (n_pairs, n_shells)

    Returns (sigma_c, sigma_qmc) each of shape (n_pairs, n_shells), or None.
    """
    # Preferred: separate c and q-c arrays
    if "sigma_c_shells" in data and "sigma_qmc_shells" in data:
        sc = np.asarray(data["sigma_c_shells"], dtype=float)
        sqmc = np.asarray(data["sigma_qmc_shells"], dtype=float)
        print(f"  q={q}: sigma_c_shells {sc.shape}, sigma_qmc_shells {sqmc.shape}")
        return sc, sqmc

    # Fallback: single pair product array — cannot decompose
    if "sigma_pair_shells" in data:
        print(f"  q={q}: only sigma_pair_shells available; r_eff requires "
              f"separate c / q-c vectors — skipping Part B")
        return None, None

    # Try generic names
    for kc, kqmc in [("sc", "sqmc"), ("sigma_c", "sigma_qmc"),
                     ("s_c_mean", "s_qmc_mean")]:
        if kc in data and kqmc in data:
            sc = np.asarray(data[kc], dtype=float)
            sqmc = np.asarray(data[kqmc], dtype=float)
            if sc.ndim >= 1:
                print(f"  q={q}: sigma vectors from ('{kc}', '{kqmc}')")
                return sc, sqmc

    print(f"  q={q}: sigma_c/sigma_{{q-c}} vectors not found in checkpoint; "
          f"r_eff measurement requires pipeline re-run with --save-sigma-shells")
    return None, None


def measure_reff_from_vectors(sigma_c, sigma_qmc, q):
    """
    Estimate the effective dimension r_eff of the trajectory
    {v_c(n) ⊗ v_{q-c}(n)} in End(V_rho) ≃ R^{d_rho^2}.

    Method: for each pair p, form the vectorised outer product
        m_p(n) = sigma_c[p, n] * sigma_qmc[p, n]   (scalar per shell)
    Stack across shells to get a matrix M of shape (n_pairs, n_shells).
    The effective rank of M estimates r_eff.

    Prediction (spin-1/2): r_eff = d_rho^2 = 4.
    """
    # sigma_c shape: (n_pairs, n_shells)
    if sigma_c.ndim == 1:
        sigma_c = sigma_c[np.newaxis, :]
    if sigma_qmc.ndim == 1:
        sigma_qmc = sigma_qmc[np.newaxis, :]

    n_pairs = sigma_c.shape[0]
    n_shells = min(sigma_c.shape[1], sigma_qmc.shape[1])
    sigma_c = sigma_c[:, :n_shells]
    sigma_qmc = sigma_qmc[:, :n_shells]

    # Outer product trajectories: shape (n_pairs, n_shells)
    M = sigma_c * sigma_qmc

    # Singular value decomposition of M (n_pairs x n_shells)
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    sv_norm = sv / sv[0]

    # Effective rank via participation ratio
    p2 = sv_norm ** 2
    p2 /= p2.sum()
    r_eff_pr = 1.0 / np.sum(p2 ** 2)

    # Hard threshold at 5% of leading singular value
    r_eff_thr = int(np.sum(sv_norm > 0.05))

    print(f"  q={q}: singular values (normalised) = "
          + ", ".join(f"{s:.4f}" for s in sv_norm[:8]))
    print(f"  q={q}: r_eff (participation ratio) = {r_eff_pr:.2f}  "
          f"r_eff (5% threshold) = {r_eff_thr}")
    print(f"  q={q}: spin-1/2 prediction r_eff = d_rho^2 = 4")

    return {"sv": sv, "sv_norm": sv_norm, "r_eff_pr": r_eff_pr,
            "r_eff_thr": r_eff_thr, "n_pairs": n_pairs, "n_shells": n_shells}


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
        r_pr = res["r_eff_pr"]
        r_thr = res["r_eff_thr"]
        ax.set_title(f"$q={q}$\n"
                     f"$r_\\mathrm{{eff}}^\\mathrm{{PR}}={r_pr:.2f}$, "
                     f"$r_\\mathrm{{eff}}^{{5\\%}}={r_thr}$")
        ax.legend(fontsize=7)
        ax.set_xlim(0.5, k + 0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(direction="in")

    fig.suptitle(r"O28 — Part B: Effective dimension $r_\mathrm{eff}$ "
                 r"of trajectory in $\mathrm{End}(V_\rho)$", fontsize=11)
    fig.tight_layout()
    outpath = os.path.join(outdir, "o28_reff_measurement.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Delta_corr summary
# ---------------------------------------------------------------------------

def compute_delta_corr(npz_dir, primes, n1_results):
    """
    Recompute delta_corr(q) = delta_pair_mean * eta * log(q) / log(n1(q))
    using the measured n1 values.
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
                delta_mean = float(np.asarray(data[key]).flat[0])
                break
        if delta_mean is None and "delta_pairs" in data:
            arr = np.asarray(data["delta_pairs"], dtype=float).ravel()
            delta_mean = float(np.nanmean(arr))
        if delta_mean is None:
            print(f"  q={q}: delta_pair_mean not found in checkpoint")
            continue

        n1 = n1_results.get(q, {}).get("n1")
        if n1 is None or n1 <= 1:
            continue

        corr_factor = ETA * np.log(q) / np.log(n1)
        delta_corr = delta_mean * corr_factor
        in_window = ADMISSIBLE_LO <= delta_corr <= ADMISSIBLE_HI
        rows.append({
            "q": q,
            "delta_pair_mean": delta_mean,
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
    lines.append("Part B: Effective dimension r_eff")
    lines.append("-" * 40)
    lines.append(f"Spin-1/2 prediction: r_eff = d_rho^2 = 4")
    lines.append("")
    if reff_results:
        lines.append(f"{'q':>6}  {'r_eff (PR)':>12}  {'r_eff (5%)':>12}  "
                     f"{'consistent with 4':>18}")
        for q, res in sorted(reff_results.items()):
            consistent = abs(res["r_eff_pr"] - 4.0) < 1.0
            lines.append(
                f"{q:>6}  {res['r_eff_pr']:>12.2f}  {res['r_eff_thr']:>12}  "
                f"{'YES' if consistent else 'NO':>18}")
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

    outpath_n1 = None
    if n1_results:
        outpath_n1 = plot_n1_calibration(n1_results, slope, intercept, args.out_dir)

    print("\n=== O28 — Part A (bis): delta_corr with measured n1 ===\n")
    delta_rows = compute_delta_corr(args.npz_dir, args.primes, n1_results)

    print("\n=== O28 — Part B: r_eff measurement ===\n")
    reff_results = {}
    for q in args.primes:
        data = load_o25_npz(args.npz_dir, q)
        if data is None:
            continue
        sc, sqmc = extract_sigma_vectors(data, q)
        if sc is not None and sqmc is not None:
            res = measure_reff_from_vectors(sc, sqmc, q)
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

    # If r_eff could not be measured, print the pipeline patch needed
    if not reff_results:
        print()
        print("[info] Part B requires sigma_c_shells and sigma_qmc_shells arrays.")
        print("       Add --save-sigma-shells to o25_paired_pipeline.py and")
        print("       save np.array(sigma_c_per_shell) and np.array(sigma_qmc_per_shell)")
        print("       under those keys in the npz checkpoint.")


if __name__ == "__main__":
    main()
