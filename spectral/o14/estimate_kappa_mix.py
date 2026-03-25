#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: pathlib.Path) -> dict[int, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    data: dict[int, np.ndarray] = {}
    for k, v in raw.items():
        n = int(k)
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"JSON entry for n={n} is not 1D")
        data[n] = arr

    return data


def load_npz(path: pathlib.Path) -> dict[int, np.ndarray]:
    z = np.load(path, allow_pickle=False)

    if "n_values" not in z or "sigma_matrix" not in z:
        raise ValueError(
            "NPZ must contain arrays 'n_values' (1D) and 'sigma_matrix' (2D)"
        )

    n_values = np.asarray(z["n_values"], dtype=int)
    sigma_matrix = np.asarray(z["sigma_matrix"], dtype=float)

    if n_values.ndim != 1:
        raise ValueError("'n_values' must be 1D")
    if sigma_matrix.ndim != 2:
        raise ValueError("'sigma_matrix' must be 2D")
    if sigma_matrix.shape[0] != len(n_values):
        raise ValueError(
            "sigma_matrix.shape[0] must equal len(n_values)"
        )

    data: dict[int, np.ndarray] = {}
    for i, n in enumerate(n_values):
        data[int(n)] = sigma_matrix[i, :]

    return data


def load_data(path: pathlib.Path) -> dict[int, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".npz":
        return load_npz(path)

    raise ValueError("Supported input formats are .json and .npz")


def compute_rows(
    sigmas_by_n: dict[int, np.ndarray],
    n0: int,
    n1: int,
    eps: float = 1e-15,
) -> list[dict]:
    rows: list[dict] = []

    for n in range(n0, n1 + 1):
        if n not in sigmas_by_n:
            continue

        vals = np.asarray(sigmas_by_n[n], dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        mean_n = float(np.mean(vals))
        var_n = float(np.var(vals, ddof=0))
        vn = var_n / max(mean_n * mean_n, eps)

        if mean_n > 0.0 and vn > 0.0:
            rows.append(
                {
                    "n": int(n),
                    "num_blocks": int(vals.size),
                    "mean_sigma": mean_n,
                    "var_sigma": var_n,
                    "Vn": vn,
                }
            )

    return rows


def ols_loglog_fit(rows: list[dict], y_key: str) -> dict:
    if len(rows) < 2:
        raise ValueError("Need at least 2 points for log-log fit")

    x = np.log(np.asarray([r["n"] for r in rows], dtype=float))
    y = np.log(np.asarray([r[y_key] for r in rows], dtype=float))

    slope, intercept = np.polyfit(x, y, 1)

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def jackknife_alpha_mix(
    sigmas_by_n: dict[int, np.ndarray],
    n0: int,
    n1: int,
) -> dict:
    ns = [n for n in range(n0, n1 + 1) if n in sigmas_by_n]
    if not ns:
        raise ValueError("No shells in requested window")

    lengths = [len(sigmas_by_n[n]) for n in ns]
    m = min(lengths)

    if m < 3:
        raise ValueError("Need at least 3 blocks for jackknife")

    estimates = []

    for j in range(m):
        reduced: dict[int, np.ndarray] = {}
        for n in ns:
            arr = np.asarray(sigmas_by_n[n], dtype=float)
            arr = arr[:m]
            reduced[n] = np.delete(arr, j)

        rows = compute_rows(reduced, n0, n1)
        fit = ols_loglog_fit(rows, "Vn")
        alpha_mix = -fit["slope"]
        kappa_mix = alpha_mix / 2.0
        estimates.append((alpha_mix, kappa_mix, fit["r2"]))

    arr = np.asarray(estimates, dtype=float)

    return {
        "alpha_mean": float(np.mean(arr[:, 0])),
        "alpha_std": float(np.std(arr[:, 0], ddof=1)),
        "kappa_mean": float(np.mean(arr[:, 1])),
        "kappa_std": float(np.std(arr[:, 1], ddof=1)),
        "r2_mean": float(np.mean(arr[:, 2])),
        "jackknife_reps": int(len(estimates)),
        "blocks_used": int(m),
    }


def save_csv(rows: list[dict], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("n,num_blocks,mean_sigma,var_sigma,Vn,log_n,log_Vn\n")
        for r in rows:
            log_n = math.log(r["n"])
            log_vn = math.log(r["Vn"])
            f.write(
                f"{r['n']},{r['num_blocks']},{r['mean_sigma']:.16e},"
                f"{r['var_sigma']:.16e},{r['Vn']:.16e},"
                f"{log_n:.16e},{log_vn:.16e}\n"
            )


def make_plot(
    rows: list[dict],
    fit: dict,
    out_pdf: pathlib.Path,
    title: str,
) -> None:
    n = np.asarray([r["n"] for r in rows], dtype=float)
    vn = np.asarray([r["Vn"] for r in rows], dtype=float)

    plt.figure(figsize=(6.5, 4.8))
    plt.loglog(n, vn, "o", label="Measured $V_n$")

    n_fit = np.linspace(np.min(n), np.max(n), 300)
    vn_fit = np.exp(fit["intercept"]) * (n_fit ** fit["slope"])
    alpha_mix = -fit["slope"]
    kappa_mix = alpha_mix / 2.0

    plt.loglog(
        n_fit,
        vn_fit,
        "--",
        label=(
            rf"Fit: $V_n \sim n^{{-{alpha_mix:.3f}}}$, "
            rf"$\kappa_{{mix}}={kappa_mix:.3f}$, "
            rf"$R^2={fit['r2']:.4f}$"
        ),
    )

    plt.xlabel("BFS depth $n$")
    plt.ylabel(r"$V_n = \mathrm{Var}_c(\Sigma_n^{(c)}) / \bar{\Sigma}_n^2$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate kappa_mix from inter-block variance decay "
            "V_n ~ n^{-alpha_mix}"
        )
    )
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="Input file (.json or .npz)",
    )
    parser.add_argument(
        "--n0",
        type=int,
        required=True,
        help="Lower bound of fitting window",
    )
    parser.add_argument(
        "--n1",
        type=int,
        required=True,
        help="Upper bound of fitting window",
    )
    parser.add_argument(
        "--out-prefix",
        type=pathlib.Path,
        default=pathlib.Path("kappa_mix"),
        help="Output prefix for CSV/PDF/TXT",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Inter-block variance decay and kappa_mix",
        help="Plot title",
    )

    args = parser.parse_args()

    sigmas_by_n = load_data(args.input)
    rows = compute_rows(sigmas_by_n, args.n0, args.n1)

    if len(rows) < 2:
        print("Not enough valid points in the requested window.", file=sys.stderr)
        return 1

    fit = ols_loglog_fit(rows, "Vn")
    alpha_mix = -fit["slope"]
    kappa_mix = alpha_mix / 2.0

    jk = jackknife_alpha_mix(sigmas_by_n, args.n0, args.n1)

    csv_path = args.out_prefix.with_suffix(".csv")
    pdf_path = args.out_prefix.with_suffix(".pdf")
    txt_path = args.out_prefix.with_suffix(".txt")

    save_csv(rows, csv_path)
    make_plot(rows, fit, pdf_path, args.title)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("kappa_mix estimation\n")
        f.write("====================\n\n")
        f.write(f"input: {args.input}\n")
        f.write(f"window: [{args.n0}, {args.n1}]\n")
        f.write(f"num_points: {len(rows)}\n\n")
        f.write("OLS log-log fit on V_n\n")
        f.write("----------------------\n")
        f.write(f"slope = {fit['slope']:.10f}\n")
        f.write(f"intercept = {fit['intercept']:.10f}\n")
        f.write(f"R^2 = {fit['r2']:.10f}\n")
        f.write(f"alpha_mix = {alpha_mix:.10f}\n")
        f.write(f"kappa_mix = {kappa_mix:.10f}\n\n")
        f.write("Jackknife over blocks\n")
        f.write("---------------------\n")
        f.write(f"jackknife_reps = {jk['jackknife_reps']}\n")
        f.write(f"blocks_used = {jk['blocks_used']}\n")
        f.write(f"alpha_mean = {jk['alpha_mean']:.10f}\n")
        f.write(f"alpha_std = {jk['alpha_std']:.10f}\n")
        f.write(f"kappa_mean = {jk['kappa_mean']:.10f}\n")
        f.write(f"kappa_std = {jk['kappa_std']:.10f}\n")
        f.write(f"R^2_mean = {jk['r2_mean']:.10f}\n")

    print()
    print("kappa_mix estimation")
    print("====================")
    print(f"input           : {args.input}")
    print(f"window          : [{args.n0}, {args.n1}]")
    print(f"num_points      : {len(rows)}")
    print(f"alpha_mix       : {alpha_mix:.6f}")
    print(f"kappa_mix       : {kappa_mix:.6f}")
    print(f"R^2             : {fit['r2']:.6f}")
    print(f"jackknife alpha : {jk['alpha_mean']:.6f} ± {jk['alpha_std']:.6f}")
    print(f"jackknife kappa : {jk['kappa_mean']:.6f} ± {jk['kappa_std']:.6f}")
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())