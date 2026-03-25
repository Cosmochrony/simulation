#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_Q = np.array([29, 61, 101, 151], dtype=float)
DEFAULT_VMAX = np.array([5.20, 2.75, 1.60, 0.65], dtype=float)


def load_json(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "q_values" in raw and "vmax_values" in raw:
            q = np.asarray(raw["q_values"], dtype=float)
            vmax = np.asarray(raw["vmax_values"], dtype=float)
            return q, vmax

        pairs = sorted((float(k), float(v)) for k, v in raw.items())
        q = np.asarray([p[0] for p in pairs], dtype=float)
        vmax = np.asarray([p[1] for p in pairs], dtype=float)
        return q, vmax

    raise ValueError("JSON must be either a dict {q: vmax} or contain q_values/vmax_values")


def load_csv(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    q_vals = []
    vmax_vals = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "q" not in reader.fieldnames or "vmax" not in reader.fieldnames:
            raise ValueError("CSV must contain columns 'q' and 'vmax'")

        for row in reader:
            q_vals.append(float(row["q"]))
            vmax_vals.append(float(row["vmax"]))

    return np.asarray(q_vals, dtype=float), np.asarray(vmax_vals, dtype=float)


def load_npz(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)

    if "q_values" not in z or "vmax_values" not in z:
        raise ValueError("NPZ must contain arrays 'q_values' and 'vmax_values'")

    q = np.asarray(z["q_values"], dtype=float)
    vmax = np.asarray(z["vmax_values"], dtype=float)
    return q, vmax


def load_input(path: pathlib.Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return DEFAULT_Q.copy(), DEFAULT_VMAX.copy()

    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".npz":
        return load_npz(path)

    raise ValueError("Supported input formats are .json, .csv, .npz")


def validate_data(q: np.ndarray, vmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if q.ndim != 1 or vmax.ndim != 1:
        raise ValueError("q and vmax must be 1D arrays")
    if len(q) != len(vmax):
        raise ValueError("q and vmax must have the same length")
    if len(q) < 2:
        raise ValueError("Need at least 2 points")
    if np.any(q <= 0) or np.any(vmax <= 0):
        raise ValueError("All q and vmax values must be strictly positive")

    order = np.argsort(q)
    return q[order], vmax[order]


def loglog_fit(q: np.ndarray, vmax: np.ndarray) -> dict:
    x = np.log(q)
    y = np.log(vmax)

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    beta_mix = -float(slope)
    c_prefactor = float(np.exp(intercept))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "beta_mix": beta_mix,
        "C": c_prefactor,
        "r2": float(r2),
        "fitted_vmax": np.exp(y_hat),
        "residuals_log": y - y_hat,
    }


def leave_one_out_fit(q: np.ndarray, vmax: np.ndarray) -> dict:
    if len(q) < 3:
        return {
            "beta_mean": math.nan,
            "beta_std": math.nan,
            "C_mean": math.nan,
            "C_std": math.nan,
            "r2_mean": math.nan,
            "reps": 0,
        }

    betas = []
    cs = []
    r2s = []

    for i in range(len(q)):
        mask = np.ones(len(q), dtype=bool)
        mask[i] = False
        fit = loglog_fit(q[mask], vmax[mask])
        betas.append(fit["beta_mix"])
        cs.append(fit["C"])
        r2s.append(fit["r2"])

    betas = np.asarray(betas, dtype=float)
    cs = np.asarray(cs, dtype=float)
    r2s = np.asarray(r2s, dtype=float)

    return {
        "beta_mean": float(np.mean(betas)),
        "beta_std": float(np.std(betas, ddof=1)),
        "C_mean": float(np.mean(cs)),
        "C_std": float(np.std(cs, ddof=1)),
        "r2_mean": float(np.mean(r2s)),
        "reps": int(len(betas)),
    }


def write_csv(q: np.ndarray, vmax: np.ndarray, fit: dict, out_csv: pathlib.Path) -> None:
    fitted = fit["fitted_vmax"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q", "vmax", "fitted_vmax", "log_residual"])
        for qi, vi, fi, ri in zip(q, vmax, fitted, fit["residuals_log"]):
            writer.writerow(
                [
                    f"{qi:.0f}",
                    f"{vi:.16e}",
                    f"{fi:.16e}",
                    f"{ri:.16e}",
                ]
            )


def make_plot(
    q: np.ndarray,
    vmax: np.ndarray,
    fit: dict,
    out_pdf: pathlib.Path,
    title: str,
) -> None:
    plt.figure(figsize=(6.6, 4.8))
    plt.loglog(q, vmax, "o", label=r"Measured $V_n^{\max}(q)$")

    q_fit = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), 300)
    vmax_fit = fit["C"] * q_fit ** (-fit["beta_mix"])

    plt.loglog(
        q_fit,
        vmax_fit,
        "--",
        label=(
            rf"Fit: $V_n^{{\max}}(q)\approx {fit['C']:.2f}\,q^{{-{fit['beta_mix']:.3f}}}$"
            "\n"
            rf"$R^2={fit['r2']:.4f}$"
        ),
    )

    for qi, vi in zip(q, vmax):
        plt.annotate(
            f"{int(qi)}",
            (qi, vi),
            textcoords="offset points",
            xytext=(5, 4),
        )

    plt.xlabel(r"Prime $q$")
    plt.ylabel(r"$V_n^{\max}(q)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def write_report(
    q: np.ndarray,
    vmax: np.ndarray,
    fit: dict,
    loo: dict,
    out_txt: pathlib.Path,
    source_desc: str,
) -> None:
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Vmax(q) power-law fit\n")
        f.write("====================\n\n")
        f.write(f"source: {source_desc}\n")
        f.write(f"num_points: {len(q)}\n\n")

        f.write("Input data\n")
        f.write("----------\n")
        for qi, vi in zip(q, vmax):
            f.write(f"q={int(qi):>4d}   vmax={vi:.10f}\n")

        f.write("\nLog-log fit\n")
        f.write("-----------\n")
        f.write(f"log(Vmax) = intercept + slope * log(q)\n")
        f.write(f"slope      = {fit['slope']:.10f}\n")
        f.write(f"intercept  = {fit['intercept']:.10f}\n")
        f.write(f"beta_mix   = {fit['beta_mix']:.10f}\n")
        f.write(f"C          = {fit['C']:.10f}\n")
        f.write(f"R^2        = {fit['r2']:.10f}\n")
        f.write("\nEmpirical law:\n")
        f.write(f"Vmax(q) ≈ {fit['C']:.6f} * q^(-{fit['beta_mix']:.6f})\n")

        if loo["reps"] > 0:
            f.write("\nLeave-one-out robustness\n")
            f.write("------------------------\n")
            f.write(f"reps       = {loo['reps']}\n")
            f.write(f"beta_mean  = {loo['beta_mean']:.10f}\n")
            f.write(f"beta_std   = {loo['beta_std']:.10f}\n")
            f.write(f"C_mean     = {loo['C_mean']:.10f}\n")
            f.write(f"C_std      = {loo['C_std']:.10f}\n")
            f.write(f"R^2_mean   = {loo['r2_mean']:.10f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Vmax(q) ~ C q^{-beta_mix} from O13-style variance data"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional input file (.json, .csv, .npz). "
            "If omitted, uses O13 default data: "
            "(29,5.20), (61,2.75), (101,1.60), (151,0.65)."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        type=pathlib.Path,
        default=pathlib.Path("vmax_q_fit"),
        help="Output prefix for CSV/PDF/TXT",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=r"Peak inter-block variance decay $V_n^{\max}(q)$",
        help="Plot title",
    )

    args = parser.parse_args()

    q, vmax = load_input(args.input)
    q, vmax = validate_data(q, vmax)

    fit = loglog_fit(q, vmax)
    loo = leave_one_out_fit(q, vmax)

    csv_path = args.out_prefix.with_suffix(".csv")
    pdf_path = args.out_prefix.with_suffix(".pdf")
    txt_path = args.out_prefix.with_suffix(".txt")

    source_desc = str(args.input) if args.input is not None else "built-in O13 default data"

    write_csv(q, vmax, fit, csv_path)
    make_plot(q, vmax, fit, pdf_path, args.title)
    write_report(q, vmax, fit, loo, txt_path, source_desc)

    print()
    print("Vmax(q) power-law fit")
    print("====================")
    print(f"source     : {source_desc}")
    print(f"num_points : {len(q)}")
    print(f"beta_mix   : {fit['beta_mix']:.6f}")
    print(f"C          : {fit['C']:.6f}")
    print(f"R^2        : {fit['r2']:.6f}")
    if loo["reps"] > 0:
        print(f"LOO beta   : {loo['beta_mean']:.6f} ± {loo['beta_std']:.6f}")
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())