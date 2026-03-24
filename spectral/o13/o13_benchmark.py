"""
o13_benchmark.py
O13 Pre-Benchmark: Scalability and Measurability Test
=====================================================
Cosmochrony Admissibility Sub-Programme -- Pre-decision script for O13.

Answers the four decision questions before committing to O13-A
(asymptotic regime extension):

  Q1. Does q=101 terminate in reasonable wall time?
  Q2. Is a genuine pre-saturation window measurable (>= 5 data points)?
  Q3. Is inter-block variance controlled?
  Q4. Is extension to q=151 or q=211 realistic?

REQUIRES: spectral_O12.py in the same directory (or on PYTHONPATH).
This script reuses the exact O12 computation kernel without modification.

Usage:
  python o13_benchmark.py
  python o13_benchmark.py --quick
  python o13_benchmark.py --add-q 151
  python o13_benchmark.py --add-q 211 --M 5
  python o13_benchmark.py --add-q 151 --add-q 211 --M 10

Outputs (saved in current directory):
  o13_benchmark_report.txt
  o13_benchmark_figure.pdf
"""

import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from spectral_O12 import (
        build_generators,
        bfs_shells,
        sample_generic_blocks,
        compute_block_capacity,
        find_fitting_window,
        ols_loglog,
    )
except ImportError:
    print("ERROR: spectral_O12.py not found. Place it in the same directory.")
    sys.exit(1)

RNG_BASE = 42


# ---------------------------------------------------------------------------
# Pilot configurations
# ---------------------------------------------------------------------------
# Each entry: (q, M_block, n_max, bfs_frac)
#   M_block : number of generic blocks
#   n_max   : BFS depth cap
#   bfs_frac: fraction of group explored by BFS

DEFAULT_PILOTS = [
    (29, 50, 8, 0.99),     # paper parameters -- validates against O12
    (61, 15, 10, 0.99),    # paper parameters
    (101, 20, 20, 0.99),   # pilot target
]

QUICK_PILOTS = [
    (29, 20, 8, 0.99),
    (61, 8, 10, 0.99),
    (101, 5, 15, 0.99),
]


# ---------------------------------------------------------------------------
# Helper functions for added primes
# ---------------------------------------------------------------------------

def extra_prime_params(q, m_override=None):
    """
    Conservative but realistic parameters for added primes.

    q=151:
      - bfs_frac ~ 1e6 / q^3 ≈ 0.29
      - n_max chosen to comfortably cover the fit window

    q=211:
      - bfs_frac ~ 1e6 / q^3 ≈ 0.106
      - n_max increased modestly to capture the longer pre-saturation window
    """
    if m_override is None:
        m_block = 10
    else:
        m_block = m_override

    if q <= 151:
        n_max = 37
    elif q <= 211:
        n_max = 50
    else:
        n_max = max(50, int(0.25 * q))

    bfs_frac = max(0.10, 1e6 / (q ** 3))
    return (q, m_block, n_max, bfs_frac)


def deduplicate_pilots(pilots):
    """
    Keep the last occurrence of each q so that command-line additions override defaults.
    """
    by_q = {}
    for entry in pilots:
        by_q[entry[0]] = entry
    return [by_q[q] for q in sorted(by_q)]


# ---------------------------------------------------------------------------
# Run one prime
# ---------------------------------------------------------------------------

def run_prime(q, m_block, n_max, bfs_frac, label=""):
    print(f"\n{'=' * 60}")
    print(
        f"  q={q}, M_block={m_block}, n_max={n_max}, bfs_frac={bfs_frac}"
        + (f"  [{label}]" if label else "")
    )
    print(f"{'=' * 60}")
    sys.stdout.flush()

    rng = np.random.default_rng(RNG_BASE + q)
    gens = build_generators(q)

    t_bfs = time.perf_counter()
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    t_bfs = time.perf_counter() - t_bfs
    n_shells = len(shells)
    n_nodes = sum(len(s) for s in shells)

    print(f"  BFS: {n_shells} shells, {n_nodes} nodes, {t_bfs:.2f}s")
    sys.stdout.flush()

    blocks = sample_generic_blocks(q, m_block, rng)

    all_sigma = []
    t_block_times = []
    t_blocks_total = time.perf_counter()

    for i, c in enumerate(blocks):
        t0 = time.perf_counter()
        sv, _, _, _ = compute_block_capacity(shells, c, q, gens, n_max=n_max)
        dt = time.perf_counter() - t0
        t_block_times.append(dt)

        pad = n_shells - len(sv)
        if pad > 0:
            sv = np.concatenate([sv, np.zeros(pad)])
        all_sigma.append(sv[:n_shells])

        elapsed = time.perf_counter() - t_blocks_total
        sys.stdout.write(
            f"\r  Block {i + 1}/{m_block}, {dt:.1f}s/block, elapsed {elapsed:.0f}s"
        )
        sys.stdout.flush()

    t_blocks_total = time.perf_counter() - t_blocks_total
    print(
        f"\n  Blocks total: {t_blocks_total:.1f}s, "
        f"mean/block: {np.mean(t_block_times):.2f}s"
    )
    sys.stdout.flush()

    arr = np.array(all_sigma)
    sigma_bar = arr.mean(axis=0)
    sigma_var = arr.var(axis=0)
    ns = np.arange(n_shells)
    V_n = np.where(sigma_bar > 1e-15, sigma_var / sigma_bar ** 2, np.nan)

    # Fit window and OLS
    n0, n1 = find_fitting_window(ns[1:], sigma_bar[1:], q)
    delta_hat, C_hat, r2 = ols_loglog(ns, sigma_bar, n0, n1)
    mask_win = (ns >= n0) & (ns <= n1) & (sigma_bar > 0)
    n_pts = int(mask_win.sum())
    v_max = float(np.nanmax(V_n[mask_win])) if mask_win.any() else np.nan

    e1_ok = n_pts >= 4
    e2_ok = (not np.isnan(v_max)) and (v_max < 1.0)
    e3_ok = (not np.isnan(r2)) and (r2 >= 0.97)

    print(
        f"  delta_hat={delta_hat:.3f}, R2={r2:.4f}, "
        f"window=[{n0},{n1}] ({n_pts} pts)"
    )
    print(f"  E1={e1_ok}, E2={e2_ok} (Vmax={v_max:.2f}), E3={e3_ok}")

    return dict(
        q=q,
        m_block=m_block,
        n_max=n_max,
        bfs_frac=bfs_frac,
        label=label,
        n_shells=n_shells,
        n_nodes=n_nodes,
        shells=shells,
        ns=ns,
        sigma_bar=sigma_bar,
        sigma_var=sigma_var,
        V_n=V_n,
        all_sigma=arr,
        n0=n0,
        n1=n1,
        delta_hat=delta_hat,
        C_hat=C_hat,
        r2=r2,
        n_pts=n_pts,
        v_max=v_max,
        e1_ok=e1_ok,
        e2_ok=e2_ok,
        e3_ok=e3_ok,
        t_bfs=t_bfs,
        t_blocks=t_blocks_total,
        t_per_block=float(np.mean(t_block_times)),
    )


# ---------------------------------------------------------------------------
# Cost extrapolation
# ---------------------------------------------------------------------------

def extrapolate_cost(results):
    """
    Extrapolate cost to larger primes.
    Observed scaling: T_per_block ~ q^5
    (O(q^4) per BFS step * O(q) steps).

    Uses the largest measured q as anchor.
    """
    valid = [r for r in results if r is not None]
    if not valid:
        return "No valid results for extrapolation."

    anchor = sorted(valid, key=lambda r: r["q"], reverse=True)[0]
    q0 = anchor["q"]
    t0 = anchor["t_per_block"]

    lines = [
        "",
        f"Cost extrapolation (anchor: q={q0}, {t0:.2f}s/block, T ~ q^5)",
        "-" * 78,
        f"{'q':>5}  {'|Gq|':>10}  {'t/block(s)':>12}  "
        f"{'M=5 (s)':>10}  {'M=10 (s)':>10}  {'M=20 (s)':>10}",
        "-" * 78,
    ]

    for r in valid:
        q = r["q"]
        t = r["t_per_block"]
        lines.append(
            f"{q:>5}  {r['n_nodes']:>10}  {t:>12.2f}  "
            f"{t * 5:>10.0f}  {t * 10:>10.0f}  {t * 20:>10.0f}  [measured]"
        )

    measured_qs = {r["q"] for r in valid}
    for qe in [101, 151, 211, 307]:
        if qe in measured_qs:
            continue
        scale = (qe / q0) ** 5
        te = t0 * scale
        lines.append(
            f"{qe:>5}  {qe ** 3:>10}  {te:>12.1f}  "
            f"{te * 5:>10.0f}  {te * 10:>10.0f}  {te * 20:>10.0f}  [extrap.]"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision summary
# ---------------------------------------------------------------------------

def decision_summary(results):
    lines = ["\n" + "=" * 65, "  O13 BENCHMARK DECISION SUMMARY", "=" * 65]

    for r in results:
        if r is None:
            continue

        q = r["q"]
        delta = r["delta_hat"]
        r2 = r["r2"]
        n_pts = r["n_pts"]
        v_max = r["v_max"]
        t_pb = r["t_per_block"]

        q1 = (
            "YES" if t_pb * 20 < 3600
            else "BORDERLINE" if t_pb * 20 < 14400
            else "NO"
        )
        q2 = "YES" if n_pts >= 5 else "MARGINAL" if n_pts >= 3 else "NO"
        q3_label = "YES (E2 sat.)" if v_max < 1.0 else "MEAN_ONLY (E2 viol.)"

        t_q151 = t_pb * (151 / q) ** 5 * 20
        q4 = (
            "YES" if t_q151 < 7200
            else f"BORDERLINE ({t_q151:.0f}s)" if t_q151 < 86400
            else f"NO ({t_q151:.0f}s)"
        )

        lines += [
            f"\n  q = {q}",
            f"    delta_hat = {delta:.3f} (O12 range: 4.28-4.80),  R2 = {r2:.4f}",
            f"    Q1 [M=20 < 1h]:         {q1}  ({t_pb * 20:.0f}s est.)",
            f"    Q2 [window >= 5 pts]:   {q2}  ({n_pts} pts, [{r['n0']},{r['n1']}])",
            f"    Q3 [Vn < 1.0]:          {q3_label}  (Vmax = {v_max:.2f})",
            f"    Q4 [q=151 M=20 < 2h]:   {q4}",
        ]

    r101 = next((r for r in results if r is not None and r["q"] == 101), None)

    lines += ["", "=" * 65, "  EDITORIAL RECOMMENDATION:"]
    if r101 is None:
        lines.append("  q=101 result unavailable; re-run with default pilots.")
    elif r101["n_pts"] >= 5 and r101["t_per_block"] * 20 < 14400:
        lines += [
            "  O13-A (asymptotic regime) is VIABLE.",
            "  Recommended O13 scope:",
            "    - q in {101, 151} with M_block = 20 if available",
            "    - q=211 with M_block = 5--10 as a robustness extension",
            "    - report delta_exact(q) convergence plot as primary result",
            "    - fit scaling law delta(q) = delta_inf + a/q^alpha",
            "  If delta_exact(q) remains flat (no drift toward [7.4,10.6]),",
            "  O13 becomes a transition paper: convergence of delta_exact,",
            "  robust tension, and motivation for O14.",
        ]
    elif r101["n_pts"] >= 3:
        lines += [
            "  O13-A is MARGINALLY viable at q=101 only.",
            "  Consider a transition paper: q=101 exact result,",
            "  honest about insufficient window length and finite-size.",
            "  Recommend optimising compute before committing to q=151/211.",
        ]
    else:
        lines += [
            "  O13-A is NOT YET viable.",
            "  Window too short at q=101; measurement is unreliable.",
            "  Recommend code optimisation before deciding O13 scope.",
        ]

    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "purple"]

    for idx, r in enumerate(results):
        if r is None:
            continue

        q = r["q"]
        ns = r["ns"]
        sb = r["sigma_bar"]
        n0, n1 = r["n0"], r["n1"]
        delta = r["delta_hat"]
        C = r["C_hat"]
        col = colors[idx % len(colors)]

        pos = sb > 1e-10
        ax1.loglog(
            ns[pos][1:],
            sb[pos][1:],
            "o",
            color=col,
            ms=4,
            label=f"$q={q}$ mean (M={r['m_block']})",
        )
        if not np.isnan(delta):
            ns_fit = np.linspace(n0, n1, 100)
            ax1.loglog(
                ns_fit,
                C * ns_fit ** (-delta),
                "--",
                color=col,
                lw=1.5,
                label=f"fit: $\\hat{{\\delta}}={delta:.2f}$, $R^2={r['r2']:.3f}$",
            )
        ax1.axvline(n0, color=col, ls=":", lw=0.8, alpha=0.5)
        ax1.axvline(n1, color=col, ls=":", lw=0.8, alpha=0.5)

        Vn = r["V_n"]
        mask2 = sb > 1e-5
        if mask2.any():
            ax2.semilogy(
                ns[mask2],
                np.where(Vn[mask2] > 0, Vn[mask2], np.nan),
                "-",
                color=col,
                label=f"$q={q}$",
            )

    ax1.set_xlabel("BFS depth $n$", fontsize=11)
    ax1.set_ylabel(r"$\bar\Sigma_n$", fontsize=11)
    ax1.set_title(r"Exact Weil capacity decay $\bar\Sigma_n$ --- O13 benchmark", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="E2 threshold $\\varepsilon_E = 1$")
    ax2.set_xlabel("BFS depth $n$", fontsize=11)
    ax2.set_ylabel(r"$V_n = \mathrm{Var}_c(\Sigma_n^{(c)}) / \bar\Sigma_n^2$", fontsize=11)
    ax2.set_title("Inter-block variance (E2) --- O13 benchmark", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="O13 Pre-Benchmark for SpectralO12 extension."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced M_block for a faster pass.",
    )
    parser.add_argument(
        "--add-q",
        type=int,
        action="append",
        default=[],
        metavar="Q",
        help="Add an extra prime Q (repeatable, e.g. --add-q 151 --add-q 211).",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Override M_block for added primes only (e.g. --M 5).",
    )
    args = parser.parse_args()

    pilots = list(QUICK_PILOTS if args.quick else DEFAULT_PILOTS)

    for q_extra in args.add_q:
        pilots.append(extra_prime_params(q_extra, m_override=args.M))

    pilots = deduplicate_pilots(pilots)

    print("O13 Pre-Benchmark")
    print("=" * 65)
    print(f"Mode: {'quick' if args.quick else 'default'}")
    print(f"Pilots: q in {[p[0] for p in pilots]}")
    print("=" * 65)

    results = []
    for q, m_block, n_max, bfs_frac in pilots:
        label = "added" if q not in {29, 61, 101} else ""
        r = run_prime(q, m_block, n_max, bfs_frac, label=label)
        results.append(r)

    cost_text = extrapolate_cost(results)
    print(cost_text)

    decision_text = decision_summary(results)
    print(decision_text)

    make_figure(results, "o13_benchmark_figure.pdf")

    with open("o13_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("O13 Pre-Benchmark Report\n")
        f.write("=" * 65 + "\n\n")
        for r in results:
            if r is None:
                f.write("[FAILED]\n\n")
                continue
            f.write(
                f"q = {r['q']}  (M={r['m_block']}, n_max={r['n_max']}, "
                f"bfs_frac={r['bfs_frac']})\n"
            )
            f.write(f"  delta_hat  = {r['delta_hat']:.4f}\n")
            f.write(f"  R2         = {r['r2']:.4f}\n")
            f.write(f"  window     = [{r['n0']},{r['n1']}], {r['n_pts']} pts\n")
            f.write(
                f"  V_n max    = {r['v_max']:.3f}  "
                f"(E2: {'OK' if r['e2_ok'] else 'violated'})\n"
            )
            f.write(f"  t/block    = {r['t_per_block']:.2f}s\n")
            f.write(f"  t_blocks   = {r['t_blocks']:.1f}s\n\n")

        f.write(cost_text + "\n")
        f.write(decision_text + "\n")

    print("\nDone. Outputs: o13_benchmark_report.txt, o13_benchmark_figure.pdf")


if __name__ == "__main__":
    main()