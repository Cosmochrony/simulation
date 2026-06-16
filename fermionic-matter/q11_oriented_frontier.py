#!/usr/bin/env python3
"""
q11_oriented_frontier.py

Tests whether the recursive cascade produces a NON-ZERO, BIAS-FREE oriented angular
signal -- the honest exit from the symmetric-capacity obstruction (where the symmetric
shell average of the central phase A_c vanishes identically).

The observable is NOT a shell average. It is a frontier transfer observable attached to
the directed outgoing edges g -> gs of the BFS cascade.
For a single element g=(a,b,z) the central phase under the central character c is
A_c(g)=c z, so along one outward step the Heisenberg law gives
    Delta A_c(g,s) = c ( z(s) + a(g) s_b ) = c a(g) s_b   (z(s)=0 for the generators),
the discrete 1/2[X,Y] cocycle = the dynamical J_3 carried by the oriented edge.

Frontiers (canonical once the origin e and admissible generators are fixed):
    d+ S_m = { (g,s): d(e,g)=m, d(e,gs)=m+1 }   (outgoing)
    d- S_m = { (g,s): d(e,g)=m, d(e,gs)=m-1 }   (incoming)

PROTOCOL (the value of the amplitude is NOT computed here):
    theta_plus_raw(m)  = < 2pi/q * Delta A_c >_{d+ S_m}     test 1: != 0 ?  (may fail)
    theta_minus_raw(m) = < 2pi/q * Delta A_c >_{d- S_m}     consistency: ~ -theta_plus
    theta_sym_raw(m)   = < 2pi/q * Delta A_c >_{d+ U d-}     ANTI-BIAS: must be 0

A valid oriented signal must satisfy BOTH
    theta_plus_raw != 0   AND   theta_sym_raw = 0.
The sign reversal alone is nearly tautological; the symmetrised cancellation is the
discriminant that attributes the signal to recursive orientation, not to sampling bias.

MAXIMAL-LOCKING BOUND (theta_max): the Ihat-normalised accumulation of the absolute
frontier increment < (2pi/q)|Delta A_c| >_{d+}. Since phi sends Delta A_c -> -Delta A_c,
|Delta A_c| is phi-even and its d+ mean equals its d+/phi mean. theta_max bounds the true
chiral signal, |Theta_chi(n)| <= theta_max(n), with equality only under the chiral-area
locking [H-orient] (sigma_L(e) = sign Delta A_c(e)). It is a structural bound, not epsilon.

GUARDRAILS: no N_A, no 1/10, no epsilon anywhere in this script.

Outputs (current directory, per q):
  q11_frontier_q{q}_profile.csv
  q11_frontier_q{q}_summary.json
  q11_frontier_q{q}_checkpoint.jsonl
  q11_frontier_q{q}.pdf
"""

import os
import json
import time
import math
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

# --------------------------------------------------------------------------- #
# PARAMETERS
# --------------------------------------------------------------------------- #
Q_LIST = [61, 101, 151]
N_MAX_CAP = 14                   # BFS depth for the frontier profile
N_BLOCKS = 24                    # blocks (c1,c2,c3) for the radial capacity sigma -> Ihat
C_CENTRAL = 1                    # central character scale c (other c rescale linearly)
SAT_FRACTION = 0.99              # n_3^obs = first n with Ihat(n) >= SAT_FRACTION
TOL = 1e-9                       # zero tolerance for the status flags
RNG_SEED = 20260607
N_WORKERS = max(1, cpu_count() - 1)

PROFILE_TMPL = "q11_frontier_q{q}_profile.csv"
SUMMARY_TMPL = "q11_frontier_q{q}_summary.json"
CKPT_TMPL = "q11_frontier_q{q}_checkpoint.jsonl"
PDF_TMPL = "q11_frontier_q{q}.pdf"


# --------------------------------------------------------------------------- #
# Conventions: import from the project pipeline if available, else local fallback
# --------------------------------------------------------------------------- #
try:
    from spectral_O12 import build_generators as _build_generators
    from spectral_O12 import heisenberg_mul_batch as _hmul_batch
    PIPELINE = "spectral_O12"
except Exception:
    PIPELINE = "local-fallback"

    def _build_generators(q):
        return [(1, 0, 0), (0, 1, 0), ((-1) % q, 0, 0), (0, (-1) % q, 0)]

    def _hmul_batch(elts, s, q):
        a, b, z = elts[:, 0], elts[:, 1], elts[:, 2]
        sa, sb, sz = s
        return np.stack([(a + sa) % q, (b + sb) % q, (z + sz + a * sb) % q], axis=1)


def signed_mod(v, q):
    """Map v in [0,q) to the symmetric range (-q/2, q/2]; the oriented (signed) value."""
    return ((v + q // 2) % q) - q // 2


def heisenberg_mul(g, s, q):
    a, b, z = g
    sa, sb, sz = s
    return ((a + sa) % q, (b + sb) % q, (z + sz + a * sb) % q)


def bfs_with_distance(q, n_max):
    """BFS to depth n_max+1; return shells (list of node lists) and dist dict."""
    gens = _build_generators(q)
    e = (0, 0, 0)
    dist = {e: 0}
    shells = [[e]]
    frontier = [e]
    for n in range(1, n_max + 2):                 # +1 so edges from shell n_max see their target
        nxt = []
        for g in frontier:
            for s in gens:
                h = heisenberg_mul(g, s, q)
                if h not in dist:
                    dist[h] = n
                    nxt.append(h)
        if not nxt:
            break
        shells.append(nxt)
        frontier = nxt
    return shells, dist, gens


# --------------------------------------------------------------------------- #
# Frontier transfer observable  (central cocycle, c = C_CENTRAL)
# --------------------------------------------------------------------------- #
def frontier_profiles(q, shells, dist, gens, n_max):
    """Per shell m: signed means over d+, d-, d+ U d- frontiers, plus the absolute
    (maximal-locking) mean of |Delta A_c| over d+.

    The absolute mean is the maximal-locking bound: since the residual reflection phi
    sends Delta A_c -> -Delta A_c, |Delta A_c| is phi-even, so its mean over d+ equals
    its mean over the quotient d+/phi. It bounds |Theta_chi| and is attained only under
    the chiral-area locking [H-orient]; it is NOT an amplitude (no N_A, no epsilon).
    """
    c = C_CENTRAL
    two_pi_q = 2.0 * math.pi / q
    th_plus, th_minus, th_sym, th_max = [], [], [], []
    counts = []
    for m in range(min(n_max + 1, len(shells))):
        dplus, dminus = [], []
        for g in shells[m]:
            zg = g[2]
            for s in gens:
                gs = heisenberg_mul(g, s, q)
                dm = dist.get(gs, None)
                if dm is None:
                    continue
                delta = signed_mod(c * ((gs[2] - zg) % q), q)   # central increment = c a(g) s_b
                if dm == m + 1:
                    dplus.append(delta)
                elif dm == m - 1:
                    dminus.append(delta)
        dp = np.array(dplus, dtype=float)
        dn = np.array(dminus, dtype=float)
        ds = np.concatenate([dp, dn]) if (dp.size + dn.size) else np.array([])
        th_plus.append(two_pi_q * dp.mean() if dp.size else 0.0)
        th_minus.append(two_pi_q * dn.mean() if dn.size else 0.0)
        th_sym.append(two_pi_q * ds.mean() if ds.size else 0.0)
        th_max.append(two_pi_q * np.abs(dp).mean() if dp.size else 0.0)
        counts.append((int(dp.size), int(dn.size)))
    return (np.array(th_plus), np.array(th_minus), np.array(th_sym),
            np.array(th_max), counts)


# --------------------------------------------------------------------------- #
# Radial capacity sigma_pair -> Ihat  (distinct-B frequency counting over blocks)
# --------------------------------------------------------------------------- #
def sigma_block(args):
    block, q, shells, n_max = args
    c1, c2, c3 = block
    gens = _build_generators(q)
    triples = [(s1, s2, s3) for s1 in gens for s2 in gens for s3 in gens]
    seen = set()
    sigma = []
    for m in range(min(n_max + 1, len(shells))):
        sh = np.array(shells[m], dtype=np.int64)
        before = len(seen)
        for (s1, s2, s3) in triples:
            e1 = _hmul_batch(sh, s1, q)
            e2 = _hmul_batch(e1, s2, q)
            e3 = _hmul_batch(e2, s3, q)
            B = (c1 * e1[:, 1] + c2 * e2[:, 1] + c3 * e3[:, 1]) % q
            seen.update(B.tolist())
        sigma.append((len(seen) - before) / len(sh))
    return {"block": [int(x) for x in block], "sigma": sigma}


def ihat_from_sigma(sigma_list, n_max):
    L = min(len(s) for s in sigma_list)
    sig = np.mean([np.array(s[:L]) for s in sigma_list], axis=0)
    sp = sig ** 2
    sp = sp / sp[0]
    I_raw = sp[0] - sp
    if I_raw[-1] <= 0:
        n_sat = L - 1
    else:
        It = I_raw / I_raw[-1]
        idx = np.where(It >= SAT_FRACTION)[0]
        n_sat = int(idx[0]) if len(idx) else L - 1
    n_sat = max(2, min(n_sat, L - 1))
    denom = sp[0] - sp[n_sat]
    Ihat = (sp[0] - sp) / denom if denom != 0 else np.zeros(L)
    return sp, Ihat, n_sat, L


# --------------------------------------------------------------------------- #
# Driver with checkpoints, progress/ETA
# --------------------------------------------------------------------------- #
def run_q(q):
    rng = np.random.default_rng(RNG_SEED + q)
    n_max = min(N_MAX_CAP, q // 2)
    t0 = time.time()
    print(f"[q={q}] pipeline={PIPELINE}  BFS depth {n_max}(+1) ...", flush=True)
    shells, dist, gens = bfs_with_distance(q, n_max)
    print(f"[q={q}] shells={len(shells)} nodes={len(dist)} ({time.time()-t0:.1f}s)", flush=True)

    th_plus, th_minus, th_sym, th_max, counts = frontier_profiles(q, shells, dist, gens, n_max)
    print(f"[q={q}] frontier profiles done ({time.time()-t0:.1f}s)", flush=True)

    # radial capacity over generic blocks (for Ihat)
    blocks = []
    while len(blocks) < min(N_BLOCKS, q - 1):
        b = tuple(int(rng.integers(1, q)) for _ in range(3))
        if sum(b) % q != 0:
            blocks.append(b)

    ckpt = CKPT_TMPL.format(q=q)
    done = {}
    if os.path.exists(ckpt):
        stale = False
        recs = []
        with open(ckpt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("pipeline") != PIPELINE:   # missing field or different pipeline -> stale
                    stale = True
                    break
                recs.append(r)
        if stale:
            bad = ckpt + ".stale"
            os.replace(ckpt, bad)
            print(f"[q={q}] WARNING: checkpoint pipeline != '{PIPELINE}'; discarding stale "
                  f"checkpoint -> {bad}; recomputing sigma", flush=True)
        else:
            for r in recs:
                done[tuple(r["block"])] = r["sigma"]
            print(f"[q={q}] resumed {len(done)} blocks (pipeline={PIPELINE})", flush=True)

    todo = [b for b in blocks if b not in done]
    args = [(b, q, shells, n_max) for b in todo]
    sig_list = list(done.values())
    t1 = time.time()
    if args:
        with Pool(N_WORKERS) as pool, open(ckpt, "a") as fh:
            for i, r in enumerate(pool.imap_unordered(sigma_block, args), 1):
                r["pipeline"] = PIPELINE                # stamp so a later run can detect mismatch
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                sig_list.append(r["sigma"])
                el = time.time() - t1
                eta = el / i * (len(args) - i)
                print(f"[q={q}] sigma block {i}/{len(args)}  elapsed={el:5.1f}s  ETA={eta:5.1f}s",
                      flush=True)

    sp, Ihat, n_sat, L = ihat_from_sigma(sig_list, n_max)
    L = min(L, len(th_plus))
    th_plus, th_minus, th_sym, th_max = th_plus[:L], th_minus[:L], th_sym[:L], th_max[:L]
    sp, Ihat = sp[:L], Ihat[:L]
    n_sat = min(n_sat, L - 1)

    # cumulative raw and capacity-normalised oriented profile
    th_plus_cum = np.cumsum(th_plus)
    dIhat = np.diff(Ihat, prepend=0.0)
    weighted = np.cumsum(th_plus * dIhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        th_q11 = np.where(Ihat > 1e-9, weighted / Ihat, np.nan)

    # maximal-locking bound: same Ihat-normalised accumulation applied to |Delta A_c|.
    # |Theta_chi(n)| <= Theta_max(n); equality holds only under [H-orient] (chiral-area
    # locking). This is a structural upper bound on the angle, NOT a value of epsilon.
    th_max_cum = np.cumsum(th_max)
    weighted_max = np.cumsum(th_max * dIhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        th_max_oriented = np.where(Ihat > 1e-9, weighted_max / Ihat, np.nan)

    # status flags
    sign_ok = bool(np.allclose(th_minus, -th_plus, atol=1e-6))
    sym_zero_ok = bool(np.max(np.abs(th_sym)) < TOL)
    max_plus = float(np.max(np.abs(th_plus)))
    if max_plus < TOL:
        auto_status = ("frontier re-paired by residual automorphism: "
                       "NO oriented signal (test 1 FAILS)")
    elif sym_zero_ok:
        auto_status = "oriented signal present and bias-free (test 1 passes, anti-bias passes)"
    else:
        auto_status = ("oriented signal present but symmetrised control non-zero: "
                       "HIDDEN SAMPLING BIAS (anti-bias FAILS)")

    return {
        "q": q, "n_sat": n_sat, "L": L,
        "theta_plus_raw": th_plus, "theta_minus_raw": th_minus, "theta_sym_raw": th_sym,
        "sigma_pair": sp, "Ihat": Ihat,
        "theta_plus_cumulative": th_plus_cum, "theta_q11_oriented": th_q11,
        "theta_plus_sat": float(th_plus_cum[n_sat]),
        "theta_max_raw": th_max, "theta_max_cumulative": th_max_cum,
        "theta_max_oriented": th_max_oriented,
        "theta_max_bound_sat": (float(th_max_oriented[n_sat])
                                if not math.isnan(float(th_max_oriented[n_sat])) else float("nan")),
        "theta_max_bound_sat_q": (float(th_max_oriented[n_sat]) * q
                                  if not math.isnan(float(th_max_oriented[n_sat])) else float("nan")),
        "max_abs_theta_plus_raw": max_plus,
        "max_abs_theta_sym_raw": float(np.max(np.abs(th_sym))),
        "sign_reversal_check": sign_ok,
        "sym_frontier_zero_check": sym_zero_ok,
        "automorphism_residual_status": auto_status,
        "edge_counts": counts[:L],
    }


def write_outputs(agg):
    q = agg["q"]
    with open(PROFILE_TMPL.format(q=q), "w") as fh:
        fh.write("m,theta_plus_raw,theta_minus_raw,theta_sym_raw,sigma_pair,Ihat,"
                 "theta_plus_cumulative,theta_q11_oriented,theta_max_raw,theta_max_oriented,"
                 "theta_max_oriented_times_q\n")
        for m in range(agg["L"]):
            tq = agg["theta_q11_oriented"][m]
            tq_s = "" if (isinstance(tq, float) and math.isnan(tq)) else f"{tq:.6g}"
            tmx = agg["theta_max_oriented"][m]
            tmx_s = "" if (isinstance(tmx, float) and math.isnan(tmx)) else f"{tmx:.6g}"
            tmxq_s = "" if (isinstance(tmx, float) and math.isnan(tmx)) else f"{tmx * agg['q']:.6g}"
            fh.write(f"{m},{agg['theta_plus_raw'][m]:.6g},{agg['theta_minus_raw'][m]:.6g},"
                     f"{agg['theta_sym_raw'][m]:.6g},{agg['sigma_pair'][m]:.6g},"
                     f"{agg['Ihat'][m]:.6g},{agg['theta_plus_cumulative'][m]:.6g},{tq_s},"
                     f"{agg['theta_max_raw'][m]:.6g},{tmx_s},{tmxq_s}\n")
    summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in agg.items()}
    with open(SUMMARY_TMPL.format(q=q), "w") as fh:
        json.dump(summary, fh, indent=2)
    _plot(agg)


def _plot(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = agg["q"]
    m = np.arange(agg["L"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(m, agg["theta_plus_raw"], "o-", ms=3, label=r"$\partial^+$")
    ax[0].plot(m, agg["theta_minus_raw"], "s--", ms=3, label=r"$\partial^-$")
    ax[0].plot(m, agg["theta_sym_raw"], "x:", ms=4, label=r"$\partial^+\cup\partial^-$ (control)")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_title(rf"frontier $\langle\Delta A_c\rangle$, $q={q}$")
    ax[0].set_xlabel("BFS depth $m$"); ax[0].legend(fontsize=8)
    ax[1].semilogy(m, agg["sigma_pair"], "o-", ms=3)
    ax[1].axvline(agg["n_sat"], ls=":", color="grey")
    ax[1].set_title(r"$\sigma_{\rm pair}(m)$ (radial, for $\widehat I$)")
    ax[1].set_xlabel("BFS depth $m$")
    ax[2].plot(m, agg["theta_plus_cumulative"], "o-", ms=3, label=r"$\Theta^{\rm raw}_+$ cumulative")
    ax[2].plot(m, agg["theta_q11_oriented"], "^-", ms=3, color="#a83232",
               label=r"$\Theta_{\rm Q11}^{\rm oriented}$")
    tmax = np.asarray(agg["theta_max_oriented"], dtype=float)
    ax[2].plot(m, tmax, "--", lw=1.1, color="#2a6f97", label=r"$\Theta_{\max}$ bound")
    ax[2].plot(m, -tmax, "--", lw=1.1, color="#2a6f97")
    ax[2].fill_between(m, -tmax, tmax, color="#2a6f97", alpha=0.08)
    ax[2].axvline(agg["n_sat"], ls=":", color="grey")
    ax[2].set_title(r"oriented profile + max-locking bound (no $\mathcal{N}_A$, no $\varepsilon$)")
    ax[2].set_xlabel("BFS depth $n$"); ax[2].legend(fontsize=8)
    fig.suptitle(rf"Q11 oriented frontier $q={q}$ -- {agg['automorphism_residual_status']}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PDF_TMPL.format(q=q))
    print(f"[q={q}] figure -> {PDF_TMPL.format(q=q)}", flush=True)


def _fallback_banner():
    print("!" * 70, flush=True)
    print("!! WARNING: pipeline = local-fallback (TOY generators / multiplication).", flush=True)
    print("!! These numbers are NOT the campaign values.", flush=True)
    print("!! Set PYTHONPATH so that spectral_O12 is importable to run the real pipeline.", flush=True)
    print("!" * 70, flush=True)


def final_summary(aggs):
    print("\n" + "=" * 70)
    print("Q11 ORIENTED FRONTIER -- oriented-signal test (no N_A, no 1/10, no epsilon)")
    print(f"PIPELINE = {PIPELINE}")
    print("=" * 70)
    for a in aggs:
        print(f"q = {a['q']}")
        print(f"  n3_obs                       = {a['n_sat']}")
        print(f"  max|theta_plus_raw|          = {a['max_abs_theta_plus_raw']:.3e}")
        print(f"  max|theta_sym_raw| (control) = {a['max_abs_theta_sym_raw']:.3e}")
        print(f"  theta_plus_cumulative(n_sat) = {a['theta_plus_sat']:+.6f}")
        print(f"  theta_max bound (n_sat)      = {a['theta_max_bound_sat']:.6f}  "
              f"(|Theta_chi| <= this; equality only under [H-orient])")
        print(f"  theta_max * q (n_sat)        = {a['theta_max_bound_sat_q']:.4f}  "
              f"(q-invariant diagnostic; should be q-stable across the prime list)")
        print(f"  sign_reversal_check          = {'passed' if a['sign_reversal_check'] else 'FAILED'}")
        print(f"  sym_frontier_zero_check      = "
              f"{'passed' if a['sym_frontier_zero_check'] else 'FAILED'}")
        print(f"  automorphism_residual_status = {a['automorphism_residual_status']}")
    tq_vals = [a["theta_max_bound_sat_q"] for a in aggs
               if not math.isnan(a["theta_max_bound_sat_q"])]
    if len(tq_vals) >= 2:
        mean = sum(tq_vals) / len(tq_vals)
        rel_spread = (max(tq_vals) - min(tq_vals)) / mean if mean else float("nan")
        print("-" * 70)
        print(f"theta_max * q across q: mean = {mean:.4f}, rel spread = {rel_spread:.2%}")
        print("(small spread => q-stable maximal-locking bound; the 1/q in theta_max is the")
        print(" 2pi/q angle convention. epsilon = N_A * theta_max still requires explicit N_A.)")
    print("=" * 70)
    print("Reading: an oriented signal is admissible only if max|theta_plus_raw| > 0 AND")
    print("the symmetrised control max|theta_sym_raw| = 0. Sign reversal alone is not enough.")
    print("theta_max is the maximal-locking bound on |Theta_chi|: a structural upper bound,")
    print("attained only under the chiral-area locking [H-orient]. It is NOT a value of epsilon.")
    print("=" * 70)
    if PIPELINE == "local-fallback":
        _fallback_banner()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "check"], default="full")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="permit running on the local-fallback pipeline (toy generators)")
    args = ap.parse_args()
    if PIPELINE == "local-fallback":
        _fallback_banner()
        if args.mode == "full" and not args.allow_fallback:
            print("REFUSING to run a full campaign on the local-fallback pipeline.\n"
                  "Make spectral_O12 importable (set PYTHONPATH), or pass --allow-fallback "
                  "to force a toy run.", flush=True)
            raise SystemExit(2)
    aggs = []
    for q in Q_LIST:
        agg = run_q(q)
        write_outputs(agg)
        if args.mode == "check":
            print(f"[q={q}] CHECK plus!=0: "
                  f"{'yes' if agg['max_abs_theta_plus_raw'] > TOL else 'no'}; "
                  f"sym==0: {'yes' if agg['sym_frontier_zero_check'] else 'no'}", flush=True)
        aggs.append(agg)
    final_summary(aggs)


if __name__ == "__main__":
    main()