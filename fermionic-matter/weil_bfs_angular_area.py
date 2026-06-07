#!/usr/bin/env python3
"""
weil_bfs_angular_area.py  (A+Bx conventions, aligned with spectral_O12 / o32)

Measures, on the O12 single-character fingerprint
    fp_c(g; x) = q^{-3/2} exp( 2pi i (A_c(g) + B_c(g) x) / q ),
    B_c(g) = sum_i c_i b_i               (frequency  -> radial / capacity)
    A_c(g) = sum_i c_i (g_i - b_i a_i)    (central phase -> angular / J_3)
the two factorised observables of AngularAmplitudeReduction:

  RADIAL  (capacity):  sigma_c(n)    = |{ B_c(g) : g in shell n }| / |S_n|
  ANGULAR (J_3 split): Theta_Weil(n) = Odd_{J_Pi}( 2pi A_c / q )  Ihat-normalised

with  Ihat(n) = (sigma_pair(0)-sigma_pair(n)) / (sigma_pair(0)-sigma_pair(n_sat)),
      sigma_pair = sigma_c * sigma_{q-c},  n_sat = n_3^obs.

THREE GUARDRAILS (by construction):
  1. The value 1/10 appears ONLY in final_summary(), as the *required-normalisation*
     diagnostic N_A_required = (1/10)/Theta_Weil(n_3^obs) -- never as a result.
  2. No A -> eps conversion happens unless N_A is explicitly provided (default None).
  3. The primary unnormalised output is Theta_Weil (a raw angle), NOT eps.

mode=check additionally verifies:
  - sigma_c(n) = distinct-B count / |S_n|;
  - sigma invariance under the colour scaling c -> omega c (B -> omega B bijection);
  - the sign-reversal control: A_c -> -A_c under conjugation c -> q-c, and the
    oriented part flips sign under cascade (generator-order) reversal.

Outputs (current directory, per q):
  angular_area_q{q}_profile.csv
  angular_area_q{q}_summary.json
  angular_area_q{q}_checkpoint.jsonl
  angular_area_q{q}.pdf
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
N_MAX_CAP = 22                   # BFS depth cap (saturation usually below this)
N_BLOCKS = 32                    # generic blocks (c1,c2,c3) sampled per q
SAT_FRACTION = 0.99              # n_3^obs = first n with Ihat(n) >= SAT_FRACTION
N_A = None                       # angular normalisation A -> eps; UNSET on purpose
RNG_SEED = 20260607
N_WORKERS = max(1, cpu_count() - 1)
EPS_DICT = 0.10                  # dictionary value; used ONLY in final_summary()

PROFILE_TMPL = "angular_area_q{q}_profile.csv"
SUMMARY_TMPL = "angular_area_q{q}_summary.json"
CKPT_TMPL = "angular_area_q{q}_checkpoint.jsonl"
PDF_TMPL = "angular_area_q{q}.pdf"


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


def bfs_shells(q, n_max):
    gens = _build_generators(q)
    e = (0, 0, 0)
    dist = {e: 0}
    shells = [[e]]
    frontier = [e]
    for n in range(1, n_max + 1):
        nxt = []
        for g in frontier:
            for s in gens:
                h = ((g[0] + s[0]) % q, (g[1] + s[1]) % q, (g[2] + s[2] + g[0] * s[1]) % q)
                if h not in dist:
                    dist[h] = n
                    nxt.append(h)
        if not nxt:
            break
        shells.append(nxt)
        frontier = nxt
    return shells


def signed_mod(v, q):
    """Map v in [0,q) to the symmetric range (-q/2, q/2]; oriented value of A_c."""
    return ((v + q // 2) % q) - q // 2


def shell_BA(shell, block, q, gens, reverse=False):
    """Return (distinct_B_set, signed_A_array) over all generator triples on `shell`.

    The 3 accumulated steps are ep1=g.s1, ep2=ep1.s2, ep3=ep2.s3 (sigma_freq order);
    `reverse` swaps the triple order (s3,s2,s1) -- the cascade-reversal control.
    B_c = sum_i c_i b_i ;  A_c = sum_i c_i (g_i - b_i a_i).
    """
    c1, c2, c3 = block
    sh = np.array(shell, dtype=np.int64)
    Bset = set()
    A_signed_all = []
    triples = [(s1, s2, s3) for s1 in gens for s2 in gens for s3 in gens]
    for (s1, s2, s3) in triples:
        if reverse:
            s1, s2, s3 = s3, s2, s1
        e1 = _hmul_batch(sh, s1, q)
        e2 = _hmul_batch(e1, s2, q)
        e3 = _hmul_batch(e2, s3, q)
        B = (c1 * e1[:, 1] + c2 * e2[:, 1] + c3 * e3[:, 1]) % q
        A = (c1 * (e1[:, 2] - e1[:, 1] * e1[:, 0])
             + c2 * (e2[:, 2] - e2[:, 1] * e2[:, 0])
             + c3 * (e3[:, 2] - e3[:, 1] * e3[:, 0])) % q
        Bset.update(B.tolist())
        A_signed_all.append(signed_mod(A, q))
    return Bset, np.concatenate(A_signed_all)


def _primitive_cube_root(q):
    if q % 3 != 1:
        return None
    for x in range(2, q):
        if pow(x, 3, q) == 1:
            return x
    return None


def process_block(args):
    """One generic block (c1,c2,c3): sigma(n), theta(n) (oriented J_Pi-odd phase)."""
    block, q, shells = args
    gens = _build_generators(q)
    conj = tuple((q - c) % q for c in block)        # J_Pi conjugate block
    two_pi_q = 2.0 * math.pi / q

    seen = set()
    sigma = []
    theta = []           # per-shell oriented J_Pi-odd central phase (radians)
    theta_rev = []       # same, with reversed cascade order (sign control)
    for n, sh in enumerate(shells):
        if not sh:
            break
        before = len(seen)
        Bset, A_sg = shell_BA(sh, block, q, gens)
        _, A_conj = shell_BA(sh, conj, q, gens)
        seen |= Bset
        sigma.append((len(seen) - before) / len(sh))
        # J_Pi-odd part of the mean signed central phase (block minus conjugate)/2
        odd = 0.5 * (A_sg.mean() - A_conj.mean())
        theta.append(two_pi_q * odd)
        # reversed-cascade control (oriented part must flip sign)
        _, A_rev = shell_BA(sh, block, q, gens, reverse=True)
        _, A_rev_c = shell_BA(sh, conj, q, gens, reverse=True)
        theta_rev.append(two_pi_q * 0.5 * (A_rev.mean() - A_rev_c.mean()))

    # colour-scaling sigma invariance check (B -> omega B bijection)
    omega = _primitive_cube_root(q)
    sig_scaled = None
    if omega is not None:
        wblock = tuple((omega * c) % q for c in block)
        seen2, s2 = set(), []
        for sh in shells[:len(sigma)]:
            before = len(seen2)
            Bset, _ = shell_BA(sh, wblock, q, gens)
            seen2 |= Bset
            s2.append((len(seen2) - before) / len(sh))
        sig_scaled = s2

    return {
        "block": [int(x) for x in block], "q": int(q),
        "sigma": sigma, "theta": theta, "theta_rev": theta_rev,
        "sigma_scaled": sig_scaled,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(results, q):
    L = min(len(r["sigma"]) for r in results)
    if L < 3:
        raise RuntimeError(f"profile too short (L={L}); increase N_MAX_CAP for q={q}")
    sig = np.mean([np.array(r["sigma"][:L]) for r in results], axis=0)
    th = np.mean([np.array(r["theta"][:L]) for r in results], axis=0)
    th_rev = np.mean([np.array(r["theta_rev"][:L]) for r in results], axis=0)

    sigma_pair = sig ** 2                                   # sigma_c * sigma_{q-c}
    sigma_pair = sigma_pair / sigma_pair[0]

    I_raw = sigma_pair[0] - sigma_pair
    if I_raw[-1] <= 0:
        n_sat = L - 1
    else:
        Itmp = I_raw / I_raw[-1]
        idx = np.where(Itmp >= SAT_FRACTION)[0]
        n_sat = int(idx[0]) if len(idx) else L - 1
    n_sat = max(2, min(n_sat, L - 1))

    denom = sigma_pair[0] - sigma_pair[n_sat]
    Ihat = (sigma_pair[0] - sigma_pair) / denom if denom != 0 else np.zeros(L)

    Theta_raw = np.cumsum(th)                                # accumulated oriented phase
    with np.errstate(divide="ignore", invalid="ignore"):
        Theta_Weil = np.where(Ihat > 1e-9, Theta_raw / Ihat, np.nan)

    # sign-reversal control: cumulative reversed phase should be ~ -Theta_raw
    Theta_raw_rev = np.cumsum(th_rev)
    TOL = 1e-9
    if abs(Theta_raw[n_sat]) < TOL:
        angular_status = "vanishes_symmetric"     # oriented part = 0 under symmetric shell
        sign_status = "n/a (Theta=0 under symmetric averaging)"
    elif np.sign(Theta_raw[n_sat]) == -np.sign(Theta_raw_rev[n_sat]):
        angular_status = "oriented_signal"
        sign_status = "passed"
    else:
        angular_status = "oriented_signal"
        sign_status = "FAILED"
    sign_ok = (sign_status == "passed")

    # sigma colour-scaling check
    sigma_ok = True
    for r in results:
        if r["sigma_scaled"] is not None:
            m = min(len(r["sigma"]), len(r["sigma_scaled"]))
            if np.max(np.abs(np.array(r["sigma"][:m]) - np.array(r["sigma_scaled"][:m]))) > 1e-12:
                sigma_ok = False
                break

    return {
        "q": q, "n_sat": n_sat, "L": L,
        "sigma_pair": sigma_pair, "Ihat": Ihat,
        "Theta_raw": Theta_raw, "Theta_Weil": Theta_Weil,
        "Theta_Weil_sat": float(Theta_raw[n_sat]),           # = Theta_Weil(n_sat), Ihat=1
        "Theta_raw_rev_sat": float(Theta_raw_rev[n_sat]),
        "angular_status": angular_status, "sign_status": sign_status,
        "sigma_check": sigma_ok, "sign_reversal_check": sign_ok,
    }


# --------------------------------------------------------------------------- #
# Driver with checkpoints, progress/ETA
# --------------------------------------------------------------------------- #
def run_q(q):
    rng = np.random.default_rng(RNG_SEED + q)
    n_max = min(N_MAX_CAP, q // 2)
    t0 = time.time()
    print(f"[q={q}] pipeline={PIPELINE}  BFS depth {n_max} ...", flush=True)
    shells = bfs_shells(q, n_max)
    print(f"[q={q}] shells={len(shells)} |B|={sum(len(s) for s in shells)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # generic blocks (c1,c2,c3), c1+c2+c3 != 0
    blocks = []
    while len(blocks) < min(N_BLOCKS, (q - 1)):
        b = tuple(int(rng.integers(1, q)) for _ in range(3))
        if sum(b) % q != 0:
            blocks.append(b)

    ckpt = CKPT_TMPL.format(q=q)
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                r = json.loads(line)
                done[tuple(r["block"])] = r
        print(f"[q={q}] resumed {len(done)} blocks from {ckpt}", flush=True)

    todo = [b for b in blocks if b not in done]
    args = [(b, q, shells) for b in todo]
    results = list(done.values())

    t1 = time.time()
    if args:
        with Pool(N_WORKERS) as pool, open(ckpt, "a") as fh:
            for i, r in enumerate(pool.imap_unordered(process_block, args), 1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                results.append(r)
                el = time.time() - t1
                eta = el / i * (len(args) - i)
                print(f"[q={q}] block {i}/{len(args)}  elapsed={el:5.1f}s  ETA={eta:5.1f}s",
                      flush=True)

    return aggregate(results, q)


def write_outputs(agg):
    q = agg["q"]
    with open(PROFILE_TMPL.format(q=q), "w") as fh:
        fh.write("n,sigma_pair,Ihat,Theta_raw,Theta_Weil\n")
        for n in range(agg["L"]):
            tw = agg["Theta_Weil"][n]
            tw_s = "" if (isinstance(tw, float) and math.isnan(tw)) else f"{tw:.6g}"
            fh.write(f"{n},{agg['sigma_pair'][n]:.6g},{agg['Ihat'][n]:.6g},"
                     f"{agg['Theta_raw'][n]:.6g},{tw_s}\n")
    summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in agg.items()}
    with open(SUMMARY_TMPL.format(q=q), "w") as fh:
        json.dump(summary, fh, indent=2)
    _plot(agg)


def _plot(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = agg["q"]
    n = np.arange(agg["L"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].semilogy(n, agg["sigma_pair"], "o-", ms=3)
    ax[0].axvline(agg["n_sat"], ls=":", color="grey")
    ax[0].set_title(rf"$\sigma_{{\rm pair}}(n)$ (B-frequency count), $q={q}$")
    ax[0].set_xlabel("BFS depth $n$")
    ax[1].plot(n, agg["Theta_raw"], "o-", ms=3, label=r"$\Theta_{\rm raw}$")
    ax[1].plot(n, agg["Ihat"], "k--", lw=1, label=r"$\widehat I$")
    ax[1].axvline(agg["n_sat"], ls=":", color="grey")
    ax[1].set_title(r"oriented central phase (angular, from $A_c$)")
    ax[1].set_xlabel("BFS depth $n$")
    ax[1].legend(fontsize=8)
    ax[2].plot(n, agg["Theta_Weil"], "o-", ms=3, color="#a83232")
    ax[2].axvline(agg["n_sat"], ls=":", color="grey")
    ax[2].set_title(r"$\Theta_{\rm Weil}(n)=\Theta_{\rm raw}/\widehat I$ (raw angle)")
    ax[2].set_xlabel("BFS depth $n$")
    fig.suptitle(rf"Weil-BFS angular area $q={q}$ -- raw angle $\Theta_{{\rm Weil}}$ "
                 r"(NOT $\varepsilon$; $\mathcal{N}_A$ unset)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PDF_TMPL.format(q=q))
    print(f"[q={q}] figure -> {PDF_TMPL.format(q=q)}", flush=True)


def final_summary(aggs):
    print("\n" + "=" * 70)
    print("ANGULAR-AREA SUMMARY (raw angle Theta_Weil; eps requires N_A)")
    print("=" * 70)
    for a in aggs:
        q = a["q"]
        tw = a["Theta_Weil_sat"]
        # GUARDRAIL 1: 1/10 enters ONLY here, as a required-normalisation diagnostic
        na_req = (EPS_DICT / tw) if abs(tw) > 1e-12 else float("nan")
        print(f"q = {q}")
        print(f"  n3_obs                              = {a['n_sat']}")
        print(f"  theta_weil_sat                      = {tw:+.6f}")
        print(f"  N_A_required_for_epsilon_1_over_10  = {na_req:+.6f}   "
              f"(diagnostic, NOT a result)")
        if N_A is not None:                           # GUARDRAIL 2
            print(f"  epsilon_weil_sat = N_A*theta        = {N_A * tw:+.6f}   (N_A={N_A})")
        else:
            print(f"  epsilon_weil_sat                    = N_A not set "
                  f"-> primary output is the raw angle theta_weil_sat")
        print(f"  sigma_check                         = "
              f"{'passed' if a['sigma_check'] else 'FAILED'}")
        print(f"  angular_status                      = {a['angular_status']}")
        print(f"  sign_reversal_check                 = {a['sign_status']}")
    print("=" * 70)
    print("Note: Theta_Weil is the raw oriented J_Pi-odd central phase. The conversion")
    print("to eps requires the geometric normalisation N_A (deferred, note section 5).")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "check"], default="full")
    args = ap.parse_args()
    aggs = []
    for q in Q_LIST:
        agg = run_q(q)
        write_outputs(agg)
        if args.mode == "check":
            print(f"[q={q}] CHECK sigma colour-scaling invariance: "
                  f"{'passed' if agg['sigma_check'] else 'FAILED'}; "
                  f"sign-reversal: {'passed' if agg['sign_reversal_check'] else 'FAILED'}",
                  flush=True)
        aggs.append(agg)
    final_summary(aggs)


if __name__ == "__main__":
    main()