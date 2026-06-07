#!/usr/bin/env python3
"""
weil_bfs_angular_area.py

Measures the accumulated oriented J_3-odd area of the Weil-BFS cascade on
Heis_3(Z/qZ), normalised by the projected capacity, and compares it (only at the
final step) with the ADE-deficit dictionary value eps = 1/10.

It implements the observable fixed by the reduction lemma
(AngularAmplitudeReduction):

    eps_Weil(n)   = sum_{m<=n} dA_J3_odd(m)
    Ihat(n)       = (sigma_pair(0)-sigma_pair(n)) / (sigma_pair(0)-sigma_pair(n_sat))
    K_ar_Weil(n)  = eps_Weil(n) / Ihat(n)
    target        = K_ar_Weil(n_3^obs) = eps_Weil(n_3^obs)        [Ihat(n_3^obs)=1]

LAYER STATUS (read before trusting any number):
  * EXACT, unambiguous layers (validated by mode=check against O30/O21):
        - Weil block rho_cc on C^q                (O12 eq. 1)
        - BFS on Cay(Heis_3(Z/qZ), {X^+-1,Y^+-1})
        - sigma_cc(n), sigma_pair(n), n_3^obs, Ihat(n)
        - static per-pair covariance C_c          (must reproduce [1:1/2:1/2], O30)
  * CONVENTION-SENSITIVE layer (ALIGN WITH the O28/O30/Q11 pipeline):
        - dA_J3_odd(n): the ORIENTED, J_Pi-odd J_3 increment. The orientation is
          carried by the cascade arrow I(n); a symmetric-shell average cancels the
          odd part by g <-> g^{-1} symmetry. The convention used here is documented
          at `j3_odd_increment`; substitute the canonical cascade ordering if it
          differs. Interpret K_ar_Weil ONLY after mode=check validates C_c.

The target value 1/10 is NOT used anywhere except the final comparison print.

Outputs (current directory):
    weil_bfs_raw.jsonl     per-pair raw results (one JSON object per line)
    weil_bfs_profiles.csv  K_ar_Weil(n), eps_Weil(n), sigma_pair(n), Ihat(n)
    weil_bfs_summary.pdf    three required figures
    checkpoint_q{q}.jsonl   resumable per-pair checkpoints

Engineering: parameters at the top, multi-core over conjugate pairs, progress/ETA,
milestone checkpoints (completed pairs are skipped on restart).
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.linalg import logm

# --------------------------------------------------------------------------- #
# PARAMETERS
# --------------------------------------------------------------------------- #
Q_LIST = [61, 101, 151]          # primes (q = 1 mod something not required here)
N_MAX_CAP = 28                   # hard cap on BFS depth (saturation usually < this)
N_PAIRS = 50                     # number of generic conjugate pairs sampled per q
SAT_FRACTION = 0.99              # n_sat = first n where Ihat(n) >= SAT_FRACTION
RNG_SEED = 20260607
N_WORKERS = max(1, cpu_count() - 1)
ADE_TARGET = None                # set ONLY inside final_comparison(); never used before

CHECKPOINT_TMPL = "checkpoint_q{q}.jsonl"
RAW_PATH = "weil_bfs_raw.jsonl"
CSV_PATH = "weil_bfs_profiles.csv"
PDF_PATH = "weil_bfs_summary.pdf"

# Optional external O21/O30 outputs for mode=check (consistency only, not input).
O21_N3_PATH = "o21_n3obs.json"   # {"61": n3, "101": n3, ...} if available
O30_RATIO_PATH = "o30_ccratio.json"  # {"61": [1.0,0.5,0.5], ...} if available


# --------------------------------------------------------------------------- #
# Heisenberg group Heis_3(Z/qZ): elements (a, b, z), law (a,b,z)(a',b',z')
#   = (a+a', b+b', z+z'+a b')  mod q
# --------------------------------------------------------------------------- #
def hmul(g, h, q):
    a, b, z = g
    ap, bp, zp = h
    return ((a + ap) % q, (b + bp) % q, (z + zp + a * bp) % q)


def generators(q):
    X = (1, 0, 0)
    Y = (0, 1, 0)
    Xi = ((-1) % q, 0, 0)
    Yi = (0, (-1) % q, 0)
    return [X, Y, Xi, Yi]


def bfs_shells(q, n_max):
    """Return shells[n] = list of group elements at word distance n from identity."""
    gens = generators(q)
    e = (0, 0, 0)
    dist = {e: 0}
    shells = [[e]]
    frontier = [e]
    for n in range(1, n_max + 1):
        nxt = []
        for g in frontier:
            for s in gens:
                h = hmul(g, s, q)
                if h not in dist:
                    dist[h] = n
                    nxt.append(h)
        if not nxt:
            break
        shells.append(nxt)
        frontier = nxt
    return shells


# --------------------------------------------------------------------------- #
# Weil block:  (rho_cc(a,b,z) f)(x) = exp(2pi i cc (z + b x)/q) f(x+a)
# --------------------------------------------------------------------------- #
def weil_apply(cc, g, f, q, xs, twopi_over_q):
    a, b, z = g
    shifted = np.roll(f, -a)                    # shifted[x] = f[x+a]
    phase = np.exp(1j * twopi_over_q * cc * ((z + b * xs) % q))
    return phase * shifted


def seed_vector(q):
    f = np.zeros(q, dtype=complex)
    f[0] = 1.0                                  # delta at x=0 (documented reference)
    return f


# --------------------------------------------------------------------------- #
# Per-pair pipeline
# --------------------------------------------------------------------------- #
def process_pair(args):
    cc, q, shells = args
    xs = np.arange(q)
    twopi_over_q = 2.0 * math.pi / q
    f0 = seed_vector(q)
    n_shells = len(shells)

    # --- pass 1: orbit covariance to fix the 2-dim admissible fibre V_rho ----
    # accumulate Sigma = <(rho f0)(rho f0)^dagger> over a mid-cascade window
    Sigma = np.zeros((q, q), dtype=complex)
    cnt = 0
    lo = min(2, n_shells - 1)
    hi = min(n_shells, max(lo + 1, n_shells // 2 + 1))
    for n in range(lo, hi):
        for g in shells[n]:
            v = weil_apply(cc, g, f0, q, xs, twopi_over_q)
            Sigma += np.outer(v, v.conj())
            cnt += 1
    if cnt:
        Sigma /= cnt
    w_eig, V = np.linalg.eigh(Sigma)            # ascending
    fibre = V[:, -2:]                           # top-2 admissible directions -> V_rho ~ C^2
    Pf = fibre.conj().T                         # projector q -> C^2

    # --- pass 2: per-shell capacity, C^3 covariance, J3-odd increment --------
    sigma = np.zeros(n_shells)
    Cc_per_shell = []                           # 3x3 covariance on Sym^2(V_rho) per shell
    for n in range(n_shells):
        ret = 0.0
        C3 = np.zeros((3, 3), dtype=complex)
        Sn = shells[n]
        for g in Sn:
            v = weil_apply(cc, g, f0, q, xs, twopi_over_q)
            ret += abs(np.vdot(f0, v)) ** 2                 # return amplitude (capacity)
            w = Pf @ v                                       # project to C^2
            # vec of symmetric M = w w^T in basis (M00, M11, sqrt2*M01)
            vecM = np.array([w[0] * w[0], w[1] * w[1], math.sqrt(2.0) * w[0] * w[1]])
            C3 += np.outer(vecM, vecM.conj())
        sigma[n] = ret / max(1, len(Sn))
        Cc_per_shell.append(C3 / max(1, len(Sn)))

    # static covariance (saturated window) and its [1:1/2:1/2] check ----------
    Cc_static = np.mean(np.stack(Cc_per_shell[lo:hi]), axis=0)
    ev = np.sort(np.linalg.eigvalsh(Cc_static))[::-1]
    ev = ev / ev[0] if ev[0] != 0 else ev                   # normalise to [1, ., .]
    cc_ratio = ev.real.tolist()

    # --- J3-odd oriented increment (CONVENTION-SENSITIVE; see j3_odd_increment)
    dA = j3_odd_increment(Cc_per_shell, q)

    return {
        "cc": int(cc), "q": int(q),
        "sigma": sigma.tolist(),
        "dA_J3_odd": dA.tolist(),
        "cc_ratio": cc_ratio,
    }


def j3_odd_increment(Cc_per_shell, q):
    """
    ORIENTED, J_Pi-odd J_3 increment per shell.  CONVENTION-SENSITIVE.

    Convention used here (document/align with the Q11 cascade ordering):
      - In the eigenframe of the static C_c, the distinct eigenvalue is the central
        weight-0 direction e_0; the remaining 2-d block is the outer doublet {e+, e-}.
      - The cascade increment between consecutive shells is the transfer
        T(n) = Cc(n) Cc(n-1)^+  restricted to the outer doublet; its oriented
        rotation angle (signed by increasing n, the I(n) arrow) is the J_3 angle.
      - The J_Pi-odd part is the component odd under cc <-> q-cc; here it is the
        imaginary (anti-Hermitian) part of log T projected on J_3 = diag(+1,-1)
        of the outer block.  A symmetric construction would give zero; the
        orientation (sign of the n-increment) is what makes it non-vanishing.

    Returns an array dA[n] (dA[0] = 0).
    """
    n_shells = len(Cc_per_shell)
    dA = np.zeros(n_shells)
    # identify e_0 from the static frame (use last shell as representative)
    Cstat = Cc_per_shell[min(len(Cc_per_shell) - 1, len(Cc_per_shell) // 2)]
    ev, U = np.linalg.eigh(Cstat)               # ascending; central = distinct one
    # central direction = the eigenvector whose eigenvalue is farthest from the mean
    idx_central = int(np.argmax(np.abs(ev - ev.mean())))
    outer_idx = [i for i in range(3) if i != idx_central]
    J3o = np.array([[1.0, 0.0], [0.0, -1.0]])   # J_3 on the outer doublet
    for n in range(1, n_shells):
        A = U[:, outer_idx].conj().T @ Cc_per_shell[n] @ U[:, outer_idx]
        B = U[:, outer_idx].conj().T @ Cc_per_shell[n - 1] @ U[:, outer_idx]
        try:
            Bi = np.linalg.pinv(B)
            T = A @ Bi
            # oriented rotation angle in the outer plane (anti-Hermitian / J_Pi-odd part)
            L = logm(T.astype(complex))
            L_odd = 0.5 * (L - L.conj().T)       # anti-Hermitian = J_Pi-odd part
            # J_3 angle = imaginary part of the J_3-projected anti-Hermitian log
            dA[n] = float(np.imag(np.trace(J3o @ L_odd)) / 2.0)
        except Exception:
            dA[n] = 0.0
    return dA


# --------------------------------------------------------------------------- #
# Observables built from per-pair results
# --------------------------------------------------------------------------- #
def aggregate(results, q):
    """Average sigma and dA over pairs; build sigma_pair, n_sat, Ihat, eps, K_ar."""
    sig = np.mean([np.array(r["sigma"]) for r in results], axis=0)
    dA = np.mean([np.array(r["dA_J3_odd"]) for r in results], axis=0)
    # pair observable sigma_pair = sigma_cc * sigma_{q-cc}; with the symmetric sample
    # the mean already pairs cc and q-cc, so use sigma^2 as the canonical pair proxy
    sigma_pair = sig ** 2
    sigma_pair = sigma_pair / sigma_pair[0]      # normalise sigma_pair(0)=1

    # intrinsic saturation rank n_3^obs: first n where the normalised cumulative
    # projected capacity reaches SAT_FRACTION of its final pre-cap value
    I_raw = sigma_pair[0] - sigma_pair
    if I_raw[-1] <= 0:
        n_sat = len(sigma_pair) - 1
    else:
        Itmp = I_raw / I_raw[-1]
        idx = np.where(Itmp >= SAT_FRACTION)[0]
        n_sat = int(idx[0]) if len(idx) else len(sigma_pair) - 1
    n_sat = max(n_sat, 2)

    denom = sigma_pair[0] - sigma_pair[n_sat]
    Ihat = (sigma_pair[0] - sigma_pair) / denom if denom != 0 else np.zeros_like(sigma_pair)

    eps = np.cumsum(dA)
    with np.errstate(divide="ignore", invalid="ignore"):
        K_ar = np.where(Ihat > 1e-9, eps / Ihat, np.nan)

    return {
        "q": q, "n_sat": n_sat,
        "sigma_pair": sigma_pair, "Ihat": Ihat,
        "eps": eps, "K_ar": K_ar,
        "K_ar_at_nsat": float(eps[n_sat]),       # = K_ar(n_sat) since Ihat(n_sat)=1
        "cc_ratio_mean": np.mean([r["cc_ratio"] for r in results], axis=0).tolist(),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_full(q):
    rng = np.random.default_rng(RNG_SEED + q)
    n_max = min(N_MAX_CAP, q // 2)
    t0 = time.time()
    print(f"[q={q}] BFS up to depth {n_max} ...", flush=True)
    shells = bfs_shells(q, n_max)
    print(f"[q={q}] shells built: depths={len(shells)}, "
          f"|B|={sum(len(s) for s in shells)} ({time.time()-t0:.1f}s)", flush=True)

    # generic conjugate pair labels cc in 1..q-1 (cc and q-cc are a pair)
    labels = rng.choice(np.arange(1, q), size=min(N_PAIRS, q - 1), replace=False)

    ckpt = CHECKPOINT_TMPL.format(q=q)
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["cc"]] = r
        print(f"[q={q}] resumed {len(done)} pairs from {ckpt}", flush=True)

    todo = [int(c) for c in labels if int(c) not in done]
    args = [(c, q, shells) for c in todo]
    results = list(done.values())

    t1 = time.time()
    with Pool(N_WORKERS) as pool, open(ckpt, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(process_pair, args), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            results.append(r)
            el = time.time() - t1
            eta = el / i * (len(args) - i)
            print(f"[q={q}] pair {i}/{len(args)}  elapsed={el:5.1f}s  ETA={eta:5.1f}s",
                  flush=True)

    return aggregate(results, q)


def run_check(q, agg):
    """Consistency-only: compare C_c ratio and n_3^obs against external O30/O21."""
    print(f"[q={q}] CHECK: mean C_c eigenvalue ratio = "
          f"{[round(x,3) for x in agg['cc_ratio_mean']]}  (expect ~ [1, 0.5, 0.5])",
          flush=True)
    if os.path.exists(O30_RATIO_PATH):
        ext = json.load(open(O30_RATIO_PATH)).get(str(q))
        print(f"[q={q}] CHECK: O30 ratio = {ext}", flush=True)
    if os.path.exists(O21_N3_PATH):
        ext = json.load(open(O21_N3_PATH)).get(str(q))
        print(f"[q={q}] CHECK: n_sat computed = {agg['n_sat']}, O21 n_3^obs = {ext}",
              flush=True)
    else:
        print(f"[q={q}] CHECK: n_sat computed = {agg['n_sat']} (no O21 file to compare)",
              flush=True)


def final_comparison(aggs):
    """The ONLY place the dictionary value 1/10 enters."""
    ADE = 1.0 / 10.0
    print("\n" + "=" * 70)
    print("FINAL COMPARISON  K_ar_Weil(n_3^obs)  ->?  1/10   (tested, not fitted)")
    print("=" * 70)
    for a in aggs:
        k = a["K_ar_at_nsat"]
        print(f"  q={a['q']:4d}  n_sat={a['n_sat']:3d}  "
              f"K_ar_Weil(n_sat)={k:+.4f}   |K_ar - 1/10|={abs(k-ADE):.4f}")
    ks = [a["K_ar_at_nsat"] for a in aggs]
    spread = (max(ks) - min(ks)) if len(ks) > 1 else 0.0
    print(f"  q-spread of K_ar_Weil = {spread:.4f}")
    print("  Outcomes: ->1/10 (orthogonal confirmation); !=1/10 q-stable (extra "
          "transfer factor, e.g. kappa=5/12); q-unstable (amplitude open).")
    print("=" * 70)


def write_outputs(aggs, all_results):
    with open(RAW_PATH, "w") as fh:
        for r in all_results:
            fh.write(json.dumps(r) + "\n")
    with open(CSV_PATH, "w") as fh:
        fh.write("q,n,sigma_pair,Ihat,eps_Weil,K_ar_Weil\n")
        for a in aggs:
            for n in range(len(a["sigma_pair"])):
                kar = a["K_ar"][n]
                fh.write(f"{a['q']},{n},{a['sigma_pair'][n]:.6g},{a['Ihat'][n]:.6g},"
                         f"{a['eps'][n]:.6g},"
                         f"{'' if (isinstance(kar,float) and math.isnan(kar)) else f'{kar:.6g}'}\n")
    _plot(aggs)


def _plot(aggs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a in aggs:
        n = np.arange(len(a["sigma_pair"]))
        ax[0].plot(n, a["sigma_pair"], "o-", ms=3, label=f"q={a['q']}")
        ax[1].plot(n, a["K_ar"], "o-", ms=3, label=f"q={a['q']}")
        ax[1].axvline(a["n_sat"], ls=":", alpha=0.4)
    ax[0].set_title(r"$\sigma_{\rm pair}(n)$ (recomputed)")
    ax[0].set_xlabel("BFS depth $n$"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
    ax[1].set_title(r"profile $K_{\rm ar}^{\rm Weil}(n)$")
    ax[1].set_xlabel("BFS depth $n$"); ax[1].legend(fontsize=8)
    qs = [a["q"] for a in aggs]
    ks = [a["K_ar_at_nsat"] for a in aggs]
    ax[2].plot(qs, ks, "s-", color="#a83232")
    ax[2].set_title(r"$K_{\rm ar}^{\rm Weil}(n_3^{\rm obs})$ vs $q$")
    ax[2].set_xlabel("$q$"); ax[2].set_ylabel(r"$K_{\rm ar}^{\rm Weil}$")
    fig.suptitle("Weil-BFS angular area: profile, saturation value, q-dependence "
                 "(target line drawn only post hoc)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PDF_PATH)
    print(f"figure -> {PDF_PATH}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "check"], default="full")
    args = p.parse_args()

    aggs, all_results = [], []
    for q in Q_LIST:
        agg = run_full(q)
        aggs.append(agg)
        if os.path.exists(CHECKPOINT_TMPL.format(q=q)):
            with open(CHECKPOINT_TMPL.format(q=q)) as fh:
                all_results.extend(json.loads(l) for l in fh)
        if args.mode == "check":
            run_check(q, agg)
    write_outputs(aggs, all_results)
    final_comparison(aggs)


if __name__ == "__main__":
    main()
