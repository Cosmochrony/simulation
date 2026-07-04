"""
trajectory-entropy/trajectory_branching.py
==========================================
Shellwise measurement of the projective trajectory-branching entropy rate
h(n) on Heisenberg cascades -- the D4 (projective distinguishability) test.

MOTIVATION (transfer of the two-regime expansion model)
--------------------------------------------------------
SpectralRelaxation 2.0 proposes a two-regime expansion model (polynomial
shell growth, then exponential branching of admissible trajectories) on LPS
expander graphs.  On Heis3(Z/qZ) the ball volume is polynomial of degree 4
(Bass-Guivarc'h), so if N_proj counts states (D1), the exponential regime
disappears.  If N_proj counts projectively distinguishable histories (D4),
the question becomes whether the entropy rate

    h(n) = (1/n) log N_dist(n),   0 <= h(n) <= log(d-1) = log 3

is positive before finite-substrate saturation.  This script measures h(n).

DEFINITION D4 (working definition, session 2026-07-04)
------------------------------------------------------
Two non-backtracking trajectories gamma, gamma' of length n from the
identity are distinguishable iff their projected profiles differ.  The
profile of gamma = (u_0, ..., u_n) is the sequence of pair residual symbols

    s_k = r_c(k) * r_{q-c}(k),   k = 1..n,

where r_c(k) is the residual norm of a probe vector at u_k against a frozen
end-of-shell basis B_{k-1} (order-free at snapshot granularity; basis index
= step index k, capped at the last computed shell).

STRUCTURAL NO-GO FOUND AT FIRST RUN (proved, kept as control variant V1)
------------------------------------------------------------------------
With the O12-exact choices (probe = single Weil vector on the UNIFORM
state, basis = O12 fingerprint span), the measurement provably collapses:

  * rho_c(a,b,gamma)|uniform> and every O12 fingerprint product have
    EXACTLY ONE nonzero Fourier coefficient (verified numerically): they
    are characters e_m up to phase (m = c.b, resp. m = c1 b1+c2 b2+c3 b3).
  * Hence the O12 span is a Fourier-aligned coordinate subspace, and ANY
    residual NORM against it is invariant under a-translation and central
    phase gamma: residuals can depend on b_k only.
  * In practice every reachable character is already spanned at low depth:
    symbols == 0 identically, h == 0 at all n, eps, q.

The central charge gamma is unconditionally erased from single-path norms
(it acts as a global phase in every irrep): the erased fibre of D4-by-norms
is exactly (gamma, part of the abelian data).  This sharpens candidate 2.

VARIANTS MEASURED
-----------------
Same Monte-Carlo path ensemble for all variants (comparability).  A fixed
generic reference state |phi0> (seeded complex Gaussian, stored in the npz)
replaces |uniform> where indicated.

  V1  uniform probe  + O12 fingerprint basis   [proved-collapse control]
  V2  generic probe  + O12 fingerprint basis   [b-shadow channel]
      Residuals depend on b_k only (theorem above), but are now graded
      (Fourier tail mass of phi0), not binary.  Analytic ceiling:
      h <= log(1+sqrt(2)) ~ 0.8814, the entropy rate of non-backtracking
      b-increment sequences (transfer matrix on {X, Y+, Y-} classes,
      lambda_max = 1+sqrt(2)).  Measures how much of the abelian b-shadow
      the graded residual channel retains.
  V3  generic probe  + Gabor basis (same phi0) [full abelian shadow]
      Basis at depth j = GS span of {rho_c(u)|phi0> : u in ball B_j},
      deduplicated by (a,b) (gamma is a phase).  This is a Gabor system;
      for prime q and generic phi0 the rank grows like the abelian ball
      (~2k^2), saturating at k ~ sqrt(q/2) (full spark).  Residuals depend
      on (a_k, b_k); the (a,b)-sequence determines the whole word, so the
      ceiling is the full log 3.  Self-consistent (one reference state for
      probe and basis) but departs from the O12-exact span: flagged.

ESTIMATORS (candidate 1 bracketed by 3 and 2, agreed protocol)
--------------------------------------------------------------
Profiles are discretized with bin width eps (sweep over EPS_LIST; a
binning-dependent verdict would be an artefact).

  h2(n)   [main, candidate 1, lower bound]: Renyi-2 (collision) entropy
      rate of discretized profile prefixes over T paths.  H2 <= H <=
      log N_dist.  Zero-collision points are censored at the floor
      log(T(T-1)/2) and flagged.
  h_low(n) [candidate 3, cruder lower bound]: distinct-count of the
      discretized accumulated divergence I(n) = sum_{k<=n} s_k.
  h_up(n)  [candidate 2 surrogate, upper bound]: per-step alphabet product
      prod min(A_k, 3), A_k = observed distinct symbols at step k.

VERDICT LOGIC
-------------
A plateau h* > 0 before basis saturation, stable across eps and q, means
the exponential regime survives the LPS -> Heisenberg transfer as a count
of histories (option B) -- in the V2 channel bounded by the b-shadow rate,
in the V3 channel by log 3.  Post-saturation decay ~ H(n_sat)/n is the
finite-substrate distinguishability bound, expected, outside the verdict
window.  Multi-q checks that the plateau is structural.

EPISTEMIC STATUS
----------------
V1 collapse: proved (character structure) + verified numerically.
V2 ceiling log(1+sqrt(2)): proved (transfer matrix).
Measured h values: numerical evidence on finite substrates, not proof.

USAGE
-----
python trajectory_branching.py                        # q in {29, 61, 101}
python trajectory_branching.py --primes 29 --T 20000  # smoke test
python trajectory_branching.py --plot-only            # figure from npz

REQUIRES: spectral_O12.py from ../o25 (imported).
Outputs (--out-dir, default traj_outputs/): q{q}_traj.npz,
trajectory_branching_h.pdf, summary tables on stdout.
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "o25"))

from spectral_O12 import (
    EPS_GS,
    build_generators,
    fingerprint_vectors_batch,
    gram_schmidt_batch,
    heisenberg_mul,
    make_psi_table,
)

DEFAULT_PRIMES = [29, 61, 101]
T_DEFAULT = 100_000
N_BLOCKS_DEFAULT = 3
EPS_LIST = [0.1, 0.05, 0.02, 0.01]
EPS_MAIN = 0.02
DEFAULT_SEED = 42
PHI0_SEED_OFFSET = 777_000   # phi0 seed = this + q (reported, stored)
OUTPUT_DIR = pathlib.Path(os.path.join(HERE, "traj_outputs"))
CHUNK = 400

VARIANTS = ["V1_uniform_o12", "V2_generic_o12", "V3_generic_gabor"]

# Path length per prime (verdict window ends at saturation anyway).
# q=101 capped at 14: the verdict window closes at Gabor saturation
# (k ~ sqrt(q/2) ~ 7); deeper shells only add the trivial H(n_sat)/n tail.
N_STEPS = {29: 8, 61: 12, 101: 14}
N_STEPS_FALLBACK = 12

LOG3 = np.log(3.0)
H_B_SHADOW = np.log(1.0 + np.sqrt(2.0))   # V2 analytic ceiling


def bfs_shells_capped(gens, q, max_depth):
    """BFS from identity, capped at depth max_depth (inclusive)."""
    identity = (0, 0, 0)
    visited = {identity}
    shells = [[identity]]
    for _ in range(max_depth):
        nxt = []
        for u in shells[-1]:
            for g in gens:
                v = heisenberg_mul(u, g, q)
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        if not nxt:
            break
        shells.append(nxt)
    return shells


def make_reference(q, seed):
    """Fixed generic reference state |phi0> (seeded complex Gaussian)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=q) + 1j * rng.normal(size=q)
    return v / np.linalg.norm(v)


def weil_ref_batch(a_arr, b_arr, gamma_arr, psi, q, phi0):
    """
    Exact Weil action rho_c(a,b,gamma)|phi0>, batched: (N, q) complex.
    [rho(a,b,g) phi](x) = psi_c(g + b(x-a)) phi(x-a).
    With phi0 = uniform this reproduces spectral_O12.weil_batch_lut.
    """
    x_out = np.arange(q, dtype=np.int64)
    x_in = (x_out[None, :] - a_arr[:, None]) % q
    arg = (gamma_arr[:, None] + b_arr[:, None] * x_in) % q
    return psi[arg] * phi0[x_in]


def build_shell_bases_o12(shells, c_block, q, gens, verbose=True):
    """
    O12 fingerprint Gram-Schmidt span, snapshot at the END of each shell
    (frozen, order-free at snapshot granularity).  Early-stops at rank q.
    Returns (bases, ranks).
    """
    gens_arr = np.array(gens, dtype=np.int64)
    c_block = np.asarray(c_block, dtype=np.int64)
    basis = None
    bases, ranks = [], []
    t0 = time.perf_counter()
    for n, shell in enumerate(shells):
        if basis is None or basis.shape[0] < q:
            shell_arr = np.array(shell, dtype=np.int64)
            for start in range(0, len(shell_arr), CHUNK):
                vecs = fingerprint_vectors_batch(
                    shell_arr[start:start + CHUNK], c_block, gens_arr, q)
                basis, _ = gram_schmidt_batch(basis, vecs, eps=EPS_GS)
                if basis.shape[0] >= q:
                    break
        bases.append(basis.copy())
        ranks.append(basis.shape[0])
        if verbose:
            dt = time.perf_counter() - t0
            print(f"    o12 basis shell {n}/{len(shells)-1}"
                  f"  rank={ranks[-1]}/{q}  elapsed={dt:.1f}s", flush=True)
    return bases, np.array(ranks, dtype=np.int64)


def build_shell_bases_gabor(shells, c, q, phi0, verbose=True):
    """
    V3 basis: GS span of probe vectors rho_c(u)|phi0> over the ball,
    deduplicated by (a, b) (gamma acts as a phase), snapshot per shell.
    Returns (bases, ranks).
    """
    psi = make_psi_table(c, q)
    seen_ab = set()
    basis = None
    bases, ranks = [], []
    t0 = time.perf_counter()
    for n, shell in enumerate(shells):
        if basis is None or basis.shape[0] < q:
            ab = []
            for (a, b, _g) in shell:
                if (a, b) not in seen_ab:
                    seen_ab.add((a, b))
                    ab.append((a, b))
            if ab:
                ab = np.array(ab, dtype=np.int64)
                vecs = weil_ref_batch(ab[:, 0], ab[:, 1],
                                      np.zeros(len(ab), dtype=np.int64),
                                      psi, q, phi0)
                basis, _ = gram_schmidt_batch(basis, vecs, eps=EPS_GS)
        if basis is None:
            basis = np.empty((0, q), dtype=np.complex128)
        bases.append(basis.copy())
        ranks.append(basis.shape[0])
        if verbose:
            dt = time.perf_counter() - t0
            print(f"    gabor basis shell {n}/{len(shells)-1}"
                  f"  rank={ranks[-1]}/{q}  elapsed={dt:.1f}s", flush=True)
    return bases, np.array(ranks, dtype=np.int64)


def sample_block(c1, q, rng, max_attempts=2000):
    """Generic block (c1,c2,c3): ci != 0, c1+c2+c3 != 0 mod q."""
    for _ in range(max_attempts):
        c2 = int(rng.integers(1, q))
        c3 = int(rng.integers(1, q))
        if (c1 + c2 + c3) % q != 0:
            return np.array([c1, c2, c3], dtype=np.int64)
    raise RuntimeError(f"cannot sample generic block c1={c1} q={q}")


def sample_paths(q, gens, T, n_steps, rng):
    """
    T non-backtracking paths of length n_steps from the identity.
    Generator order [X, X^-1, Y, Y^-1]: inverse of index i is i XOR 1.
    Returns positions (n_steps, T, 3) int64.
    """
    gens_arr = np.array(gens, dtype=np.int64)
    pos = np.zeros((T, 3), dtype=np.int64)
    out = np.zeros((n_steps, T, 3), dtype=np.int64)
    prev = None
    for k in range(n_steps):
        if prev is None:
            idx = rng.integers(0, 4, size=T)
        else:
            forbidden = prev ^ 1
            r = rng.integers(0, 3, size=T)
            idx = r + (r >= forbidden)
        for g in range(4):
            m = idx == g
            if not np.any(m):
                continue
            u = pos[m]
            a, b, gam = gens_arr[g]
            new = np.empty_like(u)
            new[:, 2] = (u[:, 2] + gam + u[:, 0] * b) % q
            new[:, 0] = (u[:, 0] + a) % q
            new[:, 1] = (u[:, 1] + b) % q
            pos[m] = new
        prev = idx
        out[k] = pos
    return out


def residual_norms(pos, basis, psi, q, phi0):
    """
    Residual norm of the unit probe rho_c(u)|phi0> against an orthonormal
    basis (r, q), batched over positions (T, 3).
    ||res||^2 = 1 - ||coeffs||^2 (unit probe, orthonormal basis).
    """
    v = weil_ref_batch(pos[:, 0], pos[:, 1], pos[:, 2], psi, q, phi0)
    if basis.shape[0] == 0:
        return np.ones(len(pos))
    coeffs = v @ basis.conj().T
    proj2 = np.einsum("ij,ij->i", coeffs, coeffs.conj()).real
    return np.sqrt(np.clip(1.0 - proj2, 0.0, 1.0))


def symbols_for(positions, bases_c, bases_qc, psi_c, psi_qc, q,
                phi0_c, phi0_qc, verbose_tag=""):
    """Pair residual symbols (T, n_steps) float32 on a fixed path ensemble."""
    n_steps, T, _ = positions.shape
    symbols = np.zeros((T, n_steps), dtype=np.float32)
    t0 = time.perf_counter()
    for k in range(1, n_steps + 1):
        j = min(k - 1, len(bases_c) - 1)
        r_c = residual_norms(positions[k - 1], bases_c[j], psi_c, q, phi0_c)
        r_qc = residual_norms(positions[k - 1], bases_qc[j], psi_qc, q,
                              phi0_qc)
        symbols[:, k - 1] = (r_c * r_qc).astype(np.float32)
    if verbose_tag:
        print(f"    symbols {verbose_tag}: {time.perf_counter()-t0:.1f}s",
              flush=True)
    return symbols


def collision_entropy(codes):
    """
    Renyi-2 entropy from integer row-codes via unbiased collision estimate.
    Returns (H2, censored): zero collisions -> censoring floor
    log(T(T-1)/2), censored=True.
    """
    T = len(codes)
    _, counts = np.unique(codes, return_counts=True)
    coll = np.sum(counts * (counts - 1.0))
    pairs = T * (T - 1.0)
    if coll <= 0:
        return np.log(pairs / 2.0), True
    return -np.log(coll / pairs), False


def prefix_codes(disc):
    """Incremental dense integer codes for row prefixes of (T, n) int16."""
    T = disc.shape[0]
    codes = np.zeros(T, dtype=np.int64)
    for n in range(1, disc.shape[1] + 1):
        pair = codes * 100003 + disc[:, n - 1].astype(np.int64)
        _, codes = np.unique(pair, return_inverse=True)
        yield n, codes


def estimators_for_eps(symbols, eps):
    """h2(n), censored flags, h_low(n), h_up(n) for one bin width eps."""
    n_steps = symbols.shape[1]
    disc = np.minimum((symbols / eps).astype(np.int16),
                      np.int16(int(1.0 / eps)))
    h2 = np.zeros(n_steps)
    cens = np.zeros(n_steps, dtype=bool)
    for n, codes in prefix_codes(disc):
        H2, c = collision_entropy(codes)
        h2[n - 1] = H2 / n
        cens[n - 1] = c
    acc = np.cumsum(symbols.astype(np.float64), axis=1)
    h_low = np.zeros(n_steps)
    for n in range(1, n_steps + 1):
        vals = (acc[:, n - 1] / eps).astype(np.int64)
        h_low[n - 1] = np.log(len(np.unique(vals))) / n
    alph = np.array([len(np.unique(disc[:, k])) for k in range(n_steps)],
                    dtype=np.float64)
    h_up = (np.cumsum(np.minimum(np.log(alph), LOG3))
            / np.arange(1, n_steps + 1))
    return h2, cens, h_low, h_up


def run_one_prime(q, T, n_blocks, seed, verbose=True, out_dir=None):
    """
    All variants, n_blocks conjugate pairs, one shared path ensemble.
    Milestone/resume: a partial checkpoint q{q}_traj_partial.npz is written
    after each block when out_dir is set; on restart, completed blocks are
    skipped (the RNG is re-advanced identically, so results are unchanged).
    """
    n_steps = N_STEPS.get(q, N_STEPS_FALLBACK)
    gens = build_generators(q)
    rng = np.random.default_rng(seed + q)
    phi0_gen = make_reference(q, PHI0_SEED_OFFSET + q)
    phi0_uni = np.ones(q, dtype=np.complex128) / np.sqrt(q)
    if verbose:
        print(f"  q={q}: BFS to depth {n_steps}...", flush=True)
    shells = bfs_shells_capped(gens, q, n_steps)
    if verbose:
        print(f"  q={q}: {len(shells)} shells,"
              f" sizes={[len(s) for s in shells]}", flush=True)
    positions = sample_paths(q, gens, T, n_steps, rng)

    nv = len(VARIANTS)
    all_symbols = np.zeros((nv, n_blocks, T, n_steps), dtype=np.float32)
    ranks_o12 = np.zeros((n_blocks, 2, len(shells)), dtype=np.int64)
    ranks_gab = np.zeros((n_blocks, 2, len(shells)), dtype=np.int64)
    cs = np.zeros(n_blocks, dtype=np.int64)

    partial = (pathlib.Path(out_dir) / f"q{q}_traj_partial.npz"
               if out_dir is not None else None)
    b_done = 0
    if partial is not None and partial.exists():
        zp = np.load(partial)
        if (int(zp["q"]) == q and int(zp["T"]) == T
                and int(zp["seed"]) == seed
                and int(zp["n_blocks"]) == n_blocks):
            b_done = int(zp["b_done"])
            all_symbols[:, :b_done] = zp["symbols"][:, :b_done]
            ranks_o12[:b_done] = zp["ranks_o12"][:b_done]
            ranks_gab[:b_done] = zp["ranks_gab"][:b_done]
            cs[:b_done] = zp["cs"][:b_done]
            if verbose:
                print(f"  q={q}: resuming, {b_done}/{n_blocks} blocks done",
                      flush=True)

    for b in range(n_blocks):
        c = int(rng.integers(1, (q - 1) // 2 + 1))
        cs[b] = c
        block_c = sample_block(c, q, rng)
        block_qc = sample_block(q - c, q, rng)
        if b < b_done:
            continue          # RNG re-advanced identically; skip compute
        psi_c, psi_qc = make_psi_table(c, q), make_psi_table(q - c, q)
        if verbose:
            print(f"  q={q} block {b+1}/{n_blocks}: pair (c={c},"
                  f" q-c={q-c})", flush=True)
        bo_c, r1 = build_shell_bases_o12(shells, block_c, q, gens, verbose)
        bo_qc, r2 = build_shell_bases_o12(shells, block_qc, q, gens, verbose)
        bg_c, r3 = build_shell_bases_gabor(shells, c, q, phi0_gen, verbose)
        bg_qc, r4 = build_shell_bases_gabor(shells, q - c, q, phi0_gen,
                                            verbose)
        ranks_o12[b, 0, :len(r1)], ranks_o12[b, 1, :len(r2)] = r1, r2
        ranks_gab[b, 0, :len(r3)], ranks_gab[b, 1, :len(r4)] = r3, r4
        all_symbols[0, b] = symbols_for(positions, bo_c, bo_qc, psi_c,
                                        psi_qc, q, phi0_uni, phi0_uni, "V1")
        all_symbols[1, b] = symbols_for(positions, bo_c, bo_qc, psi_c,
                                        psi_qc, q, phi0_gen, phi0_gen, "V2")
        all_symbols[2, b] = symbols_for(positions, bg_c, bg_qc, psi_c,
                                        psi_qc, q, phi0_gen, phi0_gen, "V3")
        if partial is not None:
            tmp = partial.with_name(partial.stem + "_tmp.npz")
            np.savez_compressed(
                tmp, q=q, T=T, seed=seed, n_blocks=n_blocks, b_done=b + 1,
                symbols=all_symbols, ranks_o12=ranks_o12,
                ranks_gab=ranks_gab, cs=cs)
            tmp.replace(partial)
            if verbose:
                print(f"  q={q}: checkpoint after block {b+1}", flush=True)
    if partial is not None and partial.exists():
        partial.unlink()
    return dict(q=q, T=T, seed=seed, n_steps=n_steps, cs=cs,
                phi0=phi0_gen, symbols=all_symbols,
                ranks_o12=ranks_o12, ranks_gab=ranks_gab,
                shell_sizes=np.array([len(s) for s in shells],
                                     dtype=np.int64))


def saturation_depth(ranks, q, n_steps):
    """First shell where all bases of a variant reach rank q (min/blocks)."""
    full = np.all(ranks >= q, axis=1)               # (n_blocks, n_shells)
    return min(int(np.argmax(f)) if np.any(f) else n_steps for f in full)


def analyse(res, eps_list=EPS_LIST, verbose=True):
    """Per-variant, per-eps estimators (block means).  Nested dict."""
    q, n_steps = int(res["q"]), int(res["n_steps"])
    out = {}
    for vi, vname in enumerate(VARIANTS):
        out[vname] = {}
        for eps in eps_list:
            h2s, cens, lows, ups = [], [], [], []
            for b in range(res["symbols"].shape[1]):
                h2, c, lo, up = estimators_for_eps(res["symbols"][vi, b],
                                                   eps)
                h2s.append(h2), cens.append(c)
                lows.append(lo), ups.append(up)
            out[vname][eps] = dict(
                h2=np.array(h2s), censored=np.array(cens),
                h_low=np.array(lows), h_up=np.array(ups))
    if verbose:
        sat_o12 = saturation_depth(res["ranks_o12"], q, n_steps)
        sat_gab = saturation_depth(res["ranks_gab"], q, n_steps)
        for vname in VARIANTS:
            sat = sat_gab if vname.endswith("gabor") else sat_o12
            ceil = (LOG3 if vname.endswith("gabor") else
                    (H_B_SHADOW if "generic" in vname else 0.0))
            print(f"\n  q={q} {vname}  (saturation shell ~{sat},"
                  f" ceiling={ceil:.3f})")
            print("  " + f"{'n':>3} " + "".join(
                f"| e={eps:<4} lo/h2/up " for eps in eps_list))
            for n in range(n_steps):
                row = f"  {n+1:>3} "
                for eps in eps_list:
                    d = out[vname][eps]
                    cf = "*" if np.any(d["censored"][:, n]) else " "
                    row += (f"| {np.mean(d['h_low'][:, n]):.2f}/"
                            f"{np.mean(d['h2'][:, n]):.2f}{cf}/"
                            f"{np.mean(d['h_up'][:, n]):.2f} ")
                print(row)
        print("  (* = censored h2: zero collisions, true value above floor)")
    return out


def save_npz(res, path):
    np.savez_compressed(path, **res)
    print(f"  Saved: {path}")


def make_figure(results, analyses, out_path):
    """Rows = variants, cols = primes; h2 main + bracket + saturation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = sorted(results.keys())
    nv = len(VARIANTS)
    fig, axes = plt.subplots(nv, len(qs),
                             figsize=(4.0 * len(qs), 2.9 * nv),
                             sharex="col", sharey=True, squeeze=False)
    for vi, vname in enumerate(VARIANTS):
        for qi, q in enumerate(qs):
            ax = axes[vi][qi]
            res, ana = results[q], analyses[q][vname]
            n_steps = int(res["n_steps"])
            ns = np.arange(1, n_steps + 1)
            d = ana[EPS_MAIN]
            h2m = np.mean(d["h2"], axis=0)
            ax.fill_between(ns, np.mean(d["h_low"], axis=0),
                            np.mean(d["h_up"], axis=0), alpha=0.2)
            ax.plot(ns, h2m, "o-", ms=3.5, label=f"h2, eps={EPS_MAIN}")
            cmask = np.any(d["censored"], axis=0)
            if np.any(cmask):
                ax.plot(ns[cmask], h2m[cmask], "v", ms=7, mfc="none")
            for eps in EPS_LIST:
                if eps != EPS_MAIN:
                    ax.plot(ns, np.mean(ana[eps]["h2"], axis=0), "-",
                            lw=0.6, alpha=0.5)
            ranks = (res["ranks_gab"] if vname.endswith("gabor")
                     else res["ranks_o12"])
            sat = saturation_depth(ranks, int(res["q"]), n_steps)
            ax.axvline(sat, color="gray", ls=":", lw=1)
            ax.axhline(LOG3, color="k", ls="--", lw=0.7)
            if "generic_o12" in vname:
                ax.axhline(H_B_SHADOW, color="C3", ls="--", lw=0.7)
            if vi == 0:
                ax.set_title(f"q = {q}")
            if vi == nv - 1:
                ax.set_xlabel("n (path length)")
            if qi == 0:
                ax.set_ylabel(vname.replace("_", " ") + "\n h(n)")
            ax.set_ylim(0, 1.25 * LOG3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Projective trajectory-branching entropy rate,"
                 r" $\mathrm{Heis}_3(\mathbb{Z}/q\mathbb{Z})$ (D4)",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Figure: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="D4 trajectory-branching entropy measurement")
    p.add_argument("--primes", type=int, nargs="+", default=DEFAULT_PRIMES)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--n-blocks", type=int, default=N_BLOCKS_DEFAULT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("trajectory_branching.py (D4 measurement)")
    print(f"primes={args.primes}  T={args.T}  n_blocks={args.n_blocks}"
          f"  seed={args.seed}  eps sweep={EPS_LIST}"
          f"  variants={VARIANTS}")

    results, analyses = {}, {}
    for q in sorted(set(args.primes)):
        path = args.out_dir / f"q{q}_traj.npz"
        print(f"\n--- q={q} ---")
        if path.exists() and not args.force:
            z = np.load(path)
            results[q] = {k: z[k] for k in z.files}
            print(f"  Loaded existing: {path}  (resume/milestone)")
        elif args.plot_only:
            print(f"  Missing {path}, skipping (plot-only)")
            continue
        else:
            t0 = time.perf_counter()
            results[q] = run_one_prime(q, args.T, args.n_blocks, args.seed,
                                       out_dir=args.out_dir)
            print(f"  q={q}: computation {time.perf_counter()-t0:.1f}s")
            save_npz(results[q], path)
        analyses[q] = analyse(results[q])

    if results:
        make_figure(results, analyses,
                    args.out_dir / "trajectory_branching_h.pdf")


if __name__ == "__main__":
    main()
