"""
O11 pipeline: exact BFS on Heis_3(Z/qZ) + Fourier-character proxy fingerprint.

Phase A: exact BFS from identity, records shell_sizes[n] and ball_sizes[n].
Phase B: for each BFS depth n, sample random triples of endpoints from S_n,
         update per-block span trackers (Gram-Schmidt), compute block-wise
         projective capacity Sigma_n^(c), inter-block variance, log-log slope.

Group law: (a,b,c)*(a',b',c') = (a+a', b+b', c+c'+a*b')  mod q
Element encoding: id(a,b,c) = a + q*b + q^2*c   (integer in [0, q^3))
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Group arithmetic
# ---------------------------------------------------------------------------

def mul(a, b, c, a2, b2, c2, q):
    return (a + a2) % q, (b + b2) % q, (c + c2 + a * b2) % q

def inv(a, b, c, q):
    """Inverse of (a,b,c): (-a, -b, -c + a*b) mod q."""
    return (-a) % q, (-b) % q, (-c + a * b) % q

def encode(a, b, c, q):
    return a + q * b + q * q * c

def decode(idx, q):
    a = idx % q
    b = (idx // q) % q
    c = idx // (q * q)
    return a, b, c

def generators(q):
    """Return S_q = {X, X^-1, Y, Y^-1} as encoded integers."""
    X  = encode(1, 0, 0, q)
    Xi = encode(q - 1, 0, 0, q)
    Y  = encode(0, 1, 0, q)
    Yi = encode(0, q - 1, 0, q)
    return [X, Xi, Y, Yi]

def apply_gen(idx, gen_abc, q):
    """Return idx * gen (right multiplication)."""
    a, b, c = decode(idx, q)
    ga, gb, gc = gen_abc
    return encode(*mul(a, b, c, ga, gb, gc, q))

# ---------------------------------------------------------------------------
# Phase A: exact BFS
# ---------------------------------------------------------------------------

def exact_bfs(q, verbose=True):
    """
    BFS from identity on Cay(G_q, S_q).
    Returns shell_sizes, ball_sizes, and shells (list of arrays of encoded ids).
    Memory: visited as uint8 array of size q^3.
    """
    N = q ** 3
    gens_encoded = generators(q)
    gens_abc = [decode(g, q) for g in gens_encoded]

    visited = np.zeros(N, dtype=np.uint8)
    identity = encode(0, 0, 0, q)
    visited[identity] = 1

    shell_sizes = []
    ball_sizes  = []
    shells      = []

    frontier = np.array([identity], dtype=np.int64)
    ball_count = 1

    n = 0
    t0 = time.time()
    while len(frontier) > 0:
        shell_sizes.append(len(frontier))
        ball_sizes.append(ball_count)
        shells.append(frontier.copy())

        next_list = []
        for idx in frontier:
            a, b, c = decode(int(idx), q)
            for ga, gb, gc in gens_abc:
                nidx = encode(*mul(a, b, c, ga, gb, gc, q), q)
                if not visited[nidx]:
                    visited[nidx] = 1
                    next_list.append(nidx)

        if next_list:
            frontier = np.array(next_list, dtype=np.int64)
            ball_count += len(frontier)
        else:
            frontier = np.array([], dtype=np.int64)

        n += 1
        if verbose and n % 10 == 0:
            elapsed = time.time() - t0
            print(f"  BFS depth {n:3d}: |S_n|={shell_sizes[-1]:>10,}  "
                  f"|B_n|={ball_sizes[-1]:>10,}  ({elapsed:.1f}s)")

    if verbose:
        print(f"  BFS complete: diameter={len(shell_sizes)-1}, "
              f"total={ball_count}, time={time.time()-t0:.1f}s")
    return shell_sizes, ball_sizes, shells

# ---------------------------------------------------------------------------
# Phase B: Fourier-character proxy fingerprint
# ---------------------------------------------------------------------------

def sample_generic_blocks(q, n_blocks, rng):
    """
    Sample n_blocks generic 6-tuples (t1,t2,t3,s1,s2,s3) for the (a,b)-proxy.
    Genericity conditions:
      - t_i != 0 and t1+t2+t3 != 0  (mod q)  [for the a-component]
      - s_i != 0 and s1+s2+s3 != 0  (mod q)  [for the b-component]
    Returns array of shape (n_blocks, 6).
    """
    blocks = []
    while len(blocks) < n_blocks:
        t = rng.integers(1, q, size=(n_blocks * 4, 3))
        s = rng.integers(1, q, size=(n_blocks * 4, 3))
        valid_t = t[t.sum(axis=1) % q != 0]
        valid_s = s[s.sum(axis=1) % q != 0]
        n_valid = min(len(valid_t), len(valid_s))
        for i in range(n_valid):
            blocks.append(np.concatenate([valid_t[i], valid_s[i]]))
    return np.array(blocks[:n_blocks], dtype=np.int64)

def proxy_feature(endpoints_abc, block, q):
    """
    Compute the Fourier-character proxy value for a triple of endpoints.
    endpoints_abc: array (3, 3) with rows (a_i, b_i, c_i)
    block:         array (3,)  = (t1, t2, t3)
    Returns a complex scalar in C.

    The proxy is phi_{t1,t2,t3}(a1,a2,a3) = exp(2*pi*i*(t1*a1+t2*a2+t3*a3)/q).
    This is the abelianisation of the Weil representation in the a-coordinate.
    """
    dot = int(block[0]) * int(endpoints_abc[0, 0]) + \
          int(block[1]) * int(endpoints_abc[1, 0]) + \
          int(block[2]) * int(endpoints_abc[2, 0])
    return np.exp(2j * np.pi * dot / q)

def gram_schmidt_rank(basis_matrix):
    """
    Current rank of the column span tracked by basis_matrix (q x k complex).
    We maintain an orthonormal basis incrementally; rank = number of columns.
    Returns the rank (already known as the number of columns stored).
    """
    return basis_matrix.shape[1] if basis_matrix.ndim == 2 else 0

def add_vector_to_span(basis, v, tol=1e-10):
    """
    Try to add vector v to the orthonormal basis (q x k).
    Returns (new_basis, added) where added=True if v increased the rank.
    basis is None if empty.
    """
    if basis is None:
        norm = np.linalg.norm(v)
        if norm < tol:
            return basis, False
        return v[:, None] / norm, True
    # project out existing basis
    coeffs = basis.conj().T @ v
    residual = v - basis @ coeffs
    norm = np.linalg.norm(residual)
    if norm < tol:
        return basis, False
    new_col = residual / norm
    return np.hstack([basis, new_col[:, None]]), True

def run_fingerprint(q, shells, shell_sizes, blocks, n_path_samples, rng, verbose=True):
    """
    Incremental novelty capacity with 2D (a,b)-proxy per Weil block.

    The proxy feature for a triple (g1,g2,g3) = ((a_i,b_i,c_i)) and a block
    (t1,t2,t3,s1,s2,s3) is the pair of dot values:

        dot_a = (t1*a1 + t2*a2 + t3*a3) mod q
        dot_b = (s1*b1 + s2*b2 + s3*b3) mod q

    The pair (dot_a, dot_b) lives in (Z/qZ)^2 and indexes a DFT basis vector
    in C^{q^2}.  Since these are orthonormal, the rank of the span equals the
    number of distinct (dot_a, dot_b) pairs seen, with max rank q^2.

    The (a,b)-proxy is richer than the a-only proxy: for q=101 the proxy space
    has dimension q^2 = 10201 and saturation occurs around n ~ (q^2/4)^{1/3} ~ 14,
    which matches n* -- giving a usable pre-saturation window.

    Incremental capacity:
        Delta_r_n^(c) = |new (dot_a,dot_b) pairs from S_n| - already seen
        Sigma_n^(c)   = Delta_r_n^(c) / |S_n|    in [0,1]

    blocks: array (n_blocks, 6) -- columns (t1,t2,t3,s1,s2,s3)
    Returns:
        sigma:      array (n_depths, n_blocks)
        sigma_mean: array (n_depths,)
        sigma_var:  array (n_depths,)
    """
    n_blocks = len(blocks)
    n_depths = len(shells)

    # cumulative coverage array: seen_pairs[b, p] = True iff pair p has been seen
    # for block b.  p = dot_a + q*dot_b in [0, q^2).
    seen_pairs = np.zeros((n_blocks, q * q), dtype=bool)

    sigma      = np.zeros((n_depths, n_blocks))
    sigma_mean = np.zeros(n_depths)
    sigma_var  = np.zeros(n_depths)

    t0 = time.time()
    for n, shell in enumerate(shells):
        Sn_size = shell_sizes[n]
        if Sn_size == 0:
            sigma_mean[n] = 0.0
            continue

        eff_samples = min(n_path_samples, max(Sn_size ** 2, 1))
        idx_rows    = rng.choice(len(shell), size=(eff_samples, 3), replace=True)
        triple_ids  = shell[idx_rows]          # (eff_samples, 3)
        a_coords    = triple_ids % q           # (eff_samples, 3)
        b_coords    = (triple_ids // q) % q    # (eff_samples, 3)

        # dot_a[b, s] = (t1*a1 + t2*a2 + t3*a3) mod q  (blocks[:, :3])
        # dot_b[b, s] = (s1*b1 + s2*b2 + s3*b3) mod q  (blocks[:, 3:])
        dot_a = (blocks[:, :3] @ a_coords.T) % q   # (n_blocks, eff_samples)
        dot_b = (blocks[:, 3:] @ b_coords.T) % q   # (n_blocks, eff_samples)

        # encoded pair: dot_a + q*dot_b  in [0, q^2)
        pair_matrix = dot_a + q * dot_b             # (n_blocks, eff_samples)

        # vectorised incremental rank update:
        # for each block, find pairs not yet seen; update seen_pairs.
        # We maintain seen_pairs as a boolean array of shape (n_blocks, q^2).
        for b in range(n_blocks):
            new_mask = ~seen_pairs[b][pair_matrix[b]]
            new_pairs_idx = pair_matrix[b][new_mask]
            delta_r = min(int(new_mask.sum()), Sn_size)
            if len(new_pairs_idx):
                seen_pairs[b][new_pairs_idx] = True
            sigma[n, b] = delta_r / Sn_size

        sigma_mean[n] = sigma[n].mean()
        sigma_var[n]  = sigma[n].var()

        if verbose and n % 5 == 0:
            print(f"  depth {n:3d}: |S_n|={Sn_size:>10,}  "
                  f"sigma_mean={sigma_mean[n]:.5f}  "
                  f"sigma_var={sigma_var[n]:.3e}  ({time.time()-t0:.1f}s)")

    return sigma, sigma_mean, sigma_var

# ---------------------------------------------------------------------------
# Log-log slope extraction
# ---------------------------------------------------------------------------

def extract_slope(depths, sigma_mean, n_star, min_window=5):
    """
    Find the pre-saturation window [n0, n1] in [1, n_star] where sigma_mean
    is strictly monotone (increasing OR decreasing), then fit
    log(sigma_mean) ~ delta * log(n) + const.
    Returns (n0, n1, delta_hat, r2) or None if no valid window.

    For the incremental capacity the expected behaviour is monotone decreasing
    after the proxy space begins to saturate.
    """
    idx = np.where((depths >= 1) & (depths <= n_star) & (sigma_mean > 1e-8))[0]
    if len(idx) < min_window:
        return None

    d  = depths[idx].astype(float)
    sm = sigma_mean[idx]

    # find longest strictly monotone sub-sequence (increasing or decreasing)
    def longest_monotone(seq):
        best_inc = (0, 0)
        best_dec = (0, 0)
        i0_inc = i0_dec = 0
        for i in range(1, len(seq)):
            if seq[i] <= seq[i - 1]:
                if i - i0_inc > best_inc[1] - best_inc[0]:
                    best_inc = (i0_inc, i - 1)
                i0_inc = i
            if seq[i] >= seq[i - 1]:
                if i - i0_dec > best_dec[1] - best_dec[0]:
                    best_dec = (i0_dec, i - 1)
                i0_dec = i
        if len(seq) - i0_inc > best_inc[1] - best_inc[0]:
            best_inc = (i0_inc, len(seq) - 1)
        if len(seq) - i0_dec > best_dec[1] - best_dec[0]:
            best_dec = (i0_dec, len(seq) - 1)
        if best_inc[1] - best_inc[0] >= best_dec[1] - best_dec[0]:
            return best_inc
        return best_dec

    i0, i1 = longest_monotone(sm)
    if i1 - i0 + 1 < min_window:
        return None

    ld  = np.log(d[i0:i1 + 1])
    lsm = np.log(sm[i0:i1 + 1])
    coeffs    = np.polyfit(ld, lsm, 1)
    delta_hat = coeffs[0]
    residuals = lsm - np.polyval(coeffs, ld)
    ss_res    = (residuals ** 2).sum()
    ss_tot    = ((lsm - lsm.mean()) ** 2).sum()
    r2        = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return int(d[i0]), int(d[i1]), delta_hat, r2

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_ball_growth(q_list, all_ball_sizes, all_shell_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # log-log ball growth
    ax = axes[0]
    for q, bs in zip(q_list, all_ball_sizes):
        ns = np.arange(1, len(bs) + 1, dtype=float)
        ax.loglog(ns, bs, label=f"$q={q}$")
    ns_ref = np.logspace(0, 2, 50)
    ax.loglog(ns_ref, ns_ref ** 4 * 0.5, "k--", label=r"$\propto n^4$")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$|B_n|$")
    ax.set_title(r"Ball growth on $\mathrm{Cay}(G_q, S_q)$")
    ax.legend(fontsize=8)

    # shell sizes
    ax = axes[1]
    for q, ss in zip(q_list, all_shell_sizes):
        ns = np.arange(1, len(ss) + 1, dtype=float)
        ax.semilogy(ns, ss, label=f"$q={q}$")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$|S_n|$")
    ax.set_title("Shell sizes")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("fig_ball_growth.pdf")
    plt.close(fig)
    print("Saved fig_ball_growth.pdf")

def fig_block_capacities(q, depths, sigma, sigma_mean, n_show=50):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_blocks = sigma.shape[1]
    idx_show = RNG.choice(n_blocks, size=min(n_show, n_blocks), replace=False)
    for i in idx_show:
        ax.plot(depths, sigma[:, i], color="steelblue", alpha=0.2, linewidth=0.7)
    ax.plot(depths, sigma_mean, color="black", linewidth=2,
            label=r"$\bar{\Sigma}_n$ (mean)")
    sd = np.sqrt(np.array([sigma[n].var() for n in range(len(depths))]))
    ax.fill_between(depths, sigma_mean - sd, sigma_mean + sd,
                    color="steelblue", alpha=0.15, label=r"$\pm 1\,\sigma$")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\Sigma_n^{(c)}$")
    ax.set_title(f"Block-wise projective capacity, $q={q}$  "
                 f"({min(n_show, n_blocks)} sampled blocks shown)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_block_capacities.pdf")
    plt.close(fig)
    print("Saved fig_block_capacities.pdf")

def fig_variance(q_list, all_depths, all_sigma_mean, all_sigma_var):
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, depths, sm, sv in zip(q_list, all_depths, all_sigma_mean, all_sigma_var):
        ratio = np.where(sm > 1e-12, sv / (sm ** 2), np.nan)
        ax.plot(depths, ratio, label=f"$q={q}$")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label=r"$\varepsilon = 1$ (B2 threshold)")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\mathrm{Var}(\Sigma_n^{(c)}) \,/\, \bar{\Sigma}_n^2$")
    ax.set_title("Inter-block variance ratio (condition B2)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("fig_variance.pdf")
    plt.close(fig)
    print("Saved fig_variance.pdf")

def fig_loglog(q_list, all_depths, all_sigma_mean, all_n_star, all_slopes):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(q_list)))
    for q, depths, sm, n_star, slope_result, col in zip(
            q_list, all_depths, all_sigma_mean, all_n_star, all_slopes, colors):
        mask = (depths >= 1) & (sm > 0)
        ax.loglog(depths[mask], sm[mask], "o-", color=col, markersize=3,
                  label=f"$q={q}$")
        if slope_result is not None:
            n0, n1, dhat, r2 = slope_result
            d_fit = np.array([n0, n1], dtype=float)
            # reconstruct fit line
            log_d  = np.log(depths[mask])
            log_sm = np.log(sm[mask])
            c_fit  = np.polyfit(np.log(depths[(depths >= n0) & (depths <= n1) & (sm > 0)]),
                                np.log(sm[(depths >= n0) & (depths <= n1) & (sm > 0)]), 1)
            ax.loglog(d_fit, np.exp(np.polyval(c_fit, np.log(d_fit))),
                      "--", color=col, linewidth=1.5,
                      label=fr"$\hat\delta={dhat:.2f}$, $R^2={r2:.3f}$ ($q={q}$)")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\bar{\Sigma}_n$")
    ax.set_title("Log-log fit of mean block capacity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_loglog.pdf")
    plt.close(fig)
    print("Saved fig_loglog.pdf")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(q, n_blocks=500, n_path_samples=200, verbose=True):
    print(f"\n{'='*60}")
    print(f"q = {q}   |G_q| = {q**3:,}")
    print(f"{'='*60}")

    print("Phase A: exact BFS ...")
    t0 = time.time()
    shell_sizes, ball_sizes, shells = exact_bfs(q, verbose=verbose)
    print(f"  diameter = {len(shell_sizes) - 1},  time = {time.time()-t0:.1f}s")

    # n_star: first n with |B_n| >= q^2
    q2 = q ** 2
    n_star = next((n for n, bs in enumerate(ball_sizes) if bs >= q2),
                  len(ball_sizes) - 1)
    print(f"  n* (|B_n*| >= q^2 = {q2:,}) = {n_star}")

    print(f"Phase B: fingerprint proxy ({n_blocks} blocks, "
          f"{n_path_samples} path samples/depth) ...")
    blocks = sample_generic_blocks(q, n_blocks, RNG)
    depths = np.arange(len(shells))

    t0 = time.time()
    sigma, sigma_mean, sigma_var = run_fingerprint(
        q, shells, shell_sizes, blocks, n_path_samples, RNG, verbose=verbose)
    print(f"  fingerprint done in {time.time()-t0:.1f}s")

    slope_result = extract_slope(depths, sigma_mean, n_star)
    if slope_result:
        n0, n1, dhat, r2 = slope_result
        print(f"  Slope: n0={n0}, n1={n1}, delta_hat={dhat:.4f}, R2={r2:.4f}")
    else:
        print("  Slope: no valid window found")

    return shell_sizes, ball_sizes, shells, depths, sigma, sigma_mean, sigma_var, \
           n_star, slope_result

def main():
    q_list = [101, 151, 211, 307]

    all_shell_sizes = []
    all_ball_sizes  = []
    all_depths      = []
    all_sigma_mean  = []
    all_sigma_var   = []
    all_sigma       = {}
    all_n_star      = []
    all_slopes      = []

    results_table = []

    for q in q_list:
        (shell_sizes, ball_sizes, shells,
         depths, sigma, sigma_mean, sigma_var,
         n_star, slope_result) = run_one(q, n_blocks=500, n_path_samples=200)

        all_shell_sizes.append(shell_sizes)
        all_ball_sizes.append(ball_sizes)
        all_depths.append(depths)
        all_sigma_mean.append(sigma_mean)
        all_sigma_var.append(sigma_var)
        all_sigma[q] = sigma
        all_n_star.append(n_star)
        all_slopes.append(slope_result)

        if slope_result:
            n0, n1, dhat, r2 = slope_result
            results_table.append((q, n_star, n0, n1, f"{dhat:.4f}", f"{r2:.4f}"))
        else:
            results_table.append((q, n_star, "--", "--", "--", "--"))

    # summary table
    print("\n\nSUMMARY TABLE")
    print(f"{'q':>6}  {'n*':>5}  {'n0':>5}  {'n1':>5}  {'d_hat':>8}  {'R2':>7}")
    for row in results_table:
        print(f"{row[0]:>6}  {row[1]:>5}  {row[2]:>5}  {row[3]:>5}  "
              f"{row[4]:>8}  {row[5]:>7}")

    # figures
    print("\nGenerating figures ...")
    fig_ball_growth(q_list, all_ball_sizes, all_shell_sizes)
    # use last q for block-level figure
    last_q = q_list[-1]
    fig_block_capacities(last_q, all_depths[-1], all_sigma[last_q], all_sigma_mean[-1])
    fig_variance(q_list, all_depths, all_sigma_mean, all_sigma_var)
    fig_loglog(q_list, all_depths, all_sigma_mean, all_n_star, all_slopes)

    print("\nDone.")

if __name__ == "__main__":
    main()
