"""
O11 pipeline: exact BFS on Heis_3(Z/qZ) + bidimensional Fourier-character proxy.

Running `python o11_pipeline.py` produces all numerical results and figures for
paper SpectralO11.  Figures are saved as PDF in the current working directory.

Group law: (a,b,c)*(a',b',c') = (a+a', b+b', c+c'+a*b')  mod q
Encoding:  id(a,b,c) = a + q*b + q^2*c  in [0, q^3)

Proxy: for a triple of endpoints (g1,g2,g3) and channel (tau,sigma)=(t1,t2,t3,s1,s2,s3),
       u_a = (t1*a1+t2*a2+t3*a3) mod q,  u_b = (s1*b1+s2*b2+s3*b3) mod q.
       Feature = pair (u_a,u_b) encoded as u_a+q*u_b in [0,q^2).
       Rank = number of distinct pairs seen (exact for this DFT-basis proxy).
       Incremental capacity: Sigma_n = Delta_r_n / |S_n|, Delta_r_n = new pairs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

SEED      = 42
N_BLOCKS  = 500
N_SAMPLES = 3000
Q_LIST    = [101, 151, 211, 307]
COLORS    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

RNG = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Group arithmetic
# ---------------------------------------------------------------------------

def encode(a, b, c, q):
    return a + q * b + q * q * c

def decode(idx, q):
    a = idx % q
    b = (idx // q) % q
    c = idx // (q * q)
    return a, b, c

def mul(a, b, c, a2, b2, c2, q):
    return (a + a2) % q, (b + b2) % q, (c + c2 + a * b2) % q

def generators(q):
    return [encode(1, 0, 0, q), encode(q-1, 0, 0, q),
            encode(0, 1, 0, q), encode(0, q-1, 0, q)]

# ---------------------------------------------------------------------------
# Phase A: exact BFS
# ---------------------------------------------------------------------------

def exact_bfs(q, verbose=True):
    """BFS from identity. Returns shell_sizes, ball_sizes, shells."""
    N        = q ** 3
    gens_abc = [decode(g, q) for g in generators(q)]
    visited  = np.zeros(N, dtype=np.uint8)
    identity = encode(0, 0, 0, q)
    visited[identity] = 1

    shell_sizes, ball_sizes, shells = [], [], []
    frontier   = np.array([identity], dtype=np.int64)
    ball_count = 1
    t0         = time.time()

    while len(frontier):
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
            frontier    = np.array(next_list, dtype=np.int64)
            ball_count += len(frontier)
        else:
            frontier = np.array([], dtype=np.int64)

    if verbose:
        print(f"  BFS: diam={len(shell_sizes)-1}  "
              f"|G_q|={ball_count:,}  ({time.time()-t0:.1f}s)")
    return shell_sizes, ball_sizes, shells

# ---------------------------------------------------------------------------
# Phase B: bidimensional proxy fingerprint
# ---------------------------------------------------------------------------

def sample_generic_blocks(q, n_blocks, rng):
    """
    Sample n_blocks generic 6-tuples (t1,t2,t3,s1,s2,s3) with all components
    non-zero, t1+t2+t3 != 0 and s1+s2+s3 != 0  (mod q).
    Returns array of shape (n_blocks, 6).
    """
    blocks = []
    while len(blocks) < n_blocks:
        t  = rng.integers(1, q, size=(n_blocks * 4, 3))
        s  = rng.integers(1, q, size=(n_blocks * 4, 3))
        vt = t[t.sum(axis=1) % q != 0]
        vs = s[s.sum(axis=1) % q != 0]
        for i in range(min(len(vt), len(vs))):
            blocks.append(np.concatenate([vt[i], vs[i]]))
    return np.array(blocks[:n_blocks], dtype=np.int64)

def run_fingerprint(q, shells, shell_sizes, blocks, n_path_samples, rng,
                    verbose=True):
    """
    Incremental novelty capacity with bidimensional (a,b)-proxy.

    At depth n, sample endpoint triples from S_n, compute proxy pairs
    (u_a, u_b) per channel, count new pairs (Delta_r_n), compute
    Sigma_n^(tau,sigma) = Delta_r_n / |S_n|.

    seen_pairs[b] is a boolean array of size q^2 tracking cumulative coverage.
    """
    n_blocks   = len(blocks)
    n_depths   = len(shells)
    seen_pairs = np.zeros((n_blocks, q * q), dtype=bool)
    sigma      = np.zeros((n_depths, n_blocks))
    sigma_mean = np.zeros(n_depths)
    sigma_var  = np.zeros(n_depths)
    t0         = time.time()

    for n, shell in enumerate(shells):
        Sn = shell_sizes[n]
        if Sn == 0:
            continue

        eff      = min(n_path_samples, max(Sn ** 2, 1))
        idx_rows = rng.choice(len(shell), size=(eff, 3), replace=True)
        ids      = shell[idx_rows]
        a_coords = ids % q
        b_coords = (ids // q) % q

        dot_a = (blocks[:, :3] @ a_coords.T) % q   # (n_blocks, eff)
        dot_b = (blocks[:, 3:] @ b_coords.T) % q
        pairs = dot_a + q * dot_b                   # (n_blocks, eff)

        for b in range(n_blocks):
            new_mask = ~seen_pairs[b, pairs[b]]
            dr = min(int(new_mask.sum()), Sn)
            if new_mask.any():
                seen_pairs[b, pairs[b, new_mask]] = True
            sigma[n, b] = dr / Sn

        sigma_mean[n] = sigma[n].mean()
        sigma_var[n]  = sigma[n].var()

        if verbose and n % 5 == 0:
            print(f"  n={n:3d}: |S_n|={Sn:>10,}  "
                  f"mean={sigma_mean[n]:.5f}  var={sigma_var[n]:.3e}  "
                  f"({time.time()-t0:.1f}s)")

    return sigma, sigma_mean, sigma_var

# ---------------------------------------------------------------------------
# Slope extraction
# ---------------------------------------------------------------------------

def extract_slope(depths, sigma_mean, n_star, min_window=5):
    """
    Find the longest strictly monotone window in [1, n_star] with
    sigma_mean > 1e-8, fit log(sigma_mean) = slope*log(n) + const.
    slope < 0; delta_cap = -slope > 0.
    Returns (n0, n1, slope, r2) or None.
    """
    idx = np.where((depths >= 1) & (depths <= n_star) & (sigma_mean > 1e-8))[0]
    if len(idx) < min_window:
        return None

    d  = depths[idx].astype(float)
    sm = sigma_mean[idx]

    def longest_mono(seq):
        bi, bd = (0, 0), (0, 0)
        i0i = i0d = 0
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                if i - i0i > bi[1] - bi[0]:
                    bi = (i0i, i-1)
                i0i = i
            if seq[i] >= seq[i-1]:
                if i - i0d > bd[1] - bd[0]:
                    bd = (i0d, i-1)
                i0d = i
        if len(seq) - i0i > bi[1] - bi[0]:
            bi = (i0i, len(seq)-1)
        if len(seq) - i0d > bd[1] - bd[0]:
            bd = (i0d, len(seq)-1)
        return bi if (bi[1]-bi[0] >= bd[1]-bd[0]) else bd

    i0, i1 = longest_mono(sm)
    if i1 - i0 + 1 < min_window:
        return None

    ld  = np.log(d[i0:i1+1])
    lsm = np.log(sm[i0:i1+1])
    c   = np.polyfit(ld, lsm, 1)
    res = lsm - np.polyval(c, ld)
    ss  = ((lsm - lsm.mean())**2).sum()
    r2  = 1.0 - res.var() * len(lsm) / ss if ss > 1e-12 else 0.0
    return int(d[i0]), int(d[i1]), float(c[0]), float(r2)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_fig_ball_growth(q_list, all_ball_sizes, all_shell_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    ax = axes[0]
    for q, bs, col in zip(q_list, all_ball_sizes, COLORS):
        ns = np.arange(1, len(bs)+1, dtype=float)
        ax.loglog(ns, bs, color=col, linewidth=1.3, label=r"$q=%d$" % q)
    ns_ref = np.logspace(0.3, 2.5, 60)
    ax.loglog(ns_ref, 0.25*ns_ref**4, "k--", linewidth=1.0, label=r"$\propto n^4$")
    ax.set_xlabel("BFS depth $n$", fontsize=9)
    ax.set_ylabel(r"$|B_n|$", fontsize=9)
    ax.set_title(r"Ball growth on $\mathrm{Cay}(G_q,S_q)$", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    for q, ss, col in zip(q_list, all_shell_sizes, COLORS):
        ns = np.arange(1, len(ss)+1, dtype=float)
        ax.semilogy(ns, ss, color=col, linewidth=1.3, label=r"$q=%d$" % q)
    ax.set_xlabel("BFS depth $n$", fontsize=9)
    ax.set_ylabel(r"$|S_n|$", fontsize=9)
    ax.set_title("Shell sizes", fontsize=9)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("fig_ball_growth.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ball_growth.pdf")

def make_fig_block_capacities(q, depths, sigma, sigma_mean, sigma_var,
                               n_show=50):
    """Zoom to the useful signal window."""
    last = np.where(sigma_mean > 1e-5)[0]
    xmax = int(depths[last[-1]]) + 8 if len(last) else 40

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    rng_show = np.random.default_rng(7)
    idx_show = rng_show.choice(sigma.shape[1], size=min(n_show, sigma.shape[1]),
                               replace=False)
    for i in idx_show:
        ax.plot(depths, sigma[:, i], color="steelblue", alpha=0.18, linewidth=0.6)
    ax.plot(depths, sigma_mean, color="black", linewidth=1.8,
            label=r"$\bar{\Sigma}_n$ (mean)")
    sd = np.sqrt(np.maximum(sigma_var, 0))
    ax.fill_between(depths, np.maximum(sigma_mean - sd, 0), sigma_mean + sd,
                    color="steelblue", alpha=0.13, label=r"$\pm1\,\sigma$")
    ax.set_xlabel("BFS depth $n$", fontsize=9)
    ax.set_ylabel(r"$\Sigma_n^{(\tau,\sigma)}$", fontsize=9)
    ax.set_title(r"Block-wise incremental capacity, $q=%d$" % q, fontsize=9)
    ax.set_xlim(1, xmax)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_block_capacities.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_block_capacities.pdf")

def make_fig_variance(q_list, all_depths, all_sigma_mean, all_sigma_var):
    """
    Variance ratio, shown only while sigma_mean > 1e-5.
    y-axis capped at 1.5.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for q, depths, sm, sv, col in zip(q_list, all_depths, all_sigma_mean,
                                      all_sigma_var, COLORS):
        mask  = (depths >= 1) & (sm > 1e-5)
        d_m   = depths[mask]
        sm_m  = sm[mask]
        sv_m  = sv[mask]
        ratio = sv_m / sm_m**2
        ax.plot(d_m, ratio, color=col, linewidth=1.2, label=r"$q=%d$" % q)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.9,
               label=r"$\varepsilon=1$ (B2)")
    ax.set_xlabel("BFS depth $n$", fontsize=9)
    ax.set_ylabel(r"$\mathrm{Var}(\Sigma_n^{(\tau,\sigma)})\,/\,\bar{\Sigma}_n^2$",
                  fontsize=9)
    ax.set_title("Inter-channel variance ratio (condition B2)", fontsize=9)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_variance.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_variance.pdf")

def make_fig_loglog(q_list, all_depths, all_sigma_mean, all_slopes):
    """Log-log decay with fit lines labelled by delta_cap = -slope."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for q, depths, sm, slope, col in zip(q_list, all_depths, all_sigma_mean,
                                         all_slopes, COLORS):
        mask = (depths >= 2) & (sm > 1e-6)
        ax.loglog(depths[mask], sm[mask], "o-", color=col, markersize=2.5,
                  linewidth=0.9, label=r"$q=%d$" % q)
        if slope:
            n0, n1, dhat, r2 = slope
            d_w = depths[(depths >= n0) & (depths <= n1) & (sm > 1e-8)].astype(float)
            s_w = sm[(depths >= n0) & (depths <= n1) & (sm > 1e-8)]
            if len(d_w) >= 2:
                c   = np.polyfit(np.log(d_w), np.log(s_w), 1)
                lbl = (r"fit $q=%d$: $\hat\delta_{\rm cap}=%.2f$, $R^2=%.3f$"
                       % (q, -dhat, r2))
                ax.loglog(d_w[[0, -1]],
                          np.exp(np.polyval(c, np.log(d_w[[0, -1]]))),
                          "--", color=col, linewidth=1.5, label=lbl)
    ax.set_xlabel("BFS depth $n$", fontsize=9)
    ax.set_ylabel(r"$\bar{\Sigma}_n$", fontsize=9)
    ax.set_title("Log-log decay of mean channel capacity", fontsize=9)
    ax.legend(fontsize=6, ncol=1, loc="lower left")
    fig.tight_layout()
    fig.savefig("fig_loglog.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_loglog.pdf")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_shell_sizes, all_ball_sizes = [], []
    all_depths, all_sigma_mean, all_sigma_var = [], [], []
    all_sigma, all_n_star, all_slopes = {}, [], []
    results_table = []

    for q in Q_LIST:
        print(f"\n{'='*52}  q={q}  |G_q|={q**3:,}")

        t0 = time.time()
        shell_sizes, ball_sizes, shells = exact_bfs(q, verbose=False)
        print(f"  BFS: diam={len(shells)-1}  ({time.time()-t0:.1f}s)")

        q2     = q ** 2
        n_star = next((n for n, bs in enumerate(ball_sizes) if bs >= q2),
                      len(ball_sizes) - 1)
        print(f"  n* = {n_star}")

        blocks = sample_generic_blocks(q, N_BLOCKS, RNG)
        depths = np.arange(len(shells))

        t0 = time.time()
        sigma, sm, sv = run_fingerprint(q, shells, shell_sizes, blocks,
                                        N_SAMPLES, RNG, verbose=True)
        print(f"  fingerprint: {time.time()-t0:.1f}s")

        slope = extract_slope(depths, sm, n_star + 5, min_window=4)
        if slope:
            n0, n1, dhat, r2 = slope
            print(f"  slope: n0={n0} n1={n1} "
                  f"delta_cap={-dhat:.4f} R2={r2:.4f}")
        else:
            print("  slope: no valid window")

        all_shell_sizes.append(shell_sizes)
        all_ball_sizes.append(ball_sizes)
        all_depths.append(depths)
        all_sigma_mean.append(sm)
        all_sigma_var.append(sv)
        all_sigma[q] = sigma
        all_n_star.append(n_star)
        all_slopes.append(slope)

        if slope:
            n0, n1, dhat, r2 = slope
            results_table.append(
                (q, n_star, n0, n1, f"{-dhat:.4f}", f"{r2:.4f}"))
        else:
            results_table.append((q, n_star, "--", "--", "--", "--"))

    print("\n\nSUMMARY TABLE")
    print(f"{'q':>6}  {'n*':>4}  {'n0':>4}  {'n1':>4}  "
          f"{'d_cap_hat':>10}  {'R2':>7}")
    for row in results_table:
        print(f"{row[0]:>6}  {row[1]:>4}  {row[2]:>4}  {row[3]:>4}  "
              f"{row[4]:>10}  {row[5]:>7}")

    print("\nGenerating figures...")
    make_fig_ball_growth(Q_LIST, all_ball_sizes, all_shell_sizes)
    make_fig_block_capacities(Q_LIST[-1], all_depths[-1],
                              all_sigma[Q_LIST[-1]],
                              all_sigma_mean[-1], all_sigma_var[-1])
    make_fig_variance(Q_LIST, all_depths, all_sigma_mean, all_sigma_var)
    make_fig_loglog(Q_LIST, all_depths, all_sigma_mean, all_slopes)
    print("\nDone. Figures saved in current directory.")

if __name__ == "__main__":
    main()
