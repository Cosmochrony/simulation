"""
Generate figures for SpectralO9.

fig_loglog_heisenberg.pdf
  Log-log plot of |B_n| vs n (shell growth) showing the n^4 law on Heis_3(Z/qZ),
  demonstrating the window depth n_max = Omega(q^{1/2}).

fig_statelaw_heisenberg.pdf
  Comparison of window depth n_max(cq^2) between LPS and Heisenberg families,
  confirming the super-polynomial improvement.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Heisenberg group arithmetic mod q
# ---------------------------------------------------------------------------

def heis_mul(a, b, q):
    x1, y1, z1 = a
    x2, y2, z2 = b
    return ((x1 + x2) % q, (y1 + y2) % q, (z1 + z2 + x1 * y2) % q)


def bfs_ball(q):
    """Full BFS on Cay(Heis_3(Z/qZ), S_q); return array ball[n] = |B_n|."""
    gens = [(1, 0, 0), ((-1) % q, 0, 0), (0, 1, 0), (0, (-1) % q, 0)]
    visited = {(0, 0, 0): 0}
    frontier = [(0, 0, 0)]
    ball_sizes = [1]
    while frontier:
        nf = []
        for v in frontier:
            for g in gens:
                w = heis_mul(v, g, q)
                if w not in visited:
                    visited[w] = len(ball_sizes)
                    nf.append(w)
        frontier = nf
        if frontier:
            ball_sizes.append(len(visited))
    return np.array(ball_sizes)

# ---------------------------------------------------------------------------
# Figure 1: log-log growth |B_n| vs n
# ---------------------------------------------------------------------------

primes = [11, 13, 17, 19, 23, 29]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(primes)))

fig, ax = plt.subplots(figsize=(6.5, 4.8))

print("Ball growth data:")
for q, col in zip(primes, colors):
    ball = bfs_ball(q)
    n_arr = np.arange(len(ball))
    # Pre-wrap regime: first 60% of the diameter
    n_cut = max(3, int(0.6 * len(ball)))
    n_fit = n_arr[1:n_cut]
    b_fit = ball[1:n_cut]
    coeffs = np.polyfit(np.log(n_fit), np.log(b_fit), 1)
    D_hat = coeffs[0]
    print(f"  q={q}: diameter={len(ball)-1}, D_hat={D_hat:.3f}")
    ax.loglog(n_arr[1:], ball[1:], "o-", markersize=3, color=col,
              linewidth=1.2, label=f"$q={q}$  ($\\hat{{D}}={D_hat:.2f}$)")

# Reference line proportional to n^4
n_ref = np.array([1.0, 10.0])
ax.loglog(n_ref, 0.7 * n_ref ** 4, "k--", linewidth=1.5, label=r"$\propto n^4$")

ax.set_xlabel(r"BFS depth $n$", fontsize=12)
ax.set_ylabel(r"$|B_n|$", fontsize=12)
ax.set_title(
    r"Ball growth on $\mathrm{Cay}(\mathrm{Heis}_3(\mathbb{Z}/q\mathbb{Z}),\,S_q)$",
    fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_loglog_heisenberg.pdf", bbox_inches="tight")
print("Saved fig_loglog_heisenberg.pdf")

# ---------------------------------------------------------------------------
# Figure 2: window-depth comparison LPS vs Heisenberg
# ---------------------------------------------------------------------------

q_vals = np.array([11, 13, 17, 19, 23, 29, 31, 37, 41])

# LPS: n_max(c*q^2) = 2 * log_p(q) + O(1), representative p = 5
p_lps = 5
n_lps = 2.0 * np.log(q_vals) / np.log(p_lps) + 1.0

# Heisenberg: diameter = q (empirical); window depth = Omega(q^{1/2}) from theorem
# Empirically diameter ~ q; we use actual measured diameters
diam_arr = []
for q in q_vals:
    b = bfs_ball(int(q))
    diam_arr.append(len(b) - 1)
diam_arr = np.array(diam_arr)

# Theoretical lower bound from Theorem: n_max(q^2) >= (q^2 / c2)^{1/4}
c2 = 1.0   # upper constant in n^4 bound
n_heis_lb = (q_vals ** 2 / c2) ** 0.25

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
ax2.plot(q_vals, diam_arr, "s-", color="steelblue", markersize=6,
         label=r"Heisenberg: actual diameter $\sim q$")
ax2.plot(q_vals, n_heis_lb, "--", color="cornflowerblue", markersize=5,
         label=r"Heisenberg: theorem lower bound $\Omega(q^{1/2})$")
ax2.plot(q_vals, n_lps, "^--", color="firebrick", markersize=6,
         label=r"LPS ($p=5$): $O(\log q)$")
ax2.fill_between(q_vals, n_lps, diam_arr, alpha=0.10, color="steelblue")

ax2.set_xlabel(r"Prime $q$", fontsize=12)
ax2.set_ylabel(r"Available BFS depth", fontsize=12)
ax2.set_title(r"Window depth: LPS vs Heisenberg Cayley graphs", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("fig_statelaw_heisenberg.pdf", bbox_inches="tight")
print("Saved fig_statelaw_heisenberg.pdf")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\nSummary: window depths")
print(f"{'q':>5}  {'n_LPS':>8}  {'n_Heis(lb)':>12}  {'diam_Heis':>12}")
for q, nl, nh, nd in zip(q_vals, n_lps, n_heis_lb, diam_arr):
    print(f"{int(q):>5}  {nl:>8.1f}  {nh:>12.1f}  {int(nd):>12}")
