"""
Numerical test: B_n^2 + Sigma_n^2 vs B_n + Sigma_n on Heis3(Z/qZ).

Definitions faithful to O7:
  B_n  = cumulative fraction of (a,b) spatial pairs covered by BFS depth n
         (spatial occupation, cumulative, non-decreasing)
  Sigma_n = residual Weil-representation novelty:
            fraction of the q-dim Weil rep space NOT yet spanned by
            {omega(g) v0 : g in union_{k<=n} S_k}
            (temporal residual, instantaneous, non-increasing)

These are INDEPENDENT observables: B_n tracks physical coverage in (a,b),
Sigma_n tracks fingerprint-space residual capacity.

Test:
  Linear law (reduced model O7): B_n + Sigma_n = 1
  Quadratic law (BI/CC prediction): B_n^2 + Sigma_n^2 = 1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Heis3 BFS ────────────────────────────────────────────────────────────────

def heis3_bfs(q):
    """Return list of BFS shells (sets of (a,b,c)) on Heis3(Z/qZ)."""
    gens = [(1,0,0), (q-1,0,0), (0,1,0), (0,q-1,0)]
    identity = (0,0,0)
    visited = {identity}
    shells = [frozenset([identity])]
    frontier = {identity}
    while frontier:
        nxt = set()
        for (a,b,c) in frontier:
            for (da,db,dc) in gens:
                na = (a+da) % q
                nb = (b+db) % q
                nc = (c + dc + a*db) % q
                g = (na,nb,nc)
                if g not in visited:
                    nxt.add(g); visited.add(g)
        if not nxt:
            break
        shells.append(frozenset(nxt))
        frontier = nxt
    return shells


# ── Weil representation omega(a,b,c) acting on C^q ───────────────────────────
# [omega(a,b,c) f](x) = exp(2pi i (c + a*x) / q) * f(x+b mod q)
# Applied to v0 = e_0 (standard basis vector):
#   omega(a,b,c) e_0 = exp(2pi i c / q) * e_{b mod q}  <-- 1-dim per (b mod q)
# Applied to v0 = uniform = (1,...,1)/sqrt(q):
#   [omega(a,b,c) v0](x) = exp(2pi i (c + a*x) / q) / sqrt(q)
# = exp(2pi i c/q) / sqrt(q) * (exp(2pi i a*0/q),...,exp(2pi i a*(q-1)/q))
# = exp(2pi i c/q) * DFT_basis[a]
# So span depends ONLY on a-values -> too degenerate for a 2D test.
#
# Use FULL MATRIX span: omega(a,b,c) as q x q matrix.
# Flatten to q^2-dimensional vector and track span dimension.

def weil_matrix(a, b, c, q):
    """Return the q x q Weil rep matrix omega(a,b,c) as a flat q^2 vector."""
    # omega(a,b,c)_{x,y} = exp(2pi i (c + a*x) / q) * delta_{y, (x+b) mod q}
    phases = np.exp(2j * np.pi * (c + a * np.arange(q)) / q)  # shape (q,)
    # column y is nonzero only at row x = (y - b) mod q
    mat = np.zeros((q, q), dtype=complex)
    for x in range(q):
        y = (x + b) % q
        mat[x, y] = phases[x]
    return mat.ravel()  # q^2 complex vector


def span_dim(vectors):
    """Dimension of the real span of a list of complex vectors (viewed as R^{2q^2})."""
    if not vectors:
        return 0
    M = np.stack(vectors, axis=0)          # (n_vecs, q^2) complex
    M_real = np.concatenate([M.real, M.imag], axis=1)  # (n_vecs, 2q^2) real
    _, s, _ = np.linalg.svd(M_real, full_matrices=False)
    return int(np.sum(s > 1e-8 * s[0]))


def compute_observables(shells, q, max_depth=None):
    """
    Compute B_n and Sigma_n for each BFS depth n.

    B_n   = fraction of q^2 spatial (a,b) pairs covered cumulatively.
    Sigma_n = 1 - (dim of Weil matrix span at depth n) / q^2
              (residual Weil-rep capacity).

    Both are in [0,1].  They are INDEPENDENT by construction.
    """
    total_ab = q * q
    total_weil_dim = 2 * q * q      # real dim of C^{q x q} viewed as R^{2q^2}

    covered_ab = set()
    weil_vectors = []               # accumulate Weil matrix vectors
    cumulative_elements = set()

    B_list = []
    S_list = []

    for n, shell in enumerate(shells):
        if max_depth is not None and n > max_depth:
            break
        cumulative_elements.update(shell)

        # B_n: cumulative (a,b) spatial coverage
        for (a,b,c) in shell:
            covered_ab.add((a,b))
        B_n = len(covered_ab) / total_ab

        # Sigma_n: residual Weil dimension
        # Add ALL Weil matrix vectors for shell elements to the span
        for g in shell:
            weil_vectors.append(weil_matrix(*g, q))
        d = span_dim(weil_vectors)
        Sigma_n = 1.0 - d / total_weil_dim

        B_list.append(B_n)
        S_list.append(Sigma_n)

    return np.array(B_list), np.array(S_list)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(q, max_depth=None):
    shells = heis3_bfs(q)
    if max_depth:
        shells = shells[:max_depth+1]
    B, S = compute_observables(shells, q, max_depth=max_depth)
    return B, S, shells


def print_table(q, B, S):
    print(f"\n{'='*60}")
    print(f"q = {q}  |G| = {q**3}")
    print(f"{'n':>4} {'B_n':>8} {'Sig_n':>8} {'B+S':>8} {'B²+S²':>8}")
    print('-'*44)
    for n in range(len(B)):
        lin = B[n] + S[n]
        quad = B[n]**2 + S[n]**2
        print(f"{n:>4} {B[n]:>8.4f} {S[n]:>8.4f} {lin:>8.4f} {quad:>8.4f}")


def main():
    # Small primes where q^3 is manageable
    primes = [7, 11, 13]
    results = {}
    for q in primes:
        print(f"\nRunning q={q} (|G|={q**3})...")
        B, S = run(q, max_depth=None)[:2]
        print_table(q, B, S)
        results[q] = (B, S)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson']

    for idx, (q, (B, S)) in enumerate(results.items()):
        c = colors[idx]
        n_arr = np.arange(len(B))

        # Normalized depth tau = n / n_max for comparability
        tau = n_arr / max(n_arr) if max(n_arr) > 0 else n_arr

        lin  = B + S
        quad = B**2 + S**2

        axes[0].plot(tau, B, '-',  color=c, label=f'q={q}', linewidth=2)
        axes[0].plot(tau, S, '--', color=c, linewidth=1.5)

        axes[1].plot(tau, lin,  color=c, label=f'q={q}', linewidth=2)
        axes[2].plot(tau, quad, color=c, label=f'q={q}', linewidth=2)

    for ax in axes[1:]:
        ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='= 1')
        ax.set_ylim([0, 1.6])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    axes[0].set_title('B_n (solid) and Σ_n (dashed)')
    axes[0].set_xlabel('normalised depth τ'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(r'Linear test  $B_n + \Sigma_n$')
    axes[1].set_xlabel('normalised depth τ')

    axes[2].set_title(r'Quadratic test  $B_n^2 + \Sigma_n^2$')
    axes[2].set_xlabel('normalised depth τ')

    plt.suptitle(r'Linear vs quadratic residual law on Heis$_3(\mathbb{Z}/q\mathbb{Z})$',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('quadratic_law_test.png', dpi=150, bbox_inches='tight')
    print('\nFigure saved: quadratic_law_test.png')

    # ── Summary statistics ────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('SUMMARY: mean deviation from 1 (mid-range depths)')
    print(f"{'q':>4}  {'|B+S - 1| mean':>18}  {'|B²+S² - 1| mean':>20}")
    print('-'*48)
    for q, (B, S) in results.items():
        # exclude first and last point
        mid = slice(1, len(B)-1)
        d_lin  = np.abs(B[mid] + S[mid] - 1).mean()
        d_quad = np.abs(B[mid]**2 + S[mid]**2 - 1).mean()
        print(f"{q:>4}  {d_lin:>18.5f}  {d_quad:>20.5f}")

    print('\nConclusion: smaller mean deviation = better law.')
    print('If quadratic < linear, the BI/CC prediction is supported.')


if __name__ == '__main__':
    main()