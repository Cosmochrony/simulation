"""
SpectralO7.py -- Projective Capacity and the Continuum Limit of Admissible Redundancy.

Companion numerical script for the paper:
``Projective Capacity and the Continuum Limit of Admissible Redundancy''

Computes:
  - k=2 path fingerprint effective novelty nov_n(gamma)
  - Discrete projective capacity Sigma_n(x)
  - Coarse-grained projected density |psi_n(x)|^2
  - Projective occupancy eta_n(x)
  - Redundancy functional R_n^(2)(x)
  - State law test: R_n^(2) vs eta_n vs Phi(eta) = 1/sqrt(1+eta^2)
  - Power-law fit of Sigma_n vs p(n)

Usage:
    python SpectralO7.py

Output:
    SpectralO7_fig1.pdf   -- 4-panel figure (state law test)
    SpectralO7_fig2.pdf   -- scaling collapse
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# LPS graph construction
# ---------------------------------------------------------------------------

def mod_sqrt_neg1(q):
    """Return i_q such that i_q^2 == -1 (mod q), or None."""
    for x in range(q):
        if (x * x + 1) % q == 0:
            return x
    return None


def lps_generators(p, q):
    """
    Compute the p+1 LPS generators in PSL(2, F_q).
    p must satisfy p = 1 (mod 4).
    Returns list of tuples ((a,b),(c,d)) representing 2x2 matrices mod q.
    """
    i_q = mod_sqrt_neg1(q)
    if i_q is None:
        raise ValueError(f"q={q} has no sqrt(-1); need q=1 mod 4")

    # Solve a0^2+a1^2+a2^2+a3^2 = p, a0>0 odd, a1,a2,a3 even
    solutions = []
    rng = range(-(p + 1), p + 2)
    for a0 in rng:
        for a1 in rng:
            for a2 in rng:
                for a3 in rng:
                    if (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p
                            and a0 > 0 and a0 % 2 == 1
                            and a1 % 2 == 0 and a2 % 2 == 0 and a3 % 2 == 0):
                        solutions.append((a0, a1, a2, a3))

    generators = []
    for (a0, a1, a2, a3) in solutions:
        m00 = (a0 + a1 * i_q) % q
        m01 = (a2 + a3 * i_q) % q
        m10 = (-a2 + a3 * i_q) % q
        m11 = (a0 - a1 * i_q) % q
        generators.append(((m00, m01), (m10, m11)))

    return generators


def mat_mul_mod(A, B, q):
    """Multiply two 2x2 matrices mod q."""
    ((a00, a01), (a10, a11)) = A
    ((b00, b01), (b10, b11)) = B
    return (
        ((a00 * b00 + a01 * b10) % q, (a00 * b01 + a01 * b11) % q),
        ((a10 * b00 + a11 * b10) % q, (a10 * b01 + a11 * b11) % q),
    )


def canonical_psl(M, q):
    """Canonical representative in PSL(2, F_q): choose min of M, -M."""
    a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
    pos = (a, b, c, d)
    neg = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
    return min(pos, neg)


def build_lps_graph(p, q):
    """
    Build the LPS Cayley graph X_{p,q} by BFS from the identity.
    Returns: (nodes, adj, gens) where
      nodes[idx] = canonical 4-tuple,
      adj[idx] = list of (neighbor_idx, gen_idx),
      gens = list of generator matrices.
    """
    gens = lps_generators(p, q)
    identity = canonical_psl(((1, 0), (0, 1)), q)

    nodes = [identity]
    node_to_idx = {identity: 0}
    adj = defaultdict(list)
    queue = deque([identity])

    while queue:
        v = queue.popleft()
        v_idx = node_to_idx[v]
        v_mat = ((v[0], v[1]), (v[2], v[3]))
        for s_idx, s in enumerate(gens):
            w_mat = mat_mul_mod(v_mat, s, q)
            w = canonical_psl(w_mat, q)
            if w not in node_to_idx:
                node_to_idx[w] = len(nodes)
                nodes.append(w)
                queue.append(w)
            w_idx = node_to_idx[w]
            adj[v_idx].append((w_idx, s_idx))

    return nodes, dict(adj), gens, node_to_idx


# ---------------------------------------------------------------------------
# Adjoint representation matrices for PSL(2, F_q)
# ---------------------------------------------------------------------------

def adjoint_matrix(g_tuple, q):
    """
    Compute the 3x3 adjoint representation matrix Ad(g) over R.
    g_tuple = (a, b, c, d) with g = [[a,b],[c,d]] in PSL(2, F_q).
    Basis for sl(2): e=[[0,1],[0,0]], f=[[0,0],[1,0]], h=[[1,0],[0,-1]].
    Ad(g) in basis (e,f,h):
      Column 0 (image of e): coefficients (a^2, -c^2, -2ac)
      Column 1 (image of f): coefficients (-b^2, d^2, 2bd)
      Column 2 (image of h): coefficients (-2ab, 2cd, ad+bc)
    """
    a, b, c, d = [float(x) for x in g_tuple]
    return np.array([
        [a * a,     -b * b,   -2 * a * b],
        [-c * c,     d * d,    2 * c * d],
        [-2 * a * c, 2 * b * d, a * d + b * c],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# BFS with k=2 path fingerprint and effective novelty
# ---------------------------------------------------------------------------

def phi_theoretical(eta):
    """Theoretical state law: Phi(eta) = 1 / sqrt(1 + eta^2)."""
    return 1.0 / np.sqrt(1.0 + eta ** 2)


def run_bfs_k2(p, q, cell_size=50, tol=1e-10, verbose=True):
    """
    Run BFS on X_{p,q} and compute capacity, occupancy, redundancy per cell.

    For each BFS step n, for each cell x:
      - |psi_n(x)|^2 = fraction of C_x visited
      - nov_n(gamma) = normalised distance of pi_2(gamma) to current span Pi_2(S_{n-1})
      - Sigma_n(x) = average nov_n over paths ending in frontier of C_x
      - eta_n(x) = |psi_n(x)|^2 / Sigma_n(x)
      - R_n^(2)(x) = fraction of frontier vertices in C_x with novel path

    Returns a dict with lists indexed by step n.
    """
    if verbose:
        print(f"\nBuilding LPS graph X_{{5,{q}}}...")
    nodes, adj, gens, node_to_idx = build_lps_graph(p, q)
    G_size = len(nodes)
    if verbose:
        print(f"  |G| = {G_size}, generators = {len(gens)}, cells = {G_size // cell_size}")

    # Precompute adjoint matrices
    Ad_mats = []
    for s in gens:
        g_t = (s[0][0], s[0][1], s[1][0], s[1][1])
        Ad_mats.append(adjoint_matrix(g_t, q))

    # BFS state
    identity_node = canonical_psl(((1, 0), (0, 1)), q)
    identity_idx = node_to_idx[identity_node]

    n_cells = (G_size + cell_size - 1) // cell_size  # ceiling division
    # Cell assignment: vertex v_idx -> cell idx = v_idx // cell_size (BFS order)
    # This is implicit: cell(v_idx) = v_idx // cell_size

    visited = np.zeros(G_size, dtype=bool)
    visited[identity_idx] = True
    S_n_set = {identity_idx}

    # Fingerprint span: global incremental QR basis (R^9 for k=2 adjoint)
    Q_basis = np.zeros((9, 0), dtype=np.float64)

    # Track per-cell counts
    cell_visited = np.zeros(n_cells, dtype=int)
    cell_visited[identity_idx // cell_size] += 1

    # Track last generator per node (for k=2 paths)
    node_last_gen = defaultdict(set)

    current_front = [identity_idx]
    step = 0

    # Output collections
    all_eta = []
    all_R = []
    p_n_vals = []  # cumulative |S_n|
    sigma_mean_vals = []  # mean Sigma_n across cells

    S_size = 1  # |S_0| = 1 (just the identity)

    while current_front:
        next_front = []
        next_paths = []  # (v_idx, s0_idx, s1_idx) for k=2 paths from step >= 2

        for u_idx in current_front:
            if u_idx not in adj:
                continue
            s0_set = node_last_gen[u_idx]  # generators used to arrive at u

            for (v_idx, s1_idx) in adj[u_idx]:
                if not visited[v_idx]:
                    visited[v_idx] = True
                    next_front.append(v_idx)
                    cell_visited[v_idx // cell_size] += 1

                # Record k=2 paths: need at least 2 steps
                if step >= 1 and s0_set:
                    for s0_idx in s0_set:
                        if s0_idx != s1_idx:  # non-backtracking approximation
                            next_paths.append((v_idx, s0_idx, s1_idx))

                node_last_gen[v_idx].add(s1_idx)

        if not next_front:
            break

        bsize = len(next_front)
        S_size += bsize
        p_n_vals.append(S_size)

        # --- Per-cell computation ---
        cell_novelties = defaultdict(list)
        cell_frontier = defaultdict(int)  # number of frontier vertices per cell

        for v_idx in next_front:
            cx = v_idx // cell_size
            cell_frontier[cx] += 1

        # Compute novelties for k=2 paths
        v_novel = {}  # v_idx -> best novelty (for R_n computation)
        path_novelties = {}  # (s0, s1) -> novelty (cache)

        for (v_idx, s0_idx, s1_idx) in next_paths:
            key = (s0_idx, s1_idx)
            if key not in path_novelties:
                fp = np.kron(Ad_mats[s0_idx], Ad_mats[s1_idx]).ravel()
                norm_fp = np.linalg.norm(fp)
                if norm_fp < 1e-14:
                    path_novelties[key] = 0.0
                else:
                    fp_unit = fp / norm_fp
                    if Q_basis.shape[1] > 0:
                        residual = fp_unit - Q_basis @ (Q_basis.T @ fp_unit)
                        nov = np.linalg.norm(residual)
                    else:
                        nov = 1.0
                    path_novelties[key] = float(nov)
            nov = path_novelties[key]

            cx = v_idx // cell_size
            cell_novelties[cx].append(nov)
            if v_idx not in v_novel or nov > v_novel[v_idx]:
                v_novel[v_idx] = nov

        # Update global Q_basis with genuinely new directions
        for (s0_idx, s1_idx), nov in path_novelties.items():
            if nov > tol:
                fp = np.kron(Ad_mats[s0_idx], Ad_mats[s1_idx]).ravel()
                norm_fp = np.linalg.norm(fp)
                if norm_fp < 1e-14:
                    continue
                fp_unit = fp / norm_fp
                if Q_basis.shape[1] == 0:
                    Q_basis = fp_unit.reshape(-1, 1)
                else:
                    residual = fp_unit - Q_basis @ (Q_basis.T @ fp_unit)
                    r_norm = np.linalg.norm(residual)
                    if r_norm > tol:
                        Q_basis = np.hstack([Q_basis, (residual / r_norm).reshape(-1, 1)])

        # Compute Sigma_n(x), |psi_n(x)|^2, eta_n(x), R_n(x) per cell
        step_eta = []
        step_R = []
        cell_sigma = []

        for cx in range(n_cells):
            visited_frac = cell_visited[cx] / cell_size if cell_size > 0 else 0.0
            psi2 = float(min(visited_frac, 1.0))

            novs = cell_novelties.get(cx, [])
            if novs:
                sigma = float(np.mean(novs))
            else:
                sigma = 0.0

            if sigma > 1e-12:
                eta = psi2 / sigma
            else:
                eta = float('inf') if psi2 > 0 else 0.0

            # R_n(x) = fraction of frontier vertices in C_x with at least one novel path
            n_frontier = cell_frontier.get(cx, 0)
            if n_frontier > 0:
                novel_count = sum(
                    1 for v_idx in next_front
                    if (v_idx // cell_size == cx and v_novel.get(v_idx, 0.0) > tol)
                )
                R = novel_count / n_frontier
            else:
                R = float('nan')

            if np.isfinite(eta) and np.isfinite(R) and n_frontier > 0:
                step_eta.append(eta)
                step_R.append(R)
                cell_sigma.append(sigma)

        if step_eta:
            all_eta.extend(step_eta)
            all_R.extend(step_R)
            sigma_mean_vals.append(float(np.mean(cell_sigma)))
        else:
            sigma_mean_vals.append(float('nan'))

        if verbose and step % 5 == 0:
            print(f"  Step {step:3d}: |dS|={bsize:6d}, |S|={S_size:7d}, "
                  f"dim_Pi2={Q_basis.shape[1]:3d}, "
                  f"n_pairs={len(step_eta):4d}")

        current_front = next_front
        step += 1

    return {
        'all_eta': np.array(all_eta),
        'all_R': np.array(all_R),
        'p_n': np.array(p_n_vals[:len(sigma_mean_vals)]),
        'sigma_mean': np.array(sigma_mean_vals),
        'G_size': G_size,
        'q': q,
        'n_steps': step,
    }


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_powerlaw(x, y, x_lo=None, x_hi=None):
    """Fit log y = log C + slope * log x in [x_lo, x_hi]. Returns (slope, C, r2)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if x_lo is not None:
        mask &= (x >= x_lo)
    if x_hi is not None:
        mask &= (x <= x_hi)
    if mask.sum() < 3:
        return None, None, None
    lx, ly = np.log(x[mask]), np.log(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    slope, log_C = coeffs
    C = np.exp(log_C)
    y_pred = slope * lx + log_C
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, C, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = 5
    qs = [29, 41, 61]
    cell_size = 50
    results = {}

    for q in qs:
        print(f"\n{'='*60}\nProcessing q={q}\n{'='*60}")
        res = run_bfs_k2(p, q, cell_size=cell_size, verbose=True)
        results[q] = res

    colors = {29: 'tab:blue', 41: 'tab:orange', 61: 'tab:green'}
    markers = {29: 'o', 41: 's', 61: '^'}

    # Theoretical Phi
    eta_th = np.linspace(0, 10, 500)
    phi_th = phi_theoretical(eta_th)

    # -----------------------------------------------------------------------
    # Figure 1: 4-panel main figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r'State law $R_n^{(2)}(x) \approx \Phi(\eta_n(x))$ on $X_{5,q}$ --- SpectralO7',
        fontsize=12, fontweight='bold'
    )

    # Panel (a): R vs eta scatter + theoretical curve
    ax = axes[0, 0]
    for q in qs:
        res = results[q]
        eta = res['all_eta']
        R = res['all_R']
        mask = np.isfinite(eta) & np.isfinite(R) & (eta < 15)
        ax.scatter(eta[mask], R[mask], s=6, alpha=0.4, color=colors[q],
                   marker=markers[q], label=rf'$q={q}$')
    ax.plot(eta_th, phi_th, 'k-', lw=2, label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$')
    ax.set_xlabel(r'Projective occupancy $\eta_n(x)$')
    ax.set_ylabel(r'$R_n^{(2)}(x)$')
    ax.set_title('State law test')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlim(-0.1, 8)
    ax.set_ylim(-0.05, 1.05)

    # Panel (b): residuals
    ax = axes[0, 1]
    for q in qs:
        res = results[q]
        eta = res['all_eta']
        R = res['all_R']
        mask = np.isfinite(eta) & np.isfinite(R) & (eta < 15)
        resid = R[mask] - phi_theoretical(eta[mask])
        ax.scatter(eta[mask], resid, s=6, alpha=0.4, color=colors[q],
                   marker=markers[q], label=rf'$q={q}$')
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xlabel(r'$\eta_n(x)$')
    ax.set_ylabel(r'$R_n^{(2)} - \Phi(\eta_n)$')
    ax.set_title('Residuals from theoretical curve')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlim(-0.1, 8)

    # Panel (c): Sigma_mean vs p(n) log-log with fit
    ax = axes[1, 0]
    param_rows = []
    for q in qs:
        res = results[q]
        pn = res['p_n']
        sm = res['sigma_mean']
        mask = np.isfinite(sm) & (sm > 0) & (pn > 1)
        if mask.sum() >= 3:
            ax.loglog(pn[mask], sm[mask], color=colors[q],
                      lw=1.5, label=rf'$q={q}$')
            slope, C, r2 = fit_powerlaw(pn, sm)
            if slope is not None:
                x_fit = np.logspace(np.log10(pn[mask].min()),
                                    np.log10(pn[mask].max()), 100)
                ax.loglog(x_fit, C * x_fit ** slope, color=colors[q],
                          ls='--', lw=1.0, alpha=0.8)
                alpha_val = -slope  # slope is negative
                beff = 1.0 / (0.5 + alpha_val) if alpha_val > 0 else float('nan')
                param_rows.append((q, res['G_size'],
                                   round(float(alpha_val), 3),
                                   round(float(beff), 3),
                                   round(float(r2), 3)))
    ax.set_xlabel(r'$p(n) = |S_n|$ (log)')
    ax.set_ylabel(r'$\bar\Sigma_n$ (log)')
    ax.set_title(r'Capacity $\bar\Sigma_n$ vs $p(n)$ (power-law fit: dashed)')
    ax.legend(fontsize=8)

    # Panel (d): parameter table
    ax = axes[1, 1]
    ax.axis('off')
    if param_rows:
        col_labels = [r'$q$', r'$|G|$', r'$\alpha$', r'$\beta_{\rm eff}$', r'$R^2$']
        table = ax.table(
            cellText=param_rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.6)
    ax.set_title('Parameter table', pad=20)

    plt.tight_layout()
    plt.savefig('SpectralO7_fig1.pdf',
                bbox_inches='tight', dpi=150)
    print("\nFigure 1 saved.")

    # -----------------------------------------------------------------------
    # Figure 2: scaling collapse R vs eta, all q
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for q in qs:
        res = results[q]
        eta = res['all_eta']
        R = res['all_R']
        mask = np.isfinite(eta) & np.isfinite(R) & (eta < 12)
        ax2.scatter(eta[mask], R[mask], s=6, alpha=0.35, color=colors[q],
                    marker=markers[q], label=rf'$q={q}$')

    ax2.plot(eta_th, phi_th, 'k-', lw=2.5, label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$')
    ax2.set_xlabel(r'Projective occupancy $\eta_n(x)$')
    ax2.set_ylabel(r'$R_n^{(2)}(x)$')
    ax2.set_title(
        r'Scaling collapse: all $q \in \{29,41,61\}$'
        '\n' + r'vs theoretical $\Phi(\eta)=1/\sqrt{1+\eta^2}$',
        fontsize=11
    )
    ax2.legend(markerscale=2, fontsize=10)
    ax2.set_xlim(-0.1, 8)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('SpectralO7_fig2.pdf',
                bbox_inches='tight', dpi=150)
    print("Figure 2 saved.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for row in param_rows:
        q, G_size, alpha, beff, r2 = row
        print(f"q={q}: |G|={G_size}, alpha={alpha:.3f}, "
              f"beta_eff={beff:.3f}, R2={r2:.3f}")

    print("\nReduced model prediction (uniform distribution):")
    print("  alpha_red = 1.0  =>  beta_eff_red = 2/3 = 0.667")
    print("  LPS corrections expected to increase alpha toward [7.4, 10.6]")


if __name__ == '__main__':
    main()
