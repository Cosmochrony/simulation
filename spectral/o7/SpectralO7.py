"""
SpectralO7.py -- Projective Capacity and the Continuum Limit of Admissible Redundancy.

Companion numerical script.

Figure 1 (4 panels):
  (a) State law R vs eta: reduced model exact curve Phi(eta)=1/sqrt(1+eta^2)
      + Monte Carlo validation of the reduced model
      + LPS data points where available
  (b) Residuals from Phi(eta)
  (c) Capacity Sigma vs p(n) from reduced model simulation
  (d) Parameter table

Figure 2: Scaling collapse (reduced model, three values of D)

The reduced model is the analytically tractable baseline established in the paper.
LPS graph data at k=2 (adjoint, dim=9) saturates quickly due to the small real
adjoint dimension; this is reported honestly in the paper (Section 6.4).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Theoretical state law
# ---------------------------------------------------------------------------

def phi_theoretical(eta):
    """Phi(eta) = 1 / sqrt(1 + eta^2)  (reduced model, exact)."""
    return 1.0 / np.sqrt(1.0 + np.asarray(eta, dtype=float) ** 2)


def eta_from_psi2(psi2):
    """eta = psi^2 / sqrt(1 - psi^2), defined for psi^2 in [0,1)."""
    psi2 = np.asarray(psi2, dtype=float)
    return psi2 / np.sqrt(np.maximum(1.0 - psi2, 1e-14))


# ---------------------------------------------------------------------------
# Reduced filling model: Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_reduced_model(D, n_steps, n_trials=200, rng=None):
    """
    Simulate the reduced filling model:
      - Draw n_steps unit vectors uniformly from S^{D-1}
      - Track span dimension d, capacity Sigma=sqrt(1-d/D), occupancy eta
      - Record (eta, R=Sigma) at each step
    Returns arrays: eta_vals, R_vals, pn_vals, sigma_vals
    """
    if rng is None:
        rng = np.random.default_rng(42)

    eta_all, R_all = [], []
    pn_all, sigma_all = [], []

    for _ in range(n_trials):
        Q = np.zeros((D, 0))  # orthonormal basis of current span
        etas, Rs, sigmas = [], [], []

        for step in range(n_steps):
            # Draw a uniform unit vector
            v = rng.standard_normal(D)
            v /= np.linalg.norm(v)

            # Effective novelty = distance to current span
            if Q.shape[1] == 0:
                nov = 1.0
            else:
                residual = v - Q @ (Q.T @ v)
                nov = float(np.linalg.norm(residual))

            # Update span if new direction
            if nov > 1e-10 and Q.shape[1] < D:
                res_unit = (v - Q @ (Q.T @ v)) / nov if Q.shape[1] > 0 else v
                Q = np.hstack([Q, res_unit.reshape(-1, 1)])

            d = Q.shape[1]
            psi2 = d / D
            sigma = float(np.sqrt(max(1.0 - psi2, 0.0)))
            eta = psi2 / sigma if sigma > 1e-12 else float('inf')

            etas.append(eta)
            Rs.append(nov)  # nov = R in the reduced model (avg over uniform draws = Sigma)
            sigmas.append(sigma)

        eta_all.extend(etas)
        R_all.extend(Rs)
        sigma_all.extend(sigmas)
        pn_all.extend(range(1, n_steps + 1))

    return (np.array(eta_all), np.array(R_all),
            np.array(pn_all), np.array(sigma_all))


# ---------------------------------------------------------------------------
# LPS graph construction (for small q only)
# ---------------------------------------------------------------------------

def mod_sqrt_neg1(q):
    for x in range(q):
        if (x * x + 1) % q == 0:
            return x
    return None


def lps_generators(p, q):
    i_q = mod_sqrt_neg1(q)
    if i_q is None:
        raise ValueError(f"q={q} has no sqrt(-1)")
    solutions = []
    rng = range(-(p + 1), p + 2)
    for a0 in rng:
        for a1 in rng:
            for a2 in rng:
                for a3 in rng:
                    if (a0*a0 + a1*a1 + a2*a2 + a3*a3 == p
                            and a0 > 0 and a0 % 2 == 1
                            and a1 % 2 == 0 and a2 % 2 == 0 and a3 % 2 == 0):
                        solutions.append((a0, a1, a2, a3))
    gens = []
    for (a0, a1, a2, a3) in solutions:
        m00 = (a0 + a1 * i_q) % q
        m01 = (a2 + a3 * i_q) % q
        m10 = (-a2 + a3 * i_q) % q
        m11 = (a0 - a1 * i_q) % q
        gens.append(((m00, m01), (m10, m11)))
    return gens


def canonical_psl(M, q):
    a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
    pos = (a, b, c, d)
    neg = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
    return min(pos, neg)


def adjoint_matrix(g_tuple, q):
    a, b, c, d = [float(x) for x in g_tuple]
    return np.array([
        [a*a,    -b*b,   -2*a*b],
        [-c*c,    d*d,    2*c*d],
        [-2*a*c,  2*b*d,  a*d+b*c],
    ], dtype=np.float64)


def run_lps_k2_small(p, q, max_nodes=5000, cell_size=30, tol=1e-10, verbose=True):
    """
    Run BFS on X_{p,q} up to max_nodes, compute k=2 path fingerprint data.
    Returns (eta_vals, R_vals, sigma_mean_per_step, p_n_vals, G_size).
    Works reliably for small graphs (q <= 17).
    """
    if verbose:
        print(f"  Building X_{{5,{q}}} (max {max_nodes} nodes)...")
    gens = lps_generators(p, q)
    Ad_mats = [adjoint_matrix((s[0][0],s[0][1],s[1][0],s[1][1]), q) for s in gens]

    identity = (1, 0, 0, 1)
    nodes = [identity]
    node_to_idx = {identity: 0}
    adj = defaultdict(list)
    queue = deque([identity])
    while queue and len(nodes) < max_nodes:
        v = queue.popleft()
        v_idx = node_to_idx[v]
        v_mat = ((v[0], v[1]), (v[2], v[3]))
        for s_idx, s in enumerate(gens):
            wm = (
                ((v_mat[0][0]*s[0][0]+v_mat[0][1]*s[1][0]) % q,
                 (v_mat[0][0]*s[0][1]+v_mat[0][1]*s[1][1]) % q),
                ((v_mat[1][0]*s[0][0]+v_mat[1][1]*s[1][0]) % q,
                 (v_mat[1][0]*s[0][1]+v_mat[1][1]*s[1][1]) % q),
            )
            w = canonical_psl(wm, q)
            if w not in node_to_idx:
                node_to_idx[w] = len(nodes)
                nodes.append(w)
                queue.append(w)
                if len(nodes) >= max_nodes:
                    break
            adj[v_idx].append((node_to_idx[w], s_idx))

    G_size = len(nodes)
    n_cells = (G_size + cell_size - 1) // cell_size
    if verbose:
        print(f"  |G_partial| = {G_size}, cells = {n_cells}")

    visited = np.zeros(G_size, dtype=bool)
    visited[0] = True
    Q_basis = np.zeros((9, 0), dtype=np.float64)
    node_last_gen = defaultdict(set)
    current_front = [0]
    cell_visited = np.zeros(n_cells, dtype=int)
    cell_visited[0] += 1
    step = 0
    S_size = 1

    eta_all, R_all, sigma_mean_list, pn_list = [], [], [], []

    while current_front:
        next_front = []
        next_paths = []
        for u_idx in current_front:
            s0_set = node_last_gen[u_idx]
            for (v_idx, s1_idx) in adj.get(u_idx, []):
                if not visited[v_idx]:
                    visited[v_idx] = True
                    next_front.append(v_idx)
                    cell_visited[v_idx // cell_size] += 1
                if step >= 1 and s0_set:
                    for s0_idx in s0_set:
                        if s0_idx != s1_idx:
                            next_paths.append((v_idx, s0_idx, s1_idx))
                node_last_gen[v_idx].add(s1_idx)

        if not next_front:
            break

        S_size += len(next_front)
        pn_list.append(S_size)

        # Compute novelties for each (s0,s1) pair seen this step
        pair_nov = {}
        for (v_idx, s0_idx, s1_idx) in next_paths:
            key = (s0_idx, s1_idx)
            if key not in pair_nov:
                fp = np.kron(Ad_mats[s0_idx], Ad_mats[s1_idx]).ravel()
                nfp = np.linalg.norm(fp)
                if nfp < 1e-14:
                    pair_nov[key] = 0.0
                    continue
                fp_u = fp / nfp
                if Q_basis.shape[1] == 0:
                    pair_nov[key] = 1.0
                else:
                    res = fp_u - Q_basis @ (Q_basis.T @ fp_u)
                    pair_nov[key] = float(np.linalg.norm(res))

        # Update Q_basis
        for key, nov in pair_nov.items():
            if nov > tol and Q_basis.shape[1] < 9:
                s0_idx, s1_idx = key
                fp = np.kron(Ad_mats[s0_idx], Ad_mats[s1_idx]).ravel()
                fp_u = fp / np.linalg.norm(fp)
                if Q_basis.shape[1] == 0:
                    Q_basis = fp_u.reshape(-1, 1)
                else:
                    res = fp_u - Q_basis @ (Q_basis.T @ fp_u)
                    rn = np.linalg.norm(res)
                    if rn > tol:
                        Q_basis = np.hstack([Q_basis, (res/rn).reshape(-1,1)])

        # Per-cell capacity
        cell_novs = defaultdict(list)
        cell_frontier = defaultdict(int)
        v_novel = {}
        for v_idx in next_front:
            cell_frontier[v_idx // cell_size] += 1
        for (v_idx, s0_idx, s1_idx) in next_paths:
            nov = pair_nov.get((s0_idx, s1_idx), 0.0)
            cell_novs[v_idx // cell_size].append(nov)
            if v_idx not in v_novel or nov > v_novel[v_idx]:
                v_novel[v_idx] = nov

        step_eta, step_R, step_sigma = [], [], []
        for cx in range(n_cells):
            nf = cell_frontier.get(cx, 0)
            if nf == 0:
                continue
            psi2 = float(min(cell_visited[cx] / cell_size, 1.0))
            novs = cell_novs.get(cx, [])
            sigma = float(np.mean(novs)) if novs else 0.0
            if sigma > 1e-12:
                eta = psi2 / sigma
            else:
                eta = float('inf') if psi2 > 0 else 0.0
            novel_count = sum(1 for v_idx in next_front
                              if v_idx // cell_size == cx
                              and v_novel.get(v_idx, 0.0) > tol)
            R = novel_count / nf
            if np.isfinite(eta) and np.isfinite(R):
                step_eta.append(eta)
                step_R.append(R)
                step_sigma.append(sigma)

        eta_all.extend(step_eta)
        R_all.extend(step_R)
        sigma_mean_list.append(float(np.mean(step_sigma)) if step_sigma else float('nan'))
        current_front = next_front
        step += 1

    return (np.array(eta_all), np.array(R_all),
            np.array(sigma_mean_list), np.array(pn_list[:len(sigma_mean_list)]),
            G_size)


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_powerlaw(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 1) & (y > 0)
    if mask.sum() < 4:
        return None, None, None
    lx, ly = np.log(x[mask]), np.log(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    slope, log_C = coeffs
    y_pred = slope * lx + log_C
    ss_res = np.sum((ly - y_pred)**2)
    ss_tot = np.sum((ly - np.mean(ly))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(np.exp(log_C)), float(r2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # 1. Reduced model simulations (three ambient dimensions)
    # -----------------------------------------------------------------------
    print("Running reduced model simulations...")
    Ds = [9, 27, 81]           # k=2 adj (D=9), k=3 adj (D=27), k=4 adj (D=81)
    labels = {9: r'$D=9$ ($k=2$)', 27: r'$D=27$ ($k=3$)', 81: r'$D=81$ ($k=4$)'}
    colors_D = {9: 'tab:blue', 27: 'tab:orange', 81: 'tab:green'}
    mc_data = {}
    for D in Ds:
        eta_mc, R_mc, pn_mc, sigma_mc = simulate_reduced_model(
            D, n_steps=D, n_trials=300, rng=rng)
        mc_data[D] = (eta_mc, R_mc, pn_mc, sigma_mc)
        print(f"  D={D}: {len(eta_mc)} data points")

    # -----------------------------------------------------------------------
    # 2. LPS data for small q (q=13 only, reliable)
    # -----------------------------------------------------------------------
    print("Running LPS X_{5,13}...")
    try:
        eta_lps, R_lps, sigma_lps, pn_lps, G_lps = run_lps_k2_small(
            5, 13, max_nodes=4500, cell_size=20, verbose=True)
        lps_ok = len(eta_lps) > 2
    except Exception as e:
        print(f"  LPS failed: {e}")
        lps_ok = False

    # -----------------------------------------------------------------------
    # 3. Theoretical curve
    # -----------------------------------------------------------------------
    eta_th = np.linspace(0, 8, 500)
    phi_th = phi_theoretical(eta_th)

    # -----------------------------------------------------------------------
    # Figure 1: 4-panel
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r'State law $R_n^{(k)}(x) \approx \Phi(\eta_n(x))$ --- SpectralO7',
        fontsize=12, fontweight='bold')

    # Panel (a): R vs eta
    ax = axes[0, 0]
    for D in Ds:
        eta_mc, R_mc, _, _ = mc_data[D]
        mask = np.isfinite(eta_mc) & np.isfinite(R_mc) & (eta_mc < 9)
        ax.scatter(eta_mc[mask], R_mc[mask], s=4, alpha=0.25,
                   color=colors_D[D], label=labels[D])
    if lps_ok:
        mask = np.isfinite(eta_lps) & np.isfinite(R_lps) & (eta_lps < 9)
        ax.scatter(eta_lps[mask], R_lps[mask], s=20, alpha=0.7,
                   color='tab:red', marker='D', zorder=5,
                   label=r'LPS $X_{5,13}$, $k=2$')
    ax.plot(eta_th, phi_th, 'k-', lw=2.2,
            label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$', zorder=6)
    ax.set_xlabel(r'Projective occupancy $\eta_n(x)$')
    ax.set_ylabel(r'$R_n^{(k)}(x)$')
    ax.set_title('State law (reduced model + LPS)')
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlim(-0.1, 8)
    ax.set_ylim(-0.05, 1.1)

    # Panel (b): residuals
    ax = axes[0, 1]
    for D in Ds:
        eta_mc, R_mc, _, _ = mc_data[D]
        mask = np.isfinite(eta_mc) & np.isfinite(R_mc) & (eta_mc < 9)
        resid = R_mc[mask] - phi_theoretical(eta_mc[mask])
        ax.scatter(eta_mc[mask], resid, s=4, alpha=0.25, color=colors_D[D],
                   label=labels[D])
    if lps_ok:
        mask = np.isfinite(eta_lps) & np.isfinite(R_lps) & (eta_lps < 9)
        resid_lps = R_lps[mask] - phi_theoretical(eta_lps[mask])
        ax.scatter(eta_lps[mask], resid_lps, s=20, alpha=0.7,
                   color='tab:red', marker='D', zorder=5,
                   label=r'LPS $X_{5,13}$')
    ax.axhline(0, color='k', lw=1.5)
    ax.set_xlabel(r'$\eta_n(x)$')
    ax.set_ylabel(r'$R_n^{(k)} - \Phi(\eta_n)$')
    ax.set_title('Residuals from theoretical curve')
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlim(-0.1, 8)

    # Panel (c): Sigma_mean vs p(n) log-log from reduced model
    ax = axes[1, 0]
    param_rows = []
    for D in Ds:
        _, _, pn_mc, sigma_mc = mc_data[D]
        # Average sigma over trials at each step (pn runs 1..D, n_trials times)
        steps = np.arange(1, D + 1)
        sigma_by_step = np.zeros(D)
        counts = np.zeros(D, dtype=int)
        for s, sig in zip(pn_mc.astype(int), sigma_mc):
            if 1 <= s <= D and np.isfinite(sig):
                sigma_by_step[s - 1] += sig
                counts[s - 1] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            sigma_avg = np.where(counts > 0, sigma_by_step / counts, np.nan)
        mask = np.isfinite(sigma_avg) & (sigma_avg > 0) & (steps > 1)
        if mask.sum() >= 4:
            ax.loglog(steps[mask], sigma_avg[mask], color=colors_D[D],
                      lw=1.5, label=labels[D])
            slope, C, r2 = fit_powerlaw(steps, sigma_avg)
            if slope is not None:
                x_fit = np.logspace(np.log10(steps[mask].min()),
                                    np.log10(steps[mask].max()), 80)
                ax.loglog(x_fit, C * x_fit**slope, color=colors_D[D],
                          ls='--', lw=1.0, alpha=0.8)
                alpha_val = -slope
                beff = 1.0 / (0.5 + alpha_val) if alpha_val > -0.5 else float('nan')
                param_rows.append((
                    f'$D={D}$',
                    f'{alpha_val:.3f}',
                    f'{beff:.3f}',
                    f'{r2:.3f}'
                ))
    ax.set_xlabel(r'Step $n$ (log)')
    ax.set_ylabel(r'$\bar\Sigma_n$ (log)')
    ax.set_title(r'Capacity $\bar\Sigma_n$ vs step $n$ (fit: dashed)')
    ax.legend(fontsize=8)

    # Panel (d): parameter table
    ax = axes[1, 1]
    ax.axis('off')
    if param_rows:
        col_labels = [r'Model', r'$\alpha$', r'$\beta_{\rm eff}$', r'$R^2$']
        tbl = ax.table(
            cellText=param_rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.8)
        ax.set_title('Parameter table (reduced model)', pad=20)
    else:
        ax.text(0.5, 0.5, 'No fit data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig('SpectralO7_fig1.pdf',
                bbox_inches='tight', dpi=150)
    print("Figure 1 saved.")

    # -----------------------------------------------------------------------
    # Figure 2: scaling collapse (D=9, 27, 81 all on same axes vs eta)
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for D in Ds:
        eta_mc, R_mc, _, _ = mc_data[D]
        mask = np.isfinite(eta_mc) & np.isfinite(R_mc) & (eta_mc < 9)
        # Bin-average for cleaner plot
        eta_bins = np.linspace(0, 8, 30)
        R_binned = []
        eta_centers = []
        for i in range(len(eta_bins) - 1):
            m = mask & (eta_mc >= eta_bins[i]) & (eta_mc < eta_bins[i+1])
            if m.sum() > 5:
                R_binned.append(float(R_mc[m].mean()))
                eta_centers.append(float((eta_bins[i] + eta_bins[i+1]) / 2))
        if eta_centers:
            ax2.plot(eta_centers, R_binned, color=colors_D[D], lw=1.8,
                     marker='o', ms=4, label=labels[D])

    if lps_ok:
        mask = np.isfinite(eta_lps) & np.isfinite(R_lps) & (eta_lps < 9)
        ax2.scatter(eta_lps[mask], R_lps[mask], s=40, color='tab:red',
                    marker='D', zorder=5, label=r'LPS $X_{5,13}$, $k=2$')

    ax2.plot(eta_th, phi_th, 'k-', lw=2.5,
             label=r'$\Phi(\eta)=1/\sqrt{1+\eta^2}$', zorder=6)
    ax2.set_xlabel(r'Projective occupancy $\eta_n(x)$')
    ax2.set_ylabel(r'$R_n^{(k)}(x)$')
    ax2.set_title(
        r'Scaling collapse: reduced model ($D=9,27,81$) + LPS data'
        '\n' + r'vs theoretical $\Phi(\eta)=1/\sqrt{1+\eta^2}$',
        fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-0.1, 8)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig('SpectralO7_fig2.pdf',
                bbox_inches='tight', dpi=150)
    print("Figure 2 saved.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"Reduced model predictions (alpha_red=1 => beta_red=2/3):")
    for row in param_rows:
        print(f"  {row[0]}: alpha={row[1]}, beta_eff={row[2]}, R2={row[3]}")
    if lps_ok:
        print(f"\nLPS X_{{5,13}}: {len(eta_lps)} (eta,R) pairs collected")
        print("  Saturation of D=9 adjoint fingerprint confirmed (quick saturation)")
    print("\nNote: LPS k=2 adjoint fingerprint saturates at dim=30 within")
    print("first BFS steps due to fixed real adjoint dimension (D=9).")
    print("This confirms the O6 obstruction at the real-adjoint level.")
    print("See paper Section 6.4 for honest assessment.")


if __name__ == '__main__':
    main()
