"""
spectral_O12.py
Exact Weil-Block Projective Capacity on Heisenberg Graphs.
O12 of the Admissibility Sub-Programme -- Cosmochrony framework.

Computes, for each prime q in {211, 307, 401}:
  - BFS on Cay(Heis3(Z/qZ), {+/-X, +/-Y})
  - Per-block exact Weil projection for M_block sampled generic blocks H_c
  - Incremental block-wise projective capacity Sigma_n^(c) via Gram-Schmidt
  - Mean capacity Sigma_bar_n and inter-block variance V_n
  - Log-log fit of Sigma_bar_n -> delta_exact with OLS over [n0, n1]
  - Sensitivity analysis (window perturbation, M_block variation)
  - Coherence length ell_gamma(n) per shell (diagnostic for Section 6.4)
  - Collapse plot data (Sigma normalised, n/n*)
  - N_seed=5 independent block-sampling seeds -> sigma_seed(delta_hat)

Outputs (saved in current directory):
  fig1_per_block.pdf      -- per-block capacity curves (Fig 1)
  fig2_variance.pdf       -- inter-block variance V_n (Fig 2)
  fig3_loglog.pdf         -- log-log fit Sigma_bar_n (Fig 3)
  fig4_collapse.pdf       -- collapse plot (Fig 4, primary figure)
  fig5_coherence.pdf      -- coherence length ell_gamma(n) (diagnostic)
  table_delta.txt         -- Table 3: delta_exact extraction results
  table_sensitivity.txt   -- Table 4: sensitivity analysis at q=307
  table_e2.txt            -- Table: condition (E2) assessment
  table_beta.txt          -- Table 5: implied beta* ranges (placeholder)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import sys

RNG_BASE = 42
EPS_GS = 1e-10
EPS_SAT = 1e-3
E2_THRESH = 1.0
R2_THRESH = 0.97
WINDOW_MIN_PTS = 15

PRIMES = [211, 307, 401]
BFS_FRACS = {211: 0.30, 307: 0.20, 401: 0.10}
M_BLOCK_MAIN = {211: 500, 307: 500, 401: 300}
N_SEED = 5

SENSITIVITY_Q = 307


def heisenberg_group_elements(q):
    """
    Enumerate all elements of Heis3(Z/qZ) as triples (a, b, gamma).
    Returns dict mapping (a,b,gamma) -> index and inverse list.
    """
    elems = []
    idx = {}
    for a in range(q):
        for b in range(q):
            for g in range(q):
                idx[(a, b, g)] = len(elems)
                elems.append((a, b, g))
    return elems, idx


def heisenberg_mul(u, v, q):
    """Group multiplication: (a,b,g)*(a',b',g') = (a+a', b+b', g+g'+a*b') mod q."""
    a, b, gam = u
    ap, bp, gamp = v
    return ((a + ap) % q, (b + bp) % q, (gam + gamp + a * bp) % q)


def heisenberg_inv(u, q):
    """Inverse of (a,b,g) in Heis3(Z/qZ)."""
    a, b, gam = u
    return ((-a) % q, (-b) % q, (a * b - gam) % q)


def build_generators(q):
    """Standard symmetric generating set S = {±X, ±Y}."""
    X = (1, 0, 0)
    Y = (0, 1, 0)
    Xinv = heisenberg_inv(X, q)
    Yinv = heisenberg_inv(Y, q)
    return [X, Xinv, Y, Yinv]


def bfs_shells(elems, idx, gens, q, max_fraction):
    """
    BFS from identity, stopping when max_fraction of |G_q| nodes visited.
    elems/idx parameters kept for API compatibility but not used internally.
    Returns list of shells; shell[n] = list of (a,b,gamma) at depth n.
    """
    identity = (0, 0, 0)
    max_nodes = int(max_fraction * q ** 3)
    visited = {identity}
    current_shell = [identity]
    shells = [current_shell]
    total = 1

    while total < max_nodes:
        next_shell = []
        for u in current_shell:
            for g in gens:
                v = heisenberg_mul(u, g, q)
                if v not in visited:
                    visited.add(v)
                    next_shell.append(v)
                    total += 1
                    if total >= max_nodes:
                        break
            if total >= max_nodes:
                break
        if not next_shell:
            break
        shells.append(next_shell)
        current_shell = next_shell

    return shells


def sample_generic_blocks(q, m_block, rng):
    """
    Sample m_block generic multi-indices (c1,c2,c3) in (Z/qZ)^3.
    Generic: all ci != 0 and c1+c2+c3 != 0 (mod q).
    Returns array of shape (m_block, 3).
    """
    blocks = []
    attempts = 0
    max_attempts = m_block * 100
    while len(blocks) < m_block and attempts < max_attempts:
        c = rng.integers(1, q, size=3)
        if (c[0] + c[1] + c[2]) % q != 0:
            blocks.append(c)
        attempts += 1
    if len(blocks) < m_block:
        raise RuntimeError(f"Could not sample {m_block} generic blocks at q={q}")
    return np.array(blocks, dtype=np.int64)


def sample_generic_blocks_stratified(q, m_block, rng):
    """
    Stratified sampling: draw (c1,c2,c3) with c1<=c2<=c3 (up to permutation).
    """
    blocks = []
    attempts = 0
    max_attempts = m_block * 200
    while len(blocks) < m_block and attempts < max_attempts:
        c = np.sort(rng.integers(1, q, size=3))
        if (int(c[0]) + int(c[1]) + int(c[2])) % q != 0:
            blocks.append(c)
        attempts += 1
    if len(blocks) < m_block:
        blocks_uniform = sample_generic_blocks(q, m_block - len(blocks), rng)
        blocks.extend(blocks_uniform.tolist())
    return np.array(blocks[:m_block], dtype=np.int64)


def heisenberg_mul_batch(u_arr, g, q):
    """
    u_arr: (N, 3) int array; g: (3,) int array or tuple.
    Returns (N, 3) result of u * g in Heis3(Z/qZ).
    """
    ap, bp, gamp = int(g[0]), int(g[1]), int(g[2])
    out = u_arr.copy()
    out[:, 2] = (u_arr[:, 2] + gamp + u_arr[:, 0] * bp) % q
    out[:, 0] = (u_arr[:, 0] + ap) % q
    out[:, 1] = (u_arr[:, 1] + bp) % q
    return out


def make_psi_table(c, q):
    """Precompute lookup table psi_c(t) = exp(2pi i c t / q) for t in Z/qZ."""
    t = np.arange(q, dtype=np.int64)
    return np.exp(2j * np.pi * c * t / q)


def weil_batch_lut(a_arr, b_arr, gamma_arr, psi_table, q):
    """
    Vectorised exact Weil action using a precomputed LUT for phases.
    a_arr, b_arr, gamma_arr: (N,) int arrays.
    psi_table: (q,) complex array, psi_table[t] = exp(2pi i c t / q).
    Returns (N, q) complex array.
    """
    x_out = np.arange(q, dtype=np.int64)
    x_in = (x_out[None, :] - a_arr[:, None]) % q
    arg = (gamma_arr[:, None] + b_arr[:, None] * x_in) % q
    return psi_table[arg] / np.sqrt(q)


def fingerprint_vectors_batch(shell_arr, c_block, gens_arr, q):
    """
    Compute all k=3 fingerprint vectors for all elements of a shell.
    Uses LUT for phase computation.
    shell_arr: (M, 3); c_block: (3,); gens_arr: (4, 3).
    Returns: (M * 4^3, q) complex array.
    """
    c1, c2, c3 = int(c_block[0]), int(c_block[1]), int(c_block[2])
    psi1 = make_psi_table(c1, q)
    psi2 = make_psi_table(c2, q)
    psi3 = make_psi_table(c3, q)
    all_vecs = []
    for s1 in gens_arr:
        ep1 = heisenberg_mul_batch(shell_arr, s1, q)
        v1 = weil_batch_lut(ep1[:, 0], ep1[:, 1], ep1[:, 2], psi1, q)
        for s2 in gens_arr:
            ep2 = heisenberg_mul_batch(ep1, s2, q)
            v2 = weil_batch_lut(ep2[:, 0], ep2[:, 1], ep2[:, 2], psi2, q)
            for s3 in gens_arr:
                ep3 = heisenberg_mul_batch(ep2, s3, q)
                v3 = weil_batch_lut(ep3[:, 0], ep3[:, 1], ep3[:, 2], psi3, q)
                all_vecs.append(v1 * v2 * v3)
    return np.concatenate(all_vecs, axis=0)


def gram_schmidt_batch(basis_mat, new_vecs, eps=EPS_GS):
    """
    Orthogonalise rows of new_vecs against current orthonormal basis_mat.
    basis_mat: (k, q) complex array of current basis (may be None if empty).
    new_vecs: (N, q) complex array.
    Returns (new_basis_mat, delta_r) where delta_r = number of new directions added.
    """
    delta_r = 0
    if basis_mat is None:
        basis_mat = np.empty((0, new_vecs.shape[1]), dtype=np.complex128)

    for vec in new_vecs:
        if basis_mat.shape[0] >= new_vecs.shape[1]:
            break
        w = vec.copy()
        if basis_mat.shape[0] > 0:
            coeffs = basis_mat.conj() @ w
            w -= (basis_mat.T @ coeffs)
        norm = np.linalg.norm(w)
        if norm > eps:
            basis_mat = np.vstack([basis_mat, (w / norm)[None, :]])
            delta_r += 1

    return basis_mat, delta_r


CHUNK_SIZE = 400  # max elements per batch to stay under ~100MB


def compute_block_capacity(shells, c_block, q, gens, n_sat_min=1):
    """
    Compute Sigma_n^(c) for one generic block H_c over all shells.
    Vectorised with chunked processing to avoid memory overflow.
    Returns arrays: sigma_vals, delta_r_vals, shell_sizes, final_rank.
    """
    gens_arr = np.array(gens, dtype=np.int64)
    c_block = np.asarray(c_block, dtype=np.int64)

    basis_mat = None
    sigma_vals = []
    delta_r_vals = []
    shell_sizes = []

    for n, shell in enumerate(shells):
        if len(shell) == 0:
            break
        shell_arr = np.array(shell, dtype=np.int64)
        delta_r = 0

        # Process shell in chunks to limit peak memory
        for start in range(0, len(shell_arr), CHUNK_SIZE):
            chunk = shell_arr[start:start + CHUNK_SIZE]
            vecs = fingerprint_vectors_batch(chunk, c_block, gens_arr, q)
            basis_mat, dr = gram_schmidt_batch(basis_mat, vecs)
            delta_r += dr
            if basis_mat is not None and basis_mat.shape[0] >= q:
                break

        sz = len(shell)
        sigma = delta_r / sz if sz > 0 else 0.0
        sigma_vals.append(sigma)
        delta_r_vals.append(delta_r)
        shell_sizes.append(sz)

        if basis_mat is not None and basis_mat.shape[0] >= q:
            break

    final_rank = 0 if basis_mat is None else basis_mat.shape[0]
    return (np.array(sigma_vals),
            np.array(delta_r_vals),
            np.array(shell_sizes),
            final_rank)


def ols_loglog(ns, sigma_bar, n0, n1):
    """
    OLS fit: log(sigma_bar) = -delta * log(n) + log(C) over [n0, n1].
    Returns (delta_hat, C_hat, R2).
    """
    mask = (ns >= n0) & (ns <= n1) & (sigma_bar > 0)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    log_n = np.log(ns[mask])
    log_s = np.log(sigma_bar[mask])
    A = np.column_stack([log_n, np.ones(mask.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_s, rcond=None)
    slope, intercept = coeffs
    log_s_pred = A @ coeffs
    ss_res = np.sum((log_s - log_s_pred) ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return -slope, np.exp(intercept), r2


def find_fitting_window(ns, sigma_bar, q):
    """
    n0: smallest n s.t. |B_n| >= q (proxy: n0 = ceil(q^(1/3)) since |B_n|~n^4/24).
    n1: last n before sigma_bar < EPS_SAT.
    """
    # Rough estimate: |B_n| ~ 4 * n^4 / 24 for Heisenberg (homogeneous degree 4)
    # We want |B_n| >= q, so n >= (q * 6)^(1/4) approx
    n0_approx = max(2, int(np.ceil((q / 4.0) ** 0.25)))
    # Find first n in ns array >= n0_approx
    n0_idx = 0
    for i, n in enumerate(ns):
        if n >= n0_approx:
            n0_idx = i
            break

    # n1: last index where sigma_bar > EPS_SAT
    n1_idx = len(ns) - 1
    for i in range(len(ns) - 1, -1, -1):
        if sigma_bar[i] > EPS_SAT:
            n1_idx = i
            break

    n0 = int(ns[n0_idx])
    n1 = int(ns[n1_idx])
    return n0, n1


def coherence_length(shells, q):
    """
    Compute ell_gamma(n) = |mean_g_in_Sn exp(2pi i gamma_g / q)| per shell.
    This is the coherence of the central coordinate across the shell.
    """
    ell = []
    for shell in shells:
        if len(shell) == 0:
            ell.append(0.0)
            continue
        phases = np.array([np.exp(2j * np.pi * g / q) for (a, b, g) in shell])
        ell.append(abs(phases.mean()))
    return np.array(ell)


def run_one_prime(q, m_block, seed, stratified=False):
    """
    Run the full O12 computation for one prime q, one seed.
    Returns dict with all results.
    """
    t0 = time.time()
    rng = np.random.default_rng(RNG_BASE + seed * 1000 + q)

    print(f"  q={q}, seed={seed}, m_block={m_block}, stratified={stratified}")
    sys.stdout.flush()

    # Build generators (no need to enumerate all group elements)
    gens = build_generators(q)

    # BFS
    frac = BFS_FRACS[q]
    shells = bfs_shells(None, None, gens, q, frac)
    n_shells = len(shells)
    ns = np.arange(n_shells)
    shell_sizes = np.array([len(s) for s in shells])

    print(f"    BFS done: {n_shells} shells, {sum(shell_sizes)} nodes")
    sys.stdout.flush()

    # Coherence length (independent of block sampling)
    ell_gam = coherence_length(shells, q)

    # Sample generic blocks
    if stratified:
        blocks = sample_generic_blocks_stratified(q, m_block, rng)
    else:
        blocks = sample_generic_blocks(q, m_block, rng)

    # Per-block capacity
    all_sigma = []
    for i, c_block in enumerate(blocks):
        if i % 50 == 0:
            print(f"    block {i}/{m_block} ...")
            sys.stdout.flush()
        sigma_vals, _, _, rank_final = compute_block_capacity(shells, c_block, q, gens)
        # Pad to n_shells length
        pad = n_shells - len(sigma_vals)
        if pad > 0:
            sigma_vals = np.concatenate([sigma_vals, np.zeros(pad)])
        all_sigma.append(sigma_vals[:n_shells])

    all_sigma = np.array(all_sigma)  # shape (m_block, n_shells)

    # Mean and variance
    sigma_bar = all_sigma.mean(axis=0)
    sigma_var = all_sigma.var(axis=0)
    V_n = np.where(sigma_bar > 1e-15, sigma_var / sigma_bar ** 2, np.nan)

    # Fitting window
    n0, n1 = find_fitting_window(ns[1:], sigma_bar[1:], q)
    n0 = max(n0, 1)
    n1 = min(n1, n_shells - 1)

    # OLS fit
    delta_hat, C_hat, r2 = ols_loglog(ns, sigma_bar, n0, n1)

    # Check conditions
    mask_win = (ns >= n0) & (ns <= n1)
    e1_ok = (n1 - n0 + 1) >= WINDOW_MIN_PTS
    v_max_win = np.nanmax(V_n[mask_win]) if mask_win.any() else np.nan
    e2_ok = not np.isnan(v_max_win) and v_max_win < E2_THRESH
    e3_ok = not np.isnan(r2) and r2 >= R2_THRESH

    t1 = time.time()
    print(f"    delta_hat={delta_hat:.3f}, R2={r2:.4f}, "
          f"[n0,n1]=[{n0},{n1}], E1={e1_ok}, E2={e2_ok}(Vmax={v_max_win:.3f}), "
          f"E3={e3_ok}, time={t1-t0:.1f}s")

    return dict(
        q=q, seed=seed, m_block=m_block, stratified=stratified,
        shells=shells, ns=ns, shell_sizes=shell_sizes,
        all_sigma=all_sigma, sigma_bar=sigma_bar, sigma_var=sigma_var, V_n=V_n,
        n0=n0, n1=n1, delta_hat=delta_hat, C_hat=C_hat, r2=r2,
        e1_ok=e1_ok, e2_ok=e2_ok, e3_ok=e3_ok, v_max_win=v_max_win,
        ell_gam=ell_gam, blocks=blocks,
    )


def run_sensitivity(q, base_result):
    """
    Sensitivity analysis for one prime q.
    Perturbations: window ±2/±3, m_block 200/1000, stratified.
    Returns list of (label, delta_hat, delta_diff).
    """
    rows = []
    baseline_delta = base_result['delta_hat']

    sigma_bar = base_result['sigma_bar']
    ns = base_result['ns']
    n0_base = base_result['n0']
    n1_base = base_result['n1']

    def perturb_window(dn0, dn1):
        n0 = max(1, n0_base + dn0)
        n1 = min(len(ns) - 1, n1_base + dn1)
        d, _, r2 = ols_loglog(ns, sigma_bar, n0, n1)
        return d, r2

    rows.append(("Baseline", baseline_delta, 0.0))
    for (dn0, dn1, label) in [
        (-2, 0, "n0 -> n0-2"),
        (+2, 0, "n0 -> n0+2"),
        (0, -3, "n1 -> n1-3"),
        (0, +3, "n1 -> n1+3"),
    ]:
        d, r2 = perturb_window(dn0, dn1)
        rows.append((label, d, d - baseline_delta if not np.isnan(d) else np.nan))

    # M_block variations: re-run with different block counts
    for mb, label in [(200, "M_block=200"), (1000, "M_block=1000")]:
        try:
            rng = np.random.default_rng(RNG_BASE + q)
            gens = build_generators(q)
            shells = base_result['shells']
            blocks = sample_generic_blocks(q, mb, rng)
            all_s = []
            n_shells = len(shells)
            for c_block in blocks:
                sigma_vals, _, _, _ = compute_block_capacity(shells, c_block, q, gens)
                pad = n_shells - len(sigma_vals)
                if pad > 0:
                    sigma_vals = np.concatenate([sigma_vals, np.zeros(pad)])
                all_s.append(sigma_vals[:n_shells])
            sb = np.array(all_s).mean(axis=0)
            d, _, r2 = ols_loglog(base_result['ns'], sb, n0_base, n1_base)
            rows.append((label, d, d - baseline_delta if not np.isnan(d) else np.nan))
        except Exception as e:
            rows.append((label, np.nan, np.nan))

    # Stratified sampling
    try:
        rng = np.random.default_rng(RNG_BASE + q + 99)
        gens = build_generators(q)
        shells = base_result['shells']
        blocks = sample_generic_blocks_stratified(q, base_result['m_block'], rng)
        all_s = []
        n_shells = len(shells)
        for c_block in blocks:
            sigma_vals, _, _, _ = compute_block_capacity(shells, c_block, q, gens)
            pad = n_shells - len(sigma_vals)
            if pad > 0:
                sigma_vals = np.concatenate([sigma_vals, np.zeros(pad)])
            all_s.append(sigma_vals[:n_shells])
        sb = np.array(all_s).mean(axis=0)
        d, _, r2 = ols_loglog(base_result['ns'], sb, n0_base, n1_base)
        rows.append(("Stratified", d, d - baseline_delta if not np.isnan(d) else np.nan))
    except Exception as e:
        rows.append(("Stratified", np.nan, np.nan))

    return rows


def make_figure1(results_by_q):
    """Per-block capacity curves (Fig 1)."""
    fig, axes = plt.subplots(1, len(results_by_q), figsize=(14, 4), sharey=True)
    if len(results_by_q) == 1:
        axes = [axes]
    for ax, (q, res) in zip(axes, results_by_q.items()):
        ns = res['ns']
        all_sigma = res['all_sigma']
        sigma_bar = res['sigma_bar']
        sigma_std = np.sqrt(res['sigma_var'])
        n0, n1 = res['n0'], res['n1']
        # Plot individual blocks (thin grey)
        for i in range(min(50, all_sigma.shape[0])):
            ax.plot(ns[1:], all_sigma[i, 1:], color='grey', alpha=0.25, lw=0.5)
        # Mean and std band
        ax.plot(ns[1:], sigma_bar[1:], 'k-', lw=2, label=r'$\bar\Sigma_n$')
        ax.fill_between(ns[1:],
                        np.clip(sigma_bar[1:] - sigma_std[1:], 0, None),
                        sigma_bar[1:] + sigma_std[1:],
                        alpha=0.3, color='steelblue', label=r'$\pm 1\sigma$')
        ax.axvline(n0, color='r', ls='--', lw=1, label=f'$n_0={n0}$')
        ax.axvline(n1, color='b', ls='--', lw=1, label=f'$n_1={n1}$')
        ax.set_xlabel('BFS depth $n$', fontsize=11)
        ax.set_title(f'$q={q}$', fontsize=12)
        ax.legend(fontsize=7, loc='upper right')
    axes[0].set_ylabel(r'$\Sigma_n^{(c)}$', fontsize=11)
    fig.suptitle(
        r'Per-block projective capacity $\Sigma_n^{(c)}$ (exact Weil projection) --- SpectralO12',
        fontsize=10)
    plt.tight_layout()
    plt.savefig('fig1_per_block.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig1_per_block.pdf")


def make_figure2(results_by_q):
    """Inter-block variance ratio V_n (Fig 2)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for (q, res), col in zip(results_by_q.items(), colors):
        ns = res['ns']
        V_n = res['V_n']
        sigma_bar = res['sigma_bar']
        mask = sigma_bar > 1e-5
        ax.plot(ns[mask], V_n[mask], color=col, lw=1.8, label=f'$q={q}$')
    ax.axhline(E2_THRESH, color='red', ls='--', lw=1.5, label=r'$\varepsilon_E=1$ (E2 threshold)')
    ax.set_xlabel('BFS depth $n$', fontsize=11)
    ax.set_ylabel(r'$V_n = \mathrm{Var}_c(\Sigma_n^{(c)}) / \bar\Sigma_n^2$', fontsize=11)
    ax.set_title(
        r'Inter-block variance ratio (condition E2) --- SpectralO12', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('fig2_variance.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig2_variance.pdf")


def make_figure3(results_by_q):
    """Log-log fit of Sigma_bar_n (Fig 3)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for (q, res), col in zip(results_by_q.items(), colors):
        ns = res['ns']
        sigma_bar = res['sigma_bar']
        n0, n1 = res['n0'], res['n1']
        delta_hat = res['delta_hat']
        C_hat = res['C_hat']
        r2 = res['r2']
        mask = sigma_bar > 0
        ax.loglog(ns[mask][1:], sigma_bar[mask][1:], 'o', color=col,
                  ms=3, label=f'$q={q}$')
        # Fit line
        ns_fit = np.linspace(n0, n1, 100)
        if not np.isnan(delta_hat):
            ax.loglog(ns_fit, C_hat * ns_fit ** (-delta_hat), '--',
                      color=col, lw=1.5,
                      label=f'fit $q={q}$: $\\hat{{\\delta}}={delta_hat:.2f}$, $R^2={r2:.3f}$')
    ax.set_xlabel('BFS depth $n$', fontsize=11)
    ax.set_ylabel(r'$\bar\Sigma_n$', fontsize=11)
    ax.set_title(
        r'Log-log decay of mean block capacity $\bar\Sigma_n$ --- SpectralO12',
        fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig3_loglog.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig3_loglog.pdf")


def make_figure4(results_by_q):
    """Collapse plot (Fig 4 -- primary figure)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['steelblue', 'darkorange', 'forestgreen']
    all_x, all_y = [], []

    for (q, res), col in zip(results_by_q.items(), colors):
        ns = res['ns']
        all_sigma = res['all_sigma']
        sigma_bar = res['sigma_bar']
        n0, n1 = res['n0'], res['n1']
        n_star = n1

        for i in range(min(50, all_sigma.shape[0])):
            sig = all_sigma[i]
            # Amplitude C_c: mean in fitting window
            mask_win = (ns >= n0) & (ns <= n1) & (sig > 0)
            if mask_win.sum() < 3:
                continue
            # Normalise: Sigma_tilde = Sigma / C_c
            ns_w = ns[mask_win].astype(float)
            sig_w = sig[mask_win]
            C_c = np.exp(np.mean(np.log(sig_w) + res['delta_hat'] * np.log(ns_w)))
            if C_c <= 0 or np.isnan(C_c):
                continue
            mask_pos = (ns >= n0) & (sig > 0)
            x_vals = ns[mask_pos] / n_star
            y_vals = sig[mask_pos] / C_c
            ax.loglog(x_vals, y_vals, color=col, alpha=0.15, lw=0.5)
            all_x.extend(x_vals.tolist())
            all_y.extend(y_vals.tolist())
        # Mean curve for this q
        mask_pos = (ns >= n0) & (sigma_bar > 0)
        ax.loglog(ns[mask_pos] / n_star, sigma_bar[mask_pos], color=col, lw=2,
                  label=f'$q={q}$ mean')

    # Master power law using converged delta
    qs_list = list(results_by_q.keys())
    deltas = [results_by_q[q]['delta_hat'] for q in qs_list
              if not np.isnan(results_by_q[q]['delta_hat'])]
    if deltas:
        delta_conv = deltas[-1]
        x_master = np.logspace(-1, 0.3, 100)
        ax.loglog(x_master, x_master ** (-delta_conv), 'k--', lw=2,
                  label=f'Master: $(n/n^*)^{{-{delta_conv:.2f}}}$')

    ax.set_xlabel(r'Rescaled depth $n/n^*$', fontsize=11)
    ax.set_ylabel(r'Normalised capacity $\tilde\Sigma_n^{(c)} = \Sigma_n^{(c)}/C_c$', fontsize=11)
    ax.set_title(
        r'Collapse plot: universality across $q$ and blocks --- SpectralO12',
        fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('fig4_collapse.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig4_collapse.pdf")


def make_figure5(results_by_q):
    """Coherence length ell_gamma(n) (Fig 5 -- diagnostic)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for (q, res), col in zip(results_by_q.items(), colors):
        ns = res['ns']
        ell = res['ell_gam']
        ax.plot(ns[1:], ell[1:], color=col, lw=1.8, label=f'$q={q}$')
    ax.axhline(1.0, color='grey', ls=':', lw=1, label='Full coherence')
    ax.axhline(0.0, color='grey', ls=':', lw=1)
    ax.set_xlabel('BFS depth $n$', fontsize=11)
    ax.set_ylabel(r'$\ell_\gamma(n)$', fontsize=11)
    ax.set_title(
        r'Central coordinate coherence $\ell_\gamma(n)$ per shell --- SpectralO12',
        fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('fig5_coherence.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig5_coherence.pdf")


def write_table_delta(seed_results_by_q):
    """
    Table 3: delta_exact extraction.
    seed_results_by_q[q] = list of per-seed result dicts.
    """
    lines = []
    lines.append("Table 3: Exact Weil-block delta_exact extraction")
    lines.append("=" * 80)
    hdr = f"{'q':>5} {'n*':>5} {'[n0,n1]':>12} {'delta_hat':>10} "
    hdr += f"{'sigma_seed':>12} {'R2':>7} {'E1':>4} {'E2':>4} {'E3':>4}"
    lines.append(hdr)
    lines.append("-" * 80)

    for q in PRIMES:
        seed_res = seed_results_by_q[q]
        deltas = [r['delta_hat'] for r in seed_res if not np.isnan(r['delta_hat'])]
        r2s = [r['r2'] for r in seed_res if not np.isnan(r['r2'])]
        n0s = [r['n0'] for r in seed_res]
        n1s = [r['n1'] for r in seed_res]
        n_stars = [r['n1'] for r in seed_res]

        delta_mean = np.mean(deltas) if deltas else np.nan
        delta_std = np.std(deltas) if len(deltas) > 1 else np.nan
        r2_mean = np.mean(r2s) if r2s else np.nan
        n0_rep = int(np.median(n0s)) if n0s else -1
        n1_rep = int(np.median(n1s)) if n1s else -1
        n_star_rep = n1_rep

        e1 = all(r['e1_ok'] for r in seed_res)
        e2 = all(r['e2_ok'] for r in seed_res)
        e3 = all(r['e3_ok'] for r in seed_res)

        row = (f"{q:>5} {n_star_rep:>5} [{n0_rep},{n1_rep}]{'':{6}} "
               f"{delta_mean:>10.3f} {delta_std:>12.4f} {r2_mean:>7.4f} "
               f"{'Yes' if e1 else 'No':>4} {'Yes' if e2 else 'No':>4} "
               f"{'Yes' if e3 else 'No':>4}")
        lines.append(row)

    lines.append("-" * 80)
    lines.append(f"{'O11 proxy':>5} {'22':>5} {'[10,27]':>12} {'3.390':>10} "
                 f"{'---':>12} {'0.987':>7} {'Yes':>4} {'Yes':>4} {'Yes':>4}")
    lines.append("")
    lines.append("sigma_seed = inter-seed std of delta_hat across N_seed=5 seeds.")

    with open('table_delta.txt', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved table_delta.txt")


def write_table_e2(seed_results_by_q):
    """Table: condition E2 assessment."""
    lines = []
    lines.append("Table E2: Inter-block variance condition (E2)")
    lines.append("=" * 60)
    lines.append(f"{'q':>5} {'max V_n (window)':>20} {'E2 satisfied?':>15}")
    lines.append("-" * 60)
    for q in PRIMES:
        seed_res = seed_results_by_q[q]
        vmaxs = [r['v_max_win'] for r in seed_res if not np.isnan(r['v_max_win'])]
        vmax_mean = np.mean(vmaxs) if vmaxs else np.nan
        e2 = all(r['e2_ok'] for r in seed_res)
        lines.append(f"{q:>5} {vmax_mean:>20.4f} {'Yes' if e2 else 'No':>15}")
    with open('table_e2.txt', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved table_e2.txt")


def write_table_sensitivity(sens_rows):
    """Table 4: sensitivity at q=SENSITIVITY_Q."""
    lines = []
    lines.append(f"Table 4: Sensitivity of delta_exact at q={SENSITIVITY_Q}")
    lines.append("=" * 55)
    lines.append(f"{'Perturbation':>30} {'delta_hat':>10} {'Delta delta':>12}")
    lines.append("-" * 55)
    for (label, dhat, ddelta) in sens_rows:
        dhat_str = f"{dhat:.3f}" if not np.isnan(dhat) else "---"
        ddelta_str = f"{ddelta:+.3f}" if not np.isnan(ddelta) else "---"
        lines.append(f"{label:>30} {dhat_str:>10} {ddelta_str:>12}")
    with open('table_sensitivity.txt', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved table_sensitivity.txt")


def write_table_beta(seed_results_by_q):
    """
    Table 5: implied beta* ranges.
    Placeholder: the f(beta) relation from O3-O7 is needed for the actual numbers.
    We write the delta values and mark beta* as [PENDING f(beta) from O3].
    """
    lines = []
    lines.append("Table 5: Implied beta* constraint from delta_exact")
    lines.append("(beta* = g(delta) via O3-O7 structural relation -- see Section 7)")
    lines.append("=" * 70)
    lines.append(f"{'q':>5} {'delta_exact':>13} {'Implied beta* range':>25} "
                 f"{'Compatible (0.09,0.13)?':>25}")
    lines.append("-" * 70)
    for q in PRIMES:
        seed_res = seed_results_by_q[q]
        deltas = [r['delta_hat'] for r in seed_res if not np.isnan(r['delta_hat'])]
        delta_mean = np.mean(deltas) if deltas else np.nan
        delta_std = np.std(deltas) if len(deltas) > 1 else 0.0
        dlo = delta_mean - delta_std if not np.isnan(delta_mean) else np.nan
        dhi = delta_mean + delta_std if not np.isnan(delta_mean) else np.nan
        delta_str = f"{delta_mean:.3f} +/- {delta_std:.3f}" if not np.isnan(delta_mean) else "---"
        lines.append(f"{q:>5} {delta_str:>13} {'[PENDING f(beta)]':>25} {'[PENDING]':>25}")
    lines.append("")
    lines.append("NOTE: f(beta) must be extracted from O3-O7 papers to fill this table.")
    lines.append("Phenomenological target: beta* in (0.09, 0.13) from O3 lepton mass ratios.")
    with open('table_beta.txt', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved table_beta.txt")


def main():
    print("O12 -- Exact Weil-Block Projective Capacity on Heisenberg Graphs")
    print("=" * 65)

    seed_results_by_q = {}
    best_result_by_q = {}

    for q in PRIMES:
        print(f"\n--- Prime q={q} ---")
        m_block = M_BLOCK_MAIN[q]
        seed_results = []
        for seed in range(N_SEED):
            res = run_one_prime(q, m_block, seed)
            seed_results.append(res)
        seed_results_by_q[q] = seed_results
        # Best result = seed 0 (used for figures)
        best_result_by_q[q] = seed_results[0]

    print("\n--- Sensitivity analysis (q={}) ---".format(SENSITIVITY_Q))
    base = best_result_by_q[SENSITIVITY_Q]
    sens_rows = run_sensitivity(SENSITIVITY_Q, base)

    print("\n--- Generating figures ---")
    make_figure1(best_result_by_q)
    make_figure2(best_result_by_q)
    make_figure3(best_result_by_q)
    make_figure4(best_result_by_q)
    make_figure5(best_result_by_q)

    print("\n--- Writing tables ---")
    write_table_delta(seed_results_by_q)
    write_table_e2(seed_results_by_q)
    write_table_sensitivity(sens_rows)
    write_table_beta(seed_results_by_q)

    print("\nDone.")


if __name__ == '__main__':
    main()
