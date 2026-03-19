#!/usr/bin/env python3
"""
SpectralO5 -- Numerical Computations
======================================
All scripts supporting the paper
  "Admissible Frontier Saturation and the Cascade Exponent"
  J. Beau, Cosmochrony O-Series, 2026

Sections:
  §1  Shared utilities (group construction, P^1 action, irreps)
  §2  Figure 1: Version A vs B (character-based) on X^{5,13}
  §3  Figure 2: Version B character, q-dependence on X^{5,q}
  §4  Figure 3: Matrix-based variants M1/M2/M3/M4 on X^{5,13}
  §5  Figure 4: Steinberg St_elem pre-saturation law, q in {13,17,29}
  §6  Main: generate all figures

Usage:
  python3 SpectralO5_computations.py          # generate all figures
  python3 SpectralO5_computations.py --fig N  # generate figure N only (1-4)

Dependencies: numpy, scipy, matplotlib (standard scientific Python stack)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# §1  SHARED UTILITIES
# ================================================================

def mod_inv(a, q):
    """Modular inverse of a mod q (q prime)."""
    return pow(int(a), q - 2, q)

def sqrt_mod(a, q):
    """Square root of a mod q using Tonelli-Shanks. Returns None if not square."""
    a = int(a) % q
    if a == 0:
        return 0
    if pow(a, (q - 1) // 2, q) != 1:
        return None
    if q % 4 == 3:
        return pow(a, (q + 1) // 4, q)
    s, n = 0, q - 1
    while n % 2 == 0:
        s += 1
        n //= 2
    z = 2
    while pow(z, (q - 1) // 2, q) != q - 1:
        z += 1
    M, c, t, R = s, pow(z, n, q), pow(a, n, q), pow(a, (n + 1) // 2, q)
    while True:
        if t == 1:
            return R
        i, tmp = 1, (t * t) % q
        while tmp != 1:
            tmp = (tmp * tmp) % q
            i += 1
        b = pow(c, 1 << (M - i - 1), q)
        M, c, t, R = i, (b * b) % q, (t * b * b) % q, (R * b) % q

def is_square(a, q):
    """Check if a is a quadratic residue mod q."""
    return a == 0 or pow(int(a) % q, (q - 1) // 2, q) == 1

def mat_mul_mod(A, B, q):
    """Multiply two 2x2 integer matrices mod q."""
    return np.array([
        [(A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0]) % q,
         (A[0, 0]*B[0, 1] + A[0, 1]*B[1, 1]) % q],
        [(A[1, 0]*B[0, 0] + A[1, 1]*B[1, 0]) % q,
         (A[1, 0]*B[0, 1] + A[1, 1]*B[1, 1]) % q]
    ], dtype=np.int64)

def mat_to_key(A, q):
    """Canonical key for a PSL(2,F_q) element (mod ±I quotient)."""
    a, b, c, d = int(A[0, 0]), int(A[0, 1]), int(A[1, 0]), int(A[1, 1])
    return min((a, b, c, d), ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q))

def four_square(p):
    """Find (a,b,c,d) with a^2+b^2+c^2+d^2=p, a>0 odd, b,c,d even."""
    from itertools import product as ip
    for a in range(1, p, 2):
        for b in range(0, p, 2):
            for c in range(0, p, 2):
                d2 = p - a*a - b*b - c*c
                if d2 < 0:
                    continue
                d = int(d2**0.5 + 0.5)
                if d*d == d2 and d % 2 == 0:
                    return a, b, c, d

def build_graph(q, p=5):
    """
    Build the LPS Cayley graph X^{p,q} on PSL(2,F_q).
    Returns: adj (adjacency list), elems (list of canonical keys),
             elem_mats (list of actual 2x2 matrices), gen_mats (generator matrices).
    """
    i_val = next(x for x in range(1, q) if (x*x + 1) % q == 0)
    a0, b0, c0, d0 = four_square(p)
    from itertools import product as ip, permutations
    parts = [a0, b0, c0, d0]
    seen = set()
    for perm in permutations(range(4)):
        for signs in ip([-1, 1], repeat=4):
            vals = tuple(signs[j]*parts[perm[j]] for j in range(4))
            if vals[0] > 0 and sum(v*v for v in vals) == p:
                seen.add(vals)
    gen_quats = list(seen)[:p + 1]
    gen_mats = [
        np.array([[(aa + bb*i_val) % q, (cc + dd*i_val) % q],
                  [(-cc + dd*i_val) % q, (aa - bb*i_val) % q]], dtype=np.int64)
        for (aa, bb, cc, dd) in gen_quats
    ]
    I = np.array([[1, 0], [0, 1]], dtype=np.int64)
    elems = [mat_to_key(I, q)]
    eidx = {elems[0]: 0}
    elem_mats = [I.copy()]
    adj = {0: []}
    queue = deque([I])
    while queue:
        u_mat = queue.popleft()
        u_key = mat_to_key(u_mat, q)
        u_idx = eidx[u_key]
        if u_idx not in adj:
            adj[u_idx] = []
        for gi, G in enumerate(gen_mats):
            v_mat = mat_mul_mod(u_mat, G, q)
            v_key = mat_to_key(v_mat, q)
            if v_key not in eidx:
                v_idx = len(elems)
                elems.append(v_key)
                elem_mats.append(v_mat.copy())
                eidx[v_key] = v_idx
                adj[v_idx] = []
                queue.append(v_mat)
            else:
                v_idx = eidx[v_key]
            adj[u_idx].append((v_idx, gi))
    return adj, elems, elem_mats, gen_mats

def build_irreps(q, p=5):
    """
    Build admissible irreps of PSL(2,F_q) and their character functions.
    Returns: reps (list of (name, dim, char_func)), adm_idx, mu_vals, lam_star, d.
    """
    d = p + 1
    g_prim = next(x for x in range(2, q) if pow(x, (q-1)//2, q) != 1)
    dlog = {}
    gj = 1
    for j in range(q - 1):
        dlog[gj] = j
        gj = (gj * g_prim) % q

    def chi_Fq(t, k):
        if t == 0:
            return 0.0
        return np.cos(2 * np.pi * k * dlog[int(t) % q] / (q - 1))

    def pi_k_char(ct, tr, k):
        if ct == 'I':
            return float(q + 1)
        if ct == 'U':
            return 1.0
        if ct == 'Ss':
            tau = int(tr) % q
            disc = (tau*tau - 4) % q
            sq = sqrt_mod(disc, q)
            if sq is None:
                return 0.0
            t1 = ((tau + sq) * mod_inv(2, q)) % q
            t2 = ((tau - sq) * mod_inv(2, q)) % q
            return chi_Fq(t1, k) + chi_Fq(t2, k)
        return 0.0

    reps = [
        ('trivial', 1, lambda ct, tr: 1.0),
        ('Steinberg', q, lambda ct, tr: float(q) if ct == 'I' else
                                        0.0 if ct == 'U' else -1.0)
    ]
    for k in range(1, (q - 3) // 2 + 1):
        kk = k
        def mk(kk):
            return lambda ct, tr: pi_k_char(ct, tr, kk)
        reps.append((f'pi_{k}', q + 1, mk(k)))
    for l in range(1, (q - 1) // 2 + 1):
        def ms(ll):
            def f(ct, tr):
                if ct == 'I':
                    return float(q - 1)
                if ct == 'U':
                    return -1.0
                return 0.0
            return f
        reps.append((f'sigma_{l}', q - 1, ms(l)))

    # Determine generator conjugacy class
    i_val = next(x for x in range(1, q) if (x*x + 1) % q == 0)
    aa, bb, cc, dd = four_square(p)
    M = np.array([[(aa + bb*i_val) % q, (cc + dd*i_val) % q],
                  [(-cc + dd*i_val) % q, (aa - bb*i_val) % q]], dtype=np.int64)
    tr = int(M[0, 0] + M[1, 1]) % q
    tr = min(tr, (q - tr) % q)
    disc = (tr*tr - 4) % q
    ct_gen = 'U' if disc == 0 else ('Ss' if is_square(disc, q) else 'Se')

    lam_star = (np.sqrt(d - 1) + 1)**2
    adm_idx = []
    mu_vals = []
    for k, (name, dim, chi) in enumerate(reps):
        mu = d * chi(ct_gen, tr) / dim
        mu_vals.append(mu)
        lam = d - mu
        if 0.01 < lam <= lam_star + 0.01 and k > 0:
            adm_idx.append(k)
    return reps, adm_idx, mu_vals, lam_star, d

def vertex_chars(elems, q, reps, adm_idx):
    """Compute pi_A(v) = (kappa_rho * chi_rho(v)) for all vertices. Returns (n, n_adm) matrix."""
    n, n_adm = len(elems), len(adm_idx)
    Pi = np.zeros((n, n_adm))
    cache = {}
    adm_reps = [reps[k] for k in adm_idx]
    for vi, key in enumerate(elems):
        a, b, c, dd = key
        tr = min(int(a + dd) % q, (q - int(a + dd) % q) % q)
        disc = (tr*tr - 4) % q
        ct = 'U' if disc == 0 else ('Ss' if is_square(disc, q) else 'Se')
        ck = (ct, tr)
        if ck not in cache:
            cache[ck] = np.array([chi(ct, tr) for (_, _, chi) in adm_reps])
        Pi[vi, :] = cache[ck]
    return Pi.real, len(cache)

def action_on_P1(M, q):
    """Permutation of P^1(F_q) induced by M in PSL(2,F_q). Returns array of length q+1."""
    a, b, c, d = int(M[0, 0]), int(M[0, 1]), int(M[1, 0]), int(M[1, 1])
    sigma = np.zeros(q + 1, dtype=np.int64)
    for i in range(q):
        num = (a*i + b) % q
        den = (c*i + d) % q
        sigma[i] = (num * mod_inv(den, q)) % q if den != 0 else q
    sigma[q] = (a * mod_inv(c, q)) % q if c != 0 else q
    return sigma

def build_steinberg_projections(elems, elem_mats, gen_mats, q):
    """Compute Steinberg projections tilde_sigma for all vertices and generators."""
    steins = np.zeros((len(elems), q + 1))
    for i, M in enumerate(elem_mats):
        s = action_on_P1(M, q).astype(float)
        steins[i] = s - s.mean()
    gen_steins = np.zeros((len(gen_mats), q + 1))
    for gi, G in enumerate(gen_mats):
        s = action_on_P1(G, q).astype(float)
        gen_steins[gi] = s - s.mean()
    return steins, gen_steins

# ----------------------------------------------------------------
# Generic BFS cascade measurement
# ----------------------------------------------------------------

def bfs_cascade(adj, fp_func, fp_dim, n_verts, max_steps=200):
    """
    BFS cascade measuring dim Pi, r_tilde, admissible count at each step.
    fp_func(u_idx, v_idx, gi) -> R^fp_dim fingerprint vector.
    Returns dict of arrays: Sn, dS, dim_Pi, rt, adm_count.
    """
    S = set([0])
    frontier = {0: (None, None)}  # v -> (parent_u, generator_idx)
    added = set([0])
    bfs_q = deque([0])
    span = np.zeros((0, fp_dim))

    res = {'Sn': [], 'dS': [], 'dim_Pi': [], 'rt': [], 'adm_count': []}

    def rank_Vt(sp):
        if len(sp) == 0:
            return 0, None
        sv = np.linalg.svd(sp, compute_uv=False)
        r = int(np.sum(sv > 1e-8))
        if r == 0:
            return 0, None
        _, _, Vt = np.linalg.svd(sp, full_matrices=False)
        return r, Vt[:r, :]

    for step in range(max_steps):
        if not frontier:
            break
        rank, Vt_r = rank_Vt(span)
        adm_c = 0
        for v, (u, gi) in frontier.items():
            if u is None:
                continue
            ev = fp_func(u, v, gi)
            if rank == 0 or np.all(np.abs(ev) < 1e-10):
                adm_c += 1
            else:
                proj = Vt_r.T @ (Vt_r @ ev)
                if np.linalg.norm(ev - proj) > 1e-8:
                    adm_c += 1
        dS = len(frontier)
        res['Sn'].append(len(S))
        res['dS'].append(dS)
        res['dim_Pi'].append(rank)
        res['adm_count'].append(adm_c)
        res['rt'].append(adm_c / dS if dS > 0 else 0)

        v_new = None
        while bfs_q:
            cand = bfs_q.popleft()
            if cand in frontier:
                v_new = cand
                break
        if v_new is None:
            v_new = next(iter(frontier))
        u_new, gi_new = frontier[v_new]
        if u_new is not None:
            new_fp = fp_func(u_new, v_new, gi_new).reshape(1, -1)
            span = np.vstack([span, new_fp]) if len(span) > 0 else new_fp
        S.add(v_new)
        del frontier[v_new]
        for (nb, gi) in adj.get(v_new, []):
            if nb < n_verts and nb not in added:
                frontier[nb] = (v_new, gi)
                bfs_q.append(nb)
                added.add(nb)

    return {k: np.array(v) for k, v in res.items()}


# ================================================================
# §2  FIGURE 1 -- Version A vs B (character-based) on X^{5,13}
# ================================================================

def figure1_A_vs_B(q=13, p=5, outfile='fig1_A_vs_B.png'):
    print(f"Figure 1: Version A vs B (character-based), q={q}")
    adj, elems, elem_mats, gen_mats = build_graph(q, p)
    n = len(elems)
    reps, adm_idx, mu_vals, lam_star, d = build_irreps(q, p)
    Pi_mat, n_conj = vertex_chars(elems, q, reps, adm_idx)
    n_adm = len(adm_idx)
    kappa = np.array([mu_vals[k] / d for k in adm_idx])
    rank_Madm = np.linalg.matrix_rank(Pi_mat, tol=1e-6)

    # Version A fingerprint
    def fp_A(u, v, gi):
        return Pi_mat[v, :]

    # Version B (character) fingerprint
    def fp_B(u, v, gi):
        return kappa * Pi_mat[u, :] * Pi_mat[v, :]

    print(f"  |G|={n}, n_adm={n_adm}, rank(M_adm)={rank_Madm}")
    res_A = bfs_cascade(adj, fp_A, n_adm, n, max_steps=100)
    res_B = bfs_cascade(adj, fp_B, n_adm, n, max_steps=100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f'Version A vs B (character-based) on $X^{{{p},{q}}}$\n'
        f'$|G|={n}$, $n_{{\\rm adm}}={n_adm}$, '
        f'$\\mathrm{{rank}}(M_{{\\rm adm}})={rank_Madm}$',
        fontsize=12
    )
    Sn_A, Sn_B = res_A['Sn'], res_B['Sn']

    ax = axes[0, 0]
    ax.plot(Sn_A, res_A['dim_Pi'], 'g-o', ms=3, label=r'$\dim\Pi_A(S_n)$')
    ax.plot(Sn_B, res_B['dim_Pi'], 'b-s', ms=3, label=r'$\dim\Pi_B(S_n)$', alpha=0.8)
    ax.axhline(rank_Madm, color='g', linestyle='--',
               label=f'$\\mathrm{{rank}}(M_{{\\rm adm}})={rank_Madm}$')
    ax.axhline(n_adm, color='b', linestyle=':',
               label=f'$n_{{\\rm adm}}={n_adm}$')
    ax.set_xlabel(r'$|S_n|$')
    ax.set_ylabel(r'$\dim\Pi(S_n)$')
    ax.set_title('Rank of admissible span')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(Sn_A, res_A['rt'], 'g-o', ms=3, label=r'$\tilde{r}^A_n$ (vertex)')
    ax.plot(Sn_B, res_B['rt'], 'b-s', ms=3, label=r'$\tilde{r}^B_n$ (character)', alpha=0.8)
    ax.set_xlabel(r'$|S_n|$')
    ax.set_ylabel(r'$\tilde{r}_n$')
    ax.set_title(r'Admissible frontier fraction')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 0]
    r_A = np.array(res_A['dim_Pi'], dtype=float) / np.maximum(res_A['Sn'], 1)
    ax.plot(Sn_A, r_A, 'g-o', ms=3)
    ax.set_xlabel(r'$|S_n|$')
    ax.set_ylabel(r'$r_n = \dim\Pi_A/|S_n|$')
    ax.set_title(r'Rank-to-size ratio $r_n \to 0$ (Corollary~1)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"KEY NUMBERS ($q={q}$, $p={p}$):\n\n"
        f"  $|G| = {n}$\n"
        f"  $n_{{\\rm adm}} = {n_adm}$\n"
        f"  $\\mathrm{{rank}}(M_{{\\rm adm}}) = {rank_Madm}$\n\n"
        f"VERSION A:\n"
        f"  Saturates at $|S^*| \\approx "
        f"{next((res_A['Sn'][i] for i,d_ in enumerate(res_A['dim_Pi']) if d_>=rank_Madm), '?')}$\n"
        f"  Late $\\tilde{{r}}^A$: {np.mean(res_A['rt'][-5:]):.3f}\n\n"
        f"VERSION B (character):\n"
        f"  Max $\\dim\\Pi_B$: {max(res_B['dim_Pi'])}\n"
        f"  Late $\\tilde{{r}}^B$: {np.mean(res_B['rt'][-5:]):.3f}\n"
        f"  (No decay -- character collapse)"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Summary')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Saved {outfile}")


# ================================================================
# §3  FIGURE 2 -- Version B character q-dependence
# ================================================================

def figure2_versionB_qdep(qs=(13, 29), p=5, outfile='fig2_versionB_sat.png'):
    print(f"Figure 2: Version B character q-dependence, q={qs}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r'Character-based Version B: $q$-dependence on $X^{5,q}$',
        fontsize=12
    )
    colors = {13: 'green', 17: 'blue', 29: 'orange', 41: 'red'}

    for q in qs:
        adj, elems, elem_mats, gen_mats = build_graph(q, p)
        n = len(elems)
        reps, adm_idx, mu_vals, lam_star, d = build_irreps(q, p)
        Pi_mat, _ = vertex_chars(elems, q, reps, adm_idx)
        n_adm = len(adm_idx)
        kappa = np.array([mu_vals[k] / d for k in adm_idx])
        rank_Madm = np.linalg.matrix_rank(Pi_mat, tol=1e-6)

        def fp_A(u, v, gi, Pi=Pi_mat):
            return Pi[v, :]

        def fp_B(u, v, gi, Pi=Pi_mat, kap=kappa):
            return kap * Pi[u, :] * Pi[v, :]

        res_A = bfs_cascade(adj, fp_A, n_adm, n, max_steps=100)
        res_B = bfs_cascade(adj, fp_B, n_adm, n, max_steps=100)
        col = colors.get(q, 'gray')

        axes[0].plot(res_A['Sn'], res_A['dim_Pi'],
                     color=col, linestyle='-', marker='o', ms=2,
                     label=f'$\\dim\\Pi_A$, $q={q}$')
        axes[0].plot(res_B['Sn'], res_B['dim_Pi'],
                     color=col, linestyle='--', marker='s', ms=2,
                     label=f'$\\dim\\Pi_B$, $q={q}$', alpha=0.7)
        axes[0].axhline(rank_Madm, color=col, linestyle=':', alpha=0.4)

        axes[1].plot(res_A['Sn'], res_A['rt'],
                     color=col, linestyle='-', marker='o', ms=2,
                     label=f'$\\tilde{{r}}^A$, $q={q}$')
        axes[1].plot(res_B['Sn'], res_B['rt'],
                     color=col, linestyle='--', marker='s', ms=2,
                     label=f'$\\tilde{{r}}^B$, $q={q}$', alpha=0.7)

    axes[0].set_xlabel(r'$|S_n|$')
    axes[0].set_ylabel(r'$\dim\Pi(S_n)$')
    axes[0].set_title('Rank of admissible spans')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'$|S_n|$')
    axes[1].set_ylabel(r'$\tilde{r}_n$')
    axes[1].set_title(r'Admissible frontier fraction')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Saved {outfile}")


# ================================================================
# §4  FIGURE 3 -- Matrix variants M1/M2/M3/M4 on X^{5,13}
# ================================================================

def figure3_matrix_variants(q=13, p=5, outfile='fig3_matB_variants.png'):
    print(f"Figure 3: Matrix variants M1/M2/M3/M4, q={q}")
    adj, elems, elem_mats, gen_mats = build_graph(q, p)
    n = len(elems)
    reps, adm_idx, mu_vals, lam_star, d = build_irreps(q, p)
    Pi_mat, _ = vertex_chars(elems, q, reps, adm_idx)
    n_adm = len(adm_idx)
    kappa = np.array([mu_vals[k] / d for k in adm_idx])

    def fp_M1(u, v, gi):
        return elem_mats[v].astype(float).flatten() / q

    def fp_M2(u, v, gi):
        Mu = elem_mats[u].astype(float) / q
        Ms = gen_mats[gi].astype(float) / q
        return np.outer(Mu.flatten(), Ms.flatten()).flatten()

    def fp_M3(u, v, gi):
        chi_u = Pi_mat[u, :]
        Ms = gen_mats[gi].astype(float) / q
        return np.outer(kappa * chi_u, Ms.flatten()).flatten()

    def fp_M4(u, v, gi):
        Mu = elem_mats[u].astype(float) / q
        Ms = gen_mats[gi].astype(float) / q
        return np.concatenate([(Mu @ Ms).flatten(),
                                (Mu.T @ Ms).flatten(),
                                (Mu @ Ms.T).flatten()])

    variants = [
        ('M1: $\\mathrm{vec}(M_v)$', fp_M1, 4, 'gray'),
        ('M2: $M_u\\otimes M_s$', fp_M2, 16, 'blue'),
        ('M3: $\\chi_u\\otimes\\mathrm{vec}(M_s)$', fp_M3, n_adm*4, 'orange'),
        ('M4: 3 products', fp_M4, 12, 'green'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Matrix-based Version B variants on $X^{{{p},{q}}}$',
        fontsize=12
    )

    for name, fp, dim, col in variants:
        print(f"  Running {name} (dim={dim})...", end=' ', flush=True)
        res = bfs_cascade(adj, fp, dim, n, max_steps=200)
        print(f"max_dim={max(res['dim_Pi'])}, late_rt={np.mean(res['rt'][-5:]):.3f}")
        Sn = res['Sn']
        axes[0].plot(Sn, res['dim_Pi'], marker='o', ms=2, color=col, label=name)
        axes[0].axhline(dim, color=col, linestyle=':', alpha=0.3)
        axes[1].plot(Sn, res['rt'], marker='o', ms=2, color=col, label=name)
        valid = np.array(res['rt']) > 0.005
        Sn_arr = np.array(Sn)
        rt_arr = np.array(res['rt'])
        if valid.sum() > 3:
            axes[2].semilogy(Sn_arr[valid], rt_arr[valid],
                             marker='o', ms=2, color=col, label=name)

    axes[0].set_xlabel(r'$|S_n|$')
    axes[0].set_ylabel(r'$\dim\Pi^{\rm mat}(S_n)$')
    axes[0].set_title('Rank of matrix span')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'$|S_n|$')
    axes[1].set_ylabel(r'$\tilde{r}_n^{\rm mat}$')
    axes[1].set_title('Admissible frontier fraction')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].set_xlabel(r'$|S_n|$')
    axes[2].set_ylabel(r'$\tilde{r}_n$ (log)')
    axes[2].set_title('Decay shape (semi-log)')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Saved {outfile}")


# ================================================================
# §5  FIGURE 4 -- Steinberg St_elem pre-saturation law
# ================================================================

def figure4_steinberg_presat(qs=(13, 17, 29), p=5, outfile='fig4_StElem_presat.png'):
    print(f"Figure 4: Steinberg St_elem pre-saturation, q={qs}")
    colors = {13: 'green', 17: 'blue', 29: 'orange'}

    all_data = {}
    for q in qs:
        print(f"  q={q}...", end=' ', flush=True)
        adj, elems, elem_mats, gen_mats = build_graph(q, p)
        n = len(elems)
        steins, gen_steins = build_steinberg_projections(elems, elem_mats, gen_mats, q)

        def fp_St(u, v, gi, st=steins, gst=gen_steins):
            return st[u] * gst[gi]

        # Dense BFS with cumulative front tracking
        S = set([0])
        frontier = {0: (None, None)}
        added = set([0])
        bfs_q_local = deque([0])
        span = np.zeros((0, q + 1))
        p_prod = 0

        Sn_l, rt_l, dim_l, pprod_l = [], [], [], []

        def rk(sp):
            if len(sp) == 0:
                return 0, None
            sv = np.linalg.svd(sp, compute_uv=False)
            r = int(np.sum(sv > 1e-8))
            if r == 0:
                return 0, None
            _, _, Vt = np.linalg.svd(sp, full_matrices=False)
            return r, Vt[:r, :]

        for step in range(min(400, n // 3)):
            if not frontier:
                break
            rank, Vt_r = rk(span)
            adm_c = 0
            for v, (u, gi) in frontier.items():
                if u is None:
                    continue
                ev = fp_St(u, v, gi)
                if rank == 0 or np.all(np.abs(ev) < 1e-10):
                    adm_c += 1
                else:
                    proj = Vt_r.T @ (Vt_r @ ev)
                    if np.linalg.norm(ev - proj) > 1e-8:
                        adm_c += 1
            dS = len(frontier)
            p_prod += adm_c
            Sn_l.append(len(S))
            rt_l.append(adm_c / dS if dS > 0 else 0)
            dim_l.append(rank)
            pprod_l.append(p_prod)

            v_new = None
            while bfs_q_local:
                cand = bfs_q_local.popleft()
                if cand in frontier:
                    v_new = cand
                    break
            if v_new is None:
                v_new = next(iter(frontier))
            u_new, gi_new = frontier[v_new]
            if u_new is not None:
                new_fp = fp_St(u_new, v_new, gi_new).reshape(1, -1)
                span = np.vstack([span, new_fp]) if len(span) > 0 else new_fp
            S.add(v_new)
            del frontier[v_new]
            for (nb, gi) in adj.get(v_new, []):
                if nb < n and nb not in added:
                    frontier[nb] = (v_new, gi)
                    bfs_q_local.append(nb)
                    added.add(nb)

        Sn_arr = np.array(Sn_l)
        rt_arr = np.array(rt_l)
        dim_arr = np.array(dim_l)
        pp_arr = np.array(pprod_l)

        sat_idx = next((i for i, d_ in enumerate(dim_arr) if d_ >= q + 1), None)
        sat_Sn = int(Sn_arr[sat_idx]) if sat_idx is not None else None

        # Fit p_prod ~ |S|^beta
        hi = sat_idx if sat_idx is not None else len(Sn_arr)
        mask = (Sn_arr[:hi] > 3) & (pp_arr[:hi] > 0)
        beta_prod = None
        if mask.sum() >= 5:
            logx = np.log(Sn_arr[:hi][mask].astype(float))
            logy = np.log(pp_arr[:hi][mask].astype(float))
            slope, _ = np.polyfit(logx, logy, 1)
            beta_prod = slope

        all_data[q] = {
            'n': n, 'Sn': Sn_arr, 'rt': rt_arr, 'dim': dim_arr,
            'pp': pp_arr, 'sat_Sn': sat_Sn, 'sat_idx': sat_idx,
            'beta_prod': beta_prod
        }
        print(f"|S*|={sat_Sn}, beta_prod={beta_prod:.3f}" if beta_prod else f"|S*|={sat_Sn}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r'Steinberg $\tilde\sigma_u\odot\tilde\sigma_s$ pre-saturation law'
        f' -- $X^{{5,q}}$, $q\\in{{{",".join(map(str,qs))}}}$',
        fontsize=12
    )

    # Panel 1: r_tilde vs |S|
    ax = axes[0, 0]
    for q, d in all_data.items():
        ax.plot(d['Sn'], d['rt'], color=colors[q], marker='o', ms=2,
                label=f'$q={q}$')
        if d['sat_Sn']:
            ax.axvline(d['sat_Sn'], color=colors[q], linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$|S_n|$')
    ax.set_ylabel(r'$\tilde{r}_n^{\rm St}$')
    ax.set_title(r'Admissible frontier fraction (linear)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: semi-log r_tilde
    ax = axes[0, 1]
    for q, d in all_data.items():
        valid = d['rt'] > 0.005
        ax.semilogy(d['Sn'][valid], d['rt'][valid], color=colors[q],
                    marker='o', ms=2, label=f'$q={q}$')
    ax.set_xlabel(r'$|S_n|$')
    ax.set_ylabel(r'$\tilde{r}_n^{\rm St}$ (log)')
    ax.set_title('Decay shape (semi-log)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: p_prod log-log
    ax = axes[1, 0]
    for q, d in all_data.items():
        valid = d['pp'] > 0
        ax.loglog(d['Sn'][valid], d['pp'][valid], color=colors[q],
                  marker='o', ms=2, label=f'$q={q}$')
        sat = d['sat_idx']
        if d['beta_prod'] and sat:
            x_fit = d['Sn'][3:sat].astype(float)
            mask_f = (x_fit > 3) & (d['pp'][3:sat] > 0)
            if mask_f.sum() > 3:
                logx = np.log(x_fit[mask_f])
                logy = np.log(d['pp'][3:sat][mask_f].astype(float))
                A = np.exp(np.polyfit(logx, logy, 1)[1])
                ax.loglog(x_fit[mask_f],
                          A * x_fit[mask_f]**d['beta_prod'], color=colors[q],
                          linestyle='--', alpha=0.6,
                          label=f"$\\beta_{{\\rm prod}}={d['beta_prod']:.2f}$")
    ax.set_xlabel(r'$|S_n|$ (log)')
    ax.set_ylabel(r'$p_n^{\rm prod}$ (log)')
    ax.set_title(r'Cumulative admissible front (log-log)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: stability table
    ax = axes[1, 1]
    ax.axis('off')
    lines = ['PARAMETER STABILITY\n']
    lines.append('Saturation thresholds vs q:')
    sat_list = [(q, d['sat_Sn']) for q, d in all_data.items() if d['sat_Sn']]
    for q_v, s_v in sat_list:
        n_v = all_data[q_v]['n']
        lines.append(f'  $q={q_v}$: $|S^*|={s_v}$, '
                     f'$|S^*|/|G|={s_v/n_v:.5f}$')
    if len(sat_list) >= 2:
        q_arr = np.array([x[0] for x in sat_list], dtype=float)
        s_arr = np.array([x[1] for x in sat_list], dtype=float)
        slope, _ = np.polyfit(np.log(q_arr), np.log(s_arr), 1)
        lines.append(f'  $|S^*| \\sim q^{{{slope:.2f}}}$')
    lines.append('')
    lines.append('$\\beta_{\\rm prod}$ stability:')
    for q, d in all_data.items():
        if d['beta_prod']:
            lines.append(f'  $q={q}$: $\\beta_{{\\rm prod}}={d["beta_prod"]:.4f}$')
    bp_vals = [d['beta_prod'] for d in all_data.values() if d['beta_prod']]
    if bp_vals:
        lines.append('  mean: ' + f'{np.mean(bp_vals):.4f}' + ', '
                     + 'std: ' + f'{np.std(bp_vals):.4f}')
    lines.append('')
    lines.append(r'Note: $\beta_{\rm prod}\approx1.7$ is outside')
    lines.append(r'$\beta^*\in(0.09,0.13)$; governed by')
    lines.append(r'$O(q)$ pre-saturation window only.')

    ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Summary')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"  Saved {outfile}")


# ================================================================
# §6  MAIN
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SpectralO5 computations')
    parser.add_argument('--fig', type=int, default=0,
                        help='Generate only figure N (1-4). 0 = all.')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for figures.')
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    def path(name):
        return os.path.join(args.outdir, name)

    figs = {
        1: lambda: figure1_A_vs_B(q=13, p=5,
                                   outfile=path('fig1_A_vs_B.pdf')),
        2: lambda: figure2_versionB_qdep(qs=(13, 29), p=5,
                                          outfile=path('fig2_versionB_sat.pdf')),
        3: lambda: figure3_matrix_variants(q=13, p=5,
                                            outfile=path('fig3_matB_variants.pdf')),
        4: lambda: figure4_steinberg_presat(qs=(13, 17, 29), p=5,
                                             outfile=path('fig4_StElem_presat.pdf')),
    }

    if args.fig == 0:
        for i in sorted(figs):
            figs[i]()
    elif args.fig in figs:
        figs[args.fig]()
    else:
        print(f"Unknown figure {args.fig}. Choose 1-4.")


if __name__ == '__main__':
    main()
