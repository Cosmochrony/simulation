"""
spectral_stratigraphy_full.py
==============================
Complete numerical implementation of the spectral stratigraphy programme.
Reproduces all figures from the SpectralStratigraphy technical note.

Sections
--------
1.  Group construction     -- 2T (order 24), 2O (48), 2I (120)
2.  Cayley graph spectrum  -- eigenvalues, multiplicities, representation content
3.  Three conditions       -- n_proj, n_sat, admissibility
4.  No-go demonstration    -- scalar power-law gives featureless profile
5.  Q8 two-level result    -- proof of concept
6.  Scan of generator sets -- systematic search for three-level cases
7.  Three-level figures    -- 2T, 2O, 2I (ord-4), 2I (ord-5)
8.  Summary table

Dependencies: numpy, matplotlib, itertools (stdlib)

Usage
-----
    python spectral_stratigraphy_full.py

All figures are saved to the current directory as PDF files.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Quaternion arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(q):
    """Return unit quaternion."""
    n = np.sqrt(sum(x**2 for x in q))
    return tuple(x / n for x in q)

def qmult(a, b):
    """Hamilton product of two unit quaternions, returned normalised."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return normalize((
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ))

def qinv(q):
    """Inverse (= conjugate) of a unit quaternion."""
    return normalize((q[0], -q[1], -q[2], -q[3]))

def find_idx(elems, q, tol=1e-7):
    """Return index of q in elems list, or None."""
    for i, e in enumerate(elems):
        if all(abs(e[k] - q[k]) < tol for k in range(4)):
            return i
    return None

def deduplicate(raw):
    """Remove duplicate quaternions (no identification of q with -q)."""
    elems = []
    for q in raw:
        if not any(all(abs(e[i] - q[i]) < 1e-8 for i in range(4)) for e in elems):
            elems.append(q)
    return elems


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Group construction
# ═══════════════════════════════════════════════════════════════════════════════

def _base_24():
    """24 unit quaternions forming 2T and the ±1,±i,±j,±k,halves core."""
    raw = []
    for s in [1, -1]:
        raw += [normalize((s,0,0,0)), normalize((0,s,0,0)),
                normalize((0,0,s,0)), normalize((0,0,0,s))]
    for a, b, c, d in [(a,b,c,d) for a in [1,-1] for b in [1,-1]
                       for c in [1,-1] for d in [1,-1]]:
        raw.append(normalize((a/2, b/2, c/2, d/2)))
    return raw


def build_2T():
    """Binary tetrahedral group 2T, order 24."""
    return deduplicate(_base_24())


def build_2O():
    """Binary octahedral group 2O, order 48."""
    raw = _base_24()
    for a, b in [(a,b) for a in [1,-1] for b in [1,-1]]:
        sq = 1 / np.sqrt(2)
        raw += [
            normalize((a*sq, b*sq, 0,    0   )),
            normalize((a*sq, 0,    b*sq, 0   )),
            normalize((a*sq, 0,    0,    b*sq)),
            normalize((0,    a*sq, b*sq, 0   )),
            normalize((0,    a*sq, 0,    b*sq)),
            normalize((0,    0,    a*sq, b*sq)),
        ]
    return deduplicate(raw)


def _sign_perm(p):
    """Sign (+1 or -1) of permutation p."""
    n = len(p)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = p[j]
                cycle_len += 1
            if cycle_len % 2 == 0:
                sign *= -1
    return sign


def build_2I():
    """Binary icosahedral group 2I = SL(2,5), order 120."""
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    raw = _base_24()
    base = [0, inv_phi, 1, phi]
    for s0, s1, s2, s3 in [(s0,s1,s2,s3)
                            for s0 in [1,-1] for s1 in [1,-1]
                            for s2 in [1,-1] for s3 in [1,-1]]:
        v = [s0*base[0], s1*base[1], s2*base[2], s3*base[3]]
        for perm in permutations(range(4)):
            if _sign_perm(perm) == 1:
                raw.append(normalize(tuple(v[perm[i]] / 2 for i in range(4))))
    return deduplicate(raw)


def build_Q8():
    """Quaternion group Q8, order 8."""
    Q8 = {
        0: ( 1, 0, 0, 0),
        1: (-1, 0, 0, 0),
        2: ( 0, 1, 0, 0),
        3: ( 0,-1, 0, 0),
        4: ( 0, 0, 1, 0),
        5: ( 0, 0,-1, 0),
        6: ( 0, 0, 0, 1),
        7: ( 0, 0, 0,-1),
    }
    return [normalize(Q8[i]) for i in range(8)]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Cayley graph spectrum
# ═══════════════════════════════════════════════════════════════════════════════

def cayley_laplacian(elems, gen_indices):
    """Build the Laplacian of Cay(G, S) where S = gen_indices."""
    n = len(elems)
    A = np.zeros((n, n))
    for g in gen_indices:
        for ei in range(n):
            prod = qmult(elems[ei], elems[g])
            t = find_idx(elems, prod)
            if t is not None:
                A[ei, t] = 1.0
    return np.diag(A.sum(axis=1)) - A


def spectrum(L, tol=1e-5):
    """
    Return (eigenvalues_sorted, distinct_list) where distinct_list is a
    list of [value, multiplicity] pairs.
    """
    eigs = np.sort(np.linalg.eigvalsh(L))
    dist = []
    for e in eigs:
        if not dist or abs(e - dist[-1][0]) > tol:
            dist.append([round(e, 5), 1])
        else:
            dist[-1][1] += 1
    return eigs, dist


def elem_orders(elems, identity_idx, max_ord=40):
    """Return list of element orders."""
    orders = []
    for idx in range(len(elems)):
        curr = elems[idx]
        for k in range(1, max_ord + 1):
            curr = qmult(curr, elems[idx])
            if find_idx(elems, curr) == identity_idx:
                orders.append(k + 1)
                break
        else:
            orders.append(-1)
    return orders


def symmetric_gens(elems, ord_list, target_order):
    """Return symmetric generator set of elements with given order."""
    S = set(i for i, o in enumerate(ord_list) if o == target_order)
    for g in list(S):
        inv = find_idx(elems, qinv(elems[g]))
        if inv is not None:
            S.add(inv)
    return list(S)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Three conditions: n_proj, n_sat, n_min
# ═══════════════════════════════════════════════════════════════════════════════

def n_proj(lam, C, alpha):
    """Earliest cascade step at which mode lambda becomes projectable."""
    return np.ceil((lam / C) ** (1.0 / alpha))


def n_sat(lam, E0, beta, E_P=1.0):
    """Latest cascade step at which enough energy remains for stabilisation."""
    return E_P / (E0 * lam ** beta)


def two_anchor_calibration(lam_h, lam_l, n_h, n_l, alpha, beta, E_P=1.0):
    """
    Calibrate (C, E0) from two empirical anchors:
      n_proj(lam_h) = n_h   (heaviest sector: stabilises earliest)
      n_sat(lam_l)  = n_l   (lightest sector: stabilises latest)
    """
    C  = lam_h / (n_h ** alpha)
    E0 = E_P / (n_l * (lam_l ** beta))
    return C, E0


def stratigraphic_levels(nz_dist, alpha=1.0, beta=0.5, n_h=1e17, n_l=1e22, E_P=1.0):
    """
    Given a list of (lambda, multiplicity) pairs (non-zero eigenvalues),
    return list of (lambda, mult, log10_n_min) for modes with non-empty window.
    """
    if len(nz_dist) < 2:
        return []
    lam_h, lam_l = nz_dist[0][0], nz_dist[-1][0]
    C, E0 = two_anchor_calibration(lam_h, lam_l, n_h, n_l, alpha, beta, E_P)
    levels = []
    for lv, mult in nz_dist:
        np_ = n_proj(lv, C, alpha)
        ns_ = n_sat(lv, E0, beta, E_P)
        if np_ <= ns_:
            levels.append((lv, mult, np.log10(np_)))
    return levels


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  No-go result: scalar power law
# ═══════════════════════════════════════════════════════════════════════════════

def nogo_figure(save=True):
    """
    Figure demonstrating the no-go result: scalar E*(lambda)=E0*lambda^beta
    with two-anchor calibration gives a featureless linear log-log profile.
    """
    lam_min, lam_max = 1e-3, 1e2
    N = 40
    lam = np.logspace(np.log10(lam_min), np.log10(lam_max), N)
    lam_top, lam_e = lam[0], lam[-1]

    alpha, beta, E_P = 1.0, 0.5, 1.0
    n_h, n_l = 1e17, 1e22
    C, E0 = two_anchor_calibration(lam_top, lam_e, n_h, n_l, alpha, beta, E_P)

    nproj = np.array([n_proj(lv, C, alpha) for lv in lam])
    nsat  = np.array([n_sat(lv, E0, beta, E_P) for lv in lam])
    mask  = nproj <= nsat
    k_all = np.arange(1, N + 1)
    lnm   = np.log10(nproj)

    slope = (lnm[-1] - lnm[0]) / (np.log10(lam[-1]) - np.log10(lam[0]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    ax = axes[0]
    ax.semilogy(k_all, nproj, "o--", ms=3, color="steelblue",
                label="$n_{\\mathrm{proj}}$", lw=0.9)
    ax.semilogy(k_all, nsat,  "s--", ms=3, color="darkorange",
                label="$n_{\\mathrm{sat}}$", lw=0.9)
    ax.semilogy(k_all[mask], nproj[mask], "^", ms=6, color="firebrick",
                label="stabilisable", zorder=5)
    ax.axhline(n_h, color="steelblue",  linestyle=":", alpha=0.5,
               label=f"top anchor $10^{{17}}$")
    ax.axhline(n_l, color="darkorange", linestyle=":", alpha=0.5,
               label=f"electron anchor $10^{{22}}$")
    ax.set_xlabel("mode index $k$")
    ax.set_ylabel("cascade depth $n$")
    ax.set_title("(a) Projectability / Saturability")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.plot(k_all[mask], lnm[mask], "^-", ms=5, color="firebrick", lw=1.3)
    ax2.axhline(17, color="gray", ls=":", lw=0.8, label="top $n\\approx 10^{17}$")
    ax2.axhline(22, color="gray", ls="--", lw=0.8, label="electron $n\\approx 10^{22}$")
    ax2.set_xlabel("stabilisable mode index")
    ax2.set_ylabel("$\\log_{10}\\,n_{\\min}(\\lambda_k)$")
    ax2.set_title("(b) Stratigraphic profile\n(main diagnostic)")
    ax2.legend(fontsize=7)

    ax3 = axes[2]
    ax3.plot(np.log10(lam[mask]), lnm[mask], "^-", ms=5,
             color="firebrick", lw=1.3)
    ax3.set_xlabel("$\\log_{10}\\,\\lambda_k$")
    ax3.set_ylabel("$\\log_{10}\\,n_{\\min}(\\lambda_k)$")
    ax3.set_title("(c) $\\log n_{\\min}$ vs $\\log\\lambda$\n"
                  "(slope $= 1/\\alpha$, no plateaus)")
    ax3.text(0.05, 0.9,
             f"slope $\\approx {slope:.3f}$ (theory: $1/\\alpha = {1/alpha:.1f}$)",
             transform=ax3.transAxes, fontsize=8)

    fig.suptitle(
        "No-go result: scalar $E^*(\\lambda)=E_0\\lambda^\\beta$ "
        "with two-anchor calibration\\n"
        "gives a featureless power-law stratigraphic profile "
        f"(slope $= 1/\\alpha$)",
        fontsize=9, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig("fig_nogo.pdf", bbox_inches="tight")
        print("Saved: fig_nogo.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Q8 two-level result
# ═══════════════════════════════════════════════════════════════════════════════

def q8_figure(save=True):
    """
    Compute Q8 Cayley spectrum and two-level stratigraphic profile.
    Compares with the no-go scalar profile.
    """
    elems_Q8 = build_Q8()
    # Generators: ±i, ±j, ±k  (indices 2..7)
    gens_Q8 = [2, 3, 4, 5, 6, 7]
    L_Q8 = cayley_laplacian(elems_Q8, gens_Q8)
    _, dist_Q8 = spectrum(L_Q8)

    print("Q8 Cayley spectrum (S={±i,±j,±k}):")
    for val, mult in dist_Q8:
        tag = "zero mode" if val < 1e-4 else ""
        print(f"  lambda={val:.4f}  mult={mult:3d}  {tag}")

    nz_Q8 = [(d[0], d[1]) for d in dist_Q8 if d[0] > 1e-4]
    levels_Q8 = stratigraphic_levels(nz_Q8)

    print("\nQ8 stratigraphic levels:")
    for lv, mult, lnm in levels_Q8:
        print(f"  lambda={lv:.1f}  mult={mult}  log10(n_min)={lnm:.3f}")
    if len(levels_Q8) >= 2:
        sep = levels_Q8[1][2] - levels_Q8[0][2]
        print(f"  Separation: {sep:.3f} cascade decades")

    # Figure
    alpha, beta, E_P = 1.0, 0.5, 1.0
    n_h, n_l = 1e17, 1e22
    colors_Q8 = {6.: "steelblue", 8.: "darkorange"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (a) spectrum
    ax = axes[0]
    lam_vals_Q8 = [6., 6., 6., 6., 8., 8., 8.]
    for k, lv in enumerate(lam_vals_Q8):
        ax.scatter(k + 1, lv, c=colors_Q8[lv], s=80, zorder=5)
    ax.axhline(6., color="steelblue",  ls=":", alpha=0.5)
    ax.axhline(8., color="darkorange", ls=":", alpha=0.5)
    ax.text(7.4, 6.15, "$\\lambda=6$ spinorial $\\rho_5$",
            fontsize=8, color="steelblue", ha="right")
    ax.text(7.4, 8.15, "$\\lambda=8$ three 1D reps",
            fontsize=8, color="darkorange", ha="right")
    ax.set_xlim(0, 8.5); ax.set_ylim(0, 10)
    ax.set_xlabel("mode index $k$")
    ax.set_ylabel("$\\lambda_k$")
    ax.set_title("(a) $Q_8$ spectrum\n$S=\\{\\pm i,\\pm j,\\pm k\\}$")

    # (b) stratigraphic profile
    ax2 = axes[1]
    lam_spec = np.array([d[0] for d in dist_Q8 if d[0] > 1e-4
                         for _ in range(d[1])])
    C_Q8, E0_Q8 = two_anchor_calibration(nz_Q8[0][0], nz_Q8[-1][0],
                                         n_h, n_l, alpha, beta, E_P)
    nproj_Q8 = np.array([n_proj(lv, C_Q8, alpha) for lv in lam_spec])
    lnm_Q8   = np.log10(nproj_Q8)
    for k_idx, (lv, lnm_val) in enumerate(zip(lam_spec, lnm_Q8)):
        ax2.scatter(k_idx + 1, lnm_val,
                    c=colors_Q8.get(round(lv), "gray"), s=80, zorder=5)
    for lv, mult, lnm_val in levels_Q8:
        ax2.axhline(lnm_val,
                    color=colors_Q8.get(round(lv), "gray"),
                    ls="--", alpha=0.5, lw=1.2,
                    label=f"$\\lambda={lv:.0f}$, mult {mult}")
    ax2.set_xlabel("mode index $k$")
    ax2.set_ylabel("$\\log_{10}\\,n_{\\min}(\\lambda_k)$")
    ax2.set_title("(b) $Q_8$ stratigraphic profile\n2 discrete levels")
    ax2.legend(fontsize=7)
    ax2.set_ylim(16.5, 18.0)

    # (c) comparison with no-go
    ax3 = axes[2]
    lam_cont = np.linspace(6., 8., 50)
    lnm_cont = np.log10((lam_cont / C_Q8) ** (1 / alpha))
    ax3.plot(lam_cont, lnm_cont, "-", color="gray", lw=1.5, alpha=0.7,
             label="scalar power law (no-go)")
    for lv, mult, lnm_val in levels_Q8:
        ax3.scatter([lv] * mult, [lnm_val] * mult,
                    c=colors_Q8.get(round(lv), "gray"), s=80, zorder=5,
                    label=f"$Q_8$: $\\lambda={lv:.0f}$ (×{mult})")
    ax3.set_xlabel("$\\lambda_k$")
    ax3.set_ylabel("$\\log_{10}\\,n_{\\min}(\\lambda_k)$")
    ax3.set_title("(c) Discrete vs continuous\nstratigraphy")
    ax3.legend(fontsize=7)

    fig.suptitle(
        "$Q_8$ Cayley graph: two spectral classes from representation theory "
        "give two discrete stratigraphic levels",
        fontsize=9, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig("fig_Q8.pdf", bbox_inches="tight")
        print("Saved: fig_Q8.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Systematic scan of generator sets
# ═══════════════════════════════════════════════════════════════════════════════

def scan_groups(verbose=True):
    """
    For each group (2T, 2O, 2I), scan all symmetric generator sets by order
    and report the number of non-zero spectral levels.
    """
    results = {}

    for name, elems in [("2T", build_2T()), ("2O", build_2O()),
                        ("2I", build_2I()), ("Q8", build_Q8())]:
        id_idx = find_idx(elems, normalize((1., 0., 0., 0.)))
        ords   = elem_orders(elems, id_idx)
        order_set = sorted(set(ords) - {1})
        results[name] = {}

        if verbose:
            print(f"\n{name} (|G|={len(elems)}), "
                  f"order distribution: {dict(sorted(Counter(ords).items()))}")

        for o in order_set:
            S = symmetric_gens(elems, ords, o)
            if not S:
                continue
            L   = cayley_laplacian(elems, S)
            _, dist = spectrum(L)
            n_levels = sum(1 for d in dist if d[0] > 1e-4)
            results[name][o] = (len(S), n_levels, dist)
            if verbose:
                lam_str = ", ".join(
                    f"{d[0]:.2f}(×{d[1]})"
                    for d in dist if d[0] > 1e-4)
                print(f"  S=ord-{o:2d} (|S|={len(S):3d}): "
                      f"{n_levels} non-zero levels  [{lam_str}]")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Three-level figure
# ═══════════════════════════════════════════════════════════════════════════════

# Hard-coded spectra for the four three-level cases (verified numerically)
THREE_LEVEL_CASES = [
    {
        "label":   "$2T$, $S$=ord-3\n$|G|=24$, $|S|=8$",
        "nz":      [(6., 8), (8., 9), (12., 6)],
        "rep":     ["2D+2D", "3D", "1D+1D+2D"],
    },
    {
        "label":   "$2O$, $S$=ord-3\n$|G|=48$, $|S|=8$",
        "nz":      [(6., 16), (8., 18), (12., 12)],
        "rep":     ["4D", "3D+3D", "2D+2D+2D"],
    },
    {
        "label":   "$2I$, $S$=ord-4\n$|G|=120$, $|S|=30$",
        "nz":      [(24., 25), (30., 76), (40., 18)],
        "rep":     ["5D", "6D+4D+4D+2D+2D+2D", "3D+3D"],
    },
    {
        "label":   "$2I$, $S$=ord-5\n$|G|=120$, $|S|=24$",
        "nz":      [(20., 54), (24., 25), (30., 40)],
        "rep":     ["6D+3D+3D", "5D", "4D+4D+2D+2D"],
    },
]

LEVEL_COLORS = ["steelblue", "darkorange", "firebrick"]


def three_level_figure(save=True, alpha=1.0, beta=0.5, n_h=1e17, n_l=1e22):
    """Four-panel figure showing the three-level stratigraphic cases."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

    for ax, case in zip(axes, THREE_LEVEL_CASES):
        nz     = case["nz"]
        levels = stratigraphic_levels(nz, alpha, beta, n_h, n_l)

        k = 1
        for j, (lv, mult, lnm) in enumerate(levels):
            col = LEVEL_COLORS[j]
            for _ in range(mult):
                ax.scatter(k, lnm, c=col, s=45, zorder=5)
                k += 1
            ax.axhline(lnm, color=col, ls="--", alpha=0.5, lw=0.9,
                       label=f"$\\lambda={lv:.0f}$ (×{mult})")
            # Representation content annotation
            ax.text(mult * 0.6 + 2, lnm + 0.004,
                    case["rep"][j], fontsize=5.5,
                    color=col, va="bottom")

        seps = [f"{levels[i+1][2] - levels[i][2]:.3f}"
                for i in range(len(levels) - 1)]
        ax.set_xlabel("mode index $k$", fontsize=8)
        ax.set_ylabel("$\\log_{10}\\,n_{\\min}$", fontsize=8)
        ax.set_title(f"{case['label']}\n{len(levels)} levels, "
                     f"seps: {seps} dec", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Three-level stratigraphic structures: $2T$, $2O$, $2I$\n"
        "Each level = a block of irreducible representations — "
        "representational mechanism confirmed",
        fontsize=9, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig("fig_3levels.pdf", bbox_inches="tight")
        print("Saved: fig_3levels.pdf")
    return fig


def comparison_2I_figure(save=True, alpha=1.0, beta=0.5, n_h=1e17, n_l=1e22):
    """
    Figure comparing Q8 (2 levels), 2I with S=order-6 (5 levels),
    and 2I with S=order-4 (3 levels).
    """
    cases_comp = [
        ("$Q_8$, $S$=ord-6\n2 non-zero levels",
         [(6., 4), (8., 3)]),
        ("$2I$, $S$=ord-6\n5 non-zero levels",
         [(10., 8), (15., 16), (20., 54), (24., 25), (25., 16)]),
        ("$2I$, $S$=ord-4\n3 non-zero levels",
         [(24., 25), (30., 76), (40., 18)]),
        ("$2I$, $S$=ord-5\n3 non-zero levels",
         [(20., 54), (24., 25), (30., 40)]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    palette = ["steelblue", "cornflowerblue", "dodgerblue",
               "darkorange", "sandybrown"]

    for ax, (label, nz) in zip(axes, cases_comp):
        levels = stratigraphic_levels(nz, alpha, beta, n_h, n_l)
        k = 1
        for j, (lv, mult, lnm) in enumerate(levels):
            col = palette[j % len(palette)]
            for _ in range(mult):
                ax.scatter(k, lnm, c=col, s=40, zorder=5)
                k += 1
            ax.axhline(lnm, color=col, ls="--", alpha=0.5, lw=0.9,
                       label=f"$\\lambda={lv:.0f}$")
        ax.set_xlabel("mode index $k$", fontsize=8)
        ax.set_ylabel("$\\log_{10}\\,n_{\\min}$", fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "$Q_8$ and $2I$: varying generator sets changes the number of "
        "stratigraphic levels",
        fontsize=9, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig("fig_2I_comparison.pdf", bbox_inches="tight")
        print("Saved: fig_2I_comparison.pdf")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(alpha=1.0, beta=0.5, n_h=1e17, n_l=1e22):
    print()
    print("SUMMARY TABLE: three-level cases")
    print("="*90)
    hdr = (f"{'Case':35s}  {'lam_h':6s}  {'lam_l':6s}  "
           f"{'ratio':6s}  {'sep1 (dec)':10s}  {'sep2 (dec)':10s}  "
           f"{'mults':20s}")
    print(hdr)
    print("-"*90)
    for case in THREE_LEVEL_CASES:
        nz     = case["nz"]
        levels = stratigraphic_levels(nz, alpha, beta, n_h, n_l)
        lam_h, lam_l = nz[0][0], nz[-1][0]
        seps = [levels[i+1][2] - levels[i][2]
                for i in range(len(levels) - 1)]
        mults = [m for _, m, _ in levels]
        label = case["label"].replace("\n", " ").replace("$", "")
        s1 = f"{seps[0]:.4f}" if len(seps) > 0 else "—"
        s2 = f"{seps[1]:.4f}" if len(seps) > 1 else "—"
        print(f"{label:35s}  {lam_h:6.1f}  {lam_l:6.1f}  "
              f"{lam_l/lam_h:6.3f}  {s1:10s}  {s2:10s}  {mults}")
    print()
    print("Notes:")
    print("  Separations are in units of log10(n_min) cascade decades.")
    print("  Two-anchor calibration: n_proj(lam_h)=10^17, n_sat(lam_l)=10^22.")
    print("  All three-level cases arise from representational mechanism (A).")
    print("  Scalar power-law ansatz produces NO plateaus (Proposition 2).")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Spectral Stratigraphy — Full Computation")
    print("="*50)

    # Section 6: scan
    print("\n── Section 6: Systematic generator-set scan ──")
    scan_groups(verbose=True)

    # Section 4: no-go
    print("\n── Section 4: No-go figure ──")
    nogo_figure(save=True)

    # Section 5: Q8
    print("\n── Section 5: Q8 two-level figure ──")
    q8_figure(save=True)

    # Section 7: three-level cases
    print("\n── Section 7: Three-level figure ──")
    three_level_figure(save=True)
    comparison_2I_figure(save=True)

    # Section 8: summary
    print_summary_table()

    print("\nAll done. Figures saved as PDF in current directory.")
