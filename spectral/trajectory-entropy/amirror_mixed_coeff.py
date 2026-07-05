"""Exact certification of the mixed two-anchor coefficient (a-mirror front).

Companion to the a-mirror separation front (recovery note
RECOVERY-NOTE-front-amirror-separation.md, step 2-3 of the fixed plan).

Setup.  Quasi-real probe in Fourier coordinates (u-lattice, frequencies c*u):
    f = sum_j e_{A_j} + i*eps*(x1 e_{A1+1} + x2 e_{A2+1}),
with m >= 2k real unit anchors A_1..A_m pairwise separated by more than the
b-window, and two satellites at lattice step Delta=1 on anchors A_1, A_2.

Claim (derived analytically, certified here at machine precision):
    Ptilde := F(a,b)F(a,-b) - F(-a,b)F(-a,-b)
            = F0(b)F0(-b) * 4 eps^2 Im(x1 conj(x2)) * S(a,b) + O(eps^4),
where F(a,b) = det Gram_c(D_{k-1} u {(a,b)}), F0 = its eps=0 value, and

    S(a,b) = [Im Q_{b-1} + Im Q_b]_{sys(a,b)}
           + [Im Q_{-b-1} + Im Q_{-b}]_{sys(a,-b)},
    Q_beta = K_beta(tau_1, tau_2) * K_{beta+1}(tau_2, tau_1),
    K_beta(s, t) = w_beta(s)^dag A_beta^{-1} w_beta(t),
    tau_j = c*(A_j + 1 + beta),
    A_beta = W_beta W_beta^dag,   W_beta[alpha, j] = zeta^{c(A_j+beta) alpha},
    w_beta(t)[alpha] = zeta^{t alpha},   alpha in Lambda_beta of the system.

The eps -> -eps symmetrisation kills all odd orders, so the measured ratio
converges to S at rate O(eps^2) until the floating-point noise floor of the
determinant ratios takes over (around eps ~ 1e-4).  The b-mirror identity
Delta_{q-c}(a,b) = Delta_c(a,-b) is also re-verified numerically.

Exit code 0 iff all geometries certify (relative error < 1e-4 at eps=1e-3,
the sweet spot between O(eps^2) truncation and the noise floor).

Usage:
    python amirror_mixed_coeff.py
"""

import sys

import numpy as np

HERE = "/".join(__file__.split("/")[:-1]) or "."
sys.path.insert(0, HERE)

from v3_exact_check import weil


def make_probe(q, c, anchors, x1, x2, eps):
    """Position-space probe for the quasi-real two-satellite family."""
    hat = np.zeros(q, complex)
    for A in anchors:
        hat[(c * A) % q] = 1.0
    hat[(c * (anchors[0] + 1)) % q] += 1j * eps * x1
    hat[(c * (anchors[1] + 1)) % q] += 1j * eps * x2
    # inverse DFT consistent with hat(nu) = q^{-1/2} sum_x phi(x) zeta^{-nu x}
    x = np.arange(q)
    F = np.exp(2j * np.pi * np.outer(x, x) / q) / np.sqrt(q)
    return F @ hat


def system_points(k, a, b):
    """D_{k-1} union {(a,b)} as a list of lattice points."""
    pts = [(al, be) for be in range(-(k - 1), k) for al in range(-(k - 1), k)
           if abs(al) + abs(be) <= k - 1]
    return pts + [(a, b)]


def gram_det(q, c, phi, pts):
    """(sign, logdet) of the Gram of {rho_c(alpha,beta,0) phi}."""
    M = np.array([weil(al % q, be % q, 0, c, q, phi) for al, be in pts])
    G = M.conj() @ M.T
    sign, ld = np.linalg.slogdet(G)
    return sign, ld


def ptilde_normalised(q, c, k, a, b, anchors, x1, x2, eps):
    """[F(a,b)F(a,-b) - F(-a,b)F(-a,-b)] / [F0(b) F0(-b)], exact dets."""
    phi = make_probe(q, c, anchors, x1, x2, eps)
    phi0 = make_probe(q, c, anchors, x1, x2, 0.0)
    out = {}
    for sa in (1, -1):
        acc = 0.0
        for sb in (1, -1):
            s1, l1 = gram_det(q, c, phi, system_points(k, sa * a, sb * b))
            s0, l0 = gram_det(q, c, phi0, system_points(k, sa * a, sb * b))
            acc += l1 - l0
        out[sa] = np.exp(acc)
    return out[1] - out[-1]


def sheet_alphas(k, a, b, beta):
    """Alpha-set of sheet beta in the system D_{k-1} union {(a,b)}."""
    al = [x for x in range(-(k - 1), k) if abs(x) + abs(beta) <= k - 1]
    if beta == b:
        al.append(a)
    return al


def kernel(q, c, anchors, alphas, beta, s, t):
    """K_beta(s,t) = w(s)^dag (W W^dag)^{-1} w(t) on the alpha-set."""
    al = np.array(alphas)
    W = np.exp(2j * np.pi * np.outer(al, [c * (A + beta) for A in anchors])
               / q)
    A = W @ W.conj().T
    ws = np.exp(2j * np.pi * s * al / q)
    wt = np.exp(2j * np.pi * t * al / q)
    return ws.conj() @ np.linalg.solve(A, wt)


def s_analytic(q, c, k, a, b, anchors):
    """The mixed two-anchor coefficient S(a,b) from the kernel formula."""
    total = 0.0
    for zb in (b, -b):
        for beta in (zb - 1, zb):
            alow = sheet_alphas(k, a, zb, beta)
            ahigh = sheet_alphas(k, a, zb, beta + 1)
            if not alow or not ahigh:
                continue
            t1 = c * (anchors[0] + 1 + beta)
            t2 = c * (anchors[1] + 1 + beta)
            K1 = kernel(q, c, anchors, alow, beta, t1, t2)
            K2 = kernel(q, c, anchors, ahigh, beta + 1, t2, t1)
            total += (K1 * K2).imag
    return total


def check_bmirror(q, c, k, a, b, anchors, rg):
    """Numerical re-check of Delta_{q-c}(a,b) = Delta_c(a,-b)."""
    phi = make_probe(q, c, anchors, 0.3 + 0.7j, -0.5 + 0.2j, 0.05)
    _, l1 = gram_det(q, q - c, phi, system_points(k, a, b))
    _, l2 = gram_det(q, c, phi, system_points(k, a, -b))
    return abs(l1 - l2)


def main():
    q, c, k = 211, 5, 3
    rg = np.random.default_rng(11)
    geoms = [
        [0, 40, 80, 120, 160, 25],
        [0, 33, 71, 104, 143, 181],
        [7, 52, 96, 130, 168, 199],
    ]
    pairs = [(2, 1), (1, 2)]
    ok = True

    print(f"q={q} c={c} k={k}  (m={len(geoms[0])} anchors)")
    bm = check_bmirror(q, c, k, 2, 1, geoms[0], rg)
    print(f"b-mirror logdet identity: |diff| = {bm:.2e}")
    ok &= bm < 1e-8

    for anchors in geoms:
        for a, b in pairs:
            x1 = complex(rg.normal(), rg.normal())
            x2 = complex(rg.normal(), rg.normal())
            S = s_analytic(q, c, k, a, b, anchors)
            print(f"anchors={anchors} pair=({a},{b})  S = {S:+.6e}")
            for eps in (1e-2, 1e-3, 1e-4):
                pp = ptilde_normalised(q, c, k, a, b, anchors, x1, x2, eps)
                pm = ptilde_normalised(q, c, k, a, b, anchors, -x1, -x2, eps)
                meas = 0.5 * (pp + pm) / (4 * eps ** 2
                                          * (x1 * np.conj(x2)).imag)
                rel = abs(meas - S) / max(abs(S), 1e-300)
                print(f"    eps={eps:.0e}: measured {meas:+.6e}"
                      f"  rel.err {rel:.2e}")
                if eps == 1e-3:
                    ok &= rel < 1e-4 and abs(S) > 0
    print("ALL PASS" if ok else "** FAIL **")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
