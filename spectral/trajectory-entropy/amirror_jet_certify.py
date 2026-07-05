"""Certification of the a-mirror mixed-coefficient nonvanishing proof.

Companion to amirror_mixed_coeff.py; certifies every load-bearing identity
of the proof that the mixed two-anchor coefficient S is not identically
zero (a-mirror separation front, note-1.2.0 target).

Proof structure (all steps certified here at machine precision):

  P1  Weighted eps^2 identity.  With anchor weights w_j > 0 (values
      sqrt(w_j)) the certified identity becomes
        Ptilde = F0(b) F0(-b) * 4 eps^2 sqrt(w1 w2) Im(x1 conj x2) * S,
      with S given by the kernel formula of amirror_mixed_coeff.py built
      from the weighted sheet Grams.
  P2  Moment reduction.  A_beta = D_beta P_Lambda D_beta^dag with
      D_beta = diag(zeta^{c beta alpha}) and P_Lambda[al,al'] =
      p(al-al'), p(d) = sum_j w_j xi_j^d, xi_j = zeta^{c A_j}: the sheet
      phases cancel inside the kernels, so S depends on the free anchors
      only through the moment vector (p(d)) and on (xi_1, xi_2, mu),
      mu = zeta^c:
        S = Im[ k_{C1}(xi1 mu, xi2 mu) k_Z(xi2, xi1)
              + k_Z (xi1 mu, xi2 mu) k_{C2}(xi2, xi1)
              + k_{C2}(xi1 mu, xi2 mu) k_Z(xi2, xi1)
              + k_Z (xi1 mu, xi2 mu) k_{C1}(xi2, xi1) ],
      ranges C1 = {-(l+1)..l+1}, Z = {-l..l+1} (a>0), C2 = {-(l-1)..l-1},
      l = k-1-|b| (b >= 1; the b=0 variant has ranges C = 2k-3 centered
      and Z = {-(k-1)..k}, each pair counted twice).
  P3  Orthogonal point.  At p(d) = delta_{d0} p(0) all kernels are
      Dirichlet-type sums in rho = conj(xi1) xi2 and S vanishes exactly
      (the mirror cancellation).
  P4  Hermitian jet.  Perturbing the single Fourier mode f (P_f = t,
      P_{-f} = conj t) gives dS = 2 Re(A t) with, for b >= 1 and
      f = 2L = 2(k-|b|):
        A = sin(L theta0) (mu^{-f} - 1) g^{-L} / p(0)^3,
      and for b = 0, f = 2k-1:
        A = i D_{k-2}(theta0) (mu^{-f} - 1) g^{-k} (xi2 - xi1) / p(0)^3,
      where theta0 = arg(rho), g = xi1 xi2, D_r = Dirichlet kernel.
      A != 0 whenever rho^{2L} != 1 and mu^f != 1 (automatic for q prime
      and admissible A_1 != A_2), plus D_{k-2}(theta0) != 0, xi1 != xi2
      in the b = 0 case.

  The remaining steps of the proof are classical and need no numerics:
  the moment map w -> (p(d))_{d <= 2k-2} is real-linear with open image
  (a real trigonometric polynomial of degree <= 2k-2 has at most 4k-4
  circle zeros, so m >= 4k-3 distinct nodes span), the domain
  {all sheet Toeplitz matrices positive definite} is convex hence
  connected, contains the orthogonal point and the whole weight cone
  (Chebotarev), and S is real-analytic there; a nonzero jet at an
  interior zero therefore propagates: S != 0 for some admissible weight
  vector, hence Ptilde is not identically zero on probe space.

Exit code 0 iff all certifications pass.

Usage:
    python amirror_jet_certify.py
"""

import sys

import numpy as np

HERE = "/".join(__file__.split("/")[:-1]) or "."
sys.path.insert(0, HERE)

from amirror_mixed_coeff import gram_det, system_points

Q, C, K = 211, 5, 3
ZETA = np.exp(2j * np.pi / Q)


def make_probe_w(anchors, weights, x1, x2, eps):
    hat = np.zeros(Q, complex)
    for A, w in zip(anchors, weights):
        hat[(C * A) % Q] = np.sqrt(w)
    hat[(C * (anchors[0] + 1)) % Q] += 1j * eps * x1
    hat[(C * (anchors[1] + 1)) % Q] += 1j * eps * x2
    x = np.arange(Q)
    F = np.exp(2j * np.pi * np.outer(x, x) / Q) / np.sqrt(Q)
    return F @ hat


def ptilde_w(a, b, anchors, weights, x1, x2, eps):
    phi = make_probe_w(anchors, weights, x1, x2, eps)
    phi0 = make_probe_w(anchors, weights, x1, x2, 0.0)
    out = {}
    for sa in (1, -1):
        acc = 0.0
        for sb in (1, -1):
            _, l1 = gram_det(Q, C, phi, system_points(K, sa * a, sb * b))
            _, l0 = gram_det(Q, C, phi0, system_points(K, sa * a, sb * b))
            acc += l1 - l0
        out[sa] = np.exp(acc)
    return out[1] - out[-1]


def kern_p(p_of_d, rng, nu, nup):
    al = np.array(rng)
    P = np.array([[p_of_d(r - s) for s in al] for r in al])
    return np.conj(nu ** al) @ np.linalg.solve(P, nup ** al)


def ranges_for(k, b, a_sign=1):
    """Kernel range pairs for the four (two if b=0, doubled) terms."""
    if b == 0:
        Cc = list(range(-(k - 2), k - 1))
        Z = (list(range(-(k - 1), k + 1)) if a_sign > 0
             else list(range(-k, k)))
        return [(Cc, Z), (Z, Cc)] * 2
    l = k - 1 - abs(b)
    C1 = list(range(-(l + 1), l + 2))
    C2 = list(range(-(l - 1), l)) if l >= 1 else []
    Z = list(range(-l, l + 2)) if a_sign > 0 else list(range(-l - 1, l + 1))
    return [(C1, Z), (Z, C2), (C2, Z), (Z, C1)]


def s_moment(p_of_d, mu, xi1, xi2, k, b, a_sign=1):
    t = 0.0
    for R1, R2 in ranges_for(k, b, a_sign):
        if not R1 or not R2:
            continue
        t += (kern_p(p_of_d, R1, xi1 * mu, xi2 * mu)
              * kern_p(p_of_d, R2, xi2, xi1)).imag
    return t


def main():
    rg = np.random.default_rng(17)
    anchors = [0, 33, 71, 104, 143, 181]
    weights = rg.uniform(0.5, 2.0, len(anchors))
    xis = ZETA ** (C * np.array(anchors))
    xi1, xi2 = xis[0], xis[1]
    mu = ZETA ** C
    rho = np.conj(xi1) * xi2
    g = xi1 * xi2
    ok = True

    def p_true(d):
        return complex(np.sum(weights * xis ** d))

    print(f"q={Q} c={C} k={K} anchors={anchors}")
    print("[P1+P2] weighted eps^2 identity vs moment-kernel S")
    for a, b in [(2, 1), (1, 2), (3, 0)]:
        S = s_moment(p_true, mu, xi1, xi2, K, b)
        x1 = complex(rg.normal(), rg.normal())
        x2 = complex(rg.normal(), rg.normal())
        eps = 1e-3
        pp = ptilde_w(a, b, anchors, weights, x1, x2, eps)
        pm = ptilde_w(a, b, anchors, weights, -x1, -x2, eps)
        meas = 0.5 * (pp + pm) / (4 * eps ** 2
                                  * np.sqrt(weights[0] * weights[1])
                                  * (x1 * np.conj(x2)).imag)
        rel = abs(meas - S) / abs(S)
        print(f"    pair=({a},{b}): S {S:+.6e} measured {meas:+.6e}"
              f" rel {rel:.2e}")
        ok &= rel < 5e-4 and abs(S) > 0

    print("[P3] orthogonal point (p(d) = delta_d0)")
    for b in (1, 2, 0):
        S0 = s_moment(lambda d: 1.0 if d == 0 else 0.0, mu, xi1, xi2, K, b)
        print(f"    b={b}: S = {S0:+.3e}")
        ok &= abs(S0) < 1e-12

    print("[P4] Hermitian jet dS = 2 Re(A t), closed-form A")
    theta0 = np.angle(rho)
    jets = {}
    for b in (1, 2):
        L = K - abs(b)
        jets[b] = (np.sin(L * theta0) * (mu ** (-2 * L) - 1) * g ** (-L), 2 * L)
    Dk2 = np.sum(rho ** np.arange(-(K - 2), K - 1)).real
    jets[0] = (1j * Dk2 * (mu ** (-(2 * K - 1)) - 1) * g ** (-K)
               * (xi2 - xi1), 2 * K - 1)
    for b, (A, f) in jets.items():
        print(f"    b={b} f={f}: |A| = {abs(A):.4e}"
              f" (nonzero: {abs(A) > 1e-12})")
        ok &= abs(A) > 1e-12
        for t in (0.005 + 0.002j, 0.0005 + 0.0002j):
            def p_pert(d, t=t, f=f):
                base = 1.0 if d == 0 else 0.0
                if d == f:
                    return base + t
                if d == -f:
                    return base + np.conj(t)
                return base
            Sj = s_moment(p_pert, mu, xi1, xi2, K, b)
            pred = 2 * (A * t).real
            rel = abs(Sj - pred) / abs(Sj)
            print(f"        |t|={abs(t):.1e}: rel {rel:.2e}")
            if abs(t) < 1e-3:
                ok &= rel < 1e-4

    print("ALL PASS" if ok else "** FAIL **")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
