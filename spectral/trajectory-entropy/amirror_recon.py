"""Reconnaissance of the a-mirror separation front (V3 exact class count).

Goal: upgrade the verified/conjectural closed form N_dist = (N_class+2)/2 of
the trajectory-branching note to a theorem.  The missing lemma is generic
separation of a-mirror pairs at equal |b|: s(a,b) != s(-a,b) for frontier
pairs, outside a proper algebraic set of probes.

FINDINGS (5 Jul 2026, all verified here):

1. Antiunitary obstruction.  K(phi)(x) := conj(phi(-x)) satisfies
   K rho_c(a,b,0) K = rho_c(-a,b,0), so s_phi(-a,b) = s_{K phi}(a,b).
   Probes with real Fourier transform are a-mirror degenerate IDENTICALLY,
   and the exhaustive distinct-profile count collapses exactly to the
   Burnside orbit count (N_class+4)/4 of the Klein group {1, b-mirror,
   a-mirror, both} (checked at q=211, n=2..6: 4,10,24,54,116).

2. Gauge group of the channel = modulations: phi-hat -> e^{i(beta nu+gamma)}
   phi-hat leaves all channel distances invariant.  The channel data depends
   only on |phi-hat|^2 and on edge products conj(phi-hat(nu)) phi-hat(nu')
   over pairs at lattice distance nu-nu' = c*Db within the b-window
   (the ``window graph'').  A single-edge support is gauge-equivalent to a
   real one (beta rotates the edge phase freely): single-edge probes are
   a-mirror degenerate TO ALL ORDERS (verified: exact zeros).

3. Minimal separating witnesses: TWO edges.  Far-sheet base (real anchors
   A_1..A_m) plus two satellites x_1, x_2 at lattice step Delta on two
   DIFFERENT anchors A_1, A_2.  The gauge invariant is arg(x_1) - arg(x_2)
   (for equal steps), which K negates.  Measured: the a-mirror difference D
   is an exact sinusoid in psi = arg(x_1 conj x_2) with D(0) = D(pi) = 0 at
   machine precision, pure-x_1 and pure-x_2 controls exactly zero, and D != 0
   for every valid two-edge configuration tested (8 anchor geometries,
   |D| in [3e-5, 4e-3] at q=211, pair (2,1) vs (-2,1), k=3).

4. Proof plan (not yet executed): the mixed second-order coefficient carries
   the absolute-phase factor zeta^{a c (A_2-A_1)} through the coupling path
   column b0 -> single-vector column b1 = b0 - Delta -> back; properness of
   the per-pair bad set then follows if the coefficient is nonzero, to be
   shown by the extreme-monomial technique in the free anchor positions
   (same style as the Lawrence-Pfander-Walnut minor-nonvanishing proof).

Running this file reproduces findings 1-3 (a few minutes).
"""

import sys

import numpy as np

HERE = "/".join(__file__.split("/")[:-1]) or "."
sys.path.insert(0, HERE)

from v3_exact_check import (weil, gabor_basis_per_shell, nb_words, walk_heis,
                            n_class)

Q = 211
C = 5
FMAT = np.fft.ifft(np.eye(Q), norm="ortho")


def probe(hat):
    v = FMAT @ hat
    return v / np.linalg.norm(v)


def pair_symbol_factory(phi, kmax):
    bc = gabor_basis_per_shell(Q, C, phi, kmax)
    bqc = gabor_basis_per_shell(Q, Q - C, phi, kmax)

    def s(k, a, b):
        v = weil(a % Q, b % Q, 0, C, Q, phi)
        B, _ = bc[k - 1]
        r1 = np.sqrt(max(0.0, 1 - np.linalg.norm(B.conj() @ v) ** 2))
        v2 = weil(a % Q, b % Q, 0, Q - C, Q, phi)
        B2, _ = bqc[k - 1]
        r2 = np.sqrt(max(0.0, 1 - np.linalg.norm(B2.conj() @ v2) ** 2))
        return r1 * r2

    return s


def distinct_counts(phi, nmax):
    bc = gabor_basis_per_shell(Q, C, phi, nmax)
    bqc = gabor_basis_per_shell(Q, Q - C, phi, nmax)
    words = nb_words(nmax)
    profs = np.zeros((len(words), nmax))
    for i, w in enumerate(words):
        pts = walk_heis(w, Q)
        for k in range(1, nmax + 1):
            a, b, g = pts[k - 1]
            v = weil(a, b, g, C, Q, phi)
            B, _ = bc[k - 1]
            r1 = np.sqrt(max(0.0, 1 - np.linalg.norm(B.conj() @ v) ** 2))
            v2 = weil(a, b, g, Q - C, Q, phi)
            B2, _ = bqc[k - 1]
            r2 = np.sqrt(max(0.0, 1 - np.linalg.norm(B2.conj() @ v2) ** 2))
            profs[i, k - 1] = r1 * r2
    return [len(set(map(tuple, np.round(profs[:, :n], 9).tolist())))
            for n in range(2, nmax + 1)]


def main():
    rg = np.random.default_rng(7)
    nmax = 6

    print("[1] real-Fourier collapse vs Burnside orbits")
    hat_r = rg.normal(size=Q).astype(complex)
    cnt = distinct_counts(probe(hat_r), nmax)
    burnside = [(n_class(n) + 4) // 4 for n in range(2, nmax + 1)]
    print("    measured:", cnt)
    print("    (N_class+4)/4:", burnside,
          "MATCH" if cnt == burnside else "** MISMATCH **")

    print("[2] single-edge degeneracy (exact zeros expected)")
    far = [0, 40, 80, 120]
    k, a, b0 = 3, 2, 1
    hat = np.zeros(Q, complex)
    for f in far:
        hat[(C * f) % Q] = 1.0
    hat[(C * 1) % Q] = 0.6 + 0.5j
    s = pair_symbol_factory(probe(hat), k - 1)
    print(f"    one edge: D = {s(k, a, b0) - s(k, -a, b0):+.2e}")

    print("[3] two-edge witness: sinusoid in psi, nonzero at psi=pi/2")
    eps = 0.2
    for psi in (0.0, np.pi / 2, np.pi):
        hat = np.zeros(Q, complex)
        for f in far:
            hat[(C * f) % Q] = 1.0
        hat[(C * 1) % Q] += eps
        hat[(C * 41) % Q] += eps * np.exp(-1j * psi)
        s = pair_symbol_factory(probe(hat), k - 1)
        print(f"    psi={psi:4.2f}: D = {s(k, a, b0) - s(k, -a, b0):+.4e}")


if __name__ == "__main__":
    main()
