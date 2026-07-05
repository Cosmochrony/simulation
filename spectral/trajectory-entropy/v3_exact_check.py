"""Exhaustive verification of the V3 full-history compression theorem.

Companion to the trajectory-branching note (Beau2026tb), section ``The
Full-History Channel: Exact Compression Rate''.  Unlike the sampled campaign
of trajectory_branching.py, this script enumerates ALL 4*3^(n-1)
non-backtracking words (no sampling) and verifies, at machine precision and
without binning:

  C1  Gabor span rank at end of shell j equals min(2j^2+2j+1, q)
      (disk-span lemma + LPW full spark).
  C2  geodesic death: the pair symbol s_k vanishes exactly iff the abelian
      shadow satisfies |z_k|_1 <= k-1; one inward step kills the profile
      forever (alive <=> L1-outward geodesic).
  C3  outward geodesic count A(m) = 2^(m+2) - 4.
  C4  distinct-profile count against the theorem bracket
      2^n <= N_dist <= 2^(n+2) and against the verified/conjectural closed
      form N_dist(n) = (N_class(n)+2)/2 for n >= 2, with
      N_class(n) = A(n) + sum_{d=3..n} (A(d-1)-4).
  C5  Chebotarev witness of the |b|-level separation lemma: an m-sparse
      far-apart Fourier probe gives s(z)=0 for a deep column and s(z')>0
      for a shallow one.

Exit code 0 iff all checks pass.  Runtime: a few minutes (q=101 and q=211,
n <= 7, exhaustive).

Usage:
    python v3_exact_check.py
"""

import sys

import numpy as np


def weil(a, b, g, c, q, phi):
    """Exact Weil action rho_c(a,b,g)|phi>."""
    x = np.arange(q)
    xin = (x - a) % q
    ph = np.exp(2j * np.pi * c * ((g + b * xin) % q) / q)
    return ph * phi[xin]


def gabor_basis_per_shell(q, c, phi, jmax):
    """Orthonormal bases (SVD) of span{rho_c(a,b,0)phi : |a|+|b| <= j}."""
    out = []
    vecs = []
    seen = set()
    for j in range(jmax + 1):
        for a in range(-j, j + 1):
            for b in range(-j, j + 1):
                if abs(a) + abs(b) == j and (a % q, b % q) not in seen:
                    seen.add((a % q, b % q))
                    vecs.append(weil(a % q, b % q, 0, c, q, phi))
        m = np.array(vecs)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        r = int((s > 1e-10 * s[0]).sum())
        out.append((vh[:r], r))
    return out


def nb_words(n):
    """All non-backtracking words over generators 0..3 = X,X^-1,Y,Y^-1."""
    words = [[g] for g in range(4)]
    for _ in range(n - 1):
        words = [w + [g] for w in words for g in range(4) if g != w[-1] ^ 1]
    return words


GEN = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def walk_heis(word, q):
    """Heisenberg walk from the identity; returns [(a,b,gamma)] per step."""
    a = b = g = 0
    out = []
    for w in word:
        da, db = GEN[w]
        g = (g + a * db) % q
        a = (a + da) % q
        b = (b + db) % q
        out.append((a, b, g))
    return out


def a_count(m):
    """Outward-geodesic word count A(m) = 2^(m+2) - 4 (m >= 1)."""
    return 2 ** (m + 2) - 4 if m >= 1 else 1


def n_class(n):
    """Unconditional class count N_class(n)."""
    return a_count(n) + sum(a_count(d - 1) - 4 for d in range(3, n + 1))


def check_prime(q, nmax, seed):
    """Run C1-C4 exhaustively at prime q; returns True iff all pass."""
    ok = True
    rg = np.random.default_rng(seed)
    phi = rg.normal(size=q) + 1j * rg.normal(size=q)
    phi /= np.linalg.norm(phi)
    c = int(rg.integers(1, q))
    bases_c = gabor_basis_per_shell(q, c, phi, nmax)
    bases_qc = gabor_basis_per_shell(q, q - c, phi, nmax)

    ranks = [r for _, r in bases_c]
    pred = [min(2 * j * j + 2 * j + 1, q) for j in range(nmax + 1)]
    c1 = ranks == pred
    ok &= c1
    print(f"q={q}  C1 rank sequence: {'PASS' if c1 else 'FAIL ' + str(ranks)}")

    words = nb_words(nmax)
    profs = np.zeros((len(words), nmax))
    c2 = True
    for i, w in enumerate(words):
        pts = walk_heis(w, q)
        aa = bb = 0
        for k in range(1, nmax + 1):
            a, b, g = pts[k - 1]
            v = weil(a, b, g, c, q, phi)
            bc, _ = bases_c[k - 1]
            r1 = np.sqrt(max(0.0, 1 - np.linalg.norm(bc.conj() @ v) ** 2))
            v2 = weil(a, b, g, q - c, q, phi)
            bqc, _ = bases_qc[k - 1]
            r2 = np.sqrt(max(0.0, 1 - np.linalg.norm(bqc.conj() @ v2) ** 2))
            s = r1 * r2
            profs[i, k - 1] = s
            da, db = GEN[w[k - 1]]
            aa += da
            bb += db
            alive = (abs(aa) + abs(bb)) == k
            if alive != (s > 1e-8):
                c2 = False
    ok &= c2
    print(f"q={q}  C2 geodesic death (alive <=> nonzero): "
          f"{'PASS' if c2 else 'FAIL'}")

    c3 = True
    for m in range(1, nmax + 1):
        cnt = 0
        for w in nb_words(m):
            aa = bb = 0
            outward = True
            for k, gidx in enumerate(w, 1):
                da, db = GEN[gidx]
                aa += da
                bb += db
                if abs(aa) + abs(bb) != k:
                    outward = False
                    break
            cnt += outward
        c3 &= cnt == a_count(m)
    ok &= c3
    print(f"q={q}  C3 A(m) = 2^(m+2)-4: {'PASS' if c3 else 'FAIL'}")

    c4 = True
    print(f"q={q}  C4 distinct profiles per horizon:")
    for n in range(2, nmax + 1):
        rows = np.round(profs[:, :n], 9)
        uniq = len(set(map(tuple, rows.tolist())))
        closed = (n_class(n) + 2) // 2
        lo, hi = 2 ** n, 2 ** (n + 2)
        good = lo <= uniq <= hi and uniq == closed
        c4 &= good
        print(f"      n={n}: distinct={uniq}  closed-form={closed}  "
              f"bracket=[{lo},{hi}]  {'ok' if good else '** FAIL **'}")
    ok &= c4
    return ok


def check_witness(q, k, d, dp):
    """C5: Chebotarev witness separating column depths d > d' at step k."""
    m = 2 * dp + 1
    assert m <= 2 * d - 1 and (2 * k - 1) ** 2 < q
    c = 3
    sig = [(j) * (2 * k - 1) for j in range(m)]
    phi = np.zeros(q, complex)
    fmat = np.fft.ifft(np.eye(q), norm="ortho")
    for s in sig:
        phi += fmat[:, (c * s) % q]
    phi /= np.linalg.norm(phi)
    bases_c = gabor_basis_per_shell(q, c, phi, k - 1)
    bases_qc = gabor_basis_per_shell(q, q - c, phi, k - 1)
    bc, _ = bases_c[k - 1]
    bqc, _ = bases_qc[k - 1]

    def sym(a, b):
        v = weil(a % q, b % q, 0, c, q, phi)
        r1 = np.sqrt(max(0.0, 1 - np.linalg.norm(bc.conj() @ v) ** 2))
        v2 = weil(a % q, b % q, 0, q - c, q, phi)
        r2 = np.sqrt(max(0.0, 1 - np.linalg.norm(bqc.conj() @ v2) ** 2))
        return r1 * r2

    s_deep = sym(d, k - d)
    s_shallow = sym(dp, k - dp)
    good = s_deep < 1e-8 < s_shallow
    print(f"q={q}  C5 witness k={k} depths ({d},{dp}): "
          f"s(deep)={s_deep:.2e}  s(shallow)={s_shallow:.3f}  "
          f"{'PASS' if good else '** FAIL **'}")
    return good


def main():
    ok = True
    ok &= check_prime(q=101, nmax=7, seed=1)
    ok &= check_prime(q=211, nmax=7, seed=2)
    ok &= check_witness(q=211, k=4, d=3, dp=1)
    ok &= check_witness(q=211, k=5, d=4, dp=2)
    print("ALL CHECKS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
