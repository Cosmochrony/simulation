"""
Numerical verification: spectral admissibility hierarchy on Cay(PSL(2,7), C_4).

Confirms Hypothesis 7.2 (O31 §7.2 / SpectralAdmissibility v2 §12):
    lambda_3 = lambda_3bar = 28  <  lambda_6 = lambda_8 = 42  <  lambda_7 = 48

where lambda = d - mu, d = |C_4| = 42, and mu is the adjacency eigenvalue.

Predicted eigenvalues from character theory:
    chi_3, chi_3bar: chi(C_4) = 1  =>  mu = 42/3 * 1 = 14  =>  lambda = 28
    chi_6, chi_8:    chi(C_4) = 0  =>  mu = 0               =>  lambda = 42
    chi_7:           chi(C_4) = -1 =>  mu = 42/7 * (-1) = -6 => lambda = 48

Multiplicity check (regular representation: mult = dim(rho)^2):
    lambda = 28: 3^2 + 3^2 = 18
    lambda = 42: 6^2 + 8^2 = 100
    lambda = 48: 7^2       = 49
    lambda =  0: 1^2       = 1  (trivial, mu = 42)
    Total: 1 + 18 + 100 + 49 = 168 = |PSL(2,7)| checksum.
"""

import time
import numpy as np
from itertools import product
from collections import Counter

F = 7


def canonical(M):
    """Canonical representative in PSL(2,7): lexicographic min of {M, -M mod F}."""
    flat = tuple(M.flatten().tolist())
    neg_flat = tuple((-M % F).flatten().tolist())
    return np.array(min(flat, neg_flat)).reshape(2, 2)


def build_psl27():
    """Enumerate all 168 elements of PSL(2, F_7) as canonical 2x2 integer matrices."""
    seen = set()
    elements = []
    for a, b, c, d in product(range(F), repeat=4):
        if (a * d - b * c) % F != 1:
            continue
        M = np.array([[a, b], [c, d]])
        key = tuple(canonical(M).flatten().tolist())
        if key not in seen:
            seen.add(key)
            elements.append(canonical(M))
    return elements


def element_order(M, elem_to_idx, I_key):
    """Return the order of element M in PSL(2,7)."""
    cur = M.copy()
    for k in range(1, 25):
        if tuple(canonical(cur).flatten().tolist()) == I_key:
            return k
        cur = cur @ M % F
    return -1


def build_adjacency(elements, elem_to_idx, generators):
    """Build the 168x168 adjacency matrix for Cay(PSL(2,7), generators)."""
    n = len(elements)
    A = np.zeros((n, n), dtype=int)
    for i, Mi in enumerate(elements):
        for s_idx in generators:
            Ms = elements[s_idx]
            prod_key = tuple(canonical(Mi @ Ms % F).flatten().tolist())
            j = elem_to_idx[prod_key]
            A[i, j] = 1
    return A


def main():
    t0 = time.time()

    # --- Build group ---
    elements = build_psl27()
    assert len(elements) == 168, f"Expected 168, got {len(elements)}"
    elem_to_idx = {tuple(M.flatten().tolist()): i for i, M in enumerate(elements)}
    print(f"|PSL(2,7)| = {len(elements)}  [expected 168]")

    # --- Conjugacy classes via element orders ---
    I_key = tuple(canonical(np.eye(2, dtype=int)).flatten().tolist())
    orders = [element_order(M, elem_to_idx, I_key) for M in elements]
    order_dist = Counter(orders)
    print(f"Order distribution: {dict(sorted(order_dist.items()))}")
    print(f"  [expected: {{1:1, 2:21, 3:56, 4:42, 7:48}}]")

    # Conjugacy class C_4 = all elements of order 4 (size 42 = degree d)
    C4_indices = [i for i, o in enumerate(orders) if o == 4]
    d = len(C4_indices)
    print(f"\nGenerating set S = C_4 (elements of order 4): |S| = {d}  [expected 42]")

    # Symmetry check: S closed under inversion
    inv_check = all(
        elem_to_idx[tuple(canonical(
            pow(int((M[0,0]*M[1,1]-M[0,1]*M[1,0]) % F), F-2, F)
            * np.array([[M[1,1],-M[0,1]],[-M[1,0],M[0,0]]]) % F
        ).flatten().tolist())] in C4_indices
        for M in (elements[i] for i in C4_indices)
    )
    print(f"S closed under inversion: {inv_check}  [expected True]")

    # --- Adjacency matrix and eigenvalues ---
    t1 = time.time()
    A = build_adjacency(elements, elem_to_idx, C4_indices)
    print(f"\nAdjacency matrix built in {time.time()-t1:.3f}s")
    print(f"Row degree (should be {d}): min={A.sum(axis=1).min()}, max={A.sum(axis=1).max()}")

    eigs = np.linalg.eigvalsh(A)
    rounded = [round(e, 6) for e in eigs]
    cnt = Counter(rounded)

    print(f"\nAdjacency eigenvalues mu (value: multiplicity):")
    for mu in sorted(cnt, reverse=True):
        lam = d - mu
        print(f"  mu = {mu:8.4f}  lambda = {lam:6.4f}  mult = {cnt[mu]}")

    # --- Verification against predictions ---
    print(f"\nVerification:")
    predicted = {42.0: 1, 14.0: 18, 0.0: 100, -6.0: 49}
    for mu_pred, mult_pred in sorted(predicted.items(), reverse=True):
        lam = d - mu_pred
        mult_obs = cnt.get(round(mu_pred, 6), 0)
        status = "OK" if mult_obs == mult_pred else "FAIL"
        print(f"  [{status}] mu={mu_pred:5.1f} (lambda={lam:.1f}): "
              f"predicted mult={mult_pred}, observed mult={mult_obs}")

    # Admissibility ratio R = sqrt(lambda_max / lambda_3)
    lam_3 = d - 14.0
    lam_max = d - (-6.0)
    R = np.sqrt(lam_max / lam_3)
    print(f"\nAdmissibility ratio R = sqrt({lam_max}/{lam_3}) = sqrt(12/7) = {R:.6f}")
    print(f"  [compare: Q8: sqrt(4/3)=1.1547, 2I: sqrt(14/9)=1.2472]")

    print(f"\nTotal runtime: {time.time()-t0:.3f}s")


if __name__ == "__main__":
    main()