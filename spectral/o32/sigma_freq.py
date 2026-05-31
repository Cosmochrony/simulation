"""
sigma_freq.py
Closed-form sigma via frequency counting -- a proof-grade simplification.

Each O12 fingerprint vector is a single additive character:
    fp[x] = q^{-3/2} exp(2pi i (A + B x)/q),  B = c1 b1 + c2 b2 + c3 b3 (mod q),
where (a_i,b_i,g_i) are the three accumulated generator steps. Additive characters
are orthonormal, so a set of fingerprint vectors is linearly independent iff their
frequencies B are distinct. Hence the Gram-Schmidt rank increment of a shell equals
the number of NEW distinct frequencies it contributes, and
    sigma_c(n) = |new distinct B at shell n| / |S_n|.
No Gram-Schmidt, no SVD: O(|S_n|) per shell instead of O(|S_n| q^2).
"""
import numpy as np
from spectral_O12 import heisenberg_mul_batch


def compute_block_capacity_freq(shells, c_block, q, gens, n_max=None):
    c1, c2, c3 = (int(c_block[0]) % q, int(c_block[1]) % q, int(c_block[2]) % q)
    gens_arr = [tuple(int(x) for x in g) for g in gens]
    seen = set()                 # distinct frequencies B realised so far
    sigma_vals, delta_r_vals, shell_sizes = [], [], []
    for n, shell in enumerate(shells):
        if n_max is not None and n > n_max:
            break
        if len(shell) == 0:
            break
        shell_arr = np.array(shell, dtype=np.int64)
        before = len(seen)
        for s1 in gens_arr:
            ep1 = heisenberg_mul_batch(shell_arr, s1, q)
            for s2 in gens_arr:
                ep2 = heisenberg_mul_batch(ep1, s2, q)
                for s3 in gens_arr:
                    ep3 = heisenberg_mul_batch(ep2, s3, q)
                    B = (c1 * ep1[:, 1] + c2 * ep2[:, 1] + c3 * ep3[:, 1]) % q
                    seen.update(B.tolist())
                    if len(seen) >= q:
                        break
                if len(seen) >= q:
                    break
            if len(seen) >= q:
                break
        delta_r = len(seen) - before
        sz = len(shell)
        sigma_vals.append(delta_r / sz if sz > 0 else 0.0)
        delta_r_vals.append(delta_r)
        shell_sizes.append(sz)
        if len(seen) >= q:
            break
    return (np.array(sigma_vals), np.array(delta_r_vals),
            np.array(shell_sizes), len(seen))