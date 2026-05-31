import numpy as np
from spectral_O12 import build_generators, bfs_shells_depth_capped, fingerprint_vectors_batch
from hcolor_exact_check import primitive_cube_root

def perm_dilation(q, omega):
    """U_omega: (U f)[x] = f[omega*x mod q]. Permutation matrix (q x q)."""
    x = np.arange(q)
    src = (omega * x) % q          # output[x] = input[src[x]]
    P = np.zeros((q, q))
    P[x, src] = 1.0
    return P

def check(q, n_shells=6):
    omega = primitive_cube_root(q); w2 = (omega*omega) % q
    gens = build_generators(q); gens_arr = np.array(gens, np.int64)
    shells = bfs_shells_depth_capped(gens, q, n_shells+1)
    U = perm_dilation(q, omega)
    c, c2, c3 = 7 % q or 1, 24 % q or 1, 52 % q or 1
    # ensure generic
    if (c+c2+c3) % q == 0: c3 = (c3+1) % q
    blk  = np.array((c, c2, c3), np.int64)
    blkW = np.array(((omega*c)%q, (omega*c2)%q, (omega*c3)%q), np.int64)

    max_resid = 0.0          # || fp_w - phase * U fp ||  after removing per-vector phase
    max_rank_gap = 0         # |#distinct frequencies B - GS-style independence count|
    for n in range(1, n_shells+1):
        shell = np.array(shells[n], np.int64)
        FP  = fingerprint_vectors_batch(shell, blk,  gens_arr, q)   # (m, q)
        FPW = fingerprint_vectors_batch(shell, blkW, gens_arr, q)   # (m, q)
        UFP = FP @ U.T                                              # (U fp) per row
        # per-vector phase = <UFP, FPW> / |<UFP, FPW>| , then residual
        ip = np.sum(UFP.conj() * FPW, axis=1)
        phase = ip / np.maximum(np.abs(ip), 1e-300)
        resid = FPW - phase[:, None] * UFP
        max_resid = max(max_resid, float(np.max(np.abs(resid))))

        # pure-Fourier-mode claim: each fp row is q^{-1} * e_B. Recover B by argmax of
        # |FFT|, and check the row equals its single-frequency reconstruction.
        # Independent check that rank == number of distinct frequencies B:
        F = np.fft.fft(FP, axis=1)                 # each row should be a delta in freq
        Bidx = np.argmax(np.abs(F), axis=1)        # dominant frequency per vector
        # verify each row is truly single-frequency (peak carries all the energy)
        peak = np.abs(F[np.arange(len(F)), Bidx])
        tot  = np.linalg.norm(F, axis=1)
        single_mode = np.max(np.abs(peak/np.maximum(tot,1e-300) - 1.0))
        # rank via distinct frequencies vs numerical rank of FP
        n_distinct = len(np.unique(Bidx))
        num_rank = np.linalg.matrix_rank(FP, tol=1e-8)
        max_rank_gap = max(max_rank_gap, abs(n_distinct - num_rank))
        if n in (2,4,6):
            print(f"  shell {n}: |fp_w - phase*U fp|={np.max(np.abs(resid)):.2e}  "
                  f"single-mode dev={single_mode:.2e}  #distinct B={n_distinct}  numrank={num_rank}")
    print(f"q={q}: MAX |fp_w - phase*U_omega fp| = {max_resid:.2e}   "
          f"max|#distinctB - rank| = {max_rank_gap}")
    return max_resid, max_rank_gap

for q in [13, 61, 151]:
    check(q, n_shells=6)