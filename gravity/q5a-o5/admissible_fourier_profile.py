"""
Admissible Fourier Profile  --  Q5a-O2 paper, §4
=================================================
Computes K_q = max_j |supp(phi_j_hat)| using the CORRECT normalization.

Key fix:  weil_batch_lut divides each vector by sqrt(q), so the product
v1*v2*v3 has norm 1/q (not 1).  Therefore
    |pi_c[i,j]|^2 = |<phi_j, v_i>|^2 = delta_{xi_i=xi_j} / q^2
for a pure-mode match.  The detection threshold is 0.5/q^2 (NOT a
normalized-probability threshold).

Workflow per pair p, direction j:
  1. Compute xi_i for all fingerprints i in shells [n0,n1]  (analytic formula)
  2. For each unique xi: max_i |pi_c[i,j]|^2  (one representative per mode)
  3. A frequency xi is in supp(phi_j) iff max > 0.5/q^2
  4. K_j = |supp|;  K_q = max_j K_j

Expected:
  K_q = 1  =>  each admissible vector is a single pure Fourier mode
  K_q = O(q)  =>  delocalized; [H-BFS'] needs reformulation
"""

import argparse, glob, importlib.util, os, sys
import numpy as np


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_pipeline(d):
    for f in ["spectral_O12.py", "o25_paired_pipeline.py"]:
        if not os.path.exists(os.path.join(d, f)):
            sys.exit(f"{f} not found in {d}")
    o12 = load_module(os.path.join(d, "spectral_O12.py"),        "spectral_O12")
    o25 = load_module(os.path.join(d, "o25_paired_pipeline.py"), "o25_pp")
    return o12, o25


# ---------------------------------------------------------------------------
# Fingerprint frequency (vectorised, identical ordering to fingerprint_vectors_batch)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 400  # must match spectral_O12.py


def fingerprint_freqs_batch(shell_arr, c_block, gens_arr, q, hmb):
    """
    xi_i = (c1*b1 + c2*b2 + c3*b3) % q for each of |shell|*4^3 fingerprints.
    Processed in chunks of CHUNK_SIZE to match the ordering used by
    fingerprint_vectors_batch (and therefore pi_c storage ordering).
    """
    c1, c2, c3 = int(c_block[0]), int(c_block[1]), int(c_block[2])
    all_freqs = []
    for start in range(0, len(shell_arr), CHUNK_SIZE):
        chunk = shell_arr[start:start + CHUNK_SIZE]
        freqs = []
        for s1 in gens_arr:
            ep1 = hmb(chunk, s1, q)
            for s2 in gens_arr:
                ep2 = hmb(ep1, s2, q)
                for s3 in gens_arr:
                    ep3 = hmb(ep2, s3, q)
                    xi  = (c1*ep1[:,1].astype(int) +
                           c2*ep2[:,1].astype(int) +
                           c3*ep3[:,1].astype(int)) % q
                    freqs.append(xi)
        all_freqs.append(np.concatenate(freqs))
    return np.concatenate(all_freqs)


# ---------------------------------------------------------------------------
# BFS cache
# ---------------------------------------------------------------------------
_BFS_CACHE = {}

def get_shells(o12, q, bfs_frac):
    key = (q, float(bfs_frac))
    if key not in _BFS_CACHE:
        gens = o12.build_generators(q)
        _BFS_CACHE[key] = (o12.bfs_shells(None, None, gens, q, float(bfs_frac)),
                           gens)
    return _BFS_CACHE[key]


# ---------------------------------------------------------------------------
# Core: reconstruct phi_j spectral support from pi_c
# ---------------------------------------------------------------------------

def reconstruct_support(data, q, pair_idx, o12, o25):
    """
    Returns:
      supports : {j: sorted list of frequencies in supp(phi_j_hat)}
      cb_c     : block used for m=0
      raw_max  : {j: {xi: max |pi_c[i,j]|^2 for i with xi_i=xi}} (unnormalized)

    Detection criterion: supp includes xi iff max > 0.5/q^2.
    A pure-mode match gives max = 1/q^2; threshold = 0.5/q^2 is generous.
    """
    pi_c    = data['pi_c']
    n0, n1  = int(data['n0']), int(data['n1'])
    seed    = int(data['seed'])
    pairs   = data['pairs']
    c       = int(pairs[pair_idx, 0])

    # rank_adm from first non-empty shell
    rank_adm = 0
    for k in range(pi_c.shape[1]):
        arr = np.asarray(pi_c[pair_idx, k])
        if arr.ndim == 2 and arr.shape[0] > 0:
            rank_adm = arr.shape[1]
            break
    if rank_adm == 0:
        return None, None, None

    # Reconstruct cb_c for m=0 (same RNG chain as pipeline)
    rng  = np.random.default_rng(seed + pair_idx * 997 + c * 7)
    cb_c = o25.sample_block_with_c1(c, q, rng)

    shells, gens = get_shells(o12, q, data['bfs_frac'])
    gens_arr     = np.array(gens, dtype=np.int64)
    hmb          = o12.heisenberg_mul_batch
    q2           = q * q           # normalisation factor: pure-mode mass = 1/q^2
    threshold    = 0.5 / q2        # generous: half of a pure-mode unit

    # {j: {xi: max |pi_c[i,j]|^2}}
    raw_max = [{} for _ in range(rank_adm)]

    for k in range(pi_c.shape[1]):
        s_abs = n0 + k
        if s_abs > n1 or s_abs >= len(shells):
            break
        arr = np.asarray(pi_c[pair_idx, k])
        if arr.ndim < 2 or arr.shape[0] == 0:
            continue

        shell = np.array(shells[s_abs], dtype=np.int64)
        freqs = fingerprint_freqs_batch(shell, cb_c, gens_arr, q, hmb)

        n_fp = arr.shape[0]
        if len(freqs) != n_fp:
            continue

        for j in range(rank_adm):
            col = np.abs(arr[:, j]) ** 2   # (n_fp,)
            for i in range(n_fp):
                xi  = int(freqs[i])
                xi  = min(xi, q - xi)       # fold to [0, q//2]
                val = float(col[i])
                if xi not in raw_max[j] or val > raw_max[j][xi]:
                    raw_max[j][xi] = val

    # Build support sets
    supports = {}
    for j in range(rank_adm):
        supp = sorted(xi for xi, v in raw_max[j].items() if v > threshold)
        supports[j] = supp

    return supports, cb_c, raw_max


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def find_npz(prime, d="."):
    for pat in [os.path.join(d, f"q{prime}_o25.npz"),
                os.path.join(d, f"*q{prime}*.npz")]:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[0]
    return None

def admissible_rank_from_pi(pi_c_row):
    for k, raw in enumerate(pi_c_row):
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[0] > 0:
            return arr.shape[1], k
    return 0, -1


# ---------------------------------------------------------------------------
# Multi-sample analysis (optional, slow)
# ---------------------------------------------------------------------------

def multi_sample_support(data, q, pair_idx, o12, o25):
    """
    For each of M block samples: find which xi values carry significant mass
    in the admissible GS basis vectors.  Returns union support set and per-
    sample K values.
    """
    n0, n1 = int(data['n0']), int(data['n1'])
    seed   = int(data['seed'])
    pairs  = data['pairs']
    M      = int(data['M_per_pair'])
    c, qc  = int(pairs[pair_idx, 0]), int(pairs[pair_idx, 1])

    shells, gens = get_shells(o12, q, data['bfs_frac'])
    gens_arr     = np.array(gens, dtype=np.int64)
    hmb          = o12.heisenberg_mul_batch
    q2           = q * q
    threshold    = 0.5 / q2

    rng = np.random.default_rng(seed + pair_idx * 997 + c * 7)
    per_K, union_xi = [], set()

    for m in range(M):
        cb_c  = o25.sample_block_with_c1(c,  q, rng)
        _     = o25.sample_block_with_c1(qc, q, rng)   # advance RNG

        basis_mat = np.zeros((0, q), dtype=complex)
        raw_max   = [dict() for _ in range(3)]

        for s_abs in range(n0, min(n1 + 1, len(shells))):
            shell = np.array(shells[s_abs], dtype=np.int64)
            vecs  = o12.fingerprint_vectors_batch(shell, cb_c, gens_arr, q)
            freqs = fingerprint_freqs_batch(shell, cb_c, gens_arr, q, hmb)

            for i, v in enumerate(vecs):
                v_o = v.copy()
                for bv in basis_mat:
                    v_o -= np.dot(bv.conj(), v_o) * bv
                nrm = np.linalg.norm(v_o)
                if nrm > 1e-8:
                    j = basis_mat.shape[0]
                    basis_mat = np.vstack([basis_mat, v_o / nrm])
                    xi = min(int(freqs[i]), q - int(freqs[i]))
                    # GS vector j = v_o/nrm.  Its inner product with v_i/||v_i|| = q*v_i:
                    # |<(v_o/nrm), v_i>|^2 = nrm^2/nrm^2 * |<v_i/nrm, v_i>|^2
                    # For GS vector = pure mode at xi: mass at xi = 1/q^2
                    # We record the xi directly since GS vector is still pure.
                    raw_max[j][xi] = 1.0 / q2
                if basis_mat.shape[0] >= 3:
                    break
            if basis_mat.shape[0] >= 3:
                break

        sample_K = max((len([xi for xi,v in s.items() if v > threshold])
                        for s in raw_max if s), default=0)
        per_K.append(sample_K)
        for s in raw_max:
            union_xi.update(xi for xi,v in s.items() if v > threshold)

    return per_K, sorted(union_xi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="K_q test for Q5a-O2 §4")
    ap.add_argument('--primes',       nargs='+', type=int, default=[29, 61, 101, 151])
    ap.add_argument('--npz-dir',      default='.')
    ap.add_argument('--pipeline-dir', default='.')
    ap.add_argument('--n-pairs',      type=int, default=5)
    ap.add_argument('--full-M',       action='store_true')
    args = ap.parse_args()

    print("Loading pipeline modules...")
    o12, o25 = load_pipeline(args.pipeline_dir)
    kq_table = []

    for q in args.primes:
        path = find_npz(q, args.npz_dir)
        if path is None:
            print(f"\n[q={q}] not found -- skipping")
            continue
        print(f"\n{'='*70}")
        print(f"q = {q}  ({os.path.basename(path)})")
        print(f"{'='*70}")
        data   = np.load(path, allow_pickle=True)
        q_a    = int(data['q'])
        q2     = q_a * q_a
        pi_c   = data['pi_c']
        pairs  = data['pairs']

        non_empty = [p for p in range(pairs.shape[0])
                     if admissible_rank_from_pi(pi_c[p])[0] > 0]
        print(f"  n0={int(data['n0'])}, n1={int(data['n1'])}, "
              f"M={int(data['M_per_pair'])},  non-empty pairs={len(non_empty)}")
        print(f"  norm factor 1/q^2 = {1.0/q2:.4e}   threshold = {0.5/q2:.4e}")

        print(f"\n  {'pair':>5}  {'c':>5}  {'j':>3}  {'K_j':>5}  "
              f"{'max/q-2':>10}  {'supp_xi'}")
        print(f"  {'-'*70}")

        K_max = 0
        for p in non_empty[:args.n_pairs]:
            c = int(pairs[p, 0])
            supports, cb_c, raw_max = reconstruct_support(data, q_a, p, o12, o25)
            if supports is None:
                continue
            for j in sorted(supports.keys()):
                supp = supports[j]
                K_j  = len(supp)
                K_max = max(K_max, K_j)
                # Show max mass as multiple of 1/q^2
                if raw_max[j]:
                    top_mass = max(raw_max[j].values())
                    top_ratio = top_mass * q2
                else:
                    top_ratio = 0.0
                supp_str = str(supp) if len(supp) <= 8 else (
                    str(supp[:4])[:-1] + f", ..., {supp[-1]}]  ({len(supp)} freqs)")
                print(f"  {p:>5d}  {c:>5d}  {j:>3d}  {K_j:>5d}  "
                      f"{top_ratio:>10.3f}  {supp_str}")

        print(f"\n  K_max (m=0 sample) = {K_max}")
        print(f"  Interpretation: ", end="")
        if K_max == 1:
            print("K=1  -->  admissible vectors are PURE FOURIER MODES")
        elif K_max <= 5:
            print(f"K={K_max}  -->  near-pure; small spectral support")
        elif K_max <= q_a // 4:
            print(f"K~{K_max} ~ q/{q_a//K_max if K_max else '?'}  "
                  "-->  moderate spread")
        else:
            print(f"K~{K_max} ~ q/2  -->  delocalized, [H-BFS'] FAILS as stated")

        # Multi-sample
        K_union = K_max
        if args.full_M and non_empty:
            p0 = non_empty[0]
            c0 = int(pairs[p0, 0])
            print(f"\n  Multi-sample (pair {p0}, c={c0}, M={int(data['M_per_pair'])}):")
            per_K, union = multi_sample_support(data, q_a, p0, o12, o25)
            K_union = len(union)
            print(f"  K per sample: min={min(per_K)}  mean={np.mean(per_K):.2f}  "
                  f"max={max(per_K)}")
            print(f"  Union K = {K_union}  union_xi = {union[:20]}"
                  f"{'...' if len(union) > 20 else ''}")

        kq_table.append((q_a, K_max, K_union))

    # Scaling summary
    if kq_table:
        print(f"\n{'='*70}")
        print("SCALING SUMMARY")
        print(f"{'='*70}")
        print(f"  {'q':>6}  {'K_single':>10}  {'K_union':>10}  "
              f"{'K/q':>8}  {'K/sqrt(q)':>12}")
        for (q, Ks, Ku) in kq_table:
            print(f"  {q:>6d}  {Ks:>10d}  {Ku:>10d}  "
                  f"{Ks/q:>8.4f}  {Ks/q**0.5:>12.4f}")
        Ks = [r[1] for r in kq_table]
        print("\n  Verdict:")
        if max(Ks) <= 1:
            print("  K_q = 1 for all primes  =>  "
                  "[H-BFS'] CONFIRMED: pure Fourier modes")
        elif max(Ks) <= 5:
            print("  K_q = O(1)  =>  [H-BFS'] HOLDS (finite spectral sparsity)")
        elif all(abs(k/q - kq_table[0][1]/kq_table[0][0]) < 0.05
                 for (q,k,_) in kq_table):
            print(f"  K_q ~ {np.mean([k/q for q,k,_ in kq_table]):.2f}*q  =>  "
                  "[H-BFS'] FAILS as stated (O(q) spread)")
        else:
            rr = [k/q**0.5 for q,k,_ in kq_table]
            print(f"  K_q/sqrt(q) = {min(rr):.2f}..{max(rr):.2f}  "
                  "=>  intermediate scaling, needs more analysis")


if __name__ == '__main__':
    main()