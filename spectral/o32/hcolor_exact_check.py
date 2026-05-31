"""
hcolor_exact_check.py
Deterministic, Monte-Carlo-free test of [H-color]_pointwise on the standard
Cayley graph of Heis_3(Z/qZ).

O31 (Prop. 4.19) proves the BFS sector RANK equality r_c(n)=r_wc(n)=r_w2c(n)
exactly. The open question (O32 open problem 3) is whether the O12 Born-Infeld
FINGERPRINT observable sigma_c(n) inherits that equality at finite q, or only up
to O(q^-1). The O32 campaign answers this with Monte-Carlo over random auxiliary
(c2,c3), whose sampling variance is exactly what muddies R_var(q).

Here we remove the sampling entirely: for matched blocks related across the
colour orbit {c, wc, w2c}, we compute sigma exactly and compare it POINTWISE,
shell by shell. Two matchings are tested:
  (A) same auxiliary:   (c,c2,c3) vs (wc,c2,c3) vs (w2c,c2,c3)
  (B) scaled auxiliary:  (c,c2,c3) vs (wc,wc2,wc3) vs (w2c,w2c2,w2c3)
If max|sigma_i(n)-sigma_j(n)| == 0 to machine precision over all base blocks,
[H-color]_pointwise holds for that matching; a nonzero value is a finite-q
counterexample. No M, no averaging: each base block is an exact comparison.

Uses compute_block_capacity_fast + bfs_shells_depth_capped from the unified
spectral_O12.py.
"""
import argparse
import sys
import numpy as np
from spectral_O12 import (build_generators, bfs_shells_depth_capped,
                          compute_block_capacity_freq)


def primitive_cube_root(q):
    if q % 3 != 1:
        return None
    for x in range(2, q):
        if pow(x, 3, q) == 1:
            return x
    return None


def orbit_reps(q, omega):
    """One representative c per cyclic orbit {c, wc, w2c}, with pairwise-distinct,
    non-conjugate, generic members (matches o32 find_colour_triplets validity)."""
    w2 = (omega * omega) % q
    seen, reps = set(), []
    for c in range(1, q):
        t = (c, (omega * c) % q, (w2 * c) % q)
        key = frozenset(t)
        if key in seen or len(key) < 3:
            continue
        seen.add(key)
        conjs = [(q - x) % q for x in t]
        if any(t[i] == conjs[j] for i in range(3) for j in range(3) if i != j):
            continue
        reps.append(c)
    return reps


def sigma_profile(shells, block, q, gens, n_max):
    sv, _, _, _ = compute_block_capacity_freq(shells, np.array(block, np.int64),
                                              q, gens, n_max=n_max)
    return sv


def run(q, n0, n1, buffer=3, n_orbits=None, aux_pairs=None, seed=0, exhaustive=False):
    omega = primitive_cube_root(q)
    if omega is None:
        print(f"q={q}: q != 1 (mod 3) -> no primitive cube root, no colour triplets. "
              f"This is a CONTROL prime for the colour sector; the [H-color] triplet "
              f"test only applies to test primes q == 1 (mod 3) "
              f"(e.g. 61, 151, 211, 307, 313, 331, 337, 349, ...).")
        return None

    w2 = (omega * omega) % q
    gens = build_generators(q)
    depth = n1 + buffer
    shells = bfs_shells_depth_capped(gens, q, depth)
    n_max = depth
    reps = orbit_reps(q, omega)
    if n_orbits:
        reps = reps[:n_orbits]
    rng = np.random.default_rng(seed)

    def aux_list(c):
        if exhaustive:
            out = []
            for c2 in range(1, q):
                for c3 in range(1, q):
                    if (c + c2 + c3) % q != 0:
                        out.append((c2, c3))
            return out
        out = []
        while len(out) < (aux_pairs or 3):
            c2 = int(rng.integers(1, q)); c3 = int(rng.integers(1, q))
            if (c + c2 + c3) % q != 0:
                out.append((c2, c3))
        return out

    maxdev = {'A': 0.0, 'B': 0.0}
    argmax = {'A': None, 'B': None}
    nblocks = 0
    for c in reps:
        cc = [c, (omega * c) % q, (w2 * c) % q]
        for (c2, c3) in aux_list(c):
            nblocks += 1
            # matching A: same auxiliary across the orbit
            sA = [sigma_profile(shells, (cc[i], c2, c3), q, gens, n_max) for i in range(3)]
            # matching B: auxiliary scaled by the same orbit factor
            facs = [1, omega, w2]
            sB = [sigma_profile(shells,
                                (cc[i], (facs[i] * c2) % q, (facs[i] * c3) % q),
                                q, gens, n_max) for i in range(3)]
            for tag, S in (('A', sA), ('B', sB)):
                L = min(len(x) for x in S)
                lo, hi = n0, min(n1, L - 1)
                if hi < lo:
                    continue
                stk = np.stack([x[lo:hi + 1] for x in S])
                dev = float(np.max(np.abs(stk - stk[0])))
                if dev > maxdev[tag]:
                    maxdev[tag] = dev
                    argmax[tag] = (c, c2, c3)
    return maxdev, argmax, nblocks, len(reps), depth, sum(len(s) for s in shells)


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--q', type=int, default=61)
    pa.add_argument('--n0', type=int, default=2)
    pa.add_argument('--n1', type=int, default=7)
    pa.add_argument('--n-orbits', type=int, default=None)
    pa.add_argument('--aux-pairs', type=int, default=3)
    pa.add_argument('--exhaustive', action='store_true')
    pa.add_argument('--seed', type=int, default=0)
    args = pa.parse_args()
    out = run(args.q, args.n0, args.n1, n_orbits=args.n_orbits,
              aux_pairs=args.aux_pairs, seed=args.seed, exhaustive=args.exhaustive)
    if out is None:
        sys.exit(0)
    md, am, nb, nr, depth, nodes = out
    print(f"q={args.q}  window [{args.n0},{args.n1}]  depth-capped BFS to {depth} "
          f"({nodes} nodes)  orbits={nr}  base blocks={nb}")
    print(f"  matching A (same aux):   max|sigma_i - sigma_j| = {md['A']:.3e}  at {am['A']}")
    print(f"  matching B (scaled aux): max|sigma_i - sigma_j| = {md['B']:.3e}  at {am['B']}")