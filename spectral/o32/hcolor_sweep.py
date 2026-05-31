"""
hcolor_sweep.py
Run the deterministic [H-color]_pointwise check across colour test primes
(q == 1 mod 3) with a live progress/ETA bar per prime. Run it like any other
script (e.g. in PyCharm); hcolor_exact_check.py must be in the same directory.

  matching A (c1 scaled alone)     -> sigma differs   (expected, nonzero)
  matching B (whole block scaled)  -> sigma identical  (the [H-color] symmetry)
"""
import time
from hcolor_exact_check import run

# Colour TEST primes only (q == 1 mod 3). 401 is a CONTROL prime (2 mod 3): skip.
PRIMES = [61, 151, 211, 307, 313, 331, 337, 349, 367, 373]
N0, N1 = 3, 13
N_ORBITS, AUX_PAIRS = 4, 3

print(f"{'q':>5}  {'matching A':>12}  {'matching B':>12}  {'verdict':>8}  "
      f"{'blocks':>7}  {'time(s)':>8}")
print("-" * 64)

t_start = time.perf_counter()
rows = []
for i, q in enumerate(PRIMES):
    t0 = time.perf_counter()
    # progress=True shows a per-prime tqdm bar with live ETA over the base blocks
    out = run(q, N0, N1, n_orbits=N_ORBITS, aux_pairs=AUX_PAIRS, progress=True)
    dt = time.perf_counter() - t0
    if out is None:                       # control prime, no colour triplets
        print(f"{q:>5}  {'(control: q != 1 mod 3, no triplets)':>40}")
        continue
    md, am, nb, nr, depth, nodes = out
    verdict = "exact" if md['B'] < 1e-12 else "BREAK"
    print(f"{q:>5}  {md['A']:>12.3e}  {md['B']:>12.3e}  {verdict:>8}  "
          f"{nb:>7}  {dt:>8.1f}")
    rows.append((q, md['B']))
    # crude overall ETA: remaining primes scaled by q (cost grows with q)
    done_q = sum(p for p in PRIMES[:i + 1])
    rem_q = sum(p for p in PRIMES[i + 1:])
    elapsed = time.perf_counter() - t_start
    if rem_q and done_q:
        eta = elapsed * rem_q / done_q
        print(f"        ... elapsed {elapsed:5.0f}s, est. remaining ~{eta:5.0f}s")

print("-" * 64)
allB = all(b < 1e-12 for _, b in rows)
print(f"matching B == 0 at all {len(rows)} test primes: "
      f"{'YES -> [H-color]_pointwise confirmed' if allB else 'NO -> inspect'}")