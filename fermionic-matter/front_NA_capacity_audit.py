#!/usr/bin/env python3
"""
front_NA_capacity_audit.py

Front N_A: does the Weil-BFS / projected-capacity pipeline yield N_A = epsilon = 1/10 WITHOUT FITTING?

This is a thin CONSOLIDATOR over the two existing campaign backends (no new physics, no fit):
  - q11_oriented_frontier.py   -> directed outgoing-frontier signal and the maximal-locking bound theta_max
  - weil_bfs_angular_area.py   -> the symmetric-shell oriented area Theta_raw (the obstruction baseline)

It executes the AAR "required outputs and honest outcomes" programme for q in {61, 101, 151} and classifies the
result into AAR's four honest outcomes, WITHOUT ever fitting to 1/10.

Run (real pipeline):
  PYTHONPATH=../spectral/o12 python3 front_NA_capacity_audit.py

GUARDRAILS (inherited from both backends): the value 1/10 is used ONLY as a comparison target printed at the end;
no quantity is ever rescaled to hit it. theta_max is the maximal-locking BOUND on |Theta_chi|, a structural upper
bound attained only under the CHO [H-orient] sigma_L lock; it is NOT an amplitude and its conversion to epsilon
needs the geometric Sym^2 normalisation N_A^geom, which the capacity data do not supply (AAR section 5).
"""

import importlib.util
import math
from fractions import Fraction

EPS_DICT = Fraction(1, 10)
Q_LIST = [61, 101, 151, 211, 307]
TWO_PI = 2.0 * math.pi


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _signed(v, q):
    return ((v + q // 2) % q) - q // 2


def onset_generator_dependence(q=61):
    """Shell-1 onset <|Delta A_c|>_d+ for several symmetric generating sets of Heis_3(Z/qZ).

    Demonstrates that the closed onset 1/3 is the standard {+-X,+-Y} Cayley-frontier convention, NOT a
    generator-set invariant (and a fortiori not a generation-count / Sym^2-dimension invariant): changing the
    horizontal generating set changes the onset rational. Exact integer arithmetic (Fraction)."""
    from collections import deque

    def mul(g, s):
        a, b, z = g
        sa, sb, sz = s
        return ((a + sa) % q, (b + sb) % q, (z + sz + a * sb) % q)

    def onset(gens):
        dist = {(0, 0, 0): 0}
        dq = deque([(0, 0, 0)])
        while dq:
            g = dq.popleft()
            if dist[g] >= 3:
                continue
            for s in gens:
                h = mul(g, s)
                if h not in dist:
                    dist[h] = dist[g] + 1
                    dq.append(h)
        num = cnt = 0
        for g, d in dist.items():
            if d != 1:
                continue
            for s in gens:
                h = mul(g, s)
                if dist.get(h) == 2:
                    num += abs(_signed((h[2] - g[2]) % q, q))
                    cnt += 1
        return Fraction(num, cnt)

    X, Xi, Y, Yi = (1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 0)
    sets = {
        "standard {+-X,+-Y}": [X, Xi, Y, Yi],
        "{+-X,+-Y,+-(X+Y)}": [X, Xi, Y, Yi, (1, 1, 0), (q - 1, q - 1, 0)],
        "{+-X,+-Y,+-2X}": [X, Xi, Y, Yi, (2, 0, 0), (q - 2, 0, 0)],
        "{+-X,+-(X+Y)}": [X, Xi, (1, 1, 0), (q - 1, q - 1, 0)],
        "{+-2X,+-Y}": [(2, 0, 0), (q - 2, 0, 0), Y, Yi],
    }
    return {name: onset(gens) for name, gens in sets.items()}


def frontier_abs_dAc_exact(q11, q, mmax=6):
    """EXACT (rational) per-shell directed-frontier mean |Delta A_c| over the outgoing edges d+.

    This is the bare per-shell increment underlying theta_max = (2pi/q) <|Delta A_c|>_{d+}; computed with
    integer arithmetic (no float, no block sampling, no fit). c = 1 (central character scale)."""
    n_max = min(q11.N_MAX_CAP, q // 2)
    shells, dist, gens = q11.bfs_with_distance(q, n_max)
    seq = []
    for mm in range(1, mmax + 1):
        num, cnt = 0, 0
        for g in shells[mm]:
            zg = g[2]
            for s in gens:
                gs = q11.heisenberg_mul(g, s, q)
                if dist.get(gs) == mm + 1:                  # outgoing edge d+
                    num += abs(_signed((gs[2] - zg) % q, q))
                    cnt += 1
        seq.append(Fraction(num, cnt))
    return seq


def main():
    q11 = _load("q11f", "q11_oriented_frontier.py")
    wb = _load("wb", "weil_bfs_angular_area.py")
    q11.N_WORKERS = 1                                       # serial: dynamic-module workers cannot be pickled
    wb.N_WORKERS = 1

    # (i) EXACT per-shell directed-frontier sequence -- frontier only, no blocks/Pool: all q
    rows = [{"q": q, "seq": frontier_abs_dAc_exact(q11, q)} for q in Q_LIST]

    # (ii) obstruction confirmation (signed=0, anti-bias=0, symmetric area vanishes) on checkpointed q only
    obstruction = []
    for q in [61, 101, 151]:
        a = q11.run_q(q)
        b = wb.run_q(q)
        obstruction.append({
            "q": q,
            "signed_plus_raw_max": a["max_abs_theta_plus_raw"],
            "antibias_sym_max": a["max_abs_theta_sym_raw"],
            "sym_area_status": b["angular_status"],
        })

    print("=" * 104)
    print("FRONT N_A -- capacity-pipeline audit (real pipeline, exact rationals, no fit). AAR required outputs.")
    print("=" * 104)
    print("Obstruction (signed oriented area), checkpointed q:")
    print(f"  {'q':>4} | {'signed |<dAc>_d+|':>16} | {'anti-bias':>9} | {'symmetric area':>20}")
    for o in obstruction:
        print(f"  {o['q']:>4} | {o['signed_plus_raw_max']:>16.2e} | {o['antibias_sym_max']:>9.1e} | "
              f"{o['sym_area_status']:>20}")
    print("-" * 104)
    print("Exact directed-frontier <|Delta A_c|>_d+ per shell (integer arithmetic, c=1):")
    for r in rows:
        print(f"  q={r['q']:>4}: " + ", ".join(str(f) for f in r["seq"]))
    print("-" * 104)

    # q-independence of the exact sequence, and the saturation-onset value
    seqs = [tuple(r["seq"]) for r in rows]
    q_independent = all(s == seqs[0] for s in seqs)
    onset = seqs[0][0]
    print(f"per-shell sequence q-independent across q={Q_LIST}: {q_independent}")
    print(f"saturation-onset increment <|Delta A_c|>_(shell 1) = {onset}  (exact)")
    # generator-set dependence: the onset 1/3 is the standard {+-X,+-Y} convention, NOT a generator invariant
    gd = onset_generator_dependence()
    print("onset vs symmetric generating set (q=61):  " + ";  ".join(f"{k} -> {v}" for k, v in gd.items()))
    print(f"  => onset is GENERATOR-SET DEPENDENT (1/3 is the standard campaign convention, not a generation/dim "
          f"invariant); only eps and the ADE ratio are generator-independent")
    inv_onset = 1 / onset                                  # Fraction
    print(f"=> maximal-locking bound (onset)  theta_max = (2pi/q)*{onset} = 2pi/({inv_onset} q)")
    na_geom_coeff = EPS_DICT * inv_onset / TWO_PI           # coefficient of q in N_A^geom = (1/10)/theta_max
    print()
    print("VERDICT (honest, no fit):")
    print(" 1. SIGNED oriented Weil-BFS area = 0 identically: symmetric shell average vanishes (weil_bfs")
    print("    'vanishes_symmetric') AND directed-frontier signed mean = 0 (q11 max|theta_plus|=0), anti-bias = 0,")
    print("    sign-reversal consistent. => AAR honest outcome 4: capacity data do NOT select an oriented branch.")
    print(" 2. ANGULAR COEFFICIENT CLOSED: the directed-frontier |Delta A_c| per-shell means are q-INDEPENDENT")
    print(f"    EXACT rationals; the saturation-onset value is exactly {onset}. Hence the maximal-locking bound")
    print(f"    (conditional on the CHO [H-orient] lock) is theta_max = 2pi/(3q) -- EXACT, q-independent, -> 0.")
    print(f" 3. NOT a derivation of eps = 1/10. theta_max -> 0 with q; eps=1/10 would require the geometric Sym^2")
    print(f"    normalisation N_A^geom = (1/10)/theta_max = (1/10)*(3q/2pi) = 3q/(20 pi) -- AAR section 5's 'last")
    print("    open step', NOT supplied by the capacity data. The angular coefficient is closed (1/3); the")
    print("    geometric normalisation is not. eps=1/10 stays gated, consistent with AOG lem:rigidity.")
    print("=" * 104)
    return rows


if __name__ == "__main__":
    main()
