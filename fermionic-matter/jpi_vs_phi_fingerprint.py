#!/usr/bin/env python3
"""
jpi_vs_phi_fingerprint.py

Ground-locking check, in the A_c + B_c x conventions of the amplitude note, of the
induced action on the O12 fingerprint of:

  - the BI parity / conjugation  J_Pi  (= H K, with K = sigma_x o conj, H = sigma_z grading),
  - the residual frontier automorphism  phi: (a,b,z) -> (-a, b, -z).

Fingerprint (single character, 3 accumulated steps ep1,ep2,ep3 on a base element):
  B_c = sum_i c_i b(ep_i)                  (frequency  -> radial / capacity)
  A_c = sum_i c_i ( z(ep_i) - b(ep_i) a(ep_i) )   (central phase -> angular)

Claims to verify EXACTLY (mod q), over random g, triples, blocks, and several q:
  (J)  conjugation c -> q-c :  A_c -> -A_c ,  B_c -> -B_c     => J_Pi : (A,B) -> (-A,-B)
  (P)  phi applied per accumulated element:  A_c -> -A_c , B_c -> +B_c
                                                              => phi  : (A,B) -> (-A,+B)
Hence  J_Pi != phi, and phi = R_b o J_Pi  with  R_b: (A,B) -> (A,-B)  (frequency reflection).
H = sigma_z is a sector grading (+/-1), it does NOT move (A,B); the only label-changing
factor in J_Pi = H K is K. This is asserted from Q14 (H=sigma_z) and is consistent with (J).

No amplitude, no 1/10, no epsilon appear here: this only fixes the two involutions.
"""

import numpy as np

Q_LIST = [31, 61, 101, 151]
N_SAMPLES = 4000
RNG_SEED = 20260607


def build_generators(q):
    return [(1, 0, 0), (0, 1, 0), ((-1) % q, 0, 0), (0, (-1) % q, 0)]


def hmul(g, s, q):
    a, b, z = g
    sa, sb, sz = s
    return ((a + sa) % q, (b + sb) % q, (z + sz + a * sb) % q)


def phi(g, q):
    a, b, z = g
    return ((-a) % q, b % q, (-z) % q)


def accumulate(g, triple, q):
    """Return the three accumulated elements ep1, ep2, ep3."""
    e1 = hmul(g, triple[0], q)
    e2 = hmul(e1, triple[1], q)
    e3 = hmul(e2, triple[2], q)
    return e1, e2, e3


def A_of(eps, block, q):
    c = block
    return sum(c[i] * ((eps[i][2] - eps[i][1] * eps[i][0]) % q) for i in range(3)) % q


def B_of(eps, block, q):
    c = block
    return sum(c[i] * eps[i][1] for i in range(3)) % q


def run_q(q):
    rng = np.random.default_rng(RNG_SEED + q)
    gens = build_generators(q)
    maxdev_J_A = maxdev_J_B = maxdev_P_A = maxdev_P_B = 0
    differ_B = False
    for _ in range(N_SAMPLES):
        g = (int(rng.integers(0, q)), int(rng.integers(0, q)), int(rng.integers(0, q)))
        triple = tuple(gens[int(rng.integers(0, 4))] for _ in range(3))
        block = tuple(int(rng.integers(1, q)) for _ in range(3))
        conj = tuple((q - c) % q for c in block)

        eps = accumulate(g, triple, q)
        A = A_of(eps, block, q)
        B = B_of(eps, block, q)

        # (J) conjugation c -> q-c
        A_conj = A_of(eps, conj, q)
        B_conj = B_of(eps, conj, q)
        maxdev_J_A = max(maxdev_J_A, (A_conj - (-A)) % q if (A_conj - (-A)) % q <= q // 2
                         else q - (A_conj - (-A)) % q)
        maxdev_J_B = max(maxdev_J_B, (B_conj - (-B)) % q if (B_conj - (-B)) % q <= q // 2
                         else q - (B_conj - (-B)) % q)

        # (P) phi applied per accumulated element (phi is an automorphism)
        eps_phi = tuple(phi(e, q) for e in eps)
        A_phi = A_of(eps_phi, block, q)
        B_phi = B_of(eps_phi, block, q)
        maxdev_P_A = max(maxdev_P_A, (A_phi - (-A)) % q if (A_phi - (-A)) % q <= q // 2
                         else q - (A_phi - (-A)) % q)
        maxdev_P_B = max(maxdev_P_B, (B_phi - (+B)) % q if (B_phi - (+B)) % q <= q // 2
                         else q - (B_phi - (+B)) % q)

        # phi vs J_Pi must differ in B whenever B != 0
        if B % q != 0 and (B_phi - B_conj) % q != 0:
            differ_B = True

    return {
        "q": q,
        "J_A_dev": maxdev_J_A, "J_B_dev": maxdev_J_B,
        "P_A_dev": maxdev_P_A, "P_B_dev": maxdev_P_B,
        "phi_neq_Jpi_in_B": differ_B,
    }


def main():
    print("=" * 72)
    print("J_Pi (BI conjugation) vs phi on the O12 fingerprint (A_c, B_c)  -- exact mod q")
    print("=" * 72)
    all_ok = True
    for q in Q_LIST:
        r = run_q(q)
        ok = (r["J_A_dev"] == 0 and r["J_B_dev"] == 0
              and r["P_A_dev"] == 0 and r["P_B_dev"] == 0 and r["phi_neq_Jpi_in_B"])
        all_ok &= ok
        print(f"q = {q}")
        print(f"  (J) conjugation c->q-c : A->-A dev={r['J_A_dev']}, B->-B dev={r['J_B_dev']}"
              f"   => J_Pi:(A,B)->(-A,-B)")
        print(f"  (P) phi (a,b,z)->(-a,b,-z): A->-A dev={r['P_A_dev']}, B->+B dev={r['P_B_dev']}"
              f"  => phi:(A,B)->(-A,+B)")
        print(f"  phi != J_Pi (differ in B): {r['phi_neq_Jpi_in_B']}")
    print("=" * 72)
    if all_ok:
        print("VERDICT: J_Pi:(A,B)->(-A,-B)  !=  phi:(A,B)->(-A,+B).")
        print("         phi = R_b o J_Pi,  R_b:(A,B)->(A,-B)  (frequency / radial reflection).")
        print("         H = sigma_z is a sector grading; it does not restore B.")
        print("         => the residual obstruction phi is the BI parity corrected by a")
        print("            frequency reflection, NOT the BI parity itself. L3a is false.")
    else:
        print("VERDICT: identities FAILED -- re-examine conventions.")
    print("=" * 72)


if __name__ == "__main__":
    main()
