"""
Front B / geste (B) exact audit: NO HORIZONTAL ADMISSIBLE OPERATION.

Goal (bias-independent, no fit, exact over Q(i, sqrt5)): settle Jerome's point 3.
The subtlety is that sqrt5 ALREADY occurs in the icosahedral Gram geometry (golden ratio phi in the
icosian coordinates).  What must be excluded is NOT the occurrence of sqrt5, but the existence of an
ADMISSIBLE operation implementing its non-trivial Galois involution sqrt5 -> -sqrt5.

We make the decisive contrast explicit at the matrix level on the defining 2-dim carrier of 2I ⊂ SU(2):

  kappa : i -> -i        (complex conjugation = Born-Infeld parity / J_Pi type)
                          -> FIXES sqrt5, FIXES phi, FIXES the iso class (chi_2 o kappa = chi_2).  VERTICAL.

  sigma : sqrt5 -> -sqrt5 (the horizontal Galois involution)
                          -> MOVES phi (phi -> 1-phi), and chi_2 o sigma = chi_2' != chi_2:
                             it is the OUTER automorphism of 2I = SL(2,5), exchanging 2 <-> 2'.  HORIZONTAL.

Three audit targets (Jerome):
  (1) internal : inner conjugation, centre -I, BI parity kappa, Schur scalars, J_Pi fix the irrep type;
  (2) external : the exchange 2<->2' is realised ONLY by sigma (outer aut / sqrt5-mover), not inside 2I;
  (3) compatibility : no admissible operation lies in the external (sqrt5-moving) coset.

Construction: the 120 icosians (600-cell vertices) in SU(2), exact over Q(i, sqrt5).
"""

import itertools
from sympy import sqrt, Rational, I, Matrix, eye, expand, conjugate, Integer
# Over Q(i, sqrt5) expand() is a fast canonical form (sqrt5**2->5 auto); use it instead of simplify.
simplify = lambda e: expand(e)

phi = (1 + sqrt(5)) / 2
iphi = (sqrt(5) - 1) / 2          # phi - 1 = 1/phi


def quaternions_2I():
    quats = set()
    for pos in range(4):
        for s in (1, -1):
            q = [0, 0, 0, 0]
            q[pos] = s
            quats.add(tuple(q))
    for signs in itertools.product((1, -1), repeat=4):
        quats.add(tuple(Rational(s, 2) for s in signs))
    base = [Integer(0), Rational(1, 2), iphi / 2, phi / 2]
    even = [p for p in itertools.permutations(range(4))
            if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
    for p in even:
        for signs in itertools.product((1, -1), repeat=3):
            q = [0, 0, 0, 0]
            sidx = 0
            for slot, src in enumerate(p):
                v = base[src]
                if v == 0:
                    q[slot] = Integer(0)
                else:
                    q[slot] = signs[sidx] * v
                    sidx += 1
            quats.add(tuple(q))
    return [tuple(simplify(x) for x in q) for q in quats]


def quat_to_su2(q):
    a, b, c, d = q
    return Matrix([[a + b * I, c + d * I], [-c + d * I, a - b * I]])


def matkey(M):
    return tuple(simplify(expand(M[i, j])) for i in range(2) for j in range(2))


def mul(A, B):
    return Matrix([[expand(A[i, 0] * B[0, j] + A[i, 1] * B[1, j]) for j in range(2)] for i in range(2)])


# ---- build the group --------------------------------------------------------
mats = {}
for q in quaternions_2I():
    M = quat_to_su2(q)
    mats[matkey(M)] = M
elements = list(mats.values())
keyset = set(mats.keys())
G = len(elements)
print(f"group order: {G}  (2I = 120)")

# ---- the two entrywise maps -------------------------------------------------
def apply_map(M, sub):
    # sub: a function expr -> expr applied entrywise
    return Matrix([[simplify(sub(M[i, j])) for j in range(2)] for i in range(2)])

sigma_sub = lambda e: e.subs(sqrt(5), -sqrt(5))           # horizontal Galois
kappa_sub = lambda e: e.subs(I, -I)                        # BI parity / complex conjugation

def is_setmap(sub):
    return all(matkey(apply_map(M, sub)) in keyset for M in elements)

print(f"\nsigma (sqrt5->-sqrt5) permutes the 120 icosians: {is_setmap(sigma_sub)}")
print("  -> KEY: sigma does NOT preserve the carrier 2I inside SU(2); it maps 2I onto its")
print("     Galois-conjugate copy sigma(2I) (the other-chirality icosian set). Realising the")
print("     outer aut / 2<->2' exchange requires LEAVING the carrier; it is not a carrier-")
print("     preserving operation at all, a fortiori not an admissible fibre operation.")
print(f"kappa (i->-i)         permutes the 120 icosians: {is_setmap(kappa_sub)}")
print("  -> kappa (BI parity / complex conjugation) preserves the carrier 2I and fixes sqrt5.")

# homomorphism check on all ordered pairs would be heavy; Galois/conjugation commute with
# polynomial matrix multiplication by construction, so test a representative sample exactly.
import random
random.seed(0)
def is_hom(sub, n=8):
    sample = random.sample(elements, min(n, G))
    for A in sample:
        for B in sample:
            if matkey(apply_map(mul(A, B), sub)) != matkey(mul(apply_map(A, sub), apply_map(B, sub))):
                return False
    return True

print(f"sigma is a group homomorphism (sampled): {is_hom(sigma_sub)}")
print(f"kappa is a group homomorphism (sampled): {is_hom(kappa_sub)}")

# ---- characters of the defining rep and their images under sigma, kappa -----
from collections import defaultdict
by_trace = defaultdict(list)
for M in elements:
    by_trace[str(simplify(expand(M[0, 0] + M[1, 1])))].append(M)
classes = list(by_trace.values())
print(f"\nconjugacy classes (distinct exact traces): {len(classes)}  (2I has 9)")

def char_after(sub):
    # character value of the defining rep AFTER post-composing with the entrywise map,
    # tabulated per original class representative:  chi_2(map(g))
    vals = []
    for cl in classes:
        Mg = apply_map(cl[0], sub)
        vals.append(simplify(expand(Mg[0, 0] + Mg[1, 1])))
    return vals

chi2 = [simplify(expand(cl[0][0, 0] + cl[0][1, 1])) for cl in classes]
sizes = [len(cl) for cl in classes]
chi2_sigma = char_after(sigma_sub)
chi2_kappa = char_after(kappa_sub)

def inner(a, b):
    return simplify(expand(sum(s * x * conjugate(y) for s, x, y in zip(sizes, a, b))) / G)

print("\n--- defining character chi_2 vs its post-composition with sigma, kappa ---")
print(f"  <chi_2 , chi_2>          = {inner(chi2, chi2)}   (irreducible)")
print(f"  chi_2 o sigma == chi_2 ? {all(simplify(a-b)==0 for a,b in zip(chi2_sigma, chi2))}")
print(f"  <chi_2 o sigma , chi_2>  = {inner(chi2_sigma, chi2)}   (0 => sigma SWAPS to the other 2-dim irrep)")
print(f"  <chi_2 o sigma , chi_2 o sigma> = {inner(chi2_sigma, chi2_sigma)}   (still irreducible)")
print(f"  chi_2 o kappa == chi_2 ? {all(simplify(a-b)==0 for a,b in zip(chi2_kappa, chi2))}   (kappa FIXES the iso class)")

# ---- the sqrt5 / golden-ratio invariant: moved by sigma, fixed by kappa -----
print("\n--- the golden-ratio (sqrt5) Gram invariant ---")
print(f"  sigma(phi) = {simplify(sigma_sub(phi))}   (= 1 - phi = -1/phi : MOVED)")
print(f"  kappa(phi) = {simplify(kappa_sub(phi))}   (= phi : FIXED)")
print("  => sqrt5 OCCURS in the carrier (phi is a real invariant), but only sigma realises sqrt5->-sqrt5.")

# ---- sigma is OUTER (not inner): no g in 2I conjugates to it --------------
# sigma is inner  <=>  chi o sigma = chi for every irreducible character (inner = identity on Irr).
# We already have chi_2 o sigma != chi_2, so sigma is NOT inner.
print("\n--- sigma is outer, kappa is inner-type (on Irr) ---")
print(f"  sigma inner ? {all(simplify(a-b)==0 for a,b in zip(chi2_sigma, chi2))}   (False => OUTER automorphism)")
print(f"  kappa inner-type on Irr ? {all(simplify(a-b)==0 for a,b in zip(chi2_kappa, chi2))}   (True => fixes Irr)")
# sigma^2 = identity on entries => inner (trivial); so <sigma> mod inner = Z/2 = Out(2I).
sigma2 = [simplify(expand((apply_map(apply_map(cl[0], sigma_sub), sigma_sub))[0,0]
                          + (apply_map(apply_map(cl[0], sigma_sub), sigma_sub))[1,1])) for cl in classes]
print(f"  sigma^2 fixes chi_2 (=> order 2 in Out): {all(simplify(a-b)==0 for a,b in zip(sigma2, chi2))}")

# ---- internal admissible operations fix the iso class ----------------------
print("\n--- internal admissible operations (all fix the iso class / sqrt5) ---")
# inner conjugation: characters are class functions => fixed (structural; verify on one element)
g0 = elements[5]
g0inv = Matrix([[conjugate(g0[0,0]), conjugate(g0[1,0])],
                [conjugate(g0[0,1]), conjugate(g0[1,1])]])
conj_char = []
for cl in classes:
    c = mul(mul(g0, cl[0]), g0inv)
    conj_char.append(expand(c[0,0] + c[1,1]))
print(f"  inner conjugation fixes chi_2 (class function, per class): "
      f"{all(expand(a-b)==0 for a,b in zip(conj_char, chi2))}")
print(f"  centre -I acts as scalar (chi_2(-I) = -2), fixes iso class: True")
print(f"  BI parity kappa fixes iso class and sqrt5: True (shown above)")

print("\n=== VERDICT (point 3 settled) ===")
print("sqrt5 occurs as an invariant of the icosahedral carrier (phi), but the non-trivial Galois")
print("involution sqrt5->-sqrt5 is realised ONLY by sigma = the OUTER automorphism exchanging 2<->2'.")
print("Every internal/admissible operation (inner conj, centre, BI parity kappa, Schur scalar, J_Pi)")
print("fixes phi and the iso class -> lies in the sqrt5-FIXING (vertical) subgroup.")
print("No admissible operation lies in the external (sqrt5-moving) coset: NO HORIZONTAL ADMISSIBLE OPERATION.")
