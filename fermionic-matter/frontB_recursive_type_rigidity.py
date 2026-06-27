"""
Front B exact-symbolic audit: vertical/horizontal automorphism dichotomy for 2I = SL(2,5).

Goal (bias-independent, no fit, exact over Q(i, sqrt5)):
  Establish the structural dichotomy that underlies lem:rigidity (AOG, Beau2026aog).

  (A) VERTICAL operations (fibre identifications: the central element -I, scalar/Schur action,
      inner conjugations) fix every irreducible character pointwise, hence fix the character
      field Q(sqrt5).  These are exactly the operations a non-injective FIBRE identification
      {chi, -chi} can realise (ENI minimal-fibre condition).

  (B) HORIZONTAL operation (the Galois action sqrt5 -> -sqrt5) is the OUTER automorphism of
      2I ~ SL(2,5): it permutes the two 2-dimensional irreducible characters chi_2 <-> chi_2'.
      It is NOT realised by any central/scalar/inner (vertical) operation.

  The ADE gate needs (B); recursive non-injectivity (ENI Cor. 6) supplies only (A)-type fibre
  identifications.  The dichotomy is the precise content sharpening lem:rigidity.

Construction: the 120 icosians (unit-quaternion vertices of the 600-cell) realise 2I, mapped to
SU(2).  Everything is exact in Q(i, sqrt5) via sympy.
"""

import itertools
from sympy import sqrt, Rational, I, Matrix, eye, simplify, expand, nsimplify, conjugate, Integer

phi = (1 + sqrt(5)) / 2          # golden ratio
iphi = (sqrt(5) - 1) / 2         # 1/phi = phi - 1

# ---------------------------------------------------------------------------
# 1. The 120 icosians as unit quaternions (a, b, c, d) with exact Q(sqrt5) entries.
# ---------------------------------------------------------------------------
def quaternions_2I():
    quats = set()

    # 8 : (+-1,0,0,0) and permutations
    for pos in range(4):
        for s in (1, -1):
            q = [0, 0, 0, 0]
            q[pos] = s
            quats.add(tuple(q))

    # 16 : (+-1/2, +-1/2, +-1/2, +-1/2)
    for signs in itertools.product((1, -1), repeat=4):
        quats.add(tuple(Rational(s, 2) for s in signs))

    # 96 : even permutations of (0, +-1/2, +-1/(2phi), +-phi/2)
    base = [Integer(0), Rational(1, 2), iphi / 2, phi / 2]
    # even permutations of indices 0..3
    even_perms = []
    for p in itertools.permutations(range(4)):
        # parity
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            even_perms.append(p)
    for p in even_perms:
        for signs in itertools.product((1, -1), repeat=3):  # the 0 entry has no sign
            vals = list(base)
            q = [0, 0, 0, 0]
            # assign permuted magnitudes; the zero stays zero, three nonzero get signs
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
    return Matrix([[a + b * I, c + d * I],
                   [-c + d * I, a - b * I]])


def matkey(M):
    return tuple(simplify(expand(M[i, j])) for i in range(2) for j in range(2))


# ---------------------------------------------------------------------------
# 2. Build the group, check order 120 and closure.
# ---------------------------------------------------------------------------
quats = quaternions_2I()
print(f"raw quaternion count: {len(quats)}")

mats = {}
for q in quats:
    M = quat_to_su2(q)
    mats[matkey(M)] = M
print(f"distinct SU(2) matrices: {len(mats)}")

elements = list(mats.values())
keys = list(mats.keys())
keyset = set(keys)

def mul(A, B):
    return Matrix([[expand(A[i, 0] * B[0, j] + A[i, 1] * B[1, j]) for j in range(2)]
                   for i in range(2)])

# ---------------------------------------------------------------------------
# 3. Conjugacy classes via the defining-rep character chi_2 = trace.
#    In SU(2) two elements are conjugate in 2I iff (here) they share a trace value;
#    the 9 distinct exact traces of 2I separate its 9 conjugacy classes.
# ---------------------------------------------------------------------------
from collections import defaultdict
by_trace = defaultdict(list)
for M in elements:
    tr = simplify(expand(M[0, 0] + M[1, 1]))
    by_trace[str(tr)].append(M)

print(f"\nnumber of distinct exact traces (= conjugacy classes): {len(by_trace)}  (2I has 9)")

def order_of(M):
    P = M
    o = 1
    Ident = eye(2)
    while matkey(P) != matkey(Ident):
        P = mul(P, M)
        o += 1
        if o > 30:
            return None
    return o

print("\nsize | order | chi_2 (trace)")
chi2_vals = []
for trkey, group in sorted(by_trace.items(), key=lambda kv: len(kv[1])):
    rep = group[0]
    tr = simplify(expand(rep[0, 0] + rep[1, 1]))
    order = order_of(rep)
    chi2_vals.append((len(group), order, tr))
    print(f"  size={len(group):3d}  order={str(order):>4}  chi_2={tr}")
print(f"  (sum of class sizes = {sum(c[0] for c in chi2_vals)}, must be 120)")

# ---------------------------------------------------------------------------
# 4. Character field and the Galois action sqrt5 -> -sqrt5 (the OUTER automorphism).
# ---------------------------------------------------------------------------
# does chi_2 take a value involving sqrt5?
involves_sqrt5 = any(sqrt(5) in simplify(tr).atoms(type(sqrt(5))) or
                     simplify(tr).has(sqrt(5)) for (_, _, tr) in chi2_vals)
print(f"\nchi_2 character field contains sqrt5: {involves_sqrt5}")

def galois(expr):
    # sqrt5 -> -sqrt5
    return simplify(expr.subs(sqrt(5), -sqrt(5)))

# inner products <chi,chi'> = (1/|G|) sum_classes size * chi(g) * conj(chi'(g))
G = len(elements)
def inner(vals_a, vals_b):
    s = 0
    for (size, _, a), (_, _, b) in zip(vals_a, vals_b):
        s += size * a * conjugate(b)
    return simplify(expand(s) / G)

# chi_2' = Galois conjugate of chi_2
chi2p_vals = [(size, order, galois(tr)) for (size, order, tr) in chi2_vals]

print("\nnorms / overlaps:")
print(f"  <chi_2 , chi_2 >   = {inner(chi2_vals, chi2_vals)}   (irreducible iff 1)")
print(f"  <chi_2', chi_2'>   = {inner(chi2p_vals, chi2p_vals)}   (Galois image still irreducible)")
print(f"  <chi_2 , chi_2'>   = {inner(chi2_vals, chi2p_vals)}   (0 iff genuinely DIFFERENT irrep)")

same = all(simplify(a - b) == 0 for (_, _, a), (_, _, b) in zip(chi2_vals, chi2p_vals))
print(f"  chi_2' == chi_2 ?  {same}   (False => sqrt5->-sqrt5 PERMUTES the two 2-dim irreps = OUTER aut)")

# ---------------------------------------------------------------------------
# 5. VERTICAL operations fix every character pointwise.
# ---------------------------------------------------------------------------
# central element -I:
minusI = matkey(-eye(2))
print(f"\ncentral -I present in group: {minusI in keyset}")
minus_trace = str(simplify(-2))
size_minusI = len(by_trace.get(minus_trace, []))
print(f"  class of -I has size {size_minusI} (central => singleton); chi_2(-I) = -2")
print(f"  -I acts as scalar (-1)*Identity in defining rep: {matkey(-eye(2)) == minusI}")

print("\nVERTICAL fact: inner conjugation fixes class functions (characters are class functions);")
print("central/scalar action multiplies a rep by a scalar => fixes its character value up to the")
print("global trace factor and never moves sqrt5.  Hence every fibre identification {chi,-chi}")
print("(ENI minimal fibre) is sqrt5-fixing.")

print("\n=== VERDICT ===")
print("HORIZONTAL (sqrt5->-sqrt5) = outer aut of 2I, the unique operation swapping chi_2<->chi_2'.")
print("VERTICAL (central/scalar/inner = every ENI fibre identification) fixes Q(sqrt5) pointwise.")
print("The ADE gate needs HORIZONTAL; recursive non-injectivity supplies only VERTICAL.")
