"""
spectral_admissibility.py
=========================
Spectral admissibility and non-abelian mode resilience on the
Cayley graphs of Q8 and 2I (binary icosahedral group).

Reproduces all results from SpectralNote – Steps 1-5
(Cosmochrony programme, 2026).

Contents
--------
Part 1  Q8 Cayley graph
  1a  Build the group and Cayley graph
  1b  Peter-Weyl eigenmodes (exact basis)
  1c  Bounded-flux admissibility envelope
  1d  DBI shape factors

Part 2  2I Cayley graph
  2a  Enumerate all 120 elements as unit quaternions
  2b  Conjugacy classes and element orders
  2c  Character table (spin-type irreps via Sym^k)
  2d  Spectral admissibility for multiple generating sets
  2e  Golden ratio identity: algebraic origin of the hierarchy
  2f  Comparison Q8 vs 2I

Part 3  DBI dynamics simulation on Q8
  3a  Symplectic (leapfrog) integrator
  3b  Sector amplitude tracking in the Peter-Weyl basis
  3c  Saturation and resilience of the quaternionic sector

Usage
-----
    python3 spectral_admissibility.py

Outputs (all in current directory):
    figures/fig1_Q8_admissibility.{png,pdf}
    figures/fig2_Q8_DBI_dynamics.{png,pdf}
    figures/fig3_hierarchy_Q8_vs_2I.{png,pdf}

Requirements: numpy, scipy (optional), matplotlib (optional)
"""

import numpy as np
from itertools import product, permutations
from collections import Counter, defaultdict
import os

# ---------------------------------------------------------------
# Quaternion arithmetic
# ---------------------------------------------------------------

def qmul(p, q):
    """Hamilton product: p * q, each = (a,b,c,d) = a+bi+cj+dk."""
    a1,b1,c1,d1 = p[0],p[1],p[2],p[3]
    a2,b2,c2,d2 = q[0],q[1],q[2],q[3]
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2,
    ])

def qinv(q):
    """Inverse of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def round_q(q, dec=7):
    return tuple(round(float(x), dec) for x in q)

def even_perms(t):
    """All even permutations of a 4-tuple."""
    base = list(t)
    result = set()
    for p in permutations(range(4)):
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            result.add(tuple(base[p[i]] for i in range(4)))
    return result


# ===============================================================
# PART 1  Q8
# ===============================================================

print("=" * 60)
print("PART 1 – Q8 Cayley graph")
print("=" * 60)

# ------ 1a. Group elements and Cayley graph ------

Q8_elements = [
    (1,0,0,0), (-1,0,0,0),
    (0,1,0,0), (0,-1,0,0),
    (0,0,1,0), (0,0,-1,0),
    (0,0,0,1), (0,0,0,-1),
]
Q8_np  = {e: np.array(e, dtype=float) for e in Q8_elements}
Q8_idx = {e: i for i, e in enumerate(Q8_elements)}
N_Q8   = 8

# Generating set S = {±i, ±j, ±k}
S_Q8 = [(0,1,0,0),(0,-1,0,0),(0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]
d_Q8 = len(S_Q8)

# Adjacency and Laplacian matrices
A_Q8 = np.zeros((N_Q8, N_Q8))
for i, g in enumerate(Q8_elements):
    for s in S_Q8:
        prod = round_q(qmul(Q8_np[g], Q8_np[s]))
        A_Q8[i, Q8_idx[prod]] += 1
L_Q8 = d_Q8 * np.eye(N_Q8) - A_Q8

# Verify group closure
for g in Q8_elements:
    for s in S_Q8:
        assert round_q(qmul(Q8_np[g], Q8_np[s])) in Q8_idx
print(f"Q8: |G|={N_Q8}, |S|={d_Q8}  (6-regular Cayley graph)  ✓")

# ------ 1b. Peter-Weyl eigenmodes (exact basis, no numerical rotation) ------

# The 2D irrep rho4: rho4(a+bi+cj+dk) = a*I + b*(iσ_x) + c*(iσ_y) + d*(iσ_z)
rho4_matrices = {
    (1,0,0,0):   np.array([[1,0],[0,1]],     dtype=complex),
    (-1,0,0,0):  np.array([[-1,0],[0,-1]],   dtype=complex),
    (0,1,0,0):   np.array([[1j,0],[0,-1j]],  dtype=complex),
    (0,-1,0,0):  np.array([[-1j,0],[0,1j]],  dtype=complex),
    (0,0,1,0):   np.array([[0,1],[-1,0]],    dtype=complex),
    (0,0,-1,0):  np.array([[0,-1],[1,0]],    dtype=complex),
    (0,0,0,1):   np.array([[0,1j],[1j,0]],   dtype=complex),
    (0,0,0,-1):  np.array([[0,-1j],[-1j,0]], dtype=complex),
}

# Abelian irreps rho1, rho2, rho3 (characters on each element)
chi_abel = {
    'rho1': {e: 1 for e in Q8_elements},
    'rho2': {e: 1 for e in Q8_elements},
    'rho3': {e: 1 for e in Q8_elements},
}
# rho1: trivial on i, sign on j,k
for e in Q8_elements:
    chi_abel['rho1'][e] = 1 if e in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0)] else -1
    chi_abel['rho2'][e] = 1 if e in [(1,0,0,0),(-1,0,0,0),(0,0,1,0),(0,0,-1,0)] else -1
    chi_abel['rho3'][e] = 1 if e in [(1,0,0,0),(-1,0,0,0),(0,0,0,1),(0,0,0,-1)] else -1

# Peter-Weyl normalization factor: sqrt(dim_rho / |G|)
pf_rho4 = np.sqrt(2.0 / N_Q8)   # = 1/2
pf_abel  = np.sqrt(1.0 / N_Q8)  # = 1/(2sqrt2)

# Build Peter-Weyl eigenmodes (real parts; imaginary parts orthogonal too)
PW = {}   # label -> (lambda_value, real vector of length 8)
for u in range(2):
    for v in range(2):
        vec = np.array([pf_rho4 * rho4_matrices[e][u, v].real
                        for e in Q8_elements])
        PW[f'rho4_{u}{v}'] = (6.0, vec)

for name, chi in chi_abel.items():
    vec = np.array([pf_abel * chi[e] for e in Q8_elements])
    PW[name] = (8.0, vec)

# Verify norms, orthogonality, and eigenvalues
print("\nPeter-Weyl modes:")
print(f"  {'Mode':12s}  lambda  ||psi||_2  ||psi||_inf  Rayleigh")
all_ok = True
vecs = list(PW.items())
for label, (lam, psi) in vecs:
    norm2   = np.linalg.norm(psi)
    inf_    = np.max(np.abs(psi))
    rayleigh = float(psi @ L_Q8 @ psi) / float(psi @ psi)
    ok = abs(norm2 - 1.0) < 1e-9 and abs(rayleigh - lam) < 1e-8
    if not ok:
        all_ok = False
    print(f"  {label:12s}  {lam:6.1f}  {norm2:.6f}   {inf_:.6f}    {rayleigh:.4f}")
if all_ok:
    print("  All norms=1, eigenvalues exact  ✓")

# ------ 1c. Admissibility envelope ------

print("\nAdmissibility envelope  A^max_n = c_chi / sqrt(lambda_n):")
for label, (lam, psi) in PW.items():
    psi_inf = np.max(np.abs(psi))
    # A^max such that A_n * ||psi_n||_inf = c_chi / sqrt(lambda_n) * ||psi_n||_inf
    # Effective amplitude A_n^eff = |a_n| * ||psi_n||_inf
    # Constraint from DBI: A_n^eff <= c_chi / sqrt(lambda_n)
    amax = f"c_chi / sqrt({lam:.4g})"
    print(f"  {label:12s}: lambda={lam:.1f},  A^max = {amax}")

lam_quat = 6.0;  lam_abel = 8.0
ratio_Q8 = np.sqrt(lam_abel / lam_quat)
print(f"\n  Key ratio: A^max_rho4 / A^max_abel = sqrt({lam_abel:.0f}/{lam_quat:.0f})"
      f" = sqrt(4/3) = {ratio_Q8:.6f}")

# ------ 1d. DBI shape factors ------

print("\nDBI shape factors kappa_n / ||psi_n||^2_inf  (expect = 1 on Q8):")
for label, (lam, psi) in PW.items():
    kappa   = np.sum(psi**4)
    psi_inf = np.max(np.abs(psi))**2
    ratio   = kappa / psi_inf
    print(f"  {label:12s}: kappa={kappa:.6f},  kappa/||psi||^2_inf = {ratio:.6f}")


# ===============================================================
# PART 2  2I (binary icosahedral group)
# ===============================================================

print("\n" + "=" * 60)
print("PART 2 – 2I Cayley graph")
print("=" * 60)

# ------ 2a. Enumerate all 120 elements ------

phi  = (1.0 + np.sqrt(5.0)) / 2.0
iphi = phi - 1.0   # = 1/phi

elems_2I = set()

# 2T subset: ±1, ±i, ±j, ±k  and  (±1 ± i ± j ± k)/2
for s in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
          (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
    elems_2I.add(round_q(s))
for s in product([1,-1], repeat=4):
    elems_2I.add(round_q(tuple(x / 2.0 for x in s)))

# 96 icosahedral elements: even permutations of (0, ±1/2, ±phi/2, ±iphi/2)
base_4 = (0.0, 0.5, phi / 2.0, iphi / 2.0)
for signs in product([1,-1], repeat=4):
    v = tuple(s * x for s, x in zip(signs, base_4))
    for perm in even_perms(v):
        if abs(sum(x**2 for x in perm) - 1.0) < 1e-9:
            elems_2I.add(round_q(perm))

assert len(elems_2I) == 120
elems_2I_list = sorted(elems_2I)
elems_2I_np   = {e: np.array(e) for e in elems_2I_list}
elem_2I_idx   = {e: i for i, e in enumerate(elems_2I_list)}
N_2I          = 120
print(f"\n2I: enumerated {N_2I} unit quaternions  ✓")

# ------ 2b. Element orders and conjugacy classes ------

id_np = np.array([1.0, 0.0, 0.0, 0.0])

def elem_order(g_np, tol=1e-5):
    curr = g_np.copy()
    for k in range(1, 25):
        if np.linalg.norm(curr - id_np) < tol:
            return k
        curr = qmul(curr, g_np)
    return -1

order_count = Counter(elem_order(elems_2I_np[e]) for e in elems_2I_list)
expected    = {1: 1, 2: 1, 3: 20, 4: 30, 5: 24, 6: 20, 10: 24}
print("\nElement orders in 2I:")
all_ok = True
for o in sorted(order_count):
    ok = "✓" if order_count[o] == expected.get(o, -1) else "✗"
    if ok == "✗":
        all_ok = False
    print(f"  order {o:2d}: {order_count[o]:3d}  {ok}")
if all_ok:
    print("  All orders match expected values  ✓")

by_order = defaultdict(list)
for e in elems_2I_list:
    by_order[elem_order(elems_2I_np[e])].append(e)

def chi2_val(e):
    """Character of spin-1/2 rep: chi_2(g) = 2 * a0."""
    return 2.0 * e[0]

o5_pos  = [e for e in by_order[5]  if chi2_val(e) > 0]
o5_neg  = [e for e in by_order[5]  if chi2_val(e) < 0]
o10_pos = [e for e in by_order[10] if chi2_val(e) > 0]
o10_neg = [e for e in by_order[10] if chi2_val(e) < 0]

class_info = [
    ('1a',  1,  by_order[1]),
    ('2a',  2,  by_order[2]),
    ('3a',  3,  by_order[3]),
    ('4a',  4,  by_order[4]),
    ('5a',  5,  o5_pos),
    ('5b',  5,  o5_neg),
    ('6a',  6,  by_order[6]),
    ('10a', 10, o10_pos),
    ('10b', 10, o10_neg),
]
class_sz   = {name: len(members) for name, _, members in class_info}
class_lbls = [name for name, _, _ in class_info]

print("\nConjugacy classes of 2I:")
for name, order, members in class_info:
    print(f"  {name:>5}  order {order:2d}  size {len(members):3d}")
assert sum(class_sz.values()) == 120

# ------ 2c. Character table (spin-type irreps) ------

# Characters computed from symmetric tensor powers Sym^k(V_2) and verified.
CT = {
    'chi1 (spin 0)':   [1,  1,  1,  1,  1,     1,     1,  1,      1    ],
    'chi2 (spin 1/2)': [2, -2, -1,  0,  iphi,  -phi,   1,  phi,  -iphi  ],
    'chi3 (spin 1)':   [3,  3,  0, -1,  phi,   -iphi,  0, -iphi,  phi   ],
    'chi4 (spin 3/2)': [4, -4,  1,  0, -1,     -1,    -1,  1,     1     ],
    'chi5 (spin 2)':   [5,  5, -1,  1,  0,      0,    -1,  0,     0     ],
    'chi6 (spin 5/2)': [6, -6,  0,  0,  1,      1,     0, -1,    -1     ],
}
class_sizes_list = [class_sz[c] for c in class_lbls]

# Verify orthogonality
all_ok = True
keys = list(CT.keys())
for i, k1 in enumerate(keys):
    for j, k2 in enumerate(keys):
        ip = sum(class_sizes_list[c] * CT[k1][c] * CT[k2][c]
                 for c in range(9))
        if abs(ip - (120 if i == j else 0)) > 1e-6:
            print(f"  FAIL <{k1},{k2}> = {ip:.4f}")
            all_ok = False
print(f"\nCharacter table orthogonality: {'✓' if all_ok else 'FAIL'}")

# ------ 2d. Spectral admissibility for various generating sets ------

def cayley_spectrum(S_class_names):
    d = sum(class_sz[c] for c in S_class_names)
    cidx = {name: i for i, name in enumerate(class_lbls)}
    out  = {}
    for irrep, row in CT.items():
        dim     = row[0]
        chi_sum = sum(class_sz[c] * row[cidx[c]] for c in S_class_names)
        mu      = chi_sum / dim
        lam     = d - mu
        out[irrep] = {'mu': mu, 'lambda': lam, 'dim': dim}
    return d, out

gen_sets = {
    'S_A order-4   (d=30)':   ['4a'],
    'S_B order-5   (d=24)':   ['5a', '5b'],
    'S_C order-3   (d=20)':   ['3a'],
    'S_D order-10  (d=24)':   ['10a', '10b'],
    'S_E order-3+5 (d=44)':   ['3a', '5a', '5b'],
    'S_F order-5+10(d=48)':   ['5a', '5b', '10a', '10b'],
}

print("\nSpectral admissibility on 2I – lambda values per sector:")
hdr = f"  {'Generating set':28s}  {'1/2':>6}{'  1':>5}{'3/2':>6}{'  2':>5}{'5/2':>6}  winner"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

all_spectra = {}
for gen_name, S_cls in gen_sets.items():
    d, spec = cayley_spectrum(S_cls)
    lv = {
        '1/2': spec['chi2 (spin 1/2)']['lambda'],
        '1':   spec['chi3 (spin 1)']['lambda'],
        '3/2': spec['chi4 (spin 3/2)']['lambda'],
        '2':   spec['chi5 (spin 2)']['lambda'],
        '5/2': spec['chi6 (spin 5/2)']['lambda'],
    }
    all_spectra[gen_name] = (d, spec, lv)
    nt     = {k: v for k, v in lv.items() if v > 1e-10}
    minlam = min(nt.values())
    wins   = '+'.join(k for k, v in nt.items() if abs(v - minlam) < 1e-6)
    print(f"  {gen_name:28s}  "
          f"{lv['1/2']:6.1f}{lv['1']:6.1f}{lv['3/2']:6.1f}"
          f"{lv['2']:6.1f}{lv['5/2']:6.1f}  {wins}")

# ------ 2e. Focus: S_D (canonical generating set) ------

print("\n" + "-" * 52)
print("S_D = order-10 elements  (canonical analogue of Q8 generators)")
print("-" * 52)
d_D, spec_D, lv_D = all_spectra['S_D order-10  (d=24)']
print(f"\n  {'Irrep':22s} dim   mu     lambda   A^max")
for name, data in spec_D.items():
    lam  = data['lambda']
    amax = "inf" if lam < 1e-10 else f"c_chi/sqrt({lam:.4g})"
    print(f"  {name:22s} {data['dim']:3d}  {data['mu']:6.3f}  {lam:7.3f}   {amax}")

# ------ 2e. Golden ratio identity ------

print("\nAlgebraic origin of mu_(1/2) = mu_(3/2) = 6:")
print(f"  phi  = {phi:.6f},  1/phi = iphi = {iphi:.6f}")
print(f"  chi2(10a) = +phi  =  {phi:.6f}")
print(f"  chi2(10b) = -iphi = {-iphi:.6f}")
print(f"  chi2(10a) + chi2(10b) = phi - iphi"
      f" = phi - (phi-1) = 1  [phi - 1/phi = {phi - 1/phi:.6f}]")
print(f"  mu_{{1/2}} = (1/2) * (12*phi + 12*(-iphi))"
      f" = 6 * (phi - iphi) = 6 * 1 = 6  ✓")
print(f"  chi4(10a) = chi4(10b) = 1")
print(f"  mu_{{3/2}} = (1/4) * 24 = 6  ✓")
print(f"  chi5(10a) = chi5(10b) = 0  =>  mu_{{2}} = 0  (spectrally neutral)")

# ------ 2f. Comparison Q8 vs 2I ------

print("\n" + "=" * 60)
print("COMPARISON Q8 vs 2I: growing spectral hierarchy")
print("=" * 60)
ratio_2I = np.sqrt(28.0 / 18.0)
print(f"""
  Q8   S={{±i,±j,±k}} d=6:
    lambda: quat=6, abel=8
    Admissibility ratio  A^max_quat / A^max_abel  =  sqrt(8/6) = sqrt(4/3)
                                                   =  {np.sqrt(4/3):.6f}
    Algebraic origin: tr(sigma_x) = tr(sigma_y) = tr(sigma_z) = 0

  2I   S=10a+10b  d=24:
    lambda: spin-1/2=spin-3/2=18, spin-1=20, spin-2=24, spin-5/2=28
    Admissibility ratio  A^max_{{1/2}} / A^max_{{5/2}}  =  sqrt(28/18) = sqrt(14/9)
                                                        =  {ratio_2I:.6f}
    Algebraic origin: phi - 1/phi = 1  (golden ratio identity)

  Trend along Q8 ⊂ 2I ⊂ SU(2):
    sqrt(4/3)  = {np.sqrt(4/3):.4f}   (Q8,  2-level hierarchy)
    sqrt(14/9) = {ratio_2I:.4f}   (2I,  4-level hierarchy)
  => The hierarchy strengthens and becomes richer at each step.
""")


# ===============================================================
# PART 3  DBI dynamics simulation on Q8
# ===============================================================

print("=" * 60)
print("PART 3 – DBI dynamics simulation on Q8")
print("=" * 60)

# Collect Peter-Weyl modes by sector
pw_quat = [(lam, psi) for label, (lam, psi) in PW.items() if 'rho4' in label]
pw_abel = [(lam, psi) for label, (lam, psi) in PW.items() if 'rho' in label and 'rho4' not in label]

print(f"\nQ8 sectors:")
print(f"  quaternionic (rho4): {len(pw_quat)} modes, lambda=6,  A^max = c_chi/sqrt(6)")
print(f"  abelian (rho1-3):    {len(pw_abel)} modes, lambda=8,  A^max = c_chi/sqrt(8)")

c_chi   = 1.0
Amax_q  = c_chi / np.sqrt(6.0)
Amax_a  = c_chi / np.sqrt(8.0)
print(f"  c_chi = {c_chi},  A^max_quat = {Amax_q:.6f},  A^max_abel = {Amax_a:.6f}")

def sector_amplitude(chi_vec, modes):
    """
    Maximum effective amplitude over all modes in a sector.
    A_n^eff = |<psi_n, chi>| * ||psi_n||_inf
    """
    A_max = 0.0
    for lam, psi in modes:
        a_n   = float(psi @ chi_vec)
        psi_inf = np.max(np.abs(psi))
        A_max = max(A_max, abs(a_n) * psi_inf)
    return A_max

def simulate_dbi(chi0, pi0, c_chi, dt, n_steps):
    """
    Symplectic (leapfrog / Stormer-Verlet) integration of the Q8 DBI system.

    Hamiltonian:
        H = sum_v c^2 * sqrt(1 + pi_v^2/c^2) + (1/2) chi^T L chi

    Equations of motion:
        d chi_v / dt = pi_v / sqrt(1 + pi_v^2/c^2)      [sub-luminal velocity]
        d pi_v  / dt = -(L chi)_v                         [restoring force]

    Returns
    -------
    t_arr     : (n_steps+1,)       time points
    chi_hist  : (n_steps+1, N_Q8)  field history
    An_quat   : (n_steps+1,)       quaternionic sector amplitude
    An_abel   : (n_steps+1,)       abelian sector amplitude
    """
    chi = chi0.copy()
    pi  = pi0.copy()

    t_arr   = np.zeros(n_steps + 1)
    An_quat = np.zeros(n_steps + 1)
    An_abel = np.zeros(n_steps + 1)

    def chi_dot(pi_vec):
        return pi_vec / np.sqrt(1.0 + (pi_vec / c_chi)**2)

    def force(chi_vec):
        return -(L_Q8 @ chi_vec)

    def record(step):
        An_quat[step] = sector_amplitude(chi, pw_quat)
        An_abel[step] = sector_amplitude(chi, pw_abel)

    record(0)
    for step in range(n_steps):
        # Leapfrog
        pi_half = pi + 0.5 * dt * force(chi)
        chi     = chi + dt * chi_dot(pi_half)
        pi      = pi_half + 0.5 * dt * force(chi)

        # Enforce sub-luminal constraint (clip if numerical drift pushes over)
        v       = chi_dot(pi)
        max_v   = np.max(np.abs(v))
        if max_v > 0.9999 * c_chi:
            pi *= (0.9999 * c_chi / max_v)

        t_arr[step + 1] = (step + 1) * dt
        record(step + 1)

    return t_arr, chi, An_quat, An_abel

# Initial condition: inject equal energy in both sectors at ~80% of A^max
np.random.seed(42)
chi0 = np.zeros(N_Q8)
pi0  = np.zeros(N_Q8)
frac = 0.80

for k, (lam, psi) in enumerate(pw_quat):
    psi_inf = np.max(np.abs(psi))
    a_n     = frac * Amax_q / psi_inf
    phase   = 0.3 * k
    chi0   += a_n * psi * np.cos(phase)
    pi0    += -a_n * np.sqrt(lam) * psi * np.sin(phase)

for k, (lam, psi) in enumerate(pw_abel):
    psi_inf = np.max(np.abs(psi))
    a_n     = frac * Amax_a / psi_inf
    phase   = 0.5 * k
    chi0   += a_n * psi * np.cos(phase)
    pi0    += -a_n * np.sqrt(lam) * psi * np.sin(phase)

dt      = 0.02
n_steps = 3000

print(f"\nSimulation: dt={dt}, n_steps={n_steps}, T_max={n_steps*dt:.1f}")
print(f"  Initial condition: {frac*100:.0f}% of A^max in each sector")

t_arr, chi_final, An_quat, An_abel = simulate_dbi(chi0, pi0, c_chi, dt, n_steps)

print(f"\n  Time series (saturation fraction A/A^max):")
print(f"  {'t':>8}  {'quat/A^max_q':>14}  {'abel/A^max_a':>14}")
checkpoints = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
for s in checkpoints:
    fq = An_quat[s] / Amax_q
    fa = An_abel[s] / Amax_a
    print(f"  {t_arr[s]:8.2f}  {fq:14.4f}  {fa:14.4f}")

# Physical interpretation
fq_end = An_quat[-1] / Amax_q
fa_end = An_abel[-1] / Amax_a
print(f"\n  Outcome: quat saturates at {fq_end:.3f} * A^max_q,"
      f"  abel at {fa_end:.3f} * A^max_a")
print(f"  The quaternionic sector retains a larger absolute amplitude")
print(f"  because A^max_q = {Amax_q:.4f} > A^max_a = {Amax_a:.4f}")


# ===============================================================
# PART 4  Plots
# ===============================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('../figures', exist_ok=True)
    plt.rcParams.update({'font.size': 11})

    # --- Figure 1: Admissibility envelope on Q8 ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_lam = np.linspace(0.3, 10.5, 400)
    ax.plot(x_lam, c_chi / np.sqrt(x_lam), 'k-', lw=2,
            label=r'$A^{\max} = c_\chi/\sqrt{\lambda}$')
    pts = [
        (6.0, Amax_q, 'C0', r'quaternionic $\rho_4$  ($\lambda=6$)'),
        (8.0, Amax_a, 'C1', r'abelian $\rho_{1,2,3}$  ($\lambda=8$)'),
    ]
    for lam, amax, col, lbl in pts:
        ax.scatter([lam], [amax], s=140, color=col, zorder=5, label=lbl)
        ax.axvline(lam,  color=col, lw=0.8, ls=':', alpha=0.7)
        ax.axhline(amax, color=col, lw=0.8, ls=':', alpha=0.7)
    ax.annotate(r'$\sqrt{4/3}\approx1.155$',
                xy=(7, (Amax_q + Amax_a) / 2),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='<->', color='grey'),
                xytext=(7, (Amax_q + Amax_a) / 2))
    ax.set_xlabel(r'Laplacian eigenvalue $\lambda_n$')
    ax.set_ylabel(r'$A^{\max}_n / c_\chi$')
    ax.set_title(r'Admissibility envelope on $Q_8$ Cayley graph')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'figures/fig1_Q8_admissibility.{ext}', dpi=150)
    plt.close()

    # --- Figure 2: DBI dynamics ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_arr, An_quat, 'C0', lw=1.5, label=r'quaternionic $\rho_4$')
    axes[0].plot(t_arr, An_abel, 'C1', lw=1.5, label=r'abelian $\rho_{1,2,3}$')
    axes[0].axhline(Amax_q, color='C0', ls='--', lw=1.2,
                    label=fr'$A^{{\max}}_{{\rho_4}}={Amax_q:.3f}$')
    axes[0].axhline(Amax_a, color='C1', ls='--', lw=1.2,
                    label=fr'$A^{{\max}}_{{\rho_k}}={Amax_a:.3f}$')
    axes[0].set_ylabel(r'Effective amplitude $A_n(t)$')
    axes[0].set_title(r'DBI dynamics on $Q_8$: effective amplitude per sector')
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].plot(t_arr, An_quat / Amax_q, 'C0', lw=1.5,
                 label=r'quat / $A^{\max}_q$')
    axes[1].plot(t_arr, An_abel / Amax_a, 'C1', lw=1.5,
                 label=r'abel / $A^{\max}_a$')
    axes[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel(r'modal time $\omega_n t$')
    axes[1].set_ylabel('Saturation fraction')
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.4)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'figures/fig2_Q8_DBI_dynamics.{ext}', dpi=150)
    plt.close()

    # --- Figure 3: Q8 vs 2I hierarchy ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def plot_hierarchy(ax, sectors, lambdas, colors, title):
        y     = np.arange(len(sectors))
        amaxs = [1.0 / np.sqrt(l) for l in lambdas]
        bars  = ax.barh(y, amaxs, color=colors, alpha=0.85, edgecolor='k', lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sectors)
        ax.set_xlabel(r'$A^{\max}/c_\chi = 1/\sqrt{\lambda}$')
        ax.set_title(title)
        for i, (amax, lam) in enumerate(zip(amaxs, lambdas)):
            ax.text(amax + 0.002, i, fr'$\lambda={lam}$',
                    va='center', fontsize=9)
        ax.set_xlim(0, max(amaxs) * 1.35)

    plot_hierarchy(axes[0],
        sectors=['abelian (×3)', r'quaternionic $\rho_4$'],
        lambdas=[8, 6],
        colors=['C1', 'C0'],
        title=r'$Q_8$   $S=\{\pm i,\pm j,\pm k\}$,  $d=6$')

    plot_hierarchy(axes[1],
        sectors=[r'spin-$\frac{5}{2}$', r'spin-$2$',
                 r'spin-$1$', r'spin-$\frac{3}{2}$', r'spin-$\frac{1}{2}$'],
        lambdas=[28, 24, 20, 18, 18],
        colors=['C4', 'C3', 'C2', 'C0', 'C0'],
        title=r'$2I$   $S=10a\cup10b$,  $d=24$')

    fig.suptitle(r'Spectral admissibility hierarchy: $Q_8 \to 2I \to \mathrm{SU}(2)$',
                 fontsize=13)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'figures/fig3_hierarchy_Q8_vs_2I.{ext}', dpi=150)
    plt.close()

    print("\nFigures saved in ./figures/")
    print("  fig1_Q8_admissibility.{png,pdf}")
    print("  fig2_Q8_DBI_dynamics.{png,pdf}")
    print("  fig3_hierarchy_Q8_vs_2I.{png,pdf}")

except ImportError:
    print("\n(matplotlib not available – skipping figures)")


# ===============================================================
# SUMMARY
# ===============================================================

print("\n" + "=" * 60)
print("SUMMARY OF ALL RESULTS")
print("=" * 60)
print(f"""
  Q8  (Steps 1-3):
    Cayley graph: 6-regular, 8 vertices
    lambda_quat = 6,   lambda_abel = 8
    A^max_quat / A^max_abel = sqrt(4/3) = {np.sqrt(4/3):.6f}
    Origin: tr(Pauli matrices) = 0  =>  mu_rho4 = 0

  2I  (Step 5):
    Cayley graph: 24-regular, 120 vertices  (S = order-10 elements)
    lambda(1/2) = lambda(3/2) = 18  <  lambda(1) = 20
                <  lambda(2) = 24   <  lambda(5/2) = 28
    A^max_{{1/2}} / A^max_{{5/2}} = sqrt(14/9) = {np.sqrt(14/9):.6f}
    Origin: phi - 1/phi = 1  (golden ratio)  =>  mu_{{1/2}} = mu_{{3/2}} = 6

  Trend  Q8 -> 2I -> SU(2):
    sqrt(4/3)  = {np.sqrt(4/3):.4f}  (Q8)
    sqrt(14/9) = {np.sqrt(14/9):.4f}  (2I)
    The hierarchy strengthens and becomes richer with each step.
    Both origins are structural character-theoretic identities.
""")
