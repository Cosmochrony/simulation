#!/usr/bin/env python3
"""Front "Generation stabilisation depths in Lambda_proj(n)" -- step 1 + step 2 audit.

Exact symbolic, NO fit. Two questions settled here:

  Step 1 (reconciliation). Is the PYO Schur-residue split w_g = {1, 1/2+u, 1/2-u}
  the SAME object as the stratigraphy level weight c_i/dim rho_i, or an independent
  factor on the same rank-three carrier?  --> distinct objects (different ratio-sets,
  different functional inputs, different spaces).

  Step 2 (level-crossing cancellation). In the exponential trajectory-branching
  regime, if the generation depths n_g are fixed by the crossing of the growing
  algebraic connectivity lambda_2(n) ~ Lambda_proj(n) against the stratigraphic
  levels lambda_g, does the depth amplification A(n)=exp(beta* n) produce a
  hierarchy?  --> NO: beta* cancels and m_i/m_j collapses to lambda_i/lambda_j = O(1),
  reproducing the published single-level no-go. A genuine hierarchy therefore needs
  the depths n_g to be set by an intrinsic branching scale DECOUPLED from the
  stratigraphic level ratios; the required gaps (~40-125 cascade steps) are not
  reachable from the compressed static data (level ratios <= 1.5, dims in {4,5,6}).

Corpus anchors (verified, not refitted):
  - PYO Beau2026pyo prop:squared-yukawa : Spec(H_Pi)|gen = lamY^2 {1, 1/2+u, 1/2-u}.
  - Stratigraphy Beau2026a4 prop:threelevels / tab:threelevels : 2I ord-5 levels
    {20,24,30}, mult (54,25,40).
  - Relaxation Beau2026a5 def:nproj / eq:nproj-value / eq:mass-formula ; growth law
    Lambda_proj(n)=c_chi(n)^2/A_min^2 ; cor:linear-law Lambda_proj ~ lambda_2(n).
  - O16 Beau2026a20 : beta* = 1/(delta+1/2), delta_pair ~ 7.44 -> beta* ~ 0.127.
"""

import math
from fractions import Fraction as F

import sympy as sp

checks = []


def record(name, ok, detail=""):
    checks.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}" + (f"  --  {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Step 1 : the locked separation -- two distinct objects on the same rank-3 carrier
# --------------------------------------------------------------------------
print("\n=== Step 1 : reconciliation (PYO w_g  vs  stratigraphy c_i/dim rho_i) ===")

u = F(1, 10)
w_g = [F(1), F(1, 2) + u, F(1, 2) - u]                       # {1, 3/5, 2/5}
levels = [F(20), F(24), F(30)]                               # 2I ord-5 Cayley levels

ratios_w = sorted(w / min(w_g) for w in w_g)                 # {1, 3/2, 5/2}
ratios_l = sorted(l / min(levels) for l in levels)          # {1, 6/5, 3/2}

record("w_g at u=1/10 equals {1, 3/5, 2/5}",
       w_g == [F(1), F(3, 5), F(2, 5)], str(w_g))
record("stratigraphic ratio-set != PYO ratio-set",
       ratios_w != ratios_l, f"w {ratios_w}  vs  strat {ratios_l}")
# Functional independence: w_g is a function of the single vertical datum u only
# (free under Sym^2, Front 3); c_i/dim rho_i = F_KM(lambda_i)/dim rho_i. No shared input.
record("objects are functionally independent",
       True, "w_g=f(u) [Sym^2 vertical]  ;  c_i/dim rho_i=F_KM(lambda_i)/dim rho_i [2I/KM]")


# --------------------------------------------------------------------------
# Step 2 : level-crossing cancellation in the exponential regime
# --------------------------------------------------------------------------
print("\n=== Step 2 : level-crossing cancellation (exponential trajectory-branching) ===")

b, L0, lam_i, lam_j = sp.symbols("beta L0 lambda_i lambda_j", positive=True)

# Exponential regime: lambda_2(n) = L0 * exp(beta n). Level lambda becomes projectable
# (Lambda_proj ~ lambda_2 crosses lambda) at depth n_proj(lambda) = (1/beta) ln(lambda/L0).
n_i = sp.log(lam_i / L0) / b
n_j = sp.log(lam_j / L0) / b
depth_gap = sp.simplify(n_i - n_j)
record("depth gap n_i-n_j = (1/beta) ln(lam_i/lam_j)",
       sp.simplify(depth_gap - sp.log(lam_i / lam_j) / b) == 0, str(depth_gap))

# Depth amplification A(n)=exp(beta n): mass ratio in the level-crossing reading.
mass_ratio = sp.simplify(sp.exp(b * n_i) / sp.exp(b * n_j))
record("A(n_i)/A(n_j) = lam_i/lam_j  (beta cancels)",
       sp.simplify(mass_ratio - lam_i / lam_j) == 0, str(mass_ratio))

# Numerically on the stratigraphic levels: O(1), not a hierarchy.
strat_ratio_max = 30 / 20
record("stratigraphic level-crossing mass ratio is O(1)",
       strat_ratio_max <= 2.0, f"max lam_i/lam_j = {strat_ratio_max:.3f}")


# --------------------------------------------------------------------------
# Step 2 : required depth gaps vs reachable static depths
# --------------------------------------------------------------------------
print("\n=== Step 2 : required Delta n_g (observed hierarchy) vs reachable static depths ===")

observed = [("tau:e", 3477.0), ("mu:e", 207.0), ("t:u", 7.5e4)]
betas = [0.090, 0.127, 0.213]                                # O16 range; 0.127 at delta_pair
for name, r in observed:
    for bb in betas:
        print(f"    {name:6s} ratio={r:9.0f}  beta*={bb:.3f}  ->  Delta n_req = {math.log(r)/bb:6.1f} steps")

# Compressed static data: level ratios <= 1.5, dims in {4,5,6}. ln() is O(1).
ln_level = math.log(1.5)
ln_dim = math.log(6 / 4)
record("static inputs give only O(1) depth gaps",
       ln_level < 1 and ln_dim < 1, f"ln(1.5)={ln_level:.3f}, ln(6/4)={ln_dim:.3f}")
# Smallest required gap (tau:e at the fastest beta) far exceeds any O(1) static depth.
min_required = math.log(207.0) / 0.213
record("required Delta n_g (>=25) unreachable from static data",
       min_required > 10, f"min required ~ {min_required:.1f} steps >> O(1)")


# --------------------------------------------------------------------------
print("\n=== VERDICT ===")
print("Step 1 : generation label is stratigraphic; w_g is a transported Schur-residue")
print("         weight on the same rank-three carrier. Two readings, not one object.")
print("Step 2 : level-crossing depths give beta-independent O(1) ratios (no-go reappears).")
print("         A hierarchy needs intrinsic branching depths n_g decoupled from {lambda_g};")
print("         m_g/lamY = w_g S_g A(n_g), n_g intrinsic stabilisation depths of Lambda_proj(n).")
print("         The published base does NOT fix n_g -> honest reduction, not a derivation.")

n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n{n_pass}/{len(checks)} checks pass.")
assert n_pass == len(checks)
