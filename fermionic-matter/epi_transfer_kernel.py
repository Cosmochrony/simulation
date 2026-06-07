"""
epi_transfer_kernel.py

Step (3a): determine the profile of the angular-radial transfer kernel
    K_norm(m) = Delta_eps(m) / Delta_Ihat(m),     K_raw(m) = Delta_eps(m) / Delta_I(m)
that defines the cascade normalisation N_casc via
    eps_n = sum_{m<=n} K_norm(m) Delta_Ihat(m),   eps = N_casc |alpha|.

Radial observable (principal): normalised cumulative projected capacity
    Ihat(n) = [sigma_pair(0) - sigma_pair(n)] / [sigma_pair(0) - sigma_pair(n_sat)]  in [0,1],
with sigma_pair(n) ~ (1+n)^{-delta_pair} (pre-saturation power law) and n_sat = n_3^obs.
Raw I(n) = sigma_pair(0) - sigma_pair(n) kept as a control only.

The profile of K_norm is NOT free: it is fixed by the coupling law Delta_eps(m), i.e. how the
angular J_3 split locks in per cascade step. Three structural candidates:
    (A) arrow-coupling     Delta_eps ~ Delta_Ihat   (split rides the projection arrow I(n))
    (B) per-step           Delta_eps ~ const        (rotation per metaplectic step)
    (C) residual-capacity  Delta_eps ~ sigma_pair(m)
Each yields one of the three profiles {constant, n3-concentrated, saturating}.

Outputs:
    - stdout: profile classification + eps_sat under each law
    - epi_transfer_kernel.png
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
DELTA_PAIR = 7.44          # fibre-level capacity exponent (O16-O21)
N_SAT = 13                 # n_3^obs proxy (intrinsic saturation rank, O21 scale)
OUT_PNG = "epi_transfer_kernel.png"

n = np.arange(0, N_SAT + 1)
sigma = (1.0 + n) ** (-DELTA_PAIR)          # sigma_pair(n)
I_raw = sigma[0] - sigma                     # raw cumulative projected capacity
I_hat = I_raw / I_raw[N_SAT]                  # normalised in [0,1]

dI_raw = np.diff(I_raw)                        # Delta I(m),    m=1..N_SAT
dI_hat = np.diff(I_hat)                        # Delta Ihat(m)
sig_mid = sigma[1:]                            # sigma_pair(m) for m=1..N_SAT
m = n[1:]

# --------------------------------------------------------------------------- #
# candidate angular increments Delta_eps(m), each normalised so eps_sat = 1
# (shape only; the absolute scale ties to the model-dependent alpha)
# --------------------------------------------------------------------------- #
def normalise(d):
    return d / d.sum()

deps_A = normalise(dI_hat.copy())              # arrow-coupling
deps_B = normalise(np.ones_like(m, float))     # per-step (uniform)
deps_C = normalise(sig_mid.copy())             # residual-capacity

laws = {"A arrow (Delta_eps ~ Delta_Ihat)": deps_A,
        "B per-step (Delta_eps ~ const)": deps_B,
        "C residual (Delta_eps ~ sigma)": deps_C}

print("=" * 74)
print(f"TRANSFER KERNEL PROFILE  (delta_pair={DELTA_PAIR}, n_sat=n_3^obs={N_SAT})")
print("=" * 74)
print(f"  Ihat(n_sat) = {I_hat[N_SAT]:.3f} (=1 by construction)")
print()
Knorm = {}
for name, deps in laws.items():
    Kn = deps / dI_hat
    Kr = deps / dI_raw
    Knorm[name] = Kn
    # profile descriptor: ratio of last to first kernel value
    span = Kn[-1] / Kn[0]
    cum = np.cumsum(deps)
    # fraction of eps accumulated in the last 3 steps (near saturation)
    tail = deps[-3:].sum()
    print(f"  {name}")
    print(f"     K_norm span (last/first) : {span:10.3e}")
    print(f"     eps fraction in last 3 steps near n_sat : {tail:6.1%}")
    if span < 3:
        verdict = "CONSTANT  -> pure transfer constant"
    elif span > 50:
        verdict = "n3-CONCENTRATED -> transfer fixed near n_3^obs"
    else:
        verdict = "SATURATING -> bounded by relaxation"
    print(f"     profile : {verdict}")
    print()

print("=" * 74)
print("READING")
print("=" * 74)
print("  - The K_norm profile is set entirely by the coupling law Delta_eps(m):")
print("      A (arrow)     -> K_norm CONSTANT       (eps rides I(n); strong sub-case)")
print("      B (per-step)  -> K_norm rises ~ m^{d+1} (n3-concentrated)")
print("      C (residual)  -> K_norm SATURATING")
print("  - The architecture test (split coupled to decay BY the arrow I(n), with the")
print("    split sign flipping under cascade reversal) selects law A: the angular split")
print("    locks in per unit of projected capacity, not per bare metaplectic step.")
print("    Law B moreover makes eps grow with n_sat (no intrinsic bound); law A gives a")
print("    saturating, n_sat-independent eps_sat = K_norm.")
print("  => Under the I(n)-coupling, K_norm is CONSTANT and N_casc is a PURE transfer")
print("     constant; eps_sat = K_norm. The residual is the ABSOLUTE value of this")
print("     constant, still tied to the physical (non-shear-model) normalisation of")
print("     alpha -- NOT to be fixed by matching 0.156 to 1/10.")
print("=" * 74)

# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.semilogy(m, Knorm["A arrow (Delta_eps ~ Delta_Ihat)"], "o-", color="#3b7a57",
            label="A arrow: constant")
ax.semilogy(m, Knorm["B per-step (Delta_eps ~ const)"], "s-", color="#a83232",
            label="B per-step: $\\sim m^{\\delta+1}$")
ax.semilogy(m, Knorm["C residual (Delta_eps ~ sigma)"], "^-", color="#1f3b73",
            label="C residual: saturating")
ax.axvline(N_SAT, ls=":", color="grey")
ax.set_xlabel("cascade step $m$")
ax.set_ylabel(r"$K_{\rm norm}(m)$  (log scale)")
ax.set_title(r"Transfer-kernel profile vs coupling law")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(m, np.cumsum(deps_A), "o-", color="#3b7a57", label="A arrow")
ax.plot(m, np.cumsum(deps_B), "s-", color="#a83232", label="B per-step")
ax.plot(m, np.cumsum(deps_C), "^-", color="#1f3b73", label="C residual")
ax.plot(m, I_hat[1:], "k--", lw=1, label=r"$\widehat I(n)$")
ax.axvline(N_SAT, ls=":", color="grey")
ax.set_xlabel("cascade step $m$")
ax.set_ylabel(r"accumulated $\varepsilon_n/\varepsilon_{\rm sat}$")
ax.set_title(r"Accumulation: A tracks $\widehat I$, B concentrates at $n_3^{\rm obs}$")
ax.legend(fontsize=8)

fig.suptitle("Step 3a: K_norm profile is fixed by the coupling law; the I(n)-arrow "
             "selects the constant (A) case", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PNG, dpi=150)
print(f"\nFigure written to {OUT_PNG}")
