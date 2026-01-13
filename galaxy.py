import numpy as np
import matplotlib.pyplot as plt

# constants
G = 4.302e-6  # kpc (km/s)^2 Msun^-1
c = 3.0e5     # km/s
H0 = 70.0 / 1000.0  # km/s/kpc
a0 = c * H0

import numpy as np

def load_rotmod(filename):
    data = np.loadtxt(filename, comments="#")  # ignore header lines

    ncol = data.shape[1]
    if ncol < 6:
        raise ValueError(f"{filename}: expected at least 6 columns, got {ncol}")

    r     = data[:, 0]
    vobs  = data[:, 1]
    ev    = data[:, 2]
    vgas  = data[:, 3]
    vdisk = data[:, 4]
    vbul  = data[:, 5]

    # Optional surface brightness columns (if present)
    sbdisk = data[:, 6] if ncol >= 7 else None
    sbbul  = data[:, 7] if ncol >= 8 else None

    return r, vobs, ev, vgas, vdisk, vbul, sbdisk, sbbul

def load_rotmod6(filename):
    data = np.loadtxt(filename)
    r, vobs, ev, vgas, vdisk, vbulge = data.T
    return r, vobs, ev, vgas, vdisk, vbulge

def v_cosmo(r, vgas, vdisk, vbul, ups_disk=0.5, ups_bul=0.7):
  vbar2 = vgas ** 2 + ups_disk * vdisk ** 2 + ups_bul * vbul ** 2
  gN = vbar2 / r
  geff = np.sqrt(gN ** 2 + a0 * gN)
  return np.sqrt(r * geff)

# example: NGC 3198
prefix = "data/Rotmod_LTG/"
r, vobs, ev, vgas, vdisk, vbulge, sbdisk, sbbul = load_rotmod(prefix + "NGC3198_rotmod.dat")

ups = 0.5  # example stellar M/L
vmodel = v_cosmo(r, vgas, vdisk, vbulge, ups)

plt.errorbar(r, vobs, yerr=ev, fmt='o', label="Observed")
plt.plot(r, vmodel, label="Cosmochrony")
plt.xlabel("r [kpc]")
plt.ylabel("V [km/s]")
plt.legend()
plt.show()
