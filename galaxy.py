import numpy as np
import matplotlib.pyplot as plt

# constants
G = 4.302e-6  # kpc (km/s)^2 Msun^-1
c = 3.0e5     # km/s
H0 = 70.0 / 1000.0  # km/s/kpc
a0 = c * H0

def load_rotmod(filename):
    data = np.loadtxt(filename)
    r, vobs, ev, vgas, vdisk, vbulge = data.T
    return r, vobs, ev, vgas, vdisk, vbulge

def v_cosmo(r, vgas, vdisk, vbulge, ups):
    vbar2 = vgas**2 + ups*vdisk**2 + ups*vbulge**2
    gN = vbar2 / r
    geff = np.sqrt(gN**2 + a0*gN)
    return np.sqrt(r * geff)

# example: NGC 3198
prefix = "data/Rotmod_LTG/"
r, vobs, ev, vgas, vdisk, vbulge = load_rotmod(prefix + "NGC3198_rotmod.dat")

ups = 0.5  # example stellar M/L
vmodel = v_cosmo(r, vgas, vdisk, vbulge, ups)

plt.errorbar(r, vobs, yerr=ev, fmt='o', label="Observed")
plt.plot(r, vmodel, label="Cosmochrony")
plt.xlabel("r [kpc]")
plt.ylabel("V [km/s]")
plt.legend()
plt.show()
