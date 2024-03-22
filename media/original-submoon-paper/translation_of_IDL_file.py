import numpy as np
import matplotlib.pyplot as plt

# this is a simple python translation of the script 'mass_planet.pro', done by ChatGPT.
# As I see it, it almost correctly reproduces the sub-figures of figure 1 of the 2019 submoon paper.
# Haven't had the time yet to correct this script.

# Constants
kmcm = 1e5
yearsec = 3.154e7
G = 6.67e-8

# Solar System data
rearthkm = 6378.  # km
rearth = 6378. * kmcm  # km
mearth = 5.974e27

Mmars = 0.107 * mearth
Rmars = 0.53 * rearth

Mluna = 0.012 * mearth
Rluna = 0.273 * rearth

mpsyche = 2.72e22
rpsyche = 113. * kmcm

mvesta = 2.589e23  # g
rvesta = 262.7 * kmcm

mceres = 1.5e-4 * mearth
rceres = 473. * kmcm

Mjup = 317.8 * mearth
Rjup = 69911. * kmcm

Msat = 95.16 * mearth
Rsat = 58232. * kmcm

Mnep = 17.15 * mearth
Rnep = 24622. * kmcm

Mura = 14.54 * mearth
Rura = 25362. * kmcm

MK1625 = 4. * Mjup
RK1625 = 11.4 * rearth

# Define age of system
T = 4.6e9 * yearsec  # 5 Gyr

# Stability parameter (in fraction of Hill radius; Domingos et al)
f = 0.4895

# Define moon-of-moon properties
rhomm = 2.
Rmma = 5.0 * kmcm
Mmma = (4. / 3.) * np.pi * rhomm * (Rmma) ** 3  # grams

Rmmb = 10. * kmcm
Mmmb = (4. / 3.) * np.pi * rhomm * (Rmmb) ** 3  # grams

Rmmc = 20. * kmcm
Mmmc = (4. / 3.) * np.pi * rhomm * (Rmmc) ** 3  # grams

Rmmd = 500.0 * kmcm
Mmmd = (4. / 3.) * np.pi * rhomm * (Rmmd) ** 3  # grams

# Jupiter
rhomoon = 2.5
k2p = 0.25
Qmoon = 100.
Rmoon = np.arange(1001) * 10. * kmcm
Mmoon = (4. / 3.) * np.pi * rhomoon * (Rmoon) ** 3  # grams
Mplan = Mjup

acrita = (1. / f) * (3. * Mplan * (13. / 2. * Mmma * (3. * k2p * T * Rmoon ** 5 * np.sqrt(G) / (Mmoon ** (8. / 3.) * Qmoon))) ** (6. / 13.)) ** (1 / 3.)
acritb = (1. / f) * (3. * Mplan * (13. / 2. * Mmmb * (3. * k2p * T * Rmoon ** 5 * np.sqrt(G) / (Mmoon ** (8. / 3.) * Qmoon))) ** (6. / 13.)) ** (1 / 3.)
acritc = (1. / f) * (3. * Mplan * (13. / 2. * Mmmc * (3. * k2p * T * Rmoon ** 5 * np.sqrt(G) / (Mmoon ** (8. / 3.) * Qmoon))) ** (6. / 13.)) ** (1 / 3.)

plt.figure(figsize=(10, 6))
plt.plot(acrita / Rjup, Rmoon / kmcm, label='Moon Size = 5 km', linestyle='-')
plt.plot(acritb / Rjup, Rmoon / kmcm, label='Moon Size = 10 km', linestyle='--')
plt.plot(acritc / Rjup, Rmoon / kmcm, label='Moon Size = 20 km', linestyle='-.')
plt.xlabel('Orbital distance (Jupiter radii)')
plt.ylabel('Radius of moon (km)')
plt.title('Jupiter')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
