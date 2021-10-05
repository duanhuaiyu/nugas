'''Physical constants.
Author: Huaiyu Duan (UNM)
Source: Particle Data Group https://pdg.lbl.gov
'''
from math import asin, sqrt, pi

# speed of light
c = 299792458e-3 # in km/s

# reduced planck constant
hbar = 6.582119569e-22 # in MeV s

# hbar * c
hbarc = hbar*c # in MeV km

# erg / MeV
erg_MeV = 1e-7 / 1.602176634e-13

# Fermi coupling constant
Gf = 1.1663787e-5 * 1e-6 # in MeV^-2

# Avogadro's number
Na = 6.02214076e23 

# neutrino mixing angles
theta12 = sqrt(asin(0.307))
theta23 = sqrt(asin(0.54))
theta13 = sqrt(asin(0.022))

# neutrino mass splittings
dm21sqr = 7.53e-5 * 1e-12 # in MeV^2
dm32sqr = 2.45e-3 * 1e-12 # Normal order

# neutrino CP phase
delta_CP = 1.36*pi