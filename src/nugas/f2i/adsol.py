'''Python module to compute adiabatic solutions for the 2-flavor oscillations in a uniform and isotropic neutrino gas.
Author: Huaiyu Duan (UNM)
'''
import numpy as np
from scipy.optimize import root

def adsol(D3, mu, omegas=(1, -1), weights=[], imo=False, D1_guess=None, Omega_guess=None, root_kargs={}):
    '''Compute the adiabatic solution.
    D3 : the 3rd component of the total polarization vector D in the mass basis.
    mu : a number or a list that gives the neutrino potential(s) μ.
    omegas[Nw] : a list of vaccum oscillation frequencies.
    weights[Nw] : a list of weights for the polarization vectors P. Sum of abs(weights)*P gives the total P. If not given, it is set as 1. The signs of weights determine the alignment of the polarization vectors.
    imo : whether the neutrino has the inverted mass order. Default is False.
    D1_guess : initial guess for the perpendicular component of the total polarization vector D in the mass basis.
    Omega_guess : initial guess for the collective frequency.
    root_kargs : dictionary of keyword arguments to be passed to scipy.optimize.root.

    return : (D1, Omega, P1, P3) for the adiabatic solution, where D1 and Omega are either numbers or arrays of the same length of mu, and P1 and P3 are arrays of shape (Nw,) or (len(mu), Nw).
    '''
    # energy bins
    Nw = len(omegas) # number of energy bins
    omegas = np.array(omegas, dtype=np.double)
    assert omegas.shape == (Nw,), "omegas has a wrong shape."
    if imo: omegas = -omegas # inverted mass ordering

    # weights for energy bins
    if len(weights) == 0: # default weights
        weights = np.ones(Nw, dtype=np.double)
    else:
        weights = np.array(weights, dtype=np.double)
        assert weights.shape == omegas.shape, "omegas and weights should have the same shape."

    if D1_guess == None: 
        # compute default D_perp
        tot = np.sum(weights)
        if tot == 0: # rare situation
            D1_guess = 0.1  
        else:
            P3 = abs(D3/tot)
            assert P3 <= 1, f"Value D3 = {D3} is not compatible with the weights specified."
            D1_guess = np.sqrt(1 - P3*P3) * tot

    # initial guess for collective frequency
    if Omega_guess == None:
        Omega_guess = omegas @ weights # use average omega as a guess

    try: # assume mu is an array
        Nm = len(mu) # number of solutions to compute
    except TypeError: # mu is a number
        Nm = 1

    if Nm == 1:
        D1, Omega, P1, P3 = _adsol_nomatter(D3, mu, omegas, weights, D1_guess, Omega_guess, root_kargs)
    else:
        P1 = np.empty((Nm, Nw), dtype=np.double) 
        P3 = np.empty((Nm, Nw), dtype=np.double) 
        Omega = np.empty(Nm, dtype=np.double)
        D1 = np.empty(Nm, dtype=np.double)
        for i, mu1 in enumerate(mu):
            D1_guess, Omega_guess, P1[i], P3[i] = _adsol_nomatter(D3, mu1, omegas, weights, D1_guess, Omega_guess, root_kargs)
            D1[i] = D1_guess; Omega[i] = Omega_guess

    return D1, Omega, P1, P3

def _Pad(D3, mu, omegas, weights, D1, Omega):
    '''Compute the polarization vectors in an adiabatic solution.
    D3 : the 3rd component of the total polarization vector in the mass basis.
    mu : a number that gives the neutrino potential μ. 
    omegas[Nw] : a list of vaccum oscillation frequencies.
    weights[Nw] : a list of weights for the polarization vectors P. Sum of weights*P gives the total P. 
    D1 : the perpendicular component of the total polarization vector D in the mass basis.
    Omega : the collective frequency.

    return : (P1[Nw], P3[Nw])
    '''
    P1 = mu * D1 * weights
    P3 = (-omegas + Omega + mu*D3) * weights
    Pnorm = np.sqrt(P1**2 + P3**2)
    return P1/Pnorm, P3/Pnorm

def _adsol_nomatter(D3, mu, omegas, weights, D1_guess, Omega_guess, root_kargs):
    '''Compute the adiabatic solution in the absence of matter.
    D3 : the 3rd component of the total polarization vector in the mass basis.
    mu : a number that gives the neutrino potential μ. 
    omegas[Nw] : a list of vaccum oscillation frequencies.
    weights[Nw] : a list of weights for the polarization vectors P. Sum of weights*P gives the total P. 
    D1_guess : initial guess for the perpendicular component of the total polarization vector D in the mass basis.
    Omega_guess : initial guess for the collective frequency.
    root_kargs : dictionary of keyword arguments to be passed to scipy.optimize.root.

    return : (D1, Omega, P1, P3) for the adiabatic solution.
    '''
    def f(x): 
        '''Function to solve x = [D1, Omega].'''
        H1 = mu * x[0]
        H3 = -omegas + x[1] + mu*D3
        Px = weights * mu / np.sqrt(H1**2 + H3**2)
        return np.sum(Px) - 1, (omegas @ Px) - x[1]

    sol = root(f, x0=(D1_guess, Omega_guess), **root_kargs)
    # check the solution
    assert sol.success, sol.message
    r = f(sol.x) # residue
    if np.sqrt(r[0]**2 + r[1]**2) >= 1e-4:
        print(f"The root {sol.x} produces residue {r} and may be invalid.")
    D1, Omega = sol.x
    P1, P3 = _Pad(D3, mu, omegas, weights, D1, Omega)
    return D1, Omega, P1, P3
