'''Python module to compute the flavor wave solution in 1D axially symmetric neutrino gases.
Author: Huaiyu Duan (UNM) 
'''

import numpy as np
from scipy import optimize

def DR_wv(N3, Ks, u, g0, alignment=+1, Omega_guess=None, F1_guess=None, F3_guess=None, ytol=1e-3, root_kargs={}):
    '''Solve the dispersion relation of the wave solution.
    N3 : lepton number density along the axis where it is constant.
    Ks : an array of wave numbers for which Omegas are to be found.
    u : angular bins.
    g0 : NumPy array of the weights time the ELN angular distribution. The sum of g0 * (polarization vectors) gives the total polarization vector.
    alignment : a number or array of +/-1 indicating whether the polarization vector is aligned or antialigned with the H vector.
    Omega_guess : an initial guess for Omega.
    F1_guess : an initial guess for F1 at Ks[0].
    F3_guess : an initial guess for F3 at Ks[0].
    ytol : tolerance for the root function residue.
    root_kargs : keyword arguments to be passed to scipy.optimize.root.
    
    return : (Ks, Omegas, F1s, F3s), where Omegas, F1s, and F3s are the solution parameters at Ks.
    '''
    Omegas = np.zeros_like(Ks)
    F1s = np.zeros_like(Ks)
    F3s = np.zeros_like(Ks)
    for i, K in enumerate(Ks):
        try:
            Omega_guess, F1_guess, F3_guess = wvsol(N3, K, u, g0, alignment, Omega_guess, F1_guess, F3_guess, calcP=False, ytol=ytol, root_kargs=root_kargs)
            Omegas[i], F1s[i], F3s[i] = Omega_guess, F1_guess, F3_guess
        except:
            print(f"Exception at i = {i}.")
            raise
    return Ks, Omegas, F1s, F3s

def wvsol(N3, K, u, g0, alignment=+1, Omega_guess=None, F1_guess=None, F3_guess=None, calcP=False, ytol=1e-3, root_kargs={}):
    '''Solve the wave solution with given N3 and K.
    N3 : lepton number density along the axis where it is constant.
    K : wave number.
    u : angular bins.
    g0 : NumPy array of the weights time the ELN angular distribution. The sum of g0 * (polarization vectors) gives the total polarization vector.
    alignment : a number or array of +/-1 indicating whether the polarization vector is aligned or antialigned with the H vector.
    Omega_guess : an initial guess for Omega.
    F1_guess : an initial guess for F1.
    F3_guess : an initial guess for F3.
    calcP: whether to return the polarization vectors. Default is True.
    ytol : tolerance for the root function residue.
    root_kargs : keyword arguments to be passed to scipy.optimize.root.

    return : (Omega, F1, F3, P1, P3) if calcP is True and (Omega, F1, F3) otherwise, where Omega is the frequency of the collective, P1 and P3 are NumPy arrays which contain the components of the polarization vectors that are normal and parallel to the conservation axis, respectively, and Fi = sum(g0*u*Pi) for i = 1, 3.
    '''
    # convert alignment to array
    alignment = np.array(alignment)
    if alignment.size == 1:
        alignment = np.ones_like(u) * alignment
    # make initial guesses
    if not Omega_guess:
        Omega_guess = K * 0.5
    if not F1_guess:
        F1_guess = 0.1
    if not F3_guess:
        F3_guess = 0.1

    def f(x):
        '''Function to be passed to root.
        x : [Omega, F1, F3]
        '''
        Omega, F1, F3 = x
        N1 = F1 * K / Omega
        coe = alignment*g0 / np.sqrt(((N3-Omega) - (F3-K)*u)**2 + (N1 - F1*u)**2)
        return [ N1 - coe @ (N1 - F1*u), 
            N3 - coe @ ((N3 - Omega) - (F3 - K)*u),
            F3 - (u * coe) @ ((N3 - Omega) - (F3 - K)*u) ]

    sol = optimize.root(f, x0=(Omega_guess, F1_guess, F3_guess), **root_kargs)
    # check the solution
    assert sol.success, sol.message
    r = f(sol.x) # residue
    if np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) > ytol:
        print(f"The root {sol.x} produces residue {r} and may be invalid.")
    Omega, F1, F3 = sol.x
    if calcP:
        P1, P3 = P_wv(Omega, K, F1, N3, F3, u, g0, alignment)
        return Omega, F1, F3, P1, P3
    else:
        return Omega, F1, F3
    
def P_wv(Omega, K, F1, N3, F3, u, g0, alignment=+1):
    '''Compute the polarization vectors for the wave solution.
    Omega : collective frequency.
    K : wave number.
    F1 : total flux in the flavor direction that is perpendicular to the conserved axis.
    N3 : total lepton number along the conserved axis.
    F3 : total flux along the conserved axis.
    u : angular bins.
    g0 : NumPy array of the weights time the ELN angular distribution. The sum of g0 * (polarization vectors) gives the total polarization vector.
    alignment : a number or array of +/-1 indicating whether the polarization vector is aligned or antialigned with the H vector.

    return : (P1, P3)     
    '''
    # convert alignment to array
    alignment = np.array(alignment)
    if alignment.size == 1:
        alignment = np.ones_like(u) * alignment
    N1 = F1 * K / Omega
    coe = alignment / np.sqrt(((N3-Omega) - (F3-K)*u)**2 + (N1 - F1*u)**2)
    P1 = coe * (N1 - F1*u)
    P3 = coe * ((N3-Omega) - (F3-K)*u)
    return P1, P3
