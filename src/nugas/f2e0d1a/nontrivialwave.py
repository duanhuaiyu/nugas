'''Python module to compute the flavor wave solution in 1D axially symmetric neutrino gases.
Author: Huaiyu Duan (UNM) 
'''

import numpy as np
from scipy import optimize, integrate

def DR_wv(N3, Ks, G, alignment=+1, Omega_guess=None, F1_guess=None, F3_guess=None, ytol=1e-3, int_kargs={}, root_kargs={}):
    '''Solve the wave solution with given N3 and K.
    N3 : lepton number density along the axis where it is constant.
    Ks : an array of wave numbers for which Omegas are to be found.
    G(u) : ELN function.
    alignment : +/-1 indicating whether the polarization vector is aligned or antialigned with the H vector.
    Omega_guess : an initial guess for Omega.
    F1_guess : an initial guess for F1.
    F3_guess : an initial guess for F3.
    ytol : tolerance for the root function residue.
    int_kargs : keyword arguments to be passed to scipy.integrate.quad
    root_kargs : keyword arguments to be passed to scipy.optimize.root.
    
    return : (Ks, Omegas, F1s, F3s), where Omegas, F1s, and F3s are the solution parameters at Ks.
    '''
    Omegas = np.zeros_like(Ks)
    F1s = np.zeros_like(Ks)
    F3s = np.zeros_like(Ks)
    for i, K in enumerate(Ks):
        try:
            Omega_guess, F1_guess, F3_guess = calcWave(N3, K, G, alignment, Omega_guess, F1_guess, F3_guess, ytol, int_kargs, root_kargs)
            Omegas[i], F1s[i], F3s[i] = Omega_guess, F1_guess, F3_guess
        except:
            print(f"Exception at i = {i}. Returned arrays are truncated.")
            break
    return Ks[:i], Omegas[:i], F1s[:i], F3s[:i]

def calcWave(N3, K, G, alignment=+1, Omega_guess=None, F1_guess=None, F3_guess=None, ytol=1e-3, int_kargs={}, root_kargs={}):
    '''Solve the wave solution with given N3 and K.
    N3 : lepton number density along the axis where it is constant.
    K : wave number.
    G(u) : ELN function.
    alignment : +/-1 indicating whether the polarization vector is aligned or antialigned with the H vector.
    Omega_guess : an initial guess for Omega.
    F1_guess : an initial guess for F1.
    F3_guess : an initial guess for F3.
    ytol : tolerance for the root function residue.
    int_kargs : keyword arguments to be passed to scipy.integrate.quad
    root_kargs : keyword arguments to be passed to scipy.optimize.root.

    return : (Omega, F1, F3), where Omega is the frequency of the collective, and Fi = integrate(G*u*Pi, (u,-1,1)) for i = 1, 3.
    '''
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
        return [ 
            N1 - _calcN1(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs),
            N3 - _calcN3(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs),
            F3 - _calcF3(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs)
        ]

    sol = optimize.root(f, x0=(Omega_guess, F1_guess, F3_guess), **root_kargs)
    # check the solution
    assert sol.success, sol.message
    r = f(sol.x) # residue
    if np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) > ytol:
        print(f"The root {sol.x} produces residue {r} and may be invalid.")

    return sol.x

def calcP1P3(u, K, Omega, N3, F1, F3, alignment):
    '''Compute the Bloch vectors of the nontrivial wave solution.
    u : z velocity component of the neutrino.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.

    return : (P1, P3) in the wave solution.
    '''
    N1 = F1 * K / Omega
    return _calcP1(u, K, Omega, N1, N3, F1, F3, alignment), _calcP3(u, K, Omega, N1, N3, F1, F3, alignment)

def _calcN1(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs={}):
    '''Compute N1 in the wave solution.
    G : ELN distribution.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.
    int_kargs : keyword arguments to be passed to scipy.integrate.quad.

    return : actual N1.
    '''
    f = lambda u: _calcP1(u, K, Omega, N1, N3, F1, F3, alignment) * G(u)
    res = integrate.quad(f, -1, 1, **int_kargs)
    return res[0]

def _calcN3(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs={}):
    '''Compute N3 in the wave solution.
    G : ELN distribution.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.
    int_kargs : keyword arguments to be passed to scipy.integrate.quad.

    return : actual N3.
    '''
    f = lambda u: _calcP3(u, K, Omega, N1, N3, F1, F3, alignment) * G(u)
    res = integrate.quad(f, -1, 1, **int_kargs)
    return res[0]

def _calcF3(G, K, Omega, N1, N3, F1, F3, alignment, int_kargs={}):
    '''Compute F3 in the wave solution.
    G : ELN distribution.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.
    int_kargs : keyword arguments to be passed to scipy.integrate.quad.

    return : actual F3.
    '''
    f = lambda u: _calcP3(u, K, Omega, N1, N3, F1, F3, alignment) * G(u) * u
    res = integrate.quad(f, -1, 1, **int_kargs)
    return res[0]

def _calcP1(u, K, Omega, N1, N3, F1, F3, alignment):
    '''Compute P1 in the wave solution.
    u : z velocity component of the neutrino.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.

    return : P1 in the wave solution.
    '''
    p3 = (N3 - Omega) - (F3 - K)*u
    p1 = N1 - F1*u
    return p1/np.sqrt(p1**2 + p3**2)*alignment

def _calcP3(u, K, Omega, N1, N3, F1, F3, alignment):
    '''Compute P3 in the wave solution.
    u : z velocity component of the neutrino.
    K : wave number.
    Omega : frequency.
    N1 : integral of P1.
    N3 : integral of P3.
    F1 : integral of u*P1.
    F3 : integral of u*P3.
    alignment : +/-1.

    return : P3 in the wave solution.
    '''
    p3 = (N3 - Omega) - (F3 - K)*u
    p1 = N1 - F1*u
    return p3/np.sqrt(p1**2 + p3**2)*alignment
