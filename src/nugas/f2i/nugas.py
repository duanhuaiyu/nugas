'''Python package to compute 2-flavor oscillations in a uniform and isotropic neutrino gas.
Author: Huaiyu Duan (UNM)
'''
import numpy as np
from ..misc.misc import INTEGRATORS

_rtol = 1e-8 # default relative error tolerance
_atol = 1e-8 # default absolute error tolerance

class NuGas:
    '''Python object that describes a two-flavor, homogeneous and isotropic gas. 
    '''
    def __init__(self, t_ini=0, P_ini=[], omegas=(1, -1), weights=[], theta=0.15, imo=False, Hmat=None, mu=1, integrator="RK45", int_kargs={}):
        '''Initialization of the model.

        omegas[Nw] : a list of vaccum oscillation frequencies. omega is positive for a neutrino and negative for an antineutrino.
        weights[Nw] : a list of weights for the polarization vectors P. Sum of weights*P gives the total P. If not given, all elements are set to ±1 depending on the signs of omegas.
        t_ini : initial time. It is set to 0 by default.
        P_ini[Nw,3] : Initial polarization vectors. If not given, they are set to in the electron flavor.
        theta : mixing angle.
        imo : whether the neutrino has the inverted mass order. Default is False.
        Hmat : a number or function that gives the matter potential λ at time t. The polarization vectors are in the vacuum mass basis if it is None (default). Otherwise, they are in the flavor basis.
        mu : a number or function mu(t) that gives the neutrino potential μ at time t. If not given, it will be  1.
        integrator : SciPy integration method for time evolution. See nugas.misc.INTEGRATORS for the possible choices. The default is "RK45".
        int_kargs : a dictionary of keyword arguments to be passed to the integrator.
        '''
        # energy bins
        Nw = len(omegas) # number of energy bins
        assert Nw > 0, "Number of energy bins must be larger than 0"
        self.omegas = np.array(omegas, dtype=np.double) # save the omega's

        # weights for energy bins
        if len(weights) == 0: # default weights
            self.weights = np.sign(self.omegas)
        else:
            assert Nw == len(weights),"omega and weights should have the same length."
            self.weights = np.array(weights, dtype=np.double) # save the weights

        # initial time
        t_ini = float(t_ini) 

        # current polarization vectors    
        # Internally, P has shape (3, Nw) instead of (Nw, 3)
        from ..misc.misc import f2m, m2f
        if len(P_ini) == 0: # default initial P
            P_ini = np.empty((Nw, 3), dtype=np.double)
            P_ini[:,0] = P_ini[:,1] = 0
            P_ini[:,2] = 1
            if Hmat == None: P_ini = f2m(P_ini, theta) # use mass basis
        else:
            P_ini = np.array(P_ini, dtype=np.double) # save the initial P
            assert P_ini.shape == (Nw, 3), "P_ini has a wrong shape."

        # vaccum Hamiltonian
        self.Hvac = np.empty((3, Nw), dtype=np.double) # Hvac uses internal P shape
        self.Hvac[0] = self.Hvac[1] = 0
        self.Hvac[2] = self.omegas if imo else -self.omegas
        if Hmat != None: self.Hvac = m2f(self.Hvac, theta, faxis=0) # use flavor basis

        # matter potential
        if Hmat == None: 
            self.lam = None
        elif callable(Hmat):
            self.lam = Hmat
        else: # Hmat is a number
            self.lam = lambda t: float(Hmat) # save matter potential

        # determine the neutrino potential
        self.mu = mu if callable(mu) else lambda t: float(mu)

        # set up the integrator
        assert integrator in INTEGRATORS, f"Unknown integrator '{integrator}'. The available integrators are {list(INTEGRATORS.keys())}."
        if 'rtol' not in int_kargs: int_kargs['rtol'] = _rtol # default relative error tolerance
        if 'atol' not in int_kargs: int_kargs['atol'] = _atol # default absolute error tolerance
        dydx = lambda x, y: self._dPdt(x, y.reshape(self.Hvac.shape)).ravel()
        self.solver = INTEGRATORS[integrator](dydx, t_ini, P_ini.T.ravel(), t_ini, **int_kargs)

        # initial history
        self.t = np.array([t_ini])
        self.P = P_ini.reshape((1,Nw,3))


    def evolve(self, t):
        '''Evolve the system.
        t : a number or array of time(s) to reach. Must be larger than the last time the system evolved to (or the initial time if it has not been evolved.)

        return : (current time, current P)
        '''        
        try: # check if t is an array
            Nt = len(t)
            tt = np.array(t, dtype=np.double)
        except TypeError: # no, t is a number
            Nt = 1
            tt = np.array([float(t)])

        # enlarge history arrays
        ti0 = len(self.t) # length of old history
        Nt += ti0 # length of the new history
        t = np.empty(Nt, dtype=np.double)
        t[:ti0] = self.t # copy the old times
        t[ti0:] = tt # new times to reach
        P = np.empty((Nt, len(self.omegas), 3), dtype=np.double)
        P[:ti0] = self.P # copy the old history

        # compute
        for ti in range(ti0, Nt):
            t[ti], P[ti] = self._evolveTo(t[ti])

        self.t = t
        self.P = P

        return t[-1], P[-1]

    def _evolveTo(self, t):
        '''Evolve the system to time t.
        t : a number or array of time(s) to reach.

        return : (current time, current P)
        '''
        assert t > self.solver.t, f"Time {t} has reached."
        self.solver.t_bound = t # set the time to reach
        self.solver.status = 'running'
        # run the integrator to the new time
        while self.solver.status == 'running': 
            msg = self.solver.step()
            if self.solver.status == 'failed':
                raise Exception(msg)
            
        return self.solver.t, self.solver.y.reshape(self.Hvac.shape).T


    def _dPdt(self, t, P):
        '''Compute the time derivative of polarization vector P at time t.
        t : time.
        P[3,Nw] : polarization vectors.

        return : dP/dt of shape (3, Nw).
        '''
        # total polarization vector

        # vacuum Hamiltonian
        H = np.array(self.Hvac, dtype=np.double) 
        # add matter potential
        if self.lam != None: 
            H[2] += self.lam(t)
        # add neutrino potential
        D = (P @ self.weights) * self.mu(t) 
        H += D[:,None]

        # compute dP/dt
        r = np.empty(H.shape, dtype=np.double)
        r[0] = H[1]*P[2] - H[2]*P[1]
        r[1] = H[2]*P[0] - H[0]*P[2]
        r[2] = H[0]*P[1] - H[1]*P[0]

        return r
