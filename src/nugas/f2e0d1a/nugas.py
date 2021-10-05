'''Python package to compute 2-flavor oscillations in 1D axially symmetric neutrino gases.
Author: Huaiyu Duan (UNM)
'''

import numpy as np
import sys

from ..misc.misc import INTEGRATORS, OLD_INTEGRATORS, logger
_rtol = 1e-8 # default relative error tolerance
_atol = 1e-8 # default absolute error tolerance

def _X(P):
    '''Change the shape of the polarization vectors from the shape used by the solver (Nz, 3, Nu) to that used by the interface and IO (..., Nu, 3).
    '''
    return P.swapaxes(1, 2)
        
class NuGas:
    _single_odes = { # flags for ODE solvers that allow only one instance
        'vode' : False,
        'zvode': False,
        'lsoda': False
    }

    def __init__(self, t=0., P=None, z=None, u=None, weights=None, g=None, pdz="fd5", integrator="RK45", int_kargs={}, data_file=None, clobber=False, load=False, attrs={}, eom_c=False, log_file=sys.stdout):
        '''Initialize a calculation.
        t : initial time.
        P[Nz,3,Nu] : NumPy array of the initial polarization vector.
        z[Nz] : NumPy array of the spatial mesh.
        u[Nu] : NumPy array of the angular mesh.
        weights[Nu] : NumPy array of the angular weights; sum of f(u)*weights gives the integral of f.
        g [Nu] : ELN distribution.
        pdz : method to compute spatial derivative.
        integrator : integration method for time evolution.
        int_kargs : keyword arguments to be passed to the integrator. Default is empty.
        data_file : name of the file to store flavor history.
        clobber : whether to overwite the existing file if starting a new history. Default is False.
        load : whether to load an existing history and append to it. Default is False.
        attrs : a dictionary of attributes to be save to the data file.
        eom_c : whether to use C++ version of eom. Default is False.
        log_file : name of the file to record log message. Default is sys.stdout.
        '''
        if load: # load and append to an existing history
            assert data_file, "Must supply a file name to load an existing history."
            t, P, dt = self._load(data_file)

        else: # start a new flavor history
            t, P, dt = self._new(t, P, z, u, weights, g, data_file, clobber, attrs)

        self._logger = logger(log_file)
        # set up the integrator
        if 'rtol' not in int_kargs: int_kargs['rtol'] = _rtol # default relative error tolerance
        if 'atol' not in int_kargs: int_kargs['atol'] = _atol # default absolute error tolerance
        if dt: int_kargs['dt'] = dt # initial step size
        self.integrator = integrator
        if integrator == 'lax42': # use Lax4 algorithm
            from .lax42 import Lax42
            self._solver = Lax42(t, _X(P), self.z, self.u, self.g0, **int_kargs);
        else: # use SciPy integrators
            # define the function to be passed on to the scipy integrator
            if eom_c:
                from .eom_c import eom
            else:
                from .eom import eom
            dPdt = eom(self.z, self.u, self.g0, pdz)

            if integrator in OLD_INTEGRATORS: # old SciPy ODE solver
                if integrator in self._single_odes: # check single-entrant solver
                    if self._single_odes[integrator]: # check if the solver is being used
                        raise Exception(f"ODE solver '{integrator}' is not re-entrant. Delete the old NuGas object to use it again.")
                    else: 
                        self._single_odes[integrator] = True # set the flag                                
                from scipy.integrate import ode
                self._solver = ode(dPdt)
                if integrator == 'RK45': self.integrator = 'dopri5'
                self._solver.set_integrator(self.integrator, **int_kargs)
                self._solver.set_initial_value(_X(P).ravel(), t)
            else: # new SciPy ODE solver
                assert integrator in INTEGRATORS, f"The integrator {integrator} is not defined."
                self._solver = INTEGRATORS[integrator](dPdt, t, _X(P).ravel(), t, **int_kargs)

    def evolve(self, t, progress=False, flush_int=0):
        '''Evolve the system.
        t : a number or array of time(s) to reach. Must be larger than the last time the system evolved to (or the initial time if it has not been evolved.)
        progress : whether to log the progress. Default is False.
        flush_int : Sync after these many snapshots. A value 0 or negative means no intermediate sync.

        return : (current time, current P)
        '''        
        try: # check if t is an array
            Nt = len(t)
            tt = np.array(t, dtype=np.double)
        except TypeError: # no, t is a number
            Nt = 1
            tt = np.array([float(t)])

        if progress: self._logger(f"{self.t[-1]:.2f}, ")
        for ti in range(Nt):
            t, P, dt = self._evolveTo(tt[ti])
            if self._history: # save history
                if flush_int > 0 and (ti+1)%flush_int == 0:
                    flush = True
                else:
                    flush = False
                self._history.addSnapshot({'t': t, 'P': P, 'dt': dt}, flush)
            if progress: 
                if self.integrator == 'lax42':
                    stepinfo = f", {self._solver.success_steps}/{self._solver.total_steps}"
                else: stepinfo = ""
                self._logger(f"{t:.2f} ({dt:g}{stepinfo}), ")

        if self._history:
            self._history.flush()
        else: # no history
            self.t[0] = t
            self.P = P.reshape((1, self.z.size, self.u.size, 3))
        
        if progress: 
            self._logger("Done.\n")
            if self.integrator == 'lax42':
                msg = f"The total number of steps and successful steps are {self._solver.total_steps} and {self._solver.success_steps}.\n"
                self._logger(msg)

        return t, P

    def _evolveTo(self, t):
        '''Evolve the system to time t.
        t : a number or array of time(s) to reach.

        return : (current time, current P in external format)
        '''
        assert t > self._solver.t, f"Time {t} has been reached."
        if self.integrator == 'lax42':
            t, P, dt = self._solver.integrate(t)
        elif self.integrator in OLD_INTEGRATORS: # use older SciPy ODE solver
            P = self._solver.integrate(t).reshape((self.z.size, 3, self.u.size))
            dt = 0.
            assert self._solver.successful(), f"ODE returns with error code {self._solver.get_return_code()}."
        else: # use newer SciPy ODE solver
            self._solver.t_bound = t # set the time to reach
            self._solver.status = 'running'
            # run the integrator to the new time
            while self._solver.status == 'running': 
                msg = self._solver.step()
                if self._solver.status == 'failed':
                    raise Exception(msg)
            # current time and polarization
            t = self._solver.t
            P = self._solver.y.reshape((self.z.size, 3, self.u.size))
            dt = self._solver.step_size
        P = _X(P) # change P to the external format

        return t, P, dt

    def _load(self, data_file):
        '''Initialize the model from a data file.
        data_file : name of the data file.

        return : (t, P), the last time and polarization, and the number of points for differencing.
        '''
        from ..misc.ionetcdf import FlavorHistory
        self._history = FlavorHistory(data_file, load=True)
        self.t = self._history.data.variables['t'] # time points
        t = self.t[-1] # current time
        dt = self._history.data.variables['dt'][-1]
        if dt <= 0: dt = None
        self.P = self._history.data.variables['P'] # time points
        P = np.array(self.P[-1]) # current P
        self.z = np.array(self._history.data.variables['z'][:]) # z mesh points
        self.u = np.array(self._history.data.variables['u'][:]) # angular mesh points
        self.g = np.array(self._history.data.variables['g'][:]) # ELN
        self.g0 = np.array(self._history.data.variables['g0'][:]) # angle weights
        return t, P, dt

    def _new(self, t, P, z, u, weights, g, data_file, clobber, attrs):
        '''Initialize the model from input parameters. See __init__() for the meanings of the arguments.
        return : t, P, dt
        '''
        self.z = np.array(z, dtype=np.double)
        assert self.z.ndim == 1, "Spatial mesh z must be a 1D array."
        self.u = np.array(u, dtype=np.double)
        assert self.u.ndim == 1, "Angular mesh u must be a 1D array."
        self.g = np.array(g, dtype=np.double)
        assert self.u.shape == self.g.shape, "ELN has a wrong shape."
        weights = np.array(weights, dtype=np.double)
        assert self.u.shape == weights.shape, "Angular weights has a wrong shape."
        self.g0 = self.g * weights
        P = np.array(P, dtype=np.double) # initial P in external shape
        assert P.shape == (len(z), len(u), 3), "P has a wrong shape."
        t = float(t) # initial t

        if data_file == None:
            self._history = None # don't save flavor history
            self.t = np.array([t]) # keep the last point
            self.P = P.reshape((1, len(z), len(u), 3))
        else: # create the data file to store history
            from ..misc.ionetcdf import FlavorHistory
            dims = { # dimensions of the variables
                't' : None, # time dimension, unlimited
                'z' : len(z), # spatial dimension
                'u' : len(u), # angular dimension
                'i' : 3 # index for flavor vector
            }
            vars = { # variables
                't' : {'type': 'f8', 'dimensions': ('t',)}, # time 
                'dt' : {'type': 'f8', 'dimensions': ('t',)}, # step size 
                'z' : {'type': 'f8', 'dimensions': ('z',)}, # space
                'u' : {'type': 'f8', 'dimensions': ('u',)}, # cos(theta)
                'g' : {'type': 'f8', 'dimensions': ('u',)}, # ELN
                'g0' : {'type': 'f8', 'dimensions': ('u',)}, 
                'P' : {'type': 'f8', 'dimensions': ('t', 'z', 'u', 'i')} # polarization
            }
            vars_ini = { # initial values
                'z': self.z, 'u' : self.u, 'g': self.g, 'g0': self.g0, 
                't': t, 'dt': 0., 'P': P               
            }
            self._history = FlavorHistory(data_file, clobber=clobber, dim=dims, var=vars, var_ini=vars_ini, attr=attrs)
            self.t = self._history.data.variables['t']
            self.P = self._history.data.variables['P']
        return t, P, None

    def __del__(self):
        try: 
            if self._fast_ode in self._single_odes:
                self._single_odes[self._fast_ode] = False # reset single-entrant solver flag
            del(self._history)
            del(self._solver)
        except:
            pass