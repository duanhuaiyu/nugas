'''Various definitions shared by different NuGas models.
Author: Huaiyu Duan
'''
import numpy as np
import numpy.linalg as la 
import scipy.integrate

INTEGRATORS = { # integrators in scipy.integrate
    'RK23': scipy.integrate.RK23,
    'RK45': scipy.integrate.RK45,
    'DOP853': scipy.integrate.DOP853,
    'Radau': scipy.integrate.Radau,
    'BDF': scipy.integrate.BDF,
    'LSODA': scipy.integrate.LSODA
    }

OLD_INTEGRATORS = [ # old scipy integrators
    'vode', 'zvode', 'lsoda', 'dopri5', 'dopr853'
] 

def Pnorm(P, faxis=-1):
    '''Compute the norm of the polarization vectors P.
    P : NumPy array of polarization vectors.
    faxis : the axis of P that corresponds to the flavor axis. Default is the last axis.
    '''
    assert P.shape[faxis] == 3, "Flavor axis must have length 3."
    return la.norm(P, axis=faxis)


def f2m(Qf, theta, faxis=-1):
    '''Change a flavor vector from the flavor basis to the mass basis.
    Qf : the flavor vector in the flavor basis.
    theta : mixing angle.
    faxis : the axis of Q that corresponds to the flavor axis. Default is the last axis.
    '''
    if not isinstance(Qf, np.ndarray): Qf = np.array(Qf) # convert to array if necessary
    assert Qf.shape[faxis] == 3, "Flavor axis must have length 3."
    Qm = np.empty(Qf.shape, dtype=np.double) # vector in the vacuum mass basis

    # get slice indices along the flavor axis
    if faxis < 0:
        faxis += Qf.ndim 
    def indx(i): # index generating function
        res = []
        for j in range(Qm.ndim):
            if j == faxis: res.append(i)
            else: res.append(slice(None))
        return tuple(res)
    i0 = indx(0); i1 = indx(1); i2 = indx(2)

    # make the transformation
    c2v = np.cos(2*theta); s2v = np.sin(2*theta)
    Qm[i0] = c2v*Qf[i0] + s2v*Qf[i2]
    Qm[i1] = Qf[i1]
    Qm[i2] = c2v*Qf[i2] - s2v*Qf[i0]

    return Qm

def m2f(Qm, theta, faxis=-1):
    '''Change a flavor vector from the mass basis to the flavor basis.
    Qm : the flavor vector in the mass basis.
    theta : mixing angle.
    faxis : the axis of Q that corresponds to the flavor axis. Default is the last axis.
    '''
    if not isinstance(Qm, np.ndarray): Qm = np.array(Qm) # convert to array if necessary
    assert Qm.shape[faxis] == 3, "Flavor axis must have length 3."
    Qf = np.empty(Qm.shape, dtype=np.double) # vector in the flavor mass basis

    # get slice indices along the flavor axis
    if faxis < 0:
        faxis += Qf.ndim 
    def indx(i): # index generating function
        res = []
        for j in range(Qm.ndim):
            if j == faxis: res.append(i)
            else: res.append(slice(None))
        return tuple(res)
    i0 = indx(0); i1 = indx(1); i2 = indx(2)
    
    # make the transformation
    c2v = np.cos(2*theta); s2v = np.sin(2*theta)
    Qf = np.empty(Qm.shape, dtype=np.double) # vector in the vacuum mass basis
    Qf[i0] = c2v*Qm[i0] - s2v*Qm[i2]
    Qf[i1] = Qm[i1]
    Qf[i2] = c2v*Qm[i2] + s2v*Qm[i0]

    return Qf


def pGaussian(x, width, x0=None, tol=1e-16):
    '''Approximate the Gaussian function exp(-(x-x0)**2/(2*width**2)) on [xmin, xmax) with a sum of periodic functions cos(2pi*m*(x-x0)/L).
    
    x: NumPy array of the position bins.
    x0: location of the peak. None implies x0 is at the center of the box L/2.
    tol: error tolerance

    return: NumPy array of length len(x) that approximates the desired Gaussian function.
    '''
    dx = x[1] - x[0] # size of x bins
    Nx = len(x) # number x bins
    L = dx * Nx # size of the periodic box
    width *= 2*np.pi/L # normalized width
    if x0 == None: x0 = L / 2 # middle of the box
    g = np.ones(Nx) # m=0
    coe = 1.
    m = 0 # moment index
    while coe > tol and m < Nx:
        m += 1
        coe = 2. * np.exp(-0.5*(m*width)**2) # coefficient 
        g += coe * np.cos(2*m*np.pi*(x-x0)/L)
    return g*width/np.sqrt(2.*np.pi)

def logger(log_file):
    '''Generate a function that log messages.
    log_file : name of the file to record log message.
    return : function log_message(msg) that write msg to log_file.
    '''
    def log_message(msg):
        if type(log_file) == str: # given a file name
            with open(log_file, 'a') as output:
                output.write(msg)
        else: # assume it is a file handle
            print(msg, end='', file=log_file)
    return log_message