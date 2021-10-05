'''Python module that approximates the first order derivatives on a periodic box using the central finite difference. 
Author: Huaiyu Duan (UNM)
'''
import numpy as np

DIFFERENTIATORS = {}

def pdz(x, method='fd5'):
    '''Generate a function that computes the first order derivative of a function on a periodic box with equal spaced mesh x. 
    x[Nx] : NumPy array of the spatial mesh points.
    method : method used to compute the derivative. Default is 'fd5'.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    assert method in DIFFERENTIATORS, f"Unknown differentiation method '{method}'. The available choices are {list(DIFFERENTIATORS)}."
    assert len(x)>1, "The number of the mesh points must be larger than 1."
    return DIFFERENTIATORS[method](x)

def _d3p(x):
    '''First order derivative with 3 points.
    x[Nx] : NumPy array of the spatial mesh points.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    Nx = len(x)
    assert Nx > 2, "The length of the x array must be larger than 2."
    dx = x[1] - x[0]
    coe = 0.5 / dx
    def Dz(y):
        assert y.size%Nx == 0, "The shape of y does not match that of x."
        dy = np.empty(y.shape) # derivative
        y = y.reshape((Nx, y.size//Nx))
        dydx = dy.reshape(y.shape)
        dydx[0] = coe * (-y[-1] + y[1])
        dydx[-1] = coe * (-y[-2] + y[0])
        dydx[1:-1] = coe * (-y[:-2] + y[2:])
        return dy
    return Dz
DIFFERENTIATORS['fd3'] = _d3p

def _d5p(x):
    '''First order derivative with 5 points.
    x[Nx] : NumPy array of the spatial mesh points.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    Nx = len(x)
    assert Nx > 4, "The length of the x array must be larger than 4."
    dx = x[1] - x[0]
    coe = 1. / (12. * dx)
    def Dz(y):
        assert y.size%Nx == 0, "The shape of y does not match that of x."
        dy = np.empty(y.shape) # derivative
        y = y.reshape((Nx, y.size//Nx))
        dydx = dy.reshape(y.shape)
        dydx[0] = coe * (y[-2] - 8.*y[-1] + 8.*y[1] - y[2])
        dydx[1] = coe * (y[-1] - 8.*y[0] + 8.*y[2] - y[3])
        dydx[-1] = coe * (y[-3] - 8.*y[-2] + 8.*y[0] - y[1])
        dydx[-2] = coe * (y[-4] - 8.*y[-3] + 8.*y[-1] - y[0])
        dydx[2:-2] = coe * (y[:-4] - 8.*y[1:-3] + 8.*y[3:-1] - y[4:])
        return dy
    return Dz
DIFFERENTIATORS['fd5'] = _d5p

def _d7p(x):
    '''First order derivative with 7 points.
    x[Nx] : NumPy array of the spatial mesh points.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    Nx = len(x)
    assert Nx > 6, "The length of the x array must be larger than 6."
    dx = x[1] - x[0]
    coe = 1. / (60. * dx)
    def Dz(y):
        assert y.size%Nx == 0, "The shape of y does not match that of x."
        dy = np.empty(y.shape) # derivative
        y = y.reshape((Nx, y.size//Nx))
        dydx = dy.reshape(y.shape)
        dydx[0] = coe * (-y[-3] + 9.*y[-2] - 45.*y[-1] + 45.*y[1] - 9.*y[2] + y[3])
        dydx[1] = coe * (-y[-2] + 9.*y[-1] - 45.*y[0] + 45.*y[2] - 9.*y[3] + y[4])
        dydx[2] = coe * (-y[-1] + 9.*y[0] - 45.*y[1] + 45.*y[3] - 9.*y[4] + y[5])
        dydx[-1] = coe * (-y[-4] + 9.*y[-3] - 45.*y[-2] + 45.*y[0] - 9.*y[1] + y[2])
        dydx[-2] = coe * (-y[-5] + 9.*y[-4] - 45.*y[-3] + 45.*y[-1] - 9.*y[0] + y[1])
        dydx[-3] = coe * (-y[-6] + 9.*y[-5] - 45.*y[-4] + 45.*y[-2] - 9.*y[-1] + y[0])
        dydx[3:-3] = coe * (-y[:-6] + 9.*y[1:-5] - 45.*y[2:-4] + 45.*y[4:-2] - 9.*y[5:-1] + y[6:])
        return dy
    return Dz
DIFFERENTIATORS['fd7'] = _d7p

def _d9p(x):
    '''First order derivative with 9 points.
    x[Nx] : NumPy array of the spatial mesh points.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    Nx = len(x)
    assert Nx > 8, "The length of the x array must be larger than 8."
    dx = x[1] - x[0]
    coe = 1. / (840. * dx)
    def Dz(y):
        assert y.size%Nx == 0, "The shape of y does not match that of x."
        dy = np.empty(y.shape) # derivative
        y = y.reshape((Nx, y.size//Nx))
        dydx = dy.reshape(y.shape)
        dydx[0] = coe * (3.*y[-4] - 32.*y[-3] + 168.*y[-2] - 672.*y[-1] + 672.*y[1] -168.*y[2] + 32.*y[3] - 3.*y[4])
        dydx[1] = coe * (3.*y[-3] - 32.*y[-2] + 168.*y[-1] - 672.*y[0] + 672.*y[2] -168.*y[3] + 32.*y[4] - 3.*y[5])
        dydx[2] = coe * (3.*y[-2] - 32.*y[-1] + 168.*y[0] - 672.*y[1] + 672.*y[3] -168.*y[4] + 32.*y[5] - 3.*y[6])
        dydx[3] = coe * (3.*y[-1] - 32.*y[0] + 168.*y[1] - 672.*y[2] + 672.*y[4] -168.*y[5] + 32.*y[6] - 3.*y[7])
        dydx[-1] = coe * (3.*y[-5] - 32.*y[-4] + 168.*y[-3] - 672.*y[-2] + 672.*y[0] -168.*y[1] + 32.*y[2] - 3.*y[3])
        dydx[-2] = coe * (3.*y[-6] - 32.*y[-5] + 168.*y[-4] - 672.*y[-3] + 672.*y[-1] -168.*y[0] + 32.*y[1] - 3.*y[2])
        dydx[-3] = coe * (3.*y[-7] - 32.*y[-6] + 168.*y[-5] - 672.*y[-4] + 672.*y[-2] -168.*y[-1] + 32.*y[0] - 3.*y[1])
        dydx[-4] = coe * (3.*y[-8] - 32.*y[-7] + 168.*y[-6] - 672.*y[-5] + 672.*y[-3] -168.*y[-2] + 32.*y[-1] - 3.*y[0])
        dydx[4:-4] = coe * (3.*y[:-8] - 32.*y[1:-7] + 168.*y[2:-6] - 672.*y[3:-5] + 672.*y[5:-3] -168.*y[6:-2] + 32.*y[7:-1] - 3.*y[8:])
        return dy
    return Dz
DIFFERENTIATORS['fd9'] = _d9p

def _pdz_fft(x):
    '''First order derivative with fft.
    x[Nx] : NumPy array of the spatial mesh points.
    return: dydx(y), where y[Nx] are the function values on x[Nx].
    '''
    from scipy.fft import rfft, irfft, rfftfreq
    from numpy import pi
    Nx = len(x) # number of the mesh points
    assert not Nx%2 and Nx > 1, "The length of the x array must be an even number larger than 1."
    k = 2.j * pi * rfftfreq(Nx, x[1]-x[0]) # wave number times j
    def Dz(y):
        assert y.size%Nx == 0, "The shape of y does not match that of x."
        yshape = y.shape
        y = y.reshape((Nx, y.size//Nx))
        return irfft(k[:,None]*rfft(y, axis=0), axis=0).reshape(yshape)
    
    return Dz
DIFFERENTIATORS['fft'] = _pdz_fft
