'''quadrature.py
Author: Huaiyu Duan (UNM)
Description: Python module that define the mesh and weights of various quadrature rules.
'''
import numpy as np

RULES = {} # dictionary of the quadrature rules

def mesh(a, b, n, rule='midpoint'):
    '''Compute the abscissas x[n] and weights w[n] using n nodes and the specified rule.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of nodes
    rule : name of the quadrature rule

    return : x[n], w[n]
    '''
    assert rule in RULES, f"Unknown quadrature rule '{rule}''. The available choices are {list(RULES)}."
    return RULES[rule](a, b, n)

def _midpoint(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using the (composite) midpoint rule.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    n = int(n)
    assert n >= 1, "Number of abscissas should be at least 1 for the midpoint rule."
    dx = (b - a) / n # mesh interval
    x = a + (np.arange(n) + 0.5) * dx
    w = np.ones_like(x) * dx
    return x, w
RULES['midpoint'] = _midpoint

def _trapezoid(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using the (composite) trapezoid rule.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    n = int(n)
    assert n >= 2, "Number of abscissas should be at least 2 for the trapezoid rule."
    x = np.linspace(a, b, n)
    w = np.ones_like(x) * (x[1] - x[0])
    w[0] *= 0.5; w[-1] *= 0.5
    return x, w
RULES['trapezoid'] = _trapezoid

def _simpson(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using the (composite) Simpson's rule.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    n = int(n)
    assert n >= 3 and n%2 ,  "The number of abscissas should be an odd number larger than or equal to 3 for Simpson's rule."
    x = np.linspace(a, b, n)
    w = np.ones_like(x) * (x[1] - x[0]) 
    w[0] *= 1/3; w[-1] *= 1/3
    w[1:-1:2] *= 4/3; w[2:-2:2] *= 2/3
    return x, w
RULES['simpson'] = _simpson

def _simpson2(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using an alternative Simpson's rule that works better for functions with narrow peaks.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    n = int(n)
    assert n >= 6,  "The number of abscissas should be at least 6 for the alternative Simpson's rule."
    x = np.linspace(a, b, n)
    w = np.ones_like(x) * (x[1] - x[0]) 
    w[0] *= 9/24; w[-1] *= 9/24
    w[1] *= 28/24; w[-2] *= 28/24
    w[2] *= 23/24; w[-3] *= 23/24
    return x, w
RULES['simpson2'] = _simpson2

def _chebyshev(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using the Chebyshev-Gaussian quadrature.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    from scipy.special import roots_chebyt
    y, w = roots_chebyt(n)
    x = (b-a)*0.5*np.array(y) + (b+a)*0.5 # transform from [-1, 1] to [a, b]
    w *= 0.5 * (b - a) * np.sqrt(1 - y**2)
    return x, w
#     # y = [np.cos(np.pi*(j+0.5)/n) for j in range(n)] # Chebyshev nodes
#     # x = (b-a)*0.5*np.array(y) + (b+a)*0.5 # transform from [-1, 1] to [a, b]
#     # w = [np.sqrt(1-y[j]**2) for j in range(n)]
#     # w = 0.5*(b-a)*(np.pi/n)*np.array(w) # weights
#     # return x, w
RULES['chebyshev'] = _chebyshev

def _legendre(a, b, n):
    '''Compute the mesh points x[n] and weights w[n] using the Legendre-Gaussian quadrature.
    The sum of f(x[i]) * w[i] gives an approximation of integral of f(x) for x from a to b.
    a : lower limit of the integral
    b : upper limit of the integral
    n : number of points

    return : x[n], w[n]
    '''
    from scipy.special import roots_legendre
    y, w = roots_legendre(n)
    x = (b-a)*0.5*np.array(y) + (b+a)*0.5 # transform from [-1, 1] to [a, b]
    w *= 0.5 * (b - a)
    return x, w
RULES['legendre'] = _legendre

