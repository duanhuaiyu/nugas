'''Dispersion relation for the symmetry preserving case
Author: Huaiyu Duan (UNM)
Ref: arXiv:1901.01546
'''
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, root_scalar, minimize_scalar


def _DR_func(K, Omega, G, int_opts={}):
    '''The function D(K, Omega) that should equal 0. See Eq.34 of Ref. 
    K : wave number
    Omega : wave frequency
    G(u) : ELN distribution function
    int_opts : dictionary of options to be passed on to the integrator

    return : value of the function
    '''
    kr, ki = K.real, K.imag
    wr, wi = Omega.real, Omega.imag
    fr = lambda p, u: G(u)*(wr-kr*u)* u**p/((wr-kr*u)**2 + (wi-ki*u)**2)
    fi = lambda p, u: G(u)*(-wi+ki*u)* u**p/((wr-kr*u)**2 + (wi-ki*u)**2)
    J = [] # integral I in Eq.34 of Ref.
    for p in range(3):
        r, *_ = quad(lambda u: fr(p, u), -1, 1, **int_opts)
        i, *_ = quad(lambda u: fi(p, u), -1, 1, **int_opts)
        J.append(complex(r, i))

    return (J[0]+1)*(J[2]-1) - J[1]**2



def _I_of_real_n(p, n, G, int_opts={}):
    '''Compute the integral of [G(u) u^p / (1 - n u)] for u in [-1, 1] with real n. See Eq.50 of Ref.
    p : integer
    n : real refractive index
    G(u) : ELN distribution function
    int_opts : dictionary of options to be passed on to the integrator

    return : integral value (real number)
    '''
    f = lambda u: G(u) * u**p / (1 - n*u)
    res, *_ = quad(f, -1, 1, **int_opts)
    return res



def _dI_dn(p, n, G, int_opts={}):
    '''Compute the derivative of the above integral with respect to n.
    p : integer
    n : real refractive index
    G(u) : ELN distribution function
    int_opts : dictionary of options to be passed on to the integrator

    return : real number
    '''
    f = lambda u: G(u) * u**(p+1) / (1 - n*u)**2
    res, *_ = quad(f, -1, 1, **int_opts)
    return res



def _Omega_of_real_n(n, G, sgn, int_opts={}):
    '''Compute the frequency Omega of the real dispersion relation. See Eq.49 of Ref.
    n : real refractive index
    G(u) : ELN distribution function
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator

    return : Omega, frequency of the wave
    '''
    sgn = np.sign(sgn)
    I = [_I_of_real_n(p, n, G, int_opts) for p in range(3)]
    D = 0 if sgn == 0 else (I[2] - I[0])**2 + 4*(I[2]*I[0] - I[1]**2)

    return 0.5*(I[2] - I[0] + sgn*np.sqrt(D))



def _dOmega_dn(n, G, sgn, int_opts={}):
    '''Calculate dOmega/dn of the real dispersion relation.
    n : refractive index
    G(u) : ELN distribution function for u in [-1, 1]
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator

    return : dOmega/dn
    '''
    sgn = np.sign(sgn)
    I = [_I_of_real_n(p, n, G, int_opts) for p in range(3)]
    dIdn = [_dI_dn(p, n, G, int_opts) for p in range(3)]

    D = (I[2] - I[0])**2 + 4*(I[2]*I[0] - I[1]**2)
    assert D > 0, f"Invalid n = {n} for real DR." 
    dDdn = 2*(I[2] - I[0])*(dIdn[2] - dIdn[0]) + 4*(dIdn[2]*I[0] + I[2]*dIdn[0] - 2*I[1]*dIdn[1])

    return 0.5*(dIdn[2] - dIdn[0]) + 0.25*sgn*dDdn/np.sqrt(D)
    


def _dK_dn(n, G, sgn, int_opts={}):
    '''Calculate dK/dn=Omega + n dOmega/dn of the real dispersion relation.
    n : refractive index
    G(u) : ELN distribution function for u in [-1, 1]
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator

    return : dK/dn
    '''
    sgn = np.sign(sgn)
    I = [_I_of_real_n(p, n, G, int_opts) for p in range(3)]
    dIdn = [_dI_dn(p, n, G, int_opts) for p in range(3)]

    D = (I[2] - I[0])**2 + 4*(I[2]*I[0] - I[1]**2)
    assert D > 0, f"Invalid n = {n} for real DR." 
    dDdn = 2*(I[2] - I[0])*(dIdn[2] - dIdn[0]) + 4*(dIdn[2]*I[0] + I[2]*dIdn[0] - 2*I[1]*dIdn[1])

    Omega = 0.5*(I[2] - I[0] + sgn*np.sqrt(D))
    dOmegadn = 0.5*(dIdn[2] - dIdn[0]) + 0.25*sgn*dDdn/np.sqrt(D)
    return Omega + n * dOmegadn



def _n_star(G, int_opts={}, eps=1e-5):
    '''Compute the critical value of refractive index n where the two real DR branches join. See Sec.III.C.2 of Ref.
    G(u) : ELN distribution function
    int_opts : dictionary of options to be passed on to the integrator
    eps : numerical error tolerance in n

    return : the critical value of n
    '''
    assert G(1)<0 and G(-1)>0, "There is no crossing."
    def Delta(n): # discriminant
        I = [_I_of_real_n(p, n, G, int_opts=int_opts) for  p in range(3)]
        return (I[2] - I[0])**2 + 4*(I[2]*I[0] - I[1]**2)

    # try to bracket the critical n between n0 and n1
    dn = 0.5 
    n0 = -1 + dn
    n1 = 1 - dn
    while Delta(n0) < 0: # make sure that n0 gives real DR
        if dn < eps:  # fail to bracket
            return None
        n1 = n0
        dn *= 0.5
        n0 = -1 + dn
    while Delta(n1) > 0: # make sure that n1 does NOT give real DR
        n0 = n1
        dn *= 0.5
        n1 = 1 - dn

    # find the critical n
    sol = root_scalar(Delta, bracket=(n0, n1))
    assert sol.converged, "Cannot find the critical n."
    return sol.root


def _n_of_real_K(K, G, sgn, int_opts={}, ns=None, Ks=None, eps=1e-5):
    '''Compute the refractive index n that corresponds to the wave number K on the real dispersion relation.
    K : wave number
    G(u) : ELN distribution function
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator
    ns : value of n where two real branches join
    Ks : value of K where two real branches join
    eps : numerical error tolerance in n

    return : n, refractive index
    '''
    if K == 0: return 0.0
    assert sgn != 0, "sgn should should be a nonreal number indicating the real branch."

    # try to bracket the desired value of n between n0 and nb
    if ns == None:
        n0 = 0; k0 = 0
        nb = np.sign(sgn * K) # bound of n        
    else:
        assert Ks != None, "Ks must be specified when ns is given."
        n0 = ns; k0 = Ks
        nb = -1
    n1 = (nb + n0) * 0.5
    k1 = n1 * _Omega_of_real_n(n1, G, sgn, int_opts=int_opts)
    while (k1 - K) * (k0 - K) > 0: # make sure k0 and k1 bracket K
        if abs(n0-nb) < eps:
            print(f"Cannot locate n exactly for K = {K} on the real branch. Use {nb} instead.")
            return nb
        n0 = n1; k0 = k1
        n1 = (nb + n0) * 0.5
        k1 = n1 * _Omega_of_real_n(n1, G, sgn, int_opts=int_opts)

    # find n
    if ns == None: ns = 2 # guard against the situation where n == ns in function f below
    f = lambda n: _Omega_of_real_n(n, G, sgn*(ns-n), int_opts=int_opts) * n - K
    sol = root_scalar(f, bracket=(n0, n1))
    assert sol.converged, f"Cannot find n for K = {K}."
    return sol.root


    
def _extremalOmega(G, sgn, int_opts={}, eps=1e-5):
    '''Find the extremal points of Omega(K) on the real dispersion relation.
    G(u) : ELN distribution function for u in [-1, 1]
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator
    eps : numerical error tolerance in n

    return : [n, K, Omega], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the extremal points when there is no crossing or tuples when there is crossing.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
 
    dOmega_dn = lambda n: _dOmega_dn(n, G, sgn, int_opts)   
    if G(1) >= 0: # no crossing, one minimum point on each real branch
        sol1 = root_scalar(dOmega_dn, bracket=(-1+eps, 1-eps))
        assert sol1.converged, "Couldn't find an extremal Omega point."
        n1 = sol1.root # value of n where Omega(K) is extreme
        w1 = _Omega_of_real_n(n1, G, sgn, int_opts)
        return [n1, n1*w1, w1] # return n, K, and Omega

    # crossing case
    ns = _n_star(G, int_opts=int_opts, eps=eps) # critical n where two real branches join
    if ns != None:
        sol = minimize_scalar(lambda n: -dOmega_dn(n)*sgn, bounds=(-1+eps, ns-eps*(ns+1)), method="bounded") 
        assert sol.success, "Couldn't find the extremal Omega points."
        n1 = sol.x # dOmega(n)/dn is extreme
        w1 = _Omega_of_real_n(n1, G, sgn, int_opts)
        val = dOmega_dn(n1)
        if abs(val) <= eps*abs(w1): # degenerate extremal points
            return [ [n1, n1], [w1*n1, w1*n1], [w1, w1]]
        elif val*sgn > 0: # two separate extremal points
            sol1 = root_scalar(dOmega_dn, bracket=(-1+eps, n1)) # find the minimum Omega point
            sol2 = root_scalar(dOmega_dn, bracket=(n1, ns-eps)) # find the maximum Omega point
            assert sol1.converged and sol2.converged, "Couldn't find both extremal points. Crossing may be too shallow."
            n1 = sol1.root; n2 = sol2.root # values of n where Omega(K) are extreme
            w1 = _Omega_of_real_n(n1, G, sgn, int_opts)
            w2 = _Omega_of_real_n(n2, G, sgn, int_opts)
            return [ [n1, n2], [w1*n1, w2*n2], [w1, w2]]

    # no extremal Omega points found
    return [None, None, None]



def _extremalK(G, int_opts={}, eps=1e-5):
    '''Find the extremal points of K(Omega) on the real dispersion relation.
    G(u) : ELN distribution function for u in [-1, 1]
    sgn : a number indicating whether the plus or minus branch will be calculated
    int_opts : dictionary of options to be passed on to the integrator
    eps : numerical error tolerance in n

    return : [n, K, Omega], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the extremal points.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    if G(1) >= 0: # no crossing
        return [None, None, None]
    ns = _n_star(G, int_opts=int_opts) # critical point where two real branches join
    if ns == None: # there is no real DR branch
        return [None, None, None]

    if ns > eps: # two extremal points on the plus and minus branches, respectively
        n_list = []; k_list = []; w_list = [] # n, K, and Omega where K(Omega) are extreme 
        for sgn in [+1, -1]: # search the plus and minus branches
            dK_dn = lambda n: _dK_dn(n, G, sgn, int_opts)
            sol = root_scalar(dK_dn, bracket=(-1+eps, ns-eps))
            assert sol.converged, f"Error in searching extremal K point on the {'+' if sgn >0 else '-'} branch."
            n = sol.root # value of n where K(Omega) is extreme
            w = _Omega_of_real_n(n, G, sgn, int_opts)
            n_list.append(n)
            w_list.append(w)
            k_list.append(n*w)
    else: # search the plus branch only
        dK_dn = lambda n: _dK_dn(n, G, +1, int_opts)
        if abs(ns) <= eps: # ns is about 0 and is an extremal point
            w2 = _Omega_of_real_n(0, G, 0, int_opts) # Omega at this point
            # find the other point
            sol1 = root_scalar(dK_dn, bracket=(-1+eps, -2*eps))
            assert sol1.converged, "Error in searching extremal K point on the + branch."
            n1 = sol1.root
            w1 = _Omega_of_real_n(n1, G, +1, int_opts)
            n_list = [n1, 0.]
            w_list = [w1, w2]
            k_list = [n1*w1, 0.]
        else:
            # check if  there are two extremal points on the + branch
            sol = minimize_scalar(dK_dn, bounds=(-1+eps, ns-eps), method='bounded')
            assert sol.success, "Error in searching extremal K points on the + branch."
            n1 = sol.x # value of n where dK/dn is minimum
            val = dK_dn(n1)
            if val > eps: # no extremal point
                n_list = k_list = w_list = None
            elif abs(val) <= eps: # degenerate extremal points
                w1 = _Omega_of_real_n(n1, G, +1, int_opts)
                n_list = [n1, n1]
                w_list = [w1, w1]
                k_list = [w1*n1, w1*n1]
            else: # two distinct extremal points
                sol1 = root_scalar(dK_dn, bracket=(-1+eps, n1-eps))
                sol2 = root_scalar(dK_dn, bracket=(n1+eps, ns-eps))
                assert sol1.converged and sol2.converged, "Error in searching extremal K points on the + branch."
                n_list = [sol1.root, sol2.root]
                w_list = [_Omega_of_real_n(n, G, +1, int_opts) for n in n_list]
                k_list = [w_list[i]*n_list[i] for i in range(2)]

    return [n_list, k_list, w_list] 



def _xing(G, int_opts={}):
    '''Find the crossing point of the ELN and the corresponding critical point in (K, Omega).
    G(u) : ELN distribution function for u in [-1, 1]
    int_opts : dictionary of options to be passed on to the integrator

    return : [(n, K, Omega), ...], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the critical point.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    if G(1) >= 0: # no crossing
        return [None, None, None]

    # find the crossing point
    sol = root_scalar(G, bracket=(-1, 1))
    assert sol.converged, "Cannot find the crossing point."
    ux = sol.root # ELN crossing point

    # K times integrals defined in Eq.36 in Ref.
    I0, *_ = quad(G, -1, 1, weight='cauchy', wvar=sol.root, **int_opts)
    I1, *_ = quad(lambda u: G(u)*u, -1, 1, weight='cauchy', wvar=sol.root, **int_opts)
    I2, *_ = quad(lambda u: G(u)*u*u, -1, 1, weight='cauchy', wvar=sol.root, **int_opts)
    D = (I2 - I0)**2 - 4*(I1*I1 - I0*I2)

    # critical points
    k_list = [ 0.5*(I0-I2+sgn*np.sqrt(D)) for sgn in [+1, -1]] # K
    n = 1/ux if ux != 0.0 else np.inf # refractive index
    w_list = [k*ux for k in k_list] # values of Omega
    return [ [n, n], k_list, w_list]


def _cplx_K_of_0(G, int_opts={}):
    '''Find the complex K with at Omega=0.
    G(u) : ELN distribution function for u in [-1, 1]
    int_opts : dictionary of options to be passed on to the integrator

    return : K
    '''
    # K times integrals defined in Eq.36 of Ref at Omega = 0
    I0, *_ = quad(G, -1, 1, weight='cauchy', wvar=0, **int_opts)
    I0 += 1j * np.pi * G(0)
    I1, *_ = quad(G, -1, 1, **int_opts)
    I2, *_ = quad(lambda u: G(u)*u, -1, 1, **int_opts)
    k_list = []
    D = (I2 - I0)**2 - 4*(I1*I1 - I0*I2)
    for sgn in [+1, -1]:
        k0 = 0.5*(I0-I2+sgn*np.sqrt(D))
        k_list.append(complex(k0.real, abs(k0.imag)))

    # k_list = [ 0.5*(I0-I2+sgn*np.sqrt(D)) for sgn in [+1, -1]]
    
    return k_list


def _cplx_K_of_Omega(Omega, G, K0=0.1j, int_opts={}, rt_opts={}):
    '''Find the complex K with given real Omega.
    Omega : wave frequency
    G(u) : ELN distribution function for u in [-1, 1]
    K0 : initial guess of K
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder

    return : K
    '''
    def eq(k): # equation for solving Omega
        res = _DR_func(complex(k[0], k[1]), Omega, G, int_opts)
        return res.real, res.imag
    sol = root(eq, x0=[K0.real, K0.imag], **rt_opts)
    assert sol.success, f"{sol.message} Couldn't find a complex K solution at Omega = {Omega}. Current solution is K = {complex(sol.x[0], abs(sol.x[1]))}."
    return complex(sol.x[0], abs(sol.x[1]))


def _cplx_K(Omega, G, K0=0.1j, int_opts={}, rt_opts={}):
    '''Find the complex K with given list of real Omegas.
    Omega[num] : NumPy array of wave frequencies
    G(u) : ELN distribution function for u in [-1, 1]
    K0 : initial guess of K
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder

    return : K[num]
    '''
    kk = np.empty(len(Omega), dtype=np.cdouble)
    for i in range(len(Omega)):
        kk[i] = K0 = _cplx_K_of_Omega(Omega[i], G, K0, int_opts=int_opts)
    return kk
    

def _cplx_Omega_of_K(K, G, Omega0=0.1j, int_opts={}, rt_opts={}):
    '''Find the complex Omega with given real K.
    K : wave number
    G(u) : ELN distribution function for u in [-1, 1]
    Omega0 : guess value of Omega
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder

    return : Omega
    '''
    def eq(w): # equation for solving Omega
        res = _DR_func(K, complex(w[0], w[1]), G, int_opts)
        return res.real, res.imag
    sol = root(eq, x0=[Omega0.real, Omega0.imag], **rt_opts)
    if not sol.success:
        print(f"WARNING: {sol.message} The complex Omega at K={K} may be off.")
    return complex(sol.x[0], abs(sol.x[1]))


def _cplx_Omega(K, G, Omega0=0.1j, int_opts={}, rt_opts={}):
    '''Find the complex Omega with given list of real K.
    K[num] : NumPy array of wave numbers
    G(u) : ELN distribution function for u in [-1, 1]
    Omega0 : guess value of Omega
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder

    return : Omega[num]
    '''
    ww = np.empty(len(K), dtype=np.cdouble)
    for i in range(len(K)):
        ww[i] = Omega0 = _cplx_Omega_of_K(K[i], G, Omega0, int_opts=int_opts)
    return ww


def DR_real(G, maxK=1, minK=-1, num_pts=100, int_opts={}, shift=True, eps=1e-5):
    '''Calculate the real dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    maxK : maximum value of wave number K to compute
    minK : minimum value of wave number K to compute
    num_pts : number of points to calculate
    int_opts : dictionary of options to be passed on to the integrator
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    eps : numerical error tolerance in n

    return : [(K[num_pts], Omega[num_pts]), ...] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."

    if not shift: # shift maxK and minK if necessary
        w0, *_ = quad(G, -1, 1, **int_opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **int_opts) # shift of the wave number
        maxK -= k0; minK -= k0
    
    dr = [] # dispersion relations
    if G(1) >= 0: # no crossing, two real branches
        for sgn in [+1, -1]:
            na = _n_of_real_K(minK, G, sgn, int_opts=int_opts, eps=eps)
            nb = _n_of_real_K(maxK, G, sgn, int_opts=int_opts, eps=eps)
            nn = np.linspace(na, nb, num_pts)
            ww = np.empty(num_pts, dtype=np.double)
            kk = np.empty(num_pts, dtype=np.double)
            kk[0] = minK; ww[0] = minK/na
            kk[-1] = maxK; ww[-1] = maxK/nb
            for i in range(1, num_pts-1):
                ww[i] = _Omega_of_real_n(nn[i], G, sgn, int_opts=int_opts)
                kk[i] = nn[i] * ww[i]
            dr.append((kk, ww))
    else: # crossing
        ns = _n_star(G, int_opts=int_opts, eps=eps) # critical n where two real branches join
        if (ns == None): return [] # deep crossing, no real DR
        ws = _Omega_of_real_n(ns, G, 0, int_opts=int_opts)
        ks = ns * ws
        for sgn, Km in zip([+1, -1], [minK, maxK]): # compute the two branches
            if (ks - Km)*sgn < 0: # no need to compute the branch
                continue
            nb = _n_of_real_K(Km, G, sgn, int_opts=int_opts, ns=ns, Ks=ks, eps=eps)
            nn = np.linspace(ns, nb, num_pts)
            ww = np.empty(num_pts, dtype=np.double)
            kk = np.empty(num_pts, dtype=np.double)
            kk[0] = ks; ww[0] = ws
            kk[-1] = Km; ww[-1] = Km/nb
            for i in range(1, num_pts-1):
                ww[i] = _Omega_of_real_n(nn[i], G, sgn, int_opts=int_opts)
                kk[i] = nn[i] * ww[i]
            dr.append((kk, ww))

    if not shift:  # shift the DR back if needed
        for kk, ww in dr:
            kk += k0
            ww += w0

    return dr

def _fix_K0(dr, br=(0, 1)):
    '''Because of the uncertainty in the branch cut of sqrt, the end points of the plus and minus complex-K DRs at Omega = 0 may be swapped. This function fixes the problem.
    dr : a list (K, Omega) with K and Omega being NumPy arrays
    br : the indices of the DR pairs to check/fix
    '''
    dr0 = dr[br[0]]; dr1 = dr[br[1]] # two complex-K branches to compare
    flag = 0
    if abs(dr0[0][-1]-dr1[0][-2]) < abs(dr0[0][-1]-dr0[0][-2]): flag += 1
    if abs(dr1[0][-1]-dr0[0][-2]) < abs(dr1[0][-1]-dr1[0][-2]): flag += 1
    if flag == 2: # looks like K0 points are swapped
        for i in range(2):
            dr0[i][-1], dr1[i][-1] = dr1[i][-1], dr0[i][-1]
    elif flag == 1:
        print("WARNING: K at Omega=0 on one of the complex-K Dr may be off.")

def DR_complexK(G, num_pts=100, int_opts={}, rt_opts={}, shift=True, eps=1e-5):
    '''Calculate the complex K dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    num_pts : number of points to calculate
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    eps : numerical error tolerance in n

    return : [(K[num_pts], Omega[num_pts]), ...] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    dr = [] # dispersion relations

    if G(1) >= 0: # no crossing, two complex K branches
        for sgn, k0 in zip([+1, -1], _cplx_K_of_0(G, int_opts=int_opts)):
            nc, kc, wc = _extremalOmega(G, sgn, int_opts=int_opts, eps=eps) # extremal Omega points on the real branch
            ww = np.linspace(wc, 0, num_pts)
            kk = np.empty(num_pts, dtype=np.cdouble)
            kk[0] = kc; kk[-1] = k0
            kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), int_opts=int_opts, rt_opts=rt_opts)
            dr.append((kk, ww))
        _fix_K0(dr) # fix possible end points error
    else: # crossing
        k0 = _cplx_K_of_0(G, int_opts) # complex K at Omega = 0
        nx, kx, wx = _xing(G, int_opts) # critical points related to crossing
        br = []
        for i, sgn in enumerate([+1, -1]):
            nc, kc, wc = _extremalOmega(G, sgn, int_opts=int_opts, eps=eps) # extremal Omega points on the real branch
            if nc == None: # deep crossing, no extremal Omega point
                assert wx[i] != 0.0, "Cannot handle the case where ELN crosses at u = 0."
                ww = np.linspace(wx[i], 0, num_pts)
                kk = np.empty(num_pts, dtype=np.cdouble)
                kk[0] = kx[i]; kk[-1] = k0[i]
                kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), int_opts=int_opts, rt_opts=rt_opts)
                dr.append((kk, ww))
                br.append(len(dr)-1)
            else: # moderate crossing, two complex K branches
                # first branch
                ww = np.linspace(wc[0], 0, num_pts)
                kk = np.empty(num_pts, dtype=np.cdouble)
                kk[0] = kc[0]; kk[-1] = k0[i]
                kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), int_opts=int_opts, rt_opts=rt_opts)
                dr.append((kk, ww))
                br.append(len(dr)-1)
                # second branch
                ww = np.linspace(wx[i], wc[1], num_pts)
                kk = np.empty(num_pts, dtype=np.cdouble)
                kk[0] = kx[i]; kk[-1] = kc[1]
                kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), int_opts=int_opts, rt_opts=rt_opts)
                dr.append((kk, ww))
        # fix possible end points error
        _fix_K0(dr, br)

    if not shift: # shift back Omega and K if needed
        w0, *_ = quad(G, -1, 1, **int_opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **int_opts) # shift of the wave number
        for kk, ww in dr:
            kk += k0
            ww += w0
    return dr


def DR_complexOmega(G, num_pts=100, int_opts={}, rt_opts={}, shift=True, eps=1e-5):
    '''Calculate the complex Omega dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    num_pts : number of points to calculate
    int_opts : dictionary of options to be passed on to the integrator
    rt_opts : dictionary of options to be passed on to the root finder
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    eps : a small number to guard against rare cases

    return : [(K[num_pts], Omega[num_pts]), ...] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    dr = [] # dispersion relations
    if G(1) < 0: # complex Omega branch exists only for crossing
        nx, kx, wx = _xing(G, int_opts=int_opts) # critical points related to crossing
        nc, kc, wc = _extremalK(G, int_opts=int_opts, eps=eps) # extremal K points on the real branch
        if nc == None: # two complex Omega branches have joined
            kk = np.linspace(kx[0], kx[1], num_pts)
            ww = np.empty(num_pts, dtype=np.cdouble)
            ww[0] = wx[0]; ww[-1] = wx[1]
            ww[1:-1] = _cplx_Omega(kk[1:-1], G, ww[0]*(1+0.1j)+eps*1j, int_opts=int_opts, rt_opts=rt_opts)
            dr.append((kk, ww))
        else: # two separate complex Omega branches
            for i in range(2):
                kk = np.linspace(kx[i], kc[i], num_pts)
                ww = np.empty(num_pts, dtype=np.cdouble)
                ww[0] = wx[i]; ww[-1] = wc[i]
                ww[1:-1] = _cplx_Omega(kk[1:-1], G, ww[0]*(1+0.1j), int_opts=int_opts, rt_opts=rt_opts)
                dr.append((kk, ww))

    if not shift: # shift back Omega and K if needed
        w0, *_ = quad(G, -1, 1, **int_opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **int_opts) # shift of the wave number
        for kk, ww in dr:
            kk += k0
            ww += w0

    return dr