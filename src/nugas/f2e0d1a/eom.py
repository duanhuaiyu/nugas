'''Python module dyn1d that provides the time derivative of the polarization vectors in the dynamic 1D model.
Author: Huaiyu Duan (UNM)
Description: 
'''
import numpy as np

def eom(z, u, g0, Dz="fd5"):
    '''Generates a function that compute the time derivative of the polarization vector.
    z[Nz] : NumPy array of the spatial mesh points
    u[Nu] : NumPy array of the angular mesh points
    g0[Nu] : P dot g0 gives the integral of P over u
    Dz : method to compute the derivative along z.
    return : dPdt(t, P)
    '''
    from ..misc.pdz import pdz
    g1 = u * g0
    Dz = pdz(z, Dz) # function to compute derivative
    Pshape = (len(z), 3, len(u)) # actual shape of polarization vector   

    def dPdt(t, P):
        '''Compute dP/dt.
        t : time
        P[3*Nu*Nz] : NumPy array of the polarization vector

        return : dP/dt
        '''
        ishape = P.shape
        P = P.reshape(Pshape)
        Ptot0 = P @ g0
        Ptot1 = P @ g1
        H = Ptot0[:,:,None] - u * Ptot1[:,:,None] # neutrino self-coupling Hamiltonian

        r = Dz(P) # ∂P/∂z 

        # compute time derivative: ∂P/∂t = H x P - u ∂P/∂z -> r
        r[:,0,:] = H[:,1,:] * P[:,2,:] - H[:,2,:] * P[:,1,:] - u * r[:,0,:]
        r[:,1,:] = H[:,2,:] * P[:,0,:] - H[:,0,:] * P[:,2,:] - u * r[:,1,:]
        r[:,2,:] = H[:,0,:] * P[:,1,:] - H[:,1,:] * P[:,0,:] - u * r[:,2,:]

        return r.reshape(ishape)

    return dPdt