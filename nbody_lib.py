######### HANDOUT 1
def periodic_boundary(x):
    #given an position, correctly
    #change it to assert periodic
    #boundary conditions
    if(x <= 0.):
        x += 1
    if(x >= 1.):
        x -= 1
    return x

def array_periodic_boundary(x):
    x[x>=1.] -=1.
    x[x<0.]  +=1.
    return x

def CIC_deposit(x,Ngrid=100):
    """
    x: position of particles in our dimensionless variable
        by definition this means that x in [0,1]
    Ngrid: number of grid elements for particle mesh

    returns rho which is a list size Ngrid that contains the mass density in a grid cell
    """
    dx = 1./Ngrid
    rho = np.zeros(Ngrid)
    m   = 1 / len(x) # assume all particles have the same mass
    
    left = x-0.5*dx
    right = left+dx
    xi = np.int32(left/dx)
    frac = (1.+xi-left/dx)
    ind = np.where(left<0.)
    frac[ind] = (-(left[ind]/dx))
    xi[ind] = Ngrid-1
    xir = xi.copy()+1
    xir[xir==Ngrid] = 0
    rho  = np.bincount(xi,  weights=frac*m, minlength=Ngrid)
    rho2 = np.bincount(xir, weights=(1.-frac)*m, minlength=Ngrid)

    rho += rho2
    
    return rho*Ngrid

import numpy as np
def solve_poisson(rho, a):
    Ngrid = len(rho)
    
    delta_l = np.fft.fft(rho - 1)
    k_ell   = 2 * np.pi * np.fft.fftfreq(Ngrid)

    phi_l = np.zeros_like(k_ell, dtype=np.complex128)
    phi_l[1:] = (- 3 / (8 * a * Ngrid**2) * delta_l[1:] / (np.sin(k_ell[1:] / 2)**2))
    phi_l[0] = 0

    phi_x = (np.fft.ifft(phi_l)).real
    return phi_x



def central_difference(y):
    return (np.roll(y,-1)-np.roll(y,1))/2

def rebin_grad_phi(x, grad_phi, Ngrid):
    left = x-0.5 / Ngrid
    xi = np.int64(left * Ngrid)
    frac = (1.+xi-left * Ngrid)
    return (frac)*(np.roll(grad_phi,0))[xi] + (1.-frac) * (np.roll(grad_phi,-1))[xi] 
