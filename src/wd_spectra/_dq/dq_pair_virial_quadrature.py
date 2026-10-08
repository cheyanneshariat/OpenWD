"""Vectorized radial quadrature of the unchanged published virial inputs.

No temperature table or interpolation is introduced. Every temperature
evaluates its actual Boltzmann factor. Each PCHIP interval is integrated
separately; r=R_last/u maps the complete long-range tail to 0<u<1.
Order doubling checks numerical accuracy and fails explicitly if unresolved.
"""
from functools import lru_cache
import numpy as np
from .dq_dense_continuum_inputs import _CORE, _R, neutral_potential_hartree, pair_response
from wd_spectra.constants import BOLTZMANN
from wd_spectra.dense_helium_continuum import HARTREE_ERG


@lru_cache(maxsize=24)
def nodes(order,short_range):
    x,w=np.polynomial.legendre.leggauss(order)
    edges=np.r_[_CORE,_R]
    if np.any(np.diff(edges)<=0):raise ValueError('Virial intervals must be ordered')
    half=.5*np.diff(edges);mid=.5*(edges[1:]+edges[:-1])
    r=(mid[:,None]+half[:,None]*x).ravel()
    weights=(half[:,None]*w).ravel()
    u=.5*(x+1);last=edges[-1]
    r=np.r_[r,last/u]
    weights=np.r_[weights,.5*w*last/u**2]
    potential=neutral_potential_hartree(r)
    coefficients=np.asarray(pair_response(r,short_range=short_range))*r[None,:]**2*weights[None,:]
    potential.flags.writeable=False;coefficients.flags.writeable=False
    return potential,coefficients


def fixed_order(temperature,order,short_range):
    potential,coefficients=nodes(order,short_range)
    t=np.asarray(temperature,float)
    thermal=t.ravel()*BOLTZMANN/HARTREE_ERG
    with np.errstate(over='raise',invalid='raise'):
        result=np.exp(-potential[None,:]/thermal[:,None])@coefficients.T
    return result.reshape(t.shape+(2,))


def pair_virial_integrals(temperature,*,short_range='constant'):
    t=np.asarray(temperature,float)
    if np.any(~np.isfinite(t)) or np.any(t<=0):
        raise ValueError('Positive finite temperatures required')
    low=fixed_order(t,16,short_range)
    for order in (32,64,128,256,512):
        high=fixed_order(t,order,short_range)
        # Tighter than the reference integrator's 1e-8 relative tolerance.
        if np.all(abs(high-low)<=1e-12+1e-10*abs(high)):
            return high
        low=high
    raise ArithmeticError('Published pair virial quadrature failed order-doubling check')
