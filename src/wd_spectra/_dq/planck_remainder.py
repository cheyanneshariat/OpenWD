"""Nonlinear Planck response of the frozen, coupled refractive transfer.

Only a proposal model. Physical accepted transfer is never frozen. Integrate
the Planck response into Chebyshev coefficients, avoiding a wavelength x
depth x depth cache. Subtract its value and tangent at the anchor so all
opacity, composition and refractivity derivatives remain exactly those of
the full physical model. No new closure, flux normalization or line scale.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from numpy.polynomial.chebyshev import chebder
from wd_spectra.spectrum import planck_lambda_angstrom
from .dq_refractive_finite_volume import validate, operators


def columns(coefficients, coordinate):
    """Clenshaw evaluation, last axis is independent source temperature."""
    b1=np.zeros_like(coefficients[0]);b2=b1.copy()
    for c in coefficients[:0:-1]:
        b0=c+2*coordinate*b1-b2;b2,b1=b1,b0
    return coefficients[0]+coordinate*b1-b2


class PlanckRemainder:
    def __init__(self, mass, wave, a, s, b, index, temperature, weights, *,
                 n_angle=4, span=.12, degree=24):
        mass,(a,s,b,index),angles,angular_weights=validate(mass,a,s,b,index,n_angle)
        self.anchor=np.log(temperature);self.span=span
        nodes=np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
        transform=2/(degree+1)*np.cos(np.arange(degree+1)[:,None]*np.arccos(nodes))
        transform[0]*=.5
        size=len(mass);cells=size-1
        self.flux=np.zeros((degree+1,size,size))
        self.heat=np.zeros((degree+1,cells,size))
        self.emission=np.zeros((degree+1,cells))
        sampled_temperature=np.exp(self.anchor[None,:]+span*nodes[:,None])
        for iw,w in enumerate(wave):
            k=a[iw]+s[iw];eps=a[iw,:-1]/k[:-1]
            lam,fl,volume=operators(mass,k,index[iw],angles,angular_weights)
            matrix=np.eye(cells)-lam[:,:-1]*(1-eps)[None,:]
            mean=np.linalg.solve(matrix,np.column_stack([lam[:,:-1]*eps[None,:],lam[:,-1]]))
            source=np.zeros((size,size));source[:-1]=(1-eps)[:,None]*mean
            source[np.arange(cells),np.arange(cells)]+=eps;source[-1,-1]=1.
            flux=fl@source
            heat=mean.copy();heat[np.arange(cells),np.arange(cells)]-=1.
            thermal=4*np.pi*eps*volume
            heat*=thermal[:,None]
            bb=planck_lambda_angstrom(float(w),sampled_temperature)
            cb=transform@bb
            self.flux+=weights[iw]*flux[None,:,:]*cb[:,None,:]
            self.heat+=weights[iw]*heat[None,:,:]*cb[:,None,:]
            self.emission+=weights[iw]*thermal[None,:]*cb[:,:-1]
        self.coefficients=(self.flux,self.heat,self.emission)
        self.derivatives=tuple(chebder(c,axis=0)/span for c in self.coefficients)
        zeros=np.zeros(size)
        self.base=tuple(columns(c,zeros if c.ndim==3 else zeros[:-1]) for c in self.coefficients)
        self.slope=tuple(columns(c,zeros if c.ndim==3 else zeros[:-1]) for c in self.derivatives)

    def __call__(self, log_temperature):
        dx=np.asarray(log_temperature)-self.anchor
        if np.max(abs(dx))>self.span*(1+1e-12):
            raise ValueError('Planck proposal outside its interpolation box')
        values=[];derivatives=[]
        for coeff,dcoeff,base,slope in zip(self.coefficients,self.derivatives,self.base,self.slope):
            x=dx if coeff.ndim==3 else dx[:-1]
            rem=columns(coeff,x/self.span)-base-slope*x
            derivative=columns(dcoeff,x/self.span)-slope
            if coeff.ndim==3:rem=rem.sum(axis=1)
            else:derivative=np.column_stack([np.diag(derivative),np.zeros(len(derivative))])
            values.append(rem);derivatives.append(derivative)
        return tuple(values),tuple(derivatives)


@contextmanager
def nonlinear_planck_trials():
    from . import nonlinear_material_model as nm
    from .dq_refractive_material import RefractiveBackend
    original_init=RefractiveBackend.__init__;original_model=nm.local_model
    def initialize(backend,material,wave):
        original_init(backend,material,wave)
        material.planck_trial_backend=backend
    def local_model(system,state,ev):
        owner=system.eos_trial_owner
        if owner is not None:
            # Cache by the actual anchor evaluation, not rounded temperatures.
            cached=getattr(system,'planck_trial_cache',None)
            if cached is None or cached[0] is not ev:
                backend=owner.planck_trial_backend
                temperature=ev.payload['atmosphere'].temperature
                matches=[c for c in backend.contexts.values() if np.array_equal(c['state'].temperature,temperature)]
                if not matches:raise RuntimeError('Missing physical Planck anchor context')
                ctx=matches[-1];wave=backend.wave
                d=np.diff(wave,prepend=wave[0],append=wave[-1]);weights=.5*(d[:-1]+d[1:])
                path=owner.experiment_metadata.get('fixed_wavelength_grid')
                if path is not None:
                    with np.load(path) as z:
                        if 'weights' in z.files:
                            np.testing.assert_array_equal(z['wavelength'],wave);weights=z['weights'].copy()
                print('Building nonlinear Planck trial response',flush=True)
                model=PlanckRemainder(ctx['mass'],wave,ctx['a'],ctx['s'],ctx['b'],ctx['n'],temperature,
                    weights,n_angle=owner.ray_angles)
                system.planck_trial_cache=(ev,model)
                print('Nonlinear Planck trial response ready',flush=True)
            system.radiation_trial_remainder=system.planck_trial_cache[1]
        return original_model(system,state,ev)
    with patch.object(RefractiveBackend,'__init__',initialize),patch.object(nm,'local_model',local_model):yield
