"""Exact-material ML2 coordinates for an explicitly selected research solve.

The coordinates are surface ln(T) and signed element velocities. Temperature
is reconstructed by solving gradient/material compatibility. Positive y has
Fconv/Fstar=y**3; negative y parameterizes stable, zero-convection cells.
No radiative or convective output flux is supplied by this coordinate map.
"""
import numpy as np
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient,ml2_auxiliary_compatibility
from wd_spectra.nonlinear import RecoverableEvaluationError


class CompatibleTemperatures:
    def __init__(self,log_temperature,pressure,materials,target):
        self.reference=np.asarray(log_temperature).copy()
        pressure=np.asarray(pressure)
        if (self.reference.ndim!=1 or self.reference.size<2 or pressure.shape!=self.reference.shape
                or np.any(~np.isfinite(self.reference)) or np.any(~np.isfinite(pressure))
                or np.any(pressure<=0) or np.any(np.diff(pressure)<=0)):
            raise ValueError('compatible coordinates require finite ordered physical profiles')
        self.materials=materials
        self.target=target
        self.n=len(self.reference)
        dp=np.diff(np.log(pressure))
        self.operator=np.zeros((self.n-1,self.n))
        indices=np.arange(self.n-1)
        self.operator[indices,indices]=-1/dp
        self.operator[indices,indices+1]=1/dp
        self.dp=dp
        coefficients=tuple(v[1:] for v in materials(self.reference))
        y=ml2_auxiliary_from_gradient(np.diff(self.reference)/dp,*coefficients,target)
        self.scale=np.maximum(1.,abs(y))
        self.initial=np.r_[self.reference[0],y/self.scale]
        self.cache={}

    def decode(self,q):
        q=np.asarray(q)
        if q.shape!=(self.n,) or np.any(~np.isfinite(q)):
            raise RecoverableEvaluationError('invalid compatible temperature coordinate')
        key=q.tobytes()
        if key in self.cache:return self.cache[key]
        logt=self.reference+(q[0]-self.reference[0])
        y=q[1:]*self.scale
        previously_small=False
        for iteration in range(24):
            full,derivatives=self.materials(logt,True)
            coefficients=tuple(v[1:] for v in full)
            responses=tuple(v[1:] for v in derivatives)
            gradient=np.diff(logt)/self.dp
            defect,tangent,dy=ml2_auxiliary_compatibility(
                gradient,y,coefficients,responses,self.operator,self.target)
            matrix=np.vstack((np.eye(self.n)[0],tangent))
            r=np.r_[logt[0]-q[0],defect]
            correction=np.linalg.solve(matrix,-r)
            actual_y=ml2_auxiliary_from_gradient(gradient,*coefficients,self.target)
            flux_defect=float(np.max(abs(np.maximum(actual_y,0.)**3-np.maximum(y,0.)**3)))
            if iteration and iteration%4==0:
                print(f'COMPATIBLE MATERIAL ITERATION {iteration}: '
                    f'flux defect {flux_defect:.6g}, max dlnT correction '
                    f'{np.max(abs(correction)):.6g} at node {np.argmax(abs(correction))}',flush=True)
            # Match the established finite-material repair's log-T accuracy;
            # nested thermodynamic differences have a noise floor well above
            # log-T roundoff. Actual flux compatibility is checked separately,
            # and the atmosphere's physical convergence gates are unchanged.
            small=np.max(abs(correction))<1e-9 and flux_defect<1e-7
            # Polish once after reaching the material-noise tolerance. This
            # preserves derivative accuracy for smooth materials rather than
            # returning a deliberately un-applied small Newton correction.
            if small and previously_small:
                rhs=np.zeros((self.n,self.n));rhs[0,0]=1.
                rhs[1:,1:]=np.diag(-dy*self.scale)
                mapping=np.linalg.solve(matrix,rhs)
                result=(logt,mapping,flux_defect)
                # Bounded cache of exact arguments; the mathematical mapping
                # always uses its fixed initial seed, not an order-dependent one.
                if len(self.cache)>=16:self.cache.pop(next(iter(self.cache)))
                self.cache[key]=result
                return result
            previously_small=small
            if np.any(~np.isfinite(correction)) or np.max(abs(correction))>1.:
                raise RecoverableEvaluationError('compatible temperature reconstruction left its local branch')
            logt=logt+correction
        raise RecoverableEvaluationError(f'exact-material temperature compatibility did not solve: '
            f'flux defect {flux_defect:.6g}; max dlnT correction {np.max(abs(correction)):.6g}; '
            f'node {np.argmax(abs(correction))}, T={np.exp(logt[np.argmax(abs(correction))]):.6g}')

    def convective_tangent(self,q):
        y=np.asarray(q)[1:]*self.scale
        result=np.zeros((self.n,self.n))
        result[1:,1:]=np.diag(3*np.maximum(y,0.)**2*self.scale*self.target)
        return result
