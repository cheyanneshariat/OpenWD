"""Monotone cubic formal integral with consistent scattering.

PCHIP source reconstruction on the existing optical-depth nodes. The opacity
integral, top constant-source cell and thermal bottom boundary are unchanged.
This does not change the atmosphere solver or enforce a bolometric scaling.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import gammainc
from .radiative_transfer import angular_quadrature, _validate_inputs, RadiationField
from ._linear_scattering import linear_lambda_operator


def slopes(x, y, *, jacobian=False):
    """Fritsch-Butland harmonic slopes with standard one-sided PCHIP edges."""
    h=np.diff(x,axis=1)
    delta=np.diff(y,axis=1)/h
    out=np.zeros_like(y)
    tangent=np.zeros((*y.shape,y.shape[1])) if jacobian else None
    if y.shape[1]==2:
        out[:]=delta
        if jacobian:
            tangent[:,:,0]=-1/h
            tangent[:,:,1]=1/h
        return (out,tangent) if jacobian else out
    same=(np.sign(delta[:,:-1])==np.sign(delta[:,1:])) & (delta[:,:-1]!=0) & (delta[:,1:]!=0)
    w1=2*h[:,1:]+h[:,:-1]
    w2=h[:,1:]+2*h[:,:-1]
    # Reciprocal scaling avoids overflow when an underflowing Wien tail has
    # vanishing slopes next to finite values.
    d0=np.where(same,delta[:,:-1],1.)
    d1=np.where(same,delta[:,1:],1.)
    smaller=np.minimum(abs(d0),abs(d1))
    ratio=(w1+w2)/(w1*smaller/d0+w2*smaller/d1)
    out[:,1:-1]=np.where(same,smaller*ratio,0.)
    if jacobian:
        p=np.where(same,w1/(w1+w2)*(out[:,1:-1]/d0)**2,0.)
        q=np.where(same,w2/(w1+w2)*(out[:,1:-1]/d1)**2,0.)
        j=np.arange(1,y.shape[1]-1)
        tangent[:,j,j-1]=-p/h[:,:-1]
        tangent[:,j,j]=p/h[:,:-1]-q/h[:,1:]
        tangent[:,j,j+1]=q/h[:,1:]
    for index,h0,h1,m0,m1 in ((0,h[:,0],h[:,1],delta[:,0],delta[:,1]),
                              (-1,h[:,-1],h[:,-2],delta[:,-1],delta[:,-2])):
        edge=((2*h0+h1)*m0-h0*m1)/(h0+h1)
        zero=np.sign(edge)!=np.sign(m0)
        edge=np.where(zero,0.,edge)
        limited=(np.sign(m0)!=np.sign(m1)) & (abs(edge)>3*abs(m0))
        out[:,index]=np.where(limited,3*m0,edge)
        if jacobian:
            p=np.where(zero,0.,np.where(limited,3.,(2*h0+h1)/(h0+h1)))
            q=np.where(zero|limited,0.,-h0/(h0+h1))
            if index==0:
                tangent[:,0,0]=-p/h0
                tangent[:,0,1]=p/h0-q/h1
                tangent[:,0,2]=q/h1
            else:
                tangent[:,-1,-1]=p/h0
                tangent[:,-1,-2]=-p/h0+q/h1
                tangent[:,-1,-3]=-q/h1
    return (out,tangent) if jacobian else out


class CubicFormal:
    def __init__(self,tau,n_angle=3):
        self.tau=np.asarray(tau,dtype=float)
        self.h=np.diff(self.tau,axis=1)
        self.mu,self.weights=angular_quadrature(n_angle)
        self.coefficients=[]
        for mu in self.mu:
            delta=self.h/mu
            moments=[-np.expm1(-delta)]
            # Integral of u^k exp(-delta*u) delta du, u in [0,1].
            # gammainc avoids subtracting nearly equal terms at small delta.
            for k in range(1,4):
                small=delta<1e-3
                value=np.zeros_like(delta)
                for j in range(7):
                    value[small]+=(-1.)**j*delta[small]**(j+1)/(math.factorial(j)*(k+j+1))
                value[~small]=math.factorial(k)*gammainc(k+1,delta[~small])*delta[~small]**(-k)
                moments.append(value)
            self.coefficients.append((np.exp(-delta),moments,
                np.exp(-self.tau[:,0]/mu),-np.expm1(-self.tau[:,0]/mu)))

    def field(self,source):
        source=np.asarray(source,dtype=float)
        derivative=slopes(self.tau,source)
        return self.field_from_slopes(source,derivative)

    def field_from_slopes(self,source,derivative):
        """Linear sweep also used for the exact piecewise PCHIP tangent."""
        h=self.h if source.ndim==2 else self.h[...,None]
        change=np.diff(source,axis=1)
        left=h*derivative[:,:-1]
        right=h*derivative[:,1:]
        # Cubic coefficients viewed inward from each cell's upper end.
        upper=(source[:,:-1],left,3*change-2*left-right,-2*change+left+right)
        # Same interpolant viewed outward from the cell's lower end.
        lower=(source[:,1:],-right,-3*change+2*right+left,2*change-right-left)
        mean=np.zeros_like(source)
        flux=np.zeros_like(source)
        surface=np.zeros((len(source),)+source.shape[2:])
        for mu,weight,(attenuation,moments,surface_attenuation,surface_emission) in zip(self.mu,self.weights,self.coefficients):
            if source.ndim==3:
                attenuation=attenuation[...,None]
                moments=[m[...,None] for m in moments]
                surface_attenuation=surface_attenuation[...,None]
                surface_emission=surface_emission[...,None]
            up_emission=sum(c*m for c,m in zip(upper,moments))
            down_emission=sum(c*m for c,m in zip(lower,moments))
            inward=np.empty_like(source);outward=np.empty_like(source)
            inward[:,0]=source[:,0]*surface_emission
            outward[:,-1]=source[:,-1]
            for j in range(source.shape[1]-1):
                inward[:,j+1]=inward[:,j]*attenuation[:,j]+down_emission[:,j]
            for j in range(source.shape[1]-2,-1,-1):
                outward[:,j]=outward[:,j+1]*attenuation[:,j]+up_emission[:,j]
            mean+=.5*weight*(inward+outward)
            flux+=2*np.pi*weight*mu*(outward-inward)
            surface+=2*np.pi*weight*mu*(outward[:,0]*surface_attenuation+source[:,0]*surface_emission)
        return RadiationField(mean_intensity=mean,flux=flux),surface


def _solve_cubic_chunk(tau,planck,absorption,scattering,*,wavelength,n_angle=3,max_iterations=32):
    tau,b=_validate_inputs(tau,planck)
    tau=np.broadcast_to(tau,b.shape)
    a,s=[np.broadcast_to(np.asarray(v,dtype=float),b.shape) for v in (absorption,scattering)]
    total=a+s
    if np.any(~np.isfinite(total)) or np.any(a<0) or np.any(s<0) or np.any(total<=0):
        raise ValueError('invalid opacities')
    epsilon=a/total;fraction=s/total
    formal=CubicFormal(tau,n_angle)
    source=b.copy()
    matrices=[]
    for begin in range(0,len(tau),64):
        stop=min(begin+64,len(tau))
        lam=linear_lambda_operator(tau[begin:stop],n_angle)
        matrix=np.eye(b.shape[1])[None,:,:]-fraction[begin:stop,:,None]*lam
        matrices.append((begin,stop,matrix))
        source[begin:stop]=np.linalg.solve(matrix,(epsilon[begin:stop]*b[begin:stop])[...,None])[...,0]
    weight=np.asarray(wavelength)[:,None]
    history=[]
    for iteration in range(max_iterations+1):
        field,flux=formal.field(source)
        defect=epsilon*b+fraction*field.mean_intensity-source
        scale=max(float(np.max(weight*abs(source))),np.finfo(float).tiny)
        error=float(np.max(weight*abs(defect))/scale)
        history.append(error)
        if error<1e-11:
            # Re-evaluate separately rather than reuse a solved source RHS.
            independent,_=CubicFormal(tau,n_angle).field(source)
            error=float(np.max(weight*abs(source-epsilon*b-fraction*independent.mean_intensity))/scale)
            if error>1e-10:
                raise RuntimeError('cubic source independent closure failed')
            return source,field,flux,dict(iterations=iteration,source_converged=True,
                independent_radiation_scaled_source_error=error,history=history)
        if iteration==max_iterations:
            raise RuntimeError(f'cubic source did not converge: {history}')
        correction=np.empty_like(source)
        for begin,stop,matrix in matrices:
            if iteration>=4:
                # Weak-scattering rows close in a few cheap defect corrections.
                # Slow corrections require the actual cubic-source Jacobian,
                # not a larger budget of linear-operator iterations.
                _,d=slopes(tau[begin:stop],source[begin:stop],jacobian=True)
                basis=np.broadcast_to(np.eye(b.shape[1]),d.shape)
                lam=CubicFormal(tau[begin:stop],n_angle).field_from_slopes(basis,d)[0].mean_intensity
                matrix=np.eye(b.shape[1])[None,:,:]-fraction[begin:stop,:,None]*lam
            correction[begin:stop]=np.linalg.solve(matrix,defect[begin:stop,...,None])[...,0]
        alpha=1.
        while alpha>=2**-12:
            trial=source+alpha*correction
            if np.all(trial>=0) and np.all(np.isfinite(trial)):
                trial_field,_=formal.field(trial)
                trial_error=float(np.max(weight*abs(epsilon*b+fraction*trial_field.mean_intensity-trial))/scale)
                if trial_error<error:
                    source=trial
                    break
            alpha*=.5
        else:
            raise RuntimeError(f'cubic source line search failed: {history}')


def solve_cubic_source(tau, planck, absorption, scattering, *, wavelength,
                       n_angle=4, max_iterations=32, wavelength_chunk_size=64):
    """Solve the nonlinear monotone-source equation with bounded storage.

    No bolometric target enters this calculation. The returned atmosphere is
    not changed, and source-closure failure raises rather than selecting a
    different transfer method.
    """
    tau, b = _validate_inputs(tau, planck)
    tau = np.broadcast_to(tau, b.shape)
    if np.any(tau[:, 0] < 0):
        raise ValueError("optical depth must be nonnegative")
    wave = np.asarray(wavelength, dtype=float)
    if wave.shape != (len(b),) or np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wavelength must be finite, positive and match the source")
    for name, value, minimum in (("wavelength_chunk_size", wavelength_chunk_size, 1),
                                 ("max_iterations", max_iterations, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    a, s = [np.broadcast_to(np.asarray(v, dtype=float), b.shape)
            for v in (absorption, scattering)]
    source = np.empty_like(b)
    mean = np.empty_like(b)
    flux = np.empty_like(b)
    surface = np.empty(len(b))
    maximum_iterations = 0
    maximum_error = 0.
    for begin in range(0, len(b), wavelength_chunk_size):
        stop = min(begin + wavelength_chunk_size, len(b))
        src, field, emergent, record = _solve_cubic_chunk(
            tau[begin:stop], b[begin:stop], a[begin:stop], s[begin:stop],
            wavelength=wave[begin:stop], n_angle=n_angle, max_iterations=max_iterations)
        source[begin:stop] = src
        mean[begin:stop] = field.mean_intensity
        flux[begin:stop] = field.flux
        surface[begin:stop] = emergent
        maximum_iterations = max(maximum_iterations, record["iterations"])
        maximum_error = max(maximum_error, record["independent_radiation_scaled_source_error"])
    return source, RadiationField(mean_intensity=mean, flux=flux), surface, {
        "iterations": maximum_iterations, "source_converged": True,
        "independent_radiation_scaled_source_error": maximum_error,
        "wavelength_chunk_size": wavelength_chunk_size,
    }
