"""Opt-in Feautrier transfer on physical column-mass control volumes.

Face optical resistances retain trapezoidal extinction integrals. Cell
emission/absorption use kappa_i * Delta_m_i, not a wavelength-dependent
Delta_tau_i inferred from neighbouring extinctions. Field, flux, material
tangent and local energy therefore discretize the SAME mass conservation
equation. No flux is repaired after the transfer solve.
"""
import numpy as np
from .radiative_transfer import RadiationField,angular_quadrature
from ._stable_feautrier import _DifferenceFactors
from ._compat import trapezoid


def validate_chunk(chunk):
    if isinstance(chunk,(bool,np.bool_)) or not isinstance(chunk,(int,np.integer)) or chunk<1:
        raise ValueError('wavelength_chunk_size must be a positive integer')


def mass_width(mass):
    m=np.asarray(mass,dtype=float)
    if m.ndim!=1 or len(m)<2 or np.any(~np.isfinite(m)) or m[0]<=0 or np.any(np.diff(m)<=0):
        raise ValueError('column mass must be finite, positive and strictly increasing')
    h=np.diff(m,prepend=0.)
    return .5*(h[:-1]+h[1:])


class MassFactors(_DifferenceFactors):
    def __init__(self,tau,fraction,n_angle,mass,extinction):
        self.mu,self.weight=angular_quadrature(n_angle)
        nw,nd=tau.shape;nr=len(self.mu)
        self.h=np.diff(tau,axis=1,prepend=0.)
        if np.any(~np.isfinite(self.h)) or np.any(self.h<=0):
            raise ValueError('optical face resistances must be positive finite')
        volume=extinction[:,:-1]*mass_width(mass)[None,:]
        self.a=np.zeros((nw,nd+1,nr));self.c=np.zeros_like(self.a)
        self.c[:,0]=self.mu/self.h[:,:1]
        self.a[:,1:-1]=self.mu**2/(self.h[:,:-1,None]*volume[:,:,None])
        self.c[:,1:-1]=self.mu**2/(self.h[:,1:,None]*volume[:,:,None])
        eye=np.eye(nr);q=np.broadcast_to(eye,(nw,nd+1,nr,nr)).copy()
        q[:,1:-1]-=fraction[:,:-1,None,None]*self.weight
        self.g=np.empty_like(q);self.p=np.empty_like(q)
        self.g[:,0]=q[:,0];self.p[:,0]=q[:,0]+self.c[:,0,:,None]*eye
        for i in range(1,nd+1):
            self.g[:,i]=q[:,i]+self.a[:,i,:,None]*np.linalg.solve(self.p[:,i-1],self.g[:,i-1])
            self.p[:,i]=self.g[:,i]+self.c[:,i,:,None]*eye


def mass_field(tau,planck,absorption,scattering,*,column_mass,n_angle=4,wavelength_chunk_size=64,
               reconstruct_intensity=False):
    validate_chunk(wavelength_chunk_size)
    b=np.asarray(planck,float);tau=np.broadcast_to(tau,b.shape)
    a=np.broadcast_to(absorption,b.shape);s=np.broadcast_to(scattering,b.shape);ext=a+s
    if b.ndim!=2 or any(np.any(~np.isfinite(x)) for x in (b,a,s,ext)) or np.any(b<0) or np.any(a<0) or np.any(s<0) or np.any(ext<=0):
        raise ValueError('field inputs must be finite nonnegative wavelength/depth arrays, with positive extinction')
    mass_width(column_mass)
    if len(column_mass)!=b.shape[1]:
        raise ValueError('mass grid mismatch')
    return _mass_emission_field(tau,a/ext*b,s/ext,ext,b[:,-1],
        column_mass=column_mass,n_angle=n_angle,wavelength_chunk_size=wavelength_chunk_size,
        reconstruct_intensity=reconstruct_intensity)


class InvalidRadiationFieldError(ValueError):
    """Material coefficients do not admit a finite nonnegative radiation field."""


def mass_emissivity_field(tau,emissivity,absorption,scattering,*,column_mass,
                          bottom_source,n_angle=4,wavelength_chunk_size=64):
    """Transfer with signed net absorption, without dividing by absorption.

    Stimulated emission may make net material absorption negative while total
    extinction stays positive. The same mass-volume equations apply; emission
    enters as eta/extinction and scattering couples through sigma/extinction.
    Reject amplification configurations without a nonnegative finite solution.
    The imposed bottom source must be supplied separately.
    """
    validate_chunk(wavelength_chunk_size)
    eta=np.asarray(emissivity,float);tau=np.broadcast_to(tau,eta.shape)
    a=np.broadcast_to(absorption,eta.shape);s=np.broadcast_to(scattering,eta.shape)
    ext=a+s;bottom=np.asarray(bottom_source,float)
    if (eta.ndim!=2 or bottom.shape!=(eta.shape[0],) or
            any(np.any(~np.isfinite(x)) for x in (eta,a,s,ext,bottom)) or
            np.any(eta<0) or np.any(s<0) or np.any(ext<=0) or np.any(bottom<0)):
        raise ValueError('emissivity transfer requires finite nonnegative emission/scattering/boundary and positive total extinction')
    mass_width(column_mass)
    if len(column_mass)!=eta.shape[1]:raise ValueError('mass grid mismatch')
    try:
        source,field=_mass_emission_field(tau,eta/ext,s/ext,ext,bottom,
            column_mass=column_mass,n_angle=n_angle,wavelength_chunk_size=wavelength_chunk_size,
            require_nonnegative=True,reconstruct_intensity=True)
    except np.linalg.LinAlgError as exc:
        raise InvalidRadiationFieldError('singular radiation field with stimulated emission') from exc
    if (any(np.any(~np.isfinite(x)) for x in
            (source,field.mean_intensity,field.flux,field.interface_flux)) or
            np.any(source<0) or np.any(field.mean_intensity<0)):
        raise InvalidRadiationFieldError('stimulated-emission transfer has no finite nonnegative radiation field')
    return source,field


def _mass_emission_field(tau,emission_source,fraction,ext,bottom_source,*,
                         column_mass,n_angle,wavelength_chunk_size,require_nonnegative=False,
                         reconstruct_intensity=False):
    nw,nd=emission_source.shape
    mean=np.empty_like(emission_source);flux=np.empty_like(mean);interface=np.empty_like(mean)
    for start in range(0,nw,wavelength_chunk_size):
        stop=min(nw,start+wavelength_chunk_size);local=slice(start,stop)
        factor=MassFactors(tau[local],fraction[local],n_angle,column_mass,ext[local])
        rhs=np.zeros((stop-start,nd+1,n_angle))
        rhs[:,1:-1]=emission_source[local,:-1,None]
        rhs[:,-1]=bottom_source[local,None]
        # Deep-Wien sources can be nonzero but subnormal during elimination.
        # Normalize only those very faint rows, leaving ordinary field
        # arithmetic unchanged. Linearity restores the physical units after
        # solving; no negative intensity or emissivity is clipped.
        peak=np.max(abs(rhs),axis=(1,2))
        faint=(peak>0)&(peak<np.sqrt(np.finfo(float).tiny))
        if np.any(faint):rhs[faint]/=peak[faint,None,None]
        u,jump=factor.solve(rhs,reconstruct_intensity=reconstruct_intensity)
        if np.any(faint):
            u[faint]*=peak[faint,None,None]
            jump[faint]*=peak[faint,None,None]
        if require_nonnegative and (np.any(~np.isfinite(u)) or np.any(u<0)):
            index=np.unravel_index(np.argmin(u),u.shape)
            error=InvalidRadiationFieldError(
                'stimulated-emission transfer has nonphysical angular intensities '
                f'(wavelength index {start+index[0]}, depth {index[1]}, '
                f'minimum {u[index]:.6g}, row maximum {np.max(abs(u[index[0]])):.6g}, '
                f'maximum scattering fraction {np.max(fraction[start+index[0]]):.6g})')
            # Small reproducer for numerical diagnostics; no full-grid copy.
            row=start+index[0]
            error.transfer_row=dict(tau=tau[row].copy(),emission_source=emission_source[row].copy(),
                fraction=fraction[row].copy(),extinction=ext[row].copy(),bottom_source=bottom_source[row],
                column_mass=np.asarray(column_mass).copy(),n_angle=n_angle)
            raise error
        mean[local]=u[:,1:]@factor.weight
        derivative=jump/factor.h[:,:,None];fw=4*np.pi*factor.weight*factor.mu**2
        interface[local]=derivative@fw
        nodal=derivative.copy()
        nodal[:,:-1]=(factor.h[:,1:,None]*derivative[:,:-1]+factor.h[:,:-1,None]*derivative[:,1:])/(factor.h[:,1:]+factor.h[:,:-1])[:,:,None]
        flux[local]=nodal@fw
    return emission_source+fraction*mean,RadiationField(mean_intensity=mean,flux=flux,interface_flux=interface)


class MassResponseOperator:
    """Reuse one frozen transfer operator across material derivative columns.

    The private copies prevent subsequent trial mutations from changing the
    anchor of cached factors. Only source/opacity derivatives vary in apply.
    """
    def __init__(self,tau,wave,source,fraction,mass,*,extinction,n_angle=4,
                 wavelength_chunk_size=16,allow_stimulated_gain=False,reconstruct_intensity=False):
        for name,value in (('tau',tau),('wave',wave),('source',source),
                           ('fraction',fraction),('mass',mass),('extinction',extinction)):
            array=np.array(value,dtype=float,copy=True)
            array.flags.writeable=False
            setattr(self,name,array)
        self.n_angle=n_angle
        self.chunk_size=wavelength_chunk_size
        self.allow_stimulated_gain=allow_stimulated_gain
        self.reconstruct_intensity=reconstruct_intensity
        self.factors={}

    def apply(self,direct,db,dk,*,return_auxiliary_response=True,mean_response_consumer=None,nodal_flux_response_consumer=None):
        return mass_response(self.tau,self.wave,self.source,direct,db,self.fraction,
            self.mass,dk,extinction=self.extinction,n_angle=self.n_angle,
            wavelength_chunk_size=self.chunk_size,allow_stimulated_gain=self.allow_stimulated_gain,
            reconstruct_intensity=self.reconstruct_intensity,
            return_auxiliary_response=return_auxiliary_response,
            mean_response_consumer=mean_response_consumer,nodal_flux_response_consumer=nodal_flux_response_consumer,_operator=self)


def mass_response(tau,wave,source,direct,db,fraction,mass,dk,*,extinction,n_angle=4,
                  wavelength_chunk_size=16,return_auxiliary_response=True,mean_response_consumer=None,
                  nodal_flux_response_consumer=None,
                  allow_stimulated_gain=False,reconstruct_intensity=False,_operator=None):
    """Exact linear response of the mass-volume field at fixed mass nodes."""
    validate_chunk(wavelength_chunk_size)
    wave=np.asarray(wave);source=np.asarray(source);mass=np.asarray(mass)
    if source.ndim!=2 or np.any(~np.isfinite(source)):raise ValueError('invalid source')
    nw,nd=source.shape
    if wave.shape!=(nw,) or nw<2 or np.any(~np.isfinite(wave)) or np.any(wave<=0) or np.any(np.diff(wave)<=0):raise ValueError('invalid wavelengths')
    mass_width(mass)
    if mass.shape!=(nd,):raise ValueError('mass grid mismatch')
    if any(np.shape(x)!=source.shape or np.any(~np.isfinite(x)) for x in (tau,direct,db,fraction,dk,extinction)):
        raise ValueError('response fields must be finite and have identical shapes')
    if (np.any(extinction<=0) or np.any(fraction<0) or
            (not allow_stimulated_gain and np.any(fraction>1))):raise ValueError('invalid material coefficients')
    if _operator is not None and (any(left is not right for left,right in (
            (tau,_operator.tau),(wave,_operator.wave),(source,_operator.source),
            (fraction,_operator.fraction),(mass,_operator.mass),(extinction,_operator.extinction)))
            or n_angle!=_operator.n_angle or wavelength_chunk_size!=_operator.chunk_size
            or reconstruct_intensity!=_operator.reconstruct_intensity):
        raise ValueError('response operator belongs to a different anchor')
    weights=np.diff(wave,prepend=wave[0],append=wave[-1]);weights=.5*(weights[:-1]+weights[1:])
    integrated=np.zeros((nd,nd))
    mean_response=np.empty((nw,nd,nd)) if return_auxiliary_response else None
    source_response=np.empty_like(mean_response) if return_auxiliary_response else None
    for start in range(0,nw,wavelength_chunk_size):
        stop=min(nw,start+wavelength_chunk_size);local=slice(start,stop);nc=stop-start
        cached=None if _operator is None else _operator.factors.get(start)
        if cached is None:
            scalar=MassFactors(tau[local],np.zeros((nc,nd)),n_angle,mass,extinction[local])
            rhs=np.zeros((nc,nd+1,n_angle));rhs[:,1:]=source[local,:,None]
            _,jumps=scalar.solve(rhs,reconstruct_intensity=reconstruct_intensity)
            coupled=MassFactors(tau[local],fraction[local],n_angle,mass,extinction[local])
            if _operator is not None:_operator.factors[start]=(coupled,jumps)
        else:
            coupled,jumps=cached
        h=coupled.h
        dh=np.zeros((nc,nd,nd));dh[:,0,0]=mass[0]*dk[local,0]
        for i in range(1,nd):
            dh[:,i,i-1]=.5*(mass[i]-mass[i-1])*dk[local,i-1]
            dh[:,i,i]=.5*(mass[i]-mass[i-1])*dk[local,i]
        tangent_rhs=np.zeros((nc,nd+1,n_angle,nd))
        for i in range(nd-1):tangent_rhs[:,i+1,:,i]=direct[local,i,None]
        tangent_rhs[:,-1,:,-1]=db[local,-1,None]
        tangent_rhs[:,0]=-coupled.mu[None,:,None]*dh[:,0,None,:]/h[:,0,None,None]**2*jumps[:,0,:,None]
        dv=np.zeros((nc,nd-1,nd))
        cells=np.arange(nd-1);dv[:,cells,cells]=dk[local,:-1]/extinction[local,:-1]
        da=-coupled.a[:,1:-1,:,None]*(dh[:,:-1,None,:]/h[:,:-1,None,None]+dv[:,:,None,:])
        dc=-coupled.c[:,1:-1,:,None]*(dh[:,1:,None,:]/h[:,1:,None,None]+dv[:,:,None,:])
        tangent_rhs[:,1:-1]+=-da*jumps[:,:-1,:,None]+dc*jumps[:,1:,:,None]
        response,jump_response=coupled.solve(tangent_rhs,reconstruct_intensity=reconstruct_intensity)
        mean=np.einsum('wdrk,r->wdk',response[:,1:],coupled.weight)
        if return_auxiliary_response:
            mean_response[local]=mean
            sr=fraction[local,:,None]*mean
            sr[:,np.arange(nd),np.arange(nd)]+=direct[local]
            source_response[local]=sr
        if mean_response_consumer is not None:
            mean.flags.writeable=False;mean_response_consumer(start,stop,mean)
        local_flux=np.einsum('wdrk,r->wdk',jump_response/h[:,:,None,None]
            -jumps[:,:,:,None]*dh[:,:,None,:]/h[:,:,None,None]**2,4*np.pi*coupled.weight*coupled.mu**2)
        if nodal_flux_response_consumer is not None:
            # Differentiate the same optical-resistance interpolation used
            # for the nodal flux in _mass_emission_field.
            face_flux=np.einsum('wdr,r->wd',jumps/h[:,:,None],4*np.pi*coupled.weight*coupled.mu**2)
            nodal=local_flux.copy()
            denom=h[:,:-1]+h[:,1:]
            value=(h[:,1:]*face_flux[:,:-1]+h[:,:-1]*face_flux[:,1:])/denom
            nodal[:,:-1]=(h[:,1:,None]*local_flux[:,:-1]+h[:,:-1,None]*local_flux[:,1:]
                +dh[:,1:]*(face_flux[:,:-1]-value)[:,:,None]
                +dh[:,:-1]*(face_flux[:,1:]-value)[:,:,None])/denom[:,:,None]
            nodal.flags.writeable=False
            nodal_flux_response_consumer(start,stop,nodal)
        integrated+=np.einsum('wdk,w->dk',local_flux,weights[local])
    return integrated,mean_response,source_response


def mass_energy(wave,mass,planck,mean,absorption):
    weight=4*np.pi*mass_width(mass)[None,:]*absorption[:,:-1]
    return (trapezoid(weight*(mean[:,:-1]-planck[:,:-1]),wave,axis=0),
        trapezoid(weight*planck[:,:-1],wave,axis=0))


def mass_emissivity_energy(wave,mass,emissivity,mean,absorption):
    """Physical cell heating and emission, valid across zero net absorption."""
    width=4*np.pi*mass_width(mass)
    return (width*trapezoid(absorption[:,:-1]*mean[:,:-1]-emissivity[:,:-1],wave,axis=0),
            width*trapezoid(emissivity[:,:-1],wave,axis=0))


def mass_energy_response(wave,tau,mass,planck,mean,source,absorption,scattering,db,da,ds,*,
                         n_angle=4,wavelength_chunk_size=16,response_solver=None):
    """Energy and emission responses use the same physical control volumes."""
    ext=absorption+scattering;dk=da+ds
    direct=(da*planck+absorption*db+ds*mean-dk*source)/ext
    nw,nd=planck.shape;energy_j=np.zeros((nd-1,nd));emission_j=np.zeros_like(energy_j)
    weights=np.diff(wave,prepend=wave[0],append=wave[-1]);weights=.5*(weights[:-1]+weights[1:])
    widths=4*np.pi*mass_width(mass);cells=np.arange(nd-1)
    def consume(start,stop,dj):
        local=slice(start,stop)
        response=absorption[local,:-1,None]*dj[:,:-1,:]
        response[:,cells,cells]+=da[local,:-1]*(mean[local,:-1]-planck[local,:-1])-absorption[local,:-1]*db[local,:-1]
        energy_j[:]+=widths[:,None]*np.einsum('wdk,w->dk',response,weights[local])
        emission_j[cells,cells]+=widths*np.einsum('wd,w->d',
            da[local,:-1]*planck[local,:-1]+absorption[local,:-1]*db[local,:-1],weights[local])
    flux_j,_,_=mass_response(tau,wave,source,direct,db,scattering/ext,mass,dk,extinction=ext,
        n_angle=n_angle,wavelength_chunk_size=wavelength_chunk_size,return_auxiliary_response=False,
        mean_response_consumer=consume)
    return flux_j,energy_j,emission_j
