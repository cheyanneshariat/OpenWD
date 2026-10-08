"""Coupled temperature/element-velocity proposal; no eliminated trial bounds.

The square augmented system contains energy equations AND exact ML2
temperature-gradient compatibility. Auxiliary flux is only a proposal;
the outer solver evaluates flux from the actual temperature gradient.
"""
import logging
import numpy as np
from scipy.optimize import least_squares
from .radiative_rate_proposal import radiative_rates
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._ml2_auxiliary import (ml2_auxiliary_from_gradient,
    ml2_auxiliary_compatibility,ml2_scaled_coefficients)


def augmented_model(evaluation,*,cell_only=False,velocity_scaled_compatibility=False,
                    thermal_scaled_compatibility=False,pseudo_time_step=None,pseudo_relax_bottom=False):
    if pseudo_time_step is not None and (not np.isfinite(pseudo_time_step) or pseudo_time_step<=0):
        raise ValueError('pseudo time step must be positive finite')
    if pseudo_relax_bottom and pseudo_time_step is None:
        raise ValueError('a thermal boundary reservoir requires pseudo time')
    p=evaluation.payload
    target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
    n=p['atmosphere'].n_depth
    tr=p['convection_transport']
    a,b,c=(tr[k][1:] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
    y0=ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],a,b,c,target)
    ys=np.maximum(1.,abs(y0))
    ca,cb=ml2_scaled_coefficients(b,c,target)
    compatibility_scale=ca*ys+cb*ys**2
    r=p['radiative_flux_interface']/target
    rj=p['radiative_flux_log_temperature_jacobian']/target
    e=p['radiative_cell_energy_defect']/target
    ej=p['radiative_cell_energy_log_temperature_jacobian']/target
    emission=p['thermal_cell_emission']/target
    emission_j=p['thermal_cell_emission_log_temperature_jacobian']/p['thermal_cell_emission'][:,None]
    gradient=p['temperature_gradient'][1:]
    operator=p['interface_gradient_operator'][1:]
    material=p['diagnostic_material_model']
    def model(z):
        dt,y=z[:n],y0+ys*z[n:]
        values,responses=material(dt)
        defect,gj,yj=ml2_auxiliary_compatibility(gradient+operator@dt,y,
            tuple(v[1:] for v in values),tuple(v[1:] for v in responses),operator,target)
        flux=np.r_[0.,np.maximum(y,0)**3]
        fj=np.zeros((n,n-1));fj[1:]=np.diag(3*np.maximum(y,0)**2*ys)
        exchange,exchange_j,thermal,thermal_j=radiative_rates(p,dt)
        cell=exchange+np.diff(flux)
        scale=thermal+flux[:-1]+flux[1:]
        q=cell/scale
        q_t=exchange_j/scale[:,None]-q[:,None]*thermal_j/scale[:,None]
        q_y=np.diff(fj,axis=0)/scale[:,None]-q[:,None]*(fj[:-1]+fj[1:])/scale[:,None]
        if pseudo_time_step is not None:
            # Frozen positive rates are a local time preconditioner. The
            # bottom stellar-flux boundary is algebraic; all other cells
            # evolve in the sign of their actual net heating.
            frozen=p['cell_energy_scale']/target
            time_j=np.zeros((n-1,n));time_j[np.arange(n-1),np.arange(n-1)]=1/pseudo_time_step
            energy=np.r_[cell/frozen-dt[:-1]/pseudo_time_step,r[-1]+rj[-1]@dt+flux[-1]-1]
            energy_j=np.vstack((np.hstack((exchange_j/frozen[:,None]-time_j,
                np.diff(fj,axis=0)/frozen[:,None])),np.r_[rj[-1],fj[-1]]))
            if pseudo_relax_bottom:
                energy[-1]+=dt[-1]/pseudo_time_step
                energy_j[-1,n-1]+=1/pseudo_time_step
        elif cell_only:
            energy=np.r_[r[0]+rj[0]@dt-1,q]
            energy_j=np.vstack((np.r_[rj[0],np.zeros(n-1)],np.hstack((q_t,q_y))))
        else:
            energy=r+rj@dt+flux-1
            energy[:-1]-=q
            energy_j=np.hstack((rj,fj));energy_j[:-1]-=np.hstack((q_t,q_y))
        if velocity_scaled_compatibility:
            from .scaled_ml2_compatibility import scaled_compatibility
            scale_options={}
            if thermal_scaled_compatibility:
                # Adjacent-cell thermal emission gives the local energy
                # resolution required of this interface's physical ML2 flux.
                face_thermal=np.r_[thermal[0],.5*(thermal[:-1]+thermal[1:]),thermal[-1]][1:]
                face_thermal_j=np.vstack((thermal_j[0],.5*(thermal_j[:-1]+thermal_j[1:]),thermal_j[-1]))[1:]
                scale_options=dict(thermal_scale=face_thermal,thermal_scale_jacobian=face_thermal_j)
            compatibility,gj,yj=scaled_compatibility(gradient+operator@dt,y,
                tuple(v[1:] for v in values),tuple(v[1:] for v in responses),operator,target,**scale_options)
            compatibility_j=np.hstack((gj,np.diag(yj*ys)))
        else:
            compatibility=defect/compatibility_scale
            compatibility_j=np.hstack((gj,np.diag(yj*ys)))/compatibility_scale[:,None]
        return np.r_[energy,compatibility],np.vstack((energy_j,compatibility_j))
    return model


def augmented_step(state,evaluation,radius,budget,*,cell_only=False,project_compatibility=False,
                   equilibrate=False,velocity_scaled_compatibility=False,thermal_scaled_compatibility=False,
                   pseudo_time_step=None,pseudo_relax_bottom=False):
    model=augmented_model(evaluation,cell_only=cell_only,
                          velocity_scaled_compatibility=velocity_scaled_compatibility,
                          thermal_scaled_compatibility=thermal_scaled_compatibility,pseudo_time_step=pseudo_time_step,
                          pseudo_relax_bottom=pseudo_relax_bottom)
    n=len(state); evaluations=0
    cached_x=cached=None
    def values(z):
        nonlocal cached_x,cached,evaluations
        if cached_x is None or not np.array_equal(cached_x,z):
            cached_x=z.copy();cached=model(z);evaluations+=1
            if evaluations % 50 == 0:
                logging.getLogger(__name__).info('Augmented ML2 evaluation %d: energy %.6g compatibility %.3g',
                    evaluations,np.max(abs(cached[0][:n])),np.max(abs(cached[0][n:])))
        return cached
    bounds=(np.r_[np.full(n,-radius),np.full(n-1,-np.inf)],
            np.r_[np.full(n,radius),np.full(n-1,np.inf)])
    result=least_squares(lambda z:values(z)[0],np.zeros(2*n-1),jac=lambda z:values(z)[1],
        bounds=bounds,x_scale='jac' if equilibrate else np.r_[np.full(n,radius),np.ones(n-1)],
        max_nfev=budget,ftol=None,xtol=1e-14,gtol=1e-10)
    dt=result.x[:n]
    p=evaluation.payload;tr=p['convection_transport']
    target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
    initial_velocity=ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],
        *(tr[k][1:] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')),target)
    p['diagnostic_augmented_proposed_velocity']=(initial_velocity+
        np.maximum(1.,abs(initial_velocity))*result.x[n:])
    if project_compatibility:
        from .inverse_ml2_proposal import inverse_model
        from .dense_helium_limits import DenseHeliumDomainError
        compatible=inverse_model(evaluation,cell_only=cell_only)
        initial=compatible(np.zeros(n))
        initial_merit=np.dot(initial[0],initial[0])
        direction=np.r_[dt[0],result.x[n:]]
        fraction=1.
        dt=np.zeros(n)
        while fraction >= 2.**-24:
            try:
                repaired=compatible(fraction*direction)
            except DenseHeliumDomainError:
                fraction*=.5
                continue
            if (np.max(abs(repaired[2])) <= radius*(1+1e-12)
                    and np.dot(repaired[0],repaired[0]) < initial_merit):
                dt=repaired[2]
                break
            fraction*=.5
        print(f'Augmented compatible projection: fraction={fraction:.6g}, '
              f'max dlnT={np.max(abs(dt)):.6g}',flush=True)
    print(f'Augmented ML2: {result.nfev} evaluations status={result.status} '
          f'energy={np.max(abs(result.fun[:n])):.6g} compatibility={np.max(abs(result.fun[n:])):.3g} '
          f'max dlnT={np.max(abs(dt)):.6g}',flush=True)
    return np.linalg.solve(evaluation.payload['log_temperature_from_state'],dt)
