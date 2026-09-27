"""Research-only elimination of ML2 gradient compatibility in the cheap model.

Unknowns are the surface log-temperature correction and signed ML2 element
velocities. Compatibility is solved, not substituted for physical energy.
Outer evaluations still recompute the actual atmospheric flux from T.
"""
import numpy as np
import logging
from scipy.optimize import least_squares, minimize
from wd_spectra._ml2_auxiliary import (
    ml2_auxiliary_from_gradient, ml2_scaled_coefficients,
    ml2_auxiliary_compatibility,
)
from wd_spectra.constants import STEFAN_BOLTZMANN
from .radiative_rate_proposal import radiative_rates


def inverse_model(evaluation, *, cell_only=False):
    p=evaluation.payload
    target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
    n=p['atmosphere'].n_depth
    gradient=p['temperature_gradient'][1:]
    operator=p['interface_gradient_operator'][1:]
    transport=p['convection_transport']
    ad,loss,coeff=(transport[k][1:] for k in
        ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
    ad_j,loss_j,coeff_j=(v[1:] for v in p['ml2_coefficient_log_temperature_responses'])
    loss_log_j=loss_j/loss[:,None]; coeff_log_j=coeff_j/coeff[:,None]
    y0=ml2_auxiliary_from_gradient(gradient,ad,loss,coeff,target)
    yscale=np.maximum(1.,abs(y0))
    radiation=p['radiative_flux_interface']/target
    radiation_j=p['radiative_flux_log_temperature_jacobian']/target
    cell=p['radiative_cell_energy_defect']/target
    cell_j=p['radiative_cell_energy_log_temperature_jacobian']/target
    emission=p['thermal_cell_emission']/target
    emission_log_j=p['thermal_cell_emission_log_temperature_jacobian']/p['thermal_cell_emission'][:,None]

    def model(z):
        y=y0+yscale*z[1:]
        dt=np.full(n,z[0])
        for _ in range(20):
            exact_material=p.get('diagnostic_material_model')
            if exact_material is None:
                a=ad+ad_j@dt
                b=loss*np.exp(loss_log_j@dt)
                c=coeff*np.exp(coeff_log_j@dt)
                responses=(ad_j,b[:,None]*loss_log_j,c[:,None]*coeff_log_j)
            else:
                if not hasattr(exact_material,'evaluate'):
                    raise ValueError('inverse material solve requires exact EOS, not a bounded surrogate')
                coefficients,full_responses=exact_material.evaluate(dt)
                a,b,c=(v[1:] for v in coefficients)
                responses=tuple(v[1:] for v in full_responses)
            defect,tangent,dy=ml2_auxiliary_compatibility(gradient+operator@dt,y,
                (a,b,c),responses,operator,target)
            equations=np.concatenate(([dt[0]-z[0]],defect))
            matrix=np.vstack((np.eye(n)[0],tangent))
            if np.max(abs(equations))<1e-13: break
            dt-=np.linalg.solve(matrix,equations)
        else:
            raise RuntimeError('Inner ML2 compatibility did not solve; no replacement state used')
        rhs=np.zeros((n,n)); rhs[0,0]=1
        rhs[1:,1:]=np.diag(-dy*yscale)
        dt_j=np.linalg.solve(matrix,rhs)
        flux=np.concatenate(([0.],np.maximum(y,0.)**3))
        flux_j=np.zeros((n,n)); flux_j[1:,1:]=np.diag(3*np.maximum(y,0.)**2*yscale)
        exchange,exchange_j,thermal,thermal_j=radiative_rates(p,dt)
        thermal_j=thermal_j@dt_j
        energy=exchange+np.diff(flux)
        energy_j=exchange_j@dt_j+np.diff(flux_j,axis=0)
        scale=thermal+flux[:-1]+flux[1:]
        scale_j=thermal_j+flux_j[:-1]+flux_j[1:]
        if cell_only:
            residual=np.concatenate(([radiation[0]+radiation_j[0]@dt-1],energy/scale))
            jacobian=np.vstack((radiation_j[0]@dt_j,
                energy_j/scale[:,None]-(energy/scale**2)[:,None]*scale_j))
            return residual,jacobian,dt,dt_j
        residual=radiation+radiation_j@dt+flux-1
        jacobian=radiation_j@dt_j+flux_j
        residual[:-1]-=energy/scale
        jacobian[:-1]-=energy_j/scale[:,None]-(energy/scale**2)[:,None]*scale_j
        return residual,jacobian,dt,dt_j
    return model


def inverse_step(state,evaluation,radius,budget,*,cell_only=False):
    model=inverse_model(evaluation,cell_only=cell_only)
    result=least_squares(lambda z:model(z)[0],np.zeros_like(state),
        jac=lambda z:model(z)[1],bounds=(-radius,radius),x_scale='jac',
        max_nfev=budget,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    values,_,dt,_=model(result.x)
    dt*=min(1.,radius/max(np.max(abs(dt)),1e-300))
    print(f'Inverse ML2 proposal: {result.nfev} evaluations, status={result.status}, '
          f'model residual={np.max(abs(values)):.6g}, max dlnT={np.max(abs(dt)):.6g}',flush=True)
    return np.linalg.solve(evaluation.payload['log_temperature_from_state'],dt)


def newton_inverse_step(state,evaluation,radius,budget,*,cell_only=False):
    """Damped Newton in compatible velocity coordinates, with true T bounds.

    The linear system is equilibrated, not regularized or fitted. Every
    line-search point solves finite-material gradient compatibility anew;
    neither its temperatures nor its convective flux are clipped afterward.
    A proposal need not solve the cheap model exactly: the outer iteration
    recomputes radiation and all materials before accepting any change.
    """
    from .dense_helium_limits import DenseHeliumDomainError
    model=inverse_model(evaluation,cell_only=cell_only)
    z=np.zeros_like(state)
    current=model(z)
    evaluations=1
    steps=0
    while evaluations < budget:
        r,j,dt,dtj=current
        if np.max(abs(r)) < 1e-10:
            break
        rows=np.max(abs(j),axis=1)
        if np.any(rows == 0):
            raise RuntimeError('singular compatible energy equation')
        equilibrated=j/rows[:,None]
        columns=np.max(abs(equilibrated),axis=0)
        if np.any(columns == 0):
            raise RuntimeError('unconstrained compatible energy coordinate')
        dz=np.linalg.solve(equilibrated/columns[None,:],-r/rows)/columns
        merit=np.dot(r,r)
        fraction=1.
        accepted=False
        while evaluations < budget and fraction >= 2.**-24:
            evaluations+=1
            try:
                trial=model(z+fraction*dz)
            except DenseHeliumDomainError:
                fraction*=.5
                continue
            if (np.max(abs(trial[2])) <= radius*(1+1e-12)
                    and np.dot(trial[0],trial[0]) <= (1-1e-4*fraction)*merit):
                z+=fraction*dz
                current=trial
                accepted=True
                steps+=1
                break
            fraction*=.5
        if not accepted:
            break
    r,_,dt,_=current
    print(f'Compatible Newton ML2: {evaluations} evaluations, {steps} steps; '
          f'model residual={np.max(abs(r)):.6g}, max dlnT={np.max(abs(dt)):.6g}',flush=True)
    return np.linalg.solve(evaluation.payload['log_temperature_from_state'],dt)


def constrained_inverse_step(state,evaluation,radius,budget,*,cell_only=False,allow_onset=False):
    """Enforce the ACTUAL temperature bound inside the compatibility solve.

    Scaling an already-compatible temperature step afterward destroys its
    convective-flux prediction. Nonlinear inequality constraints avoid that
    mismatch without changing any physical equation or acceptance gate.
    """
    model=inverse_model(evaluation,cell_only=cell_only)
    saved_x,saved_value=None,None
    def cached(z):
        nonlocal saved_x,saved_value
        if saved_x is None or not np.array_equal(saved_x,z):
            saved_x=z.copy(); saved_value=model(z)
        return saved_value
    def objective(z):
        r,j,_,_=cached(z)
        return .5*np.dot(r,r),j.T@r
    def constraints(z):
        dt=cached(z)[2]
        return np.concatenate((radius-dt,radius+dt))/radius
    def constraints_j(z):
        j=cached(z)[3]
        return np.vstack((-j,j))/radius
    bounds=[(-radius,radius)]*len(state)
    if allow_onset:
        p=evaluation.payload; tr=p['convection_transport']
        target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
        y0=ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],
            *(tr[k][1:] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')),target)
        scale=np.maximum(1.,abs(y0))
        # A temperature-radius box on fractional velocity changes can NEVER
        # cross y=0 from a stable cell with |y0|>1: it just approaches onset
        # geometrically. Admit the natural +/- stellar-flux velocity range.
        # The actual T inequalities, not this coordinate box, enforce trust.
        bounds[1:]=list(zip(-np.maximum(radius,(y0+1)/scale),
                            np.maximum(radius,(1-y0)/scale)))
    inner_iterations=0
    def progress(z):
        nonlocal inner_iterations
        inner_iterations+=1
        if inner_iterations % 10 == 0:
            values=cached(z)
            logging.getLogger(__name__).info(
                'Constrained ML2 inner iteration %d: model max %.6g; T bound excess %.3g',
                inner_iterations,np.max(abs(values[0])),np.max(abs(values[2]))-radius)
    result=minimize(objective,np.zeros_like(state),jac=True,method='SLSQP',
        bounds=bounds,
        constraints={'type':'ineq','fun':constraints,'jac':constraints_j},
        callback=progress,options={'maxiter':budget,'ftol':1e-11})
    r,_,dt,_=cached(result.x)
    if np.max(abs(dt)) > radius*(1+1e-8):
        raise RuntimeError(f'Constrained inverse proposal infeasible: {result.message}')
    print(f'Constrained inverse ML2: iterations={result.nit} status={result.status} '
          f'model residual={np.max(abs(r)):.6g} max dlnT={np.max(abs(dt)):.6g}',flush=True)
    return np.linalg.solve(evaluation.payload['log_temperature_from_state'],dt)
