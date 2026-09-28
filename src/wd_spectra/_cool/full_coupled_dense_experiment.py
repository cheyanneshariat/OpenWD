"""Fully reevaluated temperature/ML2 system, not a cheap inner proposal.

Auxiliary flux is allowed off the compatibility manifold during Newton
iteration, but both exact compatibility AND actual-gradient physical flux
and local energy must pass before convergence. No output flux is replaced.
"""
import numpy as np
from wd_spectra import _ml2_auxiliary as auxiliary
from wd_spectra.nonlinear import NonlinearEvaluation

original_auxiliary_solver=auxiliary.solve_auxiliary_ml2_experiment


def bounded_schur_step(ev,j,radius,n,mapping):
    """Eliminate linear ML2 constraints; bound physical log-T corrections.

    A huge unconstrained Newton step near a weak thermal response can waste
    every correction on a nearly null direction. BVLS minimizes the linear
    energy error inside the same trust box while satisfying all linear ML2
    constraints exactly. No physical flux or final tolerance is altered.
    """
    from scipy.optimize import lsq_linear
    temperature_j=np.linalg.solve(mapping.T,j[:,:n].T).T
    # Analytic ML2 has a diagonal velocity block. The generic driver can
    # supply a dense rank-one secant update after a rejected direction, so
    # eliminate the actual supplied block instead of assuming diagonality.
    cy=j[n:,n:]
    velocity_offset=np.linalg.solve(cy,-ev.residual[n:])
    velocity_tangent=np.linalg.solve(cy,-temperature_j[n:])
    energy_j=temperature_j[:n]+j[:n,n:]@velocity_tangent
    energy_r=ev.residual[:n]+j[:n,n:]@velocity_offset
    result=lsq_linear(energy_j,-energy_r,bounds=(-radius,radius),method='bvls',tol=1e-11)
    if not result.success:raise RuntimeError('bounded full-system energy proposal failed')
    dt=result.x;dy=velocity_offset+velocity_tangent@dt
    direction=np.r_[np.linalg.solve(mapping,dt),dy]
    print(f'BOUNDED FULL COUPLED STEP: linear energy {np.max(abs(energy_r+energy_j@dt)):.6g}; '
        f'linear compatibility {np.max(abs(ev.residual[n:]+j[n:]@direction)):.3g}; '
        f'maximum dlnT {np.max(abs(dt)):.6g}',flush=True)
    return direction


def current_augmented_energy(payload,y,scale,target,mapping=None,*,normalization=None):
    n=len(y)+1
    flux=np.r_[0.,np.maximum(y,0.)**3]
    radiation=payload['radiative_flux_interface']/target
    thermal=payload['thermal_cell_emission']/target
    exchange=payload['radiative_cell_energy_defect']/target
    norm=thermal+flux[:-1]+flux[1:] if normalization is None else np.asarray(normalization)
    if norm.shape!=thermal.shape or np.any(~np.isfinite(norm)) or np.any(norm<=0):
        raise ValueError('energy normalization must be positive, finite and match the cells')
    q=(exchange+np.diff(flux))/norm
    r=radiation+flux-1
    r[:-1]-=q
    if mapping is None: return r,None
    flux_j=np.zeros((n,n-1));flux_j[1:]=np.diag(3*np.maximum(y,0.)**2*scale)
    radiative_j=payload['radiative_flux_log_temperature_jacobian']@mapping/target
    exchange_j=payload['radiative_cell_energy_log_temperature_jacobian']@mapping/target
    thermal_j=payload['thermal_cell_emission_log_temperature_jacobian']@mapping/target
    q_t=exchange_j/norm[:,None]
    q_y=np.diff(flux_j,axis=0)/norm[:,None]
    if normalization is None:
        q_t-=q[:,None]*thermal_j/norm[:,None]
        q_y-=q[:,None]*(flux_j[:-1]+flux_j[1:])/norm[:,None]
    jacobian=np.hstack((radiative_j,flux_j))
    jacobian[:-1]-=np.hstack((q_t,q_y))
    return r,jacobian


def current_compatibility(payload,y,scale,target,mapping=None,*,thermal_scaling=False,normalization=None):
    from .scaled_ml2_compatibility import scaled_compatibility
    tr=payload['convection_transport']
    coefficients=tuple(tr[k][1:] for k in
        ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
    responses=operator=None
    if mapping is not None:
        responses=tuple(v[1:]@mapping for v in payload['ml2_coefficient_log_temperature_responses'])
        operator=payload['interface_gradient_operator'][1:]@mapping
    thermal_options={}
    if thermal_scaling:
        thermal=payload['thermal_cell_emission']/target
        thermal_options['thermal_scale']=np.r_[.5*(thermal[:-1]+thermal[1:]),thermal[-1]]
        if mapping is not None:
            tj=payload['thermal_cell_emission_log_temperature_jacobian']@mapping/target
            thermal_options['thermal_scale_jacobian']=np.vstack((.5*(tj[:-1]+tj[1:]),tj[-1]))
    closure,tangent,yj=scaled_compatibility(payload['temperature_gradient'][1:],
        y,coefficients,responses,operator,target,normalization=normalization,**thermal_options)
    jacobian=None if mapping is None else np.hstack((tangent,np.diag(yj*scale)))
    return closure,jacobian


def full_coupled_solve(initial,evaluate,solver,target_flux,*,local_energy_scaling=False,
                       velocity_scaled_compatibility=False,thermal_scaled_compatibility=False,
                       frozen_local_scaling=False,bounded_temperature_step=False,
                       material_projection_context=None,**options):
    if local_energy_scaling:
        raise ValueError('full coupled experiment uses current, not frozen, local scales')
    if thermal_scaled_compatibility and not velocity_scaled_compatibility:
        raise ValueError('thermal compatibility requires current velocity scaling')
    n=len(initial)
    def full_solver(z0,augmented_evaluate,**settings):
        first=augmented_evaluate(z0,False)
        p=first.payload;tr=p['convection_transport']
        y0=auxiliary.ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],
            *(tr[k][1:] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')),target_flux)
        scale=np.maximum(1.,abs(y0))
        weights={}
        def current(state,need_jacobian):
            ev=augmented_evaluate(state,need_jacobian)
            p=ev.payload
            if frozen_local_scaling and (need_jacobian or not weights):
                from .scaled_ml2_compatibility import compatibility_normalization
                y=state[n:]*scale
                flux=np.r_[0.,np.maximum(y,0.)**3]
                thermal=p['thermal_cell_emission']/target_flux
                coefficients=tuple(p['convection_transport'][k][1:] for k in
                    ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
                face=np.r_[.5*(thermal[:-1]+thermal[1:]),thermal[-1]]
                weights.update(energy=thermal+flux[:-1]+flux[1:],
                    compatibility=compatibility_normalization(y,coefficients,target_flux,
                        face if thermal_scaled_compatibility else None))
            energy,j=current_augmented_energy(p,state[n:]*scale,scale,target_flux,
                p['log_temperature_from_state'] if need_jacobian else None,
                normalization=weights.get('energy'))
            r=ev.residual.copy();r[:n]=energy
            if j is not None:
                j=np.vstack((j,ev.jacobian[n:]))
            if velocity_scaled_compatibility:
                closure,tangent=current_compatibility(p,state[n:]*scale,scale,target_flux,
                    p['log_temperature_from_state'] if need_jacobian else None,
                    thermal_scaling=thermal_scaled_compatibility,normalization=weights.get('compatibility'))
                r[n:]=closure
                if j is not None: j[n:]=tangent
            return NonlinearEvaluation(r,j,p)
        def step(state,ev,j,radius):
            if bounded_temperature_step:
                return bounded_schur_step(ev,j,radius,n,ev.payload['log_temperature_from_state'])
            rows=np.max(abs(j),axis=1)
            if np.any(rows == 0): raise RuntimeError('singular full coupled equation')
            a=j/rows[:,None]
            cols=np.max(abs(a),axis=0)
            if np.any(cols == 0): raise RuntimeError('unconstrained full coupled coordinate')
            return np.linalg.solve(a/cols[None,:],-ev.residual/rows)/cols
        old_convergence=settings['convergence_test']
        tolerance=settings.get('residual_tolerance',2e-3)
        def converged(state,ev,step):
            local=ev.payload['cell_energy_balance_relative_residual']
            return np.max(abs(local)) < tolerance and old_convergence(state,ev,step)
        settings.update(step_builder=step,convergence_test=converged,
            jacobian_refresh_interval=1,allow_initial_convergence=False)
        if frozen_local_scaling and not velocity_scaled_compatibility:
            raise ValueError('frozen full-system scaling requires current compatibility weights')
        if material_projection_context is not None:
            from functools import partial
            from .full_ml2_manifold_projection import project_full_trial
            # The state mapping is only published on a differentiated call.
            mapping=current(z0,True).payload['log_temperature_from_state']
            settings['trial_projector']=partial(project_full_trial,scale=scale,mapping=mapping,
                options=material_projection_context['options'])
        from .measured_proposal_guard import require_measured_proposal
        settings=require_measured_proposal(settings)
        return solver(z0,current,**settings)
    return original_auxiliary_solver(initial,evaluate,full_solver,target_flux,
        local_energy_scaling=False,**options)
