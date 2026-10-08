"""Linearly implicit local pseudo-time initialization, not a static solution.

Positive heating raises log T; cooling lowers it. Physical radiative and ML2
cell energy are reevaluated at every trial. A frozen positive local transport
rate is a time preconditioner, NOT heat capacity or a physical evolution time.
The final atmosphere still goes through the unmodified static convergence
gates. See Kelley & Keyes (1998), and PETSc TSPSEUDO's one-Newton-step method.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import RecoverableEvaluationError,NonlinearEvaluation
from .dense_helium_limits import DenseHeliumDomainError


def pseudo_direction(residual,jacobian,step,*,relax_boundary=False):
    """Thermal rows evolve; optionally relax the bottom heat reservoir too.

    Bottom residual is F_bottom/Fstar-1: opposite sign to local heating.
    Its positive inertia therefore cools an overcarrying bottom node. The
    stationary zero is unchanged, but a convection-onset boundary need not
    satisfy a nonlinear algebraic condition in one linearized step.
    """
    matrix=np.array(jacobian,copy=True)
    nodes=np.arange(len(residual)-1)
    matrix[nodes,nodes]-=1./step
    if relax_boundary:matrix[-1,-1]+=1./step
    row=np.max(abs(matrix),axis=1)
    if np.any(row==0):raise ValueError('singular pseudo-time equation')
    scaled=matrix/row[:,None];column=np.max(abs(scaled),axis=0)
    if np.any(column==0):raise ValueError('singular pseudo-time variable')
    return np.linalg.solve(scaled/column[None,:],-residual/row)/column


def physical_rows(payload,scale,target,need_jacobian):
    energy=payload['radiative_cell_energy_defect']+np.diff(payload['convective_flux_interface'])
    residual=np.r_[energy/scale,payload['total_flux_interface'][-1]/target-1.]
    if not need_jacobian:return residual,None
    # These are derivatives of the ACTUAL temperature-gradient ML2 flux.
    # The surface convective flux is identically zero. Telescope its actual
    # cell differences to recover the bottom derivative (not a flux repair).
    if payload['convective_flux_interface'][0]!=0:
        raise ValueError('pseudo-time boundary requires zero surface convective flux')
    bottom_convection=np.sum(payload['cell_energy_log_temperature_jacobian']
        -payload['radiative_cell_energy_log_temperature_jacobian'],axis=0)
    jacobian=np.vstack((payload['cell_energy_log_temperature_jacobian']/scale[:,None],
        (payload['radiative_flux_log_temperature_jacobian'][-1]+bottom_convection)/target))
    return residual,jacobian


@contextmanager
def pseudo_time_initializer(sweeps,output,*,implicit_convection=False,until_local_balance=False,
                            relax_boundary=False,tangent_convection=False):
    if implicit_convection and tangent_convection:
        raise ValueError('choose either exact dense materials or the explicit local material tangent')
    output=Path(output)
    if output.exists():raise ValueError('refusing to overwrite pseudo-time telemetry')
    output.parent.mkdir(parents=True,exist_ok=True)
    original=adaptive.solve_trust_region_newton
    used=False
    def solve(initial,evaluate,**settings):
        nonlocal used
        first=evaluate(initial,False)
        if used or not first.payload['energy_balance_is_physical_flux']:
            return original(initial,evaluate,**settings)
        used=True;state=initial.copy();step=1.
        with output.open('x') as telemetry:
            for iteration in range(sweeps):
                print(f'PSEUDO TIME initializer {iteration+1}: actual material/transfer Jacobian',flush=True)
                base=evaluate(state,True);p=base.payload
                if p.get('material_temperature_response_domain_limited',False):
                    # Pseudo-time is only a warm-up, not physical evolution.
                    # A thermal transient can approach a table boundary while
                    # the constrained static solution lies inside it. Do not
                    # squeeze that transient against the boundary indefinitely;
                    # hand the SAME valid state to the full flux/energy solve.
                    print('PSEUDO TIME material-domain boundary: handing to full static solve; '
                          'NOT a convergence claim and no EOS substitution',flush=True)
                    break
                if until_local_balance and np.max(abs(p['cell_energy_balance_relative_residual']))<settings['residual_tolerance']:
                    print('PSEUDO TIME local balance reached: handing to full static solve; NOT a convergence claim',flush=True)
                    break
                mapping=p['log_temperature_from_state'];logt=np.log(p['atmosphere'].temperature)
                target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
                scale=p['cell_energy_scale'].copy()
                residual,jacobian=physical_rows(p,scale,target,True)
                accepted=False
                # Permit the full floating-point range of useful halvings;
                # efficient convection can have a much shorter local time
                # scale than a radiative layer. Stop on an unrepresentable
                # temperature step, not an atmosphere-specific retry limit.
                for attempt in range(64):
                    if implicit_convection or tangent_convection:
                        from .augmented_ml2_proposal import augmented_step
                        if tangent_convection:
                            from .tangent_ml2_materials import TangentML2Materials
                            materials=TangentML2Materials(p)
                        else:
                            materials=p['diagnostic_dense_exact_material_model']
                        inner=NonlinearEvaluation(base.residual,base.jacobian,{**p,
                            'diagnostic_material_model':materials,
                            'diagnostic_positive_radiative_rates':True})
                        direction=augmented_step(state,inner,.04,1000,velocity_scaled_compatibility=True,
                            thermal_scaled_compatibility=True,pseudo_time_step=step,
                            pseudo_relax_bottom=relax_boundary)
                        delta=mapping@direction
                    else:
                        delta=pseudo_direction(residual,jacobian,step,relax_boundary=relax_boundary)
                    if np.any(~np.isfinite(delta)) or np.max(abs(delta))>.04:
                        if attempt%4==0:
                            print(f'PSEUDO TIME retry {attempt+1}: step={step:g}, temperature bound',flush=True)
                        step*=.5;continue
                    try:
                        candidate=state+np.linalg.solve(mapping,delta)
                        if np.array_equal(mapping@candidate,mapping@state):
                            print('PSEUDO TIME step is below temperature resolution',flush=True)
                            break
                        trial=evaluate(candidate,False)
                    except (DenseHeliumDomainError,RecoverableEvaluationError):
                        step*=.5;continue
                    updated,_=physical_rows(trial.payload,scale,target,False)
                    temporal=updated.copy();temporal[:-1]-=delta[:-1]/step
                    if relax_boundary:temporal[-1]+=delta[-1]/step
                    relative_error=float(np.max(abs(temporal))/max(np.max(abs(residual)),1e-12))
                    if relative_error>.25:
                        if attempt%4==0:
                            print(f'PSEUDO TIME retry {attempt+1}: step={step:g}, temporal defect={relative_error:g}',flush=True)
                        step*=.5;continue
                    accepted=True;state=candidate
                    row=dict(initialization_only=True,static_convergence_claim=False,sweep=iteration+1,
                        nonlinear_convective_proposal=implicit_convection or tangent_convection,
                        material_tangent_only=tangent_convection,
                        bottom_boundary_thermal_reservoir=relax_boundary,
                        handoff_on_local_balance=until_local_balance,
                        pseudo_time_step=step,nonlinear_temporal_defect=relative_error,
                        maximum_log_temperature_change=float(np.max(abs(delta))),
                        maximum_local_energy=float(np.max(abs(trial.payload['cell_energy_balance_relative_residual']))),
                        maximum_stellar_flux_defect=float(np.max(abs(trial.payload['total_flux_interface']/target-1))),
                        surface_temperature=float(trial.payload['atmosphere'].temperature[0]))
                    telemetry.write(json.dumps(row)+'\n');telemetry.flush()
                    print('PSEUDO TIME (not converged): '+json.dumps(row),flush=True)
                    atmosphere=trial.payload['atmosphere']
                    np.savez_compressed(output.with_suffix('.npz'),experimental_temperature=atmosphere.temperature,
                        experimental_gas_pressure=atmosphere.gas_pressure,
                        experimental_column_mass=atmosphere.column_mass,
                        experimental_rosseland_optical_depth=atmosphere.rosseland_optical_depth,
                        initialization_only=True,static_convergence_claim=False)
                    # Nonlinear temporal-defect control, not a requirement
                    # that static energy decrease during a thermal transient.
                    if relative_error<.0625:step*=2.
                    break
                if not accepted:
                    print('PSEUDO TIME initializer cannot take a resolved admissible step',flush=True)
                    break
                if row['maximum_local_energy']<1e-4 and row['maximum_stellar_flux_defect']<1e-4:
                    break
        settings['allow_initial_convergence']=False
        return original(state,evaluate,**settings)
    with patch.object(adaptive,'solve_trust_region_newton',solve):
        yield
