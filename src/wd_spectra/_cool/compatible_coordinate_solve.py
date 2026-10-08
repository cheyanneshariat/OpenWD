"""Fully reevaluated physical-energy solve on exact-material ML2 coordinates.

The correction residual uses the ML2-compatible flux coordinate to avoid
subtracting nearly adiabatic gradients. Its independently reconstructed
gradient flux must agree at every trial, and actual-gradient total/local
balance must pass at convergence. Output fluxes are never substituted.
"""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra import nonlinear
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import NonlinearEvaluation,RecoverableEvaluationError
from .compatible_temperature_coordinates import CompatibleTemperatures
from .convective_consistency_experiment import MaterialCoefficients
from .measured_proposal_guard import require_measured_proposal


def least_squares_merit(residual):
    """The same smooth objective minimized by the constrained proposal.

    A maximum-row term is not part of that subproblem and can reject a
    least-squares descent direction indefinitely. Worst-row physical limits
    belong to the independent convergence gates, which remain unchanged.
    """
    return .5*float(np.mean(np.asarray(residual)**2))


def linear_trust_fraction(temperature_mapping,direction,radius):
    """Bound the predicted physical step before trying a nonlinear decode.

    An unrestricted Newton direction can leave the representable chemistry
    domain before the nonlinear trust check even runs. This is only a
    proposal bound: the decoded temperature change must still pass the true
    trust radius, and a clipped proposal cannot certify stationarity.
    """
    prediction=temperature_mapping@direction
    maximum=float(np.max(abs(prediction)))
    if not np.isfinite(maximum) or not np.isfinite(radius) or radius<=0:
        raise RecoverableEvaluationError('invalid predicted physical temperature step')
    return min(1.,radius/maximum) if maximum else 1.


def physical_box_direction(jacobian,residual,temperature_mapping,radius):
    """Gauss-Newton trust subproblem bounded in physical nodal log-T.

    Keep the more accurately conditioned coordinate Newton solve when it
    fits. Otherwise solve the constrained linear least-squares problem in
    physical temperature space; do not throttle all layers by the weakest
    mode's Newton size. The nonlinear coordinate map is checked separately.
    No normal equations or approximate atmosphere supplies the accepted flux.
    """
    from scipy.optimize import lsq_linear
    rows=np.max(abs(jacobian),axis=1)
    if np.any(rows==0):raise RuntimeError('unconstrained compatible energy row')
    a=jacobian/rows[:,None];columns=np.max(abs(a),axis=0)
    if np.any(columns==0):raise RuntimeError('unconstrained compatible coordinate')
    direction=np.linalg.solve(a/columns[None,:],-residual/rows)/columns
    if linear_trust_fraction(temperature_mapping,direction,radius)==1.:
        return direction,True
    physical_jacobian=np.linalg.solve(temperature_mapping.T,jacobian.T).T
    bounded=lsq_linear(physical_jacobian,-residual,bounds=(-radius,radius),
        method='bvls',tol=1e-12,max_iter=10*len(direction))
    if not bounded.success or np.any(~np.isfinite(bounded.x)):
        raise RecoverableEvaluationError('bounded compatible temperature subproblem did not solve')
    candidate=np.linalg.solve(temperature_mapping,bounded.x)
    if np.linalg.norm(jacobian@candidate+residual)>np.linalg.norm(residual):
        raise RecoverableEvaluationError('bounded compatible subproblem did not reduce its linear model')
    return candidate,False


def energy_tangent(payload,temperature_mapping,convective_mapping,target):
    """Chain rule without cancelling large thermal-gradient ML2 tangents."""
    radiation=payload['radiative_flux_log_temperature_jacobian']@temperature_mapping
    exchange=payload['radiative_cell_energy_log_temperature_jacobian']@temperature_mapping
    thermal=payload['thermal_cell_emission_log_temperature_jacobian']@temperature_mapping
    scale=payload['cell_energy_scale']
    defect=(payload['radiative_cell_energy_defect']+np.diff(payload['convective_flux_interface']))
    tangent=(radiation+convective_mapping)/target
    tangent[:-1]-=(exchange+np.diff(convective_mapping,axis=0))/scale[:,None]
    tangent[:-1]+=(defect/scale**2)[:,None]*(thermal+convective_mapping[:-1]+convective_mapping[1:])
    return tangent


def coordinate_energy(payload,coordinate_flux,temperature_mapping,convective_mapping,target):
    # This is a solved ML2 coordinate, not a prescribed stellar-flux deficit.
    # All conservation rows remain. The caller retains the independent actual
    # atmosphere in the payload for physical acceptance/convergence checks.
    scale=payload['thermal_cell_emission']+coordinate_flux[:-1]+coordinate_flux[1:]
    values=(payload['radiative_flux_interface']+coordinate_flux)/target-1.
    values=values.copy()
    values[:-1]-=(payload['radiative_cell_energy_defect']+np.diff(coordinate_flux))/scale
    jacobian=None
    if temperature_mapping is not None:
        differentiable={**payload,'convective_flux_interface':coordinate_flux,'cell_energy_scale':scale}
        jacobian=energy_tangent(differentiable,temperature_mapping,convective_mapping,target)
    return values,jacobian


@contextmanager
def compatible_coordinate_solver(material_options):
    original=adaptive.solve_trust_region_newton
    def solve(initial,evaluate,**settings):
        print('COMPATIBLE COORDINATES: building initial physical response',flush=True)
        first=evaluate(initial,True)
        p=first.payload
        if not p['energy_balance_is_physical_flux']:
            raise ValueError('compatible coordinates require physical-only energy equations')
        material=MaterialCoefficients(*(material_options[k] for k in
            ('with_temperature','thermodynamics','rosseland_opacity','mixing_length_alpha')))
        target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
        linear=p['log_temperature_from_state']
        coords=CompatibleTemperatures(np.log(p['atmosphere'].temperature),
            p['atmosphere'].gas_pressure,material,target)
        print('COMPATIBLE COORDINATES: actual-material chart constructed',flush=True)
        def physical_state(q):return np.linalg.solve(linear,coords.decode(q)[0])
        def current(q,need_jacobian):
            logt,mapping,compatibility=coords.decode(q)
            ev=evaluate(np.linalg.solve(linear,logt),need_jacobian)
            payload={**ev.payload,'experimental_exact_material_coordinates':True,
                'experimental_coordinate_energy_flux':'independently checked ML2 coordinate',
                'coordinate_flux_compatibility_error':compatibility,
                'coordinate_temperature_mapping':mapping}
            # Confirm the incoming equation selection before constructing its
            # equivalent, independently ML2-checked coordinate representation.
            expected=payload['total_flux_interface']/target-1.
            expected=expected.copy()
            expected[:-1]-=(payload['radiative_cell_energy_defect']+
                np.diff(payload['convective_flux_interface']))/payload['cell_energy_scale']
            if not np.allclose(ev.residual,expected,rtol=1e-10,atol=1e-12):
                raise ValueError('compatible coordinate experiment requires current-energy rows')
            coordinate_flux=np.r_[0.,np.maximum(q[1:]*coords.scale,0.)**3]*target
            discrepancy=float(np.max(abs(coordinate_flux-payload['convective_flux_interface']))/target)
            if discrepancy>1e-7:
                print(f'COMPATIBLE TRIAL REJECTED: flux compatibility {discrepancy:.6g}',flush=True)
                raise RecoverableEvaluationError('coordinate flux disagrees with the actual ML2 flux')
            values,jacobian=coordinate_energy(payload,coordinate_flux,
                mapping if need_jacobian else None,coords.convective_tangent(q),target)
            if not need_jacobian:
                print(f'COMPATIBLE TRIAL: mean_square={np.mean(values**2):.9g}, '
                    f'max_flux={np.max(abs(payload["total_flux_interface"]/target-1)):.9g}, '
                    f'max_local={np.max(abs(payload["cell_energy_balance_relative_residual"])):.9g}, '
                    f'max_Fconv_over_Fstar={np.max(coordinate_flux/target):.9g}, '
                    f'compatibility={discrepancy:.3g}',flush=True)
            return NonlinearEvaluation(values,jacobian,payload)
        def measure(old,new):
            return float(np.max(abs(coords.decode(new)[0]-coords.decode(old)[0])))
        def step(q,ev,j,radius):
            direction,root_step=physical_box_direction(j,ev.residual,
                ev.payload['coordinate_temperature_mapping'],radius)
            factor=linear_trust_fraction(ev.payload['coordinate_temperature_mapping'],direction,radius)
            for _ in range(64):
                try:
                    size=measure(q,q+factor*direction)
                except RecoverableEvaluationError:
                    size=np.inf
                if size<=radius:
                    ev.payload['diagnostic_inner_proposal_root_solved']=root_step and factor==1.
                    print(f'COMPATIBLE COORDINATE PROPOSAL: fraction={factor:g}, '
                        f'max_dlnT={size:.6g}, root_step={root_step}, unscaled_linear_error='
                        f'{np.max(abs(j@direction+ev.residual)):.3g}',flush=True)
                    return factor*direction
                factor*=.5
            raise RecoverableEvaluationError('no resolved compatible temperature proposal')
        tolerance=settings['residual_tolerance']
        # The incoming gate closes over the old proposal builder. Rebuild
        # the SAME actual-flux/local-energy gate for these coordinates, then
        # apply a new measured, non-clipped-correction guard below.
        def converged(q,ev,size):
            payload=ev.payload
            return (np.max(abs(payload['total_flux_interface']/target-1))<tolerance and
                np.max(abs(payload['cell_energy_balance_relative_residual']))<tolerance)
        callback=settings.get('callback')
        if callback is not None:
            settings['callback']=lambda record,q,ev:callback(record,physical_state(q),ev)
        settings.update(step_builder=step,step_measure=measure,convergence_test=converged,
            trial_projector=None,jacobian_refresh_interval=1,allow_initial_convergence=False,
            finite_difference_fallback_step=None)
        result=original(coords.initial,current,**require_measured_proposal(settings))
        state=physical_state(result.state)
        final=evaluate(state,False)
        return replace(result,state=state,evaluation=final)
    with patch.object(adaptive,'solve_trust_region_newton',solve), \
            patch.object(nonlinear,'_residual_merit',least_squares_merit):
        yield
