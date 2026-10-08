"""Cheap nonlinear ML2 trial model with the global radiative/material tangent.

Only the trial model is approximate. Radiation/heating use their full nonlocal
temperature response. Positive material coefficients and thermal emission use
log-linear predictions matching those same first derivatives. ML2 and local
energy normalization retain their full algebraic nonlinearity. The bounded
inner solve makes no additional EOS/opacity/transfer calls; the shared outer
solver evaluates actual physics before accepting anything.
"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
from scipy.optimize import least_squares
import wd_spectra.adaptive_structure as adaptive
from wd_spectra.nonlinear import (
    NonlinearCorrection, NonlinearEvaluation, NonlinearProposal,
    RecoverableEvaluationError,
)
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from .dq_augmented_convection import coupled_direction
from .dq_explicit_gradient import ExplicitGradientSystem


def trial_bounds(system, state, evaluation, radius):
    """Bound temperatures and protect newly physical deep gradients.

    An explicit-gradient coordinate can move arbitrarily far toward the
    stable branch without changing the ML2 flux: both the auxiliary and the
    independently evaluated physical convection are exactly zero there.
    Applying the temperature trust radius to that algebraic motion can make a
    harmless compatibility update throttle the actual temperature Newton
    step.  At interfaces made physical by a same-run lower-domain extension,
    the release stops at nabla=0 so the former terminal reservoir cannot turn
    into a deep inversion.  Non-grey surface inversions remain available.
    Motion toward convection remains bounded, as do all temperatures.
    """
    lower = np.full_like(state, -radius, dtype=float)
    upper = np.full_like(state, radius, dtype=float)
    if not getattr(system, 'one_sided_stable_gradient_coordinate', False):
        return lower, upper
    n = system.n
    payload = evaluation.payload
    auxiliary = np.asarray(payload['dq_augmented_auxiliary_flux'])[1:]
    physical = np.asarray(payload['convective_flux_interface'])[1:]
    if auxiliary.shape != (n-1,) or physical.shape != (n-1,):
        raise ValueError('Explicit-gradient flux shape does not match its coordinates')
    # The closure constructs these fluxes as max(y, 0)^3.  Requiring exact
    # zeros makes this a branch-identity decision, not a numerical threshold.
    inactive = (auxiliary == 0.) & (physical == 0.)
    system._trust_inactive_gradient_coordinates = inactive.copy()
    minimum = np.asarray(system.minimum_gradient_coordinate(evaluation), dtype=float)
    if minimum.shape != (n-1,) or np.any(np.isnan(minimum)):
        raise ValueError('Explicit-gradient coordinate floor is invalid')
    # Keep delta=0 feasible if a diagnostic restart already lies infinitesimally
    # beyond the floor.  Fresh cold states are admissible, and every accepted
    # trial below is independently guarded using its actual temperature field.
    floor_step = np.minimum(minimum-state[n:], 0.)
    lower[n:] = np.maximum(lower[n:], floor_step)
    lower[n:][inactive] = floor_step[inactive]
    return lower, upper


def admissible_temperature_gradient(evaluation, first_index=None):
    """Guard only newly physical lower interfaces against inversion."""
    gradient = np.asarray(evaluation.payload['temperature_gradient'], dtype=float)
    if gradient.ndim != 1 or gradient.size < 2 or np.any(~np.isfinite(gradient)):
        raise ValueError('Physical temperature gradient is invalid')
    if first_index is None:
        return True
    if not 1 <= first_index < gradient.size:
        raise ValueError('Deep gradient guard starts outside the atmosphere')
    allowance = 128*np.finfo(float).eps*max(1., float(np.max(abs(gradient))))
    return bool(np.all(gradient[first_index:] >= -allowance))


def compatible_stationarity(system, state):
    """Return an exact auxiliary-closure projection at a physical solution.

    The explicit gradient is an algebraic solver coordinate, not atmosphere
    state.  Once the independently evaluated temperature solution passes all
    physical gates, rebuild that coordinate from the unchanged temperature
    and the exact ML2 closure.  This cannot repair or alter a physical flux;
    it only prevents a nearly null stable coordinate from blocking a measured
    stationarity claim.
    """
    tolerance = getattr(system, 'requested_residual_tolerance', None)
    if tolerance is None or not hasattr(system, 'state_from_temperature'):
        return None
    n = system.n
    physical = system.physical(state[:n], False)
    compatible = np.asarray(
        system.state_from_temperature(state[:n], physical), dtype=float
    )
    if compatible.shape != state.shape or np.any(~np.isfinite(compatible)):
        raise ValueError('Compatible explicit-gradient state is invalid')
    projected = system.evaluate(compatible, False)
    payload = projected.payload
    if (
        np.max(abs(projected.residual)) >= tolerance
        or np.max(abs(payload['total_flux_interface']/system.target-1.)) >= tolerance
        or np.max(abs(payload['cell_energy_balance_relative_residual'])) >= tolerance
        or payload['dq_augmented_flux_compatibility'] >= tolerance
    ):
        return None
    direction = compatible-state
    if np.max(abs(direction[:n])) != 0.:
        raise ValueError('Auxiliary closure projection changed temperature')
    return direction


def local_model(system, state, ev):
    n = system.n; p = ev.payload; anchor = state[:n].copy()
    # The supplied accepted evaluation already owns these exact material
    # values/tangents. A rejected physical trial may have displaced the
    # one-entry atmosphere cache: revisiting it here rebuilt EOS, opacity
    # and transfer just to fetch values already present in this payload.
    values = tuple(p['convection_transport'][key] for key in
        ('adiabatic_gradient', 'ml2_radiative_loss', 'ml2_flux_coefficient'))
    responses = p['ml2_coefficient_log_temperature_responses']
    ratios = tuple(np.divide(d, v[:, None], out=np.zeros_like(d), where=v[:, None]!=0)
                   for v, d in zip(values, responses))
    thermal0 = p['thermal_cell_emission']
    thermal_ratio = p['thermal_cell_emission_log_temperature_jacobian']/thermal0[:, None]
    radiation_remainder = getattr(system, 'radiation_trial_remainder', None)
    coefficients = None
    if getattr(system, 'eos_trial_owner', None) is not None:
        from .dq_eos_knot_model import coefficient_model
        coefficients = coefficient_model(system, state, ev, anchor_material=(values, responses))

    def predicted(logt, need):
        dx = logt-anchor
        current = tuple(v*np.exp(d@dx) for v, d in zip(values, ratios))
        derivatives = tuple(v[:, None]*d for v, d in zip(current, ratios))
        if coefficients is not None:
            current, derivatives = coefficients(logt)
        gradient = p['temperature_gradient']+system.gradient_operator@dx
        y = ml2_auxiliary_from_gradient(gradient, *current, system.target)
        convection = np.maximum(y, 0.)**3*system.target; convection[0] = 0.
        radiation = p['radiative_flux_interface']+p['radiative_flux_log_temperature_jacobian']@dx
        heating = p['radiative_cell_energy_defect']+p['radiative_cell_energy_log_temperature_jacobian']@dx
        thermal = thermal0*np.exp(thermal_ratio@dx)
        radiation_j = p['radiative_flux_log_temperature_jacobian']
        heating_j = p['radiative_cell_energy_log_temperature_jacobian']
        thermal_j = thermal[:, None]*thermal_ratio
        if radiation_remainder is not None:
            remainder, derivative = radiation_remainder(logt)
            radiation = radiation+remainder[0]
            heating = heating+remainder[1]
            thermal = thermal+remainder[2]
            radiation_j = radiation_j+derivative[0]
            heating_j = heating_j+derivative[1]
            thermal_j = thermal_j+derivative[2]
        scale = thermal+convection[:-1]+convection[1:]
        energy = (heating+np.diff(convection))/scale
        residual = (radiation+convection)/system.target-1.
        residual[:-1] -= energy
        metadata = dict(p['convection_transport'])
        metadata.update(zip(('adiabatic_gradient', 'ml2_radiative_loss', 'ml2_flux_coefficient'), current))
        fields = dict(p, atmosphere=SimpleNamespace(**{
                **vars(p['atmosphere']), 'temperature':np.exp(logt)}),
            convection_transport=metadata, ml2_coefficient_log_temperature_responses=derivatives,
            temperature_gradient=gradient, radiative_flux_interface=radiation,
            convective_flux_interface=convection, total_flux_interface=radiation+convection,
            radiative_cell_energy_defect=heating, thermal_cell_emission=thermal,
            radiative_flux_log_temperature_jacobian=radiation_j,
            radiative_cell_energy_log_temperature_jacobian=heating_j,
            thermal_cell_emission_log_temperature_jacobian=thermal_j,
            cell_energy_balance_relative_residual=energy, log_temperature_from_state=np.eye(n))
        return NonlinearEvaluation(residual, np.eye(n) if need else None, fields)

    model_type = getattr(system, 'trial_system_class', ExplicitGradientSystem)
    model = model_type(predicted(anchor, True), predicted, None)
    model.gradient_scale = system.gradient_scale.copy()
    copy_coordinates = getattr(system, 'copy_trial_coordinates', None)
    if copy_coordinates is not None:
        copy_coordinates(model)
    # Exact same origin and tangent; nonlinear model not a changed root.
    check = model.evaluate(state, True)
    np.testing.assert_allclose(check.residual, ev.residual, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(check.jacobian, ev.jacobian, atol=1e-8, rtol=1e-10)
    return model


def compatible_temperature_trial(system, model, state, radius):
    """Minimize physical rows on the exact algebraic ML2 manifold.

    Stable explicit-gradient coordinates can be extremely large when their
    frozen superadiabatic unit was established near convection onset.  They
    carry no flux but make the enlarged least-squares problem nearly singular.
    Eliminate those algebraic variables in this cheap trial only: every trial
    temperature rebuilds its exact ML2 coordinate, and the outer solver still
    evaluates and accepts the resulting full-physics state.
    """
    n = system.n
    cache = [None, None]

    def evaluated(delta):
        if cache[0] is None or not np.array_equal(cache[0], delta):
            logt = state[:n]+delta
            physical = model.physical(logt, True)
            compatible = model.state_from_temperature(logt, physical)
            full = model.evaluate(compatible, True)
            mapping = model.compatible_state_jacobian(logt, physical)
            if mapping.shape != (2*n-1, n):
                raise ValueError('Compatible-state tangent has the wrong shape')
            jacobian = full.jacobian[:n]@mapping
            cache[:] = [delta.copy(), (compatible, full, jacobian)]
        return cache[1]

    solved = least_squares(
        lambda d:evaluated(d)[1].residual[:n],
        np.zeros(n),
        jac=lambda d:evaluated(d)[2],
        bounds=(np.full(n, -radius), np.full(n, radius)),
        # Every variable is the same dimensionless quantity, delta ln(T).
        # Jacobian-column scaling can freeze a perfectly regular physical
        # direction when an inactive convection coordinate gives one column
        # a transiently extreme norm near a stable/unstable interface.
        method='trf', x_scale=1., max_nfev=1000,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    compatible, full, _ = evaluated(solved.x)
    direction = compatible-state
    if not admissible_temperature_gradient(
            full, getattr(system, 'minimum_temperature_gradient_index', None)):
        # This cheap manifold candidate is only a proposal.  Do not offer an
        # inversion to the full-physics outer solver merely because it lowers
        # the approximate residual model.
        return solved, np.zeros_like(state), np.inf
    return solved, direction, float(.5*np.sum(full.residual**2))


def compatible_temperature_correction(system, state):
    """Build an exactly compatible nonlinear path for a stored candidate."""
    candidate = getattr(system, '_compatible_temperature_correction', None)
    system._compatible_temperature_correction = None
    if candidate is None:
        return None
    anchor, temperature_direction, original_physical_cost = candidate
    if not np.array_equal(anchor, state):
        return None

    latest_physical_cost = [np.inf]

    def trial(factor):
        logt = state[:system.n]+factor*temperature_direction
        physical = system.physical(logt, False)
        latest_physical_cost[0] = float(.5*np.sum(physical.residual**2))
        return system.state_from_temperature(logt, physical)

    return NonlinearCorrection(
        trial,
        lambda ev:latest_physical_cost[0] < original_physical_cost,
    )


def solve_bounded_material_trial(evaluated, size, lower, upper):
    """Solve the cheap bounded model without Jacobian-derived variable units.

    Every entry of ``size`` is already a nondimensional native coordinate.
    In a newly extended stable layer, one superadiabatic coordinate can sit
    exactly on its one-sided physical floor while its Jacobian column is many
    orders of magnitude smaller than the temperature columns.  SciPy's
    ``x_scale='jac'`` then starts that bounded variable an infinitesimal
    distance inside its bound and may satisfy ``ftol`` before taking a useful
    temperature step.  Unit scaling preserves the declared native trust box
    and avoids that false stationary point.
    """
    return least_squares(
        lambda d:evaluated(d).residual,
        np.zeros(size),
        jac=lambda d:evaluated(d).jacobian,
        bounds=(lower, upper),
        method='trf', x_scale=1., max_nfev=1000,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )


def propose(system, state, ev, jacobian, radius):
    started = time.monotonic()
    system._compatible_temperature_correction = None
    compatible = compatible_stationarity(system, state)
    if compatible is not None:
        return NonlinearProposal(compatible), True, dict(
            ordinary_newton=True,
            exact_auxiliary_closure_projection=True,
            temperature_step=0.,
            seconds=time.monotonic()-started,
        )
    stationarity = getattr(system, 'unrestricted_stationarity_candidate', None)
    if stationarity is not None:
        measured = stationarity(ev, jacobian)
        if measured is not None:
            return NonlinearProposal(measured), True, dict(ordinary_newton=True,
                unrestricted_physical_stationarity=True)
    linear, root = coupled_direction(jacobian, ev.residual, system.n, radius)
    # Preserve ordinary local Newton roots near a verified physical solution.
    if root and np.max(abs(ev.residual)) < .002:
        return NonlinearProposal(linear), True, dict(ordinary_newton=True)
    model = local_model(system, state, ev)
    lower, upper = trial_bounds(system, state, ev, radius)
    cache = [None, None]
    def evaluated(delta):
        if cache[0] is None or not np.array_equal(cache[0], delta):
            cache[:] = [delta.copy(), model.evaluate(state+delta, True)]
        return cache[1]
    solved = solve_bounded_material_trial(
        evaluated, state.size, lower, upper
    )
    corner_info = {}
    from .dq_eos_corner_subspace import refine_corner_subspace
    solved, subspace_info = refine_corner_subspace(
        system, state, solved, radius, evaluated, bounds=(lower, upper)
    )
    # Adjoining-piece search is retained as a separate experiment: it did
    # not improve the J1311 physical gates enough to justify default cost.
    if getattr(getattr(system,'eos_trial_owner',None),'search_eos_corners',False):
        from .dq_eos_knot_search import search_corners
        solved, corner_info = search_corners(system,state,solved,radius,evaluated)
    compatible_solved, compatible_direction, compatible_cost = compatible_temperature_trial(
        system, model, state, radius
    )
    compatible_size = system.trust_step_size(state, state+compatible_direction)
    manifold_info = dict(
        compatible_temperature_manifold=False,
        compatible_temperature_success=bool(compatible_solved.success),
        compatible_temperature_status=int(compatible_solved.status),
        compatible_temperature_message=str(compatible_solved.message),
        compatible_temperature_evaluations=int(compatible_solved.nfev),
        compatible_temperature_optimality=float(compatible_solved.optimality),
        compatible_temperature_candidate_cost=compatible_cost,
        compatible_temperature_candidate_size=float(compatible_size),
        compatible_temperature_candidate_step=float(np.max(abs(
            compatible_direction[:system.n]
        ))),
        compatible_temperature_correction_available=False,
    )
    initial_cost = float(.5*np.sum(ev.residual**2))
    initial_physical_cost = float(.5*np.sum(ev.residual[:system.n]**2))
    compatible_temperature_size = float(np.max(abs(
        compatible_direction[:system.n]
    )))
    minimum_correction_size = getattr(
        system, 'requested_step_tolerance', 2e-4
    )
    if (np.all(np.isfinite(compatible_direction))
            and compatible_cost < initial_physical_cost
            and compatible_temperature_size >= minimum_correction_size):
        system._compatible_temperature_correction = (
            state.copy(), compatible_direction[:system.n].copy(),
            initial_physical_cost,
        )
        manifold_info['compatible_temperature_correction_available'] = True
    if (np.all(np.isfinite(compatible_direction))
            and compatible_size <= radius*(1+128*np.finfo(float).eps)
            and compatible_cost < solved.cost
            and compatible_cost < initial_physical_cost
            and compatible_temperature_size >= minimum_correction_size):
        solved.x = compatible_direction
        solved.cost = compatible_cost
        solved.success = compatible_solved.success
        solved.message = compatible_solved.message
        solved.nfev += compatible_solved.nfev
        manifold_info = dict(
            compatible_temperature_manifold=True,
            compatible_temperature_success=bool(compatible_solved.success),
            compatible_temperature_status=int(compatible_solved.status),
            compatible_temperature_message=str(compatible_solved.message),
            compatible_temperature_evaluations=int(compatible_solved.nfev),
            compatible_temperature_optimality=float(compatible_solved.optimality),
            compatible_temperature_cost=compatible_cost,
            compatible_temperature_candidate_cost=compatible_cost,
            compatible_temperature_candidate_size=float(compatible_size),
            compatible_temperature_candidate_step=float(np.max(abs(
                compatible_direction[:system.n]
            ))),
            compatible_temperature_step=float(np.max(abs(compatible_solved.x))),
            compatible_temperature_correction_available=False,
        )
        system._compatible_temperature_correction = None
    if not np.all(np.isfinite(solved.x)) or solved.cost >= .5*np.sum(ev.residual**2):
        raise RecoverableEvaluationError('Nonlinear material trial model did not improve')
    def residual_model(delta):
        return model.evaluate(state+delta, False).residual
    return NonlinearProposal(solved.x, limited=bool(subspace_info.get('eos_subspace_selected')),
        residual_model=residual_model), False, dict(
        ordinary_newton=False, inner_success=bool(solved.success), inner_status=solved.message,
        inner_evaluations=solved.nfev, inner_cost=float(solved.cost),
        initial_cost=initial_cost,
        native_step=float(np.max(abs(solved.x))),
        temperature_step=float(np.max(abs(solved.x[:system.n]))),
        stable_coordinate_release_count=int(np.count_nonzero(
            lower[system.n:] < -radius
        )),
        seconds=time.monotonic()-started, **corner_info, **subspace_info,
        **manifold_info)


@contextmanager
def nonlinear_material_trials():
    original = adaptive.solve_trust_region_newton
    def builder(system, q, ev, j, radius):
        result, root, info = propose(system, q, ev, j, radius)
        print('NONLINEAR MATERIAL TRIAL '+str(info), flush=True)
        return result, root
    def solve(initial, evaluate, **settings):
        system = getattr(evaluate, '__self__', None)
        if isinstance(system, ExplicitGradientSystem):
            # This model already retains nonlinear convection coupled to the
            # global radiative response. Do not interleave the separate local
            # diffusion-shaped proposal during this controlled experiment.
            settings['iteration_correction'] = (
                lambda state, ev:compatible_temperature_correction(system, state)
            )
            settings['nonlinear_model_globalization'] = True
            system.requested_step_tolerance = settings.get('step_tolerance', 2e-4)
            system.requested_residual_tolerance = settings.get('residual_tolerance', .002)
            original_acceptance = settings.get('acceptance_test')
            def physically_admissible(old, trial):
                return (admissible_temperature_gradient(trial, getattr(
                            system, 'minimum_temperature_gradient_index', None))
                    and (original_acceptance is None
                         or original_acceptance(old, trial)))
            settings['acceptance_test'] = physically_admissible
        return original(initial, evaluate, **settings)
    # Install below require_measured_proposal, never replace its wrapper.
    # Otherwise its closed-over measurement would remain false forever.
    with patch.object(ExplicitGradientSystem, 'build_proposal', builder, create=True), \
            patch.object(adaptive, 'solve_trust_region_newton', solve):
        yield
