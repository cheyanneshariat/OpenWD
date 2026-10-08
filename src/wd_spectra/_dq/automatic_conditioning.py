"""Single-invocation cold DQ experiment; no external atmosphere dependency.

Implicit thermal conditioning stops when the ordinary depth-flux tolerance
is reached.  It is repeated after a same-run lower-domain extension because
the newly physical cell changes the formal radiation field even when the
shallower atmosphere was converged. The ordinary steady solver still has to
pass *all* gates.
Energy/coordinate units are refreshed between steady phases if local units
drift by a decade, or an unconverged solve stalls. The decade is a numerical
conditioning policy, not a Teff/object/physics switch. No weights change
inside an accepted/rejected Newton trial or its Jacobian model.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
import time
from unittest.mock import patch
import numpy as np
from scipy.optimize import least_squares
from . import nonlinear_material_model as nm
import wd_spectra.adaptive_structure as adaptive
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton
from .thermal_balance import capacity, thermal_rows
from .phase_energy_scale import refresh_energy_units, scale_drift


THERMAL_RELATIVE_NORM_LIMIT = ContextVar('dq_thermal_relative_norm_limit', default=.1)


@contextmanager
def thermal_relative_norm_limit(value):
    """Use one forcing target for inner work and outer transient acceptance.

    This controls the accuracy of a temporary pseudo-time equation only.
    The steady solver and its physical convergence tolerances are untouched.
    """
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError('Thermal relative norm limit must lie strictly between zero and one')
    token = THERMAL_RELATIVE_NORM_LIMIT.set(value)
    try:
        yield
    finally:
        THERMAL_RELATIVE_NORM_LIMIT.reset(token)


def thermal_condition(system, state, tolerance, emit, maximum_steps=30, *, handoff_at_budget=False):
    n = system.n
    dt = None
    relative_limit = THERMAL_RELATIVE_NORM_LIMIT.get()
    for iteration in range(maximum_steps+1):
        ev = system.evaluate(state, True)
        p = ev.payload
        flux = float(max(abs(p['total_flux_interface']/system.target-1)))
        if flux < tolerance:
            emit(dict(phase='thermal-handoff', iterations=iteration, flux=flux,
                energy=float(max(abs(p['cell_energy_balance_relative_residual']))),
                atmosphere_certified=False))
            return state
        if iteration == maximum_steps:
            if handoff_at_budget:
                emit(dict(phase='thermal-budget-handoff',iterations=iteration,flux=flux,
                    energy=float(max(abs(p['cell_energy_balance_relative_residual']))),
                    thermal_target_reached=False,atmosphere_certified=False))
                return state.copy()
            raise RuntimeError('Thermal conditioning exceeded its step budget; no fallback')
        anchor = state.copy()
        cap = capacity(p)
        heat = p['radiative_cell_energy_defect']+np.diff(p['dq_augmented_auxiliary_flux'])
        ceiling = .04/max(float(max(abs(heat/cap))), np.finfo(float).tiny)
        dt = min(ceiling, ceiling if dt is None else dt)
        def evaluate(q, need):
            actual = system.evaluate(q, need)
            r, j = thermal_rows(system, q, actual, anchor, cap, dt)
            return NonlinearEvaluation(r, j, actual)
        def propose(q, e, j, radius):
            model = nm.local_model(system, q, e.payload)
            cache = [None, None]
            def predicted(delta):
                if cache[0] is None or not np.array_equal(cache[0], delta):
                    candidate = model.evaluate(q+delta, True)
                    cache[:] = [delta.copy(), thermal_rows(system, q+delta, candidate, anchor, cap, dt)]
                return cache[1]
            bounds = np.r_[np.full(n, radius), np.full(n-1, np.inf)]
            sol = least_squares(lambda d:predicted(d)[0], np.zeros_like(q),
                jac=lambda d:predicted(d)[1], bounds=(-bounds,bounds), x_scale='jac',
                max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            return NonlinearProposal(sol.x, limited=True, residual_model=lambda d:predicted(d)[0])
        first = evaluate(state, True)
        first_cost = .5*first.residual@first.residual
        def observed(record,q,evaluation):
            physical = evaluation.payload.payload
            emit(dict(phase='thermal-newton',thermal_iteration=iteration+1,
                inner_iteration=record.iteration,dt=float(dt),
                transient_norm=float(np.linalg.norm(evaluation.residual)),
                relative_transient_norm=float(np.linalg.norm(evaluation.residual)/np.sqrt(2*first_cost)),
                auxiliary_mismatch=float(physical['dq_augmented_flux_compatibility']),
                maximum_step=float(record.maximum_step),
                worst_residual_index=int(record.worst_residual_index),
                proposal_limited=bool(record.proposal_limited),
                atmosphere_certified=False))
        result = solve_trust_region_newton(state, evaluate, maximum_iterations=2,
            residual_tolerance=1e-7, step_tolerance=1e-7, allow_initial_convergence=False,
            initial_trust_radius=.04, maximum_trust_radius=.04,
            step_measure=lambda old,new:float(max(abs(new[:n]-old[:n]))),
            step_builder=propose, merit_function='least-squares', trust_update='model-agreement',
            broyden_updates=False, jacobian_refresh_interval=1, finite_difference_fallback_step=None,
            nonlinear_model_globalization=True, convergence_test=lambda *a:False,
            callback=observed)
        final = result.evaluation
        cost = .5*final.residual@final.residual
        p = final.payload.payload
        accepted = (cost < relative_limit**2*first_cost
                    and p['dq_augmented_flux_compatibility'] < tolerance)
        emit(dict(phase='thermal-step', iteration=iteration+1, dt=float(dt), accepted=bool(accepted),
            inner_stop_reason=getattr(result,'reason','fixed-update-budget'),
            transient_cost=float(cost), initial_cost=float(first_cost),
            relative_transient_norm=float(np.sqrt(cost/first_cost)),
            acceptance_relative_norm_limit=relative_limit,
            flux=float(max(abs(p['total_flux_interface']/system.target-1))),
            energy=float(max(abs(p['cell_energy_balance_relative_residual']))),
            worst_residual_index=int(np.argmax(abs(final.residual))),
            atmosphere_certified=False), p if accepted else None,
            rejected_payload=None if accepted else p)
        if accepted:
            state = result.state.copy()
            dt *= 2.
        else:
            dt *= .25
    raise AssertionError('Unreachable thermal loop exit')


class RefreshPhase(Exception):
    def __init__(self, state):
        self.state = state.copy()


@contextmanager
def automatic_material_trials(output):
    original_context = nm.nonlinear_material_trials
    record_path = output/'automatic-phases.json'
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    elapsed_offset = records[-1]['seconds'] if records else 0.
    domain_depths = None
    started = time.monotonic()
    def emit(row, payload=None, *, rejected_payload=None):
        row = dict(row, seconds=elapsed_offset+time.monotonic()-started,
                   domain_depths=domain_depths)
        records.append(row)
        print('AUTOMATIC DQ PHASE '+json.dumps(row), flush=True)
        record_path.write_text(json.dumps(records, indent=2)+'\n')
        if payload is not None:
            a = payload['atmosphere']
            np.savez_compressed(output/'automatic-latest.npz', temperature=a.temperature,
                gas_pressure=a.gas_pressure, column_mass=a.column_mass,
                rosseland_optical_depth=a.rosseland_optical_depth)
        if rejected_payload is not None:
            # Retain the failed trial for diagnosis, without displacing the
            # last accepted state or implying it passed the transient test.
            a = rejected_payload['atmosphere']
            np.savez_compressed(output/'automatic-last-rejected.npz',
                temperature=a.temperature,gas_pressure=a.gas_pressure,
                column_mass=a.column_mass,rosseland_optical_depth=a.rosseland_optical_depth,
                accepted=np.array(False),atmosphere_certified=np.array(False))
    with original_context():
        original_solver = adaptive.solve_trust_region_newton
        def solve(initial, evaluate, **settings):
            nonlocal domain_depths
            system = getattr(evaluate, '__self__', None)
            if system is None or not hasattr(system, 'excess_scale'):
                return original_solver(initial, evaluate, **settings)
            domain_depths = system.n
            tolerance = settings.get('residual_tolerance', .002)
            state = initial.copy()
            if getattr(system, 'same_run_domain_extension', False):
                p = system.evaluate(state, False).payload
                emit(dict(phase='thermal-start-same-run-domain-extension',
                    flux=float(max(abs(p['total_flux_interface']/system.target-1))),
                    energy=float(max(abs(p['cell_energy_balance_relative_residual']))),
                    atmosphere_certified=False))
            state = thermal_condition(system, state, tolerance, emit)
            remaining = settings['maximum_iterations']
            total = 0
            callback = settings.get('callback')
            result = None
            # Explicit resource guard: never cycle indefinitely or certify
            # a state just because a phase budget was exhausted.
            for phase in range(8):
                actual = system.physical(state[:system.n], True)
                del system.excess_scale
                state = system.state_from_temperature(state[:system.n], actual)
                anchor = system.evaluate(state, True)
                refresh_energy_units(system, anchor.payload)
                used = 0
                consecutive_small_unconverged_steps = 0
                def observed(record, q, ev):
                    nonlocal used, consecutive_small_unconverged_steps
                    used = max(used, record.iteration)
                    if callback is not None:
                        callback(replace(record, iteration=total+record.iteration), q, ev)
                    drift = scale_drift(ev.payload)
                    coordinate_drift = getattr(system,'coordinate_scale_drift',lambda p:0.)(ev.payload)
                    flux = float(max(abs(
                        ev.payload['total_flux_interface']/system.target-1.
                    )))
                    energy = float(max(abs(
                        ev.payload['cell_energy_balance_relative_residual']
                    )))
                    auxiliary = float(ev.payload.get(
                        'dq_augmented_flux_compatibility', 0.
                    ))
                    physically_unconverged = max(flux, energy, auxiliary) >= tolerance
                    step_tolerance = settings.get('step_tolerance', 2e-4)
                    if physically_unconverged and record.maximum_step < step_tolerance:
                        consecutive_small_unconverged_steps += 1
                    else:
                        consecutive_small_unconverged_steps = 0
                    if consecutive_small_unconverged_steps >= 2:
                        emit(dict(phase='refresh-request', steady_phase=phase+1,
                            reason='consecutive-small-unconverged-steps',
                            small_step_count=consecutive_small_unconverged_steps,
                            maximum_step=float(record.maximum_step), flux=flux,
                            energy=energy, auxiliary_mismatch=auxiliary), ev.payload)
                        raise RefreshPhase(q)
                    if ((drift > np.log(10.) and
                        max(abs(ev.payload['cell_energy_balance_relative_residual'])) > tolerance)
                        or coordinate_drift > np.log(10.)):
                        emit(dict(phase='refresh-request', steady_phase=phase+1,
                            reason='unit-drift',
                            unit_drift_factor=float(np.exp(drift)),
                            coordinate_unit_drift_factor=float(np.exp(coordinate_drift))), ev.payload)
                        raise RefreshPhase(q)
                emit(dict(phase='steady-start', steady_phase=phase+1, remaining_iterations=remaining))
                try:
                    result = original_solver(state, system.evaluate, **dict(settings,
                        maximum_iterations=remaining, callback=observed,
                        allow_initial_convergence=False))
                    state = result.state.copy()
                except RefreshPhase as refresh:
                    state = refresh.state
                total += used
                remaining -= max(used, 1)
                if result is not None and result.converged:
                    emit(dict(phase='steady-converged', steady_phases=phase+1,
                        total_steady_iterations=total))
                    return result
                if remaining < 1:
                    break
                result = None
            raise RuntimeError('Automatic conditioning/steady phase budget exhausted; no fallback')
        with patch.object(adaptive, 'solve_trust_region_newton', solve):
            yield
