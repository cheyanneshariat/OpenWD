"""Contracts for the optional, non-certifying LTE preconditioner handoff."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import adaptive_structure as adaptive
from wd_spectra.atmosphere import Atmosphere
from wd_spectra.nonlinear import (
    NonlinearCorrection, NonlinearEvaluation, RecoverableEvaluationError,
    solve_trust_region_newton,
)


def evaluation(state, jacobian, energy=.1):
    return NonlinearEvaluation(1e-5*(state-1.),
        np.eye(state.size)*1e-5 if jacobian else None,
        {"total_flux_interface": np.ones(state.size),
         "cell_energy_balance_relative_residual": np.full(state.size, energy)})


def run(evaluate=evaluation, **overrides):
    initial = np.zeros(2)
    callbacks = []
    controller = adaptive._PreconditionerFluxHandoff(initial, 1., .003,
        lambda record, state, ev: callbacks.append((record, state.copy(), ev)))
    options = dict(maximum_iterations=4, residual_tolerance=.003,
        step_tolerance=.0003, allow_initial_convergence=False,
        linear_regularization=0., jacobian_refresh_interval=1,
        finite_difference_fallback_step=None, convergence_test=lambda *args: False,
        callback=controller.callback, accepted_state_handoff=controller.handoff)
    options.update(overrides)
    result = solve_trust_region_newton(initial, evaluate, **options)
    return result, controller, callbacks


def test_initial_eligible_residual_needs_two_genuinely_accepted_updates():
    result, controller, callbacks = run()
    assert result.diagnostics.terminal_reason == "accepted-state-phase-handoff"
    assert not result.converged
    assert result.diagnostics.accepted_iterations == len(callbacks) == 2
    assert all(record.line_search_factor > 0 for record, _, _ in callbacks)
    np.testing.assert_array_equal(result.state, callbacks[-1][1])
    assert controller.metadata()["observed_accepted_newton_updates"] == 2
    assert len(controller.metadata()["observations"]) == 2
    for item, (record, _, _) in zip(controller.observations, callbacks):
        assert item["iteration"] == record.iteration
        assert item["line_search_factor"] == record.line_search_factor
        assert item["measured_log_temperature_step"] == record.maximum_step
    # Reusing an already consumed accepted-state token cannot advance it.
    assert not controller.handoff(result.state, result.evaluation)
    assert len(controller.metadata()["observations"]) == 2


def test_pending_token_cannot_be_used_with_a_different_returned_anchor():
    result, _, _ = run(maximum_iterations=1)
    controller = adaptive._PreconditionerFluxHandoff(np.zeros(2), 1., .003, None)
    controller.callback(result.history[0], result.state, result.evaluation)
    with pytest.raises(ValueError, match="anchor differs"):
        controller.handoff(result.state+1e-6, result.evaluation)
    assert not controller.handoff(result.state, result.evaluation)
    assert not controller.observations


def test_real_stationary_callback_remains_root_completion_not_handoff():
    def root(state, jacobian):
        return NonlinearEvaluation(state, np.eye(state.size) if jacobian else None,
                                  evaluation(state, jacobian).payload)
    result, controller, callbacks = run(root, convergence_test=lambda *args: True)
    assert result.converged
    assert result.diagnostics.terminal_reason == "stationary-residual-converged"
    assert len(callbacks) == 1 and callbacks[0][0].line_search_factor == 0
    assert controller.observed_updates == 0 and not controller.triggered


def test_material_update_does_not_count_or_leave_newton_eligibility_streak():
    def material(state, ev):
        return NonlinearCorrection(lambda factor: state+factor*.2)
    result, controller, callbacks = run(maximum_iterations=3,
                                        iteration_correction=material)
    assert [record.step_kind for record, _, _ in callbacks] == [
        "material-correction", "newton", "material-correction"]
    assert not result.converged and not controller.triggered
    assert result.diagnostics.accepted_iterations == 3
    assert controller.observed_updates == 1
    assert controller.ignored_callbacks == 2 and controller.streak == 0


def test_recoverable_trials_do_not_count_but_backtracked_accepted_steps_do():
    anchor = np.zeros(2)
    def domain(state, jacobian):
        nonlocal anchor
        if jacobian:
            anchor = state.copy()
        elif np.max(abs(state-anchor)) > .025:
            raise RecoverableEvaluationError("outside local trial domain")
        return evaluation(state, jacobian)
    result, controller, callbacks = run(domain)
    assert result.diagnostics.terminal_reason == "accepted-state-phase-handoff"
    assert not result.converged and controller.observed_updates == 2
    assert result.diagnostics.infeasible_trial_rejections > 0
    previous = np.zeros(2)
    for record, current, _ in callbacks:
        assert 0 < record.line_search_factor < 1
        assert 0 < np.max(abs(current-previous)) <= .025
        previous = current
    np.testing.assert_array_equal(result.state, previous)


def test_rejected_direction_cannot_supply_accepted_state_progress():
    result, controller, callbacks = run(maximum_iterations=1,
        acceptance_test=lambda *args: False)
    assert not result.converged and not callbacks
    assert result.diagnostics.rejected_directions == 1
    assert controller.observed_updates == 0 and not controller.triggered
    np.testing.assert_array_equal(result.state, np.zeros(2))


def test_nonfinite_physical_payload_cannot_trigger_handoff():
    result, controller, _ = run(lambda x, j: evaluation(x, j, energy=np.inf),
                                maximum_iterations=2)
    assert not result.converged and not controller.triggered
    assert controller.observed_updates == 2
    assert all(not item["eligible"] and item["energy"] is None
               for item in controller.observations)


def test_nonfinite_callback_state_cannot_leave_a_progress_token():
    result, _, _ = run(maximum_iterations=1)
    controller = adaptive._PreconditionerFluxHandoff(np.zeros(2), 1., .003, None)
    invalid = np.array([np.nan, 0.])
    controller.callback(result.history[0], invalid, result.evaluation)
    assert not controller.handoff(invalid, result.evaluation)
    assert controller.observed_updates == 0


def adapter_arguments():
    mass = np.geomspace(.01, 100., 3)
    seed = Atmosphere(effective_temperature=10000., logg=8.,
        rosseland_optical_depth=mass, column_mass=mass, gas_pressure=1e8*mass,
        temperature=np.full(3, 5000.), mass_density=np.ones(3),
        neutral_h_density=np.zeros(3), proton_density=np.zeros(3),
        electron_density=np.ones(3), metadata={})
    wave = np.array([1000., 2000., 4000., 8000.])
    options = dict(with_temperature=lambda t: replace(seed, temperature=t),
        true_absorption=lambda a: np.ones((4, 3)),
        scattering_opacity=lambda a: np.zeros((4, 3)),
        rosseland_opacity=lambda a: np.ones(3),
        thermodynamics=lambda a: SimpleNamespace(specific_heat_constant_pressure=np.full(3, 1e8),
            density_temperature_derivative=np.ones(3), adiabatic_temperature_gradient=np.full(3, .4)),
        mixing_length_alpha=1.25, max_iterations=1, temperature_tolerance=.0003,
        flux_tolerance=.003, n_angle=2, initial_temperature_was_supplied=False,
        project_initial_convective_gradient=False, use_initial_bolometric_rescaling=False,
        use_precision_polish=False, maximum_formal_flux_rosseland_depth=None,
        maximum_formal_flux_continuations=0)
    return seed, wave, options


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes"])
def test_boolean_validation_precedes_material_or_solver_work(monkeypatch, invalid):
    seed, wave, options = adapter_arguments()
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected material or solver work before validation")
    options["with_temperature"] = forbidden
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", forbidden)
    with pytest.raises(ValueError, match="flux-handoff flag must be boolean"):
        adaptive.solve_adaptive_lte_structure(seed, wave,
            **dict(options, use_preconditioner_flux_handoff=invalid))


@pytest.mark.parametrize("changes", [
    {"use_convective_gradient_preconditioner": False, "enforce_local_energy_balance": True},
    {"initial_temperature_was_supplied": True,
     "resume_supplied_structure_in_formal_flux_phase": True, "enforce_local_energy_balance": True},
    {"enforce_local_energy_balance": False},
])
def test_requested_policy_is_unused_outside_supported_preconditioner(monkeypatch, changes):
    seed, wave, options = adapter_arguments()
    class RoutingChecked(Exception):
        pass
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported phase created a handoff controller")
    def checked(initial, evaluate, **controls):
        assert controls.get("accepted_state_handoff") is None
        raise RoutingChecked()
    monkeypatch.setattr(adaptive, "_PreconditionerFluxHandoff", forbidden)
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", checked)
    with pytest.raises(RoutingChecked):
        adaptive.solve_adaptive_lte_structure(seed, wave,
            **dict(options, **changes, use_preconditioner_flux_handoff=True))


def test_default_metadata_routing_replaces_only_inherited_current_solve_policy(monkeypatch):
    seed, wave, options = adapter_arguments()
    inherited = {"convective_preconditioner_flux_handoff":
        {"requested": True, "used": True, "triggered": True},
        "retained_seed_provenance": "prior-source"}
    original = adaptive.solve_trust_region_newton
    def reject(initial, evaluate, **controls):
        assert controls.get("accepted_state_handoff") is None
        return original(initial, evaluate, **dict(controls,
            acceptance_test=lambda *args: False, finite_difference_fallback_step=None,
            maximum_iterations=1))
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", reject)
    common = dict(options, mixing_length_alpha=None, metadata=inherited)
    omitted = adaptive.solve_adaptive_lte_structure(seed, wave, **common)
    disabled = adaptive.solve_adaptive_lte_structure(seed, wave,
        **dict(common, use_preconditioner_flux_handoff=False))
    # The default records the current solve's policy, never the inherited one;
    # an explicitly disabled solve removes the inherited record.
    record = omitted.metadata["convective_preconditioner_flux_handoff"]
    assert record["requested"] is True
    assert record["used"] is False and record["triggered"] is False
    assert record["inactive_reason"] == "no-gradient-preconditioner"
    assert record["certifies_equilibrium"] is False
    assert "convective_preconditioner_flux_handoff" not in disabled.metadata
    assert omitted.metadata["retained_seed_provenance"] == "prior-source"
    assert inherited["convective_preconditioner_flux_handoff"]["used"] is True
    assert not omitted.metadata["radiative_equilibrium_converged"]
    assert not disabled.metadata["radiative_equilibrium_converged"]
    for name in ("temperature", "column_mass", "rosseland_optical_depth", "gas_pressure",
                 "mass_density", "electron_density"):
        np.testing.assert_array_equal(getattr(omitted, name), getattr(disabled, name))
    def equal(left, right):
        if isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    without_record = {key: value for key, value in omitted.metadata.items()
                      if key != "convective_preconditioner_flux_handoff"}
    equal(without_record, disabled.metadata)


@pytest.mark.parametrize("enabled", [False, True])
def test_initial_phase_hook_never_leaks_into_formal_completion(monkeypatch, enabled):
    mass = np.geomspace(.01, 100., 3)
    seed = Atmosphere(effective_temperature=10000., logg=8.,
        rosseland_optical_depth=mass, column_mass=mass, gas_pressure=1e8*mass,
        temperature=np.full(3, 5000.), mass_density=np.ones(3),
        neutral_h_density=np.zeros(3), proton_density=np.zeros(3),
        electron_density=np.ones(3), metadata={})
    calls = []
    original = adaptive.solve_trust_region_newton
    class FormalChecked(Exception):
        pass
    def checked(initial, evaluate, **options):
        calls.append(options)
        if len(calls) == 1:
            hook = options.get("accepted_state_handoff")
            assert (hook is not None) == enabled
            if enabled:
                assert isinstance(hook.__self__, adaptive._PreconditionerFluxHandoff)
            return original(initial, evaluate, **dict(options, maximum_iterations=1))
        assert options.get("accepted_state_handoff") is None
        assert options["rejected_step_handoff"] is calls[0]["rejected_step_handoff"]
        if enabled:
            assert options["callback"] is calls[0]["callback"].__self__.original_callback
        else:
            assert options["callback"] is calls[0]["callback"]
        raise FormalChecked()
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", checked)
    with pytest.raises(FormalChecked):
        adaptive.solve_adaptive_lte_structure(seed, np.array([1000., 2000., 4000., 8000.]),
            with_temperature=lambda t: replace(seed, temperature=t),
            true_absorption=lambda a: np.ones((4, 3)),
            scattering_opacity=lambda a: np.zeros((4, 3)),
            rosseland_opacity=lambda a: np.ones(3),
            thermodynamics=lambda a: SimpleNamespace(specific_heat_constant_pressure=np.full(3, 1e8),
                density_temperature_derivative=np.ones(3), adiabatic_temperature_gradient=np.full(3, .4)),
            mixing_length_alpha=1.25, max_iterations=1, temperature_tolerance=.0003,
            flux_tolerance=.003, n_angle=2, initial_temperature_was_supplied=False,
            project_initial_convective_gradient=False, use_initial_bolometric_rescaling=False,
            maximum_formal_flux_rosseland_depth=None, maximum_formal_flux_continuations=0,
            enforce_local_energy_balance=True, use_preconditioner_flux_handoff=enabled)
    assert len(calls) == 2


def test_actual_dq_dispatch_inherits_the_enabled_shared_default(monkeypatch):
    from wd_spectra._dq import base
    calls = []
    marker = object()
    def captured(seed, wavelength, **options):
        calls.append(options)
        return marker
    monkeypatch.setattr(base, "solve_adaptive_lte_structure", captured)
    result = base.DQMaterial.solve_adaptive_structure(object(), object(), object(), max_iterations=1)
    assert result is marker
    assert "use_preconditioner_flux_handoff" not in calls[0]
    assert inspect.signature(adaptive.solve_adaptive_lte_structure).parameters[
        "use_preconditioner_flux_handoff"].default is True


def test_ineligible_accepted_newton_update_resets_the_eligibility_streak():
    def temporarily_unbalanced(state, jacobian):
        ev = evaluation(state, jacobian)
        flux = 1.01 if .05 < state[0] < .15 else 1.
        return NonlinearEvaluation(ev.residual, ev.jacobian,
            {**ev.payload, "total_flux_interface": np.full(state.size, flux)})

    result, controller, callbacks = run(temporarily_unbalanced,
                                        maximum_iterations=4)
    assert result.diagnostics.terminal_reason == "accepted-state-phase-handoff"
    assert not result.converged
    assert result.diagnostics.accepted_iterations == len(callbacks) == 4
    assert controller.observed_updates == 4 and controller.eligible_updates == 3
    assert controller.ignored_callbacks == 0
    assert all(record.step_kind == "newton" and record.line_search_factor > 0
               and record.maximum_step > 0 for record, _, _ in callbacks)
    np.testing.assert_allclose([state for _, state, _ in callbacks],
        np.repeat(np.array([.04, .10, .19, .31])[:, None], 2, axis=1),
        rtol=0., atol=1e-14)
    assert [item["eligible"] for item in controller.observations] == [
        True, False, True, True]
    assert [item["streak"] for item in controller.observations] == [1, 0, 1, 2]
    assert controller.triggered
    np.testing.assert_array_equal(result.state, callbacks[-1][1])
