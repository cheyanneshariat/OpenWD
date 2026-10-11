from types import SimpleNamespace
import numpy as np
from wd_spectra._thermal_conditioning import (
    equilibrated_direction,
    frozen_energy_evaluator,
    thermal_condition,
)
from wd_spectra.nonlinear import NonlinearEvaluation
from wd_spectra.constants import STEFAN_BOLTZMANN


def linear_thermal_system(state, need_jacobian):
    target = STEFAN_BOLTZMANN
    desired = np.array([0.03, -0.02, 0.01])
    energy = target * (desired[:-1] - state[:-1])
    flux = np.full(3, target)
    flux[-1] *= 1 + state[-1] - desired[-1]
    radiation_jac = np.zeros((3, 3))
    radiation_jac[-1, -1] = target
    cell_jac = -target * np.eye(3)[:-1]
    residual = flux / target - 1
    residual[:-1] -= energy / target
    jac = np.eye(3) if need_jacobian else None
    payload = dict(
        atmosphere=SimpleNamespace(effective_temperature=1.0),
        radiative_cell_energy_defect=energy,
        convective_flux_interface=np.zeros(3),
        total_flux_interface=flux,
        cell_energy_scale=np.full(2, target),
        cell_energy_balance_relative_residual=energy / target,
        radiative_flux_log_temperature_jacobian=radiation_jac,
        convective_flux_log_temperature_jacobian=np.zeros((3, 3)),
        cell_energy_log_temperature_jacobian=cell_jac,
        log_temperature_from_state=np.eye(3),
    )
    return NonlinearEvaluation(residual, jac, payload)


def test_thermal_steps_reduce_heating_and_cool_excess_bottom_flux_without_certifying():
    records = []
    state, meta = thermal_condition(
        np.zeros(3),
        linear_thermal_system,
        local_tolerance=1e-5,
        callback=lambda record, *_: records.append(record),
    )
    assert max(abs(state - np.array([0.03, -0.02, 0.01]))) < 1e-5
    assert meta["termination"] == "local-balance-handoff"
    assert meta["iterations"] == len(records) > 0
    assert not meta["converged"]
    assert all(
        r["initialization_only"] and not r["converged"] for r in records
    )
    assert all(r["maximum_log_temperature_change"] <= 0.12 for r in records)


def test_zero_thermal_budget_does_not_evaluate_or_claim_equilibrium():
    def forbidden(*args):
        raise AssertionError("zero budget")

    initial = np.arange(3.0)
    state, meta = thermal_condition(initial, forbidden, maximum_sweeps=0)
    np.testing.assert_array_equal(state, initial)
    assert meta["iterations"] == 0 and not meta["converged"]


def test_frozen_equation_weights_and_square_direction_preserve_root():
    evaluate = frozen_energy_evaluator(linear_thermal_system)
    initial = np.zeros(3)
    ev = evaluate(initial, True)
    step = equilibrated_direction(ev.jacobian, ev.residual)
    np.testing.assert_allclose(step, [0.03, -0.02, 0.01])
    np.testing.assert_allclose(
        evaluate(initial + step, False).residual, 0.0, atol=1e-15
    )
    # Neither very large nor tiny equation units may suppress a direction.
    weights = np.array([1e-100, 1e100, 1.0])
    np.testing.assert_allclose(
        equilibrated_direction(
            ev.jacobian * weights[:, None], ev.residual * weights
        ),
        step,
    )


def test_trial_equation_weights_are_frozen_but_physical_diagnostics_are_current():
    def moving_scale(state, need_jacobian):
        ev = linear_thermal_system(state, need_jacobian)
        ev.payload["cell_energy_scale"] *= np.exp(state[0])
        ev.payload["cell_energy_balance_relative_residual"] /= np.exp(state[0])
        return ev

    evaluate = frozen_energy_evaluator(moving_scale)
    evaluate(np.zeros(3), True)
    state = np.array([0.02, -0.01, 0.0])
    trial = evaluate(state, False)
    np.testing.assert_allclose(
        trial.residual, linear_thermal_system(state, False).residual
    )
    np.testing.assert_allclose(
        trial.payload["cell_energy_balance_relative_residual"],
        moving_scale(state, False).payload[
            "cell_energy_balance_relative_residual"
        ],
    )
    refreshed = evaluate(state, True)
    np.testing.assert_allclose(
        refreshed.residual[:-1], trial.residual[:-1] / np.exp(0.02)
    )


def _scaled_thermal_system(desired, cell_slope):
    """Linear cells; ``cell_slope`` > 0 makes heating grow with temperature."""

    def system(state, need_jacobian):
        target = STEFAN_BOLTZMANN
        energy = target * cell_slope * (desired[:-1] - state[:-1])
        flux = np.full(3, target)
        flux[-1] *= 1 + state[-1] - desired[-1]
        radiation_jac = np.zeros((3, 3))
        radiation_jac[-1, -1] = target
        cell_jac = -target * cell_slope * np.eye(3)[:-1]
        residual = flux / target - 1
        residual[:-1] -= energy / target
        payload = dict(
            atmosphere=SimpleNamespace(effective_temperature=1.0),
            radiative_cell_energy_defect=energy,
            convective_flux_interface=np.zeros(3),
            total_flux_interface=flux,
            cell_energy_scale=np.full(2, target),
            cell_energy_balance_relative_residual=energy / target,
            radiative_flux_log_temperature_jacobian=radiation_jac,
            convective_flux_log_temperature_jacobian=np.zeros((3, 3)),
            cell_energy_log_temperature_jacobian=cell_jac,
            log_temperature_from_state=np.eye(3),
        )
        return NonlinearEvaluation(residual, np.eye(3) if need_jacobian else None, payload)

    return system


def test_decreasing_local_energy_permits_larger_thermal_steps():
    records = []
    desired = np.array([0.5, -0.4, 0.0])
    thermal_condition(
        np.zeros(3), _scaled_thermal_system(desired, 1.0),
        local_tolerance=1e-6, callback=lambda record, *_: records.append(record),
    )
    steps = [r["maximum_log_temperature_change"] for r in records]
    assert steps[0] <= 0.04 + 1e-12
    assert max(steps) > 0.04 and max(steps) <= 0.12 + 1e-12


def test_rising_local_energy_keeps_cautious_thermal_steps():
    records = []
    desired = np.array([0.5, -0.4, 0.0])
    thermal_condition(
        np.zeros(3), _scaled_thermal_system(desired, -1.0),
        maximum_sweeps=8, local_tolerance=1e-6,
        callback=lambda record, *_: records.append(record),
    )
    assert records
    assert all(r["maximum_log_temperature_change"] <= 0.04 + 1e-12 for r in records)


def test_failed_fast_speculation_resumes_the_cautious_trajectory():
    base = _scaled_thermal_system(np.array([0.5, -0.4, 0.0]), 1.0)

    def bumpy(state, need_jacobian):
        ev = base(state, need_jacobian)
        if state[0] > 0.12:  # an artificial region where heating rises
            ev.payload["cell_energy_balance_relative_residual"] = (
                ev.payload["cell_energy_balance_relative_residual"] + 1.0)
        return ev

    records = []
    state, meta = thermal_condition(
        np.zeros(3), bumpy, maximum_sweeps=6, local_tolerance=1e-6,
        callback=lambda record, *_: records.append(record),
    )
    flags = [r["fast_control"] for r in records]
    assert True in flags
    first_cautious_after_fast = flags.index(False, flags.index(True))
    resumed = records[first_cautious_after_fast:]
    assert resumed and not any(r["fast_control"] for r in resumed)
    assert all(r["maximum_log_temperature_change"] <= 0.04 + 1e-12 for r in resumed)
