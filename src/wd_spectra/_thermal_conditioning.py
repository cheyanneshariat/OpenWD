"""Local thermal conditioning followed by, never substituted for, equilibrium.

The artificial time scale damps poorly initialized thermal modes. Every trial
uses the original material, transfer and ML2 equations. It is not a physical
evolution calculation, and this module makes no convergence claim.
"""

import logging
import numpy as np
from .constants import STEFAN_BOLTZMANN
from .nonlinear import NonlinearEvaluation, RecoverableEvaluationError

_LOGGER = logging.getLogger(__name__)


def equilibrated_direction(matrix, residual):
    """Unpenalized square solve with row and column equilibration."""
    row = np.max(abs(matrix), axis=1)
    if np.any(row == 0):
        raise np.linalg.LinAlgError("unconstrained energy equation")
    scaled = matrix / row[:, None]
    column = np.max(abs(scaled), axis=0)
    if np.any(column == 0):
        raise np.linalg.LinAlgError("unconstrained temperature variable")
    return np.linalg.solve(scaled / column[None, :], -residual / row) / column


def frozen_energy_evaluator(evaluate):
    """Hold positive equation weights fixed within each Newton linearization.

    Removing derivatives of these numerical weights improves globalization;
    the actual, currently normalized energy is still independently certified.
    The full material/radiation/convection derivative is retained.
    """
    scale = None

    def weighted(state, need_jacobian):
        nonlocal scale
        ev = evaluate(state, need_jacobian)
        p = ev.payload
        target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
        if scale is None or need_jacobian:
            scale = np.asarray(p["cell_energy_scale"]).copy()
        defect = p["radiative_cell_energy_defect"] + np.diff(
            p["convective_flux_interface"]
        )
        residual = p["total_flux_interface"] / target - 1
        residual[:-1] -= defect / scale
        jacobian = None
        if need_jacobian:
            jacobian = (
                p["radiative_flux_log_temperature_jacobian"]
                + p["convective_flux_log_temperature_jacobian"]
            ) / target
            jacobian[:-1] -= (
                p["cell_energy_log_temperature_jacobian"] / scale[:, None]
            )
            jacobian = jacobian @ p["log_temperature_from_state"]
        return NonlinearEvaluation(residual, jacobian, p)

    return weighted


def thermal_condition(
    initial,
    evaluate,
    *,
    maximum_sweeps=40,
    local_tolerance=3e-3,
    maximum_temperature_step=0.04,
    fast_temperature_step=0.12,
    fast_tolerance_fraction=0.25,
    callback=None,
):
    """Take bounded linearly implicit thermal steps, then return to static solve.

    Cell heating advances ln(T); excess bottom flux cools the bottom thermal
    reservoir. Both artificial inertias vanish from the stationary equations.
    Temporal-defect control chooses the step, not Teff or a convection mask.

    Cautious control bounds each ln(T) change by ``maximum_temperature_step``
    and doubles the pseudo-time step only after a small temporal defect.  When
    the first (cautious) sweep lowers the largest local energy defect, later
    sweeps are tried speculatively with fast control: only the stationary
    state matters, so every accepted step doubles the pseudo-time step, the
    bound is ``fast_temperature_step`` and the handoff requires the defect to
    fall below ``fast_tolerance_fraction * local_tolerance``.  The first fast
    sweep that fails to lower the defect discards all fast sweeps and resumes
    from the state and step after the first sweep with cautious control, so
    any non-monotone case returns exactly the cautious trajectory.
    """
    state = np.asarray(initial).copy()
    step = 1.0
    history = []
    reason = "sweep-budget"
    fast = False
    restart = None
    sweeps = 0
    while sweeps < maximum_sweeps:
        sweeps += 1
        _LOGGER.info(
            "Thermal conditioning %d/%d: rebuilding actual material/transfer tangent",
            sweeps,
            maximum_sweeps,
        )
        base = evaluate(state, True)
        p = base.payload
        base_energy = float(np.max(abs(p["cell_energy_balance_relative_residual"])))
        if base_energy < local_tolerance * (fast_tolerance_fraction if fast else 1.0):
            reason = "local-balance-handoff"
            break
        target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
        scale = p["cell_energy_scale"].copy()
        mapping = p["log_temperature_from_state"]

        def thermal_rows(payload):
            energy = (
                payload["radiative_cell_energy_defect"]
                + np.diff(payload["convective_flux_interface"])
            ) / scale
            return np.r_[
                energy, payload["total_flux_interface"][-1] / target - 1
            ]

        residual = thermal_rows(p)
        jacobian = np.vstack(
            (
                p["cell_energy_log_temperature_jacobian"] / scale[:, None],
                (
                    p["radiative_flux_log_temperature_jacobian"][-1]
                    + p["convective_flux_log_temperature_jacobian"][-1]
                )
                / target,
            )
        )
        bound = fast_temperature_step if fast else maximum_temperature_step
        accepted = False
        rolled_back = False
        # The representability check is the real lower bound; this finite
        # ceiling prevents pathological evaluation loops on invalid inputs.
        for attempt in range(64):
            matrix = jacobian.copy()
            nodes = np.arange(len(state) - 1)
            matrix[nodes, nodes] -= 1 / step
            matrix[-1, -1] += 1 / step
            try:
                delta = equilibrated_direction(matrix, residual)
            except np.linalg.LinAlgError:
                step *= 0.5
                continue
            if not np.all(np.isfinite(delta)) or np.max(abs(delta)) > bound:
                step *= 0.5
                continue
            candidate = state + np.linalg.solve(mapping, delta)
            if np.array_equal(mapping @ candidate, mapping @ state):
                reason = "temperature-resolution-handoff"
                break
            try:
                trial = evaluate(candidate, False)
            except RecoverableEvaluationError:
                step *= 0.5
                continue
            temporal = thermal_rows(trial.payload)
            temporal[:-1] -= delta[:-1] / step
            temporal[-1] += delta[-1] / step
            error = float(
                np.max(abs(temporal)) / max(np.max(abs(residual)), 1e-12)
            )
            if not np.isfinite(error) or error > 0.25:
                step *= 0.5
                continue
            trial_energy = float(
                np.max(abs(trial.payload["cell_energy_balance_relative_residual"]))
            )
            if fast and trial_energy >= base_energy:
                # Speculation failed: resume the cautious trajectory.
                state, step = restart
                fast = False
                sweeps = 1
                rolled_back = True
                _LOGGER.info(
                    "Thermal conditioning: fast step raised local energy; "
                    "resuming cautious control after the first sweep"
                )
                break
            state = candidate
            record = dict(
                iteration=len(history) + 1,
                pseudo_time_step=step,
                temporal_defect=error,
                maximum_log_temperature_change=float(np.max(abs(delta))),
                maximum_local_energy=trial_energy,
                maximum_flux_error=float(
                    np.max(
                        abs(trial.payload["total_flux_interface"] / target - 1)
                    )
                ),
                fast_control=fast,
                initialization_only=True,
                converged=False,
            )
            history.append(record)
            _LOGGER.info(
                "Thermal conditioning: local energy %.6g, flux %.6g, dlnT %.6g, dt %.6g",
                record["maximum_local_energy"],
                record["maximum_flux_error"],
                record["maximum_log_temperature_change"],
                step,
            )
            if callback is not None:
                callback(record, state, trial)
            cautious_step = step * 2 if error < 0.0625 else step
            if restart is None:
                restart = (state.copy(), cautious_step)
                fast = trial_energy < base_energy
            step = step * 2 if fast else cautious_step
            accepted = True
            break
        if rolled_back:
            continue
        if not accepted:
            if fast and reason != "temperature-resolution-handoff":
                # No admissible fast step: also resume the cautious path.
                state, step = restart
                fast = False
                sweeps = 1
                continue
            if reason == "sweep-budget":
                reason = "no-admissible-thermal-step"
            break
    return state, dict(
        initialization_only=True,
        converged=False,
        termination=reason,
        iterations=len(history),
        history=history,
    )
