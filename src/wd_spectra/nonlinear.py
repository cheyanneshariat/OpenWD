"""Reusable safeguarded nonlinear solves for atmosphere structure.

The atmosphere modules supply physical residuals and approximate Jacobians;
this module owns convergence, scaling, trust-region step control, and Broyden
updates.  Keeping those policies here prevents individual spectral types from
accumulating their own damping constants and restart heuristics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Payload = TypeVar("Payload")
_LOGGER = logging.getLogger(__name__)


class RecoverableEvaluationError(ValueError):
    """Signal that a proposed nonlinear trial state is physically invalid.

    The trust-region driver treats this exception as a rejected line-search
    trial and backtracks.  Evaluations of the initial or accepted state, and
    all other exception types, still fail immediately so programming and
    physics errors are not silently hidden.
    """


@dataclass(frozen=True)
class NonlinearEvaluation(Generic[Payload]):
    """Residual, optional Jacobian, and caller-owned physical state."""

    residual: FloatArray
    jacobian: FloatArray | None
    payload: Payload


@dataclass(frozen=True)
class NonlinearProposal:
    """A custom direction with explicit nonlinear/physical limiting telemetry.

    ``limited`` reports clipping after the trust subproblem. Such a proposal
    cannot justify radius growth or an unrestricted stationarity claim.
    """

    direction: FloatArray
    limited: bool = False
    # Optional structured local model, evaluated at a displacement from the
    # current anchor. It changes trust-model prediction, never the physical
    # residual or acceptance test. None retains the linear Jacobian model.
    residual_model: Callable[[FloatArray], FloatArray] | None = None


@dataclass(frozen=True)
class NonlinearCorrection(Generic[Payload]):
    """Independent, backtracked material update between Newton iterations.

    ``trial_state(factor)`` must construct a complete candidate, including
    dependent variables, for 0 < factor <= 1. It must not mutate the anchor.
    The driver still evaluates every candidate and requires merit reduction.
    ``acceptance_test`` can impose additional, never weaker, physical checks.
    """

    trial_state: Callable[[float], FloatArray]
    acceptance_test: Callable[[NonlinearEvaluation[Payload]], bool] | None = None


@dataclass(frozen=True)
class NonlinearIteration:
    """One accepted step or measured stationary proposal (line-search factor zero)."""

    iteration: int
    residual_rms: float
    residual_maximum: float
    maximum_step: float
    trust_radius: float
    line_search_factor: float
    jacobian_recomputed: bool
    residual_merit: float
    worst_residual_index: int
    rejected_trial_evaluations: int
    model_agreement: float | None
    unrestricted_maximum_step: float | None = None
    proposal_limited: bool = False
    step_kind: Literal["newton", "material-correction"] = "newton"


NonlinearTerminalReason = Literal[
    "initial-state-converged",
    "accepted-state-phase-handoff",
    "rejected-step-phase-handoff",
    "residual-and-step-converged",
    "stationary-warm-start-complete",
    "stationary-residual-converged",
    "trust-region-collapsed",
    "finite-difference-domain-exhausted",
    "iteration-limit-converged",
    "maximum-iterations-exhausted",
]


@dataclass(frozen=True)
class NonlinearDiagnostics:
    """Structured accounting for one bounded nonlinear-solver segment."""

    terminal_reason: NonlinearTerminalReason
    residual_evaluations: int
    jacobian_evaluations: int
    accepted_iterations: int
    rejected_trial_evaluations: int
    infeasible_trial_rejections: int
    rejected_directions: int
    merit_rejections: int
    acceptance_test_rejections: int
    analytic_restarts: int
    finite_difference_jacobian_rebuilds: int
    jacobian_refreshes: int
    smallest_trust_radius: float
    final_trust_radius: float
    final_residual_rms: float
    final_residual_maximum: float
    final_worst_residual_index: int
    rejected_trial_line_search_factors: tuple[float, ...]
    rejected_trial_residual_maxima: tuple[float, ...]
    rejected_trial_worst_residual_indices: tuple[int, ...]
    model_screened_trials: int = 0


@dataclass(frozen=True)
class NonlinearResult(Generic[Payload]):
    """Final nonlinear state and convergence history."""

    state: FloatArray
    evaluation: NonlinearEvaluation[Payload]
    converged: bool
    iterations: int
    history: tuple[NonlinearIteration, ...]
    diagnostics: NonlinearDiagnostics


def nonlinear_result_metadata(
    result: NonlinearResult[object],
) -> dict[str, object]:
    """Return JSON-friendly telemetry for one nonlinear-solver segment."""

    diagnostics = result.diagnostics
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "terminal_reason": diagnostics.terminal_reason,
        "residual_evaluations": diagnostics.residual_evaluations,
        "jacobian_evaluations": diagnostics.jacobian_evaluations,
        "accepted_iterations": diagnostics.accepted_iterations,
        "rejected_trial_evaluations": (
            diagnostics.rejected_trial_evaluations
        ),
        "infeasible_trial_rejections": (
            diagnostics.infeasible_trial_rejections
        ),
        "rejected_directions": diagnostics.rejected_directions,
        "merit_rejections": diagnostics.merit_rejections,
        "acceptance_test_rejections": (
            diagnostics.acceptance_test_rejections
        ),
        "analytic_restarts": diagnostics.analytic_restarts,
        "model_screened_trials": diagnostics.model_screened_trials,
        "finite_difference_jacobian_rebuilds": (
            diagnostics.finite_difference_jacobian_rebuilds
        ),
        "jacobian_refreshes": diagnostics.jacobian_refreshes,
        "smallest_trust_radius": diagnostics.smallest_trust_radius,
        "final_trust_radius": diagnostics.final_trust_radius,
        "final_residual_rms": diagnostics.final_residual_rms,
        "final_residual_maximum": diagnostics.final_residual_maximum,
        "final_worst_residual_index": (
            diagnostics.final_worst_residual_index
        ),
        "rejected_trial_line_search_factors": (
            diagnostics.rejected_trial_line_search_factors
        ),
        "rejected_trial_residual_maxima": (
            # An infeasible trial has no measured residual. Keep the
            # internal infinity sentinel, but export an explicit JSON null
            # rather than making even a successful checkpoint unsaveable.
            tuple(value if np.isfinite(value) else None
                  for value in diagnostics.rejected_trial_residual_maxima)
        ),
        "rejected_trial_worst_residual_indices": (
            diagnostics.rejected_trial_worst_residual_indices
        ),
        "iteration_history": tuple(
            {
                "iteration": record.iteration,
                "residual_rms": record.residual_rms,
                "residual_maximum": record.residual_maximum,
                "residual_merit": record.residual_merit,
                "worst_residual_index": record.worst_residual_index,
                "maximum_step": record.maximum_step,
                "unrestricted_maximum_step": record.unrestricted_maximum_step,
                "trust_radius": record.trust_radius,
                "line_search_factor": record.line_search_factor,
                "rejected_trial_evaluations": (
                    record.rejected_trial_evaluations
                ),
                "jacobian_recomputed": record.jacobian_recomputed,
                "model_agreement": record.model_agreement,
                "proposal_limited": record.proposal_limited,
                "step_kind": record.step_kind,
            }
            for record in result.history
        ),
    }


def _validated_evaluation(
    evaluation: NonlinearEvaluation[Payload],
    state_size: int,
    *,
    require_jacobian: bool,
) -> NonlinearEvaluation[Payload]:
    residual = np.asarray(evaluation.residual, dtype=np.float64)
    if residual.shape != (state_size,) or np.any(~np.isfinite(residual)):
        raise ValueError("nonlinear residual must be a finite state-sized vector")
    jacobian = evaluation.jacobian
    if require_jacobian and jacobian is None:
        raise ValueError("a Jacobian was requested but not returned")
    if jacobian is not None:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        if jacobian.shape != (state_size, state_size):
            raise ValueError("nonlinear Jacobian must be square and state-sized")
        if np.any(~np.isfinite(jacobian)):
            raise ValueError("nonlinear Jacobian must be finite")
    return NonlinearEvaluation(residual, jacobian, evaluation.payload)


def _residual_merit(residual: FloatArray) -> float:
    """Balance global progress against a localized failed depth cell."""

    rms = float(np.sqrt(np.mean(residual**2)))
    maximum = float(np.max(np.abs(residual)))
    return rms + 0.25 * maximum


def solve_trust_region_newton(
    initial_state: FloatArray,
    evaluate: Callable[[FloatArray, bool], NonlinearEvaluation[Payload]],
    *,
    maximum_iterations: int = 60,
    residual_tolerance: float = 2.0e-3,
    step_tolerance: float = 2.0e-4,
    initial_trust_radius: float = 0.04,
    maximum_trust_radius: float = 0.12,
    minimum_trust_radius: float = 1.0e-5,
    jacobian_refresh_interval: int = 4,
    minimum_line_search_factor: float = 1.0 / 64.0,
    finite_difference_fallback_step: float | None = 1.0e-4,
    callback: Callable[
        [NonlinearIteration, FloatArray, NonlinearEvaluation[Payload]], None
    ]
    | None = None,
    convergence_test: Callable[
        [FloatArray, NonlinearEvaluation[Payload], float], bool
    ]
    | None = None,
    acceptance_test: Callable[
        [NonlinearEvaluation[Payload], NonlinearEvaluation[Payload]], bool
    ]
    | None = None,
    rejected_step_handoff: Callable[
        [FloatArray, NonlinearEvaluation[Payload]], bool
    ]
    | None = None,
    accepted_state_handoff: Callable[
        [FloatArray, NonlinearEvaluation[Payload]], bool
    ]
    | None = None,
    step_measure: Callable[[FloatArray, FloatArray], float] | None = None,
    trust_step_measure: Callable[[FloatArray, FloatArray], float] | None = None,
    stationary_completion_iterations: int | None = None,
    allow_initial_convergence: bool = True,
    linear_regularization: float = 1.0e-8,
    trial_projector: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    step_builder: Callable[
        [FloatArray, NonlinearEvaluation[Payload], FloatArray, float], FloatArray | NonlinearProposal
    ] | None = None,
    merit_function: Literal["rms-max", "least-squares"] = "rms-max",
    trust_update: Literal["legacy", "model-agreement"] = "legacy",
    broyden_updates: bool = True,
    nonlinear_model_globalization: bool = False,
    iteration_correction: Callable[
        [FloatArray, NonlinearEvaluation[Payload]], NonlinearCorrection[Payload] | None
    ] | None = None,
    converge_when_proposal_limited: bool = False,
) -> NonlinearResult[Payload]:
    """Solve a square nonlinear system with automatic globalization.

    The caller controls the physics through ``evaluate(state, jacobian)``.
    Newton steps are row-scaled and solved in a least-squares sense, making a
    singular provisional atmosphere recoverable.  A trust radius limits the
    largest state change; backtracking accepts only reductions of a merit
    function containing both RMS and maximum residuals.  Between exact or
    approximate Jacobian refreshes, a rank-one Broyden update communicates
    the observed opacity/EOS response without extra formal transfer solves.
    ``stationary_completion_iterations`` is reserved for explicitly
    approximate warm-start phases: it permits a caller to move on after that
    many consecutive accepted physical steps fall below ``step_tolerance``,
    even though the provisional residual is not the final equation root.
    An optional ``trial_projector(old, proposed)`` may repair a trial state
    before evaluation. Its actual displacement must satisfy the trust radius;
    the repaired state, not the original proposal, is evaluated and retained.
    This hook cannot bypass merit reduction or the convergence tests.
    ``accepted_state_handoff`` may end a provisional continuation phase at
    the initial or a genuinely accepted state. It returns ``converged=False``
    and cannot certify equilibrium or change trial acceptance. Final root
    solves must leave this optional hook disabled.
    ``rejected_step_handoff`` may request a different residual representation
    after a whole direction is rejected. It returns the unchanged accepted
    state with ``converged=False``; it cannot certify or accept a rejected trial.
    An optional ``step_builder(state, evaluation, jacobian, radius)`` supplies
    an alternative proposal direction. It still passes through the same
    physical step measure, trust limit, trial evaluation and acceptance gates.
    ``allow_initial_convergence=False`` requires a computed correction before
    certifying stationarity, for example after an unfinished solver phase.
    ``linear_regularization=0`` selects an unpenalized, rank-revealing linear
    solve; it never removes the physical trust bound or trial acceptance tests.
    ``merit_function='least-squares'`` uses half the mean squared residual for
    acceptance AND model agreement, matching an unweighted bounded subproblem.
    ``trust_update='model-agreement'`` requires a positive predicted reduction,
    good agreement and an unmodified boundary-sized step before radius growth.
    ``broyden_updates=False`` preserves fresh analytic tangents, including after
    rejected trials. Existing approximate-tangent callers retain their defaults.
    ``trust_step_measure`` optionally measures the full trust-region geometry
    independently of the physical ``step_measure`` used for stationarity and
    history. Use it when the bounded subproblem includes auxiliary variables.
    Clipping, trial bounds and boundary-triggered radius growth all use this
    same measure. None preserves the existing physical-step geometry.
    ``iteration_correction`` optionally supplies a separate material update,
    with its own backtracking rather than Newton's trust bound. It is tried
    at most once between Newton attempts, after checking measured stationarity.
    Accepted corrections require full residual evaluation, all acceptance
    guards and a fresh Jacobian. They never establish convergence or alter
    Newton's trust radius. The default None leaves existing solves unchanged.
    ``nonlinear_model_globalization`` screens trials with the supplied nonlinear
    residual model, and recomputes a smaller bounded proposal after rejection.
    It requires fresh analytic tangents (no Broyden updates). Screening can
    only reject: every accepted step still requires a physical evaluation.
    Default False preserves the existing rejection/restart policy.
    """

    state = np.asarray(initial_state, dtype=np.float64).copy()
    if state.ndim != 1 or state.size < 2 or np.any(~np.isfinite(state)):
        raise ValueError("initial_state must be a finite one-dimensional vector")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not isinstance(allow_initial_convergence, bool):
        raise ValueError("allow_initial_convergence must be boolean")
    if merit_function not in ("rms-max", "least-squares"):
        raise ValueError("unknown nonlinear merit function")
    if trust_update not in ("legacy", "model-agreement"):
        raise ValueError("unknown trust update policy")
    if not isinstance(broyden_updates, bool):
        raise ValueError("broyden_updates must be boolean")
    if not isinstance(nonlinear_model_globalization, bool):
        raise ValueError("nonlinear_model_globalization must be boolean")
    if nonlinear_model_globalization and broyden_updates:
        raise ValueError("nonlinear model globalization requires broyden_updates=False")
    if nonlinear_model_globalization and jacobian_refresh_interval != 1:
        raise ValueError("nonlinear model globalization requires jacobian_refresh_interval=1")
    if not np.isfinite(linear_regularization) or linear_regularization < 0:
        raise ValueError("linear_regularization must be finite and nonnegative")
    if residual_tolerance <= 0.0 or step_tolerance <= 0.0:
        raise ValueError("nonlinear tolerances must be positive")
    if not 0.0 < minimum_line_search_factor <= 1.0:
        raise ValueError("minimum_line_search_factor must lie in (0, 1]")
    if not 0.0 < minimum_trust_radius <= initial_trust_radius:
        raise ValueError("trust radii must be positive and ordered")
    if maximum_trust_radius < initial_trust_radius:
        raise ValueError("maximum_trust_radius must exceed the initial value")
    if jacobian_refresh_interval < 1:
        raise ValueError("jacobian_refresh_interval must be positive")
    if finite_difference_fallback_step is not None and (
        not np.isfinite(finite_difference_fallback_step)
        or finite_difference_fallback_step <= 0.0
    ):
        raise ValueError(
            "finite_difference_fallback_step must be positive or None"
        )
    if (
        stationary_completion_iterations is not None
        and stationary_completion_iterations < 1
    ):
        raise ValueError(
            "stationary_completion_iterations must be positive or None"
        )

    def merit(residual):
        if merit_function == "least-squares":
            return 0.5 * float(np.mean(residual**2))
        return _residual_merit(residual)

    trust_measure = step_measure if trust_step_measure is None else trust_step_measure

    def trust_size(old, new):
        value = (float(np.max(abs(new-old))) if trust_measure is None
                 else float(trust_measure(old, new)))
        if not np.isfinite(value) or value < 0:
            raise ValueError("trust step measure must return a finite non-negative value")
        return value

    trust_radius = float(initial_trust_radius)
    smallest_trust_radius = trust_radius
    history: list[NonlinearIteration] = []
    residual_evaluations = 0
    jacobian_evaluations = 0
    rejected_trial_evaluations = 0
    infeasible_trial_rejections = 0
    rejected_directions = 0
    merit_rejections = 0
    acceptance_test_rejections = 0
    analytic_restarts = 0
    model_screened_trials = 0
    finite_difference_jacobian_rebuilds = 0
    jacobian_refreshes = 0
    rejected_trial_line_search_factors: list[float] = []
    rejected_trial_residual_maxima: list[float] = []
    rejected_trial_worst_residual_indices: list[int] = []

    def evaluated(
        candidate_state: FloatArray,
        need_jacobian: bool,
    ) -> NonlinearEvaluation[Payload]:
        nonlocal residual_evaluations, jacobian_evaluations
        residual_evaluations += 1
        jacobian_evaluations += int(need_jacobian)
        started = time.monotonic()
        if need_jacobian:
            _LOGGER.info("Building Jacobian (evaluation %d, %d variables)",
                         residual_evaluations, state.size)
        result = _validated_evaluation(
            evaluate(candidate_state, need_jacobian),
            state.size,
            require_jacobian=need_jacobian,
        )
        if need_jacobian:
            _LOGGER.info("Jacobian ready in %.2fs; maximum residual %.6g",
                         time.monotonic() - started,
                         np.max(np.abs(result.residual)))
        return result

    def finished(
        terminal_reason: NonlinearTerminalReason,
        converged: bool,
        iterations: int,
    ) -> NonlinearResult[Payload]:
        residual = evaluation.residual
        absolute_residual = np.abs(residual)
        diagnostics = NonlinearDiagnostics(
            terminal_reason=terminal_reason,
            residual_evaluations=residual_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            accepted_iterations=sum(
                record.line_search_factor > 0.0 for record in history
            ),
            rejected_trial_evaluations=rejected_trial_evaluations,
            infeasible_trial_rejections=infeasible_trial_rejections,
            rejected_directions=rejected_directions,
            merit_rejections=merit_rejections,
            acceptance_test_rejections=acceptance_test_rejections,
            analytic_restarts=analytic_restarts,
            model_screened_trials=model_screened_trials,
            finite_difference_jacobian_rebuilds=(
                finite_difference_jacobian_rebuilds
            ),
            jacobian_refreshes=jacobian_refreshes,
            smallest_trust_radius=smallest_trust_radius,
            final_trust_radius=trust_radius,
            final_residual_rms=float(np.sqrt(np.mean(residual**2))),
            final_residual_maximum=float(np.max(absolute_residual)),
            final_worst_residual_index=int(np.argmax(absolute_residual)),
            rejected_trial_line_search_factors=tuple(
                rejected_trial_line_search_factors
            ),
            rejected_trial_residual_maxima=tuple(
                rejected_trial_residual_maxima
            ),
            rejected_trial_worst_residual_indices=tuple(
                rejected_trial_worst_residual_indices
            ),
        )
        return NonlinearResult(
            state,
            evaluation,
            converged,
            iterations,
            tuple(history),
            diagnostics,
        )

    def finite_difference_jacobian(
        current_state: FloatArray,
        current_evaluation: NonlinearEvaluation[Payload],
    ) -> FloatArray:
        assert finite_difference_fallback_step is not None
        numerical = np.empty((state.size, state.size), dtype=np.float64)
        for column in range(state.size):
            # The caller chooses state variables with useful numerical units
            # (atmospheres use ln T), so this is an absolute state increment.
            # Scaling by the value itself would make a log-state perturbation
            # depend spuriously on the choice of dimensional zero point.
            step = finite_difference_fallback_step
            trial_state = current_state.copy()
            trial_state[column] += step
            try:
                trial = evaluated(trial_state, False)
            except RecoverableEvaluationError:
                # A valid atmosphere can lie next to a material-table or
                # chemical-approximation boundary. Differentiate the SAME
                # residual from the admissible side, not a substituted EOS.
                # Ordinary forward differences remain bit-for-bit unchanged.
                step = -step
                trial_state = current_state.copy()
                trial_state[column] += step
                trial = evaluated(trial_state, False)
            numerical[:, column] = (
                trial.residual - current_evaluation.residual
            ) / step
        return numerical

    # A restart may already satisfy the requested physical tolerances.  Test
    # that state with one residual evaluation before paying for a Jacobian;
    # in atmosphere problems the latter can require one expensive opacity
    # and transfer derivative per depth point.  Passing a zero step is the
    # mathematically appropriate stationarity test for an unchanged restart.
    evaluation = evaluated(state, False)
    initial_residual_maximum = float(np.max(np.abs(evaluation.residual)))
    if accepted_state_handoff is not None and accepted_state_handoff(state.copy(), evaluation):
        return finished("accepted-state-phase-handoff", False, 0)
    if (
        allow_initial_convergence
        and initial_residual_maximum < residual_tolerance
        and (
            convergence_test is None
            or convergence_test(state, evaluation, 0.0)
        )
    ):
        return finished("initial-state-converged", True, 0)

    evaluation = evaluated(state, True)
    assert evaluation.jacobian is not None
    jacobian = evaluation.jacobian.copy()
    last_step_maximum = np.inf
    last_proposal_limited = False
    numerical_fallback_used_at_state = False
    rejected_secant_repairs_at_state = 0
    analytic_restart_used_at_state = False
    correction_pending = iteration_correction is not None

    def accepted_step_converged() -> bool:
        """Apply the same accepted-state predicate at every iteration exit."""
        return bool(
            np.max(np.abs(evaluation.residual)) < residual_tolerance
            and last_step_maximum < step_tolerance
            # Preserve the caller's explicit limited-proposal certificate
            # policy at the iteration cap as well as between iterations.
            and (not last_proposal_limited or converge_when_proposal_limited)
            and (
                convergence_test is None
                or convergence_test(state, evaluation, last_step_maximum)
            )
        )

    def stationary_result_if_converged(
        iteration: int,
        stationary_step: float,
    ) -> NonlinearResult[Payload] | None:
        """Certify a measured small proposal, never a collapsed trust radius."""

        residual = evaluation.residual
        converged = bool(
            np.max(np.abs(residual)) < residual_tolerance
            and stationary_step < step_tolerance
            and (
                convergence_test is None
                or convergence_test(state, evaluation, stationary_step)
            )
        )
        if not converged:
            return None
        record = NonlinearIteration(
            iteration=iteration,
            residual_rms=float(np.sqrt(np.mean(residual**2))),
            residual_maximum=float(np.max(np.abs(residual))),
            maximum_step=stationary_step,
            trust_radius=trust_radius,
            line_search_factor=0.0,
            jacobian_recomputed=False,
            residual_merit=merit(residual),
            worst_residual_index=int(np.argmax(np.abs(residual))),
            rejected_trial_evaluations=0,
            model_agreement=None,
            unrestricted_maximum_step=stationary_step,
        )
        history.append(record)
        if callback is not None:
            callback(record, state.copy(), evaluation)
        return finished("stationary-residual-converged", True, iteration)

    for iteration in range(1, maximum_iterations + 1):
        residual = evaluation.residual
        if accepted_step_converged():
            return finished(
                "residual-and-step-converged", True, iteration - 1
            )
        if (
            stationary_completion_iterations is not None
            and len(history) >= stationary_completion_iterations
            and all(
                record.maximum_step < step_tolerance and not record.proposal_limited
                for record in history[-stationary_completion_iterations:]
            )
            and (
                convergence_test is None
                or convergence_test(state, evaluation, last_step_maximum)
            )
        ):
            # Some callers use this solver for an explicitly approximate
            # warm-start phase before imposing the final equations.  Once
            # repeated accepted physical changes are below the requested
            # structural tolerance, continuing to chase a nonzero residual
            # of that provisional system only consumes expensive opacity and
            # transfer evaluations.  The option is disabled for ordinary
            # root solves; atmosphere preconditioners opt in explicitly.
            return finished(
                "stationary-warm-start-complete", True, iteration - 1
            )

        proposal_limited = False
        proposal_model = None
        if step_builder is None:
            row_scale = np.maximum(
                np.max(np.abs(jacobian), axis=1),
                np.finfo(np.float64).tiny,
            )
            scaled_jacobian = jacobian / row_scale[:, np.newaxis]
            scaled_residual = residual / row_scale
            # A small diagonal Tikhonov term selects a smooth finite step when a
            # provisional convection boundary makes the local Jacobian singular.
            augmented_matrix = scaled_jacobian
            augmented_rhs = -scaled_residual
            if linear_regularization > 0:
                augmented_matrix = np.vstack(
                    (
                        scaled_jacobian,
                        np.sqrt(linear_regularization) * np.eye(state.size),
                    )
                )
                augmented_rhs = np.concatenate(
                    (-scaled_residual, np.zeros(state.size))
                )
            step = np.linalg.lstsq(
                augmented_matrix, augmented_rhs, rcond=1.0e-10
            )[0]
        else:
            proposed = step_builder(state.copy(), evaluation, jacobian.copy(), trust_radius)
            if isinstance(proposed, NonlinearProposal):
                if not isinstance(proposed.limited, bool):
                    raise ValueError("proposal limited flag must be boolean")
                proposal_limited = proposed.limited
                proposal_model = proposed.residual_model
                proposed = proposed.direction
            step = np.asarray(
                proposed,
                dtype=np.float64,
            ).copy()
            if step.shape != state.shape or np.any(~np.isfinite(step)):
                raise ValueError(
                    "step_builder must return a finite direction with the state shape"
                )
            if proposal_model is not None:
                if not callable(proposal_model):
                    raise ValueError("proposal residual model must be callable")
                origin = np.asarray(proposal_model(np.zeros_like(state)), dtype=float)
                if (origin.shape != evaluation.residual.shape or np.any(~np.isfinite(origin))
                        or not np.allclose(origin, evaluation.residual, rtol=1e-10, atol=1e-13)):
                    raise ValueError("proposal residual model must match the physical anchor residual")
        step_maximum = (
            float(np.max(np.abs(step)))
            if step_measure is None
            else float(step_measure(state, state + step))
        )
        if not np.isfinite(step_maximum) or step_maximum < 0.0:
            raise ValueError(
                "step_measure must return a finite non-negative value"
            )
        # At an accurate root a genuine Newton correction may be too small
        # to reduce the residual further. Check that computed direction BEFORE
        # trust clipping or backtracking. A user step builder may itself be
        # bounded, so a sub-tolerance trust region cannot certify its size.
        if not proposal_limited and (step_builder is None or trust_radius >= step_tolerance):
            stationary = stationary_result_if_converged(iteration, step_maximum)
            if stationary is not None:
                return stationary
        if correction_pending:
            correction_pending = False
            try:
                correction = iteration_correction(state.copy(), evaluation)
            except RecoverableEvaluationError as error:
                _LOGGER.info("Material correction unavailable: %s", error)
                correction = None
            if correction is not None:
                if not isinstance(correction, NonlinearCorrection):
                    raise ValueError("iteration_correction must return NonlinearCorrection or None")
                factor = 1.0
                correction_rejections = 0
                old_merit = merit(evaluation.residual)
                while factor >= minimum_line_search_factor:
                    try:
                        candidate = np.asarray(correction.trial_state(factor), dtype=float).copy()
                        if candidate.shape != state.shape or np.any(~np.isfinite(candidate)):
                            raise ValueError("material correction must return a finite state of the original shape")
                        size = (float(np.max(abs(candidate-state))) if step_measure is None
                                else float(step_measure(state, candidate)))
                        if not np.isfinite(size) or size < 0:
                            raise ValueError("step_measure must return a finite non-negative value")
                        trial = evaluated(candidate, False)
                    except RecoverableEvaluationError:
                        infeasible_trial_rejections += 1
                        trial = None
                    if trial is not None:
                        improves = merit(trial.residual) < old_merit
                        guarded = (improves
                            and (acceptance_test is None or acceptance_test(evaluation, trial))
                            and (correction.acceptance_test is None or correction.acceptance_test(trial)))
                        if guarded:
                            # A nonlinear material map has no Newton model
                            # ratio. Discard that model, not its trust radius.
                            state = candidate
                            evaluation = evaluated(state, True)
                            jacobian = evaluation.jacobian.copy()
                            jacobian_refreshes += 1
                            last_step_maximum = np.inf
                            last_proposal_limited = True
                            numerical_fallback_used_at_state = False
                            rejected_secant_repairs_at_state = 0
                            analytic_restart_used_at_state = False
                            record = NonlinearIteration(
                                iteration=iteration,
                                residual_rms=float(np.sqrt(np.mean(evaluation.residual**2))),
                                residual_maximum=float(np.max(abs(evaluation.residual))),
                                maximum_step=size, trust_radius=trust_radius,
                                line_search_factor=factor, jacobian_recomputed=True,
                                residual_merit=merit(evaluation.residual),
                                worst_residual_index=int(np.argmax(abs(evaluation.residual))),
                                rejected_trial_evaluations=correction_rejections,
                                model_agreement=None, proposal_limited=True,
                                step_kind="material-correction")
                            history.append(record)
                            if callback is not None:
                                callback(record, state.copy(), evaluation)
                            break
                        merit_rejections += int(not improves)
                        acceptance_test_rejections += int(improves)
                    rejected_trial_evaluations += 1
                    correction_rejections += 1
                    rejected_trial_line_search_factors.append(float(factor))
                    rejected_trial_residual_maxima.append(
                        float(np.max(abs(trial.residual))) if trial is not None else float("inf"))
                    rejected_trial_worst_residual_indices.append(
                        int(np.argmax(abs(trial.residual))) if trial is not None else -1)
                    factor *= 0.5
                else:
                    _LOGGER.info("Material correction rejected after %d trials", correction_rejections)
                if factor >= minimum_line_search_factor:
                    # Count the material update as a real iteration. A global
                    # Newton attempt is mandatory before another correction.
                    continue
        # Keep the default arithmetic unchanged; an enlarged system can supply
        # its own native geometry without changing physical stationarity.
        bounded_size = (step_maximum if trust_step_measure is None
                        else trust_size(state, state + step))
        if bounded_size > trust_radius:
            step *= trust_radius / bounded_size

        old_state = state
        old_evaluation = evaluation
        old_merit = merit(old_evaluation.residual)
        factor = 1.0
        accepted = False
        rejected_trials_this_iteration = 0
        trial_state = old_state
        trial = old_evaluation
        last_valid_rejected_state: FloatArray | None = None
        last_valid_rejected_evaluation: NonlinearEvaluation[Payload] | None = (
            None
        )
        while factor >= minimum_line_search_factor:
            trial_state = old_state + factor * step
            try:
                if trust_update == "model-agreement":
                    # Nonlinear coordinates need not scale proportionally
                    # under backtracking. Check the actual decoded trial.
                    trial_size = trust_size(old_state, trial_state)
                    if trial_size > trust_radius*(1+128*np.finfo(float).eps):
                        raise RecoverableEvaluationError("trial exceeds physical trust radius")
                if trial_projector is not None:
                    unprojected_state = trial_state
                    trial_state = np.asarray(
                        trial_projector(old_state.copy(), trial_state.copy()),
                        dtype=np.float64,
                    ).copy()
                    if trial_state.shape != state.shape or np.any(~np.isfinite(trial_state)):
                        raise ValueError("trial_projector must return a finite state of the original shape")
                    projected_size = trust_size(old_state, trial_state)
                    if not np.isfinite(projected_size) or projected_size < 0.:
                        raise ValueError("projected step measure must be finite and non-negative")
                    roundoff_allowance = 128.*np.finfo(float).eps*max(
                        1., float(np.max(abs(old_state))), float(np.max(abs(trial_state))))
                    if (not np.array_equal(trial_state, unprojected_state)
                            and projected_size > trust_radius+roundoff_allowance):
                        raise RecoverableEvaluationError("projected trial exceeds trust radius")
                if nonlinear_model_globalization and proposal_model is not None:
                    prediction = np.asarray(proposal_model(trial_state-old_state), dtype=float)
                    if prediction.shape != residual.shape or np.any(~np.isfinite(prediction)):
                        raise ValueError("proposal residual model must return a finite state-sized vector")
                    if merit(prediction) >= old_merit:
                        model_screened_trials += 1
                        _LOGGER.info("Skipped model-uphill trial (factor %.6g)", factor)
                        factor *= 0.5
                        continue
                trial = evaluated(trial_state, False)
            except RecoverableEvaluationError:
                # The proposed state is outside the caller's physical
                # domain.  This is a normal globalization event: reject it
                # and try a smaller step.  Do not catch ValueError or other
                # exceptions here, because those indicate a broken solver
                # contract rather than an inadmissible trial atmosphere.
                rejected_trial_evaluations += 1
                infeasible_trial_rejections += 1
                rejected_trials_this_iteration += 1
                rejected_trial_line_search_factors.append(float(factor))
                rejected_trial_residual_maxima.append(float("inf"))
                rejected_trial_worst_residual_indices.append(-1)
                factor *= 0.5
                continue
            merit_improved = merit(trial.residual) < old_merit
            accepted_by_guard = bool(
                merit_improved
                and (
                    acceptance_test is None
                    or acceptance_test(old_evaluation, trial)
                )
            )
            if accepted_by_guard:
                state = trial_state
                evaluation = trial
                accepted = True
                break
            rejected_trial_evaluations += 1
            rejected_trials_this_iteration += 1
            rejected_trial_line_search_factors.append(float(factor))
            rejected_trial_residual_maxima.append(
                float(np.max(np.abs(trial.residual)))
            )
            rejected_trial_worst_residual_indices.append(
                int(np.argmax(np.abs(trial.residual)))
            )
            if merit_improved:
                acceptance_test_rejections += 1
            else:
                merit_rejections += 1
            last_valid_rejected_state = trial_state
            last_valid_rejected_evaluation = trial
            factor *= 0.5

        if not accepted:
            correction_pending = iteration_correction is not None
            rejected_directions += 1
            _LOGGER.info(
                "Iteration %d: direction rejected after %d trials; "
                "maximum residual %.6g, trust radius %.6g",
                iteration, rejected_trials_this_iteration,
                np.max(np.abs(evaluation.residual)), trust_radius,
            )
            if (rejected_step_handoff is not None
                    and rejected_step_handoff(state.copy(), evaluation)):
                return finished("rejected-step-phase-handoff", False, iteration - 1)
            if nonlinear_model_globalization and proposal_model is not None:
                # At an unchanged physical anchor the exact analytic tangent
                # has not become stale. Re-solving a smaller nonlinear box is
                # different from scaling the rejected direction through a
                # strongly curved constitutive response.
                trust_radius *= 0.25
                smallest_trust_radius = min(smallest_trust_radius, trust_radius)
                if trust_radius < minimum_trust_radius:
                    return finished("trust-region-collapsed", False, iteration - 1)
                _LOGGER.info("Recomputing nonlinear proposal at radius %.6g", trust_radius)
                continue
            # The backtracking evaluations have measured a true directional
            # derivative even though none of their steps lowered the merit.
            # Retain that information instead of immediately rebuilding the
            # same approximate Jacobian and proposing the same uphill step.
            # Successive rejected directions build an inexpensive low-rank
            # recovery for large systems where a complete finite-difference
            # Jacobian would require one transfer solve per depth point.
            if (
                last_valid_rejected_state is not None
                and last_valid_rejected_evaluation is not None
            ):
                sampled_step = last_valid_rejected_state - old_state
                sampled_denominator = float(sampled_step @ sampled_step)
            else:
                sampled_step = np.zeros_like(old_state)
                sampled_denominator = 0.0
            if broyden_updates and sampled_denominator > np.finfo(np.float64).tiny:
                sampled_residual_change = (
                    last_valid_rejected_evaluation.residual
                    - old_evaluation.residual
                )
                jacobian += np.outer(
                    sampled_residual_change - jacobian @ sampled_step,
                    sampled_step,
                ) / sampled_denominator
                rejected_secant_repairs_at_state += 1
                trust_radius *= 0.5
                smallest_trust_radius = min(
                    smallest_trust_radius, trust_radius
                )
                if trust_radius < minimum_trust_radius:
                    return finished(
                        "trust-region-collapsed", False, iteration - 1
                    )
                if rejected_secant_repairs_at_state < 2:
                    continue
            # Large atmosphere systems deliberately disable a column-by-
            # column finite-difference Jacobian: one such fallback would
            # require a complete opacity/transfer solve for every depth
            # point.  Previously that also made the branch above retain
            # rejected rank-one repairs until the trust radius collapsed.
            # A fresh analytic tangent is much cheaper and is equivalent to
            # restarting from the last accepted atmosphere with no stale
            # Broyden history.  Permit one such restart at each physical
            # state; if it still fails, the ordinary trust collapse below
            # remains the bounded failure mode.
            if (
                finite_difference_fallback_step is None
                and not analytic_restart_used_at_state
            ):
                evaluation = evaluated(state, True)
                assert evaluation.jacobian is not None
                jacobian = evaluation.jacobian.copy()
                trust_radius = max(trust_radius, initial_trust_radius)
                rejected_secant_repairs_at_state = 0
                analytic_restart_used_at_state = True
                analytic_restarts += 1
                jacobian_refreshes += 1
                continue
            # A frozen-opacity or other approximate physics block can point
            # uphill after a large EOS/opacity change.  Before making the
            # trust region arbitrarily small, measure the complete residual
            # derivative once at this state.  This is deliberately a recovery
            # path rather than the normal iteration cost.
            if (
                finite_difference_fallback_step is not None
                and not numerical_fallback_used_at_state
            ):
                try:
                    jacobian = finite_difference_jacobian(state, evaluation)
                except RecoverableEvaluationError:
                    # Neither probe is admissible at the requested numerical
                    # scale. Keep the last evaluated state, explicitly
                    # unconverged; do not invent a derivative or flux.
                    return finished("finite-difference-domain-exhausted", False, iteration - 1)
                numerical_fallback_used_at_state = True
                finite_difference_jacobian_rebuilds += 1
                jacobian_refreshes += 1
                continue
            trust_radius *= 0.25
            smallest_trust_radius = min(smallest_trust_radius, trust_radius)
            if trust_radius < minimum_trust_radius:
                return finished(
                    "trust-region-collapsed", False, iteration - 1
                )
            # A rejected linearization is stale; rebuild it at the unchanged
            # physical state before proposing another direction.
            evaluation = evaluated(state, True)
            assert evaluation.jacobian is not None
            jacobian = evaluation.jacobian.copy()
            jacobian_refreshes += 1
            numerical_fallback_used_at_state = False
            rejected_secant_repairs_at_state = 0
            continue

        accepted_step = state - old_state
        correction_pending = iteration_correction is not None
        last_proposal_limited = proposal_limited
        last_step_maximum = (
            float(np.max(np.abs(accepted_step)))
            if step_measure is None
            else float(step_measure(old_state, state))
        )
        if not np.isfinite(last_step_maximum) or last_step_maximum < 0.0:
            raise ValueError("step_measure must return a finite non-negative value")
        numerical_fallback_used_at_state = False
        rejected_secant_repairs_at_state = 0
        analytic_restart_used_at_state = False
        actual_reduction = old_merit - merit(evaluation.residual)
        linear_residual = (old_evaluation.residual + jacobian @ accepted_step
            if proposal_model is None else np.asarray(proposal_model(accepted_step), dtype=float))
        if linear_residual.shape != old_evaluation.residual.shape or np.any(~np.isfinite(linear_residual)):
            raise ValueError("proposal residual model must return a finite state-sized vector")
        predicted_reduction = old_merit - merit(linear_residual)
        if trust_update == "model-agreement":
            agreement = (actual_reduction / predicted_reduction
                if predicted_reduction > 0 else None)
        else:
            agreement = actual_reduction / max(
                predicted_reduction, np.finfo(np.float64).tiny
            )
        # In the legacy policy, backtracking has established that an accepted
        # step lowers the nonlinear merit. A poor model ratio requests a
        # Jacobian refresh, but does not by itself collapse the trust radius;
        # doing both makes a systematically approximate Kantorovich block
        # creep forward in minimum-sized steps even while every full step is
        # successful.
        if trust_update == "model-agreement":
            projected = trial_projector is not None and not np.array_equal(
                state, old_state + factor * step)
            if agreement is None or agreement < 0.25 or factor < 0.5:
                trust_radius = max(minimum_trust_radius, 0.5*trust_radius)
            elif (agreement > 0.75 and factor == 1.0 and not proposal_limited
                  and not projected and
                  (last_step_maximum if trust_step_measure is None else trust_size(old_state, state))
                  >= 0.9*trust_radius):
                trust_radius = min(maximum_trust_radius, 1.5*trust_radius)
        elif factor < 0.5:
            trust_radius = max(minimum_trust_radius, 0.5 * trust_radius)
        elif factor == 1.0:
            # Backtracking has directly verified this direction.  Permit a
            # modest recovery from an overly small trust region even while
            # the approximate Jacobian is still learning the local response.
            growth = 1.5 if agreement > 0.75 else 1.25
            trust_radius = min(maximum_trust_radius, growth * trust_radius)
        smallest_trust_radius = min(smallest_trust_radius, trust_radius)

        # A heavily backtracked step is still a measured, downhill secant and
        # is often the most useful information available for an approximate
        # atmosphere tangent.  The trust radius has already been reduced
        # above.  Rebuilding solely because ``factor < 0.5`` discarded that
        # information and made expensive opacity-aware Jacobians repeat after
        # every accepted step.  Refresh on the scheduled cadence or genuinely
        # poor predicted/actual agreement; otherwise let Broyden learn from
        # the accepted nonlinear response below.
        refresh = (
            iteration % jacobian_refresh_interval == 0
            or agreement is None or agreement < 0.1
        )
        if refresh:
            evaluation = evaluated(state, True)
            assert evaluation.jacobian is not None
            jacobian = evaluation.jacobian.copy()
            jacobian_refreshes += 1
            # The legacy Broyden policy also retains the just-measured secant.
            # Exact-tangent callers explicitly disable that modification.
            denominator = float(accepted_step @ accepted_step)
            if broyden_updates and denominator > np.finfo(np.float64).tiny:
                residual_change = (
                    evaluation.residual - old_evaluation.residual
                )
                jacobian += np.outer(
                    residual_change - jacobian @ accepted_step,
                    accepted_step,
                ) / denominator
        else:
            residual_change = evaluation.residual - old_evaluation.residual
            denominator = float(accepted_step @ accepted_step)
            if broyden_updates and denominator > np.finfo(np.float64).tiny:
                jacobian += np.outer(
                    residual_change - jacobian @ accepted_step,
                    accepted_step,
                ) / denominator

        record = NonlinearIteration(
            iteration=iteration,
            residual_rms=float(np.sqrt(np.mean(evaluation.residual**2))),
            residual_maximum=float(np.max(np.abs(evaluation.residual))),
            maximum_step=last_step_maximum,
            trust_radius=trust_radius,
            line_search_factor=factor,
            jacobian_recomputed=refresh,
            residual_merit=merit(evaluation.residual),
            worst_residual_index=int(
                np.argmax(np.abs(evaluation.residual))
            ),
            rejected_trial_evaluations=rejected_trials_this_iteration,
            model_agreement=(
                float(agreement) if agreement is not None and np.isfinite(agreement) else None
            ),
            unrestricted_maximum_step=step_maximum,
            proposal_limited=proposal_limited,
        )
        history.append(record)
        if callback is not None:
            callback(record, state.copy(), evaluation)
        if accepted_state_handoff is not None and accepted_state_handoff(state.copy(), evaluation):
            return finished("accepted-state-phase-handoff", False, iteration)

    final_converged = accepted_step_converged()
    return finished(
        (
            "iteration-limit-converged"
            if final_converged
            else "maximum-iterations-exhausted"
        ),
        final_converged,
        maximum_iterations,
    )
