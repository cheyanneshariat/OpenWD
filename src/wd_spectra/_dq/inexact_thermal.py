"""Stop an implicit conditioning step by its error, not two Newton updates.

The transient acceptance uses the controller's common relative L2 residual
target plus physical auxiliary-flux compatibility. The default is 0.1;
explicit experiments may select another contraction. A solved transient is
never a steady-atmosphere certificate. Eight updates are a resource ceiling;
convergence can stop earlier. Steady equations and physical gates are intact.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch
import numpy as np
from . import automatic_conditioning as controller


@dataclass(frozen=True)
class TransientStep:
    """A conditioning result, deliberately without a steady `converged` flag."""
    state: np.ndarray
    evaluation: object
    history: tuple
    transient_accepted: bool
    reason: str = 'budget-exhausted'


def predicts_budget_exhaustion(previous,current,remaining,target):
    """Geometric residual contraction estimate; failure prediction, not proof.

    Like a Newton iteration controller, estimate whether the measured rate
    can achieve its fixed target in the remaining work budget. This may reject
    a solvable large transient step, but never accepts an inaccurate one.
    """
    if not (0<previous and 0<=current and 0<target) or remaining<0:
        raise ValueError('Positive norms/target and nonnegative budget required')
    if not np.all(np.isfinite([previous,current,target])):
        raise ValueError('Finite residual norms required')
    if current<target:
        return False
    rate=current/previous
    return rate>=1. or np.log(current/target)+remaining*np.log(rate)>=0.


def solve_inexact(solver,initial,evaluate,*,compatibility_tolerance,
                  predict_exhaustion=False,**settings):
    if not np.isfinite(compatibility_tolerance) or compatibility_tolerance <= 0:
        raise ValueError('Positive finite compatibility tolerance required')
    first = evaluate(initial,True)
    norm = float(np.linalg.norm(first.residual))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Positive finite initial transient norm required')
    target = controller.THERMAL_RELATIVE_NORM_LIMIT.get()*norm
    # The caller's two-step solve has no convergence certificate on purpose.
    # This replacement certifies ONLY its temporary equation, with exactly
    # the same acceptance test used by thermal_condition after the solve.
    def transient_complete(state,evaluation,step):
        physical = evaluation.payload.payload
        return (np.linalg.norm(evaluation.residual) < target
                and physical['dq_augmented_flux_compatibility'] < compatibility_tolerance)
    # The shared solver correctly refuses to infer root stationarity from
    # bounded proposals. Transient acceptance is different: it requires an
    # actual residual reduction, not a small unrestricted Newton step. Check
    # only its accepted-state callback; never relabel a limited proposal.
    class Complete(Exception):
        def __init__(self,state,evaluation,accepted,reason):
            self.state,self.evaluation = state.copy(),evaluation
            self.accepted,self.reason = accepted,reason
    history = []
    norms = []
    callback = settings.get('callback')
    def observed(record,state,evaluation):
        history.append(record)
        if callback is not None:
            callback(record,state,evaluation)
        if transient_complete(state,evaluation,record.maximum_step):
            raise Complete(state,evaluation,True,'transient-error-target')
        current = float(np.linalg.norm(evaluation.residual))
        if (predict_exhaustion and norms and predicts_budget_exhaustion(
                norms[-1],current,8-len(history),target)):
            raise Complete(state,evaluation,False,'predicted-inner-budget-exhaustion')
        norms.append(current)
    try:
        result = solver(initial,evaluate,**dict(settings,maximum_iterations=8,
            callback=observed,convergence_test=lambda *a:False))
    except Complete as completed:
        return TransientStep(completed.state,completed.evaluation,tuple(history),
                             completed.accepted,completed.reason)
    return TransientStep(result.state,result.evaluation,tuple(result.history),
        bool(transient_complete(result.state,result.evaluation,None)))


@contextmanager
def error_controlled_thermal_steps(*,predict_exhaustion=False):
    original = controller.solve_trust_region_newton
    original_condition = controller.thermal_condition
    active_tolerance = [None]
    def condition(system,state,tolerance,emit,maximum_steps=30):
        previous = active_tolerance[0]
        active_tolerance[0] = tolerance
        try:
            return original_condition(system,state,tolerance,emit,maximum_steps)
        finally:
            active_tolerance[0] = previous
    def solve(initial,evaluate,**settings):
        if active_tolerance[0] is None:
            raise RuntimeError('Inexact transient solve outside thermal conditioning')
        return solve_inexact(original,initial,evaluate,
            compatibility_tolerance=active_tolerance[0],predict_exhaustion=predict_exhaustion,
            **settings)
    with patch.object(controller,'solve_trust_region_newton',solve), \
            patch.object(controller,'thermal_condition',condition):
        yield
