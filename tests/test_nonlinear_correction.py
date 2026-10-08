"""Independent material updates must not weaken Newton certification."""
import numpy as np
import pytest
from wd_spectra.nonlinear import (
    NonlinearCorrection, NonlinearEvaluation, RecoverableEvaluationError,
    nonlinear_result_metadata, solve_trust_region_newton,
)


def linear(x, need):
    return NonlinearEvaluation(x-np.ones(2), np.eye(2) if need else None, {})


def solve(**options):
    return solve_trust_region_newton(np.zeros(2), linear,
        initial_trust_radius=.01, maximum_trust_radius=.1,
        linear_regularization=0, **options)


def test_default_and_noop_correction_are_identical():
    a, b = solve(), solve(iteration_correction=lambda x, ev: None)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.history == b.history
    assert a.diagnostics == b.diagnostics


def test_correction_exceeds_newton_radius_but_never_certifies_root():
    hook = lambda x, ev: NonlinearCorrection(lambda factor: x+factor*(1-x))
    result = solve(maximum_iterations=1, iteration_correction=hook)
    np.testing.assert_array_equal(result.state, np.ones(2))
    assert not result.converged
    assert result.history[0].maximum_step == 1.
    assert result.history[0].trust_radius == .01
    assert result.history[0].step_kind == 'material-correction'
    assert result.history[0].proposal_limited
    assert result.diagnostics.jacobian_evaluations == 2
    assert result.diagnostics.final_trust_radius == .01
    assert nonlinear_result_metadata(result)['iteration_history'][0]['step_kind'] == 'material-correction'
    measured = solve(maximum_iterations=3, iteration_correction=hook)
    assert measured.converged
    assert measured.history[-1].step_kind == 'newton'
    assert measured.history[-1].line_search_factor == 0


@pytest.mark.parametrize('rejection', ['merit', 'global-guard', 'local-guard', 'domain'])
def test_correction_backtracks_full_physics_and_guards(rejection):
    factors = []
    def hook(x, ev):
        def trial(factor):
            factors.append(factor)
            if rejection == 'domain' and factor == 1:
                raise RecoverableEvaluationError('outside material domain')
            return x+factor*np.full(2, 3. if rejection == 'merit' else 1.5)
        return NonlinearCorrection(trial,
            (lambda ev: ev.residual[0] < 0) if rejection == 'local-guard' else None)
    result = solve(maximum_iterations=1, iteration_correction=hook,
        acceptance_test=(lambda old, new: new.residual[0] < 0) if rejection == 'global-guard' else None)
    assert factors == [1., .5]
    assert result.history[0].line_search_factor == .5
    assert result.diagnostics.rejected_trial_evaluations == 1
    assert not result.converged


def test_rejected_correction_leaves_newton_state_radius_and_history_unchanged():
    rejected = lambda x, ev: NonlinearCorrection(lambda f: x-f*np.ones(2))
    a = solve(maximum_iterations=1)
    b = solve(maximum_iterations=1, iteration_correction=rejected)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.history == b.history
    assert b.diagnostics.rejected_trial_evaluations == 7


def test_global_newton_attempt_between_material_updates():
    hook = lambda x, ev: NonlinearCorrection(lambda f: x+f*.1*(1-x))
    result = solve(maximum_iterations=5, iteration_correction=hook)
    assert [r.step_kind for r in result.history] == [
        'material-correction', 'newton', 'material-correction', 'newton', 'material-correction']


def test_actual_post_correction_jacobian_is_used():
    probes = []
    def nonlinear(x, need):
        if need:
            probes.append(x.copy())
        return NonlinearEvaluation(x**2-4, np.diag(2*x) if need else None, {})
    result = solve_trust_region_newton(np.ones(2), nonlinear,
        maximum_iterations=2, initial_trust_radius=.01, linear_regularization=0,
        iteration_correction=lambda x, ev: NonlinearCorrection(lambda f: x+f*.5))
    np.testing.assert_array_equal(probes[1], np.full(2, 1.5))
    assert result.history[-1].step_kind == 'newton'


@pytest.mark.parametrize('bad', [np.ones(3), np.full(2, np.nan)])
def test_malformed_candidate_is_programming_error_not_hidden_rejection(bad):
    with pytest.raises(ValueError, match='finite state'):
        solve(iteration_correction=lambda x, ev: NonlinearCorrection(lambda f: bad))


def test_hook_programming_error_propagates():
    def bad(x, ev):
        raise RuntimeError('broken implementation')
    with pytest.raises(RuntimeError, match='broken implementation'):
        solve(iteration_correction=bad)
