"""Check derivative freshness without changing dormant public Newton paths."""
from types import SimpleNamespace

import numpy as np

from wd_spectra._hot_recovery_fresh import FreshTangentFallback
from wd_spectra._hot_recovery import ProjectedThermalPopulationCorrection
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton


class FreshProjectedFallback(FreshTangentFallback):
    """Isolate the projected Newton path in the focused tests."""
    def correction(self, x, evaluation):
        return None


def direction(x, e, j, radius):
    return NonlinearProposal(np.linalg.solve(j, -e.residual))


def test_dormant_fresh_fallback_is_bit_identical():
    calls = []
    eq = SimpleNamespace(jacobian_key=None)
    def evaluate(x, jacobian):
        calls.append((x.copy(), jacobian))
        if jacobian:eq.jacobian_key = x.tobytes()
        return NonlinearEvaluation(x-np.array([.1, .2]), np.eye(2) if jacobian else None, None)
    options = dict(maximum_iterations=20, residual_tolerance=1e-12, step_tolerance=1e-10,
                   allow_initial_convergence=False, finite_difference_fallback_step=None,
                   broyden_updates=False, merit_function='least-squares')
    ordinary = solve_trust_region_newton(np.zeros(2), evaluate, step_builder=direction, **options)
    ordinary_calls = calls.copy()
    calls.clear()
    gate = FreshProjectedFallback(ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10), eq)
    recovered = solve_trust_region_newton(np.zeros(2), evaluate,
        step_builder=gate.direction, trial_projector=gate.project,
        rejected_step_handoff=gate.rejected, **options)
    assert ordinary.converged and recovered.converged
    assert not gate.active and gate.extra_jacobian_evaluations == 0
    assert recovered.history == ordinary.history
    np.testing.assert_array_equal(recovered.state, ordinary.state)
    assert len(calls) == len(ordinary_calls)
    for (x, need), (expected, expected_need) in zip(calls, ordinary_calls):
        np.testing.assert_array_equal(x, expected)
        assert need == expected_need


def test_active_recovery_refreshes_a_changed_state_but_reuses_current_tangent():
    eq = SimpleNamespace(jacobian_key=None)
    calls = []
    current = np.array([[2., 0.], [0., 3.]])
    def evaluate(x, need):
        calls.append(need)
        eq.jacobian_key = x.tobytes()
        return NonlinearEvaluation(np.ones(2), current, None)
    gate = FreshProjectedFallback(ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10), eq)
    gate.active = True
    x = np.zeros(2)
    stale = NonlinearEvaluation(np.ones(2), np.eye(2), None)
    proposal = gate.direction(x, stale, stale.jacobian, .12)
    np.testing.assert_allclose(proposal.direction, [-.5, -1./3.])
    gate.direction(x, evaluate(x, False), current, .12)
    assert calls == [True, False]
    assert gate.extra_jacobian_evaluations == 1
    assert gate.correction(x, stale) is None


def test_rejected_curved_population_steps_recover_with_fresh_projected_newton():
    eq = SimpleNamespace(jacobian_key=None)
    def evaluate(x, need):
        t, p = x
        if need:eq.jacobian_key = x.tobytes()
        return NonlinearEvaluation(np.array([1e-8*(t-1.), p-10*t*t]),
            np.array([[1e-8, 0.], [-20*t, 1.]]) if need else None,
            (SimpleNamespace(n_depth=1), None, {'nlte_continuation_fraction': 1.}))
    gate = FreshProjectedFallback(ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10), eq)
    result = solve_trust_region_newton(np.zeros(2), evaluate, maximum_iterations=5,
        residual_tolerance=1e-12, step_tolerance=1e-10, allow_initial_convergence=False,
        finite_difference_fallback_step=None, broyden_updates=False,
        jacobian_refresh_interval=10, merit_function='least-squares',
        step_builder=gate.direction, trial_projector=gate.project, rejected_step_handoff=gate.rejected)
    assert gate.active and gate.rejected_directions >= 2
    assert gate.extra_jacobian_evaluations >= 1
    assert result.state[0] > 0
    assert all(record.step_kind == 'newton' for record in result.history)
    assert not result.converged


def test_material_recovery_uses_the_refreshed_anchor():
    eq = SimpleNamespace(jacobian_key=None)
    fresh = NonlinearEvaluation(np.ones(2), 2*np.eye(2), None)
    def evaluate(x, need):
        assert need
        eq.jacobian_key = x.tobytes()
        return fresh
    proposals = ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10)
    def correction(x, e):
        np.testing.assert_array_equal(proposals.anchor[2], fresh.jacobian)
        return 'diagnostic sentinel'
    proposals.correction = correction
    gate = FreshTangentFallback(proposals, eq)
    x = np.zeros(2)
    assert gate.correction(x, fresh) is None
    gate.active = True
    gate.direction(x, fresh, np.eye(2), .12)
    assert gate.correction(x, fresh) == 'diagnostic sentinel'
