"""Check dormant-path invariance and rejection-triggered recovery."""
from types import SimpleNamespace

import numpy as np

from wd_spectra._hot_recovery import ProjectedThermalPopulationCorrection
from wd_spectra._hot_recovery import RejectedDirectionFallback
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton


def options():
    return dict(maximum_iterations=20, residual_tolerance=1e-12, step_tolerance=1e-10,
                allow_initial_convergence=False, finite_difference_fallback_step=None,
                broyden_updates=False, merit_function='least-squares')


def direction(x, e, j, radius):
    return NonlinearProposal(np.linalg.solve(j, -e.residual))


def test_clean_newton_path_is_bit_identical_and_does_not_do_extra_physics():
    calls = []
    def evaluate(x, jacobian):
        calls.append((x.copy(), jacobian))
        return NonlinearEvaluation(x-np.array([.1, .2]), np.eye(2) if jacobian else None, None)
    original = solve_trust_region_newton(np.zeros(2), evaluate, step_builder=direction, **options())
    original_calls = calls.copy()
    calls.clear()
    gate = RejectedDirectionFallback(ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10))
    candidate = solve_trust_region_newton(np.zeros(2), evaluate,
        step_builder=gate.direction, iteration_correction=gate.correction,
        trial_projector=gate.project, rejected_step_handoff=gate.rejected, **options())
    assert original.converged and candidate.converged
    assert not gate.active
    assert gate.proposals.extra_residual_evaluations == 0
    np.testing.assert_array_equal(candidate.state, original.state)
    assert len(calls) == len(original_calls)
    for (x, jacobian), (expected, expected_jacobian) in zip(calls, original_calls):
        np.testing.assert_array_equal(x, expected)
        assert jacobian == expected_jacobian
    assert candidate.history == original.history


def test_two_rejections_activate_recovery_without_certifying_the_material_step():
    def evaluate(x, jacobian):
        t, p = x
        return NonlinearEvaluation(np.array([1e-8*(t-1.), p-10*t*t]),
            np.array([[1e-8, 0.], [-20*t, 1.]]) if jacobian else None,
            (SimpleNamespace(n_depth=1), None, {'nlte_continuation_fraction': 1.}))
    proposals = ProjectedThermalPopulationCorrection(evaluate, direction, 1e-10)
    gate = RejectedDirectionFallback(proposals)
    kwargs = options()
    kwargs['maximum_iterations'] = 4
    result = solve_trust_region_newton(np.zeros(2), evaluate,
        step_builder=gate.direction, iteration_correction=gate.correction,
        trial_projector=gate.project, rejected_step_handoff=gate.rejected, **kwargs)
    assert gate.active
    assert gate.rejected_directions >= 2
    material = [i for i, record in enumerate(result.history) if record.step_kind == 'material-correction']
    assert material
    first = material[0]
    assert result.history[first].jacobian_recomputed
    assert result.history[first].unrestricted_maximum_step is None
    assert result.history[first+1].step_kind == 'newton'
    assert not result.converged
