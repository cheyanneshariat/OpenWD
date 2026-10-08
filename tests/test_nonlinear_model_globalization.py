import numpy as np
import pytest
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton


def test_recompute_smaller_box_without_rebuilding_identical_tangent():
    calls=[]; radii=[]
    def evaluate(x, need):
        calls.append((x.copy(), need))
        return NonlinearEvaluation(x-1, np.eye(2) if need else None, {})
    def propose(x, ev, j, radius):
        radii.append(radius)
        if radius > .02:
            # Full trial optimistically predicted downhill; all damped trials
            # predicted uphill. Actual full trial also uphill.
            def model(d):
                return ev.residual-d if abs(d[0]) > .03 else ev.residual+d
            return NonlinearProposal(np.array([-radius, 0.]), residual_model=model)
        return NonlinearProposal(np.array([radius, 0.]), residual_model=lambda d: ev.residual+d)
    result=solve_trust_region_newton(np.array([0.,1.]),evaluate,
        step_builder=propose, maximum_iterations=2, broyden_updates=False,
        finite_difference_fallback_step=None, nonlinear_model_globalization=True,
        jacobian_refresh_interval=1)
    assert radii == pytest.approx([.04,.01])
    assert result.state[0] == pytest.approx(.01)
    assert result.diagnostics.analytic_restarts == 0
    assert result.diagnostics.model_screened_trials == 6
    assert result.diagnostics.rejected_trial_evaluations == 1
    assert len([x for x,need in calls if not need]) == 3  # Includes initial gate check.
    assert not result.converged


def test_screening_never_bypasses_physical_acceptance_guard():
    def evaluate(x,need):
        return NonlinearEvaluation(x-1,np.eye(2) if need else None,{})
    result=solve_trust_region_newton(np.array([0.,1.]),evaluate,
        step_builder=lambda x,ev,j,r: NonlinearProposal(np.array([r,0.]),
            residual_model=lambda d:ev.residual+d),
        nonlinear_model_globalization=True,broyden_updates=False,jacobian_refresh_interval=1,
        maximum_iterations=2,acceptance_test=lambda old,new:False)
    np.testing.assert_array_equal(result.state,[0.,1.])
    assert not result.converged and not result.history
    assert result.diagnostics.acceptance_test_rejections>0


@pytest.mark.parametrize('options',[dict(nonlinear_model_globalization=1),
    dict(nonlinear_model_globalization=True,broyden_updates=True),
    dict(nonlinear_model_globalization=True,broyden_updates=False,jacobian_refresh_interval=4)])
def test_invalid_policy(options):
    with pytest.raises(ValueError):
        solve_trust_region_newton(np.zeros(2),lambda *a:None,**options)
