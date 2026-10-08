"""Opt-in exact-tangent/controller contracts; legacy defaults stay covered."""
import numpy as np
import pytest
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton


def test_fresh_analytic_jacobian_is_not_overwritten_by_previous_secant():
    seen=[]
    def evaluate(x,need):
        return NonlinearEvaluation(x**3-1,np.diag(3*x*x) if need else None,{})
    def propose(x,ev,j,radius):
        seen.append((j.copy(),np.diag(3*x*x)))
        return np.linalg.solve(j,-ev.residual)
    solve_trust_region_newton(np.array([2.,1.]),evaluate,step_builder=propose,
        maximum_iterations=2,initial_trust_radius=1.,maximum_trust_radius=1.,
        jacobian_refresh_interval=1,broyden_updates=False)
    assert len(seen)==2
    for actual,expected in seen:np.testing.assert_array_equal(actual,expected)


def test_rejected_trials_do_not_modify_exact_tangent_policy():
    matrices=[]
    def evaluate(x,need):return NonlinearEvaluation(x-1,np.eye(2) if need else None,{})
    def propose(x,ev,j,radius):
        matrices.append(j.copy());return np.array([-radius,0.])
    result=solve_trust_region_newton(np.array([0.,1.]),evaluate,step_builder=propose,
        maximum_iterations=3,broyden_updates=False,finite_difference_fallback_step=None)
    assert not result.converged
    assert result.diagnostics.rejected_trial_evaluations>0
    for actual in matrices:np.testing.assert_array_equal(actual,np.eye(2))


def test_bounded_least_squares_acceptance_uses_its_own_objective():
    start=np.array([1.,0.,0.,0.]);end=np.array([.7,.49,.49,.49])
    def evaluate(x,need):
        jac=np.eye(4);jac[:,0]=end-start
        return NonlinearEvaluation(start+x[0]*(end-start),jac if need else None,{})
    options=dict(maximum_iterations=1,initial_trust_radius=1.,maximum_trust_radius=1.,
        step_builder=lambda *args:np.array([1.,0.,0.,0.]),broyden_updates=False)
    legacy=solve_trust_region_newton(np.zeros(4),evaluate,**options)
    squared=solve_trust_region_newton(np.zeros(4),evaluate,merit_function='least-squares',**options)
    assert legacy.history[0].line_search_factor==1.
    assert squared.history[0].line_search_factor==.5
    assert np.mean(squared.evaluation.residual**2)<np.mean(start**2)
    assert not squared.converged  # Physical max-residual test remains independent.


@pytest.mark.parametrize('size,limited,expected',[(.04,False,.06),(.04,True,.04),(.001,False,.04)])
def test_radius_growth_requires_unlimited_boundary_contact(size,limited,expected):
    def evaluate(x,need):return NonlinearEvaluation(x-1,np.eye(2) if need else None,{})
    result=solve_trust_region_newton(np.array([0.,1.]),evaluate,maximum_iterations=1,
        merit_function='least-squares',trust_update='model-agreement',broyden_updates=False,
        step_builder=lambda *args:NonlinearProposal(np.array([size,0.]),limited))
    assert result.history[0].trust_radius==pytest.approx(expected)
    assert result.history[0].proposal_limited is limited


@pytest.mark.parametrize('slope',[-1.,100.])
def test_bad_linear_model_cannot_grow_radius(slope):
    def evaluate(x,need):return NonlinearEvaluation(x-1,slope*np.eye(2) if need else None,{})
    result=solve_trust_region_newton(np.array([0.,1.]),evaluate,maximum_iterations=1,
        merit_function='least-squares',trust_update='model-agreement',broyden_updates=False,
        step_builder=lambda *args:np.array([.005,0.]))
    assert result.history[0].trust_radius==.02
    if slope<0:assert result.history[0].model_agreement is None
    else:assert result.history[0].model_agreement<.25


@pytest.mark.parametrize('maximum_iterations',[1,2])
def test_small_limited_accepted_step_cannot_certify_stationarity(maximum_iterations):
    def evaluate(x,need):return NonlinearEvaluation(x,np.eye(2) if need else None,{})
    result=solve_trust_region_newton(np.array([1e-4,0.]),evaluate,
        maximum_iterations=maximum_iterations,allow_initial_convergence=False,
        step_builder=lambda *args:NonlinearProposal(np.array([-1e-8,0.]),True))
    assert not result.converged
    assert result.history and all(r.proposal_limited for r in result.history)


@pytest.mark.parametrize('maximum_iterations', [1, 2])
def test_limited_physical_certificate_is_consistent_at_iteration_limit(maximum_iterations):
    def evaluate(x, need):
        return NonlinearEvaluation(x, np.eye(2) if need else None, {})
    options = dict(
        maximum_iterations=maximum_iterations,
        allow_initial_convergence=False,
        step_builder=lambda *args: NonlinearProposal(np.array([-1e-8, 0.]), True),
        converge_when_proposal_limited=True,
    )
    accepted = solve_trust_region_newton(
        np.array([1e-4, 0.]), evaluate,
        convergence_test=lambda *args: True, **options,
    )
    assert accepted.converged
    assert accepted.iterations == 1
    assert accepted.history[0].proposal_limited
    refused = solve_trust_region_newton(
        np.array([1e-4, 0.]), evaluate,
        convergence_test=lambda *args: False, **options,
    )
    assert not refused.converged


@pytest.mark.parametrize('settings',[dict(merit_function='invalid'),
    dict(trust_update='invalid'),dict(broyden_updates=1)])
def test_invalid_policies_fail_before_evaluation(settings):
    def forbidden(*args):raise AssertionError('invalid policy evaluated')
    with pytest.raises(ValueError):solve_trust_region_newton(np.zeros(2),forbidden,**settings)


def test_auxiliary_boundary_contact_grows_radius_without_changing_physical_step():
    def evaluate(x, need):
        return NonlinearEvaluation(x-np.array([0., 1.]), np.eye(2) if need else None, {})
    options = dict(maximum_iterations=1, merit_function='least-squares',
        trust_update='model-agreement', broyden_updates=False,
        step_measure=lambda a,b: abs(b[0]-a[0]),
        step_builder=lambda *args: np.array([0., .04]))
    legacy = solve_trust_region_newton(np.zeros(2), evaluate, **options)
    fixed = solve_trust_region_newton(np.zeros(2), evaluate,
        trust_step_measure=lambda a,b: float(np.max(abs(b-a))), **options)
    np.testing.assert_array_equal(fixed.state, legacy.state)
    assert fixed.history[0].maximum_step == legacy.history[0].maximum_step == 0.
    assert fixed.history[0].trust_radius == pytest.approx(.06)
    assert legacy.history[0].trust_radius == .04
    assert not fixed.converged


def test_native_geometry_clips_auxiliary_proposal_and_projected_trial():
    def evaluate(x, need):
        return NonlinearEvaluation(x-np.array([0., 1.]), np.eye(2) if need else None, {})
    options = dict(maximum_iterations=1, trust_update='model-agreement',
        broyden_updates=False, step_measure=lambda a,b: abs(b[0]-a[0]),
        trust_step_measure=lambda a,b: float(np.max(abs(b-a))),
        step_builder=lambda *args: np.array([0., 1.]))
    result = solve_trust_region_newton(np.zeros(2), evaluate, **options)
    assert result.state[1] == pytest.approx(.04)
    projected = solve_trust_region_newton(np.zeros(2), evaluate,
        trial_projector=lambda old,new: 2*new, **options)
    assert projected.state[1] == pytest.approx(.04)
    assert projected.diagnostics.infeasible_trial_rejections == 1


@pytest.mark.parametrize('invalid', [-1., np.inf, np.nan])
def test_invalid_native_step_geometry_rejected(invalid):
    def evaluate(x, need):
        return NonlinearEvaluation(x-1, np.eye(2) if need else None, {})
    with pytest.raises(ValueError, match='trust step measure'):
        solve_trust_region_newton(np.zeros(2), evaluate,
            trust_step_measure=lambda *args: invalid)


def test_structured_model_controls_prediction_not_physical_acceptance():
    def evaluate(x, need):
        return NonlinearEvaluation(x*x-1, np.diag(2*x) if need else None, {})
    def proposal(x, ev, j, radius):
        return NonlinearProposal(np.array([radius, 0.]),
            residual_model=lambda displacement: (x+displacement)**2-1)
    result = solve_trust_region_newton(np.array([.5, 1.]), evaluate,
        maximum_iterations=1, step_builder=proposal, trust_update='model-agreement',
        merit_function='least-squares', broyden_updates=False)
    assert result.history[0].model_agreement == pytest.approx(1.)
    assert result.history[0].trust_radius == pytest.approx(.06)


def test_optimistic_model_cannot_accept_an_actual_uphill_trial():
    def evaluate(x, need):
        return NonlinearEvaluation(x-1, np.eye(2) if need else None, {})
    def proposal(x, ev, j, radius):
        return NonlinearProposal(np.array([-radius, 0.]),
            residual_model=lambda displacement: ev.residual-displacement)
    result = solve_trust_region_newton(np.array([0., 1.]), evaluate,
        maximum_iterations=1, step_builder=proposal, trust_update='model-agreement',
        merit_function='least-squares', broyden_updates=False)
    np.testing.assert_array_equal(result.state, [0., 1.])
    assert not result.converged and not result.history


@pytest.mark.parametrize('bad', [lambda d: np.zeros(2), lambda d: np.zeros(3), 4])
def test_structured_model_must_match_anchor(bad):
    def evaluate(x, need):
        return NonlinearEvaluation(x-1, np.eye(2) if need else None, {})
    with pytest.raises(ValueError, match='proposal residual model'):
        solve_trust_region_newton(np.zeros(2), evaluate, step_builder=lambda *args:
            NonlinearProposal(np.ones(2)*.04, residual_model=bad))
