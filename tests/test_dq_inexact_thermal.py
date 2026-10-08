from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra._dq.inexact_thermal import solve_inexact,predicts_budget_exhaustion
from wd_spectra.nonlinear import NonlinearEvaluation,NonlinearProposal,solve_trust_region_newton


def evaluation(residual,compatibility=0.):
    return SimpleNamespace(residual=np.asarray(residual),payload=SimpleNamespace(
        payload={'dq_augmented_flux_compatibility':compatibility}))


def test_transient_stops_on_original_l2_and_compatibility_not_number_of_steps():
    settings={}
    seen=[]
    def solver(initial,evaluate,**kwargs):
        settings.update(kwargs)
        callback=kwargs['callback']
        record=SimpleNamespace(maximum_step=.04,proposal_limited=True)
        callback(record,initial,evaluation([.3,.3],.003))
        callback(record,initial,evaluation([.4,.4]))
        # Equality does not satisfy the strict cost-ratio acceptance.
        callback(record,initial,evaluation([.3,.4]))
        callback(record,initial,evaluation([.3,.3]))
        raise AssertionError('Transient acceptance must stop further expensive updates')
    first=evaluation([3.,4.])
    result=solve_inexact(solver,np.zeros(2),lambda q,need:first,
        compatibility_tolerance=.002,
        maximum_iterations=2,maximum_trust_radius=.04,convergence_test=lambda *a:False,
        callback=lambda *a:seen.append(a))
    assert settings['maximum_iterations']==8
    assert result.transient_accepted
    assert not hasattr(result,'converged')
    assert len(result.history)==len(seen)==4
    assert all(record.proposal_limited for record in result.history)


@pytest.mark.parametrize('residual', [[0.,0.],[float('nan'),1.],[float('inf'),1.]])
def test_invalid_transient_norm_is_not_replaced_by_a_fallback(residual):
    with pytest.raises(ValueError):
        solve_inexact(None,np.zeros(2),lambda q,need:evaluation(residual),
            maximum_trust_radius=.04,compatibility_tolerance=.002)


@pytest.mark.parametrize('tolerance', [0.,-1.,float('nan'),float('inf')])
def test_invalid_compatibility_tolerance_cannot_bypass_physical_check(tolerance):
    with pytest.raises(ValueError,match='compatibility tolerance'):
        solve_inexact(None,np.zeros(2),lambda q,need:evaluation([1.,1.]),
            maximum_trust_radius=.04,compatibility_tolerance=tolerance)


def test_real_shared_newton_can_finish_a_step_needing_more_than_two_updates():
    physical=SimpleNamespace(payload={'dq_augmented_flux_compatibility':0.})
    def evaluate(q,need):
        exponential=np.exp(q[0])
        return NonlinearEvaluation(np.array([exponential-1e4,q[1]]),
            np.diag([exponential,1.]) if need else None,physical)
    initial=np.array([8.,0.]);first=np.linalg.norm(evaluate(initial,True).residual)
    def bounded(q,e,j,radius):
        step=np.linalg.solve(j,-e.residual)
        step*=min(1.,radius/max(abs(step)))
        return NonlinearProposal(step,limited=True)
    result=solve_inexact(solve_trust_region_newton,initial,evaluate,
        compatibility_tolerance=.002,maximum_iterations=2,initial_trust_radius=.4,
        maximum_trust_radius=.4,broyden_updates=False,jacobian_refresh_interval=1,
        finite_difference_fallback_step=None,allow_initial_convergence=False,
        merit_function='least-squares',trust_update='model-agreement',step_builder=bounded)
    assert result.transient_accepted
    assert 2<len(result.history)<8
    assert all(record.proposal_limited for record in result.history)
    assert np.linalg.norm(result.evaluation.residual)<.1*first


def test_exhausted_inner_budget_is_not_transient_acceptance():
    first=evaluation([3.,4.])
    def exhausted(state,evaluate,**settings):
        return SimpleNamespace(state=state,evaluation=first,history=())
    result=solve_inexact(exhausted,np.zeros(2),lambda *a:first,compatibility_tolerance=.002)
    assert not result.transient_accepted
    assert not hasattr(result,'converged')


def test_prediction_retains_measured_j1225_progress_and_rejects_j0804_plateau():
    norms=[.32745,.26885,.15921,.08709]
    assert not any(predicts_budget_exhaustion(a,b,8-i,.1)
        for i,(a,b) in enumerate(zip(norms[:-1],norms[1:]),2))
    assert predicts_budget_exhaustion(.53525,.46589,6,.1)
    assert not predicts_budget_exhaustion(.28226,.09212,6,.1)


def test_prediction_is_only_an_early_failure_not_a_convergence_certificate():
    def solver(state,evaluate,**settings):
        record=SimpleNamespace(maximum_step=.01,proposal_limited=True)
        settings['callback'](record,state,evaluation([.5,.5]))
        settings['callback'](record,state,evaluation([.49,.49]))
        raise AssertionError('Unproductive inner updates should return early')
    result=solve_inexact(solver,np.zeros(2),lambda *a:evaluation([1.,1.]),
        compatibility_tolerance=.002,predict_exhaustion=True)
    assert not result.transient_accepted
    assert result.reason=='predicted-inner-budget-exhaustion'
    assert not hasattr(result,'converged')


@pytest.mark.parametrize('previous,current,remaining,target',[(0.,1.,2,.1),
    (1.,float('nan'),2,.1),(1.,.5,-1,.1),(1.,.5,2,0.)])
def test_invalid_prediction_input_is_rejected(previous,current,remaining,target):
    with pytest.raises(ValueError):predicts_budget_exhaustion(previous,current,remaining,target)
