"""Globalization checks on an independently specified curved material law."""
from types import SimpleNamespace
import numpy as np
from wd_spectra.nonlinear import (NonlinearEvaluation,NonlinearProposal,
                                RecoverableEvaluationError,solve_trust_region_newton)
from wd_spectra._hot_recovery import ThermalPopulationCorrection
from wd_spectra._hot_recovery import ProjectedThermalPopulationCorrection


def test_curved_population_trial_is_bounded_and_reduces_the_actual_residual():
    # Thermal balance is weak compared with curvature in the population law.
    # A straight Newton trial loses merit even though its tangent is exact.
    def evaluate(x,jacobian):
        t,p=x
        return NonlinearEvaluation(np.array([1e-3*(t-1),p-10*t*t]),
            np.array([[1e-3,0],[-20*t,1]]) if jacobian else None,
            (SimpleNamespace(n_depth=1),None,{'nlte_continuation_fraction':1.}))
    def direction(x,e,j,radius):return NonlinearProposal(np.linalg.solve(j,-e.residual))
    x=np.zeros(2);e=evaluate(x,True)
    correction=ThermalPopulationCorrection(evaluate,direction,1e-8)
    raw=correction.direction(x,e,e.jacobian,.04).direction
    straight=x+raw*.04/max(abs(raw))
    assert np.linalg.norm(evaluate(straight,False).residual)>np.linalg.norm(e.residual)
    repaired=correction.correction(x,e).trial_state(1.)
    assert max(abs(repaired-x))<=.04
    assert repaired[0]>.01
    assert abs(repaired[1]-10*repaired[0]**2)<1e-14
    assert np.linalg.norm(evaluate(repaired,False).residual)<np.linalg.norm(e.residual)
    np.testing.assert_array_equal(x,np.zeros(2))
    result=solve_trust_region_newton(x,evaluate,step_builder=correction.direction,
        iteration_correction=correction.correction,maximum_iterations=2,
        residual_tolerance=1e-8,step_tolerance=1e-8,allow_initial_convergence=False,
        finite_difference_fallback_step=None,broyden_updates=False,merit_function='least-squares')
    assert result.history[0].step_kind=='material-correction'
    assert result.history[0].jacobian_recomputed
    assert result.history[0].unrestricted_maximum_step is None
    assert result.history[1].step_kind=='newton'
    assert not result.converged
    assert correction.extra_residual_evaluations>0


def test_singular_population_block_defers_to_global_newton():
    e=NonlinearEvaluation(np.ones(2),np.zeros((2,2)),
        (SimpleNamespace(n_depth=1),None,{'nlte_continuation_fraction':1.}))
    correction=ThermalPopulationCorrection(lambda *args:e,
        lambda *args:NonlinearProposal(np.ones(2)),1e-8)
    x=np.zeros(2)
    correction.direction(x,e,e.jacobian,.04)
    assert correction.correction(x,e) is None


def test_newton_population_projection_preserves_temperature_and_trust_bound():
    def evaluate(x,jacobian):
        t,p=x
        return NonlinearEvaluation(np.array([1e-3*(t-1),p-10*t*t]),
            np.array([[1e-3,0],[-20*t,1]]) if jacobian else None,
            (SimpleNamespace(n_depth=1),None,{'nlte_continuation_fraction':1.}))
    direction=lambda x,e,j,radius:NonlinearProposal(np.linalg.solve(j,-e.residual))
    correction=ProjectedThermalPopulationCorrection(evaluate,direction,1e-8)
    old=np.zeros(2);e=evaluate(old,True)
    correction.direction(old,e,e.jacobian,.04)
    trial=np.array([.04,0.]);repaired=correction.project(old,trial)
    assert repaired[0]==trial[0]
    assert max(abs(repaired-old))<=.04
    assert np.linalg.norm(evaluate(repaired,False).residual)<np.linalg.norm(e.residual)
    np.testing.assert_array_equal(old,np.zeros(2))
    np.testing.assert_array_equal(trial,np.array([.04,0.]))


def test_inadmissible_projected_newton_trial_is_backtracked_by_the_shared_driver():
    def evaluate(x,jacobian):
        t,p=x
        if t>.02:raise RecoverableEvaluationError('Synthetic material-domain boundary')
        return NonlinearEvaluation(np.array([1e-3*(t-1),p-10*t*t]),
            np.array([[1e-3,0],[-20*t,1]]) if jacobian else None,
            (SimpleNamespace(n_depth=1),None,{'nlte_continuation_fraction':1.}))
    correction=ProjectedThermalPopulationCorrection(evaluate,
        lambda x,e,j,radius:NonlinearProposal(np.linalg.solve(j,-e.residual)),1e-8)
    result=solve_trust_region_newton(np.zeros(2),evaluate,step_builder=correction.direction,
        trial_projector=correction.project,maximum_iterations=1,initial_trust_radius=.04,
        residual_tolerance=1e-8,step_tolerance=1e-8,allow_initial_convergence=False,
        finite_difference_fallback_step=None,broyden_updates=False,merit_function='least-squares')
    assert result.diagnostics.infeasible_trial_rejections==1
    assert result.history[0].line_search_factor==.5
    assert result.state[0]==.02
    assert not result.converged


def test_projection_at_log_temperature_trust_boundary_allows_subtraction_roundoff():
    def evaluate(x,jacobian):
        t=x[0]-10.;p=x[1]
        return NonlinearEvaluation(np.array([1e-3*(t-1),p-10*t*t]),
            np.array([[1e-3,0],[-20*t,1]]) if jacobian else None,
            (SimpleNamespace(n_depth=1),None,{'nlte_continuation_fraction':1.}))
    correction=ProjectedThermalPopulationCorrection(evaluate,
        lambda x,e,j,radius:NonlinearProposal(np.linalg.solve(j,-e.residual)),1e-8)
    old=np.array([10.,0.]);trial=np.array([10.06,0.]);e=evaluate(old,True)
    assert trial[0]-old[0]>.06  # subtraction is just outside the nominal box
    correction.direction(old,e,e.jacobian,.06)
    repaired=correction.project(old,trial)
    assert repaired[0]==trial[0]
    assert abs(evaluate(repaired,False).residual[1])<1e-14
    assert max(abs(repaired-old))<.06+1e-12
