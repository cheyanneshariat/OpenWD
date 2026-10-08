"""Atomic failures must reach the common driver's trial-domain safeguard."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest

from wd_spectra._hot_structure import HotEquations
from wd_spectra.nlte_core import NonphysicalPopulationError
from wd_spectra.nonlinear import (
    NonlinearEvaluation, RecoverableEvaluationError, solve_trust_region_newton)
from wd_spectra.atmosphere import gray_helium_atmosphere, gray_hydrogen_helium_atmosphere
from wd_spectra import helium_nlte, multilevel_nlte
from test_hot_nlte import atom


@pytest.mark.parametrize('species', ['helium', 'hydrogen'])
@pytest.mark.parametrize('solution', [-1., np.nan, 1e100])
def test_atomic_population_and_normalization_failures_have_specific_type(atom,monkeypatch,species,solution):
    a=(gray_helium_atmosphere(60000.,8.,n_depth=3) if species=='helium' else
       gray_hydrogen_helium_atmosphere(60000.,8.,2.,n_depth=3))
    def invalid(matrix,rhs):
        return np.full_like(rhs,solution)
    monkeypatch.setattr(np.linalg,'solve',invalid)
    match='normalization' if solution==1e100 else 'non-physical'
    with pytest.raises(NonphysicalPopulationError,match=match):
        if species=='helium':
            helium_nlte.solve_coupled_helium_statistical_equilibrium(a,atom.collision_data,
                maximum_helium_ii_level=3)
        else:
            multilevel_nlte.solve_multilevel_hydrogen_statistical_equilibrium(a,atom.collision_data,
                maximum_level=3)


def error_equations(route,error,*,invalid_initial=False):
    """Inject an atomic failure at each real dispatch route, with a known root.

    Only the expensive transfer/thermal residual is replaced. The hot residual
    exception boundary, preparation/rate dispatch and common driver are real.
    """
    e=HotEquations.__new__(HotEquations)
    a=object();groups=[]
    def bad(*args,**kwargs):raise error('SE failure')
    e.seed=None;e.wave=np.ones(1);e.nd=2;e.fixed_temperature=False;e.prepared={}
    e.model=SimpleNamespace(_rate_state=bad,rebuild_atmosphere=lambda *args:a)
    e.profile_cache=SimpleNamespace(groups=groups,fields=lambda mean:[{}, {}, {}])
    e.rate_cache=SimpleNamespace(atmosphere=a,state=bad) if route=='prepared' else None
    def interior(state,*args):
        if invalid_initial or state[0]>1.5:
            if route=='preparation':
                e.prepare(state)
            else:
                e.rate_residual(state,a,None,groups,np.ones((1,2)))
        return NonlinearEvaluation(state-1.,None,None)
    e._residual=interior
    def evaluate(state,jacobian):
        return replace(e.residual(state),jacobian=.5*np.eye(2) if jacobian else None)
    return evaluate


@pytest.mark.parametrize('route',['preparation','prepared','uncached'])
def test_nonphysical_trial_backtracks_through_hot_boundary(route):
    result=solve_trust_region_newton(np.zeros(2),error_equations(route,NonphysicalPopulationError),
        maximum_iterations=2,residual_tolerance=1e-10,step_tolerance=1.1,
        initial_trust_radius=2.,maximum_trust_radius=2.)
    assert result.converged
    np.testing.assert_allclose(result.state,np.ones(2),rtol=0.,atol=4*np.finfo(float).eps)
    assert result.history[0].line_search_factor==.5
    assert result.diagnostics.infeasible_trial_rejections==1


@pytest.mark.parametrize('route',['preparation','prepared','uncached'])
def test_nonphysical_initial_state_is_still_fatal(route):
    with pytest.raises(RecoverableEvaluationError,match='SE failure') as raised:
        solve_trust_region_newton(np.zeros(2),
            error_equations(route,NonphysicalPopulationError,invalid_initial=True))
    assert isinstance(raised.value.__cause__,NonphysicalPopulationError)


def test_nonphysical_accepted_state_is_still_fatal():
    invalid=error_equations('prepared',NonphysicalPopulationError,invalid_initial=True)
    def evaluate(state,jacobian):
        if jacobian and state[0]>0:
            return invalid(state,jacobian)
        return NonlinearEvaluation(state-1.,np.eye(2) if jacobian else None,None)
    with pytest.raises(RecoverableEvaluationError,match='SE failure'):
        solve_trust_region_newton(np.zeros(2),evaluate,
            initial_trust_radius=.1,maximum_trust_radius=.1)


@pytest.mark.parametrize('error',[RuntimeError,ValueError])
@pytest.mark.parametrize('route',['preparation','prepared','uncached'])
def test_other_trial_errors_are_not_backtracked(route,error):
    with pytest.raises(error,match='SE failure') as raised:
        solve_trust_region_newton(np.zeros(2),error_equations(route,error),
            initial_trust_radius=2.,maximum_trust_radius=2.)
    assert type(raised.value) is error
