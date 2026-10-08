from types import SimpleNamespace
import numpy as np
import pytest
import wd_spectra._dq.dq_thermal_corner_experiment as experiment


def fixture():
    state = np.zeros(3)
    fun = lambda delta:np.array([1.+delta[0], 0., 0.])
    jac = lambda delta:np.diag([1., 0., 0.])
    solved = SimpleNamespace(x=np.zeros(3), cost=.5)
    bounds = (np.array([-.04, -.04, -np.inf]), np.array([.04, .04, np.inf]))
    return state, fun, jac, solved, bounds


def test_existing_successful_thermal_proposal_is_returned_unchanged(monkeypatch):
    state, fun, jac, solved, bounds = fixture()
    # A successful ordinary proposal must not inspect table corners at all.
    monkeypatch.setattr(experiment, 'corner_columns', lambda *a:pytest.fail('corner search'))
    result, info = experiment.repair_proposal(SimpleNamespace(n=2), None, state,
        lambda d:np.array([1. if np.all(d == 0) else .1]), jac,
        SimpleNamespace(x=np.array([.01, 0., 0.])), bounds, .8)
    assert info['mode'] == 'ordinary'
    assert info['corners'] == []
    np.testing.assert_array_equal(result.x, [.01, 0., 0.])


def test_no_eos_leaves_unsuccessful_original_unchanged():
    state, fun, jac, solved, bounds = fixture()
    result, info = experiment.repair_proposal(SimpleNamespace(n=2, eos_trial_owner=None),
        None, state, fun, jac, solved, bounds, .8)
    assert result is solved
    assert info['mode'] == 'ordinary'


@pytest.mark.parametrize('admissible', [True, False])
def test_corner_candidate_still_requires_model_deep_gradient_guard(monkeypatch, admissible):
    state, fun, jac, solved, bounds = fixture()
    fun = lambda d:np.array([1.+20*d[0], 0., 0.])
    improved = SimpleNamespace(x=np.array([-.02, 0., 0.]), cost=.18)
    monkeypatch.setattr(experiment, 'corner_columns', lambda *a:np.array([0]))
    monkeypatch.setattr(experiment, 'refine_corner_subspace', lambda *a,**k:(improved, {}))
    monkeypatch.setattr(experiment, 'search_thermal_pieces', lambda *a,**k:(None, []))
    monkeypatch.setattr(experiment.controller.nm, 'admissible_temperature_gradient',
                        lambda *a:admissible)
    model = SimpleNamespace(evaluate=lambda *a:SimpleNamespace(payload={}))
    result, info = experiment.repair_proposal(SimpleNamespace(n=2), model, state,
        fun, jac, solved, bounds, .8)
    assert result is (improved if admissible else solved)
    assert info['atmosphere_certified'] is False


def test_experimental_patches_restore_after_error():
    original_model = experiment.controller.nm.local_model
    original_solver = experiment.controller.least_squares
    with pytest.raises(RuntimeError):
        with experiment.corner_aware_thermal_proposals(lambda row:None):
            assert experiment.controller.least_squares is not original_solver
            raise RuntimeError('interrupted diagnostic')
    assert experiment.controller.nm.local_model is original_model
    assert experiment.controller.least_squares is original_solver


def test_completed_domain_releases_observer_anchor(monkeypatch):
    import gc
    import weakref
    from wd_spectra._dq import automatic_conditioning as controller

    class Node:
        pass

    def build(system, state, ev):
        model=Node();model.system=system;model.anchor=ev
        system.cycle=system
        return model

    monkeypatch.setattr(controller.nm,'local_model',build)
    with experiment.corner_aware_thermal_proposals(lambda row:None) as release:
        for _ in range(2):
            system=Node();anchor=Node()
            refs=(weakref.ref(system),weakref.ref(anchor))
            model=controller.nm.local_model(system,np.zeros(2),anchor)
            del model,system,anchor
            gc.collect()
            assert all(ref() is not None for ref in refs)
            release();gc.collect()
            assert all(ref() is None for ref in refs)
