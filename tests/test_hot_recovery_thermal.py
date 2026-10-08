"""Synthetic checks for the isolated bracketed thermal experiment."""
from types import SimpleNamespace

import numpy as np
import pytest

import wd_spectra._hot_recovery as module
from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, RecoverableEvaluationError


def experiment(monkeypatch, *, radiation_root_shift=0., thermal_coupling=0., correction_type=None):
    origin = np.log(25000.)
    anchor = np.r_[np.full(3, origin), np.zeros(3)]
    def heat(q):
        # A true root at -.22 and a nonzero turning point near +.02.
        return -(q+.22)*((q-.02)**2+.0001)
    def atmosphere(t):
        return SimpleNamespace(n_depth=3, temperature=np.exp(t),
                               rosseland_optical_depth=np.array([1e-4, 1e-3, 1.]))
    def coefficients(a, wave, population, **kwargs):
        h = heat(np.log(a.temperature[1])-origin)
        absorption = np.ones((2, 3))
        absorption[:, 1] += h
        return SimpleNamespace(true_absorption=absorption, thermal_emissivity=np.ones((2, 3)))
    def prepare(t):
        a = atmosphere(t)
        return a, .1*(t-origin), None, None
    def evaluate(x, jacobian):
        t = x[:3]-origin
        residual = np.r_[t[0]+thermal_coupling*t[1], heat(t[1]-radiation_root_shift),
                         t[2], x[3:]-.1*t]
        return NonlinearEvaluation(residual, None, (atmosphere(x[:3]), x[3:],
                                                   {'nlte_continuation_fraction': 1.}))
    eq = SimpleNamespace(fixed_temperature=False, nlte_fraction=1., wave=np.array([1., 2.]),
                         model=SimpleNamespace(transfer_coefficients=coefficients, n_angle=3),
                         prepare=prepare, prepared={}, coordinates=lambda p: p,
                         rate_residual=lambda x, a, ref, groups, mean: (ref, None))
    monkeypatch.setattr(module, 'transfer_field', lambda *a, **k: (
        None, SimpleNamespace(mean_intensity=np.ones((2, 3))), None))
    j = np.eye(6)
    j[0, 1] = thermal_coupling
    j[3:, :3] = -.1*np.eye(3)
    correction = (correction_type or module.BracketedThermalPopulationCorrection)(
        evaluate, lambda *a: NonlinearProposal(np.array([0., .1, 0., 0., 0., 0.])), 3e-4, eq)
    correction.direction(anchor, evaluate(anchor, False), j, .12)
    return correction, anchor, evaluate


def test_bracket_escapes_nonzero_turning_point_with_full_merit_reduction(monkeypatch):
    correction, x, evaluate = experiment(monkeypatch)
    original = x.copy()
    before = evaluate(x, False)
    proposal = correction.correction(x, before)
    trial = proposal.trial_state(1.)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(trial[1]-x[1], -.22, atol=1e-7)
    np.testing.assert_array_equal(trial[[0, 2]], x[[0, 2]])
    assert max(abs(trial[:3]-x[:3])) <= .35
    assert np.linalg.norm(evaluate(trial, False).residual) < np.linalg.norm(before.residual)*1e-4
    assert correction.thermal_probes > 7
    assert correction.extra_residual_evaluations >= 1


def test_bracket_backtracking_rebuilds_dependent_populations(monkeypatch):
    correction, x, evaluate = experiment(monkeypatch)
    proposal = correction.correction(x, evaluate(x, False))
    trial = proposal.trial_state(.5)
    np.testing.assert_allclose(trial[1]-x[1], -.11, atol=1e-7)
    np.testing.assert_allclose(evaluate(trial, False).residual[3:], 0., atol=1e-14)


def test_full_transfer_refines_the_frozen_radiation_root(monkeypatch):
    correction, x, evaluate = experiment(monkeypatch, radiation_root_shift=.06)
    proposal = correction.correction(x, evaluate(x, False))
    trial = proposal.trial_state(1.)
    np.testing.assert_allclose(trial[1]-x[1], -.16, atol=1e-6)
    assert np.linalg.norm(evaluate(trial, False).residual) < 1e-8


@pytest.mark.parametrize('attribute,value', [('fixed_temperature', True), ('nlte_fraction', .8)])
def test_nonthermal_or_intermediate_stage_does_not_scan(monkeypatch, attribute, value):
    correction, x, evaluate = experiment(monkeypatch)
    monkeypatch.setattr(module.ProjectedThermalPopulationCorrection, 'correction', lambda *a: None)
    setattr(correction.equations, attribute, value)
    assert correction.correction(x, evaluate(x, False)) is None
    assert correction.thermal_probes == 0


def test_exhausted_local_root_is_recoverable():
    with pytest.raises(RecoverableEvaluationError, match='did not settle'):
        module._bracket_root(lambda x: x**3-2., 0., 2., maxiter=1)


def test_unexpected_physics_error_is_not_hidden():
    def broken(x):
        raise RuntimeError('unexpected evaluator defect')
    with pytest.raises(RuntimeError, match='unexpected evaluator defect'):
        module._bracket_root(broken, 0., 2.)


def test_singular_population_restoration_is_recoverable(monkeypatch):
    correction, x, evaluate = experiment(monkeypatch)
    def singular(*args, **kwargs):
        raise module.LinAlgWarning('singular population block')
    monkeypatch.setattr(module, 'lu_factor', singular)
    with pytest.raises(RecoverableEvaluationError, match='Singular bracketed population tangent'):
        correction.correction(x, evaluate(x, False))


def test_neighbor_compensation_makes_the_full_root_reduce_coupled_merit(monkeypatch):
    from wd_spectra._hot_recovery import CoupledBracketedThermalPopulationCorrection
    bare, x, evaluate = experiment(monkeypatch, thermal_coupling=.02)
    before = evaluate(x, False)
    isolated = bare.correction(x, before).trial_state(1.)
    assert np.linalg.norm(evaluate(isolated, False).residual) > np.linalg.norm(before.residual)
    corrected, x, evaluate = experiment(monkeypatch, thermal_coupling=.02,
        correction_type=CoupledBracketedThermalPopulationCorrection)
    original = x.copy()
    trial = corrected.correction(x, evaluate(x, False)).trial_state(1.)
    np.testing.assert_allclose(trial[1]-x[1], -.22, atol=1e-7)
    np.testing.assert_allclose(trial[0]-x[0], .0044, atol=1e-7)
    np.testing.assert_array_equal(x, original)
    assert np.linalg.norm(evaluate(trial, False).residual) < 1e-8
    assert corrected.neighbor_adjustments == 1
    assert max(abs(trial[:3]-x[:3])) <= .35


def test_inadmissible_neighbor_correction_keeps_the_original_proposal(monkeypatch):
    from wd_spectra._hot_recovery import CoupledBracketedThermalPopulationCorrection
    correction, x, evaluate = experiment(monkeypatch, thermal_coupling=.02,
        correction_type=CoupledBracketedThermalPopulationCorrection)
    def guarded(state, jacobian):
        if state[0]-x[0] > .001:
            raise RecoverableEvaluationError('Synthetic neighbor domain')
        return evaluate(state, jacobian)
    correction.evaluate = guarded
    trial = correction.correction(x, evaluate(x, False)).trial_state(1.)
    assert trial[0] == x[0]
    np.testing.assert_allclose(trial[1]-x[1], -.22, atol=1e-7)
    assert correction.neighbor_adjustments == 0
