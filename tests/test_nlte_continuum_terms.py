"""Continuum batching retains the original complete-grid numerical result."""
import numpy as np
import pytest

from wd_spectra.multilevel_nlte import _ContinuumTransferProblem, _nlte_continuum_terms


@pytest.mark.parametrize('wavelengths', [1, 256, 513])
@pytest.mark.parametrize('levels', [3, 20, 32])
@pytest.mark.parametrize('signed', [False, True])
def test_batched_continuum_matches_complete_grid(wavelengths, levels, signed):
    rng = np.random.default_rng(81)
    shape = (wavelengths, 7)
    coefficient = np.exp(rng.normal(size=shape+(levels,)))
    background = np.full(shape, .02)
    exp_minus = rng.uniform(.01, .9, size=shape)
    planck = np.exp(rng.normal(size=shape))
    spontaneous = planck*(1-exp_minus)/exp_minus
    bound = rng.uniform(1., 2., size=(7, levels))
    ion = rng.uniform(.5, 1., size=7)
    if signed:
        bound[:, 0] = .001
        ion *= 10.
    problem = _ContinuumTransferProblem(np.arange(wavelengths)+1., background,
        coefficient, exp_minus, spontaneous, np.ones(shape), planck)
    expected_absorption = background + np.sum(coefficient*(bound[None]-
        ion[None, :, None]*exp_minus[:, :, None]), axis=2)
    expected_emissivity = background*planck + np.sum(coefficient*
        ion[None, :, None]*exp_minus[:, :, None], axis=2)*spontaneous
    absorption, emissivity = _nlte_continuum_terms(problem, bound, ion,
        allow_signed_absorption=signed)
    np.testing.assert_array_equal(absorption, expected_absorption)
    np.testing.assert_array_equal(emissivity, expected_emissivity)
    if signed:
        with pytest.raises(RuntimeError, match='non-positive total continuum'):
            _nlte_continuum_terms(problem, bound, ion)
