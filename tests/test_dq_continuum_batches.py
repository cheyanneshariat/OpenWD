"""Wavelength batching changes allocation only, never continuum physics."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from wd_spectra import DQConfig, gray_helium_atmosphere
from wd_spectra.models.common import ModelData
from wd_spectra._dq import base
from wd_spectra._dq.runtime import make_material
from wd_spectra._dq.continuum_batches import helium_continuum_absorption, CONTINUUM_BATCH_SIZE


@pytest.mark.parametrize('dense', [False, True])
def test_continuum_batches_are_bit_identical(tmp_path, monkeypatch, dense):
    fixture = Path(__file__).parent/'data/dq_regressions/j1235-fixed.npz'
    mat = make_material(DQConfig(9347., 8.041, -4.107), ModelData.default(), tmp_path)
    with np.load(fixture) as z:
        seed = gray_helium_atmosphere(9347., 8.041, n_depth=len(z['temperature']))
        a = replace(seed, **{k: z[k].copy() for k in
            ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')})
    a = mat.chemistry(a)[0]
    wave = np.geomspace(1000., 100000., 2*CONTINUUM_BATCH_SIZE+19)
    correction = (mat.dense_continuum.correction(wave, a.temperature,
                  a.helium_lte_state.neutral_he_density) if dense else None)
    original = base.helium_continuum_mass_absorption_coefficient
    expected = original(a, wave, include_electron_scattering=False,
        include_rayleigh_scattering=False, helium_minus_correction=correction)
    sizes = []
    def observe(state, wavelengths, **kwargs):
        sizes.append(len(wavelengths))
        return original(state, wavelengths, **kwargs)
    monkeypatch.setattr(base, 'helium_continuum_mass_absorption_coefficient', observe)
    actual = helium_continuum_absorption(a, wave, correction)
    np.testing.assert_array_equal(actual, expected)
    assert sizes == [CONTINUUM_BATCH_SIZE, CONTINUUM_BATCH_SIZE, 19]
    with pytest.raises(ValueError, match='wavelength-by-depth'):
        helium_continuum_absorption(a, wave, np.ones((len(wave), 1)))
    with pytest.raises(ValueError, match='positive 1D'):
        helium_continuum_absorption(a, np.array([1000., np.nan]))
