from importlib.resources import files

import numpy as np
import pytest

from wd_spectra.spectrum import StokesSpectrum
from wd_spectra.validation.magnetic_da import score_dah_circular_polarization

WAVE = np.arange(3000.0, 8000.0, 1.0)
GRID = np.arange(3700.0, 6600.0, 3.3)
SIGMA = np.full(GRID.size, 0.005)


def v_over_i(wave):
    return 0.1 * np.exp(-0.5 * ((wave - 5700.0) / 150.0) ** 2) - 0.05 * np.exp(
        -0.5 * ((wave - 4500.0) / 100.0) ** 2
    )


def model(sign=1.0):
    stokes_i = WAVE**-2.0
    zero = np.zeros_like(WAVE)
    return StokesSpectrum(WAVE, stokes_i, {}, zero, zero, sign * stokes_i * v_over_i(WAVE))


def observed(shift=0.0):
    return v_over_i(GRID - shift) + np.random.default_rng(1).normal(0.0, 0.005, GRID.size)


def score(data, spectrum, convention="optical"):
    return score_dah_circular_polarization(
        GRID, data, SIGMA, spectrum, observed_convention=convention, resolution_fwhm_angstrom=25.0
    )


def test_noise_only_scores_unity_scale_and_no_shift():
    result = score(observed(), model())
    assert result["chi2_v"] == pytest.approx(1.0, abs=0.15)
    assert result["v_scale"] == pytest.approx(1.0, abs=0.02)
    assert abs(result["shift_angstrom"]) < 1.0


def test_negated_model_v_gives_negative_scale():
    assert score(observed(), model(-1.0))["v_scale"] == pytest.approx(-1.0, abs=0.02)


def test_shift_is_recovered_within_one_pixel():
    assert score(observed(10.0), model())["shift_angstrom"] == pytest.approx(10.0, abs=3.3)


def test_iau_and_optical_inputs_agree():
    data = observed()
    assert score(-data, model(), "iau") == score(data, model(), "optical")


def test_convention_is_required():
    with pytest.raises(TypeError):
        score_dah_circular_polarization(GRID, observed(), SIGMA, model())


def test_bundled_re_j0317_cyclotron_peak_is_negative_in_optical_convention():
    path = files("wd_spectra").joinpath("data/validation/dah/re-j0317-853-vi.npz")
    with np.load(str(path)) as saved:
        wave, data = saved["wavelength_angstrom"], saved["v_over_i"]
    assert np.mean(data[(wave > 5650.0) & (wave < 5750.0)]) < -0.1
