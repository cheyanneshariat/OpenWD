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


def test_noise_only_scores_unity_and_statistically_zero_shift():
    result = score(observed(), model())
    assert result["chi2_v"] == pytest.approx(1.0, abs=0.15)
    assert result["v_scale"] == pytest.approx(1.0, abs=0.02)
    # Independent nonlinear least squares, convolving I/V at each trial
    # wavelength, gives -1.18523 A for this fixed noise realization. Its
    # Fisher uncertainty is about 1.01 A; noise need not fit an exact zero.
    assert result["shift_angstrom"] == pytest.approx(-1.18523, abs=0.02)


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


def gaussian(wave, *, centre=5500.0, width=30.0, amplitude=0.1):
    return amplitude * np.exp(-0.5 * ((wave - centre) / width) ** 2)


def constant_i_model(profile):
    zero = np.zeros_like(WAVE)
    return StokesSpectrum(WAVE, np.ones_like(WAVE), {}, zero, zero, profile)


def analytic_score(wave, data, spectrum, sigma=0.001, **kwargs):
    return score_dah_circular_polarization(
        wave, data, np.broadcast_to(sigma, wave.shape), spectrum,
        observed_convention="optical", **kwargs,
    )


def broadened_gaussian(wave, fwhm, **kwargs):
    if fwhm is None:
        return gaussian(wave, **kwargs)
    # Independent Gaussian quadrature in log wavelength, with constant I.
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    sigma_log = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / wave
    samples = wave[:, None] * np.exp(np.sqrt(2.0) * sigma_log[:, None] * nodes)
    return np.sum(weights * gaussian(samples, **kwargs), axis=1) / np.sqrt(np.pi)


@pytest.mark.parametrize("fwhm", [None, 25.0])
@pytest.mark.parametrize("shift", [-50.0, 50.0])
def test_shift_uses_wavelengths_on_an_uneven_grid(shift, fwhm):
    wave = np.concatenate((np.arange(3820.0, 5000.0),
                           np.arange(5000.0, 6000.0, 10.0),
                           np.arange(6000.0, 6950.0)))
    data = broadened_gaussian(wave - shift, fwhm)
    result = analytic_score(wave, data, constant_i_model(gaussian(WAVE)),
                            resolution_fwhm_angstrom=fwhm)
    assert result["shift_angstrom"] == pytest.approx(shift, abs=0.05)


@pytest.mark.parametrize("fwhm", [None, 25.0])
def test_alignment_downweights_a_half_sigma_pixel(fwhm):
    wave = np.arange(3820.0, 6950.0)
    profile = lambda w: gaussian(w, centre=5000.0, width=10.0, amplitude=0.001)
    data = broadened_gaussian(wave, fwhm, centre=5000.0, width=10.0, amplitude=0.001)
    sigma = np.full_like(wave, 0.0001)
    bad = np.flatnonzero(wave == 5500.0)[0]
    data[bad] += 0.1
    sigma[bad] = 0.2
    result = analytic_score(wave, data, constant_i_model(profile(WAVE)), sigma,
                            max_shift_angstrom=600.0, resolution_fwhm_angstrom=fwhm)
    assert result["shift_angstrom"] == pytest.approx(0.0, abs=0.02)
    assert result["v_scale"] == pytest.approx(1.0, abs=0.001)


@pytest.mark.parametrize("level", [0.0, 0.125])
def test_flat_polarization_has_no_measurable_shift(level):
    data = np.full_like(GRID, level)
    with np.errstate(all="raise"):
        result = analytic_score(GRID, data, constant_i_model(np.full_like(WAVE, level)))
    assert result["chi2_v"] == 0.0
    assert np.isnan(result["shift_angstrom"])
    if level == 0.0:
        assert np.isnan(result["v_scale"])
    else:
        assert result["v_scale"] == 1.0


def test_flat_observation_cannot_align_a_structured_model():
    result = analytic_score(GRID, np.zeros_like(GRID), constant_i_model(gaussian(WAVE)))
    assert np.isnan(result["shift_angstrom"])
    assert result["v_scale"] == 0.0


def test_zero_model_cannot_align_noise_or_fit_an_amplitude():
    data = np.random.default_rng(2).normal(0.0, 0.005, GRID.size)
    with np.errstate(all="raise"):
        result = analytic_score(GRID, data, constant_i_model(np.zeros_like(WAVE)))
    assert np.isfinite(result["chi2_v"])
    assert np.isnan(result["v_scale"])
    assert np.isnan(result["shift_angstrom"])


def test_periodic_alignment_with_equal_aliases_is_unconstrained():
    profile = lambda w: 0.1 * np.sin(2.0 * np.pi * w / 50.0)
    result = analytic_score(GRID, profile(GRID), constant_i_model(profile(WAVE)))
    assert np.isnan(result["shift_angstrom"])


def test_noise_free_observation_has_zero_shift():
    result = analytic_score(GRID, gaussian(GRID), constant_i_model(gaussian(WAVE)))
    assert result["shift_angstrom"] == pytest.approx(0.0, abs=0.02)


def test_too_little_alignment_coverage_is_unconstrained():
    wave = np.arange(4000.0, 4201.0)
    profile = gaussian(wave, centre=4100.0)
    zero = np.zeros_like(wave)
    spectrum = StokesSpectrum(wave, np.ones_like(wave), {}, zero, zero, profile)
    result = analytic_score(wave, profile, spectrum, max_shift_angstrom=100.0)
    assert result["chi2_v"] == 0.0
    assert result["v_scale"] == 1.0
    assert np.isnan(result["shift_angstrom"])


def test_alignment_preserves_information_in_tiny_polarization():
    result = analytic_score(GRID, gaussian(GRID - 20.0, amplitude=1e-12),
                            constant_i_model(gaussian(WAVE, amplitude=1e-12)), sigma=1e-14)
    assert result["shift_angstrom"] == pytest.approx(20.0, abs=0.05)


def test_search_limit_is_explicit_and_does_not_change_unshifted_scores():
    spectrum = constant_i_model(gaussian(WAVE))
    data = gaussian(GRID - 150.0)
    narrow = analytic_score(GRID, data, spectrum, max_shift_angstrom=100.0)
    wide = analytic_score(GRID, data, spectrum, max_shift_angstrom=200.0)
    assert np.isnan(narrow["shift_angstrom"])
    assert wide["shift_angstrom"] == pytest.approx(150.0, abs=0.05)
    assert narrow["chi2_v"] == wide["chi2_v"]
    assert narrow["v_scale"] == wide["v_scale"]


def test_alignment_ignores_masked_pixels_and_recovers_an_offset_profile():
    data = 2.0 * gaussian(GRID - 20.0) + 0.02
    sigma = np.full_like(GRID, 0.001)
    data[200:220] = np.nan
    sigma[300:320] = np.inf
    result = analytic_score(GRID, data, constant_i_model(gaussian(WAVE)), sigma)
    assert result["shift_angstrom"] == pytest.approx(20.0, abs=0.05)


def test_empty_comparison_window_has_a_clear_error():
    with pytest.raises(ValueError, match="valid.*pixels"):
        analytic_score(GRID, gaussian(GRID), constant_i_model(gaussian(WAVE)),
                       window_angstrom=(7000.0, 7100.0))


def test_all_masked_pixels_have_a_clear_error():
    with pytest.raises(ValueError, match="valid.*pixels"):
        analytic_score(GRID, gaussian(GRID), constant_i_model(gaussian(WAVE)), sigma=np.inf)


def test_mismatched_observed_arrays_are_rejected():
    with pytest.raises(ValueError, match="matching 1D arrays"):
        analytic_score(GRID, gaussian(GRID[:-1]), constant_i_model(gaussian(WAVE)))


def test_nonpositive_model_intensity_is_rejected():
    zero = np.zeros_like(WAVE)
    spectrum = StokesSpectrum(WAVE, zero, {}, zero, zero, gaussian(WAVE))
    with pytest.raises(ValueError, match="I must be finite/positive"):
        analytic_score(GRID, gaussian(GRID), spectrum)


@pytest.mark.parametrize("limit", [0.0, -1.0, np.inf, np.nan])
def test_shift_limit_must_be_finite_and_positive(limit):
    with pytest.raises(ValueError, match="max_shift_angstrom"):
        analytic_score(GRID, gaussian(GRID), constant_i_model(gaussian(WAVE)),
                       max_shift_angstrom=limit)
