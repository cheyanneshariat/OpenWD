import numpy as np
import pytest

from wd_spectra import _rt


pytestmark = pytest.mark.skipif(
    _rt is None or not hasattr(_rt, "metal_line_mean_intensity"),
    reason="optional compiled line-mean kernel is not built",
)


def _line_mean_inputs():
    n_wave, n_depth, n_line = 30, 4, 3
    wavelength = np.linspace(5000.0, 5010.0, n_wave)
    intensity = np.ones((n_wave, n_depth), dtype=np.float64)
    lambda_diagonal = np.full((n_wave, n_depth), 2.0, dtype=np.float64)
    center = np.array([5003.0, 5005.0, 5007.0], dtype=np.float64)
    sigma = np.full((n_line, n_depth), 0.05, dtype=np.float64)
    gamma = np.full((n_line, n_depth), 0.01, dtype=np.float64)
    mean = np.empty((n_line, n_depth), dtype=np.float64)
    mean_lambda = np.empty((n_line, n_depth), dtype=np.float64)
    return (
        wavelength,
        intensity,
        lambda_diagonal,
        center,
        sigma,
        gamma,
        mean,
        mean_lambda,
    )


@pytest.mark.parametrize(
    "bad_input",
    [
        "nan_center",
        "zero_center",
        "nan_wavelength",
        "zero_wavelength",
        "duplicate_wavelength",
        "decreasing_wavelength",
    ],
)
def test_native_line_mean_rejects_nonphysical_interpolation_inputs(bad_input):
    arrays = list(_line_mean_inputs())
    if bad_input == "nan_center":
        arrays[3][1] = np.nan
    elif bad_input == "zero_center":
        arrays[3][1] = 0.0
    elif bad_input == "nan_wavelength":
        arrays[0][5] = np.nan
    elif bad_input == "zero_wavelength":
        arrays[0][5] = 0.0
    elif bad_input == "duplicate_wavelength":
        arrays[0][5] = arrays[0][4]
    elif bad_input == "decreasing_wavelength":
        arrays[0][5] = arrays[0][6] + 1.0
    with pytest.raises(ValueError, match="line-mean|line centers"):
        _rt.metal_line_mean_intensity(*arrays, False)


def test_native_line_mean_accepts_finite_positive_ordered_inputs():
    arrays = _line_mean_inputs()
    _rt.metal_line_mean_intensity(*arrays, True)
    np.testing.assert_allclose(arrays[6], 1.0)
    np.testing.assert_allclose(arrays[7], 2.0)
