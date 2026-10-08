"""Resource-lifetime checks for the optional compiled metal-profile kernel."""

import sys

import numpy as np
import pytest

_rt = pytest.importorskip("wd_spectra._rt")


def _profile_arrays():
    # Distinct allocations let us check every borrowed buffer independently.
    return [
        np.array([4999., 5000., 5001.]),  # wavelength
        np.ones((3, 1)),                # Planck function
        np.array([5000.]),              # center
        np.array([1.]),                 # strength
        np.full((1, 1), 0.1),           # Gaussian sigma
        np.full((1, 1), 0.01),          # Lorentz HWHM
        np.array([2.]),                 # minimum half-window
        np.zeros((1, 1)),               # static scale
        np.zeros((1, 1)),               # static amplitude
        np.ones((1, 1)),                # population scale
        np.ones((1, 1)),                # lower departure
        np.ones((1, 1)),                # upper departure
        np.full((1, 1), 0.2),           # Boltzmann factor
        np.array([1.]),                 # quadrature argument
        np.array([1.]),                 # quadrature weight
        np.zeros((3, 1)),               # absorption output
        np.zeros((3, 1)),               # emissivity output
    ]


@pytest.mark.parametrize("failure", [None, "shape", "readonly", "strided", *range(17)])
def test_metal_profile_releases_every_buffer(failure):
    arrays = _profile_arrays()
    if failure == "shape":
        # Error after all 17 buffers have been acquired.
        arrays[-1] = np.zeros((2, 1))
    elif failure == "readonly":
        arrays[-1].flags.writeable = False
    elif failure == "strided":
        arrays[-1] = np.zeros((6, 1))[::2]
    elif failure is not None:
        # Includes validation failure on the final emissivity buffer.
        arrays[failure] = arrays[failure].astype(np.float32)
    before = [sys.getrefcount(array) for array in arrays]
    for _ in range(3):
        if failure is None:
            _rt.accumulate_metal_line_profiles(*arrays, False)
        else:
            with pytest.raises((TypeError, ValueError, BufferError)):
                _rt.accumulate_metal_line_profiles(*arrays, False)
    assert [sys.getrefcount(array) for array in arrays] == before
    if failure is None:
        assert np.all(arrays[-2] >= 0)
        assert arrays[-2][1, 0] > 0
        np.testing.assert_allclose(arrays[-1], arrays[-2])
