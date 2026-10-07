"""Doppler-convolved table edges must keep their physical wavelength scale."""
import numpy as np
import pytest

from wd_spectra.stark import default_balmer_stark_table


@pytest.mark.parametrize("transition,center", [((2, 3), 6564.636), ((2, 4), 4862.683)])
@pytest.mark.parametrize("temperature", [5000.0, 6000.0, 20000.0])
@pytest.mark.parametrize("electron_density", [1e6, 1e8, 1e9])
def test_below_density_grid_keeps_entire_doppler_convolved_edge(
    transition, center, temperature, electron_density
):
    line = default_balmer_stark_table()[transition]
    edge_density = 10.0 ** line.log_electron_density[0]
    wavelength = center + np.linspace(-5.0, 5.0, 4001)
    edge = line.wavelength_profile(wavelength, center, temperature, edge_density)
    below = line.wavelength_profile(wavelength, center, temperature, electron_density)
    # The profile already contains the edge's thermal convolution. A query
    # outside the density grid must not squeeze that profile as ne^(2/3).
    np.testing.assert_array_equal(below, edge)


@pytest.mark.parametrize("density", [1e10, 3.2e10, 1e14, 1e18, 1e20])
def test_density_floor_preserves_in_range_and_upper_edge_field_scaling(density):
    line = default_balmer_stark_table()[(2, 3)]
    field, _ = line._local_profile_state(6000.0, density)
    assert field == 1.25e-9 * density ** (2.0 / 3.0)


def test_low_density_profile_has_finite_width_instead_of_collapsing():
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    offset = np.linspace(-0.5, 0.5, 10001)
    profile = line.wavelength_profile(center + offset, center, 5000.0, 1e8)
    half_maximum = offset[profile >= profile.max() / 2.0]
    width = half_maximum[-1] - half_maximum[0]
    # Independent hydrogen thermal FWHM at 5000 K is about 0.331 A.
    # A broad bound avoids asserting perfect accuracy of table interpolation.
    assert 0.30 < width < 0.36
