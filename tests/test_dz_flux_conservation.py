"""Numerical pieces that keep polluted-helium spectra flux conserving."""

import numpy as np
import pytest

from wd_spectra.atmosphere import (
    _metal_line_opacity_sampling_grid,
    photosphere_concentrated_optical_depths,
)
from wd_spectra.radiative_transfer import emergent_flux
from wd_spectra.spectrum import _depth_refined_transfer_inputs, planck_lambda_angstrom


def test_zero_concentration_is_the_historical_uniform_grid():
    np.testing.assert_allclose(
        photosphere_concentrated_optical_depths(1.0e-8, 1.0e2, 40, 0.0),
        np.geomspace(1.0e-8, 1.0e2, 40),
    )


def test_concentration_moves_points_into_the_photosphere():
    uniform = photosphere_concentrated_optical_depths(1.0e-8, 1.0e2, 40, 0.0)
    dense = photosphere_concentrated_optical_depths(1.0e-8, 1.0e2, 40, 1.0)
    assert dense.size == 40
    assert dense[0] == 1.0e-8 and dense[-1] == 1.0e2
    assert np.all(np.diff(dense) > 0.0)

    def photospheric(tau):
        return int(np.sum((tau > 0.01) & (tau < 10.0)))

    assert photospheric(dense) >= photospheric(uniform) + 6
    # Points come from the thin upper layers, never from the deep interior.
    deep_step = np.max(np.diff(np.log(dense[dense > 1.0])))
    assert deep_step <= np.log(uniform[-1] / uniform[-2]) * (1.0 + 1.0e-9)
    with pytest.raises(ValueError):
        photosphere_concentrated_optical_depths(1.0e-8, 1.0e2, 40, -1.0)


def test_opacity_sampling_grid_is_uniform_and_independent_of_line_count():
    rng = np.random.default_rng(1)
    few = np.sort(rng.uniform(1500.0, 6000.0, 200))
    many = np.sort(rng.uniform(1500.0, 6000.0, 20_000))
    grids = [
        _metal_line_opacity_sampling_grid(
            centers, opacity_sampling_resolution=1000.0,
            effective_temperature=12_000.0,
        )
        for centers in (few, many)
    ]
    for grid, wing_sampled in grids:
        assert wing_sampled == 0
        step = np.diff(np.log(grid))
        np.testing.assert_allclose(step, 1.0e-3, rtol=1.0e-9)
    # The grid spans the line centers, so 100x more lines over nearly the
    # same interval add at most a few per cent more samples.
    assert grids[1][0].size < 1.05 * grids[0][0].size
    # Without opacity sampling the stencil grid grows with the line list.
    stencil, _ = _metal_line_opacity_sampling_grid(many)
    assert stencil.size > 10 * grids[1][0].size


def test_depth_refinement_preserves_nodes_and_converges_the_emergent_flux():
    column_mass = np.geomspace(1.0e-6, 1.0e2, 25)
    temperature = 6000.0 * (1.0 + (column_mass / 0.3) ** 0.45)
    wavelength = np.asarray([3000.0, 5000.0, 9000.0])
    absorption = 0.4 * np.ones((wavelength.size, column_mass.size)) * (
        column_mass / column_mass[0]
    ) ** 0.2
    scattering = 1.0e-3 * np.ones_like(absorption)
    planck = planck_lambda_angstrom(wavelength[:, None], temperature[None, :])

    same = _depth_refined_transfer_inputs(
        column_mass, temperature, wavelength, absorption, scattering, planck, 1
    )
    assert same[0] is column_mass and same[3] is planck

    def flux(factor):
        mass, kappa, sigma, source = _depth_refined_transfer_inputs(
            column_mass, temperature, wavelength, absorption, scattering,
            planck, factor,
        )
        total = kappa + sigma
        tau = np.concatenate(
            (total[:, :1] * mass[0], total[:, :1] * mass[0] + np.cumsum(
                0.5 * (total[:, 1:] + total[:, :-1]) * np.diff(mass), axis=1
            )),
            axis=1,
        )
        return mass, kappa, source, emergent_flux(tau, source, n_angle=4)

    mass2, kappa2, source2, _ = flux(2)
    np.testing.assert_allclose(mass2[::2], column_mass, rtol=1.0e-12)
    np.testing.assert_allclose(kappa2[:, ::2], absorption, rtol=1.0e-10)
    np.testing.assert_allclose(source2[:, ::2], planck, rtol=1.0e-10)
    reference = flux(32)[3]
    coarse_error = np.max(np.abs(flux(1)[3] / reference - 1.0))
    refined_error = np.max(np.abs(flux(4)[3] / reference - 1.0))
    assert refined_error < 0.25 * coarse_error
