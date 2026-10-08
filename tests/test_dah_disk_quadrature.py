"""Opt-in drift-resolved DAH disk quadrature (geometry only; no synthesis)."""

import numpy as np
import pytest

from wd_spectra import DAHConfig
from wd_spectra.magnetic import (
    LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS,
    WEAK_FIELD_MAXIMUM_MEGAGAUSS,
    balmer_component_drift_rate,
    drift_resolved_field_edges,
    field_binned_surface_cells,
    dipole_surface_cells,
)
from wd_spectra.magnetic_atomic import read_h2db_transition_database
from wd_spectra.models.common import ModelData
from wd_spectra.models.dah import dah_surface_cells


@pytest.fixture(scope="module")
def transitions():
    return read_h2db_transition_database(ModelData.default().h2db_balmer_subset)


def test_default_quadrature_is_unchanged():
    config = DAHConfig(magnetic_field_megagauss=368.52, field_geometry="dipole",
                       dipole_inclination_deg=34.0, dipole_offset_radius=(0.0, 0.0, 0.07))
    assert config.disk_component_drift_angstrom is None
    cells = dah_surface_cells(config)
    reference = dipole_surface_cells(368.52, inclination_deg=34.0, offset_vector_radius=(0.0, 0.0, 0.07))
    np.testing.assert_array_equal(cells.field_strength_megagauss, reference.field_strength_megagauss)
    assert cells.projected_weight.size == 21


def test_normal_triplet_drift_is_analytic():
    # Halpha alone in the window: d lambda/dB = lambda^2 e/(4 pi m_e c^2).
    field = np.linspace(0.1, 0.9, 9)
    drift = balmer_component_drift_rate(field, (6000.0, 7000.0), None)
    expected = 6564.6**2 * LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS * 1.0e6 / 2.99792458e10 * 1.0e-8
    np.testing.assert_allclose(drift, expected, rtol=1e-4)
    assert 20.0 < expected < 20.3  # about 20 A/MG at Halpha


def test_h2db_drift_matches_component_finite_difference(transitions):
    # Independent check through the synthesis accessor at 300 MG.
    temperature = np.asarray([12_000.0])
    a = transitions.balmer_components(3, 6564.6, 300.0, temperature, minimum_relative_strength=0.0)
    b = transitions.balmer_components(3, 6564.6, 301.0, temperature, minimum_relative_strength=0.0)
    window = (a.wavelength_angstrom > 3600.0) & (a.wavelength_angstrom < 7000.0)
    strong = window & (a.normalized_strength[:, 0] > 0.05)
    expected = np.max(np.abs(b.wavelength_angstrom - a.wavelength_angstrom)[strong])
    drift = balmer_component_drift_rate(np.linspace(299.0, 302.0, 31), (3600.0, 7000.0), transitions,
                                        maximum_upper_level=3, minimum_relative_strength=0.05)
    assert drift[10] == pytest.approx(expected, rel=0.15)


def test_edges_bound_accumulated_drift():
    edges = drift_resolved_field_edges((10.0, 30.0), lambda b: np.full(b.size, 3.0), 2.0)
    # 60 A of drift in steps of at most 2 A: 30 bins of 2/3 MG.
    assert edges.size == 31
    np.testing.assert_allclose(np.diff(edges), 2.0 / 3.0, rtol=1e-9)
    crossing = drift_resolved_field_edges((0.5, 1.5), lambda b: np.full(b.size, 20.0), 4.0)
    assert WEAK_FIELD_MAXIMUM_MEGAGAUSS in crossing
    with pytest.raises(ValueError):
        drift_resolved_field_edges((1.0, 2.0), lambda b: np.full(b.size, 1.0), 0.0)


def test_binned_cells_conserve_weight_and_mean_field(transitions):
    raw = dipole_surface_cells(368.52, inclination_deg=34.0, offset_vector_radius=(0.0, 0.0, 0.07),
                               n_mu=96, n_azimuth=192, n_field_bins=None)
    edges = np.linspace(raw.field_strength_megagauss.min(), raw.field_strength_megagauss.max(), 50)
    binned = field_binned_surface_cells(raw, edges)
    assert binned.projected_weight.sum() == pytest.approx(1.0, abs=1e-12)
    raw_mean = np.sum(raw.projected_weight * raw.field_strength_megagauss)
    assert np.sum(binned.projected_weight * binned.field_strength_megagauss) == pytest.approx(raw_mean, rel=1e-12)
    assert binned.field_bounds_megagauss == raw.field_bounds_megagauss


def test_drift_resolved_config_bounds_component_steps(transitions):
    config = DAHConfig(magnetic_field_megagauss=437.1, field_geometry="dipole",
                       dipole_inclination_deg=10.0, dipole_offset_radius=(0.0, 0.0, -0.15),
                       disk_component_drift_angstrom=8.0)
    cells = dah_surface_cells(config, transitions=transitions, wavelength_range_angstrom=(3500.0, 7100.0))
    fields = np.sort(cells.field_strength_megagauss)
    lower, upper = cells.field_bounds_megagauss
    assert lower <= fields[0] and fields[-1] <= upper
    grid = np.linspace(fields[0], fields[-1], 4001)
    rate = balmer_component_drift_rate(grid, (3500.0, 7100.0), transitions)
    accumulated = np.concatenate(([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(grid))))
    steps = np.diff(np.interp(fields, grid, accumulated))
    # Bin means of adjacent intervals can differ by up to two tolerances.
    assert np.max(steps) <= 2.0 * 8.0 + 1e-6
    assert cells.projected_weight.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dah_surface_cells(DAHConfig(magnetic_field_megagauss=10.0, disk_component_drift_angstrom=2.0))


def test_domain_notes_flag_documented_high_field_approximations():
    from wd_spectra.models.dah import _domain_notes

    high = dict(magnetic_field_megagauss=368.52, field_geometry="dipole",
                dipole_inclination_deg=34.0, dipole_offset_radius=(0.0, 0.0, 0.07))
    default = DAHConfig(**high)
    notes = _domain_notes(default, dah_surface_cells(default), True)
    assert any("pseudo-continuum" in note for note in notes)
    assert any("equal-weight field bins" in note for note in notes)
    rwa = DAHConfig(**high, include_rwa_photoionization=True, disk_component_drift_angstrom=16.0)
    notes = _domain_notes(rwa, dah_surface_cells(DAHConfig(**high)), True)
    assert len(notes) == 1 and "not a flux-error bound" in notes[0]
    low = DAHConfig(magnetic_field_megagauss=0.5, field_geometry="dipole")
    assert _domain_notes(low, dah_surface_cells(low), False) == []


def test_new_option_preserves_positional_config_arguments():
    from dataclasses import fields

    assert fields(DAHConfig)[10].name == "atmosphere_structure"
    assert fields(DAHConfig)[-1].name == "disk_component_drift_angstrom"


def test_zero_field_drift_quadrature_is_finite():
    config = DAHConfig(magnetic_field_megagauss=0.0, field_geometry="dipole",
                       disk_component_drift_angstrom=16.0)
    cells = dah_surface_cells(config)
    reference = dah_surface_cells(DAHConfig(magnetic_field_megagauss=0.0, field_geometry="dipole"))
    np.testing.assert_array_equal(cells.field_strength_megagauss, reference.field_strength_megagauss)
    np.testing.assert_array_equal(cells.ray_mu, reference.ray_mu)
    np.testing.assert_array_equal(cells.projected_weight, reference.projected_weight)
    assert cells.field_bounds_megagauss == (0.0, 0.0)


def test_binning_does_not_merge_exact_atomic_boundary():
    from wd_spectra.magnetic import SurfaceCells

    raw = SurfaceCells(np.array([0.9, 1.0, 1.1]), np.ones(3), np.ones(3),
                       np.ones(3) / 3, (0.9, 1.1), True)
    cells = field_binned_surface_cells(raw, [0.9, 1.0, 1.1])
    np.testing.assert_allclose(cells.field_strength_megagauss, [0.9, 1.0, 1.1])
    np.testing.assert_allclose(cells.projected_weight.sum(), 1.0)


@pytest.mark.parametrize("drift", [0.0, -1.0, np.nan, np.inf])
def test_invalid_drift_is_rejected_before_geometry(drift):
    with pytest.raises(ValueError, match="finite and positive"):
        dah_surface_cells(DAHConfig(field_geometry="dipole", disk_component_drift_angstrom=drift))


@pytest.mark.parametrize("offset", [(0.0, 0.0, 1.0), (np.nan, 0.0, 0.0), (0.0, 0.0)])
def test_invalid_offset_is_rejected_before_dense_allocation(offset):
    with pytest.raises(ValueError, match="modulus < 0.8"):
        dah_surface_cells(DAHConfig(field_geometry="dipole", dipole_offset_radius=offset,
                                    disk_component_drift_angstrom=16.0))


def test_invalid_drift_grid_and_interval_are_rejected():
    with pytest.raises(ValueError):
        balmer_component_drift_rate([1.0, np.nan], (3400.0, 8000.0), None)
    with pytest.raises(ValueError):
        balmer_component_drift_rate([1.0, 2.0], (8000.0, 3400.0), None)
    with pytest.raises(ValueError):
        drift_resolved_field_edges((1.0, 2.0), lambda b: b, 16.0, n_field_samples=1)
    with pytest.raises(ValueError, match="more than 4096"):
        drift_resolved_field_edges((1.0, 2.0), lambda b: np.ones_like(b), 1e-6)


def test_binning_rejects_nonfinite_or_incomplete_edges():
    raw = dipole_surface_cells(0.5)
    with pytest.raises(ValueError):
        field_binned_surface_cells(raw, [0.0, np.nan])
    with pytest.raises(ValueError, match="cover every surface field"):
        field_binned_surface_cells(raw, [0.0, 0.1])
