from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra.h2_lines import (
    H2_RADIATION_GRID_ANGSTROM,
    h2_line_mass_absorption,
    h2_partition_function,
    h2_source_function_ratio,
    read_h2_lyman_werner_data,
)
from wd_spectra.models import DAConfig, DAZConfig


@pytest.fixture(scope="module")
def data():
    return read_h2_lyman_werner_data()


def _upper(data, energy, tolerance=0.5):
    """Upper-level indices whose lines reach the given upper-level energy."""
    lower = data.lower_energy_wavenumber
    upper_energy = lower + 1.0e8 / data.wavelength_angstrom  # measured-level lines
    lines = np.flatnonzero(np.abs(upper_energy - energy) < tolerance)
    return np.unique(data.upper_index[lines])


def test_bundled_lines_are_measured_level_lyman_and_werner(data):
    assert data.n_lines == 10259 and data.branch_wavelength_angstrom.size == 28486
    assert 840.0 < data.wavelength_angstrom.min() and data.wavelength_angstrom.max() < 1850.0
    assert np.all(data.lower_energy_wavenumber >= 0.0)
    assert "measured levels" in data.source


def test_kurucz_gf_convention_gives_j_independent_lifetimes(data):
    # gf = g_ns (2J''+1) f / 4: B(v'=0) levels J'=1, 2, 3 must share one radiative
    # lifetime, close to Abgrall, Roueff & Drira (2000): A = 1.87e9 s^-1 (0.53 ns).
    lifetimes = []
    for energy in (90242.380, 90319.671, 90434.799):
        upper = _upper(data, energy, tolerance=0.3)
        assert upper.size == 1
        u = upper[0]
        total = data.upper_bound_decay_rate[u] + data.upper_dissociation_rate[u]
        lifetimes.append(1.0 / total)
    lifetimes = np.array(lifetimes)
    assert np.ptp(lifetimes) / lifetimes.mean() < 0.05
    assert lifetimes.mean() == pytest.approx(1.0 / 1.87e9, rel=0.1)


def test_source_ratio_is_one_in_a_thermal_radiation_field(data):
    temperature = np.array([7000.0, 9000.0, 12000.0])
    ratio = h2_source_function_ratio(
        data, H2_RADIATION_GRID_ANGSTROM,
        np.ones((H2_RADIATION_GRID_ANGSTROM.size, 3)), temperature, np.array([1e12, 1e15, 1e17]),
    )
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-12)


def test_collisions_thermalize_and_radiation_fills_in(data):
    temperature = np.full(2, 9000.0)
    bright = np.full((H2_RADIATION_GRID_ANGSTROM.size, 2), 10.0)
    ratio = h2_source_function_ratio(
        data, H2_RADIATION_GRID_ANGSTROM, bright, temperature, np.array([1.0e10, 1.0e22])
    )
    # Radiation-dominated: S/B approaches the branch-weighted J/B (less dissociative decay).
    assert np.median(ratio[:, 0]) > 5.0
    # Collision-dominated: LTE.
    np.testing.assert_allclose(np.median(ratio[:, 1]), 1.0, atol=0.05)


def test_partition_function_uses_the_eos_normalization():
    # Barklem & Collet divide by the nuclear-spin weight 4; the line populations
    # use the full sum, so Q is four times the tabulated value.
    from wd_spectra.molecules import molecular_hydrogen_internal_partition_function

    assert h2_partition_function(8000.0)[0] == pytest.approx(
        4.0 * molecular_hydrogen_internal_partition_function(8000.0)
    )


def _atmosphere(h2_density):
    n = 3
    state = SimpleNamespace(molecular_hydrogen_density=h2_density)
    return SimpleNamespace(
        n_depth=n, temperature=np.full(n, 9000.0), mass_density=np.full(n, 1.0e-7),
        hydrogen_lte_state=state,
    )


def test_opacity_requires_molecular_hydrogen(data):
    wave = np.arange(1340.0, 1360.0, 0.01)
    kappa, weighted = h2_line_mass_absorption(data, _atmosphere(None), wave)
    assert not np.any(kappa) and not np.any(weighted)
    kappa, weighted = h2_line_mass_absorption(data, _atmosphere(np.full(3, 1.0e12)), wave)
    assert np.all(kappa >= 0.0) and kappa.max() > 0.0
    np.testing.assert_array_equal(kappa, weighted)


def test_option_is_opt_in_and_requires_molecules():
    assert DAConfig().h2_lyman_werner_lines is False
    assert DAZConfig().h2_lyman_werner_lines is False
