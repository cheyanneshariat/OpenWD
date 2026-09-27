"""Linear-Stark (quasi-static ionic) treatment of Rydberg metal lines."""

import numpy as np
import pytest

from wd_spectra.d6 import SDSS_J1637_LOG_NUMBER_ABUNDANCE
from wd_spectra.metals import (
    _metal_holtsmark_microfield_distribution,
    _pseudo_voigt_profile_grid,
    _pseudo_voigt_profile_per_angstrom,
    _tabulated_holtsmark_distribution,
    linear_stark_mixing_fraction,
    linear_stark_rydberg_level,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs


@pytest.fixture(scope="module")
def database():
    return _atomic_inputs(ModelData.default(), tuple(SDSS_J1637_LOG_NUMBER_ABUNDANCE))[0]


def _level(database, key, fragment):
    ion = database.ions[key]
    return ion, next(level for level in ion.levels if fragment in level.label)


def test_only_genuine_rydberg_levels_qualify(database):
    ion, level = _level(database, ("O", 0), ".(4So).9d.(5Do")
    principal, angular, effective_n, gap = linear_stark_rydberg_level(database, ion, level)
    assert (principal, angular) == (9, 2)
    assert effective_n == pytest.approx(8.96, abs=0.01)
    # l < 2, core-excited, and excited-parent levels are excluded.
    for fragment in (".(4So).3s.(5So", ".(2Do).3s.(3Do"):
        ion, level = _level(database, ("O", 0), fragment)
        assert linear_stark_rydberg_level(database, ion, level) is None


def test_ionic_defects_use_the_core_charge(database):
    ion, level = _level(database, ("Mg", 1), ".7f.(2Fo")
    principal, angular, effective_n, gap = linear_stark_rydberg_level(database, ion, level)
    assert effective_n == pytest.approx(7.0, abs=0.01)
    assert gap < 0.01


def test_highest_l_level_mixes_only_with_its_existing_neighbour(database):
    # Mg II 4f has no 4g partner; its gap is the defect difference to 4d.
    ion, level = _level(database, ("Mg", 1), ".4f.(2Fo")
    gap = linear_stark_rydberg_level(database, ion, level)[3]
    assert gap > 0.02


def test_mixing_fraction_limits_and_monotonicity():
    field = np.geomspace(1.0, 1.0e14, 60)
    mixing = linear_stark_mixing_fraction(9.0, 0.04, 1.0, field)
    assert mixing[0] < 1.0e-6
    assert mixing[-1] > 0.999
    assert np.all(np.diff(mixing) >= 0.0)
    # A larger quantum-defect gap needs a stronger field to mix.
    assert np.all(linear_stark_mixing_fraction(9.0, 0.2, 1.0, field) <= mixing)


def test_vectorized_profiles_match_scalar_definitions():
    beta = np.linspace(0.0, 12.0, 20_001)
    np.testing.assert_allclose(
        _tabulated_holtsmark_distribution(beta),
        _metal_holtsmark_microfield_distribution(beta),
        atol=1.0e-7,
    )
    wavelength = np.linspace(4990.0, 5010.0, 2001)
    sigma = np.array([0.02, 0.05, 0.2])
    hwhm = np.array([0.0, 0.3, 1.0])
    grid = _pseudo_voigt_profile_grid(wavelength, 5000.0, sigma, hwhm)
    for index in range(sigma.size):
        np.testing.assert_array_equal(
            grid[:, index],
            _pseudo_voigt_profile_per_angstrom(wavelength, 5000.0, sigma[index], hwhm[index]),
        )
