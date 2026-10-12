import numpy as np
import pytest

from wd_spectra.metals import (
    atomic_database_with_kurucz_electron_stark,
    atomic_database_with_si_ii_he_unified_profile,
    read_si_ii_he_unified_profile,
    read_stout_atomic_database,
)
from wd_spectra.models import DZConfig, ModelData

C = 2.99792458e10


@pytest.fixture(scope="module")
def databases():
    base = atomic_database_with_kurucz_electron_stark(read_stout_atomic_database(
        ModelData.default().stout, elements=("Si",), maximum_charge=2))
    return base, atomic_database_with_si_ii_he_unified_profile(base)


@pytest.mark.parametrize("temperature,density", [(12000.0, 1.0e19), (16000.0, 3.0e19), (20000.0, 8.0e19)])
def test_profile_is_normalized_and_positive(temperature, density):
    table = read_si_ii_he_unified_profile()
    detuning = np.concatenate([-np.geomspace(6000, 1e-4, 4000), [0.0], np.geomspace(1e-4, 6000, 4000)])
    gamma = 1.0e11
    profile = table.profile(detuning, np.array([temperature]), np.array([density]), np.array([gamma]))[:, 0]
    assert np.all(profile >= 0.0)
    area = np.trapz(profile, 2 * np.pi * C * detuning) if hasattr(np, "trapz") else np.trapezoid(profile, 2 * np.pi * C * detuning)
    assert area == pytest.approx(1.0, abs=0.01)


def test_only_the_3p_3d_triplet_is_attached(databases):
    base, unified = databases
    lines = [t for t in unified.ions["Si", 1].transitions if t.helium_unified_profile is not None]
    assert sorted(round(t.wavelength_vacuum_angstrom, 2) for t in lines) == [1260.42, 1264.74, 1265.0]
    for old, new in zip(base.ions["Si", 1].transitions, unified.ions["Si", 1].transitions):
        assert new.absorption_oscillator_strength == old.absorption_oscillator_strength
        assert new.electron_stark_rate_coefficient_cm3_s == old.electron_stark_rate_coefficient_cm3_s
    assert unified.ions["Si", 2] is base.ions["Si", 2]


def test_option_defaults_off():
    assert DZConfig().si_ii_he_unified_profile is False
