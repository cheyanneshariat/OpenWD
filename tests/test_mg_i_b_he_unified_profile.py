"""Allard et al. (2016) Mg I b--He unified profiles (figure vectors)."""
import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.constants import LIGHT_SPEED, PI
from wd_spectra.metals import (
    MgIbHeUnifiedProfile,
    atomic_database_with_mg_i_b_he_unified_profile,
    read_stout_atomic_database,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.stellar import DZConfig

DATA = ModelData.default()
X = np.arange(-3000.0, 3000.0, 0.5)


@pytest.mark.parametrize("component", [0, 1, 2])
def test_unit_area_and_continuity(component):
    p = MgIbHeUnifiedProfile(DATA.mg_i_b_he_density_profiles, component)
    n = np.array([1e19, 1e20, 0.9999e21, 1e21, 2.5e21, 4e21, 2e22])
    I = p.profile(X, np.full(n.size, 6000.0), n, np.zeros(n.size))
    area = trapezoid(I, X, axis=0) * 2 * PI * LIGHT_SPEED
    assert np.allclose(area, 1.0, atol=0.005)
    assert np.max(np.abs(I[:, 2] - I[:, 3])) < 0.01 * I[:, 3].max()


def test_satellite_moves_blue_with_density():
    p = MgIbHeUnifiedProfile(DATA.mg_i_b_he_density_profiles, 2)
    n = np.array([1e21, 2e21, 3e21, 4e21])
    I = p.profile(X, np.full(4, 6000.0), n, np.zeros(4))
    peaks = X[np.argmax(I, axis=0)]
    assert np.all(np.diff(peaks) > 0)
    assert peaks[-1] > 400.0      # Allard Fig. 11: J2 peak near 5051 A at 4e21


def test_attaches_only_the_triplet():
    atomic = read_stout_atomic_database(DATA.stout, elements=("Mg",), maximum_charge=1)
    out = atomic_database_with_mg_i_b_he_unified_profile(atomic, DATA.mg_i_b_he_density_profiles)
    tagged = [t for t in out.ions["Mg", 0].transitions if t.helium_unified_profile is not None]
    centers = sorted(round(t.wavelength_vacuum_angstrom, 1) for t in tagged)
    assert centers == [5168.8, 5174.1, 5185.0]
    assert sorted(t.helium_unified_profile.component for t in tagged) == [0, 1, 2]


def test_default_off():
    assert DZConfig().mg_i_b_he_unified_profile is False
