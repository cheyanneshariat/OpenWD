"""Computed Mg I b--He unified-profile table (MgIbHeComputedProfile)."""
import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.constants import LIGHT_SPEED, PI
from wd_spectra.metals import MgIbHeComputedProfile
from wd_spectra.models.common import ModelData

DATA = ModelData.default()


@pytest.fixture(scope="module")
def table():
    return MgIbHeComputedProfile(DATA.mg_i_b_he_unified_profiles, 1)


def test_grid_is_fine_enough(table):
    assert np.max(np.diff(np.log10(table.densities))) <= 0.05 + 1e-9
    assert {4500.0, 5500.0} <= set(table.temperatures)
    assert "Sigma dV consistent" in table.source


def test_interpolation_conserves_band_area(table):
    """Between nodes the returned area stays within 4% of the bilinear node areas."""
    x = table.detuning_cm1
    node_area = trapezoid(table.profiles, x, axis=2)
    lt, ln = np.log(table.temperatures), np.log(table.densities)
    for T in (4250.0, 5250.0, 6250.0):
        n = np.geomspace(1e21, 3e22, 23)
        prof = table.profile(x, np.full(n.size, T), n, np.zeros(n.size)) * 2 * PI * LIGHT_SPEED
        area = trapezoid(prof, x, axis=0)
        i = np.searchsorted(lt, np.log(T)) - 1
        j = np.searchsorted(ln, np.log(n)) - 1
        wt = (np.log(T) - lt[i]) / (lt[i + 1] - lt[i])
        wn = (np.log(n) - ln[j]) / (ln[j + 1] - ln[j])
        ref = ((1 - wt) * (1 - wn) * node_area[i, j] + wt * (1 - wn) * node_area[i + 1, j]
               + (1 - wt) * wn * node_area[i, j + 1] + wt * wn * node_area[i + 1, j + 1])
        assert np.all(np.abs(area / ref - 1.0) < 0.04)


def test_physical_band_area_falls_with_density(table):
    x = table.detuning_cm1
    n = np.array([1e20, 1e21, 1e22, 1e23])
    area = trapezoid(table.profile(x, np.full(4, 5000.0), n, np.zeros(4)), x, axis=0) * 2 * PI * LIGHT_SPEED
    assert abs(area[0] - 1.0) < 0.01
    assert np.all(np.diff(area) < 0)


def test_absorption_maximum_moves_blue_with_density(table):
    x = table.detuning_cm1
    n = np.array([1e21, 3e21, 1e22])
    prof = table.profile(x, np.full(3, 6000.0), n, np.zeros(3))
    peaks = x[np.argmax(prof, axis=0)]
    assert np.all(np.diff(peaks) > 0)
    assert 300.0 < peaks[1] < 700.0 and peaks[2] > 1200.0
