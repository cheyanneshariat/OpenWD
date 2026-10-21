"""Computed Mg II h&k and Ca II H&K + He unified-profile tables."""
import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.metals import UnifiedIonicHeProfileGrid
from wd_spectra.models.common import ModelData

DATA = ModelData.default()
# pi e^2 / (m_e c) [cm^2 Hz]; summed f of the doublets.
PI_E2_MC = 0.02654
F_SUM = {"ca": 0.682 + 0.330, "mg": 0.608 + 0.303}


def _tables():
    return {"ca": UnifiedIonicHeProfileGrid(DATA.ca_ii_he_unified_profiles),
            "mg": UnifiedIonicHeProfileGrid(DATA.mg_ii_he_unified_profiles)}


def _figure_sum(path, value, wavelength):
    data = np.loadtxt(path, comments="#")
    sigma = np.zeros_like(wavelength)
    for j in (0.5, 1.5):
        rows = data[np.isclose(data[:, 0], value) & (data[:, 1] == j)]
        order = np.argsort(rows[:, 2])
        sigma += np.interp(wavelength, rows[order, 2], rows[order, 3], left=0.0, right=0.0)
    return sigma


@pytest.mark.parametrize("ion", ["ca", "mg"])
@pytest.mark.parametrize("temperature", [4000.0, 8000.0])
def test_oscillator_strength_sum_rule_at_low_density(ion, temperature):
    table = _tables()[ion]
    w = table.wavelength_angstrom
    sigma = table.cross_section(w, np.array([1e20]), temperature=np.array([temperature]))[:, 0]
    area = -trapezoid(sigma, LIGHT_SPEED / (w * 1e-8))
    assert area == pytest.approx(PI_E2_MC * F_SUM[ion], rel=0.01)


def test_dense_fluid_band_area():
    """The R-dependent 4s-4p dipole moves Ca II H&K strength into other bands as He packs
    around the ion (Mg II: constant dipole, area conserved)."""
    tables = _tables()
    areas = {}
    for ion, table in tables.items():
        w = table.wavelength_angstrom
        sigma = table.cross_section(w, np.array([1e21, 1e22, 1e23]), temperature=np.full(3, 6000.0))
        areas[ion] = -trapezoid(sigma, LIGHT_SPEED / (w * 1e-8), axis=0) / (PI_E2_MC * F_SUM[ion])
    assert np.all(np.diff(areas["ca"]) < 0) and 0.3 < areas["ca"][-1] < 0.6
    assert areas["ca"][0] == pytest.approx(0.994, abs=0.01)
    assert np.allclose(areas["mg"], 1.0, atol=0.06)


@pytest.mark.parametrize("ion", ["ca", "mg"])
def test_grid_nodes_and_density_scaling(ion):
    table = _tables()[ion]
    w = table.wavelength_angstrom[::7]
    i, j = 3, 8
    t, n = table.temperatures[i], table.densities[j]
    node = table.cross_section(w, np.array([n]), temperature=np.array([t]))[:, 0]
    assert np.allclose(node, table.cross_section_by_condition[i, j, ::7], rtol=1e-12, atol=0)
    top, low = table.densities[-1], table.densities[0]
    pair = table.cross_section(w, np.array([top, 3 * top, low, 0.1 * low]),
                               temperature=np.full(4, t))
    assert np.allclose(pair[:, 1], 3 * pair[:, 0])
    assert np.allclose(pair[:, 3], 0.1 * pair[:, 2])
    with pytest.raises(ValueError):
        table.cross_section(w, np.array([n]))


def test_far_wings_grow_with_density():
    # Below the dense-fluid regime (by 1e23 correlations and the band-area loss
    # cut the red wing, and the near wings saturate as the line spreads).
    table = _tables()["ca"]
    w = np.array([2800.0, 3000.0, 4800.0, 5200.0])
    n = np.array([1e20, 1e21, 1e22])
    sigma = table.cross_section(w, n, temperature=np.full(3, 6000.0))
    assert np.all(np.diff(sigma, axis=1) > 0)


@pytest.mark.parametrize("temperature", [4000.0, 6000.0, 10000.0])
def test_ca_ii_matches_allard_alekseev_2014(temperature):
    """Same potentials and dipoles as Allard & Alekseev (2014) Fig. 5 (5e21)."""
    table = _tables()["ca"]
    w = np.linspace(2600.0, 4400.0, 1800)
    ours = table.cross_section(w, np.array([5e21]), temperature=np.array([temperature]))[:, 0]
    ref = _figure_sum(DATA.cache / "metal-opacity/unified-metal-he/caii-hk-he-5e21-temperature.dat",
                      temperature, w)
    for lo, hi in ((2600.0, 3700.0), (4050.0, 4400.0)):
        m = (w > lo) & (w < hi) & (ref > 0)
        assert 0.8 < np.median(ours[m] / ref[m]) < 1.25


@pytest.mark.parametrize("density", [1e21, 1e22])
def test_mg_ii_matches_allard_2016(density):
    """Same GG potentials as Allard et al. (2016) Figs. 5-6 (6000 K)."""
    table = _tables()["mg"]
    w = np.linspace(2850.0, 3600.0, 750)
    ours = table.cross_section(w, np.array([density]), temperature=np.array([6000.0]))[:, 0]
    ref = _figure_sum(DATA.cache / "metal-opacity/unified-metal-he/mgii-hk-he-6000K-density.dat",
                      density, w)
    m = ref > 0
    assert 0.8 < np.median(ours[m] / ref[m]) < 1.25
