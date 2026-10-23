"""Dense-He He- free-free correction in the DZ preset (opt-in)."""
import numpy as np
import pytest

from wd_spectra.constants import HELIUM_MASS
from wd_spectra.dense_helium_continuum import (
    DenseHeliumCorrectionTable,
    bounded_helium_minus_correction,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.stellar import DZConfig


@pytest.fixture(scope="module")
def table():
    return DenseHeliumCorrectionTable(ModelData.default().helium_minus_dense_free_free)


def test_table_is_rogers_young_without_dielectric(table):
    assert table.metadata["schema"] == "dense-He-ff-v1"
    assert table.sha256 == (
        "46e7fb094d7385e91ff13b4bfdddccbaf25de11650eefc178ee71cf93a7760b0"
    )
    # v1 isotherms, the 2412/2649 K cool-DZ extension and the 861-2196 K
    # dense-db extension (tools/extend_dense_heminus_table.py)
    assert np.exp(table.axes[0][0]) == pytest.approx(861.222, rel=1e-6)


def test_cool_isotherms_refuse_unconverged_fluid_density(table):
    limit = table.maximum_valid_log1p_density
    full = table.axes[1][-1]
    cool = np.exp(table.axes[0]) < 2300
    assert np.all(limit[~cool] == full) and np.any(limit[cool] < full)
    wave = np.array([5500.0])
    f = bounded_helium_minus_correction(table, wave, np.array([1000.0, 1000.0]),
                                        np.array([0.0, 0.3]) / HELIUM_MASS)
    assert f[0, 0] == 1.0 and 0.25 < f[0, 1] < 0.45
    with pytest.raises(ValueError, match="converged fluid density"):
        bounded_helium_minus_correction(table, wave, np.array([900.0]), np.array([1.3 / HELIUM_MASS]))


def test_dilute_limit_and_monotone_density(table):
    wave = np.array([3000.0, 6000.0])
    t = np.full(5, 5000.0)
    rho = np.array([0.0, 0.05, 0.2, 0.4, 0.8])
    f = bounded_helium_minus_correction(table, wave, t, rho / HELIUM_MASS)
    assert np.allclose(f[:, 0], 1.0)
    assert np.all(np.diff(f, axis=1) < 0)
    # Rogers-Young S(k) at 0.4 g/cm^3, 5000 K: about half the ideal value.
    assert 0.40 < f[1, 3] < 0.52


def test_edges_are_held_not_extrapolated(table):
    n = np.array([0.3 / HELIUM_MASS])
    lo, hi = table.wavelength_bounds
    held = bounded_helium_minus_correction(table, np.array([500.0, lo]), np.array([5000.0]), n)
    assert held[0, 0] == pytest.approx(held[1, 0])
    t_lo = np.exp(table.axes[0][0])
    cold = bounded_helium_minus_correction(table, np.array([5000.0]), np.array([700.0, t_lo]),
                                           np.array([1e20, 1e20]))
    assert cold[0, 0] == pytest.approx(cold[0, 1])
    with pytest.raises(ValueError):
        bounded_helium_minus_correction(table, np.array([5000.0]), np.array([5000.0]),
                                        np.array([5.0 / HELIUM_MASS]))


def test_config_default_and_validation():
    assert DZConfig().dense_helium_free_free == "off"
    from wd_spectra.models.stellar import compute_dz
    with pytest.raises(ValueError, match="dense_helium_free_free"):
        compute_dz(DZConfig(dense_helium_free_free="bogus"), relax_atmosphere=False)
