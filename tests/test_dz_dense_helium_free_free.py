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
        "d6337a48a729015be117522f63b525ad8d585e95bea3427b5f8f0595b4a0f4e9"
    )


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
    cold = bounded_helium_minus_correction(table, np.array([5000.0]), np.array([2000.0, t_lo]),
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
