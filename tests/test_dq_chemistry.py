"""DQ material tests run without an atmosphere iteration or external data."""

from dataclasses import replace
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.carbon_molecular import (
    C2CrossSectionTable,
    carbon_atomic_pool,
    carbon_helium_lte_state,
    c2_dissociation_constant,
)
from wd_spectra.models.common import ModelData
from wd_spectra.metals import read_stout_atomic_database
from wd_spectra.constants import BOLTZMANN


@pytest.fixture
def table():
    return C2CrossSectionTable(
        np.array([3000, 5000, 10000.0]),
        np.array([1000, 100000.0]),
        np.ones((3, 2)) * 1e-18,
        np.array([1000, 100000.0]),
        np.array([500, 500000.0]),
    )


def test_atomic_pool_preserves_nuclei_and_ionized_fraction():
    n = np.geomspace(1e3, 1e25, 31)
    for f in (0.0, 1e-10, 0.1, 1.0):
        k = 1e14
        atoms = carbon_atomic_pool(n, f, k)
        np.testing.assert_allclose(atoms + 2 * (f * atoms) ** 2 / k, n, rtol=1e-14)


def test_c2_mass_action_uses_measured_ground_state_dissociation_energy():
    from wd_spectra.carbon_molecular import C2_DISSOCIATION_ENERGY_EV
    from wd_spectra.constants import PLANCK, LIGHT_SPEED, HYDROGEN_MASS

    ev = 1.602176634e-12
    energy = 50390.5 * PLANCK * LIGHT_SPEED
    np.testing.assert_allclose(C2_DISSOCIATION_ENERGY_EV, energy / ev, rtol=1e-15)
    t = np.array([4000.0, 6000.0, 10000.0])
    translation = (2 * np.pi * 6 * HYDROGEN_MASS * BOLTZMANN * t / PLANCK**2) ** 1.5
    k = c2_dissociation_constant(t, 9.0, 2000.0)
    np.testing.assert_allclose(
        k, translation * 81 / 2000 * np.exp(-energy / (BOLTZMANN * t)), rtol=1e-13
    )
    old = translation * 81 / 2000 * np.exp(-6.297 * ev / (BOLTZMANN * t))
    # Lower binding energy increases dissociation, not formation.
    assert np.all(k > old)
    np.testing.assert_allclose(
        k / old, np.exp((6.297 * ev - energy) / (BOLTZMANN * t)), rtol=1e-13
    )


def test_table_refuses_silent_temperature_clamping_and_nonfinite_data(table):
    with pytest.raises(ValueError, match="clamping"):
        table.cross_section_for_wavelength_temperature([5000], [500])
    with pytest.raises(ValueError):
        table.molecular_partition_function([np.nan])
    with pytest.raises(ValueError):
        replace(table, partition_function=np.array([1.0, np.nan]))
    assert np.all(
        table.cross_section_for_wavelength_temperature([2000, 5000, 11000], [7000])[
            [0, 2]
        ]
        == 0
    )


@pytest.mark.parametrize("teff", [6000.0, 8000.0, 10000.0])
def test_carbon_molecules_charge_and_pressure_are_coupled(table, teff):
    seed = gray_helium_atmosphere(teff, 8, n_depth=12)
    db = read_stout_atomic_database(
        ModelData.default().stout, elements=("C",), maximum_charge=3
    )
    a, c, c2 = carbon_helium_lte_state(seed, db, -4.5, table)
    ions = c.ion_number_density["C"]
    nhe = a.helium_lte_state.helium_nuclei_density
    np.testing.assert_allclose(
        np.sum(ions, axis=0) + 2 * c2, 10**-4.5 * nhe, rtol=1e-12
    )
    charge = (
        np.sum(np.arange(len(ions))[:, None] * ions, axis=0)
        + c.host_ion_number_density[1]
        + 2 * c.host_ion_number_density[2]
    )
    np.testing.assert_allclose(charge, a.electron_density, rtol=1e-9)
    pressure = (
        (nhe + np.sum(ions, axis=0) + c2 + a.electron_density)
        * BOLTZMANN
        * a.temperature
    )
    np.testing.assert_allclose(pressure, a.gas_pressure, rtol=1e-9)
    k = c2_dissociation_constant(
        a.temperature,
        c.partition_function[("C", 0)],
        table.molecular_partition_function(a.temperature),
    )
    np.testing.assert_allclose(ions[0] ** 2 / c2, k, rtol=1e-12)


















def test_table_is_immutable_and_preserves_zero_opacity(table):
    with pytest.raises(ValueError):
        table.cross_section[:] = 0
    zero = replace(table, cross_section=np.zeros_like(table.cross_section))
    assert np.all(
        zero.cross_section_for_wavelength_temperature([4000, 5000], [5000]) == 0
    )












