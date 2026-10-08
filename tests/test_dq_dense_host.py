"""DQ-only bulk-EOS closure; no atmosphere iterations or new downloads."""

from dataclasses import replace
import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.carbon_molecular import (
    C2CrossSectionTable,
    carbon_helium_lte_state,
    _reos_host_state,
)
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from wd_spectra.dense_eos import read_helium_reos3_table
from wd_spectra.metals import read_stout_atomic_database
from wd_spectra.models.common import ModelData
from wd_spectra._dq.base import DQConfig, DQMaterial


@pytest.fixture(scope="module")
def inputs():
    data = ModelData.default()
    atomic = read_stout_atomic_database(data.stout, elements=("C",), maximum_charge=3)
    bulk = read_helium_reos3_table(data.helium_reos3)
    molecular = C2CrossSectionTable(
        np.array([3000.0, 5000.0, 10000.0]),
        np.array([1000.0, 100000.0]),
        np.ones((3, 2)) * 1e-18,
        np.array([1000.0, 100000.0]),
        np.array([500.0, 500000.0]),
    )
    seed = gray_helium_atmosphere(7000, 8, n_depth=8)
    seed = replace(
        seed,
        temperature=np.linspace(4500.0, 14000.0, 8),
        gas_pressure=np.geomspace(1e7, 1e12, 8),
    )
    return data, atomic, bulk, molecular, seed


@pytest.mark.parametrize("abundance", [-8.0, -5.0, -3.0])
def test_dense_host_preserves_charge_nuclei_and_total_pressure(inputs, abundance):
    _, atomic, bulk, molecular, seed = inputs
    a, c, c2 = carbon_helium_lte_state(
        seed, atomic, abundance, molecular, helium_reos3_table=bulk
    )
    he = a.helium_lte_state
    ions = c.ion_number_density["C"]
    np.testing.assert_allclose(
        ions.sum(axis=0) + 2 * c2, 10**abundance * he.helium_nuclei_density, rtol=1e-12
    )
    charge = (
        he.singly_ionized_he_density
        + 2 * he.doubly_ionized_he_density
        + (np.arange(len(ions))[:, None] * ions).sum(axis=0)
    )
    np.testing.assert_allclose(charge, a.electron_density, rtol=1e-9)
    p_host = np.asarray(a.metadata["dq_helium_host_pressure"])
    ne_host = np.asarray(a.metadata["dq_helium_host_electron_density"])
    p_total = (
        p_host
        + (ions.sum(axis=0) + c2 + a.electron_density - ne_host)
        * BOLTZMANN
        * a.temperature
    )
    np.testing.assert_allclose(p_total, a.gas_pressure, rtol=1e-10)
    rho, _, covered = bulk.evaluate(p_host, a.temperature)
    assert np.all(covered)
    np.testing.assert_allclose(he.helium_nuclei_density * HELIUM_MASS, rho, rtol=1e-12)
    repeated = carbon_helium_lte_state(
        a, atomic, abundance, molecular, helium_reos3_table=bulk
    )
    np.testing.assert_allclose(repeated[0].mass_density, a.mass_density, rtol=1e-12)
    np.testing.assert_allclose(repeated[2], c2, rtol=1e-12)


def test_dense_host_tends_to_tabulated_pure_helium(inputs):
    _, atomic, bulk, molecular, seed = inputs
    a, _, _ = carbon_helium_lte_state(
        seed, atomic, -12.0, molecular, helium_reos3_table=bulk
    )
    rho, _, _ = bulk.evaluate(seed.gas_pressure, seed.temperature)
    np.testing.assert_allclose(a.mass_density, rho, rtol=1e-9)


def test_dense_host_never_substitutes_an_out_of_domain_eos(inputs):
    _, _, bulk, _, _ = inputs
    with pytest.raises(ValueError, match="outside supplied EOS domain"):
        _reos_host_state(np.array([1.0]), np.array([1e12]), bulk)




def test_dense_material_uses_the_same_host_eos_for_convection(inputs):
    data, _, bulk, molecular, seed = inputs
    material = DQMaterial(
        DQConfig(helium_eos="reos3", molecular_line_profile="line_bin"), data, molecular
    )
    a = material.with_temperature(seed, seed.temperature)
    thermo = material.thermodynamics(a)
    from wd_spectra.eos import hummer_mihalas_helium_thermodynamics

    expected = hummer_mihalas_helium_thermodynamics(
        a.temperature,
        np.asarray(a.metadata["dq_helium_host_pressure"]),
        helium_reos3_table=bulk,
        correlated_microfields=True,
        neutral_radius_scale=0.5,
    )
    np.testing.assert_array_equal(
        thermo.specific_heat_constant_pressure, expected.specific_heat_constant_pressure
    )
    np.testing.assert_array_equal(
        thermo.adiabatic_temperature_gradient, expected.adiabatic_temperature_gradient
    )
    assert np.all(thermo.specific_heat_constant_pressure > 0)


def test_dense_continuum_uses_updated_eos_and_same_structure_synthesis_path(inputs, tmp_path):
    """Synthetic correction tests coupling, not the physical dense-He model."""
    import json
    from wd_spectra.dense_helium_continuum import DenseHeliumCorrectionTable
    data, _, _, molecular, seed = inputs
    logt = np.linspace(np.log(2000), np.log(40000), 7)
    logn = np.linspace(0, np.log1p(2/.001), 9)
    logw = np.linspace(np.log(1000), np.log(100000), 7)
    factor = np.exp(-.01*logn[None,:,None]*(1+.02*logt[:,None,None])
                    *np.ones((1,1,len(logw))))
    path = tmp_path/"synthetic-not-physical.npz"
    np.savez(path, log_temperature=logt, log1p_density=logn, log_wavelength=logw,
             density_scale_g_cm3=.001, factor=factor,
             metadata_json=json.dumps(dict(schema="dense-He-ff-v1", qualified=False,
                                           scope="synthetic coupling test")))
    config = DQConfig(helium_eos="reos3", molecular_line_profile="line_bin",
                      helium_dense_continuum_path=str(path))
    material = DQMaterial(config, data, molecular)
    a = material.with_temperature(seed, seed.temperature)
    wave = np.array([4339., 6200.])
    calls = []
    evaluate = material.dense_continuum.correction
    def record(w, t, n):
        calls.append((np.array(t), np.array(n)))
        return evaluate(w, t, n)
    material.dense_continuum.correction = record
    structure = material.absorption(a, wave, include_c2=False, structure=True)
    material.spectrum(a, wave, 3, include_c2=False)
    for t, n in calls:
        np.testing.assert_array_equal(t, a.temperature)
        np.testing.assert_array_equal(n, a.helium_lte_state.neutral_he_density)
    assert len(calls) == 2
    changed = material.with_temperature(seed, a.temperature*1.01)
    trial = material.absorption(changed, wave, include_c2=False, structure=True)
    np.testing.assert_array_equal(calls[-1][0], changed.temperature)
    np.testing.assert_array_equal(calls[-1][1], changed.helium_lte_state.neutral_he_density)
    assert not np.array_equal(structure, trial)
    with pytest.raises(ValueError, match="outside supplied table"):
        material.absorption(a, np.array([800.]))
    with pytest.raises(ValueError, match="requires helium_eos"):
        DQMaterial(replace(config, helium_eos="ideal"), data, molecular)
    zero = DenseHeliumCorrectionTable(path).correction(wave, a.temperature, np.zeros(a.n_depth))
    np.testing.assert_array_equal(zero, 1.)


def test_invalid_dense_table_is_not_substituted(inputs, tmp_path):
    data, _, _, molecular, _ = inputs
    with pytest.raises(FileNotFoundError):
        DQMaterial(DQConfig(helium_eos="reos3", helium_dense_continuum_path=str(tmp_path/"missing.npz")), data, molecular)
