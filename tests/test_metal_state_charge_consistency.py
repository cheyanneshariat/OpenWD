from types import MappingProxyType

import numpy as np

from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.metals import (
    ATOMIC_MASS_U,
    IONIZATION_ENERGY_EV,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    atmosphere_with_metal_electrons,
    helium_metal_lte_state_from_mass_fractions,
)


def _ground_state_database(elements):
    ions = {}
    for element in elements:
        for charge in range(len(IONIZATION_ENERGY_EV[element]) + 1):
            ions[(element, charge)] = AtomicIon(
                element=element,
                charge=charge,
                atomic_mass_u=ATOMIC_MASS_U[element],
                ionization_energy_ev=(
                    IONIZATION_ENERGY_EV[element][charge]
                    if charge < len(IONIZATION_ENERGY_EV[element])
                    else None
                ),
                levels=(AtomicLevel(1, 0.0, 1.0, "ground"),),
                transitions=(),
            )
    return AtomicDatabase(MappingProxyType(ions))


def test_bulk_he_metal_handoff_uses_replaced_host_nuclei_density():
    atmosphere = gray_helium_atmosphere(120_000.0, 7.0, n_depth=8)
    state = helium_metal_lte_state_from_mass_fractions(
        atmosphere,
        _ground_state_database(("C", "O", "Ne")),
        {"He": 0.329, "C": 0.50, "O": 0.17, "Ne": 0.001},
    )
    host = state.host_ion_number_density
    assert host is not None
    electron_density = state.electron_density.copy()
    host_before = host.copy()

    enriched = atmosphere_with_metal_electrons(atmosphere, state)
    enriched_he = enriched.helium_lte_state
    assert enriched_he is not None
    expected_mean_charge = (host[1] + 2.0 * host[2]) / np.sum(host, axis=0)

    np.testing.assert_allclose(
        enriched_he.mean_ion_charge, expected_mean_charge, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(enriched_he.helium_nuclei_density, np.sum(host, axis=0))
    np.testing.assert_array_equal(state.electron_density, electron_density)
    np.testing.assert_array_equal(state.host_ion_number_density, host_before)
