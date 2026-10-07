"""Independent Saha/conservation checks for a highly ionized mixed EOS."""
import numpy as np
import pytest

from wd_spectra.constants import (
    BOLTZMANN, ELECTRON_MASS, HELIUM_MASS, HYDROGEN_IONIZATION_ENERGY,
    HYDROGEN_MASS, PI, PLANCK,
)
from wd_spectra.eos import hummer_mihalas_hydrogen_helium_lte


SURFACE_TEMPERATURE = 50453.78510442453


def saha_constant_from_physical_constants(temperature):
    """Ground-state degeneracy and electron spin cancel in this convention."""
    return (2*PI*ELECTRON_MASS*BOLTZMANN*temperature/PLANCK**2)**1.5 * np.exp(
        -HYDROGEN_IONIZATION_ENERGY/(BOLTZMANN*temperature)
    )


@pytest.mark.parametrize("log_hydrogen_to_helium", [2., -2.])
def test_highly_ionized_mixture_retains_saha_neutral_tail_and_conservation(log_hydrogen_to_helium):
    # First point is the actual DAO gray surface where material probes became
    # noisy. The dilute hot points retain neutral tails below machine epsilon
    # as fractions of all H nuclei, while their absolute densities are finite.
    temperature = np.array([SURFACE_TEMPERATURE, 60000., 1.e6])
    pressure = np.array([10., 1.e-6, 1.e-8])
    state = hummer_mihalas_hydrogen_helium_lte(temperature, pressure, log_hydrogen_to_helium)
    hydrogen, helium = state.hydrogen_lte_state, state.helium_lte_state
    neutral = hydrogen.neutral_h_density
    assert np.all(np.isfinite(neutral)) and np.all(neutral > 0)
    assert np.all(neutral/state.hydrogen_nuclei_density < 1.e-5)
    assert neutral[-1]/state.hydrogen_nuclei_density[-1] < np.finfo(float).eps
    assert np.all(hydrogen.level_population_density > 0)

    # Check the physical equilibrium equation by products, without evaluating
    # an ionization sigmoid or deriving neutral H from a complementary fraction.
    saha_balance = hydrogen.proton_density*state.electron_density*hydrogen.internal_partition_function
    np.testing.assert_allclose(
        neutral*saha_constant_from_physical_constants(temperature), saha_balance, rtol=5.e-12, atol=0,
    )
    np.testing.assert_allclose(
        neutral+hydrogen.proton_density, state.hydrogen_nuclei_density, rtol=2.e-14, atol=0,
    )
    np.testing.assert_allclose(
        np.sum(hydrogen.level_population_density, axis=-1), neutral, rtol=5.e-12, atol=0,
    )
    np.testing.assert_allclose(
        helium.neutral_he_density+helium.singly_ionized_he_density+helium.doubly_ionized_he_density,
        state.helium_nuclei_density, rtol=2.e-14, atol=0,
    )
    np.testing.assert_allclose(
        hydrogen.proton_density+helium.singly_ionized_he_density+2*helium.doubly_ionized_he_density,
        state.electron_density, rtol=2.e-11, atol=0,
    )
    np.testing.assert_allclose(
        BOLTZMANN*temperature*(state.hydrogen_nuclei_density+state.helium_nuclei_density+state.electron_density),
        pressure, rtol=2.e-14, atol=0,
    )
    np.testing.assert_allclose(
        state.hydrogen_nuclei_density/state.helium_nuclei_density, 10**log_hydrogen_to_helium, rtol=2.e-14, atol=0,
    )
    np.testing.assert_allclose(
        HYDROGEN_MASS*state.hydrogen_nuclei_density+HELIUM_MASS*state.helium_nuclei_density,
        state.mass_density, rtol=2.e-14, atol=0,
    )


def test_neutral_and_partly_ionized_mixture_satisfies_saha_and_conservation():
    # The complementary fraction also serves the neutral and transition
    # regimes. These warm atomic states cover both sides of that transition.
    temperature = np.array([10000., 20000., 40000.])
    pressure = 1.e8
    state = hummer_mihalas_hydrogen_helium_lte(temperature, pressure, 2.)
    hydrogen, helium = state.hydrogen_lte_state, state.helium_lte_state
    neutral_fraction = hydrogen.neutral_h_density/state.hydrogen_nuclei_density
    assert neutral_fraction[0] > .99
    assert neutral_fraction[-1] < .5
    assert np.all(np.isfinite(neutral_fraction))
    assert np.all((neutral_fraction > 0) & (neutral_fraction < 1))

    np.testing.assert_allclose(
        hydrogen.neutral_h_density*saha_constant_from_physical_constants(temperature),
        hydrogen.proton_density*state.electron_density*hydrogen.internal_partition_function,
        rtol=5.e-12, atol=0,
    )
    np.testing.assert_allclose(
        hydrogen.neutral_h_density+hydrogen.proton_density,
        state.hydrogen_nuclei_density, rtol=2.e-14, atol=0,
    )
    np.testing.assert_allclose(
        hydrogen.proton_density+helium.singly_ionized_he_density+2*helium.doubly_ionized_he_density,
        state.electron_density, rtol=2.e-11, atol=0,
    )
    np.testing.assert_allclose(
        BOLTZMANN*temperature*(state.hydrogen_nuclei_density+state.helium_nuclei_density+state.electron_density),
        pressure, rtol=2.e-14, atol=0,
    )


@pytest.mark.parametrize("log_temperature_step", [8.60552030043332e-9, 8.60552030043332e-10])
def test_surface_neutral_temperature_response_satisfies_differentiated_saha_balance(log_temperature_step):
    # These are the surface ln(T) displacements in the recorded DAO Newton
    # direction at physical probe sizes 1e-4 and 1e-5. Pressure/composition stay
    # fixed as in the joint hot solver's material response.
    temperature = SURFACE_TEMPERATURE*np.exp(np.array([-1., 1.])*log_temperature_step)
    state = hummer_mihalas_hydrogen_helium_lte(temperature, 10., 2.)
    hydrogen = state.hydrogen_lte_state
    measured = np.log(hydrogen.neutral_h_density[1]/hydrogen.neutral_h_density[0])/(2*log_temperature_step)
    # Differentiate n_H0 K(T) = n_p n_e U_H. The explicit derivative of the
    # translational/ionization factor is independent of the neutral-tail code.
    material_response = (
        np.log(hydrogen.proton_density[1]/hydrogen.proton_density[0])
        + np.log(state.electron_density[1]/state.electron_density[0])
        + np.log(hydrogen.internal_partition_function[1]/hydrogen.internal_partition_function[0])
    )/(2*log_temperature_step)
    saha_response = 1.5+HYDROGEN_IONIZATION_ENERGY/(BOLTZMANN*SURFACE_TEMPERATURE)
    np.testing.assert_allclose(measured, material_response-saha_response, rtol=0, atol=3.e-6)
