import numpy as np
import pytest

from wd_spectra.eos import (
    HELIUM_I_LOW_TERM_ENERGY,
    HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT,
    HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    charged_particle_hydrogen_occupation_probability,
    helium_occupation_probability,
    hydrogen_level_distribution,
    hydrogen_occupation_probability,
    hummer_mihalas_helium_lte,
    hummer_mihalas_helium_thermodynamics,
    hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_helium_thermodynamics,
    hummer_mihalas_hydrogen_lte,
    hummer_mihalas_hydrogen_thermodynamics,
    ideal_hydrogen_lte,
    ideal_hydrogen_thermodynamics,
    hydrogen_saha_constant,
)
from wd_spectra.constants import BOLTZMANN, BOHR_RADIUS, PI
from wd_spectra.molecules import (
    molecular_hydrogen_dissociation_constant,
    molecular_hydrogen_ion_dissociation_constant,
)


def test_hydrogen_eos_conserves_charge_particles_and_pressure():
    temperature = np.geomspace(4_000.0, 50_000.0, 30)
    pressure = np.geomspace(1.0e2, 1.0e10, 30)
    state = ideal_hydrogen_lte(temperature, pressure)

    np.testing.assert_allclose(state.electron_density, state.proton_density)
    np.testing.assert_allclose(
        state.neutral_h_density + state.proton_density,
        state.hydrogen_nuclei_density,
        rtol=2e-15,
    )
    reconstructed_pressure = (
        state.neutral_h_density + state.proton_density + state.electron_density
    ) * BOLTZMANN * temperature
    np.testing.assert_allclose(reconstructed_pressure, pressure, rtol=2e-15)


def test_ionization_increases_with_temperature_at_fixed_pressure():
    state = ideal_hydrogen_lte(
        np.array([5_000.0, 10_000.0, 20_000.0, 40_000.0]), 1.0e6
    )
    assert np.all(np.diff(state.ionization_fraction) > 0.0)
    assert state.ionization_fraction[0] < 1.0e-4
    assert state.ionization_fraction[-1] > 0.99


def test_hummer_mihalas_eos_uses_finite_partition_and_conserves_hydrogen():
    temperature = np.array([8_000.0, 12_000.0, 20_000.0])
    pressure = np.array([1.0e6, 3.0e6, 1.0e7])
    state = hummer_mihalas_hydrogen_lte(temperature, pressure)
    assert state.level_population_density is not None
    assert state.level_occupation_probability is not None
    assert state.internal_partition_function is not None
    np.testing.assert_allclose(
        np.sum(state.level_population_density, axis=-1),
        state.neutral_h_density,
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(state.electron_density, state.proton_density)
    np.testing.assert_allclose(
        state.neutral_h_density + state.proton_density,
        state.hydrogen_nuclei_density,
        rtol=3.0e-15,
    )
    assert np.all(np.isfinite(state.internal_partition_function))
    assert np.all(state.internal_partition_function > 0.99)


def test_hummer_mihalas_pressure_includes_adopted_neutral_virial_term():
    temperature = np.array([6_000.0, 8_000.0, 12_000.0])
    pressure = np.array([1.0e8, 3.0e8, 1.0e9])
    state = hummer_mihalas_hydrogen_lte(temperature, pressure)
    assert state.level_population_density is not None
    level = np.arange(1, state.level_population_density.shape[-1] + 1)
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (
            HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            * BOHR_RADIUS
            * (level**2 + 1.0)
        )
        ** 3
    )
    fraction = state.level_population_density / state.neutral_h_density[:, None]
    mean_excluded_volume = np.sum(fraction * excluded_volume, axis=-1)
    reconstructed = BOLTZMANN * temperature * (
        state.neutral_h_density
        + state.proton_density
        + state.electron_density
        + 0.5 * state.neutral_h_density**2 * mean_excluded_volume
    )
    np.testing.assert_allclose(reconstructed, pressure, rtol=2.0e-10)


def test_molecular_hm_eos_conserves_charge_nuclei_and_pressure():
    temperature = np.array([3_500.0, 5_000.0, 8_000.0, 12_000.0])
    pressure = np.array([1.0e6, 1.0e7, 1.0e8, 1.0e8])
    state = hummer_mihalas_hydrogen_lte(
        temperature, pressure, include_molecules=True
    )
    assert state.molecular_hydrogen_density is not None
    assert state.molecular_hydrogen_ion_density is not None
    np.testing.assert_allclose(
        state.electron_density,
        state.proton_density + state.molecular_hydrogen_ion_density,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        state.neutral_h_density
        + state.proton_density
        + 2.0 * state.molecular_hydrogen_density
        + 2.0 * state.molecular_hydrogen_ion_density,
        state.hydrogen_nuclei_density,
        rtol=2.0e-10,
    )
    assert state.level_population_density is not None
    level = np.arange(1, state.level_population_density.shape[-1] + 1)
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (
            HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            * BOHR_RADIUS
            * (level**2 + 1.0)
        )
        ** 3
    )
    fraction = state.level_population_density / state.neutral_h_density[:, None]
    mean_excluded_volume = np.sum(fraction * excluded_volume, axis=-1)
    reconstructed = BOLTZMANN * temperature * (
        state.neutral_h_density
        + state.proton_density
        + state.electron_density
        + state.molecular_hydrogen_density
        + state.molecular_hydrogen_ion_density
        + 0.5 * state.neutral_h_density**2 * mean_excluded_volume
    )
    np.testing.assert_allclose(reconstructed, pressure, rtol=3.0e-9)
    assert state.internal_partition_function is not None
    np.testing.assert_allclose(
        state.proton_density
        * state.electron_density
        / state.neutral_h_density,
        hydrogen_saha_constant(temperature)
        / state.internal_partition_function,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        state.neutral_h_density**2 / state.molecular_hydrogen_density,
        molecular_hydrogen_dissociation_constant(
            temperature,
            atomic_internal_partition_function=state.internal_partition_function,
        ),
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        state.neutral_h_density
        * state.proton_density
        / state.molecular_hydrogen_ion_density,
        molecular_hydrogen_ion_dissociation_constant(
            temperature,
            atomic_internal_partition_function=state.internal_partition_function,
        ),
        rtol=2.0e-9,
    )
    assert state.molecular_hydrogen_density[0] > state.neutral_h_density[0]
    assert state.molecular_hydrogen_density[-1] < state.neutral_h_density[-1]


def test_complex_cool_hydrogen_eos_conserves_charge_nuclei_and_pressure():
    temperature = np.array([4_000.0, 5_000.0, 6_000.0])
    pressure = np.array([1.0e8, 2.5e8, 5.0e8])
    state = hummer_mihalas_hydrogen_lte(
        temperature,
        pressure,
        include_molecules=True,
        include_negative_hydrogen=True,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    assert state.molecular_hydrogen_density is not None
    assert state.molecular_hydrogen_ion_density is not None
    assert state.negative_hydrogen_density is not None
    assert state.trihydrogen_ion_density is not None
    np.testing.assert_allclose(
        state.electron_density + state.negative_hydrogen_density,
        state.proton_density
        + state.molecular_hydrogen_ion_density
        + state.trihydrogen_ion_density,
        rtol=3.0e-9,
    )
    np.testing.assert_allclose(
        state.neutral_h_density
        + state.proton_density
        + state.negative_hydrogen_density
        + 2.0 * state.molecular_hydrogen_density
        + 2.0 * state.molecular_hydrogen_ion_density
        + 3.0 * state.trihydrogen_ion_density,
        state.hydrogen_nuclei_density,
        rtol=3.0e-9,
    )
    assert state.level_population_density is not None
    level = np.arange(1, state.level_population_density.shape[-1] + 1)
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (
            HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            * BOHR_RADIUS
            * (level**2 + 1.0)
        )
        ** 3
    )
    fraction = state.level_population_density / state.neutral_h_density[:, None]
    mean_excluded_volume = np.sum(fraction * excluded_volume, axis=-1)
    particle_density = (
        state.neutral_h_density
        + state.proton_density
        + state.electron_density
        + state.negative_hydrogen_density
        + state.molecular_hydrogen_density
        + state.molecular_hydrogen_ion_density
        + state.trihydrogen_ion_density
        + 0.5 * state.neutral_h_density**2 * mean_excluded_volume
    )
    np.testing.assert_allclose(
        BOLTZMANN * temperature * particle_density,
        pressure,
        rtol=3.0e-9,
    )
    assert np.all(state.trihydrogen_ion_density > 0.0)


def test_molecular_equilibrium_constants_increase_with_temperature():
    temperature = np.array([3_000.0, 5_000.0, 8_000.0, 12_000.0])
    assert np.all(np.diff(molecular_hydrogen_dissociation_constant(temperature)) > 0.0)
    assert np.all(
        np.diff(molecular_hydrogen_ion_dissociation_constant(temperature)) > 0.0
    )


def test_hummer_mihalas_level_distribution_dissolves_high_states_smoothly():
    distribution = hydrogen_level_distribution(
        1.0e19, 1.0e17, 8_000.0, maximum_level=30
    )
    probability = distribution.occupation_probability
    assert np.all(np.diff(probability) <= 0.0)
    assert probability[0] > 0.99
    assert probability[-1] < 1.0e-20
    np.testing.assert_allclose(
        np.sum(distribution.population_fraction), 1.0, rtol=1.0e-15
    )


def test_neutral_occupation_uses_configured_hard_sphere_radius():
    neutral_density = 3.0e18
    level = np.array([1.0, 3.0, 6.0])
    probability = hydrogen_occupation_probability(
        neutral_density,
        0.0,
        6_000.0,
        level,
    )
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (
            HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            * BOHR_RADIUS
            * (level**2 + 1.0)
        )
        ** 3
    )
    np.testing.assert_allclose(
        probability,
        np.exp(-neutral_density * excluded_volume),
        rtol=2.0e-15,
    )


def test_qmhd_correlations_increase_high_level_survival():
    level = np.array([1.0, 3.0, 6.0, 10.0])
    holtsmark = charged_particle_hydrogen_occupation_probability(
        1.0e17, level
    )
    correlated = charged_particle_hydrogen_occupation_probability(
        1.0e17, level, 10_000.0
    )
    assert np.all(correlated >= holtsmark)
    assert correlated[-1] > 2.0 * holtsmark[-1]
    np.testing.assert_allclose(correlated[0], holtsmark[0], rtol=2.0e-8)


@pytest.mark.parametrize("ionic_charge", [1.0, 2.0, 6.0])
@pytest.mark.parametrize("temperature", [None, 10_000.0, 60_000.0])
def test_qmhd_hydrogenic_charge_scaling_matches_tlusty_wn(
    ionic_charge, temperature
):
    # TLUSTY reference manual II (2017), eqs. 107--108, with BERGFC=1.
    # Use its independently tabulated prefactor, not the implementation's
    # physical-constant expression. TLUSTY 200 hardcoded the now-obsolete
    # factor of two; it is NOT the undoubled reference convention.
    density = np.geomspace(1e12, 1e20, 9)[:, None]
    level = np.array([1., 3., 8., 10., 11., 20.])[None, :]
    correction = np.where(level <= 3, 1., 16 * level / (3 * (level + 1)**2))
    beta = 8.59e14 * ionic_charge**3 * density**(-2/3) * correction / level**4
    a = 0. if temperature is None else np.minimum(
        0.09 * density**(1/6) / np.sqrt(temperature), 0.8
    )
    x = (1 + a)**3.15
    f = 0.1402 * (x + 4 * (ionic_charge - 1) * a**3) * beta**3 / (
        1 + 0.1285 * x * beta**1.5
    )
    expected = f / (1 + f)
    actual = charged_particle_hydrogen_occupation_probability(
        density, level, temperature, ionic_charge=ionic_charge
    )
    # Our modern constants give 8.580024e14 rather than the rounded 8.59e14;
    # the largest propagated relative difference is 0.35% in the beta**3 limit.
    np.testing.assert_allclose(actual, expected, rtol=0.004, atol=0.)


def test_hummer_mihalas_thermodynamics_is_physical():
    result = hummer_mihalas_hydrogen_thermodynamics(
        np.array([8_000.0, 12_000.0, 20_000.0]),
        np.array([1.0e6, 3.0e6, 1.0e7]),
    )
    assert np.all(result.specific_heat_constant_pressure > 0.0)
    assert np.all(result.density_temperature_derivative > 0.0)
    assert np.all(result.adiabatic_temperature_gradient > 0.0)
    assert np.all(result.adiabatic_temperature_gradient < 0.4)


def test_molecular_hm_thermodynamics_is_physical():
    result = hummer_mihalas_hydrogen_thermodynamics(
        np.array([3_500.0, 5_000.0, 8_000.0]),
        np.array([1.0e6, 1.0e7, 1.0e8]),
        include_molecules=True,
    )
    assert np.all(result.specific_heat_constant_pressure > 0.0)
    assert np.all(result.density_temperature_derivative > 0.0)
    assert np.all(result.adiabatic_temperature_gradient > 0.0)
    assert np.all(result.adiabatic_temperature_gradient < 0.4)


def test_ideal_hydrogen_thermodynamic_derivatives_match_finite_differences():
    temperature = np.array([8_000.0, 12_000.0, 20_000.0])
    pressure = np.array([1.0e5, 3.0e5, 1.0e6])
    result = ideal_hydrogen_thermodynamics(temperature, pressure)
    epsilon = 1.0e-5
    cool = ideal_hydrogen_lte(temperature * np.exp(-epsilon), pressure)
    hot = ideal_hydrogen_lte(temperature * np.exp(epsilon), pressure)
    numerical_q = -(
        np.log(hot.mass_density) - np.log(cool.mass_density)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result.density_temperature_derivative, numerical_q, rtol=2.0e-9
    )
    assert np.all(result.specific_heat_constant_pressure > 0.0)
    assert np.all((result.adiabatic_temperature_gradient > 0.0))
    assert np.all((result.adiabatic_temperature_gradient < 0.4))


def test_helium_eos_conserves_nuclei_charge_and_pressure():
    temperature = np.geomspace(8_000.0, 80_000.0, 16)
    pressure = np.geomspace(1.0e4, 1.0e10, 16)
    state = hummer_mihalas_helium_lte(temperature, pressure)
    np.testing.assert_allclose(
        state.neutral_he_density
        + state.singly_ionized_he_density
        + state.doubly_ionized_he_density,
        state.helium_nuclei_density,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        state.electron_density,
        state.singly_ionized_he_density + 2.0 * state.doubly_ionized_he_density,
        rtol=3.0e-14,
    )
    reconstructed_pressure = BOLTZMANN * temperature * (
        state.helium_nuclei_density + state.electron_density
    )
    np.testing.assert_allclose(reconstructed_pressure, pressure, rtol=3.0e-14)


def test_helium_n2_term_populations_follow_hm_boltzmann_distribution():
    temperature = np.array([12_000.0, 20_000.0, 35_000.0])
    state = hummer_mihalas_helium_lte(temperature, 1.0e7)
    expected_fraction = (
        HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * state.neutral_level_occupation_probability
        * np.exp(
            -HELIUM_I_LOW_TERM_ENERGY
            / (BOLTZMANN * temperature[:, np.newaxis])
        )
        / state.neutral_partition_function[:, np.newaxis]
    )
    np.testing.assert_allclose(
        state.neutral_level_population_density
        / state.neutral_he_density[:, np.newaxis],
        expected_fraction,
        rtol=2.0e-15,
    )
    assert np.all(np.diff(state.neutral_level_population_density[:, 3]) > 0.0)


def test_neutral_helium_perturbations_dissolve_rydberg_levels_first():
    levels = np.array([1.0, 2.0, 5.0, 10.0])
    probability = helium_occupation_probability(
        1.0e20, 1.0e16, 12_000.0, levels
    )
    assert np.all(np.diff(probability) <= 0.0)
    assert probability[0] > 0.99
    assert probability[-1] < 2.0e-4


def test_helium_thermodynamics_is_physical():
    result = hummer_mihalas_helium_thermodynamics(
        np.array([12_000.0, 20_000.0, 35_000.0]),
        np.array([1.0e6, 1.0e7, 1.0e8]),
    )
    assert np.all(result.specific_heat_constant_pressure > 0.0)
    assert np.all(result.density_temperature_derivative > 0.0)
    assert np.all(result.adiabatic_temperature_gradient > 0.0)
    assert np.all(result.adiabatic_temperature_gradient < 0.4)


def test_mixed_hydrogen_helium_eos_conserves_pressure_nuclei_and_charge():
    temperature = np.array([10_000.0, 20_000.0, 40_000.0])
    pressure = np.array([1.0e6, 1.0e7, 1.0e8])
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature, pressure, -2.0
    )
    hydrogen = state.hydrogen_lte_state
    helium = state.helium_lte_state
    np.testing.assert_allclose(
        state.hydrogen_nuclei_density / state.helium_nuclei_density,
        1.0e-2,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        hydrogen.neutral_h_density + hydrogen.proton_density,
        state.hydrogen_nuclei_density,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        helium.neutral_he_density
        + helium.singly_ionized_he_density
        + helium.doubly_ionized_he_density,
        state.helium_nuclei_density,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        state.electron_density,
        hydrogen.proton_density
        + helium.singly_ionized_he_density
        + 2.0 * helium.doubly_ionized_he_density,
        rtol=2.0e-11,
    )
    reconstructed_pressure = BOLTZMANN * temperature * (
        state.hydrogen_nuclei_density
        + state.helium_nuclei_density
        + state.electron_density
    )
    np.testing.assert_allclose(reconstructed_pressure, pressure, rtol=2.0e-14)
    np.testing.assert_allclose(
        np.sum(hydrogen.level_population_density, axis=-1),
        hydrogen.neutral_h_density,
        rtol=3.0e-14,
    )


def test_mixed_eos_approaches_pure_composition_limits():
    temperature = np.array([20_000.0])
    pressure = np.array([1.0e4])
    mixed_he = hummer_mihalas_hydrogen_helium_lte(
        temperature, pressure, -14.0
    )
    pure_he = hummer_mihalas_helium_lte(
        temperature, pressure, correlated_microfields=True
    )
    np.testing.assert_allclose(
        mixed_he.mass_density, pure_he.mass_density, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        mixed_he.electron_density, pure_he.electron_density, rtol=2.0e-11
    )

    mixed_h = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        pressure,
        14.0,
        hydrogen_neutral_radius_scale=HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
        correlated_microfields=False,
    )
    pure_h = hummer_mihalas_hydrogen_lte(
        temperature, pressure, correlated_microfields=False
    )
    np.testing.assert_allclose(
        mixed_h.mass_density, pure_h.mass_density, rtol=2.0e-10
    )
    np.testing.assert_allclose(
        mixed_h.electron_density, pure_h.electron_density, rtol=2.0e-10
    )


def test_mixed_hydrogen_helium_thermodynamics_is_physical():
    result = hummer_mihalas_hydrogen_helium_thermodynamics(
        np.array([12_000.0, 20_000.0, 35_000.0]),
        np.array([1.0e6, 1.0e7, 1.0e8]),
        -2.0,
    )
    assert np.all(result.specific_heat_constant_pressure > 0.0)
    assert np.all(result.density_temperature_derivative > 0.0)
    assert np.all(result.adiabatic_temperature_gradient > 0.0)
    assert np.all(result.adiabatic_temperature_gradient < 0.4)


@pytest.mark.parametrize("temperature, pressure", [(0.0, 1.0), (1.0, 0.0)])
def test_eos_rejects_nonphysical_inputs(temperature, pressure):
    with pytest.raises(ValueError):
        ideal_hydrogen_lte(temperature, pressure)
