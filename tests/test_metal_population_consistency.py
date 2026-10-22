"""Material/population identities; no atmosphere solve or spectral fitting.

Expected Saha factors use the published Blouin coefficients independently.
Opacity checks hold the returned ion density fixed and count the populated
levels, so ionization and opacity errors cannot cancel each other.
"""

from dataclasses import replace
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import metals
from wd_spectra.atmosphere import Atmosphere
from wd_spectra.constants import (
    BOLTZMANN, ELECTRON_MASS, HELIUM_MASS, LIGHT_SPEED, PLANCK,
)
from wd_spectra.d6 import (
    TOPbaseLevelPhotoionization, TOPbasePhotoionizationDatabase,
    topbase_bound_free_mass_absorption_coefficient,
)
from wd_spectra.eos import (
    charged_particle_hydrogen_occupation_probability,
    hummer_mihalas_helium_lte_at_nuclei_density,
)
from wd_spectra.models.common import ModelData


@pytest.fixture(scope="module")
def atomic_data():
    return metals.read_stout_atomic_database(
        ModelData.default().stout, elements=("Na", "Mg", "Al", "Si", "Ca"),
        maximum_charge=2, include_default_supplements=False,
    )


def _restricted_atom(database, element, count=1, maximum_charge=1):
    ions = {}
    for key, ion in database.ions.items():
        if ion.element != element or ion.charge > maximum_charge:
            continue
        # Both native partition methods use the same 0.1-eV cutoff. Only
        # levels actually included in Q are part of a normalization identity.
        selected = [
            level for level in ion.levels
            if np.isfinite(level.energy_wavenumber)
            and level.statistical_weight > 0.0
            and 0.0 <= level.energy_wavenumber
            <= max(ion.ionization_energy_ev - 0.1, 0.0) * metals.EV_TO_WAVENUMBER
        ]
        selected.sort(key=lambda level: level.energy_wavenumber)
        ions[key] = replace(ion, levels=tuple(selected[:count]), transitions=())
    return metals.AtomicDatabase(MappingProxyType(ions))


def _atmosphere(density=(0.01, 0.3, 1.0), temperature=5000.0):
    density = np.asarray(density, dtype=float)
    temperature = np.full(density.size, temperature)
    helium = hummer_mihalas_helium_lte_at_nuclei_density(
        temperature, density / HELIUM_MASS,
        correlated_microfields=True, neutral_radius_scale=0.5,
    )
    pressure = (helium.helium_nuclei_density + helium.electron_density) * BOLTZMANN * temperature
    grid = np.arange(1, density.size + 1, dtype=float)
    return Atmosphere(
        float(temperature[0]), 8.0, grid, grid, temperature, pressure,
        density, np.zeros_like(grid), np.zeros_like(grid), helium.electron_density,
        {"scope": "independent material states"}, helium_lte_state=helium,
    )


def _state(atmosphere, atomic, element, *, occupation=True, dense=True):
    return metals.metal_lte_state(
        atmosphere, atomic, {element: -9.0},
        occupation_probability_partitions=occupation,
        include_dense_helium_ionization=dense,
    )


def _blouin_shift(element, atmosphere):
    # Blouin, Dufour & Allard (2018), Table 2. These test states are inside
    # the published fit domain; no boundary continuation is being tested.
    coefficients = {
        "Na": (-0.52305, -0.62471e-4, 0.04833),
        "Mg": (0.45809, -0.85522e-4, -1.01958),
    }
    a, b, c = coefficients[element]
    density = HELIUM_MASS * atmosphere.helium_lte_state.helium_nuclei_density
    return np.minimum((a + b * atmosphere.temperature) * density + c * density**2, 0.0)


def _saha_translational(temperature):
    return 2.0 * (2.0 * np.pi * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2)**1.5


def _charged_ground(ion, state, temperature):
    effective_n = max(1.0, np.sqrt(13.598434599702 * (ion.charge + 1)**2 / ion.ionization_energy_ev))
    return charged_particle_hydrogen_occupation_probability(
        state.electron_density, effective_n, temperature,
        ionic_charge=ion.charge + 1,
    )


def _expected_level_population(ion, level, state, temperature):
    weight = level.statistical_weight * np.exp(
        -level.energy_wavenumber * metals.WAVENUMBER_TO_ERG / (BOLTZMANN * temperature)
    )
    if state.metal_level_dissolution:
        weight *= metals.metal_rydberg_level_occupation_probability(
            ion, level, state.electron_density, temperature,
            neutral_perturber_number_density=state.neutral_perturber_number_density,
            neutral_perturber_radius_cm=state.neutral_perturber_radius_cm,
        )
    return state.ion_number_density[ion.element][ion.charge] * weight / state.partition_function[(ion.element, ion.charge)]


@pytest.mark.parametrize("element", ["Na", "Mg"])
def test_ground_only_saha_retains_blouin_and_charged_factors(atomic_data, element):
    atmosphere = _atmosphere(density=(1.0e-8, 0.01, 0.3, 1.0))
    atomic = _restricted_atom(atomic_data, element)
    state = _state(atmosphere, atomic, element)
    lower, upper = atomic.ion_stages(element)
    expected = (
        _saha_translational(atmosphere.temperature)
        * upper.levels[0].statistical_weight / lower.levels[0].statistical_weight
        * _charged_ground(upper, state, atmosphere.temperature)
        / _charged_ground(lower, state, atmosphere.temperature)
        * np.exp(-(lower.ionization_energy_ev + _blouin_shift(element, atmosphere))
                 * metals.EV_TO_ERG / (BOLTZMANN * atmosphere.temperature))
    )
    populations = state.ion_number_density[element]
    np.testing.assert_allclose(
        populations[1] * state.electron_density / populations[0], expected,
        rtol=2.0e-8,
    )
    # The within-ion partition must remain absolute, not normalized to w0.
    for ion in (lower, upper):
        direct = ion.occupation_weighted_partition_function(
            atmosphere.temperature, state.electron_density,
            neutral_perturber_number_density=state.neutral_perturber_number_density,
            neutral_perturber_radius_cm=state.neutral_perturber_radius_cm,
        )
        np.testing.assert_allclose(state.partition_function[(element, ion.charge)], direct, rtol=2.0e-8)


@pytest.mark.parametrize("element,dense,occupation", [
    ("Na", False, True), ("Na", True, False), ("Al", True, True),
])
def test_unmodified_saha_control_modes(atomic_data, element, dense, occupation):
    atmosphere = _atmosphere()
    atomic = _restricted_atom(atomic_data, element)
    state = _state(atmosphere, atomic, element, dense=dense, occupation=occupation)
    lower, upper = atomic.ion_stages(element)
    shift = _blouin_shift(element, atmosphere) if dense and element == "Na" else 0.0
    expected = (
        _saha_translational(atmosphere.temperature)
        * state.partition_function[(element, 1)] / state.partition_function[(element, 0)]
        * np.exp(-(lower.ionization_energy_ev + shift) * metals.EV_TO_ERG
                 / (BOLTZMANN * atmosphere.temperature))
    )
    population = state.ion_number_density[element]
    np.testing.assert_allclose(population[1] * state.electron_density / population[0], expected, rtol=2.0e-8)


def test_higher_ionization_ratio_keeps_its_existing_partition_convention(atomic_data):
    atmosphere = _atmosphere(density=(0.01, 0.1), temperature=8000.0)
    atomic = _restricted_atom(atomic_data, "Mg", count=3, maximum_charge=2)
    state = _state(atmosphere, atomic, "Mg")
    singly = atomic.ions[("Mg", 1)]
    population = state.ion_number_density["Mg"]
    expected = (
        _saha_translational(atmosphere.temperature)
        * state.partition_function[("Mg", 2)] / state.partition_function[("Mg", 1)]
        * np.exp(-singly.ionization_energy_ev * metals.EV_TO_ERG / (BOLTZMANN * atmosphere.temperature))
    )
    np.testing.assert_allclose(population[2] * state.electron_density / population[1], expected, rtol=2.0e-8)


def test_first_ionization_retains_differential_excited_state_contributions(atomic_data):
    atmosphere = _atmosphere(density=(1e-5, 0.001, 0.03), temperature=8000.0)
    atomic = _restricted_atom(atomic_data, "Na", count=5)
    state = _state(atmosphere, atomic, "Na")
    neutral, singly = atomic.ion_stages("Na")
    exponential = np.exp(
        -(neutral.ionization_energy_ev + _blouin_shift("Na", atmosphere))
        * metals.EV_TO_ERG / (BOLTZMANN * atmosphere.temperature)
    )
    expected = (
        _saha_translational(atmosphere.temperature)
        * state.partition_function[("Na", 1)] / state.partition_function[("Na", 0)]
        * exponential
    )
    population = state.ion_number_density["Na"]
    actual = population[1] * state.electron_density / population[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-8)
    ground_only = (
        _saha_translational(atmosphere.temperature)
        * singly.levels[0].statistical_weight / neutral.levels[0].statistical_weight
        * _charged_ground(singly, state, atmosphere.temperature)
        / _charged_ground(neutral, state, atmosphere.temperature) * exponential
    )
    assert np.max(np.abs(actual / ground_only - 1.0)) > 0.05


@pytest.mark.parametrize("element", ["Na", "Si"])
def test_included_levels_sum_to_each_ion_and_charge_is_closed(atomic_data, element):
    atmosphere = _atmosphere(density=(0.001, 0.03, 0.3))
    atomic = _restricted_atom(atomic_data, element, count=5)
    state = _state(atmosphere, atomic, element)
    for ion in atomic.ion_stages(element):
        populations = [
            metals.metal_lte_level_population(ion, level, state, atmosphere.temperature)
            for level in ion.levels
        ]
        np.testing.assert_allclose(
            np.sum(populations, axis=0), state.ion_number_density[element][ion.charge],
            rtol=2.0e-8,
        )
    ion_population = state.ion_number_density[element]
    np.testing.assert_allclose(np.sum(ion_population, axis=0), state.element_number_density[element], rtol=1e-13)
    metal_charge = np.sum(np.arange(ion_population.shape[0])[:, None] * ion_population, axis=0)
    host = state.host_ion_number_density
    host_charge = np.sum(np.arange(host.shape[0])[:, None] * host, axis=0)
    np.testing.assert_allclose(state.electron_density, host_charge + metal_charge, rtol=1e-9)


def test_trace_multicharge_population_uses_the_partition_perturber_convention():
    temperature = np.full(4, 15000.0)
    electron_density = np.array([1e12, 1e14, 1e16, 1e18])
    levels = (
        metals.AtomicLevel(1, 0.0, 1.0, "2s2.(1S<0>)"),
        metals.AtomicLevel(2, 79.0 * metals.EV_TO_WAVENUMBER, 100.0, "11p.(1Po<1>)"),
    )
    ion = metals.AtomicIon("Mg", 2, 24.305, 80.0, levels, ())
    partition = ion.occupation_weighted_partition_function(temperature, electron_density)
    population = np.zeros((3, 4))
    population[2] = 0.5 * electron_density
    state = metals.MetalLTEState(
        reference_species="He", log_number_abundance={"Mg": -9.0},
        element_number_density={"Mg": population[2]}, ion_number_density={"Mg": population},
        partition_function={("Mg", 2): partition}, electron_density=electron_density,
        metal_electron_density=electron_density, metal_level_dissolution=True,
    )
    # Here the ionic moment differs from ne, but Q was built with ne.
    np.testing.assert_allclose(metals.metal_ionic_microfield_perturber_density(state), np.sqrt(2.0) * electron_density)
    level_population = sum(
        metals.metal_lte_level_population(ion, level, state, temperature) for level in levels
    )
    np.testing.assert_allclose(level_population, population[2], rtol=2e-13)


@pytest.mark.parametrize("element,count", [("Na", 1), ("Si", 5)])
def test_verner_ground_term_uses_normalized_populated_levels(atomic_data, element, count):
    atmosphere = _atmosphere()
    atomic = _restricted_atom(atomic_data, element, count=count)
    state = _state(atmosphere, atomic, element)
    ion = atomic.ions[(element, 0)]
    photo = metals.read_verner_photoionization_database(
        ModelData.default().verner_photoionization, elements=(element,), maximum_charge=0,
    )
    wave = np.array([1000.0, 1100.0])
    energy = PLANCK * LIGHT_SPEED / (wave * 1e-8) / metals.EV_TO_ERG
    ground_population = sum(
        _expected_level_population(ion, level, state, atmosphere.temperature)
        for level in metals.ground_term_levels(ion)
    )
    stimulated = -np.expm1(-energy[:, None] * metals.EV_TO_ERG / (BOLTZMANN * atmosphere.temperature))
    expected = photo.fits[(element, 0)].cross_section(energy)[:, None] * ground_population * stimulated / atmosphere.mass_density
    actual = metals.metal_bound_free_mass_absorption_coefficient(atmosphere, wave, atomic, state, photo)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("element,charge,table_argument", [
    ("Mg", 1, "mg_ii_he_profile_table"),
    ("Ca", 0, "ca_i_he_profile_table"),
    ("Ca", 1, "ca_ii_he_profile_table"),
    ("Na", 0, "na_i_he_profile_table"),
])
@pytest.mark.parametrize("profile_dissolution", [True, False])
def test_special_unified_profile_uses_normalized_ground_population(
    atomic_data, element, charge, table_argument, profile_dissolution,
):
    # No ordinary transitions: the table's constant cross section isolates
    # the special branch's lower population, independent of profile shape.
    atmosphere = _atmosphere(density=(0.01, 0.1, 0.3))
    atomic = _restricted_atom(atomic_data, element)
    state = _state(atmosphere, atomic, element)
    wave = np.array([3900.0, 4100.0])
    cross_section = 1.0e-18
    table = SimpleNamespace(
        densities=np.array([1e20]), temperatures=np.empty(0), lines=(),
        cross_section=lambda wavelength, density, **kwargs: np.full(
            (wavelength.size, density.size), cross_section,
        ),
    )
    options = {
        table_argument: table,
        "include_rydberg_dissolution": profile_dissolution,
        "rydberg_dissolution_elements": None if profile_dissolution else (),
    }
    stimulated = -np.expm1(
        -PLANCK * LIGHT_SPEED / (wave[:, None] * 1e-8 * BOLTZMANN * atmosphere.temperature)
    )
    expected = (
        cross_section * state.ion_number_density[element][charge]
        * stimulated / atmosphere.mass_density
    )
    actual = metals.metal_line_mass_absorption_coefficient(
        atmosphere, wave, atomic, state, **options,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)

    # A prescribed population scaling must be linear at fixed material and
    # profile inputs. This is not an assertion of charge closure after scaling.
    scaled_populations = {key: value.copy() for key, value in state.ion_number_density.items()}
    scaled_populations[element][charge] *= 3.0
    scaled = replace(state, ion_number_density=scaled_populations)
    changed = metals.metal_line_mass_absorption_coefficient(
        atmosphere, wave, atomic, scaled, **options,
    )
    np.testing.assert_allclose(changed, 3.0 * actual, rtol=2e-13, atol=0.0)


@pytest.mark.parametrize("level_offset,parent_excitation,occupation", [
    (0, 0.0, True), (1, 0.0, True), (1, 2.0, True), (1, 2.0, False),
])
def test_topbase_section_identity_and_channel_threshold_do_not_define_population(
    atomic_data, level_offset, parent_excitation, occupation,
):
    atmosphere = _atmosphere(density=(0.001, 0.03, 0.1))
    atomic = _restricted_atom(atomic_data, "Na", count=4)
    state = _state(atmosphere, atomic, "Na", occupation=occupation)
    ion = atomic.ions[("Na", 0)]
    level = ion.levels[level_offset]
    excitation = level.energy_wavenumber / metals.EV_TO_WAVENUMBER
    channel_threshold = ion.ionization_energy_ev - excitation + parent_excitation
    section = TOPbaseLevelPhotoionization(
        "Na", 0, 98765, excitation, level.statistical_weight,
        channel_threshold, channel_threshold,
        np.array([channel_threshold, 20.0]), np.full(2, 1e-18),
    )
    database = TOPbasePhotoionizationDatabase((section,), "synthetic constant cross section")
    wave = np.array([1000.0, 1100.0])
    energy = PLANCK * LIGHT_SPEED / (wave * 1e-8) / metals.EV_TO_ERG
    lower_population = _expected_level_population(ion, level, state, atmosphere.temperature)
    stimulated = -np.expm1(-energy[:, None] * metals.EV_TO_ERG / (BOLTZMANN * atmosphere.temperature))
    expected = section.cross_section(energy)[:, None] * lower_population * stimulated / atmosphere.mass_density
    actual = topbase_bound_free_mass_absorption_coefficient(atmosphere, wave, state, database)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=0.0)


def test_neutral_occupation_retains_low_levels_and_suppresses_rydberg_levels():
    cutoff = metals.NEUTRAL_HARD_SPHERE_MINIMUM_LEVEL_RADIUS_CM
    radius = np.array([0., np.nextafter(cutoff, 0.), cutoff, 2. * cutoff])
    density = {"He": np.full(4, 1e21)}
    perturber = {"He": 0.5 * metals.BOHR_RADIUS_CM}
    actual = metals.metal_neutral_hard_sphere_occupation_probability(radius, density, perturber)
    np.testing.assert_array_equal(actual[:2], np.ones(2))
    expected = np.exp(-4. * np.pi / 3. * density["He"][2:] * (radius[2:] + perturber["He"])**3)
    np.testing.assert_allclose(actual[2:], expected, rtol=1e-14)
