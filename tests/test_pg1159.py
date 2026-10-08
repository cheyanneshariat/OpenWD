from types import MappingProxyType

import numpy as np

from wd_spectra.light_metal_nlte import (
    LightMetalNLTEState,
    ReducedLightMetalLevelState,
)
from wd_spectra.atmosphere import Atmosphere
from wd_spectra.constants import BOLTZMANN
from wd_spectra.eos import hummer_mihalas_helium_lte
from wd_spectra.helium_nlte import CoupledHeliumNLTEState
from wd_spectra.metals import (
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    MetalLTEState,
)
from wd_spectra.pg1159 import (
    PG1159NLTEModel,
    PG1159NLTEState,
    _add_missing_lte_ion_ladders,
    _align_reduced_level_state,
    _damp_reduced_level_state,
    _finite_departure_ratio,
    _formal_population_transition_keys,
    _formal_spectrum_transition_keys,
    _ion_departures_with_explicit_stage_closure,
    _maximum_coupled_helium_population_change,
    _coupled_helium_population_change_diagnostic,
    _replace_light_metal_elements,
    _pg1159_nlte_charge_feedback,
)


def test_pg1159_atmosphere_population_effort_tightens_near_convergence():
    model = PG1159NLTEModel(
        helium_model=object(),
        atomic_database=AtomicDatabase(MappingProxyType({}), "test"),
        metal_population_iterations=30,
        metal_population_minimum_iterations=3,
        metal_population_relative_tolerance=3.0e-3,
        adaptive_atmosphere_population_effort=True,
        atmosphere_coarse_population_iterations=6,
        atmosphere_coarse_population_relative_tolerance=2.0e-2,
        atmosphere_population_tightening_threshold=1.5e-2,
    )

    coarse = model.atmosphere_iteration_model(1, None)
    assert coarse.metal_population_iterations == 6
    assert coarse.metal_population_relative_tolerance == 2.0e-2
    assert not coarse.atmosphere_population_accuracy_final

    final = model.atmosphere_iteration_model(2, 1.0e-2)
    assert final.metal_population_iterations == 30
    assert final.metal_population_relative_tolerance == 3.0e-3
    assert final.atmosphere_population_accuracy_final

    # A small local integral-balance residual does not imply that the global
    # flux mode is converged.  Retain coarse populations while the proposed
    # temperature correction is still large.
    global_mode_coarse = model.atmosphere_iteration_model(2, 3.0e-2, 1.0e-3)
    assert global_mode_coarse.metal_population_iterations == 6
    assert not global_mode_coarse.atmosphere_population_accuracy_final

    both_final = model.atmosphere_iteration_model(2, 1.0e-2, 1.0e-3)
    assert both_final.atmosphere_population_accuracy_final


def test_pg1159_nlte_charge_feedback_preserves_fixed_pressure():
    temperature = np.asarray((50_000.0, 80_000.0))
    gas_pressure = np.asarray((1.0e5, 3.0e5))
    helium = hummer_mihalas_helium_lte(temperature, gas_pressure)
    atmosphere = Atmosphere(
        effective_temperature=60_000.0,
        logg=8.0,
        rosseland_optical_depth=np.asarray((1.0e-3, 1.0)),
        column_mass=gas_pressure / 1.0e8,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=helium.mass_density,
        neutral_h_density=np.zeros(2),
        proton_density=np.zeros(2),
        electron_density=helium.electron_density,
        helium_lte_state=helium,
        metadata={},
    )
    helium_density = helium.helium_nuclei_density
    host_lte = np.stack(
        (
            0.05 * helium_density,
            0.75 * helium_density,
            0.20 * helium_density,
        )
    )
    carbon_density = 0.1 * helium_density
    carbon_lte = np.stack(
        (
            0.10 * carbon_density,
            0.70 * carbon_density,
            0.20 * carbon_density,
        )
    )
    metal = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({"C": -1.0}),
        element_number_density=MappingProxyType({"C": carbon_density}),
        ion_number_density=MappingProxyType({"C": carbon_lte}),
        partition_function=MappingProxyType({}),
        electron_density=host_lte[1] + 2.0 * host_lte[2]
        + carbon_lte[1] + 2.0 * carbon_lte[2],
        metal_electron_density=carbon_lte[1] + 2.0 * carbon_lte[2],
        host_ion_number_density=host_lte,
        composition_mode="bulk",
        mass_fraction=MappingProxyType({"He": 0.77, "C": 0.23}),
        total_mass_density=helium.mass_density * 1.3,
    )
    helium_state = CoupledHeliumNLTEState(
        neutral_population_density=(0.01 * helium_density)[:, None],
        singly_ionized_population_density=(0.29 * helium_density)[:, None],
        doubly_ionized_he_density=0.70 * helium_density,
        lte_neutral_population_density=(0.05 * helium_density)[:, None],
        lte_singly_ionized_population_density=(0.75 * helium_density)[:, None],
        lte_doubly_ionized_he_density=0.20 * helium_density,
        neutral_departure_coefficient=np.full((2, 1), 0.2),
        singly_ionized_departure_coefficient=np.full((2, 1), 0.29 / 0.75),
        doubly_ionized_departure_coefficient=np.full(2, 3.5),
        iterations=1,
        converged=True,
        maximum_relative_population_change=0.0,
        metadata={},
    )
    carbon_nlte = np.stack(
        (
            0.01 * carbon_density,
            0.19 * carbon_density,
            0.80 * carbon_density,
        )
    )
    light = LightMetalNLTEState(
        ion_number_density=MappingProxyType({"C": carbon_nlte}),
        lte_ion_number_density=MappingProxyType({"C": carbon_lte}),
        ion_departure_coefficient=MappingProxyType(
            {
                ("C", charge): carbon_nlte[charge] / carbon_lte[charge]
                for charge in range(3)
            }
        ),
        photoionization_rate=MappingProxyType({}),
        radiative_recombination_rate=MappingProxyType({}),
        collisional_ionization_rate=MappingProxyType({}),
        three_body_recombination_rate=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    previous = PG1159NLTEState(
        helium_state=helium_state,
        metal_state=metal,
        light_metal_state=light,
        carbon_level_state=None,
        oxygen_level_state=None,
        metadata={},
    )

    adjusted_atmosphere, adjusted_metal = _pg1159_nlte_charge_feedback(
        atmosphere, metal, previous
    )

    helium_total = np.sum(adjusted_metal.host_ion_number_density, axis=0)
    carbon_total = adjusted_metal.element_number_density["C"]
    particle_pressure = BOLTZMANN * temperature * (
        adjusted_atmosphere.electron_density + helium_total + carbon_total
    )
    np.testing.assert_allclose(particle_pressure, gas_pressure, rtol=2.0e-14)
    np.testing.assert_allclose(carbon_total / helium_total, 0.1)
    # The NLTE electrons must also enter the LTE reference Saha ratios
    # used to construct inverse radiative and collisional rates.
    electron_ratio = metal.electron_density / adjusted_metal.electron_density
    for old, new in ((host_lte, adjusted_metal.host_ion_number_density),
                     (carbon_lte, adjusted_metal.ion_number_density["C"])):
        np.testing.assert_allclose((new[1:] / new[:-1]) / (old[1:] / old[:-1]),
            np.broadcast_to(electron_ratio, old[1:].shape), rtol=2e-13)
    helium_b = np.array([.01/.05, .29/.75, .70/.20])[:, None]
    he_actual = adjusted_metal.host_ion_number_density * helium_b
    he_actual *= helium_total / he_actual.sum(axis=0)
    c_actual = adjusted_metal.ion_number_density["C"] * (carbon_nlte / carbon_lte)
    c_actual *= carbon_total / c_actual.sum(axis=0)
    required_electrons = he_actual[1] + 2*he_actual[2] + c_actual[1] + 2*c_actual[2]
    np.testing.assert_allclose(adjusted_metal.electron_density, required_electrons, rtol=2e-13)
    host = adjusted_metal.host_ion_number_density
    np.testing.assert_allclose(adjusted_atmosphere.helium_lte_state.mean_ion_charge,
        (host[1] + 2 * host[2]) / host.sum(axis=0), rtol=2e-13)
    repeated_atmosphere, repeated_metal = _pg1159_nlte_charge_feedback(
        adjusted_atmosphere, adjusted_metal, previous)
    np.testing.assert_allclose(repeated_metal.electron_density,
        adjusted_metal.electron_density, rtol=2e-13)
    assert adjusted_atmosphere.metadata[
        "nlte_charge_feedback_maximum_pressure_residual"
    ] < 2.0e-14


def test_population_transfer_uses_only_mapped_formal_components():
    population_ion = AtomicIon(
        "C",
        3,
        12.0,
        64.0,
        (
            AtomicLevel(1, 0.0, 2.0, "lower"),
            AtomicLevel(2, 20_000.0, 4.0, "retained upper"),
            AtomicLevel(3, 30_000.0, 6.0, "LTE upper"),
        ),
        (AtomicTransition(1, 2, 1.0e8, "E1", 5005.0, 0.3),),
    )
    formal_ion = AtomicIon(
        "C",
        3,
        12.0,
        64.0,
        (
            AtomicLevel(10, 0.0, 2.0, "lower"),
            AtomicLevel(11, 20_000.0, 4.0, "retained upper"),
            AtomicLevel(12, 30_000.0, 6.0, "LTE upper"),
        ),
        (
            AtomicTransition(10, 11, 1.0e8, "E1", 5000.0, 0.2),
            AtomicTransition(10, 12, 1.0e8, "E1", 4000.0, 0.1),
        ),
    )
    population_database = AtomicDatabase(
        MappingProxyType({("C", 3): population_ion}), "population"
    )
    formal_database = AtomicDatabase(
        MappingProxyType({("C", 3): formal_ion}), "formal"
    )
    mapping = MappingProxyType({
        ("C", 3, 1): (("C", 3, 10),),
        ("C", 3, 2): (("C", 3, 11),),
        ("C", 3, 3): (("C", 3, 12),),
    })

    keys = _formal_population_transition_keys(
        population_database, formal_database, mapping, "C", {3: 2}
    )

    assert keys == frozenset({("C", 3, 10, 11)})


def test_formal_spectrum_drops_lines_attached_to_omitted_rate_levels():
    population_ion = AtomicIon(
        "C",
        3,
        12.0,
        64.0,
        (
            AtomicLevel(1, 0.0, 2.0, "lower"),
            AtomicLevel(2, 20_000.0, 4.0, "retained upper"),
            AtomicLevel(3, 30_000.0, 6.0, "omitted upper"),
        ),
        (AtomicTransition(1, 2, 1.0e8, "E1", 5005.0, 0.3),),
    )
    formal_ion = AtomicIon(
        "C",
        3,
        12.0,
        64.0,
        (
            AtomicLevel(10, 0.0, 2.0, "lower"),
            AtomicLevel(11, 20_000.0, 4.0, "retained upper"),
            AtomicLevel(12, 30_000.0, 6.0, "omitted upper"),
            AtomicLevel(13, 40_000.0, 8.0, "LTE child"),
        ),
        (
            AtomicTransition(10, 11, 1.0e8, "E1", 5000.0, 0.2),
            AtomicTransition(10, 12, 1.0e8, "E1", 4000.0, 0.1),
            AtomicTransition(11, 13, 1.0e8, "E1", 3000.0, 0.05),
        ),
    )
    population_database = AtomicDatabase(
        MappingProxyType({("C", 3): population_ion}), "population"
    )
    formal_database = AtomicDatabase(
        MappingProxyType({("C", 3): formal_ion}), "formal"
    )
    mapping = MappingProxyType({
        ("C", 3, 1): (("C", 3, 10),),
        ("C", 3, 2): (("C", 3, 11),),
        ("C", 3, 3): (("C", 3, 12),),
    })

    keys = _formal_spectrum_transition_keys(
        formal_database,
        ("C",),
        {"C": population_database},
        {"C": mapping},
        {"C": {3: 2}},
        {"C": {("C", 3, 13): ("C", 3, 2)}},
    )

    assert keys == frozenset({("C", 3, 10, 11), ("C", 3, 11, 13)})


def test_departure_ratio_remains_finite_when_lte_tail_underflows():
    ratio = _finite_departure_ratio(
        np.asarray((1.0, 1.0e-250, 0.0)),
        np.asarray((0.0, 1.0e-300, 0.0)),
    )
    assert np.all(np.isfinite(ratio))
    assert ratio[0] == 1.0e100
    assert ratio[1] == 1.0e50
    assert ratio[2] == 0.0


def test_passenger_ion_ladder_can_be_added_and_updated_directly():
    carbon = np.asarray(((2.0, 2.0), (8.0, 8.0)))
    nitrogen_lte = np.asarray(((3.0, 3.0), (7.0, 7.0)))
    state = LightMetalNLTEState(
        ion_number_density=MappingProxyType({"C": carbon}),
        lte_ion_number_density=MappingProxyType({"C": carbon}),
        ion_departure_coefficient=MappingProxyType({
            ("C", 0): np.ones(2), ("C", 1): np.ones(2)
        }),
        photoionization_rate=MappingProxyType({}),
        radiative_recombination_rate=MappingProxyType({}),
        collisional_ionization_rate=MappingProxyType({}),
        three_body_recombination_rate=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    metal = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({"C": -1.0, "N": -2.0}),
        element_number_density=MappingProxyType({
            "C": carbon.sum(axis=0), "N": nitrogen_lte.sum(axis=0)
        }),
        ion_number_density=MappingProxyType({"C": carbon, "N": nitrogen_lte}),
        partition_function=MappingProxyType({}),
        electron_density=np.ones(2),
        metal_electron_density=np.ones(2),
    )
    augmented = _add_missing_lte_ion_ladders(state, metal, ("C", "N"))
    np.testing.assert_allclose(augmented.ion_number_density["C"], carbon)
    np.testing.assert_allclose(augmented.ion_number_density["N"], nitrogen_lte)
    direct_nitrogen = np.asarray(((1.0, 1.0), (9.0, 9.0)))
    source = LightMetalNLTEState(
        ion_number_density=MappingProxyType({"C": carbon, "N": direct_nitrogen}),
        lte_ion_number_density=MappingProxyType({"C": carbon, "N": nitrogen_lte}),
        ion_departure_coefficient=MappingProxyType({
            ("C", 0): np.ones(2), ("C", 1): np.ones(2),
            ("N", 0): np.full(2, 1.0 / 3.0),
            ("N", 1): np.full(2, 9.0 / 7.0),
        }),
        photoionization_rate=MappingProxyType({}),
        radiative_recombination_rate=MappingProxyType({}),
        collisional_ionization_rate=MappingProxyType({}),
        three_body_recombination_rate=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    updated = _replace_light_metal_elements(augmented, source, ("N",))
    np.testing.assert_allclose(updated.ion_number_density["N"], direct_nitrogen)
    np.testing.assert_allclose(updated.ion_number_density["C"], carbon)


def test_explicit_atom_stage_departure_replaces_independent_ion_ladder():
    independent = LightMetalNLTEState(
        ion_number_density=MappingProxyType({"C": np.ones((4, 2))}),
        lte_ion_number_density=MappingProxyType({"C": np.ones((4, 2))}),
        ion_departure_coefficient=MappingProxyType({
            ("C", charge): np.full(2, 10.0 + charge) for charge in range(4)
        }),
        photoionization_rate=MappingProxyType({}),
        radiative_recombination_rate=MappingProxyType({}),
        collisional_ionization_rate=MappingProxyType({}),
        three_body_recombination_rate=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    explicit = ReducedLightMetalLevelState(
        element="C",
        level_key=(("C", 2, 1), ("C", 2, 2), ("C", 3, 1)),
        population_density=np.asarray(((2.0, 4.0), (1.0, 2.0), (8.0, 4.0))),
        lte_population_density=np.asarray(((1.0, 1.0), (2.0, 2.0), (4.0, 4.0))),
        level_departure_coefficient=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    departure = _ion_departures_with_explicit_stage_closure(
        independent, explicit
    )
    assert departure is not None
    np.testing.assert_allclose(departure[("C", 2)], (1.0, 2.0))
    np.testing.assert_allclose(departure[("C", 3)], (2.0, 1.0))
    np.testing.assert_allclose(departure[("C", 1)], (11.0, 11.0))


def test_explicit_stage_departure_includes_omitted_level_reservoir():
    explicit = ReducedLightMetalLevelState(
        element="C",
        level_key=(("C", 2, 1), ("C", 2, 2), ("C", 3, 1)),
        population_density=np.asarray(((2.0, 4.0), (1.0, 2.0), (8.0, 4.0))),
        lte_population_density=np.asarray(((1.0, 1.0), (2.0, 2.0), (4.0, 4.0))),
        level_departure_coefficient=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
        conservation_weight=np.asarray(((1.0, 1.0), (3.0, 3.0), (2.0, 2.0))),
    )
    departure = _ion_departures_with_explicit_stage_closure(None, explicit)
    assert departure is not None
    np.testing.assert_allclose(departure[("C", 2)], (5.0 / 7.0, 10.0 / 7.0))
    np.testing.assert_allclose(departure[("C", 3)], (2.0, 1.0))


def test_helium_change_norm_is_symmetric_for_a_collapsing_level():
    reference = np.asarray([[4.0]])
    reference_ion = np.asarray([[3.0]])
    reference_continuum = np.asarray([3.0])
    def state(neutral, continuum):
        return CoupledHeliumNLTEState(
            neutral_population_density=np.asarray([[neutral]]),
            singly_ionized_population_density=np.asarray([[3.0]]),
            doubly_ionized_he_density=np.asarray([continuum]),
            lte_neutral_population_density=reference,
            lte_singly_ionized_population_density=reference_ion,
            lte_doubly_ionized_he_density=reference_continuum,
            neutral_departure_coefficient=np.asarray([[neutral / 4.0]]),
            singly_ionized_departure_coefficient=np.asarray([[1.0]]),
            doubly_ionized_departure_coefficient=np.asarray([continuum / 3.0]),
            iterations=1,
            converged=False,
            maximum_relative_population_change=0.0,
            metadata={},
        )
    previous = state(2.0, 5.0)
    current = state(1.0, 6.0)
    change = _maximum_coupled_helium_population_change(previous, current)
    diagnostic = _coupled_helium_population_change_diagnostic(previous, current)
    np.testing.assert_allclose(change, 0.5)
    np.testing.assert_allclose(diagnostic["relative_change"], change)
    assert diagnostic["ion"] == "He I"


def test_tmad_formal_level_mapping_survives_population_damping():
    population_key = ("C", 3, 1)
    mapped_key = ("C", 3, 21)
    lte_parent_key = ("C", 3, 22)
    common = dict(
        element="C",
        level_key=(population_key,),
        lte_population_density=np.asarray(((2.0, 4.0),)),
        level_departure_coefficient=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
        formal_level_mapping=MappingProxyType({
            population_key: (mapped_key,),
        }),
        formal_lte_parent_mapping=MappingProxyType({
            lte_parent_key: population_key,
        }),
    )
    previous = ReducedLightMetalLevelState(
        population_density=np.asarray(((2.0, 4.0),)),
        **common,
    )
    current = ReducedLightMetalLevelState(
        population_density=np.asarray(((6.0, 12.0),)),
        **common,
    )

    damped = _damp_reduced_level_state(previous, current, 0.5)

    np.testing.assert_allclose(
        damped.population_level_departure_coefficient[population_key],
        (2.0, 2.0),
    )
    np.testing.assert_allclose(
        damped.level_departure_coefficient[mapped_key], (2.0, 2.0)
    )
    np.testing.assert_allclose(
        damped.level_departure_coefficient[lte_parent_key], (2.0, 2.0)
    )


def test_population_damping_warm_starts_newly_promoted_term_in_lte():
    shared = ("O", 3, 1)
    promoted = ("O", 3, 2)
    previous = ReducedLightMetalLevelState(
        element="O",
        level_key=(shared,),
        population_density=np.asarray(((4.0, 8.0),)),
        lte_population_density=np.asarray(((2.0, 4.0),)),
        level_departure_coefficient=MappingProxyType({shared: np.asarray((2.0, 2.0))}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    current = ReducedLightMetalLevelState(
        element="O",
        level_key=(shared, promoted),
        population_density=np.asarray(((8.0, 16.0), (6.0, 12.0))),
        lte_population_density=np.asarray(((2.0, 4.0), (2.0, 4.0))),
        level_departure_coefficient=MappingProxyType({}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )

    damped = _damp_reduced_level_state(previous, current, 0.5)

    np.testing.assert_allclose(damped.population_density[0], (6.0, 12.0))
    # The newly promoted term starts from b=1 before applying the damping.
    np.testing.assert_allclose(damped.population_density[1], (4.0, 8.0))
    aligned = _align_reduced_level_state(previous, current)
    assert aligned is not None
    np.testing.assert_allclose(aligned.population_density[0], (4.0, 8.0))
    np.testing.assert_allclose(aligned.population_density[1], (2.0, 4.0))


def test_population_alignment_initializes_a_new_element_from_first_solution():
    key = ("O", 5, 1)
    current = ReducedLightMetalLevelState(
        element="O",
        level_key=(key,),
        population_density=np.asarray(((7.0, 8.0),)),
        lte_population_density=np.asarray(((2.0, 4.0),)),
        level_departure_coefficient=MappingProxyType({key: np.asarray((3.5, 2.0))}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    aligned = _align_reduced_level_state(None, current)
    assert aligned is not None
    np.testing.assert_array_equal(aligned.population_density, current.population_density)
    assert aligned.metadata["warm_start_alignment"] == (
        "new element initialized from first SE solution"
    )


def test_population_alignment_drops_an_element_absent_from_current_model():
    key = ("O", 5, 1)
    previous = ReducedLightMetalLevelState(
        element="O",
        level_key=(key,),
        population_density=np.asarray(((7.0, 8.0),)),
        lte_population_density=np.asarray(((2.0, 4.0),)),
        level_departure_coefficient=MappingProxyType({key: np.asarray((3.5, 2.0))}),
        maximum_lte_recovery_error=0.0,
        metadata={},
    )
    assert _align_reduced_level_state(previous, None) is None


def test_coupled_helium_acceleration_preserves_particles_and_fixed_point():
    from wd_spectra.pg1159 import _coupled_population_update, _damp_coupled_helium_state
    reference = np.array([[.08, .02, .3, .6], [.04, .01, .45, .5]]) * np.array([1e15, 2e16])[:, None]
    def state(pop):
        dep = pop / reference
        return CoupledHeliumNLTEState(pop[:, :2], pop[:, 2:3], pop[:, 3],
            reference[:, :2], reference[:, 2:3], reference[:, 3],
            dep[:, :2], dep[:, 2:3], dep[:, 3], 1, False, 1., {})
    initial = reference * np.array([2., 2., 1.1, .8])
    initial *= reference.sum(axis=1)[:, None] / initial.sum(axis=1)[:, None]
    slow = accelerated = state(initial)
    history = []
    def update(s):
        pop = np.column_stack((s.neutral_population_density, s.singly_ionized_population_density, s.doubly_ionized_he_density))
        return state(.9 * pop + .1 * reference)
    for _ in range(20):
        slow = _damp_coupled_helium_state(slow, update(slow), .4)
        candidate = update(accelerated)
        proposal = _coupled_population_update(np.zeros(1), np.zeros(1), accelerated, candidate, history, 4)
        accelerated = (_damp_coupled_helium_state(accelerated, candidate, .4) if proposal is None else proposal[1])
        pop = np.column_stack((accelerated.neutral_population_density, accelerated.singly_ionized_population_density, accelerated.doubly_ionized_he_density))
        assert np.all(pop > 0)
        np.testing.assert_allclose(pop.sum(axis=1), reference.sum(axis=1), rtol=1e-14)
    assert np.max(abs(accelerated.neutral_departure_coefficient - 1)) < .05 * np.max(abs(slow.neutral_departure_coefficient - 1))
    fixed_history = []
    for _ in range(4):
        fixed = _coupled_population_update(np.zeros(1), np.zeros(1), state(reference), state(reference), fixed_history, 4)
        if fixed is not None:
            np.testing.assert_allclose(fixed[1].neutral_departure_coefficient, 1.)


def test_population_acceleration_weights_prevent_empty_level_noise_dominating_fit():
    from wd_spectra.multilevel_nlte import _anderson_log_population_update
    # First population has a slow physical mode. The second is an effectively
    # empty level with alternating relative noise below the convergence floor.
    def proposal(weights):
        history = []
        for iteration in range(3):
            x = np.array([.01 * iteration, 0.])
            residual = np.array([.1 * (1 - x[0]), (-1.) ** iteration])
            result, _ = _anderson_log_population_update(x, x + residual, history,
                depth=4, mixing=.65, maximum_step=10., residual_weights=weights)
        return result
    weighted = proposal(np.array([1., 1e-10]))
    unweighted = proposal(None)
    assert weighted is not None and unweighted is not None
    assert abs(weighted[0] - 1.) < 1e-3
    assert abs(unweighted[0] - 1.) > .1


def test_accelerated_levels_preserve_omitted_reservoir_particle_constraint():
    from wd_spectra.light_metal_nlte import ReducedLightMetalLevelState
    from wd_spectra.pg1159 import _states_from_population_vector
    keys = (('O', 4, 1), ('O', 4, 2), ('O', 5, 1))
    population = np.array([[.3], [.3], [.4]])
    weight = np.array([[1.], [1.], [11.]])
    departure = {key: np.ones(1) for key in keys}
    state = ReducedLightMetalLevelState('O', keys, population, population,
        departure, 0., {}, departure, conservation_weight=weight)
    _, _, mixed = _states_from_population_vector(
        np.log(np.array([.1, .1, .8])), None, None, state, None)
    np.testing.assert_allclose(np.sum(weight*mixed.population_density, axis=0),
        np.sum(weight*population, axis=0), rtol=1e-14)
    assert not np.isclose(mixed.population_density.sum(), population.sum())
    # Decoding an exact SE image leaves the physical solution unchanged.
    _, _, fixed = _states_from_population_vector(np.log(population.ravel()),
        None, None, state, None)
    np.testing.assert_allclose(fixed.population_density, population, rtol=1e-14)
