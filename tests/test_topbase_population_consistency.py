"""Population conservation across independent opacity representations."""
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra.atmosphere import Atmosphere
from wd_spectra.constants import BOLTZMANN, LIGHT_SPEED, PLANCK
from wd_spectra.d6 import (
    TOPbaseLevelPhotoionization,
    TOPbasePhotoionizationDatabase,
    _topbase_native_population_levels,
    read_tlusty_rap_lte_photoionization,
    read_tlusty_topbase_lte_photoionization,
    tlusty_observed_term_excitation_overrides,
    topbase_bound_free_mass_absorption_coefficient,
    topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    topbase_lte_level_population,
    topbase_population_provenance,
    resonance_average_topbase_photoionization,
    merge_topbase_photoionization_databases,
)
from wd_spectra.metals import (
    AtomicDatabase, AtomicIon, AtomicLevel, BOHR_RADIUS_CM,
    EV_TO_ERG, EV_TO_WAVENUMBER, MetalLTEState,
    ground_term_levels, metal_lte_level_population, read_stout_atomic_database,
    read_stout_atomic_ion,
)
from wd_spectra.models.common import ModelData


def _fixture(levels, *, nonideal=True, neutral_density=1.0e22):
    ion = AtomicIon("Mg", 0, 24.305, 7.64624, tuple(levels), ())
    parent = AtomicIon("Mg", 1, 24.305, 15.0353,
                       (AtomicLevel(1, 0., 2., "2p6.3s.(2S<1/2>)"),), ())
    atomic = AtomicDatabase(MappingProxyType({("Mg", 0): ion, ("Mg", 1): parent}))
    temperature = np.array([8000.])
    electrons = np.array([1.0e15])
    neutrals = {"He": np.array([neutral_density])} if neutral_density else None
    radii = {"He": .5 * BOHR_RADIUS_CM} if neutral_density else None
    partition = {
        key: (part.occupation_weighted_partition_function(
            temperature, electrons, neutral_perturber_number_density=neutrals,
            neutral_perturber_radius_cm=radii,
        ) if nonideal else part.partition_function(temperature))
        for key, part in atomic.ions.items()
    }
    state = MetalLTEState(
        reference_species="He", log_number_abundance={"Mg": -7.},
        element_number_density={"Mg": np.array([1.01e15])},
        ion_number_density={"Mg": np.array([[1.0e13], [1.0e15]])},
        partition_function=partition, electron_density=electrons,
        metal_electron_density=electrons, metal_level_dissolution=nonideal,
        neutral_perturber_number_density=neutrals,
        neutral_perturber_radius_cm=radii, atomic_database=atomic,
    )
    atm = Atmosphere(8000., 8., np.array([1.]), np.array([1.]), temperature,
                     np.array([1.0e10]), np.array([.1]), np.zeros(1),
                     np.zeros(1), electrons, {})
    return atm, state, atomic


def _section(ion, level, *, source_index=987, parent_energy=0.):
    excitation = level.energy_wavenumber / EV_TO_WAVENUMBER
    edge = ion.ionization_energy_ev - excitation + parent_energy
    return TOPbaseLevelPhotoionization(
        ion.element, ion.charge, source_index, excitation, level.statistical_weight,
        edge, edge, np.array([edge, 30.]), np.array([1e-17, 1e-17]),
    )


@pytest.mark.parametrize("nonideal", [False, True])
def test_one_ground_level_continuum_uses_entire_ion_population(nonideal):
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground], nonideal=nonideal)
    section = _section(atomic.ions[("Mg", 0)], ground)
    photon = 12.
    wavelength = np.array([PLANCK * LIGHT_SPEED / (photon * EV_TO_ERG) * 1e8])
    actual = topbase_bound_free_mass_absorption_coefficient(
        atm, wavelength, state, TOPbasePhotoionizationDatabase((section,), "test"),
    )
    stimulated = -np.expm1(-photon * EV_TO_ERG / (BOLTZMANN * atm.temperature))
    expected = 1e-17 * state.ion_number_density["Mg"][0] * stimulated / atm.mass_density
    np.testing.assert_allclose(actual[0], expected, rtol=3e-14)


def test_identical_cross_sections_sum_to_all_bound_ion_population():
    levels = [AtomicLevel(8, 0., 1., "3s2.(1S<0>)"),
              AtomicLevel(19, 4.0 * EV_TO_WAVENUMBER, 3., "3s.3p.(1Po<1>)")]
    atm, state, atomic = _fixture(levels)
    ion = atomic.ions[("Mg", 0)]
    sections = tuple(_section(ion, level, source_index=i+70) for i, level in enumerate(levels))
    photon = 12.
    wavelength = np.array([PLANCK * LIGHT_SPEED / (photon * EV_TO_ERG) * 1e8])
    opacity = topbase_bound_free_mass_absorption_coefficient(
        atm, wavelength, state, TOPbasePhotoionizationDatabase(sections, "test"),
    )
    stimulated = -np.expm1(-photon * EV_TO_ERG / (BOLTZMANN * atm.temperature))
    recovered = opacity[0] * atm.mass_density / (1e-17 * stimulated)
    np.testing.assert_allclose(recovered, state.ion_number_density["Mg"][0], rtol=3e-14)


def test_channel_threshold_does_not_change_the_lower_population():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground], neutral_density=0.)
    ion = atomic.ions[("Mg", 0)]
    # An unmatched theoretical model term exercises the charged-only
    # continuation, whose true binding is I-E, independent of parent.
    model = AtomicLevel(997, 4.0 * EV_TO_WAVENUMBER, 3., "unmapped source term")
    a = _section(ion, model)
    b = _section(ion, model, parent_energy=5.)
    pa = topbase_lte_level_population(a, state, atm.temperature)
    pb = topbase_lte_level_population(b, state, atm.temperature)
    np.testing.assert_array_equal(pa, pb)


def test_ls_term_population_sums_its_native_fine_structure_components():
    levels = [AtomicLevel(8, 0., 1., "3s2.(1S<0>)")]
    for index, j, energy in [(19, 0, 2.6), (31, 1, 2.7), (57, 2, 2.8)]:
        levels.append(AtomicLevel(index, energy * EV_TO_WAVENUMBER,
                                  2.*j+1., f"3s.3p.(3Po<{j}>)"))
    atm, state, atomic = _fixture(levels)
    ion = atomic.ions[("Mg", 0)]
    term = levels[1:]
    weight = sum(level.statistical_weight for level in term)
    energy = sum(level.energy_wavenumber * level.statistical_weight for level in term) / weight
    section = _section(ion, AtomicLevel(1, energy, weight, "model term"))
    actual = topbase_lte_level_population(section, state, atm.temperature)
    expected = sum(metal_lte_level_population(ion, level, state, atm.temperature) for level in term)
    assert {level.index for level in _topbase_native_population_levels(section, ion)} == {19, 31, 57}
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_pseudocontinuum_and_true_continuum_share_lower_population_at_edge():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground])
    section = _section(atomic.ions[("Mg", 0)], ground)
    edge = PLANCK * LIGHT_SPEED / (section.threshold_energy_ev * EV_TO_ERG) * 1e8
    database = TOPbasePhotoionizationDatabase((section,), "test")
    bound_free = topbase_bound_free_mass_absorption_coefficient(atm, np.array([edge]), state, database)
    pseudo = topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atm, np.array([edge * (1 + 1e-8)]), state, database,
    )
    assert bound_free[0, 0] > 0
    np.testing.assert_allclose(pseudo, bound_free, rtol=1e-6)


def test_nonideal_manual_state_requires_the_actual_partition_atom():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground])
    state = replace(state, atomic_database=None)
    section = _section(atomic.ions[("Mg", 0)], ground)
    with pytest.raises(ValueError, match="atomic_database used to build metal_state"):
        topbase_lte_level_population(section, state, atm.temperature)


def test_real_tlusty_magnesium_and_silicon_terms_use_physical_identity():
    data = ModelData.default()
    atomic = read_stout_atomic_database(data.stout, elements=("Mg", "Si"), maximum_charge=1)
    datasets = [(data.tlusty_atoms / "mg1_12+6lev.dat", "Mg", 0),
                (data.tlusty_atoms / "si1_16+6lev.dat", "Si", 0)]
    database = read_tlusty_topbase_lte_photoionization(
        datasets, atomic, excitation_energy_overrides_ev=tlusty_observed_term_excitation_overrides(datasets, atomic),
    )
    selected = {(s.element, s.level_index): s for s in database.sections}
    # Mg model level2 is the complete3P term (Stout2/3/4), not Stout2.
    mg = _topbase_native_population_levels(selected[("Mg", 2)], atomic.ions[("Mg", 0)])
    assert len(mg) == 3
    assert sum(level.statistical_weight for level in mg) == 9.
    # Si model level2 is1D; its Stout level index is4.
    si = _topbase_native_population_levels(selected[("Si", 2)], atomic.ions[("Si", 0)])
    assert len(si) == 1 and si[0].index == 4
    assert "(1D<2>)" in si[0].label

    # The complete native bundles also contain effective aggregate levels.
    # None is silently omitted merely because it lacks a Stout identity.
    atm, state, _ = _fixture([AtomicLevel(8, 0., 1., "3s2.(1S<0>)")])
    partition = {key: ion.occupation_weighted_partition_function(
        atm.temperature, state.electron_density,
        neutral_perturber_number_density=state.neutral_perturber_number_density,
        neutral_perturber_radius_cm=state.neutral_perturber_radius_cm,
    ) for key, ion in atomic.ions.items()}
    state = replace(state, atomic_database=atomic,
                    ion_number_density={"Mg": state.ion_number_density["Mg"],
                                        "Si": state.ion_number_density["Mg"]},
                    partition_function=partition)
    populations = [topbase_lte_level_population(s, state, atm.temperature) for s in database.sections]
    assert len(populations) == len(database.sections) == 40
    assert np.all(np.isfinite(populations)) and np.all(np.array(populations) >= 0)
    opacity = topbase_bound_free_mass_absorption_coefficient(
        atm, np.array([1000., 2000., 3500., 5000.]), state, database,
    )
    assert np.all(np.isfinite(opacity)) and np.any(opacity > 0)
    provenance = topbase_population_provenance(database, atomic)
    assert provenance["unresolved_model_level_sections"] == 12
    assert provenance["native_level_sections"] + provenance["native_LS_term_sections"] == 28


def test_effective_level_survival_has_the_native_shell_radius_density_scaling():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    recovered = []
    densities = (5e20, 1e21)
    for density in densities:
        atm, state, atomic = _fixture([ground], neutral_density=density)
        ion = atomic.ions[("Mg", 0)]
        model = AtomicLevel(999, 7.0 * EV_TO_WAVENUMBER, 9., "effective aggregate")
        section = _section(ion, model)
        population = topbase_lte_level_population(section, state, atm.temperature)
        assert np.all(np.isfinite(population)) and np.all(population > 0)
        # Remove Q and the source-model Boltzmann factor. The ratio of the
        # remaining survival at two He densities has an analytic exponential.
        recovered.append(population * state.partition_function[("Mg", 0)]
                         / state.ion_number_density["Mg"][0]
                         / (section.statistical_weight * np.exp(
                             -section.excitation_energy_ev * EV_TO_ERG / (BOLTZMANN * atm.temperature))))
    n_squared = 13.605693122994 / (7.64624 - 7.)
    shell_radius = BOHR_RADIUS_CM * (5. * n_squared + 1.) / 4.
    excluded_volume = 4. * np.pi / 3. * (shell_radius + .5 * BOHR_RADIUS_CM)**3
    expected = np.exp(-(densities[1]-densities[0]) * excluded_volume)
    np.testing.assert_allclose(recovered[1] / recovered[0], expected, rtol=2e-14)


def test_effective_level_occupation_recovers_the_source_model_dilute_limit():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground], neutral_density=1e3)
    state = replace(state, electron_density=np.array([1e3]))
    ion = atomic.ions[("Mg", 0)]
    section = _section(ion, AtomicLevel(999, 3. * EV_TO_WAVENUMBER, 9., "effective aggregate"))
    actual = topbase_lte_level_population(section, state, atm.temperature)
    ideal = topbase_lte_level_population(section, replace(state, metal_level_dissolution=False), atm.temperature)
    np.testing.assert_allclose(actual, ideal, rtol=1e-12)


def test_strict_audit_policy_rejects_unidentified_superlevels():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground])
    section = _section(atomic.ions[("Mg", 0)], AtomicLevel(999, 3. * EV_TO_WAVENUMBER, 9., "effective aggregate"))
    database = TOPbasePhotoionizationDatabase((section,), "test", unmapped_population_policy="strict")
    with pytest.raises(ValueError, match="strict population policy"):
        topbase_bound_free_mass_absorption_coefficient(atm, np.array([1000.]), state, database)


def test_policy_and_provenance_survive_photoionization_transformations():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    _, _, atomic = _fixture([ground])
    ion = atomic.ions[("Mg", 0)]
    sections = (_section(ion, ground), _section(ion, AtomicLevel(999, 3. * EV_TO_WAVENUMBER, 9., "effective aggregate")))
    database = TOPbasePhotoionizationDatabase(sections, "test", unmapped_population_policy="strict")
    averaged = resonance_average_topbase_photoionization(database, .01)
    merged = merge_topbase_photoionization_databases(averaged)
    assert merged.unmapped_population_policy == "strict"
    provenance = topbase_population_provenance(merged, atomic)
    assert provenance["section_count"] == 2
    assert provenance["native_level_sections"] == 1
    assert provenance["unresolved_model_level_sections"] == 1
    assert provenance["nonideal_unmapped_population_policy"] == "strict"
    effective = topbase_population_provenance(replace(database, unmapped_population_policy="effective-level"), atomic)
    assert "not exact native-partition" in effective["unresolved_population"]


def test_bundled_rap_feii_excitation_is_already_above_lowest_j():
    """Actual RAP absolute energies must not receive a second LS offset."""
    data = ModelData.default()
    ion = read_stout_atomic_ion(data.stout, "Fe", 1)
    # Native indices deliberately differ from source-local model indices.
    ion = replace(ion, levels=tuple(replace(level, index=level.index+1000)
                                   for level in ion.levels), transitions=())
    atom = AtomicDatabase(MappingProxyType({("Fe", 1): ion}))
    path = data.tlusty_atoms / "fe2p_14+11lev.rap"
    loaded = read_tlusty_rap_lte_photoionization([(path, "Fe", 1)], atom)
    np.testing.assert_allclose(
        [section.excitation_energy_ev for section in loaded.sections[:3]],
        np.array([416.299, 2416.571, 8320.469]) / EV_TO_WAVENUMBER,
        rtol=0., atol=1e-14,
    )
    assert [section.statistical_weight for section in loaded.sections[:3]] == [30., 28., 20.]
    ground = loaded.sections[0]
    members = _topbase_native_population_levels(ground, ion)
    assert {level.index for level in members} == {1001, 1002, 1003, 1004, 1005}
    assert members == ground_term_levels(ion)
    assert all("a6D" in level.label for level in members)
    assert ground.threshold_energy_ev == pytest.approx(
        ion.ionization_energy_ev - 416.299 / EV_TO_WAVENUMBER, abs=1e-13,
    )

    # Keep every channel and the original absolute photon/cross-section grid.
    # A companion atom has a slightly older IP; the first positive sample is
    # not redefined as I-E here and no level-identity tolerance is relaxed.
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[2])
    assert count == len(loaded.sections) == 25
    cursor = 1
    for section in loaded.sections:
        index, energy, weight, points = lines[cursor].split()
        cursor += 1
        samples = np.array([tuple(map(float, row.split()))
                            for row in lines[cursor:cursor+int(points)]])
        cursor += int(points)
        first = int(np.flatnonzero(samples[:, 1] > 0.)[0])
        start = max(0, first-1)
        assert section.level_index == int(index)
        assert section.statistical_weight == float(weight)
        assert section.excitation_energy_ev == float(energy) / EV_TO_WAVENUMBER
        np.testing.assert_array_equal(section.photon_energy_ev,
                                      PLANCK * samples[start:, 0] / EV_TO_ERG)
        np.testing.assert_array_equal(section.cross_section_cm2,
                                      samples[start:, 1] * 1e-18)
        assert section.absolute_photon_grid
        positive = section.cross_section_cm2 > 0.
        np.testing.assert_allclose(
            section.cross_section(section.photon_energy_ev[positive]),
            section.cross_section_cm2[positive], rtol=3e-13, atol=0.,
        )
        opening = section.photon_energy_ev[np.flatnonzero(positive)[0]]
        assert section.sampling_edge_energy_ev == opening
        assert section.cross_section(np.nextafter(section.photon_energy_ev[0], 0.)) == 0.
        first = int(np.flatnonzero(positive)[0])
        if first:
            midpoint = .5 * (section.photon_energy_ev[first-1] + opening)
            # The narrowest ramps make the represented midpoint differ
            # from an exact half by ~1e-10 of the interval. Evaluate the
            # reference at that actual floating-point coordinate.
            expected_midpoint = np.interp(midpoint,
                section.photon_energy_ev[first-1:first+1],
                section.cross_section_cm2[first-1:first+1])
            np.testing.assert_allclose(section.cross_section(midpoint),
                expected_midpoint, rtol=3e-13, atol=0.)
        # Nominal population bindings cannot clip or shift an aggregate's
        # absolute photon table, including below the mean lower binding.
        shifted = replace(section, threshold_energy_ev=100.,
                          source_threshold_energy_ev=101.)
        np.testing.assert_array_equal(
            shifted.cross_section(section.photon_energy_ev),
            section.cross_section(section.photon_energy_ev),
        )

    # Actual Fe II aggregate6 has positive support below its mean I-E.
    section = loaded.sections[5]
    photon = PLANCK * LIGHT_SPEED / (958.9365366146347e-8) / EV_TO_ERG
    assert photon < section.threshold_energy_ev
    positive = section.cross_section_cm2 > 0.
    expected = np.exp(np.interp(np.log(photon),
        np.log(section.photon_energy_ev[positive]),
        np.log(section.cross_section_cm2[positive])))
    np.testing.assert_allclose(section.cross_section(photon), expected,
                               rtol=3e-13, atol=0.)
    assert expected > 2e-18
    averaged = resonance_average_topbase_photoionization(loaded, .01)
    assert all(s.absolute_photon_grid for s in averaged.sections)


def test_absolute_table_interpolates_zero_pairs_without_inventing_coverage():
    section = TOPbaseLevelPhotoionization(
        "Fe", 1, 1, 3., 78., 100., 101.,
        np.array([2., 4., 6., 8., 10.]),
        np.array([1., 0., 0., 4., 2.]) * 1e-18,
        absolute_photon_grid=True,
    )
    np.testing.assert_allclose(section.cross_section([1., 2., 3., 5., 7., 8., 20.]),
        np.array([0., 1., .5, 0., 2., 4., .25]) * 1e-18,
        rtol=3e-13, atol=0.)
    assert section.structure_edge_energies_ev == (2.,)


def test_absolute_aggregate_table_rejects_undefined_pseudocontinuum():
    ground = AtomicLevel(8, 0., 1., "3s2.(1S<0>)")
    atm, state, atomic = _fixture([ground])
    section = replace(_section(atomic.ions[("Mg", 0)], ground),
                      absolute_photon_grid=True)
    with pytest.raises(ValueError, match="do not define a unique continuum edge"):
        topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
            atm, [2000.], state,
            TOPbasePhotoionizationDatabase((section,), "synthetic RAP"),
        )


def test_rap_feiii_primary_header_has_nonzero_absolute_ground_centroid(tmp_path):
    """Fe III is a source-format control, not an extra bundled opacity ion.

    Header and first samples: TLUSTY database/atom/fe3p_22+7lev.rap, linked
    from https://tlusty.oca.eu/tlusty/Tlusty2002/data-iron.html. The fixture
    excerpts one four-point section; the declared counts match that excerpt.
    """
    ion = read_stout_atomic_ion(ModelData.default().stout, "Fe", 2)
    atom = AtomicDatabase(MappingProxyType({("Fe", 2): ion}))
    rap = tmp_path / "feiii-ground-format-control.rap"
    rap.write_text(
        "26 3 1\n"
        "1 422.896 25.0 4\n"
        "9.99999996e11 0.0\n"
        "6.99692532e15 0.0\n"
        "6.99693230e15 3.942\n"
        "7.20841434e15 5.454\n",
        encoding="ascii",
    )
    members = ground_term_levels(ion)
    assert sum(level.statistical_weight for level in members) == 25.
    assert min(level.energy_wavenumber for level in members) == 0.
    assert max(level.energy_wavenumber for level in members) > 1000.
    section, = read_tlusty_rap_lte_photoionization([(rap, "Fe", 2)], atom).sections
    assert section.excitation_energy_ev == pytest.approx(422.896 / EV_TO_WAVENUMBER, abs=1e-14)
    assert section.threshold_energy_ev == pytest.approx(
        ion.ionization_energy_ev-422.896 / EV_TO_WAVENUMBER, abs=1e-13,
    )
    np.testing.assert_array_equal(section.cross_section_cm2,
                                  np.array([0., 3.942, 5.454]) * 1e-18)
