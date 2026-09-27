from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import (
    gray_helium_atmosphere,
    gray_hydrogen_atmosphere,
    radiative_equilibrium_helium_atmosphere,
    radiative_equilibrium_hydrogen_atmosphere,
)
from wd_spectra.adaptive_structure import rosseland_mean_from_opacity_grid
from wd_spectra.metals import (
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    ATOMIC_MASS_U,
    ATOMIC_NUMBER,
    IONIZATION_ENERGY_EV,
    _hydrogen_state_with_trace_metal_electrons,
    atomic_database_with_melendez_barbuy_feii_oscillator_strengths,
    augment_oxygen_i_6258_6271_multiplet,
    atmosphere_with_metal_electrons,
    ca_ii_electron_stark_rate_coefficient,
    ca_ii_helium_impact_rate_coefficient,
    chondritic_metal_abundances,
    dense_helium_ionization_potential_shift_ev,
    helium_metal_lte_state_from_mass_fractions,
    helium_metal_number_abundances_from_mass_fractions,
    metal_bound_free_mass_absorption_coefficient,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    mg_i_3835_electron_stark_rate_coefficient,
    mg_i_optical_electron_stark_rate_coefficient,
    mg_ii_4481_electron_stark_rate_coefficient,
    mg_ii_4852_electron_stark_rate_coefficient,
    na_i_optical_electron_stark_rate_coefficient,
    na_i_d_neon_impact_rate_coefficient,
    neutral_impact_rate_from_hydrogen,
    o_i_optical_electron_stark_rate_coefficient,
    read_ca_i_he_profile_table,
    read_ca_ii_he_profile_grid,
    read_ca_ii_he_profile_table,
    read_ca_ii_he_profile_temperature_grid,
    read_barklem_neutral_hydrogen_broadening,
    read_mg_he_red_wing_table,
    read_mg_ii_he_profile_table,
    read_kurucz_gf100_atomic_database,
    read_nist_asd_strong_atomic_database,
    read_stout_atomic_database,
    read_stout_atomic_ion,
    read_verner_photoionization_database,
    selected_metal_lines,
    strong_uv_resonance_minimum_half_window_angstrom,
    unsold_helium_impact_rate_coefficient,
    unsold_hydrogen_impact_rate_coefficient,
    unsold_neutral_metal_impact_rate_coefficient,
)


def test_melendez_barbuy_feii_strength_replacement_is_matched_and_idempotent():
    lower_energy = 2.8912 * 8065.544005
    wavelength_vacuum = 5019.84
    upper_energy = lower_energy + 1.0e8 / wavelength_vacuum
    original = AtomicTransition(
        1,
        2,
        2.0e6,
        "E1",
        wavelength_vacuum,
        1.0e-3,
    )
    ion = AtomicIon(
        "Fe",
        1,
        55.845,
        16.1878,
        (
            AtomicLevel(1, lower_energy, 6.0, "a 6S J=5/2"),
            AtomicLevel(2, upper_energy, 8.0, "z 6P J=7/2"),
        ),
        (original,),
    )
    database = AtomicDatabase(MappingProxyType({("Fe", 1): ion}))

    updated = atomic_database_with_melendez_barbuy_feii_oscillator_strengths(
        database
    )
    transition = updated.ions[("Fe", 1)].transitions[0]
    expected_f = 10.0**-1.10 / 6.0
    assert transition.absorption_oscillator_strength == pytest.approx(expected_f)
    assert transition.einstein_a == pytest.approx(
        original.einstein_a * expected_f / original.absorption_oscillator_strength
    )
    assert "1/142" in updated.source
    assert (
        atomic_database_with_melendez_barbuy_feii_oscillator_strengths(updated)
        is updated
    )


def test_common_rocky_pollutants_have_atomic_constants():
    for element in (
        "O", "Al", "P", "S", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn",
    ):
        assert ATOMIC_MASS_U[element] > 0.0
        assert len(IONIZATION_ENERGY_EV[element]) >= 3
        assert np.all(np.diff(IONIZATION_ENERGY_EV[element]) > 0.0)


def test_opacity_grid_rosseland_mean_recovers_gray_extinction():
    wavelength = np.geomspace(100.0, 100_000.0, 500)
    temperature = np.asarray([8_000.0, 15_000.0])
    gray_extinction = np.broadcast_to(
        np.asarray([0.17, 2.3])[np.newaxis, :],
        (wavelength.size, temperature.size),
    )

    result = rosseland_mean_from_opacity_grid(
        wavelength, gray_extinction, temperature
    )

    np.testing.assert_allclose(result, [0.17, 2.3], rtol=2.0e-14)


def test_oxygen_i_6258_6271_supplement_is_complete_and_idempotent():
    levels = (
        AtomicLevel(240, 113714.444, 9.0, "lower J=4"),
        AtomicLevel(241, 113721.413, 7.0, "lower J=3"),
        AtomicLevel(242, 113727.165, 5.0, "lower J=2"),
        AtomicLevel(286, 129666.907, 9.0, "upper J=4"),
        AtomicLevel(287, 129679.841, 9.0, "upper J=4"),
        AtomicLevel(288, 129680.522, 11.0, "upper J=5"),
        AtomicLevel(292, 129693.488, 7.0, "upper J=3"),
    )
    ion = AtomicIon("O", 0, ATOMIC_MASS_U["O"], 13.618055, levels, ())
    database = AtomicDatabase(MappingProxyType({("O", 0): ion}))
    original_count = len(database.ions[("O", 0)].transitions)
    augmented = augment_oxygen_i_6258_6271_multiplet(database)
    transitions = augmented.ions[("O", 0)].transitions
    added = transitions[original_count:]

    assert len(added) == 8
    assert min(line.wavelength_vacuum_angstrom for line in added) == pytest.approx(
        6258.197, abs=0.002
    )
    assert max(line.wavelength_vacuum_angstrom for line in added) == pytest.approx(
        6271.364, abs=0.002
    )
    assert all(line.absorption_oscillator_strength > 0.0 for line in added)
    assert all(line.radiative_damping_rate_s is not None for line in added)
    assert augment_oxygen_i_6258_6271_multiplet(augmented) is augmented


def test_hot_light_element_ladders_reach_bare_nuclei():
    assert len(IONIZATION_ENERGY_EV["C"]) == 6
    assert len(IONIZATION_ENERGY_EV["F"]) == 9
    assert len(IONIZATION_ENERGY_EV["O"]) == 8
    assert len(IONIZATION_ENERGY_EV["Ne"]) == 10
    for element in ("Mg", "Si", "P", "S", "Ar", "Ca", "Fe"):
        assert len(IONIZATION_ENERGY_EV[element]) == ATOMIC_NUMBER[element]


def test_strong_uv_resonance_lines_receive_extended_formal_support():
    ground = AtomicLevel(1, 0.0, 2.0, "ground")
    fine_ground = AtomicLevel(2, 100.0, 4.0, "ground fine structure")
    excited = AtomicLevel(3, 20_000.0, 2.0, "excited")
    upper = AtomicLevel(4, 100_000.0, 4.0, "upper")
    ion = AtomicIon("O", 5, 15.999, 138.1, (ground, fine_ground, excited, upper), ())

    resonance = AtomicTransition(1, 4, 2.0e9, "E1", 1033.8, 0.13)
    fine_resonance = AtomicTransition(2, 4, 2.0e9, "E1", 1033.8, 0.06)
    weak = AtomicTransition(1, 4, 2.0e9, "E1", 1033.8, 0.01)
    subordinate = AtomicTransition(3, 4, 2.0e9, "E1", 1033.8, 0.13)
    optical = AtomicTransition(1, 4, 2.0e9, "E1", 4000.0, 0.13)

    assert strong_uv_resonance_minimum_half_window_angstrom(
        ion, resonance, ground
    ) == pytest.approx(5.0)
    assert strong_uv_resonance_minimum_half_window_angstrom(
        ion, fine_resonance, fine_ground
    ) == pytest.approx(5.0)
    assert strong_uv_resonance_minimum_half_window_angstrom(
        ion, weak, ground
    ) == 0.0
    assert strong_uv_resonance_minimum_half_window_angstrom(
        ion, subordinate, excited
    ) == 0.0
    assert strong_uv_resonance_minimum_half_window_angstrom(
        ion, optical, ground
    ) == 0.0


def test_stout_reader_synthesizes_bare_nucleus_without_a_file(tmp_path: Path):
    ion = read_stout_atomic_ion(tmp_path, "C", 6)
    assert ion.charge == 6
    assert ion.ionization_energy_ev is None
    assert len(ion.levels) == 1
    assert ion.transitions == ()


def test_barklem_reader_attaches_width_and_temperature_exponent(tmp_path):
    lower = AtomicLevel(1, 0.0, 1.0, "lower")
    upper = AtomicLevel(2, 25_000.0, 3.0, "upper")
    transition = AtomicTransition(1, 2, 1.0e7, "E1", 4_000.0, 0.1)
    ion = AtomicIon(
        "Mg", 0, ATOMIC_MASS_U["Mg"], IONIZATION_ENERGY_EV["Mg"][0],
        (lower, upper), (transition,),
    )
    database = AtomicDatabase(MappingProxyType({("Mg", 0): ion}))
    path = tmp_path / "hlist"
    path.write_text(
        "12.00 3998.869 0.0 1.0 0.000 3.100 0.000 60000.000 "
        "25000.000 60000.000 0 1 500. 0.250 -7.300 0.375\n",
        encoding="ascii",
    )

    merged = read_barklem_neutral_hydrogen_broadening(path, database)
    result = merged.ions[("Mg", 0)].transitions[0]
    assert result.neutral_h_vdw_rate_coefficient_cm3_s == pytest.approx(
        10.0**-7.3
    )
    assert result.neutral_h_vdw_temperature_exponent == pytest.approx(0.375)
    assert "Barklem-Piskunov-O'Mara" in merged.source


def test_kurucz_gf100_reader_replaces_optical_lines_and_keeps_damping(tmp_path):
    lower = AtomicLevel(1, 0.0, 1.0, "lower")
    upper = AtomicLevel(2, 25_000.0, 3.0, "upper")
    old = AtomicTransition(1, 2, 1.0e7, "E1", 4_000.0, 0.1)
    ion = AtomicIon(
        "Mg", 0, ATOMIC_MASS_U["Mg"], IONIZATION_ENERGY_EV["Mg"][0],
        (lower, upper), (old,),
    )
    database = AtomicDatabase(MappingProxyType({("Mg", 0): ion}))
    line = (
        f"{399.8900:11.4f}{-0.301:7.3f}{12.00:6.2f}"
        f"{0.0:12.3f}{0.0:5.1f}{'lower':<10}"
        f"{25_000.0:12.3f}{1.0:5.1f} {'upper':<10}"
        f"{7.00:6.2f}{-5.00:6.2f}{-7.00:6.2f}{'TEST':<4}\n"
    )
    path = tmp_path / "gf0300.100"
    path.write_text(line, encoding="latin1")

    merged = read_kurucz_gf100_atomic_database([path], database)
    transition = merged.ions[("Mg", 0)].transitions[0]
    assert len(merged.ions[("Mg", 0)].transitions) == 1
    assert transition.wavelength_vacuum_angstrom == pytest.approx(4_000.0)
    assert transition.absorption_oscillator_strength == pytest.approx(
        10.0**-0.301
    )
    assert transition.radiative_damping_rate_s == pytest.approx(1.0e7)
    assert transition.electron_stark_rate_coefficient_cm3_s == pytest.approx(
        1.0e-5
    )

    damping_only = read_kurucz_gf100_atomic_database(
        [path], database, replace_transitions=False
    )
    retained = damping_only.ions[("Mg", 0)].transitions[0]
    assert retained.absorption_oscillator_strength == pytest.approx(0.1)
    assert retained.electron_stark_rate_coefficient_cm3_s == pytest.approx(
        1.0e-5
    )

    matched_atomic = read_kurucz_gf100_atomic_database(
        [path],
        database,
        replace_transitions=False,
        replace_matched_atomic_data=True,
    )
    matched = matched_atomic.ions[("Mg", 0)].transitions[0]
    assert len(matched_atomic.ions[("Mg", 0)].transitions) == 1
    assert matched.absorption_oscillator_strength == pytest.approx(10.0**-0.301)
    assert matched.einstein_a != old.einstein_a
    assert matched.electron_stark_rate_coefficient_cm3_s == pytest.approx(1.0e-5)
    assert "matched atomic data" in matched_atomic.source

    supplemented = read_kurucz_gf100_atomic_database(
        [path],
        database,
        replace_transitions=False,
        supplement_missing_transitions=True,
    )
    # The energy-pair match prevents an existing Stout transition from being
    # duplicated even though its oscillator strength differs from Kurucz.
    assert supplemented.ions[("Mg", 0)].transitions == (old,)

    second_line = (
        f"{499.8625:11.4f}{-0.602:7.3f}{12.00:6.2f}"
        f"{5_000.0:12.3f}{1.0:5.1f}{'new lower':<10}"
        f"{25_000.0:12.3f}{1.0:5.1f} {'upper':<10}"
        f"{6.50:6.2f}{-5.50:6.2f}{-7.50:6.2f}{'TEST':<4}\n"
    )
    path.write_text(line + second_line, encoding="latin1")
    supplemented = read_kurucz_gf100_atomic_database(
        [path],
        database,
        replace_transitions=False,
        supplement_missing_transitions=True,
    )
    assert len(supplemented.ions[("Mg", 0)].transitions) == 2
    assert supplemented.ions[("Mg", 0)].transitions[0] is old


def test_kurucz_reader_accepts_current_expanded_high_energy_layout(tmp_path):
    lower = AtomicLevel(1, 86_625.757, 3.0, "(4S)3p 5P")
    upper = AtomicLevel(2, 108_731.530, 3.0, "(4S)10d 5D")
    old = AtomicTransition(1, 2, 1.0e7, "E1", 1.0e8 / 22_105.773, 1.0e-4)
    ion = AtomicIon(
        "O", 0, ATOMIC_MASS_U["O"], IONIZATION_ENERGY_EV["O"][0],
        (lower, upper), (old,),
    )
    database = AtomicDatabase(MappingProxyType({("O", 0): ion}))
    line = (
        f"{452.2437:11.4f}{-2.690:7.3f}{8.00:6.2f}"
        f"{86_625.757:12.3f}{1.0:5.1f}{'(4S)3p 5P':<10}"
        f"{108_731.530:13.3f}{1.0:5.1f} {'(4S)10d 5D':<10}"
        f"{7.60:6.2f}{-3.04:6.2f}{-6.75:6.2f}{'K13':<4}\n"
    )
    path = tmp_path / "gf0800.all"
    path.write_text(line, encoding="latin1")
    merged = read_kurucz_gf100_atomic_database(
        [path], database, replace_transitions=False
    )
    transition = merged.ions[("O", 0)].transitions[0]
    assert transition.absorption_oscillator_strength == old.absorption_oscillator_strength
    assert transition.electron_stark_rate_coefficient_cm3_s == pytest.approx(
        10.0**-3.04
    )


def test_kurucz_reader_accepts_overflowed_first_level_label(tmp_path):
    lower = AtomicLevel(1, 16_956.170, 2.0, "2p6.3p 2P")
    upper = AtomicLevel(2, 40_348.918, 4.0, "2p6.10d 2D")
    old = AtomicTransition(
        1, 2, 1.0e6, "E1", 1.0e8 / (40_348.918 - 16_956.170), 1.0e-2
    )
    ion = AtomicIon(
        "Na", 0, ATOMIC_MASS_U["Na"], IONIZATION_ENERGY_EV["Na"][0],
        (lower, upper), (old,),
    )
    database = AtomicDatabase(MappingProxyType({("Na", 0): ion}))
    # Real current-Kurucz layout: the 11-column first label shifts the
    # ordinary 12-column second energy one place to the right.
    line = (
        "   427.3626 -2.511 11.00   40348.918  1.5 2p6.10d 2D"
        "   16956.170  0.5 2p6.3p 2P   7.83 -1.64 -6.59K12\n"
    )
    path = tmp_path / "gf1100.all"
    path.write_text(line, encoding="latin1")
    merged = read_kurucz_gf100_atomic_database(
        [path], database, replace_transitions=True
    )
    transition = merged.ions[("Na", 0)].transitions[0]
    assert transition.wavelength_vacuum_angstrom == pytest.approx(
        old.wavelength_vacuum_angstrom
    )
    assert transition.absorption_oscillator_strength == pytest.approx(
        10.0**-2.511 / 2.0
    )
    assert transition.electron_stark_rate_coefficient_cm3_s == pytest.approx(
        10.0**-1.64
    )


def test_nist_asd_strong_reader_requires_exact_levels_and_accuracy(tmp_path):
    lower = AtomicLevel(1, 0.0, 1.0, "3p6.4s2 1S")
    upper = AtomicLevel(2, 23_652.304, 3.0, "3p6.4s.4p 1P*")
    old = AtomicTransition(1, 2, 2.0e8, "E1", 1.0e8 / 23_652.304, 1.2)
    ion = AtomicIon(
        "Ca", 0, ATOMIC_MASS_U["Ca"], IONIZATION_ENERGY_EV["Ca"][0],
        (lower, upper), (old,),
    )
    database = AtomicDatabase(MappingProxyType({("Ca", 0): ion}))
    header = (
        "obs_wl_air(nm)\tritz_wl_air(nm)\tintens\tAki(s^-1)\tfik\t"
        "log_gf\tAcc\tEi(cm-1)\tEk(cm-1)\tconf_i\tterm_i\tJ_i\t"
        "conf_k\tterm_k\tJ_k\tType\ttp_ref\tline_ref\t\n"
    )
    row = (
        "422.673\t422.673\t100\t2.18e+08\t1.75\t0.243\tB+\t"
        "0.000\t23652.304\t3p6.4s2\t1S\t0\t3p6.4s.4p\t1P*\t1\t"
        "\tT\tL\t\n"
    )
    path = tmp_path / "nist-asd-ca1.tsv"
    path.write_text(header + row, encoding="utf-8")
    merged = read_nist_asd_strong_atomic_database([path], database)
    transition = merged.ions[("Ca", 0)].transitions[0]
    assert transition.absorption_oscillator_strength == pytest.approx(1.75)
    assert transition.einstein_a == pytest.approx(2.18e8)
    assert transition.radiative_damping_rate_s is None
    assert "NIST ASD 5.12" in merged.source

    inaccurate = path.read_text(encoding="utf-8").replace("\tB+\t", "\tD\t")
    path.write_text(inaccurate, encoding="utf-8")
    with pytest.raises(ValueError, match="no usable NIST"):
        read_nist_asd_strong_atomic_database([path], database)


def _ground_state_database(elements: tuple[str, ...]) -> AtomicDatabase:
    ions = {}
    for element in elements:
        for charge in range(len(IONIZATION_ENERGY_EV[element]) + 1):
            ions[(element, charge)] = AtomicIon(
                element=element,
                charge=charge,
                atomic_mass_u=ATOMIC_MASS_U[element],
                ionization_energy_ev=(
                    IONIZATION_ENERGY_EV[element][charge]
                    if charge < len(IONIZATION_ENERGY_EV[element]) else None
                ),
                levels=(AtomicLevel(1, 0.0, 1.0, "ground"),),
                transitions=(),
            )
    return AtomicDatabase(MappingProxyType(ions))


def test_bulk_helium_carbon_oxygen_state_closes_pressure_mass_and_charge():
    atmosphere = gray_helium_atmosphere(120_000.0, 7.0, n_depth=8)
    database = _ground_state_database(("C", "O"))
    state = helium_metal_lte_state_from_mass_fractions(
        atmosphere, database, {"He": 0.33, "C": 0.50, "O": 0.17}
    )
    host = state.host_ion_number_density
    assert host is not None
    n_he = np.sum(host, axis=0)
    total_nuclei = n_he + sum(state.element_number_density.values())
    particle_pressure = (
        total_nuclei + state.electron_density
    ) * 1.380649e-16 * atmosphere.temperature
    np.testing.assert_allclose(particle_pressure, atmosphere.gas_pressure, rtol=3e-12)

    charge = host[1] + 2.0 * host[2]
    for population in state.ion_number_density.values():
        charge += np.sum(
            np.arange(population.shape[0])[:, None] * population, axis=0
        )
    np.testing.assert_allclose(charge, state.electron_density, rtol=3e-12)
    assert state.composition_mode == "bulk"
    assert state.mass_fraction == pytest.approx({"He": 0.33, "C": 0.50, "O": 0.17})


def test_pg1159_mass_fractions_convert_to_number_ratios():
    abundance = helium_metal_number_abundances_from_mass_fractions(
        {"He": 0.33, "C": 0.50, "O": 0.17}
    )
    assert 10.0**abundance["C"] == pytest.approx(
        (0.50 / ATOMIC_MASS_U["C"]) / (0.33 / 4.002602)
    )
    assert 10.0**abundance["O"] == pytest.approx(
        (0.17 / ATOMIC_MASS_U["O"]) / (0.33 / 4.002602)
    )


def test_chondritic_mixture_is_scaled_to_calcium_and_accepts_overrides():
    abundance = chondritic_metal_abundances(
        -10.0, overrides={"Mg": -8.1}, elements=("Mg", "Ca", "Fe")
    )
    assert abundance["Ca"] == -10.0
    assert abundance["Mg"] == -8.1
    assert abundance["Fe"] == pytest.approx(-8.840, abs=0.002)


def test_helium_checkpoint_restart_requires_complete_structure():
    with pytest.raises(ValueError, match="checkpoint restart requires"):
        radiative_equilibrium_helium_atmosphere(
            8000.0,
            8.0,
            stark_table=None,
            n_depth=6,
            max_iterations=1,
            n_continuum_wavelength=80,
            initial_gas_pressure=np.ones(6),
        )


def test_complete_helium_checkpoint_skips_hydrostatic_seed():
    n_depth = 6
    column_mass = np.geomspace(1.0e-6, 1.0, n_depth)
    atmosphere = radiative_equilibrium_helium_atmosphere(
        8000.0,
        8.0,
        stark_table=None,
        n_depth=n_depth,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_lines=False,
        mixing_length_alpha=None,
        initial_temperature=np.full(n_depth, 8000.0),
        initial_column_mass=column_mass,
        initial_gas_pressure=1.0e8 * column_mass,
        initial_rosseland_optical_depth=np.geomspace(1.0e-8, 100.0, n_depth),
    )
    assert atmosphere.metadata["checkpoint_skips_hydrostatic_seed"]


def test_helium_checkpoint_resamples_all_structure_coordinates():
    source_depth = 9
    target_depth = 6
    source_tau = np.geomspace(1.0e-8, 100.0, source_depth)
    source_mass = np.geomspace(1.0e-6, 1.0, source_depth)
    atmosphere = radiative_equilibrium_helium_atmosphere(
        8000.0,
        8.0,
        stark_table=None,
        n_depth=target_depth,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_lines=False,
        mixing_length_alpha=None,
        structure_solver="adaptive-newton",
        n_angle=1,
        initial_temperature=8000.0 * (1.0 + 0.1 * source_tau**0.1),
        initial_column_mass=source_mass,
        initial_gas_pressure=1.0e8 * source_mass,
        initial_rosseland_optical_depth=source_tau,
    )

    assert atmosphere.n_depth == target_depth
    assert atmosphere.metadata["checkpoint_source_depth_points"] == source_depth
    assert atmosphere.metadata["checkpoint_resampled_to_depth_points"] == target_depth
    segments = atmosphere.metadata["nonlinear_solver_segments"]
    assert segments
    assert atmosphere.metadata["nonlinear_solver_terminal_reason"] == (
        segments[-1]["terminal_reason"]
    )
    rejected = atmosphere.metadata[
        "nonlinear_solver_rejected_trial_evaluations"
    ]
    assert rejected == sum(
        segment["rejected_trial_evaluations"] for segment in segments
    )
    assert (
        atmosphere.metadata[
            "electron_scattering_source_iterations_per_evaluation"
        ]
        == 1
    )
    assert (
        atmosphere.metadata[
            "electron_scattering_source_final_maximum_relative_residual"
        ]
        >= 0.0
    )
    np.testing.assert_allclose(
        atmosphere.gas_pressure,
        1.0e8 * atmosphere.column_mass,
        rtol=2.0e-14,
    )


def test_helium_checkpoint_still_requires_measured_equilibrium_certificate():
    n_depth = 6
    column_mass = np.geomspace(1.0e-6, 1.0, n_depth)
    atmosphere = radiative_equilibrium_helium_atmosphere(
        8000.0,
        8.0,
        stark_table=None,
        n_depth=n_depth,
        max_iterations=10,
        temperature_tolerance=1.0,
        flux_tolerance=100.0,
        n_continuum_wavelength=80,
        include_lines=False,
        mixing_length_alpha=None,
        initial_temperature=np.full(n_depth, 8000.0),
        initial_column_mass=column_mass,
        initial_gas_pressure=1.0e8 * column_mass,
        initial_rosseland_optical_depth=np.geomspace(1.0e-8, 100.0, n_depth),
    )
    certificate = atmosphere.metadata["equilibrium_certificate"]
    assert atmosphere.metadata["radiative_equilibrium_converged"] == certificate["verified"]
    assert not certificate["verified"]
    assert certificate["failures"]


def _write_stout_ion(
    root: Path,
    stage: int,
    *,
    upper_energy: float,
    upper_weight: float,
    einstein_a: float,
) -> None:
    directory = root / "stout" / "mg" / f"mg_{stage}"
    directory.mkdir(parents=True)
    (directory / f"mg_{stage}.nrg").write_text(
        "17 09 05\n"
        "1 0.000 1 \"ground\"\n"
        f"2 {upper_energy:.6f} {upper_weight:.1f} \"upper\"\n",
        encoding="utf-8",
    )
    (directory / f"mg_{stage}.tp").write_text(
        "17 09 05\n"
        f"A 1 2 {einstein_a:.8e} E1\n",
        encoding="utf-8",
    )


@pytest.fixture
def atomic_root(tmp_path: Path) -> Path:
    _write_stout_ion(
        tmp_path, 1, upper_energy=35051.264, upper_weight=3.0, einstein_a=4.67e8
    )
    _write_stout_ion(
        tmp_path, 2, upper_energy=35669.31, upper_weight=2.0, einstein_a=2.58e8
    )
    _write_stout_ion(
        tmp_path, 3, upper_energy=40000.0, upper_weight=3.0, einstein_a=1.0e8
    )
    return tmp_path


def test_stout_reader_derives_mg_resonance_wavelength_and_oscillator_strength(
    atomic_root: Path,
):
    ion = read_stout_atomic_ion(atomic_root, "mg", 0)
    assert ion.element == "Mg"
    assert ion.charge == 0
    assert ion.transitions[0].wavelength_vacuum_angstrom == pytest.approx(
        2852.96416, rel=2e-8
    )
    assert ion.transitions[0].absorption_oscillator_strength == pytest.approx(
        1.70958, rel=2e-5
    )
    assert ion.partition_function(5000.0) == pytest.approx(1.00013, rel=2e-4)


def test_capped_line_selection_prefers_populated_lower_levels(tmp_path: Path):
    directory = tmp_path / "stout" / "mg" / "mg_1"
    directory.mkdir(parents=True)
    (directory / "mg_1.nrg").write_text(
        "17 09 05\n"
        "1 0.0 1 ground\n"
        "2 20000.0 3 low-upper\n"
        "3 80000.0 1 excited-lower\n"
        "4 100000.0 3 excited-upper\n",
        encoding="ascii",
    )
    (directory / "mg_1.tp").write_text(
        "17 09 05\n"
        "A 1 2 1.0e6 E1\n"
        "A 3 4 1.0e9 E1\n",
        encoding="ascii",
    )
    ion = read_stout_atomic_ion(tmp_path, "Mg", 0)
    database = AtomicDatabase(MappingProxyType({("Mg", 0): ion}))
    selected = selected_metal_lines(
        database, {"Mg": -7.0}, 4000.0, 6000.0, 1.0e-8, 1, 6000.0
    )
    assert selected[0][1].lower_index == 1


def test_capped_line_selection_accounts_for_ion_stage_population(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    selected = selected_metal_lines(
        database,
        {"Mg": -7.0},
        2000.0,
        5000.0,
        1.0e-8,
        1,
        10_000.0,
        {("Mg", 0): 1.0e-8, ("Mg", 1): 1.0, ("Mg", 2): 1.0e-4},
    )
    assert selected[0][0].charge == 1


def test_flux_weighted_structure_selection_prefers_the_sed_peak():
    levels = (
        AtomicLevel(1, 0.0, 1.0, "ground"),
        AtomicLevel(2, 50_000.0, 3.0, "upper-a"),
        AtomicLevel(3, 55_000.0, 3.0, "upper-b"),
    )
    ultraviolet = AtomicTransition(1, 2, 1.0e7, "E1", 2000.0, 0.1)
    infrared = AtomicTransition(1, 3, 1.0e7, "E1", 20_000.0, 0.1)
    ion = AtomicIon(
        "Mg", 0, ATOMIC_MASS_U["Mg"], IONIZATION_ENERGY_EV["Mg"][0],
        levels, (ultraviolet, infrared),
    )
    database = AtomicDatabase(MappingProxyType({("Mg", 0): ion}))
    selected = selected_metal_lines(
        database,
        {"Mg": 0.0},
        100.0,
        100_000.0,
        1.0e-8,
        1,
        15_680.0,
        flux_weighted=True,
    )
    assert selected[0][1] is ultraviolet


def test_dense_helium_ionization_fit_has_published_sign_and_coefficients():
    shift = dense_helium_ionization_potential_shift_ev(
        "Mg", np.asarray([0.0, 0.1, 1.0]), 8000.0
    )
    np.testing.assert_allclose(shift, [0.0, -0.0328044, -1.245666], rtol=2e-7)
    assert dense_helium_ionization_potential_shift_ev("Si", 1.0, 6000.0) == 0.0
    assert dense_helium_ionization_potential_shift_ev("Mg", 1.0, 15_000.0) == 0.0


def test_ca_ii_helium_impact_width_uses_hammond_laboratory_measurements():
    conversion = 4.0 * np.pi * 2.99792458e10
    assert ca_ii_helium_impact_rate_coefficient(3934.777, 5200.0) == pytest.approx(
        conversion * 1.71e-20
    )
    assert ca_ii_helium_impact_rate_coefficient(3969.591, 5200.0) == pytest.approx(
        conversion * 1.28e-20
    )
    assert ca_ii_helium_impact_rate_coefficient(3934.777, 10_400.0) == pytest.approx(
        conversion * 1.71e-20 * 2.0**0.2285
    )
    with pytest.raises(ValueError, match="Ca II H or K"):
        ca_ii_helium_impact_rate_coefficient(4227.0, 5200.0)


def test_ca_ii_electron_stark_width_uses_experimental_fwhm():
    for wavelength, fwhm in (
        (3159.779, 0.53),
        (3180.251, 0.49),
        (3707.079, 0.66),
        (3737.964, 0.67),
        (3934.777, 0.17),
        (3969.591, 0.16),
    ):
        rate = (
            ca_ii_electron_stark_rate_coefficient(wavelength, 14_000.0)
            * 1.0e17
        )
        center_cm = wavelength * 1.0e-8
        recovered_fwhm = (
            2.0 * center_cm**2 * rate
            / (4.0 * np.pi * 2.99792458e10) * 1.0e8
        )
        assert recovered_fwhm == pytest.approx(fwhm)
    with pytest.raises(ValueError, match="measured Ca II line"):
        ca_ii_electron_stark_rate_coefficient(4227.0, 14_000.0)


def test_mg_ii_4481_electron_stark_width_uses_published_fwhm():
    wavelength = 4482.383
    rate = (
        mg_ii_4481_electron_stark_rate_coefficient(wavelength, 16_900.0)
        * 1.35e17
    )
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        2.0 * center_cm**2 * rate
        / (4.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(2.50)


def test_mg_ii_4852_electron_stark_width_uses_kurucz_coefficient():
    wavelength = 4852.426
    coefficient = mg_ii_4852_electron_stark_rate_coefficient(
        wavelength, 10_000.0
    )
    assert coefficient == pytest.approx(10.0**-2.59)
    rate = coefficient * 1.0e16
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        2.0 * center_cm**2 * rate
        / (4.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(32.13, rel=2.0e-4)


def test_mg_ii_4852_stark_width_rejects_other_wavelengths():
    with pytest.raises(ValueError, match="Mg II 4852"):
        mg_ii_4852_electron_stark_rate_coefficient(4482.4, 10_000.0)


def test_mg_i_3835_electron_stark_width_uses_experimental_fwhm():
    wavelength = 3833.391
    rate = (
        mg_i_3835_electron_stark_rate_coefficient(wavelength, 6_370.0)
        * 1.0e17
    )
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        2.0 * center_cm**2 * rate
        / (4.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(1.55)


@pytest.mark.parametrize(
    ("wavelength", "published_fwhm"),
    ((4704.3, 7.89e-6), (5179.6, 5.32e-7)),
)
def test_mg_i_optical_electron_stark_widths_use_published_values(
    wavelength, published_fwhm
):
    coefficient = mg_i_optical_electron_stark_rate_coefficient(
        wavelength, 10_000.0
    )
    assert coefficient is not None
    rate = coefficient * 1.0e11
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        2.0 * center_cm**2 * rate
        / (4.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(published_fwhm)


def test_mg_i_optical_electron_stark_returns_none_away_from_multiplets():
    assert mg_i_optical_electron_stark_rate_coefficient(
        5528.0, 10_000.0
    ) is None


@pytest.mark.parametrize(
    ("wavelength", "published_fwhm"),
    ((5686.4, 2.11), (5891.8, 0.0249)),
)
def test_na_i_optical_electron_stark_widths_use_published_values(
    wavelength, published_fwhm
):
    coefficient = na_i_optical_electron_stark_rate_coefficient(
        wavelength, 10_000.0
    )
    assert coefficient is not None
    rate = coefficient * 1.0e16
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        2.0 * center_cm**2 * rate
        / (4.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(published_fwhm)


def test_na_i_optical_electron_stark_returns_none_away_from_multiplets():
    assert na_i_optical_electron_stark_rate_coefficient(
        6154.0, 10_000.0
    ) is None


@pytest.mark.parametrize(
    ("wavelength", "reference_rate"),
    ((5891.58, 2.37e-8), (5897.56, 2.27e-8)),
)
def test_na_i_d_neon_impact_rate_uses_measured_450k_width(
    wavelength, reference_rate
):
    rate = na_i_d_neon_impact_rate_coefficient(wavelength, 450.0)
    assert rate is not None
    assert float(rate) == pytest.approx(reference_rate)
    hotter = na_i_d_neon_impact_rate_coefficient(wavelength, 10_000.0)
    assert float(hotter) == pytest.approx(
        reference_rate * (10_000.0 / 450.0) ** 0.3
    )


def test_na_i_d_neon_impact_rate_is_resonance_specific():
    assert na_i_d_neon_impact_rate_coefficient(5689.8, 7_000.0) is None


@pytest.mark.parametrize(
    ("wavelength", "published_fwhm"),
    ((4369.5, 0.0130), (7775.5, 0.0697278), (8448.8, 0.0822)),
)
def test_o_i_optical_electron_stark_widths_use_literature_values(
    wavelength, published_fwhm
):
    ion = AtomicIon(
        "O", 0, ATOMIC_MASS_U["O"], IONIZATION_ENERGY_EV["O"][0], (), ()
    )
    line = AtomicTransition(1, 2, 1.0e7, "E1", wavelength, 0.1)
    coefficient = o_i_optical_electron_stark_rate_coefficient(
        ion, line, np.asarray([10_000.0])
    )
    assert coefficient is not None
    center_cm = wavelength * 1.0e-8
    recovered_fwhm = (
        center_cm**2 * coefficient[0] * 1.0e16
        / (2.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert recovered_fwhm == pytest.approx(published_fwhm)


def test_unsold_helium_width_reproduces_ca_k_laboratory_scale():
    levels = (
        AtomicLevel(1, 0.0, 2.0, "3p6.4s.(2S<1/2>)"),
        AtomicLevel(2, 25_414.4, 4.0, "3p6.4p.(2Po<3/2>)"),
        AtomicLevel(3, 52_166.93, 2.0, "3p6.5s.(2S<1/2>)"),
    )
    ca_k = AtomicTransition(1, 2, 1.4e8, "E1", 3934.777, 0.65)
    excited = AtomicTransition(2, 3, 9.56e7, "E1", 3707.079, 0.197)
    ion = AtomicIon(
        "Ca", 1, 40.078, 11.871719, levels, (ca_k, excited)
    )
    predicted = unsold_helium_impact_rate_coefficient(ion, ca_k, 5200.0)
    hydrogen = unsold_hydrogen_impact_rate_coefficient(ion, ca_k, 5200.0)
    measured = ca_ii_helium_impact_rate_coefficient(3934.777, 5200.0)
    assert predicted is not None
    assert hydrogen is not None
    assert float(predicted) == pytest.approx(float(measured), rel=0.10)
    assert float(hydrogen) > float(predicted)

    excited_rate = unsold_helium_impact_rate_coefficient(ion, excited, 5200.0)
    assert excited_rate is not None
    assert float(excited_rate) > 2.0 * float(predicted)


def test_unsold_bulk_oxygen_perturber_has_polarizability_mass_scaling():
    ion = AtomicIon(
        "Ca",
        1,
        ATOMIC_MASS_U["Ca"],
        IONIZATION_ENERGY_EV["Ca"][1],
        (
            AtomicLevel(1, 0.0, 2.0, "4s 2S"),
            AtomicLevel(2, 25_414.4, 4.0, "4p 2P"),
        ),
        (),
    )
    line = AtomicTransition(1, 2, 1.4e8, "E1", 3934.8, 0.63)
    hydrogen = unsold_hydrogen_impact_rate_coefficient(ion, line, 15_000.0)
    oxygen = unsold_neutral_metal_impact_rate_coefficient(
        ion, line, 15_000.0, "O"
    )
    assert hydrogen is not None and oxygen is not None
    reduced_h = ATOMIC_MASS_U["Ca"] * 1.00784 / (
        ATOMIC_MASS_U["Ca"] + 1.00784
    )
    reduced_o = ATOMIC_MASS_U["Ca"] * ATOMIC_MASS_U["O"] / (
        ATOMIC_MASS_U["Ca"] + ATOMIC_MASS_U["O"]
    )
    expected_scale = (0.802 / 0.666793) ** 0.4 * (
        reduced_h / reduced_o
    ) ** 0.3
    assert oxygen / hydrogen == pytest.approx(expected_scale)


def test_unsold_bulk_neon_perturber_has_polarizability_mass_scaling():
    ion = AtomicIon(
        "Mg",
        0,
        ATOMIC_MASS_U["Mg"],
        IONIZATION_ENERGY_EV["Mg"][0],
        (
            AtomicLevel(1, 21_850.4, 3.0, "3s.3p.(3Po)"),
            AtomicLevel(2, 47_957.1, 3.0, "3s.3d.(3D)"),
        ),
        (),
    )
    line = AtomicTransition(1, 2, 9.0e7, "E1", 3830.44, 0.59)
    hydrogen = unsold_hydrogen_impact_rate_coefficient(ion, line, 12_000.0)
    neon = unsold_neutral_metal_impact_rate_coefficient(
        ion, line, 12_000.0, "Ne"
    )
    assert hydrogen is not None and neon is not None
    reduced_h = ATOMIC_MASS_U["Mg"] * 1.00784 / (
        ATOMIC_MASS_U["Mg"] + 1.00784
    )
    reduced_ne = ATOMIC_MASS_U["Mg"] * ATOMIC_MASS_U["Ne"] / (
        ATOMIC_MASS_U["Mg"] + ATOMIC_MASS_U["Ne"]
    )
    expected_scale = (0.394299 / 0.666793) ** 0.4 * (
        reduced_h / reduced_ne
    ) ** 0.3
    assert neon / hydrogen == pytest.approx(expected_scale)


def test_explicit_hydrogen_width_can_be_rescaled_to_neutral_neon():
    ion = AtomicIon(
        "Mg", 0, ATOMIC_MASS_U["Mg"], IONIZATION_ENERGY_EV["Mg"][0], (), ()
    )
    hydrogen = np.asarray([7.6e-8, 8.0e-8])
    neon = neutral_impact_rate_from_hydrogen(
        ion,
        hydrogen,
        perturber_atomic_mass_u=ATOMIC_MASS_U["Ne"],
        perturber_static_polarizability_a3=0.394299,
    )
    reduced_h = ATOMIC_MASS_U["Mg"] * 1.00784 / (
        ATOMIC_MASS_U["Mg"] + 1.00784
    )
    reduced_ne = ATOMIC_MASS_U["Mg"] * ATOMIC_MASS_U["Ne"] / (
        ATOMIC_MASS_U["Mg"] + ATOMIC_MASS_U["Ne"]
    )
    expected_scale = (0.394299 / 0.666793) ** 0.4 * (
        reduced_h / reduced_ne
    ) ** 0.3
    assert neon == pytest.approx(hydrogen * expected_scale)


def test_metal_eos_conserves_nuclei_and_charge_and_donates_electrons(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(
        8000.0, 8.0, n_depth=12, rosseland_opacity=0.1
    )
    state = metal_lte_state(atmosphere, database, {"Mg": -7.0})
    np.testing.assert_allclose(
        np.sum(state.ion_number_density["Mg"], axis=0),
        state.element_number_density["Mg"],
        rtol=3e-15,
    )
    assert state.host_ion_number_density is not None
    required_electrons = (
        state.host_ion_number_density[1]
        + 2.0 * state.host_ion_number_density[2]
        + state.metal_electron_density
    )
    np.testing.assert_allclose(state.electron_density, required_electrons, rtol=4e-12)
    assert np.all(state.electron_density >= atmosphere.electron_density)

    enriched = atmosphere_with_metal_electrons(atmosphere, state)
    np.testing.assert_allclose(enriched.electron_density, state.electron_density)
    assert enriched.metadata["metal_abundances"] == {"Mg": -7.0}


def test_warm_trace_hydrogen_is_included_in_shared_charge_neutrality(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(
        8000.0, 8.0, n_depth=12, rosseland_opacity=0.1
    )
    # The charge-neutrality bracket visits almost electron-free trial states;
    # the trace-H Saha closure must remain finite there rather than relying on
    # a subsequently rejected overflowing ratio.
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        state = metal_lte_state(
            atmosphere,
            database,
            {"Mg": -7.0},
            log_hydrogen_abundance=-3.5,
        )
    assert state.trace_hydrogen_state is not None
    hydrogen = state.trace_hydrogen_state
    np.testing.assert_allclose(
        hydrogen.neutral_h_density + hydrogen.proton_density,
        hydrogen.hydrogen_nuclei_density,
        rtol=2e-14,
    )
    assert state.host_ion_number_density is not None
    host = state.host_ion_number_density
    required = (
        state.metal_electron_density
        + hydrogen.proton_density
        + host[1]
        + 2.0 * host[2]
    )
    np.testing.assert_allclose(state.electron_density, required, rtol=3e-12)
    enriched = atmosphere_with_metal_electrons(atmosphere, state)
    assert enriched.hydrogen_lte_state is not None
    np.testing.assert_allclose(enriched.neutral_h_density, hydrogen.neutral_h_density)


def test_hydrogen_host_is_reclosed_with_metal_donated_electrons(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_hydrogen_atmosphere(
        6_000.0,
        8.0,
        n_depth=12,
        rosseland_opacity=0.1,
        include_molecules=True,
        include_negative_hydrogen=True,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    state = metal_lte_state(
        atmosphere,
        database,
        {"Mg": -4.0},
        reference_species="H",
        include_dense_helium_ionization=False,
    )

    assert state.host_hydrogen_state is not None
    hydrogen = state.host_hydrogen_state
    required = hydrogen.proton_density + state.metal_electron_density
    if hydrogen.molecular_hydrogen_ion_density is not None:
        required += hydrogen.molecular_hydrogen_ion_density
    if hydrogen.trihydrogen_ion_density is not None:
        required += hydrogen.trihydrogen_ion_density
    if hydrogen.negative_hydrogen_density is not None:
        required -= hydrogen.negative_hydrogen_density
    np.testing.assert_allclose(state.electron_density, required, rtol=2.0e-6)
    np.testing.assert_allclose(
        hydrogen.hydrogen_nuclei_density,
        atmosphere.hydrogen_lte_state.hydrogen_nuclei_density,
        rtol=2.0e-14,
    )
    counted_nuclei = hydrogen.neutral_h_density + hydrogen.proton_density
    counted_nuclei += hydrogen.negative_hydrogen_density
    counted_nuclei += 2.0 * hydrogen.molecular_hydrogen_density
    counted_nuclei += 2.0 * hydrogen.molecular_hydrogen_ion_density
    counted_nuclei += 3.0 * hydrogen.trihydrogen_ion_density
    np.testing.assert_allclose(
        counted_nuclei, hydrogen.hydrogen_nuclei_density, rtol=2.0e-12
    )
    assert np.all(state.electron_density >= atmosphere.electron_density)

    enriched = atmosphere_with_metal_electrons(atmosphere, state)
    assert enriched.hydrogen_lte_state is hydrogen
    np.testing.assert_allclose(enriched.electron_density, state.electron_density)
    np.testing.assert_allclose(enriched.proton_density, hydrogen.proton_density)


def test_hydrogen_host_reclosure_conserves_nuclei_when_newton_falls_back():
    atmosphere = gray_hydrogen_atmosphere(
        6_000.0,
        8.0,
        n_depth=12,
        rosseland_opacity=0.1,
        include_molecules=True,
        include_negative_hydrogen=True,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    template = atmosphere.hydrogen_lte_state
    assert template is not None
    trial_electron_density = np.geomspace(1.0e12, 1.0e22, 12)
    hydrogen, _ = _hydrogen_state_with_trace_metal_electrons(
        template, atmosphere.temperature, trial_electron_density
    )

    counted_nuclei = hydrogen.neutral_h_density + hydrogen.proton_density
    assert hydrogen.negative_hydrogen_density is not None
    assert hydrogen.molecular_hydrogen_density is not None
    assert hydrogen.molecular_hydrogen_ion_density is not None
    assert hydrogen.trihydrogen_ion_density is not None
    counted_nuclei += hydrogen.negative_hydrogen_density
    counted_nuclei += 2.0 * hydrogen.molecular_hydrogen_density
    counted_nuclei += 2.0 * hydrogen.molecular_hydrogen_ion_density
    counted_nuclei += 3.0 * hydrogen.trihydrogen_ion_density
    np.testing.assert_allclose(
        counted_nuclei, hydrogen.hydrogen_nuclei_density, rtol=2.0e-12
    )


def test_adaptive_hydrogen_structure_retains_trace_metal_feedback_metadata(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        include_lyman_lines=False,
        include_metal_lines=True,
        mixing_length_alpha=None,
        structure_solver="adaptive-newton",
        n_angle=1,
        metal_database=database,
        metal_abundances={"Mg": -5.0},
    )

    assert atmosphere.metadata["composition"] == "metal-polluted-hydrogen"
    assert atmosphere.metadata["metal_electron_feedback"] == (
        "fixed-H-nuclei shared H/metal charge closure"
    )
    assert atmosphere.metadata["metal_thermodynamic_derivatives"] == (
        "trace-metal approximation: Q-MHD hydrogen derivatives"
    )
    assert atmosphere.hydrogen_lte_state is not None
    assert "+trace-metal-charge-neutral" in (
        atmosphere.hydrogen_lte_state.chemical_model
    )
    assert atmosphere.metadata[
        "rosseland_opacity_includes_metal_bound_bound_and_bound_free"
    ]


def test_adaptive_helium_structure_retains_trace_metal_feedback_metadata(
    atomic_root: Path,
):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = radiative_equilibrium_helium_atmosphere(
        12_000.0,
        8.0,
        stark_table=None,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_lines=False,
        include_metal_lines=True,
        mixing_length_alpha=None,
        structure_solver="adaptive-newton",
        n_angle=1,
        metal_database=database,
        metal_abundances={"Mg": -7.0},
    )

    assert atmosphere.metadata["composition"] == "metal-polluted-helium"
    assert atmosphere.metadata["structure_solver"] == (
        "adaptive-trust-region-newton"
    )
    assert atmosphere.metadata["adaptive_structure_driver"] == "shared-lte"
    assert atmosphere.metadata[
        "initial_convective_gradient_projection_mode"
    ] == "interface-transport"
    assert atmosphere.metadata["metal_electron_feedback"] == (
        "charge-neutral EOS and continuum opacity"
    )
    assert atmosphere.metadata["metal_thermodynamic_derivatives"] == (
        "trace-metal approximation: Q-MHD helium derivatives"
    )
    assert atmosphere.metadata[
        "rosseland_opacity_includes_metal_bound_bound_and_bound_free"
    ]


def test_adaptive_helium_checkpoint_resumes_in_formal_flux_phase():
    seed = gray_helium_atmosphere(9_000.0, 8.0, n_depth=8)
    atmosphere = radiative_equilibrium_helium_atmosphere(
        9_000.0,
        8.0,
        stark_table=None,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_lines=False,
        include_helium_ii_lines=False,
        mixing_length_alpha=1.25,
        structure_solver="adaptive-newton",
        n_angle=1,
        initial_temperature=seed.temperature,
        initial_column_mass=seed.column_mass,
        initial_gas_pressure=seed.gas_pressure,
        initial_rosseland_optical_depth=seed.rosseland_optical_depth,
    )

    assert atmosphere.metadata["resumed_directly_in_formal_flux_phase"]
    assert atmosphere.metadata["initial_temperature_was_supplied"]
    assert not atmosphere.metadata["initial_convective_gradient_projection"]
    assert atmosphere.metadata["convective_preconditioner_iterations"] == 0
    assert atmosphere.metadata["convective_preconditioner_iteration_limit"] == 0
    assert atmosphere.metadata["formal_flux_continuations"] == 2
    assert atmosphere.metadata["radiative_equilibrium_iterations"] == 3
    assert not atmosphere.metadata["radiative_equilibrium_converged"]


def test_nonideal_dense_helium_increases_first_ionization(atomic_root: Path):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(
        6000.0, 8.0, n_depth=12, rosseland_opacity=0.01
    )
    ideal = metal_lte_state(
        atmosphere,
        database,
        {"Mg": -10.0},
        include_dense_helium_ionization=False,
    )
    nonideal = metal_lte_state(
        atmosphere,
        database,
        {"Mg": -10.0},
        include_dense_helium_ionization=True,
    )
    helium_mass_density = (
        atmosphere.helium_lte_state.helium_nuclei_density * 6.64647699e-24
    )
    tested_layer = int(np.argmin(dense_helium_ionization_potential_shift_ev(
        "Mg", helium_mass_density, atmosphere.temperature
    )))
    assert (
        nonideal.ion_number_density["Mg"][1:, tested_layer].sum()
        > ideal.ion_number_density["Mg"][1:, tested_layer].sum()
    )


def test_mg_he_table_interpolates_temperature_and_density(tmp_path: Path):
    table_path = tmp_path / "fig5.dat"
    table_path.write_text(
        " 4000 0.3000000E+04 0.1000000E-19\n"
        " 4000 0.4000000E+04 0.3000000E-19\n"
        " 8000 0.3000000E+04 0.4000000E-19\n"
        " 8000 0.4000000E+04 0.1200000E-18\n",
        encoding="utf-8",
    )
    table = read_mg_he_red_wing_table(table_path)
    cross_section = table.cross_section(
        np.asarray([3000.0, 3500.0, 4000.0]),
        np.sqrt(4000.0 * 8000.0),
        5.0e20,
    )
    np.testing.assert_allclose(cross_section, [1.0e-20, 2.0e-20, 3.0e-20])

    capped = table.cross_section(
        np.asarray([3000.0, 3500.0, 4000.0]),
        np.sqrt(4000.0 * 8000.0),
        1.0e22,
    )
    np.testing.assert_allclose(capped, [2.0e-20, 4.0e-20, 6.0e-20])

    density_path = tmp_path / "fig6.dat"
    density_rows = []
    for density, sigma in ((1.0e21, 2.0e-20), (1.0e22, 2.0e-19)):
        for wavelength in np.linspace(3000.0, 4000.0, 60):
            density_rows.append(f"{density:.8e} {wavelength:.8f} {sigma:.10e}\n")
    density_path.write_text("".join(density_rows), encoding="utf-8")
    density_table = read_mg_he_red_wing_table(table_path, density_path)
    interpolated_density = density_table.cross_section(
        np.asarray([3500.0]), 6000.0, np.sqrt(1.0e21 * 1.0e22)
    )
    np.testing.assert_allclose(interpolated_density, [np.sqrt(4.0e-39)])
    high_density = density_table.cross_section(
        np.asarray([3500.0]), 6000.0, 1.0e23
    )
    np.testing.assert_allclose(high_density, [2.0e-19])


def test_metal_line_opacity_is_positive_and_peaks_at_resonance(atomic_root: Path):
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(8000.0, 8.0, n_depth=8)
    state = metal_lte_state(atmosphere, database, {"Mg": -7.0})
    center = database.ions[("Mg", 0)].transitions[0].wavelength_vacuum_angstrom
    neutral_database = AtomicDatabase(MappingProxyType({
        ("Mg", 0): database.ions[("Mg", 0)]
    }))
    wavelength = np.asarray([center - 10.0, center, center + 10.0])
    opacity = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, neutral_database, state, maximum_lines=None
    )
    assert opacity.shape == (3, atmosphere.n_depth)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.all(opacity[1] > opacity[0])
    assert np.all(opacity[1] > opacity[2])
    excluded = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        neutral_database,
        state,
        maximum_lines=None,
        excluded_elements=("Mg",),
    )
    assert np.all(excluded == 0.0)
    selected_transition = database.ions[("Mg", 0)].transitions[0]
    filtered = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        neutral_database,
        state,
        maximum_lines=None,
        transition_keys=((
            "Mg",
            0,
            selected_transition.lower_index,
            selected_transition.upper_index,
        ),),
    )
    assert np.all(filtered >= 0.0)
    assert np.all(filtered <= opacity * (1.0 + 1.0e-13))
    explicitly_selected_despite_cap = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        neutral_database,
        state,
        maximum_lines=0,
        transition_keys=((
            "Mg",
            0,
            selected_transition.lower_index,
            selected_transition.upper_index,
        ),),
    )
    np.testing.assert_allclose(explicitly_selected_despite_cap, filtered)
    absent = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        neutral_database,
        state,
        maximum_lines=None,
        transition_keys=(("Mg", 0, -1, -2),),
    )
    assert np.all(absent == 0.0)


def test_explicit_transition_declaration_is_not_filtered_by_query_endpoints(
    monkeypatch, atomic_root: Path
):
    import wd_spectra.metals as metals_module

    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(8000.0, 8.0, n_depth=8)
    state = metal_lte_state(atmosphere, database, {"Mg": -7.0})
    line = database.ions[("Mg", 0)].transitions[0]
    key = (("Mg", 0, line.lower_index, line.upper_index),)
    centers = []
    original = metals_module._accumulate_lte_metal_line_profiles

    def record(wavelength, center, *args, **kwargs):
        centers.extend(np.asarray(center).tolist())
        return original(wavelength, center, *args, **kwargs)

    monkeypatch.setattr(
        metals_module, "_accumulate_lte_metal_line_profiles", record
    )
    metals_module.metal_line_mass_absorption_coefficient(
        atmosphere,
        np.array([8000.0, 8100.0]),
        database,
        state,
        maximum_lines=0,
        transition_keys=key,
    )
    assert line.wavelength_vacuum_angstrom in centers


def test_adaptive_profile_support_can_be_restricted_by_complete_element():
    def synthetic_ion(element, charge, center=None):
        levels = [AtomicLevel(1, 0.0, 2.0, "ground")]
        transitions = []
        if center is not None:
            levels.append(AtomicLevel(2, 1.0e8 / center, 4.0, "upper"))
            transitions.append(
                AtomicTransition(1, 2, 1.0e8, "E1", center, 0.5)
            )
        return AtomicIon(
            element,
            charge,
            ATOMIC_MASS_U[element],
            IONIZATION_ENERGY_EV[element][charge],
            tuple(levels),
            tuple(transitions),
        )

    database = AtomicDatabase(MappingProxyType({
        ("Mg", 0): synthetic_ion("Mg", 0, 3830.0),
        ("Mg", 1): synthetic_ion("Mg", 1),
        ("Ca", 0): synthetic_ion("Ca", 0, 4228.0),
        ("Ca", 1): synthetic_ion("Ca", 1),
    }))
    atmosphere = gray_helium_atmosphere(8000.0, 8.0, n_depth=8)
    state = metal_lte_state(
        atmosphere, database, {"Mg": -2.0, "Ca": -2.0}
    )
    magnesium = database.ions[("Mg", 0)].transitions[0]
    calcium = database.ions[("Ca", 0)].transitions[0]
    transition_keys = (
        ("Mg", 0, magnesium.lower_index, magnesium.upper_index),
        ("Ca", 0, calcium.lower_index, calcium.upper_index),
    )
    wavelength = np.arange(3800.0, 4250.1, 0.25)
    common = {
        "maximum_lines": None,
        "transition_keys": transition_keys,
        "include_classical_electron_stark": True,
    }
    fixed = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, database, state, **common
    )
    empty = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        database,
        state,
        profile_edge_optical_depth=0.01,
        profile_edge_optical_depth_elements=(),
        **common,
    )
    magnesium_only = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        database,
        state,
        profile_edge_optical_depth=0.01,
        profile_edge_optical_depth_elements=("Mg",),
        **common,
    )
    magnesium_neutral_only = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        database,
        state,
        profile_edge_optical_depth=0.01,
        profile_edge_optical_depth_ions=(("Mg", 0),),
        **common,
    )
    all_elements = metal_line_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        database,
        state,
        profile_edge_optical_depth=0.01,
        **common,
    )

    np.testing.assert_array_equal(empty, fixed)
    magnesium_region = wavelength < 3900.0
    calcium_region = wavelength > 4150.0
    assert np.max(np.abs(
        magnesium_only[magnesium_region] - fixed[magnesium_region]
    )) > 0.0
    np.testing.assert_array_equal(
        magnesium_only[calcium_region], fixed[calcium_region]
    )
    np.testing.assert_array_equal(magnesium_neutral_only, magnesium_only)
    np.testing.assert_array_equal(
        all_elements[magnesium_region], magnesium_only[magnesium_region]
    )
    assert np.max(np.abs(
        all_elements[calcium_region] - magnesium_only[calcium_region]
    )) > 0.0

    with pytest.raises(ValueError, match="mutually exclusive"):
        metal_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            database,
            state,
            profile_edge_optical_depth=0.01,
            profile_edge_optical_depth_elements=("Mg",),
            profile_edge_optical_depth_ions=(("Mg", 0),),
            **common,
        )


def test_compiled_lte_metal_profile_batch_matches_python(monkeypatch):
    import wd_spectra.metals as metals_module

    compiled = getattr(
        getattr(metals_module, "_rt", None),
        "accumulate_lte_metal_line_profiles",
        None,
    )
    if compiled is None:
        pytest.skip("optional LTE metal-profile C kernel is unavailable")
    wavelength = np.linspace(4998.0, 5008.0, 401)
    center = np.asarray([5000.0, 5005.0])
    strength = np.asarray([0.02654, 0.011])
    sigma = np.asarray([[0.08, 0.11], [0.06, 0.09]])
    gamma = np.asarray([[0.03, 0.08], [0.01, 0.04]])
    half_window = np.asarray([0.4, 0.25])
    population = np.asarray([[2.0, 1.0], [0.5, 1.5]])

    accelerated = np.zeros((wavelength.size, 2))
    metals_module._accumulate_lte_metal_line_profiles(
        wavelength,
        center,
        strength,
        sigma,
        gamma,
        half_window,
        population,
        accelerated,
    )
    monkeypatch.setattr(metals_module, "_rt", None)
    reference = np.zeros_like(accelerated)
    metals_module._accumulate_lte_metal_line_profiles(
        wavelength,
        center,
        strength,
        sigma,
        gamma,
        half_window,
        population,
        reference,
    )
    np.testing.assert_allclose(accelerated, reference, rtol=2.0e-14, atol=0.0)


def test_verner_ground_state_fit_and_bound_free_edge(
    atomic_root: Path, tmp_path: Path
):
    table_path = tmp_path / "photo.dat"
    table_path.write_text(
        " 1  1 1.360E+01 5.000E+04 4.298E-01 5.475E+04 "
        "3.288E+01 2.963E+00 0.000E+00 0.000E+00 0.000E+00\n"
        "12 12 7.646E+00 5.490E+01 1.197E+01 1.372E+08 "
        "2.228E-01 1.574E+01 2.805E-01 0.000E+00 0.000E+00\n",
        encoding="ascii",
    )
    hydrogen = read_verner_photoionization_database(
        table_path, elements=("H",), maximum_charge=0
    )
    assert hydrogen.fits[("H", 0)].cross_section(13.6) == pytest.approx(
        6.346e-18, rel=2e-3
    )

    photoionization = read_verner_photoionization_database(
        table_path, elements=("Mg",), maximum_charge=0
    )
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(6000.0, 8.0, n_depth=8)
    state = metal_lte_state(atmosphere, database, {"Mg": -7.0})
    opacity = metal_bound_free_mass_absorption_coefficient(
        atmosphere,
        np.asarray([1500.0, 2000.0]),
        database,
        state,
        photoionization,
    )
    assert np.all(opacity[0] > 0.0)
    assert np.all(opacity[1] == 0.0)
    excluded = metal_bound_free_mass_absorption_coefficient(
        atmosphere,
        np.asarray([1500.0, 2000.0]),
        database,
        state,
        photoionization,
        excluded_elements=("Mg",),
    )
    assert np.all(excluded == 0.0)
    excluded_stage = metal_bound_free_mass_absorption_coefficient(
        atmosphere,
        np.asarray([1500.0, 2000.0]),
        database,
        state,
        photoionization,
        excluded_ions=(("Mg", 0),),
    )
    assert np.all(excluded_stage == 0.0)
    retained_stage = metal_bound_free_mass_absorption_coefficient(
        atmosphere,
        np.asarray([1500.0, 2000.0]),
        database,
        state,
        photoionization,
        excluded_ions=(("Mg", 1),),
    )
    np.testing.assert_allclose(retained_stage, opacity)


def test_verner_reader_allows_documented_element_gaps_unless_strict(
    tmp_path: Path,
):
    table_path = tmp_path / "photo.dat"
    table_path.write_text(
        "12 12 7.646E+00 5.490E+01 1.197E+01 1.372E+08 "
        "2.228E-01 1.574E+01 2.805E-01 0.000E+00 0.000E+00\n",
        encoding="ascii",
    )
    partial = read_verner_photoionization_database(
        table_path, elements=("Mg", "Ti"), maximum_charge=0
    )
    assert set(partial.fits) == {("Mg", 0)}
    with pytest.raises(ValueError, match="missing requested elements: Ti"):
        read_verner_photoionization_database(
            table_path,
            elements=("Mg", "Ti"),
            maximum_charge=0,
            require_all_elements=True,
        )


def test_verner_edges_are_sampled_by_helium_structure_solver(
    atomic_root: Path, tmp_path: Path
):
    table_path = tmp_path / "photo.dat"
    table_path.write_text(
        "12 12 7.646E+00 5.490E+01 1.197E+01 1.372E+08 "
        "2.228E-01 1.574E+01 2.805E-01 0.000E+00 0.000E+00\n",
        encoding="ascii",
    )
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    photoionization = read_verner_photoionization_database(
        table_path, elements=("Mg",), maximum_charge=0
    )
    atmosphere = radiative_equilibrium_helium_atmosphere(
        8000.0,
        8.0,
        stark_table=None,
        n_depth=6,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_lines=False,
        include_helium_ii_lines=False,
        mixing_length_alpha=None,
        metal_database=database,
        metal_abundances={"Mg": -7.0},
        include_metal_lines=False,
        metal_photoionization_database=photoionization,
    )
    assert atmosphere.metadata["radiative_equilibrium_includes_metal_bound_free"]


def test_mg_ii_figure_profile_scales_linearly_with_helium_density(tmp_path: Path):
    table_path = tmp_path / "mgii.dat"
    rows = [f"{1800.0 + 10.0 * i:.1f} {1.0e-18 * (i + 1):.8e}" for i in range(30)]
    table_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    table = read_mg_ii_he_profile_table(table_path)
    cross_section = table.cross_section(
        np.asarray([1800.0, 1900.0]), np.asarray([1.0e21, 2.0e21])
    )
    assert cross_section.shape == (2, 2)
    assert cross_section[0, 0] == pytest.approx(0.5e-18)
    assert cross_section[0, 1] == pytest.approx(1.0e-18)
    assert cross_section[1, 1] == pytest.approx(11.0e-18)


def test_unified_bridge_retains_ordinary_profile_outside_figure_table(
    atomic_root: Path, tmp_path: Path,
):
    table_path = tmp_path / "mgii-limited.dat"
    rows = [f"{1800.0 + 10.0 * i:.1f} 1.0e-18" for i in range(30)]
    table_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    table = read_mg_ii_he_profile_table(table_path)
    database = read_stout_atomic_database(
        atomic_root, elements=("Mg",), maximum_charge=2
    )
    atmosphere = gray_helium_atmosphere(8000.0, 8.0, n_depth=8)
    state = metal_lte_state(atmosphere, database, {"Mg": -7.0})
    wavelength = np.asarray([2790.0, 2803.0, 2810.0])
    ordinary = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, database, state
    )
    bridged = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, database, state,
        mg_ii_he_profile_table=table,
    )
    np.testing.assert_allclose(bridged, ordinary, rtol=2e-14, atol=0.0)


def test_mg_ii_reader_recovers_condition_and_continues_truncated_figure_tails(
    tmp_path: Path,
):
    table_path = tmp_path / "mgii-blouin-figure.dat"
    rows = [
        f"{2600.0 + 10.0 * i:.1f} {1.0e-18 * (1 + min(i, 30 - i)):.8e}"
        for i in range(31)
    ]
    table_path.write_text(
        "# summed Mg II from Blouin, Dufour & Allard 2018, Fig. 1\n"
        "# T=6000 K; n_He=1e22 cm^-3\n"
        + "\n".join(rows) + "\n",
        encoding="ascii",
    )
    table = read_mg_ii_he_profile_table(table_path)
    assert table.temperature_kelvin == pytest.approx(6000.0)
    assert table.helium_density_cm3 == pytest.approx(1.0e22)
    assert table.continue_fitted_log_linear_tails
    cross_section = table.cross_section(
        np.asarray([2500.0, 2600.0, 2900.0, 3000.0]), np.asarray([1.0e22])
    )[:, 0]
    assert cross_section[0] > 0.0
    assert cross_section[3] > 0.0
    assert cross_section[0] < cross_section[1]
    assert cross_section[3] < cross_section[2]


def test_ca_i_figure_profiles_interpolate_in_density(tmp_path: Path):
    table_path = tmp_path / "cai.dat"
    rows = []
    for density, sigma in ((1.0e21, 1.0e-18), (1.0e23, 1.0e-16)):
        rows.extend(
            f"{density:.8e} {3500.0 + 10.0 * i:.1f} {sigma:.8e}"
            for i in range(60)
        )
    table_path.write_text("\n".join(rows) + "\n", encoding="ascii")
    table = read_ca_i_he_profile_table(table_path)
    cross_section = table.cross_section(
        np.asarray([3800.0]),
        np.asarray([5.0e20, 1.0e22, 2.0e23]),
    )
    assert cross_section.shape == (1, 3)
    assert cross_section[0, 0] == pytest.approx(0.5e-18)
    assert cross_section[0, 1] == pytest.approx(1.0e-17)
    assert cross_section[0, 2] == pytest.approx(1.0e-16)


def test_ca_i_profile_combines_published_temperature_and_density_slices(
    tmp_path: Path,
):
    density_path = tmp_path / "cai-density.dat"
    density_rows = []
    for density, sigma in ((1.0e21, 1.0e-18), (1.0e22, 1.0e-17)):
        density_rows.extend(
            f"{density:.8e} {3500.0 + 10.0 * i:.1f} {sigma:.8e}"
            for i in range(60)
        )
    density_path.write_text("\n".join(density_rows) + "\n", encoding="ascii")
    temperature_path = tmp_path / "cai-temperature.dat"
    temperature_rows = []
    for temperature, sigma in ((4000.0, 5.0e-18), (6000.0, 2.0e-17)):
        temperature_rows.extend(
            f"{temperature:.1f} {3500.0 + 10.0 * i:.1f} {sigma:.8e}"
            for i in range(60)
        )
    temperature_path.write_text(
        "\n".join(temperature_rows) + "\n", encoding="ascii"
    )
    table = read_ca_i_he_profile_table(density_path, temperature_path)
    wavelength = np.asarray([3800.0])
    at_reference = table.cross_section(wavelength, 1.0e22, temperature=4000.0)
    at_hot_node = table.cross_section(wavelength, 1.0e22, temperature=6000.0)
    at_midpoint = table.cross_section(
        wavelength, 1.0e22, temperature=np.sqrt(4000.0 * 6000.0)
    )
    np.testing.assert_allclose(at_reference, [1.0e-17])
    np.testing.assert_allclose(at_hot_node, [4.0e-17])
    np.testing.assert_allclose(at_midpoint, [2.0e-17])
    # Outside the calculated temperature range the nearest published profile
    # is retained rather than extrapolating its nonlinear shape.
    np.testing.assert_allclose(
        table.cross_section(wavelength, 1.0e22, temperature=20_000.0),
        at_hot_node,
    )


def test_ca_ii_figure_profile_scales_from_published_density(tmp_path: Path):
    table_path = tmp_path / "caii.dat"
    rows = [f"{3700.0 + 10.0 * i:.1f} 2.0e-17" for i in range(30)]
    table_path.write_text(
        "# figure bridge\n# T=10000 K; n_He=5e21 cm^-3\n"
        + "\n".join(rows) + "\n",
        encoding="ascii",
    )
    table = read_ca_ii_he_profile_table(table_path)
    assert table.temperature_kelvin == pytest.approx(10_000.0)
    assert table.helium_density_cm3 == pytest.approx(5.0e21)
    cross_section = table.cross_section(
        np.asarray([3800.0]), np.asarray([5.0e21, 2.0e22])
    )
    assert cross_section[0, 0] == pytest.approx(2.0e-17)
    assert cross_section[0, 1] == pytest.approx(2.0e-17)


def test_ca_ii_temperature_grid_interpolates_and_scales_density(
    tmp_path: Path,
):
    paths = []
    wavelength = np.linspace(3700.0, 4000.0, 31)
    for temperature, sigma in ((4000.0, 1.0e-17), (10000.0, 4.0e-17)):
        path = tmp_path / f"caii-{temperature:.0f}.dat"
        path.write_text(
            f"# test curve\n# T={temperature:g} K; n_He=5e21 cm^-3\n"
            + "\n".join(
                f"{local_wavelength:.8f} {sigma:.8e}"
                for local_wavelength in wavelength
            )
            + "\n",
            encoding="ascii",
        )
        paths.append(path)
    table = read_ca_ii_he_profile_temperature_grid(paths)
    geometric_temperature = np.sqrt(4000.0 * 10000.0)
    cross_section = table.cross_section(
        np.asarray([3850.0]),
        np.asarray([2.5e21, 1.0e22]),
        temperature=np.asarray([geometric_temperature, 4000.0]),
    )
    assert cross_section.shape == (1, 2)
    assert cross_section[0, 0] == pytest.approx(1.0e-17)
    assert cross_section[0, 1] == pytest.approx(1.0e-17)


def test_ca_ii_temperature_density_grid_interpolates_both_dimensions(
    tmp_path: Path,
):
    paths = []
    wavelength = np.linspace(3700.0, 4000.0, 31)
    for temperature, temperature_scale in ((4000.0, 1.0), (10000.0, 4.0)):
        for density, density_scale in ((1.0e21, 1.0), (1.0e22, 10.0)):
            path = tmp_path / f"caii-{temperature:.0f}-{density:.0e}.dat"
            sigma = 1.0e-18 * temperature_scale * density_scale
            path.write_text(
                f"# test curve\n# T={temperature:g} K; n_He={density:g} cm^-3\n"
                + "\n".join(
                    f"{local_wavelength:.8f} {sigma:.8e}"
                    for local_wavelength in wavelength
                )
                + "\n",
                encoding="ascii",
            )
            paths.append(path)
    table = read_ca_ii_he_profile_grid(paths)
    cross_section = table.cross_section(
        np.asarray([3850.0]),
        np.asarray([np.sqrt(1.0e21 * 1.0e22), 5.0e20, 2.0e22]),
        temperature=np.asarray([
            np.sqrt(4000.0 * 10000.0),
            4000.0,
            10000.0,
        ]),
    )

    assert cross_section.shape == (1, 3)
    assert cross_section[0, 0] == pytest.approx(np.sqrt(4.0e-18 * 4.0e-17))
    assert cross_section[0, 1] == pytest.approx(0.5e-18)
    assert cross_section[0, 2] == pytest.approx(4.0e-17)
