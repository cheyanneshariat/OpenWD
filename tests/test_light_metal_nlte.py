from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import (
    ChiantiScaledCollisionComponent,
    ChiantiTermCollisionStrength,
    TmadStructureModelAtom,
    atomic_database_with_chianti_radiative_transitions,
    atomic_database_with_tmad_formal_ions,
    atomic_database_with_tmad_profile_parameters,
    barklem_oi_electron_collision_data,
    fine_structure_collision_data_from_tmad,
    btm_quadrupole_l_mixing_rate_coefficient,
    gray_helium_atmosphere,
    light_metal_free_free_charge_kernel,
    light_metal_free_free_mass_absorption_coefficient,
    planck_lambda_angstrom,
    psm20_debye_l_mixing_rate_coefficient,
    psm20_l_mixing_rate_coefficient,
    read_chianti_term_collision_strengths,
    read_barklem_mgi_electron_collision_data,
    read_norad_oxygen_vi_photoionization_data,
    read_sirocco_topbase_photoionization_data,
    read_tmad_structure_model_atom,
    read_tlusty_forbidden_collision_strengths,
    read_tlusty_photoionization_threshold_data,
    reduced_light_metal_wavelength,
    rydberg_angular_momentum_mixing_collision_data,
    rydberg_quadrupole_angular_momentum_mixing_collision_data,
    solve_light_metal_ionization_nlte,
    solve_reduced_light_metal_levels_nlte,
    TmadEffectiveDielectronicCoupling,
    TmadLTEBoundBoundCoupling,
)
from wd_spectra.constants import LIGHT_SPEED, PLANCK
from wd_spectra._compat import trapezoid
from wd_spectra.metals import (
    ATOMIC_MASS_U,
    EV_TO_ERG,
    IONIZATION_ENERGY_EV,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    MetalLTEState,
    VernerPhotoionizationDatabase,
    VernerPhotoionizationFit,
)
from wd_spectra.light_metal_nlte import (
    _accumulate_metal_line_profiles_python,
    _holtsmark_microfield_distribution,
    _ion_dynamic_holtsmark_distribution,
    hot_metal_line_nlte_coefficients,
    _impact_profile_damping_rate,
    _mapped_fine_structure_components,
    _photoionization_rates,
    _photoionization_rates_python,
    _profile_weighted_line_means,
    _tmap_seaton_collisional_ionization_rate,
    _profile_weighted_line_means_python,
    _tabulated_electron_stark_fwhm_angstrom,
)

try:
    from wd_spectra import _rt
except ImportError:  # pragma: no cover - source-only installation
    _rt = None


def test_chianti_scaled_collision_component_descales_type_two_midpoint():
    component = ChiantiScaledCollisionComponent(
        transition_energy_rydberg=1.0,
        transition_type=2,
        scaling_parameter=1.0,
        scaled_temperature=(0.0, 0.5, 1.0),
        scaled_upsilon=(1.0, 2.0, 3.0),
    )
    # kT / Delta-E = 1 gives the Burgess--Tully type-2 coordinate x=1/2.
    assert component.effective_collision_strength(1.57888e5) == pytest.approx(2.0)


def test_psm20_debye_rate_matches_badnell_high_l_benchmark():
    # Badnell et al. (2021), Table 1: He(n=30,l=29)+p at 10^4 K and
    # N_H=100 cm^-3 has q(29->28)=6.18 cm^3/s.  The published value includes
    # more precise mass/plasma conventions, so use their quoted few-percent
    # comparison accuracy rather than asserting spurious extra digits.
    coefficient = psm20_debye_l_mixing_rate_coefficient(
        30,
        29,
        28,
        1.0e4,
        100.0,
        4.002_602,
        1.0,
        1.007_84,
        1.0,
    )
    assert coefficient == pytest.approx(6.18, rel=0.04)
    integrated = psm20_l_mixing_rate_coefficient(
        30,
        29,
        28,
        1.0e4,
        100.0,
        4.002_602,
        1.0,
        1.007_84,
        1.0,
    )
    assert integrated == pytest.approx(coefficient, rel=2.0e-3)


def test_psm20_term_splitting_reduces_non_degenerate_rate():
    debye_only = psm20_debye_l_mixing_rate_coefficient(
        7, 4, 3, 8.3e4, 9.4e14, 15.999, 2.0, 4.002_602, 6.0
    )
    split = psm20_l_mixing_rate_coefficient(
        7,
        4,
        3,
        8.3e4,
        9.4e14,
        15.999,
        2.0,
        4.002_602,
        6.0,
        energy_splitting_wavenumber=3.0,
    )
    assert split < 0.03 * debye_only


@pytest.mark.parametrize(
    ("initial_l", "final_l", "published_coefficient"),
    ((1, 3, 3.04), (9, 11, 1.20), (17, 19, 0.0524)),
)
def test_btm_quadrupole_rate_matches_published_benchmarks(
    initial_l: int, final_l: int, published_coefficient: float
):
    # Deliporanidou et al. (2025), Table 1: H(n=20,l)+p at 100 K.
    coefficient = btm_quadrupole_l_mixing_rate_coefficient(
        20,
        initial_l,
        final_l,
        100.0,
        1.007_84,
        1.0,
        1.007_84,
        1.0,
    )
    assert coefficient == pytest.approx(published_coefficient, rel=0.025)


def test_rydberg_l_mixing_builder_uses_only_adjacent_high_l_terms():
    ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][5],
        levels=(
            AtomicLevel(1, 100.0, 10.0, "O607D  2D"),
            AtomicLevel(2, 103.0, 14.0, "O607F  2FO"),
            AtomicLevel(3, 102.0, 18.0, "O607G  2G"),
            AtomicLevel(4, 104.0, 22.0, "O607H  2HO"),
            AtomicLevel(5, 200.0, 14.0, "O608F  2FO"),
            AtomicLevel(6, 201.0, 18.0, "O608G  2G"),
        ),
        transitions=(),
        source="synthetic O VI Rydberg atom",
    )
    database = AtomicDatabase(
        MappingProxyType({("O", 5): ion}), source="synthetic O VI atom"
    )
    collisions = rydberg_angular_momentum_mixing_collision_data(
        database, "O", 5
    )
    assert set(collisions) == {
        ("O", 5, 3, 2),  # n=7 F-G, ordered by term energy.
        ("O", 5, 3, 4),  # n=7 G-H.
        ("O", 5, 5, 6),  # n=8 F-G.
    }
    assert all(record.target_core_charge == 6.0 for record in collisions.values())


def test_barklem_oi_collision_builder_preserves_ls_term_rates():
    levels = (
        AtomicLevel(1, 0.0, 5.0, "2s2.2p4.(3P<2>)"),
        AtomicLevel(2, 100.0, 3.0, "2s2.2p4.(3P<1>)"),
        AtomicLevel(3, 200.0, 1.0, "2s2.2p4.(3P<0>)"),
        AtomicLevel(4, 15_000.0, 5.0, "2s2.2p4.(1D<2>)"),
        AtomicLevel(5, 33_000.0, 1.0, "2s2.2p4.(1S<0>)"),
        AtomicLevel(6, 73_000.0, 5.0, "2s2.2p3.3s.(5So<2>)"),
        AtomicLevel(7, 76_000.0, 3.0, "2s2.2p3.3s.(3So<1>)"),
        AtomicLevel(8, 86_000.0, 3.0, "2s2.2p3.3p.(5P<1>)"),
        AtomicLevel(9, 86_100.0, 5.0, "2s2.2p3.3p.(5P<2>)"),
        AtomicLevel(10, 86_200.0, 7.0, "2s2.2p3.3p.(5P<3>)"),
        AtomicLevel(11, 88_000.0, 3.0, "2s2.2p3.3p.(3P<1>)"),
        AtomicLevel(12, 88_100.0, 5.0, "2s2.2p3.3p.(3P<2>)"),
        AtomicLevel(13, 88_200.0, 1.0, "2s2.2p3.3p.(3P<0>)"),
    )
    ion = AtomicIon(
        element="O",
        charge=0,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][0],
        levels=levels,
        transitions=(),
        source="synthetic O I fine atom",
    )
    database = AtomicDatabase(
        MappingProxyType({("O", 0): ion}), source="synthetic O I atom"
    )
    collisions = barklem_oi_electron_collision_data(database)
    quintet_rates = sum(
        collisions[("O", 0, 6, upper)].upward_rate_coefficient(12_000.0)
        for upper in (8, 9, 10)
    )
    triplet_rates = sum(
        collisions[("O", 0, 7, upper)].upward_rate_coefficient(12_000.0)
        for upper in (11, 12, 13)
    )
    assert quintet_rates == pytest.approx(1.20e-7)
    assert triplet_rates == pytest.approx(1.26e-7)
    record = collisions[("O", 0, 6, 8)]
    assert record.upward_rate_coefficient(np.sqrt(8_000.0 * 12_000.0)) == pytest.approx(
        np.sqrt(
            record.upward_rate_coefficient_cm3_s[3]
            * record.upward_rate_coefficient_cm3_s[4]
        )
    )


def test_barklem_mgi_collision_reader_preserves_term_rate(tmp_path: Path):
    ion = AtomicIon(
        element="Mg",
        charge=0,
        atomic_mass_u=ATOMIC_MASS_U["Mg"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["Mg"][0],
        levels=(
            AtomicLevel(1, 0.0, 1.0, "3s2.(1S<0>)"),
            AtomicLevel(2, 21_850.0, 1.0, "3s.3p.(3Po<0>)"),
            AtomicLevel(3, 21_870.0, 3.0, "3s.3p.(3Po<1>)"),
            AtomicLevel(4, 21_910.0, 5.0, "3s.3p.(3Po<2>)"),
        ),
        transitions=(),
        source="synthetic Mg I fine atom",
    )
    database = AtomicDatabase(
        MappingProxyType({("Mg", 0): ion}), source="synthetic Mg I atom"
    )
    (tmp_path / "states.dat").write_text(
        "  1    3s2_1S    1  0.000\n"
        "  2    3s3p_3P   9  2.714\n",
        encoding="ascii",
    )
    (tmp_path / "1000k_c.dat").write_text(
        "0.0 2.0\n2.0 0.0\n", encoding="ascii"
    )
    (tmp_path / "2000k_c.dat").write_text(
        "0.0 4.0\n4.0 0.0\n", encoding="ascii"
    )
    collisions = read_barklem_mgi_electron_collision_data(
        tmp_path, database
    )
    summed_rate = sum(
        collisions[("Mg", 0, 1, upper)].upward_rate_coefficient(1_000.0)
        for upper in (2, 3, 4)
    )
    expected = (
        8.63e-6
        * 2.0
        / np.sqrt(1_000.0)
        * np.exp(-2.714 * EV_TO_ERG / (1.380_649e-16 * 1_000.0))
    )
    assert summed_rate == pytest.approx(expected)
    assert len(collisions) == 3


def test_rydberg_quadrupole_builder_connects_delta_l_two_terms():
    ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][5],
        levels=(
            AtomicLevel(1, 100.0, 14.0, "O607F  2FO"),
            AtomicLevel(2, 101.0, 18.0, "O607G  2G"),
            AtomicLevel(3, 102.0, 22.0, "O607H  2HO"),
            AtomicLevel(4, 103.0, 26.0, "O607I  2I"),
            AtomicLevel(5, 200.0, 14.0, "O608F  2FO"),
            AtomicLevel(6, 202.0, 22.0, "O608H  2HO"),
        ),
        transitions=(),
        source="synthetic O VI Rydberg atom",
    )
    database = AtomicDatabase(
        MappingProxyType({("O", 5): ion}), source="synthetic O VI atom"
    )
    collisions = rydberg_quadrupole_angular_momentum_mixing_collision_data(
        database, "O", 5
    )
    assert set(collisions) == {
        ("O", 5, 1, 3),  # n=7 F-H.
        ("O", 5, 2, 4),  # n=7 G-I.
        ("O", 5, 5, 6),  # n=8 F-H.
    }
    assert all(record.target_core_charge == 6.0 for record in collisions.values())


def test_tlusty_reader_retains_collision_only_forbidden_link(tmp_path: Path):
    ionization_frequency = IONIZATION_ENERGY_EV["C"][2] * EV_TO_ERG / PLANCK
    excited_frequency = ionization_frequency - 1.0e15
    path = tmp_path / "c3.dat"
    path.write_text(
        f"""****** Levels
 {ionization_frequency:.12E} 1. 3 'CIII 1Se 1' 0 0. 0.
 {excited_frequency:.12E} 9. 3 'CIII 3Po 1' 0 0. 0.
****** Continuum transitions
*** Line transitions
 1 2 0 0 4 0 0 0.000E+00 5.000E-02
""",
        encoding="ascii",
    )
    ion = AtomicIon(
        element="C",
        charge=2,
        atomic_mass_u=ATOMIC_MASS_U["C"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["C"][2],
        levels=(
            AtomicLevel(1, 0.0, 1.0, "C302S2 1S"),
            AtomicLevel(2, 1.0e15 / LIGHT_SPEED, 9.0, "C302P  3PO"),
        ),
        transitions=(),
        source="synthetic population atom",
    )
    model_atom = TmadStructureModelAtom(
        atomic_database=AtomicDatabase(
            MappingProxyType({("C", 2): ion}),
            source="synthetic population atom",
        ),
        levels_per_charge=MappingProxyType({2: 2}),
        formal_level_mapping=MappingProxyType({}),
        formal_lte_parent_mapping=MappingProxyType({}),
        continuum_parent_mapping=MappingProxyType({}),
        lte_level_reservoir=MappingProxyType({}),
        lte_bound_bound_couplings=(),
        collision_data=MappingProxyType({}),
        source="synthetic population atom",
    )
    collisions = read_tlusty_forbidden_collision_strengths(
        path, model_atom.atomic_database, "C", 2
    )
    record = collisions[("C", 2, 1, 2)]
    assert record.effective_collision_strength(100_000.0) == pytest.approx(0.05)


def test_sirocco_topbase_reader_collapses_identical_fine_components(
    tmp_path: Path,
):
    levels_path = tmp_path / "o_6_levels.dat"
    levels_path.write_text(
        """LevMacro 8 6 1 -138.1 0.0 2.0 1e21 1s2.2s_2S0.5 200 1
LevMacro 8 6 2 -126.1 1.0 2.0 1e-9 1s2.2p_2P0.5 211 1
LevMacro 8 6 3 -126.1 1.0 4.0 1e-9 1s2.2p_2P1.5 211 1
LevMacro 8 6 4 -54.4 2.0 4.0 1e-9 1s2.3d_2D1.5 220 1
""",
        encoding="ascii",
    )
    photo_path = tmp_path / "o_6_phot.dat"
    photo_path.write_text(
        """PhotMacS 8 6 1 1 138.0 2
PhotMac 138.0 3.0e-19
PhotMac 276.0 3.0e-20
PhotMacS 8 6 2 1 125.999 2
PhotMac 126.0 4.0e-19
PhotMac 252.0 4.0e-20
PhotMacS 8 6 3 1 125.999 2
PhotMac 126.0 4.0e-19
PhotMac 252.0 4.0e-20
PhotMacS 8 6 4 1 54.0 2
PhotMac 54.0 5.0e-19
PhotMac 108.0 5.0e-20
""",
        encoding="ascii",
    )
    ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][5],
        levels=(
            AtomicLevel(1, 0.0, 2.0, "O602S  2S"),
            AtomicLevel(2, 96_000.0, 6.0, "O602P  2PO"),
            AtomicLevel(3, 670_000.0, 10.0, "O603D  2D"),
            AtomicLevel(4, 675_000.0, 14.0, "O603F  2FO"),
        ),
        transitions=(),
        source="synthetic TMAD term atom",
    )
    data = read_sirocco_topbase_photoionization_data(
        levels_path, photo_path, ion
    )
    assert data.level_label == ("O602S  2S", "O602P  2PO", "O603D  2D")
    np.testing.assert_allclose(
        data.threshold_cross_section_cm2,
        (3.0e-19, 4.0e-19, 5.0e-19),
    )
    assert data.threshold_cross_section_includes_gbar is False
    threshold = data.threshold_frequency_hz[1]
    evaluated = data.cross_section(
        np.asarray((threshold, threshold * 252.0 / 125.999)),
        threshold,
        level_label="O602P  2PO",
    )
    np.testing.assert_allclose(evaluated, (4.0e-19, 4.0e-20))

    tlusty_collision_data = replace(
        data,
        line_wavelength_vacuum_angstrom=np.asarray((1031.9, 1037.6)),
        line_collision_gbar=np.asarray((0.31, 0.31)),
    )
    with_collisions = read_sirocco_topbase_photoionization_data(
        levels_path,
        photo_path,
        ion,
        line_collision_data=tlusty_collision_data,
    )
    np.testing.assert_allclose(
        with_collisions.line_wavelength_vacuum_angstrom, (1031.9, 1037.6)
    )
    np.testing.assert_allclose(with_collisions.line_collision_gbar, (0.31, 0.31))


def test_norad_oxygen_vi_reader_maps_partial_sections_by_n_and_l(
    tmp_path: Path,
):
    energy_path = tmp_path / "o6.en.ls.txt"
    energy_path.write_text(
        """8 2 E
2 3 1 2
1 0.000000E+00
1 T 1 7 3 -7.34723E-01
2 T 1 8 3 -5.62518E-01
2 4 0 1
1 0.000000E+00
1 T 1 8 4 -5.62495E-01
0 0 0 0
""",
        encoding="ascii",
    )
    photo_path = tmp_path / "o6.ptpx.ls.txt"
    photo_path.write_text(
        """8 2 P
2 3 1 1
1 3 3
7.34723E-01 0.1000
7.34723E-01 2.40
1.469446E+00 0.60
2.204169E+00 0.20
2 3 1 2
1 3 3
5.62518E-01 0.1000
5.62519E-01 2.60
1.125036E+00 0.65
1.687554E+00 0.21
2 4 0 1
1 3 3
5.62495E-01 0.1000
5.62495E-01 0.63
1.124990E+00 0.16
1.687485E+00 0.05
0 0 0 0
""",
        encoding="ascii",
    )
    ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][5],
        levels=(
            AtomicLevel(1, 0.0, 2.0, "O602S  2S"),
            AtomicLevel(2, 800_000.0, 14.0, "O607F  2FO"),
            AtomicLevel(3, 820_000.0, 14.0, "O608F  2FO"),
            AtomicLevel(4, 820_100.0, 18.0, "O608G  2G"),
        ),
        transitions=(),
        source="synthetic TMAD O VI term atom",
    )

    data = read_norad_oxygen_vi_photoionization_data(
        energy_path,
        photo_path,
        ion,
    )

    assert data.level_label == ("O607F  2FO", "O608F  2FO", "O608G  2G")
    np.testing.assert_allclose(
        data.threshold_cross_section_cm2,
        (2.40e-18, 2.60e-18, 0.63e-18),
    )
    threshold = data.threshold_frequency_hz[1]
    evaluated = data.cross_section(
        np.asarray((threshold, 2.0 * threshold)),
        threshold,
        level_label="O608F  2FO",
    )
    np.testing.assert_allclose(evaluated, (2.60e-18, 0.65e-18))


def test_chianti_reader_sums_fine_components_onto_population_terms(
    tmp_path: Path,
):
    population_ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=ATOMIC_MASS_U["O"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["O"][5],
        levels=(
            AtomicLevel(1, 0.0, 2.0, "2s 2S"),
            AtomicLevel(2, 100_000.0, 6.0, "2p 2P"),
        ),
        transitions=(),
        source="synthetic population atom",
    )
    population_database = AtomicDatabase(
        MappingProxyType({("O", 5): population_ion}),
        source="synthetic population atom",
    )
    model_atom = TmadStructureModelAtom(
        atomic_database=population_database,
        levels_per_charge=MappingProxyType({5: 2}),
        formal_level_mapping=MappingProxyType({
            ("O", 5, 1): (("O", 5, 1),),
            ("O", 5, 2): (("O", 5, 2), ("O", 5, 3)),
        }),
        formal_lte_parent_mapping=MappingProxyType({}),
        continuum_parent_mapping=MappingProxyType({}),
        lte_level_reservoir=MappingProxyType({}),
        lte_bound_bound_couplings=(),
        collision_data=MappingProxyType({}),
        source="synthetic term map",
    )
    scups = tmp_path / "o_6.scups"
    scups.write_text(
        """1 2 1.0 0.1 0.2 3 2 1.0
0.0 0.5 1.0
1.0 1.0 1.0
1 3 1.0 0.1 0.2 3 2 1.0
0.0 0.5 1.0
2.0 2.0 2.0
-1
""",
        encoding="ascii",
    )
    collisions = read_chianti_term_collision_strengths(
        scups, model_atom, "O", 5
    )
    record = collisions[("O", 5, 1, 2)]
    assert isinstance(record, ChiantiTermCollisionStrength)
    assert len(record.components) == 2
    assert record.effective_collision_strength(1.57888e5) == pytest.approx(3.0)


def test_chianti_radiative_reader_fills_only_missing_endpoint_pairs(
    tmp_path: Path,
):
    ion = AtomicIon(
        element="P",
        charge=4,
        atomic_mass_u=ATOMIC_MASS_U["P"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["P"][4],
        levels=(
            AtomicLevel(1, 0.0, 2.0, "3s 2S"),
            AtomicLevel(2, 100_000.0, 4.0, "3p 2P"),
        ),
        transitions=(),
        source="synthetic Stout ion",
    )
    database = AtomicDatabase(
        MappingProxyType({("P", 4): ion}), source="synthetic database"
    )
    wgfa = tmp_path / "p_5.wgfa"
    wgfa.write_text(
        "1 2 1000.0 6.0e-1 1.0e8 3s 2S - 3p 2P\n-1\n",
        encoding="ascii",
    )
    augmented = atomic_database_with_chianti_radiative_transitions(
        database, wgfa, "P", 4
    )
    transition = augmented.ions[("P", 4)].transitions[0]
    assert transition.wavelength_vacuum_angstrom == pytest.approx(1000.0)
    assert transition.absorption_oscillator_strength == pytest.approx(0.3)
    assert transition.einstein_a == pytest.approx(1.0e8)
    assert database.ions[("P", 4)].transitions == ()


def test_holtsmark_microfield_distribution_matches_reference_values():
    beta = np.asarray((0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0))
    expected = np.asarray((
        4.2439353e-5,
        4.2245323e-3,
        2.7122081e-1,
        3.3693878e-1,
        4.1180237e-2,
        5.5613463e-3,
        1.5036946e-5,
    ))
    np.testing.assert_allclose(
        _holtsmark_microfield_distribution(beta), expected, rtol=2.0e-3
    )
    integration_grid = np.geomspace(1.0e-6, 1.0e5, 20_000)
    integral = trapezoid(
        _holtsmark_microfield_distribution(integration_grid), integration_grid
    )
    assert abs(integral - 1.0) < 2.0e-3


def test_formula4_uses_half_the_microfield_magnitude_distribution_per_side():
    # The Holtsmark field-magnitude distribution integrates to unity over
    # beta >= 0.  TMAP formula 4 uses it symmetrically in frequency, so each
    # side must carry half the area.  Its 0.0368/1.385 prefactor then recovers
    # pi e^2/(m_e c), rather than twice the oscillator-strength integral.
    beta = np.geomspace(1.0e-6, 1.0e5, 20_000)
    two_sided_area = 2.0 * trapezoid(
        0.5 * _holtsmark_microfield_distribution(beta), beta
    )
    assert two_sided_area == pytest.approx(1.0, abs=2.0e-3)
    assert 0.0368 / 1.385 == pytest.approx(
        np.pi * 4.803_204_712_57e-10**2
        / (9.109_383_7139e-28 * 2.997_924_58e10),
        rel=2.0e-3,
    )


def test_ion_motion_fills_holtsmark_center_and_preserves_profile_area():
    beta = np.linspace(0.0, 80.0, 160_001)
    static = _ion_dynamic_holtsmark_distribution(beta, 0.0)
    moving = _ion_dynamic_holtsmark_distribution(beta, 0.15)

    assert static[0] == 0.0
    assert moving[0] > 0.02
    assert 2.0 * trapezoid(moving, beta) == pytest.approx(1.0, rel=5.0e-3)
    # Over the beta<=30 support used by formula 4 the dynamic correction is
    # still core dominated; the static wing remains the leading term.
    assert moving[np.searchsorted(beta, 10.0)] < 1.25 * static[
        np.searchsorted(beta, 10.0)
    ]


def test_c_iv_resonance_uses_published_electron_impact_widths():
    assert _tabulated_electron_stark_fwhm_angstrom(
        "C", 3, 1548.20, 1.0e17, 100_000.0
    ) == pytest.approx(0.00561)
    assert _tabulated_electron_stark_fwhm_angstrom(
        "C", 3, 1550.77, 2.0e17, 200_000.0
    ) == pytest.approx(0.00860)


def test_extended_impact_support_does_not_extrapolate_static_holtsmark_wing():
    center = 1_000.0
    center_cm = center * 1.0e-8
    field_scale = (
        1.0 * 2.997_924_58e10 / (30.0 * center_cm**2 * 1.0e8)
    )
    wavelength = np.ascontiguousarray((999.5, 1_000.0, 1_000.5, 1_002.0))
    planck = np.ones((wavelength.size, 1))
    absorption = np.zeros_like(planck)
    emissivity = np.zeros_like(planck)
    _accumulate_metal_line_profiles_python(
        wavelength,
        planck,
        np.asarray((center,)),
        np.asarray((0.0,)),
        np.asarray(((0.01,),)),
        np.asarray(((0.0,),)),
        np.asarray((3.0,)),
        np.asarray(((field_scale,),)),
        np.asarray(((1.0e-18,),)),
        np.zeros((1, 1)),
        np.asarray(((1.0e10,),)),
        np.asarray(((1.0,),)),
        np.asarray(((0.0,),)),
        np.asarray(((0.0,),)),
        absorption,
        emissivity,
        False,
    )
    assert absorption[0, 0] > 0.0
    assert absorption[2, 0] > 0.0
    assert absorption[3, 0] == 0.0


def test_compiled_metal_line_profile_accumulation_matches_python():
    if _rt is None or not hasattr(_rt, "accumulate_metal_line_profiles"):
        return
    wavelength = np.ascontiguousarray(np.concatenate((
        np.linspace(1025.0, 1040.0, 400),
        np.linspace(5780.0, 5820.0, 800),
    )))
    n_depth = 3
    planck = np.ascontiguousarray(
        1.0e10
        * (1.0 + wavelength[:, np.newaxis] / 6000.0)
        * np.asarray((1.0, 1.5, 2.0))[np.newaxis, :]
    )
    center = np.ascontiguousarray((1031.9, 5801.3))
    strength = np.ascontiguousarray((2.0e-3, 7.0e-3))
    gaussian = np.ascontiguousarray((
        (0.025, 0.035, 0.05),
        (0.06, 0.08, 0.1),
    ))
    lorentz = np.ascontiguousarray((
        (0.001, 0.01, 0.03),
        (0.004, 0.02, 0.08),
    ))
    minimum_half_window = np.ascontiguousarray((2.0, 0.0))
    static_scale = np.ascontiguousarray((
        (0.0, 2.0e12, 8.0e12),
        (0.0, 0.0, 0.0),
    ))
    static_amplitude = np.ascontiguousarray((
        (0.0, 2.0e-18, 8.0e-19),
        (0.0, 0.0, 0.0),
    ))
    population = np.ascontiguousarray((
        (1.0e8, 2.0e8, 4.0e8),
        (3.0e7, 7.0e7, 1.1e8),
    ))
    lower_departure = np.ascontiguousarray((
        (0.7, 0.9, 1.2),
        (1.1, 0.8, 0.6),
    ))
    upper_departure = np.ascontiguousarray((
        (0.5, 0.7, 0.9),
        (1.3, 0.6, 0.4),
    ))
    exponential = np.ascontiguousarray((
        (0.1, 0.2, 0.3),
        (0.5, 0.7, 0.9),
    ))
    python_absorption = np.zeros_like(planck)
    python_emissivity = np.zeros_like(planck)
    _accumulate_metal_line_profiles_python(
        wavelength, planck, center, strength, gaussian, lorentz,
        minimum_half_window,
        static_scale, static_amplitude, np.zeros_like(static_scale),
        population, lower_departure,
        upper_departure, exponential, python_absorption, python_emissivity,
        True,
    )
    far_wing = int(np.argmin(np.abs(wavelength - (center[0] + 1.0))))
    assert python_absorption[far_wing, 0] > 0.0
    compiled_absorption = np.zeros_like(planck)
    compiled_emissivity = np.zeros_like(planck)
    from wd_spectra.light_metal_nlte import (
        _HOLTSMARK_COMPILED_ARGUMENT,
        _HOLTSMARK_COMPILED_WEIGHT,
    )
    _rt.accumulate_metal_line_profiles(
        wavelength, planck, center, strength, gaussian, lorentz,
        minimum_half_window,
        static_scale, static_amplitude, population, lower_departure,
        upper_departure, exponential, _HOLTSMARK_COMPILED_ARGUMENT,
        _HOLTSMARK_COMPILED_WEIGHT, compiled_absorption,
        compiled_emissivity, True,
    )
    np.testing.assert_allclose(
        compiled_absorption, python_absorption, rtol=3.0e-13, atol=0.0
    )
    np.testing.assert_allclose(
        compiled_emissivity, python_emissivity, rtol=3.0e-13, atol=0.0
    )
    if getattr(_rt, "METAL_PROFILE_DEPTH_MAJOR", 0):
        depth_absorption = np.zeros((n_depth, wavelength.size))
        depth_emissivity = np.zeros_like(depth_absorption)
        _rt.accumulate_metal_line_profiles(
            wavelength, np.ascontiguousarray(planck.T), center, strength,
            gaussian, lorentz, minimum_half_window, static_scale,
            static_amplitude, population, lower_departure, upper_departure,
            exponential, _HOLTSMARK_COMPILED_ARGUMENT,
            _HOLTSMARK_COMPILED_WEIGHT, depth_absorption,
            depth_emissivity, True, True,
        )
        np.testing.assert_array_equal(depth_absorption.T, compiled_absorption)
        np.testing.assert_array_equal(depth_emissivity.T, compiled_emissivity)


def test_compiled_metal_line_rate_means_match_python():
    wavelength = np.ascontiguousarray(np.concatenate((
        np.linspace(1025.0, 1040.0, 400),
        np.linspace(5780.0, 5820.0, 800),
    )))
    intensity = np.ascontiguousarray(
        (1.0 + 0.2 * np.sin(wavelength[:, np.newaxis] / 7.0))
        * np.asarray((1.0, 1.7, 2.4))[np.newaxis, :]
    )
    lambda_diagonal = np.ascontiguousarray(
        (0.2 + 0.1 * np.cos(wavelength[:, np.newaxis] / 11.0))
        * np.asarray((0.7, 1.0, 1.2))[np.newaxis, :]
    )
    center = np.ascontiguousarray((1031.9, 5801.3, 1039.7))
    gaussian = np.ascontiguousarray((
        (0.025, 0.035, 0.05),
        (0.06, 0.08, 0.1),
        (1.0e-5, 1.0e-5, 1.0e-5),
    ))
    lorentz = np.ascontiguousarray((
        (0.001, 0.01, 0.03),
        (0.004, 0.02, 0.08),
        (1.0e-7, 1.0e-7, 1.0e-7),
    ))
    expected_j, expected_lambda = _profile_weighted_line_means_python(
        wavelength, intensity, lambda_diagonal, center, gaussian, lorentz
    )
    actual_j, actual_lambda = _profile_weighted_line_means(
        wavelength, intensity, lambda_diagonal, center, gaussian, lorentz
    )
    np.testing.assert_allclose(actual_j, expected_j, rtol=5.0e-14)
    np.testing.assert_allclose(actual_lambda, expected_lambda, rtol=5.0e-14)
    no_lambda_j, no_lambda = _profile_weighted_line_means(
        wavelength, intensity, None, center, gaussian, lorentz
    )
    np.testing.assert_allclose(no_lambda_j, expected_j, rtol=5.0e-14)
    np.testing.assert_array_equal(no_lambda, 0.0)


def test_formula4_static_wings_enter_metal_line_rate_mean():
    wavelength = np.ascontiguousarray(np.linspace(999.0, 1001.0, 2001))
    distance = np.abs(wavelength - 1000.0)
    intensity = np.ascontiguousarray((1.0 + distance)[:, np.newaxis])
    lambda_diagonal = np.ascontiguousarray(
        (0.1 + 0.3 * distance)[:, np.newaxis]
    )
    center = np.ascontiguousarray((1000.0,))
    gaussian = np.ascontiguousarray(((0.015,),))
    lorentz = np.ascontiguousarray(((0.001,),))

    impact_j, impact_lambda = _profile_weighted_line_means(
        wavelength,
        intensity,
        lambda_diagonal,
        center,
        gaussian,
        lorentz,
    )
    static_j, static_lambda = _profile_weighted_line_means(
        wavelength,
        intensity,
        lambda_diagonal,
        center,
        gaussian,
        lorentz,
        integrated_strength=np.ascontiguousarray((2.5e-2,)),
        static_frequency_scale=np.ascontiguousarray(((1.0e11,),)),
        static_amplitude=np.ascontiguousarray(((1.0,),)),
    )

    assert static_j[0, 0] > impact_j[0, 0] + 0.05
    assert static_lambda[0, 0] > impact_lambda[0, 0] + 0.015


def test_formula4_ion_dynamic_scale_contracts_only_selected_series():
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=3)
    ions = {}
    for charge in range(4):
        levels = (AtomicLevel(1, 0.0, 2.0, "ground"),)
        transitions = ()
        if charge == 3:
            levels = (
                AtomicLevel(1, 0.0, 2.0, "4d"),
                AtomicLevel(2, 100_000.0, 4.0, "9f"),
            )
            transitions = (
                AtomicTransition(
                    1,
                    2,
                    1.0e8,
                    "E1",
                    1135.0,
                    0.02,
                    4,
                    (0.0, 1.0e8, 1.0, 4.0, 4.0, 9.0),
                ),
            )
        ions[("C", charge)] = AtomicIon(
            "C",
            charge,
            ATOMIC_MASS_U["C"],
            IONIZATION_ENERGY_EV["C"][charge],
            levels,
            transitions,
            "synthetic formula-4 atom",
        )
    database = AtomicDatabase(MappingProxyType(ions), "synthetic formula-4 atom")
    populations = np.zeros((4, atmosphere.n_depth))
    populations[3] = 1.0e12
    lte_state = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({"C": -1.0}),
        element_number_density=MappingProxyType({"C": populations.sum(axis=0)}),
        ion_number_density=MappingProxyType({"C": populations}),
        partition_function=MappingProxyType({
            ("C", charge): ion.partition_function(atmosphere.temperature)
            for (_, charge), ion in ions.items()
        }),
        electron_density=atmosphere.electron_density,
        metal_electron_density=np.zeros(atmosphere.n_depth),
    )
    departures = MappingProxyType({
        ("C", charge): np.ones(atmosphere.n_depth) for charge in range(4)
    })
    wavelength = np.linspace(1125.0, 1145.0, 2001)
    static, _ = hot_metal_line_nlte_coefficients(
        atmosphere,
        wavelength,
        database,
        lte_state,
        departures,
        minimum_oscillator_strength=1.0e-6,
    )
    narrowed, _ = hot_metal_line_nlte_coefficients(
        atmosphere,
        wavelength,
        database,
        lte_state,
        departures,
        minimum_oscillator_strength=1.0e-6,
        static_linear_stark_frequency_scales={("C", 3, 4, 9): 0.25},
    )
    unmatched, _ = hot_metal_line_nlte_coefficients(
        atmosphere,
        wavelength,
        database,
        lte_state,
        departures,
        minimum_oscillator_strength=1.0e-6,
        static_linear_stark_frequency_scales={("C", 3, 4, 10): 0.25},
    )
    center = np.argmin(np.abs(wavelength - 1135.0))
    wing = np.argmin(np.abs(wavelength - 1138.0))
    np.testing.assert_allclose(narrowed[center], static[center])
    # The middle layer is static-wing dominated; the deepest layer is already
    # impact dominated and should therefore remain unchanged by this option.
    assert narrowed[wing, 1] < 0.2 * static[wing, 1]
    np.testing.assert_allclose(unmatched, static)


@pytest.mark.parametrize("n_depth", [1, 4, 8, 40, 41])
def test_compiled_photoionization_rates_match_python(n_depth):
    wavelength = np.ascontiguousarray(np.geomspace(90.0, 3000.0, 1600))
    temperature_shape = np.geomspace(0.7, 2.0, n_depth)[np.newaxis, :]
    intensity = np.ascontiguousarray(
        (wavelength[:, np.newaxis] / 500.0) ** -1.3 * temperature_shape
    )
    cross_section = np.ascontiguousarray(np.asarray((
        np.where(wavelength < 900.0, 2.0e-18 * (wavelength / 900.0) ** 3, 0.0),
        np.where(wavelength < 300.0, 7.0e-19 * (wavelength / 300.0) ** 2, 0.0),
        np.zeros_like(wavelength),
    )))
    expected = _photoionization_rates_python(
        wavelength, intensity, cross_section
    )
    actual = _photoionization_rates(wavelength, intensity, cross_section)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=0.0)


def test_tmad_profile_flags_control_impact_damping():
    def transition(formula: int, parameters: tuple[float, ...]) -> AtomicTransition:
        return AtomicTransition(
            1, 2, 1.0e8, "E1", 5000.0, 0.1,
            profile_formula=formula,
            profile_parameters=parameters,
        )

    common = dict(
        upper_radiative_rate=9.0e8,
        generic_stark_rate_per_electron=2.0e-8,
        electron_density=1.0e16,
        temperature=1.0e5,
    )
    assert _impact_profile_damping_rate(
        transition(1, (0.1,)), **common
    ) == 0.0
    assert _impact_profile_damping_rate(
        transition(2, (0.1, 3.0e8)), **common
    ) == 3.0e8
    cowley = _impact_profile_damping_rate(
        transition(3, (0.1, 3.0e8, 4.0)), **common
    )
    assert cowley > 3.0e8
    assert _impact_profile_damping_rate(
        transition(0, ()), **common
    ) == 1.1e9


def test_published_ovi_high_series_width_correction_is_selective():
    high_series = AtomicTransition(
        1, 2, 1.0e8, "E1", 5291.28, 1.0,
        profile_formula=4,
        profile_parameters=(1.0, 8.0e7, 83.6, 6.0, 7.0, 8.0),
    )
    low_series = replace(
        high_series,
        wavelength_vacuum_angstrom=3811.35,
        profile_parameters=(1.0, 8.0e7, 5.0, 6.0, 3.0, 3.0),
    )
    common = dict(
        upper_radiative_rate=8.0e7,
        generic_stark_rate_per_electron=0.0,
        electron_density=1.0e17,
        temperature=100_000.0,
        element="O",
        charge=5,
    )
    cowley = _impact_profile_damping_rate(high_series, **common)
    corrected = _impact_profile_damping_rate(
        high_series,
        include_semiclassical_ovi_stark_widths=True,
        **common,
    )
    assert (corrected - 8.0e7) / (cowley - 8.0e7) == pytest.approx(
        3.86716, rel=2.0e-6
    )
    assert _impact_profile_damping_rate(
        low_series,
        include_semiclassical_ovi_stark_widths=True,
        **common,
    ) == _impact_profile_damping_rate(low_series, **common)


def test_tmad_formal_mapping_preserves_nearly_degenerate_rydberg_term(tmp_path):
    """The 8h component must not be mapped onto a degenerate 8i level."""

    ground_threshold = 1.0e16
    target_energy = 10_000.1
    upper_threshold = ground_threshold - target_energy * LIGHT_SPEED
    path = tmp_path / "O_VI_syn"
    path.write_text(
        "L\n"
        f"{'O607G 72G':10s}{'O707S  1S':10s}{ground_threshold:20.12E}     8.00000\n"
        f"{'O608H112HO':10s}{'O707S  1S':10s}{upper_threshold:20.12E}    12.00000\n"
        "RBB\n"
        f"{'O607G 72G':10s}{'O608H112HO':10s} 4 6   1.0000E+00   8.0000E+07  80.0  6  7  8 "
        "WAVELENGTH: 5290.98292 A\n"
        "0\n",
        encoding="ascii",
    )
    ion = AtomicIon(
        element="O",
        charge=5,
        atomic_mass_u=16.0,
        ionization_energy_ev=138.1,
        levels=(
            AtomicLevel(1, 0.0, 8.0, "1s2.7g.(2G<7/2>)"),
            # Put the wrong same-weight term first and at a marginally closer
            # energy to reproduce the historical energy-only permutation.
            AtomicLevel(2, 10_000.0, 12.0, "1s2.8i.(2I<11/2>)"),
            AtomicLevel(3, target_energy, 12.0, "1s2.8h.(2Ho<11/2>)"),
        ),
        transitions=(),
        source="degenerate Rydberg mapping fixture",
    )
    database = AtomicDatabase(MappingProxyType({("O", 5): ion}), source="test")
    formal = atomic_database_with_tmad_formal_ions(
        database, {("O", 5): path}
    )
    transition = formal.ions[("O", 5)].transitions[0]
    assert transition.lower_index == 1
    assert transition.upper_index == 3


def test_diagnostic_li_like_doublets_use_tabulated_stark_widths():
    np.testing.assert_allclose(
        _tabulated_electron_stark_fwhm_angstrom(
            "C", 3, 5802.9, 1.0e17, 100_000.0
        ),
        0.463,
    )
    np.testing.assert_allclose(
        _tabulated_electron_stark_fwhm_angstrom(
            "O", 5, 3812.4, 1.42e17, 79_700.0
        ),
        0.171,
    )
    np.testing.assert_allclose(
        _tabulated_electron_stark_fwhm_angstrom(
            "O", 5, 1031.9, 1.0e17, 100_000.0
        ),
        0.00146,
    )
    # The published high-density entries for this resonance multiplet are
    # linear through 1e20 cm-3 (0.146 A at 1e19 cm-3 and 100,000 K).
    np.testing.assert_allclose(
        _tabulated_electron_stark_fwhm_angstrom(
            "O", 5, 1037.6, 1.0e19, 100_000.0
        ),
        0.146,
    )
    np.testing.assert_allclose(
        _tabulated_electron_stark_fwhm_angstrom(
            "O", 4, 5591.4, 1.0e17, 100_000.0
        ),
        0.189,
    )
    assert _tabulated_electron_stark_fwhm_angstrom(
        "O", 4, 3812.4, 1.0e17, 100_000.0
    ) is None

    line = AtomicTransition(
        1, 2, 1.0e8, "E1", 5802.9, 0.1,
        profile_formula=4,
        profile_parameters=(0.1, 3.0e8, 3.08, 4.0, 3.0, 3.0),
    )
    damping = _impact_profile_damping_rate(
        line,
        upper_radiative_rate=9.0e8,
        generic_stark_rate_per_electron=2.0e-8,
        electron_density=1.0e17,
        temperature=100_000.0,
        element="C",
        charge=3,
    )
    fwhm = (
        (5802.9e-8) ** 2
        * (damping - 3.0e8)
        / (2.0 * np.pi * 2.99792458e10)
        * 1.0e8
    )
    np.testing.assert_allclose(fwhm, 0.463)

    half_damping = _impact_profile_damping_rate(
        line,
        upper_radiative_rate=9.0e8,
        generic_stark_rate_per_electron=2.0e-8,
        electron_density=1.0e17,
        temperature=100_000.0,
        element="C",
        charge=3,
        tabulated_electron_stark_width_scale=0.5,
    )
    np.testing.assert_allclose(
        half_damping - line.profile_parameters[1],
        0.5 * (damping - line.profile_parameters[1]),
    )


def test_term_transition_maps_to_actual_fine_structure_components():
    term_line = AtomicTransition(1, 2, 2.0e8, "E1", 5005.0, 0.3)
    components = (
        AtomicTransition(10, 11, 1.0e8, "E1", 5000.0, 0.1),
        AtomicTransition(10, 12, 2.0e8, "E1", 5010.0, 0.2),
    )
    formal_ion = AtomicIon(
        "C",
        3,
        ATOMIC_MASS_U["C"],
        IONIZATION_ENERGY_EV["C"][3],
        (
            AtomicLevel(10, 0.0, 2.0, "lower"),
            AtomicLevel(11, 20_000.0, 2.0, "upper-a"),
            AtomicLevel(12, 20_100.0, 4.0, "upper-b"),
        ),
        components,
        "fine test",
    )
    database = AtomicDatabase(MappingProxyType({("C", 3): formal_ion}), "fine test")
    mapping = {
        ("C", 3, 1): (("C", 3, 10),),
        ("C", 3, 2): (("C", 3, 11), ("C", 3, 12)),
    }
    assert _mapped_fine_structure_components(
        "C", 3, term_line, database, mapping
    ) == components


def test_light_metal_free_free_uses_nlte_ion_populations():
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=3)
    population = np.zeros((3, atmosphere.n_depth))
    population[1] = 2.0e13
    population[2] = 3.0e13
    state = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({"C": -1.0}),
        element_number_density=MappingProxyType({"C": np.sum(population, axis=0)}),
        ion_number_density=MappingProxyType({"C": population}),
        partition_function=MappingProxyType({}),
        electron_density=atmosphere.electron_density,
        metal_electron_density=np.zeros(atmosphere.n_depth),
    )
    wavelength = np.asarray((1000.0, 5000.0))
    lte = light_metal_free_free_mass_absorption_coefficient(
        atmosphere, wavelength, state
    )
    doubled = light_metal_free_free_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        state,
        {
            ("C", 1): np.full(atmosphere.n_depth, 2.0),
            ("C", 2): np.full(atmosphere.n_depth, 2.0),
        },
    )
    assert np.all(lte > 0.0)
    np.testing.assert_allclose(doubled, 2.0 * lte)
    kernel = light_metal_free_free_charge_kernel(
        atmosphere, wavelength, (1, 2)
    )
    cached = light_metal_free_free_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        state,
        precomputed_charge_kernel=kernel,
    )
    np.testing.assert_allclose(cached, lte)


def test_ground_ion_solver_brackets_lte_supported_stages():
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=3)
    element = "C"
    ions = {
        (element, charge): AtomicIon(
            element,
            charge,
            12.0,
            None if charge == 6 else 20.0 + charge,
            (AtomicLevel(1, 0.0, 1.0, "ground"),),
            (),
        )
        for charge in range(7)
    }
    database = AtomicDatabase(MappingProxyType(ions), source="test")
    fits = {
        (element, charge): VernerPhotoionizationFit(
            element,
            charge,
            10.0 + charge,
            10_000.0,
            20.0,
            1.0,
            1.0,
            2.0,
            0.1,
            0.0,
            1.0,
        )
        for charge in range(6)
    }
    photo = VernerPhotoionizationDatabase(MappingProxyType(fits), source="test")
    fraction = np.asarray((1.0e-30, 1.0e-20, 0.1, 0.8, 0.1, 1.0e-20, 1.0e-30))
    total = np.full(atmosphere.n_depth, 1.0e12)
    lte_population = fraction[:, np.newaxis] * total[np.newaxis, :]
    lte_state = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({element: -1.0}),
        element_number_density=MappingProxyType({element: total}),
        ion_number_density=MappingProxyType({element: lte_population}),
        partition_function=MappingProxyType({
            (element, charge): np.ones(atmosphere.n_depth)
            for charge in range(7)
        }),
        electron_density=atmosphere.electron_density,
        metal_electron_density=np.zeros(atmosphere.n_depth),
    )
    wavelength = np.geomspace(10.0, 1000.0, 300)
    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    state = solve_light_metal_ionization_nlte(
        atmosphere,
        database,
        lte_state,
        photo,
        wavelength,
        planck,
        elements=(element,),
        minimum_active_lte_ion_fraction=1.0e-12,
        active_ion_stage_margin=1,
    )
    assert state.metadata["active_stage_ranges"][element] == (1, 5)
    np.testing.assert_allclose(
        state.ion_number_density[element], lte_population, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        state.ion_departure_coefficient[(element, 0)], 1.0
    )
    np.testing.assert_allclose(
        state.ion_departure_coefficient[(element, 6)], 1.0
    )


def test_tlusty_threshold_reader_extracts_continuum_normalizations(
    tmp_path: Path,
):
    path = tmp_path / "c4.dat"
    path.write_text(
        """****** Levels
 6.00000000E+15  2.  3  'lower one' 0 0. 0.
 5.00000000E+15  6.  3  'lower two' 0 0. 0.
 1.00000000E+15  1.  4  'continuum' 0 0. 0.
****** Continuum transitions
 1 3 1 102 0 0 0 3.000E-19 0.000E+00
 0.0 1.0
 0.0 -3.0
 2 3 1 2 0 0 0 5.000E-19 0.000E+00
 3.0 2.0 1.0 0.0
*** Line transitions
 1 2 -1 0 1 0 0 0.1 0.2
""",
        encoding="ascii",
    )
    data = read_tlusty_photoionization_threshold_data(path)
    np.testing.assert_allclose(data.threshold_frequency_hz, (6.0e15, 5.0e15))
    np.testing.assert_allclose(data.statistical_weight, (2.0, 6.0))
    np.testing.assert_allclose(
        data.threshold_cross_section_cm2, (3.0e-19, 5.0e-19)
    )
    assert data.cross_section_for_threshold(5.98e15) == 3.0e-19
    assert data.cross_section_for_threshold(4.0e15) is None
    np.testing.assert_allclose(
        data.cross_section(np.asarray((5.98e15, 5.98e16)), 5.98e15),
        (1.0e-18, 1.0e-21),
        rtol=0.02,
    )
    # IFANCY=2 is TLUSTY's Peach expression.  The continuum-record value
    # (5e-19) is the effective Seaton collision normalization, while the
    # radiative cross section at threshold is the independent 2e-18 fit.
    np.testing.assert_allclose(
        data.cross_section(np.asarray((4.99e15, 9.98e15)), 4.99e15),
        (2.0e-18, 2.5e-19),
        rtol=2.0e-12,
    )
    expected_wavelength = 2.99792458e18 / 1.0e15
    assert data.collision_gbar_for_wavelength(expected_wavelength) == 0.2


def test_tlusty_tabulated_photoionization_is_zero_below_first_fit_point(
    tmp_path: Path,
):
    path = tmp_path / "delayed-continuum.dat"
    path.write_text(
        """****** Levels
 6.00000000E+15  5.  2  'metastable lower' 0 0. 0.
 1.00000000E+15  1.  3  'continuum' 0 0. 0.
****** Continuum transitions
 1 2 1 102 0 0 0 1.000E-18 0.000E+00
 0.2 0.5
 1.0 0.0
*** Line transitions
""",
        encoding="ascii",
    )
    data = read_tlusty_photoionization_threshold_data(path)
    threshold = 6.0e15
    frequency = threshold * 10.0 ** np.asarray((0.0, 0.199, 0.2, 0.5))
    cross_section = data.cross_section(frequency, threshold)
    np.testing.assert_array_equal(cross_section[:2], 0.0)
    np.testing.assert_allclose(cross_section[2:], (1.0e-17, 1.0e-18))


def test_tmap_seaton_rate_does_not_double_count_tlusty_gbar():
    temperature = np.asarray((80_000.0,))
    electron_density = np.asarray((2.0e16,))
    threshold_u = np.asarray((3.0,))
    raw_cross_section = 1.0e-18
    folded_cross_section = 0.3 * raw_cross_section
    from_raw_tmad = _tmap_seaton_collisional_ionization_rate(
        electron_density,
        temperature,
        threshold_u,
        raw_cross_section,
        5,
    )
    from_tlusty_record = _tmap_seaton_collisional_ionization_rate(
        electron_density,
        temperature,
        threshold_u,
        folded_cross_section,
        5,
        cross_section_includes_gbar=True,
    )
    np.testing.assert_allclose(from_tlusty_record, from_raw_tmad)


def test_tlusty_threshold_reader_rejects_missing_sections(tmp_path: Path):
    path = tmp_path / "broken.dat"
    path.write_text("****** Levels\n1e15 2.\n", encoding="ascii")
    try:
        read_tlusty_photoionization_threshold_data(path)
    except ValueError as error:
        assert "not a supported TLUSTY model atom" in str(error)
    else:
        raise AssertionError("malformed atom should have failed")


def test_tlusty_threshold_match_preserves_superlevel_spin(tmp_path: Path):
    path = tmp_path / "o5.dat"
    path.write_text(
        """****** Levels
 1.00000000E+15 219. 6 'O V +3__ 4' 0 0. 0.
 9.80000000E+14  73. 6 'O V +1__ 3' 0 0. 0.
 1.00000000E+14   1. 7 'continuum' 0 0. 0.
****** Continuum transitions
 1 3 1 1 0 0 0 2.000E-19 0.000E+00
 2 3 1 1 0 0 0 7.000E-19 0.000E+00
*** Line transitions
""",
        encoding="ascii",
    )
    data = read_tlusty_photoionization_threshold_data(path)
    # Energy alone favors the singlet threshold for this deliberately
    # intermediate query.  A triplet TMAD term must still select the triplet
    # superlevel (and conversely for the singlet).
    assert data.cross_section_for_threshold(
        9.89e14, relative_tolerance=0.02, level_label="O506G  3G"
    ) == 2.0e-19
    assert data.cross_section_for_threshold(
        9.89e14, relative_tolerance=0.02, level_label="O506P  1PO"
    ) == 7.0e-19


def test_tlusty_threshold_match_preserves_orbital_term(tmp_path: Path):
    path = tmp_path / "c4.dat"
    path.write_text(
        """****** Levels
 1.00000000E+15   2. 6 'C IV 2Se 2' 0 0. 0.
 9.90000000E+14   6. 6 'C IV 2Po 2' 0 0. 0.
 1.00000000E+14   1. 7 'continuum' 0 0. 0.
****** Continuum transitions
 1 3 1 1 0 0 0 2.000E-19 0.000E+00
 2 3 1 1 0 0 0 7.000E-19 0.000E+00
*** Line transitions
""",
        encoding="ascii",
    )
    data = read_tlusty_photoionization_threshold_data(path)
    assert data.cross_section_for_threshold(
        9.96e14, relative_tolerance=0.02, level_label="C403P  2PO"
    ) == 7.0e-19
    assert data.cross_section_for_threshold(
        9.94e14, relative_tolerance=0.02, level_label="C403S  2S"
    ) == 2.0e-19


def test_tmad_structure_atom_maps_terms_to_formal_levels(tmp_path: Path):
    path = tmp_path / "C_III-IV"
    path.write_text(
        """ATOM
C  2  12.011
L
C302S2 1S C402S  2S   1.000000000000E+16     1.00000
C302P  1POC402S  2S   9.000000000000E+15     3.00000
0
RBB
C302S2 1S C302P  1PO 3 3   5.0000E-01   1.0000E+08       0.2000
0
CBB
C302S2 1S C302P  1PO 1 2   5.0000E-01   0.2000
0
LTE
C303S  1S C402S  2S   8.000000000000E+15     1.00000
C304S' 1POC402S  2S   1.000000000000E+14     3.00000
0
RDI
C302S2 1S C304S' 1PO    1 1   1.50E-03
0
RBB
C302P  1POC303S  1S  3 3   1.0000E-01   5.0000E+07       0.2000
0
CBB
C302P  1POC303S  1S  1 2   1.0000E-01   0.2000
0
CBX
C302P  1POC303S  1S  1 2   1.0000E-01   0.7000
0
L
C402S  2S NONE             0.0 2.0
0
""",
        encoding="ascii",
    )
    ions = {}
    excitation_wavenumber = 1.0e15 / 2.997_924_58e10
    for charge in range(7):
        levels = (AtomicLevel(1, 0.0, 2.0, "ground"),)
        if charge == 2:
            levels = (
                AtomicLevel(1, 0.0, 1.0, "ground"),
                AtomicLevel(2, excitation_wavenumber, 3.0, "excited"),
                AtomicLevel(3, 2.0 * excitation_wavenumber, 1.0, "higher"),
            )
        ions[("C", charge)] = AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=levels,
            transitions=(),
            source="synthetic formal atom",
        )
    formal = AtomicDatabase(MappingProxyType(ions), source="synthetic formal atom")
    model = read_tmad_structure_model_atom(path, formal)
    assert dict(model.levels_per_charge) == {2: 2, 3: 1}
    assert len(model.atomic_database.ions[("C", 2)].transitions) == 1
    assert model.formal_level_mapping[("C", 2, 2)] == (("C", 2, 2),)
    structure_transition = model.atomic_database.ions[("C", 2)].transitions[0]
    expected_structure_a = (
        8.0
        * np.pi**2
        * 4.803_204_712_570_263e-10**2
        / (9.109_383_713_9e-28 * 2.997_924_58e10)
        * (1.0 / 3.0)
        * 0.5
        / (structure_transition.wavelength_vacuum_angstrom * 1.0e-8) ** 2
    )
    assert np.isclose(structure_transition.einstein_a, expected_structure_a)
    assert structure_transition.einstein_a != 1.0e8
    assert structure_transition.profile_parameters == (0.5, 1.0e8, 0.2)
    assert model.continuum_parent_mapping[("C", 2, 2)] == ("C", 3, 1)
    assert sum(len(levels) for levels in model.lte_level_reservoir.values()) == 1
    assert len(model.lte_bound_bound_couplings) == 1
    assert model.autoionizing_lte_level_labels == ("C304S' 1PO",)
    assert len(model.radiative_dielectronic_records) == 1
    dielectronic = model.radiative_dielectronic_records[0]
    assert dielectronic.lower_level_label == "C302S2 1S"
    assert dielectronic.autoionizing_level_label == "C304S' 1PO"
    assert dielectronic.formula == 1
    assert np.isclose(dielectronic.oscillator_strength, 1.5e-3)
    assert len(model.effective_dielectronic_couplings) == 1
    effective = model.effective_dielectronic_couplings[0]
    assert effective.lower_level_key == ("C", 2, 1)
    assert effective.continuum_parent_key == ("C", 3, 1)
    assert effective.transition_frequency_hz == pytest.approx(9.9e15)
    coupling = model.lte_bound_bound_couplings[0]
    assert coupling.explicit_level_key == ("C", 2, 2)
    assert coupling.lte_parent_key == ("C", 3, 1)
    assert coupling.explicit_is_lower
    assert coupling.collision_record == (1, (0.1, 0.7))
    assert model.collision_data[("C", 2, 1, 2)] == (1, (0.5, 0.2))
    no_cbx_path = tmp_path / "C_III-IV-no-cbx"
    no_cbx_path.write_text(
        path.read_text(encoding="ascii").replace(
            "CBX\nC302P  1POC303S  1S  1 2   1.0000E-01   0.7000\n0\n",
            "CBX\n0\n",
        ),
        encoding="ascii",
    )
    no_cbx = read_tmad_structure_model_atom(no_cbx_path, formal)
    assert no_cbx.lte_bound_bound_couplings[0].collision_record == (
        1, (0.1, 0.2)
    )
    processed_no_cbx = read_tmad_structure_model_atom(
        no_cbx_path,
        formal,
        infer_lte_collisions_from_cbb=False,
    )
    assert processed_no_cbx.lte_bound_bound_couplings[0].collision_record is None
    promoted = read_tmad_structure_model_atom(
        path, formal, promote_lte_levels_per_charge={2: 1}
    )
    assert promoted.levels_per_charge[2] == 3
    assert len(promoted.atomic_database.ions[("C", 2)].transitions) == 2
    assert promoted.formal_level_mapping[("C", 2, 3)] == (("C", 2, 3),)
    assert promoted.lte_level_reservoir == {}
    assert promoted.lte_bound_bound_couplings == ()

    label_promoted = read_tmad_structure_model_atom(
        path, formal, promote_lte_level_labels=("C303S  1S",)
    )
    assert label_promoted.levels_per_charge[2] == 3
    assert len(label_promoted.atomic_database.ions[("C", 2)].transitions) == 2
    assert label_promoted.formal_level_mapping[("C", 2, 3)] == (("C", 2, 3),)
    assert label_promoted.lte_level_reservoir == {}

    selected_after_demote = read_tmad_structure_model_atom(
        path,
        formal,
        target_nlte_levels_per_charge={2: 1, 3: 1},
        promote_lte_level_labels=("C303S  1S",),
    )
    assert selected_after_demote.levels_per_charge[2] == 2
    assert selected_after_demote.atomic_database.ions[("C", 2)].levels[1].label == (
        "C303S  1S"
    )

    try:
        read_tmad_structure_model_atom(
            path, formal, promote_lte_level_labels=("C399X  9X",)
        )
    except ValueError as error:
        assert "no LTE level" in str(error)
    else:
        raise AssertionError("unknown targeted LTE term should have failed")

    try:
        read_tmad_structure_model_atom(
            path, formal, promote_lte_level_labels=("C304S' 1PO",)
        )
    except ValueError as error:
        assert "autoionizing state" in str(error)
    else:
        raise AssertionError("RDI autoionizing term should not be promoted")

    historical_control = read_tmad_structure_model_atom(
        path,
        formal,
        target_nlte_levels_per_charge={2: 4, 3: 1},
        diagnostic_allow_rdi_level_promotion=True,
    )
    assert historical_control.levels_per_charge[2] == 4

    targeted = read_tmad_structure_model_atom(
        path, formal, target_nlte_levels_per_charge={2: 3, 3: 1}
    )
    assert dict(targeted.levels_per_charge) == {2: 3, 3: 1}
    assert len(targeted.atomic_database.ions[("C", 2)].transitions) == 2
    assert targeted.lte_level_reservoir == {}

    demoted = read_tmad_structure_model_atom(
        path, formal, target_nlte_levels_per_charge={2: 1, 3: 1}
    )
    assert dict(demoted.levels_per_charge) == {2: 1, 3: 1}
    assert demoted.atomic_database.ions[("C", 2)].transitions == ()
    assert len(demoted.lte_bound_bound_couplings) == 1
    assert sum(
        len(levels) for levels in demoted.lte_level_reservoir.values()
    ) == 2

    formal_path = tmp_path / "C_III_syn"
    formal_path.write_text(
        """L
C302S2 1S C402S  2S   1.000000000000E+16     1.00000
C302P  1POC402S  2S   9.000000000000E+15     3.00000
C303D  1D C402S  2S   5.000000000000E+15     5.00000
RBB
C302S2 1S C302P  1PO 4 6   5.0000E-01   1.0000E+08  3.0  3 2 2 WAVELENGTH:  3000.00000 A
C302P  1POC303D 21D  3 3   1.0000E-01   5.0000E+07  0.2 WAVELENGTH:  4500.00000 A
0
""",
        encoding="ascii",
    )
    formal_lines = atomic_database_with_tmad_formal_ions(
        formal, {("C", 2): formal_path}
    )
    assert len(formal_lines.ions[("C", 2)].levels) == 4
    assert len(formal_lines.ions[("C", 2)].transitions) == 2
    transition = formal_lines.ions[("C", 2)].transitions[0]
    assert transition.wavelength_vacuum_angstrom == 3000.0
    assert transition.profile_formula == 4
    assert transition.profile_parameters == (
        0.5, 1.0e8, 3.0, 3.0, 2.0, 2.0
    )
    expected_formal_a = (
        8.0
        * np.pi**2
        * 4.803_204_712_570_263e-10**2
        / (9.109_383_713_9e-28 * 2.997_924_58e10)
        * (1.0 / 3.0)
        * 0.5
        / (3000.0e-8) ** 2
    )
    assert np.isclose(transition.einstein_a, expected_formal_a)
    assert transition.einstein_a != 1.0e8
    base_ions = dict(formal.ions)
    base_transition = AtomicTransition(
        1,
        2,
        3.1e7,
        "E1",
        3000.03,
        0.48,
    )
    base_ions[("C", 2)] = replace(
        base_ions[("C", 2)], transitions=(base_transition,)
    )
    profiled = atomic_database_with_tmad_profile_parameters(
        AtomicDatabase(MappingProxyType(base_ions), source="profile base"),
        {("C", 2): formal_path},
    )
    retained = profiled.ions[("C", 2)].transitions[0]
    assert retained.wavelength_vacuum_angstrom == 3000.03
    assert retained.absorption_oscillator_strength == 0.48
    assert retained.einstein_a == 3.1e7
    assert retained.profile_formula == 4
    assert retained.profile_parameters == (
        0.48, 1.0e8, 3.0, 3.0, 2.0, 2.0
    )
    selected_profiled = atomic_database_with_tmad_profile_parameters(
        AtomicDatabase(MappingProxyType(base_ions), source="profile base"),
        {("C", 2): formal_path},
        series_lower_principal_quantum_number=2,
        minimum_upper_principal_quantum_number=2,
    )
    assert selected_profiled.ions[("C", 2)].transitions[0].profile_formula == 4
    interval_profiled = atomic_database_with_tmad_profile_parameters(
        AtomicDatabase(MappingProxyType(base_ions), source="profile base"),
        {("C", 2): formal_path},
        wavelength_intervals_angstrom=((2999.0, 3001.0),),
    )
    assert interval_profiled.ions[("C", 2)].transitions[0].profile_formula == 4
    with pytest.raises(ValueError, match="no secure profile matches"):
        atomic_database_with_tmad_profile_parameters(
            AtomicDatabase(MappingProxyType(base_ions), source="profile base"),
            {("C", 2): formal_path},
            wavelength_intervals_angstrom=((1174.0, 1177.0),),
        )
    with pytest.raises(ValueError, match="no secure profile matches"):
        atomic_database_with_tmad_profile_parameters(
            AtomicDatabase(MappingProxyType(base_ions), source="profile base"),
            {("C", 2): formal_path},
            series_lower_principal_quantum_number=3,
            minimum_upper_principal_quantum_number=8,
        )
    projected_collision = fine_structure_collision_data_from_tmad(
        model, formal_lines
    )
    assert projected_collision[("C", 2, 1, 2)] == (1, (0.5, 0.2))


def test_tmad_structure_retains_rbb_with_only_oscillator_strength(
    tmp_path: Path,
):
    path = tmp_path / "C_minimal_rbb"
    path.write_text(
        """ATOM
C  2  12.011
L
C302S2 1S C402S  2S   1.000000000000E+16     1.00000
C302P  1POC402S  2S   9.000000000000E+15     3.00000
0
RBB
C302S2 1S C302P  1PO 1 1   5.0000E-01
0
CBB
0
L
C402S  2S NONE             0.0 2.0
0
""",
        encoding="ascii",
    )
    ions = {}
    excitation_wavenumber = 1.0e15 / LIGHT_SPEED
    for charge in range(7):
        levels = (AtomicLevel(1, 0.0, 2.0, "ground"),)
        if charge == 2:
            levels = (
                AtomicLevel(1, 0.0, 1.0, "ground"),
                AtomicLevel(2, excitation_wavenumber, 3.0, "excited"),
            )
        ions[("C", charge)] = AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=levels,
            transitions=(),
            source="synthetic formal atom",
        )
    formal = AtomicDatabase(MappingProxyType(ions), source="synthetic formal atom")
    model = read_tmad_structure_model_atom(path, formal)
    transitions = model.atomic_database.ions[("C", 2)].transitions
    assert len(transitions) == 1
    assert transitions[0].absorption_oscillator_strength == 0.5
    assert transitions[0].einstein_a > 0.0
    assert transitions[0].profile_parameters == (0.5,)


def test_tmad_mapping_uses_labels_for_nearly_degenerate_rydberg_terms(
    tmp_path: Path,
):
    path = tmp_path / "C_IV-V"
    ground_threshold = 1.0e16
    f_threshold = ground_threshold - 100_000.0 * 2.997_924_58e10
    g_threshold = ground_threshold - 100_000.2 * 2.997_924_58e10
    path.write_text(
        f"""ATOM
C  2  12.011
L
C402S  2S C502S  2S {ground_threshold:20.12E}     2.00000
C406F  2FOC502S  2S {f_threshold:20.12E}    14.00000
C406G  2G C502S  2S {g_threshold:20.12E}    18.00000
0
L
C502S  2S NONE             0.0 2.0
0
""",
        encoding="ascii",
    )
    ions = {}
    for charge in range(7):
        levels = (AtomicLevel(1, 0.0, 2.0, "ground"),)
        if charge == 3:
            # The 6g J=7/2 component is deliberately closer in energy to
            # the 6f centroid than the correct 6f J=7/2 component.  An
            # energy-only subset match therefore scrambles the two terms.
            levels = (
                AtomicLevel(1, 0.0, 2.0, "1s2.2s.(2S<1/2>)"),
                AtomicLevel(2, 100_000.0, 6.0, "1s2.6f.(2Fo<5/2>)"),
                AtomicLevel(3, 100_000.1, 8.0, "1s2.6g.(2Go<7/2>)"),
                AtomicLevel(4, 100_001.0, 8.0, "1s2.6f.(2Fo<7/2>)"),
                AtomicLevel(5, 100_001.1, 10.0, "1s2.6g.(2Go<9/2>)"),
            )
        ions[("C", charge)] = AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=levels,
            transitions=(),
            source="synthetic formal atom",
        )
    formal = AtomicDatabase(MappingProxyType(ions), source="synthetic formal atom")

    model = read_tmad_structure_model_atom(path, formal)

    assert set(model.formal_level_mapping[("C", 3, 2)]) == {
        ("C", 3, 2),
        ("C", 3, 4),
    }
    assert set(model.formal_level_mapping[("C", 3, 3)]) == {
        ("C", 3, 3),
        ("C", 3, 5),
    }


def test_promoted_tmad_superlevel_without_formal_components_is_retained(
    tmp_path: Path,
):
    path = tmp_path / "C_III-IV"
    path.write_text(
        """ATOM
C  2  12.011
L
C302S2 1S C402S  2S   1.000000000000E+16     1.00000
0
LTE
C303X  7X C402S  2S   8.000000000000E+15     7.00000
0
L
C402S  2S NONE             0.0 2.0
0
""",
        encoding="ascii",
    )
    ions = {
        ("C", charge): AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=(AtomicLevel(1, 0.0, 1.0 if charge == 2 else 2.0, "ground"),),
            transitions=(),
            source="synthetic formal atom",
        )
        for charge in range(7)
    }
    formal = AtomicDatabase(MappingProxyType(ions), source="synthetic formal atom")
    promoted = read_tmad_structure_model_atom(
        path, formal, promote_lte_levels_per_charge={2: 1}
    )
    assert promoted.levels_per_charge[2] == 2
    assert ("C", 2, 2) not in promoted.formal_level_mapping


def test_explicit_tmad_superlevel_can_be_retained_without_formal_components(
    tmp_path: Path,
):
    path = tmp_path / "C_III-IV"
    path.write_text(
        """ATOM
C  2  12.011
L
C302S2 1S C402S  2S   1.000000000000E+16     1.00000
C303X  7X C402S  2S   8.000000000000E+15     7.00000
0
L
C402S  2S NONE             0.0 2.0
0
""",
        encoding="ascii",
    )
    ions = {
        ("C", charge): AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=(
                AtomicLevel(1, 0.0, 1.0 if charge == 2 else 2.0, "ground"),
            ),
            transitions=(),
            source="synthetic formal atom",
        )
        for charge in range(7)
    }
    formal = AtomicDatabase(MappingProxyType(ions), source="synthetic formal atom")
    with pytest.raises(ValueError, match="cannot map TMAD term"):
        read_tmad_structure_model_atom(path, formal)
    enlarged = read_tmad_structure_model_atom(
        path, formal, allow_unmapped_nlte_terms=True
    )
    assert enlarged.levels_per_charge[2] == 2
    assert ("C", 2, 2) not in enlarged.formal_level_mapping


def test_reduced_atom_grid_resolves_opacity_project_fit_knots(tmp_path: Path):
    path = tmp_path / "c3.dat"
    path.write_text(
        """****** Levels
 6.00000000E+15  2.  3  'ground' 0 0. 0.
 8.58128911E+15  4.  3  'excited' 0 0. 0.
 1.00000000E+15  1.  4  'continuum' 0 0. 0.
****** Continuum transitions
 2 3 1 103 0 0 0 5.000E-19 0.000E+00
 0.0 0.2 0.6
 0.0 1.0 -1.0
*** Line transitions
 1 2 -1 0 1 0 0 0.1 0.2
""",
        encoding="ascii",
    )
    threshold_data = read_tlusty_photoionization_threshold_data(path)
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=3)
    ions = {
        ("C", charge): AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=IONIZATION_ENERGY_EV["C"][charge],
            levels=(AtomicLevel(1, 0.0, 1.0, "ground"),),
            transitions=(),
            source="synthetic test atom",
        )
        for charge in (0, 1)
    }
    ions.update({
        ("C", 2): AtomicIon(
            element="C",
            charge=2,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=IONIZATION_ENERGY_EV["C"][2],
            levels=(
                AtomicLevel(1, 0.0, 2.0, "ground"),
                AtomicLevel(2, 100_000.0, 4.0, "excited"),
            ),
            transitions=(),
            source="synthetic test atom",
        ),
        ("C", 3): AtomicIon(
            element="C",
            charge=3,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=IONIZATION_ENERGY_EV["C"][3],
            levels=(AtomicLevel(1, 0.0, 2.0, "ground"),),
            transitions=(),
            source="synthetic test atom",
        ),
    })
    database = AtomicDatabase(MappingProxyType(ions), source="synthetic test atom")
    photoionization = VernerPhotoionizationDatabase(
        MappingProxyType({
            ("C", 2): VernerPhotoionizationFit(
                "C", 2, IONIZATION_ENERGY_EV["C"][2], 1.0e4,
                50.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0,
            )
        }),
        source="synthetic test fit",
    )
    wavelength = reduced_light_metal_wavelength(
        database,
        "C",
        {2: 2, 3: 1},
        n_continuum_wavelength=120,
        photoionization_threshold_data={2: threshold_data},
    )
    threshold_ev = (
        IONIZATION_ENERGY_EV["C"][2]
        - 100_000.0 * 1.986_445_857_148_928_6e-16 / 1.602_176_634e-12
    )
    threshold_wavelength = (
        6.626_070_15e-27 * 2.997_924_58e10
        / (threshold_ev * 1.602_176_634e-12) * 1.0e8
    )
    for log_ratio in (0.0, 0.1, 0.2, 0.4, 0.6):
        expected = threshold_wavelength / 10.0**log_ratio
        assert np.min(np.abs(wavelength / expected - 1.0)) < 1.0e-12


@pytest.mark.parametrize("line_wavelength", [1000., 1001.])
def test_reduced_atom_planck_field_recovers_full_partition_lte_populations(line_wavelength):
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=6)
    ions = {}
    for charge in range(7):
        levels = (AtomicLevel(1, 0.0, 2.0, "ground"),)
        transitions = ()
        if charge == 2:
            # The third level is deliberately omitted from the reduced atom.
            # Its Boltzmann weight must remain in the full partition function.
            levels = (
                AtomicLevel(1, 0.0, 2.0, "ground"),
                AtomicLevel(2, 100_000.0, 4.0, "explicit excited"),
                AtomicLevel(3, 150_000.0, 6.0, "omitted excited"),
            )
            transitions = (
                AtomicTransition(1, 2, 1.0e8, "E1", line_wavelength, 0.1),
                AtomicTransition(1, 2, 1.0e3, "M1", 1000.0, 0.0),
            )
        ions[("C", charge)] = AtomicIon(
            element="C",
            charge=charge,
            atomic_mass_u=ATOMIC_MASS_U["C"],
            ionization_energy_ev=(
                IONIZATION_ENERGY_EV["C"][charge] if charge < 6 else None
            ),
            levels=levels,
            transitions=transitions,
            source="synthetic test atom",
        )
    database = AtomicDatabase(MappingProxyType(ions), source="synthetic test atom")
    n_depth = atmosphere.n_depth
    ion_population = np.ones((7, n_depth))
    ion_population[2] = 1.0e12
    ion_population[3] = 5.0e11
    element_population = np.sum(ion_population, axis=0)
    partitions = {
        ("C", charge): ion.partition_function(atmosphere.temperature)
        for (element, charge), ion in ions.items()
    }
    lte_state = MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType({"C": -1.0}),
        element_number_density=MappingProxyType({"C": element_population}),
        ion_number_density=MappingProxyType({"C": ion_population}),
        partition_function=MappingProxyType(partitions),
        electron_density=atmosphere.electron_density,
        metal_electron_density=np.zeros(n_depth),
    )
    photoionization = VernerPhotoionizationDatabase(
        MappingProxyType({
            ("C", 2): VernerPhotoionizationFit(
                "C", 2, IONIZATION_ENERGY_EV["C"][2], 1.0e4,
                50.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0,
            )
        }),
        source="synthetic test fit",
    )
    wavelength = reduced_light_metal_wavelength(
        database,
        "C",
        {2: 2, 3: 1},
        n_continuum_wavelength=120,
    )
    mean_intensity = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        wavelength,
        mean_intensity,
        "C",
        {2: 2, 3: 1},
    )
    np.testing.assert_allclose(
        state.population_density,
        state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    rdi_coupling = TmadEffectiveDielectronicCoupling(
        lower_level_key=("C", 2, 2),
        continuum_parent_key=("C", 3, 1),
        transition_frequency_hz=2.5e15,
        oscillator_strength=0.15,
        lower_level_label="synthetic explicit excited",
        autoionizing_level_label="synthetic auto state",
    )
    rdi_wavelength = reduced_light_metal_wavelength(
        database,
        "C",
        {2: 2, 3: 1},
        n_continuum_wavelength=120,
        effective_dielectronic_couplings=(rdi_coupling,),
    )
    rdi_intensity = planck_lambda_angstrom(
        rdi_wavelength[:, np.newaxis],
        atmosphere.temperature[np.newaxis, :],
    )
    rdi_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        rdi_wavelength,
        rdi_intensity,
        "C",
        {2: 2, 3: 1},
        effective_dielectronic_couplings=(rdi_coupling,),
    )
    np.testing.assert_allclose(
        rdi_state.population_density,
        rdi_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert rdi_state.metadata["effective_dielectronic_couplings"] == 1
    collision_ions = dict(ions)
    collision_ions[("C", 2)] = AtomicIon(
        element="C",
        charge=2,
        atomic_mass_u=ATOMIC_MASS_U["C"],
        ionization_energy_ev=IONIZATION_ENERGY_EV["C"][2],
        levels=ions[("C", 2)].levels,
        transitions=(),
        source="synthetic collision-only atom",
    )
    collision_only_database = AtomicDatabase(
        MappingProxyType(collision_ions), source="synthetic collision-only atom"
    )
    collision_only_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        collision_only_database,
        lte_state,
        photoionization,
        wavelength,
        mean_intensity,
        "C",
        {2: 2, 3: 1},
        collision_data=MappingProxyType({
            ("C", 2, 1, 2): ChiantiTermCollisionStrength(
                components=(ChiantiScaledCollisionComponent(
                    transition_energy_rydberg=1.0,
                    transition_type=2,
                    scaling_parameter=1.0,
                    scaled_temperature=(0.0, 1.0),
                    scaled_upsilon=(1.0, 1.0),
                ),),
                source="synthetic R-matrix collision",
            )
        }),
    )
    np.testing.assert_allclose(
        collision_only_state.population_density,
        collision_only_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert collision_only_state.metadata["collision_only_transitions"] == 1
    tmad_collision_only_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        collision_only_database,
        lte_state,
        photoionization,
        wavelength,
        mean_intensity,
        "C",
        {2: 2, 3: 1},
        collision_data=MappingProxyType({
            # TMAD formula 1: absorption oscillator strength, minimum g-bar.
            ("C", 2, 1, 2): (1, (0.12, 0.7)),
        }),
    )
    np.testing.assert_allclose(
        tmad_collision_only_state.population_density,
        tmad_collision_only_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert tmad_collision_only_state.metadata[
        "collision_only_transitions"
    ] == 1
    mali_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        wavelength,
        mean_intensity,
        "C",
        {2: 2, 3: 1},
        approximate_lambda_diagonal=np.full_like(mean_intensity, 0.7),
        previous_population_state=state,
    )
    np.testing.assert_allclose(
        mali_state.population_density,
        mali_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert "MALI" in mali_state.metadata["statistical_equilibrium_iteration"]
    assert np.all(
        np.sum(state.population_density, axis=0)
        < ion_population[2] + ion_population[3]
    )
    reservoir_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        wavelength,
        mean_intensity,
        "C",
        {2: 2, 3: 1},
        lte_level_reservoir={
            ("C", 3, 1): ((2, 150_000.0, 6.0),),
        },
    )
    np.testing.assert_allclose(
        reservoir_state.population_density,
        reservoir_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert reservoir_state.metadata["lte_reservoir_levels"] == 1
    closure_wavelength = np.unique(np.concatenate((
        wavelength,
        np.linspace(1999.0, 2001.0, 201),
    )))
    closure_mean_intensity = planck_lambda_angstrom(
        closure_wavelength[:, np.newaxis],
        atmosphere.temperature[np.newaxis, :],
    )
    closure_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        closure_wavelength,
        closure_mean_intensity,
        "C",
        {2: 2, 3: 1},
        lte_level_reservoir={
            ("C", 3, 1): ((2, 150_000.0, 6.0),),
        },
        lte_bound_bound_couplings=(TmadLTEBoundBoundCoupling(
            explicit_level_key=("C", 2, 2),
            lte_parent_key=("C", 3, 1),
            explicit_is_lower=True,
            lte_charge=2,
            lte_energy_wavenumber=150_000.0,
            lte_statistical_weight=6.0,
            lte_level_label="omitted excited",
            transition=AtomicTransition(
                2, 0, 5.0e7, "E1", 2000.0, 0.1
            ),
            collision_record=(1, (0.1, 0.2)),
        ),),
    )
    np.testing.assert_allclose(
        closure_state.population_density,
        closure_state.lte_population_density,
        rtol=5.0e-7,
        atol=1.0e-3,
    )
    assert closure_state.metadata["lte_bound_bound_couplings"] == 1
    non_lte_reservoir_state = solve_reduced_light_metal_levels_nlte(
        atmosphere,
        database,
        lte_state,
        photoionization,
        wavelength,
        0.35 * mean_intensity,
        "C",
        {2: 2, 3: 1},
        lte_level_reservoir={
            ("C", 3, 1): ((2, 150_000.0, 6.0),),
        },
    )
    parent_index = non_lte_reservoir_state.level_key.index(("C", 3, 1))
    reservoir_lte = (
        ion_population[2]
        * 6.0
        * np.exp(
            -150_000.0
            * 6.626_070_15e-27
            * 2.997_924_58e10
            / (1.380_649e-16 * atmosphere.temperature)
        )
        / partitions[("C", 2)]
    )
    conservation_weight = np.ones_like(
        non_lte_reservoir_state.population_density
    )
    conservation_weight[parent_index] += reservoir_lte / np.maximum(
        non_lte_reservoir_state.lte_population_density[parent_index],
        np.finfo(float).tiny,
    )
    np.testing.assert_allclose(
        np.sum(
            conservation_weight
            * non_lte_reservoir_state.population_density,
            axis=0,
        ),
        np.sum(non_lte_reservoir_state.lte_population_density, axis=0)
        + reservoir_lte,
        rtol=2.0e-12,
    )


def test_rate_equilibrium_keeps_weak_links_and_reservoir_conservation():
    from wd_spectra.light_metal_nlte import _positive_rate_equilibrium
    weights=np.geomspace(1e-120,1.,18)
    rng=np.random.default_rng(42)
    flux=10.**rng.uniform(-50,30,(len(weights),len(weights)))
    flux=flux+flux.T
    np.fill_diagonal(flux,0.)
    rates=flux/weights[None,:]
    np.fill_diagonal(rates,-rates.sum(axis=0))
    reservoir=np.geomspace(1.,100.,len(weights))
    result=_positive_rate_equilibrium(rates,reservoir,2.3e15)
    expected=weights*(2.3e15/np.dot(reservoir,weights))
    np.testing.assert_allclose(result,expected,rtol=2e-13,atol=0.)
    assert np.dot(reservoir,result)==pytest.approx(2.3e15)


@pytest.mark.parametrize("quadrature_scale", [1., .5, 2., 1.])
def test_compiled_holtsmark_interpolation_preserves_central_and_wing_profiles(quadrature_scale):
    if _rt is None or not hasattr(_rt, 'accumulate_metal_line_profiles'):
        pytest.skip('compiled backend unavailable')
    from wd_spectra.light_metal_nlte import _HOLTSMARK_COMPILED_ARGUMENT, _HOLTSMARK_COMPILED_WEIGHT
    beta=np.unique(np.r_[np.geomspace(1e-5,30.,30000),.001,8.])
    center=np.array([1031.9]);scale=np.array([[1e12]])
    wave=np.sort(2.99792458e18/(2.99792458e18/center[0]+beta*scale[0,0]))
    planck=np.ones((len(wave),1));one=np.ones((1,1));zero=np.zeros((1,1))
    absorption=np.zeros_like(planck);emission=np.zeros_like(planck)
    expected_a=np.zeros_like(planck);expected_e=np.zeros_like(planck)
    args=(wave,planck,center,np.array([0.]),one*.1,one*.01,np.array([30.]),scale,one)
    _accumulate_metal_line_profiles_python(*args,zero,one,one,one,one*.2,expected_a,expected_e,True)
    _rt.accumulate_metal_line_profiles(*args,one,one,one,one*.2,
        _HOLTSMARK_COMPILED_ARGUMENT,_HOLTSMARK_COMPILED_WEIGHT*quadrature_scale,absorption,emission,True)
    actual_beta=np.abs(2.99792458e10/(wave*1e-8)-2.99792458e10/(center[0]*1e-8))/scale[0,0]
    central=(actual_beta>=.001)&(actual_beta<=8.)
    expected_a[central]*=quadrature_scale;expected_e[central]*=quadrature_scale
    np.testing.assert_allclose(absorption,expected_a,rtol=3e-13,atol=0.)
    np.testing.assert_allclose(emission,expected_e,rtol=3e-13,atol=0.)


@pytest.mark.parametrize("photon,inverse,old_source,diagonal", [
    (.01,.02,10.,.8), (10.,.01,10.,.8), (2.,3.,2.,.7), (.1,1.1,1e8,1.)])
def test_mali_preserves_positive_rates_and_unpreconditioned_particle_balance(photon,inverse,old_source,diagonal):
    from wd_spectra.light_metal_nlte import _mali_radiative_occupations
    upward,downward=_mali_radiative_occupations(photon,inverse,diagonal,old_source)
    assert upward>=0 and downward>=0
    lower=1.;weight_ratio=2.
    upper=weight_ratio*lower*old_source/(1.+old_source)
    original=weight_ratio*lower*photon-upper*inverse
    preconditioned=weight_ratio*lower*upward-upper*downward
    np.testing.assert_allclose(preconditioned,original,rtol=3e-14,atol=3e-15)


def test_compiled_bound_free_accumulation_matches_numpy(monkeypatch):
    import wd_spectra.light_metal_nlte as metal
    if metal._rt is None or not hasattr(metal._rt, 'accumulate_bound_free_nlte'):
        pytest.skip('compiled bound-free kernel unavailable')
    rng = np.random.default_rng(117)
    wave, depth = 113, 17
    exponential = rng.uniform(0., 1., (wave, depth))
    exponential[0] = 0.
    exponential[1] = 1.
    planck = np.exp(rng.uniform(-100., 30., (wave, depth)))
    absorption = rng.random((wave, depth)); emission = absorption.copy()
    reference_absorption = absorption.copy(); reference_emission = emission.copy()
    levels = [(np.exp(rng.uniform(-50., -30., wave)),
               np.exp(rng.uniform(5., 45., depth)),
               np.exp(rng.uniform(-20., 0., depth)),
               np.exp(rng.uniform(-8., 8., depth)),
               np.exp(rng.uniform(-8., 8., depth))) for _ in range(4)]
    for values in levels:
        metal._accumulate_bound_free_nlte(*values, exponential, planck, absorption, emission)
    monkeypatch.setattr(metal, '_rt', None)
    for values in levels:
        metal._accumulate_bound_free_nlte(*values, exponential, planck, reference_absorption, reference_emission)
    # The compiled multiply/subtract may fuse near cancellation; allow the
    # resulting sub-picorelative rounding difference.
    np.testing.assert_allclose(absorption, reference_absorption, rtol=3e-13, atol=0.)
    np.testing.assert_allclose(emission, reference_emission, rtol=3e-13, atol=0.)


def test_compiled_bound_free_rejects_mismatched_output():
    import wd_spectra.light_metal_nlte as metal
    if metal._rt is None or not hasattr(metal._rt, 'accumulate_bound_free_nlte'):
        pytest.skip('compiled bound-free kernel unavailable')
    with pytest.raises(ValueError, match='wavelength/depth arrays must match'):
        metal._rt.accumulate_bound_free_nlte(np.ones(3), *[np.ones(2) for _ in range(4)],
            np.ones((3, 2)), np.ones((3, 2)), np.zeros((3, 1)), np.zeros((3, 2)))
