from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra.d6 import (
    D6_STOUT_MAXIMUM_CHARGE,
    D6_TLUSTY_TOPBASE_FILES,
    D6_TLUSTY_RAP_FILES,
    SDSS_J1637_LOG_NUMBER_ABUNDANCE,
    TOPbaseLevelPhotoionization,
    TOPbasePhotoionizationDatabase,
    atmosphere_with_bulk_metal_state,
    bulk_metal_lte_state,
    bulk_metal_thermodynamics,
    d6_tlusty_excitation_energy_overrides,
    gray_d6_atmosphere,
    _structure_wavelength_grid,
    merge_topbase_photoionization_databases,
    read_norad_ls_photoionization,
    read_sirocco_topbase_lte_photoionization,
    resonance_average_topbase_photoionization,
    stout_ls_term_excitation_overrides,
    read_tlusty_topbase_lte_photoionization,
    read_tlusty_rap_lte_photoionization,
    retain_d6_structure_line_stages,
    topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
)
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.models.common import ModelData
from wd_spectra import (
    gray_helium_atmosphere,
    radiative_equilibrium_helium_atmosphere,
    synthesize_helium_spectrum,
)
from wd_spectra.convection import (
    ml2_convective_flux_gradient_derivative_from_thermodynamics,
    ml2_convective_flux_for_gradient_from_thermodynamics,
    ml2_temperature_gradient_for_total_flux_from_thermodynamics,
)
from wd_spectra.metals import (
    ATOMIC_MASS_U,
    IONIZATION_ENERGY_EV,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    MetalLTEState,
    cool_metal_negative_ion_mass_absorption_coefficient,
    hydrogenic_metal_level_mean_radius_cm,
    metal_ionic_microfield_perturber_density,
    metal_neutral_hard_sphere_occupation_probability,
    metal_rydberg_level_occupation_probability,
    metal_rydberg_transition_survival_probability,
    o_i_3p5p_nd5d_electron_stark_rate_coefficient,
    oxygen_negative_ion_lte_number_density,
    read_stout_atomic_database,
)


def _ground_state_database(elements: tuple[str, ...]) -> AtomicDatabase:
    ions = {}
    for element in elements:
        for charge in range(len(IONIZATION_ENERGY_EV[element]) + 1):
            ions[(element, charge)] = AtomicIon(
                element,
                charge,
                ATOMIC_MASS_U[element],
                (
                    IONIZATION_ENERGY_EV[element][charge]
                    if charge < len(IONIZATION_ENERGY_EV[element])
                    else None
                ),
                (AtomicLevel(1, 0.0, 1.0, "ground"),),
                (),
            )
    return AtomicDatabase(MappingProxyType(ions))


def test_helium_formal_spectrum_accepts_level_resolved_topbase_opacity():
    """Exercise the public spectrum path, not only the D6 opacity helper."""

    database = _ground_state_database(("O",))
    threshold = IONIZATION_ENERGY_EV["O"][0]
    topbase = TOPbasePhotoionizationDatabase(
        (
            TOPbaseLevelPhotoionization(
                "O",
                0,
                1,
                0.0,
                1.0,
                threshold,
                threshold,
                np.asarray((threshold, 20.0, 30.0)),
                np.asarray((2.0e-18, 1.0e-18, 4.0e-19)),
            ),
        ),
        "synthetic TOPbase regression table",
    )
    atmosphere = gray_helium_atmosphere(12_000.0, 8.0, n_depth=8)
    spectrum = synthesize_helium_spectrum(
        atmosphere,
        np.linspace(700.0, 1_000.0, 9),
        stark_table=None,
        include_lines=False,
        metal_database=database,
        metal_abundances={"O": -6.0},
        metal_topbase_photoionization_database=topbase,
        n_angle=1,
    )
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))
    assert spectrum.metadata["metal_bound_free"].endswith(
        "synthetic TOPbase regression table"
    )

    structure = radiative_equilibrium_helium_atmosphere(
        12_000.0,
        8.0,
        stark_table=None,
        n_depth=8,
        n_continuum_wavelength=80,
        max_iterations=1,
        structure_solver="adaptive-newton",
        include_lines=False,
        include_metal_lines=False,
        mixing_length_alpha=None,
        metal_database=database,
        metal_abundances={"O": -6.0},
        metal_topbase_photoionization_database=topbase,
        n_angle=1,
    )
    assert structure.metadata["radiative_equilibrium_metal_opacity"]


def test_published_d6_composition_is_carbon_referenced_and_oxygen_dominated():
    assert SDSS_J1637_LOG_NUMBER_ABUNDANCE["C"] == 0.0
    assert 10.0 ** SDSS_J1637_LOG_NUMBER_ABUNDANCE["O"] == pytest.approx(
        1.862, rel=1.0e-3
    )
    assert "H" not in SDSS_J1637_LOG_NUMBER_ABUNDANCE
    assert "He" not in SDSS_J1637_LOG_NUMBER_ABUNDANCE


def test_d6_bulk_electron_donors_have_complete_saha_ladders():
    assert D6_STOUT_MAXIMUM_CHARGE["C"] == 6
    assert D6_STOUT_MAXIMUM_CHARGE["O"] == 8
    assert D6_STOUT_MAXIMUM_CHARGE["Ne"] == 10
    assert D6_STOUT_MAXIMUM_CHARGE["Mg"] == 2


def test_d6_high_ion_levels_survive_when_their_lines_are_suppressed():
    line = AtomicTransition(1, 2, 1.0e7, "E1", 1200.0, 0.1)
    levels = (
        AtomicLevel(1, 0.0, 1.0, "ground"),
        AtomicLevel(2, 10_000.0, 3.0, "upper"),
    )
    database = AtomicDatabase(MappingProxyType({
        ("C", 0): AtomicIon("C", 0, 12.011, 11.26, levels, (line,)),
        ("C", 3): AtomicIon("C", 3, 12.011, 64.49, levels, (line,)),
    }))
    restricted = retain_d6_structure_line_stages(database)
    assert restricted.ions[("C", 0)].transitions == (line,)
    assert restricted.ions[("C", 3)].transitions == ()
    assert restricted.ions[("C", 3)].levels == levels


def test_d6_uses_expanded_tlusty_neutral_carbon_oxygen_photoionization_atoms():
    carbon_url, carbon_checksum, carbon_element, carbon_charge = (
        D6_TLUSTY_TOPBASE_FILES["c1_28+12lev.dat"]
    )
    assert "c1_28%2B12lev.dat" in carbon_url
    assert len(carbon_checksum) == 64
    assert (carbon_element, carbon_charge) == ("C", 0)
    url, checksum, element, charge = D6_TLUSTY_TOPBASE_FILES[
        "o1_23+10lev.dat"
    ]
    assert "o1_23%2B10lev.dat" in url
    assert len(checksum) == 64
    assert (element, charge) == ("O", 0)
    assert D6_TLUSTY_RAP_FILES["fe2p_14+11lev.rap"][2:] == ("Fe", 1)



def test_bundled_norad_tables_match_their_pinned_checksums():
    import hashlib
    import lzma

    from wd_spectra.d6 import NORAD_LEVEL_RESOLVED_FILES

    root = ModelData.default().norad
    for name, (_, checksum, _, _) in NORAD_LEVEL_RESOLVED_FILES.items():
        assert name.endswith(".xz")
        digest = hashlib.sha256()
        with lzma.open(root / name, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        assert digest.hexdigest() == checksum, name

def test_bulk_metal_state_closes_particle_pressure_charge_nuclei_and_mass():
    database = _ground_state_database(("C", "O"))
    atmosphere = gray_d6_atmosphere(
        15_680.0,
        6.3,
        database,
        {"C": 0.0, "O": 0.27},
        n_depth=8,
    )
    state = bulk_metal_lte_state(
        atmosphere, database, {"C": 0.0, "O": 0.27}
    )

    total_nuclei = sum(state.element_number_density.values())
    recovered_pressure = (
        total_nuclei + state.electron_density
    ) * 1.380649e-16 * atmosphere.temperature
    np.testing.assert_allclose(recovered_pressure, atmosphere.gas_pressure, rtol=3e-12)

    charge = np.zeros_like(atmosphere.temperature)
    for element, population in state.ion_number_density.items():
        np.testing.assert_allclose(
            np.sum(population, axis=0),
            state.element_number_density[element],
            rtol=3e-15,
        )
        charge += np.sum(
            np.arange(population.shape[0])[:, np.newaxis] * population,
            axis=0,
        )
    np.testing.assert_allclose(charge, state.electron_density, rtol=3e-12)

    expected_mass = (
        ATOMIC_MASS_U["C"] * state.element_number_density["C"]
        + ATOMIC_MASS_U["O"] * state.element_number_density["O"]
    ) * 1.66053906892e-24
    np.testing.assert_allclose(state.total_mass_density, expected_mass, rtol=2e-15)
    assert sum(state.mass_fraction.values()) == pytest.approx(1.0)
    assert state.reference_species == "metal"
    assert state.composition_mode == "bulk"


def test_bulk_metal_atmosphere_has_no_hidden_hydrogen_or_helium_host():
    database = _ground_state_database(("C", "O"))
    atmosphere = gray_d6_atmosphere(
        15_680.0,
        6.3,
        database,
        {"C": 0.0, "O": 0.27},
        n_depth=6,
    )
    state = bulk_metal_lte_state(
        atmosphere, database, {"C": 0.0, "O": 0.27}
    )
    attached = atmosphere_with_bulk_metal_state(atmosphere, state)
    assert attached.hydrogen_lte_state is None
    assert attached.helium_lte_state is None
    assert np.all(attached.neutral_h_density == 0.0)
    assert np.all(attached.proton_density == 0.0)
    assert attached.metadata["composition"] == "hydrogen-helium-free-bulk-metals"


def test_cool_bulk_metal_negative_ion_continuum_is_physical():
    database = _ground_state_database(("O", "Ne", "Na"))
    abundance = {"O": 0.0, "Ne": 0.3, "Na": -1.1}
    atmosphere = gray_d6_atmosphere(
        10_000.0,
        5.5,
        database,
        abundance,
        reference_element="O",
        n_depth=8,
    )
    state = bulk_metal_lte_state(
        atmosphere, database, abundance, reference_element="O"
    )
    atmosphere = atmosphere_with_bulk_metal_state(atmosphere, state)
    oxygen_minus = oxygen_negative_ion_lte_number_density(atmosphere, state)
    assert oxygen_minus.shape == atmosphere.temperature.shape
    assert np.all(np.isfinite(oxygen_minus))
    assert np.all(oxygen_minus > 0.0)
    assert np.all(oxygen_minus < state.ion_number_density["O"][0])

    wavelength = np.asarray([3800.0, 5320.0, 6620.0, 9000.0])
    opacity = cool_metal_negative_ion_mass_absorption_coefficient(
        atmosphere, wavelength, state
    )
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.any(opacity > 0.0)


def test_bulk_metal_reference_abundance_must_be_zero():
    database = _ground_state_database(("C", "O"))
    atmosphere = gray_d6_atmosphere(
        15_680.0,
        6.3,
        database,
        {"C": 0.0, "O": 0.27},
        n_depth=6,
    )
    with pytest.raises(ValueError, match="reference element"):
        bulk_metal_lte_state(
            atmosphere, database, {"C": -0.2, "O": 0.27}
        )


def test_structure_line_ranking_respects_noncarbon_reference_element():
    database = _ground_state_database(("O",))
    atmosphere = gray_d6_atmosphere(
        10_000.0,
        5.5,
        database,
        {"O": 0.0},
        reference_element="O",
        n_depth=6,
    )
    wavelength = _structure_wavelength_grid(
        atmosphere,
        database,
        {"O": 0.0},
        80,
        1.0e-4,
        0,
        reference_element="O",
        flux_weighted_line_ranking=True,
    )
    assert wavelength.size == 80
    assert wavelength[0] == pytest.approx(100.0)
    assert wavelength[-1] == pytest.approx(100_000.0)


def test_structure_grid_resolves_minimum_metal_line_profile_support():
    database = _ground_state_database(("C",))
    levels = (
        AtomicLevel(1, 0.0, 1.0, "ground"),
        AtomicLevel(2, 10_000.0, 3.0, "upper"),
    )
    center = 1500.0
    line = AtomicTransition(1, 2, 1.0e8, "E1", center, 0.5)
    ions = dict(database.ions)
    ions[("C", 0)] = AtomicIon(
        "C", 0, ATOMIC_MASS_U["C"], IONIZATION_ENERGY_EV["C"][0], levels, (line,)
    )
    database = AtomicDatabase(MappingProxyType(ions))
    atmosphere = gray_d6_atmosphere(
        15_000.0, 6.5, database, {"C": 0.0}, n_depth=6
    )
    wavelength = _structure_wavelength_grid(
        atmosphere,
        database,
        {"C": 0.0},
        80,
        1.0e-4,
        10,
        flux_weighted_line_ranking=False,
    )
    for offset in (-0.251, -0.25, -0.125, 0.0, 0.125, 0.25, 0.251):
        assert np.min(np.abs(wavelength - (center + offset))) < 1.0e-10


def test_topbase_reader_shifts_edge_but_preserves_resonance_coordinate(tmp_path):
    ions = {
        ("C", 0): AtomicIon(
            "C",
            0,
            ATOMIC_MASS_U["C"],
            11.2603,
            (
                AtomicLevel(1, 0.0, 1.0, "ground"),
                AtomicLevel(2, 8065.54394, 3.0, "excited"),
            ),
            (),
        ),
        ("C", 1): AtomicIon(
            "C", 1, ATOMIC_MASS_U["C"], 24.383,
            (
                AtomicLevel(1, 0.0, 2.0, "ground"),
                # A 5-eV excited parent level of the next ion.
                AtomicLevel(2, 5.0 * 8065.54394, 4.0, "parent"),
            ),
            (),
        ),
    }
    database = AtomicDatabase(MappingProxyType(ions))
    levels = tmp_path / "c_1_levels.dat"
    levels.write_text(
        "LevMacro 6 1 1 -11.2603 0.0 1.0 1e21 ground\n"
        "LevMacro 6 1 2 -10.2603 1.0 3.0 1e-9 excited\n",
        encoding="ascii",
    )
    photo = tmp_path / "c_1_phot.dat"
    photo.write_text(
        # Ground-parent channel with a 0.26-eV theoretical threshold error.
        "PhotMacS 6 1 2 1 10.0 3\n"
        "PhotMac 10.0 1e-17\n"
        "PhotMac 18.0 2e-17\n"
        "PhotMac 34.0 1e-17\n"
        # Excited-parent channel of the ground level: 11.26 + 5.0 eV.
        "PhotMacS 6 1 1 1 16.3 2\n"
        "PhotMac 16.3 3e-18\n"
        "PhotMac 30.0 1e-18\n"
        # A table matching no parent channel of the excited level.
        "PhotMacS 6 1 2 1 12.5 2\n"
        "PhotMac 12.5 1e-17\n"
        "PhotMac 30.0 1e-17\n",
        encoding="ascii",
    )
    loaded = read_sirocco_topbase_lte_photoionization(
        [(levels, photo, "C", 0)], database
    )
    assert len(loaded.sections) == 2
    assert "1 sections without a parent channel omitted" in loaded.source
    section, parent = loaded.sections
    assert isinstance(section, TOPbaseLevelPhotoionization)
    assert section.threshold_energy_ev == pytest.approx(10.2603)
    assert section.source_threshold_energy_ev == pytest.approx(10.0)
    assert section.cross_section(10.0) == pytest.approx(0.0)
    assert section.cross_section(section.threshold_energy_ev) == pytest.approx(1e-17)
    assert section.cross_section(
        section.threshold_energy_ev + 8.0
    ) == pytest.approx(2e-17)
    # The excited-parent channel opens at its own experimental energy, not
    # at the ground-parent limit of its lower level.
    assert parent.threshold_energy_ev == pytest.approx(16.2603)
    assert parent.cross_section(12.0) == pytest.approx(0.0)

def test_sirocco_topbase_reader_rezeros_ionized_stage_excitation(tmp_path):
    database = _ground_state_database(("C",))
    levels = tmp_path / "c_2_levels.dat"
    levels.write_text(
        "LevMacro 6 2 1 -24.3830 11.2603 2.0 1e21 ground\n"
        "LevMacro 6 2 2 -23.3830 12.2603 4.0 1e-9 excited\n"
        "LevMacro 6 2 3 -5.6433 30.0000 6.0 1e-9 high\n",
        encoding="ascii",
    )
    photo = tmp_path / "c_2_phot.dat"
    photo.write_text(
        "PhotMacS 6 2 2 1 23.3830 2\n"
        "PhotMac 23.3830 1e-17\n"
        "PhotMac 46.7660 5e-18\n"
        "PhotMacS 6 2 3 1 5.6433 2\n"
        "PhotMac 5.6433 2e-17\n"
        "PhotMac 11.2866 1e-17\n",
        encoding="ascii",
    )
    loaded = read_sirocco_topbase_lte_photoionization(
        [(levels, photo, "C", 1)], database
    )
    assert len(loaded.sections) == 2
    assert loaded.sections[0].excitation_energy_ev == pytest.approx(1.0)
    assert loaded.sections[0].threshold_energy_ev == pytest.approx(
        IONIZATION_ENERGY_EV["C"][1] - 1.0
    )
    assert loaded.sections[1].excitation_energy_ev == pytest.approx(18.7397)


def test_sirocco_topbase_reader_accepts_simple_atom_format(tmp_path):
    database = _ground_state_database(("Si",))
    levels = tmp_path / "topbase_levels_si.dat"
    levels.write_text(
        "LevTop 14 1 310 1 -8.18 0.0 9 1.0 1e21 ground\n"
        "LevTop 14 1 120 1 -7.18 1.0 5 1.0 1e21 excited\n",
        encoding="ascii",
    )
    photo = tmp_path / "topbase_si_phot.dat"
    photo.write_text(
        "PhotTopS 14 1 120 1 7.0 2\n"
        "PhotTop 7.0 2e-17\n"
        "PhotTop 14.0 1e-17\n",
        encoding="ascii",
    )
    loaded = read_sirocco_topbase_lte_photoionization(
        [(levels, photo, "Si", 0)], database
    )
    assert len(loaded.sections) == 1
    section = loaded.sections[0]
    assert section.excitation_energy_ev == pytest.approx(1.0)
    assert section.statistical_weight == pytest.approx(5.0)
    assert section.threshold_energy_ev == pytest.approx(
        IONIZATION_ENERGY_EV["Si"][0] - 1.0
    )
    assert section.cross_section(section.threshold_energy_ev) == pytest.approx(
        2.0e-17
    )


def test_tlusty_topbase_reader_preserves_relative_fit_coordinate(tmp_path):
    database = _ground_state_database(("Mg",))
    atom = tmp_path / "mg1.dat"
    atom.write_text(
        "****** Levels\n"
        " 1.00000000E+15 1. 2 'Mg I ground' 0 0. 0\n"
        " 5.00000000E+14 3. 3 'Mg I excited' 0 0. 0\n"
        "****** Continuum transitions\n"
        " 1 3 1 103 0 0 0 1.0E-18 0.0\n"
        " 0.0 0.1 0.3\n"
        " 0.0 1.0 0.0\n"
        " 2 3 1 103 0 0 0 2.0E-18 0.0\n"
        " 0.0 0.1 0.3\n"
        " 0.30103 0.0 -1.0\n"
        "****** Line transitions\n",
        encoding="ascii",
    )
    loaded = read_tlusty_topbase_lte_photoionization(
        [(atom, "Mg", 0)], database
    )
    assert len(loaded.sections) == 2
    ground = loaded.sections[0]
    assert ground.threshold_energy_ev == pytest.approx(
        IONIZATION_ENERGY_EV["Mg"][0]
    )
    source_edge = ground.source_threshold_energy_ev
    resonance_energy = ground.threshold_energy_ev + source_edge * (
        10.0**0.1 - 1.0
    )
    assert ground.cross_section(resonance_energy) == pytest.approx(1.0e-17)

    with pytest.raises(ValueError, match="duplicate"):
        merge_topbase_photoionization_databases(loaded, loaded)


def test_tlusty_topbase_reader_accepts_observed_excitation_override(tmp_path):
    database = _ground_state_database(("Mg",))
    atom = tmp_path / "mg1.dat"
    atom.write_text(
        "****** Levels\n"
        " 1.00000000E+15 1. 2 'Mg I ground' 0 0. 0\n"
        " 5.00000000E+14 3. 3 'Mg I excited' 0 0. 0\n"
        "****** Continuum transitions\n"
        " 1 3 1 102 0 0 0 1.0E-18 0.0\n"
        " 0.0 0.3\n"
        " 0.0 -1.0\n"
        " 2 3 1 102 0 0 0 2.0E-18 0.0\n"
        " 0.0 0.3\n"
        " 0.30103 -1.0\n"
        "****** Line transitions\n",
        encoding="ascii",
    )
    observed_excitation = 4.25
    loaded = read_tlusty_topbase_lte_photoionization(
        [(atom, "Mg", 0)],
        database,
        excitation_energy_overrides_ev={("Mg", 0, 2): observed_excitation},
    )
    excited = loaded.sections[1]
    assert excited.excitation_energy_ev == pytest.approx(observed_excitation)
    assert excited.threshold_energy_ev == pytest.approx(
        IONIZATION_ENERGY_EV["Mg"][0] - observed_excitation
    )
    source_offset = excited.source_threshold_energy_ev * (10.0**0.3 - 1.0)
    assert excited.cross_section(excited.threshold_energy_ev + source_offset) == (
        pytest.approx(excited.cross_section_cm2[1])
    )


def test_tlusty_topbase_reader_preserves_delayed_continuum_opening(tmp_path):
    database = _ground_state_database(("C",))
    atom = tmp_path / "c1.dat"
    atom.write_text(
        "****** Levels\n"
        " 1.00000000E+15 5. 2 'C I 5So' 0 0. 0\n"
        " 1.00000000E+14 1. 3 'C II continuum' 0 0. 0\n"
        "****** Continuum transitions\n"
        " 1 2 1 102 0 0 0 1.0E-18 0.0\n"
        " 0.2 0.5\n"
        " 1.0 0.0\n"
        "****** Line transitions\n",
        encoding="ascii",
    )
    loaded = read_tlusty_topbase_lte_photoionization(
        [(atom, "C", 0)], database
    )
    assert len(loaded.sections) == 1
    section = loaded.sections[0]
    source_opening = section.source_threshold_energy_ev * 10.0**0.2
    target_opening = section.threshold_energy_ev + (
        source_opening - section.source_threshold_energy_ev
    )
    assert section.cross_section(section.threshold_energy_ev) == 0.0
    assert section.cross_section(target_opening - 1.0e-6) == 0.0
    assert section.cross_section(target_opening) == pytest.approx(1.0e-17)


def test_topbase_delayed_opening_tolerates_rebasing_roundoff():
    section = TOPbaseLevelPhotoionization(
        element="C",
        charge=0,
        level_index=33,
        excitation_energy_ev=0.0,
        statistical_weight=1.0,
        threshold_energy_ev=0.3825532461758616,
        source_threshold_energy_ev=0.37838240595773026,
        photon_energy_ev=np.asarray((
            0.37838240595773026,
            0.38453062545981226,
            0.8,
        )),
        cross_section_cm2=np.asarray((0.0, 2.0e-18, 1.0e-18)),
    )
    target_opening = section.threshold_energy_ev + (
        section.photon_energy_ev[1] - section.source_threshold_energy_ev
    )
    assert section.cross_section(section.threshold_energy_ev) == 0.0
    assert section.cross_section(target_opening - 1.0e-8) == 0.0
    assert section.cross_section(target_opening) == pytest.approx(2.0e-18)


def test_topbase_exact_rebased_threshold_keeps_first_cross_section():
    section = TOPbaseLevelPhotoionization(
        element="O",
        charge=0,
        level_index=6,
        excitation_energy_ev=10.740638193725818,
        statistical_weight=15.0,
        threshold_energy_ev=2.877416806274182,
        source_threshold_energy_ev=2.877315336606309,
        photon_energy_ev=np.asarray((2.877315336606309, 3.0, 4.0)),
        cross_section_cm2=np.asarray((4.260673837994696e-18, 3.0e-18, 1.0e-18)),
    )
    assert section.cross_section(section.threshold_energy_ev) == pytest.approx(
        section.cross_section_cm2[0]
    )


def test_topbase_positive_table_start_above_nominal_threshold_is_delayed():
    section = TOPbaseLevelPhotoionization(
        element="C",
        charge=0,
        level_index=6,
        excitation_energy_ev=4.182656,
        statistical_weight=5.0,
        threshold_energy_ev=7.077632,
        source_threshold_energy_ev=5.156666,
        photon_energy_ev=np.asarray((12.5112, 14.4732, 17.4366)),
        cross_section_cm2=np.asarray((1.953e-17, 1.877e-17, 1.612e-17)),
    )
    opening = section.threshold_energy_ev + (
        section.photon_energy_ev[0] - section.source_threshold_energy_ev
    )
    assert section.cross_section(section.threshold_energy_ev) == 0.0
    assert section.cross_section(opening - 1.0e-8) == 0.0
    assert section.cross_section(opening) == pytest.approx(1.953e-17)


def test_resonance_average_preserves_delayed_continuum_opening():
    section = TOPbaseLevelPhotoionization(
        element="C",
        charge=0,
        level_index=3,
        excitation_energy_ev=4.18,
        statistical_weight=5.0,
        threshold_energy_ev=7.08,
        source_threshold_energy_ev=7.08,
        photon_energy_ev=np.asarray((7.08, 8.0, 11.8, 12.0, 12.5, 14.0)),
        cross_section_cm2=np.asarray((0.0, 0.0, 1.0e-18, 5.0e-18, 2.0e-18, 1.0e-18)),
    )
    averaged = resonance_average_topbase_photoionization(
        TOPbasePhotoionizationDatabase((section,), "synthetic delayed C I"),
        0.03,
    ).sections[0]
    opening = section.threshold_energy_ev + (
        section.photon_energy_ev[2] - section.source_threshold_energy_ev
    )
    assert averaged.cross_section(section.threshold_energy_ev) == 0.0
    assert averaged.cross_section(opening - 1.0e-8) == 0.0
    assert averaged.cross_section(opening) > 0.0
    assert averaged.cross_section_cm2[0] == 0.0
    assert averaged.cross_section_cm2[1] == 0.0


def test_tlusty_rap_reader_recovers_level_resolved_cross_section(tmp_path):
    database = _ground_state_database(("Fe",))
    rap = tmp_path / "fe2.rap"
    rap.write_text(
        "26 2 1\n"
        "1 1000.0 10.0 4\n"
        "1.0e12 0.0\n"
        "3.0e15 0.0\n"
        "3.0001e15 2.0\n"
        "6.0e15 1.0\n",
        encoding="ascii",
    )
    loaded = read_tlusty_rap_lte_photoionization(
        [(rap, "Fe", 1)], database
    )
    assert len(loaded.sections) == 1
    section = loaded.sections[0]
    assert section.excitation_energy_ev == pytest.approx(
        1000.0 / 8065.543937
    )
    assert section.statistical_weight == 10.0
    assert section.cross_section(section.threshold_energy_ev) == pytest.approx(
        2.0e-18
    )
    assert loaded.ion_stages == frozenset({("Fe", 1)})


def test_norad_ls_reader_rebases_terms_and_converts_megabarns(tmp_path):
    database = _ground_state_database(("Ni",))
    photo = tmp_path / "ni2.px.txt"
    photo.write_text(
        "28 26 1\n"
        "0.0\n"
        "2 2 0 1\n"
        "-1.20 3\n"
        "0.01\n"
        "1.20 0.0\n"
        "1.30 2.0\n"
        "2.00 1.0\n"
        "28 26 1\n"
        "0.0\n"
        "4 1 1 1\n"
        "-0.40 3\n"
        "0.01\n"
        "0.60 0.0\n"
        "0.70 3.0\n"
        "1.00 1.0\n",
        encoding="ascii",
    )
    loaded = read_norad_ls_photoionization(
        photo, "Ni", 1, database, maximum_photon_energy_ev=20.0
    )
    assert len(loaded.sections) == 2
    ground, excited = loaded.sections
    assert ground.excitation_energy_ev == pytest.approx(0.0)
    assert ground.statistical_weight == pytest.approx(10.0)
    assert excited.excitation_energy_ev == pytest.approx(0.8 * 13.605693122994)
    assert excited.statistical_weight == pytest.approx(12.0)
    assert excited.threshold_energy_ev == pytest.approx(
        database.ions[("Ni", 1)].ionization_energy_ev
        - excited.excitation_energy_ev
    )
    target_opening = excited.threshold_energy_ev + (
        0.70 * 13.605693122994 - excited.source_threshold_energy_ev
    )
    assert excited.cross_section(excited.threshold_energy_ev) == 0.0
    assert excited.cross_section(target_opening) == pytest.approx(3.0e-18)
    assert loaded.ion_stages == frozenset({("Ni", 1)})


def test_norad_ls_reader_accepts_legacy_global_header(tmp_path):
    database = _ground_state_database(("Fe",))
    photo = tmp_path / "fe2.px.txt"
    photo.write_text(
        "26 24 P\n"
        "6 0 0 1\n"
        "1 3\n"
        "1.00 0.01\n"
        "1.00 0.0\n"
        "1.10 4.0\n"
        "2.00 1.0\n"
        "0 0 0 0\n",
        encoding="ascii",
    )
    loaded = read_norad_ls_photoionization(photo, "Fe", 1, database)
    assert len(loaded.sections) == 1
    section = loaded.sections[0]
    assert section.statistical_weight == pytest.approx(6.0)
    assert section.excitation_energy_ev == pytest.approx(0.0)
    target_opening = section.threshold_energy_ev + 0.1 * 13.605693122994
    assert section.cross_section(section.threshold_energy_ev) == 0.0
    assert section.cross_section(target_opening) == pytest.approx(4.0e-18)

    observed = read_norad_ls_photoionization(
        photo,
        "Fe",
        1,
        database,
        excitation_energy_overrides_ev={(6, 0, 0, 1): 2.0},
        only_excitation_overrides=True,
    )
    assert len(observed.sections) == 1
    assert observed.sections[0].excitation_energy_ev == pytest.approx(2.0)
    assert observed.sections[0].threshold_energy_ev == pytest.approx(
        database.ions[("Fe", 1)].ionization_energy_ev - 2.0
    )


def test_stout_ls_term_excitation_overrides_group_fine_structure():
    levels = (
        AtomicLevel(1, 0.0, 6.0, "3d9.(2D<5/2>)"),
        AtomicLevel(2, 8065.543937, 4.0, "3d9.(2D<3/2>)"),
        AtomicLevel(3, 2.0 * 8065.543937, 10.0, "3d8.4s.(2D<9/2>)"),
        AtomicLevel(4, 8065.543937, 28.0, "3d8.4s.(4F<13/2>)"),
        AtomicLevel(5, 3.0 * 8065.543937, 12.0, "3d8.4p.(4Do<11/2>)"),
        AtomicLevel(6, 100.0, 1.0, "unclassified"),
    )
    database = AtomicDatabase(MappingProxyType({
        ("Ni", 1): AtomicIon("Ni", 1, 58.6934, 18.168838, levels, ()),
    }))
    overrides = stout_ls_term_excitation_overrides(database, "Ni", 1)
    assert overrides[(2, 2, 0, 1)] == pytest.approx(0.4)
    assert overrides[(2, 2, 0, 2)] == pytest.approx(2.0)
    assert overrides[(4, 3, 0, 1)] == pytest.approx(1.0)
    assert overrides[(4, 2, 1, 1)] == pytest.approx(3.0)

    restricted = stout_ls_term_excitation_overrides(
        database, "Ni", 1, maximum_excitation_energy_ev=1.5
    )
    assert set(restricted) == {(2, 2, 0, 1), (4, 3, 0, 1)}


def test_d6_tlusty_observed_energy_override_identifies_o_i_3p_5p():
    oxygen = AtomicIon(
        "O",
        0,
        ATOMIC_MASS_U["O"],
        IONIZATION_ENERGY_EV["O"][0],
        (
            AtomicLevel(1, 0.0, 5.0, "2s2.2p4.(3P<2>)"),
            AtomicLevel(8, 86625.757, 3.0, "2s2.2p3.(4So).3p.(5P<1>)"),
            AtomicLevel(9, 86627.778, 5.0, "2s2.2p3.(4So).3p.(5P<2>)"),
            AtomicLevel(10, 86631.454, 7.0, "2s2.2p3.(4So).3p.(5P<3>)"),
        ),
        (),
    )
    database = AtomicDatabase(MappingProxyType({("O", 0): oxygen}))
    overrides = d6_tlusty_excitation_energy_overrides(database)
    observed_centroid = np.average(
        np.asarray((86625.757, 86627.778, 86631.454)),
        weights=np.asarray((3.0, 5.0, 7.0)),
    ) / 8065.543937
    assert overrides == {("O", 0, 6): pytest.approx(observed_centroid)}


def test_bulk_metal_thermodynamics_is_positive_and_recovers_monatomic_limit():
    database = _ground_state_database(("C", "O"))
    atmosphere = gray_d6_atmosphere(
        4_000.0,
        6.3,
        database,
        {"C": 0.0, "O": 0.27},
        n_depth=6,
        tau_min=1.0e-5,
        tau_max=1.0e-3,
        seed_rosseland_opacity=1.0e-8,
    )
    thermodynamics = bulk_metal_thermodynamics(
        atmosphere, database, {"C": 0.0, "O": 0.27}
    )
    assert np.all(np.isfinite(thermodynamics.specific_heat_constant_pressure))
    assert np.all(thermodynamics.specific_heat_constant_pressure > 0.0)
    np.testing.assert_allclose(
        thermodynamics.density_temperature_derivative, 1.0, rtol=2.0e-4
    )
    np.testing.assert_allclose(
        thermodynamics.adiabatic_temperature_gradient, 0.4, rtol=2.0e-4
    )


def test_bulk_metal_ml2_coupled_gradient_carries_requested_total_flux():
    database = _ground_state_database(("C", "O"))
    atmosphere = gray_d6_atmosphere(
        12_000.0,
        6.3,
        database,
        {"C": 0.0, "O": 0.27},
        n_depth=8,
        tau_min=0.1,
        tau_max=100.0,
    )
    thermodynamics = bulk_metal_thermodynamics(
        atmosphere, database, {"C": 0.0, "O": 0.27}
    )
    opacity = np.full(atmosphere.n_depth, 10.0)
    requested = np.full(
        atmosphere.n_depth,
        STEFAN_BOLTZMANN * atmosphere.effective_temperature**4,
    )
    gradient = ml2_temperature_gradient_for_total_flux_from_thermodynamics(
        atmosphere,
        opacity,
        requested,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
    )
    radiative = (
        16.0
        * STEFAN_BOLTZMANN
        * atmosphere.gravity
        * atmosphere.temperature**4
        / (3.0 * opacity * atmosphere.gas_pressure)
        * gradient
    )
    convective = ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere,
        opacity,
        gradient,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
    )
    np.testing.assert_allclose(radiative + convective, requested, rtol=2.0e-12)
    assert np.any(convective > 0.0)
    derivative = ml2_convective_flux_gradient_derivative_from_thermodynamics(
        atmosphere,
        opacity,
        gradient,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
    )
    gradient_step = 1.0e-7
    hotter_convective = ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere,
        opacity,
        gradient + gradient_step,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
    )
    np.testing.assert_allclose(
        derivative,
        (hotter_convective - convective) / gradient_step,
        rtol=3.0e-4,
        atol=1.0e-12,
    )

    # A full transfer solution can provide a more accurate local linearized
    # radiative response than diffusion in the tau_R ~ 1 transition layers.
    transfer_coefficient = radiative / gradient * 0.73
    transfer_gradient = (
        ml2_temperature_gradient_for_total_flux_from_thermodynamics(
            atmosphere,
            opacity,
            requested,
            thermodynamics.specific_heat_constant_pressure,
            thermodynamics.density_temperature_derivative,
            thermodynamics.adiabatic_temperature_gradient,
            radiative_flux_coefficient=transfer_coefficient,
        )
    )
    transfer_radiative = transfer_coefficient * transfer_gradient
    transfer_convective = ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere,
        opacity,
        transfer_gradient,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
    )
    np.testing.assert_allclose(
        transfer_radiative + transfer_convective, requested, rtol=2.0e-12
    )


def test_metal_rydberg_occupation_dissolves_high_levels_first():
    ion = AtomicIon(
        "O",
        0,
        ATOMIC_MASS_U["O"],
        IONIZATION_ENERGY_EV["O"][0],
        (),
        (),
    )
    ground = AtomicLevel(1, 0.0, 5.0, "ground")
    rydberg = AtomicLevel(
        2,
        (IONIZATION_ENERGY_EV["O"][0] - 0.14) * 8065.544005,
        7.0,
        "10d",
    )
    density = np.asarray([1.0e12, 1.0e16, 1.0e18])
    temperature = np.full_like(density, 13_000.0)
    ground_survival = metal_rydberg_level_occupation_probability(
        ion, ground, density, temperature
    )
    rydberg_survival = metal_rydberg_level_occupation_probability(
        ion, rydberg, density, temperature
    )
    assert np.all(np.diff(rydberg_survival) < 0.0)
    assert np.all(ground_survival > rydberg_survival)
    assert rydberg_survival[0] > 0.99
    assert rydberg_survival[-1] < 0.01
    holtsmark = metal_rydberg_level_occupation_probability(
        ion,
        rydberg,
        np.asarray([1.0e16]),
        np.asarray([13_000.0]),
        correlated_microfields=False,
    )
    correlated = metal_rydberg_level_occupation_probability(
        ion,
        rydberg,
        np.asarray([1.0e16]),
        np.asarray([13_000.0]),
    )
    assert holtsmark[0] < correlated[0]


def test_metal_ionic_microfield_density_uses_multicharge_moment():
    singly = np.asarray([3.0e15, 2.0e15])
    doubly = np.asarray([1.0e15, 4.0e15])
    electron_density = singly + 2.0 * doubly
    state = MetalLTEState(
        reference_species="metal",
        log_number_abundance=MappingProxyType({"O": 0.0}),
        element_number_density=MappingProxyType(
            {"O": singly + doubly}
        ),
        ion_number_density=MappingProxyType(
            {"O": np.stack((np.zeros(2), singly, doubly))}
        ),
        partition_function=MappingProxyType({}),
        electron_density=electron_density,
        metal_electron_density=electron_density,
        composition_mode="bulk",
    )
    expected = singly + 2.0**1.5 * doubly
    np.testing.assert_allclose(
        metal_ionic_microfield_perturber_density(state), expected
    )
    assert np.all(expected > electron_density)


def test_neutral_hard_sphere_dissolution_targets_large_rydberg_orbits():
    ion = AtomicIon(
        "O",
        0,
        ATOMIC_MASS_U["O"],
        IONIZATION_ENERGY_EV["O"][0],
        (),
        (),
    )
    lower = AtomicLevel(
        1,
        (IONIZATION_ENERGY_EV["O"][0] - 2.877) * 8065.544005,
        5.0,
        "2s2.2p3.(4So).3p.(5P<2>)",
    )
    upper = AtomicLevel(
        2,
        (IONIZATION_ENERGY_EV["O"][0] - 0.137) * 8065.544005,
        7.0,
        "2s2.2p3.(4So).10d.(5Do<3>)",
    )
    lower_radius = hydrogenic_metal_level_mean_radius_cm(ion, lower)
    upper_radius = hydrogenic_metal_level_mean_radius_cm(ion, upper)
    assert upper_radius > 20.0 * lower_radius
    neutral_density = {"O": np.asarray([8.0e14, 1.0e18])}
    neutral_radius = {"O": 3.0e-9}
    lower_survival = metal_neutral_hard_sphere_occupation_probability(
        lower_radius, neutral_density, neutral_radius
    )
    upper_survival = metal_neutral_hard_sphere_occupation_probability(
        upper_radius, neutral_density, neutral_radius
    )
    assert lower_survival[1] > 0.999
    # J1637's neutral density is only about 8e14 cm^-3, so this term is
    # negligible there; it becomes important in genuinely neutral-dense gas.
    assert upper_survival[0] > 0.998
    assert upper_survival[1] < 0.2

    joint = metal_rydberg_level_occupation_probability(
        ion,
        upper,
        np.asarray([1.0e16]),
        np.asarray([14_000.0]),
        neutral_perturber_number_density={"O": np.asarray([8.0e14])},
        neutral_perturber_radius_cm=neutral_radius,
    )
    charged_only = metal_rydberg_level_occupation_probability(
        ion,
        upper,
        np.asarray([1.0e16]),
        np.asarray([14_000.0]),
    )
    assert joint[0] == pytest.approx(charged_only[0], rel=2.0e-3)


def test_metal_nonideal_partition_smoothly_removes_rydberg_weight():
    ion = AtomicIon(
        "O",
        0,
        ATOMIC_MASS_U["O"],
        IONIZATION_ENERGY_EV["O"][0],
        (
            AtomicLevel(1, 0.0, 5.0, "ground"),
            AtomicLevel(
                2,
                (IONIZATION_ENERGY_EV["O"][0] - 0.14) * 8065.544005,
                80.0,
                "10d bundle",
            ),
        ),
        (),
    )
    temperature = np.asarray([13_000.0, 13_000.0])
    electron_density = np.asarray([1.0e12, 1.0e18])
    ideal = ion.partition_function(temperature)
    nonideal = ion.occupation_weighted_partition_function(
        temperature, electron_density
    )
    assert nonideal[0] == pytest.approx(ideal[0], rel=2.0e-5)
    assert nonideal[1] < ideal[1]
    assert nonideal[1] > 4.99


def test_metal_dissolution_retains_autoionizing_resonances():
    ion = AtomicIon(
        "O",
        0,
        ATOMIC_MASS_U["O"],
        IONIZATION_ENERGY_EV["O"][0],
        (),
        (),
    )
    lower = AtomicLevel(1, 102_662.026, 5.0, "bound 3s 1D")
    autoionizing_upper = AtomicLevel(2, 116_631.094, 5.0, "autoionizing 3p 1D")
    autoionizing_lower = AtomicLevel(3, 113_204.445, 3.0, "autoionizing 3p 1P")
    density = np.asarray([1.0e12, 1.0e18])
    temperature = np.asarray([13_000.0, 13_000.0])
    np.testing.assert_array_equal(
        metal_rydberg_transition_survival_probability(
            ion, lower, autoionizing_upper, density, temperature
        ),
        np.ones(2),
    )
    np.testing.assert_array_equal(
        metal_rydberg_transition_survival_probability(
            ion, autoionizing_lower, autoionizing_upper, density, temperature
        ),
        np.ones(2),
    )


def test_metal_dissolution_continuum_cutoff_uses_local_microfields():
    ion = AtomicIon(
        "O", 0, ATOMIC_MASS_U["O"], IONIZATION_ENERGY_EV["O"][0], (), ()
    )
    lower = AtomicLevel(1, 0.0, 5.0, "ground")
    upper = AtomicLevel(
        2,
        (IONIZATION_ENERGY_EV["O"][0] - 0.14) * 8065.544005,
        7.0,
        "10d",
    )
    survival = metal_rydberg_transition_survival_probability(
        ion,
        lower,
        upper,
        np.asarray([1.0e12, 1.0e18]),
        np.asarray([13_000.0, 13_000.0]),
        continuum_cutoff_probability=0.5,
    )
    np.testing.assert_array_equal(survival, [1.0, 0.0])


def test_topbase_dissolved_pseudocontinuum_is_confined_redward_of_edge():
    database = _ground_state_database(("O",))
    atmosphere = gray_d6_atmosphere(
        15_680.0,
        6.3,
        database,
        {"O": 0.0},
        reference_element="O",
        n_depth=8,
    )
    state = bulk_metal_lte_state(
        atmosphere,
        database,
        {"O": 0.0},
        reference_element="O",
    )
    section = TOPbaseLevelPhotoionization(
        "O",
        0,
        2,
        10.0,
        5.0,
        IONIZATION_ENERGY_EV["O"][0] - 10.0,
        IONIZATION_ENERGY_EV["O"][0] - 10.0,
        np.asarray([3.6181, 7.2362]),
        np.asarray([1.0e-17, 2.0e-18]),
    )
    topbase = TOPbasePhotoionizationDatabase((section,), "synthetic O I")
    wavelength = np.asarray([3300.0, 3500.0, 3800.0, 4200.0])
    opacity = topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        state,
        topbase,
    )
    np.testing.assert_array_equal(opacity[0], 0.0)
    assert np.all(opacity[1:3] >= 0.0)
    assert np.any(opacity[1:3] > 0.0)
    np.testing.assert_array_equal(opacity[3], 0.0)


def test_topbase_dissolved_pseudocontinuum_includes_magnesium_by_default():
    database = _ground_state_database(("O", "Mg"))
    atmosphere = gray_d6_atmosphere(
        12_000.0,
        5.5,
        database,
        {"O": 0.0, "Mg": -1.0},
        reference_element="O",
        n_depth=8,
    )
    state = bulk_metal_lte_state(
        atmosphere,
        database,
        {"O": 0.0, "Mg": -1.0},
        reference_element="O",
    )
    excitation = 4.34571
    threshold = IONIZATION_ENERGY_EV["Mg"][0] - excitation
    section = TOPbaseLevelPhotoionization(
        "Mg",
        0,
        3,
        excitation,
        3.0,
        threshold,
        threshold,
        np.asarray([threshold, threshold + 1.0]),
        np.asarray([7.4e-17, 2.0e-17]),
    )
    opacity = topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere,
        np.asarray([3800.0, 4000.0, 4500.0]),
        state,
        TOPbasePhotoionizationDatabase((section,), "synthetic Mg I"),
    )
    assert np.any(opacity > 0.0)



def test_o_i_3p5p_4d5d_stark_rate_reproduces_published_width():
    database = read_stout_atomic_database(
        ModelData.default().stout,
        elements=("O",),
        maximum_charge=2,
    )
    ion = database.ions[("O", 0)]
    line = next(
        line
        for line in ion.transitions
        if 6157.0 < line.wavelength_vacuum_angstrom < 6159.0
        and line.absorption_oscillator_strength > 0.02
    )
    rate = o_i_3p5p_nd5d_electron_stark_rate_coefficient(
        ion, line, np.asarray([10_000.0])
    )
    assert rate is not None
    center_cm = 6159.0e-8
    fwhm = (
        center_cm**2 * rate[0] * 1.0e16
        / (2.0 * np.pi * 2.99792458e10) * 1.0e8
    )
    assert fwhm == pytest.approx(0.192, rel=2.0e-4)


def test_o_i_series_stark_effective_n_cutoff_preserves_lower_members():
    database = read_stout_atomic_database(
        ModelData.default().stout,
        elements=("O",),
        maximum_charge=2,
    )
    ion = database.ions[("O", 0)]
    line_4d = next(
        line
        for line in ion.transitions
        if 6157.0 < line.wavelength_vacuum_angstrom < 6159.0
        and line.absorption_oscillator_strength > 0.02
    )
    line_9d = next(
        line
        for line in ion.transitions
        if 4577.0 < line.wavelength_vacuum_angstrom < 4579.0
    )
    temperature = np.asarray([14_000.0])
    assert o_i_3p5p_nd5d_electron_stark_rate_coefficient(
        ion,
        line_4d,
        temperature,
        minimum_effective_n=7.5,
    ) is None
    assert o_i_3p5p_nd5d_electron_stark_rate_coefficient(
        ion,
        line_9d,
        temperature,
        minimum_effective_n=7.5,
    ) is not None
