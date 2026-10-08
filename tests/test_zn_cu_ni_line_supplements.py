"""Opt-in E1 lines for Zn, Cu and Ni ions whose Stout files are forbidden-only."""

import math
import re

import numpy as np
import pytest

from wd_spectra import DABConfig
from wd_spectra.atmosphere import hydrogen_helium_continuum_atmosphere
from wd_spectra.metals import (
    KURUCZ_IRON_GROUP_POSITION_FILES,
    RAUCH_ZN_CU_ATOMIC_DATA_FILES,
    RAUCH_ZN_CU_ION_FILES,
    _rauch_data_directory,
    _rauch_records,
    _stout_level_parity,
    atomic_database_with_kurucz_iron_group_positions,
    atomic_database_with_rauch_zn_cu_transitions,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    rauch_zn_cu_line_counts,
    read_stout_atomic_database,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.stellar import _dab_trace_metals

DATA = ModelData.default()


@pytest.fixture(scope="module")
def stout():
    return read_stout_atomic_database(
        DATA.stout, elements=("Ni", "Cu", "Zn"), maximum_charge=6
    )


@pytest.fixture(scope="module")
def rauch(stout):
    return atomic_database_with_rauch_zn_cu_transitions(stout, DATA.stout)


def _e1(ion):
    return [line for line in ion.transitions if line.transition_type == "E1"]


def _line_near(ion, wavelength, tolerance=0.002):
    matches = [
        line for line in _e1(ion)
        if abs(line.wavelength_vacuum_angstrom - wavelength) <= tolerance
    ]
    assert len(matches) == 1, wavelength
    return matches[0]


def test_stout_iv_to_vi_ions_of_ni_cu_zn_have_no_e1_lines(stout):
    for element in ("Ni", "Cu", "Zn"):
        for charge in (3, 4, 5):
            ion = stout.ions[(element, charge)]
            assert ion.transitions and not _e1(ion), (element, charge)
    assert len(stout.ions[("Zn", 3)].transitions) == 10
    assert len(stout.ions[("Cu", 3)].transitions) == 23


def test_rauch_files_are_pinned_and_complete():
    directory = _rauch_data_directory(DATA.stout)
    assert sorted(path.name for path in directory.iterdir() if path.name != "README.md") == sorted(
        RAUCH_ZN_CU_ATOMIC_DATA_FILES
    )
    records = _rauch_records(str(directory.resolve()))
    # Row counts of CDS J/A+A/564/A41 and of the TOSS Cu IV-VI tables.
    assert {key: len(rows) for key, rows in records.items()} == {
        ("Zn", 3): 400, ("Zn", 4): 1879,
        ("Cu", 3): 8785, ("Cu", 4): 5456, ("Cu", 5): 3797,
    }


def test_stout_term_parity_agrees_with_the_configuration(stout):
    orbital = re.compile(r"^\d+([spdfghik])(\d*)$")
    for element, charge in RAUCH_ZN_CU_ION_FILES:
        for level in stout.ions[(element, charge)].levels:
            parity = sum(
                "spdfghik".index(match.group(1)) * int(match.group(2) or 1)
                for match in map(orbital.match, level.label.split("."))
                if match
            ) % 2
            assert _stout_level_parity(level.label) == "eo"[parity], level


def test_rauch_lines_attach_to_stout_levels(stout, rauch):
    counts = rauch_zn_cu_line_counts(stout, DATA.stout)
    assert {key: value["attached"] for key, value in counts.items()} == {
        "Zn 3": 400, "Zn 4": 1651, "Cu 3": 7576, "Cu 4": 4218, "Cu 5": 2717,
    }
    for element, charge in RAUCH_ZN_CU_ION_FILES:
        before = stout.ions[(element, charge)]
        after = rauch.ions[(element, charge)]
        assert after.levels == before.levels
        assert after.transitions[:len(before.transitions)] == before.transitions
        added = after.transitions[len(before.transitions):]
        assert len(added) == counts[f"{element} {charge}"]["attached"]
        assert len({(line.lower_index, line.upper_index) for line in added}) == len(added)
    assert rauch.ions[("Cu", 6)] == stout.ions[("Cu", 6)]
    assert atomic_database_with_rauch_zn_cu_transitions(rauch, DATA.stout) is rauch


def test_zn_iv_reproduces_the_cds_table(rauch):
    """Every CDS Zn IV row maps to one Stout pair with its J, parity and gf."""

    ion = rauch.ions[("Zn", 3)]
    lines = {(line.lower_index, line.upper_index): line for line in _e1(ion)}
    records = _rauch_records(str(_rauch_data_directory(DATA.stout).resolve()))[("Zn", 3)]

    def level(record, prefix):
        matches = [
            candidate for candidate in ion.levels
            if abs(candidate.energy_wavenumber - record[f"{prefix}_energy"]) <= 1.0
            and candidate.statistical_weight == 2 * record[f"{prefix}_j"] + 1
            and _stout_level_parity(candidate.label) == record[f"{prefix}_parity"]
        ]
        assert len(matches) == 1
        return matches[0]

    for record in records:
        lower, upper = level(record, "lower"), level(record, "upper")
        line = lines[(lower.index, upper.index)]
        # CDS wavelengths follow from energies rounded to 1 cm^-1.
        assert abs(1e8 / line.wavelength_vacuum_angstrom - 1e8 / record["wavelength"]) < 1.0
        log_gf = math.log10(lower.statistical_weight * line.absorption_oscillator_strength)
        # log gf is rounded to 0.01 dex and gA to three digits.
        assert log_gf == pytest.approx(record["log_gf"], abs=0.012)
    assert len(lines) == len(records)


def test_wavelength_matching_reproduces_energy_matching_for_zn():
    """The Cu method (no level energies) recovers the Zn pairs from wavelengths."""

    database = read_stout_atomic_database(DATA.stout, elements=("Zn",), maximum_charge=4)
    directory = str(_rauch_data_directory(DATA.stout).resolve())
    for charge in (3, 4):
        ion = database.ions[("Zn", charge)]
        groups = {}
        for level in ion.levels:
            key = (_stout_level_parity(level.label), (level.statistical_weight - 1) / 2)
            groups.setdefault(key, []).append(level)
        energy_pairs, wavelength_pairs = [], []
        for record in _rauch_records(directory)[("Zn", charge)]:
            def matches(prefix):
                return [
                    level for level in groups[(record[f"{prefix}_parity"], record[f"{prefix}_j"])]
                    if abs(level.energy_wavenumber - record[f"{prefix}_energy"]) <= 1.0
                ]
            by_energy = [(a.index, b.index) for a in matches("lower") for b in matches("upper")]
            by_wavelength = [
                (a.index, b.index)
                for a in groups[(record["lower_parity"], record["lower_j"])]
                for b in groups[(record["upper_parity"], record["upper_j"])]
                if abs(b.energy_wavenumber - a.energy_wavenumber - 1e8 / record["wavelength"]) <= 0.6
            ]
            if len(by_energy) == 1 and len(by_wavelength) == 1:
                energy_pairs.append(by_energy[0])
                wavelength_pairs.append(by_wavelength[0])
            elif len(by_energy) == 1:
                # Wavelength-only matching may lose a line but never mislabels one.
                assert by_energy[0] in by_wavelength or not by_wavelength
        assert wavelength_pairs == energy_pairs
        assert len(energy_pairs) >= {3: 399, 4: 1640}[charge]


def test_williams_figure_1_zn_iv_and_cu_iv_lines_are_present(stout, rauch):
    # Extended Data Table 2 identifications in the 1365-1370.5 A window.
    for element, wavelength in (("Zn", 1365.253), ("Zn", 1369.510), ("Cu", 1367.519)):
        line = _line_near(rauch.ions[(element, 3)], wavelength, tolerance=0.006)
        assert line.absorption_oscillator_strength > 0.05
        assert not [
            other for other in _e1(stout.ions[(element, 3)])
            if abs(other.wavelength_vacuum_angstrom - wavelength) < 1.0
        ]


def test_kurucz_supplement_adds_ni_iv_1452(stout):
    database = read_stout_atomic_database(DATA.stout, elements=("Ni",), maximum_charge=6)
    merged = atomic_database_with_kurucz_iron_group_positions(database)
    ion = merged.ions[("Ni", 3)]
    levels = {level.index: level for level in ion.levels}
    line = _line_near(ion, 1452.220)
    lower, upper = levels[line.lower_index], levels[line.upper_index]
    # Both levels are Stout's; Kurucz gf2803.pos gives log gf = +0.696.
    assert lower.label == "3d6.(3H).4s.(4H<13/2>)"
    assert upper.label == "3d6.(3H).4p.(4Io<15/2>)"
    assert math.log10(lower.statistical_weight * line.absorption_oscillator_strength) == (
        pytest.approx(0.696, abs=1e-6)
    )
    assert line.electron_stark_rate_coefficient_cm3_s is not None
    original = database.ions[("Ni", 3)]
    assert ion.levels[:len(original.levels)] == original.levels
    assert ion.transitions[:len(original.transitions)] == original.transitions
    assert len(_e1(ion)) == 5659
    assert [len(merged.ions[("Ni", charge)].levels) - len(database.ions[("Ni", charge)].levels)
            for charge in (3, 4, 5)] == [31, 20, 140]
    # Ni VII is excluded: its Kurucz ground term would duplicate Stout's.
    assert ("Ni", 6) not in KURUCZ_IRON_GROUP_POSITION_FILES
    assert merged.ions[("Ni", 6)] == database.ions[("Ni", 6)]
    temperature = np.array([35_800.0])
    for charge, limit in ((3, 1.001), (4, 1.0001), (5, 1.02)):
        ratio = (merged.ions[("Ni", charge)].partition_function(temperature)
                 / database.ions[("Ni", charge)].partition_function(temperature))
        assert 1.0 <= ratio[0] < limit
    assert atomic_database_with_kurucz_iron_group_positions(merged) is merged


def test_supplements_are_opt_in_for_the_dab_preset():
    abundances = {"Ni": -6.32, "Cu": -6.46, "Zn": -6.24}
    charges = {element: 4 for element in abundances}
    assert DABConfig().metal_line_supplements == ()
    plain = _dab_trace_metals(
        DABConfig(log_hydrogen_to_helium=1.9, abundances=abundances,
                  maximum_metal_charge=charges), DATA,
    )["database"]
    assert all(not _e1(plain.ions[(element, 3)]) for element in abundances)
    full = _dab_trace_metals(
        DABConfig(log_hydrogen_to_helium=1.9, abundances=abundances,
                  maximum_metal_charge=charges,
                  metal_line_supplements=("rauch-zn-cu", "kurucz-fe-ni")), DATA,
    )["database"]
    assert all(_e1(full.ions[(element, charge)])
               for element in abundances for charge in (3, 4))
    with pytest.raises(ValueError, match="require trace-metal abundances"):
        _dab_trace_metals(DABConfig(metal_line_supplements=("rauch-zn-cu",)), DATA)
    for bad in (("rauch-zn-cu", "rauch-zn-cu"), ("nist",), ["rauch-zn-cu"]):
        with pytest.raises(ValueError, match="metal_line_supplements"):
            _dab_trace_metals(
                DABConfig(log_hydrogen_to_helium=1.9, abundances=abundances,
                          metal_line_supplements=bad), DATA,
            )


def test_supplemented_zn_iv_line_enters_the_opacity():
    atmosphere = hydrogen_helium_continuum_atmosphere(35_800.0, 7.9, 1.9, n_depth=20)
    stout = read_stout_atomic_database(DATA.stout, elements=("Zn",), maximum_charge=4)
    rauch = atomic_database_with_rauch_zn_cu_transitions(stout, DATA.stout)
    wave = np.array([1369.0, 1369.510, 1370.1])
    photosphere = np.argmin(abs(np.log(atmosphere.rosseland_optical_depth / 0.1)))
    opacity = {}
    for label, database in (("stout", stout), ("rauch", rauch)):
        # Williams et al. log Zn/H = -6.24 is -4.34 relative to He here.
        state = metal_lte_state(atmosphere, database, {"Zn": -4.34}, log_hydrogen_abundance=1.9)
        opacity[label] = metal_line_mass_absorption_coefficient(atmosphere, wave, database, state)
    assert opacity["rauch"][1, photosphere] > 1e3 * opacity["stout"][1, photosphere]
    assert opacity["rauch"][1, photosphere] > 1e2 * opacity["rauch"][0, photosphere]
