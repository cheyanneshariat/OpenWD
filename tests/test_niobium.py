"""Bundled Nb atomic data and the homogeneous H/He host for trace metals."""

import csv
import math

import numpy as np
import pytest

from wd_spectra import DABConfig
from wd_spectra.atmosphere import (
    hydrogen_helium_continuum_atmosphere,
    radiative_equilibrium_hydrogen_helium_atmosphere,
)
from wd_spectra.metals import (
    IONIZATION_ENERGY_EV,
    NIOBIUM_ATOMIC_DATA_FILES,
    _niobium_data_directory,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_niobium_atomic_ion,
    read_stout_atomic_database,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.stellar import _dab_trace_metals


DATA = ModelData.default()

# Williams et al. (2026, arXiv:2610.07161), Extended Data Table 2.
PAPER_NB_III = (1451.628, 1456.692, 1513.831, 1524.927, 1639.512)
PAPER_NB_IV = (
    981.270, 992.567, 993.538, 1002.756, 1005.700, 1007.015, 1010.178,
    1013.808, 1030.271, 1035.216, 1044.904, 1049.610, 1050.976, 1054.384,
    1055.874, 1063.050, 1065.569, 1086.748, 1094.622, 1103.044, 1107.844,
    1116.081, 1119.835, 1120.243, 1125.274, 1127.498, 1128.611, 1136.689,
    1316.906, 1330.601, 1349.634, 1363.762, 1369.762, 1379.492, 1386.238,
    1418.884, 1420.662, 1424.368, 1434.140, 1434.223, 1444.491, 1447.491,
    1466.015, 1472.246, 1472.664, 1476.974, 1487.218, 1487.260, 1500.579,
    1508.721, 1510.832, 1517.461, 1524.383, 1532.606, 1532.981, 1534.059,
)


def _niobium_rows(name, delimiter="\t"):
    directory = _niobium_data_directory(DATA.stout)
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        return [
            {key: (value or "").strip().strip('"').lstrip("=").strip('"')
             for key, value in row.items()}
            for row in csv.DictReader(stream, delimiter=delimiter)
        ]


def test_niobium_is_a_consecutive_stout_compatible_ladder():
    database = read_stout_atomic_database(DATA.stout, elements=("Nb",), maximum_charge=5)
    stages = database.ion_stages("Nb")
    assert [ion.charge for ion in stages] == list(range(6))
    assert [len(ion.transitions) for ion in stages] == [0, 0, 76, 819, 0, 0]
    assert all(ion.levels[0].energy_wavenumber == 0.0 for ion in stages)
    assert stages[3].levels[0].label == "4d2.(3F<2>)"
    temperature = np.array([20_000.0, 35_000.0])
    # Closed-shell Nb VI: the first excited level lies at 272187 cm^-1.
    assert np.allclose(stages[5].partition_function(temperature), 1.0, rtol=1e-3)
    # Nb V ground term 4d 2D (g = 4 + 6 exp(-1867.4 hc/kT)) dominates; the
    # 5s level at 75930 cm^-1 adds about one percent at 35000 K.
    cool = np.array([5_000.0])
    expected = 4.0 + 6.0 * np.exp(-1867.4 * 1.438776877 / cool)
    assert np.allclose(stages[4].partition_function(cool), expected, rtol=1e-8)
    hot = stages[4].partition_function(temperature)
    ground = 4.0 + 6.0 * np.exp(-1867.4 * 1.438776877 / temperature)
    assert np.all((hot > ground) & (hot < 1.02 * ground))
    with pytest.raises(ValueError, match="charges 0-5"):
        read_niobium_atomic_ion(DATA.stout, 6)
    with pytest.raises(ValueError, match="charges 0-5"):
        read_stout_atomic_database(DATA.stout, elements=("Nb",), maximum_charge=6)


def test_niobium_ionization_energies_match_the_pinned_nist_snapshot():
    rows = _niobium_rows("nist-asd-nb-ionization.csv", delimiter=",")
    values = [float(row["Ionization Energy (eV)"]) for row in rows[:6]]
    assert IONIZATION_ENERGY_EV["Nb"] == tuple(values)
    assert set(NIOBIUM_ATOMIC_DATA_FILES) == {
        *(f"nist-asd-nb{stage}-levels.tsv" for stage in range(1, 7)),
        "nist-asd-nb4-lines.tsv", "nist-asd-nb-ionization.csv",
        "nilsson2010-nb3-table7.csv",
    }


def _printed_precision(text):
    """Half a unit in the last printed digit, as a log10 interval."""

    mantissa = text.lower().split("e")[0].rstrip(".")
    digits = len(mantissa.replace(".", "").replace("-", "").lstrip("0"))
    return math.log10(1.0 + 0.5 * 10.0 ** (1 - digits) / abs(float(mantissa)))


def test_niobium_iv_lines_reproduce_nist_wavelengths_and_strengths():
    ion = read_niobium_atomic_ion(DATA.stout, 3)
    rows = _niobium_rows("nist-asd-nb4-lines.tsv")
    assert len(rows) == len(ion.transitions) == 819
    weights = {level.index: level.statistical_weight for level in ion.levels}
    for row, line in zip(rows, ion.transitions):
        ritz = float(row["ritz_wl_vac(nm)"]) * 10.0
        assert line.wavelength_vacuum_angstrom == pytest.approx(ritz, abs=2.0e-3)
        assert line.einstein_a == float(row["Aki(s^-1)"])
        log_gf = math.log10(weights[line.lower_index] * line.absorption_oscillator_strength)
        # This export's "fik" column holds the line strength S (atomic units).
        strength = 10.0**log_gf * line.wavelength_vacuum_angstrom / 303.756
        assert math.log10(strength / float(row["fik"])) == pytest.approx(
            0.0, abs=_printed_precision(row["Aki(s^-1)"])
            + _printed_precision(row["fik"]) + 1e-3
        )
        if row["ritz_wl_vac(nm)"] == "195.5929":
            # NIST prints log gf = 1.76 here, although its own A and S both
            # give log gf = 0.00; f follows from A, as for every line.
            assert row["log_gf"] == "1.76" and abs(log_gf) < 0.01
            continue
        decimals = len(row["log_gf"].split(".")[1]) if "." in row["log_gf"] else 0
        assert log_gf == pytest.approx(
            float(row["log_gf"]),
            abs=0.5 * 10.0**-decimals + _printed_precision(row["Aki(s^-1)"]) + 1e-3,
        )


def test_niobium_iii_reproduces_nilsson_table_7():
    ion = read_niobium_atomic_ion(DATA.stout, 2)
    rows = _niobium_rows("nilsson2010-nb3-table7.csv", delimiter=",")
    assert len(rows) == len(ion.transitions) == 76
    weights = {level.index: level.statistical_weight for level in ion.levels}
    pairs = {(line.lower_index, line.upper_index) for line in ion.transitions}
    assert len(pairs) == 76
    for row, line in zip(rows, ion.transitions):
        log_gf = math.log10(weights[line.lower_index] * line.absorption_oscillator_strength)
        assert log_gf == pytest.approx(float(row["log_gf"]), abs=0.025)
        assert weights[line.upper_index] * line.einstein_a == pytest.approx(float(row["gA_s"]))


def test_every_published_hs0209_niobium_line_has_an_atomic_match():
    for charge, wavelengths in ((2, PAPER_NB_III), (3, PAPER_NB_IV)):
        centers = np.array([
            line.wavelength_vacuum_angstrom
            for line in read_niobium_atomic_ion(DATA.stout, charge).transitions
        ])
        offsets = [np.min(abs(centers - wavelength)) for wavelength in wavelengths]
        assert max(offsets) < 0.015


def _hs0209_seed():
    return hydrogen_helium_continuum_atmosphere(35_800.0, 7.9, 1.9, n_depth=20)


def test_homogeneous_host_metal_closure_reduces_to_the_mixed_eos():
    atmosphere = _hs0209_seed()
    elements = ("Ca", "Nb")
    database = read_stout_atomic_database(
        DATA.stout, elements=elements, maximum_charge={"Ca": 4, "Nb": 5}
    )
    negligible = metal_lte_state(
        atmosphere, database, {element: -25.0 for element in elements},
        log_hydrogen_abundance=1.9,
    )
    assert np.allclose(negligible.electron_density, atmosphere.electron_density, rtol=1e-7)
    assert np.allclose(
        negligible.trace_hydrogen_state.proton_density,
        atmosphere.hydrogen_lte_state.proton_density, rtol=1e-7,
    )
    assert np.allclose(
        negligible.trace_hydrogen_state.hydrogen_nuclei_density,
        atmosphere.hydrogen_lte_state.hydrogen_nuclei_density, rtol=1e-12,
    )
    # log N(Nb)/N(H) = -6.33 is log N(Nb)/N(He) = -4.43 for H/He = 10^1.9.
    state = metal_lte_state(
        atmosphere, database, {"Ca": -4.64 + 1.9, "Nb": -6.33 + 1.9},
        log_hydrogen_abundance=1.9,
    )
    niobium = state.ion_number_density["Nb"]
    assert np.allclose(
        niobium.sum(axis=0),
        10.0**-6.33 * atmosphere.hydrogen_lte_state.hydrogen_nuclei_density,
        rtol=1e-9,
    )
    photosphere = np.argmin(abs(np.log(atmosphere.rosseland_optical_depth / 0.1)))
    fractions = niobium[:, photosphere] / niobium[:, photosphere].sum()
    assert np.argmax(fractions) == 3
    assert fractions[2] > 1e-3 and fractions[4] > 0.05 and fractions[5] < 0.01


def test_niobium_lines_enter_the_metal_line_opacity():
    atmosphere = _hs0209_seed()
    database = read_stout_atomic_database(DATA.stout, elements=("Nb",), maximum_charge=5)
    state = metal_lte_state(atmosphere, database, {"Nb": -4.43}, log_hydrogen_abundance=1.9)
    wave = np.array([1433.9, 1434.140, 1434.223, 1434.45])
    without = metal_line_mass_absorption_coefficient(atmosphere, wave, database, state)
    with_stark = metal_line_mass_absorption_coefficient(
        atmosphere, wave, database, state, include_classical_electron_stark=True
    )
    photosphere = np.argmin(abs(np.log(atmosphere.rosseland_optical_depth / 0.1)))
    assert without[1, photosphere] > 1e3 * without[0, photosphere]
    assert without[2, photosphere] > 1e3 * without[3, photosphere]
    # Stark wings raise the opacity between the cores of the Nb IV doublet.
    assert with_stark[3, photosphere] > without[3, photosphere]


def test_homogeneous_metal_host_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="requires log_hydrogen_abundance and metals"):
        radiative_equilibrium_hydrogen_helium_atmosphere(
            35_800.0, 7.9, 1.9, stark_table=None, homogeneous_metal_host=True,
        )
    with pytest.raises(ValueError, match="molecular"):
        _dab_trace_metals(
            DABConfig(abundances={"Nb": -6.0}, include_molecules=True), DATA
        )
    with pytest.raises(ValueError, match="log_hydrogen_to_helium"):
        _dab_trace_metals(DABConfig(abundances={"He": -1.0}), DATA)
    metals = _dab_trace_metals(
        DABConfig(log_hydrogen_to_helium=1.9, abundances={"Ni": -6.32, "Nb": -6.33},
                  maximum_metal_charge={"Ni": 4, "Nb": 5}),
        DATA,
    )
    assert metals["helium_abundances"] == pytest.approx({"Ni": -4.42, "Nb": -4.43})
    assert "Ni 3" not in metals["ions_without_photoionization"]
    assert {f"Nb {charge}" for charge in range(6)} <= set(
        metals["ions_without_photoionization"]
    )


def test_metal_free_dab_configuration_is_unchanged():
    assert DABConfig().abundances is None
    assert _dab_trace_metals(DABConfig(), DATA) is None
