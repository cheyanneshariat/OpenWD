"""Mg I 2852 + He: the impact Voigt is enveloped only within the Weisskopf frequency.

Beyond v/rho_W the line is quasi-static and the Allard unified table is the wing; the impact
Lorentzian must not supply opacity thousands of angstroms away at n_He ~ 1e23 cm^-3.
"""
from pathlib import Path
from types import MappingProxyType

import numpy as np

from test_metals import _write_stout_ion
from wd_spectra import gray_helium_atmosphere
from wd_spectra.metals import (
    AtomicDatabase,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_mg_he_red_wing_table,
    read_stout_atomic_database,
)


def test_dense_mg_resonance_far_wing_comes_only_from_the_unified_table(tmp_path: Path):
    _write_stout_ion(tmp_path, 1, upper_energy=35051.264, upper_weight=3.0, einstein_a=4.67e8)
    _write_stout_ion(tmp_path, 2, upper_energy=35669.31, upper_weight=2.0, einstein_a=2.58e8)
    _write_stout_ion(tmp_path, 3, upper_energy=40000.0, upper_weight=3.0, einstein_a=1.0e8)
    table_path = tmp_path / "fig5.dat"
    table_path.write_text("".join(f" {t} {w:.7E} 0.1000000E-19\n" for t in (4000, 8000)
                                  for w in (2900.0, 3500.0, 4200.0)), encoding="utf-8")
    table = read_mg_he_red_wing_table(table_path)
    database = read_stout_atomic_database(tmp_path, elements=("Mg",), maximum_charge=2)
    neutral = AtomicDatabase(MappingProxyType({("Mg", 0): database.ions[("Mg", 0)]}))
    atmosphere = gray_helium_atmosphere(4500.0, 8.5, n_depth=12)
    state = metal_lte_state(atmosphere, database, {"Mg": -8.0})
    center = database.ions[("Mg", 0)].transitions[0].wavelength_vacuum_angstrom
    wavelength = np.asarray([center + 2.0, 3500.0, 6000.0])
    # dense layer inside the table's temperature range (the old 100-HWHM envelope reached ~3600 A here)
    dense = (atmosphere.helium_lte_state.neutral_he_density > 1.0e22) & (atmosphere.temperature < 8000.0)
    assert dense.any()
    deep = int(np.argmax(dense))

    def opacity(mg_table):
        return metal_line_mass_absorption_coefficient(
            atmosphere, wavelength, neutral, state, maximum_lines=None, mg_he_red_wing_table=mg_table)

    with_table = opacity(table)
    impact_only = opacity(None)
    # without a unified table the (impact) Lorentzian reaches 6000 A in the dense layers
    assert impact_only[2, deep] > 1.0e-3 * impact_only[1, deep]
    # with it, 3000 A from the line nothing beyond the table is left, the table wing and core remain
    assert with_table[2, deep] < 1.0e-8 * with_table[1, deep]
    assert with_table[1, deep] > 0.0 and with_table[0, deep] > with_table[1, deep]


def test_mg_resonance_below_the_table_temperatures_keeps_the_unified_wing(tmp_path: Path):
    _write_stout_ion(tmp_path, 1, upper_energy=35051.264, upper_weight=3.0, einstein_a=4.67e8)
    _write_stout_ion(tmp_path, 2, upper_energy=35669.31, upper_weight=2.0, einstein_a=2.58e8)
    _write_stout_ion(tmp_path, 3, upper_energy=40000.0, upper_weight=3.0, einstein_a=1.0e8)
    table_path = tmp_path / "fig5.dat"
    table_path.write_text("".join(f" {t} {w:.7E} 0.1000000E-19\n" for t in (5000, 8000)
                                  for w in (2900.0, 3500.0, 4200.0)), encoding="utf-8")
    table = read_mg_he_red_wing_table(table_path)
    database = read_stout_atomic_database(tmp_path, elements=("Mg",), maximum_charge=2)
    neutral = AtomicDatabase(MappingProxyType({("Mg", 0): database.ions[("Mg", 0)]}))
    atmosphere = gray_helium_atmosphere(4500.0, 8.5, n_depth=12)
    state = metal_lte_state(atmosphere, database, {"Mg": -8.0})
    cool = (atmosphere.temperature < 5000.0) & (atmosphere.helium_lte_state.neutral_he_density > 1.0e21)
    assert cool.any()
    layer = int(np.argmax(cool))
    opacity = metal_line_mass_absorption_coefficient(
        atmosphere, np.asarray([3500.0, 3600.0]), neutral, state, maximum_lines=None, mg_he_red_wing_table=table)
    # 650 A from the line is far outside the impact window; only the (temperature-clamped) table is there
    assert opacity[0, layer] > 0.0


def test_mg_wing_below_the_coolest_table_follows_the_boltzmann_law(tmp_path: Path):
    table_path = tmp_path / "fig5.dat"
    rows = {4000: (1.0e-19, 4.0e-20, 0.0), 6000: (2.0e-19, 2.0e-19, 5.0e-21)}
    table_path.write_text("".join(f" {t} {w:.7E} {s:.7E}\n" for t, sig in rows.items()
                                  for w, s in zip((2900.0, 3500.0, 4200.0), sig)), encoding="utf-8")
    table = read_mg_he_red_wing_table(table_path)
    wavelength = np.asarray([2900.0, 3500.0, 4200.0])
    at_3000 = table.cross_section(wavelength, 3000.0, 1.0e21).ravel()
    slope = np.log(np.asarray(rows[6000][:2]) / rows[4000][:2]) / (1 / 4000 - 1 / 6000)
    np.testing.assert_allclose(at_3000[:2], np.asarray(rows[4000][:2]) * np.exp(-slope * (1 / 3000 - 1 / 4000)))
    assert at_3000[2] == 0.0  # no wing where the coolest table has none
    np.testing.assert_allclose(table.cross_section(wavelength, 4000.0, 1.0e21).ravel(), rows[4000])


def test_mg_wing_has_no_edges_where_the_figure_curves_end(tmp_path: Path):
    # CDS temperature curves ending at 3950 A (4000 K) and 4250 A (6000 K), density curves at 4400 A
    cds = tmp_path / "fig5.dat"
    cds.write_text("".join(f" {t} {w:.7E} {1e-18 * np.exp(-(w - 2900.0) / 300.0) * (t / 6000.0) ** 4:.7E}\n"
                           for t, end in ((4000, 3950.0), (6000, 4250.0)) for w in np.arange(2900.0, end + 1.0, 25.0)),
                   encoding="utf-8")
    density = tmp_path / "fig6.dat"
    density.write_text("".join(f" {n:.1e} {w:.1f} {2e-18 * n / 1e22 * np.exp(-(w - 2900.0) / 300.0):.7E}\n"
                               for n in (1e21, 1e22) for w in np.arange(2900.0, 4401.0, 25.0)), encoding="utf-8")
    table = read_mg_he_red_wing_table(cds, density)
    wavelength = np.arange(3800.0, 4700.0, 5.0)
    for temperature in (3000.0, 4000.0, 5000.0):
        sigma = table.cross_section(wavelength, temperature, 1.0e22).ravel()
        assert np.all(sigma > 0.0)
        step = np.abs(np.diff(np.log(sigma)))
        assert step.max() < 0.1  # smooth exponential wing: 5 A steps change ln sigma by 1/60
        assert np.all(np.diff(sigma[wavelength > 4400.0]) < 0.0)  # continues to fall beyond the figure
