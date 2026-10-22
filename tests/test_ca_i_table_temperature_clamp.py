"""Ca I 4227 + He: below the published temperature range the unified profile stays the line.

The table clamps temperature at its edges.  Replacing the line by the impact profile there put a
Lorentzian far wing into the 2700-3000 K upper layers of the coolest DZs.
"""
from pathlib import Path

import numpy as np

from wd_spectra import gray_helium_atmosphere
from wd_spectra.metals import (
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_ca_i_he_profile_table,
    read_stout_atomic_database,
)


def _write_ca_ion(root: Path, stage: int, upper_energy: float, upper_weight: float, einstein_a: float):
    directory = root / "stout" / "ca" / f"ca_{stage}"
    directory.mkdir(parents=True)
    (directory / f"ca_{stage}.nrg").write_text(
        f"20 09 05\n1 0.000 1 \"ground\"\n2 {upper_energy:.6f} {upper_weight:.1f} \"upper\"\n", encoding="utf-8")
    (directory / f"ca_{stage}.tp").write_text(f"20 09 05\nA 1 2 {einstein_a:.8e} E1\n", encoding="utf-8")


def test_cool_layer_keeps_the_unified_ca_i_profile(tmp_path: Path):
    _write_ca_ion(tmp_path, 1, 23652.304, 3.0, 2.18e8)  # Ca I 4227
    _write_ca_ion(tmp_path, 2, 60000.0, 4.0, 1.0e8)  # keep Ca II lines out of the window
    _write_ca_ion(tmp_path, 3, 200000.0, 3.0, 1.0e8)
    density_path = tmp_path / "cai-density.dat"
    density_path.write_text("".join(f"{n:.8e} {3500.0 + 10.0 * i:.1f} {s:.8e}\n" for n, s in
                                    ((1.0e21, 1.0e-20), (1.0e23, 1.0e-18)) for i in range(60)), encoding="ascii")
    temperature_path = tmp_path / "cai-temperature.dat"
    temperature_path.write_text("".join(f"{t:.1f} {3500.0 + 10.0 * i:.1f} 1.0e-19\n" for t in (4000.0, 6000.0)
                                        for i in range(60)), encoding="ascii")
    table = read_ca_i_he_profile_table(density_path, temperature_path)
    database = read_stout_atomic_database(tmp_path, elements=("Ca",), maximum_charge=2)
    atmosphere = gray_helium_atmosphere(3500.0, 8.0, n_depth=12)
    state = metal_lte_state(atmosphere, database, {"Ca": -9.0})
    wavelength = np.asarray([3600.0, 3900.0])
    n_he = atmosphere.helium_lte_state.neutral_he_density
    cool = (atmosphere.temperature < table.temperatures[0]) & (n_he > 1.0e21) & (n_he < 1.0e23)
    assert cool.any()
    layer = int(np.argmax(cool))
    opacity = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, database, state, maximum_lines=None, ca_i_he_profile_table=table)
    # the flat unified profile, not an impact Lorentzian (ratio ~3.7 between these points)
    assert opacity[0, layer] > 0.0
    np.testing.assert_allclose(opacity[0, layer], opacity[1, layer], rtol=0.05)
