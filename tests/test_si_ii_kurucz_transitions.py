import numpy as np
import pytest

from wd_spectra.metals import (
    augment_silicon_ii_kurucz_transitions,
    read_stout_atomic_database,
)
from wd_spectra.models import DAZConfig, DZConfig, ModelData


@pytest.fixture(scope="module")
def stout():
    return read_stout_atomic_database(
        ModelData.default().stout, elements=("Si", "Mg"), maximum_charge=2,
        include_default_supplements=False,
    )


def _gf(ion, wavelength):
    levels = {level.index: level for level in ion.levels}
    found = [
        t for t in ion.transitions
        if t.transition_type == "E1"
        and abs(t.wavelength_vacuum_angstrom - wavelength) < 0.005
    ]
    return [
        t.absorption_oscillator_strength * levels[t.lower_index].statistical_weight
        for t in found
    ]


def test_stout_lacks_the_2d_2f_lines(stout):
    ion = stout.ions["Si", 1]
    for energy in (131677.1, 131918.8):
        level = [l for l in ion.levels if abs(l.energy_wavenumber - energy) < 0.1]
        assert len(level) == 1
        index = level[0].index
        assert not [t for t in ion.transitions if index in (t.lower_index, t.upper_index)]
    for wavelength in (1305.592, 1309.453, 1309.725):
        assert _gf(ion, wavelength) == []


def test_adds_kurucz_lines_between_existing_stout_levels(stout):
    augmented = augment_silicon_ii_kurucz_transitions(stout)
    before, after = stout.ions["Si", 1], augmented.ions["Si", 1]
    # Kurucz gf1401.all log gf: +0.474, +0.318, -0.862.
    for wavelength, log_gf in ((1305.592, 0.474), (1309.453, 0.318), (1309.725, -0.862)):
        (gf,) = _gf(after, wavelength)
        assert np.isclose(np.log10(gf), log_gf, atol=0.002)
    # Levels (and therefore partition functions) are unchanged; Stout lines
    # are retained first and unchanged.
    assert after.levels == before.levels
    assert after.transitions[: len(before.transitions)] == before.transitions
    assert len(after.transitions) > len(before.transitions)
    stout_levels = {level.index for level in before.levels}
    assert all(
        t.lower_index in stout_levels and t.upper_index in stout_levels
        for t in after.transitions
    )
    # Lines to Kurucz-only autoionizing levels (Si II 1399.94/1400.27 to
    # 150770 cm^-1) are not imported.
    assert _gf(after, 1400.266) == [] and _gf(after, 1399.943) == []
    assert augmented.ions["Mg", 1] is stout.ions["Mg", 1]
    assert "gf1401.all" in after.source


def test_idempotent_and_no_op_without_si_ii(stout):
    once = augment_silicon_ii_kurucz_transitions(stout)
    assert augment_silicon_ii_kurucz_transitions(once) is once
    magnesium = read_stout_atomic_database(
        ModelData.default().stout, elements=("Mg",), maximum_charge=2
    )
    assert augment_silicon_ii_kurucz_transitions(magnesium) is magnesium


def test_option_defaults_on():
    assert DZConfig().si_ii_kurucz_transitions is True
    assert DAZConfig().si_ii_kurucz_transitions is True
