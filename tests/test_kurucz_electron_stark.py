from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

import wd_spectra.models.stellar as stellar
from wd_spectra.metals import (
    AtomicDatabase,
    atomic_database_with_kurucz_electron_stark,
    read_kurucz_stark_table,
    read_stout_atomic_database,
)
from wd_spectra.models import DZConfig, ModelData


TRIPLET = (1260.4221, 1264.7375, 1265.0018)


@pytest.fixture(scope="module")
def stout_si_ii():
    return read_stout_atomic_database(
        ModelData.default().stout, elements=("Si",), maximum_charge=2
    )


def _triplet(ion):
    return [
        t for t in ion.transitions
        if min(abs(t.wavelength_vacuum_angstrom - w) for w in TRIPLET) < 0.01
    ]


def test_table_contains_vald_kurucz_value_for_si_ii_1260():
    table = read_kurucz_stark_table()
    assert len(table) > 100_000
    # VALD3/Kurucz K14: log Gamma_e/n_e = -5.75 for the 3p 2P - 3d 2D triplet.
    entries = [v for k, v in table.items() if k[:2] == ("Si", 1)]
    triplet = [v for v in entries if np.isclose(v[2] - v[0], 1.0e8 / 1260.4221, atol=0.1)]
    assert triplet and all(np.isclose(v[-1], 10.0 ** -5.75) for v in triplet)


def test_attaches_stark_only_where_missing(stout_si_ii):
    ion = stout_si_ii.ions["Si", 1]
    lines = _triplet(ion)
    assert len(lines) == 3
    assert all(t.electron_stark_rate_coefficient_cm3_s is None for t in lines)
    updated = atomic_database_with_kurucz_electron_stark(stout_si_ii)
    new = _triplet(updated.ions["Si", 1])
    for old, line in zip(lines, new):
        assert np.isclose(line.electron_stark_rate_coefficient_cm3_s, 10.0 ** -5.75)
        # nothing else is changed
        assert line.absorption_oscillator_strength == old.absorption_oscillator_strength
        assert line.einstein_a == old.einstein_a
        assert line.neutral_h_vdw_rate_coefficient_cm3_s == old.neutral_h_vdw_rate_coefficient_cm3_s
        assert line.radiative_damping_rate_s == old.radiative_damping_rate_s


def test_existing_stark_width_takes_precedence(stout_si_ii):
    ion = stout_si_ii.ions["Si", 1]
    target = _triplet(ion)[0]
    sentinel = 1.234e-5
    transitions = tuple(
        replace(t, electron_stark_rate_coefficient_cm3_s=sentinel) if t is target else t
        for t in ion.transitions
    )
    ions = dict(stout_si_ii.ions)
    ions["Si", 1] = replace(ion, transitions=transitions)
    database = AtomicDatabase(MappingProxyType(ions), source=stout_si_ii.source)
    updated = atomic_database_with_kurucz_electron_stark(database)
    kept = [
        t for t in updated.ions["Si", 1].transitions
        if (t.lower_index, t.upper_index) == (target.lower_index, target.upper_index)
    ]
    assert kept[0].electron_stark_rate_coefficient_cm3_s == sentinel


def test_unmatched_levels_are_left_unchanged(stout_si_ii):
    ion = stout_si_ii.ions["Si", 1]
    shifted = tuple(replace(level, energy_wavenumber=level.energy_wavenumber + 1.0e4)
                    for level in ion.levels)
    ions = dict(stout_si_ii.ions)
    ions["Si", 1] = replace(ion, levels=shifted)
    database = AtomicDatabase(MappingProxyType(ions), source=stout_si_ii.source)
    updated = atomic_database_with_kurucz_electron_stark(database)
    assert all(
        a.electron_stark_rate_coefficient_cm3_s == b.electron_stark_rate_coefficient_cm3_s
        for a, b in zip(updated.ions["Si", 1].transitions, ion.transitions)
    )


class _Stop(Exception):
    pass


@pytest.mark.parametrize("enabled", [False, True])
def test_dz_config_option_controls_attachment(monkeypatch, enabled):
    calls = []

    def fake_attach(database):
        calls.append(database)
        return database

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(stellar, "atomic_database_with_kurucz_electron_stark", fake_attach)
    monkeypatch.setattr(stellar, "read_verner_photoionization_database", stop)
    assert DZConfig().kurucz_electron_stark is False
    with pytest.raises(_Stop):
        stellar.compute_dz(DZConfig(kurucz_electron_stark=enabled))
    assert len(calls) == (1 if enabled else 0)


def test_forbidden_lines_sharing_levels_are_not_broadened():
    # Stout lists Si III 1298.89 both as E1 and as an M2 line between the same
    # levels; only the E1 transition was matched to Kurucz.
    database = read_stout_atomic_database(
        ModelData.default().stout, elements=("Si",), maximum_charge=2
    )
    updated = atomic_database_with_kurucz_electron_stark(database)
    lines = [
        t for t in updated.ions["Si", 2].transitions
        if abs(t.wavelength_vacuum_angstrom - 1298.892) < 0.01
    ]
    types = {t.transition_type: t.electron_stark_rate_coefficient_cm3_s for t in lines}
    assert types["E1"] is not None
    assert types["M2"] is None
