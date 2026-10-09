"""The default line list must retain Stout and recover the HIRES Cr II lines."""

import hashlib
from importlib.resources import files
import lzma

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.metals import (
    KURUCZ_CR_II_SHA256,
    augment_chromium_ii_kurucz_transitions,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_kurucz_gf100_atomic_database,
    read_stout_atomic_database,
    selected_metal_lines,
)
from wd_spectra.models import ModelData


@pytest.fixture(scope="module")
def chromium_databases():
    root = ModelData.default().stout
    raw = read_stout_atomic_database(
        root, elements=("Cr", "Mg"), maximum_charge=3,
        include_default_supplements=False,
    )
    default = read_stout_atomic_database(
        root, elements=("Cr", "Mg"), maximum_charge=3,
    )
    return raw, default


def test_default_crii_preserves_stout_and_other_ions(chromium_databases):
    raw, default = chromium_databases
    original, supplemented = raw.ions["Cr", 1], default.ions["Cr", 1]
    assert len(original.levels) == 913
    assert len(original.transitions) == 138
    assert len(supplemented.levels) == 914
    assert len(supplemented.transitions) == 90559
    assert supplemented.levels[:len(original.levels)] == original.levels
    assert supplemented.transitions[:len(original.transitions)] == original.transitions
    for key in raw.ions:
        if key != ("Cr", 1):
            assert default.ions[key] == raw.ions[key]
    assert KURUCZ_CR_II_SHA256 in supplemented.source
    fractional_change = (
        supplemented.partition_function(15000.0)
        / original.partition_function(15000.0) - 1.0
    )
    assert 0.0 < fractional_change < 1.5e-5


def test_default_crii_matches_the_full_explicit_kurucz_import(
    chromium_databases, tmp_path,
):
    raw, default = chromium_databases
    content = lzma.decompress(
        files("wd_spectra").joinpath("data/atomic/gf2401.all.xz").read_bytes()
    )
    assert hashlib.sha256(content).hexdigest() == KURUCZ_CR_II_SHA256
    path = tmp_path / "gf2401.all"
    path.write_bytes(content)
    explicit = read_kurucz_gf100_atomic_database(
        [path], raw, elements=("Cr",), replace_transitions=False,
        supplement_missing_transitions=True,
    )
    assert explicit.ions["Cr", 1].levels == default.ions["Cr", 1].levels
    assert explicit.ions["Cr", 1].transitions == default.ions["Cr", 1].transitions


def test_default_crii_is_idempotent(chromium_databases):
    _, default = chromium_databases
    assert augment_chromium_ii_kurucz_transitions(default) is default


def test_no_chromium_does_not_load_the_supplement(monkeypatch):
    import wd_spectra.metals as metals

    raw = read_stout_atomic_database(
        ModelData.default().stout, elements=("Mg",),
        include_default_supplements=False,
    )

    def unexpected_read(*args, **kwargs):
        pytest.fail("Cr-free models must not read the Kurucz resource")

    monkeypatch.setattr(metals, "files", unexpected_read)
    assert augment_chromium_ii_kurucz_transitions(raw) is raw
    default = read_stout_atomic_database(ModelData.default().stout, elements=("Mg",))
    assert default == raw


def test_crii_rejects_a_checksum_mismatch(chromium_databases, monkeypatch):
    import wd_spectra.metals as metals

    raw, _ = chromium_databases
    monkeypatch.setattr(metals, "KURUCZ_CR_II_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Cr II checksum mismatch"):
        augment_chromium_ii_kurucz_transitions(raw)


def test_missing_hires_crii_lines_are_selected_and_produce_opacity(
    chromium_databases,
):
    raw, default = chromium_databases
    pairs = {(10, 92), (8, 86), (15, 92)}  # Air 3368.049, 3433.309, 3585.294 Å.
    original_pairs = {
        (line.lower_index, line.upper_index) for line in raw.ions["Cr", 1].transitions
    }
    assert pairs.isdisjoint(original_pairs)
    abundances = {
        "O": -5.14, "Mg": -6.09, "Si": -6.36, "Ca": -7.69,
        "Ti": -8.96, "Cr": -8.16, "Mn": -8.54, "Fe": -6.45,
    }
    default = read_stout_atomic_database(
        ModelData.default().stout, elements=tuple(abundances), maximum_charge=3,
    )
    selected = selected_metal_lines(
        default, abundances, 3200.0, 8000.0,
        1.0e-4, 20000, 11787.0,
    )
    targets = [
        line for ion, line in selected
        if (ion.element, ion.charge) == ("Cr", 1)
        and (line.lower_index, line.upper_index) in pairs
    ]
    assert len(targets) == 3
    assert all(line.radiative_damping_rate_s is not None for line in targets)
    atmosphere = gray_helium_atmosphere(11787.0, 8.30, n_depth=8)
    state = metal_lte_state(atmosphere, default, {"Cr": -8.16})
    wavelength = np.asarray([line.wavelength_vacuum_angstrom for line in targets])
    keys = [("Cr", 1, lower, upper) for lower, upper in pairs]
    added_opacity = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, default, state, transition_keys=keys,
    )
    missing_opacity = metal_line_mass_absorption_coefficient(
        atmosphere, wavelength, raw, state, transition_keys=keys,
    )
    assert np.all(np.isfinite(added_opacity))
    assert np.all(added_opacity > 0.0)
    assert np.all(missing_opacity == 0.0)
