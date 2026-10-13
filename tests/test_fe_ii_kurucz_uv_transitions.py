"""Missing UV opacity, preservation and warm-cache regressions for Fe II."""
from importlib.resources import files
import hashlib
import json
import lzma
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.metals import (
    AtomicDatabase,
    KURUCZ_FE_II_SHA256,
    augment_iron_ii_kurucz_uv_transitions,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_stout_atomic_database,
    selected_metal_lines,
)
from wd_spectra.models import ModelData
import wd_spectra.metals as metals_module


@pytest.fixture(scope="module")
def original():
    return read_stout_atomic_database(
        ModelData.default().stout, elements=("Fe", "Mg"), maximum_charge=2,
        include_default_supplements=False,
    )


def _near(ion, wavelength):
    return [t for t in ion.transitions
            if t.transition_type == "E1"
            and abs(t.wavelength_vacuum_angstrom - wavelength) < 0.002]


def test_bundled_slice_checksum_and_provenance():
    resource = files("wd_spectra").joinpath("data/atomic")
    content = lzma.decompress(resource.joinpath("gf2601_uv_missing.all.xz").read_bytes())
    provenance = json.loads(resource.joinpath("gf2601_uv_missing.provenance.json").read_text())
    assert len(content.splitlines()) == provenance["rows"] == 145
    assert hashlib.sha256(content).hexdigest() == KURUCZ_FE_II_SHA256
    assert provenance["decompressed_sha256"] == KURUCZ_FE_II_SHA256
    assert provenance["source_full_sha256"] == (
        "5a7d9d26eef58a2a32da523f8221f50cb08d3316cff20b45e22c8913e5390892"
    )


@pytest.mark.parametrize("wavelength,loggf", [
    (1190.882083, -1.033), (1193.574539, -0.918),
    (1357.760845, -0.317), (1357.796131, -0.391), (1361.373692, -0.580),
])
def test_recovers_missing_components_with_source_strengths(original, wavelength, loggf):
    before = original.ions["Fe", 1]
    assert not _near(before, wavelength)
    after = augment_iron_ii_kurucz_uv_transitions(original).ions["Fe", 1]
    (transition,) = _near(after, wavelength)
    lower = next(l for l in after.levels if l.index == transition.lower_index)
    assert np.log10(lower.statistical_weight * transition.absorption_oscillator_strength) == (
        pytest.approx(loggf, abs=1e-12)
    )
    assert transition.radiative_damping_rate_s > 0
    assert transition.electron_stark_rate_coefficient_cm3_s > 0
    assert transition.neutral_h_vdw_rate_coefficient_cm3_s > 0


def test_preserves_stout_and_other_ions_and_adds_only_one_level(original):
    updated = augment_iron_ii_kurucz_uv_transitions(original)
    before, after = original.ions["Fe", 1], updated.ions["Fe", 1]
    assert after.transitions[:len(before.transitions)] == before.transitions
    assert after.levels[:len(before.levels)] == before.levels
    assert len(after.transitions) - len(before.transitions) == 145
    assert len(after.levels) - len(before.levels) == 1
    assert all(updated.ions[key] is ion for key, ion in original.ions.items()
               if key != ("Fe", 1))
    # This existing, NIST-evaluated line must retain its f and damping data.
    assert _near(after, 1360.857567) == _near(before, 1360.857567)
    (new_level,) = after.levels[len(before.levels):]
    assert new_level.energy_wavenumber == pytest.approx(106524.4)
    assert new_level.statistical_weight == 16
    (new_component,) = _near(after, 1357.796131)
    assert new_component.upper_index == new_level.index


def test_added_level_partition_effect_is_small_and_explicit(original):
    before = original.ions["Fe", 1]
    after = augment_iron_ii_kurucz_uv_transitions(original).ions["Fe", 1]
    temperature = np.asarray([10800.0, 15300.0, 17280.0, 21000.0])
    fractional = after.partition_function(temperature) / before.partition_function(temperature) - 1
    assert np.all((fractional > 0) & (fractional < 1e-4))
    assert fractional[1] == pytest.approx(6.929726558801974e-6, rel=1e-8)


def test_warm_selection_and_width_caches_do_not_hide_added_lines(original):
    warmed = AtomicDatabase(original.ions, source=original.source)
    abundances = {"Fe": -6.47}
    args = (abundances, 1189.0, 1362.0, 1e-4, 20000)
    selected_metal_lines(warmed, *args, selection_temperature=15300)
    assert warmed._line_selection_cache
    warmed._unsold_hydrogen_coefficient_cache[("Fe", 1, 14, 293)] = None
    updated = augment_iron_ii_kurucz_uv_transitions(warmed)
    assert updated._line_selection_cache == {}
    assert updated._unsold_hydrogen_coefficient_cache == {}
    selected = selected_metal_lines(updated, *args, selection_temperature=15300)
    for wavelength in (1190.882083, 1193.574539, 1357.760845, 1357.796131, 1361.373692):
        assert any(ion.element == "Fe" and ion.charge == 1
                   and abs(t.wavelength_vacuum_angstrom - wavelength) < .002
                   for ion, t in selected)


@pytest.mark.parametrize("wavelength", [1193.574539, 1357.796131, 1361.373692])
def test_missing_component_contributes_real_positive_opacity(original, wavelength):
    updated = augment_iron_ii_kurucz_uv_transitions(original)
    ion = updated.ions["Fe", 1]
    (line,) = _near(ion, wavelength)
    atmosphere = gray_helium_atmosphere(15300.0, 8.0, n_depth=8)
    state = metal_lte_state(atmosphere, updated, {"Fe": -6.47})
    # Isolate the new component, while keeping identical populations in the
    # opacity comparison. This exercises the actual metal opacity path.
    key = (("Fe", 1, line.lower_index, line.upper_index),)
    wave = line.wavelength_vacuum_angstrom + np.asarray([-.05, 0, .05])
    absent = metal_line_mass_absorption_coefficient(
        atmosphere, wave, original, state, maximum_lines=None,
        transition_keys=key,
    )
    present = metal_line_mass_absorption_coefficient(
        atmosphere, wave, updated, state, maximum_lines=None,
        transition_keys=key,
    )
    assert np.all(absent == 0)
    assert np.all(np.isfinite(present))
    assert np.all(present >= 0)
    assert np.all(present[1] > 0)


def test_default_reader_opt_out_and_idempotence(original):
    updated = read_stout_atomic_database(
        ModelData.default().stout, elements=("Fe", "Mg"), maximum_charge=2,
    )
    assert len(updated.ions["Fe", 1].transitions) == len(original.ions["Fe", 1].transitions) + 145
    assert augment_iron_ii_kurucz_uv_transitions(updated) is updated


def test_missing_fe_is_noop_before_checksum_and_bad_checksum_fails(original, monkeypatch):
    magnesium = AtomicDatabase(MappingProxyType({key: ion for key, ion in original.ions.items()
                                               if key[0] == "Mg"}))
    monkeypatch.setattr(metals_module, "KURUCZ_FE_II_SHA256", "0" * 64)
    assert augment_iron_ii_kurucz_uv_transitions(magnesium) is magnesium
    with pytest.raises(ValueError, match="Fe II checksum mismatch"):
        augment_iron_ii_kurucz_uv_transitions(original)
