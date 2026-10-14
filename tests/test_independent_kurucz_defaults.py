"""Default data policy, preservation, widths and explicit opacity declarations."""
from dataclasses import replace
from importlib.resources import files
import hashlib
import json
import lzma

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra import metals as M
from wd_spectra.models import ModelData
from wd_spectra._kurucz_supplements import augment, catalogue, manifest, screen


@pytest.fixture(scope="module")
def atoms():
    elements = ("C", "Si", "S", "Ti", "Cr", "Fe")
    root = ModelData.default().stout
    raw = M.read_stout_atomic_database(root, elements=elements, maximum_charge=3,
        include_default_supplements=False)
    current = M.read_stout_atomic_database(root, elements=elements, maximum_charge=3)
    return raw, current


def test_default_inventory_preserves_original_records_and_known_levels(atoms):
    raw, current = atoms
    counts = {}
    for key, ion in current.ions.items():
        original = raw.ions[key]
        assert ion.transitions[:len(original.transitions)] == original.transitions
        assert ion.levels[:len(original.levels)] == original.levels
        added = [t for t in ion.transitions if t.independently_validated_supplement]
        if added:
            counts[key] = len(added)
            name = f"{key[0]}_{key[1]}"
            _, levels = catalogue(name)
            np.testing.assert_array_equal(levels, np.array([
                [l.index, l.energy_wavenumber, l.statistical_weight] for l in ion.levels]))
    assert counts == {("C", 0): 1, ("Si", 0): 433, ("Si", 1): 17,
        ("S", 0): 7, ("Ti", 0): 4, ("Cr", 1): 14891, ("Fe", 1): 4613}
    assert sum(len(i.supplemental_downward_rates) for i in current.ions.values()) == 1106
    assert augment(current) is current
    # Configuration compatibility must not re-import unqualified legacy data.
    for legacy in (M.augment_chromium_ii_kurucz_transitions,
                   M.augment_iron_ii_kurucz_uv_transitions,
                   M.augment_silicon_ii_kurucz_transitions):
        assert legacy(current) is current


def test_every_packaged_record_has_qualified_independent_evidence():
    root = files("wd_spectra").joinpath("data/atomic/independent-kurucz")
    for key, info in manifest()["ions"].items():
        rows, levels = catalogue(key)
        energy = {int(i): e for i, e, g in levels}
        element, charge = info["element"], info["charge"]
        limit = M.IONIZATION_ENERGY_EV[element][charge] * M.EV_TO_WAVENUMBER
        payload = root.joinpath(key + ".evidence.json.xz").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == info["evidence_sha256"]
        evidence = json.loads(lzma.decompress(payload))
        assert len(rows) == len(evidence)
        for row, proof in zip(rows, evidence):
            assert [int(row[0]), int(row[1])] == proof["pair"]
            assert row[3] == proof["Kurucz_f"]
            assert 0.1 <= proof["ratio"] <= 10
            assert proof["primary_evidence"]["source"]
            assert max(energy[int(row[0])], energy[int(row[1])]) < limit


def test_known_chromium_recoveries_use_the_common_policy(atoms):
    _, current = atoms
    cr = current.ions["Cr", 1]
    for pair in ((10, 92), (8, 86), (15, 92)):
        (line,) = [t for t in cr.transitions if (t.lower_index, t.upper_index) == pair]
        assert line.independently_validated_supplement
        assert line.radiative_damping_rate_s > 0
    # Previously imported resonance above the bare Si II threshold is held out.
    si = current.ions["Si", 1]
    assert not any(abs(t.wavelength_vacuum_angstrom - 1305.592) < 0.005
                   for t in si.transitions)
    for w in (1309.453, 1309.725):
        assert any(abs(t.wavelength_vacuum_angstrom - w) < 0.005 for t in si.transitions)


def test_subset_atom_does_not_gain_an_unrelated_level_extension(atoms):
    from types import MappingProxyType

    raw, _ = atoms
    cr = raw.ions["Cr", 1]
    levels = cr.levels[:100]
    indices = {level.index for level in levels}
    subset = replace(cr, levels=levels, transitions=tuple(t for t in cr.transitions
        if t.lower_index in indices and t.upper_index in indices))
    augmented = augment(M.AtomicDatabase(MappingProxyType({("Cr", 1): subset})))
    assert augmented.ions["Cr", 1].levels == levels
    assert all(t.lower_index in indices and t.upper_index in indices
               for t in augmented.ions["Cr", 1].transitions)


def test_declared_lines_are_never_rescreened_and_rates_do_not_accumulate(atoms):
    _, database = atoms
    atmosphere = gray_helium_atmosphere(15300, 8, n_depth=12)
    state = M.metal_lte_state(atmosphere, database, {"Fe": -6.47}, reference_species="He")
    wave = np.linspace(1387, 1389, 25)
    lines = M.selected_metal_lines(database, {"Fe": -6.47}, 1387, 1389, 1e-4, 20000)
    keys = tuple((i.element, i.charge, t.lower_index, t.upper_index) for i, t in lines)
    assert keys
    plain = M.metal_line_mass_absorption_coefficient(atmosphere, wave, database, state,
        transition_keys=keys)
    screened = M.metal_line_mass_absorption_coefficient(atmosphere, wave, database, state,
        transition_keys=keys, supplement_background_absorption=np.full((25, 12), 1e100))
    np.testing.assert_array_equal(plain, screened)
    repeated = M.metal_line_mass_absorption_coefficient(atmosphere, wave, database, state,
        transition_keys=keys)
    np.testing.assert_array_equal(plain, repeated)
    assert np.max(plain) > 0


def test_screen_protects_original_and_unified_or_ground_resonance_lines(atoms):
    _, database = atoms
    atmosphere = gray_helium_atmosphere(15300, 8, n_depth=12)
    state = M.metal_lte_state(atmosphere, database, {"Fe": -6.47}, reference_species="He")
    selected = M.selected_metal_lines(database, {"Fe": -6.47}, 1370, 1405, 1e-4, 20000)
    wave = np.linspace(1370, 1405, 10)
    retained = screen(atmosphere, wave, state, selected, np.full((10, 12), 1e100))
    assert all(pair in retained for pair in selected if not pair[1].independently_validated_supplement)
    for ion, line in retained:
        if line.independently_validated_supplement:
            lower = next(l for l in ion.levels if l.index == line.lower_index)
            assert line.helium_unified_profile is not None or (
                line.absorption_oscillator_strength >= .1 and lower.energy_wavenumber <= 500)
    with pytest.raises(ValueError, match="axes"):
        screen(atmosphere, wave, state, selected, np.ones((10, 11)))


def test_nlte_departure_coefficients_bypass_lte_supplement_screen(atoms):
    _, database = atoms
    atmosphere = gray_helium_atmosphere(15300, 8, n_depth=12)
    state = M.metal_lte_state(atmosphere, database, {"Fe": -6.47}, reference_species="He")
    wave = np.linspace(1387, 1389, 25)
    plain = M.metal_line_mass_absorption_coefficient(atmosphere, wave, database, state)
    bypass = M.metal_line_mass_absorption_coefficient(atmosphere, wave, database, state,
        level_departure_coefficient={}, supplement_background_absorption=np.full((25, 12), 1e100))
    np.testing.assert_array_equal(plain, bypass)


def test_excluding_calcium_also_excludes_its_coupled_line_emissivity():
    from wd_spectra.spectrum import synthesize_helium_spectrum

    atmosphere = gray_helium_atmosphere(15300, 8, n_depth=12)
    database = M.read_stout_atomic_database(ModelData.default().stout,
        elements=("Ca",), maximum_charge=2)
    wave = np.linspace(3965, 3972, 12)
    common = dict(include_lines=False, stark_table=None, metal_database=database, metal_abundances={"Ca": -6.9},
                  excluded_metal_line_elements=("Ca",), n_angle=2)
    plain = synthesize_helium_spectrum(atmosphere, wave, **common)
    excluded = synthesize_helium_spectrum(atmosphere, wave,
        ca_ii_resonance_collision_strengths="must-not-be-opened", **common)
    np.testing.assert_array_equal(plain.surface_flux_lambda, excluded.surface_flux_lambda)
