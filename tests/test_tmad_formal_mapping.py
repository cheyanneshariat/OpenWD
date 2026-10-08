"""TMAD population terms map only onto formal levels of the same term."""

import pytest

from wd_spectra.light_metal_nlte import (
    _TMAD_WAVELENGTH_PATTERN,
    _spectroscopic_term_labels_match,
    _term_orbital_letter,
    _term_principal_quantum_number,
    _term_spin_multiplicity,
)
from wd_spectra.models import ModelData
from wd_spectra._pg1159_builder import _model_atom, build_model
from wd_spectra.pg1159_presets import PG1424_BEST_LINE_ATOM_COUNTS, mutable_atom_counts


def test_label_fields():
    assert _term_principal_quantum_number("C3010P 3PO") == 10
    assert _term_principal_quantum_number("O506S  3S") == 6
    assert _spectroscopic_term_labels_match("O506S  3S", "2s.6p.(3Po<1>)") is False
    assert _spectroscopic_term_labels_match("O507P  3PO", "2s.7h.(3Ho<4>)") is False
    assert _spectroscopic_term_labels_match("O507H  1HO", "2s.7g.(3G<5>)") is False
    assert _spectroscopic_term_labels_match("O506G  3G", "2s.6g.(3G<4>)") is True


@pytest.mark.parametrize("label, multiplicity", [
    ("O507H 51HO", 1),   # 7h J=5 1Ho: the J field precedes 2S+1
    ("O507H 53HO", 3),
    ("O609L152L", 2),    # 2J=15
    ("O710D 13D", 3),
    ("O403P\"14DO", 4),
    ("O504F'41G", 1),
    ("O507H  1HO", 1),
    ("C3010P 3PO", 3),
    ("2s.7h.(3Ho<5>)", 3),
])
def test_tmad_fine_structure_keys_carry_their_multiplicity(label, multiplicity):
    assert _term_spin_multiplicity(label) == multiplicity
    assert _spectroscopic_term_labels_match("O507H 51HO", "2s.7h.(3Ho<5>)") is False


@pytest.fixture(scope="module")
def pg1159_model():
    composition = {"He": 0.33, "C": 0.50, "O": 0.17}
    model, _ = build_model(
        ModelData.default(), composition, structure_only=False,
        population_iterations=120, oxygen_atom_preset="extended54-complete",
    )
    return model


def _mapping_labels(model, element, charge):
    population = (
        model.oxygen_population_atomic_database if element == "O"
        else model.carbon_population_atomic_database
    ).ions[(element, charge)]
    formal = model.atomic_database.ions[(element, charge)]
    formal_label = {level.index: level.label for level in formal.levels}
    mapping = (
        model.oxygen_formal_level_mapping if element == "O"
        else model.carbon_formal_level_mapping
    )
    return {
        level.label: [formal_label[key[2]] for key in mapping.get((element, charge, level.index), ())]
        for level in population.levels
    }


def test_no_mapped_formal_level_contradicts_its_term(pg1159_model):
    for element, charge in (("C", 2), ("C", 3), ("O", 3), ("O", 4), ("O", 5)):
        for term, formal_levels in _mapping_labels(pg1159_model, element, charge).items():
            for label in formal_levels:
                assert _spectroscopic_term_labels_match(term, label) is not False, (
                    f"{element}{charge + 1} {term!r} mapped onto {label!r}"
                )


def test_o_v_high_terms_keep_their_own_levels(pg1159_model):
    labels = _mapping_labels(pg1159_model, "O", 4)
    assert sorted(labels["O506P  3PO"]) == ["2s.6p.(3Po<0>)", "2s.6p.(3Po<1>)", "2s.6p.(3Po<2>)"]
    assert sorted(labels["O507H  3HO"]) == ["2s.7h.(3Ho<4>)", "2s.7h.(3Ho<5>)", "2s.7h.(3Ho<6>)"]
    assert labels["O507H  1HO"] == ["2s.7h.(1Ho<5>)"]
    # Stout has no 2s6s 3S level. The importer appends the TMAD level
    # instead of borrowing a 2s6p component, and the term maps onto it.
    assert labels["O506S  3S"] == ["O506S  3S"]


def _tmad_rbb_keys(path):
    """(wavelength, f) -> TMAD (lower, upper) keys of every RBB record."""
    keys = {}
    in_rbb = False
    for record in path.read_text(errors="replace").splitlines():
        if record.strip() == "RBB":
            in_rbb = True
            continue
        if in_rbb and record.strip().startswith("0"):
            in_rbb = False
            continue
        if not in_rbb or not record or record.lstrip().startswith("."):
            continue
        match = _TMAD_WAVELENGTH_PATTERN.search(record)
        fields = record[20:].split()
        if match is None or len(fields) < 3:
            continue
        try:
            f_value = float(fields[2])
        except ValueError:
            continue
        key = (round(float(match.group(1)), 3), round(f_value, 6))
        keys.setdefault(key, set()).add((record[:10].strip(), record[10:20].strip()))
    return keys


@pytest.mark.parametrize("charge, roman", [(2, "III"), (3, "IV"), (4, "V"), (5, "VI"), (6, "VII")])
def test_formal_importer_keeps_each_line_on_its_own_terms(pg1159_model, charge, roman):
    path = ModelData.default().cache / f"tmad-atoms/O_{roman}_syn"
    rbb = _tmad_rbb_keys(path)
    ion = pg1159_model.atomic_database.ions[("O", charge)]
    label = {level.index: level.label for level in ion.levels}
    checked = 0
    for transition in ion.transitions:
        pairs = rbb.get((
            round(transition.wavelength_vacuum_angstrom, 3),
            round(transition.absorption_oscillator_strength, 6),
        ))
        if not pairs:
            continue
        lower, upper = label[transition.lower_index], label[transition.upper_index]
        assert any(
            _spectroscopic_term_labels_match(lo, lower) is not False
            and _spectroscopic_term_labels_match(up, upper) is not False
            for lo, up in pairs
        ), (f"O {roman} {transition.wavelength_vacuum_angstrom:.3f} A: "
            f"{sorted(pairs)} imported onto {lower!r} -> {upper!r}")
        checked += 1
    assert checked > 50


def test_lte_parent_mapping_never_contradicts_its_term(pg1159_model):
    cache = ModelData.default().cache
    counts = mutable_atom_counts(PG1424_BEST_LINE_ATOM_COUNTS)
    for element, name, attribute in (
        ("C", "C_III-V", "carbon_formal_lte_parent_mapping"),
        ("O", "O_III-VII", "oxygen_formal_lte_parent_mapping"),
    ):
        atom = _model_atom(
            element, cache / f"tmad-atoms/{name}", pg1159_model.atomic_database, counts[element]
        )
        assert set(atom.formal_lte_term_labels) == set(atom.formal_lte_parent_mapping)
        formal = {
            (element, charge, level.index): level.label
            for (symbol, charge), ion in pg1159_model.atomic_database.ions.items()
            if symbol == element
            for level in ion.levels
        }
        for formal_key, term in atom.formal_lte_term_labels.items():
            assert _spectroscopic_term_labels_match(term, formal[formal_key]) is not False, (
                f"LTE term {term!r} mapped onto {formal[formal_key]!r}"
            )
        # The model uses the same assignments.
        assert set(getattr(pg1159_model, attribute)) <= set(atom.formal_lte_parent_mapping)


_ORBITAL = "SPDFGHIKLMNOQ"


def test_strong_formal_oxygen_lines_obey_ls_dipole_rules(pg1159_model):
    """Swapped near-degenerate levels show up as forbidden strong lines."""
    checked = 0
    for charge in (2, 3, 4, 5, 6):
        ion = pg1159_model.atomic_database.ions[("O", charge)]
        label = {level.index: level.label for level in ion.levels}
        for transition in ion.transitions:
            if transition.absorption_oscillator_strength < 0.01:
                continue
            lower, upper = label[transition.lower_index], label[transition.upper_index]
            fields = (_term_spin_multiplicity(lower), _term_spin_multiplicity(upper),
                      _term_orbital_letter(lower), _term_orbital_letter(upper))
            if None in fields:
                continue
            spin_lower, spin_upper = fields[:2]
            l_lower, l_upper = (_ORBITAL.index(value) for value in fields[2:])
            assert spin_lower == spin_upper and abs(l_lower - l_upper) <= 1 and l_lower + l_upper > 0, (
                f"O {charge + 1} {transition.wavelength_vacuum_angstrom:.3f} A: {lower!r} -> {upper!r}"
            )
            checked += 1
    assert checked > 2000
