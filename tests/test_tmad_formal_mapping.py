"""TMAD population terms map only onto formal levels of the same term."""

import pytest

from wd_spectra.light_metal_nlte import (
    _spectroscopic_term_labels_match,
    _term_principal_quantum_number,
)
from wd_spectra.models import ModelData
from wd_spectra._pg1159_builder import build_model


def test_label_fields():
    assert _term_principal_quantum_number("C3010P 3PO") == 10
    assert _term_principal_quantum_number("O506S  3S") == 6
    assert _spectroscopic_term_labels_match("O506S  3S", "2s.6p.(3Po<1>)") is False
    assert _spectroscopic_term_labels_match("O507P  3PO", "2s.7h.(3Ho<4>)") is False
    assert _spectroscopic_term_labels_match("O507H  1HO", "2s.7g.(3G<5>)") is False
    assert _spectroscopic_term_labels_match("O506G  3G", "2s.6g.(3G<4>)") is True


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
    # No formal 2s6s 3S level exists; the term stays unmapped rather than
    # borrowing a 2s6p component.
    assert labels["O506S  3S"] == []
