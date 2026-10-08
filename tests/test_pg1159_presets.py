from __future__ import annotations

import pytest

from wd_spectra import (
    PG1424_BEST_LINE_ATOM_COUNTS,
    PG1424_COMPOSITION_LADDER_ATOM_COUNTS,
    PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS,
    PG1424_PUBLISHED_MASS_FRACTIONS,
    PG1424_RELAXED_STRUCTURE_ATOM_COUNTS,
    WERNER2015_LINE_ATOM_COUNTS,
    mutable_atom_counts,
)


def test_pg1424_best_atom_is_an_immutable_accepted_configuration() -> None:
    assert dict(PG1424_BEST_LINE_ATOM_COUNTS["C"]) == {
        2: 97,
        3: 54,
        4: 1,
    }
    assert dict(PG1424_BEST_LINE_ATOM_COUNTS["O"]) == {
        2: 47,
        3: 83,
        4: 105,
        5: 14,
        6: 1,
    }
    assert WERNER2015_LINE_ATOM_COUNTS["O"][5] == 9
    with pytest.raises(TypeError):
        PG1424_BEST_LINE_ATOM_COUNTS["O"][5] = 9  # type: ignore[index]


def test_pg1424_mutable_atom_copy_cannot_change_the_preset() -> None:
    copied = mutable_atom_counts(PG1424_BEST_LINE_ATOM_COUNTS)
    copied["O"][5] = 9
    assert PG1424_BEST_LINE_ATOM_COUNTS["O"][5] == 14


def test_pg1424_atom_stages_keep_structure_history_explicit() -> None:
    assert PG1424_COMPOSITION_LADDER_ATOM_COUNTS["O"][5] == 4
    assert PG1424_RELAXED_STRUCTURE_ATOM_COUNTS["C"][2] == 97
    assert PG1424_RELAXED_STRUCTURE_ATOM_COUNTS["O"][5] == 14
    assert PG1424_BEST_LINE_ATOM_COUNTS["O"][5] == 14


def test_pg1424_trace_scope_records_the_accepted_population_treatment() -> None:
    assert PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS == (
        "F",
        "Ne",
        "Si",
        "S",
        "Ar",
        "Fe",
    )
    assert "P" in PG1424_PUBLISHED_MASS_FRACTIONS
    assert "P" not in PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS
