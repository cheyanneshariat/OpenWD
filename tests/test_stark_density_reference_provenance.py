"""Versioned density-edge outputs preserve earlier controls and structures."""

import hashlib
import json
from pathlib import Path

import numpy as np

from wd_spectra.models.common import (
    _MODEL_FAMILY_PHYSICS_REVISIONS,
    _MODEL_PHYSICS_REVISION,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "tests/data/approved_regressions/stark_density_floor_2026_10_05/fixed"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_density_edge_reference_provenance_and_preserved_old_controls():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    assert manifest["physics_revision"] == _MODEL_PHYSICS_REVISION
    assert manifest["density_policy_revision"] == _MODEL_FAMILY_PHYSICS_REVISIONS["DA"]
    assert manifest["density_policy_revision"] == _MODEL_FAMILY_PHYSICS_REVISIONS["DAB"]
    assert manifest["approval"]["scope"] == "three versioned fixed-atmosphere references"
    records = manifest["records"]
    assert {record["case"] for record in records} == {"da-4000", "da-5000", "dab-9000"}
    assert len(records) == 3
    expected_counts = {"da-4000": 19, "da-5000": 4, "dab-9000": 7}
    for record in records:
        frozen = DIRECTORY / (record["case"] + ".npz")
        previous = ROOT / record["previous_approved_file"]
        assert digest(frozen) == record["output_sha256"]
        assert digest(previous) == record["previous_approved_sha256"]
        assert digest(ROOT / record["historical_input"]) == record["historical_sha256"]
        assert record["fixed_state_only"]
        assert record["fresh_equilibrium_not_claimed"]
        assert record["source_error"] < 1e-10
        assert record["review"]["same_fixed_structure"]
        assert record["review"]["changed_guard_samples"] == expected_counts[record["case"]]
        assert record["review"]["max_optical_relative_change"] < 5.2e-9
        with np.load(frozen) as new, np.load(previous) as old:
            for key in ("wavelength", "gas_pressure", "column_mass"):
                np.testing.assert_array_equal(new[key], old[key])
            assert np.all(np.isfinite(new["surface_flux"]))
            assert np.all(new["surface_flux"] > 0)
