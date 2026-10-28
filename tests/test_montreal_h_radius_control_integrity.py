"""Versioned hydrogen-neutral-default controls retain their inputs and predecessors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "tests/data/approved_regressions/montreal_h_radius_2026_10_27"
MANIFEST = json.loads((DIRECTORY / "manifest.json").read_text())
DAH_KEYS = json.loads((ROOT / "tests/data/dah_paper/manifest.json").read_text())["targets"]
EXPECTED = (
    {f"fixed/{case}" for case in (
        "da-3000", "da-4000", "da-5000", "da-20000", "dab-20000", "dz-pg1225", "dz-j0738")}
    | {"daz/daz-g149_28", "da_cubic/da-12000-public"}
    | {f"{preset}/{key}" for preset in ("dah_default", "dah_paper") for key in DAH_KEYS}
)
# The predecessor recovery bound of each consuming test, in significant flux.
RECOVERY_BOUND = {"fixed": 2e-6, "daz": 1e-3, "da_cubic": 1e-3,
                  "dah_default": 2e-8, "dah_paper": 2e-8}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_covers_every_changed_control():
    assert set(MANIFEST["cases"]) == EXPECTED


@pytest.mark.parametrize("label", sorted(EXPECTED))
def test_control_retains_its_input_and_predecessor(label):
    record = MANIFEST["cases"][label]
    group = label.split("/")[0]
    fixture = DIRECTORY / (label + ".npz")
    assert digest(fixture) == record["fixture_sha256"]
    assert digest(ROOT / record["structure_input"]) == record["structure_sha256"]
    assert digest(ROOT / record["previous_control"]) == record["previous_control_sha256"]
    assert record["tolerances_changed"] is False
    assert record["atmosphere_reconverged"] is False
    assert record["cold_start_qualification"] is False
    assert (
        record["previous_recovery"]["maximum_significant_relative_change"]
        <= RECOVERY_BOUND[group]
    )
    with np.load(fixture, allow_pickle=False) as saved:
        assert np.all(np.isfinite(saved["surface_flux"]))
        assert np.all(saved["surface_flux"] > 0.0)


COLD = json.loads((DIRECTORY / "cold/manifest.json").read_text())


@pytest.mark.parametrize("case", sorted(COLD["cases"]))
def test_cold_reference_is_a_certified_cold_start_with_its_predecessor(case):
    record = COLD["cases"][case]
    assert digest(DIRECTORY / "cold" / (case + ".npz")) == record["output_sha256"]
    assert digest(ROOT / record["previous_cold_reference"]) == (
        record["previous_cold_reference_sha256"]
    )
    assert record["source_equilibrium_certificate"]["verified"] is True
    assert record["source_public_cold_start"] is True
    assert record["tolerances_changed"] is False
