"""Offline capture of cold-start references after the 2026-10-27 hydrogen defaults.

The Montreal neutral-hydrogen radius enters the structure solve, so fresh cold
starts move by a few 1e-4 in pressure. Each replacement is a completed public
cold start from the protected canary (OPENWD_TEST_ARTIFACTS), which
``capture_release_controls.capture`` verifies (equilibrium certificate, gray
public initialization, no checkpoint seed) and re-synthesizes on the canary
wavelength grid. The previous cold references are untouched; the manifest
records their hashes and the measured change. Never run by tests or CI.

Usage: PYTHONPATH=src python research/capture_montreal_h_radius_cold_controls.py \
    OUTPUT SOURCES.json   (SOURCES maps case -> completed cold atmosphere.npz)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from capture_release_controls import capture, digest, source_identity  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.constants import STEFAN_BOLTZMANN  # noqa: E402

PREVIOUS = ROOT / "tests/data/approved_regressions/cold"


def change(new_path, old_path, effective_temperature):
    with np.load(new_path) as new, np.load(old_path) as old:
        structure = {
            name: float(np.max(np.abs(new[name][: len(old[name])] / old[name] - 1.0)))
            for name in ("gas_pressure", "column_mass")
        }
        wave, i, j = np.intersect1d(new["wavelength"], old["wavelength"], return_indices=True)
        actual, expected = new["surface_flux"][i], old["surface_flux"][j]
    target = STEFAN_BOLTZMANN * effective_temperature**4
    return dict(
        maximum_structure_relative_change=structure,
        energy_scale_maximum_change=float(np.max(wave * abs(actual - expected))
                                          / np.max(wave * expected)),
        integrated_absolute_change=float(trapezoid(abs(actual - expected), wave) / target),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for case, source in sources.items():
        record = capture(case, "cold", args.output, {case: source})
        previous = PREVIOUS / (case + ".npz")
        record["previous_cold_reference"] = str(previous.relative_to(ROOT))
        record["previous_cold_reference_sha256"] = digest(previous)
        record["change_from_previous"] = change(
            args.output / (case + ".npz"), previous, record["config"]["effective_temperature"])
        record["tolerances_changed"] = False
        records[case] = record
    manifest = dict(
        description=__doc__, date="2026-10-27",
        reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                 text=True).strip(),
        source_identity=source_identity(), python=platform.python_version(),
        numpy=np.__version__, cases=records)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
