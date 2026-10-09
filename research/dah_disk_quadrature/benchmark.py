"""Bounded, fixed-atmosphere DAH disk diagnostics; no new regression controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
KEYS = ("j1018+0111", "j1351+5419")
TAGS = ("baseline", "drift-32", "drift-16", "drift-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(output):
    from wd_spectra.constants import LIGHT_SPEED
    from wd_spectra.validation.magnetic_da import load_dah_observation
    from wd_spectra.validation.observed import convolve_variable_log_gaussian

    report = {
        "scope": "archived fixed-atmosphere disk-quadrature diagnostics; not equilibrium qualification",
        "definition": "(convolved F_nu / 8-A convolved F_nu - 1); no per-variant flux rescaling",
        "window_vacuum_angstrom": [3820.0, 6950.0],
        "objects": {},
    }
    for key in KEYS:
        directory = DATA / key
        observation = load_dah_observation(key)
        records = {tag: json.loads((directory / (tag + ".json")).read_text()) for tag in TAGS}
        velocity = records["baseline"]["score"]["velocity_kms"]
        rest = observation.wavelength_angstrom / (1.0 + velocity * 1e5 / LIGHT_SPEED)
        selected = (rest >= 3820.0) & (rest <= 6950.0)
        rest = rest[selected]
        convolved = {}
        hashes = {}
        for tag in TAGS:
            path = directory / (tag + ".npz")
            with np.load(path) as saved:
                wave, flux = saved["wavelength"], saved["flux"]
            convolved[tag] = convolve_variable_log_gaussian(
                wave, flux * wave**2 * 1e-8 / LIGHT_SPEED,
                rest, observation.sigma_log_wavelength[selected],
            )
            hashes[tag] = {"spectrum": digest(path), "run_record": digest(directory / (tag + ".json"))}
        reference = convolved["drift-8"]
        mask = (rest >= 3820.0) & (rest <= 6950.0) & np.isfinite(reference) & (reference > 0.0)
        if not np.any(mask):
            raise ValueError("comparison has no valid wavelengths")
        rows = {}
        for tag in TAGS:
            error = convolved[tag][mask] / reference[mask] - 1.0
            if np.any(~np.isfinite(error)):
                raise ValueError("nonfinite comparison flux")
            rows[tag] = {
                "maximum_relative_difference": float(np.max(np.abs(error))),
                "rms_relative_difference": float(np.sqrt(np.mean(error**2))),
                "surface_cells": records[tag]["surface_cells"],
                "original_synthesis_seconds": records[tag]["synthesis_seconds"],
                "sha256": hashes[tag],
            }
        report["objects"][key] = {"velocity_kms": velocity, "comparison_pixels": int(mask.sum()), "runs": rows}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def synthesize(key, drift, output):
    from wd_spectra import DAHConfig, compute_dah
    from wd_spectra.models.common import load_atmosphere_checkpoint

    manifest_path = ROOT / "tests/data/dah_paper/manifest.json"
    target = json.loads(manifest_path.read_text())["targets"][key]
    names = ("effective_temperature", "logg", "magnetic_field_megagauss", "quality",
             "field_geometry", "dipole_inclination_deg", "dipole_offset_radius")
    values = {name: target["config"][name] for name in names}
    values["dipole_offset_radius"] = tuple(values["dipole_offset_radius"])
    config = replace(DAHConfig(**values), disk_component_drift_angstrom=drift)
    path = ROOT / "tests/data/dah_paper" / target["reference_file"]
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, "hydrogen",
        include_molecules=False, include_negative_hydrogen=False,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    with np.load(path) as saved:
        wavelength = saved["wavelength"]
    start = time.perf_counter()
    # Preserve the provenance warning: this is deliberately not a cold start.
    result = compute_dah(config, wavelength, initial_atmosphere=atmosphere, relax_atmosphere=False)
    seconds = time.perf_counter() - start
    np.savez_compressed(output / "spectrum.npz", wavelength=wavelength,
                        flux=result.spectrum.surface_flux_lambda)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "key": key, "config": asdict(config), "synthesis_seconds": seconds,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "surface_cells": result.spectrum.metadata["surface_cells"],
        "atmosphere_input_sha256": digest(path), "python": platform.python_version(),
        "source_sha256": {name: digest(ROOT / name) for name in
                          ("src/wd_spectra/magnetic.py", "src/wd_spectra/models/dah.py")},
        "scope": "saved-atmosphere diagnostic; no cold-start or magnetic-equilibrium claim",
    }
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--output", type=Path, required=True)
    synthesis = sub.add_parser("synthesize")
    synthesis.add_argument("key", choices=KEYS)
    synthesis.add_argument("--drift", type=float)
    synthesis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "compare":
        compare(args.output)
    else:
        synthesize(args.key, args.drift, args.output)


if __name__ == "__main__":
    main()
