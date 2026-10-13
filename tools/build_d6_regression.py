"""Offline, explicit construction of a new frozen J1637 warm-regression reference.

Tests never call this builder. Existing outputs are never overwritten.
The source must be a certified standard-resolution J1637 cold-start atmosphere,
or an explicitly selected, checksum-verified frozen capsule of that atmosphere.
Its archived certificate documents the seed, not current-code convergence.
Current equations generate the bounded perturbed trajectory and fixed spectrum.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source(directory, config):
    """Read a checksum-verified seed capsule, preserving its archived provenance."""
    import numpy as np

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seed_path = directory.with_name(directory.name + "-seed.npz")
    reference_path = directory / "j1637.npz"
    if (manifest["schema"] != 1
            or manifest["config"] != json.loads(json.dumps(config))
            or manifest["source_request"]["config"] != manifest["config"]
            or digest(seed_path) != manifest["seed_sha256"]
            or digest(reference_path) != manifest["output_sha256"]):
        raise ValueError("frozen source integrity or configuration mismatch")
    certificate = manifest["source_cold_certificate"]
    if not certificate["verified"] or certificate["failures"]:
        raise ValueError("frozen seed lacks an archived verified cold certificate")
    required = {"all_depth_flux", "local_energy", "temperature_stationarity",
                "source_closure", "boundary_screening"}
    if set(certificate["required_checks"]) != required:
        raise ValueError("frozen seed must retain all required archived certificate checks")
    for key in certificate["required_checks"]:
        check = certificate["checks"][key]
        if not (check["measured"] and check["passed"]
                and 0 <= check["value"] < check["tolerance"]):
            raise ValueError("frozen seed has an invalid archived certificate check")
    with np.load(seed_path, allow_pickle=False) as seed:
        with np.load(reference_path, allow_pickle=False) as reference:
            for key in seed.files:
                np.testing.assert_array_equal(seed[key], reference[key])
        arrays = {key: seed[key].copy() for key in seed.files}
    for key in ("column_mass", "optical_depth", "base_temperature", "initial_temperature"):
        if (arrays[key].shape != (48,) or not np.all(np.isfinite(arrays[key]))
                or not np.all(arrays[key] > 0)):
            raise ValueError("frozen seed must contain positive finite 48-layer arrays")
    perturbation = 0.015 * np.exp(-((np.log10(arrays["optical_depth"]) + 2) / 1)**2)
    np.testing.assert_allclose(arrays["initial_temperature"],
        arrays["base_temperature"] * np.exp(perturbation), rtol=1e-14, atol=0)
    archive = dict(name=directory.name, manifest_sha256=digest(manifest_path),
        output_sha256=manifest["output_sha256"], seed_sha256=manifest["seed_sha256"],
        reference_commit=manifest["reference_commit"])
    return arrays, manifest, archive


def trajectory_difference(records, reference):
    import numpy as np
    from d6_regression_support import ARRAYS, METRICS

    profiles = {}
    for key in ARRAYS:
        actual = np.stack([record[key] for record in records])
        relative = np.abs(actual / reference[key] - 1)
        profiles[key] = dict(maximum_absolute=float(np.max(np.abs(actual - reference[key]))),
            maximum_relative=float(np.max(relative)),
            maximum_relative_by_step=np.max(relative, axis=1).tolist(),
            maximum_relative_location=list(map(int, np.unravel_index(np.argmax(relative), relative.shape))))
    metrics = np.stack([record["metrics"] for record in records])
    return dict(profiles=profiles, metrics=list(METRICS),
                previous_metrics=reference["metrics"].tolist(), current_metrics=metrics.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-frozen-reference", action="store_true",
        help="Reuse a checksum-verified versioned seed and its historical cold certificate")
    parser.add_argument("--verify-legacy-uv-controls", action="store_true",
        help="First reproduce the frozen pre-Fe-supplement trajectory at unchanged tolerances")
    args = parser.parse_args()
    if args.verify_legacy_uv_controls and not args.from_frozen_reference:
        parser.error("legacy verification requires --from-frozen-reference")
    seed_path = args.output.with_name(args.output.name + "-seed.npz")
    if args.output.exists() or seed_path.exists():
        parser.error("output already exists; choose a new versioned directory")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENWD_NUM_THREADS",
                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"

    import numpy as np
    from wd_spectra import D6Config
    from wd_spectra.models.common import (AtmosphereConvergenceWarning,
                                         ModelData, model_request_fingerprint)
    from wd_spectra.models.d6 import default_d6_wavelength_grid
    from d6_regression_support import (ARRAYS, METRICS, STEPS, atomic_input_scope,
        assert_trajectory_matches, fixed_spectrum, trajectory)

    config = D6Config()
    archived_reference = None
    if args.from_frozen_reference:
        arrays, source_manifest, archived_reference = frozen_source(args.source, asdict(config))
        column_mass = arrays["column_mass"]
        optical_depth = arrays["optical_depth"]
        temperature = arrays["base_temperature"]
        certificate = source_manifest["source_cold_certificate"]
        source_request = source_manifest["source_request"]
        source_name = source_manifest["source_name"]
        source_sha256 = source_manifest["source_sha256"]
    else:
        with np.load(args.source, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["atmosphere_metadata_json"]))
            certificate = metadata["equilibrium_certificate"]
            assert certificate["verified"] and not certificate["failures"]
            assert metadata["initial_temperature_was_supplied"] is False
            assert metadata["radiative_equilibrium_selected_metal_lines"] == 25_000
            assert metadata["metal_abundances"] == dict(config.abundances)
            assert float(saved["effective_temperature"]) == config.effective_temperature
            assert float(saved["logg"]) == config.logg
            column_mass = saved["column_mass"].copy()
            optical_depth = saved["rosseland_optical_depth"].copy()
            temperature = saved["temperature"].copy()
            assert temperature.shape == (48,)
        source_request = metadata["model_request_fingerprint"]
        source_name = f"{args.source.parent.name}/{args.source.name}"
        source_sha256 = digest(args.source)

    fingerprint = model_request_fingerprint("D6", config, ModelData.default())
    assert source_request["config"] == json.loads(json.dumps(asdict(config)))

    # A smooth 1.5% perturbation in the line-forming layers exercises real
    # thermal corrections instead of an already-converged zero-step resume.
    perturbation = 0.015 * np.exp(-((np.log10(optical_depth) + 2.0) / 1.0)**2)
    initial_temperature = (arrays["initial_temperature"].copy()
        if args.from_frozen_reference else temperature * np.exp(perturbation))
    legacy_recovery = None
    if args.verify_legacy_uv_controls:
        print("Verifying the immutable pre-Fe-supplement warm trajectory", flush=True)
        started = time.monotonic()
        with atomic_input_scope(legacy=True):
            legacy_records = trajectory(column_mass, optical_depth, initial_temperature)
        with np.load(args.source / "j1637.npz", allow_pickle=False) as previous:
            assert_trajectory_matches(legacy_records, previous)
            legacy_recovery = trajectory_difference(legacy_records, previous)
        legacy_recovery.update(passed=True, tolerances_changed=False,
            reference_output_sha256=archived_reference["output_sha256"],
            elapsed_seconds=time.monotonic() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name(args.output.name + "-legacy-recovery.json").write_text(
            json.dumps(legacy_recovery, indent=2) + "\n")
        print(json.dumps(dict(legacy_recovery=legacy_recovery)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_path, column_mass=column_mass, optical_depth=optical_depth,
        base_temperature=temperature, initial_temperature=initial_temperature)
    print(f"Building {STEPS} perturbed warm updates with full-resolution physics", flush=True)
    started = time.monotonic()
    with atomic_input_scope():
        records = trajectory(column_mass, optical_depth, initial_temperature)
    trajectory_seconds = time.monotonic() - started
    assert len(records) == STEPS
    for record in records:
        for key in ARRAYS:
            assert record[key].shape == (48,)
            assert np.all(np.isfinite(record[key])) and np.all(record[key] > 0)
        assert np.all(np.isfinite(record["metrics"])) and np.all(record["metrics"] >= 0)
    assert any(np.max(np.abs(record["temperature"] / initial_temperature - 1.0)) > 1e-4
               for record in records)
    # Preserve expensive trajectory evidence if the subsequent spectrum fails.
    args.output.mkdir(parents=True)
    partial = args.output / "trajectory.partial.npz"
    np.savez_compressed(partial, column_mass=column_mass, optical_depth=optical_depth,
        base_temperature=temperature, initial_temperature=initial_temperature,
        iterations=np.array([r["iteration"] for r in records]),
        phases=np.array([r["phase"] for r in records]),
        metrics=np.stack([r["metrics"] for r in records]),
        **{key: np.stack([r[key] for r in records]) for key in ARRAYS})
    wavelength = default_d6_wavelength_grid(config.quality)
    started = time.monotonic()
    with atomic_input_scope(), warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        spectrum = fixed_spectrum(column_mass, optical_depth, temperature, wavelength)
    spectrum_seconds = time.monotonic() - started
    assert spectrum.metadata["source_converged"]
    assert spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10

    path = args.output / "j1637.npz"
    np.savez_compressed(path, column_mass=column_mass, optical_depth=optical_depth,
        base_temperature=temperature, initial_temperature=initial_temperature,
        iterations=np.array([r["iteration"] for r in records]),
        phases=np.array([r["phase"] for r in records]),
        metrics=np.stack([r["metrics"] for r in records]),
        **{key: np.stack([r[key] for r in records]) for key in ARRAYS},
        wavelength=wavelength, surface_flux=spectrum.surface_flux_lambda)
    manifest = dict(schema=1, scope="bounded warm trajectory and fixed-atmosphere spectrum",
        cold_start_qualification=False, steps=STEPS, config=asdict(config),
        source_name=source_name, source_sha256=source_sha256,
        source_cold_certificate=certificate,
        source_request={key: source_request[key] for key in
                        ("physics_revision", "code_sha256", "data_sha256", "config")},
        reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physics_revision=fingerprint["physics_revision"],
        data_sha256=fingerprint["data_sha256"], code_sha256=fingerprint["code_sha256"],
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        perturbation="T * exp(0.015 * exp(-((log10(tau) + 2) / 1)**2))",
        metrics=list(METRICS), output_sha256=digest(path), seed_sha256=digest(seed_path),
        timings_seconds=dict(trajectory=trajectory_seconds, fixed_spectrum=spectrum_seconds))
    if archived_reference is not None:
        manifest["archived_reference"] = archived_reference
        with np.load(args.source / "j1637.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal([r["iteration"] for r in records], previous["iterations"])
            np.testing.assert_array_equal([r["phase"] for r in records], previous["phases"])
            manifest["difference_from_previous"] = trajectory_difference(records, previous)
    if legacy_recovery is not None:
        manifest["legacy_recovery"] = legacy_recovery
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial.unlink()
    print(json.dumps(manifest["timings_seconds"]), flush=True)


if __name__ == "__main__":
    main()
