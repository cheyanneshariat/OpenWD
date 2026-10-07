"""Offline, explicit construction of a new frozen J1637 warm-regression reference.

Tests never call this builder. Existing outputs are never overwritten.
The source must be a certified standard-resolution J1637 cold-start atmosphere.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
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
    from d6_regression_support import ARRAYS, METRICS, STEPS, fixed_spectrum, trajectory

    config = D6Config()
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

    fingerprint = model_request_fingerprint("D6", config, ModelData.default())
    source_request = metadata["model_request_fingerprint"]
    assert source_request["config"] == json.loads(json.dumps(asdict(config)))

    # A smooth 1.5% perturbation in the line-forming layers exercises real
    # thermal corrections instead of an already-converged zero-step resume.
    perturbation = 0.015 * np.exp(-((np.log10(optical_depth) + 2.0) / 1.0)**2)
    initial_temperature = temperature * np.exp(perturbation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_path, column_mass=column_mass, optical_depth=optical_depth,
        base_temperature=temperature, initial_temperature=initial_temperature)
    print(f"Building {STEPS} perturbed warm updates with full-resolution physics", flush=True)
    started = time.monotonic()
    records = trajectory(column_mass, optical_depth, initial_temperature)
    trajectory_seconds = time.monotonic() - started
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
    with warnings.catch_warnings():
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
        source_name=f"{args.source.parent.name}/{args.source.name}", source_sha256=digest(args.source),
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
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial.unlink()
    print(json.dumps(manifest["timings_seconds"]), flush=True)


if __name__ == "__main__":
    main()
