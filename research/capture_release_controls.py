"""Explicitly capture reviewed microfield/cubic regression outputs.

Never run by pytest or CI. Historical inputs are read-only; output directories
must be new. Fixed-state synthesis is not an equilibrium certificate. A separate
set of reference structures checks fresh cold runs under unchanged tolerances.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from wd_spectra import DAConfig, DAZConfig, DBConfig, DABConfig, DZConfig
from wd_spectra import compute_da, compute_daz, compute_db, compute_dab, compute_dz
from wd_spectra.models import load_atmosphere_checkpoint, AtmosphereConvergenceWarning
from wd_spectra.models.common import _MODEL_PHYSICS_REVISION
from wd_spectra._compat import trapezoid

ROOT = Path(__file__).resolve().parents[1]
FIXED = ("da-3000", "da-4000", "da-5000", "da-20000", "db-10000",
         "db-22000", "dab-9000", "dab-20000", "dz-pg1225", "dz-j0738",
         "daz-g149_28", "daz-galex1931")
COLD = ("da-3000", "da-4000", "da-5000", "da-20000", "db-10000",
        "db-22000", "db-22000-standard", "dab-20000")
MODELS = {
    "DA": (DAConfig, compute_da, "hydrogen"),
    "DAZ": (DAZConfig, compute_daz, "hydrogen"),
    "DB": (DBConfig, compute_db, "helium"),
    "DAB": (DABConfig, compute_dab, "mixed"),
    "DZ": (DZConfig, compute_dz, "helium"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in ("src", "csrc") for p in sorted((ROOT / folder).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def capture(case, mode, output, cold_sources):
    print(f"[{mode}/{case}] synthesizing reviewed default", flush=True)
    kind = case.split("-")[0].upper()
    cls, compute, composition = MODELS[kind]
    historical = ROOT / "tests/data" / (
        "daz_regressions" if kind == "DAZ" else "spectral_regressions"
    ) / ((case[4:] if kind == "DAZ" else case) + ".npz")
    with np.load(historical) as saved:
        config = cls(**json.loads(str(saved["config_json"])))
        wavelength = saved["wavelength"].copy()
        old_flux = saved["original_surface_flux"].copy()
    source = Path(cold_sources[case]) if mode == "cold" else historical
    record = dict(case=case, mode=mode, physics_revision=_MODEL_PHYSICS_REVISION,
                  historical_input=str(historical.relative_to(ROOT)),
                  historical_sha256=digest(historical),
                  structure_input=str(source), structure_sha256=digest(source),
                  fixed_state_only=True, fresh_equilibrium_not_claimed=True,
                  config=asdict(config))
    if source != historical:
        metadata_file = source.with_name("metadata.json")
        saved_metadata = json.loads(metadata_file.read_text())
        metadata = saved_metadata["atmosphere_metadata"]
        model_metadata = saved_metadata["model_metadata"]
        certificate = metadata["equilibrium_certificate"]
        # A same-run bottom extension passes its own preceding state internally.
        # Its public cold-start flag remains authoritative; this is not a seed
        # from another stellar model. DA records its gray public initialization.
        public_cold = metadata.get("cold_start") is True or (
            model_metadata.get("atmosphere_initialization") == "gray"
            and metadata.get("initial_temperature_was_supplied") is False
        )
        if (not certificate["verified"] or not public_cold
                or model_metadata.get("checkpoint_matches_model_request") is True):
            raise ValueError(f"{case}: replacement structure must be a verified cold start")
        record["source_equilibrium_certificate"] = certificate
        record["source_public_cold_start"] = public_cold
        record["source_metadata_sha256"] = digest(metadata_file)
    if mode == "cold":
        wavelength = np.unique(np.r_[np.geomspace(900., 300000., 1100),
                                     np.arange(3700., 7000., 2.)])
    molecular = kind in {"DA", "DAZ"} and config.effective_temperature <= 12000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        atmosphere = load_atmosphere_checkpoint(
            source, config.effective_temperature, config.logg, composition,
            include_molecules=molecular, include_negative_hydrogen=molecular,
            log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
            trihydrogen_ion_partition_model=(
                "neale-tennyson-1995" if kind in {"DA", "DAZ"} else None),
        )
        # Deliberately no transfer override: capture the actual public default.
        result = compute(config, wavelength, initial_atmosphere=atmosphere,
                         relax_atmosphere=False)
    spectrum = result.spectrum
    assert spectrum.metadata["source_converged"]
    assert spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10
    assert spectrum.metadata["transfer_discretization"] == (
        "formal-pchip" if kind == "DA" else "formal-linear")
    record["transfer_discretization"] = spectrum.metadata["transfer_discretization"]
    record["source_error"] = spectrum.metadata["independent_radiation_scaled_source_error"]
    flux = spectrum.surface_flux_lambda
    if mode == "fixed":
        important = wavelength * old_flux > .01 * np.max(wavelength * old_flux)
        record["max_significant_change_from_historical"] = float(
            np.max(abs(flux[important] / old_flux[important] - 1)))
        record["integrated_absolute_change_from_historical"] = float(
            trapezoid(abs(flux-old_flux), wavelength) / trapezoid(old_flux, wavelength))
    path = output / (case + ".npz")
    np.savez_compressed(path, wavelength=wavelength, surface_flux=flux,
                        gas_pressure=result.atmosphere.gas_pressure,
                        column_mass=result.atmosphere.column_mass)
    record["output_sha256"] = digest(path)
    print(f"[{mode}/{case}] captured; source error={record['source_error']:.3g}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixed", "cold"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cold-sources", type=Path,
                        help="JSON mapping case to a completed corrected cold atmosphere.npz")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    sources = {} if args.cold_sources is None else json.loads(args.cold_sources.read_text())
    if args.mode == "cold" and set(sources) != set(COLD):
        parser.error("--cold-sources must explicitly name every corrected cold control")
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_identity()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(capture, c, args.mode, args.output, sources)
                   for c in (FIXED if args.mode == "fixed" else COLD)]
        records = [future.result() for future in as_completed(pending)]
    if source_identity() != before:
        raise RuntimeError("source changed during capture; outputs cannot be approved")
    (args.output / "manifest.json").write_text(json.dumps(dict(
        physics_revision=_MODEL_PHYSICS_REVISION, source_files=before,
        records=sorted(records, key=lambda r: r["case"])), indent=2) + "\n")


if __name__ == "__main__":
    main()
