"""Offline capture of fixed-atmosphere controls after the 2026-10-27 hydrogen defaults.

The public hydrogen neutral defaults changed to the Montreal Hummer--Mihalas
hard-sphere radius scale 0.5 and Ali--Griem self-broadening at every
temperature. Each changed control is synthesized on its frozen structure with
atmosphere iteration forbidden, twice: inside
``previous_hydrogen_neutral_defaults`` (structure EOS reloaded there) it must
first recover the previously approved spectrum at that test's own tolerance;
then the current default is recorded. Existing controls are never overwritten,
and tests/CI never run this script. No atmosphere is reconverged or certified.

Usage: PYTHONPATH=src python research/capture_montreal_h_radius_controls.py OUTPUT
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import numpy as np

from wd_spectra import compiled_backend_available, paper_dah_config
from wd_spectra._compat import trapezoid
from wd_spectra.models import (
    AtmosphereConvergenceWarning, DABConfig, DAConfig, DAHConfig, DAZConfig, DZConfig,
    compute_da, compute_dab, compute_dah, compute_daz, compute_dz, load_atmosphere_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from montreal_h_radius_control_legacy import previous_hydrogen_neutral_defaults  # noqa: E402

DATA = ROOT / "tests/data"
APPROVED = DATA / "approved_regressions"
FIXED_PREVIOUS = {
    "da-3000": APPROVED / "fixed",
    "da-4000": APPROVED / "stark_density_floor_2026_10_05/fixed",
    "da-5000": APPROVED / "stark_density_floor_2026_10_05/fixed",
    "da-20000": APPROVED / "fixed",
    "dab-20000": APPROVED / "fixed",
    "dz-pg1225": APPROVED / "metal_populations_2026_10_24/fixed",
    "dz-j0738": APPROVED / "metal_populations_2026_10_24/fixed",
}
DAZ_CASES = ("g149_28",)
DA_CUBIC_CASES = ("da-12000-public",)
BANDS = ((900, 1300), (1300, 3000), (3000, 3700), (3700, 4200), (4200, 5000),
         (5000, 7000), (7000, 25000))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(wavelength, actual, expected):
    relative = actual / expected - 1.0
    significant = wavelength * expected > 0.01 * np.max(wavelength * expected)
    index = int(np.argmax(np.abs(relative)))
    sig_index = int(np.argmax(np.abs(np.where(significant, relative, 0.0))))
    bands = {}
    for lo, hi in BANDS:
        take = (wavelength >= lo) & (wavelength < hi)
        if np.count_nonzero(take) > 1:
            bands[f"{lo}-{hi}"] = float(trapezoid((actual - expected)[take], wavelength[take])
                                        / trapezoid(expected[take], wavelength[take]))
    return dict(
        maximum_relative_change=float(np.max(np.abs(relative))),
        maximum_change_wavelength_angstrom=float(wavelength[index]),
        maximum_significant_relative_change=float(np.abs(relative[sig_index])),
        maximum_significant_change_wavelength_angstrom=float(wavelength[sig_index]),
        integrated_flux_change=float(trapezoid(actual - expected, wavelength)
                                     / trapezoid(expected, wavelength)),
        band_fractional_changes=bands,
    )


def synthesize(run, config, wavelength, load):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        result = run(config, wavelength, initial_atmosphere=load(), relax_atmosphere=False)
    flux = result.spectrum.surface_flux_lambda.copy()
    assert np.all(np.isfinite(flux)) and np.all(flux > 0.0)
    return flux


def fixed_cases():
    for case, previous_directory in FIXED_PREVIOUS.items():
        path = DATA / "spectral_regressions" / (case + ".npz")
        with np.load(path) as saved:
            kind = str(saved["spectral_type"])
            config = {"DA": DAConfig, "DAB": DABConfig, "DZ": DZConfig}[kind](
                **json.loads(str(saved["config_json"])))
            wavelength = saved["wavelength"].copy()
        molecular = kind == "DA" and config.effective_temperature <= 12000

        def load(path=path, config=config, kind=kind, molecular=molecular):
            return load_atmosphere_checkpoint(
                path, config.effective_temperature, config.logg,
                {"DA": "hydrogen", "DAB": "mixed", "DZ": "helium"}[kind],
                include_molecules=molecular, include_negative_hydrogen=molecular,
                log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
                trihydrogen_ion_partition_model="neale-tennyson-1995" if kind == "DA" else None)

        previous = previous_directory / (case + ".npz")
        with np.load(previous) as control:
            expected = control["surface_flux"].copy()
        run = {"DA": compute_da, "DAB": compute_dab, "DZ": compute_dz}[kind]

        def recovered(flux, expected=expected):
            np.testing.assert_allclose(flux, expected, rtol=2e-6, atol=1e-12 * np.max(expected))

        yield "fixed", case, path, previous, config, run, wavelength, load, expected, recovered


def daz_cases():
    for case in DAZ_CASES:
        path = DATA / "daz_regressions" / (case + ".npz")
        with np.load(path) as saved:
            config = DAZConfig(**json.loads(str(saved["config_json"])))
            wavelength = saved["wavelength"].copy()
        molecular = config.effective_temperature <= 12000

        def load(path=path, config=config, molecular=molecular):
            return load_atmosphere_checkpoint(
                path, config.effective_temperature, config.logg, "hydrogen",
                include_molecules=molecular, include_negative_hydrogen=molecular,
                trihydrogen_ion_partition_model="neale-tennyson-1995")

        previous = APPROVED / "independent_kurucz_2026_10_14/fixed" / ("daz-" + case + ".npz")
        with np.load(previous) as control:
            expected = control["surface_flux"].copy()

        def recovered(flux, expected=expected, wavelength=wavelength):
            significant = wavelength * expected > 0.01 * np.max(wavelength * expected)
            np.testing.assert_allclose(flux[significant], expected[significant], rtol=1e-3, atol=0)

        yield "daz", "daz-" + case, path, previous, config, compute_daz, wavelength, load, expected, recovered


def da_cubic_cases():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_da_public_spectrum import assert_spectrum_preserved

    for case in DA_CUBIC_CASES:
        path = DATA / "da_cubic_regressions" / (case + ".npz")
        with np.load(path) as saved:
            config = DAConfig(**json.loads(str(saved["config_json"])))
            wavelength = saved["wavelength"].copy()
            expected = saved["original_surface_flux"].copy()
        molecular = config.effective_temperature <= 12000

        def load(path=path, config=config, molecular=molecular):
            return load_atmosphere_checkpoint(
                path, config.effective_temperature, config.logg, "hydrogen",
                include_molecules=molecular, include_negative_hydrogen=molecular,
                trihydrogen_ion_partition_model="neale-tennyson-1995")

        def recovered(flux, expected=expected, wavelength=wavelength):
            assert_spectrum_preserved(wavelength, flux, expected)

        yield "da_cubic", case, path, path, config, compute_da, wavelength, load, expected, recovered


def dah_cases():
    targets = json.loads((DATA / "dah_paper/manifest.json").read_text())["targets"]
    names = ("effective_temperature", "logg", "magnetic_field_megagauss", "quality",
             "field_geometry", "dipole_inclination_deg", "dipole_offset_radius")
    for preset in ("dah_default", "dah_paper"):
        for key, record in targets.items():
            parameters = {name: record["config"][name] for name in names}
            parameters["dipole_offset_radius"] = tuple(parameters["dipole_offset_radius"])
            config = paper_dah_config(**parameters) if preset == "dah_paper" else DAHConfig(**parameters)
            path = DATA / "dah_paper" / record["reference_file"]
            if preset == "dah_paper":
                previous = path
                with np.load(path) as saved:
                    wavelength, expected = saved["wavelength"].copy(), saved["flux"].copy()
            else:
                previous = APPROVED / "dah_rwa_default_2026_10_25/fixed" / (key + ".npz")
                with np.load(previous) as saved:
                    wavelength, expected = saved["wavelength"].copy(), saved["surface_flux"].copy()

            def load(path=path, config=config):
                return load_atmosphere_checkpoint(
                    path, config.effective_temperature, config.logg, "hydrogen",
                    include_molecules=False, include_negative_hydrogen=False,
                    trihydrogen_ion_partition_model="neale-tennyson-1995")

            def recovered(flux, expected=expected):
                np.testing.assert_allclose(flux, expected, rtol=2e-8, atol=0.0)

            yield preset, key, path, previous, config, compute_dah, wavelength, load, expected, recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise SystemExit(f"{args.output} is already complete")
    assert compiled_backend_available(), "use the compiled backend for the saved controls"
    partial = args.output / "manifest.partial.json"
    cases = json.loads(partial.read_text()) if partial.exists() else {}
    for group in (fixed_cases, daz_cases, da_cubic_cases, dah_cases):
        for (subdir, case, structure, previous, config, run, wavelength, load,
             expected, recovered) in group():
            out = args.output / subdir / (case + ".npz")
            label = f"{subdir}/{case}"
            if label in cases and out.exists() and digest(out) == cases[label]["fixture_sha256"]:
                print(label, "already captured", flush=True)
                continue
            with previous_hydrogen_neutral_defaults():
                legacy = synthesize(run, config, wavelength, load)
            recovered(legacy)
            flux = synthesize(run, config, wavelength, load)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, wavelength=wavelength, surface_flux=flux)
            cases[label] = dict(
                structure_input=str(Path(structure).relative_to(ROOT)),
                structure_sha256=digest(structure),
                previous_control=str(Path(previous).relative_to(ROOT)),
                previous_control_sha256=digest(previous),
                fixture_sha256=digest(out),
                config=json.loads(json.dumps(asdict(config))),
                previous_recovery=difference(wavelength, legacy, expected),
                change_from_previous=difference(wavelength, flux, expected),
                atmosphere_reconverged=False, cold_start_qualification=False,
                tolerances_changed=False)
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(cases, indent=2) + "\n")
            change = cases[label]["change_from_previous"]
            print(label, "captured; max significant change",
                  f"{change['maximum_significant_relative_change']:.2e}",
                  f"at {change['maximum_significant_change_wavelength_angstrom']:.1f} A", flush=True)
    manifest = dict(
        description=__doc__, date="2026-10-27",
        reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p)
                       for p in sorted((ROOT / "src/wd_spectra").rglob("*.py"))},
        python=platform.python_version(), numpy=np.__version__, cases=cases)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial.unlink()


if __name__ == "__main__":
    main()
