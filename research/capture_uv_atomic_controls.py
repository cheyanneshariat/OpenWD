"""Capture Fe II missing-transition and trace-H metal-width fixed controls.

Every saved atmosphere, configuration, wavelength grid and tolerance stays
unchanged. First prove that scoped legacy atomic opacity recovers the preceding
DZ microfield-host and DAZ Si-Kurucz controls. Then capture the new defaults.
No atmosphere is re-relaxed; these are not cold-start qualifications. The
script refuses existing output directories and is never run by tests or CI.

usage: PYTHONPATH=src python research/capture_uv_atomic_controls.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENWD_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy

from wd_spectra import _rt
from wd_spectra._compat import trapezoid
from wd_spectra.metals import KURUCZ_STARK_TABLE_RESOURCE, SI_II_HE_UNIFIED_PROFILE_RESOURCE
from uv_atomic_control_legacy import legacy_uv_atomic_controls
from wd_spectra.models import (
    AtmosphereConvergenceWarning,
    DAZConfig,
    DZConfig,
    compute_daz,
    compute_dz,
    load_atmosphere_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
OUT = DATA / "approved_regressions/uv_atomic_2026_10_12"
CASES = {
    "dz-pg1225": ("DZ", DATA / "spectral_regressions/dz-pg1225.npz",
                  DATA / "approved_regressions/microfield_hosts_2026_10_12/fixed/dz-pg1225.npz"),
    "dz-j0738": ("DZ", DATA / "spectral_regressions/dz-j0738.npz",
                 DATA / "approved_regressions/microfield_hosts_2026_10_12/fixed/dz-j0738.npz"),
    "daz-g149_28": ("DAZ", DATA / "daz_regressions/g149_28.npz",
                    DATA / "approved_regressions/si_kurucz_2026_10_12/fixed/daz-g149_28.npz"),
    "daz-galex1931": ("DAZ", DATA / "daz_regressions/galex1931.npz",
                      DATA / "approved_regressions/si_kurucz_2026_10_12/fixed/daz-galex1931.npz"),
}
SOURCES = (
    "src/wd_spectra/metals.py",
    "src/wd_spectra/models/stellar.py",
    "src/wd_spectra/models/daz.py",
    "src/wd_spectra/models/common.py",
    "src/wd_spectra/" + KURUCZ_STARK_TABLE_RESOURCE,
    "src/wd_spectra/" + SI_II_HE_UNIFIED_PROFILE_RESOURCE,
    "src/wd_spectra/data/atomic/gf1401_observed.all.xz",
    "src/wd_spectra/data/atomic/gf2601_uv_missing.all.xz",
    "src/wd_spectra/data/atomic/gf2601_uv_missing.provenance.json",
    "research/uv_atomic_control_legacy.py",
    "research/capture_uv_atomic_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(kind, path, config, wave):
    if kind == "DZ":
        atmosphere = load_atmosphere_checkpoint(
            path, config.effective_temperature, config.logg, "helium",
            include_molecules=False, include_negative_hydrogen=False,
            log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
            trihydrogen_ion_partition_model=None,
        )
        compute = compute_dz
    else:
        molecules = config.effective_temperature <= 12000
        atmosphere = load_atmosphere_checkpoint(
            path, config.effective_temperature, config.logg, "hydrogen",
            include_molecules=molecules, include_negative_hydrogen=molecules,
            trihydrogen_ion_partition_model="neale-tennyson-1995",
        )
        compute = compute_daz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        return compute(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)


def main() -> None:
    assert all(hasattr(_rt, name) for name in (
        "frequency_voigt_profile", "expand_metal_line_blocks", "metal_line_profile_means",
    )), "Build the current C backend before capturing controls"
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "fixed").mkdir()
    records = {}
    for case, (kind, path, previous_path) in CASES.items():
        with np.load(path) as saved:
            config = (DZConfig if kind == "DZ" else DAZConfig)(
                **json.loads(str(saved["config_json"]))
            )
            wave = saved["wavelength"]
        with np.load(previous_path) as previous:
            np.testing.assert_array_equal(wave, previous["wavelength"])
            old = previous["surface_flux"]
        # Disable only the two new atomic-opacity changes, preserving EOS.
        with legacy_uv_atomic_controls():
            off = synthesize(kind, path, config, wave)
        off_flux = off.spectrum.surface_flux_lambda
        assert np.all(np.isfinite(off_flux)) and np.all(off_flux > 0)
        assert off.spectrum.metadata["source_converged"]
        if kind == "DZ":
            np.testing.assert_allclose(off_flux, old, rtol=2e-6, atol=1e-12 * np.max(old))
        else:
            significant = wave * old > 0.01 * np.max(wave * old)
            np.testing.assert_allclose(off_flux[significant], old[significant], rtol=1e-3, atol=0)
        result = synthesize(kind, path, config, wave)
        new = result.spectrum.surface_flux_lambda
        assert np.all(np.isfinite(new)) and np.all(new > 0)
        assert result.spectrum.metadata["source_converged"]
        output = OUT / "fixed" / (case + ".npz")
        np.savez_compressed(output, wavelength=wave, surface_flux=new)
        bands = {}
        for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
            take = (wave >= lo) & (wave <= hi)
            if np.count_nonzero(take) >= 2:
                bands[f"{lo}-{hi}"] = float(
                    trapezoid(new[take] - old[take], wave[take]) / trapezoid(old[take], wave[take])
                )
        records[case] = dict(
            family=kind,
            historical_config_json_sha256=hashlib.sha256(json.dumps(asdict(config), sort_keys=True, default=str).encode()).hexdigest(),
            wavelength_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            saved_structure_unchanged=True,
            atmosphere_reconverged=False,
            regression_rtol=2e-6 if kind == "DZ" else 1e-3,
            regression_atol=1e-12 * float(np.max(old)) if kind == "DZ" else 0.0,
            input_sha256=sha256(path),
            previous_control=str(previous_path.relative_to(ROOT)),
            previous_control_sha256=sha256(previous_path),
            output_sha256=sha256(output),
            legacy_source_converged=bool(off.spectrum.metadata["source_converged"]),
            legacy_atomic_opacity_recovers_previous_control=True,
            legacy_atomic_opacity_max_fractional_difference=float(np.max(np.abs(off_flux / old - 1.0))),
            max_fractional_change=float(np.max(np.abs(new / old - 1.0))),
            band_fractional_changes=bands,
            source_converged=bool(result.spectrum.metadata["source_converged"]),
            atomic_lines=result.metadata["atomic_lines"],
            spectrum_metadata=result.spectrum.metadata,
            cold_start_qualification=False,
            config=json.loads(json.dumps(asdict(config), default=str)),
        )
        print(case, json.dumps({k: records[case][k] for k in (
            "max_fractional_change", "band_fractional_changes",
            "legacy_atomic_opacity_max_fractional_difference")}), flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    manifest = dict(
        base_commit=commit,
        description=("Default missing Fe II UV transitions and neutral trace-H metal-line widths; "
                     "fixed historical atmospheres only; all preceding controls retained"),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        cases=records,
    )
    manifest["native_backend_sha256"] = sha256(Path(_rt.__file__))
    from wd_spectra.models.common import _jsonable
    (OUT / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(
        "# UV atomic fixed-atmosphere controls (2026-10-12)\n\n"
        "Missing Fe II ultraviolet transitions and neutral trace-H metal-line widths. "
        "Historical structures, configurations, grids and regression tolerances are unchanged. "
        "Disabling only these two changes under the scoped legacy helper recovers the "
        "previous DZ microfield-host and DAZ Si-Kurucz controls before capture. "
        "Earlier controls remain intact. The DAZ grids are optical; their Fe II "
        "partition inputs include the single added upper level.\n\n"
        "The manifest records source/input/output hashes, numerical versions, "
        "configuration, spectrum diagnostics and the legacy-control proof. "
        "These spectra are not a cold-start or atmosphere-equilibrium qualification. "
        "Capture uses research/capture_uv_atomic_controls.py and refuses an existing directory.\n"
    )


if __name__ == "__main__":
    sys.exit(main())
