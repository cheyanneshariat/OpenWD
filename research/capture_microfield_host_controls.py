"""Capture the DZ fixed-atmosphere controls after the host-ion microfield fix.

metal_ionic_microfield_perturber_density now counts the host He+/He++ (and
other singly charged) ions, not only the metals. Each unchanged historical
DZ structure is synthesized on its saved wavelength grid with its saved
configuration. With the previous metal-only moment patched in, the result
must recover the previous approved control exactly before the new array is
written. DAZ controls do not change (no metal-level dissolution or
linear-Stark manifold in DAZ). The script refuses an existing output
directory and is never called by tests or CI.

usage: PYTHONPATH=src python research/capture_microfield_host_controls.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy

import wd_spectra.metals as metals_module

from wd_spectra._compat import trapezoid
from wd_spectra.metals import KURUCZ_STARK_TABLE_RESOURCE, SI_II_HE_UNIFIED_PROFILE_RESOURCE
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
OUT = DATA / "approved_regressions/microfield_hosts_2026_10_12"
CASES = {
    "dz-pg1225": ("DZ", DATA / "spectral_regressions/dz-pg1225.npz",
                  DATA / "approved_regressions/si_kurucz_2026_10_12/fixed/dz-pg1225.npz"),
    "dz-j0738": ("DZ", DATA / "spectral_regressions/dz-j0738.npz",
                 DATA / "approved_regressions/si_kurucz_2026_10_12/fixed/dz-j0738.npz"),
}


def legacy_metal_only_moment(state):
    """The previous moment: metal ions only."""
    result = np.zeros_like(state.electron_density, dtype=np.float64)
    for populations in state.ion_number_density.values():
        for charge in range(1, populations.shape[0]):
            result += charge**1.5 * populations[charge]
    return result


SOURCES = (
    "src/wd_spectra/metals.py",
    "src/wd_spectra/models/stellar.py",
    "src/wd_spectra/models/daz.py",
    "src/wd_spectra/models/common.py",
    "src/wd_spectra/" + KURUCZ_STARK_TABLE_RESOURCE,
    "src/wd_spectra/" + SI_II_HE_UNIFIED_PROFILE_RESOURCE,
    "src/wd_spectra/data/atomic/gf1401_observed.all.xz",
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
        # Disabling only the new option must recover the previous control.
        current = metals_module.metal_ionic_microfield_perturber_density
        metals_module.metal_ionic_microfield_perturber_density = legacy_metal_only_moment
        try:
            off = synthesize(kind, path, config, wave)
        finally:
            metals_module.metal_ionic_microfield_perturber_density = current
        off_flux = off.spectrum.surface_flux_lambda
        if kind == "DZ":
            np.testing.assert_allclose(off_flux, old, rtol=2e-6, atol=1e-12 * np.max(old))
        else:
            significant = wave * old > 0.01 * np.max(wave * old)
            np.testing.assert_allclose(off_flux[significant], old[significant], rtol=1e-3, atol=0)
        result = synthesize(kind, path, config, wave)
        new = result.spectrum.surface_flux_lambda
        assert np.all(np.isfinite(new)) and np.all(new >= 0)
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
            input_sha256=sha256(path),
            previous_control=str(previous_path.relative_to(ROOT)),
            previous_control_sha256=sha256(previous_path),
            output_sha256=sha256(output),
            legacy_moment_recovers_previous_control=True,
            legacy_moment_max_fractional_difference=float(np.max(np.abs(off_flux / old - 1.0))),
            max_fractional_change=float(np.max(np.abs(new / old - 1.0))),
            band_fractional_changes=bands,
            source_converged=bool(result.spectrum.metadata["source_converged"]),
            atomic_lines=result.metadata["atomic_lines"],
            cold_start_qualification=False,
            config=json.loads(json.dumps(asdict(config), default=str)),
        )
        print(case, json.dumps({k: records[case][k] for k in (
            "max_fractional_change", "band_fractional_changes",
            "legacy_moment_max_fractional_difference")}), flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    manifest = dict(
        base_commit=commit,
        description=("Host-ion microfield moment for metal-level dissolution (DZ); fixed historical "
                     "atmospheres only; previous controls retained"),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        cases=records,
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
