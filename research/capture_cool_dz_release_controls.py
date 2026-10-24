"""Re-approve the fixed-spectrum controls changed by the cool-DZ release.

DZ controls (dz-pg1225, dz-j0738): every saved atmosphere, configuration and
wavelength grid stays unchanged; DZConfig now defaults to the validated
dense-helium physics (REOS3, Rogers-Young and partial-wave He- free-free,
computed Mg I b / Mg II / Ca II / Na I D + He tables), and the dense-He metal
populations are corrected (Rydberg-only neutral hard sphere).  New spectra go
to a new dated directory; earlier approved directories are not modified.

DQ controls (J1235 historical, J1311 completed default): only the flux arrays
are recomputed, on the unchanged stored structures, as on the 2026-10-01
re-approval.  The change comes from the smooth (C1) He-REOS.3 interpolation
between isotherms that DQ models share.  The previous hash is appended to each
manifest's ``reapprovals``.

No atmosphere is re-relaxed; these are not cold-start qualifications.  The
script refuses an existing DZ output directory and is never run by tests or CI.

usage: PYTHONPATH=src python research/capture_cool_dz_release_controls.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENWD_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import scipy

from wd_spectra import _rt, gray_helium_atmosphere
from wd_spectra._compat import trapezoid
from wd_spectra._dq.provenance import digest
from wd_spectra._dq.runtime import make_material, numerical_policy
from wd_spectra._dq.data import data_root
from wd_spectra._dq.swan_strength_cache import CachedSwan
from wd_spectra.models import AtmosphereConvergenceWarning, DZConfig, compute_dz, load_atmosphere_checkpoint
from wd_spectra.models.common import ModelData
from wd_spectra.models.dq import DQConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
DATE = "2026-10-23"
OUT = DATA / "approved_regressions/cool_dz_release_2026_10_23"
PREVIOUS = DATA / "approved_regressions/independent_kurucz_2026_10_14/fixed"
DZ_CASES = ("dz-pg1225", "dz-j0738")
DQ_ROOT = DATA / "dq_regressions"
REVISION = "cool-dz-release-2026-10-23"
SOURCES = (
    "src/wd_spectra/models/stellar.py",
    "src/wd_spectra/metals.py",
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/dense_helium_continuum.py",
    "src/wd_spectra/d6.py",
    "src/wd_spectra/cool_metal_nlte.py",
    "research/capture_cool_dz_release_controls.py",
)
BANDS = ((1000, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5500), (5500, 7000), (7000, 10000), (10000, 100000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band_changes(wave, new, old):
    out = {}
    for lo, hi in BANDS:
        k = (wave >= lo) & (wave < hi)
        if np.count_nonzero(k) >= 2:
            out[f"{lo}-{hi}"] = float(trapezoid(new[k], wave[k]) / trapezoid(old[k], wave[k]) - 1.0)
    return out


def capture_dz() -> dict:
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "fixed").mkdir()
    cases = {}
    for case in DZ_CASES:
        path = DATA / "spectral_regressions" / f"{case}.npz"
        with np.load(path) as saved:
            config = DZConfig(**json.loads(str(saved["config_json"])))
            wave = saved["wavelength"]
        with np.load(PREVIOUS / f"{case}.npz") as previous:
            np.testing.assert_array_equal(wave, previous["wavelength"])
            old = previous["surface_flux"]
        atmosphere = load_atmosphere_checkpoint(
            path, config.effective_temperature, config.logg, "helium",
            include_molecules=False, include_negative_hydrogen=False,
            log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
            trihydrogen_ion_partition_model=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
            result = compute_dz(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
        new = result.spectrum.surface_flux_lambda
        assert np.all(np.isfinite(new)) and np.all(new > 0)
        assert result.spectrum.metadata["source_converged"]
        out = OUT / "fixed" / f"{case}.npz"
        np.savez_compressed(out, wavelength=wave, surface_flux=new)
        cases[case] = {
            "structure_input": str(path.relative_to(ROOT)),
            "structure_sha256": sha256(path),
            "previous_approved": str((PREVIOUS / f"{case}.npz").relative_to(ROOT)),
            "previous_sha256": sha256(PREVIOUS / f"{case}.npz"),
            "fixture_sha256": sha256(out),
            "config": json.loads(json.dumps(asdict(config), default=str)),
            "band_fractional_changes": band_changes(wave, new, old),
            "max_relative_change": float(np.max(np.abs(new / old - 1.0))),
            "atmosphere_reconverged": False,
            "cold_start_qualification": False,
        }
        print(case, "max |new/old-1| =", cases[case]["max_relative_change"], cases[case]["band_fractional_changes"])
    return cases


def dq_flux(fixture: Path, manifest: dict, historical: bool, workdir: Path):
    if historical:
        config = DQConfig(**manifest["parameters"], include_c2_ca=False)
    else:
        config = DQConfig(**manifest["parameters"])
    with np.load(fixture, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    seed = gray_helium_atmosphere(config.effective_temperature, config.logg, n_depth=len(arrays["temperature"]))
    state = replace(seed, **{k: arrays[k].copy() for k in
                             ("temperature", "gas_pressure", "column_mass", "rosseland_optical_depth")})
    with numerical_policy(workdir):
        material = make_material(config, ModelData.default(), workdir)
        if historical:
            material.integrated_swan = CachedSwan(material.table, branches=data_root() / "hornkohl_calibrated.npz",
                                                  cache_grids=192)
            new = material.spectrum(state, arrays["wavelength"], 3).surface_flux_lambda
        else:
            state = material.chemistry(state)[0]
            new = material.spectrum(state, arrays["wavelength"], manifest["n_angle"]).surface_flux_lambda
    return arrays, np.asarray(new)


def reapprove_dq() -> dict:
    summary = {}
    for name, fixture, manifest_name, historical in (
        ("j1235", DQ_ROOT / "j1235-fixed.npz", "manifest.json", True),
        ("j1311", DQ_ROOT / "j1311-completed-fixed.npz", "completed-manifest.json", False),
    ):
        manifest_path = DQ_ROOT / manifest_name
        manifest = json.loads(manifest_path.read_text())
        assert digest(fixture) == manifest["fixture_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            arrays, new = dq_flux(fixture, manifest, historical, Path(tmp))
        old = arrays["flux"]
        assert np.all(np.isfinite(new)) and np.all(new > 0)
        changes = band_changes(arrays["wavelength"], new, old)
        arrays["flux"] = new
        previous = manifest["fixture_sha256"]
        np.savez_compressed(fixture, **arrays)              # same keys, same order; structure arrays unchanged
        manifest.setdefault("reapprovals", []).append({
            "date": DATE,
            "previous_fixture_sha256": previous,
            "physics_revision": REVISION,
            "reason": ("Smooth (C1) He-REOS.3 interpolation of the non-ideal part between isotherms "
                       "(41a6cde), shared by DQ models; the previous linear-in-logT interpolation had "
                       "~30% jumps in nabla_ad at the isotherms. Structure arrays unchanged."),
            "band_fractional_changes": changes,
            "max_relative_change": float(np.max(np.abs(new / old - 1.0))),
        })
        manifest["fixture_sha256"] = digest(fixture)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        summary[name] = manifest["reapprovals"][-1]
        print(name, "max |new/old-1| =", summary[name]["max_relative_change"], changes)
    return summary


def main() -> None:
    assert all(hasattr(_rt, n) for n in ("frequency_voigt_profile", "expand_metal_line_blocks")), \
        "Build the current C backend before capturing controls"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dz = capture_dz()
    (OUT / "manifest.json").write_text(json.dumps({
        "description": __doc__.split("\n\n")[1].replace("\n", " "),
        "base_commit": commit,
        "cases": dz,
        "source_sha256": {p: sha256(ROOT / p) for p in SOURCES},
        "native_backend_sha256": sha256(Path(_rt.__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }, indent=1) + "\n")
    reapprove_dq()


if __name__ == "__main__":
    main()
