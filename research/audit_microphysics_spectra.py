"""Measure a physics correction against immutable fixed-atmosphere controls.

This is a synthesis-only audit, never a cold-start convergence certificate.
It does not overwrite any baseline or change regression-test tolerances.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import warnings

import numpy as np
from wd_spectra._compat import trapezoid
from wd_spectra.models import (
    DAConfig, DAZConfig, DBConfig, DABConfig, DZConfig,
    compute_da, compute_daz, compute_db, compute_dab, compute_dz,
    load_atmosphere_checkpoint, AtmosphereConvergenceWarning,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "da-3000", "da-4000", "da-5000", "da-12000-public", "da-20000",
    "db-10000", "db-22000", "dab-9000", "dab-20000",
    "dz-pg1225", "dz-j0738", "daz-g149_28", "daz-galex1931",
)


def compare(case, output, synthesis_transfer=None):
    start = time.monotonic()
    print(f"[{case}] starting fixed-atmosphere synthesis", flush=True)
    daz = case.startswith("daz-")
    path = ROOT / "tests/data" / (
        "daz_regressions" if daz else "spectral_regressions"
    ) / ((case[4:] if daz else case) + ".npz")
    with np.load(path) as saved:
        kind = "DAZ" if daz else str(saved["spectral_type"])
        cls, compute, composition = {
            "DA": (DAConfig, compute_da, "hydrogen"),
            "DAZ": (DAZConfig, compute_daz, "hydrogen"),
            "DB": (DBConfig, compute_db, "helium"),
            "DAB": (DABConfig, compute_dab, "mixed"),
            "DZ": (DZConfig, compute_dz, "helium"),
        }[kind]
        config = cls(**json.loads(str(saved["config_json"])))
        wave = saved["wavelength"]
        original = saved["original_surface_flux"]
        expected = (
            saved["checked_surface_flux"]
            if "checked_surface_flux" in saved else original
        )
    molecular = kind in {"DA", "DAZ"} and config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, composition,
        include_molecules=molecular, include_negative_hydrogen=molecular,
        log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
        trihydrogen_ion_partition_model=(
            "neale-tennyson-1995" if kind in {"DA", "DAZ"} else None
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        if synthesis_transfer is not None and kind not in {"DA", "DZ"}:
            raise ValueError("explicit synthesis_transfer is supported here for DA/DZ only")
        result = compute(config, wave, initial_atmosphere=atmosphere,
                         relax_atmosphere=False,
                         **({"synthesis_transfer": synthesis_transfer}
                            if synthesis_transfer is not None else {}))
    new = result.spectrum.surface_flux_lambda
    difference = new - expected
    peak = np.max(wave * expected)
    significant = wave * expected > 0.01 * peak
    maximum = int(np.argmax(wave * abs(difference)))
    source = result.spectrum.metadata
    report = dict(
        case=case, elapsed_seconds=time.monotonic() - start,
        fixed_atmosphere_only=True, atmosphere_convergence_not_tested=True,
        source_converged=bool(source["source_converged"]),
        transfer_discretization=source.get("transfer_discretization"),
        max_peak_scaled_change=float(np.max(wave * abs(difference)) / peak),
        wavelength_of_maximum_change=float(wave[maximum]),
        max_significant_relative_change=float(
            np.max(abs(difference[significant] / expected[significant]))
        ),
        fixed_flux_array_matches_reference=bool(
            np.allclose(new, expected, rtol=2e-6, atol=1e-12 * np.max(expected))
            if not daz else np.allclose(new[significant], expected[significant],
                                       rtol=1e-3, atol=0.)
        ),
        integrated_absolute_change_fraction=float(
            trapezoid(abs(difference), wave) / trapezoid(expected, wave)
        ),
        band_fractional_changes={},
    )
    for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
        take = (wave >= lo) & (wave <= hi)
        denominator = trapezoid(expected[take], wave[take])
        report["band_fractional_changes"][f"{lo}-{hi}"] = (
            float(trapezoid(difference[take], wave[take]) / denominator)
            if np.count_nonzero(take) >= 2 and denominator > 0 else None
        )
    np.savez_compressed(output / (case + ".npz"), wavelength=wave,
                        old_flux=expected, new_flux=new, original_flux=original)
    (output / (case + ".json")).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--synthesis-transfer", choices=("formal-linear", "formal-pchip"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = [f.result() for f in as_completed(
            [pool.submit(compare, c, args.output, args.synthesis_transfer) for c in args.case or CASES]
        )]
    (args.output / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
