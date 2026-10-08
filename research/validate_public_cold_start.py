"""Fresh public-API validation; stored structures are never solver inputs."""

import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from wd_spectra import (
    DAConfig,
    DBConfig,
    DABConfig,
    DZConfig,
    compute_da,
    compute_db,
    compute_dab,
    compute_dz,
    save_model_result,
)
from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error("refusing to overwrite a validation")
args.output.mkdir(parents=True)
kind, teff, *resolution = args.case.split("-")
quality = "standard" if resolution else "production"
if kind == "dz":
    # Object parameters/abundances only, never the saved atmosphere arrays.
    with np.load(
        Path("tests/data/spectral_regressions") / (args.case + ".npz")
    ) as saved:
        config = DZConfig(**json.loads(str(saved["config_json"])))
    teff = config.effective_temperature
else:
    config = {"da": DAConfig, "db": DBConfig, "dab": DABConfig}[kind](
        effective_temperature=float(teff), quality=quality
    )
compute = {
    "da": compute_da,
    "db": compute_db,
    "dab": compute_dab,
    "dz": compute_dz,
}[kind]
# Independent requested wavelengths, not the atmosphere or grid of a prior model.
wave = np.unique(
    np.r_[np.geomspace(900.0, 300000.0, 1100), np.arange(3700.0, 7000.0, 2.0)]
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
started = time.monotonic()


def report(iteration, atmosphere, diagnostics):
    record = dict(
        elapsed=time.monotonic() - started, iteration=iteration, **diagnostics
    )
    print(json.dumps(record), flush=True)
    with (args.output / "iterations.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


print(
    "PUBLIC COLD START: no stored/neighboring atmosphere supplied",
    config,
    flush=True,
)
result = compute(config, wave, iteration_callback=report)
save_model_result(result, args.output)
metadata = result.atmosphere.metadata
summary = dict(
    cold_start=True,
    elapsed_seconds=time.monotonic() - started,
    certificate=metadata.get(
        "equilibrium_certificate",
        {"verified": False, "failures": ["legacy-no-certificate"]},
    ),
    iterations=metadata.get(
        "radiative_equilibrium_iterations_including_domain_adaptation",
        metadata["radiative_equilibrium_iterations"],
    ),
    thermal_sweeps=(metadata.get("thermal_conditioning") or {}).get(
        "iterations", 0
    ),
)
# References are opened only AFTER the fresh calculation has finished.
# DZ remains a separate paper-comparison diagnostic; its historical reference
# has not been promoted to a cold-canary baseline. Preserve that explicit check.
reference_directory = (
    "tests/data/spectral_regressions" if kind == "dz"
    else "tests/data/approved_regressions/cold"
)
with np.load(
    Path(reference_directory) / (args.case + ".npz")
) as saved:
    w, i, j = np.intersect1d(wave, saved["wavelength"], return_indices=True)
    old = saved["checked_surface_flux" if kind == "dz" else "surface_flux"][j]
    new = result.spectrum.surface_flux_lambda[i]
    peak = np.max(w * old)
    significant = w * old > 0.01 * peak
    summary["spectral_maximum_peak_scaled_change"] = float(
        np.max(w * abs(new - old)) / peak
    )
    summary["spectral_maximum_significant_relative_change"] = float(
        np.max(abs(new[significant] / old[significant] - 1))
    )
    target = STEFAN_BOLTZMANN * float(teff) ** 4
    summary["spectral_integrated_absolute_difference_over_stellar_flux"] = (
        float(trapezoid(abs(new - old), w) / target)
    )
    summary["bands"] = []
    for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
        q = (w >= lo) & (w <= hi)
        if np.sum(q) < 2:
            continue
        integral = trapezoid(old[q], w[q])
        summary["bands"].append(
            dict(
                lo=lo,
                hi=hi,
                old_fraction_of_stellar_flux=float(integral / target),
                fractional_change=float(
                    trapezoid((new - old)[q], w[q]) / integral
                ),
            )
        )
(args.output / "validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print("PUBLIC VALIDATION: " + json.dumps(summary), flush=True)
assert summary["certificate"]["verified"], summary["certificate"]["failures"]
assert summary["spectral_maximum_peak_scaled_change"] < 0.003, summary
assert (
    summary["spectral_integrated_absolute_difference_over_stellar_flux"]
    < 0.003
), summary
for band in summary["bands"]:
    if band["old_fraction_of_stellar_flux"] >= 1e-3:
        assert abs(band["fractional_change"]) < 0.003, band
