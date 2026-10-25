#!/usr/bin/env python3
"""Compute one cool DZ/DAZ paper model at its published parameters.

The atmosphere is solved from a cold start with the default DZConfig
(helium host) or DAZConfig (hydrogen host) at standard quality.  The output
directory receives the usual model files (save_model_result), the emergent
spectrum on a 0.05-A grid over 3000-10000 A (spectrum_fine.npz, vacuum
wavelengths, erg s^-1 cm^-2 A^-1), the convergence certificate status
(summary.json) and one JSON line per structure iteration (iterations.jsonl).
Numerical libraries are limited to one thread.

usage: python research/cool_dz_paper/run_model.py KEY OUTDIR [--quality standard]
"""

import argparse
import json
import os
import time
from pathlib import Path

for _variable in ("OPENWD_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

import wd_spectra.metals  # noqa: E402
from wd_spectra import DAZConfig, DZConfig, compute_daz, compute_dz  # noqa: E402
from wd_spectra.models.common import default_wavelength_grid, save_model_result  # noqa: E402

# the paper models used the compiled transfer kernel; without it a run is several times slower
assert wd_spectra.metals._rt is not None, "build the C kernel (wd_spectra._rt) before running"

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--quality", default="standard")
    args = parser.parse_args()
    entry = json.loads((HERE / "objects.json").read_text())["objects"][args.key]
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    wavelength = np.unique(np.r_[
        default_wavelength_grid(900.0, 30000.0, continuum_points=2500, optical_step=1.0),
        np.arange(3000.0, 10000.0, 0.05)])
    if entry["host"] == "He":
        config = DZConfig(effective_temperature=entry["teff"], logg=entry["logg"], abundances=entry["abundances"],
                          log_hydrogen_abundance=entry["log_h"], quality=args.quality)
        compute = compute_dz
    else:
        config = DAZConfig(effective_temperature=entry["teff"], logg=entry["logg"], abundances=entry["abundances"],
                           quality=args.quality)
        compute = compute_daz
    start = time.time()

    def record(iteration, atmosphere, info):
        values = {k: v for k, v in dict(info).items() if isinstance(v, (int, float, str, bool)) or v is None}
        with open(out / "iterations.jsonl", "a") as handle:
            handle.write(json.dumps({"iteration": int(iteration), "wall_s": round(time.time() - start, 1),
                                     **values}, default=str) + "\n")

    result = compute(config, wavelength, iteration_callback=record)
    save_model_result(result, out)
    np.savez_compressed(out / "spectrum_fine.npz", wavelength=result.spectrum.wavelength_angstrom,
                        flux=result.spectrum.surface_flux_lambda)
    certificate = result.atmosphere.metadata.get("equilibrium_certificate", {})
    summary = {"key": args.key, "wall_s": round(time.time() - start, 1),
               "verified": bool(certificate.get("verified")) if isinstance(certificate, dict) else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
