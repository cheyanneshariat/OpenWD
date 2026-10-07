#!/usr/bin/env python3
"""HS 0209+0832 at the published parameters: H/He LTE host with trace metals and Nb.

    python research/hs0209_niobium.py model --output DIR [--quality standard]
    python research/hs0209_niobium.py windows --output DIR
    python research/hs0209_niobium.py flux --output DIR
    python research/hs0209_niobium.py model --control metal-free --output DIR2
    python research/hs0209_niobium.py model --control negligible-metals --output DIR3

``model`` relaxes one homogeneous H/He atmosphere from scratch with the nine
detected metals of Williams et al. (2026, arXiv:2610.07161, Extended Data
Tables 1 and 3) in its charge balance and opacity, and saves the ordinary
model outputs plus a pickled atmosphere.  Nothing is fitted: T_eff, log g,
He/H and every abundance are the paper's values.  ``windows`` synthesizes the
six Figure 1 windows on that unchanged atmosphere, with and without Nb, at a
spacing that resolves the thermal Nb line cores.  ``flux`` integrates the
emergent flux from 100 A to 100 microns.  The two controls relax the same
H/He atmosphere without metals (the established DAB preset) and with every
metal at log N(Z)/N(H) = -20, which must reproduce it.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import resource
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

import wd_spectra
from wd_spectra import DABConfig, save_model_result
from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.models.stellar import compute_dab

# Extended Data Table 1 (T_eff, log g, He/H) and Table 3 (log N(Z)/N(H)).
PAPER_ABUNDANCES = {
    "C": -6.04, "Al": -7.37, "Si": -8.26, "Ca": -4.64, "Ti": -6.24,
    "Ni": -6.32, "Cu": -6.46, "Zn": -6.24, "Nb": -6.33,
}
# Stage V matters for Si, Ti, Ni, Cu and Zn at this temperature, and Nb V
# for Nb (its closed-shell Nb VI tops the ladder).
MAXIMUM_CHARGE = {**{element: 4 for element in PAPER_ABUNDANCES}, "Nb": 5}
# Figure 1 panels (observed-frame limits) padded for velocity and the LSF.
WINDOWS = {
    "fuse_1003": (1001.5, 1009.0),
    "fuse_1050": (1048.0, 1058.0),
    "stis_1366": (1363.0, 1372.0),
    "stis_1434": (1431.5, 1437.5),
    "stis_1452": (1450.0, 1455.5),
    "stis_1640": (1637.0, 1645.0),
}
WINDOW_STEP_ANGSTROM = 0.003


def config(quality: str, abundances=PAPER_ABUNDANCES) -> DABConfig:
    if abundances is None:
        return DABConfig(effective_temperature=35_800.0, logg=7.90,
                         log_hydrogen_to_helium=1.90, quality=quality)
    return DABConfig(
        effective_temperature=35_800.0,
        logg=7.90,
        log_hydrogen_to_helium=1.90,
        quality=quality,
        abundances=dict(abundances),
        maximum_metal_charge={element: MAXIMUM_CHARGE[element] for element in abundances},
        metal_classical_electron_stark=True,
    )


def _provenance() -> dict:
    root = Path(wd_spectra.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    return {
        "source": str(root), "commit": commit, "src_modified": bool(dirty),
        "compiled_backend": wd_spectra.compiled_backend_available(),
        "python": platform.python_version(), "numpy": np.__version__,
    }


CONTROLS = {
    "paper": PAPER_ABUNDANCES,
    "metal-free": None,
    "negligible-metals": {element: -20.0 for element in PAPER_ABUNDANCES},
}


def run_model(output: Path, quality: str, control: str = "paper") -> None:
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_dab(config(quality, CONTROLS[control]))
    elapsed = time.monotonic() - start
    save_model_result(result, output / "model")
    with (output / "atmosphere.pkl").open("wb") as stream:
        pickle.dump(result.atmosphere, stream)
    certificate = result.atmosphere.metadata.get("equilibrium_certificate", {})
    record = {
        "control": control,
        "config": json.loads(json.dumps(result.config.__dict__, default=str)),
        "wall_seconds": elapsed,
        # ru_maxrss is in bytes on macOS and KiB on Linux.
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**20 if platform.system() == "Darwin" else 2**10),
        "certificate_verified": bool(certificate.get("verified")),
        "certificate": certificate,
        "surface_flux_over_sigma_teff4": float(
            trapezoid(result.spectrum.surface_flux_lambda, result.spectrum.wavelength_angstrom)
            / (STEFAN_BOLTZMANN * 35_800.0**4)
        ),
        "warnings": [str(item.message) for item in caught],
        "provenance": _provenance(),
    }
    (output / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps({key: record[key] for key in
                      ("wall_seconds", "peak_rss_mib", "certificate_verified",
                       "surface_flux_over_sigma_teff4")}, indent=2))


def run_windows(output: Path) -> None:
    with (output / "atmosphere.pkl").open("rb") as stream:
        atmosphere = pickle.load(stream)
    quality = json.loads((output / "run.json").read_text())["config"]["quality"]
    without_nb = {key: value for key, value in PAPER_ABUNDANCES.items() if key != "Nb"}
    arrays, timing = {}, {}
    for name, (low, high) in WINDOWS.items():
        wave = np.arange(low, high, WINDOW_STEP_ANGSTROM)
        arrays[f"{name}_wavelength"] = wave
        for label, abundances in (("full", PAPER_ABUNDANCES), ("no_nb", without_nb)):
            start = time.monotonic()
            with warnings.catch_warnings():
                # Fixed-structure synthesis of the converged atmosphere above.
                warnings.simplefilter("ignore")
                result = compute_dab(config(quality, abundances), wave,
                                     initial_atmosphere=atmosphere, relax_atmosphere=False)
            arrays[f"{name}_{label}"] = result.spectrum.surface_flux_lambda
            timing[f"{name}_{label}"] = time.monotonic() - start
            print(name, label, f"{timing[f'{name}_{label}']:.1f} s", flush=True)
    np.savez_compressed(output / "windows.npz", **arrays)
    (output / "windows.json").write_text(json.dumps({
        "windows": WINDOWS, "step_angstrom": WINDOW_STEP_ANGSTROM,
        "frame": "stellar rest frame, vacuum", "flux": "surface F_lambda",
        "no_nb": "same relaxed atmosphere; Nb removed from the synthesis only",
        "seconds": timing, "provenance": _provenance(),
    }, indent=2) + "\n")


def run_flux(output: Path) -> None:
    with (output / "atmosphere.pkl").open("rb") as stream:
        atmosphere = pickle.load(stream)
    record = json.loads((output / "run.json").read_text())
    wave = np.geomspace(100.0, 1.0e6, 6000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_dab(config(record["config"]["quality"],
                                    CONTROLS[record.get("control", "paper")]),
                             wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    total = trapezoid(result.spectrum.surface_flux_lambda, wave)
    below = wave < 900.0
    summary = {
        "grid": "geomspace(100 A, 1e6 A, 6000)",
        "surface_flux_over_sigma_teff4": float(total / (STEFAN_BOLTZMANN * 35_800.0**4)),
        "fraction_below_900_angstrom": float(
            trapezoid(result.spectrum.surface_flux_lambda[below], wave[below]) / total
        ),
    }
    (output / "flux.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("model", "windows", "flux"))
    parser.add_argument("--control", default="paper", choices=tuple(CONTROLS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", default="standard",
                        choices=("quick", "standard", "production"))
    args = parser.parse_args()
    if args.step == "model":
        run_model(args.output, args.quality, args.control)
    elif args.step == "flux":
        run_flux(args.output)
    else:
        run_windows(args.output)


if __name__ == "__main__":
    main()
