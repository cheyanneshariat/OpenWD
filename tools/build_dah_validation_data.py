#!/usr/bin/env python3
"""Package the observed DAH validation spectra as compact bundled arrays.

Inputs (development checkout ``.cache/magnetic-validation``):

* original SDSS/BOSS spectrum FITS files (vacuum wavelengths), a
  photometry-recalibrated copy of the J1034+0327 BOSS spectrum, and the
  Gianninas GH Leo optical spectrum mirrored by the MWDD;
* the Hardy, Dufour & Jordan (2023) model curves digitized from the official
  CDS panels (air wavelengths; converted to vacuum here);
* two SPY/UVES ESO Phase 3 epochs each of GD 9 and G 76-48, cut to the
  Halpha--Hdelta windows (air converted to vacuum by the reader).

Survey spectra of the Hardy targets are de-reddened with the per-object MWDD
E(B-V) (Fitzpatrick 1999, R_V=3.1) as in Hardy et al.; the two
Vera-Rueda & Rohrmann (2024) targets are used as observed.  Each output file
holds the rest-frame-independent observation (``wavelength_angstrom``,
``flux_nu``, ``sigma_log_wavelength``) and, where available, the Hardy curve.
Requires ``astropy`` and ``extinction`` (build time only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from extinction import fitzpatrick99

from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.validation import (
    physical_air_to_vacuum,
    read_eso_phase3_spectrum,
    read_sdss_spectrum,
)
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_TARGETS,
    DAH_WEAK_FIELD_TARGETS,
    DAH_WEAK_LINE_WINDOWS,
)


def _observation(root: Path, target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / target.source_file
    if target.source_format == "mwdd-fnu-csv":
        table = np.genfromtxt(path, delimiter=",", skip_header=2)
        valid = np.isfinite(table[:, 0]) & np.isfinite(table[:, 1]) & (table[:, 1] > 0.0)
        wave = np.asarray(table[valid, 0], dtype=np.float64)
        flux_nu = np.asarray(table[valid, 1], dtype=np.float64)
        sigma_log = np.full_like(wave, target.resolution_fwhm_angstrom / 2.354820045 / wave)
    else:
        observed = read_sdss_spectrum(path)
        valid = (
            (observed.inverse_variance > 0.0)
            & np.isfinite(observed.flux_lambda)
            & (observed.flux_lambda > 0.0)
        )
        wave = np.asarray(observed.wavelength_angstrom[valid], dtype=np.float64)
        flux_nu = (
            observed.flux_lambda[valid] * 1.0e-17 * wave**2 * 1.0e-8 / LIGHT_SPEED
        )
        sigma_log = np.asarray(observed.resolution_sigma_log_wavelength[valid])
    keep = (wave >= 3_500.0) & (wave <= 7_500.0)
    wave, flux_nu, sigma_log = wave[keep], flux_nu[keep], sigma_log[keep]
    if target.ebv > 0.0:
        flux_nu = flux_nu * 10.0 ** (0.4 * fitzpatrick99(wave, 3.1 * target.ebv, 3.1))
    return wave, flux_nu, sigma_log


_WEAK_CACHE = ".cache/magnetic-validation"


def build(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for target in DAH_VALIDATION_TARGETS:
        wave, flux_nu, sigma_log = _observation(root, target)
        arrays = dict(
            wavelength_angstrom=wave,
            flux_nu=flux_nu,
            sigma_log_wavelength=sigma_log,
        )
        if target.hardy_curve_file is not None:
            with np.load(root / target.hardy_curve_file) as saved:
                arrays["hardy_wavelength_angstrom"] = physical_air_to_vacuum(
                    np.asarray(saved["model_wavelength_angstrom"], dtype=np.float64)
                )
                arrays["hardy_flux"] = np.asarray(
                    saved["model_flux_plot_units"], dtype=np.float64
                )
        path = output / f"{target.key}.npz"
        np.savez_compressed(path, **arrays)
        print(f"{target.key}: {wave.size} pixels -> {path} ({path.stat().st_size / 1e3:.0f} kB)")
    for target in DAH_WEAK_FIELD_TARGETS:
        arrays = {"n_epochs": np.int64(len(target.product_pairs))}
        for index, pair in enumerate(target.product_pairs):
            observed = read_eso_phase3_spectrum(
                [root / _WEAK_CACHE / target.key / f"{name}.fits" for name in pair],
                survey=f"SPY/UVES (ESO Phase 3; {target.display})",
            )
            wave = observed.wavelength_angstrom
            keep = np.zeros(wave.size, dtype=bool)
            for _, center, half, _ in DAH_WEAK_LINE_WINDOWS:
                keep |= np.abs(wave - center) <= half + 12.0
            keep &= np.isfinite(observed.flux_lambda) & np.isfinite(observed.inverse_variance)
            arrays[f"epoch{index}_wavelength_angstrom"] = wave[keep]
            arrays[f"epoch{index}_flux"] = observed.flux_lambda[keep]
            arrays[f"epoch{index}_inverse_variance"] = observed.inverse_variance[keep]
        path = output / f"{target.key}.npz"
        np.savez_compressed(path, **arrays)
        print(f"{target.key}: {len(target.product_pairs)} UVES epochs -> {path} ({path.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help=".cache/magnetic-validation of a dev checkout")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("src/wd_spectra/data/validation/dah"),
    )
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == "__main__":
    main()
