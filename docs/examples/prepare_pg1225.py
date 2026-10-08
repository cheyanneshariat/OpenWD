"""Prepare the bundled UVES coadd; no atmosphere calculation is performed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import urllib.request

import numpy as np

C_KMS = 299_792.458

ESO_URL = "https://dataportal.eso.org/dataPortal/file/{}"

UVES_PRODUCTS = {  # ESO Phase 3 product: SHA-256 of the FITS file
    "ADP.2020-09-14T19:45:12.879": "5b4132ec7e21914fc70679c0e3d6420d144c43cc3cabc1432d121718fc33267b",
    "ADP.2020-09-14T19:48:29.184": "cd540f0fe4502e2877801440b15530f69a500b34000c74baf30f869bad3128e1",
    "ADP.2021-10-25T14:23:51.506": "3a6a0909f123578adde0ebd91092f84997b7b1b0e1f2b548094b614c5a87ec5d",
    "ADP.2020-08-14T12:26:01.950": "4c0a658513e146cb1eac5fb7c32ce058a8b274f20e81e44e46a72b715fff5852",
}

def verified_tls_context():
    # Honor the user's trust configuration; never create an unverified context.
    for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        configured = os.environ.get(variable)
        if configured and not Path(configured).exists():
            raise RuntimeError(f"{variable} points to a missing CA path: {configured}")
    context = ssl.create_default_context()
    defaults = ssl.get_default_verify_paths()
    if context.cert_store_stats()["x509_ca"] or defaults.capath:
        return context
    try:
        import certifi
    except ImportError:
        certifi = None
    bundle = (Path(certifi.where()) if certifi is not None else
              Path("/etc/ssl/cert.pem") if sys.platform == "darwin" else None)
    if bundle is None or not bundle.is_file():
        raise RuntimeError(
            "No CA trust store is available. Set SSL_CERT_FILE to a trusted CA bundle "
            "or use Python's Install Certificates.command; cached FITS files need no HTTPS.")
    context.load_verify_locations(cafile=str(bundle))
    return context

def download(url, path):
    try:
        context = verified_tls_context()
        with urllib.request.urlopen(url, context=context, timeout=300) as response:
            path.write_bytes(response.read())
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not download {url} to {path}: {error}. "
            f"You can save the original archive file as {path.with_suffix('.fits')}; "
            "its pinned checksum will still be checked.") from error

def fetch(product, sha256):
    path = DATA_DIR / f"{product}.fits"
    if not path.is_file():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        download(ESO_URL.format(product), partial)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"{partial} does not match its pinned checksum; cache unchanged")
        partial.rename(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise RuntimeError(f"{path} does not match its pinned checksum")
    return path

def air_to_vacuum(wave_air):
    # IAU standard (Morton 1991), iterated because n is defined in vacuum.
    vacuum = np.array(wave_air, dtype=float)
    for _ in range(5):
        s2 = vacuum**-2
        vacuum = wave_air * (1 + 2.735182e-4 + 131.4182 * s2 + 2.76249e8 * s2**2)
    return vacuum

def prepare(data_dir, output):
    from astropy.io import fits

    global DATA_DIR
    DATA_DIR = Path(data_dir)
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output path; existing data are not overwritten")
    grid = np.arange(3300.0, 4500.0, 0.03)  # vacuum, barycentric Angstrom
    fluxes, weights, provenance = [], [], []
    source_headers = {}
    for product, sha256 in UVES_PRODUCTS.items():
        path = fetch(product, sha256)
        with fits.open(path) as hdul:
            header, table = hdul[0].header, hdul[1].data
            source_headers[product] = {"primary": header.tostring(sep="\n", padding=False),
                                       "spectrum_table": hdul[1].header.tostring(sep="\n", padding=False)}
            if (header["SPECSYS"] != "TOPOCENT" or
                    hdul[1].header["TUCD1"] != "em.wl;obs.atmos" or
                    hdul[1].header["TUNIT1"].lower() != "angstrom"):
                raise ValueError(f"Unexpected wavelength convention in {product}; inspect its headers.")
            barycentric = header["ESO QC VRAD BARYCOR"]  # km/s
            wave = air_to_vacuum(table["WAVE"][0].astype(float)) * (1 + barycentric / C_KMS)
            flux = table["FLUX"][0].astype(float)
            error = table["ERR"][0].astype(float)
        if not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
            raise ValueError(f"Non-finite or unordered wavelengths in {product}")
        good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
        fluxes.append(np.interp(grid, wave[good], flux[good], left=np.nan, right=np.nan))
        weights.append(np.interp(grid, wave[good], error[good], left=np.nan, right=np.nan) ** -2)
        provenance.append({"product": product, "url": ESO_URL.format(product), "sha256": sha256,
                           "date_obs": header["DATE-OBS"], "programme": header["PROG_ID"],
                           "target": header["OBJECT"], "exposure_seconds": header["EXPTIME"],
                           "wavelength_input": "air, topocentric, Angstrom",
                           "wavelength_output": "vacuum, barycentric, Angstrom",
                           "resolving_power": header["SPEC_RES"], "snr": header["SNR"],
                           "barycentric_correction_kms": barycentric})
        print(f"{product}  {header['DATE-OBS'][:10]}  S/N {header['SNR']:.1f}  "
              f"barycentric correction {barycentric:+.2f} km/s")

    fluxes, weights = np.array(fluxes), np.array(weights)
    reference = (grid > 4150) & (grid < 4400)
    scale = np.nanmedian(fluxes[0, reference]) / np.nanmedian(fluxes[:, reference], axis=1)
    fluxes *= scale[:, None]
    weights /= scale[:, None] ** 2
    weights[~np.isfinite(fluxes) | ~np.isfinite(weights)] = 0.0
    fluxes[weights == 0] = 0.0
    covered = weights.sum(axis=0) > 0
    obs_wave = grid[covered]
    obs_flux = (fluxes * weights).sum(axis=0)[covered] / weights.sum(axis=0)[covered]
    obs_error = weights.sum(axis=0)[covered] ** -0.5

    RESOLVING_POWER = provenance[0]["resolving_power"]
    assert all(p["resolving_power"] == RESOLVING_POWER for p in provenance)
    print(f"Exposure scale factors: {np.round(scale, 3)}")
    print(f"Coadd: {obs_wave[0]:.1f}-{obs_wave[-1]:.1f} Angstrom, "
          f"median S/N per pixel at 4150-4400 Angstrom = "
          f"{np.median((obs_flux / obs_error)[(obs_wave > 4150) & (obs_wave < 4400)]):.0f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, wave=obs_wave, flux=obs_flux, error=obs_error)
    metadata = {
        "schema_version": 1, "target": "PG 1225-079", "copyright": "ESO",
        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_policy": "https://archive.eso.org/cms/eso-data-access-policy.html",
        "collection_doi": "https://doi.org/10.18727/archive/50",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "wavelength_frame": "vacuum, barycentric", "wavelength_unit": "Angstrom",
        "flux_unit": "1e-16 erg cm^-2 s^-1 Angstrom^-1", "error_unit": "same as flux",
        "resolving_power": RESOLVING_POWER, "n_pixels": len(obs_wave),
        "grid_step_angstrom": 0.03, "scale_factors": scale.tolist(),
        "processing": "Iterated OpenWD air-to-vacuum formula; wavelength multiplied by 1+barycor/c; linear flux/error-amplitude interpolation; scaled to first exposure in 4150-4400 Angstrom; inverse-variance mean.",
        "error_treatment": "Approximate diagonal pipeline weights; resampling covariance is not propagated.",
        "checksums_recorded": "2026-10-07", "sources": provenance,
        "source_fits_headers": source_headers,
        "acknowledgement": "Based on observations collected at the European Southern Observatory under programmes 165.H-0588(A) and 167.D-0407(A), obtained from the ESO Science Archive Facility.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return obs_wave, obs_flux, obs_error, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Cache for the four original FITS products")
    parser.add_argument("--output", type=Path, required=True, help="New .npz output path")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output)
