#!/usr/bin/env python3
"""Co-add ESO X-shooter Phase 3 (IDP) spectra of one arm into one barycentric spectrum.

    python research/sdb_coadd_xshooter.py --arm UVB --proposal "096.D-0055(A)" \
        --input results/sdb/obs/hd4539 --output results/sdb/obs/hd4539/coadd-uvb.fits

Each spectrum is shifted from the topocentric to the barycentric frame with the
pipeline's ESO QC VRAD BARYCOR, linearly interpolated onto the wavelength grid of
the highest-S/N exposure, and averaged with inverse-variance weights.  Pixels
with nonzero QUAL or non-positive error are excluded.  Wavelengths stay in air
(the IDP convention) and are written in Angstrom; the flux calibration is kept.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

LIGHT_SPEED_KMS = 299792.458


def read(path):
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[1].data
        wave = np.ravel(data['WAVE']).astype(float) * 10.0
        flux = np.ravel(data['FLUX']).astype(float)
        error = np.ravel(data['ERR']).astype(float)
        quality = np.ravel(data['QUAL']) if 'QUAL' in data.columns.names else np.zeros(wave.size, int)
    barycentric = float(header['HIERARCH ESO QC VRAD BARYCOR'])
    good = (quality == 0) & (error > 0) & np.isfinite(flux)
    return dict(wave=wave * (1.0 + barycentric / LIGHT_SPEED_KMS), flux=flux, error=error,
                good=good, snr=float(header.get('SNR', 0.0)), mjd=float(header['MJD-OBS']),
                exptime=float(header['EXPTIME']), barycentric=barycentric, name=path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['UVB', 'VIS'], required=True)
    parser.add_argument('--proposal', required=True)
    args = parser.parse_args()

    spectra = []
    for path in sorted(args.input.glob('ADP.*.fits')):
        header = fits.getheader(path, 0)
        if header.get('HIERARCH ESO SEQ ARM') == args.arm and header.get('HIERARCH ESO OBS PROG ID') == args.proposal:
            spectra.append(read(path))
    if not spectra:
        raise SystemExit('no matching spectra')
    reference = max(spectra, key=lambda s: s['snr'])
    grid = reference['wave']
    weight_sum = np.zeros_like(grid)
    weighted_flux = np.zeros_like(grid)
    for spectrum in spectra:
        with np.errstate(divide='ignore'):
            inverse_variance = np.where(spectrum['good'], spectrum['error'] ** -2.0, 0.0)
        flux = np.interp(grid, spectrum['wave'], spectrum['flux'], left=np.nan, right=np.nan)
        weight = np.interp(grid, spectrum['wave'], inverse_variance, left=0.0, right=0.0)
        # A pixel adjacent to a rejected one gets zero weight rather than a blend.
        bad = np.interp(grid, spectrum['wave'], (~spectrum['good']).astype(float), left=1.0, right=1.0) > 0
        weight[bad | ~np.isfinite(flux)] = 0.0
        weighted_flux += weight * np.nan_to_num(flux)
        weight_sum += weight
    covered = weight_sum > 0
    flux = np.where(covered, weighted_flux / np.where(covered, weight_sum, 1.0), np.nan)
    with np.errstate(divide='ignore'):
        error = np.where(covered, weight_sum ** -0.5, np.nan)

    header = fits.Header()
    header['OBJECT'] = fits.getheader(args.input / reference['name'], 0).get('OBJECT')
    header['ARM'] = args.arm
    header['PROPOSAL'] = args.proposal
    header['NCOMBINE'] = len(spectra)
    header['WAVEFRAM'] = ('air, barycentric', 'wavelength convention')
    header['WAVEUNIT'] = 'Angstrom'
    header['FLUXUNIT'] = 'erg s^-1 cm^-2 Angstrom^-1'
    header['REFSPEC'] = reference['name']
    for index, spectrum in enumerate(spectra):
        header[f'IN{index:02d}'] = spectrum['name']
    columns = [fits.Column(name='WAVE', format='D', array=grid), fits.Column(name='FLUX', format='D', array=flux),
               fits.Column(name='ERR', format='D', array=error)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns)]).writeto(
        args.output, overwrite=True)
    median_snr = np.nanmedian(flux / error)
    print(f'{args.arm}: {len(spectra)} spectra -> {args.output} (median S/N per pixel {median_snr:.0f}; '
          f'barycentric corrections {min(s["barycentric"] for s in spectra):.2f} to '
          f'{max(s["barycentric"] for s in spectra):.2f} km/s)')


if __name__ == '__main__':
    main()
