#!/usr/bin/env python3
"""Fetch and inspect the Preval et al. G191-B2B benchmark (no model fitting).

Run from the repository root. Raw observations remain under ignored results/.
The manifest pins the published parameter/abundance convention and file hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import urllib.request
import numpy as np

PAPER = 'https://doi.org/10.1093/mnras/stt1604'
ARCHIVE = 'https://archive.stsci.edu/prepds/wd-linelist/'
BASE = 'https://archive.stsci.edu/missions/hlsp/wd-linelist/'
FILES = {
    'spectrum': ('hst_coadd/hlsp_wd-linelist_hst_stis_g191-b2b_e140h_v1_coadd-spec.fits',
                 '7e633ed5f29f3376b2ccc142c98d9e314176e3613e9e53a17baac7759d19dddb'),
    'lines': ('linelists/hlsp_wd-linelist_hst_stis_g191-b2b_e140h_v1_linelist.txt',
              '02a28cd46d7e66b765883400c335f93c6b943c089914c55196c209d0a734298a'),
}
BENCHMARK = dict(
    target='G191-B2B', reference=PAPER, archive=ARCHIVE,
    effective_temperature=52500., logg=7.53, helium_to_hydrogen=1e-5,
    abundance_convention='N(element)/N(H), inferred from the indicated ion; not ionic fractions',
    carbon_from_ciii=dict(value=1.72e-7, minus_1sigma=0.02e-7, plus_1sigma=0.02e-7),
    silicon_from_siiv=dict(value=3.68e-7, minus_1sigma=0.14e-7, plus_1sigma=0.13e-7),
    carbon_from_civ=dict(value=2.13e-7, minus_1sigma=0.15e-7, plus_1sigma=0.29e-7),
    silicon_from_siiii=dict(value=3.16e-7, minus_1sigma=0.30e-7, plus_1sigma=0.31e-7),
    photospheric_velocity_kms=23.8, nominal_resolving_power=144000,
    observation='MAST HLSP v1 STIS E140H coadd; vacuum Angstrom, observer-frame F_lambda',
    qualifications=[
        'Use the 52500 K, logg=7.53 parameters associated with these abundances; do not mix with a later 60000 K analysis.',
        'Published background is metal blanketed (including Fe/Ni); an H/He+C/Si prototype is an incomplete comparison.',
        'Si IV resonance lines have a non-photospheric component near +8 to +9 km/s (paper Table 11).',
        'Several C III 1175 components have possible Fe/Ni blends in the published catalogue.',
        'Ly alpha requires ISM absorption and airglow handling; no Ly alpha abundance-fit score is produced here.',
        'STIS data test the physics; a later COS comparison requires the actual COS LSF.',
    ])
WINDOWS = {'ciii_1175': (1173.8, 1177.0), 'lyalpha': (1190., 1242.),
           'siiv_1393': (1393.25, 1394.35), 'siiv_1402': (1402.25, 1403.35)}


def checked_files(directory, fetch=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, (relative, expected) in FILES.items():
        path = directory / Path(relative).name
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f'{path}; rerun with --fetch')
            with urllib.request.urlopen(BASE + relative, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'MAST checksum mismatch for {relative}')
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'checksum mismatch: {path}')
        paths[kind] = path
    return paths


def read_observation(path):
    from astropy.io import fits
    with fits.open(path) as hdus:
        if hdus[0].header.get('AIRORVAC') != 'VAC':
            raise ValueError('benchmark must use vacuum wavelengths')
        table = hdus[1].data
        wave, flux, error = (np.asarray(table[k], dtype=float).copy() for k in ('WAVE', 'FLUX', 'ERROR'))
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    wave, flux, error = wave[good], flux[good], error[good]
    if np.any(np.diff(wave) <= 0):
        raise ValueError('observed wavelengths are not strictly increasing')
    return wave, flux, error


def diagnostic_lines(path):
    rows = []
    for text in Path(path).read_text().splitlines():
        if not text.strip() or text.lstrip().startswith('#'):
            continue
        f = shlex.split(text)
        observed = float(f[0])
        if any(lo <= observed <= hi for key, (lo, hi) in WINDOWS.items() if key != 'lyalpha'):
            rows.append(dict(observed_wavelength=observed, equivalent_width_mA=float(f[2]),
                             equivalent_width_error_mA=float(f[3]), element=f[4], ion=f[5],
                             rest_wavelength=float(f[6]), velocity_kms=float(f[8]), origin=f[11]))
    return rows


def prepare(directory, fetch=False, plot=True):
    paths = checked_files(directory, fetch)
    wave, flux, error = read_observation(paths['spectrum'])
    directory = Path(directory)
    selected = np.zeros(wave.size, dtype=bool)
    for lo, hi in WINDOWS.values():
        selected |= (wave >= lo) & (wave <= hi)
    # Candidate masks only. A full fit should explicitly model these absorbers.
    mask_nonphotospheric = np.zeros_like(selected)
    for center in (1393.795, 1402.809):
        mask_nonphotospheric |= abs(wave-center) < center * 8. / 299792.458
    mask_lyalpha_core = (wave >= 1215.) & (wave <= 1217.)
    np.savez(directory/'observed_windows.npz', wavelength=wave[selected],
             flux=flux[selected], error=error[selected],
             candidate_nonphotospheric_mask=mask_nonphotospheric[selected],
             candidate_lyalpha_core_mask=mask_lyalpha_core[selected])
    manifest = {**BENCHMARK, 'files': {k: dict(url=BASE+r, sha256=s) for k,(r,s) in FILES.items()},
                'retained_pixels': int(selected.sum()), 'windows_observer_angstrom': WINDOWS,
                'candidate_mask_policy': 'Si IV: +/-8 km/s around Table 11 centers; Ly alpha: 1215-1217 A. Not validated fit masks.',
                'diagnostic_line_catalogue': diagnostic_lines(paths['lines']),
                'model_comparison_completed': False}
    (directory/'benchmark.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
        for ax, (name, (lo, hi)) in zip(axes.flat, WINDOWS.items()):
            take = (wave >= lo) & (wave <= hi)
            scale = 1e-11
            ax.plot(wave[take], flux[take]/scale, color='#244b73', lw=0.8)
            ax.fill_between(wave[take], (flux[take]-error[take])/scale,
                            (flux[take]+error[take])/scale, alpha=.2, color='#244b73')
            if name.startswith('siiv'):
                for center in (1393.795, 1402.809):
                    if lo < center < hi:
                        width = center*8/299792.458
                        ax.axvspan(center-width, center+width, color='#c16a30', alpha=.2,
                                   label='Non-photospheric component region')
                ax.legend(fontsize=8)
            if name == 'lyalpha':
                ax.axvspan(1215.,1217.,color='#c16a30',alpha=.2,label='Core: ISM / airglow')
                ax.legend(fontsize=8)
            ax.set(title=name.replace('_',' '), xlabel='Observed vacuum wavelength (A)',
                   ylabel=r'$F_\lambda$ ($10^{-11}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)')
            ax.set_xlim(lo,hi)
        fig.suptitle('G191-B2B | Preval et al. (2013), MAST STIS E140H\nObserved validation data — no atmosphere model fitted', fontsize=13)
        fig.savefig(directory/'observed_validation_windows.png',dpi=160)
        plt.close(fig)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/hot-daz/g191-b2b'))
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    result = prepare(args.output, args.fetch, not args.no_plot)
    print(json.dumps({k: result[k] for k in ('target','effective_temperature','logg','retained_pixels')}, indent=2))

if __name__ == '__main__':
    main()
