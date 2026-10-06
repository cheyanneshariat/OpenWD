#!/usr/bin/env python3
"""Compare sdB model spectra with the co-added HD 4539 X-shooter spectrum at fixed parameters.

    python research/sdb_compare_hd4539.py LABEL=results/sdb/hd4539-lte-hhe-40/spectrum.npz [...] \
        --output results/sdb/compare-hd4539.pdf

No stellar parameter is fitted.  Each model (vacuum wavelengths) is converted to
air, convolved with a Gaussian of FWHM = lambda/R for the arm that covers the
line, shifted by one radial velocity fitted to the first model over all line
windows, and normalized line by line: a linear continuum through the outer
edges of each window is fitted separately to the observation and to the model.
Prints the rms (observed - model) in each line core and wing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIGHT_SPEED_KMS = 299792.458
OBS = Path('results/sdb/obs/hd4539')
RESOLVING_POWER = {'uvb': 9861.0, 'vis': 18340.0}
# (label, air centre, half window); He I windows avoid neighbouring Balmer cores where possible.
LINES = [('H9', 3835.39, 18), ('H8', 3889.05, 20), ('Hε', 3970.07, 22), ('Hδ', 4101.74, 45),
         ('Hγ', 4340.47, 55), ('Hβ', 4861.33, 70), ('Hα', 6562.80, 60),
         ('He I 4026', 4026.19, 10), ('He I 4388', 4387.93, 8), ('He I 4471', 4471.48, 12),
         ('He I 4713', 4713.15, 8), ('He I 4922', 4921.93, 10), ('He I 5016', 5015.68, 8),
         ('He I 5876', 5875.62, 10), ('He I 6678', 6678.15, 10)]
EDGE_FRACTION = 0.2


def vacuum_to_air(wavelength):
    s2 = (1e4 / wavelength) ** 2
    return wavelength / (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def gaussian_convolve(wavelength, flux, resolving_power):
    # Uniform grid in log(lambda) so that a constant R is a constant kernel width.
    log_grid = np.arange(np.log(wavelength[0]), np.log(wavelength[-1]), 1.0 / (resolving_power * 10))
    sampled = np.interp(log_grid, np.log(wavelength), flux)
    sigma = 10 / (2 * np.sqrt(2 * np.log(2)))
    half = int(np.ceil(5 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(sampled, kernel, mode='same')
    return np.exp(log_grid), smoothed


def load_observation(directory=OBS):
    arms = {}
    for arm in ('uvb', 'vis'):
        data = fits.getdata(Path(directory) / f'coadd-{arm}.fits', 1)
        good = np.isfinite(data['FLUX']) & np.isfinite(data['ERR'])
        arms[arm] = (data['WAVE'][good], data['FLUX'][good], data['ERR'][good])
    return arms


def arm_for(centre):
    return 'uvb' if centre < 5500 else 'vis'


def linear_continuum(wavelength, flux, centre, half):
    offset = np.abs(wavelength - centre)
    edge = offset > (1 - EDGE_FRACTION) * half
    coefficients = np.polyfit(wavelength[edge] - centre, flux[edge], 1)
    return np.polyval(coefficients, wavelength - centre)


def prepared_model(path):
    saved = np.load(path)
    air = vacuum_to_air(saved['wavelength_vacuum'])
    return {arm: gaussian_convolve(air, saved['flux'], power) for arm, power in RESOLVING_POWER.items()}


def normalized_window(observation, model, centre, half, velocity):
    arm = arm_for(centre)
    wave, flux, error = observation[arm]
    inside = np.abs(wave - centre) <= half
    wave, flux, error = wave[inside], flux[inside], error[inside]
    model_wave, model_flux = model[arm]
    shifted = np.interp(wave, model_wave * (1 + velocity / LIGHT_SPEED_KMS), model_flux)
    observed_continuum = linear_continuum(wave, flux, centre, half)
    model_continuum = linear_continuum(wave, shifted, centre, half)
    return wave, flux / observed_continuum, error / observed_continuum, shifted / model_continuum


def fit_velocity(observation, model):
    velocities = np.arange(-60.0, 60.01, 0.5)
    chi2 = []
    for velocity in velocities:
        total = 0.0
        for _, centre, half in LINES:
            _, flux, error, shifted = normalized_window(observation, model, centre, min(half, 8.0), velocity)
            total += np.sum(((flux - shifted) / error) ** 2)
        chi2.append(total)
    return float(velocities[int(np.argmin(chi2))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', help='LABEL=path/to/spectrum.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observation = load_observation()
    models = []
    for item in args.models:
        label, path = item.split('=', 1)
        models.append((label, prepared_model(path)))
    velocity = fit_velocity(observation, models[0][1])
    print(f'radial velocity (fitted to {models[0][0]} line cores): {velocity:+.1f} km/s')

    columns = 5
    rows = int(np.ceil(len(LINES) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.0 * rows))
    header = 'line          ' + ''.join(f'{label:>22s}' for label, _ in models)
    print(header + '\n' + ' ' * 14 + ''.join(f'{"core rms   wing rms":>22s}' for _ in models))
    for axis, (name, centre, half) in zip(axes.ravel(), LINES):
        text = f'{name:13s} '
        for index, (label, model) in enumerate(models):
            wave, flux, error, shifted = normalized_window(observation, model, centre, half, velocity)
            if index == 0:
                axis.plot(wave - centre, flux, color='k', lw=0.6, label='X-shooter')
            axis.plot(wave - centre, shifted, lw=1.0, label=label)
            core = np.abs(wave - centre) < 0.25 * half
            wing = (~core) & (np.abs(wave - centre) < (1 - EDGE_FRACTION) * half)
            text += f'{np.sqrt(np.mean((flux - shifted)[core] ** 2)):11.4f}{np.sqrt(np.mean((flux - shifted)[wing] ** 2)):11.4f}'
        print(text)
        axis.set_title(name, fontsize=9)
        axis.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7)
    for axis in axes.ravel()[len(LINES):]:
        axis.axis('off')
    figure.suptitle(f'HD 4539, fixed parameters (Teff 23200 K, log g 5.20, log He/H -2.27); RV {velocity:+.1f} km/s',
                    fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix('.png'), dpi=110)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
