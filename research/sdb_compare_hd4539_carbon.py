#!/usr/bin/env python3
"""Compare sdB carbon NLTE models with the HD 4539 X-shooter co-add (fixed parameters, no fit).

    python research/sdb_compare_hd4539_carbon.py LABEL=results/sdb/hd4539-carbon/spectrum.npz [...] \
        --output results/sdb/compare-hd4539-carbon.png

Uses the conventions of sdb_compare_hd4539.py (air wavelengths, instrumental
convolution per arm, RV -3.0 km/s, linear continuum from the window edges).
The first model's host-only spectrum (no carbon) is overplotted dotted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sdb_compare_hd4539 import (load_observation, normalized_window, gaussian_convolve, vacuum_to_air,
                                RESOLVING_POWER, fit_velocity)

# (title, window centre, half width, line positions measured).  Blended pairs
# share one window so the continuum comes from beyond both lines.
WINDOWS = [('C II 3918/20', 3919.8, 7, (3918.97, 3920.68)), ('C II 4267', 4267.26, 6, (4267.26,)),
           ('C II 6578/83', 6580.5, 9, (6578.05, 6582.88)), ('C II 4074/76', 4075.4, 6, (4074.52, 4075.85)),
           ('C III 4647-51', 4649.5, 7, (4647.42, 4650.25)), ('C III 5696', 5695.92, 5, (5695.92,)),
           ('C III 4069', 4068.92, 5, (4068.92,)), ('C III 4187', 4186.90, 5, (4186.90,))]
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
VELOCITY = -3.0


def prepared(wavelength, flux):
    air = vacuum_to_air(wavelength)
    return {arm: gaussian_convolve(air, flux, power) for arm, power in RESOLVING_POWER.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', help='LABEL=spectrum.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observation', type=Path, default=Path('results/sdb/obs/hd4539'))
    parser.add_argument('--title', default='HD 4539 carbon lines, fixed Teff 23200 K, log g 5.20, log He/H -2.27')
    parser.add_argument('--velocity', type=float, help='km/s; default: fitted to the H/He lines of the host')
    args = parser.parse_args()
    observation = load_observation(args.observation)
    models = []
    for item in args.models:
        label, path = item.split('=', 1)
        saved = np.load(path)
        models.append((label, prepared(saved['wavelength_vacuum'], saved['flux'])))
        if len(models) == 1:
            host = prepared(saved['wavelength_vacuum'], saved['host_flux'])
    velocity = fit_velocity(observation, host) if args.velocity is None else args.velocity
    print(f'radial velocity {velocity:+.1f} km/s')
    figure, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    print(f"{'window':14s} {'observed':>9s} " + ' '.join(f'{label[:16]:>16s}' for label, _ in models) + '   (core depth)')
    for axis, (name, centre, half, positions) in zip(axes.ravel(), WINDOWS):
        wave, flux, error, host_flux = normalized_window(observation, host, centre, half, velocity)
        axis.plot(wave - centre, flux, color='#0b0b0b', lw=0.7, label='X-shooter')
        axis.plot(wave - centre, host_flux, color='#52514e', lw=0.8, ls=':', label='H/He host only')
        spectra = []
        for (label, model), colour in zip(models, COLOURS):
            _, _, _, shifted = normalized_window(observation, model, centre, half, velocity)
            axis.plot(wave - centre, shifted, color=colour, lw=1.1, label=label)
            spectra.append(shifted)
        for position in positions:
            core = np.abs(wave - position) < 0.4
            if not core.any():
                print(f'{name[:9]} {position:7.1f}  (no usable observed pixels)')
                continue
            print(f'{name[:9]} {position:7.1f} {1 - flux[core].min():9.3f} '
                  + ' '.join(f'{1 - m[core].min():16.3f}' for m in spectra))
        axis.set_title(name, fontsize=9, loc='left')
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False)
    figure.suptitle(f'{args.title}; RV {velocity:+.1f} km/s', fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output, dpi=110)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
