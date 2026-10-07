#!/usr/bin/env python3
"""Plot the optical spectra of finished sdB grid models (hybrid NLTE H/He).

    python research/sdb_grid_plot_spectra.py results/sdb/grid --output results/sdb/grid/spectra-so-far.png
    python research/sdb_grid_plot_spectra.py results/sdb/grid --models metals --output results/sdb/grid/spectra-metals.png

Top: continuum-normalized 3700-7000 A spectra (default R = 3000), offset vertically,
direct-labelled.  Bottom: key lines at R = 10,000 overlaid (NLTE populations).
``--models metals`` plots the <tag>-metals models of sdb_metal_grid.sh instead
(typical sdB metals on the same hosts), with metal-line zooms.
Colour encodes Teff (fixed slot per temperature; the two He-abundance
points get their own slots); line style encodes log g (solid: lower,
dotted: higher at that Teff).  The pseudo-continuum is a smoothed upper envelope of the
model itself (display only; it dips slightly in the high Balmer series).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from sdb_compare_hd4539 import gaussian_convolve, vacuum_to_air

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
METAL_ZOOMS = [('C II 4267', 4267.3, 3), ('He I 4471 / Mg II 4481', 4476.0, 9), ('Si III 4552-75', 4564.0, 14),
               ('N II / O II / C III 4630-51', 4641.0, 13), ('He II 4686', 4685.70, 6)]
ZOOMS = [('Hβ', 4861.33, 40), ('He I 4471', 4471.48, 8), ('He II 4686', 4685.70, 6),
         ('He I 5876', 5875.62, 6), ('He I 6678', 6678.15, 6)]


def continuum_normalized(wavelength, flux, window_angstrom=180.0):
    step = np.median(np.diff(wavelength))
    width = max(3, int(window_angstrom / step))
    envelope = uniform_filter1d(maximum_filter1d(flux, width), width)
    return flux / envelope


def load(path, resolving_power, lo=3650.0, hi=7100.0):
    saved = np.load(path)
    air = vacuum_to_air(saved['wavelength_vacuum'])
    keep = (air > lo) & (air < hi)
    return gaussian_convolve(air[keep], saved['flux'][keep], resolving_power)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('grid', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', choices=['hybrid', 'metals'], default='hybrid')
    parser.add_argument('--resolving-power', type=float, default=3000.0, help='overview panel')
    args = parser.parse_args()
    zooms = ZOOMS if args.models == 'hybrid' else METAL_ZOOMS
    finished, skipped = [], []
    for summary in args.grid.glob(f'*-{args.models}/run-summary.json'):
        tag = summary.parent.name[:-len(args.models) - 1]
        if json.loads(summary.read_text()).get('converged', True) is False:
            skipped.append(tag)
            continue
        teff, logg, he = (float(x) for x in re.match(r't(\d+)-g([\d.]+)-he([-\d.]+)', tag).groups())
        finished.append((teff, logg, he, summary.parent))
    finished.sort()
    if not finished:
        raise SystemExit('no finished grid models')
    # Fixed colour per Teff (EHB track points share log He/H by Teff); the 28 kK
    # He-abundance probes get slots of their own, keyed by abundance.
    track = sorted({t for t, _, he, _ in finished if t != 28000.0})
    colour_of, style_of = {}, {}
    for teff, logg, he, directory in finished:
        key = (teff, he)
        if teff == 28000.0:
            colour_of[key] = SERIES[5] if he < -2.5 else SERIES[6]
        else:
            colour_of[key] = SERIES[track.index(teff)]
        gravities = sorted(g for t, g, h, _ in finished if (t, h) == key)
        style_of[(teff, logg, he)] = '-' if logg == gravities[0] else ':'

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_MUTED, 'axes.labelcolor': INK,
                         'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED})
    figure = plt.figure(figsize=(11, 4.0 + 0.55 * len(finished)))
    grid = figure.add_gridspec(2, len(zooms), height_ratios=[1.6 + 0.25 * len(finished), 1.0], hspace=0.42, wspace=0.28)
    overview = figure.add_subplot(grid[0, :])
    offset = 0.45
    for index, (teff, logg, he, directory) in enumerate(finished):
        colour = colour_of[(teff, he)]
        wave, flux = load(directory / 'spectrum.npz', args.resolving_power)
        shift = offset * (len(finished) - 1 - index)
        overview.plot(wave, continuum_normalized(wave, flux) + shift, color=colour, lw=1.0)
        overview.text(7120, 1.0 + shift, f'{teff/1e3:.0f} kK, log g {logg:.1f}, log He/H {he:+.1f}',
                      color=INK, fontsize=8, va='center')
    markers = [('Hδ', 4101.7), ('Hγ', 4340.5), ('Hβ', 4861.3), ('Hα', 6562.8), ('He I 4026', 4026.2),
               ('4471', 4471.5), ('He II 4686', 4685.7), ('4922', 4921.9), ('5876', 5875.6), ('6678', 6678.2)]
    for name, centre in markers:
        overview.axvline(centre, color=GRID, lw=0.8, zorder=0)
        overview.text(centre, 1.12 + offset * (len(finished) - 1), name, rotation=90, fontsize=7,
                      color=INK_MUTED, ha='center', va='bottom')
    overview.set_xlim(3700, 7000)
    overview.set_ylim(0.35, 1.12 + offset * (len(finished) - 1) + 0.45)
    overview.set_xlabel('Air wavelength (Å)')
    overview.set_ylabel('Normalized flux + offset')
    overview.set_title(('sdB grid, hybrid NLTE H+He (LTE structure + MALI populations)' if args.models == 'hybrid' else
                        'sdB grid, hybrid NLTE H+He + typical metals (Geier 2013 medians; NLTE C N O Si S, '
                        'LTE Mg Al Fe; no blanketing)') + f', R = {args.resolving_power:.0f}',
                       color=INK, fontsize=10, loc='left')
    for side in ('top', 'right'):
        overview.spines[side].set_visible(False)

    for column, (name, centre, half) in enumerate(zooms):
        axis = figure.add_subplot(grid[1, column])
        for index, (teff, logg, he, directory) in enumerate(finished):
            colour = colour_of[(teff, he)]
            for filename, style, width in (('spectrum.npz', style_of[(teff, logg, he)], 1.2),):
                wave, flux = load(directory / filename, 10000.0, centre - 2 * half, centre + 2 * half)
                inside = np.abs(wave - centre) <= half
                edge = np.abs(wave[inside] - centre) > 0.8 * half
                continuum = np.polyval(np.polyfit(wave[inside][edge] - centre, flux[inside][edge], 1),
                                       wave[inside] - centre)
                axis.plot(wave[inside] - centre, flux[inside] / continuum, color=colour, ls=style, lw=width,
                          label=f'{teff/1e3:.0f} kK, log g {logg:.1f}, log He/H {he:+.1f}')
        axis.set_title(name, color=INK, fontsize=9, loc='left')
        axis.ticklabel_format(useOffset=False)
        low, high = axis.get_ylim()
        if high - low < 0.02:  # line essentially absent: keep a readable 2% window
            axis.set_ylim(0.98, 1.005)
        axis.set_xlabel('Δλ (Å)')
        axis.grid(color=GRID, lw=0.5)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    figure.axes[1].set_ylabel('Normalized flux')
    handles, labels = figure.axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, frameon=False,
                  bbox_to_anchor=(0.5, -0.06))
    figure.text(0.99, 0.005, 'line zooms: NLTE populations, R = 10,000; colour = Teff, solid/dotted = lower/higher log g',
                ha='right', fontsize=7, color=INK_MUTED)
    if skipped:
        figure.text(0.01, 0.005, 'not converged, omitted: ' + ', '.join(sorted(skipped)), ha='left', fontsize=7,
                    color=INK_MUTED)
    figure.savefig(args.output, dpi=130, bbox_inches='tight')
    print(f'wrote {args.output} ({len(finished)} models; omitted unconverged: {sorted(skipped)})')


if __name__ == '__main__':
    main()
