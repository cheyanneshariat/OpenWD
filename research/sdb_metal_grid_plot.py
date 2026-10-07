#!/usr/bin/env python3
"""Plot and tabulate the typical-sdB metal grid (sdb_metal_grid.sh).

    python research/sdb_metal_grid_plot.py results/sdb/grid --output results/sdb/grid/metal-lines.png

Each panel overlays one metal-line window for every finished <tag>-metals model,
continuum-normalized by a straight line through the window edges, at R = 20,000.  Colours follow sdb_grid_plot_spectra.py (Teff
slot; the two 28 kK He-abundance probes have their own); solid/dotted = lower/
higher log g.  Also prints convergence and NLTE-metal equivalent widths (mA) per model.
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

from wd_spectra._compat import trapezoid

from sdb_compare_hd4539 import gaussian_convolve, vacuum_to_air
from sdb_grid_plot_spectra import SERIES, INK, INK_MUTED, GRID

# (title, air centre, half width, EW lines (air) measured over +-0.6 A)
WINDOWS = [('C II 4267', 4267.3, 2.5, (4267.26,)), ('C III + O II 4647-51', 4649.0, 4.5, (4647.42, 4649.13, 4650.25)),
           ('N II 3995', 3995.0, 2.5, (3994.99,)), ('N III 4097 (Hδ wing)', 4097.4, 2.5, (4097.36,)),
           ('O II 4414/17', 4416.0, 3.5, (4414.90, 4416.97)), ('Si III 4552/68/75', 4564.0, 13.0, (4552.62, 4567.84, 4574.76)),
           ('Si IV 4089 (Hδ wing)', 4088.9, 2.5, (4088.86,)), ('Si IV 4116 (Hδ wing)', 4116.1, 2.5, (4116.10,)),
           ('S III 4254', 4253.6, 2.5, (4253.59,)), ('Mg II 4481 (LTE)', 4481.2, 2.5, (4481.20,)),
           ('Al III 4512/29 (LTE)', 4520.5, 10.5, (4512.57, 4529.19)), ('Fe III 4164 (LTE)', 4164.7, 2.5, (4164.73,))]
EW_EXTRA = {'S III 4285': 4284.98, 'C III 5696': 5695.92, 'C II 6578': 6578.05}


def finished_models(grid):
    found = []
    for summary in grid.glob('*-metals/run-summary.json'):
        tag = summary.parent.name[:-7]
        teff, logg, he = (float(x) for x in re.match(r't(\d+)-g([\d.]+)-he([-\d.]+)', tag).groups())
        found.append((teff, logg, he, summary.parent))
    return sorted(found)


def styles(models):
    track = sorted({t for t, _, he, _ in models if t != 28000.0})
    colour, style = {}, {}
    for teff, logg, he, _ in models:
        colour[(teff, he)] = (SERIES[5] if he < -2.5 else SERIES[6]) if teff == 28000.0 else SERIES[track.index(teff)]
        gravities = sorted(g for t, g, h, _ in models if (t, h) == (teff, he))
        style[(teff, logg, he)] = '-' if logg == gravities[0] else ':'
    return colour, style


def normalized(wave, flux, centre, half):
    inside = np.abs(wave - centre) <= half
    edge = np.abs(wave[inside] - centre) > 0.85 * half
    continuum = np.polyval(np.polyfit(wave[inside][edge] - centre, flux[inside][edge], 1), wave[inside] - centre)
    return wave[inside], flux[inside] / continuum


def equivalent_width_mA(wave, flux, background, line):
    """EW against the fixed background (H/He host plus LTE Mg/Al/Fe), which removes the H/He
    line under a metal line (NLTE-metal lines only: the LTE Mg/Al/Fe lines are in it)."""
    window = np.abs(wave - line) <= 0.6
    return 1e3 * trapezoid(1.0 - flux[window] / background[window], wave[window])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('grid', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models = finished_models(args.grid)
    if not models:
        raise SystemExit('no finished metal models')
    colour, style = styles(models)
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK_MUTED, 'axes.labelcolor': INK,
                         'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED})
    figure, axes = plt.subplots(3, 4, figsize=(13, 9))
    lines = sorted({x for w in WINDOWS if '(LTE)' not in w[0] for x in w[3]} | set(EW_EXTRA.values()))
    rows = []
    for teff, logg, he, directory in models:
        saved = np.load(directory / 'spectrum.npz')
        air = vacuum_to_air(saved['wavelength_vacuum'])
        summary = json.loads((directory / 'run-summary.json').read_text())
        rows.append((teff, logg, he, summary['converged'], summary['iterations'], summary['population_defect'],
                     summary['elapsed_seconds'],
                     [equivalent_width_mA(air, saved['flux'], saved['host_flux'], x) for x in lines]))
        label = f'{teff/1e3:.0f} kK, log g {logg:.1f}, log He/H {he:+.1f}'
        for axis, (name, centre, half, _) in zip(axes.ravel(), WINDOWS):
            lo, hi = centre - half - 3, centre + half + 3
            keep = (air > lo) & (air < hi)
            wave, flux = normalized(*gaussian_convolve(air[keep], saved['flux'][keep], 20000.0), centre, half)
            axis.plot(wave - centre, flux, color=colour[(teff, he)], ls=style[(teff, logg, he)], lw=1.1, label=label)
    for axis, (name, centre, half, positions) in zip(axes.ravel(), WINDOWS):
        for position in positions:
            axis.axvline(position - centre, color=GRID, lw=0.8, zorder=0)
        axis.set_title(name, color=INK, fontsize=9, loc='left')
        axis.set_xlabel(f'Δλ from {centre:.1f} Å')
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle('sdB grid with typical metals (Geier 2013 medians; NLTE C N O Si S, LTE Mg Al Fe), '
                    'no blanketing, R = 20,000', fontsize=10, color=INK)
    figure.tight_layout()
    figure.savefig(args.output, dpi=130, bbox_inches='tight')

    print(f"{'model':28s} conv  its  defect   min   " + ' '.join(f'{x:7.1f}' for x in lines))
    for teff, logg, he, converged, iterations, defect, seconds, ews in rows:
        print(f'{teff:5.0f} {logg:4.1f} {he:+5.1f}            {"y" if converged else "N"} {iterations:4d} {defect:8.1e} '
              f'{seconds/60:5.1f} ' + ' '.join(f'{w:7.0f}' for w in ews))
    print(f'wrote {args.output} ({len(models)} models); EWs in mA against the fixed background, +-0.6 A')


if __name__ == '__main__':
    main()
