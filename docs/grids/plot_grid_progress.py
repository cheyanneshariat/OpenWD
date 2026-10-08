"""Reproduce the progress figure from the compact status table."""
import argparse

from collections import Counter

import csv

import hashlib

import json

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ORDER = ('DA', 'DB', 'DZ', 'DAO', 'DO', 'DAB', 'DAZ', 'DQ', 'D6', 'DAH', 'PG1159')

DISPLAY_ORDER = ORDER[:8]

STYLES = {
    'completed': ('#237a57', 'o', 'Completed'),
    'numerical_failure': ('#ba3838', 'x', 'Numerical check failed'),
    'timed_out': ('#8d8d8d', 's', 'Timed out'),
    'unsupported': ('#a5a5a5', '^', 'Unsupported'),
    'not_started': ('#bcbcbc', 'o', 'Not started'),
    'running': ('#8d8d8d', '>', 'Running'),
    'unfinished': ('#bcbcbc', 'D', 'Awaiting qualification'),
}

def panel(ax, rows, family):
    subset = [r for r in rows if r['family'] == family]
    counts = Counter(r['plot_status'] for r in subset)
    dense = len(subset) > 100
    layers = ('not_started', 'unsupported', 'unfinished', 'running', 'timed_out', 'completed', 'numerical_failure')
    for key in layers:
        points = sorted({(float(r['teff_K']), float(r['logg']))
                         for r in subset if r['plot_status'] == key})
        if not points:
            continue
        color, marker, _ = STYLES[key]
        options = dict(marker=marker, zorder=layers.index(key)+1)
        if key == 'timed_out':
            options.update(s=48 if dense else 100, facecolors='none', edgecolors=color, linewidths=1.2)
        elif key == 'numerical_failure':
            options.update(s=25 if dense else 58, color=color, linewidths=1.1)
        else:
            options.update(s=(13 if dense else 44) if key == 'completed' else (28 if dense else 70),
                           color=color, edgecolors='none')
        ax.scatter([p[0] for p in points], [p[1] for p in points], **options)
    ax.set_xscale('log')
    ax.set_xlim(2200, 140000)
    ax.set_ylim(4.75, 10.4)
    ax.xaxis.set_major_locator(FixedLocator([3000, 10000, 30000, 100000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _:f'{x/1000:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([5, 6, 7, 8, 9, 10])
    ax.set_xlabel(r'$T_{\rm eff}$ (kK)', labelpad=4)
    ax.set_ylabel(r'$\log g$ (cgs)', labelpad=4)
    label = 'DAB / DBA' if family == 'DAB' else family
    ax.set_title(f'{label}  ({counts["completed"]}/{len(subset)})', fontsize=15, pad=10)
    return {key: counts[key] for key in STYLES}

def main():
    parser = argparse.ArgumentParser(description="Plot a saved per-request grid status table.")
    parser.add_argument('--table', type=Path, default=Path(__file__).with_name('progress.csv'))
    parser.add_argument('--style', type=Path, default=Path(__file__).with_name('grid-style.mplstyle'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.table.open()))
    assert len(rows) == len({r['model_id'] for r in rows})
    assert all(r['plot_status'] in STYLES for r in rows)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.style.use(args.style)
    plt.rcParams.update({'axes.labelsize':13, 'xtick.labelsize':11,
                         'ytick.labelsize':11, 'axes.linewidth':.8})
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
    for ax, family in zip(axes.flat, DISPLAY_ORDER):
        panel(ax, rows, family)
    handles = [Line2D([], [], linestyle='none', marker=marker, color=color,
                      markerfacecolor='none' if key == 'timed_out' else color,
                      markersize=7, label=label)
               for key, (color, marker, label) in STYLES.items()
               if any(r['family'] in DISPLAY_ORDER and r['plot_status'] == key for r in rows)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .012),
               frameon=False, fontsize=11, ncol=len(handles), columnspacing=2)
    fig.subplots_adjust(left=.055, right=.985, bottom=.14, top=.93, wspace=.35, hspace=.4)
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'grid_progress.{ext}',dpi=250)
    plt.close(fig)


if __name__ == '__main__':
    main()
