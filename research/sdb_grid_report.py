#!/usr/bin/env python3
"""Summarize the sdB robustness grid written by research/sdb_grid_check.sh.

    python research/sdb_grid_report.py results/sdb/grid [--fixed-point]

Prints, per grid point, the LTE structure convergence and run time, the hybrid
NLTE population convergence (iterations, final change, spectral drift between
the last two chunks) and, with --fixed-point, the change of the converged
populations under one unpreconditioned statistical-equilibrium update.  It
measures LTE and NLTE line-core depths at R ~ 10,000 and writes a figure of
normalized profiles (NLTE solid, LTE dashed) across the grid.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sdb_compare_hd4539 import gaussian_convolve, vacuum_to_air

LINES = [('Hβ', 4861.33, 60), ('Hγ', 4340.47, 50), ('He I 4471', 4471.48, 10), ('He I 4922', 4921.93, 8),
         ('He I 5876', 5875.62, 8), ('He I 6678', 6678.15, 8), ('He II 4686', 4685.70, 8)]
RESOLVING_POWER = 9861.0


def normalized(wavelength, flux, centre, half):
    inside = np.abs(wavelength - centre) <= half
    w, f = wavelength[inside], flux[inside]
    edge = np.abs(w - centre) > 0.8 * half
    continuum = np.polyval(np.polyfit(w[edge] - centre, f[edge], 1), w - centre)
    return w - centre, f / continuum


def smoothed(path):
    saved = np.load(path)
    air = vacuum_to_air(saved['wavelength_vacuum'])
    keep = (air > 4200) & (air < 6800)
    return gaussian_convolve(air[keep], saved['flux'][keep], RESOLVING_POWER)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('grid', type=Path)
    parser.add_argument('--fixed-point', action='store_true')
    args = parser.parse_args()
    points = []
    for lte in sorted(args.grid.glob('*-lte'), key=lambda p: (float(re.search(r't(\d+)', p.name).group(1)), p.name)):
        tag = lte.name[:-4]
        teff, logg, he = (float(x) for x in re.match(r't(\d+)-g([\d.]+)-he([-\d.]+)', tag).groups())
        record = dict(tag=tag, teff=teff, logg=logg, he=he)
        if (lte / 'run-summary.json').exists():
            summary = json.loads((lte / 'run-summary.json').read_text())
            record.update(lte_status=summary['atmosphere_convergence_status'], lte_s=summary['elapsed_seconds'],
                          lte_iterations=summary['iterations'])
        hybrid = args.grid / f'{tag}-hybrid'
        log = args.grid / f'{tag}-hybrid.log'
        if (hybrid / 'run-summary.json').exists():
            summary = json.loads((hybrid / 'run-summary.json').read_text())
            drifts = re.findall(r'max spectral drift since previous chunk=([\de.+-]+|None)', log.read_text())
            record.update(nlte_converged=summary['populations_converged'], nlte_iterations=summary['population_iterations'],
                          nlte_change=summary['population_change'], nlte_s=summary['total_seconds'],
                          last_drift=None if not drifts or drifts[-1] == 'None' else float(drifts[-1]))
            if args.fixed_point:
                output = subprocess.run(
                    [sys.executable, 'research/sdb_fixed_point_check.py', str(lte), str(hybrid / 'populations.npz'),
                     '--data-root', 'src/wd_spectra/data/runtime'], capture_output=True, text=True).stdout
                found = re.findall(r'(He I|He II|He III|H) ([\de.+-]+)', output)
                record['fixed_point'] = {k: float(v) for k, v in found}
        points.append(record)

    print(f"{'point':26s} {'LTE':>10s} {'t_LTE':>6s} {'NLTE':>5s} {'its':>4s} {'change':>8s} {'drift':>8s} {'t_NLTE':>6s}  fixed-point (He I/He II/H)")
    for r in points:
        fp = r.get('fixed_point', {})
        print(f"{r['tag']:26s} {str(r.get('lte_status', '-')):>10s} {r.get('lte_s', 0)/60:6.1f} "
              f"{str(r.get('nlte_converged', '-')):>5s} {r.get('nlte_iterations', 0):4d} {r.get('nlte_change', float('nan')):8.1e} "
              f"{(r.get('last_drift') if r.get('last_drift') is not None else float('nan')):8.1e} {r.get('nlte_s', 0)/60:6.1f}  "
              + ('/'.join(f'{fp.get(k, float("nan")):.0e}' for k in ('He I', 'He II', 'H')) if fp else ''))

    done = [r for r in points if 'nlte_converged' in r]
    print(f"\n{'point':26s} " + ' '.join(f'{n:>15s}' for n, _, _ in LINES) + '\n' + ' ' * 27 + ' '.join(f'{"LTE  NLTE":>15s}' for _ in LINES))
    figure, axes = plt.subplots(len(LINES), 1, figsize=(7, 2.3 * len(LINES)))
    colours = plt.cm.plasma(np.linspace(0, 0.9, len(done)))
    for colour, r in zip(colours, done):
        nlte = smoothed(args.grid / f"{r['tag']}-hybrid/spectrum.npz")
        lte = smoothed(args.grid / f"{r['tag']}-hybrid/spectrum-lte.npz")
        depths = []
        for axis, (name, centre, half) in zip(axes, LINES):
            x, y = normalized(*nlte, centre, half)
            xl, yl = normalized(*lte, centre, half)
            core = np.abs(x) < 1.0
            depths.append((1 - yl[np.abs(xl) < 1.0].min(), 1 - y[core].min()))
            axis.plot(x, y, color=colour, lw=1, label=f"{r['teff']/1e3:.0f}k/{r['logg']}/{r['he']}")
            axis.plot(xl, yl, color=colour, lw=0.6, ls='--')
            axis.set_ylabel(name, fontsize=8)
        print(f"{r['tag']:26s} " + ' '.join(f'{a:7.3f}{b:8.3f}' for a, b in depths))
    axes[0].legend(fontsize=6, ncol=3)
    axes[-1].set_xlabel('Δλ (Å)')
    figure.tight_layout()
    figure.savefig(args.grid / 'grid-profiles.png', dpi=110)
    print(f"\nwrote {args.grid / 'grid-profiles.png'}")


if __name__ == '__main__':
    main()
