#!/usr/bin/env python3
"""Reference line atlas: observed sdB spectrum vs OpenWD in 150 A panels, strong lines labelled.

    python research/plot_sdb_line_atlas.py lsiv14116 --maximum-defect 2e-3
    python research/plot_sdb_line_atlas.py hd4539

Uses exactly the prepared spectra of the paper figure (plot_sdb_paper_comparison.prepare),
but plots them in air wavelengths, the convention for optical line identifications.

Labels: every model absorption feature deeper than --depth (relative to the
local maximum within +-1.5 A) is identified with the strongest candidate
transition within --tolerance A among the lines the model actually contains
(H I, He I, He II; the NLTE atoms' selected transitions; the LTE line lists;
sdb_heavy_lines.py).  Candidates are ranked by an LTE strength estimate,
gf n_lower lambda at tau_Ross = 0.3; for NLTE elements this is only a ranking
proxy.  Interstellar lines are marked IS; telluric bands are light gray.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wd_spectra.hot_trace_metals import _atom_selection, fixed_electron_metal_reference
from wd_spectra.metals import read_pg1159_atomic_database
from wd_spectra.models.common import ModelData

import plot_sdb_paper_comparison as figure
from sdb_heavy_lines import HeavyLines
from sdb_trace_metals import NIST_IONIZATION_EV, load_host

ROMAN = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII')
BOLTZMANN, WAVENUMBER_TO_ERG = 1.380649e-16, 6.62607015e-27 * 2.99792458e10
# H I and He I/II optical lines (air A).
HHE = ([('H' + n, w) for n, w in (('α', 6562.79), ('β', 4861.35), ('γ', 4340.47), ('δ', 4101.74), ('ε', 3970.07),
                                    ('8', 3889.05), ('9', 3835.38), ('10', 3797.90))]
       + [('He I', w) for w in (3819.61, 3867.48, 3926.54, 3935.91, 3964.73, 4009.26, 4026.19, 4120.82, 4143.76,
                                4387.93, 4437.55, 4471.48, 4713.15, 4921.93, 5015.68, 5047.74, 5875.62, 6678.15)]
       + [('He II', w) for w in (3923.48, 4025.60, 4199.83, 4338.67, 4541.59, 4685.70, 4859.32, 5411.52, 6560.10)])


def vacuum_to_air(vacuum):
    s2 = (1e4 / np.asarray(vacuum)) ** 2
    return vacuum / (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def candidates(summary, data):
    """(air A, label, strength) for every metal line the model contains."""
    host = summary['host']
    model, atmosphere, state, _ = load_host(Path(host['lte']), Path(host['hybrid']), data)
    depth = int(np.argmin(np.abs(np.log(np.asarray(atmosphere.rosseland_optical_depth) / 0.3))))
    temperature = float(atmosphere.temperature[depth])
    out = []
    nlte, lte = summary['abundances'], summary.get('lte_abundances', {})
    elements = tuple(nlte) + tuple(lte)
    database = read_pg1159_atomic_database(data.stout, elements=elements)
    database = replace(database, ions={k: (replace(v, ionization_energy_ev=NIST_IONIZATION_EV[k])
                                           if v.ionization_energy_ev is None and k in NIST_IONIZATION_EV else v)
                                       for k, v in database.ions.items()},
                       _line_selection_cache={}, _unsold_hydrogen_coefficient_cache={})
    top = {e: min((i.charge for i in database.ion_stages(e) if i.ionization_energy_ev is None), default=None)
           for e in elements}
    database = replace(database, ions={k: v for k, v in database.ions.items() if top[k[0]] is None or k[1] <= top[k[0]]},
                       _line_selection_cache={}, _unsold_hydrogen_coefficient_cache={})
    reference = fixed_electron_metal_reference(atmosphere, database, {**nlte, **lte})
    for element in elements:
        if element in nlte:
            counts = {int(q): n for q, n in summary['levels_per_charge'][element].items()}
            selected = _atom_selection(database, element, counts)[0]
        for index, ion in enumerate(database.ion_stages(element)):
            population = reference.ion_number_density[element][index][depth]
            partition = reference.partition_function[(element, ion.charge)][depth]
            if population <= 0:
                continue
            levels = {level.index: level for level in ion.levels}
            for line in ion.transitions:
                if element in nlte and (element, ion.charge, line.lower_index, line.upper_index) not in selected:
                    continue
                if element not in nlte and line.absorption_oscillator_strength < 1e-4:
                    continue
                w = line.wavelength_vacuum_angstrom
                if not 3700 < w < 7000 or line.absorption_oscillator_strength <= 0:
                    continue
                lower = levels[line.lower_index]
                n_lower = (population * lower.statistical_weight / partition
                           * np.exp(-lower.energy_wavenumber * WAVENUMBER_TO_ERG / (BOLTZMANN * temperature)))
                out.append((float(vacuum_to_air(w)), f'{element} {ROMAN[ion.charge]}',
                            line.absorption_oscillator_strength * n_lower * w))
    heavy = summary.get('heavy_abundances', {})
    if heavy:
        lines = HeavyLines(atmosphere, heavy)
        for line in lines.lines:
            population, partition = lines.populations[(line['element'], line['charge'])]
            n_lower = (population[depth] * line['lower_weight'] / partition[depth]
                       * np.exp(-line['lower_energy'] * WAVENUMBER_TO_ERG / (BOLTZMANN * temperature)))
            out.append((float(vacuum_to_air(line['wavelength'])), f'{line["element"]} {ROMAN[line["charge"]]}',
                        10 ** line['log_gf'] / line['lower_weight'] * n_lower * line['wavelength']))
    return out


def labels_for(air, model, metals, depth_limit, tolerance):
    """[(air A, text, strength)] for model features deeper than depth_limit."""
    good = np.isfinite(model)
    w, m = air[good], model[good]
    step = np.median(np.diff(w))
    half = max(2, int(1.5 / step))
    found = []
    for i in range(1, len(m) - 1):
        if m[i] < m[i - 1] and m[i] <= m[i + 1]:
            local = m[max(0, i - half):i + half + 1].max()
            if local - m[i] >= depth_limit:
                found.append((w[i], local - m[i]))
    metal_w = np.array([c[0] for c in metals]) if metals else np.empty(0)
    labels = []
    for position, depth in found:
        hhe = [(abs(position - x), name, x) for name, x in HHE if abs(position - x) < max(tolerance, 1.0)]
        if hhe:
            _, name, x = min(hhe)
            labels.append((x, f'{name} {x:.1f}' if name.startswith('He') else name, 1e9 + depth))
            continue
        near = np.flatnonzero(np.abs(metal_w - position) < tolerance)
        if near.size:
            best = max(near, key=lambda k: metals[k][2])
            labels.append((metals[best][0], f'{metals[best][1]} {metals[best][0]:.1f}', depth))
    # One label per transition; drop exact duplicates.
    unique = {}
    for x, text, strength in labels:
        if text not in unique or strength > unique[text][2]:
            unique[text] = (x, text, strength)
    return sorted(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', choices=figure.ORDER)
    parser.add_argument('--models', type=Path, default=figure.ROOT / 'results/sdb/paper')
    parser.add_argument('--output', type=Path, default=figure.ROOT / 'results/sdb/paper/figure')
    parser.add_argument('--maximum-defect', type=float)
    parser.add_argument('--width', type=float, default=150.0)
    parser.add_argument('--depth', type=float, default=0.02, help='minimum labelled model depth')
    parser.add_argument('--tolerance', type=float, help='A; default 0.3 (UVES) or 0.6 (X-shooter)')
    parser.add_argument('--per-page', type=int, default=5)
    args = parser.parse_args()
    normalize, _ = figure.paper_normalizer()
    prepared = figure.prepare(args.target, args.models, normalize, args.maximum_defect)
    target, summary, host = prepared['target'], prepared['summary'], prepared['host']
    tolerance = args.tolerance or (0.3 if target['instrument'] == 'UVES' else 0.6)
    air = vacuum_to_air(prepared['wave'])
    data = ModelData.default(figure.ROOT / 'src/wd_spectra/data/runtime')
    labels = labels_for(air, prepared['model'], candidates(summary, data), args.depth, tolerance)
    interstellar = target.get('interstellar', ())
    plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'font.size': 9,
                         'axes.linewidth': .7, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    starts = np.arange(3800.0, 6800.0, args.width)
    finite = np.isfinite(prepared['observed'])
    starts = [s for s in starts if np.any(finite & (air >= s) & (air < s + args.width))]
    pages = [starts[i:i + args.per_page] for i in range(0, len(starts), args.per_page)]
    title = (f'{target["name"]} ({target["instrument"]}): Teff {host["teff"]:.0f} K, log g {host["logg"]:.2f}, '
             f'log He/H {host["log_he_h"]:.2f} [{target["source"]}]\nGray: observed (light gray: telluric); '
             'orange: OpenWD.  Labels: model features (air A); IS: interstellar.')
    name = f'{args.target}_line_atlas'
    with PdfPages(args.output / f'{name}.pdf') as pdf:
        for number, page in enumerate(pages, 1):
            fig, axes = plt.subplots(args.per_page, 1, figsize=(8.5, 11), squeeze=False)
            fig.subplots_adjust(left=.07, right=.985, top=.945, bottom=.045, hspace=.42)
            fig.suptitle(title, fontsize=7.5, x=.07, ha='left', y=.992, va='top')
            for ax, start in zip(axes[:, 0], page + [None] * (args.per_page - len(page))):
                if start is None:
                    ax.axis('off'); continue
                ax.plot(air, prepared['clean'], color='.30', lw=.55, rasterized=True)
                ax.plot(air, prepared['telluric'], color='.75', lw=.55, rasterized=True)
                ax.plot(air, prepared['model'], color='#d95f02', lw=.65)
                ax.set_xlim(start, start + args.width); ax.set_ylim(0.25, 1.32)
                ax.set_yticks((.5, .75, 1.0))
                ax.tick_params(top=True, right=True, length=3, labelsize=8)
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
                ax.tick_params(which='minor', length=1.5, top=True)
                inside = [lab for lab in labels if start <= lab[0] <= start + args.width]
                # Strongest first; skip a label closer than 1.1 A to one already placed.
                placed = []
                for x, text, strength in sorted(inside, key=lambda lab: -lab[2]):
                    if all(abs(x - p) > 1.1 for p in placed):
                        placed.append(x)
                        ax.plot((x, x), (1.045, 1.075), color='.4', lw=.4)
                        ax.text(x, 1.08, text, rotation=90, ha='center', va='bottom', fontsize=4.6, color='.15')
                for line in interstellar:  # at the bottom, clear of the line labels
                    if start <= line <= start + args.width:
                        ax.plot((line, line), (0.27, 0.31), color='tab:blue', lw=.5)
                        ax.text(line, 0.32, 'IS', ha='center', va='bottom', fontsize=5, color='tab:blue')
            axes[-1, 0].set_xlabel('air wavelength [Å]', fontsize=9)
            fig.supylabel('normalized flux', fontsize=9, x=.012)
            pdf.savefig(fig, dpi=400)  # resolution of the rasterized observed spectrum
            fig.savefig(args.output / f'{name}_p{number}.png', dpi=170)
            plt.close(fig)
    (args.output / f'{name}-labels.json').write_text(json.dumps(
        [dict(air_angstrom=round(x, 3), label=t, model_depth=round(s % 1e9, 4)) for x, t, s in labels], indent=1) + '\n')
    print(args.output / f'{name}.pdf', f'{len(pages)} pages, {len(labels)} labelled features')


if __name__ == '__main__':
    main()
