"""Plot verified saved spectra and interpolation predictions; no model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--style', type=Path, required=True)
a = ap.parse_args()
R, O = a.inputs, a.output
O.mkdir(exist_ok=False)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plt.style.use(a.style)
BLUE, ORANGE = '#2b628c', '#c57535'


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(O / (name + '.' + ext), dpi=250)
    plt.close(fig)


d = read(R / 'DAB_INTERPOLATION.json')
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), layout='constrained')
for ax, metric, label, limit in [
        (axes[0], 'integrated_absolute_error_fraction', 'Integrated absolute error', 1),
        (axes[1], 'maximum_relative_error_fraction', 'Maximum relative error', 3)]:
    values = [limit]
    for method, color, marker, text in [('linear_F_Tg', BLUE, 'o', 'Linear flux'),
            ('pchip_logF_local_T_then_g', ORANGE, '^', 'Log-flux PCHIP')]:
        x = [100 * r['metrics']['coarse_' + method][metric] for r in d['records']]
        y = [100 * r['metrics']['refined_' + method][metric] for r in d['records']]
        values.extend(x + y)
        ax.scatter(x, y, color=color, marker=marker, s=50, label=text, zorder=3)
    lo, hi = min(values) * .7, max(values) * 1.3
    ax.plot([lo, hi], [lo, hi], color='.65', lw=.9, zorder=0)
    ax.axhline(limit, color='.4', ls=':', lw=1, zorder=0)
    ax.axvline(limit, color='.4', ls=':', lw=1, zorder=0)
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi),
        xlabel='Coarse grid error (%)', ylabel='Refined grid error (%)', title=label)
axes[0].legend(fontsize=11, loc='upper left')
fig.suptitle(r'Four independent DAB spectra; $\log(n_{\rm H}/n_{\rm He})=-3$', fontsize=14)
save(fig, 'dab_interpolation')

pairs = [x for x in read(R / 'NATIVE_PAIRS.json')['records'] if x['family'] in ['DZ', 'DAZ']]
fig, axes = plt.subplots(5, 2, figsize=(10.8, 12.5), layout='constrained')
for row, (left, right) in zip(pairs, axes):
    with np.load(R / row['arrays_file'], allow_pickle=False) as z:
        w, old, new = z['old_wavelength_angstrom'], z['old_F_lambda'], z['new_F_lambda']
        assert np.array_equal(w, z['new_wavelength_angstrom'])
    lo, hi = (1150, 3000) if row['family'] == 'DZ' else (3000, 10000)
    m = (w >= lo) & (w <= hi)
    left.plot(w[m], old[m] / 1e8, color=ORANGE, lw=1, label='Previous accepted')
    left.plot(w[m], new[m] / 1e8, color=BLUE, lw=.9, label='00503a0')
    left.set_ylim(bottom=0)
    left.set_title(f"{row['family']}, {row['config']['effective_temperature']:g} K, log g = {row['config']['logg']:g}", loc='left', fontsize=12)
    right.axhline(0, color='.6', lw=.8)
    right.plot(w[m], 100 * (new[m] / old[m] - 1), color=BLUE, lw=.8)
    left.set_ylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)', fontsize=11)
    right.set_ylabel('Flux change (%)', fontsize=12)
    for ax in [left, right]:
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)', fontsize=12)
        ax.tick_params(labelsize=11)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False, fontsize=12)
save(fig, 'cold_metal_changes')

hot = read(R / 'HOT_SPECTRA.json')['records']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
for row, color, ls in zip(hot, [BLUE, ORANGE, '#607c52', '#775a88'], ['-', '--', '-.', ':']):
    with np.load(R / row['arrays_file'], allow_pickle=False) as z:
        w, f = z['wavelength_angstrom'], z['F_lambda']
    for ax, lo, hi in [(axes[0], 1150, 3000), (axes[1], 3000, 10000)]:
        m = (w >= lo) & (w <= hi)
        ax.plot(w[m], f[m] / 1e9, color=color, ls=ls, lw=1,
            label=f"{row['family']} {row['config']['effective_temperature']/1000:g} kK")
        ax.set(xlim=(lo, hi), yscale='log', xlabel=r'Vacuum wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel(r'$F_\lambda$ ($10^9$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)', fontsize=12)
axes[0].legend(fontsize=11, loc='lower left')
fig.suptitle('Completed hot models at log g = 8', fontsize=14)
save(fig, 'hot_spectra')

(O / 'CAPTIONS.md').write_text('''DAB interpolation: Four independent spectra test one cell from 15,000 to 20,000 K and log g 7 to 7.5, at log H/He = −3. The coarse mesh uses four corners; refinement uses nine input spectra. Circles show linear flux interpolation; triangles show log-flux PCHIP. The diagonal marks equal errors. Dotted lines mark the 1% integrated and 3% maximum error targets over 3000–10000 Å. Refinement reduces errors, but none of the four targets passes both limits. This cell does not establish whole-grid accuracy. Only band endpoints are evaluated linearly for the historical error metric; saved spectra and predictions retain their native wavelengths.

Cold metal changes: The five fresh DZ/DAZ anchors use the same requested configurations as their accepted predecessors. Orange shows the predecessor; blue shows source 00503a0. Left panels show absolute surface flux; right panels show fractional changes. DZ panels cover the ultraviolet and DAZ panels the optical. All native samples are retained. The comparison includes all intervening source changes and does not isolate one change to the code or atomic data.

Hot spectra: Four completed DO/DAO models passed the native structure, source and population-convergence checks. The DAO model has log H/He = 2. Curves show absolute surface flux on native vacuum wavelengths, without normalization or resampling. The DAO model at 40,000 K timed out and is excluded. These numerical checks do not establish depth convergence or observational accuracy.
''')
(O / 'FIGURE_INPUTS.json').write_text(json.dumps(dict(
    input_reports={n: sha(R / n) for n in ['DAB_INTERPOLATION.json', 'NATIVE_PAIRS.json', 'HOT_SPECTRA.json']},
    figures={p.name: sha(p) for p in O.iterdir() if p.suffix in ['.png', '.pdf']},
    actual_png_inspection_pending=True, actual_pdf_inspection_pending=True), indent=2) + '\n')
