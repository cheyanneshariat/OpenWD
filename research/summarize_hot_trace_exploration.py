#!/usr/bin/env python3
"""Collate reproducible short G191-B2B screens without promoting their status."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CASES = [
    ('baseline', 'Baseline 40 / 54', 'g191-b2b-higher-ions', 'g191-b2b-higher-ions-comparison'),
    ('control', 'Short control 40 / 54', 'explore-control', None),
    ('op', 'OP tables, 40 / 54', 'explore-op', None),
    ('ciii', 'C III enlarged, 80 / 54', 'explore-ciii-only', None),
    ('large', 'Enlarged, 80 / 100', 'explore-large-c-continued', None),
    ('large_op', 'OP tables, 80 / 100', 'explore-large-c-op', None),
    ('iron', '80 / 100 + Fe continuum', 'explore-large-c-fe', None),
    ('larger_op', 'OP tables, 120 / 150', 'explore-larger-c-op', None),
]

COLLISION_CASES = [
    ('baseline', 'Original 40 / 54', 'g191-b2b-higher-ions', 'g191-b2b-higher-ions-comparison'),
    ('previous', 'Previous OP 120 / 150', 'explore-larger-c-op', None),
    ('control', 'OP 180 / 200, corrected', 'explore-c180-control-fixed', None),
    ('low', '180 / 200, low C III collisions', 'explore-c180-low-collisions', None),
    ('c120', '120 / 150 + CHIANTI', 'explore-c120-chianti', None),
    ('c180', '180 / 200 + CHIANTI', 'explore-c180-chianti', None),
    ('fine', '180 / 200, doubled grid', 'explore-c180-chianti-fine', None),
    ('c240', '240 / 243 + CHIANTI', 'explore-c240-chianti', None),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/hot-daz'))
    p.add_argument('--output', type=Path, default=Path('results/hot-daz/exploration-summary'))
    p.add_argument('--suite', choices=('initial','collisions'), default='initial')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from compare_hot_daz_benchmark import verify_benchmark_parameters
    cases, arrays = {}, {}
    for key, label, name, comparison in CASES if args.suite=='initial' else COLLISION_CASES:
        directory = args.root / name
        comp = args.root / comparison if comparison else directory / 'comparison'
        report = json.loads((comp / 'comparison.json').read_text())
        meta = report['model_metadata']
        verify_benchmark_parameters(meta)
        with np.load(comp / 'comparison_arrays.npz', allow_pickle=False) as data:
            arrays[key] = {k: data[k].copy() for k in data.files}
        with np.load(directory / 'spectrum.npz', allow_pickle=False) as raw:
            with fits.open(comp / 'prediction.fits', checksum=True) as h:
                assert all(x.verify_checksum() == 1 and x.verify_datasum() == 1 for x in h)
                assert bool(h[0].header['METCONV']) == bool(meta['converged'])
                np.testing.assert_array_equal(h[1].data['SURFACE_FLUX'], raw['flux'])
                np.testing.assert_array_equal(h[1].data['WAVELENGTH'], raw['wavelength'])
        with fits.open(comp / 'comparison.fits', checksum=True) as h:
            assert all(x.verify_checksum() == 1 and x.verify_datasum() == 1 for x in h)
            assert bool(h[0].header['METCONV']) == bool(meta['converged'])
            for region in report['metrics']:
                for field in ('wavelength','observed','error','model','photosphere','score_mask'):
                    np.testing.assert_array_equal(h[region.upper()].data[field.upper()],
                                                  arrays[key][region+'_'+field])
        record = dict(label=label, model_directory=str(directory), comparison_directory=str(comp),
                      metadata=meta, metrics=report['metrics'], fits_roundtrip_verified=True,
                      spectrum_sha256=report['spectrum_sha256'],
                      population_sha256=hashlib.sha256((directory/'populations.npz').read_bytes()).hexdigest())
        if (directory / 'progress.json').exists():
            record['saved_iterates'] = json.loads((directory/'progress.json').read_text())
        if (directory / 'collision-audit.json').exists():
            audit=json.loads((directory/'collision-audit.json').read_text())
            record['collision_coverage']={q:{k:v for k,v in a.items() if k not in ('level_mapping','collisions')}
                                          for q,a in audit.items()}
        cases[key] = record
    summary = dict(target='G191-B2B', date='2026-10-02', observationally_validated=False,
                   reference='https://doi.org/10.1093/mnras/stt1604',
                   scope='Fixed published parameters and abundances; short metal NLTE iterations on a fixed H/He host',
                   cases=cases,
                   limitations=['All new screens are unconverged; EW stability is not full population convergence.',
                                'Atomic completeness and realistic collision rates remain unresolved.',
                                'OP tables cover selected levels; high levels still use approximate cross-sections.',
                                'Fe test includes LTE ground bound-free opacity only; no Fe/Ni line forest or thermal feedback.',
                                'Ni continuum screen could not be run: cached Ni IV+ ionization energies and Verner fits absent.',
                                'No diffusion equilibrium or levitation-modified abundance profile was imposed.',
                                'High levels retain approximate collision/photoionization rates and an ideal metal reference.'])
    summary['suite']=args.suite
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13.6, 8.3), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1))
    cax = fig.add_subplot(grid[0, :])
    sax = fig.add_subplot(grid[1, 1])
    bax = fig.add_subplot(grid[1, 0])
    colors = {'baseline': '#31659b', 'large': '#b26812', 'large_op': '#9180b6', 'larger_op': '#158071'}
    if args.suite=='collisions':
        colors={'baseline':'#31659b','previous':'#9180b6','c180':'#158071','c240':'#b26812'}
    for ax, region, xlim in [(cax, 'ciii_1175', (1174.8,1176.6)),
                              (sax, 'siiv_1393', (1393.5,1394.12))]:
        a = arrays['baseline']; w = a[region+'_wavelength']
        ax.plot(w, a[region+'_observed'], color='.23', lw=.7, label='STIS observation')
        ax.fill_between(w, a[region+'_observed']-a[region+'_error'],
                        a[region+'_observed']+a[region+'_error'], color='.3', alpha=.15)
        ax.fill_between(w, 0, 1, where=~a[region+'_score_mask'], transform=ax.get_xaxis_transform(),
                        color='#ccb5d8', alpha=.19, step='mid')
        for key, color in colors.items():
            ax.plot(w, arrays[key][region+'_model'], color=color, lw=1.5,
                    ls='--' if key=='baseline' else '-', label=cases[key]['label'])
        ax.set(xlim=xlim, ylim=(.38,1.08) if region=='ciii_1175' else (.23,1.06),
               xlabel='Observed vacuum wavelength (Å)', ylabel='Locally normalized flux')
        ax.set_title('C III 1175 multiplet' if region=='ciii_1175' else 'Si IV 1393.75 — little change')
    cax.legend(loc='lower left', ncol=3, fontsize=9, framealpha=.95)
    keys = list(cases)
    for offset, component, color in [(-.16,'1175.987','#31659b'),(.16,'1176.370','#b26812')]:
        values = [cases[k]['metrics']['ciii_1175']['isolated_components'][component]
                  ['predicted_to_observed_ew_ratio'] for k in keys]
        bax.barh(np.arange(len(keys))+offset, values, height=.29, color=color, label=component+' Å')
    bax.axvline(1, color='.2', ls=':', lw=1)
    bax.set(yticks=np.arange(len(keys)), yticklabels=[cases[k]['label'] for k in keys],
            xlabel='Predicted / observed equivalent width', xlim=(0, 1.07))
    bax.invert_yaxis()
    bax.legend(loc='lower right', fontsize=8)
    bax.set_title('Two cleaner C III components; ±20 km/s apertures')
    fig.suptitle(('G191-B2B: carbon levels and calculated electron collisions\n' if args.suite=='collisions' else
                 'G191-B2B: screening carbon atomic physics at fixed published abundances\n')+
                 '52,500 K · log g = 7.53 · C/H = 1.72 × 10⁻⁷ · Si/H = 3.68 × 10⁻⁷', fontsize=14)
    fig.supxlabel('New models are UNCONVERGED exploratory iterates. Labels give C III / C IV level counts.\n'
                  'Gaussian STIS response; local continuum normalization; shading marks pixels excluded from scores.', fontsize=9)
    fig.savefig(args.output/'exploration.png', dpi=180)
    fig.savefig(args.output/'exploration.pdf')
    plt.close(fig)
    for key, case in cases.items():
        m = case['metrics']; c = m['ciii_1175']
        print(key, 'defect', case['metadata']['population_defect'],
              'EWs', [v['predicted_aperture_ew_mA'] for v in c['isolated_components'].values()],
              'RMS', c['normalized_rms'])


if __name__ == '__main__':
    main()
