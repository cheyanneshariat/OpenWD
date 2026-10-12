"""Reproduce the DAB refinement figures from the checksummed native archive.

No atmosphere calculation is launched. The archive also contains all prediction
arrays, predictor inputs, native requests, certificates and original attempts.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
PLAN = None
NATIVE = None
RESULTS = None
OUT = None
ARCHIVE_SHA256 = '029ab23abb72320696d974321747ebbeffe4e091af66eb47fb7d5e5da06451f4'
BLUE, ORANGE, GRAY = '#23659a', '#cf772f', '#777777'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / (name + '.' + ext), dpi=250)
    plt.close(fig)


def knot_map(plan, accepted, report):
    models = {r['model_id']: r for r in plan['models']}
    tests = {r['model_id']: r for r in report['records']}
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.2))
    rows = []
    for ax, cell in zip(axes.flat, plan['cells']):
        ts = np.asarray(cell['temperature_knots']) / 1000
        gs = cell['gravity_knots']
        # These lines show predictor cells, not a decorative plotting grid.
        ax.plot([ts[0], ts[2], ts[2], ts[0], ts[0]],
                [gs[0], gs[0], gs[2], gs[2], gs[0]],
                color='.7', lw=.9, zorder=0)
        ax.plot([ts[1], ts[1]], [gs[0], gs[2]], color=BLUE,
                lw=.7, alpha=.4, zorder=0)
        ax.plot([ts[0], ts[2]], [gs[1], gs[1]], color=BLUE,
                lw=.7, alpha=.4, zorder=0)
        for corner in cell['coarse_corners']:
            t, g, h = corner['parameters']
            ax.scatter(t / 1000, g, s=37, color=GRAY, marker='o', zorder=3)
            rows.append(dict(model_id=corner['model_id'], cell_id=cell['cell_id'],
                             teff_K=t, logg=g, log_H_He=h, role='original-corner',
                             numerical_status='PREVIOUSLY_VERIFIED'))
        ax.scatter(ts[1], gs[1], s=38, color='black', marker='D', zorder=3)
        rows.append(dict(model_id=cell['reused_center'], cell_id=cell['cell_id'],
                         teff_K=1000*ts[1], logg=gs[1],
                         log_H_He=cell['log_H_He'], role='reused-center',
                         numerical_status='PREVIOUSLY_VERIFIED'))
        for mid in cell['refinement_model_ids']:
            r = models[mid]
            assert mid in accepted and accepted[mid]['status'] == 'COMPLETE_NUMERICAL'
            t, g = r['config']['effective_temperature'], r['config']['logg']
            ax.scatter(t / 1000, g, s=47, color=BLUE, marker='s', zorder=4)
            rows.append(dict(model_id=mid, cell_id=cell['cell_id'], teff_K=t,
                             logg=g, log_H_He=cell['log_H_He'], role='new-grid-point',
                             numerical_status='COMPLETE_NUMERICAL'))
        for mid in cell['validation_model_ids']:
            r = tests[mid]
            t, g = r['parameters']['teff_K'], r['parameters']['logg']
            done = r['comparison_status'] == 'MEASURED'
            if done:
                assert mid in accepted and r['target_native_status'] == 'COMPLETE_NUMERICAL'
                ax.scatter(t / 1000, g, s=48, facecolors='none', edgecolors=ORANGE,
                           linewidths=1.3, marker='^', zorder=3)
            else:
                assert r['target_native_status'] == 'COMPLETE_NUMERICAL' and r['comparison_status']=='TARGET_PROFILE_NOT_RETRIEVED'
                ax.scatter(t / 1000, g, s=43, color=GRAY, linewidths=1.4,
                           marker='x', zorder=3)
            rows.append(dict(model_id=mid, cell_id=cell['cell_id'], teff_K=t,
                             logg=g, log_H_He=cell['log_H_He'], role='independent-test',
                             numerical_status=r['target_native_status']))
        h = cell['log_H_He']
        ax.set_title(r'$\log_{10}(n_{\rm H}/n_{\rm He}) = ' + f'{h:+d}' + '$', fontsize=13)
        ax.set_xlim(ts[0]-.45, ts[2]+.45)
        ax.set_ylim(gs[0]-.055, gs[2]+.055)
        ax.set_xticks(ts)
        ax.set_yticks(gs)
        ax.tick_params(labelsize=11)
    fig.supxlabel(r'$T_{\rm eff}$ (kK)', fontsize=15, y=.045)
    fig.supylabel(r'$\log g$', fontsize=15, x=.015)
    handles = [
        Line2D([], [], ls='none', marker='o', color=GRAY, label='Original grid'),
        Line2D([], [], ls='none', marker='s', color=BLUE, label='New grid point'),
        Line2D([], [], ls='none', marker='D', color='black', label='Reused center'),
        Line2D([], [], ls='none', marker='^', mfc='none', mec=ORANGE,
               color=ORANGE, label='Independent test: completed'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965),
               ncol=2, fontsize=11, columnspacing=1.8, handletextpad=.6)
    fig.text(.5, .995, '32 new DAB/DBA grid points in eight selected cells',
             ha='center', va='top', fontsize=15)
    fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.81,
                        wspace=.34, hspace=.38)
    save(fig, 'new_dab_points')
    fields = ['model_id', 'cell_id', 'teff_K', 'logg', 'log_H_He', 'role', 'numerical_status']
    with (OUT / 'plotted_points.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    assert Counter(r['role'] for r in rows) == {
        'original-corner': 32, 'reused-center': 8,
        'new-grid-point': 32, 'independent-test': 32}


def paired_errors(report):
    measured = [r for r in report['records'] if r['comparison_status'] == 'MEASURED']
    assert len(measured) == 32
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.5))
    for ax, metric, title, target in [
        (axes[0], 'integrated_absolute_error_fraction', 'Integrated absolute error', 1),
        (axes[1], 'maximum_relative_error_fraction', 'Maximum relative error', 3),
    ]:
        extrema = []
        for method, color, marker, label in [
            ('linear_F_Tg', BLUE, 'o', 'Linear flux'),
            ('pchip_logF_local_T_then_g', ORANGE, '^', 'Local log-flux PCHIP'),
        ]:
            x = np.array([100*r['metrics']['coarse_'+method][metric] for r in measured])
            y = np.array([100*r['metrics']['refined_'+method][metric] for r in measured])
            assert np.all(x > 0) and np.all(y > 0)
            extrema += [*x, *y, target]
            ax.scatter(x, y, s=32, color=color, marker=marker, alpha=.85, label=label)
        lo, hi = min(extrema)*.6, max(extrema)*1.7
        ax.plot([lo, hi], [lo, hi], color='.65', lw=.9, zorder=0)
        ax.axhline(target, color='.4', ls=':', lw=.9, zorder=0)
        ax.axvline(target, color='.4', ls=':', lw=.9, zorder=0)
        ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi))
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Before refinement (%)', fontsize=14)
        ax.set_ylabel('After refinement (%)', fontsize=14)
    axes[0].legend(loc='upper left', fontsize=11)
    fig.text(.5, .01, '32 independent spectra; 3000–10000 Å. Points below the diagonal have smaller errors.',
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.90, wspace=.31)
    save(fig, 'interpolation_gain')


def spectrum_examples(report):
    cases = ['dab-81faee5ab152418c8ef23fd7', 'dab-56c6ec857f07aaca72ac4421']
    records = {r['model_id']: r for r in report['records']}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4),
                             gridspec_kw={'height_ratios': [1.6, 1]}, sharex='col')
    examples = []
    for col, mid in enumerate(cases):
        r = records[mid]
        assert r['comparison_status'] == 'MEASURED'
        path = RESULTS / (mid + '.npz')
        leaf = NATIVE / mid / 'attempt-v1'
        meta = read(leaf / 'model/metadata.json')['spectrum_metadata']
        assert meta['flux_convention'] == 'surface F_lambda'
        assert meta['flux_unit'] == 'erg s^-1 cm^-2 Angstrom^-1'
        assert meta['wavelength_medium'] == 'vacuum'
        worker = read(leaf / 'worker_result.json')
        native_path = leaf / 'model' / worker['audit']['spectrum']['path']
        assert digest(native_path) == r['direct_provenance']['spectrum_sha256']
        native = np.loadtxt(native_path)
        with np.load(path, allow_pickle=False) as bank:
            wave, direct = bank['wavelength_angstrom'], bank['direct_F_lambda']
            assert np.array_equal(wave, native[:, 0]) and np.array_equal(direct, native[:, 1])
            mask = (wave >= 4250) & (wave <= 4950)
            ax, residual = axes[0, col], axes[1, col]
            for name, color, style, label in [
                ('coarse_linear_F_Tg', ORANGE, '--', 'Coarse interpolation'),
                ('refined_linear_F_Tg', BLUE, '-', 'Refined interpolation'),
            ]:
                flux = bank[name]
                ax.plot(wave[mask], flux[mask], color=color, ls=style, lw=1.5, label=label)
                residual.plot(wave[mask], 100*(flux[mask]-direct[mask])/direct[mask],
                              color=color, ls=style, lw=1.3)
                full = (wave >= 3000) & (wave <= 10000)
                peak = np.max(np.abs((flux[full]-direct[full])/direct[full]))
                assert np.isclose(peak, r['metrics'][name]['maximum_relative_error_fraction'],
                                  rtol=1e-12, atol=1e-14)
            ax.plot(wave[mask], direct[mask], color='black', ls=':', lw=1.7,
                    label='Direct OpenWD calculation', zorder=4)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            for bound in [-3, 3]:
                residual.axhline(bound, color='.55', ls=':', lw=.7)
            residual.axhline(0, color='.7', lw=.7, zorder=0)
            residual.set_xlabel('Vacuum wavelength (Å)', fontsize=14)
            residual.set_xlim(4250, 4950)
            params = r['parameters']
            ax.set_title(f"{params['teff_K']:,.0f} K; log g = {params['logg']:.3f}; "
                         f"log(H/He) = {params['log_H_He']:+d}", fontsize=12)
            examples.append(dict(model_id=mid, parameters=params, metrics={
                k: r['metrics'][k] for k in ['coarse_linear_F_Tg', 'refined_linear_F_Tg']},
                direct_spectrum_sha256=digest(native_path), plotted_band_angstrom=[4250, 4950],
                metric_band_angstrom=[3000, 10000], selection='Illustrative recovered pass and remaining large-error case'))
    axes[0, 0].set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=13)
    axes[1, 0].set_ylabel('Flux error (%)', fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=3, fontsize=11)
    fig.text(.5, .018, 'Linear interpolation; absolute surface flux. Dotted residual lines mark ±3%.',
             ha='center', fontsize=11)
    fig.subplots_adjust(left=.085, right=.985, bottom=.11, top=.865,
                        hspace=.08, wspace=.26)
    save(fig, 'spectrum_examples')
    (OUT / 'spectrum_examples.json').write_text(json.dumps(examples, indent=2)+'\n')


def check_predictions(report):
    """Recompute optical errors from all saved absolute-flux prediction arrays."""
    n_arrays = 0
    for row in report['records']:
        assert row['comparison_status'] == 'MEASURED'
        leaf = NATIVE / row['model_id'] / 'attempt-v1'
        worker = read(leaf / 'worker_result.json')
        native_path = leaf / 'model' / worker['audit']['spectrum']['path']
        assert (leaf/'model').resolve() in native_path.resolve().parents
        assert digest(native_path) == row['direct_provenance']['spectrum_sha256']
        native = np.loadtxt(native_path)
        with np.load(RESULTS/(row['model_id']+'.npz'), allow_pickle=False) as data:
            wave, direct = data['wavelength_angstrom'], data['direct_F_lambda']
            assert np.array_equal(wave,native[:,0]) and np.array_equal(direct,native[:,1])
            assert np.all(np.diff(wave)>0) and np.all(direct>0)
            x = np.r_[3000.,wave[(wave>3000)&(wave<10000)],10000.]
            a = np.interp(x,wave,direct)
            for key, recorded in row['metrics'].items():
                pred = data[key]
                assert np.isfinite(pred).all() and np.all(pred>0)
                b = np.interp(x,wave,pred)
                peak = float(np.max(np.abs((b-a)/a)))
                integrated = float(np.trapz(np.abs(b-a),x)/np.trapz(a,x))
                assert np.isclose(peak,recorded['maximum_relative_error_fraction'],rtol=1e-12)
                assert np.isclose(integrated,recorded['integrated_absolute_error_fraction'],rtol=1e-12)
                assert (peak<=.03 and integrated<=.01)==recorded['both_targets_pass']
            n_arrays += len(data.files)
    return n_arrays


def main():
    import tarfile
    import tempfile
    global OUT, PLAN, NATIVE, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--style',type=Path,default=ROOT/'grid-style.mplstyle')
    args=parser.parse_args()
    assert digest(args.archive)==ARCHIVE_SHA256,'Archive hash mismatch'
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    plt.style.use(args.style)
    with tempfile.TemporaryDirectory(prefix='openwd-refinement-') as temporary:
        root=Path(temporary)
        with tarfile.open(args.archive) as bank:
            for member in bank.getmembers():
                destination=(root/member.name).resolve()
                assert root.resolve() in destination.parents
                assert member.isfile() or member.isdir(),'Links are not supported'
            bank.extractall(root)  # Every member was restricted to regular files/directories above.
        for name,expected in read(root/'model-and-product-hashes.json').items():
            assert digest(root/name)==expected,name
        PLAN=root/'execution-v1/plan.json'
        NATIVE=root/'native-products'
        RESULTS=root/'new-validation-v1'
        plan,report=read(PLAN),read(RESULTS/'comparison.json')
        records=read(root/'native-models.json')
        accepted={r['model_id']:r for r in records}
        assert len(accepted)==104 and all(r['native_certificate_and_spectrum_checked'] for r in records)
        assert report['measured']==32 and report['requested_new_independent_targets']==32
        arrays=check_predictions(report)
        knot_map(plan,accepted,report)
        paired_errors(report)
        spectrum_examples(report)
    (OUT/'reproduction.json').write_text(json.dumps({
        'archive_sha256':ARCHIVE_SHA256,'independent_spectra_checked':32,
        'prediction_arrays_read':arrays,'methods_checked_per_target':8,
        'normalization':'none','extrapolation':False,
        'metric_interval_angstrom':[3000,10000],
        'independent_depth_convergence_verified':False,
        'physical_accuracy_verified':False,
    },indent=2)+'\n')
    print('Verified 32 native test spectra and 256 saved predictions; wrote three PNG/PDF figures.')


if __name__ == '__main__':
    main()
