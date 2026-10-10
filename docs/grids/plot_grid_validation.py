"""Reproduce DAB interpolation diagnostics from checksummed saved arrays.

This script calculates errors and plots saved direct/predicted surface fluxes.
It performs no atmosphere solve, normalization or wavelength resampling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def metrics(wave, direct, prediction):
    """Integrate over 3000--10000 Angstrom, with exact interval endpoints."""
    low, high = 3000., 10000.
    assert wave[0] <= low < high <= wave[-1]
    inside = (wave > low) & (wave < high)
    x = np.r_[low, wave[inside], high]
    a = np.r_[np.interp(low, wave, direct), direct[inside], np.interp(high, wave, direct)]
    b = np.r_[np.interp(low, wave, prediction), prediction[inside], np.interp(high, wave, prediction)]
    relative = (b - a) / a
    # NumPy 1.x and 2.x have different preferred names for trapezoidal integration.
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return dict(
        integrated_absolute_error_fraction=float(integrate(np.abs(b - a), x) / integrate(a, x)),
        maximum_relative_error_fraction=float(np.max(np.abs(relative))))


def check_arrays(wave, direct, prediction):
    assert wave.ndim == direct.ndim == prediction.ndim == 1
    assert wave.shape == direct.shape == prediction.shape
    assert all(np.isfinite(a).all() for a in (wave, direct, prediction))
    assert (np.diff(wave) > 0).all() and (direct > 0).all() and (prediction > 0).all()


def compare_metrics(actual, recorded):
    for key, value in actual.items():
        np.testing.assert_allclose(value, recorded[key], rtol=1e-12, atol=1e-14)


def save(fig, output, name):
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{ext}', dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('validation-2026-10-09'))
    parser.add_argument('--style', type=Path, default=Path(__file__).with_name('grid-style.mplstyle'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for line in (args.inputs / 'INPUT_SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((args.inputs / name).read_bytes()).hexdigest() == expected, name
    report = json.loads((args.inputs / 'interpolation/comparison.json').read_text())
    records = report['records']
    assert len(records) == report['measured'] == 8
    arrays = {}
    computed = {}
    for row in records:
        assert row['comparison_status'] == 'MEASURED' and row['status'] == 'COMPLETE_NUMERICAL'
        with np.load(args.inputs / 'interpolation' / f'{row["model_id"]}.npz', allow_pickle=False) as source:
            wave, direct, prediction = [source[k] for k in
                ('wavelength_angstrom', 'independent_F_lambda', 'interpolated_F_lambda')]
        check_arrays(wave, direct, prediction)
        measured = metrics(wave, direct, prediction)
        compare_metrics(measured, row['optical'])
        computed[row['model_id']] = measured
        arrays[row['model_id']] = wave, direct, prediction
    args.output.mkdir(parents=True, exist_ok=False)
    plt.style.use(args.style)
    for kind in ('spectra', 'residuals'):
        fig, axes = plt.subplots(4, 2, figsize=(11, 11.7), sharex=True)
        for ax, row in zip(axes.flat, records):
            p = row['parameters']
            annotation = (rf'$T_{{\rm eff}}={p["teff_K"]/1000:g}\,{{\rm kK}},\ \log g={p["logg"]:g}$'
                          + '\n' + rf'$\log_{{10}}[N({{\rm H}})/N({{\rm He}})]={p["log_H_He"]:g}$')
            ax.text(.97, .94, annotation, ha='right', va='top', transform=ax.transAxes, fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            wave, direct, prediction = arrays[row['model_id']]
            mask = (wave >= 3000) & (wave <= 10000)
            if kind == 'spectra':
                ax.plot(wave[mask], direct[mask] / 1e8, color='#23659a', lw=1.1, label='Independent OpenWD model')
                ax.plot(wave[mask], prediction[mask] / 1e8, color='#cf772f', ls='--', lw=1.1, label='Grid interpolation')
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0, color='.65', lw=.7)
                ax.plot(wave[mask], 100 * (prediction[mask] - direct[mask]) / direct[mask], color='#23659a', lw=.8)
                for target in (3, -3):
                    ax.axhline(target, color='#cf772f', ls='--', lw=.6)
            ax.set_xlim(3000, 10000)
        for ax in axes[-1]:
            ax.set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)', fontsize=15)
        fig.supylabel(r'$F_\lambda$ ($10^8$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)' if kind == 'spectra'
                      else r'$(F_{\lambda,\mathrm{interp}}/F_{\lambda,\mathrm{direct}}-1)$ (%)', fontsize=15, x=.02)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.10, right=.985, bottom=.085, top=.99, wspace=.19, hspace=.18)
        save(fig, args.output, f'dab_interpolation_{kind}')

    abundance = json.loads((args.inputs / 'abundance/comparison.json').read_text())
    with np.load(args.inputs / 'abundance/inputs.npz', allow_pickle=False) as source:
        wave = source['wavelength_angstrom']
        cases = [(source['direct_half'], source['predicted_half'], .5, '1 dex spacing', 'one_dex'),
                 (source['direct_quarter'], source['predicted_quarter'], .25, '0.5 dex spacing', 'half_dex')]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    mask = (wave >= 3000) & (wave <= 10000)
    for ax, (direct, prediction, h, label, key) in zip(axes, cases):
        check_arrays(wave, direct, prediction)
        measured = metrics(wave, direct, prediction)
        compare_metrics(measured, abundance[key]['metrics'])
        computed[key] = measured
        ax.axhline(0, color='.65', lw=.7)
        for target in (3, -3):
            ax.axhline(target, color='.65', ls='--', lw=.5)
        ax.plot(wave[mask], 100 * (prediction[mask] / direct[mask] - 1), color='#23659a', lw=.9)
        ax.text(.97, .94, label + '\n' + rf'$\log_{{10}}[N({{\rm H}})/N({{\rm He}})]={h:g}$',
                ha='right', va='top', transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
        ax.set_xlim(3000, 10000)
        ax.set_ylabel('Interpolation error (%)', fontsize=15)
    axes[1].set_xlabel(r'Vacuum wavelength ($\mathrm{\AA}$)', fontsize=16)
    fig.text(.5, .975, r'$T_{\rm eff}=22.5\,\mathrm{kK},\quad \log g=6.25$', ha='center', fontsize=15)
    fig.subplots_adjust(left=.11, right=.985, bottom=.11, top=.935, hspace=.15)
    save(fig, args.output, 'pure_abundance_error')
    (args.output / 'recomputed-metrics.json').write_text(json.dumps(computed, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(comparisons=10, metrics_match=True, figures=3)))


if __name__ == '__main__':
    main()
