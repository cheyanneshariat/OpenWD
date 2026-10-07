#!/usr/bin/env python3
"""sdB paper figure: observed optical spectra vs OpenWD hybrid-NLTE predictions.

    python research/plot_sdb_paper_comparison.py --models results/sdb/paper --output results/sdb/paper/figure

Same layout and normalization as the DO/DAO paper figure
(research/plot_hot_paper_comparison.py on main): stacked full-width panels,
observation gray, OpenWD orange, each spectrum independently normalized with
the paper's broad upper-envelope helper, observed vacuum wavelengths, gaps
left blank.  Stellar parameters and abundances are fixed at published values
(see research/sdb_paper_models.sh); nothing is fitted here.  Models are
convolved with a Gaussian at each instrument's resolving power and shifted by
fixed radial velocities: X-shooter co-adds are in the observed (air) frame,
the Dorsch et al. (2020) UVES co-adds in the stellar rest (air) frame.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIGHT_SPEED_KMS = 299792.458
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
OBS = ROOT / 'results/sdb/obs'
# key: (label, observation, instrument, resolving power below/above the split, split (A, air), RV km/s)
# The paper's X-shooter display mask (validate_metal_suite._broad_observation_mask)
# within this range: the UVB/VIS dichroic join of the public Phase-3 products.
XSHOOTER_MASKS = ((5330.0, 5590.0),)
# X-shooter covers the whole range: three equal bands.
XSHOOTER_BANDS = ((3800.0, 4800.0), (4800.0, 5800.0), (5800.0, 6800.0))
TARGETS = {
    'hd4539': dict(name='HD 4539', source='Schneider et al. 2018; Geier 2013', instrument='X-shooter',
                   observation=OBS / 'hd4539', power=(9861., 18340.), split=5500., velocity=-3.0,
                   masks=XSHOOTER_MASKS, bands=XSHOOTER_BANDS,
                   # Interstellar Ca II K, H and Na D2, D1 (observed frame, air A).
                   interstellar=(3933.66, 3968.47, 5889.95, 5895.92)),
    'feige38': dict(name='Feige 38', source='Schneider et al. 2018; Geier 2013', instrument='X-shooter',
                    observation=OBS / 'feige38', power=(9861., 18340.), split=5500., velocity=8.5,
                    masks=XSHOOTER_MASKS, bands=XSHOOTER_BANDS),
    'lsiv14116': dict(name='LS IV−14°116', source='Dorsch et al. 2020', instrument='UVES',
                      observation=OBS / 'dorsch2020/lsiv_uves_srnt.fits', power=(40970., 42310.), split=4650.,
                      velocity=0.0, vsini=9.0, frame_velocity=-154.0,
                      # Interstellar Ca II K, H and Na D2, D1 in the stellar rest frame
                      # (v_rad = -154 km/s, Dorsch et al. 2020), observed air A.
                      interstellar=(3935.71, 3970.54, 5892.73, 5898.70)),
    'feige46': dict(name='Feige 46', source='Latour et al. 2019; Dorsch et al. 2020', instrument='UVES',
                    observation=OBS / 'dorsch2020/f46_uves_sr.fits', power=(40970., 42310.), split=4570.,
                    velocity=0.0, edge_trim=40.0, vsini=9.0, frame_velocity=89.0),
}
ORDER = ('hd4539', 'feige38', 'lsiv14116', 'feige46')
# Telluric bands (observed air A): H2O 5880-5990 and O2 gamma 6270-6330.  The
# UVES co-adds were shifted to the stellar rest frame, which moves these
# bands by -frame_velocity (the stellar radial velocity removed, km/s).
TELLURIC_AIR = ((5880.0, 5990.0), (6270.0, 6330.0))
ELEMENT_ORDER = ('C', 'N', 'O', 'Ne', 'Mg', 'Al', 'Si', 'P', 'S', 'Ar', 'Ca', 'Ti', 'Cr', 'Fe', 'Co', 'Ni', 'Zn',
                 'Ge', 'Sr', 'Y', 'Zr', 'Sn')
# Feige 46's UVES co-add is order-merged but not continuum-corrected: its flux
# falls by 30-50% within 20-40 A of each chip edge, which no broad continuum
# follows; 'edge_trim' (A) removes those edges inside the range.
LIMITS = (3800.0, 6800.0)
# Three stacked panels per star, one per UVES chip range of the two He-rich
# stars (vacuum A), each with its own wavelength axis.
BANDS = ((3800.0, 4535.0), (4650.0, 5765.0), (5830.0, 6800.0))


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def paper_normalizer():
    """The DZ/DQ/DO paper figures' broad upper-envelope helper, loaded alone."""
    path = WORKSPACE / 'scripts/validate_metal_suite.py'
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_upper_envelope_normalize')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[node.name], identity(path)


def air_to_vacuum(air):
    """Inverse of the IAU/Morton vacuum-to-air relation used elsewhere (two fixed-point passes)."""
    vacuum = np.asarray(air, dtype=float).copy()
    for _ in range(3):
        s2 = (1e4 / vacuum) ** 2
        vacuum = air * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
    return vacuum


def observation(target):
    """Observed wavelengths (air, as tabulated) and flux, joined across arms."""
    path = target['observation']
    if path.suffix == '.fits':
        data = fits.getdata(path, 1)
        wave, flux = np.asarray(data['WAVE'], float), np.asarray(data['FLUX'], float)
    else:
        parts = []
        for arm, keep in (('uvb', lambda w: w < target['split']), ('vis', lambda w: w >= target['split'])):
            data = fits.getdata(path / f'coadd-{arm}.fits', 1)
            w, f = np.asarray(data['WAVE'], float), np.asarray(data['FLUX'], float)
            parts.append((w[keep(w)], f[keep(w)]))
        wave, flux = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    good = np.isfinite(wave) & np.isfinite(flux)
    order = np.argsort(wave[good])
    return wave[good][order], flux[good][order]


def gaussian_convolve(wavelength, flux, resolving_power):
    """Constant-R Gaussian (FWHM = lambda/R) on a log-lambda grid, 10 samples per FWHM."""
    log_grid = np.arange(np.log(wavelength[0]), np.log(wavelength[-1]), 1.0 / (resolving_power * 10))
    sampled = np.interp(log_grid, np.log(wavelength), flux)
    sigma = 10 / (2 * np.sqrt(2 * np.log(2)))
    half = int(np.ceil(5 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return np.exp(log_grid), np.convolve(sampled, kernel / kernel.sum(), mode='same')


def rotation_broadened(wavelength, flux, vsini, limb_darkening=0.6):
    """Gray (2005) rotation profile on a uniform log-lambda grid (0.1 km/s per sample)."""
    step = 0.1 / LIGHT_SPEED_KMS
    log_grid = np.arange(np.log(wavelength[0]), np.log(wavelength[-1]), step)
    sampled = np.interp(log_grid, np.log(wavelength), flux)
    x = np.arange(-int(vsini / 0.1), int(vsini / 0.1) + 1) * 0.1 / vsini
    x = np.clip(x, -1.0, 1.0)
    kernel = (2 * (1 - limb_darkening) * np.sqrt(1 - x ** 2) + 0.5 * np.pi * limb_darkening * (1 - x ** 2))
    return np.exp(log_grid), np.convolve(sampled, kernel / kernel.sum(), mode='same')


def predicted(model_spectrum, target, observed_air):
    """Model on the observed (air) grid: per-arm convolution, fixed velocity shift."""
    saved = np.load(model_spectrum)
    vacuum, flux = saved['wavelength_vacuum'], saved['flux']
    # vacuum -> air with the same relation as sdb_compare_hd4539.py
    s2 = (1e4 / vacuum) ** 2
    air = vacuum / (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
    if target.get('vsini'):
        air, flux = rotation_broadened(air, flux, target['vsini'])
    shift = 1 + target['velocity'] / LIGHT_SPEED_KMS
    out = np.empty_like(observed_air)
    for power, keep in ((target['power'][0], observed_air < target['split']),
                        (target['power'][1], observed_air >= target['split'])):
        if keep.any():
            w, f = gaussian_convolve(air, flux, power)
            out[keep] = np.interp(observed_air[keep], w * shift, f)
    return out


def prepare(key, models, normalize, maximum_defect=None):
    """Observed and predicted spectra of one target on the observed grid (vacuum A), normalized
    per contiguous segment; NaN breaks at gaps; 'clean'/'telluric' split the observation."""
    target = TARGETS[key]
    model_dir = models / f'{key}-metals'
    summary = json.loads((model_dir / 'run-summary.json').read_text())
    if not summary['converged'] and not (maximum_defect and summary['population_defect'] <= maximum_defect):
        raise SystemExit(f'{key}: metal populations not converged (defect {summary["population_defect"]:.3g})')
    host = summary['host']
    air, observed = observation(target)
    model = predicted(model_dir / 'spectrum.npz', target, air)
    wave = air_to_vacuum(air)
    use = (wave >= LIMITS[0]) & (wave <= LIMITS[1])
    wave, observed, model = wave[use], observed[use], model[use]
    keep = np.ones(wave.size, dtype=bool)
    for lower, upper in target.get('masks', ()):
        keep &= ~((air[use] >= lower) & (air[use] <= upper))
    wave, observed, model = wave[keep], observed[keep], model[keep]
    # Chip/order gaps (> 20 A) bound the normalization segments; smaller
    # gaps (masked pixels) only break the plotted line.
    segments = np.split(np.arange(wave.size), np.flatnonzero(np.diff(wave) > 20.0) + 1)
    trim = target.get('edge_trim', 0.0)
    if trim:
        # Every chip edge inside the display range; not the range limits.
        keep = np.ones(wave.size, dtype=bool)
        for segment in segments:
            w = wave[segment]
            if w[0] > LIMITS[0] + 1.0:
                keep[segment[w < w[0] + trim]] = False
            if w[-1] < LIMITS[1] - 1.0:
                keep[segment[w > w[-1] - trim]] = False
        wave, observed, model = wave[keep], observed[keep], model[keep]
        segments = np.split(np.arange(wave.size), np.flatnonzero(np.diff(wave) > 20.0) + 1)
    observed = np.concatenate([normalize(wave[s], observed[s]) for s in segments])
    model = np.concatenate([normalize(wave[s], model[s]) for s in segments])
    steps = np.diff(wave)
    gaps = np.flatnonzero(steps > 5 * np.median(steps[steps > 0])) + 1
    wave, observed, model = (np.insert(a, gaps, np.nan) for a in (wave, observed, model))
    bands = target.get('bands', BANDS)
    # Observed flux inside telluric bands in light gray (frame of the plotted spectrum).
    shift = 1 - target.get('frame_velocity', 0.0) / LIGHT_SPEED_KMS
    telluric = np.zeros(wave.size, dtype=bool)
    for lower, upper in TELLURIC_AIR:
        telluric |= (wave >= air_to_vacuum(lower) * shift) & (wave <= air_to_vacuum(upper) * shift)
    for line in target.get('interstellar', ()):  # interstellar, not telluric: keep dark
        telluric &= np.abs(wave - air_to_vacuum(line)) > 1.0
    clean = np.where(telluric, np.nan, observed)
    # Overlap one sample at each band edge so the two traces join.
    edges = np.flatnonzero(np.diff(telluric.astype(int)) != 0)
    affected = telluric.copy(); affected[edges] = True; affected[np.minimum(edges + 1, wave.size - 1)] = True
    tell = np.where(affected, observed, np.nan)
    return dict(target=target, summary=summary, host=host, model_dir=model_dir, wave=wave,
                observed=observed, model=model, clean=clean, telluric=tell, gaps=gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', type=Path, default=ROOT / 'results/sdb/paper')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/sdb/paper/figure')
    # The paper figure: one He-poor and one He-rich star (Feige 38 is a 3He
    # star, whose He I isotope shifts the 4He models do not include).
    parser.add_argument('--targets', nargs='+', choices=ORDER, default=['hd4539', 'lsiv14116'])
    parser.add_argument('--name', default='sdb_two_star_optical')
    parser.add_argument('--maximum-defect', type=float, default=None,
                        help='also accept a metal solve that stopped with this final population defect '
                             '(solver tolerance 1e-3); recorded in the provenance')
    args = parser.parse_args()
    order = [key for key in ORDER if key in args.targets]
    args.output.mkdir(parents=True, exist_ok=True)
    normalize, normalizer_source = paper_normalizer()
    plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'font.size': 10,
                         'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    report = dict(scope=__doc__.strip(), normalizer=normalizer_source, plotter=identity(Path(__file__)), cases={})
    stored = {}
    # Rows: for each star its three band panels, then a spacer (not after the last).
    panel, spacer = 1.15, .30
    ratios = []
    for index in range(len(order)):
        ratios += [panel] * len(BANDS) + ([spacer] if index < len(order) - 1 else [])
    height = sum(ratios) + 1.0
    fig = plt.figure(figsize=(8., height))
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=.36,
                            left=.105, right=.965, bottom=.62 / height, top=1 - .32 / height)
    axes, row_index = [], 0
    for index in range(len(order)):
        axes.append([fig.add_subplot(grid[row_index + band, 0]) for band in range(len(BANDS))])
        row_index += len(BANDS) + 1
    for row, key in zip(axes, order):
        prepared = prepare(key, args.models, normalize, args.maximum_defect)
        target, summary, host, model_dir = (prepared[k] for k in ('target', 'summary', 'host', 'model_dir'))
        wave, observed, model, gaps = (prepared[k] for k in ('wave', 'observed', 'model', 'gaps'))
        clean, tell = prepared['clean'], prepared['telluric']
        bands = target.get('bands', BANDS)
        for ax, (lower, upper) in zip(row, bands):
            ax.plot(wave, clean, color='.30', lw=.50, rasterized=True)
            ax.plot(wave, tell, color='.75', lw=.50, rasterized=True)
            ax.plot(wave, model, color='#d95f02', lw=.60)
            # Crop each panel to this star's data inside the band.
            inside = wave[(wave >= lower) & (wave <= upper) & np.isfinite(observed)]
            lower, upper = max(lower, inside.min() - 5.0), min(upper, inside.max() + 5.0)
            ax.set_xlim(lower, upper); ax.set_ylim(0, 1.25); ax.set_yticks((0, .5, 1))
            ax.set_xticks(np.arange(np.ceil(lower / 100 + 0.25) * 100, upper - 20, 100))
            ax.tick_params(top=True, right=True, length=4, direction='in', labelsize=9)
        # One label per group of interstellar lines closer than 10 A (Na D1/D2).
        groups = []
        for line in sorted(float(air_to_vacuum(x)) for x in target.get('interstellar', ())):
            if groups and line - groups[-1][-1] < 10.0:
                groups[-1].append(line)
            else:
                groups.append([line])
        for vacuum in (float(np.mean(g)) for g in groups):
            for ax, (lower, upper) in zip(row, bands):
                if lower <= vacuum <= upper:
                    ax.annotate('IS', (vacuum, 1.05), ha='center', va='bottom', fontsize=7, color='.35')
        label = (f'{target["name"]} ({target["instrument"]})   '
                 + rf'$T_{{\rm eff}}={host["teff"]:.0f}\,$K, $\log g={host["logg"]:.2f}$, '
                 + rf'$\log(N_{{\rm He}}/N_{{\rm H}})={host["log_he_h"]:.2f}$')
        row[0].set_title(label + '   [' + target['source'] + ']', fontsize=10, loc='left', pad=3)
        # Adopted metal abundances, log N(X)/N(H): NLTE elements, then LTE line opacity.
        nlte = summary['abundances']
        lte = {**summary.get('lte_abundances', {}),
               **{e: v for e, v in summary.get('heavy_abundances', {}).items() if e not in nlte}}
        fmt = lambda values: ', '.join(f'{e} {values[e]:.2f}'.replace('-', '\u2212')
                                       for e in ELEMENT_ORDER if e in values)
        row[1].text(.008, .05, r'$\log N_{\rm X}/N_{\rm H}$ (NLTE): ' + fmt(nlte) + '\n'
                    + r'$\log N_{\rm X}/N_{\rm H}$ (LTE): ' + fmt(lte), transform=row[1].transAxes,
                    ha='left', va='bottom', fontsize=6.3, color='.2', linespacing=1.3)
        stored[key + '__wavelength'], stored[key + '__observed'], stored[key + '__model'] = wave, observed, model
        report['cases'][key] = dict(name=target['name'], instrument=target['instrument'],
                                    resolving_power=target['power'], arm_split_air=target['split'],
                                    velocity_km_s=target['velocity'], masks_air=target.get('masks', ()),
                                    edge_trim_angstrom=target.get('edge_trim', 0.0),
                                    vsini_km_s=target.get('vsini', 0.0),
                                    telluric_bands_air=TELLURIC_AIR,
                                    telluric_frame_velocity_km_s=target.get('frame_velocity', 0.0),
                                    heavy_abundances=summary.get('heavy_abundances', {}),
                                    normalization='paper upper-envelope helper per contiguous segment',
                                    display_bands_vacuum=target.get('bands', BANDS),
                                    parameters=host, abundances=summary['abundances'],
                                    lte_abundances=summary.get('lte_abundances', {}),
                                    metal_iterations=summary['iterations'], gap_breaks=int(len(gaps)),
                                    metal_converged=summary['converged'],
                                    metal_population_defect=summary['population_defect'],
                                    accepted_maximum_defect=args.maximum_defect,
                                    model=identity(model_dir / 'spectrum.npz'),
                                    observation=[identity(p) for p in (
                                        [target['observation']] if target['observation'].suffix == '.fits' else
                                        [target['observation'] / f'coadd-{a}.fits' for a in ('uvb', 'vis')])])
    axes[0][0].legend(handles=[Line2D([], [], color='.30', lw=.8, label='Observed'),
                               Line2D([], [], color='#d95f02', lw=.8, label='OpenWD')],
                      loc='lower right', ncol=2, frameon=False, fontsize=9, handlelength=2.3, columnspacing=1.4)
    fig.supxlabel(r'vacuum wavelength [$\mathrm{\AA}$]', fontsize=11, y=.06 / height)
    fig.supylabel('normalized flux', fontsize=11, x=.015)
    fig.savefig(args.output / f'{args.name}.pdf', dpi=400)  # rasterized observed spectra
    fig.savefig(args.output / f'{args.name}.png', dpi=220)
    np.savez_compressed(args.output / f'{args.name}-arrays.npz', **stored)
    report['figure'] = identity(args.output / f'{args.name}.pdf')
    (args.output / f'{args.name}-provenance.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(args.output / f'{args.name}.pdf')


if __name__ == '__main__':
    main()
