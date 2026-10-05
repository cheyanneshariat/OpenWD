#!/usr/bin/env python3
"""Panelled G191-B2B UV overview with one shared, documented flux scale."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_hot_composition import line_catalogue
from compare_hot_daz_benchmark import (
    C_KMS, doppler_factor, instrument_sample, ism_lyalpha,
    verify_benchmark_parameters,
)
from hot_daz_benchmark import checked_files, read_observation
from verify_hot_composition_convergence import check_record


VELOCITY = 23.8
FEATURES = [
    ('Ly series', [937.8, 949.7]), ('Ly gamma / C III', [972.54, 977.02]),
    ('Ly beta / O VI', [1025.72, 1031.93, 1037.62]),
    ('S IV', [1062.662, 1072.974]), ('P V', [1117.977, 1128.008]),
    ('C III', [1175.7]), ('Si III', [1206.5]), ('Ly alpha', [1215.6701]),
    ('N V', [1238.821, 1242.804]), ('Ni V', [1306.624]),
    ('O IV', [1338.615, 1343.514]), ('Si IV', [1393.755, 1402.770]),
    ('Fe V', [1409.453]), ('C IV', [1548.195, 1550.772]),
    ('Al III', [1854.716, 1862.790]),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_segments(wavelength):
    """Never average an observed bin across a removed-data gap."""
    if len(wavelength) < 2:
        return []
    steps = np.diff(wavelength)
    boundaries = np.r_[0, np.flatnonzero(steps > 3 * np.median(steps)) + 1, len(wavelength)]
    return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b-a >= 2]


def sample_segments(model_wave, model_flux, wave, resolving_power):
    sampled = np.full(len(wave), np.nan)
    sigma = C_KMS / resolving_power / np.sqrt(8*np.log(2))
    shifted = model_wave * doppler_factor(VELOCITY)
    for segment in split_segments(wave):
        indices = np.arange(len(wave))[segment]
        local = wave[segment]
        half_bin = .5 * np.max(np.diff(local))
        keep = ((local-half_bin > shifted[0]*np.exp(6*sigma/C_KMS)) &
                (local+half_bin < shifted[-1]*np.exp(-6*sigma/C_KMS)))
        indices = indices[keep]
        if len(indices) >= 2:
            sampled[indices] = instrument_sample(model_wave, model_flux, wave[indices],
                velocity=VELOCITY, resolving_power=resolving_power, transmission=ism_lyalpha)
    return sampled


def shared_scale(record, catalogue):
    """One dilution/calibration parameter, estimated only from STIS continuum candidates."""
    wave, flux, error, model = (record[k] for k in ('wavelength', 'observed', 'error', 'unscaled_model'))
    anchors = (wave >= 1300.) & (wave <= 1600.) & np.isfinite(model) & (model > 0)
    for row in catalogue:
        anchors &= abs(wave-row['observed']) > row['observed'] * 20. / C_KMS
    ratio = np.divide(flux, model, out=np.full_like(flux, np.nan), where=np.isfinite(model) & (model > 0))
    uncertainty = np.divide(error, model, out=np.full_like(error, np.nan), where=np.isfinite(model) & (model > 0))
    used = anchors.copy()
    for _ in range(6):
        center = np.median(ratio[used])
        scatter = max(1.4826*np.median(abs(ratio[used]-center)), np.median(uncertainty[used]))
        updated = anchors & (abs(ratio-center) < 4*scatter)
        if np.array_equal(updated, used):
            break
        used = updated
    if used.sum() < 100:
        raise ValueError('too few continuum candidates for the shared scale')
    scale = float(np.average(ratio[used], weights=1/uncertainty[used]**2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid shared flux scale')
    return scale, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('results/hot-daz/composition-converged-full'))
    parser.add_argument('--data', type=Path, default=Path('results/hot-daz/multimetal-data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout', choices=('overview', 'fine-stis'), default='overview',
                        help='fine-stis uses narrower STIS panels and a separate broad FUSE page')
    parser.add_argument('--stis-panel-width', type=float, default=20., help='STIS detail panel width in Angstrom')
    args = parser.parse_args()
    if not np.isfinite(args.stis_panel_width) or not 0 < args.stis_panel_width <= 830.:
        raise ValueError('STIS panel width must be finite, positive and at most 830 A')
    metadata = json.loads((args.model/'metadata.json').read_text())
    check_record(metadata)
    verify_benchmark_parameters(metadata)
    with np.load(args.model/'spectrum.npz') as data:
        model_wave, model_flux = data['wavelength'].copy(), data['flux'].copy()
    if model_wave[0] > 910.001 or model_wave[-1] < 1989.99 or np.max(np.diff(model_wave)) > .01001:
        raise ValueError('continuous 910--1990 A synthesis is required')

    stis = checked_files('results/hot-daz/g191-b2b')
    manifest = json.loads((args.data/'manifest.json').read_text())
    specs = [
        ('FUSE', args.data/'hlsp_wd-linelist_fuse_spec_g191-b2b_fuv_v1_coadd-spec.fits', 910., 1160., 20000.),
        ('STIS E140H', stis['spectrum'], 1160., 1685., 144000.),
        ('STIS E230H', args.data/'hlsp_wd-linelist_hst_stis_g191-b2b_e230h_v1_coadd-spec.fits', 1685., 1990.001, 144000.),
    ]
    records = {}
    for label, path, lo, hi, resolution in specs:
        if path.parent == args.data and sha256(path) != manifest[path.name]['sha256']:
            raise ValueError(f'input checksum mismatch: {path}')
        wave, flux, error = read_observation(path)
        keep = (wave >= lo) & (wave < hi)
        wave, flux, error = wave[keep], flux[keep], error[keep]
        model = sample_segments(model_wave, model_flux, wave, resolution)
        records[label] = dict(wavelength=wave, observed=flux, error=error, unscaled_model=model,
                              resolving_power=resolution, input_path=str(path), input_sha256=sha256(path))
    scale, anchors = shared_scale(records['STIS E140H'], line_catalogue(stis['lines']))
    records['STIS E140H']['scale_anchor_mask'] = anchors
    for record in records.values():
        record['model'] = record['unscaled_model'] * scale

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {f'{label.replace(" ", "_")}_{key}': value for label, record in records.items()
              for key, value in record.items() if isinstance(value, np.ndarray)}
    np.savez(args.output/'full-uv-arrays.npz', **arrays)
    if args.layout == 'fine-stis':
        stis_edges = np.r_[np.arange(1160., 1990., args.stis_panel_width), 1990.]
        stis_panels = [list(x) for x in zip(stis_edges[:-1], stis_edges[1:])]
        page_panels = [[[910., 1000.], [1000., 1090.], [1090., 1160.]]]
        page_panels.extend(stis_panels[i:i+4] for i in range(0, len(stis_panels), 4))
    else:
        broad = [list(x) for x in zip(np.arange(910., 1990., 90.), np.arange(1000., 2080., 90.))]
        page_panels = [broad[i:i+4] for i in range(0, len(broad), 4)]
    report = dict(model_directory=str(args.model), model_metadata_sha256=sha256(args.model/'metadata.json'),
        model_spectrum_sha256=sha256(args.model/'spectrum.npz'), source_sha256=sha256(__file__),
        wavelength_range_angstrom=[910., 1990.], panels=[panel for page in page_panels for panel in page],
        layout=args.layout, stis_panel_width_angstrom=args.stis_panel_width if args.layout == 'fine-stis' else 90.,
        stis_vertical_limits='local full data/model range with padding' if args.layout == 'fine-stis' else 'zero to padded maximum',
        flux_convention='observed F_lambda in erg s^-1 cm^-2 Angstrom^-1; model multiplied by one shared scale',
        shared_surface_to_observed_flux_scale=scale, scale_fit=dict(instrument='STIS E140H', window=[1300., 1600.],
            line_mask_half_width_kms=20., retained_pixels=int(anchors.sum()), robust_ratio_clipping_sigma=4.),
        velocity_kms=VELOCITY, ism_lyalpha=dict(log_column=18.18, b_kms=10., velocity_kms=19.4),
        inputs={label:{k:v for k,v in record.items() if not isinstance(v, np.ndarray)} for label,record in records.items()},
        model_sample_counts={label:int(np.isfinite(r['model']).sum()) for label,r in records.items()},
        qualifications=[
            'Only the available model interval 910--1990 A is plotted; the E230H observations extend beyond it.',
            'Higher-resolution E140H is preferred in instrument overlaps; E230H starts at 1685 A.',
            'One scale is used throughout, with no per-panel continuum normalization or abundance adjustment.',
            'Native observed bins; Gaussian instrumental response approximates the actual line-spread functions.',
            'The previously adopted H I Ly alpha absorber is included; other ISM absorption and airglow are not modeled.',
            'The same 23.8 km/s photospheric shift is used throughout. FUSE segment offsets are not recalibrated.',
            'The earlier local P V 1128 catalogue-to-coadd correction is not a global FUSE wavelength solution and is not extrapolated here.',
            'Only trace populations are converged. Metal feedback on the H/He temperature structure remains unsolved.',
            'A few blue-edge samples have no model because convolution would require extrapolation beyond the synthesis.',
        ])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'path.simplify':False, 'pdf.fonttype':42})
    def draw(ax, lo, hi):
        selected = []
        peaks = []; floors = []
        for label, record in records.items():
            w = record['wavelength']; mask = (w >= lo) & (w <= hi)
            if not mask.any():
                continue
            selected.append(label)
            x, obs, model = w[mask], record['observed'][mask]/1e-11, record['model'][mask]/1e-11
            for segment in split_segments(x):
                ax.plot(x[segment], obs[segment], c='#3f464d', lw=.6, alpha=.9)
                ax.plot(x[segment], model[segment], c='#cb4335', lw=.8, alpha=.95)
            peaks.extend([float(np.max(obs)), float(np.nanmax(model))])
            floors.extend([float(np.min(obs)), float(np.nanmin(model))])
        fine = args.layout == 'fine-stis' and lo >= 1160.
        if fine:
            lower, upper = min(floors), max(peaks)
            span = max(upper-lower, .01*upper)
            limits = (max(0., lower-.06*span), upper+.20*span)
        else:
            limits = (0., 1.20*max(peaks))
        ax.set(xlim=(lo, hi), ylim=limits)
        ax.set_title(f'{lo:.0f}–{hi:.0f} Å  |  '+ ' / '.join(selected), loc='left', fontsize=10.5)
        ax.xaxis.set_major_locator(MultipleLocator((2 if hi-lo <= 12 else 5) if fine else 20))
        ax.xaxis.set_minor_locator(MultipleLocator((.5 if hi-lo <= 12 else 1) if fine else 5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='y', color='.92', lw=.5)
        for boundary in (1160., 1685.):
            if lo < boundary < hi:
                ax.axvline(boundary, c='.55', ls=':', lw=.8)
        placed = []
        for label, centers in FEATURES:
            positions = np.array(centers)*doppler_factor(VELOCITY)
            margin = .015*(hi-lo) if fine else .5
            positions = positions[(positions > lo+margin) & (positions < hi-margin)]
            if not len(positions):
                continue
            center = float(np.mean(positions))
            height = .94 if not placed or center-placed[-1] > .14*(hi-lo) else .83
            placed.append(center)
            ax.plot(positions, np.full(len(positions), height-.035), '|', color='#666666',
                    ms=5, transform=ax.get_xaxis_transform())
            text_center = np.clip(center, lo+.035*(hi-lo), hi-.035*(hi-lo))
            ax.text(text_center, height, label, ha='center', va='bottom', fontsize=8,
                    color='#555555', transform=ax.get_xaxis_transform())

    def decorate(fig, start, stop):
        height = fig.get_figheight()
        fig.text(.015,.5,r'$F_\lambda$ ($10^{-11}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)',
                 rotation=90,va='center',ha='center',fontsize=11)
        kind = (f'STIS detail: {args.stis_panel_width:g} Å panels' if args.layout == 'fine-stis' and start >= 1160.
                else 'UV overview')
        fig.suptitle(f'G191-B2B | {kind} | {start:.0f}–{stop:.0f} Å', y=1-.20/height, fontsize=15)
        fig.text(.5, 1-.52/height, 'Converged trace populations on the fixed 52,500 K / log g 7.53 atmosphere; Fe and Ni in LTE',
                 ha='center', fontsize=10)
        handles = [Line2D([],[],c='#3f464d',lw=1,label='Observed'),
                   Line2D([],[],c='#cb4335',lw=1,label='Model + adopted ISM Ly alpha; one shared flux scale')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,1-.645/height), ncol=2, frameon=False, fontsize=9)
        fig.text(.5,.32/height,'Native observed bins; Gaussian instrumental response. Other ISM/airglow and FUSE segment shifts remain unmodeled.',
                 ha='center',fontsize=8)
        footer = ('Local y-axis zoom; one shared flux scale, no panel normalization. Metal thermal feedback is not solved.'
                  if args.layout == 'fine-stis' and start >= 1160. else
                  'One STIS continuum scale is applied throughout; no panel is independently normalized. Metal thermal feedback is not solved.')
        fig.text(.5,.14/height,footer,
                 ha='center',fontsize=8)

    pages = []; stis_pages = []
    with ExitStack() as stack:
        pdf = stack.enter_context(PdfPages(args.output/'full-uv-comparison.pdf'))
        stis_pdf = (stack.enter_context(PdfPages(args.output/'stis-detail.pdf'))
                    if args.layout == 'fine-stis' else None)
        for subset in page_panels:
            fig, axes = plt.subplots(len(subset),1,figsize=(14,2.5+2.25*len(subset)),squeeze=False)
            axes = axes.ravel()
            # Keep header/footer space fixed in inches on shorter final pages.
            height = fig.get_figheight()
            fig.subplots_adjust(left=.075,right=.985,bottom=1.035/height,top=1-1.3225/height,hspace=.48)
            for ax,(lo,hi) in zip(axes,subset):
                draw(ax,lo,hi)
            axes[-1].set_xlabel('Observed vacuum wavelength (Å)')
            decorate(fig,subset[0][0],subset[-1][1])
            name=f'full-uv-{int(subset[0][0])}-{int(subset[-1][1])}'
            fig.savefig(args.output/(name+'.png'),dpi=200)
            fig.savefig(args.output/(name+'.pdf'))
            pdf.savefig(fig)
            if stis_pdf is not None and subset[0][0] >= 1160.:
                stis_pdf.savefig(fig)
                stis_pages.append(name)
            pages.append(name)
            plt.close(fig)
    report['pages'] = pages
    report['stis_pages'] = stis_pages
    (args.output/'full-uv-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(shared_flux_scale=scale,continuum_anchor_pixels=int(anchors.sum()),pages=pages),indent=2))


if __name__ == '__main__':
    main()
