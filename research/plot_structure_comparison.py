#!/usr/bin/env python3
"""Observed G191-B2B UV vs trace-metal models with and without metals in the structure.

20-A panels, 910-1990 A. Each spectrum is normalized within each panel by a
linear fit to its own upper envelope, so line depths compare directly. Models
are shifted, convolved and bin-averaged exactly as in plot_hot_daz_uv_atlas.py
(including the adopted ISM Ly-alpha absorber).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_hot_daz_uv_atlas import sample_segments
from hot_daz_benchmark import read_observation

DATA = [('FUSE', 'results/hot-daz/multimetal-data/hlsp_wd-linelist_fuse_spec_g191-b2b_fuv_v1_coadd-spec.fits', 910., 1160., 20000.),
        ('STIS E140H', 'results/hot-daz/g191-b2b/hlsp_wd-linelist_hst_stis_g191-b2b_e140h_v1_coadd-spec.fits', 1160., 1685., 144000.),
        ('STIS E230H', 'results/hot-daz/multimetal-data/hlsp_wd-linelist_hst_stis_g191-b2b_e230h_v1_coadd-spec.fits', 1685., 1990.001, 144000.)]


def envelope_normalize(w, f, bins=5, level=85.):
    """Divide by a line through the upper envelope (a high percentile in sub-bins)."""
    good = np.isfinite(f)
    if good.sum() < 10:
        return f * np.nan
    x, y = w[good], f[good]
    chunks = np.array_split(np.arange(x.size), bins)
    cx = np.array([np.median(x[c]) for c in chunks])
    cy = np.array([np.percentile(y[c], level) for c in chunks])
    fit = np.polyfit(cx - x.mean(), cy, 1)
    return f / np.polyval(fit, w - x.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fixed-host', type=Path, required=True)
    p.add_argument('--blanketed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel-width', type=float, default=20.)
    p.add_argument('--panels-per-page', type=int, default=6)
    args = p.parse_args()
    models = {}
    for label, path in (('without metals in structure', args.fixed_host), ('with metals in structure', args.blanketed)):
        with np.load(path) as z:
            models[label] = (z['wavelength'].copy(), z['flux'].copy())
    observed = []
    for name, path, lo, hi, resolution in DATA:
        w, f, e = read_observation(path)
        k = (w >= lo) & (w < hi) & np.isfinite(f)
        w, f = w[k], f[k]
        sampled = {lab: sample_segments(mw, mf, w, resolution) for lab, (mw, mf) in models.items()}
        observed.append((name, w, f, sampled))
    edges = np.arange(910., 1990.001, args.panel_width)
    panels = list(zip(edges[:-1], edges[1:]))
    args.output.mkdir(parents=True, exist_ok=True)
    colors = {'without metals in structure': '#1f6fb4', 'with metals in structure': '#cb4335'}
    pages = [panels[i:i + args.panels_per_page] for i in range(0, len(panels), args.panels_per_page)]
    with PdfPages(args.output / 'structure-comparison.pdf') as pdf:
        for number, page in enumerate(pages, 1):
            fig, axes = plt.subplots(len(page), 1, figsize=(15, 2.3 * len(page) + 0.6), squeeze=False)
            labelled = set()
            for ax, (lo, hi) in zip(axes[:, 0], page):
                for name, w, f, sampled in observed:
                    k = (w >= lo) & (w < hi)
                    if k.sum() < 10:
                        continue
                    ax.plot(w[k], envelope_normalize(w[k], f[k]), color='0.25', lw=0.6,
                            label=None if 'observed' in labelled else 'observed')
                    labelled.add('observed')
                    for lab, mf in sampled.items():
                        ax.plot(w[k], envelope_normalize(w[k], mf[k]), color=colors[lab], lw=0.9,
                                alpha=0.9, label=None if lab in labelled else lab)
                        labelled.add(lab)
                    ax.text(0.003 if w[k][0] < lo + 1 else (w[k][0] - lo) / (hi - lo) + 0.003, 0.06, name,
                            transform=ax.transAxes, fontsize=8, color='0.3')
                ax.set_xlim(lo, hi)
                ax.set_ylim(0.0 if lo < 1240 else 0.35, 1.12)
                ax.tick_params(labelsize=8)
            axes[0, 0].legend(fontsize=8, loc='lower right', ncol=3, framealpha=0.9)
            axes[-1, 0].set_xlabel('observed vacuum wavelength (A)')
            fig.suptitle('G191-B2B: observed vs trace-metal NLTE model with/without metals in the temperature structure '
                         f'(40 depths; page {number}/{len(pages)})', fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(args.output / f'page-{number:02d}.png', dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
    print(f'wrote {len(pages)} pages to {args.output}')


if __name__ == '__main__':
    main()
