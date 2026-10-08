#!/usr/bin/env python3
"""Native-pixel Balmer comparisons to public CALSPEC STIS observations.

No atmospheric parameters or radial velocities are fitted. Model extensions of
CALSPEC composite SEDs are excluded by explicit, header-verified optical bounds.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wd_spectra.validation.observed import local_normalize, normalized_profile_metrics
from wd_spectra._compat import trapezoid
from wd_spectra._convergence import recorded_equilibrium_status

TARGETS = {
    'gd153': dict(teff=40204., logg=7.82, velocity=8.3,
                  reference='TMAP pure-H NLTE', resolution='gd153_stisnic_006.fits'),
    'g191b2b': dict(teff=59000., logg=7.60, velocity=22.1,
                   reference='TLUSTY metal-blanketed NLTE', resolution=None),
}
LINES = [('H delta', 4000., 4200.), ('H gamma', 4220., 4460.),
         ('H beta', 4700., 5020.), ('H alpha', 6400., 6740.)]


def read_observation(directory, star):
    path = directory / (star+'_stiswfcnic_004.fits')
    with fits.open(path) as h:
        table = h[1].data.copy()
        pedigree = '\n'.join(str(c.value) for c in h[0].header.cards
                             if c.keyword in ('HISTORY', 'COMMENT'))
    # The checked headers identify 3500--7500 A as STIS for both objects.
    w = np.asarray(table['WAVELENGTH'], float)
    selected = (w >= 3500) & (w <= 7500)
    selected &= (table['DATAQUAL'] == 1) & (table['STATERROR'] > 0)
    selected &= np.isfinite(table['FLUX']) & (table['FLUX'] > 0)
    w = w[selected]
    fwhm = np.asarray(table['FWHM'][selected], float)
    resolution_path = path
    if TARGETS[star]['resolution']:
        # Current GD153 FWHM is an exact copy of FLUX, including in version003.
        # Use the same star's earlier STIS G430L/G750L resolution vector.
        # This is an approximate Gaussian LSF; the substitution is recorded.
        resolution_path = directory / TARGETS[star]['resolution']
        old = fits.getdata(resolution_path)
        fwhm = np.interp(w, old['WAVELENGTH'], old['FWHM'])
    if np.any(~np.isfinite(fwhm)) or np.any((fwhm < 1) | (fwhm > 20)):
        raise ValueError('Invalid optical resolution metadata')
    return (w, np.asarray(table['FLUX'][selected], float),
            np.asarray(table['STATERROR'][selected], float), fwhm,
            dict(observation=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 pedigree=pedigree, resolution_file=str(resolution_path.resolve()),
                 resolution_sha256=hashlib.sha256(resolution_path.read_bytes()).hexdigest(),
                 resolution_method='Gaussian using archived FWHM; GD153 uses earlier same-star STIS resolution',
                 selection='3500--7500 A; DATAQUAL=1; finite positive flux and statistical error',
                 error_model='STATERROR only; chi-square is diagnostic, excludes correlated calibration/continuum uncertainty'))


def convolve(wave, flux, output, fwhm):
    result = np.empty_like(output)
    for i, (center, width) in enumerate(zip(output, fwhm)):
        sigma = width / np.sqrt(8*np.log(2))
        m = (wave >= center-5*sigma) & (wave <= center+5*sigma)
        if m.sum() < 3:
            raise ValueError('Model undersamples the observed LSF')
        weights = np.exp(-.5*((wave[m]-center)/sigma)**2)
        result[i] = trapezoid(flux[m]*weights, wave[m])/trapezoid(weights, wave[m])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--star', choices=TARGETS)
    p.add_argument('--model', type=Path)
    args = p.parse_args()
    if args.model and not args.star:
        p.error('--model requires --star')
    stars = [args.star] if args.star else list(TARGETS)
    fig, axes = plt.subplots(len(stars), len(LINES), figsize=(16, 4*len(stars)),
                             squeeze=False, constrained_layout=True)
    report = dict(scope='fixed-parameter normalized optical Balmer profiles',
                  comparison_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), targets={})
    for row, star in enumerate(stars):
        w, f, error, widths, source = read_observation(args.data, star)
        refpath = args.data / (star+'_mod_012.fits')
        ref = fits.getdata(refpath)
        # Archive model wavelengths already include the header radial velocity.
        predictions = {TARGETS[star]['reference']: convolve(ref['WAVELENGTH'], ref['FLUX'], w, widths)}
        candidate = None
        if args.model:
            record = json.loads((args.model/'metadata.json').read_text())
            cfg = record['config']
            if cfg['effective_temperature'] != TARGETS[star]['teff'] or cfg['logg'] != TARGETS[star]['logg']:
                raise ValueError('Candidate parameters differ from fixed target parameters')
            mw, mf = np.loadtxt(args.model/'spectrum.txt', unpack=True)
            if (mw.ndim!=1 or len(mw)<3 or np.any(~np.isfinite(mw)) or
                    np.any(mw<=0) or np.any(np.diff(mw)<=0) or
                    np.any(~np.isfinite(mf)) or np.any(mf<=0)):
                raise ValueError('Candidate spectrum must have increasing finite wavelengths and finite positive flux')
            status = recorded_equilibrium_status(record['atmosphere_metadata'])
            predictions['OpenWD ('+status+')'] = convolve(mw*(1+TARGETS[star]['velocity']/299792.458), mf, w, widths)
            candidate = dict(path=str(args.model.resolve()), config=cfg, metadata=record['model_metadata'],
                             spectrum_sha256=hashlib.sha256((args.model/'spectrum.txt').read_bytes()).hexdigest(),
                             atmosphere_convergence_status=status,
                             equilibrium_certificate=record['atmosphere_metadata'].get('equilibrium_certificate'))
        target = dict(parameters=TARGETS[star], source=source, candidate=candidate,
                      reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(), metrics={})
        for col, (line, lo, hi) in enumerate(LINES):
            ax = axes[row,col]
            ow, of = local_normalize(w,f,lo,hi)
            ax.plot(ow,of,'k.',ms=2,label='Observed STIS')
            for label, prediction in predictions.items():
                mw,mf = local_normalize(w,prediction,lo,hi)
                ax.plot(mw,mf,lw=1,label=label)
                target['metrics'].setdefault(label,{})[line] = normalized_profile_metrics(
                    w,f,w,prediction,lo,hi,observed_inverse_variance=error**-2)
            ax.set_title(star+' '+line);ax.set_xlabel('Vacuum wavelength (A)')
            if col == 0:ax.legend(fontsize=8)
        report['targets'][star] = target
    args.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(args.output/'balmer-profiles.png',dpi=150)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(args.output/'comparison.json')

if __name__ == '__main__':
    main()
