"""Compare the accepted D6 spectrum with newly reduced GTC/OSIRIS arrays.

This is a fixed-parameter, continuum-conditioned comparison. It does not fit
temperature, gravity, abundances, velocity, distance or reddening.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

MODEL_SHA = '6a8d28f7b0066d2e9e05749b22c715cfb2651fbd94e128c292ca64029ac28a2d'
DATA_SHA = 'd7f638b7393544a28a3c15230ba7aeb6b6b4076b99d826fa70b72097ed684757'
MEMBER = 'native/d6-a94fa52da97edec3e1dfd7f5/attempt-v1/model/spectrum.txt'
WINDOWS = {'blue': (3580., 4600.), 'green': (4600., 5600.), 'red': (5600., 7500.)}
TELLURIC = [(6860., 6960.), (7160., 7400.), (7580., 7700.)]
SKY = [5578.886770459, 5891.583, 5897.558, 6302.046463603, 6365.535508181]
RV = 384.0
C_KMS = 299792.458


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convolve(wave, flux, resolving_power):
    grid = np.linspace(np.log(wave[0]), np.log(wave[-1]), len(wave))
    sampled = np.interp(grid, np.log(wave), flux)
    sigma = 1 / (2.354820045 * resolving_power * (grid[1] - grid[0]))
    broadened = gaussian_filter1d(sampled, sigma, mode='reflect', truncate=4.)
    return np.interp(np.log(wave), grid, broadened)


def continuum_ratio(wave, data, model, valid):
    """Fifth-order smooth ratio, fit to 50-A median bins; retain narrow residuals."""
    x = 2*(wave-wave.min())/np.ptp(wave)-1
    edges = np.arange(wave.min(), wave.max()+50., 50.)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = valid & (wave >= lo) & (wave < hi)
        if g.sum() >= 5:
            bx.append(np.median(x[g])); by.append(np.median(data[g]/model[g]))
    bx, by = np.array(bx), np.array(by)
    keep = np.isfinite(by) & (by > 0)
    assert keep.sum() >= 10
    for _ in range(4):
        coefficient = np.polynomial.polynomial.polyfit(bx[keep], by[keep], 5)
        residual = by-np.polynomial.polynomial.polyval(bx, coefficient)
        sigma = 1.4826*np.median(np.abs(residual[keep]-np.median(residual[keep])))
        if sigma <= 0: break
        new = np.isfinite(by) & (by > 0) & (np.abs(residual) <= 4*sigma)
        if new.sum() < 10 or np.array_equal(new, keep): break
        keep = new
    coefficient = np.polynomial.polynomial.polyfit(bx[keep], by[keep], 5)
    scale = np.polynomial.polynomial.polyval(x, coefficient)
    assert np.all(scale > 0)
    return scale, coefficient


def bin_pairs(wave, data, ivar, model, valid):
    rows = []
    for j in range(0, len(wave)-1, 2):
        sl = slice(j, j+2)
        if not np.all(valid[sl]):
            rows.append([np.mean(wave[sl]), np.nan, np.nan, np.nan]); continue
        w = ivar[sl]
        rows.append([np.sum(wave[sl]*w)/sum(w), np.sum(data[sl]*w)/sum(w),
                     1/np.sqrt(sum(w)), np.sum(model[sl]*w)/sum(w)])
    return np.asarray(rows).T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Verified native spectrum.txt or diagnostic tar.gz')
    p.add_argument('--data', type=Path, default=Path(__file__).with_name('gtc-coadds.npz'))
    p.add_argument('--output', type=Path, default=Path('.'))
    a = p.parse_args(); a.output.mkdir(exist_ok=True, parents=True)
    assert sha(a.data) == DATA_SHA
    if str(a.model).endswith('.tar.gz'):
        with tarfile.open(a.model) as bank:
            members = [m for m in bank.getmembers() if m.name.endswith('/d6-a94fa52da97edec3e1dfd7f5/attempt-v1/model/spectrum.txt')]
            assert len(members) == 1
            raw = bank.extractfile(members[0]).read()
    else:
        raw = Path(a.model).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MODEL_SHA
    mw, mf = np.loadtxt(io.BytesIO(raw)).T
    assert np.all(np.diff(mw) > 0) and np.all(np.isfinite(mf)) and np.all(mf >= 0)
    base = convolve(mw, mf, 2500.)
    doppler = np.sqrt((1+RV/C_KMS)/(1-RV/C_KMS))
    plt.rcParams.update({'font.family':'DejaVu Serif', 'mathtext.fontset':'cm',
                         'font.size':12, 'axes.labelsize':14, 'axes.titlesize':15,
                         'xtick.direction':'in', 'ytick.direction':'in',
                         'xtick.top':True, 'ytick.right':True, 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), gridspec_kw={'width_ratios':[1.8,1]})
    results = {'model_sha256':MODEL_SHA, 'data_sha256':DATA_SHA,
               'source_commit':'14e206729d06f0989e03a0d2f1744a9cc518a136',
               'parameters_fitted':False, 'rv_fixed_kms':RV, 'resolution_R':2500.,
               'continuum_fit':'Fifth-order data/model ratio fit separately per grism to 50-A median bins; 4-sigma clipping of continuum bins only.',
               'metric':'Unweighted RMS (data-model)/smooth scaled-model continuum on all good unbinned coadd pixels in the stated windows after sky/telluric masks. No residual-based pixel clipping.',
               'telluric_observer_vacuum_A':TELLURIC, 'sky_observer_vacuum_A_plusminus4':SKY,
               'uncertainties':'Pipeline statistical diagonal variances; excludes calibration, continuum-fit, correlated-resampling and model uncertainties.',
               'grisms':{}}
    compared = {}
    with np.load(a.data, allow_pickle=False) as saved:
        for row,(label,(lo,hi)) in enumerate(WINDOWS.items()):
            ow = saved[label+'_wave']; wave = ow/doppler
            data = saved[label+'_flux']*doppler  # flux-density coordinate Jacobian
            ivar = saved[label+'_ivar']/doppler**2
            good = saved[label+'_mask'].astype(bool) & (ivar > 0) & np.isfinite(data)
            good &= (wave >= lo) & (wave <= hi) & (wave >= mw[0]) & (wave <= mw[-1])
            for l,h in TELLURIC: good &= ~((ow >= l) & (ow <= h))
            for line in SKY: good &= np.abs(ow-line) > 4.
            scope = (wave >= lo) & (wave <= hi)
            wave,data,ivar,good = wave[scope],data[scope],ivar[scope],good[scope]
            sampled = np.interp(wave,mw,base)
            scale,coeff = continuum_ratio(wave,data,sampled,good)
            model = sampled*scale
            smooth = gaussian_filter1d(sampled, 50./np.median(np.diff(wave)), mode='reflect')*scale
            residual = (data-model)/smooth
            rms = float(np.sqrt(np.mean(residual[good]**2)))
            sensitivity = {}
            for resolving_power in [2200.,2800.]:
                trial = np.interp(wave,mw,convolve(mw,mf,resolving_power))
                ts,_ = continuum_ratio(wave,data,trial,good)
                sensitivity[str(int(resolving_power))] = float(np.sqrt(np.mean(((data-trial*ts)/smooth)[good]**2)))
            results['grisms'][label] = dict(window_rest_vacuum_A=[lo,hi],good_native_pixels=int(good.sum()),
                    rms_continuum_fraction=rms, nominal_noise_rms=float(np.sqrt(np.mean((1/np.sqrt(ivar[good])/smooth[good])**2))),
                    resolution_sensitivity_rms=sensitivity, continuum_coefficients=coeff.tolist(),
                    median_sn=float(np.median(data[good]*np.sqrt(ivar[good]))))
            bw,bd,be,bm = bin_pairs(wave,data,ivar,model,good)
            left,right = axes[row]
            left.fill_between(bw,bd-be,bd+be,color='0.78',alpha=.55,lw=0)
            left.plot(bw,bd,color='0.15',lw=.8)
            left.plot(bw,bm,color='#2378a8',lw=1.1,ls='--')
            left.set(xlim=(lo,hi),title=f'GTC {dict(blue="R2500U",green="R2500V",red="R2500R")[label]} and D6 model',
                     ylabel=r'$F_\lambda$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
            g = good & np.isfinite(data)
            ymin,ymax = np.min((data-1/np.sqrt(ivar))[g]),np.max((data+1/np.sqrt(ivar))[g])
            left.set_ylim(ymin-.05*(ymax-ymin),ymax+.05*(ymax-ymin))
            right.axhline(0,color='#2378a8',ls='--',lw=.8)
            br = (bd-bm)/np.interp(bw,wave,smooth)
            bs = be/np.interp(bw,wave,smooth)
            right.fill_between(bw,br-bs,br+bs,color='0.78',alpha=.55,lw=0)
            right.plot(bw,br,color='0.15',lw=.7)
            right.set(xlim=(lo,hi),ylabel=r'$(F_{\rm data}-F_{\rm model})/F_{\rm smooth}$')
            for ax in [left,right]:
                ax.minorticks_on();ax.set_xlabel('Rest-frame vacuum wavelength (Å)')
            compared[label+'_wave']=wave;compared[label+'_data']=data;compared[label+'_model']=model
            compared[label+'_good']=good;compared[label+'_ivar']=ivar;compared[label+'_smooth']=smooth
    fig.tight_layout(h_pad=1.6,w_pad=1.8)
    for suffix in ['png','pdf']:fig.savefig(a.output/f'd6_gtc_comparison.{suffix}',dpi=250)
    plt.close(fig)
    np.savez_compressed(a.output/'comparison-arrays.npz',**compared)
    (a.output/'COMPARISON.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results['grisms'],indent=2))


if __name__ == '__main__': main()
