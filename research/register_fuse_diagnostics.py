#!/usr/bin/env python3
"""Measure a local catalogue-to-coadd offset without using a stellar model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from hot_daz_benchmark import read_observation
from compare_hot_composition import line_catalogue


def gaussian_bins(lower,upper,center,sigma):
    return sigma*np.sqrt(np.pi/2)/(upper-lower)*(erf((upper-center)/(np.sqrt(2)*sigma))
                                               -erf((lower-center)/(np.sqrt(2)*sigma)))


def fit_centroid(wavelength,flux,error,catalogue_center):
    """Bin-integrated empirical absorption profile; no atmosphere input."""
    wave=np.asarray(wavelength);flux=np.asarray(flux);error=np.asarray(error)
    if np.any(np.diff(wave)<=0) or np.any(error<=0):raise ValueError('invalid observation grid/errors')
    edges=np.r_[wave[0]-(wave[1]-wave[0])/2,(wave[1:]+wave[:-1])/2,
                wave[-1]+(wave[-1]-wave[-2])/2]
    # Pilot search found the local displacement lies near +0.05 A. The
    # fit centre remains free; this window keeps adjacent lines outside it.
    selected=abs(wave-catalogue_center-.05)<.20
    if selected.sum()<9:raise ValueError('too few samples to register line')
    x=wave[selected]-catalogue_center
    lo=edges[:-1][selected]-catalogue_center;hi=edges[1:][selected]-catalogue_center
    norm=np.median(flux[selected]);y=flux[selected]/norm;e=error[selected]/norm
    def model(x,continuum,slope,depth,offset,sigma):
        return continuum+slope*x-depth*gaussian_bins(lo,hi,offset,sigma)
    parameters,covariance=curve_fit(model,x,y,sigma=e,absolute_sigma=True,
        p0=[1.,0.,.2,.05,.035],bounds=([.5,-2.,0.,-.1,.015],[2.,2.,1.8,.16,.09]),maxfev=10000)
    residual=(model(x,*parameters)-y)/e
    reduced_chi2=float(residual@residual/(len(x)-len(parameters)))
    uncertainties=np.sqrt(np.diag(covariance)*max(1.,reduced_chi2))
    snr=float(parameters[2]/uncertainties[2])
    accepted=bool(snr>5 and uncertainties[3]<.015 and .0151<parameters[4]<.0899
                  and -.099<parameters[3]<.159)
    result=dict(catalogue_center=catalogue_center,coadd_center=float(catalogue_center+parameters[3]),
        offset_angstrom=float(parameters[3]),inflated_center_error_angstrom=float(uncertainties[3]),
        gaussian_sigma_angstrom=float(parameters[4]),depth_snr=snr,
        reduced_chi2=reduced_chi2,accepted=accepted,parameters=parameters.tolist())
    return result,(x,y,e,model(x,*parameters))


def register(directory,output):
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prefix='hlsp_wd-linelist_fuse_spec_g191-b2b_fuv_v1_'
    spectrum=directory/(prefix+'coadd-spec.fits');catalogue=directory/(prefix+'linelist.txt')
    manifest=json.loads((directory/'manifest.json').read_text())
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (spectrum,catalogue)}
    for name,checksum in inputs.items():
        if checksum!=manifest[name]['sha256']:raise ValueError('observation input checksum changed')
    wave,flux,error=read_observation(spectrum);rows=line_catalogue(catalogue)
    specifications=[('N','IV',1122.055),('Si','IV',1122.485),('Si','IV',1128.340)]
    anchors=[];profiles=[]
    for element,ion,rest in specifications+[('P','V',1128.008)]:
        matching=[r for r in rows if r['element']==element and r['ion']==ion
                  and r['origin']=='PHOT' and abs(r['rest']-rest)<.002]
        if len(matching)!=1:raise ValueError('ambiguous registration feature')
        fit,profile=fit_centroid(wave,flux,error,matching[0]['observed'])
        fit.update(element=element,ion=ion,rest_wavelength=rest)
        if element!='P':
            if not fit['accepted']:raise ValueError('registration anchor failed quality check')
            anchors.append(fit)
        else:held_out=fit
        profiles.append((fit,profile))
    offsets=np.array([r['offset_angstrom'] for r in anchors])
    offset=float(np.median(offsets))
    report=dict(method='median catalogue-to-coadd centroid offset of three non-phosphorus photospheric lines',
        input_sha256=inputs,anchors=anchors,held_out_PV1128=held_out,
        regions=[dict(rest_min=1121.5,rest_max=1129.2,offset_angstrom=offset,
                      anchor_offset_standard_deviation_angstrom=float(np.std(offsets,ddof=1)))],
        qualifications=['Local comparison registration, not a recalibrated FUSE spectrum or stellar RV measurement.',
            'P V is excluded from the shift estimate; its centroid is only a held-out check.',
            'Gaussian profiles and coarse 0.04 A bins limit accuracy; anchor scatter is not a full systematic error.',
            'Do not extrapolate this offset to P V 1117 or the S IV windows.',
            'The input coadd and intrinsic model spectra are unchanged.'],
        source='https://archive.stsci.edu/prepds/wd-linelist/',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'registration.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(fit,(x,y,e,model)) in zip(axes.flat,profiles):
        ax.errorbar(x,y,e,color='.25',lw=.8,marker='.',label='FUSE coadd')
        ax.plot(x,model,c='#389076',label='Empirical centroid fit')
        ax.axvline(0,c='#7c90a0',ls='--',label='Catalogue position')
        ax.axvline(offset,c='#c3473d',ls='--',label='Common local correction')
        suffix=' (held out)' if fit['element']=='P' else ' (anchor)'
        ax.set(title=f'{fit["element"]} {fit["ion"]} {fit["rest_wavelength"]:.3f} Å'+suffix,
               xlabel='Offset from catalogue position (Å)',ylabel='Locally scaled flux')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Independent local FUSE registration: +{offset:.4f} Å\n'
                 'No stellar model or phosphorus line used to estimate the correction')
    fig.savefig(output/'registration.png',dpi=160);fig.savefig(output/'registration.pdf');plt.close(fig)
    print(json.dumps(dict(offset_angstrom=offset,anchor_scatter_angstrom=float(np.std(offsets,ddof=1)),
                         held_out_PV_residual_angstrom=held_out['offset_angstrom']-offset)))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='results/hot-daz/multimetal-data')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();register(args.data,args.output)
