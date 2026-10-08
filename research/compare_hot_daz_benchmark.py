#!/usr/bin/env python3
"""Compare a fixed-parameter OpenWD prediction with the G191-B2B STIS atlas.

Only local continuum nuisance parameters are fitted. Teff, logg, C/H, Si/H,
photospheric velocity and resolution stay fixed. The Gaussian R=144000
instrumental approximation is explicitly distinguished from an exact STIS LSF.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz
from hot_daz_benchmark import BENCHMARK, checked_files, read_observation

C_KMS = 299792.458
REGIONS = {
    'ciii_1175': dict(label='C III 1175 multiplet', window=(1173.8,1177.),
                     anchors=((1173.8,1174.55),(1176.65,1177.)), score=(1174.85,1176.6)),
    'lyalpha': dict(label=r'Ly$\alpha$', window=(1190.,1242.),
                   anchors=((1190.,1195.),(1236.,1242.)), score=(1195.,1236.)),
    'siiv_1393': dict(label='Si IV 1393.75', window=(1393.25,1394.35),
                     anchors=((1393.25,1393.52),(1394.12,1394.35)), score=(1393.55,1394.1)),
    'siiv_1402': dict(label='Si IV 1402.77', window=(1402.25,1403.35),
                     anchors=((1402.25,1402.4),(1403.12,1403.35)), score=(1402.6,1403.1)),
}


def doppler_factor(velocity):
    beta = velocity/C_KMS
    if not np.isfinite(beta) or abs(beta) >= 1:
        raise ValueError('velocity must be finite and subluminal')
    return np.sqrt((1+beta)/(1-beta))


def ism_lyalpha(wave, log_column=18.18, velocity=19.4, b_kms=10.):
    """Single effective H I Voigt absorber; published total column, assumed b.

    N(H I) is from Lemoine et al. 2002, ApJS 140, 67 (astro-ph/0112180).
    b=10 km/s and v=19.4 km/s are diagnostic choices; the unresolved cloud
    structure is not inferred. The core is excluded from the comparison score.
    """
    if not np.isfinite(log_column) or not np.isfinite(b_kms) or b_kms <= 0:
        raise ValueError('invalid ISM column or Doppler parameter')
    c = 2.99792458e10
    nu0 = c/(1215.6701e-8)
    nu = c/(np.asarray(wave)/doppler_factor(velocity)*1e-8)
    width = nu0*b_kms/C_KMS
    profile = wofz((nu-nu0)/width + 1j*6.265e8/(4*np.pi*width)).real/(np.sqrt(np.pi)*width)
    # pi e^2 / (m_e c) in cgs; oscillator strength f_12=0.4164.
    cross_section = 0.02654008856*0.4164*profile
    return np.exp(-10.**log_column*cross_section)


def instrument_sample(model_wave, model_flux, observed_wave, *, velocity=23.8,
                      resolving_power=144000., transmission=None, velocity_step=.15):
    """Doppler-shift, convolve at constant R, and average over observed bins.

    Each region is processed separately. Model padding and gap coverage are
    checked so interpolation never bridges an absent spectral segment.
    """
    w, f, obs = map(lambda x: np.asarray(x,dtype=float),(model_wave,model_flux,observed_wave))
    if (w.ndim != 1 or f.shape != w.shape or obs.ndim != 1 or len(obs)<2
            or np.any(np.diff(w)<=0) or np.any(np.diff(obs)<=0)
            or np.any(~np.isfinite(w)) or np.any(~np.isfinite(f)) or np.any(~np.isfinite(obs))
            or not np.isfinite(resolving_power) or resolving_power<=0
            or not np.isfinite(velocity_step) or velocity_step<=0):
        raise ValueError('invalid spectrum or sampling parameters')
    factor = doppler_factor(velocity)
    edges = np.r_[obs[0]-(obs[1]-obs[0])/2,(obs[1:]+obs[:-1])/2,obs[-1]+(obs[-1]-obs[-2])/2]
    sigma_v = C_KMS/resolving_power/np.sqrt(8*np.log(2))
    lo = edges[0]*np.exp(-6*sigma_v/C_KMS)
    hi = edges[-1]*np.exp(6*sigma_v/C_KMS)
    shifted = w*factor
    if shifted[0] > lo or shifted[-1] < hi:
        raise ValueError('model does not cover region plus convolution padding')
    i = max(0,np.searchsorted(shifted,lo)-1)
    j = min(len(w),np.searchsorted(shifted,hi)+1)
    if np.max(np.diff(shifted[i:j])) > .1:
        raise ValueError('model has an unsynthesized gap in this region')
    grid = np.exp(np.arange(np.log(lo),np.log(hi)+velocity_step/C_KMS,velocity_step/C_KMS))
    flux = np.interp(grid,shifted,f/factor)
    if transmission is not None:
        flux *= transmission(grid)
    smooth = gaussian_filter1d(flux,sigma_v/velocity_step,mode='nearest',truncate=5.)
    integral = cumulative_trapezoid(smooth,grid,initial=0.)
    return np.diff(np.interp(edges,grid,integral))/np.diff(edges)


def fit_continuum(wave, flux, error, anchors):
    """Linear local continuum with downward clipping of unlisted absorption."""
    x = (wave-np.mean(wave))/np.ptp(wave)
    design = np.column_stack((np.ones_like(x),x))
    good = anchors.copy()
    if np.count_nonzero(good)<8:
        raise ValueError('too few uncontaminated continuum samples')
    for _ in range(5):
        coeff = np.linalg.lstsq(design[good]/error[good,None],flux[good]/error[good],rcond=None)[0]
        model = design@coeff
        residual = (flux-model)/error
        updated = anchors & (residual > -2.) & (residual < 4.)
        if np.count_nonzero(updated)<8 or np.array_equal(updated,good):
            break
        good=updated
    if np.any(model<=0):
        raise ValueError('nonpositive fitted continuum')
    return model,good


def blend_mask(wave, lines, target_element, target_ion, half_width_kms=6.):
    mask = np.zeros(len(wave),dtype=bool)
    for row in lines:
        if row['element']==target_element and row['ion']==target_ion and row['origin']=='PHOT':
            continue
        center=row['observed_wavelength']
        mask |= abs(wave-center)<center*half_width_kms/C_KMS
    return mask


def verify_benchmark_parameters(metadata):
    expected = dict(teff=BENCHMARK['effective_temperature'],logg=BENCHMARK['logg'],log_h_he=5.)
    for name,value in expected.items():
        if not np.isclose(metadata.get('host_parameters',{}).get(name,np.nan),value,
                          rtol=0,atol=1e-10):
            raise ValueError(f'model {name} does not match the declared G191-B2B benchmark')
    for element,value in [('C',BENCHMARK['carbon_from_ciii']['value']),
                          ('Si',BENCHMARK['silicon_from_siiv']['value'])]:
        if not np.isclose(metadata.get('abundances',{}).get(element,np.nan),np.log10(value),
                          rtol=0,atol=1e-10):
            raise ValueError(f'model {element}/H does not match the declared G191-B2B benchmark')


def compare(model_directory, data_directory, output, *, allow_unconverged=False, resolving_power=144000.):
    model_directory, output = Path(model_directory),Path(output)
    metadata=json.loads((model_directory/'metadata.json').read_text())
    verify_benchmark_parameters(metadata)
    if not allow_unconverged and (metadata.get('smoke_only') or not metadata.get('converged')
            or metadata.get('host_atmosphere_convergence')!='converged'):
        raise ValueError('comparison requires a converged physical host and metal populations')
    predicted=np.load(model_directory/'spectrum.npz')
    model_wave=predicted['wavelength']; model_flux=predicted['flux']; background=predicted['background_flux']
    paths=checked_files(data_directory)
    wave,flux,error=read_observation(paths['spectrum'])
    # Read the full catalogue for continuum and Ly alpha contaminant masks.
    import shlex
    lines=[]
    for text in paths['lines'].read_text().splitlines():
        if not text.strip() or text.lstrip().startswith('#'):continue
        f=shlex.split(text)
        lines.append(dict(observed_wavelength=float(f[0]),element=f[4],ion=f[5],origin=f[11]))
    output.mkdir(parents=True,exist_ok=True)
    metrics={}; arrays={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,8),layout='constrained')
    outer=fig.add_gridspec(2,2)
    for index,(name,region) in enumerate(REGIONS.items()):
        lo,hi=region['window'];take=(wave>=lo)&(wave<=hi)
        w,y,e=wave[take],flux[take],error[take]
        photosphere=instrument_sample(model_wave,model_flux,w,resolving_power=resolving_power)
        host=instrument_sample(model_wave,background,w,resolving_power=resolving_power)
        target=('C','III') if name=='ciii_1175' else ('Si','IV') if name.startswith('siiv') else ('H','I')
        contaminants=blend_mask(w,lines,*target)
        anchors=np.zeros(len(w),dtype=bool)
        for a,b in region['anchors']: anchors|=(w>=a)&(w<=b)
        anchors &= ~blend_mask(w,lines,'','')
        continuum,anchor_used=fit_continuum(w,y,e,anchors)
        # Two nuisance coefficients for surface-flux dilution/local calibration,
        # fitted ONLY in the prescribed continuum windows, never in line cores.
        x=(w-np.mean(w))/np.ptp(w)
        design=photosphere[:,None]*np.column_stack((np.ones(len(w)),x))
        scale_coeff=np.linalg.lstsq(design[anchor_used]/e[anchor_used,None],
                                  y[anchor_used]/e[anchor_used],rcond=None)[0]
        scale=scale_coeff[0]+scale_coeff[1]*x
        if np.any(scale<=0):raise ValueError('unphysical local continuum scaling')
        pure=photosphere*scale/continuum
        model=pure.copy()
        if name=='lyalpha':
            model=instrument_sample(model_wave,model_flux,w,resolving_power=resolving_power,
                                    transmission=ism_lyalpha)*scale/continuum
            contaminants |= (w>1214.)&(w<1217.5)
        if name.startswith('siiv'):
            center=1393.795 if name=='siiv_1393' else 1402.809
            contaminants |= abs(w-center)<center*10./C_KMS
        observed=y/continuum; uncertainty=e/continuum
        score=(w>=region['score'][0])&(w<=region['score'][1])&~contaminants
        residual=observed-model
        dw=np.diff(np.r_[w[0]-(w[1]-w[0])/2,(w[1:]+w[:-1])/2,w[-1]+(w[-1]-w[-2])/2])
        whole=(w>=region['score'][0])&(w<=region['score'][1])
        metrics[name]=dict(
            retained_pixels=int(score.sum()),excluded_pixels=int((whole&contaminants).sum()),
            normalized_rms=float(np.sqrt(np.mean(residual[score]**2))),
            median_absolute_residual=float(np.median(abs(residual[score]))),
            chi2_per_pixel=float(np.mean((residual[score]/uncertainty[score])**2)),
            observed_full_window_equivalent_width_mA=float(np.sum((1-observed[whole])*dw[whole])*1000),
            predicted_full_window_equivalent_width_mA=float(np.sum((1-model[whole])*dw[whole])*1000),
            continuum_coefficients=scale_coeff.tolist(),
            equivalent_width_scope='full window includes blends/nonphotospheric absorption; not a clean elemental EW',
            score_scope='conditional on local continuum; pixel correlations/systematic errors not included')
        if name == 'ciii_1175':
            # These two main components have no competing Fe/Ni identification
            # in the atlas. The fixed +/-20 km/s apertures avoid the nearby weak
            # 1176.100-A C III feature that is absent from the compact atom.
            components = {}
            for rest in (1175.98705003,1176.36969723):
                center = rest*doppler_factor(23.8)
                aperture = abs(w-center) < center*20./C_KMS
                observed_ew = float(np.sum((1-observed[aperture])*dw[aperture])*1000)
                predicted_ew = float(np.sum((1-model[aperture])*dw[aperture])*1000)
                components[f'{rest:.3f}'] = dict(
                    observed_aperture_ew_mA=observed_ew,
                    predicted_aperture_ew_mA=predicted_ew,
                    observed_statistical_error_mA=float(np.linalg.norm(
                        uncertainty[aperture]*dw[aperture])*1000),
                    predicted_to_observed_ew_ratio=predicted_ew/observed_ew,
                    half_width_kms=20.,
                    scope='fixed aperture; statistical error excludes continuum uncertainty')
            metrics[name]['isolated_components'] = components
        arrays[name+'_wavelength']=w;arrays[name+'_observed']=observed
        arrays[name+'_error']=uncertainty;arrays[name+'_model']=model
        arrays[name+'_photosphere']=pure;arrays[name+'_score_mask']=score
        sub=outer[index//2,index%2].subgridspec(2,1,height_ratios=(3,1),hspace=.02)
        ax=fig.add_subplot(sub[0]);res=fig.add_subplot(sub[1],sharex=ax)
        ax.plot(w,observed,color='.25',lw=.75,label='STIS')
        ax.fill_between(w,observed-uncertainty,observed+uncertainty,color='.5',alpha=.2)
        if name=='lyalpha':
            ax.plot(w,pure,color='#3e80b6',ls='--',lw=1.1,label='Photosphere only')
        ax.plot(w,model,color='#d25333',lw=1.4,label='OpenWD + H I screen' if name=='lyalpha' else 'OpenWD')
        if name!='lyalpha':
            ax.plot(w,host*scale/continuum,color='#3e80b6',ls=':',lw=1.,label='H/He background')
        res.axhline(0,color='.6',lw=.7)
        res.plot(w,residual,color='.7',lw=.6)
        res.plot(w,np.where(score,residual,np.nan),color='#244b73',lw=.8)
        # Contaminant markers show the excluded regions without concealing data.
        ax.fill_between(w,0,1,where=contaminants,transform=ax.get_xaxis_transform(),
                        color='#c9b5d4',alpha=.2,step='mid')
        ax.set(title=region['label'],ylabel='Locally normalized flux',xlim=(lo,hi))
        ax.tick_params(labelbottom=False);ax.legend(fontsize=8,loc='best')
        res.set(xlabel='Observed vacuum wavelength (Angstrom)',ylabel='Data − model')
        ax.text(.98,.96,f'RMS {metrics[name]["normalized_rms"]:.3f}',
                transform=ax.transAxes,ha='right',va='top',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=2.))
    status='CONVERGED FIXED-HOST PROTOTYPE' if (metadata.get('converged')
        and not metadata.get('smoke_only') and metadata.get('host_atmosphere_convergence')=='converged') else 'UNCONVERGED / NUMERICAL DIAGNOSTIC'
    fig.suptitle('G191-B2B: fixed published parameters and C/Si abundances\n'
                 f'T = 52,500 K, log g = 7.53, C/H = 1.72e−7, Si/H = 3.68e−7 | {status}\n'
                 'Shading: catalogue blends / non-photospheric components excluded from the score',fontsize=11)
    fig.savefig(output/'comparison.png',dpi=170)
    fig.savefig(output/'comparison.pdf');plt.close(fig)
    report=dict(target='G191-B2B',reference=BENCHMARK['reference'],model_metadata=metadata,
                comparison_completed=True,observationally_validated=False,
                spectrum_sha256=hashlib.sha256((model_directory/'spectrum.npz').read_bytes()).hexdigest(),
                observation_files={name:dict(path=str(path.resolve()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in paths.items()},
                velocity_kms=23.8,resolution=dict(model='Gaussian constant resolving power approximation',R=resolving_power),
                ism=dict(log_n_hi=18.18,velocity_kms=19.4,b_kms=10.,
                         column_reference='https://arxiv.org/abs/astro-ph/0112180',
                         note='single effective component, fixed diagnostic b; core excluded'),
                fitted_quantities=['two local continuum nuisance coefficients per region'],
                fixed_quantities=['Teff','logg','He/H',*[e+'/H' for e in metadata.get('abundances',{'C':0,'Si':0})],
                                  'photospheric velocity','resolving power','ISM screen'],
                masked_regions='catalogue contaminants +/-6 km/s; nonphotospheric Si IV +/-10 km/s; Ly alpha 1214-1217.5 A',
                limitations=[('C/Si-only opacity; other metal blanketing/blends missing'
                              if set(metadata.get('abundances',{})) <= {'C','Si'} else
                              'Additional metals included; see model metadata for NLTE/LTE treatment'),
                             'provisional compact atoms and approximate collision/photoionization data',
                             'fixed host; no metal thermal or charge feedback',
                             'Gaussian instrumental approximation; coadded exposure LSF not reconstructed',
                             'continuum normalization suppresses absolute-flux information'],metrics=metrics)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'comparison_arrays.npz',**arrays)
    from astropy.io import fits
    primary = fits.PrimaryHDU()
    for key,value in [('OBJECT','G191-B2B'),('TEFF',52500.),('LOGG',7.53),
                      ('C_H',1.72e-7),('SI_H',3.68e-7),('HE_H',1e-5),
                      ('VSTAR',23.8),('RESPOWER',resolving_power),
                      ('METCONV',bool(metadata.get('converged'))),
                      ('HOSTCONV',metadata.get('host_atmosphere_convergence','unknown')),
                      ('SMOKE',bool(metadata.get('smoke_only'))),
                      ('VALIDATE',False)]:
        primary.header[key] = value
    for element,log_abundance in metadata.get('abundances',{}).items():
        primary.header[element.upper()+'_H'] = (10.**log_abundance, 'elemental number ratio relative to H')
    primary.header['ABUNCON'] = 'N(element)/N(H)'
    primary.header['NLTEELEM'] = ','.join(metadata.get('nlte_elements',metadata.get('levels_per_charge',{})))
    primary.header['LTEELEM'] = ','.join(metadata.get('lte_background_elements',[]))
    primary.header['HOSTFIX'] = not metadata.get('atmosphere_recomputed',False)
    primary.header['MTHERM'] = (False,'Metal thermal feedback solved')
    primary.header['MCHARGE'] = (False,'Metal charge feedback solved')
    primary.header['HISTORY'] = 'Fixed published abundances; only local continuum nuisance parameters fitted.'
    primary.header['HISTORY'] = 'Gaussian STIS response approximation; see comparison.json for masks and caveats.'
    hdus = [primary]
    for name in REGIONS:
        columns = []
        for field in ('wavelength','observed','error','model','photosphere','score_mask'):
            columns.append(fits.Column(name=field.upper(),
                format='L' if field=='score_mask' else 'D',
                unit='Angstrom' if field=='wavelength' else None,
                array=arrays[name+'_'+field]))
        hdu = fits.BinTableHDU.from_columns(columns,name=name.upper())
        hdu.header['FLUXTYPE'] = 'locally normalized; dimensionless'
        hdu.header['WAVEMED'] = 'vacuum, observed frame'
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(output/'comparison.fits',overwrite=True,checksum=True)
    rest = fits.BinTableHDU.from_columns([
        fits.Column(name='WAVELENGTH',format='D',unit='Angstrom',array=model_wave),
        fits.Column(name='SURFACE_FLUX',format='D',unit='erg s-1 cm-2 Angstrom-1',array=model_flux),
        fits.Column(name='H_HE_FLUX',format='D',unit='erg s-1 cm-2 Angstrom-1',array=background),
    ],name='PREDICTION')
    rest.header['WAVEMED']='vacuum, stellar rest frame'
    rest.header['FLUXTYPE']='emergent surface F_lambda'
    rest_primary=fits.PrimaryHDU(header=primary.header.copy())
    for key in ('HISTORY','VSTAR','RESPOWER'):
        if key in rest_primary.header:del rest_primary.header[key]
    rest_primary.header['HISTORY']='Intrinsic prediction: no radial velocity, instrument or ISM applied.'
    rest_primary.header['HISTORY']='No continuum normalization or observed-flux scale is applied to these surface fluxes.'
    rest_primary.header['HISTORY']='METCONV certifies fixed-host trace populations only; MTHERM and MCHARGE are false.'
    fits.HDUList([rest_primary,rest]).writeto(output/'prediction.fits',overwrite=True,checksum=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('results/hot-daz/g191-b2b-model'))
    parser.add_argument('--data',type=Path,default=Path('results/hot-daz/g191-b2b'))
    parser.add_argument('--output',type=Path,default=Path('results/hot-daz/g191-b2b-comparison'))
    parser.add_argument('--allow-unconverged',action='store_true')
    parser.add_argument('--resolving-power',type=float,default=144000.)
    args=parser.parse_args()
    report=compare(args.model,args.data,args.output,allow_unconverged=args.allow_unconverged,
                   resolving_power=args.resolving_power)
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__': main()
