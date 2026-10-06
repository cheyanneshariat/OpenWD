#!/usr/bin/env python3
"""Compare published new-metal diagnostics without fitting their abundances."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import numpy as np
from hot_daz_benchmark import checked_files,read_observation
from compare_hot_daz_benchmark import doppler_factor,instrument_sample,fit_continuum,C_KMS

# Laboratory wavelengths in the atlas, selected before seeing new predictions.
DIAGNOSTICS=[
 ('NV1238','N','V',1238.821,'e140h'),('NV1242','N','V',1242.804,'e140h'),
 ('OIV1338','O','IV',1338.615,'e140h'),('OIV1343','O','IV',1343.514,'e140h'),
 ('AlIII1854','Al','III',1854.716,'e230h'),('AlIII1862','Al','III',1862.790,'e230h'),
 ('PV1117','P','V',1117.977,'fuv'),('PV1128','P','V',1128.008,'fuv'),
 ('SIV1062','S','IV',1062.662,'fuv'),('SIV1072','S','IV',1072.974,'fuv'),
 ('FeV1409','Fe','V',1409.453,'e140h'),('NiV1306','Ni','V',1306.624,'e140h')]


def line_catalogue(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        f=shlex.split(line)
        result.append(dict(observed=float(f[0]),element=f[4],ion=f[5],rest=float(f[6]),
                           velocity=float(f[8]),origin=f[11]))
    return result


def registration_offset(registration, rest_wavelength):
    matches=[region for region in registration.get('regions',[])
             if region['rest_min']<=rest_wavelength<=region['rest_max']]
    if len(matches)>1:raise ValueError('overlapping local registration regions')
    return float(matches[0]['offset_angstrom']) if matches else 0.


def comparison(models, directory, output, *, fuse_registration=None):
    directory,output=Path(directory),Path(output)
    pinned=json.loads((directory/'manifest.json').read_text())
    for name,item in pinned.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'input checksum mismatch: {name}')
    registration={} if fuse_registration is None else json.loads(Path(fuse_registration).read_text())
    for name,checksum in registration.get('input_sha256',{}).items():
        if name not in pinned or pinned[name]['sha256']!=checksum:
            raise ValueError('registration applies to different observation inputs')
    hst=checked_files('results/hot-daz/g191-b2b')
    observations={'e140h':read_observation(hst['spectrum'])}
    catalogues={'e140h':line_catalogue(hst['lines'])}
    for instrument,prefix in [('e230h','hst_stis'),('fuv','fuse_spec')]:
        base=f'hlsp_wd-linelist_{prefix}_g191-b2b_{instrument}_v1_'
        observations[instrument]=read_observation(directory/(base+'coadd-spec.fits'))
        catalogues[instrument]=line_catalogue(directory/(base+'linelist.txt'))
    spectra={};metadata={}
    for label,path in models.items():
        path=Path(path);m=json.loads((path/'metadata.json').read_text())
        if m['host_parameters']!={'teff':52500.,'logg':7.53,'log_h_he':5.}:
            raise ValueError('inconsistent benchmark host')
        metadata[label]=m
        with np.load(path/'spectrum.npz') as p:
            spectra[label]=(p['wavelength'].copy(),p['flux'].copy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(14,12),layout='constrained')
    report={};arrays={}
    for ax,diag in zip(axes.flat,DIAGNOSTICS):
        name,element,ion,rest,instrument=diag
        cat=catalogues[instrument]
        rows=[r for r in cat if r['element']==element and r['ion']==ion
              and abs(r['rest']-rest)<.002 and r['origin']=='PHOT']
        if len(rows)!=1:raise ValueError(f'nonunique target catalogue entry {name}')
        # FUSE archive warns of uncorrected segment offsets. Use each target's
        # published measured velocity, never choose a shift to improve a model.
        # It does not remove all catalogue/coadd offsets (notably P V).
        velocity=rows[0]['velocity'] if instrument=='fuv' else 23.8
        catalogue_velocity=velocity
        offset=registration_offset(registration,rest) if instrument=='fuv' else 0.
        if offset:
            factor=doppler_factor(velocity)+offset/rest
            velocity=C_KMS*(factor**2-1)/(factor**2+1)
        resolution=20000. if instrument=='fuv' else 144000.
        center=rest*doppler_factor(velocity)
        w,y,e=observations[instrument];sel=abs(w-center)<1.2
        w,y,e=w[sel],y[sel],e[sel]
        anchors=(abs(w-center)>.4)&(abs(w-center)<1.15)
        blends=np.zeros(len(w),dtype=bool)
        mask_width=15. if instrument=='fuv' else 6.
        for row in cat:
            row_offset=registration_offset(registration,row['rest']) if instrument=='fuv' else 0.
            near=abs(w-row['observed']-row_offset)<row['observed']*mask_width/C_KMS
            anchors&=~near
            if not(row['element']==element and row['ion']==ion and row['origin']=='PHOT'):
                blends|=near
        continuum,used=fit_continuum(w,y,e,anchors)
        observed=y/continuum;error=e/continuum
        halfwidth=35. if instrument=='fuv' else 20.
        aperture=abs(w-center)<center*halfwidth/C_KMS
        score=aperture&~blends
        dw=np.diff(np.r_[w[0]-(w[1]-w[0])/2,(w[1:]+w[:-1])/2,w[-1]+(w[-1]-w[-2])/2])
        report[name]=dict(element=element,ion=ion,rest_wavelength=rest,instrument=instrument,
            velocity_kms=velocity,catalogue_velocity_kms=catalogue_velocity,
            registration_offset_angstrom=offset,
            velocity_source=('published catalogue plus independent local coadd registration' if offset else
                             'published line catalogue' if instrument=='fuv' else 'fixed photospheric 23.8 km/s'),
            gaussian_resolving_power=resolution,half_width_kms=halfwidth,retained_pixels=int(score.sum()),
            observed_aperture_ew_mA=float(np.sum((1-observed[aperture])*dw[aperture])*1000),
            observed_statistical_error_mA=float(np.linalg.norm(error[aperture]*dw[aperture])*1000),
            aperture_blend_pixels=int(np.count_nonzero(aperture&blends)),predictions={})
        arrays[name+'_wavelength']=w;arrays[name+'_observed']=observed;arrays[name+'_error']=error
        arrays[name+'_score_mask']=score;arrays[name+'_aperture']=aperture
        ax.plot(w-center,observed,color='.25',lw=.85,label='Observed')
        colors=['#7c90a0','#d59324','#c3473d','#389076','#7958a3','#3274a1']
        if len(spectra)==1:colors=['#c3473d']
        minimum_flux=float(np.min(observed[aperture]))
        labels={'CSi':'C + Si','Light':'+ N/O/Al/P/S (NLTE)','All':'+ Fe/Ni (LTE)'}
        for index,(label,(mw,mf)) in enumerate(spectra.items()):
            color=colors[index % len(colors)]
            sampled=instrument_sample(mw,mf,w,velocity=velocity,resolving_power=resolution)
            x=(w-np.mean(w))/np.ptp(w)
            design=sampled[:,None]*np.column_stack([np.ones(len(w)),x])
            coef=np.linalg.lstsq(design[used]/e[used,None],y[used]/e[used],rcond=None)[0]
            scale=coef[0]+coef[1]*x
            if np.any(scale<=0):raise ValueError('nonpositive continuum scale')
            normalized=sampled*scale/continuum
            report[name]['predictions'][label]=dict(
                normalized_rms=float(np.sqrt(np.mean((normalized[score]-observed[score])**2))),
                aperture_ew_mA=float(np.sum((1-normalized[aperture])*dw[aperture])*1000),
                continuum_coefficients=coef.tolist(),converged=metadata[label]['converged'])
            arrays[name+'_'+label]=normalized
            minimum_flux=min(minimum_flux,float(np.min(normalized[aperture])))
            ax.plot(w-center,normalized,color=color,lw=1.25,label=labels.get(label,label))
        ax.set(title=f'{element} {ion} {rest:.3f} Å ({instrument.upper()})',xlim=(-.32,.32),ylim=(max(0.,minimum_flux-.08),1.10))
        ax.axhline(1.,c='.75',lw=.5);ax.set_xlabel('Offset from adopted line center (Å)')
        ax.set_ylabel('Continuum-normalized flux')
    axes.flat[0].legend(fontsize=8)
    convergence_caption=('All trace-population solutions converged' if all(m['converged'] for m in metadata.values())
                         else 'Includes unconverged iterates; convergence status recorded per model')
    fig.suptitle('G191-B2B: published abundances, fixed 52,500 K / log g 7.53 host\n'
                 +convergence_caption+'; no abundance fitting'
                 +('\nP V 1128: independent local FUSE registration applied' if registration else ''),fontsize=13)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'additional-elements.png',dpi=160)
    fig.savefig(output/'additional-elements.pdf');plt.close(fig)
    np.savez(output/'additional-elements.npz',**arrays)
    result=dict(metrics=report,fuse_registration=registration,
        models={k:dict(directory=str(v),converged=metadata[k]['converged'],
        population_defect=metadata[k]['population_defect'],
        nlte_elements=metadata[k].get('nlte_elements',[]),
        lte_background_elements=metadata[k].get('lte_background_elements',[]),
        oxygen_op=metadata[k].get('oxygen_op',False),
        oxygen_collisions=metadata[k].get('oxygen_collisions'),
        abundance_choices=metadata[k].get('abundance_choices',{})) for k,v in models.items()},
        qualifications=['conditional local continuum fit on sidebands only',
            'aperture EWs include unmasked blends; RMS excludes catalogue contaminants',
            'Gaussian instrumental approximation; FUSE resolution assumed R=20000',
            ('P V 1128 registration measured from nearby non-P lines; other FUSE windows retain catalogue velocities'
             if registration else 'FUSE uses published line velocities; residual catalogue/coadd offsets remain, especially P V'),
            'no added ISM absorbers in these windows',
            'convergence flags refer to fixed-host trace populations, not metal-blanketed atmospheric equilibrium',
            'unconverged iterates, when included, are labelled in each model record'])
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,help='LABEL=DIRECTORY')
    p.add_argument('--data',default='results/hot-daz/multimetal-data')
    p.add_argument('--output',required=True)
    p.add_argument('--fuse-registration',type=Path)
    a=p.parse_args();comparison(dict(x.split('=',1) for x in a.model),a.data,a.output,
                               fuse_registration=a.fuse_registration)
