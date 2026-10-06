#!/usr/bin/env python3
"""Audit and plot the oxygen, iron-group, and nitrogen diagnostic experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from compare_hot_daz_benchmark import REGIONS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(models, output, verification, unregistered=None):
    output=Path(output)
    diagnostics=json.loads((output/'metrics.json').read_text())
    if list(models)!=list(diagnostics['models']):
        raise ValueError('comparison models differ from requested summary')
    cases={};arrays={};reference=None
    for label,directory in models.items():
        path=Path(directory)
        if str(path)!=diagnostics['models'][label]['directory']:
            raise ValueError('comparison directory changed')
        metadata=json.loads((path/'metadata.json').read_text())
        comparison=json.loads((path/'comparison/comparison.json').read_text())
        with fits.open(path/'comparison/prediction.fits',checksum=True) as hdus:
            for hdu in hdus:
                if hdu.verify_checksum()!=1 or hdu.verify_datasum()!=1:
                    raise ValueError('invalid FITS checksum')
            header=hdus[0].header
            assert header['METCONV']==metadata['converged']
            assert header['NLTEELEM']==','.join(metadata['nlte_elements'])
            assert header['LTEELEM']==','.join(metadata['lte_background_elements'])
            assert not header['VALIDATE'] and header['HOSTFIX']
            for element,abundance in metadata['abundances'].items():
                np.testing.assert_allclose(header[element.upper()+'_H'],10**abundance,rtol=1e-13,atol=0)
        with np.load(path/'spectrum.npz') as data:
            wave=data['wavelength'];background=data['background_flux']
            if np.any(~np.isfinite(data['flux'])) or np.any(data['flux']<=0):
                raise ValueError('nonpositive or nonfinite predicted flux')
            if reference is None:reference=(wave.copy(),background.copy())
            common,a,b=np.intersect1d(reference[0],wave,return_indices=True)
            if not len(common):raise ValueError('no shared host wavelengths')
            np.testing.assert_array_equal(reference[1][a],background[b])
        closure={}
        with np.load(path/'populations.npz') as data:
            for element in metadata['nlte_elements']:
                population=data[element+'_population_density']
                lte=data[element+'_lte_population_density']
                if np.any(~np.isfinite(population)) or np.any(population<0):
                    raise ValueError('invalid population')
                closure[element]=float(np.max(abs(population.sum(axis=0)/lte.sum(axis=0)-1)))
                if closure[element]>1e-12:raise ValueError('represented particles not conserved')
        snapshots=path/'source-snapshots'
        for name,expected in metadata.get('research_code_sha256',{}).items():
            if digest(snapshots/name)!=expected:raise ValueError('research source snapshot changed')
        with np.load(path/'comparison/comparison_arrays.npz') as data:
            arrays[label]={key:data[key].copy() for key in data.files}
        cases[label]=dict(directory=str(path),metadata=metadata,
            progress=json.loads((path/'progress.json').read_text()),
            original_diagnostic_metrics=comparison['metrics'],
            checks=dict(fits_checksums_and_metadata=True,finite_positive_flux=True,
                identical_host_flux_shared_wavelengths=len(common),
                maximum_represented_particle_closure=closure,source_snapshots_verified=True),
            file_sha256={name:digest(path/name) for name in
                ('spectrum.npz','populations.npz','metadata.json','comparison/prediction.fits')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#7c90a0','#d59324','#c3473d','#389076','#7958a3','#3274a1']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    base=next(iter(arrays.values()))
    for ax,(region,config) in zip(axes.flat,REGIONS.items()):
        wave=base[region+'_wavelength'];observed=base[region+'_observed']
        ax.plot(wave,observed,c='.25',lw=.75,label='Observed STIS')
        minimum=float(np.min(observed))
        for index,(label,data) in enumerate(arrays.items()):
            np.testing.assert_array_equal(data[region+'_wavelength'],wave)
            flux=data[region+'_model'];minimum=min(minimum,float(np.min(flux)))
            ax.plot(wave,flux,c=colors[index%len(colors)],lw=1.1,label=label)
        ax.set(title=config['label'],xlabel='Observed vacuum wavelength (Å)',
               ylabel='Normalized flux',ylim=(max(0.,minimum-.05),1.12))
        ax.axhline(1.,c='.8',lw=.5)
    axes.flat[0].legend(fontsize=8,ncol=2)
    fig.suptitle('G191-B2B: cross-checks on C III, Si IV and Ly α\n'
                 'Fixed H/He host; all metal solutions remain exploratory')
    fig.savefig(output/'carbon-silicon.png',dpi=160)
    fig.savefig(output/'carbon-silicon.pdf');plt.close(fig)
    registration_effect=None
    if unregistered is not None:
        before=json.loads((Path(unregistered)/'metrics.json').read_text())
        label=next(iter(models))
        if before['models'][label]['directory']!=str(models[label]):
            raise ValueError('registration comparison changed the intrinsic model')
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,directory,report,title in zip(axes,(Path(unregistered),output),
                (before,diagnostics),('Catalogue velocity','Independent local registration')):
            metric=report['metrics']['PV1128']
            with np.load(directory/'additional-elements.npz') as data:
                wave=data['PV1128_wavelength']
                ax.plot(wave,data['PV1128_observed'],color='.25',marker='.',lw=1,label='FUSE coadd')
                ax.plot(wave,data['PV1128_'+label],color='#c3473d',lw=1.4,label='Same control model')
            ax.set(xlim=(1127.9,1128.3),ylim=(.35,1.1),xlabel='Observed vacuum wavelength (Å)',
                   ylabel='Continuum-normalized flux',title=title)
            ax.set_xticks(np.arange(1127.9,1128.31,.1))
            ax.text(.04,.08,f'Diagnostic RMS: {100*metric["predictions"][label]["normalized_rms"]:.1f}%',
                    transform=ax.transAxes)
            ax.axhline(1.,c='.8',lw=.5)
        axes[0].legend(fontsize=9,loc='lower left',bbox_to_anchor=(0.,.20))
        fig.suptitle('P V 1128: wavelength registration removes most of the profile mismatch\n'
                     'Same phosphorus abundance and populations; correction derived from N IV / Si IV')
        fig.savefig(output/'pv1128-registration.png',dpi=160)
        fig.savefig(output/'pv1128-registration.pdf');plt.close(fig)
        registration_effect=dict(unregistered_directory=str(unregistered),
            unregistered_metrics_sha256=digest(Path(unregistered)/'metrics.json'),
            before=before['metrics']['PV1128'],after=diagnostics['metrics']['PV1128'],
            qualification='Each diagnostic aperture and blend mask follows its adopted line centre.')
    source_files=['research/summarize_hot_trace_repairs.py','research/compare_hot_composition.py',
                  'research/compare_hot_daz_benchmark.py','src/wd_spectra/hot_trace_metals.py']
    report=dict(cases=cases,additional_diagnostics=diagnostics,verification=verification,
        registration_effect=registration_effect,
        analysis_source_sha256={name:digest(name) for name in source_files},
        qualifications=[
            'Same converged H/He atmosphere at 52500 K, log g 7.53; metals do not update structure.',
            'All saved metal states remain unconverged; line stability is not a convergence certificate.',
            'Fe/Ni NLTE replaces only closed explicit transitions and continua; remaining opacity stays LTE.',
            'Nitrogen alternative uses a second published abundance, not a newly fitted value.',
            'Oxygen, iron, nickel and nitrogen changes are separate experiments, not a single combined model.',
            'Diagnostic-only spectra contain separated wavelength windows, not a continuous full UV prediction.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',action='append',required=True,help='LABEL=DIRECTORY')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--verification',required=True,type=Path)
    parser.add_argument('--history',type=Path)
    parser.add_argument('--unregistered-comparison',type=Path)
    args=parser.parse_args()
    report=summarize(dict(item.split('=',1) for item in args.model),args.output,
                     json.loads(args.verification.read_text()),args.unregistered_comparison)
    if args.history:args.history.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved and verified {len(report["cases"])} exploratory cases in {args.output}')
