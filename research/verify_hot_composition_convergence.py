#!/usr/bin/env python3
"""Audit a converged trace solution and an independently re-evaluated restart."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from compare_hot_daz_benchmark import compare
from wd_spectra.models.common import ModelData
from resynthesize_hot_trace import load_frozen_host


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_record(metadata):
    """Do not infer convergence from a small damped step or a saved Boolean."""
    tolerance=metadata.get('requested_population_tolerance')
    defect=metadata.get('population_defect')
    if (not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance)
            or not 0<tolerance<=1e-4 or not isinstance(defect,(int,float))
            or not np.isfinite(defect) or not 0<=defect<tolerance):
        raise ValueError('undamped population residual does not satisfy the requested tolerance')
    if not metadata.get('converged') or metadata.get('smoke_only'):
        raise ValueError('expected a converged physical trace solution')
    if metadata.get('convergence_scope')!='fixed-host metal populations':
        raise ValueError('unrecognized convergence scope')
    element=metadata.get('element_population_defects',{})
    if set(element)!=set(metadata['nlte_elements']) or not all(np.isfinite(x) and 0<=x<tolerance for x in element.values()):
        raise ValueError('missing or failing element residual')
    if max(element.values())!=defect:raise ValueError('inconsistent worst element residual')
    for key,limit in [('source_closure_residual',1e-6),('maximum_particle_conservation_error',1e-12)]:
        value=metadata.get(key)
        if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<limit:
            raise ValueError(f'failed convergence audit: {key}')


def continuation_history(directory):
    """Retain the warm-start work; do not present four final maps as the whole solve."""
    stages=[];seen=set()
    while True:
        directory=Path(directory)
        if str(directory) in seen:raise ValueError('cyclic seed provenance')
        seen.add(str(directory))
        metadata=json.loads((directory/'metadata.json').read_text())
        if 'requested_population_tolerance' not in metadata:break
        end=metadata['iterations']
        root=directory.parent if directory.name.startswith('iteration-') else directory
        root_metadata=root/'metadata.json'
        complete=json.loads(root_metadata.read_text()) if root_metadata.exists() else metadata
        history=complete.get('element_population_defect_history')
        if history is not None:
            points=[(i+1,row) for i,row in enumerate(history[:end])]
        else:
            points=[(row['iteration'],row['element_defects'])
                    for row in json.loads((root/'progress.json').read_text()) if row['iteration']<=end]
        if not points or points[-1][0]!=end:raise ValueError('incomplete continuation history')
        stages.append(dict(directory=str(directory),evaluations=end,points=points,
            method=('ion-stage Anderson' if metadata.get('ionwise_acceleration') else
                    'element Anderson' if metadata.get('elementwise_acceleration') else 'joint Anderson'),
            history_depth=metadata['requested_acceleration_depth'],metadata_sha256=sha256(directory/'metadata.json')))
        parent=Path(metadata['seed_directory'])
        prior=json.loads((parent/'metadata.json').read_text())
        if any(metadata[k]!=prior[k] for k in ('abundances','levels_per_charge','oxygen_op','oxygen_collisions')):break
        directory=parent
    return list(reversed(stages))


def verify(model,replay,previous,output):
    model,replay,previous,output=map(Path,(model,replay,previous,output))
    records=[json.loads((directory/'metadata.json').read_text()) for directory in (model,replay)]
    for directory,metadata in zip((model,replay),records):
        check_record(metadata)
        for name,expected in metadata['research_code_sha256'].items():
            if sha256(directory/'source-snapshots'/name)!=expected:
                raise ValueError('source snapshot mismatch')
    original,fresh=records
    settings=('abundances','levels_per_charge','nlte_elements','lte_background_elements',
              'oxygen_op','oxygen_collisions','nlte_iron_group','numerical_identity',
              'host_parameters','host_directory','population_wavelength_points')
    if any(original[key]!=fresh[key] for key in settings):
        raise ValueError('restart changed the physical or numerical problem')
    if fresh['iterations']!=1 or fresh['seed_population_sha256']!=sha256(model/'populations.npz'):
        raise ValueError('expected a fresh, single-evaluation restart of this model')
    # This rebuilds the saved EOS and rechecks the host certificate/code/data.
    host=load_frozen_host(Path(original['host_directory']),ModelData.default(),allow_trace_only_changes=True)
    host_certificate=host.atmosphere.metadata['equilibrium_certificate']
    output.mkdir(parents=True,exist_ok=True)
    strict=compare(replay,'results/hot-daz/g191-b2b',output/'strict-comparison')
    with np.load(model/'spectrum.npz') as a,np.load(replay/'spectrum.npz') as b:
        shared,ia,ib=np.intersect1d(a['wavelength'],b['wavelength'],return_indices=True)
        if len(shared)!=len(a['wavelength']):raise ValueError('replay does not cover the original spectrum')
        np.testing.assert_array_equal(a['background_flux'][ia],b['background_flux'][ib])
        wave=b['wavelength'].copy();flux=b['flux'].copy()
        if np.any(~np.isfinite(flux)) or np.any(flux<=0):raise ValueError('invalid surface flux')
        chosen=wave[(wave>=910.)&(wave<=1990.001)]
        if chosen[0]>910.001 or chosen[-1]<1989.999 or np.max(np.diff(chosen))>.01001:
            raise ValueError('full 910--1990 A spectrum is missing or has gaps')
        spectrum_difference=float(np.max(abs(b['flux'][ib]/a['flux'][ia]-1)))
        if spectrum_difference>1e-8:raise ValueError('fresh restart changed converged spectrum unexpectedly')
    with np.load(model/'populations.npz') as a,np.load(replay/'populations.npz') as b:
        population_differences={}
        for element in original['nlte_elements']:
            np.testing.assert_array_equal(a[element+'_level_key'],b[element+'_level_key'])
            population=a[element+'_population_density'];other=b[element+'_population_density']
            floor=1e-12*a[element+'_lte_population_density'].sum(axis=0)
            difference=float(np.max(abs(other-population)/np.maximum(population,floor)))
            if difference>1e-8:raise ValueError('fresh restart changed saved populations unexpectedly')
            population_differences[element]=difference
    with fits.open(output/'strict-comparison/prediction.fits',checksum=True) as hdus:
        for hdu in hdus:
            if hdu.verify_checksum()!=1 or hdu.verify_datasum()!=1:raise ValueError('FITS checksum failure')
        header=hdus[0].header
        if (not header['METCONV'] or header['HOSTCONV']!='converged' or not header['HOSTFIX']
                or header['VALIDATE'] or header['MTHERM'] or header['MCHARGE']):
            raise ValueError('incorrect FITS convergence/validation flags')
    prior=json.loads((previous/'comparison/comparison.json').read_text())
    stages=continuation_history(model)
    report=dict(scope='fixed-host trace-metal statistical equilibrium and radiative transfer',
        verified=True,model_directory=str(model),replay_directory=str(replay),previous_directory=str(previous),
        full_spectrum_directory=str(replay),
        convergence_tolerance=original['requested_population_tolerance'],
        undamped_population_defect=original['population_defect'],
        independently_recomputed_defect=fresh['population_defect'],
        element_population_defects=original['element_population_defects'],
        source_closure_residual=original['source_closure_residual'],
        maximum_particle_conservation_error=original['maximum_particle_conservation_error'],
        restart_maximum_relative_flux_difference=spectrum_difference,
        restart_shared_wavelength_points=len(shared),
        restart_population_differences=population_differences,
        host_equilibrium_certificate=host_certificate,
        full_uv_coverage_angstrom=[float(chosen[0]),float(chosen[-1])],
        wavelength_points=len(wave),abundance_choices=original['abundance_choices'],
        nlte_elements=original['nlte_elements'],lte_background_elements=original['lte_background_elements'],
        final_metrics=strict['metrics'],previous_metrics=prior['metrics'],
        spectrum_sha256=sha256(replay/'spectrum.npz'),populations_sha256=sha256(replay/'populations.npz'),
        converged_seed_spectrum_sha256=sha256(model/'spectrum.npz'),
        replay_metadata_sha256=sha256(replay/'metadata.json'),
        continuation_stages=stages,
        audit_script_sha256=sha256(__file__),
        qualifications=['Fe/Ni opacity remains LTE; seven other metals are solved in NLTE.',
            'The converged H/He structure is fixed; metal thermal and charge feedback is not solved.',
            'Fresh restart recomputes the same equations, not an independent atmosphere code.',
            'Numerical convergence does not validate incomplete atoms or establish an observed abundance fit.'])
    energy_path=replay/'energy-audit/energy-balance.json'
    if energy_path.exists():
        energy=json.loads(energy_path.read_text())
        if energy.get('atmosphere_recomputed') or energy.get('thermal_equilibrium_with_metals_certified'):
            raise ValueError('a frozen-state energy diagnostic cannot certify thermal equilibrium')
        report['frozen_state_energy_audit']=energy
        report['energy_audit_sha256']=sha256(energy_path)
        report['energy_audit_directory']=str(energy_path.parent)
    (output/'convergence-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    histories=[];iteration_numbers=[];offset=0
    for stage in stages:
        iteration_numbers.extend(offset+i for i,row in stage['points'])
        histories.extend(row for i,row in stage['points'])
        if offset:axes[0].axvline(offset+.5,c='.8',lw=.7)
        offset+=stage['evaluations']
    for element in original['nlte_elements']:
        axes[0].semilogy(iteration_numbers,[x[element] for x in histories],label=element)
    axes[0].axhline(original['requested_population_tolerance'],c='.2',ls='--',label='Required tolerance')
    axes[0].set(xlabel='Map evaluation along the continuation chain',ylabel='Maximum undamped population residual',
                title='Every solved element passes (saved checkpoints shown)')
    axes[0].legend(ncol=4,fontsize=8)
    old=np.load(previous/'comparison/comparison_arrays.npz')
    final=np.load(output/'strict-comparison/comparison_arrays.npz')
    name='ciii_1175';w=final[name+'_wavelength']
    axes[1].plot(w,final[name+'_observed'],c='.3',lw=.75,label='Observed STIS')
    axes[1].plot(w,old[name+'_model'],c='#d59324',lw=1,label='Previous 8-iteration screen')
    axes[1].plot(w,final[name+'_model'],c='#c3473d',lw=1.15,label='Converged trace solution')
    axes[1].set(xlabel='Observed vacuum wavelength (Å)',ylabel='Normalized flux',title='C III 1175 multiplet')
    axes[1].legend(fontsize=8);old.close();final.close()
    fig.suptitle('G191-B2B: numerical convergence on the fixed H/He atmosphere')
    fig.savefig(output/'convergence.png',dpi=160);fig.savefig(output/'convergence.pdf');plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','replay','previous','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--history',type=Path)
    args=parser.parse_args();report=verify(args.model,args.replay,args.previous,args.output)
    if args.history:args.history.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('verified','undamped_population_defect','independently_recomputed_defect')}))
