#!/usr/bin/env python3
"""Short, warm-started physical sensitivity experiments for G191-B2B.

Intermediate spectra are saved with measured population defects. A promising
observed fit is not a convergence certificate. No abundance is optimized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra.constants import PLANCK, LIGHT_SPEED
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import solve_hot_trace_metals, fixed_electron_metal_reference
from wd_spectra.light_metal_nlte import (ReducedLightMetalLevelState,
    read_tlusty_photoionization_threshold_data,reduced_light_metal_wavelength)
from wd_spectra.metals import (read_pg1159_atomic_database,read_verner_photoionization_database,
    metal_bound_free_mass_absorption_coefficient,EV_TO_ERG)
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.spectrum import planck_lambda_angstrom
from resynthesize_hot_trace import load_frozen_host,identity
from compare_hot_daz_benchmark import compare


def load_guesses(directory, elements=None):
    result={}
    with np.load(Path(directory)/'populations.npz',allow_pickle=False) as p:
        available={k[:-len('_level_key')] for k in p.files if k.endswith('_level_key')}
        for e in sorted(available if elements is None else available.intersection(elements)):
            keys=tuple((str(k[0]),int(k[1]),int(k[2])) for k in p[e+'_level_key'])
            n=p[e+'_population_density'];lte=p[e+'_lte_population_density']
            dep={key:n[i]/np.maximum(lte[i],1e-300) for i,key in enumerate(keys)}
            result[e]=ReducedLightMetalLevelState(e,keys,n,lte,dep,0.,{'warm_guess_only':True})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('control','large-c','op','large-c-op','fe-continuum'),required=True)
    p.add_argument('--seed',type=Path,default=Path('results/hot-daz/g191-b2b-higher-ions'))
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--carbon-levels',type=int,nargs=4,metavar=('CIII','CIV','CV','CVI'),
                   help='Override the four carbon stage sizes to screen atom completeness')
    p.add_argument('--chianti-carbon',type=Path,help='Directory with pinned CHIANTI elvlc/scups files and manifest')
    p.add_argument('--collision-charges',type=int,nargs='+',default=[2,3])
    p.add_argument('--collision-low-levels',type=int,help='Use only collision pairs within the first N Stout indices')
    p.add_argument('--refine-grid',action='store_true',help='Add geometric midpoints to the metal population grid')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('choose a new output directory; preserve earlier experiments')
    data=ModelData.default();ident=identity(data)
    seedmeta=json.loads((args.seed/'metadata.json').read_text())
    host=load_frozen_host(Path(seedmeta['host_directory']),data,allow_trace_only_changes=True)
    counts={e:{int(k):v for k,v in n.items()} for e,n in seedmeta['levels_per_charge'].items()}
    if args.case in ('large-c','large-c-op'):counts['C']={2:80,3:100,4:10,5:1}
    if args.carbon_levels is not None:counts['C']=dict(zip((2,3,4,5),args.carbon_levels))
    db=read_pg1159_atomic_database(data.stout,elements=('C','Si'))
    collisions=None;collision_audit=None
    if args.chianti_carbon:
        from hot_trace_collisions import carbon_collisions
        collisions,collision_audit=carbon_collisions(args.chianti_carbon,db,counts['C'],
            charges=args.collision_charges,low_level_limit=args.collision_low_levels)
    thresholds={}
    if args.case in ('op','large-c-op'):
        thresholds['C']={2:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c3.dat'),
                         3:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c4_35+2lev.dat')}
    wave=host.spectrum.wavelength_angstrom
    model=_model_from_config(host.config,data)
    def host_background(w):return model.transfer_coefficients(host.atmosphere,w,host.population_state)
    background=host_background
    bg_flux=transfer_field(host.atmosphere,host_background(wave),n_angle=3)[1].interface_flux[:,0]
    extra_abundances={}
    extra_grid=[]
    if args.case=='fe-continuum':
        # Preval 2013 Table 10: elemental number ratio inferred from Fe V.
        # The local data lack the Ni IV+ ionization ladder and Ni Verner fits;
        # do not silently invent a nickel opacity or call this full blanketing.
        extra_abundances={'Fe':np.log10(5.00e-6)}
        extra_db=read_pg1159_atomic_database(data.stout,elements=tuple(extra_abundances))
        extra_ref=fixed_electron_metal_reference(host.atmosphere,extra_db,extra_abundances)
        extra_photo=read_verner_photoionization_database(data.verner_photoionization,
            elements=tuple(extra_abundances),require_all_elements=True)
        for e in extra_abundances:
            for ion in extra_db.ion_stages(e):
                if ion.ionization_energy_ev and 2<=ion.charge<=7:
                    edge=PLANCK*LIGHT_SPEED/EV_TO_ERG/ion.ionization_energy_ev*1e8
                    extra_grid.extend(edge*np.array([.99999,1.,1.00001]))
        def background(w):
            base=host_background(w)
            a=metal_bound_free_mass_absorption_coefficient(host.atmosphere,w,extra_db,extra_ref,extra_photo)
            return NLTETransferCoefficients(w,base.true_absorption+a,
                base.thermal_emissivity+a*planck_lambda_angstrom(w[:,None],host.atmosphere.temperature),
                base.scattering,{'extra_opacity':'LTE Fe ground bound-free only; no line forest or thermal feedback'})
    grid=np.unique(np.concatenate([np.asarray(extra_grid)]+[reduced_light_metal_wavelength(
        db,e,n,photoionization_threshold_data=thresholds.get(e)) for e,n in counts.items()]))
    if args.refine_grid:grid=np.unique(np.r_[grid,np.sqrt(grid[:-1]*grid[1:])])
    common=dict(experimental=True,exploratory=True,smoke_only=False,case=args.case,
        host_parameters=seedmeta['host_parameters'],abundances=seedmeta['abundances'],
        extra_background_abundances=extra_abundances,levels_per_charge=counts,
        host_directory=seedmeta['host_directory'],host_atmosphere_convergence='converged',
        host_reuse_scope=host.metadata['frozen_host_reuse_scope'],
        atmosphere_recomputed=False,host_populations_updated=False,electron_density_updated=False,
        observationally_validated=False,tabulated_photoionization_stages={e:list(t) for e,t in thresholds.items()},
        population_wavelength_points=len(grid),numerical_identity=ident,
        population_grid_midpoints_added=args.refine_grid,
        seed_population_sha256=hashlib.sha256((args.seed/'populations.npz').read_bytes()).hexdigest(),
        limitations=['fixed H/He structure and populations','no complete Fe/Ni line blanketing',
                     'provisional atoms and collisions','intermediate iterates are exploratory'])
    if extra_abundances:common['background_experiment']='LTE Fe bound-free continuum only'
    args.output.mkdir(parents=True)
    if collision_audit is not None:
        (args.output/'collision-audit.json').write_text(json.dumps(collision_audit,indent=2)+'\n')
        common['collision_audit_sha256']=hashlib.sha256((args.output/'collision-audit.json').read_bytes()).hexdigest()
        common['collision_adapter_sha256']=hashlib.sha256(Path(__file__).with_name('hot_trace_collisions.py').read_bytes()).hexdigest()
        common['collision_data_records']=len(collisions)
    history=[]
    def save(directory,spectrum,states,metadata):
        directory.mkdir(parents=True,exist_ok=True)
        np.savez(directory/'spectrum.npz',wavelength=wave,flux=spectrum.surface_flux_lambda,background_flux=bg_flux)
        np.savez(directory/'populations.npz',**{f'{e}_{key}':getattr(s,key)
            for e,s in states.items() for key in ('level_key','population_density','lte_population_density')})
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        report=compare(directory,'results/hot-daz/g191-b2b',directory/'comparison',allow_unconverged=True)
        return report['metrics']
    def callback(i,defect,states,synthesize):
        print(f'{args.case}: iteration {i}, defect {defect:.5g}',flush=True)
        if i in {1,4,8,args.iterations}:
            spectrum=synthesize()
            meta={**common,**spectrum.metadata,'converged':defect<1e-4,'iterations':i,'population_defect':defect}
            metrics=save(args.output/f'iteration-{i:03d}',spectrum,states,meta)
            widths={k:v['predicted_aperture_ew_mA'] for k,v in metrics['ciii_1175']['isolated_components'].items()}
            history.append(dict(iteration=i,defect=defect,ciii_equivalent_widths_mA=widths,
                                normalized_rms={k:v['normalized_rms'] for k,v in metrics.items()}))
            (args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print(f'  C III EWs: {widths}',flush=True)
        return False
    result=solve_hot_trace_metals(host.atmosphere,background,seedmeta['abundances'],wave,data=data,
        atomic_database=db,levels_per_charge=counts,population_wavelength=grid,
        initial_populations=load_guesses(args.seed,elements=counts),photoionization_threshold_data=thresholds,
        collision_data=collisions,
        maximum_iterations=args.iterations,tolerance=1e-4,require_convergence=False,state_callback=callback)
    if identity(data)!=ident:raise RuntimeError('code/data changed during experiment')
    metadata={**result.metadata,**common,'converged':result.converged,'iterations':result.iterations,
              'population_defect':result.population_defect}
    metrics=save(args.output,result.spectrum,result.populations,metadata)
    print(json.dumps({'case':args.case,'converged':result.converged,
        'population_defect':result.population_defect,'ciii':metrics['ciii_1175']['isolated_components']},indent=2))


if __name__=='__main__':main()
