#!/usr/bin/env python3
"""Repeat trace-metal synthesis on a saved, unchanged, converged H/He host.

This research reader requires a code/data identity recorded during the host
run. Its explicit metal-only exception verifies unchanged H/He code, tables,
and rebuilt EOS against the original manifest; strict identity is the default.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from wd_spectra import DAOConfig, ModelResult
from wd_spectra._provenance import numerical_code_identity, model_data_identity
from wd_spectra.helium_nlte import CoupledHeliumNLTEState
from wd_spectra.hot_nlte import HotPopulationState
from wd_spectra.hot_trace_metals import synthesize_dao_trace_metals
from wd_spectra.models.common import ModelData, load_atmosphere_checkpoint, atmosphere_convergence_status
from wd_spectra.multilevel_nlte import MultilevelHydrogenNLTEState
from wd_spectra.spectrum import Spectrum


def identity(data):
    return dict(numerical_code=numerical_code_identity(),physical_data=model_data_identity(data))


def record_identity(directory,data):
    record=identity(data)
    (Path(directory)/'numerical-identity.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def verify_frozen_identity(directory,data,allow_trace_only_changes=False):
    directory=Path(directory)
    previous=json.loads((directory/'numerical-identity.json').read_text())
    current=identity(data)
    if previous==current:return 'identical code and data'
    manifest_path=directory/'validated-code-manifest.json'
    if allow_trace_only_changes and previous['physical_data']==current['physical_data'] and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        package=Path(__file__).resolve().parents[1]/'src/wd_spectra'
        files=list(package.rglob('*.py'))+list(package.glob('_rt*.so'))+list(package.glob('_rt*.pyd'))
        now={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        old=manifest['files']
        # Neither module participates in the pure H/He DAO host equations.
        # They are used only after reconstructing the independently certified host.
        trace_modules={'hot_trace_metals.py','light_metal_nlte.py'}
        if (manifest['identity']==previous and set(old)==set(now)
                and all(old[k]==now[k] for k in old if k not in trace_modules)):
            changed=', '.join(sorted(k for k in old if old[k]!=now[k]))
            return f'verified identical H/He code and data; only {changed} changed'
    raise ValueError('host code/data changed; a new equilibrium solve is required')


def load_frozen_host(directory,data,*,allow_trace_only_changes=False):
    directory=Path(directory)
    reuse_scope=verify_frozen_identity(directory,data,allow_trace_only_changes)
    folder=directory/'host'
    metadata=json.loads((folder/'metadata.json').read_text())
    if metadata['spectral_type']!='DAO':raise ValueError('expected a DAO host')
    config=DAOConfig(**metadata['config'])
    atmosphere=load_atmosphere_checkpoint(folder/'atmosphere.npz',config.effective_temperature,
        config.logg,'mixed',log_hydrogen_to_helium=config.log_hydrogen_to_helium)
    if atmosphere_convergence_status(atmosphere)!='converged':
        raise ValueError('saved host is not converged or its rebuilt EOS changed')
    with np.load(folder/'populations.npz',allow_pickle=False) as saved:
        def decode(value):
            if isinstance(value,dict):
                if set(value)=={'array','shape'}:
                    array=saved[value['array']].copy()
                    if list(array.shape)!=value['shape']:raise ValueError('population shape mismatch')
                    return array
                return {key:decode(item) for key,item in value.items()}
            if isinstance(value,list):return [decode(item) for item in value]
            return value
        state=decode(json.loads(str(saved['metadata_json'])))
    state['helium']=CoupledHeliumNLTEState(**state['helium'])
    state['hydrogen']=MultilevelHydrogenNLTEState(**state['hydrogen'])
    populations=HotPopulationState(**state)
    metadata['model_metadata']['frozen_host_reuse_scope']=reuse_scope
    spectrum=np.loadtxt(folder/'spectrum.txt')
    return ModelResult('DAO',atmosphere,Spectrum(spectrum[:,0],spectrum[:,1],metadata['spectrum_metadata']),
                       config,metadata['model_metadata'],populations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expanded-atoms',action='store_true',
                        help='C III/IV/V=(40,54,1), Si III/IV/V=(50,40,1) level-sensitivity check')
    parser.add_argument('--higher-ions',action='store_true',
                        help='C III/IV/V/VI=(40,54,10,1), Si III/IV/V/VI=(50,40,30,1); overrides expanded-atoms')
    parser.add_argument('--refine-radiation-grid',action='store_true',
                        help='include the host rate/structure grid and double sampling between all points')
    parser.add_argument('--full-uv',action='store_true',
                        help='also synthesize 1150--1700 A every 0.01 A, retaining fine diagnostic sampling')
    parser.add_argument('--tolerance',type=float,default=1e-3)
    parser.add_argument('--allow-trace-only-changes',action='store_true',
                        help='reuse host only if its recorded manifest proves H/He code and physical tables unchanged')
    args=parser.parse_args()
    data=ModelData.default();host=load_frozen_host(args.host,data,
        allow_trace_only_changes=args.allow_trace_only_changes)
    original=json.loads((args.host/'metadata.json').read_text())
    options=dict(maximum_iterations=200,tolerance=args.tolerance,
                 iteration_callback=lambda i,d:print(f'metal iteration {i}: {d:.6g}',flush=True))
    if args.expanded_atoms:options['levels_per_charge']={'C':{2:40,3:54,4:1},'Si':{2:50,3:40,4:1}}
    if args.higher_ions:options['levels_per_charge']={'C':{2:40,3:54,4:10,5:1},'Si':{2:50,3:40,4:30,5:1}}
    if args.refine_radiation_grid:
        from wd_spectra._hot_structure import HotEquations
        from wd_spectra.models.hot import _model_from_config, structure_wavelength
        from wd_spectra.models.common import numerical_resolution
        from wd_spectra.metals import read_pg1159_atomic_database
        from wd_spectra.light_metal_nlte import reduced_light_metal_wavelength
        model=_model_from_config(host.config,data)
        grid=HotEquations(host.atmosphere,model,structure_wavelength(
            host.config,numerical_resolution(host.config.quality).n_continuum)).wave
        counts=options.get('levels_per_charge',{'C':{2:20,3:30,4:1},'Si':{2:30,3:23,4:1}})
        database=read_pg1159_atomic_database(data.stout,elements=tuple(counts))
        grid=np.unique(np.concatenate([grid]+[reduced_light_metal_wavelength(
            database,e,count,n_continuum_wavelength=640) for e,count in counts.items()]))
        options['population_wavelength']=np.unique(np.r_[grid,np.sqrt(grid[:-1]*grid[1:])])
    wave=host.spectrum.wavelength_angstrom
    if args.full_uv:wave=np.unique(np.r_[wave,np.arange(1150.,1700.005,.01)])
    result=synthesize_dao_trace_metals(host,original['abundances'],wave,
                                     data=data,**options)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'spectrum.npz',wavelength=result.spectrum.wavelength_angstrom,
             flux=result.spectrum.surface_flux_lambda,background_flux=result.background_spectrum.surface_flux_lambda)
    np.savez(args.output/'populations.npz',**{
        f'{e}_{field}':getattr(state,field) for e,state in result.populations.items()
        for field in ('population_density','lte_population_density','level_key')})
    metadata={**result.metadata,'smoke_only':False,'abundances':original['abundances'],
              'host_parameters':original['host_parameters'],'host_directory':str(args.host.resolve()),
              'converged':result.converged,'iterations':result.iterations,
              'population_defect':result.population_defect,'comparison_to_observation_completed':False}
    metadata['radiation_grid_refined_with_host']=args.refine_radiation_grid
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(converged=result.converged,iterations=result.iterations,
                         population_defect=result.population_defect),indent=2))

if __name__=='__main__':main()
