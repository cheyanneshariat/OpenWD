#!/usr/bin/env python3
"""Re-solve full H/He NLTE from a tagged G191-B2B numerical checkpoint.

This is a research warm start, not the public cold-start API. No convergence
certificate is inherited: the final full-NLTE solve must independently pass
its population, flux, source-closure and temperature-correction checks.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import time
import numpy as np
from wd_spectra import DAOConfig, ModelResult, save_model_result
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import population_arrays, with_departures, transfer_field
from wd_spectra.hot_trace_metals import synthesize_dao_trace_metals
from wd_spectra.models.common import ModelData, numerical_resolution, warn_if_atmosphere_not_converged
from wd_spectra.models.hot import _model_from_config, structure_wavelength
from wd_spectra.spectrum import Spectrum
from wd_spectra._hot_structure import solve
from hot_daz_benchmark import BENCHMARK
from run_hot_public_diagnostic import save_state


def load_seed(path, config, data):
    """Restore arrays under the explicitly declared H/He composition.

    The numerical checkpoint format records atom sizes and Teff/logg, but not
    the H/He ratio. The caller must provide the original configuration. The
    independent LTE-reference check also rejects inconsistent compositions.
    """
    model = _model_from_config(config, data)
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in [('effective_temperature',config.effective_temperature),
                              ('logg',config.logg),
                              ('maximum_hydrogen_level',config.maximum_hydrogen_level),
                              ('maximum_helium_ii_level',config.maximum_helium_ii_level)]:
            if float(saved[key]) != expected:
                raise ValueError(f'checkpoint {key} differs from configuration')
        seed = gray_hydrogen_helium_atmosphere(config.effective_temperature, config.logg,
                    config.log_hydrogen_to_helium, n_depth=len(saved['temperature']))
        seed = replace(seed, column_mass=saved['column_mass'], gas_pressure=saved['gas_pressure'],
                       rosseland_optical_depth=saved['rosseland_optical_depth'])
        seed = model.rebuild_atmosphere(seed, saved['temperature'])
        reference = model._rate_state(seed)
        _, expected = population_arrays(reference)
        np.testing.assert_allclose(saved['lte_population'],expected,rtol=1e-10,atol=0,
                                   err_msg='checkpoint EOS/composition/atom reference mismatch')
        populations = with_departures(reference,saved['population']/saved['lte_population'])
        actual,_ = population_arrays(populations)
        np.testing.assert_allclose(actual,saved['population'],rtol=1e-10,atol=0)
    return model,seed,populations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-iterations',type=int,default=120)
    parser.add_argument('--equilibrate-populations',action='store_true',
                        help='prepare fixed-temperature NLTE populations before the coupled solve')
    args=parser.parse_args()
    if (args.output/'host').exists() or (args.output/'metadata.json').exists():
        raise FileExistsError('choose a new output directory to preserve existing model provenance')
    args.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    config=DAOConfig(effective_temperature=BENCHMARK['effective_temperature'],
                     logg=BENCHMARK['logg'],log_hydrogen_to_helium=5.)
    data=ModelData.default()
    from resynthesize_hot_trace import record_identity, identity
    initial_identity=record_identity(args.output,data)
    model,seed,populations=load_seed(args.checkpoint,config,data)
    manifest=dict(cold_start=False,checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  fixed_temperature_population_initialization=args.equilibrate_populations,
                  method='full NLTE from a numerical checkpoint; independent final certificate required')
    (args.output/'initialization.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.monotonic()
    if args.equilibrate_populations:
        populations=model.solve_populations(seed,populations)
        save_state(args.output/'host-population-initialization.npz',seed,populations,model)
        print(f'Fixed-temperature population initializer: converged={populations.converged}, '
              f'defect={populations.maximum_relative_population_change:.6g}',flush=True)
    def progress(i,a,p,d):
        save_state(args.output/'host-accepted.npz',a,p,model)
        (args.output/'host-progress.json').write_text(json.dumps(dict(
            elapsed_seconds=time.monotonic()-start,iteration=i,**d),indent=2)+'\n')
    def jacobian(x,e):
        np.savez_compressed(args.output/'host-jacobian.npz',x=x,residual=e.residual,jacobian=e.jacobian)
        save_state(args.output/'host-jacobian-state.npz',e.payload[0],e.payload[1],model)
    resolution=numerical_resolution(config.quality)
    answer=solve(seed,model,structure_wavelength(config,resolution.n_continuum),
                 populations=populations,nlte_fraction=1.,maximum_iterations=args.maximum_iterations,
                 iteration_callback=progress,jacobian_callback=jacobian)
    a,p=answer.atmosphere,answer.population_state
    if identity(data) != initial_identity:
        raise RuntimeError('numerical code or physical tables changed during the host solve')
    save_state(args.output/'host-final.npz',a,p,model)
    status=warn_if_atmosphere_not_converged(a,'DAO')
    wave=np.unique(np.concatenate((np.arange(1173.,1178.,.002),np.arange(1187.,1245.,.02),
        np.arange(1214.5,1217.,.002),np.arange(1392.,1395.5,.002),np.arange(1401.,1404.5,.002))))
    _,field,closure=transfer_field(a,model.transfer_coefficients(a,wave,p),n_angle=model.n_angle)
    host=ModelResult('DAO',a,Spectrum(wave,field.interface_flux[:,0],dict(source_closure_residual=closure)),
                     config,dict(atmosphere_convergence_status=status,nlte_populations_converged=p.converged,
                                 initialization=manifest),p)
    save_model_result(host,args.output/'host')
    abundance={'C':np.log10(BENCHMARK['carbon_from_ciii']['value']),
               'Si':np.log10(BENCHMARK['silicon_from_siiv']['value'])}
    result=synthesize_dao_trace_metals(host,abundance,wave,data=data,maximum_iterations=160,
                 iteration_callback=lambda i,d:print(f'metal iteration {i}: {d:.6g}',flush=True))
    np.savez(args.output/'spectrum.npz',wavelength=wave,flux=result.spectrum.surface_flux_lambda,
             background_flux=result.background_spectrum.surface_flux_lambda)
    np.savez(args.output/'populations.npz',**{
        f'{e}_{field}':getattr(state,field) for e,state in result.populations.items()
        for field in ('population_density','lte_population_density')})
    metadata={**result.metadata,'smoke_only':False,'abundances':abundance,
              'host_parameters':dict(teff=config.effective_temperature,logg=config.logg,log_h_he=5.),
              'converged':result.converged,'iterations':result.iterations,
              'population_defect':result.population_defect,'initialization':manifest,
              'elapsed_seconds':time.monotonic()-start,'comparison_to_observation_completed':False}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(converged=result.converged,iterations=result.iterations,
                         population_defect=result.population_defect),indent=2))

if __name__=='__main__':main()
