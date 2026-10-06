#!/usr/bin/env python3
"""Experimental G191-B2B C/Si line formation; --smoke is not a stellar model.

The ordinary command computes and saves a fresh converged DAO background.
--smoke instead uses a small gray atmosphere with Planck-initialized H/He
populations solely to exercise the numerical plumbing without an hours-long
host solve. Its output must never be scored as a physical benchmark fit.
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np
from wd_spectra import DAOConfig, compute_dao, save_model_result
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_trace_metals import solve_hot_trace_metals, synthesize_dao_trace_metals
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
from hot_daz_benchmark import BENCHMARK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/hot-daz/g191-b2b-model'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--maximum-iterations', type=int, default=80)
    args = parser.parse_args()
    if (args.output/'host').exists() or (args.output/'metadata.json').exists():
        raise FileExistsError('choose a new output directory to preserve existing model provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    data = ModelData.default()
    from resynthesize_hot_trace import record_identity, identity
    initial_identity = record_identity(args.output,data)
    wave = np.unique(np.concatenate((np.arange(1173.0,1178.0,.002),
        np.arange(1187.,1245.,.02),np.arange(1214.5,1217.,.002),
        np.arange(1392.0,1395.5,.002),np.arange(1401.0,1404.5,.002))))
    abundance = {'C': np.log10(BENCHMARK['carbon_from_ciii']['value']),
                 'Si': np.log10(BENCHMARK['silicon_from_siiv']['value'])}
    config = DAOConfig(effective_temperature=BENCHMARK['effective_temperature'],
                       logg=BENCHMARK['logg'], log_hydrogen_to_helium=5.,
                       maximum_helium_ii_level=4 if args.smoke else 32,
                       maximum_hydrogen_level=4 if args.smoke else 8)
    options = dict(maximum_iterations=args.maximum_iterations,
                   require_convergence=not args.smoke,
                   iteration_callback=lambda i,d: print(f'metal iteration {i}: undamped defect {d:.5g}', flush=True))
    if args.smoke:
        atmosphere = gray_hydrogen_helium_atmosphere(
            config.effective_temperature, config.logg, config.log_hydrogen_to_helium,
            n_depth=8, tau_max=100.)
        model = _model_from_config(config, data)
        populations = model._rate_state(atmosphere)
        result = solve_hot_trace_metals(atmosphere,
            lambda w: model.transfer_coefficients(atmosphere, w, populations),
            abundance, wave, data=data, n_angle=2, **options)
    else:
        # Instrumentation only: retain accepted states without changing the solve.
        import wd_spectra._hot_structure as joint
        from run_hot_public_diagnostic import save_state
        original_solve = joint.solve
        started = time.monotonic()
        def instrumented(seed, model, grid, **kwargs):
            fraction = kwargs.get('nlte_fraction', 1.)
            callback = kwargs.get('iteration_callback')
            def accepted(i, a, populations, diagnostics):
                save_state(args.output/f'host-stage-{fraction:.3f}-accepted.npz', a, populations, model)
                progress = dict(elapsed_seconds=time.monotonic()-started, stage=fraction,
                                iteration=i, **diagnostics)
                (args.output/'host-progress.json').write_text(json.dumps(progress, indent=2,
                    default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
                if callback is not None:
                    callback(i,a,populations,diagnostics)
            kwargs['iteration_callback'] = accepted
            answer = original_solve(seed,model,grid,**kwargs)
            save_state(args.output/f'host-stage-{fraction:.3f}-final.npz',
                       answer.atmosphere,answer.population_state,model)
            return answer
        joint.solve = instrumented
        host = compute_dao(config, wave, data=data,
                           iteration_callback=lambda i,a,d: print(f'host iteration {i}: {d}',flush=True))
        joint.solve = original_solve
        if identity(data) != initial_identity:
            raise RuntimeError('numerical code or physical tables changed during the host solve')
        save_model_result(host, args.output/'host')
        result = synthesize_dao_trace_metals(host, abundance, wave, data=data, **options)
    metadata = {**result.metadata, 'smoke_only': args.smoke,
                'host_parameters': dict(teff=config.effective_temperature,logg=config.logg,log_h_he=5.),
                'abundances': abundance,'converged':result.converged,
                'iterations':result.iterations,'population_defect':result.population_defect,
                'comparison_to_observation_completed':False}
    np.savez(args.output/'spectrum.npz', wavelength=wave,flux=result.spectrum.surface_flux_lambda,
             background_flux=result.background_spectrum.surface_flux_lambda)
    np.savez(args.output/'populations.npz', **{
        f'{e}_{field}':getattr(state,field) for e,state in result.populations.items()
        for field in ('population_density','lte_population_density')})
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(converged=result.converged, iterations=result.iterations,
                          population_defect=result.population_defect, smoke_only=args.smoke),indent=2))

if __name__ == '__main__':
    main()
