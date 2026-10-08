"""Run the unchanged public cold-start API while retaining numerical checkpoints.

Instrumentation only: no checkpoint is read, no initial model is supplied,
and no solver budget, equation or tolerance is changed.
"""
import argparse,json,logging,time,hashlib,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from wd_spectra import DOConfig,DAOConfig,ModelData,run_model
from wd_spectra.hot_nlte import population_arrays
import wd_spectra._hot_structure as joint


def save_state(path,a,p,model):
    actual,reference=population_arrays(p)
    np.savez_compressed(path,temperature=a.temperature,column_mass=a.column_mass,
        gas_pressure=a.gas_pressure,rosseland_optical_depth=a.rosseland_optical_depth,
        population=actual,lte_population=reference,effective_temperature=a.effective_temperature,logg=a.logg,
        maximum_helium_ii_level=model.maximum_helium_ii_level,
        maximum_hydrogen_level=model.maximum_hydrogen_level)


def main(*,experiment=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teff',type=float,default=60000.)
    p.add_argument('--logg',type=float,default=8.)
    p.add_argument('--ratio',type=float)
    p.add_argument('--quality',choices=['quick','standard','production'],default='standard')
    p.add_argument('--helium-levels',type=int,default=32)
    p.add_argument('--hydrogen-levels',type=int,default=8)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    original=joint.solve
    directory=args.output/'numerical-diagnostics'
    start=time.monotonic()
    stages=[]
    def progress_record(record):
        directory.mkdir(parents=True,exist_ok=True)
        record=dict(elapsed_seconds=time.monotonic()-start,**record)
        with (directory/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        (directory/'latest-progress.json').write_text(json.dumps(record,indent=2)+'\n')
    def instrumented(seed,model,wave,**kwargs):
        directory.mkdir(parents=True,exist_ok=True)
        manifest.update(status='running',elapsed_seconds=time.monotonic()-start)
        (args.output/'cold-start-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
        fraction=kwargs.get('nlte_fraction',1.)
        prefix=directory/f'stage-{fraction:.3f}'
        population=kwargs.get('populations')
        save_state(str(prefix)+'-initial.npz',seed,model._rate_state(seed) if population is None else population,model)
        callback=kwargs.get('iteration_callback')
        def progress(i,a,p,d):
            save_state(str(prefix)+'-accepted.npz',a,p,model)
            progress_record(dict(iteration=i,**d))
            if callback is not None:callback(i,a,p,d)
        def jacobian(x,e):
            np.savez_compressed(str(prefix)+'-jacobian.npz',x=x,residual=e.residual,jacobian=e.jacobian)
            save_state(str(prefix)+'-jacobian-state.npz',e.payload[0],e.payload[1],model)
        kwargs.update(iteration_callback=progress,jacobian_callback=jacobian)
        result=original(seed,model,wave,**kwargs)
        save_state(str(prefix)+'-final.npz',result.atmosphere,result.population_state,model)
        stages.append(dict(nlte_fraction=fraction,elapsed_seconds=time.monotonic()-start,
                           atmosphere_metadata=result.atmosphere.metadata))
        (directory/'stages.json').write_text(json.dumps(stages,indent=2,
            default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n')
        return result
    joint.solve=instrumented
    kwargs=dict(effective_temperature=args.teff,logg=args.logg,quality=args.quality,
                maximum_helium_ii_level=args.helium_levels)
    config=DOConfig(**kwargs) if args.ratio is None else DAOConfig(**kwargs,log_hydrogen_to_helium=args.ratio,
                                                               maximum_hydrogen_level=args.hydrogen_levels)
    source=Path(joint.__file__).resolve().parent
    manifest=dict(cold_start=True,initial_model_supplied=False,checkpoint_inputs=[],config=asdict(config),
                  experiment=experiment,
                  source_sha256={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source.rglob('*.py')})
    try:
        run_model(config,args.output,data=ModelData.default(args.data_root),require_convergence=True)
        manifest['status']='converged'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        error=traceback.format_exc())
        raise
    finally:
        joint.solve=original
        if args.output.is_dir():
            manifest['elapsed_seconds']=time.monotonic()-start
            (args.output/'cold-start-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
