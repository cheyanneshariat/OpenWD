"""Extend structure quadrature beyond 10 micron without moving existing nodes.

The original thermal interval is inadequate for opacity-weighted heating and
cooling, even when it captures nearly all bolometric flux. This scoped research
wrapper extends only that continuum quadrature, at the existing log spacing;
all line grids, opacity callbacks, physics and convergence gates are retained.
"""
import argparse
from contextlib import contextmanager
import sys
from unittest.mock import patch
import numpy as np
from wd_spectra import atmosphere


def extend_grid(grid,maximum):
    grid=np.asarray(grid)
    if maximum<=grid[-1]: raise ValueError('thermal extension must exceed the existing upper wavelength')
    step=np.log(grid[-1]/grid[-2])
    count=int(np.ceil(np.log(maximum/grid[-1])/step))
    tail=np.geomspace(grid[-1],maximum,count+1)[1:]
    return np.r_[grid,tail]


class StructureNumpy:
    def __init__(self,original,maximum,lyman_core_subdivisions=1):
        self.original,self.maximum=original,maximum
        self.lyman_core_subdivisions=lyman_core_subdivisions
    def __getattr__(self,name):return getattr(self.original,name)
    def arange(self,*args,**kwargs):
        grid=self.original.arange(*args,**kwargs)
        if args==(-3.,3.0001,.05) and self.lyman_core_subdivisions>1:
            interior=grid[:-1,None]+np.diff(grid)[:,None]*np.arange(1,self.lyman_core_subdivisions)/self.lyman_core_subdivisions
            refined=np.unique(np.r_[grid,interior.ravel()])
            print(f'LYMAN CORE QUADRATURE: {len(grid)} -> {len(refined)} offsets; original nodes retained',flush=True)
            return refined
        return grid
    def geomspace(self,start,stop,num=50,*args,**kwargs):
        grid=self.original.geomspace(start,stop,num,*args,**kwargs)
        if start==100. and stop==100000.:
            extended=extend_grid(grid,self.maximum)
            print(f'EXTENDED THERMAL QUADRATURE: {len(grid)} -> {len(extended)} continuum nodes; '
                  f'upper wavelength {self.maximum:g} A; original nodes unchanged',flush=True)
            return extended
        return grid


@contextmanager
def extended_quadrature(maximum,*,lyman_core_subdivisions=1):
    if not isinstance(lyman_core_subdivisions,int) or lyman_core_subdivisions<1:
        raise ValueError('Lyman core subdivisions must be a positive integer')
    with patch.object(atmosphere,'np',StructureNumpy(atmosphere.np,maximum,lyman_core_subdivisions)):
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--thermal-maximum-wavelength',type=float,default=1e8)
    parser.add_argument('--thermal-sweeps',type=int,default=0)
    parser.add_argument('--structure-angles',type=int,default=3)
    parser.add_argument('--project-thermal-convection',action='store_true')
    parser.add_argument('--temperature-profile-predictor',type=str)
    parser.add_argument('--resume-dense-iteration',type=str)
    parser.add_argument('--resume-pseudo-time-snapshot',action='store_true')
    parser.add_argument('--resume-thermal-sweeps',type=int,default=0)
    parser.add_argument('--thermal-branch-escape',action='store_true')
    parser.add_argument('--thermal-branch-preserve-convection',action='store_true')
    parser.add_argument('--predictor-preserve-convection',action='store_true')
    parser.add_argument('--predictor-reuse-pressure',action='store_true')
    parser.add_argument('--predictor-truncate-optical-depth',type=float)
    parser.add_argument('--predictor-refine-pressure',action='store_true')
    parser.add_argument('--predictor-unscaled-temperature',action='store_true')
    parser.add_argument('--direct-spectrum',action='store_true')
    parser.add_argument('--mass-conservative-transfer',action='store_true')
    parser.add_argument('--smooth-heminus-join',action='store_true')
    parser.add_argument('--pseudo-time-sweeps',type=int,default=0)
    parser.add_argument('--pseudo-time-output',type=str)
    parser.add_argument('--pseudo-time-implicit-convection',action='store_true')
    parser.add_argument('--pseudo-time-until-local-balance',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.pseudo_time_until_local_balance and not args.pseudo_time_sweeps:
        parser.error('local-balance handoff requires explicit pseudo-time initialization')
    if args.resume_pseudo_time_snapshot and not args.resume_dense_iteration:
        parser.error('pseudo-time snapshot selection requires its explicit source run directory')
    if args.pseudo_time_implicit_convection and (not args.pseudo_time_sweeps or '--exact-convection-tangent' not in remaining):
        parser.error('implicit pseudo-time convection requires pseudo-time sweeps and exact material tangent')
    if args.pseudo_time_sweeps<0 or bool(args.pseudo_time_sweeps)!=bool(args.pseudo_time_output):
        parser.error('pseudo-time initialization requires positive sweeps and a separate telemetry output')
    if args.pseudo_time_sweeps and (args.thermal_sweeps or args.resume_thermal_sweeps or '--full-coupled-newton' in remaining):
        parser.error('pseudo-time initialization must be tested separately from other initializer/auxiliary variants')
    if args.mass_conservative_transfer and ('--stable-transfer' not in remaining or not args.direct_spectrum):
        parser.error('mass-conservative transfer requires stable-transfer scope and consistent direct synthesis')
    if not np.isfinite(args.thermal_maximum_wavelength) or args.thermal_maximum_wavelength<=1e5:
        parser.error('thermal upper wavelength must be finite and exceed 1e5 A')
    if args.thermal_sweeps<0:parser.error('thermal sweeps must be nonnegative')
    if args.structure_angles<2:parser.error('at least two structure angles are required')
    if args.project_thermal_convection and not args.thermal_sweeps:
        parser.error('thermal convective projection requires thermal initialization')
    from contextlib import ExitStack
    from wd_spectra import adaptive_structure as adaptive
    predictor=None
    if args.resume_thermal_sweeps<0:
        parser.error('resume thermal sweeps must be nonnegative')
    if args.resume_thermal_sweeps and (not args.resume_dense_iteration or '--full-coupled-newton' in remaining):
        parser.error('resumed thermal initialization requires an iteration source and the original-sized solver')
    if args.thermal_branch_escape and not args.resume_thermal_sweeps:
        parser.error('thermal branch escape requires explicit resumed thermal initialization')
    if args.thermal_branch_preserve_convection and not args.thermal_branch_escape:
        parser.error('transport-preserving initialization requires explicit thermal branch escape')
    if args.resume_dense_iteration:
        if (args.temperature_profile_predictor or args.thermal_sweeps or args.predictor_preserve_convection
                or args.predictor_reuse_pressure or args.predictor_refine_pressure
                or args.predictor_unscaled_temperature or args.predictor_truncate_optical_depth is not None
                or '--mesh-hierarchy' in remaining or '--discrete-transport-seed' in remaining
                or '--physical-only' not in remaining):
            parser.error('iteration continuation must retain its exact state and enter the physical phase')
        table_parser=argparse.ArgumentParser(add_help=False)
        table_parser.add_argument('--interaction-table',required=True)
        table_args,_=table_parser.parse_known_args(remaining)
        from .dense_iteration_resume import IterationResume
        predictor=IterationResume(args.resume_dense_iteration,table_args.interaction_table,
            pseudo_time_snapshot=args.resume_pseudo_time_snapshot)
    if args.predictor_unscaled_temperature and not args.temperature_profile_predictor:
        parser.error('unscaled continuation requires an explicit temperature source')
    if (args.predictor_truncate_optical_depth is not None or args.predictor_refine_pressure) and not args.predictor_reuse_pressure:
        parser.error('pressure-domain truncation/refinement requires retained-pressure continuation')
    if (args.predictor_preserve_convection or args.predictor_reuse_pressure) and not args.temperature_profile_predictor:
        parser.error('preserving predictor convection requires a temperature predictor')
    if args.temperature_profile_predictor:
        table_parser=argparse.ArgumentParser(add_help=False)
        table_parser.add_argument('--interaction-table',required=True)
        table_args,_=table_parser.parse_known_args(remaining)
        if '--mesh-hierarchy' in remaining:
            parser.error('external temperature predictor currently supports an explicit single grid only')
        if args.predictor_reuse_pressure and '--discrete-transport-seed' in remaining:
            parser.error('retained-pressure continuation must not overwrite its temperatures with a diffusion initializer')
        from .dense_temperature_predictor import TemperaturePredictor
        predictor=TemperaturePredictor(args.temperature_profile_predictor,table_args.interaction_table,
            preserve_convection=args.predictor_preserve_convection,reuse_pressure=args.predictor_reuse_pressure,
            truncate_optical_depth=args.predictor_truncate_optical_depth,refine_pressure=args.predictor_refine_pressure,
            unscaled_temperature=args.predictor_unscaled_temperature)
    original=adaptive.solve_adaptive_lte_structure
    if predictor is not None:
        from .heminus_join_experiment import validate_continuation_policy,HARD_JOIN,SMOOTH_JOIN
        if validate_continuation_policy(predictor.heminus_join_policy,SMOOTH_JOIN if args.smooth_heminus_join else HARD_JOIN):
            predictor.description+='; EXPLICIT change from historical to smooth He-minus wavelength join'
    def record(seed,wave,**options):
        if wave[-1]!=args.thermal_maximum_wavelength:
            raise ValueError('structure wavelength extension was not applied')
        options['metadata']={**options.get('metadata',{}),
            'experimental_thermal_wavelength_maximum_angstrom':float(wave[-1]),
            'experimental_thermal_wavelength_count':len(wave),
            'experimental_structure_n_angle':options['n_angle']}
        from .heminus_join_experiment import METADATA_KEY,active_policy
        options['metadata'][METADATA_KEY]=active_policy()
        if predictor is not None:
            options['metadata']['experimental_source_heminus_wavelength_join']=predictor.heminus_join_policy
        if args.mass_conservative_transfer:
            options['compute_local_energy_response']=True
            options['metadata']['experimental_mass_conservative_transfer']=True
        if args.pseudo_time_sweeps:
            options['metadata'].update(experimental_pseudo_time_initializer_sweeps=args.pseudo_time_sweeps,
                experimental_pseudo_time_nonlinear_convection=args.pseudo_time_implicit_convection,
                experimental_pseudo_time_handoff_on_local_balance=args.pseudo_time_until_local_balance,
                experimental_pseudo_time_is_physical_evolution=False,
                experimental_pseudo_time_is_static_convergence=False)
        if predictor is not None:
            seed,options=predictor(seed,options)
        return original(seed,wave,**options)
    with ExitStack() as stack:
        if args.smooth_heminus_join:
            from .heminus_join_experiment import heminus_join_scope
            stack.enter_context(heminus_join_scope())
            print('EXPLICIT He-minus opacity join experiment: smooth 0.5063--1 micron overlap; '
                  'both source prescriptions and all other opacity ranges unchanged',flush=True)
        if args.pseudo_time_sweeps:
            from .pseudo_time_dense_experiment import pseudo_time_initializer
            stack.enter_context(pseudo_time_initializer(args.pseudo_time_sweeps,args.pseudo_time_output,
                implicit_convection=args.pseudo_time_implicit_convection,
                until_local_balance=args.pseudo_time_until_local_balance))
        if args.mass_conservative_transfer:
            from . import stable_feautrier_experiment as stable
            from .mass_transfer_experiment import mass_transfer_experiment
            stack.enter_context(patch.object(stable,'stable_transfer_experiment',mass_transfer_experiment))
        stack.enter_context(extended_quadrature(args.thermal_maximum_wavelength))
        stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',record))
        if args.thermal_sweeps or args.resume_thermal_sweeps:
            from .thermal_relaxation_experiment import thermal_initializer
            from .dense_helium_fluid_experiment import KB_EV
            stack.enter_context(thermal_initializer(args.thermal_sweeps or args.resume_thermal_sweeps,
                np.nextafter(.1/KB_EV,np.inf),17000.,
                project_convection=args.project_thermal_convection or args.thermal_branch_escape,
                allow_energy_barrier=args.thermal_branch_escape,
                preserve_convective_transport=args.thermal_branch_preserve_convection))
        from functools import partial
        from . import check_cool_db_transport_seed as runner
        if args.direct_spectrum:
            from .dense_direct_spectrum import direct_result
            original_compute=runner.compute_db
            def direct_compute(*arguments,**options):
                return direct_result(original_compute(*arguments,**options),args.structure_angles,
                    mass_conservative=args.mass_conservative_transfer)
            stack.enter_context(patch.object(runner,'compute_db',direct_compute))
        if predictor is not None:
            if predictor.reuse_pressure:
                stack.enter_context(patch.object(runner,'transport_seed',predictor.seed))
        original_run=runner.run
        def labeled_run(settings):
            settings.experimental_thermal_driver_options=vars(args).copy()
            if predictor is not None:
                settings.experimental_initializer_description=predictor.description
            return original_run(settings)
        stack.enter_context(patch.object(runner,'run',labeled_run))
        stack.enter_context(patch.object(runner,'radiative_equilibrium_helium_atmosphere',
            partial(runner.radiative_equilibrium_helium_atmosphere,n_angle=args.structure_angles)))
        from . import dense_helium_molecular_experiment as experiment
        sys.argv=[sys.argv[0]]+remaining
        experiment.main()


if __name__=='__main__':main()
