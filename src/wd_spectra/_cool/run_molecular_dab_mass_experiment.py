"""Explicit molecular H/He test of conservative transfer and thermal initialization.

Keeps the same mixed EOS: this does not substitute the pure-He dense EOS.
All stellar-flux, local-energy and measured-step gates remain in force.
Use a separate process per run because the legacy research runner has scopes.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from wd_spectra import adaptive_structure as adaptive
from wd_spectra._compat import trapezoid
from wd_spectra._mass_feautrier import mass_field
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.spectrum import planck_lambda_angstrom
from .extended_thermal_wavelength_experiment import extended_quadrature
from .mass_transfer_experiment import mass_transfer_experiment
from .pseudo_time_dense_experiment import pseudo_time_initializer
from . import stable_feautrier_experiment as stable
from . import run_molecular_dab_experiment as molecular_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--thermal-maximum', type=float, default=1e8)
    parser.add_argument('--angles', type=int, default=8)
    parser.add_argument('--require-convergence',action='store_true',
                        help='exit nonzero on failed numerical convergence; preserve exploratory outputs')
    parser.add_argument('--pseudo-sweeps', type=int, default=0)
    parser.add_argument('--relax-bottom', action='store_true')
    parser.add_argument('--tangent-convection', action='store_true')
    parser.add_argument('--positive-rates', action='store_true',
                        help='factor positive heating/cooling in proposals; accepted transfer is unchanged')
    parser.add_argument('--measured-proposals',action='store_true',
                        help='experimental secant correction to the nonlinear proposal; off by default')
    parser.add_argument('--augmented-convection', action='store_true',
                        help='couple ML2 velocity/temperature in the finite-material proposal')
    parser.add_argument('--discrete-initializer', action='store_true',
                        help='solve the sampled diffusion/ML2 seed with the SAME mixed materials')
    parser.add_argument('--newton-direction',action='store_true',
                        help='square equilibrated Newton direction, without a least-squares inner model')
    parser.add_argument('--iteration-weights',action='store_true',
                        help='refresh fixed physical energy weights between Newton linearizations')
    parser.add_argument('--material-probe',type=float,
                        help='explicit ln-T material difference for derivative diagnostics, not a physics change')
    parser.add_argument('--consistent-stark-edge',action='store_true',
                        help='keep the full Doppler-convolved profile fixed below its density table')
    parser.add_argument('--fully-coupled',action='store_true',
                        help='reevaluate the full temperature/ML2 auxiliary system at every outer trial')
    parser.add_argument('--bounded-coupled-temperature',action='store_true',
                        help='bound full-system temperature corrections while eliminating linear ML2 constraints')
    parser.add_argument('--physical-detuning-lyman',action='store_true',
                        help='interpolate Doppler-convolved Lyman profiles in physical wavelength coordinates')
    parser.add_argument('--lyman-core-subdivisions',type=int,default=1,
                        help='explicit independent refinement of each existing Lyman core quadrature interval')
    parser.add_argument('--condition-convection', action='store_true',
                        help='explicit initial gradient-conditioning stage, followed by unchanged physical gates')
    parser.add_argument('--compatible-coordinates',action='store_true',
                        help='solve temperature/ML2 compatibility at every actual trial; research only')
    parser.add_argument('--chemistry-tolerance',type=float,
                        help='explicit tighter molecular conservation accuracy; equilibrium equations unchanged')
    parser.add_argument('--least-squares-merit',action='store_true',
                        help='match globalization to the least-squares proposal; physical gates unchanged')
    args, remaining = parser.parse_known_args()
    target_parser = argparse.ArgumentParser(add_help=False)
    target_parser.add_argument('temperature', type=int)
    target_parser.add_argument('--output-root', type=Path, required=True)
    target_parser.add_argument('--log-h-he', type=float, required=True)
    run, _ = target_parser.parse_known_args(remaining)
    if args.angles < 2 or args.pseudo_sweeps < 0:
        parser.error('angles must be >=2 and pseudo sweeps nonnegative')
    if args.chemistry_tolerance is not None and (not np.isfinite(args.chemistry_tolerance)
            or not 0 < args.chemistry_tolerance <= 2e-11):
        parser.error('chemistry tolerance may only tighten the default 2e-11')
    if args.material_probe is not None and (not np.isfinite(args.material_probe) or args.material_probe<=0):
        parser.error('material probe must be finite positive')
    if '--stable-transfer' not in remaining or '--physical-only' not in remaining:
        parser.error('requires explicit --stable-transfer and --physical-only')
    if args.augmented_convection and '--nonlinear-materials' not in remaining:
        parser.error('augmented convection requires explicit --nonlinear-materials')
    if args.newton_direction and (args.augmented_convection or args.positive_rates or '--nonlinear-materials' in remaining):
        parser.error('square Newton direction is a separate proposal experiment')
    if args.iteration_weights and ('nonlinear-convection-direct-energy' not in remaining or args.newton_direction):
        parser.error('iteration weights require direct-energy payload proposals, without secant Newton directions')
    if args.fully_coupled and ('--auxiliary-ml2' not in remaining or
            any((args.newton_direction,args.iteration_weights,args.augmented_convection,args.pseudo_sweeps))):
        parser.error('full coupled system requires --auxiliary-ml2 and no other proposal/time experiment')
    if args.bounded_coupled_temperature and not args.fully_coupled:
        parser.error('bounded coupled temperatures require --fully-coupled')
    if args.compatible_coordinates and ('nonlinear-convection-current-energy' not in remaining or
            any((args.fully_coupled,args.pseudo_sweeps,args.condition_convection,args.iteration_weights,
                 args.measured_proposals,args.newton_direction,args.discrete_initializer)) or
            any(v in remaining for v in ('--auxiliary-ml2','--finite-material-repair','--inverse-ml2-step'))):
        parser.error('compatible coordinates require an independent current-energy physical solve')
    root = run.output_root/str(run.temperature)
    root.mkdir(parents=True, exist_ok=True)
    policy_path = root/'transport-options.json'
    if policy_path.exists():
        parser.error('refusing to overwrite experiment')
    from .research_paths import REPOSITORY, source_paths
    sources = source_paths()
    from .experiment_source_archive import archive_sources
    policy = dict(vars(args), argv=remaining,
        dense_pure_helium_eos_substituted=False,
        source_archive=str(root/'python-sources.tar.gz'),
        source_sha256=archive_sources(sources,root/'python-sources.tar.gz',base_directory=REPOSITORY))
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    original = adaptive.solve_adaptive_lte_structure
    coordinate_options={}
    original_newton = adaptive.solve_trust_region_newton
    def positive_newton(initial,evaluate,**settings):
        from wd_spectra.nonlinear import NonlinearEvaluation
        def marked(state,jacobian):
            ev=evaluate(state,jacobian)
            return NonlinearEvaluation(ev.residual,ev.jacobian,
                {**ev.payload,'diagnostic_positive_radiative_rates':args.positive_rates,
                 'diagnostic_measured_proposals':args.measured_proposals})
        return original_newton(initial,marked,**settings)
    def weighted_newton(initial,evaluate,**settings):
        from .frozen_energy_linearization import frozen_energy_evaluator
        return (positive_newton if args.positive_rates or args.measured_proposals else original_newton)(
            initial,frozen_energy_evaluator(evaluate),**settings)
    def solve(seed, wave, **options):
        options.update(n_angle=args.angles, transfer_discretization='column-mass',
                       compute_local_energy_response=True)
        coordinate_options.update(options)
        options['metadata']={**options.get('metadata',{}),
            'experimental_molecular_transport_policy':str(policy_path),
            'experimental_structure_n_angle':args.angles,
            'experimental_thermal_wavelength_maximum_angstrom':float(wave[-1]),
            'experimental_hydrogen_stark_low_density_policy':(
                'whole-profile-edge' if args.consistent_stark_edge else 'scaled-alpha'),
            'experimental_lyman_profile_interpolation':(
                'physical-detuning' if args.physical_detuning_lyman else 'field-scaled'),
            'experimental_lyman_core_subdivisions':args.lyman_core_subdivisions,
            'experimental_synthesis_matches_structure_transfer':True,
            'experimental_helium_ii_lines':seed.effective_temperature>=15000.,
            'experimental_lyman_profile_source':'stark'}
        options['metadata']['experimental_exact_material_coordinates']=args.compatible_coordinates
        options['metadata']['experimental_nonlinear_merit']=(
            'least-squares' if args.least_squares_merit or args.compatible_coordinates
            else 'legacy rms plus maximum')
        if args.compatible_coordinates:
            options['metadata']['experimental_coordinate_energy_flux']='independently checked ML2 coordinate'
            options['metadata']['experimental_coordinate_trust_subproblem']='physical nodal box'
            options['metadata']['experimental_coordinate_merit']='matching least squares'
        if args.chemistry_tolerance is not None:
            options['metadata']['experimental_molecular_conservation_tolerance']=args.chemistry_tolerance
        if args.condition_convection:
            options.update(use_convective_gradient_preconditioner=True,
                project_initial_convective_gradient=True,
                resume_supplied_structure_in_formal_flux_phase=False)
        if args.discrete_initializer:
            from dataclasses import replace
            from functools import partial
            from types import SimpleNamespace
            from .discrete_dense_transport_seed import discrete_seed
            local_runner=SimpleNamespace(atmosphere_at=partial(
                molecular_runner.runner.atmosphere_at,log_h_he=run.log_h_he))
            seed,error=discrete_seed(seed,local_runner,options)
            seed=replace(seed,metadata={**seed.metadata,'initialization_only':True,
                'radiative_equilibrium_converged':False})
            np.savez_compressed(root/'discrete-seed.npz',temperature=seed.temperature,
                gas_pressure=seed.gas_pressure,column_mass=seed.column_mass,
                rosseland_optical_depth=seed.rosseland_optical_depth,
                initialization_only=True,static_convergence_claim=False)
            options['metadata']={**options.get('metadata',{}),
                'experimental_discrete_mixed_seed':True,
                'experimental_discrete_diffusion_seed_error':error}
        result = original(seed, wave, **options)
        absorption = options['true_absorption'](result)
        scattering = options['scattering_opacity'](result)
        tau = optical_depth_from_mass_opacity(result.column_mass, absorption+scattering)
        planck = planck_lambda_angstrom(wave[:,None],result.temperature[None,:])
        source, field = mass_field(tau, planck, absorption, scattering,
            column_mass=result.column_mass, n_angle=args.angles)
        _, prescribed = mass_field(tau, source, absorption+scattering, np.zeros_like(scattering),
            column_mass=result.column_mass, n_angle=args.angles)
        defect = source-(absorption*planck+scattering*prescribed.mean_intensity)/(absorption+scattering)
        closure = float(np.max(wave[:,None]*abs(defect))/np.max(wave[:,None]*source))
        np.savetxt(root/'structure-grid-spectrum.txt',np.c_[wave,field.interface_flux[:,0]])
        target = STEFAN_BOLTZMANN*result.effective_temperature**4
        report = dict(structure_grid_only=True, independent_wavelength_validation=False,
            source_closure_scaled_error=closure,
            spectrum_integral_over_sigma_teff4=float(trapezoid(field.interface_flux[:,0],wave)/target),
            converged=bool(result.metadata['radiative_equilibrium_converged']))
        (root/'direct-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MOLECULAR MASS TRANSFER: '+json.dumps(report),flush=True)
        return result
    sys.argv=[sys.argv[0]]+remaining
    with ExitStack() as stack:
        if args.least_squares_merit:
            from wd_spectra import nonlinear
            from .compatible_coordinate_solve import least_squares_merit
            stack.enter_context(patch.object(nonlinear,'_residual_merit',least_squares_merit))
        if args.chemistry_tolerance is not None:
            from .precise_molecular_chemistry import precise_chemistry
            stack.enter_context(precise_chemistry(args.chemistry_tolerance))
        from .matched_molecular_synthesis import matched_synthesis
        stack.enter_context(matched_synthesis(molecular_runner.runner,args.angles))
        if args.compatible_coordinates:
            from .compatible_coordinate_solve import compatible_coordinate_solver
            stack.enter_context(compatible_coordinate_solver(coordinate_options))
        if args.physical_detuning_lyman:
            from .physical_detuning_stark_experiment import physical_detuning_lyman
            stack.enter_context(physical_detuning_lyman())
        if args.fully_coupled:
            from functools import partial
            from wd_spectra import _ml2_auxiliary as auxiliary
            from .full_coupled_dense_experiment import full_coupled_solve
            run_case=molecular_runner.runner.run
            def full_case(arguments):
                arguments.experimental_full_coupled_newton=True
                return run_case(arguments)
            stack.enter_context(patch.object(molecular_runner.runner,'run',full_case))
            stack.enter_context(patch.object(auxiliary,'solve_auxiliary_ml2_experiment',
                partial(full_coupled_solve,velocity_scaled_compatibility=True,
                    thermal_scaled_compatibility=True,frozen_local_scaling=True,
                    bounded_temperature_step=args.bounded_coupled_temperature)))
        if args.consistent_stark_edge:
            from .stark_density_edge_experiment import consistent_stark_density_edge
            stack.enter_context(consistent_stark_density_edge())
        if args.material_probe is not None:
            from wd_spectra._material_response import temperature_response_probes
            def probe(evaluate_offset,step,*,centered):
                return temperature_response_probes(evaluate_offset,args.material_probe,centered=centered)
            stack.enter_context(patch.object(adaptive,'temperature_response_probes',probe))
        if args.newton_direction:
            from . import solver_step_experiments as proposals
            from .equilibrated_temperature_step import equilibrated_step
            stack.enter_context(patch.object(proposals,'make_step',equilibrated_step))
        if args.augmented_convection:
            from . import solver_step_experiments as proposals
            from .augmented_ml2_proposal import augmented_step
            def augmented(state, ev, jac, radius, method, **settings):
                if method != 'nonlinear-convection-current-energy':
                    raise ValueError('augmented proposal requires current local energy rows')
                return augmented_step(state, ev, radius, settings['inner_max_evaluations'],
                    velocity_scaled_compatibility=True, thermal_scaled_compatibility=True)
            stack.enter_context(patch.object(proposals,'make_step',augmented))
        if args.iteration_weights:
            stack.enter_context(patch.object(adaptive,'solve_trust_region_newton',weighted_newton))
        elif args.positive_rates or args.measured_proposals:
            stack.enter_context(patch.object(adaptive,'solve_trust_region_newton',positive_newton))
        stack.enter_context(extended_quadrature(args.thermal_maximum,
            lyman_core_subdivisions=args.lyman_core_subdivisions))
        stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',solve))
        stack.enter_context(patch.object(stable,'stable_transfer_experiment',mass_transfer_experiment))
        if args.pseudo_sweeps:
            stack.enter_context(pseudo_time_initializer(args.pseudo_sweeps,root/'pseudo-time.jsonl',
                until_local_balance=True,relax_boundary=args.relax_bottom,
                tangent_convection=args.tangent_convection))
        molecular_runner.main()
    if args.require_convergence:
        from .check_molecular_dab_result import check_result
        if not check_result(root):raise SystemExit(1)


if __name__ == '__main__':
    main()
