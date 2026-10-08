"""Explicit REOS + HNC He/He+/He2+/e research experiment, NOT production.

No density or electron floors; no automatic alternative EOS. HNC interactions
use Young1981 neutral, Bruno2010 atomic-ion and Chang2002 molecular-ion
potentials, not the precise interaction model of Kowalski2007. The latter's
thesis electron fit is used as printed. Collective He- ff/refraction and
in-medium molecular cross sections remain absent. Neutral-dominated bulk
thermodynamics comes from REOS; trace reaction energy is neglected only
within the explicit <=0.1% ionization validity guard.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import hashlib
from pathlib import Path
import sys
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from wd_spectra import eos, helium_molecular as molecular
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS, PLANCK, LIGHT_SPEED
from wd_spectra.dense_eos import read_helium_reos3_table
from .dense_helium_atomic_experiment import AtomicDenseEOS, atomic_dense_experiment
from .dense_helium_chemical_equilibrium import ExcessPotentials, solve_chemistry
from .dense_helium_fluid_experiment import (
    RadialGrid,solve_hnc,neutral_pair_ev,ion_pair_ev,electron_insertion_ev,KB_EV,
)
from .chang_helium_potential import molecular_ion_neutral_pair_ev,BOHR_ANGSTROM,HARTREE_EV
from .helium_dimer_state_sum import calculate_bound_states, DimerBoundStates

PHYSICS=('EXPERIMENTAL REOS3 + HNC He/He+/He2+ + thesis electron insertion; '
         'Chang bound-state molecular equilibrium; Stancil bf/ff with explicit donors; '
         'cross sections held below 4200 K; collective He- ff/refraction ABSENT; '
         'not validated dense-He atmosphere physics')
TABLE_VERSION='molecular-hnc-young1981-bruno2010-chang2002-tail-electron-domain-v4'


def insertion_tail_corrections(grid,temperature,density):
    """Long-range contribution to the HNC chemical-potential functional.

    At large r, c=-V/kT+O(r^-8), while h*gamma is higher order.
    Integrate the documented neutral r^-6 and molecular-ion polarization
    tails analytically, including the half-weight boundary omitted by DST.
    This does NOT add a missing tail to Bruno's short-range ion fit.
    Verify remaining finite-box terms by extending the entire HNC domain.
    """
    from .dense_helium_fluid_experiment import KB_EV
    n=np.asarray(density)/(HELIUM_MASS*1e24)
    r,dr=grid.extent,grid.dr
    c6=10.8*KB_EV*13.1/(13.1-6)*2.9673**6
    neutral=-4*np.pi*n*c6*(1/(3*r**3)+.5*dr/r**4)
    c4=1.3793/2*HARTREE_EV*BOHR_ANGSTROM**4
    a=2.046179*BOHR_ANGSTROM/2
    integral=r/(2*(r*r-a*a))+np.log((r+a)/(r-a))/(4*a)
    dimer=-4*np.pi*n*c4*(integral+.5*dr*r*r/(r*r-a*a)**2)
    return np.stack((neutral,np.zeros_like(neutral),dimer),axis=-1)


def molecular_isotherm(temperature,densities,*,grid=None,pair=None):
    grid=RadialGrid() if grid is None else grid
    pair=molecular_ion_neutral_pair_ev(grid.r) if pair is None else pair
    rho=np.asarray(densities,float)
    if (rho.ndim != 1 or not len(rho) or np.any(~np.isfinite(rho))
            or rho[0] < 0 or np.any(np.diff(rho) <= 0)):
        raise ValueError('densities must be nonnegative, finite and increasing')
    previous=[None]*3
    mu=np.empty((len(rho),3)); s0=np.empty(len(rho))
    potentials=[neutral_pair_ev(grid.r),ion_pair_ev(grid.r),pair]
    for i,rhoi in enumerate(rho):
        number=rhoi/(HELIUM_MASS*1e24)
        neutral=None
        errors=[]
        for j,potential in enumerate(potentials):
            state,mu[i,j]=solve_hnc(grid,temperature,number,potential,
                                  solvent=neutral,initial=previous[j])
            previous[j]=state.gamma
            errors.append(state.residual)
            if j == 0:
                neutral=state
                s0[i]=1/(1-number*grid.zero(state.direct))
        if i % 40 == 0 or i == len(rho)-1:
            print(f'molecular HNC T={temperature:g} rho={rhoi:.6g} '
                  f'mu={mu[i]} closure={max(errors):.3g}',flush=True)
    return mu+insertion_tail_corrections(grid,temperature,rho),s0


def build_table(path,workers=2):
    path=Path(path)
    if path.exists():
        raise ValueError('refusing to overwrite interaction table')
    if workers < 1:
        raise ValueError('workers must be positive')
    # Use the electron model's documented lower domain, rather than a
    # guessed lower atmospheric temperature or a fitted Teff switch.
    temperature=np.geomspace(np.nextafter(.1/KB_EV,np.inf),17000.,81)
    density=np.geomspace(1e-12,2.,161)
    grid=RadialGrid()
    pair=molecular_ion_neutral_pair_ev(grid.r)
    worker=partial(molecular_isotherm,densities=density,grid=grid,pair=pair)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(worker,temperature))
    levels=calculate_bound_states()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,temperature=temperature,density=density,
        mu=np.stack([r[0] for r in results]),structure_zero=np.stack([r[1] for r in results]),
        version=TABLE_VERSION,physics=PHYSICS,radial_points=grid.points,radial_extent=grid.extent,
        molecular_pair_ev=pair,bound_energy_ev=levels.energy_ev,
        bound_rotational_quantum_number=levels.rotational_quantum_number,
        bound_state_source=levels.source)


class MolecularDenseEOS(AtomicDenseEOS):
    table_version=TABLE_VERSION
    species_count=3
    physics=PHYSICS
    dimer_description='explicit He2+ bound-free and He*He+ free-free; no ideal-equilibrium donor substitution'
    output_stem='molecular-dense'

    def __init__(self,bulk,interaction_path):
        super().__init__(bulk,interaction_path)
        with np.load(self.path,allow_pickle=False) as table:
            self.levels=DimerBoundStates(table['bound_energy_ev'].copy(),
                table['bound_rotational_quantum_number'].copy(),str(table['bound_state_source']))

    def chemistry(self,temperature,density):
        mu=self.excess_potentials(temperature,density)
        q0=self.level_weights(temperature).sum(axis=-1)
        return solve_chemistry(temperature,density,
            ExcessPotentials(*mu,electron_insertion_ev(temperature,density),self.physics),
            equilibrium=self.levels.equilibrium,neutral_partition=q0)

    def level_weights(self,temperature):
        weights=eos.HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT*np.exp(
            -eos.HELIUM_I_LOW_TERM_ENERGY/(BOLTZMANN*np.asarray(temperature)[...,None]))
        if np.any(np.sum(weights[...,1:],axis=-1) > 1e-3):
            from .dense_helium_limits import DenseHeliumDomainError
            raise DenseHeliumDomainError('ground-state chemical approximation exceeded excited-state domain')
        return weights

    def lte(self,temperature,pressure,**options):
        t,p=np.broadcast_arrays(np.asarray(temperature,float),np.asarray(pressure,float))
        rho,_=self.bulk_state(t,p)
        chemical=self.chemistry(t,rho)
        weights=self.level_weights(t)
        partition=np.sum(weights,axis=-1)
        ne=chemical.electron
        return eos.HeliumLTEState(mass_density=rho,helium_nuclei_density=rho/HELIUM_MASS,
            neutral_he_density=chemical.neutral,singly_ionized_he_density=chemical.atomic_ion,
            doubly_ionized_he_density=np.zeros_like(ne),electron_density=ne,
            mean_ion_charge=ne/(rho/HELIUM_MASS),neutral_partition_function=partition,
            singly_ionized_partition_function=np.full_like(ne,2.),
            neutral_level_occupation_probability=np.ones_like(weights),
            neutral_level_population_density=chemical.neutral[...,None]*weights/partition[...,None],
            microfield_model='qmhd',
            neutral_radius_scale=options.get('neutral_radius_scale',eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE))

    def opacity_factory(self,original):
        def absorption(atmosphere,wavelength,**options):
            include=options.pop('include_helium_dimer_ion',True)
            result=original(atmosphere,wavelength,include_helium_dimer_ion=False,**options)
            if include:
                chemistry=self.chemistry(atmosphere.temperature,atmosphere.mass_density)
                result+=molecular_absorption(wavelength,atmosphere.temperature,chemistry)/atmosphere.mass_density
            return result
        return absorption


def molecular_absorption(wavelength,temperature,chemistry):
    """cm^-1; cross sections per actual molecular/atomic donor, respectively.

    Retains the existing table's explicit low-T cross-section hold. This is
    an acknowledged opacity limitation, not an extrapolated equilibrium law.
    Chemical potential shifts change populations, NOT photoionization edges.
    """
    wave=np.asarray(wavelength,float)[:,None]
    temp=np.asarray(temperature,float)[None,:]
    tab_t=np.maximum(temp,molecular._TEMPERATURE[0])
    bf=molecular._linear_interpolate_table(wave,tab_t,molecular._WAVELENGTH_BOUND_FREE,molecular._BOUND_FREE)
    ff=molecular._linear_interpolate_table(wave,tab_t,molecular._WAVELENGTH_FREE_FREE,molecular._FREE_FREE)
    bf=np.where((wave >= molecular._WAVELENGTH_BOUND_FREE[0]) & (wave <= molecular._WAVELENGTH_BOUND_FREE[-1]),bf,0.)
    ff=np.where((wave >= molecular._WAVELENGTH_FREE_FREE[0]) & (wave <= molecular._WAVELENGTH_FREE_FREE[-1]),ff,0.)
    stimulated=-np.expm1(-PLANCK*LIGHT_SPEED/(wave*1e-8*BOLTZMANN*temp))
    return (bf*chemistry.molecular_ion+ff*chemistry.neutral*chemistry.atomic_ion)*stimulated


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--interaction-table',type=Path,required=True)
    parser.add_argument('--build-table',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--least-squares-merit',action='store_true')
    parser.add_argument('--exact-dense-materials',action='store_true')
    parser.add_argument('--exact-convection-tangent',action='store_true')
    parser.add_argument('--constrained-inverse-proposal',action='store_true')
    parser.add_argument('--discrete-transport-seed',action='store_true')
    parser.add_argument('--convective-onset-proposal',action='store_true')
    parser.add_argument('--augmented-convection-proposal',action='store_true')
    parser.add_argument('--compatible-newton-proposal',action='store_true')
    parser.add_argument('--project-augmented-compatibility',action='store_true')
    parser.add_argument('--positive-radiative-rates',action='store_true')
    parser.add_argument('--equilibrate-augmented-step',action='store_true')
    parser.add_argument('--full-coupled-newton',action='store_true')
    parser.add_argument('--frozen-full-coupled-scaling',action='store_true')
    parser.add_argument('--bounded-full-coupled-step',action='store_true')
    parser.add_argument('--project-full-coupled-trials',action='store_true')
    parser.add_argument('--velocity-scaled-compatibility',action='store_true')
    parser.add_argument('--thermal-scaled-compatibility',action='store_true')
    parser.add_argument('--exact-ml2-trial-projection',action='store_true')
    parser.add_argument('--mesh-hierarchy',action='store_true')
    parser.add_argument('--hierarchy-coarse-maximum',type=int,default=80)
    parser.add_argument('--convective-prolongation',action='store_true')
    parser.add_argument('--allow-incomplete-coarse',action='store_true')
    parser.add_argument('--coarse-physical-iterations',type=int,default=20)
    args,remaining=parser.parse_known_args()
    if args.build_table:
        if remaining:
            parser.error('table build does not accept atmosphere arguments')
        return build_table(args.interaction_table,args.workers)
    forbidden=('--reos3','--helium-dimer','--log-h-he','--reuse-fresh-seed','--resume-atmosphere',
        '--refine-atmosphere','--extend-atmosphere','--stable-helium-thermodynamics','--smooth-h2','--state-sum-h2')
    if any(x.split('=')[0] in forbidden for x in remaining):
        parser.error('dense chemical experiment requires fresh pure helium, no EOS substitutions')
    from . import check_cool_db_transport_seed as runner
    from wd_spectra.models import ModelData
    from .smooth_reos3_experiment import SmoothREOS3
    bulk=read_helium_reos3_table(ModelData.default().helium_reos3)
    model=MolecularDenseEOS(SmoothREOS3(bulk,.1/KB_EV,17000.),args.interaction_table)
    model.exact_dense_materials=args.exact_dense_materials
    model.positive_radiative_rates=args.positive_radiative_rates
    model.full_coupled_newton=args.full_coupled_newton
    model.experiment_options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    from .research_paths import REPOSITORY, source_paths
    model.source_sha256={str(p):hashlib.sha256((REPOSITORY/p).read_bytes()).hexdigest()
                         for p in source_paths()}
    run_parser=argparse.ArgumentParser(add_help=False)
    run_parser.add_argument('temperature',type=int)
    run_parser.add_argument('--output-root',type=Path,required=True)
    run_parser.add_argument('--n-depth',type=int,default=80)
    run_args,_=run_parser.parse_known_args(remaining)
    print(model.physics,flush=True)
    sys.argv=[sys.argv[0]]+remaining
    with ExitStack() as stack:
        if args.exact_convection_tangent:
            if '--stable-transfer' not in remaining:
                parser.error('exact convection tangent requires the late stable-transfer research scope')
            from . import stable_feautrier_experiment as stable
            from contextlib import contextmanager
            from .exact_dense_tangent_experiment import exact_dense_tangent_experiment
            original_transfer_scope=stable.stable_transfer_experiment
            @contextmanager
            def tangent_scope():
                with original_transfer_scope(),exact_dense_tangent_experiment(model.bulk):
                    yield
            stack.enter_context(patch.object(stable,'stable_transfer_experiment',tangent_scope))
        if args.allow_incomplete_coarse:
            if not args.mesh_hierarchy or args.coarse_physical_iterations<1:
                parser.error('incomplete coarse stages require a mesh hierarchy and positive iteration budget')
            from wd_spectra import adaptive_structure as adaptive
            original_newton=adaptive.solve_trust_region_newton
            def bounded_coarse(initial,evaluate,**settings):
                first=evaluate(initial,False)
                if (first.payload['energy_balance_is_physical_flux']
                        and first.payload['atmosphere'].n_depth<run_args.n_depth):
                    settings['maximum_iterations']=min(settings['maximum_iterations'],args.coarse_physical_iterations)
                return original_newton(initial,evaluate,**settings)
            stack.enter_context(patch.object(adaptive,'solve_trust_region_newton',bounded_coarse))
        if args.convective_prolongation and not args.mesh_hierarchy:
            parser.error('convective prolongation requires a fresh mesh hierarchy')
        if args.mesh_hierarchy:
            if not args.discrete_transport_seed:
                parser.error('mesh hierarchy requires a discrete fresh seed')
            from .dense_mesh_hierarchy import hierarchy_solve
            stack.enter_context(patch.object(runner,'radiative_equilibrium_helium_atmosphere',
                partial(hierarchy_solve,runner.radiative_equilibrium_helium_atmosphere,
                        coarse_maximum=args.hierarchy_coarse_maximum,
                        allow_incomplete_coarse=args.allow_incomplete_coarse,
                        before_level=lambda parent:setattr(model,'mesh_parent',parent))))
        if args.velocity_scaled_compatibility and not (args.full_coupled_newton or args.augmented_convection_proposal):
            parser.error('current compatibility scaling requires an augmented proposal or full coupled Newton')
        if args.thermal_scaled_compatibility and not ((args.augmented_convection_proposal or args.full_coupled_newton) and args.velocity_scaled_compatibility):
            parser.error('thermal compatibility scaling requires a velocity-scaled augmented system')
        if args.full_coupled_newton:
            if '--auxiliary-ml2' not in remaining or args.exact_dense_materials:
                parser.error('full coupled Newton requires auxiliary dispatch, not cheap material proposals')
            from wd_spectra import _ml2_auxiliary as auxiliary
            from wd_spectra import adaptive_structure as adaptive
            from .full_coupled_dense_experiment import full_coupled_solve
            full_material_context={} if args.project_full_coupled_trials else None
            stack.enter_context(patch.object(auxiliary,'solve_auxiliary_ml2_experiment',
                partial(full_coupled_solve,velocity_scaled_compatibility=args.velocity_scaled_compatibility,
                    thermal_scaled_compatibility=args.thermal_scaled_compatibility,
                    frozen_local_scaling=args.frozen_full_coupled_scaling,
                    bounded_temperature_step=args.bounded_full_coupled_step,
                    material_projection_context=full_material_context)))
            original_full_adaptive=adaptive.solve_adaptive_lte_structure
            def full_response(seed,wave,**options):
                options['compute_local_energy_response']=True
                if full_material_context is not None:
                    full_material_context['options']=options
                    options['metadata']={**options.get('metadata',{}),
                        'experimental_full_ml2_material_projection':True,
                        'experimental_full_ml2_projection_changes_flux':False}
                return original_full_adaptive(seed,wave,**options)
            stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',full_response))
        elif args.frozen_full_coupled_scaling or args.bounded_full_coupled_step or args.project_full_coupled_trials:
            parser.error('full-system scaling/step options require full coupled Newton')
        if args.project_augmented_compatibility and not args.augmented_convection_proposal:
            parser.error('compatibility projection requires the augmented proposal')
        if args.equilibrate_augmented_step and not args.augmented_convection_proposal:
            parser.error('augmented equilibration requires the augmented proposal')
        if args.compatible_newton_proposal:
            if args.augmented_convection_proposal or args.constrained_inverse_proposal or not args.exact_dense_materials or '--inverse-ml2-step' not in remaining:
                parser.error('compatible Newton requires exact dense materials and inverse dispatch, no other inverse variant')
            from . import inverse_ml2_proposal as inverse
            stack.enter_context(patch.object(inverse,'inverse_step',inverse.newton_inverse_step))
        if args.augmented_convection_proposal:
            if args.constrained_inverse_proposal or not args.exact_dense_materials or '--inverse-ml2-step' not in remaining:
                parser.error('augmented proposal requires exact dense materials and inverse dispatch, not constrained inverse')
            from . import inverse_ml2_proposal as inverse
            from .augmented_ml2_proposal import augmented_step
            stack.enter_context(patch.object(inverse,'inverse_step',
                partial(augmented_step,project_compatibility=args.project_augmented_compatibility,
                        equilibrate=args.equilibrate_augmented_step,
                        velocity_scaled_compatibility=args.velocity_scaled_compatibility,
                        thermal_scaled_compatibility=args.thermal_scaled_compatibility)))
        if args.exact_ml2_trial_projection:
            if not args.augmented_convection_proposal or args.project_augmented_compatibility:
                parser.error('exact trial projection requires an augmented proposal, not an approximate projection')
            from .exact_ml2_trial_projector import exact_trial_projection
            stack.enter_context(exact_trial_projection())
        if args.discrete_transport_seed:
            from wd_spectra import adaptive_structure as adaptive
            from .discrete_dense_transport_seed import discrete_seed
            original_adaptive=adaptive.solve_adaptive_lte_structure
            def initialize(seed,wavelength,**options):
                if args.full_coupled_newton:
                    options['compute_local_energy_response']=True
                refining = args.mesh_hierarchy and options.get('resume_supplied_structure_in_formal_flux_phase',False)
                if refining:
                    # A coarse physical solution supplies the initializer;
                    # replacing it with a diffusion atmosphere would discard
                    # the hierarchy's non-gray information.
                    if args.convective_prolongation:
                        from .convective_mesh_prolongation import prolongate
                        seed,error=prolongate(seed,model.mesh_parent,runner,options)
                        options['metadata']={**options.get('metadata',{}),
                            'experimental_convective_prolongation_error':error}
                    return original_adaptive(seed,wavelength,**options)
                initialized,defect=discrete_seed(seed,runner,options)
                suffix = f'-{seed.n_depth}' if args.mesh_hierarchy else ''
                checkpoint=run_args.output_root/str(run_args.temperature)/f'experimental-discrete-seed{suffix}.npz'
                if checkpoint.exists():
                    raise ValueError('refusing to overwrite discrete initializer checkpoint')
                np.savez_compressed(checkpoint,
                    experimental_temperature=initialized.temperature,
                    experimental_pressure=initialized.gas_pressure,
                    experimental_density=initialized.mass_density,
                    experimental_column_mass=initialized.column_mass,
                    experimental_tau=initialized.rosseland_optical_depth,
                    experimental_diffusion_ml2_defect=defect,
                    experimental_physics=model.physics,
                    interaction_table_sha256=model.sha256)
                options['metadata']={**options.get('metadata',{}),
                    'experimental_discrete_transport_seed':True,
                    'experimental_seed_diffusion_ml2_defect':defect}
                return original_adaptive(initialized,wavelength,**options)
            stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',initialize))
        if args.constrained_inverse_proposal:
            if '--inverse-ml2-step' not in remaining or not args.exact_dense_materials:
                parser.error('constrained inverse requires --inverse-ml2-step --exact-dense-materials')
            from . import inverse_ml2_proposal as inverse
            stack.enter_context(patch.object(inverse,'inverse_step',
                partial(inverse.constrained_inverse_step,allow_onset=args.convective_onset_proposal)))
        elif args.convective_onset_proposal:
            parser.error('--convective-onset-proposal requires --constrained-inverse-proposal')
        if args.exact_dense_materials:
            if '--nonlinear-materials' not in remaining:
                parser.error('--exact-dense-materials requires --nonlinear-materials')
            from . import convective_consistency_experiment as consistency
            from .dense_helium_materials import DenseExactMaterials
            stack.enter_context(patch.object(consistency,'TabulatedMaterials',
                partial(DenseExactMaterials,bulk=model.bulk)))
        stack.enter_context(atomic_dense_experiment(runner,model,least_squares_merit=args.least_squares_merit,
                                 opacity_factory=model.opacity_factory))
        runner.main()
    path=run_args.output_root/str(run_args.temperature)/'summary.json'
    summary=json.loads(path.read_text())
    summary.update(physics=model.physics,validated_full_physics=False,interaction_table_sha256=model.sha256)
    path.write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
