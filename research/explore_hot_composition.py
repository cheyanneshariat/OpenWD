#!/usr/bin/env python3
"""Screen additional published G191-B2B elements on the certified H/He host."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import solve_hot_trace_metals, fixed_electron_metal_reference, _atom_selection
from wd_spectra.light_metal_nlte import (read_tlusty_photoionization_threshold_data,
    reduced_light_metal_wavelength, hot_metal_line_nlte_coefficients,
    light_metal_bound_free_nlte_coefficients)
from wd_spectra.metals import ATOMIC_MASS_U, ATOMIC_NUMBER, EV_TO_WAVENUMBER
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
from wd_spectra.nlte_core import NLTETransferCoefficients
from resynthesize_hot_trace import load_frozen_host, identity
from explore_hot_trace import load_guesses
from hot_trace_collisions import carbon_collisions
from hot_trace_composition import ABUNDANCES, ALTERNATIVES, LIGHT_COUNTS, load_composition_data
from compare_hot_daz_benchmark import compare

# Fixed before inspecting predictions; cover lines used in the published study.
DIAGNOSTICS = [
 ('NV1238','N','V',1238.821,'e140h',144000.,23.8),
 ('NV1242','N','V',1242.804,'e140h',144000.,23.8),
 ('OIV1338','O','IV',1338.615,'e140h',144000.,23.8),
 ('OIV1343','O','IV',1343.514,'e140h',144000.,23.8),
 ('AlIII1854','Al','III',1854.716,'e230h',144000.,23.8),
 ('AlIII1862','Al','III',1862.790,'e230h',144000.,23.8),
 ('PV1117','P','V',1117.977,'fuv',20000.,None),
 ('PV1128','P','V',1128.008,'fuv',20000.,None),
 ('SIV1062','S','IV',1062.662,'fuv',20000.,None),
 ('SIV1072','S','IV',1072.974,'fuv',20000.,None),
 ('FeV1409','Fe','V',1409.453,'e140h',144000.,23.8),
 ('NiV1306','Ni','V',1306.624,'e140h',144000.,23.8),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('csi','light','all'),default='all')
    p.add_argument('--seed',type=Path,default=Path('results/hot-daz/explore-c240-chianti'))
    p.add_argument('--atomic-data',type=Path,default=Path('results/hot-daz/multimetal-data'))
    p.add_argument('--host-directory',type=Path,default=None,
        help='certified frozen H/He host to use instead of the one recorded in the seed metadata')
    p.add_argument('--accelerated-lambda',action='store_true',
        help='precondition line rates with the diagonal approximate lambda operator (MALI)')
    p.add_argument('--convergence-criterion',choices=('population','opacity','flux'),default='population')
    p.add_argument('--iron-group-top-charge',type=int,default=None,
        help='NLTE Fe/Ni: all bound levels of charges 3..N-1 and the ground of charge N; '
             'LTE remainder opacity of charges >= N is dropped (default: legacy N=6, remainder kept)')
    p.add_argument('--cold-start',action='store_true',
        help='start every metal from LTE populations (the seed still supplies host and atom sizes)')
    p.add_argument('--profile-block-tolerance',type=float,default=None,
        help='line-profile block quadrature tolerance for the iteration (solver default if omitted; 0 = exact)')
    p.add_argument('--chianti-recombination',type=Path,default=None,
        help='CHIANTI rr/drparams directory: total recombination for NLTE Fe/Ni beyond the explicit atom')
    p.add_argument('--kurucz-positions',type=Path,default=None,
        help='directory of checksummed Kurucz Fe/Ni IV-VII .pos files to supplement EUV transitions')
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--tolerance',type=float,default=None,
        help='convergence tolerance (default: 3e-3 for the flux criterion, otherwise 1e-4)')
    p.add_argument('--damping',type=float,default=.5)
    p.add_argument('--acceleration-depth',type=int,default=6)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--require-convergence',action='store_true')
    p.add_argument('--energy-audit',action='store_true',help='measure frozen-host energy imbalance after the solve')
    p.add_argument('--elementwise-acceleration',action='store_true')
    p.add_argument('--ionwise-acceleration',action='store_true')
    p.add_argument('--alternative-abundances',action='store_true')
    p.add_argument('--oxygen-op',action='store_true')
    p.add_argument('--oxygen-collisions',type=Path)
    p.add_argument('--published-alternatives',nargs='+',choices=tuple(ALTERNATIVES),default=[])
    p.add_argument('--oxygen-levels',type=int,nargs=5)
    p.add_argument('--nlte-iron-group',nargs='+',choices=('Fe','Ni'),default=[])
    p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.checkpoint_every<0:raise ValueError('checkpoint interval must be nonnegative')
    if args.elementwise_acceleration and args.ionwise_acceleration:
        raise ValueError('choose element or ion acceleration groups, not both')
    if args.output.exists():raise ValueError('preserve earlier experiments; choose a new output directory')
    if args.tolerance is None:
        from wd_spectra.hot_trace_metals import FLUX_TOLERANCE
        args.tolerance=FLUX_TOLERANCE if args.convergence_criterion=='flux' else 1e-4
    tolerance=args.tolerance
    source_files=('explore_hot_composition.py','hot_trace_composition.py','hot_trace_collisions.py',
                  'explore_hot_trace.py','hot_trace_oxygen_collisions.py')
    if args.energy_audit:source_files+=('hot_trace_energy_balance.py',)
    if args.chianti_recombination is not None:source_files+=('hot_trace_recombination.py',)
    if args.elementwise_acceleration or args.ionwise_acceleration:source_files+=('hot_trace_acceleration.py',)
    source_snapshot={f:Path(__file__).with_name(f).read_bytes() for f in source_files}
    data=ModelData.default();ident=identity(data)
    seedmeta=json.loads((args.seed/'metadata.json').read_text())
    if args.mode!='csi':
        args.oxygen_op=args.oxygen_op or seedmeta.get('oxygen_op',False)
        if args.oxygen_collisions is None and seedmeta.get('oxygen_collisions'):
            args.oxygen_collisions=Path(seedmeta['oxygen_collisions'])
    if args.mode=='all':args.nlte_iron_group=args.nlte_iron_group or seedmeta.get('nlte_iron_group',[])
    if args.host_directory is not None:seedmeta['host_directory']=str(args.host_directory.resolve())
    host=load_frozen_host(Path(seedmeta['host_directory']),data,allow_trace_only_changes=True)
    db,photo,audit=load_composition_data(data,args.atomic_data,iron_group=args.mode=='all',
        kurucz_positions=args.kurucz_positions)
    counts={e:{int(q):n for q,n in seedmeta['levels_per_charge'][e].items()} for e in ('C','Si')}
    if args.mode!='csi':
        counts.update({e:{int(q):n for q,n in seedmeta.get('levels_per_charge',{}).get(e,v).items()}
                       for e,v in LIGHT_COUNTS.items()})
    if args.oxygen_levels:
        if args.mode=='csi':raise ValueError('oxygen options require a multi-element mode')
        counts['O']=dict(zip(range(2,7),args.oxygen_levels))
    if args.oxygen_op and args.mode=='csi':raise ValueError('oxygen OP requires a multi-element mode')
    if args.nlte_iron_group and args.mode!='all':raise ValueError('NLTE Fe/Ni requires --mode all')
    top_charge=6 if args.iron_group_top_charge is None else args.iron_group_top_charge
    if top_charge<5:raise ValueError('iron-group top charge must be at least 5')
    for e in args.nlte_iron_group:
        counts[e]={q:int(sum(l.energy_wavenumber/EV_TO_WAVENUMBER < db.ions[e,q].ionization_energy_ev-.1
                         for l in db.ions[e,q].levels)) for q in range(3,top_charge)}
        counts[e][top_charge]=1
    for e,stages in counts.items():
        for q,n in stages.items():
            ion=db.ions[e,q]
            chosen=sorted(ion.levels,key=lambda l:l.energy_wavenumber)[:n]
            if len(chosen)!=n or (ion.ionization_energy_ev is not None and
                    any(l.energy_wavenumber/EV_TO_WAVENUMBER >= ion.ionization_energy_ev-.05 for l in chosen)):
                raise ValueError(f'explicit {e} {q} atom contains unbound levels or invalid size')
    composition=dict(ABUNDANCES)
    composition.update({e:(choice['number_ratio'],choice['inferred_from_ion'])
                        for e,choice in seedmeta.get('abundance_choices',{}).items()})
    if args.alternative_abundances:composition.update(ALTERNATIVES)
    composition.update({e:ALTERNATIVES[e] for e in args.published_alternatives})
    abundances={e:float(np.log10(composition[e][0])) for e in counts}
    extras={e:float(np.log10(composition[e][0])) for e in ('Fe','Ni') if e not in counts} if args.mode=='all' else {}
    # Include LTE opacity species in the same trace-budget guard as NLTE metals.
    ref=fixed_electron_metal_reference(host.atmosphere,db,{**abundances,**extras})
    mass=sum(ref.element_number_density[e]*ATOMIC_MASS_U[e]*1.66053906660e-24 for e in ref.element_number_density)/host.atmosphere.mass_density
    charge=sum(ref.element_number_density[e]*ATOMIC_NUMBER[e] for e in ref.element_number_density)/host.atmosphere.electron_density
    if max(mass)>1e-3 or max(charge)>1e-2:raise ValueError('total mixture exceeds trace budgets')
    collisions,collision_audit=carbon_collisions(Path('results/hot-daz/chianti-carbon'),db,counts['C'])
    total_recombination=None
    if args.chianti_recombination is not None:
        if not args.nlte_iron_group:raise ValueError('--chianti-recombination applies to NLTE Fe/Ni')
        from hot_trace_recombination import chianti_total_recombination
        total_recombination={};recombination_audit={}
        for e in args.nlte_iron_group:
            stages=sorted(counts[e]);parents=[q for q in stages[1:]]
            total_recombination[e],recombination_audit[e]=chianti_total_recombination(args.chianti_recombination,e,parents)
        collision_audit['total_recombination']=recombination_audit
    if args.oxygen_collisions:
        from hot_trace_oxygen_collisions import oxygen_collisions
        oxygen_rates,oxygen_audit=oxygen_collisions(args.oxygen_collisions,db,counts['O'])
        collisions.update(oxygen_rates);collision_audit['oxygen']=oxygen_audit
    thresholds={'C':{2:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c3.dat'),
                     3:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c4_35+2lev.dat')}}
    if args.oxygen_op:
        thresholds['O']={q:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/f'o{q+1}.dat')
                         for q in (3,4,5)}
    wave=np.unique(np.concatenate([host.spectrum.wavelength_angstrom]+([] if args.diagnostic_only else [np.arange(910.,1990.001,.01)])
        +[np.arange(d[3]-1.5,d[3]+1.501,.002) for d in DIAGNOSTICS]))
    model=_model_from_config(host.config,data)
    def host_background(w):return model.transfer_coefficients(host.atmosphere,w,host.population_state)
    host_final=host_background(wave)
    # The host's own angular quadrature (3 standard, 4 production).
    n_angle=model.n_angle
    bg_flux=transfer_field(host.atmosphere,host_final,n_angle=n_angle)[1].interface_flux[:,0]
    # Preserve LTE opacity outside every promoted iron-group atom; replace
    # only its closed lines and explicit ground continua, never double count.
    background_abundances={e:float(np.log10(composition[e][0])) for e in ('Fe','Ni')} if args.mode=='all' else {}
    extra_ref=fixed_electron_metal_reference(host.atmosphere,db,background_abundances) if background_abundances else None
    unity={(e,s.charge):np.ones(host.atmosphere.n_depth) for e in background_abundances for s in db.ion_stages(e)}
    closed={e:_atom_selection(db,e,counts[e])[0] for e in args.nlte_iron_group}
    # With an explicit top charge, the NLTE atom represents every stage >= N
    # by the ground of N (a recombination sink): drop their Saha-populated LTE
    # line/edge opacity instead of pairing it with an LTE source function.
    drop_charge=None if args.iron_group_top_charge is None else top_charge
    retained={e:{(e,ion.charge,l.lower_index,l.upper_index) for ion in db.ion_stages(e)
                 if drop_charge is None or ion.charge<drop_charge
                 for l in ion.transitions} - closed[e] for e in closed}
    remainder_block_tolerance=1e-4 if args.profile_block_tolerance is None else args.profile_block_tolerance
    def background(w):
        base=host_final if np.array_equal(w,wave) else host_background(w)
        if not background_abundances:return base
        absorption=base.true_absorption.copy();emission=base.thermal_emissivity.copy()
        for e in background_abundances:
            # Rate grid: the same block quadrature as the NLTE iteration; the
            # emergent-spectrum grid stays exact.
            a,j=hot_metal_line_nlte_coefficients(host.atmosphere,w,db,extra_ref,unity,
                elements=(e,),minimum_oscillator_strength=1e-4,maximum_lines=None,
                transition_keys=retained.get(e),include_static_linear_stark=False,
                profile_block_tolerance=0. if np.array_equal(w,wave) else remainder_block_tolerance)
            bf_counts=({s.charge:(0 if s.charge in sorted(counts[e])[:-1]
                                  or (drop_charge is not None and s.charge>=drop_charge) else 1)
                        for s in db.ion_stages(e)} if e in closed else None)
            b,k=light_metal_bound_free_nlte_coefficients(host.atmosphere,w,db,extra_ref,photo,unity,
                elements=(e,),levels_per_charge=bf_counts,include_explicit_kramers=True)
            absorption+=a+b;emission+=j+k
        return NLTETransferCoefficients(w,absorption,emission,base.scattering,
            {'iron_group_LTE_scope':'all opacity of unpromoted species; nonexplicit opacity of promoted species'})
    grid=np.unique(np.concatenate([reduced_light_metal_wavelength(db,e,n,
        photoionization_threshold_data=thresholds.get(e)) for e,n in counts.items()]
        # Resolve the LTE UV line forest on the radiation mesh as well.
        +([np.arange(880.,1990.,.02)] if background_abundances else [])))
    args.output.mkdir(parents=True)
    (args.output/'source-snapshots').mkdir()
    for name,content in source_snapshot.items():(args.output/'source-snapshots'/name).write_bytes(content)
    (args.output/'atomic-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.output/'collision-audit.json').write_text(json.dumps(collision_audit,indent=2)+'\n')
    common=dict(experimental=True,exploratory=True,smoke_only=False,case='composition-'+args.mode,
        host_parameters=seedmeta['host_parameters'],abundances={**abundances,**extras},
        nlte_elements=list(abundances),lte_background_elements=list(extras),levels_per_charge=counts,
        lte_reservoir_opacity_elements=list(closed),
        promoted_iron_group_scope=('explicit levels/closed lines in NLTE; other lines/stages retain LTE opacity'
            if drop_charge is None else f'explicit levels/closed lines in NLTE through charge {top_charge} (ground); '
            f'LTE opacity of charges < {top_charge} outside the atom; charges >= {top_charge} have no LTE remainder'),
        iron_group_top_charge=top_charge,
        abundance_choices={e:dict(number_ratio=composition[e][0],inferred_from_ion=composition[e][1])
                           for e in [*abundances,*extras]},
        host_directory=seedmeta['host_directory'],host_atmosphere_convergence='converged',
        host_reuse_scope=host.metadata['frozen_host_reuse_scope'],atmosphere_recomputed=False,
        host_populations_updated=False,electron_density_updated=False,observationally_validated=False,
        numerical_identity=ident,population_wavelength_points=len(grid),n_angle=n_angle,
        oxygen_op=args.oxygen_op,nlte_iron_group=args.nlte_iron_group,
        kurucz_positions=None if args.kurucz_positions is None else str(args.kurucz_positions),
        accelerated_lambda=args.accelerated_lambda,convergence_criterion=args.convergence_criterion,
        chianti_recombination=None if args.chianti_recombination is None else str(args.chianti_recombination),diagnostic_only=args.diagnostic_only,
        oxygen_collisions=str(args.oxygen_collisions) if args.oxygen_collisions else None,
        seed_directory=str(args.seed),
        requested_population_tolerance=args.tolerance,maximum_iterations=args.iterations,
        requested_damping=args.damping,requested_acceleration_depth=args.acceleration_depth,
        elementwise_acceleration=args.elementwise_acceleration,
        ionwise_acceleration=args.ionwise_acceleration,
        convergence_required=args.require_convergence,
        published_alternative_elements=args.published_alternatives,
        initial_guess_elements=[] if args.cold_start else sorted(load_guesses(args.seed,elements=counts)),
        cold_start=args.cold_start,
        all_metals_maximum_mass_ratio=float(max(mass)),all_metals_maximum_electron_bound=float(max(charge)),
        seed_population_sha256=hashlib.sha256((args.seed/'populations.npz').read_bytes()).hexdigest(),
        research_code_sha256={f:hashlib.sha256(content).hexdigest() for f,content in source_snapshot.items()},
        limitations=[x for x in audit['missing_physics'] if x!='Fe/Ni NLTE populations']
            +(['LTE populations for '+','.join(extras)] if extras else [])
            +['incomplete Fe/Ni atoms and approximate excited continua',
              'fixed H/He atmosphere; no metal feedback on structure; no abundance fit'])
    history=[];element_acceleration=None
    def save(directory,spectrum,states,metadata):
        directory.mkdir(parents=True,exist_ok=True)
        np.savez(directory/'spectrum.npz',wavelength=wave,flux=spectrum.surface_flux_lambda,background_flux=bg_flux)
        np.savez(directory/'populations.npz',**{f'{e}_{key}':getattr(s,key)
            for e,s in states.items() for key in ('level_key','population_density','lte_population_density')})
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return compare(directory,'results/hot-daz/g191-b2b',directory/'comparison',allow_unconverged=True)['metrics']
    def callback(i,defect,states,synthesize):
        from datetime import datetime
        print(f'{datetime.now().isoformat(timespec="seconds")} {args.mode}: iteration {i}, defect {defect:.5g}',flush=True)
        if element_acceleration is not None:
            (args.output/'acceleration-progress.json').write_text(json.dumps(element_acceleration.diagnostics(),indent=2)+'\n')
        if i in {1,4,args.iterations} or defect<args.tolerance or (args.checkpoint_every and i%args.checkpoint_every==0):
            spectrum=synthesize()
            meta={**common,**spectrum.metadata,'converged':defect<args.tolerance,'iterations':i,'population_defect':defect}
            metrics=save(args.output/f'iteration-{i:03d}',spectrum,states,meta)
            ew={k:v['predicted_aperture_ew_mA'] for k,v in metrics['ciii_1175']['isolated_components'].items()}
            history.append(dict(iteration=i,defect=defect,ciii_equivalent_widths_mA=ew,
                element_defects=spectrum.metadata.get('element_population_defects'),
                worst=spectrum.metadata.get('worst_population_defect')))
            (args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
            print('  C III EWs:',ew,flush=True)
            print('  Element defects:',spectrum.metadata.get('element_population_defects'),flush=True)
            print('  Worst:',spectrum.metadata.get('worst_population_defect'),flush=True)
        return False
    print(f'Preparing {len(wave)} synthesis points; {len(grid)} rate points; {counts}',flush=True)
    import wd_spectra.hot_trace_metals as trace_impl
    ordinary_acceleration=trace_impl.population_update
    if args.elementwise_acceleration or args.ionwise_acceleration:
        from hot_trace_acceleration import ElementPopulationAcceleration
        groups=({f'{e}:{q}':n[q] for e,n in counts.items() for q in sorted(n)}
                if args.ionwise_acceleration else {e:sum(n.values()) for e,n in counts.items()})
        element_acceleration=ElementPopulationAcceleration(groups)
        trace_impl.population_update=element_acceleration
    try:
        result=solve_hot_trace_metals(host.atmosphere,background,abundances,wave,data=data,
            atomic_database=db,photoionization_database=photo,levels_per_charge=counts,
            population_wavelength=grid,
            initial_populations=None if args.cold_start else load_guesses(args.seed,elements=counts),
            **({} if args.profile_block_tolerance is None else dict(profile_block_tolerance=args.profile_block_tolerance)),
            photoionization_threshold_data=thresholds,collision_data=collisions,
            total_recombination=total_recombination,
            accelerated_lambda=args.accelerated_lambda,convergence_criterion=args.convergence_criterion,
            n_angle=n_angle,maximum_iterations=args.iterations,tolerance=tolerance,damping=args.damping,
            acceleration_depth=args.acceleration_depth,require_convergence=False,state_callback=callback)
    finally:
        trace_impl.population_update=ordinary_acceleration
    if identity(data)!=ident:raise RuntimeError('code/data changed during experiment')
    meta={**result.metadata,**common,'converged':result.converged,'iterations':result.iterations,
          'population_defect':result.population_defect}
    if element_acceleration is not None:meta['element_acceleration_diagnostics']=element_acceleration.diagnostics()
    save(args.output,result.spectrum,result.populations,meta)
    print(f'Saved {args.output}; converged={result.converged}',flush=True)
    if args.require_convergence and not result.converged:
        raise RuntimeError(f'Population convergence required but not achieved: {result.population_defect:g}')
    if args.energy_audit:
        from hot_trace_energy_balance import audit_energy_balance
        print('Measuring omitted metal feedback on atmospheric energy balance',flush=True)
        audit_energy_balance(host,model,db,photo,abundances,counts,thresholds,result.populations,
                             grid,background,host_background,args.output/'energy-audit')

if __name__=='__main__':main()
