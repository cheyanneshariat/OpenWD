"""Choose physical rows for a verified same-calculation domain extension.

No files or neighboring models are loaded. The caller retains only its own
last completed domain; the new seed must be exactly its prescribed extension.
All new layers are solved with actual energy/flux and stationarity checks.
This is a research-only initialization policy, not a convergence shortcut.
"""
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from wd_spectra.nonlinear import RecoverableEvaluationError
from wd_spectra._convergence import equilibrium_certificate
from wd_spectra._domain import append_lower_domain


def physical_extension_options(previous,seed,options):
    if previous is None or seed.n_depth<=previous.n_depth:
        return options
    if (seed.effective_temperature!=previous.effective_temperature
            or seed.logg!=previous.logg):
        return options
    certificate=equilibrium_certificate(previous.metadata,
        flux_tolerance=options['flux_tolerance'],
        temperature_tolerance=options['temperature_tolerance'])
    if certificate['failures']!=['boundary_screening']:
        return options
    expected=append_lower_domain(previous)
    # A domain controller may deliberately solve and screen the first
    # canonical cell before appending the rest of the pressure-doubling
    # proposal. Accept only an exact nonempty prefix of that proposal.
    if seed.n_depth>expected['n_depth']:
        return options
    for field in ('temperature','gas_pressure','column_mass','rosseland_optical_depth'):
        if not np.array_equal(
            getattr(seed,field),expected['initial_'+field][:seed.n_depth]
        ):
            return options
    print(f'DQ internal domain {previous.n_depth}->{seed.n_depth}: '
        'solve physical energy rows; no external atmosphere',flush=True)
    return dict(options,use_convective_gradient_preconditioner=False,
        project_initial_convective_gradient=False,use_initial_bolometric_rescaling=False,
        initial_temperature_was_supplied=True,resume_supplied_structure_in_formal_flux_phase=False,
        metadata={**options.get('metadata',{}),
            'dq_domain_initialization':'physical energy after same-run interior qualification'})


def flux_balanced_extension(seed,old_depth,material):
    """Initialize newly physical lower nodes with the actual EOS and ML2 law.

    The terminal node of the completed domain was the transfer solver's LTE
    reservoir, not a material control volume.  Extending the domain turns
    that node into a real cell. Test its scalar radiative-diffusion + ML2
    balance with the actual material: preserve the old exact formal-boundary
    temperature when this candidate is strictly stable, but use the balanced
    value when the new cell must carry convection. Then balance every appended
    node inward, rebuilding the material coefficients as each temperature
    changes. Every formerly physical node is retained bit for bit. The
    Rosseland coefficient already includes refraction where enabled. This is
    a provisional starting profile, never a substitute for the formal field
    or a convergence certificate. There are no stored/external-model inputs.
    """
    from scipy.optimize import brentq
    if not 1<old_depth<seed.n_depth:
        raise ValueError('Domain initialization requires existing and appended nodes')
    temperatures=np.array(seed.temperature,dtype=float,copy=True)
    logp=np.log(seed.gas_pressure)
    target=STEFAN_BOLTZMANN*seed.effective_temperature**4
    records=[]
    first_new_cell=old_depth-1
    for i in range(first_new_cell,seed.n_depth):
        left=np.log(temperatures[i-1]);spacing=logp[i]-logp[i-1]
        if not spacing>0:raise ValueError('Ordered domain pressure required')
        def fluxes(delta):
            trial=temperatures.copy();trial[i]=np.exp(left+delta)
            current,fields=material.fields(np.log(trial))
            ad,loss,coefficient=material.assemble(current,fields)
            gradient=delta/spacing
            y=ml2_auxiliary_from_gradient(np.array([gradient]),ad[i:i+1],
                loss[i:i+1],coefficient[i:i+1],target)[0]
            convective=max(y,0.)**3
            tmid=np.sqrt(trial[i-1]*trial[i])
            pmid=np.sqrt(seed.gas_pressure[i-1]*seed.gas_pressure[i])
            kappa=np.sqrt(fields[1,i-1]*fields[1,i])
            radiative=16*STEFAN_BOLTZMANN*tmid**4*seed.gravity*gradient/(3*kappa*pmid*target)
            if not np.isfinite(radiative+convective):
                raise RecoverableEvaluationError('Nonfinite domain-initialization flux')
            return float(radiative),float(convective)
        def defect(delta):return sum(fluxes(delta))-1.
        high=max(float(np.log(temperatures[i])-left),.01)
        # A failure of the declared material domain is propagated, never
        # replaced by a different EOS or the previous extrapolated profile.
        for _ in range(16):
            if defect(high)>0:break
            high*=2
        else:raise RecoverableEvaluationError('Cannot bracket flux-balanced domain temperature')
        delta=brentq(defect,0.,high,xtol=2e-14,rtol=8*np.finfo(float).eps,maxiter=100)
        radiative,convective=fluxes(delta)
        balanced_temperature=float(np.exp(left+delta))
        balanced_radiative=float(radiative)
        balanced_convective=float(convective)
        if abs(radiative+convective-1)>1e-6:
            raise RecoverableEvaluationError('Domain seed failed its local flux-balance check')
        role = (
            'reactivated_terminal_reservoir'
            if i==first_new_cell else 'appended_node'
        )
        # A stable terminal reservoir already supplied the exact formal
        # boundary intensity of the completed shallower calculation. Replacing
        # it with a diffusion temperature can create a large heating impulse
        # when it becomes a cell, without adding the convective flux that
        # motivates reactivation. Preserve that value exactly on the stable
        # branch. If the actual ML2 closure is convective, the reservoir lacks
        # a physical cell flux and the balanced candidate is required.
        if i==first_new_cell and convective==0.:
            temperatures[i]=seed.temperature[i]
            radiative,convective=fluxes(np.log(temperatures[i])-left)
            role='preserved_terminal_reservoir_stable'
        else:
            temperatures[i]=np.exp(left+delta)
        records.append(dict(node=i,
            role=role,
            temperature=float(temperatures[i]),
            radiative_diffusion_over_target=radiative,convective_over_target=convective,
            balanced_candidate_temperature=balanced_temperature,
            balanced_candidate_radiative_diffusion_over_target=balanced_radiative,
            balanced_candidate_convective_over_target=balanced_convective))
        print(f'DQ domain seed node {i}: T={temperatures[i]:.3f} K, '
            f'Fdiff/Fstar={radiative:.6g}, Fconv/Fstar={convective:.6g}',flush=True)
    np.testing.assert_array_equal(
        temperatures[:first_new_cell],seed.temperature[:first_new_cell]
    )
    return temperatures,records


def initialize_physical_extension(previous,seed,options,material_factory):
    """Apply flux-balanced initialization only to a qualified same-run extension."""
    changed=physical_extension_options(previous,seed,options)
    if changed is options:return seed,options
    material=material_factory(*(options[k] for k in
        ('with_temperature','thermodynamics','rosseland_opacity','mixing_length_alpha')))
    temperature,records=flux_balanced_extension(seed,previous.n_depth,material)
    initialized=options['with_temperature'](temperature)
    for name in ('gas_pressure','column_mass','rosseland_optical_depth'):
        np.testing.assert_array_equal(getattr(initialized,name),getattr(seed,name))
    current_floor=min(record['node'] for record in records)
    previous_floor=previous.metadata.get(
        'dq_minimum_temperature_gradient_index', current_floor
    )
    if (not isinstance(previous_floor,(int,np.integer))
            or not 1<=previous_floor<initialized.n_depth):
        raise ValueError('Previous deep-gradient floor is invalid')
    return initialized,dict(changed,metadata={**changed['metadata'],
        'dq_domain_initialization':'same-run appended cells: reactivated terminal reservoir plus actual-material diffusion+ML2 seed; full formal solve required',
        'dq_domain_seed_fluxes':records,
        # Once a terminal reservoir becomes a physical cell, later domain
        # extensions must not allow that same interface to become an
        # inversion merely because it is no longer the newest appended cell.
        'dq_minimum_temperature_gradient_index':int(min(
            current_floor,previous_floor
        ))})
