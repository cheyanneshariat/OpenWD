"""Actual-material ML2 compatibility at EVERY outer backtracking trial.

The cheap proposal linearizes Rosseland opacity. Project its auxiliary
velocity with the actual EOS and actual opacity before evaluating radiation.
Scaling an already projected temperature step would lose this property.
Only temperatures change; the outer evaluator still computes actual ML2.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from .convective_mesh_prolongation import transport_profile


@contextmanager
def exact_trial_projection():
    original_adaptive=adaptive.solve_adaptive_lte_structure
    original_newton=adaptive.solve_trust_region_newton
    context={}
    def capture(seed,wave,**options):
        options['metadata']={**options.get('metadata',{}),'experimental_exact_ml2_trial_projection':True,
            'experimental_trial_projection_changes_flux':False}
        context['options']=options
        return original_adaptive(seed,wave,**options)
    def solve(initial,evaluate,**settings):
        first=evaluate(initial,False)
        if not first.payload['energy_balance_is_physical_flux']:
            return original_newton(initial,evaluate,**settings)
        if settings.get('trial_projector') is not None:
            raise ValueError('do not combine distinct trial projectors')
        builder=settings['step_builder'];pending={}
        def step(state,ev,jacobian,radius):
            direction=builder(state,ev,jacobian,radius)
            p=ev.payload;tr=p['convection_transport']
            target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
            velocity=ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],
                *(tr[k][1:] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')),target)
            proposed=p.get('diagnostic_augmented_proposed_velocity')
            if proposed is None:raise ValueError('exact trial projection requires an augmented velocity proposal')
            pending.update(direction=direction.copy(),velocity=velocity.copy(),
                proposed=proposed.copy(),mapping=p['log_temperature_from_state'])
            return direction
        def project(old,trial):
            direction=pending['direction'];denominator=float(direction@direction)
            if denominator==0:return trial
            fraction=float((trial-old)@direction/denominator)
            np.testing.assert_allclose(trial-old,fraction*direction,rtol=1e-8,atol=1e-12)
            velocity=pending['velocity']+fraction*(pending['proposed']-pending['velocity'])
            desired=np.r_[0.,np.maximum(velocity,0.)**3]
            options=context['options'];mapping=pending['mapping']
            candidate=options['with_temperature'](np.exp(mapping@trial))
            from . import check_cool_db_transport_seed as runner
            corrected,error=transport_profile(candidate,desired,runner,options,project_stable=True)
            projected=np.linalg.solve(mapping,np.log(corrected.temperature))
            print(f'ACTUAL-MATERIAL ML2 TRIAL: factor={fraction:.6g}; '
                f'maximum dlnT={np.max(abs(mapping@(projected-old))):.6g}; '
                f'actual convective-fraction defect={error:.3g}',flush=True)
            return projected
        settings.update(step_builder=step,trial_projector=project,allow_initial_convergence=False)
        return original_newton(initial,evaluate,**settings)
    with patch.object(adaptive,'solve_adaptive_lte_structure',capture), \
            patch.object(adaptive,'solve_trust_region_newton',solve):
        yield
