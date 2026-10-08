"""Cold DQ: use flux-balanced convection as a preconditioner, not an end point.

The phase transition uses the solver's existing ALL-depth physical flux
tolerance, never Teff or an object-specific iteration count. The second
phase belongs to the same cold solve, on the same mass/pressure grid; no
external atmosphere is loaded. Existing convective trial correction is
enabled in both phases. All final physical convergence gates are unchanged.
"""
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import wd_spectra.adaptive_structure as adaptive


class FluxBalancedHandoff(Exception):
    def __init__(self,iteration,state,reason='all-depth flux gate passed'):
        self.iteration,self.state,self.reason=iteration,state,reason


def gated_solve(original,seed,wave,*,finish_in_energy_equations=False,**options):
    callback=options.get('iteration_callback')
    latest_iteration=0
    def monitor(i,state,diagnostics):
        nonlocal latest_iteration
        latest_iteration=i
        if callback is not None:callback(i,state,diagnostics)
        if (diagnostics.get('solver_phase')=='convective-gradient-preconditioner'
            and diagnostics['maximum_all_depth_total_flux_residual']<=options['flux_tolerance']):
            raise FluxBalancedHandoff(i,state)
    options=dict(options,use_convective_trial_correction=True,iteration_callback=monitor)
    nonlinear=adaptive.solve_trust_region_newton
    def finish_conditioning(initial,evaluate,**settings):
        first=evaluate(initial,False)
        if (first.payload.get('energy_balance_is_physical_flux')
                and not first.payload.get('local_energy_equations_enforced')):
            # The shared driver has already terminated its initializer. Do
            # not insert its flux-only temperature-coordinate solve before
            # the coupled physical-energy method selected by this experiment.
            raise FluxBalancedHandoff(latest_iteration,first.payload['atmosphere'],
                'preconditioner terminated; enter coupled physical energy')
        return nonlinear(initial,evaluate,**settings)
    try:
        with (patch.object(adaptive,'solve_trust_region_newton',finish_conditioning)
              if finish_in_energy_equations else nullcontext()):
            return original(seed,wave,**options)
    except FluxBalancedHandoff as handoff:
        remaining=options['max_iterations']-handoff.iteration
        if remaining<1:
            raise RuntimeError('Iteration budget exhausted before physical-energy completion')
        if callback is not None:
            options['iteration_callback']=lambda i,a,d:callback(i+handoff.iteration,a,d)
        else:options['iteration_callback']=None
        options.update(use_convective_gradient_preconditioner=False,
            project_initial_convective_gradient=False,use_initial_bolometric_rescaling=False,
            initial_temperature_was_supplied=True,resume_supplied_structure_in_formal_flux_phase=False,
            max_iterations=remaining)
        print(f'DQ physical-energy handoff at iteration {handoff.iteration}: {handoff.reason}',flush=True)
        result=original(handoff.state,wave,**options)
        return replace(result,metadata={**result.metadata,
            'radiative_equilibrium_iterations':handoff.iteration+int(result.metadata.get('radiative_equilibrium_iterations',0)),
            'dq_internal_flux_gate_iteration':handoff.iteration,
            'dq_physical_energy_handoff_reason':handoff.reason})




