#!/usr/bin/env python3
"""Cold-start a hot DA/DAO host with an adaptive Planck-to-NLTE continuation.

Research experiment for the shared H/He solver. The public driver
(``solve_hot_nlte_atmosphere``) uses fixed NLTE fractions 0, 0.1, ..., 1 for
DAO hosts. On the 80-depth G191-B2B cold start every intermediate stage
converged in five Newton iterations with two coupled Jacobians (~25 min), so
the fixed schedule spends most of the run on easy stages. Here the fraction
step adapts: it doubles after a stage needs at most ``grow_iterations`` Newton
iterations, is kept up to ``keep_iterations`` and halved otherwise; a failed
attempt restarts from the last converged stage with half the step. Only the
path to full NLTE changes. The final stage is the same full-NLTE solve with
the same tolerances and certificate. Everything else is the instrumented
public cold start of ``run_hot_public_diagnostic.py`` (same arguments).
"""
from __future__ import annotations
from dataclasses import replace
import sys
import numpy as np
import wd_spectra.models.hot as hot_models
import run_hot_public_diagnostic as public

OPTIONS = dict(initial_step=0.25, minimum_step=0.025, grow_iterations=6, keep_iterations=10,
               intermediate_budget=12, final_budget=30)


def adaptive_solve_hot_nlte_atmosphere(seed, model, wavelength, *, maximum_iterations=120,
                                       flux_tolerance=3e-3, temperature_tolerance=3e-4,
                                       iteration_callback=None):
    from wd_spectra._hot_structure import solve
    if (isinstance(maximum_iterations, (bool, np.bool_)) or
            not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1):
        raise ValueError('maximum_iterations must be a positive integer')
    budget = min(maximum_iterations, model.population_maximum_iterations)
    stages, offset = [], 0
    state, fraction, step = None, 0.0, OPTIONS['initial_step']

    def callback(iteration, atmosphere, populations, diagnostics):
        if iteration_callback is not None:
            iteration_callback(offset + iteration, atmosphere, diagnostics)

    def attempt(target, limit):
        nonlocal offset
        result = solve(seed, model, wavelength, maximum_iterations=limit, populations=state,
                       nlte_fraction=float(target), iteration_callback=callback,
                       flux_tolerance=flux_tolerance, temperature_tolerance=temperature_tolerance)
        converged = bool(result.nonlinear_result.converged)
        stages.append(dict(nlte_fraction=float(target), solver_converged=converged,
                           iterations=result.nonlinear_result.iterations, step=float(target - fraction)))
        offset += result.nonlinear_result.iterations
        return result, converged

    # Planck (LTE-like) initialization exactly as in the public schedule.
    result, converged = attempt(0.0, budget)
    if converged:
        seed, state = result.atmosphere, result.population_state
    while converged and fraction < 1.0:
        target = min(1.0, fraction + step)
        final = target == 1.0
        result, ok = attempt(target, min(budget, OPTIONS['final_budget'] if final
                                         else OPTIONS['intermediate_budget']))
        if ok:
            seed, state, fraction = result.atmosphere, result.population_state, target
            used = stages[-1]['iterations']
            if used <= OPTIONS['grow_iterations']:
                step *= 2.0
            elif used > OPTIONS['keep_iterations']:
                step /= 2.0
            continue
        step /= 2.0
        if step < OPTIONS['minimum_step']:
            converged = False
    metadata = {**result.atmosphere.metadata, 'nlte_continuation_stages': stages,
                'nlte_continuation_schedule': dict(kind='adaptive', **OPTIONS),
                'radiative_equilibrium_iterations': offset,
                'nlte_population_tolerance': model.population_tolerance}
    return replace(result, atmosphere=replace(result.atmosphere, metadata=metadata))


if __name__ == '__main__':
    hot_models.solve_hot_nlte_atmosphere = adaptive_solve_hot_nlte_atmosphere
    public.main(experiment=dict(scope='research: adaptive NLTE continuation schedule',
                                script='run_hot_adaptive_continuation.py', options=OPTIONS))
