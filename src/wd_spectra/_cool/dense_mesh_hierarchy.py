"""Fresh coarse-to-fine nonlinear solves; every level uses the same physics.

No external checkpoint, failed-level substitution, or convergence inheritance.
Every grid solves its own formal-transfer and actual-gradient convection
equations. By default each must pass every gate. Explicitly requested incomplete
coarse stages may initialize finer grids; the final grid must always pass.
"""
from dataclasses import replace
import numpy as np
from scipy.interpolate import PchipInterpolator


def nested_indices(size, coarse_maximum=80):
    if size < 3 or coarse_maximum < 3:
        raise ValueError('mesh hierarchy requires at least three layers')
    levels = [np.arange(size)]
    while len(levels[-1]) > coarse_maximum:
        previous = levels[-1]
        levels.append(np.unique(np.r_[previous[::2], previous[-1]]))
    return levels[::-1]


def hierarchy_solve(solve, *arguments, coarse_maximum=80, before_level=None,
                    allow_incomplete_coarse=False, **options):
    pressure = np.asarray(options['initial_gas_pressure'])
    mass = np.asarray(options['initial_column_mass'])
    tau = np.asarray(options['initial_rosseland_optical_depth'])
    temperature = np.asarray(options['initial_temperature'])
    levels = nested_indices(len(pressure),coarse_maximum)
    callback = options.get('iteration_callback')
    total_iterations = 0
    histories = []
    result = None
    for level, indices in enumerate(levels):
        if before_level is not None:
            before_level(result)
        if result is not None:
            interpolator = PchipInterpolator(np.log(result.gas_pressure), np.log(result.temperature), extrapolate=False)
            values = np.exp(interpolator(np.log(pressure[indices])))
        else:
            values = temperature[indices]
        def report(iteration, atmosphere, diagnostics):
            nonlocal total_iterations
            total_iterations += 1
            if callback is not None:
                callback(total_iterations, atmosphere, {**diagnostics,
                    'mesh_hierarchy_level': level, 'mesh_hierarchy_layer_count': len(indices),
                    'mesh_level_iteration': iteration})
        current = {**options, 'n_depth': len(indices), 'initial_temperature': values,
            'initial_column_mass': mass[indices], 'initial_gas_pressure': pressure[indices],
            'initial_rosseland_optical_depth': tau[indices],
            'resume_supplied_structure_in_formal_flux_phase': level > 0,
            'iteration_callback': report}
        print(f'Fresh dense mesh hierarchy: level {level+1}/{len(levels)}, {len(indices)} layers; '
              + ('fresh transport initialization' if level == 0 else 'interpolated same-physics coarse solution; not yet converged'), flush=True)
        result = solve(*arguments, **current)
        diagnostics = result.metadata
        flux_tolerance = options.get('flux_tolerance', 3e-3)
        step_tolerance = options.get('temperature_tolerance', 3e-4)
        passed = bool(diagnostics['radiative_equilibrium_converged']
            and diagnostics['maximum_all_depth_total_flux_residual'] < flux_tolerance
            and diagnostics['maximum_relative_cell_energy_balance_residual'] < flux_tolerance
            and diagnostics['radiative_equilibrium_maximum_log_temperature_correction'] < step_tolerance)
        histories.append(dict(n_depth=len(indices), converged=passed,
            maximum_flux_residual=result.metadata['maximum_all_depth_total_flux_residual'],
            maximum_local_energy_residual=result.metadata['maximum_relative_cell_energy_balance_residual'],
            maximum_log_temperature_correction=diagnostics['radiative_equilibrium_maximum_log_temperature_correction']))
        if not passed and (not allow_incomplete_coarse or level==len(levels)-1):
            raise RuntimeError(f'Fresh mesh hierarchy stopped: {len(indices)}-layer solve failed; no coarse-result substitution')
        if not passed:
            print(f'INCOMPLETE coarse grid ({len(indices)} layers): initialize next level only; '
                  'not a converged result and not a substitute for the requested final grid',flush=True)
    return replace(result, metadata={**result.metadata, 'experimental_mesh_hierarchy': histories,
        'experimental_mesh_hierarchy_fresh_start': True})
