"""Full-resolution D6 operations shared by frozen tests and their offline builder."""

from functools import lru_cache
import json
import time

import numpy as np

from wd_spectra import D6Config, compute_d6
from wd_spectra.d6 import _bulk_atmosphere, radiative_equilibrium_d6_atmosphere
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.d6 import _atomic_inputs, _N_DEPTH, _STRUCTURE_LINES

STEPS = 2
ARRAYS = ("temperature", "gas_pressure", "mass_density", "electron_density")
METRICS = (
    "maximum_all_depth_total_flux_residual",
    "maximum_relative_cell_energy_balance_residual",
    "maximum_log_temperature_correction",
    "scattering_source_maximum_relative_residual",
)


@lru_cache(maxsize=1)
def atomic_inputs():
    return _atomic_inputs(ModelData.default(), tuple(D6Config().abundances))


def solve_structure(column_mass, optical_depth, temperature, callback=None):
    """Recompute the public standard material closures, with an explicit warm seed.

    The low-level API explicitly selects the conservative warm-start phase.
    Frozen test inputs do not claim a matching current-code checkpoint hash.
    """
    config = D6Config()
    database, photoionization, topbase = atomic_inputs()
    resolution = numerical_resolution(config.quality)
    return radiative_equilibrium_d6_atmosphere(
        config.effective_temperature, config.logg, database, photoionization,
        config.abundances, reference_element=config.reference_element,
        n_depth=_N_DEPTH[config.quality], max_iterations=resolution.maximum_iterations,
        maximum_metal_lines=_STRUCTURE_LINES[config.quality],
        minimum_metal_oscillator_strength=1.0e-4,
        include_rydberg_dissolution=config.include_rydberg_dissolution,
        include_metal_series_pseudocontinuum=config.include_metal_series_pseudocontinuum,
        metal_series_pseudocontinuum_elements=config.metal_series_pseudocontinuum_elements,
        topbase_photoionization_database=topbase,
        include_linear_stark_quasistatic=config.include_linear_stark_quasistatic,
        linear_stark_profile=config.linear_stark_profile,
        microturbulent_velocity_kms=config.microturbulent_velocity_kms,
        include_oxygen_i_series_stark=config.include_oxygen_i_series_stark,
        profile_edge_optical_depth=config.line_profile_edge_optical_depth,
        n_angle=min(resolution.n_angle, 3), mixing_length_alpha=config.mixing_length_alpha,
        initial_temperature=temperature, initial_column_mass=column_mass,
        initial_rosseland_optical_depth=optical_depth,
        resume_supplied_structure_in_formal_flux_phase=True,
        iteration_callback=callback,
    )


class TrajectoryComplete(Exception):
    """Stop after the requested measured updates, without claiming convergence."""


def trajectory(column_mass, optical_depth, temperature):
    records = []
    started = time.monotonic()

    def record(iteration, atmosphere, diagnostic):
        records.append(dict(
            iteration=iteration, phase=diagnostic["solver_phase"],
            metrics=np.array([diagnostic[key] for key in METRICS]),
            **{key: getattr(atmosphere, key).copy() for key in ARRAYS},
        ))
        print(json.dumps(dict(iteration=iteration, phase=diagnostic["solver_phase"],
                              elapsed_seconds=time.monotonic() - started,
                              **{key: diagnostic[key] for key in METRICS})), flush=True)
        if len(records) == STEPS:
            raise TrajectoryComplete

    try:
        solve_structure(column_mass, optical_depth, temperature, record)
    except TrajectoryComplete:
        return records
    raise AssertionError(f"D6 returned before {STEPS} measured trajectory steps")


def fixed_spectrum(column_mass, optical_depth, temperature, wavelength):
    config = D6Config()
    database, _, _ = atomic_inputs()
    atmosphere, _ = _bulk_atmosphere(
        config.effective_temperature, config.logg, optical_depth, column_mass,
        temperature, database, config.abundances,
        reference_element=config.reference_element,
        metadata={"model": "frozen-d6-regression-input",
                  "radiative_equilibrium_converged": False},
    )
    return compute_d6(config, wavelength, initial_atmosphere=atmosphere,
                      relax_atmosphere=False).spectrum
