"""Planck-field reference for PG 1159, without a transfer iteration."""
from dataclasses import replace
import numpy as np
from .pg1159 import PG1159NLTEState, pg1159_composition_atmosphere
from .helium_nlte import solve_coupled_helium_statistical_equilibrium
from .light_metal_nlte import (
    solve_light_metal_ionization_nlte,
    solve_reduced_light_metal_levels_nlte,
)
from .spectrum import planck_lambda_angstrom


def planck_state(model, atmosphere, wavelength):
    a, metal = pg1159_composition_atmosphere(
        atmosphere, model.atomic_database, model.mass_fractions
    )
    h = model.helium_model
    helium = solve_coupled_helium_statistical_equilibrium(
        a,
        h.collision_data,
        maximum_helium_ii_level=h.maximum_helium_ii_level,
        helium_i_collision_data=h.helium_i_collision_data,
        hydrogenic_collision_model=h.hydrogenic_collision_model,
        neutral_collision_strength_scale=h.neutral_collision_strength_scale,
    )
    mean = planck_lambda_angstrom(wavelength[:, None], a.temperature[None, :])
    light = solve_light_metal_ionization_nlte(
        a,
        model.atomic_database,
        metal,
        model.photoionization_database,
        wavelength,
        mean,
        elements=model.nlte_metal_elements,
        photoionization_threshold_data=model.trace_photoionization_threshold_data,
    )
    levels = []
    for element, prefix in [("C", "carbon"), ("O", "oxygen")]:
        kwargs = {
            key: getattr(model, prefix + "_" + key)
            for key in (
                "photoionization_threshold_data",
                "formal_level_mapping",
                "formal_lte_parent_mapping",
                "continuum_parent_mapping",
                "lte_level_reservoir",
                "lte_bound_bound_couplings",
                "effective_dielectronic_couplings",
                "collision_data",
            )
        }
        levels.append(
            solve_reduced_light_metal_levels_nlte(
                a,
                getattr(model, prefix + "_population_atomic_database"),
                metal,
                model.photoionization_database,
                wavelength,
                mean,
                element,
                getattr(model, prefix + "_levels_per_charge"),
                **kwargs
            )
        )
    return (
        a,
        PG1159NLTEState(
            helium,
            metal,
            light,
            *levels,
            metadata={
                "metal_population_converged": True,
                "metal_population_iterations": 1,
                "metal_population_relative_change": 0.0,
                "undamped_population_defect": 0.0,
                "electron_closure_residual": 0.0,
                "population_reference": "local Planck field",
            }
        ),
    )


def gray_opacity_seed(model, atmosphere, *, iterations=4, wavelength_points=300):
    """Precondition a fresh continuum seed with its own He/C/O opacity.

    Only the provisional temperature is changed. The requested Teff, gravity,
    mixture and hydrostatic mass/pressure grid stay fixed. A sparse opacity
    quadrature suffices for this gray estimate; it never certifies equilibrium.
    The common solver subsequently evaluates the complete structural grid.
    """
    from dataclasses import replace
    import logging
    import numpy as np
    from ._rosseland import rosseland_mean_from_opacity_grid

    logger = logging.getLogger(__name__)
    wavelength = np.geomspace(10., 200000., wavelength_points)
    current = atmosphere
    for iteration in range(iterations):
        current, population = planck_state(model, current, wavelength)
        coefficients = model.transfer_coefficients(current, wavelength, population)
        rosseland = rosseland_mean_from_opacity_grid(
            wavelength, coefficients.total_extinction, current.temperature)
        surface = rosseland[0] * current.column_mass[0]
        tau = np.r_[surface, surface + np.cumsum(
            .5 * (rosseland[1:] + rosseland[:-1]) * np.diff(current.column_mass))]
        gray_temperature = current.effective_temperature * (.75 * (tau + 2./3.)) ** .25
        temperature = np.sqrt(current.temperature * gray_temperature)
        logger.info('PG1159 gray opacity initialization %d: maximum log-T change %.4g',
            iteration + 1, np.max(abs(np.log(temperature / current.temperature))))
        current = model.rebuild_atmosphere(current, temperature, None)
    return replace(current, metadata={**current.metadata,
        'pg1159_gray_opacity_initialization': {
            'iterations': iterations, 'wavelength_points': wavelength_points,
            'mass_and_pressure_grid_preserved': True,
            'equilibrium_certified': False,
        }})
