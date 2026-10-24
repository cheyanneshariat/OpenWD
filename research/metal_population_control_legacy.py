"""Recover pre-normalization opacity populations for immutable controls.

Research/test scope only. This restores main at 6e66459 (PR #38): it keeps
that release's Rydberg-only neutral occupation and corrected RAP reader.
It does not change ionization, partition functions, line profiles or grids.
The legacy equations are deliberately inconsistent with occupation-weighted
partitions; current public models must never select this scope.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
from functools import wraps
from inspect import signature
import sys
from unittest.mock import patch

import numpy as np

from wd_spectra import d6, metals
from wd_spectra.constants import BOLTZMANN


def _ideal_level_population(ion, level, state, temperature_kelvin):
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    return (state.ion_number_density[ion.element][ion.charge]
            * level.statistical_weight
            * np.exp(-level.energy_wavenumber * metals.WAVENUMBER_TO_ERG
                     / (BOLTZMANN * temperature))
            / state.partition_function[(ion.element, ion.charge)])


def _ideal_topbase_population(section, state, temperature_kelvin, **_kwargs):
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    return (state.ion_number_density[section.element][section.charge]
            * section.statistical_weight
            * np.exp(-section.excitation_energy_ev * d6.EV_TO_ERG
                     / (BOLTZMANN * temperature))
            / state.partition_function[(section.element, section.charge)])


def _previous_lines(function):
    parameters = signature(function)

    @wraps(function)
    def previous(*args, **kwargs):
        values = parameters.bind(*args, **kwargs)
        values.apply_defaults()
        options = values.arguments
        state = options['metal_state']
        database = options['atomic_database']
        include = options['include_rydberg_dissolution'] or (
            state.metal_level_dissolution and state.composition_mode == 'trace')
        selected = options['rydberg_dissolution_elements']
        selected = (None if selected is None else
                    frozenset(metals._canonical_element(item) for item in selected))
        if options['rydberg_neutral_perturbers']:
            neutrals, radii = metals.bulk_metal_neutral_perturber_data(database, state)
        else:
            neutrals, radii = (state.neutral_perturber_number_density,
                               state.neutral_perturber_radius_cm)

        def lower_population(ion, level, material, temperature):
            population = _ideal_level_population(ion, level, material, temperature)
            if (material.metal_level_dissolution and include
                    and (selected is None or ion.element in selected)
                    and (ion.ionization_energy_ev is None or level.energy_wavenumber
                         < ion.ionization_energy_ev * metals.EV_TO_WAVENUMBER)):
                population = population * metals.metal_rydberg_level_occupation_probability(
                    ion, level, metals.metal_ionic_microfield_perturber_density(material),
                    temperature, correlated_microfields=options['rydberg_correlated_microfields'],
                    neutral_perturber_number_density=neutrals,
                    neutral_perturber_radius_cm=radii)
            return population

        with patch.object(metals, 'metal_lte_level_population', lower_population):
            return function(*args, **kwargs)
    return previous


def _previous_pseudocontinuum(function):
    @wraps(function)
    def previous(atmosphere, wavelength_angstrom, metal_state, database, **kwargs):
        # Before normalization the channel threshold supplied n*_lower;
        # the lower population itself was ideal g exp(-E/kT)/Q. This flag
        # selects both old branches without changing the material arrays.
        return function(atmosphere, wavelength_angstrom,
                        replace(metal_state, metal_level_dissolution=False),
                        database, **kwargs)
    return previous


def _previous_native_continuum(function):
    @wraps(function)
    def previous(*args, **kwargs):
        with patch.object(metals, 'metal_lte_level_population', _ideal_level_population):
            return function(*args, **kwargs)
    return previous


@contextmanager
def previous_metal_opacity_populations(*, components=('lines', 'verner', 'oxygen', 'topbase')):
    """Restore old opacity at every imported binding; subsets support audits."""
    allowed = {'lines', 'verner', 'oxygen', 'topbase'}
    if set(components) - allowed:
        raise ValueError('unknown historical population component')
    swaps = {}
    if 'lines' in components:
        swaps[metals.metal_line_mass_absorption_coefficient] = _previous_lines(
            metals.metal_line_mass_absorption_coefficient)
    for name, function in (
        ('verner', metals.metal_bound_free_mass_absorption_coefficient),
        ('oxygen', metals.oxygen_i_dissolved_series_mass_absorption_coefficient),
    ):
        if name in components:
            swaps[function] = _previous_native_continuum(function)
    if 'topbase' in components:
        swaps[d6.topbase_lte_level_population] = _ideal_topbase_population
        swaps[d6.topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient] = (
            _previous_pseudocontinuum(
                d6.topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient))
    with ExitStack() as stack:
        for module in tuple(sys.modules.values()):
            if module is None or not getattr(module, '__name__', '').startswith('wd_spectra'):
                continue
            for name, value in tuple(vars(module).items()):
                for current, previous in swaps.items():
                    if value is current:
                        stack.enter_context(patch.object(module, name, previous))
        yield
