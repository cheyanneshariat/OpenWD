"""Hot homogeneous He/C/O atmospheres for PG 1159 stars.

The first implementation couples a bulk He/C/O fixed-pressure EOS to the
existing term-resolved helium NLTE atom.  Carbon and oxygen ion stages are
solved in statistical equilibrium, and a reduced explicit C III--V atom
provides the diagnostic optical carbon-line source functions.  The common
interface keeps their opacity inside the NLTE atmosphere loop so this model
can be progressively enlarged without another atmosphere driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .constants import BOLTZMANN
from .helium_nlte import (
    CoupledHeliumNLTEState,
    _prepare_helium_line_transfer_problem,
    default_helium_ii_continuum_wavelength,
    default_neutral_helium_continuum_wavelength,
    helium_ii_shell_transition,
    remap_coupled_helium_state,
    solve_coupled_helium_statistical_equilibrium,
)
from .light_metal_nlte import (
    BTMQuadrupoleAngularMomentumMixingCollision,
    ChiantiTermCollisionStrength,
    ConstantEffectiveCollisionStrength,
    LightMetalNLTEState,
    ReducedLightMetalLevelState,
    PSM20AngularMomentumMixingCollision,
    TmadEffectiveDielectronicCoupling,
    TmadLTEBoundBoundCoupling,
    TlustyPhotoionizationThresholdData,
    default_light_metal_ionization_wavelength,
    hot_metal_line_nlte_coefficients,
    light_metal_bound_free_nlte_coefficients,
    light_metal_free_free_charge_kernel,
    light_metal_free_free_mass_absorption_coefficient,
    reduced_light_metal_wavelength,
    solve_reduced_light_metal_levels_nlte,
    solve_light_metal_ionization_nlte,
)
from .metals import (
    AtomicDatabase,
    MetalLTEState,
    VernerPhotoionizationDatabase,
    atmosphere_with_metal_electrons,
    helium_metal_lte_state_from_mass_fractions,
    metal_bound_free_mass_absorption_coefficient,
    metal_line_mass_absorption_coefficient,
)
from ._pg1159_helium import CoupledHeliumNLTEModel, PG1159HeliumTransferCache
from .nlte_core import NLTETransferCoefficients, NonphysicalPopulationError
from .nlte import _profile_averaged_mean_intensity_nu
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import Backend, emergent_flux, radiation_field, RadiationField
from .spectrum import Spectrum, planck_lambda_angstrom


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
_MAX_FINITE_DEPARTURE = 1.0e100


PG1159_035_MASS_FRACTIONS = MappingProxyType(
    {"He": 0.33, "C": 0.50, "O": 0.17}
)


def _finite_departure_ratio(
    population: NDArray[np.float64],
    lte_population: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a finite departure ratio when an LTE tail underflows."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = population / np.maximum(lte_population, np.finfo(np.float64).tiny)
    return np.asarray(np.nan_to_num(
        value,
        nan=0.0,
        posinf=_MAX_FINITE_DEPARTURE,
        neginf=0.0,
    ).clip(0.0, _MAX_FINITE_DEPARTURE))


def _interpolate_mean_intensity(
    source_wavelength: FloatArray,
    source_mean_intensity: FloatArray,
    target_wavelength: FloatArray,
) -> FloatArray:
    """Interpolate a depth-dependent radiation field in wavelength."""

    return np.ascontiguousarray(np.column_stack([
        np.interp(target_wavelength, source_wavelength, source_mean_intensity[:, depth])
        for depth in range(source_mean_intensity.shape[1])
    ]))


def _population_transfer_extinction(true_absorption, emissivity, scattering):
    """Validate a trial SE radiation field and return its total extinction."""
    total = true_absorption + scattering
    if (
        np.any(~np.isfinite(total))
        or np.any(total <= 0)
        or np.any(~np.isfinite(emissivity))
        or np.any(emissivity < 0)
        or np.any(~np.isfinite(scattering))
        or np.any(scattering < 0)
        or np.any(true_absorption[:, -1] <= 0)
    ):
        raise NonphysicalPopulationError(
            "inadmissible He/C/O population-transfer coefficients"
        )
    return total


def _maximum_coupled_helium_population_change(
    previous: CoupledHeliumNLTEState,
    current: CoupledHeliumNLTEState,
    *,
    relative_floor: float = 1.0e-12,
) -> float:
    reference_total = (
        np.sum(current.lte_neutral_population_density, axis=1)
        + np.sum(current.lte_singly_ionized_population_density, axis=1)
        + current.lte_doubly_ionized_he_density
    )
    floor = relative_floor * reference_total
    maximum = 0.0
    for old, new in (
        (previous.neutral_population_density, current.neutral_population_density),
        (
            previous.singly_ionized_population_density,
            current.singly_ionized_population_density,
        ),
        (previous.doubly_ionized_he_density, current.doubly_ionized_he_density),
    ):
        local_floor = floor[..., np.newaxis] if new.ndim == 2 else floor
        scale = np.maximum(np.maximum(np.abs(old), np.abs(new)), local_floor)
        maximum = max(
            maximum,
            float(np.max(np.abs(new - old) / np.maximum(scale, 1.0e-300))),
        )
    return maximum


def _coupled_helium_population_change_diagnostic(
    previous: CoupledHeliumNLTEState,
    current: CoupledHeliumNLTEState,
    *,
    relative_floor: float = 1.0e-12,
) -> Mapping[str, object]:
    """Describe the He state responsible for the maximum convergence norm."""

    reference_total = (
        np.sum(current.lte_neutral_population_density, axis=1)
        + np.sum(current.lte_singly_ionized_population_density, axis=1)
        + current.lte_doubly_ionized_he_density
    )
    best: dict[str, object] = {"relative_change": 0.0}

    def inspect_levels(
        ion: str,
        old: NDArray[np.float64],
        new: NDArray[np.float64],
    ) -> None:
        floor = relative_floor * reference_total[:, np.newaxis]
        scale = np.maximum(np.maximum(np.abs(old), np.abs(new)), floor)
        relative = np.abs(new - old) / np.maximum(scale, 1.0e-300)
        depth, level = np.unravel_index(int(np.argmax(relative)), relative.shape)
        value = float(relative[depth, level])
        if value > float(best["relative_change"]):
            best.update({
                "relative_change": value,
                "ion": ion,
                "level_index": int(level),
                "depth_index": int(depth),
                "current_fraction": float(new[depth, level] / reference_total[depth]),
                "previous_fraction": float(old[depth, level] / reference_total[depth]),
            })

    inspect_levels(
        "He I", previous.neutral_population_density, current.neutral_population_density
    )
    inspect_levels(
        "He II",
        previous.singly_ionized_population_density,
        current.singly_ionized_population_density,
    )
    old = previous.doubly_ionized_he_density
    new = current.doubly_ionized_he_density
    scale = np.maximum(
        np.maximum(np.abs(old), np.abs(new)), relative_floor * reference_total
    )
    relative = np.abs(new - old) / np.maximum(scale, 1.0e-300)
    depth = int(np.argmax(relative))
    value = float(relative[depth])
    if value > float(best["relative_change"]):
        best.update({
            "relative_change": value,
            "ion": "He III",
            "level_index": 0,
            "depth_index": depth,
            "current_fraction": float(new[depth] / reference_total[depth]),
            "previous_fraction": float(old[depth] / reference_total[depth]),
        })
    return MappingProxyType(best)


def _damp_coupled_helium_state(
    previous: CoupledHeliumNLTEState,
    current: CoupledHeliumNLTEState,
    damping: float,
) -> CoupledHeliumNLTEState:
    """Linearly damp coupled-He populations while preserving particle count."""

    if damping >= 1.0:
        return current
    neutral = (
        (1.0 - damping) * previous.neutral_population_density
        + damping * current.neutral_population_density
    )
    ion = (
        (1.0 - damping) * previous.singly_ionized_population_density
        + damping * current.singly_ionized_population_density
    )
    continuum = (
        (1.0 - damping) * previous.doubly_ionized_he_density
        + damping * current.doubly_ionized_he_density
    )
    return replace(
        current,
        neutral_population_density=np.ascontiguousarray(neutral),
        singly_ionized_population_density=np.ascontiguousarray(ion),
        doubly_ionized_he_density=np.ascontiguousarray(continuum),
        neutral_departure_coefficient=np.ascontiguousarray(
            neutral / np.maximum(current.lte_neutral_population_density, 1.0e-300)
        ),
        singly_ionized_departure_coefficient=np.ascontiguousarray(
            ion / np.maximum(current.lte_singly_ionized_population_density, 1.0e-300)
        ),
        doubly_ionized_departure_coefficient=np.ascontiguousarray(
            continuum / np.maximum(current.lte_doubly_ionized_he_density, 1.0e-300)
        ),
    )


def _coupled_population_update(x_vector, g_vector, previous_helium, next_helium, history, depth,
                               *, metal_relative_floors=None, helium_relative_floor=1e-12):
    """Accelerate the common radiation fixed point for all three elements."""
    from ._pg1159_acceleration import population_update as _anderson_log_population_update
    def helium_logs(state):
        return np.log(np.maximum(np.column_stack((state.neutral_departure_coefficient,
            state.singly_ionized_departure_coefficient,
            state.doubly_ionized_departure_coefficient)), 1e-300))
    old_helium = helium_logs(previous_helium)
    raw_helium = helium_logs(next_helium)
    weights = None
    if metal_relative_floors is not None:
        fraction = np.exp(np.maximum(x_vector, g_vector))
        metal_weights = fraction / np.maximum(fraction, metal_relative_floors)
        def helium_fraction(state):
            population = np.column_stack((state.neutral_population_density,
                state.singly_ionized_population_density, state.doubly_ionized_he_density))
            return population / np.maximum(population.sum(axis=1, keepdims=True), 1e-300)
        fraction = np.maximum(helium_fraction(previous_helium), helium_fraction(next_helium))
        helium_weights = fraction / np.maximum(fraction, helium_relative_floor)
        weights = np.r_[metal_weights, helium_weights.ravel()]
    proposed, _ = _anderson_log_population_update(
        np.r_[x_vector, old_helium.ravel()], np.r_[g_vector, raw_helium.ravel()], history,
        depth=depth, mixing=.65, maximum_step=.75, residual_weights=weights)
    if proposed is None:
        return None
    return proposed[:x_vector.size], _helium_from_log_departures(next_helium,
        proposed[x_vector.size:].reshape(old_helium.shape))


def _helium_from_log_departures(current, proposed):
    """Decode positive helium departures with exact particle conservation."""
    from .helium_nlte import _replace_coupled_departures
    neutral_end = current.neutral_population_density.shape[1]
    ion_end = neutral_end + current.singly_ionized_population_density.shape[1]
    neutral = np.exp(proposed[:, :neutral_end])
    ion = np.exp(proposed[:, neutral_end:ion_end])
    continuum = np.exp(proposed[:, ion_end])
    total = (current.lte_neutral_population_density.sum(axis=1)
        + current.lte_singly_ionized_population_density.sum(axis=1)
        + current.lte_doubly_ionized_he_density)
    mixed = ((current.lte_neutral_population_density * neutral).sum(axis=1)
        + (current.lte_singly_ionized_population_density * ion).sum(axis=1)
        + current.lte_doubly_ionized_he_density * continuum)
    normalization = total / mixed
    return _replace_coupled_departures(current, neutral * normalization[:, None],
        ion * normalization[:, None], continuum * normalization,
        iterations=current.iterations, converged=False,
        maximum_change=current.maximum_relative_population_change, metadata=current.metadata)


def _formal_departures_from_population(
    state: ReducedLightMetalLevelState,
    population_departure: Mapping[tuple[str, int, int], NDArray[np.float64]],
) -> Mapping[tuple[str, int, int], NDArray[np.float64]]:
    """Expand compact model-atom term departures onto synthesis levels."""

    if state.formal_level_mapping is None:
        return MappingProxyType(dict(population_departure))
    departure: dict[tuple[str, int, int], NDArray[np.float64]] = {}
    for population_key, value in population_departure.items():
        for formal_key in state.formal_level_mapping.get(population_key, ()):
            departure[formal_key] = value
    if state.formal_lte_parent_mapping is not None:
        for formal_key, parent_key in state.formal_lte_parent_mapping.items():
            if parent_key in population_departure:
                departure[formal_key] = population_departure[parent_key]
    return MappingProxyType(departure)


def _formal_population_transition_keys(
    population_atomic_database: AtomicDatabase,
    formal_atomic_database: AtomicDatabase,
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ],
    element: str,
    levels_per_charge: Mapping[int, int],
) -> frozenset[tuple[str, int, int, int]]:
    """Identify formal components connecting retained population terms."""

    symbol = element.strip().capitalize()
    represented_formal_levels: dict[int, set[int]] = {}
    for charge, count in levels_per_charge.items():
        population_ion = population_atomic_database.ions[(symbol, charge)]
        selected = sorted(
            population_ion.levels, key=lambda level: level.energy_wavenumber
        )[: int(count)]
        represented_formal_levels[charge] = {
            formal_key[2]
            for level in selected
            for formal_key in formal_level_mapping.get(
                (symbol, charge, level.index), ()
            )
        }
    return frozenset(
        (symbol, charge, line.lower_index, line.upper_index)
        for charge, represented in represented_formal_levels.items()
        for line in formal_atomic_database.ions[(symbol, charge)].transitions
        if line.lower_index in represented
        and line.upper_index in represented
        and line.einstein_a > 0.0
    )


def _formal_spectrum_transition_keys(
    formal_atomic_database: AtomicDatabase,
    elements: Iterable[str],
    population_atoms: Mapping[str, AtomicDatabase | None],
    formal_level_mappings: Mapping[
        str,
        Mapping[tuple[str, int, int], tuple[tuple[str, int, int], ...]] | None,
    ],
    levels_per_charge: Mapping[str, Mapping[int, int]],
    formal_lte_parent_mappings: Mapping[
        str, Mapping[tuple[str, int, int], tuple[str, int, int]] | None
    ] | None = None,
) -> frozenset[tuple[str, int, int, int]]:
    """Keep only formal lines whose two endpoints exist in the rate atom.

    TMAP's formal atom is generally larger than its statistical-equilibrium
    atom.  Lines attached to an omitted level must not silently re-enter the
    final spectrum with an ion-mean departure coefficient: doing so creates
    strong high-series features that were absent from the population solve.
    Elements without a configured population/formal mapping retain all of
    their lines, which preserves the ion-only trace-element treatment.
    """

    allowed: set[tuple[str, int, int, int]] = set()
    for raw_element in elements:
        element = raw_element.strip().capitalize()
        population_atom = population_atoms.get(element)
        mapping = formal_level_mappings.get(element)
        if population_atom is not None and mapping is not None:
            selected_population_levels = {
                (element, charge, level.index)
                for charge, count in levels_per_charge.get(element, {}).items()
                for level in sorted(
                    population_atom.ions[(element, charge)].levels,
                    key=lambda item: item.energy_wavenumber,
                )[: int(count)]
            }
            represented_formal_levels = {
                formal_key
                for population_key in selected_population_levels
                for formal_key in mapping.get(population_key, ())
            }
            lte_parent_mapping = (
                None
                if formal_lte_parent_mappings is None
                else formal_lte_parent_mappings.get(element)
            )
            if lte_parent_mapping is not None:
                represented_formal_levels.update(
                    formal_key
                    for formal_key, parent_key in lte_parent_mapping.items()
                    if parent_key in selected_population_levels
                )
            for (candidate, _), ion in formal_atomic_database.ions.items():
                if candidate != element:
                    continue
                allowed.update(
                    (element, ion.charge, line.lower_index, line.upper_index)
                    for line in ion.transitions
                    if (element, ion.charge, line.lower_index)
                    in represented_formal_levels
                    and (element, ion.charge, line.upper_index)
                    in represented_formal_levels
                    and line.einstein_a > 0.0
                )
            continue
        for (candidate, _), ion in formal_atomic_database.ions.items():
            if candidate != element:
                continue
            allowed.update(
                (element, ion.charge, line.lower_index, line.upper_index)
                for line in ion.transitions
            )
    return frozenset(allowed)


def _add_missing_lte_ion_ladders(
    state: LightMetalNLTEState,
    metal_state: MetalLTEState,
    elements: tuple[str, ...],
) -> LightMetalNLTEState:
    """Extend an old checkpoint without discarding its converged ladders."""

    population = dict(state.ion_number_density)
    lte_population = dict(state.lte_ion_number_density)
    departure = dict(state.ion_departure_coefficient)
    added = []
    for element in elements:
        if element in population:
            continue
        local_lte = np.asarray(metal_state.ion_number_density[element])
        population[element] = local_lte.copy()
        lte_population[element] = local_lte.copy()
        for charge in range(local_lte.shape[0]):
            departure[(element, charge)] = np.ones(local_lte.shape[1])
        added.append(element)
    if not added:
        return state
    return replace(
        state,
        ion_number_density=MappingProxyType(population),
        lte_ion_number_density=MappingProxyType(lte_population),
        ion_departure_coefficient=MappingProxyType(departure),
        metadata={
            **state.metadata,
            "lte_initialized_checkpoint_elements": tuple(added),
        },
    )


def _replace_light_metal_elements(
    state: LightMetalNLTEState,
    source: LightMetalNLTEState,
    elements: tuple[str, ...],
) -> LightMetalNLTEState:
    """Copy passenger ion ladders from a direct SE solution."""

    population = dict(state.ion_number_density)
    lte_population = dict(state.lte_ion_number_density)
    departure = dict(state.ion_departure_coefficient)
    for element in elements:
        population[element] = source.ion_number_density[element]
        lte_population[element] = source.lte_ion_number_density[element]
        for charge in range(population[element].shape[0]):
            departure[(element, charge)] = source.ion_departure_coefficient[
                (element, charge)
            ]
    return replace(
        state,
        ion_number_density=MappingProxyType(population),
        lte_ion_number_density=MappingProxyType(lte_population),
        ion_departure_coefficient=MappingProxyType(departure),
    )


def _damp_reduced_level_state(
    previous: ReducedLightMetalLevelState | None,
    current: ReducedLightMetalLevelState,
    damping: float,
) -> ReducedLightMetalLevelState:
    if previous is None or damping >= 1.0:
        return current
    if previous.level_key == current.level_key:
        previous_population = previous.population_density
    else:
        previous_by_key = {
            key: previous.population_density[index]
            for index, key in enumerate(previous.level_key)
        }
        previous_population = np.asarray([
            previous_by_key.get(key, current.lte_population_density[index])
            for index, key in enumerate(current.level_key)
        ])
    population = (
        (1.0 - damping) * previous_population
        + damping * current.population_density
    )
    population_departures = {
        key: _finite_departure_ratio(
            population[index], current.lte_population_density[index]
        )
        for index, key in enumerate(current.level_key)
    }
    metadata = dict(current.metadata)
    metadata["population_damping"] = damping
    return replace(
        current,
        population_density=np.asarray(population),
        population_level_departure_coefficient=MappingProxyType(
            population_departures
        ),
        level_departure_coefficient=_formal_departures_from_population(
            current, population_departures
        ),
        metadata=metadata,
    )


def _align_reduced_level_state(
    previous: ReducedLightMetalLevelState | None,
    current_template: ReducedLightMetalLevelState | None,
) -> ReducedLightMetalLevelState | None:
    """Align a warm-start state after terms have been promoted.

    Shared terms retain their populations; newly promoted terms start from
    the current LTE reference.  The returned state uses the current model's
    mappings and key order, so vector damping/acceleration has equal shapes.
    """

    if current_template is None:
        return None
    if previous is None:
        # A checkpoint may predate an element that has just been enabled
        # (the controlled C -> C+O bridge is the canonical example).  There
        # is no old vector block to damp against, so adopt the first physical
        # statistical-equilibrium solution as that block's warm start.  The
        # remaining, already-present species are still mixed normally.
        return replace(
            current_template,
            metadata={
                **current_template.metadata,
                "warm_start_alignment": (
                    "new element initialized from first SE solution"
                ),
            },
        )
    if previous.level_key == current_template.level_key:
        return previous
    previous_by_key = {
        key: previous.population_density[index]
        for index, key in enumerate(previous.level_key)
    }
    population = np.asarray([
        previous_by_key.get(key, current_template.lte_population_density[index])
        for index, key in enumerate(current_template.level_key)
    ])
    population_departures = MappingProxyType({
        key: _finite_departure_ratio(
            population[index], current_template.lte_population_density[index]
        )
        for index, key in enumerate(current_template.level_key)
    })
    return replace(
        current_template,
        population_density=population,
        population_level_departure_coefficient=population_departures,
        level_departure_coefficient=_formal_departures_from_population(
            current_template, population_departures
        ),
        metadata={
            **current_template.metadata,
            "warm_start_alignment": "shared terms retained; promoted terms initialized in LTE",
        },
    )


def _population_relative_change(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    total_by_depth: NDArray[np.float64],
    *,
    relative_floor: float = 1.0e-12,
) -> float:
    """Return a stable fractional change without magnifying empty levels."""

    floor = relative_floor * total_by_depth[np.newaxis, :]
    scale = np.maximum(np.maximum(np.abs(previous), np.abs(current)), floor)
    return float(np.max(np.abs(current - previous) / np.maximum(scale, 1.0e-300)))


def _maximum_population_state_change(
    previous_light: LightMetalNLTEState | None,
    previous_carbon: ReducedLightMetalLevelState | None,
    previous_oxygen: ReducedLightMetalLevelState | None,
    current_light: LightMetalNLTEState | None,
    current_carbon: ReducedLightMetalLevelState | None,
    current_oxygen: ReducedLightMetalLevelState | None,
    metal_state: MetalLTEState,
    *,
    relative_floor: float = 1.0e-12,
    ion_relative_floor: float | None = None,
    level_relative_floor: float | None = None,
) -> float:
    """Return the largest physically weighted change across all metal states."""

    changes = []
    ion_floor = relative_floor if ion_relative_floor is None else ion_relative_floor
    level_floor = (
        relative_floor if level_relative_floor is None else level_relative_floor
    )
    if previous_light is not None and current_light is not None:
        for element, population in current_light.ion_number_density.items():
            changes.append(_population_relative_change(
                previous_light.ion_number_density[element],
                population,
                np.asarray(metal_state.element_number_density[element]),
                relative_floor=ion_floor,
            ))
    for previous, current in (
        (previous_carbon, current_carbon),
        (previous_oxygen, current_oxygen),
    ):
        if previous is not None and current is not None:
            if previous.level_key == current.level_key:
                previous_population = previous.population_density
            else:
                previous_by_key = {
                    key: previous.population_density[index]
                    for index, key in enumerate(previous.level_key)
                }
                previous_population = np.asarray([
                    previous_by_key.get(
                        key, current.lte_population_density[index]
                    )
                    for index, key in enumerate(current.level_key)
                ])
            changes.append(_population_relative_change(
                previous_population,
                current.population_density,
                np.sum(current.population_density, axis=0),
                relative_floor=level_floor,
            ))
    return max(changes, default=np.inf)


def _population_state_change_diagnostic(
    previous_light: LightMetalNLTEState | None,
    previous_carbon: ReducedLightMetalLevelState | None,
    previous_oxygen: ReducedLightMetalLevelState | None,
    current_light: LightMetalNLTEState | None,
    current_carbon: ReducedLightMetalLevelState | None,
    current_oxygen: ReducedLightMetalLevelState | None,
    metal_state: MetalLTEState,
    *,
    relative_floor: float = 1.0e-12,
    ion_relative_floor: float | None = None,
    level_relative_floor: float | None = None,
) -> Mapping[str, object]:
    """Describe the component responsible for the maximum convergence norm."""

    best: dict[str, object] = {"relative_change": 0.0}
    ion_floor = relative_floor if ion_relative_floor is None else ion_relative_floor
    level_floor = (
        relative_floor if level_relative_floor is None else level_relative_floor
    )

    def inspect(
        label: str,
        keys: tuple[object, ...],
        previous: NDArray[np.float64],
        current: NDArray[np.float64],
        total: NDArray[np.float64],
        local_relative_floor: float,
    ) -> None:
        floor = local_relative_floor * total[np.newaxis, :]
        scale = np.maximum(np.maximum(np.abs(previous), np.abs(current)), floor)
        relative = np.abs(current - previous) / np.maximum(scale, 1.0e-300)
        flat_index = int(np.argmax(relative))
        component, depth = np.unravel_index(flat_index, relative.shape)
        value = float(relative[component, depth])
        if value <= float(best["relative_change"]):
            return
        best.update({
            "relative_change": value,
            "state": label,
            "component": keys[component],
            "depth_index": int(depth),
            "current_fraction": float(
                current[component, depth] / max(total[depth], 1.0e-300)
            ),
            "previous_fraction": float(
                previous[component, depth] / max(total[depth], 1.0e-300)
            ),
        })

    if previous_light is not None and current_light is not None:
        for element, population in current_light.ion_number_density.items():
            inspect(
                "ion_stage",
                tuple((element, charge) for charge in range(population.shape[0])),
                previous_light.ion_number_density[element],
                population,
                np.asarray(metal_state.element_number_density[element]),
                ion_floor,
            )
    for label, previous, current in (
        ("carbon_level", previous_carbon, current_carbon),
        ("oxygen_level", previous_oxygen, current_oxygen),
    ):
        if (
            previous is not None
            and current is not None
            and previous.level_key == current.level_key
        ):
            inspect(
                label,
                tuple(current.level_key),
                previous.population_density,
                current.population_density,
                np.sum(current.population_density, axis=0),
                level_floor,
            )
    return MappingProxyType(best)


def _pack_population_vector(
    light_state: LightMetalNLTEState | None,
    carbon_state: ReducedLightMetalLevelState | None,
    oxygen_state: ReducedLightMetalLevelState | None,
    metal_state: MetalLTEState,
) -> NDArray[np.float64]:
    parts = []
    if light_state is not None:
        for element in sorted(light_state.ion_number_density):
            total = np.maximum(metal_state.element_number_density[element], 1.0e-300)
            fraction = light_state.ion_number_density[element] / total[np.newaxis, :]
            parts.append(np.ravel(np.log(np.maximum(fraction, 1.0e-30))))
    for state in (carbon_state, oxygen_state):
        if state is not None:
            total = np.maximum(np.sum(state.population_density, axis=0), 1.0e-300)
            fraction = state.population_density / total[np.newaxis, :]
            parts.append(np.ravel(np.log(np.maximum(fraction, 1.0e-30))))
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)


def _states_from_population_vector(
    vector: NDArray[np.float64],
    light_template: LightMetalNLTEState | None,
    carbon_template: ReducedLightMetalLevelState | None,
    oxygen_template: ReducedLightMetalLevelState | None,
    metal_state: MetalLTEState,
) -> tuple[
    LightMetalNLTEState | None,
    ReducedLightMetalLevelState | None,
    ReducedLightMetalLevelState | None,
]:
    """Decode positive, normalized state fractions after Ng/Anderson mixing."""

    offset = 0
    light_state = light_template
    if light_template is not None:
        populations = {}
        for element in sorted(light_template.ion_number_density):
            shape = light_template.ion_number_density[element].shape
            size = int(np.prod(shape))
            log_fraction = vector[offset : offset + size].reshape(shape)
            offset += size
            log_fraction = log_fraction - np.max(
                log_fraction, axis=0, keepdims=True
            )
            fraction = np.exp(np.clip(log_fraction, -80.0, 0.0))
            fraction /= np.sum(fraction, axis=0, keepdims=True)
            populations[element] = fraction * metal_state.element_number_density[
                element
            ][np.newaxis, :]
        departures = {
            (element, charge): _finite_departure_ratio(
                population[charge],
                light_template.lte_ion_number_density[element][charge],
            )
            for element, population in populations.items()
            for charge in range(population.shape[0])
        }
        light_state = replace(
            light_template,
            ion_number_density=MappingProxyType(populations),
            ion_departure_coefficient=MappingProxyType(departures),
        )

    decoded_levels = []
    for template in (carbon_template, oxygen_template):
        if template is None:
            decoded_levels.append(None)
            continue
        shape = template.population_density.shape
        size = int(np.prod(shape))
        log_fraction = vector[offset : offset + size].reshape(shape)
        offset += size
        log_fraction = log_fraction - np.max(log_fraction, axis=0, keepdims=True)
        fraction = np.exp(np.clip(log_fraction, -80.0, 0.0))
        fraction /= np.sum(fraction, axis=0, keepdims=True)
        weight = template.conservation_weight
        if weight is None:
            total = np.sum(template.population_density, axis=0)
            population = fraction * total[np.newaxis, :]
        else:
            # Preserve the same explicit-plus-reservoir particle constraint
            # as the rate solve. Keeping the raw explicit sum instead leaves
            # a hidden, lagged normalization outside the accelerated vector.
            total = np.sum(weight * template.population_density, axis=0)
            normalization = total / np.sum(weight * fraction, axis=0)
            population = fraction * normalization[np.newaxis, :]
        population_departures = {
            key: _finite_departure_ratio(
                population[index], template.lte_population_density[index]
            )
            for index, key in enumerate(template.level_key)
        }
        decoded_levels.append(replace(
            template,
            population_density=np.asarray(population),
            population_level_departure_coefficient=MappingProxyType(
                population_departures
            ),
            level_departure_coefficient=_formal_departures_from_population(
                template, population_departures
            ),
        ))
    if offset != vector.size:
        raise ValueError("population acceleration vector has an inconsistent size")
    return light_state, decoded_levels[0], decoded_levels[1]


def _ion_departures_with_explicit_stage_closure(
    light_state: LightMetalNLTEState | None,
    *level_states: ReducedLightMetalLevelState | None,
) -> Mapping[tuple[str, int], NDArray[np.float64]] | None:
    """Use explicit-atom ion populations for levels outside the reduced atom.

    The independent ground-state ion ladder is useful as a complete fallback,
    but once an explicit atom has solved a charge stage its summed population
    is the more consistent ionization balance. Assigning the independent
    ladder departure to omitted levels mixed two statistical-equilibrium
    solutions inside one ion, particularly for optical O V lines immediately
    outside the reduced atom.
    """

    departures = (
        {}
        if light_state is None
        else dict(light_state.ion_departure_coefficient)
    )
    for state in level_states:
        if state is None:
            continue
        for charge in sorted({int(key[1]) for key in state.level_key}):
            selected = np.asarray(
                [int(key[1]) == charge for key in state.level_key], dtype=bool
            )
            weight = (
                1.0
                if state.conservation_weight is None
                else state.conservation_weight[selected]
            )
            population = np.sum(
                weight * state.population_density[selected], axis=0
            )
            lte_population = np.sum(
                weight * state.lte_population_density[selected], axis=0
            )
            departures[(state.element, charge)] = _finite_departure_ratio(
                population, lte_population
            )
    return MappingProxyType(departures) if departures else None


@dataclass(frozen=True)
class PG1159NLTEState:
    """Helium NLTE populations and the common He/C/O reference state."""

    helium_state: CoupledHeliumNLTEState
    metal_state: MetalLTEState
    light_metal_state: LightMetalNLTEState | None
    carbon_level_state: ReducedLightMetalLevelState | None
    oxygen_level_state: ReducedLightMetalLevelState | None
    metadata: dict[str, object]
    trace_level_states: Mapping[
        str, ReducedLightMetalLevelState
    ] = field(default_factory=lambda: MappingProxyType({}))

    radiation_wavelength: FloatArray | None = None
    radiation_mean_intensity: FloatArray | None = None
    # Local approximate lambda operator of the last population field (diagonal
    # Lambda* and total source function on radiation_wavelength); recorded only
    # when the population iteration uses ALI.
    radiation_diagonal_lambda: FloatArray | None = None
    radiation_source_function: FloatArray | None = None

    @property
    def converged(self) -> bool:
        """Whether the coupled He/C/O population iteration converged."""

        return bool(self.metadata.get("metal_population_converged", True))

    @property
    def iterations(self) -> int:
        """Number of coupled population iterations used for this state."""

        return int(self.metadata.get("metal_population_iterations", 1))


def pg1159_composition_atmosphere(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    mass_fractions: Mapping[str, float] = PG1159_035_MASS_FRACTIONS,
) -> tuple[Atmosphere, MetalLTEState]:
    """Apply a homogeneous bulk He/C/O closure to a helium atmosphere."""

    state = helium_metal_lte_state_from_mass_fractions(
        atmosphere, atomic_database, mass_fractions
    )
    composed = atmosphere_with_metal_electrons(atmosphere, state)
    metadata = dict(composed.metadata)
    metadata.update(
        {
            "composition": "homogeneous-helium-carbon-oxygen",
            "pg1159_mass_fractions": dict(state.mass_fraction or {}),
            "carbon_oxygen_population_model": "LTE reference",
        }
    )
    return replace(composed, metadata=metadata), state


def _pg1159_nlte_charge_feedback(
    atmosphere: Atmosphere,
    metal_state: MetalLTEState,
    previous_state: PG1159NLTEState,
) -> tuple[Atmosphere, MetalLTEState]:
    """Close pressure and charge with departures fixed and consistent LTE ratios.

    At fixed temperature, adjacent Saha ratios scale inversely with electron
    density. Recompute that reference while closing the NLTE charge: inverse
    radiative and collisional rates must use the same electrons as forward
    collisional rates. Only the PG mixture closure is affected.
    """
    host_lte = metal_state.host_ion_number_density
    if host_lte is None:
        raise ValueError("PG 1159 NLTE charge feedback requires a helium host")
    tiny = np.finfo(np.float64).tiny
    helium = previous_state.helium_state
    previous_helium_population = np.stack((
        helium.neutral_population_density.sum(axis=1),
        helium.singly_ionized_population_density.sum(axis=1),
        helium.doubly_ionized_he_density,
    ))
    previous_helium_lte = np.stack((
        helium.lte_neutral_population_density.sum(axis=1),
        helium.lte_singly_ionized_population_density.sum(axis=1),
        helium.lte_doubly_ionized_he_density,
    ))
    helium_departure = _finite_departure_ratio(
        previous_helium_population, previous_helium_lte
    )
    ion_departure = _ion_departures_with_explicit_stage_closure(
        previous_state.light_metal_state,
        previous_state.carbon_level_state,
        previous_state.oxygen_level_state,
        *previous_state.trace_level_states.values(),
    )
    references = {"He": host_lte, **dict(metal_state.ion_number_density)}
    departures = {"He": helium_departure}
    for element, population in metal_state.ion_number_density.items():
        departures[element] = np.stack([
            np.ones(atmosphere.n_depth) if ion_departure is None
            else np.asarray(ion_departure.get((element, q), np.ones(atmosphere.n_depth)))
            for q in range(population.shape[0])
        ])
    old_helium_density = host_lte.sum(axis=0)
    number_ratio = {"He": np.ones(atmosphere.n_depth)}
    number_ratio.update({
        element: density / np.maximum(old_helium_density, tiny)
        for element, density in metal_state.element_number_density.items()
    })
    nuclei_per_helium = sum(number_ratio.values())
    log_reference_electrons = np.log(metal_state.electron_density)
    with np.errstate(divide="ignore"):
        log_reference = {e: np.log(p) for e, p in references.items()}
        log_departure = {e: np.log(b) for e, b in departures.items()}
    charges = {e: np.arange(len(p))[:, None] for e, p in references.items()}

    def normalized(log_population):
        fraction = np.exp(log_population - np.max(log_population, axis=0))
        return fraction / fraction.sum(axis=0)

    def fractions_and_charge(log_electrons):
        fractions = {}
        mean_charge = np.zeros(atmosphere.n_depth)
        for element, reference in log_reference.items():
            shifted = reference - charges[element] * (
                log_electrons - log_reference_electrons
            )
            fractions[element] = normalized(shifted)
            actual = normalized(shifted + log_departure[element])
            mean_charge += number_ratio[element] * np.sum(charges[element] * actual, axis=0)
        return fractions, mean_charge

    particle_density = atmosphere.gas_pressure / (BOLTZMANN * atmosphere.temperature)
    lower = np.full(atmosphere.n_depth, np.log(tiny))
    upper = np.log(particle_density)
    for _ in range(64):
        middle = .5 * (lower + upper)
        _, mean_charge = fractions_and_charge(middle)
        required = particle_density * mean_charge / (nuclei_per_helium + mean_charge)
        above = np.exp(middle) > required
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle)
    electron_density = np.exp(.5 * (lower + upper))
    fractions, mean_charge = fractions_and_charge(np.log(electron_density))
    helium_density = particle_density / (nuclei_per_helium + mean_charge)
    scale = helium_density / np.maximum(old_helium_density, tiny)
    element_density = {
        element: helium_density * number_ratio[element]
        for element in metal_state.element_number_density
    }
    ion_density = {
        element: density[None, :] * fractions[element]
        for element, density in element_density.items()
    }
    # This field describes electrons in the LTE reference, as before. The
    # atmosphere's electron_density above is the actual NLTE closure value.
    reference_metal_electrons = sum(
        np.sum(charges[e] * p, axis=0) for e, p in ion_density.items()
    )
    adjusted_state = replace(
        metal_state,
        element_number_density=MappingProxyType(element_density),
        ion_number_density=MappingProxyType(ion_density),
        electron_density=electron_density,
        metal_electron_density=np.asarray(reference_metal_electrons),
        host_ion_number_density=helium_density[None, :] * fractions["He"],
        total_mass_density=(None if metal_state.total_mass_density is None
                            else metal_state.total_mass_density * scale),
    )
    adjusted_atmosphere = atmosphere_with_metal_electrons(atmosphere, adjusted_state)
    # Recompute this ratio using the new nuclei density. The generic adapter
    # also serves fixed-host mixtures and retains its original denominator.
    adjusted_atmosphere = replace(adjusted_atmosphere, helium_lte_state=replace(
        adjusted_atmosphere.helium_lte_state,
        mean_ion_charge=fractions["He"][1] + 2 * fractions["He"][2],
    ))
    closure = (electron_density + helium_density * nuclei_per_helium) / particle_density
    metadata = {
        **adjusted_atmosphere.metadata,
        "nlte_charge_feedback": "fixed-pressure charge closure with Saha-consistent LTE references",
        "nlte_charge_feedback_electron_ratio": electron_density / metal_state.electron_density,
        "nlte_charge_feedback_maximum_pressure_residual": float(np.max(abs(closure - 1.0))),
        "nlte_reference_electron_density_consistent": True,
    }
    return replace(adjusted_atmosphere, metadata=metadata), adjusted_state


# Shared immutable defaults for PG1159NLTEModel.  Python >= 3.11 rejects
# unhashable dataclass defaults such as MappingProxyType, so the fields use
# default factories that return these same objects: every instance still
# shares one read-only mapping, exactly as with the former plain defaults.
_EMPTY_MAPPING: Mapping = MappingProxyType({})
_DEFAULT_CARBON_LEVELS_PER_CHARGE: Mapping[int, int] = MappingProxyType(
    {2: 6, 3: 54, 4: 1}
)
_DEFAULT_OXYGEN_LEVELS_PER_CHARGE: Mapping[int, int] = MappingProxyType(
    {3: 8, 4: 6, 5: 27, 6: 1}
)


@dataclass(frozen=True)
class PG1159NLTEModel:
    """Reduced He/C/O NLTE provider for the common atmosphere loop.

    This is a controlled first milestone, not yet a complete PG 1159 atom:
    helium, C/O ion stages, selected C III--V levels, and the O VI optical
    system embedded between O V/O VII reservoirs are solved in statistical
    equilibrium.  Levels outside the reduced atoms retain their LTE
    distribution within each NLTE ion stage.  All C/O opacity participates
    in every temperature update.
    """

    helium_model: CoupledHeliumNLTEModel
    atomic_database: AtomicDatabase
    mass_fractions: Mapping[str, float] = field(
        default_factory=lambda: PG1159_035_MASS_FRACTIONS
    )
    photoionization_database: VernerPhotoionizationDatabase | None = None
    minimum_metal_oscillator_strength: float = 1.0e-6
    maximum_metal_lines: int | None = 20_000
    maximum_population_trace_lines: int | None = 5_000
    nlte_metal_elements: tuple[str, ...] = ("C", "O")
    trace_opacity_in_population_radiation: bool = False
    use_explicit_metal_line_source_functions: bool = True
    include_static_linear_stark: bool = True
    include_semiclassical_ovi_stark_widths: bool = False
    include_ovi_high_series_ion_dephasing: bool = False
    include_ion_dynamic_stark_core: bool = False
    hydrogenic_linear_stark_components: bool = False
    static_linear_stark_frequency_scales: Mapping[
        tuple[str, int, int, int], float
    ] = field(default_factory=lambda: _EMPTY_MAPPING)
    tabulated_electron_stark_width_scale: float = 1.0
    extend_strong_uv_resonance_wings: bool = True
    strong_uv_resonance_core_optical_depth: float = 1.0e3
    solve_carbon_oxygen_ionization_nlte: bool = True
    solve_carbon_levels_nlte: bool = True
    solve_oxygen_levels_nlte: bool = True
    carbon_levels_per_charge: Mapping[int, int] = field(
        default_factory=lambda: _DEFAULT_CARBON_LEVELS_PER_CHARGE
    )
    carbon_population_atomic_database: AtomicDatabase | None = None
    carbon_formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None = None
    carbon_formal_lte_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None
    carbon_continuum_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None
    carbon_lte_level_reservoir: Mapping[
        tuple[str, int, int], tuple[tuple[int, float, float], ...]
    ] | None = None
    carbon_lte_bound_bound_couplings: tuple[
        TmadLTEBoundBoundCoupling, ...
    ] | None = None
    carbon_effective_dielectronic_couplings: tuple[
        TmadEffectiveDielectronicCoupling, ...
    ] | None = None
    carbon_collision_data: Mapping[
        tuple[str, int, int, int],
        tuple[int, tuple[float, ...]]
        | ChiantiTermCollisionStrength
        | ConstantEffectiveCollisionStrength
        | PSM20AngularMomentumMixingCollision
        | BTMQuadrupoleAngularMomentumMixingCollision,
    ] | None = None
    carbon_photoionization_threshold_data: Mapping[
        int, TlustyPhotoionizationThresholdData
    ] | None = None
    oxygen_levels_per_charge: Mapping[int, int] = field(
        default_factory=lambda: _DEFAULT_OXYGEN_LEVELS_PER_CHARGE
    )
    oxygen_population_atomic_database: AtomicDatabase | None = None
    oxygen_formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None = None
    oxygen_formal_lte_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None
    oxygen_continuum_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None
    oxygen_lte_level_reservoir: Mapping[
        tuple[str, int, int], tuple[tuple[int, float, float], ...]
    ] | None = None
    oxygen_lte_bound_bound_couplings: tuple[
        TmadLTEBoundBoundCoupling, ...
    ] | None = None
    oxygen_effective_dielectronic_couplings: tuple[
        TmadEffectiveDielectronicCoupling, ...
    ] | None = None
    oxygen_collision_data: Mapping[
        tuple[str, int, int, int],
        tuple[int, tuple[float, ...]]
        | ChiantiTermCollisionStrength
        | ConstantEffectiveCollisionStrength
        | PSM20AngularMomentumMixingCollision
        | BTMQuadrupoleAngularMomentumMixingCollision,
    ] | None = None
    oxygen_photoionization_threshold_data: Mapping[
        int, TlustyPhotoionizationThresholdData
    ] | None = None
    trace_photoionization_threshold_data: Mapping[
        tuple[str, int], TlustyPhotoionizationThresholdData
    ] = field(default_factory=lambda: _EMPTY_MAPPING)
    ion_stage_range_overrides: Mapping[
        str, tuple[int, int]
    ] = field(default_factory=lambda: _EMPTY_MAPPING)
    ionization_wavelength_points: int = 360
    metal_rate_line_velocity_samples_kms: tuple[float, ...] | None = None
    ionization_scattering_iterations: int = 4
    population_transfer: str = "legacy"
    population_nlte_fraction: float = 1.0
    use_population_ali: bool = True
    minimum_active_lte_ion_fraction: float = 1.0e-12
    active_ion_stage_margin: int = 1
    recouple_helium_radiation: bool = True
    nlte_charge_feedback: bool = False
    coupled_population_acceleration_depth: int = 0
    helium_population_damping: float = 0.4
    helium_population_relative_floor: float = 1.0e-12
    metal_population_iterations: int = 3
    metal_population_minimum_iterations: int = 3
    metal_population_relative_tolerance: float = 3.0e-3
    metal_population_relative_floor: float = 1.0e-12
    metal_ion_population_relative_floor: float | None = None
    metal_level_population_relative_floor: float | None = None
    metal_population_damping: float = 1.0
    metal_population_minimum_damping: float = 1.0e-3
    metal_population_acceleration_depth: int = 0
    metal_population_acceleration_damping: float = 0.7
    adaptive_atmosphere_population_effort: bool = False
    # Full-NLTE atmosphere stages can use the transfer response to solve the
    # bolometric flux profile directly.  Initializer stages retain the cheap
    # operator-split map and override this flag in the structure driver.
    use_pg1159_response_jacobian: bool = False
    pg1159_population_response_probes: int = 1
    atmosphere_coarse_population_iterations: int = 12
    atmosphere_coarse_population_relative_tolerance: float = 2.0e-2
    atmosphere_population_tightening_threshold: float = 1.5e-2
    atmosphere_population_accuracy_final: bool = True

    @property
    def name(self) -> str:
        return "pg1159-he-co-reduced-nlte"

    def atmosphere_iteration_model(
        self,
        iteration: int,
        previous_temperature_correction: float | None,
        previous_local_energy_residual: float | None = None,
    ) -> PG1159NLTEModel:
        """Choose inexact or final population accuracy for an atmosphere step.

        Accurate statistical equilibrium is wasteful while the thermal
        structure is still moving by several percent.  The final-accuracy
        flag prevents the outer solver from converging on a state obtained
        with the deliberately loose early tolerance.
        """

        del iteration
        if not self.adaptive_atmosphere_population_effort:
            return self
        temperature_near_convergence = (
            previous_temperature_correction is not None
            and previous_temperature_correction
            <= self.atmosphere_population_tightening_threshold
        )
        local_balance_near_convergence = (
            previous_local_energy_residual is None
            or previous_local_energy_residual
            <= self.atmosphere_population_tightening_threshold
        )
        if temperature_near_convergence and local_balance_near_convergence:
            return replace(self, atmosphere_population_accuracy_final=True)
        coarse_iterations = min(
            self.metal_population_iterations,
            self.atmosphere_coarse_population_iterations,
        )
        return replace(
            self,
            metal_population_iterations=coarse_iterations,
            metal_population_minimum_iterations=min(
                self.metal_population_minimum_iterations,
                coarse_iterations,
            ),
            metal_population_relative_tolerance=max(
                self.metal_population_relative_tolerance,
                self.atmosphere_coarse_population_relative_tolerance,
            ),
            atmosphere_population_accuracy_final=False,
        )

    def rebuild_atmosphere(
        self,
        template: Atmosphere,
        temperature: FloatArray,
        previous_state: Any | None,
    ) -> Atmosphere:
        previous_helium = (
            previous_state.helium_state
            if isinstance(previous_state, PG1159NLTEState)
            else None
        )
        helium_atmosphere = self.helium_model.rebuild_atmosphere(
            template, temperature, previous_helium
        )
        composed, metal_state = pg1159_composition_atmosphere(
            helium_atmosphere, self.atomic_database, self.mass_fractions
        )
        if self.nlte_charge_feedback and isinstance(
            previous_state, PG1159NLTEState
        ):
            composed, _ = _pg1159_nlte_charge_feedback(
                composed, metal_state, previous_state
            )
        metadata = dict(composed.metadata)
        metadata.update(
            {
                "nlte_ion_stage_scope": "coupled He I/II/III; LTE C/O reference",
                "nlte_charge_feedback": (
                    "previous NLTE He/C/O departures"
                    if self.nlte_charge_feedback
                    and isinstance(previous_state, PG1159NLTEState)
                    else "LTE He/C/O closure"
                ),
            }
        )
        return replace(composed, metadata=metadata)

    def solve_populations(
        self,
        atmosphere: Atmosphere,
        previous_state: Any | None,
        *,
        iteration_callback: Callable[[int, PG1159NLTEState], None] | None = None,
        _fixed_radiation: tuple[FloatArray,FloatArray] | None = None,
        _local_lambda_operator: tuple[FloatArray, FloatArray] | None = None,
        _stop_iteration: Callable[[int, PG1159NLTEState], bool] | None = None,
        _coupled_population_history: Any | None = None,
        _coupled_population_near_root_history: int | None = None,
        _coupled_population_near_root_threshold: float | None = None,
    ) -> PG1159NLTEState:
        if not 0.0 < self.metal_population_damping <= 1.0:
            raise ValueError("metal_population_damping must be in (0, 1]")
        if not 0.0 < self.metal_population_minimum_damping <= self.metal_population_damping:
            raise ValueError(
                "metal_population_minimum_damping must be positive and no larger "
                "than metal_population_damping"
            )
        if self.coupled_population_acceleration_depth != 0 and self.coupled_population_acceleration_depth < 2:
            raise ValueError("coupled_population_acceleration_depth must be zero or at least two")
        if self.metal_population_acceleration_depth < 0:
            raise ValueError("metal_population_acceleration_depth must be non-negative")
        if not 0.0 < self.metal_population_acceleration_damping <= 1.0:
            raise ValueError("metal_population_acceleration_damping must be in (0, 1]")
        if self.metal_population_iterations < 1:
            raise ValueError("metal_population_iterations must be positive")
        if self.atmosphere_coarse_population_iterations < 1:
            raise ValueError(
                "atmosphere_coarse_population_iterations must be positive"
            )
        if self.atmosphere_coarse_population_relative_tolerance <= 0.0:
            raise ValueError(
                "atmosphere_coarse_population_relative_tolerance must be positive"
            )
        if self.atmosphere_population_tightening_threshold <= 0.0:
            raise ValueError(
                "atmosphere_population_tightening_threshold must be positive"
            )
        if (
            self.maximum_population_trace_lines is not None
            and self.maximum_population_trace_lines < 1
        ):
            raise ValueError("maximum_population_trace_lines must be positive")
        if not 1 <= self.metal_population_minimum_iterations <= self.metal_population_iterations:
            raise ValueError(
                "metal_population_minimum_iterations must lie between 1 and the maximum"
            )
        if self.metal_population_relative_tolerance <= 0.0:
            raise ValueError("metal_population_relative_tolerance must be positive")
        if not 0.0 < self.metal_population_relative_floor < 1.0:
            raise ValueError("metal_population_relative_floor must lie in (0, 1)")
        for name, value in (
            (
                "metal_ion_population_relative_floor",
                self.metal_ion_population_relative_floor,
            ),
            (
                "metal_level_population_relative_floor",
                self.metal_level_population_relative_floor,
            ),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if not 0.0 < self.helium_population_relative_floor < 1.0:
            raise ValueError("helium_population_relative_floor must lie in (0, 1)")
        composed, metal_state = pg1159_composition_atmosphere(
            atmosphere, self.atomic_database, self.mass_fractions
        )
        if self.nlte_charge_feedback and isinstance(
            previous_state, PG1159NLTEState
        ):
            composed, metal_state = _pg1159_nlte_charge_feedback(
                composed, metal_state, previous_state
            )
        previous_helium = (
            previous_state.helium_state
            if isinstance(previous_state, PG1159NLTEState)
            else None
        )
        if not 0.0 < self.helium_population_damping <= 1.0:
            raise ValueError("helium_population_damping must lie in (0, 1]")
        helium_state = (
            self.helium_model.solve_populations(composed, None)
            if previous_helium is None
            else remap_coupled_helium_state(composed, previous_helium)
        )
        if isinstance(previous_state, PG1159NLTEState):
            light_metal_state = previous_state.light_metal_state
            carbon_level_state = previous_state.carbon_level_state
            oxygen_level_state = previous_state.oxygen_level_state
            trace_level_states = previous_state.trace_level_states
        else:
            light_metal_state = None
            carbon_level_state = None
            oxygen_level_state = None
            trace_level_states = MappingProxyType({})
        if (
            light_metal_state is not None
            and not set(self.nlte_metal_elements).issubset(
                light_metal_state.ion_number_density
            )
        ):
            # Old checkpoints contained only the C/O ground-state ladder.
            # Retain their explicit term populations as a warm start, but
            # rebuild the ion ladder when newly enabled trace species are
            # meant to participate in the NLTE radiation field.
            light_metal_state = _add_missing_lte_ion_ladders(
                light_metal_state,
                metal_state,
                self.nlte_metal_elements,
            )
        rate_wavelength = rate_mean_intensity = None
        rate_diagonal_lambda = rate_source_function = None
        if _local_lambda_operator is not None and _fixed_radiation is None:
            raise ValueError("a local lambda operator needs its fixed radiation field")
        population_iteration_count = 0
        population_relative_change = np.inf
        undamped_population_change = np.inf
        population_converged = False
        population_relative_change_history: list[float] = []
        metal_only_population_change_history: list[float] = []
        coupled_acceleration_history = (
            [] if _coupled_population_history is None else _coupled_population_history
        )
        near_root_acceleration = False
        recycled_population_secants = len(getattr(coupled_acceleration_history, "retained", ()))
        helium_population_change_history: list[float] = []
        population_worst_change: Mapping[str, object] = MappingProxyType({})
        helium_population_worst_change: Mapping[str, object] = MappingProxyType({})
        undamped_population_worst_change: Mapping[str, object] = MappingProxyType({})
        undamped_helium_population_worst_change: Mapping[str, object] = MappingProxyType({})
        effective_population_damping = self.metal_population_damping
        population_x_history: list[NDArray[np.float64]] = []
        population_g_history: list[NDArray[np.float64]] = []
        if (
            self.solve_carbon_oxygen_ionization_nlte
            or self.solve_carbon_levels_nlte
            or self.solve_oxygen_levels_nlte
        ):
            if self.photoionization_database is None:
                raise ValueError(
                    "C/O NLTE requires a photoionization database"
                )
            wavelength_parts = []
            if self.solve_carbon_oxygen_ionization_nlte:
                wavelength_parts.append(default_light_metal_ionization_wavelength(
                    self.photoionization_database,
                    elements=self.nlte_metal_elements,
                    n_wavelength=self.ionization_wavelength_points,
                    photoionization_threshold_data=(
                        self.trace_photoionization_threshold_data
                    ),
                ))
            if self.solve_carbon_levels_nlte:
                wavelength_parts.append(reduced_light_metal_wavelength(
                    (
                        self.atomic_database
                        if self.carbon_population_atomic_database is None
                        else self.carbon_population_atomic_database
                    ),
                    "C",
                    self.carbon_levels_per_charge,
                    n_continuum_wavelength=self.ionization_wavelength_points,
                    photoionization_threshold_data=(
                        self.carbon_photoionization_threshold_data
                    ),
                    lte_bound_bound_couplings=(
                        self.carbon_lte_bound_bound_couplings
                    ),
                    effective_dielectronic_couplings=(
                        self.carbon_effective_dielectronic_couplings
                    ),
                    line_velocity_samples_kms=(
                        self.metal_rate_line_velocity_samples_kms
                    ),
                ))
            if self.solve_oxygen_levels_nlte:
                wavelength_parts.append(reduced_light_metal_wavelength(
                    (
                        self.atomic_database
                        if self.oxygen_population_atomic_database is None
                        else self.oxygen_population_atomic_database
                    ),
                    "O",
                    self.oxygen_levels_per_charge,
                    n_continuum_wavelength=self.ionization_wavelength_points,
                    photoionization_threshold_data=(
                        self.oxygen_photoionization_threshold_data
                    ),
                    lte_bound_bound_couplings=(
                        self.oxygen_lte_bound_bound_couplings
                    ),
                    effective_dielectronic_couplings=(
                        self.oxygen_effective_dielectronic_couplings
                    ),
                    line_velocity_samples_kms=(
                        self.metal_rate_line_velocity_samples_kms
                    ),
                ))
            helium_neutral_wavelength = None
            helium_ion_wavelength = None
            helium_ion_line_problems = ()
            if self.recouple_helium_radiation:
                helium_neutral_wavelength = (
                    default_neutral_helium_continuum_wavelength()
                )
                helium_ion_wavelength = default_helium_ii_continuum_wavelength(
                    self.helium_model.maximum_helium_ii_level
                )
                # Population-rate closure is larger than the compact named
                # He II line list used by the emergent-spectrum synthesizer.
                # The hydrogenic fallback provides exact shell wavelengths
                # and f-values for every remaining pair.
                helium_ion_line_problems = tuple(
                    _prepare_helium_line_transfer_problem(
                        composed,
                        helium_ii_shell_transition(lower, upper),
                        self.helium_model.maximum_helium_ii_level,
                        stark_table=self.helium_model.helium_ii_stark_table,
                    )
                    for lower in range(
                        1,
                        self.helium_model.population_explicit_maximum_lower_level
                        + 1,
                    )
                    for upper in range(
                        lower + 1,
                        self.helium_model.maximum_helium_ii_level + 1,
                    )
                )
                wavelength_parts.extend((
                    helium_neutral_wavelength,
                    helium_ion_wavelength,
                    *(problem.continuum.wavelength_angstrom
                      for problem in helium_ion_line_problems),
                ))
            ionization_wavelength = np.unique(np.concatenate(wavelength_parts))
            planck = planck_lambda_angstrom(
                ionization_wavelength[:, np.newaxis],
                composed.temperature[np.newaxis, :],
            )
            free_free_charges = {
                charge
                for element in metal_state.log_number_abundance
                for charge in range(
                    1,
                    metal_state.ion_number_density[element].shape[0],
                )
            }
            metal_free_free_charge_kernel = (
                light_metal_free_free_charge_kernel(
                    composed,
                    ionization_wavelength,
                    free_free_charges,
                )
            )
            trace_elements = set(metal_state.log_number_abundance) - set(
                self.nlte_metal_elements
            )
            non_c_o_elements = set(metal_state.log_number_abundance) - {"C", "O"}
            population_feedback_elements = (
                set(self.nlte_metal_elements)
                if self.trace_opacity_in_population_radiation
                else {"C", "O"}
            )
            # Trace species remain in LTE during the present C/O solve.  At
            # fixed structure their opacity is therefore invariant across
            # the outer population iteration; evaluating tens of thousands
            # of iron-group profiles on every C/O update was pure duplicate
            # work and made trace-blanketed validation prohibitively slow.
            trace_bound_free_opacity = np.zeros_like(planck)
            trace_line_opacity = np.zeros_like(planck)
            if trace_elements and self.trace_opacity_in_population_radiation:
                trace_bound_free_opacity = (
                    metal_bound_free_mass_absorption_coefficient(
                        composed,
                        ionization_wavelength,
                        self.atomic_database,
                        metal_state,
                        self.photoionization_database,
                        excluded_elements=self.nlte_metal_elements,
                    )
                )
                trace_line_opacity = metal_line_mass_absorption_coefficient(
                    composed,
                    ionization_wavelength,
                    self.atomic_database,
                    metal_state,
                    minimum_oscillator_strength=(
                        self.minimum_metal_oscillator_strength
                    ),
                    maximum_lines=self.maximum_population_trace_lines,
                    excluded_elements=self.nlte_metal_elements,
                )
            helium_transfer_options = ({'_cache':PG1159HeliumTransferCache(composed,ionization_wavelength)}
                if isinstance(self.helium_model,CoupledHeliumNLTEModel) else {})
            for iteration in range(self.metal_population_iterations):
                previous_helium_state = helium_state
                previous_light_metal_state = light_metal_state
                previous_carbon_level_state = carbon_level_state
                previous_oxygen_level_state = oxygen_level_state
                if _fixed_radiation is None or _local_lambda_operator is not None:
                    helium_coefficients = self.helium_model.transfer_coefficients(
                        composed, ionization_wavelength, helium_state, **helium_transfer_options
                    )
                    element_line_absorption: dict[str, FloatArray] = {}
                    if light_metal_state is None:
                        excluded_from_initial_field = (
                            ()
                            if self.trace_opacity_in_population_radiation
                            else tuple(non_c_o_elements)
                        )
                        metal_bound_free = metal_bound_free_mass_absorption_coefficient(
                            composed,
                            ionization_wavelength,
                            self.atomic_database,
                            metal_state,
                            self.photoionization_database,
                            excluded_elements=excluded_from_initial_field,
                        )
                        metal_emissivity = metal_bound_free * planck
                        metal_lines = metal_line_mass_absorption_coefficient(
                            composed,
                            ionization_wavelength,
                            self.atomic_database,
                            metal_state,
                            minimum_oscillator_strength=(
                                self.minimum_metal_oscillator_strength
                            ),
                            maximum_lines=self.maximum_metal_lines,
                            excluded_elements=excluded_from_initial_field,
                        )
                        metal_line_emissivity = metal_lines * planck
                        free_free_departure = None
                    else:
                        combined_ion_departure = (
                            _ion_departures_with_explicit_stage_closure(
                                light_metal_state,
                                carbon_level_state,
                                oxygen_level_state,
                                *trace_level_states.values(),
                            )
                        )
                        if combined_ion_departure is None:
                            raise RuntimeError(
                                "internal PG1159 state has no active NLTE ion populations"
                            )
                        free_free_departure = combined_ion_departure
                        # TMAP deliberately uses compact, term-resolved atoms in
                        # the atmosphere/SE iteration and separate fine-structure
                        # atoms only for the formal spectrum.  Feeding every
                        # Stout synthesis line back into this radiation field
                        # over-counts multiplets and couples levels absent from
                        # the compact rate matrix.  Mirror the structure/formal
                        # split whenever a TMAD population atom is available.
                        metal_bound_free = np.zeros_like(planck)
                        metal_emissivity = np.zeros_like(planck)
                        metal_lines = np.zeros_like(planck)
                        metal_line_emissivity = np.zeros_like(planck)
                        element_atoms = (
                            (
                                "C",
                                self.atomic_database
                                if self.carbon_population_atomic_database is None
                                else self.carbon_population_atomic_database,
                                carbon_level_state,
                            ),
                            (
                                "O",
                                self.atomic_database
                                if self.oxygen_population_atomic_database is None
                                else self.oxygen_population_atomic_database,
                                oxygen_level_state,
                            ),
                        )
                        for element, population_atom, level_state in element_atoms:
                            if element not in metal_state.log_number_abundance:
                                continue
                            if (
                                population_atom is self.atomic_database
                                or level_state is None
                                or level_state.population_level_departure_coefficient is None
                            ):
                                level_departure = (
                                    None
                                    if level_state is None
                                    else level_state.level_departure_coefficient
                                )
                            else:
                                level_departure = (
                                    level_state.population_level_departure_coefficient
                                )
                            # Match TMAP's documented two-stage calculation.  The
                            # atmosphere and statistical-equilibrium iterations
                            # use the unsplit LS-term population atom.  Only the
                            # final formal solution introduces fine structure and
                            # distributes each term population over its J levels
                            # by statistical weight (Werner & Rauch 2014, Sect. 3).
                            line_atom = population_atom
                            line_level_departure = level_departure
                            local_bound_free, local_bound_free_emissivity = (
                                light_metal_bound_free_nlte_coefficients(
                                    composed,
                                    ionization_wavelength,
                                    population_atom,
                                    metal_state,
                                    self.photoionization_database,
                                    combined_ion_departure,
                                    level_departure_coefficient=level_departure,
                                    elements=(element,),
                                    photoionization_threshold_data=(
                                        self.carbon_photoionization_threshold_data
                                        if element == "C"
                                        else self.oxygen_photoionization_threshold_data
                                    ),
                                    levels_per_charge=(
                                        self.carbon_levels_per_charge
                                        if element == "C"
                                        else self.oxygen_levels_per_charge
                                    ),
                                    continuum_parent_mapping=(
                                        self.carbon_continuum_parent_mapping
                                        if element == "C"
                                        else self.oxygen_continuum_parent_mapping
                                    ),
                                )
                            )
                            local_lines, local_line_emissivity = (
                                hot_metal_line_nlte_coefficients(
                                    composed,
                                    ionization_wavelength,
                                    line_atom,
                                    metal_state,
                                    combined_ion_departure,
                                    level_departure_coefficient=line_level_departure,
                                    minimum_oscillator_strength=(
                                        self.minimum_metal_oscillator_strength
                                    ),
                                    maximum_lines=self.maximum_metal_lines,
                                    include_static_linear_stark=(
                                        self.include_static_linear_stark
                                    ),
                                    static_linear_stark_frequency_scales=(
                                        self.static_linear_stark_frequency_scales
                                    ),
                                    tabulated_electron_stark_width_scale=(
                                        self.tabulated_electron_stark_width_scale
                                    ),
                                    include_semiclassical_ovi_stark_widths=(
                                        self.include_semiclassical_ovi_stark_widths
                                    ),
                                    include_ovi_high_series_ion_dephasing=(
                                        self.include_ovi_high_series_ion_dephasing
                                    ),
                                    extend_strong_uv_resonance_wings=(
                                        self.extend_strong_uv_resonance_wings
                                    ),
                                    strong_uv_resonance_core_optical_depth=(
                                        self.strong_uv_resonance_core_optical_depth
                                    ),
                                    elements=(element,),
                                    include_ion_dynamic_stark_core=(
                                        self.include_ion_dynamic_stark_core
                                    ),
                                    hydrogenic_linear_stark_components=(
                                        self.hydrogenic_linear_stark_components
                                    ),
                                )
                            )
                            metal_bound_free += local_bound_free
                            metal_emissivity += local_bound_free_emissivity
                            metal_lines += local_lines
                            metal_line_emissivity += local_line_emissivity
                            element_line_absorption[element] = local_lines
                        # The optional research feedback mode lets ion-only
                        # trace species contribute opacity to their own SE
                        # radiation field. In the public trace approximation
                        # they remain passenger populations and this set is
                        # deliberately empty; the final formal transfer still
                        # includes their NLTE opacity.
                        ion_only_elements = tuple(
                            element
                            for element in self.nlte_metal_elements
                            if element not in {"C", "O"}
                            and element in metal_state.log_number_abundance
                            and element in population_feedback_elements
                        )
                        if ion_only_elements:
                            local_bound_free, local_bound_free_emissivity = (
                                light_metal_bound_free_nlte_coefficients(
                                    composed,
                                    ionization_wavelength,
                                    self.atomic_database,
                                    metal_state,
                                    self.photoionization_database,
                                    combined_ion_departure,
                                    elements=ion_only_elements,
                                )
                            )
                            local_lines, local_line_emissivity = (
                                hot_metal_line_nlte_coefficients(
                                    composed,
                                    ionization_wavelength,
                                    self.atomic_database,
                                    metal_state,
                                    combined_ion_departure,
                                    minimum_oscillator_strength=(
                                        self.minimum_metal_oscillator_strength
                                    ),
                                    maximum_lines=self.maximum_population_trace_lines,
                                    include_static_linear_stark=(
                                        self.include_static_linear_stark
                                    ),
                                    static_linear_stark_frequency_scales=(
                                        self.static_linear_stark_frequency_scales
                                    ),
                                    tabulated_electron_stark_width_scale=(
                                        self.tabulated_electron_stark_width_scale
                                    ),
                                    include_semiclassical_ovi_stark_widths=(
                                        self.include_semiclassical_ovi_stark_widths
                                    ),
                                    include_ovi_high_series_ion_dephasing=(
                                        self.include_ovi_high_series_ion_dephasing
                                    ),
                                    extend_strong_uv_resonance_wings=(
                                        self.extend_strong_uv_resonance_wings
                                    ),
                                    strong_uv_resonance_core_optical_depth=(
                                        self.strong_uv_resonance_core_optical_depth
                                    ),
                                    elements=ion_only_elements,
                                    include_ion_dynamic_stark_core=(
                                        self.include_ion_dynamic_stark_core
                                    ),
                                    hydrogenic_linear_stark_components=(
                                        self.hydrogenic_linear_stark_components
                                    ),
                                )
                            )
                            metal_bound_free += local_bound_free
                            metal_emissivity += local_bound_free_emissivity
                            metal_lines += local_lines
                            metal_line_emissivity += local_line_emissivity
                        if trace_elements and self.trace_opacity_in_population_radiation:
                            metal_bound_free += trace_bound_free_opacity
                            metal_emissivity += trace_bound_free_opacity * planck
                            metal_lines += trace_line_opacity
                            metal_line_emissivity += trace_line_opacity * planck
                    metal_free_free = (
                        light_metal_free_free_mass_absorption_coefficient(
                            composed,
                            ionization_wavelength,
                            metal_state,
                            free_free_departure,
                            elements=tuple(sorted(population_feedback_elements)),
                            precomputed_charge_kernel=(
                                metal_free_free_charge_kernel
                            ),
                        )
                    )
                    metal_bound_free += metal_free_free
                    metal_emissivity += metal_free_free * planck
                    true_absorption = (
                        helium_coefficients.true_absorption
                        + metal_bound_free
                        + metal_lines
                    )
                    emissivity = (
                        helium_coefficients.thermal_emissivity
                        + metal_emissivity
                        + metal_line_emissivity
                    )
                    total = _population_transfer_extinction(
                        true_absorption,
                        emissivity,
                        helium_coefficients.scattering,
                    )
                    optical_depth = optical_depth_from_mass_opacity(
                        composed.column_mass, total
                    )
                    source = np.ascontiguousarray(
                        (emissivity + helium_coefficients.scattering * planck) / total
                    )
                    field = None
                    need_legacy_operator = _local_lambda_operator is None and (
                        self.population_transfer == "legacy" or
                        (self.use_population_ali and self.population_nlte_fraction == 1.))
                    if _local_lambda_operator is not None:
                        # Local ALO response (TMAP-style linearization): the
                        # radiation follows this state's own source function
                        # through the anchor's diagonal operator,
                        #   J = J0 + Lambda* (S - S0),  S = (eta + sigma J) / chi,
                        # solved in closed form at every wavelength and depth.
                        # No cross-depth coupling enters, so depth columns of a
                        # simultaneous perturbation stay separable.
                        old_wave, old_mean = _fixed_radiation
                        anchor_mean = _interpolate_mean_intensity(
                            old_wave, old_mean, ionization_wavelength)
                        operator, anchor_source = _local_lambda_operator
                        if (np.shape(operator) != np.shape(anchor_mean)
                                or np.shape(anchor_source) != np.shape(anchor_mean)):
                            raise ValueError("local lambda operator has the wrong shape")
                        scattering_fraction = helium_coefficients.scattering / total
                        denominator = 1.0 - operator * scattering_fraction
                        local_mean = (
                            anchor_mean + operator * (emissivity / total - anchor_source)
                        ) / np.maximum(denominator, 1e-12)
                        field = RadiationField(
                            mean_intensity=np.maximum(local_mean, 0.0),
                            flux=np.zeros_like(local_mean))
                    elif need_legacy_operator:
                        operator_depth = optical_depth
                        if self.population_transfer == "mass":
                            # This field only supplies the approximate Lambda
                            # diagonal; the rates use the exact mass-transfer
                            # field below.  Keep its optical depth strictly
                            # increasing where rounding or a local inversion
                            # gives a non-positive increment.
                            increment = np.diff(optical_depth, axis=-1)
                            floor = 1e-14 * np.maximum(abs(optical_depth[..., 1:]), 1e-300)
                            if np.any(increment <= floor):
                                operator_depth = np.concatenate(
                                    (optical_depth[..., :1],
                                     optical_depth[..., :1]
                                     + np.cumsum(np.maximum(increment, floor), axis=-1)),
                                    axis=-1)
                        for _ in range(1 if self.population_transfer == "mass" else self.ionization_scattering_iterations):
                            field = radiation_field(operator_depth, source,
                                n_angle=self.helium_model.population_n_angle,
                                calculate_diagonal_lambda=self.use_population_ali)
                            source = np.ascontiguousarray((emissivity
                                + helium_coefficients.scattering * field.mean_intensity) / total)
                    if self.population_transfer == "mass" and _local_lambda_operator is None:
                        from ._pg1159_transfer import transfer_field
                        coefficients = NLTETransferCoefficients(
                            ionization_wavelength, true_absorption, emissivity,
                            helium_coefficients.scattering, {})
                        _, exact_field, _ = transfer_field(
                            composed, coefficients,
                            n_angle=self.helium_model.population_n_angle,
                            check_source=False)
                        # The old diagonal is only an approximate preconditioner.
                        # The actual rates see the same mass-volume discretization
                        # and exact scattering closure as the atmosphere equations.
                        field = (exact_field if field is None else
                            replace(exact_field, diagonal_lambda=field.diagonal_lambda))
                    elif (self.population_transfer != "legacy"
                          and _local_lambda_operator is None):
                        raise ValueError("unknown PG1159 population transfer")
                else:
                    old_wave,old_mean=_fixed_radiation
                    if np.shape(old_mean)!=(len(old_wave),composed.n_depth):
                        raise ValueError('fixed population radiation has the wrong shape')
                    mean=_interpolate_mean_intensity(old_wave,old_mean,ionization_wavelength)
                    field=RadiationField(mean_intensity=mean,flux=np.zeros_like(mean))
                    element_line_absorption={}
                if self.population_transfer=='mass':
                    rate_wavelength=ionization_wavelength
                    rate_mean_intensity=field.mean_intensity
                    if (_fixed_radiation is None
                            and getattr(field, "diagonal_lambda", None) is not None):
                        rate_diagonal_lambda = field.diagonal_lambda
                        rate_source_function = (
                            emissivity + helium_coefficients.scattering * field.mean_intensity
                        ) / total
                if not 0 <= self.population_nlte_fraction <= 1:
                    raise ValueError("population_nlte_fraction must lie in [0, 1]")
                if self.population_nlte_fraction != 1:
                    field = replace(field, mean_intensity=(
                        self.population_nlte_fraction * field.mean_intensity
                        + (1-self.population_nlte_fraction) * planck),
                        diagonal_lambda=None)
                helium_population_change = 0.0
                if self.recouple_helium_radiation:
                    assert helium_neutral_wavelength is not None
                    assert helium_ion_wavelength is not None
                    neutral_continuum_mean = _interpolate_mean_intensity(
                        ionization_wavelength,
                        field.mean_intensity,
                        helium_neutral_wavelength,
                    )
                    ion_continuum_mean = _interpolate_mean_intensity(
                        ionization_wavelength,
                        field.mean_intensity,
                        helium_ion_wavelength,
                    )
                    ion_line_fields = {}
                    for problem in helium_ion_line_problems:
                        local_mean = _interpolate_mean_intensity(
                            ionization_wavelength,
                            field.mean_intensity,
                            problem.continuum.wavelength_angstrom,
                        )
                        ion_line_fields[(
                            problem.line.lower_level,
                            problem.line.upper_level,
                        )] = _profile_averaged_mean_intensity_nu(
                            problem.continuum.wavelength_angstrom,
                            problem.lte_line_opacity,
                            local_mean,
                        )
                    next_helium_state = (
                        solve_coupled_helium_statistical_equilibrium(
                            composed,
                            self.helium_model.collision_data,
                            maximum_helium_ii_level=(
                                self.helium_model.maximum_helium_ii_level
                            ),
                            helium_ii_line_mean_intensity_nu=ion_line_fields,
                            neutral_continuum_wavelength_angstrom=(
                                helium_neutral_wavelength
                            ),
                            neutral_continuum_mean_intensity_lambda=(
                                neutral_continuum_mean
                            ),
                            helium_ii_continuum_wavelength_angstrom=(
                                helium_ion_wavelength
                            ),
                            helium_ii_continuum_mean_intensity_lambda=(
                                ion_continuum_mean
                            ),
                            neutral_collision_strength_scale=(
                                self.helium_model.neutral_collision_strength_scale
                            ),
                            helium_i_collision_data=(
                                self.helium_model.helium_i_collision_data
                            ),
                            hydrogenic_collision_model=(
                                self.helium_model.hydrogenic_collision_model
                            ),
                            hydrogenic_collision_rate_multiplier=(
                                self.helium_model.hydrogenic_collision_rate_multiplier
                            ),
                            hydrogenic_excitation_collision_rate_multiplier=(
                                self.helium_model.hydrogenic_excitation_collision_rate_multiplier
                            ),
                            hydrogenic_ionization_collision_rate_multiplier=(
                                self.helium_model.hydrogenic_ionization_collision_rate_multiplier
                            ),
                        )
                    )
                    helium_state = _damp_coupled_helium_state(
                        helium_state, next_helium_state, self.helium_population_damping)
                    helium_population_change = (
                        _maximum_coupled_helium_population_change(
                            previous_helium_state,
                            helium_state,
                            relative_floor=self.helium_population_relative_floor,
                        )
                    )
                    helium_population_worst_change = (
                        _coupled_helium_population_change_diagnostic(
                            previous_helium_state,
                            helium_state,
                            relative_floor=self.helium_population_relative_floor,
                        )
                    )
                next_light_metal_state = None
                next_carbon_level_state = None
                next_oxygen_level_state = None
                if self.solve_carbon_oxygen_ionization_nlte:
                    next_light_metal_state = solve_light_metal_ionization_nlte(
                        composed,
                        self.atomic_database,
                        metal_state,
                        self.photoionization_database,
                        ionization_wavelength,
                        field.mean_intensity,
                        elements=self.nlte_metal_elements,
                        minimum_active_lte_ion_fraction=(
                            self.minimum_active_lte_ion_fraction
                        ),
                        active_ion_stage_margin=self.active_ion_stage_margin,
                        photoionization_threshold_data=(
                            self.trace_photoionization_threshold_data
                        ),
                        active_stage_range_overrides=(
                            self.ion_stage_range_overrides
                        ),
                    )
                if self.solve_carbon_levels_nlte:
                    next_carbon_level_state = solve_reduced_light_metal_levels_nlte(
                        composed,
                        (
                            self.atomic_database
                            if self.carbon_population_atomic_database is None
                            else self.carbon_population_atomic_database
                        ),
                        metal_state,
                        self.photoionization_database,
                        ionization_wavelength,
                        field.mean_intensity,
                        "C",
                        self.carbon_levels_per_charge,
                        photoionization_threshold_data=(
                            self.carbon_photoionization_threshold_data
                        ),
                        formal_level_mapping=(
                            self.carbon_formal_level_mapping
                        ),
                        formal_lte_parent_mapping=(
                            self.carbon_formal_lte_parent_mapping
                        ),
                        continuum_parent_mapping=(
                            self.carbon_continuum_parent_mapping
                        ),
                        lte_level_reservoir=self.carbon_lte_level_reservoir,
                        lte_bound_bound_couplings=(
                            self.carbon_lte_bound_bound_couplings
                        ),
                        effective_dielectronic_couplings=(
                            self.carbon_effective_dielectronic_couplings
                        ),
                        collision_data=self.carbon_collision_data,
                        static_linear_stark_frequency_scales=(
                            self.static_linear_stark_frequency_scales
                        ),
                        include_semiclassical_ovi_stark_widths=(
                            self.include_semiclassical_ovi_stark_widths
                        ),
                        include_ovi_high_series_ion_dephasing=(
                            self.include_ovi_high_series_ion_dephasing
                        ),
                        include_ion_dynamic_stark_core=(
                            self.include_ion_dynamic_stark_core
                        ),
                        hydrogenic_linear_stark_components=(
                            self.hydrogenic_linear_stark_components
                        ),
                        approximate_lambda_diagonal=(
                            None
                            if not self.use_population_ali
                            or carbon_level_state is None
                            or field.diagonal_lambda is None
                            or "C" not in element_line_absorption
                            else field.diagonal_lambda
                            * np.maximum(element_line_absorption["C"], 0.0)
                            / np.maximum(total, np.finfo(np.float64).tiny)
                        ),
                        previous_population_state=carbon_level_state,
                    )
                if self.solve_oxygen_levels_nlte:
                    next_oxygen_level_state = solve_reduced_light_metal_levels_nlte(
                        composed,
                        (
                            self.atomic_database
                            if self.oxygen_population_atomic_database is None
                            else self.oxygen_population_atomic_database
                        ),
                        metal_state,
                        self.photoionization_database,
                        ionization_wavelength,
                        field.mean_intensity,
                        "O",
                        self.oxygen_levels_per_charge,
                        photoionization_threshold_data=(
                            self.oxygen_photoionization_threshold_data
                        ),
                        formal_level_mapping=(
                            self.oxygen_formal_level_mapping
                        ),
                        formal_lte_parent_mapping=(
                            self.oxygen_formal_lte_parent_mapping
                        ),
                        continuum_parent_mapping=(
                            self.oxygen_continuum_parent_mapping
                        ),
                        lte_level_reservoir=self.oxygen_lte_level_reservoir,
                        lte_bound_bound_couplings=(
                            self.oxygen_lte_bound_bound_couplings
                        ),
                        effective_dielectronic_couplings=(
                            self.oxygen_effective_dielectronic_couplings
                        ),
                        collision_data=self.oxygen_collision_data,
                        static_linear_stark_frequency_scales=(
                            self.static_linear_stark_frequency_scales
                        ),
                        include_semiclassical_ovi_stark_widths=(
                            self.include_semiclassical_ovi_stark_widths
                        ),
                        include_ovi_high_series_ion_dephasing=(
                            self.include_ovi_high_series_ion_dephasing
                        ),
                        include_ion_dynamic_stark_core=(
                            self.include_ion_dynamic_stark_core
                        ),
                        hydrogenic_linear_stark_components=(
                            self.hydrogenic_linear_stark_components
                        ),
                        approximate_lambda_diagonal=(
                            None
                            if not self.use_population_ali
                            or oxygen_level_state is None
                            or field.diagonal_lambda is None
                            or "O" not in element_line_absorption
                            else field.diagonal_lambda
                            * np.maximum(element_line_absorption["O"], 0.0)
                            / np.maximum(total, np.finfo(np.float64).tiny)
                        ),
                        previous_population_state=oxygen_level_state,
                    )
                if light_metal_state is None:
                    light_metal_state = next_light_metal_state
                    carbon_level_state = next_carbon_level_state
                    oxygen_level_state = next_oxygen_level_state
                else:
                    carbon_level_state = _align_reduced_level_state(
                        carbon_level_state, next_carbon_level_state
                    )
                    oxygen_level_state = _align_reduced_level_state(
                        oxygen_level_state, next_oxygen_level_state
                    )
                    x_vector = _pack_population_vector(
                        light_metal_state,
                        carbon_level_state,
                        oxygen_level_state,
                        metal_state,
                    )
                    g_vector = _pack_population_vector(
                        next_light_metal_state,
                        next_carbon_level_state,
                        next_oxygen_level_state,
                        metal_state,
                    )
                    population_x_history.append(x_vector)
                    population_g_history.append(g_vector)
                    maximum_history = self.metal_population_acceleration_depth + 1
                    population_x_history = population_x_history[-maximum_history:]
                    population_g_history = population_g_history[-maximum_history:]
                    mixed_vector = (
                        (1.0 - self.metal_population_damping) * x_vector
                        + self.metal_population_damping * g_vector
                    )
                    if (
                        self.metal_population_acceleration_depth > 0
                        and len(population_x_history) >= 2
                    ):
                        residual_history = [
                            g_value - x_value
                            for x_value, g_value in zip(
                                population_x_history, population_g_history
                            )
                        ]
                        residual_difference = np.column_stack([
                            residual_history[index + 1] - residual_history[index]
                            for index in range(len(residual_history) - 1)
                        ])
                        image_difference = np.column_stack([
                            population_g_history[index + 1]
                            - population_g_history[index]
                            for index in range(len(population_g_history) - 1)
                        ])
                        try:
                            coefficient = np.linalg.lstsq(
                                residual_difference,
                                residual_history[-1],
                                rcond=1.0e-10,
                            )[0]
                            accelerated = (
                                population_g_history[-1]
                                - image_difference @ coefficient
                            )
                            direct_step_norm = float(np.linalg.norm(
                                g_vector - x_vector
                            ))
                            accelerated_step_norm = float(np.linalg.norm(
                                accelerated - x_vector
                            ))
                            if (
                                np.all(np.isfinite(accelerated))
                                and accelerated_step_norm
                                <= 4.0 * max(direct_step_norm, 1.0e-12)
                            ):
                                beta = self.metal_population_acceleration_damping
                                mixed_vector = (
                                    (1.0 - beta) * x_vector + beta * accelerated
                                )
                                effective_population_damping = beta
                        except np.linalg.LinAlgError:
                            pass
                    if self.coupled_population_acceleration_depth and self.recouple_helium_radiation:
                        if (
                            not near_root_acceleration
                            and _coupled_population_near_root_history is not None
                            and _coupled_population_near_root_threshold is not None
                            and undamped_population_change
                            <= _coupled_population_near_root_threshold
                        ):
                            near_root_acceleration = True
                            LOGGER.info(
                                "PG1159 population defect %.4g reached the local "
                                "closure regime; using acceleration depth %d",
                                undamped_population_change,
                                _coupled_population_near_root_history,
                            )
                            if hasattr(coupled_acceleration_history, "restart"):
                                coupled_acceleration_history.restart(
                                    _coupled_population_near_root_history,
                                    retain=True,
                                )
                            elif _coupled_population_near_root_history == 0:
                                coupled_acceleration_history.clear()
                            else:
                                del coupled_acceleration_history[
                                    : -_coupled_population_near_root_history
                                ]
                        acceleration_depth = (
                            _coupled_population_near_root_history
                            if near_root_acceleration
                            else self.coupled_population_acceleration_depth
                        )
                        ion_floor = (self.metal_population_relative_floor if self.metal_ion_population_relative_floor is None
                            else self.metal_ion_population_relative_floor)
                        level_floor = (self.metal_population_relative_floor if self.metal_level_population_relative_floor is None
                            else self.metal_level_population_relative_floor)
                        floor_parts = [np.full(light_metal_state.ion_number_density[element].size, ion_floor)
                            for element in sorted(light_metal_state.ion_number_density)]
                        floor_parts.extend(np.full(level.population_density.size, level_floor)
                            for level in (carbon_level_state, oxygen_level_state) if level is not None)
                        proposed = _coupled_population_update(x_vector, g_vector,
                            previous_helium_state, next_helium_state,
                            coupled_acceleration_history, acceleration_depth,
                            metal_relative_floors=np.concatenate(floor_parts),
                            helium_relative_floor=self.helium_population_relative_floor)
                        if proposed is not None:
                            mixed_vector, helium_state = proposed
                            helium_population_change = _maximum_coupled_helium_population_change(
                                previous_helium_state, helium_state,
                                relative_floor=self.helium_population_relative_floor)
                            helium_population_worst_change = _coupled_helium_population_change_diagnostic(
                                previous_helium_state, helium_state,
                                relative_floor=self.helium_population_relative_floor)
                    (
                        light_metal_state,
                        carbon_level_state,
                        oxygen_level_state,
                    ) = _states_from_population_vector(
                        mixed_vector,
                        next_light_metal_state,
                        next_carbon_level_state,
                        next_oxygen_level_state,
                        metal_state,
                    )
                    if (
                        self.metal_population_acceleration_depth == 0
                        and self.coupled_population_acceleration_depth == 0
                        and population_relative_change_history
                    ):
                        trial_change = _maximum_population_state_change(
                            previous_light_metal_state,
                            previous_carbon_level_state,
                            previous_oxygen_level_state,
                            light_metal_state,
                            carbon_level_state,
                            oxygen_level_state,
                            metal_state,
                            relative_floor=self.metal_population_relative_floor,
                            ion_relative_floor=(
                                self.metal_ion_population_relative_floor
                            ),
                            level_relative_floor=(
                                self.metal_level_population_relative_floor
                            ),
                        )
                        allowed_change = max(
                            2.0 * population_relative_change_history[-1], 0.01
                        )
                        trial_damping = self.metal_population_damping
                        while (
                            trial_change > allowed_change
                            and trial_damping > self.metal_population_minimum_damping
                        ):
                            trial_damping *= 0.5
                            trial_vector = (
                                (1.0 - trial_damping) * x_vector
                                + trial_damping * g_vector
                            )
                            (
                                light_metal_state,
                                carbon_level_state,
                                oxygen_level_state,
                            ) = _states_from_population_vector(
                                trial_vector,
                                next_light_metal_state,
                                next_carbon_level_state,
                                next_oxygen_level_state,
                                metal_state,
                            )
                            trial_change = _maximum_population_state_change(
                                previous_light_metal_state,
                                previous_carbon_level_state,
                                previous_oxygen_level_state,
                                light_metal_state,
                                carbon_level_state,
                                oxygen_level_state,
                                metal_state,
                                relative_floor=(
                                    self.metal_population_relative_floor
                                ),
                                ion_relative_floor=(
                                    self.metal_ion_population_relative_floor
                                ),
                                level_relative_floor=(
                                    self.metal_level_population_relative_floor
                                ),
                            )
                        effective_population_damping = trial_damping
                    if (
                        not self.trace_opacity_in_population_radiation
                        and next_light_metal_state is not None
                    ):
                        passenger_elements = tuple(
                            element
                            for element in self.nlte_metal_elements
                            if element not in {"C", "O"}
                        )
                        if passenger_elements:
                            light_metal_state = _replace_light_metal_elements(
                                light_metal_state,
                                next_light_metal_state,
                                passenger_elements,
                            )
                population_iteration_count = iteration + 1
                population_relative_change = _maximum_population_state_change(
                    previous_light_metal_state,
                    previous_carbon_level_state,
                    previous_oxygen_level_state,
                    light_metal_state,
                    carbon_level_state,
                    oxygen_level_state,
                    metal_state,
                    relative_floor=self.metal_population_relative_floor,
                    ion_relative_floor=(
                        self.metal_ion_population_relative_floor
                    ),
                    level_relative_floor=(
                        self.metal_level_population_relative_floor
                    ),
                )
                metal_only_population_change_history.append(
                    population_relative_change
                )
                population_worst_change = _population_state_change_diagnostic(
                    previous_light_metal_state,
                    previous_carbon_level_state,
                    previous_oxygen_level_state,
                    light_metal_state,
                    carbon_level_state,
                    oxygen_level_state,
                    metal_state,
                    relative_floor=self.metal_population_relative_floor,
                    ion_relative_floor=(
                        self.metal_ion_population_relative_floor
                    ),
                    level_relative_floor=(
                        self.metal_level_population_relative_floor
                    ),
                )
                helium_population_change_history.append(
                    helium_population_change
                )
                population_relative_change = max(
                    population_relative_change,
                    helium_population_change,
                )
                population_relative_change_history.append(
                    population_relative_change
                )
                undamped_population_change = _maximum_population_state_change(
                    previous_light_metal_state, previous_carbon_level_state,
                    previous_oxygen_level_state, next_light_metal_state,
                    next_carbon_level_state, next_oxygen_level_state, metal_state,
                    relative_floor=self.metal_population_relative_floor,
                    ion_relative_floor=self.metal_ion_population_relative_floor,
                    level_relative_floor=self.metal_level_population_relative_floor)
                undamped_population_worst_change = _population_state_change_diagnostic(
                    previous_light_metal_state,
                    previous_carbon_level_state,
                    previous_oxygen_level_state,
                    next_light_metal_state,
                    next_carbon_level_state,
                    next_oxygen_level_state,
                    metal_state,
                    relative_floor=self.metal_population_relative_floor,
                    ion_relative_floor=self.metal_ion_population_relative_floor,
                    level_relative_floor=self.metal_level_population_relative_floor,
                )
                if self.recouple_helium_radiation:
                    undamped_population_change = max(undamped_population_change,
                        _maximum_coupled_helium_population_change(
                            previous_helium_state, next_helium_state,
                            relative_floor=self.helium_population_relative_floor))
                    undamped_helium_population_worst_change = (
                        _coupled_helium_population_change_diagnostic(
                            previous_helium_state,
                            next_helium_state,
                            relative_floor=self.helium_population_relative_floor,
                        )
                    )
                measured_input_available = (
                    (previous_light_metal_state is not None or next_light_metal_state is None)
                    and all(new is None or (old is not None and old.level_key == new.level_key)
                        for old, new in ((previous_carbon_level_state, next_carbon_level_state),
                                         (previous_oxygen_level_state, next_oxygen_level_state)))
                )
                iteration_converged = (
                    population_iteration_count >= self.metal_population_minimum_iterations
                    and undamped_population_change <= self.metal_population_relative_tolerance
                    and measured_input_available
                )
                handoff_requested = False
                if (_stop_iteration is not None and not iteration_converged
                        and measured_input_available and self.metal_population_iterations > 1):
                    # Both the measured rate defect and the proposed EOS
                    # refresh must refer to the same input populations.
                    measured_input = PG1159NLTEState(
                        helium_state=previous_helium_state,
                        metal_state=metal_state,
                        light_metal_state=previous_light_metal_state,
                        carbon_level_state=previous_carbon_level_state,
                        oxygen_level_state=previous_oxygen_level_state,
                        trace_level_states=trace_level_states,
                        metadata={
                            "metal_population_iterations": population_iteration_count,
                            "metal_population_converged": False,
                            "undamped_population_relative_change": undamped_population_change,
                            "metal_population_relative_change": undamped_population_change,
                        },
                    )
                    handoff_requested = _stop_iteration(population_iteration_count, measured_input)
                iteration_budget_exhausted = (
                    population_iteration_count == self.metal_population_iterations
                )
                if (
                    iteration_converged
                    or handoff_requested
                    or iteration_budget_exhausted
                ) and self.metal_population_iterations > 1:
                    # Single-pass callers measure the rate map or its tangent;
                    # they must always receive g(x), even when g(x)-x is small.
                    # The just-measured rate map belongs to the INPUT state.
                    # Return that independently measured state, rather than
                    # extrapolating away from it with an Anderson step whose
                    # size can exceed the rate residual near a weak mode. This
                    # also applies at the iteration limit: the unmeasured final
                    # proposal is not a better restart merely because the
                    # iteration budget ended after it was formed.
                    helium_state = previous_helium_state
                    light_metal_state = previous_light_metal_state
                    carbon_level_state = previous_carbon_level_state
                    oxygen_level_state = previous_oxygen_level_state
                    population_relative_change = undamped_population_change
                    population_relative_change_history[-1] = undamped_population_change
                    population_worst_change = _population_state_change_diagnostic(
                        previous_light_metal_state, previous_carbon_level_state,
                        previous_oxygen_level_state, next_light_metal_state,
                        next_carbon_level_state, next_oxygen_level_state, metal_state,
                        relative_floor=self.metal_population_relative_floor,
                        ion_relative_floor=self.metal_ion_population_relative_floor,
                        level_relative_floor=self.metal_level_population_relative_floor)
                if iteration_callback is not None or _stop_iteration is not None:
                    progress_state = PG1159NLTEState(
                            helium_state=helium_state,
                            metal_state=metal_state,
                            light_metal_state=light_metal_state,
                            carbon_level_state=carbon_level_state,
                            oxygen_level_state=oxygen_level_state,
                            metadata={
                                "metal_population_iterations": (
                                    population_iteration_count
                                ),
                                "metal_population_converged": iteration_converged,
                                "undamped_population_relative_change": undamped_population_change,
                                "metal_population_relative_change": (
                                    population_relative_change
                                ),
                                "metal_population_relative_change_history": tuple(
                                    population_relative_change_history
                                ),
                            "metal_population_worst_change": dict(
                                population_worst_change
                            ),
                            "helium_population_worst_change": dict(
                                helium_population_worst_change
                            ),
                            "undamped_metal_population_worst_change": dict(
                                undamped_population_worst_change
                            ),
                            "undamped_helium_population_worst_change": dict(
                                undamped_helium_population_worst_change
                            ),
                            },
                            trace_level_states=trace_level_states,
                        )
                    if iteration_callback is not None:
                        iteration_callback(population_iteration_count, progress_state)
                    if handoff_requested:
                        # The handed-off state is the measured input above,
                        # still explicitly unfinished until independent checks.
                        break
                if iteration_converged:
                    population_converged = True
                    break
        # The atmosphere atom is term-resolved, while the formal atom is
        # fine-structure resolved.  Losing this map makes the synthesizer
        # interpret unrelated adjacent term indices as the line endpoints;
        # C IV 5801/5812 is especially sensitive to that failure.
        for element, configured_mapping, level_state in (
            ("C", self.carbon_formal_level_mapping, carbon_level_state),
            ("O", self.oxygen_formal_level_mapping, oxygen_level_state),
        ):
            if (
                configured_mapping is not None
                and level_state is not None
                and level_state.formal_level_mapping is None
            ):
                raise RuntimeError(
                    f"internal PG1159 error: {element} formal-level mapping "
                    "was lost during the NLTE solve"
                )
        return PG1159NLTEState(
            radiation_wavelength=rate_wavelength,
            radiation_mean_intensity=rate_mean_intensity,
            radiation_diagonal_lambda=rate_diagonal_lambda,
            radiation_source_function=rate_source_function,
            helium_state=helium_state,
            metal_state=metal_state,
            light_metal_state=light_metal_state,
            carbon_level_state=carbon_level_state,
            oxygen_level_state=oxygen_level_state,
            metadata={
                "model_atom": self.name,
                "helium_model_atom": helium_state.metadata.get("model_atom"),
                "carbon_oxygen_population_model": (
                    "explicit C III-V and O III-VII reduced atoms"
                    if carbon_level_state is not None
                    and oxygen_level_state is not None
                    else (
                        "explicit C III-V levels; NLTE ion stages for "
                        + ", ".join(self.nlte_metal_elements)
                    )
                    if carbon_level_state is not None
                    else (
                        "explicit O III-VII levels; NLTE ion stages for "
                        + ", ".join(self.nlte_metal_elements)
                    )
                    if oxygen_level_state is not None
                    else "NLTE ion stages; LTE excitation within stages"
                    if light_metal_state is not None
                    else "LTE ionization and excitation"
                ),
                "mass_fractions": dict(metal_state.mass_fraction or {}),
                "carbon_oxygen_lte_recovery_error": (
                    light_metal_state.maximum_lte_recovery_error
                    if light_metal_state is not None else 0.0
                ),
                "carbon_bound_bound_transitions": (
                    carbon_level_state.metadata.get("bound_bound_transitions")
                    if carbon_level_state is not None else 0
                ),
                "oxygen_bound_bound_transitions": (
                    oxygen_level_state.metadata.get("bound_bound_transitions")
                    if oxygen_level_state is not None else 0
                ),
                "carbon_population_atom": (
                    self.atomic_database.source
                    if self.carbon_population_atomic_database is None
                    else self.carbon_population_atomic_database.source
                ),
                "oxygen_population_atom": (
                    self.atomic_database.source
                    if self.oxygen_population_atomic_database is None
                    else self.oxygen_population_atomic_database.source
                ),
                "metal_population_iterations": population_iteration_count,
                "metal_population_converged": population_converged,
                "iteration_limit_returned_measured_input": bool(
                    self.metal_population_iterations > 1
                    and population_iteration_count == self.metal_population_iterations
                    and not population_converged
                ),
                "metal_population_relative_change": population_relative_change,
                "undamped_population_relative_change": undamped_population_change,
                "metal_population_relative_tolerance": (
                    self.metal_population_relative_tolerance
                ),
                "population_accuracy_final": (
                    self.atmosphere_population_accuracy_final
                ),
                "metal_population_relative_change_history": tuple(
                    population_relative_change_history
                ),
                "metal_only_population_change_history": tuple(
                    metal_only_population_change_history
                ),
                "helium_population_change_history": tuple(
                    helium_population_change_history
                ),
                "helium_population_worst_change": dict(
                    helium_population_worst_change
                ),
                "helium_population_relative_floor": (
                    self.helium_population_relative_floor
                ),
                "metal_population_worst_change": dict(
                    population_worst_change
                ),
                "undamped_metal_population_worst_change": dict(
                    undamped_population_worst_change
                ),
                "undamped_helium_population_worst_change": dict(
                    undamped_helium_population_worst_change
                ),
                "metal_population_damping": effective_population_damping,
                "metal_population_minimum_damping": (
                    self.metal_population_minimum_damping
                ),
                "metal_population_relative_floor": (
                    self.metal_population_relative_floor
                ),
                "metal_ion_population_relative_floor": (
                    self.metal_population_relative_floor
                    if self.metal_ion_population_relative_floor is None
                    else self.metal_ion_population_relative_floor
                ),
                "metal_level_population_relative_floor": (
                    self.metal_population_relative_floor
                    if self.metal_level_population_relative_floor is None
                    else self.metal_level_population_relative_floor
                ),
                "helium_radiation_recoupled": self.recouple_helium_radiation,
                "helium_population_damping": self.helium_population_damping,
                "coupled_population_acceleration_depth": self.coupled_population_acceleration_depth,
                "recycled_population_secants": recycled_population_secants,
                "coupled_population_acceleration_method": (
                    "weighted-svd-anderson" if self.coupled_population_acceleration_depth
                    and self.recouple_helium_radiation else "none"
                ),
                "coupled_population_near_root_history": (
                    _coupled_population_near_root_history
                ),
                "coupled_population_near_root_threshold": (
                    _coupled_population_near_root_threshold
                ),
                "coupled_population_near_root_activated": near_root_acceleration,
                "population_ali": self.use_population_ali,
                "nlte_ion_stage_elements": tuple(self.nlte_metal_elements),
                "active_ion_stage_ranges": (
                    {}
                    if light_metal_state is None
                    else light_metal_state.metadata.get(
                        "active_stage_ranges", {}
                    )
                ),
                "minimum_active_lte_ion_fraction": (
                    self.minimum_active_lte_ion_fraction
                ),
                "active_ion_stage_margin": self.active_ion_stage_margin,
                "ion_stage_range_overrides": dict(
                    self.ion_stage_range_overrides
                ),
                "trace_opacity_in_population_radiation": (
                    self.trace_opacity_in_population_radiation
                ),
                "population_trace_line_limit": self.maximum_population_trace_lines,
            },
            trace_level_states=trace_level_states,
        )

    def transfer_coefficients(
        self,
        atmosphere: Atmosphere,
        wavelength_angstrom: FloatArray,
        state: Any,
    ) -> NLTETransferCoefficients:
        if not isinstance(state, PG1159NLTEState):
            raise TypeError("PG1159NLTEModel requires a PG1159NLTEState")
        base = self.helium_model.transfer_coefficients(
            atmosphere, wavelength_angstrom, state.helium_state
        )
        ion_departure = _ion_departures_with_explicit_stage_closure(
            state.light_metal_state,
            state.carbon_level_state,
            state.oxygen_level_state,
            *state.trace_level_states.values(),
        )
        combined_level_departure = {}
        if state.carbon_level_state is not None:
            combined_level_departure.update(
                state.carbon_level_state.level_departure_coefficient
            )
        if state.oxygen_level_state is not None:
            combined_level_departure.update(
                state.oxygen_level_state.level_departure_coefficient
            )
        for trace_state in state.trace_level_states.values():
            combined_level_departure.update(
                trace_state.level_departure_coefficient
            )
        level_departure = (
            combined_level_departure or None
            if self.use_explicit_metal_line_source_functions else None
        )
        planck = planck_lambda_angstrom(
            wavelength_angstrom[:, np.newaxis],
            atmosphere.temperature[np.newaxis, :],
        )
        if ion_departure is None:
            metal_line_absorption = metal_line_mass_absorption_coefficient(
                atmosphere,
                wavelength_angstrom,
                self.atomic_database,
                state.metal_state,
                minimum_oscillator_strength=self.minimum_metal_oscillator_strength,
                maximum_lines=self.maximum_metal_lines,
            )
            metal_line_emissivity = metal_line_absorption * planck
        else:
            formal_transition_keys = _formal_spectrum_transition_keys(
                self.atomic_database,
                self.nlte_metal_elements,
                {
                    "C": self.carbon_population_atomic_database,
                    "O": self.oxygen_population_atomic_database,
                },
                {
                    "C": self.carbon_formal_level_mapping,
                    "O": self.oxygen_formal_level_mapping,
                },
                {
                    "C": self.carbon_levels_per_charge,
                    "O": self.oxygen_levels_per_charge,
                },
                {
                    "C": self.carbon_formal_lte_parent_mapping,
                    "O": self.oxygen_formal_lte_parent_mapping,
                },
            )
            metal_line_absorption, metal_line_emissivity = (
                hot_metal_line_nlte_coefficients(
                    atmosphere,
                    wavelength_angstrom,
                    self.atomic_database,
                    state.metal_state,
                    ion_departure,
                    level_departure_coefficient=level_departure,
                    minimum_oscillator_strength=(
                        self.minimum_metal_oscillator_strength
                    ),
                    maximum_lines=self.maximum_metal_lines,
                    include_static_linear_stark=(
                        self.include_static_linear_stark
                    ),
                    static_linear_stark_frequency_scales=(
                        self.static_linear_stark_frequency_scales
                    ),
                    tabulated_electron_stark_width_scale=(
                        self.tabulated_electron_stark_width_scale
                    ),
                    include_semiclassical_ovi_stark_widths=(
                        self.include_semiclassical_ovi_stark_widths
                    ),
                    include_ovi_high_series_ion_dephasing=(
                        self.include_ovi_high_series_ion_dephasing
                    ),
                    extend_strong_uv_resonance_wings=(
                        self.extend_strong_uv_resonance_wings
                    ),
                    strong_uv_resonance_core_optical_depth=(
                        self.strong_uv_resonance_core_optical_depth
                    ),
                    elements=self.nlte_metal_elements,
                    transition_keys=formal_transition_keys,
                    include_ion_dynamic_stark_core=(
                        self.include_ion_dynamic_stark_core
                    ),
                    hydrogenic_linear_stark_components=(
                        self.hydrogenic_linear_stark_components
                    ),
                )
            )
            if set(state.metal_state.log_number_abundance) - set(
                self.nlte_metal_elements
            ):
                trace_line_absorption = metal_line_mass_absorption_coefficient(
                    atmosphere,
                    wavelength_angstrom,
                    self.atomic_database,
                    state.metal_state,
                    minimum_oscillator_strength=(
                        self.minimum_metal_oscillator_strength
                    ),
                    maximum_lines=self.maximum_metal_lines,
                    excluded_elements=self.nlte_metal_elements,
                )
                metal_line_absorption += trace_line_absorption
                metal_line_emissivity += trace_line_absorption * planck
        metal_absorption = metal_line_absorption
        metal_emissivity = metal_line_emissivity
        metal_free_free = light_metal_free_free_mass_absorption_coefficient(
            atmosphere,
            wavelength_angstrom,
            state.metal_state,
            ion_departure,
        )
        metal_absorption = metal_absorption + metal_free_free
        metal_emissivity = metal_emissivity + metal_free_free * planck
        if self.photoionization_database is not None:
            if ion_departure is None:
                metal_bound_free = metal_bound_free_mass_absorption_coefficient(
                    atmosphere,
                    wavelength_angstrom,
                    self.atomic_database,
                    state.metal_state,
                    self.photoionization_database,
                )
                metal_bound_free_emissivity = metal_bound_free * planck
            else:
                # Use the same level-resolved compact C/O atoms in the
                # formal transfer that supplied bound-free opacity to the
                # statistical-equilibrium radiation field.  Previously this
                # path omitted the TMAD threshold tables and silently fell
                # back to Verner ground-state fits.  The populations then saw
                # an excited-level UV continuum which was absent from both
                # the atmosphere energy balance and emergent spectrum.
                metal_bound_free = np.zeros_like(planck)
                metal_bound_free_emissivity = np.zeros_like(planck)
                for element, population_atom, level_state in (
                    (
                        "C",
                        self.atomic_database
                        if self.carbon_population_atomic_database is None
                        else self.carbon_population_atomic_database,
                        state.carbon_level_state,
                    ),
                    (
                        "O",
                        self.atomic_database
                        if self.oxygen_population_atomic_database is None
                        else self.oxygen_population_atomic_database,
                        state.oxygen_level_state,
                    ),
                ):
                    if element not in self.nlte_metal_elements:
                        continue
                    local_level_departure = (
                        None
                        if level_state is None
                        else (
                            level_state.level_departure_coefficient
                            if population_atom is self.atomic_database
                            or level_state.population_level_departure_coefficient
                            is None
                            else level_state.population_level_departure_coefficient
                        )
                    )
                    local_absorption, local_emissivity = (
                        light_metal_bound_free_nlte_coefficients(
                            atmosphere,
                            wavelength_angstrom,
                            population_atom,
                            state.metal_state,
                            self.photoionization_database,
                            ion_departure,
                            level_departure_coefficient=local_level_departure,
                            elements=(element,),
                            photoionization_threshold_data=(
                                self.carbon_photoionization_threshold_data
                                if element == "C"
                                else self.oxygen_photoionization_threshold_data
                            ),
                            levels_per_charge=(
                                self.carbon_levels_per_charge
                                if element == "C"
                                else self.oxygen_levels_per_charge
                            ),
                            continuum_parent_mapping=(
                                self.carbon_continuum_parent_mapping
                                if element == "C"
                                else self.oxygen_continuum_parent_mapping
                            ),
                        )
                    )
                    metal_bound_free += local_absorption
                    metal_bound_free_emissivity += local_emissivity
                ion_only_elements = tuple(
                    element
                    for element in self.nlte_metal_elements
                    if element not in {"C", "O"}
                    and element in state.metal_state.log_number_abundance
                )
                if ion_only_elements:
                    local_absorption, local_emissivity = (
                        light_metal_bound_free_nlte_coefficients(
                            atmosphere,
                            wavelength_angstrom,
                            self.atomic_database,
                            state.metal_state,
                            self.photoionization_database,
                            ion_departure,
                            level_departure_coefficient=level_departure,
                            elements=ion_only_elements,
                        )
                    )
                    metal_bound_free += local_absorption
                    metal_bound_free_emissivity += local_emissivity
                if set(state.metal_state.log_number_abundance) - set(
                    self.nlte_metal_elements
                ):
                    trace_bound_free = (
                        metal_bound_free_mass_absorption_coefficient(
                            atmosphere,
                            wavelength_angstrom,
                            self.atomic_database,
                            state.metal_state,
                            self.photoionization_database,
                            excluded_elements=self.nlte_metal_elements,
                        )
                    )
                    metal_bound_free += trace_bound_free
                    metal_bound_free_emissivity += trace_bound_free * planck
            metal_absorption = metal_absorption + metal_bound_free
            metal_emissivity = metal_emissivity + metal_bound_free_emissivity
        metadata = dict(base.metadata)
        metadata.update(
            {
                **state.metadata,
                "metal_lines": self.atomic_database.source,
                "metal_bound_free": (
                    self.photoionization_database.source
                    if self.photoionization_database is not None else "disabled"
                ),
                "metal_free_free": (
                    "charge-weighted C/O ionic bremsstrahlung with van Hoof Gaunt factors"
                ),
                "metal_emissivity": (
                    "departure-dependent bound-bound/bound-free source"
                    if ion_departure is not None
                    else "LTE Kirchhoff source"
                ),
            }
        )
        return NLTETransferCoefficients(
            wavelength_angstrom=np.asarray(base.wavelength_angstrom),
            true_absorption=np.ascontiguousarray(
                base.true_absorption + metal_absorption
            ),
            thermal_emissivity=np.ascontiguousarray(
                base.thermal_emissivity + metal_emissivity
            ),
            scattering=np.asarray(base.scattering),
            metadata=metadata,
        )


def synthesize_pg1159_nlte_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    state: PG1159NLTEState,
    model: PG1159NLTEModel,
    *,
    n_angle: int = 4,
    scattering_iterations: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize a fixed-structure PG 1159 spectrum from solved populations."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if scattering_iterations < 1:
        raise ValueError("scattering_iterations must be positive")
    coefficients = model.transfer_coefficients(atmosphere, wavelength, state)
    total = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    source = np.ascontiguousarray(
        (coefficients.thermal_emissivity + coefficients.scattering * planck) / total
    )
    for _ in range(scattering_iterations):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total
        )
    flux = emergent_flux(optical_depth, source, n_angle=n_angle, backend=backend)
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            **coefficients.metadata,
            "composition": "homogeneous-helium-carbon-oxygen",
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "transfer": "reduced He/C/O NLTE plus coherent scattering",
            "n_angle": int(n_angle),
        },
    )
