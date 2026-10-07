"""Experimental trace-metal NLTE line formation on a fixed hydrogen atmosphere.

The caller supplies H/He absorption, emissivity and scattering on any requested
vacuum wavelength grid. These can come directly from HotNLTEModel and its
matched population state. Neither the host populations nor T, rho or ne are
updated. Metal opacity DOES enter the radiation field used for metal rates.

This first, deliberately low-level interface uses compact Stout fine-structure
atoms, Verner ground continua, Kramers excited continua and approximate electron
collisions from light_metal_nlte. Omitted levels remain inert LTE reservoirs;
only transitions closed within the explicit atom contribute line opacity.
It is a development calculation, not a validated abundance-fitting preset.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping
import logging
import warnings

import numpy as np

from .atmosphere import Atmosphere
from ._pg1159_acceleration import population_update
from .constants import BOLTZMANN, ELECTRON_MASS, PLANCK, LIGHT_SPEED
from .hot_nlte import transfer_field
from .light_metal_nlte import (
    ReducedLightMetalLevelState,
    hot_metal_line_nlte_coefficients,
    light_metal_bound_free_nlte_coefficients,
    metal_thermal_arrays,
    reduced_light_metal_wavelength,
    solve_reduced_light_metal_levels_nlte,
)
from .metals import (
    ATOMIC_MASS_U, ATOMIC_NUMBER, EV_TO_ERG, AtomicDatabase, MetalLTEState,
    VernerPhotoionizationDatabase, read_pg1159_atomic_database,
    read_verner_photoionization_database,
)
from .models.common import ModelData
from .nlte_core import NLTETransferCoefficients
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import radiation_field
from .spectrum import Spectrum


_LOGGER = logging.getLogger(__name__)


# The population criterion tests relative level changes only where they can
# matter: levels holding at least POPULATION_DEFECT_FLOOR of the element (at
# each depth), in layers at Rosseland depth >= POPULATION_DEFECT_MINIMUM_TAU.
# With the former 1e-12 floor and every layer tested, 80-depth ZTF J1539
# models stalled at defects of 0.05-0.1: the limiting levels held 1e-13 to
# 1e-6 of the carbon (mostly the C VI ground) in the outermost layers, where
# they oscillate under acceleration but form no observed line. Re-evaluated
# with these limits, those stalled states pass while genuinely unconverged
# photospheric levels (e.g. Si VI holding 1.4e-5 of Si) are still flagged.
POPULATION_DEFECT_FLOOR = 1e-6
POPULATION_DEFECT_MINIMUM_TAU = 1e-6

# Default largest relative change of the emergent flux allowed by the
# undamped update under convergence_criterion="flux".
FLUX_TOLERANCE = 3e-3


class TraceMetalConvergenceWarning(RuntimeWarning):
    """The undamped metal population defect exceeds the requested tolerance."""


@dataclass(frozen=True)
class TraceMetalResult:
    spectrum: Spectrum
    background_spectrum: Spectrum
    populations: Mapping[str, ReducedLightMetalLevelState]
    lte_reference: MetalLTEState | None
    converged: bool
    iterations: int
    population_defect: float
    metadata: dict


def synthesize_dao_trace_metals(host, abundances, wavelength, *, data=None,
                              require_host_convergence=True, **options):
    """Use a compute_dao result as the fixed H/He NLTE background.

    Use the same ModelData as for the host. An explicit convergence opt-out
    allows smoke diagnostics without implying atmosphere validation.
    """
    from .models.hot import DAOConfig, _model_from_config
    from .hot_nlte import HotPopulationState
    from .multilevel_nlte import _validate_state_atmosphere
    if (not isinstance(host.config, DAOConfig)
            or not isinstance(host.population_state, HotPopulationState)
            or host.population_state.hydrogen is None):
        raise ValueError("a DAO result with H/He NLTE populations is required")
    host_status = host.metadata.get("atmosphere_convergence_status", "unknown")
    if require_host_convergence and (host_status != "converged" or not host.population_state.converged):
        raise ValueError("host must have converged atmosphere and H/He populations")
    _validate_state_atmosphere(host.atmosphere, host.population_state.hydrogen)
    model = _model_from_config(host.config, ModelData.default() if data is None else data)
    options.setdefault("n_angle", model.n_angle)
    result = solve_hot_trace_metals(
        host.atmosphere,
        lambda wave: model.transfer_coefficients(host.atmosphere, wave, host.population_state),
        abundances, wavelength, data=data, **options)
    result.metadata.update(host_atmosphere_convergence=host_status,
                           host_effective_temperature=host.config.effective_temperature,
                           host_logg=host.config.logg,
                           host_log_hydrogen_to_helium=host.config.log_hydrogen_to_helium)
    result.spectrum.metadata.update(result.metadata)
    return result


def fixed_electron_metal_reference(atmosphere, database, abundances):
    """Saha reference at the *unchanged* host ne and hydrogen nuclei density.

    Calling metal_lte_state here would reclose the host EOS and make its LTE
    reference inconsistent with the ne used by the statistical-equilibrium
    rates. This reference is ideal for metals and includes the full ion ladder.
    """
    if atmosphere.hydrogen_lte_state is None:
        raise ValueError("trace abundances require a hydrogen host")
    temperature = np.asarray(atmosphere.temperature)
    electrons = np.asarray(atmosphere.electron_density)
    if np.any(~np.isfinite(electrons)) or np.any(electrons <= 0):
        raise ValueError("host electron density must be finite and positive")
    nh = atmosphere.hydrogen_lte_state.hydrogen_nuclei_density
    partitions, densities, ions = {}, {}, {}
    metal_electrons = np.zeros_like(electrons)
    translation = 1.5 * np.log(2 * np.pi * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2)
    for element, abundance in abundances.items():
        stages = database.ion_stages(element)
        densities[element] = nh * 10.0**abundance
        for ion in stages:
            partitions[element, ion.charge] = ion.partition_function(temperature)
        logs = [np.zeros_like(temperature)]
        for lower, upper in zip(stages[:-1], stages[1:]):
            if lower.ionization_energy_ev is None:
                raise ValueError(f"missing ionization energy for {element} {lower.charge}")
            logs.append(logs[-1] + np.log(2.) + translation - np.log(electrons)
                        + np.log(partitions[element, upper.charge] / partitions[element, lower.charge])
                        - lower.ionization_energy_ev * EV_TO_ERG / (BOLTZMANN * temperature))
        logs = np.asarray(logs)
        fractions = np.exp(logs - np.max(logs, axis=0))
        fractions /= fractions.sum(axis=0)
        ions[element] = densities[element] * fractions
        metal_electrons += np.sum(np.arange(len(stages))[:, None] * ions[element], axis=0)
    return MetalLTEState("H", dict(abundances), densities, ions, partitions,
                         electrons.copy(), metal_electrons)


def _wavelength(values):
    wave = np.asarray(values, dtype=float)
    if (wave.ndim != 1 or wave.size < 2 or np.any(~np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError("wavelength must be finite, positive and strictly increasing")
    return wave


def _atom_selection(database, element, counts):
    charges = sorted(counts)
    if (any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) for q in charges)
            or len(charges) < 2 or charges != list(range(charges[0], charges[-1] + 1))):
        raise ValueError("explicit atoms require at least two consecutive charge stages")
    selected = {}
    for charge in charges:
        ion = database.ions.get((element, charge))
        count = counts[charge]
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or ion is None or not 1 <= count <= len(ion.levels)):
            raise ValueError(f"invalid explicit level count for {element} charge {charge}")
        selected[charge] = {level.index for level in sorted(
            ion.levels, key=lambda level: level.energy_wavenumber)[:count]}
    transitions = frozenset(
        (element, charge, line.lower_index, line.upper_index)
        for charge in charges for line in database.ions[element, charge].transitions
        if line.lower_index in selected[charge] and line.upper_index in selected[charge]
    )
    # The highest explicit ion is a continuum sink, as in the rate solver.
    bf_counts = {ion.charge: (counts.get(ion.charge, 0) if ion.charge < charges[-1] else 0)
                 for ion in database.ion_stages(element)}
    return transitions, bf_counts


def _cross_element_overlaps(line_wavelength, velocity_kms):
    """Keys of lines whose centre is within ``velocity_kms`` of another element's line."""
    items = sorted(line_wavelength.items(), key=lambda item: item[1])
    centres = np.array([wavelength for _, wavelength in items])
    elements = np.array([key[0] for key, _ in items])
    half = centres * velocity_kms / (LIGHT_SPEED * 1e-5)
    lower = np.searchsorted(centres, centres - half)
    upper = np.searchsorted(centres, centres + half, side='right')
    return frozenset(key for index, (key, _) in enumerate(items)
                     if np.any(elements[lower[index]:upper[index]] != key[0]))


def _diagonal_lambda_operator(atmosphere, coefficients, n_angle):
    """Diagonal Lambda* of the current extinction (two-stream sweep).

    Used only to precondition the line rates; the rates themselves use the
    exact mass-coordinate transfer field.
    """
    total = np.maximum(coefficients.true_absorption + coefficients.scattering, 1e-300)
    depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    increment = np.diff(depth, axis=-1)
    floor = 1e-14 * np.maximum(abs(depth[..., 1:]), 1e-300)
    if np.any(increment <= floor):
        depth = np.concatenate((depth[..., :1], depth[..., :1]
                                + np.cumsum(np.maximum(increment, floor), axis=-1)), axis=-1)
    source = np.ascontiguousarray(coefficients.thermal_emissivity / total)
    return radiation_field(depth, source, n_angle=n_angle,
                           calculate_diagonal_lambda=True).diagonal_lambda


def trace_metal_coefficients(atmosphere, wavelength, base, database, photoionization_database,
                             reference, selection, departures, elements, *, thresholds=None,
                             line_center_opacity=None, metadata=None, profile_block_tolerance=0.0):
    """Host coefficients plus explicit-atom metal line/bound-free opacity and emissivity.

    ``selection`` maps element -> (closed transition keys, bound-free level counts)
    as returned by :func:`_atom_selection`; ``departures`` maps element ->
    level departure coefficients relative to ``reference`` (missing element:
    LTE). Evaluating the same departures with the reference of another
    atmosphere gives the frozen-departure coefficients at that temperature.
    ``profile_block_tolerance`` selects the line-profile block quadrature of
    :func:`hot_metal_line_nlte_coefficients` (0: exact point accumulation).
    """
    thresholds = {} if thresholds is None else thresholds
    absorption = base.true_absorption.copy()
    emission = base.thermal_emissivity.copy()
    unity = {(e, ion.charge): np.ones(atmosphere.n_depth)
             for e in elements for ion in database.ion_stages(e)}
    thermal = metal_thermal_arrays(atmosphere, wavelength)
    for e in elements:
        departure = departures.get(e)
        lines, bound_counts = selection[e]
        a, j = hot_metal_line_nlte_coefficients(
            atmosphere, wavelength, database, reference, unity,
            level_departure_coefficient=departure, elements=(e,),
            transition_keys=lines, minimum_oscillator_strength=0., maximum_lines=None,
            include_static_linear_stark=False, retain_inverted_emissivity=True,
            line_center_opacity=line_center_opacity,
            profile_block_tolerance=profile_block_tolerance, thermal_arrays=thermal)
        b, k = light_metal_bound_free_nlte_coefficients(
            atmosphere, wavelength, database, reference, photoionization_database, unity,
            elements=(e,), level_departure_coefficient=departure,
            levels_per_charge=bound_counts, include_explicit_kramers=True,
            photoionization_threshold_data=thresholds.get(e), kramers_cumulative=True,
            thermal_arrays=thermal)
        absorption += a + b
        emission += j + k
    return NLTETransferCoefficients(wavelength, absorption, emission, base.scattering,
                                    {} if metadata is None else metadata)


def _damped(previous, proposed, damping):
    if previous is None:
        old = proposed.lte_population_density
    else:
        old = previous.population_density
    population = (1 - damping) * old + damping * proposed.population_density
    return _with_population(proposed, population)


def _relaxed(previous, proposed, damping):
    """Damped step toward the proposed populations, at least halfway in log.

    The arithmetic mean moves a rising population most of the way to its
    target, but a population that must fall by many decades only halves per
    step (17 iterations for 5 decades).  Falling populations therefore use
    the geometric mean, i.e. the same fraction of the way in log as the
    log-space Anderson mixing.  Each element's represented particles are
    restored afterwards. The fixed point is unchanged.
    """
    old = previous.population_density
    new = proposed.population_density
    population = np.where(new >= old, (1 - damping) * old + damping * new,
                          old ** (1 - damping) * new ** damping)
    population *= proposed.lte_population_density.sum(axis=0) / population.sum(axis=0)
    return _with_population(proposed, population)


def _with_population(state, population):
    departures = {key: population[i] / np.maximum(state.lte_population_density[i], 1e-300)
                  for i, key in enumerate(state.level_key)}
    return replace(state, population_density=population,
                   level_departure_coefficient=departures,
                   population_level_departure_coefficient=departures)


def _initial_state(atmosphere, database, reference, element, counts, previous):
    """Map a population guess to the requested atom; never inherit convergence."""
    keys, rows = [], []
    for charge in sorted(counts):
        ion = database.ions[element, charge]
        for level in sorted(ion.levels, key=lambda x: x.energy_wavenumber)[:counts[charge]]:
            keys.append((element, charge, level.index))
            rows.append(reference.ion_number_density[element][charge] * level.statistical_weight
                * np.exp(-level.energy_wavenumber*PLANCK*LIGHT_SPEED/(BOLTZMANN*atmosphere.temperature))
                / reference.partition_function[element, charge])
    lte = np.asarray(rows)
    if previous is None:
        state = ReducedLightMetalLevelState(element,tuple(keys),lte,lte,{},0.,
                                            {"initial_guess_only":True,"initial_guess":"LTE"})
        return _with_population(state,lte.copy())
    if (previous.element != element or previous.population_density.shape != previous.lte_population_density.shape
            or previous.population_density.shape != (len(previous.level_key),atmosphere.n_depth)
            or any(np.any(~np.isfinite(a)) or np.any(a<0) for a in
                   (previous.population_density,previous.lte_population_density))):
        raise ValueError('invalid initial population guess')
    departures = {key: previous.population_density[i]/np.maximum(previous.lte_population_density[i],1e-300)
                  for i,key in enumerate(previous.level_key)}
    population = lte*np.asarray([departures.get(key,np.ones(atmosphere.n_depth)) for key in keys])
    if np.any(population.sum(axis=0)<=0):
        raise ValueError('initial populations must contain represented particles')
    population *= lte.sum(axis=0)/population.sum(axis=0)
    state = ReducedLightMetalLevelState(element,tuple(keys),lte,lte,{},0.,
                                        {'initial_guess_only':True})
    return _with_population(state,population)


def solve_hot_trace_metals(
    atmosphere: Atmosphere,
    background: Callable[[np.ndarray], NLTETransferCoefficients],
    abundances: Mapping[str, float],
    wavelength,
    *,
    data: ModelData | None = None,
    atomic_database: AtomicDatabase | None = None,
    photoionization_database: VernerPhotoionizationDatabase | None = None,
    levels_per_charge: Mapping[str, Mapping[int, int]] | None = None,
    population_wavelength=None,
    initial_populations=None,
    photoionization_threshold_data=None,
    collision_data=None,
    total_recombination=None,
    accelerated_lambda: bool = False,
    mali_overlap_velocity: float | None = None,
    convergence_criterion: str = "population",
    profile_block_tolerance: float = 1e-4,
    opacity_check_gate: float | None = 3.0,
    flux_wavelength_range: tuple[float, float] = (900.0, 1.0e5),
    maximum_iterations: int = 80,
    tolerance: float | None = None,
    damping: float = 0.5,
    acceleration_depth: int = 6,
    n_angle: int = 3,
    require_convergence: bool = True,
    iteration_callback=None,
    state_callback=None,
    trace_mass_limit: float = 1e-3,
) -> TraceMetalResult:
    """Iterate explicit trace-metal statistical equilibrium and combined transfer.

    Abundances are log10 N(element)/N(H). No abundances returns precisely the
    supplied background. ``converged`` tests the UNDAMPED fixed-point defect (relative population
    changes of levels above ``POPULATION_DEFECT_FLOOR`` of the element, in
    layers at Rosseland depth >= ``POPULATION_DEFECT_MINIMUM_TAU``);
    it does not certify the atmosphere, atomic completeness or an observed fit.
    A population grid, if supplied, must cover all edges/lines of the chosen
    atoms; the generated grid is preferable for actual development runs.
    ``initial_populations`` supplies guesses, including from smaller atoms;
    shared departures are mapped and represented particles re-normalized.
    Elements absent from a partial initial guess start from LTE. Additional
    elements beyond C/Si require explicit stage sizes and complete atomic data.
    ``state_callback(iteration, defect, states, synthesize)`` sees the evaluated
    state and can synthesize its spectrum lazily. Treat states as read-only.
    Returning True stops exploration; it does not change convergence criteria.
    Optional per-element TLUSTY/OP tables enter both rates and emissivity.
    ``collision_data`` supplies explicit excitation strengths/rates keyed by
    (element, charge, lower level, upper level), including collision-only
    links. Missing pairs retain the existing approximate prescription.
    ``total_recombination`` maps element -> {recombining charge: alpha(T)}
    with published total recombination coefficients; any excess over the
    explicit atom's own radiative recombination is added (with its LTE
    inverse) as recombination to omitted levels cascading to the ground.
    ``accelerated_lambda`` preconditions the line rates with the diagonal
    approximate lambda operator of the current radiation field (MALI, as in
    the PG 1159 populations). It changes the iteration path, not the fixed
    point or the convergence test.
    ``mali_overlap_velocity`` (km/s; default None: off) withholds that
    preconditioning from every line whose centre lies within this velocity of
    a line of a *different* element.  There the field is shared with the other
    species' source function, so a single-line diagonal operator over-corrects:
    S III 702.78/702.82 on O III 702.84 made a 30 kK, log g 5.3 sdB model
    flip-flop.  Those lines take the ordinary Lambda step; the fixed point is
    unchanged.
    ``convergence_criterion="opacity"`` tests the undamped fixed-point
    residual in the quantities that enter the transfer: the change in metal
    absorption/emissivity produced by the proposed populations, relative to the
    total (host + metal) extinction and emission at every population
    wavelength and depth. The population defect is still recorded. The default
    ("population") keeps the relative-population test.
    The undamped opacity residual needs a second full coefficient
    evaluation.  With ``opacity_check_gate`` set, it is evaluated only once
    the free iterate-to-iterate change of the same opacity measure falls
    below ``opacity_check_gate * tolerance`` (and in the last allowed
    iteration); convergence is still declared only on the undamped residual.
    ``None`` evaluates it every iteration.
    ``convergence_criterion="flux"`` tests the observable instead: the
    undamped proposed populations may change the emergent flux by at most
    ``tolerance`` (relative; default ``FLUX_TOLERANCE`` = 3e-3, otherwise
    1e-3) at every population wavelength inside ``flux_wavelength_range``
    (Angstrom). The same gate applies, using the
    free iterate-to-iterate change of that flux. Populations of levels that
    leave the emergent spectrum unchanged may then still be drifting; the
    opacity criterion remains available for such studies.
    ``profile_block_tolerance`` is the relative accuracy of the linear-profile
    block quadrature used for the metal line opacity on the population grid
    and for the profile-averaged rate integrals (see
    :func:`hot_metal_line_nlte_coefficients`).  Deep-layer Fe/Ni lines are
    Stark broadened to ~1 A, and their windows (up to 10% of the wavelength)
    otherwise cost ~1e10 profile evaluations per iteration.  Upward and
    downward rates share the quadrature (exact detailed balance).  On the
    G191-B2B nine-element model the default 1e-4 changes the rate-equation
    populations by <4e-7 relative and halves the rate-solve time. The
    emergent spectrum is always computed exactly; 0 makes the iteration
    exact as well.
    """
    wave = _wavelength(wavelength)
    if tolerance is None:
        tolerance = FLUX_TOLERANCE if convergence_criterion == "flux" else 1e-3
    if (isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int)
            or maximum_iterations < 1 or not np.isfinite(tolerance) or not 0 < tolerance < 1
            or not np.isfinite(damping) or not 0 < damping <= 1
            or isinstance(acceleration_depth, bool) or not isinstance(acceleration_depth, int)
            or acceleration_depth < 0
            or (mali_overlap_velocity is not None and not (np.isfinite(mali_overlap_velocity)
                                                         and mali_overlap_velocity > 0))
            or isinstance(n_angle, bool) or not isinstance(n_angle, int) or n_angle < 1):
        raise ValueError("invalid iteration, tolerance, damping or angular settings")
    if set(abundances) - set(ATOMIC_NUMBER) or any(
            not np.isfinite(value) or value > 0 for value in abundances.values()):
        raise ValueError("provide supported elements with finite log10 abundances relative to H, at most zero")
    metadata = dict(experimental=True, atmosphere_recomputed=False,
                    host_populations_updated=False, electron_density_updated=False,
                    observationally_validated=False, metal_free_free_included=False,
                    abundance_convention="log10 N(Z)/N(H)",
                    atom="compact Stout; Verner ground/Kramers excited continua; approximate collisions",
                    omitted_levels="inert LTE reservoir; no omitted-level line opacity")
    base_final = background(wave)
    if not np.array_equal(base_final.wavelength_angstrom, wave):
        raise ValueError("background returned a different wavelength grid")
    _, base_field, _ = transfer_field(atmosphere, base_final, n_angle=n_angle)
    spectrum_meta = dict(flux_unit="erg s^-1 cm^-2 Angstrom^-1",
                         flux_convention="surface F_lambda", wavelength_medium="vacuum")
    base_spectrum = Spectrum(wave, base_field.interface_flux[:, 0], spectrum_meta)
    if not abundances:
        return TraceMetalResult(base_spectrum, base_spectrum, {}, None, True, 0, 0., metadata)
    data = ModelData.default() if data is None else data
    database = atomic_database if atomic_database is not None else read_pg1159_atomic_database(
        data.stout, elements=tuple(abundances))
    photo = photoionization_database if photoionization_database is not None else read_verner_photoionization_database(
        data.verner_photoionization, elements=tuple(abundances), require_all_elements=True)
    reference = fixed_electron_metal_reference(atmosphere, database, abundances)
    mass_ratio = sum(reference.element_number_density[e] * ATOMIC_MASS_U[e] * 1.66053906660e-24
                     for e in abundances) / atmosphere.mass_density
    charge_bound = sum(reference.element_number_density[e] * ATOMIC_NUMBER[e]
                       for e in abundances) / atmosphere.electron_density
    metadata.update(maximum_trace_mass_ratio=float(np.max(mass_ratio)),
                    maximum_trace_electron_fraction_bound=float(np.max(charge_bound)))
    # ``trace_mass_limit`` bounds the metal mass fraction neglected by the fixed
    # host (1e-3 suits white dwarfs; solar-like sdB C/N/O is ~1e-3 to 1e-2).
    if np.max(mass_ratio) > trace_mass_limit or np.max(charge_bound) > 1e-2:
        raise ValueError(f"mixture exceeds fixed-background trace limits (mass {trace_mass_limit:g}; electrons 1e-2)")
    defaults = {"C": {2: 20, 3: 30, 4: 1}, "Si": {2: 30, 3: 23, 4: 1}}
    if levels_per_charge is None and set(abundances) - set(defaults):
        raise ValueError("additional elements require explicit levels_per_charge")
    if levels_per_charge is not None and set(levels_per_charge) != set(abundances):
        raise ValueError("levels_per_charge must match the abundance elements")
    counts = {e: dict((defaults if levels_per_charge is None else levels_per_charge)[e]) for e in abundances}
    thresholds = {} if photoionization_threshold_data is None else photoionization_threshold_data
    if set(thresholds)-set(abundances) or (initial_populations is not None and set(initial_populations)-set(abundances)):
        raise ValueError('initial populations / photoionization tables must match abundance elements')
    selection = {e: _atom_selection(database, e, counts[e]) for e in abundances}
    for e in abundances:
        for q in sorted(counts[e])[:-1]:
            if (e, q) not in photo.fits:
                raise ValueError(f"missing ground-state photoionization data for {e} charge {q}")
    metadata["levels_per_charge"] = counts
    metadata['initial_population_guess_supplied'] = initial_populations is not None
    metadata['tabulated_photoionization_stages'] = {e:sorted(t) for e,t in thresholds.items()}
    if collision_data is not None and any(key[0] not in abundances for key in collision_data):
        raise ValueError('collision data must belong to the abundance elements')
    if total_recombination is not None and set(total_recombination) - set(abundances):
        raise ValueError('total recombination data must belong to the abundance elements')
    metadata['total_recombination_stages'] = {e: sorted(int(q) for q in v)
                                              for e, v in (total_recombination or {}).items()}
    metadata['explicit_collision_records'] = len(collision_data or {})
    metadata['explicit_collision_sources'] = sorted({str(getattr(r,'source','user supplied'))
        for r in (collision_data or {}).values()})
    if thresholds:
        metadata['atom'] = 'Stout; selected TLUSTY/OP photoionization with Verner/Kramers fallbacks; approximate collisions'
    if population_wavelength is None:
        pop_wave = np.unique(np.concatenate([
            reduced_light_metal_wavelength(database, e, counts[e],
                photoionization_threshold_data=thresholds.get(e)) for e in abundances]))
    else:
        pop_wave = _wavelength(population_wavelength)
    base_pop = background(pop_wave)
    if not np.array_equal(base_pop.wavelength_angstrom, pop_wave):
        raise ValueError("background returned a different population wavelength grid")
    unity = {(e, ion.charge): np.ones(atmosphere.n_depth)
             for e in abundances for ion in database.ion_stages(e)}

    if not 0.0 <= profile_block_tolerance < 1.0:
        raise ValueError("profile_block_tolerance must be in [0, 1)")
    if opacity_check_gate is not None and not opacity_check_gate > 0.0:
        raise ValueError("opacity_check_gate must be positive or None")
    metadata['opacity_check_gate'] = opacity_check_gate
    metadata['profile_block_tolerance'] = float(profile_block_tolerance)

    def coefficients(grid, base, states, centers=None, block_tolerance=0.0):
        return trace_metal_coefficients(
            atmosphere, grid, base, database, photo, reference, selection,
            {e: states[e].level_departure_coefficient for e in states}, abundances,
            thresholds=thresholds, line_center_opacity=centers, metadata=metadata,
            profile_block_tolerance=block_tolerance)

    line_wavelength = {
        (e, charge, line.lower_index, line.upper_index): line.wavelength_vacuum_angstrom
        for e in abundances for charge in sorted(counts[e])
        for line in database.ions[e, charge].transitions}
    overlapping_lines = frozenset() if mali_overlap_velocity is None else _cross_element_overlaps(
        {key: line_wavelength[key] for e in abundances for key in selection[e][0]}, mali_overlap_velocity)
    metadata['mali_overlap_velocity'] = mali_overlap_velocity
    metadata['mali_overlap_excluded_lines'] = len(overlapping_lines)
    states = {} if initial_populations is None else {
        e:_initial_state(atmosphere,database,reference,e,counts[e],initial_populations.get(e)) for e in abundances}
    history = []
    element_history = []
    acceleration_history = []
    accelerated_updates = 0
    stopped_by_callback = False
    metadata['accelerated_lambda'] = bool(accelerated_lambda)
    if convergence_criterion not in ("population", "opacity", "flux"):
        raise ValueError("convergence_criterion must be 'population', 'opacity' or 'flux'")
    flux_band = (pop_wave >= flux_wavelength_range[0]) & (pop_wave <= flux_wavelength_range[1])
    if convergence_criterion == "flux" and not np.any(flux_band):
        raise ValueError("flux_wavelength_range contains no population wavelength")
    metadata['flux_wavelength_range'] = [float(x) for x in flux_wavelength_range]
    metadata['convergence_criterion'] = convergence_criterion
    metadata['convergence_tolerance'] = float(tolerance)
    metadata['population_defect_limits'] = dict(element_fraction_floor=POPULATION_DEFECT_FLOOR,
        minimum_rosseland_depth=POPULATION_DEFECT_MINIMUM_TAU)
    opacity_history = []
    opacity_defect = np.inf
    step_history = []
    previous_metal = None
    previous_flux = None
    flux_history = []
    flux_step_history = []
    # The host is fixed, so electron-impact rates are reused across iterations.
    rate_caches = {e: {} for e in abundances}
    for iteration in range(1, maximum_iterations + 1):
        centers = {} if accelerated_lambda and states else None
        current = coefficients(pop_wave, base_pop, states, centers, profile_block_tolerance)
        # The independent closure check is reserved for the emergent spectrum.
        _, radiation, _ = transfer_field(atmosphere, current, n_angle=n_angle, check_source=False)
        operator = (_diagonal_lambda_operator(atmosphere, current, n_angle)
                    if accelerated_lambda and states else None)
        line_fraction = None
        if operator is not None:
            # Weight each line's operator by its share of the total extinction
            # at line centre (Rybicki & Hummer 1991); overlapping lines and
            # continua otherwise over-correct weak lines in a dense forest.
            total = current.true_absorption + current.scattering
            line_fraction = {}
            for key, value in centers.items():
                index = int(np.clip(np.searchsorted(pop_wave, line_wavelength[key]), 1, len(pop_wave)-1))
                if abs(pop_wave[index-1]-line_wavelength[key]) < abs(pop_wave[index]-line_wavelength[key]):
                    index -= 1
                if key not in overlapping_lines:
                    line_fraction[key] = value / np.maximum(total[index], 1e-300)
            del total
        proposals = {e: solve_reduced_light_metal_levels_nlte(
            atmosphere, database, reference, photo, pop_wave, radiation.mean_intensity,
            e, counts[e],photoionization_threshold_data=thresholds.get(e),
            collision_data=collision_data,
            total_recombination_rate_coefficients=(total_recombination or {}).get(e),
            approximate_lambda_diagonal=operator,
            lambda_line_fraction=line_fraction,
            profile_block_tolerance=profile_block_tolerance,
            rate_cache=rate_caches[e], kramers_cumulative=True,
            previous_population_state=states.get(e) if operator is not None else None)
            for e in abundances}
        if convergence_criterion == "flux":
            flux = radiation.interface_flux[:, 0]
            flux_step = (np.inf if previous_flux is None else float(np.max(
                abs(flux - previous_flux)[flux_band] / np.maximum(abs(previous_flux[flux_band]), 1e-300))))
            previous_flux = flux.copy()
            flux_step_history.append(flux_step)
            metadata['iterate_flux_change_history'] = list(flux_step_history)
            check = (opacity_check_gate is None or iteration == maximum_iterations
                     or flux_step < opacity_check_gate * tolerance)
            opacity_defect = np.inf
            if check:
                proposed = coefficients(pop_wave, base_pop, proposals,
                                        block_tolerance=profile_block_tolerance)
                _, proposed_field, _ = transfer_field(atmosphere, proposed, n_angle=n_angle,
                                                      check_source=False)
                change = (abs(proposed_field.interface_flux[:, 0] - flux)[flux_band]
                          / np.maximum(abs(flux[flux_band]), 1e-300))
                opacity_defect = float(np.max(change))
                worst = int(np.argmax(change))
                metadata['worst_flux_defect'] = dict(
                    wavelength_angstrom=float(pop_wave[flux_band][worst]),
                    relative_change=opacity_defect)
                del proposed, proposed_field, change
            flux_history.append(None if not check else opacity_defect)
            metadata['flux_defect_history'] = list(flux_history)
            _LOGGER.info('trace metals iteration %d: iterate flux change %.4g, undamped flux defect %s',
                         iteration, flux_step, 'not evaluated' if not check else '%.4g' % opacity_defect)
        if convergence_criterion == "opacity":
            extinction = current.true_absorption + current.scattering
            emission = current.thermal_emissivity + current.scattering * radiation.mean_intensity
            step_change = np.inf
            if previous_metal is not None:
                step_change = float(max(
                    np.max(abs(current.true_absorption - previous_metal[0])
                           / np.maximum(abs(extinction), 1e-300)),
                    np.max(abs(current.thermal_emissivity - previous_metal[1])
                           / np.maximum(abs(emission), 1e-300))))
            step_history.append(step_change)
            metadata['iterate_opacity_change_history'] = list(step_history)
            previous_metal = (current.true_absorption, current.thermal_emissivity)
            check = (opacity_check_gate is None or iteration == maximum_iterations
                     or step_change < opacity_check_gate * tolerance)
            opacity_defect = np.inf
        if convergence_criterion == "opacity" and not check:
            opacity_history.append(None)
            metadata['opacity_defect_history'] = list(opacity_history)
            _LOGGER.info('trace metals iteration %d: iterate opacity change %.4g '
                         '(undamped residual not evaluated)', iteration, step_change)
            del extinction, emission
        elif convergence_criterion == "opacity":
            proposed = coefficients(pop_wave, base_pop, proposals,
                                    block_tolerance=profile_block_tolerance)
            absorption_change = (abs(proposed.true_absorption - current.true_absorption)
                                 / np.maximum(abs(extinction), 1e-300))
            emission_change = (abs(proposed.thermal_emissivity - current.thermal_emissivity)
                               / np.maximum(abs(emission), 1e-300))
            opacity_defect = float(max(np.max(absorption_change), np.max(emission_change)))
            worst_kind, worst_change = (("absorption", absorption_change)
                if np.max(absorption_change) >= np.max(emission_change) else ("emission", emission_change))
            w_index, d_index = np.unravel_index(np.argmax(worst_change), worst_change.shape)
            metadata['worst_opacity_defect'] = dict(kind=worst_kind,
                wavelength_angstrom=float(pop_wave[w_index]), depth_index=int(d_index),
                column_mass=float(atmosphere.column_mass[d_index]),
                metal_share=float(1.0 - (base_pop.true_absorption[w_index, d_index]
                    if worst_kind == "absorption" else base_pop.thermal_emissivity[w_index, d_index])
                    / max(abs(extinction[w_index, d_index]) if worst_kind == "absorption"
                          else abs(emission[w_index, d_index]), 1e-300)))
            del absorption_change, emission_change, worst_change
            opacity_history.append(opacity_defect)
            # Visible to callbacks and checkpoints, not only the final result.
            metadata['opacity_defect_history'] = list(opacity_history)
            _LOGGER.info('trace metals iteration %d: worst opacity change %s', iteration,
                         metadata['worst_opacity_defect'])
            _LOGGER.info('trace metals iteration %d: opacity defect %.4g, population defect %.4g',
                         iteration, opacity_defect, max(
                             float(np.max(np.abs(p.population_density - states[e].population_density)
                                          / np.maximum(np.maximum(p.population_density, states[e].population_density),
                                                       POPULATION_DEFECT_FLOOR * reference.element_number_density[e])))
                             if e in states else 1.0 for e, p in proposals.items()))
            del proposed, extinction, emission
        del current
        residuals = {e: np.abs(p.population_density - (
            states[e].population_density if e in states else p.lte_population_density)) /
            np.maximum(np.maximum(p.population_density,
                states[e].population_density if e in states else p.lte_population_density),
                POPULATION_DEFECT_FLOOR * reference.element_number_density[e]) for e, p in proposals.items()}
        tested_layers = np.asarray(atmosphere.rosseland_optical_depth) >= POPULATION_DEFECT_MINIMUM_TAU
        residuals = {e: np.where(tested_layers[np.newaxis, :], r, 0.0) for e, r in residuals.items()}
        element_defects = {e:float(np.max(r)) for e,r in residuals.items()}
        defect = max(element_defects.values())
        worst_element = max(element_defects,key=element_defects.get)
        row,depth = np.unravel_index(np.argmax(residuals[worst_element]),residuals[worst_element].shape)
        proposal = proposals[worst_element]
        old_population = (states[worst_element].population_density
                          if worst_element in states else proposal.lte_population_density)
        metadata['element_population_defects'] = element_defects
        metadata['worst_population_defect'] = dict(element=worst_element,
            level_key=list(proposal.level_key[row]),depth_index=int(depth),
            temperature=float(atmosphere.temperature[depth]),
            column_mass=float(atmosphere.column_mass[depth]),
            current_element_fraction=float(old_population[row,depth]/reference.element_number_density[worst_element][depth]),
            proposed_element_fraction=float(proposal.population_density[row,depth]/reference.element_number_density[worst_element][depth]))
        element_history.append(dict(element_defects))
        history.append(defect)
        if iteration_callback is not None:
            iteration_callback(iteration, defect)
        if state_callback is not None:
            evaluated = states or {e:_damped(None,p,0.) for e,p in proposals.items()}
            def synthesize_evaluated():
                _,field,closure = transfer_field(atmosphere,coefficients(wave,base_final,evaluated),n_angle=n_angle)
                return Spectrum(wave,field.interface_flux[:,0],{**spectrum_meta,**metadata,
                    'exploratory_iteration':iteration,'population_defect':defect,
                    'source_closure_residual':float(closure)})
            if state_callback(iteration,defect,evaluated,synthesize_evaluated):
                states = evaluated
                stopped_by_callback = True
                break
        # Keep the evaluated state: its own radiation supplied the defect.
        measure = opacity_defect if convergence_criterion in ("opacity", "flux") else defect
        if states and measure < tolerance:
            break
        if iteration < maximum_iterations:
            old = np.concatenate([states[e].population_density if e in states else p.lte_population_density
                                  for e, p in proposals.items()])
            target = np.concatenate([p.population_density for p in proposals.values()])
            floors = np.concatenate([np.broadcast_to(1e-12 * reference.element_number_density[e],
                                                      p.population_density.shape)
                                     for e, p in proposals.items()])
            if len(history) > 1 and history[-1] > 2 * history[-2]:
                acceleration_history.clear()
            accelerated = None if not states else population_update(
                np.log(np.maximum(old, 1e-300)), np.log(np.maximum(target, 1e-300)),
                acceleration_history, depth=acceleration_depth, mixing=damping, maximum_step=1.,
                residual_weights=np.minimum(1., np.maximum(old, target)/floors))[0]
            if not states:
                # Cold start: the LTE starting point carries no information
                # worth averaging with, so take the first statistical-
                # equilibrium solution in full.
                states = {e: _damped(None, p, 1.) for e, p in proposals.items()}
                metadata['cold_start_first_step'] = 'undamped'
            elif accelerated is None:
                states = {e: _relaxed(states[e], p, damping) for e, p in proposals.items()}
            else:
                offset = 0
                for e, p in proposals.items():
                    size = len(p.level_key)
                    population = np.exp(accelerated[offset:offset+size])
                    # Positivity decoding must preserve each element's represented particles.
                    population *= p.lte_population_density.sum(axis=0) / population.sum(axis=0)
                    states[e] = _with_population(p, population)
                    offset += size
                accelerated_updates += 1
    converged = bool(states) and (
        opacity_defect if convergence_criterion in ("opacity", "flux") else defect) < tolerance
    if not states:
        # One iteration still returns the evaluated LTE state, not an unchecked update.
        states = {e: _damped(None, p, 0.) for e, p in proposals.items()}
    metadata.update(population_defect_history=history,
                    opacity_defect_history=opacity_history,
                    element_population_defect_history=element_history,
                    stopped_by_callback=stopped_by_callback,
                    accelerated_updates=accelerated_updates, acceleration_depth=acceleration_depth,
                    population_wavelength_points=len(pop_wave), convergence_scope="fixed-host metal populations")
    metadata["represented_lte_fraction_minimum"] = {
        e: float(np.min(s.lte_population_density.sum(axis=0) / reference.element_number_density[e]))
        for e, s in states.items()}
    metadata["maximum_particle_conservation_error"] = max(
        float(np.max(abs(s.population_density.sum(axis=0) - s.lte_population_density.sum(axis=0))
                     / reference.element_number_density[e])) for e, s in states.items())
    metadata["atomic_database_source"] = database.source
    metadata["photoionization_database_source"] = photo.source
    if not converged:
        message = f"trace-metal populations not converged after {iteration} iterations: undamped defect={defect:.3g}"
        if require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, TraceMetalConvergenceWarning, stacklevel=2)
    _, final_field, closure = transfer_field(atmosphere, coefficients(wave, base_final, states), n_angle=n_angle)
    metadata["source_closure_residual"] = float(closure)
    spectrum = Spectrum(wave, final_field.interface_flux[:, 0], {**spectrum_meta, **metadata})
    return TraceMetalResult(spectrum, base_spectrum, states, reference,
                            converged, iteration, defect, metadata)
