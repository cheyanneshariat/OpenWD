"""Hot H/He NLTE adapters for the shared OpenWD nonlinear solver.

The retained model atoms conserve each element separately. H and He see one
radiation field, mass density, and LTE electron/pressure closure. This is a
restricted NLTE treatment: ionization departures do not reclose the EOS.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)

from ._mass_feautrier import mass_field, mass_emissivity_field
from .atmosphere import _atmosphere_from_hydrogen_helium_state
from .eos import hummer_mihalas_helium_lte, hummer_mihalas_hydrogen_helium_lte
from . import helium_nlte as he
from . import multilevel_nlte as h
from .nlte import _profile_averaged_mean_intensity_nu
from .nlte_core import NLTETransferCoefficients, _FixedTransferCache
from .opacity import optical_depth_from_mass_opacity, electron_scattering_mass_coefficient
from .spectrum import planck_lambda_angstrom


@dataclass(frozen=True)
class HotPopulationState:
    helium: he.CoupledHeliumNLTEState
    hydrogen: h.MultilevelHydrogenNLTEState | None = None
    converged: bool = False
    iterations: int = 0
    maximum_relative_population_change: float = float('inf')


def transfer_field(atmosphere, coefficients, *, n_angle=3, check_source=True, wavelength_chunk_size=512):
    """Solve scattering directly with the release mass-volume transfer solver."""
    a, eta, s = (np.asarray(x) for x in (
        coefficients.true_absorption, coefficients.thermal_emissivity,
        coefficients.scattering))
    shape = (len(coefficients.wavelength_angstrom), atmosphere.n_depth)
    if any(x.shape != shape or np.any(~np.isfinite(x)) for x in (a, eta, s)):
        raise ValueError('NLTE coefficients must be finite wavelength-by-depth arrays')
    if np.any(a+s <= 0) or np.any(a[:,-1] <= 0) or np.any(eta < 0) or np.any(s < 0):
        raise ValueError('NLTE transfer requires positive total extinction and bottom absorption, and nonnegative emission/scattering')
    tau = optical_depth_from_mass_opacity(atmosphere.column_mass, a+s)
    # Use eta directly: stimulated emission can change the sign of net
    # absorption inside the atmosphere. Only the thermal bottom condition
    # requires eta/kappa, where positive absorption is checked above.
    source, field = mass_emissivity_field(tau, eta, a, s,bottom_source=eta[:,-1]/a[:,-1],
                              column_mass=atmosphere.column_mass, n_angle=n_angle,
                              wavelength_chunk_size=wavelength_chunk_size)
    if not check_source:
        return source, field, None
    # Independent fixed-source formal solution on the SAME mass volumes.
    _, check = mass_field(tau, source, a+s, np.zeros_like(s),
                         column_mass=atmosphere.column_mass, n_angle=n_angle,
                         wavelength_chunk_size=wavelength_chunk_size, reconstruct_intensity=True)
    closure = (eta+s*check.mean_intensity)/(a+s)
    error = float(np.max(abs(source-closure)/np.maximum(abs(source), 1e-100)))
    return source, field, error


@dataclass(frozen=True)
class HotNLTEModel:
    collision_data: h.HydrogenElectronCollisionData
    helium_i_stark_table: object
    helium_ii_stark_table: object
    helium_i_collision_data: object
    maximum_helium_ii_level: int = 32
    maximum_hydrogen_level: int = 8
    log_hydrogen_to_helium: float | None = None
    n_angle: int = 3
    population_maximum_iterations: int = 120
    population_tolerance: float = 1e-4
    hydrogenic_collision_model: str = "tlusty-mihalas"
    # None: TLUSTY 14-term He I.  A different atom needs matching collision
    # data (read with that atom file for the ionization scale).
    helium_i_atom: object = None
    # "last" (default) or "dominant": which helium equation particle
    # conservation replaces (see solve_coupled_helium_statistical_equilibrium).
    helium_conservation_row: str = "last"

    def __post_init__(self):
        if self.hydrogenic_collision_model not in ("ccc-scaled", "tlusty-mihalas"):
            raise ValueError("unknown He II collision prescription")
        for value, lower, upper in (
            (self.maximum_helium_ii_level, 2, 32),
            (self.maximum_hydrogen_level, 2, 20),
            (self.population_maximum_iterations, 1, None),
            (self.n_angle, 1, None)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower or (upper is not None and value > upper):
                raise ValueError("invalid hot-atom level/iteration/angular count")
        if not np.isfinite(self.population_tolerance) or self.population_tolerance <= 0:
            raise ValueError("population_tolerance must be finite and positive")
        if self.log_hydrogen_to_helium is not None and not np.isfinite(self.log_hydrogen_to_helium):
            raise ValueError("log_hydrogen_to_helium must be finite")
        scale = getattr(self.helium_i_collision_data, 'ionization_scale', None)
        if self.helium_i_atom is not None and scale is not None and scale.size != self.helium_i_atom.n_terms:
            raise ValueError("He I collision ionization scale does not match the He I atom")

    def rebuild_atmosphere(self, template, temperature, previous_state=None):
        metadata = {**template.metadata, 'nlte_charge_feedback': False,
                    'nlte_electron_closure': 'temperature-updated LTE H/He EOS',
                    'radiative_equilibrium_converged': False,
                    'radiative_equilibrium_solver_converged': False}
        # A seed certificate belongs to its original temperatures/equations.
        # In particular a fresh LTE initialization cannot certify NLTE trials.
        metadata.pop('equilibrium_certificate', None)
        metadata.pop('model_request_fingerprint', None)
        if self.log_hydrogen_to_helium is not None:
            eos = hummer_mihalas_hydrogen_helium_lte(
                temperature, template.gas_pressure, self.log_hydrogen_to_helium)
            return _atmosphere_from_hydrogen_helium_state(
                template.effective_temperature, template.logg,
                template.rosseland_optical_depth, template.column_mass,
                temperature, template.gas_pressure, eos, metadata)
        if template.hydrogen_lte_state is not None or np.any(template.neutral_h_density > 0) or np.any(template.proton_density > 0):
            raise ValueError("pure-He model requires a hydrogen-free template")
        eos = hummer_mihalas_helium_lte(temperature, template.gas_pressure,
                                      correlated_microfields=True)
        return replace(template, temperature=temperature, mass_density=eos.mass_density,
                       electron_density=eos.electron_density, helium_lte_state=eos,
                       metadata=metadata)

    def remap(self, atmosphere, state):
        if (state.hydrogen is None) != (self.log_hydrogen_to_helium is None):
            raise ValueError("population state and model compositions differ")
        return HotPopulationState(
            he.remap_coupled_helium_state(atmosphere, state.helium),
            None if state.hydrogen is None else h.remap_multilevel_hydrogen_state(atmosphere, state.hydrogen))

    def transfer_coefficients(self, atmosphere, wavelength, state, *, lines=True, _cache=None):
        # Stimulated continuum terms may be negative for a single ion/species.
        # Combine all material contributions before transfer_field validates
        # the positive total extinction. Do not clip or discard signed terms.
        helium = he.coupled_helium_nlte_transfer_coefficients(
            atmosphere, wavelength, state.helium, _cache=_cache,
            _allow_signed_continuum=True,
            helium_i_stark_table=self.helium_i_stark_table,
            helium_ii_stark_table=self.helium_ii_stark_table,
            include_helium_i_lines=lines, include_helium_i_resonance_lines=lines,
            include_helium_ii_lines=lines)
        if state.hydrogen is None:
            return helium
        hydrogen = h.hydrogen_nlte_transfer_coefficients(
            atmosphere, wavelength, state.hydrogen, _cache=_cache,
            _allow_signed_continuum=True,
            maximum_level=self.maximum_hydrogen_level, include_lines=lines,
            include_paschen=self.maximum_hydrogen_level >= 3,
            include_brackett=self.maximum_hydrogen_level >= 4)
        # Both species coefficient builders include the same Thomson opacity.
        scattering = helium.scattering + hydrogen.scattering - electron_scattering_mass_coefficient(atmosphere)[None, :]
        return NLTETransferCoefficients(
            np.asarray(wavelength), helium.true_absorption+hydrogen.true_absorption,
            helium.thermal_emissivity+hydrogen.thermal_emissivity, scattering,
            {'composition': 'homogeneous-hydrogen-helium', 'electron_scattering_count': 1})

    def _rate_state(self, atmosphere, wavelength=None, mean=None, neutral=None, ion=None, hydrogen=None):
        helium = he.solve_coupled_helium_statistical_equilibrium(
            atmosphere, self.collision_data, maximum_helium_ii_level=self.maximum_helium_ii_level,
            helium_i_collision_data=self.helium_i_collision_data, helium_i_atom=self.helium_i_atom,
            conservation_row=self.helium_conservation_row,
            hydrogenic_collision_model=self.hydrogenic_collision_model,
            neutral_line_mean_intensity_nu=neutral, helium_ii_line_mean_intensity_nu=ion,
            neutral_continuum_wavelength_angstrom=wavelength,
            neutral_continuum_mean_intensity_lambda=mean,
            helium_ii_continuum_wavelength_angstrom=wavelength,
            helium_ii_continuum_mean_intensity_lambda=mean)
        hydrogen_state = None
        if self.log_hydrogen_to_helium is not None:
            hydrogen_state = h.solve_multilevel_hydrogen_statistical_equilibrium(
                atmosphere, self.collision_data, maximum_level=self.maximum_hydrogen_level,
                line_mean_intensity_nu=hydrogen, continuum_wavelength_angstrom=wavelength,
                continuum_mean_intensity_lambda=mean)
        return HotPopulationState(helium, hydrogen_state)

    def _line_problems(self, atmosphere):
        neutral = he._neutral_helium_line_components(atmosphere, self.helium_i_stark_table, self.helium_i_atom)
        ion = {(lower, upper): (he._prepare_helium_line_transfer_problem(
            atmosphere, he.helium_ii_shell_transition(lower, upper),
            self.maximum_helium_ii_level, stark_table=self.helium_ii_stark_table),)
            for lower, upper in sorted({
                (line.lower_principal_quantum_number, line.upper_principal_quantum_number)
                for line in he.coupled_helium_ii_lines(self.maximum_helium_ii_level)})}
        hydrogen = {}
        if self.log_hydrogen_to_helium is not None:
            # Couple only transitions with retained opacity/profile support.
            # For the larger atom, e.g. Brackett n>14, the existing atomic
            # solver retains its Planck-field closure rather than inventing
            # unprovided Stark profiles.
            transitions = sorted({(line.lower_level, line.upper_level)
                for line in (*h.LYMAN_LINES, *h.BALMER_LINES, *h.PASCHEN_LINES, *h.BRACKETT_LINES)
                if line.upper_level <= self.maximum_hydrogen_level})
            hydrogen = {(lower, upper): (h._prepare_line_transfer_problem(
                atmosphere, (h.hydrogen_shell_transition(lower, upper) if upper <= 9 else
                             h._extended_hydrogen_shell_transition(lower, upper)),
                self.maximum_hydrogen_level),)
                for lower, upper in transitions}
        return neutral, ion, hydrogen

    def solve_populations(self, atmosphere, previous_state=None, *, accelerated_lambda=False):
        """Fixed-temperature populations by iterated statistical equilibrium.

        ``accelerated_lambda`` (opt-in) preconditions the bound-bound rates
        with the diagonal approximate lambda operator (MALI, see
        ``_hot_mali``).  It changes the iteration path, not the fixed point
        or the convergence test; the default leaves this method unchanged.
        """
        groups = self._line_problems(atmosphere)
        grids = [he.default_neutral_helium_continuum_wavelength(self.helium_i_atom),
                 he.default_helium_ii_continuum_wavelength(self.maximum_helium_ii_level)]
        if self.log_hydrogen_to_helium is not None:
            grids.append(h._default_continuum_wavelength(self.maximum_hydrogen_level))
        grids.extend(p.continuum.wavelength_angstrom for group in groups
                     for problems in group.values() for p in problems)
        wave = np.unique(np.concatenate(grids))
        current = self._rate_state(atmosphere) if previous_state is None else self.remap(atmosphere, previous_state)
        history = []
        coefficient_cache = _FixedTransferCache(atmosphere, wave)
        accelerator = None
        if accelerated_lambda:
            from ._hot_mali import HotMALI
            accelerator = HotMALI(self, atmosphere, wave, groups)
        for iteration in range(1, self.population_maximum_iterations+1):
            c = self.transfer_coefficients(atmosphere, wave, current, _cache=coefficient_cache)
            _, field, _ = transfer_field(atmosphere, c, n_angle=self.n_angle, check_source=False)
            fields = []
            for group in groups:
                fields.append({key: np.average([
                    _profile_averaged_mean_intensity_nu(
                        p.continuum.wavelength_angstrom, p.lte_line_opacity,
                        field.mean_intensity[np.searchsorted(wave, p.continuum.wavelength_angstrom)])
                    for p in problems], axis=0,
                    weights=[p.line.absorption_oscillator_strength for p in problems])
                    for key, problems in group.items()})
            candidate = (self._rate_state(atmosphere, wave, field.mean_intensity, *fields)
                         if accelerator is None else
                         accelerator.state(current, c, field.mean_intensity, fields))
            old, reference = population_arrays(current)
            new, _ = population_arrays(candidate)
            relative_change = abs(new-old)/np.maximum(new, 1e-12*reference.sum(axis=1)[:, None])
            change = float(np.max(relative_change))
            worst_depth, worst_level = np.unravel_index(np.argmax(relative_change), relative_change.shape)
            if iteration == 1 or iteration % 10 == 0:
                _LOGGER.info("NLTE populations: iteration %d, residual %.6g (depth %d, state %d)",
                             iteration, change, worst_depth, worst_level)
            if change < self.population_tolerance:
                # Return the state whose radiation produced the checked rate
                # defect, rather than an unchecked last fixed-point update.
                return population_status(current, True, iteration, change)
            if iteration == self.population_maximum_iterations:
                break
            log_old = np.log(old/reference)
            log_new = np.log(new/reference)
            proposal, _ = h._anderson_log_population_update(
                log_old, log_new, history, depth=4, mixing=.65, maximum_step=.75)
            if proposal is None:
                proposal = log_old + .4*(log_new-log_old)
            current = with_departures(candidate, np.exp(proposal))
        return population_status(current, False, iteration, change)


def population_status(state, converged, iterations, change):
    values = dict(converged=converged, iterations=iterations,
                  maximum_relative_population_change=change)
    return replace(state, helium=replace(state.helium, **values),
                   hydrogen=None if state.hydrogen is None else replace(state.hydrogen, **values),
                   **values)


def population_arrays(state):
    he_state = state.helium
    actual = [he_state.neutral_population_density, he_state.singly_ionized_population_density,
              he_state.doubly_ionized_he_density[:, None]]
    reference = [he_state.lte_neutral_population_density, he_state.lte_singly_ionized_population_density,
                 he_state.lte_doubly_ionized_he_density[:, None]]
    if state.hydrogen is not None:
        actual.extend((state.hydrogen.population_density, state.hydrogen.proton_density[:, None]))
        reference.extend((state.hydrogen.lte_population_density, state.hydrogen.lte_proton_density[:, None]))
    return np.column_stack(actual), np.column_stack(reference)


def with_departures(state, departures):
    """Normalize each elemental reservoir independently after acceleration."""
    _, reference = population_arrays(state)
    n_neutral = state.helium.neutral_population_density.shape[1]
    nhe = n_neutral+1+state.helium.singly_ionized_population_density.shape[1]
    departures = departures.copy()
    for section in (slice(0, nhe), slice(nhe, None)):
        if reference[:, section].size:
            departures[:, section] *= (reference[:, section].sum(axis=1)/
                (reference[:, section]*departures[:, section]).sum(axis=1))[:, None]
    helium = he._replace_coupled_departures(
        state.helium, departures[:, :n_neutral], departures[:, n_neutral:nhe-1], departures[:, nhe-1],
        iterations=0, converged=False, maximum_change=float('inf'), metadata=state.helium.metadata)
    hydrogen = state.hydrogen
    if hydrogen is not None:
        hydrogen = replace(hydrogen, departure_coefficient=departures[:, nhe:-1],
                           continuum_departure_coefficient=departures[:, -1],
                           population_density=reference[:, nhe:-1]*departures[:, nhe:-1],
                           proton_density=reference[:, -1]*departures[:, -1], converged=False)
    return HotPopulationState(helium, hydrogen)


@dataclass(frozen=True)
class HotAtmosphereResult:
    atmosphere: object
    population_state: HotPopulationState
    nonlinear_result: object


def solve_hot_nlte_atmosphere(seed, model, wavelength, *, maximum_iterations=120,
                              flux_tolerance=3e-3, temperature_tolerance=3e-4,
                              iteration_callback=None):
    """Solve simultaneous temperatures/populations with the common driver.

    A Planck-to-NLTE continuation initializes the full equations. Intermediate
    stages cannot certify an atmosphere. The bounded two-step smoke budget
    attempts full NLTE directly and normally returns an unconverged diagnostic.
    """
    from ._hot_structure import solve
    if (isinstance(maximum_iterations, (bool, np.bool_)) or
            not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations<1):
        raise ValueError('maximum_iterations must be a positive integer')
    budget=min(maximum_iterations,model.population_maximum_iterations)
    fractions=([1.] if maximum_iterations<=2 else
               np.linspace(0.,1.,6 if model.log_hydrogen_to_helium is None else 11))
    state=None
    stages=[]
    offset=0
    def callback(iteration,atmosphere,populations,diagnostics):
        if iteration_callback is not None:
            iteration_callback(offset+iteration,atmosphere,diagnostics)
    for fraction in fractions:
        result=solve(seed,model,wavelength,maximum_iterations=budget,populations=state,
                     nlte_fraction=float(fraction),iteration_callback=callback,
                     flux_tolerance=flux_tolerance,temperature_tolerance=temperature_tolerance)
        stages.append(dict(nlte_fraction=float(fraction),
                           solver_converged=bool(result.nonlinear_result.converged),
                           iterations=result.nonlinear_result.iterations))
        offset+=result.nonlinear_result.iterations
        if not result.nonlinear_result.converged:
            break
        seed,state=result.atmosphere,result.population_state
    metadata={**result.atmosphere.metadata,'nlte_continuation_stages':stages,
              'radiative_equilibrium_iterations':offset,
              'nlte_population_tolerance':model.population_tolerance}
    return replace(result,atmosphere=replace(result.atmosphere,metadata=metadata))
