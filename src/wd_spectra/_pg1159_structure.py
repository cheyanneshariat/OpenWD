"""He/C/O material equations for the common trust-region atmosphere solver.

Statistical equilibrium is eliminated by an inner solve. The inexpensive
material Jacobian re-solves local SE at fixed radiation. A measured
population-response subspace and the common driver's secant updates include
the remaining nonlocal coupling. Every accepted trial re-solves
SE, and termination additionally requires an undamped, unpreconditioned rate
check, electron closure, hydrostatic balance, and the shared energy checks.
"""
from dataclasses import dataclass, replace
import logging
import time
import numpy as np
from ._compat import trapezoid
from .constants import STEFAN_BOLTZMANN, LIGHT_SPEED, PLANCK, BOLTZMANN
from .nonlinear import (
    NonlinearEvaluation,
    RecoverableEvaluationError,
    solve_trust_region_newton,
    nonlinear_result_metadata,
)
from .nlte_core import NonphysicalPopulationError
from ._mass_feautrier import (
    mass_emissivity_energy,
    mass_emissivity_field,
    InvalidRadiationFieldError,
)
from ._pg1159_transfer import transfer_field
from ._convergence import equilibrium_certificate
from .opacity import optical_depth_from_mass_opacity
from .helium_nlte import remap_coupled_helium_state
from .spectrum import planck_lambda_angstrom
from .pg1159 import (
    pg1159_composition_atmosphere,
    _pg1159_nlte_charge_feedback,
    _maximum_population_state_change,
    _maximum_coupled_helium_population_change,
    _coupled_helium_population_change_diagnostic,
    _population_state_change_diagnostic,
    _formal_departures_from_population,
    _finite_departure_ratio,
)
from types import MappingProxyType

LOGGER = logging.getLogger(__name__)
PG1159_LOCAL_ENERGY_MINIMUM_ROSSELAND_DEPTH = 1.0e-5


class PG1159MaterialClosureError(RecoverableEvaluationError):
    """A physical PG1159 trial whose eliminated material did not converge."""


def measured_directional_response(
    probe, x, residual, direction, *, maximum_log_step=1e-3, attempts=4
):
    """Measure one coupled response, shrinking only for trial-domain failures.

    Material closure exhaustion is independent of the finite-difference step:
    repeating the expensive solve at smaller steps cannot make its fixed-point
    tolerance easier to satisfy. Let the caller retain its approximate
    Jacobian immediately in that case.
    """
    scale = float(np.max(abs(direction)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid PG1159 response direction")
    h = maximum_log_step / scale
    for attempt in range(attempts):
        try:
            return (np.asarray(probe(x + h * direction)) - residual) / h
        except PG1159MaterialClosureError:
            raise
        except RecoverableEvaluationError:
            if attempt == attempts - 1:
                raise
            h *= .5


TRACE_LENGTH = 16
MAXIMUM_SECANT_CORRECTIONS = 6
MAXIMUM_SECANT_REPAIRS = 6


def secant_corrected(matrix, corrections):
    """Return the least change of ``matrix`` that reproduces measured actions.

    ``corrections`` holds (unit direction, measured residual derivative) pairs.
    The corrected matrix maps every direction to its measured derivative and
    is unchanged on their orthogonal complement.
    """
    q = np.column_stack([c[0] for c in corrections])
    m = np.column_stack([c[1] for c in corrections])
    coefficients = np.linalg.lstsq(q, np.eye(q.shape[0]), rcond=1e-8)[0]
    return matrix + (m - matrix @ q) @ coefficients


def rejected_direction_secant(trace, anchor):
    """Fit the residual derivative along the last rejected line search.

    The trials after the anchor evaluation lie on anchor + f * direction.
    A per-component fit dR = f a + f^2 b separates the directional derivative
    ``a`` from the curvature the backtracking factors also sample.  Returns
    (unit direction, derivative per unit length) or None.
    """
    anchor = np.asarray(anchor, dtype=float)
    base = None
    for index, (x, residual) in enumerate(trace):
        if np.array_equal(x, anchor):
            base = index
    if base is None:
        return None
    r0 = trace[base][1]
    trials = [(x - anchor, residual - r0) for x, residual in trace[base + 1:]]
    if not trials:
        return None
    last = trials[-1][0]
    batch = [
        (dx, dr) for dx, dr in trials
        if np.linalg.norm(dx) > 0
        and float(dx @ last) / (np.linalg.norm(dx) * np.linalg.norm(last)) > 0.999
    ]
    if len(batch) < 2:
        return None
    direction = max((dx for dx, _ in batch), key=np.linalg.norm)
    factors = np.array([float(dx @ direction) / float(direction @ direction) for dx, _ in batch])
    design = np.column_stack([factors, factors ** 2])
    changes = np.vstack([dr for _, dr in batch])
    derivative = np.linalg.lstsq(design, changes, rcond=None)[0][0]
    length = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(derivative)) or length <= 0:
        return None
    return direction / length, derivative / length


def _population_response_probe_limit(fraction, certification_stage=None):
    """Bound provisional work while resolving late-continuation curvature."""
    if certification_stage is None:
        certification_stage = fraction == 1.0
    # One measured current-direction action captures the leading eliminated
    # population coupling.  Further directions cost another material closure
    # each and previously added minutes without validating the noisy response.
    return 1


def _local_energy_mask(atmosphere, model):
    """Cells where the cancellation-prone local heating equation is useful."""
    tau = np.asarray(atmosphere.rosseland_optical_depth[:-1])
    lower = float(
        getattr(
            model,
            "atmosphere_local_energy_minimum_rosseland_depth",
            PG1159_LOCAL_ENERGY_MINIMUM_ROSSELAND_DEPTH,
        )
    )
    upper = float(
        getattr(model, "atmosphere_local_energy_maximum_rosseland_depth", 10.0)
    )
    if not 0.0 <= lower < upper:
        raise ValueError("PG1159 local-energy Rosseland bounds must be ordered")
    mask = (tau >= lower) & (tau < upper)
    if not np.any(mask):
        raise ValueError("PG1159 local-energy window contains no structure cells")
    return mask


def _photospheric_flux_mask(atmosphere, model):
    """Interfaces in the declared spectrum-forming structure window."""
    tau = np.asarray(atmosphere.rosseland_optical_depth)
    upper = float(
        getattr(model, "atmosphere_local_energy_maximum_rosseland_depth", 10.0)
    )
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("PG1159 photospheric Rosseland bound must be positive")
    mask = tau < upper
    if not np.any(mask):
        raise ValueError("PG1159 photospheric flux window contains no interfaces")
    return mask


def _legacy_operator_split_correction(
    atmosphere,
    model,
    wavelength,
    coefficients,
    field,
    flux,
    target_pressure,
):
    """Return the validated legacy PG1159 temperature/pressure iteration map.

    The shared nonlinear driver still evaluates and backtracks every proposed
    physical state.  This function only supplies the inexpensive direction
    that the legacy atmosphere code used successfully.
    """
    temperature = atmosphere.temperature
    planck = planck_lambda_angstrom(wavelength[:, None], temperature[None, :])
    hotter = planck_lambda_angstrom(
        wavelength[:, None], (1.001 * temperature)[None, :]
    )
    heating = trapezoid(
        coefficients.true_absorption * field.mean_intensity
        - coefficients.thermal_emissivity,
        wavelength,
        axis=0,
    )
    derivative = trapezoid(
        coefficients.true_absorption * (hotter - planck),
        wavelength,
        axis=0,
    ) / 0.001
    local = np.divide(
        heating,
        np.maximum(derivative, np.finfo(float).tiny),
    )
    active = atmosphere.rosseland_optical_depth < 10.0
    local = np.where(active, np.clip(local, -0.025, 0.025), 0.0)
    smoothed = local.copy()
    smoothed[1:-1] = 0.25 * local[:-2] + 0.5 * local[1:-1] + 0.25 * local[2:]
    smoothed[active] -= np.mean(smoothed[active])
    global_correction = float(
        np.clip(-0.25 * np.log(max(flux[0] + 1.0, np.finfo(float).tiny)), -0.05, 0.05)
    )
    temperature_correction = np.clip(
        0.20 * smoothed + global_correction, -0.05, 0.05
    )
    proposed = max(float(np.max(abs(smoothed[active]))), abs(global_correction))
    if target_pressure is None:
        return temperature_correction, proposed
    pressure_correction = 0.35 * np.clip(
        np.log(target_pressure / atmosphere.gas_pressure), -0.05, 0.05
    )
    return np.r_[temperature_correction, pressure_correction], proposed


def population_defect(old, new, model):
    return max(
        _maximum_coupled_helium_population_change(
            old.helium_state,
            new.helium_state,
            relative_floor=model.helium_population_relative_floor,
        ),
        _maximum_population_state_change(
            old.light_metal_state,
            old.carbon_level_state,
            old.oxygen_level_state,
            new.light_metal_state,
            new.carbon_level_state,
            new.oxygen_level_state,
            new.metal_state,
            relative_floor=model.metal_population_relative_floor,
            ion_relative_floor=model.metal_ion_population_relative_floor,
            level_relative_floor=model.metal_level_population_relative_floor,
        ),
    )


def population_defect_diagnostic(old, new, model):
    if not hasattr(old, "helium_state") or not hasattr(new, "helium_state"):
        return {"family": "unavailable", "relative_change": 0.0}
    helium = dict(_coupled_helium_population_change_diagnostic(
        old.helium_state,
        new.helium_state,
        relative_floor=model.helium_population_relative_floor,
    ))
    helium["family"] = "helium"
    metal = dict(_population_state_change_diagnostic(
        old.light_metal_state,
        old.carbon_level_state,
        old.oxygen_level_state,
        new.light_metal_state,
        new.carbon_level_state,
        new.oxygen_level_state,
        new.metal_state,
        relative_floor=model.metal_population_relative_floor,
        ion_relative_floor=model.metal_ion_population_relative_floor,
        level_relative_floor=model.metal_level_population_relative_floor,
    ))
    metal["family"] = "metal"
    return max(
        (helium, metal),
        key=lambda item: float(item.get("relative_change", 0.0)),
    )


def remap_material(model, template, temperature, pressure, state):
    """A local, frozen-departure-coefficient approximation to material response."""
    a = model.rebuild_atmosphere(
        replace(template, gas_pressure=pressure), temperature, state
    )
    _, metal = pg1159_composition_atmosphere(
        a, model.atomic_database, model.mass_fractions
    )
    if model.nlte_charge_feedback:
        a, metal = _pg1159_nlte_charge_feedback(a, metal, state)
    helium = remap_coupled_helium_state(a, state.helium_state)
    light = state.light_metal_state
    if light is not None:
        populations = {}
        for element, old in light.ion_number_density.items():
            reference = metal.ion_number_density[element]
            departure = _finite_departure_ratio(
                old, light.lte_ion_number_density[element]
            )
            trial = departure * reference
            trial *= metal.element_number_density[element] / np.maximum(
                trial.sum(axis=0), 1e-300
            )
            populations[element] = trial
        light = replace(
            light,
            ion_number_density=MappingProxyType(populations),
            lte_ion_number_density=metal.ion_number_density,
            ion_departure_coefficient=MappingProxyType(
                {
                    (e, q): _finite_departure_ratio(
                        v[q], metal.ion_number_density[e][q]
                    )
                    for e, v in populations.items()
                    for q in range(len(v))
                }
            ),
        )
    levels = []
    for prefix, level in (
        ("carbon", state.carbon_level_state),
        ("oxygen", state.oxygen_level_state),
    ):
        if level is None:
            levels.append(None)
            continue
        database = getattr(model, prefix + "_population_atomic_database")
        by_key = {
            (element, charge, l.index): l
            for (element, charge), ion in database.ions.items()
            for l in ion.levels
        }
        reference = np.array(
            [
                metal.ion_number_density[element][charge]
                * by_key[(element, charge, index)].statistical_weight
                * np.exp(
                    -by_key[(element, charge, index)].energy_wavenumber
                    * PLANCK
                    * LIGHT_SPEED
                    / (BOLTZMANN * temperature)
                )
                / metal.partition_function[(element, charge)]
                for element, charge, index in level.level_key
            ]
        )
        reservoir = np.zeros_like(reference)
        index_by_key = {key: i for i, key in enumerate(level.level_key)}
        for parent, terms in (
            getattr(model, prefix + "_lte_level_reservoir") or {}
        ).items():
            for charge, energy, weight in terms:
                reservoir[index_by_key[parent]] += (
                    metal.ion_number_density[level.element][charge]
                    * weight
                    * np.exp(-energy * PLANCK * LIGHT_SPEED / (BOLTZMANN * temperature))
                    / metal.partition_function[(level.element, charge)]
                )
        departure = _finite_departure_ratio(
            level.population_density, level.lte_population_density
        )
        # Keep the atom's complete explicit-plus-LTE-reservoir conservation.
        norm = (reference + reservoir).sum(axis=0) / np.maximum(
            ((reference + reservoir) * departure).sum(axis=0), 1e-300
        )
        departure = departure * norm
        mapping = MappingProxyType(
            {key: departure[i] for i, key in enumerate(level.level_key)}
        )
        levels.append(
            replace(
                level,
                population_density=reference * departure,
                lte_population_density=reference,
                conservation_weight=1. + reservoir / np.maximum(reference, 1e-300),
                population_level_departure_coefficient=mapping,
                level_departure_coefficient=_formal_departures_from_population(
                    level, mapping
                ),
            )
        )
    return (
        a,
        replace(
            state,
            helium_state=helium,
            metal_state=metal,
            light_metal_state=light,
            carbon_level_state=levels[0],
            oxygen_level_state=levels[1],
        ),
    )


def rate_response_material(model, template, temperature, pressure, state, *,
                           local_lambda_operator=False):
    """Local SE/EOS tangent with the accepted radiation field held fixed.

    All depth temperatures can be perturbed together: the frozen radiation
    removes cross-depth coupling from this local material response. Transfer
    supplies that coupling, with the additional measured directional response.
    """
    if state.radiation_wavelength is None or state.radiation_mean_intensity is None:
        raise ValueError(
            "NLTE material response requires its population radiation field"
        )
    radiation = (state.radiation_wavelength, state.radiation_mean_intensity)
    # A diagonal-Lambda* local radiation response was tested here (2026-09-24)
    # and made the upper-atmosphere columns worse against finite differences;
    # the missing damping is non-local.  Keep the frozen-field tangent.
    operator = (
        (state.radiation_diagonal_lambda, state.radiation_source_function)
        if local_lambda_operator
        and getattr(state, "radiation_diagonal_lambda", None) is not None
        and getattr(state, "radiation_source_function", None) is not None
        else None
    )
    if local_lambda_operator and operator is None:
        raise ValueError("the anchor population state has no local lambda operator")
    local = replace(
        model,
        metal_population_iterations=1,
        metal_population_minimum_iterations=1,
        metal_population_damping=1.0,
        helium_population_damping=1.0,
        metal_population_acceleration_depth=0,
        coupled_population_acceleration_depth=0,
        use_population_ali=False,
    )
    template = replace(template, gas_pressure=pressure)
    p = state
    for iteration in range(6):
        a = local.rebuild_atmosphere(template, temperature, p)
        next_state = local.solve_populations(
            a, p, _fixed_radiation=radiation,
            **({"_local_lambda_operator": operator} if operator is not None else {}),
        )
        refreshed = local.rebuild_atmosphere(template, temperature, next_state)
        charge = float(np.max(abs(refreshed.electron_density / a.electron_density - 1)))
        defect = population_defect(p, next_state, local)
        p = next_state
        if charge < 1e-7 and defect < 1e-5:
            break
    return a, p


@dataclass(frozen=True)
class PG1159AtmosphereResult:
    atmosphere: object
    population_state: object
    nonlinear_result: object
    # Material closure a stage ended on after a rejected-direction response
    # (tolerance, resolved flag, secant corrections); None otherwise.  The next
    # full-NLTE stage starts from it instead of re-opening the loose closure.
    closure_state: object = None

    @property
    def converged(self):
        return self.atmosphere.metadata["equilibrium_certificate"]["verified"]


class PG1159Equations:
    def __init__(
        self,
        seed,
        model,
        wave,
        *,
        radiative_acceleration=True,
        radiative_acceleration_scale=1.0,
        certification_stage=None,
        material_tolerance_ceiling=None,
        resolved_material_closure=False,
    ):
        self.seed, self.model, self.wave = seed, model, np.asarray(wave)
        # A resolved closure keeps the coupled population history through the
        # root and measures population-response directions near it.  It is
        # used only after a rejected direction at the minimum material
        # tolerance; see _solve_stage.
        self.resolved_material_closure = bool(resolved_material_closure)
        # Residuals of recent full evaluations, and measured directional
        # derivatives (unit direction, residual action) that correct the
        # analytic tangent in a resolved closure.
        self.trace = []
        self.secant_corrections = []
        # Cells [first, last) whose full-NLTE rows are local heating instead
        # of interface flux (None: flux profile only).
        self.hybrid_band = None
        # Resolved-closure numerics: population ALI in the material closure.
        self.population_ali = False
        self.nd = seed.n_depth
        self.radiative_acceleration = radiative_acceleration
        self.radiative_acceleration_scale = radiative_acceleration_scale
        self.certification_stage = (
            getattr(model, "population_nlte_fraction", 1.0) == 1.0
            if certification_stage is None
            else bool(certification_stage)
        )
        self.material_tolerance_ceiling = (
            getattr(model, "metal_population_relative_tolerance", 1e-4)
            if self.certification_stage
            else 1e-4
        ) if material_tolerance_ceiling is None else float(
            material_tolerance_ceiling
        )
        if self.material_tolerance_ceiling <= 0.0:
            raise ValueError("material_tolerance_ceiling must be positive")
        self.target = STEFAN_BOLTZMANN * seed.effective_temperature ** 4
        self.cache = {}
        self.coefficients = {}
        self.anchor = None
        self.evaluations = 0

    def initial_state(self):
        return (
            np.r_[np.log(self.seed.temperature), np.log(self.seed.gas_pressure)]
            if self.radiative_acceleration
            else np.log(self.seed.temperature)
        )

    def unpack(self, x):
        return (
            np.exp(x[: self.nd]),
            (
                np.exp(x[self.nd :])
                if self.radiative_acceleration
                else self.seed.gas_pressure
            ),
        )

    def material(
        self,
        x,
        *,
        maximum_closure_iterations=8,
        maximum_population_iterations=None,
    ):
        temperature, pressure = self.unpack(x)
        template = replace(self.seed, gas_pressure=pressure)
        state = self.anchor
        if state is None or self.model.population_nlte_fraction == 0:
            from ._pg1159_reference import planck_state

            a = self.model.rebuild_atmosphere(template, temperature, None)
            a, state = planck_state(self.model, a, self.wave)
            if self.model.population_nlte_fraction == 0:
                return a, state
        # Use the model's declared material target at every stage.  On the
        # PG1424 validation case the plain map has a 4.2e-3 numerical floor;
        # ten further iterations change the normalized optical spectrum by
        # 1.6e-4 RMS (1.7e-3 maximum).  Tighter targets therefore spend the
        # runtime fitting iteration noise rather than changing the spectrum.
        # The independent raw-map and electron rebuild below still test every
        # returned state, including temporary continuation states.
        material_ceiling = self.material_tolerance_ceiling
        rate_tolerance = min(
            self.model.metal_population_relative_tolerance,
            material_ceiling,
        )
        raw_rate_tolerance = rate_tolerance
        charge_tolerance = rate_tolerance
        # A supplied warm population is already a legitimate fixed-point
        # candidate. Requiring the public three-iteration warmup can move an
        # independently closed state away from its root before testing it.
        # The explicit one-pass defect and electron checks below still certify
        # every returned material state.
        closure_model = replace(
            self.model,
            use_population_ali=bool(self.model.use_population_ali or self.population_ali),
            metal_population_relative_tolerance=rate_tolerance,
            metal_population_minimum_iterations=1,
            metal_population_iterations=(
                self.model.metal_population_iterations
                if maximum_population_iterations is None
                else min(
                    self.model.metal_population_iterations,
                    maximum_population_iterations,
                )
            ),
        )
        from ._pg1159_acceleration import PopulationHistory
        history = PopulationHistory()
        previous_electrons = None
        defect = accepted_update = charge = np.inf
        # The inner loop stops on its own iterate update, which can sit just
        # below the gate while the independent plain-Lambda defect sits just
        # above it; each such near miss then tightens the inner stop.
        inner_stop = [raw_rate_tolerance]
        stopped_on_inner_target = [False]
        for closure_iteration in range(maximum_closure_iterations):
            stopped_on_inner_target[0] = False
            a = self.model.rebuild_atmosphere(template, temperature, state)
            if previous_electrons is not None:
                small_density_change = np.max(abs(np.log(
                    a.electron_density / previous_electrons))) < 1e-3
                history.restart(self.model.coupled_population_acceleration_depth,
                                retain=small_density_change)
                if history.retained:
                    LOGGER.info("Reusing %d PG1159 population secants after electron refresh",
                                len(history.retained))
            previous_electrons = a.electron_density.copy()

            def population_progress(iteration, population):
                if iteration == 1 or iteration % 10 == 0:
                    candidates = (
                        population.metadata.get("undamped_metal_population_worst_change", {}),
                        population.metadata.get("undamped_helium_population_worst_change", {}),
                    )
                    worst = max(
                        candidates,
                        key=lambda item: float(item.get("relative_change", 0.0)),
                    )
                    LOGGER.info(
                        "PG1159 population iteration %d: update %.4g, raw rate update %.4g, worst %s",
                        iteration,
                        population.metadata["metal_population_relative_change"],
                        population.metadata["undamped_population_relative_change"],
                        worst,
                    )

            raw_updates = []

            def restart_stagnant_history(update):
                # Secants retained across an EOS refresh can dominate the
                # Anderson fit for a whole history depth and cancel the step:
                # the raw update then stays constant for tens of iterations.
                # Discard the history when ten iterations gain under 5%.
                raw_updates.append(update)
                if (
                    self.resolved_material_closure
                    and len(raw_updates) >= 10
                    and update > 0.95 * raw_updates[-10]
                    and (len(history) or history.retained)
                ):
                    LOGGER.info(
                        "PG1159 population update stagnated at %.4g; restarting "
                        "the coupled history (%d retained, %d samples)",
                        update, len(history.retained), len(history),
                    )
                    history.restart(0, retain=False)
                    raw_updates.clear()

            def refresh_charge_before_oversolving(iteration, population):
                # Once the remaining population update is smaller than the
                # stale electron-EOS error, update the EOS before spending
                # further iterations on rates at the old density. This does
                # not certify either state: the independent checks below
                # retain the final population and charge tolerances.
                update = population.metadata["undamped_population_relative_change"]
                accepted = population.metadata["metal_population_relative_change"]
                restart_stagnant_history(update)
                if update < inner_stop[0]:
                    stopped_on_inner_target[0] = True
                    LOGGER.info(
                        "PG1159 population update %.4g and raw defect %.4g "
                        "satisfy the material gate",
                        accepted,
                        update,
                    )
                    return True
                if iteration < 3 or update >= .01:
                    return False
                refreshed = self.model.rebuild_atmosphere(template, temperature, population)
                pending_charge = float(np.max(abs(refreshed.electron_density / a.electron_density - 1)))
                if pending_charge > max(charge_tolerance, 4 * update):
                    LOGGER.info("Refreshing PG1159 electron EOS: rate update %.4g, charge %.4g", update, pending_charge)
                    return True
                return False

            candidate = replace(
                closure_model, metal_population_relative_tolerance=inner_stop[0]
            ).solve_populations(
                a, state, iteration_callback=population_progress,
                _stop_iteration=refresh_charge_before_oversolving,
                _coupled_population_history=history,
                _coupled_population_near_root_history=(
                    0
                    if (
                        not self.resolved_material_closure
                        and self.model.population_nlte_fraction > 0.0
                        and closure_model.coupled_population_acceleration_depth
                    )
                    else None
                ),
                _coupled_population_near_root_threshold=(
                    3.0 * rate_tolerance
                    if (
                        not self.resolved_material_closure
                        and self.model.population_nlte_fraction > 0.0
                        and closure_model.coupled_population_acceleration_depth
                    )
                    else None
                ),
            )
            # An undamped plain-Lambda update measures the actual SE defect;
            # damped/ALI iteration sizes alone cannot certify convergence.
            check_model = replace(
                self.model,
                metal_population_iterations=1,
                metal_population_minimum_iterations=1,
                metal_population_damping=1.0,
                metal_population_acceleration_depth=0,
                coupled_population_acceleration_depth=0,
                helium_population_damping=1.0,
                use_population_ali=False,
            )
            check = check_model.solve_populations(a, candidate)
            defect = population_defect(candidate, check, self.model)
            accepted_update = float(
                candidate.metadata.get(
                    "metal_population_relative_change", defect
                )
            )
            defect_worst = population_defect_diagnostic(candidate, check, self.model)
            refreshed = self.model.rebuild_atmosphere(template, temperature, candidate)
            charge = float(
                np.max(abs(refreshed.electron_density / a.electron_density - 1))
            )
            state = candidate
            LOGGER.info(
                "PG1159 material %d: rate defect %.4g, charge %.4g, worst %s",
                closure_iteration,
                defect,
                charge,
                defect_worst,
            )
            if (
                defect < raw_rate_tolerance
                and charge < charge_tolerance
            ):
                break
            # Tighten only after a near miss: the inner loop met its own
            # target but the independent check did not.  A pass that ran out
            # of iterations far from the root says nothing about the target,
            # and tightening then can push it below the plain map's
            # reachable floor.
            if defect >= raw_rate_tolerance and stopped_on_inner_target[0]:
                inner_stop[0] *= 0.5
        state = replace(
            state,
            metadata={
                **state.metadata,
                "metal_population_converged": bool(
                    defect < raw_rate_tolerance
                    and charge < charge_tolerance
                ),
                "accepted_population_update": accepted_update,
                "undamped_population_defect": defect,
                "electron_closure_residual": charge,
                "material_population_tolerance": rate_tolerance,
                "material_undamped_population_tolerance": raw_rate_tolerance,
                "material_electron_tolerance": charge_tolerance,
                "material_population_worst_change": defect_worst,
            },
        )
        if not state.converged:
            raise PG1159MaterialClosureError(
                "PG1159 material did not close: rate defect %.6g, charge %.6g"
                % (defect, charge)
            )
        return a, state

    def directional_residual(self, x):
        """Return a bounded-cost residual for optional Jacobian refinement."""
        try:
            a, p = self.material(
                x,
                maximum_closure_iterations=2,
                maximum_population_iterations=20,
            )
            c = self.model.transfer_coefficients(a, self.wave, p)
            return self.field_residual(a, p, coefficients=c).residual
        except (NonphysicalPopulationError, InvalidRadiationFieldError) as exc:
            raise RecoverableEvaluationError(str(exc)) from exc

    def field_residual(self, a, p, *, diagnostics=True, coefficients=None):
        c = (
            self.model.transfer_coefficients(a, self.wave, p)
            if coefficients is None
            else coefficients
        )
        if (
            np.any(~np.isfinite(c.true_absorption))
            or np.any(~np.isfinite(c.thermal_emissivity))
            or np.any(~np.isfinite(c.scattering))
            or np.any(c.total_extinction <= 0)
            or np.any(c.true_absorption[:, -1] <= 0)
            or np.any(c.thermal_emissivity < 0)
            or np.any(c.scattering < 0)
        ):
            raise RecoverableEvaluationError(
                "inadmissible PG1159 trial opacity/emissivity"
            )
        source, field, closure = transfer_field(
            a,
            c,
            n_angle=self.model.helium_model.population_n_angle,
            check_source=diagnostics,
        )
        flux = trapezoid(field.interface_flux, self.wave, axis=0) / self.target - 1
        energy, emission = mass_emissivity_energy(
            self.wave,
            a.column_mass,
            c.thermal_emissivity,
            field.mean_intensity,
            c.true_absorption,
        )
        scaled = energy / np.maximum(abs(emission), 1e-30 * self.target)
        local_energy_mask = _local_energy_mask(a, self.model)
        excluded_surface_mask = (
            np.asarray(a.rosseland_optical_depth[:-1])
            < float(
                getattr(
                    self.model,
                    "atmosphere_local_energy_minimum_rosseland_depth",
                    PG1159_LOCAL_ENERGY_MINIMUM_ROSSELAND_DEPTH,
                )
            )
        )
        photospheric_flux_mask = _photospheric_flux_mask(a, self.model)
        local_indices = np.flatnonzero(local_energy_mask)
        maximum_local_index = int(
            local_indices[np.argmax(abs(scaled[local_energy_mask]))]
        )
        largest_count = min(8, local_indices.size)
        largest_local_indices = local_indices[
            np.argsort(abs(scaled[local_energy_mask]))[-largest_count:][::-1]
        ]
        # At full NLTE solve exactly the quantity used by the release gate:
        # the bolometric flux at every interface.  The former local-heating
        # residual could shrink while the flux profile worsened because its
        # per-cell normalization hid accumulated absolute flux drift.  Local
        # heating remains an independently recorded qualification check.
        full_nlte_flux_solve = self.model.population_nlte_fraction == 1.0
        physical_residual = flux if full_nlte_flux_solve else np.r_[
            np.where(local_energy_mask, scaled, 0.0), flux[0]
        ]
        if full_nlte_flux_solve and self.hybrid_band == "auto":
            # Freeze the band at the stage's first (anchor) evaluation so the
            # residual rows stay fixed for the whole solve.
            first = int(np.argmax(np.r_[local_energy_mask, True]))
            last = int(np.argmax(np.r_[abs(emission) >= self.target, True]))
            self.hybrid_band = (first, last) if first < last else None
        if full_nlte_flux_solve and self.hybrid_band is not None:
            first, last = self.hybrid_band
            physical_residual = np.r_[flux[:first], scaled[first:last], flux[last:]]
        acceleration = (
            trapezoid(c.total_extinction * field.flux, self.wave, axis=0) / LIGHT_SPEED
        )
        hydro = 0.0
        target_pressure = None
        if self.radiative_acceleration:
            effective = a.gravity - self.radiative_acceleration_scale * acceleration
            target_pressure = np.r_[
                effective[0] * a.column_mass[0],
                effective[0] * a.column_mass[0]
                + np.cumsum(
                    0.5 * (effective[1:] + effective[:-1]) * np.diff(a.column_mass)
                ),
            ]
            if np.any(target_pressure <= 0):
                raise RecoverableEvaluationError(
                    "radiative force gives nonpositive hydrostatic gas pressure"
                )
            pressure = np.log(a.gas_pressure / target_pressure)
            physical_residual = np.r_[physical_residual, pressure]
            hydro = float(np.max(abs(pressure)))
        operator_correction, proposed_correction = _legacy_operator_split_correction(
            a, self.model, self.wave, c, field, flux, target_pressure
        )
        # Temporary continuation stages solve the legacy fixed-point map, so
        # expose that map's correction as their nonlinear residual. This lets
        # the common driver's merit test accept a contraction even when an
        # individual physical residual is briefly non-monotone. The final
        # full-NLTE certification stage always exposes the physical flux,
        # heating, and hydrostatic residuals below.
        residual = (
            physical_residual
            if self.certification_stage or full_nlte_flux_solve
            else np.asarray(operator_correction, dtype=float)
        )
        d = dict(
            maximum_all_depth_total_flux_residual=float(np.max(abs(flux))),
            maximum_photospheric_total_flux_residual=float(
                np.max(abs(flux[photospheric_flux_mask]))
            ),
            photospheric_flux_maximum_rosseland_depth=float(
                getattr(
                    self.model,
                    "atmosphere_local_energy_maximum_rosseland_depth",
                    10.0,
                )
            ),
            maximum_relative_cell_energy_balance_residual=float(
                np.max(abs(scaled[local_energy_mask]))
            ),
            maximum_local_energy_balance_depth_index=maximum_local_index,
            maximum_local_energy_balance_rosseland_depth=float(
                a.rosseland_optical_depth[maximum_local_index]
            ),
            maximum_local_energy_balance_column_mass=float(
                a.column_mass[maximum_local_index]
            ),
            maximum_local_energy_balance_temperature=float(
                a.temperature[maximum_local_index]
            ),
            maximum_local_energy_balance_signed_residual=float(
                scaled[maximum_local_index]
            ),
            maximum_local_energy_balance_net_heating=float(
                energy[maximum_local_index]
            ),
            maximum_local_energy_balance_emission_scale=float(
                emission[maximum_local_index]
            ),
            largest_local_energy_balance_cells=tuple(
                {
                    "depth_index": int(depth_index),
                    "rosseland_depth": float(
                        a.rosseland_optical_depth[depth_index]
                    ),
                    "signed_residual": float(scaled[depth_index]),
                }
                for depth_index in largest_local_indices
            ),
            maximum_all_cell_energy_balance_residual=float(np.max(abs(scaled))),
            maximum_excluded_surface_cell_energy_balance_residual=(
                float(np.max(abs(scaled[excluded_surface_mask])))
                if np.any(excluded_surface_mask)
                else None
            ),
            local_energy_minimum_rosseland_depth=float(
                getattr(
                    self.model,
                    "atmosphere_local_energy_minimum_rosseland_depth",
                    PG1159_LOCAL_ENERGY_MINIMUM_ROSSELAND_DEPTH,
                )
            ),
            local_energy_maximum_rosseland_depth=float(
                getattr(
                    self.model,
                    "atmosphere_local_energy_maximum_rosseland_depth",
                    10.0,
                )
            ),
            maximum_legacy_proposed_log_temperature_correction=proposed_correction,
            pg1159_hybrid_band=self.hybrid_band,
            _operator_split_correction=operator_correction,
            # Reused by the hybrid temperature correction, which would
            # otherwise repeat this formal solution.
            _transfer_coefficients=c,
            _radiation_field=field,
            electron_scattering_source_final_maximum_relative_residual=closure,
            maximum_hydrostatic_log_pressure_residual=hydro,
            maximum_radiative_acceleration_fraction=float(
                np.max(acceleration / a.gravity)
            ),
            nlte_populations_converged=p.converged,
            nlte_maximum_relative_population_change=p.metadata.get(
                "undamped_population_defect"
            ),
            electron_closure_residual=p.metadata.get("electron_closure_residual"),
            surface_flux_ratio=float(flux[0] + 1),
        )
        if diagnostics:
            _, boundary = mass_emissivity_field(
                optical_depth_from_mass_opacity(a.column_mass, c.total_extinction),
                np.zeros_like(c.thermal_emissivity),
                c.true_absorption,
                c.scattering,
                bottom_source=c.thermal_emissivity[:, -1] / c.true_absorption[:, -1],
                column_mass=a.column_mass,
                n_angle=self.model.helium_model.population_n_angle,
                wavelength_chunk_size=512,
            )
            d["lower_boundary_absorption_escape_bound"] = float(
                abs(trapezoid(boundary.interface_flux[:, 0], self.wave)) / self.target
            )
        return NonlinearEvaluation(residual, None, (a, p, d))

    def evaluate(self, x, jacobian):
        key = x.tobytes()
        try:
            if key not in self.cache:
                self.evaluations += 1
                a, p = self.material(x)
                c = self.model.transfer_coefficients(a, self.wave, p)
                self.cache[key] = self.field_residual(a, p, coefficients=c)
                trial = self.cache[key]
                self.trace = [*self.trace[-(TRACE_LENGTH - 1):],
                              (x.copy(), trial.residual.copy())]
                trial_diagnostics = trial.payload[2]
                LOGGER.info(
                    "PG1159 residual evaluation %d: merit %.6g, maximum residual %.6g, "
                    "photospheric flux %.6g, all-depth flux %.6g, local energy %.6g, "
                    "proposed log-T correction %.6g",
                    self.evaluations,
                    0.5 * float(np.mean(trial.residual**2)),
                    float(np.max(abs(trial.residual))),
                    trial_diagnostics["maximum_photospheric_total_flux_residual"],
                    trial_diagnostics["maximum_all_depth_total_flux_residual"],
                    trial_diagnostics["maximum_relative_cell_energy_balance_residual"],
                    trial_diagnostics["maximum_legacy_proposed_log_temperature_correction"],
                )
                self.coefficients[key] = c
                if len(self.coefficients) > 2:
                    old = next(iter(self.coefficients))
                    del self.coefficients[old]
                if len(self.cache) > 4:
                    old = next(iter(self.cache))
                    del self.cache[old]
            result = self.cache[key]
            if not jacobian:
                return result
            a, p, d = result.payload
            if not getattr(self.model, "use_pg1159_response_jacobian", False):
                # The first physical trial in a stage must start from the
                # material state just closed at its anchor. Otherwise it
                # repeats the entire population transition from the previous
                # continuation fraction before every first trial.
                self.anchor = p
                diagnostics = {
                    **d,
                    "population_response_probe_count": 0,
                    "population_response_directional_defects": (),
                    "population_response_direction_validated": False,
                    "population_response_measured_rank": 0,
                    "population_response_probe_failed": False,
                    "population_response_probe_failure": None,
                }
                # The PG-specific operator split supplies the proposal. The
                # common driver still requires a finite square Jacobian for
                # bookkeeping, but the proposal does not consume it. Avoid
                # expensive, noisy response probes in production stages.
                return NonlinearEvaluation(
                    result.residual, np.eye(len(x)), (a, p, diagnostics)
                )
            # The common driver requests derivatives at accepted states.
            # Reuse this closed material state for the first trial too, rather
            # than repeating the previous continuation stage's population solve.
            self.anchor = p
            step = 1e-4
            from ._pg1159_response import thermal_response

            c = self.coefficients.get(key)
            if c is None:
                c = self.model.transfer_coefficients(a, self.wave, p)
            probes = []
            failed_local_probes = 0
            for offset in range(0, len(x), self.nd):
                local_step = step
                pair = None
                failure = None
                for _attempt in range(4):
                    candidate_pair = []
                    try:
                        for sign in (1, -1):
                            trial = x.copy()
                            trial[offset : offset + self.nd] += sign * local_step
                            t, pressure = self.unpack(trial)
                            if self.model.population_nlte_fraction == 0:
                                from ._pg1159_reference import planck_state

                                pa = self.model.rebuild_atmosphere(
                                    replace(a, gas_pressure=pressure), t, None
                                )
                                pa, pp = planck_state(self.model, pa, self.wave)
                            else:
                                pa, pp = rate_response_material(
                                    self.model, a, t, pressure, p
                                )
                            candidate_pair.append(
                                self.model.transfer_coefficients(pa, self.wave, pp)
                            )
                    except (
                        RecoverableEvaluationError,
                        NonphysicalPopulationError,
                        InvalidRadiationFieldError,
                    ) as exc:
                        failure = exc
                        local_step *= .5
                        continue
                    pair = candidate_pair
                    break
                if pair is None:
                    # This is only the local coefficient part of an approximate
                    # Jacobian. A zero derivative is conservative here; the
                    # coupled directional probe and the common driver's full
                    # physical trial still measure the missing response.
                    failed_local_probes += 1
                    LOGGER.info(
                        "PG1159 local material-response probe unavailable: %s",
                        failure,
                    )
                    pair = [c, c]
                probes.append((*pair, local_step))
            matrix = thermal_response(
                a,
                self.wave,
                c,
                probes,
                self.target,
                self.model.helium_model.population_n_angle,
                hydrostatic=self.radiative_acceleration,
                radiative_acceleration_scale=self.radiative_acceleration_scale,
                local_energy_mask=_local_energy_mask(a, self.model),
                flux_profile_residual=(
                    self.model.population_nlte_fraction == 1.0
                ),
                hybrid_band=self.hybrid_band,
            )
            d = {
                **d,
                "local_material_response_probe_failures": failed_local_probes,
            }
            if self.model.population_nlte_fraction > 0.0:
                from ._pg1159_response import refine_population_response

                def measure(direction):
                    return measured_directional_response(
                        self.directional_residual,
                        x,
                        result.residual,
                        direction,
                    )

                def propose(current):
                    return damped_thermal_direction(x, result, current, .04).direction

                def logged_measure(direction):
                    measured = measure(direction)
                    LOGGER.info("Measured PG1159 coupled thermal response direction")
                    return measured

                probe_limit = int(
                    getattr(
                        self.model,
                        "pg1159_population_response_probes",
                        _population_response_probe_limit(
                            self.model.population_nlte_fraction,
                            self.certification_stage,
                        ),
                    )
                )
                if probe_limit < 0:
                    raise ValueError(
                        "pg1159_population_response_probes must be nonnegative"
                    )
                # The fixed-radiation material/transfer tangent already gave
                # a tenfold flux reduction once the profile residual fell
                # below 0.5.  A coupled population probe at that point doubled
                # Jacobian time without improving the accepted direction.
                # Retain one measured correction only for far-out states.
                population_probe_residual_threshold = 0.5
                if self.resolved_material_closure:
                    # Short-closure population probes are too noisy near the
                    # root; the resolved closure instead corrects the tangent
                    # with directional derivatives measured by the rejected
                    # line searches themselves (see secant_corrected).
                    probe_limit = 0
                elif (
                    float(np.max(abs(result.residual)))
                    < population_probe_residual_threshold
                ):
                    probe_limit = 0
                if probe_limit:
                    matrix, response_diagnostics = refine_population_response(
                        matrix,
                        result.residual,
                        logged_measure,
                        propose,
                        maximum_probes=probe_limit,
                    )
                else:
                    response_diagnostics = {
                        "population_response_probe_count": 0,
                        "population_response_directional_defects": (),
                        "population_response_direction_validated": False,
                        "population_response_measured_rank": 0,
                        "population_response_probe_failed": False,
                        "population_response_probe_failure": None,
                    }
                d = {**d, **response_diagnostics}
                d["population_response_probe_residual_threshold"] = (
                    population_probe_residual_threshold
                )
                LOGGER.info("PG1159 coupled thermal response: %s", response_diagnostics)
            if self.resolved_material_closure and self.secant_corrections:
                matrix = secant_corrected(matrix, self.secant_corrections)
                d = {**d, "pg1159_secant_corrections": len(self.secant_corrections)}
                LOGGER.info("Applied %d measured PG1159 secant corrections",
                            len(self.secant_corrections))
            return NonlinearEvaluation(result.residual, matrix, (a, p, d))
        except (NonphysicalPopulationError, InvalidRadiationFieldError) as exc:
            LOGGER.info("Rejected PG1159 material trial: %s", exc)
            raise RecoverableEvaluationError(str(exc)) from exc
        except ValueError as exc:
            # A trial whose NLTE opacity gives non-positive transfer faces
            # (a local population inversion) is outside the physical domain,
            # like a nonphysical population: shorten the step, do not abort.
            if "optical face resistances" not in str(exc):
                raise
            LOGGER.info("Rejected PG1159 transfer trial: %s", exc)
            raise RecoverableEvaluationError(str(exc)) from exc


def damped_thermal_direction(state, evaluation, jacobian, radius):
    """Minimize the local residual model within the thermal trust bound.

    Scaling one large Newton direction can retain a weak, strongly curved
    thermal mode. Tikhonov damping instead solves the whole local least-squares
    model. The shared nonlinear driver still measures and accepts every trial.
    """
    from .nonlinear import NonlinearProposal

    u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    projected = u.T @ (-evaluation.residual)
    cutoff = max(float(singular[0]), np.finfo(float).tiny) * 1e-10
    inverse = np.divide(
        1.0, singular, out=np.zeros_like(singular), where=singular > cutoff
    )
    newton = vt.T @ (inverse * projected)
    if np.all(singular > cutoff) and np.max(abs(newton)) <= radius:
        return NonlinearProposal(newton)

    def damped(penalty):
        return vt.T @ (singular / (singular * singular + penalty) * projected)

    low = 0.0
    high = max(float(singular[0] ** 2) * 1e-12, 1e-16)
    while np.max(abs(damped(high))) > radius:
        high *= 10.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.max(abs(damped(middle))) > radius:
            low = middle
        else:
            high = middle
    return NonlinearProposal(damped(high), limited=True)


def _initializer_ready(
    evaluation,
    flux_tolerance,
    local_energy_tolerance=None,
    *,
    require_flux_profile=False,
):
    """Test physical handoff metrics, independent of residual coordinates."""
    if local_energy_tolerance is None:
        local_energy_tolerance = flux_tolerance
    diagnostics = evaluation.payload[2]
    ready = bool(
        evaluation.payload[1].converged
        and abs(diagnostics["surface_flux_ratio"] - 1.0) < flux_tolerance
        and diagnostics["maximum_relative_cell_energy_balance_residual"]
        < local_energy_tolerance
        and diagnostics["maximum_hydrostatic_log_pressure_residual"]
        < flux_tolerance
    )
    if require_flux_profile:
        ready = bool(
            ready
            and diagnostics["maximum_photospheric_total_flux_residual"]
            < flux_tolerance
            and diagnostics["maximum_all_depth_total_flux_residual"]
            < flux_tolerance
        )
    return ready


def _operator_split_direction(state, evaluation, jacobian, radius):
    """Supply the legacy PG1159 map to the common globalization driver."""
    del state, jacobian, radius
    return np.asarray(
        evaluation.payload[2]["_operator_split_correction"], dtype=float
    )


def _solve_stage_driver(
    eq,
    state,
    maximum_iterations,
    *,
    model,
    residual_tolerance,
    step_tolerance,
    qualified,
    ready_for_next_phase,
    certification_stage,
    callback,
    rejected_step_handoff,
):
    """Run the shared trust-region driver for one PG1159 stage residual."""
    return solve_trust_region_newton(
        state,
        eq.evaluate,
        maximum_iterations=maximum_iterations,
        # A four-percent temperature proposal drives the eliminated He/C/O
        # populations several tenths away from their rate fixed point and can
        # spend many EOS cycles relearning weak modes. Start final NLTE at one
        # percent and permit conservative growth after successful trials.
        initial_trust_radius=(
            0.05 if model.population_nlte_fraction == 1.0
            else 0.04
        ),
        maximum_trust_radius=(
            0.05 if model.population_nlte_fraction == 1.0
            else 0.12
        ),
        residual_tolerance=residual_tolerance,
        step_tolerance=step_tolerance,
        # The certification test checks every certificate gate itself, so a
        # proposal-limited sub-tolerance step (e.g. a radiation-pressure
        # re-balance after refinement) need not run to the iteration cap.
        converge_when_proposal_limited=bool(certification_stage),
        convergence_test=qualified,
        # A provisional Planck-blended problem only prepares the next phase;
        # it need not have a stationary root of its own. The explicit handoff
        # reports converged=False. Full NLTE still requires the independent
        # residual certificate and material checks.
        accepted_state_handoff=(
            ready_for_next_phase if not certification_stage else None
        ),
        allow_initial_convergence=True,
        jacobian_refresh_interval=1 if model.population_nlte_fraction == 0.0 else 4,
        finite_difference_fallback_step=None,
        broyden_updates=False,
        linear_regularization=0.0,
        step_builder=(
            _operator_split_direction
            if not getattr(model, "use_pg1159_response_jacobian", False)
            else damped_thermal_direction
        ),
        merit_function="least-squares",
        # The common model-agreement policy intentionally refuses radius growth
        # for limited proposals. LM proposals are intrinsically bounded, so
        # retain the driver's backtracking-based growth policy here.
        trust_update="legacy",
        callback=callback,
        rejected_step_handoff=rejected_step_handoff,
    )


def _solve_stage(
    seed,
    model,
    wave,
    *,
    maximum_iterations=60,
    include_radiative_acceleration=True,
    iteration_callback=None,
    initial_population_state=None,
    radiative_acceleration_scale=1.0,
    certification_stage=None,
    material_tolerance_ceiling=None,
    provisional_thermal_tolerance=1e-2,
    stage_name=None,
    closure_state=None,
):
    start = time.monotonic()
    if closure_state is not None:
        carried_tolerance = float(closure_state["tolerance"])
        model = replace(
            model,
            metal_population_relative_tolerance=min(
                model.metal_population_relative_tolerance, carried_tolerance
            ),
        )
        material_tolerance_ceiling = (
            carried_tolerance
            if material_tolerance_ceiling is None
            else min(float(material_tolerance_ceiling), carried_tolerance)
        )
    certification_stage = (
        model.population_nlte_fraction == 1.0
        if certification_stage is None
        else bool(certification_stage)
    )
    if stage_name is None:
        stage_name = "certification" if certification_stage else "initializer"
    eq = PG1159Equations(
        seed,
        model,
        wave,
        radiative_acceleration=include_radiative_acceleration,
        radiative_acceleration_scale=radiative_acceleration_scale,
        certification_stage=certification_stage,
        material_tolerance_ceiling=material_tolerance_ceiling,
        resolved_material_closure=bool(
            closure_state is not None and closure_state.get("resolved")
        ),
    )
    eq.anchor = initial_population_state
    # A spectrum-qualified final state must conserve the bolometric flux
    # through the declared structure, not only at the surface.  One percent
    # admits the measured cold-origin PG 1424 solution while rejecting the
    # legacy-shape state whose internal flux changes by nearly a factor of two.
    flux_tolerance = 1.0e-2 if certification_stage else provisional_thermal_tolerance
    local_energy_tolerance = (
        3.0e-3
        if certification_stage or stage_name == "full-nlte-relaxation"
        else provisional_thermal_tolerance
    )
    # The generic driver threshold is evaluated before the PG1159-specific
    # convergence test.  It must therefore admit the looser of the two
    # physical residual gates; otherwise it silently reinstates the surface
    # flux tolerance for the independently-scoped local-energy residual.
    residual_tolerance = max(flux_tolerance, local_energy_tolerance)
    step_tolerance = 5.1e-2 if certification_stage else 3e-2

    def ready_for_next_phase(x, e):
        if stage_name in (
            "planck-initializer",
            "half-nlte-population-bridge",
        ):
            return bool(not certification_stage and e.payload[1].converged)
        return bool(
            not certification_stage
            and _initializer_ready(
                e,
                flux_tolerance,
                local_energy_tolerance,
                require_flux_profile=(stage_name == "full-nlte-relaxation"),
            )
        )

    def qualified(x, e, step):
        d = e.payload[2]
        return bool(
            e.payload[1].converged
            and abs(d["surface_flux_ratio"] - 1.0) < flux_tolerance
            and d["maximum_photospheric_total_flux_residual"] < flux_tolerance
            and d["maximum_all_depth_total_flux_residual"] < flux_tolerance
            and d["maximum_relative_cell_energy_balance_residual"] < local_energy_tolerance
            and d["maximum_hydrostatic_log_pressure_residual"] < flux_tolerance
            and step < step_tolerance
        )

    # Iterations already accepted by earlier solves of this stage; nonzero
    # only after a rejected-step material tightening restarts the driver.
    iteration_offset = 0

    def callback(it, x, e):
        eq.anchor = e.payload[1]
        it = replace(it, iteration=it.iteration + iteration_offset)
        d = {
            **{k: v for k, v in e.payload[2].items() if not k.startswith("_")},
            "solver_phase": "shared-pg1159-newton",
            "nlte_continuation_fraction": model.population_nlte_fraction,
            "certification_stage": certification_stage,
            "continuation_stage": stage_name,
            "elapsed_seconds": time.monotonic() - start,
            "nonlinear_step": it.maximum_step,
            "unrestricted_nonlinear_step": it.unrestricted_maximum_step,
            "proposal_limited": it.proposal_limited,
        }
        LOGGER.info("PG1159 iteration %d: %s", it.iteration, d)
        if iteration_callback:
            iteration_callback(it.iteration, e.payload[0], e.payload[1], d)

    # The full-NLTE flux residual is evaluated through an eliminated material
    # closure that stops once one undamped rate update falls below the
    # material tolerance.  Near the root the plain map contracts slowly
    # (spectral radius ~0.97 at PG 1159-035), so a one-percent update can
    # leave the populations several tens of percent from their fixed point
    # and bias the bolometric flux by more than the flux gate itself.  Once
    # that bias dominates, every temperature trial of a whole direction is
    # uphill, including 1/64 steps.  Respond to exactly that event by
    # re-closing the anchor with a tighter material tolerance and restarting
    # the driver on the more accurate residual.  States that never reject a
    # direction (PG 1424, PG 1707) are unaffected.
    #
    # At the minimum tolerance the plain near-root map is almost neutral
    # (spectral radius ~0.9993 at PG 1159-035), so a tighter update gate
    # cannot be reached, and the stopped populations still lag along the
    # slow mode.  That lag reversed the measured slope of the true descent
    # direction.  A rejection there therefore switches once to a resolved
    # closure: coupled Anderson history through the root at
    # pg1159_resolved_material_tolerance.  The analytic tangent also has the
    # wrong sign along that mode, so every rejected line search there adds
    # its measured directional derivative as a persistent secant correction
    # of the tangent (at most MAXIMUM_SECANT_REPAIRS restarts per stage).
    material_tightening = float(
        getattr(model, "pg1159_rejected_step_material_tightening", 0.1)
    )
    minimum_material_tolerance = float(
        getattr(model, "pg1159_minimum_material_tolerance", 1.0e-3)
    )
    resolved_material_tolerance = float(
        getattr(model, "pg1159_resolved_material_tolerance", 1.0e-4)
    )
    if (
        not 0.0 < material_tightening < 1.0
        or minimum_material_tolerance <= 0.0
        or resolved_material_tolerance <= 0.0
    ):
        raise ValueError("PG1159 material tightening settings are invalid")
    material_tightenings = []

    def effective_material_tolerance(equations):
        return min(
            equations.model.metal_population_relative_tolerance,
            equations.material_tolerance_ceiling,
        )

    def can_tighten(equations):
        return (
            effective_material_tolerance(equations) * material_tightening
            >= minimum_material_tolerance * (1.0 - 1.0e-12)
        )

    hybrid_band = (
        closure_state.get("hybrid_band") if closure_state is not None else None
    )
    if hybrid_band is not None and hybrid_band != "auto":
        hybrid_band = tuple(int(v) for v in hybrid_band)
    if (
        hybrid_band is None
        and model.population_nlte_fraction == 1.0
        and bool(getattr(model, "pg1159_hybrid_surface_energy", True))
    ):
        hybrid_band = "auto"
    eq.hybrid_band = hybrid_band
    population_ali = bool(
        closure_state.get("population_ali", False) if closure_state is not None else False
    )
    eq.population_ali = population_ali
    secant_repairs = 0
    secant_corrections = list(
        closure_state.get("secant_corrections", ()) if closure_state is not None else ()
    )
    eq.secant_corrections = list(secant_corrections)

    def tighten_material_on_rejection(x, e):
        if model.population_nlte_fraction != 1.0:
            return False
        if can_tighten(eq) or not eq.resolved_material_closure:
            return True
        return bool(
            secant_repairs < MAXIMUM_SECANT_REPAIRS
            and rejected_direction_secant(getattr(eq, "trace", ()), x) is not None
        )

    state = eq.initial_state()
    remaining_iterations = maximum_iterations
    while True:
        result = _solve_stage_driver(
            eq,
            state,
            remaining_iterations,
            model=model,
            residual_tolerance=residual_tolerance,
            step_tolerance=step_tolerance,
            qualified=qualified,
            ready_for_next_phase=ready_for_next_phase,
            certification_stage=certification_stage,
            callback=callback,
            rejected_step_handoff=tighten_material_on_rejection,
        )
        if result.diagnostics.terminal_reason != "rejected-step-phase-handoff":
            break
        state = result.state
        iteration_offset += result.iterations
        remaining_iterations = max(1, remaining_iterations - result.iterations)
        secant = rejected_direction_secant(getattr(eq, "trace", ()), result.state)
        hybrid_band = getattr(eq, "hybrid_band", hybrid_band)
        if can_tighten(eq):
            action, resolved = "tighten", False
            tolerance = effective_material_tolerance(eq) * material_tightening
        elif not eq.resolved_material_closure:
            action, resolved = "resolve", True
            tolerance = min(effective_material_tolerance(eq), resolved_material_tolerance)
        else:
            action, resolved = "secant", True
            tolerance = effective_material_tolerance(eq)
            secant_repairs += 1
        if resolved and secant is not None:
            secant_corrections = [*secant_corrections, secant][-MAXIMUM_SECANT_CORRECTIONS:]
        LOGGER.info(
            "PG1159 %s: whole direction rejected after %d accepted iterations; "
            "%s: material tolerance %.3g, %d secant corrections",
            stage_name,
            iteration_offset,
            action,
            tolerance,
            len(secant_corrections) if resolved else 0,
        )
        material_tightenings.append(
            {
                "after_iteration": iteration_offset,
                "action": action,
                "material_tolerance": tolerance,
                "resolved_closure": resolved,
                "secant_corrections": len(secant_corrections) if resolved else 0,
                "hybrid_band": hybrid_band,
            }
        )
        anchor = eq.anchor
        evaluations = eq.evaluations
        eq = PG1159Equations(
            seed,
            replace(model, metal_population_relative_tolerance=tolerance),
            wave,
            radiative_acceleration=include_radiative_acceleration,
            radiative_acceleration_scale=radiative_acceleration_scale,
            certification_stage=certification_stage,
            material_tolerance_ceiling=tolerance,
            resolved_material_closure=resolved,
        )
        eq.anchor = anchor
        eq.evaluations = evaluations
        eq.hybrid_band = hybrid_band
        eq.population_ali = population_ali
        if resolved:
            eq.secant_corrections = list(secant_corrections)
    # Only a successful full-physics stage can be certified. Initializers
    # and failed solves do not need another expensive stationarity Jacobian.
    measure_stationarity = False
    final = eq.evaluate(result.state, measure_stationarity)
    a, p, private_diagnostics = final.payload
    d = {k: v for k, v in private_diagnostics.items() if not k.startswith("_")}
    correction = None
    measured = False
    if measure_stationarity:
        correction, _, rank, _ = np.linalg.lstsq(
            final.jacobian, -final.residual, rcond=None
        )
        response_validated = bool(
            model.population_nlte_fraction == 0.0
            or d.get("population_response_direction_validated") is True
        )
        measured = bool(
            response_validated
            and rank == len(result.state)
            and np.all(np.isfinite(correction))
        )
    metadata = {
        **dict(a.metadata),
        **d,
        "radiative_equilibrium_solver": "shared-trust-region-pg1159-nlte",
        "radiative_equilibrium_solver_converged": bool(
            result.converged
            and p.converged
            and model.population_nlte_fraction == 1.0
            and certification_stage
            and radiative_acceleration_scale == 1.0
        ),
        "continuation_initializer_ready": ready_for_next_phase(result.state, final),
        "stage_residual_tolerance": residual_tolerance,
        "stage_flux_tolerance": flux_tolerance,
        "stage_local_energy_tolerance": local_energy_tolerance,
        "stage_step_tolerance": step_tolerance,
        "temperature_correction_measured": bool(measured),
        "temperature_population_response_validated": bool(
            model.population_nlte_fraction == 0.0
            or d.get("population_response_direction_validated") is True
        ),
        "maximum_unrestricted_log_temperature_correction": float(
            np.max(abs(correction[: a.n_depth]))
        )
        if measured
        else None,
        "temperature_tangent": (
            "LTE material derivative"
            if model.population_nlte_fraction == 0.0
            else (
                "fixed-radiation SE/EOS response plus measured population direction"
                if d.get("population_response_measured_rank", 0) > 0
                else "fixed-radiation SE/EOS response with physical trial acceptance"
            )
        ),
        "nonlinear_solver": nonlinear_result_metadata(result),
        "nlte_continuation_fraction": model.population_nlte_fraction,
        "certification_stage": certification_stage,
        "continuation_stage": stage_name,
        "stage_material_tolerance_ceiling": eq.material_tolerance_ceiling,
        "radiative_acceleration_in_hydrostatics": include_radiative_acceleration,
        "radiative_acceleration_scale": radiative_acceleration_scale,
        "elapsed_seconds": time.monotonic() - start,
        "residual_evaluations": eq.evaluations,
        "material_tolerance_tightenings": tuple(material_tightenings),
        "independent_grid_validation": False,
        "full_physics_validation": False,
    }
    metadata["equilibrium_certificate"] = equilibrium_certificate(
        metadata,
        flux_tolerance=flux_tolerance,
        surface_flux_tolerance=flux_tolerance,
        photospheric_flux_tolerance=flux_tolerance,
        local_energy_tolerance=local_energy_tolerance,
        required_checks=(
            "surface_flux",
            "photospheric_flux",
            "all_depth_flux",
            "local_energy",
            "source_closure",
            "boundary_screening",
        ),
        profile="pg1159-spectrum-gate-v1",
    )
    metadata["spectrum_qualified"] = metadata["equilibrium_certificate"]["verified"]
    metadata["radiative_equilibrium_converged"] = False
    hybrid_band = getattr(eq, "hybrid_band", hybrid_band)
    carried_closure = (
        None
        if closure_state is None and not material_tightenings
        else {
            "tolerance": effective_material_tolerance(eq),
            "resolved": eq.resolved_material_closure,
            "secant_corrections": tuple(secant_corrections),
            "hybrid_band": hybrid_band,
            "population_ali": population_ali,
        }
    )
    return PG1159AtmosphereResult(
        replace(a, metadata=metadata), p, result, closure_state=carried_closure
    )


def solve_pg1159_atmosphere(
    seed,
    model,
    wave,
    *,
    maximum_iterations=60,
    include_radiative_acceleration=False,
    iteration_callback=None,
    cold_start=False,
    refinement=None,
):
    """Cold initialization followed by Planck-to-NLTE continuation.

    All stages belong to the requested atmosphere. Stellar parameters,
    composition, atomic rates, and final tolerances remain fixed. Only the
    final fraction-one equations can receive an equilibrium certificate.

    ``refinement=(model, wave, options)`` adds two stages after the
    certification stage: the upper atmosphere is converged to local radiative
    equilibrium on ``model``/``wave`` (:func:`refine_upper_atmosphere`, with
    ``options`` as keyword arguments), and the refined state is certified
    again on the same equations.  The returned certificate then belongs to
    the refined equations.
    """
    stages = (
        (
            (
                0.0,
                False,
                model.metal_population_relative_tolerance,
                5e-1,
                "planck-initializer",
            ),
            (
                0.5,
                False,
                1e-2,
                5e-1,
                "half-nlte-population-bridge",
            ),
            (
                1.0,
                False,
                1e-2,
                1e-2,
                "full-nlte-relaxation",
            ),
            (
                1.0,
                True,
                model.metal_population_relative_tolerance,
                None,
                "full-nlte-certification",
            ),
        )
        if model.population_nlte_fraction == 1.0
        else ((model.population_nlte_fraction, False, 1e-4, 1e-2, "initializer"),)
    )
    gray_initialization = seed.metadata.get("pg1159_gray_opacity_initialization")
    population = None
    closure_state = None
    history = []
    started = time.monotonic()
    for (
        fraction,
        certification_stage,
        material_tolerance_ceiling,
        provisional_thermal_tolerance,
        stage_name,
    ) in stages:
        stage_model = replace(model, population_nlte_fraction=fraction)
        if stage_name in (
            "half-nlte-population-bridge",
            "full-nlte-relaxation",
        ):
            stage_model = replace(
                stage_model,
                metal_population_relative_tolerance=material_tolerance_ceiling,
            )
        if 0.0 < fraction:
            # Use the configured joint history for the half-NLTE bridge and
            # exact full-NLTE map.  The public model selects the validated
            # 80-state weighted-SVD history; truncating it to six here lost
            # the weak population modes and erased most of that speed-up.
            # Near the declared target material() still switches back to the
            # plain physical map.
            stage_model = replace(
                stage_model,
                metal_population_damping=1.0,
                helium_population_damping=1.0,
                metal_population_acceleration_depth=0,
                coupled_population_acceleration_depth=(
                    model.coupled_population_acceleration_depth
                ),
                use_pg1159_response_jacobian=(fraction == 1.0),
            )
        LOGGER.info(
            "PG1159 cold continuation fraction %.3g (%s)",
            fraction,
            "certification" if certification_stage else "initializer",
        )
        result = _solve_stage(
            seed,
            stage_model,
            wave,
            maximum_iterations=maximum_iterations,
            include_radiative_acceleration=include_radiative_acceleration
            and fraction > 0.0,
            radiative_acceleration_scale=fraction
            if include_radiative_acceleration
            else 1.0,
            iteration_callback=iteration_callback,
            initial_population_state=population,
            certification_stage=certification_stage,
            material_tolerance_ceiling=material_tolerance_ceiling,
            provisional_thermal_tolerance=(
                1e-2
                if provisional_thermal_tolerance is None
                else provisional_thermal_tolerance
            ),
            stage_name=stage_name,
            **(
                {"closure_state": closure_state}
                if closure_state is not None and fraction == 1.0
                else {}
            ),
        )
        history.append(
            {
                "fraction": fraction,
                "certification_stage": certification_stage,
                "stage": stage_name,
                "material_tolerance_ceiling": material_tolerance_ceiling,
                "nonlinear_converged": result.nonlinear_result.converged,
                "initializer_ready": result.atmosphere.metadata.get("continuation_initializer_ready", False),
                "iterations": result.nonlinear_result.iterations,
                "elapsed_seconds": result.atmosphere.metadata["elapsed_seconds"],
            }
        )
        seed = result.atmosphere
        population = result.population_state
        closure_state = getattr(result, "closure_state", None)
        if not (result.nonlinear_result.converged
                or result.atmosphere.metadata.get("continuation_initializer_ready", False)):
            break
    refinement_info = None
    if (
        refinement is not None
        and history
        and history[-1]["stage"] == "full-nlte-certification"
    ):
        result, refinement_info, refinement_history = _refine_and_certify(
            seed,
            population,
            refinement,
            maximum_iterations=maximum_iterations,
            include_radiative_acceleration=include_radiative_acceleration,
            iteration_callback=iteration_callback,
        )
        history.extend(refinement_history)
        seed = result.atmosphere
        population = result.population_state
    # Only the public entry point, which constructs its own continuum seed,
    # establishes cold provenance. An arbitrary caller-supplied seed cannot.
    metadata = {
        **seed.metadata,
        "cold_start": bool(cold_start),
        "initialization": {
            "method": (
                "fresh continuum" if cold_start else "caller-supplied atmosphere"
            )
            + " and internal Planck-to-NLTE continuation",
            "previous_model_supplied": False if cold_start else None,
            "stellar_parameters_fixed": True,
            "gray_opacity_preconditioning": gray_initialization,
        },
        "continuation_history": history,
        **(
            {"upper_atmosphere_refinement": {k: v for k, v in refinement_info.items() if k != "history"}}
            if refinement_info is not None
            else {}
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    return replace(result, atmosphere=replace(seed, metadata=metadata))


def _refine_and_certify(
    atmosphere,
    population,
    refinement,
    *,
    maximum_iterations=60,
    include_radiative_acceleration=False,
    iteration_callback=None,
):
    """Upper-atmosphere refinement of a certified state, then re-certification.

    ``refinement=(model, wave, options)``.  Returns
    ``(stage result, refinement info, continuation-history entries)``.
    """
    from ._pg1159_ali_temperature import refine_upper_atmosphere

    refinement_model, refinement_wave, refinement_options = refinement
    LOGGER.info("PG1159 upper-atmosphere refinement")
    started = time.monotonic()
    options = dict(refinement_options or {})
    if iteration_callback is not None and "iteration_callback" not in options:
        options["iteration_callback"] = iteration_callback
    # The temperature refinement holds the pressure structure fixed; radiation
    # pressure enters the coupled Newton re-certification below.  Updating the
    # pressure inside the refinement (options["include_radiative_acceleration"])
    # is experimental and did not converge in the PG 1159-035 tests.
    options.setdefault("include_radiative_acceleration", False)
    refined_atmosphere, refined_population, info = refine_upper_atmosphere(
        atmosphere,
        population,
        refinement_model,
        refinement_wave,
        **options,
    )
    history = [
        {
            "fraction": 1.0,
            "certification_stage": False,
            "stage": "upper-atmosphere-refinement",
            "material_tolerance_ceiling": info["material_tolerance"],
            "nonlinear_converged": info["converged"],
            "initializer_ready": True,
            "iterations": info["iterations"],
            "elapsed_seconds": time.monotonic() - started,
        }
    ]
    certification_model = replace(
        refinement_model,
        population_nlte_fraction=1.0,
        metal_population_damping=1.0,
        helium_population_damping=1.0,
        metal_population_acceleration_depth=0,
        coupled_population_acceleration_depth=(
            refinement_model.coupled_population_acceleration_depth
        ),
        use_pg1159_response_jacobian=True,
    )
    result = _solve_stage(
        refined_atmosphere,
        certification_model,
        refinement_wave,
        maximum_iterations=maximum_iterations,
        include_radiative_acceleration=include_radiative_acceleration,
        iteration_callback=iteration_callback,
        initial_population_state=refined_population,
        certification_stage=True,
        material_tolerance_ceiling=refinement_model.metal_population_relative_tolerance,
        provisional_thermal_tolerance=1e-2,
        stage_name="upper-atmosphere-certification",
    )
    history.append(
        {
            "fraction": 1.0,
            "certification_stage": True,
            "stage": "upper-atmosphere-certification",
            "material_tolerance_ceiling": refinement_model.metal_population_relative_tolerance,
            "nonlinear_converged": result.nonlinear_result.converged,
            "initializer_ready": result.atmosphere.metadata.get("continuation_initializer_ready", False),
            "iterations": result.nonlinear_result.iterations,
            "elapsed_seconds": result.atmosphere.metadata["elapsed_seconds"],
        }
    )
    return result, info, history
