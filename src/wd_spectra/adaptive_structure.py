"""Composition-independent adaptive LTE atmosphere-structure solver.

The hydrogen and helium modules supply their own EOS and opacity closures;
this module owns the conservative radiative/convective flux residual, the
surface-temperature plus pressure-gradient variables, and the safeguarded
trust-region iteration.  Keeping those numerical choices here prevents each
spectral-type module from accumulating a separate collection of damping and
restart heuristics.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from ._compat import trapezoid

_LOGGER = logging.getLogger(__name__)
from ._material_response import temperature_response_probes
from ._energy_balance import (
    discrete_radiative_cell_energy_balance,
    integrated_radiative_cell_energy_state_response,
)
from ._rosseland import rosseland_mean_from_opacity_grid
from ._stable_feautrier import (
    cancellation_safe_field,
    cancellation_safe_scalar_field,
    cancellation_safe_response,
)
from .atmosphere import Atmosphere, _upper_interface_values_on_nodes
from .constants import STEFAN_BOLTZMANN
from .convection import (
    ML2CoefficientFunction,
    _ml2_flux_coefficient_response,
    _ml2_local_coefficients_from_thermodynamics,
    ml2_convective_flux_gradient_derivative_from_thermodynamics,
    ml2_convective_flux_for_gradient_from_thermodynamics,
    ml2_temperature_gradient_for_total_flux_from_thermodynamics,
)
from .nonlinear import (
    NonlinearEvaluation,
    NonlinearIteration,
    RecoverableEvaluationError,
    nonlinear_result_metadata,
    solve_trust_region_newton,
)
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import (
    coherent_scattering_feautrier_field,
    emergent_flux,
    feautrier_radiation_field,
    integrated_coherent_scattering_feautrier_state_response,
    integrated_emergent_flux_state_response,
    integrated_feautrier_interface_state_response,
)
from .spectrum import planck_lambda_angstrom


FloatArray = NDArray[np.float64]
IterationCallback = Callable[[int, Atmosphere, Mapping[str, object]], None]
_SCATTERING_SOURCE_NEGATIVE_RELATIVE_TOLERANCE = 1.0e-8


def _positive_interface_values(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    interface = np.empty_like(array)
    interface[0] = array[0]
    interface[1:] = np.sqrt(array[:-1] * array[1:])
    return interface


def _arithmetic_interface_values(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    interface = np.empty_like(array)
    interface[0] = array[0]
    interface[1:] = 0.5 * (array[:-1] + array[1:])
    return interface


def _adiabatic_asymptotic_convection_mask(
    desired_convective_flux: FloatArray,
    desired_gradient: FloatArray,
    adiabatic_gradient: FloatArray,
    log_pressure_step: FloatArray,
    *,
    target_flux: float,
    flux_tolerance: float,
    temperature_tolerance: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Identify ML2 flux rows whose superadiabaticity is unresolved.

    In an extremely efficient convection zone, a sub-tolerance change in the
    nodal temperatures can change the directly evaluated ML2 flux by many
    stellar fluxes.  The local ML2-gradient equation has the same root but is
    numerically well conditioned there.  Return both the implied nodal
    superadiabatic increment and the rows where that equivalent equation is
    required.
    """

    interface_log_pressure_step = np.zeros_like(desired_gradient)
    interface_log_pressure_step[1:] = log_pressure_step
    superadiabatic_temperature_increment = (
        np.maximum(desired_gradient - adiabatic_gradient, 0.0)
        * interface_log_pressure_step
    )
    asymptotic = (desired_convective_flux > flux_tolerance * target_flux) & (
        superadiabatic_temperature_increment <= 0.25 * temperature_tolerance
    )
    asymptotic[0] = False
    return superadiabatic_temperature_increment, asymptotic


def _clip_roundoff_negative_scattering_source(
    source: FloatArray,
) -> tuple[FloatArray, int, float]:
    """Clip only wavelength-local negative source values at numerical scale."""

    candidate = np.asarray(source, dtype=np.float64)
    if np.any(~np.isfinite(candidate)):
        raise ValueError("scattering source solve must remain finite")
    wavelength_scale = np.maximum(
        np.max(np.abs(candidate), axis=1), np.finfo(np.float64).tiny
    )
    negative_relative = np.maximum(
        -candidate / wavelength_scale[:, np.newaxis], 0.0
    )
    maximum_negative_relative = float(np.max(negative_relative))
    if maximum_negative_relative > (
        _SCATTERING_SOURCE_NEGATIVE_RELATIVE_TOLERANCE
    ):
        raise RecoverableEvaluationError(
            "scattering source solve produced materially negative values"
        )
    clipped_count = int(np.count_nonzero(candidate < 0.0))
    if clipped_count:
        candidate = np.maximum(candidate, 0.0)
    return candidate, clipped_count, maximum_negative_relative


def _cap_overcarrying_ml2_gradient(
    gradient: FloatArray,
    transport: Mapping[str, FloatArray],
    target_flux: float,
) -> FloatArray:
    """Propose gradients with at most one stellar flux for frozen local ML2.

    Only overcarrying cells change. Updated temperatures/material properties
    must subsequently be re-evaluated; this never caps or replaces a flux.
    """
    overcarrying = np.asarray(transport["convective_flux"]) > target_flux
    overcarrying[0] = False
    candidate = np.asarray(gradient, dtype=np.float64).copy()
    if np.any(overcarrying):
        element = np.cbrt(
            target_flux / transport["ml2_flux_coefficient"][overcarrying]
        )
        cap = (
            transport["adiabatic_gradient"][overcarrying]
            + transport["ml2_radiative_loss"][overcarrying] * element
            + element**2
        )
        candidate[overcarrying] = np.minimum(candidate[overcarrying], cap)
    return candidate


def _thermal_boundary_absorption_escape_bound(
    wavelength: FloatArray,
    column_mass: FloatArray,
    absorption: FloatArray,
    bottom_planck: FloatArray,
    target_flux: float,
) -> float:
    """Conservative lower-boundary transparency diagnostic, in stellar fluxes.

    Integrate pi B(T_bottom) exp(-tau_abs). Oblique paths and scattering only
    lengthen the absorption path, so this bounds directly escaping thermal
    boundary radiation. A small result is a sufficient screening criterion;
    a large result calls for a deeper-domain test, not a measured flux error.
    This is independent of the discrete total-flux residual and does not
    rescale the spectrum or alter the solved equations.
    """
    bottom_absorption_depth = absorption[:, 0] * column_mass[0] + trapezoid(
        absorption, column_mass, axis=1
    )
    return float(
        trapezoid(
            np.pi * bottom_planck * np.exp(-bottom_absorption_depth),
            wavelength,
        )
        / target_flux
    )


class _PreconditionerFluxHandoff:
    """Caller-scoped progress policy; never an equilibrium certificate."""

    def __init__(
        self,
        initial_state: FloatArray,
        target_flux: float,
        tolerance: float,
        callback: Callable[
            [NonlinearIteration, FloatArray, NonlinearEvaluation[dict[str, object]]],
            None,
        ]
        | None,
    ) -> None:
        self.previous_state = initial_state.copy()
        self.target_flux = target_flux
        self.tolerance = tolerance
        self.original_callback = callback
        self.pending_observation: dict[str, object] | None = None
        self.streak = 0
        self.observed_updates = 0
        self.eligible_updates = 0
        self.ignored_callbacks = 0
        self.triggered = False
        self.observations: list[dict[str, object]] = []

    def callback(
        self,
        record: NonlinearIteration,
        current: FloatArray,
        evaluation: NonlinearEvaluation[dict[str, object]],
    ) -> None:
        genuine = (
            record.step_kind == "newton"
            and np.isfinite(record.line_search_factor)
            and record.line_search_factor > 0.0
            and np.isfinite(record.maximum_step)
            and record.maximum_step > 0.0
            and np.all(np.isfinite(current))
            and not np.array_equal(self.previous_state, current)
        )
        if genuine:
            self.pending_observation = {
                "iteration": int(record.iteration),
                "line_search_factor": float(record.line_search_factor),
                "measured_log_temperature_step": float(record.maximum_step),
                "step_kind": record.step_kind,
                "proposal_limited": bool(record.proposal_limited),
            }
            self.observed_updates += 1
        else:
            self.pending_observation = None
            self.streak = 0
            self.ignored_callbacks += 1
        self.previous_state = current.copy()
        if self.original_callback is not None:
            self.original_callback(record, current, evaluation)

    def handoff(
        self,
        current: FloatArray,
        evaluation: NonlinearEvaluation[dict[str, object]],
    ) -> bool:
        if self.pending_observation is None:
            return False
        observation = self.pending_observation
        self.pending_observation = None
        if not np.array_equal(self.previous_state, current):
            raise ValueError("preconditioner handoff anchor differs from accepted callback")
        flux = float(
            np.max(
                np.abs(
                    np.asarray(evaluation.payload["total_flux_interface"])
                    / self.target_flux
                    - 1.0
                )
            )
        )
        residual = float(np.max(np.abs(evaluation.residual)))
        energy = float(
            np.max(
                np.abs(evaluation.payload["cell_energy_balance_relative_residual"])
            )
        )
        finite = bool(np.all(np.isfinite((flux, residual, energy))))
        eligible = bool(
            finite
            and flux < self.tolerance
            and residual < self.tolerance
            and energy >= self.tolerance
        )
        self.streak = self.streak + 1 if eligible else 0
        self.eligible_updates += int(eligible)
        self.observations.append(
            {
                **observation,
                "eligible": eligible,
                "streak": self.streak,
                "flux": flux if np.isfinite(flux) else None,
                "phase_residual": residual if np.isfinite(residual) else None,
                "energy": energy if np.isfinite(energy) else None,
            }
        )
        self.triggered = self.streak >= 2
        return self.triggered

    def metadata(self) -> dict[str, object]:
        return {
            "observed_accepted_newton_updates": self.observed_updates,
            "eligible_accepted_newton_updates": self.eligible_updates,
            "consecutive_eligible_updates": self.streak,
            "ignored_callback_observations": self.ignored_callbacks,
            "triggered": self.triggered,
            "observations": tuple(dict(item) for item in self.observations),
        }


def solve_adaptive_lte_structure(
    seed: Atmosphere,
    wavelength: FloatArray,
    *,
    with_temperature: Callable[[FloatArray], Atmosphere],
    true_absorption: Callable[[Atmosphere], FloatArray],
    scattering_opacity: Callable[[Atmosphere], FloatArray],
    rosseland_opacity: Callable[[Atmosphere], FloatArray],
    thermodynamics: Callable[[Atmosphere], object],
    mixing_length_alpha: float | None,
    max_iterations: int,
    temperature_tolerance: float,
    flux_tolerance: float,
    n_angle: int,
    initial_temperature_was_supplied: bool,
    resume_supplied_structure_in_formal_flux_phase: bool | None = None,
    temperature_normalization_anchor_rosseland_depth: float | None = None,
    maximum_formal_flux_rosseland_depth: float | None = None,
    formal_flux_taper_start_rosseland_depth: float | None = None,
    project_initial_convective_gradient: bool = True,
    initial_convective_gradient_projection_mode: Literal[
        "interface-transport", "unstable-node-gradient"
    ] = "interface-transport",
    use_convective_gradient_preconditioner: bool = True,
    maximum_convective_preconditioner_iterations: int | None = None,
    preconditioner_stationary_completion_iterations: int | None = 2,
    use_preconditioner_flux_handoff: bool = True,
    maximum_formal_flux_continuations: int = 2,
    use_adiabatic_asymptotic_conditioning: bool = False,
    use_initial_bolometric_rescaling: bool = True,
    safeguard_surface_flux: bool = False,
    use_convective_trial_correction: bool = False,
    compute_local_energy_response: bool = False,
    reuse_material_probe_rosseland: bool = False,
    enforce_local_energy_balance: bool = False,
    physical_temperature_coordinates: bool = True,
    energy_thermal_sweeps: int = 40,
    use_precision_polish: bool = True,
    transfer_discretization: Literal[
        "optical-depth", "column-mass"
    ] = "optical-depth",
    column_mass_radiation_backend: object | None = None,
    ml2_coefficient_function: ML2CoefficientFunction | None = None,
    iteration_callback: IterationCallback | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Atmosphere:
    """Solve one LTE atmosphere with adaptive Newton/ML2 transport.

    The nonlinear variables are one normalization value of ``ln(T)`` and one
    ``dln(T)/dln(P)`` value per interface.  The normalization is at the
    surface by default.  A composition module may instead anchor it at a
    specified Rosseland depth when an optically thin surface must not be
    driven by the homologous/global part of an initial Newton step.  In
    convective cells a local ML2 gradient equation first preconditions the
    nearly adiabatic structure; a second phase then imposes the conservative
    formal-transfer total flux at every interface.  Composition enters only
    through the callbacks. ``initial_temperature_was_supplied`` records seed
    provenance; the separate resume option decides whether that seed is an
    exact checkpoint that may bypass conditioning. The projection mode and
    stationary-completion setting preserve validated composition-specific
    initialization policies without duplicating the nonlinear solver.
    ``use_preconditioner_flux_handoff`` (enabled by default) is a progress
    policy for the first convective-gradient preconditioner. With
    local energy enforcement enabled, two genuine accepted Newton updates
    with balanced actual flux and provisional residual but unresolved local
    energy end that phase as unconverged. Formal and energy completion plus
    all original final qualification gates remain authoritative. A requested
    policy is unused without that preconditioner or local energy enforcement.
    ``use_convective_trial_correction`` is an experimental, disabled-by-default
    repair of overcarrying ML2 trial gradients. It does not cap evaluated
    fluxes or bypass any acceptance or final physical convergence tests.
    ``compute_local_energy_response`` adds a direct thermal-energy tangent
    to the evaluation payload for explicitly requested research formulations.
    It is off by default and does not change the residual equations.
    ``reuse_material_probe_rosseland`` retains each centered material probe's
    Rosseland vector while its opacity is still cached by the composition
    callback. This avoids rebuilding its opacity for the convection response.
    Only depth-sized vectors are retained within the current linearization;
    the default preserves the callback order of composition-owned adapters.
    ``enforce_local_energy_balance`` imposes flux and local energy from the
    first step when convection is disabled. Convective starts add a completion
    phase when the original formal-flux solution fails cell-local energy or
    has no measured final correction. Bounded thermal conditioning stabilizes
    the temperature field,
    then an unpenalized Newton solve imposes both flux and direct cell energy.
    Positive equation weights are frozen within each linearization, never the
    physical material/transfer response. Final certification uses fresh actual
    flux, local energy and the unrestricted correction. ``energy_thermal_sweeps``
    bounds initialization work; it cannot certify equilibrium. Nodal log-T
    coordinates are used by default for this phase, independently of Teff.
    ``use_precision_polish`` completes an already flux-balanced
    warm start with cancellation-resistant transfer arithmetic and an
    unpenalized Newton tangent when its last temperature step is oversized.
    All physical and step checks remain active. The initial conditioning
    trajectory is unchanged; this option is not a temperature/composition split.
    ``transfer_discretization="column-mass"`` explicitly selects physical
    mass control volumes for the field, independent source check, tangent,
    and local energy balance together. It changes no material physics and
    never switches during an iteration. Spectrum synthesis must use the same
    discretization. The established optical-depth default is unchanged.
    ``column_mass_radiation_backend`` and ``ml2_coefficient_function`` are
    explicit composition-owned physics injection points.  They let a DQ
    calculation supply refractive transfer and its corresponding grey ML2
    closure without replacing module globals.  The backend is valid only for
    column-mass transfer; omitting both arguments preserves the established
    operators exactly.
    """

    if transfer_discretization not in ("optical-depth", "column-mass"):
        raise ValueError("unknown transfer discretization")
    mass_transfer = transfer_discretization == "column-mass"
    if column_mass_radiation_backend is not None and not mass_transfer:
        raise ValueError(
            "a column-mass radiation backend requires column-mass transfer"
        )
    if ml2_coefficient_function is None:
        ml2_coefficient_function = _ml2_local_coefficients_from_thermodynamics
    if not callable(ml2_coefficient_function):
        raise ValueError("ml2 coefficient function must be callable")
    if not isinstance(use_preconditioner_flux_handoff, bool):
        raise ValueError("preconditioner flux-handoff flag must be boolean")
    if not isinstance(enforce_local_energy_balance, bool):
        raise ValueError("local energy enforcement flag must be boolean")
    if not isinstance(reuse_material_probe_rosseland, bool):
        raise ValueError("material-probe Rosseland reuse flag must be boolean")
    if not isinstance(physical_temperature_coordinates, bool):
        raise ValueError("temperature-coordinate selection must be boolean")
    if (
        isinstance(energy_thermal_sweeps, bool)
        or not isinstance(energy_thermal_sweeps, int)
        or energy_thermal_sweeps < 0
    ):
        raise ValueError("energy thermal sweeps must be a nonnegative integer")
    if mass_transfer:
        if column_mass_radiation_backend is None:
            from ._mass_feautrier import (
                mass_field,
                mass_response,
                mass_energy,
                mass_energy_response,
            )
            mass_boundary = _thermal_boundary_absorption_escape_bound
        else:
            mass_field = column_mass_radiation_backend.field
            mass_response = column_mass_radiation_backend.response
            mass_energy = column_mass_radiation_backend.energy
            mass_energy_response = column_mass_radiation_backend.energy_response
            mass_boundary = column_mass_radiation_backend.boundary

    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength must be a one-dimensional grid")
    if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be finite and strictly increasing")

    target_flux = STEFAN_BOLTZMANN * seed.effective_temperature**4
    log_pressure = np.log(seed.gas_pressure)
    log_pressure_step = np.diff(log_pressure)
    if np.any(~np.isfinite(log_pressure_step)) or np.any(
        log_pressure_step <= 0.0
    ):
        raise ValueError("seed pressure must be finite and increase inward")
    if temperature_normalization_anchor_rosseland_depth is not None and (
        not np.isfinite(temperature_normalization_anchor_rosseland_depth)
        or temperature_normalization_anchor_rosseland_depth <= 0.0
    ):
        raise ValueError(
            "temperature normalization anchor depth must be positive or None"
        )
    if maximum_formal_flux_rosseland_depth is not None and (
        not np.isfinite(maximum_formal_flux_rosseland_depth)
        or maximum_formal_flux_rosseland_depth <= 0.0
    ):
        raise ValueError(
            "maximum formal-flux Rosseland depth must be positive or None"
        )
    if formal_flux_taper_start_rosseland_depth is not None:
        if maximum_formal_flux_rosseland_depth is None:
            raise ValueError(
                "a formal-flux taper requires a maximum formal-flux depth"
            )
        if (
            not np.isfinite(formal_flux_taper_start_rosseland_depth)
            or formal_flux_taper_start_rosseland_depth <= 0.0
            or formal_flux_taper_start_rosseland_depth
            >= maximum_formal_flux_rosseland_depth
        ):
            raise ValueError(
                "formal-flux taper start must be positive and shallower than "
                "the maximum formal-flux depth"
            )
    if maximum_formal_flux_continuations < 0:
        raise ValueError(
            "maximum formal-flux continuations must be non-negative"
        )
    if not isinstance(use_adiabatic_asymptotic_conditioning, bool):
        raise ValueError(
            "adiabatic-asymptotic conditioning flag must be boolean"
        )
    if not isinstance(use_initial_bolometric_rescaling, bool):
        raise ValueError("initial bolometric rescaling flag must be boolean")
    if not isinstance(use_convective_trial_correction, bool):
        raise ValueError("convective trial correction flag must be boolean")
    if not isinstance(compute_local_energy_response, bool):
        raise ValueError("local energy response flag must be boolean")
    if not isinstance(use_precision_polish, bool):
        raise ValueError("precision polish flag must be boolean")
    if not isinstance(initial_temperature_was_supplied, bool):
        raise ValueError("initial_temperature_was_supplied must be boolean")
    if (
        resume_supplied_structure_in_formal_flux_phase is not None
        and not isinstance(
            resume_supplied_structure_in_formal_flux_phase, bool
        )
    ):
        raise ValueError(
            "resume_supplied_structure_in_formal_flux_phase must be boolean "
            "or None"
        )
    if initial_convective_gradient_projection_mode not in (
        "interface-transport",
        "unstable-node-gradient",
    ):
        raise ValueError(
            "initial convective-gradient projection mode must be "
            "'interface-transport' or 'unstable-node-gradient'"
        )
    if (
        maximum_convective_preconditioner_iterations is not None
        and maximum_convective_preconditioner_iterations < 1
    ):
        raise ValueError(
            "maximum convective-preconditioner iterations must be positive"
        )
    if (
        preconditioner_stationary_completion_iterations is not None
        and preconditioner_stationary_completion_iterations < 1
    ):
        raise ValueError(
            "preconditioner stationary-completion iterations must be "
            "positive or None"
        )
    # Without ML2 transport there is no gradient to condition. Leaving this
    # switch on also postpones local-energy equations and stable transfer:
    # nearly identical flux rows then leave optically thin temperatures
    # poorly constrained, even as the bolometric flux improves.
    use_convective_gradient_preconditioner = bool(
        use_convective_gradient_preconditioner
        and mixing_length_alpha is not None
    )
    preconditioner_iteration_limit = min(
        max_iterations,
        (
            min(20, max(8, seed.n_depth // 2))
            if maximum_convective_preconditioner_iterations is None
            else maximum_convective_preconditioner_iterations
        ),
    )
    requested_formal_flux_resume = (
        initial_temperature_was_supplied
        if resume_supplied_structure_in_formal_flux_phase is None
        else resume_supplied_structure_in_formal_flux_phase
    )
    if requested_formal_flux_resume and not initial_temperature_was_supplied:
        raise ValueError(
            "formal-flux resume requires a supplied initial temperature"
        )
    resume_in_formal_flux_phase = bool(
        use_convective_gradient_preconditioner and requested_formal_flux_resume
    )
    if resume_in_formal_flux_phase:
        # A complete supplied structure is already the warm start.  Repeating
        # the approximate local-gradient phase after every checkpoint or
        # interpolated-grid restart can consume the budget indefinitely;
        # resume directly with the conservative equation the atmosphere must
        # finally obey.  Fresh gray/continuum seeds still receive the local
        # convective preconditioner below.
        use_convective_gradient_preconditioner = False
    normalization_anchor_index = 0
    if temperature_normalization_anchor_rosseland_depth is not None:
        at_or_below_anchor = np.flatnonzero(
            seed.rosseland_optical_depth
            >= temperature_normalization_anchor_rosseland_depth
        )
        normalization_anchor_index = int(
            at_or_below_anchor[0]
            if at_or_below_anchor.size
            else seed.n_depth - 1
        )

    interface_gradient_operator = np.zeros(
        (seed.n_depth, seed.n_depth), dtype=np.float64
    )
    interface_depth = np.arange(1, seed.n_depth)
    interface_gradient_operator[interface_depth, interface_depth - 1] = (
        -1.0 / log_pressure_step
    )
    interface_gradient_operator[interface_depth, interface_depth] = (
        1.0 / log_pressure_step
    )

    use_physical_flux_residual = not use_convective_gradient_preconditioner
    # Convective starts retain their gradient conditioner. Radiative starts
    # impose local energy immediately, including in optically thin layers.
    energy_completion_active = bool(
        enforce_local_energy_balance
        and not use_convective_gradient_preconditioner
    )
    # A deep ML2-gradient residual is useful only as a nonlinear conditioner.
    # Keep it as a separate switch from ``use_physical_flux_residual`` so the
    # final phase can restore the exact discretized total-flux equation at
    # every interface.  In particular, a diffusion-based desired gradient is
    # not algebraically identical to a finite-grid formal-transfer flux at
    # large optical depth, even though the two approach the same continuum
    # limit.
    use_deep_gradient_conditioner = False
    precision_polish_active = False
    solver_phase = (
        "convective-gradient-preconditioner"
        if use_convective_gradient_preconditioner
        else "formal-radiative-flux"
    )
    solver_iteration_offset = 0
    cached_no_jacobian_log_temperature: FloatArray | None = None
    cached_no_jacobian_solver_phase: str | None = None
    cached_no_jacobian_evaluation: (
        NonlinearEvaluation[dict[str, object]] | None
    ) = None
    cached_base_work: dict[str, object] | None = None

    def interface_atmosphere(current: Atmosphere) -> Atmosphere:
        return Atmosphere(
            effective_temperature=current.effective_temperature,
            logg=current.logg,
            rosseland_optical_depth=_positive_interface_values(
                current.rosseland_optical_depth
            ),
            column_mass=_positive_interface_values(current.column_mass),
            temperature=_positive_interface_values(current.temperature),
            gas_pressure=_positive_interface_values(current.gas_pressure),
            mass_density=_positive_interface_values(current.mass_density),
            neutral_h_density=_positive_interface_values(
                current.neutral_h_density
            ),
            proton_density=_positive_interface_values(current.proton_density),
            electron_density=_positive_interface_values(
                current.electron_density
            ),
            metadata=current.metadata,
        )

    def convection_transport(
        current: Atmosphere,
        temperature_gradient: FloatArray,
        radiative_flux_interface: FloatArray | None = None,
    ) -> dict[str, FloatArray] | None:
        if mixing_length_alpha is None:
            return None
        rosseland = np.asarray(rosseland_opacity(current), dtype=np.float64)
        thermo = thermodynamics(current)
        current_interface = interface_atmosphere(current)
        rosseland_interface = _positive_interface_values(rosseland)
        heat_capacity_interface = _arithmetic_interface_values(
            np.asarray(
                thermo.specific_heat_constant_pressure, dtype=np.float64
            )
        )
        expansion_interface = _arithmetic_interface_values(
            np.asarray(thermo.density_temperature_derivative, dtype=np.float64)
        )
        adiabatic_gradient_interface = _arithmetic_interface_values(
            np.asarray(thermo.adiabatic_temperature_gradient, dtype=np.float64)
        )
        convective_flux_interface = (
            ml2_convective_flux_for_gradient_from_thermodynamics(
                current_interface,
                rosseland_interface,
                temperature_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                coefficient_function=ml2_coefficient_function,
            )
        )
        convective_flux_interface[0] = 0.0
        diffusion_radiative_coefficient = (
            16.0
            * STEFAN_BOLTZMANN
            * current_interface.gravity
            * current_interface.temperature**4
            / (3.0 * rosseland_interface * current_interface.gas_pressure)
        )
        formal_radiative_coefficient = diffusion_radiative_coefficient
        if radiative_flux_interface is not None:
            formal_radiative_coefficient = np.where(
                (temperature_gradient > 1.0e-8)
                & (radiative_flux_interface > 0.0),
                radiative_flux_interface
                / np.maximum(temperature_gradient, 1.0e-8),
                diffusion_radiative_coefficient,
            )
        desired_gradient = (
            ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                current_interface,
                rosseland_interface,
                np.full(current.n_depth, target_flux),
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                radiative_flux_coefficient=formal_radiative_coefficient,
                coefficient_function=ml2_coefficient_function,
            )
        )
        desired_convective_flux = (
            ml2_convective_flux_for_gradient_from_thermodynamics(
                current_interface,
                rosseland_interface,
                desired_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                coefficient_function=ml2_coefficient_function,
            )
        )
        (
            superadiabatic_temperature_increment,
            adiabatic_asymptotic_convection,
        ) = _adiabatic_asymptotic_convection_mask(
            desired_convective_flux,
            desired_gradient,
            adiabatic_gradient_interface,
            log_pressure_step,
            target_flux=target_flux,
            flux_tolerance=flux_tolerance,
            temperature_tolerance=temperature_tolerance,
        )
        if not use_adiabatic_asymptotic_conditioning:
            adiabatic_asymptotic_convection[:] = False
        actual_convective_flux_gradient_derivative = (
            ml2_convective_flux_gradient_derivative_from_thermodynamics(
                current_interface,
                rosseland_interface,
                temperature_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                coefficient_function=ml2_coefficient_function,
            )
        )
        convective_flux_gradient_derivative = (
            ml2_convective_flux_gradient_derivative_from_thermodynamics(
                current_interface,
                rosseland_interface,
                desired_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                coefficient_function=ml2_coefficient_function,
            )
        )
        # Convection cannot carry flux through the surface boundary, so the
        # first Jacobian row remains purely radiative.
        actual_convective_flux_gradient_derivative[0] = 0.0
        convective_flux_gradient_derivative[0] = 0.0
        _, ml2_loss, ml2_coefficient = (
            ml2_coefficient_function(
                current_interface,
                rosseland_interface,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha,
            )
        )
        return {
            "rosseland": rosseland,
            "specific_heat": np.asarray(
                thermo.specific_heat_constant_pressure
            ),
            "expansion": np.asarray(thermo.density_temperature_derivative),
            "node_adiabatic_gradient": np.asarray(
                thermo.adiabatic_temperature_gradient
            ),
            "ml2_radiative_loss": ml2_loss,
            "ml2_flux_coefficient": ml2_coefficient,
            "adiabatic_gradient": adiabatic_gradient_interface,
            "convective_flux": convective_flux_interface,
            "desired_gradient": desired_gradient,
            "desired_convective_flux": desired_convective_flux,
            "superadiabatic_temperature_increment": (
                superadiabatic_temperature_increment
            ),
            "adiabatic_asymptotic_convection": (
                adiabatic_asymptotic_convection
            ),
            "formal_radiative_coefficient": formal_radiative_coefficient,
            "actual_convective_flux_gradient_derivative": (
                actual_convective_flux_gradient_derivative
            ),
            "convective_flux_gradient_derivative": (
                convective_flux_gradient_derivative
            ),
        }

    def convection_coefficient_response(
        current: Atmosphere,
        hotter: Atmosphere,
        transport: Mapping[str, FloatArray],
        logarithmic_step: float,
        probe_rosseland: FloatArray | None = None,
    ) -> tuple[
        tuple[FloatArray, FloatArray, FloatArray],
        tuple[FloatArray, FloatArray, FloatArray],
    ]:
        """Local material tangent, including both nodes of every interface.

        EOS and opacity closures are local at fixed pressure. Reuse the
        all-nodes hotter opacity evaluation already needed for radiation,
        then assemble even/odd nodal perturbations of their material fields.
        Each interface sees exactly one changed endpoint per color. This
        costs one extra thermodynamics call, not one spectral solve per node.
        Only smooth coefficients are differenced; the potentially tiny
        superadiabatic excess is differentiated analytically.
        """
        hot_thermo = thermodynamics(hotter)
        hot_rosseland = (
            np.asarray(rosseland_opacity(hotter), dtype=np.float64)
            if probe_rosseland is None
            else probe_rosseland
        )
        base_fields = (
            current.temperature,
            current.mass_density,
            transport["rosseland"],
            transport["specific_heat"],
            transport["expansion"],
            transport["node_adiabatic_gradient"],
        )
        hot_fields = (
            hotter.temperature,
            hotter.mass_density,
            hot_rosseland,
            hot_thermo.specific_heat_constant_pressure,
            hot_thermo.density_temperature_derivative,
            hot_thermo.adiabatic_temperature_gradient,
        )
        base_coefficients = (
            transport["adiabatic_gradient"],
            transport["ml2_radiative_loss"],
            transport["ml2_flux_coefficient"],
        )
        responses = tuple(
            np.zeros((seed.n_depth, seed.n_depth)) for _ in range(3)
        )
        interface = interface_atmosphere(current)
        for parity in (0, 1):
            mask = np.arange(seed.n_depth) % 2 == parity
            temperature, density, opacity, cp, expansion, adiabatic = (
                np.where(mask, hot, base)
                for base, hot in zip(base_fields, hot_fields)
            )
            perturbed_interface = replace(
                interface,
                temperature=_positive_interface_values(temperature),
                mass_density=_positive_interface_values(density),
            )
            perturbed_coefficients = (
                ml2_coefficient_function(
                    perturbed_interface,
                    _positive_interface_values(opacity),
                    _arithmetic_interface_values(cp),
                    _arithmetic_interface_values(expansion),
                    _arithmetic_interface_values(adiabatic),
                    mixing_length_alpha,
                )
            )
            column = np.where(
                interface_depth % 2 == parity,
                interface_depth,
                interface_depth - 1,
            )
            for response, perturbed, base in zip(
                responses, perturbed_coefficients, base_coefficients
            ):
                response[interface_depth, column] = (
                    perturbed[interface_depth] - base[interface_depth]
                ) / logarithmic_step
        return base_coefficients, responses

    def evaluate_log_temperature(
        log_temperature: FloatArray, need_jacobian: bool
    ) -> NonlinearEvaluation[dict[str, object]]:
        nonlocal cached_no_jacobian_log_temperature
        nonlocal cached_no_jacobian_solver_phase
        nonlocal cached_no_jacobian_evaluation
        nonlocal cached_base_work
        # Warm-start selection and the nonlinear driver can request the same
        # residual several times in succession.  A structure residual is
        # deterministic at fixed ln(T) and solver phase, while its opacity and
        # transfer calculation is expensive.  Retain the most recent
        # residual-only evaluation.  When the nonlinear driver immediately
        # requests a Jacobian at that identical state, reuse the already
        # evaluated EOS, opacity, scattering source, transfer solution, and
        # convection closure; only the hotter tangent state and exact response
        # remain to be built.
        energy_equations_active = bool(
            energy_completion_active
            and use_physical_flux_residual
            and not use_deep_gradient_conditioner
        )
        stable_arithmetic_active = bool(
            precision_polish_active or energy_equations_active
        )
        same_as_cached_residual = (
            cached_no_jacobian_evaluation is not None
            and cached_no_jacobian_solver_phase == solver_phase
            and cached_no_jacobian_log_temperature is not None
            and np.array_equal(
                np.asarray(log_temperature),
                cached_no_jacobian_log_temperature,
            )
        )
        if not need_jacobian and same_as_cached_residual:
            return cached_no_jacobian_evaluation
        reuse_base = bool(
            need_jacobian
            and same_as_cached_residual
            and cached_base_work is not None
        )
        if reuse_base:
            assert cached_base_work is not None
            current_temperature = cached_base_work["temperature"]
            current = cached_base_work["atmosphere"]
            absorption = cached_base_work["absorption"]
            scattering = cached_base_work["scattering"]
            extinction = cached_base_work["extinction"]
            optical_depth = cached_base_work["optical_depth"]
            planck = cached_base_work["planck"]
            source = cached_base_work["source"]
            field = cached_base_work["field"]
            source_relative_residual = cached_base_work[
                "source_relative_residual"
            ]
            source_worst_wavelength_index = cached_base_work[
                "source_worst_wavelength_index"
            ]
            source_worst_depth_index = cached_base_work[
                "source_worst_depth_index"
            ]
            source_negative_clipped_count = cached_base_work[
                "source_negative_clipped_count"
            ]
            source_maximum_negative_relative = cached_base_work[
                "source_maximum_negative_relative"
            ]
            radiative_flux_interface = cached_base_work[
                "radiative_flux_interface"
            ]
            temperature_gradient = cached_base_work["temperature_gradient"]
            convective_flux = cached_base_work["convective_flux"]
            convective_flux_interface = cached_base_work[
                "convective_flux_interface"
            ]
            transport = cached_base_work["transport"]
        else:
            current_temperature = np.exp(log_temperature)
            current = with_temperature(current_temperature)
            absorption = np.asarray(true_absorption(current), dtype=np.float64)
            scattering = np.asarray(
                scattering_opacity(current), dtype=np.float64
            )
            expected = (wavelength.size, current.n_depth)
            if absorption.shape != expected or scattering.shape != expected:
                raise ValueError(
                    "opacity callbacks must return (wavelength, depth)"
                )
            extinction = absorption + scattering
            extinction = np.maximum(extinction, np.finfo(np.float64).tiny)
            optical_depth = optical_depth_from_mass_opacity(
                current.column_mass, extinction
            )
            if np.any(~np.isfinite(optical_depth)) or np.any(
                np.diff(optical_depth, axis=1) <= 0.0
            ):
                # Positive local opacities do not guarantee representable
                # cumulative increments on a severely compressed mass grid.
                # This particular atmosphere is an inadmissible trial: let
                # the line search shorten the step, without catching unrelated
                # transfer/opacity contract errors or inventing optical depth.
                raise RecoverableEvaluationError(
                    "trial atmosphere has non-finite or unresolved optical-depth increments"
                )
            planck = planck_lambda_angstrom(
                wavelength[:, np.newaxis], current_temperature[np.newaxis, :]
            )
            field_solver = (
                cancellation_safe_field
                if stable_arithmetic_active
                else coherent_scattering_feautrier_field
            )
            field_options = {}
            if mass_transfer:
                field_solver = mass_field
                field_options["column_mass"] = current.column_mass
            source, field = field_solver(
                optical_depth,
                planck,
                absorption,
                scattering,
                n_angle=n_angle,
                **field_options,
            )
            (
                source,
                source_negative_clipped_count,
                (source_maximum_negative_relative),
            ) = _clip_roundoff_negative_scattering_source(source)
            # Check closure with the independent fixed-source formal solver.
            # Using the coupled J that constructed S would test an algebraic
            # identity, not whether the returned source solves transfer.
            closure_field = (
                mass_field(
                    optical_depth,
                    source,
                    extinction,
                    np.zeros_like(extinction),
                    column_mass=current.column_mass,
                    n_angle=n_angle,
                )[1]
                if mass_transfer and np.any(scattering > 0.0)
                else (
                    (
                        cancellation_safe_scalar_field
                        if stable_arithmetic_active
                        else feautrier_radiation_field
                    )(optical_depth, source, n_angle=n_angle)
                    if np.any(scattering > 0.0)
                    else field
                )
            )
            source_equation = (
                absorption * planck + scattering * closure_field.mean_intensity
            ) / extinction
            source_residual_floor = np.maximum(
                np.max(planck, axis=1, keepdims=True) * 1.0e-12,
                np.finfo(np.float64).tiny,
            )
            source_relative_residual = np.abs(
                source_equation - source
            ) / np.maximum(
                np.maximum(np.abs(source_equation), np.abs(source)),
                source_residual_floor,
            )
            source_worst_flat_index = int(np.argmax(source_relative_residual))
            source_worst_wavelength_index, source_worst_depth_index = (
                np.unravel_index(source_worst_flat_index, source.shape)
            )
            if (
                field.interface_flux is None
            ):  # pragma: no cover - API invariant
                raise RuntimeError(
                    "Feautrier solver did not return interface fluxes"
                )
            radiative_flux_interface = trapezoid(
                field.interface_flux, wavelength, axis=0
            )

            temperature_gradient = np.empty_like(log_temperature)
            temperature_gradient[0] = 0.0
            temperature_gradient[1:] = (
                np.diff(log_temperature) / log_pressure_step
            )
            convective_flux = np.zeros_like(current_temperature)
            convective_flux_interface = np.zeros_like(current_temperature)
            transport = convection_transport(
                current,
                temperature_gradient,
                radiative_flux_interface,
            )
            if transport is not None:
                convective_flux_interface = transport["convective_flux"]
                convective_flux = _upper_interface_values_on_nodes(
                    convective_flux_interface
                )

        total_flux_interface = (
            radiative_flux_interface + convective_flux_interface
        )
        flux_residual = total_flux_interface / target_flux - 1.0
        residual = flux_residual.copy()
        transport_gradient_scale = None
        gradient_preconditioned = np.zeros(seed.n_depth, dtype=bool)
        physical_flux_weight = np.ones(seed.n_depth, dtype=np.float64)
        if transport is not None:
            desired_gradient = transport["desired_gradient"]
            adiabatic_gradient = transport["adiabatic_gradient"]
            transport_gradient_scale = np.maximum(
                desired_gradient, adiabatic_gradient
            )
            asymptotic = np.asarray(
                transport["adiabatic_asymptotic_convection"], dtype=bool
            )
            if use_physical_flux_residual and np.any(asymptotic):
                # Normalize the equivalent gradient defect by the change in
                # gradient that carries one stellar flux.  In the asymptotic
                # ML2 limit, scaling by nabla itself admits a harmless-looking
                # residual whose raw flux error can be thousands of times the
                # target because dF_conv/dnabla is enormous.
                stiff_flux_derivative = np.asarray(
                    transport["formal_radiative_coefficient"],
                    dtype=np.float64,
                ) + np.asarray(
                    transport["convective_flux_gradient_derivative"],
                    dtype=np.float64,
                )
                transport_gradient_scale[asymptotic] = (
                    target_flux
                    / np.maximum(
                        stiff_flux_derivative[asymptotic],
                        np.finfo(np.float64).tiny,
                    )
                )
            gradient_preconditioned = (
                transport["desired_convective_flux"] > 0.0
            )
            if not use_physical_flux_residual:
                physical_flux_weight[gradient_preconditioned] = 0.0
            else:
                if (
                    not use_deep_gradient_conditioner
                    or maximum_formal_flux_rosseland_depth is None
                ):
                    gradient_preconditioned[:] = False
                elif formal_flux_taper_start_rosseland_depth is not None:
                    candidate = gradient_preconditioned.copy()
                    physical_flux_weight[candidate] = np.clip(
                        np.log(
                            maximum_formal_flux_rosseland_depth
                            / np.maximum(
                                seed.rosseland_optical_depth[candidate],
                                np.finfo(np.float64).tiny,
                            )
                        )
                        / np.log(
                            maximum_formal_flux_rosseland_depth
                            / formal_flux_taper_start_rosseland_depth
                        ),
                        0.0,
                        1.0,
                    )
                    gradient_preconditioned &= physical_flux_weight < 1.0
                else:
                    gradient_preconditioned &= (
                        seed.rosseland_optical_depth
                        > maximum_formal_flux_rosseland_depth
                    )
                    physical_flux_weight[gradient_preconditioned] = 0.0
            gradient_preconditioned |= asymptotic
            physical_flux_weight[asymptotic] = 0.0
            gradient_preconditioned[0] = False
            gradient_residual = (
                temperature_gradient[gradient_preconditioned]
                - desired_gradient[gradient_preconditioned]
            ) / transport_gradient_scale[gradient_preconditioned]
            residual[gradient_preconditioned] = (
                physical_flux_weight[gradient_preconditioned]
                * flux_residual[gradient_preconditioned]
                + (1.0 - physical_flux_weight[gradient_preconditioned])
                * gradient_residual
            )

        local_energy_active = bool(
            compute_local_energy_response
            or (
                energy_completion_active
                and use_physical_flux_residual
                and not use_deep_gradient_conditioner
            )
        )
        jacobian = None
        if need_jacobian:
            probe_rosseland: dict[int, FloatArray] = {}
            retain_probe_rosseland = bool(
                reuse_material_probe_rosseland
                and local_energy_active
                and transport is not None
            )

            def material_probe(offset):
                point = with_temperature(current_temperature * np.exp(offset))
                probe = (
                    point,
                    np.asarray(true_absorption(point), dtype=np.float64),
                    np.asarray(scattering_opacity(point), dtype=np.float64),
                )
                if retain_probe_rosseland:
                    # The opposite probe otherwise displaces this state from
                    # the composition's one-entry opacity cache. Retain its
                    # small mean-opacity vector, not another spectral grid.
                    probe_rosseland[id(point)] = np.asarray(
                        rosseland_opacity(point), dtype=np.float64
                    ).copy()
                return probe

            primary_probe, opposite_probe, logarithmic_step = (
                temperature_response_probes(
                    material_probe,
                    2.0e-4,
                    centered=local_energy_active,
                )
            )
            if column_mass_radiation_backend is not None:
                column_mass_radiation_backend.record_temperature_response_probes(
                    (primary_probe, opposite_probe, logarithmic_step)
                )
            hotter, hotter_absorption, hotter_scattering = primary_probe
            hotter_temperature = current_temperature * np.exp(logarithmic_step)
            hotter_planck = planck_lambda_angstrom(
                wavelength[:, np.newaxis], hotter_temperature[np.newaxis, :]
            )
            planck_derivative = (hotter_planck - planck) / logarithmic_step
            absorption_derivative = (
                hotter_absorption - absorption
            ) / logarithmic_step
            scattering_derivative = (
                hotter_scattering - scattering
            ) / logarithmic_step
            centered_material_response = opposite_probe is not None
            if centered_material_response:
                # Local heating rows can cancel leading contributions that
                # are harmless in a flux-only tangent. Use centered material
                # differences for the experimental direct-energy response;
                # retain the established default tangent unchanged.
                colder_temperature = current_temperature * np.exp(
                    -logarithmic_step
                )
                colder, colder_absorption, colder_scattering = opposite_probe
                colder_planck = planck_lambda_angstrom(
                    wavelength[:, None], colder_temperature[None, :]
                )
                planck_derivative = (hotter_planck - colder_planck) / (
                    2 * logarithmic_step
                )
                absorption_derivative = (
                    hotter_absorption - colder_absorption
                ) / (2 * logarithmic_step)
                scattering_derivative = (
                    hotter_scattering - colder_scattering
                ) / (2 * logarithmic_step)
            extinction_derivative = (
                absorption_derivative + scattering_derivative
            )
            source_derivative = (
                absorption_derivative * planck
                + absorption * planck_derivative
                + scattering_derivative * field.mean_intensity
                - extinction_derivative * source
            ) / extinction
            if local_energy_active:
                energy_response_solver = (
                    mass_energy_response
                    if mass_transfer
                    else integrated_radiative_cell_energy_state_response
                )
                (
                    radiative_flux_jacobian,
                    radiative_cell_energy_jacobian,
                    thermal_emission_jacobian,
                ) = energy_response_solver(
                    wavelength,
                    optical_depth,
                    current.column_mass,
                    planck,
                    field.mean_intensity,
                    source,
                    absorption,
                    scattering,
                    planck_derivative,
                    absorption_derivative,
                    scattering_derivative,
                    n_angle=n_angle,
                    response_solver=(
                        cancellation_safe_response
                        if stable_arithmetic_active and not mass_transfer
                        else None
                    ),
                )
            elif (
                mass_transfer
                or stable_arithmetic_active
                or np.any(scattering > 0.0)
            ):
                response_solver = (
                    cancellation_safe_response
                    if stable_arithmetic_active
                    else integrated_coherent_scattering_feautrier_state_response
                )
                response_options = {}
                if mass_transfer:
                    response_solver = mass_response
                    response_options["extinction"] = extinction
                radiative_flux_jacobian, _, _ = response_solver(
                    optical_depth,
                    wavelength,
                    source,
                    source_derivative,
                    planck_derivative,
                    scattering / extinction,
                    current.column_mass,
                    extinction_derivative,
                    n_angle=n_angle,
                    return_auxiliary_response=False,
                    **response_options,
                )
            else:
                radiative_flux_jacobian = (
                    integrated_feautrier_interface_state_response(
                        optical_depth,
                        wavelength,
                        source,
                        source_derivative,
                        current.column_mass,
                        extinction_derivative,
                        n_angle=n_angle,
                    )
                )
            jacobian = radiative_flux_jacobian / target_flux
            if transport is not None:
                ml2_coefficients, ml2_responses = (
                    convection_coefficient_response(
                        current,
                        hotter,
                        transport,
                        logarithmic_step,
                        probe_rosseland.get(id(hotter)),
                    )
                )
                if centered_material_response:
                    _, cold_ml2_responses = convection_coefficient_response(
                        current, colder, transport, -logarithmic_step,
                        probe_rosseland.get(id(colder)),
                    )
                    ml2_responses = tuple(
                        0.5 * (hot + cold)
                        for hot, cold in zip(ml2_responses, cold_ml2_responses)
                    )
            if transport is not None and use_physical_flux_residual:
                # Differentiate both the gradient and the material state.
                # Freezing cp, Q, nabla_ad, density and opacity can even give
                # the wrong sign for Newton's convective response.
                jacobian += (
                    transport["actual_convective_flux_gradient_derivative"][
                        :, np.newaxis
                    ]
                    * interface_gradient_operator
                    / target_flux
                )
                jacobian += (
                    _ml2_flux_coefficient_response(
                        temperature_gradient, ml2_coefficients, ml2_responses
                    )
                    / target_flux
                )
            if (
                transport is not None
                and transport_gradient_scale is not None
                and np.any(gradient_preconditioned)
            ):
                formal_coefficient = transport["formal_radiative_coefficient"]
                convective_derivative = transport[
                    "convective_flux_gradient_derivative"
                ]
                safe_gradient = np.maximum(temperature_gradient, 1.0e-8)
                formal_coefficient_derivative = np.zeros_like(
                    radiative_flux_jacobian
                )
                formal_branch = (temperature_gradient > 1.0e-8) & (
                    radiative_flux_interface > 0.0
                )
                formal_coefficient_derivative[formal_branch] = (
                    radiative_flux_jacobian[formal_branch]
                    * safe_gradient[formal_branch, np.newaxis]
                    - radiative_flux_interface[formal_branch, np.newaxis]
                    * interface_gradient_operator[formal_branch]
                ) / safe_gradient[formal_branch, np.newaxis] ** 2
                desired_gradient_derivative = (
                    -transport["desired_gradient"][:, np.newaxis]
                    * formal_coefficient_derivative
                    - _ml2_flux_coefficient_response(
                        transport["desired_gradient"],
                        ml2_coefficients,
                        ml2_responses,
                    )
                ) / (formal_coefficient + convective_derivative)[:, np.newaxis]
                gradient_jacobian = (
                    interface_gradient_operator[gradient_preconditioned]
                    - desired_gradient_derivative[gradient_preconditioned]
                ) / transport_gradient_scale[
                    gradient_preconditioned, np.newaxis
                ]
                jacobian[gradient_preconditioned] = (
                    physical_flux_weight[gradient_preconditioned, np.newaxis]
                    * jacobian[gradient_preconditioned]
                    + (
                        1.0
                        - physical_flux_weight[
                            gradient_preconditioned, np.newaxis
                        ]
                    )
                    * gradient_jacobian
                )

        physical_flux_monitored = np.ones(seed.n_depth, dtype=bool)
        if (
            use_deep_gradient_conditioner
            and maximum_formal_flux_rosseland_depth is not None
        ):
            if formal_flux_taper_start_rosseland_depth is None:
                physical_flux_monitored = (
                    seed.rosseland_optical_depth
                    <= maximum_formal_flux_rosseland_depth
                )
            else:
                physical_flux_monitored = physical_flux_weight >= 0.5

        payload: dict[str, object] = {
            "atmosphere": current,
            "lower_boundary_absorption_escape_bound": (
                (mass_boundary if mass_transfer else _thermal_boundary_absorption_escape_bound)(
                    wavelength,
                    current.column_mass,
                    absorption,
                    planck[:, -1],
                    target_flux,
                )
            ),
            "radiative_flux_interface": radiative_flux_interface,
            "convective_flux": convective_flux,
            "convective_flux_interface": convective_flux_interface,
            "total_flux_interface": total_flux_interface,
            "convection_transport": transport,
            "temperature_gradient": temperature_gradient,
            "physical_flux_monitored": physical_flux_monitored,
            "physical_flux_residual_weight": physical_flux_weight,
            "scattering_source_iterations": 1,
            "scattering_source_maximum_relative_residual": float(
                np.max(source_relative_residual)
            ),
            "scattering_source_worst_wavelength_index": int(
                source_worst_wavelength_index
            ),
            "scattering_source_worst_depth_index": int(
                source_worst_depth_index
            ),
            "source_negative_clipped_count": int(
                source_negative_clipped_count
            ),
            "source_maximum_negative_relative": float(
                source_maximum_negative_relative
            ),
        }
        if mass_transfer:
            radiative_cell_defect, thermal_cell_emission = mass_energy(
                wavelength,
                current.column_mass,
                planck,
                field.mean_intensity,
                absorption,
            )
        else:
            radiative_cell_defect, thermal_cell_emission = (
                discrete_radiative_cell_energy_balance(
                    wavelength,
                    optical_depth,
                    planck,
                    field.mean_intensity,
                    absorption / extinction,
                )
            )
        cell_defect = radiative_cell_defect + np.diff(
            convective_flux_interface
        )
        cell_scale = np.maximum(
            thermal_cell_emission
            + np.abs(convective_flux_interface[:-1])
            + np.abs(convective_flux_interface[1:]),
            np.finfo(float).tiny,
        )
        payload["radiative_cell_energy_defect"] = radiative_cell_defect
        payload["thermal_cell_emission"] = thermal_cell_emission
        payload["cell_energy_scale"] = cell_scale
        payload["cell_energy_balance_relative_residual"] = (
            cell_defect / cell_scale
        )
        payload["cell_energy_balance_defect_in_stellar_flux"] = (
            cell_defect / target_flux
        )
        payload["energy_balance_is_physical_flux"] = bool(
            use_physical_flux_residual and not np.any(gradient_preconditioned)
        )
        if need_jacobian and transport is not None:
            payload["radiative_flux_log_temperature_jacobian"] = (
                radiative_flux_jacobian
            )
            payload["ml2_coefficient_log_temperature_responses"] = (
                ml2_responses
            )
        if need_jacobian and local_energy_active:
            payload["radiative_flux_log_temperature_jacobian"] = (
                radiative_flux_jacobian
            )
            payload["convective_flux_log_temperature_jacobian"] = (
                np.zeros_like(radiative_flux_jacobian)
            )
            payload["material_temperature_response_domain_limited"] = (
                not centered_material_response
            )
            payload["radiative_cell_energy_log_temperature_jacobian"] = (
                radiative_cell_energy_jacobian
            )
            cell_jacobian = radiative_cell_energy_jacobian.copy()
            cell_scale_jacobian = thermal_emission_jacobian.copy()
            if transport is not None:
                actual_convection_jacobian = transport[
                    "actual_convective_flux_gradient_derivative"
                ][
                    :, None
                ] * interface_gradient_operator + _ml2_flux_coefficient_response(
                    temperature_gradient, ml2_coefficients, ml2_responses
                )
                payload["convective_flux_log_temperature_jacobian"] = (
                    actual_convection_jacobian
                )
                cell_jacobian += np.diff(actual_convection_jacobian, axis=0)
                cell_scale_jacobian += (
                    actual_convection_jacobian[:-1]
                    + actual_convection_jacobian[1:]
                )
            payload["cell_energy_log_temperature_jacobian"] = cell_jacobian
            payload["thermal_cell_emission_log_temperature_jacobian"] = (
                thermal_emission_jacobian
            )
            payload["cell_energy_scale_log_temperature_jacobian"] = (
                cell_scale_jacobian
            )
        if (
            energy_completion_active
            and use_physical_flux_residual
            and not use_deep_gradient_conditioner
        ):
            # An invertible combination of constant-flux and directly
            # evaluated cell-heating equations. No Teff/convection mask and
            # no subtraction of nearly identical flux-Jacobian rows.
            residual = flux_residual.copy()
            residual[:-1] -= cell_defect / cell_scale
            if need_jacobian:
                jacobian = radiative_flux_jacobian.copy() / target_flux
                if transport is not None:
                    jacobian += actual_convection_jacobian / target_flux
                jacobian[:-1] -= cell_jacobian / cell_scale[:, None]
                jacobian[:-1] += (cell_defect / cell_scale**2)[
                    :, None
                ] * cell_scale_jacobian
            payload["energy_balance_is_physical_flux"] = True
            payload["local_energy_equations_enforced"] = True
            payload["physical_flux_residual_weight"] = np.ones(seed.n_depth)
        evaluation = NonlinearEvaluation(residual, jacobian, payload)
        if not need_jacobian:
            cached_no_jacobian_log_temperature = np.asarray(
                log_temperature, dtype=np.float64
            ).copy()
            cached_no_jacobian_solver_phase = solver_phase
            cached_no_jacobian_evaluation = evaluation
            cached_base_work = {
                "temperature": current_temperature,
                "atmosphere": current,
                "absorption": absorption,
                "scattering": scattering,
                "extinction": extinction,
                "optical_depth": optical_depth,
                "planck": planck,
                "source": source,
                "field": field,
                "source_relative_residual": source_relative_residual,
                "source_worst_wavelength_index": (
                    source_worst_wavelength_index
                ),
                "source_worst_depth_index": source_worst_depth_index,
                "source_negative_clipped_count": (
                    source_negative_clipped_count
                ),
                "source_maximum_negative_relative": (
                    source_maximum_negative_relative
                ),
                "radiative_flux_interface": radiative_flux_interface,
                "temperature_gradient": temperature_gradient,
                "convective_flux": convective_flux,
                "convective_flux_interface": convective_flux_interface,
                "transport": transport,
            }
        return evaluation

    # Every structure evaluation (opacity, transfer and convection at one
    # temperature) is logged with its phase and wall time, so long phases
    # that accept no iteration for many evaluations remain observable.
    untimed_evaluate_log_temperature = evaluate_log_temperature
    structure_evaluation_count = 0

    def evaluate_log_temperature(
        log_temperature: FloatArray, need_jacobian: bool
    ) -> NonlinearEvaluation[dict[str, object]]:
        nonlocal structure_evaluation_count
        started = time.perf_counter()
        evaluation = untimed_evaluate_log_temperature(log_temperature, need_jacobian)
        structure_evaluation_count += 1
        _LOGGER.info(
            "structure evaluation %d: phase %s, jacobian %s, %.1f s",
            structure_evaluation_count,
            solver_phase,
            need_jacobian,
            time.perf_counter() - started,
        )
        return evaluation

    log_temperature_from_state = np.zeros(
        (seed.n_depth, seed.n_depth), dtype=np.float64
    )
    log_temperature_from_state[:, 0] = 1.0
    for interface in range(1, seed.n_depth):
        pressure_step = log_pressure_step[interface - 1]
        if interface <= normalization_anchor_index:
            log_temperature_from_state[:interface, interface] = -pressure_step
        else:
            log_temperature_from_state[interface:, interface] = pressure_step

    nodal_coordinates = bool(
        physical_temperature_coordinates
        and energy_completion_active
        and use_physical_flux_residual
        and not use_deep_gradient_conditioner
    )
    identity_temperature_map = np.eye(seed.n_depth)

    def temperature_map():
        return (
            identity_temperature_map
            if nodal_coordinates
            else log_temperature_from_state
        )

    def state_from_log_temperature(log_temperature: FloatArray) -> FloatArray:
        if nodal_coordinates:
            return np.asarray(log_temperature).copy()
        state = np.empty_like(log_temperature)
        state[0] = log_temperature[normalization_anchor_index]
        state[1:] = np.diff(log_temperature) / log_pressure_step
        return state

    def evaluate_state(
        state: FloatArray, need_jacobian: bool
    ) -> NonlinearEvaluation[dict[str, object]]:
        evaluation = evaluate_log_temperature(
            temperature_map() @ state, need_jacobian
        )
        jacobian = evaluation.jacobian
        payload = evaluation.payload
        if jacobian is not None:
            jacobian = jacobian @ temperature_map()
            if "radiative_flux_log_temperature_jacobian" in payload:
                payload = dict(payload)
                # Expose existing tangents to isolated auxiliary-variable
                # experiments without extra matrix products on the default path.
                payload["log_temperature_from_state"] = temperature_map()
                payload["interface_gradient_operator"] = (
                    interface_gradient_operator
                )
        return NonlinearEvaluation(evaluation.residual, jacobian, payload)

    def report_iteration(record, state, evaluation) -> None:
        if iteration_callback is None:
            return
        payload = evaluation.payload
        current = payload["atmosphere"]
        total = np.asarray(payload["total_flux_interface"], dtype=np.float64)
        radiative = np.asarray(
            payload["radiative_flux_interface"], dtype=np.float64
        )
        convective = np.asarray(
            payload["convective_flux_interface"], dtype=np.float64
        )
        flux_residual = total / target_flux - 1.0
        monitored = np.asarray(payload["physical_flux_monitored"], dtype=bool)
        maximum_depth = int(np.argmax(np.abs(flux_residual)))
        transport = payload["convection_transport"]
        desired_gradient = (
            np.asarray(transport["desired_gradient"], dtype=np.float64)
            if transport is not None
            else np.full(seed.n_depth, np.nan)
        )
        actual_gradient = np.asarray(
            payload["temperature_gradient"], dtype=np.float64
        )
        asymptotic = (
            np.asarray(
                transport["adiabatic_asymptotic_convection"], dtype=bool
            )
            if transport is not None
            else np.zeros(seed.n_depth, dtype=bool)
        )
        iteration_callback(
            solver_iteration_offset + record.iteration,
            current,
            {
                "solver_residual_rms": record.residual_rms,
                "solver_residual_maximum": record.residual_maximum,
                "flux_ratio": float(total[0] / target_flux),
                "maximum_log_temperature_correction": record.maximum_step,
                "maximum_flux_residual_depth_index": maximum_depth,
                # Retain the historical key; it has always located the flux
                # residual, not the largest temperature correction.
                "maximum_correction_depth_index": maximum_depth,
                "maximum_total_flux_residual": float(
                    np.max(np.abs(flux_residual))
                ),
                "maximum_formal_region_total_flux_residual": float(
                    np.max(np.abs(flux_residual[monitored]))
                ),
                "maximum_all_depth_total_flux_residual": float(
                    np.max(np.abs(flux_residual))
                ),
                "maximum_relative_cell_energy_balance_residual": float(
                    np.max(
                        np.abs(
                            payload["cell_energy_balance_relative_residual"]
                        )
                    )
                ),
                "bottom_radiative_flux_ratio": float(
                    radiative[-1] / target_flux
                ),
                "bottom_convective_flux_ratio": float(
                    convective[-1] / target_flux
                ),
                "maximum_flux_depth_temperature_gradient": float(
                    actual_gradient[maximum_depth]
                ),
                "maximum_flux_depth_desired_gradient": float(
                    desired_gradient[maximum_depth]
                ),
                "adiabatic_asymptotic_interfaces": int(np.sum(asymptotic)),
                "trust_radius": record.trust_radius,
                "line_search_factor": record.line_search_factor,
                "jacobian_recomputed": record.jacobian_recomputed,
                "scattering_source_iterations": int(
                    payload["scattering_source_iterations"]
                ),
                "scattering_source_maximum_relative_residual": float(
                    payload["scattering_source_maximum_relative_residual"]
                ),
                "scattering_source_negative_clipped_count": int(
                    payload["source_negative_clipped_count"]
                ),
                "scattering_source_maximum_negative_relative": float(
                    payload["source_maximum_negative_relative"]
                ),
                "solver_phase": solver_phase,
                "converged": bool(
                    solver_phase != "thermal-conditioning"
                    and record.residual_maximum < flux_tolerance
                    and record.maximum_step < temperature_tolerance
                    and np.max(np.abs(flux_residual)) < flux_tolerance
                ),
            },
        )

    def physical_flux_converged(state, evaluation, maximum_step) -> bool:
        total = np.asarray(
            evaluation.payload["total_flux_interface"], dtype=np.float64
        )
        return bool(np.max(np.abs(total / target_flux - 1.0)) < flux_tolerance)

    def phase_converged(state, evaluation, maximum_step) -> bool:
        if not use_physical_flux_residual or use_deep_gradient_conditioner:
            return True
        return bool(
            physical_flux_converged(state, evaluation, maximum_step)
            and (
                not energy_completion_active
                or np.max(
                    np.abs(
                        evaluation.payload[
                            "cell_energy_balance_relative_residual"
                        ]
                    )
                )
                < flux_tolerance
            )
        )

    def local_energy_handoff(state, evaluation) -> bool:
        # Once flux is already balanced, a failed flux-only direction can be
        # caused by unresolved local heating. Use its direct equations rather
        # than repeat this representation through trust collapse/restarts.
        # A successful original direction, or an already-qualified atmosphere,
        # follows the original path unchanged. No Teff or new tolerance here.
        return bool(
            enforce_local_energy_balance
            and not energy_completion_active
            and use_physical_flux_residual
            and not use_deep_gradient_conditioner
            and physical_flux_converged(state, evaluation, 0.0)
            and np.max(
                np.abs(
                    evaluation.payload["cell_energy_balance_relative_residual"]
                )
            )
            >= flux_tolerance
        )

    def surface_flux_error(evaluation) -> float:
        total = np.asarray(
            evaluation.payload["total_flux_interface"], dtype=np.float64
        )
        return float(abs(total[0] / target_flux - 1.0))

    def seed_merit(evaluation) -> float:
        residual = evaluation.residual
        return float(
            np.sqrt(np.mean(residual**2)) + 0.25 * np.max(np.abs(residual))
        )

    initial_log_temperature = np.log(seed.temperature)
    initial_bolometric_temperature_scale = 1.0
    initial_bolometric_rescaling_evaluations = 0
    initial_convective_gradient_projection_damping = 0.0
    initial_convective_gradient_projection_rejected_infeasible = 0
    initial_convective_gradient_projection_attempted = bool(
        mixing_length_alpha is not None
        and project_initial_convective_gradient
        and not resume_in_formal_flux_phase
    )
    if initial_convective_gradient_projection_attempted:
        initial = with_temperature(seed.temperature)
        if (
            initial_convective_gradient_projection_mode
            == "unstable-node-gradient"
        ):
            # The original DA cold-start construction evaluated the ML2
            # proposal on depth nodes and changed only gradients that were
            # already Schwarzschild-unstable.  Keep that validated seed
            # policy as a composition option while the nonlinear equations,
            # transfer response, continuation, and convergence checks remain
            # owned by this shared driver.
            initial_gradient = np.gradient(
                initial_log_temperature, log_pressure, edge_order=2
            )
            initial_thermodynamics = thermodynamics(initial)
            initial_rosseland = np.asarray(
                rosseland_opacity(initial), dtype=np.float64
            )
            desired_gradient = (
                ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                    initial,
                    initial_rosseland,
                    np.full(seed.n_depth, target_flux),
                    np.asarray(
                        initial_thermodynamics.specific_heat_constant_pressure,
                        dtype=np.float64,
                    ),
                    np.asarray(
                        initial_thermodynamics.density_temperature_derivative,
                        dtype=np.float64,
                    ),
                    np.asarray(
                        initial_thermodynamics.adiabatic_temperature_gradient,
                        dtype=np.float64,
                    ),
                    mixing_length_alpha=mixing_length_alpha,
                    coefficient_function=ml2_coefficient_function,
                )
            )
            adiabatic_gradient = np.asarray(
                initial_thermodynamics.adiabatic_temperature_gradient,
                dtype=np.float64,
            )
            unstable = initial_gradient > adiabatic_gradient
            projected_gradient = np.where(
                unstable,
                np.minimum(initial_gradient, desired_gradient),
                initial_gradient,
            )
        else:
            initial_gradient = np.empty(seed.n_depth, dtype=np.float64)
            initial_gradient[0] = 0.0
            initial_gradient[1:] = (
                np.diff(initial_log_temperature) / log_pressure_step
            )
            transport = convection_transport(initial, initial_gradient)
            assert transport is not None
            desired_gradient = transport["desired_gradient"]
            adiabatic_gradient = transport["adiabatic_gradient"]
            unstable = initial_gradient > adiabatic_gradient
            convection_required = desired_gradient > adiabatic_gradient
            projected_gradient = np.where(
                unstable,
                np.minimum(initial_gradient, desired_gradient),
                np.where(
                    convection_required, desired_gradient, initial_gradient
                ),
            )
        ordinary_evaluation = evaluate_log_temperature(
            initial_log_temperature, False
        )
        ordinary_merit = seed_merit(ordinary_evaluation)
        # The local ML2 inversion is a proposal based on the seed EOS and
        # opacity.  Its full application can be too large, particularly when
        # the seed spans many pressure scale heights.  Backtrack this
        # provisional correction exactly as an ordinary nonlinear step, and
        # regard a nonphysical trial opacity grid as a rejected proposal
        # rather than allowing an optional warm start to abort the solve.
        for projection_damping in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
        ):
            candidate_gradient = initial_gradient + projection_damping * (
                projected_gradient - initial_gradient
            )
            candidate_log_temperature = np.empty_like(initial_log_temperature)
            candidate_log_temperature[0] = initial_log_temperature[0]
            for depth in range(1, seed.n_depth):
                if (
                    initial_convective_gradient_projection_mode
                    == "unstable-node-gradient"
                ):
                    integrated_gradient = 0.5 * (
                        candidate_gradient[depth - 1]
                        + candidate_gradient[depth]
                    )
                else:
                    integrated_gradient = candidate_gradient[depth]
                candidate_log_temperature[depth] = (
                    candidate_log_temperature[depth - 1]
                    + integrated_gradient * log_pressure_step[depth - 1]
                )
            try:
                candidate_evaluation = evaluate_log_temperature(
                    candidate_log_temperature, False
                )
            except ValueError:
                initial_convective_gradient_projection_rejected_infeasible += 1
                continue
            projection_improves_surface = (
                not safeguard_surface_flux
                or surface_flux_error(candidate_evaluation)
                <= surface_flux_error(ordinary_evaluation) + 1.0e-12
            )
            if (
                seed_merit(candidate_evaluation) < ordinary_merit
                and projection_improves_surface
            ):
                initial_log_temperature = candidate_log_temperature
                initial_convective_gradient_projection_damping = (
                    projection_damping
                )
                break

    # A checkpoint interpolated in Teff/log(g), or relaxed at a neighboring
    # composition, can have a good dimensionless gradient but the wrong
    # homologous temperature normalization.  In that case an unrestricted
    # Newton solve wastes many opacity/transfer evaluations rediscovering the
    # gray F ~ T^4 scaling through small local steps.  Treat the gray scaling
    # only as a proposed warm start: evaluate the complete non-gray transfer
    # problem at the full correction and two damped corrections, and retain a
    # candidate only when it improves both the physical surface-flux error and
    # the merit of the complete current residual.  The latter guard is
    # essential for a convective restart: a homologous temperature change can
    # repair the surface flux while destroying an already accurate deep ML2
    # closure.  Thus the gray estimate never replaces or weakens the equations
    # used for convergence.
    initial_evaluation = evaluate_log_temperature(
        initial_log_temperature, False
    )
    for _ in range(6 if use_initial_bolometric_rescaling else 0):
        if resume_in_formal_flux_phase:
            # The caller has identified this as a complete checkpoint rather
            # than a neighboring/interpolated seed.  Preserve its thermal
            # normalization and honor the direct formal-flux resume contract;
            # a homologous rescaling would turn the checkpoint back into a
            # fresh convective-gradient warm start.
            break
        initial_total_flux = np.asarray(
            initial_evaluation.payload["total_flux_interface"],
            dtype=np.float64,
        )
        initial_flux_ratio = float(initial_total_flux[0] / target_flux)
        if (
            not np.isfinite(initial_flux_ratio)
            or initial_flux_ratio <= 0.0
            or abs(initial_flux_ratio - 1.0) < flux_tolerance
        ):
            break
        gray_log_correction = float(
            np.clip(-0.25 * np.log(initial_flux_ratio), -0.12, 0.12)
        )
        accepted = False
        for damping in (1.0, 0.5, 0.25):
            log_correction = damping * gray_log_correction
            if abs(log_correction) <= np.finfo(np.float64).eps:
                continue
            candidate_log_temperature = (
                initial_log_temperature + log_correction
            )
            candidate_evaluation = evaluate_log_temperature(
                candidate_log_temperature, False
            )
            initial_bolometric_rescaling_evaluations += 1
            if surface_flux_error(candidate_evaluation) < surface_flux_error(
                initial_evaluation
            ) and seed_merit(candidate_evaluation) < seed_merit(
                initial_evaluation
            ):
                initial_log_temperature = candidate_log_temperature
                initial_evaluation = candidate_evaluation
                initial_bolometric_temperature_scale *= float(
                    np.exp(log_correction)
                )
                accepted = True
                break
        if not accepted:
            break

    restart_reconditioned_after_bolometric_rescaling = False
    if (
        resume_in_formal_flux_phase
        and mixing_length_alpha is not None
        and initial_bolometric_temperature_scale != 1.0
    ):
        # A supplied checkpoint can resume directly in the exact flux phase
        # only while its thermal normalization is retained.  If the guarded
        # formal-transfer check above accepts a homologous rescaling, the EOS,
        # opacity, and ML2 efficiency have all changed; the old convective
        # gradient is no longer a consistent warm start.  Re-run the same
        # bounded conditioner used for a fresh atmosphere automatically.
        use_convective_gradient_preconditioner = True
        use_physical_flux_residual = False
        use_deep_gradient_conditioner = False
        solver_phase = "convective-gradient-preconditioner"
        restart_reconditioned_after_bolometric_rescaling = True

    convective_trial_correction_count = 0

    def correct_convective_trial(old_state, proposed_state):
        nonlocal convective_trial_correction_count
        log_temperature = temperature_map() @ proposed_state
        current = with_temperature(np.exp(log_temperature))
        gradient = interface_gradient_operator @ log_temperature
        transport = convection_transport(current, gradient)
        if transport is None:
            return proposed_state
        corrected = _cap_overcarrying_ml2_gradient(
            gradient, transport, target_flux
        )
        if np.array_equal(corrected, gradient):
            return proposed_state
        # Preserve the normalization degree of freedom. Changing the actual
        # gradient variables changes T consistently at all affected depths.
        result = proposed_state.copy()
        if nodal_coordinates:
            gradient_state = np.r_[
                log_temperature[normalization_anchor_index], corrected[1:]
            ]
            result = log_temperature_from_state @ gradient_state
        else:
            result[1:] = corrected[1:]
        convective_trial_correction_count += 1
        return result

    nonlinear_options = dict(
        maximum_iterations=max_iterations,
        residual_tolerance=flux_tolerance,
        step_tolerance=temperature_tolerance,
        initial_trust_radius=0.04,
        maximum_trust_radius=0.12,
        jacobian_refresh_interval=4,
        finite_difference_fallback_step=(
            5.0e-3 if seed.n_depth <= 12 else None
        ),
        callback=report_iteration,
        convergence_test=phase_converged,
        rejected_step_handoff=local_energy_handoff,
        acceptance_test=(
            (
                lambda old, trial: surface_flux_error(trial)
                <= surface_flux_error(old) + 1.0e-12
            )
            if safeguard_surface_flux
            else None
        ),
        step_measure=lambda old_state, new_state: float(
            np.max(np.abs(temperature_map() @ (new_state - old_state)))
        ),
    )
    nonlinear_evaluator = evaluate_state
    if energy_completion_active:
        from ._thermal_conditioning import (
            frozen_energy_evaluator,
            equilibrated_direction,
        )

        # Use the same unpenalized, freshly linearized energy solve as the
        # completion phase below. A regularized flux-oriented step can stall
        # while thin layers still have significant heating/cooling defects.
        solver_phase = "local-energy-completion"
        nonlinear_options.update(
            allow_initial_convergence=False,
            linear_regularization=0.0,
            jacobian_refresh_interval=1,
            # Frozen row weights are renewed with each tangent. Secants
            # between differently weighted residuals cannot update that J.
            broyden_updates=False,
            step_builder=lambda state, ev, jac, radius: equilibrated_direction(
                jac, ev.residual
            ),
        )
        nonlinear_evaluator = frozen_energy_evaluator(evaluate_state)
    if use_convective_trial_correction and mixing_length_alpha is not None:
        nonlinear_options["trial_projector"] = correct_convective_trial
    initial_options = dict(nonlinear_options)
    initial_state = state_from_log_temperature(initial_log_temperature)
    preconditioner_handoff: _PreconditionerFluxHandoff | None = None
    handoff_inactive_reason = None
    if not use_preconditioner_flux_handoff:
        handoff_inactive_reason = "not-requested"
    elif not use_convective_gradient_preconditioner:
        handoff_inactive_reason = "no-gradient-preconditioner"
    elif not enforce_local_energy_balance:
        handoff_inactive_reason = "local-energy-enforcement-disabled"
    handoff_metadata: dict[str, object] = {
        "requested": use_preconditioner_flux_handoff,
        "used": False,
        "triggered": False,
        "phase": "convective-gradient-preconditioner",
        "required_accepted_newton_updates": 2,
        "observed_accepted_newton_updates": 0,
        "eligible_accepted_newton_updates": 0,
        "consecutive_eligible_updates": 0,
        "ignored_callback_observations": 0,
        "observations": (),
        "certifies_equilibrium": False,
        "inactive_reason": handoff_inactive_reason,
    }
    if use_convective_gradient_preconditioner:
        # This phase solves a deliberately approximate local ML2-gradient
        # problem.  It is a warm-start construction, not a second atmosphere
        # solve, so spending the full production iteration budget here when
        # the local objective stalls only delays the conservative flux solve.
        initial_options["maximum_iterations"] = preconditioner_iteration_limit
        initial_options["stationary_completion_iterations"] = (
            preconditioner_stationary_completion_iterations
        )
        if use_preconditioner_flux_handoff and enforce_local_energy_balance:
            if initial_options.get("accepted_state_handoff") is not None:
                raise ValueError("preconditioner already has an accepted-state handoff")
            preconditioner_handoff = _PreconditionerFluxHandoff(
                initial_state, target_flux, flux_tolerance,
                initial_options.get("callback"),
            )
            initial_options["callback"] = preconditioner_handoff.callback
            initial_options["accepted_state_handoff"] = preconditioner_handoff.handoff
            handoff_metadata["used"] = True
    nonlinear_solver_segments: list[dict[str, object]] = []
    result = solve_trust_region_newton(
        initial_state,
        nonlinear_evaluator,
        **initial_options,
    )
    nonlinear_solver_segments.append(
        {
            "phase": solver_phase,
            **nonlinear_result_metadata(result),
        }
    )
    if preconditioner_handoff is not None:
        handoff_metadata.update(preconditioner_handoff.metadata())
    preconditioner_iterations = (
        result.iterations if use_convective_gradient_preconditioner else 0
    )
    conditioned_transport_iterations = 0
    formal_flux_continuations = 0
    if (
        use_convective_gradient_preconditioner
        and mixing_length_alpha is not None
    ):
        # An approximate conditioning phase can never be the final authority,
        # even when its incidental physical-flux residual is already below
        # tolerance.  Always enter the exact formal-flux phase at least once;
        # its physical equations supply the authoritative convergence status.
        # A large last conditioning step must not become a fictitious zero
        # correction merely because a new solver segment starts.
        use_physical_flux_residual = True
        solver_iteration_offset = preconditioner_iterations
        if (
            maximum_formal_flux_rosseland_depth is not None
            and not physical_flux_converged(
                result.state, result.evaluation, 0.0
            )
        ):
            # First impose the exact transfer flux through the radiative
            # surface and transition layers while retaining the local ML2
            # equation in the optically thick convective interior. This is
            # another bounded warm start: its diffusion-based deep residual
            # is not used to declare physical convergence.
            use_deep_gradient_conditioner = True
            solver_phase = "conditioned-transport-warm-start"
            conditioned_options = dict(nonlinear_options)
            conditioned_options["maximum_iterations"] = min(
                max_iterations, max(12, min(30, seed.n_depth))
            )
            conditioned_options["stationary_completion_iterations"] = 2
            result = solve_trust_region_newton(
                np.asarray(result.state, dtype=np.float64).copy(),
                evaluate_state,
                **conditioned_options,
            )
            nonlinear_solver_segments.append(
                {
                    "phase": solver_phase,
                    **nonlinear_result_metadata(result),
                }
            )
            conditioned_transport_iterations = result.iterations
            solver_iteration_offset += result.iterations

        # The converged equations are always the conservative formal-transfer
        # radiative flux plus the ML2 convective flux at every interface.
        # Explicitly turn off the deep residual replacement before the final
        # solve; otherwise a stationary diffusion warm start can masquerade
        # as a failed physical completion on a finite depth grid.
        use_deep_gradient_conditioner = False
        solver_phase = "formal-radiative-flux-completion"
        completion_state = np.asarray(result.state, dtype=np.float64).copy()
        if physical_temperature_coordinates and energy_completion_active:
            completion_state = log_temperature_from_state @ completion_state
            nodal_coordinates = True
        completion_options = dict(nonlinear_options)
        completion_options["allow_initial_convergence"] = bool(
            not energy_completion_active
            and result.history
            and result.history[-1].maximum_step < temperature_tolerance
        )
        precision_polish_active = bool(
            use_precision_polish
            and not completion_options["allow_initial_convergence"]
            and physical_flux_converged(result.state, result.evaluation, 0.0)
        )
        if precision_polish_active:
            # This phase change also invalidates the warm-start field cache.
            # No temperature/composition threshold and no relaxed equations:
            # the trust region and full rebuilt flux still accept every step.
            completion_options["linear_regularization"] = 0.0
        result = solve_trust_region_newton(
            completion_state, evaluate_state, **completion_options
        )
        nonlinear_solver_segments.append(
            {
                "phase": solver_phase,
                **nonlinear_result_metadata(result),
            }
        )

    # A trust-region collapse means that the accumulated Broyden history,
    # not the accepted atmosphere, has become unhelpful.  Rebuilding the
    # tangent and trust state from the last accepted atmosphere is a general
    # Newton restart within this calculation; no external atmosphere or
    # neighboring stellar model is involved. Apply it to every physical-flux
    # solve, including a composition-matched checkpoint that correctly
    # bypassed the provisional convective-gradient phase above.  Previously
    # that direct-resume path accidentally bypassed continuation as well.
    # Keep the number bounded so a genuinely inconsistent physical closure
    # still returns as unconverged rather than looping indefinitely.
    # Energy solves rebuild an exact tangent on every step. Repeating that
    # solve cannot repair stale Broyden history; return an unresolved grid to
    # its composition driver instead of restarting the same stalled system.
    if (
        use_physical_flux_residual
        and not use_deep_gradient_conditioner
        and not energy_completion_active
    ):
        for _ in range(maximum_formal_flux_continuations):
            if (
                result.diagnostics.terminal_reason
                == "rejected-step-phase-handoff"
            ):
                break
            if result.converged:
                break
            if result.iterations == 0:
                break
            solver_iteration_offset += result.iterations
            formal_flux_continuations += 1
            continuation_options = dict(nonlinear_options)
            if precision_polish_active:
                continuation_options["linear_regularization"] = 0.0
            continuation_options["allow_initial_convergence"] = bool(
                not energy_completion_active
                and result.history
                and result.history[-1].maximum_step < temperature_tolerance
            )
            result = solve_trust_region_newton(
                result.state, nonlinear_evaluator, **continuation_options
            )
            nonlinear_solver_segments.append(
                {
                    "phase": (
                        "formal-radiative-flux-continuation-"
                        f"{formal_flux_continuations}"
                    ),
                    **nonlinear_result_metadata(result),
                }
            )

    thermal_metadata = None
    # The formal phase stops when its accepted (line-searched) step is small.
    # In optically thin outer layers the flux equations barely constrain T,
    # so a residual already at round-off can leave a large unrestricted
    # Newton proposal that the line search keeps rejecting.  The certificate
    # measures that proposal, so complete with local energy balance (which
    # does constrain those layers) whenever it is not yet stationary.
    final_unrestricted_step = (
        result.history[-1].unrestricted_maximum_step
        if result.history else None
    )
    if (
        enforce_local_energy_balance
        and not energy_completion_active
        and (
            not result.converged
            or not result.history
            or np.max(
                np.abs(
                    result.evaluation.payload[
                        "cell_energy_balance_relative_residual"
                    ]
                )
            )
            >= flux_tolerance
            or (
                final_unrestricted_step is not None
                and final_unrestricted_step >= temperature_tolerance
            )
        )
    ):
        solver_iteration_offset += result.iterations
        completion_state = temperature_map() @ result.state
        energy_completion_active = True
        use_physical_flux_residual = True
        use_deep_gradient_conditioner = False
        solver_phase = "local-energy-completion"
        nodal_coordinates = bool(physical_temperature_coordinates)
        completion_state = state_from_log_temperature(completion_state)
        energy_options = dict(
            nonlinear_options,
            allow_initial_convergence=False,
            linear_regularization=0.0,
        )
        from ._thermal_conditioning import (
            thermal_condition,
            frozen_energy_evaluator,
            equilibrated_direction,
        )
        from .nonlinear import NonlinearIteration

        solver_phase = "thermal-conditioning"

        def thermal_report(record, state, ev):
            report_iteration(
                NonlinearIteration(
                    iteration=record["iteration"],
                    residual_rms=float(np.sqrt(np.mean(ev.residual**2))),
                    residual_maximum=float(np.max(abs(ev.residual))),
                    maximum_step=record["maximum_log_temperature_change"],
                    trust_radius=0.04,
                    line_search_factor=1.0,
                    jacobian_recomputed=True,
                    residual_merit=float(np.linalg.norm(ev.residual)),
                    worst_residual_index=int(np.argmax(abs(ev.residual))),
                    rejected_trial_evaluations=0,
                    model_agreement=None,
                ),
                state,
                ev,
            )

        completion_state, thermal_metadata = thermal_condition(
            completion_state,
            evaluate_state,
            maximum_sweeps=min(max_iterations, energy_thermal_sweeps),
            local_tolerance=flux_tolerance,
            callback=thermal_report,
        )
        solver_iteration_offset += thermal_metadata["iterations"]
        solver_phase = "local-energy-completion"
        energy_options.update(
            jacobian_refresh_interval=1,
            broyden_updates=False,
            step_builder=lambda state, ev, jac, radius: equilibrated_direction(
                jac, ev.residual
            ),
        )
        result = solve_trust_region_newton(
            completion_state,
            frozen_energy_evaluator(evaluate_state),
            **energy_options,
        )
        nonlinear_solver_segments.append(
            dict(phase=solver_phase, **nonlinear_result_metadata(result))
        )

    total_solver_iterations = solver_iteration_offset + result.iterations
    payload = result.evaluation.payload
    final_atmosphere = payload["atmosphere"]
    assert isinstance(final_atmosphere, Atmosphere)
    total_flux_interface = np.asarray(
        payload["total_flux_interface"], dtype=np.float64
    )
    radiative_flux_interface = np.asarray(
        payload["radiative_flux_interface"], dtype=np.float64
    )
    convective_flux_interface = np.asarray(
        payload["convective_flux_interface"], dtype=np.float64
    )
    convective_flux = np.asarray(payload["convective_flux"], dtype=np.float64)
    final_transport = payload["convection_transport"]
    if final_transport is not None:
        # The ML2 residual already evaluated the exact Rosseland mean at this
        # accepted atmosphere.  Reuse it, especially when the final
        # evaluation also built a Jacobian and the composition callback's
        # one-entry opacity cache now contains the hotter tangent state.
        final_rosseland_opacity = np.asarray(
            final_transport["rosseland"], dtype=np.float64
        )
    else:
        final_rosseland_opacity = np.asarray(
            rosseland_opacity(final_atmosphere), dtype=np.float64
        )
    final_asymptotic = (
        np.asarray(
            final_transport["adiabatic_asymptotic_convection"], dtype=bool
        )
        if final_transport is not None
        else np.zeros(seed.n_depth, dtype=bool)
    )
    final_superadiabatic_increment = (
        np.asarray(
            final_transport["superadiabatic_temperature_increment"],
            dtype=np.float64,
        )
        if final_transport is not None
        else np.zeros(seed.n_depth, dtype=np.float64)
    )
    rosseland_depth = np.empty_like(final_atmosphere.column_mass)
    rosseland_depth[0] = (
        final_rosseland_opacity[0] * final_atmosphere.column_mass[0]
    )
    rosseland_depth[1:] = rosseland_depth[0] + np.cumsum(
        0.5
        * (final_rosseland_opacity[1:] + final_rosseland_opacity[:-1])
        * np.diff(final_atmosphere.column_mass)
    )
    final_residual = total_flux_interface / target_flux - 1.0
    final_step = result.history[-1].maximum_step if result.history else None
    common_metadata: dict[str, object] = {
        "model": (
            "non-gray-radiative-convective-equilibrium"
            if mixing_length_alpha is not None
            else "non-gray-radiative-equilibrium"
        ),
        "rosseland_opacity_cm2_g": final_rosseland_opacity,
        "convective_rosseland_quadrature": "composition callback on fixed spectral grid",
        "convective_jacobian": "ML2 gradient and local material coefficients",
        "scattering_source_verification": "independent scalar Feautrier closure",
        "cell_energy_balance_relative_residual": payload[
            "cell_energy_balance_relative_residual"
        ],
        "maximum_relative_cell_energy_balance_residual": float(
            np.max(np.abs(payload["cell_energy_balance_relative_residual"]))
        ),
        "maximum_cell_energy_balance_residual_depth_index": int(
            np.argmax(np.abs(payload["cell_energy_balance_relative_residual"]))
        ),
        "maximum_cell_energy_balance_defect_in_stellar_flux": float(
            np.max(
                np.abs(payload["cell_energy_balance_defect_in_stellar_flux"])
            )
        ),
        "lower_boundary_absorption_escape_bound": float(
            payload["lower_boundary_absorption_escape_bound"]
        ),
        "lower_boundary_thermalization_verified_by_absorption": bool(
            payload["lower_boundary_absorption_escape_bound"] < flux_tolerance
        ),
        "lower_boundary_screening_tolerance": float(flux_tolerance),
        "structure_solver": "adaptive-trust-region-newton",
        "convective_trial_correction_enabled": bool(
            use_convective_trial_correction
        ),
        "convective_trial_correction_proposals": int(
            convective_trial_correction_count
        ),
        "adaptive_structure_driver": "shared-lte",
        "local_energy_enforcement_requested": enforce_local_energy_balance,
        "local_energy_completion_used": energy_completion_active,
        "thermal_conditioning": thermal_metadata,
        "final_temperature_coordinates": (
            "nodal-log-temperature"
            if nodal_coordinates
            else "normalization-and-gradients"
        ),
        "structure_residual": (
            "conservative interface flux minus directly evaluated local energy; no replaced ML2 flux"
            if energy_completion_active
            else "bounded ML2-gradient warm starts followed by conservative "
            "formal interface total flux, with an equivalent ML2-gradient "
            "equation only where superadiabaticity is sub-resolution"
        ),
        "structure_jacobian": (
            "opacity-aware tangent transfer plus analytic ML2 gradient "
            "response"
        ),
        "radiative_equilibrium_iterations": total_solver_iterations,
        "convective_preconditioner_iterations": preconditioner_iterations,
        "conditioned_transport_warm_start_iterations": (
            conditioned_transport_iterations
        ),
        "convective_preconditioner_iteration_limit": (
            preconditioner_iteration_limit
            if use_convective_gradient_preconditioner
            else 0
        ),
        "resumed_directly_in_formal_flux_phase": (
            resume_in_formal_flux_phase
            and not restart_reconditioned_after_bolometric_rescaling
        ),
        "restart_reconditioned_after_bolometric_rescaling": (
            restart_reconditioned_after_bolometric_rescaling
        ),
        "formal_flux_completion_used": bool(
            any(
                segment["phase"] == "formal-radiative-flux-completion"
                for segment in nonlinear_solver_segments
            )
        ),
        "formal_flux_continuations": formal_flux_continuations,
        "precision_polish_requested": use_precision_polish,
        "precision_polish_used": precision_polish_active,
        "nonlinear_solver_terminal_reason": (
            result.diagnostics.terminal_reason
        ),
        "nonlinear_solver_final_worst_residual_depth_index": int(
            result.diagnostics.final_worst_residual_index
        ),
        "nonlinear_solver_final_worst_residual_rosseland_depth": float(
            rosseland_depth[result.diagnostics.final_worst_residual_index]
        ),
        "nonlinear_solver_final_worst_residual_temperature_K": float(
            final_atmosphere.temperature[
                result.diagnostics.final_worst_residual_index
            ]
        ),
        "nonlinear_solver_final_worst_residual_convective_flux_fraction": float(
            convective_flux_interface[
                result.diagnostics.final_worst_residual_index
            ]
            / target_flux
        ),
        "nonlinear_solver_residual_evaluations": int(
            sum(
                int(segment["residual_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_jacobian_evaluations": int(
            sum(
                int(segment["jacobian_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_accepted_iterations": int(
            sum(
                int(segment["accepted_iterations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_rejected_trial_evaluations": int(
            sum(
                int(segment["rejected_trial_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_infeasible_trial_rejections": int(
            sum(
                int(segment["infeasible_trial_rejections"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_rejected_directions": int(
            sum(
                int(segment["rejected_directions"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_segments": tuple(nonlinear_solver_segments),
        "radiative_equilibrium_converged": bool(
            result.converged
            and np.max(np.abs(final_residual)) < flux_tolerance
        ),
        # No accepted step means there is no measured correction. Infinity
        # cannot be serialized in a standards-compliant checkpoint, and zero
        # would falsely suggest a measured stationary solution.
        "radiative_equilibrium_maximum_log_temperature_correction": (
            None if final_step is None else float(final_step)
        ),
        "temperature_correction_measured": bool(
            result.history
            and result.history[-1].unrestricted_maximum_step is not None
            and result.history[-1].trust_radius >= temperature_tolerance
        ),
        "maximum_unrestricted_log_temperature_correction": (
            result.history[-1].unrestricted_maximum_step
            if result.history
            else None
        ),
        "radiative_equilibrium_flux_ratio": float(
            total_flux_interface[0] / target_flux
        ),
        "radiative_equilibrium_wavelength_points": int(wavelength.size),
        "radiative_equilibrium_maximum_total_flux_residual": float(
            np.max(np.abs(final_residual))
        ),
        "maximum_total_flux_residual": float(np.max(np.abs(final_residual))),
        "maximum_all_depth_total_flux_residual": float(
            np.max(np.abs(final_residual))
        ),
        "maximum_total_flux_residual_depth_index": int(
            np.argmax(np.abs(final_residual))
        ),
        "maximum_convective_flux_fraction": float(
            np.max(convective_flux) / target_flux
        ),
        "adiabatic_asymptotic_conditioning_enabled": bool(
            use_adiabatic_asymptotic_conditioning
        ),
        "adiabatic_asymptotic_interfaces": int(np.sum(final_asymptotic)),
        "adiabatic_asymptotic_maximum_omitted_log_temperature_increment": (
            float(np.max(final_superadiabatic_increment[final_asymptotic]))
            if np.any(final_asymptotic)
            else 0.0
        ),
        "radiative_flux_fraction_by_interface": (
            radiative_flux_interface / target_flux
        ),
        "convective_flux_fraction_by_interface": (
            convective_flux_interface / target_flux
        ),
        "total_flux_fraction_by_interface": total_flux_interface / target_flux,
        "convection": (
            "ML2-Bergeron-1992" if mixing_length_alpha is not None else "none"
        ),
        "mixing_length_alpha": mixing_length_alpha,
        "initial_temperature_was_supplied": bool(
            initial_temperature_was_supplied
        ),
        "initial_convective_gradient_projection": bool(
            initial_convective_gradient_projection_attempted
        ),
        "initial_convective_gradient_projection_mode": (
            initial_convective_gradient_projection_mode
        ),
        "initial_convective_gradient_projection_damping": float(
            initial_convective_gradient_projection_damping
        ),
        "initial_convective_gradient_projection_rejected_infeasible": int(
            initial_convective_gradient_projection_rejected_infeasible
        ),
        "initial_bolometric_temperature_scale": float(
            initial_bolometric_temperature_scale
        ),
        "initial_bolometric_rescaling_enabled": bool(
            use_initial_bolometric_rescaling
        ),
        "initial_bolometric_rescaling_evaluations": int(
            initial_bolometric_rescaling_evaluations
        ),
        "convective_gradient_preconditioner_used": bool(
            use_convective_gradient_preconditioner
        ),
        "maximum_formal_flux_continuations": int(
            maximum_formal_flux_continuations
        ),
        "temperature_normalization_anchor_depth_index": int(
            normalization_anchor_index
        ),
        "temperature_normalization_anchor_rosseland_depth": float(
            seed.rosseland_optical_depth[normalization_anchor_index]
        ),
        "maximum_formal_flux_rosseland_depth": (
            None
            if maximum_formal_flux_rosseland_depth is None
            else float(maximum_formal_flux_rosseland_depth)
        ),
        "formal_flux_taper_start_rosseland_depth": (
            None
            if formal_flux_taper_start_rosseland_depth is None
            else float(formal_flux_taper_start_rosseland_depth)
        ),
        "electron_scattering_source": (
            "coherent-isotropic direct block Feautrier solve"
        ),
        "electron_scattering_source_iterations_per_evaluation": int(
            payload["scattering_source_iterations"]
        ),
        "electron_scattering_source_final_maximum_relative_residual": float(
            payload["scattering_source_maximum_relative_residual"]
        ),
        "electron_scattering_source_final_negative_clipped_count": int(
            payload["source_negative_clipped_count"]
        ),
        "electron_scattering_source_final_maximum_negative_relative": float(
            payload["source_maximum_negative_relative"]
        ),
        "electron_scattering_source_final_worst_wavelength_index": int(
            payload["scattering_source_worst_wavelength_index"]
        ),
        "electron_scattering_source_final_worst_depth_index": int(
            payload["scattering_source_worst_depth_index"]
        ),
        "electron_scattering_source_final_worst_wavelength_angstrom": float(
            wavelength[payload["scattering_source_worst_wavelength_index"]]
        ),
        "electron_scattering_source_final_worst_rosseland_depth": float(
            rosseland_depth[payload["scattering_source_worst_depth_index"]]
        ),
    }
    if metadata is not None:
        # Caller/seed provenance must never overwrite newly measured results.
        common_metadata = {**metadata, **common_metadata}
    if mass_transfer:
        # Do not let copied seed metadata mislabel the equations actually used.
        common_metadata["transfer_discretization"] = "column-mass"
    # This policy record describes the current solve, not inherited seed history.
    if use_preconditioner_flux_handoff:
        common_metadata["convective_preconditioner_flux_handoff"] = handoff_metadata
    else:
        common_metadata.pop("convective_preconditioner_flux_handoff", None)
    from ._convergence import equilibrium_certificate

    common_metadata["radiative_equilibrium_solver_converged"] = (
        common_metadata["radiative_equilibrium_converged"]
    )
    common_metadata["equilibrium_certificate"] = equilibrium_certificate(
        common_metadata,
        flux_tolerance=flux_tolerance,
        temperature_tolerance=temperature_tolerance,
    )
    common_metadata["radiative_equilibrium_converged"] = common_metadata[
        "equilibrium_certificate"
    ]["verified"]
    return Atmosphere(
        effective_temperature=final_atmosphere.effective_temperature,
        logg=final_atmosphere.logg,
        rosseland_optical_depth=rosseland_depth,
        column_mass=final_atmosphere.column_mass,
        temperature=final_atmosphere.temperature,
        gas_pressure=final_atmosphere.gas_pressure,
        mass_density=final_atmosphere.mass_density,
        neutral_h_density=final_atmosphere.neutral_h_density,
        proton_density=final_atmosphere.proton_density,
        electron_density=final_atmosphere.electron_density,
        metadata=common_metadata,
        hydrogen_lte_state=final_atmosphere.hydrogen_lte_state,
        helium_lte_state=final_atmosphere.helium_lte_state,
    )
