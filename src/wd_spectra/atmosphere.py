"""One-dimensional white-dwarf atmosphere structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid
from .eos import (
    HeliumLTEState,
    HydrogenHeliumLTEState,
    HydrogenLTEState,
    hummer_mihalas_helium_lte,
    hummer_mihalas_helium_lte_with_reos3,
    hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_lte,
    ideal_hydrogen_lte,
)

if TYPE_CHECKING:
    from .dense_eos import HeliumREOS3Table
    from .d6 import TOPbasePhotoionizationDatabase
    from .metals import (
        AtomicDatabase,
        CaIHeProfileTable,
        CaIIHeProfileTable,
        MgHeRedWingTable,
        MgIIHeProfileTable,
        VernerPhotoionizationDatabase,
    )
    from .molecules import H2H2CollisionInducedAbsorptionTable
    from .hydrogen_self import BarklemSelfBroadeningTable
    from .jackson_lyman import JacksonLymanProfileTable
    from .quasimolecular import AllardUnifiedLymanTable


FloatArray = NDArray[np.float64]


def _metal_line_opacity_sampling_grid(
    line_centers_angstrom: FloatArray,
    *,
    maximum_wing_sampled_lines: int = 1_000,
) -> tuple[FloatArray, int]:
    """Return a bounded transfer grid for a ranked structural line list.

    ``selected_metal_lines`` returns lines from strongest to weakest.  Five
    samples are retained for the strongest lines, whose resolved cores and
    near wings materially affect the flux integral.  The remaining line
    forest receives a compact three-point core stencil.  Keeping bracketing
    samples is important: a single isolated line-center point would acquire
    an arbitrary trapezoidal wavelength weight set by the next unrelated
    transition.  This bounds the grid growth without changing the selected
    opacity contributors or assigning weak lines artificially broad bins.
    """

    centers = np.asarray(line_centers_angstrom, dtype=np.float64)
    if centers.ndim != 1 or np.any(~np.isfinite(centers)):
        raise ValueError("metal line centers must be a finite one-dimensional array")
    if maximum_wing_sampled_lines < 0:
        raise ValueError("maximum_wing_sampled_lines must be non-negative")
    if centers.size == 0:
        return centers.copy(), 0
    n_wing = min(int(maximum_wing_sampled_lines), int(centers.size))
    samples: list[FloatArray] = []
    if n_wing:
        offsets = np.asarray((-2.0, -0.5, 0.0, 0.5, 2.0))
        samples.append((centers[:n_wing, np.newaxis] + offsets).ravel())
    if n_wing < centers.size:
        core_offsets = np.asarray((-0.05, 0.0, 0.05))
        samples.append(
            (centers[n_wing:, np.newaxis] + core_offsets).ravel()
        )
    return np.unique(np.concatenate(samples)), n_wing


def _solve_bracketed_log_root(
    residual: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    residual_tolerance: float = 2.0e-11,
    interval_tolerance: float = 2.0e-11,
    maximum_iterations: int = 40,
) -> float:
    """Solve a monotone scalar residual with safeguarded secant steps.

    The hydrostatic equation is best conditioned in the logarithm of the
    pressure increment.  Its residual is nearly linear there, so bisection's
    fixed 45 opacity evaluations are unnecessarily expensive.  Secant steps
    converge rapidly, while the retained sign-changing bracket makes the
    solve as robust as the former bisection.
    """

    function = residual  # Keep the hot-loop calls below compact.
    f_lower = float(function(lower))
    while f_lower > 0.0:
        lower -= np.log(10.0)
        f_lower = float(function(lower))
    f_upper = float(function(upper))
    while f_upper < 0.0:
        upper += np.log(10.0)
        f_upper = float(function(upper))

    replaced_side = 0
    for _ in range(maximum_iterations):
        width = upper - lower
        if width <= interval_tolerance:
            return 0.5 * (lower + upper)
        candidate = upper - f_upper * width / (f_upper - f_lower)
        if not lower < candidate < upper:
            candidate = 0.5 * (lower + upper)
        f_candidate = float(function(candidate))
        if abs(f_candidate) <= residual_tolerance:
            return candidate
        if f_candidate < 0.0:
            lower, f_lower = candidate, f_candidate
            if replaced_side < 0:
                f_upper *= 0.5
            replaced_side = -1
        else:
            upper, f_upper = candidate, f_candidate
            if replaced_side > 0:
                f_lower *= 0.5
            replaced_side = 1
    return 0.5 * (lower + upper)


def _node_values_on_upper_interfaces(
    values: FloatArray,
    *,
    surface_value: float | None = None,
) -> FloatArray:
    """Map depth-node values to the upper cell interfaces."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional depth array")
    result = np.empty_like(array)
    result[0] = array[0] if surface_value is None else float(surface_value)
    result[1:] = 0.5 * (array[:-1] + array[1:])
    return result


def _upper_interface_values_on_nodes(values: FloatArray) -> FloatArray:
    """Map upper-interface values back to the atmosphere depth nodes."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional depth array")
    result = np.empty_like(array)
    result[:-1] = 0.5 * (array[:-1] + array[1:])
    result[-1] = array[-1]
    return result


@dataclass(frozen=True)
class Atmosphere:
    """A plane-parallel atmosphere sampled from the surface inward.

    All quantities use cgs units.  The optical-depth and column-mass arrays
    must increase inward.
    """

    effective_temperature: float
    logg: float
    rosseland_optical_depth: FloatArray
    column_mass: FloatArray
    temperature: FloatArray
    gas_pressure: FloatArray
    mass_density: FloatArray
    neutral_h_density: FloatArray
    proton_density: FloatArray
    electron_density: FloatArray
    metadata: dict[str, object]
    hydrogen_lte_state: HydrogenLTEState | None = None
    helium_lte_state: HeliumLTEState | None = None

    @property
    def gravity(self) -> float:
        """Surface gravity in cm s^-2."""

        return 10.0**self.logg

    @property
    def n_depth(self) -> int:
        return int(self.temperature.size)


def _atmosphere_from_hydrogen_helium_state(
    effective_temperature: float,
    logg: float,
    rosseland_optical_depth: FloatArray,
    column_mass: FloatArray,
    temperature: FloatArray,
    gas_pressure: FloatArray,
    state: HydrogenHeliumLTEState,
    metadata: dict[str, object],
) -> Atmosphere:
    """Build an :class:`Atmosphere` from one shared mixed-composition EOS."""

    hydrogen = state.hydrogen_lte_state
    return Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=rosseland_optical_depth,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=state.mass_density,
        neutral_h_density=hydrogen.neutral_h_density,
        proton_density=hydrogen.proton_density,
        electron_density=state.electron_density,
        metadata=({**metadata, "includes_molecular_equilibrium": True,
                   "mixed_chemical_model": state.chemical_model}
                  if state.chemical_model == "molecular-h-he-hm" else metadata),
        hydrogen_lte_state=hydrogen,
        helium_lte_state=state.helium_lte_state,
    )


def gray_hydrogen_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-6,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
) -> Atmosphere:
    r"""Construct an LTE, hydrostatic Eddington-gray pure-H atmosphere.

    The structure obeys

    .. math:: T^4 = \frac{3}{4}T_\mathrm{eff}^4(\tau_R + q)

    and ``P_g = g tau_R / kappa_R``.  Constant Rosseland opacity is a
    controlled approximation for solver validation, not yet a realistic DA
    atmosphere.
    """

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")

    tau = np.geomspace(tau_min, tau_max, n_depth, dtype=np.float64)
    temperature = effective_temperature * (
        0.75 * (tau + hopf_constant)
    ) ** 0.25
    column_mass = tau / rosseland_opacity
    gas_pressure = 10.0**logg * column_mass
    eos = hummer_mihalas_hydrogen_lte(
        temperature,
        gas_pressure,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )

    return Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={
            "model": "eddington-gray",
            "composition": "pure-hydrogen",
            "eos": (
                "q-mhd-correlated-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "includes_molecular_equilibrium": bool(include_molecules),
        },
        hydrogen_lte_state=eos,
    )


def hydrogen_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-6,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
) -> Atmosphere:
    r"""Construct a gray-temperature atmosphere with a physical pressure scale.

    The Eddington temperature law is retained, but hydrostatic equilibrium is
    integrated using a local Rosseland mean computed from the implemented LTE
    hydrogen continuum.  This removes the arbitrary constant opacity from the
    density structure while stopping short of non-gray radiative equilibrium.
    """

    # Reuse the gray constructor for input validation and the temperature grid.
    seed = gray_hydrogen_atmosphere(
        effective_temperature,
        logg,
        n_depth=n_depth,
        tau_min=tau_min,
        tau_max=tau_max,
        rosseland_opacity=1.0,
        hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    from .opacity import rosseland_mean_hydrogen_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def opacity_at(temp: float, pressure: float) -> float:
        eos = hummer_mihalas_hydrogen_lte(
            np.array([temp]),
            np.array([pressure]),
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
        )
        point = Atmosphere(
            effective_temperature=float(effective_temperature),
            logg=float(logg),
            rosseland_optical_depth=np.array([1.0]),
            column_mass=np.array([pressure / gravity]),
            temperature=np.array([temp]),
            gas_pressure=np.array([pressure]),
            mass_density=np.atleast_1d(eos.mass_density),
            neutral_h_density=np.atleast_1d(eos.neutral_h_density),
            proton_density=np.atleast_1d(eos.proton_density),
            electron_density=np.atleast_1d(eos.electron_density),
            metadata={},
            hydrogen_lte_state=eos,
        )
        return float(
            rosseland_mean_hydrogen_continuum_opacity(
                point, h2_h2_cia_table=h2_h2_cia_table
            )[0]
        )

    def solve_log_increment(
        previous_pressure: float, delta_tau: float, midpoint_temperature: float
    ) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            midpoint_pressure = previous_pressure + 0.5 * increment
            return log_increment + np.log(opacity_at(midpoint_temperature, midpoint_pressure)) - np.log(target)

        lower = np.log(max(target / 1.0e12, 1.0e-20))
        upper = np.log(max(target / 1.0e-12, 1.0e-19))
        return float(np.exp(_solve_bracketed_log_root(residual, lower, upper)))

    # The surface pressure is the integral across the first optical-depth bin.
    gas_pressure = np.empty_like(tau)
    gas_pressure[0] = solve_log_increment(0.0, float(tau[0]), float(temperature[0]))
    for index in range(1, tau.size):
        gas_pressure[index] = gas_pressure[index - 1] + solve_log_increment(
            float(gas_pressure[index - 1]),
            float(tau[index] - tau[index - 1]),
            float(0.5 * (temperature[index] + temperature[index - 1])),
        )

    column_mass = gas_pressure / gravity
    eos = hummer_mihalas_hydrogen_lte(
        temperature,
        gas_pressure,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    provisional = Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={},
        hydrogen_lte_state=eos,
    )
    rosseland_opacity = rosseland_mean_hydrogen_continuum_opacity(
        provisional, h2_h2_cia_table=h2_h2_cia_table
    )
    return Atmosphere(
        effective_temperature=provisional.effective_temperature,
        logg=provisional.logg,
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={
            "model": "eddington-gray-temperature/hydrogen-continuum-hydrostatic",
            "composition": "pure-hydrogen",
            "eos": (
                "q-mhd-correlated-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": rosseland_opacity,
            "hopf_constant": float(hopf_constant),
            "includes_molecular_equilibrium": bool(include_molecules),
            "includes_h2_h2_collision_induced_absorption": bool(
                include_molecules and h2_h2_cia_table is not None
            ),
        },
        hydrogen_lte_state=eos,
    )


def gray_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    helium_reos3_table: HeliumREOS3Table | None = None,
) -> Atmosphere:
    """Construct an LTE Eddington-gray, hydrostatic pure-He atmosphere."""

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")
    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (0.75 * (tau + hopf_constant)) ** 0.25
    column_mass = tau / rosseland_opacity
    pressure = 10.0**logg * column_mass
    eos_function = (
        hummer_mihalas_helium_lte
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3
    )
    eos = eos_function(
        temperature,
        pressure,
        *((helium_reos3_table,) if helium_reos3_table is not None else ()),
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    return Atmosphere(
        float(effective_temperature), float(logg), tau, column_mass,
        temperature, pressure, eos.mass_density,
        np.zeros(n_depth), np.zeros(n_depth), eos.electron_density,
        {
            "model": "eddington-gray",
            "composition": "pure-helium",
            "eos": "q-mhd-helium-occupation-probability" if correlated_microfields else "hummer-mihalas-helium-occupation-probability",
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
        },
        helium_lte_state=eos,
    )


def gray_hydrogen_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = 0.5,
) -> Atmosphere:
    """Construct a homogeneous atomic H/He Eddington-gray atmosphere."""

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg) or not np.isfinite(log_hydrogen_to_helium):
        raise ValueError("logg and log_hydrogen_to_helium must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")
    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (
        0.75 * (tau + hopf_constant)
    ) ** 0.25
    column_mass = tau / rosseland_opacity
    pressure = 10.0**logg * column_mass
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        pressure,
        log_hydrogen_to_helium,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    return _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {
            "model": "eddington-gray",
            "composition": "homogeneous-hydrogen-helium",
            "log_hydrogen_to_helium": float(log_hydrogen_to_helium),
            "eos": (
                "q-mhd-hydrogen-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "hydrogen_neutral_radius_scale": float(
                hydrogen_neutral_radius_scale
            ),
            "helium_neutral_radius_scale": float(helium_neutral_radius_scale),
            "mixed_molecular_chemistry": False,
        },
    )


def hydrogen_helium_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = 0.5,
    rosseland_frequency_points: int = 160,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
) -> Atmosphere:
    """Gray-temperature mixed atmosphere with a physical pressure scale."""

    seed = gray_hydrogen_helium_atmosphere(
        effective_temperature,
        logg,
        log_hydrogen_to_helium,
        n_depth=n_depth,
        tau_min=tau_min,
        tau_max=tau_max,
        rosseland_opacity=1.0,
        hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
    )
    from .mixture import rosseland_mean_hydrogen_helium_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def point_at(temp: float, pressure: float) -> Atmosphere:
        state = hummer_mihalas_hydrogen_helium_lte(
            np.asarray([temp]),
            np.asarray([pressure]),
            log_hydrogen_to_helium,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            helium_neutral_radius_scale=helium_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        return _atmosphere_from_hydrogen_helium_state(
            effective_temperature,
            logg,
            np.ones(1),
            np.asarray([pressure / gravity]),
            np.asarray([temp]),
            np.asarray([pressure]),
            state,
            {},
        )

    def opacity_at(temp: float, pressure: float) -> float:
        return float(
            rosseland_mean_hydrogen_helium_continuum_opacity(
                point_at(temp, pressure),
                n_frequency=rosseland_frequency_points,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )[0]
        )

    def solve_increment(previous: float, delta_tau: float, temp: float) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            opacity = opacity_at(temp, previous + 0.5 * increment)
            return log_increment + np.log(opacity) - np.log(target)

        return float(
            np.exp(
                _solve_bracketed_log_root(
                    residual,
                    np.log(max(target / 1.0e12, 1.0e-20)),
                    np.log(max(target / 1.0e-12, 1.0e-19)),
                )
            )
        )

    pressure = np.empty_like(tau)
    pressure[0] = solve_increment(0.0, float(tau[0]), float(temperature[0]))
    for depth in range(1, n_depth):
        pressure[depth] = pressure[depth - 1] + solve_increment(
            float(pressure[depth - 1]),
            float(tau[depth] - tau[depth - 1]),
            float(0.5 * (temperature[depth] + temperature[depth - 1])),
        )
    column_mass = pressure / gravity
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        pressure,
        log_hydrogen_to_helium,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    provisional = _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {},
    )
    rosseland = rosseland_mean_hydrogen_helium_continuum_opacity(
        provisional,
        n_frequency=rosseland_frequency_points,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    return _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {
            "model": "eddington-gray-temperature/hydrogen-helium-continuum-hydrostatic",
            "composition": "homogeneous-hydrogen-helium",
            "log_hydrogen_to_helium": float(log_hydrogen_to_helium),
            "eos": (
                "q-mhd-hydrogen-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": rosseland,
            "hopf_constant": float(hopf_constant),
            "hydrogen_neutral_radius_scale": float(
                hydrogen_neutral_radius_scale
            ),
            "helium_neutral_radius_scale": float(helium_neutral_radius_scale),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "mixed_molecular_chemistry": False,
        },
    )


def helium_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    rosseland_frequency_points: int = 160,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    log_hydrogen_abundance: float | None = None,
    include_dense_helium_metal_ionization: bool = True,
    helium_reos3_table: HeliumREOS3Table | None = None,
) -> Atmosphere:
    """Gray-temperature He atmosphere with a physical pressure scale.

    When metal data and abundances are supplied, metal electron donation and
    the resulting He-minus/free-free continuum are included during the
    hydrostatic integration.  Metal lines are intentionally omitted from this
    Rosseland-mean seed; their structural blanketing belongs in the subsequent
    non-gray radiative-equilibrium calculation.
    """

    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if log_hydrogen_abundance is not None and metal_database is None:
        return hydrogen_helium_continuum_atmosphere(
            effective_temperature,
            logg,
            log_hydrogen_abundance,
            n_depth=n_depth,
            tau_min=tau_min,
            tau_max=tau_max,
            hopf_constant=hopf_constant,
            correlated_microfields=correlated_microfields,
            helium_neutral_radius_scale=neutral_radius_scale,
            rosseland_frequency_points=rosseland_frequency_points,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )

    seed = gray_helium_atmosphere(
        effective_temperature, logg, n_depth=n_depth, tau_min=tau_min,
        tau_max=tau_max, rosseland_opacity=1.0, hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        neutral_radius_scale=neutral_radius_scale,
        helium_reos3_table=helium_reos3_table,
    )
    from .helium import rosseland_mean_helium_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def opacity_at(temp: float, pressure: float) -> float:
        eos = (
            hummer_mihalas_helium_lte(
                np.asarray([temp]), np.asarray([pressure]),
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            if helium_reos3_table is None
            else hummer_mihalas_helium_lte_with_reos3(
                np.asarray([temp]), np.asarray([pressure]), helium_reos3_table,
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
        )
        point = Atmosphere(
            float(effective_temperature), float(logg), np.ones(1),
            np.asarray([pressure / gravity]), np.asarray([temp]),
            np.asarray([pressure]), eos.mass_density, np.zeros(1), np.zeros(1),
            eos.electron_density, {}, helium_lte_state=eos,
        )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                point,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            point = atmosphere_with_metal_electrons(point, metal_state)
        return float(
            rosseland_mean_helium_continuum_opacity(
                point,
                n_frequency=rosseland_frequency_points,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )[0]
        )

    def solve_increment(previous: float, delta_tau: float, temp: float) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            opacity = opacity_at(temp, previous + 0.5 * increment)
            return log_increment + np.log(opacity) - np.log(target)

        return float(np.exp(_solve_bracketed_log_root(
            residual,
            np.log(max(target / 1.0e12, 1.0e-20)),
            np.log(max(target / 1.0e-12, 1.0e-19)),
        )))

    pressure = np.empty_like(tau)
    pressure[0] = solve_increment(0.0, float(tau[0]), float(temperature[0]))
    for depth in range(1, n_depth):
        pressure[depth] = pressure[depth - 1] + solve_increment(
            float(pressure[depth - 1]), float(tau[depth] - tau[depth - 1]),
            float(0.5 * (temperature[depth] + temperature[depth - 1])),
        )
    column_mass = pressure / gravity
    eos = (
        hummer_mihalas_helium_lte(
            temperature, pressure, neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3(
            temperature, pressure, helium_reos3_table,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    )
    provisional = Atmosphere(
        float(effective_temperature), float(logg), tau, column_mass,
        temperature, pressure, eos.mass_density, np.zeros(n_depth),
        np.zeros(n_depth), eos.electron_density, {}, helium_lte_state=eos,
    )
    metal_state = None
    if metal_database is not None and metal_abundances is not None:
        from .metals import atmosphere_with_metal_electrons, metal_lte_state

        metal_state = metal_lte_state(
            provisional,
            metal_database,
            metal_abundances,
            reference_species="He",
            include_dense_helium_ionization=include_dense_helium_metal_ionization,
            log_hydrogen_abundance=log_hydrogen_abundance,
        )
        provisional = atmosphere_with_metal_electrons(provisional, metal_state)
    rosseland = rosseland_mean_helium_continuum_opacity(
        provisional,
        n_frequency=rosseland_frequency_points,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    return Atmosphere(
        provisional.effective_temperature, provisional.logg, tau, column_mass,
        temperature, pressure, provisional.mass_density,
        provisional.neutral_h_density, provisional.proton_density,
        provisional.electron_density,
        {
            "model": "eddington-gray-temperature/helium-continuum-hydrostatic",
            "composition": (
                "metal-polluted-helium" if metal_state is not None else "pure-helium"
            ),
            "eos": "q-mhd-helium-occupation-probability" if correlated_microfields else "hummer-mihalas-helium-occupation-probability",
            "rosseland_opacity_cm2_g": rosseland,
            "hopf_constant": float(hopf_constant),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "metal_abundances": (
                dict(metal_state.log_number_abundance)
                if metal_state is not None else {}
            ),
            "log_hydrogen_abundance": log_hydrogen_abundance,
            "metal_electron_feedback": (
                "hydrostatic continuum opacity and charge neutrality"
                if metal_state is not None else "disabled"
            ),
        },
        hydrogen_lte_state=provisional.hydrogen_lte_state,
        helium_lte_state=provisional.helium_lte_state,
    )


def radiative_equilibrium_hydrogen_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-10,
    max_iterations: int = 500,
    temperature_tolerance: float = 2.0e-4,
    flux_tolerance: float = 2.0e-3,
    enforce_local_energy_balance: bool = True,
    n_continuum_wavelength: int = 600,
    include_balmer_lines: bool = True,
    include_paschen_lines: bool = True,
    include_brackett_lines: bool = True,
    include_balmer_self_broadening: bool = True,
    balmer_self_broadening_quadrature_order: int = 32,
    balmer_self_broadening_prescription: str = "barklem",
    balmer_self_broadening_truncation_closure: str = "renormalize",
    barklem_self_table: BarklemSelfBroadeningTable | None = None,
    include_lyman_lines: bool = True,
    resolve_balmer_line_cores: bool = True,
    balmer_line_core_step_angstrom: float = 0.10,
    resolve_lyman_line_cores: bool | None = None,
    include_series_pseudocontinuum: bool = False,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
    include_neutral_lyman_alpha_wing: bool = True,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    jackson_lyman_table: JacksonLymanProfileTable | None = None,
    mixing_length_alpha: float | None = 0.7,
    n_angle: int = 3,
    initial_temperature: FloatArray | None = None,
    initial_column_mass: FloatArray | None = None,
    initial_atmosphere: Atmosphere | None = None,
    hydrogen_eos_function: Callable[
        [FloatArray, FloatArray], HydrogenLTEState
    ]
    | None = None,
    balmer_opacity_function: Callable[[Atmosphere, FloatArray], FloatArray]
    | None = None,
    continuum_opacity_function: Callable[[Atmosphere, FloatArray], FloatArray]
    | None = None,
    additional_structure_wavelength_angstrom: ArrayLike | None = None,
    scattering_opacity_function: Callable[[Atmosphere, FloatArray], FloatArray]
    | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    include_metal_lines: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-2,
    maximum_metal_lines: int | None = 1_000,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
    structure_solver: Literal["adaptive-newton"] = "adaptive-newton",
) -> Atmosphere:
    """Relax an H-dominated atmosphere toward non-gray energy equilibrium.

    Hydrostatic pressure remains exact on a fixed column-mass grid. With
    ``structure_solver="adaptive-newton"``, the unknowns are surface
    ``ln(T)`` and the layer logarithmic temperature gradients. A conservative
    Feautrier radiative flux and interface ML2 flux are solved together.  A
    local ML2-gradient phase first preconditions the nearly adiabatic interior;
    if necessary, the same-grid solve then restores the exact formal-flux
    residual in radiative layers.  The tangent Feautrier operator includes
    both source-function and opacity/optical-depth motion, with trust-region
    and backtracking globalization. The directly evaluated total flux must satisfy the
    requested tolerance. The default ``alpha=0.7`` matches the
    current Koester DA calibration; pass ``None`` for a strictly radiative
    control model.  The Lyman through Brackett series are included in the
    structural opacity by default.  The radiative-equilibrium
    grid extends to Rosseland optical depth 1e-10 by default because a grid
    beginning at 1e-6 can already be optically thick in the Lyman continuum
    of cool, high-gravity models.  Narrow Balmer-core points are included in
    the structure grid because omitting them produces an artificially warm
    upper atmosphere and broad, shallow optical cores.  Fine Lyman-core
    sampling is selected automatically at ``Teff >= 30000 K``.  In that
    regime it removes a warm bias from the Balmer-core-forming layers without
    changing the deeper line wings.  It remains off by default below 30000 K
    because the present strict-LTE solver can develop a neutral-opacity/
    cooling runaway in cool outer layers.  Pass an explicit boolean to
    override the automatic choice.

    When ``metal_database`` and ``metal_abundances`` are supplied, metal
    electron donation, bound--free opacity, and the selected structural metal
    lines are recomputed at every trial structure. The H chemical equilibrium
    is reclosed at the shared electron density. Metals remain trace in the
    pressure and thermodynamic derivatives used by ML2.

    If supplied, ``iteration_callback`` receives the updated atmosphere and a
    compact convergence record after every accepted correction. It can be
    used to write resumable checkpoints or inspect spectra during long cool-DA
    relaxations without changing the numerical iteration.  A caller may
    provide ``hydrogen_eos_function(T, P)`` to replace the ordinary HM/Q-MHD
    chemical-equilibrium solve while retaining the same hydrostatic and
    radiative-equilibrium machinery; the DAH module uses this controlled hook
    for its stationary-state magnetic EOS experiment.  Likewise,
    ``balmer_opacity_function(atmosphere, wavelength)`` can replace the
    zero-field structural Balmer opacity while leaving the remaining series
    and continuum controls unchanged.  ``continuum_opacity_function`` is the
    corresponding controlled hook for replacing the thermal continuum.  It
    must return the complete true-absorption continuum (not scattering), so a
    magnetic implementation can replace H I bound-free opacity without
    losing the otherwise validated free-free and molecular terms.  When it
    is supplied, the adaptive solver's Rosseland mean (its depth coordinate)
    is taken from the same continuum rather than the zero-field one.
    ``additional_structure_wavelength_angstrom`` adds frequency points to the
    structure mesh, e.g. around displaced magnetic Balmer components.
    ``scattering_opacity_function(atmosphere, wavelength)`` returns a change
    to the electron plus Rayleigh coherent scattering (e.g. magnetic Thomson
    scattering; adaptive-newton solver only).
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if structure_solver != "adaptive-newton":
        raise ValueError(
            "Only structure_solver='adaptive-newton' is supported"
        )
    if temperature_tolerance <= 0.0 or flux_tolerance <= 0.0:
        raise ValueError("convergence tolerances must be positive")
    if n_continuum_wavelength < 80:
        raise ValueError("n_continuum_wavelength must be at least 80")
    if n_angle < 1:
        raise ValueError("n_angle must be positive")
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError(
            "metal_database and metal_abundances must be supplied together"
        )
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError(
            "metal photoionization requires metal_database and abundances"
        )
    if minimum_metal_oscillator_strength <= 0.0:
        raise ValueError("minimum_metal_oscillator_strength must be positive")
    if maximum_metal_lines is not None and maximum_metal_lines < 1:
        raise ValueError("maximum_metal_lines must be positive or None")
    if balmer_self_broadening_quadrature_order < 8:
        raise ValueError(
            "balmer_self_broadening_quadrature_order must be at least 8"
        )
    if (
        not np.isfinite(balmer_line_core_step_angstrom)
        or balmer_line_core_step_angstrom <= 0.0
    ):
        raise ValueError(
            "balmer_line_core_step_angstrom must be finite and positive"
        )
    if mixing_length_alpha is not None and (
        not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be finite and positive")
    if initial_atmosphere is not None and (
        initial_temperature is not None or initial_column_mass is not None
    ):
        raise ValueError(
            "initial_atmosphere cannot be combined with initial temperature/grid"
        )
    resolved_lyman_line_cores = (
        effective_temperature >= 30_000.0
        if resolve_lyman_line_cores is None
        else bool(resolve_lyman_line_cores)
    )

    from .constants import LIGHT_SPEED, PLANCK
    from .opacity import BALMER_LINES, BRACKETT_LINES, LYMAN_LINES, PASCHEN_LINES, balmer_mass_absorption_coefficient, brackett_mass_absorption_coefficient, electron_scattering_mass_coefficient, hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient, hydrogen_continuum_mass_absorption_coefficient, hydrogen_rayleigh_scattering_mass_coefficient, lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient, paschen_mass_absorption_coefficient, lyman_mass_absorption_coefficient, rosseland_mean_hydrogen_continuum_opacity
    if initial_atmosphere is None:
        seed = hydrogen_continuum_atmosphere(
            effective_temperature,
            logg,
            n_depth=n_depth,
            tau_min=tau_min,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
            h2_h2_cia_table=h2_h2_cia_table,
        )
    else:
        if initial_atmosphere.hydrogen_lte_state is None:
            raise ValueError("initial_atmosphere must be a pure-H LTE structure")
        if not np.isclose(
            initial_atmosphere.effective_temperature,
            effective_temperature,
            rtol=0.0,
            atol=1.0e-8,
        ) or not np.isclose(
            initial_atmosphere.logg, logg, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "initial_atmosphere must match effective temperature and log(g)"
            )
        seed = initial_atmosphere
    relaxation_depth = seed.rosseland_optical_depth.copy()
    wavelength = np.geomspace(
        100.0, 100_000.0, n_continuum_wavelength, dtype=np.float64
    )
    if include_balmer_lines:
        wavelength = np.unique(
            np.concatenate((wavelength, np.arange(3500.0, 7000.1, 10.0)))
        )
        # Line cores control the optically thin temperature through
        # radiative-equilibrium cooling.  A 10-A wing mesh alone entirely
        # misses their sub-Angstrom opacity peaks.
        if resolve_balmer_line_cores:
            wavelength = np.unique(
                np.concatenate(
                    (
                        wavelength,
                        *(
                            line.wavelength_vacuum_angstrom
                            + np.arange(
                                -5.0,
                                5.0 + 0.5 * balmer_line_core_step_angstrom,
                                balmer_line_core_step_angstrom,
                            )
                            for line in BALMER_LINES[:4]
                        ),
                    )
                )
            )
    if additional_structure_wavelength_angstrom is not None:
        extra_wavelength = np.asarray(
            additional_structure_wavelength_angstrom, dtype=np.float64
        ).ravel()
        if np.any(~np.isfinite(extra_wavelength)) or np.any(
            extra_wavelength <= 0.0
        ):
            raise ValueError(
                "additional_structure_wavelength_angstrom must be finite and positive"
            )
        wavelength = np.unique(np.concatenate((wavelength, extra_wavelength)))
    infrared_line_offsets = np.asarray(
        (
            -300.0,
            -150.0,
            -75.0,
            -30.0,
            -10.0,
            -3.0,
            -1.0,
            0.0,
            1.0,
            3.0,
            10.0,
            30.0,
            75.0,
            150.0,
            300.0,
        ),
        dtype=np.float64,
    )
    if include_paschen_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    *(
                        line.wavelength_vacuum_angstrom
                        + infrared_line_offsets
                        for line in PASCHEN_LINES
                    ),
                )
            )
        )
    if include_brackett_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    *(
                        line.wavelength_vacuum_angstrom
                        + infrared_line_offsets
                        for line in BRACKETT_LINES
                    ),
                )
            )
        )
    if include_lyman_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    np.arange(900.0, 1250.1, 1.0),
                    np.arange(1255.0, 3000.1, 5.0),
                )
            )
        )
        if resolved_lyman_line_cores:
            wavelength = np.unique(
                np.concatenate(
                    (
                        wavelength,
                        *(
                            line.wavelength_vacuum_angstrom
                            + np.arange(-3.0, 3.0001, 0.05)
                            for line in LYMAN_LINES[:3]
                        ),
                    )
                )
            )
    metal_wing_sampled_lines = 0
    if metal_database is not None and metal_abundances is not None:
        from .metals import metal_lte_state, selected_metal_lines

        selection_state = metal_lte_state(
            seed,
            metal_database,
            metal_abundances,
            reference_species="H",
            include_dense_helium_ionization=False,
        )
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge]
                / np.maximum(
                    selection_state.element_number_density[element],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in selection_state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }
        structure_lines = selected_metal_lines(
            metal_database,
            metal_abundances,
            100.0,
            100_000.0,
            minimum_metal_oscillator_strength,
            maximum_metal_lines,
            effective_temperature,
            ion_stage_weight,
            flux_weighted=True,
        ) if include_metal_lines else []
        metal_centers = np.asarray(
            [line.wavelength_vacuum_angstrom for _, line in structure_lines]
        )
        if metal_centers.size:
            metal_grid, metal_wing_sampled_lines = (
                _metal_line_opacity_sampling_grid(metal_centers)
            )
            wavelength = np.unique(np.concatenate((
                wavelength,
                metal_grid,
            )))
    if metal_photoionization_database is not None:
        metal_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (fit.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for fit in metal_photoionization_database.fits.values()
            if metal_abundances is not None and fit.element in metal_abundances
        ])
        if metal_edge_grid.size:
            wavelength = np.unique(np.concatenate((
                wavelength,
                (
                    metal_edge_grid[:, np.newaxis]
                    * (1.0 + np.asarray((-1.0e-4, 1.0e-4)))
                ).ravel(),
            )))
    if metal_topbase_photoionization_database is not None:
        topbase_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (section.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for section in metal_topbase_photoionization_database.sections
            if metal_abundances is not None and section.element in metal_abundances
        ])
        if topbase_edge_grid.size:
            wavelength = np.unique(np.concatenate((
                wavelength,
                (
                    topbase_edge_grid[:, np.newaxis]
                    * (1.0 + np.asarray((-1.0e-4, 1.0e-4)))
                ).ravel(),
            )))
    if initial_temperature is None:
        temperature = seed.temperature.copy()
    else:
        temperature = np.asarray(initial_temperature, dtype=np.float64).copy()
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError(
                "initial_temperature must contain finite positive values"
            )
        if initial_column_mass is not None:
            initial_mass = np.asarray(initial_column_mass, dtype=np.float64)
            if (
                initial_mass.shape != temperature.shape
                or np.any(~np.isfinite(initial_mass))
                or np.any(initial_mass <= 0.0)
                or np.any(np.diff(initial_mass) <= 0.0)
            ):
                raise ValueError(
                    "initial_column_mass must be positive, increasing, and match initial_temperature"
                )
            temperature = np.interp(
                np.log(seed.column_mass),
                np.log(initial_mass),
                temperature,
                left=temperature[0],
                right=temperature[-1],
            )
        elif temperature.shape != seed.temperature.shape:
            raise ValueError(
                "initial_temperature must have one value per depth when initial_column_mass is omitted"
            )
    def with_temperature(values: FloatArray) -> Atmosphere:
        eos = (
            hummer_mihalas_hydrogen_lte(
                values,
                seed.gas_pressure,
                correlated_microfields=correlated_microfields,
                include_molecules=include_molecules,
                include_negative_hydrogen=include_negative_hydrogen,
                trihydrogen_ion_partition_model=(
                    trihydrogen_ion_partition_model
                ),
            )
            if hydrogen_eos_function is None
            else hydrogen_eos_function(values, seed.gas_pressure)
        )
        result = Atmosphere(
            effective_temperature=seed.effective_temperature,
            logg=seed.logg,
            rosseland_optical_depth=relaxation_depth,
            column_mass=seed.column_mass,
            temperature=values,
            gas_pressure=seed.gas_pressure,
            mass_density=eos.mass_density,
            neutral_h_density=eos.neutral_h_density,
            proton_density=eos.proton_density,
            electron_density=eos.electron_density,
            metadata=seed.metadata,
            hydrogen_lte_state=eos,
        )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                result,
                metal_database,
                metal_abundances,
                reference_species="H",
                include_dense_helium_ionization=False,
            )
            result = atmosphere_with_metal_electrons(result, metal_state)
        return result

    hydrogen_structure_absorption_cache: dict[str, object] = {}

    def true_absorption(current_atmosphere: Atmosphere) -> FloatArray:
        """Evaluate every thermal opacity used by the structure solver."""

        absorption = (
            hydrogen_continuum_mass_absorption_coefficient(
                current_atmosphere,
                wavelength,
                include_electron_scattering=False,
                include_rayleigh_scattering=False,
                h2_h2_cia_table=h2_h2_cia_table,
            )
            if continuum_opacity_function is None
            else continuum_opacity_function(current_atmosphere, wavelength)
        )
        if absorption.shape != (wavelength.size, current_atmosphere.n_depth):
            raise ValueError(
                "continuum_opacity_function must return shape "
                "(wavelength, depth)"
            )
        if np.any(~np.isfinite(absorption)) or np.any(absorption < 0.0):
            raise ValueError(
                "continuum_opacity_function returned non-finite or negative opacity"
            )
        if include_balmer_lines:
            absorption += (
                balmer_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    include_self_broadening=include_balmer_self_broadening,
                    self_broadening_quadrature_order=(
                        balmer_self_broadening_quadrature_order
                    ),
                    self_broadening_prescription=(
                        balmer_self_broadening_prescription
                    ),
                    self_broadening_truncation_closure=(
                        balmer_self_broadening_truncation_closure
                    ),
                    barklem_self_table=barklem_self_table,
                    profile_edge_optical_depth=1.0e-4,
                )
                if balmer_opacity_function is None
                else balmer_opacity_function(current_atmosphere, wavelength)
            )
        if include_paschen_lines:
            absorption += paschen_mass_absorption_coefficient(
                current_atmosphere, wavelength
            )
        if include_brackett_lines:
            absorption += brackett_mass_absorption_coefficient(
                current_atmosphere, wavelength
            )
        if include_lyman_lines:
            absorption += lyman_mass_absorption_coefficient(
                current_atmosphere,
                wavelength,
                unified_allard_table=unified_allard_table,
                jackson_lyman_table=jackson_lyman_table,
            )
            if include_series_pseudocontinuum:
                absorption += (
                    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                        current_atmosphere, wavelength
                    )
                )
            if include_neutral_lyman_alpha_wing:
                absorption += (
                    lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
                        current_atmosphere,
                        wavelength,
                        allard_table=(
                            unified_allard_table.lines.get((1, 2))
                            if (
                                unified_allard_table is not None
                                and not (
                                    jackson_lyman_table is not None
                                    and (1, 2) in jackson_lyman_table.lines
                                )
                            )
                            else None
                        ),
                    )
                )
        if metal_database is not None and metal_abundances is not None:
            from .metals import (
                metal_bound_free_mass_absorption_coefficient,
                metal_line_mass_absorption_coefficient,
                metal_lte_state,
            )

            metal_state = metal_lte_state(
                current_atmosphere,
                metal_database,
                metal_abundances,
                reference_species="H",
                include_dense_helium_ionization=False,
            )
            if metal_photoionization_database is not None:
                absorption += metal_bound_free_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_database,
                    metal_state,
                    metal_photoionization_database,
                    excluded_ions=(
                        ()
                        if metal_topbase_photoionization_database is None
                        else metal_topbase_photoionization_database.ion_stages
                    ),
                )
            if metal_topbase_photoionization_database is not None:
                from .d6 import topbase_bound_free_mass_absorption_coefficient

                absorption += topbase_bound_free_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_state,
                    metal_topbase_photoionization_database,
                )
            if include_metal_lines:
                absorption += metal_line_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_database,
                    metal_state,
                    minimum_oscillator_strength=(
                        minimum_metal_oscillator_strength
                    ),
                    maximum_lines=maximum_metal_lines,
                )
        hydrogen_structure_absorption_cache.clear()
        hydrogen_structure_absorption_cache.update(
            atmosphere=current_atmosphere,
            absorption=absorption,
        )
        return absorption

    from .adaptive_structure import (
        rosseland_mean_from_opacity_grid,
        solve_adaptive_lte_structure,
    )
    from .eos import hummer_mihalas_hydrogen_thermodynamics

    def scattering_opacity(current: Atmosphere) -> FloatArray:
        result = (
            electron_scattering_mass_coefficient(current)[np.newaxis, :]
            + hydrogen_rayleigh_scattering_mass_coefficient(current, wavelength)
        )
        if scattering_opacity_function is not None:
            extra = np.asarray(
                scattering_opacity_function(current, wavelength), dtype=np.float64
            )
            if extra.shape != (wavelength.size, current.n_depth) or np.any(
                ~np.isfinite(extra)
            ) or np.any(result + extra < 0.0):
                raise ValueError(
                    "scattering_opacity_function must return a finite "
                    "(wavelength, depth) change that keeps scattering nonnegative"
                )
            result = result + extra
        return result

    explicit_metal_rosseland_opacity = (
        metal_database is not None
        and (
            include_metal_lines
            or metal_photoionization_database is not None
            or metal_topbase_photoionization_database is not None
        )
    )

    def rosseland_opacity(current: Atmosphere) -> FloatArray:
        if continuum_opacity_function is not None and not explicit_metal_rosseland_opacity:
            # Use the same continuum physics for the Rosseland depth coordinate.
            return rosseland_mean_from_opacity_grid(
                wavelength,
                continuum_opacity_function(current, wavelength)
                + scattering_opacity(current),
                current.temperature,
            )
        if explicit_metal_rosseland_opacity:
            if (
                hydrogen_structure_absorption_cache.get("atmosphere")
                is current
            ):
                absorption = np.asarray(
                    hydrogen_structure_absorption_cache["absorption"],
                    dtype=np.float64,
                )
            else:
                absorption = true_absorption(current)
            return rosseland_mean_from_opacity_grid(
                wavelength,
                absorption + scattering_opacity(current),
                current.temperature,
            )
        return rosseland_mean_hydrogen_continuum_opacity(
            current, h2_h2_cia_table=h2_h2_cia_table,
            wavelength_angstrom=wavelength,
        )

    def thermodynamics(current: Atmosphere) -> object:
        return hummer_mihalas_hydrogen_thermodynamics(
            current.temperature,
            current.gas_pressure,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=(
                trihydrogen_ion_partition_model
            ),
            central_state=current.hydrogen_lte_state,
        )

    adaptive_seed = with_temperature(temperature)
    return solve_adaptive_lte_structure(
        adaptive_seed,
        wavelength,
        enforce_local_energy_balance=enforce_local_energy_balance,
        with_temperature=with_temperature,
        true_absorption=true_absorption,
        scattering_opacity=scattering_opacity,
        rosseland_opacity=rosseland_opacity,
        thermodynamics=thermodynamics,
        mixing_length_alpha=mixing_length_alpha,
        max_iterations=max_iterations,
        temperature_tolerance=temperature_tolerance,
        flux_tolerance=flux_tolerance,
        n_angle=n_angle,
        initial_temperature_was_supplied=initial_temperature is not None,
        # The DA API historically treats a supplied temperature as a
        # same-physics warm start, not proof of an exactly matching
        # completed checkpoint.  Preserve that contract: it still gets
        # the bounded ML2 conditioner before exact flux completion.
        resume_supplied_structure_in_formal_flux_phase=False,
        # The very efficient convection zones below 5000 K require the
        # interface-based projection recovered from the successful
        # ultracool DA solver.  Preserve the validated node-gradient path
        # at and above 5000 K.
        initial_convective_gradient_projection_mode=(
            "interface-transport"
            if effective_temperature < 5_000.0
            or initial_temperature is not None
            else "unstable-node-gradient"
        ),
        maximum_convective_preconditioner_iterations=max_iterations,
        preconditioner_stationary_completion_iterations=None,
        use_adiabatic_asymptotic_conditioning=(
            effective_temperature < 5_000.0
        ),
        use_initial_bolometric_rescaling=(
            effective_temperature >= 5_000.0
        ),
        iteration_callback=iteration_callback,
        metadata={
            **{
                key: value
                for key, value in seed.metadata.items()
                if key not in ("model", "composition", "eos")
            },
            "composition": (
                "metal-polluted-hydrogen"
                if metal_database is not None
                else "pure-hydrogen"
            ),
            "eos": (
                adaptive_seed.hydrogen_lte_state.chemical_model
                if adaptive_seed.hydrogen_lte_state is not None
                else "hummer-mihalas-occupation-probability"
            ),
            "radiative_equilibrium_seed_tau_min": float(tau_min),
            "radiative_equilibrium_custom_hydrogen_eos": hydrogen_eos_function is not None,
            "radiative_equilibrium_resolves_balmer_line_cores": bool(
                include_balmer_lines and resolve_balmer_line_cores
            ),
            "radiative_equilibrium_resolves_lyman_line_cores": bool(
                include_lyman_lines and resolved_lyman_line_cores
            ),
            "radiative_equilibrium_includes_balmer_lines": bool(
                include_balmer_lines
            ),
            "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                1.0e-4 if include_balmer_lines else None
            ),
            "radiative_equilibrium_includes_paschen_lines": bool(
                include_paschen_lines
            ),
            "radiative_equilibrium_includes_brackett_lines": bool(
                include_brackett_lines
            ),
            "radiative_equilibrium_includes_lyman_lines": bool(
                include_lyman_lines
            ),
            "radiative_equilibrium_includes_series_pseudocontinuum": bool(
                include_lyman_lines and include_series_pseudocontinuum
            ),
            "radiative_equilibrium_includes_molecular_equilibrium_and_opacity": bool(
                include_molecules
            ),
            "radiative_equilibrium_metal_opacity": bool(
                metal_database is not None
            ),
            "metal_abundances": (
                dict(metal_abundances)
                if metal_abundances is not None
                else {}
            ),
            "metal_electron_feedback": (
                "fixed-H-nuclei shared H/metal charge closure"
                if metal_database is not None
                else "disabled"
            ),
            "hydrogen_metal_eos_solver": (
                "depth-local Newton in log(ne) and log(H partition)"
                if metal_database is not None
                else "disabled"
            ),
            "metal_thermodynamic_derivatives": (
                "trace-metal approximation: Q-MHD hydrogen derivatives"
                if metal_database is not None
                else "not applicable"
            ),
            "rosseland_opacity_includes_metal_bound_bound_and_bound_free": (
                explicit_metal_rosseland_opacity
            ),
            "radiative_equilibrium_includes_metal_lines": bool(
                metal_database is not None and include_metal_lines
            ),
            "radiative_equilibrium_includes_metal_bound_free": bool(
                metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            ),
            "radiative_equilibrium_level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                minimum_metal_oscillator_strength
            ),
            "radiative_equilibrium_maximum_metal_lines": (
                maximum_metal_lines
            ),
            "radiative_equilibrium_wing_sampled_metal_lines": int(
                metal_wing_sampled_lines
            ),
            "radiative_equilibrium_metal_line_selection": (
                "abundance-gf-boltzmann-ion-fraction-Planck-flux at Teff"
            ),
            "radiative_equilibrium_includes_negative_hydrogen_charge_equilibrium": bool(
                include_negative_hydrogen
            ),
            "radiative_equilibrium_h3plus_partition_model": (
                trihydrogen_ion_partition_model
            ),
            "radiative_equilibrium_h_h2_lyman_alpha_temperature_dependence": (
                "Sahu-et-al-2025 inverse-temperature correction to "
                "Rohrmann-et-al-2011 6000-K profile"
                if include_molecules and include_lyman_lines
                else None
            ),
        },
    )


def radiative_equilibrium_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    stark_table: object,
    helium_ii_stark_table: object | None = None,
    n_depth: int = 80,
    tau_min: float = 1.0e-8,
    max_iterations: int = 300,
    structure_solver: Literal["adaptive-newton"] = "adaptive-newton",
    temperature_tolerance: float = 3.0e-4,
    flux_tolerance: float = 3.0e-3,
    enforce_local_energy_balance: bool = True,
    n_continuum_wavelength: int = 500,
    include_lines: bool = True,
    include_helium_ii_lines: bool = True,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    mixing_length_alpha: float | None = 1.25,
    n_angle: int = 3,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    neutral_line_broadening: Literal["none", "unsold", "montreal"] = "unsold",
    initial_temperature: FloatArray | None = None,
    initial_column_mass: FloatArray | None = None,
    initial_gas_pressure: FloatArray | None = None,
    initial_rosseland_optical_depth: FloatArray | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    log_hydrogen_abundance: float | None = None,
    molecular_h_he: object | None = None,
    include_trace_hydrogen_lines: bool = True,
    include_hydrogen_self_broadening: bool = True,
    include_hydrogen_neutral_helium_broadening: bool = True,
    hydrogen_self_broadening_quadrature_order: int = 32,
    hydrogen_self_broadening_impact_validity_fraction: float = 1.0,
    hydrogen_self_broadening_prescription: str = "barklem",
    hydrogen_self_broadening_truncation_closure: str = "renormalize",
    include_hydrogen_series_pseudocontinuum: bool = False,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    allard_stark_weight: float = 0.5,
    include_dense_helium_metal_ionization: bool = True,
    include_metal_lines: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-2,
    maximum_metal_lines: int | None = 1_000,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    helium_reos3_table: HeliumREOS3Table | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    resume_supplied_structure_in_formal_flux_phase: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> Atmosphere:
    """Relax an LTE He atmosphere toward non-gray radiative equilibrium.

    This is the helium counterpart of the transparent DA reference solver.
    Hydrostatic equilibrium is exact on a fixed column-mass grid.  The
    released line-dissolved Beauchamp profiles should normally be supplied as
    ``stark_table``.  The default ML2 mixing length of 1.25 is the standard
    Montreal/ATMO DB calibration; pass ``None`` for a radiative control model.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not isinstance(resume_supplied_structure_in_formal_flux_phase, bool):
        raise TypeError(
            "resume_supplied_structure_in_formal_flux_phase must be boolean"
        )
    if structure_solver != "adaptive-newton":
        raise ValueError(
            "Only structure_solver='adaptive-newton' is supported"
        )
    if n_continuum_wavelength < 80:
        raise ValueError("n_continuum_wavelength must be at least 80")
    if mixing_length_alpha is not None and (
        not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be finite and positive")
    if neutral_line_broadening not in ("none", "unsold", "montreal"):
        raise ValueError(
            "neutral_line_broadening must be 'none', 'unsold', or 'montreal'"
        )
    if hydrogen_self_broadening_quadrature_order < 8:
        raise ValueError(
            "hydrogen_self_broadening_quadrature_order must be at least 8"
        )
    if (
        not np.isfinite(allard_stark_weight)
        or not 0.0 <= allard_stark_weight <= 1.0
    ):
        raise ValueError("allard_stark_weight must lie in [0, 1]")
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError("metal photoionization requires metal_database and abundances")
    homogeneous_mixture = (
        log_hydrogen_abundance is not None and metal_database is None
    )
    if molecular_h_he is not None and (
        not homogeneous_mixture or helium_reos3_table is not None
    ):
        raise ValueError("Molecular H/He requires a homogeneous adaptive mixture without bulk-EOS substitution")
    mixed_lte = (hummer_mihalas_hydrogen_helium_lte if molecular_h_he is None
                 else molecular_h_he.lte)
    from .constants import (
        BOLTZMANN,
        ELECTRON_MASS,
        ELEMENTARY_CHARGE_ESU,
        HELIUM_MASS,
        HYDROGEN_IONIZATION_ENERGY,
        LIGHT_SPEED,
        PI,
        PLANCK,
        STEFAN_BOLTZMANN,
    )
    from .helium import HELIUM_I_LINES, HELIUM_II_LINES, _helium_ii_level_distribution, helium_continuum_mass_absorption_coefficient, helium_i_line_mass_absorption_coefficient, helium_i_resonance_line_mass_absorption_coefficient, helium_ii_line_mass_absorption_coefficient, helium_rayleigh_scattering_mass_coefficient, rosseland_mean_helium_continuum_opacity
    if homogeneous_mixture:
        from .mixture import rosseland_mean_hydrogen_helium_continuum_opacity
    from .opacity import BALMER_LINES, BRACKETT_LINES, LYMAN_LINES, PASCHEN_LINES, balmer_mass_absorption_coefficient, brackett_mass_absorption_coefficient, electron_scattering_mass_coefficient, hydrogen_continuum_mass_absorption_coefficient, hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient, hydrogen_rayleigh_scattering_mass_coefficient, lyman_mass_absorption_coefficient, paschen_mass_absorption_coefficient
    from .spectrum import planck_lambda_angstrom
    restart_seed_requested = (
        initial_gas_pressure is not None
        or initial_rosseland_optical_depth is not None
    )
    if restart_seed_requested:
        if any(
            value is None for value in (
                initial_temperature,
                initial_column_mass,
                initial_gas_pressure,
                initial_rosseland_optical_depth,
            )
        ):
            raise ValueError(
                "a checkpoint restart requires temperature, column mass, "
                "gas pressure, and Rosseland optical depth"
            )
        restart_temperature = np.asarray(initial_temperature, dtype=np.float64)
        restart_mass = np.asarray(initial_column_mass, dtype=np.float64)
        restart_pressure = np.asarray(initial_gas_pressure, dtype=np.float64)
        restart_tau = np.asarray(
            initial_rosseland_optical_depth, dtype=np.float64
        )
        restart_shape = restart_temperature.shape
        if (
            restart_temperature.ndim != 1
            or restart_temperature.size < 3
            or any(
                value.shape != restart_shape
                for value in (
                    restart_mass, restart_pressure, restart_tau
                )
            )
        ):
            raise ValueError(
                "checkpoint arrays must be one dimensional and have the "
                "same length"
            )
        if (
            np.any(~np.isfinite(restart_temperature))
            or np.any(restart_temperature <= 0.0)
            or np.any(~np.isfinite(restart_mass))
            or np.any(restart_mass <= 0.0)
            or np.any(np.diff(restart_mass) <= 0.0)
            or np.any(~np.isfinite(restart_pressure))
            or np.any(restart_pressure <= 0.0)
            or np.any(~np.isfinite(restart_tau))
            or np.any(restart_tau <= 0.0)
            or np.any(np.diff(restart_tau) <= 0.0)
        ):
            raise ValueError("checkpoint arrays must be finite, positive, and increasing")
        restart_source_depth = int(restart_temperature.size)
        if restart_source_depth != n_depth:
            # A converged production checkpoint is often a better warm start
            # for a standard-resolution calculation than a lower-resolution
            # atmosphere with the same nominal parameters.  Resample every
            # hydrostatic coordinate together on log Rosseland depth; do not
            # silently discard pressure or column-mass provenance by using
            # only the supplied temperature array.
            source_log_tau = np.log(restart_tau)
            target_log_tau = np.linspace(
                source_log_tau[0], source_log_tau[-1], n_depth
            )
            restart_temperature = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_temperature),
            ))
            restart_mass = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_mass),
            ))
            restart_pressure = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_pressure),
            ))
            restart_tau = np.exp(target_log_tau)
        restart_metadata = {
            "model": "checkpoint-restart-seed",
            "composition": (
                "homogeneous-hydrogen-helium"
                if homogeneous_mixture else "helium"
            ),
            "log_hydrogen_to_helium": (
                float(log_hydrogen_abundance)
                if homogeneous_mixture else None
            ),
            "checkpoint_skips_hydrostatic_seed": True,
            "checkpoint_source_depth_points": restart_source_depth,
            "checkpoint_resampled_to_depth_points": int(n_depth),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
        }
        if homogeneous_mixture:
            assert log_hydrogen_abundance is not None
            mixed_restart_eos = mixed_lte(
                restart_temperature,
                restart_pressure,
                log_hydrogen_abundance,
                helium_neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            seed = _atmosphere_from_hydrogen_helium_state(
                effective_temperature,
                logg,
                restart_tau.copy(),
                restart_mass.copy(),
                restart_temperature.copy(),
                restart_pressure.copy(),
                mixed_restart_eos,
                restart_metadata,
            )
        else:
            restart_eos = (
                hummer_mihalas_helium_lte(
                    restart_temperature,
                    restart_pressure,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
                if helium_reos3_table is None
                else hummer_mihalas_helium_lte_with_reos3(
                    restart_temperature,
                    restart_pressure,
                    helium_reos3_table,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            seed = Atmosphere(
                float(effective_temperature),
                float(logg),
                restart_tau.copy(),
                restart_mass.copy(),
                restart_temperature.copy(),
                restart_pressure.copy(),
                restart_eos.mass_density,
                np.zeros(n_depth),
                np.zeros(n_depth),
                restart_eos.electron_density,
                restart_metadata,
                helium_lte_state=restart_eos,
            )
    else:
        seed = helium_continuum_atmosphere(
            effective_temperature, logg, n_depth=n_depth, tau_min=tau_min,
            correlated_microfields=correlated_microfields,
            neutral_radius_scale=neutral_radius_scale,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
            helium_reos3_table=helium_reos3_table,
            metal_database=metal_database,
            metal_abundances=metal_abundances,
            log_hydrogen_abundance=log_hydrogen_abundance,
            include_dense_helium_metal_ionization=(
                include_dense_helium_metal_ionization
            ),
        )
    relaxation_depth = seed.rosseland_optical_depth.copy()
    structure_helium_i_lines = HELIUM_I_LINES
    structure_helium_ii_lines = (
        HELIUM_II_LINES if include_helium_ii_lines else ()
    )
    if include_lines:
        # Screen structurally irrelevant helium transitions using a strict
        # upper bound on line-centre optical depth above tau_R ~= 1.  Lines
        # below 1e-3 cannot alter a structure solved to 3e-3 flux accuracy;
        # final spectrum synthesis remains unscreened.
        helium_state = seed.helium_lte_state
        if helium_state is not None:
            first_below_photosphere = min(
                int(
                    np.searchsorted(
                        seed.rosseland_optical_depth, 1.0, side="right"
                    )
                )
                + 1,
                seed.n_depth,
            )
            thermal_velocity_fraction = np.sqrt(
                BOLTZMANN * 10_000.0 / HELIUM_MASS
            ) / LIGHT_SPEED
            integrated_cross_section = (
                PI * ELEMENTARY_CHARGE_ESU**2
                / (ELECTRON_MASS * LIGHT_SPEED)
            )
            retained_lines = []
            for line in HELIUM_I_LINES:
                center = line.wavelength_vacuum_angstrom
                center_cm = center * 1.0e-8
                doppler_peak_per_angstrom = 1.0 / (
                    np.sqrt(2.0 * PI)
                    * center
                    * thermal_velocity_fraction
                )
                stimulated = -np.expm1(
                    -PLANCK
                    * LIGHT_SPEED
                    / (center_cm * BOLTZMANN * seed.temperature)
                )
                upper_mass_opacity = (
                    integrated_cross_section
                    * line.absorption_oscillator_strength
                    * helium_state.neutral_level_population_density[
                        :, line.lower_term_index
                    ]
                    * stimulated
                    * doppler_peak_per_angstrom
                    * 1.0e8
                    * center_cm**2
                    / LIGHT_SPEED
                    / seed.mass_density
                )
                upper_optical_depth = trapezoid(
                    upper_mass_opacity[:first_below_photosphere],
                    seed.column_mass[:first_below_photosphere],
                )
                if upper_optical_depth >= 1.0e-3:
                    retained_lines.append(line)
            structure_helium_i_lines = tuple(retained_lines)
            if include_helium_ii_lines:
                maximum_helium_ii_level = max(
                    line.upper_principal_quantum_number
                    for line in HELIUM_II_LINES
                )
                helium_ii_population, _ = _helium_ii_level_distribution(
                    seed, maximum_helium_ii_level
                )
                retained_helium_ii_lines = []
                for line in HELIUM_II_LINES:
                    center = line.wavelength_vacuum_angstrom
                    center_cm = center * 1.0e-8
                    doppler_peak_per_angstrom = 1.0 / (
                        np.sqrt(2.0 * PI)
                        * center
                        * thermal_velocity_fraction
                    )
                    stimulated = -np.expm1(
                        -PLANCK
                        * LIGHT_SPEED
                        / (center_cm * BOLTZMANN * seed.temperature)
                    )
                    upper_mass_opacity = (
                        integrated_cross_section
                        * line.absorption_oscillator_strength
                        * helium_ii_population[
                            :, line.lower_principal_quantum_number - 1
                        ]
                        * stimulated
                        * doppler_peak_per_angstrom
                        * 1.0e8
                        * center_cm**2
                        / LIGHT_SPEED
                        / seed.mass_density
                    )
                    upper_optical_depth = trapezoid(
                        upper_mass_opacity[:first_below_photosphere],
                        seed.column_mass[:first_below_photosphere],
                    )
                    if upper_optical_depth >= 1.0e-3:
                        retained_helium_ii_lines.append(line)
                structure_helium_ii_lines = tuple(retained_helium_ii_lines)
    wavelength = np.geomspace(100.0, 100_000.0, n_continuum_wavelength)
    if include_lines:
        core_offsets = np.arange(-5.0, 5.0001, 0.2)
        line_grids = [
            *(
                line.wavelength_vacuum_angstrom + core_offsets
                for line in structure_helium_i_lines
            )
        ]
        if structure_helium_i_lines:
            line_grids.append(np.arange(2600.0, 7500.1, 10.0))
        uv_resonance_grid = np.arange(480.0, 700.0001, 0.5)
        uv_flux_fraction = float(
            PI
            * trapezoid(
                planck_lambda_angstrom(
                    uv_resonance_grid, effective_temperature
                ),
                uv_resonance_grid,
            )
            / (STEFAN_BOLTZMANN * effective_temperature**4)
        )
        if uv_flux_fraction >= 1.0e-8:
            line_grids.append(uv_resonance_grid)
        if structure_helium_ii_lines:
            line_grids.extend(
                line.wavelength_vacuum_angstrom + core_offsets
                for line in structure_helium_ii_lines
            )
        if line_grids:
            wavelength = np.unique(np.concatenate((wavelength, *line_grids)))
    metal_wing_sampled_lines = 0
    if (
        metal_database is not None
        and metal_abundances is not None
        and include_metal_lines
    ):
        from .metals import metal_lte_state, selected_metal_lines

        selection_state = metal_lte_state(
            seed,
            metal_database,
            metal_abundances,
            log_hydrogen_abundance=log_hydrogen_abundance,
            include_dense_helium_ionization=(
                include_dense_helium_metal_ionization
            ),
        )
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge]
                / np.maximum(
                    selection_state.element_number_density[element],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in selection_state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }

        structure_lines = selected_metal_lines(
            metal_database,
            metal_abundances,
            100.0,
            100_000.0,
            minimum_metal_oscillator_strength,
            maximum_metal_lines,
            effective_temperature,
            ion_stage_weight,
            flux_weighted=True,
        )
        metal_centers = np.asarray(
            [line.wavelength_vacuum_angstrom for _, line in structure_lines]
        )
        if metal_centers.size:
            metal_grid, metal_wing_sampled_lines = (
                _metal_line_opacity_sampling_grid(metal_centers)
            )
            wavelength = np.unique(np.concatenate((wavelength, metal_grid)))
    if log_hydrogen_abundance is not None and include_trace_hydrogen_lines:
        hydrogen_centers = np.asarray([
            line.wavelength_vacuum_angstrom
            for line in (*LYMAN_LINES, *BALMER_LINES, *PASCHEN_LINES, *BRACKETT_LINES)
        ])
        hydrogen_offsets = np.asarray([-5.0, -1.0, 0.0, 1.0, 5.0])
        hydrogen_grid = (
            hydrogen_centers[:, np.newaxis] + hydrogen_offsets
        ).ravel()
        wavelength = np.unique(np.concatenate((wavelength, hydrogen_grid)))
        if homogeneous_mixture:
            # H opacity controls the optically thin temperature in a DBA just
            # as it does in a DA.  Five samples per line miss the narrow cores
            # and severely under-resolve the broad Lyman blanketing.  Match
            # the mature hydrogen solver and explicitly bracket the first
            # four bound-free thresholds so quadrature never interpolates
            # across a discontinuity.
            hydrogen_core_offsets = np.arange(-5.0, 5.0001, 0.2)
            lyman_core_offsets = np.arange(-3.0, 3.0001, 0.05)
            lyman_limit = (
                PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
            )
            series_limits = lyman_limit * np.arange(1.0, 5.0) ** 2
            edge_samples = (
                series_limits[:, np.newaxis]
                * np.asarray((1.0 - 1.0e-4, 1.0 + 1.0e-4))[np.newaxis, :]
            ).ravel()
            wavelength = np.unique(np.concatenate((
                wavelength,
                np.arange(900.0, 1250.1, 1.0),
                np.arange(1255.0, 3000.1, 5.0),
                edge_samples,
                *(
                    line.wavelength_vacuum_angstrom + hydrogen_core_offsets
                    for line in BALMER_LINES[:4]
                ),
                *(
                    line.wavelength_vacuum_angstrom + lyman_core_offsets
                    for line in LYMAN_LINES[:3]
                ),
            )))
    if metal_photoionization_database is not None:
        edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED / (fit.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for fit in metal_photoionization_database.fits.values()
            if fit.element in metal_abundances
        ])
        if edge_grid.size:
            edge_samples = (
                edge_grid[np.newaxis, :]
                * (1.0 + np.asarray([-1.0e-4, 1.0e-4]))[:, np.newaxis]
            ).ravel()
            wavelength = np.unique(np.concatenate((wavelength, edge_samples)))
    if metal_topbase_photoionization_database is not None:
        topbase_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (section.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for section in metal_topbase_photoionization_database.sections
            if section.element in metal_abundances
        ])
        if topbase_edge_grid.size:
            topbase_edge_samples = (
                topbase_edge_grid[np.newaxis, :]
                * (1.0 + np.asarray([-1.0e-4, 1.0e-4]))[:, np.newaxis]
            ).ravel()
            wavelength = np.unique(
                np.concatenate((wavelength, topbase_edge_samples))
            )
    # Broad unified metal profiles carry flux over hundreds of Angstroms.
    # Sample their actual vector grids in the structure solution, rather than
    # relying only on five conventional line-center points.  A deterministic
    # stride caps the transfer cost without discarding satellites or edges.
    unified_profile_grids: list[FloatArray] = []
    if mg_he_red_wing_table is not None:
        for local_wavelength in (
            mg_he_red_wing_table.wavelength_by_temperature.values()
        ):
            step = max(1, local_wavelength.size // 100)
            unified_profile_grids.append(local_wavelength[::step])
        if mg_he_red_wing_table.wavelength_by_density is not None:
            for local_wavelength in (
                mg_he_red_wing_table.wavelength_by_density.values()
            ):
                step = max(1, local_wavelength.size // 100)
                unified_profile_grids.append(local_wavelength[::step])
    if mg_ii_he_profile_table is not None:
        step = max(1, mg_ii_he_profile_table.wavelength_angstrom.size // 400)
        unified_profile_grids.append(mg_ii_he_profile_table.wavelength_angstrom[::step])
    if ca_i_he_profile_table is not None:
        for local_wavelength in ca_i_he_profile_table.wavelength_by_density.values():
            step = max(1, local_wavelength.size // 100)
            unified_profile_grids.append(local_wavelength[::step])
        if ca_i_he_profile_table.wavelength_by_temperature is not None:
            for local_wavelength in (
                ca_i_he_profile_table.wavelength_by_temperature.values()
            ):
                step = max(1, local_wavelength.size // 100)
                unified_profile_grids.append(local_wavelength[::step])
    if ca_ii_he_profile_table is not None:
        unified_profile_grids.append(ca_ii_he_profile_table.wavelength_angstrom)
    if unified_profile_grids:
        wavelength = np.unique(np.concatenate((wavelength, *unified_profile_grids)))
    if initial_temperature is None:
        temperature = seed.temperature.copy()
    else:
        supplied = np.asarray(initial_temperature, dtype=np.float64)
        if np.any(~np.isfinite(supplied)) or np.any(supplied <= 0.0):
            raise ValueError("initial_temperature must contain finite positive values")
        if initial_column_mass is not None:
            mass = np.asarray(initial_column_mass, dtype=np.float64)
            if mass.shape != supplied.shape or np.any(np.diff(mass) <= 0.0):
                raise ValueError("initial_column_mass must increase and match initial_temperature")
            temperature = np.interp(
                np.log(seed.column_mass), np.log(mass), supplied,
                left=supplied[0], right=supplied[-1],
            )
        elif supplied.shape == seed.temperature.shape:
            temperature = supplied.copy()
        else:
            raise ValueError("initial_temperature must have one value per depth")

    def with_temperature(values: FloatArray) -> Atmosphere:
        if homogeneous_mixture:
            assert log_hydrogen_abundance is not None
            mixed_eos = mixed_lte(
                values,
                seed.gas_pressure,
                log_hydrogen_abundance,
                helium_neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            result = _atmosphere_from_hydrogen_helium_state(
                seed.effective_temperature,
                seed.logg,
                relaxation_depth,
                seed.column_mass,
                values,
                seed.gas_pressure,
                mixed_eos,
                seed.metadata,
            )
        else:
            eos = (
                hummer_mihalas_helium_lte(
                    values, seed.gas_pressure,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
                if helium_reos3_table is None
                else hummer_mihalas_helium_lte_with_reos3(
                    values, seed.gas_pressure, helium_reos3_table,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            result = Atmosphere(
                seed.effective_temperature, seed.logg, relaxation_depth,
                seed.column_mass, values, seed.gas_pressure, eos.mass_density,
                np.zeros(n_depth), np.zeros(n_depth), eos.electron_density,
                seed.metadata, helium_lte_state=eos,
            )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                result,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            result = atmosphere_with_metal_electrons(result, metal_state)
        return result

    structure_absorption_cache: dict[str, object] = {}

    def true_absorption(current: Atmosphere) -> FloatArray:
        result = helium_continuum_mass_absorption_coefficient(
            current, wavelength, include_electron_scattering=False,
            include_rayleigh_scattering=False,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )
        if include_lines:
            result += helium_i_line_mass_absorption_coefficient(
                current, wavelength, stark_table,
                lines=structure_helium_i_lines,
                include_occupation_probability=True,
                # This convolution is expensive but is required for flux
                # consistency in cool, dense DB atmospheres.  Callers can
                # explicitly select "none" for fast warm-star diagnostics.
                neutral_broadening=neutral_line_broadening,
            )
            result += helium_i_resonance_line_mass_absorption_coefficient(
                current,
                wavelength,
                include_occupation_probability=True,
            )
            if include_helium_ii_lines:
                result += helium_ii_line_mass_absorption_coefficient(
                    current,
                    wavelength,
                    lines=structure_helium_ii_lines,
                    stark_table=helium_ii_stark_table,
                    include_occupation_probability=True,
                )
        if metal_database is not None and metal_abundances is not None and (
            include_metal_lines
            or metal_photoionization_database is not None
            or metal_topbase_photoionization_database is not None
        ):
            from .metals import (
                metal_bound_free_mass_absorption_coefficient,
                metal_line_mass_absorption_coefficient,
                metal_lte_state,
            )

            metal_state = metal_lte_state(
                current,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            if metal_photoionization_database is not None:
                result += metal_bound_free_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_database,
                    metal_state,
                    metal_photoionization_database,
                    excluded_ions=(
                        ()
                        if metal_topbase_photoionization_database is None
                        else metal_topbase_photoionization_database.ion_stages
                    ),
                )
            if metal_topbase_photoionization_database is not None:
                from .d6 import topbase_bound_free_mass_absorption_coefficient

                result += topbase_bound_free_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_state,
                    metal_topbase_photoionization_database,
                )
            if include_metal_lines:
                result += metal_line_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_database,
                    metal_state,
                    mg_he_red_wing_table=mg_he_red_wing_table,
                    mg_ii_he_profile_table=mg_ii_he_profile_table,
                    ca_i_he_profile_table=ca_i_he_profile_table,
                    ca_ii_he_profile_table=ca_ii_he_profile_table,
                    minimum_oscillator_strength=minimum_metal_oscillator_strength,
                    maximum_lines=maximum_metal_lines,
                )
        if current.hydrogen_lte_state is not None:
            hydrogen_opacity = (hydrogen_continuum_mass_absorption_coefficient
                if molecular_h_he is None else molecular_h_he.hydrogen_opacity)
            result += hydrogen_opacity(
                current,
                wavelength,
                include_electron_scattering=False,
                include_rayleigh_scattering=False,
                include_molecular_absorption=False,
                **({} if molecular_h_he is None else dict(unified_allard_table=unified_allard_table)),
            )
            if include_trace_hydrogen_lines:
                result += balmer_mass_absorption_coefficient(
                    current,
                    wavelength,
                    include_self_broadening=include_hydrogen_self_broadening,
                    include_neutral_helium_broadening=(
                        include_hydrogen_neutral_helium_broadening
                    ),
                    self_broadening_quadrature_order=(
                        hydrogen_self_broadening_quadrature_order
                    ),
                    self_broadening_impact_validity_fraction=(
                        hydrogen_self_broadening_impact_validity_fraction
                    ),
                    self_broadening_prescription=(
                        hydrogen_self_broadening_prescription
                    ),
                    self_broadening_truncation_closure=(
                        hydrogen_self_broadening_truncation_closure
                    ),
                    profile_edge_optical_depth=1.0e-4,
                )
                result += lyman_mass_absorption_coefficient(
                    current,
                    wavelength,
                    unified_allard_table=unified_allard_table,
                    allard_stark_weight=allard_stark_weight,
                )
                result += paschen_mass_absorption_coefficient(current, wavelength)
                result += brackett_mass_absorption_coefficient(current, wavelength)
                if include_hydrogen_series_pseudocontinuum:
                    result += (
                        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                            current, wavelength
                        )
                    )
        structure_absorption_cache.clear()
        structure_absorption_cache.update(
            atmosphere=current,
            absorption=result,
        )
        return result

    from .adaptive_structure import (
        rosseland_mean_from_opacity_grid,
        solve_adaptive_lte_structure,
    )
    from .eos import (
        hummer_mihalas_helium_thermodynamics,
        hummer_mihalas_hydrogen_helium_thermodynamics,
    )

    def scattering_opacity(current: Atmosphere) -> FloatArray:
        scattering = (
            electron_scattering_mass_coefficient(current)[np.newaxis, :]
            + helium_rayleigh_scattering_mass_coefficient(
                current, wavelength
            )
        )
        if current.hydrogen_lte_state is not None:
            scattering += hydrogen_rayleigh_scattering_mass_coefficient(
                current, wavelength
            )
        return scattering

    def rosseland_opacity(current: Atmosphere) -> FloatArray:
        metal_opacity_is_explicit = (
            metal_database is not None
            and (
                include_metal_lines
                or metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            )
        )
        if metal_opacity_is_explicit:
            if structure_absorption_cache.get("atmosphere") is current:
                absorption = np.asarray(
                    structure_absorption_cache["absorption"],
                    dtype=np.float64,
                )
            else:
                absorption = true_absorption(current)
            return rosseland_mean_from_opacity_grid(
                wavelength,
                absorption + scattering_opacity(current),
                current.temperature,
            )
        function = (
            rosseland_mean_hydrogen_helium_continuum_opacity
            if homogeneous_mixture
            else rosseland_mean_helium_continuum_opacity
        )
        return function(
            current,
            n_frequency=120,
            wavelength_angstrom=wavelength,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
            **({} if molecular_h_he is None else dict(molecular_h_he=molecular_h_he)),
        )

    def thermodynamics(current: Atmosphere) -> object:
        if homogeneous_mixture:
            assert log_hydrogen_abundance is not None
            thermo_function = (hummer_mihalas_hydrogen_helium_thermodynamics
                if molecular_h_he is None else molecular_h_he.thermodynamics)
            return thermo_function(
                current.temperature,
                current.gas_pressure,
                log_hydrogen_abundance,
                helium_neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
        return hummer_mihalas_helium_thermodynamics(
            current.temperature,
            current.gas_pressure,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
            helium_reos3_table=helium_reos3_table,
        )

    adaptive_seed = with_temperature(temperature)
    return solve_adaptive_lte_structure(
        adaptive_seed,
        wavelength,
        enforce_local_energy_balance=enforce_local_energy_balance,
        with_temperature=with_temperature,
        true_absorption=true_absorption,
        scattering_opacity=scattering_opacity,
        rosseland_opacity=rosseland_opacity,
        thermodynamics=thermodynamics,
        mixing_length_alpha=mixing_length_alpha,
        max_iterations=max_iterations,
        temperature_tolerance=temperature_tolerance,
        flux_tolerance=flux_tolerance,
        n_angle=n_angle,
        initial_temperature_was_supplied=initial_temperature is not None,
        resume_supplied_structure_in_formal_flux_phase=(
            initial_temperature is not None
            and resume_supplied_structure_in_formal_flux_phase
        ),
        iteration_callback=iteration_callback,
        metadata={
            **{
                key: value
                for key, value in seed.metadata.items()
                if key not in ("model", "composition", "eos")
            },
            "composition": (
                "metal-polluted-helium"
                if metal_database is not None
                else "homogeneous-hydrogen-helium"
                if homogeneous_mixture
                else "pure-helium"
            ),
            "eos": (
                "trace-metal-charge-neutral-q-mhd-helium-occupation-probability"
                if metal_database is not None
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
                if homogeneous_mixture
                else "q-mhd-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-helium-occupation-probability"
            ),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "includes_molecular_equilibrium": molecular_h_he is not None,
            "mixed_chemical_model": ("molecular-h-he-hm" if molecular_h_he is not None else "atomic-h-he-hm"),
            "radiative_equilibrium_includes_helium_lines": bool(
                include_lines
            ),
            "radiative_equilibrium_retained_helium_i_lines": int(
                len(structure_helium_i_lines)
            ),
            "radiative_equilibrium_retained_helium_ii_lines": int(
                len(structure_helium_ii_lines)
            ),
            "radiative_equilibrium_neutral_helium_line_broadening": (
                neutral_line_broadening if include_lines else "disabled"
            ),
            "helium_stark_profiles": (
                "Tremblay-2026/Beauchamp-2025 explicit table"
            ),
            "helium_i_ground_resonance_broadening": (
                "Dimitrijevic-Sahal-Brechot-1989 electron/He-II impact"
            ),
            "helium_ii_profiles": (
                "Schoning-Butler/SYNSPEC table"
                if include_lines
                and include_helium_ii_lines
                and helium_ii_stark_table is not None
                else "hydrogenic Z^-5 transform of unified hydrogen tables"
                if include_lines and include_helium_ii_lines
                else "disabled"
            ),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "radiative_equilibrium_hydrogen_abundance": (
                float(log_hydrogen_abundance)
                if log_hydrogen_abundance is not None
                else None
            ),
            "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                1.0e-4 if include_trace_hydrogen_lines else None
            ),
            "radiative_equilibrium_metal_opacity": bool(
                metal_database is not None
            ),
            "radiative_equilibrium_includes_metal_lines": bool(
                metal_database is not None and include_metal_lines
            ),
            "radiative_equilibrium_includes_metal_bound_free": bool(
                metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            ),
            "radiative_equilibrium_level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                minimum_metal_oscillator_strength
            ),
            "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
            "radiative_equilibrium_wing_sampled_metal_lines": int(
                metal_wing_sampled_lines
            ),
            "radiative_equilibrium_metal_line_selection": (
                "abundance-gf-boltzmann-ion-fraction-Planck-flux at Teff"
            ),
            "metal_abundances": (
                dict(metal_abundances)
                if metal_abundances is not None else {}
            ),
            "metal_electron_feedback": (
                "charge-neutral EOS and continuum opacity"
                if metal_database is not None else "disabled"
            ),
            "metal_thermodynamic_derivatives": (
                "trace-metal approximation: Q-MHD helium derivatives"
                if metal_database is not None else "not applicable"
            ),
            "rosseland_opacity_includes_metal_bound_bound_and_bound_free": (
                metal_database is not None
                and (
                    include_metal_lines
                    or metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                )
            ),
        },
    )


def radiative_equilibrium_hydrogen_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    stark_table: object,
    **kwargs: object,
) -> Atmosphere:
    """Relax a homogeneous warm H/He atmosphere to non-gray equilibrium.

    This named entry point makes the composition convention explicit while
    sharing the mature DB atmosphere iteration.  Keyword options are the same
    as :func:`radiative_equilibrium_helium_atmosphere`.
    """

    if "log_hydrogen_abundance" in kwargs:
        raise TypeError(
            "pass log_hydrogen_to_helium positionally, not "
            "log_hydrogen_abundance"
        )
    kwargs.setdefault("include_hydrogen_series_pseudocontinuum", True)
    # He II contributes less than 1e-12 of the local heating throughout the
    # tested 10 kK mixed structure, yet its tabulated profiles are expensive.
    # Retain it automatically once He ionization becomes structurally relevant.
    kwargs.setdefault("include_helium_ii_lines", effective_temperature >= 15_000.0)
    return radiative_equilibrium_helium_atmosphere(
        effective_temperature,
        logg,
        stark_table=stark_table,
        log_hydrogen_abundance=log_hydrogen_to_helium,
        **kwargs,
    )
