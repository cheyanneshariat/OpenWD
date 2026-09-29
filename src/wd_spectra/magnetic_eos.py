"""Magnetic chemical equilibrium for pure-hydrogen atmospheres.

This module implements the part of the Vera-Rueda & Rohrmann (2020) magnetic
Saha equation that can be reproduced from the public H2db archive:
Landau-quantized electron/proton translation, field-dependent stationary
bound-state energies (with the Vera-Rueda--Rohrmann centered-state fits for
states missing from the archive) and, by default, the centered-motion
transverse-mass factor.  Hummer--Mihalas/Q-MHD occupation probabilities stay
coupled to the partition function and the pressure solve.  H2db energies are
in infinite-mass Rydbergs; they are converted with the reduced-mass hydrogen
Rydberg so that the zero-field limit is exactly the ordinary EOS.

Thermally decentered atoms are not included: no radiative cross sections for
them exist, and populations without matching opacity are not a consistent
improvement.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .atmosphere import Atmosphere
from .constants import (
    BOHR_RADIUS,
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HYDROGEN_IONIZATION_ENERGY,
    HYDROGEN_MASS,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import (
    HM_MAX_BOUND_LEVEL,
    HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    HydrogenLevelDistribution,
    HydrogenLTEState,
    _ionization_fraction_from_saha_ratio,
    hydrogen_level_distribution,
    hydrogen_occupation_probability,
    hydrogen_saha_constant,
    hummer_mihalas_hydrogen_lte,
)
from .magnetic_atomic import (
    H2DB_REFERENCE_FIELD_MEGAGAUSS,
    H2dbEnergyDatabase,
    _longitudinal_quantum_number,
)


# H2db energies are in units of the infinite-mass Rydberg; the ordinary EOS,
# line centers and continua all use the reduced-mass hydrogen value.
_RYDBERG_ENERGY_ERG = HYDROGEN_IONIZATION_ENERGY
_ELECTRON_TO_PROTON_MASS = ELECTRON_MASS / (HYDROGEN_MASS - ELECTRON_MASS)
_ELECTRON_TO_HYDROGEN_MASS = ELECTRON_MASS / HYDROGEN_MASS
_HYDROGEN_MASS_IN_ELECTRON_MASSES = HYDROGEN_MASS / ELECTRON_MASS


def magnetic_landau_saha_factor(
    temperature: ArrayLike, field_strength_megagauss: float
) -> np.ndarray:
    r"""Return the Vera-Rueda--Rohrmann factor ``f(eta)``.

    The magnetic Saha equation contains

    ``f = tanh(eta)/eta * (1-exp(-q eta))/(q eta)``,

    where ``eta = hbar omega_e/(2 k T)`` and ``q=2 m_e/m_p``.  Stable series
    limits are used at zero field.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    field = float(field_strength_megagauss)
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or not np.isfinite(field)
        or field < 0.0
    ):
        raise ValueError("temperature and magnetic field must be physical")
    if field == 0.0:
        return np.ones_like(temperature)
    hbar = PLANCK / (2.0 * PI)
    electron_cyclotron = (
        ELEMENTARY_CHARGE_ESU * field * 1.0e6
        / (ELECTRON_MASS * LIGHT_SPEED)
    )
    eta = hbar * electron_cyclotron / (2.0 * BOLTZMANN * temperature)
    proton_mass = HYDROGEN_MASS - ELECTRON_MASS
    q = 2.0 * ELECTRON_MASS / proton_mass
    first = np.tanh(eta) / eta
    q_eta = q * eta
    second = -np.expm1(-q_eta) / q_eta
    return np.asarray(first * second)


def magnetic_bound_electron_spin_factor(
    temperature: ArrayLike, field_strength_megagauss: float
) -> np.ndarray:
    r"""Return the bound-electron spin sum in the zero-field EOS convention.

    ``hydrogen_saha_constant`` uses the convention in which the two free-
    electron spin states cancel the two neutral-H ground spin states.  In a
    field, the spin-up bound state is raised by one electron cyclotron energy,
    so a spin-down-only H2db sum must be multiplied by
    ``(1 + exp(-hbar*omega_e/kT))/2``.  The factor is exactly unity at zero
    field and approaches one half when the spin-up family is depopulated.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    field = float(field_strength_megagauss)
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or not np.isfinite(field)
        or field < 0.0
    ):
        raise ValueError("temperature and magnetic field must be physical")
    if field == 0.0:
        return np.ones_like(temperature)
    hbar = PLANCK / (2.0 * PI)
    electron_cyclotron = (
        ELEMENTARY_CHARGE_ESU * field * 1.0e6
        / (ELECTRON_MASS * LIGHT_SPEED)
    )
    return 0.5 * (
        1.0
        + np.exp(
            -hbar * electron_cyclotron / (BOLTZMANN * temperature)
        )
    )


def _state_from_longitudinal_quantum_numbers(
    longitudinal_quantum_number: int,
    magnetic_quantum_number: int,
) -> tuple[int, int]:
    """Invert the Vera-Rueda--Rohrmann ``(n,l,m) <-> (nu,m)`` map."""

    longitudinal = int(longitudinal_quantum_number)
    absolute_m = abs(int(magnetic_quantum_number))
    if longitudinal < 0:
        raise ValueError("longitudinal quantum number must be nonnegative")
    if longitudinal % 2 == 0:
        base_principal = math.floor(
            1.0
            + 2.0 * longitudinal
            / (1.0 + math.sqrt(2.0 * longitudinal + 1.0))
            + 1.0e-12
        )
    else:
        base_principal = math.floor(
            2.0
            + (2.0 * longitudinal - 2.0)
            / (1.0 + math.sqrt(2.0 * longitudinal - 1.0))
            + 1.0e-12
        )
    principal = base_principal + absolute_m
    orbital = [
        candidate
        for candidate in range(absolute_m, principal)
        if _longitudinal_quantum_number(
            principal, candidate, absolute_m
        )
        == longitudinal
    ]
    if len(orbital) != 1:
        raise ValueError("longitudinal state does not have a unique Coulomb label")
    return principal, orbital[0]


def centered_transverse_mass_ratio(
    database: H2dbEnergyDatabase,
    principal_quantum_number: int,
    orbital_quantum_number: int,
    magnetic_quantum_number: int,
    field_strength_megagauss: float,
) -> float:
    """Return ``M_perp/M`` from Vera-Rueda & Rohrmann (2020), Eq. 23."""

    field = float(field_strength_megagauss)
    if field == 0.0:
        return 1.0
    principal = int(principal_quantum_number)
    orbital = int(orbital_quantum_number)
    magnetic = int(magnetic_quantum_number)
    longitudinal = _longitudinal_quantum_number(
        principal, orbital, abs(magnetic)
    )

    def energy_at_m(candidate_magnetic: int) -> float:
        candidate_principal, candidate_orbital = (
            _state_from_longitudinal_quantum_numbers(
                longitudinal, candidate_magnetic
            )
        )
        return database.energy_rydberg_with_analytic_fallback(
            candidate_principal,
            candidate_orbital,
            candidate_magnetic,
            -0.5,
            field,
        )

    energy = energy_at_m(magnetic)
    beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
    proton_cyclotron_rydberg = 4.0 * beta * _ELECTRON_TO_PROTON_MASS
    hydrogen_cyclotron_rydberg = 4.0 * beta * _ELECTRON_TO_HYDROGEN_MASS
    denominator_minus = (
        energy_at_m(magnetic - 1) - energy + proton_cyclotron_rydberg
    )
    denominator_plus = (
        energy_at_m(magnetic + 1) - energy - proton_cyclotron_rydberg
    )
    if min(abs(denominator_minus), abs(denominator_plus)) < 1.0e-12:
        return 1.0
    alpha = hydrogen_cyclotron_rydberg * (
        (1.0 - magnetic) / denominator_minus
        - magnetic / denominator_plus
    )
    ratio = 1.0 / (1.0 - alpha)
    # The perturbation expansion itself has failed if it predicts a negative
    # mass.  Such a state should be treated by a non-perturbative energy curve;
    # the present implementation is deliberately restricted to n<=5, where
    # this guard is not reached over the magnetic-white-dwarf field range.
    return float(ratio) if np.isfinite(ratio) and ratio > 0.0 else 1.0


def centered_pseudomomentum_shell_boltzmann_weight(
    temperature: ArrayLike,
    field_strength_megagauss: float,
    database: H2dbEnergyDatabase,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    maximum_centered_level: int = 17,
) -> np.ndarray:
    """Return shell sums with the centered-state ``M_perp/M`` factor.

    This is the finite-pseudomomentum approximation used by Rohrmann (2026)
    in the RWA opacity calculation.  It integrates the quadratic transverse
    kinetic energy analytically, leaving one effective-mass multiplier per
    stationary substate.  Decentered states are deliberately not included.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    field = float(field_strength_megagauss)
    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")
    if not 1 <= maximum_centered_level <= 17:
        raise ValueError("maximum_centered_level must lie in 1..17")
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or not np.isfinite(field)
        or field < 0.0
    ):
        raise ValueError("temperature and field must be physical")
    stationary = database.shell_boltzmann_weight(
        maximum_level, field, temperature
    )
    if field == 0.0:
        return stationary
    output = np.array(stationary, copy=True)
    ground = database.energy_rydberg(1, 0, 0, -0.5, field)
    for principal in range(1, min(maximum_level, maximum_centered_level) + 1):
        shell = np.zeros_like(temperature)
        for orbital in range(principal):
            for magnetic in range(-orbital, orbital + 1):
                energy = database.energy_rydberg_with_analytic_fallback(
                    principal, orbital, magnetic, -0.5, field
                )
                try:
                    mass_ratio = centered_transverse_mass_ratio(
                        database, principal, orbital, magnetic, field
                    )
                except (KeyError, ValueError):
                    # Unity is the controlled stationary-state limit if an
                    # adjacent state needed by the perturbative mass formula
                    # is unavailable or the expansion becomes singular.
                    mass_ratio = 1.0
                shell += mass_ratio * np.exp(
                    -(energy - ground)
                    * _RYDBERG_ENERGY_ERG
                    / (BOLTZMANN * temperature)
                )
        output[..., principal - 1] = shell
    return output


def magnetic_hydrogen_level_distribution(
    neutral_h_density: ArrayLike,
    electron_density: ArrayLike,
    temperature: ArrayLike,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase,
    *,
    include_centered_motion: bool = True,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    correlated_microfields: bool = True,
) -> HydrogenLevelDistribution:
    """Return shell populations for the stationary/centered magnetic EOS."""

    if float(field_strength_megagauss) == 0.0:
        return hydrogen_level_distribution(
            neutral_h_density,
            electron_density,
            temperature,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    neutral, electron, temperature = np.broadcast_arrays(
        np.asarray(neutral_h_density, dtype=np.float64),
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(neutral))
        or np.any(neutral < 0.0)
        or np.any(~np.isfinite(electron))
        or np.any(electron < 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("densities and temperature must be physical")
    level = np.arange(1, maximum_level + 1, dtype=np.float64)
    expanded_level = level.reshape((1,) * temperature.ndim + (-1,))
    occupation = hydrogen_occupation_probability(
        neutral[..., np.newaxis],
        electron[..., np.newaxis],
        temperature[..., np.newaxis],
        expanded_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    if include_centered_motion:
        boltzmann_sum = centered_pseudomomentum_shell_boltzmann_weight(
            temperature,
            field_strength_megagauss,
            energy_database,
            maximum_level=maximum_level,
        )
    else:
        boltzmann_sum = energy_database.shell_boltzmann_weight(
            maximum_level,
            field_strength_megagauss,
            temperature,
        )
    boltzmann_sum *= magnetic_bound_electron_spin_factor(
        temperature, field_strength_megagauss
    )[..., np.newaxis]
    weight = occupation * boltzmann_sum
    partition = np.sum(weight, axis=-1)
    fraction = weight / partition[..., np.newaxis]
    return HydrogenLevelDistribution(
        principal_quantum_number=level,
        occupation_probability=np.asarray(occupation),
        population_fraction=np.asarray(fraction),
        population_density=np.asarray(neutral[..., np.newaxis] * fraction),
        internal_partition_function=np.asarray(partition),
    )


def magnetic_hummer_mihalas_hydrogen_lte(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase,
    *,
    include_centered_motion: bool = True,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    correlated_microfields: bool = True,
) -> HydrogenLTEState:
    """Solve magnetic Saha/HM equilibrium of H, H+ and e- at fixed pressure.

    The zero-field call delegates to the ordinary HM/Q-MHD EOS exactly.
    Molecules and negative ions are absent, as in the published magnetic
    chemical model being reproduced.
    """

    field = float(field_strength_megagauss)
    if field == 0.0:
        return hummer_mihalas_hydrogen_lte(
            temperature,
            gas_pressure,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
            include_molecules=False,
        )
    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(gas_pressure))
        or np.any(gas_pressure <= 0.0)
        or not np.isfinite(field)
        or field < 0.0
    ):
        raise ValueError("temperature, pressure, and magnetic field must be physical")

    particle_pressure_density = gas_pressure / (BOLTZMANN * temperature)
    saha_zero_ground = hydrogen_saha_constant(temperature)
    binding = energy_database.ground_state_binding_energy_erg(field)
    landau = magnetic_landau_saha_factor(temperature, field)
    magnetic_ground_saha = saha_zero_ground * np.exp(
        -(binding - HYDROGEN_IONIZATION_ENERGY) / (BOLTZMANN * temperature)
    ) / landau
    ideal_ionization = np.sqrt(
        magnetic_ground_saha / (particle_pressure_density + magnetic_ground_saha)
    )
    nuclei_density = particle_pressure_density / (1.0 + ideal_ionization)
    ionization = ideal_ionization
    levels = np.arange(1, maximum_level + 1, dtype=np.float64)
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (neutral_radius_scale * BOHR_RADIUS * (levels**2 + 1.0)) ** 3
    )

    def distribution_at(neutral, electron):
        return magnetic_hydrogen_level_distribution(
            neutral,
            electron,
            temperature,
            field,
            energy_database,
            include_centered_motion=include_centered_motion,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )

    converged = False
    for _ in range(200):
        neutral = (1.0 - ionization) * nuclei_density
        electron = ionization * nuclei_density
        distribution = distribution_at(neutral, electron)
        saha = magnetic_ground_saha / distribution.internal_partition_function
        candidate_ionization = _ionization_fraction_from_saha_ratio(
            saha / nuclei_density
        )
        mean_excluded_volume = np.sum(
            distribution.population_fraction * excluded_volume,
            axis=-1,
        )
        quadratic = (
            0.5
            * (1.0 - candidate_ionization) ** 2
            * mean_excluded_volume
        )
        linear = 1.0 + candidate_ionization
        candidate_nuclei = np.where(
            quadratic > np.finfo(np.float64).tiny,
            2.0
            * particle_pressure_density
            / (
                linear
                + np.sqrt(
                    linear**2 + 4.0 * quadratic * particle_pressure_density
                )
            ),
            particle_pressure_density / linear,
        )
        change = np.maximum(
            np.abs(candidate_ionization - ionization),
            np.abs(candidate_nuclei - nuclei_density)
            / np.maximum(candidate_nuclei, np.finfo(np.float64).tiny),
        )
        ionization = 0.55 * ionization + 0.45 * candidate_ionization
        nuclei_density = 0.55 * nuclei_density + 0.45 * candidate_nuclei
        if np.all(change < 2.0e-11):
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "magnetic hydrogen Saha/pressure iteration did not converge "
            f"(maximum change {float(np.max(change)):.3g})"
        )

    neutral = (1.0 - ionization) * nuclei_density
    proton = ionization * nuclei_density
    distribution = distribution_at(neutral, proton)
    return HydrogenLTEState(
        mass_density=np.asarray(HYDROGEN_MASS * nuclei_density),
        hydrogen_nuclei_density=np.asarray(nuclei_density),
        neutral_h_density=np.asarray(neutral),
        proton_density=np.asarray(proton),
        electron_density=np.asarray(proton),
        ionization_fraction=np.asarray(ionization),
        internal_partition_function=distribution.internal_partition_function,
        level_occupation_probability=distribution.occupation_probability,
        level_population_density=distribution.population_density,
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        chemical_model=(
            "magnetic-centered-motion-h-hplus-hm"
            if include_centered_motion
            else "magnetic-stationary-h-hplus-hm"
        ),
        neutral_radius_scale=float(neutral_radius_scale),
    )


def atmosphere_with_magnetic_hydrogen_eos(
    atmosphere: Atmosphere,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase,
    *,
    include_centered_motion: bool = True,
) -> Atmosphere:
    """Remap an atmosphere's fixed ``T,P,m`` structure to the magnetic EOS.

    Temperature, pressure and column mass (hence hydrostatic equilibrium) are
    unchanged.  Densities and level populations are recomputed.  The
    Rosseland depth scale is not recomputed; callers that need a
    self-consistent structure should relax it with the magnetic physics.
    """

    from dataclasses import replace

    state = magnetic_hummer_mihalas_hydrogen_lte(
        atmosphere.temperature,
        atmosphere.gas_pressure,
        field_strength_megagauss,
        energy_database,
        include_centered_motion=include_centered_motion,
    )
    return replace(
        atmosphere,
        mass_density=state.mass_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        hydrogen_lte_state=state,
        metadata={
            **atmosphere.metadata,
            "eos": state.chemical_model,
            "magnetic_eos_field_megagauss": float(field_strength_megagauss),
        },
    )


__all__ = [
    "atmosphere_with_magnetic_hydrogen_eos",
    "magnetic_hummer_mihalas_hydrogen_lte",
    "magnetic_hydrogen_level_distribution",
    "magnetic_bound_electron_spin_factor",
    "centered_pseudomomentum_shell_boltzmann_weight",
    "centered_transverse_mass_ratio",
    "magnetic_landau_saha_factor",
]
