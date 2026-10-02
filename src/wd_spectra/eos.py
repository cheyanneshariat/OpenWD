"""Equations of state used by the atmosphere solver.

The ideal ground-state Saha solution is retained as a compact regression
reference.  The atmosphere calculations use chemical-picture H, He, and
homogeneous atomic H/He equations of state in which the same Hummer--Mihalas
occupation probabilities determine the internal partition functions,
ionization balance, and bound-level populations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dense_eos import HeliumREOS3Table
from .constants import (
    BOLTZMANN,
    BOHR_RADIUS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HELIUM_FIRST_IONIZATION_ENERGY,
    HELIUM_MASS,
    HELIUM_SECOND_IONIZATION_ENERGY,
    HYDROGEN_IONIZATION_ENERGY,
    HYDROGEN_MASS,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .molecules import (
    H2_DISSOCIATION_ENERGY,
    H2_PLUS_DISSOCIATION_ENERGY,
    H3_PLUS_DISSOCIATION_ENERGY,
    H_MINUS_DETACHMENT_ENERGY,
    molecular_hydrogen_dissociation_constant,
    molecular_hydrogen_ion_dissociation_constant,
    molecular_hydrogen_ion_rovibrational_energy,
    molecular_hydrogen_rovibrational_energy,
    negative_hydrogen_ionization_constant,
    trihydrogen_ion_dissociation_constant,
    trihydrogen_ion_rovibrational_energy,
)


FloatArray = NDArray[np.float64]
HM_MAX_BOUND_LEVEL = 40
# Effective radii are a model closure in the HM hard-sphere term.  Keep the
# direct hydrogenic choice r_n = n^2 a_0 as the grid-wide default.  The common
# Montreal/ATMO r_B=0.5 calibration was screened against the Koester grid: it
# helps selected 6000 K lines, but degrades Hgamma at 8000 K once the structure
# is relaxed.  It is therefore not hidden in the default EOS.
HM_NEUTRAL_HYDROGEN_RADIUS_SCALE = 1.0
HM_HELIUM_NEUTRAL_RADIUS_SCALE = 0.5


@dataclass(frozen=True)
class HydrogenLTEState:
    """Thermodynamic state of partially ionized pure hydrogen.

    Number densities are in cm^-3 and mass density is in g cm^-3.
    ``hydrogen_nuclei_density`` counts nuclei in every atomic and molecular
    species.  Molecular densities are ``None`` for the atomic-only control
    EOS retained for reproducible validation.
    """

    mass_density: FloatArray
    hydrogen_nuclei_density: FloatArray
    neutral_h_density: FloatArray
    proton_density: FloatArray
    electron_density: FloatArray
    ionization_fraction: FloatArray
    internal_partition_function: FloatArray | None = None
    level_occupation_probability: FloatArray | None = None
    level_population_density: FloatArray | None = None
    microfield_model: str = "holtsmark"
    molecular_hydrogen_density: FloatArray | None = None
    molecular_hydrogen_ion_density: FloatArray | None = None
    negative_hydrogen_density: FloatArray | None = None
    trihydrogen_ion_density: FloatArray | None = None
    trihydrogen_ion_partition_model: str | None = None
    chemical_model: str = "atomic-hm"
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE


@dataclass(frozen=True)
class HydrogenLevelDistribution:
    """Occupation probabilities and LTE populations of bound H I levels.

    The final array axis corresponds to principal quantum numbers starting at
    one.  ``internal_partition_function`` uses statistical weights relative
    to the ground-state weight, so its ideal low-temperature limit is unity.
    """

    principal_quantum_number: FloatArray
    occupation_probability: FloatArray
    population_fraction: FloatArray
    population_density: FloatArray
    internal_partition_function: FloatArray


@dataclass(frozen=True)
class HydrogenThermodynamics:
    r"""Thermodynamic derivatives for ionizing pure hydrogen.

    ``density_temperature_derivative`` is the conventional MLT quantity
    :math:`Q = -(\partial \ln \rho/\partial \ln T)_P`.  The specific heat is
    per gram and uses cgs units.
    """

    specific_heat_constant_pressure: FloatArray
    density_temperature_derivative: FloatArray
    adiabatic_temperature_gradient: FloatArray


@dataclass(frozen=True)
class HeliumLTEState:
    """Thermodynamic state of pure helium in LTE.

    Number densities are in cm^-3 and mass density is in g cm^-3.  The
    neutral-level arrays contain the He I ground state followed by the
    2s triplet, 2s singlet, 2p triplet, and 2p singlet terms.  These are the
    lower terms of the optical lines used by the DB spectrum synthesizer.
    """

    mass_density: FloatArray
    helium_nuclei_density: FloatArray
    neutral_he_density: FloatArray
    singly_ionized_he_density: FloatArray
    doubly_ionized_he_density: FloatArray
    electron_density: FloatArray
    mean_ion_charge: FloatArray
    neutral_partition_function: FloatArray
    singly_ionized_partition_function: FloatArray
    neutral_level_occupation_probability: FloatArray
    neutral_level_population_density: FloatArray
    microfield_model: str = "qmhd"
    neutral_radius_scale: float = 0.5


@dataclass(frozen=True)
class HeliumThermodynamics:
    r"""Thermodynamic derivatives for ionizing pure helium."""

    specific_heat_constant_pressure: FloatArray
    density_temperature_derivative: FloatArray
    adiabatic_temperature_gradient: FloatArray


@dataclass(frozen=True)
class HydrogenHeliumLTEState:
    """Self-consistent homogeneous atomic H/He state in LTE.

    The component states share one electron density and use the same neutral
    perturber populations in their Hummer--Mihalas occupation probabilities.
    ``mass_density`` includes both nuclei; the component ``mass_density``
    fields retain the mass belonging to that element alone.
    """

    mass_density: FloatArray
    hydrogen_nuclei_density: FloatArray
    helium_nuclei_density: FloatArray
    electron_density: FloatArray
    hydrogen_to_helium_number_ratio: FloatArray
    hydrogen_lte_state: HydrogenLTEState
    helium_lte_state: HeliumLTEState
    microfield_model: str = "qmhd"
    chemical_model: str = "atomic-h-he-hm"


@dataclass(frozen=True)
class HydrogenHeliumThermodynamics:
    r"""Thermodynamic derivatives for a homogeneous atomic H/He mixture."""

    specific_heat_constant_pressure: FloatArray
    density_temperature_derivative: FloatArray
    adiabatic_temperature_gradient: FloatArray


def hydrogen_saha_constant(temperature: ArrayLike) -> FloatArray:
    """Return ``n_p n_e / n_H`` for ground-state ideal hydrogen in cm^-3.

    Electron spin and the ground-state degeneracy of neutral hydrogen cancel
    in this convention. Excited levels and pressure ionization are not yet
    included.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must contain finite positive values")

    translational = (
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    ) ** 1.5
    return translational * np.exp(
        -HYDROGEN_IONIZATION_ENERGY / (BOLTZMANN * temperature)
    )


def hydrogenic_critical_microfield_beta(
    perturber_density: ArrayLike,
    effective_principal_quantum_number: ArrayLike,
    ionic_charge: float = 1.0,
) -> FloatArray:
    """Hummer--Mihalas critical microfield of level ``n`` in normal-field units.

    Fields above ``beta_critical`` times ``F0 = (4 pi N / 3)^(2/3) e`` (the
    Holtsmark normal field) dissolve the level; the Q-MHD occupation
    probability is the fraction of microfields below it.  ``perturber_density``
    must be positive.
    """

    density = np.asarray(perturber_density, dtype=np.float64)
    level = np.asarray(effective_principal_quantum_number, dtype=np.float64)
    correction = np.where(
        level <= 3.0,
        1.0,
        16.0 * level / (3.0 * (level + 1.0) ** 2),
    )
    binding_energy = HYDROGEN_IONIZATION_ENERGY / level**2
    return (
        (3.0 / (4.0 * PI)) ** (2.0 / 3.0)
        * correction
        * binding_energy**2
        / (4.0 * ELEMENTARY_CHARGE_ESU**4)
        * ionic_charge**3
        * density ** (-2.0 / 3.0)
    )


def hooper_microfield_cumulative_probability(
    beta: ArrayLike,
    correlation: ArrayLike,
    ionic_charge: float = 1.0,
) -> FloatArray:
    """Fraction of microfields below ``beta`` normal-field units (Hooper fit).

    This is the Nayfonov et al. (1999) rational fit to Hooper's correlated
    microfield distribution used by the Q-MHD occupation probability,
    ``x / (1 + x)`` with ``x = C1 beta^3 / (1 + C2 beta^1.5)``.  At zero
    correlation it reproduces the Holtsmark limits (``4 beta^3 / 9 pi`` at
    weak fields, ``1 - O(beta^-1.5)`` at strong fields).  Its derivative is
    the field distribution consistent with the occupation probabilities.
    """

    beta = np.asarray(beta, dtype=np.float64)
    correlation = np.asarray(correlation, dtype=np.float64)
    correlation_factor = (1.0 + correlation) ** 3.15
    coefficient_1 = 0.1402 * (
        correlation_factor
        + 4.0 * (ionic_charge - 1.0) * correlation**3
    )
    coefficient_2 = 0.1285 * correlation_factor
    # Evaluate the Hooper rational fit in log space. Charge-neutrality
    # bisections deliberately probe extremely small trial densities, where
    # beta is enormous and the direct beta**3 expression overflows
    # even though the physical probability simply tends to one.
    log_beta = np.log(np.maximum(beta, np.finfo(np.float64).tiny))
    log_ratio = (
        np.log(coefficient_1)
        + 3.0 * log_beta
        - np.logaddexp(0.0, np.log(coefficient_2) + 1.5 * log_beta)
    )
    # Both logistic branches need the same decaying exponential. Reuse it
    # without changing the original +/-745 clamps or either division; this
    # also avoids overflow in the unused branch of np.where.
    exponential = np.exp(-np.minimum(np.abs(log_ratio), 745.0))
    denominator = 1.0 + exponential
    probability = np.where(
        log_ratio >= 0.0, 1.0 / denominator, exponential / denominator
    )
    return probability


def charged_particle_hydrogen_occupation_probability(
    electron_density: ArrayLike,
    effective_principal_quantum_number: ArrayLike,
    temperature: ArrayLike | None = None,
    *,
    ionic_charge: float = 1.0,
) -> FloatArray:
    """Q-MHD occupation probability for charged perturbers.

    This evaluates the analytic Hooper microfield fit of Nayfonov et al.
    (1999) for a neutral radiator.  Its correlation parameter is
    ``a = 0.09 ne**(1/6) / sqrt(T)`` in cgs units and is clipped to the
    published fit domain ``a <= 0.8``.  ``ionic_charge`` is the nuclear charge
    of a hydrogenic radiator (one for H I, two for He II).  Omitting
    ``temperature`` selects ``a=0``, the original MHD/Holtsmark limit.  The
    normalization and charge-dependent correlation term follow TLUSTY's
    modern ``WN`` convention: BERGFC=1 (reference manual II, 2017,
    eqs. 107--108). The historical empirical doubling of the critical field
    is not applied, consistently with the non-ideal Tremblay--Bergeron
    hydrogen line profiles.
    """

    electron_density = np.asarray(electron_density, dtype=np.float64)
    level = np.asarray(effective_principal_quantum_number, dtype=np.float64)
    if temperature is None:
        electron_density, level = np.broadcast_arrays(electron_density, level)
        correlation = np.zeros_like(electron_density)
    else:
        electron_density, level, temperature = np.broadcast_arrays(
            electron_density,
            level,
            np.asarray(temperature, dtype=np.float64),
        )
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature must be finite and positive")
        correlation = np.clip(
            0.09 * electron_density ** (1.0 / 6.0) / np.sqrt(temperature),
            0.0,
            0.8,
        )
    if (
        np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(level))
        or np.any(level < 1.0)
        or not np.isfinite(ionic_charge)
        or ionic_charge < 1.0
    ):
        raise ValueError(
            "electron density must be non-negative and principal quantum "
            "number and ionic charge must be finite and positive"
        )

    positive_density = electron_density > 0.0
    beta_critical = hydrogenic_critical_microfield_beta(
        np.where(positive_density, electron_density, 1.0), level, ionic_charge
    )
    probability = hooper_microfield_cumulative_probability(
        beta_critical, correlation, ionic_charge
    )
    return np.where(positive_density, probability, 1.0)


def hydrogen_occupation_probability(
    neutral_h_density: ArrayLike,
    electron_density: ArrayLike,
    temperature: ArrayLike,
    effective_principal_quantum_number: ArrayLike,
    *,
    neutral_he_density: ArrayLike = 0.0,
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    helium_neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = True,
) -> FloatArray:
    """Combined charged- and neutral-perturber HM occupation probability.

    ``neutral_he_density`` supplies the cross-species excluded-volume term
    needed by a homogeneous H/He atmosphere.  It defaults to zero so the
    established pure-H result is unchanged.
    """

    neutral_h_density, neutral_he_density, electron_density, temperature, level = (
        np.broadcast_arrays(
            np.asarray(neutral_h_density, dtype=np.float64),
            np.asarray(neutral_he_density, dtype=np.float64),
            np.asarray(electron_density, dtype=np.float64),
            np.asarray(temperature, dtype=np.float64),
            np.asarray(effective_principal_quantum_number, dtype=np.float64),
        )
    )
    if (
        np.any(~np.isfinite(neutral_h_density))
        or np.any(neutral_h_density < 0.0)
        or np.any(~np.isfinite(neutral_he_density))
        or np.any(neutral_he_density < 0.0)
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(level))
        or np.any(level < 1.0)
        or not np.isfinite(neutral_radius_scale)
        or neutral_radius_scale <= 0.0
        or not np.isfinite(helium_neutral_radius_scale)
        or helium_neutral_radius_scale <= 0.0
    ):
        raise ValueError("densities, temperature, and level must be physical")

    charged_probability = charged_particle_hydrogen_occupation_probability(
        electron_density,
        level,
        temperature if correlated_microfields else None,
    )
    level_radius = neutral_radius_scale * BOHR_RADIUS * level**2
    hydrogen_ground_radius = neutral_radius_scale * BOHR_RADIUS
    helium_ground_radius = helium_neutral_radius_scale * BOHR_RADIUS
    hydrogen_excluded_volume = (
        4.0 / 3.0 * PI * (level_radius + hydrogen_ground_radius) ** 3
    )
    helium_excluded_volume = (
        4.0 / 3.0 * PI * (level_radius + helium_ground_radius) ** 3
    )
    return charged_probability * np.exp(
        -neutral_h_density * hydrogen_excluded_volume
        -neutral_he_density * helium_excluded_volume
    )


def hydrogen_level_distribution(
    neutral_h_density: ArrayLike,
    electron_density: ArrayLike,
    temperature: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_he_density: ArrayLike = 0.0,
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    helium_neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = False,
) -> HydrogenLevelDistribution:
    """Return HM occupation probabilities and normalized H I populations.

    Charged-particle dissolution uses the Holtsmark MHD microfield integral.
    Neutral-particle dissolution uses the low-density hard-sphere term of
    HM88 with the direct hydrogenic interaction radius ``r_n = n^2 a_0``.
    Ground-state H dominates the neutral perturber population throughout the
    current DA grid, so the neutral term uses that controlled approximation
    while retaining all target levels in the partition sum.
    """

    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")
    neutral_h_density, neutral_he_density, electron_density, temperature = np.broadcast_arrays(
        np.asarray(neutral_h_density, dtype=np.float64),
        np.asarray(neutral_he_density, dtype=np.float64),
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(neutral_h_density))
        or np.any(neutral_h_density < 0.0)
        or np.any(~np.isfinite(neutral_he_density))
        or np.any(neutral_he_density < 0.0)
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("densities and temperature must be physical")

    level = np.arange(1, maximum_level + 1, dtype=np.float64)
    expanded_level = level.reshape((1,) * temperature.ndim + (maximum_level,))
    expanded_neutral = neutral_h_density[..., np.newaxis]
    expanded_neutral_he = neutral_he_density[..., np.newaxis]
    expanded_electron = electron_density[..., np.newaxis]
    expanded_temperature = temperature[..., np.newaxis]

    occupation_probability = hydrogen_occupation_probability(
        expanded_neutral,
        expanded_electron,
        expanded_temperature,
        expanded_level,
        neutral_he_density=expanded_neutral_he,
        neutral_radius_scale=neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )

    excitation_energy = HYDROGEN_IONIZATION_ENERGY * (
        1.0 - 1.0 / expanded_level**2
    )
    statistical_weight = expanded_level**2
    log_weight = (
        2.0 * np.log(expanded_level)
        + np.log(np.maximum(occupation_probability, np.finfo(np.float64).tiny))
        - excitation_energy / (BOLTZMANN * expanded_temperature)
    )
    maximum_log_weight = np.max(log_weight, axis=-1, keepdims=True)
    scaled_weight = np.exp(log_weight - maximum_log_weight)
    scaled_partition = np.sum(scaled_weight, axis=-1)
    population_fraction = scaled_weight / scaled_partition[..., np.newaxis]
    partition = (
        np.exp(maximum_log_weight[..., 0]) * scaled_partition
    )
    return HydrogenLevelDistribution(
        principal_quantum_number=level,
        occupation_probability=np.asarray(occupation_probability),
        population_fraction=np.asarray(population_fraction),
        population_density=np.asarray(
            expanded_neutral * population_fraction
        ),
        internal_partition_function=np.asarray(partition),
    )


def _ionization_fraction_from_saha_ratio(saha_over_nuclei: FloatArray) -> FloatArray:
    """Solve ``x^2 / (1-x) = S/N`` without cancellation."""

    positive = saha_over_nuclei > 0.0
    safe = np.where(positive, saha_over_nuclei, 1.0)
    ionization = 2.0 / (np.sqrt(1.0 + 4.0 / safe) + 1.0)
    return np.where(positive, ionization, 0.0)


def _molecular_hydrogen_pressure_root(
    particle_pressure_density: FloatArray,
    saha: FloatArray,
    h2_dissociation: FloatArray,
    h2plus_dissociation: FloatArray,
    mean_excluded_volume: FloatArray,
    initial_neutral_density: FloatArray,
    hminus_ionization: FloatArray | None = None,
    h3plus_dissociation: FloatArray | None = None,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """Solve the fixed-partition molecular pressure equation.

    The equilibrium densities are analytic functions of ``n(H)``.  A
    safeguarded Newton iteration in ``ln n(H)`` replaces the former 72-step
    bisection.  The pressure residual is monotonic, and the maintained bracket
    makes this path just as robust while normally requiring fewer than ten
    iterations.
    """

    tiny = np.finfo(np.float64).tiny
    pressure_density = np.asarray(
        particle_pressure_density, dtype=np.float64
    )
    density_cap = np.maximum(10.0 * pressure_density, 1.0)
    log_density_cap = np.log(density_cap)
    log_saha = np.log(np.maximum(saha, tiny))
    log_h2 = np.log(np.maximum(h2_dissociation, tiny))
    log_lower = np.log(np.maximum(pressure_density * 1.0e-30, tiny))
    log_upper = np.log(np.maximum(2.0 * pressure_density, 1.0))
    log_atom = np.clip(
        np.log(np.maximum(initial_neutral_density, tiny)),
        log_lower,
        log_upper,
    )

    for _ in range(40):
        atom = np.exp(log_atom)
        molecule = np.exp(
            np.minimum(2.0 * log_atom - log_h2, log_density_cap)
        )
        log_h2plus_ratio = log_atom - np.log(
            np.maximum(h2plus_dissociation, tiny)
        )
        log_h3plus_ratio = (
            np.full_like(atom, -np.inf)
            if h3plus_dissociation is None
            else np.log(np.maximum(molecule, tiny))
            - np.log(np.maximum(h3plus_dissociation, tiny))
        )
        log_hminus_ratio = (
            np.full_like(atom, -np.inf)
            if hminus_ionization is None
            else log_atom - np.log(np.maximum(hminus_ionization, tiny))
        )
        log_positive_charge_factor = np.logaddexp(
            np.logaddexp(0.0, log_h2plus_ratio), log_h3plus_ratio
        )
        log_negative_charge_factor = np.logaddexp(0.0, log_hminus_ratio)
        log_proton = 0.5 * (
            log_saha
            + log_atom
            + log_negative_charge_factor
            - log_positive_charge_factor
        )
        proton = np.exp(np.minimum(log_proton, log_density_cap))
        molecular_ion = np.exp(
            np.minimum(log_proton + log_h2plus_ratio, log_density_cap)
        )
        trihydrogen_ion = np.exp(
            np.minimum(log_proton + log_h3plus_ratio, log_density_cap)
        )
        electron = np.exp(
            np.minimum(
                log_proton
                + log_positive_charge_factor
                - log_negative_charge_factor,
                log_density_cap,
            )
        )
        negative_hydrogen = np.exp(
            np.minimum(
                np.log(np.maximum(electron, tiny)) + log_hminus_ratio,
                log_density_cap,
            )
        )
        virial = 0.5 * atom**2 * mean_excluded_volume
        residual = (
            atom
            + proton
            + electron
            + molecular_ion
            + trihydrogen_ion
            + negative_hydrogen
            + molecule
            + virial
            - pressure_density
        )
        log_upper = np.where(residual > 0.0, log_atom, log_upper)
        log_lower = np.where(residual > 0.0, log_lower, log_atom)
        if np.all(
            np.abs(residual) / np.maximum(pressure_density, 1.0)
            < 2.0e-13
        ):
            break

        positive_log_derivative = (
            np.exp(log_h2plus_ratio - log_positive_charge_factor)
            + 2.0 * np.exp(log_h3plus_ratio - log_positive_charge_factor)
        )
        negative_log_derivative = np.exp(
            log_hminus_ratio - log_negative_charge_factor
        )
        proton_log_derivative = 0.5 * (
            1.0
            + negative_log_derivative
            - positive_log_derivative
        )
        electron_log_derivative = 0.5 * (
            1.0
            + positive_log_derivative
            - negative_log_derivative
        )
        derivative = (
            atom
            + proton * proton_log_derivative
            + electron * electron_log_derivative
            + molecular_ion * (1.0 + proton_log_derivative)
            + trihydrogen_ion * (2.0 + proton_log_derivative)
            + negative_hydrogen * (1.0 + electron_log_derivative)
            + 2.0 * molecule
            + 2.0 * virial
        )
        newton = log_atom - residual / np.maximum(derivative, tiny)
        midpoint = 0.5 * (log_lower + log_upper)
        usable = (
            np.isfinite(newton)
            & (newton > log_lower)
            & (newton < log_upper)
        )
        log_atom = np.where(usable, newton, midpoint)

    atom = np.exp(log_atom)
    molecule = np.exp(
        np.minimum(2.0 * log_atom - log_h2, log_density_cap)
    )
    log_h2plus_ratio = log_atom - np.log(
        np.maximum(h2plus_dissociation, tiny)
    )
    log_h3plus_ratio = (
        np.full_like(atom, -np.inf)
        if h3plus_dissociation is None
        else np.log(np.maximum(molecule, tiny))
        - np.log(np.maximum(h3plus_dissociation, tiny))
    )
    log_hminus_ratio = (
        np.full_like(atom, -np.inf)
        if hminus_ionization is None
        else log_atom - np.log(np.maximum(hminus_ionization, tiny))
    )
    log_positive_charge_factor = np.logaddexp(
        np.logaddexp(0.0, log_h2plus_ratio), log_h3plus_ratio
    )
    log_negative_charge_factor = np.logaddexp(0.0, log_hminus_ratio)
    log_proton = 0.5 * (
        log_saha
        + log_atom
        + log_negative_charge_factor
        - log_positive_charge_factor
    )
    proton = np.exp(np.minimum(log_proton, log_density_cap))
    molecular_ion = np.exp(
        np.minimum(log_proton + log_h2plus_ratio, log_density_cap)
    )
    trihydrogen_ion = np.exp(
        np.minimum(log_proton + log_h3plus_ratio, log_density_cap)
    )
    log_electron = (
        log_proton + log_positive_charge_factor - log_negative_charge_factor
    )
    electron = np.exp(np.minimum(log_electron, log_density_cap))
    negative_hydrogen = np.exp(
        np.minimum(log_electron + log_hminus_ratio, log_density_cap)
    )
    return (
        atom,
        proton,
        electron,
        molecule,
        molecular_ion,
        negative_hydrogen,
        trihydrogen_ion,
    )


def hummer_mihalas_hydrogen_lte(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
) -> HydrogenLTEState:
    """Solve a pure-H occupation-probability EOS at fixed gas pressure.

    The modified internal partition function is used inside the Saha balance,
    rather than applying occupation probabilities after solving an ideal EOS.
    The pressure includes the second-virial neutral hard-sphere term that
    generates the adopted neutral occupation probability.  With
    ``include_molecules=True``, the coupled equilibria H <-> H+ + e-,
    2H <-> H2, and H + H+ <-> H2+ are solved together with charge neutrality,
    nuclei conservation, and the gas-pressure constraint.  The optional
    negative-H and H3+ terms add H + e <-> H- and H2 + H+ <-> H3+ to the same
    nonlinear solve.  Electron degeneracy and Debye--Huckel Coulomb pressure
    remain outside this implementation.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(gas_pressure))
        or np.any(gas_pressure <= 0.0)
    ):
        raise ValueError("temperature and gas_pressure must be finite and positive")
    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")
    if not np.isfinite(neutral_radius_scale) or neutral_radius_scale <= 0.0:
        raise ValueError("neutral_radius_scale must be finite and positive")

    particle_pressure_density = gas_pressure / (BOLTZMANN * temperature)
    saha_ground = hydrogen_saha_constant(temperature)
    ideal_ionization = np.sqrt(
        saha_ground / (particle_pressure_density + saha_ground)
    )
    nuclei_density = particle_pressure_density / (1.0 + ideal_ionization)
    ionization = ideal_ionization
    levels = np.arange(1, maximum_level + 1, dtype=np.float64)
    excluded_volume = (
        4.0
        / 3.0
        * PI
        * (
            neutral_radius_scale
            * BOHR_RADIUS
            * (levels**2 + 1.0)
        )
        ** 3
    )

    for _ in range(48):
        neutral_density = (1.0 - ionization) * nuclei_density
        electron_density = ionization * nuclei_density
        distribution = hydrogen_level_distribution(
            neutral_density,
            electron_density,
            temperature,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        saha = np.exp(
            np.clip(
                np.log(np.maximum(saha_ground, np.finfo(np.float64).tiny))
                - np.log(
                    np.maximum(
                        distribution.internal_partition_function,
                        np.finfo(np.float64).tiny,
                    )
                ),
                np.log(np.finfo(np.float64).tiny),
                np.log(np.finfo(np.float64).max),
            )
        )
        candidate_ionization = _ionization_fraction_from_saha_ratio(
            saha / nuclei_density
        )
        mean_excluded_volume = np.sum(
            distribution.population_fraction * excluded_volume, axis=-1
        )
        quadratic = 0.5 * (1.0 - candidate_ionization) ** 2 * mean_excluded_volume
        linear = 1.0 + candidate_ionization
        candidate_nuclei = np.where(
            quadratic > np.finfo(np.float64).tiny,
            2.0
            * particle_pressure_density
            / (
                linear
                + np.sqrt(
                    linear**2
                    + 4.0 * quadratic * particle_pressure_density
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
            break

    neutral_density = (1.0 - ionization) * nuclei_density
    proton_density = ionization * nuclei_density
    electron_density = proton_density
    molecular_density: FloatArray | None = None
    molecular_ion_density: FloatArray | None = None
    negative_hydrogen_density: FloatArray | None = None
    trihydrogen_ion_density: FloatArray | None = None
    if include_molecules:
        # At fixed atomic HM partition and excluded volume, all five species
        # are functions of n(H).  The pressure residual is monotonic, so a
        # vectorized logarithmic bisection provides a robust inner solve.  The
        # outer iteration updates the density-dependent occupation
        # probabilities and therefore all three equilibrium constants.
        tiny = np.finfo(np.float64).tiny
        molecular_density = np.zeros_like(neutral_density)
        molecular_ion_density = np.zeros_like(neutral_density)
        negative_hydrogen_density = np.zeros_like(neutral_density)
        trihydrogen_ion_density = np.zeros_like(neutral_density)
        for _ in range(64):
            distribution = hydrogen_level_distribution(
                neutral_density,
                electron_density,
                temperature,
                maximum_level=maximum_level,
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            atomic_partition = distribution.internal_partition_function
            saha = np.exp(
                np.clip(
                    np.log(np.maximum(saha_ground, tiny))
                    - np.log(np.maximum(atomic_partition, tiny)),
                    np.log(tiny),
                    np.log(np.finfo(np.float64).max),
                )
            )
            h2_dissociation = molecular_hydrogen_dissociation_constant(
                temperature,
                atomic_internal_partition_function=atomic_partition,
            )
            h2plus_dissociation = (
                molecular_hydrogen_ion_dissociation_constant(
                    temperature,
                    atomic_internal_partition_function=atomic_partition,
                )
            )
            hminus_ionization = (
                negative_hydrogen_ionization_constant(
                    temperature,
                    atomic_internal_partition_function=atomic_partition,
                )
                if include_negative_hydrogen
                else None
            )
            h3plus_dissociation = (
                trihydrogen_ion_dissociation_constant(
                    temperature,
                    partition_model=trihydrogen_ion_partition_model,
                )
                if trihydrogen_ion_partition_model is not None
                else None
            )
            mean_excluded_volume = np.sum(
                distribution.population_fraction * excluded_volume, axis=-1
            )

            (
                candidate_neutral,
                candidate_proton,
                candidate_electron,
                candidate_molecular,
                candidate_molecular_ion,
                candidate_negative_hydrogen,
                candidate_trihydrogen_ion,
            ) = _molecular_hydrogen_pressure_root(
                particle_pressure_density,
                saha,
                h2_dissociation,
                h2plus_dissociation,
                mean_excluded_volume,
                neutral_density,
                hminus_ionization=hminus_ionization,
                h3plus_dissociation=h3plus_dissociation,
            )
            scale = np.maximum(particle_pressure_density, 1.0)
            change = np.maximum.reduce(
                (
                    np.abs(candidate_neutral - neutral_density) / scale,
                    np.abs(candidate_proton - proton_density) / scale,
                    np.abs(candidate_electron - electron_density) / scale,
                    np.abs(candidate_molecular - molecular_density) / scale,
                    np.abs(candidate_molecular_ion - molecular_ion_density)
                    / scale,
                    np.abs(
                        candidate_negative_hydrogen
                        - negative_hydrogen_density
                    ) / scale,
                    np.abs(
                        candidate_trihydrogen_ion
                        - trihydrogen_ion_density
                    ) / scale,
                )
            )
            neutral_density = 0.45 * candidate_neutral + 0.55 * neutral_density
            proton_density = 0.45 * candidate_proton + 0.55 * proton_density
            electron_density = 0.45 * candidate_electron + 0.55 * electron_density
            molecular_density = (
                0.45 * candidate_molecular + 0.55 * molecular_density
            )
            molecular_ion_density = (
                0.45 * candidate_molecular_ion
                + 0.55 * molecular_ion_density
            )
            negative_hydrogen_density = (
                0.45 * candidate_negative_hydrogen
                + 0.55 * negative_hydrogen_density
            )
            trihydrogen_ion_density = (
                0.45 * candidate_trihydrogen_ion
                + 0.55 * trihydrogen_ion_density
            )
            if np.all(change < 2.0e-11):
                break
        nuclei_density = (
            neutral_density
            + proton_density
            + negative_hydrogen_density
            + 2.0 * molecular_density
            + 2.0 * molecular_ion_density
            + 3.0 * trihydrogen_ion_density
        )
        ionization = electron_density / nuclei_density

    distribution = hydrogen_level_distribution(
        neutral_density,
        electron_density,
        temperature,
        maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    return HydrogenLTEState(
        mass_density=np.asarray(HYDROGEN_MASS * nuclei_density),
        hydrogen_nuclei_density=np.asarray(nuclei_density),
        neutral_h_density=np.asarray(neutral_density),
        proton_density=np.asarray(proton_density),
        electron_density=np.asarray(electron_density),
        ionization_fraction=np.asarray(ionization),
        internal_partition_function=distribution.internal_partition_function,
        level_occupation_probability=distribution.occupation_probability,
        level_population_density=distribution.population_density,
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        molecular_hydrogen_density=(
            None if molecular_density is None else np.asarray(molecular_density)
        ),
        molecular_hydrogen_ion_density=(
            None
            if molecular_ion_density is None
            else np.asarray(molecular_ion_density)
        ),
        negative_hydrogen_density=(
            None
            if negative_hydrogen_density is None
            else np.asarray(negative_hydrogen_density)
        ),
        trihydrogen_ion_density=(
            None
            if trihydrogen_ion_density is None
            else np.asarray(trihydrogen_ion_density)
        ),
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
        chemical_model=(
            "h-hplus-hminus-h2-h2plus-h3plus-hm/"
            + str(trihydrogen_ion_partition_model)
            if include_molecules
            and include_negative_hydrogen
            and trihydrogen_ion_partition_model is not None
            else "h-hplus-hminus-h2-h2plus-hm"
            if include_molecules and include_negative_hydrogen
            else "h-hplus-h2-h2plus-hm"
            if include_molecules
            else "atomic-hm"
        ),
        neutral_radius_scale=float(neutral_radius_scale),
    )


def _hummer_mihalas_specific_enthalpy(
    temperature: FloatArray,
    gas_pressure: FloatArray,
    *,
    maximum_level: int,
    correlated_microfields: bool,
    include_molecules: bool,
    include_negative_hydrogen: bool,
    trihydrogen_ion_partition_model: str | None,
) -> tuple[FloatArray, HydrogenLTEState]:
    """Return specific enthalpy for the adopted HM chemical-picture EOS."""

    state = hummer_mihalas_hydrogen_lte(
        temperature,
        gas_pressure,
        maximum_level=maximum_level,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    assert state.level_population_density is not None
    level = np.arange(1, maximum_level + 1, dtype=np.float64)
    excitation_energy = HYDROGEN_IONIZATION_ENERGY * (
        1.0 - 1.0 / level**2
    )
    excitation_density = np.sum(
        state.level_population_density * excitation_energy, axis=-1
    )
    molecular_density = (
        np.zeros_like(state.neutral_h_density)
        if state.molecular_hydrogen_density is None
        else state.molecular_hydrogen_density
    )
    molecular_ion_density = (
        np.zeros_like(state.neutral_h_density)
        if state.molecular_hydrogen_ion_density is None
        else state.molecular_hydrogen_ion_density
    )
    negative_hydrogen_density = (
        np.zeros_like(state.neutral_h_density)
        if state.negative_hydrogen_density is None
        else state.negative_hydrogen_density
    )
    trihydrogen_ion_density = (
        np.zeros_like(state.neutral_h_density)
        if state.trihydrogen_ion_density is None
        else state.trihydrogen_ion_density
    )
    ideal_particle_density = (
        state.neutral_h_density
        + state.proton_density
        + state.electron_density
        + molecular_density
        + molecular_ion_density
        + negative_hydrogen_density
        + trihydrogen_ion_density
    )
    internal_energy_density = (
        1.5 * BOLTZMANN * temperature * ideal_particle_density
        + HYDROGEN_IONIZATION_ENERGY
        * (
            state.proton_density
            + molecular_ion_density
            + trihydrogen_ion_density
        )
        + excitation_density
        - H2_DISSOCIATION_ENERGY * molecular_density
        - H2_PLUS_DISSOCIATION_ENERGY * molecular_ion_density
        - H_MINUS_DETACHMENT_ENERGY * negative_hydrogen_density
        - (
            H2_DISSOCIATION_ENERGY + H3_PLUS_DISSOCIATION_ENERGY
        ) * trihydrogen_ion_density
        + molecular_hydrogen_rovibrational_energy(temperature)
        * molecular_density
        + molecular_hydrogen_ion_rovibrational_energy(temperature)
        * molecular_ion_density
        + (
            trihydrogen_ion_rovibrational_energy(
                temperature,
                partition_model=trihydrogen_ion_partition_model,
            )
            * trihydrogen_ion_density
            if trihydrogen_ion_partition_model is not None
            else 0.0
        )
    )
    return (
        (internal_energy_density + gas_pressure) / state.mass_density,
        state,
    )


def hummer_mihalas_hydrogen_thermodynamics(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
    central_state: HydrogenLTEState | None = None,
) -> HydrogenThermodynamics:
    r"""Return numerical :math:`c_P`, :math:`Q`, and :math:`\nabla_\mathrm{ad}`.

    Centered logarithmic derivatives are evaluated with the same
    occupation-probability EOS and level distribution used for the atmosphere
    state.  This keeps the ML2 stability criterion consistent with the
    ionization and excitation populations without duplicating the lengthy MHD
    free-energy derivative algebra.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(gas_pressure))
        or np.any(gas_pressure <= 0.0)
    ):
        raise ValueError("temperature and gas_pressure must be finite and positive")
    epsilon = 2.0e-4
    cooler_temperature = temperature * np.exp(-epsilon)
    hotter_temperature = temperature * np.exp(epsilon)
    cooler_enthalpy, cooler = _hummer_mihalas_specific_enthalpy(
        cooler_temperature,
        gas_pressure,
        maximum_level=maximum_level,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    hotter_enthalpy, hotter = _hummer_mihalas_specific_enthalpy(
        hotter_temperature,
        gas_pressure,
        maximum_level=maximum_level,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    if central_state is None:
        state = hummer_mihalas_hydrogen_lte(
            temperature,
            gas_pressure,
            maximum_level=maximum_level,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
        )
    else:
        if central_state.mass_density.shape != temperature.shape:
            raise ValueError("central_state and thermodynamic grid must match")
        state = central_state
    specific_heat = (
        hotter_enthalpy - cooler_enthalpy
    ) / (hotter_temperature - cooler_temperature)
    density_temperature_derivative = -(
        np.log(hotter.mass_density) - np.log(cooler.mass_density)
    ) / (2.0 * epsilon)
    adiabatic_gradient = (
        gas_pressure
        * density_temperature_derivative
        / (state.mass_density * temperature * specific_heat)
    )
    return HydrogenThermodynamics(
        specific_heat_constant_pressure=np.asarray(specific_heat),
        density_temperature_derivative=np.asarray(
            density_temperature_derivative
        ),
        adiabatic_temperature_gradient=np.asarray(adiabatic_gradient),
    )


def ideal_hydrogen_lte(temperature: ArrayLike, gas_pressure: ArrayLike) -> HydrogenLTEState:
    """Solve the ideal pure-hydrogen Saha equation at fixed gas pressure.

    The analytic pressure-form solution avoids a nonlinear root solve.  If
    ``x = n_p / (n_H + n_p)`` and ``N = P / kT``, then
    ``x**2 = S / (N + S)`` and ``n_nuclei = N / (1 + x)``.
    Inputs broadcast according to NumPy rules.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    if np.any(~np.isfinite(gas_pressure)) or np.any(gas_pressure <= 0.0):
        raise ValueError("gas_pressure must contain finite positive values")

    saha = hydrogen_saha_constant(temperature)
    particle_density = gas_pressure / (BOLTZMANN * temperature)
    ionization_fraction = np.sqrt(saha / (particle_density + saha))
    nuclei_density = particle_density / (1.0 + ionization_fraction)
    proton_density = ionization_fraction * nuclei_density
    neutral_density = nuclei_density - proton_density

    return HydrogenLTEState(
        mass_density=np.asarray(HYDROGEN_MASS * nuclei_density),
        hydrogen_nuclei_density=np.asarray(nuclei_density),
        neutral_h_density=np.asarray(neutral_density),
        proton_density=np.asarray(proton_density),
        electron_density=np.asarray(proton_density),
        ionization_fraction=np.asarray(ionization_fraction),
    )


def ideal_hydrogen_thermodynamics(
    temperature: ArrayLike, gas_pressure: ArrayLike
) -> HydrogenThermodynamics:
    r"""Return :math:`c_P`, :math:`Q`, and :math:`\nabla_\mathrm{ad}`.

    The derivatives are analytic for the same ground-state Saha mixture used
    by :func:`ideal_hydrogen_lte`.  Ionization energy is included in the
    enthalpy, which produces the low adiabatic gradient in the hydrogen
    partial-ionization zone that drives DA-atmosphere convection.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    state = ideal_hydrogen_lte(temperature, gas_pressure)
    ionization = state.ionization_fraction
    logarithmic_saha_pressure_derivative = (
        2.5 + HYDROGEN_IONIZATION_ENERGY / (BOLTZMANN * temperature)
    )
    ionization_log_temperature_derivative = (
        0.5
        * ionization
        * (1.0 - ionization**2)
        * logarithmic_saha_pressure_derivative
    )
    density_temperature_derivative = (
        1.0
        + ionization_log_temperature_derivative / (1.0 + ionization)
    )
    specific_heat = (
        2.5 * BOLTZMANN * (1.0 + ionization)
        + (2.5 * BOLTZMANN * temperature + HYDROGEN_IONIZATION_ENERGY)
        * ionization_log_temperature_derivative
        / temperature
    ) / HYDROGEN_MASS
    adiabatic_gradient = (
        (1.0 + ionization)
        * BOLTZMANN
        * density_temperature_derivative
        / (HYDROGEN_MASS * specific_heat)
    )
    return HydrogenThermodynamics(
        specific_heat_constant_pressure=np.asarray(specific_heat),
        density_temperature_derivative=np.asarray(
            density_temperature_derivative
        ),
        adiabatic_temperature_gradient=np.asarray(adiabatic_gradient),
    )


# NIST ASD term energies for the lower states of the optical He I spectrum.
# The triplet-P energy is statistical-weight averaged over J=0,1,2.  The
# first entry is the 1s^2 ground state.  Statistical weights are for complete
# LS terms rather than individual fine-structure components.
HELIUM_I_LOW_TERM_WAVENUMBER = np.asarray(
    [0.0, 159_855.971_776, 166_277.437_635, 169_086.907_646, 171_134.894_441],
    dtype=np.float64,
)
HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT = np.asarray(
    [1.0, 3.0, 1.0, 9.0, 3.0], dtype=np.float64
)
HELIUM_I_LOW_TERM_ENERGY = (
    PLANCK * LIGHT_SPEED * HELIUM_I_LOW_TERM_WAVENUMBER
)
HELIUM_I_LOW_TERM_LABELS = (
    "1s2_1S",
    "1s2s_3S",
    "1s2s_1S",
    "1s2p_3P",
    "1s2p_1P",
)
def helium_occupation_probability(
    neutral_he_density: ArrayLike,
    electron_density: ArrayLike,
    temperature: ArrayLike,
    effective_principal_quantum_number: ArrayLike,
    *,
    neutral_h_density: ArrayLike = 0.0,
    neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    hydrogen_neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    correlated_microfields: bool = False,
) -> FloatArray:
    """Return the HM occupation probability for a He I bound state.

    The neutral hard-sphere radius is ``r_B n_eff^2 a_0``.  ``r_B=0.5`` is
    the standard ATMO/Montreal DB choice.  The charged term uses the same
    MHD/Q-MHD microfield integral as a neutral hydrogenic Rydberg electron,
    with ``n_eff`` inferred from the binding energy.
    """

    neutral_he_density, neutral_h_density, electron_density, temperature, effective_level = (
        np.broadcast_arrays(
            np.asarray(neutral_he_density, dtype=np.float64),
            np.asarray(neutral_h_density, dtype=np.float64),
            np.asarray(electron_density, dtype=np.float64),
            np.asarray(temperature, dtype=np.float64),
            np.asarray(effective_principal_quantum_number, dtype=np.float64),
        )
    )
    if (
        np.any(~np.isfinite(neutral_he_density))
        or np.any(neutral_he_density < 0.0)
        or np.any(~np.isfinite(neutral_h_density))
        or np.any(neutral_h_density < 0.0)
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(effective_level))
        or np.any(effective_level <= 0.0)
        or not np.isfinite(neutral_radius_scale)
        or neutral_radius_scale <= 0.0
        or not np.isfinite(hydrogen_neutral_radius_scale)
        or hydrogen_neutral_radius_scale <= 0.0
    ):
        raise ValueError("helium occupation-probability inputs must be physical")

    charged = charged_particle_hydrogen_occupation_probability(
        electron_density,
        np.maximum(effective_level, 1.0),
        temperature if correlated_microfields else None,
    )
    level_radius = neutral_radius_scale * BOHR_RADIUS * effective_level**2
    helium_ground_radius = neutral_radius_scale * BOHR_RADIUS
    hydrogen_ground_radius = hydrogen_neutral_radius_scale * BOHR_RADIUS
    helium_excluded_volume = (
        4.0 / 3.0 * PI * (level_radius + helium_ground_radius) ** 3
    )
    hydrogen_excluded_volume = (
        4.0 / 3.0 * PI * (level_radius + hydrogen_ground_radius) ** 3
    )
    return charged * np.exp(
        -neutral_he_density * helium_excluded_volume
        -neutral_h_density * hydrogen_excluded_volume
    )


def _helium_partition_functions(
    temperature: FloatArray,
    electron_density: FloatArray,
    neutral_he_density: FloatArray,
    *,
    neutral_h_density: ArrayLike = 0.0,
    maximum_level: int,
    neutral_radius_scale: float,
    hydrogen_neutral_radius_scale: float = HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    correlated_microfields: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return He I/II partition functions and mean excitation energies."""

    shape = temperature.shape
    low_energy = HELIUM_I_LOW_TERM_ENERGY
    low_binding = np.maximum(
        HELIUM_FIRST_IONIZATION_ENERGY - low_energy,
        np.finfo(np.float64).tiny,
    )
    low_effective_level = np.sqrt(HYDROGEN_IONIZATION_ENERGY / low_binding)
    low_occupation = helium_occupation_probability(
        neutral_he_density[..., np.newaxis],
        electron_density[..., np.newaxis],
        temperature[..., np.newaxis],
        low_effective_level,
        neutral_h_density=np.asarray(neutral_h_density)[..., np.newaxis],
        neutral_radius_scale=neutral_radius_scale,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    low_terms = (
        HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * low_occupation
        * np.exp(-low_energy / (BOLTZMANN * temperature[..., np.newaxis]))
    )
    neutral_partition = np.sum(low_terms, axis=-1)
    neutral_excitation_numerator = np.sum(low_terms * low_energy, axis=-1)

    if maximum_level >= 3:
        principal = np.arange(3, maximum_level + 1, dtype=np.float64)
        shell_energy = (
            HELIUM_FIRST_IONIZATION_ENERGY
            - HYDROGEN_IONIZATION_ENERGY / principal**2
        )
        shell_weight = 4.0 * principal**2
        shell_occupation = helium_occupation_probability(
            neutral_he_density[..., np.newaxis],
            electron_density[..., np.newaxis],
            temperature[..., np.newaxis],
            principal,
            neutral_h_density=np.asarray(neutral_h_density)[..., np.newaxis],
            neutral_radius_scale=neutral_radius_scale,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        shell_terms = (
            shell_weight
            * shell_occupation
            * np.exp(
                -shell_energy
                / (BOLTZMANN * temperature[..., np.newaxis])
            )
        )
        neutral_partition += np.sum(shell_terms, axis=-1)
        neutral_excitation_numerator += np.sum(
            shell_terms * shell_energy, axis=-1
        )

    principal_ion = np.arange(1, maximum_level + 1, dtype=np.float64)
    ion_energy = HELIUM_SECOND_IONIZATION_ENERGY * (
        1.0 - 1.0 / principal_ion**2
    )
    ion_weight = 2.0 * principal_ion**2
    ion_charged = charged_particle_hydrogen_occupation_probability(
        electron_density[..., np.newaxis],
        principal_ion,
        temperature[..., np.newaxis] if correlated_microfields else None,
        ionic_charge=2.0,
    )
    ion_radius = (
        neutral_radius_scale
        * BOHR_RADIUS
        * principal_ion**2
        / 2.0
    )
    neutral_ground_radius = neutral_radius_scale * BOHR_RADIUS
    hydrogen_ground_radius = hydrogen_neutral_radius_scale * BOHR_RADIUS
    ion_neutral = np.exp(
        -neutral_he_density[..., np.newaxis]
        * 4.0
        / 3.0
        * PI
        * (ion_radius + neutral_ground_radius) ** 3
        - np.asarray(neutral_h_density)[..., np.newaxis]
        * 4.0
        / 3.0
        * PI
        * (ion_radius + hydrogen_ground_radius) ** 3
    )
    ion_terms = (
        ion_weight
        * ion_charged
        * ion_neutral
        * np.exp(-ion_energy / (BOLTZMANN * temperature[..., np.newaxis]))
    )
    # At deliberately extreme trial pressures used while bracketing the
    # hydrostatic solution, every explicit state can be pressure-dissolved.
    # A tiny floor is the log-domain representation of that limiting case and
    # prevents 0/0 from poisoning the otherwise well-bracketed EOS solve.
    tiny = np.finfo(np.float64).tiny
    ion_partition = np.maximum(np.sum(ion_terms, axis=-1), tiny)
    neutral_partition = np.maximum(neutral_partition, tiny)
    ion_excitation = np.sum(ion_terms * ion_energy, axis=-1) / ion_partition
    neutral_excitation = neutral_excitation_numerator / neutral_partition
    return (
        neutral_partition.reshape(shape),
        ion_partition.reshape(shape),
        low_occupation,
        neutral_excitation.reshape(shape),
        ion_excitation.reshape(shape),
    )


def hummer_mihalas_helium_lte(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = False,
) -> HeliumLTEState:
    """Solve a pure-He HM occupation-probability EOS at fixed gas pressure.

    He I, He II, He III, and electrons are in simultaneous Saha equilibrium.
    The internal partition functions use NIST term energies for the optical
    ``n=2`` terms and a hydrogenic Rydberg completion through
    ``maximum_level``.  Charge neutrality is solved in logarithmic electron
    density, which remains stable from nearly neutral DC atmospheres through
    the He II ionization zone.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(gas_pressure))
        or np.any(gas_pressure <= 0.0)
        or maximum_level < 3
    ):
        raise ValueError("temperature and gas_pressure must be positive and maximum_level >= 3")

    particle_density = gas_pressure / (BOLTZMANN * temperature)
    log_lower = np.log(particle_density) - 90.0
    log_upper = np.log(2.0 * particle_density / 3.0)
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )

    def state_at_electron_density(
        electron_density: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        nuclei_density = np.maximum(
            particle_density - electron_density,
            np.finfo(np.float64).tiny,
        )
        neutral_perturber = nuclei_density.copy()
        for _ in range(3):
            neutral_partition, ion_partition, low_occupation, _, _ = (
                _helium_partition_functions(
                    temperature,
                    electron_density,
                    neutral_perturber,
                    maximum_level=maximum_level,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            log_ratio_one = (
                np.log(2.0)
                + translational_log
                + np.log(ion_partition)
                - np.log(neutral_partition)
                - HELIUM_FIRST_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            log_ratio_two = (
                np.log(2.0)
                + translational_log
                - np.log(ion_partition)
                - HELIUM_SECOND_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            terms = np.stack(
                (
                    np.zeros_like(log_ratio_one),
                    log_ratio_one,
                    log_ratio_one + log_ratio_two,
                ),
                axis=-1,
            )
            maximum = np.max(terms, axis=-1, keepdims=True)
            weights = np.exp(np.clip(terms - maximum, -745.0, 0.0))
            fractions = weights / np.sum(weights, axis=-1, keepdims=True)
            neutral_perturber = nuclei_density * fractions[..., 0]
        mean_charge = fractions[..., 1] + 2.0 * fractions[..., 2]
        return (
            electron_density - nuclei_density * mean_charge,
            nuclei_density,
            fractions,
            neutral_partition,
            ion_partition,
            low_occupation,
            mean_charge,
        )

    # Fifty-six log-bisection steps give far more precision than the
    # opacity and profile data warrant while avoiding a large cost in the
    # hydrostatic and radiative-equilibrium inner loops.
    for _ in range(56):
        log_middle = 0.5 * (log_lower + log_upper)
        middle = np.exp(log_middle)
        residual, *_ = state_at_electron_density(middle)
        positive = residual > 0.0
        log_upper = np.where(positive, log_middle, log_upper)
        log_lower = np.where(positive, log_lower, log_middle)

    electron_density = np.exp(0.5 * (log_lower + log_upper))
    (
        _,
        nuclei_density,
        fractions,
        neutral_partition,
        ion_partition,
        low_occupation,
        mean_charge,
    ) = state_at_electron_density(electron_density)
    neutral_density = nuclei_density * fractions[..., 0]
    singly_ionized_density = nuclei_density * fractions[..., 1]
    doubly_ionized_density = nuclei_density * fractions[..., 2]
    low_terms = (
        HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * low_occupation
        * np.exp(
            -HELIUM_I_LOW_TERM_ENERGY
            / (BOLTZMANN * temperature[..., np.newaxis])
        )
    )
    low_population = (
        neutral_density[..., np.newaxis]
        * low_terms
        / neutral_partition[..., np.newaxis]
    )
    return HeliumLTEState(
        mass_density=np.asarray(HELIUM_MASS * nuclei_density),
        helium_nuclei_density=np.asarray(nuclei_density),
        neutral_he_density=np.asarray(neutral_density),
        singly_ionized_he_density=np.asarray(singly_ionized_density),
        doubly_ionized_he_density=np.asarray(doubly_ionized_density),
        electron_density=np.asarray(electron_density),
        mean_ion_charge=np.asarray(mean_charge),
        neutral_partition_function=np.asarray(neutral_partition),
        singly_ionized_partition_function=np.asarray(ion_partition),
        neutral_level_occupation_probability=np.asarray(low_occupation),
        neutral_level_population_density=np.asarray(low_population),
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        neutral_radius_scale=float(neutral_radius_scale),
    )


def hummer_mihalas_helium_lte_at_nuclei_density(
    temperature: ArrayLike,
    helium_nuclei_density: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = False,
) -> HeliumLTEState:
    """Solve the chemical-picture helium populations at fixed bulk density.

    This form is needed when a physical dense-fluid EOS supplies the mass
    density independently of the ideal particle-pressure equation.  It uses
    the same HM/Q-MHD partition functions and Saha relations as
    :func:`hummer_mihalas_helium_lte`; only the thermodynamic independent
    variable differs.
    """

    temperature, nuclei_density = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(helium_nuclei_density, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(nuclei_density))
        or np.any(nuclei_density <= 0.0)
        or maximum_level < 3
    ):
        raise ValueError(
            "temperature and helium_nuclei_density must be positive and "
            "maximum_level >= 3"
        )
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )

    def state_at_electron_density(
        electron_density: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        neutral_perturber = nuclei_density.copy()
        for _ in range(3):
            neutral_partition, ion_partition, low_occupation, _, _ = (
                _helium_partition_functions(
                    temperature,
                    electron_density,
                    neutral_perturber,
                    maximum_level=maximum_level,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            log_ratio_one = (
                np.log(2.0)
                + translational_log
                + np.log(ion_partition)
                - np.log(neutral_partition)
                - HELIUM_FIRST_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            log_ratio_two = (
                np.log(2.0)
                + translational_log
                - np.log(ion_partition)
                - HELIUM_SECOND_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            terms = np.stack(
                (
                    np.zeros_like(log_ratio_one),
                    log_ratio_one,
                    log_ratio_one + log_ratio_two,
                ),
                axis=-1,
            )
            maximum = np.max(terms, axis=-1, keepdims=True)
            weights = np.exp(np.clip(terms - maximum, -745.0, 0.0))
            fractions = weights / np.sum(weights, axis=-1, keepdims=True)
            neutral_perturber = nuclei_density * fractions[..., 0]
        mean_charge = fractions[..., 1] + 2.0 * fractions[..., 2]
        return (
            electron_density - nuclei_density * mean_charge,
            fractions,
            neutral_partition,
            ion_partition,
            low_occupation,
            mean_charge,
        )

    log_lower = np.log(nuclei_density) - 90.0
    log_upper = np.log(2.0 * nuclei_density)
    for _ in range(56):
        log_middle = 0.5 * (log_lower + log_upper)
        residual, *_ = state_at_electron_density(np.exp(log_middle))
        positive = residual > 0.0
        log_upper = np.where(positive, log_middle, log_upper)
        log_lower = np.where(positive, log_lower, log_middle)
    electron_density = np.exp(0.5 * (log_lower + log_upper))
    (
        _,
        fractions,
        neutral_partition,
        ion_partition,
        low_occupation,
        mean_charge,
    ) = state_at_electron_density(electron_density)
    neutral_density = nuclei_density * fractions[..., 0]
    singly_ionized_density = nuclei_density * fractions[..., 1]
    doubly_ionized_density = nuclei_density * fractions[..., 2]
    low_terms = (
        HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * low_occupation
        * np.exp(
            -HELIUM_I_LOW_TERM_ENERGY
            / (BOLTZMANN * temperature[..., np.newaxis])
        )
    )
    low_population = (
        neutral_density[..., np.newaxis]
        * low_terms
        / neutral_partition[..., np.newaxis]
    )
    return HeliumLTEState(
        mass_density=np.asarray(HELIUM_MASS * nuclei_density),
        helium_nuclei_density=np.asarray(nuclei_density),
        neutral_he_density=np.asarray(neutral_density),
        singly_ionized_he_density=np.asarray(singly_ionized_density),
        doubly_ionized_he_density=np.asarray(doubly_ionized_density),
        electron_density=np.asarray(electron_density),
        mean_ion_charge=np.asarray(mean_charge),
        neutral_partition_function=np.asarray(neutral_partition),
        singly_ionized_partition_function=np.asarray(ion_partition),
        neutral_level_occupation_probability=np.asarray(low_occupation),
        neutral_level_population_density=np.asarray(low_population),
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        neutral_radius_scale=float(neutral_radius_scale),
    )


def hummer_mihalas_helium_lte_with_reos3(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    table: HeliumREOS3Table,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = False,
) -> HeliumLTEState:
    """Combine He-REOS.3 bulk density with HM/Q-MHD level populations."""

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    tabulated_density, _, inside = table.evaluate(gas_pressure, temperature)
    if np.all(inside):
        mass_density = tabulated_density
    elif not np.any(inside):
        return hummer_mihalas_helium_lte(
            temperature,
            gas_pressure,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    else:
        outside = ~inside
        dilute = hummer_mihalas_helium_lte(
            temperature[outside],
            gas_pressure[outside],
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        mass_density = np.asarray(tabulated_density).copy()
        mass_density[outside] = dilute.mass_density
    return hummer_mihalas_helium_lte_at_nuclei_density(
        temperature,
        mass_density / HELIUM_MASS,
        maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )


def _hummer_mihalas_helium_specific_enthalpy(
    temperature: FloatArray,
    gas_pressure: FloatArray,
    *,
    maximum_level: int,
    neutral_radius_scale: float,
    correlated_microfields: bool,
    helium_reos3_table: HeliumREOS3Table | None = None,
    separate_neutral_translation: bool = False,
) -> tuple[FloatArray, HeliumLTEState]:
    state = (
        hummer_mihalas_helium_lte(
            temperature,
            gas_pressure,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3(
            temperature,
            gas_pressure,
            helium_reos3_table,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    )
    _, _, _, neutral_excitation, ion_excitation = _helium_partition_functions(
        temperature,
        state.electron_density,
        state.neutral_he_density,
        maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    ideal_particle_density = state.helium_nuclei_density + state.electron_density
    reaction_energy_density = (
        state.singly_ionized_he_density
        * (HELIUM_FIRST_IONIZATION_ENERGY + ion_excitation)
        + state.doubly_ionized_he_density
        * (HELIUM_FIRST_IONIZATION_ENERGY + HELIUM_SECOND_IONIZATION_ENERGY)
        + state.neutral_he_density * neutral_excitation
    )
    if separate_neutral_translation:
        if helium_reos3_table is not None:
            raise ValueError(
                "Neutral translation separation requires the ideal-pressure HM EOS"
            )
        # Compute h - (5/2) kT/m_He without subtracting that large ideal
        # contribution from h. The remaining particle enthalpy is due only
        # to electrons; reaction/excitation terms retain the same populations.
        enthalpy = (
            reaction_energy_density
            + 2.5 * BOLTZMANN * temperature * state.electron_density
        ) / state.mass_density
    else:
        internal_energy_density = (
            1.5 * BOLTZMANN * temperature * ideal_particle_density
            + reaction_energy_density
        )
        enthalpy = (internal_energy_density + gas_pressure) / state.mass_density
    if helium_reos3_table is not None:
        _, tabulated_internal_energy, inside = helium_reos3_table.evaluate(
            gas_pressure, temperature
        )
        enthalpy = np.where(
            inside,
            tabulated_internal_energy + gas_pressure / state.mass_density,
            enthalpy,
        )
    return np.asarray(enthalpy), state


def hummer_mihalas_helium_thermodynamics(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = False,
    helium_reos3_table: HeliumREOS3Table | None = None,
) -> HeliumThermodynamics:
    r"""Return :math:`c_P`, :math:`Q`, and :math:`\nabla_ad` for He.

    For ideal-pressure HM, differentiate reaction/ionization contributions
    numerically but handle neutral translation analytically. This avoids
    noise from differencing two nearly equal ideal-gas enthalpies/densities
    in efficient convection. No temperature or ionization threshold is used.
    The tabulated REOS derivative path is unchanged.
    """

    temperature, gas_pressure = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
    )
    epsilon = 2.0e-4
    cooler_temperature = temperature * np.exp(-epsilon)
    hotter_temperature = temperature * np.exp(epsilon)
    cooler_enthalpy, cooler = _hummer_mihalas_helium_specific_enthalpy(
        cooler_temperature,
        gas_pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
        helium_reos3_table=helium_reos3_table,
        separate_neutral_translation=helium_reos3_table is None,
    )
    hotter_enthalpy, hotter = _hummer_mihalas_helium_specific_enthalpy(
        hotter_temperature,
        gas_pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
        helium_reos3_table=helium_reos3_table,
        separate_neutral_translation=helium_reos3_table is None,
    )
    state = (
        hummer_mihalas_helium_lte(
            temperature,
            gas_pressure,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3(
            temperature,
            gas_pressure,
            helium_reos3_table,
            maximum_level=maximum_level,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    )
    specific_heat = (
        (hotter_enthalpy - cooler_enthalpy)
        / (hotter_temperature - cooler_temperature)
    )
    if helium_reos3_table is None:
        specific_heat += 2.5 * BOLTZMANN / HELIUM_MASS
        # rho = m_He P/[kT(1+Z)]. log1p retains the response of trace ions
        # instead of losing it underneath the explicit -ln T dependence.
        expansion = 1.0 + (
            np.log1p(hotter.mean_ion_charge) - np.log1p(cooler.mean_ion_charge)
        ) / (2.0 * epsilon)
        adiabatic_gradient = (
            BOLTZMANN * (1.0 + state.mean_ion_charge) * expansion
            / (HELIUM_MASS * specific_heat)
        )
    else:
        expansion = -(
            np.log(hotter.mass_density) - np.log(cooler.mass_density)
        ) / (2.0 * epsilon)
        adiabatic_gradient = (
            gas_pressure * expansion / (state.mass_density * temperature * specific_heat)
        )
    return HeliumThermodynamics(
        specific_heat_constant_pressure=np.asarray(specific_heat),
        density_temperature_derivative=np.asarray(expansion),
        adiabatic_temperature_gradient=np.asarray(adiabatic_gradient),
    )


def _hydrogen_helium_number_fractions(
    log_hydrogen_to_helium: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return numerically stable H and He nuclei fractions."""

    natural_log_ratio = np.log(10.0) * log_hydrogen_to_helium
    hydrogen_fraction = np.exp(
        natural_log_ratio - np.logaddexp(0.0, natural_log_ratio)
    )
    helium_fraction = np.exp(-np.logaddexp(0.0, natural_log_ratio))
    return hydrogen_fraction, helium_fraction


def hummer_mihalas_hydrogen_helium_lte(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    log_hydrogen_to_helium: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = True,
) -> HydrogenHeliumLTEState:
    """Solve a homogeneous atomic H/He occupation-probability EOS.

    ``log_hydrogen_to_helium`` is the base-ten nuclei number ratio.  Both
    elements share one electron density, and their neutral hard-sphere
    occupation probabilities include H and He perturbers.  This first mixed
    milestone deliberately excludes molecules and dense-fluid corrections;
    it is intended for warm DAB/DBA atmospheres.

    The translational pressure is ``kT (N_H + N_He + n_e)``.  As in the
    established pure-He branch, the hard-sphere term is used to truncate
    internal partition functions but is not separately added to the pressure.
    """

    temperature, gas_pressure, log_ratio = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
        np.asarray(log_hydrogen_to_helium, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(gas_pressure))
        or np.any(gas_pressure <= 0.0)
        or np.any(~np.isfinite(log_ratio))
        or maximum_level < 3
        or not np.isfinite(hydrogen_neutral_radius_scale)
        or hydrogen_neutral_radius_scale <= 0.0
        or not np.isfinite(helium_neutral_radius_scale)
        or helium_neutral_radius_scale <= 0.0
    ):
        raise ValueError(
            "mixed H/He EOS inputs must be finite and positive, with "
            "maximum_level >= 3"
        )

    hydrogen_fraction, helium_fraction = _hydrogen_helium_number_fractions(
        log_ratio
    )
    particle_density = gas_pressure / (BOLTZMANN * temperature)
    maximum_mean_charge = hydrogen_fraction + 2.0 * helium_fraction
    maximum_electron_density = (
        particle_density
        * maximum_mean_charge
        / (1.0 + maximum_mean_charge)
    )
    tiny = np.finfo(np.float64).tiny
    log_lower = np.log(np.maximum(particle_density, 1.0)) - 90.0
    log_upper = np.log(np.maximum(maximum_electron_density, tiny))
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )
    log_hydrogen_saha = np.log(
        np.maximum(hydrogen_saha_constant(temperature), tiny)
    )

    def composition_at_electron_density(
        electron_density: FloatArray,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        HydrogenLevelDistribution,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        nuclei_density = np.maximum(particle_density - electron_density, tiny)
        hydrogen_nuclei = hydrogen_fraction * nuclei_density
        helium_nuclei = helium_fraction * nuclei_density
        neutral_hydrogen = hydrogen_nuclei.copy()
        neutral_helium = helium_nuclei.copy()
        hydrogen_ion_fraction = np.zeros_like(nuclei_density)
        helium_ion_fractions = np.stack(
            (
                np.ones_like(nuclei_density),
                np.zeros_like(nuclei_density),
                np.zeros_like(nuclei_density),
            ),
            axis=-1,
        )
        hydrogen_distribution: HydrogenLevelDistribution | None = None
        neutral_partition = np.ones_like(nuclei_density)
        ion_partition = np.ones_like(nuclei_density)
        low_occupation = np.ones(nuclei_density.shape + (5,))

        # Cross-species occupation probabilities make the neutral populations
        # a short fixed-point problem at a trial electron density.
        for _ in range(4):
            hydrogen_distribution = hydrogen_level_distribution(
                neutral_hydrogen,
                electron_density,
                temperature,
                maximum_level=maximum_level,
                neutral_he_density=neutral_helium,
                neutral_radius_scale=hydrogen_neutral_radius_scale,
                helium_neutral_radius_scale=helium_neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            log_hydrogen_ion_ratio = (
                log_hydrogen_saha
                - np.log(
                    np.maximum(
                        hydrogen_distribution.internal_partition_function,
                        tiny,
                    )
                )
                - np.log(electron_density)
            )
            hydrogen_ion_fraction = np.exp(
                log_hydrogen_ion_ratio
                - np.logaddexp(0.0, log_hydrogen_ion_ratio)
            )

            (
                neutral_partition,
                ion_partition,
                low_occupation,
                _,
                _,
            ) = _helium_partition_functions(
                temperature,
                electron_density,
                neutral_helium,
                neutral_h_density=neutral_hydrogen,
                maximum_level=maximum_level,
                neutral_radius_scale=helium_neutral_radius_scale,
                hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            log_ratio_one = (
                np.log(2.0)
                + translational_log
                + np.log(ion_partition)
                - np.log(neutral_partition)
                - HELIUM_FIRST_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            log_ratio_two = (
                np.log(2.0)
                + translational_log
                - np.log(ion_partition)
                - HELIUM_SECOND_IONIZATION_ENERGY / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            terms = np.stack(
                (
                    np.zeros_like(log_ratio_one),
                    log_ratio_one,
                    log_ratio_one + log_ratio_two,
                ),
                axis=-1,
            )
            maximum = np.max(terms, axis=-1, keepdims=True)
            weights = np.exp(np.clip(terms - maximum, -745.0, 0.0))
            helium_ion_fractions = weights / np.sum(
                weights, axis=-1, keepdims=True
            )
            candidate_neutral_hydrogen = (
                hydrogen_nuclei * (1.0 - hydrogen_ion_fraction)
            )
            candidate_neutral_helium = (
                helium_nuclei * helium_ion_fractions[..., 0]
            )
            neutral_hydrogen = candidate_neutral_hydrogen
            neutral_helium = candidate_neutral_helium

        assert hydrogen_distribution is not None
        # Re-evaluate both partitions at the final perturber populations so
        # returned levels and ion fractions describe exactly the same state.
        hydrogen_distribution = hydrogen_level_distribution(
            neutral_hydrogen,
            electron_density,
            temperature,
            maximum_level=maximum_level,
            neutral_he_density=neutral_helium,
            neutral_radius_scale=hydrogen_neutral_radius_scale,
            helium_neutral_radius_scale=helium_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        log_hydrogen_ion_ratio = (
            log_hydrogen_saha
            - np.log(
                np.maximum(
                    hydrogen_distribution.internal_partition_function, tiny
                )
            )
            - np.log(electron_density)
        )
        hydrogen_ion_fraction = np.exp(
            log_hydrogen_ion_ratio
            - np.logaddexp(0.0, log_hydrogen_ion_ratio)
        )
        neutral_hydrogen = hydrogen_nuclei * (1.0 - hydrogen_ion_fraction)
        proton_density = hydrogen_nuclei * hydrogen_ion_fraction
        (
            neutral_partition,
            ion_partition,
            low_occupation,
            _,
            _,
        ) = _helium_partition_functions(
            temperature,
            electron_density,
            neutral_helium,
            neutral_h_density=neutral_hydrogen,
            maximum_level=maximum_level,
            neutral_radius_scale=helium_neutral_radius_scale,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        log_ratio_one = (
            np.log(2.0)
            + translational_log
            + np.log(ion_partition)
            - np.log(neutral_partition)
            - HELIUM_FIRST_IONIZATION_ENERGY / (BOLTZMANN * temperature)
            - np.log(electron_density)
        )
        log_ratio_two = (
            np.log(2.0)
            + translational_log
            - np.log(ion_partition)
            - HELIUM_SECOND_IONIZATION_ENERGY / (BOLTZMANN * temperature)
            - np.log(electron_density)
        )
        terms = np.stack(
            (
                np.zeros_like(log_ratio_one),
                log_ratio_one,
                log_ratio_one + log_ratio_two,
            ),
            axis=-1,
        )
        maximum = np.max(terms, axis=-1, keepdims=True)
        weights = np.exp(np.clip(terms - maximum, -745.0, 0.0))
        helium_ion_fractions = weights / np.sum(weights, axis=-1, keepdims=True)
        neutral_helium = helium_nuclei * helium_ion_fractions[..., 0]
        helium_one = helium_nuclei * helium_ion_fractions[..., 1]
        helium_two = helium_nuclei * helium_ion_fractions[..., 2]
        charge_density = proton_density + helium_one + 2.0 * helium_two
        return (
            electron_density - charge_density,
            hydrogen_nuclei,
            helium_nuclei,
            neutral_hydrogen,
            proton_density,
            neutral_helium,
            helium_one,
            hydrogen_distribution,
            helium_two,
            neutral_partition,
            ion_partition,
        )

    for _ in range(44):
        log_middle = 0.5 * (log_lower + log_upper)
        middle = np.exp(log_middle)
        residual, *_ = composition_at_electron_density(middle)
        positive = residual > 0.0
        log_upper = np.where(positive, log_middle, log_upper)
        log_lower = np.where(positive, log_lower, log_middle)

    electron_density = np.exp(0.5 * (log_lower + log_upper))
    (
        _,
        hydrogen_nuclei,
        helium_nuclei,
        neutral_hydrogen,
        proton_density,
        neutral_helium,
        helium_one,
        hydrogen_distribution,
        helium_two,
        neutral_partition,
        ion_partition,
    ) = composition_at_electron_density(electron_density)
    # The final He low-term populations use the occupation probabilities from
    # the final cross-perturber state, not a penultimate fixed-point iterate.
    (
        _,
        _,
        low_occupation,
        _,
        _,
    ) = _helium_partition_functions(
        temperature,
        electron_density,
        neutral_helium,
        neutral_h_density=neutral_hydrogen,
        maximum_level=maximum_level,
        neutral_radius_scale=helium_neutral_radius_scale,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    low_terms = (
        HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * low_occupation
        * np.exp(
            -HELIUM_I_LOW_TERM_ENERGY
            / (BOLTZMANN * temperature[..., np.newaxis])
        )
    )
    low_population = (
        neutral_helium[..., np.newaxis]
        * low_terms
        / neutral_partition[..., np.newaxis]
    )
    hydrogen_state = HydrogenLTEState(
        mass_density=np.asarray(HYDROGEN_MASS * hydrogen_nuclei),
        hydrogen_nuclei_density=np.asarray(hydrogen_nuclei),
        neutral_h_density=np.asarray(neutral_hydrogen),
        proton_density=np.asarray(proton_density),
        electron_density=np.asarray(electron_density),
        ionization_fraction=np.asarray(
            proton_density / np.maximum(hydrogen_nuclei, tiny)
        ),
        internal_partition_function=np.asarray(
            hydrogen_distribution.internal_partition_function
        ),
        level_occupation_probability=np.asarray(
            hydrogen_distribution.occupation_probability
        ),
        level_population_density=np.asarray(
            hydrogen_distribution.population_density
        ),
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        chemical_model="atomic-hm-mixed-h-he",
        neutral_radius_scale=float(hydrogen_neutral_radius_scale),
    )
    helium_mean_charge = (
        helium_one + 2.0 * helium_two
    ) / np.maximum(helium_nuclei, tiny)
    helium_state = HeliumLTEState(
        mass_density=np.asarray(HELIUM_MASS * helium_nuclei),
        helium_nuclei_density=np.asarray(helium_nuclei),
        neutral_he_density=np.asarray(neutral_helium),
        singly_ionized_he_density=np.asarray(helium_one),
        doubly_ionized_he_density=np.asarray(helium_two),
        electron_density=np.asarray(electron_density),
        mean_ion_charge=np.asarray(helium_mean_charge),
        neutral_partition_function=np.asarray(neutral_partition),
        singly_ionized_partition_function=np.asarray(ion_partition),
        neutral_level_occupation_probability=np.asarray(low_occupation),
        neutral_level_population_density=np.asarray(low_population),
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
        neutral_radius_scale=float(helium_neutral_radius_scale),
    )
    ratio = np.exp(np.clip(np.log(10.0) * log_ratio, -745.0, 709.0))
    return HydrogenHeliumLTEState(
        mass_density=np.asarray(
            HYDROGEN_MASS * hydrogen_nuclei + HELIUM_MASS * helium_nuclei
        ),
        hydrogen_nuclei_density=np.asarray(hydrogen_nuclei),
        helium_nuclei_density=np.asarray(helium_nuclei),
        electron_density=np.asarray(electron_density),
        hydrogen_to_helium_number_ratio=np.asarray(ratio),
        hydrogen_lte_state=hydrogen_state,
        helium_lte_state=helium_state,
        microfield_model=("qmhd" if correlated_microfields else "holtsmark"),
    )


def _hydrogen_helium_specific_enthalpy(
    temperature: FloatArray,
    gas_pressure: FloatArray,
    log_hydrogen_to_helium: FloatArray,
    *,
    maximum_level: int,
    hydrogen_neutral_radius_scale: float,
    helium_neutral_radius_scale: float,
    correlated_microfields: bool,
) -> tuple[FloatArray, HydrogenHeliumLTEState]:
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        gas_pressure,
        log_hydrogen_to_helium,
        maximum_level=maximum_level,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    hydrogen = state.hydrogen_lte_state
    helium = state.helium_lte_state
    assert hydrogen.level_population_density is not None
    level = np.arange(1, maximum_level + 1, dtype=np.float64)
    hydrogen_excitation_energy = HYDROGEN_IONIZATION_ENERGY * (
        1.0 - 1.0 / level**2
    )
    hydrogen_excitation_density = np.sum(
        hydrogen.level_population_density * hydrogen_excitation_energy,
        axis=-1,
    )
    _, _, _, helium_neutral_excitation, helium_ion_excitation = (
        _helium_partition_functions(
            temperature,
            state.electron_density,
            helium.neutral_he_density,
            neutral_h_density=hydrogen.neutral_h_density,
            maximum_level=maximum_level,
            neutral_radius_scale=helium_neutral_radius_scale,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    )
    ideal_particle_density = (
        state.hydrogen_nuclei_density
        + state.helium_nuclei_density
        + state.electron_density
    )
    internal_energy_density = (
        1.5 * BOLTZMANN * temperature * ideal_particle_density
        + hydrogen.proton_density * HYDROGEN_IONIZATION_ENERGY
        + hydrogen_excitation_density
        + helium.singly_ionized_he_density
        * (HELIUM_FIRST_IONIZATION_ENERGY + helium_ion_excitation)
        + helium.doubly_ionized_he_density
        * (HELIUM_FIRST_IONIZATION_ENERGY + HELIUM_SECOND_IONIZATION_ENERGY)
        + helium.neutral_he_density * helium_neutral_excitation
    )
    return (
        (internal_energy_density + gas_pressure) / state.mass_density,
        state,
    )


def hummer_mihalas_hydrogen_helium_thermodynamics(
    temperature: ArrayLike,
    gas_pressure: ArrayLike,
    log_hydrogen_to_helium: ArrayLike,
    *,
    maximum_level: int = HM_MAX_BOUND_LEVEL,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields: bool = True,
) -> HydrogenHeliumThermodynamics:
    r"""Return numerical ``c_P``, ``Q``, and ``nabla_ad`` for atomic H/He."""

    temperature, gas_pressure, log_ratio = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(gas_pressure, dtype=np.float64),
        np.asarray(log_hydrogen_to_helium, dtype=np.float64),
    )
    epsilon = 2.0e-4
    cooler_temperature = temperature * np.exp(-epsilon)
    hotter_temperature = temperature * np.exp(epsilon)
    cooler_enthalpy, cooler = _hydrogen_helium_specific_enthalpy(
        cooler_temperature,
        gas_pressure,
        log_ratio,
        maximum_level=maximum_level,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    hotter_enthalpy, hotter = _hydrogen_helium_specific_enthalpy(
        hotter_temperature,
        gas_pressure,
        log_ratio,
        maximum_level=maximum_level,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        gas_pressure,
        log_ratio,
        maximum_level=maximum_level,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    specific_heat = (
        hotter_enthalpy - cooler_enthalpy
    ) / (hotter_temperature - cooler_temperature)
    expansion = -(
        np.log(hotter.mass_density) - np.log(cooler.mass_density)
    ) / (2.0 * epsilon)
    adiabatic_gradient = (
        gas_pressure
        * expansion
        / (state.mass_density * temperature * specific_heat)
    )
    return HydrogenHeliumThermodynamics(
        specific_heat_constant_pressure=np.asarray(specific_heat),
        density_temperature_derivative=np.asarray(expansion),
        adiabatic_temperature_gradient=np.asarray(adiabatic_gradient),
    )
