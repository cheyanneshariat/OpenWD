"""Restricted-NLTE model atom for hydrogenic He II and He III.

This module is the first helium block for the species-independent atmosphere
driver.  It treats shell-averaged He II levels and the He III continuum in
statistical equilibrium while retaining He I in the LTE background.  The
electron-impact data are the public hydrogen CCC cross sections transformed
with the exact non-relativistic hydrogenic scaling
``q_Z(T) = Z**-3 q_H(T / Z**2)``.  Reverse rates are formed by detailed
balance against the same non-ideal He II reference populations used by the
opacity calculation.

The restricted ion-stage scope is deliberate.  A coupled LS-resolved He I +
He II + He III atom is the next layer; keeping this smaller block separately
testable gives it a strict Planck-field LTE regression and makes mistakes in
the ion/continuum coupling much easier to isolate.
"""

from __future__ import annotations
from ._nlte_radiative_integrals import continuum_integrals

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from ._compat import trapezoid
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .eos import helium_occupation_probability
from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HELIUM_FIRST_IONIZATION_ENERGY,
    HELIUM_SECOND_IONIZATION_ENERGY,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .helium import (
    _HE_I_P_TO_S_LINE_KEYS,
    helium_i_rydberg_bound_free_linear_absorption_coefficient,
    HELIUM_I_LINES,
    HELIUM_I_RESONANCE_LINES,
    HELIUM_II_LINES,
    HeliumIILine,
    _helium_ii_level_distribution,
    _helium_i_op_shell_cross_section,
    helium_continuum_mass_absorption_coefficient,
    helium_i_line_mass_absorption_coefficient,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
    helium_rayleigh_scattering_mass_coefficient,
    neutral_helium_photoionization_cross_section,
)
from .helium_ii_stark import HeliumIIStarkTable, read_helium_ii_stark_table
from .helium_collisions import _TERM_FINE_INDEX, TlustyHeliumCollisionData
from .helium_i_atom import HeliumIAtom
from .helium_stark import HeliumStarkTable
from .multilevel_nlte import (
    HydrogenElectronCollisionData,
    _ContinuumTransferProblem,
    _LineTransferProblem,
    _anderson_log_population_update,
    _bound_bound_radiative_rates,
    _continuum_radiation_field,
    _departure_line_factors,
    _exponential_integral_e1,
    _excitation_rate_coefficient,
    _extended_hydrogen_shell_oscillator_strength,
    _finite_population_ratio,
    _ionization_rate_coefficient,
    _line_grid,
    _line_mean_intensity,
    _nlte_continuum_terms,
    _normalize_departure_coefficients,
    _planck_nu_at_wavelength,
    _validate_state_atmosphere,
    atmosphere_structure_fingerprint,
)
from .nlte_core import NLTETransferCoefficients, OpacitySpan, _cached_transfer, NonphysicalPopulationError
from .opacity import (
    HydrogenLine,
    electron_scattering_mass_coefficient,
    hydrogen_ground_state_photoionization_cross_section,
    optical_depth_from_mass_opacity,
)
from .radiative_transfer import Backend, emergent_flux, radiation_field
from .gaunt import hydrogen_bound_free_gaunt_factor
from .spectrum import Spectrum, planck_lambda_angstrom


FloatArray = NDArray[np.float64]
_LIGHT_SPEED_ANGSTROM_PER_SECOND = LIGHT_SPEED * 1.0e8
_HELIUM_NUCLEAR_CHARGE = 2.0
HydrogenicHeliumCollisionModel = Literal["ccc-scaled", "tlusty-mihalas"]

# Coefficients copied from TLUSTY 200's COLHE routine.  The public He II
# model atom selects the Werner/Mihalas polynomial (ICOL=1) for the ground
# continuum and the original Mihalas--Heasley--Auer fits (ICOL=0) for the
# excited continua.  Keeping these constants here makes the alternative
# collision prescription independently testable instead of hiding it behind
# empirical rate multipliers.
_TLUSTY_HEII_IONIZATION_G0 = np.asarray(
    [7.339_952_1e-2, 1.725_286_7, 8.633_508_7], dtype=np.float64
)
_TLUSTY_HEII_IONIZATION_G1 = np.asarray(
    [1.459_276_3e-7, 2.094_411_7e-6, 2.757_554_4e-5], dtype=np.float64
)
_TLUSTY_HEII_IONIZATION_G2 = np.asarray(
    [7.662_129_9e5, 5.425_487_9e6, 6.639_551_9e6], dtype=np.float64
)
_TLUSTY_HEII_IONIZATION_G3 = np.asarray(
    [2.377_543_9e2, 2.217_789_1e3, 5.207_25e3], dtype=np.float64
)
_TLUSTY_HEII_WERNER_POLYNOMIAL = np.asarray(
    [
        [-8.5931587, 85.014091, 923.64099, 2018.6470, 1551.5061,
         -2327.4819, -10701.481, -27619.789, -41099.602, -61599.023],
        [9.3868790, -78.834488, -969.18451, -2243.1768, -2059.9768,
         1546.7107, 9834.3447, 27067.436, 41421.254, 63594.133],
        [-4.0027571, 28.360615, 401.23965, 983.83374, 1051.4103,
         -204.82320, -3335.4211, -10100.119, -15863.257, -24949.125],
        [0.83941799, -4.7963457, -81.122566, -209.86169, -251.30855,
         -43.175175, 530.37292, 1826.1049, 2941.6460, 4740.8364],
        [-0.086396709, 0.37385577, 8.0078983, 21.757591, 28.375637,
         11.890312, -39.536087, -161.52513, -266.86011, -440.88257],
        [0.0034853835, -0.010401310, -0.30957383, -0.87988985,
         -1.2254572, -0.72724497, 1.0879648, 5.6239786, 9.5323009,
         16.150818],
    ],
    dtype=np.float64,
)

# Public TLUSTY 14-level He I atom (he1_14lev.dat).  The first column is the
# threshold frequency from each term to He II.  Levels 6--11 are separate
# singlet/triplet n=3--5 superlevels; n=6--8 are multiplicity-combined.
HELIUM_I_14_THRESHOLD_FREQUENCY_HZ = np.asarray(
    [
        5.94503520e15,
        1.15267210e15,
        9.60145430e14,
        8.75933720e14,
        8.14536220e14,
        3.80746460e14,
        3.68828050e14,
        2.08996590e14,
        2.06223510e14,
        1.31522000e14,
        1.31522000e14,
        9.13347220e13,
        6.71030610e13,
        5.13757810e13,
    ],
    dtype=np.float64,
)
HELIUM_I_14_STATISTICAL_WEIGHT = np.asarray(
    [1, 3, 1, 9, 3, 27, 9, 48, 16, 75, 25, 144, 196, 256],
    dtype=np.float64,
)
HELIUM_I_14_PRINCIPAL_QUANTUM_NUMBER = np.asarray(
    [1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    dtype=np.float64,
)
HELIUM_I_14_LABEL = (
    "1s2 1S",
    "2s 3S",
    "2s 1S",
    "2p 3P",
    "2p 1P",
    "n=3 triplet",
    "n=3 singlet",
    "n=4 triplet",
    "n=4 singlet",
    "n=5 triplet",
    "n=5 singlet",
    "n=6 combined",
    "n=7 combined",
    "n=8 combined",
)
HELIUM_I_14_OSCILLATOR_STRENGTH: dict[tuple[int, int], float] = {
    (1, 5): 0.2762,
    (1, 7): 0.0734,
    (1, 9): 0.0302,
    (1, 11): 0.0153,
    (1, 12): 0.00848,
    (1, 13): 0.00593,
    (1, 14): 0.00399,
    (2, 4): 0.5391,
    (2, 6): 0.06446,
    (2, 8): 0.0231,
    (2, 10): 0.0114,
    (2, 12): 0.00608,
    (2, 13): 0.00381,
    (2, 14): 0.00260,
    (3, 5): 0.3764,
    (3, 7): 0.1514,
    (3, 9): 0.0507,
    (3, 11): 0.0221,
    (3, 12): 0.0128,
    (3, 13): 0.0066,
    (3, 14): 0.0044,
    (4, 6): 0.6783,
    (4, 8): 0.1368,
    (4, 10): 0.05105,
    (4, 12): 0.02326,
    (4, 13): 0.01520,
    (4, 14): 0.009341,
    (5, 7): 0.7590,
    (5, 9): 0.12034,
    (5, 11): 0.04688,
    (5, 12): 0.02283,
    (5, 13): 0.01208,
    (5, 14): 0.008056,
}


@dataclass(frozen=True)
class HydrogenicHeliumNLTEState:
    """Depth-dependent He II shell and He III populations.

    He I remains in the atmosphere's LTE EOS in this first restricted atom.
    The particle-conservation equation therefore applies to the active
    ``He II + He III`` reservoir only.
    """

    principal_quantum_number: FloatArray
    population_density: FloatArray
    doubly_ionized_he_density: FloatArray
    lte_population_density: FloatArray
    lte_doubly_ionized_he_density: FloatArray
    departure_coefficient: FloatArray
    continuum_departure_coefficient: FloatArray
    iterations: int
    converged: bool
    maximum_relative_population_change: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class NeutralHeliumNLTEState:
    """Populations of the TLUSTY 14-term He I atom and He II continuum."""

    term_label: tuple[str, ...]
    population_density: FloatArray
    singly_ionized_he_density: FloatArray
    lte_population_density: FloatArray
    lte_singly_ionized_he_density: FloatArray
    departure_coefficient: FloatArray
    continuum_departure_coefficient: FloatArray
    iterations: int
    converged: bool
    maximum_relative_population_change: float
    metadata: dict[str, object]
    helium_i_atom: HeliumIAtom | None = None  # None: TLUSTY 14-term atom


@dataclass(frozen=True)
class CoupledHeliumNLTEState:
    """One particle-conserving He I + He II + He III rate solution."""

    neutral_population_density: FloatArray
    singly_ionized_population_density: FloatArray
    doubly_ionized_he_density: FloatArray
    lte_neutral_population_density: FloatArray
    lte_singly_ionized_population_density: FloatArray
    lte_doubly_ionized_he_density: FloatArray
    neutral_departure_coefficient: FloatArray
    singly_ionized_departure_coefficient: FloatArray
    doubly_ionized_departure_coefficient: FloatArray
    iterations: int
    converged: bool
    maximum_relative_population_change: float
    metadata: dict[str, object]
    helium_i_atom: HeliumIAtom | None = None  # None: TLUSTY 14-term atom


def helium_ii_shell_transition(lower_level: int, upper_level: int) -> HydrogenLine:
    """Return a shell-averaged He II transition usable by the rate solver."""

    if not 1 <= lower_level < upper_level:
        raise ValueError("He II levels must satisfy 1 <= lower < upper")
    matching = next(
        (
            line
            for line in HELIUM_II_LINES
            if line.lower_principal_quantum_number == lower_level
            and line.upper_principal_quantum_number == upper_level
        ),
        None,
    )
    if matching is None:
        oscillator_strength = _extended_hydrogen_shell_oscillator_strength(
            lower_level, upper_level
        )
        transition_energy = HELIUM_SECOND_IONIZATION_ENERGY * (
            1.0 / lower_level**2 - 1.0 / upper_level**2
        )
        wavelength = PLANCK * LIGHT_SPEED / transition_energy * 1.0e8
    else:
        oscillator_strength = matching.absorption_oscillator_strength
        wavelength = matching.wavelength_vacuum_angstrom
    return HydrogenLine(
        name=f"HeII{lower_level}-{upper_level}",
        lower_level=lower_level,
        upper_level=upper_level,
        wavelength_vacuum_angstrom=float(wavelength),
        absorption_oscillator_strength=float(oscillator_strength),
    )


def _neutral_helium_reference_populations(
    atmosphere: Atmosphere,
    helium_i_atom: HeliumIAtom | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    helium = atmosphere.helium_lte_state
    if helium is None:
        raise ValueError("a pure-helium atmosphere with helium_lte_state is required")
    atom = _helium_i_atom(helium_i_atom)
    binding_energy = PLANCK * atom.threshold_frequency_hz
    excitation_energy = np.maximum(
        HELIUM_FIRST_IONIZATION_ENERGY - binding_energy, 0.0
    )
    effective_level = np.sqrt(
        HYDROGEN_IONIZATION_ENERGY / np.maximum(binding_energy, 1.0e-99)
    )
    occupation = helium_occupation_probability(
        helium.neutral_he_density[:, np.newaxis],
        helium.electron_density[:, np.newaxis],
        atmosphere.temperature[:, np.newaxis],
        effective_level[np.newaxis, :],
        neutral_h_density=atmosphere.neutral_h_density[:, np.newaxis],
        neutral_radius_scale=helium.neutral_radius_scale,
        correlated_microfields=(helium.microfield_model == "qmhd"),
    )
    terms = (
        atom.statistical_weight[np.newaxis, :]
        * occupation
        * np.exp(
            -excitation_energy[np.newaxis, :]
            / (BOLTZMANN * atmosphere.temperature[:, np.newaxis])
        )
    )
    population = (
        helium.neutral_he_density[:, np.newaxis]
        * terms
        / helium.neutral_partition_function[:, np.newaxis]
    )
    tiny = np.finfo(np.float64).tiny
    return (
        np.maximum(np.ascontiguousarray(population), tiny),
        np.maximum(
            np.ascontiguousarray(helium.singly_ionized_he_density), tiny
        ),
        np.ascontiguousarray(occupation),
    )


def neutral_helium_term_photoionization_cross_section(
    term_index: int, frequency_hz: ArrayLike
) -> FloatArray:
    """Return the TLUSTY-14 He I term photoionization cross section."""

    if not 0 <= term_index < 14:
        raise ValueError("term_index must lie in [0, 13]")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency must be finite and positive")
    if term_index < 5:
        return neutral_helium_photoionization_cross_section(
            frequency, term_index
        )
    principal = int(HELIUM_I_14_PRINCIPAL_QUANTUM_NUMBER[term_index])
    if principal in (3, 4):
        multiplicity = 3 if term_index in (5, 7) else 1
        cross_section = _helium_i_op_shell_cross_section(
            frequency, principal, multiplicity=multiplicity
        )
    else:
        cross_section = 2.815e29 / (frequency**3 * principal**5)
    return np.where(
        frequency >= HELIUM_I_14_THRESHOLD_FREQUENCY_HZ[term_index],
        cross_section,
        0.0,
    )




# The coupled He I + He II + He III solvers take a ``helium_i_atom``; None is
# this TLUSTY 14-term atom, whose data and term mapping are unchanged.
HELIUM_I_14 = HeliumIAtom(
    name="TLUSTY 14-term He I (he1_14lev.dat)",
    threshold_frequency_hz=HELIUM_I_14_THRESHOLD_FREQUENCY_HZ,
    statistical_weight=HELIUM_I_14_STATISTICAL_WEIGHT,
    principal_quantum_number=HELIUM_I_14_PRINCIPAL_QUANTUM_NUMBER,
    multiplicity=np.asarray([1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 0, 0, 0]),
    orbital_angular_momentum=np.asarray([0, 0, 0, 1, 1] + [-1] * 9),
    label=HELIUM_I_14_LABEL,
    oscillator_strength=HELIUM_I_14_OSCILLATOR_STRENGTH,
    collision_fine_groups=_TERM_FINE_INDEX,
    photoionization=neutral_helium_term_photoionization_cross_section,
    source="wd_spectra.helium_nlte HELIUM_I_14_* constants",
)


def _helium_i_atom(helium_i_atom: HeliumIAtom | None) -> HeliumIAtom:
    return HELIUM_I_14 if helium_i_atom is None else helium_i_atom


def _helium_i_line_term_indices(component, atom: HeliumIAtom) -> tuple[int, int]:
    """Lower and upper model-atom terms of one ``HELIUM_I_LINES`` component."""
    lower = component.lower_term_index
    principal = component.upper_principal_quantum_number
    if atom is HELIUM_I_14:
        return lower, _neutral_helium_upper_term_index(principal, triplet=lower in (1, 3))
    # 2s terms (1, 2) go to np; 2p terms (3, 4) to ns or nd.
    orbital = 1 if lower in (1, 2) else (0 if component.table_key_angstrom in _HE_I_P_TO_S_LINE_KEYS else 2)
    return lower, atom.term_index(principal, 3 if lower in (1, 3) else 1, orbital)


def _helium_i_resonance_upper_index(component, atom: HeliumIAtom) -> int:
    principal = component.upper_principal_quantum_number
    if atom is HELIUM_I_14:
        return 4 if principal == 2 else _neutral_helium_upper_term_index(principal, triplet=False)
    return atom.term_index(principal, 1, 1)
def _neutral_helium_bound_bound_radiative_rates(
    atmosphere: Atmosphere,
    lower_index: int,
    upper_index: int,
    oscillator_strength: float,
    lte_population: FloatArray,
    occupation: FloatArray,
    mean_intensity_nu: FloatArray,
    helium_i_atom: HeliumIAtom | None = None,
) -> tuple[FloatArray, FloatArray]:
    atom = _helium_i_atom(helium_i_atom)
    lower_weight = atom.statistical_weight[lower_index]
    upper_weight = atom.statistical_weight[upper_index]
    frequency = (
        atom.threshold_frequency_hz[lower_index]
        - atom.threshold_frequency_hz[upper_index]
    )
    spontaneous = (
        8.0
        * PI**2
        * ELEMENTARY_CHARGE_ESU**2
        * frequency**2
        / (ELECTRON_MASS * LIGHT_SPEED**3)
        * lower_weight
        / upper_weight
        * oscillator_strength
    )
    stimulated_down = spontaneous * LIGHT_SPEED**2 / (
        2.0 * PLANCK * frequency**3
    )
    absorption = upper_weight / lower_weight * stimulated_down
    lower_survival = np.clip(occupation[:, lower_index], 0.0, 1.0)
    upper_survival = np.clip(occupation[:, upper_index], 0.0, 1.0)
    planck_nu = (
        2.0
        * PLANCK
        * frequency**3
        / LIGHT_SPEED**2
        / np.expm1(
            PLANCK * frequency / (BOLTZMANN * atmosphere.temperature)
        )
    )
    upward = upper_survival * absorption * mean_intensity_nu
    physical_downward = lower_survival * (
        spontaneous + stimulated_down * mean_intensity_nu
    )
    lte_upward = upper_survival * absorption * planck_nu
    normalization = np.divide(
        _finite_population_ratio(
            lte_population[:, lower_index],
            lte_population[:, upper_index],
        )
        * lte_upward,
        lower_survival * (spontaneous + stimulated_down * planck_nu),
        out=np.ones_like(planck_nu),
        where=physical_downward > 0.0,
    )
    return np.asarray(upward), np.asarray(physical_downward * normalization)


def _neutral_helium_excitation_rate_coefficient(
    temperature: FloatArray,
    lower_index: int,
    upper_index: int,
    oscillator_strength: float,
    *,
    collision_strength_scale: float,
    helium_i_atom: HeliumIAtom | None = None,
) -> FloatArray:
    """Van-Regemorter-style closure for missing He I collision data.

    The effective collision strength scales as Ry / delta-E, not its
    square. With the rate conversion below this is the T**(-3/2) / u
    dependence of TLUSTY's CREGER (at the same approximate Gaunt factor).
    """

    atom = _helium_i_atom(helium_i_atom)
    energy = PLANCK * (
        atom.threshold_frequency_hz[lower_index]
        - atom.threshold_frequency_hz[upper_index]
    )
    reduced = energy / (BOLTZMANN * temperature)
    lower_weight = atom.statistical_weight[lower_index]
    if oscillator_strength > 0.0:
        gaunt = np.clip(0.2 + 0.28 * np.log1p(1.0 / reduced), 0.2, 5.0)
        effective_collision_strength = (
            14.5
            * oscillator_strength
            * lower_weight
            * (HYDROGEN_IONIZATION_ENERGY / energy)
            * gaunt
        )
    else:
        # TLUSTY marks these pairs as collision-only.  A small constant
        # effective strength is a conventional provisional closure.
        effective_collision_strength = np.full_like(temperature, 0.1)
    return (
        collision_strength_scale
        * 8.629e-6
        * effective_collision_strength
        / (lower_weight * np.sqrt(temperature))
        * np.exp(-np.minimum(reduced, 745.0))
    )


def _neutral_helium_continuum_radiative_rates(
    atmosphere: Atmosphere,
    lte_population: FloatArray,
    lte_continuum: FloatArray,
    wavelength_angstrom: FloatArray,
    mean_intensity_lambda: FloatArray,
    helium_i_atom: HeliumIAtom | None = None,
) -> tuple[FloatArray, FloatArray]:
    atom = _helium_i_atom(helium_i_atom)
    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    mean_lambda = np.asarray(mean_intensity_lambda, dtype=np.float64)
    if mean_lambda.shape != (wavelength.size, atmosphere.n_depth):
        raise ValueError("continuum mean intensity has the wrong shape")
    upward, recombination = continuum_integrals(
        wavelength, atmosphere.temperature, mean_lambda, atom.n_terms, atom.photoionization, first_level=0)
    downward = recombination * _finite_population_ratio(lte_population, lte_continuum[:,None])
    return upward, downward


def solve_neutral_helium_statistical_equilibrium(
    atmosphere: Atmosphere,
    *,
    line_mean_intensity_nu: Mapping[tuple[int, int], ArrayLike] | None = None,
    continuum_wavelength_angstrom: ArrayLike | None = None,
    continuum_mean_intensity_lambda: ArrayLike | None = None,
    collision_strength_scale: float = 1.0,
    helium_i_collision_data: TlustyHeliumCollisionData | None = None,
    fix_continuum_departure: bool = False,
) -> NeutralHeliumNLTEState:
    """Solve the TLUSTY 14-term He I atom plus its He II continuum."""

    if not np.isfinite(collision_strength_scale) or collision_strength_scale < 0.0:
        raise ValueError("collision_strength_scale must be finite and non-negative")
    lte_population, lte_continuum, occupation = (
        _neutral_helium_reference_populations(atmosphere)
    )
    n_depth = atmosphere.n_depth
    n_bound = 14
    n_state = n_bound + 1
    rate = np.zeros((n_depth, n_state, n_state), dtype=np.float64)
    fitted_collision_rate = (
        None
        if helium_i_collision_data is None
        else helium_i_collision_data.term_rate_matrix(atmosphere.temperature)
    )
    supplied_fields = {} if line_mean_intensity_nu is None else line_mean_intensity_nu
    for lower in range(n_bound - 1):
        for upper in range(lower + 1, n_bound):
            oscillator_strength = HELIUM_I_14_OSCILLATOR_STRENGTH.get(
                (lower + 1, upper + 1), 0.0
            )
            frequency = (
                HELIUM_I_14_THRESHOLD_FREQUENCY_HZ[lower]
                - HELIUM_I_14_THRESHOLD_FREQUENCY_HZ[upper]
            )
            supplied = supplied_fields.get((lower + 1, upper + 1))
            if oscillator_strength <= 0.0:
                mean_intensity = np.zeros(n_depth)
            elif supplied is None:
                mean_intensity = (
                    2.0
                    * PLANCK
                    * frequency**3
                    / LIGHT_SPEED**2
                    / np.expm1(
                        PLANCK
                        * frequency
                        / (BOLTZMANN * atmosphere.temperature)
                    )
                )
            else:
                mean_intensity = np.asarray(supplied, dtype=np.float64)
                if mean_intensity.shape != (n_depth,):
                    raise ValueError("each He I line field must have one value per depth")
            if oscillator_strength > 0.0:
                radiative_up, radiative_down = (
                    _neutral_helium_bound_bound_radiative_rates(
                        atmosphere,
                        lower,
                        upper,
                        oscillator_strength,
                        lte_population,
                        occupation,
                        mean_intensity,
                    )
                )
            else:
                radiative_up = np.zeros(n_depth)
                radiative_down = np.zeros(n_depth)
            rate_coefficient = (
                fitted_collision_rate[:, lower, upper]
                * collision_strength_scale
                if fitted_collision_rate is not None and upper < 9
                else _neutral_helium_excitation_rate_coefficient(
                    atmosphere.temperature,
                    lower,
                    upper,
                    oscillator_strength,
                    collision_strength_scale=collision_strength_scale,
                )
            )
            collisional_up = (
                atmosphere.electron_density
                * rate_coefficient
                * np.clip(occupation[:, upper], 0.0, 1.0)
            )
            collisional_down = collisional_up * _finite_population_ratio(
                lte_population[:, lower], lte_population[:, upper]
            )
            rate[:, lower, upper] += radiative_up + collisional_up
            rate[:, upper, lower] += radiative_down + collisional_down

    if continuum_wavelength_angstrom is None:
        edges = 1.025 * _LIGHT_SPEED_ANGSTROM_PER_SECOND / HELIUM_I_14_THRESHOLD_FREQUENCY_HZ
        base = np.geomspace(100.0, float(np.max(edges)), 520)
        edge_points = np.ravel(
            np.column_stack((edges * (1.0 - 2.0e-5), edges, edges * (1.0 + 2.0e-5)))
        )
        continuum_wavelength = np.unique(np.r_[base, edge_points])
    else:
        continuum_wavelength = np.asarray(
            continuum_wavelength_angstrom, dtype=np.float64
        )
    if continuum_mean_intensity_lambda is None:
        continuum_mean = planck_lambda_angstrom(
            continuum_wavelength[:, np.newaxis],
            atmosphere.temperature[np.newaxis, :],
        )
    else:
        continuum_mean = np.asarray(
            continuum_mean_intensity_lambda, dtype=np.float64
        )
    photo_up, radiative_recombination = (
        _neutral_helium_continuum_radiative_rates(
            atmosphere,
            lte_population,
            lte_continuum,
            np.ascontiguousarray(continuum_wavelength),
            continuum_mean,
        )
    )
    continuum_index = n_bound
    binding_energy = PLANCK * HELIUM_I_14_THRESHOLD_FREQUENCY_HZ
    effective_level = np.sqrt(
        HYDROGEN_IONIZATION_ENERGY / binding_energy
    )
    fitted_ionization_rate = (
        helium_i_collision_data.ionization_rate_coefficient(
            atmosphere.temperature, binding_energy
        )
        if helium_i_collision_data is not None
        and helium_i_collision_data.ionization_scale is not None
        else None
    )
    for term in range(n_bound):
        if fitted_ionization_rate is not None:
            ionization_rate = fitted_ionization_rate[:, term]
        else:
            ionization_rate = (
                5.465e-11
                * np.sqrt(atmosphere.temperature)
                * effective_level[term] ** 3
                * np.exp(
                    -np.minimum(
                        binding_energy[term]
                        / (BOLTZMANN * atmosphere.temperature),
                        745.0,
                    )
                )
            )
        collisional_ionization = (
            atmosphere.electron_density
            * collision_strength_scale
            * ionization_rate
        )
        three_body = collisional_ionization * _finite_population_ratio(
            lte_population[:, term], lte_continuum
        )
        rate[:, term, continuum_index] += photo_up[:, term] + collisional_ionization
        rate[:, continuum_index, term] += radiative_recombination[:, term] + three_body

    reference = np.column_stack((lte_population, lte_continuum))
    active_density = np.sum(reference, axis=1)
    populations = np.empty_like(reference)
    for depth in range(n_depth):
        matrix = rate[depth].T.copy()
        matrix[np.diag_indices(n_state)] -= np.sum(rate[depth], axis=1)
        matrix *= reference[depth][np.newaxis, :]
        rhs = np.zeros(n_state)
        if fix_continuum_departure:
            matrix[-1] = 0.0
            matrix[-1, -1] = reference[depth, -1]
            rhs[-1] = reference[depth, -1]
        else:
            matrix[-1] = reference[depth]
            rhs[-1] = active_density[depth]
        scale = np.maximum(np.max(np.abs(matrix), axis=1), np.finfo(np.float64).tiny)
        try:
            departure = np.linalg.solve(matrix / scale[:, np.newaxis], rhs / scale)
        except np.linalg.LinAlgError:
            departure = np.linalg.lstsq(matrix / scale[:, np.newaxis], rhs / scale, rcond=None)[0]
        if np.any(~np.isfinite(departure)) or np.any(departure <= 0.0):
            raise NonphysicalPopulationError(f"non-physical He I rate solution at depth {depth}")
        populations[depth] = departure * reference[depth]
    if fix_continuum_departure:
        populations[:, -1] = lte_continuum
    else:
        populations[:, -1] = active_density - np.sum(populations[:, :-1], axis=1)
    if np.any(populations[:, -1] <= 0.0):
        raise NonphysicalPopulationError("He I normalization removed the He II continuum")
    departure = populations / reference
    return NeutralHeliumNLTEState(
        term_label=HELIUM_I_14_LABEL,
        population_density=np.ascontiguousarray(populations[:, :-1]),
        singly_ionized_he_density=np.ascontiguousarray(populations[:, -1]),
        lte_population_density=lte_population,
        lte_singly_ionized_he_density=lte_continuum,
        departure_coefficient=np.ascontiguousarray(departure[:, :-1]),
        continuum_departure_coefficient=np.ascontiguousarray(departure[:, -1]),
        iterations=1,
        converged=True,
        maximum_relative_population_change=float(np.max(np.abs(departure - 1.0))),
        metadata={
            "model_atom": "TLUSTY 14-term He I plus He II continuum",
            "atomic_data": "public TLUSTY he1_14lev.dat architecture",
            "photoionization": "Opacity Project/Seaton-Fernley plus hydrogenic Rydberg",
            "collisions": (
                "Storey-Hummer/Berrington-Kingston excitation through n=4; "
                "TLUSTY/Mihalas ionization; provisional high-level excitation"
                if helium_i_collision_data is not None
                and helium_i_collision_data.ionization_scale is not None
                else "Storey-Hummer/Berrington-Kingston excitation through n=4; "
                "provisional high-level excitation and Seaton ionization"
                if helium_i_collision_data is not None
                else "provisional Van Regemorter/constant-strength/Seaton closure"
            ),
            "helium_i_collision_source": (
                None
                if helium_i_collision_data is None
                else helium_i_collision_data.source_path
            ),
            "helium_i_collision_extrapolation": (
                None if helium_i_collision_data is None else
                "constant effective collision strength outside 1000--50000 K; kinetic factors at actual temperature"
            ),
            "collision_strength_scale": float(collision_strength_scale),
            "continuum_population_closure": (
                "diagnostic He II departure fixed to unity"
                if fix_continuum_departure
                else "active explicit He I plus He II conservation"
            ),
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(atmosphere),
        },
    )


def solve_coupled_helium_statistical_equilibrium(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    maximum_helium_ii_level: int = 8,
    neutral_line_mean_intensity_nu: Mapping[tuple[int, int], ArrayLike] | None = None,
    helium_ii_line_mean_intensity_nu: Mapping[tuple[int, int], ArrayLike] | None = None,
    neutral_continuum_wavelength_angstrom: ArrayLike | None = None,
    neutral_continuum_mean_intensity_lambda: ArrayLike | None = None,
    helium_ii_continuum_wavelength_angstrom: ArrayLike | None = None,
    helium_ii_continuum_mean_intensity_lambda: ArrayLike | None = None,
    neutral_collision_strength_scale: float = 1.0,
    helium_i_collision_data: TlustyHeliumCollisionData | None = None,
    hydrogenic_collision_model: HydrogenicHeliumCollisionModel = "ccc-scaled",
    hydrogenic_collision_rate_multiplier: float = 1.0,
    hydrogenic_excitation_collision_rate_multiplier: float | None = None,
    hydrogenic_ionization_collision_rate_multiplier: float | None = None,
    _rate_matrix=None,
    _return_rate_matrix=False,
    helium_i_atom: HeliumIAtom | None = None,
    conservation_row: str = "last",
) -> CoupledHeliumNLTEState:
    """Solve a single He I + He II + He III statistical-equilibrium matrix.

    With no supplied radiation fields, local Planck fields are used and the
    solution is an exact LTE regression.  He I photoionization/recombination
    connects to the explicit He II ground shell—not a duplicated aggregate
    continuum—so all three ion stages share one particle-conservation row.
    """

    if not 2 <= maximum_helium_ii_level <= 32:
        raise ValueError("maximum_helium_ii_level must lie in [2, 32]")
    # "last" replaces the He III equation by particle conservation (the
    # historical choice, natural where He III dominates).  "dominant" replaces
    # the equation of the most populous state instead, so a trace He III or
    # high He II level keeps its own balance equation; in cool sdB surface
    # layers (He III/He ~ 1e-18) the "last" system is singular to roundoff.
    if conservation_row not in ("last", "dominant"):
        raise ValueError("conservation_row must be 'last' or 'dominant'")
    excitation_collision_multiplier = (
        hydrogenic_collision_rate_multiplier
        if hydrogenic_excitation_collision_rate_multiplier is None
        else hydrogenic_excitation_collision_rate_multiplier
    )
    ionization_collision_multiplier = (
        hydrogenic_collision_rate_multiplier
        if hydrogenic_ionization_collision_rate_multiplier is None
        else hydrogenic_ionization_collision_rate_multiplier
    )
    if (
        not np.isfinite(excitation_collision_multiplier)
        or excitation_collision_multiplier <= 0.0
        or not np.isfinite(ionization_collision_multiplier)
        or ionization_collision_multiplier <= 0.0
    ):
        raise ValueError("hydrogenic collision multipliers must be finite and positive")
    atom = _helium_i_atom(helium_i_atom)
    neutral_lte, helium_ii_total_lte, neutral_occupation = (
        _neutral_helium_reference_populations(atmosphere, atom)
    )
    helium_ii_lte, helium_iii_lte, helium_ii_occupation = (
        _reference_populations(atmosphere, maximum_helium_ii_level)
    )
    n_depth = atmosphere.n_depth
    n_neutral = atom.n_terms
    n_ion = maximum_helium_ii_level
    ion_start = n_neutral
    continuum_index = n_neutral + n_ion
    n_state = continuum_index + 1
    neutral_fields = (
        {} if neutral_line_mean_intensity_nu is None else neutral_line_mean_intensity_nu
    )
    ion_fields = (
        {} if helium_ii_line_mean_intensity_nu is None else helium_ii_line_mean_intensity_nu
    )
    if _rate_matrix is None:
        rate = np.zeros((n_depth, n_state, n_state), dtype=np.float64)
        fitted_neutral_collision_rate = (
            None
            if helium_i_collision_data is None
            else helium_i_collision_data.term_rate_matrix(
                atmosphere.temperature, atom.collision_fine_groups, atom.n_terms
            )
        )

        for lower in range(n_neutral - 1):
            for upper in range(lower + 1, n_neutral):
                oscillator_strength = atom.oscillator_strength.get(
                    (lower + 1, upper + 1), 0.0
                )
                frequency = (
                    atom.threshold_frequency_hz[lower]
                    - atom.threshold_frequency_hz[upper]
                )
                supplied = neutral_fields.get((lower + 1, upper + 1))
                if oscillator_strength <= 0.0:
                    mean_intensity = np.zeros(n_depth)
                elif supplied is None:
                    mean_intensity = (
                        2.0
                        * PLANCK
                        * frequency**3
                        / LIGHT_SPEED**2
                        / np.expm1(
                            PLANCK * frequency
                            / (BOLTZMANN * atmosphere.temperature)
                        )
                    )
                else:
                    mean_intensity = np.asarray(supplied, dtype=np.float64)
                if oscillator_strength > 0.0:
                    radiative_up, radiative_down = (
                        _neutral_helium_bound_bound_radiative_rates(
                            atmosphere,
                            lower,
                            upper,
                            oscillator_strength,
                            neutral_lte,
                            neutral_occupation,
                            mean_intensity,
                            atom,
                        )
                    )
                else:
                    radiative_up = np.zeros(n_depth)
                    radiative_down = np.zeros(n_depth)
                rate_coefficient = (
                    fitted_neutral_collision_rate[:, lower, upper]
                    * neutral_collision_strength_scale
                    if fitted_neutral_collision_rate is not None
                    and upper < atom.n_fitted_collision_terms
                    else _neutral_helium_excitation_rate_coefficient(
                        atmosphere.temperature,
                        lower,
                        upper,
                        oscillator_strength,
                        collision_strength_scale=neutral_collision_strength_scale,
                        helium_i_atom=atom,
                    )
                )
                collisional_up = (
                    atmosphere.electron_density
                    * rate_coefficient
                    * np.clip(neutral_occupation[:, upper], 0.0, 1.0)
                )
                collisional_down = collisional_up * _finite_population_ratio(
                    neutral_lte[:, lower], neutral_lte[:, upper]
                )
                rate[:, lower, upper] += radiative_up + collisional_up
                rate[:, upper, lower] += radiative_down + collisional_down

        for lower in range(1, n_ion):
            for upper in range(lower + 1, n_ion + 1):
                line = helium_ii_shell_transition(lower, upper)
                supplied = ion_fields.get((lower, upper))
                if supplied is None:
                    mean_intensity = _planck_nu_at_wavelength(
                        line.wavelength_vacuum_angstrom,
                        atmosphere.temperature,
                    )
                else:
                    mean_intensity = np.asarray(supplied, dtype=np.float64)
                radiative_up, radiative_down = _bound_bound_radiative_rates(
                    atmosphere.temperature,
                    line,
                    helium_ii_lte[:, lower - 1],
                    helium_ii_lte[:, upper - 1],
                    helium_ii_occupation[:, lower - 1],
                    helium_ii_occupation[:, upper - 1],
                    mean_intensity,
                )
                collisional_up = (
                    excitation_collision_multiplier
                    * atmosphere.electron_density
                    * helium_ii_excitation_collision_rate_coefficient(
                        collision_data,
                        atmosphere.temperature,
                        lower,
                        upper,
                        model=hydrogenic_collision_model,
                    )
                    * np.clip(helium_ii_occupation[:, upper - 1], 0.0, 1.0)
                )
                collisional_down = collisional_up * _finite_population_ratio(
                    helium_ii_lte[:, lower - 1],
                    helium_ii_lte[:, upper - 1],
                )
                lower_index = ion_start + lower - 1
                upper_index = ion_start + upper - 1
                rate[:, lower_index, upper_index] += radiative_up + collisional_up
                rate[:, upper_index, lower_index] += radiative_down + collisional_down

        neutral_wavelength = (
            default_neutral_helium_continuum_wavelength(atom)
            if neutral_continuum_wavelength_angstrom is None
            else np.asarray(neutral_continuum_wavelength_angstrom, dtype=np.float64)
        )
        neutral_mean = (
            planck_lambda_angstrom(
                neutral_wavelength[:, np.newaxis],
                atmosphere.temperature[np.newaxis, :],
            )
            if neutral_continuum_mean_intensity_lambda is None
            else np.asarray(neutral_continuum_mean_intensity_lambda, dtype=np.float64)
        )
        neutral_photo_up, neutral_recombination_to_total_ion = (
            _neutral_helium_continuum_radiative_rates(
                atmosphere,
                neutral_lte,
                helium_ii_total_lte,
                np.ascontiguousarray(neutral_wavelength),
                neutral_mean,
                atom,
            )
        )
        ion_ground_index = ion_start
        ground_fraction = helium_ii_lte[:, 0] / np.maximum(
            helium_ii_total_lte, np.finfo(np.float64).tiny
        )
        binding_energy = PLANCK * atom.threshold_frequency_hz
        effective_level = np.sqrt(HYDROGEN_IONIZATION_ENERGY / binding_energy)
        fitted_neutral_ionization_rate = (
            helium_i_collision_data.ionization_rate_coefficient(
                atmosphere.temperature, binding_energy
            )
            if helium_i_collision_data is not None
            and helium_i_collision_data.ionization_scale is not None
            else None
        )
        for term in range(n_neutral):
            if fitted_neutral_ionization_rate is not None:
                ionization_rate = fitted_neutral_ionization_rate[:, term]
            else:
                ionization_rate = (
                    5.465e-11
                    * np.sqrt(atmosphere.temperature)
                    * effective_level[term] ** 3
                    * np.exp(
                        -np.minimum(
                            binding_energy[term]
                            / (BOLTZMANN * atmosphere.temperature),
                            745.0,
                        )
                    )
                )
            collisional_ionization = (
                atmosphere.electron_density
                * neutral_collision_strength_scale
                * ionization_rate
            )
            # Convert the old rate per total He II ion to a rate per explicit
            # ground-shell ion while preserving LTE detailed balance.
            radiative_recombination = (
                neutral_recombination_to_total_ion[:, term]
                / np.maximum(ground_fraction, np.finfo(np.float64).tiny)
            )
            three_body = collisional_ionization * _finite_population_ratio(
                neutral_lte[:, term], helium_ii_lte[:, 0]
            )
            rate[:, term, ion_ground_index] += (
                neutral_photo_up[:, term] + collisional_ionization
            )
            rate[:, ion_ground_index, term] += radiative_recombination + three_body

        ion_wavelength = (
            default_helium_ii_continuum_wavelength(maximum_helium_ii_level)
            if helium_ii_continuum_wavelength_angstrom is None
            else np.asarray(helium_ii_continuum_wavelength_angstrom, dtype=np.float64)
        )
        ion_mean = (
            planck_lambda_angstrom(
                ion_wavelength[:, np.newaxis],
                atmosphere.temperature[np.newaxis, :],
            )
            if helium_ii_continuum_mean_intensity_lambda is None
            else np.asarray(helium_ii_continuum_mean_intensity_lambda, dtype=np.float64)
        )
        ion_photo_up, ion_radiative_recombination = _continuum_radiative_rates(
            atmosphere,
            helium_ii_lte,
            helium_iii_lte,
            np.ascontiguousarray(ion_wavelength),
            ion_mean,
        )
        for level in range(1, n_ion + 1):
            collisional_ionization = (
                ionization_collision_multiplier
                * atmosphere.electron_density
                * helium_ii_ionization_collision_rate_coefficient(
                    collision_data,
                    atmosphere.temperature,
                    level,
                    model=hydrogenic_collision_model,
                )
            )
            three_body = collisional_ionization * _finite_population_ratio(
                helium_ii_lte[:, level - 1], helium_iii_lte
            )
            level_index = ion_start + level - 1
            rate[:, level_index, continuum_index] += (
                ion_photo_up[:, level - 1] + collisional_ionization
            )
            rate[:, continuum_index, level_index] += (
                ion_radiative_recombination[:, level - 1] + three_body
            )

    else:
        rate = np.asarray(_rate_matrix, dtype=np.float64)
        if rate.shape != (n_depth, n_state, n_state) or np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError(f"invalid prepared helium rate matrix: minimum {np.min(rate):.6g} at {np.unravel_index(np.argmin(rate),rate.shape)}")
    if _return_rate_matrix:
        return rate

    reference = np.column_stack((neutral_lte, helium_ii_lte, helium_iii_lte))
    active_density = np.sum(reference, axis=1)
    populations = np.empty_like(reference)
    for depth in range(n_depth):
        matrix = rate[depth].T.copy()
        matrix[np.diag_indices(n_state)] -= np.sum(rate[depth], axis=1)
        matrix *= reference[depth][np.newaxis, :]
        row = -1 if conservation_row == "last" else int(np.argmax(reference[depth]))
        matrix[row] = reference[depth]
        rhs = np.zeros(n_state)
        rhs[row] = active_density[depth]
        scale = np.maximum(
            np.max(np.abs(matrix), axis=1), np.finfo(np.float64).tiny
        )
        try:
            departure = np.linalg.solve(
                matrix / scale[:, np.newaxis], rhs / scale
            )
        except np.linalg.LinAlgError:
            departure = np.linalg.lstsq(
                matrix / scale[:, np.newaxis], rhs / scale, rcond=None
            )[0]
        if np.any(~np.isfinite(departure)) or np.any(departure <= 0.0):
            raise NonphysicalPopulationError(
                f"non-physical coupled helium rate solution at depth {depth}"
            )
        populations[depth] = departure * reference[depth]
    # Enforce particle conservation by a common scale factor.  Recovering
    # He III as the difference between the total and the other states loses
    # all precision when He III is a trace ion (e.g. ~1e-6 of He in sdB
    # atmospheres near 30 kK), and solver roundoff can then make it negative.
    # The conservation row is solved to roundoff (~1e-15); a larger defect
    # means the linear solution itself is invalid.
    total = np.sum(populations, axis=1)
    if np.any(~(np.abs(total / active_density - 1.0) <= 1.0e-8)):
        raise NonphysicalPopulationError(
            "coupled helium normalization: rate solution violates particle conservation")
    populations *= (active_density / total)[:, np.newaxis]
    departure = populations / reference
    return CoupledHeliumNLTEState(
        helium_i_atom=None if atom is HELIUM_I_14 else atom,
        neutral_population_density=np.ascontiguousarray(
            populations[:, :n_neutral]
        ),
        singly_ionized_population_density=np.ascontiguousarray(
            populations[:, ion_start:continuum_index]
        ),
        doubly_ionized_he_density=np.ascontiguousarray(
            populations[:, continuum_index]
        ),
        lte_neutral_population_density=neutral_lte,
        lte_singly_ionized_population_density=helium_ii_lte,
        lte_doubly_ionized_he_density=helium_iii_lte,
        neutral_departure_coefficient=np.ascontiguousarray(
            departure[:, :n_neutral]
        ),
        singly_ionized_departure_coefficient=np.ascontiguousarray(
            departure[:, ion_start:continuum_index]
        ),
        doubly_ionized_departure_coefficient=np.ascontiguousarray(
            departure[:, continuum_index]
        ),
        iterations=1,
        converged=True,
        maximum_relative_population_change=float(
            np.max(np.abs(departure - 1.0))
        ),
        metadata={
            "model_atom": (
                (f"TLUSTY-14 He I + hydrogenic He II n=1-{n_ion} + He III"
                 if atom is HELIUM_I_14 else
                 f"{atom.name} + hydrogenic He II n=1-{n_ion} + He III")
            ),
            "helium_ii_radiation_coupled_transition_count": len(
                ion_fields
            ),
            "rate_coupling": "single statistical-equilibrium matrix",
            "particle_conservation": "one shared He I + He II + He III row",
            "photoionization": "Opacity Project He I and hydrogenic He II",
            "neutral_collisions": (
                "Storey-Hummer/Berrington-Kingston excitation through n=4; "
                "TLUSTY/Mihalas ionization; provisional high-level excitation"
                if helium_i_collision_data is not None
                and helium_i_collision_data.ionization_scale is not None
                else "Storey-Hummer/Berrington-Kingston excitation through n=4; "
                "provisional high-level excitation and Seaton ionization"
                if helium_i_collision_data is not None
                else "provisional Van Regemorter/constant-strength/Seaton closure"
            ),
            "helium_i_collision_source": (
                None
                if helium_i_collision_data is None
                else helium_i_collision_data.source_path
            ),
            "hydrogenic_collisions": (
                "CCC with exact Z=2 scaling"
                if hydrogenic_collision_model == "ccc-scaled"
                else "TLUSTY COLHE Mihalas/Werner He II fits"
            ),
            "helium_i_collision_extrapolation": (
                None if helium_i_collision_data is None else
                "constant effective collision strength outside 1000--50000 K; kinetic factors at actual temperature"
            ),
            "hydrogenic_collision_model": hydrogenic_collision_model,
            "hydrogenic_excitation_collision_rate_multiplier": float(
                excitation_collision_multiplier
            ),
            "hydrogenic_ionization_collision_rate_multiplier": float(
                ionization_collision_multiplier
            ),
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(
                atmosphere
            ),
        },
    )


def default_neutral_helium_continuum_wavelength(
    helium_i_atom: HeliumIAtom | None = None,
) -> FloatArray:
    """Return a grid resolving every He I photoionization threshold of the atom."""

    edges = (
        _LIGHT_SPEED_ANGSTROM_PER_SECOND
        / _helium_i_atom(helium_i_atom).threshold_frequency_hz
    )
    base = np.geomspace(100.0, 1.025 * float(np.max(edges)), 520)
    edge_points = np.ravel(
        np.column_stack(
            (edges * (1.0 - 2.0e-5), edges, edges * (1.0 + 2.0e-5))
        )
    )
    return np.unique(np.r_[base, edge_points])


def _prepare_neutral_helium_continuum_transfer_problem(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lte_absorption: FloatArray | None = None,
    helium_i_atom: HeliumIAtom | None = None,
) -> _ContinuumTransferProblem:
    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("continuum wavelength must be positive and increasing")
    # ``lte_absorption`` may be supplied by a caller that builds both the
    # He I and He II problems on the same atmosphere and grid; it is exactly
    # this LTE continuum, so the value is unchanged.
    if lte_absorption is None:
        lte_absorption = helium_continuum_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            include_electron_scattering=False,
            include_rayleigh_scattering=False,
        )
    atom = _helium_i_atom(helium_i_atom)
    lte_population, _, _ = _neutral_helium_reference_populations(atmosphere, atom)
    frequency = _LIGHT_SPEED_ANGSTROM_PER_SECOND / wavelength
    exponent = (
        PLANCK
        * frequency[:, np.newaxis]
        / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
    )
    exp_minus = np.exp(-np.minimum(exponent, 745.0))
    coefficient = np.empty(
        (wavelength.size, atmosphere.n_depth, atom.n_terms), dtype=np.float64
    )
    for term in range(atom.n_terms):
        coefficient[:, :, term] = (
            atom.photoionization(term, frequency)[
                :, np.newaxis
            ]
            * lte_population[np.newaxis, :, term]
            / atmosphere.mass_density[np.newaxis, :]
        )
    if atom is HELIUM_I_14:
        lte_explicit_bound_free = np.sum(
            coefficient * (1.0 - exp_minus)[:, :, np.newaxis], axis=2
        )
    else:
        # A term-resolved atom replaces the EOS representation of the same
        # shells (its five n <= 2 terms and hydrogenic n = 3..n_max shells)
        # rather than its own LTE sum: resolved n >= 3 terms are more tightly
        # bound than the hydrogenic shells, so their explicit LTE opacity can
        # locally exceed the EOS shell share.  The background is then
        # non-negative by construction and J = B still recovers LTE.
        helium = atmosphere.helium_lte_state
        stimulated = 1.0 - exp_minus
        lte_explicit_bound_free = np.zeros_like(lte_absorption)
        for term_index in range(5):
            lte_explicit_bound_free += (
                helium.neutral_level_population_density[np.newaxis, :, term_index]
                * neutral_helium_photoionization_cross_section(frequency, term_index)[:, np.newaxis]
                * stimulated
            )
        lte_explicit_bound_free += helium_i_rydberg_bound_free_linear_absorption_coefficient(
            atmosphere, wavelength, minimum_principal_quantum_number=3,
            maximum_principal_quantum_number=int(np.max(atom.principal_quantum_number)),
        )
        lte_explicit_bound_free /= atmosphere.mass_density[np.newaxis, :]
    thermal_background = lte_absorption - lte_explicit_bound_free
    tolerance = 3.0e-12 * np.maximum(lte_absorption, 1.0e-40)
    if np.any(thermal_background < -tolerance):
        mismatch = float(
            np.min(
                thermal_background / np.maximum(lte_absorption, 1.0e-40)
            )
        )
        raise NonphysicalPopulationError(
            "He I bound-free decomposition is inconsistent with LTE "
            f"continuum opacity ({mismatch:.3e})"
        )
    thermal_background = np.maximum(thermal_background, 0.0)
    spontaneous_nu = 2.0 * PLANCK * frequency**3 / LIGHT_SPEED**2
    spontaneous_lambda = (
        spontaneous_nu * _LIGHT_SPEED_ANGSTROM_PER_SECOND / wavelength**2
    )
    scattering = (
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
        + helium_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
    )
    return _ContinuumTransferProblem(
        wavelength_angstrom=wavelength,
        thermal_background_absorption=np.ascontiguousarray(
            thermal_background
        ),
        bound_free_coefficient=np.ascontiguousarray(coefficient),
        exp_minus_photon_energy=np.ascontiguousarray(exp_minus),
        spontaneous_intensity_lambda=np.ascontiguousarray(
            spontaneous_lambda[:, np.newaxis]
        ),
        scattering=np.ascontiguousarray(scattering),
        planck_lambda=np.ascontiguousarray(
            planck_lambda_angstrom(
                wavelength[:, np.newaxis],
                atmosphere.temperature[np.newaxis, :],
            )
        ),
    )


def _neutral_helium_upper_term_index(
    upper_principal_quantum_number: int, *, triplet: bool
) -> int:
    principal = upper_principal_quantum_number
    if principal <= 5:
        return {3: (5, 6), 4: (7, 8), 5: (9, 10)}[principal][
            0 if triplet else 1
        ]
    return min(principal, 8) + 5


def _neutral_helium_line_components(
    atmosphere: Atmosphere,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_i_atom: HeliumIAtom | None = None,
) -> dict[tuple[int, int], tuple[_LineTransferProblem, ...]]:
    atom = _helium_i_atom(helium_i_atom)
    grouped: dict[tuple[int, int], list[_LineTransferProblem]] = {}
    for component in HELIUM_I_LINES:
        lower_index, upper_index = _helium_i_line_term_indices(component, atom)
        key = (lower_index + 1, upper_index + 1)
        if key not in atom.oscillator_strength:
            continue
        center = component.wavelength_vacuum_angstrom
        wavelength = np.unique(
            np.r_[
                np.linspace(center - 100.0, center + 100.0, 81),
                np.linspace(center - 6.0, center + 6.0, 81),
            ]
        )
        line = HydrogenLine(
            name=component.name,
            lower_level=1,
            upper_level=2,
            wavelength_vacuum_angstrom=center,
            absorption_oscillator_strength=(
                component.absorption_oscillator_strength
            ),
        )
        grouped.setdefault(key, []).append(
            _LineTransferProblem(
                line=line,
                continuum=_prepare_neutral_helium_continuum_transfer_problem(
                    atmosphere, wavelength, helium_i_atom=helium_i_atom
                ),
                lte_line_opacity=helium_i_line_mass_absorption_coefficient(
                    atmosphere,
                    wavelength,
                    helium_i_stark_table,
                    lines=(component,),
                ),
                lower_state_index=lower_index,
                upper_state_index=upper_index,
            )
        )
    for component in HELIUM_I_RESONANCE_LINES:
        lower_index = 0
        upper_index = _helium_i_resonance_upper_index(component, atom)
        key = (1, upper_index + 1)
        center = component.wavelength_vacuum_angstrom
        wavelength = np.unique(
            np.r_[
                np.linspace(center - 25.0, center + 25.0, 81),
                np.linspace(center - 2.0, center + 2.0, 81),
            ]
        )
        line = HydrogenLine(
            name=component.name,
            lower_level=1,
            upper_level=2,
            wavelength_vacuum_angstrom=center,
            absorption_oscillator_strength=(
                component.absorption_oscillator_strength
            ),
        )
        grouped.setdefault(key, []).append(
            _LineTransferProblem(
                line=line,
                continuum=_prepare_neutral_helium_continuum_transfer_problem(
                    atmosphere, wavelength, helium_i_atom=helium_i_atom
                ),
                lte_line_opacity=(
                    helium_i_resonance_line_mass_absorption_coefficient(
                        atmosphere, wavelength, lines=(component,)
                    )
                ),
                lower_state_index=lower_index,
                upper_state_index=upper_index,
            )
        )
    return {key: tuple(value) for key, value in grouped.items()}


def solve_neutral_helium_nlte(
    atmosphere: Atmosphere,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    n_angle: int = 3,
    maximum_iterations: int = 120,
    relative_tolerance: float = 5.0e-3,
    population_damping: float = 0.4,
    collision_strength_scale: float = 1.0,
    helium_i_collision_data: TlustyHeliumCollisionData | None = None,
    initial_departure_coefficient: ArrayLike | None = None,
    initial_continuum_departure_coefficient: ArrayLike | None = None,
) -> NeutralHeliumNLTEState:
    """Iterate the TLUSTY-14 He I atom with continuum and line transfer."""

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not 0.0 < population_damping <= 1.0:
        raise ValueError("population_damping must lie in (0, 1]")
    continuum_wavelength = default_neutral_helium_continuum_wavelength()
    continuum_problem = _prepare_neutral_helium_continuum_transfer_problem(
        atmosphere, continuum_wavelength
    )
    component_groups = _neutral_helium_line_components(
        atmosphere, helium_i_stark_table
    )
    lte_population, lte_continuum, _ = (
        _neutral_helium_reference_populations(atmosphere)
    )
    if (initial_departure_coefficient is None) != (
        initial_continuum_departure_coefficient is None
    ):
        raise ValueError("both warm-start departure arrays must be supplied")
    if initial_departure_coefficient is None:
        departure = np.ones_like(lte_population)
        continuum_departure = np.ones_like(lte_continuum)
        warm_started = False
    else:
        departure = np.asarray(initial_departure_coefficient, dtype=np.float64).copy()
        continuum_departure = np.asarray(initial_continuum_departure_coefficient, dtype=np.float64).copy()
        if departure.shape != lte_population.shape or continuum_departure.shape != lte_continuum.shape:
            raise ValueError("warm-start departures do not match the atom")
        departure, continuum_departure = _normalize_departure_coefficients(
            lte_population, lte_continuum, departure, continuum_departure
        )
        warm_started = True
    converged = False
    maximum_change = np.inf
    solution = None
    for iteration in range(1, maximum_iterations + 1):
        continuum_mean = _continuum_radiation_field(
            atmosphere,
            continuum_problem,
            departure,
            continuum_departure,
            n_angle=n_angle,
        )
        line_fields: dict[tuple[int, int], FloatArray] = {}
        for key, problems in component_groups.items():
            intensities = []
            strengths = []
            for problem in problems:
                intensities.append(
                    _line_mean_intensity(
                        atmosphere,
                        problem,
                        departure,
                        continuum_departure,
                        n_angle=n_angle,
                    )
                )
                strengths.append(problem.line.absorption_oscillator_strength)
            line_fields[key] = np.average(
                np.asarray(intensities), axis=0, weights=np.asarray(strengths)
            )
        candidate = solve_neutral_helium_statistical_equilibrium(
            atmosphere,
            line_mean_intensity_nu=line_fields,
            continuum_wavelength_angstrom=continuum_wavelength,
            continuum_mean_intensity_lambda=continuum_mean,
            collision_strength_scale=collision_strength_scale,
            helium_i_collision_data=helium_i_collision_data,
        )
        candidate_departure = candidate.departure_coefficient
        candidate_continuum = candidate.continuum_departure_coefficient
        maximum_change = float(
            max(
                np.max(np.abs(np.log(candidate_departure / departure))),
                np.max(np.abs(np.log(candidate_continuum / continuum_departure))),
            )
        )
        solution = candidate
        if maximum_change < relative_tolerance:
            departure = candidate_departure
            continuum_departure = candidate_continuum
            converged = True
            break
        departure = np.exp(
            (1.0 - population_damping) * np.log(departure)
            + population_damping * np.log(candidate_departure)
        )
        continuum_departure = np.exp(
            (1.0 - population_damping) * np.log(continuum_departure)
            + population_damping * np.log(candidate_continuum)
        )
        departure, continuum_departure = _normalize_departure_coefficients(
            lte_population, lte_continuum, departure, continuum_departure
        )
    assert solution is not None
    metadata = dict(solution.metadata)
    metadata.update(
        {
            "structure": "fixed LTE helium atmosphere and electron density",
            "population_transfer": "He I bound-free plus resolved He I lines",
            "warm_started": warm_started,
            "nlte_status": "He I/II restricted atom; He II excitation and He III separate",
        }
    )
    return replace(
        solution,
        population_density=np.ascontiguousarray(lte_population * departure),
        singly_ionized_he_density=np.ascontiguousarray(
            lte_continuum * continuum_departure
        ),
        departure_coefficient=np.ascontiguousarray(departure),
        continuum_departure_coefficient=np.ascontiguousarray(continuum_departure),
        iterations=iteration,
        converged=converged,
        maximum_relative_population_change=maximum_change,
        metadata=metadata,
    )


def neutral_helium_nlte_transfer_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    state: NeutralHeliumNLTEState,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
) -> NLTETransferCoefficients:
    """Assemble NLTE He I plus LTE He II/III transfer coefficients."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength must be positive and increasing")
    _validate_state_atmosphere(atmosphere, state)
    atom = _helium_i_atom(state.helium_i_atom)
    continuum_problem = _prepare_neutral_helium_continuum_transfer_problem(
        atmosphere, wavelength, helium_i_atom=state.helium_i_atom
    )
    absorption, emissivity = _nlte_continuum_terms(
        continuum_problem,
        state.departure_coefficient,
        state.continuum_departure_coefficient,
    )
    planck = continuum_problem.planck_lambda
    suppressed_inversions = 0
    if include_helium_i_lines:
        for component in HELIUM_I_LINES:
            lower_index, upper_index = _helium_i_line_term_indices(component, atom)
            line = HydrogenLine(
                component.name,
                1,
                2,
                component.wavelength_vacuum_angstrom,
                component.absorption_oscillator_strength,
            )
            opacity_factor, source_factor = _departure_line_factors(
                atmosphere,
                line,
                state.departure_coefficient,
                lower_state_index=lower_index,
                upper_state_index=upper_index,
                suppress_population_inversion=True,
            )
            suppressed_inversions += int(
                np.count_nonzero(opacity_factor == 0.0)
            )
            lte_opacity = helium_i_line_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                helium_i_stark_table,
                lines=(component,),
            )
            line_opacity = lte_opacity * opacity_factor[np.newaxis, :]
            absorption += line_opacity
            emissivity += line_opacity * planck * source_factor[np.newaxis, :]
    if include_helium_i_resonance_lines:
        for component in HELIUM_I_RESONANCE_LINES:
            upper_index = _helium_i_resonance_upper_index(component, atom)
            line = HydrogenLine(
                component.name,
                1,
                2,
                component.wavelength_vacuum_angstrom,
                component.absorption_oscillator_strength,
            )
            opacity_factor, source_factor = _departure_line_factors(
                atmosphere,
                line,
                state.departure_coefficient,
                lower_state_index=0,
                upper_state_index=upper_index,
                suppress_population_inversion=True,
            )
            suppressed_inversions += int(
                np.count_nonzero(opacity_factor == 0.0)
            )
            lte_opacity = helium_i_resonance_line_mass_absorption_coefficient(
                atmosphere, wavelength, lines=(component,)
            )
            line_opacity = lte_opacity * opacity_factor[np.newaxis, :]
            absorption += line_opacity
            emissivity += line_opacity * planck * source_factor[np.newaxis, :]
    if include_helium_ii_lines:
        helium_ii_opacity = helium_ii_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            stark_table=helium_ii_stark_table,
        )
        absorption += helium_ii_opacity
        emissivity += helium_ii_opacity * planck
    return NLTETransferCoefficients(
        wavelength_angstrom=wavelength,
        true_absorption=np.ascontiguousarray(absorption),
        thermal_emissivity=np.ascontiguousarray(emissivity),
        scattering=continuum_problem.scattering,
        metadata={
            "model_atom": state.metadata.get("model_atom"),
            "ion_stage_scope": "He I/II NLTE; He II excitation and He III LTE",
            "helium_i_lines": bool(include_helium_i_lines),
            "helium_i_resonance_lines": bool(
                include_helium_i_resonance_lines
            ),
            "helium_ii_lines_lte": bool(include_helium_ii_lines),
            "suppressed_line_inversions": suppressed_inversions,
        },
    )


def synthesize_neutral_helium_nlte_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    nlte_state: NeutralHeliumNLTEState | None = None,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize a restricted He I/II NLTE spectrum."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if nlte_state is None:
        nlte_state = solve_neutral_helium_nlte(
            atmosphere,
            helium_i_stark_table=helium_i_stark_table,
            n_angle=max(1, min(n_angle, 3)),
        )
    coefficients = neutral_helium_nlte_transfer_coefficients(
        atmosphere,
        wavelength,
        nlte_state,
        helium_i_stark_table=helium_i_stark_table,
        helium_ii_stark_table=helium_ii_stark_table,
        include_helium_i_lines=include_helium_i_lines,
        include_helium_i_resonance_lines=include_helium_i_resonance_lines,
        include_helium_ii_lines=include_helium_ii_lines,
    )
    total = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total
    )
    source = (
        coefficients.thermal_emissivity
        + coefficients.scattering
        * planck_lambda_angstrom(
            wavelength[:, np.newaxis],
            atmosphere.temperature[np.newaxis, :],
        )
    ) / total
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total
        )
    flux = emergent_flux(
        optical_depth, source, n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "composition": "pure-helium",
            "line_formation": "restricted He I/II NLTE; He II excitation/He III LTE",
            "model_atom": nlte_state.metadata.get("model_atom"),
            "population_iterations": nlte_state.iterations,
            "population_converged": nlte_state.converged,
            "maximum_log_population_change": (
                nlte_state.maximum_relative_population_change
            ),
            "n_angle": int(n_angle),
        },
    )



def coupled_helium_ii_lines(maximum_level):
    """Include every explicitly transferred rate transition in the opacity.

    The compact optical line list omits some UV high-shell links. A common
    field must contain their extinction too, as the legacy per-line solves did.
    """
    lines = [line for line in HELIUM_II_LINES
             if line.upper_principal_quantum_number <= maximum_level]
    pairs = {(line.lower_principal_quantum_number, line.upper_principal_quantum_number)
             for line in lines}
    for lower in range(1, min(3, maximum_level-1)+1):
        for upper in range(lower+1, maximum_level+1):
            if (lower, upper) not in pairs:
                transition = helium_ii_shell_transition(lower, upper)
                lines.append(HeliumIILine(
                    transition.name, lower, upper, transition.wavelength_vacuum_angstrom,
                    transition.absorption_oscillator_strength))
    return tuple(lines)


def _add_line_opacity(absorption, emissivity, planck, lte_opacity, opacity_factor, source_factor):
    """Add one NLTE-scaled line; compact spans update only their nonzero rows."""
    if isinstance(lte_opacity, OpacitySpan):
        rows = lte_opacity.rows
        line_opacity = lte_opacity.block * opacity_factor[np.newaxis, :]
        absorption[rows] += line_opacity
        emissivity[rows] += (
            line_opacity * planck[rows] * source_factor[np.newaxis, :]
        )
        return
    line_opacity = lte_opacity * opacity_factor[np.newaxis, :]
    absorption += line_opacity
    emissivity += line_opacity * planck * source_factor[np.newaxis, :]


def combined_helium_nlte_transfer_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    neutral_state: NeutralHeliumNLTEState,
    hydrogenic_state: HydrogenicHeliumNLTEState,
    *,
    _cache=None,
    _allow_signed_continuum=False,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    maximum_helium_ii_level: int = 8,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
) -> NLTETransferCoefficients:
    """Combine converged restricted He I/II and He II/III populations.

    This is an operator-split bridge to the forthcoming single coupled rate
    matrix.  Each ion's bound-free and bound-bound opacity uses its own
    departure coefficients; the small inconsistency between the independently
    solved shared He II reservoirs is reported in metadata.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if _cache is not None:
        _cache.validate(atmosphere, wavelength)
    _validate_state_atmosphere(atmosphere, neutral_state)
    _validate_state_atmosphere(atmosphere, hydrogenic_state)
    # Both continuum problems need the same LTE He continuum on this grid;
    # evaluate it at most once, and only if a problem is not already cached.
    shared_lte_continuum = []

    def lte_continuum():
        if not shared_lte_continuum:
            shared_lte_continuum.append(helium_continuum_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                include_electron_scattering=False,
                include_rayleigh_scattering=False,
            ))
        return shared_lte_continuum[0]

    atom = _helium_i_atom(neutral_state.helium_i_atom)
    he_i_problem = _cached_transfer(_cache, ('he-i-continuum', atom.n_terms), lambda: _prepare_neutral_helium_continuum_transfer_problem(
        atmosphere, wavelength, lte_absorption=lte_continuum(),
        helium_i_atom=neutral_state.helium_i_atom,
    ))
    absorption, emissivity = _nlte_continuum_terms(
        he_i_problem,
        neutral_state.departure_coefficient,
        neutral_state.continuum_departure_coefficient,
        allow_signed_absorption=_allow_signed_continuum,
    )
    maximum_helium_ii_level = min(
        maximum_helium_ii_level,
        hydrogenic_state.departure_coefficient.shape[1],
    )
    he_ii_problem = _cached_transfer(_cache, ('he-ii-continuum', maximum_helium_ii_level), lambda: _prepare_helium_continuum_transfer_problem(
        atmosphere, wavelength, maximum_helium_ii_level,
        lte_absorption=lte_continuum(),
    ))
    ones_bound = np.ones(
        (atmosphere.n_depth, maximum_helium_ii_level), dtype=np.float64
    )
    ones_continuum = np.ones(atmosphere.n_depth, dtype=np.float64)
    he_ii_lte_absorption, he_ii_lte_emissivity = _nlte_continuum_terms(
        he_ii_problem, ones_bound, ones_continuum
    )
    he_ii_nlte_absorption, he_ii_nlte_emissivity = _nlte_continuum_terms(
        he_ii_problem,
        hydrogenic_state.departure_coefficient,
        hydrogenic_state.continuum_departure_coefficient,
        allow_signed_absorption=_allow_signed_continuum,
    )
    absorption += he_ii_nlte_absorption - he_ii_lte_absorption
    emissivity += he_ii_nlte_emissivity - he_ii_lte_emissivity
    planck = he_i_problem.planck_lambda
    suppressed_inversions = 0
    for component in HELIUM_I_LINES if include_helium_i_lines else ():
        lower_index, upper_index = _helium_i_line_term_indices(component, atom)
        line = HydrogenLine(
            component.name,
            1,
            2,
            component.wavelength_vacuum_angstrom,
            component.absorption_oscillator_strength,
        )
        opacity_factor, source_factor = _departure_line_factors(
            atmosphere,
            line,
            neutral_state.departure_coefficient,
            lower_state_index=lower_index,
            upper_state_index=upper_index,
            suppress_population_inversion=True,
        )
        suppressed_inversions += int(np.count_nonzero(opacity_factor == 0.0))
        lte_opacity = _cached_transfer(_cache, ('he-i-line', component.name), lambda: helium_i_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            helium_i_stark_table,
            lines=(component,),
        ))
        _add_line_opacity(
            absorption, emissivity, planck, lte_opacity, opacity_factor, source_factor
        )
    for component in (
        HELIUM_I_RESONANCE_LINES if include_helium_i_resonance_lines else ()
    ):
        upper_index = _helium_i_resonance_upper_index(component, atom)
        line = HydrogenLine(
            component.name,
            1,
            2,
            component.wavelength_vacuum_angstrom,
            component.absorption_oscillator_strength,
        )
        opacity_factor, source_factor = _departure_line_factors(
            atmosphere,
            line,
            neutral_state.departure_coefficient,
            lower_state_index=0,
            upper_state_index=upper_index,
            suppress_population_inversion=True,
        )
        suppressed_inversions += int(np.count_nonzero(opacity_factor == 0.0))
        lte_opacity = _cached_transfer(_cache, ('he-i-resonance', component.name), lambda: helium_i_resonance_line_mass_absorption_coefficient(
            atmosphere, wavelength, lines=(component,)
        ))
        _add_line_opacity(
            absorption, emissivity, planck, lte_opacity, opacity_factor, source_factor
        )
    for helium_line in coupled_helium_ii_lines(maximum_helium_ii_level) if include_helium_ii_lines else ():
        lower = helium_line.lower_principal_quantum_number
        upper = helium_line.upper_principal_quantum_number
        if upper > maximum_helium_ii_level:
            continue
        line = helium_ii_shell_transition(lower, upper)
        opacity_factor, source_factor = _departure_line_factors(
            atmosphere,
            line,
            hydrogenic_state.departure_coefficient,
            suppress_population_inversion=True,
        )
        suppressed_inversions += int(np.count_nonzero(opacity_factor == 0.0))
        lte_opacity = _cached_transfer(_cache, ('he-ii-line', lower, upper), lambda: helium_ii_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=(helium_line,),
            stark_table=helium_ii_stark_table,
        ))
        _add_line_opacity(
            absorption, emissivity, planck, lte_opacity, opacity_factor, source_factor
        )
    shared_he_ii_mismatch = np.max(
        np.abs(
            neutral_state.singly_ionized_he_density
            - np.sum(hydrogenic_state.population_density, axis=1)
        )
        / np.maximum(
            neutral_state.singly_ionized_he_density,
            np.finfo(np.float64).tiny,
        )
    )
    return NLTETransferCoefficients(
        wavelength_angstrom=wavelength,
        true_absorption=np.ascontiguousarray(absorption),
        thermal_emissivity=np.ascontiguousarray(emissivity),
        scattering=he_i_problem.scattering,
        metadata={
            "model_atom": "operator-split TLUSTY-14 He I + hydrogenic He II/III",
            "ion_stage_scope": "He I, He II, and He III NLTE transfer",
            "rate_coupling": "independent restricted atoms; coupled matrix pending",
            "shared_he_ii_maximum_relative_mismatch": float(
                shared_he_ii_mismatch
            ),
            "suppressed_line_inversions": suppressed_inversions,
            "helium_i_lines": bool(include_helium_i_lines),
            "helium_i_resonance_lines": bool(
                include_helium_i_resonance_lines
            ),
            "helium_ii_lines": bool(include_helium_ii_lines),
        },
    )


def synthesize_combined_helium_nlte_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    neutral_state: NeutralHeliumNLTEState,
    hydrogenic_state: HydrogenicHeliumNLTEState,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    maximum_helium_ii_level: int = 8,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize one spectrum using both restricted helium atoms."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    coefficients = combined_helium_nlte_transfer_coefficients(
        atmosphere,
        wavelength,
        neutral_state,
        hydrogenic_state,
        helium_i_stark_table=helium_i_stark_table,
        helium_ii_stark_table=helium_ii_stark_table,
        maximum_helium_ii_level=maximum_helium_ii_level,
    )
    total = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total
    )
    source = (
        coefficients.thermal_emissivity
        + coefficients.scattering
        * planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    ) / total
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total
        )
    flux = emergent_flux(
        optical_depth, source, n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            **coefficients.metadata,
            "composition": "pure-helium",
            "n_angle": int(n_angle),
        },
    )


def _coupled_state_views(
    atmosphere: Atmosphere, state: CoupledHeliumNLTEState
) -> tuple[NeutralHeliumNLTEState, HydrogenicHeliumNLTEState]:
    """Expose a coupled state through the two tested opacity interfaces."""

    helium = atmosphere.helium_lte_state
    if helium is None:
        raise ValueError("a pure-helium atmosphere is required")
    fingerprint = atmosphere_structure_fingerprint(atmosphere)
    he_ii_ground_departure = state.singly_ionized_departure_coefficient[:, 0]
    neutral_view = NeutralHeliumNLTEState(
        term_label=_helium_i_atom(state.helium_i_atom).label,
        helium_i_atom=state.helium_i_atom,
        population_density=state.neutral_population_density,
        singly_ionized_he_density=np.ascontiguousarray(
            helium.singly_ionized_he_density * he_ii_ground_departure
        ),
        lte_population_density=state.lte_neutral_population_density,
        lte_singly_ionized_he_density=np.ascontiguousarray(
            helium.singly_ionized_he_density
        ),
        departure_coefficient=state.neutral_departure_coefficient,
        continuum_departure_coefficient=np.ascontiguousarray(
            he_ii_ground_departure
        ),
        iterations=state.iterations,
        converged=state.converged,
        maximum_relative_population_change=state.maximum_relative_population_change,
        metadata={
            **state.metadata,
            "atmosphere_structure_sha256": fingerprint,
        },
    )
    hydrogenic_view = HydrogenicHeliumNLTEState(
        principal_quantum_number=np.arange(
            1,
            state.singly_ionized_departure_coefficient.shape[1] + 1,
            dtype=np.float64,
        ),
        population_density=state.singly_ionized_population_density,
        doubly_ionized_he_density=state.doubly_ionized_he_density,
        lte_population_density=state.lte_singly_ionized_population_density,
        lte_doubly_ionized_he_density=state.lte_doubly_ionized_he_density,
        departure_coefficient=state.singly_ionized_departure_coefficient,
        continuum_departure_coefficient=state.doubly_ionized_departure_coefficient,
        iterations=state.iterations,
        converged=state.converged,
        maximum_relative_population_change=state.maximum_relative_population_change,
        metadata={
            **state.metadata,
            "explicit_maximum_level": int(
                state.singly_ionized_departure_coefficient.shape[1]
            ),
            "atmosphere_structure_sha256": fingerprint,
        },
    )
    return neutral_view, hydrogenic_view


def coupled_helium_nlte_transfer_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    state: CoupledHeliumNLTEState,
    *,
    _cache=None,
    _allow_signed_continuum=False,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
) -> NLTETransferCoefficients:
    """Assemble transfer coefficients from the single coupled helium atom."""

    neutral_view, hydrogenic_view = _coupled_state_views(atmosphere, state)
    coefficients = combined_helium_nlte_transfer_coefficients(
        atmosphere,
        wavelength_angstrom,
        neutral_view,
        hydrogenic_view,
        _cache=_cache, _allow_signed_continuum=_allow_signed_continuum,
        helium_i_stark_table=helium_i_stark_table,
        helium_ii_stark_table=helium_ii_stark_table,
        maximum_helium_ii_level=(
            state.singly_ionized_departure_coefficient.shape[1]
        ),
        include_helium_i_lines=include_helium_i_lines,
        include_helium_i_resonance_lines=include_helium_i_resonance_lines,
        include_helium_ii_lines=include_helium_ii_lines,
    )
    metadata = dict(coefficients.metadata)
    metadata.update(
        {
            "model_atom": state.metadata.get("model_atom"),
            "rate_coupling": "single statistical-equilibrium matrix",
            "shared_he_ii_maximum_relative_mismatch": 0.0,
        }
    )
    return replace(coefficients, metadata=metadata)


def _coupled_continuum_mean_intensity(
    atmosphere: Atmosphere,
    wavelength: FloatArray,
    state: CoupledHeliumNLTEState,
    *,
    n_angle: int,
) -> FloatArray:
    """Formal continuum solution including both He I and He II departures."""

    neutral_view, hydrogenic_view = _coupled_state_views(atmosphere, state)
    lte_continuum = helium_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
    )
    he_i_problem = _prepare_neutral_helium_continuum_transfer_problem(
        atmosphere, wavelength, lte_absorption=lte_continuum
    )
    absorption, emissivity = _nlte_continuum_terms(
        he_i_problem,
        neutral_view.departure_coefficient,
        neutral_view.continuum_departure_coefficient,
    )
    n_ion = hydrogenic_view.departure_coefficient.shape[1]
    he_ii_problem = _prepare_helium_continuum_transfer_problem(
        atmosphere, wavelength, n_ion, lte_absorption=lte_continuum
    )
    lte_absorption, lte_emissivity = _nlte_continuum_terms(
        he_ii_problem,
        np.ones_like(hydrogenic_view.departure_coefficient),
        np.ones_like(hydrogenic_view.continuum_departure_coefficient),
    )
    nlte_absorption, nlte_emissivity = _nlte_continuum_terms(
        he_ii_problem,
        hydrogenic_view.departure_coefficient,
        hydrogenic_view.continuum_departure_coefficient,
    )
    absorption += nlte_absorption - lte_absorption
    emissivity += nlte_emissivity - lte_emissivity
    total = absorption + he_i_problem.scattering
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total
    )
    source = (
        emissivity + he_i_problem.scattering * he_i_problem.planck_lambda
    ) / total
    field = None
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (emissivity + he_i_problem.scattering * field.mean_intensity)
            / total
        )
    assert field is not None
    return field.mean_intensity


def _replace_coupled_departures(
    state: CoupledHeliumNLTEState,
    neutral_departure: FloatArray,
    ion_departure: FloatArray,
    continuum_departure: FloatArray,
    *,
    iterations: int,
    converged: bool,
    maximum_change: float,
    metadata: dict[str, object],
) -> CoupledHeliumNLTEState:
    return replace(
        state,
        neutral_population_density=np.ascontiguousarray(
            state.lte_neutral_population_density * neutral_departure
        ),
        singly_ionized_population_density=np.ascontiguousarray(
            state.lte_singly_ionized_population_density * ion_departure
        ),
        doubly_ionized_he_density=np.ascontiguousarray(
            state.lte_doubly_ionized_he_density * continuum_departure
        ),
        neutral_departure_coefficient=np.ascontiguousarray(
            neutral_departure
        ),
        singly_ionized_departure_coefficient=np.ascontiguousarray(
            ion_departure
        ),
        doubly_ionized_departure_coefficient=np.ascontiguousarray(
            continuum_departure
        ),
        iterations=iterations,
        converged=converged,
        maximum_relative_population_change=maximum_change,
        metadata=metadata,
    )


def _coupled_population_change(
    current: CoupledHeliumNLTEState,
    candidate: CoupledHeliumNLTEState,
    *,
    relative_population_floor: float,
) -> tuple[float, float]:
    """Return a physically weighted and a formal departure change.

    A departure coefficient can change by orders of magnitude when its LTE
    reference population is effectively zero.  That is a poor stopping
    criterion: it can keep an otherwise stationary radiation/population
    solution running indefinitely.  The first result is therefore the
    largest actual population change relative to the larger of the candidate
    population and a small fraction of the local helium reservoir.  The
    second result retains the unweighted log-departure diagnostic.
    """

    reference_total = (
        np.sum(candidate.lte_neutral_population_density, axis=1)
        + np.sum(candidate.lte_singly_ionized_population_density, axis=1)
        + candidate.lte_doubly_ionized_he_density
    )
    floor = relative_population_floor * reference_total
    weighted_change = 0.0
    formal_change = 0.0
    for old_population, new_population, old_departure, new_departure in (
        (
            current.neutral_population_density,
            candidate.neutral_population_density,
            current.neutral_departure_coefficient,
            candidate.neutral_departure_coefficient,
        ),
        (
            current.singly_ionized_population_density,
            candidate.singly_ionized_population_density,
            current.singly_ionized_departure_coefficient,
            candidate.singly_ionized_departure_coefficient,
        ),
        (
            current.doubly_ionized_he_density[:, np.newaxis],
            candidate.doubly_ionized_he_density[:, np.newaxis],
            current.doubly_ionized_departure_coefficient[:, np.newaxis],
            candidate.doubly_ionized_departure_coefficient[:, np.newaxis],
        ),
    ):
        scale = np.maximum(np.abs(new_population), floor[:, np.newaxis])
        weighted_change = max(
            weighted_change,
            float(np.max(np.abs(new_population - old_population) / scale)),
        )
        formal_change = max(
            formal_change,
            float(np.max(np.abs(np.log(new_departure / old_departure)))),
        )
    return weighted_change, formal_change


def solve_coupled_helium_nlte(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    maximum_helium_ii_level: int = 8,
    helium_ii_explicit_maximum_lower_level: int = 3,
    n_angle: int = 3,
    maximum_iterations: int = 120,
    relative_tolerance: float = 5.0e-3,
    relative_population_floor: float = 1.0e-12,
    population_damping: float = 0.4,
    acceleration: Literal["none", "anderson"] = "anderson",
    anderson_depth: int = 4,
    anderson_mixing: float = 0.65,
    anderson_maximum_log_step: float = 0.75,
    neutral_collision_strength_scale: float = 1.0,
    helium_i_collision_data: TlustyHeliumCollisionData | None = None,
    hydrogenic_collision_model: HydrogenicHeliumCollisionModel = "ccc-scaled",
    hydrogenic_collision_rate_multiplier: float = 1.0,
    hydrogenic_excitation_collision_rate_multiplier: float | None = None,
    hydrogenic_ionization_collision_rate_multiplier: float | None = None,
    initial_state: CoupledHeliumNLTEState | None = None,
) -> CoupledHeliumNLTEState:
    """Iterate the shared 14 + n + 1 helium atom with radiative transfer."""

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not 0.0 < population_damping <= 1.0:
        raise ValueError("population_damping must lie in (0, 1]")
    if not 0.0 < relative_population_floor < 1.0:
        raise ValueError("relative_population_floor must lie in (0, 1)")
    if acceleration not in ("none", "anderson"):
        raise ValueError("acceleration must be 'none' or 'anderson'")
    if anderson_depth < 2:
        raise ValueError("anderson_depth must be at least two")
    if not 0.0 <= anderson_mixing <= 1.0:
        raise ValueError("anderson_mixing must lie in [0, 1]")
    if anderson_maximum_log_step <= 0.0:
        raise ValueError("anderson_maximum_log_step must be positive")
    if not 1 <= helium_ii_explicit_maximum_lower_level <= min(
        4, maximum_helium_ii_level - 1
    ):
        raise ValueError("helium_ii_explicit_maximum_lower_level lies outside the atom")
    neutral_wavelength = default_neutral_helium_continuum_wavelength()
    ion_wavelength = default_helium_ii_continuum_wavelength(
        maximum_helium_ii_level
    )
    neutral_line_groups = _neutral_helium_line_components(
        atmosphere, helium_i_stark_table
    )
    # ``helium_ii_shell_transition`` supplies the exact hydrogenic wavelength
    # and oscillator strength when a transition is absent from the compact
    # formal-synthesis line list.  Restricting the population transfer to that
    # list used to leave, for example, 1--9 through 1--14 and 3--13/14 at the
    # local Planck field in a 14-shell atom.  Those links belong to the rate
    # atom even though they do not need a named emergent-spectrum profile.
    ion_transitions = tuple(
        helium_ii_shell_transition(lower, upper)
        for lower in range(1, helium_ii_explicit_maximum_lower_level + 1)
        for upper in range(lower + 1, maximum_helium_ii_level + 1)
    )
    ion_line_problems = tuple(
        _prepare_helium_line_transfer_problem(
            atmosphere,
            line,
            maximum_helium_ii_level,
            stark_table=helium_ii_stark_table,
        )
        for line in ion_transitions
    )
    if initial_state is None:
        current = solve_coupled_helium_statistical_equilibrium(
            atmosphere,
            collision_data,
            maximum_helium_ii_level=maximum_helium_ii_level,
            neutral_collision_strength_scale=neutral_collision_strength_scale,
            helium_i_collision_data=helium_i_collision_data,
            hydrogenic_collision_model=hydrogenic_collision_model,
            hydrogenic_collision_rate_multiplier=(
                hydrogenic_collision_rate_multiplier
            ),
            hydrogenic_excitation_collision_rate_multiplier=(
                hydrogenic_excitation_collision_rate_multiplier
            ),
            hydrogenic_ionization_collision_rate_multiplier=(
                hydrogenic_ionization_collision_rate_multiplier
            ),
        )
        warm_started = False
    else:
        if (
            initial_state.neutral_departure_coefficient.shape
            != (atmosphere.n_depth, 14)
            or initial_state.singly_ionized_departure_coefficient.shape
            != (atmosphere.n_depth, maximum_helium_ii_level)
        ):
            raise ValueError("initial coupled helium state has the wrong shape")
        if initial_state.metadata.get("atmosphere_structure_sha256") != atmosphere_structure_fingerprint(atmosphere):
            raise ValueError("initial coupled helium state belongs to a different atmosphere")
        current = initial_state
        warm_started = True
    converged = False
    maximum_change = np.inf
    formal_maximum_change = np.inf
    acceleration_history: list[tuple[FloatArray, FloatArray]] = []
    accelerated_steps = 0
    acceleration_rejections: dict[str, int] = {}
    for iteration in range(1, maximum_iterations + 1):
        neutral_view, hydrogenic_view = _coupled_state_views(
            atmosphere, current
        )
        neutral_continuum_mean = _coupled_continuum_mean_intensity(
            atmosphere,
            neutral_wavelength,
            current,
            n_angle=n_angle,
        )
        ion_continuum_mean = _coupled_continuum_mean_intensity(
            atmosphere,
            ion_wavelength,
            current,
            n_angle=n_angle,
        )
        neutral_line_fields: dict[tuple[int, int], FloatArray] = {}
        for key, problems in neutral_line_groups.items():
            component_fields = [
                _line_mean_intensity(
                    atmosphere,
                    problem,
                    neutral_view.departure_coefficient,
                    neutral_view.continuum_departure_coefficient,
                    n_angle=n_angle,
                )
                for problem in problems
            ]
            component_strengths = np.asarray(
                [
                    problem.line.absorption_oscillator_strength
                    for problem in problems
                ]
            )
            neutral_line_fields[key] = np.average(
                np.asarray(component_fields),
                axis=0,
                weights=component_strengths,
            )
        ion_line_fields = {
            (problem.line.lower_level, problem.line.upper_level): (
                _line_mean_intensity(
                    atmosphere,
                    problem,
                    hydrogenic_view.departure_coefficient,
                    hydrogenic_view.continuum_departure_coefficient,
                    n_angle=n_angle,
                )
            )
            for problem in ion_line_problems
        }
        candidate = solve_coupled_helium_statistical_equilibrium(
            atmosphere,
            collision_data,
            maximum_helium_ii_level=maximum_helium_ii_level,
            neutral_line_mean_intensity_nu=neutral_line_fields,
            helium_ii_line_mean_intensity_nu=ion_line_fields,
            neutral_continuum_wavelength_angstrom=neutral_wavelength,
            neutral_continuum_mean_intensity_lambda=neutral_continuum_mean,
            helium_ii_continuum_wavelength_angstrom=ion_wavelength,
            helium_ii_continuum_mean_intensity_lambda=ion_continuum_mean,
            neutral_collision_strength_scale=neutral_collision_strength_scale,
            helium_i_collision_data=helium_i_collision_data,
            hydrogenic_collision_model=hydrogenic_collision_model,
            hydrogenic_collision_rate_multiplier=(
                hydrogenic_collision_rate_multiplier
            ),
            hydrogenic_excitation_collision_rate_multiplier=(
                hydrogenic_excitation_collision_rate_multiplier
            ),
            hydrogenic_ionization_collision_rate_multiplier=(
                hydrogenic_ionization_collision_rate_multiplier
            ),
        )
        maximum_change, formal_maximum_change = _coupled_population_change(
            current,
            candidate,
            relative_population_floor=relative_population_floor,
        )
        if maximum_change < relative_tolerance:
            current = candidate
            converged = True
            break
        log_population = np.column_stack(
            (
                np.log(current.neutral_departure_coefficient),
                np.log(current.singly_ionized_departure_coefficient),
                np.log(current.doubly_ionized_departure_coefficient),
            )
        )
        log_fixed_point = np.column_stack(
            (
                np.log(candidate.neutral_departure_coefficient),
                np.log(candidate.singly_ionized_departure_coefficient),
                np.log(candidate.doubly_ionized_departure_coefficient),
            )
        )
        proposed = None
        if acceleration == "anderson":
            proposed, acceleration_status = _anderson_log_population_update(
                log_population,
                log_fixed_point,
                acceleration_history,
                depth=anderson_depth,
                mixing=anderson_mixing,
                maximum_step=anderson_maximum_log_step,
            )
            if proposed is None:
                acceleration_rejections[acceleration_status] = (
                    acceleration_rejections.get(acceleration_status, 0) + 1
                )
        if proposed is None:
            proposed = log_population + population_damping * (
                log_fixed_point - log_population
            )
        else:
            accelerated_steps += 1
        neutral_departure = np.exp(proposed[:, :14])
        ion_end = 14 + maximum_helium_ii_level
        ion_departure = np.exp(proposed[:, 14:ion_end])
        continuum_departure = np.exp(proposed[:, ion_end])
        reference_total = (
            np.sum(candidate.lte_neutral_population_density, axis=1)
            + np.sum(candidate.lte_singly_ionized_population_density, axis=1)
            + candidate.lte_doubly_ionized_he_density
        )
        mixed_total = (
            np.sum(
                candidate.lte_neutral_population_density
                * neutral_departure,
                axis=1,
            )
            + np.sum(
                candidate.lte_singly_ionized_population_density
                * ion_departure,
                axis=1,
            )
            + candidate.lte_doubly_ionized_he_density
            * continuum_departure
        )
        normalization = reference_total / mixed_total
        neutral_departure *= normalization[:, np.newaxis]
        ion_departure *= normalization[:, np.newaxis]
        continuum_departure *= normalization
        current = _replace_coupled_departures(
            candidate,
            neutral_departure,
            ion_departure,
            continuum_departure,
            iterations=iteration,
            converged=False,
            maximum_change=maximum_change,
            metadata=candidate.metadata,
        )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "population_transfer": "shared continua plus resolved He I/II lines",
            "helium_ii_radiation_coupled_transition_count": len(
                ion_transitions
            ),
            "warm_started": warm_started,
            "nlte_status": "coupled He I/II/III fixed-structure solution",
            "convergence_measure": (
                "actual population change with local helium-reservoir floor"
            ),
            "relative_population_floor": relative_population_floor,
            "formal_maximum_log_departure_change_all_states": (
                formal_maximum_change
            ),
            "acceleration": acceleration,
            "anderson_depth": int(anderson_depth),
            "accelerated_steps": int(accelerated_steps),
            "acceleration_rejections": acceleration_rejections,
        }
    )
    return _replace_coupled_departures(
        current,
        current.neutral_departure_coefficient,
        current.singly_ionized_departure_coefficient,
        current.doubly_ionized_departure_coefficient,
        iterations=iteration,
        converged=converged,
        maximum_change=maximum_change,
        metadata=metadata,
    )


def synthesize_coupled_helium_nlte_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    state: CoupledHeliumNLTEState,
    *,
    helium_i_stark_table: HeliumStarkTable | str | Path,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize a spectrum from one coupled He I/II/III population state."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    coefficients = coupled_helium_nlte_transfer_coefficients(
        atmosphere,
        wavelength,
        state,
        helium_i_stark_table=helium_i_stark_table,
        helium_ii_stark_table=helium_ii_stark_table,
    )
    total = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total
    )
    source = (
        coefficients.thermal_emissivity
        + coefficients.scattering
        * planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    ) / total
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total
        )
    flux = emergent_flux(
        optical_depth, source, n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            **coefficients.metadata,
            "composition": "pure-helium",
            "population_iterations": state.iterations,
            "population_converged": state.converged,
            "n_angle": int(n_angle),
        },
    )


def remap_coupled_helium_state(
    atmosphere: Atmosphere, state: CoupledHeliumNLTEState
) -> CoupledHeliumNLTEState:
    """Carry a coupled helium solution to a nearby LTE structure.

    The departure coefficients are frozen and renormalized with one common
    He I + He II + He III particle-conservation factor at every depth.  This
    is the warm start needed by the outer temperature iteration.
    """

    maximum_helium_ii_level = (
        state.singly_ionized_departure_coefficient.shape[1]
    )
    neutral_lte, _, _ = _neutral_helium_reference_populations(atmosphere, state.helium_i_atom)
    ion_lte, continuum_lte, _ = _reference_populations(
        atmosphere, maximum_helium_ii_level
    )
    if neutral_lte.shape != state.neutral_departure_coefficient.shape:
        raise ValueError("coupled helium state has a different He I atom")
    reference_total = (
        np.sum(neutral_lte, axis=1)
        + np.sum(ion_lte, axis=1)
        + continuum_lte
    )
    trial_total = (
        np.sum(neutral_lte * state.neutral_departure_coefficient, axis=1)
        + np.sum(
            ion_lte * state.singly_ionized_departure_coefficient, axis=1
        )
        + continuum_lte * state.doubly_ionized_departure_coefficient
    )
    normalization = reference_total / np.maximum(
        trial_total, np.finfo(np.float64).tiny
    )
    neutral_departure = np.ascontiguousarray(
        state.neutral_departure_coefficient * normalization[:, np.newaxis]
    )
    ion_departure = np.ascontiguousarray(
        state.singly_ionized_departure_coefficient
        * normalization[:, np.newaxis]
    )
    continuum_departure = np.ascontiguousarray(
        state.doubly_ionized_departure_coefficient * normalization
    )
    metadata = dict(state.metadata)
    metadata.update(
        {
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(
                atmosphere
            ),
            "state_remap": (
                "frozen coupled departures with shared particle normalization"
            ),
        }
    )
    return replace(
        state,
        neutral_population_density=np.ascontiguousarray(
            neutral_lte * neutral_departure
        ),
        singly_ionized_population_density=np.ascontiguousarray(
            ion_lte * ion_departure
        ),
        doubly_ionized_he_density=np.ascontiguousarray(
            continuum_lte * continuum_departure
        ),
        lte_neutral_population_density=neutral_lte,
        lte_singly_ionized_population_density=ion_lte,
        lte_doubly_ionized_he_density=continuum_lte,
        neutral_departure_coefficient=neutral_departure,
        singly_ionized_departure_coefficient=ion_departure,
        doubly_ionized_departure_coefficient=continuum_departure,
        iterations=0,
        converged=False,
        maximum_relative_population_change=0.0,
        metadata=metadata,
    )


def _reference_populations(
    atmosphere: Atmosphere, maximum_level: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    helium = atmosphere.helium_lte_state
    if helium is None:
        raise ValueError("a pure-helium atmosphere with helium_lte_state is required")
    population, occupation = _helium_ii_level_distribution(
        atmosphere, maximum_level
    )
    tiny = np.finfo(np.float64).tiny
    return (
        np.maximum(np.ascontiguousarray(population), tiny),
        np.maximum(
            np.ascontiguousarray(helium.doubly_ionized_he_density), tiny
        ),
        np.ascontiguousarray(occupation),
    )


def _scaled_excitation_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: FloatArray,
    lower_level: int,
    upper_level: int,
) -> FloatArray:
    scaled_temperature = temperature / _HELIUM_NUCLEAR_CHARGE**2
    return (
        _excitation_rate_coefficient(
            collision_data,
            scaled_temperature,
            lower_level,
            upper_level,
        )
        / _HELIUM_NUCLEAR_CHARGE**3
    )


def _scaled_ionization_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: FloatArray,
    level: int,
) -> FloatArray:
    scaled_temperature = temperature / _HELIUM_NUCLEAR_CHARGE**2
    return (
        _ionization_rate_coefficient(
            collision_data, scaled_temperature, level
        )
        / _HELIUM_NUCLEAR_CHARGE**3
    )


def _tlusty_mihalas_heii_excitation_rate_coefficient(
    temperature: FloatArray,
    lower_level: int,
    upper_level: int,
) -> FloatArray:
    """Reproduce TLUSTY 200 ``COLHE`` He II bound-bound rates.

    The expression is the Mihalas--Heasley--Auer hydrogenic fit used for
    non-negative ``ICOL`` transitions in the public TLUSTY He II atom.  Its
    return value is the upward Maxwellian rate coefficient in cm3 s-1.
    """

    if not 1 <= lower_level < upper_level:
        raise ValueError("He II levels must satisfy 1 <= lower < upper")
    value = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("temperature must be finite and positive")
    reduced_energy = (
        HELIUM_SECOND_IONIZATION_ENERGY
        * (1.0 / lower_level**2 - 1.0 / upper_level**2)
        / (BOLTZMANN * value)
    )
    oscillator_strength = _extended_hydrogen_shell_oscillator_strength(
        lower_level, upper_level
    )
    lower = float(lower_level)
    upper = float(upper_level)
    collision_factor = lower - (lower - 1.0) / (upper - lower)
    collision_factor = min(collision_factor, upper - lower)
    if lower_level > 1:
        collision_factor *= 1.1
    coefficient = (
        3.703_648_9
        / value**1.5
        / reduced_energy
        * oscillator_strength
        * (
            0.693 * np.exp(-np.minimum(reduced_energy, 745.0))
            + _exponential_integral_e1(reduced_energy)
        )
        * collision_factor
    )
    return np.maximum(np.asarray(coefficient, dtype=np.float64), 0.0)


def _tlusty_mihalas_heii_ionization_rate_coefficient(
    temperature: FloatArray,
    level: int,
) -> FloatArray:
    """Reproduce TLUSTY ``COLHE`` He II continuum rates.

    The Mihalas/Werner fit is used through 100 kK.  Above that temperature,
    the public source switches to its Sampson--Zhang/XSTAR ``SZIRC`` fit for
    explicit shells below the n=16 pseudo-continuum.  Our larger optional atom
    retains the Mihalas closure for n>=16, where that pseudo-continuum formula
    is not defined.
    """

    if level < 1:
        raise ValueError("He II level must be positive")
    value = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("temperature must be finite and positive")
    log_temperature = np.log10(value)
    if level == 1:
        gamma = np.polynomial.polynomial.polyval(
            log_temperature, _TLUSTY_HEII_WERNER_POLYNOMIAL[:, 0]
        )
    elif level <= 3:
        index = level - 1
        gamma = (
            _TLUSTY_HEII_IONIZATION_G0[index]
            - _TLUSTY_HEII_IONIZATION_G1[index] * value
            + (
                _TLUSTY_HEII_IONIZATION_G2[index] / value
                - _TLUSTY_HEII_IONIZATION_G3[index]
            )
            / value
        )
    elif level == 4:
        gamma = -95.23828 + (62.656249 - 8.1454078 * log_temperature) * log_temperature
    elif level == 5:
        gamma = 472.99219 - 74.144287 * log_temperature - 1869.6562 / log_temperature**2
    elif level == 6:
        gamma = 825.17186 - 134.23096 * log_temperature - 2739.4375 / log_temperature**2
    elif level == 7:
        gamma = 1181.3516 - 200.71191 * log_temperature - 2810.7812 / log_temperature**2
    elif level == 8:
        gamma = 1440.1016 - 259.75781 * log_temperature - 1283.5625 / log_temperature**2
    elif level == 9:
        gamma = 2492.1250 - 624.84375 * log_temperature + 30.101562 * log_temperature**2
    elif level == 10:
        gamma = 4663.3129 - 1390.1250 * log_temperature + 97.671874 * log_temperature**2
    else:
        gamma = np.full_like(value, float(level**3))
    reduced_threshold = (
        HELIUM_SECOND_IONIZATION_ENERGY
        / (level**2 * BOLTZMANN * value)
    )
    low_temperature_coefficient = (
        5.465e-11
        * np.sqrt(value)
        * np.exp(-np.minimum(reduced_threshold, 745.0))
        * np.maximum(gamma, 0.0)
    )
    if level >= 16 or not np.any(value > 1.0e5):
        return np.maximum(
            np.asarray(low_temperature_coefficient, dtype=np.float64), 0.0
        )

    abethe = np.asarray(
        [1.134, 0.603, 0.412, 0.313, 0.252, 0.211, 0.181, 0.159,
         0.142, 0.128, 1.307],
        dtype=np.float64,
    )
    hbethe = np.asarray(
        [1.48, 3.64, 5.93, 8.32, 10.75, 12.90, 15.05, 17.20,
         19.35, 21.50, 2.15],
        dtype=np.float64,
    )
    rbethe = np.asarray(
        [2.20, 1.90, 1.73, 1.65, 1.60, 1.56, 1.54, 1.52, 1.52,
         1.52, 1.52],
        dtype=np.float64,
    )
    if level < 11:
        bethe_a = abethe[level - 1]
        bethe_h = hbethe[level - 1]
        bethe_r = rbethe[level - 1]
    else:
        bethe_a = abethe[10] / level
        bethe_h = hbethe[10] * level
        bethe_r = rbethe[10]
    continuum_level = 16.0
    shell = float(level)
    reduced = (
        _HELIUM_NUCLEAR_CHARGE**2
        * HYDROGEN_IONIZATION_ENERGY
        / (BOLTZMANN * value)
        * (
            1.0 / shell**2
            - 1.0 / continuum_level**2
            - 0.25
            * (
                1.0 / (continuum_level - 1.0) ** 2
                - 1.0 / continuum_level**2
            )
        )
    )
    exponential = np.exp(-np.minimum(reduced, 745.0))
    integral_1 = _exponential_integral_e1(reduced)
    integral_2 = exponential - reduced * integral_1
    integral_3 = 0.5 * (exponential - reduced * integral_2)
    high_temperature_coefficient = (
        4.6513e-3
        * np.sqrt(BOLTZMANN * value)
        * shell**5
        / _HELIUM_NUCLEAR_CHARGE**4
        * bethe_a
        * reduced
        * (
            integral_1 / shell
            - (exponential - reduced * integral_3) / (3.0 * shell)
            + (
                reduced * integral_2
                - 2.0 * reduced * integral_1
                + exponential
            )
            * 3.0
            * bethe_h
            / shell
            / (3.0 - bethe_r)
            + (integral_1 - integral_2) * 3.36 * reduced
        )
    )
    coefficient = np.where(
        value > 1.0e5,
        high_temperature_coefficient,
        low_temperature_coefficient,
    )
    return np.maximum(np.asarray(coefficient, dtype=np.float64), 0.0)


def helium_ii_excitation_collision_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: ArrayLike,
    lower_level: int,
    upper_level: int,
    *,
    model: HydrogenicHeliumCollisionModel = "ccc-scaled",
) -> FloatArray:
    """Return an upward He II electron-impact excitation coefficient."""

    value = np.asarray(temperature, dtype=np.float64)
    if model == "ccc-scaled":
        return _scaled_excitation_rate_coefficient(
            collision_data, value, lower_level, upper_level
        )
    if model == "tlusty-mihalas":
        return _tlusty_mihalas_heii_excitation_rate_coefficient(
            value, lower_level, upper_level
        )
    raise ValueError("unknown hydrogenic helium collision model")


def helium_ii_ionization_collision_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: ArrayLike,
    level: int,
    *,
    model: HydrogenicHeliumCollisionModel = "ccc-scaled",
) -> FloatArray:
    """Return a He II electron-impact ionization coefficient."""

    value = np.asarray(temperature, dtype=np.float64)
    if model == "ccc-scaled":
        return _scaled_ionization_rate_coefficient(collision_data, value, level)
    if model == "tlusty-mihalas":
        return _tlusty_mihalas_heii_ionization_rate_coefficient(value, level)
    raise ValueError("unknown hydrogenic helium collision model")


def helium_ii_photoionization_cross_section(
    level: int, frequency_hz: ArrayLike
) -> FloatArray:
    """Hydrogenic He II photoionization cross section in cm2."""

    if level < 1:
        raise ValueError("level must be positive")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    threshold = HELIUM_SECOND_IONIZATION_ENERGY / (PLANCK * level**2)
    if level == 1:
        cross_section = (
            hydrogen_ground_state_photoionization_cross_section(
                frequency / _HELIUM_NUCLEAR_CHARGE**2
            )
            / _HELIUM_NUCLEAR_CHARGE**2
        )
    else:
        cross_section = (
            2.815e29
            * _HELIUM_NUCLEAR_CHARGE**4
            * frequency**-3
            / level**5
            * hydrogen_bound_free_gaunt_factor(
                level, frequency / _HELIUM_NUCLEAR_CHARGE**2
            )
        )
    return np.where(frequency >= threshold, cross_section, 0.0)


def default_helium_ii_continuum_wavelength(
    maximum_level: int,
) -> FloatArray:
    """Return a frequency grid resolving every He II bound-free edge."""

    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")
    ground_edge = (
        PLANCK * LIGHT_SPEED / HELIUM_SECOND_IONIZATION_ENERGY * 1.0e8
    )
    upper = 1.025 * ground_edge * maximum_level**2
    base = np.geomspace(25.0, upper, 420)
    edges: list[float] = []
    for level in range(1, maximum_level + 1):
        edge = ground_edge * level**2
        edges.extend((edge * (1.0 - 2.0e-5), edge, edge * (1.0 + 2.0e-5)))
    return np.unique(np.asarray([*base, *edges], dtype=np.float64))


def _continuum_radiative_rates(
    atmosphere: Atmosphere,
    lte_population: FloatArray,
    lte_continuum: FloatArray,
    wavelength_angstrom: FloatArray,
    mean_intensity_lambda: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    mean_lambda = np.asarray(mean_intensity_lambda, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 8
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("continuum wavelength must be positive and increasing")
    if mean_lambda.shape != (wavelength.size, atmosphere.n_depth):
        raise ValueError("continuum mean intensity has the wrong shape")
    if np.any(~np.isfinite(mean_lambda)) or np.any(mean_lambda < 0.0):
        raise ValueError("continuum mean intensity must be finite and non-negative")

    upward, recombination = continuum_integrals(
        wavelength, atmosphere.temperature, mean_lambda, lte_population.shape[1], helium_ii_photoionization_cross_section)
    downward = recombination * _finite_population_ratio(lte_population, lte_continuum[:,None])
    return upward, downward


def solve_hydrogenic_helium_statistical_equilibrium(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    maximum_level: int = 8,
    line_mean_intensity_nu: Mapping[tuple[int, int], ArrayLike] | None = None,
    continuum_wavelength_angstrom: ArrayLike | None = None,
    continuum_mean_intensity_lambda: ArrayLike | None = None,
    collision_model: HydrogenicHeliumCollisionModel = "ccc-scaled",
    collision_rate_multiplier: float = 1.0,
    fix_continuum_departure: bool = False,
) -> HydrogenicHeliumNLTEState:
    """Solve He II/III statistical equilibrium for a prescribed radiation field.

    Missing radiation fields use the local Planck function.  Consequently a
    call with no supplied fields is an exact detailed-balance regression and
    must return unit departure coefficients.
    """

    if not 2 <= maximum_level <= 32:
        raise ValueError("maximum_level must lie in [2, 32]")
    if not np.isfinite(collision_rate_multiplier) or collision_rate_multiplier < 0.0:
        raise ValueError("collision_rate_multiplier must be finite and non-negative")
    lte_population, lte_continuum, occupation = _reference_populations(
        atmosphere, maximum_level
    )
    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    n_depth = atmosphere.n_depth
    n_state = maximum_level + 1
    rate = np.zeros((n_depth, n_state, n_state), dtype=np.float64)
    supplied_line_fields = (
        {} if line_mean_intensity_nu is None else line_mean_intensity_nu
    )

    for lower in range(1, maximum_level):
        for upper in range(lower + 1, maximum_level + 1):
            line = helium_ii_shell_transition(lower, upper)
            supplied = supplied_line_fields.get((lower, upper))
            if supplied is None:
                mean_intensity = _planck_nu_at_wavelength(
                    line.wavelength_vacuum_angstrom, temperature
                )
            else:
                mean_intensity = np.asarray(supplied, dtype=np.float64)
                if mean_intensity.shape != (n_depth,):
                    raise ValueError("each line field must have one value per depth")
                if np.any(~np.isfinite(mean_intensity)) or np.any(mean_intensity < 0.0):
                    raise ValueError("line fields must be finite and non-negative")
            radiative_up, radiative_down = _bound_bound_radiative_rates(
                temperature,
                line,
                lte_population[:, lower - 1],
                lte_population[:, upper - 1],
                occupation[:, lower - 1],
                occupation[:, upper - 1],
                mean_intensity,
            )
            collisional_up = (
                collision_rate_multiplier
                * electron_density
                * helium_ii_excitation_collision_rate_coefficient(
                    collision_data,
                    temperature,
                    lower,
                    upper,
                    model=collision_model,
                )
                * np.clip(occupation[:, upper - 1], 0.0, 1.0)
            )
            collisional_down = collisional_up * _finite_population_ratio(
                lte_population[:, lower - 1],
                lte_population[:, upper - 1],
            )
            rate[:, lower - 1, upper - 1] += radiative_up + collisional_up
            rate[:, upper - 1, lower - 1] += radiative_down + collisional_down

    if continuum_wavelength_angstrom is None:
        continuum_wavelength = default_helium_ii_continuum_wavelength(
            maximum_level
        )
    else:
        continuum_wavelength = np.asarray(
            continuum_wavelength_angstrom, dtype=np.float64
        )
    if continuum_mean_intensity_lambda is None:
        continuum_mean = planck_lambda_angstrom(
            continuum_wavelength[:, np.newaxis],
            temperature[np.newaxis, :],
        )
    else:
        continuum_mean = np.asarray(
            continuum_mean_intensity_lambda, dtype=np.float64
        )
    photo_up, radiative_recombination = _continuum_radiative_rates(
        atmosphere,
        lte_population,
        lte_continuum,
        np.ascontiguousarray(continuum_wavelength),
        continuum_mean,
    )
    continuum_index = maximum_level
    for level in range(1, maximum_level + 1):
        collisional_ionization = (
            collision_rate_multiplier
            * electron_density
            * helium_ii_ionization_collision_rate_coefficient(
                collision_data,
                temperature,
                level,
                model=collision_model,
            )
        )
        three_body_recombination = (
            collisional_ionization
            * _finite_population_ratio(
                lte_population[:, level - 1], lte_continuum
            )
        )
        rate[:, level - 1, continuum_index] += (
            photo_up[:, level - 1] + collisional_ionization
        )
        rate[:, continuum_index, level - 1] += (
            radiative_recombination[:, level - 1]
            + three_body_recombination
        )

    reference = np.column_stack((lte_population, lte_continuum))
    active_density = np.sum(reference, axis=1)
    populations = np.empty_like(reference)
    for depth in range(n_depth):
        matrix = rate[depth].T.copy()
        matrix[np.diag_indices(n_state)] -= np.sum(rate[depth], axis=1)
        matrix *= reference[depth][np.newaxis, :]
        right_hand_side = np.zeros(n_state, dtype=np.float64)
        if fix_continuum_departure:
            matrix[-1] = 0.0
            matrix[-1, -1] = reference[depth, -1]
            right_hand_side[-1] = reference[depth, -1]
        else:
            matrix[-1] = reference[depth]
            right_hand_side[-1] = active_density[depth]
        row_scale = np.maximum(
            np.max(np.abs(matrix), axis=1), np.finfo(np.float64).tiny
        )
        matrix /= row_scale[:, np.newaxis]
        right_hand_side /= row_scale
        try:
            departure = np.linalg.solve(matrix, right_hand_side)
        except np.linalg.LinAlgError:
            departure = np.linalg.lstsq(matrix, right_hand_side, rcond=None)[0]
        if np.any(~np.isfinite(departure)) or np.any(departure <= 0.0):
            raise NonphysicalPopulationError(
                f"non-physical He II statistical-equilibrium solution at depth {depth}"
            )
        populations[depth] = departure * reference[depth]

    inactive = lte_population < active_density[:, np.newaxis] * 1.0e-60
    populations[:, :maximum_level][inactive] = lte_population[inactive]
    if fix_continuum_departure:
        populations[:, continuum_index] = lte_continuum
    else:
        populations[:, continuum_index] = active_density - np.sum(
            populations[:, :maximum_level], axis=1
        )
    if np.any(populations[:, continuum_index] <= 0.0):
        raise NonphysicalPopulationError("He II statistical-equilibrium normalization removed He III")
    departure = populations / reference
    return HydrogenicHeliumNLTEState(
        principal_quantum_number=np.arange(
            1, maximum_level + 1, dtype=np.float64
        ),
        population_density=np.ascontiguousarray(
            populations[:, :maximum_level]
        ),
        doubly_ionized_he_density=np.ascontiguousarray(
            populations[:, continuum_index]
        ),
        lte_population_density=lte_population,
        lte_doubly_ionized_he_density=lte_continuum,
        departure_coefficient=np.ascontiguousarray(
            departure[:, :maximum_level]
        ),
        continuum_departure_coefficient=np.ascontiguousarray(
            departure[:, continuum_index]
        ),
        iterations=1,
        converged=True,
        maximum_relative_population_change=float(
            np.max(np.abs(departure - 1.0))
        ),
        metadata={
            "model_atom": (
                f"restricted He II n=1-{maximum_level} plus He III; He I LTE"
            ),
            "explicit_maximum_level": int(maximum_level),
            "collision_data": collision_data.source,
            "collision_scaling": (
                "hydrogenic q_Z(T)=Z^-3 q_H(T/Z^2), Z=2"
                if collision_model == "ccc-scaled"
                else "TLUSTY COLHE Mihalas/Werner He II fits"
            ),
            "hydrogenic_collision_model": collision_model,
            "radiation_field": (
                "supplied line/continuum fields with local-Planck fallback"
            ),
            "continuum_population_closure": (
                "diagnostic He III departure fixed to unity"
                if fix_continuum_departure
                else "active He II plus He III particle conservation"
            ),
            "ion_stage_scope": "He I retained in LTE background",
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(
                atmosphere
            ),
        },
    )


def _prepare_helium_continuum_transfer_problem(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    maximum_level: int,
    *,
    lte_absorption: FloatArray | None = None,
) -> _ContinuumTransferProblem:
    """Separate explicit He II bound-free terms from the LTE He background."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("continuum wavelength must be positive and increasing")
    # ``lte_absorption`` may be supplied by a caller that builds both the
    # He I and He II problems on the same atmosphere and grid; it is exactly
    # this LTE continuum, so the value is unchanged.
    if lte_absorption is None:
        lte_absorption = helium_continuum_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            include_electron_scattering=False,
            include_rayleigh_scattering=False,
        )
    lte_population, _, _ = _reference_populations(atmosphere, maximum_level)
    frequency = _LIGHT_SPEED_ANGSTROM_PER_SECOND / wavelength
    exponent = (
        PLANCK
        * frequency[:, np.newaxis]
        / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
    )
    exp_minus = np.exp(-np.minimum(exponent, 745.0))
    coefficient = np.empty(
        (wavelength.size, atmosphere.n_depth, maximum_level),
        dtype=np.float64,
    )
    for level in range(1, maximum_level + 1):
        coefficient[:, :, level - 1] = (
            helium_ii_photoionization_cross_section(level, frequency)[
                :, np.newaxis
            ]
            * lte_population[np.newaxis, :, level - 1]
            / atmosphere.mass_density[np.newaxis, :]
        )
    lte_explicit_bound_free = np.sum(
        coefficient * (1.0 - exp_minus)[:, :, np.newaxis], axis=2
    )
    thermal_background = lte_absorption - lte_explicit_bound_free
    tolerance = 3.0e-12 * np.maximum(lte_absorption, 1.0e-40)
    if np.any(thermal_background < -tolerance):
        mismatch = float(
            np.min(
                thermal_background / np.maximum(lte_absorption, 1.0e-40)
            )
        )
        raise NonphysicalPopulationError(
            "He II bound-free decomposition is inconsistent with LTE "
            f"continuum opacity ({mismatch:.3e})"
        )
    thermal_background = np.maximum(thermal_background, 0.0)
    spontaneous_nu = 2.0 * PLANCK * frequency**3 / LIGHT_SPEED**2
    spontaneous_lambda = (
        spontaneous_nu * _LIGHT_SPEED_ANGSTROM_PER_SECOND / wavelength**2
    )
    scattering = (
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
        + helium_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
    )
    return _ContinuumTransferProblem(
        wavelength_angstrom=wavelength,
        thermal_background_absorption=np.ascontiguousarray(
            thermal_background
        ),
        bound_free_coefficient=np.ascontiguousarray(coefficient),
        exp_minus_photon_energy=np.ascontiguousarray(exp_minus),
        spontaneous_intensity_lambda=np.ascontiguousarray(
            spontaneous_lambda[:, np.newaxis]
        ),
        scattering=np.ascontiguousarray(scattering),
        planck_lambda=np.ascontiguousarray(
            planck_lambda_angstrom(
                wavelength[:, np.newaxis],
                atmosphere.temperature[np.newaxis, :],
            )
        ),
    )


def _prepare_helium_line_transfer_problem(
    atmosphere: Atmosphere,
    line: HydrogenLine,
    maximum_level: int,
    *,
    stark_table: HeliumIIStarkTable | str | Path | None,
) -> _LineTransferProblem:
    resolved_stark_table = stark_table
    if (
        resolved_stark_table is not None
        and not isinstance(resolved_stark_table, HeliumIIStarkTable)
    ):
        resolved_stark_table = read_helium_ii_stark_table(
            resolved_stark_table
        )
    transition = (line.lower_level, line.upper_level)
    if (
        isinstance(resolved_stark_table, HeliumIIStarkTable)
        and transition in resolved_stark_table.lines
    ):
        extent = float(
            resolved_stark_table[transition].wavelength_offset_angstrom[-1]
        )
        logarithmic_offset = np.geomspace(1.0e-4, extent, 120)
        wavelength = np.unique(
            np.r_[
                line.wavelength_vacuum_angstrom - logarithmic_offset[::-1],
                np.linspace(
                    line.wavelength_vacuum_angstrom - 6.0,
                    line.wavelength_vacuum_angstrom + 6.0,
                    121,
                ),
                line.wavelength_vacuum_angstrom + logarithmic_offset,
            ]
        )
    else:
        wavelength = _line_grid(line)
    helium_line = next(
        (
            item
            for item in HELIUM_II_LINES
            if item.lower_principal_quantum_number == line.lower_level
            and item.upper_principal_quantum_number == line.upper_level
        ),
        None,
    )
    if helium_line is None:
        # The compact public line tuple names only the members commonly used
        # for emergent-spectrum synthesis.  Population-rate transfer also
        # needs the remaining hydrogenic links; the opacity routine already
        # provides its documented transformed-unified-profile fallback.
        helium_line = HeliumIILine(
            line.name,
            line.lower_level,
            line.upper_level,
            line.wavelength_vacuum_angstrom,
            line.absorption_oscillator_strength,
        )
    return _LineTransferProblem(
        line=line,
        continuum=_prepare_helium_continuum_transfer_problem(
            atmosphere, wavelength, maximum_level
        ),
        lte_line_opacity=helium_ii_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=(helium_line,),
            stark_table=resolved_stark_table,
        ),
        lower_state_index=line.lower_level - 1,
        upper_state_index=line.upper_level - 1,
    )


def solve_hydrogenic_helium_nlte(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    maximum_level: int = 8,
    explicit_maximum_lower_level: int = 3,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    n_angle: int = 3,
    maximum_iterations: int = 120,
    relative_tolerance: float = 5.0e-3,
    population_damping: float = 0.45,
    collision_model: HydrogenicHeliumCollisionModel = "ccc-scaled",
    collision_rate_multiplier: float = 1.0,
    initial_departure_coefficient: ArrayLike | None = None,
    initial_continuum_departure_coefficient: ArrayLike | None = None,
) -> HydrogenicHeliumNLTEState:
    """Iterate restricted He II/III populations with their radiation field."""

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not 2 <= maximum_level <= 32:
        raise ValueError("maximum_level must lie in [2, 32]")
    if not 1 <= explicit_maximum_lower_level <= min(4, maximum_level - 1):
        raise ValueError("explicit_maximum_lower_level lies outside the atom")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(population_damping) or not 0.0 < population_damping <= 1.0:
        raise ValueError("population_damping must lie in (0, 1]")

    continuum_wavelength = default_helium_ii_continuum_wavelength(
        maximum_level
    )
    continuum_problem = _prepare_helium_continuum_transfer_problem(
        atmosphere, continuum_wavelength, maximum_level
    )
    available = {
        (line.lower_principal_quantum_number, line.upper_principal_quantum_number)
        for line in HELIUM_II_LINES
    }
    transitions = tuple(
        helium_ii_shell_transition(lower, upper)
        for lower in range(1, explicit_maximum_lower_level + 1)
        for upper in range(lower + 1, maximum_level + 1)
        if (lower, upper) in available
    )
    line_problems = tuple(
        _prepare_helium_line_transfer_problem(
            atmosphere,
            line,
            maximum_level,
            stark_table=helium_ii_stark_table,
        )
        for line in transitions
    )
    lte_population, lte_continuum, _ = _reference_populations(
        atmosphere, maximum_level
    )
    if (initial_departure_coefficient is None) != (
        initial_continuum_departure_coefficient is None
    ):
        raise ValueError("both warm-start departure arrays must be supplied")
    if initial_departure_coefficient is None:
        departure = np.ones_like(lte_population)
        continuum_departure = np.ones_like(lte_continuum)
        warm_started = False
    else:
        departure = np.asarray(
            initial_departure_coefficient, dtype=np.float64
        ).copy()
        continuum_departure = np.asarray(
            initial_continuum_departure_coefficient, dtype=np.float64
        ).copy()
        if departure.shape != lte_population.shape or continuum_departure.shape != lte_continuum.shape:
            raise ValueError("warm-start departures do not match this atmosphere/atom")
        if (
            np.any(~np.isfinite(departure))
            or np.any(departure <= 0.0)
            or np.any(~np.isfinite(continuum_departure))
            or np.any(continuum_departure <= 0.0)
        ):
            raise ValueError("warm-start departures must be finite and positive")
        departure, continuum_departure = _normalize_departure_coefficients(
            lte_population,
            lte_continuum,
            departure,
            continuum_departure,
        )
        warm_started = True

    converged = False
    maximum_change = np.inf
    solution = None
    for iteration in range(1, maximum_iterations + 1):
        continuum_mean = _continuum_radiation_field(
            atmosphere,
            continuum_problem,
            departure,
            continuum_departure,
            n_angle=n_angle,
        )
        line_fields = {
            (problem.line.lower_level, problem.line.upper_level): (
                _line_mean_intensity(
                    atmosphere,
                    problem,
                    departure,
                    continuum_departure,
                    n_angle=n_angle,
                )
            )
            for problem in line_problems
        }
        candidate = solve_hydrogenic_helium_statistical_equilibrium(
            atmosphere,
            collision_data,
            maximum_level=maximum_level,
            line_mean_intensity_nu=line_fields,
            continuum_wavelength_angstrom=continuum_wavelength,
            continuum_mean_intensity_lambda=continuum_mean,
            collision_model=collision_model,
            collision_rate_multiplier=collision_rate_multiplier,
        )
        candidate_departure = candidate.departure_coefficient
        candidate_continuum = candidate.continuum_departure_coefficient
        log_change = np.maximum(
            np.abs(np.log(candidate_departure / departure)),
            0.0,
        )
        maximum_change = float(
            max(
                np.max(log_change),
                np.max(np.abs(np.log(candidate_continuum / continuum_departure))),
            )
        )
        solution = candidate
        if maximum_change < relative_tolerance:
            departure = candidate_departure
            continuum_departure = candidate_continuum
            converged = True
            break
        departure = np.exp(
            (1.0 - population_damping) * np.log(departure)
            + population_damping * np.log(candidate_departure)
        )
        continuum_departure = np.exp(
            (1.0 - population_damping) * np.log(continuum_departure)
            + population_damping * np.log(candidate_continuum)
        )
        departure, continuum_departure = _normalize_departure_coefficients(
            lte_population,
            lte_continuum,
            departure,
            continuum_departure,
        )

    assert solution is not None
    population = lte_population * departure
    continuum_population = lte_continuum * continuum_departure
    metadata = dict(solution.metadata)
    metadata.update(
        {
            "structure": "fixed LTE helium atmosphere and electron density",
            "population_transfer": (
                "He II bound-free plus explicit He II lines"
            ),
            "explicit_maximum_lower_level": int(
                explicit_maximum_lower_level
            ),
            "warm_started": warm_started,
            "nlte_status": (
                "restricted He II/III milestone; He I and charge feedback pending"
            ),
        }
    )
    return replace(
        solution,
        population_density=np.ascontiguousarray(population),
        doubly_ionized_he_density=np.ascontiguousarray(
            continuum_population
        ),
        departure_coefficient=np.ascontiguousarray(departure),
        continuum_departure_coefficient=np.ascontiguousarray(
            continuum_departure
        ),
        iterations=iteration,
        converged=converged,
        maximum_relative_population_change=maximum_change,
        metadata=metadata,
    )


def remap_hydrogenic_helium_state(
    atmosphere: Atmosphere, state: HydrogenicHeliumNLTEState
) -> HydrogenicHeliumNLTEState:
    """Carry He II/III departure coefficients to a nearby LTE structure."""

    maximum_level = int(
        state.metadata.get(
            "explicit_maximum_level", state.departure_coefficient.shape[1]
        )
    )
    lte_population, lte_continuum, _ = _reference_populations(
        atmosphere, maximum_level
    )
    departure, continuum_departure = _normalize_departure_coefficients(
        lte_population,
        lte_continuum,
        state.departure_coefficient,
        state.continuum_departure_coefficient,
    )
    metadata = dict(state.metadata)
    metadata["atmosphere_structure_sha256"] = atmosphere_structure_fingerprint(
        atmosphere
    )
    metadata["state_remap"] = "frozen He II/III departure coefficients"
    return replace(
        state,
        population_density=np.ascontiguousarray(
            lte_population * departure
        ),
        doubly_ionized_he_density=np.ascontiguousarray(
            lte_continuum * continuum_departure
        ),
        lte_population_density=lte_population,
        lte_doubly_ionized_he_density=lte_continuum,
        departure_coefficient=departure,
        continuum_departure_coefficient=continuum_departure,
        metadata=metadata,
    )


def helium_nlte_transfer_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    state: HydrogenicHeliumNLTEState,
    *,
    maximum_level: int = 8,
    helium_i_stark_table: HeliumStarkTable | str | Path | None = None,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
) -> NLTETransferCoefficients:
    """Assemble He LTE-background plus NLTE He II/III transfer coefficients."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength must be positive and increasing")
    _validate_state_atmosphere(atmosphere, state)
    maximum_level = min(
        maximum_level,
        int(state.metadata.get("explicit_maximum_level", maximum_level)),
    )
    continuum_problem = _prepare_helium_continuum_transfer_problem(
        atmosphere, wavelength, maximum_level
    )
    absorption, emissivity = _nlte_continuum_terms(
        continuum_problem,
        state.departure_coefficient,
        state.continuum_departure_coefficient,
    )
    planck = continuum_problem.planck_lambda
    suppressed_inversions = 0
    if include_helium_ii_lines:
        for helium_line in HELIUM_II_LINES:
            lower = helium_line.lower_principal_quantum_number
            upper = helium_line.upper_principal_quantum_number
            if upper > maximum_level:
                continue
            line = helium_ii_shell_transition(lower, upper)
            lte_opacity = helium_ii_line_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                lines=(helium_line,),
                stark_table=helium_ii_stark_table,
            )
            opacity_factor, source_factor = _departure_line_factors(
                atmosphere,
                line,
                state.departure_coefficient,
                suppress_population_inversion=True,
            )
            suppressed_inversions += int(
                np.count_nonzero(opacity_factor == 0.0)
            )
            line_opacity = lte_opacity * opacity_factor[np.newaxis, :]
            absorption += line_opacity
            emissivity += (
                line_opacity
                * planck
                * source_factor[np.newaxis, :]
            )
    if include_helium_i_lines and helium_i_stark_table is not None:
        helium_i_opacity = helium_i_line_mass_absorption_coefficient(
            atmosphere, wavelength, helium_i_stark_table
        )
        absorption += helium_i_opacity
        emissivity += helium_i_opacity * planck
    if include_helium_i_resonance_lines:
        resonance_opacity = helium_i_resonance_line_mass_absorption_coefficient(
            atmosphere, wavelength
        )
        absorption += resonance_opacity
        emissivity += resonance_opacity * planck
    return NLTETransferCoefficients(
        wavelength_angstrom=wavelength,
        true_absorption=np.ascontiguousarray(absorption),
        thermal_emissivity=np.ascontiguousarray(emissivity),
        scattering=continuum_problem.scattering,
        metadata={
            "model_atom": state.metadata.get("model_atom"),
            "ion_stage_scope": "He II/III NLTE; He I LTE background",
            "helium_i_optical_lines": bool(
                include_helium_i_lines and helium_i_stark_table is not None
            ),
            "helium_i_resonance_lines": bool(
                include_helium_i_resonance_lines
            ),
            "helium_ii_lines": bool(include_helium_ii_lines),
            "suppressed_line_inversions": suppressed_inversions,
        },
    )


def synthesize_hydrogenic_helium_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    collision_data: HydrogenElectronCollisionData,
    *,
    nlte_state: HydrogenicHeliumNLTEState | None = None,
    maximum_level: int = 8,
    explicit_maximum_lower_level: int = 3,
    helium_i_stark_table: HeliumStarkTable | str | Path | None = None,
    helium_ii_stark_table: HeliumIIStarkTable | str | Path | None = None,
    include_helium_i_lines: bool = True,
    include_helium_i_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize a restricted He II/III NLTE pure-helium spectrum."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if nlte_state is None:
        nlte_state = solve_hydrogenic_helium_nlte(
            atmosphere,
            collision_data,
            maximum_level=maximum_level,
            explicit_maximum_lower_level=explicit_maximum_lower_level,
            helium_ii_stark_table=helium_ii_stark_table,
            n_angle=max(1, min(n_angle, 3)),
        )
    coefficients = helium_nlte_transfer_coefficients(
        atmosphere,
        wavelength,
        nlte_state,
        maximum_level=maximum_level,
        helium_i_stark_table=helium_i_stark_table,
        helium_ii_stark_table=helium_ii_stark_table,
        include_helium_i_lines=include_helium_i_lines,
        include_helium_i_resonance_lines=include_helium_i_resonance_lines,
        include_helium_ii_lines=include_helium_ii_lines,
    )
    total = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total
    )
    source = (
        coefficients.thermal_emissivity
        + coefficients.scattering
        * planck_lambda_angstrom(
            wavelength[:, np.newaxis],
            atmosphere.temperature[np.newaxis, :],
        )
    ) / total
    field = None
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total
        )
    flux = emergent_flux(
        optical_depth, source, n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "composition": "pure-helium",
            "line_formation": (
                "restricted He II/III NLTE; He I LTE background"
            ),
            "model_atom": nlte_state.metadata.get("model_atom"),
            "population_iterations": nlte_state.iterations,
            "population_converged": nlte_state.converged,
            "maximum_log_population_change": (
                nlte_state.maximum_relative_population_change
            ),
            "n_angle": int(n_angle),
        },
    )
