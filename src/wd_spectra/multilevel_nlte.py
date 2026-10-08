"""Fixed-structure multilevel non-LTE hydrogen line formation.

This module is the first predictive replacement for the adjustable two-level
experiment in :mod:`wd_spectra.nlte`.  It solves statistical equilibrium for
shell-averaged H I levels and an H II continuum while keeping the atmospheric
temperature, pressure, electron density, and molecular populations fixed.

Electron-impact excitation and ionization are integrated directly from the
convergent-close-coupling (CCC) cross sections distributed by the Curtin CCC
database. OpenWD bundles the release archive with permission;
:func:`read_ccc_hydrogen_collision_data` also accepts another ZIP archive
without extracting it.

The transfer iteration updates H I bound-free extinction and Milne emissivity
with the populations while retaining the LTE H-minus, free-free, and molecular
thermal background.  It explicitly iterates the Lyman and Balmer transitions;
Paschen and Brackett transfer can be enabled as diagnostics.  A safeguarded
Anderson iteration accelerates the coupled radiation/population fixed point.
The thermal structure and electron density remain prescribed, but the default
restricted-NLTE closure conserves the total active hydrogen population and
therefore allows the H II population to respond to the rate equations.  This
matches TLUSTY's fixed-structure formal solution; fixing the H II departure
coefficient to unity remains available as an explicit diagnostic.  An
optional ninth rate state merges the Hummer--Mihalas n=9--40
Rydberg population: its bound-free terms are summed shell by shell, CCC data
remain in force through n=9, and the unavailable higher collision rates use
the explicitly reported TLUSTY/Mihalas modified-rate closure.  The solver
remains opt-in and reports all of these choices in its metadata.
"""

from __future__ import annotations
from ._nlte_radiative_integrals import continuum_integrals

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, Mapping
from zipfile import ZipFile

import numpy as np
from ._compat import trapezoid
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .constants import (
    BOHR_RADIUS,
    BOLTZMANN,
    ELECTRON_MASS,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import HM_MAX_BOUND_LEVEL
from .gaunt import hydrogen_bound_free_gaunt_factor
from .nlte import (
    _profile_averaged_mean_intensity_nu,
    einstein_a_from_absorption_oscillator_strength,
)
from .nlte_core import NLTETransferCoefficients, _cached_transfer, NonphysicalPopulationError
from .opacity import (
    BALMER_LINES,
    BRACKETT_LINES,
    LYMAN_LINES,
    PASCHEN_LINES,
    HydrogenLine,
    _atmosphere_level_distribution,
    balmer_mass_absorption_coefficient,
    brackett_mass_absorption_coefficient,
    electron_scattering_mass_coefficient,
    hydrogen_continuum_mass_absorption_coefficient,
    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    hydrogen_ground_state_photoionization_cross_section,
    hydrogen_rayleigh_scattering_mass_coefficient,
    lyman_mass_absorption_coefficient,
    optical_depth_from_mass_opacity,
    paschen_mass_absorption_coefficient,
)
from .radiative_transfer import Backend, emergent_flux, radiation_field
from .spectrum import Spectrum, planck_lambda_angstrom


FloatArray = NDArray[np.float64]
_EV_TO_ERG = 1.602_176_634e-12
_LIGHT_SPEED_ANGSTROM_PER_SECOND = LIGHT_SPEED * 1.0e8
_RYDBERG_WAVENUMBER = 109_678.77


# Exact non-relativistic shell-averaged hydrogen absorption oscillator
# strengths.  They were evaluated by summing the analytic dipole radial
# integrals over every l -> l +/- 1 component.  Entries through n=4 reproduce
# the Lyman, Balmer, Paschen, and Brackett constants already used by opacity.py.
_SHELL_OSCILLATOR_STRENGTH = {
    (1, 2): 0.416196717979983,
    (1, 3): 0.0791015624999999,
    (1, 4): 0.028991029248,
    (1, 5): 0.0139383438752513,
    (1, 6): 0.00779949271874015,
    (1, 7): 0.00481395081442315,
    (1, 8): 0.00318342730451621,
    (1, 9): 0.00221610878663833,
    (2, 3): 0.640747044864,
    (2, 4): 0.119321144112231,
    (2, 5): 0.0446702946311311,
    (2, 6): 0.0220928192138672,
    (2, 7): 0.012704624210421,
    (2, 8): 0.00803555852030092,
    (2, 9): 0.00542894641045432,
    (3, 4): 0.842096357532275,
    (3, 5): 0.150584082803107,
    (3, 6): 0.0558402477747258,
    (3, 7): 0.027684887963411,
    (3, 8): 0.0160358274083518,
    (3, 9): 0.0102341026067734,
    (4, 5): 1.03773625884025,
    (4, 6): 0.179252374278725,
    (4, 7): 0.065485989731148,
    (4, 8): 0.0322951583475062,
    (4, 9): 0.0186995864209933,
    (5, 6): 1.23123652646566,
    (5, 7): 0.206892276195322,
    (5, 8): 0.0744792411144178,
    (5, 9): 0.0364540070611578,
    (6, 7): 1.42373131904772,
    (6, 8): 0.234039818868826,
    (6, 9): 0.0831462647974112,
    (7, 8): 1.61568147363079,
    (7, 9): 0.260920065935122,
    (8, 9): 1.80730635264355,
}

# TLUSTY 208's public hydrogenic OSH table extends the precise low-level
# values above to n=20.  Its rounded entries are used only beyond n=9 and are
# the same anchors employed by TLUSTY for its high-Rydberg continuation.
_TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH = (
    (),
    (0.4162,),
    (0.0791, 0.6407),
    (0.02899, 0.1193, 0.8421),
    (0.01394, 0.04467, 0.1506, 1.038),
    (0.007799, 0.02209, 0.05584, 0.1793, 1.231),
    (0.004814, 0.0127, 0.02768, 0.06549, 0.2069, 1.424),
    (0.003183, 0.008036, 0.01604, 0.0323, 0.07448, 0.234, 1.616),
    (0.002216, 0.005429, 0.01023, 0.0187, 0.03645, 0.08315, 0.2609, 1.807),
    (0.001605, 0.003851, 0.00698, 0.01196, 0.02104, 0.04038, 0.09163, 0.2876, 1.999),
    (0.001201, 0.002835, 0.004996, 0.008187, 0.01344, 0.0232, 0.04416, 0.1, 0.3143, 2.19),
    (0.0009214, 0.002151, 0.003711, 0.005886, 0.009209, 0.01479, 0.02525, 0.04787, 0.1083, 0.3408, 2.381),
    (0.0007227, 0.001672, 0.002839, 0.004393, 0.006631, 0.01012, 0.01605, 0.02724, 0.05152, 0.1166, 0.3673, 2.572),
    (0.0005744, 0.001326, 0.002224, 0.003375, 0.004959, 0.007289, 0.01097, 0.01726, 0.02918, 0.05513, 0.1248, 0.3938, 2.763),
    (0.0004686, 0.00107, 0.001776, 0.002656, 0.003821, 0.005455, 0.007891, 0.01177, 0.01843, 0.03109, 0.05872, 0.133, 0.4202, 2.954),
    (0.0003856, 0.0008764, 0.001443, 0.002131, 0.003014, 0.004207, 0.005905, 0.008456, 0.01254, 0.01958, 0.03298, 0.06228, 0.1412, 0.4467, 3.145),
    (0.0003211, 0.000727, 0.001188, 0.001739, 0.002425, 0.003324, 0.004556, 0.006323, 0.008995, 0.01328, 0.0207, 0.03486, 0.06584, 0.1494, 0.4731, 3.336),
    (0.0002702, 0.0006099, 0.0009916, 0.001439, 0.001984, 0.002679, 0.003602, 0.004877, 0.006719, 0.009515, 0.01402, 0.02182, 0.03672, 0.06938, 0.1575, 0.4995, 3.527),
    (0.0002296, 0.0005167, 0.0008361, 0.001204, 0.001646, 0.002196, 0.002905, 0.003856, 0.00518, 0.007099, 0.01002, 0.01474, 0.02292, 0.03858, 0.07292, 0.1657, 0.5259, 3.718),
    (0.0001967, 0.0004416, 0.0007118, 0.001019, 0.001382, 0.001825, 0.002383, 0.003112, 0.004094, 0.005468, 0.007468, 0.01052, 0.01545, 0.02402, 0.04043, 0.07644, 0.1738, 0.5523, 3.909),
)


@dataclass(frozen=True)
class CollisionCrossSection:
    """One shell-averaged electron-impact cross section."""

    incident_energy_ev: FloatArray
    cross_section_cm2: FloatArray


@dataclass(frozen=True)
class HydrogenElectronCollisionData:
    """CCC excitation and ionization cross sections for a hydrogen atom."""

    excitation: Mapping[tuple[int, int], CollisionCrossSection]
    ionization: Mapping[int, CollisionCrossSection]
    maximum_level: int
    source: str

    def excitation_rate_coefficient(
        self,
        temperature: ArrayLike,
        lower_level: int,
        upper_level: int,
    ) -> FloatArray:
        """Return Maxwellian ``<sigma v>`` in cm^3 s^-1."""

        if not 1 <= lower_level < upper_level <= self.maximum_level:
            raise ValueError("collision levels lie outside the loaded atom")
        return _maxwellian_rate_coefficient(
            self.excitation[(lower_level, upper_level)], temperature
        )

    def ionization_rate_coefficient(
        self, temperature: ArrayLike, level: int
    ) -> FloatArray:
        """Return electron-impact ionization ``<sigma v>`` in cm^3 s^-1."""

        if not 1 <= level <= self.maximum_level:
            raise ValueError("ionization level lies outside the loaded atom")
        return _maxwellian_rate_coefficient(self.ionization[level], temperature)


@dataclass(frozen=True)
class MultilevelHydrogenNLTEState:
    """Depth-dependent populations from the fixed-structure rate solve."""

    principal_quantum_number: FloatArray
    population_density: FloatArray
    proton_density: FloatArray
    lte_population_density: FloatArray
    lte_proton_density: FloatArray
    departure_coefficient: FloatArray
    continuum_departure_coefficient: FloatArray
    iterations: int
    converged: bool
    maximum_relative_population_change: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class _ContinuumTransferProblem:
    wavelength_angstrom: FloatArray
    thermal_background_absorption: FloatArray
    bound_free_coefficient: FloatArray
    exp_minus_photon_energy: FloatArray
    spontaneous_intensity_lambda: FloatArray
    scattering: FloatArray
    planck_lambda: FloatArray


@dataclass(frozen=True)
class _LineTransferProblem:
    line: HydrogenLine
    continuum: _ContinuumTransferProblem
    lte_line_opacity: FloatArray
    lower_state_index: int
    upper_state_index: int


def atmosphere_structure_fingerprint(atmosphere: Atmosphere) -> str:
    """Return a stable digest for fixed-structure NLTE state compatibility."""

    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [atmosphere.effective_temperature, atmosphere.logg],
            dtype="<f8",
        ).tobytes()
    )
    for values in (
        atmosphere.column_mass,
        atmosphere.temperature,
        atmosphere.gas_pressure,
        atmosphere.mass_density,
        atmosphere.electron_density,
    ):
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_state_atmosphere(
    atmosphere: Atmosphere, state: MultilevelHydrogenNLTEState
) -> None:
    saved = state.metadata.get("atmosphere_structure_sha256")
    if saved is None:
        raise ValueError(
            "NLTE state does not record its atmosphere structure; recompute it"
        )
    if saved != atmosphere_structure_fingerprint(atmosphere):
        raise ValueError(
            "NLTE state was solved on a different atmosphere structure"
        )


def hydrogen_shell_oscillator_strength(
    lower_level: int, upper_level: int
) -> float:
    """Return the exact shell-averaged H I oscillator strength through n=9."""

    try:
        return _SHELL_OSCILLATOR_STRENGTH[(lower_level, upper_level)]
    except KeyError as error:
        raise ValueError("oscillator strengths are implemented for 1 <= n < n' <= 9") from error


def hydrogen_shell_transition(
    lower_level: int, upper_level: int
) -> HydrogenLine:
    """Construct one shell-averaged hydrogenic radiative transition."""

    oscillator_strength = hydrogen_shell_oscillator_strength(
        lower_level, upper_level
    )
    canonical_series = {
        1: LYMAN_LINES,
        2: BALMER_LINES,
        3: PASCHEN_LINES,
        4: BRACKETT_LINES,
    }
    wavelength = None
    for existing in canonical_series.get(lower_level, ()):
        if existing.upper_level == upper_level:
            wavelength = existing.wavelength_vacuum_angstrom
            break
    if wavelength is None:
        wavelength = 1.0e8 / (
            _RYDBERG_WAVENUMBER
            * (1.0 / lower_level**2 - 1.0 / upper_level**2)
        )
    return HydrogenLine(
        name=f"H{lower_level}-{upper_level}",
        lower_level=lower_level,
        upper_level=upper_level,
        wavelength_vacuum_angstrom=wavelength,
        absorption_oscillator_strength=oscillator_strength,
    )


def _extended_hydrogen_shell_oscillator_strength(
    lower_level: int, upper_level: int
) -> float:
    """Return a shell oscillator strength beyond the explicit CCC atom.

    Exact evaluated values already used by the opacity module are preferred.
    Above the last tabulated member, the hydrogenic ``n^-3`` continuation used
    by TLUSTY is anchored to that member.  This approximation is confined to
    transitions into the optional merged Rydberg reservoir.
    """

    if not 1 <= lower_level < upper_level:
        raise ValueError("hydrogen transition levels must satisfy 1 <= lower < upper")
    exact = _SHELL_OSCILLATOR_STRENGTH.get((lower_level, upper_level))
    if exact is not None:
        return exact
    series = {
        1: LYMAN_LINES,
        2: BALMER_LINES,
        3: PASCHEN_LINES,
        4: BRACKETT_LINES,
    }.get(lower_level, ())
    for line in series:
        if line.upper_level == upper_level:
            return line.absorption_oscillator_strength

    if upper_level <= len(_TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH):
        return _TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH[upper_level - 1][
            lower_level - 1
        ]

    if lower_level >= len(_TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH):
        # TLUSTY/SYNSPEC STARK0 asymptote for transitions whose lower shell
        # itself lies above the public n=20 OSH table.
        lower = float(lower_level)
        upper = float(upper_level)
        return float(
            1.96
            * lower
            * (upper / (upper**2 - lower**2)) ** 3
        )

    anchors = [line for line in series if line.upper_level < upper_level]
    if anchors:
        anchor = anchors[-1]
    else:
        anchor_upper = len(_TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH)
        anchor = _extended_hydrogen_shell_transition(
            lower_level, anchor_upper
        )
    lower_squared = float(lower_level**2)
    anchor_upper = float(anchor.upper_level)
    upper = float(upper_level)
    scale = (
        (anchor_upper**2 - lower_squared)
        / anchor_upper
        * upper
        / (upper**2 - lower_squared)
    ) ** 3
    return float(anchor.absorption_oscillator_strength * scale)


def _extended_hydrogen_shell_transition(
    lower_level: int, upper_level: int
) -> HydrogenLine:
    """Construct a physical shell transition used by the merged reservoir."""

    oscillator_strength = _extended_hydrogen_shell_oscillator_strength(
        lower_level, upper_level
    )
    wavelength = 1.0e8 / (
        _RYDBERG_WAVENUMBER
        * (1.0 / lower_level**2 - 1.0 / upper_level**2)
    )
    return HydrogenLine(
        name=f"H{lower_level}-{upper_level}",
        lower_level=lower_level,
        upper_level=upper_level,
        wavelength_vacuum_angstrom=wavelength,
        absorption_oscillator_strength=oscillator_strength,
    )


def _read_cross_section(handle: ZipFile, name: str) -> CollisionCrossSection:
    rows: list[tuple[float, float]] = []
    try:
        contents = handle.read(name).decode("ascii")
    except KeyError as error:
        raise ValueError(f"CCC archive is missing {name}") from error
    for raw_line in contents.splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        columns = raw_line.split()
        if len(columns) < 2:
            continue
        try:
            energy = float(columns[0])
            cross_section = float(columns[1])
        except ValueError:
            continue
        rows.append((energy, cross_section))
    if len(rows) < 3:
        raise ValueError(f"CCC table {name} contains fewer than three data rows")
    values = np.asarray(rows, dtype=np.float64)
    if (
        np.any(~np.isfinite(values))
        or np.any(values[:, 0] <= 0.0)
        or np.any(values[:, 1] < 0.0)
        or np.any(np.diff(values[:, 0]) <= 0.0)
    ):
        raise ValueError(f"CCC table {name} contains invalid cross sections")
    return CollisionCrossSection(
        incident_energy_ev=np.ascontiguousarray(values[:, 0]),
        cross_section_cm2=np.ascontiguousarray(values[:, 1] * BOHR_RADIUS**2),
    )


def read_ccc_hydrogen_collision_data(
    path: str | Path, *, maximum_level: int = 8
) -> HydrogenElectronCollisionData:
    """Read shell-averaged CCC e--H cross sections from the public ZIP archive.

    Files named ``upper.lower`` contain excitation cross sections averaged over
    the initial shell and summed over the final shell.  ``TICS.lower`` contains
    the corresponding total ionization cross section.  Energies are in eV and
    cross sections in ``a0^2`` in the upstream archive.
    """

    if not 2 <= maximum_level <= 9:
        raise ValueError("CCC shell-averaged data support 2 <= maximum_level <= 9")
    source = Path(path)
    excitation: dict[tuple[int, int], CollisionCrossSection] = {}
    ionization: dict[int, CollisionCrossSection] = {}
    with ZipFile(source) as handle:
        for lower in range(1, maximum_level + 1):
            ionization[lower] = _read_cross_section(handle, f"TICS.{lower}")
            for upper in range(lower + 1, maximum_level + 1):
                excitation[(lower, upper)] = _read_cross_section(
                    handle, f"{upper}.{lower}"
                )
    return HydrogenElectronCollisionData(
        excitation=excitation,
        ionization=ionization,
        maximum_level=maximum_level,
        source=str(source),
    )


def _maxwellian_rate_coefficient(
    cross_section: CollisionCrossSection, temperature: ArrayLike
) -> FloatArray:
    temperature_array = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(temperature_array)) or np.any(temperature_array <= 0.0):
        raise ValueError("temperature must contain finite positive values")
    energy = cross_section.incident_energy_ev * _EV_TO_ERG
    thermal_energy = BOLTZMANN * temperature_array
    exponent = energy.reshape((-1,) + (1,) * thermal_energy.ndim) / thermal_energy
    integrand = (
        cross_section.cross_section_cm2.reshape((-1,) + (1,) * thermal_energy.ndim)
        * energy.reshape((-1,) + (1,) * thermal_energy.ndim)
        * np.exp(-np.minimum(exponent, 745.0))
    )
    integral = trapezoid(integrand, energy, axis=0)
    return np.asarray(
        np.sqrt(8.0 / (PI * ELECTRON_MASS))
        * integral
        / thermal_energy**1.5,
        dtype=np.float64,
    )


def _reference_populations(
    atmosphere: Atmosphere,
    maximum_level: int,
    merged_rydberg_maximum_level: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    physical_maximum = (
        maximum_level
        if merged_rydberg_maximum_level is None
        else merged_rydberg_maximum_level
    )
    distribution = _atmosphere_level_distribution(atmosphere, physical_maximum)
    populations = distribution.population_density[:, :maximum_level]
    occupation = distribution.occupation_probability[:, :maximum_level]
    if merged_rydberg_maximum_level is not None:
        populations = np.column_stack(
            (
                populations,
                np.sum(
                    distribution.population_density[
                        :, maximum_level:merged_rydberg_maximum_level
                    ],
                    axis=1,
                ),
            )
        )
        # The merged state's individual shell probabilities are retained in
        # the rate aggregation.  This population-weighted value is diagnostic
        # only; no physical transition is assigned this single probability.
        merged_population = populations[:, -1]
        merged_occupation = np.divide(
            np.sum(
                distribution.population_density[
                    :, maximum_level:merged_rydberg_maximum_level
                ]
                * distribution.occupation_probability[
                    :, maximum_level:merged_rydberg_maximum_level
                ],
                axis=1,
            ),
            merged_population,
            out=np.zeros_like(merged_population),
            where=merged_population > 0.0,
        )
        occupation = np.column_stack((occupation, merged_occupation))
    proton = np.ascontiguousarray(atmosphere.proton_density)
    tiny = np.finfo(np.float64).tiny
    return (
        np.maximum(np.ascontiguousarray(populations), tiny),
        np.maximum(proton, tiny),
        np.ascontiguousarray(occupation),
    )


def _validate_merged_rydberg_atom(
    maximum_level: int,
    merged_rydberg_maximum_level: int | None,
    collision_maximum_level: int,
) -> int:
    """Validate the model atom and return its physical maximum shell."""

    if merged_rydberg_maximum_level is None:
        return maximum_level
    if not maximum_level < merged_rydberg_maximum_level <= HM_MAX_BOUND_LEVEL:
        raise ValueError(
            "merged_rydberg_maximum_level must exceed maximum_level and not "
            f"exceed the HM EOS limit ({HM_MAX_BOUND_LEVEL})"
        )
    return merged_rydberg_maximum_level


def _excitation_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: FloatArray,
    lower_level: int,
    upper_level: int,
) -> FloatArray:
    """Use CCC where available and the TLUSTY/Mihalas closure above it."""

    if upper_level <= collision_data.maximum_level:
        return collision_data.excitation_rate_coefficient(
            temperature, lower_level, upper_level
        )
    return _mihalas_high_n_excitation_rate_coefficient(
        temperature, lower_level, upper_level
    )


def _ionization_rate_coefficient(
    collision_data: HydrogenElectronCollisionData,
    temperature: FloatArray,
    level: int,
) -> FloatArray:
    """Use CCC ionization through n=9 and TLUSTY/Mihalas above it."""

    if level <= collision_data.maximum_level:
        return collision_data.ionization_rate_coefficient(temperature, level)
    return _mihalas_high_n_ionization_rate_coefficient(temperature, level)


def _finite_population_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Return a bounded detailed-balance population ratio.

    Completely dissolved HM levels can underflow to zero in deep layers.  A
    formally infinite reverse rate for a transition whose survival probability
    is zero is both numerically harmful and physically irrelevant.
    """

    tiny = np.finfo(np.float64).tiny
    log_ratio = (
        np.log(np.maximum(numerator, tiny))
        - np.log(np.maximum(denominator, tiny))
    )
    return np.exp(np.clip(log_ratio, -700.0, np.log(1.0e200)))


def _planck_nu_at_wavelength(
    wavelength_angstrom: float, temperature: FloatArray
) -> FloatArray:
    planck_lambda = planck_lambda_angstrom(wavelength_angstrom, temperature)
    return np.asarray(
        planck_lambda * wavelength_angstrom**2 / _LIGHT_SPEED_ANGSTROM_PER_SECOND,
        dtype=np.float64,
    )


def _exponential_integral_e1(argument: FloatArray) -> FloatArray:
    """TLUSTY's stable rational approximation to E1 for positive arguments."""

    value = np.asarray(argument, dtype=np.float64)
    small = value <= 1.0
    result = np.empty_like(value)
    if np.any(small):
        x = value[small]
        result[small] = (
            -np.log(x)
            - 0.577_215_66
            + x
            * (
                0.999_991_93
                + x
                * (
                    -0.249_910_55
                    + x
                    * (0.055_199_68 + x * (-0.009_760_04 + x * 0.001_078_57))
                )
            )
        )
    if np.any(~small):
        x = value[~small]
        numerator = 0.267_773_434_3 + x * (
            8.634_760_892_5
            + x * (18.059_016_973 + x * (8.573_328_740_1 + x))
        )
        denominator = 3.958_496_922_8 + x * (
            21.099_653_082_7
            + x * (25.632_956_148_6 + x * (9.573_322_345_4 + x))
        )
        result[~small] = np.exp(-x) * numerator / denominator / x
    return result


def _mihalas_high_n_excitation_rate_coefficient(
    temperature: FloatArray,
    lower_level: int,
    upper_level: int,
) -> FloatArray:
    """Approximate an omitted high-n excitation rate in cm^3 s^-1.

    This is the standard Mihalas--Heasley--Auer expression used by TLUSTY for
    its modified hydrogen collision rate.  CCC rates remain in force for all
    transitions present in the downloaded archive; this formula is used only
    beyond that boundary.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    energy_rydberg = 1.0 / lower_level**2 - 1.0 / upper_level**2
    reduced_energy = (
        HYDROGEN_IONIZATION_ENERGY
        * energy_rydberg
        / (BOLTZMANN * temperature)
    )
    exponential = np.exp(-np.minimum(reduced_energy, 745.0))
    e1 = _exponential_integral_e1(reduced_energy)
    e5 = e1
    for order in range(1, 5):
        e5 = (exponential - reduced_energy * e5) / order
    oscillator_strength = _extended_hydrogen_shell_oscillator_strength(
        lower_level, upper_level
    )
    coefficient = (
        4.0
        * 5.465e-11
        * np.sqrt(temperature)
        * oscillator_strength
        / energy_rydberg**2
        * reduced_energy
        * (e1 + 0.148 * reduced_energy * e5)
    )
    if upper_level - lower_level != 1:
        lower = float(lower_level)
        upper = float(upper_level)
        alpha = 1.8 - 0.4 / lower**2
        beta = 3.0 - 1.2 / lower
        coefficient *= beta + 2.0 * (alpha - beta) / (upper - lower)
    return np.maximum(np.asarray(coefficient, dtype=np.float64), 0.0)


def _mihalas_high_n_ionization_rate_coefficient(
    temperature: FloatArray, level: int
) -> FloatArray:
    """Return the physical-shell rate underlying TLUSTY's merged ionization."""

    temperature = np.asarray(temperature, dtype=np.float64)
    reduced_threshold = (
        HYDROGEN_IONIZATION_ENERGY
        / (level**2 * BOLTZMANN * temperature)
    )
    return (
        5.465e-11
        * np.sqrt(temperature)
        * level**3
        * np.exp(-np.minimum(reduced_threshold, 745.0))
    )


def _photoionization_cross_section(
    level: int, frequency_hz: FloatArray
) -> FloatArray:
    threshold = HYDROGEN_IONIZATION_ENERGY / (PLANCK * level**2)
    if level == 1:
        return hydrogen_ground_state_photoionization_cross_section(frequency_hz)
    cross_section = (
        2.815e29
        * frequency_hz**-3
        / level**5
        * hydrogen_bound_free_gaunt_factor(level, frequency_hz)
    )
    return np.where(frequency_hz >= threshold, cross_section, 0.0)


def _default_continuum_wavelength(maximum_level: int) -> FloatArray:
    lyman_limit = PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
    upper = 1.025 * lyman_limit * maximum_level**2
    base = np.geomspace(100.0, upper, 420)
    edges = []
    for level in range(1, maximum_level + 1):
        edge = lyman_limit * level**2
        edges.extend((edge * (1.0 - 2.0e-5), edge, edge * (1.0 + 2.0e-5)))
    return np.unique(np.asarray([*base, *edges], dtype=np.float64))


def _continuum_radiative_rates(
    atmosphere: Atmosphere,
    lte_population: FloatArray,
    lte_proton: FloatArray,
    maximum_level: int,
    continuum_wavelength_angstrom: FloatArray,
    continuum_mean_intensity_lambda: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    wavelength = np.asarray(continuum_wavelength_angstrom, dtype=np.float64)
    mean_lambda = np.asarray(continuum_mean_intensity_lambda, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 8
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("continuum wavelength must be a positive increasing grid")
    if mean_lambda.shape != (wavelength.size, atmosphere.n_depth):
        raise ValueError("continuum mean intensity must have shape (wavelength, depth)")
    if np.any(~np.isfinite(mean_lambda)) or np.any(mean_lambda < 0.0):
        raise ValueError("continuum mean intensity must be finite and non-negative")

    upward, recombination = continuum_integrals(
        wavelength, atmosphere.temperature, mean_lambda, maximum_level, _photoionization_cross_section)
    downward = recombination * _finite_population_ratio(lte_population, lte_proton[:,None])
    return upward, downward


def _bound_bound_radiative_rates(
    temperature: FloatArray,
    line: HydrogenLine,
    lower_population: FloatArray,
    upper_population: FloatArray,
    lower_occupation: FloatArray,
    upper_occupation: FloatArray,
    mean_intensity_nu: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return physical HM bound-bound rates.

    Hummer--Mihalas occupation probabilities multiply a transition by the
    probability that its *final* state exists.  Thus absorption carries
    ``w_upper`` and emission carries ``w_lower``.  This is the rate form used
    by TLUSTY and it recovers exact LTE balance because the reference level
    populations themselves contain ``w_level``.  A former implementation
    multiplied both directions by ``w_upper / w_lower`` and then modified the
    spontaneous rate to force local detailed balance.  That common-factor
    shortcut is harmless for an isolated two-level atom, but changes the
    relative rates of transitions connected to different lower levels and
    biased the cool-DA Balmer departures.
    """

    lower_weight = 2.0 * line.lower_level**2
    upper_weight = 2.0 * line.upper_level**2
    frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
    spontaneous = einstein_a_from_absorption_oscillator_strength(line)
    stimulated_down = spontaneous * LIGHT_SPEED**2 / (
        2.0 * PLANCK * frequency**3
    )
    absorption = upper_weight / lower_weight * stimulated_down
    lower_survival = np.clip(lower_occupation, 0.0, 1.0)
    upper_survival = np.clip(upper_occupation, 0.0, 1.0)
    planck_nu = _planck_nu_at_wavelength(
        line.wavelength_vacuum_angstrom, temperature
    )
    lte_upward = upper_survival * absorption * planck_nu
    physical_lte_downward = lower_survival * (
        spontaneous + stimulated_down * planck_nu
    )
    # The EOS uses the hydrogen ionization energy while canonical line
    # centers use measured vacuum wavelengths.  Remove the resulting ppm
    # inconsistency so a Planck field remains an exact regression invariant;
    # this factor is unity when the level and line energies are identical.
    lte_normalization = np.divide(
        _finite_population_ratio(lower_population, upper_population)
        * lte_upward,
        physical_lte_downward,
        out=np.ones_like(physical_lte_downward),
        where=physical_lte_downward > 0.0,
    )
    upward = upper_survival * absorption * mean_intensity_nu
    downward = lower_survival * (
        spontaneous + stimulated_down * mean_intensity_nu
    ) * lte_normalization
    return np.asarray(upward), np.asarray(downward)


def solve_multilevel_hydrogen_statistical_equilibrium(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    maximum_level: int = 8,
    merged_rydberg_maximum_level: int | None = None,
    line_mean_intensity_nu: Mapping[tuple[int, int], ArrayLike] | None = None,
    continuum_wavelength_angstrom: ArrayLike | None = None,
    continuum_mean_intensity_lambda: ArrayLike | None = None,
    collision_rate_multiplier: float = 1.0,
    fix_continuum_departure: bool = False,
    _return_rate_matrix: bool = False,
    _rate_matrix_transform=None,
) -> MultilevelHydrogenNLTEState:
    """Solve one fixed-radiation statistical-equilibrium rate matrix.

    Missing bound-bound radiation fields default to the local Planck function.
    If no continuum field is supplied, a Planck field is used as well.  That
    mode is valuable as an exact detailed-balance regression: all departure
    coefficients must return unity even with Hummer--Mihalas populations.
    When requested, the final bound state aggregates physical shells above
    ``maximum_level`` through ``merged_rydberg_maximum_level``.
    """

    if not 2 <= maximum_level <= len(
        _TLUSTY_HYDROGEN_OSCILLATOR_STRENGTH
    ):
        raise ValueError("maximum_level must lie in [2, 20]")
    physical_maximum_level = _validate_merged_rydberg_atom(
        maximum_level,
        merged_rydberg_maximum_level,
        collision_data.maximum_level,
    )
    if not np.isfinite(collision_rate_multiplier) or collision_rate_multiplier < 0.0:
        raise ValueError("collision_rate_multiplier must be finite and non-negative")
    lte_population, lte_proton, occupation = _reference_populations(
        atmosphere, maximum_level, merged_rydberg_maximum_level
    )
    physical_distribution = _atmosphere_level_distribution(
        atmosphere, physical_maximum_level
    )
    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    n_depth = atmosphere.n_depth
    bound_state_count = lte_population.shape[1]
    n_state = bound_state_count + 1
    rate = np.zeros((n_depth, n_state, n_state), dtype=np.float64)
    supplied_line_fields = {} if line_mean_intensity_nu is None else line_mean_intensity_nu

    for lower in range(1, maximum_level):
        for upper in range(lower + 1, maximum_level + 1):
            line = (
                hydrogen_shell_transition(lower, upper)
                if upper <= 9
                else _extended_hydrogen_shell_transition(lower, upper)
            )
            field = supplied_line_fields.get((lower, upper))
            if field is None:
                mean_intensity = _planck_nu_at_wavelength(
                    line.wavelength_vacuum_angstrom, temperature
                )
            else:
                mean_intensity = np.asarray(field, dtype=np.float64)
                if mean_intensity.shape != (n_depth,):
                    raise ValueError("each line mean intensity must have one value per depth")
                if np.any(~np.isfinite(mean_intensity)) or np.any(mean_intensity < 0.0):
                    raise ValueError("line mean intensities must be finite and non-negative")

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
                * _excitation_rate_coefficient(
                    collision_data,
                    temperature, lower, upper
                )
                * np.clip(occupation[:, upper - 1], 0.0, 1.0)
            )
            collisional_down = collisional_up * _finite_population_ratio(
                lte_population[:, lower - 1],
                lte_population[:, upper - 1],
            )
            rate[:, lower - 1, upper - 1] += radiative_up + collisional_up
            rate[:, upper - 1, lower - 1] += radiative_down + collisional_down

    if merged_rydberg_maximum_level is not None:
        merged_index = maximum_level
        merged_start = maximum_level + 1
        merged_reference = lte_population[:, merged_index]
        for lower in range(1, maximum_level + 1):
            radiative_up_sum = np.zeros(n_depth, dtype=np.float64)
            radiative_down_sum = np.zeros(n_depth, dtype=np.float64)
            collisional_up_sum = np.zeros(n_depth, dtype=np.float64)
            for upper in range(merged_start, physical_maximum_level + 1):
                line = _extended_hydrogen_shell_transition(lower, upper)
                field = supplied_line_fields.get((lower, upper))
                if field is None:
                    mean_intensity = _planck_nu_at_wavelength(
                        line.wavelength_vacuum_angstrom, temperature
                    )
                else:
                    mean_intensity = np.asarray(field, dtype=np.float64)
                    if mean_intensity.shape != (n_depth,):
                        raise ValueError(
                            "each line mean intensity must have one value per depth"
                        )
                    if np.any(~np.isfinite(mean_intensity)) or np.any(
                        mean_intensity < 0.0
                    ):
                        raise ValueError(
                            "line mean intensities must be finite and non-negative"
                        )
                component_up, component_down = _bound_bound_radiative_rates(
                    temperature,
                    line,
                    physical_distribution.population_density[:, lower - 1],
                    physical_distribution.population_density[:, upper - 1],
                    physical_distribution.occupation_probability[:, lower - 1],
                    physical_distribution.occupation_probability[:, upper - 1],
                    mean_intensity,
                )
                radiative_up_sum += component_up
                radiative_down_sum += component_down * np.divide(
                    physical_distribution.population_density[:, upper - 1],
                    merged_reference,
                    out=np.zeros(n_depth, dtype=np.float64),
                    where=merged_reference > 0.0,
                )
                coefficient = _excitation_rate_coefficient(
                    collision_data, temperature, lower, upper
                )
                collisional_up_sum += (
                    electron_density
                    * coefficient
                    * np.clip(
                        physical_distribution.occupation_probability[
                            :, upper - 1
                        ],
                        0.0,
                        1.0,
                    )
                )

            collisional_up_sum *= collision_rate_multiplier
            collisional_down_sum = (
                collisional_up_sum
                * _finite_population_ratio(
                    lte_population[:, lower - 1], merged_reference
                )
            )
            rate[:, lower - 1, merged_index] += (
                radiative_up_sum + collisional_up_sum
            )
            rate[:, merged_index, lower - 1] += (
                radiative_down_sum + collisional_down_sum
            )

    if continuum_wavelength_angstrom is None:
        continuum_wavelength = _default_continuum_wavelength(
            physical_maximum_level
        )
    else:
        continuum_wavelength = np.asarray(
            continuum_wavelength_angstrom, dtype=np.float64
        )
    if continuum_mean_intensity_lambda is None:
        continuum_mean = planck_lambda_angstrom(
            continuum_wavelength[:, np.newaxis], temperature[np.newaxis, :]
        )
    else:
        continuum_mean = np.asarray(continuum_mean_intensity_lambda, dtype=np.float64)
    photo_up, radiative_recombination = _continuum_radiative_rates(
        atmosphere,
        np.ascontiguousarray(physical_distribution.population_density),
        lte_proton,
        physical_maximum_level,
        continuum_wavelength,
        continuum_mean,
    )
    if merged_rydberg_maximum_level is not None:
        merged_slice = slice(maximum_level, physical_maximum_level)
        merged_reference = lte_population[:, maximum_level]
        photo_up = np.column_stack(
            (
                photo_up[:, :maximum_level],
                np.sum(
                    photo_up[:, merged_slice]
                    * physical_distribution.population_density[:, merged_slice],
                    axis=1,
                )
                / merged_reference,
            )
        )
        radiative_recombination = np.column_stack(
            (
                radiative_recombination[:, :maximum_level],
                np.sum(radiative_recombination[:, merged_slice], axis=1),
            )
        )
    continuum_index = bound_state_count
    for level in range(1, maximum_level + 1):
        collisional_ionization = (
            collision_rate_multiplier
            * electron_density
            * _ionization_rate_coefficient(
                collision_data, temperature, level
            )
        )
        three_body_recombination = collisional_ionization * _finite_population_ratio(
            lte_population[:, level - 1], lte_proton
        )
        rate[:, level - 1, continuum_index] += (
            photo_up[:, level - 1] + collisional_ionization
        )
        rate[:, continuum_index, level - 1] += (
            radiative_recombination[:, level - 1]
            + three_body_recombination
        )

    if merged_rydberg_maximum_level is not None:
        merged_index = maximum_level
        merged_start = maximum_level + 1
        merged_reference = lte_population[:, merged_index]
        weighted_ionization = np.zeros(n_depth, dtype=np.float64)
        for level in range(merged_start, physical_maximum_level + 1):
            coefficient = _ionization_rate_coefficient(
                collision_data, temperature, level
            )
            weighted_ionization += (
                physical_distribution.population_density[:, level - 1]
                * coefficient
            )
        collisional_ionization = (
            collision_rate_multiplier
            * electron_density
            * weighted_ionization
            / merged_reference
        )
        three_body_recombination = (
            collisional_ionization
            * _finite_population_ratio(merged_reference, lte_proton)
        )
        rate[:, merged_index, continuum_index] += (
            photo_up[:, merged_index] + collisional_ionization
        )
        rate[:, continuum_index, merged_index] += (
            radiative_recombination[:, merged_index]
            + three_body_recombination
        )

    if _rate_matrix_transform is not None:
        # Research hook (e.g. MALI preconditioning of the line rates); None
        # leaves the assembled matrix untouched.
        rate = _rate_matrix_transform(rate)
    if _return_rate_matrix:
        return rate
    reference = np.column_stack((lte_population, lte_proton))
    active_density = np.sum(reference, axis=1)
    populations = np.empty_like(reference)
    for depth in range(n_depth):
        matrix = rate[depth].T.copy()
        matrix[np.diag_indices(n_state)] -= np.sum(rate[depth], axis=1)
        # Solve for departure coefficients rather than absolute populations.
        # This removes hundreds of orders of magnitude of avoidable column
        # scaling when high HM levels are almost completely dissolved.
        matrix *= reference[depth][np.newaxis, :]
        right_hand_side = np.zeros(n_state, dtype=np.float64)
        if fix_continuum_departure:
            # Diagnostic closure only: pin b(H II)=1.  A restricted-NLTE
            # calculation normally fixes the thermal/electron structure but
            # still solves the hydrogen particle-conservation equation, as
            # TLUSTY does when its atmospheric structure is held fixed.
            matrix[-1] = 0.0
            matrix[-1, -1] = reference[depth, -1]
            right_hand_side[-1] = reference[depth, -1]
        else:
            matrix[-1] = reference[depth]
            right_hand_side[-1] = active_density[depth]
        row_scale = np.max(np.abs(matrix), axis=1)
        matrix /= np.maximum(row_scale[:, np.newaxis], np.finfo(np.float64).tiny)
        right_hand_side /= np.maximum(row_scale, np.finfo(np.float64).tiny)
        try:
            departure_solution = np.linalg.solve(matrix, right_hand_side)
        except np.linalg.LinAlgError:
            departure_solution = np.linalg.lstsq(
                matrix, right_hand_side, rcond=None
            )[0]
        if (
            np.any(~np.isfinite(departure_solution))
            or np.any(departure_solution <= 0.0)
        ):
            raise NonphysicalPopulationError(f"non-physical statistical-equilibrium solution at depth {depth}")
        populations[depth] = departure_solution * reference[depth]

    # Once an HM level contains less than 1e-60 of the active atomic density,
    # its departure coefficient is numerically undefined and spectroscopically
    # irrelevant.  Pin only those completely dissolved states to their LTE
    # reference.  In the particle-conserving research closure, put the
    # sub-machine-precision normalization difference in the continuum.
    inactive = lte_population < active_density[:, np.newaxis] * 1.0e-60
    populations[:, :bound_state_count][inactive] = lte_population[inactive]
    if fix_continuum_departure:
        populations[:, continuum_index] = lte_proton
    else:
        populations[:, continuum_index] = (
            active_density - np.sum(populations[:, :bound_state_count], axis=1)
        )
    if np.any(populations[:, continuum_index] <= 0.0):
        raise NonphysicalPopulationError("statistical-equilibrium normalization removed the continuum")

    departure = populations / reference
    return MultilevelHydrogenNLTEState(
        principal_quantum_number=np.arange(1, bound_state_count + 1, dtype=np.float64),
        population_density=np.ascontiguousarray(populations[:, :bound_state_count]),
        proton_density=np.ascontiguousarray(populations[:, continuum_index]),
        lte_population_density=lte_population,
        lte_proton_density=lte_proton,
        departure_coefficient=np.ascontiguousarray(departure[:, :bound_state_count]),
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
                f"H I n=1-{maximum_level} plus H II"
                if merged_rydberg_maximum_level is None
                else (
                    f"H I n=1-{maximum_level}, merged "
                    f"n={maximum_level + 1}-{merged_rydberg_maximum_level}, plus H II"
                )
            ),
            "explicit_maximum_level": int(maximum_level),
            "ccc_maximum_level": int(collision_data.maximum_level),
            "collision_closure": (
                f"CCC through n={collision_data.maximum_level}; "
                "TLUSTY/Mihalas modified high-n excitation and ionization above CCC"
            ),
            "merged_rydberg_maximum_level": merged_rydberg_maximum_level,
            "merged_collision_closure": (
                None
                if merged_rydberg_maximum_level is None
                else (
                    f"CCC through n={collision_data.maximum_level}; "
                    "TLUSTY/Mihalas modified high-n excitation and ionization beyond CCC"
                )
            ),
            "structure": "fixed LTE atmosphere and electron density",
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(
                atmosphere
            ),
            "collision_data": collision_data.source,
            "collision_rate_multiplier": float(collision_rate_multiplier),
            "radiation_field": (
                "supplied line/continuum fields with local-Planck fallback"
            ),
            "occupation_probability_closure": (
                "HM LTE reference populations and exact local detailed balance"
            ),
            "continuum_population_closure": (
                "diagnostic H II departure coefficient fixed to unity"
                if fix_continuum_departure
                else (
                    "total active hydrogen conserved with variable H II; "
                    "thermal and electron structure prescribed"
                )
            ),
        },
    )


def _line_grid(line: HydrogenLine) -> FloatArray:
    center = line.wavelength_vacuum_angstrom
    if line.lower_level == 1:
        half_width = max(18.0, 0.025 * center)
    elif line.lower_level == 2:
        half_width = max(45.0, 0.0125 * center)
    else:
        half_width = max(100.0, 0.0125 * center)
    core_width = min(6.0, 0.08 * half_width)
    return np.unique(
        np.r_[
            np.linspace(center - half_width, center + half_width, 81),
            np.linspace(center - core_width, center + core_width, 81),
        ]
    )


def _lte_line_opacity(
    atmosphere: Atmosphere,
    wavelength: FloatArray,
    line: HydrogenLine,
    *,
    include_balmer_self_broadening: bool = True,
    unified_allard_table: object | None = None,
) -> FloatArray:
    if line.lower_level == 1:
        return lyman_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=(line,),
            unified_allard_table=unified_allard_table,
        )
    if line.lower_level == 2:
        return balmer_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=(line,),
            include_self_broadening=include_balmer_self_broadening,
            self_broadening_quadrature_order=32,
        )
    if line.lower_level == 3:
        return paschen_mass_absorption_coefficient(
            atmosphere, wavelength, lines=(line,)
        )
    if line.lower_level == 4:
        return brackett_mass_absorption_coefficient(
            atmosphere, wavelength, lines=(line,)
        )
    raise ValueError("explicit transfer profiles stop at the Brackett series")


def _departure_line_factors(
    atmosphere: Atmosphere,
    line: HydrogenLine,
    departure_coefficient: FloatArray,
    *,
    lower_state_index: int | None = None,
    upper_state_index: int | None = None,
    suppress_population_inversion: bool = False,
) -> tuple[FloatArray, FloatArray]:
    lower_index = (
        line.lower_level - 1 if lower_state_index is None else lower_state_index
    )
    upper_index = (
        line.upper_level - 1 if upper_state_index is None else upper_state_index
    )
    lower = departure_coefficient[:, lower_index]
    upper = departure_coefficient[:, upper_index]
    frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
    exponent = PLANCK * frequency / (BOLTZMANN * atmosphere.temperature)
    stimulated = np.exp(-np.minimum(exponent, 745.0))
    opacity_factor = (lower - upper * stimulated) / (1.0 - stimulated)
    upper_to_lower = upper / lower
    source_denominator = np.exp(exponent) / upper_to_lower - 1.0
    inverted = (
        ~np.isfinite(opacity_factor)
        | (opacity_factor <= 0.0)
        | ~np.isfinite(source_denominator)
        | (source_denominator <= 0.0)
    )
    if np.any(inverted) and not suppress_population_inversion:
        raise NonphysicalPopulationError(
            "multilevel solution produced a bound-bound population inversion "
            f"in {line.lower_level}->{line.upper_level}"
        )
    safe_denominator = np.where(inverted, 1.0, source_denominator)
    source_factor = np.expm1(exponent) / safe_denominator
    if suppress_population_inversion:
        # TLUSTY applies the same local safeguard (its LASER branch): an
        # inverted transition is removed from opacity and emissivity rather
        # than allowing an unsaturated maser into the stellar-atmosphere
        # Lambda iteration.  Retain a finite source placeholder because its
        # opacity is exactly zero in those layers.
        opacity_factor = np.where(inverted, 0.0, opacity_factor)
        source_factor = np.where(inverted, 1.0, source_factor)
    if np.any(~np.isfinite(source_factor)) or np.any(source_factor <= 0.0):
        raise NonphysicalPopulationError("multilevel line source is non-physical")
    return opacity_factor, source_factor


def _prepare_continuum_transfer_problem(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    maximum_level: int,
    merged_rydberg_maximum_level: int | None = None,
) -> _ContinuumTransferProblem:
    """Separate LTE hydrogen bound-free opacity from the thermal background."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("continuum wavelength must be a positive increasing grid")
    physical_maximum_level = (
        maximum_level
        if merged_rydberg_maximum_level is None
        else merged_rydberg_maximum_level
    )
    lte_absorption = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        maximum_level=physical_maximum_level,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
    )
    physical_distribution = _atmosphere_level_distribution(
        atmosphere, physical_maximum_level
    )
    frequency = _LIGHT_SPEED_ANGSTROM_PER_SECOND / wavelength
    temperature = atmosphere.temperature[np.newaxis, :]
    exponent = (
        PLANCK
        * frequency[:, np.newaxis]
        / (BOLTZMANN * temperature)
    )
    exp_minus = np.exp(-np.minimum(exponent, 745.0))
    bound_state_count = maximum_level + int(
        merged_rydberg_maximum_level is not None
    )
    coefficient = np.zeros(
        (wavelength.size, atmosphere.n_depth, bound_state_count), dtype=np.float64
    )
    density = atmosphere.mass_density[np.newaxis, :]
    for level in range(1, physical_maximum_level + 1):
        cross_section = _photoionization_cross_section(level, frequency)
        state_index = (
            level - 1 if level <= maximum_level else maximum_level
        )
        coefficient[:, :, state_index] += (
            cross_section[:, np.newaxis]
            * physical_distribution.population_density[
                np.newaxis, :, level - 1
            ]
            / density
        )
    lte_bound_free = np.sum(
        coefficient * (1.0 - exp_minus)[:, :, np.newaxis], axis=2
    )
    # Both terms above use the same LTE populations and cross sections.  Keep
    # only round-off-scale negative residuals from contaminating the thermal
    # H-/free-free/molecular background after subtraction.
    thermal_background = lte_absorption - lte_bound_free
    tolerance = 2.0e-12 * np.maximum(lte_absorption, 1.0e-40)
    if np.any(thermal_background < -tolerance):
        mismatch = float(np.min(thermal_background / np.maximum(lte_absorption, 1e-40)))
        raise NonphysicalPopulationError(
            "bound-free decomposition is inconsistent with LTE continuum "
            f"opacity (minimum fractional residual {mismatch:.3e})"
        )
    thermal_background = np.maximum(thermal_background, 0.0)
    spontaneous_nu = (
        2.0 * PLANCK * frequency**3 / LIGHT_SPEED**2
    )
    spontaneous_lambda = (
        spontaneous_nu
        * _LIGHT_SPEED_ANGSTROM_PER_SECOND
        / wavelength**2
    )
    scattering = (
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
        + hydrogen_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
    )
    return _ContinuumTransferProblem(
        wavelength_angstrom=wavelength,
        thermal_background_absorption=np.ascontiguousarray(thermal_background),
        bound_free_coefficient=np.ascontiguousarray(coefficient),
        exp_minus_photon_energy=np.ascontiguousarray(exp_minus),
        spontaneous_intensity_lambda=np.ascontiguousarray(
            spontaneous_lambda[:, np.newaxis]
        ),
        scattering=np.ascontiguousarray(scattering),
        planck_lambda=np.ascontiguousarray(
            planck_lambda_angstrom(wavelength[:, np.newaxis], temperature)
        ),
    )


def _nlte_continuum_terms(
    problem: _ContinuumTransferProblem,
    departure_coefficient: FloatArray,
    continuum_departure_coefficient: FloatArray,
    *, allow_signed_absorption: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Return true continuum extinction and thermal emissivity per unit mass."""

    maximum_level = problem.bound_free_coefficient.shape[2]
    if departure_coefficient.shape[1] < maximum_level:
        raise ValueError("departure coefficients do not cover the continuum atom")
    if departure_coefficient.shape[0] != continuum_departure_coefficient.size:
        raise ValueError("bound and continuum departure depth grids differ")
    bound = departure_coefficient[:, :maximum_level][np.newaxis, :, :]
    ion = continuum_departure_coefficient[np.newaxis, :, np.newaxis]
    # The level reduction is local to each wavelength/depth pair. Batch only
    # wavelengths so its arithmetic/order is unchanged, without allocating
    # several full wavelength x depth x level temporaries for every probe.
    extinction = np.empty_like(problem.planck_lambda)
    emissivity = np.empty_like(extinction)
    invalid_extinction = False
    for start in range(0, extinction.shape[0], 256):
        local = slice(start, start + 256)
        coefficient = problem.bound_free_coefficient[local]
        exp_minus = problem.exp_minus_photon_energy[local, :, np.newaxis]
        background = problem.thermal_background_absorption[local]
        extinction[local] = background + np.sum(
            coefficient * (bound - ion * exp_minus), axis=2
        )
        emissivity[local] = background * problem.planck_lambda[local] + np.sum(
            coefficient * ion * exp_minus, axis=2
        ) * problem.spontaneous_intensity_lambda[local]
        if not allow_signed_absorption:
            scale = np.maximum(background + np.sum(coefficient, axis=2), 1.0e-40)
            invalid_extinction |= bool(np.any(extinction[local] <= 1.0e-14 * scale))
    if invalid_extinction:
        raise NonphysicalPopulationError(
            "multilevel solution produced a non-positive total continuum extinction"
        )
    if np.any(~np.isfinite(emissivity)) or np.any(emissivity < 0.0):
        raise NonphysicalPopulationError("multilevel continuum emissivity is non-physical")
    return np.ascontiguousarray(extinction), np.ascontiguousarray(emissivity)


def _nlte_pseudocontinuum_terms(
    atmosphere: Atmosphere,
    problem: _ContinuumTransferProblem,
    departure_coefficient: FloatArray,
    continuum_departure_coefficient: FloatArray,
    *,
    _cache=None,
    lower_levels: tuple[int, ...],
) -> tuple[FloatArray, FloatArray, int]:
    """Return dissolved-series extinction and emissivity in NLTE.

    The DAM/HM pseudo-continuum is treated as a continuation of bound-free
    opacity.  This retains the LTE dissolved oscillator-strength
    redistribution while replacing its stimulated-emission factor by
    ``b_lower - b_HII exp(-h nu/kT)`` and its Milne emissivity by the H II
    departure coefficient.  Locally inverted pseudo-continuum transitions
    use the same zero-opacity safeguard as explicit inverted lines.
    """

    if not lower_levels:
        zeros = np.zeros_like(problem.planck_lambda)
        return zeros, zeros.copy(), 0
    if max(lower_levels) > departure_coefficient.shape[1]:
        raise ValueError("departure coefficients do not cover pseudo-continuum lower levels")
    exp_minus = problem.exp_minus_photon_energy
    spontaneous = problem.spontaneous_intensity_lambda
    ion = continuum_departure_coefficient[np.newaxis, :]
    extinction = np.zeros_like(problem.planck_lambda)
    emissivity = np.zeros_like(problem.planck_lambda)
    suppressed = 0
    for lower in lower_levels:
        lte_opacity = _cached_transfer(_cache, ('h-pseudocontinuum', lower), lambda:
            hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                atmosphere,
                problem.wavelength_angstrom,
                lower_levels=(lower,),
            )
        )
        coefficient = np.divide(
            lte_opacity,
            1.0 - exp_minus,
            out=np.zeros_like(lte_opacity),
            where=(1.0 - exp_minus) > np.finfo(np.float64).eps,
        )
        factor = (
            departure_coefficient[:, lower - 1][np.newaxis, :]
            - ion * exp_minus
        )
        inverted = (coefficient > 0.0) & (factor <= 0.0)
        suppressed += int(np.count_nonzero(inverted))
        active = ~inverted
        extinction += np.where(active, coefficient * factor, 0.0)
        emissivity += np.where(
            active,
            coefficient * ion * exp_minus * spontaneous,
            0.0,
        )
    return (
        np.ascontiguousarray(extinction),
        np.ascontiguousarray(emissivity),
        suppressed,
    )


def _prepare_line_transfer_problem(
    atmosphere: Atmosphere,
    line: HydrogenLine,
    maximum_level: int,
    merged_rydberg_maximum_level: int | None = None,
) -> _LineTransferProblem:
    wavelength = _line_grid(line)
    return _LineTransferProblem(
        line=line,
        continuum=_prepare_continuum_transfer_problem(
            atmosphere,
            wavelength,
            maximum_level,
            merged_rydberg_maximum_level,
        ),
        lte_line_opacity=_lte_line_opacity(atmosphere, wavelength, line),
        lower_state_index=line.lower_level - 1,
        upper_state_index=(
            line.upper_level - 1
            if line.upper_level <= maximum_level
            else maximum_level
        ),
    )


def _line_mean_intensity(
    atmosphere: Atmosphere,
    problem: _LineTransferProblem,
    departure_coefficient: FloatArray,
    continuum_departure_coefficient: FloatArray,
    *,
    n_angle: int,
) -> FloatArray:
    opacity_factor, source_factor = _departure_line_factors(
        atmosphere,
        problem.line,
        departure_coefficient,
        lower_state_index=problem.lower_state_index,
        upper_state_index=problem.upper_state_index,
        suppress_population_inversion=True,
    )
    continuum_absorption, continuum_emissivity = _nlte_continuum_terms(
        problem.continuum,
        departure_coefficient,
        continuum_departure_coefficient,
    )
    line_opacity = problem.lte_line_opacity * opacity_factor[np.newaxis, :]
    line_source = (
        problem.continuum.planck_lambda * source_factor[np.newaxis, :]
    )
    total_opacity = (
        continuum_absorption + problem.continuum.scattering + line_opacity
    )
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total_opacity
    )
    source = (
        continuum_emissivity
        + line_opacity * line_source
        + problem.continuum.scattering * problem.continuum.planck_lambda
    ) / total_opacity
    field = None
    for _ in range(3):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                continuum_emissivity
                + line_opacity * line_source
                + problem.continuum.scattering * field.mean_intensity
            )
            / total_opacity
        )
    assert field is not None
    return _profile_averaged_mean_intensity_nu(
        problem.continuum.wavelength_angstrom,
        line_opacity,
        field.mean_intensity,
    )


def _continuum_radiation_field(
    atmosphere: Atmosphere,
    problem: _ContinuumTransferProblem,
    departure_coefficient: FloatArray,
    continuum_departure_coefficient: FloatArray,
    *,
    n_angle: int,
) -> FloatArray:
    absorption, emissivity = _nlte_continuum_terms(
        problem, departure_coefficient, continuum_departure_coefficient
    )
    total = absorption + problem.scattering
    optical_depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    source = (emissivity + problem.scattering * problem.planck_lambda) / total
    field = None
    for _ in range(4):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (emissivity + problem.scattering * field.mean_intensity) / total
        )
    assert field is not None
    return field.mean_intensity


def _normalize_departure_coefficients(
    lte_population: FloatArray,
    lte_proton: FloatArray,
    departure_coefficient: FloatArray,
    continuum_departure_coefficient: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Restore atomic-particle conservation after a nonlinear population mix."""

    reference_total = np.sum(lte_population, axis=1) + lte_proton
    mixed_total = (
        np.sum(lte_population * departure_coefficient, axis=1)
        + lte_proton * continuum_departure_coefficient
    )
    scale = reference_total / mixed_total
    return (
        np.ascontiguousarray(departure_coefficient * scale[:, np.newaxis]),
        np.ascontiguousarray(continuum_departure_coefficient * scale),
    )


def remap_multilevel_hydrogen_state(
    atmosphere: Atmosphere,
    state: MultilevelHydrogenNLTEState,
) -> MultilevelHydrogenNLTEState:
    """Carry departure coefficients to a nearby structure.

    This is a frozen-departure linearization helper, not a statistical-
    equilibrium solve.  The LTE reference populations and particle
    normalization are rebuilt on ``atmosphere`` so opacity/emissivity
    derivatives can include the EOS and Saha response without accepting a
    stale structure fingerprint.
    """

    maximum_level = int(
        state.metadata.get(
            "explicit_maximum_level", state.departure_coefficient.shape[1]
        )
    )
    merged_value = state.metadata.get("merged_rydberg_maximum_level")
    merged_maximum = None if merged_value is None else int(merged_value)
    lte_population, lte_proton, _ = _reference_populations(
        atmosphere, maximum_level, merged_maximum
    )
    if lte_population.shape != state.departure_coefficient.shape:
        raise ValueError("NLTE state and target atmosphere model atoms differ")
    departure, continuum_departure = _normalize_departure_coefficients(
        lte_population,
        lte_proton,
        state.departure_coefficient,
        state.continuum_departure_coefficient,
    )
    metadata = dict(state.metadata)
    metadata.update(
        {
            "atmosphere_structure_sha256": atmosphere_structure_fingerprint(
                atmosphere
            ),
            "population_mapping": (
                "frozen departure coefficients remapped to local LTE reference"
            ),
        }
    )
    return MultilevelHydrogenNLTEState(
        principal_quantum_number=state.principal_quantum_number,
        population_density=np.ascontiguousarray(lte_population * departure),
        proton_density=np.ascontiguousarray(
            lte_proton * continuum_departure
        ),
        lte_population_density=lte_population,
        lte_proton_density=lte_proton,
        departure_coefficient=departure,
        continuum_departure_coefficient=continuum_departure,
        iterations=0,
        converged=False,
        maximum_relative_population_change=0.0,
        metadata=metadata,
    )


def _anderson_log_population_update(
    log_population: FloatArray,
    log_fixed_point: FloatArray,
    history: list[tuple[FloatArray, FloatArray]],
    *,
    depth: int,
    mixing: float,
    maximum_step: float,
    residual_weights: FloatArray | None = None,
) -> tuple[FloatArray | None, str]:
    """Return a safeguarded type-II Anderson population update.

    Optional weights scale the residual fit, not the physical populations or
    fixed point. PG1159 uses its population convergence floors here so nearly
    empty levels cannot dominate the acceleration's least-squares objective.
    """

    residual = np.ascontiguousarray(log_fixed_point - log_population)
    history.append((log_population.copy(), residual.copy()))
    if len(history) > depth:
        del history[:-depth]
    if depth < 2 or len(history) < 3:
        return None, "history"
    states = np.column_stack([item[0].ravel() for item in history])
    residuals = np.column_stack([item[1].ravel() for item in history])
    state_difference = np.diff(states, axis=1)
    residual_difference = np.diff(residuals, axis=1)
    fit_difference = residual_difference
    fit_residual = residual.ravel()
    if residual_weights is not None:
        weights = np.asarray(residual_weights, dtype=float).ravel()
        if (weights.shape != fit_residual.shape or np.any(~np.isfinite(weights))
                or np.any(weights < 0)):
            raise ValueError("Anderson residual weights must be finite, nonnegative and match the state")
        fit_difference = weights[:, None] * residual_difference
        fit_residual = weights * fit_residual
    gram = fit_difference.T @ fit_difference / residual_difference.shape[0]
    right_hand_side = fit_difference.T @ fit_residual / residual_difference.shape[0]
    regularization = 1.0e-4 * max(
        float(np.trace(gram) / gram.shape[0]), 1.0e-20
    )
    gram[np.diag_indices(gram.shape[0])] += regularization
    try:
        coefficient = np.linalg.solve(gram, right_hand_side)
    except np.linalg.LinAlgError:
        return None, "linear_solve"
    if np.any(~np.isfinite(coefficient)):
        return None, "coefficient"
    predicted_residual = fit_residual - fit_difference @ coefficient
    if np.linalg.norm(predicted_residual) >= np.linalg.norm(fit_residual):
        return None, "prediction"
    step = (
        mixing * residual.ravel()
        - (state_difference + mixing * residual_difference) @ coefficient
    )
    proposed = log_population + step.reshape(log_population.shape)
    if (
        np.any(~np.isfinite(proposed))
        or np.max(np.abs(proposed - log_population)) > maximum_step
    ):
        return None, "step"
    return np.ascontiguousarray(proposed), "accepted"


def solve_multilevel_hydrogen_nlte(
    atmosphere: Atmosphere,
    collision_data: HydrogenElectronCollisionData,
    *,
    maximum_level: int = 8,
    merged_rydberg_maximum_level: int | None = None,
    merged_transfer_maximum_level: int | None = None,
    n_angle: int = 3,
    maximum_iterations: int = 160,
    relative_tolerance: float = 5.0e-3,
    population_damping: float = 0.45,
    collision_rate_multiplier: float = 1.0,
    fix_continuum_departure: bool = False,
    explicit_maximum_lower_level: int = 2,
    acceleration: Literal["none", "anderson"] = "anderson",
    anderson_depth: int = 4,
    anderson_mixing: float = 0.65,
    anderson_maximum_log_step: float = 0.75,
    initial_departure_coefficient: ArrayLike | None = None,
    initial_continuum_departure_coefficient: ArrayLike | None = None,
) -> MultilevelHydrogenNLTEState:
    """Iterate multilevel populations with line and bound-free transfer.

    Set ``merged_rydberg_maximum_level=40`` to append one rate state whose LTE
    reference population and continuum rates sum physical shells above
    ``maximum_level``.  ``merged_transfer_maximum_level`` independently sets
    how far into that reservoir the available line profiles are transferred;
    its default transfers only the first merged member for speed.
    """

    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    physical_maximum_level = _validate_merged_rydberg_atom(
        maximum_level,
        merged_rydberg_maximum_level,
        collision_data.maximum_level,
    )
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(population_damping) or not 0.0 < population_damping <= 1.0:
        raise ValueError("population_damping must lie in (0, 1]")
    if not 1 <= explicit_maximum_lower_level <= min(4, maximum_level - 1):
        raise ValueError("explicit_maximum_lower_level must lie in [1, min(4, n-1)]")
    if acceleration not in ("none", "anderson"):
        raise ValueError("acceleration must be 'none' or 'anderson'")
    if anderson_depth < 2:
        raise ValueError("anderson_depth must be at least two")
    if not np.isfinite(anderson_mixing) or not 0.0 <= anderson_mixing <= 1.0:
        raise ValueError("anderson_mixing must lie in [0, 1]")
    if (
        not np.isfinite(anderson_maximum_log_step)
        or anderson_maximum_log_step <= 0.0
    ):
        raise ValueError("anderson_maximum_log_step must be finite and positive")

    if merged_rydberg_maximum_level is None:
        if merged_transfer_maximum_level is not None:
            raise ValueError(
                "merged_transfer_maximum_level requires a merged Rydberg reservoir"
            )
        transfer_maximum_level = maximum_level
    else:
        transfer_maximum_level = (
            maximum_level + 1
            if merged_transfer_maximum_level is None
            else merged_transfer_maximum_level
        )
        if not maximum_level <= transfer_maximum_level <= physical_maximum_level:
            raise ValueError(
                "merged_transfer_maximum_level must not lie below maximum_level "
                "and within the merged reservoir"
            )

    continuum_wavelength = _default_continuum_wavelength(
        physical_maximum_level
    )
    continuum_problem = _prepare_continuum_transfer_problem(
        atmosphere,
        continuum_wavelength,
        maximum_level,
        merged_rydberg_maximum_level,
    )
    profile_maximum = {1: 21, 2: 22, 3: 22, 4: 14}
    transitions = tuple(
        (
            hydrogen_shell_transition(lower, upper)
            if upper <= 9
            else _extended_hydrogen_shell_transition(lower, upper)
        )
        for lower in range(1, explicit_maximum_lower_level + 1)
        for upper in range(
            lower + 1,
            min(transfer_maximum_level, profile_maximum[lower]) + 1,
        )
    )
    problems = tuple(
        _prepare_line_transfer_problem(
            atmosphere,
            line,
            maximum_level,
            merged_rydberg_maximum_level,
        )
        for line in transitions
    )
    lte_population, lte_proton, _ = _reference_populations(
        atmosphere, maximum_level, merged_rydberg_maximum_level
    )
    bound_state_count = lte_population.shape[1]
    if (initial_departure_coefficient is None) != (
        initial_continuum_departure_coefficient is None
    ):
        raise ValueError(
            "initial bound and continuum departure coefficients must be supplied together"
        )
    warm_started = initial_departure_coefficient is not None
    if warm_started:
        departure = np.asarray(
            initial_departure_coefficient, dtype=np.float64
        ).copy()
        continuum_departure = np.asarray(
            initial_continuum_departure_coefficient, dtype=np.float64
        ).copy()
        if departure.shape != lte_population.shape:
            raise ValueError(
                "initial departure coefficients do not match the current depth/model-atom grid"
            )
        if continuum_departure.shape != lte_proton.shape:
            raise ValueError(
                "initial continuum departure coefficient does not match the depth grid"
            )
        if (
            np.any(~np.isfinite(departure))
            or np.any(departure <= 0.0)
            or np.any(~np.isfinite(continuum_departure))
            or np.any(continuum_departure <= 0.0)
        ):
            raise ValueError("initial departure coefficients must be finite and positive")
        departure, continuum_departure = _normalize_departure_coefficients(
            lte_population,
            lte_proton,
            departure,
            continuum_departure,
        )
    else:
        departure = np.ones_like(lte_population)
        continuum_departure = np.ones_like(lte_proton)
    converged = False
    maximum_change = np.inf
    solution = None
    acceleration_history: list[tuple[FloatArray, FloatArray]] = []
    accelerated_steps = 0
    acceleration_rejections: dict[str, int] = {}
    for iteration in range(1, maximum_iterations + 1):
        continuum_mean = _continuum_radiation_field(
            atmosphere,
            continuum_problem,
            departure,
            continuum_departure,
            n_angle=n_angle,
        )
        line_fields = {
            (problem.line.lower_level, problem.line.upper_level): _line_mean_intensity(
                atmosphere,
                problem,
                departure,
                continuum_departure,
                n_angle=n_angle,
            )
            for problem in problems
        }
        candidate = solve_multilevel_hydrogen_statistical_equilibrium(
            atmosphere,
            collision_data,
            maximum_level=maximum_level,
            merged_rydberg_maximum_level=merged_rydberg_maximum_level,
            line_mean_intensity_nu=line_fields,
            continuum_wavelength_angstrom=continuum_wavelength,
            continuum_mean_intensity_lambda=continuum_mean,
            collision_rate_multiplier=collision_rate_multiplier,
            fix_continuum_departure=fix_continuum_departure,
        )
        candidate_departure = candidate.departure_coefficient
        candidate_continuum = candidate.continuum_departure_coefficient
        log_population = np.column_stack(
            (np.log(departure), np.log(continuum_departure))
        )
        log_fixed_point = np.column_stack(
            (np.log(candidate_departure), np.log(candidate_continuum))
        )
        residual = log_fixed_point - log_population
        maximum_change = float(population_damping * np.max(np.abs(residual)))
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
            proposed = log_population + population_damping * residual
        else:
            accelerated_steps += 1
        proposed_departure = np.exp(proposed[:, :bound_state_count])
        proposed_continuum = np.exp(proposed[:, bound_state_count])
        if fix_continuum_departure:
            departure = np.ascontiguousarray(proposed_departure)
            continuum_departure = np.ones_like(lte_proton)
        else:
            departure, continuum_departure = _normalize_departure_coefficients(
                lte_population,
                lte_proton,
                proposed_departure,
                proposed_continuum,
            )
        solution = candidate
        if maximum_change < relative_tolerance:
            converged = True
            break
    assert solution is not None
    return MultilevelHydrogenNLTEState(
        principal_quantum_number=solution.principal_quantum_number,
        population_density=np.ascontiguousarray(lte_population * departure),
        proton_density=np.ascontiguousarray(lte_proton * continuum_departure),
        lte_population_density=lte_population,
        lte_proton_density=lte_proton,
        departure_coefficient=np.ascontiguousarray(departure),
        continuum_departure_coefficient=np.ascontiguousarray(continuum_departure),
        iterations=iteration,
        converged=converged,
        maximum_relative_population_change=maximum_change,
        metadata={
            **solution.metadata,
            "transfer_iteration": (
                "Lambda iteration with safeguarded Anderson population mixing"
                if acceleration == "anderson"
                else "ordinary damped Lambda iteration"
            ),
            "explicit_bound_bound_transfer": (
                f"series with lower levels 1--{explicit_maximum_lower_level} "
                f"through physical n={transfer_maximum_level}"
            ),
            "merged_transfer_maximum_level": (
                None
                if merged_rydberg_maximum_level is None
                else int(transfer_maximum_level)
            ),
            "higher_series_closure": (
                f"lower levels above {explicit_maximum_lower_level} held at "
                "local detailed balance"
            ),
            "population_inversion_policy": (
                "TLUSTY LASER-style local suppression of inverted transitions "
                "during the rate-transfer iteration"
            ),
            "continuum_transfer": (
                "population-dependent H I bound-free extinction and Milne "
                "emissivity; fixed thermal H-/free-free/molecular background"
            ),
            "continuum_population_closure": (
                "diagnostic H II departure coefficient fixed to unity"
                if fix_continuum_departure
                else (
                    "total active hydrogen conserved with variable H II; "
                    "thermal and electron structure prescribed"
                )
            ),
            "acceleration": acceleration,
            "anderson_depth": int(anderson_depth),
            "warm_started": bool(warm_started),
            "accelerated_steps": int(accelerated_steps),
            "acceleration_rejections": acceleration_rejections,
            "iterations": iteration,
            "converged": converged,
            "maximum_log_population_change": maximum_change,
            "n_angle": int(n_angle),
        },
    )


def synthesize_multilevel_balmer_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    collision_data: HydrogenElectronCollisionData,
    *,
    nlte_state: MultilevelHydrogenNLTEState | None = None,
    maximum_level: int = 8,
    merged_rydberg_maximum_level: int | None = None,
    merged_transfer_maximum_level: int | None = None,
    include_balmer_self_broadening: bool = True,
    n_angle: int = 4,
    backend: Backend = "auto",
    solver_n_angle: int = 3,
    solver_maximum_iterations: int = 160,
    solver_relative_tolerance: float = 5.0e-3,
    solver_population_damping: float = 0.45,
    collision_rate_multiplier: float = 1.0,
    fix_continuum_departure: bool = False,
) -> Spectrum:
    """Synthesize Balmer lines with multilevel opacity and source departures."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a positive increasing 1D grid")
    if nlte_state is None:
        nlte_state = solve_multilevel_hydrogen_nlte(
            atmosphere,
            collision_data,
            maximum_level=maximum_level,
            merged_rydberg_maximum_level=merged_rydberg_maximum_level,
            merged_transfer_maximum_level=merged_transfer_maximum_level,
            n_angle=solver_n_angle,
            maximum_iterations=solver_maximum_iterations,
            relative_tolerance=solver_relative_tolerance,
            population_damping=solver_population_damping,
            collision_rate_multiplier=collision_rate_multiplier,
            fix_continuum_departure=fix_continuum_departure,
        )
    if nlte_state.departure_coefficient.shape[0] != atmosphere.n_depth:
        raise ValueError("NLTE state and atmosphere depth grids differ")
    _validate_state_atmosphere(atmosphere, nlte_state)
    state_explicit_maximum = int(
        nlte_state.metadata.get("explicit_maximum_level", maximum_level)
    )
    state_merged_maximum = nlte_state.metadata.get(
        "merged_rydberg_maximum_level"
    )
    if merged_rydberg_maximum_level is None and state_merged_maximum is not None:
        merged_rydberg_maximum_level = int(state_merged_maximum)
    maximum_level = min(maximum_level, state_explicit_maximum)
    expected_bound_states = maximum_level + int(
        merged_rydberg_maximum_level is not None
    )
    if nlte_state.departure_coefficient.shape[1] < expected_bound_states:
        raise ValueError("NLTE state does not cover the requested model atom")

    continuum_problem = _prepare_continuum_transfer_problem(
        atmosphere,
        wavelength,
        maximum_level,
        merged_rydberg_maximum_level,
    )
    continuum_absorption, continuum_emissivity = _nlte_continuum_terms(
        continuum_problem,
        nlte_state.departure_coefficient,
        nlte_state.continuum_departure_coefficient,
    )
    scattering = continuum_problem.scattering
    planck = continuum_problem.planck_lambda
    line_opacity = np.zeros_like(continuum_absorption)
    line_emissivity = np.zeros_like(continuum_absorption)
    balmer_maximum_level = (
        maximum_level
        if merged_rydberg_maximum_level is None
        else min(merged_rydberg_maximum_level, BALMER_LINES[-1].upper_level)
    )
    for upper in range(3, balmer_maximum_level + 1):
        line = (
            hydrogen_shell_transition(2, upper)
            if upper <= 9
            else _extended_hydrogen_shell_transition(2, upper)
        )
        lte_opacity = balmer_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=(line,),
            include_self_broadening=include_balmer_self_broadening,
        )
        opacity_factor, source_factor = _departure_line_factors(
            atmosphere,
            line,
            nlte_state.departure_coefficient,
            upper_state_index=(
                upper - 1 if upper <= maximum_level else maximum_level
            ),
        )
        opacity = lte_opacity * opacity_factor[np.newaxis, :]
        line_opacity += opacity
        line_emissivity += opacity * planck * source_factor[np.newaxis, :]

    total_opacity = continuum_absorption + scattering + line_opacity
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total_opacity
    )
    source = (
        continuum_emissivity
        + line_emissivity
        + scattering * planck
    ) / total_opacity
    for _ in range(5):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                continuum_emissivity
                + line_emissivity
                + scattering * field.mean_intensity
            )
            / total_opacity
        )
    flux = emergent_flux(optical_depth, source, n_angle=n_angle, backend=backend)
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "transfer": (
                "fixed-structure multilevel H I statistical equilibrium with "
                "population-dependent bound-free transfer"
            ),
            "nlte_status": "experimental opt-in; not used by default DA synthesis",
            "applicability_warning": (
                "fixed-LTE-structure line-formation experiment; not validated "
                "for Balmer-wing fitting or as a temperature-switched "
                "production correction"
            ),
            "model_atom": nlte_state.metadata.get("model_atom"),
            "merged_rydberg_maximum_level": merged_rydberg_maximum_level,
            "collision_data": collision_data.source,
            "continuum_population_closure": nlte_state.metadata.get(
                "continuum_population_closure"
            ),
            "population_iterations": nlte_state.iterations,
            "population_converged": nlte_state.converged,
            "maximum_log_population_change": (
                nlte_state.maximum_relative_population_change
            ),
            "minimum_n2_departure": float(
                np.min(nlte_state.departure_coefficient[:, 1])
            ),
            "maximum_n2_departure": float(
                np.max(nlte_state.departure_coefficient[:, 1])
            ),
            "n_angle": int(n_angle),
        },
    )


def hydrogen_nlte_transfer_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    nlte_state: MultilevelHydrogenNLTEState,
    *,
    _cache=None,
    _allow_signed_continuum=False,
    maximum_level: int = 8,
    merged_rydberg_maximum_level: int | None = None,
    include_lines: bool = True,
    include_lyman: bool = True,
    include_balmer: bool = True,
    include_paschen: bool = True,
    include_brackett: bool = True,
    include_balmer_self_broadening: bool = True,
    include_series_pseudocontinuum: bool = True,
    unified_allard_table: object | None = None,
) -> NLTETransferCoefficients:
    """Assemble population-dependent H opacity and emissivity.

    This is the common physics boundary used by fixed-structure spectrum
    synthesis and by the self-consistent atmosphere iteration.  Returning
    true absorption, emissivity, and scattering separately is essential:
    radiative equilibrium in NLTE is ``integral(kappa J - eta) dlambda = 0``,
    not the LTE-only ``integral kappa(J-B) dlambda = 0``.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if _cache is not None:
        _cache.validate(atmosphere, wavelength)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a positive increasing 1D grid")
    if not any((include_lyman, include_balmer, include_paschen, include_brackett)):
        raise ValueError("at least one hydrogen series must be enabled")
    enabled_lower_levels = tuple(
        lower
        for lower, enabled in enumerate(
            (include_lyman, include_balmer, include_paschen, include_brackett),
            start=1,
        )
        if enabled
    )
    if nlte_state.departure_coefficient.shape[0] != atmosphere.n_depth:
        raise ValueError("NLTE state and atmosphere depth grids differ")
    _validate_state_atmosphere(atmosphere, nlte_state)
    state_explicit_maximum = int(
        nlte_state.metadata.get("explicit_maximum_level", maximum_level)
    )
    state_merged_maximum = nlte_state.metadata.get("merged_rydberg_maximum_level")
    if merged_rydberg_maximum_level is None and state_merged_maximum is not None:
        merged_rydberg_maximum_level = int(state_merged_maximum)
    maximum_level = min(maximum_level, state_explicit_maximum)
    expected_bound_states = maximum_level + int(
        merged_rydberg_maximum_level is not None
    )
    if nlte_state.departure_coefficient.shape[1] < expected_bound_states:
        raise ValueError("NLTE state does not cover the requested model atom")

    continuum_problem = _cached_transfer(_cache, ('h-continuum', maximum_level, merged_rydberg_maximum_level), lambda: _prepare_continuum_transfer_problem(
        atmosphere,
        wavelength,
        maximum_level,
        merged_rydberg_maximum_level,
    ))
    continuum_absorption, continuum_emissivity = _nlte_continuum_terms(
        continuum_problem,
        nlte_state.departure_coefficient,
        nlte_state.continuum_departure_coefficient,
        allow_signed_absorption=_allow_signed_continuum,
    )
    suppressed_pseudocontinuum_inversions = 0
    if include_series_pseudocontinuum:
        pseudo_absorption, pseudo_emissivity, suppressed_pseudocontinuum_inversions = (
            _nlte_pseudocontinuum_terms(
                atmosphere,
                continuum_problem,
                nlte_state.departure_coefficient,
                nlte_state.continuum_departure_coefficient,
                lower_levels=enabled_lower_levels, _cache=_cache,
            )
        )
        continuum_absorption += pseudo_absorption
        continuum_emissivity += pseudo_emissivity

    line_opacity = np.zeros_like(continuum_absorption)
    line_emissivity = np.zeros_like(continuum_absorption)
    physical_maximum = (
        maximum_level
        if merged_rydberg_maximum_level is None
        else merged_rydberg_maximum_level
    )
    profile_maximum = {1: 21, 2: 22, 3: 22, 4: 14}
    suppressed_line_inversions = 0
    planck = continuum_problem.planck_lambda
    for lower in enabled_lower_levels if include_lines else ():
        for upper in range(
            lower + 1,
            min(physical_maximum, profile_maximum[lower]) + 1,
        ):
            line = (
                hydrogen_shell_transition(lower, upper)
                if upper <= 9
                else _extended_hydrogen_shell_transition(lower, upper)
            )
            lte_opacity = _cached_transfer(_cache, ('h-line', lower, upper), lambda: _lte_line_opacity(
                atmosphere,
                wavelength,
                line,
                include_balmer_self_broadening=include_balmer_self_broadening,
                unified_allard_table=(
                    unified_allard_table if lower == 1 else None
                ),
            ))
            opacity_factor, source_factor = _departure_line_factors(
                atmosphere,
                line,
                nlte_state.departure_coefficient,
                upper_state_index=(
                    upper - 1 if upper <= maximum_level else maximum_level
                ),
                suppress_population_inversion=True,
            )
            suppressed_line_inversions += int(
                np.count_nonzero(opacity_factor == 0.0)
            )
            opacity = lte_opacity * opacity_factor[np.newaxis, :]
            line_opacity += opacity
            line_emissivity += opacity * planck * source_factor[np.newaxis, :]

    # Thomson scattering follows the charge/electron closure in Atmosphere.
    # H I Rayleigh scattering follows the NLTE ground-state population; the
    # LTE coefficient returned by the opacity module therefore carries b1.
    scattering = (
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
        + hydrogen_rayleigh_scattering_mass_coefficient(
            atmosphere, wavelength
        )
        * nlte_state.departure_coefficient[:, 0][np.newaxis, :]
    )
    return NLTETransferCoefficients(
        wavelength_angstrom=wavelength,
        true_absorption=np.ascontiguousarray(
            continuum_absorption + line_opacity
        ),
        thermal_emissivity=np.ascontiguousarray(
            continuum_emissivity + line_emissivity
        ),
        scattering=np.ascontiguousarray(scattering),
        metadata={
            "hydrogen_series_lower_levels": enabled_lower_levels,
            "allard_lyman_profiles": bool(unified_allard_table is not None),
            "suppressed_pseudocontinuum_inversions": int(
                suppressed_pseudocontinuum_inversions
            ),
            "suppressed_line_inversions": int(suppressed_line_inversions),
            "dissolved_level_pseudocontinuum": bool(
                include_series_pseudocontinuum
            ),
        },
    )


def synthesize_multilevel_hydrogen_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    collision_data: HydrogenElectronCollisionData,
    *,
    nlte_state: MultilevelHydrogenNLTEState | None = None,
    maximum_level: int = 8,
    merged_rydberg_maximum_level: int | None = None,
    merged_transfer_maximum_level: int | None = None,
    include_lyman: bool = True,
    include_balmer: bool = True,
    include_paschen: bool = True,
    include_brackett: bool = True,
    include_balmer_self_broadening: bool = True,
    include_series_pseudocontinuum: bool = True,
    unified_allard_table: object | None = None,
    n_angle: int = 4,
    backend: Backend = "auto",
    solver_n_angle: int = 3,
    solver_maximum_iterations: int = 160,
    solver_relative_tolerance: float = 5.0e-3,
    solver_population_damping: float = 0.45,
    collision_rate_multiplier: float = 1.0,
    fix_continuum_departure: bool = False,
) -> Spectrum:
    """Synthesize NLTE Lyman through Brackett lines on a fixed structure.

    This is the broad-wavelength counterpart of
    :func:`synthesize_multilevel_balmer_spectrum`.  It uses the same
    depth-dependent departure coefficients for every explicitly represented
    series and includes line overlap in one formal solution.  The atmosphere
    remains an LTE radiative-equilibrium structure.  When requested, the
    dissolved-level pseudo-continuum is treated consistently as a bound-free
    process using the lower-level and H II departure coefficients.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a positive increasing 1D grid")
    if not any((include_lyman, include_balmer, include_paschen, include_brackett)):
        raise ValueError("at least one hydrogen series must be enabled")
    enabled_lower_levels = tuple(
        lower
        for lower, enabled in enumerate(
            (include_lyman, include_balmer, include_paschen, include_brackett),
            start=1,
        )
        if enabled
    )
    if nlte_state is None:
        nlte_state = solve_multilevel_hydrogen_nlte(
            atmosphere,
            collision_data,
            maximum_level=maximum_level,
            merged_rydberg_maximum_level=merged_rydberg_maximum_level,
            merged_transfer_maximum_level=merged_transfer_maximum_level,
            explicit_maximum_lower_level=max(enabled_lower_levels),
            n_angle=solver_n_angle,
            maximum_iterations=solver_maximum_iterations,
            relative_tolerance=solver_relative_tolerance,
            population_damping=solver_population_damping,
            collision_rate_multiplier=collision_rate_multiplier,
            fix_continuum_departure=fix_continuum_departure,
        )
    coefficients = hydrogen_nlte_transfer_coefficients(
        atmosphere,
        wavelength,
        nlte_state,
        include_lyman=include_lyman,
        include_balmer=include_balmer,
        include_paschen=include_paschen,
        include_brackett=include_brackett,
        maximum_level=maximum_level,
        merged_rydberg_maximum_level=merged_rydberg_maximum_level,
        include_balmer_self_broadening=include_balmer_self_broadening,
        include_series_pseudocontinuum=include_series_pseudocontinuum,
        unified_allard_table=unified_allard_table,
    )
    total_opacity = coefficients.total_extinction
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total_opacity
    )
    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    source = (
        coefficients.thermal_emissivity + coefficients.scattering * planck
    ) / total_opacity
    for _ in range(5):
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        source = np.ascontiguousarray(
            (
                coefficients.thermal_emissivity
                + coefficients.scattering * field.mean_intensity
            )
            / total_opacity
        )
    flux = emergent_flux(optical_depth, source, n_angle=n_angle, backend=backend)
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "transfer": (
                "fixed-structure multilevel H I statistical equilibrium with "
                "population-dependent line and bound-free transfer"
            ),
            "nlte_status": "experimental opt-in; LTE atmosphere structure",
            "hydrogen_series_lower_levels": enabled_lower_levels,
            "allard_lyman_profiles": bool(unified_allard_table is not None),
            "dissolved_level_pseudocontinuum": (
                "NLTE DAM/HM bound-free-like redistribution"
                if include_series_pseudocontinuum
                else "not included"
            ),
            "suppressed_pseudocontinuum_inversions": coefficients.metadata[
                "suppressed_pseudocontinuum_inversions"
            ],
            "suppressed_line_inversions": coefficients.metadata[
                "suppressed_line_inversions"
            ],
            "model_atom": nlte_state.metadata.get("model_atom"),
            "merged_rydberg_maximum_level": merged_rydberg_maximum_level,
            "collision_data": collision_data.source,
            "population_iterations": nlte_state.iterations,
            "population_converged": nlte_state.converged,
            "maximum_log_population_change": (
                nlte_state.maximum_relative_population_change
            ),
            "n_angle": int(n_angle),
        },
    )
