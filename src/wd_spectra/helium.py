"""LTE opacity for pure-helium white-dwarf atmospheres.

The visible He I line profiles use the Beauchamp/Tremblay tables released
with Tremblay et al. (2026).  The continuum includes He I and He II
photoionization, He II/III free-free, the John (1968, 1994) He-minus
free-free coefficients, the Stancil (1994) He2+ continuum, Thomson
scattering, and neutral-helium Rayleigh scattering.  Hydrogenically scaled
unified profiles provide a provisional He II bound-bound treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid
from .atmosphere import Atmosphere
from .constants import (
    BOHR_RADIUS,
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HELIUM_FIRST_IONIZATION_ENERGY,
    HELIUM_MASS,
    HELIUM_SECOND_IONIZATION_ENERGY,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import (
    HM_MAX_BOUND_LEVEL,
    HELIUM_I_LOW_TERM_ENERGY,
    HeliumLTEState,
    charged_particle_hydrogen_occupation_probability,
    helium_occupation_probability,
)
from .gaunt import (
    hydrogen_bound_free_gaunt_factor,
    hydrogen_free_free_gaunt_factor,
)
from .helium_stark import HeliumStarkTable, read_helium_stark_table
from .helium_ii_stark import HeliumIIStarkTable, read_helium_ii_stark_table
from .helium_molecular import (
    helium_dimer_ion_continuum_coefficient,
    helium_three_body_cia_linear_absorption_coefficient,
)
from .helium_neutral import deridder_van_rensbergen_helium_hwhm_angstrom
from .opacity import electron_scattering_mass_coefficient


FloatArray = NDArray[np.float64]

# Seaton--Fernley cubic fits to the Opacity Project He I photoionization
# calculations, as distributed in TLUSTY 208's HEPHOT routine.  Rows are the
# five explicitly populated terms in the EOS.  The fit is in x =
# log10(nu / 3.28805e15) - FL0 and returns log10(sigma / 1e-18 cm2).
_HE_I_PHOTOIONIZATION_FL0 = np.asarray(
    [0.2521, -0.4555, -0.5381, -0.5749, -0.6065]
)
_HE_I_PHOTOIONIZATION_XFIT = np.asarray(
    [0.3262, 0.7228, 0.6135, 0.9302, 0.7360]
)
_HE_I_PHOTOIONIZATION_CUBIC = np.asarray(
    [
        [0.8734, -1.545, -1.093, 0.5918],
        [0.7377, -0.9327, -1.466, 0.6891],
        [0.9771, -1.567, -0.4739, -0.1302],
        [1.204, -2.809, -0.3094, 0.1100],
        [1.129, -3.149, -0.1910, -0.5244],
    ]
)
_HE_I_PHOTOIONIZATION_TAIL_A = np.asarray(
    [0.695319, 0.901802, 1.13101, 1.16294, 1.79027]
)
_HE_I_PHOTOIONIZATION_TAIL_B = np.asarray(
    [-1.29000, -1.85905, -2.15771, -2.95758, -4.47251]
)
_HE_I_LOW_TERM_FREQUENCY = np.asarray(
    [5.94503520e15, 1.15267210e15, 9.60145430e14, 8.75933720e14, 8.14536220e14]
)

# Selected n=3 and n=4 Seaton--Fernley/Opacity Project HEPHOT fits from
# TLUSTY 208.  Rows are (n, multiplicity, l); columns are FL0, XFIT, A, B,
# then the four cubic coefficients.  Higher shells in the public TLUSTY He I
# atom use the hydrogenic expression directly.
_HE_I_RYDBERG_OP_KEYS = (
    (3, 1, 0), (3, 1, 1), (3, 1, 2),
    (3, 3, 0), (3, 3, 1), (3, 3, 2),
    (4, 1, 0), (4, 1, 1), (4, 1, 2),
    (4, 3, 0), (4, 3, 1), (4, 3, 2),
)
_HE_I_RYDBERG_OP_FIT = np.asarray(
    [
        [-.9139,.9233,1.36313,-2.13263,1.174,-1.638,-.2831,-.03281],
        [-.9578,1.041,2.23543,-3.87960,1.431,-2.511,-.3710,-.1933],
        [-.9538,.9586,1.83599,-4.58608,1.258,-3.442,-.4731,-.09522],
        [-.8622,1.076,1.25389,-2.04057,.9031,-1.157,-.7151,.1832],
        [-.9352,1.144,1.86467,-3.07110,1.455,-2.254,-.4795,.06872],
        [-.9537,.9585,1.74181,-4.41218,1.267,-3.417,-.5038,-.01797],
        [-1.175,.8438,1.51684,-2.10272,1.324,-1.692,-.2916,.09027],
        [-1.207,1.272,2.63942,-3.71668,1.620,-2.303,-.3045,-.1391],
        [-1.204,1.187,2.50403,-4.40022,1.553,-2.781,-.6841,-.004083],
        [-1.137,1.206,1.39033,-2.02189,1.031,-1.313,-.4517,.09207],
        [-1.190,1.028,2.02110,-2.87157,1.619,-2.109,-.3357,-.02532],
        [-1.204,1.041,2.25756,-4.12940,1.565,-2.781,-.6497,-.005979],
    ], dtype=np.float64,
)
_HE_I_RYDBERG_OP_LOOKUP = dict(zip(_HE_I_RYDBERG_OP_KEYS, _HE_I_RYDBERG_OP_FIT))


@dataclass(frozen=True)
class HeliumLine:
    """One LS-term-averaged He I transition."""

    name: str
    table_key_angstrom: int
    wavelength_vacuum_angstrom: float
    absorption_oscillator_strength: float
    lower_term_index: int
    upper_principal_quantum_number: int


@dataclass(frozen=True)
class HeliumResonanceLine:
    """One ground-state He I ``1s^2 1S - 1s np 1P`` transition."""

    name: str
    wavelength_vacuum_angstrom: float
    absorption_oscillator_strength: float
    upper_principal_quantum_number: int


@dataclass(frozen=True)
class HeliumIILine:
    """One hydrogenic He II shell-averaged transition."""

    name: str
    lower_principal_quantum_number: int
    upper_principal_quantum_number: int
    wavelength_vacuum_angstrom: float
    absorption_oscillator_strength: float


# Vacuum centers and LS-term oscillator strengths derived from the NIST ASD
# fine-structure components (Kramida et al. 2024, ver. 5.12).  Lower-term
# indices follow HeliumLTEState.neutral_level_population_density.
HELIUM_I_LINES = (
    # 2s 3S - np 3P
    HeliumLine("HeI2723", 2723, 2723.998527, 0.00283659, 1, 8),
    HeliumLine("HeI2764", 2764, 2764.619451, 0.00429964, 1, 7),
    HeliumLine("HeI2829", 2829, 2829.913169, 0.00698365, 1, 6),
    HeliumLine("HeI2945", 2945, 2945.964409, 0.0124929, 1, 5),
    HeliumLine("HeI3188", 3188, 3188.665408, 0.0257739, 1, 4),
    HeliumLine("HeI3889", 3889, 3889.744810, 0.0644736, 1, 3),
    # 2s 1S - np 1P
    HeliumLine("HeI3355", 3355, 3355.519207, 0.0073616, 2, 7),
    HeliumLine("HeI3448", 3448, 3448.576937, 0.012137034947, 2, 6),
    HeliumLine("HeI3614", 3614, 3614.672701, 0.022343, 2, 5),
    HeliumLine("HeI3965", 3965, 3965.850589, 0.049168, 2, 4),
    HeliumLine("HeI5016", 5016, 5017.076951, 0.15138, 2, 3),
    # 2p 3P - ns 3S
    HeliumLine("HeI3652", 3652, 3653.041224, 0.000649831111111, 3, 8),
    HeliumLine("HeI3733", 3733, 3733.944963, 0.00103781111111, 3, 7),
    HeliumLine("HeI3868", 3868, 3868.590265, 0.00182977777778, 3, 6),
    HeliumLine("HeI4121", 4121, 4121.997765, 0.00378086666667, 3, 5),
    HeliumLine("HeI4713", 4713, 4714.489791, 0.0105751111111, 3, 4),
    HeliumLine("HeI7065", 7065, 7067.197026, 0.0695194444444, 3, 3),
    # 2p 3P - nd 3D
    HeliumLine("HeI3634", 3634, 3635.285940, 0.008605855, 3, 8),
    HeliumLine("HeI3705", 3705, 3706.069636, 0.0135652166667, 3, 7),
    HeliumLine("HeI3820", 3820, 3820.707582, 0.0234721388889, 3, 6),
    HeliumLine("HeI4026", 4026, 4027.346770, 0.0470127388889, 3, 5),
    HeliumLine("HeI4471", 4471, 4472.756993, 0.122855333333, 3, 4),
    HeliumLine("HeI5876", 5877, 5877.289448, 0.610234777778, 3, 3),
    # 2p 1P - ns 1S
    HeliumLine("HeI3936", 3936, 3937.059473, 0.00057689, 4, 8),
    HeliumLine("HeI4024", 4024, 4025.117031, 0.00091336, 4, 7),
    HeliumLine("HeI4169", 4169, 4170.146714, 0.0015902, 4, 6),
    HeliumLine("HeI4438", 4438, 4438.799269, 0.0032186, 4, 5),
    HeliumLine("HeI5048", 5048, 5049.146038, 0.0086265, 4, 4),
    HeliumLine("HeI7281", 7281, 7283.357121, 0.048509, 4, 3),
    # 2p 1P - nd 1D
    HeliumLine("HeI3872", 3872, 3872.883939, 0.0050168, 4, 9),
    HeliumLine("HeI3926", 3926, 3927.656186, 0.0074666, 4, 8),
    HeliumLine("HeI4009", 4009, 4010.389904, 0.0119, 4, 7),
    HeliumLine("HeI4144", 4144, 4144.927650, 0.020954, 4, 6),
    HeliumLine("HeI4388", 4388, 4389.161905, 0.043269, 4, 5),
    HeliumLine("HeI4922", 4922, 4923.305068, 0.1203, 4, 4),
    HeliumLine("HeI6678", 6678, 6679.996000, 0.71028, 4, 3),
)

_HE_I_P_TO_S_LINE_KEYS = frozenset(
    {
        3652, 3733, 3868, 4121, 4713, 7065,
        3936, 4024, 4169, 4438, 5048, 7281,
    }
)
_HE_I_UNSOLD_ONLY_LINE_KEYS = frozenset({4121, 4713})


# TLUSTY's public 14-level He I model atom, based on NIST energies and
# oscillator strengths.  These EUV lines carry most of the discrete
# ground-state oscillator strength and materially affect DB blanketing even
# though essentially no observable flux escapes in their cores.
HELIUM_I_RESONANCE_LINES = tuple(
    HeliumResonanceLine(f"HeI1s-{n}p", wavelength, oscillator_strength, n)
    for n, wavelength, oscillator_strength in zip(
        range(2, 9),
        (584.33391989, 537.62790717, 522.39465972, 515.68208016,
         512.14177959, 510.03048506, 508.66946440),
        (0.2762, 0.0734, 0.0302, 0.0153, 0.00848, 0.00593, 0.00399),
    )
)

# Dimitrijevic & Sahal-Brechot (1989, Bull. Obs. Astron. Belgrade 141,
# 57--86), Table 1: electron-impact FWHM in Angstrom at Ne=1e16 cm^-3.
# The density dependence is linear over the tabulated impact regime.  Higher
# series members cease to be isolated at progressively smaller densities, so
# their extrapolation is capped at the last density represented in the table.
_HE_I_RESONANCE_IMPACT_TEMPERATURE = np.asarray(
    [10_000.0, 20_000.0, 40_000.0]
)
_HE_I_RESONANCE_ELECTRON_FWHM_1E16 = np.asarray(
    [
        [3.30e-4, 3.63e-4, 3.95e-4],
        [6.68e-3, 6.30e-3, 5.79e-3],
        [2.56e-2, 2.41e-2, 2.20e-2],
        [6.32e-2, 6.10e-2, 5.63e-2],
        [1.22e-1, 1.22e-1, 1.15e-1],
        [2.02e-1, 2.12e-1, 2.05e-1],
        [3.01e-1, 3.32e-1, 3.31e-1],
    ],
    dtype=np.float64,
)
_HE_I_RESONANCE_ELECTRON_MAX_DENSITY = np.asarray(
    [1.0e19, 1.0e17, 1.0e17, 1.0e16, 1.0e16, 1.0e16, 1.0e16]
)
# The 584-A He-II impact component remains tabulated over the white-dwarf
# regime.  Values are FWHM at n(He II)=1e16 cm^-3; the ionic contribution to
# higher series members leaves the isolated-line impact regime too early to
# represent safely with a Lorentzian.
_HE_I_584_HE_II_FWHM_1E16 = np.asarray(
    [9.64e-5, 9.66e-5, 9.69e-5]
)


def helium_i_resonance_stark_hwhm_angstrom(
    temperature: ArrayLike,
    electron_density: ArrayLike,
    singly_ionized_helium_density: ArrayLike,
    upper_principal_quantum_number: int,
) -> FloatArray:
    """Return impact Stark HWHM for a He I ground resonance line.

    Widths are interpolated in log temperature and scaled with perturber
    density, following Table 1 of Dimitrijević & Sahal-Bréchot (1989).  The
    high-density cap prevents extrapolating an isolated-line impact profile
    beyond the range for which the source table reports that approximation.
    """

    temperature, electron_density, ion_density = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(singly_ionized_helium_density, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(ion_density))
        or np.any(ion_density < 0.0)
        or not 2 <= upper_principal_quantum_number <= 8
    ):
        raise ValueError("invalid thermodynamic state or resonance upper level")
    index = upper_principal_quantum_number - 2
    log_temperature = np.clip(
        np.log10(temperature),
        np.log10(_HE_I_RESONANCE_IMPACT_TEMPERATURE[0]),
        np.log10(_HE_I_RESONANCE_IMPACT_TEMPERATURE[-1]),
    )
    electron_fwhm = np.power(
        10.0,
        np.interp(
            log_temperature,
            np.log10(_HE_I_RESONANCE_IMPACT_TEMPERATURE),
            np.log10(_HE_I_RESONANCE_ELECTRON_FWHM_1E16[index]),
        ),
    )
    electron_fwhm *= (
        np.minimum(
            electron_density,
            _HE_I_RESONANCE_ELECTRON_MAX_DENSITY[index],
        )
        / 1.0e16
    )
    ion_fwhm = np.zeros_like(electron_fwhm)
    if upper_principal_quantum_number == 2:
        ion_fwhm = np.power(
            10.0,
            np.interp(
                log_temperature,
                np.log10(_HE_I_RESONANCE_IMPACT_TEMPERATURE),
                np.log10(_HE_I_584_HE_II_FWHM_1E16),
            ),
        )
        ion_fwhm *= np.minimum(ion_density, 1.0e19) / 1.0e16
    return 0.5 * (electron_fwhm + ion_fwhm)


def _helium_ii_wavelength_angstrom(lower: int, upper: int) -> float:
    transition_energy = HELIUM_SECOND_IONIZATION_ENERGY * (
        1.0 / lower**2 - 1.0 / upper**2
    )
    return PLANCK * LIGHT_SPEED / transition_energy * 1.0e8


# Exact non-relativistic shell-averaged hydrogenic oscillator strengths.  A
# hydrogenic ion has the same dimensionless f values as H I.  These cover the
# Balmer-, Paschen-, and Pickering-like He II features relevant to the public
# Montreal DB grid, including 1640, 4686, 5412, and 6560 A.
_HELIUM_II_SERIES_OSCILLATOR_STRENGTHS = {
    1: tuple(
        2.0**8
        * float(upper) ** 5
        * (float(upper) - 1.0) ** (2.0 * float(upper) - 4.0)
        / (3.0 * (float(upper) + 1.0) ** (2.0 * float(upper) + 4.0))
        for upper in range(2, 9)
    ),
    2: (
        0.640747044864, 0.119321144112, 0.0446702946311,
        0.0220928192139, 0.0127046242104, 0.0080355585203,
        0.00542894641045, 0.003851, 0.002835, 0.002151,
        0.001672, 0.001326,
    ),
    3: (
        0.842096357532266, 0.150584082803105, 0.0558402477747247,
        0.0276848879634103, 0.0160358274083514, 0.0102341026067731,
        0.00697970910768096, 0.00499641990757861, 0.00371148300653813,
    ),
    4: (
        1.03773625884024, 0.179252374278725, 0.0654859897311483,
        0.0322951583475062, 0.0186995864209931, 0.0119611130243852,
        0.00818718978958117, 0.00588609864644224, 0.00439250880501866,
        0.00337542785213278, 0.002656,
    ),
}
HELIUM_II_LINES = tuple(
    HeliumIILine(
        f"HeII{lower}-{upper}",
        lower,
        upper,
        _helium_ii_wavelength_angstrom(lower, upper),
        oscillator_strength,
    )
    for lower, strengths in _HELIUM_II_SERIES_OSCILLATOR_STRENGTHS.items()
    for upper, oscillator_strength in enumerate(strengths, start=lower + 1)
)


def helium_i_resonance_hwhm_angstrom(
    observed_wavelength_angstrom: ArrayLike,
    resonance_wavelength_angstrom: ArrayLike,
    resonance_oscillator_strength: ArrayLike,
    ground_state_number_density: ArrayLike,
) -> FloatArray:
    """Return the Ali--Griem He I resonance Lorentz HWHM in Angstrom.

    This is the standard NIST estimate for a singlet level connected to the
    ``1s2 1S`` ground state.  The published coefficient gives a *full* width
    at half maximum; the factor of one half below converts it to the HWHM
    convention used by the profile convolutions in this module.  The
    statistical-weight factor is ``sqrt(1/3)`` for all of the allowed
    ``1s2 1S - 1s np 1P`` resonance transitions considered here.
    """

    observed = np.asarray(observed_wavelength_angstrom, dtype=np.float64)
    resonance = np.asarray(resonance_wavelength_angstrom, dtype=np.float64)
    oscillator_strength = np.asarray(
        resonance_oscillator_strength, dtype=np.float64
    )
    ground_density = np.asarray(ground_state_number_density, dtype=np.float64)
    if (
        np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(resonance))
        or np.any(~np.isfinite(oscillator_strength))
        or np.any(~np.isfinite(ground_density))
        or np.any(observed <= 0.0)
        or np.any(resonance <= 0.0)
        or np.any(oscillator_strength < 0.0)
        or np.any(ground_density < 0.0)
    ):
        raise ValueError("resonance-width inputs must be finite and non-negative")
    return (
        0.5
        * 8.6e-30
        * np.sqrt(1.0 / 3.0)
        * observed**2
        * resonance
        * oscillator_strength
        * ground_density
    )


def _singlet_resonance_transition(line: HeliumLine) -> HeliumResonanceLine | None:
    """Return the ground-state transition coupled to a singlet line level."""

    if line.lower_term_index == 4:
        # The optical line starts from 1s2p 1P, connected to the ground state
        # by the 584-A resonance line.
        return HELIUM_I_RESONANCE_LINES[0]
    if line.lower_term_index == 2:
        # The optical line ends in 1s np 1P.  Its matching ground-state
        # transition is the corresponding member of the resonance series.
        index = line.upper_principal_quantum_number - 2
        if 0 <= index < len(HELIUM_I_RESONANCE_LINES):
            return HELIUM_I_RESONANCE_LINES[index]
    return None


def _require_helium_state(atmosphere: Atmosphere) -> HeliumLTEState:
    state = atmosphere.helium_lte_state
    if state is None:
        raise ValueError("a pure-helium Atmosphere with helium_lte_state is required")
    return state


def neutral_helium_photoionization_cross_section(
    frequency_hz: ArrayLike, term_index: int
) -> FloatArray:
    """Return the Opacity Project He I bound-free fit in cm^2.

    ``term_index`` follows :class:`HeliumLTEState`: ground, 2s triplet,
    2s singlet, 2p triplet, and 2p singlet.  The resonance-free cubic fits
    are due to Seaton and Fernley and are the standard TLUSTY/ATMO data.
    Above their fitted interval the prescribed log-log tail is used.
    """

    if term_index < 0 or term_index >= _HE_I_LOW_TERM_FREQUENCY.size:
        raise ValueError("term_index must be between 0 and 4")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must contain finite positive values")
    x = (
        np.log10(frequency / 3.28805e15)
        - _HE_I_PHOTOIONIZATION_FL0[term_index]
    )
    coefficients = _HE_I_PHOTOIONIZATION_CUBIC[term_index]
    polynomial = (
        coefficients[0]
        + x * (coefficients[1] + x * (coefficients[2] + x * coefficients[3]))
    )
    log_cross_section_mb = np.where(
        x < _HE_I_PHOTOIONIZATION_XFIT[term_index],
        polynomial,
        _HE_I_PHOTOIONIZATION_TAIL_A[term_index]
        + _HE_I_PHOTOIONIZATION_TAIL_B[term_index] * x,
    )
    return np.where(
        frequency >= _HE_I_LOW_TERM_FREQUENCY[term_index],
        np.power(10.0, log_cross_section_mb) * 1.0e-18,
        0.0,
    )


# John (1994), MNRAS 269, 871, Table 2.  Coefficients are in units of
# 1e-26 cm^4 dyne^-1 per neutral He atom per electron pressure.
_JOHN_THETA = np.asarray([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])
_JOHN_WAVELENGTH_UM = np.asarray(
    [0.5063, 0.5695, 0.6509, 0.7594, 0.9113, 1.1391, 1.5188, 1.8225,
     2.2782, 3.0376, 3.6451, 4.5564, 6.0751, 9.1127, 11.3909, 15.1878]
)
_JOHN_COEFFICIENT = np.asarray(
    [
        [.033,.036,.043,.049,.055,.061,.066,.072,.078,.100,.121],
        [.041,.045,.053,.061,.067,.074,.081,.087,.094,.120,.145],
        [.053,.059,.069,.077,.086,.094,.102,.109,.117,.148,.178],
        [.072,.079,.092,.103,.114,.124,.133,.143,.152,.190,.227],
        [.102,.113,.131,.147,.160,.173,.186,.198,.210,.258,.305],
        [.159,.176,.204,.227,.247,.266,.283,.300,.316,.380,.444],
        [.282,.311,.360,.400,.435,.466,.495,.522,.547,.643,.737],
        [.405,.447,.518,.576,.625,.670,.710,.747,.782,.910,1.030],
        [.632,.698,.808,.899,.977,1.045,1.108,1.165,1.218,1.405,1.574],
        [1.121,1.239,1.435,1.597,1.737,1.860,1.971,2.073,2.167,2.490,2.765],
        [1.614,1.783,2.065,2.299,2.502,2.681,2.842,2.990,3.126,3.592,3.979],
        [2.520,2.784,3.226,3.593,3.910,4.193,4.448,4.681,4.897,5.632,6.234],
        [4.479,4.947,5.733,6.387,6.955,7.460,7.918,8.338,8.728,10.059,11.147],
        [10.074,11.128,12.897,14.372,15.653,16.798,17.838,18.795,19.685,22.747,25.268],
        [15.739,17.386,20.151,22.456,24.461,26.252,27.882,29.384,30.782,35.606,39.598],
        [27.979,30.907,35.822,39.921,43.488,46.678,49.583,52.262,54.757,63.395,70.580],
    ], dtype=np.float64
)
_JOHN_LONG_WAVELENGTH_FACTOR = np.asarray(
    [.121,.134,.155,.173,.189,.202,.215,.227,.238,.275,.307]
)


def _john_1968_helium_minus_free_free_coefficient(
    wavelength_micron: FloatArray, temperature: FloatArray
) -> FloatArray:
    """Return the Carbon et al. (1969) fit to John (1968).

    Carbon et al. fitted the John (1968) calculation as a polynomial in
    temperature and inverse frequency.  The original fit returns the
    absorption coefficient per ``n_e n(He I)`` in cm^5.  Division by the
    electron pressure factor ``kT`` converts it to the same cm^4 dyne^-1
    convention used in the John (1994) table below.  The fit already includes
    stimulated emission, as is evident from its use without a separate
    stimulated-emission factor in the standard ATLAS ``HEMIOP`` routine.
    """

    frequency = LIGHT_SPEED / (wavelength_micron * 1.0e-4)
    coefficient_a = (
        3.397e-46 + (-5.216e-31 + 7.039e-15 / frequency) / frequency
    )
    coefficient_b = (
        -4.116e-42 + (1.067e-26 + 8.135e-11 / frequency) / frequency
    )
    coefficient_c = (
        5.081e-37 + (-8.724e-23 - 5.659e-8 / frequency) / frequency
    )
    coefficient_cm5 = (
        coefficient_a * temperature
        + coefficient_b
        + coefficient_c / temperature
    )
    return np.maximum(coefficient_cm5, 0.0) / (BOLTZMANN * temperature)


def helium_minus_free_free_coefficient(
    wavelength_micron: ArrayLike,
    temperature: ArrayLike,
    *,
    prescription: Literal["automatic", "john1968", "john1994"] = "automatic",
) -> FloatArray:
    """Return a He-minus free-free coefficient in cm^4 dyne^-1.

    ``prescription='automatic'`` uses the more accurate John (1994) table
    only inside its published wavelength and temperature domain
    (lambda >= 0.5063 micron and 1400 <= T <= 10080 K).  Elsewhere it uses
    the Carbon et al. (1969) polynomial fit to John (1968).  This follows the
    two-source treatment documented for the Montreal DB calculations without
    extrapolating the cool John table through hotter DB photospheres or into
    the ultraviolet.  The explicit prescriptions are retained for controlled
    comparisons.

    Inputs broadcast normally.  The returned coefficient has units
    cm^4 dyne^-1 and multiplies ``n(He I) * electron_pressure`` to give an
    inverse-length absorption coefficient.
    """

    wavelength, temperature = np.broadcast_arrays(
        np.asarray(wavelength_micron, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_micron must contain finite positive values")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must contain finite positive values")
    if prescription not in ("automatic", "john1968", "john1994"):
        raise ValueError(
            "prescription must be 'automatic', 'john1968', or 'john1994'"
        )
    john_1968 = _john_1968_helium_minus_free_free_coefficient(
        wavelength, temperature
    )
    if prescription == "john1968":
        return john_1968
    theta = np.clip(5040.0 / temperature, _JOHN_THETA[0], _JOHN_THETA[-1])
    theta_upper = np.searchsorted(_JOHN_THETA, theta, side="right")
    theta_upper = np.clip(theta_upper, 1, _JOHN_THETA.size - 1)
    theta_lower = theta_upper - 1
    theta_fraction = (
        (theta - _JOHN_THETA[theta_lower])
        / (_JOHN_THETA[theta_upper] - _JOHN_THETA[theta_lower])
    )
    wavelength_upper = np.searchsorted(
        _JOHN_WAVELENGTH_UM, wavelength, side="right"
    )
    wavelength_upper = np.clip(
        wavelength_upper, 1, _JOHN_WAVELENGTH_UM.size - 1
    )
    wavelength_lower = wavelength_upper - 1
    wavelength_fraction = (
        (wavelength - _JOHN_WAVELENGTH_UM[wavelength_lower])
        / (
            _JOHN_WAVELENGTH_UM[wavelength_upper]
            - _JOHN_WAVELENGTH_UM[wavelength_lower]
        )
    )
    lower_wavelength_value = (
        (1.0 - theta_fraction)
        * _JOHN_COEFFICIENT[wavelength_lower, theta_lower]
        + theta_fraction
        * _JOHN_COEFFICIENT[wavelength_lower, theta_upper]
    )
    upper_wavelength_value = (
        (1.0 - theta_fraction)
        * _JOHN_COEFFICIENT[wavelength_upper, theta_lower]
        + theta_fraction
        * _JOHN_COEFFICIENT[wavelength_upper, theta_upper]
    )
    result = (
        (1.0 - wavelength_fraction) * lower_wavelength_value
        + wavelength_fraction * upper_wavelength_value
    )
    # John's calculation is tabulated only from 0.5063 micron.  The opacity
    # is negligible in the UV; tapering avoids a false edge.
    short = wavelength < _JOHN_WAVELENGTH_UM[0]
    edge = (
        (1.0 - theta_fraction) * _JOHN_COEFFICIENT[0, theta_lower]
        + theta_fraction * _JOHN_COEFFICIENT[0, theta_upper]
    )
    result = np.where(
        short,
        edge * (wavelength / _JOHN_WAVELENGTH_UM[0]) ** 2,
        result,
    )
    long = wavelength > _JOHN_WAVELENGTH_UM[-1]
    long_factor = (
        (1.0 - theta_fraction) * _JOHN_LONG_WAVELENGTH_FACTOR[theta_lower]
        + theta_fraction * _JOHN_LONG_WAVELENGTH_FACTOR[theta_upper]
    )
    result = np.where(long, long_factor * wavelength**2, result)
    john_1994 = result * 1.0e-26
    if prescription == "john1994":
        return john_1994
    john_1994_domain = (
        (wavelength >= _JOHN_WAVELENGTH_UM[0])
        & (temperature >= 5040.0 / _JOHN_THETA[-1])
        & (temperature <= 5040.0 / _JOHN_THETA[0])
    )
    return np.where(john_1994_domain, john_1994, john_1968)


def _helium_ii_level_distribution(
    atmosphere: Atmosphere,
    maximum_principal_quantum_number: int = HM_MAX_BOUND_LEVEL,
) -> tuple[FloatArray, FloatArray]:
    """Return He II shell populations and HM occupation probabilities.

    This exactly mirrors the hydrogenic He II completion used in the helium
    EOS: statistical weight ``2 n^2``, binding charge ``Z=2``, Q-MHD charged
    perturbers, and excluded-volume dissolution by neutral He I.
    """

    state = _require_helium_state(atmosphere)
    if maximum_principal_quantum_number < 1:
        raise ValueError("maximum_principal_quantum_number must be positive")
    principal = np.arange(
        1, maximum_principal_quantum_number + 1, dtype=np.float64
    )
    temperature = atmosphere.temperature[:, np.newaxis]
    charged = charged_particle_hydrogen_occupation_probability(
        state.electron_density[:, np.newaxis],
        principal[np.newaxis, :],
        temperature if state.microfield_model == "qmhd" else None,
        ionic_charge=2.0,
    )
    orbital_radius = (
        state.neutral_radius_scale
        * BOHR_RADIUS
        * principal[np.newaxis, :] ** 2
        / 2.0
    )
    ground_neutral_radius = state.neutral_radius_scale * BOHR_RADIUS
    hydrogen_neutral_radius = (
        atmosphere.hydrogen_lte_state.neutral_radius_scale * BOHR_RADIUS
        if atmosphere.hydrogen_lte_state is not None else BOHR_RADIUS
    )
    neutral = np.exp(
        -state.neutral_he_density[:, np.newaxis]
        * 4.0
        / 3.0
        * PI
        * (orbital_radius + ground_neutral_radius) ** 3
        - atmosphere.neutral_h_density[:, np.newaxis]
        * 4.0
        / 3.0
        * PI
        * (orbital_radius + hydrogen_neutral_radius) ** 3
    )
    occupation = charged * neutral
    excitation_energy = HELIUM_SECOND_IONIZATION_ENERGY * (
        1.0 - 1.0 / principal**2
    )
    population = (
        state.singly_ionized_he_density[:, np.newaxis]
        / state.singly_ionized_partition_function[:, np.newaxis]
        * 2.0
        * principal**2
        * occupation
        * np.exp(-excitation_energy[np.newaxis, :] / (BOLTZMANN * temperature))
    )
    return population, occupation


def helium_ii_bound_free_linear_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    maximum_principal_quantum_number: int = HM_MAX_BOUND_LEVEL,
) -> FloatArray:
    """Return occupation-probability He II bound-free absorption in cm^-1.

    The ground state uses the exact hydrogenic Stobbe cross section scaled by
    ``Z^-2``.  Excited shells use the Mihalas/Karzas--Latter Gaunt fits with
    the hydrogenic ``Z^4`` Kramers scaling.
    """

    # Imported locally to avoid making the neutral-helium opacity module the
    # owner of the exact hydrogenic 1s expression.
    from .opacity import hydrogen_ground_state_photoionization_cross_section

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    population, _ = _helium_ii_level_distribution(
        atmosphere, maximum_principal_quantum_number
    )
    frequency = LIGHT_SPEED / (wavelength[:, np.newaxis] * 1.0e-8)
    temperature = atmosphere.temperature[np.newaxis, :]
    stimulated = -np.expm1(-PLANCK * frequency / (BOLTZMANN * temperature))
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    threshold_ground = HELIUM_SECOND_IONIZATION_ENERGY / PLANCK
    for level in range(1, maximum_principal_quantum_number + 1):
        threshold = threshold_ground / level**2
        if level == 1:
            cross_section = (
                hydrogen_ground_state_photoionization_cross_section(
                    frequency / 4.0
                )
                / 4.0
            )
        else:
            cross_section = (
                2.815e29
                * 2.0**4
                * frequency**-3
                / level**5
                * hydrogen_bound_free_gaunt_factor(level, frequency / 4.0)
            )
        opacity += np.where(
            frequency >= threshold,
            population[np.newaxis, :, level - 1]
            * cross_section
            * stimulated,
            0.0,
        )
    return opacity


def helium_continuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    include_electron_scattering: bool = True,
    include_rayleigh_scattering: bool = True,
    include_dissolved_level_pseudocontinuum: bool = True,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    helium_minus_correction: ArrayLike | None = None,
) -> FloatArray:
    """Return the implemented LTE pure-He continuum in cm^2 g^-1.

    An explicit dense-fluid ``helium_minus_correction`` multiplies only
    John He-minus free-free absorption. Its shape must be (wavelength,
    depth); omitted corrections retain the existing arithmetic exactly.
    """

    state = _require_helium_state(atmosphere)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    temperature = atmosphere.temperature[np.newaxis, :]
    density = atmosphere.mass_density[np.newaxis, :]
    electrons = atmosphere.electron_density[np.newaxis, :]
    neutral = state.neutral_he_density[np.newaxis, :]
    he_ii = state.singly_ionized_he_density[np.newaxis, :]
    he_iii = state.doubly_ionized_he_density[np.newaxis, :]
    stimulated = -np.expm1(-PLANCK * frequency / (BOLTZMANN * temperature))

    gaunt_z1 = hydrogen_free_free_gaunt_factor(
        wavelength[:, np.newaxis], temperature, ionic_charge=1.0
    )
    gaunt_z2 = hydrogen_free_free_gaunt_factor(
        wavelength[:, np.newaxis], temperature, ionic_charge=2.0
    )
    free_free = (
        3.692e8 * temperature**-0.5 * electrons
        * (he_ii * gaunt_z1 + 4.0 * he_iii * gaunt_z2)
        * frequency**-3
        * stimulated
    )

    # Seaton--Fernley fits to the Opacity Project cross sections for the
    # ground and four low n=2 terms.
    bound_free = np.zeros_like(free_free)
    for term_index in range(5):
        cross_section = neutral_helium_photoionization_cross_section(
            frequency, term_index
        )
        bound_free += (
            state.neutral_level_population_density[np.newaxis, :, term_index]
            * cross_section * stimulated
        )
    if include_rydberg_bound_free:
        bound_free += helium_i_rydberg_bound_free_linear_absorption_coefficient(
            atmosphere, wavelength
        )
    bound_free += helium_ii_bound_free_linear_absorption_coefficient(
        atmosphere, wavelength
    )

    electron_pressure = electrons * BOLTZMANN * temperature
    helium_minus = (
        helium_minus_free_free_coefficient(wavelength[:, np.newaxis] * 1.0e-4, temperature)
        * neutral * electron_pressure
    )
    if helium_minus_correction is not None:
        correction = np.asarray(helium_minus_correction, dtype=float)
        if (correction.shape != helium_minus.shape or np.any(~np.isfinite(correction))
                or np.any(correction <= 0)):
            raise ValueError("helium_minus_correction must be positive finite wavelength-by-depth factors")
        helium_minus *= correction
    helium_dimer_ion = np.zeros_like(free_free)
    if include_helium_dimer_ion:
        helium_dimer_ion = (
            helium_dimer_ion_continuum_coefficient(
                wavelength[:, np.newaxis], temperature
            )
            * neutral
            * he_ii
            * stimulated
        )
    helium_three_body_cia = np.zeros_like(free_free)
    if include_helium_three_body_cia:
        helium_three_body_cia = (
            helium_three_body_cia_linear_absorption_coefficient(
                wavelength[:, np.newaxis], temperature, neutral
            )
        )
    result = (
        free_free + bound_free + helium_minus + helium_dimer_ion
        + helium_three_body_cia
    ) / density
    if include_dissolved_level_pseudocontinuum:
        result += helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
            atmosphere, wavelength
        )
        result += helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
            atmosphere, wavelength
        )
    if include_electron_scattering:
        result += electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
    if include_rayleigh_scattering:
        result += helium_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
    return np.maximum(result, np.finfo(np.float64).tiny)


def _helium_i_op_shell_cross_section(
    frequency_hz: FloatArray,
    principal_quantum_number: int,
    *,
    multiplicity: int | None = None,
) -> FloatArray:
    """Return TLUSTY's OP-averaged n=3 or n=4 He I cross section.

    With ``multiplicity=None`` this is the singlet+triplet shell average used
    by the LTE Rydberg opacity.  Values one and three reproduce the separate
    superlevel averages in TLUSTY's 14-term He I atom.
    """

    if multiplicity not in (None, 1, 3):
        raise ValueError("multiplicity must be one, three, or None")

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    contributions = np.zeros_like(frequency)
    multiplicities = (1, 3) if multiplicity is None else (multiplicity,)
    for spin_multiplicity in multiplicities:
        for angular_momentum in range(principal_quantum_number):
            angular_weight = 2 * angular_momentum + 1
            if angular_momentum <= 2:
                fit = _HE_I_RYDBERG_OP_LOOKUP[
                    (
                        principal_quantum_number,
                        spin_multiplicity,
                        angular_momentum,
                    )
                ]
                x = np.log10(frequency / 3.28805e15) - fit[0]
                polynomial = (
                    fit[4]
                    + x * (fit[5] + x * (fit[6] + x * fit[7]))
                )
                log_cross_section_mb = np.where(
                    x < fit[1], polynomial, fit[2] + fit[3] * x
                )
                cross_section = np.where(
                    x >= -0.001,
                    np.power(10.0, log_cross_section_mb) * 1.0e-18,
                    0.0,
                )
            else:
                # This reproduces HEPHOT's high-l branch, including its
                # internal relative-population factor, before SBFHE1 applies
                # the outer statistical average.
                cross_section = (
                    2.815e29
                    / (frequency**3 * principal_quantum_number**5)
                    * spin_multiplicity
                    * angular_weight
                    / (2.0 * principal_quantum_number**2)
                )
                cross_section = np.where(
                    frequency
                    >= HYDROGEN_IONIZATION_ENERGY
                    / (PLANCK * principal_quantum_number**2),
                    cross_section,
                    0.0,
                )
            outer_weight = (
                spin_multiplicity * angular_weight
                if multiplicity is None
                else angular_weight
            )
            contributions += outer_weight * cross_section
    denominator = (
        4.0 * principal_quantum_number**2
        if multiplicity is None
        else float(principal_quantum_number**2)
    )
    return contributions / denominator


def helium_i_rydberg_bound_free_linear_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    minimum_principal_quantum_number: int = 3,
    maximum_principal_quantum_number: int = HM_MAX_BOUND_LEVEL,
) -> FloatArray:
    """Return He I Rydberg-shell bound-free absorption in cm^-1.

    The shell populations are exactly the hydrogenic completion used by the
    helium HM EOS: statistical weight ``4 n^2``, the same occupation
    probability, and binding energy ``13.598 eV/n^2``.  The cross section is
    TLUSTY's OP-averaged HEPHOT fits are used for n=3 and n=4, while the
    hydrogenic ``2.815e29 / (nu^3 n^5)`` expression is used from n=5 upward,
    exactly matching the continuum switches in its public 14-level He atom.
    """

    state = _require_helium_state(atmosphere)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or minimum_principal_quantum_number < 3
        or maximum_principal_quantum_number < minimum_principal_quantum_number
    ):
        raise ValueError("invalid wavelength or Rydberg-shell range")
    principal = np.arange(
        minimum_principal_quantum_number,
        maximum_principal_quantum_number + 1,
        dtype=np.float64,
    )
    temperature = atmosphere.temperature[:, np.newaxis]
    occupation = helium_occupation_probability(
        state.neutral_he_density[:, np.newaxis],
        state.electron_density[:, np.newaxis],
        temperature,
        principal[np.newaxis, :],
        neutral_h_density=atmosphere.neutral_h_density[:, np.newaxis],
        neutral_radius_scale=state.neutral_radius_scale,
        hydrogen_neutral_radius_scale=(
            atmosphere.hydrogen_lte_state.neutral_radius_scale
            if atmosphere.hydrogen_lte_state is not None else 1.0
        ),
        correlated_microfields=(state.microfield_model == "qmhd"),
    )
    excitation_energy = (
        HELIUM_FIRST_IONIZATION_ENERGY
        - HYDROGEN_IONIZATION_ENERGY / principal**2
    )
    shell_population = (
        state.neutral_he_density[:, np.newaxis]
        / state.neutral_partition_function[:, np.newaxis]
        * 4.0
        * principal**2
        * occupation
        * np.exp(-excitation_energy[np.newaxis, :] / (BOLTZMANN * temperature))
    )
    frequency = LIGHT_SPEED / (wavelength[:, np.newaxis, np.newaxis] * 1.0e-8)
    threshold = (
        HYDROGEN_IONIZATION_ENERGY
        / (PLANCK * principal[np.newaxis, np.newaxis, :] ** 2)
    )
    cross_section = np.where(
        frequency >= threshold,
        2.815e29
        / (frequency**3 * principal[np.newaxis, np.newaxis, :] ** 5),
        0.0,
    )
    for shell in (3, 4):
        selected_shell = np.flatnonzero(principal == shell)
        if selected_shell.size:
            cross_section[:, :, selected_shell[0]] = (
                _helium_i_op_shell_cross_section(frequency[:, :, 0], shell)
            )
    stimulated = -np.expm1(
        -PLANCK
        * frequency
        / (BOLTZMANN * atmosphere.temperature[np.newaxis, :, np.newaxis])
    )
    return np.sum(
        shell_population[np.newaxis, :, :] * cross_section * stimulated,
        axis=2,
    )


def helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
    atmosphere: Atmosphere, wavelength_angstrom: ArrayLike
) -> FloatArray:
    """Return HM/DAM redistributed He I bound-free opacity.

    A photon redward of a low term's ordinary photoionization edge is mapped
    to a fictitious Rydberg state.  The dissolved fraction
    ``1 - w_upper / w_lower`` moves the corresponding oscillator strength
    into a pseudo-continuum.  The continuation is restricted to the first
    allowed resonance of each series, avoiding the unphysical indefinite
    infrared extension of the raw DAM prescription.
    """

    state = _require_helium_state(atmosphere)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    opacity = np.zeros((wavelength.size, atmosphere.n_depth))
    first_resonance = np.asarray(
        [584.334, 3889.744810, 5017.076951, 7067.197026, 7283.357121]
    )
    for term_index in range(5):
        threshold_frequency = _HE_I_LOW_TERM_FREQUENCY[term_index]
        series_limit = LIGHT_SPEED / threshold_frequency * 1.0e8
        valid = (
            (wavelength > series_limit)
            & (wavelength < first_resonance[term_index])
        )
        if not np.any(valid):
            continue
        selected = wavelength[valid]
        photon_energy = PLANCK * LIGHT_SPEED / (selected * 1.0e-8)
        upper_binding_energy = np.maximum(
            PLANCK * threshold_frequency - photon_energy,
            np.finfo(np.float64).tiny,
        )
        effective_upper_level = np.sqrt(
            HYDROGEN_IONIZATION_ENERGY / upper_binding_energy
        )
        # Exactly at a series limit the fictitious DAM upper state is
        # n=infinity and its occupation probability is zero.  A finite cap
        # represents that limit without overflowing the excluded-volume
        # radius before exp(-volume) underflows to zero.
        effective_upper_level = np.minimum(effective_upper_level, 1.0e6)
        upper_probability = helium_occupation_probability(
            state.neutral_he_density[np.newaxis, :],
            state.electron_density[np.newaxis, :],
            atmosphere.temperature[np.newaxis, :],
            effective_upper_level[:, np.newaxis],
            neutral_h_density=atmosphere.neutral_h_density[np.newaxis, :],
            neutral_radius_scale=state.neutral_radius_scale,
            hydrogen_neutral_radius_scale=(
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None else 1.0
            ),
            correlated_microfields=(state.microfield_model == "qmhd"),
        )
        lower_probability = state.neutral_level_occupation_probability[
            :, term_index
        ]
        dissolved_fraction = np.clip(
            1.0
            - upper_probability
            / np.maximum(lower_probability[np.newaxis, :], np.finfo(np.float64).tiny),
            0.0,
            1.0,
        )
        threshold_cross_section = float(
            neutral_helium_photoionization_cross_section(
                threshold_frequency, term_index
            )
        )
        extrapolated_cross_section = (
            threshold_cross_section * (selected / series_limit) ** 3
        )
        frequency = LIGHT_SPEED / (selected[:, np.newaxis] * 1.0e-8)
        stimulated = -np.expm1(
            -PLANCK * frequency
            / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
        )
        opacity[valid] += (
            state.neutral_level_population_density[np.newaxis, :, term_index]
            / atmosphere.mass_density[np.newaxis, :]
            * extrapolated_cross_section[:, np.newaxis]
            * dissolved_fraction
            * stimulated
        )
    return opacity


def helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    maximum_lower_level: int = 4,
) -> FloatArray:
    """Return DAM pseudo-continuum opacity for dissolved He II levels.

    The continuation of each hydrogenic bound-free edge is truncated at the
    first discrete member of that series, as in the neutral-helium and
    hydrogen implementations.  This is particularly relevant to the He II
    ``n=2``--``4`` continua in hot DB atmospheres.
    """

    from .opacity import hydrogen_ground_state_photoionization_cross_section

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or maximum_lower_level < 1
    ):
        raise ValueError("invalid wavelength or maximum_lower_level")
    population, occupation = _helium_ii_level_distribution(
        atmosphere, max(HM_MAX_BOUND_LEVEL, maximum_lower_level + 1)
    )
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    ground_threshold = HELIUM_SECOND_IONIZATION_ENERGY / PLANCK
    for lower_level in range(1, maximum_lower_level + 1):
        threshold_frequency = ground_threshold / lower_level**2
        series_limit = LIGHT_SPEED / threshold_frequency * 1.0e8
        first_line_frequency = ground_threshold * (
            1.0 / lower_level**2 - 1.0 / (lower_level + 1.0) ** 2
        )
        first_line_wavelength = LIGHT_SPEED / first_line_frequency * 1.0e8
        valid = (
            (wavelength > series_limit)
            & (wavelength < first_line_wavelength)
        )
        if not np.any(valid):
            continue
        selected = wavelength[valid]
        photon_energy = PLANCK * LIGHT_SPEED / (selected * 1.0e-8)
        upper_binding_energy = np.maximum(
            PLANCK * threshold_frequency - photon_energy,
            np.finfo(np.float64).tiny,
        )
        effective_upper_level = np.sqrt(
            4.0 * HYDROGEN_IONIZATION_ENERGY / upper_binding_energy
        )
        effective_upper_level = np.minimum(effective_upper_level, 1.0e6)
        state = _require_helium_state(atmosphere)
        charged = charged_particle_hydrogen_occupation_probability(
            state.electron_density[np.newaxis, :],
            np.maximum(effective_upper_level[:, np.newaxis], 1.0),
            atmosphere.temperature[np.newaxis, :]
            if state.microfield_model == "qmhd"
            else None,
            ionic_charge=2.0,
        )
        upper_radius = (
            state.neutral_radius_scale
            * BOHR_RADIUS
            * effective_upper_level[:, np.newaxis] ** 2
            / 2.0
        )
        neutral = np.exp(
            -state.neutral_he_density[np.newaxis, :]
            * 4.0
            / 3.0
            * PI
            * (upper_radius + state.neutral_radius_scale * BOHR_RADIUS) ** 3
        )
        upper_probability = charged * neutral
        lower_probability = occupation[:, lower_level - 1]
        dissolved_fraction = np.clip(
            1.0
            - upper_probability
            / np.maximum(
                lower_probability[np.newaxis, :],
                np.finfo(np.float64).tiny,
            ),
            0.0,
            1.0,
        )
        if lower_level == 1:
            threshold_cross_section = float(
                hydrogen_ground_state_photoionization_cross_section(
                    threshold_frequency / 4.0
                )
                / 4.0
            )
        else:
            threshold_cross_section = float(
                2.815e29
                * 2.0**4
                * threshold_frequency**-3
                / lower_level**5
                * hydrogen_bound_free_gaunt_factor(
                    lower_level, threshold_frequency / 4.0
                )
            )
        extrapolated_cross_section = (
            threshold_cross_section * (selected / series_limit) ** 3
        )
        frequency = LIGHT_SPEED / (selected[:, np.newaxis] * 1.0e-8)
        stimulated = -np.expm1(
            -PLANCK
            * frequency
            / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
        )
        opacity[valid] += (
            population[np.newaxis, :, lower_level - 1]
            / atmosphere.mass_density[np.newaxis, :]
            * extrapolated_cross_section[:, np.newaxis]
            * dissolved_fraction
            * stimulated
        )
    return opacity


def helium_rayleigh_scattering_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Long-wavelength static-polarizability He I Rayleigh cross section."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must contain finite positive values")
    wavelength_cm = wavelength * 1.0e-8
    polarizability_volume = 1.383192174 * BOHR_RADIUS**3
    return 128.0 * PI**5 / 3.0 * polarizability_volume**2 / wavelength_cm**4


def helium_ii_rayleigh_scattering_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return ground-state hydrogenic He II Rayleigh scattering in cm^2.

    Dynamic hydrogenic polarizability scales as ``Z^-4`` while transition
    frequencies scale as ``Z^2``.  Consequently the full cross section at
    wavelength ``lambda`` is the H I cross section at ``Z^2 lambda``.  This
    retains the accurate preresonance fit and its smooth handoff to the
    explicit He II 304-A line.
    """

    from .opacity import hydrogen_rayleigh_scattering_cross_section

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must contain finite positive values")
    return hydrogen_rayleigh_scattering_cross_section(4.0 * wavelength)


def helium_rayleigh_scattering_mass_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    include_fluid_correlation: bool = True,
) -> FloatArray:
    """Return He I/II Rayleigh scattering with the dense-fluid correction.

    For neutral helium the low-frequency Monte Carlo fit of Rohrmann (2018)
    gives ``S(0)=(1+rho)^(-46.67685/T^0.3128)`` with mass density in
    g cm^-3.  Blouin, Dufour & Allard (2018) apply this factor to cool DZ
    atmospheres.  It approaches unity continuously in the dilute limit.
    The much smaller ionic Rayleigh term is left unmodified.
    """

    state = _require_helium_state(atmosphere)
    cross_section = helium_rayleigh_scattering_cross_section(wavelength_angstrom)
    if cross_section.ndim != 1:
        raise ValueError("wavelength_angstrom must be a 1D array")
    neutral_ground = state.neutral_level_population_density[:, 0]
    ion_population, _ = _helium_ii_level_distribution(atmosphere, 1)
    ion_cross_section = helium_ii_rayleigh_scattering_cross_section(
        wavelength_angstrom
    )
    structure_factor = np.ones(atmosphere.n_depth, dtype=np.float64)
    if include_fluid_correlation:
        exponent = 46.67685 / atmosphere.temperature**0.3128
        structure_factor = (1.0 + atmosphere.mass_density) ** (-exponent)
    return (
        cross_section[:, np.newaxis]
        * neutral_ground[np.newaxis, :]
        * structure_factor[np.newaxis, :]
        + ion_cross_section[:, np.newaxis] * ion_population[np.newaxis, :, 0]
    ) / atmosphere.mass_density[np.newaxis, :]


def _pseudo_voigt_profile_per_angstrom(
    wavelength_angstrom: FloatArray,
    center_angstrom: float,
    gaussian_sigma_angstrom: float,
    lorentz_hwhm_angstrom: float,
) -> FloatArray:
    """Return an area-normalized Thompson pseudo-Voigt profile."""

    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * gaussian_sigma_angstrom
    lorentz_fwhm = 2.0 * lorentz_hwhm_angstrom
    voigt_fwhm = (
        gaussian_fwhm**5
        + 2.69269 * gaussian_fwhm**4 * lorentz_fwhm
        + 2.42843 * gaussian_fwhm**3 * lorentz_fwhm**2
        + 4.47163 * gaussian_fwhm**2 * lorentz_fwhm**3
        + 0.07842 * gaussian_fwhm * lorentz_fwhm**4
        + lorentz_fwhm**5
    ) ** 0.2
    ratio = lorentz_fwhm / max(voigt_fwhm, np.finfo(np.float64).tiny)
    mixing = np.clip(
        1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3,
        0.0,
        1.0,
    )
    offset = wavelength_angstrom - center_angstrom
    gaussian = (
        2.0 * np.sqrt(np.log(2.0))
        / (np.sqrt(np.pi) * voigt_fwhm)
        * np.exp(-4.0 * np.log(2.0) * (offset / voigt_fwhm) ** 2)
    )
    lorentz = (
        2.0 / (np.pi * voigt_fwhm)
        / (1.0 + 4.0 * (offset / voigt_fwhm) ** 2)
    )
    return (1.0 - mixing) * gaussian + mixing * lorentz


def helium_i_resonance_line_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lines: tuple[HeliumResonanceLine, ...] = HELIUM_I_RESONANCE_LINES,
    include_occupation_probability: bool = True,
    include_stark_broadening: bool = True,
    maximum_stark_upper_level: int = 8,
) -> FloatArray:
    """Return approximate ground-state He I resonance-series opacity.

    The wavelengths and oscillator strengths are those in TLUSTY's public
    He I model atom.  Thermal Doppler, classical radiative, Ali--Griem
    resonance, and tabulated electron-impact Stark broadening are represented
    by an area-conserving pseudo-Voigt.  The 584-A line also includes the
    tabulated He II-perturber impact width.
    """

    state = _require_helium_state(atmosphere)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    integrated_frequency_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    ground_population = state.neutral_level_population_density[:, 0]
    ground_occupation = state.neutral_level_occupation_probability[:, 0]
    helium_atomic_mass = np.full(atmosphere.n_depth, HELIUM_MASS)
    for line in lines:
        center = line.wavelength_vacuum_angstrom
        center_cm = center * 1.0e-8
        upper_occupation = helium_occupation_probability(
            state.neutral_he_density,
            state.electron_density,
            atmosphere.temperature,
            float(line.upper_principal_quantum_number),
            neutral_h_density=atmosphere.neutral_h_density,
            neutral_radius_scale=state.neutral_radius_scale,
            hydrogen_neutral_radius_scale=(
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None else 1.0
            ),
            correlated_microfields=(state.microfield_model == "qmhd"),
        )
        survival = np.ones(atmosphere.n_depth)
        if include_occupation_probability:
            survival = np.minimum(
                upper_occupation
                / np.maximum(ground_occupation, np.finfo(np.float64).tiny),
                1.0,
            )
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED
            / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        for depth in range(atmosphere.n_depth):
            gaussian_sigma = center * np.sqrt(
                BOLTZMANN * atmosphere.temperature[depth]
                / (helium_atomic_mass[depth] * LIGHT_SPEED**2)
            )
            line_frequency = LIGHT_SPEED / center_cm
            radiative_rate = 2.47342e-22 * line_frequency**2
            radiative_hwhm = (
                center_cm**2 * radiative_rate / (4.0 * PI * LIGHT_SPEED)
                * 1.0e8
            )
            resonance_hwhm = helium_i_resonance_hwhm_angstrom(
                center,
                center,
                line.absorption_oscillator_strength,
                ground_population[depth],
            )
            stark_hwhm = 0.0
            if (
                include_stark_broadening
                and line.upper_principal_quantum_number
                <= maximum_stark_upper_level
            ):
                stark_hwhm = helium_i_resonance_stark_hwhm_angstrom(
                    atmosphere.temperature[depth],
                    state.electron_density[depth],
                    state.singly_ionized_he_density[depth],
                    line.upper_principal_quantum_number,
                )
            profile_lambda = _pseudo_voigt_profile_per_angstrom(
                wavelength,
                center,
                float(gaussian_sigma),
                float(radiative_hwhm + resonance_hwhm + stark_hwhm),
            )
            # The impact form is not valid arbitrarily far from the compact
            # EUV series.  Keep its blanketing within the ground-series band.
            profile_lambda = np.where(
                (wavelength >= 480.0) & (wavelength <= 700.0),
                profile_lambda,
                0.0,
            )
            profile_frequency = profile_lambda * 1.0e8 * center_cm**2 / LIGHT_SPEED
            result[:, depth] += (
                integrated_frequency_cross_section
                * line.absorption_oscillator_strength
                * ground_population[depth]
                * survival[depth]
                * stimulated[depth]
                * profile_frequency
                / atmosphere.mass_density[depth]
            )
    return result


def helium_i_line_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    table: HeliumStarkTable | str | Path,
    *,
    lines: tuple[HeliumLine, ...] = HELIUM_I_LINES,
    include_occupation_probability: bool = True,
    neutral_broadening: Literal["none", "unsold", "montreal"] = "unsold",
) -> FloatArray:
    """Return the 36 tabulated He I line opacities in cm^2 g^-1.

    The released profiles contain Stark and thermal Doppler broadening.
    ``neutral_broadening='unsold'`` adds just the classical van der Waals
    width.  ``'montreal'`` uses the modified Montreal prescription: the
    larger of the Unsold and Deridder--van Rensbergen widths (except 4121 and
    4713, which remain Unsold-only), followed by the larger of that result
    and the Ali--Griem resonance HWHM for singlets.
    """

    state = _require_helium_state(atmosphere)
    if not isinstance(table, HeliumStarkTable):
        table = read_helium_stark_table(table)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if neutral_broadening not in ("none", "unsold", "montreal"):
        raise ValueError(
            "neutral_broadening must be 'none', 'unsold', or 'montreal'"
        )
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    integrated_frequency_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    for line in lines:
        profile_table = table[line.table_key_angstrom]
        lower_population = state.neutral_level_population_density[:, line.lower_term_index]
        lower_occupation = state.neutral_level_occupation_probability[:, line.lower_term_index]
        upper_occupation = helium_occupation_probability(
            state.neutral_he_density,
            state.electron_density,
            atmosphere.temperature,
            float(line.upper_principal_quantum_number),
            neutral_h_density=atmosphere.neutral_h_density,
            neutral_radius_scale=state.neutral_radius_scale,
            hydrogen_neutral_radius_scale=(
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None else 1.0
            ),
            correlated_microfields=(state.microfield_model == "qmhd"),
        )
        survival = np.ones(atmosphere.n_depth)
        if include_occupation_probability:
            survival = np.minimum(
                upper_occupation / np.maximum(lower_occupation, np.finfo(np.float64).tiny),
                1.0,
            )
        center_cm = line.wavelength_vacuum_angstrom * 1.0e-8
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED
            / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        for depth in range(atmosphere.n_depth):
            lorentz_hwhm = 0.0
            if neutral_broadening != "none":
                upper_energy = (
                    HELIUM_I_LOW_TERM_ENERGY[line.lower_term_index]
                    + PLANCK * LIGHT_SPEED / center_cm
                )
                upper_binding = max(
                    HELIUM_FIRST_IONIZATION_ENERGY - upper_energy,
                    np.finfo(np.float64).tiny,
                )
                upper_effective_level = np.sqrt(
                    HYDROGEN_IONIZATION_ENERGY / upper_binding
                )
                mean_square_radius = 2.5 * upper_effective_level**4
                neutral_damping_rate = (
                    4.5e-9
                    * 0.42
                    * state.neutral_level_population_density[depth, 0]
                    * (atmosphere.temperature[depth] / 10_000.0) ** 0.3
                    * mean_square_radius**0.4
                )
                neutral_hwhm = (
                    center_cm**2
                    * neutral_damping_rate
                    / (4.0 * PI * LIGHT_SPEED)
                    * 1.0e8
                )
                if (
                    neutral_broadening == "montreal"
                    and line.table_key_angstrom
                    not in _HE_I_UNSOLD_ONLY_LINE_KEYS
                ):
                    lower_binding = max(
                        HELIUM_FIRST_IONIZATION_ENERGY
                        - HELIUM_I_LOW_TERM_ENERGY[line.lower_term_index],
                        np.finfo(np.float64).tiny,
                    )
                    lower_effective_level = np.sqrt(
                        HYDROGEN_IONIZATION_ENERGY / lower_binding
                    )
                    if line.lower_term_index in (1, 2):
                        lower_orbital = "s"
                        upper_orbital = "p"
                    elif line.table_key_angstrom in _HE_I_P_TO_S_LINE_KEYS:
                        lower_orbital = "p"
                        upper_orbital = "s"
                    else:
                        lower_orbital = "p"
                        upper_orbital = "d"
                    deridder_hwhm = (
                        deridder_van_rensbergen_helium_hwhm_angstrom(
                            atmosphere.temperature[depth],
                            state.neutral_level_population_density[depth, 0],
                            line.wavelength_vacuum_angstrom,
                            float(lower_effective_level),
                            float(upper_effective_level),
                            lower_orbital,
                            upper_orbital,
                        )
                    )
                    neutral_hwhm = max(
                        float(neutral_hwhm), float(deridder_hwhm)
                    )
                resonance_transition = _singlet_resonance_transition(line)
                if (
                    neutral_broadening == "montreal"
                    and resonance_transition is not None
                ):
                    resonance_hwhm = helium_i_resonance_hwhm_angstrom(
                        line.wavelength_vacuum_angstrom,
                        resonance_transition.wavelength_vacuum_angstrom,
                        resonance_transition.absorption_oscillator_strength,
                        state.neutral_level_population_density[depth, 0],
                    )
                    neutral_hwhm = max(
                        float(neutral_hwhm), float(resonance_hwhm)
                    )
                line_frequency = LIGHT_SPEED / center_cm
                radiative_damping_rate = 2.47342e-22 * line_frequency**2
                radiative_hwhm = (
                    center_cm**2
                    * radiative_damping_rate
                    / (4.0 * PI * LIGHT_SPEED)
                    * 1.0e8
                )
                lorentz_hwhm = neutral_hwhm + radiative_hwhm
            blue_extent = abs(profile_table.wavelength_offset_angstrom[0])
            red_extent = profile_table.wavelength_offset_angstrom[-1]
            extra_extent = 25.0 * lorentz_hwhm
            selected = (
                (wavelength >= line.wavelength_vacuum_angstrom - blue_extent - extra_extent)
                & (wavelength <= line.wavelength_vacuum_angstrom + red_extent + extra_extent)
            )
            if not np.any(selected):
                continue
            profile_lambda = profile_table.wavelength_profile(
                wavelength[selected],
                line.wavelength_vacuum_angstrom,
                float(atmosphere.temperature[depth]),
                float(max(atmosphere.electron_density[depth], 1.0)),
                lorentz_hwhm_angstrom=float(lorentz_hwhm),
            )
            # phi_lambda is per Angstrom.  Convert it to phi_nu before using
            # the frequency-integrated oscillator-strength cross section.
            profile_frequency = profile_lambda * 1.0e8 * center_cm**2 / LIGHT_SPEED
            result[selected, depth] += (
                integrated_frequency_cross_section
                * line.absorption_oscillator_strength
                * lower_population[depth]
                * survival[depth]
                * stimulated[depth]
                * profile_frequency
                / atmosphere.mass_density[depth]
            )
    return result


def helium_ii_line_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lines: tuple[HeliumIILine, ...] = HELIUM_II_LINES,
    stark_table: HeliumIIStarkTable | str | Path | None = None,
    include_occupation_probability: bool = True,
) -> FloatArray:
    """Return approximate non-ideal hydrogenic He II line opacity.

    When ``stark_table`` is supplied, the public Schönning--Butler profiles
    distributed with SYNSPEC are used for their 19 transitions.  Remaining
    series members fall back to transformed unified hydrogen profiles with
    the linear-Stark hydrogenic wavelength scaling
    ``delta_lambda proportional to Z^-5``.  The fallback preserves profile
    area and is considerably more realistic in dense DB atmospheres than a
    Voigt approximation, but its Doppler core is only approximate.
    """

    from .stark import (
        default_balmer_stark_table,
        default_brackett_stark_table,
        default_lyman_stark_table,
        default_paschen_stark_table,
    )

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if not lines:
        return np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    maximum_level = max(line.upper_principal_quantum_number for line in lines)
    if stark_table is not None and not isinstance(stark_table, HeliumIIStarkTable):
        stark_table = read_helium_ii_stark_table(stark_table)
    population, occupation = _helium_ii_level_distribution(
        atmosphere, maximum_level
    )
    tables = {
        1: default_lyman_stark_table(),
        2: default_balmer_stark_table(),
        3: default_paschen_stark_table(),
        4: default_brackett_stark_table(),
    }
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    integrated_frequency_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    wavelength_scale = 2.0**-5
    rydberg_hydrogen_cm = 109_678.77
    for line in lines:
        transition = (
            line.lower_principal_quantum_number,
            line.upper_principal_quantum_number,
        )
        exact_line = (
            stark_table[transition]
            if stark_table is not None and transition in stark_table.lines
            else None
        )
        table_line = None
        if exact_line is None:
            hydrogen_table = tables[line.lower_principal_quantum_number]
            try:
                table_line = hydrogen_table[transition]
            except KeyError:
                # The public He II table reaches 4->15, one member beyond the
                # bundled Brackett grid.  If that optional table is absent,
                # continue the nearest high-series unified shape.
                available_upper = max(
                    upper
                    for lower, upper in hydrogen_table.lines
                    if lower == line.lower_principal_quantum_number
                    and upper < line.upper_principal_quantum_number
                )
                table_line = hydrogen_table[
                    (line.lower_principal_quantum_number, available_upper)
                ]
        lower_index = line.lower_principal_quantum_number - 1
        upper_index = line.upper_principal_quantum_number - 1
        lower_population = population[:, lower_index]
        survival = np.ones(atmosphere.n_depth, dtype=np.float64)
        if include_occupation_probability:
            survival = np.clip(
                occupation[:, upper_index]
                / np.maximum(
                    occupation[:, lower_index], np.finfo(np.float64).tiny
                ),
                0.0,
                1.0,
            )
        center = line.wavelength_vacuum_angstrom
        center_cm = center * 1.0e-8
        hydrogen_center = 1.0e8 / (
            rydberg_hydrogen_cm
            * (
                1.0 / line.lower_principal_quantum_number**2
                - 1.0 / line.upper_principal_quantum_number**2
            )
        )
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED
            / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        for depth in range(atmosphere.n_depth):
            if exact_line is not None:
                selected = (
                    np.abs(wavelength - center)
                    <= exact_line.wavelength_offset_angstrom[-1]
                )
                if not np.any(selected):
                    continue
                profile_frequency = exact_line.frequency_profile(
                    wavelength[selected],
                    center,
                    float(atmosphere.temperature[depth]),
                    float(max(atmosphere.electron_density[depth], 1.0)),
                )
                result[selected, depth] += (
                    integrated_frequency_cross_section
                    * line.absorption_oscillator_strength
                    * lower_population[depth]
                    * survival[depth]
                    * stimulated[depth]
                    * profile_frequency
                    / atmosphere.mass_density[depth]
                )
                continue
            assert table_line is not None
            field_strength = 1.25e-9 * max(
                atmosphere.electron_density[depth], 1.0
            ) ** (2.0 / 3.0)
            extent = (
                field_strength
                * 10.0 ** table_line.log_alpha[-1]
                * wavelength_scale
            )
            mapped_wavelength = hydrogen_center + (
                wavelength - center
            ) / wavelength_scale
            selected = (
                (np.abs(wavelength - center) <= extent)
                & (mapped_wavelength > 0.0)
            )
            if not np.any(selected):
                continue
            mapped_hydrogen_wavelength = mapped_wavelength[selected]
            profile_lambda = (
                table_line.wavelength_profile(
                    mapped_hydrogen_wavelength,
                    hydrogen_center,
                    float(atmosphere.temperature[depth]),
                    float(max(atmosphere.electron_density[depth], 1.0)),
                )
                / wavelength_scale
            )
            profile_frequency = (
                profile_lambda * 1.0e8 * center_cm**2 / LIGHT_SPEED
            )
            result[selected, depth] += (
                integrated_frequency_cross_section
                * line.absorption_oscillator_strength
                * lower_population[depth]
                * survival[depth]
                * stimulated[depth]
                * profile_frequency
                / atmosphere.mass_density[depth]
            )
    return result


def rosseland_mean_helium_continuum_opacity(
    atmosphere: Atmosphere,
    *,
    n_frequency: int = 240,
    wavelength_angstrom: ArrayLike | None = None,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
) -> FloatArray:
    """Return the Rosseland mean of the implemented pure-He continuum."""

    state = _require_helium_state(atmosphere)
    if n_frequency < 40:
        raise ValueError("n_frequency must be at least 40")
    if wavelength_angstrom is not None:
        from ._rosseland import rosseland_mean_from_opacity_grid

        return rosseland_mean_from_opacity_grid(
            wavelength_angstrom,
            helium_continuum_mass_absorption_coefficient(
                atmosphere, wavelength_angstrom,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            ),
            atmosphere.temperature,
        )
    x = np.geomspace(0.1, 30.0, n_frequency)
    exponential = np.exp(x)
    weight = x**4 * exponential / np.expm1(x) ** 2
    result = np.empty(atmosphere.n_depth)
    for depth in range(atmosphere.n_depth):
        wavelength = PLANCK * LIGHT_SPEED / (BOLTZMANN * atmosphere.temperature[depth] * x) * 1.0e8
        order = np.argsort(wavelength)
        local_state = HeliumLTEState(
            **{
                field: getattr(state, field)[depth:depth + 1]
                if isinstance(getattr(state, field), np.ndarray)
                else getattr(state, field)
                for field in state.__dataclass_fields__
            }
        )
        point = Atmosphere(
            atmosphere.effective_temperature, atmosphere.logg,
            atmosphere.rosseland_optical_depth[depth:depth + 1],
            atmosphere.column_mass[depth:depth + 1],
            atmosphere.temperature[depth:depth + 1],
            atmosphere.gas_pressure[depth:depth + 1],
            atmosphere.mass_density[depth:depth + 1],
            np.zeros(1), np.zeros(1), atmosphere.electron_density[depth:depth + 1],
            atmosphere.metadata, helium_lte_state=local_state,
        )
        opacity = helium_continuum_mass_absorption_coefficient(
            point,
            wavelength[order],
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )[:, 0]
        inverse = trapezoid(
            weight[order] / np.maximum(opacity, 1.0e-30), x[order]
        ) / trapezoid(weight[order], x[order])
        result[depth] = 1.0 / inverse
    return result
