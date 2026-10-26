"""LTE hydrogen continuum and line opacity."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import gamma as gamma_function
import os
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid
from .atmosphere import Atmosphere
from .eos import (
    HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    HydrogenLevelDistribution,
    HydrogenLTEState,
    charged_particle_hydrogen_occupation_probability,
    hydrogen_level_distribution,
    hydrogen_occupation_probability,
)
from .constants import (
    BOLTZMANN,
    BOHR_RADIUS,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    FINE_STRUCTURE_CONSTANT,
    HYDROGEN_IONIZATION_ENERGY,
    HYDROGEN_MASS,
    HELIUM_MASS,
    LIGHT_SPEED,
    PI,
    PLANCK,
    THOMSON_CROSS_SECTION,
)
from .molecules import (
    H2H2CollisionInducedAbsorptionTable,
    LOSCHMIDT_NUMBER_DENSITY,
    molecular_hydrogen_ion_photoabsorption_cross_section,
    molecular_hydrogen_photoabsorption_cross_section,
)
from .hydrogen_self import BarklemSelfBroadeningTable
from .gaunt import (
    hydrogen_bound_free_gaunt_factor,
    hydrogen_free_free_gaunt_factor,
)
from .stark import (
    HydrogenStarkTable,
    default_balmer_stark_table,
    default_lyman_stark_table,
)

try:  # Optional acceleration built by setup.py.
    from . import _rt
except ImportError:  # pragma: no cover - exercised in source-only installs
    _rt = None

if TYPE_CHECKING:
    from .jackson_lyman import JacksonLymanProfileTable
    from .quasimolecular import (
        AllardNeutralLymanAlphaTable,
        AllardUnifiedLymanTable,
    )


FloatArray = NDArray[np.float64]
def _atmosphere_level_distribution(
    atmosphere: Atmosphere, maximum_level: int
) -> HydrogenLevelDistribution:
    """Reuse the HM populations already produced by the atmosphere EOS."""

    state = atmosphere.hydrogen_lte_state
    if (
        state is None
        or state.level_occupation_probability is None
        or state.level_population_density is None
        or state.internal_partition_function is None
        or state.level_population_density.ndim != 2
        or state.level_population_density.shape[0] != atmosphere.n_depth
        or state.level_population_density.shape[1] < maximum_level
    ):
        distribution = hydrogen_level_distribution(
            atmosphere.neutral_h_density,
            atmosphere.electron_density,
            atmosphere.temperature,
            # The HM partition function is defined by the package-wide
            # 40-shell atom even when a caller requests only a few population
            # columns.  Normalizing a freshly constructed atmosphere to the
            # requested subset would make the same level population depend on
            # which opacity happened to ask for it first.
            maximum_level=max(40, maximum_level),
            neutral_he_density=(
                atmosphere.helium_lte_state.neutral_he_density
                if atmosphere.helium_lte_state is not None
                else 0.0
            ),
            neutral_radius_scale=(
                state.neutral_radius_scale
                if state is not None
                else HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            ),
            helium_neutral_radius_scale=(
                atmosphere.helium_lte_state.neutral_radius_scale
                if atmosphere.helium_lte_state is not None
                else 0.5
            ),
            correlated_microfields=(
                state is not None and state.microfield_model == "qmhd"
            ),
        )
        return HydrogenLevelDistribution(
            principal_quantum_number=(
                distribution.principal_quantum_number[:maximum_level]
            ),
            occupation_probability=(
                distribution.occupation_probability[:, :maximum_level]
            ),
            population_fraction=(
                distribution.population_fraction[:, :maximum_level]
            ),
            population_density=(
                distribution.population_density[:, :maximum_level]
            ),
            internal_partition_function=distribution.internal_partition_function,
        )

    population = state.level_population_density[:, :maximum_level]
    occupation = state.level_occupation_probability[:, :maximum_level]
    fraction = np.divide(
        population,
        atmosphere.neutral_h_density[:, np.newaxis],
        out=np.zeros_like(population),
        where=atmosphere.neutral_h_density[:, np.newaxis] > 0.0,
    )
    return HydrogenLevelDistribution(
        principal_quantum_number=np.arange(
            1, maximum_level + 1, dtype=np.float64
        ),
        occupation_probability=occupation,
        population_fraction=fraction,
        population_density=population,
        internal_partition_function=state.internal_partition_function,
    )


def _slice_hydrogen_lte_state(
    state: HydrogenLTEState | None, depth: int
) -> HydrogenLTEState | None:
    """Return a one-depth view of a cached EOS state."""

    if state is None:
        return None

    def sliced(values: FloatArray | None) -> FloatArray | None:
        return None if values is None else values[depth : depth + 1]

    return HydrogenLTEState(
        mass_density=state.mass_density[depth : depth + 1],
        hydrogen_nuclei_density=state.hydrogen_nuclei_density[depth : depth + 1],
        neutral_h_density=state.neutral_h_density[depth : depth + 1],
        proton_density=state.proton_density[depth : depth + 1],
        electron_density=state.electron_density[depth : depth + 1],
        ionization_fraction=state.ionization_fraction[depth : depth + 1],
        internal_partition_function=sliced(state.internal_partition_function),
        level_occupation_probability=sliced(state.level_occupation_probability),
        level_population_density=sliced(state.level_population_density),
        microfield_model=state.microfield_model,
        molecular_hydrogen_density=sliced(state.molecular_hydrogen_density),
        molecular_hydrogen_ion_density=sliced(
            state.molecular_hydrogen_ion_density
        ),
        negative_hydrogen_density=sliced(state.negative_hydrogen_density),
        trihydrogen_ion_density=sliced(state.trihydrogen_ion_density),
        trihydrogen_ion_partition_model=(
            state.trihydrogen_ion_partition_model
        ),
        chemical_model=state.chemical_model,
        neutral_radius_scale=state.neutral_radius_scale,
    )


@dataclass(frozen=True)
class HydrogenLine:
    name: str
    lower_level: int
    upper_level: int
    wavelength_vacuum_angstrom: float
    absorption_oscillator_strength: float


BALMER_LINES = (
    HydrogenLine("Halpha", 2, 3, 6564.636, 0.640747044864),
    HydrogenLine("Hbeta", 2, 4, 4862.694, 0.119321144112),
    HydrogenLine("Hgamma", 2, 5, 4341.691, 0.0446702946311),
    HydrogenLine("Hdelta", 2, 6, 4102.898, 0.0220928192139),
    HydrogenLine("Hepsilon", 2, 7, 3971.200, 0.0127046242104),
    HydrogenLine("H8", 2, 8, 3890.155, 0.0080355585203),
    HydrogenLine("H9", 2, 9, 3836.476, 0.00542894641045),
    HydrogenLine("H10", 2, 10, 3798.979, 0.00385060974669),
    HydrogenLine("H11", 2, 11, 3771.705, 0.00283535426889),
    HydrogenLine("H12", 2, 12, 3751.221, 0.00215095165274),
    HydrogenLine("H13", 2, 13, 3735.433, 0.00167194833581),
    HydrogenLine("H14", 2, 14, 3723.000, 0.00132623212496),
    HydrogenLine("H15", 2, 15, 3713.030, 0.00107021839496),
    HydrogenLine("H16", 2, 16, 3704.909, 0.000876448450437),
    HydrogenLine("H17", 2, 17, 3698.206, 0.000727008178425),
    HydrogenLine("H18", 2, 18, 3692.608, 0.000609856418053),
    HydrogenLine("H19", 2, 19, 3687.883, 0.000516688000504),
    HydrogenLine("H20", 2, 20, 3683.859, 0.000441643912075),
    HydrogenLine("H21", 2, 21, 3680.403, 0.000380507757107),
    HydrogenLine("H22", 2, 22, 3677.412, 0.000330191084858),
)


def _lyman_oscillator_strength(upper_level: int) -> float:
    """Exact shell-averaged hydrogenic oscillator strength for 1s->n."""

    n = float(upper_level)
    return float(
        2.0**8
        * n**5
        * (n - 1.0) ** (2.0 * n - 4.0)
        / (3.0 * (n + 1.0) ** (2.0 * n + 4.0))
    )


_LYMAN_NAMES = ("Lyalpha", "Lybeta", "Lygamma", "Lydelta", "Lyepsilon")
LYMAN_LINES = tuple(
    HydrogenLine(
        _LYMAN_NAMES[upper - 2] if upper <= 6 else f"Ly{upper}",
        1,
        upper,
        1.0e8 / (109_678.77 * (1.0 - 1.0 / upper**2)),
        _lyman_oscillator_strength(upper),
    )
    for upper in range(2, 22)
)

# Exact shell-averaged non-relativistic hydrogenic absorption oscillator
# strengths.  They are evaluated by summing the analytic dipole radial
# integrals over every degenerate l -> l +/- 1 component of the lower shell.
# The first three values reproduce the Green et al. compilation quoted by
# Rubino-Martin et al. (2005): 0.8421, 0.1506, and 0.0558 for Paschen and
# 1.0377, 0.1793, and 0.0655 for Brackett.  Keeping the evaluated constants
# here avoids introducing a special-function runtime dependency.
_PASCHEN_OSCILLATOR_STRENGTHS = (
    0.842096357532266,
    0.150584082803105,
    0.0558402477747247,
    0.0276848879634103,
    0.0160358274083514,
    0.0102341026067731,
    0.00697970910768096,
    0.00499641990757861,
    0.00371148300653813,
    0.00283882995035591,
    0.00222355447286657,
    0.00177629440342198,
    0.00144280308894986,
    0.00118873916624181,
    0.000991583947274797,
    0.000836122308552389,
    0.000711805017359386,
    0.000611151435641504,
    0.000528750858802903,
)
_BRACKETT_OSCILLATOR_STRENGTHS = (
    1.03773625884024,
    0.179252374278725,
    0.0654859897311483,
    0.0322951583475062,
    0.0186995864209931,
    0.0119611130243852,
    0.00818718978958117,
    0.00588609864644224,
    0.00439250880501866,
    0.00337542785213278,
)

_PASCHEN_NAMES = ("Paalpha", "Pabeta", "Pagamma", "Padelta", "Paepsilon")
PASCHEN_LINES = tuple(
    HydrogenLine(
        _PASCHEN_NAMES[upper - 4] if upper <= 8 else f"Pa{upper}",
        3,
        upper,
        1.0e8 / (109_678.77 * (1.0 / 3.0**2 - 1.0 / upper**2)),
        oscillator_strength,
    )
    for upper, oscillator_strength in zip(
        range(4, 23), _PASCHEN_OSCILLATOR_STRENGTHS
    )
)

_BRACKETT_NAMES = ("Bralpha", "Brbeta", "Brgamma", "Brdelta", "Brepsilon")
BRACKETT_LINES = tuple(
    HydrogenLine(
        _BRACKETT_NAMES[upper - 5] if upper <= 9 else f"Br{upper}",
        4,
        upper,
        1.0e8 / (109_678.77 * (1.0 / 4.0**2 - 1.0 / upper**2)),
        oscillator_strength,
    )
    for upper, oscillator_strength in zip(
        range(5, 15), _BRACKETT_OSCILLATOR_STRENGTHS
    )
)

# Dominant b^3 Sigma_u -> a^3 Sigma_g contribution to the neutral-H
# collision-induced Lyalpha red wing.  These values are a digitized,
# piecewise-linear representation of the T=6000 K, n_H=1e20 cm^-3 curve in
# Fig. 6 of Rohrmann, Althaus & Kepler (2011, MNRAS 411, 781).  The ordinate
# is log10(sigma / n_perturber) in cm^5.  That paper reports close agreement
# with Kowalski & Saumon for this far-red-wing contribution.
_LYMAN_ALPHA_H_H_WAVELENGTH_NM = np.array(
    [
        125.0,
        130.0,
        140.0,
        150.0,
        160.0,
        162.3,
        170.0,
        180.0,
        200.0,
        225.0,
        250.0,
        275.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
    ],
    dtype=np.float64,
)
_LYMAN_ALPHA_H_H_LOG_CROSS_SECTION_PER_PERTURBER = np.array(
    [
        -39.26260,
        -39.86647,
        -40.41808,
        -40.78944,
        -41.08712,
        -41.14466,
        -41.35117,
        -41.60045,
        -42.03993,
        -42.51278,
        -42.92833,
        -43.28721,
        -43.61340,
        -44.17085,
        -44.64702,
        -45.04283,
        -45.40680,
        -45.72525,
        -46.01688,
    ],
    dtype=np.float64,
)

_LYMAN_ALPHA_H_H_X_B_WAVELENGTH_NM = np.array(
    [
        125.0,
        130.0,
        140.0,
        150.0,
        155.0,
        159.0,
        162.3,
        165.0,
        167.5,
        170.0,
        172.5,
        175.0,
        177.5,
        180.0,
        190.0,
        200.0,
    ],
    dtype=np.float64,
)
_LYMAN_ALPHA_H_H_X_B_LOG_CROSS_SECTION_PER_PERTURBER = np.array(
    [
        -37.8743,
        -38.0709,
        -38.4169,
        -38.6967,
        -38.7675,
        -38.7280,
        -38.7200,
        -40.0000,
        -40.5000,
        -40.8000,
        -41.1000,
        -41.4000,
        -41.8000,
        -42.3000,
        -44.0000,
        -45.7000,
    ],
    dtype=np.float64,
)

# Angle-averaged H--H2 contribution to the Lyalpha red wing.  This is a
# compact digitization of the sum of the solid E1--E3 and E1--E4 KG--RK
# curves in the upper panel of Fig. 8 of Rohrmann, Althaus & Kepler (2011).
# The published calculation is for T=6000 K and n(H2)=1e20 cm^-3; as for the
# H--H curves above, the ordinate stored here is log10(sigma/n_perturber) in
# cm^5.  The molecular wing starts at the isolated Lyalpha wavelength, is
# strongest in the 180--400 nm region, and remains measurable to 600 nm.
# It is kept separate from the Allard H/H+ coverage switch because H2 is an
# additional perturber species absent from those tables.
_LYMAN_ALPHA_H_H2_WAVELENGTH_NM = np.array(
    [
        121.6,
        125.0,
        130.0,
        140.0,
        150.0,
        175.0,
        200.0,
        225.0,
        250.0,
        275.0,
        300.0,
        325.0,
        350.0,
        375.0,
        400.0,
        425.0,
        450.0,
        475.0,
        500.0,
        525.0,
        550.0,
        575.0,
        600.0,
    ],
    dtype=np.float64,
)
_LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_PER_PERTURBER = np.array(
    [
        -37.95,
        -38.03,
        -38.78,
        -39.52,
        -39.90,
        -40.06,
        -40.42,
        -40.70,
        -41.01,
        -41.28,
        -41.58,
        -41.88,
        -42.10,
        -42.31,
        -42.55,
        -42.79,
        -43.03,
        -43.40,
        -43.64,
        -43.75,
        -43.91,
        -44.12,
        -44.28,
    ],
    dtype=np.float64,
)

# The Rohrmann et al. (2011) compact profile above is published only at
# 6000 K.  In the quasi-static limit the probability of the close H--H2
# configuration contains exp[-E_1(r)/(kT)], so its far-wing strength cannot
# be held fixed in a cool atmosphere.  These two curves are a vector-path
# digitization of the standard KS06 profiles plotted by Sahu et al. (2025,
# their Fig. 4) at 3000 and 6000 K.  Their 4000 and 5000 K curves verify that
# log(sigma/n_pert) is linear in 1/T to within 0.01 dex.  Only the difference
# between the curves is used below: the absolute 6000-K normalization and
# KG--RK wavelength dependence remain those of Rohrmann et al. above.
_LYMAN_ALPHA_H_H2_TEMPERATURE_WAVELENGTH_NM = np.array(
    [
        130.0,
        140.0,
        150.0,
        175.0,
        200.0,
        225.0,
        250.0,
        275.0,
        300.0,
        325.0,
        350.0,
        375.0,
        400.0,
        425.0,
        450.0,
        475.0,
        500.0,
        525.0,
        550.0,
        575.0,
        600.0,
    ],
    dtype=np.float64,
)
_LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_3000_K = np.array(
    [
        -39.1983,
        -39.5102,
        -39.8231,
        -40.5968,
        -41.3031,
        -41.9252,
        -42.4655,
        -42.9383,
        -43.3856,
        -43.7930,
        -44.1636,
        -44.4905,
        -44.7927,
        -45.0681,
        -45.3343,
        -45.5933,
        -45.8398,
        -46.0782,
        -46.3044,
        -46.5236,
        -46.7330,
    ],
    dtype=np.float64,
)
_LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_6000_K = np.array(
    [
        -39.0654,
        -39.2601,
        -39.4553,
        -39.9394,
        -40.3914,
        -40.7617,
        -41.0685,
        -41.3320,
        -41.5887,
        -41.8277,
        -42.0427,
        -42.2282,
        -42.3998,
        -42.5515,
        -42.7004,
        -42.8438,
        -42.9819,
        -43.1109,
        -43.2375,
        -43.3590,
        -43.4759,
    ],
    dtype=np.float64,
)


def _h_h2_lyman_alpha_log_temperature_correction(
    wavelength_nm: FloatArray, temperature: FloatArray
) -> FloatArray:
    """Return the local-temperature correction to the 6000-K H--H2 wing.

    The result has shape ``(n_wavelength, n_depth)`` and is logarithmic in
    base ten.  The measured 3000--6000 K dependence is interpolated in
    inverse temperature, as required by the quasi-static pair probability.
    It is conservatively held fixed outside the published temperature range.
    """

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    local_temperature = np.asarray(temperature, dtype=np.float64)
    log_difference = np.interp(
        wavelength,
        _LYMAN_ALPHA_H_H2_TEMPERATURE_WAVELENGTH_NM,
        (
            _LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_3000_K
            - _LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_6000_K
        ),
        left=0.0,
        right=(
            _LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_3000_K[-1]
            - _LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_6000_K[-1]
        ),
    )
    clipped_temperature = np.clip(local_temperature, 3000.0, 6000.0)
    inverse_temperature_fraction = (
        1.0 / clipped_temperature - 1.0 / 6000.0
    ) / (1.0 / 3000.0 - 1.0 / 6000.0)
    return log_difference[:, np.newaxis] * inverse_temperature_fraction[np.newaxis, :]

# Barklem, Piskunov & O'Mara (2000), their Table 3.  Cross sections are in
# a0^2 at 10 km/s; alpha is the exponent sigma(v) proportional to v^-alpha;
# the final value is their approximate maximum detuning for validity of the
# impact approximation at 14 km/s.
_BALMER_SELF_BROADENING = {
    3: (1180.0, 0.677, 35.0),
    4: (2320.0, 0.455, 13.2),
    5: (4208.0, 0.380, 7.7),
}

# Allard, Kielkopf, Cayrel & van 't Veer-Menneret (2008), A&A 480,
# 581, Table 2.  These are the thermally averaged total Halpha HWHM per
# neutral-H perturber in 1e-8 rad s^-1 cm^3.  Unlike the mean-collision-speed
# column of that table, the values below perform the required Maxwellian
# velocity average.  The published calculation covers 3000--12000 K.
_ALLARD_2008_HALPHA_TEMPERATURE_K = np.arange(3_000.0, 12_000.1, 1_000.0)
_ALLARD_2008_HALPHA_ANGULAR_HWHM_PER_PERTURBER = 1.0e-8 * np.asarray(
    [4.03, 4.14, 4.21, 4.26, 4.30, 4.35, 4.40, 4.43, 4.45, 4.49],
    dtype=np.float64,
)

# Classic Ali--Griem resonant-dipole fallback for Balmer members above Hgamma,
# where the Barklem, Piskunov & O'Mara calculations are unavailable.  In cgs,
# C3(1,n) = 2.015e7 f(1,n) / nu(1,n), and the angular damping constant of a
# level is 24.086 sqrt(g1/gn) n_1 C3.  A transition's two level widths add.
_ALI_GRIEM_C3_COEFFICIENT = 2.015e7
_ALI_GRIEM_DAMPING_COEFFICIENT = 24.086

# Static dipole polarizabilities in A^3.  Their ratio converts the classical
# Unsold/Warner neutral-H van-der-Waals rate to a neutral-He perturber.  This is
# the same externally checked scaling used by the metal-line implementation.
_HYDROGEN_STATIC_POLARIZABILITY_A3 = 0.666_793
_HELIUM_STATIC_POLARIZABILITY_A3 = 0.204_956

_CAUCHY_QUADRATURE_NODES, _CAUCHY_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(128)
)
_CAUCHY_QUADRATURE_BLOCK_SIZE = 16
_IMPACT_VALIDITY_TAPER_FRACTION = 0.40


def _hydrogen_profile_worker_count(n_depth: int) -> int:
    """Return the bounded worker count for compiled hydrogen profiles."""

    configured = os.environ.get("OPENWD_NUM_THREADS")
    if configured is None:
        requested = min(8, os.cpu_count() or 1)
    else:
        try:
            requested = int(configured)
        except ValueError as exc:
            raise ValueError(
                "OPENWD_NUM_THREADS must be a positive integer"
            ) from exc
        if requested < 1:
            raise ValueError("OPENWD_NUM_THREADS must be a positive integer")
    return min(n_depth, requested)


@lru_cache(maxsize=None)
def _cauchy_quadrature(order: int) -> tuple[FloatArray, FloatArray]:
    """Return cached Gauss--Legendre nodes for the Cauchy convolution."""

    if order < 8:
        raise ValueError("Cauchy quadrature order must be at least 8")
    if order == _CAUCHY_QUADRATURE_NODES.size:
        return _CAUCHY_QUADRATURE_NODES, _CAUCHY_QUADRATURE_WEIGHTS
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _lorentz_convolved_stark_profile(
    stark_line: object,
    wavelength_angstrom: FloatArray,
    line_center_angstrom: float,
    temperature: float,
    electron_density: float,
    lorentz_hwhm_angstrom: float,
    maximum_impact_shift_angstrom: float | None = None,
    quadrature_order: int = 128,
    truncation_closure: str = "renormalize",
) -> FloatArray:
    """Convolve a Stark profile with a normalized Lorentz profile.

    With ``u = gamma tan(theta)``, the Cauchy convolution becomes a smooth
    finite integral, ``(1/pi) integral P(x-u) dtheta``.  Gauss-Legendre
    quadrature therefore preserves the line strength without constructing a
    large auxiliary uniform wavelength grid.  When an impact-validity limit
    is supplied, the Cauchy kernel is smoothly tapered over its outermost 40
    per cent.  ``renormalize`` rescales that supported interval to unit area.
    ``stark-core`` instead leaves the supported Cauchy probability unchanged
    and returns the missing probability to the unshifted Stark profile.  Both
    closures conserve line strength.  A hard cutoff translates the narrow
    Stark core to exactly plus/minus the cutoff and creates an artificial
    shoulder there; the compact taper avoids that numerical feature.
    """

    # A sub-micro-resolution Lorentz kernel is numerically indistinguishable
    # from a delta function on both the bundled Stark tables and the atmosphere
    # wavelength grids.  Skipping the 128-point convolution is especially
    # important for trace-H DBA layers, where the neutral-H width is positive
    # but many orders of magnitude below the Stark/Doppler core.
    if truncation_closure not in {"renormalize", "stark-core"}:
        raise ValueError(
            "truncation_closure must be 'renormalize' or 'stark-core'"
        )
    if lorentz_hwhm_angstrom <= 1.0e-5:
        return stark_line.wavelength_profile(
            wavelength_angstrom,
            line_center_angstrom,
            temperature,
            electron_density,
        )
    quadrature_nodes, quadrature_weights = _cauchy_quadrature(quadrature_order)
    profile = np.zeros_like(wavelength_angstrom)
    detuning = wavelength_angstrom - line_center_angstrom
    angle_limit = 0.5 * PI
    if maximum_impact_shift_angstrom is not None:
        if maximum_impact_shift_angstrom <= 0.0:
            raise ValueError("maximum_impact_shift_angstrom must be positive")
        angle_limit = np.arctan(
            maximum_impact_shift_angstrom / lorentz_hwhm_angstrom
        )
    valid_kernel_measure = 2.0 * angle_limit
    if maximum_impact_shift_angstrom is not None:
        # The transformed Cauchy measure is uniform in angle.  Apodizing its
        # shift-space boundary removes the feature produced when the sharply
        # peaked Stark core crosses a hard kernel edge.  A quintic smoothstep
        # has zero first and second derivative at both ends of the taper.
        normalization_nodes, normalization_weights = _cauchy_quadrature(
            max(128, quadrature_order)
        )
        normalization_angle = angle_limit * normalization_nodes
        normalization_shift = np.abs(
            lorentz_hwhm_angstrom * np.tan(normalization_angle)
        )
        taper_start = (
            (1.0 - _IMPACT_VALIDITY_TAPER_FRACTION)
            * maximum_impact_shift_angstrom
        )
        taper_coordinate = np.clip(
            (normalization_shift - taper_start)
            / (maximum_impact_shift_angstrom - taper_start),
            0.0,
            1.0,
        )
        taper_weight = 1.0 - taper_coordinate**3 * (
            10.0 + taper_coordinate * (-15.0 + 6.0 * taper_coordinate)
        )
        valid_kernel_measure = angle_limit * float(
            np.sum(normalization_weights * taper_weight)
        )
    kernel_normalization = (
        valid_kernel_measure
        if truncation_closure == "renormalize"
        else PI
    )
    unresolved_probability = 0.0
    if (
        maximum_impact_shift_angstrom is not None
        and truncation_closure == "stark-core"
    ):
        unresolved_probability = float(
            np.clip(1.0 - valid_kernel_measure / PI, 0.0, 1.0)
        )
    compiled = (
        None
        if _rt is None
        else getattr(_rt, "hydrogen_stark_lorentz_convolution", None)
    )
    local_profile_state = getattr(stark_line, "_local_profile_state", None)
    if compiled is not None and local_profile_state is not None:
        field_strength, local_log_profile = local_profile_state(
            temperature, electron_density
        )
        return np.asarray(
            compiled(
                np.ascontiguousarray(detuning),
                np.ascontiguousarray(stark_line.log_alpha),
                np.ascontiguousarray(local_log_profile),
                np.ascontiguousarray(quadrature_nodes),
                np.ascontiguousarray(quadrature_weights),
                float(field_strength),
                float(lorentz_hwhm_angstrom),
                float(angle_limit),
                float(kernel_normalization),
                (
                    0.0
                    if maximum_impact_shift_angstrom is None
                    else float(maximum_impact_shift_angstrom)
                ),
                (
                    0.0
                    if maximum_impact_shift_angstrom is None
                    else float(taper_start)
                ),
                unresolved_probability,
            ),
            dtype=np.float64,
        )
    # Split each Cauchy integral where its shifted Stark profile reaches the
    # line center.  Treating that narrow cusp as an interval boundary removes
    # the fixed-node aliasing that otherwise appears as spikes on irregular
    # wavelength grids, especially for cool Halpha profiles.
    cusp_angle = np.clip(
        np.arctan(detuning / lorentz_hwhm_angstrom),
        -angle_limit,
        angle_limit,
    )
    for lower, upper in (
        (-angle_limit * np.ones_like(cusp_angle), cusp_angle),
        (cusp_angle, angle_limit * np.ones_like(cusp_angle)),
    ):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        for start in range(
            0,
            quadrature_nodes.size,
            _CAUCHY_QUADRATURE_BLOCK_SIZE,
        ):
            stop = start + _CAUCHY_QUADRATURE_BLOCK_SIZE
            node = quadrature_nodes[start:stop, np.newaxis]
            weight = quadrature_weights[start:stop, np.newaxis]
            angle = midpoint[np.newaxis, :] + half_width[np.newaxis, :] * node
            shifted_detuning = (
                detuning[np.newaxis, :]
                - lorentz_hwhm_angstrom * np.tan(angle)
            )
            impact_weight: FloatArray | float = 1.0
            if maximum_impact_shift_angstrom is not None:
                impact_shift = np.abs(lorentz_hwhm_angstrom * np.tan(angle))
                taper_coordinate = np.clip(
                    (impact_shift - taper_start)
                    / (maximum_impact_shift_angstrom - taper_start),
                    0.0,
                    1.0,
                )
                impact_weight = 1.0 - taper_coordinate**3 * (
                    10.0 + taper_coordinate * (-15.0 + 6.0 * taper_coordinate)
                )
            safe_center = max(
                line_center_angstrom,
                1.0 + float(np.max(np.abs(shifted_detuning))),
            )
            profile += (
                np.sum(
                    weight
                    * half_width[np.newaxis, :]
                    * impact_weight
                    / kernel_normalization
                    * stark_line.wavelength_profile(
                        safe_center + shifted_detuning,
                        safe_center,
                        temperature,
                        electron_density,
                    ),
                    axis=0,
                )
            )
    if (
        maximum_impact_shift_angstrom is not None
        and truncation_closure == "stark-core"
    ):
        # The impact calculation does not specify the line shape outside its
        # stated detuning range.  Reassigning that unknown probability to an
        # unperturbed (delta-function) neutral kernel conserves oscillator
        # strength without artificially amplifying every supported shift.
        # Convolving that delta component simply restores the corresponding
        # fraction of the original Stark profile.
        profile += unresolved_probability * stark_line.wavelength_profile(
            wavelength_angstrom,
            line_center_angstrom,
            temperature,
            electron_density,
        )
    return profile


def _tabulated_convolved_stark_profile(
    stark_line: object,
    wavelength_angstrom: FloatArray,
    line_center_angstrom: float,
    temperature: float,
    electron_density: float,
    kernel_offset_angstrom: FloatArray,
    kernel_profile_per_angstrom: FloatArray,
) -> FloatArray:
    """Convolve a Stark profile with one normalized tabulated kernel."""

    if (
        kernel_offset_angstrom.ndim != 1
        or kernel_profile_per_angstrom.shape != kernel_offset_angstrom.shape
        or kernel_offset_angstrom.size < 3
        or np.any(np.diff(kernel_offset_angstrom) <= 0.0)
        or np.any(kernel_profile_per_angstrom < 0.0)
    ):
        raise ValueError("invalid tabulated self-broadening kernel")
    integration_weight = np.empty_like(kernel_offset_angstrom)
    integration_weight[0] = 0.5 * (
        kernel_offset_angstrom[1] - kernel_offset_angstrom[0]
    )
    integration_weight[-1] = 0.5 * (
        kernel_offset_angstrom[-1] - kernel_offset_angstrom[-2]
    )
    integration_weight[1:-1] = 0.5 * (
        kernel_offset_angstrom[2:] - kernel_offset_angstrom[:-2]
    )
    integration_weight *= kernel_profile_per_angstrom
    normalization = float(np.sum(integration_weight))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("tabulated self-broadening kernel is not normalized")
    integration_weight /= normalization

    detuning = wavelength_angstrom - line_center_angstrom
    result = np.zeros_like(wavelength_angstrom)
    for start in range(
        0, kernel_offset_angstrom.size, _CAUCHY_QUADRATURE_BLOCK_SIZE
    ):
        stop = start + _CAUCHY_QUADRATURE_BLOCK_SIZE
        shift = kernel_offset_angstrom[start:stop, np.newaxis]
        weight = integration_weight[start:stop, np.newaxis]
        shifted_detuning = detuning[np.newaxis, :] - shift
        safe_center = max(
            line_center_angstrom,
            1.0 + float(np.max(np.abs(shifted_detuning))),
        )
        result += np.sum(
            weight
            * stark_line.wavelength_profile(
                safe_center + shifted_detuning,
                safe_center,
                temperature,
                electron_density,
            ),
            axis=0,
        )
    return result


def neutral_hydrogen_self_broadening_hwhm(
    atmosphere: Atmosphere,
    line: HydrogenLine,
    *,
    prescription: str = "barklem",
) -> FloatArray:
    """Return neutral-H impact-broadening HWHM in Angstrom.

    With the default ``"barklem"`` prescription, Halpha, Hbeta, and Hgamma
    use the Barklem et al. p--d self-broadening calculation and higher Balmer
    members use the older Ali--Griem resonance approximation.  The
    ``"allard-2008"`` control substitutes the thermally averaged unified-theory
    Halpha width from Allard et al. (2008), Table 2, over its published
    3000--12000 K range; Hbeta and Hgamma retain Barklem.  It changes the
    impact-core width only and is not a reconstruction of Allard's unavailable
    non-Lorentzian far-wing profile.  The explicit ``"ali-griem"`` control
    applies resonance broadening to every member; it matches the neutral-H
    treatment used in published cool-DA SYNSPEC NLTE calculations and is useful
    for separating line-profile and atmospheric-structure effects.
    """

    if prescription not in {"barklem", "allard-2008", "ali-griem"}:
        raise ValueError(
            "self-broadening prescription must be 'barklem', "
            "'allard-2008', or 'ali-griem'"
        )
    if (
        prescription == "allard-2008"
        and line.lower_level == 2
        and line.upper_level == 3
    ):
        angular_hwhm_per_perturber = np.interp(
            np.clip(
                atmosphere.temperature,
                _ALLARD_2008_HALPHA_TEMPERATURE_K[0],
                _ALLARD_2008_HALPHA_TEMPERATURE_K[-1],
            ),
            _ALLARD_2008_HALPHA_TEMPERATURE_K,
            _ALLARD_2008_HALPHA_ANGULAR_HWHM_PER_PERTURBER,
        )
        frequency_hwhm = (
            angular_hwhm_per_perturber
            * atmosphere.neutral_h_density
            / (2.0 * PI)
        )
        return (
            line.wavelength_vacuum_angstrom**2
            * 1.0e-8
            / LIGHT_SPEED
            * frequency_hwhm
        )
    broadening = (
        _BALMER_SELF_BROADENING.get(line.upper_level)
        if line.lower_level == 2 and prescription in {"barklem", "allard-2008"}
        else None
    )
    if broadening is None:
        level_distribution = _atmosphere_level_distribution(
            atmosphere,
            maximum_level=max(40, line.upper_level),
        )
        ground_population = level_distribution.population_density[:, 0]
        angular_damping = np.zeros(atmosphere.n_depth, dtype=np.float64)
        for level in (line.lower_level, line.upper_level):
            transition_frequency = LIGHT_SPEED / (
                1.0e8
                / (109_678.77 * (1.0 - 1.0 / level**2))
                * 1.0e-8
            )
            resonance_c3 = (
                _ALI_GRIEM_C3_COEFFICIENT
                * _lyman_oscillator_strength(level)
                / transition_frequency
            )
            angular_damping += (
                _ALI_GRIEM_DAMPING_COEFFICIENT
                / float(level)
                * ground_population
                * resonance_c3
            )
        frequency_hwhm = angular_damping / (4.0 * PI)
        return (
            line.wavelength_vacuum_angstrom**2
            * 1.0e-8
            / LIGHT_SPEED
            * frequency_hwhm
        )
    cross_section, velocity_exponent = broadening[:2]
    reference_velocity = 1.0e6  # cm s^-1
    mean_relative_velocity = np.sqrt(
        8.0
        * BOLTZMANN
        * atmosphere.temperature
        / (PI * (0.5 * HYDROGEN_MASS))
    )
    angular_hwhm_per_perturber = (
        cross_section
        * BOHR_RADIUS**2
        * reference_velocity
        * (4.0 / PI) ** (0.5 * velocity_exponent)
        * gamma_function(2.0 - 0.5 * velocity_exponent)
        * (mean_relative_velocity / reference_velocity)
        ** (1.0 - velocity_exponent)
    )
    frequency_hwhm = (
        angular_hwhm_per_perturber
        * atmosphere.neutral_h_density
        / (2.0 * PI)
    )
    return (
        line.wavelength_vacuum_angstrom**2
        * 1.0e-8
        / LIGHT_SPEED
        * frequency_hwhm
    )


def neutral_helium_balmer_broadening_hwhm(
    atmosphere: Atmosphere, line: HydrogenLine
) -> FloatArray:
    """Return classical neutral-He impact HWHM for a hydrogen line.

    The shell-averaged hydrogenic mean-square radii supply the Unsold/Warner
    van-der-Waals rate.  Its neutral-H result is rescaled by the measured He/H
    static-polarizability ratio and the radiator--perturber reduced mass.  This
    is an impact approximation, not a replacement for unavailable unified
    H--He profiles, but it includes the leading neutral-He width that is absent
    from a self-broadening-only DBA calculation.
    """

    helium = atmosphere.helium_lte_state
    if helium is None:
        return np.zeros(atmosphere.n_depth, dtype=np.float64)
    if line.lower_level < 1 or line.upper_level <= line.lower_level:
        raise ValueError("hydrogen line levels must be positive and increasing")

    def shell_mean_square_radius(principal_quantum_number: int) -> float:
        # Degeneracy-weighted average over l=0,...,n-1 of
        # <r^2>/a0^2 = n^2 [5 n^2 + 1 - 3 l(l+1)] / 2.
        n_squared = float(principal_quantum_number**2)
        return 1.75 * n_squared**2 + 1.25 * n_squared

    radius_difference = (
        shell_mean_square_radius(line.upper_level)
        - shell_mean_square_radius(line.lower_level)
    )
    hydrogen_rate = (
        10.0**-9.53
        * radius_difference**0.4
        * atmosphere.temperature**0.3
    )
    reduced_mass_h = 0.5 * HYDROGEN_MASS
    reduced_mass_he = HYDROGEN_MASS * HELIUM_MASS / (
        HYDROGEN_MASS + HELIUM_MASS
    )
    perturber_scale = (
        (_HELIUM_STATIC_POLARIZABILITY_A3 / _HYDROGEN_STATIC_POLARIZABILITY_A3)
        ** 0.4
        * (reduced_mass_h / reduced_mass_he) ** 0.3
    )
    collision_damping_rate = (
        hydrogen_rate * perturber_scale * helium.neutral_he_density
    )
    frequency_hwhm = collision_damping_rate / (4.0 * PI)
    return (
        line.wavelength_vacuum_angstrom**2
        * 1.0e-8
        / LIGHT_SPEED
        * frequency_hwhm
    )


def _hminus_cross_section_per_electron_pressure(
    wavelength_um: FloatArray, temperature: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return John (1988) H-minus bound-free and free-free coefficients.

    Both coefficients have units cm^4 dyne^-1.  Multiplication by electron
    pressure and the neutral-H number density gives an absorption coefficient
    in cm^-1.  The analytic fits are evaluated only over their stated short-
    wavelength limits; the long-wavelength free-free expression is retained.
    """

    wavelength_um, temperature = np.broadcast_arrays(wavelength_um, temperature)
    alpha = PLANCK * LIGHT_SPEED / BOLTZMANN * 1.0e4  # micron K
    threshold = 1.6419
    energy = np.maximum(1.0 / wavelength_um - 1.0 / threshold, 0.0)
    coefficients = np.array(
        [0.0, 152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
    )
    polynomial = np.zeros_like(energy)
    for index in range(1, 7):
        polynomial += coefficients[index] * energy ** ((index - 1.0) / 2.0)
    photodetachment_cross_section = (
        1.0e-18 * wavelength_um**3 * energy**1.5 * polynomial
    )
    bound_free = (
        0.750
        * temperature**-2.5
        * np.exp(alpha / (threshold * temperature))
        * -np.expm1(-alpha / (wavelength_um * temperature))
        * photodetachment_cross_section
    )
    bound_free = np.where(
        (wavelength_um > 0.125) & (wavelength_um <= threshold),
        bound_free,
        0.0,
    )

    long_coefficients = np.array(
        [
            [0, 0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830],
            [0, 0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170],
            [0, 0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8650],
            [0, 0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880],
            [0, 0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880],
            [0, 0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850],
        ],
        dtype=np.float64,
    )
    short_coefficients = np.array(
        [
            [0, 518.1021, 473.2636, -482.2089, 115.5291, 0, 0],
            [0, -734.8666, 1443.4137, -737.1616, 169.6374, 0, 0],
            [0, 1021.1775, -1977.3395, 1096.8827, -245.6490, 0, 0],
            [0, -479.0721, 922.3575, -521.1341, 114.2430, 0, 0],
            [0, 93.1373, -178.9275, 101.7963, -21.9972, 0, 0],
            [0, -6.4285, 12.3600, -7.0571, 1.5097, 0, 0],
        ],
        dtype=np.float64,
    )

    def evaluate_free_free(coefficient_table: FloatArray) -> FloatArray:
        powers = np.stack(
            (
                wavelength_um**2,
                np.ones_like(wavelength_um),
                wavelength_um**-1,
                wavelength_um**-2,
                wavelength_um**-3,
                wavelength_um**-4,
            )
        )
        result = np.zeros_like(wavelength_um)
        theta = 5040.0 / temperature
        for index in range(1, 7):
            result += theta ** ((index + 1.0) / 2.0) * np.sum(
                coefficient_table[:, index, np.newaxis] * powers.reshape(6, -1),
                axis=0,
            ).reshape(wavelength_um.shape)
        return 1.0e-29 * result

    free_free = np.where(
        wavelength_um > 0.3645,
        evaluate_free_free(long_coefficients),
        0.0,
    )
    free_free += np.where(
        (wavelength_um >= 0.1823) & (wavelength_um <= 0.3645),
        evaluate_free_free(short_coefficients),
        0.0,
    )
    return np.maximum(bound_free, 0.0), np.maximum(free_free, 0.0)


def hydrogen_ground_state_photoionization_cross_section(
    frequency_hz: ArrayLike,
) -> FloatArray:
    """Return the exact non-relativistic H(1s) cross section in cm^2.

    This is the Stobbe hydrogenic result, including its frequency-dependent
    bound-free Gaunt factor.  It approaches 6.30e-18 cm^2 at the Lyman edge;
    unlike the Kramers ``nu^-3`` approximation it remains accurate farther
    into the extreme ultraviolet.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must contain finite positive values")

    threshold_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    ratio = frequency / threshold_frequency
    cross_section = np.zeros_like(ratio)
    ionizing = ratio >= 1.0
    if not np.any(ionizing):
        return cross_section

    epsilon = np.sqrt(np.maximum(ratio[ionizing] - 1.0, 0.0))
    phase = np.ones_like(epsilon)
    nonzero = epsilon > 1.0e-8
    phase[nonzero] = np.arctan(epsilon[nonzero]) / epsilon[nonzero]
    denominator = np.ones_like(epsilon)
    denominator[nonzero] = -np.expm1(-2.0 * PI / epsilon[nonzero])
    threshold_cross_section = (
        2.0**9
        * PI**2
        / (3.0 * np.exp(4.0))
        * FINE_STRUCTURE_CONSTANT
        * BOHR_RADIUS**2
    )
    cross_section[ionizing] = (
        threshold_cross_section
        * ratio[ionizing] ** -4
        * np.exp(4.0 * (1.0 - phase))
        / denominator
    )
    return cross_section


def hydrogen_shell_photoionization_cross_section(
    level: int, frequency_hz: ArrayLike
) -> FloatArray:
    """Return the H I shell photoionization cross section in cm^2.

    ``level=1`` is the exact Stobbe ground-state result; excited shells use
    the Kramers normalization with the shell-averaged Mihalas/Karzas--Latter
    Gaunt factor. Threshold masking is left to the caller.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if level == 1:
        return hydrogen_ground_state_photoionization_cross_section(frequency)
    return (
        2.815e29
        * frequency**-3
        / level**5
        * hydrogen_bound_free_gaunt_factor(level, frequency)
    )


def hydrogen_free_free_extinction_per_cm(
    atmosphere: Atmosphere, wavelength_angstrom: ArrayLike
) -> FloatArray:
    """Return thermal electron--proton free--free extinction in cm^-1.

    The non-relativistic, thermally averaged Gaunt factor is interpolated
    from van Hoof et al. (2014) rather than fixed to unity. Stimulated
    emission is included.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    temperature = atmosphere.temperature[np.newaxis, :]
    stimulated_emission = -np.expm1(-PLANCK * frequency / (BOLTZMANN * temperature))
    return (
        3.692e8
        * temperature**-0.5
        * atmosphere.electron_density[np.newaxis, :]
        * atmosphere.proton_density[np.newaxis, :]
        * frequency**-3
        * stimulated_emission
        * hydrogen_free_free_gaunt_factor(wavelength_cm * 1.0e8, temperature)
    )


def hydrogen_continuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    maximum_level: int = 8,
    include_electron_scattering: bool = True,
    include_rayleigh_scattering: bool = True,
    include_molecular_absorption: bool = True,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
) -> FloatArray:
    """Return a first LTE pure-H continuum extinction in cm^2 g^-1.

    Included processes are exact Stobbe H I ground-state bound-free,
    hydrogenic excited-state bound-free with Mihalas/Karzas--Latter Gaunt
    factors, electron-proton free-free with van Hoof Gaunt factors,
    H-minus bound-free and free-free following John (1988), stable H2 and H2+
    ultraviolet photoabsorption when molecular EOS populations are present,
    optional Borysow H2--H2 collision-induced absorption, Thomson scattering,
    and ground-state H I Rayleigh scattering.
    Scattering is included as extinction when requested; the spectrum and
    atmosphere solvers evaluate it separately for a coherent source function.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")

    wavelength_um = wavelength[:, np.newaxis] * 1.0e-4
    temperature = atmosphere.temperature[np.newaxis, :]
    density = atmosphere.mass_density[np.newaxis, :]
    neutral_h = atmosphere.neutral_h_density[np.newaxis, :]
    electrons = atmosphere.electron_density[np.newaxis, :]

    free_free_per_cm = hydrogen_free_free_extinction_per_cm(atmosphere, wavelength)

    bound_free = hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        maximum_level=maximum_level,
    )

    hminus_bf, hminus_ff = _hminus_cross_section_per_electron_pressure(
        wavelength_um, temperature
    )
    electron_pressure = electrons * BOLTZMANN * temperature
    hminus_per_cm = (hminus_bf + hminus_ff) * electron_pressure * neutral_h
    extinction = (
        free_free_per_cm
        + hminus_per_cm
    ) / density
    extinction += bound_free
    if include_molecular_absorption:
        extinction += molecular_hydrogen_mass_absorption_coefficient(
            atmosphere, wavelength
        )
        if h2_h2_cia_table is not None:
            extinction += h2_h2_cia_mass_absorption_coefficient(
                atmosphere, wavelength, h2_h2_cia_table
            )
    if include_electron_scattering:
        extinction += electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
    if include_rayleigh_scattering:
        extinction += hydrogen_rayleigh_scattering_mass_coefficient(
            atmosphere, wavelength
        )
    return extinction


def hydrogen_bound_free_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    maximum_level: int = 8,
) -> FloatArray:
    """Return the ordinary LTE H I bound-free opacity in cm^2 g^-1.

    The ground state uses the exact Stobbe cross section. Excited states use
    shell-averaged Mihalas/Karzas--Latter Gaunt factors with the Kramers
    normalization.  Stimulated emission and the atmosphere's current
    Hummer--Mihalas level populations are included.  Keeping this term
    separately accessible lets magnetic transfer replace only the
    polarization-dependent photoionization opacity without double counting
    the otherwise validated continuum.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if maximum_level < 1:
        raise ValueError("maximum_level must be positive")

    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    temperature = atmosphere.temperature[np.newaxis, :]
    stimulated_emission = -np.expm1(
        -PLANCK * frequency / (BOLTZMANN * temperature)
    )
    bound_free_per_cm = np.zeros(
        (wavelength.size, atmosphere.n_depth), dtype=np.float64
    )
    lyman_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    level_distribution = _atmosphere_level_distribution(
        atmosphere,
        maximum_level=max(40, maximum_level),
    )
    for level in range(1, maximum_level + 1):
        threshold_frequency = lyman_frequency / level**2
        cross_section = hydrogen_shell_photoionization_cross_section(level, frequency)
        population = level_distribution.population_density[..., level - 1]
        bound_free_per_cm += np.where(
            frequency >= threshold_frequency,
            population * cross_section * stimulated_emission,
            0.0,
        )
    return bound_free_per_cm / atmosphere.mass_density[np.newaxis, :]


def molecular_hydrogen_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return stable H2 and H2+ ultraviolet absorption in cm^2 g^-1.

    The H2 cross section is the 1-nm-binned Leiden Lyman/Werner
    photoabsorption table, while H2+ uses the database's v=0
    photodissociation calculation.  These stable-species populations are
    distinct from the transient H--H+ collision complexes represented by the
    unified quasi-molecular Lyman profiles elsewhere in this module.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    state = atmosphere.hydrogen_lte_state
    if state is None or (
        state.molecular_hydrogen_density is None
        and state.molecular_hydrogen_ion_density is None
    ):
        return np.zeros(
            (wavelength.size, atmosphere.n_depth), dtype=np.float64
        )

    h2_density = (
        np.zeros(atmosphere.n_depth, dtype=np.float64)
        if state.molecular_hydrogen_density is None
        else state.molecular_hydrogen_density
    )
    h2plus_density = (
        np.zeros(atmosphere.n_depth, dtype=np.float64)
        if state.molecular_hydrogen_ion_density is None
        else state.molecular_hydrogen_ion_density
    )
    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    stimulated_emission = -np.expm1(
        -PLANCK
        * frequency
        / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
    )
    per_length = (
        molecular_hydrogen_photoabsorption_cross_section(wavelength)[
            :, np.newaxis
        ]
        * h2_density[np.newaxis, :]
        + molecular_hydrogen_ion_photoabsorption_cross_section(wavelength)[
            :, np.newaxis
        ]
        * h2plus_density[np.newaxis, :]
    )
    return (
        per_length
        * stimulated_emission
        / atmosphere.mass_density[np.newaxis, :]
    )


def h2_h2_cia_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    table: H2H2CollisionInducedAbsorptionTable,
) -> FloatArray:
    """Return H2--H2 collision-induced absorption in cm^2 g^-1.

    Borysow's tabulation gives the net binary absorption coefficient in
    cm^-1 amagat^-2, including its temperature dependence.  Multiplication by
    ``(n(H2) / n_amagat)^2`` converts it to inverse length; no separate
    stimulated-emission factor is required.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    state = atmosphere.hydrogen_lte_state
    if state is None or state.molecular_hydrogen_density is None:
        return np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)

    coefficient = table.coefficient_for_wavelength_temperature(
        wavelength, atmosphere.temperature
    )
    molecular_density_amagat = (
        state.molecular_hydrogen_density / LOSCHMIDT_NUMBER_DENSITY
    )
    return (
        coefficient
        * molecular_density_amagat[np.newaxis, :] ** 2
        / atmosphere.mass_density[np.newaxis, :]
    )


def hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lower_levels: tuple[int, ...] = (1, 2, 3, 4),
) -> FloatArray:
    """Return series-wide DAM/HM dissolved-level opacity in cm^2 g^-1.

    For each populated lower level ``i``, a photon redward of its series limit
    is mapped to the fictitious upper level ``n*`` of Daeppen, Anderson &
    Mihalas (1987).  The full dissolved fraction
    ``1 - w(n*) / w(i)`` is retained through the transition to ``i + 3``, the
    default pseudo-continuum cutoff used by SYNSPEC for explicit hydrogen
    levels.  Between that cutoff and the first series member, only the
    neutral-perturber part of the dissolved fraction is retained.  This
    separates the long neutral-H tail, which is absent from the charged-only
    Tremblay--Bergeron Stark profiles, from charged dissolution that would
    otherwise become an artificial background beneath the first two members
    of a series.  Lyman, Balmer, Paschen, and Brackett series are included by
    default.  Charged and neutral perturbers use the same Q-MHD/HM occupation
    probabilities as the EOS.  The Lyman
    redistribution is cut off at 925 A by default: the first three Lyman
    profiles are supplied as ideal profiles, and extrapolating the raw DAM
    term across the whole Ly-alpha interval transfers far too much strength.

    The hydrogenic cross section is continued from the ordinary threshold of
    each lower level only across these finite intervals.  This deliberately
    avoids the unphysical indefinite optical/infrared extension of the raw
    DAM prescription discussed by Kowalski (2006).
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")

    if not lower_levels or any(level < 1 for level in lower_levels):
        raise ValueError("lower_levels must contain positive integers")
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    distribution = _atmosphere_level_distribution(
        atmosphere, max(40, max(lower_levels))
    )
    lyman_limit = (
        PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
    )
    lyman_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    for lower_level in lower_levels:
        series_limit = lyman_limit * lower_level**2
        charged_red_limit = series_limit / (
            1.0 - (lower_level / (lower_level + 3.0)) ** 2
        )
        first_line = series_limit / (
            1.0 - (lower_level / (lower_level + 1.0)) ** 2
        )
        red_limit = first_line
        if lower_level == 1:
            red_limit = min(red_limit, 925.0)
        valid = (wavelength > series_limit) & (wavelength < red_limit)
        if not np.any(valid):
            continue

        selected = wavelength[valid]
        effective_level = lower_level / np.sqrt(
            1.0 - series_limit / selected
        )
        upper_probability = hydrogen_occupation_probability(
            atmosphere.neutral_h_density[np.newaxis, :],
            atmosphere.electron_density[np.newaxis, :],
            atmosphere.temperature[np.newaxis, :],
            effective_level[:, np.newaxis],
            neutral_he_density=(
                atmosphere.helium_lte_state.neutral_he_density[np.newaxis, :]
                if atmosphere.helium_lte_state is not None
                else 0.0
            ),
            neutral_radius_scale=(
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None
                else HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            ),
            helium_neutral_radius_scale=(
                atmosphere.helium_lte_state.neutral_radius_scale
                if atmosphere.helium_lte_state is not None
                else 0.5
            ),
            correlated_microfields=(
                atmosphere.hydrogen_lte_state is not None
                and atmosphere.hydrogen_lte_state.microfield_model == "qmhd"
            ),
        )
        lower_probability = distribution.occupation_probability[
            :, lower_level - 1
        ]
        total_survival = np.clip(
            upper_probability / lower_probability[np.newaxis, :],
            0.0,
            1.0,
        )
        correlated_microfields = (
            atmosphere.hydrogen_lte_state is not None
            and atmosphere.hydrogen_lte_state.microfield_model == "qmhd"
        )
        upper_charged_probability = (
            charged_particle_hydrogen_occupation_probability(
                atmosphere.electron_density[np.newaxis, :],
                effective_level[:, np.newaxis],
                (
                    atmosphere.temperature[np.newaxis, :]
                    if correlated_microfields
                    else None
                ),
            )
        )
        lower_charged_probability = (
            charged_particle_hydrogen_occupation_probability(
                atmosphere.electron_density,
                float(lower_level),
                atmosphere.temperature if correlated_microfields else None,
            )
        )
        charged_survival = np.clip(
            upper_charged_probability
            / lower_charged_probability[np.newaxis, :],
            0.0,
            1.0,
        )
        full_dissolved_fraction = 1.0 - total_survival
        neutral_dissolved_fraction = np.clip(
            charged_survival - total_survival,
            0.0,
            1.0,
        )
        dissolved_fraction = np.where(
            (selected < charged_red_limit)[:, np.newaxis],
            full_dissolved_fraction,
            neutral_dissolved_fraction,
        )
        frequency = LIGHT_SPEED / (selected[:, np.newaxis] * 1.0e-8)
        stimulated_emission = -np.expm1(
            -PLANCK
            * frequency
            / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
        )
        threshold_frequency = lyman_frequency / lower_level**2
        if lower_level == 1:
            threshold_cross_section = float(
                hydrogen_ground_state_photoionization_cross_section(
                    threshold_frequency
                )
            )
        else:
            threshold_cross_section = float(
                2.815e29
                * threshold_frequency**-3
                / lower_level**5
                * hydrogen_bound_free_gaunt_factor(
                    lower_level, threshold_frequency
                )
            )
        extrapolated_cross_section = (
            threshold_cross_section * (selected / series_limit) ** 3
        )
        lower_population = distribution.population_density[
            :, lower_level - 1
        ]
        opacity[valid] += (
            lower_population[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
            * extrapolated_cross_section[:, np.newaxis]
            * dissolved_fraction
            * stimulated_emission
        )
    return opacity


def electron_scattering_mass_coefficient(atmosphere: Atmosphere) -> FloatArray:
    """Return Thomson scattering extinction in cm^2 g^-1."""

    return (
        THOMSON_CROSS_SECTION
        * atmosphere.electron_density
        / atmosphere.mass_density
    )


def hydrogen_rayleigh_scattering_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return the ground-state H I Rayleigh cross section in cm^2.

    The analytic dynamic-polarizability fit is Eq. 30 of Rohrmann & Vera
    Rueda (2022, A&A 667, A3).  This branch is valid redward of Lyalpha.  Its
    stated energy boundary lies about 0.65 Angstrom redward of the line center,
    where a literal hard switch would turn on the divergent isolated-atom
    resonance wing abruptly.  That resonance is already represented by the
    explicit Stark/unified Lyman opacity, so a smooth detuning switch assigns
    the inner 10 Angstrom to the line profile and prevents double counting.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must contain finite positive values")

    # Photon energy in Rydbergs. The ionization energy used here sets the
    # vacuum Lyman-limit wavelength consistently with the rest of the code.
    rydberg_wavelength = (
        PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
    )
    energy = rydberg_wavelength / wavelength
    valid = energy < 0.7496

    correction = np.zeros_like(energy)
    first = valid & (energy <= 0.48083)
    correction[first] = (
        0.0017 * np.sin(8.2 * energy[first] ** 1.33) - 0.000093
    )
    second = valid & (energy > 0.48083) & (energy < 0.73)
    correction[second] = -0.00163 * np.sin(
        16.86 * np.abs(energy[second] - 0.48083) ** 1.2
    )
    fourth = valid & (energy >= 0.745)
    correction[fourth] = -10.0 ** (
        -4.9 + 0.205 * (0.7501 - energy[fourth]) ** -0.3
    )

    polarizability = np.zeros_like(energy)
    polarizability[valid] = (
        1.46486 / (0.950713 - energy[valid] ** 2.172)
        + 1.66478 / (0.75**2 - energy[valid] ** 2)
    ) / (1.0 - correction[valid])
    cross_section_bohr2 = (
        PI
        / 6.0
        * FINE_STRUCTURE_CONSTANT**4
        * energy**4
        * polarizability**2
    )
    cross_section = cross_section_bohr2 * BOHR_RADIUS**2

    # The preresonance polarizability fit is an isolated-atom expression and
    # therefore diverges toward Ly-alpha.  A hard application of its formal
    # E < 0.7496 range used to create an artificial opacity edge at 1216.32 A.
    # Hand the resonance smoothly to the explicit pressure-broadened Lyman
    # line.  The fourth-power switch strongly suppresses the singular term at
    # the fit boundary, while becoming effectively unity by 20 A detuning.
    red_detuning = np.maximum(
        wavelength - LYMAN_LINES[0].wavelength_vacuum_angstrom,
        0.0,
    )
    line_to_continuum_switch = 1.0 - np.exp(
        -(red_detuning / 10.0) ** 4
    )
    return cross_section * line_to_continuum_switch


def molecular_hydrogen_rayleigh_scattering_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return the ground-state H2 Rayleigh cross section in cm^2.

    This is the long-wavelength expansion of Dalgarno & Williams (1962),
    evaluated with wavelength in cm.  Their calculation applies at and
    redward of Lyalpha; shorter wavelengths are left to the explicit H2
    electronic photoabsorption data.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must contain finite positive values")
    wavelength_cm = wavelength * 1.0e-8
    valid = wavelength >= LYMAN_LINES[0].wavelength_vacuum_angstrom
    cross_section = np.zeros_like(wavelength)
    cross_section[valid] = (
        8.14e-45 / wavelength_cm[valid] ** 4
        + 1.28e-54 / wavelength_cm[valid] ** 6
        + 1.61e-64 / wavelength_cm[valid] ** 8
    )
    return cross_section


def hydrogen_rayleigh_scattering_mass_coefficient(
    atmosphere: Atmosphere, wavelength_angstrom: ArrayLike
) -> FloatArray:
    """Return coherent H I plus H2 Rayleigh opacity in cm^2 g^-1."""

    cross_section = hydrogen_rayleigh_scattering_cross_section(
        wavelength_angstrom
    )
    if cross_section.ndim != 1:
        raise ValueError("wavelength_angstrom must be a 1D array")
    ground_population = _atmosphere_level_distribution(
        atmosphere, 40
    ).population_density[:, 0]
    opacity = (
        cross_section[:, np.newaxis]
        * ground_population[np.newaxis, :]
        / atmosphere.mass_density[np.newaxis, :]
    )
    state = atmosphere.hydrogen_lte_state
    if state is not None and state.molecular_hydrogen_density is not None:
        molecular_cross_section = (
            molecular_hydrogen_rayleigh_scattering_cross_section(
                wavelength_angstrom
            )
        )
        opacity += (
            molecular_cross_section[:, np.newaxis]
            * state.molecular_hydrogen_density[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
        )
    return opacity


def rosseland_mean_hydrogen_continuum_opacity(
    atmosphere: Atmosphere,
    *,
    n_frequency: int = 240,
    wavelength_angstrom: ArrayLike | None = None,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
) -> FloatArray:
    """Return the Rosseland mean of the implemented hydrogen continuum."""

    if n_frequency < 40:
        raise ValueError("n_frequency must be at least 40")
    if wavelength_angstrom is not None:
        from ._rosseland import rosseland_mean_from_opacity_grid

        return rosseland_mean_from_opacity_grid(
            wavelength_angstrom,
            hydrogen_continuum_mass_absorption_coefficient(
                atmosphere, wavelength_angstrom, h2_h2_cia_table=h2_h2_cia_table,
            ),
            atmosphere.temperature,
        )
    result = np.empty(atmosphere.n_depth, dtype=np.float64)
    dimensionless_frequency = np.geomspace(0.1, 30.0, n_frequency)
    exponential = np.exp(dimensionless_frequency)
    weight = (
        dimensionless_frequency**4
        * exponential
        / np.expm1(dimensionless_frequency) ** 2
    )
    for depth in range(atmosphere.n_depth):
        wavelength = (
            PLANCK
            * LIGHT_SPEED
            / (BOLTZMANN * atmosphere.temperature[depth] * dimensionless_frequency)
            * 1.0e8
        )
        order = np.argsort(wavelength)
        point = Atmosphere(
            effective_temperature=atmosphere.effective_temperature,
            logg=atmosphere.logg,
            rosseland_optical_depth=atmosphere.rosseland_optical_depth[depth:depth + 1],
            column_mass=atmosphere.column_mass[depth:depth + 1],
            temperature=atmosphere.temperature[depth:depth + 1],
            gas_pressure=atmosphere.gas_pressure[depth:depth + 1],
            mass_density=atmosphere.mass_density[depth:depth + 1],
            neutral_h_density=atmosphere.neutral_h_density[depth:depth + 1],
            proton_density=atmosphere.proton_density[depth:depth + 1],
            electron_density=atmosphere.electron_density[depth:depth + 1],
            metadata=atmosphere.metadata,
            hydrogen_lte_state=_slice_hydrogen_lte_state(
                atmosphere.hydrogen_lte_state, depth
            ),
        )
        opacity = hydrogen_continuum_mass_absorption_coefficient(
            point,
            wavelength[order],
            h2_h2_cia_table=h2_h2_cia_table,
        )[:, 0]
        inverse_mean = trapezoid(
            weight[order] / np.maximum(opacity, 1.0e-30),
            dimensionless_frequency[order],
        ) / trapezoid(weight[order], dimensionless_frequency[order])
        result[depth] = 1.0 / inverse_mean
    return result


def hydrogen_level_population(
    neutral_h_density: ArrayLike,
    temperature: ArrayLike,
    principal_quantum_number: int,
    *,
    electron_density: ArrayLike | None = None,
) -> FloatArray:
    """LTE population of an H I level.

    When ``electron_density`` is supplied, the result is normalized with the
    same Hummer--Mihalas internal partition function used by the atmosphere
    EOS.  Omitting it retains the ideal ground-state-relative calculation for
    backward compatibility and controlled tests.
    """

    if principal_quantum_number < 1:
        raise ValueError("principal_quantum_number must be positive")
    neutral_h_density, temperature = np.broadcast_arrays(
        np.asarray(neutral_h_density, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if np.any(neutral_h_density < 0.0) or np.any(temperature <= 0.0):
        raise ValueError("density and temperature must be physical")
    if electron_density is not None:
        neutral_h_density, temperature, electron_density = np.broadcast_arrays(
            neutral_h_density,
            temperature,
            np.asarray(electron_density, dtype=np.float64),
        )
        if np.any(electron_density < 0.0):
            raise ValueError("electron density must be physical")
        distribution = hydrogen_level_distribution(
            neutral_h_density,
            electron_density,
            temperature,
            maximum_level=max(40, principal_quantum_number),
        )
        return distribution.population_density[..., principal_quantum_number - 1]
    n = float(principal_quantum_number)
    degeneracy_ratio = n * n
    excitation_energy = HYDROGEN_IONIZATION_ENERGY * (1.0 - 1.0 / n**2)
    return neutral_h_density * degeneracy_ratio * np.exp(
        -excitation_energy / (BOLTZMANN * temperature)
    )


def balmer_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: HydrogenStarkTable | None = None,
    lines: tuple[HydrogenLine, ...] = BALMER_LINES,
    include_self_broadening: bool = True,
    include_neutral_helium_broadening: bool = True,
    self_broadening_quadrature_order: int = 128,
    self_broadening_impact_validity_fraction: float | None = 1.0,
    self_broadening_prescription: str = "barklem",
    self_broadening_truncation_closure: str = "renormalize",
    barklem_self_table: BarklemSelfBroadeningTable | None = None,
    profile_edge_optical_depth: float | None = None,
    profile_support_maximum_rosseland_optical_depth: float = 2.0,
    profile_support_maximum_half_window_angstrom: float = 5000.0,
) -> FloatArray:
    """Return LTE Balmer true-absorption opacity in cm^2 g^-1.

    Halpha through H22 use the occupation-probability lower-level population
    and the bound-bound survival probability ``w_upper / w_lower``.  The
    bundled Tremblay-Bergeron Stark profiles already include their published
    non-ideal profile-shape corrections.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a positive 1D array")
    if self_broadening_quadrature_order < 8:
        raise ValueError("self_broadening_quadrature_order must be at least 8")
    if (
        self_broadening_impact_validity_fraction is not None
        and not 0.0 < self_broadening_impact_validity_fraction <= 1.0
    ):
        raise ValueError(
            "self_broadening_impact_validity_fraction must be None or in (0, 1]"
        )
    if self_broadening_truncation_closure not in {
        "renormalize",
        "stark-core",
    }:
        raise ValueError(
            "self_broadening_truncation_closure must be 'renormalize' "
            "or 'stark-core'"
        )
    if self_broadening_prescription not in {
        "barklem",
        "barklem-grid",
        "allard-2008",
        "ali-griem",
    }:
        raise ValueError(
            "self_broadening_prescription must be 'barklem', "
            "'barklem-grid', 'allard-2008', or 'ali-griem'"
        )
    if self_broadening_prescription == "barklem-grid" and barklem_self_table is None:
        raise ValueError("barklem-grid requires barklem_self_table")
    if profile_edge_optical_depth is not None and (
        not np.isfinite(profile_edge_optical_depth)
        or profile_edge_optical_depth <= 0.0
    ):
        raise ValueError("profile_edge_optical_depth must be positive")
    if (
        not np.isfinite(profile_support_maximum_rosseland_optical_depth)
        or profile_support_maximum_rosseland_optical_depth <= 0.0
        or not np.isfinite(profile_support_maximum_half_window_angstrom)
        or profile_support_maximum_half_window_angstrom <= 0.0
    ):
        raise ValueError("profile-support limits must be positive")
    table = default_balmer_stark_table() if stark_table is None else stark_table
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    level_distribution = _atmosphere_level_distribution(
        atmosphere,
        maximum_level=max(40, max(line.upper_level for line in lines)),
    )
    lower_population = level_distribution.population_density[:, 1]
    lower_probability = level_distribution.occupation_probability[:, 1]
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )

    for line in lines:
        table_line = table[(line.lower_level, line.upper_level)]
        bound_bound_survival = np.clip(
            level_distribution.occupation_probability[:, line.upper_level - 1]
            / lower_probability,
            0.0,
            1.0,
        )
        hwhm_prescription = (
            "barklem"
            if self_broadening_prescription == "barklem-grid"
            else self_broadening_prescription
        )
        self_broadening_hwhm = (
            neutral_hydrogen_self_broadening_hwhm(
                atmosphere,
                line,
                prescription=hwhm_prescription,
            )
            if include_self_broadening
            else np.zeros(atmosphere.n_depth, dtype=np.float64)
        )
        neutral_helium_hwhm = (
            neutral_helium_balmer_broadening_hwhm(atmosphere, line)
            if include_neutral_helium_broadening
            else np.zeros(atmosphere.n_depth, dtype=np.float64)
        )
        neutral_impact_hwhm = self_broadening_hwhm + neutral_helium_hwhm
        broadening_characteristics = (
            _BALMER_SELF_BROADENING.get(line.upper_level)
            if self_broadening_prescription in {
                "barklem",
                "barklem-grid",
                "allard-2008",
            }
            else None
        )
        maximum_impact_shift = None
        if (
            broadening_characteristics is not None
            and len(broadening_characteristics) > 2
            and self_broadening_impact_validity_fraction is not None
        ):
            # Barklem et al. (2000), Table 3 gives the quoted validity limit
            # at v=14 km/s.  Their Eq. 18 and sigma(v) proportional to v^-alpha
            # imply Delta-lambda_max proportional to v^(1+alpha/2).  Applying
            # one fixed wavelength cutoff at every depth was both physically
            # inconsistent and made its numerical shoulder especially sharp.
            velocity_exponent = broadening_characteristics[1]
            impact_collision_velocity = np.sqrt(
                8.0
                * BOLTZMANN
                * atmosphere.temperature
                / (PI * (0.5 * HYDROGEN_MASS))
            )
            maximum_impact_shift = (
                self_broadening_impact_validity_fraction
                * broadening_characteristics[2]
                * (impact_collision_velocity / 1.4e6)
                ** (1.0 + 0.5 * velocity_exponent)
            )
        line_frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
        stimulated_emission = 1.0 - np.exp(
            -PLANCK * line_frequency / (BOLTZMANN * atmosphere.temperature)
        )
        line_start = 0
        line_stop = wavelength.size
        if profile_edge_optical_depth is not None:
            # Opacity-sampling grids contain many continuum and metal points
            # far from every Balmer member.  The final formal spectrum keeps
            # the complete tabulated profile, but a structure iteration need
            # not repeat the expensive Stark-plus-neutral convolution where
            # the line's vertical optical depth is demonstrably negligible.
            # Expanding to an optical-depth threshold makes the support adapt
            # to abundance, gravity, temperature, and line strength instead
            # of imposing a spectral-type or fixed-wavelength cutoff.
            relevant = np.flatnonzero(
                atmosphere.rosseland_optical_depth
                <= profile_support_maximum_rosseland_optical_depth
            )
            relevant_stop = int(relevant[-1]) + 1 if relevant.size else 1
            local_mass = atmosphere.column_mass[:relevant_stop]
            line_strength_per_mass = (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * lower_population[:relevant_stop]
                * bound_bound_survival[:relevant_stop]
                * stimulated_emission[:relevant_stop]
                / atmosphere.mass_density[:relevant_stop]
            )
            half_window = min(
                5.0, profile_support_maximum_half_window_angstrom
            )
            while True:
                edge_wavelength = np.asarray(
                    (
                        max(
                            np.nextafter(0.0, 1.0),
                            line.wavelength_vacuum_angstrom - half_window,
                        ),
                        line.wavelength_vacuum_angstrom + half_window,
                    ),
                    dtype=np.float64,
                )
                edge_opacity = np.empty(
                    (2, relevant_stop), dtype=np.float64
                )
                for support_depth in range(relevant_stop):
                    support_profile = _lorentz_convolved_stark_profile(
                        table_line,
                        edge_wavelength,
                        line.wavelength_vacuum_angstrom,
                        float(atmosphere.temperature[support_depth]),
                        float(atmosphere.electron_density[support_depth]),
                        float(neutral_impact_hwhm[support_depth]),
                        (
                            float(maximum_impact_shift[support_depth])
                            if maximum_impact_shift is not None
                            else None
                        ),
                        self_broadening_quadrature_order,
                        self_broadening_truncation_closure,
                    )
                    edge_cm = edge_wavelength * 1.0e-8
                    edge_opacity[:, support_depth] = (
                        line_strength_per_mass[support_depth]
                        * support_profile
                        * 1.0e8
                        * edge_cm**2
                        / LIGHT_SPEED
                    )
                edge_tau = edge_opacity[:, 0] * local_mass[0]
                if relevant_stop > 1:
                    edge_tau += np.sum(
                        0.5
                        * (edge_opacity[:, 1:] + edge_opacity[:, :-1])
                        * np.diff(local_mass)[np.newaxis, :],
                        axis=1,
                    )
                if (
                    float(np.max(edge_tau)) <= profile_edge_optical_depth
                    or half_window
                    >= profile_support_maximum_half_window_angstrom
                ):
                    break
                half_window = min(
                    2.0 * half_window,
                    profile_support_maximum_half_window_angstrom,
                )
            line_start = int(np.searchsorted(
                wavelength,
                line.wavelength_vacuum_angstrom - half_window,
            ))
            line_stop = int(np.searchsorted(
                wavelength,
                line.wavelength_vacuum_angstrom + half_window,
                side="right",
            ))
            if line_stop <= line_start:
                continue
        line_wavelength = wavelength[line_start:line_stop]
        line_wavelength_cm = line_wavelength * 1.0e-8

        def profile_at_depth(depth: int) -> FloatArray:
            use_barklem_grid = bool(
                include_self_broadening
                and self_broadening_prescription == "barklem-grid"
                and barklem_self_table is not None
                and neutral_helium_hwhm[depth] <= 1.0e-5
                and barklem_self_table.contains_state(
                    (line.lower_level, line.upper_level),
                    float(atmosphere.temperature[depth]),
                    float(atmosphere.neutral_h_density[depth]),
                )
            )
            if use_barklem_grid:
                assert barklem_self_table is not None
                kernel = barklem_self_table.profile_at_state(
                    (line.lower_level, line.upper_level),
                    float(atmosphere.temperature[depth]),
                    float(atmosphere.neutral_h_density[depth]),
                )
                return _tabulated_convolved_stark_profile(
                    table_line,
                    line_wavelength,
                    line.wavelength_vacuum_angstrom,
                    float(atmosphere.temperature[depth]),
                    float(atmosphere.electron_density[depth]),
                    barklem_self_table.wavelength_offset_angstrom,
                    kernel,
                )
            return _lorentz_convolved_stark_profile(
                table_line,
                line_wavelength,
                line.wavelength_vacuum_angstrom,
                float(atmosphere.temperature[depth]),
                float(atmosphere.electron_density[depth]),
                float(neutral_impact_hwhm[depth]),
                (
                    float(maximum_impact_shift[depth])
                    if maximum_impact_shift is not None
                    else None
                ),
                self_broadening_quadrature_order,
                self_broadening_truncation_closure,
            )

        workers = _hydrogen_profile_worker_count(atmosphere.n_depth)
        use_parallel_profiles = bool(
            workers > 1
            and line_wavelength.size >= 256
            and _rt is not None
            and hasattr(_rt, "hydrogen_stark_lorentz_convolution")
            and np.any(neutral_impact_hwhm > 1.0e-5)
        )
        if use_parallel_profiles:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                depth_profiles = executor.map(
                    profile_at_depth, range(atmosphere.n_depth)
                )
                depth_profile_pairs = enumerate(depth_profiles)
                for depth, profile_per_angstrom in depth_profile_pairs:
                    profile_per_hz = (
                        profile_per_angstrom
                        * 1.0e8
                        * line_wavelength_cm**2
                        / LIGHT_SPEED
                    )
                    absorption_per_cm = (
                        integrated_cross_section
                        * line.absorption_oscillator_strength
                        * lower_population[depth]
                        * bound_bound_survival[depth]
                        * stimulated_emission[depth]
                        * profile_per_hz
                    )
                    opacity[line_start:line_stop, depth] += (
                        absorption_per_cm / atmosphere.mass_density[depth]
                    )
            continue

        for depth in range(atmosphere.n_depth):
            profile_per_angstrom = profile_at_depth(depth)
            profile_per_hz = (
                profile_per_angstrom
                * 1.0e8
                * line_wavelength_cm**2
                / LIGHT_SPEED
            )
            absorption_per_cm = (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * lower_population[depth]
                * bound_bound_survival[depth]
                * stimulated_emission[depth]
                * profile_per_hz
            )
            opacity[line_start:line_stop, depth] += (
                absorption_per_cm / atmosphere.mass_density[depth]
            )

    return opacity


def _higher_hydrogen_series_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: HydrogenStarkTable,
    lines: tuple[HydrogenLine, ...],
    wavelength_limits_angstrom: tuple[float, float],
    edge_taper_width_angstrom: tuple[float, float],
) -> FloatArray:
    """Return a Doppler-convolved non-ideal Stark series opacity.

    The 2015 Tremblay--Bergeron Paschen and Brackett release contains the
    Stark-plus-Doppler profile but no dedicated neutral-H self-broadening
    calculation.  Keep that distinction explicit rather than applying the
    lower-level-specific Barklem Balmer data to an infrared transition.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a positive 1D array")
    if not lines:
        return np.zeros(
            (wavelength.size, atmosphere.n_depth), dtype=np.float64
        )
    lower_level = lines[0].lower_level
    if any(line.lower_level != lower_level for line in lines):
        raise ValueError("all hydrogen-series lines must share a lower level")

    lower_limit, upper_limit = wavelength_limits_angstrom
    selected = (wavelength >= lower_limit) & (wavelength <= upper_limit)
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    if not np.any(selected):
        return opacity

    selected_wavelength = wavelength[selected]
    selected_wavelength_cm = selected_wavelength * 1.0e-8
    selected_opacity = np.zeros(
        (selected_wavelength.size, atmosphere.n_depth), dtype=np.float64
    )
    level_distribution = _atmosphere_level_distribution(
        atmosphere,
        maximum_level=max(40, max(line.upper_level for line in lines)),
    )
    lower_population = level_distribution.population_density[
        :, lower_level - 1
    ]
    lower_probability = level_distribution.occupation_probability[
        :, lower_level - 1
    ]
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )

    for line in lines:
        table_line = stark_table[(line.lower_level, line.upper_level)]
        bound_bound_survival = np.clip(
            level_distribution.occupation_probability[
                :, line.upper_level - 1
            ]
            / lower_probability,
            0.0,
            1.0,
        )
        line_frequency = LIGHT_SPEED / (
            line.wavelength_vacuum_angstrom * 1.0e-8
        )
        stimulated_emission = 1.0 - np.exp(
            -PLANCK
            * line_frequency
            / (BOLTZMANN * atmosphere.temperature)
        )
        for depth in range(atmosphere.n_depth):
            profile_per_angstrom = table_line.wavelength_profile(
                selected_wavelength,
                line.wavelength_vacuum_angstrom,
                float(atmosphere.temperature[depth]),
                float(atmosphere.electron_density[depth]),
            )
            profile_per_hz = (
                profile_per_angstrom
                * 1.0e8
                * selected_wavelength_cm**2
                / LIGHT_SPEED
            )
            absorption_per_cm = (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * lower_population[depth]
                * bound_bound_survival[depth]
                * stimulated_emission[depth]
                * profile_per_hz
            )
            selected_opacity[:, depth] += (
                absorption_per_cm / atmosphere.mass_density[depth]
            )
    lower_width, upper_width = edge_taper_width_angstrom
    lower_phase = np.clip(
        (selected_wavelength - lower_limit) / lower_width, 0.0, 1.0
    )
    upper_phase = np.clip(
        (upper_limit - selected_wavelength) / upper_width, 0.0, 1.0
    )
    edge_taper = (
        np.sin(0.5 * PI * lower_phase) ** 2
        * np.sin(0.5 * PI * upper_phase) ** 2
    )
    opacity[selected] = selected_opacity * edge_taper[:, np.newaxis]
    return opacity


def paschen_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: HydrogenStarkTable | None = None,
    lines: tuple[HydrogenLine, ...] = PASCHEN_LINES,
) -> FloatArray:
    """Return LTE Paschen-series Stark opacity in cm^2 g^-1."""

    from .stark import default_paschen_stark_table

    table = default_paschen_stark_table() if stark_table is None else stark_table
    return _higher_hydrogen_series_mass_absorption_coefficient(
        atmosphere,
        wavelength_angstrom,
        stark_table=table,
        lines=lines,
        wavelength_limits_angstrom=(7_500.0, 25_000.0),
        edge_taper_width_angstrom=(1_000.0, 2_000.0),
    )


def brackett_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: HydrogenStarkTable | None = None,
    lines: tuple[HydrogenLine, ...] = BRACKETT_LINES,
) -> FloatArray:
    """Return LTE Brackett-series Stark opacity in cm^2 g^-1."""

    from .stark import default_brackett_stark_table

    table = (
        default_brackett_stark_table() if stark_table is None else stark_table
    )
    return _higher_hydrogen_series_mass_absorption_coefficient(
        atmosphere,
        wavelength_angstrom,
        stark_table=table,
        lines=lines,
        wavelength_limits_angstrom=(13_500.0, 50_000.0),
        edge_taper_width_angstrom=(2_000.0, 5_000.0),
    )


def lyman_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: HydrogenStarkTable | None = None,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    jackson_lyman_table: JacksonLymanProfileTable | None = None,
    jackson_include_doppler: bool = True,
    allard_stark_weight: float = 0.5,
    lines: tuple[HydrogenLine, ...] = LYMAN_LINES,
) -> FloatArray:
    """Return LTE Lyman-series true-absorption opacity in cm^2 g^-1.

    Lyalpha through the 1->21 transition normally use the bundled
    Doppler-convolved Tremblay-Bergeron profiles.  A supplied Jackson table
    replaces the complete charged-particle profile of Lyalpha and Lybeta with
    the Xenomorph calculation, including electron Stark, ion dynamics,
    quasi-H2+ satellites, and (by default) its delivered Doppler convolution.
    It is not added to the ordinary Stark line because that would double-count
    both electron and proton broadening.  For transitions absent from the
    Jackson delivery, a supplied temperature-dependent Allard table retains
    the existing half-Stark additive approximation; otherwise the ordinary
    Tremblay-Bergeron profile is used.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a positive 1D array")
    if (
        not np.isfinite(allard_stark_weight)
        or not 0.0 <= allard_stark_weight <= 1.0
    ):
        raise ValueError("allard_stark_weight must lie in [0, 1]")
    table = default_lyman_stark_table() if stark_table is None else stark_table
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    level_distribution = _atmosphere_level_distribution(
        atmosphere,
        maximum_level=max(40, max(line.upper_level for line in lines)),
    )
    lower_population = level_distribution.population_density[:, 0]
    lower_probability = level_distribution.occupation_probability[:, 0]
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    wavelength_cm = wavelength * 1.0e-8

    for line in lines:
        table_line = table[(line.lower_level, line.upper_level)]
        # The distributed Tremblay--Bergeron table deliberately leaves
        # non-ideal dissolution out of Lyalpha, Lybeta, and Lygamma.  Its
        # README instructs atmosphere codes to use optical occupation
        # probability one for those transitions.  Applying w_upper/w_lower
        # here would contradict that convention.  Higher Lyman members keep
        # the EOS-consistent survival factor used by this opacity interface.
        if line.upper_level <= 4:
            bound_bound_survival = np.ones(atmosphere.n_depth)
        else:
            bound_bound_survival = np.clip(
                level_distribution.occupation_probability[
                    :, line.upper_level - 1
                ]
                / lower_probability,
                0.0,
                1.0,
            )
        line_frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
        stimulated_emission = 1.0 - np.exp(
            -PLANCK * line_frequency / (BOLTZMANN * atmosphere.temperature)
        )
        transition = (line.lower_level, line.upper_level)
        jackson_line = (
            jackson_lyman_table[transition]
            if (
                jackson_lyman_table is not None
                and transition in jackson_lyman_table.lines
            )
            else None
        )
        if jackson_line is not None:
            profile_per_hz = jackson_line.profile_per_hz(
                wavelength,
                atmosphere.temperature,
                atmosphere.electron_density,
                lyman_alpha_wavelength_angstrom=(
                    LYMAN_LINES[0].wavelength_vacuum_angstrom
                ),
                include_doppler=jackson_include_doppler,
            )
            # White et al.'s absorption profile omits the trivial omega/omega_ij
            # factor.  In wavelength coordinates that ratio is lambda_ij/lambda.
            frequency_ratio = line.wavelength_vacuum_angstrom / wavelength
            opacity += (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * profile_per_hz
                * frequency_ratio[:, np.newaxis]
                * lower_population[np.newaxis, :]
                * bound_bound_survival[np.newaxis, :]
                * stimulated_emission[np.newaxis, :]
                / atmosphere.mass_density[np.newaxis, :]
            )
            continue
        allard_line = (
            unified_allard_table[transition]
            if (
                unified_allard_table is not None
                and transition in unified_allard_table.lines
            )
            else None
        )
        allard_temperature = (
            _allard_profile_temperature_by_depth(allard_line, atmosphere)
            if allard_line is not None
            else atmosphere.temperature
        )
        for depth in range(atmosphere.n_depth):
            profile_per_angstrom = table_line.wavelength_profile(
                wavelength,
                line.wavelength_vacuum_angstrom,
                float(atmosphere.temperature[depth]),
                float(atmosphere.electron_density[depth]),
            )
            stark_weight = allard_stark_weight if allard_line is not None else 1.0
            profile_per_hz = (
                profile_per_angstrom
                * 1.0e8
                * wavelength_cm**2
                / LIGHT_SPEED
            )
            opacity[:, depth] += (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * lower_population[depth]
                * bound_bound_survival[depth]
                * stimulated_emission[depth]
                * profile_per_hz
                * stark_weight
                / atmosphere.mass_density[depth]
            )
        if allard_line is not None:
            cross_section = allard_line.profile_cross_section(
                wavelength,
                allard_temperature,
                lower_population,
                atmosphere.proton_density,
                extend_lyman_alpha_red_wing=True,
            )
            opacity += (
                cross_section
                * lower_population[np.newaxis, :]
                * bound_bound_survival[np.newaxis, :]
                * stimulated_emission[np.newaxis, :]
                / atmosphere.mass_density[np.newaxis, :]
            )
    return opacity


def _allard_profile_temperature_by_depth(
    allard_line: object,
    atmosphere: Atmosphere,
) -> FloatArray:
    """Return the profile-table temperature assigned to each depth.

    Native SYNSPEC reads one file per transition and has no depth-dependent
    interpolation.  ``effective-nearest`` therefore selects one delivered
    file, rather than blending files that can come from different calculation
    generations.
    """

    if len(allard_line.profiles) == 1:
        profile_temperature = allard_line.profiles[0].temperature_K
        selected = (
            atmosphere.effective_temperature
            if profile_temperature is None
            else profile_temperature
        )
    else:
        grid = allard_line.temperatures_K
        selected = float(
            grid[np.argmin(np.abs(grid - atmosphere.effective_temperature))]
        )
    return np.full(atmosphere.n_depth, selected, dtype=np.float64)


def _allard_table_coverage_switch(
    allard_line: object,
    wavelength_angstrom: FloatArray,
    line_center_angstrom: float,
    *,
    taper_fraction: float = 0.10,
    extend_red_wing: bool = False,
) -> FloatArray:
    """Fade a finite Allard table back to the ordinary Stark wing.

    The TLUSTY205 tables end while their density-expansion coefficients are
    still non-zero (Ly-alpha ends at about 1727 Angstrom).  In the
    conservative replacement hybrid, abruptly setting the Allard cross
    section to zero while retaining the perturbed-absorber suppression of the
    Stark profile creates an unphysical opacity and flux step.  This switch
    uses the wavelength interval common to every temperature profile and a
    cubic smoothstep over the outer ten percent of each finite side.  When the
    native Lyalpha red-tail continuation is active, only the blue edge is
    tapered; otherwise the full Stark opacity is restored beyond both edges.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    lower = max(profile.wavelength_angstrom[0] for profile in allard_line.profiles)
    upper = min(profile.wavelength_angstrom[-1] for profile in allard_line.profiles)
    if not lower < line_center_angstrom < upper:
        raise ValueError("Allard table must bracket the hydrogen line center")
    if not 0.0 < taper_fraction <= 0.5:
        raise ValueError("taper_fraction must lie in (0, 0.5]")

    blue_width = taper_fraction * (line_center_angstrom - lower)
    red_width = taper_fraction * (upper - line_center_angstrom)
    blue_coordinate = np.clip((wavelength - lower) / blue_width, 0.0, 1.0)
    blue_smoothstep = blue_coordinate**2 * (3.0 - 2.0 * blue_coordinate)
    if extend_red_wing:
        return blue_smoothstep
    red_coordinate = np.clip((upper - wavelength) / red_width, 0.0, 1.0)
    red_smoothstep = red_coordinate**2 * (3.0 - 2.0 * red_coordinate)
    return blue_smoothstep * red_smoothstep


def lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    allard_table: AllardNeutralLymanAlphaTable | None = None,
) -> FloatArray:
    """Approximate the collision-induced H-H and H-H2 Lyalpha red wing.

    This combines the one-perturber red ``X-B`` and ``b-a`` contributions.
    The curves are tabulated per neutral-H perturber at 6000 K by Rohrmann et
    al. (2011).  They capture the 1623 Angstrom satellite/cutoff and the long
    red wing needed near the cool edge of the current DA milestone.  The
    opacity is intentionally separate from the Stark profile.  If the EOS
    contains molecular hydrogen, the angle-averaged 6000-K H--H2 ``E1-E3``
    plus ``E1-E4`` curve from Fig. 8 of the same paper is included with its
    physical ``n(H) n(H2) / rho`` scaling.  The local 3000--6000 K dependence
    follows the standard KS06 profile family in Sahu et al. (2025, Fig. 4).
    When
    ``allard_table`` is supplied, the H--H approximation is used only in the
    smoothly complementary region outside the finite Allard wavelength
    coverage.  The H--H2 term is not suppressed by the Allard-coverage switch
    because that perturber is absent from the Allard H/H+ table.  Its coarse
    digitized far-wing curve is nevertheless faded in from line center to
    1250 Angstrom so that it cannot create a spurious unresolved core notch.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    wavelength_nm = wavelength * 0.1
    valid = (
        (wavelength_nm >= _LYMAN_ALPHA_H_H_WAVELENGTH_NM[0])
        & (wavelength_nm <= _LYMAN_ALPHA_H_H_WAVELENGTH_NM[-1])
    )
    b_a_cross_section_per_perturber = np.zeros_like(wavelength)
    x_b_cross_section_per_perturber = np.zeros_like(wavelength)
    b_a_cross_section_per_perturber[valid] = 10.0 ** np.interp(
        wavelength_nm[valid],
        _LYMAN_ALPHA_H_H_WAVELENGTH_NM,
        _LYMAN_ALPHA_H_H_LOG_CROSS_SECTION_PER_PERTURBER,
    )
    x_b_valid = (
        (wavelength_nm >= _LYMAN_ALPHA_H_H_X_B_WAVELENGTH_NM[0])
        & (wavelength_nm <= _LYMAN_ALPHA_H_H_X_B_WAVELENGTH_NM[-1])
    )
    x_b_cross_section_per_perturber[x_b_valid] += 10.0 ** np.interp(
        wavelength_nm[x_b_valid],
        _LYMAN_ALPHA_H_H_X_B_WAVELENGTH_NM,
        _LYMAN_ALPHA_H_H_X_B_LOG_CROSS_SECTION_PER_PERTURBER,
    )
    atomic_cross_section_per_perturber = (
        b_a_cross_section_per_perturber[:, np.newaxis]
        + x_b_cross_section_per_perturber[:, np.newaxis]
    )
    if allard_table is not None:
        allard_coverage = _allard_table_coverage_switch(
            allard_table,
            wavelength,
            LYMAN_LINES[0].wavelength_vacuum_angstrom,
            extend_red_wing=True,
        )
        atomic_cross_section_per_perturber *= (
            1.0 - allard_coverage[:, np.newaxis]
        )
    opacity = (
        atomic_cross_section_per_perturber
        * atmosphere.neutral_h_density[np.newaxis, :] ** 2
        / atmosphere.mass_density[np.newaxis, :]
    )

    state = atmosphere.hydrogen_lte_state
    if state is not None and state.molecular_hydrogen_density is not None:
        molecular_valid = (
            (wavelength_nm >= _LYMAN_ALPHA_H_H2_WAVELENGTH_NM[0])
            & (wavelength_nm <= _LYMAN_ALPHA_H_H2_WAVELENGTH_NM[-1])
        )
        molecular_cross_section_per_perturber = np.zeros(
            (wavelength.size, atmosphere.n_depth), dtype=np.float64
        )
        base_log_cross_section = np.interp(
            wavelength_nm[molecular_valid],
            _LYMAN_ALPHA_H_H2_WAVELENGTH_NM,
            _LYMAN_ALPHA_H_H2_LOG_CROSS_SECTION_PER_PERTURBER,
        )
        temperature_correction = _h_h2_lyman_alpha_log_temperature_correction(
            wavelength_nm[molecular_valid], atmosphere.temperature
        )
        molecular_cross_section_per_perturber[molecular_valid, :] = 10.0 ** (
            base_log_cross_section[:, np.newaxis] + temperature_correction
        )
        # The first point was digitized from a broad far-wing figure and is
        # not a resolved unified line-core calculation.  Applying its finite
        # value abruptly at 1216 A creates a false sub-Angstrom notch in cool
        # spectra.  Fade this quasi-static H-H2 approximation in between the
        # isolated line and the next resolved figure point (1250 A); the
        # ordinary Stark/Allard calculation owns the inner core.
        molecular_core_coordinate = np.clip(
            (
                wavelength_nm
                - 0.1 * LYMAN_LINES[0].wavelength_vacuum_angstrom
            )
            / (
                _LYMAN_ALPHA_H_H2_WAVELENGTH_NM[1]
                - 0.1 * LYMAN_LINES[0].wavelength_vacuum_angstrom
            ),
            0.0,
            1.0,
        )
        molecular_core_switch = molecular_core_coordinate**2 * (
            3.0 - 2.0 * molecular_core_coordinate
        )
        molecular_cross_section_per_perturber *= molecular_core_switch[:, np.newaxis]
        opacity += (
            molecular_cross_section_per_perturber
            * atmosphere.neutral_h_density[np.newaxis, :]
            * state.molecular_hydrogen_density[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
        )
    return opacity


def allard_unified_lyman_alpha_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    unified_table: AllardNeutralLymanAlphaTable,
) -> FloatArray:
    """Evaluate external SYNSPEC-format H/H+ Lyalpha opacity in cm2 g-1.

    This compatibility function evaluates only the external Lyalpha cross
    section.  Production synthesis should pass an ``AllardUnifiedLymanTable``
    to ``lyman_mass_absorption_coefficient`` so it can select the requested
    Stark/perturber combination explicitly.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    level_distribution = _atmosphere_level_distribution(atmosphere, maximum_level=2)
    ground_population = level_distribution.population_density[:, 0]
    line_frequency = LIGHT_SPEED / (
        LYMAN_LINES[0].wavelength_vacuum_angstrom * 1.0e-8
    )
    stimulated_emission = 1.0 - np.exp(
        -PLANCK * line_frequency / (BOLTZMANN * atmosphere.temperature)
    )
    profile_temperature = _allard_profile_temperature_by_depth(
        unified_table, atmosphere
    )
    # GETLAL/ALLARD is tabulated for H(1s) perturbers.  Native SYNSPEC passes
    # PJ(1), the ground-state population, both here and as the absorbing
    # population below; using the total neutral atomic density would quietly
    # count excited H atoms as ground-state perturbers in hot/deep layers.
    cross_section = unified_table.profile_cross_section(
        wavelength,
        profile_temperature,
        ground_population,
        atmosphere.proton_density,
        extend_lyman_alpha_red_wing=True,
    )
    return (
        cross_section
        * ground_population[np.newaxis, :]
        * stimulated_emission[np.newaxis, :]
        / atmosphere.mass_density[np.newaxis, :]
    )


def optical_depth_from_mass_opacity(
    column_mass: ArrayLike, mass_opacity: ArrayLike
) -> FloatArray:
    """Integrate monochromatic mass opacity over column mass."""

    column_mass = np.asarray(column_mass, dtype=np.float64)
    opacity = np.asarray(mass_opacity, dtype=np.float64)
    if column_mass.ndim != 1 or opacity.ndim != 2 or opacity.shape[1] != column_mass.size:
        raise ValueError("mass_opacity must have shape (wavelength, depth)")
    if np.any(np.diff(column_mass) <= 0.0) or np.any(opacity <= 0.0):
        raise ValueError("column mass and opacity must be strictly positive")
    optical_depth = np.empty_like(opacity)
    optical_depth[:, 0] = opacity[:, 0] * column_mass[0]
    increments = 0.5 * (opacity[:, 1:] + opacity[:, :-1]) * np.diff(column_mass)
    optical_depth[:, 1:] = optical_depth[:, [0]] + np.cumsum(increments, axis=1)
    return optical_depth
