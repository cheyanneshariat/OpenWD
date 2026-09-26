"""Reduced NLTE ionization ladders for light metals in hot atmospheres.

The ion-stage solver uses Verner et al. ground-state photoionization fits.
The explicit-level solver retains selected Stout levels and lines, with
optional Opacity-Project threshold cross sections read from public TLUSTY
model atoms.  Inverse rates are fixed by detailed balance against the local
LTE reference, so a Planck field recovers the LTE populations.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Sequence, Iterable, Mapping

import numpy as np
from ._compat import trapezoid
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .chianti import ChiantiScaledCollisionComponent
from .constants import (
    BOHR_RADIUS,
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .gaunt import hydrogen_free_free_gaunt_factor
from ._hydrogenic_stark import (
    COMPOSITE_LOG_BETA_MAX,
    COMPOSITE_LOG_BETA_MIN,
    COMPOSITE_POINTS,
    composite_static_profile,
    composite_static_profile_value,
    formula4_extreme_shift,
    hydrogenic_stark_pattern,
)
from .metals import (
    LINE_WINDOW_MAX_FRACTION,
    ATOMIC_MASS_U,
    EV_TO_ERG,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    MetalLTEState,
    VernerPhotoionizationDatabase,
    WAVENUMBER_TO_ERG,
    _pseudo_voigt_profile_per_angstrom,
    selected_metal_lines,
    strong_uv_resonance_minimum_half_window_angstrom,
)
from .spectrum import planck_lambda_angstrom
from .nlte_core import NonphysicalPopulationError

try:  # The package remains usable without a compiler.
    from . import _rt
except ImportError:  # pragma: no cover - exercised by source-only installs
    _rt = None
# Extensions built before the depth-major metal-profile interface lack this flag.
_METAL_PROFILE_DEPTH_MAJOR = bool(getattr(_rt, "METAL_PROFILE_DEPTH_MAJOR", 0))


FloatArray = NDArray[np.float64]
_TMAD_WAVELENGTH_PATTERN = re.compile(r"WAVELENGTH:\s*([0-9.Ee+-]+)")
# A departure coefficient beyond this point only means that its LTE
# reference population has underflowed.  Retaining an infinity is actively
# harmful: downstream opacity algebra then encounters 0 * inf.  Populations
# large enough to affect a spectrum are many orders of magnitude below this
# conservative ceiling.
_MAX_FINITE_DEPARTURE = 1.0e100
_TABULATED_STARK_MEMBER_TOLERANCE_ANGSTROM = 0.1
# Dimitrijevic & Sahal-Brechot (1992, A&AS 93, 359) give electron-impact
# FWHM=0.365 and 0.437 A for the O VI 5s-6p and 5p-6s multiplets at
# T=100,000 K and ne=1e17 cm^-3.  The Cowley formula and the corresponding
# TMAD n_eff parameters give 0.08965 and 0.11897 A.  Their geometric-mean
# ratio supplies a series-level correction without using any stellar data.
_OVI_HIGH_SERIES_SEMICLASSICAL_WIDTH_SCALE = math.sqrt(
    (0.365 / 0.089_652_84) * (0.437 / 0.118_967_13)
)


def _einstein_a_from_absorption_oscillator_strength(
    oscillator_strength: float,
    wavelength_angstrom: float,
    lower_statistical_weight: float,
    upper_statistical_weight: float,
) -> float:
    """Convert an absorption oscillator strength to the line's ``A_ul``.

    TMAD RBB formulae 3--5 store ``f_lu`` followed by the *total radiative
    damping constant of the upper level*.  The latter is appropriate for the
    Voigt width, but is not the Einstein coefficient of every transition
    terminating on that level.  Confusing the two makes very weak lines exert
    the same statistical-equilibrium rate as strong ones.
    """

    wavelength_cm = float(wavelength_angstrom) * 1.0e-8
    return float(
        8.0
        * PI**2
        * ELEMENTARY_CHARGE_ESU**2
        / (ELECTRON_MASS * LIGHT_SPEED * wavelength_cm**2)
        * float(lower_statistical_weight)
        / float(upper_statistical_weight)
        * float(oscillator_strength)
    )


# A fixed Gauss-Laguerre quadrature gives an accurate, dependency-free
# evaluation of the Holtsmark microfield distribution over the core of the
# distribution.  Its large-beta expansion is both faster and more accurate
# in the wings, where direct quadrature becomes oscillatory.
_HOLTSMARK_LAGUERRE_ABSCISSA, _HOLTSMARK_LAGUERRE_WEIGHT = (
    np.polynomial.laguerre.laggauss(96)
)
_HOLTSMARK_COMPILED_ARGUMENT = np.ascontiguousarray(
    _HOLTSMARK_LAGUERRE_ABSCISSA ** (2.0 / 3.0)
)
_HOLTSMARK_COMPILED_WEIGHT = np.ascontiguousarray(
    _HOLTSMARK_LAGUERRE_WEIGHT
    * _HOLTSMARK_LAGUERRE_ABSCISSA ** (1.0 / 3.0)
)


def _holtsmark_microfield_distribution(beta: ArrayLike) -> FloatArray:
    r"""Return the normalized Holtsmark microfield distribution ``U(beta)``.

    TMAP's linear-Stark formula uses

    ``U(beta) = 2 beta / pi int x sin(beta x) exp(-x**(3/2)) dx``.

    The transformed integral is evaluated with Gauss-Laguerre quadrature for
    ``beta <= 8``.  A six-term asymptotic expansion is used farther out; the
    omitted fourth and eighth terms vanish identically.  The small-field
    limit is ``4 beta**2 / (3 pi)``.
    """

    value = np.asarray(beta, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("beta must be finite and non-negative")
    flat = value.ravel()
    result = np.empty_like(flat)
    small = flat < 1.0e-3
    central = (flat >= 1.0e-3) & (flat <= 8.0)
    wing = flat > 8.0
    result[small] = 4.0 / (3.0 * PI) * flat[small] ** 2
    if np.any(central):
        y = _HOLTSMARK_LAGUERRE_ABSCISSA
        weight = _HOLTSMARK_LAGUERRE_WEIGHT
        argument = flat[central, np.newaxis] * y[np.newaxis, :] ** (2.0 / 3.0)
        result[central] = (
            4.0
            * flat[central]
            / (3.0 * PI)
            * np.sum(
                weight[np.newaxis, :]
                * y[np.newaxis, :] ** (1.0 / 3.0)
                * np.sin(argument),
                axis=1,
            )
        )
    if np.any(wing):
        inverse = 1.0 / flat[wing]
        # The asymptotic coefficients follow by expanding exp(-x**3/2)
        # inside the defining sine transform.  Terms n=4 and n=8 are zero.
        result[wing] = (
            1.496_033_551_505_373 * inverse**2.5
            + 7.639_437_268_410_976 * inverse**4.0
            + 21.598_984_399_858_832 * inverse**5.5
            - 447.503_958_034_575_65 * inverse**8.5
            - 3_208.563_652_732_61 * inverse**10.0
            - 12_222.451_853_819_328 * inverse**11.5
        )
    return np.maximum(result.reshape(value.shape), 0.0)


@lru_cache(maxsize=1)
def _ion_dynamic_holtsmark_table() -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Tabulate a moving-ion extension of the static Holtsmark profile.

    The formula-4 Holtsmark function is a distribution of positive
    microfield magnitudes.  Its symmetric line profile is therefore
    ``U(abs(beta))/2``.  A finite microfield correlation time replaces the
    static delta response of each field realization by an impact-limit
    Lorentzian.  Convolving the *static ion term alone* provides the desired
    unified behavior: ion motion fills the otherwise empty line center,
    while the Holtsmark wings are recovered outside the dynamic core.

    The tabulation is lazy because ordinary DA/DB/DZ calculations never use
    this hot-ion correction.  A wide periodic grid makes FFT wraparound
    negligible over the beta<=30 support used by the TMAP approximation.
    """

    full_beta = np.linspace(-160.0, 160.0, 65_537, dtype=np.float64)
    spacing = float(full_beta[1] - full_beta[0])
    static_profile = 0.5 * _holtsmark_microfield_distribution(
        np.abs(full_beta)
    )
    static_profile /= trapezoid(static_profile, full_beta)
    static_transform = np.fft.rfft(np.fft.ifftshift(static_profile))
    dynamic_hwhm = np.geomspace(3.0e-3, 12.0, 37)
    nonnegative = full_beta >= 0.0
    table = np.empty((dynamic_hwhm.size, np.count_nonzero(nonnegative)))
    for index, hwhm in enumerate(dynamic_hwhm):
        lorentz = hwhm / (PI * (full_beta**2 + hwhm**2))
        lorentz /= trapezoid(lorentz, full_beta)
        convolved = np.fft.fftshift(np.fft.irfft(
            static_transform * np.fft.rfft(np.fft.ifftshift(lorentz)),
            n=full_beta.size,
        )) * spacing
        convolved = np.maximum(convolved, 0.0)
        convolved /= trapezoid(convolved, full_beta)
        table[index] = convolved[nonnegative]
    return (
        np.ascontiguousarray(full_beta[nonnegative]),
        np.ascontiguousarray(dynamic_hwhm),
        np.ascontiguousarray(table),
    )


def _ion_dynamic_holtsmark_distribution(
    beta: ArrayLike,
    lorentz_hwhm_beta: float,
) -> FloatArray:
    """Return the symmetric Holtsmark profile with a moving-ion core.

    ``lorentz_hwhm_beta`` is the microfield decorrelation HWHM divided by
    the transition's static Stark frequency scale.  The zero-width limit is
    exactly the normalized two-sided static profile.
    """

    value = np.asarray(beta, dtype=np.float64)
    hwhm = float(lorentz_hwhm_beta)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(hwhm) or hwhm < 0.0:
        raise ValueError("lorentz_hwhm_beta must be finite and non-negative")
    if hwhm == 0.0:
        return 0.5 * _holtsmark_microfield_distribution(value)
    beta_grid, hwhm_grid, table = _ion_dynamic_holtsmark_table()
    if hwhm <= hwhm_grid[0]:
        fraction = hwhm / hwhm_grid[0]
        static = 0.5 * _holtsmark_microfield_distribution(value)
        first = np.interp(
            value, beta_grid, table[0], left=table[0, 0], right=0.0
        )
        return (1.0 - fraction) * static + fraction * first
    upper = int(np.searchsorted(hwhm_grid, hwhm, side="right"))
    upper = min(upper, hwhm_grid.size - 1)
    lower = upper - 1
    log_fraction = (
        (np.log(hwhm) - np.log(hwhm_grid[lower]))
        / (np.log(hwhm_grid[upper]) - np.log(hwhm_grid[lower]))
        if upper != lower else 0.0
    )
    lower_profile = np.interp(
        value, beta_grid, table[lower], left=table[lower, 0], right=0.0
    )
    upper_profile = np.interp(
        value, beta_grid, table[upper], left=table[upper, 0], right=0.0
    )
    return (1.0 - log_fraction) * lower_profile + log_fraction * upper_profile


def _hydrogenic_stark_pattern_key(
    transition: AtomicTransition,
    enabled: bool,
) -> tuple[int, int] | None:
    """Return ``(lower_n, upper_n)`` when a formula-4 line uses component patterns."""

    if not enabled or transition.profile_formula != 4:
        return None
    parameters = transition.profile_parameters
    if len(parameters) < 6:
        return None
    lower_n, upper_n = float(parameters[4]), float(parameters[5])
    if (
        abs(lower_n - round(lower_n)) > 1.0e-6
        or abs(upper_n - round(upper_n)) > 1.0e-6
        or round(lower_n) < 1
        or round(upper_n) < round(lower_n)
    ):
        return None
    return int(round(lower_n)), int(round(upper_n))


def _static_stark_profile(
    beta: FloatArray,
    lorentz_hwhm_beta: float,
    pattern: tuple[int, int] | None,
) -> FloatArray:
    """Two-sided static profile: formula-4 Holtsmark or hydrogenic components.

    ``beta`` is in units of the formula-4 (outermost-component) frequency
    scale.  With a pattern, each shifted component contributes its own
    Holtsmark distribution at its own scale (see ``_hydrogenic_stark``).
    """

    if pattern is None:
        return _ion_dynamic_holtsmark_distribution(beta, lorentz_hwhm_beta)
    lower_n, upper_n = pattern
    if float(lorentz_hwhm_beta) == 0.0:
        return 0.5 * composite_static_profile_value(
            composite_static_profile(lower_n, upper_n), beta
        )
    shifts, strengths = hydrogenic_stark_pattern(lower_n, upper_n)
    extreme = formula4_extreme_shift(lower_n, upper_n)
    value = np.zeros_like(np.asarray(beta, dtype=np.float64))
    for shift, strength in zip(shifts, strengths):
        if shift <= 0.0:
            continue
        ratio = shift / extreme
        value += strength / ratio * _ion_dynamic_holtsmark_distribution(
            np.asarray(beta) / ratio, float(lorentz_hwhm_beta) / ratio
        )
    return value


def _static_pattern_arrays(
    static_pattern: Sequence[tuple[int, int] | None] | None,
) -> tuple[FloatArray, FloatArray] | None:
    """Per-line table row index (-1 = Holtsmark) and the composite table."""

    if static_pattern is None or all(key is None for key in static_pattern):
        return None
    rows: dict[tuple[int, int], int] = {}
    index = np.full(len(static_pattern), -1.0, dtype=np.float64)
    for line, key in enumerate(static_pattern):
        if key is not None:
            index[line] = float(rows.setdefault(key, len(rows)))
    table = np.stack([
        composite_static_profile(*key) for key in sorted(rows, key=rows.get)
    ])
    return np.ascontiguousarray(index), np.ascontiguousarray(table)


def _accumulate_metal_line_profiles_python(
    wavelength: FloatArray,
    planck: FloatArray,
    center: FloatArray,
    integrated_strength: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
    minimum_half_window: FloatArray,
    static_frequency_scale: FloatArray,
    static_amplitude: FloatArray,
    static_ion_motion_hwhm_beta: FloatArray,
    population_scale: FloatArray,
    lower_departure: FloatArray,
    upper_departure: FloatArray,
    exponential: FloatArray,
    absorption: FloatArray,
    emissivity: FloatArray,
    retain_inverted_emissivity: bool,
    static_pattern: Sequence[tuple[int, int] | None] | None = None,
) -> None:
    """Reference implementation of the element-independent profile kernel."""

    for line_index, line_center in enumerate(center):
        center_cm = line_center * 1.0e-8
        for depth in range(planck.shape[1]):
            field_scale = static_frequency_scale[line_index, depth]
            static_half_window = (
                30.0 * field_scale * center_cm**2 / LIGHT_SPEED * 1.0e8
            )
            half_window = max(
                0.25,
                minimum_half_window[line_index],
                10.0 * gaussian_sigma[line_index, depth],
                100.0 * lorentz_hwhm[line_index, depth],
                static_half_window,
            )
            half_window = min(half_window, LINE_WINDOW_MAX_FRACTION * line_center)
            start = int(np.searchsorted(wavelength, line_center - half_window))
            stop = int(np.searchsorted(
                wavelength, line_center + half_window, side="right"
            ))
            if stop <= start:
                continue
            profile_lambda = _pseudo_voigt_profile_per_angstrom(
                wavelength[start:stop],
                float(line_center),
                float(gaussian_sigma[line_index, depth]),
                float(lorentz_hwhm[line_index, depth]),
            )
            cross_section = (
                integrated_strength[line_index]
                * profile_lambda
                * 1.0e8
                * center_cm**2
                / LIGHT_SPEED
            )
            if field_scale > 0.0:
                frequency = LIGHT_SPEED / (wavelength[start:stop] * 1.0e-8)
                beta = np.abs(frequency - LIGHT_SPEED / center_cm) / field_scale
                # TMAD's analytic static profile is specified only to
                # beta=30.  A wider impact-profile support must not silently
                # extrapolate that separate approximation into its far wing.
                inside_static_support = beta <= 30.0
                if np.any(inside_static_support):
                    static_cross_section = np.zeros_like(cross_section)
                    # U(beta) is the normalized distribution of *positive*
                    # microfield magnitudes.  Formula 4 is a symmetric line
                    # profile, so its red and blue branches each carry half
                    # of that area.  The published 0.0368/1.385 prefactor
                    # already assumes this two-sided normalization.
                    static_cross_section[inside_static_support] = (
                        static_amplitude[line_index, depth]
                        * _static_stark_profile(
                            beta[inside_static_support],
                            static_ion_motion_hwhm_beta[line_index, depth],
                            None if static_pattern is None
                            else static_pattern[line_index],
                        )
                    )
                    cross_section = np.maximum(
                        cross_section, static_cross_section
                    )
            net_departure = (
                lower_departure[line_index, depth]
                - upper_departure[line_index, depth]
                * exponential[line_index, depth]
            )
            if net_departure > 0.0:
                absorption[start:stop, depth] += (
                    cross_section
                    * population_scale[line_index, depth]
                    * net_departure
                )
            elif not retain_inverted_emissivity:
                continue
            emissivity[start:stop, depth] += (
                cross_section
                * population_scale[line_index, depth]
                * (1.0 - exponential[line_index, depth])
                * planck[start:stop, depth]
                * upper_departure[line_index, depth]
            )


def _accumulate_metal_line_profiles(
    wavelength: FloatArray,
    planck: FloatArray,
    center: FloatArray,
    integrated_strength: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
    minimum_half_window: FloatArray,
    static_frequency_scale: FloatArray,
    static_amplitude: FloatArray,
    static_ion_motion_hwhm_beta: FloatArray,
    population_scale: FloatArray,
    lower_departure: FloatArray,
    upper_departure: FloatArray,
    exponential: FloatArray,
    absorption: FloatArray,
    emissivity: FloatArray,
    retain_inverted_emissivity: bool,
    *,
    depth_major: bool = False,
    planck_depth_major: FloatArray | None = None,
    static_pattern: Sequence[tuple[int, int] | None] | None = None,
) -> None:
    """Dispatch profile accumulation to C while retaining a NumPy fallback.

    With ``depth_major=True`` the ``absorption`` and ``emissivity`` outputs are
    C-contiguous ``(depth, wavelength)`` arrays and ``planck_depth_major`` is
    the matching transpose of ``planck``.  The compiled kernel then accumulates
    with unit stride; results are bit-identical to the ``(wavelength, depth)``
    layout, which ``planck`` keeps for the NumPy fallback.
    """

    compiled = (
        None
        if _rt is None or np.any(static_ion_motion_hwhm_beta > 0.0)
        else getattr(_rt, "accumulate_metal_line_profiles", None)
    )
    if compiled is not None and depth_major and not _METAL_PROFILE_DEPTH_MAJOR:
        # A compiled extension built before the depth-major interface: use it
        # through contiguous (wavelength, depth) copies of the outputs.
        wave_absorption = np.ascontiguousarray(absorption.T)
        wave_emissivity = np.ascontiguousarray(emissivity.T)
        _accumulate_metal_line_profiles(
            wavelength, planck, center, integrated_strength, gaussian_sigma,
            lorentz_hwhm, minimum_half_window, static_frequency_scale,
            static_amplitude, static_ion_motion_hwhm_beta, population_scale,
            lower_departure, upper_departure, exponential,
            wave_absorption, wave_emissivity, retain_inverted_emissivity,
            static_pattern=static_pattern,
        )
        absorption[...] = wave_absorption.T
        emissivity[...] = wave_emissivity.T
        return
    if compiled is None:
        if depth_major:
            absorption, emissivity = absorption.T, emissivity.T
        _accumulate_metal_line_profiles_python(
            wavelength,
            planck,
            center,
            integrated_strength,
            gaussian_sigma,
            lorentz_hwhm,
            minimum_half_window,
            static_frequency_scale,
            static_amplitude,
            static_ion_motion_hwhm_beta,
            population_scale,
            lower_departure,
            upper_departure,
            exponential,
            absorption,
            emissivity,
            retain_inverted_emissivity,
            static_pattern=static_pattern,
        )
        return
    pattern_arrays = _static_pattern_arrays(static_pattern)
    pattern_arguments = () if pattern_arrays is None else (
        bool(depth_major),
        pattern_arrays[0],
        pattern_arrays[1],
        float(COMPOSITE_LOG_BETA_MIN),
        float(
            (COMPOSITE_LOG_BETA_MAX - COMPOSITE_LOG_BETA_MIN)
            / (COMPOSITE_POINTS - 1)
        ),
    )
    compiled(
        *(
            np.ascontiguousarray(value, dtype=np.float64)
            for value in (
                wavelength,
                planck_depth_major if depth_major else planck,
                center,
                integrated_strength,
                gaussian_sigma,
                lorentz_hwhm,
                minimum_half_window,
                static_frequency_scale,
                static_amplitude,
                population_scale,
                lower_departure,
                upper_departure,
                exponential,
            )
        ),
        _HOLTSMARK_COMPILED_ARGUMENT,
        _HOLTSMARK_COMPILED_WEIGHT,
        absorption,
        emissivity,
        bool(retain_inverted_emissivity),
        # Only depth-major-capable extensions accept the trailing flag.
        *(
            pattern_arguments if pattern_arguments
            else ((True,) if depth_major else ())
        ),
    )


def _line_center_vertical_optical_depth(
    atmosphere: Atmosphere,
    line_center_angstrom: float,
    integrated_strength: float,
    gaussian_sigma_angstrom: FloatArray,
    lorentz_hwhm_angstrom: FloatArray,
    population_scale: FloatArray,
    lower_departure: FloatArray,
    upper_departure: FloatArray,
    exponential: FloatArray,
) -> float:
    """Estimate the bottom vertical optical depth at a line center.

    This inexpensive scalar diagnostic is used only to decide whether the
    formally tiny Voigt area beyond the ordinary 100-HWHM cutoff can still
    matter.  It deliberately excludes the optional static profile: the
    strong UV ground-term lines for which extended support is intended use
    the impact profile, while static-profile lines already set their own
    support from the microfield scale.
    """

    center_profile = np.asarray([
        _pseudo_voigt_profile_per_angstrom(
            np.asarray([line_center_angstrom]),
            line_center_angstrom,
            float(sigma),
            float(hwhm),
        )[0]
        for sigma, hwhm in zip(
            gaussian_sigma_angstrom, lorentz_hwhm_angstrom
        )
    ])
    center_cm = line_center_angstrom * 1.0e-8
    center_cross_section = (
        integrated_strength
        * center_profile
        * 1.0e8
        * center_cm**2
        / LIGHT_SPEED
    )
    net_departure = np.maximum(
        lower_departure - upper_departure * exponential, 0.0
    )
    opacity = center_cross_section * population_scale * net_departure
    column_mass = atmosphere.column_mass
    return float(
        opacity[0] * column_mass[0]
        + np.sum(
            0.5
            * (opacity[1:] + opacity[:-1])
            * np.diff(column_mass)
        )
    )


def _profile_weighted_line_means_python(
    wavelength: FloatArray,
    intensity: FloatArray,
    lambda_diagonal: FloatArray | None,
    center: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Reference profile means used by metal bound-bound rate equations."""

    mean_intensity = np.zeros_like(gaussian_sigma)
    mean_lambda = np.zeros_like(gaussian_sigma)
    for line_index, line_center in enumerate(center):
        for depth in range(intensity.shape[1]):
            half_width = min(max(
                7.0 * gaussian_sigma[line_index, depth],
                100.0 * lorentz_hwhm[line_index, depth],
            ), LINE_WINDOW_MAX_FRACTION * line_center)
            start = int(np.searchsorted(wavelength, line_center - half_width))
            stop = int(np.searchsorted(
                wavelength, line_center + half_width, side="right"
            ))
            if stop - start >= 3:
                local_wavelength = wavelength[start:stop]
                profile = _pseudo_voigt_profile_per_angstrom(
                    local_wavelength,
                    float(line_center),
                    float(gaussian_sigma[line_index, depth]),
                    float(lorentz_hwhm[line_index, depth]),
                )
                normalization = trapezoid(profile, local_wavelength)
                mean_intensity[line_index, depth] = trapezoid(
                    intensity[start:stop, depth] * profile, local_wavelength
                ) / normalization
                if lambda_diagonal is not None:
                    mean_lambda[line_index, depth] = trapezoid(
                        lambda_diagonal[start:stop, depth] * profile,
                        local_wavelength,
                    ) / normalization
            else:
                mean_intensity[line_index, depth] = np.interp(
                    line_center, wavelength, intensity[:, depth]
                )
                if lambda_diagonal is not None:
                    mean_lambda[line_index, depth] = np.interp(
                        line_center, wavelength, lambda_diagonal[:, depth]
                    )
    return mean_intensity, mean_lambda


def _profile_weighted_line_means(
    wavelength: FloatArray,
    intensity: FloatArray,
    lambda_diagonal: FloatArray | None,
    center: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
    *,
    integrated_strength: FloatArray | None = None,
    static_frequency_scale: FloatArray | None = None,
    static_amplitude: FloatArray | None = None,
    static_ion_motion_hwhm_beta: FloatArray | None = None,
    static_pattern: Sequence[tuple[int, int] | None] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute radiative-rate means with the formal-solution profile.

    The compiled kernel supplies the ordinary impact/Voigt means.  TMAD
    formula-4 lines use the pointwise maximum of that impact cross-section
    and a quasi-static linear-Stark cross-section.  Override only those few
    components in Python so their statistical-equilibrium radiative rates
    and final formal solution use the same profile definition.
    """

    compiled = None if _rt is None else getattr(
        _rt, "metal_line_mean_intensity", None
    )
    if compiled is None:
        mean_intensity, mean_lambda = _profile_weighted_line_means_python(
            wavelength,
            intensity,
            lambda_diagonal,
            center,
            gaussian_sigma,
            lorentz_hwhm,
        )
    else:
        wavelength = np.ascontiguousarray(wavelength, dtype=np.float64)
        intensity = np.ascontiguousarray(intensity, dtype=np.float64)
        center = np.ascontiguousarray(center, dtype=np.float64)
        gaussian_sigma = np.ascontiguousarray(gaussian_sigma, dtype=np.float64)
        lorentz_hwhm = np.ascontiguousarray(lorentz_hwhm, dtype=np.float64)
        mean_intensity = np.empty_like(gaussian_sigma)
        mean_lambda = np.empty_like(gaussian_sigma)
        compiled(
            wavelength,
            intensity,
            (
                intensity
                if lambda_diagonal is None
                else np.ascontiguousarray(lambda_diagonal, dtype=np.float64)
            ),
            center,
            gaussian_sigma,
            lorentz_hwhm,
            mean_intensity,
            mean_lambda,
            lambda_diagonal is not None,
        )

    static_inputs = (
        integrated_strength,
        static_frequency_scale,
        static_amplitude,
    )
    if all(value is None for value in static_inputs):
        return mean_intensity, mean_lambda
    if any(value is None for value in static_inputs):
        raise ValueError(
            "integrated_strength, static_frequency_scale, and "
            "static_amplitude must be supplied together"
        )
    strength = np.asarray(integrated_strength, dtype=np.float64)
    field_scale = np.asarray(static_frequency_scale, dtype=np.float64)
    amplitude = np.asarray(static_amplitude, dtype=np.float64)
    ion_motion_hwhm = (
        np.zeros_like(field_scale)
        if static_ion_motion_hwhm_beta is None
        else np.asarray(static_ion_motion_hwhm_beta, dtype=np.float64)
    )
    if strength.shape != center.shape:
        raise ValueError("integrated_strength must have one value per line")
    if (
        field_scale.shape != gaussian_sigma.shape
        or amplitude.shape != gaussian_sigma.shape
        or ion_motion_hwhm.shape != gaussian_sigma.shape
    ):
        raise ValueError("static profile arrays must have shape (line, depth)")

    for line_index, line_center in enumerate(center):
        if not np.any((field_scale[line_index] > 0.0) & (amplitude[line_index] > 0.0)):
            continue
        center_cm = line_center * 1.0e-8
        center_frequency = LIGHT_SPEED / center_cm
        for depth in range(intensity.shape[1]):
            local_field = field_scale[line_index, depth]
            local_amplitude = amplitude[line_index, depth]
            if local_field <= 0.0 or local_amplitude <= 0.0:
                continue
            static_half_width = (
                30.0 * local_field * center_cm**2 / LIGHT_SPEED * 1.0e8
            )
            half_width = min(max(
                7.0 * gaussian_sigma[line_index, depth],
                100.0 * lorentz_hwhm[line_index, depth],
                static_half_width,
            ), LINE_WINDOW_MAX_FRACTION * line_center)
            start = int(np.searchsorted(wavelength, line_center - half_width))
            stop = int(np.searchsorted(
                wavelength, line_center + half_width, side="right"
            ))
            if stop - start < 3:
                continue
            local_wavelength = wavelength[start:stop]
            impact_profile = _pseudo_voigt_profile_per_angstrom(
                local_wavelength,
                float(line_center),
                float(gaussian_sigma[line_index, depth]),
                float(lorentz_hwhm[line_index, depth]),
            )
            impact_cross_section = (
                strength[line_index]
                * impact_profile
                * 1.0e8
                * center_cm**2
                / LIGHT_SPEED
            )
            frequency = LIGHT_SPEED / (local_wavelength * 1.0e-8)
            beta = np.abs(frequency - center_frequency) / local_field
            static_cross_section = np.zeros_like(impact_cross_section)
            inside = beta <= 30.0
            # See the identical two-sided normalization in the formal
            # profile accumulator above.
            static_cross_section[inside] = (
                local_amplitude
                * _static_stark_profile(
                    beta[inside], ion_motion_hwhm[line_index, depth],
                    None if static_pattern is None
                    else static_pattern[line_index],
                )
            )
            profile_weight = np.maximum(
                impact_cross_section, static_cross_section
            )
            normalization = trapezoid(profile_weight, local_wavelength)
            if normalization <= 0.0:
                continue
            mean_intensity[line_index, depth] = trapezoid(
                intensity[start:stop, depth] * profile_weight,
                local_wavelength,
            ) / normalization
            if lambda_diagonal is not None:
                mean_lambda[line_index, depth] = trapezoid(
                    lambda_diagonal[start:stop, depth] * profile_weight,
                    local_wavelength,
                ) / normalization
    return mean_intensity, mean_lambda


def _classical_electron_stark_rate_per_electron(
    charge: int,
    ionization_energy_ev: float | None,
    upper_energy_wavenumber: float,
) -> float:
    """Return the default SYNSPEC classical Stark damping coefficient.

    Hubeny & Lanz's implementation uses ``Gamma_e/n_e = 1e-8 n_eff**5``
    with ``n_eff**2 = Z_eff**2 Ry / chi_upper`` and clips ``n_eff**2`` at
    25.  The result has units cm3 s-1.
    """

    if ionization_energy_ev is None:
        return 0.0
    binding_energy_ev = (
        ionization_energy_ev
        - upper_energy_wavenumber * WAVENUMBER_TO_ERG / EV_TO_ERG
    )
    if binding_energy_ev <= 0.0:
        return 0.0
    effective_n_squared = np.clip(
        (charge + 1.0) ** 2 * 13.595 / binding_energy_ev,
        0.0,
        25.0,
    )
    return float(1.0e-8 * effective_n_squared**2.5)


def _tabulated_electron_stark_fwhm_angstrom(
    element: str,
    charge: int,
    wavelength_angstrom: float,
    electron_density: float,
    temperature: float,
) -> float | None:
    """Return a measured/tabulated electron-impact FWHM for key multiplets.

    The public TMAD synthesis atoms mark C IV 3s--3p with the approximate
    linear-Stark formula and O VI 2s--2p and 3s--3p with the Cowley impact
    formula.  TMAP can replace these by the Dimitrijevic/Sahal-Brechot tables.
    Those tables differ substantially from the generic coefficients precisely
    for the O VI resonance doublet and the optical C IV 5801/5812 and
    O VI 3811/3834 diagnostics.

    C IV values are the electron widths in Dimitrijevic, Sahal-Brechot &
    Bommier (1991), normalized to ``n_e=1e17 cm-3``.  O V 3p--3d uses the
    machine-readable Dimitrijevic & Sahal-Brechot (1995) CDS table at that
    same density.  O VI 2s--2p uses Dimitrijevic & Sahal-Brechot (1992), whose
    tables show linear density scaling from 1e17 through 1e20 cm-3.  O VI
    3s--3p uses the critically compiled measurements of Wiese et al. (2002),
    individually normalized by their measured densities.  Impact widths are
    linear in density over the line-forming regime.  Log interpolation avoids
    artificial temperature kinks; end values are held fixed rather than
    extrapolating sparse data.
    """

    symbol = element.strip().capitalize()
    center = float(wavelength_angstrom)
    if symbol == "C" and charge == 3 and 1_540.0 < center < 1_560.0:
        members = (1_548.203, 1_550.777, 1_549.05)
        # C IV 2s--2p resonance doublet.  Elabidi, Sahal-Brechot &
        # Ben Nessib (2011), table 7; their quantum widths agree with the
        # Dimitrijevic, Sahal-Brechot & Bommier (1991) semiclassical values
        # to 3--13 percent over this temperature interval.
        table_temperature = np.asarray(
            (20_000.0, 40_000.0, 50_000.0, 80_000.0,
             100_000.0, 150_000.0, 200_000.0)
        )
        fwhm_per_1e17 = np.asarray(
            (0.0118, 0.00844, 0.00760, 0.00616,
             0.00561, 0.00478, 0.00430)
        )
    elif symbol == "C" and charge == 3 and 5_750.0 < center < 5_850.0:
        members = (5_802.955, 5_813.582, 5_806.47)
        table_temperature = np.asarray(
            (10_000.0, 20_000.0, 50_000.0, 80_000.0, 100_000.0, 150_000.0, 200_000.0)
        )
        fwhm_per_1e17 = np.asarray((1.23, 0.893, 0.603, 0.502, 0.463, 0.402, 0.366))
    elif symbol == "O" and charge == 4 and 5_500.0 < center < 5_680.0:
        members = (5_573.351, 5_581.657, 5_584.774, 5_599.441, 5_605.813,
                   5_608.957, 5_591.44)
        # O V 3p--3d multiplet; CDS J/A+AS/109/551, table 1.
        table_temperature = np.asarray((40_000.0, 100_000.0, 200_000.0, 500_000.0))
        fwhm_per_1e17 = np.asarray((0.282, 0.189, 0.143, 0.105))
    elif symbol == "O" and charge == 5 and 1_025.0 < center < 1_045.0:
        members = (1_031.912, 1_037.613, 1_033.81)
        # O VI 2s--2p resonance multiplet; Dimitrijevic & Sahal-Brechot
        # (1992), A&AS 93, 359, table 1.  Values below are FWHM at
        # n_e=1e17 cm-3; their higher-density entries are exactly linear
        # through 1e20 cm-3 to the printed precision.
        table_temperature = np.asarray(
            (100_000.0, 200_000.0, 500_000.0, 1_000_000.0)
        )
        fwhm_per_1e17 = np.asarray((0.00146, 0.00105, 0.000694, 0.000525))
    elif symbol == "O" and charge == 5 and 3_775.0 < center < 3_875.0:
        members = (3_812.429, 3_835.326, 3_820.03)
        table_temperature = np.asarray(
            (61_900.0, 65_500.0, 79_700.0, 96_300.0, 133_400.0, 181_000.0, 203_100.0)
        )
        # Each laboratory FWHM is divided by its independently measured
        # electron density in units of 1e17 cm-3.
        fwhm_per_1e17 = np.asarray(
            (0.178 / 1.38, 0.136 / 1.09, 0.171 / 1.42,
             1.0 / 10.0, 1.4 / 13.0, 1.8 / 21.0, 2.1 / 24.0)
        )
    else:
        return None
    # The windows also contain unrelated high-n lines (e.g. O VI 5-8 at
    # 1036-1043 A, O V 2p4s-2s6d at 5616 A); only the tabulated multiplet's
    # own components (and the term centroid used by the rate atoms) qualify.
    if min(abs(center - member) for member in members) > _TABULATED_STARK_MEMBER_TOLERANCE_ANGSTROM:
        return None
    log_width = np.interp(
        np.log(np.clip(float(temperature), table_temperature[0], table_temperature[-1])),
        np.log(table_temperature),
        np.log(fwhm_per_1e17),
    )
    return float(np.exp(log_width) * max(float(electron_density), 0.0) / 1.0e17)


def _mapped_fine_structure_components(
    element: str,
    charge: int,
    transition: AtomicTransition,
    formal_atomic_database: AtomicDatabase | None,
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None,
) -> tuple[AtomicTransition, ...]:
    """Return formal components joining the mapped endpoints of a term line.

    A compact TMAD population atom carries one LS transition at its
    term-centroid wavelength, whereas the opacity atom carries the actual
    fine-structure components. Sampling the radiation field at the centroid
    can land between a widely split doublet.
    """

    if formal_atomic_database is None or formal_level_mapping is None:
        return ()
    symbol = element.strip().capitalize()
    lower = formal_level_mapping.get((symbol, charge, transition.lower_index), ())
    upper = formal_level_mapping.get((symbol, charge, transition.upper_index), ())
    if not lower or not upper:
        return ()
    formal_ion = formal_atomic_database.ions.get((symbol, charge))
    if formal_ion is None:
        return ()
    lower_index = {key[2] for key in lower}
    upper_index = {key[2] for key in upper}
    return tuple(
        component
        for component in formal_ion.transitions
        if component.lower_index in lower_index
        and component.upper_index in upper_index
        and component.einstein_a > 0.0
    )


def _impact_profile_damping_rate(
    transition: AtomicTransition,
    upper_radiative_rate: float,
    generic_stark_rate_per_electron: float,
    electron_density: float,
    temperature: float,
    *,
    include_electron_stark: bool = True,
    element: str | None = None,
    charge: int | None = None,
    tabulated_electron_stark_width_scale: float = 1.0,
    include_semiclassical_ovi_stark_widths: bool = False,
) -> float:
    """Return the Lorentz damping rate prescribed by a TMAD profile flag."""

    formula = transition.profile_formula
    parameters = transition.profile_parameters
    if formula == 1:
        return 0.0
    if formula == 2:
        return float(parameters[1]) if len(parameters) >= 2 else 0.0
    tabulated_fwhm = (
        None
        if not include_electron_stark or element is None or charge is None
        else _tabulated_electron_stark_fwhm_angstrom(
            element,
            charge,
            transition.wavelength_vacuum_angstrom,
            electron_density,
            temperature,
        )
    )
    if tabulated_fwhm is not None:
        if not np.isfinite(tabulated_electron_stark_width_scale) or (
            tabulated_electron_stark_width_scale <= 0.0
        ):
            raise ValueError(
                "tabulated_electron_stark_width_scale must be finite and positive"
            )
        tabulated_fwhm *= tabulated_electron_stark_width_scale
        center_cm = transition.wavelength_vacuum_angstrom * 1.0e-8
        electron_rate = (
            tabulated_fwhm
            * 2.0
            * PI
            * LIGHT_SPEED
            / (center_cm**2 * 1.0e8)
        )
        radiative_rate = (
            float(parameters[1])
            if formula in (3, 4, 5) and len(parameters) >= 2
            else float(upper_radiative_rate)
        )
        return radiative_rate + electron_rate
    if formula in (3, 4, 5) and len(parameters) >= 3:
        electron_rate = (
            6.11e-5
            * electron_density
            / np.sqrt(temperature)
            * parameters[2]
            if include_electron_stark else 0.0
        )
        if (
            include_semiclassical_ovi_stark_widths
            and _is_ovi_high_series_formula4(transition, element, charge)
        ):
            electron_rate *= _OVI_HIGH_SERIES_SEMICLASSICAL_WIDTH_SCALE
        return float(parameters[1] + electron_rate)
    return float(
        upper_radiative_rate
        + (
            generic_stark_rate_per_electron * electron_density
            if include_electron_stark else 0.0
        )
    )


def _is_ovi_high_series_formula4(
    transition: AtomicTransition,
    element: str | None,
    charge: int | None,
) -> bool:
    """Identify the adjacent high-Rydberg O VI transitions at issue here."""

    parameters = transition.profile_parameters
    return bool(
        element == "O"
        and charge == 5
        and transition.profile_formula == 4
        and len(parameters) >= 6
        and parameters[4] >= 5.0
        and abs(parameters[5] - parameters[4] - 1.0) < 0.25
    )



def _static_linear_stark_scale(
    scales: Mapping[tuple, float] | None,
    element: str,
    charge: int,
    lower_n: float,
    upper_n: float,
) -> float:
    """Frequency scale of TMAP's formula-4 quasi-static wing for one line.

    ``scales`` is keyed by ``(element, charge, lower_n, upper_n)`` or, for a
    whole ion, ``(element, charge)``; the transition key takes precedence.
    A scale of 0 removes the quasi-static wing, leaving the impact profile
    (TMAP's behaviour when an ion's lines use tabulated Stark widths, e.g.
    the LINE1 ``STARK OVI`` option).
    """

    if scales is None:
        return 1.0
    scale = scales.get(
        (element, int(charge), int(round(lower_n)), int(round(upper_n))),
        scales.get((element, int(charge)), 1.0),
    )
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "static linear-Stark frequency scales must be finite and nonnegative"
        )
    return scale


def _ion_microfield_motion_rate(
    atmosphere: Atmosphere,
    lte_state: MetalLTEState,
    radiator_atomic_mass_u: float,
) -> FloatArray:
    """Return the inverse ion-microfield correlation time ``v/R0``.

    A charged perturber changes the local microfield on approximately its
    ion-sphere crossing time.  The velocity is averaged with the same
    ``Z**(3/2) n_i`` weights that define TMAP's Holtsmark field.  The rate is
    converted to a Lorentz HWHM only for the quasi-static ion profile; it is
    not added to the electron-impact damping constant.
    """

    atomic_mass_unit = 1.660_539_068_92e-24
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    charged_density = np.zeros(atmosphere.n_depth, dtype=np.float64)
    microfield_weight = np.zeros_like(charged_density)
    speed_weight = np.zeros_like(charged_density)

    def add_population(
        population: FloatArray,
        charge: int,
        perturber_atomic_mass_u: float,
    ) -> None:
        if charge <= 0:
            return
        density = np.asarray(population, dtype=np.float64)
        reduced_mass_u = (
            radiator_atomic_mass_u * perturber_atomic_mass_u
            / (radiator_atomic_mass_u + perturber_atomic_mass_u)
        )
        mean_relative_speed = np.sqrt(
            8.0
            * BOLTZMANN
            * temperature
            / (PI * reduced_mass_u * atomic_mass_unit)
        )
        weight = charge**1.5 * density
        charged_density[:] += density
        microfield_weight[:] += weight
        speed_weight[:] += weight * mean_relative_speed

    host_population = lte_state.host_ion_number_density
    if host_population is not None:
        host_mass = (
            _HYDROGEN_ATOMIC_MASS_U
            if lte_state.reference_species == "H"
            else _HELIUM_ATOMIC_MASS_U
        )
        for charge in range(1, host_population.shape[0]):
            add_population(host_population[charge], charge, host_mass)
    for element, populations in lte_state.ion_number_density.items():
        mass = ATOMIC_MASS_U.get(element)
        if mass is None:
            continue
        for charge in range(1, populations.shape[0]):
            add_population(populations[charge], charge, float(mass))

    valid = (charged_density > 0.0) & (microfield_weight > 0.0)
    result = np.zeros(atmosphere.n_depth, dtype=np.float64)
    if np.any(valid):
        ion_sphere_radius = (
            3.0 / (4.0 * PI * charged_density[valid])
        ) ** (1.0 / 3.0)
        effective_speed = speed_weight[valid] / microfield_weight[valid]
        result[valid] = effective_speed / ion_sphere_radius
    return np.ascontiguousarray(result)


def _tlusty_van_regemorter_gbar(u: float, minimum: float = 0.25) -> float:
    """Return TLUSTY's ICOL=0 Gaunt factor for an allowed transition."""

    value = max(float(u), 1.0e-12)
    if value <= 1.0:
        exponential_integral = (
            -np.log(value)
            - 0.577_215_66
            + value
            * (
                0.999_991_93
                + value
                * (
                    -0.249_910_55
                    + value
                    * (
                        0.055_199_68
                        + value * (-0.009_760_04 + value * 0.001_078_57)
                    )
                )
            )
        )
    else:
        numerator = (
            0.267_773_4343
            + value
            * (
                8.634_760_8925
                + value * (18.059_016_973 + value * (8.573_328_7401 + value))
            )
        )
        denominator = (
            3.958_496_9228
            + value
            * (
                21.099_653_0827
                + value * (25.632_956_1486 + value * (9.573_322_3454 + value))
            )
        )
        exponential_integral = np.exp(-value) * numerator / denominator / value
    return max(minimum, 0.276 * np.exp(value) * exponential_integral)


TLUSTY_CIII_ATOM_URL = (
    "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/c3.dat"
)
TLUSTY_CIII_ATOM_SHA256 = (
    "6e8e9517c3d2f5d5e25a2200d53a302f43451d8cc2a6f947113d8a6863f1395e"
)
TLUSTY_CIV_ATOM_URL = (
    "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/c4_35+2lev.dat"
)
TLUSTY_CIV_ATOM_SHA256 = (
    "024b8335b3d416f8d079afa15d6935597e9b786ae4738f2d0a2991c9a1554979"
)
TLUSTY_OV_ATOM_URL = (
    "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/o5.dat"
)
TLUSTY_OV_ATOM_SHA256 = (
    "bda1ff41b35833dcfd55d470a6d6d3399800eb620d32bf80629ce4126b88cb6a"
)
TLUSTY_OVI_ATOM_URL = (
    "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/o6.dat"
)
TLUSTY_OVI_ATOM_SHA256 = (
    "3c955a831b2e260ed465a01584aefe1526ebe3b49b0bfd61ddd0f3dd16c6d8e8"
)
TLUSTY_OIV_ATOM_URL = (
    "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/o4.dat"
)
TLUSTY_OIV_ATOM_SHA256 = (
    "c2b6304ffafc1ddf0a736b47d27928d20121be3a3f5021d406998d69e6cda616"
)


def _term_spin_multiplicity(label: str | None) -> int | None:
    """Extract ``2S+1`` from ordinary or TLUSTY superlevel term labels."""

    if not label:
        return None
    text = label.upper()
    # Stout configuration labels put the actual term in parentheses.  Do
    # this before the generic expression so ``1s2.6f.(2Fo<5/2>)`` is read as
    # a doublet F term, not as multiplicity one from the closed 1s shell.
    parenthesized = re.findall(
        r"\(([1-9])\s*[SPDFGHIKLMNOQ](?:[EO])?",
        text,
    )
    if parenthesized:
        return int(parenthesized[-1])
    matches = re.findall(r"(?:^|\s)([1-9])\s*[SPDFGHI]", text)
    if matches:
        return int(matches[-1])
    superlevel = re.search(r"\+([1-9])__", text)
    return None if superlevel is None else int(superlevel.group(1))


def _term_orbital_letter(label: str | None) -> str | None:
    """Extract the LS-term orbital letter from TMAD/TLUSTY level labels."""

    if not label:
        return None
    matches = re.findall(r"[1-9]\s*([SPDFGHIKLMNOQ])(?:[EO])?\b", label.upper())
    return None if not matches else matches[-1]


def _term_principal_quantum_number(label: str | None) -> int | None:
    """Extract the outer-shell principal quantum number from atom labels."""

    if not label:
        return None
    text = label.upper()
    # Fixed-width TMAD labels begin with element, spectroscopic stage, then
    # a two-digit outer-shell principal quantum number (for example C406F).
    # C III writes n >= 10 with three digits (C3010P).
    tmad = re.match(r"^[A-Z]{1,2}\d(\d{2,3})[SPDFGHIKLMNOQ]", text)
    if tmad is not None:
        return int(tmad.group(1))
    # Stout labels retain the configuration explicitly, for example
    # ``1s2.6f.(2Fo<5/2>)``. The last configuration orbital is the outer
    # electron for the C IV/O VI Rydberg terms relevant here.
    configurations = re.findall(
        r"(?<![A-Z0-9])(\d+)[SPDFGHIKLMNOQ](?:\d+)?(?=\.|\()",
        text,
    )
    return None if not configurations else int(configurations[-1])


def _spectroscopic_term_labels_match(
    term_label: str | None,
    formal_label: str | None,
) -> bool | None:
    """Compare the available n, multiplicity, and L term-label fields.

    ``None`` means that the labels contain no common spectroscopic field and
    the caller should retain the historical energy/statistical-weight match.
    """

    term_fields = (
        _term_principal_quantum_number(term_label),
        _term_spin_multiplicity(term_label),
        _term_orbital_letter(term_label),
    )
    formal_fields = (
        _term_principal_quantum_number(formal_label),
        _term_spin_multiplicity(formal_label),
        _term_orbital_letter(formal_label),
    )
    compared = False
    for term_value, formal_value in zip(term_fields, formal_fields):
        if term_value is None or formal_value is None:
            continue
        compared = True
        if term_value != formal_value:
            return False
    return True if compared else None


@dataclass(frozen=True)
class TlustyPhotoionizationThresholdData:
    """Opacity-Project photoionization fits from one TLUSTY model atom.

    TLUSTY stores detailed fits as ``x=log10(nu/nu0)`` and
    ``y=log10(sigma/1 Mb)``.  Its continuum-record ``OSC`` value is the
    effective Seaton collisional-ionization normalization (the TMAP g-bar is
    already folded into it), not necessarily the radiative cross section at
    threshold.  Analytic Peach/Henry/Butler/DETAIL photoionization fits are
    retained separately.
    """

    threshold_frequency_hz: FloatArray
    statistical_weight: FloatArray
    level_label: tuple[str, ...]
    threshold_cross_section_cm2: FloatArray
    log_frequency_ratio: tuple[FloatArray | None, ...]
    log_cross_section_megabar: tuple[FloatArray | None, ...]
    photoionization_formula: tuple[int, ...]
    photoionization_parameters: tuple[FloatArray | None, ...]
    line_wavelength_vacuum_angstrom: FloatArray
    line_collision_gbar: FloatArray
    source: str
    # TLUSTY continuum-record ``OSC`` values already include its Seaton
    # collisional-ionization g-bar. Raw TOPbase/SIROCCO photoionization
    # cross-sections do not.
    threshold_cross_section_includes_gbar: bool = True

    def _matching_index(
        self,
        threshold_frequency_hz: float,
        relative_tolerance: float,
        *,
        level_label: str | None = None,
    ) -> int | None:
        difference = np.abs(
            self.threshold_frequency_hz / threshold_frequency_hz - 1.0
        )
        candidates = difference <= relative_tolerance
        requested_spin = _term_spin_multiplicity(level_label)
        if requested_spin is not None:
            tabulated_spin = np.asarray(
                [_term_spin_multiplicity(label) for label in self.level_label],
                dtype=object,
            )
            spin_matched = candidates & (tabulated_spin == requested_spin)
            if np.any(spin_matched):
                candidates = spin_matched
            elif any(spin is not None for spin in tabulated_spin):
                # A nearby threshold in the opposite spin system is not a
                # valid substitute.  This matters for the paired O V n=5--7
                # singlet/triplet superlevels in the public TLUSTY atom.
                return None
        requested_orbital = _term_orbital_letter(level_label)
        if requested_orbital is not None:
            tabulated_orbital = np.asarray(
                [_term_orbital_letter(label) for label in self.level_label],
                dtype=object,
            )
            orbital_matched = candidates & (
                tabulated_orbital == requested_orbital
            )
            if np.any(orbital_matched):
                candidates = orbital_matched
            else:
                superlevel_matched = candidates & np.asarray(
                    [orbital is None for orbital in tabulated_orbital]
                )
                if np.any(superlevel_matched):
                    # A TLUSTY ``+M__`` superlevel intentionally bundles
                    # several orbital terms of the requested multiplicity.
                    candidates = superlevel_matched
                elif any(
                    orbital is not None for orbital in tabulated_orbital
                ):
                    # Near-degenerate Rydberg terms of different l can have
                    # very different OP cross sections. Threshold proximity
                    # alone is therefore not a safe high-level match.
                    return None
        if not np.any(candidates):
            return None
        indices = np.flatnonzero(candidates)
        return int(indices[np.argmin(difference[indices])])

    def cross_section_for_threshold(
        self,
        threshold_frequency_hz: float,
        *,
        relative_tolerance: float = 0.01,
        level_label: str | None = None,
    ) -> float | None:
        index = self._matching_index(
            threshold_frequency_hz,
            relative_tolerance,
            level_label=level_label,
        )
        if index is None:
            return None
        return float(self.threshold_cross_section_cm2[index])

    def cross_section(
        self,
        photon_frequency_hz: ArrayLike,
        threshold_frequency_hz: float,
        *,
        relative_tolerance: float = 0.01,
        level_label: str | None = None,
    ) -> FloatArray | None:
        """Evaluate the matched OP fit on an arbitrary frequency array."""

        index = self._matching_index(
            threshold_frequency_hz,
            relative_tolerance,
            level_label=level_label,
        )
        if index is None:
            return None
        frequency = np.asarray(photon_frequency_hz, dtype=np.float64)
        ratio = frequency / threshold_frequency_hz
        result = np.zeros_like(frequency)
        selected = ratio >= 1.0
        if not np.any(selected):
            return result
        x_fit = self.log_frequency_ratio[index]
        y_fit = self.log_cross_section_megabar[index]
        formula = self.photoionization_formula[index]
        parameters = self.photoionization_parameters[index]
        if formula in (2, 3, 4, 6) and parameters is not None:
            frequency_selected = frequency[selected]
            relative = threshold_frequency_hz / frequency_selected
            if formula == 2:
                exponent, amplitude, shape, alternate_edge = parameters
                valid = np.ones_like(relative, dtype=bool)
                if alternate_edge > 0.0:
                    alternate_frequency = (
                        LIGHT_SPEED * 1.0e8 / alternate_edge
                        if alternate_edge < 1.0e6
                        else alternate_edge
                    )
                    valid = frequency_selected >= alternate_frequency
                    relative = alternate_frequency / frequency_selected
                local = np.zeros_like(relative)
                local[valid] = (
                    amplitude
                    * relative[valid] ** exponent
                    * (
                        shape
                        + relative[valid] * (1.0 - shape)
                    )
                    * 1.0e-18
                )
                result[selected] = local
                return result
            log_relative = np.log(np.maximum(relative, 1.0e-300))
            if formula == 3:
                exponent, amplitude, linear, constant = parameters
                result[selected] = (
                    amplitude
                    * relative**exponent
                    * (
                        constant
                        + relative
                        * (
                            linear
                            - 2.0 * constant
                            + relative * (1.0 + constant - linear)
                        )
                    )
                    * 1.0e-18
                )
                return result
            if formula == 4:
                constant, linear, quadratic, _ = parameters
                result[selected] = np.exp(
                    constant + log_relative * (linear + log_relative * quadratic)
                )
                return result
            powers = np.stack([
                log_relative**order for order in range(len(parameters))
            ])
            result[selected] = np.exp(np.sum(parameters[:, np.newaxis] * powers, axis=0))
            return result
        if x_fit is None or y_fit is None:
            result[selected] = (
                self.threshold_cross_section_cm2[index] * ratio[selected] ** -3.0
            )
            return result
        x = np.log10(ratio[selected])
        # TLUSTY defines tabulated OP sections to be exactly zero below their
        # first fit point.  This matters when the nominal continuum threshold
        # is spin-forbidden and the first allowed residual-ion channel opens
        # at a higher frequency.  NumPy's default left extrapolation would
        # otherwise copy the first non-zero cross section down to threshold.
        covered = (x >= x_fit[0]) | np.isclose(
            x, x_fit[0], rtol=0.0, atol=1.0e-12
        )
        if not np.any(covered):
            return result
        local = np.zeros_like(x)
        # TLUSTY interpolates TOPbase fits in log-log space. Continue the
        # final declining segment above the last fit point.
        covered_x = x[covered]
        y = np.interp(covered_x, x_fit, y_fit)
        above = covered_x > x_fit[-1]
        if np.any(above) and x_fit.size > 1:
            slope = (y_fit[-1] - y_fit[-2]) / (x_fit[-1] - x_fit[-2])
            y[above] = y_fit[-1] + slope * (covered_x[above] - x_fit[-1])
        local[covered] = 1.0e-18 * 10.0 ** np.clip(y, -300.0, 100.0)
        result[selected] = local
        return result

    def collision_gbar_for_wavelength(
        self, wavelength_vacuum_angstrom: float, *, relative_tolerance: float = 0.01
    ) -> float | None:
        """Return a matched TLUSTY ICOL=1 g-bar for a bound-bound line."""

        if self.line_wavelength_vacuum_angstrom.size == 0:
            return None
        difference = np.abs(
            self.line_wavelength_vacuum_angstrom / wavelength_vacuum_angstrom - 1.0
        )
        index = int(np.argmin(difference))
        if difference[index] > relative_tolerance:
            return None
        return float(self.line_collision_gbar[index])


def read_tlusty_photoionization_threshold_data(
    path: str | Path,
) -> TlustyPhotoionizationThresholdData:
    """Read level thresholds and frequency-dependent OP fits from TLUSTY."""

    lines = Path(path).read_text(encoding="ascii").splitlines()
    try:
        level_start = next(
            index for index, line in enumerate(lines) if line.startswith("****** Levels")
        ) + 1
        continuum_start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("****** Continuum transitions")
        )
        continuum_stop = next(
            index
            for index, line in enumerate(lines[continuum_start + 1 :], continuum_start + 1)
            if "Line transitions" in line
        )
    except StopIteration as error:
        raise ValueError(f"{path} is not a supported TLUSTY model atom") from error

    thresholds = []
    weights = []
    labels = []
    for line in lines[level_start:continuum_start]:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            thresholds.append(float(fields[0]))
            weights.append(float(fields[1]))
            label_match = re.search(r"'([^']*)'", line)
            labels.append("" if label_match is None else label_match.group(1).strip())
        except ValueError as error:
            raise ValueError(f"malformed TLUSTY level record in {path}: {line}") from error

    cross_sections = np.full(len(thresholds), np.nan)
    frequency_fits: list[FloatArray | None] = [None] * len(thresholds)
    cross_section_fits: list[FloatArray | None] = [None] * len(thresholds)
    photoionization_formula = np.zeros(len(thresholds), dtype=np.int64)
    photoionization_parameters: list[FloatArray | None] = [None] * len(thresholds)
    line_index = continuum_start + 1

    def read_values(start: int, count: int) -> tuple[FloatArray, int]:
        values: list[float] = []
        while start < continuum_stop and len(values) < count:
            values.extend(float(value) for value in lines[start].split())
            start += 1
        if len(values) != count:
            raise ValueError(f"truncated TOPbase fit in {path}")
        return np.asarray(values, dtype=np.float64), start

    while line_index < continuum_stop:
        fields = lines[line_index].split()
        line_index += 1
        if len(fields) != 9:
            continue
        try:
            lower = int(fields[0])
            int(fields[1])
            int(fields[2])
            ifancy = int(fields[3])
            tuple(int(value) for value in fields[4:7])
            sigma = float(fields[7])
            float(fields[8])
        except ValueError:
            continue
        if not 1 <= lower <= len(cross_sections):
            continue
        if sigma > 0.0:
            cross_sections[lower - 1] = sigma
        photoionization_formula[lower - 1] = ifancy
        if ifancy > 100:
            fit_count = ifancy - 100
            x_fit, line_index = read_values(line_index, fit_count)
            y_fit, line_index = read_values(line_index, fit_count)
            frequency_fits[lower - 1] = x_fit
            cross_section_fits[lower - 1] = y_fit
        elif 2 <= ifancy <= 4:
            parameters, line_index = read_values(line_index, 4)
            photoionization_parameters[lower - 1] = parameters
        elif ifancy == 6:
            parameters, line_index = read_values(line_index, 6)
            photoionization_parameters[lower - 1] = parameters
    valid = np.isfinite(cross_sections)
    if not np.any(valid):
        raise ValueError(f"{path} contains no readable continuum thresholds")
    line_wavelength = []
    line_gbar = []
    for line in lines[continuum_stop + 1 :]:
        fields = line.split()
        if len(fields) != 9:
            continue
        try:
            lower = int(fields[0])
            upper = int(fields[1])
            tuple(int(value) for value in fields[2:7])
            oscillator_strength = float(fields[7])
            collision_parameter = float(fields[8])
            collision_type = int(fields[4])
        except ValueError:
            continue
        if (
            collision_type != 1
            or oscillator_strength <= 0.0
            or collision_parameter <= 0.0
            or not (1 <= lower <= len(thresholds))
            or not (1 <= upper <= len(thresholds))
        ):
            continue
        frequency = abs(thresholds[lower - 1] - thresholds[upper - 1])
        if frequency > 0.0:
            line_wavelength.append(LIGHT_SPEED * 1.0e8 / frequency)
            line_gbar.append(collision_parameter)

    return TlustyPhotoionizationThresholdData(
        threshold_frequency_hz=np.asarray(thresholds)[valid],
        statistical_weight=np.asarray(weights)[valid],
        level_label=tuple(labels[index] for index in np.flatnonzero(valid)),
        threshold_cross_section_cm2=cross_sections[valid],
        log_frequency_ratio=tuple(
            frequency_fits[index] for index in np.flatnonzero(valid)
        ),
        log_cross_section_megabar=tuple(
            cross_section_fits[index] for index in np.flatnonzero(valid)
        ),
        photoionization_formula=tuple(
            int(photoionization_formula[index])
            for index in np.flatnonzero(valid)
        ),
        photoionization_parameters=tuple(
            photoionization_parameters[index] for index in np.flatnonzero(valid)
        ),
        line_wavelength_vacuum_angstrom=np.asarray(line_wavelength),
        line_collision_gbar=np.asarray(line_gbar),
        source=f"TLUSTY/Opacity Project frequency-dependent data: {Path(path).name}",
    )


def read_sirocco_topbase_photoionization_data(
    level_path: str | Path,
    photoionization_path: str | Path,
    population_ion: AtomicIon,
    *,
    line_collision_data: TlustyPhotoionizationThresholdData | None = None,
) -> TlustyPhotoionizationThresholdData:
    """Map SIROCCO's level-resolved TOPbase data onto one LS-term ion.

    SIROCCO distributes TOPbase photoionization cross-sections on its
    fine-structure level indices. The O VI components belonging to one LS
    term share the same cross-section grid, so one component can be mapped to
    the corresponding term in a TMAD population atom. Terms absent from the
    returned table deliberately remain absent; the reduced-level solver then
    applies its hydrogenic excited-level fallback, matching TMAP's documented
    ``MISSING HYDROGENIC`` treatment rather than borrowing a neighbouring
    shell-superlevel cross-section.
    """

    level_source = Path(level_path)
    photo_source = Path(photoionization_path)
    level_pattern = re.compile(
        r"^1s2\.(?P<n>\d+)(?P<orbital>[spdfghiklmnoq])_",
        re.IGNORECASE,
    )
    level_terms: dict[int, tuple[int, str]] = {}
    for line in level_source.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) < 11 or fields[0] != "LevMacro":
            continue
        try:
            atomic_number = int(fields[1])
            ion_stage = int(fields[2])
            level_index = int(fields[3])
        except ValueError as error:
            raise ValueError(
                f"malformed SIROCCO level record in {level_source}: {line}"
            ) from error
        if atomic_number != 8 or ion_stage != 6:
            continue
        match = level_pattern.match(fields[8])
        if match is not None:
            level_terms[level_index] = (
                int(match.group("n")), match.group("orbital").upper()
            )

    sections: dict[tuple[int, str], tuple[float, FloatArray, FloatArray]] = {}
    lines = photo_source.read_text(encoding="ascii").splitlines()
    line_index = 0
    while line_index < len(lines):
        fields = lines[line_index].split()
        line_index += 1
        if not fields or fields[0] != "PhotMacS":
            continue
        if len(fields) != 7:
            raise ValueError(
                f"malformed SIROCCO photoionization header in "
                f"{photo_source}: {lines[line_index - 1]}"
            )
        try:
            atomic_number = int(fields[1])
            ion_stage = int(fields[2])
            source_level = int(fields[3])
            threshold_ev = float(fields[5])
            point_count = int(fields[6])
        except ValueError as error:
            raise ValueError(
                f"malformed SIROCCO photoionization header in "
                f"{photo_source}: {lines[line_index - 1]}"
            ) from error
        energies: list[float] = []
        cross_sections: list[float] = []
        for _ in range(point_count):
            if line_index >= len(lines):
                raise ValueError(
                    f"truncated SIROCCO photoionization section in {photo_source}"
                )
            point = lines[line_index].split()
            line_index += 1
            if len(point) != 3 or point[0] != "PhotMac":
                raise ValueError(
                    f"malformed SIROCCO photoionization point in "
                    f"{photo_source}: {lines[line_index - 1]}"
                )
            energies.append(float(point[1]))
            cross_sections.append(float(point[2]))
        if atomic_number != 8 or ion_stage != 6 or source_level not in level_terms:
            continue
        energy = np.asarray(energies, dtype=np.float64)
        cross_section = np.asarray(cross_sections, dtype=np.float64)
        if (
            threshold_ev <= 0.0
            or np.any(~np.isfinite(energy))
            or np.any(~np.isfinite(cross_section))
            or np.any(energy <= 0.0)
            or np.any(cross_section <= 0.0)
            or np.any(np.diff(energy) <= 0.0)
        ):
            raise ValueError(
                f"invalid SIROCCO photoionization section for level "
                f"{source_level} in {photo_source}"
            )
        key = level_terms[source_level]
        section = (threshold_ev, energy, cross_section)
        if key in sections:
            existing = sections[key]
            # Fine-structure components must represent the same LS-term
            # cross-section before it is safe to collapse them.
            if (
                not np.isclose(existing[0], threshold_ev, rtol=2.0e-3)
                or existing[1].shape != energy.shape
                or not np.allclose(existing[1], energy, rtol=2.0e-3)
                or not np.allclose(existing[2], cross_section, rtol=2.0e-3)
            ):
                raise ValueError(
                    "fine-structure TOPbase sections disagree for "
                    f"n={key[0]} {key[1]} in {photo_source}"
                )
        else:
            sections[key] = section

    if not sections:
        raise ValueError(
            f"{photo_source} contains no O VI TOPbase sections matched by "
            f"{level_source}"
        )
    if population_ion.element != "O" or population_ion.charge != 5:
        raise ValueError("SIROCCO O VI data require an O +5 population ion")
    if population_ion.ionization_energy_ev is None:
        raise ValueError("the population ion has no O VI ionization energy")

    thresholds: list[float] = []
    weights: list[float] = []
    labels: list[str] = []
    sigma0: list[float] = []
    frequency_fits: list[FloatArray | None] = []
    cross_section_fits: list[FloatArray | None] = []
    for level in population_ion.levels:
        principal_quantum_number = _term_principal_quantum_number(level.label)
        orbital = _term_orbital_letter(level.label)
        if principal_quantum_number is None or orbital is None:
            continue
        section = sections.get((principal_quantum_number, orbital.upper()))
        if section is None:
            continue
        source_threshold_ev, energy, cross_section = section
        model_threshold_ev = (
            float(population_ion.ionization_energy_ev)
            - level.energy_wavenumber * PLANCK * LIGHT_SPEED / EV_TO_ERG
        )
        if model_threshold_ev <= 0.0:
            continue
        log_ratio = np.log10(energy / source_threshold_ev)
        log_sigma_megabar = np.log10(cross_section / 1.0e-18)
        if log_ratio[0] > 0.0:
            # SIROCCO stores an explicitly positive threshold section, but
            # rounds the header threshold and first photon-energy sample
            # independently.  Encode the intended threshold value directly
            # so it is not mistaken for TLUSTY's explicit delayed-opening
            # convention by the shared fit evaluator.
            log_ratio = np.concatenate((np.asarray((0.0,)), log_ratio))
            log_sigma_megabar = np.concatenate((
                np.asarray((log_sigma_megabar[0],)), log_sigma_megabar
            ))
        thresholds.append(model_threshold_ev * EV_TO_ERG / PLANCK)
        weights.append(level.statistical_weight)
        labels.append(level.label)
        sigma0.append(
            float(1.0e-18 * 10.0 ** np.interp(0.0, log_ratio, log_sigma_megabar))
        )
        frequency_fits.append(log_ratio)
        cross_section_fits.append(log_sigma_megabar)

    if not thresholds:
        raise ValueError(
            f"no SIROCCO O VI TOPbase sections match {population_ion.source}"
        )
    line_wavelength = (
        np.asarray((), dtype=np.float64)
        if line_collision_data is None
        else np.ascontiguousarray(
            line_collision_data.line_wavelength_vacuum_angstrom
        )
    )
    line_gbar = (
        np.asarray((), dtype=np.float64)
        if line_collision_data is None
        else np.ascontiguousarray(line_collision_data.line_collision_gbar)
    )
    collision_source = (
        ""
        if line_collision_data is None
        else "; bound-bound collision g-bars retained from "
        + line_collision_data.source
    )
    return TlustyPhotoionizationThresholdData(
        threshold_frequency_hz=np.asarray(thresholds, dtype=np.float64),
        statistical_weight=np.asarray(weights, dtype=np.float64),
        level_label=tuple(labels),
        threshold_cross_section_cm2=np.asarray(sigma0, dtype=np.float64),
        log_frequency_ratio=tuple(frequency_fits),
        log_cross_section_megabar=tuple(cross_section_fits),
        photoionization_formula=tuple(0 for _ in thresholds),
        photoionization_parameters=tuple(None for _ in thresholds),
        line_wavelength_vacuum_angstrom=line_wavelength,
        line_collision_gbar=line_gbar,
        source=(
            "SIROCCO level-resolved TOPbase photoionization data: "
            f"{photo_source.name}{collision_source}"
        ),
        threshold_cross_section_includes_gbar=False,
    )


def read_norad_oxygen_vi_photoionization_data(
    energy_path: str | Path,
    partial_photoionization_path: str | Path,
    population_ion: AtomicIon,
    *,
    line_collision_data: TlustyPhotoionizationThresholdData | None = None,
) -> TlustyPhotoionizationThresholdData:
    """Read Nahar's level-specific LS O VI photoionization cross sections.

    NORAD's ``o6.ptpx.ls`` file contains partial cross sections leaving the
    O VII core in its ground state.  That is the correct channel for the
    one-parent O VI population atom; using the published *total* cross
    section would silently assign excited-core ionization to the O VII
    ground level.  The external level energies are used only to identify
    ``n,l`` and define ``nu/nu_threshold``.  Each table is then rebased onto
    the corresponding TMAD threshold, preserving the atmosphere model's
    internally consistent level energies.
    """

    if population_ion.element != "O" or population_ion.charge != 5:
        raise ValueError("NORAD O VI data require an O +5 population ion")
    if population_ion.ionization_energy_ev is None:
        raise ValueError("the population ion has no O VI ionization energy")

    def integer_fields(line: str, count: int) -> tuple[int, ...] | None:
        fields = line.split()
        if len(fields) != count:
            return None
        try:
            return tuple(int(value) for value in fields)
        except ValueError:
            return None

    energy_lines = Path(energy_path).read_text(encoding="ascii").splitlines()
    try:
        cursor = next(
            index
            for index, line in enumerate(energy_lines)
            if integer_fields(line, 3) == (8, 2, 0)
            or line.split() == ["8", "2", "E"]
        ) + 1
    except StopIteration as error:
        raise ValueError(f"{energy_path} is not a NORAD O VI LS energy file") from error

    state_by_symmetry: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    while cursor < len(energy_lines):
        symmetry = integer_fields(energy_lines[cursor], 4)
        cursor += 1
        if symmetry is None:
            continue
        multiplicity, angular_momentum, parity, state_count = symmetry
        if symmetry == (0, 0, 0, 0):
            break
        if state_count < 0 or cursor >= len(energy_lines):
            raise ValueError(f"malformed NORAD O VI symmetry in {energy_path}")
        # The next record identifies the target/core state for this symmetry.
        cursor += 1
        for _ in range(state_count):
            if cursor >= len(energy_lines):
                raise ValueError(f"truncated NORAD O VI energy file {energy_path}")
            fields = energy_lines[cursor].split()
            cursor += 1
            if len(fields) != 6:
                raise ValueError(
                    f"malformed NORAD O VI energy record in {energy_path}: "
                    + " ".join(fields)
                )
            try:
                state_index = int(fields[0])
                core_index = int(fields[2])
                principal_quantum_number = int(fields[3])
                orbital_angular_momentum = int(fields[4])
                energy_rydberg = float(fields[5])
            except ValueError as error:
                raise ValueError(
                    f"malformed NORAD O VI energy record in {energy_path}: "
                    + " ".join(fields)
                ) from error
            if fields[1].upper() == "T" and core_index == 1 and energy_rydberg < 0.0:
                state_by_symmetry[(
                    multiplicity,
                    angular_momentum,
                    parity,
                    state_index,
                )] = (principal_quantum_number, orbital_angular_momentum)

    photo_lines = Path(partial_photoionization_path).read_text(
        encoding="ascii"
    ).splitlines()
    try:
        cursor = next(
            index
            for index, line in enumerate(photo_lines)
            if line.split() == ["8", "2", "P"]
        ) + 1
    except StopIteration as error:
        raise ValueError(
            f"{partial_photoionization_path} is not a NORAD O VI LS "
            "photoionization file"
        ) from error

    sections: dict[tuple[int, int], tuple[float, FloatArray, FloatArray]] = {}
    while cursor < len(photo_lines):
        header = integer_fields(photo_lines[cursor], 4)
        cursor += 1
        if header is None:
            continue
        if header == (0, 0, 0, 0):
            break
        if cursor + 1 >= len(photo_lines):
            raise ValueError(
                f"truncated NORAD O VI section in {partial_photoionization_path}"
            )
        counts = photo_lines[cursor].split()
        cursor += 1
        if len(counts) != 3:
            raise ValueError(
                f"malformed NORAD O VI point count in "
                f"{partial_photoionization_path}"
            )
        point_count = int(counts[1])
        binding_fields = photo_lines[cursor].split()
        cursor += 1
        if len(binding_fields) != 2:
            raise ValueError(
                f"malformed NORAD O VI binding energy in "
                f"{partial_photoionization_path}"
            )
        binding_rydberg = float(binding_fields[0])
        energy = np.empty(point_count, dtype=np.float64)
        cross_section = np.empty(point_count, dtype=np.float64)
        for point_index in range(point_count):
            if cursor >= len(photo_lines):
                raise ValueError(
                    f"truncated NORAD O VI cross section in "
                    f"{partial_photoionization_path}"
                )
            fields = photo_lines[cursor].split()
            cursor += 1
            if len(fields) != 2:
                raise ValueError(
                    f"malformed NORAD O VI cross-section point in "
                    f"{partial_photoionization_path}"
                )
            energy[point_index], cross_section[point_index] = map(float, fields)
        state = state_by_symmetry.get(header)
        if state is None:
            continue
        valid = (
            np.isfinite(energy)
            & np.isfinite(cross_section)
            & (energy >= binding_rydberg)
            & (cross_section > 0.0)
        )
        energy = energy[valid]
        cross_section = cross_section[valid]
        if (
            binding_rydberg <= 0.0
            or energy.size < 2
            or np.any(np.diff(energy) <= 0.0)
        ):
            raise ValueError(
                f"invalid NORAD O VI section {header} in "
                f"{partial_photoionization_path}"
            )
        sections[state] = (binding_rydberg, energy, cross_section)

    thresholds: list[float] = []
    weights: list[float] = []
    labels: list[str] = []
    sigma0: list[float] = []
    frequency_fits: list[FloatArray] = []
    cross_section_fits: list[FloatArray] = []
    for level in population_ion.levels:
        n = _term_principal_quantum_number(level.label)
        orbital = _term_orbital_letter(level.label)
        if n is None or orbital is None:
            continue
        angular_momentum = _RYDBERG_LABEL_ANGULAR_MOMENTUM.get(orbital.upper())
        if angular_momentum is None or (n, angular_momentum) not in sections:
            continue
        binding_rydberg, energy, cross_section_megabar = sections[(
            n, angular_momentum
        )]
        model_threshold_ev = (
            float(population_ion.ionization_energy_ev)
            - level.energy_wavenumber * PLANCK * LIGHT_SPEED / EV_TO_ERG
        )
        if model_threshold_ev <= 0.0:
            continue
        log_ratio = np.log10(energy / binding_rydberg)
        log_sigma = np.log10(cross_section_megabar)
        if log_ratio[0] > 0.0:
            # NORAD sections begin with a positive sample at their bound-state
            # threshold.  Preserve that left boundary explicitly if the two
            # independently printed energies differ by rounding.
            log_ratio = np.concatenate((np.asarray((0.0,)), log_ratio))
            log_sigma = np.concatenate((
                np.asarray((log_sigma[0],)), log_sigma
            ))
        thresholds.append(model_threshold_ev * EV_TO_ERG / PLANCK)
        weights.append(float(level.statistical_weight))
        labels.append(level.label)
        sigma0.append(float(1.0e-18 * 10.0 ** np.interp(0.0, log_ratio, log_sigma)))
        frequency_fits.append(np.ascontiguousarray(log_ratio))
        cross_section_fits.append(np.ascontiguousarray(log_sigma))

    if not thresholds:
        raise ValueError(
            f"no NORAD O VI sections match {population_ion.source}"
        )
    return TlustyPhotoionizationThresholdData(
        threshold_frequency_hz=np.asarray(thresholds, dtype=np.float64),
        statistical_weight=np.asarray(weights, dtype=np.float64),
        level_label=tuple(labels),
        threshold_cross_section_cm2=np.asarray(sigma0, dtype=np.float64),
        log_frequency_ratio=tuple(frequency_fits),
        log_cross_section_megabar=tuple(cross_section_fits),
        photoionization_formula=tuple(0 for _ in thresholds),
        photoionization_parameters=tuple(None for _ in thresholds),
        line_wavelength_vacuum_angstrom=(
            np.asarray((), dtype=np.float64)
            if line_collision_data is None
            else np.ascontiguousarray(
                line_collision_data.line_wavelength_vacuum_angstrom
            )
        ),
        line_collision_gbar=(
            np.asarray((), dtype=np.float64)
            if line_collision_data is None
            else np.ascontiguousarray(line_collision_data.line_collision_gbar)
        ),
        source=(
            "NORAD/Nahar level-specific O VI partial photoionization to "
            "O VII ground: "
            f"{Path(partial_photoionization_path).name}"
        ),
        threshold_cross_section_includes_gbar=False,
    )


def read_tlusty_forbidden_collision_strengths(
    path: str | Path,
    atomic_database: AtomicDatabase,
    element: str,
    charge: int,
    *,
    threshold_relative_tolerance: float = 0.02,
) -> Mapping[
    tuple[str, int, int, int], ConstantEffectiveCollisionStrength
]:
    """Map TLUSTY ``ICOL=4`` collision-only links onto a TMAD term atom.

    The public TLUSTY light-metal atoms include low forbidden links with a
    constant effective collision strength even when no radiative transition
    exists.  TMAD's compact RBB/CBB representation can omit those pairs.  A
    level is matched by continuum threshold, statistical weight, and the LS
    fields present in both labels; ambiguous matches are rejected.
    """

    source_path = Path(path)
    lines = source_path.read_text(encoding="ascii").splitlines()
    try:
        level_start = next(
            index for index, line in enumerate(lines)
            if line.startswith("****** Levels")
        ) + 1
        continuum_start = next(
            index for index, line in enumerate(lines)
            if line.startswith("****** Continuum transitions")
        )
        line_start = next(
            index for index, line in enumerate(lines[continuum_start + 1 :], continuum_start + 1)
            if "Line transitions" in line
        ) + 1
    except StopIteration as error:
        raise ValueError(f"{path} is not a supported TLUSTY model atom") from error

    tlusty_levels: list[tuple[float, float, str]] = []
    for line in lines[level_start:continuum_start]:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            threshold = float(fields[0])
            weight = float(fields[1])
        except ValueError as error:
            raise ValueError(
                f"malformed TLUSTY level record in {path}: {line}"
            ) from error
        label_match = re.search(r"'([^']*)'", line)
        tlusty_levels.append((
            threshold,
            weight,
            "" if label_match is None else label_match.group(1).strip(),
        ))

    symbol = element.strip().capitalize()
    ion = atomic_database.ions.get((symbol, int(charge)))
    if ion is None:
        raise ValueError(
            f"model atom has no {symbol} ion with charge {int(charge)}"
        )
    ionization_frequency = (
        float(ion.ionization_energy_ev) * EV_TO_ERG / PLANCK
    )
    target_threshold = np.asarray([
        ionization_frequency - level.energy_wavenumber * LIGHT_SPEED
        for level in ion.levels
    ])

    def matched_level(record: tuple[float, float, str]) -> int | None:
        threshold, weight, label = record
        relative = np.abs(target_threshold / threshold - 1.0)
        candidates = relative <= float(threshold_relative_tolerance)
        candidates &= np.isclose(
            np.asarray([level.statistical_weight for level in ion.levels]),
            weight,
            rtol=0.0,
            atol=1.0e-5,
        )
        for index, level in enumerate(ion.levels):
            if not candidates[index]:
                continue
            label_match = _spectroscopic_term_labels_match(level.label, label)
            if label_match is False:
                candidates[index] = False
        indices = np.flatnonzero(candidates)
        if indices.size != 1:
            return None
        return int(ion.levels[int(indices[0])].index)

    level_mapping = tuple(matched_level(record) for record in tlusty_levels)
    result: dict[
        tuple[str, int, int, int], ConstantEffectiveCollisionStrength
    ] = {}
    for line in lines[line_start:]:
        fields = line.split()
        if len(fields) != 9:
            continue
        try:
            lower = int(fields[0])
            upper = int(fields[1])
            collision_type = int(fields[4])
            collision_strength = float(fields[8])
        except ValueError:
            continue
        if (
            collision_type != 4
            or collision_strength <= 0.0
            or not (1 <= lower <= len(level_mapping))
            or not (1 <= upper <= len(level_mapping))
        ):
            continue
        mapped_lower = level_mapping[lower - 1]
        mapped_upper = level_mapping[upper - 1]
        if mapped_lower is None or mapped_upper is None or mapped_lower == mapped_upper:
            continue
        lower_index, upper_index = sorted(
            (mapped_lower, mapped_upper),
            key=lambda index: next(
                level.energy_wavenumber for level in ion.levels
                if level.index == index
            ),
        )
        result[(symbol, int(charge), lower_index, upper_index)] = (
            ConstantEffectiveCollisionStrength(
                value=collision_strength,
                source=f"TLUSTY ICOL=4: {source_path.name}",
            )
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class TmadLTEBoundBoundCoupling:
    """One TMAD line connecting an explicit term to an LTE reservoir term.

    The LTE term is not an independent statistical-equilibrium unknown.  Its
    population follows the departure coefficient of ``lte_parent_key``.  The
    rate solver can therefore collapse this physical transition into an
    effective pair between ``explicit_level_key`` and that parent without
    discarding the radiative or collisional cascade through the LTE term.
    """

    explicit_level_key: tuple[str, int, int]
    lte_parent_key: tuple[str, int, int]
    explicit_is_lower: bool
    lte_charge: int
    lte_energy_wavenumber: float
    lte_statistical_weight: float
    lte_level_label: str
    transition: AtomicTransition
    collision_record: tuple[int, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class TmadRadiativeDielectronicRecord:
    """One raw TMAD effective dielectronic-transition declaration.

    TMAP eliminates the autoionizing upper state from its explicit rate
    vector.  The raw record is retained separately from ordinary RBB lines so
    it cannot be mistaken for a bound NLTE transition and so the future
    effective-rate implementation has a typed input.
    """

    lower_level_label: str
    autoionizing_level_label: str
    oscillator_strength: float
    formula: int


@dataclass(frozen=True)
class TmadEffectiveDielectronicCoupling:
    """Resolved TMAP effective DR/autoionization rate pair.

    The eliminated autoionizing state supplies only the stabilizing
    transition frequency and oscillator strength.  Statistical equilibrium
    therefore contains a direct pair between the retained lower-ion level
    and the named continuum parent.  This is the Mihalas--Hummer ``RDI``
    formulation used by TMAP, not an ordinary bound-bound line.
    """

    lower_level_key: tuple[str, int, int]
    continuum_parent_key: tuple[str, int, int]
    transition_frequency_hz: float
    oscillator_strength: float
    lower_level_label: str
    autoionizing_level_label: str


@dataclass(frozen=True)
class ChiantiTermCollisionStrength:
    """Fine-component CHIANTI upsilons summed onto one LS term pair."""

    components: tuple[ChiantiScaledCollisionComponent, ...]
    source: str

    def effective_collision_strength(self, temperature: float) -> float:
        return float(sum(
            component.effective_collision_strength(temperature)
            for component in self.components
        ))


@dataclass(frozen=True)
class ConstantEffectiveCollisionStrength:
    """A temperature-independent effective collision strength.

    TLUSTY model atoms use ``ICOL=4`` with an explicit constant ``Omega``
    for low forbidden and intercombination links.  These transitions have no
    radiative record, so they must be carried independently into a reduced
    statistical-equilibrium atom instead of being inferred from an
    oscillator strength.
    """

    value: float
    source: str

    def effective_collision_strength(self, temperature: float) -> float:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        return float(max(self.value, 0.0))


@dataclass(frozen=True)
class TabulatedElectronExcitationRateCoefficient:
    """Temperature-dependent upward electron-excitation coefficient.

    Some R-matrix publications tabulate Maxwellian rate coefficients directly
    rather than effective collision strengths.  Retaining that representation
    avoids a lossy or opaque conversion and lets the statistical-equilibrium
    solver impose the inverse coefficient by detailed balance.  Positive
    rates are interpolated logarithmically and clipped at the tabulated
    temperature limits.
    """

    temperature_kelvin: tuple[float, ...]
    upward_rate_coefficient_cm3_s: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        grid = self.temperature_kelvin
        rate = self.upward_rate_coefficient_cm3_s
        if (
            len(grid) < 2
            or len(grid) != len(rate)
            or any(not math.isfinite(value) or value <= 0.0 for value in grid)
            or any(not math.isfinite(value) or value <= 0.0 for value in rate)
            or any(right <= left for left, right in zip(grid, grid[1:]))
        ):
            raise ValueError(
                "tabulated excitation rates must be positive and ordered"
            )

    def upward_rate_coefficient(self, temperature: float) -> float:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        grid = self.temperature_kelvin
        rate = self.upward_rate_coefficient_cm3_s
        if temperature <= grid[0]:
            return float(rate[0])
        if temperature >= grid[-1]:
            return float(rate[-1])
        lower = bisect_right(grid, temperature) - 1
        weight = math.log(temperature / grid[lower]) / math.log(
            grid[lower + 1] / grid[lower]
        )
        return float(math.exp(
            math.log(rate[lower])
            + weight * math.log(rate[lower + 1] / rate[lower])
        ))


_BARKLEM_OI_TEMPERATURE_K = (
    1_000.0, 3_000.0, 5_000.0, 8_000.0, 12_000.0, 20_000.0, 50_000.0,
)

# Upward LS-term rate coefficients from Barklem (2007), A&A 462, 781,
# Table 4. Terms are ordered 2p4 3P, 2p4 1D, 2p4 1S, 3s 5So, 3s 3So,
# 3p 5P, and 3p 3P. Exact zeros in the LS calculation are omitted.
_BARKLEM_OI_TERM_RATE_CM3_S = MappingProxyType({
    (0, 1): (5.72e-21, 2.98e-13, 1.38e-11, 1.24e-10, 4.31e-10, 1.18e-9, 2.71e-9),
    (0, 2): (5.13e-31, 4.28e-17, 3.07e-14, 1.34e-12, 1.10e-11, 5.84e-11, 2.35e-10),
    (0, 3): (1.03e-55, 9.33e-25, 1.30e-18, 3.42e-15, 2.53e-13, 7.37e-12, 1.32e-10),
    (0, 4): (3.40e-57, 3.46e-25, 7.53e-19, 2.66e-15, 2.47e-13, 9.56e-12, 3.04e-10),
    (0, 5): (6.35e-64, 9.03e-28, 1.53e-20, 1.72e-16, 2.99e-14, 1.91e-12, 8.95e-11),
    (0, 6): (5.48e-65, 4.89e-28, 1.26e-20, 1.88e-16, 3.96e-14, 3.00e-12, 1.77e-10),
    (1, 2): (1.91e-20, 4.17e-13, 1.24e-11, 8.25e-11, 2.31e-10, 5.15e-10, 1.05e-9),
    (1, 4): (3.34e-49, 6.87e-24, 6.11e-19, 3.75e-16, 1.50e-14, 3.24e-13, 5.85e-12),
    (1, 6): (6.34e-56, 9.66e-26, 9.93e-20, 2.63e-16, 2.18e-14, 8.02e-13, 2.57e-11),
    (2, 6): (8.85e-45, 4.53e-22, 1.41e-17, 7.98e-15, 3.30e-13, 6.05e-12, 6.50e-11),
    (3, 4): (9.98e-10, 2.36e-8, 3.89e-8, 4.57e-8, 4.48e-8, 3.74e-8, 2.03e-8),
    (3, 5): (1.11e-15, 6.23e-10, 1.02e-8, 4.97e-8, 1.20e-7, 2.38e-7, 4.12e-7),
    (3, 6): (1.98e-17, 4.43e-11, 8.34e-10, 3.95e-9, 8.36e-9, 1.30e-8, 1.26e-8),
    (4, 5): (9.91e-14, 1.65e-9, 1.09e-8, 2.86e-8, 4.38e-8, 5.25e-8, 4.02e-8),
    (4, 6): (3.44e-15, 9.26e-10, 1.29e-8, 5.65e-8, 1.26e-7, 2.31e-7, 3.62e-7),
    (5, 6): (2.56e-9, 3.51e-8, 5.83e-8, 6.98e-8, 6.89e-8, 5.81e-8, 3.23e-8),
})


def barklem_oi_electron_collision_data(
    atomic_database: AtomicDatabase,
) -> Mapping[
    tuple[str, int, int, int], TabulatedElectronExcitationRateCoefficient
]:
    """Project Barklem's seven-term O I R-matrix rates onto fine levels.

    Barklem (2007) tabulates LS-term excitation coefficients. Within each
    nearly degenerate upper term, the total coefficient is distributed in
    proportion to final-state statistical weight. Applying that same sum from
    every lower fine component preserves the published term rate when the
    lower term has a statistical population, while the solver retains exact
    fine-level detailed balance.
    """

    ion = atomic_database.ions.get(("O", 0))
    if ion is None:
        raise ValueError("atomic database has no O I ion")
    predicates = (
        lambda label: ".2p4.(3P<" in label,
        lambda label: ".2p4.(1D<" in label,
        lambda label: ".2p4.(1S<" in label,
        lambda label: ".3s.(5So<" in label,
        lambda label: ".3s.(3So<" in label,
        lambda label: ".3p.(5P<" in label,
        lambda label: ".3p.(3P<" in label,
    )
    terms = tuple(tuple(
        level for level in ion.levels if predicate(level.label)
    ) for predicate in predicates)
    expected_weights = (9.0, 5.0, 1.0, 5.0, 3.0, 15.0, 9.0)
    term_weights = tuple(sum(
        level.statistical_weight for level in levels
    ) for levels in terms)
    if any(not levels for levels in terms) or not np.allclose(
        term_weights, expected_weights, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "O I fine levels do not match Barklem's seven LS terms: "
            f"weights={term_weights!r}"
        )

    source = "Barklem (2007) A&A 462, 781, Table 4 R-matrix rates"
    output = {}
    for (lower_term, upper_term), term_rates in (
        _BARKLEM_OI_TERM_RATE_CM3_S.items()
    ):
        upper_weight = term_weights[upper_term]
        for lower in terms[lower_term]:
            for upper in terms[upper_term]:
                fine_fraction = upper.statistical_weight / upper_weight
                output[("O", 0, lower.index, upper.index)] = (
                    TabulatedElectronExcitationRateCoefficient(
                        temperature_kelvin=_BARKLEM_OI_TEMPERATURE_K,
                        upward_rate_coefficient_cm3_s=tuple(
                            value * fine_fraction for value in term_rates
                        ),
                        source=source,
                    )
                )
    return MappingProxyType(output)


def read_barklem_mgi_electron_collision_data(
    directory: str | Path,
    atomic_database: AtomicDatabase,
    *,
    calculation: str = "ccc",
) -> Mapping[
    tuple[str, int, int, int], TabulatedElectronExcitationRateCoefficient
]:
    """Read Barklem et al. (2017) Mg I close-coupling collision rates.

    The CDS ``J/A+A/606/A11`` delivery tabulates LS-term effective collision
    strengths for the 25 lowest Mg I terms.  The selected CCC or BSR matrices
    are converted to upward rate coefficients using the published term
    energies.  Each term rate is then projected onto Stout fine levels so its
    sum is preserved exactly for every lower fine component.  Reverse rates
    retain exact fine-level detailed balance in the population solver.
    """

    method = calculation.strip().lower()
    suffix_by_method = {"ccc": "c", "bsr": "b"}
    if method not in suffix_by_method:
        raise ValueError("calculation must be 'ccc' or 'bsr'")
    root = Path(directory)
    states_path = root / "states.dat"
    states = []
    for line in states_path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(f"malformed Barklem Mg I state: {line}")
        index, label, weight, energy_ev = fields
        states.append((int(index), label, float(weight), float(energy_ev)))
    if len(states) < 2 or [row[0] for row in states] != list(
        range(1, len(states) + 1)
    ):
        raise ValueError("Barklem Mg I states must be sequential and nonempty")

    ion = atomic_database.ions.get(("Mg", 0))
    if ion is None:
        raise ValueError("atomic database has no Mg I ion")

    terms = []
    for _, label, expected_weight, _ in states:
        try:
            configuration, term = label.split("_", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"unsupported Barklem Mg I label: {label}") from error
        if configuration == "3s2":
            configuration_prefix = "3s2."
        elif configuration.startswith("3s"):
            configuration_prefix = f"3s.{configuration[2:]}."
        else:
            raise ValueError(f"unsupported Barklem Mg I configuration: {label}")
        term_prefix = f"({term[0]}{term[1]}"
        levels = tuple(
            level
            for level in ion.levels
            if (
                level.label.startswith(configuration_prefix)
                and term_prefix in level.label
            )
        )
        actual_weight = sum(level.statistical_weight for level in levels)
        if not levels or not np.isclose(
            actual_weight, expected_weight, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"Mg I levels do not match Barklem term {label}: "
                f"weight={actual_weight!r}, expected={expected_weight!r}"
            )
        terms.append(levels)

    suffix = suffix_by_method[method]
    table_pattern = re.compile(rf"^(\d+)k_{suffix}\.dat$")
    temperature_and_matrix = []
    for path in root.iterdir():
        match = table_pattern.match(path.name)
        if match is None:
            continue
        matrix = np.loadtxt(path, dtype=np.float64)
        expected_shape = (len(states), len(states))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{path} has shape {matrix.shape}, expected {expected_shape}"
            )
        if (
            np.any(~np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"{path} is not a finite symmetric rate matrix")
        temperature_and_matrix.append((float(match.group(1)), matrix))
    temperature_and_matrix.sort(key=lambda item: item[0])
    if len(temperature_and_matrix) < 2:
        raise ValueError(
            f"fewer than two Barklem Mg I {method.upper()} rate matrices"
        )
    temperature_grid = tuple(item[0] for item in temperature_and_matrix)
    source = (
        f"Barklem et al. (2017) A&A 606, A11, CDS J/A+A/606/A11 "
        f"{method.upper()} rates"
    )
    output = {}
    for lower_term in range(len(states)):
        _, _, lower_weight, lower_energy_ev = states[lower_term]
        for upper_term in range(lower_term + 1, len(states)):
            _, _, _, upper_energy_ev = states[upper_term]
            term_rates = tuple(
                8.63e-6
                * matrix[lower_term, upper_term]
                / (lower_weight * math.sqrt(temperature))
                * math.exp(
                    -(upper_energy_ev - lower_energy_ev)
                    * EV_TO_ERG
                    / (BOLTZMANN * temperature)
                )
                for temperature, matrix in temperature_and_matrix
            )
            if any(rate <= 0.0 for rate in term_rates):
                continue
            upper_weight = sum(
                level.statistical_weight for level in terms[upper_term]
            )
            for lower in terms[lower_term]:
                for upper in terms[upper_term]:
                    fine_fraction = upper.statistical_weight / upper_weight
                    output[("Mg", 0, lower.index, upper.index)] = (
                        TabulatedElectronExcitationRateCoefficient(
                            temperature_kelvin=temperature_grid,
                            upward_rate_coefficient_cm3_s=tuple(
                                value * fine_fraction for value in term_rates
                            ),
                            source=source,
                        )
                    )
    return MappingProxyType(output)


@dataclass(frozen=True)
class PSM20AngularMomentumMixingCollision:
    """One heavy-ion ``nl -> nl'`` angular-momentum mixing link.

    ``higher_l`` identifies the direction for which the final-state-resolved
    modified Pengelly--Seaton rate is evaluated.  The reverse rate is imposed
    by detailed balance in the statistical-equilibrium solver.  The rate is
    density dependent because distant encounters are cut off at the local
    Debye radius; this is therefore not an electron effective collision
    strength.
    """

    principal_quantum_number: int
    higher_l: int
    lower_energy_l: int
    upper_energy_l: int
    energy_splitting_wavenumber: float
    radiative_lifetime_s: float | None
    target_core_charge: float
    target_atomic_mass_u: float
    source: str = "Badnell et al. (2021) PSM20, Debye cut-off"


@dataclass(frozen=True)
class BTMQuadrupoleAngularMomentumMixingCollision:
    """One heavy-ion ``nl -> nl+/-2`` quadrupole mixing link.

    The BTM rate is the analytic quadrupole-impact approximation of
    Deliporanidou et al. (2025).  Unlike PSM20 dipole mixing it is finite
    without a distant-collision cut-off.  The record follows the actual term
    energy ordering so the upward Boltzmann factor and reverse detailed
    balance can be applied consistently by the rate-equation solver.
    """

    principal_quantum_number: int
    lower_energy_l: int
    upper_energy_l: int
    energy_splitting_wavenumber: float
    target_core_charge: float
    target_atomic_mass_u: float
    source: str = "Deliporanidou et al. (2025) BTM quadrupole l mixing"


_ATOMIC_MASS_UNIT_G = 1.660_539_068_92e-24
_HYDROGEN_ATOMIC_MASS_U = 1.007_84
_HELIUM_ATOMIC_MASS_U = 4.002_602
_RYDBERG_ENERGY_ERG = 13.605_693_122_994 * EV_TO_ERG
_EULER_MASCHERONI = 0.577_215_664_901_532_9
_RYDBERG_LABEL_ANGULAR_MOMENTUM = MappingProxyType({
    symbol: angular_momentum
    for angular_momentum, symbol in enumerate("SPDFGHIKLMNOQ")
})
_PSM20_LAGUERRE_ENERGY, _PSM20_LAGUERRE_WEIGHT = (
    np.polynomial.laguerre.laggauss(48)
)


def _exponential_integral_e1(value: float) -> float:
    """Return ``E1(value)`` without adding SciPy as a core dependency."""

    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError("the exponential-integral argument must be positive")
    if x <= 1.0:
        # E1(x) = -gamma - log(x) - sum_k (-x)^k / (k k!).
        term = -x
        series = term
        for order in range(2, 200):
            term *= -x / order
            increment = term / order
            series += increment
            if abs(increment) <= 2.0e-16 * max(abs(series), 1.0):
                break
        return float(-_EULER_MASCHERONI - np.log(x) - series)

    # Lentz continued fraction for Gamma(0, x) = E1(x).
    tiny = np.finfo(np.float64).tiny
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for order in range(1, 200):
        coefficient = -float(order * order)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        change = c * d
        fraction *= change
        if abs(change - 1.0) <= 2.0e-15:
            break
    return float(np.exp(-x) * fraction)


def _psm20_debye_bracket(um: float) -> float:
    """Stable bracketed factor in Badnell et al. (2021), equation 9."""

    value = float(um)
    if value < 1.0e-3:
        # Expanding the two individually divergent terms avoids cancellation.
        return float(
            -_EULER_MASCHERONI
            - np.log(value)
            + 2.0 / 3.0
            + 3.0 / 5.0 * value
            - 3.0 / 28.0 * value**2
            + 1.0 / 54.0 * value**3
        )
    return float(
        np.sqrt(PI)
        / 2.0
        * value**-1.5
        * math.erf(np.sqrt(value))
        - np.exp(-value) / value
        + _exponential_integral_e1(value)
    )


def psm20_debye_l_mixing_rate_coefficient(
    principal_quantum_number: int,
    initial_l: int,
    final_l: int,
    temperature: float,
    electron_density: float,
    target_atomic_mass_u: float,
    collider_charge: float,
    collider_atomic_mass_u: float,
    target_core_charge: float,
) -> float:
    """Return a final-state-resolved heavy-ion l-mixing rate in cm3/s.

    This implements equation (9) and the resolved probability bound in
    equations (16)--(18) of Badnell et al. (2021).  It uses the local Debye
    radius as the distant-collision cut-off.  The caller should evaluate the
    downward-in-``l`` direction and obtain the reverse rate by reciprocity.
    """

    n = int(principal_quantum_number)
    l_initial = int(initial_l)
    l_final = int(final_l)
    if n < 2 or not (0 <= l_initial < n) or not (0 <= l_final < n):
        raise ValueError("n and l must describe bound hydrogenic states")
    if abs(l_final - l_initial) != 1:
        raise ValueError("PSM20 dipole l mixing requires |delta l| = 1")
    if (
        temperature <= 0.0
        or electron_density <= 0.0
        or target_atomic_mass_u <= 0.0
        or collider_charge <= 0.0
        or collider_atomic_mass_u <= 0.0
        or target_core_charge <= 0.0
    ):
        raise ValueError("plasma, mass, and charge inputs must be positive")

    reduced_mass_u = (
        target_atomic_mass_u
        * collider_atomic_mass_u
        / (target_atomic_mass_u + collider_atomic_mass_u)
    )
    reduced_mass_electron = (
        reduced_mass_u * _ATOMIC_MASS_UNIT_G / ELECTRON_MASS
    )
    l_greater = max(l_initial, l_final)
    omega_initial = 2.0 * l_initial + 1.0
    charge_ratio_squared = (collider_charge / target_core_charge) ** 2
    dipole_factor = (
        charge_ratio_squared
        * 6.0
        * n**2
        * l_greater
        * (n**2 - l_greater**2)
    )
    unresolved_dipole = (
        charge_ratio_squared
        * 6.0
        * n**2
        * (n**2 - l_initial**2 - l_initial - 1.0)
    )
    branching_probability = (
        0.5 * dipole_factor / (omega_initial * unresolved_dipole)
    )
    debye_radius_squared = (
        BOLTZMANN
        * temperature
        / (
            8.0
            * PI
            * BOHR_RADIUS
            * _RYDBERG_ENERGY_ERG
            * electron_density
        )
    )
    minimum_energy = (
        BOHR_RADIUS**2
        * reduced_mass_electron
        * _RYDBERG_ENERGY_ERG
        * dipole_factor
        / (
            2.0
            * branching_probability
            * omega_initial
            * debye_radius_squared
        )
    )
    um = minimum_energy / (BOLTZMANN * temperature)
    bohr_time = PLANCK / (4.0 * PI * _RYDBERG_ENERGY_ERG)
    coefficient = (
        BOHR_RADIUS**3
        / bohr_time
        * np.sqrt(
            PI
            * reduced_mass_electron
            * _RYDBERG_ENERGY_ERG
            / (BOLTZMANN * temperature)
        )
        * dipole_factor
        / omega_initial
        * _psm20_debye_bracket(um)
    )
    return float(max(coefficient, 0.0))


def psm20_l_mixing_rate_coefficient(
    principal_quantum_number: int,
    initial_l: int,
    final_l: int,
    temperature: float,
    electron_density: float,
    target_atomic_mass_u: float,
    collider_charge: float,
    collider_atomic_mass_u: float,
    target_core_charge: float,
    *,
    energy_splitting_wavenumber: float = 0.0,
    radiative_lifetime_s: float | None = None,
) -> float:
    """Return a PSM20 rate with Debye and splitting/lifetime cut-offs.

    The cross sections in Badnell et al. (2021), equations (6)--(7), are
    integrated over a Maxwell distribution with fixed Gauss--Laguerre nodes.
    This numerical form is used only when an energy-dependent cut-off is
    tighter than the Debye radius; otherwise the analytic equation (9) result
    is returned exactly.
    """

    if energy_splitting_wavenumber < 0.0:
        raise ValueError("energy splitting must be non-negative")
    if radiative_lifetime_s is not None and radiative_lifetime_s <= 0.0:
        raise ValueError("radiative lifetime must be positive")
    cutoff_time = np.inf
    if energy_splitting_wavenumber > 0.0:
        cutoff_time = min(
            cutoff_time,
            1.12
            * PLANCK
            / (2.0 * PI)
            / (
                PLANCK
                * LIGHT_SPEED
                * energy_splitting_wavenumber
            ),
        )
    if radiative_lifetime_s is not None:
        cutoff_time = min(cutoff_time, 0.72 * radiative_lifetime_s)
    if not np.isfinite(cutoff_time):
        return psm20_debye_l_mixing_rate_coefficient(
            principal_quantum_number,
            initial_l,
            final_l,
            temperature,
            electron_density,
            target_atomic_mass_u,
            collider_charge,
            collider_atomic_mass_u,
            target_core_charge,
        )

    n = int(principal_quantum_number)
    l_initial = int(initial_l)
    l_final = int(final_l)
    if n < 2 or not (0 <= l_initial < n) or not (0 <= l_final < n):
        raise ValueError("n and l must describe bound hydrogenic states")
    if abs(l_final - l_initial) != 1:
        raise ValueError("PSM20 dipole l mixing requires |delta l| = 1")
    if (
        temperature <= 0.0
        or electron_density <= 0.0
        or target_atomic_mass_u <= 0.0
        or collider_charge <= 0.0
        or collider_atomic_mass_u <= 0.0
        or target_core_charge <= 0.0
    ):
        raise ValueError("plasma, mass, and charge inputs must be positive")

    reduced_mass_u = (
        target_atomic_mass_u
        * collider_atomic_mass_u
        / (target_atomic_mass_u + collider_atomic_mass_u)
    )
    reduced_mass_g = reduced_mass_u * _ATOMIC_MASS_UNIT_G
    reduced_mass_electron = reduced_mass_g / ELECTRON_MASS
    l_greater = max(l_initial, l_final)
    omega_initial = 2.0 * l_initial + 1.0
    charge_ratio_squared = (collider_charge / target_core_charge) ** 2
    dipole_factor = (
        charge_ratio_squared
        * 6.0
        * n**2
        * l_greater
        * (n**2 - l_greater**2)
    )
    unresolved_dipole = (
        charge_ratio_squared
        * 6.0
        * n**2
        * (n**2 - l_initial**2 - l_initial - 1.0)
    )
    probability_bound = (
        0.5 * dipole_factor / (omega_initial * unresolved_dipole)
    )
    debye_radius = np.sqrt(
        BOLTZMANN
        * temperature
        / (
            8.0
            * PI
            * BOHR_RADIUS
            * _RYDBERG_ENERGY_ERG
            * electron_density
        )
    )
    energy = _PSM20_LAGUERRE_ENERGY * BOLTZMANN * temperature
    velocity = np.sqrt(2.0 * energy / reduced_mass_g)
    cutoff_radius = np.minimum(debye_radius, velocity * cutoff_time)
    matching_radius_squared = (
        BOHR_RADIUS**2
        * reduced_mass_electron
        * _RYDBERG_ENERGY_ERG
        * dipole_factor
        / (2.0 * omega_initial * probability_bound * energy)
    )
    radius_ratio = cutoff_radius / np.sqrt(matching_radius_squared)
    cross_section = np.empty_like(radius_ratio)
    distant = radius_ratio >= 1.0
    cross_section[distant] = (
        PI
        * probability_bound
        * matching_radius_squared[distant]
        * (2.0 / 3.0 + 2.0 * np.log(radius_ratio[distant]))
    )
    cross_section[~distant] = (
        PI
        * probability_bound
        * matching_radius_squared[~distant]
        * radius_ratio[~distant] ** 3
        * 2.0
        / 3.0
    )
    coefficient = (
        2.0
        / np.sqrt(PI)
        * np.sqrt(2.0 * BOLTZMANN * temperature / reduced_mass_g)
        * np.sum(
            _PSM20_LAGUERRE_WEIGHT
            * cross_section
            * _PSM20_LAGUERRE_ENERGY
        )
    )
    return float(max(coefficient, 0.0))


def btm_quadrupole_l_mixing_rate_coefficient(
    principal_quantum_number: int,
    initial_l: int,
    final_l: int,
    temperature: float,
    target_atomic_mass_u: float,
    collider_charge: float,
    collider_atomic_mass_u: float,
    target_core_charge: float,
    *,
    energy_splitting_wavenumber: float = 0.0,
) -> float:
    """Return a BTM quadrupole heavy-ion l-mixing rate in cm3/s.

    This implements equations (6)--(10) of Deliporanidou et al. (2025).
    ``initial_l`` is the initial term, so its ``2*l+1`` statistical weight is
    used explicitly.  A small measured term splitting may be supplied for an
    upward transition; the published degenerate-shell result is recovered
    when it is zero.
    """

    n = int(principal_quantum_number)
    l_initial = int(initial_l)
    l_final = int(final_l)
    if n < 3 or not (0 <= l_initial < n) or not (0 <= l_final < n):
        raise ValueError("n and l must describe bound hydrogenic states")
    if abs(l_final - l_initial) != 2:
        raise ValueError("BTM quadrupole l mixing requires |delta l| = 2")
    if (
        temperature <= 0.0
        or target_atomic_mass_u <= 0.0
        or collider_charge <= 0.0
        or collider_atomic_mass_u <= 0.0
        or target_core_charge <= 0.0
        or energy_splitting_wavenumber < 0.0
    ):
        raise ValueError("plasma, mass, charge, and splitting inputs are invalid")

    reduced_mass_u = (
        target_atomic_mass_u
        * collider_atomic_mass_u
        / (target_atomic_mass_u + collider_atomic_mass_u)
    )
    reduced_mass_electron = (
        reduced_mass_u * _ATOMIC_MASS_UNIT_G / ELECTRON_MASS
    )
    l_greater = max(l_initial, l_final)
    l_smaller = min(l_initial, l_final)
    oscillator_strength_sum = (
        150.0
        * (n / (2.0 * target_core_charge)) ** 4
        * l_greater
        * (l_greater - 1.0)
        / (2.0 * l_greater - 1.0)
        * (n**2 - l_greater**2)
        * (n**2 - (l_greater - 1.0) ** 2)
    )
    matching_radius_factor = 1.0 + (
        1.0 - l_smaller * (l_smaller + 1.0) / n**2
    ) ** 0.25
    matching_radius = matching_radius_factor * n**2 / target_core_charge
    collision_strength = (
        4.0
        / 5.0
        * (reduced_mass_electron * collider_charge / matching_radius) ** 2
        * oscillator_strength_sum
    )
    bohr_time = PLANCK / (4.0 * PI * _RYDBERG_ENERGY_ERG)
    coefficient = (
        2.0
        / (2.0 * l_initial + 1.0)
        * np.sqrt(
            PI * _RYDBERG_ENERGY_ERG / (BOLTZMANN * temperature)
        )
        * reduced_mass_electron**-1.5
        * collision_strength
        * BOHR_RADIUS**3
        / bohr_time
    )
    if energy_splitting_wavenumber > 0.0:
        coefficient *= np.exp(
            -PLANCK
            * LIGHT_SPEED
            * energy_splitting_wavenumber
            / (BOLTZMANN * temperature)
        )
    return float(max(coefficient, 0.0))


def rydberg_angular_momentum_mixing_collision_data(
    atomic_database: AtomicDatabase,
    element: str,
    charge: int,
    *,
    minimum_principal_quantum_number: int = 7,
    maximum_principal_quantum_number: int | None = None,
    minimum_angular_momentum: int = 3,
) -> Mapping[
    tuple[str, int, int, int], PSM20AngularMomentumMixingCollision
]:
    """Construct adjacent high-l PSM20 links for a term-resolved ion.

    Low-l terms are deliberately excluded by default because their quantum
    defects require a lifetime/splitting cut-off rather than the degenerate
    Debye-only expression.  Labels are interpreted using the TMAD convention
    ``O607F`` = O VI, n=7, l=3.
    """

    symbol = element.strip().capitalize()
    ion = atomic_database.ions[(symbol, int(charge))]
    prefix = f"{symbol.upper()}{int(charge) + 1}"
    by_shell_and_l: dict[tuple[int, int], AtomicLevel] = {}
    for level in ion.levels:
        label = level.label.strip().upper()
        if not label.startswith(prefix):
            continue
        match = re.match(r"(\d{2})([SPDFGHIKLMNOQ])", label[len(prefix):])
        if match is None:
            continue
        n = int(match.group(1))
        angular_momentum = _RYDBERG_LABEL_ANGULAR_MOMENTUM[match.group(2)]
        if n < minimum_principal_quantum_number:
            continue
        if maximum_principal_quantum_number is not None and (
            n > maximum_principal_quantum_number
        ):
            continue
        by_shell_and_l[(n, angular_momentum)] = level

    result = {}
    radiative_decay_rate: dict[int, float] = {}
    for transition in ion.transitions:
        radiative_decay_rate[transition.upper_index] = (
            radiative_decay_rate.get(transition.upper_index, 0.0)
            + max(float(transition.einstein_a), 0.0)
        )
    for (n, angular_momentum), first in by_shell_and_l.items():
        if angular_momentum < minimum_angular_momentum:
            continue
        second = by_shell_and_l.get((n, angular_momentum + 1))
        if second is None:
            continue
        lower, upper = sorted(
            (first, second), key=lambda level: level.energy_wavenumber
        )
        result[(symbol, int(charge), lower.index, upper.index)] = (
            PSM20AngularMomentumMixingCollision(
                principal_quantum_number=n,
                higher_l=angular_momentum + 1,
                lower_energy_l=(
                    angular_momentum
                    if lower is first
                    else angular_momentum + 1
                ),
                upper_energy_l=(
                    angular_momentum + 1
                    if lower is first
                    else angular_momentum
                ),
                energy_splitting_wavenumber=float(
                    upper.energy_wavenumber - lower.energy_wavenumber
                ),
                radiative_lifetime_s=(
                    None
                    if radiative_decay_rate.get(upper.index, 0.0) <= 0.0
                    else 1.0 / radiative_decay_rate[upper.index]
                ),
                target_core_charge=float(int(charge) + 1),
                target_atomic_mass_u=float(ion.atomic_mass_u),
            )
        )
    return MappingProxyType(result)


def rydberg_quadrupole_angular_momentum_mixing_collision_data(
    atomic_database: AtomicDatabase,
    element: str,
    charge: int,
    *,
    minimum_principal_quantum_number: int = 7,
    maximum_principal_quantum_number: int | None = None,
    minimum_angular_momentum: int = 3,
) -> Mapping[
    tuple[str, int, int, int], BTMQuadrupoleAngularMomentumMixingCollision
]:
    """Construct same-shell ``delta-l=2`` BTM heavy-ion collision links."""

    symbol = element.strip().capitalize()
    ion = atomic_database.ions[(symbol, int(charge))]
    prefix = f"{symbol.upper()}{int(charge) + 1}"
    by_shell_and_l: dict[tuple[int, int], AtomicLevel] = {}
    for level in ion.levels:
        label = level.label.strip().upper()
        if not label.startswith(prefix):
            continue
        match = re.match(r"(\d{2})([SPDFGHIKLMNOQ])", label[len(prefix):])
        if match is None:
            continue
        n = int(match.group(1))
        angular_momentum = _RYDBERG_LABEL_ANGULAR_MOMENTUM[match.group(2)]
        if n < minimum_principal_quantum_number:
            continue
        if maximum_principal_quantum_number is not None and (
            n > maximum_principal_quantum_number
        ):
            continue
        by_shell_and_l[(n, angular_momentum)] = level

    result = {}
    for (n, angular_momentum), first in by_shell_and_l.items():
        if angular_momentum < minimum_angular_momentum:
            continue
        second = by_shell_and_l.get((n, angular_momentum + 2))
        if second is None:
            continue
        lower, upper = sorted(
            (first, second), key=lambda level: level.energy_wavenumber
        )
        result[(symbol, int(charge), lower.index, upper.index)] = (
            BTMQuadrupoleAngularMomentumMixingCollision(
                principal_quantum_number=n,
                lower_energy_l=(
                    angular_momentum
                    if lower is first
                    else angular_momentum + 2
                ),
                upper_energy_l=(
                    angular_momentum + 2
                    if lower is first
                    else angular_momentum
                ),
                energy_splitting_wavenumber=float(
                    upper.energy_wavenumber - lower.energy_wavenumber
                ),
                target_core_charge=float(int(charge) + 1),
                target_atomic_mass_u=float(ion.atomic_mass_u),
            )
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class TmadStructureModelAtom:
    """TMAD term atom plus its mapping to formal fine-structure levels."""

    atomic_database: AtomicDatabase
    levels_per_charge: Mapping[int, int]
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ]
    formal_lte_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ]
    continuum_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ]
    lte_level_reservoir: Mapping[
        tuple[str, int, int], tuple[tuple[int, float, float], ...]
    ]
    lte_bound_bound_couplings: tuple[TmadLTEBoundBoundCoupling, ...]
    collision_data: Mapping[
        tuple[str, int, int, int],
        tuple[int, tuple[float, ...]]
        | ChiantiTermCollisionStrength
        | ConstantEffectiveCollisionStrength
        | TabulatedElectronExcitationRateCoefficient
        | PSM20AngularMomentumMixingCollision
        | BTMQuadrupoleAngularMomentumMixingCollision,
    ]
    source: str
    autoionizing_lte_level_labels: tuple[str, ...] = ()
    radiative_dielectronic_records: tuple[
        TmadRadiativeDielectronicRecord, ...
    ] = ()
    effective_dielectronic_couplings: tuple[
        TmadEffectiveDielectronicCoupling, ...
    ] = ()


def read_chianti_term_collision_strengths(
    path: str | Path,
    model_atom: TmadStructureModelAtom,
    element: str,
    charge: int,
) -> Mapping[
    tuple[str, int, int, int], ChiantiTermCollisionStrength
]:
    """Project CHIANTI fine-level SCUPS data onto a TMAD LS-term atom.

    Effective collision strengths add over fine-structure components when
    the sublevels of each LS term share a statistical population.  The
    existing TMAD-to-formal mapping supplies that projection without relying
    on labels or coincident line wavelengths.
    """

    source_path = Path(path)
    symbol = element.strip().capitalize()
    reverse_mapping: dict[
        tuple[str, int, int], tuple[str, int, int]
    ] = {}
    for population_key, formal_keys in model_atom.formal_level_mapping.items():
        if population_key[:2] != (symbol, int(charge)):
            continue
        for formal_key in formal_keys:
            if formal_key in reverse_mapping:
                raise ValueError(
                    f"formal level {formal_key} maps to more than one population term"
                )
            reverse_mapping[formal_key] = population_key

    lines = source_path.read_text(encoding="ascii").splitlines()
    grouped: dict[
        tuple[str, int, int, int], list[ChiantiScaledCollisionComponent]
    ] = {}
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].strip()
        cursor += 1
        if not header or header.startswith("%"):
            continue
        if header == "-1":
            break
        fields = header.split()
        if len(fields) < 8 or cursor + 1 >= len(lines):
            raise ValueError(f"malformed CHIANTI SCUPS header in {path}: {header}")
        try:
            lower_index = int(fields[0])
            upper_index = int(fields[1])
            energy = float(fields[2])
            node_count = int(fields[5])
            transition_type = int(fields[6])
            scaling_parameter = float(fields[7])
            scaled_temperature = tuple(
                float(value) for value in lines[cursor].split()
            )
            scaled_upsilon = tuple(
                float(value) for value in lines[cursor + 1].split()
            )
        except ValueError as error:
            raise ValueError(
                f"malformed CHIANTI SCUPS record in {path}: {header}"
            ) from error
        cursor += 2
        if (
            energy <= 0.0
            or node_count < 2
            or len(scaled_temperature) != node_count
            or len(scaled_upsilon) != node_count
        ):
            raise ValueError(f"incomplete CHIANTI SCUPS record in {path}: {header}")
        lower = reverse_mapping.get((symbol, int(charge), lower_index))
        upper = reverse_mapping.get((symbol, int(charge), upper_index))
        if lower is None or upper is None or lower == upper:
            continue
        population_ion = model_atom.atomic_database.ions[(symbol, int(charge))]
        level_by_index = {level.index: level for level in population_ion.levels}
        if (
            level_by_index[lower[2]].energy_wavenumber
            > level_by_index[upper[2]].energy_wavenumber
        ):
            lower, upper = upper, lower
        key = (symbol, int(charge), lower[2], upper[2])
        grouped.setdefault(key, []).append(ChiantiScaledCollisionComponent(
            transition_energy_rydberg=energy,
            transition_type=transition_type,
            scaling_parameter=scaling_parameter,
            scaled_temperature=scaled_temperature,
            scaled_upsilon=scaled_upsilon,
        ))
    if not grouped:
        raise ValueError(
            f"{path} contains no collisions mapped onto {symbol} {charge:+d}"
        )
    return MappingProxyType({
        key: ChiantiTermCollisionStrength(
            components=tuple(components),
            source=f"CHIANTI SCUPS: {source_path.name}",
        )
        for key, components in grouped.items()
    })


def atomic_database_with_chianti_radiative_transitions(
    atomic_database: AtomicDatabase,
    path: str | Path,
    element: str,
    charge: int,
) -> AtomicDatabase:
    """Fill missing bound-bound transitions from a CHIANTI ``wgfa`` file.

    CHIANTI and Stout use the same energy-ordered fine-structure indices for
    several ions assembled from the CHIANTI level set.  Some local line-list
    snapshots can nevertheless omit individual radiative records.  This helper only adds a line
    when both indexed levels already exist and that endpoint pair is absent,
    so it cannot alter the partition function or duplicate an existing line.
    The CHIANTI ``gf`` value is converted to the absorption oscillator
    strength by dividing by the lower-level statistical weight.
    """

    symbol = element.strip().capitalize()
    ion_key = (symbol, int(charge))
    if ion_key not in atomic_database.ions:
        raise ValueError(f"atomic database has no {symbol} {charge:+d} ion")
    ion = atomic_database.ions[ion_key]
    level_by_index = {level.index: level for level in ion.levels}
    existing_pairs = {
        (transition.lower_index, transition.upper_index)
        for transition in ion.transitions
    }
    additions = []
    source_path = Path(path)
    for raw_line in source_path.read_text(encoding="ascii").splitlines():
        record = raw_line.strip()
        if not record or record.startswith("%"):
            continue
        if record == "-1":
            break
        fields = record.split()
        if len(fields) < 5:
            continue
        try:
            lower_index = int(fields[0])
            upper_index = int(fields[1])
            wavelength = abs(float(fields[2]))
            weighted_oscillator_strength = float(fields[3])
            einstein_a = float(fields[4])
        except ValueError:
            continue
        pair = (lower_index, upper_index)
        if (
            lower_index not in level_by_index
            or upper_index not in level_by_index
            or pair in existing_pairs
            or wavelength <= 0.0
            or weighted_oscillator_strength <= 0.0
            or einstein_a <= 0.0
        ):
            continue
        lower_level = level_by_index[lower_index]
        upper_level = level_by_index[upper_index]
        delta_wavenumber = (
            upper_level.energy_wavenumber - lower_level.energy_wavenumber
        )
        if delta_wavenumber <= 0.0:
            continue
        implied_wavelength = 1.0e8 / delta_wavenumber
        # Refuse to combine unrelated index conventions.  A modest allowance
        # covers theoretical versus observed wavelengths without permitting a
        # silently wrong level mapping.
        if abs(implied_wavelength / wavelength - 1.0) > 0.03:
            # Level lists assembled from the same source often share a
            # leading block and diverge only when one database inserts extra
            # configurations.  Retain individually verified pairs from that
            # common block and skip later incompatible indices.
            continue
        additions.append(AtomicTransition(
            lower_index=lower_index,
            upper_index=upper_index,
            einstein_a=einstein_a,
            transition_type="E1",
            wavelength_vacuum_angstrom=wavelength,
            absorption_oscillator_strength=(
                weighted_oscillator_strength
                / lower_level.statistical_weight
            ),
        ))
        existing_pairs.add(pair)
    if not additions:
        raise ValueError(
            f"{path} adds no transitions to {symbol} {charge:+d}"
        )
    ions = dict(atomic_database.ions)
    ions[ion_key] = AtomicIon(
        element=ion.element,
        charge=ion.charge,
        atomic_mass_u=ion.atomic_mass_u,
        ionization_energy_ev=ion.ionization_energy_ev,
        levels=ion.levels,
        transitions=tuple(sorted(
            (*ion.transitions, *additions),
            key=lambda transition: (
                transition.wavelength_vacuum_angstrom,
                transition.lower_index,
                transition.upper_index,
            ),
        )),
        source=f"{ion.source}; missing radiative lines from CHIANTI {source_path.name}",
    )
    return AtomicDatabase(
        MappingProxyType(ions),
        source=f"{atomic_database.source}; augmented with CHIANTI {source_path.name}",
    )


def fine_structure_collision_data_from_tmad(
    model_atom: TmadStructureModelAtom,
    formal_atomic_database: AtomicDatabase,
) -> Mapping[tuple[str, int, int, int], tuple[int, tuple[float, ...]]]:
    """Project term-level TMAD Van Regemorter data onto formal components.

    The public formal atoms split an LS term into its fine-structure levels,
    while the compact CBB record supplies one term-averaged ``gbar``.  Each
    fine component must retain its own oscillator strength but inherit that
    term ``gbar``.  Falling back to the generic value can under-estimate the
    O V 3p--3d triplet collision rate by nearly a factor of three.

    General fitted effective collision strengths cannot be distributed over
    fine components without additional branching data, so formula 26 records
    deliberately remain on the compact atom.
    """

    projected = {}
    for compact_key, record in model_atom.collision_data.items():
        formula, parameters = record
        if formula != 1 or len(parameters) < 2:
            continue
        symbol, charge, lower_index, upper_index = compact_key
        lower_components = set(model_atom.formal_level_mapping.get(
            (symbol, charge, lower_index), ()
        ))
        upper_components = set(model_atom.formal_level_mapping.get(
            (symbol, charge, upper_index), ()
        ))
        if not lower_components or not upper_components:
            continue
        ion = formal_atomic_database.ions[(symbol, charge)]
        for transition in ion.transitions:
            lower_key = (symbol, charge, transition.lower_index)
            upper_key = (symbol, charge, transition.upper_index)
            if lower_key in lower_components and upper_key in upper_components:
                projected[(
                    symbol,
                    charge,
                    transition.lower_index,
                    transition.upper_index,
                )] = (
                    1,
                    (
                        transition.absorption_oscillator_strength,
                        parameters[1],
                    ),
                )
    return MappingProxyType(projected)


def read_tmad_structure_model_atom(
    path: str | Path,
    formal_atomic_database: AtomicDatabase,
    *,
    promote_lte_levels_per_charge: Mapping[int, int] | None = None,
    promote_lte_level_labels: Iterable[str] | None = None,
    target_nlte_levels_per_charge: Mapping[int, int] | None = None,
    diagnostic_allow_rdi_level_promotion: bool = False,
    allow_unmapped_nlte_terms: bool = False,
    infer_lte_collisions_from_cbb: bool = True,
) -> TmadStructureModelAtom:
    """Read a TMAD atmosphere atom and map its terms onto Stout levels.

    TMAD atmosphere atoms omit fine structure, whereas the separate formal
    solution uses split levels and observed wavelengths.  This reader keeps
    the compact TMAD rate-equation atom and constructs a statistical-weight
    preserving energy match to the Stout fine levels used by the local formal
    solution.  Individual LTE terms may be promoted by their fixed-width
    TMAD labels when a strategic line lies far down the energy-ordered LTE
    reservoir.  It supports the public C III--V and O III--VII TMAD files.
    ``diagnostic_allow_rdi_level_promotion`` exists solely to reproduce an
    older, physically inconsistent local checkpoint in controlled A/B tests;
    production callers must leave it false.  ``allow_unmapped_nlte_terms``
    supports a larger population atom than the independently supplied formal
    synthesis atom: such terms remain in the rate/cascade network but cannot
    donate departure coefficients to formal lines.  It is false by default so
    that an accidental mismatch in a core atmosphere atom remains an error.
    Public TMAD downloads are pre-``ATOMS2`` input atoms.  Their ``CBB``
    blocks can contain a transition whose endpoint is subsequently classified
    as an LTE reservoir level; by default such a record is retained as the
    corresponding ``CBX`` rate.  Set ``infer_lte_collisions_from_cbb=False``
    only for an already processed atom whose separate ``CBX`` block is
    authoritative.
    """

    source_path = Path(path)
    lines = source_path.read_text(encoding="ascii").splitlines()
    try:
        atom_index = next(
            index for index, line in enumerate(lines) if line.strip() == "ATOM"
        )
        header = lines[atom_index + 1].split()
        symbol = header[0].capitalize()
        atomic_mass = float(header[2])
    except (StopIteration, IndexError, ValueError) as error:
        raise ValueError(f"{path} is not a supported TMAD model atom") from error

    raw_levels: dict[str, tuple[int, float, float]] = {}
    raw_continuum_parent: dict[str, str] = {}
    ordered_keys: dict[int, list[str]] = {}
    for index, line in enumerate(lines):
        if line.strip() != "L":
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("0"):
            record = lines[cursor]
            cursor += 1
            if not record or record.lstrip().startswith("."):
                continue
            key = record[:10].strip()
            parent = record[10:20].strip()
            fields = record[20:].split()
            try:
                charge = int(key[len(symbol)]) - 1
                threshold_frequency = float(fields[0])
                statistical_weight = float(fields[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"malformed TMAD NLTE level in {path}: {record}"
                ) from error
            raw_levels[key] = (charge, threshold_frequency, statistical_weight)
            raw_continuum_parent[key] = parent
            ordered_keys.setdefault(charge, []).append(key)
    if not ordered_keys:
        raise ValueError(f"{path} contains no TMAD NLTE levels")

    raw_lte_levels: dict[str, tuple[int, str, float, float]] = {}
    ordered_lte_keys: dict[int, list[str]] = {}
    for index, line in enumerate(lines):
        if line.strip() != "LTE":
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("0"):
            record = lines[cursor]
            cursor += 1
            if not record or record.lstrip().startswith("."):
                continue
            key = record[:10].strip()
            target = record[10:20].strip()
            fields = record[20:].split()
            try:
                charge = int(key[len(symbol)]) - 1
                threshold_frequency = float(fields[0])
                statistical_weight = float(fields[1])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"malformed TMAD LTE level in {path}: {record}"
                ) from error
            raw_lte_levels[key] = (
                charge, target, threshold_frequency, statistical_weight
            )
            ordered_lte_keys.setdefault(charge, []).append(key)

    # RDI upper endpoints are doubly excited autoionizing states.  TMAP keeps
    # them in LTE with the recombining ion and eliminates them from the rate
    # equations (Werner et al. 2003, Sect. 3.2).  They must therefore never be
    # promoted by the generic "first N LTE levels" convenience used to mimic
    # a historical large line-formation atom.  Some public records reuse an
    # LTE level label for the eliminated endpoint, which is why the explicit
    # RDI declaration, rather than an energy heuristic, owns this decision.
    autoionizing_lte_keys: set[str] = set()
    radiative_dielectronic_records: list[
        TmadRadiativeDielectronicRecord
    ] = []
    for index, line in enumerate(lines):
        if line.strip() != "RDI":
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("0"):
            record = lines[cursor]
            cursor += 1
            if not record or record.lstrip().startswith("."):
                continue
            lower_key = record[:10].strip()
            upper_key = record[10:20].strip()
            fields = record[20:].split()
            try:
                formula = int(fields[0])
                parameter_count = int(fields[1])
                parameters = tuple(
                    float(value)
                    for value in fields[2 : 2 + parameter_count]
                )
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"malformed TMAD RDI transition in {path}: {record}"
                ) from error
            if len(parameters) != parameter_count or not parameters:
                raise ValueError(
                    f"incomplete TMAD RDI transition in {path}: {record}"
                )
            radiative_dielectronic_records.append(
                TmadRadiativeDielectronicRecord(
                    lower_level_label=lower_key,
                    autoionizing_level_label=upper_key,
                    oscillator_strength=parameters[0],
                    formula=formula,
                )
            )
            if upper_key in raw_lte_levels:
                autoionizing_lte_keys.add(upper_key)

    if (
        promote_lte_levels_per_charge is not None
        and target_nlte_levels_per_charge is not None
    ):
        raise ValueError(
            "specify either promoted or target TMAD NLTE level counts, not both"
        )

    # TMAP's fixed-structure line-formation step changes the number of terms
    # in selected ions while retaining one population per unsplit LS term.
    # The public atoms are energy ordered within each ion.  Exact target
    # counts are useful for reproducing a published calculation because the
    # public atom can have changed since an archival spectrum was generated
    # (the oxygen atom used by Werner & Rauch 2014 had 9 O VI terms, whereas
    # the present public structure atom has 14).  Excess present-day NLTE
    # terms are returned to the named-parent LTE reservoir; missing terms are
    # promoted from that reservoir in file order.
    promoted_lte_keys: set[str] = set()
    if target_nlte_levels_per_charge is not None:
        for charge_value, target_value in target_nlte_levels_per_charge.items():
            charge = int(charge_value)
            target = int(target_value)
            if target < 1:
                raise ValueError("target TMAD NLTE level counts must be positive")
            current_keys = ordered_keys.get(charge, [])
            if not current_keys:
                raise ValueError(f"TMAD atom has no NLTE ion with charge {charge}")
            if target < len(current_keys):
                demoted = current_keys[target:]
                ordered_keys[charge] = current_keys[:target]
                ordered_lte_keys[charge] = demoted + ordered_lte_keys.get(
                    charge, []
                )
                for key in demoted:
                    level_charge, threshold_frequency, statistical_weight = (
                        raw_levels.pop(key)
                    )
                    parent = raw_continuum_parent.pop(key)
                    raw_lte_levels[key] = (
                        level_charge,
                        parent,
                        threshold_frequency,
                        statistical_weight,
                    )
            elif target > len(current_keys):
                needed = target - len(current_keys)
                candidates = [
                    key
                    for key in ordered_lte_keys.get(charge, [])
                    if (
                        diagnostic_allow_rdi_level_promotion
                        or key not in autoionizing_lte_keys
                    )
                ]
                if needed > len(candidates):
                    raise ValueError(
                        f"TMAD charge {charge} has only "
                        f"{len(current_keys) + len(candidates)} non-autoionizing "
                        f"terms, cannot select {target}; RDI upper states remain "
                        "LTE by construction"
                    )
                for key in candidates[:needed]:
                    (
                        level_charge,
                        parent,
                        threshold_frequency,
                        statistical_weight,
                    ) = raw_lte_levels[key]
                    raw_levels[key] = (
                        level_charge, threshold_frequency, statistical_weight
                    )
                    raw_continuum_parent[key] = parent
                    ordered_keys[charge].append(key)
                    promoted_lte_keys.add(key)
                promoted = set(candidates[:needed])
                ordered_lte_keys[charge] = [
                    key for key in ordered_lte_keys.get(charge, [])
                    if key not in promoted
                ]
    if promote_lte_levels_per_charge is not None:
        for charge_value, count_value in promote_lte_levels_per_charge.items():
            charge = int(charge_value)
            count = int(count_value)
            if count < 0:
                raise ValueError("promoted TMAD LTE level counts must be non-negative")
            candidates = [
                key
                for key in ordered_lte_keys.get(charge, ())
                if (
                    diagnostic_allow_rdi_level_promotion
                    or key not in autoionizing_lte_keys
                )
            ]
            for key in candidates[:count]:
                level_charge, parent, threshold_frequency, statistical_weight = (
                    raw_lte_levels[key]
                )
                raw_levels[key] = (
                    level_charge, threshold_frequency, statistical_weight
                )
                raw_continuum_parent[key] = parent
                ordered_keys.setdefault(level_charge, []).append(key)
                promoted_lte_keys.add(key)
    if promote_lte_level_labels is not None:
        requested_labels = tuple(
            str(label).strip() for label in promote_lte_level_labels
        )
        if len(set(requested_labels)) != len(requested_labels):
            raise ValueError("targeted TMAD LTE level labels must be unique")
        for key in requested_labels:
            if key not in raw_lte_levels:
                raise ValueError(
                    f"TMAD atom has no LTE level with label {key!r}"
                )
            if key in promoted_lte_keys:
                continue
            if (
                key in autoionizing_lte_keys
                and not diagnostic_allow_rdi_level_promotion
            ):
                raise ValueError(
                    f"TMAD level {key!r} is an RDI autoionizing state and "
                    "must remain LTE"
                )
            level_charge, parent, threshold_frequency, statistical_weight = (
                raw_lte_levels[key]
            )
            raw_levels[key] = (
                level_charge, threshold_frequency, statistical_weight
            )
            raw_continuum_parent[key] = parent
            ordered_keys.setdefault(level_charge, []).append(key)
            promoted_lte_keys.add(key)
    if promoted_lte_keys:
        ordered_lte_keys = {
            charge: [key for key in keys if key not in promoted_lte_keys]
            for charge, keys in ordered_lte_keys.items()
        }

    # A composite TMAD file can span an entire isonuclear sequence (the
    # public sulfur atom, for example, contains S I--XVII).  Supplying exact
    # target counts is also an explicit request for the charge stages that
    # participate in this reduced atom.  Do not require unrelated neutral or
    # very highly ionized terms to map onto a formal database assembled only
    # for the adjacent stages of interest.
    if target_nlte_levels_per_charge is not None:
        selected_charges = {
            int(charge) for charge in target_nlte_levels_per_charge
        }
        ordered_keys = {
            charge: keys
            for charge, keys in ordered_keys.items()
            if charge in selected_charges
        }
        ordered_lte_keys = {
            charge: keys
            for charge, keys in ordered_lte_keys.items()
            if charge in selected_charges
        }
        retained_keys = {
            key for keys in ordered_keys.values() for key in keys
        }
        raw_levels = {
            key: value for key, value in raw_levels.items()
            if key in retained_keys
        }
        raw_continuum_parent = {
            key: value for key, value in raw_continuum_parent.items()
            if key in retained_keys
        }

    term_levels: dict[int, tuple[AtomicLevel, ...]] = {}
    key_to_index: dict[str, int] = {}
    for charge, keys in ordered_keys.items():
        ground_threshold = raw_levels[keys[0]][1]
        levels = []
        for local_index, key in enumerate(keys, 1):
            _, threshold_frequency, statistical_weight = raw_levels[key]
            # TMAD's threshold coordinate is measured from the next ion's
            # ground continuum even when the parent-term label names an
            # excited core. Consequently the excitation is simply the
            # difference from the first (ground-level) threshold.
            energy_wavenumber = (
                ground_threshold - threshold_frequency
            ) / LIGHT_SPEED
            if abs(energy_wavenumber) < 1.0e-7:
                energy_wavenumber = 0.0
            key_to_index[key] = local_index
            levels.append(AtomicLevel(
                local_index,
                float(max(energy_wavenumber, 0.0)),
                statistical_weight,
                key,
            ))
        term_levels[charge] = tuple(levels)

    # Parse the complete CBB block before selecting rate-equation records.
    # TMAD deliberately lists radiative/collisional transitions whose other
    # endpoint is an LTE term.  Once target level counts promote or demote a
    # term, looking only at ``raw_levels`` would silently discard those
    # one-sided cascade rates.
    def read_collision_block(
        keyword: str,
    ) -> dict[tuple[str, str], tuple[int, tuple[float, ...]]]:
        result: dict[tuple[str, str], tuple[int, tuple[float, ...]]] = {}
        for index, line in enumerate(lines):
            if line.strip() != keyword:
                continue
            cursor = index + 1
            while (
                cursor < len(lines)
                and not lines[cursor].strip().startswith("0")
            ):
                record = lines[cursor]
                cursor += 1
                if not record or record.lstrip().startswith("."):
                    continue
                lower_key = record[:10].strip()
                upper_key = record[10:20].strip()
                fields = record[20:].split()
                try:
                    formula = int(fields[0])
                    parameter_count = int(fields[1])
                    parameters = tuple(
                        float(value)
                        for value in fields[2 : 2 + parameter_count]
                    )
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        f"malformed TMAD {keyword} transition in "
                        f"{path}: {record}"
                    ) from error
                if len(parameters) != parameter_count:
                    raise ValueError(
                        f"incomplete TMAD {keyword} transition in "
                        f"{path}: {record}"
                    )
                result[(lower_key, upper_key)] = (formula, parameters)
        return result

    # Processed TMAP atoms distinguish collisions among explicit NLTE levels
    # (CBB) from NLTE-to-LTE reservoir collisions (CBX).  Public TMAD input
    # atoms still require ATOMS2, however, and can leave the latter records in
    # CBB for ATOMS2 to classify after the requested NLTE/LTE split is known.
    # An explicit CBX record always wins; the optional CBB fallback reproduces
    # that classification for raw public atoms.
    raw_collision_data = read_collision_block("CBB")
    raw_lte_collision_data = read_collision_block("CBX")

    transitions_by_charge: dict[int, list[AtomicTransition]] = {
        charge: [] for charge in term_levels
    }
    lte_bound_bound_couplings: list[TmadLTEBoundBoundCoupling] = []
    active_lte_keys = {
        key
        for keys in ordered_lte_keys.values()
        for key in keys
        if key not in autoionizing_lte_keys
    }
    for index, line in enumerate(lines):
        if line.strip() != "RBB":
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("0"):
            record = lines[cursor]
            cursor += 1
            if not record or record.lstrip().startswith("."):
                continue
            lower_key = record[:10].strip()
            upper_key = record[10:20].strip()
            lower_explicit = lower_key in raw_levels
            upper_explicit = upper_key in raw_levels
            lower_lte = lower_key in active_lte_keys
            upper_lte = upper_key in active_lte_keys
            if not (
                (lower_explicit and upper_explicit)
                or (lower_explicit and upper_lte)
                or (lower_lte and upper_explicit)
            ):
                continue
            lower_charge = (
                raw_levels[lower_key][0]
                if lower_explicit else raw_lte_levels[lower_key][0]
            )
            upper_charge = (
                raw_levels[upper_key][0]
                if upper_explicit else raw_lte_levels[upper_key][0]
            )
            if upper_charge != lower_charge:
                continue
            fields = record[20:].split()
            try:
                formula = int(fields[0])
                parameter_count = int(fields[1])
                # Even formula 1 records carry Gamma_rad and the Cowley
                # Stark factor after their sole formal input parameter.
                # They are fixed auxiliary columns in the public TMAD files.
                stored_parameter_count = max(parameter_count, 3)
                parameters = tuple(
                    float(value)
                    for value in fields[2 : 2 + stored_parameter_count]
                )
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"malformed TMAD RBB transition in {path}: {record}"
                ) from error
            # A few valid public TMAD formula-1 records provide only the
            # oscillator strength and omit optional Gamma/Cowley profile
            # columns.  Their wavelength and Einstein A are recoverable from
            # the retained endpoint energies, so dropping them changes the
            # rate atom (notably O VI 7f--9d and 7f--10d).
            if not parameters or parameters[0] <= 0.0:
                continue
            oscillator_strength = parameters[0]
            ground_threshold = raw_levels[ordered_keys[lower_charge][0]][1]
            if lower_explicit:
                lower_index = key_to_index[lower_key]
                lower_level = term_levels[lower_charge][lower_index - 1]
            else:
                _, _, threshold_frequency, statistical_weight = (
                    raw_lte_levels[lower_key]
                )
                lower_index = 0
                lower_level = AtomicLevel(
                    0,
                    float(max(
                        (ground_threshold - threshold_frequency) / LIGHT_SPEED,
                        0.0,
                    )),
                    statistical_weight,
                    lower_key,
                )
            if upper_explicit:
                upper_index = key_to_index[upper_key]
                upper_level = term_levels[lower_charge][upper_index - 1]
            else:
                _, _, threshold_frequency, statistical_weight = (
                    raw_lte_levels[upper_key]
                )
                upper_index = 0
                upper_level = AtomicLevel(
                    0,
                    float(max(
                        (ground_threshold - threshold_frequency) / LIGHT_SPEED,
                        0.0,
                    )),
                    statistical_weight,
                    upper_key,
                )
            delta_wavenumber = (
                upper_level.energy_wavenumber - lower_level.energy_wavenumber
            )
            if delta_wavenumber <= 0.0:
                continue
            match = _TMAD_WAVELENGTH_PATTERN.search(record)
            wavelength = (
                1.0e8 / delta_wavenumber
                if match is None else float(match.group(1))
            )
            einstein_a = _einstein_a_from_absorption_oscillator_strength(
                oscillator_strength,
                wavelength,
                lower_level.statistical_weight,
                upper_level.statistical_weight,
            )
            transition = AtomicTransition(
                lower_index,
                upper_index,
                einstein_a,
                "E1",
                wavelength,
                oscillator_strength,
                profile_formula=formula,
                profile_parameters=parameters,
            )
            if lower_explicit and upper_explicit:
                transitions_by_charge[lower_charge].append(transition)
                continue

            explicit_key = lower_key if lower_explicit else upper_key
            lte_key = upper_key if upper_lte else lower_key
            _, parent, _, lte_statistical_weight = raw_lte_levels[lte_key]
            if parent not in raw_levels:
                # An LTE term tied to another omitted term cannot be reduced
                # to the present explicit state vector without recursively
                # eliminating that parent.  It remains part of the inert LTE
                # reservoir and formal opacity only.
                continue
            lte_level = upper_level if upper_lte else lower_level
            lte_bound_bound_couplings.append(TmadLTEBoundBoundCoupling(
                explicit_level_key=(
                    symbol,
                    lower_charge,
                    key_to_index[explicit_key],
                ),
                lte_parent_key=(
                    symbol,
                    raw_levels[parent][0],
                    key_to_index[parent],
                ),
                explicit_is_lower=lower_explicit,
                lte_charge=lower_charge,
                lte_energy_wavenumber=lte_level.energy_wavenumber,
                lte_statistical_weight=lte_statistical_weight,
                lte_level_label=lte_key,
                transition=transition,
                collision_record=(
                    raw_lte_collision_data.get((lower_key, upper_key))
                    if not infer_lte_collisions_from_cbb
                    else raw_lte_collision_data.get(
                        (lower_key, upper_key),
                        raw_collision_data.get((lower_key, upper_key)),
                    )
                ),
            ))

    model_ions = {}
    maximum_charge = max(
        charge for element, charge in formal_atomic_database.ions if element == symbol
    )
    for charge in range(maximum_charge + 1):
        formal_ion = formal_atomic_database.ions[(symbol, charge)]
        levels = term_levels.get(
            charge, (AtomicLevel(1, 0.0, formal_ion.levels[0].statistical_weight, "ground"),)
        )
        ionization_energy = (
            raw_levels[ordered_keys[charge][0]][1] * PLANCK / EV_TO_ERG
            if charge in ordered_keys and charge < maximum_charge
            else formal_ion.ionization_energy_ev
        )
        model_ions[(symbol, charge)] = AtomicIon(
            element=symbol,
            charge=charge,
            atomic_mass_u=atomic_mass,
            ionization_energy_ev=ionization_energy,
            levels=levels,
            transitions=tuple(transitions_by_charge.get(charge, ())),
            source=f"TMAD structure atom: {source_path.name}",
        )
    population_database = AtomicDatabase(
        MappingProxyType(model_ions),
        source=f"TMAD structure atom: {source_path.name}",
    )

    continuum_parent_mapping = {}
    for key, parent in raw_continuum_parent.items():
        if parent not in raw_levels:
            continue
        charge = raw_levels[key][0]
        parent_charge = raw_levels[parent][0]
        continuum_parent_mapping[(symbol, charge, key_to_index[key])] = (
            symbol,
            parent_charge,
            key_to_index[parent],
        )

    # Resolve the subset of raw RDI declarations for which the public atom
    # actually supplies the eliminated autoionizing state's threshold.  The
    # transition frequency is nu_i - nu_c because TMAD stores both entries as
    # ionization frequencies relative to the same upper-ion continuum.  The
    # auto state remains absent from ``term_levels`` and from the rate vector.
    effective_dielectronic_couplings: list[
        TmadEffectiveDielectronicCoupling
    ] = []
    for record in radiative_dielectronic_records:
        lower_key = record.lower_level_label
        auto_key = record.autoionizing_level_label
        if (
            lower_key not in raw_levels
            or auto_key not in raw_lte_levels
            or auto_key in promoted_lte_keys
        ):
            continue
        lower_charge, lower_threshold, _ = raw_levels[lower_key]
        auto_charge, parent, auto_threshold, _ = raw_lte_levels[auto_key]
        if auto_charge != lower_charge or parent not in raw_levels:
            continue
        parent_charge = raw_levels[parent][0]
        if parent_charge != lower_charge + 1:
            continue
        transition_frequency = lower_threshold - auto_threshold
        if transition_frequency <= 0.0:
            continue
        effective_dielectronic_couplings.append(
            TmadEffectiveDielectronicCoupling(
                lower_level_key=(
                    symbol, lower_charge, key_to_index[lower_key]
                ),
                continuum_parent_key=(
                    symbol, parent_charge, key_to_index[parent]
                ),
                transition_frequency_hz=float(transition_frequency),
                oscillator_strength=record.oscillator_strength,
                lower_level_label=lower_key,
                autoionizing_level_label=auto_key,
            )
        )

    lte_level_reservoir_lists: dict[
        tuple[str, int, int], list[tuple[int, float, float]]
    ] = {}
    for charge, keys in ordered_lte_keys.items():
        if charge not in ordered_keys:
            continue
        ground_threshold = raw_levels[ordered_keys[charge][0]][1]
        for key in keys:
            if key in autoionizing_lte_keys:
                continue
            _, parent, threshold_frequency, statistical_weight = raw_lte_levels[key]
            if parent not in raw_levels:
                continue
            parent_charge = raw_levels[parent][0]
            parent_key = (symbol, parent_charge, key_to_index[parent])
            excitation_wavenumber = (
                ground_threshold - threshold_frequency
            ) / LIGHT_SPEED
            lte_level_reservoir_lists.setdefault(parent_key, []).append((
                charge,
                float(max(excitation_wavenumber, 0.0)),
                statistical_weight,
            ))
    lte_level_reservoir = MappingProxyType({
        parent: tuple(levels)
        for parent, levels in lte_level_reservoir_lists.items()
    })

    collision_data = {}
    for (lower_key, upper_key), collision_record in raw_collision_data.items():
        if lower_key not in raw_levels or upper_key not in raw_levels:
            continue
        charge = raw_levels[lower_key][0]
        if raw_levels[upper_key][0] != charge:
            continue
        collision_data[(
            symbol,
            charge,
            key_to_index[lower_key],
            key_to_index[upper_key],
        )] = collision_record

    formal_mapping = {}
    available_by_charge = {}
    energy_offset_by_charge = {}
    for charge, levels in term_levels.items():
        formal_levels = formal_atomic_database.ions[(symbol, charge)].levels
        available = {level.index: level for level in formal_levels}
        energy_offset = 0.0
        for term_position, term in enumerate(levels):
            best = None
            label_candidates = tuple(
                level
                for level in available.values()
                if _spectroscopic_term_labels_match(term.label, level.label) is True
            )
            # Fall back on the energy/statistical-weight match only among
            # formal levels whose labels carry no comparable spectroscopic
            # field.  A level whose n, multiplicity or L contradicts the term
            # must never be taken: near-degenerate high-n O V terms otherwise
            # borrow unrelated levels (2s6p for a missing 2s6s 3S, 2s7h for
            # 7p) and the formal lines between them inherit arbitrary
            # departure ratios.
            uninformative_candidates = tuple(
                level
                for level in available.values()
                if _spectroscopic_term_labels_match(term.label, level.label) is None
            )
            candidate_pools = tuple(
                pool for pool in (label_candidates, uninformative_candidates) if pool
            )
            for pool in candidate_pools:
                candidates = sorted(
                    pool,
                    key=lambda level: abs(
                        level.energy_wavenumber
                        - (term.energy_wavenumber + energy_offset)
                    ),
                )[:14]
                for count in range(1, min(6, len(candidates)) + 1):
                    for subset in combinations(candidates, count):
                        weight = sum(level.statistical_weight for level in subset)
                        if abs(weight - term.statistical_weight) > 1.0e-5:
                            continue
                        centroid = sum(
                            level.statistical_weight * level.energy_wavenumber
                            for level in subset
                        ) / term.statistical_weight
                        error = abs(
                            centroid - (term.energy_wavenumber + energy_offset)
                        )
                        if best is None or error < best[0]:
                            best = (error, subset, centroid)
                if best is not None:
                    break
            if best is None:
                # The population atom can contain a high LS superlevel for
                # which the independently supplied fine-structure atom has
                # no components with the same summed statistical weight.
                # Such a promoted term still belongs in the rate/cascade
                # network; it simply cannot provide a departure coefficient
                # to the formal line list.  Core atmosphere terms, by
                # contrast, must always have a valid mapping.
                if term.label in promoted_lte_keys or allow_unmapped_nlte_terms:
                    continue
                raise ValueError(
                    f"cannot map TMAD term {term.label!r} onto {symbol} {charge:+d}"
                )
            error, subset, centroid = best
            if term_position == 0:
                # A fine-split ground term is referenced to its lowest J
                # component by Stout but to the unsplit term by TMAD.
                energy_offset = centroid - term.energy_wavenumber
                error = 0.0
            if error > 3.0e3:
                # A promoted high term can be useful in the rate/cascade
                # network even when the independent formal atom has no
                # secure fine-structure counterpart.  Leaving just that term
                # unmapped also allows later securely matched terms (for
                # example O IV 6d after the absent/mismatched 6p term) to be
                # promoted.  Core structure-atom terms must still map.
                if term.label in promoted_lte_keys or allow_unmapped_nlte_terms:
                    continue
                raise ValueError(
                    f"TMAD term {term.label!r} differs from formal levels by {error:g} cm-1"
                )
            model_key = (symbol, charge, term.index)
            formal_mapping[model_key] = tuple(
                (symbol, charge, level.index) for level in subset
            )
            for level in subset:
                available.pop(level.index, None)
        available_by_charge[charge] = available
        energy_offset_by_charge[charge] = energy_offset

    formal_lte_parent_mapping = {}
    for charge, keys in ordered_lte_keys.items():
        if charge not in available_by_charge or charge not in ordered_keys:
            continue
        ground_threshold = raw_levels[ordered_keys[charge][0]][1]
        available = available_by_charge[charge]
        energy_offset = energy_offset_by_charge[charge]
        for key in keys:
            if key in autoionizing_lte_keys:
                continue
            _, parent, threshold_frequency, statistical_weight = raw_lte_levels[key]
            if parent not in raw_levels:
                continue
            energy_wavenumber = (
                ground_threshold - threshold_frequency
            ) / LIGHT_SPEED
            best = None
            label_candidates = tuple(
                level
                for level in available.values()
                if _spectroscopic_term_labels_match(key, level.label) is True
            )
            candidate_pools = (
                (label_candidates, tuple(available.values()))
                if label_candidates
                else (tuple(available.values()),)
            )
            for pool in candidate_pools:
                candidates = sorted(
                    pool,
                    key=lambda level: abs(
                        level.energy_wavenumber
                        - (energy_wavenumber + energy_offset)
                    ),
                )[:14]
                for count in range(1, min(6, len(candidates)) + 1):
                    for subset in combinations(candidates, count):
                        weight = sum(level.statistical_weight for level in subset)
                        if abs(weight - statistical_weight) > 1.0e-5:
                            continue
                        centroid = sum(
                            level.statistical_weight * level.energy_wavenumber
                            for level in subset
                        ) / statistical_weight
                        error = abs(
                            centroid - (energy_wavenumber + energy_offset)
                        )
                        if best is None or error < best[0]:
                            best = (error, subset)
                if best is not None:
                    break
            # Very high TMAD shell superlevels have no one-to-one formal
            # counterpart. Leave those on the ion-stage fallback; individual
            # LTE terms with a secure energy/weight match inherit the named
            # continuum parent's departure coefficient.
            if best is None or best[0] > 2.0e3:
                continue
            _, subset = best
            parent_charge = raw_levels[parent][0]
            parent_key = (symbol, parent_charge, key_to_index[parent])
            for level in subset:
                formal_key = (symbol, charge, level.index)
                formal_lte_parent_mapping[formal_key] = parent_key
                available.pop(level.index, None)

    return TmadStructureModelAtom(
        atomic_database=population_database,
        levels_per_charge=MappingProxyType({
            charge: len(levels) for charge, levels in term_levels.items()
        }),
        formal_level_mapping=MappingProxyType(formal_mapping),
        formal_lte_parent_mapping=MappingProxyType(formal_lte_parent_mapping),
        continuum_parent_mapping=MappingProxyType(continuum_parent_mapping),
        lte_level_reservoir=lte_level_reservoir,
        lte_bound_bound_couplings=tuple(lte_bound_bound_couplings),
        collision_data=MappingProxyType(collision_data),
        source=f"TMAD structure atom: {source_path.name}",
        autoionizing_lte_level_labels=tuple(sorted(autoionizing_lte_keys)),
        radiative_dielectronic_records=tuple(
            radiative_dielectronic_records
        ),
        effective_dielectronic_couplings=tuple(
            effective_dielectronic_couplings
        ),
    )


def atomic_database_with_tmad_formal_ions(
    atomic_database: AtomicDatabase,
    paths: Mapping[tuple[str, int], str | Path],
) -> AtomicDatabase:
    """Replace selected Stout line lists with public TMAD formal atoms.

    TMAD fine-structure levels are first matched to the Stout thermodynamic
    levels by statistical weight, shared spectroscopic n/L fields, and then
    energy.  A TMAD level with no secure Stout counterpart is appended to the
    ion rather than silently dropping every transition connected to it.  The
    TMAP wavelengths, oscillator strengths, radiative rates, and
    profile-formula parameters are retained.
    """

    ions = dict(atomic_database.ions)
    sources = []
    for requested_key, path_value in paths.items():
        symbol = requested_key[0].strip().capitalize()
        charge = int(requested_key[1])
        path = Path(path_value)
        lines = path.read_text(encoding="ascii").splitlines()
        try:
            level_start = next(
                index for index, line in enumerate(lines) if line.strip() == "L"
            ) + 1
            level_stop = next(
                index
                for index, line in enumerate(lines[level_start:], level_start)
                if line.strip() == "RBB"
            )
        except StopIteration as error:
            raise ValueError(f"{path} is not a supported TMAD formal atom") from error

        raw_levels = []
        for record in lines[level_start:level_stop]:
            if not record or record.lstrip().startswith("."):
                continue
            key = record[:10].strip()
            fields = record[20:].split()
            if not key.startswith(symbol) or len(fields) < 2:
                continue
            try:
                record_charge = int(key[len(symbol)]) - 1
                threshold_frequency = float(fields[0])
                statistical_weight = float(fields[1])
            except (IndexError, ValueError):
                continue
            if record_charge == charge:
                raw_levels.append((key, threshold_frequency, statistical_weight))
        if not raw_levels:
            raise ValueError(f"{path} contains no {symbol} {charge:+d} formal levels")

        # Some formal atoms declare an unresolved LS term in ``L`` but use
        # its J-resolved keys in ``RBB`` (for example O507F  3FO versus
        # O507F 31FO).  The seventh key character encodes 2J for half-integer
        # terms and J for integer terms, from which the component weight is
        # recovered using the adjacent spin-multiplicity digit.
        # Expand such a parent before matching levels, otherwise all lines
        # referring to its components appear to have undeclared endpoints.
        referenced_keys: set[str] = set()
        in_rbb = False
        for record in lines:
            if record.strip() == "RBB":
                in_rbb = True
                continue
            if in_rbb and record.strip().startswith("0"):
                in_rbb = False
                continue
            if in_rbb and record and not record.lstrip().startswith("."):
                referenced_keys.update((
                    record[:10].strip(),
                    record[10:20].strip(),
                ))
        raw_level_key = {key for key, _, _ in raw_levels}
        components_by_parent: dict[str, set[str]] = {}

        def component_statistical_weight(key: str) -> int:
            j_code = int(key[6])
            multiplicity = int(key[7])
            # Odd multiplicity has integer J encoded directly; even
            # multiplicity has half-integer J encoded as 2J.
            return 2 * j_code + 1 if multiplicity % 2 else j_code + 1

        for key in referenced_keys - raw_level_key:
            if len(key) <= 7 or not key[6].isdigit():
                continue
            parent = key[:6] + " " + key[7:]
            if parent in raw_level_key:
                components_by_parent.setdefault(parent, set()).add(key)
        if components_by_parent:
            expanded_levels = []
            for key, threshold_frequency, statistical_weight in raw_levels:
                components = sorted(components_by_parent.get(key, ()))
                if not components:
                    expanded_levels.append(
                        (key, threshold_frequency, statistical_weight)
                    )
                    continue
                component_weight = sum(
                    component_statistical_weight(component)
                    for component in components
                )
                if component_weight > statistical_weight + 1.0e-5:
                    raise ValueError(
                        f"formal components of {key!r} exceed its statistical weight"
                    )
                expanded_levels.extend(
                    (
                        component,
                        threshold_frequency,
                        float(component_statistical_weight(component)),
                    )
                    for component in components
                )
                remaining_weight = statistical_weight - component_weight
                if remaining_weight > 1.0e-5:
                    # Retain unresolved, line-free components in the partition.
                    expanded_levels.append(
                        (key, threshold_frequency, float(remaining_weight))
                    )
            raw_levels = expanded_levels
        ground_threshold = raw_levels[0][1]
        formal_energy = {
            key: max((ground_threshold - threshold) / LIGHT_SPEED, 0.0)
            for key, threshold, _ in raw_levels
        }
        stout_ion = ions[(symbol, charge)]
        available = {level.index: level for level in stout_ion.levels}
        level_mapping = {}
        for key, _, statistical_weight in raw_levels:
            weight_candidates = [
                level for level in available.values()
                if abs(level.statistical_weight - statistical_weight) <= 1.0e-5
            ]
            if not weight_candidates:
                continue
            # High-Rydberg terms of a given n are nearly degenerate.  Energy
            # and statistical weight alone can therefore permute h, i, k,
            # ... components (for example O VI 8h J=11/2 onto 8i J=11/2).
            # That silently assigns the wrong NLTE departure coefficient to
            # a valid formal transition.  Prefer the common n/L fields in
            # the fixed-width TMAD key and the Stout configuration label;
            # retain the historical energy match only when no such label
            # candidate exists.
            label_candidates = [
                level
                for level in weight_candidates
                if _spectroscopic_term_labels_match(key, level.label) is True
            ]
            candidates = label_candidates or weight_candidates
            matched = min(
                candidates,
                key=lambda level: abs(
                    level.energy_wavenumber - formal_energy[key]
                ),
            )
            if abs(matched.energy_wavenumber - formal_energy[key]) > 3.0e3:
                continue
            level_mapping[key] = matched.index
            available.pop(matched.index, None)

        # The public O V/O VI synthesis atoms contain high fine-structure
        # levels absent from (or displaced relative to) the Stout list.  The
        # old importer discarded every line touching such a level: 25 optical
        # components in the distributed atoms, including the O VI
        # 4545--4552-A complex.  Retain the authoritative TMAD level itself.
        formal_levels = list(stout_ion.levels)
        next_level_index = max(level.index for level in formal_levels) + 1
        for key, _, statistical_weight in raw_levels:
            if key in level_mapping:
                continue
            level_mapping[key] = next_level_index
            formal_levels.append(AtomicLevel(
                next_level_index,
                formal_energy[key],
                statistical_weight,
                key,
            ))
            next_level_index += 1
        formal_level_by_index = {
            level.index: level for level in formal_levels
        }

        transitions = []
        in_rbb = False
        for record in lines:
            if record.strip() == "RBB":
                in_rbb = True
                continue
            if in_rbb and record.strip().startswith("0"):
                in_rbb = False
                continue
            if not in_rbb or not record or record.lstrip().startswith("."):
                continue
            lower_key = record[:10].strip()
            upper_key = record[10:20].strip()
            if lower_key not in level_mapping or upper_key not in level_mapping:
                continue
            fields = record[20:].split()
            try:
                formula = int(fields[0])
                parameter_count = int(fields[1])
                parameters = tuple(
                    float(value) for value in fields[2 : 2 + parameter_count]
                )
            except (IndexError, ValueError):
                continue
            if len(parameters) < 2 or parameters[0] <= 0.0 or parameters[1] <= 0.0:
                continue
            match = _TMAD_WAVELENGTH_PATTERN.search(record)
            if match is None:
                delta = formal_energy[upper_key] - formal_energy[lower_key]
                if delta <= 0.0:
                    continue
                wavelength = 1.0e8 / delta
            else:
                wavelength = float(match.group(1))
            lower_level = formal_level_by_index[level_mapping[lower_key]]
            upper_level = formal_level_by_index[level_mapping[upper_key]]
            einstein_a = _einstein_a_from_absorption_oscillator_strength(
                parameters[0],
                wavelength,
                lower_level.statistical_weight,
                upper_level.statistical_weight,
            )
            transitions.append(AtomicTransition(
                level_mapping[lower_key],
                level_mapping[upper_key],
                einstein_a,
                "E1",
                wavelength,
                parameters[0],
                profile_formula=formula,
                profile_parameters=parameters,
            ))
        if not transitions:
            raise ValueError(f"{path} contains no mapped TMAD formal transitions")
        ions[(symbol, charge)] = replace(
            stout_ion,
            levels=tuple(formal_levels),
            transitions=tuple(transitions),
            source=f"TMAD formal-solution atom: {path.name}",
        )
        sources.append(path.name)
    return AtomicDatabase(
        MappingProxyType(ions),
        source=(
            atomic_database.source
            + "; TMAD formal ions "
            + ", ".join(sources)
        ),
    )


def atomic_database_with_tmad_profile_parameters(
    atomic_database: AtomicDatabase,
    paths: Mapping[tuple[str, int], str | Path],
    *,
    maximum_wavelength_difference_angstrom: float = 0.10,
    series_lower_principal_quantum_number: int | None = None,
    minimum_upper_principal_quantum_number: int | None = None,
    wavelength_intervals_angstrom: tuple[tuple[float, float], ...] | None = None,
) -> AtomicDatabase:
    """Attach TMAD broadening prescriptions without replacing a line list.

    The PG 1159 carbon synthesis currently retains the more stable Stout
    fine-structure representation and its mapping to the reduced NLTE rate
    atom.  Stout does not, however, carry TMAP's RBB profile flags.  In
    particular, the C IV 4d/4f--n=9 complex near 1135 A then loses its
    formula-4 quasi-static Stark wings.  This helper transfers only the
    profile formula and parameters from wavelength/strength-matched TMAD
    components.  Wavelengths, oscillator strengths, Einstein coefficients,
    level identities, and hence NLTE departure mappings remain those of the
    input database.  The optional principal-quantum-number selectors apply to
    TMAP formula-4 profiles, whose last two parameters are the lower and upper
    principal quantum numbers.  They permit a documented Rydberg series to be
    promoted without changing unrelated optical transitions.  Alternatively,
    ``wavelength_intervals_angstrom`` can select an identified multiplet when
    principal quantum numbers are not encoded in the profile record (for
    example the C III 1175-A fine-structure complex).
    """

    tolerance = float(maximum_wavelength_difference_angstrom)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "maximum_wavelength_difference_angstrom must be finite and positive"
        )
    if (
        series_lower_principal_quantum_number is not None
        and series_lower_principal_quantum_number <= 0
    ):
        raise ValueError(
            "series_lower_principal_quantum_number must be positive"
        )
    if (
        minimum_upper_principal_quantum_number is not None
        and minimum_upper_principal_quantum_number <= 0
    ):
        raise ValueError(
            "minimum_upper_principal_quantum_number must be positive"
        )
    intervals = None
    if wavelength_intervals_angstrom is not None:
        intervals = tuple(
            (float(lower), float(upper))
            for lower, upper in wavelength_intervals_angstrom
        )
        if not intervals or any(
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
            for lower, upper in intervals
        ):
            raise ValueError(
                "wavelength_intervals_angstrom must contain finite, "
                "increasing intervals"
            )

    def selected_profile(transition: AtomicTransition) -> bool:
        if intervals is not None and not any(
            lower <= transition.wavelength_vacuum_angstrom <= upper
            for lower, upper in intervals
        ):
            return False
        if (
            series_lower_principal_quantum_number is None
            and minimum_upper_principal_quantum_number is None
        ):
            return True
        if transition.profile_formula != 4 or len(transition.profile_parameters) < 2:
            return False
        lower_n = int(round(transition.profile_parameters[-2]))
        upper_n = int(round(transition.profile_parameters[-1]))
        return (
            series_lower_principal_quantum_number is None
            or lower_n == series_lower_principal_quantum_number
        ) and (
            minimum_upper_principal_quantum_number is None
            or upper_n >= minimum_upper_principal_quantum_number
        )

    formal = atomic_database_with_tmad_formal_ions(atomic_database, paths)
    ions = dict(atomic_database.ions)
    sources: list[str] = []
    for requested_key, path_value in paths.items():
        key = (requested_key[0].strip().capitalize(), int(requested_key[1]))
        original_ion = ions[key]
        formal_ion = formal.ions[key]
        usable = tuple(
            transition
            for transition in formal_ion.transitions
            if transition.profile_formula is not None
            and transition.profile_parameters
            and transition.absorption_oscillator_strength > 0.0
            and selected_profile(transition)
        )
        updated: list[AtomicTransition] = []
        matched = 0
        for transition in original_ion.transitions:
            candidates = tuple(
                candidate
                for candidate in usable
                if abs(
                    candidate.wavelength_vacuum_angstrom
                    - transition.wavelength_vacuum_angstrom
                ) <= tolerance
            )
            if not candidates or transition.absorption_oscillator_strength <= 0.0:
                updated.append(transition)
                continue
            candidate = min(
                candidates,
                key=lambda item: (
                    abs(
                        item.wavelength_vacuum_angstrom
                        - transition.wavelength_vacuum_angstrom
                    )
                    / tolerance
                    + abs(
                        np.log(
                            item.absorption_oscillator_strength
                            / transition.absorption_oscillator_strength
                        )
                    )
                ),
            )
            # A coincidental wavelength match with a very different line
            # strength is not a secure component identification.
            strength_ratio = (
                candidate.absorption_oscillator_strength
                / transition.absorption_oscillator_strength
            )
            if not 0.5 <= strength_ratio <= 2.0:
                updated.append(transition)
                continue
            parameters = (
                transition.absorption_oscillator_strength,
                *candidate.profile_parameters[1:],
            )
            updated.append(replace(
                transition,
                profile_formula=candidate.profile_formula,
                profile_parameters=tuple(float(value) for value in parameters),
            ))
            matched += 1
        if matched == 0:
            raise ValueError(
                f"{Path(path_value)} has no secure profile matches for {key}"
            )
        ions[key] = replace(
            original_ion,
            transitions=tuple(updated),
            source=(
                original_ion.source
                + f"; {matched} TMAD profile prescriptions from "
                + Path(path_value).name
            ),
        )
        sources.append(Path(path_value).name)
    return AtomicDatabase(
        MappingProxyType(ions),
        source=(
            atomic_database.source
            + "; TMAD profile prescriptions "
            + ", ".join(sources)
        ),
    )


@dataclass(frozen=True)
class LightMetalNLTEState:
    """Ion-stage NLTE populations and departures for several elements."""

    ion_number_density: Mapping[str, FloatArray]
    lte_ion_number_density: Mapping[str, FloatArray]
    ion_departure_coefficient: Mapping[tuple[str, int], FloatArray]
    photoionization_rate: Mapping[tuple[str, int], FloatArray]
    radiative_recombination_rate: Mapping[tuple[str, int], FloatArray]
    collisional_ionization_rate: Mapping[tuple[str, int], FloatArray]
    three_body_recombination_rate: Mapping[tuple[str, int], FloatArray]
    maximum_lte_recovery_error: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class ReducedLightMetalLevelState:
    """Explicit-level populations for one reduced light-metal model atom."""

    element: str
    level_key: tuple[tuple[str, int, int], ...]
    population_density: FloatArray
    lte_population_density: FloatArray
    level_departure_coefficient: Mapping[tuple[str, int, int], FloatArray]
    maximum_lte_recovery_error: float
    metadata: dict[str, object]
    population_level_departure_coefficient: Mapping[
        tuple[str, int, int], FloatArray
    ] | None = None
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None = None
    formal_lte_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None

    # Explicit terms plus omitted LTE terms tied to continuum parents.
    conservation_weight: FloatArray | None = None


def light_metal_free_free_charge_kernel(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    charges: Iterable[int],
) -> Mapping[int, FloatArray]:
    """Precompute the temperature-dependent free-free kernel per ion charge."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and 1D")
    selected_charges = tuple(sorted(set(int(value) for value in charges)))
    if any(charge < 1 for charge in selected_charges):
        raise ValueError("free-free ionic charges must be positive")
    frequency = LIGHT_SPEED / (wavelength[:, np.newaxis] * 1.0e-8)
    temperature = atmosphere.temperature[np.newaxis, :]
    stimulated = -np.expm1(
        -PLANCK * frequency / (BOLTZMANN * temperature)
    )
    common = (
        3.692e8
        * temperature**-0.5
        * atmosphere.electron_density[np.newaxis, :]
        * frequency**-3
        * stimulated
        / atmosphere.mass_density[np.newaxis, :]
    )
    return MappingProxyType({
        charge: np.ascontiguousarray(
            common
            * charge**2
            * hydrogen_free_free_gaunt_factor(
                wavelength[:, np.newaxis],
                temperature,
                ionic_charge=float(charge),
            )
        )
        for charge in selected_charges
    })


def light_metal_free_free_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    lte_state: MetalLTEState,
    ion_departure_coefficient: Mapping[tuple[str, int], ArrayLike] | None = None,
    *,
    elements: tuple[str, ...] | None = None,
    precomputed_charge_kernel: Mapping[int, ArrayLike] | None = None,
) -> FloatArray:
    """Return charge-weighted ionic free-free opacity in ``cm2 g-1``.

    The helium continuum already contains He II/III bremsstrahlung, but it
    cannot represent the substantial C/O ionic charge density in a PG 1159
    mixture.  This adds ``sum(q**2 n_q g_ff(q))`` for the selected metals,
    using the same van Hoof et al. thermally averaged Gaunt factors as the H
    and He continua.  Optional ion departures make the opacity consistent
    with the explicit NLTE ion-stage closure.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and 1D")
    selected = (
        tuple(lte_state.log_number_abundance)
        if elements is None
        else tuple(value.strip().capitalize() for value in elements)
    )
    population_by_charge: dict[int, FloatArray] = {}
    for element in selected:
        populations = lte_state.ion_number_density.get(element)
        if populations is None:
            continue
        for charge in range(1, populations.shape[0]):
            departure = (
                1.0
                if ion_departure_coefficient is None
                else ion_departure_coefficient.get((element, charge), 1.0)
            )
            actual_population = populations[charge] * np.asarray(
                departure, dtype=np.float64
            )
            population_by_charge[charge] = (
                population_by_charge.get(charge, 0.0) + actual_population
            )
    charge_kernel = (
        light_metal_free_free_charge_kernel(
            atmosphere, wavelength, population_by_charge
        )
        if precomputed_charge_kernel is None
        else precomputed_charge_kernel
    )
    absorption = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    for charge, actual_population in population_by_charge.items():
        if charge not in charge_kernel:
            raise ValueError(
                f"precomputed free-free kernel is missing charge {charge}"
            )
        kernel = np.asarray(charge_kernel[charge], dtype=np.float64)
        if kernel.shape != absorption.shape:
            raise ValueError(
                "precomputed free-free kernels must have shape (wavelength, depth)"
            )
        absorption += (
            kernel * actual_population[np.newaxis, :]
        )
    return np.ascontiguousarray(absorption)


def _accumulate_bound_free_nlte(cross_section, lower_population, mass_density,
        lower_departure, upper_departure, exponential, planck, absorption, emissivity):
    """Accumulate the unchanged bound-free formula with bounded working memory."""
    if _rt is not None and hasattr(_rt, "accumulate_bound_free_nlte"):
        _rt.accumulate_bound_free_nlte(
            *(np.ascontiguousarray(value, dtype=np.float64) for value in
              (cross_section, lower_population, mass_density, lower_departure,
               upper_departure, exponential, planck)), absorption, emissivity)
        return
    base = cross_section[:, None] * lower_population[None, :] / mass_density[None, :]
    local = base * (lower_departure[None, :] - upper_departure[None, :] * exponential)
    absorption += np.maximum(local, 0.)
    emissivity += base * (1. - exponential) * planck * upper_departure[None, :]


def light_metal_bound_free_nlte_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    lte_state: MetalLTEState,
    photoionization_database: VernerPhotoionizationDatabase,
    ion_departure_coefficient: Mapping[tuple[str, int], ArrayLike],
    *,
    level_departure_coefficient: Mapping[tuple[str, int, int], ArrayLike] | None = None,
    elements: tuple[str, ...] | None = None,
    photoionization_threshold_data: Mapping[
        int, TlustyPhotoionizationThresholdData
    ] | None = None,
    levels_per_charge: Mapping[int, int] | None = None,
    continuum_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return NLTE bound-free absorption and emissivity.

    For lower and continuum departures ``b_i`` and ``b_c``, the direct
    detailed-balance form is

    ``chi = n_i^* sigma (b_i - b_c exp(-h nu/kT)) / rho`` and
    ``eta = b_c chi_LTE B``.

    The second expression is normalized to the stimulated-emission LTE
    opacity and therefore recovers Kirchhoff's law when both departures are
    unity.  By default only Verner ground-state fits are used.  Supplying a
    TLUSTY threshold table and explicit level counts adds the same
    level-resolved OP/analytic cross sections used by the statistical-
    equilibrium rates; this is the TMAP RBF closure needed by compact C/O
    population atoms.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    wavelength_cm = wavelength * 1.0e-8
    photon_energy_ev = PLANCK * LIGHT_SPEED / wavelength_cm / EV_TO_ERG
    exponential = np.exp(
        -PLANCK * LIGHT_SPEED
        / (wavelength_cm[:, np.newaxis] * BOLTZMANN * atmosphere.temperature)
    )
    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    absorption = np.zeros((wavelength.size, atmosphere.n_depth))
    emissivity = np.zeros_like(absorption)
    selected_elements = (
        tuple(lte_state.log_number_abundance)
        if elements is None else tuple(element.strip().capitalize() for element in elements)
    )
    for element in selected_elements:
        stages = atomic_database.ion_stages(element)
        populations = lte_state.ion_number_density[element]
        for ion in stages[:-1]:
            fit = photoionization_database.fits.get((element, ion.charge))
            ground = min(ion.levels, key=lambda level: level.energy_wavenumber)
            partition = lte_state.partition_function[(element, ion.charge)]
            upper_ion = stages[ion.charge + 1]
            upper_ground = min(
                upper_ion.levels, key=lambda level: level.energy_wavenumber
            )
            threshold_table = (
                None
                if photoionization_threshold_data is None
                else photoionization_threshold_data.get(ion.charge)
            )
            explicit_count = (
                None
                if levels_per_charge is None
                else levels_per_charge.get(ion.charge)
            )
            levels = (
                (ground,)
                if threshold_table is None or explicit_count is None
                else tuple(sorted(
                    ion.levels, key=lambda level: level.energy_wavenumber
                )[: int(explicit_count)])
            )

            def departure(
                level_key: tuple[str, int, int], ion_key: tuple[str, int]
            ) -> FloatArray:
                if (
                    level_departure_coefficient is not None
                    and level_key in level_departure_coefficient
                ):
                    return np.asarray(level_departure_coefficient[level_key])
                return np.asarray(ion_departure_coefficient[ion_key])

            for level in levels:
                lower_key = (element, ion.charge, level.index)
                default_upper_key = (
                    element, upper_ion.charge, upper_ground.index
                )
                upper_key = (
                    default_upper_key
                    if continuum_parent_mapping is None
                    else continuum_parent_mapping.get(
                        lower_key, default_upper_key
                    )
                )
                threshold_ev = (
                    ion.ionization_energy_ev
                    - level.energy_wavenumber
                    * PLANCK
                    * LIGHT_SPEED
                    / EV_TO_ERG
                )
                if threshold_ev <= 0.05:
                    continue
                threshold_frequency = threshold_ev * EV_TO_ERG / PLANCK
                cross_section = (
                    None
                    if threshold_table is None
                    else threshold_table.cross_section(
                        photon_energy_ev * EV_TO_ERG / PLANCK,
                        threshold_frequency,
                        relative_tolerance=0.02,
                        level_label=level.label,
                    )
                )
                if cross_section is None and level.index == ground.index:
                    if fit is None:
                        continue
                    cross_section = fit.cross_section(photon_energy_ev)
                elif cross_section is None:
                    effective_charge = ion.charge + 1.0
                    effective_n = np.sqrt(
                        13.605_693_122_994
                        * effective_charge**2
                        / threshold_ev
                    )
                    sigma0 = 6.30e-18 * effective_n / effective_charge**2
                    cross_section = np.where(
                        photon_energy_ev >= threshold_ev,
                        sigma0 * (threshold_ev / photon_energy_ev) ** 3,
                        0.0,
                    )
                lower_population = (
                    populations[ion.charge]
                    * level.statistical_weight
                    * np.exp(
                        -level.energy_wavenumber
                        * PLANCK
                        * LIGHT_SPEED
                        / (BOLTZMANN * atmosphere.temperature)
                    )
                    / partition
                )
                lower_departure = departure(
                    lower_key, (element, ion.charge)
                )
                upper_departure = departure(
                    upper_key, (element, upper_ion.charge)
                )
                _accumulate_bound_free_nlte(cross_section, lower_population,
                    atmosphere.mass_density, lower_departure, upper_departure,
                    exponential, planck, absorption, emissivity)
    return np.ascontiguousarray(absorption), np.ascontiguousarray(emissivity)


def hot_metal_line_nlte_coefficients(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    lte_state: MetalLTEState,
    ion_departure_coefficient: Mapping[tuple[str, int], ArrayLike],
    *,
    level_departure_coefficient: Mapping[tuple[str, int, int], ArrayLike] | None = None,
    minimum_oscillator_strength: float = 1.0e-4,
    maximum_lines: int | None = 20_000,
    include_classical_electron_stark: bool = True,
    include_static_linear_stark: bool = True,
    static_linear_stark_frequency_scales: Mapping[
        tuple[str, int, int, int], float
    ] | None = None,
    tabulated_electron_stark_width_scale: float = 1.0,
    include_semiclassical_ovi_stark_widths: bool = False,
    include_ovi_high_series_ion_dephasing: bool = False,
    elements: tuple[str, ...] | None = None,
    transition_keys: Iterable[tuple[str, int, int, int]] | None = None,
    retain_inverted_emissivity: bool = False,
    include_ion_dynamic_stark_core: bool = False,
    hydrogenic_linear_stark_components: bool = False,
    extend_strong_uv_resonance_wings: bool = True,
    strong_uv_resonance_core_optical_depth: float = 1.0e3,
) -> tuple[FloatArray, FloatArray]:
    """Return ordinary hot-metal NLTE line absorption and emissivity.

    Thermal Doppler, natural damping, and the standard SYNSPEC classical
    electron-impact Stark damping are included.  The latter uses
    ``Gamma_e / n_e = 1e-8 n_eff**5`` and is particularly important for the
    highly excited C IV/O VI lines in compact PG1159 atmospheres.  TMAD RBB
    formula 4 additionally uses TMAP's quasi-static linear-Stark wing: the
    pointwise maximum of the impact Voigt cross-section and the Holtsmark
    microfield cross-section.  ``static_linear_stark_frequency_scales`` can
    supply ion-dynamic corrections keyed by ``(element, charge, lower_n,
    upper_n)`` or ``(element, charge)``; a scale of 0 removes the wing.  A
    scale contracts the quasi-static frequency coordinate and
    raises its amplitude by the reciprocal factor, preserving the area of
    the pure static component.  Neutral-perturber unified profiles belong to
    the cool DZ path and are intentionally absent.
    For O VI high-series lines, the optional ion-dephasing term adds the
    strong-collision limit set by the local microfield correlation time to
    the impact core.  This closes a consistency gap when the same-shell
    heavy-ion collisions are present in the statistical-equilibrium atom:
    those collisions redistribute the level populations and also destroy
    line coherence.  The rate is capped by field decorrelation rather than
    by the much larger sum of independent ``l``-mixing rates, avoiding a
    double count of the quasi-static microfield wing.

    The lower/upper departure form recovers the existing LTE opacity and
    Kirchhoff source when all departures are unity.  The optional ion-motion
    correction convolves the quasi-static ion term with the independently
    calculated microfield decorrelation width; it does not alter oscillator
    strengths or add empirical damping to the electron-impact core.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be one-dimensional and increasing")
    selected_elements = (
        tuple(lte_state.log_number_abundance)
        if elements is None else tuple(element.strip().capitalize() for element in elements)
    )
    selected_abundance = {
        element: lte_state.log_number_abundance[element]
        for element in selected_elements
    }
    selected = selected_metal_lines(
        atomic_database,
        selected_abundance,
        float(wavelength[0]),
        float(wavelength[-1]),
        minimum_oscillator_strength,
        maximum_lines,
        atmosphere.effective_temperature,
        {
            (element, charge): float(np.max(
                population[charge]
                * np.asarray(ion_departure_coefficient[(element, charge)])
                / np.maximum(lte_state.element_number_density[element], 1.0)
            ))
            for element, population in lte_state.ion_number_density.items()
            if element in selected_elements
            for charge in range(population.shape[0])
        },
    )
    if transition_keys is not None:
        allowed = frozenset(transition_keys)
        selected = [
            (ion, line)
            for ion, line in selected
            if (ion.element, ion.charge, line.lower_index, line.upper_index)
            in allowed
        ]
    # Accumulate in (depth, wavelength) order: the compiled profile kernel
    # then writes with unit stride.  Transposed once on return.
    absorption = np.zeros((atmosphere.n_depth, wavelength.size))
    emissivity = np.zeros_like(absorption)
    atomic_mass_unit = 1.660_539_068_92e-24
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    ions = {(ion.element, ion.charge): ion for ion, _ in selected}
    levels = {
        key: {level.index: level for level in ion.levels}
        for key, ion in ions.items()
    }
    upper_rate = {}
    for ion in ions.values():
        for transition in ion.transitions:
            key = (ion.element, ion.charge, transition.upper_index)
            upper_rate[key] = upper_rate.get(key, 0.0) + transition.einstein_a
    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    planck_depth_major = np.ascontiguousarray(planck.T)
    lte_population_cache = {}

    # TMAP's z_Mikro = [sum_i Z_i^(3/2) n_i]^(2/3).  Explicit C/O ion
    # populations provide their contributions.  The remaining charge is
    # overwhelmingly He++ in the hot PG1159 regime and is assigned Z=2; this
    # is exact in that limit and much closer than treating every electron as
    # a singly charged perturber.
    microfield_charge_sum = np.zeros(atmosphere.n_depth, dtype=np.float64)
    represented_electron_density = np.zeros_like(microfield_charge_sum)
    for element in selected_elements:
        ion_population = lte_state.ion_number_density.get(element)
        if ion_population is None:
            continue
        for charge in range(1, ion_population.shape[0]):
            departure = ion_departure_coefficient.get((element, charge))
            if departure is None:
                continue
            actual_population = ion_population[charge] * np.asarray(
                departure, dtype=np.float64
            )
            microfield_charge_sum += charge**1.5 * actual_population
            represented_electron_density += charge * actual_population
    helium_electron_density = np.maximum(
        atmosphere.electron_density - represented_electron_density, 0.0
    )
    microfield_charge_sum += np.sqrt(2.0) * helium_electron_density
    microfield_scale = np.maximum(microfield_charge_sum, 0.0) ** (2.0 / 3.0)
    ion_motion_rate_by_mass: dict[float, FloatArray] = {}

    def ion_motion_rate(atomic_mass_u: float) -> FloatArray:
        rate = ion_motion_rate_by_mass.get(atomic_mass_u)
        if rate is None:
            rate = _ion_microfield_motion_rate(
                atmosphere, lte_state, atomic_mass_u
            )
            ion_motion_rate_by_mass[atomic_mass_u] = rate
        return rate

    profile_batch: dict[str, list[FloatArray | float]] = {
        "center": [],
        "integrated_strength": [],
        "gaussian_sigma": [],
        "lorentz_hwhm": [],
        "minimum_half_window": [],
        "static_frequency_scale": [],
        "static_amplitude": [],
        "static_ion_motion_hwhm_beta": [],
        "static_pattern": [],
        "population_scale": [],
        "lower_departure": [],
        "upper_departure": [],
        "exponential": [],
    }

    def flush_profile_batch() -> None:
        if not profile_batch["center"]:
            return
        _accumulate_metal_line_profiles(
            wavelength,
            planck,
            np.asarray(profile_batch["center"], dtype=np.float64),
            np.asarray(profile_batch["integrated_strength"], dtype=np.float64),
            np.stack(profile_batch["gaussian_sigma"]),
            np.stack(profile_batch["lorentz_hwhm"]),
            np.asarray(profile_batch["minimum_half_window"], dtype=np.float64),
            np.stack(profile_batch["static_frequency_scale"]),
            np.stack(profile_batch["static_amplitude"]),
            np.stack(profile_batch["static_ion_motion_hwhm_beta"]),
            np.stack(profile_batch["population_scale"]),
            np.stack(profile_batch["lower_departure"]),
            np.stack(profile_batch["upper_departure"]),
            np.stack(profile_batch["exponential"]),
            absorption,
            emissivity,
            retain_inverted_emissivity,
            depth_major=True,
            planck_depth_major=planck_depth_major,
            static_pattern=list(profile_batch["static_pattern"]),
        )
        for values in profile_batch.values():
            values.clear()

    for ion, line in selected:
        lower_key = (ion.element, ion.charge, line.lower_index)
        upper_key = (ion.element, ion.charge, line.upper_index)
        lower_population = lte_population_cache.get(lower_key)
        if lower_population is None:
            level = levels[(ion.element, ion.charge)][line.lower_index]
            partition = lte_state.partition_function[(ion.element, ion.charge)]
            lower_population = (
                lte_state.ion_number_density[ion.element][ion.charge]
                * level.statistical_weight
                * np.exp(
                    -level.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                )
                / partition
            )
            lte_population_cache[lower_key] = lower_population
        # A reduced atom only supplies a physically closed line transition
        # when *both* endpoints were retained in its rate equations.  Mixing
        # an explicit-level departure coefficient at one endpoint with an
        # ion-mean coefficient at the other creates an artificial source
        # function discontinuity at the reduced-atom boundary.
        if (
            level_departure_coefficient is not None
            and lower_key in level_departure_coefficient
            and upper_key in level_departure_coefficient
        ):
            lower_departure = np.asarray(
                level_departure_coefficient[lower_key], dtype=np.float64
            )
            upper_departure = np.asarray(
                level_departure_coefficient[upper_key], dtype=np.float64
            )
        else:
            ion_departure = np.asarray(
                ion_departure_coefficient[(ion.element, ion.charge)],
                dtype=np.float64,
            )
            lower_departure = ion_departure
            upper_departure = ion_departure
        center = line.wavelength_vacuum_angstrom
        center_cm = center * 1.0e-8
        upper_level = levels[(ion.element, ion.charge)][line.upper_index]
        stark_rate_per_electron = (
            _classical_electron_stark_rate_per_electron(
                ion.charge,
                ion.ionization_energy_ev,
                upper_level.energy_wavenumber,
            )
            if include_classical_electron_stark else 0.0
        )
        exponential = np.exp(
            -PLANCK * LIGHT_SPEED
            / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        gaussian_sigma = center * np.sqrt(
            BOLTZMANN * atmosphere.temperature
            / (ion.atomic_mass_u * atomic_mass_unit * LIGHT_SPEED**2)
        )
        damping_rate = np.asarray([
            _impact_profile_damping_rate(
                line,
                upper_rate[upper_key],
                stark_rate_per_electron,
                float(electron_density),
                float(temperature),
                include_electron_stark=include_classical_electron_stark,
                element=ion.element,
                charge=ion.charge,
                tabulated_electron_stark_width_scale=(
                    tabulated_electron_stark_width_scale
                ),
                include_semiclassical_ovi_stark_widths=(
                    include_semiclassical_ovi_stark_widths
                ),
            )
            for electron_density, temperature in zip(
                atmosphere.electron_density, atmosphere.temperature
            )
        ])
        if (
            include_ovi_high_series_ion_dephasing
            and _is_ovi_high_series_formula4(line, ion.element, ion.charge)
        ):
            # A field realization loses phase memory at v/R0.  In the
            # damping-rate convention used by the Voigt profile, an angular
            # coherence HWHM of 1/tau corresponds to Gamma=2/tau.
            damping_rate += 2.0 * ion_motion_rate(ion.atomic_mass_u)
        lorentz_hwhm = (
            center_cm**2 * damping_rate / (4.0 * PI * LIGHT_SPEED) * 1.0e8
        )
        # A cutoff at 100 Lorentz HWHM loses less than one percent of a
        # normalized Voigt profile and is an excellent speed optimization
        # for ordinary metal lines.  It is not an optical-depth criterion,
        # however: a large ground-term column can leave those nominally small
        # wings photospherically thick.  Preserve wider support only for the
        # strong UV resonance transitions selected by the shared LTE/NLTE
        # policy; ordinary and optical lines retain the fast local cutoff.
        lower_level = levels[(ion.element, ion.charge)][line.lower_index]
        minimum_half_window = 0.0
        resonance_half_window = (
            strong_uv_resonance_minimum_half_window_angstrom(
                ion, line, lower_level
            )
            if extend_strong_uv_resonance_wings else 0.0
        )
        if resonance_half_window > 0.0:
            center_optical_depth = _line_center_vertical_optical_depth(
                atmosphere,
                center,
                integrated_cross_section * line.absorption_oscillator_strength,
                gaussian_sigma,
                lorentz_hwhm,
                lower_population / atmosphere.mass_density,
                lower_departure,
                upper_departure,
                exponential,
            )
            if center_optical_depth >= strong_uv_resonance_core_optical_depth:
                minimum_half_window = resonance_half_window
        static_frequency_scale = np.zeros(
            atmosphere.n_depth, dtype=np.float64
        )
        static_amplitude = np.zeros_like(static_frequency_scale)
        static_ion_motion_hwhm_beta = np.zeros_like(static_frequency_scale)
        static_parameters = (
            include_static_linear_stark
            and line.profile_formula == 4
            and len(line.profile_parameters) >= 6
            and line.profile_parameters[3] > 0.0
            and line.profile_parameters[4] > 0.0
            and line.profile_parameters[5] > 0.0
        )
        if static_parameters:
            stark_charge = line.profile_parameters[3]
            lower_n = line.profile_parameters[4]
            upper_n = line.profile_parameters[5]
            stark_sum = (
                upper_n * (upper_n - 1.0) + lower_n * (lower_n - 1.0)
            )
            dynamic_scale = _static_linear_stark_scale(
                static_linear_stark_frequency_scales,
                ion.element, ion.charge, lower_n, upper_n,
            )
            valid = microfield_scale > 0.0
            if dynamic_scale > 0.0 and stark_sum > 0.0 and np.any(valid):
                static_frequency_scale[valid] = (
                    dynamic_scale
                    * stark_sum
                    * microfield_scale[valid]
                    / (1.385 * stark_charge)
                )
                static_amplitude[valid] = (
                    0.0368
                    * stark_charge
                    * line.absorption_oscillator_strength
                    / (
                        dynamic_scale
                        * stark_sum
                        * microfield_scale[valid]
                    )
                )
                if include_ion_dynamic_stark_core:
                    static_ion_motion_hwhm_beta[valid] = (
                        ion_motion_rate(ion.atomic_mass_u)[valid]
                        / (2.0 * PI * static_frequency_scale[valid])
                    )

        profile_batch["center"].append(center)
        profile_batch["integrated_strength"].append(
            integrated_cross_section * line.absorption_oscillator_strength
        )
        profile_batch["gaussian_sigma"].append(gaussian_sigma)
        profile_batch["lorentz_hwhm"].append(lorentz_hwhm)
        profile_batch["minimum_half_window"].append(minimum_half_window)
        profile_batch["static_frequency_scale"].append(static_frequency_scale)
        profile_batch["static_amplitude"].append(static_amplitude)
        profile_batch["static_ion_motion_hwhm_beta"].append(
            static_ion_motion_hwhm_beta
        )
        profile_batch["static_pattern"].append(
            _hydrogenic_stark_pattern_key(
                line, hydrogenic_linear_stark_components
            ) if np.any(static_amplitude > 0.0) else None
        )
        profile_batch["population_scale"].append(
            lower_population / atmosphere.mass_density
        )
        profile_batch["lower_departure"].append(lower_departure)
        profile_batch["upper_departure"].append(upper_departure)
        profile_batch["exponential"].append(exponential)
        # Keep peak memory independent of the ultimately selected line list.
        if len(profile_batch["center"]) >= 256:
            flush_profile_batch()
    flush_profile_batch()
    return np.ascontiguousarray(absorption.T), np.ascontiguousarray(emissivity.T)


def default_light_metal_ionization_wavelength(
    photoionization_database: VernerPhotoionizationDatabase,
    *,
    elements: Iterable[str] = ("C", "O"),
    n_wavelength: int = 360,
    photoionization_threshold_data: Mapping[
        tuple[str, int], TlustyPhotoionizationThresholdData
    ] | None = None,
) -> FloatArray:
    """Return an EUV/X-ray grid resolving every requested ionization edge."""

    if n_wavelength < 120:
        raise ValueError("n_wavelength must be at least 120")
    symbols = tuple(element.strip().capitalize() for element in elements)
    fits = [
        fit
        for (element, _), fit in photoionization_database.fits.items()
        if element in symbols
    ]
    tabulated_threshold_frequency = np.asarray([
        frequency
        for (element, _), table in (
            ()
            if photoionization_threshold_data is None
            else photoionization_threshold_data.items()
        )
        if element in symbols
        for frequency in table.threshold_frequency_hz
        if frequency > 0.0
    ], dtype=np.float64)
    if not fits and tabulated_threshold_frequency.size == 0:
        raise ValueError("no requested photoionization fits are available")
    threshold = np.concatenate((np.asarray(
        [PLANCK * LIGHT_SPEED / (fit.threshold_energy_ev * EV_TO_ERG) * 1.0e8
         for fit in fits],
        dtype=np.float64,
    ), (
        PLANCK * LIGHT_SPEED / tabulated_threshold_frequency * 1.0e8
        if tabulated_threshold_frequency.size else np.asarray(())
    )))
    minimum = max(1.0, float(np.min(threshold)) / 30.0)
    maximum = max(2500.0, float(np.max(threshold)) * 1.25)
    edge_samples = np.concatenate(
        [value * np.asarray((0.96, 0.985, 0.997, 1.0, 1.003)) for value in threshold]
    )
    wavelength = np.unique(
        np.concatenate((np.geomspace(minimum, maximum, n_wavelength), edge_samples))
    )
    return np.ascontiguousarray(wavelength[wavelength > 0.0])


def _fractions_from_log_ratios(log_ratios: FloatArray) -> FloatArray:
    cumulative = np.concatenate(
        (np.zeros((1, log_ratios.shape[1])), np.cumsum(log_ratios, axis=0)),
        axis=0,
    )
    maximum = np.max(cumulative, axis=0, keepdims=True)
    weight = np.exp(np.clip(cumulative - maximum, -745.0, 0.0))
    return weight / np.sum(weight, axis=0, keepdims=True)


def _photoionization_rates_python(
    wavelength_angstrom: FloatArray,
    mean_intensity: FloatArray,
    cross_section: FloatArray,
) -> FloatArray:
    """Reference bound-free rates for cross sections shaped (level, wave)."""

    photon_factor = (
        4.0
        * np.pi
        * wavelength_angstrom[:, np.newaxis]
        * 1.0e-8
        / (PLANCK * LIGHT_SPEED)
    )
    return np.asarray([
        trapezoid(
            mean_intensity * local_cross_section[:, np.newaxis] * photon_factor,
            wavelength_angstrom,
            axis=0,
        )
        for local_cross_section in cross_section
    ])


def _photoionization_rates(
    wavelength_angstrom: FloatArray,
    mean_intensity: FloatArray,
    cross_section: FloatArray,
) -> FloatArray:
    """Dispatch element-independent bound-free rate integration to C."""

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    intensity = np.ascontiguousarray(mean_intensity, dtype=np.float64)
    cross_section = np.ascontiguousarray(cross_section, dtype=np.float64)
    compiled = None if _rt is None else getattr(_rt, "photoionization_rates", None)
    if compiled is None:
        return _photoionization_rates_python(
            wavelength, intensity, cross_section
        )
    rate = np.empty((cross_section.shape[0], intensity.shape[1]))
    compiled(wavelength, intensity, cross_section, rate)
    return rate


def _photoionization_rate(
    wavelength_angstrom: FloatArray,
    mean_intensity: FloatArray,
    cross_section: FloatArray,
) -> FloatArray:
    return _photoionization_rates(
        wavelength_angstrom,
        mean_intensity,
        np.asarray(cross_section, dtype=np.float64)[np.newaxis, :],
    )[0]


def _tmap_seaton_collisional_ionization_rate(
    electron_density: ArrayLike,
    temperature: ArrayLike,
    threshold_u: ArrayLike,
    threshold_cross_section_cm2: float,
    ion_charge: int,
    *,
    cross_section_includes_gbar: bool = False,
) -> FloatArray:
    """Evaluate TMAP User Guide A.2.4 without double-counting g-bar.

    Raw TMAD CBF records supply ``sigma_0`` and ``gbar`` independently.  By
    contrast, the public TLUSTY atoms used here put their effective product
    into the continuum record's ``OSC`` column.  The distinction matters by
    a factor 3.33 for C/O ions and is especially important for Rydberg-term
    continuum coupling.
    """

    density = np.asarray(electron_density, dtype=np.float64)
    local_temperature = np.asarray(temperature, dtype=np.float64)
    u = np.asarray(threshold_u, dtype=np.float64)
    gbar = (
        1.0
        if cross_section_includes_gbar
        else min(0.1 * (int(ion_charge) + 1), 0.3)
    )
    return np.ascontiguousarray(
        density
        * 1.55e13
        * gbar
        / np.sqrt(local_temperature)
        * np.exp(-u)
        / np.maximum(u, 1.0e-12)
        * max(float(threshold_cross_section_cm2), 0.0)
    )


def solve_light_metal_ionization_nlte(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    lte_state: MetalLTEState,
    photoionization_database: VernerPhotoionizationDatabase,
    wavelength_angstrom: ArrayLike,
    mean_intensity: ArrayLike,
    *,
    elements: Iterable[str] = ("C", "O"),
    minimum_active_lte_ion_fraction: float = 1.0e-12,
    active_ion_stage_margin: int = 1,
    photoionization_threshold_data: Mapping[
        tuple[str, int], TlustyPhotoionizationThresholdData
    ] | None = None,
    active_stage_range_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> LightMetalNLTEState:
    """Solve adjacent ion-stage statistical equilibrium at every depth.

    Radiative recombination is represented by the exact inverse rate required
    for detailed balance with the supplied LTE reference.  This includes
    spontaneous plus stimulated recombination in one effective downward
    coefficient.  Ground-state electron-impact ionization uses the same
    Seaton/TLUSTY approximation as the explicit-level solver, with inverse
    three-body recombination fixed by LTE detailed balance.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    intensity = np.asarray(mean_intensity, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if intensity.shape != (wavelength.size, atmosphere.n_depth):
        raise ValueError("mean_intensity must have shape (wavelength, depth)")
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise ValueError("mean_intensity must be finite and non-negative")
    if not 0.0 < minimum_active_lte_ion_fraction < 1.0:
        raise ValueError("minimum_active_lte_ion_fraction must lie in (0, 1)")
    if active_ion_stage_margin < 0:
        raise ValueError("active_ion_stage_margin must be non-negative")

    planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
    )
    photon_energy_ev = (
        PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / EV_TO_ERG
    )
    populations: dict[str, FloatArray] = {}
    lte_populations: dict[str, FloatArray] = {}
    departure: dict[tuple[str, int], FloatArray] = {}
    photo_rates: dict[tuple[str, int], FloatArray] = {}
    recombination_rates: dict[tuple[str, int], FloatArray] = {}
    collisional_rates: dict[tuple[str, int], FloatArray] = {}
    three_body_rates: dict[tuple[str, int], FloatArray] = {}
    maximum_recovery_error = 0.0
    active_stage_ranges: dict[str, tuple[int, int]] = {}
    active_stage_ranges_by_depth: dict[str, tuple[tuple[int, int], ...]] = {}
    tiny = np.finfo(np.float64).tiny
    requested_elements = tuple(
        raw_element.strip().capitalize() for raw_element in elements
    )

    for element in requested_elements:
        all_stages = atomic_database.ion_stages(element)
        lte = np.asarray(lte_state.ion_number_density[element], dtype=np.float64)
        if lte.shape != (len(all_stages), atmosphere.n_depth):
            raise ValueError(f"LTE ion ladder for {element} does not match database")
        total = np.maximum(
            np.asarray(lte_state.element_number_density[element]), tiny
        )
        maximum_lte_fraction = np.max(lte / total[np.newaxis, :], axis=1)
        override = (
            None
            if active_stage_range_overrides is None
            else active_stage_range_overrides.get(element)
        )
        if override is not None:
            first_charge, last_charge = map(int, override)
            if not 0 <= first_charge < last_charge < len(all_stages):
                raise ValueError(
                    f"invalid active ion-stage override for {element}: {override}"
                )
        else:
            supported = np.flatnonzero(
                maximum_lte_fraction >= minimum_active_lte_ion_fraction
            )
            if supported.size == 0:
                supported = np.asarray((int(np.argmax(maximum_lte_fraction)),))
            first_charge = max(0, int(supported[0]) - active_ion_stage_margin)
            last_charge = min(
                len(all_stages) - 1,
                int(supported[-1]) + active_ion_stage_margin,
            )
            if first_charge == last_charge:
                if last_charge < len(all_stages) - 1:
                    last_charge += 1
                elif first_charge > 0:
                    first_charge -= 1
        stages = all_stages[first_charge : last_charge + 1]
        active_stage_ranges[element] = (first_charge, last_charge)
        local_stage_ranges = []
        for depth in range(atmosphere.n_depth):
            if override is not None:
                local_stage_ranges.append((first_charge, last_charge))
                continue
            local_fraction = lte[:, depth] / total[depth]
            local_supported = np.flatnonzero(
                local_fraction >= minimum_active_lte_ion_fraction
            )
            if local_supported.size == 0:
                local_supported = np.asarray((int(np.argmax(local_fraction)),))
            local_first = max(
                0, int(local_supported[0]) - active_ion_stage_margin
            )
            local_last = min(
                len(all_stages) - 1,
                int(local_supported[-1]) + active_ion_stage_margin,
            )
            if local_first == local_last:
                if local_last < len(all_stages) - 1:
                    local_last += 1
                elif local_first > 0:
                    local_first -= 1
            local_stage_ranges.append((local_first, local_last))
        active_stage_ranges_by_depth[element] = tuple(local_stage_ranges)
        upward = []
        downward = []
        planck_upward = []
        for lower in stages[:-1]:
            fit = photoionization_database.fits.get((element, lower.charge))
            threshold_table = (
                None
                if photoionization_threshold_data is None
                else photoionization_threshold_data.get((element, lower.charge))
            )
            if fit is None and threshold_table is None:
                raise ValueError(
                    f"missing photoionization data for {element} {lower.charge:+d}"
                )
            if fit is not None:
                threshold_energy_ev = fit.threshold_energy_ev
                cross_section = fit.cross_section(photon_energy_ev)
                threshold_cross_section = float(fit.cross_section(np.asarray((
                    threshold_energy_ev * (1.0 + 1.0e-10),
                )))[0])
                threshold_cross_section_includes_gbar = False
            else:
                if lower.ionization_energy_ev is None:
                    raise ValueError(
                        f"missing ionization threshold for {element} "
                        f"{lower.charge:+d}"
                    )
                threshold_energy_ev = float(lower.ionization_energy_ev)
                threshold_frequency = threshold_energy_ev * EV_TO_ERG / PLANCK
                cross_section = threshold_table.cross_section(
                    photon_energy_ev * EV_TO_ERG / PLANCK,
                    threshold_frequency,
                    relative_tolerance=0.03,
                )
                threshold_cross_section = (
                    threshold_table.cross_section_for_threshold(
                        threshold_frequency, relative_tolerance=0.03
                    )
                )
                if cross_section is None or threshold_cross_section is None:
                    raise ValueError(
                        f"tabulated photoionization threshold does not match "
                        f"{element} {lower.charge:+d}"
                    )
                threshold_cross_section_includes_gbar = (
                    threshold_table.threshold_cross_section_includes_gbar
                )
            actual_rate = _photoionization_rate(wavelength, intensity, cross_section)
            thermal_rate = _photoionization_rate(wavelength, planck, cross_section)
            threshold_u = (
                threshold_energy_ev * EV_TO_ERG
                / (BOLTZMANN * atmosphere.temperature)
            )
            collision_rate = _tmap_seaton_collisional_ionization_rate(
                atmosphere.electron_density,
                atmosphere.temperature,
                threshold_u,
                threshold_cross_section,
                lower.charge,
                cross_section_includes_gbar=(
                    threshold_cross_section_includes_gbar
                ),
            )
            population_ratio = lte[lower.charge] / np.maximum(
                lte[lower.charge + 1], tiny
            )
            radiative_reverse_rate = thermal_rate * population_ratio
            three_body_rate = collision_rate * population_ratio
            upward_rate = actual_rate + collision_rate
            reverse_rate = radiative_reverse_rate + three_body_rate
            upward.append(np.maximum(upward_rate, tiny))
            downward.append(np.maximum(reverse_rate, tiny))
            planck_upward.append(np.maximum(thermal_rate + collision_rate, tiny))
            photo_rates[(element, lower.charge)] = np.asarray(actual_rate)
            recombination_rates[(element, lower.charge + 1)] = np.asarray(
                radiative_reverse_rate
            )
            collisional_rates[(element, lower.charge)] = np.asarray(collision_rate)
            three_body_rates[(element, lower.charge + 1)] = np.asarray(
                three_body_rate
            )

        log_ratio = np.log(np.stack(upward)) - np.log(np.stack(downward))
        population = np.array(lte, copy=True)
        for depth, (local_first, local_last) in enumerate(local_stage_ranges):
            local_log_ratio = log_ratio[
                local_first - first_charge : local_last - first_charge,
                depth : depth + 1,
            ]
            local_fraction = _fractions_from_log_ratios(local_log_ratio)[:, 0]
            local_total = np.sum(lte[local_first : local_last + 1, depth])
            population[local_first : local_last + 1, depth] = (
                local_fraction * local_total
            )
        populations[element] = np.asarray(population)
        lte_populations[element] = lte
        for ion in all_stages:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                raw_departure = (
                    population[ion.charge]
                    / np.maximum(lte[ion.charge], tiny)
                )
            departure[(element, ion.charge)] = np.asarray(
                np.nan_to_num(
                    raw_departure,
                    nan=0.0,
                    posinf=_MAX_FINITE_DEPARTURE,
                    neginf=0.0,
                ).clip(0.0, _MAX_FINITE_DEPARTURE)
            )

        planck_log_ratio = np.log(np.stack(planck_upward)) - np.log(
            np.stack(downward)
        )
        recovered = np.array(lte, copy=True)
        for depth, (local_first, local_last) in enumerate(local_stage_ranges):
            local_log_ratio = planck_log_ratio[
                local_first - first_charge : local_last - first_charge,
                depth : depth + 1,
            ]
            local_fraction = _fractions_from_log_ratios(local_log_ratio)[:, 0]
            local_total = np.sum(lte[local_first : local_last + 1, depth])
            recovered[local_first : local_last + 1, depth] = (
                local_fraction * local_total
            )
        recovery_error = np.max(
            np.abs(recovered - lte) / np.maximum(total[np.newaxis, :], tiny)
        )
        maximum_recovery_error = max(maximum_recovery_error, float(recovery_error))

    return LightMetalNLTEState(
        ion_number_density=MappingProxyType(populations),
        lte_ion_number_density=MappingProxyType(lte_populations),
        ion_departure_coefficient=MappingProxyType(departure),
        photoionization_rate=MappingProxyType(photo_rates),
        radiative_recombination_rate=MappingProxyType(recombination_rates),
        collisional_ionization_rate=MappingProxyType(collisional_rates),
        three_body_recombination_rate=MappingProxyType(three_body_rates),
        maximum_lte_recovery_error=maximum_recovery_error,
        metadata={
            "model_atom": (
                "adjacent ion-stage statistical equilibrium for "
                + ", ".join(requested_elements)
            ),
            "photoionization": photoionization_database.source,
            "recombination": "effective inverse rate from LTE detailed balance",
            "collisional_ionization": (
                "TLUSTY ICOL=0 Seaton ground-state rate with inverse "
                "three-body detailed balance"
            ),
            "excitation_within_ions": "LTE Boltzmann reference",
            "active_stage_ranges": active_stage_ranges,
            "active_stage_ranges_by_depth": active_stage_ranges_by_depth,
            "active_stage_range_overrides": (
                {} if active_stage_range_overrides is None
                else dict(active_stage_range_overrides)
            ),
            "minimum_active_lte_ion_fraction": minimum_active_lte_ion_fraction,
            "active_ion_stage_margin": active_ion_stage_margin,
            "lte_recovery_maximum_fractional_error": maximum_recovery_error,
        },
    )


def reduced_light_metal_wavelength(
    atomic_database: AtomicDatabase,
    element: str,
    levels_per_charge: Mapping[int, int],
    *,
    n_continuum_wavelength: int = 320,
    photoionization_threshold_data: Mapping[
        int, TlustyPhotoionizationThresholdData
    ] | None = None,
    formal_atomic_database: AtomicDatabase | None = None,
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None = None,
    lte_bound_bound_couplings: Iterable[
        TmadLTEBoundBoundCoupling
    ] | None = None,
    effective_dielectronic_couplings: Iterable[
        TmadEffectiveDielectronicCoupling
    ] | None = None,
    line_velocity_samples_kms: ArrayLike | None = None,
) -> FloatArray:
    """Return a transfer grid for one reduced explicit-level model atom."""

    if n_continuum_wavelength < 120:
        raise ValueError("n_continuum_wavelength must be at least 120")
    symbol = element.strip().capitalize()
    stages = {ion.charge: ion for ion in atomic_database.ion_stages(symbol)}
    thresholds = []
    opacity_project_samples = []
    line_centers = []
    for charge, count in levels_per_charge.items():
        ion = stages[charge]
        selected = tuple(sorted(ion.levels, key=lambda level: level.energy_wavenumber)[:count])
        selected_index = {level.index for level in selected}
        if ion.ionization_energy_ev is not None:
            for level in selected:
                threshold_ev = (
                    ion.ionization_energy_ev
                    - level.energy_wavenumber * PLANCK * LIGHT_SPEED / EV_TO_ERG
                )
                if threshold_ev > 0.05:
                    threshold_wavelength = (
                        PLANCK * LIGHT_SPEED
                        / (threshold_ev * EV_TO_ERG)
                        * 1.0e8
                    )
                    thresholds.append(threshold_wavelength)
                    threshold_table = (
                        None
                        if photoionization_threshold_data is None
                        else photoionization_threshold_data.get(charge)
                    )
                    if threshold_table is not None:
                        threshold_frequency = (
                            threshold_ev * EV_TO_ERG / PLANCK
                        )
                        match = threshold_table._matching_index(
                            threshold_frequency,
                            0.02,
                            level_label=level.label,
                        )
                        if match is not None:
                            fit_x = threshold_table.log_frequency_ratio[match]
                            if fit_x is not None:
                                # The OP tables are piecewise linear in
                                # log(nu/nu0).  Put every tabulated knot on
                                # the transfer grid, plus interval midpoints,
                                # so narrow resonant structure contributes to
                                # the photoionization integral instead of
                                # being missed by the generic logarithmic
                                # continuum grid.
                                samples_x = np.unique(np.concatenate((
                                    fit_x,
                                    0.5 * (fit_x[:-1] + fit_x[1:]),
                                )))
                                opacity_project_samples.extend(
                                    threshold_wavelength / 10.0**samples_x
                                )
        for transition in ion.transitions:
            if (
                transition.lower_index in selected_index
                and transition.upper_index in selected_index
                and transition.einstein_a > 0.0
            ):
                components = _mapped_fine_structure_components(
                    symbol,
                    charge,
                    transition,
                    formal_atomic_database,
                    formal_level_mapping,
                )
                line_centers.extend(
                    component.wavelength_vacuum_angstrom
                    for component in components
                )
                if not components:
                    line_centers.append(transition.wavelength_vacuum_angstrom)
    if lte_bound_bound_couplings is not None:
        line_centers.extend(
            coupling.transition.wavelength_vacuum_angstrom
            for coupling in lte_bound_bound_couplings
        )
    if effective_dielectronic_couplings is not None:
        line_centers.extend(
            LIGHT_SPEED / coupling.transition_frequency_hz * 1.0e8
            for coupling in effective_dielectronic_couplings
        )
    if not thresholds:
        raise ValueError("reduced atom has no bound-free thresholds")
    threshold = np.asarray(thresholds)
    minimum = max(1.0, float(np.min(threshold)) / 30.0)
    maximum = max(10_000.0, float(np.max(threshold)) * 1.08)
    edges = np.concatenate(
        [value * np.asarray((0.97, 0.99, 0.998, 1.0, 1.002)) for value in threshold]
    )
    # Statistical-equilibrium bound-bound rates require profile-averaged J,
    # not a single intensity sample at line center. Resolve each thermal core
    # on a symmetric velocity grid; the solver performs the final
    # depth-dependent Gaussian quadrature on these samples.
    velocity_samples_kms = np.asarray(
        (
            -1200.0, -600.0, -300.0, -150.0, -80.0, -40.0, -20.0,
            -10.0, 0.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0,
            600.0, 1200.0,
        )
        if line_velocity_samples_kms is None
        else line_velocity_samples_kms,
        dtype=np.float64,
    )
    if (
        velocity_samples_kms.ndim != 1
        or velocity_samples_kms.size < 3
        or np.any(~np.isfinite(velocity_samples_kms))
        or np.any(np.diff(velocity_samples_kms) <= 0.0)
        or not np.any(velocity_samples_kms == 0.0)
    ):
        raise ValueError(
            "line_velocity_samples_kms must be a finite increasing vector "
            "of at least three samples including zero"
        )
    resolved_lines = (
        np.concatenate(
            [
                center * (1.0 + velocity_samples_kms / 299_792.458)
                for center in line_centers
            ]
        )
        if line_centers
        else np.asarray([], dtype=np.float64)
    )
    wavelength = np.unique(
        np.concatenate(
            (
                np.geomspace(minimum, maximum, n_continuum_wavelength),
                edges,
                np.asarray(opacity_project_samples, dtype=np.float64),
                resolved_lines,
            )
        )
    )
    return np.ascontiguousarray(wavelength[wavelength > 0.0])


def _mali_radiative_occupations(photon, inverse, diagonal, old_source):
    """Positive MALI rates with the same unpreconditioned fixed point.

    The two subtracted terms must use one lambda. Independently clipping
    the external photon occupation breaks the rate cancellation, especially
    when the profile-integrated thermodynamic inverse differs from 1+J.
    """
    if old_source<0 or not np.isfinite(old_source):
        return photon,inverse
    coefficient=min(max(diagonal,0.),.999,inverse/(1.+old_source))
    if old_source>0:
        coefficient=min(coefficient,photon/old_source)
    coefficient*=1.-16.*np.finfo(float).eps
    return photon-coefficient*old_source,inverse-coefficient*(1.+old_source)


def _positive_rate_equilibrium(matrix, conservation_weight, total_population):
    """Dispatch the positive rate solve to C, retaining the NumPy reference."""
    compiled = None if _rt is None else getattr(_rt, "positive_rate_equilibrium", None)
    if compiled is None:
        return _positive_rate_equilibrium_python(matrix, conservation_weight, total_population)
    rates = np.ascontiguousarray(matrix, dtype=np.float64)
    weights = np.ascontiguousarray(conservation_weight, dtype=np.float64)
    population = np.empty(len(rates), dtype=np.float64)
    try:
        compiled(rates, weights, float(total_population), population)
    except RuntimeError as exc:
        raise NonphysicalPopulationError(str(exc)) from exc
    return population


def _positive_rate_equilibrium_python(matrix, conservation_weight, total_population):
    """Solve a connected rate system using subtraction-free GTH elimination.

    Off-diagonal entry (i,j) is the rate j -> i. Eliminating a state only
    adds positive indirect transitions. Unlike Gaussian elimination of the
    generator's nearly cancelling diagonal, this retains weak ionization
    links when fast bound-bound rates span many orders of magnitude.
    """
    rates=np.array(matrix,dtype=float,copy=True)
    np.fill_diagonal(rates,0.)
    if np.any(~np.isfinite(rates)) or np.any(rates<0):
        raise NonphysicalPopulationError("statistical-equilibrium transition rates must be finite and nonnegative")
    count=len(rates)
    exits=np.zeros(count)
    for k in range(count-1,0,-1):
        exits[k]=np.sum(rates[:k,k])
        if exits[k]<=0 or not np.isfinite(exits[k]):
            raise NonphysicalPopulationError("disconnected statistical-equilibrium rate system")
        probabilities=rates[:k,k]/exits[k]
        rates[:k,:k]+=probabilities[:,None]*rates[k,:k][None,:]
        np.fill_diagonal(rates[:k,:k],0.)
    population=np.ones(count)
    for k in range(1,count):
        population[k]=np.dot(rates[k,:k],population[:k])/exits[k]
        # Homogeneous normalization avoids overflow without modifying ratios.
        largest=float(np.max(population[:k+1]))
        if largest>1e100:population[:k+1]/=largest
    population*=total_population/np.dot(conservation_weight,population)
    if np.any(~np.isfinite(population)) or np.any(population<0):
        raise NonphysicalPopulationError("nonphysical statistical-equilibrium populations")
    return population


def solve_reduced_light_metal_levels_nlte(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    lte_state: MetalLTEState,
    photoionization_database: VernerPhotoionizationDatabase,
    wavelength_angstrom: ArrayLike,
    mean_intensity: ArrayLike,
    element: str,
    levels_per_charge: Mapping[int, int],
    *,
    photoionization_threshold_data: Mapping[
        int, TlustyPhotoionizationThresholdData
    ] | None = None,
    formal_level_mapping: Mapping[
        tuple[str, int, int], tuple[tuple[str, int, int], ...]
    ] | None = None,
    formal_atomic_database: AtomicDatabase | None = None,
    formal_lte_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None,
    continuum_parent_mapping: Mapping[
        tuple[str, int, int], tuple[str, int, int]
    ] | None = None,
    lte_level_reservoir: Mapping[
        tuple[str, int, int], tuple[tuple[int, float, float], ...]
    ] | None = None,
    lte_bound_bound_couplings: Iterable[TmadLTEBoundBoundCoupling] | None = None,
    effective_dielectronic_couplings: Iterable[
        TmadEffectiveDielectronicCoupling
    ] | None = None,
    collision_data: Mapping[
        tuple[str, int, int, int],
        tuple[int, tuple[float, ...]]
        | ChiantiTermCollisionStrength
        | ConstantEffectiveCollisionStrength
        | TabulatedElectronExcitationRateCoefficient
        | PSM20AngularMomentumMixingCollision
        | BTMQuadrupoleAngularMomentumMixingCollision,
    ] | None = None,
    static_linear_stark_frequency_scales: Mapping[
        tuple[str, int, int, int], float
    ] | None = None,
    include_semiclassical_ovi_stark_widths: bool = False,
    include_ovi_high_series_ion_dephasing: bool = False,
    include_ion_dynamic_stark_core: bool = False,
    hydrogenic_linear_stark_components: bool = False,
    electron_excitation_collision_scale: float = 1.0,
    approximate_lambda_diagonal: ArrayLike | None = None,
    previous_population_state: ReducedLightMetalLevelState | None = None,
) -> ReducedLightMetalLevelState:
    """Solve an explicit-level bound-bound/bound-free statistical equilibrium.

    The selected levels are the lowest-energy Stout levels of each requested
    ion.  Their LTE reference populations use the *full* Stout partition
    function, so omitted levels remain an inert LTE reservoir rather than
    being spuriously poured into the explicit states.  Ground-state
    photoionization uses the Verner fit; excited levels use an explicitly
    labelled hydrogenic Kramers approximation.  Allowed electron-impact
    excitation uses TLUSTY's Van Regemorter prescription.
    """

    symbol = element.strip().capitalize()
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    intensity = np.asarray(mean_intensity, dtype=np.float64)
    if intensity.shape != (wavelength.size, atmosphere.n_depth):
        raise ValueError("mean_intensity must have shape (wavelength, depth)")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be strictly increasing")
    collision_scale = float(electron_excitation_collision_scale)
    if not np.isfinite(collision_scale) or collision_scale < 0.0:
        raise ValueError(
            "electron_excitation_collision_scale must be finite and non-negative"
        )
    lambda_diagonal = (
        None
        if approximate_lambda_diagonal is None
        else np.asarray(approximate_lambda_diagonal, dtype=np.float64)
    )
    if lambda_diagonal is not None:
        if lambda_diagonal.shape != intensity.shape:
            raise ValueError(
                "approximate_lambda_diagonal must match mean_intensity"
            )
        if np.any(~np.isfinite(lambda_diagonal)) or np.any(
            lambda_diagonal < 0.0
        ):
            raise ValueError(
                "approximate_lambda_diagonal must be finite and non-negative"
            )
    stages = {ion.charge: ion for ion in atomic_database.ion_stages(symbol)}
    charges = tuple(sorted(int(value) for value in levels_per_charge))
    if not charges or charges != tuple(range(charges[0], charges[-1] + 1)):
        raise ValueError("levels_per_charge must contain consecutive ion stages")
    if any(charge not in stages or levels_per_charge[charge] < 1 for charge in charges):
        raise ValueError("requested ion stage is unavailable or has no levels")

    selected_levels = {}
    state_keys = []
    state_charge = []
    state_level = []
    state_index = {}
    for charge in charges:
        ion = stages[charge]
        levels = tuple(
            sorted(ion.levels, key=lambda level: level.energy_wavenumber)[
                : int(levels_per_charge[charge])
            ]
        )
        selected_levels[charge] = levels
        for level in levels:
            key = (symbol, charge, level.index)
            state_index[key] = len(state_keys)
            state_keys.append(key)
            state_charge.append(charge)
            state_level.append(level)
    n_state = len(state_keys)
    previous_population = None
    previous_by_key = None
    if previous_population_state is not None:
        if previous_population_state.element != symbol:
            raise ValueError("previous population state is for another element")
        previous_by_key = {
            key: previous_population_state.population_density[index]
            for index, key in enumerate(previous_population_state.level_key)
        }
    temperature = atmosphere.temperature
    electron_density = atmosphere.electron_density
    tiny = np.finfo(np.float64).tiny

    lte_level = np.zeros((n_state, atmosphere.n_depth), dtype=np.float64)
    lte_ion = lte_state.ion_number_density[symbol]
    for charge in charges:
        levels = selected_levels[charge]
        energy = np.asarray([level.energy_wavenumber for level in levels])
        weight = np.asarray([level.statistical_weight for level in levels])
        boltzmann = weight[:, np.newaxis] * np.exp(
            -energy[:, np.newaxis] * PLANCK * LIGHT_SPEED
            / (BOLTZMANN * temperature[np.newaxis, :])
        )
        # This denominator must be the complete ion partition function used
        # by the opacity calculation.  Normalizing over only the retained
        # levels assigned all omitted high-level populations to the reduced
        # atom (by factors up to roughly 100 in hot PG1159 surface layers),
        # while the formal solution still referenced the full partition.
        boltzmann /= lte_state.partition_function[(symbol, charge)][np.newaxis, :]
        for local_index, level in enumerate(levels):
            lte_level[state_index[(symbol, charge, level.index)]] = (
                lte_ion[charge] * boltzmann[local_index]
            )
    if previous_by_key is not None:
        # A promoted-atom continuation normally adds only a few terms.  Keep
        # every shared converged population and initialize genuinely new
        # terms in LTE instead of discarding the entire warm start because
        # the key sets are not identical.
        previous_population = np.asarray([
            previous_by_key.get(key, lte_level[index])
            for index, key in enumerate(state_keys)
        ], dtype=np.float64)
    represented_population = np.sum(lte_level, axis=0)
    # TMAD keeps selected high levels in LTE with the departure coefficient
    # of a named continuum parent.  They are not rate-equation unknowns, but
    # they must appear in particle conservation.  Omitting this reservoir
    # allows optical upper and lower terms to acquire arbitrarily different
    # normalizations even in an otherwise exact TMAD atom.
    reservoir_lte_by_parent = np.zeros_like(lte_level)
    if lte_level_reservoir is not None:
        for parent_key, reservoir_levels in lte_level_reservoir.items():
            if parent_key not in state_index:
                raise ValueError(
                    f"LTE reservoir parent {parent_key!r} is absent from the reduced atom"
                )
            parent_index = state_index[parent_key]
            for reservoir_charge, energy_wavenumber, statistical_weight in reservoir_levels:
                reservoir_lte_by_parent[parent_index] += (
                    lte_ion[reservoir_charge]
                    * statistical_weight
                    * np.exp(
                        -energy_wavenumber * PLANCK * LIGHT_SPEED
                        / (BOLTZMANN * temperature)
                    )
                    / lte_state.partition_function[(symbol, reservoir_charge)]
                )
    represented_system_population = (
        represented_population + np.sum(reservoir_lte_by_parent, axis=0)
    )
    conservation_weight = 1.0 + reservoir_lte_by_parent / np.maximum(
        lte_level, tiny
    )

    # An LTE-reservoir term has n_lte / n_lte* = n_parent / n_parent*.
    # Consequently a physical explicit<->LTE transition can be represented
    # exactly by a pair between the explicit state and the named parent, with
    # the LTE endpoint's rate multiplied by n_lte*/n_parent*.  Keeping these
    # one-sided rates closes high-level cascades without turning every formal
    # line-formation term into an expensive rate-equation unknown.
    reservoir_bound_bound = []
    if lte_bound_bound_couplings is not None:
        for coupling in lte_bound_bound_couplings:
            if coupling.explicit_level_key not in state_index:
                raise ValueError(
                    "LTE bound-bound explicit endpoint "
                    f"{coupling.explicit_level_key!r} is absent from the reduced atom"
                )
            if coupling.lte_parent_key not in state_index:
                raise ValueError(
                    "LTE bound-bound parent "
                    f"{coupling.lte_parent_key!r} is absent from the reduced atom"
                )
            explicit = state_index[coupling.explicit_level_key]
            parent = state_index[coupling.lte_parent_key]
            lte_population = (
                lte_ion[coupling.lte_charge]
                * coupling.lte_statistical_weight
                * np.exp(
                    -coupling.lte_energy_wavenumber
                    * PLANCK * LIGHT_SPEED
                    / (BOLTZMANN * temperature)
                )
                / lte_state.partition_function[(symbol, coupling.lte_charge)]
            )
            lte_to_parent_ratio = lte_population / np.maximum(
                lte_level[parent], tiny
            )
            lte_term = AtomicLevel(
                0,
                coupling.lte_energy_wavenumber,
                coupling.lte_statistical_weight,
                coupling.lte_level_label,
            )
            reservoir_bound_bound.append((
                explicit,
                parent,
                coupling,
                lte_term,
                lte_to_parent_ratio,
            ))

    # TMAP's effective RDI treatment eliminates the autoionizing level and
    # inserts a direct radiative pair between the stabilized bound level and
    # the recombining-ion parent (Werner et al. 2003, Eqs. 22--23).  Expressed
    # with photon occupation nbar, the upward coefficient is
    #
    #   8 pi^2 e^2 / (m_e c^3) f nu^2 nbar,
    #
    # and the downward coefficient is the same prefactor times
    # exp(-h nu/kT) (1+nbar) n_i^*/n_parent^*.  At J=B these obey detailed
    # balance algebraically, without ever adding the auto state to the rate
    # vector.
    effective_dielectronic = []
    if effective_dielectronic_couplings is not None:
        local_planck = planck_lambda_angstrom(wavelength[:,None],temperature[None,:])
        radiation_ratio = np.divide(intensity,local_planck,
            out=np.zeros_like(intensity),where=local_planck>0)
        for coupling in effective_dielectronic_couplings:
            if coupling.lower_level_key not in state_index:
                continue
            if coupling.continuum_parent_key not in state_index:
                continue
            frequency = float(coupling.transition_frequency_hz)
            if frequency <= 0.0 or coupling.oscillator_strength <= 0.0:
                continue
            center = LIGHT_SPEED / frequency * 1.0e8
            mean_lambda = planck_lambda_angstrom(np.asarray(center),temperature) * np.asarray([
                np.interp(center, wavelength, radiation_ratio[:, depth])
                for depth in range(atmosphere.n_depth)
            ])
            center_cm = center * 1.0e-8
            vacuum_lambda = (
                2.0 * PLANCK * LIGHT_SPEED**2 / center_cm**5 * 1.0e-8
            )
            photon_occupation = np.maximum(mean_lambda / vacuum_lambda, 0.0)
            rate_scale = (
                8.0
                * PI**2
                * ELEMENTARY_CHARGE_ESU**2
                / (ELECTRON_MASS * LIGHT_SPEED**3)
                * coupling.oscillator_strength
                * frequency**2
            )
            effective_dielectronic.append((
                state_index[coupling.lower_level_key],
                state_index[coupling.continuum_parent_key],
                frequency,
                rate_scale,
                photon_occupation,
            ))

    # Bound-bound transitions between explicitly retained levels.
    bound_bound = []
    selected_upper_rate: dict[tuple[int, int], float] = {}
    for charge in charges:
        ion = stages[charge]
        selected = {level.index for level in selected_levels[charge]}
        for transition in ion.transitions:
            if transition.upper_index in selected:
                key = (charge, transition.upper_index)
                selected_upper_rate[key] = (
                    selected_upper_rate.get(key, 0.0) + transition.einstein_a
                )
            if (
                transition.lower_index in selected
                and transition.upper_index in selected
                and transition.einstein_a > 0.0
            ):
                lower = state_index[(symbol, charge, transition.lower_index)]
                upper = state_index[(symbol, charge, transition.upper_index)]
                components = _mapped_fine_structure_components(
                    symbol,
                    charge,
                    transition,
                    formal_atomic_database,
                    formal_level_mapping,
                )
                bound_bound.append((lower, upper, transition, components))

    # R-matrix data contain important same-parity and spin-changing electron
    # collisions that have no radiative counterpart.  They still belong in
    # the rate matrix; restricting collision_data to the RBB loop silently
    # discarded precisely those thermalizing links.
    radiative_collision_keys = {
        (
            symbol,
            int(state_charge[lower]),
            state_level[lower].index,
            state_level[upper].index,
        )
        for lower, upper, _, _ in bound_bound
    }
    collision_only = []
    if collision_data is not None:
        for key, record in collision_data.items():
            if (
                key in radiative_collision_keys
                or not isinstance(
                    record,
                    (
                        tuple,
                        ChiantiTermCollisionStrength,
                        ConstantEffectiveCollisionStrength,
                        TabulatedElectronExcitationRateCoefficient,
                        PSM20AngularMomentumMixingCollision,
                        BTMQuadrupoleAngularMomentumMixingCollision,
                    ),
                )
            ):
                continue
            lower_key = (key[0], key[1], key[2])
            upper_key = (key[0], key[1], key[3])
            if lower_key in state_index and upper_key in state_index:
                collision_only.append((
                    state_index[lower_key], state_index[upper_key], record
                ))

    formal_upper_rate: dict[tuple[int, int], float] = {}
    formal_level_weight: dict[tuple[int, int], float] = {}
    formal_level_by_index: dict[tuple[int, int], AtomicLevel] = {}
    if formal_atomic_database is not None:
        for charge in charges:
            formal_ion = formal_atomic_database.ions.get((symbol, charge))
            if formal_ion is None:
                continue
            for level in formal_ion.levels:
                formal_level_weight[(charge, level.index)] = level.statistical_weight
                formal_level_by_index[(charge, level.index)] = level
            for component in formal_ion.transitions:
                key = (charge, component.upper_index)
                formal_upper_rate[key] = (
                    formal_upper_rate.get(key, 0.0) + component.einstein_a
                )

    # Sample every population-atom line profile in one numerical batch.  The
    # resulting means depend only on the numeric profile parameters and thus
    # use the same compiled kernel for C, O, and future metal atoms.  Atomic
    # term mapping and rate semantics deliberately remain here in Python.
    component_range: list[tuple[int, int]] = []
    component_center: list[float] = []
    component_gaussian: list[FloatArray] = []
    component_lorentz: list[FloatArray] = []
    component_weight: list[float] = []
    component_integrated_strength: list[float] = []
    component_static_frequency_scale: list[FloatArray] = []
    component_static_amplitude: list[FloatArray] = []
    component_static_ion_motion_hwhm_beta: list[FloatArray] = []
    component_static_pattern: list[tuple[int, int] | None] = []
    atomic_mass_unit = 1.660_539_068_92e-24
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )

    # The formula-4 radiative rates require an ionic microfield estimate.  The
    # level solver has the complete LTE ion ladder but not the outer
    # iteration's ion-stage departures, so use its charge closure and assign
    # any unrepresented electrons to He++ (the dominant background in a
    # PG1159 atmosphere).  A direct PG1424 audit found that using the outer
    # C/O departures changes this scale by less than 0.5 per cent, well below
    # the uncertainty of the approximate formula-4 profile itself.
    rate_microfield_charge_sum = np.zeros(atmosphere.n_depth, dtype=np.float64)
    rate_represented_electron_density = np.zeros_like(
        rate_microfield_charge_sum
    )
    for rate_element, ion_population in lte_state.ion_number_density.items():
        for ion_charge in range(1, ion_population.shape[0]):
            population = ion_population[ion_charge]
            rate_microfield_charge_sum += ion_charge**1.5 * population
            rate_represented_electron_density += ion_charge * population
    rate_helium_electron_density = np.maximum(
        electron_density - rate_represented_electron_density, 0.0
    )
    rate_microfield_charge_sum += (
        np.sqrt(2.0) * rate_helium_electron_density
    )
    rate_microfield_scale = np.maximum(
        rate_microfield_charge_sum, 0.0
    ) ** (2.0 / 3.0)
    rate_ion_motion = (
        _ion_microfield_motion_rate(
            atmosphere,
            lte_state,
            stages[next(iter(charges))].atomic_mass_u,
        )
        if (
            include_ion_dynamic_stark_core
            or include_ovi_high_series_ion_dephasing
        )
        else np.zeros(atmosphere.n_depth, dtype=np.float64)
    )

    def static_rate_profile_parameters(
        line: AtomicTransition,
        charge: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        frequency_scale = np.zeros(atmosphere.n_depth, dtype=np.float64)
        static_amplitude = np.zeros_like(frequency_scale)
        ion_motion_hwhm_beta = np.zeros_like(frequency_scale)
        if (
            line.profile_formula != 4
            or len(line.profile_parameters) < 6
            or line.profile_parameters[3] <= 0.0
            or line.profile_parameters[4] <= 0.0
            or line.profile_parameters[5] <= 0.0
        ):
            return frequency_scale, static_amplitude, ion_motion_hwhm_beta
        stark_charge = line.profile_parameters[3]
        lower_n = line.profile_parameters[4]
        upper_n = line.profile_parameters[5]
        stark_sum = (
            upper_n * (upper_n - 1.0)
            + lower_n * (lower_n - 1.0)
        )
        dynamic_scale = _static_linear_stark_scale(
            static_linear_stark_frequency_scales, symbol, charge, lower_n, upper_n,
        )
        valid = rate_microfield_scale > 0.0
        if dynamic_scale > 0.0 and stark_sum > 0.0 and np.any(valid):
            frequency_scale[valid] = (
                dynamic_scale
                * stark_sum
                * rate_microfield_scale[valid]
                / (1.385 * stark_charge)
            )
            static_amplitude[valid] = (
                0.0368
                * stark_charge
                * line.absorption_oscillator_strength
                / (
                    dynamic_scale
                    * stark_sum
                    * rate_microfield_scale[valid]
                )
            )
            if include_ion_dynamic_stark_core:
                ion_motion_hwhm_beta[valid] = (
                    rate_ion_motion[valid]
                    / (2.0 * PI * frequency_scale[valid])
                )
        return frequency_scale, static_amplitude, ion_motion_hwhm_beta

    for lower, upper, transition, fine_components in bound_bound:
        first_component = len(component_center)
        ion = stages[int(state_charge[lower])]
        upper_level = state_level[upper]
        for rate_line in fine_components or (transition,):
            rate_center = rate_line.wavelength_vacuum_angstrom
            rate_center_cm = rate_center * 1.0e-8
            if fine_components:
                rate_upper = formal_upper_rate.get(
                    (ion.charge, rate_line.upper_index), rate_line.einstein_a
                )
                rate_upper_level = formal_level_by_index[
                    (ion.charge, rate_line.upper_index)
                ]
            else:
                rate_upper = selected_upper_rate.get(
                    (int(state_charge[upper]), upper_level.index),
                    transition.einstein_a,
                )
                rate_upper_level = upper_level
            gaussian = rate_center * np.sqrt(
                BOLTZMANN * temperature
                / (ion.atomic_mass_u * atomic_mass_unit * LIGHT_SPEED**2)
            )
            stark_rate = _classical_electron_stark_rate_per_electron(
                ion.charge,
                ion.ionization_energy_ev,
                rate_upper_level.energy_wavenumber,
            )
            damping = np.asarray([
                _impact_profile_damping_rate(
                    rate_line,
                    rate_upper,
                    stark_rate,
                    float(local_electron_density),
                    float(local_temperature),
                    element=ion.element,
                    charge=ion.charge,
                    include_semiclassical_ovi_stark_widths=(
                        include_semiclassical_ovi_stark_widths
                    ),
                )
                for local_electron_density, local_temperature in zip(
                    electron_density, temperature
                )
            ])
            if (
                include_ovi_high_series_ion_dephasing
                and _is_ovi_high_series_formula4(
                    rate_line, ion.element, ion.charge
                )
            ):
                damping += 2.0 * rate_ion_motion
            component_center.append(rate_center)
            component_gaussian.append(gaussian)
            component_lorentz.append(
                rate_center_cm**2 * damping
                / (4.0 * PI * LIGHT_SPEED) * 1.0e8
            )
            (
                static_scale,
                static_amplitude,
                static_ion_motion_hwhm_beta,
            ) = static_rate_profile_parameters(rate_line, ion.charge)
            component_integrated_strength.append(
                integrated_cross_section
                * rate_line.absorption_oscillator_strength
            )
            component_static_frequency_scale.append(static_scale)
            component_static_amplitude.append(static_amplitude)
            component_static_ion_motion_hwhm_beta.append(
                static_ion_motion_hwhm_beta
            )
            component_static_pattern.append(
                _hydrogenic_stark_pattern_key(
                    rate_line, hydrogenic_linear_stark_components
                ) if np.any(static_amplitude > 0.0) else None
            )
            component_weight.append(
                formal_level_weight.get(
                    (ion.charge, rate_line.lower_index), 1.0
                ) * rate_line.absorption_oscillator_strength
            )
        component_range.append((first_component, len(component_center)))

    reservoir_component_range: list[tuple[int, int]] = []
    for explicit, _, coupling, reservoir_level, _ in reservoir_bound_bound:
        first_component = len(component_center)
        transition = coupling.transition
        ion = stages[coupling.lte_charge]
        explicit_level = state_level[explicit]
        upper_level = (
            reservoir_level if coupling.explicit_is_lower else explicit_level
        )
        lower_level = (
            explicit_level if coupling.explicit_is_lower else reservoir_level
        )
        rate_center = transition.wavelength_vacuum_angstrom
        rate_center_cm = rate_center * 1.0e-8
        gaussian = rate_center * np.sqrt(
            BOLTZMANN * temperature
            / (ion.atomic_mass_u * atomic_mass_unit * LIGHT_SPEED**2)
        )
        stark_rate = _classical_electron_stark_rate_per_electron(
            ion.charge,
            ion.ionization_energy_ev,
            upper_level.energy_wavenumber,
        )
        damping = np.asarray([
            _impact_profile_damping_rate(
                transition,
                transition.einstein_a,
                stark_rate,
                float(local_electron_density),
                float(local_temperature),
                element=ion.element,
                charge=ion.charge,
                include_semiclassical_ovi_stark_widths=(
                    include_semiclassical_ovi_stark_widths
                ),
            )
            for local_electron_density, local_temperature in zip(
                electron_density, temperature
            )
        ])
        if (
            include_ovi_high_series_ion_dephasing
            and _is_ovi_high_series_formula4(
                transition, ion.element, ion.charge
            )
        ):
            damping += 2.0 * rate_ion_motion
        component_center.append(rate_center)
        component_gaussian.append(gaussian)
        component_lorentz.append(
            rate_center_cm**2 * damping
            / (4.0 * PI * LIGHT_SPEED) * 1.0e8
        )
        (
            static_scale,
            static_amplitude,
            static_ion_motion_hwhm_beta,
        ) = static_rate_profile_parameters(transition, ion.charge)
        component_integrated_strength.append(
            integrated_cross_section
            * transition.absorption_oscillator_strength
        )
        component_static_frequency_scale.append(static_scale)
        component_static_amplitude.append(static_amplitude)
        component_static_ion_motion_hwhm_beta.append(
            static_ion_motion_hwhm_beta
        )
        component_static_pattern.append(
            _hydrogenic_stark_pattern_key(
                transition, hydrogenic_linear_stark_components
            ) if np.any(static_amplitude > 0.0) else None
        )
        component_weight.append(
            lower_level.statistical_weight
            * transition.absorption_oscillator_strength
        )
        reservoir_component_range.append(
            (first_component, len(component_center))
        )

    all_bound_count = len(bound_bound) + len(reservoir_bound_bound)
    all_bound_photon_occupation = np.zeros(
        (all_bound_count, atmosphere.n_depth), dtype=np.float64
    )
    all_bound_lambda_diagonal = np.zeros_like(all_bound_photon_occupation)
    all_bound_inverse_occupation = np.zeros_like(all_bound_photon_occupation)
    if component_center:
        wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
        vacuum_radiation = (
            2.0 * PLANCK * LIGHT_SPEED**2 / wavelength_cm**5 * 1.0e-8
        )
        # The Einstein-rate equations below are written in photon
        # occupation number.  Average that quantity over the line profile
        # directly.  Averaging J_lambda first and dividing by its line-centre
        # vacuum prefactor instead spuriously changes the occupation across a
        # finite-width profile and prevents exact detailed balance for broad
        # or sparsely sampled transitions.
        photon_occupation_field = np.maximum(
            intensity / vacuum_radiation, 0.0
        )
        component_center_array = np.asarray(component_center, dtype=np.float64)
        component_occupation, component_mean_lambda = (
            _profile_weighted_line_means(
                wavelength,
                photon_occupation_field,
                lambda_diagonal,
                component_center_array,
                np.stack(component_gaussian),
                np.stack(component_lorentz),
                integrated_strength=np.asarray(
                    component_integrated_strength, dtype=np.float64
                ),
                static_frequency_scale=np.stack(
                    component_static_frequency_scale
                ),
                static_amplitude=np.stack(component_static_amplitude),
                static_ion_motion_hwhm_beta=np.stack(
                    component_static_ion_motion_hwhm_beta
                ),
                static_pattern=component_static_pattern,
            )
        )
        # Integrate spontaneous+stimulated downward rates with the same
        # profile and local thermodynamic factor as the upward rate. In a
        # Planck field exp(-h nu/kT)*(1+n_nu) == n_nu pointwise. Replacing
        # this integral by 1+<n_nu> at the nominal line centre breaks detailed
        # balance for broad/sparse profiles and rounded tabulated wavelengths.
        inverse_occupation_field = np.exp(-PLANCK*LIGHT_SPEED /
            (wavelength_cm*BOLTZMANN*temperature[None,:])) * (1+photon_occupation_field)
        component_inverse, _ = (
            _profile_weighted_line_means(
                wavelength,
                inverse_occupation_field,
                None,
                component_center_array,
                np.stack(component_gaussian),
                np.stack(component_lorentz),
                integrated_strength=np.asarray(
                    component_integrated_strength, dtype=np.float64
                ),
                static_frequency_scale=np.stack(
                    component_static_frequency_scale
                ),
                static_amplitude=np.stack(component_static_amplitude),
                static_ion_motion_hwhm_beta=np.stack(
                    component_static_ion_motion_hwhm_beta
                ),
                static_pattern=component_static_pattern,
            )
        )
        component_weight_array = np.asarray(component_weight, dtype=np.float64)
        all_component_range = component_range + reservoir_component_range
        for bound_index, (first_component, last_component) in enumerate(
            all_component_range
        ):
            weight = component_weight_array[first_component:last_component].copy()
            if np.sum(weight) <= 0.0:
                weight[:] = 1.0
            weight /= np.sum(weight)
            all_bound_photon_occupation[bound_index] = np.sum(
                weight[:, np.newaxis]
                * component_occupation[first_component:last_component],
                axis=0,
            )
            all_bound_inverse_occupation[bound_index] = np.sum(
                weight[:, np.newaxis] * component_inverse[first_component:last_component],axis=0)
            all_bound_lambda_diagonal[bound_index] = np.sum(
                weight[:, np.newaxis]
                * component_mean_lambda[first_component:last_component],
                axis=0,
            )
    bound_photon_occupation = all_bound_photon_occupation[:len(bound_bound)]
    bound_lambda_diagonal = all_bound_lambda_diagonal[:len(bound_bound)]
    bound_inverse_occupation = all_bound_inverse_occupation[:len(bound_bound)]
    reservoir_inverse_occupation = all_bound_inverse_occupation[len(bound_bound):]
    reservoir_photon_occupation = all_bound_photon_occupation[len(bound_bound):]

    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    photon_exponent = (
        PLANCK * LIGHT_SPEED
        / (wavelength_cm * BOLTZMANN * temperature[np.newaxis, :])
    )
    vacuum_radiation = (
        2.0 * PLANCK * LIGHT_SPEED**2 / wavelength_cm**5 * 1.0e-8
    )
    # Milne inverse of photoionization.  The downward radiative kernel is
    # [2 h nu^3/c^2 + J_nu] exp(-h nu/kT), expressed here per wavelength.
    # It includes spontaneous and stimulated recombination in the *actual*
    # radiation field.  At J=B this reduces algebraically to B and therefore
    # retains exact LTE recovery.  The former Planck-only inverse could be
    # badly wrong for the 5--10-eV Rydberg thresholds that feed optical O V.
    recombination_radiation = (
        vacuum_radiation + intensity
    ) * np.exp(-photon_exponent)
    photon_energy_ev = PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / EV_TO_ERG
    photo_actual = {}
    photo_recombination = {}
    collisional_ionization = {}
    continuum_target = {}
    cross_section_source = {}
    for charge in charges[:-1]:
        ion = stages[charge]
        next_ground = selected_levels[charge + 1][0]
        ground = selected_levels[charge][0]
        ground_fit = photoionization_database.fits.get((symbol, charge))
        for level in selected_levels[charge]:
            index = state_index[(symbol, charge, level.index)]
            default_target_key = (symbol, charge + 1, next_ground.index)
            target_key = (
                default_target_key
                if continuum_parent_mapping is None
                else continuum_parent_mapping.get(
                    (symbol, charge, level.index), default_target_key
                )
            )
            if target_key not in state_index:
                raise ValueError(
                    f"continuum parent {target_key!r} is absent from the reduced atom"
                )
            target = state_index[target_key]
            threshold_ev = (
                ion.ionization_energy_ev
                - level.energy_wavenumber * PLANCK * LIGHT_SPEED / EV_TO_ERG
            )
            if threshold_ev <= 0.05:
                continue
            threshold_frequency = threshold_ev * EV_TO_ERG / PLANCK
            threshold_table = (
                None
                if photoionization_threshold_data is None
                else photoionization_threshold_data.get(charge)
            )
            # Prefer the term-matched Opacity Project data for *all* levels,
            # including the ground term.  TMAP's public RBF atoms use these
            # cross-sections.  The old ground-level special case substituted
            # a Verner fit and sometimes sampled it just below its rounded
            # threshold, producing a spurious zero Seaton cross-section.
            cross_section = (
                None
                if threshold_table is None
                else threshold_table.cross_section(
                    photon_energy_ev * EV_TO_ERG / PLANCK,
                    threshold_frequency,
                    relative_tolerance=0.02,
                    level_label=level.label,
                )
            )
            if cross_section is not None:
                matched_threshold = threshold_table.cross_section_for_threshold(
                    threshold_frequency,
                    relative_tolerance=0.02,
                    level_label=level.label,
                )
                threshold_cross_section = (
                    float(matched_threshold)
                    if matched_threshold is not None
                    else float(np.max(cross_section))
                )
                source = threshold_table.source
                threshold_cross_section_includes_gbar = (
                    threshold_table.threshold_cross_section_includes_gbar
                )
            elif level.index == ground.index:
                if ground_fit is None:
                    raise ValueError(
                        f"missing ground-state photoionization data for "
                        f"{symbol} {charge:+d}: no matched threshold table "
                        "and no Verner fit"
                    )
                cross_section = ground_fit.cross_section(photon_energy_ev)
                threshold_cross_section = float(
                    ground_fit.cross_section(np.asarray((
                        max(threshold_ev, ground_fit.threshold_energy_ev)
                        * (1.0 + 1.0e-10),
                    )))[0]
                )
                source = "Verner ground-state fit"
                threshold_cross_section_includes_gbar = False
            else:
                effective_charge = charge + 1.0
                effective_n = np.sqrt(
                    13.605_693_122_994 * effective_charge**2 / threshold_ev
                )
                sigma0 = 6.30e-18 * effective_n / effective_charge**2
                threshold_cross_section = sigma0
                source = "hydrogenic Kramers excited-level approximation"
                cross_section = np.where(
                    photon_energy_ev >= threshold_ev,
                    sigma0 * (threshold_ev / photon_energy_ev) ** 3,
                    0.0,
                )
                threshold_cross_section_includes_gbar = False
            photo_actual[index] = _photoionization_rate(
                wavelength, intensity, cross_section
            )
            photo_recombination[index] = _photoionization_rate(
                wavelength, recombination_radiation, cross_section
            )
            # The inverse three-body rate is constructed below by detailed
            # balance, just like the radiative recombination rate.
            threshold_u = (
                threshold_ev * EV_TO_ERG / (BOLTZMANN * temperature)
            )
            collisional_ionization[index] = (
                _tmap_seaton_collisional_ionization_rate(
                    electron_density,
                    temperature,
                    threshold_u,
                    threshold_cross_section,
                    charge,
                    cross_section_includes_gbar=(
                        threshold_cross_section_includes_gbar
                    ),
                )
            )
            continuum_target[index] = target
            cross_section_source[index] = source

    def psm20_l_mixing_coefficients(
        record: PSM20AngularMomentumMixingCollision,
        lower_level: AtomicLevel,
        upper_level: AtomicLevel,
        depth: int,
    ) -> tuple[float, float]:
        """Return equivalent per-electron coefficients for heavy-ion mixing."""

        local_temperature = float(temperature[depth])
        local_electron_density = float(electron_density[depth])
        direct_rate = 0.0

        def add_collider(
            density: float, collider_charge: int, atomic_mass_u: float
        ) -> None:
            nonlocal direct_rate
            if density <= 0.0 or collider_charge <= 0:
                return
            direct_rate += density * psm20_l_mixing_rate_coefficient(
                record.principal_quantum_number,
                record.higher_l,
                record.higher_l - 1,
                local_temperature,
                local_electron_density,
                record.target_atomic_mass_u,
                float(collider_charge),
                atomic_mass_u,
                record.target_core_charge,
                energy_splitting_wavenumber=(
                    record.energy_splitting_wavenumber
                ),
                radiative_lifetime_s=record.radiative_lifetime_s,
            )

        host_population = lte_state.host_ion_number_density
        if host_population is not None:
            host_mass = (
                _HYDROGEN_ATOMIC_MASS_U
                if lte_state.reference_species == "H"
                else _HELIUM_ATOMIC_MASS_U
            )
            for collider_charge in range(1, host_population.shape[0]):
                add_collider(
                    float(host_population[collider_charge, depth]),
                    collider_charge,
                    host_mass,
                )
        for collider_element, collider_population in (
            lte_state.ion_number_density.items()
        ):
            collider_mass = ATOMIC_MASS_U.get(collider_element)
            if collider_mass is None:
                continue
            for collider_charge in range(1, collider_population.shape[0]):
                add_collider(
                    float(collider_population[collider_charge, depth]),
                    collider_charge,
                    float(collider_mass),
                )

        delta_energy_over_kt = (
            (upper_level.energy_wavenumber - lower_level.energy_wavenumber)
            * PLANCK
            * LIGHT_SPEED
            / (BOLTZMANN * local_temperature)
        )
        if record.lower_energy_l == record.higher_l:
            upward_rate = direct_rate
            downward_rate = (
                direct_rate
                * lower_level.statistical_weight
                / upper_level.statistical_weight
                * np.exp(min(delta_energy_over_kt, 700.0))
            )
        elif record.upper_energy_l == record.higher_l:
            downward_rate = direct_rate
            upward_rate = (
                direct_rate
                * upper_level.statistical_weight
                / lower_level.statistical_weight
                * np.exp(-delta_energy_over_kt)
            )
        else:  # pragma: no cover - protected by the collision-data builder.
            raise ValueError("PSM20 link does not contain its stated higher-l term")
        return (
            upward_rate / local_electron_density,
            downward_rate / local_electron_density,
        )

    def btm_quadrupole_l_mixing_coefficients(
        record: BTMQuadrupoleAngularMomentumMixingCollision,
        lower_level: AtomicLevel,
        upper_level: AtomicLevel,
        depth: int,
    ) -> tuple[float, float]:
        """Return equivalent per-electron BTM quadrupole coefficients."""

        local_temperature = float(temperature[depth])
        local_electron_density = float(electron_density[depth])
        direct_upward_rate = 0.0

        def add_collider(
            density: float, collider_charge: int, atomic_mass_u: float
        ) -> None:
            nonlocal direct_upward_rate
            if density <= 0.0 or collider_charge <= 0:
                return
            direct_upward_rate += (
                density
                * btm_quadrupole_l_mixing_rate_coefficient(
                    record.principal_quantum_number,
                    record.lower_energy_l,
                    record.upper_energy_l,
                    local_temperature,
                    record.target_atomic_mass_u,
                    float(collider_charge),
                    atomic_mass_u,
                    record.target_core_charge,
                    energy_splitting_wavenumber=(
                        record.energy_splitting_wavenumber
                    ),
                )
            )

        host_population = lte_state.host_ion_number_density
        if host_population is not None:
            host_mass = (
                _HYDROGEN_ATOMIC_MASS_U
                if lte_state.reference_species == "H"
                else _HELIUM_ATOMIC_MASS_U
            )
            for collider_charge in range(1, host_population.shape[0]):
                add_collider(
                    float(host_population[collider_charge, depth]),
                    collider_charge,
                    host_mass,
                )
        for collider_element, collider_population in (
            lte_state.ion_number_density.items()
        ):
            collider_mass = ATOMIC_MASS_U.get(collider_element)
            if collider_mass is None:
                continue
            for collider_charge in range(1, collider_population.shape[0]):
                add_collider(
                    float(collider_population[collider_charge, depth]),
                    collider_charge,
                    float(collider_mass),
                )

        delta_energy_over_kt = (
            (upper_level.energy_wavenumber - lower_level.energy_wavenumber)
            * PLANCK
            * LIGHT_SPEED
            / (BOLTZMANN * local_temperature)
        )
        downward_rate = (
            direct_upward_rate
            * lower_level.statistical_weight
            / upper_level.statistical_weight
            * np.exp(min(delta_energy_over_kt, 700.0))
        )
        return (
            direct_upward_rate / local_electron_density,
            downward_rate / local_electron_density,
        )

    def collision_coefficients(
        transition: AtomicTransition,
        lower_level: AtomicLevel,
        upper_level: AtomicLevel,
        charge: int,
        depth: int,
        collision_record: (
            tuple[int, tuple[float, ...]]
            | ChiantiTermCollisionStrength
            | ConstantEffectiveCollisionStrength
            | TabulatedElectronExcitationRateCoefficient
            | PSM20AngularMomentumMixingCollision
            | BTMQuadrupoleAngularMomentumMixingCollision
            | None
        ),
    ) -> tuple[float, float]:
        """Return upward/downward electron-impact coefficients in cm3/s."""

        delta_ev = (
            (upper_level.energy_wavenumber - lower_level.energy_wavenumber)
            * PLANCK * LIGHT_SPEED / EV_TO_ERG
        )
        u = delta_ev * EV_TO_ERG / (BOLTZMANN * temperature[depth])
        if isinstance(
            collision_record, PSM20AngularMomentumMixingCollision
        ):
            return psm20_l_mixing_coefficients(
                collision_record, lower_level, upper_level, depth
            )
        if isinstance(
            collision_record, BTMQuadrupoleAngularMomentumMixingCollision
        ):
            return btm_quadrupole_l_mixing_coefficients(
                collision_record, lower_level, upper_level, depth
            )
        if isinstance(
            collision_record, TabulatedElectronExcitationRateCoefficient
        ):
            q_up = collision_record.upward_rate_coefficient(
                float(temperature[depth])
            )
            q_down = (
                q_up
                * lower_level.statistical_weight
                / upper_level.statistical_weight
                * np.exp(min(u, 700.0))
            )
            return q_up, q_down
        if isinstance(
            collision_record,
            (ChiantiTermCollisionStrength, ConstantEffectiveCollisionStrength),
        ):
            collision_strength = collision_record.effective_collision_strength(
                float(temperature[depth])
            )
            common = 8.629e-6 * collision_strength / np.sqrt(temperature[depth])
            return (
                common / lower_level.statistical_weight * np.exp(-u),
                common / upper_level.statistical_weight,
            )
        if transition.transition_type == "E1":
            if collision_record is not None and collision_record[0] == 26:
                parameters = collision_record[1]
                if len(parameters) < 3:
                    raise ValueError(
                        "TMAD CBB formula 26 requires T1, NFIT, and coefficients"
                    )
                temperature_origin = parameters[0]
                coefficient_count = int(round(parameters[1]))
                coefficients = parameters[2 : 2 + coefficient_count]
                if len(coefficients) != coefficient_count:
                    raise ValueError(
                        "TMAD CBB formula 26 has incomplete coefficients"
                    )
                x = np.log10(temperature[depth]) - temperature_origin
                collision_strength = max(
                    float(sum(
                        value * x**order
                        for order, value in enumerate(coefficients)
                    )),
                    0.0,
                )
                common = (
                    8.629e-6 * collision_strength / np.sqrt(temperature[depth])
                )
                return (
                    common / lower_level.statistical_weight * np.exp(-u),
                    common / upper_level.statistical_weight,
                )

            collision_table = (
                None
                if photoionization_threshold_data is None
                else photoionization_threshold_data.get(charge)
            )
            exact_gbar = (
                None
                if collision_record is None or collision_record[0] != 1
                else collision_record[1][1]
            )
            if exact_gbar is None and collision_table is not None:
                exact_gbar = collision_table.collision_gbar_for_wavelength(
                    transition.wavelength_vacuum_angstrom
                )
            gbar = _tlusty_van_regemorter_gbar(
                u, 0.25 if exact_gbar is None else exact_gbar
            )
            q_up = (
                19.7363
                * temperature[depth] ** -1.5
                * np.exp(-u)
                / max(u, 1.0e-12)
                * gbar
                * transition.absorption_oscillator_strength
            )
            return (
                q_up,
                q_up
                * lower_level.statistical_weight
                / upper_level.statistical_weight
                * np.exp(min(u, 700.0)),
            )

        # TLUSTY's model atoms use the Allen constant-effective-collision-
        # strength prescription (ICOL=4, Omega=0.05) for low forbidden links.
        common = 8.629e-6 * 0.05 / np.sqrt(temperature[depth])
        return (
            common / lower_level.statistical_weight * np.exp(-u),
            common / upper_level.statistical_weight,
        )

    def tmad_collision_only_coefficients(
        record: tuple[int, tuple[float, ...]],
        lower_level: AtomicLevel,
        upper_level: AtomicLevel,
        depth: int,
    ) -> tuple[float, float]:
        """Evaluate a TMAD CBB record without a radiative RBB partner.

        ATOMS2-processed atoms can retain near-degenerate allowed collision
        links for which no line is present in the formal RBB block.  Formula
        1 stores ``f_lu`` and the Van Regemorter minimum Gaunt factor, so it
        contains everything needed for the electron-impact rate even without
        an :class:`AtomicTransition`.  Dropping these records leaves holes in
        the statistical-equilibrium matrix relative to the TMAP atom.
        """

        formula, parameters = record
        if formula != 1 or len(parameters) < 2:
            raise ValueError(
                "collision-only TMAD CBB records currently require formula "
                "1 with oscillator strength and minimum Gaunt factor"
            )
        delta_energy = (
            (upper_level.energy_wavenumber - lower_level.energy_wavenumber)
            * PLANCK
            * LIGHT_SPEED
        )
        u = delta_energy / (BOLTZMANN * temperature[depth])
        oscillator_strength, minimum_gbar = parameters[:2]
        gbar = _tlusty_van_regemorter_gbar(u, float(minimum_gbar))
        q_up = (
            19.7363
            * temperature[depth] ** -1.5
            * np.exp(-u)
            / max(u, 1.0e-12)
            * gbar
            * float(oscillator_strength)
        )
        return (
            q_up,
            q_up
            * lower_level.statistical_weight
            / upper_level.statistical_weight
            * np.exp(min(u, 700.0)),
        )

    populations = np.empty_like(lte_level)
    for depth in range(atmosphere.n_depth):
        matrix = np.zeros((n_state, n_state), dtype=np.float64)

        def add_pair(lower: int, upper: int, upward: float, downward: float) -> None:
            matrix[lower, lower] -= upward
            matrix[upper, lower] += upward
            matrix[upper, upper] -= downward
            matrix[lower, upper] += downward

        for bound_index, (
            lower, upper, transition, fine_components
        ) in enumerate(bound_bound):
            center = transition.wavelength_vacuum_angstrom
            upper_level = state_level[upper]
            photon_occupation = float(
                bound_photon_occupation[bound_index, depth]
            )
            local_lambda = float(bound_lambda_diagonal[bound_index, depth])
            lower_level = state_level[lower]
            energy_gap = (upper_level.energy_wavenumber-lower_level.energy_wavenumber)*PLANCK*LIGHT_SPEED
            inverse_occupation = np.exp(min(energy_gap/(BOLTZMANN*temperature[depth]),700.)) * bound_inverse_occupation[bound_index,depth]
            effective_photon_occupation = photon_occupation
            effective_inverse_occupation = inverse_occupation
            if previous_population is not None and local_lambda > 0.0:
                old_lower = previous_population[lower, depth]
                old_upper = previous_population[upper, depth]
                denominator = (
                    upper_level.statistical_weight
                    / lower_level.statistical_weight
                    * old_lower
                    - old_upper
                )
                if denominator > 0.0:
                    effective_photon_occupation,effective_inverse_occupation = _mali_radiative_occupations(
                        photon_occupation,inverse_occupation,local_lambda,old_upper/denominator)
            radiative_up = (
                upper_level.statistical_weight / lower_level.statistical_weight
                * transition.einstein_a * effective_photon_occupation
            )
            radiative_down = transition.einstein_a * effective_inverse_occupation
            collision_key = (
                symbol,
                int(state_charge[lower]),
                lower_level.index,
                upper_level.index,
            )
            collision_record = (
                None if collision_data is None else collision_data.get(collision_key)
            )
            q_up, q_down = collision_coefficients(
                transition,
                lower_level,
                upper_level,
                int(state_charge[lower]),
                depth,
                collision_record,
            )
            add_pair(
                lower,
                upper,
                radiative_up + collision_scale * electron_density[depth] * q_up,
                radiative_down + collision_scale * electron_density[depth] * q_down,
            )

        for lower, upper, record in collision_only:
            lower_level = state_level[lower]
            upper_level = state_level[upper]
            if isinstance(record, tuple):
                q_up, q_down = tmad_collision_only_coefficients(
                    record, lower_level, upper_level, depth
                )
            elif isinstance(record, PSM20AngularMomentumMixingCollision):
                q_up, q_down = psm20_l_mixing_coefficients(
                    record, lower_level, upper_level, depth
                )
            elif isinstance(
                record, BTMQuadrupoleAngularMomentumMixingCollision
            ):
                q_up, q_down = btm_quadrupole_l_mixing_coefficients(
                    record, lower_level, upper_level, depth
                )
            elif isinstance(
                record, TabulatedElectronExcitationRateCoefficient
            ):
                delta_ev = (
                    (
                        upper_level.energy_wavenumber
                        - lower_level.energy_wavenumber
                    )
                    * PLANCK
                    * LIGHT_SPEED
                    / EV_TO_ERG
                )
                u = delta_ev * EV_TO_ERG / (
                    BOLTZMANN * temperature[depth]
                )
                q_up = record.upward_rate_coefficient(
                    float(temperature[depth])
                )
                q_down = (
                    q_up
                    * lower_level.statistical_weight
                    / upper_level.statistical_weight
                    * np.exp(min(u, 700.0))
                )
            else:
                delta_ev = (
                    (
                        upper_level.energy_wavenumber
                        - lower_level.energy_wavenumber
                    )
                    * PLANCK
                    * LIGHT_SPEED
                    / EV_TO_ERG
                )
                u = delta_ev * EV_TO_ERG / (
                    BOLTZMANN * temperature[depth]
                )
                collision_strength = record.effective_collision_strength(
                    float(temperature[depth])
                )
                common = (
                    8.629e-6
                    * collision_strength
                    / np.sqrt(temperature[depth])
                )
                q_up = (
                    common / lower_level.statistical_weight * np.exp(-u)
                )
                q_down = common / upper_level.statistical_weight
            add_pair(
                lower,
                upper,
                collision_scale * electron_density[depth] * q_up,
                collision_scale * electron_density[depth] * q_down,
            )

        for coupling_index, (
            explicit,
            parent,
            coupling,
            reservoir_level,
            lte_to_parent_ratio,
        ) in enumerate(reservoir_bound_bound):
            explicit_level = state_level[explicit]
            lower_level = (
                explicit_level if coupling.explicit_is_lower else reservoir_level
            )
            upper_level = (
                reservoir_level if coupling.explicit_is_lower else explicit_level
            )
            transition = coupling.transition
            photon_occupation = float(
                reservoir_photon_occupation[coupling_index, depth]
            )
            radiative_up = (
                upper_level.statistical_weight
                / lower_level.statistical_weight
                * transition.einstein_a
                * photon_occupation
            )
            energy_gap = (upper_level.energy_wavenumber-lower_level.energy_wavenumber)*PLANCK*LIGHT_SPEED
            radiative_down = transition.einstein_a * np.exp(min(
                energy_gap/(BOLTZMANN*temperature[depth]),700.)) * reservoir_inverse_occupation[coupling_index,depth]
            # ``CBX`` is the explicit TMAP channel for collisions from an
            # NLTE level to an LTE reservoir level.  An RBB record by itself
            # supplies the radiative rate only; applying the ordinary
            # Van-Regemorter fallback here invents a collisional coupling
            # that is absent from the atom.
            if coupling.collision_record is None:
                q_up = q_down = 0.0
            else:
                q_up, q_down = collision_coefficients(
                    transition,
                    lower_level,
                    upper_level,
                    coupling.lte_charge,
                    depth,
                    coupling.collision_record,
                )
            upward = (
                radiative_up
                + collision_scale * electron_density[depth] * q_up
            )
            downward = (
                radiative_down
                + collision_scale * electron_density[depth] * q_down
            )
            ratio = float(lte_to_parent_ratio[depth])
            if coupling.explicit_is_lower:
                add_pair(explicit, parent, upward, ratio * downward)
            else:
                add_pair(parent, explicit, ratio * upward, downward)

        for (
            lower,
            parent,
            frequency,
            rate_scale,
            photon_occupation,
        ) in effective_dielectronic:
            occupation = float(photon_occupation[depth])
            upward = rate_scale * occupation
            downward = (
                rate_scale
                * np.exp(
                    -PLANCK * frequency
                    / (BOLTZMANN * temperature[depth])
                )
                * (1.0 + occupation)
                * lte_level[lower, depth]
                / max(lte_level[parent, depth], tiny)
            )
            add_pair(lower, parent, max(upward, tiny), max(downward, tiny))

        for lower, target in continuum_target.items():
            radiative_upward = float(photo_actual[lower][depth])
            radiative_downward = (
                float(photo_recombination[lower][depth])
                * lte_level[lower, depth]
                / max(lte_level[target, depth], tiny)
            )
            collision_upward = float(collisional_ionization[lower][depth])
            collision_downward = (
                collision_upward
                * lte_level[lower, depth]
                / max(lte_level[target, depth], tiny)
            )
            upward = max(radiative_upward + collision_upward, tiny)
            downward = max(radiative_downward + collision_downward, tiny)
            add_pair(lower, target, upward, downward)

        solution = _positive_rate_equilibrium(matrix,conservation_weight[:,depth],
            represented_system_population[depth])
        populations[:, depth] = solution

    population_departure = {
        key: np.asarray(populations[index] / np.maximum(lte_level[index], tiny))
        for index, key in enumerate(state_keys)
    }
    if formal_level_mapping is None:
        departure = population_departure
    else:
        departure = {}
        for key, value in population_departure.items():
            for formal_key in formal_level_mapping.get(key, ()):
                departure[formal_key] = value
        if formal_lte_parent_mapping is not None:
            for formal_key, parent_key in formal_lte_parent_mapping.items():
                if parent_key in population_departure:
                    departure[formal_key] = population_departure[parent_key]

    # A separate Planck solve is deliberately omitted: every inverse
    # continuum rate was constructed from the LTE reference, and the Einstein
    # plus collision relations recover Boltzmann excitation algebraically.
    represented_lte_fraction = represented_population / np.maximum(
        lte_state.element_number_density[symbol], tiny
    )
    return ReducedLightMetalLevelState(
        element=symbol,
        level_key=tuple(state_keys),
        population_density=np.asarray(populations),
        lte_population_density=np.asarray(lte_level),
        conservation_weight=np.asarray(conservation_weight),
        level_departure_coefficient=MappingProxyType(departure),
        maximum_lte_recovery_error=0.0,
        metadata={
            "model_atom": (
                f"reduced explicit {symbol} atom with "
                + ", ".join(
                    f"{charge:+d}:{levels_per_charge[charge]} levels"
                    for charge in charges
                )
            ),
            "bound_bound_transitions": len(bound_bound),
            "lte_bound_bound_couplings": len(reservoir_bound_bound),
            "effective_dielectronic_couplings": len(effective_dielectronic),
            "formal_level_mapping": (
                "native population levels"
                if formal_level_mapping is None
                else f"{len(departure)} mapped fine-structure levels"
            ),
            "formal_lte_parent_levels": (
                0
                if formal_lte_parent_mapping is None
                else len(formal_lte_parent_mapping)
            ),
            "photoionized_levels": len(photo_actual),
            "ground_photoionization": sorted({
                source
                for index, source in cross_section_source.items()
                if state_level[index].index
                == selected_levels[int(state_charge[index])][0].index
            }),
            "excited_photoionization": (
                sorted({
                    source
                    for index, source in cross_section_source.items()
                    if state_level[index].index
                    != selected_levels[int(state_charge[index])][0].index
                })
            ),
            "electron_excitation": (
                "tabulated R-matrix rate coefficients or CHIANTI effective "
                "collision strengths where supplied, then "
                "TMAD CBB or TLUSTY Van Regemorter allowed rates plus Allen "
                "Omega=0.05 for forbidden M1/E2/M2 couplings"
            ),
            "electron_excitation_collision_scale": collision_scale,
            "tabulated_electron_excitation_transitions": sum(
                isinstance(
                    record, TabulatedElectronExcitationRateCoefficient
                )
                for record in (collision_data or {}).values()
            ),
            "tabulated_electron_excitation_sources": sorted({
                record.source
                for record in (collision_data or {}).values()
                if isinstance(
                    record, TabulatedElectronExcitationRateCoefficient
                )
            }),
            "collision_only_transitions": len(collision_only),
            "heavy_ion_angular_momentum_mixing_transitions": sum(
                isinstance(
                    record,
                    (
                        PSM20AngularMomentumMixingCollision,
                        BTMQuadrupoleAngularMomentumMixingCollision,
                    ),
                )
                for record in (collision_data or {}).values()
            ),
            "heavy_ion_dipole_angular_momentum_mixing_transitions": sum(
                isinstance(record, PSM20AngularMomentumMixingCollision)
                for record in (collision_data or {}).values()
            ),
            "heavy_ion_quadrupole_angular_momentum_mixing_transitions": sum(
                isinstance(
                    record, BTMQuadrupoleAngularMomentumMixingCollision
                )
                for record in (collision_data or {}).values()
            ),
            "heavy_ion_angular_momentum_mixing": (
                "Badnell et al. (2021) final-state-resolved PSM20 rates; "
                "local Debye plus term-splitting/lifetime cut-offs and "
                "Deliporanidou et al. (2025) BTM quadrupole rates where "
                "supplied; explicit H/He/metal ionic colliders"
                if any(
                    isinstance(
                        record,
                        (
                            PSM20AngularMomentumMixingCollision,
                            BTMQuadrupoleAngularMomentumMixingCollision,
                        ),
                    )
                    for record in (collision_data or {}).values()
                )
                else "not included"
            ),
            "collisional_ionization": (
                "TLUSTY ICOL=0 Seaton rate with inverse three-body detailed balance"
            ),
            "represented_lte_fraction_minimum": float(np.min(represented_lte_fraction)),
            "represented_lte_fraction_maximum": float(np.max(represented_lte_fraction)),
            "lte_reservoir_levels": (
                0
                if lte_level_reservoir is None
                else sum(len(levels) for levels in lte_level_reservoir.values())
            ),
            "lte_reservoir_rate_closure": (
                "explicit-to-LTE radiative and collisional transitions "
                "collapsed onto named continuum parents"
                if reservoir_bound_bound
                else "no one-sided LTE bound-bound transitions supplied"
            ),
            "lte_reservoir_fraction_minimum": float(np.min(
                np.sum(reservoir_lte_by_parent, axis=0)
                / np.maximum(represented_system_population, tiny)
            )),
            "lte_reservoir_fraction_maximum": float(np.max(
                np.sum(reservoir_lte_by_parent, axis=0)
                / np.maximum(represented_system_population, tiny)
            )),
            "line_source_function": (
                "explicit lower/upper departures for closed transitions; "
                "ion-stage departure otherwise"
            ),
            "statistical_equilibrium_iteration": (
                "diagonal MALI with line-opacity-weighted lambda operator"
                if lambda_diagonal is not None
                and previous_population is not None
                else "ordinary Lambda iteration"
            ),
        },
        population_level_departure_coefficient=MappingProxyType(
            population_departure
        ),
        formal_level_mapping=(
            None
            if formal_level_mapping is None
            else MappingProxyType(dict(formal_level_mapping))
        ),
        formal_lte_parent_mapping=(
            None
            if formal_lte_parent_mapping is None
            else MappingProxyType(dict(formal_lte_parent_mapping))
        ),
    )
