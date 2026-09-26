"""Hydrogenic linear-Stark component patterns for quasi-static line wings.

TMAP's formula-4 wing spreads a whole ``n_l -> n_u`` line with one Holtsmark
distribution whose frequency scale is that of the *outermost* Stark
component, ``X_max = n_u (n_u - 1) + n_l (n_l - 1)`` in units of
``3 e a0 F / (2 Z)``.  A hydrogenic line instead splits into components with
shifts ``X = n_u (n1 - n2)_u - n_l (n1 - n2)_l`` and strengths fixed by the
parabolic-state dipole matrix elements.  The quasi-static profile is the
strength-weighted sum of Holtsmark distributions, one per component.

Strengths are obtained by expanding parabolic states in spherical states
(Clebsch-Gordan coefficients with ``j = (n - 1)/2``) and summing the squared
dipole matrix elements over polarizations (an isotropic microfield).  The
hydrogen radial integrals are integrated numerically; they scale as ``1/Z``
for every component, so the relative pattern is independent of the ion.
The implementation reproduces the Lyman-alpha pattern (2/3 unshifted,
1/6 at each of ``X = +-2``) and the Balmer-alpha pattern tabulated by
Bethe & Salpeter (1957).
"""

from __future__ import annotations

from functools import lru_cache
from math import lgamma, exp, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre, gammaln

FloatArray = NDArray[np.float64]

# Composite profile tabulation: C(beta) on a logarithmic grid spanning the
# formula-4 support beta <= 30 (beta relative to the outermost component).
COMPOSITE_LOG_BETA_MIN = -6.0
COMPOSITE_LOG_BETA_MAX = np.log10(30.0)
COMPOSITE_POINTS = 2048


def _wigner_3j_doubled(
    two_j1: int, two_j2: int, two_j3: int, two_m1: int, two_m2: int, two_m3: int
) -> float:
    """Wigner 3j symbol with all arguments doubled (Racah formula)."""

    if two_m1 + two_m2 + two_m3 != 0:
        return 0.0
    if two_j3 < abs(two_j1 - two_j2) or two_j3 > two_j1 + two_j2:
        return 0.0
    if abs(two_m1) > two_j1 or abs(two_m2) > two_j2 or abs(two_m3) > two_j3:
        return 0.0
    if (two_j1 + two_j2 + two_j3) % 2 or (two_j1 - two_m1) % 2:
        return 0.0
    if (two_j2 - two_m2) % 2 or (two_j3 - two_m3) % 2:
        return 0.0
    a = (two_j1 + two_j2 - two_j3) // 2
    b = (two_j1 - two_j2 + two_j3) // 2
    c = (-two_j1 + two_j2 + two_j3) // 2
    d = (two_j1 + two_j2 + two_j3) // 2 + 1
    log_triangle = lgamma(a + 1) + lgamma(b + 1) + lgamma(c + 1) - lgamma(d + 1)
    log_m = (
        lgamma((two_j1 + two_m1) // 2 + 1) + lgamma((two_j1 - two_m1) // 2 + 1)
        + lgamma((two_j2 + two_m2) // 2 + 1) + lgamma((two_j2 - two_m2) // 2 + 1)
        + lgamma((two_j3 + two_m3) // 2 + 1) + lgamma((two_j3 - two_m3) // 2 + 1)
    )
    k_min = max(0, (two_j2 - two_j3 - two_m1) // 2, (two_j1 - two_j3 + two_m2) // 2)
    k_max = min(a, (two_j1 - two_m1) // 2, (two_j2 + two_m2) // 2)
    total = 0.0
    for k in range(k_min, k_max + 1):
        log_term = (
            lgamma(k + 1)
            + lgamma(a - k + 1)
            + lgamma((two_j1 - two_m1) // 2 - k + 1)
            + lgamma((two_j2 + two_m2) // 2 - k + 1)
            + lgamma((two_j3 - two_j2 + two_m1) // 2 + k + 1)
            + lgamma((two_j3 - two_j1 - two_m2) // 2 + k + 1)
        )
        total += (-1.0) ** k * exp(0.5 * (log_triangle + log_m) - log_term)
    phase = (two_j1 - two_j2 - two_m3) // 2
    return (-1.0) ** phase * total


def _clebsch_gordan_doubled(
    two_j1: int, two_m1: int, two_j2: int, two_m2: int, two_j: int, two_m: int
) -> float:
    value = _wigner_3j_doubled(two_j1, two_j2, two_j, two_m1, two_m2, -two_m)
    phase = (two_j1 - two_j2 + two_m) // 2
    return (-1.0) ** phase * sqrt(two_j + 1.0) * value


def _parabolic_states(n: int) -> list[tuple[int, int, dict[int, float]]]:
    """Return ``(n1 - n2, m, {l: coefficient})`` for all parabolic states."""

    two_j = n - 1
    states = []
    for m in range(-(n - 1), n):
        for n1 in range(0, n - abs(m)):
            n2 = n - abs(m) - 1 - n1
            two_m1 = m + n1 - n2
            two_m2 = m - n1 + n2
            coefficients = {}
            for l in range(abs(m), n):
                value = _clebsch_gordan_doubled(
                    two_j, two_m1, two_j, two_m2, 2 * l, 2 * m
                )
                if value != 0.0:
                    coefficients[l] = value
            states.append((n1 - n2, m, coefficients))
    return states


def _radial_functions(n: int, radius: FloatArray) -> list[FloatArray]:
    x = 2.0 * radius / n
    functions = []
    for l in range(n):
        log_norm = 0.5 * (
            3.0 * np.log(2.0 / n) + gammaln(n - l) - np.log(2.0 * n) - gammaln(n + l + 1)
        )
        functions.append(
            np.exp(log_norm - radius / n) * x**l
            * eval_genlaguerre(n - l - 1, 2 * l + 1, x)
        )
    return functions


def _dipole_angular(l_upper: int, m_upper: int, l_lower: int, m_lower: int, q: int) -> float:
    """``<l_u m_u | C^1_q | l_l m_l>``."""

    return (
        (-1.0) ** m_upper
        * sqrt((2 * l_upper + 1) * (2 * l_lower + 1))
        * _wigner_3j_doubled(2 * l_upper, 2, 2 * l_lower, -2 * m_upper, 2 * q, 2 * m_lower)
        * _wigner_3j_doubled(2 * l_upper, 2, 2 * l_lower, 0, 0, 0)
    )


@lru_cache(maxsize=None)
def hydrogenic_stark_pattern(lower_n: int, upper_n: int) -> tuple[FloatArray, FloatArray]:
    """Return ``(|X|, strength)`` of the ``lower_n -> upper_n`` Stark pattern.

    Components with equal ``|X|`` are combined; strengths sum to one and
    include the unshifted (``X = 0``) component when it exists.
    """

    lower_n, upper_n = int(lower_n), int(upper_n)
    if lower_n < 1 or upper_n < lower_n:
        raise ValueError("require 1 <= lower_n <= upper_n")
    radius = np.linspace(0.0, 8.0 * upper_n**2 + 40.0, 40_001)
    upper_radial = _radial_functions(upper_n, radius)
    lower_radial = upper_radial if lower_n == upper_n else _radial_functions(lower_n, radius)
    radial_integral: dict[tuple[int, int], float] = {}

    def radial(l_upper: int, l_lower: int) -> float:
        key = (l_upper, l_lower)
        if key not in radial_integral:
            radial_integral[key] = float(np.trapz(
                upper_radial[l_upper] * lower_radial[l_lower] * radius**3, radius
            ))
        return radial_integral[key]

    pattern: dict[int, float] = {}
    lower_states = _parabolic_states(lower_n)
    for k_upper, m_upper, c_upper in _parabolic_states(upper_n):
        for k_lower, m_lower, c_lower in lower_states:
            q = m_upper - m_lower
            if abs(q) > 1:
                continue
            amplitude = 0.0
            for l_upper, a in c_upper.items():
                for l_lower, b in c_lower.items():
                    if abs(l_upper - l_lower) != 1:
                        continue
                    amplitude += (
                        a * b * _dipole_angular(l_upper, m_upper, l_lower, m_lower, q)
                        * radial(l_upper, l_lower)
                    )
            if amplitude == 0.0:
                continue
            shift = abs(upper_n * k_upper - lower_n * k_lower)
            pattern[shift] = pattern.get(shift, 0.0) + amplitude * amplitude
    total = sum(pattern.values())
    if total <= 0.0:
        raise ValueError(f"no dipole components for {lower_n}->{upper_n}")
    shifts = np.asarray(sorted(pattern), dtype=np.float64)
    strengths = np.asarray([pattern[int(s)] / total for s in shifts], dtype=np.float64)
    keep = strengths > 1.0e-12
    return shifts[keep], strengths[keep]


def formula4_extreme_shift(lower_n: int, upper_n: int) -> float:
    """TMAP formula-4 Stark sum, the outermost component's ``|X|``."""

    return float(upper_n * (upper_n - 1) + lower_n * (lower_n - 1))


@lru_cache(maxsize=None)
def composite_static_profile(lower_n: int, upper_n: int) -> FloatArray:
    """Tabulate ``C(beta) = sum_X w_X U(beta / r_X) / r_X`` on the log-beta grid.

    ``beta`` is measured in units of the formula-4 (outermost-component)
    frequency scale and ``r_X = |X| / X_max``.  The unshifted component has
    no quasi-static wing and is left to the impact profile.  ``U`` is the
    Holtsmark distribution used by formula 4, so replacing ``U(beta)`` by
    ``C(beta)`` in the formula-4 cross-section changes only the component
    structure.  The integral of ``C`` is the shifted-component strength.
    """

    from .light_metal_nlte import _holtsmark_microfield_distribution

    shifts, strengths = hydrogenic_stark_pattern(lower_n, upper_n)
    extreme = formula4_extreme_shift(lower_n, upper_n)
    beta = np.logspace(COMPOSITE_LOG_BETA_MIN, COMPOSITE_LOG_BETA_MAX, COMPOSITE_POINTS)
    profile = np.zeros_like(beta)
    for shift, strength in zip(shifts, strengths):
        if shift <= 0.0:
            continue
        ratio = shift / extreme
        profile += strength / ratio * _holtsmark_microfield_distribution(beta / ratio)
    return np.ascontiguousarray(profile)


def composite_static_profile_value(profile: FloatArray, beta: FloatArray) -> FloatArray:
    """Interpolate a tabulated composite profile (log-log, quadratic below the grid)."""

    value = np.asarray(beta, dtype=np.float64)
    log_beta = np.log10(np.maximum(value, 1.0e-300))
    step = (COMPOSITE_LOG_BETA_MAX - COMPOSITE_LOG_BETA_MIN) / (COMPOSITE_POINTS - 1)
    position = (log_beta - COMPOSITE_LOG_BETA_MIN) / step
    result = np.zeros_like(value)
    inside = (position >= 0.0) & (position <= COMPOSITE_POINTS - 1)
    left = np.clip(np.floor(position[inside]).astype(np.int64), 0, COMPOSITE_POINTS - 2)
    fraction = position[inside] - left
    lo, hi = profile[left], profile[left + 1]
    positive = (lo > 0.0) & (hi > 0.0)
    interpolated = (1.0 - fraction) * lo + fraction * hi
    interpolated[positive] = np.exp(
        (1.0 - fraction[positive]) * np.log(lo[positive])
        + fraction[positive] * np.log(hi[positive])
    )
    result[inside] = interpolated
    below = position < 0.0
    result[below] = profile[0] * (value[below] / 10.0**COMPOSITE_LOG_BETA_MIN) ** 2
    return result
