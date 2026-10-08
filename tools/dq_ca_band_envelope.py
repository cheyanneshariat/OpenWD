"""Finite-bin rigid-rotor P/R envelope used by the historical C–A estimate.

Integrates the continuous rotational population over each bin. Asymptotic
P/R weights are approximate; no collision or pressure-shift law is implied.
"""

import numpy as np

HC_OVER_K_CM_K = 1.438776877

def polynomial(origin, upper_b, lower_b, branch):
    """Return c0,c1,c2 for nu(J) on one physical P/R branch."""
    if not all(np.isfinite((origin, upper_b, lower_b))) or upper_b <= 0 or lower_b <= 0:
        raise ValueError("finite positive rigid-rotor constants required")
    bsum, bdiff = upper_b + lower_b, upper_b - lower_b
    if branch == -1:  # m=-J, J>=1
        return float(origin), float(-bsum), float(bdiff), 1.0
    if branch == 1:  # m=J+1, J>=0
        return float(origin + bsum + bdiff), float(bsum + 2 * bdiff), float(bdiff), 0.0
    raise ValueError("only P (-1) and R (+1) branches are present")

def value(coefficients, j):
    c0, c1, c2 = coefficients
    return c0 + c1 * j + c2 * j**2

def monotonic_segments(origin, upper_b, lower_b, branch):
    c0, c1, c2, start = polynomial(origin, upper_b, lower_b, branch)
    coefficients = (c0, c1, c2)
    if c2 == 0:
        if c1 == 0:
            raise ValueError("zero Delta B and zero branch slope")
        return [(coefficients, start, np.inf)]
    turning = -c1 / (2 * c2)
    if turning > start:
        return [(coefficients, start, turning), (coefficients, turning, np.inf)]
    return [(coefficients, start, np.inf)]

def inverse_on_segment(coefficients, frequency, lower, upper):
    """Invert a monotonic quadratic, including the Delta-B=0 linear case."""
    c0, c1, c2 = coefficients
    frequency = np.asarray(frequency, dtype=float)
    if c2 == 0:
        if c1 == 0:
            raise ValueError("constant frequency has no invertible J mapping")
        answer = (frequency - c0) / c1
    else:
        discriminant = c1**2 - 4 * c2 * (c0 - frequency)
        scale = max(c1**2, np.max(np.abs(4 * c2 * (c0 - frequency))), 1.0)
        if np.any(discriminant < -2e-12 * scale):
            raise FloatingPointError("negative inverse discriminant inside segment range")
        root = np.sqrt(np.maximum(discriminant, 0))
        first = (-c1 + root) / (2 * c2)
        second = (-c1 - root) / (2 * c2)
        tolerance = 2e-8 * max(1.0, lower, upper if np.isfinite(upper) else lower)
        valid_first = (first >= lower - tolerance) & (
            first <= upper + tolerance if np.isfinite(upper) else True
        )
        valid_second = (second >= lower - tolerance) & (
            second <= upper + tolerance if np.isfinite(upper) else True
        )
        if np.any(~(valid_first | valid_second)):
            raise FloatingPointError("no quadratic root belongs to monotonic segment")
        answer = np.where(valid_first, first, second)
    answer = np.maximum(answer, lower)
    if np.isfinite(upper):
        answer = np.minimum(answer, upper)
    residual = np.max(np.abs(value(coefficients, answer) - frequency))
    if residual > 3e-7 * max(np.max(abs(frequency)), 1.0):
        raise FloatingPointError("quadratic inverse failed reconstruction check")
    return answer

def segment_cell_intervals(edges, coefficients, lower, upper):
    """Return table-bin indices and exact J intervals for one monotonic piece."""
    low_frequency = value(coefficients, lower)
    c0, c1, c2 = coefficients
    if np.isfinite(upper):
        high_frequency = value(coefficients, upper)
    elif c2 > 0 or (c2 == 0 and c1 > 0):
        high_frequency = np.inf
    else:
        high_frequency = -np.inf
    minimum, maximum = min(low_frequency, high_frequency), max(low_frequency, high_frequency)
    if maximum <= edges[0] or minimum >= edges[-1]:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    mapped_frequency = np.clip(edges, minimum, maximum)
    j_edge = inverse_on_segment(
        coefficients, mapped_frequency, lower, upper
    )
    ja = np.minimum(j_edge[:-1], j_edge[1:])
    jb = np.maximum(j_edge[:-1], j_edge[1:])
    use = jb > ja + 2e-12
    return np.flatnonzero(use), ja[use], jb[use]

def geometry_for_band(edges, origin, upper_b, lower_b):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("frequency edges must be a one-dimensional cell grid")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("frequency edges must be finite and strictly increasing")
    geometry = []
    for branch in (-1, 1):
        for coefficients, lower, upper in monotonic_segments(
            origin, upper_b, lower_b, branch
        ):
            index, ja, jb = segment_cell_intervals(
                edges, coefficients, lower, upper
            )
            geometry.append((index, ja, jb))
    return geometry

def exact_band_fractions(geometry, lower_b, temperature, number_of_bins):
    """Exact unregularized finite-bin mass normalized on physical P+R J."""
    if (
        not np.isfinite(temperature) or not np.isfinite(lower_b)
        or temperature <= 0 or lower_b <= 0
    ):
        raise ValueError("finite positive temperature and B required")
    if not isinstance(number_of_bins, (int, np.integer)) or number_of_bins <= 0:
        raise ValueError("positive integer number_of_bins required")
    a = HC_OVER_K_CM_K * lower_b / temperature
    # P begins at J=1, R at J=0; alpha_P=alpha_R=1/2.
    normalization = 0.5 * (np.exp(-2 * a) + 1.0)
    result = np.zeros(number_of_bins)
    for index, ja, jb in geometry:
        mass = 0.5 * (
            np.exp(-a * ja * (ja + 1)) - np.exp(-a * jb * (jb + 1))
        ) / normalization
        np.add.at(result, index, mass)
    if result.min() < -1e-14 or result.sum() > 1 + 2e-12:
        raise FloatingPointError("invalid exact finite-grid probability")
    return np.maximum(result, 0), float(1 - result.sum())
