"""Atomic radiative-rate helpers shared by the NLTE model atoms."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ._compat import trapezoid
from .constants import ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI
from .opacity import HydrogenLine
FloatArray = NDArray[np.float64]
_LIGHT_SPEED_ANGSTROM_PER_SECOND = LIGHT_SPEED * 1e8

def einstein_a_from_absorption_oscillator_strength(line: HydrogenLine) -> float:
    """Return the shell-averaged Einstein A coefficient in s^-1."""

    lower_weight = 2.0 * line.lower_level**2
    upper_weight = 2.0 * line.upper_level**2
    frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
    return float(
        8.0
        * PI**2
        * ELEMENTARY_CHARGE_ESU**2
        * frequency**2
        / (ELECTRON_MASS * LIGHT_SPEED**3)
        * lower_weight
        / upper_weight
        * line.absorption_oscillator_strength
    )

def _profile_averaged_mean_intensity_nu(
    wavelength_angstrom: FloatArray,
    line_opacity: FloatArray,
    mean_intensity_lambda: FloatArray,
) -> FloatArray:
    """Average J_nu over the local line profile at each depth."""

    frequency_jacobian = (
        _LIGHT_SPEED_ANGSTROM_PER_SECOND
        / wavelength_angstrom[:, np.newaxis] ** 2
    )
    numerator = trapezoid(
        line_opacity * mean_intensity_lambda,
        wavelength_angstrom,
        axis=0,
    )
    denominator = trapezoid(
        line_opacity * frequency_jacobian,
        wavelength_angstrom,
        axis=0,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
