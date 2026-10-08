"""Hydrogenic bound--free and thermally averaged free--free Gaunt factors."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BOLTZMANN,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PLANCK,
)


try:
    from . import _rt
except ImportError:
    _rt = None


FloatArray = NDArray[np.float64]
_TABLE_PATH = Path(__file__).with_name("data") / "gaunt" / "gauntff.dat"

# Shell-averaged hydrogenic bound-free fits of Mihalas (1967), as used by
# TLUSTY/SYNSPEC for levels n=1--10.  The columns multiply
# x^-3, x^-2, x^-1, 1, x, x^2, and x^3, with x=nu/(2.99793e14 Hz).
_BOUND_FREE_COEFFICIENTS = np.asarray(
    [
        [0.0, 12.803223, -5.5759888, 1.2302628, -2.9094219e-3, 7.3993579e-6, -8.7356966e-9],
        [-2.0244141, 2.1325684, -1.2709045, 1.1595421, -2.0735860e-3, 2.7033384e-6, 0.0],
        [-0.23387146, 0.52471924, -0.55936432, 1.1450949, -1.9366592e-3, 2.3572356e-6, 0.0],
        [-5.4418565e-2, 0.19683564, -0.31190730, 1.1306695, -1.3482273e-3, -4.6949424e-6, 2.3548636e-8],
        [-8.9182854e-3, 5.5545091e-2, -0.16051018, 1.1190904, -1.0401085e-3, -6.9943488e-6, 2.8496742e-8],
        [-5.5303574e-3, 4.1921183e-2, -0.13075417, 1.1168376, -8.9466573e-4, -8.8393133e-6, 3.4696768e-8],
        [-2.2752881e-3, 2.3350812e-2, -9.5441161e-2, 1.1128632, -7.4833260e-4, -1.0244504e-5, 3.8595771e-8],
        [-9.7200274e-4, 1.3298411e-2, -7.1010560e-2, 1.1093137, -6.2619148e-4, -1.1342068e-5, 4.1477731e-8],
        [-4.9576163e-4, 8.5139736e-3, -5.6046560e-2, 1.1078717, -5.4837392e-4, -1.2157943e-5, 4.3796716e-8],
        [-2.9467046e-4, 6.1516856e-3, -4.7326370e-2, 1.1052734, -4.4341570e-4, -1.3235905e-5, 4.7003140e-8],
    ],
    dtype=np.float64,
)


def hydrogen_bound_free_gaunt_factor(
    principal_quantum_number: int,
    frequency_hz: ArrayLike,
) -> FloatArray:
    """Return the shell-averaged hydrogenic bound-free Gaunt factor.

    The Mihalas (1967) polynomial fits reproduce the Karzas--Latter quantum
    calculations to about one percent for wavelengths longer than 50 A for
    hydrogen.  Fits are supplied for ``n=1..10``; the high-level asymptotic
    value is unity.
    """

    if principal_quantum_number < 1:
        raise ValueError("principal_quantum_number must be positive")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must be finite and positive")
    if principal_quantum_number > _BOUND_FREE_COEFFICIENTS.shape[0]:
        return np.ones_like(frequency)

    x = frequency / 2.99793e14
    basis = np.stack(
        (x**-3, x**-2, x**-1, np.ones_like(x), x, x**2, x**3),
        axis=0,
    )
    result = np.tensordot(
        _BOUND_FREE_COEFFICIENTS[principal_quantum_number - 1],
        basis,
        axes=(0, 0),
    )
    # The published polynomial accuracy is quoted only longward of 50 A for
    # hydrogen.  Retain Kramers' asymptotic factor rather than extrapolating
    # the polynomial into the X-ray regime, where its positive powers of x
    # are not physical.
    shortward_of_fit = frequency > LIGHT_SPEED / (50.0e-8)
    return np.where(shortward_of_fit, 1.0, np.maximum(result, 0.0))


@lru_cache(maxsize=1)
def _read_van_hoof_table() -> tuple[FloatArray, float, float, float]:
    """Read the van Hoof et al. (2014) non-relativistic table.

    The bundled data retain their original BSD-style notice.  Values following
    the main 146 by 81 table are quoted numerical uncertainties and are not
    needed for opacity interpolation.
    """

    tokens: list[str] = []
    with _TABLE_PATH.open("r", encoding="ascii") as stream:
        for line in stream:
            content = line.split("#", 1)[0].strip()
            if content:
                tokens.extend(content.split())
    magic = int(tokens[0])
    n_gamma = int(tokens[1])
    n_u = int(tokens[2])
    log_gamma_min = float(tokens[3])
    log_u_min = float(tokens[4])
    step = float(tokens[5])
    if magic != 20140210 or (n_gamma, n_u) != (81, 146) or step <= 0.0:
        raise ValueError(f"unexpected van Hoof Gaunt table header in {_TABLE_PATH}")
    count = n_gamma * n_u
    values = np.asarray(tokens[6 : 6 + count], dtype=np.float64)
    if values.size != count or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"invalid van Hoof Gaunt table in {_TABLE_PATH}")
    return values.reshape(n_u, n_gamma), log_gamma_min, log_u_min, step


def _cubic_grid_indices_and_weights(
    coordinate: FloatArray,
    minimum: float,
    step: float,
    size: int,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Return the four nodes and Lagrange weights used by van Hoof's code."""

    maximum = minimum + (size - 1) * step
    clipped = np.clip(coordinate, minimum, maximum)
    cell = np.floor((clipped - minimum) / step).astype(np.int64)
    base = np.clip(cell - 1, 0, size - 4)
    nodes = base[..., np.newaxis] + np.arange(4, dtype=np.int64)
    node_coordinates = minimum + nodes * step
    weights = np.ones(node_coordinates.shape, dtype=np.float64)
    for index in range(4):
        for other in range(4):
            if index != other:
                weights[..., index] *= (
                    (clipped - node_coordinates[..., other])
                    / (node_coordinates[..., index] - node_coordinates[..., other])
                )
    return nodes, weights


def hydrogen_free_free_gaunt_factor(
    wavelength_angstrom: ArrayLike,
    temperature: ArrayLike,
    *,
    ionic_charge: float = 1.0,
) -> FloatArray:
    r"""Return the thermally averaged non-relativistic free--free factor.

    This evaluates the van Hoof et al. (2014) table using their
    reference third-order interpolation in ``log10(gamma^2)`` and
    ``log10(u)``, where ``gamma^2 = Z^2 Ry/(kT)`` and ``u = h nu/(kT)``.
    Inputs broadcast normally.  The enormous published grid fully contains
    white-dwarf atmosphere conditions; values outside it are conservatively
    clipped to the nearest tabulated boundary.
    """

    wavelength, temperature_array = np.broadcast_arrays(
        np.asarray(wavelength_angstrom, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(~np.isfinite(temperature_array))
        or np.any(temperature_array <= 0.0)
        or not np.isfinite(ionic_charge)
        or ionic_charge <= 0.0
    ):
        raise ValueError(
            "wavelength, temperature, and ionic_charge must be finite and positive"
        )

    table, log_gamma_min, log_u_min, step = _read_van_hoof_table()
    gamma_squared = ionic_charge**2 * HYDROGEN_IONIZATION_ENERGY / (
        BOLTZMANN * temperature_array
    )
    u = (
        PLANCK
        * LIGHT_SPEED
        / (wavelength * 1.0e-8 * BOLTZMANN * temperature_array)
    )
    compiled = None if _rt is None else getattr(_rt, "cubic_table_interpolate", None)
    if compiled is not None:
        result = np.empty(wavelength.shape, dtype=np.float64)
        compiled(table, np.ascontiguousarray(np.log10(gamma_squared)).ravel(),
            np.ascontiguousarray(np.log10(u)).ravel(),
            log_gamma_min, log_u_min, step, result.ravel())
        return np.maximum(result, 0.0)
    gamma_nodes, gamma_weights = _cubic_grid_indices_and_weights(
        np.log10(gamma_squared), log_gamma_min, step, table.shape[1]
    )
    u_nodes, u_weights = _cubic_grid_indices_and_weights(
        np.log10(u), log_u_min, step, table.shape[0]
    )
    result = np.zeros(wavelength.shape, dtype=np.float64)
    for u_index in range(4):
        for gamma_index in range(4):
            result += (
                u_weights[..., u_index]
                * gamma_weights[..., gamma_index]
                * table[
                    u_nodes[..., u_index],
                    gamma_nodes[..., gamma_index],
                ]
            )
    return np.maximum(result, 0.0)
