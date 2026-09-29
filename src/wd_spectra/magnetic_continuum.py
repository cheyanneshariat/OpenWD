"""Magnetic hydrogen continuum opacity: RWA photoionization and cyclotron.

The bound-free opacity follows the stationary-state rigid-wavefunction
approximation (RWA) of Rohrmann (2026, A&A, doi:10.1051/0004-6361/202658917):
field-dependent H2db state energies and polarization-specific thresholds,
Wigner geometric factors and the published branching fractions, applied to the
atmosphere's own (field-dependent) shell populations.  Electron cyclotron
absorption between adjacent Landau levels follows Vera-Rueda & Rohrmann
(2024, A&A 687, A141), with its causal dispersion partner.
"""

from __future__ import annotations


from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .magnetic_atomic import H2dbEnergyDatabase, read_h2db_energy_database


FloatArray = NDArray[np.float64]
Polarization = Literal[-1, 0, 1]

ROHRMANN_2026_DOI = "10.1051/0004-6361/202658917"
ROHRMANN_2026_ZENODO_DOI = "10.5281/zenodo.19005375"
MAGNETIC_ATOMIC_FIELD_MEGAGAUSS = 4701.03
# The zero-field pseudo-continuum dissolves the Lyman--Brackett edges.
DISSOLVED_LEVEL_MAXIMUM_LOWER_LEVEL = 4
_ROZSNYAI_N8_GAUNT_COEFFICIENTS = np.asarray(
    [
        [2.1517760, 0.27905785, 1.9565383, 1.5898075e3, 1.2587007e2, 2.6365069e1],
        [8.9141668, 0.77348612, 2.4251235, 5.5361197e4, 6.6688290e1, 2.6365069e1],
        [8.4783079, 1.0621360, 1.7183014, 8.4346771e4, 1.9797032e1, 6.1582953],
        [7.0937963, 1.3847772, 1.2168604, 1.2401665e5, 1.0649873e1, 2.9762973],
        [4.3320977, 1.6934970, 0.69386273, 1.3504603e5, 6.9042520, 2.0691130],
        [2.7645291, 2.3188395, 0.46265837, 1.0915438e5, 5.0277618, 2.0691130],
        [1.0102198, 2.9697488, 0.23124628, 4.5814475e4, 3.7403664, 2.0691130],
        [0.16520881, 3.6600979, 0.045867853, 7.9957177e3, 2.8425964, 2.0691130],
    ],
    dtype=np.float64,
)


def _rozsnyai_n8_sublevel_gaunt_ratio(
    orbital_quantum_number: int,
    quasi_principal_squared: FloatArray,
) -> FloatArray:
    """Return ``g_8l/g_8`` from Rozsnyai--Jacobs Table 1."""

    orbital = int(orbital_quantum_number)
    if not 0 <= orbital < 8:
        raise ValueError("n=8 requires 0<=l<8")
    quasi_squared = np.asarray(quasi_principal_squared, dtype=np.float64)
    omega = 1.0 + 64.0 / np.maximum(quasi_squared, np.finfo(float).tiny)

    def sublevel_gaunt(index: int) -> FloatArray:
        a1, a3, a2, a4, a5, omega_m = _ROZSNYAI_N8_GAUNT_COEFFICIENTS[index]
        low = a1 * omega * (a2 + omega) ** (-a3)
        high = a4 * omega * (a5 + omega) ** (-index - 1.5)
        return np.where(omega <= omega_m, low, high)

    all_gaunt = np.stack([sublevel_gaunt(index) for index in range(8)])
    shell_gaunt = np.sum(
        (2.0 * np.arange(8, dtype=np.float64) + 1.0)[:, np.newaxis]
        * all_gaunt.reshape(8, -1),
        axis=0,
    ).reshape(omega.shape) / 64.0
    return sublevel_gaunt(orbital) / np.maximum(
        shell_gaunt, np.finfo(float).tiny
    )


def _rwa_geometric_factors(
    orbital_quantum_number: int,
    magnetic_quantum_number: int,
    polarization: Polarization,
) -> tuple[float, float]:
    """Return Rohrmann (2026) ``A`` and ``B`` Wigner weights."""

    orbital = int(orbital_quantum_number)
    magnetic = int(magnetic_quantum_number)
    if orbital < 0 or abs(magnetic) > orbital or polarization not in {-1, 0, 1}:
        raise ValueError("invalid dipole-transition quantum numbers")
    if polarization == -1:
        factor_a = (
            3.0 * (orbital - magnetic + 2) * (orbital - magnetic + 1)
            / (2.0 * (orbital + 1) * (2 * orbital + 3))
        )
    elif polarization == 0:
        factor_a = (
            3.0 * ((orbital + 1) ** 2 - magnetic**2)
            / ((orbital + 1) * (2 * orbital + 3))
        )
    else:
        factor_a = (
            3.0 * (orbital + magnetic + 2) * (orbital + magnetic + 1)
            / (2.0 * (orbital + 1) * (2 * orbital + 3))
        )
    if orbital == 0:
        return factor_a, 0.0
    if polarization == -1:
        factor_b = (
            3.0 * (orbital + magnetic) * (orbital + magnetic - 1)
            / (2.0 * orbital * (2 * orbital - 1))
        )
    elif polarization == 0:
        factor_b = (
            3.0 * (orbital**2 - magnetic**2)
            / (orbital * (2 * orbital - 1))
        )
    else:
        factor_b = (
            3.0 * (orbital - magnetic) * (orbital - magnetic - 1)
            / (2.0 * orbital * (2 * orbital - 1))
        )
    return factor_a, factor_b


def _rwa_branching_fractions(
    principal_quantum_number: int,
    orbital_quantum_number: int,
    quasi_principal_squared: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return ``l+1``/``l-1`` branches for the RWA continuum.

    These are Tables 3, 4, and 5 of Rohrmann (2026).  The longer expressions
    for ``n=6,7`` matter because dissolved high levels dominate much of the
    optical bound-free opacity in warm hydrogen atmospheres.  For ``n>=8``
    the leading high-n closure of their Eq. 18 is used: the ``l+1`` branch
    dominates.  The Rozsnyai--Jacobs sublevel Gaunt factors are explicit for
    the important ``n=8`` shell; a unity sublevel-to-shell ratio above it
    preserves the exact shell sum pending transcription of the remaining
    rows of their table.
    """

    n = int(principal_quantum_number)
    l = int(orbital_quantum_number)
    x = np.minimum(np.asarray(quasi_principal_squared, dtype=np.float64), 1.0e12)
    if n == 1:
        denominator = np.ones_like(x)
        plus = np.ones_like(x)
        minus = np.zeros_like(x)
    elif n == 2:
        denominator = (4.0 + 3.0 * x) * (4.0 + 5.0 * x)
        if l == 0:
            plus = 16.0 * (1.0 + x) * (4.0 + x)
            minus = np.zeros_like(x)
        else:
            plus = 128.0 * x * (1.0 + x) / 9.0
            minus = 4.0 * x * (4.0 + x) / 9.0
    elif n == 3:
        denominator = (81.0 + 78.0 * x + 13.0 * x**2) * (
            81.0 + 126.0 * x + 29.0 * x**2
        )
        if l == 0:
            plus = 9.0 * (1.0 + x) * (9.0 + x) * (27.0 + 7.0 * x) ** 2
            minus = np.zeros_like(x)
        elif l == 1:
            plus = 432.0 * x * (1.0 + x) * (4.0 + x) * (9.0 + x)
            minus = 24.0 * x * (3.0 + x) ** 2 * (9.0 + x)
        else:
            plus = 7776.0 * x**2 * (1.0 + x) * (4.0 + x) / 25.0
            minus = 144.0 * x**2 * (1.0 + x) * (9.0 + x) / 25.0
    elif n == 4:
        denominator = (
            (12288.0 + 13056.0 * x + 3152.0 * x**2 + 197.0 * x**3)
            * (12288.0 + 20736.0 * x + 6800.0 * x**2 + 539.0 * x**3)
            / 9.0
        )
        if l == 0:
            plus = (
                256.0 * (1.0 + x) * (16.0 + x)
                * (768.0 + 288.0 * x + 23.0 * x**2) ** 2 / 9.0
            )
            minus = np.zeros_like(x)
        elif l == 1:
            plus = (
                8192.0 * x * (1.0 + x) * (4.0 + x) * (16.0 + x)
                * (80.0 + 9.0 * x) ** 2 / 45.0
            )
            minus = (
                16.0 * x * (16.0 + x)
                * (1280.0 + 608.0 * x + 57.0 * x**2) ** 2 / 45.0
            )
        elif l == 2:
            plus = (
                1048576.0 * x**2 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) / 75.0
            )
            minus = (
                2048.0 * x**2 * (1.0 + x) * (16.0 + x)
                * (48.0 + 7.0 * x) ** 2 / 225.0
            )
        else:
            plus = (
                16777216.0 * x**3 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) / 2205.0
            )
            minus = (
                65536.0 * x**3 * (1.0 + x) * (4.0 + x)
                * (16.0 + x) / 735.0
            )
    elif n == 5:
        denominator = (
            (1171875.0 + 1312500.0*x + 372250.0*x**2 + 36100.0*x**3 + 1083.0*x**4)
            * (1171875.0 + 2062500.0*x + 786250.0*x**2 + 95700.0*x**3 + 3467.0*x**4)
            / 9.0
        )
        if l == 0:
            plus = (
                625.0 * (1.0 + x) * (25.0 + x)
                * (46875.0 + 20625.0*x + 2545.0*x**2 + 91.0*x**3) ** 2 / 9.0
            )
            minus = np.zeros_like(x)
        elif l == 1:
            plus = (
                1.0e4 * x * (1.0 + x) * (4.0 + x) * (25.0 + x)
                * (9375.0 + 1650.0*x + 67.0*x**2) ** 2 / 81.0
            )
            minus = (
                200.0 * x * (25.0 + x)
                * (46875.0 + 25875.0*x + 3725.0*x**2 + 149.0*x**3) ** 2 / 81.0
            )
        elif l == 2:
            plus = (
                1.0e5 * x**2 * (1.0 + x) * (4.0 + x) * (9.0 + x)
                * (25.0 + x) * (175.0 + 11.0*x) ** 2 / 21.0
            )
            minus = (
                2000.0 * x**2 * (1.0 + x) * (25.0 + x)
                * (2625.0 + 590.0*x + 29.0*x**2) ** 2 / 63.0
            )
        elif l == 3:
            plus = (
                2.0e8 * x**3 * (1.0 + x) * (4.0 + x) * (9.0 + x)
                * (16.0 + x) * (25.0 + x) / 441.0
            )
            minus = (
                320000.0 * x**3 * (1.0 + x) * (4.0 + x) * (25.0 + x)
                * (25.0 + 2.0*x) ** 2 / 147.0
            )
        else:
            plus = (
                1.0e9 * x**4 * (1.0 + x) * (4.0 + x) * (9.0 + x)
                * (16.0 + x) / 5103.0
            )
            minus = (
                8.0e6 * x**4 * (1.0 + x) * (4.0 + x) * (9.0 + x)
                * (25.0 + x) / 5103.0
            )
    elif n == 6:
        denominator = (
            (
                302330880.0 + 349920000.0*x + 108708480.0*x**2
                + 12909024.0*x**3 + 628260.0*x**4 + 10471.0*x**5
            )
            * (
                302330880.0 + 545875200.0*x + 226281600.0*x**2
                + 33480864.0*x**3 + 1953540.0*x**4 + 38081.0*x**5
            )
            / 25.0
        )
        if l == 0:
            plus = (
                36.0 * (1.0 + x) * (36.0 + x)
                * (
                    50388480.0 + 24261120.0*x + 3654720.0*x**2
                    + 211104.0*x**3 + 4046.0*x**4
                ) ** 2 / 25.0
            )
            minus = np.zeros_like(x)
        elif l == 1:
            plus = (
                384.0 * x * (1.0 + x) * (4.0 + x) * (36.0 + x)
                * (4898880.0 + 1061424.0*x + 70092.0*x**2 + 1425.0*x**3) ** 2
                / 35.0
            )
            minus = (
                12.0 * x * (36.0 + x)
                * (
                    19595520.0 + 11757312.0*x + 2057184.0*x**2
                    + 132528.0*x**3 + 2761.0*x**4
                ) ** 2 / 35.0
            )
        elif l == 2:
            plus = (
                2.0**15 * 3.0**5 * x**2 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (36.0 + x)
                * (9072.0 + 936.0*x + 23.0*x**2) ** 2 / 175.0
            )
            minus = (
                2.0**10 * 3.0**2 * x**2 * (1.0 + x) * (36.0 + x)
                * (326592.0 + 89424.0*x + 7092.0*x**2 + 167.0*x**3) ** 2
                / 175.0
            )
        elif l == 3:
            plus = (
                2.0**17 * 3.0**5 * x**3 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (36.0 + x)
                * (324.0 + 13.0*x) ** 2 / 245.0
            )
            minus = (
                2.0**13 * 3.0**6 * x**3 * (1.0 + x) * (4.0 + x)
                * (36.0 + x) * (1296.0 + 168.0*x + 5.0*x**2) ** 2
                / 245.0
            )
        elif l == 4:
            plus = (
                2.0**21 * 3.0**5 * x**4 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) * (36.0 + x)
                / 35.0
            )
            minus = (
                2.0**17 * 3.0**5 * x**4 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (36.0 + x) * (20.0 + x) ** 2 / 175.0
            )
        else:
            plus = (
                2.0**24 * 3.0**8 * x**5 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) / 21175.0
            )
            minus = (
                2.0**19 * 3.0**5 * x**5 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (36.0 + x) / 4235.0
            )
    elif n == 7:
        denominator = (
            (
                622857924045.0 + 737260399890.0*x + 242899890135.0*x**2
                + 32480603148.0*x**3 + 1993432651.0*x**4
                + 55606082.0*x**5 + 567409.0*x**6
            )
            * (
                622857924045.0 + 1144024758450.0*x + 500240606775.0*x**2
                + 82901603148.0*x**3 + 6067218955.0*x**4
                + 196890722.0*x**5 + 2297425.0*x**6
            )
            / 2025.0
        )
        if l == 0:
            plus = (
                7.0**4 * (1.0 + x) * (49.0 + x)
                * (
                    12711386205.0 + 6485401125.0*x + 1097665170.0*x**2
                    + 79704282.0*x**3 + 2547265.0*x**4 + 29233.0*x**5
                ) ** 2 / 2025.0
            )
            minus = np.zeros_like(x)
        elif l == 1:
            plus = (
                2.0**5 * 7.0**6 * x * (1.0 + x) * (4.0 + x) * (49.0 + x)
                * (86472015.0 + 21176820.0*x + 1766450.0*x**2 + 60116.0*x**3 + 711.0*x**4) ** 2
                / 2025.0
            )
            minus = (
                2.0**4 * 7.0**2 * x * (49.0 + x)
                * (
                    4237128735.0 + 2680632465.0*x + 525218750.0*x**2
                    + 42435274.0*x**3 + 1471715.0*x**4 + 18021.0*x**5
                ) ** 2 / 2025.0
            )
        elif l == 2:
            plus = (
                2.0**6 * 7.0**6 * x**2 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (49.0 + x)
                * (7411887.0 + 972405.0*x + 40229.0*x**2 + 527.0*x**3) ** 2
                / 2025.0
            )
            minus = (
                2.0**5 * 7.0**4 * x**2 * (1.0 + x) * (49.0 + x)
                * (155649627.0 + 47799108.0*x + 4760154.0*x**2 + 186788.0*x**3 + 2483.0*x**4) ** 2
                / 6075.0
            )
        elif l == 3:
            plus = (
                2.0**9 * 7.0**7 * x**3 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (49.0 + x)
                * (108045.0 + 7350.0*x + 121.0*x**2) ** 2 / 6075.0
            )
            minus = (
                2.0**9 * 7.0**5 * x**3 * (1.0 + x) * (4.0 + x)
                * (49.0 + x) * (21.0 + x) ** 2
                * (36015.0 + 4165.0*x + 94.0*x**2) ** 2 / 2025.0
            )
        elif l == 4:
            plus = (
                2.0**9 * 7.0**10 * x**4 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) * (49.0 + x)
                * (539.0 + 15.0*x) ** 2 / 40095.0
            )
            minus = (
                2.0**9 * 7.0**6 * x**4 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (49.0 + x)
                * (132055.0 + 11074.0*x + 219.0*x**2) ** 2 / 200475.0
            )
        elif l == 5:
            plus = (
                2.0**13 * 7.0**12 * x**5 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) * (36.0 + x)
                * (49.0 + x) / 245025.0
            )
            minus = (
                2.0**12 * 7.0**8 * x**5 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (49.0 + x)
                * (147.0 + 5.0*x) ** 2 / 147015.0
            )
        else:
            plus = (
                2.0**14 * 7.0**13 * x**6 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) * (36.0 + x)
                / 11293425.0
            )
            minus = (
                2.0**13 * 7.0**10 * x**6 * (1.0 + x) * (4.0 + x)
                * (9.0 + x) * (16.0 + x) * (25.0 + x) * (49.0 + x)
                / 11293425.0
            )
    elif n == 8:
        if not 0 <= l < n:
            raise ValueError("invalid orbital quantum number")
        denominator = np.ones_like(x)
        plus = _rozsnyai_n8_sublevel_gaunt_ratio(l, x)
        minus = np.zeros_like(x)
    elif n > 8:
        if not 0 <= l < n:
            raise ValueError("invalid orbital quantum number")
        denominator = np.ones_like(x)
        plus = np.ones_like(x)
        minus = np.zeros_like(x)
    else:
        raise ValueError("principal quantum number must be positive")
    return plus / denominator, minus / denominator


def explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    polarization: Polarization,
    energy_database: H2dbEnergyDatabase,
    *,
    maximum_level: int = 8,
    include_dissolved_levels: bool = True,
    include_centered_motion: bool = True,
) -> FloatArray:
    """Evaluate the explicit stationary-state RWA H I continuum.

    H2db state energies, polarization-specific continuum thresholds, Wigner
    geometric factors, and the analytic branching fractions of Rohrmann
    (2026) are combined with the atmosphere's local shell populations. Both
    spin partners are included. Through ``n=5`` every energy is tabulated. At
    higher shells the public archive's missing states are completed with the
    arbitrary-level centered-state fits of Vera-Rueda & Rohrmann (2020).
    Exact branching fractions are used through ``n=7`` and the published
    high-n, ``l+1``-dominant closure above it. Decentered states remain
    omitted.  With ``include_dissolved_levels`` the n=1--4 edges are
    continued below each substate's shifted threshold by the dissolved-level
    (pseudo-continuum) prescription of the zero-field model.  No stellar
    parameter or opacity normalization is fitted. ``include_centered_motion``
    uses the same transverse-mass population weights as the magnetic EOS;
    disabling it uses stationary-state Boltzmann weights throughout.
    """

    from .opacity import _atmosphere_level_distribution

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    field = float(field_strength_megagauss)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if not np.isfinite(field) or field < 0.0:
        raise ValueError("field_strength_megagauss must be finite and nonnegative")
    if polarization not in {-1, 0, 1}:
        raise ValueError("polarization must be -1, 0, or +1")
    # The ordinary continuum carries explicit H I bound-free opacity through
    # n=8 (higher shells enter as dissolved-level/ free-free closures).  The
    # magnetic calculation replaces exactly those shells, never more.
    if not 1 <= maximum_level <= 8:
        raise ValueError("explicit RWA is implemented for maximum_level=1..8")
    photon_energy = PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / HYDROGEN_IONIZATION_ENERGY
    temperature = atmosphere.temperature
    stimulated_emission = -np.expm1(
        -photon_energy[:, np.newaxis]
        * HYDROGEN_IONIZATION_ENERGY
        / (BOLTZMANN * temperature[np.newaxis, :])
    )
    levels = _atmosphere_level_distribution(
        atmosphere, maximum_level=max(40, maximum_level)
    )
    extinction = np.zeros(
        (wavelength.size, atmosphere.n_depth), dtype=np.float64
    )
    kernels = _explicit_rwa_atomic_kernels(
        str(energy_database.root.resolve()),
        tuple(float(value) for value in wavelength),
        field,
        int(polarization),
        int(maximum_level),
        bool(include_centered_motion),
    )
    for principal, (
        state_energy,
        mass_ratio,
        atomic_kernel,
        state_threshold,
        threshold_kernel,
    ) in enumerate(kernels, start=1):
        shell_population = levels.population_density[..., principal - 1]
        relative_energy = state_energy - np.min(state_energy)
        boltzmann = np.exp(
            -relative_energy[:, np.newaxis]
            * HYDROGEN_IONIZATION_ENERGY
            / (BOLTZMANN * temperature[np.newaxis, :])
        )
        state_weight = mass_ratio[:, np.newaxis] * boltzmann
        state_population = (
            shell_population[np.newaxis, :]
            * state_weight
            / np.sum(state_weight, axis=0)
        )
        extinction += atomic_kernel.T @ state_population
        if include_dissolved_levels and principal <= DISSOLVED_LEVEL_MAXIMUM_LOWER_LEVEL:
            extinction += _dissolved_rwa_extinction(
                atmosphere,
                levels,
                principal,
                photon_energy,
                state_threshold,
                threshold_kernel,
                state_population,
            )
    return (
        extinction
        * stimulated_emission
        / atmosphere.mass_density[np.newaxis, :]
    )


def _dissolved_rwa_extinction(
    atmosphere: Atmosphere,
    levels: object,
    principal: int,
    photon_energy: FloatArray,
    state_threshold: FloatArray,
    threshold_kernel: FloatArray,
    state_population: FloatArray,
) -> FloatArray:
    """Daeppen--Anderson--Mihalas dissolved-level continuation of each edge.

    This is the magnetic counterpart of
    :func:`hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient`:
    below each substate's own threshold ``T`` a photon of energy ``E`` maps
    to the fictitious level ``n*^2 = 1/(T - E)``; the threshold cross
    section, continued as ``(T/E)^3``, is multiplied by the full dissolved
    fraction ``1 - w(n*)/w(n)`` for ``n* > n + 3`` and by its neutral part for
    ``n + 1 < n* <= n + 3`` (Lyman: only above the zero-field 925-A
    equivalent).  At zero field every threshold is ``1/n^2`` and the result is
    identical to the ordinary pseudo-continuum.
    """

    from .constants import LIGHT_SPEED as _c
    from .eos import (
        HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
        charged_particle_hydrogen_occupation_probability,
        hydrogen_occupation_probability,
    )

    state = atmosphere.hydrogen_lte_state
    correlated = state is not None and state.microfield_model == "qmhd"
    radius_scale = (
        state.neutral_radius_scale if state is not None else HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
    )
    minimum_level = float(principal + 1)
    if principal == 1:
        lyman_limit = PLANCK * _c / HYDROGEN_IONIZATION_ENERGY * 1.0e8
        minimum_level = max(minimum_level, 1.0 / np.sqrt(1.0 - lyman_limit / 925.0))
    grid = np.geomspace(minimum_level, 400.0, 240)
    upper = hydrogen_occupation_probability(
        atmosphere.neutral_h_density[np.newaxis, :],
        atmosphere.electron_density[np.newaxis, :],
        atmosphere.temperature[np.newaxis, :],
        grid[:, np.newaxis],
        neutral_radius_scale=radius_scale,
        correlated_microfields=correlated,
    )
    lower = levels.occupation_probability[:, principal - 1]  # type: ignore[attr-defined]
    total_survival = np.clip(upper / lower[np.newaxis, :], 0.0, 1.0)
    upper_charged = charged_particle_hydrogen_occupation_probability(
        atmosphere.electron_density[np.newaxis, :],
        grid[:, np.newaxis],
        atmosphere.temperature[np.newaxis, :] if correlated else None,
    )
    lower_charged = charged_particle_hydrogen_occupation_probability(
        atmosphere.electron_density,
        float(principal),
        atmosphere.temperature if correlated else None,
    )
    charged_survival = np.clip(upper_charged / lower_charged[np.newaxis, :], 0.0, 1.0)
    full_dissolved = 1.0 - total_survival
    neutral_dissolved = np.clip(charged_survival - total_survival, 0.0, 1.0)
    log_grid = np.log(grid)
    result = np.zeros((photon_energy.size, atmosphere.n_depth), dtype=np.float64)
    for threshold in np.unique(state_threshold):
        members = state_threshold == threshold
        weight = threshold_kernel[members] @ state_population[members]
        deficit = threshold - photon_energy
        selected = np.flatnonzero(
            (deficit > 0.0) & (deficit < 1.0 / minimum_level**2)
        )
        if selected.size == 0 or not np.any(weight):
            continue
        effective = np.log(1.0 / np.sqrt(deficit[selected]))
        position = np.clip(np.searchsorted(log_grid, effective) - 1, 0, grid.size - 2)
        fraction = np.clip(
            (effective - log_grid[position]) / (log_grid[position + 1] - log_grid[position]),
            0.0,
            1.0,
        )[:, np.newaxis]
        def sample(table: FloatArray) -> FloatArray:
            return (1.0 - fraction) * table[position] + fraction * table[position + 1]

        local = np.where(
            (np.exp(effective) > principal + 3.0)[:, np.newaxis],
            sample(full_dissolved),
            sample(neutral_dissolved),
        )
        continuation = (threshold / photon_energy[selected]) ** 3
        result[selected] += continuation[:, np.newaxis] * local * weight[np.newaxis, :]
    return result


# Retain at most one three-polarization grid. A disk visits many fields and
# two meshes per field; keeping all their (408, n_wave) arrays costs GBs.
# Three entries still reuse all kernels during a fixed-grid structure solve.
@lru_cache(maxsize=3)
def _explicit_rwa_atomic_kernels(
    energy_root: str,
    wavelength_angstrom: tuple[float, ...],
    field_strength_megagauss: float,
    polarization: int,
    maximum_level: int,
    include_centered_motion: bool = True,
) -> tuple[tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], ...]:
    """Cache temperature-independent RWA kernels by field and wavelength.

    Per shell: state energies, centered-motion mass ratios, the kernels
    ``(state, wavelength)``, each state's threshold (Ry) and the kernel's
    limit at that threshold.
    """

    from .gaunt import hydrogen_bound_free_gaunt_factor
    from .opacity import hydrogen_ground_state_photoionization_cross_section
    from .magnetic_eos import centered_transverse_mass_ratio

    database = read_h2db_energy_database(energy_root)
    field = float(field_strength_megagauss)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    photon_energy = PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / HYDROGEN_IONIZATION_ENERGY
    rydberg_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    output: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for principal in range(1, maximum_level + 1):
        states: list[tuple[int, int, float]] = []
        energies: list[float] = []
        transverse_mass_ratio: list[float] = []
        for orbital in range(principal):
            for magnetic in range(-orbital, orbital + 1):
                for spin in (-0.5, 0.5):
                    states.append((orbital, magnetic, spin))
                    energies.append(
                        database.energy_rydberg_with_analytic_fallback(
                            principal, orbital, magnetic, spin, field
                        )
                    )
                    mass_ratio = 1.0
                    if include_centered_motion:
                        try:
                            mass_ratio = centered_transverse_mass_ratio(
                                database, principal, orbital, magnetic, field
                            )
                        except (KeyError, ValueError):
                            pass
                    transverse_mass_ratio.append(mass_ratio)
        kernel = np.zeros((len(states), wavelength.size), dtype=np.float64)
        state_threshold = np.zeros(len(states), dtype=np.float64)
        threshold_kernel = np.zeros(len(states), dtype=np.float64)
        zero_threshold = 1.0 / principal**2
        for state_index, (orbital, magnetic, _spin) in enumerate(states):
            threshold = database.photoionization_threshold_with_analytic_fallback_rydberg(
                principal, orbital, magnetic, field, polarization
            )
            shifted_energy = photon_energy - (threshold - zero_threshold)
            valid = shifted_energy >= zero_threshold
            quasi_squared = 1.0 / np.maximum(
                shifted_energy - zero_threshold, 1.0e-14
            )
            branch_plus, branch_minus = _rwa_branching_fractions(
                principal, orbital, quasi_squared
            )
            factor_a, factor_b = _rwa_geometric_factors(
                orbital, magnetic, polarization  # type: ignore[arg-type]
            )
            frequency = np.maximum(shifted_energy, zero_threshold) * rydberg_frequency
            if principal == 1:
                shell_cross_section = hydrogen_ground_state_photoionization_cross_section(
                    frequency
                )
            else:
                shell_cross_section = (
                    2.815e29
                    * frequency**-3
                    / principal**5
                    * hydrogen_bound_free_gaunt_factor(principal, frequency)
                )
            kernel[state_index] = np.where(
                valid,
                photon_energy
                / np.maximum(shifted_energy, np.finfo(np.float64).tiny)
                * (factor_a * branch_plus + factor_b * branch_minus)
                * shell_cross_section,
                0.0,
            )
            # Limit of the same expression at this state's own threshold,
            # continued below it by the dissolved-level extension.
            edge_plus, edge_minus = _rwa_branching_fractions(
                principal, orbital, np.asarray([1.0e14])
            )
            edge_frequency = np.asarray([zero_threshold * rydberg_frequency])
            if principal == 1:
                edge_cross_section = hydrogen_ground_state_photoionization_cross_section(
                    edge_frequency
                )
            else:
                edge_cross_section = (
                    2.815e29
                    * edge_frequency**-3
                    / principal**5
                    * hydrogen_bound_free_gaunt_factor(principal, edge_frequency)
                )
            state_threshold[state_index] = threshold
            threshold_kernel[state_index] = float(
                (threshold / zero_threshold)
                * (factor_a * edge_plus[0] + factor_b * edge_minus[0])
                * edge_cross_section[0]
            )
        state_energy = np.ascontiguousarray(energies, dtype=np.float64)
        mass_ratio = np.ascontiguousarray(
            transverse_mass_ratio, dtype=np.float64
        )
        for array in (kernel, state_energy, mass_ratio, state_threshold, threshold_kernel):
            array.setflags(write=False)
        output.append(
            (state_energy, mass_ratio, kernel, state_threshold, threshold_kernel)
        )
    return tuple(output)


def explicit_rwa_photoionization_structure_continuum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase,
    *,
    maximum_level: int = 8,
    include_centered_motion: bool = True,
) -> FloatArray:
    """Return a polarization-averaged explicit RWA continuum at one field.

    Bound-free opacity is replaced through ``maximum_level`` (at most eight,
    the shells carried explicitly by the ordinary continuum).
    Ordinary higher-level bound-free, free-free, H-minus, and molecular terms
    are retained. Equal q weights are appropriate for the isotropic radiation
    field used in the 1D structural iteration.
    """

    from .opacity import (
        hydrogen_bound_free_mass_absorption_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
    )

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    continuum = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
    )
    ordinary_replaced = hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wavelength, maximum_level=maximum_level
    )
    magnetic_replaced = sum(
        explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            field_strength_megagauss,
            polarization,
            energy_database,
            maximum_level=maximum_level,
            include_centered_motion=include_centered_motion,
        )
        for polarization in (-1, 0, 1)
    ) / 3.0
    return np.maximum(0.0, continuum - ordinary_replaced) + magnetic_replaced


@dataclass(frozen=True)
class FreeElectronManifolds:
    """Magneto-ionic free-electron coefficients of the three circular modes.

    ``absorption`` (thermal free--free), ``scattering`` (magnetic Thomson)
    and ``dispersion`` are :class:`~wd_spectra.magnetic.PolarizedOpacity`-
    compatible ``(minus, pi, plus)`` triples in cm^2 g^-1.
    """

    absorption: tuple[FloatArray, FloatArray, FloatArray]
    scattering: tuple[FloatArray, FloatArray, FloatArray]
    dispersion: tuple[FloatArray, FloatArray, FloatArray]


def hydrogen_free_free_mass_absorption_coefficient(
    atmosphere: Atmosphere, wavelength_angstrom: ArrayLike
) -> FloatArray:
    """Zero-field e--p bremsstrahlung exactly as in the ordinary continuum."""

    from .opacity import hydrogen_free_free_gaunt_factor

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    temperature = atmosphere.temperature[np.newaxis, :]
    stimulated = -np.expm1(-PLANCK * frequency / (BOLTZMANN * temperature))
    per_cm = (
        3.692e8
        * temperature**-0.5
        * atmosphere.electron_density[np.newaxis, :]
        * atmosphere.proton_density[np.newaxis, :]
        * frequency**-3
        * stimulated
        * hydrogen_free_free_gaunt_factor(wavelength_cm * 1.0e8, temperature)
    )
    return per_cm / atmosphere.mass_density[np.newaxis, :]


def magnetized_free_electron_manifolds(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    field_ray_cosine: float,
) -> FreeElectronManifolds:
    """Free-electron absorption, scattering and dispersion in a field.

    Cold magneto-ionic (Drude) response of electrons with collision
    frequency ``nu`` and radiation damping ``gamma_rad = 2 e^2 w^2/(3 m c^3)``:
    the circular mode ``q`` (``q=+1`` co-rotating) sees

    ``R_q = (w^2 + G^2) / ((w - q w_c)^2 + G^2)``,   ``G = nu + gamma_rad``,

    times its zero-field value, for both thermal free--free absorption and
    Thomson scattering, and the Kramers--Kronig dispersion of the same
    response.  ``nu`` is taken from the package's zero-field free--free
    opacity itself (``kappa_ff / sigma_T = nu / gamma_rad``), so ``B -> 0``
    recovers the ordinary free--free and Thomson opacities exactly, the
    ``q=+1`` resonance carries the classical integrated strength
    ``4 pi^2 e^2 / (m c)`` per electron, its thermal fraction is
    ``nu/(nu + gamma_rad)``, and far from resonance the Faraday rotation
    falls as ``w_c / w^2``.  The co-rotating resonance is Doppler broadened
    (``sigma = w_c sqrt(kT/m c^2) |cos|``) with a Voigt profile; the other
    modes are far from resonance.  Quantizing-field corrections to the
    free--free Gaunt factor are not included.
    """

    from .opacity import electron_scattering_mass_coefficient

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    field = float(field_strength_megagauss)
    cosine = abs(float(field_ray_cosine))
    free_free = hydrogen_free_free_mass_absorption_coefficient(atmosphere, wavelength)
    thomson = np.broadcast_to(
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :], free_free.shape
    )
    omega = 2.0 * PI * LIGHT_SPEED / (wavelength[:, np.newaxis] * 1.0e-8)
    radiative = 2.0 * ELEMENTARY_CHARGE_ESU**2 * omega**2 / (
        3.0 * ELECTRON_MASS * LIGHT_SPEED**3
    )
    collision = radiative * free_free / np.maximum(thomson, np.finfo(float).tiny)
    damping = collision + radiative
    cyclotron = ELEMENTARY_CHARGE_ESU * field * 1.0e6 / (ELECTRON_MASS * LIGHT_SPEED)
    numerator = omega**2 + damping**2
    total = free_free + thomson
    absorption, scattering, dispersion = [], [], []
    for q in (-1, 0, 1):
        detuning = omega - q * cyclotron
        if q == 1 and field > 0.0:
            doppler = (
                cyclotron
                * np.sqrt(BOLTZMANN * atmosphere.temperature[np.newaxis, :] / (ELECTRON_MASS * LIGHT_SPEED**2))
                * cosine
            ) * np.ones_like(omega)
            profile, phase = _complex_voigt_profile_per_angular_frequency(
                -detuning, doppler, damping
            )
            factor = numerator * PI / damping * profile
            phase_factor = numerator * PI / damping * phase
        else:
            lorentz = detuning**2 + damping**2
            factor = numerator / lorentz
            phase_factor = numerator / damping * (-detuning) / lorentz
        absorption.append(free_free * factor)
        scattering.append(thomson * factor)
        dispersion.append(total * phase_factor)
    return FreeElectronManifolds(tuple(absorption), tuple(scattering), tuple(dispersion))


def isotropic_free_electron_opacity(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    *,
    angular_quadrature_order: int = 12,
) -> tuple[FloatArray, FloatArray]:
    """Angle-averaged Stokes-I free-electron (absorption, scattering).

    Averages ``eta_I`` over ``0 <= |cos| <= 1`` because the Doppler width of
    the co-rotating resonance depends on the angle.
    """

    nodes, weights = np.polynomial.legendre.leggauss(int(angular_quadrature_order))
    cosines = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    absorption = scattering = 0.0
    for cosine, weight in zip(cosines, weights):
        manifolds = magnetized_free_electron_manifolds(
            atmosphere, wavelength_angstrom, field_strength_megagauss, cosine
        )
        c2 = cosine**2
        for store, triple in (("a", manifolds.absorption), ("s", manifolds.scattering)):
            minus, pi_, plus = triple
            eta = 0.5 * pi_ * (1.0 - c2) + 0.25 * (minus + plus) * (1.0 + c2)
            if store == "a":
                absorption = absorption + weight * eta
            else:
                scattering = scattering + weight * eta
    return np.asarray(absorption), np.asarray(scattering)


def _humlicek_complex_probability(argument: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate ``w(z)=exp(-z^2) erfc(-iz)`` without a SciPy dependency.

    This is Humlicek's four-region rational approximation.  Its accuracy is
    ample for the much larger uncertainties in the collision width, while it
    supplies the causal imaginary profile that a pseudo-Voigt approximation
    alone cannot provide.
    """

    z = np.asarray(argument, dtype=np.complex128)
    if np.any(~np.isfinite(z)) or np.any(np.imag(z) < 0.0):
        raise ValueError("complex probability arguments require Im(z)>=0")
    x = np.real(z)
    y = np.imag(z)
    t = y - 1j * x
    s = np.abs(x) + y
    result = np.empty(z.shape, dtype=np.complex128)

    region_1 = s >= 15.0
    current = t[region_1]
    result[region_1] = current * 0.5641896 / (0.5 + current * current)

    region_2 = (~region_1) & (s >= 5.5)
    current = t[region_2]
    result[region_2] = (
        current * (1.410474 + 0.5641896 * current * current)
        / (0.75 + current * current * (3.0 + current * current))
    )

    region_3 = (~region_1) & (~region_2) & (
        y >= 0.195 * np.abs(x) - 0.176
    )
    current = t[region_3]
    result[region_3] = (
        16.4955
        + current
        * (
            20.20933
            + current
            * (11.96482 + current * (3.778987 + current * 0.5642236))
        )
    ) / (
        16.4955
        + current
        * (
            38.82363
            + current
            * (
                39.27121
                + current
                * (21.69274 + current * (6.699398 + current))
            )
        )
    )

    region_4 = ~(region_1 | region_2 | region_3)
    current = t[region_4]
    squared = current * current
    result[region_4] = np.exp(squared) - current * (
        36183.31
        - squared
        * (
            3321.9905
            - squared
            * (
                1540.787
                - squared
                * (
                    219.0313
                    - squared
                    * (
                        35.7668
                        - squared * (1.320522 - squared * 0.56419)
                    )
                )
            )
        )
    ) / (
        32066.6
        - squared
        * (
            24322.8
            - squared
            * (
                9022.23
                - squared
                * (
                    2186.18
                    - squared
                    * (
                        364.219
                        - squared
                        * (61.5704 - squared * (1.84144 - squared))
                    )
                )
            )
        )
    )
    return result


def _complex_voigt_profile_per_angular_frequency(
    offset: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return area-normalized Voigt absorption and its dispersion partner."""

    frequency_offset, sigma, gamma = np.broadcast_arrays(
        np.asarray(offset, dtype=np.float64),
        np.asarray(gaussian_sigma, dtype=np.float64),
        np.asarray(lorentz_hwhm, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(frequency_offset))
        or np.any(~np.isfinite(sigma))
        or np.any(~np.isfinite(gamma))
        or np.any(sigma < 0.0)
        or np.any(gamma <= 0.0)
    ):
        raise ValueError("Voigt widths must be finite with sigma>=0 and gamma>0")

    absorption = np.empty_like(frequency_offset)
    dispersion = np.empty_like(frequency_offset)
    lorentz_limit = sigma <= np.sqrt(np.finfo(np.float64).eps) * gamma
    denominator = frequency_offset**2 + gamma**2
    absorption[lorentz_limit] = (
        gamma[lorentz_limit] / (PI * denominator[lorentz_limit])
    )
    dispersion[lorentz_limit] = (
        frequency_offset[lorentz_limit]
        / (PI * denominator[lorentz_limit])
    )

    finite_doppler = ~lorentz_limit
    if np.any(finite_doppler):
        normalization = np.sqrt(2.0) * sigma[finite_doppler]
        argument = (
            frequency_offset[finite_doppler]
            + 1j * gamma[finite_doppler]
        ) / normalization
        probability = _humlicek_complex_probability(argument)
        profile_normalization = np.sqrt(2.0 * PI) * sigma[finite_doppler]
        absorption[finite_doppler] = (
            np.real(probability) / profile_normalization
        )
        dispersion[finite_doppler] = (
            np.imag(probability) / profile_normalization
        )
    return absorption, dispersion


__all__ = [
    "FreeElectronManifolds",
    "hydrogen_free_free_mass_absorption_coefficient",
    "isotropic_free_electron_opacity",
    "magnetized_free_electron_manifolds",
    "MAGNETIC_ATOMIC_FIELD_MEGAGAUSS",
    "ROHRMANN_2026_DOI",
    "ROHRMANN_2026_ZENODO_DOI",
    "explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient",
    "explicit_rwa_photoionization_structure_continuum",
]
