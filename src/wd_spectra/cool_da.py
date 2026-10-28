"""Published structure diagnostics for cool hydrogen atmospheres.

This module deliberately separates literature-controlled mean-3D structure
experiments from the production one-dimensional atmosphere solver.  It does
not turn a 1D atmosphere into a hydrodynamical calculation; it provides a
reproducible way to test the temperature-structure difference shown by
Tremblay et al. (2013, A&A 552, A13, Fig. 7).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere, _solve_bracketed_log_root
from .eos import hummer_mihalas_hydrogen_lte


FloatArray = NDArray[np.float64]


# Digitized from the vector paths of Tremblay et al. (2013), Fig. 7, which
# shows the <3D> (red) and 1D ML2/alpha=0.8 (black dashed) structures of all
# twelve log(g)=8 CO5BOLD simulations of their Table 2.  Each pair is shifted by
# the same vertical amount, so the difference is unaffected by the offsets; the
# 5997--8032 K rows reproduce the earlier three-model digitization to 2 K.
# Below tau_R ~ 3 the curves are too steep in the figure for a reliable
# vertical difference (and there they mostly reflect the 1D convective
# efficiency), so the table stops at log tau_R = 0.5.  The paper supplies
# log(g)=8 simulations only.  Values are rounded to the nearest kelvin.
TREMBLAY_2013_LOG_ROSSELAND_DEPTH = np.linspace(-4.0, 0.5, 19)
TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K = np.asarray(
    [5_997.0, 7_012.0, 8_032.0, 9_035.0, 9_520.0, 10_018.0,
     10_530.0, 11_004.0, 11_531.0, 12_022.0, 12_505.0, 12_999.0],
    dtype=np.float64,
)
TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K = np.asarray(
    [
        [
            -641, -592, -533, -439, -359, -278, -199, -98, -8, 64,
            71, 13, -2, -27, -38, -42, -38, -40, -35,
        ],
        [
            -1056, -1001, -924, -849, -755, -653, -531, -382, -253, -124,
            7, 125, 181, 81, -7, -68, -79, -91, -79,
        ],
        [
            -1527, -1459, -1358, -1215, -1076, -904, -735, -560, -371, -184,
            -10, 134, 227, 207, -4, -86, -112, -87, -57,
        ],
        [
            -1754, -1611, -1462, -1287, -1111, -917, -725, -501, -297, -89,
            86, 213, 305, 137, -58, -145, -122, -35, 81,
        ],
        [
            -1659, -1474, -1310, -1170, -1007, -830, -637, -425, -234, -59,
            99, 202, 272, 47, -120, -166, -127, 9, 141,
        ],
        [
            -1432, -1282, -1155, -1024, -863, -687, -511, -324, -156, 10,
            131, 219, 213, -23, -162, -199, -117, 46, 225,
        ],
        [
            -1226, -1088, -966, -837, -675, -496, -338, -203, -69, 61,
            160, 226, 174, -28, -172, -218, -139, 37, 233,
        ],
        [
            -773, -741, -703, -646, -540, -419, -301, -185, -81, 16,
            88, 143, 77, -80, -205, -258, -177, 12, 205,
        ],
        [
            -509, -511, -491, -451, -387, -291, -207, -128, -33, 45,
            108, 157, 67, -86, -188, -250, -181, -6, 212,
        ],
        [
            -510, -482, -438, -364, -301, -234, -165, -89, -15, 59,
            93, 140, 20, -61, -170, -218, -146, 35, 260,
        ],
        [
            -451, -399, -336, -263, -189, -118, -43, 5, 64, 123,
            186, 179, 101, 3, -69, -126, -76, 95, 286,
        ],
        [
            -283, -250, -211, -145, -79, -17, 32, 73, 115, 166,
            212, 210, 157, 103, 44, 38, 92, 226, 154,
        ],
    ],
    dtype=np.float64,
)
# Tremblay et al. (2013c, 2015) attribute the <3D> deviation at
# tau_R < 1e-2 to convective-overshoot cooling.  The cool SPY sample in the
# OpenWD validation favors it below about 9000 K, but at 9--11.5 kK it makes the
# Halpha and Hbeta cores too deep, as Tremblay et al. (2013c) also found.
# "photospheric" keeps the 1D temperatures above this depth.
TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH = 1.0e-2
_UPPER_LAYER_TRANSITION_DEX = 0.15


def tremblay_2013_mean_3d_temperature_difference(
    effective_temperature: float,
    rosseland_optical_depth: ArrayLike,
    *,
    layers: str = "all",
) -> FloatArray:
    """Return the published mean-3D minus 1D temperature difference.

    The difference is interpolated linearly in Teff across the twelve
    log(g)=8 simulations of Tremblay et al. (2013, Fig. 7), 5997--12999 K.
    ``layers="photospheric"`` smoothly removes it above
    ``TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH`` (0.15 dex logistic), keeping
    the 1D upper atmosphere.  It is a digitized structure differential, not a
    substitute for a CO5BOLD structure grid.
    """

    teff = float(effective_temperature)
    tau = np.asarray(rosseland_optical_depth, dtype=np.float64)
    if layers not in ("all", "photospheric"):
        raise ValueError("layers must be 'all' or 'photospheric'")
    if not np.isfinite(teff):
        raise ValueError("effective_temperature must be finite")
    if (
        teff < TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[0]
        or teff > TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[-1]
    ):
        raise ValueError(
            "the digitized Tremblay et al. (2013) cool-DA differential "
            "covers 5997--12999 K"
        )
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("rosseland_optical_depth must be finite and positive")

    log_tau = np.log10(tau)
    depth_curves = np.vstack(
        [
            np.interp(
                log_tau,
                TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
                row,
                left=row[0],
                right=row[-1],
            )
            for row in TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K
        ]
    )
    upper = int(
        np.searchsorted(
            TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K, teff, side="right"
        )
    )
    upper = min(max(upper, 1), TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K.size - 1)
    lower = upper - 1
    fraction = (
        (teff - TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[lower])
        / (
            TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[upper]
            - TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[lower]
        )
    )
    difference = (
        (1.0 - fraction) * depth_curves[lower] + fraction * depth_curves[upper]
    )
    if layers == "photospheric":
        upper_layer_weight = 1.0 / (
            1.0
            + np.exp(
                (log_tau - np.log10(TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH))
                / _UPPER_LAYER_TRANSITION_DEX
            )
        )
        difference = difference * (1.0 - upper_layer_weight)
    return np.asarray(difference, dtype=np.float64)


def atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
    atmosphere: Atmosphere,
    *,
    neutral_radius_scale: float | None = None,
    maximum_logg_offset: float = 0.4,
    layers: str = "all",
) -> Atmosphere:
    """Apply the Fig. 7 temperature differential and recompute the H EOS.

    Gas pressure, column mass, and the original Rosseland coordinate remain
    fixed.  This is the same kind of differential experiment used in the
    paper, but it lacks the published mean-3D pressure structure and is not
    guaranteed to satisfy radiative equilibrium.  The guard against large
    gravity offsets prevents silent use far from the sole published log(g)=8
    sequence.
    """

    if atmosphere.hydrogen_lte_state is None:
        raise ValueError("a pure-hydrogen LTE atmosphere is required")
    if abs(atmosphere.logg - 8.0) > float(maximum_logg_offset):
        raise ValueError(
            "the Tremblay et al. (2013) structure differential is only "
            "published at log(g)=8"
        )
    difference = tremblay_2013_mean_3d_temperature_difference(
        atmosphere.effective_temperature,
        atmosphere.rosseland_optical_depth,
        layers=layers,
    )
    temperature = atmosphere.temperature + difference
    if np.any(temperature <= 0.0):
        raise RuntimeError("the mean-3D temperature differential is unphysical")
    original = atmosphere.hydrogen_lte_state
    radius_scale = (
        original.neutral_radius_scale
        if neutral_radius_scale is None
        else float(neutral_radius_scale)
    )
    maximum_level = (
        original.level_population_density.shape[-1]
        if original.level_population_density is not None
        else 40
    )
    state = hummer_mihalas_hydrogen_lte(
        temperature,
        atmosphere.gas_pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=radius_scale,
        correlated_microfields=(original.microfield_model == "qmhd"),
        include_molecules=(original.molecular_hydrogen_density is not None),
        include_negative_hydrogen=(
            original.negative_hydrogen_density is not None
        ),
        trihydrogen_ion_partition_model=(
            original.trihydrogen_ion_partition_model
        ),
    )
    return replace(
        atmosphere,
        temperature=temperature,
        mass_density=state.mass_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        hydrogen_lte_state=state,
        metadata={
            **atmosphere.metadata,
            "mean_3d_temperature_differential": (
                "Tremblay et al. 2013 Fig. 7 vector digitization"
            ),
            "mean_3d_temperature_differential_reference_logg": 8.0,
            "mean_3d_temperature_differential_layers": layers,
            "mean_3d_temperature_differential_preserves_pressure": True,
            "mean_3d_temperature_differential_is_equilibrium_model": False,
            "mean_3d_temperature_differential_neutral_radius_scale": (
                radius_scale
            ),
        },
    )


def hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
    atmosphere: Atmosphere,
    *,
    h2_h2_cia_table=None,
    maximum_logg_offset: float = 0.4,
    layers: str = "all",
) -> Atmosphere:
    """Apply the published T difference and reintegrate hydrostatic balance.

    Unlike :func:`atmosphere_with_tremblay_2013_mean_3d_temperature_difference`,
    this variant treats the published Rosseland-depth coordinate as fixed and
    solves ``dP/dtau_R = g/kappa_R`` with the corrected temperature and the
    implemented continuum Rosseland opacity.  Cooler upper layers have less
    H- opacity, so a given tau_R lies under more mass: the neutral-H density
    rises (x2.6 at tau_R=1e-2 and x9 above 1e-4 at 7000 K), as in the paper's
    <3D> structures.  ``layers`` is as in
    :func:`tremblay_2013_mean_3d_temperature_difference`.  The paper's
    turbulent pressure is not included.
    """

    from .opacity import rosseland_mean_hydrogen_continuum_opacity

    fixed_pressure = (
        atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere,
            maximum_logg_offset=maximum_logg_offset,
            layers=layers,
        )
    )
    temperature = fixed_pressure.temperature
    tau = atmosphere.rosseland_optical_depth
    gravity = atmosphere.gravity
    original = atmosphere.hydrogen_lte_state
    assert original is not None
    correlated_microfields = original.microfield_model == "qmhd"
    include_molecules = original.molecular_hydrogen_density is not None
    maximum_level = (
        original.level_population_density.shape[-1]
        if original.level_population_density is not None
        else 40
    )

    def opacity_at(temp: float, pressure: float) -> float:
        state = hummer_mihalas_hydrogen_lte(
            np.asarray([temp]),
            np.asarray([pressure]),
            maximum_level=maximum_level,
            neutral_radius_scale=original.neutral_radius_scale,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=(
                original.negative_hydrogen_density is not None
            ),
            trihydrogen_ion_partition_model=(
                original.trihydrogen_ion_partition_model
            ),
        )
        point = Atmosphere(
            effective_temperature=atmosphere.effective_temperature,
            logg=atmosphere.logg,
            rosseland_optical_depth=np.asarray([1.0]),
            column_mass=np.asarray([pressure / gravity]),
            temperature=np.asarray([temp]),
            gas_pressure=np.asarray([pressure]),
            mass_density=np.atleast_1d(state.mass_density),
            neutral_h_density=np.atleast_1d(state.neutral_h_density),
            proton_density=np.atleast_1d(state.proton_density),
            electron_density=np.atleast_1d(state.electron_density),
            metadata={},
            hydrogen_lte_state=state,
        )
        return float(
            rosseland_mean_hydrogen_continuum_opacity(
                point,
                h2_h2_cia_table=h2_h2_cia_table,
            )[0]
        )

    def pressure_increment(
        previous_pressure: float,
        delta_tau: float,
        midpoint_temperature: float,
    ) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            midpoint_pressure = previous_pressure + 0.5 * increment
            return (
                log_increment
                + np.log(opacity_at(midpoint_temperature, midpoint_pressure))
                - np.log(target)
            )

        lower = np.log(max(target / 1.0e12, 1.0e-20))
        upper = np.log(max(target / 1.0e-12, 1.0e-19))
        return float(
            np.exp(_solve_bracketed_log_root(residual, lower, upper))
        )

    pressure = np.empty_like(tau)
    pressure[0] = pressure_increment(0.0, float(tau[0]), float(temperature[0]))
    for index in range(1, tau.size):
        pressure[index] = pressure[index - 1] + pressure_increment(
            float(pressure[index - 1]),
            float(tau[index] - tau[index - 1]),
            float(0.5 * (temperature[index] + temperature[index - 1])),
        )
    state = hummer_mihalas_hydrogen_lte(
        temperature,
        pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=original.neutral_radius_scale,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=(
            original.negative_hydrogen_density is not None
        ),
        trihydrogen_ion_partition_model=(
            original.trihydrogen_ion_partition_model
        ),
    )
    return replace(
        fixed_pressure,
        gas_pressure=pressure,
        column_mass=pressure / gravity,
        mass_density=state.mass_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        hydrogen_lte_state=state,
        metadata={
            **fixed_pressure.metadata,
            "mean_3d_temperature_differential_preserves_pressure": False,
            "mean_3d_temperature_differential_hydrostatic_reintegration": (
                "dP/dtau_R=g/kappa_R; continuum Rosseland mean"
            ),
            "mean_3d_temperature_differential_turbulent_pressure": False,
        },
    )


__all__ = [
    "TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K",
    "TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K",
    "TREMBLAY_2013_LOG_ROSSELAND_DEPTH",
    "TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH",
    "atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "tremblay_2013_mean_3d_temperature_difference",
]
