"""Local mixing-length convection for one-dimensional stellar atmospheres."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable

from .atmosphere import Atmosphere
from .constants import STEFAN_BOLTZMANN
from .eos import (
    hummer_mihalas_helium_thermodynamics,
    hummer_mihalas_hydrogen_helium_thermodynamics,
    hummer_mihalas_hydrogen_thermodynamics,
)


FloatArray = NDArray[np.float64]
ML2CoefficientFunction = Callable[
    [Atmosphere, ArrayLike, ArrayLike, ArrayLike, ArrayLike, float],
    tuple[FloatArray, FloatArray, FloatArray],
]


def _ml2_contrast_and_root(
    excess: FloatArray, loss: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Positive ML2 quadratic root without subtracting nearly equal numbers.

    x² + loss*x = excess. The direct expression sqrt(loss²/4+excess)
    - loss/2 loses the entire convective flux when excess << loss².
    Rationalization preserves that same root; hypot also avoids loss² overflow.
    Callers supply nonnegative excess, including zero on the stable branch.
    """
    root = np.hypot(0.5 * loss, np.sqrt(excess))
    contrast = excess / np.maximum(root + 0.5 * loss, np.finfo(float).tiny)
    return contrast, root


def _ml2_local_coefficients_from_thermodynamics(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    specific_heat_constant_pressure: ArrayLike,
    density_temperature_derivative: ArrayLike,
    adiabatic_temperature_gradient: ArrayLike,
    mixing_length_alpha: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return local ML2 coefficients for an externally supplied EOS.

    The hydrogen and helium paths obtain these three thermodynamic derivatives
    from the Hummer--Mihalas EOS.  A bulk C/O atmosphere needs the same ML2
    closure but has a different EOS, so keeping the algebra here avoids a
    composition-specific copy of the convection equations.
    """

    if not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0:
        raise ValueError("mixing_length_alpha must be finite and positive")
    opacity = np.broadcast_to(
        np.asarray(rosseland_opacity, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    specific_heat = np.broadcast_to(
        np.asarray(specific_heat_constant_pressure, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    expansion = np.broadcast_to(
        np.asarray(density_temperature_derivative, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    adiabatic_gradient = np.broadcast_to(
        np.asarray(adiabatic_temperature_gradient, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    for name, values in (
        ("rosseland_opacity", opacity),
        ("specific_heat_constant_pressure", specific_heat),
        ("density_temperature_derivative", expansion),
        ("adiabatic_temperature_gradient", adiabatic_gradient),
    ):
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must contain finite positive values")

    temperature = atmosphere.temperature
    pressure = atmosphere.gas_pressure
    density = atmosphere.mass_density
    gravity = atmosphere.gravity
    pressure_scale_height = pressure / (density * gravity)
    mixing_length = mixing_length_alpha * pressure_scale_height
    cell_optical_depth = opacity * mixing_length * density
    thin_cell_factor = (
        8.0 * cell_optical_depth**2
        / (1.0 + 8.0 * cell_optical_depth**2 / 16.0)
    )
    radiative_loss = (
        STEFAN_BOLTZMANN
        * temperature**3
        * thin_cell_factor
        / (
            density
            * mixing_length
            * cell_optical_depth
            * specific_heat
        )
        * np.sqrt(pressure_scale_height / (gravity * expansion))
    )
    flux_coefficient = (
        2.0
        * specific_heat
        * density
        * temperature
        * mixing_length**2
        / pressure_scale_height
        * np.sqrt(gravity * expansion / pressure_scale_height)
    )
    return adiabatic_gradient, radiative_loss, flux_coefficient


def _ml2_local_coefficients(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    mixing_length_alpha: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the adiabatic gradient and the local ML2 B and C factors."""

    if not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0:
        raise ValueError("mixing_length_alpha must be finite and positive")
    opacity = np.broadcast_to(
        np.asarray(rosseland_opacity, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    if np.any(~np.isfinite(opacity)) or np.any(opacity <= 0.0):
        raise ValueError("rosseland_opacity must contain finite positive values")

    temperature = atmosphere.temperature
    pressure = atmosphere.gas_pressure
    if (
        atmosphere.hydrogen_lte_state is not None
        and atmosphere.helium_lte_state is not None
    ):
        hydrogen = atmosphere.hydrogen_lte_state
        helium = atmosphere.helium_lte_state
        log_hydrogen_to_helium = np.log10(
            hydrogen.hydrogen_nuclei_density
            / np.maximum(
                helium.helium_nuclei_density, np.finfo(np.float64).tiny
            )
        )
        thermodynamics = hummer_mihalas_hydrogen_helium_thermodynamics(
            temperature,
            pressure,
            log_hydrogen_to_helium,
            hydrogen_neutral_radius_scale=hydrogen.neutral_radius_scale,
            helium_neutral_radius_scale=helium.neutral_radius_scale,
            correlated_microfields=(
                hydrogen.microfield_model == "qmhd"
                and helium.microfield_model == "qmhd"
            ),
        )
    elif atmosphere.helium_lte_state is not None:
        thermodynamics = hummer_mihalas_helium_thermodynamics(
            temperature,
            pressure,
            neutral_radius_scale=atmosphere.helium_lte_state.neutral_radius_scale,
            correlated_microfields=(
                atmosphere.helium_lte_state.microfield_model == "qmhd"
            ),
        )
    else:
        thermodynamics = hummer_mihalas_hydrogen_thermodynamics(
            temperature,
            pressure,
            correlated_microfields=(
                atmosphere.hydrogen_lte_state is not None
                and atmosphere.hydrogen_lte_state.microfield_model == "qmhd"
            ),
            include_molecules=(
                atmosphere.hydrogen_lte_state is not None
                and atmosphere.hydrogen_lte_state.molecular_hydrogen_density
                is not None
            ),
        )
    return _ml2_local_coefficients_from_thermodynamics(
        atmosphere,
        opacity,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
        mixing_length_alpha,
    )


def ml2_convective_flux_for_gradient_from_thermodynamics(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    temperature_gradient: ArrayLike,
    specific_heat_constant_pressure: ArrayLike,
    density_temperature_derivative: ArrayLike,
    adiabatic_temperature_gradient: ArrayLike,
    *,
    mixing_length_alpha: float = 1.25,
    coefficient_function: ML2CoefficientFunction | None = None,
) -> FloatArray:
    """Return ML2 flux for a supplied EOS and temperature gradient."""

    gradient = np.asarray(temperature_gradient, dtype=np.float64)
    if gradient.shape[-1:] != atmosphere.temperature.shape:
        raise ValueError(
            "temperature_gradient must end with the atmosphere depth dimension"
        )
    if np.any(~np.isfinite(gradient)):
        raise ValueError("temperature_gradient must contain finite values")
    if coefficient_function is None:
        coefficient_function = _ml2_local_coefficients_from_thermodynamics
    adiabatic_gradient, radiative_loss, flux_coefficient = (
        coefficient_function(
            atmosphere,
            rosseland_opacity,
            specific_heat_constant_pressure,
            density_temperature_derivative,
            adiabatic_temperature_gradient,
            mixing_length_alpha,
        )
    )
    superadiabatic_excess = np.maximum(gradient - adiabatic_gradient, 0.0)
    element_environment_difference, _ = _ml2_contrast_and_root(
        superadiabatic_excess, radiative_loss,
    )
    flux = flux_coefficient * element_environment_difference**3
    return np.where(superadiabatic_excess > 0.0, flux, 0.0)


def _ml2_flux_coefficient_response(
    temperature_gradient: FloatArray,
    coefficients: tuple[FloatArray, FloatArray, FloatArray],
    coefficient_response: tuple[FloatArray, FloatArray, FloatArray],
) -> FloatArray:
    """Differentiate ML2 material coefficients at fixed temperature gradient.

    If x is the element/environment temperature contrast, ML2 obeys
    x**2 + B*x = nabla - nabla_ad and F_conv = C*x**3. Differentiate
    these identities rather than differencing the flux across the convection
    boundary. Responses have shape (interface, independent variable).
    """

    adiabatic, loss, coefficient = coefficients
    d_adiabatic, d_loss, d_coefficient = coefficient_response
    excess = np.maximum(temperature_gradient - adiabatic, 0.0)
    contrast, root = _ml2_contrast_and_root(excess, loss)
    gradient_response = 3.0 * coefficient * contrast**2 / np.maximum(
        2.0 * root, np.finfo(float).tiny
    )
    return (
        contrast[:, np.newaxis]**3 * d_coefficient
        - gradient_response[:, np.newaxis]
        * (d_adiabatic + contrast[:, np.newaxis] * d_loss)
    )


def ml2_convective_flux_gradient_derivative_from_thermodynamics(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    temperature_gradient: ArrayLike,
    specific_heat_constant_pressure: ArrayLike,
    density_temperature_derivative: ArrayLike,
    adiabatic_temperature_gradient: ArrayLike,
    *,
    mixing_length_alpha: float = 1.25,
    coefficient_function: ML2CoefficientFunction | None = None,
) -> FloatArray:
    r"""Return the analytic ML2 derivative ``dF_conv/dnabla``.

    The finite-difference derivative is especially inaccurate in efficient
    convection, where the superadiabatic excess can be much smaller than a
    practical differencing step.  Differentiating the local ML2 cubic avoids
    crossing the convection boundary and supplies the response needed by the
    atmosphere Newton system.
    """

    gradient = np.asarray(temperature_gradient, dtype=np.float64)
    if gradient.shape[-1:] != atmosphere.temperature.shape:
        raise ValueError(
            "temperature_gradient must end with the atmosphere depth dimension"
        )
    if np.any(~np.isfinite(gradient)):
        raise ValueError("temperature_gradient must contain finite values")
    if coefficient_function is None:
        coefficient_function = _ml2_local_coefficients_from_thermodynamics
    adiabatic_gradient, radiative_loss, flux_coefficient = (
        coefficient_function(
            atmosphere,
            rosseland_opacity,
            specific_heat_constant_pressure,
            density_temperature_derivative,
            adiabatic_temperature_gradient,
            mixing_length_alpha,
        )
    )
    superadiabatic_excess = np.maximum(
        gradient - adiabatic_gradient, 0.0
    )
    element_environment_difference, square_root = _ml2_contrast_and_root(
        superadiabatic_excess, radiative_loss,
    )
    derivative = (
        3.0
        * flux_coefficient
        * element_environment_difference**2
        / np.maximum(2.0 * square_root, np.finfo(np.float64).tiny)
    )
    return np.where(superadiabatic_excess > 0.0, derivative, 0.0)


def ml2_temperature_gradient_for_flux_from_thermodynamics(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    convective_flux: ArrayLike,
    specific_heat_constant_pressure: ArrayLike,
    density_temperature_derivative: ArrayLike,
    adiabatic_temperature_gradient: ArrayLike,
    *,
    mixing_length_alpha: float = 1.25,
    coefficient_function: ML2CoefficientFunction | None = None,
) -> FloatArray:
    """Invert the local ML2 closure for an externally supplied EOS."""

    requested_flux = np.broadcast_to(
        np.asarray(convective_flux, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    if np.any(~np.isfinite(requested_flux)) or np.any(requested_flux < 0.0):
        raise ValueError("convective_flux must contain finite non-negative values")
    if coefficient_function is None:
        coefficient_function = _ml2_local_coefficients_from_thermodynamics
    adiabatic_gradient, radiative_loss, flux_coefficient = (
        coefficient_function(
            atmosphere,
            rosseland_opacity,
            specific_heat_constant_pressure,
            density_temperature_derivative,
            adiabatic_temperature_gradient,
            mixing_length_alpha,
        )
    )
    element_environment_difference = np.cbrt(
        requested_flux / flux_coefficient
    )
    return (
        adiabatic_gradient
        + radiative_loss * element_environment_difference
        + element_environment_difference**2
    )


def ml2_temperature_gradient_for_total_flux_from_thermodynamics(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    total_flux: ArrayLike,
    specific_heat_constant_pressure: ArrayLike,
    density_temperature_derivative: ArrayLike,
    adiabatic_temperature_gradient: ArrayLike,
    *,
    mixing_length_alpha: float = 1.25,
    bisection_iterations: int = 48,
    radiative_flux_coefficient: ArrayLike | None = None,
    coefficient_function: ML2CoefficientFunction | None = None,
) -> FloatArray:
    r"""Solve ``F_rad(nabla) + F_conv(nabla) = F_total`` locally.

    In optically thick layers the diffusion flux is linear in the logarithmic
    temperature gradient,

    ``F_rad = 16 sigma g T**4 nabla / (3 kappa_R P)``.

    Combining that term with the monotonic ML2 flux avoids assigning the
    entire *current* radiative-flux deficit to convection.  That shortcut
    steepens the gradient and increases the radiative flux at the same time,
    so it systematically over-carries the requested total flux.
    """

    if bisection_iterations < 8:
        raise ValueError("bisection_iterations must be at least eight")
    requested_flux = np.broadcast_to(
        np.asarray(total_flux, dtype=np.float64), atmosphere.temperature.shape
    )
    opacity = np.broadcast_to(
        np.asarray(rosseland_opacity, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    if (
        np.any(~np.isfinite(requested_flux))
        or np.any(requested_flux < 0.0)
        or np.any(~np.isfinite(opacity))
        or np.any(opacity <= 0.0)
    ):
        raise ValueError("total flux and Rosseland opacity must be physical")

    if radiative_flux_coefficient is None:
        radiative_coefficient = (
            16.0
            * STEFAN_BOLTZMANN
            * atmosphere.gravity
            * atmosphere.temperature**4
            / (3.0 * opacity * atmosphere.gas_pressure)
        )
    else:
        radiative_coefficient = np.broadcast_to(
            np.asarray(radiative_flux_coefficient, dtype=np.float64),
            atmosphere.temperature.shape,
        )
        if np.any(~np.isfinite(radiative_coefficient)) or np.any(
            radiative_coefficient <= 0.0
        ):
            raise ValueError(
                "radiative_flux_coefficient must contain finite positive values"
            )
    pure_radiative_gradient = requested_flux / radiative_coefficient
    pure_convective_gradient = (
        ml2_temperature_gradient_for_flux_from_thermodynamics(
            atmosphere,
            opacity,
            requested_flux,
            specific_heat_constant_pressure,
            density_temperature_derivative,
            adiabatic_temperature_gradient,
            mixing_length_alpha=mixing_length_alpha,
            coefficient_function=coefficient_function,
        )
    )
    lower = np.zeros_like(requested_flux)
    # At either pure-flux solution the sum is at least the target, so their
    # minimum safely brackets the coupled monotonic root.
    upper = np.minimum(pure_radiative_gradient, pure_convective_gradient)
    for _ in range(bisection_iterations):
        middle = 0.5 * (lower + upper)
        convective_flux = (
            ml2_convective_flux_for_gradient_from_thermodynamics(
                atmosphere,
                opacity,
                middle,
                specific_heat_constant_pressure,
                density_temperature_derivative,
                adiabatic_temperature_gradient,
                mixing_length_alpha=mixing_length_alpha,
                coefficient_function=coefficient_function,
            )
        )
        carried_flux = radiative_coefficient * middle + convective_flux
        below = carried_flux < requested_flux
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return 0.5 * (lower + upper)


def ml2_convective_flux(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    *,
    mixing_length_alpha: float = 0.7,
) -> FloatArray:
    r"""Return the local ML2 convective flux in ``erg cm^-2 s^-1``.

    This implements equations (1)--(5) of Bergeron, Wesemael & Fontaine
    (1992), including their optically thin-cell correction.  ML2 has
    ``a=1``, ``b=2``, and ``c=16``; the Koester DA grid currently uses
    :math:`\ell/H_P=0.7`.
    """

    stored_gradient = atmosphere.metadata.get(
        "nonlocal_log_temperature_pressure_gradient"
    )
    if stored_gradient is None:
        actual_gradient = np.gradient(
            np.log(atmosphere.temperature),
            np.log(atmosphere.gas_pressure),
            edge_order=2,
        )
    else:
        actual_gradient = np.broadcast_to(
            np.asarray(stored_gradient, dtype=np.float64),
            atmosphere.temperature.shape,
        )
    return ml2_convective_flux_for_gradient(
        atmosphere,
        rosseland_opacity,
        actual_gradient,
        mixing_length_alpha=mixing_length_alpha,
    )


def ml2_convective_flux_for_gradient(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    temperature_gradient: ArrayLike,
    *,
    mixing_length_alpha: float = 0.7,
) -> FloatArray:
    """Return ML2 flux for supplied ``d ln(T) / d ln(P)`` values.

    The leading dimensions of ``temperature_gradient`` are broadcast over
    the atmosphere depth dimension.  This supports a cheap local line search
    over several proposed gradient corrections without repeating the EOS or
    opacity calculation for every trial.
    """

    gradient = np.asarray(temperature_gradient, dtype=np.float64)
    if gradient.shape[-1:] != atmosphere.temperature.shape:
        raise ValueError(
            "temperature_gradient must end with the atmosphere depth dimension"
        )
    if np.any(~np.isfinite(gradient)):
        raise ValueError("temperature_gradient must contain finite values")
    adiabatic_gradient, radiative_loss, flux_coefficient = (
        _ml2_local_coefficients(
            atmosphere, rosseland_opacity, mixing_length_alpha
        )
    )
    superadiabatic_excess = np.maximum(
        gradient - adiabatic_gradient, 0.0
    )

    element_environment_difference, _ = _ml2_contrast_and_root(
        superadiabatic_excess, radiative_loss,
    )
    flux = flux_coefficient * element_environment_difference**3
    return np.where(superadiabatic_excess > 0.0, flux, 0.0)


def ml2_temperature_gradient_for_flux(
    atmosphere: Atmosphere,
    rosseland_opacity: ArrayLike,
    convective_flux: ArrayLike,
    *,
    mixing_length_alpha: float = 0.7,
) -> FloatArray:
    r"""Invert the local ML2 equations for a requested convective flux.

    If :math:`y=(\nabla-\nabla')^{1/2}`, equations (1)--(3) give
    ``F_conv = C y**3`` and
    ``nabla = nabla_ad + B*y + y**2``.  This analytic inverse is useful for
    stable flux correction in a full radiative-transfer atmosphere.
    """

    requested_flux = np.broadcast_to(
        np.asarray(convective_flux, dtype=np.float64),
        atmosphere.temperature.shape,
    )
    if np.any(~np.isfinite(requested_flux)) or np.any(requested_flux < 0.0):
        raise ValueError("convective_flux must contain finite non-negative values")
    adiabatic_gradient, radiative_loss, flux_coefficient = (
        _ml2_local_coefficients(
            atmosphere, rosseland_opacity, mixing_length_alpha
        )
    )
    element_environment_difference = np.cbrt(
        requested_flux / flux_coefficient
    )
    return (
        adiabatic_gradient
        + radiative_loss * element_environment_difference
        + element_environment_difference**2
    )
