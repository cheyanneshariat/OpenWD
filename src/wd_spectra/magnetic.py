"""Magnetic hydrogen (DAH) line opacity, surface geometry and polarized transfer.

Balmer opacity is represented by three Zeeman polarization manifolds
(``Delta m = -1, 0, +1``).  Each manifold array is the opacity the line would
have if all its oscillator strength were in that polarization, so an
unpolarized isotropic radiation field sees their arithmetic mean and a ray at
angle ``psi`` to the field sees

``eta_I = kappa_pi sin^2(psi)/2 + (kappa_- + kappa_+)(1 + cos^2 psi)/4``,
``eta_Q = [kappa_pi/2 - (kappa_- + kappa_+)/4] sin^2 psi``,
``eta_V = (kappa_+ - kappa_-) cos(psi) / 2``,

with the magneto-optical ``rho_Q, rho_V`` built identically from the
Kramers--Kronig (Hilbert) partners of the manifolds.  ``Delta m = +1`` is the
blue (``q = +1``) component throughout, for lines, the RWA continuum and the
cyclotron resonance, and every dispersion profile is positive on the
low-frequency side, as in Landi Degl'Innocenti & Landolfi (2004).

Two atomic regimes share this representation.

* ``B <= 1 MG`` everywhere on the visible disk: the complete zero-field Balmer
  opacity (Tremblay--Bergeron Stark profiles, neutral broadening, HM/Q-MHD
  populations) is translated in frequency by the normal-triplet displacement
  ``e B / (4 pi m_e c)``.  ``B = 0`` is exactly the DA opacity.
* stronger fields: Halpha--H12 components, their wavelengths, relative
  strengths and field-dependent total strength come from the public
  Schimeczek--Wunner H2db calculation, with field-dependent lower-substate
  Boltzmann factors.  Each component carries the zero-field unified profile of
  its parent line (no general simultaneous Stark--Zeeman theory is
  available).  H13 and higher are omitted: already at ~1 MG their l-mixed,
  field-dominated upper levels no longer form discrete zero-field lines, and
  H2db has no data for them.  The dissolved-level pseudo-continuum is kept.

The emergent flux is a projected-area sum of specific intensities over
surface cells, each with its own field modulus, field--ray angle and limb
cosine.  Coherent scattering is solved exactly for the angle-averaged opacity
of each cell's atmosphere and enters every ray as an unpolarized source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere
from .constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI, PLANCK
from .magnetic_atomic import H2dbEnergyDatabase, H2dbTransitionDatabase
from .magnetic_continuum import (
    explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient,
    hydrogen_free_free_mass_absorption_coefficient,
    isotropic_free_electron_opacity,
    magnetized_free_electron_manifolds,
)
from .molecules import H2H2CollisionInducedAbsorptionTable
from .opacity import BALMER_LINES, balmer_mass_absorption_coefficient
from .spectrum import Spectrum


FloatArray = NDArray[np.float64]

# Normal Zeeman frequency displacement, e B / (4 pi m_e c), in Hz/G.
LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS = (
    ELEMENTARY_CHARGE_ESU / (4.0 * PI * ELECTRON_MASS * LIGHT_SPEED)
)
# Above this local field the quadratic Zeeman effect is no longer small for
# the upper Balmer levels, and the H2db atom is used.
WEAK_FIELD_MAXIMUM_MEGAGAUSS = 1.0
MAXIMUM_H2DB_BALMER_UPPER_LEVEL = 12
DAH_WAVELENGTH_RANGE_ANGSTROM = (3_400.0, 8_000.0)
# Magneto-optical profiles are nonlocal in frequency; line manifolds are built
# on this range and continuum manifolds on the wider one before transforming.
_CONTINUUM_DISPERSION_RANGE_ANGSTROM = (100.0, 500_000.0)
RWA_MAXIMUM_LEVEL = 8
# The isotropic cyclotron profile is a superposition of Gaussians whose widths
# scale with |cos(field, ray)|; enough nodes make that average smooth.
CYCLOTRON_ANGULAR_NODES = 12

PolarizedTransfer = Literal["full-stokes-iquv", "scalar-stokes-i"]


@dataclass(frozen=True)
class LinearZeemanTriplet:
    """Vacuum component wavelengths for a normal Zeeman triplet."""

    sigma_red_angstrom: float
    pi_angstrom: float
    sigma_blue_angstrom: float


def linear_zeeman_triplet(
    rest_wavelength_angstrom: float,
    field_strength_megagauss: float,
) -> LinearZeemanTriplet:
    """Return frequency-shifted normal-triplet wavelengths (``B <= 1 MG``)."""

    field = _validate_field(field_strength_megagauss)
    if field > WEAK_FIELD_MAXIMUM_MEGAGAUSS:
        raise ValueError(
            f"the normal triplet is limited to B <= {WEAK_FIELD_MAXIMUM_MEGAGAUSS:g} MG"
        )
    wavelength = float(rest_wavelength_angstrom)
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("rest_wavelength_angstrom must be finite and positive")
    center = LIGHT_SPEED / (wavelength * 1.0e-8)
    shift = LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS * field * 1.0e6
    return LinearZeemanTriplet(
        LIGHT_SPEED / (center - shift) / 1.0e-8,
        wavelength,
        LIGHT_SPEED / (center + shift) / 1.0e-8,
    )


def _validate_field(field_strength_megagauss: float) -> float:
    field = float(field_strength_megagauss)
    if not np.isfinite(field) or field < 0.0:
        raise ValueError("field_strength_megagauss must be finite and nonnegative")
    return field


def _validate_grid(wavelength_angstrom: ArrayLike) -> FloatArray:
    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError(
            "wavelength_angstrom must be a finite, positive, increasing 1D array"
        )
    return wavelength


# ---------------------------------------------------------------------------
# Polarization manifolds


@dataclass(frozen=True)
class PolarizedOpacity:
    """Opacity of the ``Delta m = -1, 0, +1`` manifolds in cm^2 g^-1.

    Arrays have shape ``(wavelength, depth)``.  Their mean is the opacity seen
    by unpolarized isotropic radiation.
    """

    minus: FloatArray
    pi: FloatArray
    plus: FloatArray

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "PolarizedOpacity":
        return cls(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    @classmethod
    def unpolarized(cls, opacity: FloatArray) -> "PolarizedOpacity":
        return cls(opacity, opacity, opacity)

    def isotropic(self) -> FloatArray:
        return (self.minus + self.pi + self.plus) / 3.0

    def __add__(self, other: "PolarizedOpacity") -> "PolarizedOpacity":
        return PolarizedOpacity(
            self.minus + other.minus, self.pi + other.pi, self.plus + other.plus
        )

    def scaled(self, factor: FloatArray) -> "PolarizedOpacity":
        return PolarizedOpacity(
            self.minus * factor, self.pi * factor, self.plus * factor
        )

    def subset(self, indices: NDArray[np.int64]) -> "PolarizedOpacity":
        return PolarizedOpacity(
            self.minus[indices], self.pi[indices], self.plus[indices]
        )

    def stokes(self, cosine: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(eta_I, eta_Q, eta_V)`` for a ray--field angle cosine."""

        cosine_squared = float(cosine) ** 2
        sine_squared = 1.0 - cosine_squared
        sigma = self.minus + self.plus
        eta_i = 0.5 * self.pi * sine_squared + 0.25 * sigma * (1.0 + cosine_squared)
        eta_q = (0.5 * self.pi - 0.25 * sigma) * sine_squared
        eta_v = 0.5 * (self.plus - self.minus) * float(cosine)
        return eta_i, eta_q, eta_v

    def hilbert(self, wavelength_angstrom: FloatArray) -> "PolarizedOpacity":
        """Return the magneto-optical partners of the three manifolds."""

        return PolarizedOpacity(
            frequency_hilbert_dispersion(wavelength_angstrom, self.minus),
            frequency_hilbert_dispersion(wavelength_angstrom, self.pi),
            frequency_hilbert_dispersion(wavelength_angstrom, self.plus),
        )


def frequency_hilbert_dispersion(
    wavelength_angstrom: FloatArray,
    absorption_profile: FloatArray,
    *,
    chunk: int = 512,
) -> FloatArray:
    """Return the Kramers--Kronig partner of a sampled absorption profile.

    The profile is taken as piecewise linear in frequency between the given
    (non-uniform) nodes and zero outside them, for which the principal-value
    Hilbert integral has the closed form
    ``H(x) = [sum_j c_j(x) ln|x - y_j| - (f_last - f_first)] / pi``, with
    ``c_j`` the jump of the local linear interpolant at node ``j``.  This
    resolves every structure the nodes resolve (an FFT on a uniform
    frequency mesh spanning the optical cannot).  The result
    ``psi = -H`` is positive on the low-frequency side of an isolated line,
    the Faraday--Voigt convention of
    :func:`emergent_stokes_specific_intensity`.
    """

    profile = np.asarray(absorption_profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[0] != wavelength_angstrom.size:
        raise ValueError("absorption_profile must have shape (wavelength, depth)")
    if not np.any(profile):
        return np.zeros_like(profile)
    # Ascending frequency in units of 1e15 Hz keeps the logarithms well scaled.
    frequency = (LIGHT_SPEED / (wavelength_angstrom * 1.0e-8))[::-1] / 1.0e15
    values = profile[::-1]
    step = np.diff(frequency)
    slope = np.diff(values, axis=0) / step[:, np.newaxis]
    n = frequency.size
    left_slope = np.vstack((np.zeros((1, values.shape[1])), slope))  # segment j-1
    right_slope = np.vstack((slope, np.zeros((1, values.shape[1]))))  # segment j
    left_value = np.vstack((np.zeros((1, values.shape[1])), values[:-1] + slope * step[:, None]))
    right_value = values.copy()
    right_value[-1] = 0.0
    # c_j(x) = [f_j + s_j (x - y_j)] - [f_{j-1}(y_j) + s_{j-1} (x - y_j)]
    alpha = right_value - left_value - (right_slope - left_slope) * frequency[:, None]
    beta = right_slope - left_slope
    result = np.empty_like(values)
    constant = (values[-1] - values[0])[np.newaxis, :]
    for start in range(0, n, chunk):
        x = frequency[start : start + chunk]
        distance = np.abs(x[:, None] - frequency[None, :])
        logarithm = np.log(np.where(distance > 0.0, distance, 1.0))
        result[start : start + chunk] = (
            logarithm @ alpha + x[:, None] * (logarithm @ beta) - constant
        ) / PI
    return np.asarray(-result[::-1])


# ---------------------------------------------------------------------------
# Balmer line manifolds


@dataclass(frozen=True)
class BalmerLineTemplate:
    """Zero-field opacity of one Balmer line on its own wavelength mesh."""

    wavelength_angstrom: FloatArray
    mass_absorption_coefficient: FloatArray
    dispersion: FloatArray | None = None


@dataclass(frozen=True)
class BalmerBroadening:
    """Zero-field Balmer profile options shared by every magnetic branch."""

    include_self_broadening: bool = True
    self_broadening_prescription: str = "barklem"
    self_broadening_truncation_closure: str = "stark-core"

    def kwargs(self) -> dict[str, object]:
        return {
            "include_self_broadening": self.include_self_broadening,
            "self_broadening_prescription": self.self_broadening_prescription,
            "self_broadening_truncation_closure": (
                self.self_broadening_truncation_closure
            ),
        }


def balmer_line_templates(
    atmosphere: Atmosphere,
    *,
    broadening: BalmerBroadening = BalmerBroadening(),
    maximum_upper_level: int = MAXIMUM_H2DB_BALMER_UPPER_LEVEL,
    include_dispersion: bool = False,
) -> dict[int, BalmerLineTemplate]:
    """Return zero-field single-line opacities over 900--25000 A.

    The mesh is independent of the requested output grid: 0.02 A within
    20 A of the line and geometric in detuning beyond, over the whole range
    in which the synthesis evaluates Balmer wings.  H2db components displaced
    by thousands of Angstroms therefore still sample a complete profile
    instead of stopping at a template edge.
    """

    templates: dict[int, BalmerLineTemplate] = {}
    for line in BALMER_LINES:
        if line.upper_level > maximum_upper_level:
            continue
        center = line.wavelength_vacuum_angstrom
        mesh = np.unique(
            np.concatenate(
                (
                    center + np.arange(-20.0, 20.0001, 0.02),
                    center - np.geomspace(20.0, center - 900.0, 300),
                    center + np.geomspace(20.0, 25_000.0 - center, 400),
                )
            )
        )
        opacity = balmer_mass_absorption_coefficient(
            atmosphere, mesh, lines=(line,), **broadening.kwargs()
        )
        templates[line.upper_level] = BalmerLineTemplate(
            mesh,
            opacity,
            frequency_hilbert_dispersion(mesh, opacity) if include_dispersion else None,
        )
    return templates


def _translated_profile(
    template: BalmerLineTemplate,
    wavelength: FloatArray,
    frequency_shift_hz: float,
    values: FloatArray | None = None,
) -> FloatArray:
    """Sample a template at equal frequency detuning from a shifted center."""

    values = template.mass_absorption_coefficient if values is None else values
    frequency = LIGHT_SPEED / (wavelength * 1.0e-8) - frequency_shift_hz
    result = np.zeros((wavelength.size, template.mass_absorption_coefficient.shape[1]))
    valid = frequency > 0.0
    source = LIGHT_SPEED / frequency[valid] / 1.0e-8
    mesh = template.wavelength_angstrom
    inside = (source >= mesh[0]) & (source <= mesh[-1])
    if not np.any(inside):
        return result
    target = np.flatnonzero(valid)[inside]
    position = source[inside]
    index = np.clip(np.searchsorted(mesh, position) - 1, 0, mesh.size - 2)
    weight = ((position - mesh[index]) / (mesh[index + 1] - mesh[index]))[:, np.newaxis]
    result[target] = (1.0 - weight) * values[index] + weight * values[index + 1]
    return result


def weak_field_balmer_manifolds(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    *,
    broadening: BalmerBroadening = BalmerBroadening(),
    templates: dict[int, BalmerLineTemplate] | None = None,
    dispersion: bool = False,
) -> PolarizedOpacity | tuple[PolarizedOpacity, PolarizedOpacity]:
    """Return normal-triplet manifolds of the complete H3--H22 opacity.

    Every line is translated by the same normal-triplet displacement.  With
    ``dispersion`` the magneto-optical manifolds are returned as well,
    translated from each line's exact Hilbert partner.
    """

    wavelength = _validate_grid(wavelength_angstrom)
    field = _validate_field(field_strength_megagauss)
    if field > WEAK_FIELD_MAXIMUM_MEGAGAUSS:
        raise ValueError("the linear-Zeeman branch is limited to B <= 1 MG")
    if templates is None:
        templates = balmer_line_templates(
            atmosphere, broadening=broadening, maximum_upper_level=22,
            include_dispersion=dispersion,
        )
    shift = LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS * field * 1.0e6
    shape = (wavelength.size, atmosphere.n_depth)
    opacity = {q: np.zeros(shape) for q in (-1, 0, 1)}
    phase = {q: np.zeros(shape) for q in (-1, 0, 1)}
    rest_frequencies = {
        line.upper_level: LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
        for line in BALMER_LINES
    }
    for upper_level, template in templates.items():
        rest_frequency = rest_frequencies[upper_level]
        for q in (-1, 0, 1):
            correction = _stimulated_emission_ratio(
                atmosphere.temperature, rest_frequency, rest_frequency + q * shift
            )[np.newaxis, :]
            opacity[q] += correction * _translated_profile(template, wavelength, q * shift)
            if dispersion:
                phase[q] += correction * _translated_profile(template, wavelength, q * shift, template.dispersion)
    result = PolarizedOpacity(opacity[-1], opacity[0], opacity[1])
    if dispersion:
        return result, PolarizedOpacity(phase[-1], phase[0], phase[1])
    return result


def _stimulated_emission_ratio(
    temperature: FloatArray, rest_frequency: float, component_frequency: float
) -> FloatArray:
    """Replace the rest-frequency factor already included in a DA template."""

    inverse_thermal_frequency = PLANCK / (BOLTZMANN * temperature)
    return np.expm1(-component_frequency * inverse_thermal_frequency) / np.expm1(
        -rest_frequency * inverse_thermal_frequency
    )


def h2db_balmer_manifolds(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    database: H2dbTransitionDatabase,
    *,
    templates: dict[int, BalmerLineTemplate] | None = None,
    broadening: BalmerBroadening = BalmerBroadening(),
    maximum_upper_level: int = MAXIMUM_H2DB_BALMER_UPPER_LEVEL,
    dispersion: bool = False,
) -> PolarizedOpacity | tuple[PolarizedOpacity, PolarizedOpacity]:
    """Return H2db Halpha--H12 manifolds at one local field.

    Component ``i`` of line ``n`` contributes ``3 s_i S_n(B) phi_n`` to its
    manifold, where ``s_i`` are the normalized Boltzmann-weighted H2db
    strengths, ``S_n`` the field-dependent total strength relative to zero
    field, and ``phi_n`` the zero-field line opacity translated in frequency
    with stimulated emission evaluated at the component frequency.
    The isotropic mean is therefore the zero-field line opacity as
    ``B -> 0``.
    """

    wavelength = _validate_grid(wavelength_angstrom)
    field = _validate_field(field_strength_megagauss)
    if not 0 < maximum_upper_level <= MAXIMUM_H2DB_BALMER_UPPER_LEVEL:
        raise ValueError("maximum_upper_level must lie in 3..12")
    if templates is None:
        templates = balmer_line_templates(
            atmosphere, broadening=broadening, maximum_upper_level=maximum_upper_level,
            include_dispersion=dispersion,
        )
    shape = (wavelength.size, atmosphere.n_depth)
    manifolds = {-1: np.zeros(shape), 0: np.zeros(shape), 1: np.zeros(shape)}
    phases = {-1: np.zeros(shape), 0: np.zeros(shape), 1: np.zeros(shape)}
    for line in BALMER_LINES:
        level = line.upper_level
        if level > maximum_upper_level or level not in database.transitions_by_upper_level:
            continue
        components = database.balmer_components(
            level, line.wavelength_vacuum_angstrom, field, atmosphere.temperature
        )
        rest_frequency = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1.0e-8)
        template = templates[level]
        for index, (component_wavelength, delta_m) in enumerate(
            zip(components.wavelength_angstrom, components.delta_m)
        ):
            component_frequency = LIGHT_SPEED / (component_wavelength * 1.0e-8)
            shift = component_frequency - rest_frequency
            weight = 3.0 * (
                components.normalized_strength[index] * components.line_strength_scale
                * _stimulated_emission_ratio(
                    atmosphere.temperature, rest_frequency, component_frequency
                )
            )[np.newaxis, :]
            q = int(np.sign(delta_m))
            manifolds[q] += weight * _translated_profile(template, wavelength, shift)
            if dispersion:
                phases[q] += weight * _translated_profile(
                    template, wavelength, shift, template.dispersion
                )
    result = PolarizedOpacity(manifolds[-1], manifolds[0], manifolds[1])
    if dispersion:
        return result, PolarizedOpacity(phases[-1], phases[0], phases[1])
    return result


def h2db_component_wavelengths(
    field_strength_megagauss: float,
    database: H2dbTransitionDatabase,
    temperature: float,
    *,
    maximum_upper_level: int = 6,
    minimum_strength: float = 0.01,
) -> FloatArray:
    """Return the H2db component centers carrying >= ``minimum_strength``."""

    centers: list[float] = []
    for line in BALMER_LINES:
        if line.upper_level > maximum_upper_level:
            continue
        components = database.balmer_components(
            line.upper_level,
            line.wavelength_vacuum_angstrom,
            field_strength_megagauss,
            np.asarray([float(temperature)]),
        )
        strength = components.normalized_strength[:, 0]
        centers.extend(components.wavelength_angstrom[strength >= minimum_strength])
    return np.asarray(sorted(centers), dtype=np.float64)


# ---------------------------------------------------------------------------
# Continuum


def magnetic_continuum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase | None,
    *,
    include_rwa_photoionization: bool,
    include_molecular_absorption: bool = False,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
    include_series_pseudocontinuum: bool = True,
    include_centered_motion: bool = True,
) -> tuple[FloatArray, PolarizedOpacity]:
    """Return the unpolarized true-absorption continuum and the RWA manifolds.

    With the RWA, the ordinary H I bound-free opacity of n=1..8 is replaced
    by the polarization-resolved stationary-state RWA calculation evaluated
    with the atmosphere's own shell populations; free-free, H-minus and
    higher-shell terms are kept.  The dissolved-level pseudo-continuum is
    then attached to each shifted RWA edge instead of the zero-field edges.
    Scattering is excluded.
    """

    from .opacity import (
        hydrogen_bound_free_mass_absorption_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    )

    wavelength = _validate_grid(wavelength_angstrom)
    field = _validate_field(field_strength_megagauss)
    continuum = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
        include_molecular_absorption=include_molecular_absorption,
        h2_h2_cia_table=h2_h2_cia_table,
    )
    shape = (wavelength.size, atmosphere.n_depth)
    if not include_rwa_photoionization or field == 0.0:
        if include_series_pseudocontinuum:
            continuum = continuum + (
                hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                    atmosphere, wavelength
                )
            )
        return continuum, PolarizedOpacity.zeros(shape)
    if energy_database is None:
        raise ValueError("RWA photoionization requires the H2db energy database")
    ordinary = hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wavelength, maximum_level=RWA_MAXIMUM_LEVEL
    )
    rwa = PolarizedOpacity(
        *(
            explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                field,
                polarization,
                energy_database,
                maximum_level=RWA_MAXIMUM_LEVEL,
                include_dissolved_levels=include_series_pseudocontinuum,
                include_centered_motion=include_centered_motion,
            )
            for polarization in (-1, 0, 1)
        )
    )
    return np.maximum(0.0, continuum - ordinary), rwa


def rwa_dispersion(
    atmosphere: Atmosphere,
    wavelength_angstrom: FloatArray,
    field_strength_megagauss: float,
    energy_database: H2dbEnergyDatabase,
    *,
    include_centered_motion: bool = True,
) -> PolarizedOpacity:
    """Return the causal dispersion of the RWA manifolds on the output grid."""

    wavelength = _validate_grid(wavelength_angstrom)
    blue, red = _CONTINUUM_DISPERSION_RANGE_ANGSTROM
    extended = np.unique(
        np.concatenate(
            (
                np.geomspace(blue, wavelength[0], 601)[:-1],
                wavelength,
                np.geomspace(wavelength[-1], red, 401)[1:],
            )
        )
    )
    indices = np.searchsorted(extended, wavelength)
    manifolds = PolarizedOpacity(
        *(
            explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
                atmosphere,
                extended,
                field_strength_megagauss,
                polarization,
                energy_database,
                maximum_level=RWA_MAXIMUM_LEVEL,
                include_centered_motion=include_centered_motion,
            )
            for polarization in (-1, 0, 1)
        )
    )
    return manifolds.hilbert(extended).subset(indices)


# ---------------------------------------------------------------------------
# Surface geometry


@dataclass(frozen=True)
class SurfaceCells:
    """Visible-disk quadrature: one emergent ray per cell.

    ``projected_weight`` sums to one; the surface flux is
    ``pi * sum(projected_weight * I(ray))``.
    """

    field_strength_megagauss: FloatArray
    field_ray_cosine: FloatArray
    ray_mu: FloatArray
    projected_weight: FloatArray


def uniform_field_surface_cells(
    field_strength_megagauss: float,
    *,
    field_angle_deg: float | None = None,
    n_mu: int = 4,
    n_direction: int = 3,
) -> SurfaceCells:
    """Constant field modulus over the disk.

    Without ``field_angle_deg`` the field direction is unknown and taken as
    isotropic relative to each ray; Stokes I depends only on ``|cos psi|``,
    which is then uniform on [0, 1] and integrated with Gauss--Legendre
    nodes.  A specified angle applies to every ray.
    """

    field = _validate_field(field_strength_megagauss)
    nodes, weights = np.polynomial.legendre.leggauss(int(n_mu))
    mu = 0.5 * (nodes + 1.0)
    mu_weight = 0.5 * weights * mu
    mu_weight /= np.sum(mu_weight)
    if field_angle_deg is None:
        direction, direction_weight = np.polynomial.legendre.leggauss(int(n_direction))
        cosine = 0.5 * (direction + 1.0)
        direction_weight = 0.5 * direction_weight
    else:
        angle = float(field_angle_deg)
        if not np.isfinite(angle) or not 0.0 <= angle <= 180.0:
            raise ValueError("field_angle_deg must lie between 0 and 180 degrees")
        cosine = np.asarray([abs(np.cos(np.deg2rad(angle)))])
        direction_weight = np.asarray([1.0])
    mu_grid, cosine_grid = np.meshgrid(mu, cosine, indexing="ij")
    weight = mu_weight[:, np.newaxis] * direction_weight[np.newaxis, :]
    return SurfaceCells(
        np.full(mu_grid.size, field),
        cosine_grid.ravel(),
        mu_grid.ravel(),
        weight.ravel() / np.sum(weight),
    )


def dipole_surface_cells(
    field_strength_megagauss: float,
    *,
    field_strength_definition: Literal["visible-mean", "dipole-polar"] = "dipole-polar",
    inclination_deg: float = 60.0,
    offset_vector_radius: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_mu: int = 8,
    n_azimuth: int = 16,
    n_field_bins: int | None = 21,
) -> SurfaceCells:
    """Discretize an offset dipole over the visible disk.

    ``offset_vector_radius = (ax, ay, az)`` displaces the dipole in stellar
    radii in the magnetic-axis frame of Vera-Rueda & Rohrmann (2024): z along
    the dipole axis, the line of sight in the x--z plane.  Hardy et al.
    (2023) axial offsets are ``(0, 0, az)``.  ``dipole-polar`` scales the
    field so that an undisplaced dipole has that polar field; ``visible-mean``
    instead fixes the projected-area mean modulus.

    The surface integral uses Gauss--Legendre ``mu`` nodes and uniform
    azimuths.  With ``n_field_bins`` the cells are compressed into bins of
    equal projected weight in field modulus, each keeping its weighted mean
    field, rms field--ray cosine and mean limb cosine (a J0732 test differed
    from 128 uncompressed cells by 0.7 percent RMS).  ``None`` keeps every
    cell.
    """

    field = _validate_field(field_strength_megagauss)
    inclination = float(inclination_deg)
    if not np.isfinite(inclination) or not 0.0 <= inclination <= 180.0:
        raise ValueError("inclination_deg must lie between 0 and 180 degrees")
    offset = np.asarray(offset_vector_radius, dtype=np.float64)
    if offset.shape != (3,) or np.any(~np.isfinite(offset)) or np.linalg.norm(offset) >= 0.8:
        raise ValueError("offset_vector_radius must be three finite values with modulus < 0.8")
    if field_strength_definition not in ("visible-mean", "dipole-polar"):
        raise ValueError("field_strength_definition must be 'visible-mean' or 'dipole-polar'")
    if n_mu < 2 or n_azimuth < 4 or (n_field_bins is not None and n_field_bins < 1):
        raise ValueError("disk quadrature requires n_mu>=2, n_azimuth>=4, n_field_bins>=1")

    nodes, node_weights = np.polynomial.legendre.leggauss(int(n_mu))
    mu = 0.5 * (nodes + 1.0)
    azimuth = 2.0 * PI * (np.arange(int(n_azimuth)) + 0.5) / int(n_azimuth)
    mu_grid, azimuth_grid = np.meshgrid(mu, azimuth, indexing="ij")
    radial = np.sqrt(np.maximum(0.0, 1.0 - mu_grid**2))
    position = np.stack(
        (radial * np.cos(azimuth_grid), radial * np.sin(azimuth_grid), mu_grid), axis=-1
    )
    tilt = np.deg2rad(inclination)
    axis = np.asarray((np.sin(tilt), 0.0, np.cos(tilt)))
    x_axis = np.asarray((np.cos(tilt), 0.0, -np.sin(tilt)))
    y_axis = np.asarray((0.0, 1.0, 0.0))
    center = offset[0] * x_axis + offset[1] * y_axis + offset[2] * axis
    displacement = position - center
    distance = np.linalg.norm(displacement, axis=-1)
    direction = displacement / distance[..., np.newaxis]
    projection = np.sum(direction * axis, axis=-1)
    # Unit polar field for an undisplaced dipole (pole 1, equator 1/2).
    vector = (
        0.5
        * (3.0 * projection[..., np.newaxis] * direction - axis)
        / distance[..., np.newaxis] ** 3
    )
    modulus = np.linalg.norm(vector, axis=-1)
    cosine = np.clip(vector[..., 2] / modulus, -1.0, 1.0)
    weight = mu_grid * (0.5 * node_weights)[:, np.newaxis]
    weight = weight / np.sum(weight)
    if field_strength_definition == "visible-mean":
        modulus = modulus * field / float(np.sum(weight * modulus))
    else:
        modulus = modulus * field

    flat_field = modulus.ravel()
    flat_cosine = np.abs(cosine.ravel())
    flat_mu = mu_grid.ravel()
    flat_weight = weight.ravel()
    if n_field_bins is None:
        return SurfaceCells(flat_field, flat_cosine, flat_mu, flat_weight)
    order = np.argsort(flat_field, kind="stable")
    cumulative = np.cumsum(flat_weight[order])
    bins = np.minimum((cumulative * n_field_bins).astype(int), n_field_bins - 1)
    fields, cosines, mus, weights = [], [], [], []
    for index in range(n_field_bins):
        selected = order[bins == index]
        if selected.size == 0:
            continue
        w = flat_weight[selected]
        total = float(np.sum(w))
        fields.append(float(np.sum(w * flat_field[selected]) / total))
        cosines.append(float(np.sqrt(np.sum(w * flat_cosine[selected] ** 2) / total)))
        mus.append(float(np.sum(w * flat_mu[selected]) / total))
        weights.append(total)
    weights_array = np.asarray(weights)
    return SurfaceCells(
        np.asarray(fields),
        np.asarray(cosines),
        np.asarray(mus),
        weights_array / np.sum(weights_array),
    )


# ---------------------------------------------------------------------------
# Emergent spectrum


@dataclass(frozen=True)
class MagneticPhysics:
    """Atomic data and switches for one magnetic synthesis or structure."""

    transitions: H2dbTransitionDatabase | None
    energies: H2dbEnergyDatabase | None
    broadening: BalmerBroadening = BalmerBroadening()
    include_rwa_photoionization: bool = True
    include_cyclotron_absorption: bool = False
    include_magnetic_eos: bool = True
    include_centered_motion: bool = True
    include_molecular_absorption: bool = False
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None
    # A disk model selects this once from its maximum visible field.  The
    # mean-field structure must not independently fall back to weak physics.
    line_regime: Literal["linear-zeeman", "h2db"] | None = None

    def regime(self, maximum_field_megagauss: float) -> str:
        if self.line_regime is not None:
            return self.line_regime
        return (
            "linear-zeeman"
            if maximum_field_megagauss <= WEAK_FIELD_MAXIMUM_MEGAGAUSS
            else "h2db"
        )


def _line_manifolds(
    atmosphere: Atmosphere,
    wavelength: FloatArray,
    field: float,
    physics: MagneticPhysics,
    regime: str,
    cache: dict[str, object],
    dispersion: bool,
) -> tuple[PolarizedOpacity, PolarizedOpacity | None]:
    """Line manifolds (and optional dispersion) from cached line templates."""

    key = (regime, dispersion)
    templates = cache.get(key)
    if templates is None:
        templates = balmer_line_templates(
            atmosphere,
            broadening=physics.broadening,
            maximum_upper_level=22 if regime == "linear-zeeman" else MAXIMUM_H2DB_BALMER_UPPER_LEVEL,
            include_dispersion=dispersion,
        )
        cache[key] = templates
    if regime == "linear-zeeman":
        result = weak_field_balmer_manifolds(
            atmosphere, wavelength, field, templates=templates, dispersion=dispersion  # type: ignore[arg-type]
        )
    else:
        if physics.transitions is None:
            raise ValueError("fields above 1 MG require the H2db transition database")
        result = h2db_balmer_manifolds(
            atmosphere, wavelength, field, physics.transitions,
            templates=templates, dispersion=dispersion,  # type: ignore[arg-type]
        )
    return result if dispersion else (result, None)  # type: ignore[return-value]


def _local_atmosphere(
    atmosphere: Atmosphere, field: float, physics: MagneticPhysics, regime: str
) -> Atmosphere:
    if not physics.include_magnetic_eos or regime != "h2db" or field == 0.0:
        return atmosphere
    from .magnetic_eos import atmosphere_with_magnetic_hydrogen_eos

    if physics.energies is None:
        raise ValueError("the magnetic EOS requires the H2db energy database")
    return atmosphere_with_magnetic_hydrogen_eos(
        atmosphere,
        field,
        physics.energies,
        include_centered_motion=physics.include_centered_motion,
    )


def _population_scale(reference: Atmosphere, local: Atmosphere) -> FloatArray:
    """Local/reference n=2 population per gram (Balmer lower level)."""

    if local is reference:
        return np.ones(reference.n_depth)
    from .opacity import _atmosphere_level_distribution

    reference_n2 = _atmosphere_level_distribution(reference, 2).population_density[..., 1]
    local_n2 = _atmosphere_level_distribution(local, 2).population_density[..., 1]
    return (local_n2 / local.mass_density) / (reference_n2 / reference.mass_density)


def synthesize_magnetic_hydrogen_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    cells: SurfaceCells,
    physics: MagneticPhysics,
    *,
    polarized_transfer: PolarizedTransfer = "full-stokes-iquv",
    n_angle: int = 4,
    progress: object | None = None,
) -> Spectrum:
    """Return the disk-integrated Stokes-I surface flux of a DAH.

    Each surface cell uses the local-field chemical equilibrium on the shared
    ``T(P)`` structure (above 1 MG, when enabled), its own Balmer manifolds,
    RWA continuum and cyclotron resonance, an exact coherent-scattering
    source for its angle-averaged opacity, and one ray at its limb cosine and
    field--ray angle.  Zero-field Stark kernels are computed once on the
    shared structure and rescaled by the local n=2 population per gram (the
    local charged-particle densities that set the widths change by <0.5%
    across a 100-MG disk).
    """

    from ._spectrum_source import solve_spectrum_source
    from .opacity import (
        electron_scattering_mass_coefficient,
        hydrogen_rayleigh_scattering_mass_coefficient,
        optical_depth_from_mass_opacity,
    )
    from .radiative_transfer import (
        emergent_specific_intensity,
        emergent_stokes_specific_intensity,
    )
    from .spectrum import planck_lambda_angstrom

    wavelength = _validate_grid(wavelength_angstrom)
    lower, upper = DAH_WAVELENGTH_RANGE_ANGSTROM
    if wavelength[0] < lower or wavelength[-1] > upper:
        raise ValueError(f"DAH synthesis supports {lower:g}--{upper:g} A")
    if polarized_transfer not in ("full-stokes-iquv", "scalar-stokes-i"):
        raise ValueError("polarized_transfer must be 'full-stokes-iquv' or 'scalar-stokes-i'")
    polarized = polarized_transfer == "full-stokes-iquv"
    regime = physics.regime(float(np.max(cells.field_strength_megagauss)))
    cache: dict[str, object] = {}
    flux = np.zeros(wavelength.size)
    # Cells sharing a field (uniform geometry) share every local opacity.
    field_cache: dict[float, tuple] = {}
    for cell in range(cells.projected_weight.size):
        field = float(cells.field_strength_megagauss[cell])
        cosine = float(cells.field_ray_cosine[cell])
        mu = float(cells.ray_mu[cell])
        if field not in field_cache:
            local = _local_atmosphere(atmosphere, field, physics, regime)
            lines, line_dispersion = _line_manifolds(
                atmosphere, wavelength, field, physics, regime, cache,
                polarized and field > 0.0,
            )
            population = _population_scale(atmosphere, local)[np.newaxis, :]
            lines = lines.scaled(population)
            if line_dispersion is not None:
                line_dispersion = line_dispersion.scaled(population)
            unpolarized, rwa = magnetic_continuum(
                local,
                wavelength,
                field,
                physics.energies,
                include_rwa_photoionization=(
                    physics.include_rwa_photoionization and regime == "h2db"
                ),
                include_molecular_absorption=physics.include_molecular_absorption,
                h2_h2_cia_table=physics.h2_h2_cia_table,
                include_centered_motion=physics.include_centered_motion,
            )
            continuum_dispersion = (
                rwa_dispersion(
                    local, wavelength, field, physics.energies,  # type: ignore[arg-type]
                    include_centered_motion=physics.include_centered_motion,
                )
                if polarized and physics.include_rwa_photoionization and regime == "h2db" and field > 0.0
                else None
            )
            if physics.include_rwa_photoionization and regime == "h2db":
                polarized_opacity = lines + rwa
            else:
                polarized_opacity = lines
            # Free electrons: with a field their free-free absorption and
            # Thomson scattering become the magneto-ionic mode coefficients,
            # replacing the zero-field values in the continuum.
            include_cyclotron = physics.include_cyclotron_absorption and field > 0.0
            rayleigh = hydrogen_rayleigh_scattering_mass_coefficient(local, wavelength)
            if include_cyclotron:
                unpolarized = unpolarized - hydrogen_free_free_mass_absorption_coefficient(
                    local, wavelength
                )
                scattering = rayleigh
                free_absorption, free_scattering = isotropic_free_electron_opacity(
                    local, wavelength, field,
                    angular_quadrature_order=CYCLOTRON_ANGULAR_NODES,
                )
            else:
                scattering = electron_scattering_mass_coefficient(local)[np.newaxis, :] + rayleigh
                free_absorption = free_scattering = 0.0
            thermal_isotropic = (
                unpolarized + polarized_opacity.isotropic() + free_absorption
            )
            isotropic_scattering = scattering + free_scattering
            planck = np.ascontiguousarray(
                planck_lambda_angstrom(
                    wavelength[:, np.newaxis], local.temperature[np.newaxis, :]
                )
            )
            depth = optical_depth_from_mass_opacity(
                local.column_mass, thermal_isotropic + isotropic_scattering
            )
            _, radiation, _ = solve_spectrum_source(
                depth,
                planck,
                thermal_isotropic,
                isotropic_scattering,
                wavelength=wavelength,
                n_angle=n_angle,
                discretization="formal-linear",
            )
            dispersion = None
            if line_dispersion is not None:
                dispersion = (
                    line_dispersion
                    if continuum_dispersion is None
                    else line_dispersion + continuum_dispersion
                )
            field_cache.clear()
            field_cache[field] = (
                local,
                unpolarized,
                polarized_opacity,
                dispersion,
                scattering,
                planck,
                radiation.mean_intensity,
                include_cyclotron,
            )
        (
            local,
            unpolarized,
            polarized_opacity,
            dispersion,
            scattering,
            planck,
            mean_intensity,
            include_cyclotron,
        ) = field_cache[field]
        eta_i, eta_q, eta_v = polarized_opacity.stokes(cosine)
        eta_i = eta_i + unpolarized
        thermal_q, thermal_v = eta_q, eta_v
        ray_scattering = scattering
        scattered_q = scattered_v = 0.0
        rho_q = np.zeros_like(eta_i)
        rho_v = np.zeros_like(eta_i)
        if dispersion is not None:
            _, rho_q, rho_v = dispersion.stokes(cosine)
        if include_cyclotron:
            free = magnetized_free_electron_manifolds(local, wavelength, field, cosine)
            a_i, a_q, a_v = PolarizedOpacity(*free.absorption).stokes(cosine)
            s_i, s_q, s_v = PolarizedOpacity(*free.scattering).stokes(cosine)
            # Extinction carries both parts with their dichroism; the thermal
            # part emits B and the scattered part re-emits J in its own
            # polarization (unpolarized re-emission would feed the mode the
            # resonance cannot absorb).
            eta_i = eta_i + a_i
            eta_q = eta_q + a_q + s_q
            eta_v = eta_v + a_v + s_v
            thermal_q = thermal_q + a_q
            thermal_v = thermal_v + a_v
            ray_scattering = scattering + s_i
            scattered_q, scattered_v = s_q, s_v
            if polarized:
                _, d_q, d_v = PolarizedOpacity(*free.dispersion).stokes(cosine)
                rho_q = rho_q + d_q
                rho_v = rho_v + d_v
        extinction = eta_i + ray_scattering
        emission_i = eta_i * planck + ray_scattering * mean_intensity
        source = emission_i / extinction
        if polarized:
            emission = np.zeros(extinction.shape + (4,))
            emission[..., 0] = emission_i
            emission[..., 1] = thermal_q * planck + scattered_q * mean_intensity
            emission[..., 3] = thermal_v * planck + scattered_v * mean_intensity
            intensity = emergent_stokes_specific_intensity(
                local.column_mass,
                extinction,
                eta_q,
                eta_v,
                rho_q,
                rho_v,
                source,
                mu,
                emission_stokes=emission,
                formal_solver="delo-linear",
            ).i
        else:
            intensity = emergent_specific_intensity(
                optical_depth_from_mass_opacity(local.column_mass, extinction),
                source,
                mu,
            )
        flux += cells.projected_weight[cell] * PI * intensity
        if callable(progress):
            progress(cell + 1, cells.projected_weight.size)
    return Spectrum(
        wavelength,
        flux,
        {
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda (projected-area sum of Stokes I)",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "magnetic_line_regime": regime,
            "lines": (
                "Halpha-H22 normal linear-Zeeman triplets"
                if regime == "linear-zeeman"
                else "Halpha-H12 H2db components; H13+ omitted"
            ),
            "polarized_transfer": (
                "coupled IQUV, matrix exponential with linear equilibrium source, line and continuum "
                "dichroism and magneto-optical dispersion"
                if polarized
                else "scalar Stokes I with ray-specific pi/sigma opacity"
            ),
            "surface_cells": int(cells.projected_weight.size),
            "surface_field_megagauss": cells.field_strength_megagauss.tolist(),
            "surface_projected_weight": cells.projected_weight.tolist(),
            "magnetic_bound_free": (
                f"stationary-state Rohrmann (2026) RWA through n={RWA_MAXIMUM_LEVEL}"
                if physics.include_rwa_photoionization and regime == "h2db"
                else "ordinary zero-field H I bound-free"
            ),
            "free_electrons": (
                "magneto-ionic free-free absorption, Thomson scattering and "
                "dispersion per circular mode, Doppler-broadened cyclotron "
                "resonance"
                if physics.include_cyclotron_absorption
                else "zero-field free-free and Thomson scattering"
            ),
            "surface_chemistry": (
                "local-field magnetic Saha/HM equilibrium on the shared T(P)"
                if physics.include_magnetic_eos and regime == "h2db"
                else "equilibrium of the supplied atmosphere"
            ),
            "combined_stark_zeeman": (
                "zero-field Tremblay-Bergeron profile translated to each component"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Structure opacity


def magnetic_structure_opacity(
    field_strength_megagauss: float,
    physics: MagneticPhysics,
) -> tuple[object, object, object]:
    """Return ``(balmer, continuum, scattering)`` opacity callables ``(atm, wave)``.

    With a field, free electrons use the magneto-ionic mode coefficients:
    their angle-averaged free--free absorption replaces the zero-field value
    in the continuum, and the third callable returns the change of their
    Thomson scattering relative to the solver's zero-field value.

    Both are angle averages for the isotropic radiation field of a plane-
    parallel structure solve and use the same manifolds as the synthesis.
    With the RWA the dissolved-level pseudo-continuum is attached to the RWA
    edges here, and the solver's zero-field pseudo-continuum must be off
    (see :func:`structure_uses_rwa_dissolution`); otherwise the solver's own
    switch supplies it.
    """

    field = _validate_field(field_strength_megagauss)
    regime = physics.regime(field)
    rwa_active = structure_uses_rwa_dissolution(field, physics)

    def balmer(current: Atmosphere, wavelength: FloatArray) -> FloatArray:
        grid = np.ascontiguousarray(wavelength, dtype=np.float64)
        if regime == "linear-zeeman":
            return weak_field_balmer_manifolds(
                current, grid, field, broadening=physics.broadening
            ).isotropic()
        return h2db_balmer_manifolds(
            current,
            grid,
            field,
            physics.transitions,  # type: ignore[arg-type]
            broadening=physics.broadening,
        ).isotropic()

    def continuum(current: Atmosphere, wavelength: FloatArray) -> FloatArray:
        grid = np.ascontiguousarray(wavelength, dtype=np.float64)
        unpolarized, rwa = magnetic_continuum(
            current,
            grid,
            field,
            physics.energies,
            include_rwa_photoionization=(
                physics.include_rwa_photoionization and regime == "h2db"
            ),
            include_molecular_absorption=physics.include_molecular_absorption,
            h2_h2_cia_table=physics.h2_h2_cia_table,
            include_series_pseudocontinuum=rwa_active,
            include_centered_motion=physics.include_centered_motion,
        )
        result = unpolarized + rwa.isotropic()
        if physics.include_cyclotron_absorption and field > 0.0:
            absorption, _ = isotropic_free_electron_opacity(
                current, grid, field, angular_quadrature_order=CYCLOTRON_ANGULAR_NODES
            )
            result = (
                result
                - hydrogen_free_free_mass_absorption_coefficient(current, grid)
                + absorption
            )
        return np.maximum(result, 0.0)

    def scattering(current: Atmosphere, wavelength: FloatArray) -> FloatArray:
        """Magnetic Thomson scattering minus the solver's zero-field Thomson."""

        from .opacity import electron_scattering_mass_coefficient

        grid = np.ascontiguousarray(wavelength, dtype=np.float64)
        if not (physics.include_cyclotron_absorption and field > 0.0):
            return np.zeros((grid.size, current.n_depth))
        _, magnetic = isotropic_free_electron_opacity(
            current, grid, field, angular_quadrature_order=CYCLOTRON_ANGULAR_NODES
        )
        return magnetic - electron_scattering_mass_coefficient(current)[np.newaxis, :]

    return balmer, continuum, scattering


def cyclotron_structure_wavelengths(field_strength_megagauss: float) -> FloatArray:
    """Mesh resolving the cyclotron resonance of one field for a structure.

    Its thermal width is ~1e-3 lambda_c |cos| and shrinks to the natural
    width along the field-perpendicular direction, far below a ~2% opacity-
    sampling mesh.  Sampling it at a few arbitrary detunings gives a wrong,
    extremely temperature-sensitive (stiff) energy exchange.  The mesh is
    geometric in detuning from 0.05 A to 2% of lambda_c on both sides.
    """

    field = _validate_field(field_strength_megagauss)
    if field <= 0.0:
        return np.zeros(0)
    center = 2.0 * PI * LIGHT_SPEED**2 * ELECTRON_MASS / (
        ELEMENTARY_CHARGE_ESU * field * 1.0e6
    ) * 1.0e8
    offsets = np.geomspace(0.05, 0.02 * center, 160)
    return np.concatenate((center - offsets[::-1], [center], center + offsets))


def structure_uses_rwa_dissolution(
    field_strength_megagauss: float, physics: MagneticPhysics
) -> bool:
    """Whether the structure continuum carries its own dissolved RWA edges."""

    field = _validate_field(field_strength_megagauss)
    return (
        physics.include_rwa_photoionization
        and physics.regime(field) == "h2db"
        and field > 0.0
    )


def default_magnetic_da_wavelength_grid() -> FloatArray:
    """Return a 3600--7000 A grid resolving magnetic Balmer components."""

    broad = np.arange(3_600.0, 7_000.0001, 0.25)
    cores = [
        np.arange(
            line.wavelength_vacuum_angstrom - 35.0,
            line.wavelength_vacuum_angstrom + 35.0001,
            0.05,
        )
        for line in BALMER_LINES[:4]
    ]
    return np.unique(np.round(np.concatenate((broad, *cores)), 6))


__all__ = [
    "BalmerBroadening",
    "BalmerLineTemplate",
    "DAH_WAVELENGTH_RANGE_ANGSTROM",
    "LINEAR_ZEEMAN_FREQUENCY_HZ_PER_GAUSS",
    "LinearZeemanTriplet",
    "MAXIMUM_H2DB_BALMER_UPPER_LEVEL",
    "MagneticPhysics",
    "PolarizedOpacity",
    "SurfaceCells",
    "WEAK_FIELD_MAXIMUM_MEGAGAUSS",
    "balmer_line_templates",
    "cyclotron_structure_wavelengths",
    "default_magnetic_da_wavelength_grid",
    "dipole_surface_cells",
    "frequency_hilbert_dispersion",
    "h2db_balmer_manifolds",
    "h2db_component_wavelengths",
    "linear_zeeman_triplet",
    "magnetic_continuum",
    "magnetic_structure_opacity",
    "rwa_dispersion",
    "structure_uses_rwa_dissolution",
    "synthesize_magnetic_hydrogen_spectrum",
    "uniform_field_surface_cells",
    "weak_field_balmer_manifolds",
]
