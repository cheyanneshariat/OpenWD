"""Spectrum synthesis on a converged atmosphere structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid
from .atmosphere import Atmosphere
from .constants import BOLTZMANN, LIGHT_SPEED, PI, PLANCK, STEFAN_BOLTZMANN
from .radiative_transfer import Backend, emergent_flux
from ._spectrum_source import solve_spectrum_source

if TYPE_CHECKING:
    from .d6 import TOPbasePhotoionizationDatabase
    from .metals import (
        AtomicDatabase,
        CaIHeProfileTable,
        CaIIHeProfileTable,
        MgIIHeProfileTable,
        MgHeRedWingTable,
        VernerPhotoionizationDatabase,
    )
    from .molecules import H2H2CollisionInducedAbsorptionTable
    from .hydrogen_self import BarklemSelfBroadeningTable
    from .jackson_lyman import JacksonLymanProfileTable
    from .quasimolecular import AllardUnifiedLymanTable


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Spectrum:
    """Rest-frame surface flux density on a vacuum-wavelength grid."""

    wavelength_angstrom: FloatArray
    surface_flux_lambda: FloatArray
    metadata: dict[str, object]

    @property
    def bolometric_flux(self) -> float:
        """Numerically integrated surface flux in erg cm^-2 s^-1."""

        return float(
            trapezoid(self.surface_flux_lambda, self.wavelength_angstrom)
        )

    @property
    def flux_effective_temperature(self) -> float:
        """Effective temperature inferred from the integrated wavelength grid."""

        return (self.bolometric_flux / STEFAN_BOLTZMANN) ** 0.25


def planck_lambda_angstrom(wavelength_angstrom: ArrayLike, temperature: ArrayLike) -> FloatArray:
    r"""Planck intensity ``B_lambda`` per Angstrom in cgs units.

    Inputs broadcast normally.  The result has units
    erg s^-1 cm^-2 sr^-1 Angstrom^-1.
    """

    wavelength_angstrom = np.asarray(wavelength_angstrom, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(wavelength_angstrom)) or np.any(wavelength_angstrom <= 0.0):
        raise ValueError("wavelength_angstrom must contain finite positive values")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must contain finite positive values")

    wavelength_cm = wavelength_angstrom * 1.0e-8
    exponent = PLANCK * LIGHT_SPEED / (wavelength_cm * BOLTZMANN * temperature)
    inverse_expm1 = np.empty(np.broadcast_shapes(exponent.shape, wavelength_cm.shape))
    exponent = np.broadcast_to(exponent, inverse_expm1.shape)
    small = exponent < 50.0
    inverse_expm1[small] = 1.0 / np.expm1(exponent[small])
    inverse_expm1[~small] = np.exp(-exponent[~small])
    return (
        2.0
        * PLANCK
        * LIGHT_SPEED**2
        / np.broadcast_to(wavelength_cm, inverse_expm1.shape) ** 5
        * inverse_expm1
        * 1.0e-8
    )


def _depth_refined_transfer_inputs(
    column_mass,
    temperature,
    wavelength,
    absorption,
    scattering,
    emission_source,
    factor: int,
):
    """Interpolate converged-structure transfer inputs onto a finer depth grid.

    Structure grids carry roughly 40 points over ten decades of column mass,
    i.e. a factor of about two in optical depth per interval near the
    photosphere.  That spacing is adequate for the structure's second-order
    Feautrier flux but the piecewise-linear formal solution then misses
    several per cent of the emergent flux of cool, steep, convective helium
    atmospheres.  Opacities are interpolated linearly in log-log against
    column mass, the temperature likewise, the Planck function is evaluated
    exactly at the interpolated temperature, and any departure of the
    emission source from Planck (CRD Ca II lines) is interpolated as a ratio.
    """

    if factor == 1:
        return column_mass, absorption, scattering, emission_source
    log_mass = np.log(column_mass)
    fine = np.interp(
        np.linspace(0.0, log_mass.size - 1.0, (log_mass.size - 1) * factor + 1),
        np.arange(log_mass.size, dtype=np.float64),
        log_mass,
    )
    tiny = np.finfo(np.float64).tiny

    def log_interpolate(values):
        logs = np.log(np.maximum(values, tiny))
        right = np.searchsorted(log_mass, fine, side="right").clip(1, log_mass.size - 1)
        left = right - 1
        weight = (fine - log_mass[left]) / (log_mass[right] - log_mass[left])
        return np.exp(logs[..., left] * (1.0 - weight) + logs[..., right] * weight)

    fine_temperature = log_interpolate(temperature)
    native_planck = planck_lambda_angstrom(
        wavelength[:, np.newaxis], np.asarray(temperature)[np.newaxis, :]
    )
    resolved = native_planck > tiny * 1.0e20
    ratio = np.where(
        resolved, emission_source / np.where(resolved, native_planck, 1.0), 1.0
    )
    right = np.searchsorted(log_mass, fine, side="right").clip(1, log_mass.size - 1)
    left = right - 1
    weight = (fine - log_mass[left]) / (log_mass[right] - log_mass[left])
    fine_ratio = ratio[:, left] * (1.0 - weight) + ratio[:, right] * weight
    fine_emission = np.ascontiguousarray(
        planck_lambda_angstrom(wavelength[:, np.newaxis], fine_temperature[np.newaxis, :])
        * fine_ratio
    )
    return (
        np.exp(fine),
        np.ascontiguousarray(log_interpolate(absorption)),
        np.ascontiguousarray(log_interpolate(scattering)),
        fine_emission,
    )


def _ca_ii_crd_emission_source(
    atmosphere,
    wavelength,
    absorption,
    scattering,
    planck,
    probabilities,
    line_extinction,
    n_angle,
):
    """Return the thermal-emission source including CRD Ca II H/K lines.

    The linear solver forms ``S = (kappa B + sigma J)/chi``.  With the H/K
    source functions fixed by complete redistribution, the lines are pure
    extinction whose emissivity is ``kappa_L S_L``; folding that into the
    emission term leaves the coherent background scattering unchanged.
    """

    if probabilities is None:
        return planck
    from .cool_metal_nlte import ca_ii_crd_resonance_source_functions

    sources = ca_ii_crd_resonance_source_functions(
        atmosphere, wavelength, absorption, scattering, line_extinction,
        probabilities, n_angle=n_angle,
    )
    lines = line_extinction(wavelength)
    per_angstrom = LIGHT_SPEED / (wavelength * 1.0e-8) ** 2 * 1.0e-8
    line_sum = sum(lines.values())
    emission = sum(
        lines[key] * sources[key][np.newaxis, :] for key in lines
    ) * per_angstrom[:, np.newaxis]
    return np.ascontiguousarray(
        ((absorption - line_sum) * planck + emission)
        / np.maximum(absorption, np.finfo(np.float64).tiny)
    )


def synthesize_gray_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    opacity_ratio: ArrayLike | float = 1.0,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> Spectrum:
    """Synthesize an LTE spectrum with prescribed monochromatic opacity ratios.

    ``opacity_ratio`` is ``kappa_lambda / kappa_R`` and may be a scalar or one
    value per wavelength.  The default is a genuinely gray spectrum.  This
    argument is the extension point for the forthcoming hydrogen continuum
    and line-opacity modules.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength_angstrom must be a 1D array with at least two points")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_angstrom must increase strictly")

    ratio = np.asarray(opacity_ratio, dtype=np.float64)
    if ratio.ndim == 0:
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("opacity_ratio must be finite and positive")
        optical_depth: FloatArray = np.ascontiguousarray(
            float(ratio) * atmosphere.rosseland_optical_depth
        )
    else:
        if ratio.shape != wavelength.shape:
            raise ValueError("array opacity_ratio must have one value per wavelength")
        if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
            raise ValueError("opacity_ratio must contain finite positive values")
        optical_depth = np.ascontiguousarray(
            ratio[:, np.newaxis] * atmosphere.rosseland_optical_depth[np.newaxis, :]
        )

    source = np.ascontiguousarray(
        planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    )
    flux = emergent_flux(
        optical_depth, source, n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "transfer": "pure-absorption, piecewise-linear source",
            "n_angle": int(n_angle),
            "backend": (
                "c" if backend == "c" else
                "python" if backend == "python" else "auto"
            ),
        },
    )


def synthesize_balmer_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    continuum_opacity: ArrayLike | float | None = None,
    precomputed_balmer_opacity: ArrayLike | None = None,
    precomputed_lyman_opacity: ArrayLike | None = None,
    include_molecular_absorption: bool = True,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
    include_balmer_self_broadening: bool = True,
    balmer_self_broadening_impact_validity_fraction: float = 1.0,
    balmer_self_broadening_prescription: str = "barklem",
    balmer_self_broadening_truncation_closure: str = "renormalize",
    barklem_self_table: BarklemSelfBroadeningTable | None = None,
    include_lyman: bool = False,
    include_paschen: bool = False,
    include_brackett: bool = False,
    include_series_pseudocontinuum: bool = False,
    include_neutral_lyman_alpha_wing: bool = True,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    jackson_lyman_table: JacksonLymanProfileTable | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    ca_ii_helium_impact_scale: float = 1.0,
    ca_ii_resonance_scattering_fraction: float = 0.0,
    ca_ii_resonance_collision_strengths: str | None = None,
    include_dense_helium_metal_ionization: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-4,
    maximum_metal_lines: int | None = 20_000,
    excluded_metal_line_elements: Iterable[str] = (),
    emergent_ray_mu: float | None = None,
    n_angle: int = 4,
    backend: Backend = "auto",
    transfer_discretization: Literal["formal-linear", "formal-pchip"] = "formal-linear",
    transfer_depth_refinement: int = 1,
) -> Spectrum:
    """Synthesize hydrogen lines on a supplied pure-H atmosphere structure.

    By default the continuum includes LTE H I bound-free/free-free, H-minus,
    and electron scattering. The line optical depths use the same
    Hummer--Mihalas LTE level populations as the atmosphere EOS and the
    bundled Tremblay-Bergeron Stark profiles. Rayleigh and electron scattering
    use a coherent, isotropic source. The atmospheric structure may be gray or
    independently frequency-converged.
    """

    if transfer_discretization not in ("formal-linear", "formal-pchip"):
        raise ValueError("unsupported hydrogen transfer_discretization")
    if transfer_discretization == "formal-pchip" and emergent_ray_mu is not None:
        raise ValueError("formal-pchip currently supports angle-integrated spectra only")

    from .opacity import (
        balmer_mass_absorption_coefficient,
        brackett_mass_absorption_coefficient,
        electron_scattering_mass_coefficient,
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
        hydrogen_rayleigh_scattering_mass_coefficient,
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient,
        lyman_mass_absorption_coefficient,
        optical_depth_from_mass_opacity,
        paschen_mass_absorption_coefficient,
    )

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_angstrom must be a strictly increasing 1D array")
    metal_state = None
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError("metal photoionization requires metal_database and abundances")
    if (
        not np.isfinite(ca_ii_resonance_scattering_fraction)
        or not 0.0 <= ca_ii_resonance_scattering_fraction <= 1.0
    ):
        raise ValueError("Ca II resonance scattering fraction must be in [0, 1]")
    if (
        ca_ii_resonance_collision_strengths is not None
        and ca_ii_resonance_scattering_fraction > 0.0
    ):
        raise ValueError(
            "choose either CHIANTI Ca II source functions or a diagnostic "
            "constant scattering fraction"
        )
    if metal_database is not None and metal_abundances is not None:
        from .metals import atmosphere_with_metal_electrons, metal_lte_state

        metal_state = metal_lte_state(
            atmosphere,
            metal_database,
            metal_abundances,
            include_dense_helium_ionization=include_dense_helium_metal_ionization,
        )
        atmosphere = atmosphere_with_metal_electrons(atmosphere, metal_state)
    if continuum_opacity is None:
        continuum_absorption = hydrogen_continuum_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            include_electron_scattering=False,
            include_rayleigh_scattering=False,
            include_molecular_absorption=include_molecular_absorption,
            h2_h2_cia_table=h2_h2_cia_table,
        )
        scattering = (
            electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
            + hydrogen_rayleigh_scattering_mass_coefficient(
                atmosphere, wavelength
            )
        )
        continuum_description = (
            "LTE H I/H-minus plus coherent electron and H I/H2 Rayleigh scattering"
        )
    else:
        continuum_absorption = np.asarray(continuum_opacity, dtype=np.float64)
        if continuum_absorption.ndim == 0:
            continuum_absorption = np.full(
                (wavelength.size, atmosphere.n_depth),
                float(continuum_absorption),
            )
        elif continuum_absorption.shape != (wavelength.size, atmosphere.n_depth):
            raise ValueError("continuum_opacity must be scalar or have shape (wavelength, depth)")
        if np.any(~np.isfinite(continuum_absorption)) or np.any(continuum_absorption <= 0.0):
            raise ValueError("continuum_opacity must contain finite positive values")
        scattering = np.zeros((1, atmosphere.n_depth), dtype=np.float64)
        continuum_description = "user-supplied mass opacity"

    if precomputed_balmer_opacity is None:
        line_opacity = balmer_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            include_self_broadening=include_balmer_self_broadening,
            self_broadening_impact_validity_fraction=(
                balmer_self_broadening_impact_validity_fraction
            ),
            self_broadening_prescription=(
                balmer_self_broadening_prescription
            ),
            self_broadening_truncation_closure=(
                balmer_self_broadening_truncation_closure
            ),
            barklem_self_table=barklem_self_table,
        )
    else:
        line_opacity = np.asarray(precomputed_balmer_opacity, dtype=np.float64)
        if line_opacity.shape != (wavelength.size, atmosphere.n_depth):
            raise ValueError(
                "precomputed_balmer_opacity must have shape (wavelength, depth)"
            )
        if np.any(~np.isfinite(line_opacity)) or np.any(line_opacity < 0.0):
            raise ValueError(
                "precomputed_balmer_opacity must contain finite nonnegative values"
            )
        line_opacity = np.array(line_opacity, copy=True)
    if include_paschen:
        line_opacity += paschen_mass_absorption_coefficient(
            atmosphere, wavelength
        )
    if include_brackett:
        line_opacity += brackett_mass_absorption_coefficient(
            atmosphere, wavelength
        )
    if include_lyman:
        if precomputed_lyman_opacity is None:
            line_opacity += lyman_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                unified_allard_table=unified_allard_table,
                jackson_lyman_table=jackson_lyman_table,
            )
            if include_neutral_lyman_alpha_wing:
                line_opacity += (
                    lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
                        atmosphere,
                        wavelength,
                        allard_table=(
                            unified_allard_table.lines.get((1, 2))
                            if (
                                unified_allard_table is not None
                                and not (
                                    jackson_lyman_table is not None
                                    and (1, 2) in jackson_lyman_table.lines
                                )
                            )
                            else None
                        ),
                    )
                )
        else:
            supplied_lyman = np.asarray(
                precomputed_lyman_opacity, dtype=np.float64
            )
            if supplied_lyman.shape != (wavelength.size, atmosphere.n_depth):
                raise ValueError(
                    "precomputed_lyman_opacity must have shape (wavelength, depth)"
                )
            if np.any(~np.isfinite(supplied_lyman)) or np.any(supplied_lyman < 0.0):
                raise ValueError(
                    "precomputed_lyman_opacity must contain finite nonnegative values"
                )
            line_opacity += supplied_lyman
    # The DAM/HM dissolved-level opacity is series-wide and affects the
    # Balmer/Paschen continua even when a formal spectrum intentionally omits
    # explicit Lyman lines.  Keeping it inside ``include_lyman`` made optical-
    # only magnetic spectra lose their Balmer pseudo-continuum.
    if include_series_pseudocontinuum:
        line_opacity += (
            hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                atmosphere, wavelength
            )
        )
    absorption = line_opacity + continuum_absorption
    metal_line_scattering = np.zeros_like(absorption)
    ca_ii_probabilities = None
    if metal_database is not None and metal_state is not None:
        from .metals import (
            metal_bound_free_mass_absorption_coefficient,
            metal_line_mass_absorption_coefficient,
        )

        if metal_photoionization_database is not None:
            absorption += metal_bound_free_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                metal_database,
                metal_state,
                metal_photoionization_database,
                excluded_ions=(
                    ()
                    if metal_topbase_photoionization_database is None
                    else metal_topbase_photoionization_database.ion_stages
                ),
            )
        if metal_topbase_photoionization_database is not None:
            from .d6 import topbase_bound_free_mass_absorption_coefficient

            absorption += topbase_bound_free_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                metal_state,
                metal_topbase_photoionization_database,
            )

        metal_lines = metal_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            metal_database,
            metal_state,
            mg_he_red_wing_table=mg_he_red_wing_table,
            mg_ii_he_profile_table=mg_ii_he_profile_table,
            ca_i_he_profile_table=ca_i_he_profile_table,
            ca_ii_he_profile_table=ca_ii_he_profile_table,
            ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
            minimum_oscillator_strength=minimum_metal_oscillator_strength,
            maximum_lines=maximum_metal_lines,
            excluded_elements=excluded_metal_line_elements,
        )
        absorption += metal_lines
        if ca_ii_resonance_collision_strengths is not None:
            from .cool_metal_nlte import (
                ca_ii_resonance_scattering_probabilities,
            )

            ca_ii_probabilities = ca_ii_resonance_scattering_probabilities(
                atmosphere,
                metal_database,
                ca_ii_resonance_collision_strengths,
            )
            ca_ii_crd_atmosphere = atmosphere

            def ca_ii_line_extinction(grid):
                return {
                    (lower_index, upper_index): metal_line_mass_absorption_coefficient(
                        ca_ii_crd_atmosphere,
                        grid,
                        metal_database,
                        metal_state,
                        ca_ii_he_profile_table=ca_ii_he_profile_table,
                        ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
                        minimum_oscillator_strength=(
                            minimum_metal_oscillator_strength
                        ),
                        maximum_lines=None,
                        transition_keys=(("Ca", 1, lower_index, upper_index),),
                    )
                    for lower_index, upper_index in ca_ii_probabilities
                }
        elif ca_ii_resonance_scattering_fraction > 0.0:
            ca_ii = metal_database.ions.get(("Ca", 1))
            if ca_ii is not None:
                resonance_keys = tuple(
                    ("Ca", 1, transition.lower_index, transition.upper_index)
                    for transition in ca_ii.transitions
                    if 3920.0
                    < transition.wavelength_vacuum_angstrom
                    < 3990.0
                )
                if resonance_keys:
                    ca_ii_extinction = metal_line_mass_absorption_coefficient(
                        atmosphere,
                        wavelength,
                        metal_database,
                        metal_state,
                        ca_ii_he_profile_table=ca_ii_he_profile_table,
                        ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
                        minimum_oscillator_strength=(
                            minimum_metal_oscillator_strength
                        ),
                        maximum_lines=None,
                        transition_keys=resonance_keys,
                    )
                    metal_line_scattering = (
                        ca_ii_resonance_scattering_fraction
                        * ca_ii_extinction
                    )
                    absorption -= metal_line_scattering
    scattering = scattering + metal_line_scattering
    total_opacity = absorption + scattering
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, total_opacity
    )
    planck = np.ascontiguousarray(
        planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    )
    emission_source = _ca_ii_crd_emission_source(
        atmosphere, wavelength, absorption, scattering, planck,
        ca_ii_probabilities,
        ca_ii_line_extinction if ca_ii_probabilities is not None else None,
        n_angle,
    )
    if int(transfer_depth_refinement) != transfer_depth_refinement or transfer_depth_refinement < 1:
        raise ValueError("transfer_depth_refinement must be a positive integer")
    if transfer_depth_refinement > 1:
        # Formal solution only, on subdivided structure intervals; see
        # _depth_refined_transfer_inputs.
        (
            transfer_column_mass, absorption, scattering, emission_source,
        ) = _depth_refined_transfer_inputs(
            atmosphere.column_mass, atmosphere.temperature, wavelength,
            absorption, scattering, emission_source,
            int(transfer_depth_refinement),
        )
        optical_depth = optical_depth_from_mass_opacity(
            transfer_column_mass, absorption + scattering
        )
    source, coupled, transfer_metadata = solve_spectrum_source(
        optical_depth, emission_source, absorption, scattering,
        wavelength=wavelength, n_angle=n_angle, discretization=transfer_discretization,
    )
    transfer_metadata["transfer_depth_refinement"] = int(transfer_depth_refinement)
    if transfer_discretization == "formal-pchip":
        from ._monotone_formal import CubicFormal

        flux = CubicFormal(optical_depth, n_angle).field(source)[1]
        flux_convention = "surface F_lambda"
    elif emergent_ray_mu is None:
        flux = emergent_flux(optical_depth, source, n_angle=n_angle, backend=backend)
        flux_convention = "surface F_lambda"
    else:
        from .radiative_transfer import emergent_specific_intensity

        # pi times the projected-disk mean intensity has the same surface-flux
        # convention as the angle-integrated formal solution.  Returning this
        # normalization lets magnetic disk integration average local spectra
        # with projected-area weights without changing the public Spectrum API.
        flux = PI * emergent_specific_intensity(
            optical_depth, source, emergent_ray_mu
        )
        flux_convention = "pi times line-of-sight I_lambda"
    line_series = ["Halpha-H22"]
    if include_lyman:
        line_series.insert(0, "Lyalpha-Ly21")
    if include_paschen:
        line_series.append("Paalpha-Pa22")
    if include_brackett:
        line_series.append("Bralpha-Br14")
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            "wavelength_medium": "vacuum",
            "flux_convention": flux_convention,
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "transfer": (
                "LTE absorption plus coherent-isotropic continuum/line scattering"
                if np.any(metal_line_scattering > 0.0)
                else "LTE absorption plus coherent-isotropic electron scattering"
            ),
            "continuum": continuum_description,
            "lines": ", ".join(line_series),
            "balmer_opacity_source": (
                "package LTE Balmer opacity"
                if precomputed_balmer_opacity is None
                else "caller-supplied Balmer opacity"
            ),
            "stark_profiles": "Tremblay-Bergeron 2009/2015, Doppler-convolved",
            "infrared_line_wing_windows_angstrom": (
                "Paschen 7500-25000; Brackett 13500-50000; smooth sin^2 edges"
                if include_paschen or include_brackett
                else "disabled"
            ),
            "infrared_neutral_h_broadening": (
                "not included; no dedicated Paschen/Brackett data supplied"
                if include_paschen or include_brackett
                else "disabled"
            ),
            "balmer_self_broadening": (
                (
                    "Barklem-Piskunov-O'Mara 2000 Halpha-Hgamma; "
                    "velocity-scaled smooth impact-validity taper; "
                    "Ali-Griem resonance fallback Hdelta-H22"
                    if balmer_self_broadening_prescription == "barklem"
                    else (
                        "Barklem-Piskunov-O'Mara 2000 full Halpha-Hgamma "
                        "profile grid; p-d fallback outside table; "
                        "Ali-Griem fallback Hdelta-H22"
                        if balmer_self_broadening_prescription == "barklem-grid"
                        else (
                        "Allard et al. 2008 thermally averaged Halpha impact "
                        "width; Barklem Hbeta-Hgamma; velocity-scaled smooth "
                        "impact-validity taper; Ali-Griem fallback Hdelta-H22"
                        if balmer_self_broadening_prescription == "allard-2008"
                        else "Ali-Griem resonance Halpha-H22"
                        )
                    )
                )
                if include_balmer_self_broadening
                else "disabled"
            ),
            "balmer_self_broadening_prescription": (
                balmer_self_broadening_prescription
            ),
            "balmer_self_broadening_impact_validity_fraction": (
                balmer_self_broadening_impact_validity_fraction
            ),
            "balmer_self_broadening_truncation_closure": (
                balmer_self_broadening_truncation_closure
            ),
            "unified_allard_profiles": (
                (
                    "half-Stark additive; effective-nearest; "
                    "temperature-dependent Allard H-H/H-H+ "
                    "Lyalpha-Lygamma profiles"
                    if any(
                        len(line.profiles) > 1
                        for line in unified_allard_table.lines.values()
                    )
                    else "half-Stark additive; TLUSTY205 "
                    "Allard/Koester fixed-temperature H-H/H-H+ "
                    "Lyalpha-Lygamma profiles"
                )
                if include_lyman and unified_allard_table is not None
                else "disabled"
            ),
            "jackson_lyman_profiles": (
                "Xenomorph electron+ion Stark with quasi-H2+; "
                "depth-interpolated in log(T), log(ne); Doppler-convolved"
                if include_lyman and jackson_lyman_table is not None
                else "disabled"
            ),
            "neutral_lyman_alpha_wing": (
                "Rohrmann-Althaus-Kepler 2011 H-H b-a/X-B plus H-H2 E1-E3/E1-E4 at 6000 K"
                if (
                    include_lyman
                    and include_neutral_lyman_alpha_wing
                )
                else "disabled"
            ),
            "h2_h2_collision_induced_absorption": (
                "Borysow 60--7000 K table"
                if include_molecular_absorption and h2_h2_cia_table is not None
                else "disabled"
            ),
            "dissolved_level_pseudocontinuum": (
                "series-wide DAM/Q-MHD Lyman--Brackett approximation"
                if include_series_pseudocontinuum
                else "disabled"
            ),
            "metal_abundances": (
                dict(metal_state.log_number_abundance)
                if metal_state is not None else {}
            ),
            "metal_lines": (
                metal_database.source if metal_database is not None else "disabled"
            ),
            "metal_bound_free": (
                (
                    metal_photoionization_database.source
                    if metal_photoionization_database is not None
                    else ""
                )
                + (
                    "; " + metal_topbase_photoionization_database.source
                    if metal_topbase_photoionization_database is not None
                    else ""
                )
                if (
                    metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                )
                else "disabled"
            ),
            "level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "ca_ii_resonance_scattering_fraction": (
                ca_ii_resonance_scattering_fraction
            ),
            "ca_ii_resonance_collision_strengths": (
                ca_ii_resonance_collision_strengths or "disabled"
            ),
            **transfer_metadata,
            "n_angle": int(n_angle),
            "emergent_ray_mu": emergent_ray_mu,
        },
    )


def synthesize_hydrogen_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    continuum_opacity: ArrayLike | float | None = None,
    precomputed_lyman_opacity: ArrayLike | None = None,
    include_molecular_absorption: bool = True,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
    include_balmer_self_broadening: bool = True,
    balmer_self_broadening_impact_validity_fraction: float = 1.0,
    balmer_self_broadening_prescription: str = "barklem",
    balmer_self_broadening_truncation_closure: str = "renormalize",
    barklem_self_table: BarklemSelfBroadeningTable | None = None,
    include_paschen: bool = True,
    include_brackett: bool = True,
    include_series_pseudocontinuum: bool = False,
    include_neutral_lyman_alpha_wing: bool = True,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    jackson_lyman_table: JacksonLymanProfileTable | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    ca_ii_helium_impact_scale: float = 1.0,
    ca_ii_resonance_scattering_fraction: float = 0.0,
    ca_ii_resonance_collision_strengths: str | None = None,
    include_dense_helium_metal_ionization: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-4,
    maximum_metal_lines: int | None = 20_000,
    excluded_metal_line_elements: Iterable[str] = (),
    n_angle: int = 4,
    backend: Backend = "auto",
    transfer_discretization: Literal["formal-linear", "formal-pchip"] = "formal-linear",
    transfer_depth_refinement: int = 1,
) -> Spectrum:
    """Synthesize the Lyman through Brackett series on a pure-H atmosphere."""

    return synthesize_balmer_spectrum(
        atmosphere,
        wavelength_angstrom,
        continuum_opacity=continuum_opacity,
        precomputed_lyman_opacity=precomputed_lyman_opacity,
        include_molecular_absorption=include_molecular_absorption,
        h2_h2_cia_table=h2_h2_cia_table,
        include_balmer_self_broadening=include_balmer_self_broadening,
        balmer_self_broadening_impact_validity_fraction=(
            balmer_self_broadening_impact_validity_fraction
        ),
        balmer_self_broadening_prescription=(
            balmer_self_broadening_prescription
        ),
        balmer_self_broadening_truncation_closure=(
            balmer_self_broadening_truncation_closure
        ),
        barklem_self_table=barklem_self_table,
        include_lyman=True,
        include_paschen=include_paschen,
        include_brackett=include_brackett,
        include_series_pseudocontinuum=include_series_pseudocontinuum,
        include_neutral_lyman_alpha_wing=include_neutral_lyman_alpha_wing,
        unified_allard_table=unified_allard_table,
        jackson_lyman_table=jackson_lyman_table,
        metal_database=metal_database,
        metal_abundances=metal_abundances,
        metal_photoionization_database=metal_photoionization_database,
        metal_topbase_photoionization_database=(
            metal_topbase_photoionization_database
        ),
        mg_he_red_wing_table=mg_he_red_wing_table,
        mg_ii_he_profile_table=mg_ii_he_profile_table,
        ca_i_he_profile_table=ca_i_he_profile_table,
        ca_ii_he_profile_table=ca_ii_he_profile_table,
        ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
        ca_ii_resonance_scattering_fraction=(
            ca_ii_resonance_scattering_fraction
        ),
        ca_ii_resonance_collision_strengths=(
            ca_ii_resonance_collision_strengths
        ),
        include_dense_helium_metal_ionization=include_dense_helium_metal_ionization,
        minimum_metal_oscillator_strength=minimum_metal_oscillator_strength,
        maximum_metal_lines=maximum_metal_lines,
        excluded_metal_line_elements=excluded_metal_line_elements,
        n_angle=n_angle,
        backend=backend,
        transfer_discretization=transfer_discretization,
        transfer_depth_refinement=transfer_depth_refinement,
    )


def synthesize_helium_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: object,
    helium_ii_stark_table: object | None = None,
    include_lines: bool = True,
    include_uv_resonance_lines: bool = True,
    include_helium_ii_lines: bool = True,
    include_occupation_probability: bool = True,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    neutral_broadening: Literal["none", "unsold", "montreal"] = "unsold",
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    log_hydrogen_abundance: float | None = None,
    molecular_h_he: object | None = None,
    include_trace_hydrogen_lines: bool = True,
    include_hydrogen_self_broadening: bool = True,
    include_hydrogen_neutral_helium_broadening: bool = True,
    hydrogen_self_broadening_impact_validity_fraction: float = 1.0,
    hydrogen_self_broadening_prescription: str = "barklem",
    hydrogen_self_broadening_truncation_closure: str = "renormalize",
    include_hydrogen_series_pseudocontinuum: bool = False,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    allard_stark_weight: float = 0.5,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    ca_ii_helium_impact_scale: float = 1.0,
    ca_ii_resonance_scattering_fraction: float = 0.0,
    ca_ii_resonance_collision_strengths: str | None = None,
    ca_ii_resonance_redistribution: Literal["complete", "coherent"] = "complete",
    c2_cross_section_table: object | None = None,
    include_dense_helium_metal_ionization: bool = True,
    metal_occupation_probability_partitions: bool = False,
    minimum_metal_oscillator_strength: float = 1.0e-4,
    maximum_metal_lines: int | None = 20_000,
    excluded_metal_line_elements: Iterable[str] = (),
    metal_classical_electron_stark: bool = False,
    n_angle: int = 4,
    backend: Backend = "auto",
    transfer_discretization: Literal["formal-linear", "formal-pchip", "optical-depth", "feautrier-optical-depth", "column-mass"] = "formal-linear",
    transfer_depth_refinement: int = 1,
) -> Spectrum:
    """Synthesize an LTE pure-He spectrum with tabulated He I profiles.

    ``transfer_depth_refinement`` subdivides each structure depth interval
    for the final transfer solution only (see
    :func:`_depth_refined_transfer_inputs`); opacities are still evaluated on
    the structure's own depth points.

    ``stark_table`` may be a parsed :class:`HeliumStarkTable` or a path to
    ``Beauchamp25_LD.txt``.  Requiring an explicit table keeps the optional
    CC-BY profile data out of the source distribution while making the exact
    profile provenance unambiguous.
    """

    if transfer_discretization not in ("formal-linear", "formal-pchip", "optical-depth", "feautrier-optical-depth", "column-mass"):
        raise ValueError("unsupported spectrum transfer_discretization")
    # Preserve the original explicit keyword's piecewise-linear equations.
    # A new Feautrier formal calculation must be requested by its own name.
    if transfer_discretization == "optical-depth":
        transfer_discretization = "formal-linear"
    h_state = atmosphere.hydrogen_lte_state
    molecular_state = (h_state is not None and h_state.chemical_model == "molecular-h-he-hm")
    if molecular_state != (molecular_h_he is not None):
        raise ValueError("Molecular mixed-atmosphere synthesis requires matching molecular_h_he physics and chemistry")

    from .helium import (
        helium_continuum_mass_absorption_coefficient,
        helium_i_line_mass_absorption_coefficient,
        helium_i_resonance_line_mass_absorption_coefficient,
        helium_ii_line_mass_absorption_coefficient,
        helium_rayleigh_scattering_mass_coefficient,
    )
    from .opacity import (
        balmer_mass_absorption_coefficient,
        brackett_mass_absorption_coefficient,
        electron_scattering_mass_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
        hydrogen_rayleigh_scattering_mass_coefficient,
        lyman_mass_absorption_coefficient,
        optical_depth_from_mass_opacity,
        paschen_mass_absorption_coefficient,
    )

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_angstrom must be a strictly increasing 1D array")
    metal_state = None
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError("metal photoionization requires metal_database and abundances")
    if log_hydrogen_abundance is not None and metal_database is None:
        raise ValueError("trace hydrogen currently requires a metal LTE database")
    if (
        not np.isfinite(ca_ii_resonance_scattering_fraction)
        or not 0.0 <= ca_ii_resonance_scattering_fraction <= 1.0
    ):
        raise ValueError("Ca II resonance scattering fraction must be in [0, 1]")
    if (
        ca_ii_resonance_collision_strengths is not None
        and ca_ii_resonance_scattering_fraction > 0.0
    ):
        raise ValueError(
            "choose either CHIANTI Ca II source functions or a diagnostic "
            "constant scattering fraction"
        )
    if metal_database is not None and metal_abundances is not None:
        from .metals import atmosphere_with_metal_electrons, metal_lte_state

        metal_state = metal_lte_state(
            atmosphere,
            metal_database,
            metal_abundances,
            include_dense_helium_ionization=include_dense_helium_metal_ionization,
            log_hydrogen_abundance=log_hydrogen_abundance,
            occupation_probability_partitions=metal_occupation_probability_partitions,
        )
        atmosphere = atmosphere_with_metal_electrons(atmosphere, metal_state)
    absorption = helium_continuum_mass_absorption_coefficient(
        atmosphere, wavelength, include_electron_scattering=False,
        include_rayleigh_scattering=False,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    if include_lines:
        absorption += helium_i_line_mass_absorption_coefficient(
            atmosphere, wavelength, stark_table,
            include_occupation_probability=include_occupation_probability,
            neutral_broadening=neutral_broadening,
        )
        if include_uv_resonance_lines:
            absorption += helium_i_resonance_line_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                include_occupation_probability=include_occupation_probability,
            )
        if include_helium_ii_lines:
            absorption += helium_ii_line_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                stark_table=helium_ii_stark_table,
                include_occupation_probability=include_occupation_probability,
            )
    metal_line_scattering = np.zeros_like(absorption)
    ca_ii_probabilities = None
    if metal_database is not None and metal_state is not None:
        from .metals import (
            metal_bound_free_mass_absorption_coefficient,
            metal_line_mass_absorption_coefficient,
        )

        if metal_photoionization_database is not None:
            absorption += metal_bound_free_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                metal_database,
                metal_state,
                metal_photoionization_database,
                excluded_ions=(
                    ()
                    if metal_topbase_photoionization_database is None
                    else metal_topbase_photoionization_database.ion_stages
                ),
            )
        if metal_topbase_photoionization_database is not None:
            from .d6 import topbase_bound_free_mass_absorption_coefficient

            absorption += topbase_bound_free_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                metal_state,
                metal_topbase_photoionization_database,
            )

        metal_lines = metal_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            metal_database,
            metal_state,
            mg_he_red_wing_table=mg_he_red_wing_table,
            mg_ii_he_profile_table=mg_ii_he_profile_table,
            ca_i_he_profile_table=ca_i_he_profile_table,
            ca_ii_he_profile_table=ca_ii_he_profile_table,
            ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
            minimum_oscillator_strength=minimum_metal_oscillator_strength,
            maximum_lines=maximum_metal_lines,
            excluded_elements=excluded_metal_line_elements,
            include_classical_electron_stark=metal_classical_electron_stark,
        )
        absorption += metal_lines
        if ca_ii_resonance_collision_strengths is not None:
            from .cool_metal_nlte import (
                ca_ii_resonance_scattering_probabilities,
            )

            ca_ii_probabilities = ca_ii_resonance_scattering_probabilities(
                atmosphere,
                metal_database,
                ca_ii_resonance_collision_strengths,
            )
            ca_ii_crd_atmosphere = atmosphere
            if ca_ii_resonance_redistribution not in ("complete", "coherent"):
                raise ValueError("ca_ii_resonance_redistribution must be 'complete' or 'coherent'")

            def ca_ii_line_extinction(grid):
                return {
                    (lower_index, upper_index): metal_line_mass_absorption_coefficient(
                        ca_ii_crd_atmosphere,
                        grid,
                        metal_database,
                        metal_state,
                        ca_ii_he_profile_table=ca_ii_he_profile_table,
                        ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
                        minimum_oscillator_strength=(
                            minimum_metal_oscillator_strength
                        ),
                        maximum_lines=None,
                        transition_keys=(("Ca", 1, lower_index, upper_index),),
                        include_classical_electron_stark=(
                            metal_classical_electron_stark
                        ),
                    )
                    for lower_index, upper_index in ca_ii_probabilities
                }

            if ca_ii_resonance_redistribution == "coherent":
                # Monochromatic coherent scattering with the same
                # probabilities (the pre-2026-09-30 treatment), for controls.
                for key, extinction in ca_ii_line_extinction(wavelength).items():
                    metal_line_scattering += (
                        ca_ii_probabilities[key].probability[np.newaxis, :]
                        * extinction
                    )
                absorption -= metal_line_scattering
                ca_ii_probabilities = None
        elif ca_ii_resonance_scattering_fraction > 0.0:
            ca_ii = metal_database.ions.get(("Ca", 1))
            if ca_ii is not None:
                resonance_keys = tuple(
                    ("Ca", 1, transition.lower_index, transition.upper_index)
                    for transition in ca_ii.transitions
                    if 3920.0 < transition.wavelength_vacuum_angstrom < 3990.0
                )
                if resonance_keys:
                    ca_ii_extinction = metal_line_mass_absorption_coefficient(
                        atmosphere,
                        wavelength,
                        metal_database,
                        metal_state,
                        ca_ii_he_profile_table=ca_ii_he_profile_table,
                        ca_ii_helium_impact_scale=ca_ii_helium_impact_scale,
                        minimum_oscillator_strength=(
                            minimum_metal_oscillator_strength
                        ),
                        maximum_lines=None,
                        transition_keys=resonance_keys,
                        include_classical_electron_stark=(
                            metal_classical_electron_stark
                        ),
                    )
                    metal_line_scattering = (
                        ca_ii_resonance_scattering_fraction * ca_ii_extinction
                    )
                    absorption -= metal_line_scattering
        if c2_cross_section_table is not None:
            from .carbon_molecular import c2_band_mass_absorption_coefficient

            absorption += c2_band_mass_absorption_coefficient(
                atmosphere, wavelength, metal_state, c2_cross_section_table
            )
    elif c2_cross_section_table is not None:
        raise ValueError("C2 opacity requires a carbon-bearing metal LTE state")
    if atmosphere.hydrogen_lte_state is not None:
        hydrogen_opacity = (hydrogen_continuum_mass_absorption_coefficient
            if molecular_h_he is None else molecular_h_he.hydrogen_opacity)
        absorption += hydrogen_opacity(
            atmosphere,
            wavelength,
            include_electron_scattering=False,
            include_rayleigh_scattering=False,
            include_molecular_absorption=False,
            **({} if molecular_h_he is None else dict(unified_allard_table=unified_allard_table)),
        )
        if include_trace_hydrogen_lines:
            absorption += balmer_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                include_self_broadening=include_hydrogen_self_broadening,
                include_neutral_helium_broadening=(
                    include_hydrogen_neutral_helium_broadening
                ),
                self_broadening_impact_validity_fraction=(
                    hydrogen_self_broadening_impact_validity_fraction
                ),
                self_broadening_prescription=(
                    hydrogen_self_broadening_prescription
                ),
                self_broadening_truncation_closure=(
                    hydrogen_self_broadening_truncation_closure
                ),
            )
            absorption += lyman_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                unified_allard_table=unified_allard_table,
                allard_stark_weight=allard_stark_weight,
            )
            absorption += paschen_mass_absorption_coefficient(atmosphere, wavelength)
            absorption += brackett_mass_absorption_coefficient(atmosphere, wavelength)
            if include_hydrogen_series_pseudocontinuum:
                absorption += (
                    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                        atmosphere, wavelength
                    )
                )
    scattering = (
        electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
        + helium_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
        + metal_line_scattering
    )
    if atmosphere.hydrogen_lte_state is not None:
        scattering += hydrogen_rayleigh_scattering_mass_coefficient(
            atmosphere, wavelength
        )
    total = absorption + scattering
    optical_depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    planck = np.ascontiguousarray(
        planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    )
    emission_source = _ca_ii_crd_emission_source(
        atmosphere, wavelength, absorption, scattering, planck,
        ca_ii_probabilities,
        ca_ii_line_extinction if ca_ii_probabilities is not None else None,
        n_angle,
    )
    if int(transfer_depth_refinement) != transfer_depth_refinement or transfer_depth_refinement < 1:
        raise ValueError("transfer_depth_refinement must be a positive integer")
    transfer_column_mass = atmosphere.column_mass
    if transfer_depth_refinement > 1:
        (
            transfer_column_mass, absorption, scattering, emission_source,
        ) = _depth_refined_transfer_inputs(
            atmosphere.column_mass, atmosphere.temperature, wavelength,
            absorption, scattering, emission_source,
            int(transfer_depth_refinement),
        )
        optical_depth = optical_depth_from_mass_opacity(
            transfer_column_mass, absorption + scattering
        )
    source, coupled, transfer_metadata = solve_spectrum_source(
        optical_depth, emission_source, absorption, scattering,
        wavelength=wavelength, n_angle=n_angle,
        column_mass=(transfer_column_mass if transfer_discretization == "column-mass" else None),
        discretization=("optical-depth" if transfer_discretization == "feautrier-optical-depth" else transfer_discretization),
    )
    transfer_metadata["transfer_discretization"] = transfer_discretization
    transfer_metadata["transfer_depth_refinement"] = int(transfer_depth_refinement)
    if transfer_discretization == "formal-linear":
        flux = emergent_flux(optical_depth, source, n_angle=n_angle, backend=backend)
    elif transfer_discretization == "formal-pchip":
        from ._monotone_formal import CubicFormal
        flux = CubicFormal(optical_depth, n_angle).field(source)[1]
    else:
        flux = coupled.interface_flux[:, 0]
    return Spectrum(
        wavelength_angstrom=wavelength,
        surface_flux_lambda=flux,
        metadata={
            **transfer_metadata,
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "composition": (
                "metal-polluted-helium"
                if metal_state is not None
                else "homogeneous-hydrogen-helium"
                if atmosphere.hydrogen_lte_state is not None
                else "pure-helium"
            ),
            "metal_abundances": (
                dict(metal_state.log_number_abundance)
                if metal_state is not None else {}
            ),
            "log_hydrogen_abundance": (
                log_hydrogen_abundance
                if log_hydrogen_abundance is not None
                else atmosphere.metadata.get("log_hydrogen_to_helium")
            ),
            "log_hydrogen_to_helium": atmosphere.metadata.get(
                "log_hydrogen_to_helium"
            ),
            "metal_lines": (
                metal_database.source if metal_database is not None else "disabled"
            ),
            "metal_bound_free": (
                (
                    metal_photoionization_database.source
                    if metal_photoionization_database is not None
                    else ""
                )
                + (
                    "; " + metal_topbase_photoionization_database.source
                    if metal_topbase_photoionization_database is not None
                    else ""
                )
                if (
                    metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                )
                else "disabled"
            ),
            "level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "mg_ii_he_unified_profile": (
                mg_ii_he_profile_table.source
                if mg_ii_he_profile_table is not None else "disabled"
            ),
            "mg_i_he_unified_profile": (
                mg_he_red_wing_table.source
                if mg_he_red_wing_table is not None else "disabled"
            ),
            "ca_i_he_unified_profile": (
                ca_i_he_profile_table.source
                if ca_i_he_profile_table is not None else "disabled"
            ),
            "ca_ii_he_unified_profile": (
                ca_ii_he_profile_table.source
                if ca_ii_he_profile_table is not None else "disabled"
            ),
            "ca_ii_resonance_scattering_fraction": (
                ca_ii_resonance_scattering_fraction
            ),
            "ca_ii_resonance_collision_strengths": (
                ca_ii_resonance_collision_strengths or "disabled"
            ),
            "c2_electronic_bands": (
                c2_cross_section_table.source
                if c2_cross_section_table is not None else "disabled"
            ),
            "metal_electron_feedback": (
                "charge-neutral LTE on fixed structure"
                if metal_state is not None else "disabled"
            ),
            "transfer": "LTE absorption plus coherent-isotropic scattering",
            "continuum": (
                "He I/II bound-free, He II/III and He-minus free-free, "
                "He-He+ molecular continuum, He-He-He infrared CIA"
            ),
            "lines": (
                "36 He I transitions plus hydrogenic He II series"
                if include_lines and include_helium_ii_lines
                else "36 He I transitions"
                if include_lines
                else "disabled"
            ),
            "helium_i_ground_resonance_series": bool(
                include_lines and include_uv_resonance_lines
            ),
            "helium_i_ground_resonance_broadening": (
                "Dimitrijevic-Sahal-Brechot-1989 electron/He-II impact"
                if include_lines and include_uv_resonance_lines
                else "disabled"
            ),
            "helium_ii_lines": bool(include_lines and include_helium_ii_lines),
            "helium_ii_stark_profiles": (
                "Schönning-Butler/SYNSPEC table"
                if helium_ii_stark_table is not None
                else "hydrogenic transformed fallback"
            ),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "stark_profiles": "explicit Tremblay-2026/Beauchamp-2025 table",
            "line_occupation_probability": bool(include_occupation_probability),
            "hydrogen_series_pseudocontinuum": bool(
                atmosphere.hydrogen_lte_state is not None
                and include_trace_hydrogen_lines
                and include_hydrogen_series_pseudocontinuum
            ),
            "hydrogen_neutral_helium_broadening": bool(
                atmosphere.hydrogen_lte_state is not None
                and include_trace_hydrogen_lines
                and include_hydrogen_neutral_helium_broadening
            ),
            "hydrogen_self_broadening": bool(
                atmosphere.hydrogen_lte_state is not None
                and include_trace_hydrogen_lines
                and include_hydrogen_self_broadening
            ),
            "hydrogen_self_broadening_prescription": (
                hydrogen_self_broadening_prescription
            ),
            "hydrogen_self_broadening_truncation_closure": (
                hydrogen_self_broadening_truncation_closure
            ),
            "hydrogen_allard_lyman_profiles": bool(
                atmosphere.hydrogen_lte_state is not None
                and include_trace_hydrogen_lines
                and unified_allard_table is not None
            ),
            "hydrogen_allard_stark_weight": float(allard_stark_weight),
            "neutral_helium_line_broadening": neutral_broadening,
            "neutral_helium_line_broadening_provenance": (
                "max(Unsold, Deridder-van-Rensbergen-1976); "
                "singlets max with Ali-Griem resonance; 4121/4713 Unsold-only"
                if neutral_broadening == "montreal"
                else "Unsold-1955"
                if neutral_broadening == "unsold"
                else "disabled"
            ),
            "n_angle": int(n_angle),
        },
    )


def synthesize_hydrogen_helium_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    stark_table: object,
    **kwargs: object,
) -> Spectrum:
    """Synthesize both H and He opacity on a homogeneous mixed atmosphere."""

    if (
        atmosphere.hydrogen_lte_state is None
        or atmosphere.helium_lte_state is None
    ):
        raise ValueError(
            "a homogeneous atmosphere with both H and He LTE states is required"
        )
    if "log_hydrogen_abundance" in kwargs:
        raise TypeError(
            "the H/He ratio belongs to the atmosphere EOS, not spectrum synthesis"
        )
    kwargs.setdefault("include_hydrogen_series_pseudocontinuum", True)
    if kwargs.get("metal_database") is not None:
        # Trace metals re-solve the H/He/metal charge balance at the fixed
        # H and He nuclei densities of the mixture.
        log_ratio = np.log10(
            atmosphere.hydrogen_lte_state.hydrogen_nuclei_density
            / atmosphere.helium_lte_state.helium_nuclei_density
        )
        if np.ptp(log_ratio) > 1.0e-9:
            raise ValueError("metal synthesis requires a homogeneous H/He ratio")
        kwargs["log_hydrogen_abundance"] = float(np.mean(log_ratio))
    spectrum = synthesize_helium_spectrum(
        atmosphere,
        wavelength_angstrom,
        stark_table=stark_table,
        **kwargs,
    )
    metadata = dict(spectrum.metadata)
    metadata["composition"] = (
        "metal-polluted-homogeneous-hydrogen-helium"
        if kwargs.get("metal_database") is not None
        else "homogeneous-hydrogen-helium"
    )
    metadata["log_hydrogen_to_helium"] = atmosphere.metadata.get(
        "log_hydrogen_to_helium"
    )
    return Spectrum(
        wavelength_angstrom=spectrum.wavelength_angstrom,
        surface_flux_lambda=spectrum.surface_flux_lambda,
        metadata=metadata,
    )
