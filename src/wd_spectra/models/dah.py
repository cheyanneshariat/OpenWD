"""Magnetic hydrogen-atmosphere (DAH) models on the shared structure solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..atmosphere import Atmosphere, radiative_equilibrium_hydrogen_atmosphere
from ..magnetic import (
    DAH_WAVELENGTH_RANGE_ANGSTROM,
    WEAK_FIELD_MAXIMUM_MEGAGAUSS,
    BalmerBroadening,
    MagneticPhysics,
    SurfaceCells,
    default_magnetic_da_wavelength_grid,
    dipole_surface_cells,
    h2db_component_wavelengths,
    cyclotron_structure_wavelengths,
    magnetic_structure_opacity,
    structure_uses_rwa_dissolution,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from ..magnetic_atomic import (
    read_h2db_energy_database,
    read_h2db_transition_database,
)
from .common import (
    ModelData,
    ModelResult,
    Quality,
    atmosphere_with_model_request_fingerprint,
    fixed_synthesis_atmosphere,
    model_request_fingerprint,
    numerical_resolution,
    warn_if_atmosphere_not_converged,
)
from .stellar import (
    DAConfig,
    _allard_lyman_profiles_for_effective_temperature,
    _da_self_broadening_prescription,
    compute_da,
)


# Tremblay et al. (2015) find that 1--50 kG already inhibits photospheric
# convection; Gentile Fusillo et al. (2018) need a radiative atmosphere for a
# ~43 kG DAH.
CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS = 0.05


@dataclass(frozen=True)
class DAHConfig:
    """Magnetic pure-hydrogen white dwarf.

    ``field_geometry="uniform"`` gives every surface element the field
    ``magnetic_field_megagauss`` with an unknown (isotropic) direction, or
    the fixed ``field_angle_deg`` to the line of sight.  ``"dipole"`` is an
    offset dipole with inclination ``dipole_inclination_deg`` between its
    axis and the line of sight and displacement ``dipole_offset_radius`` =
    (ax, ay, az) in stellar radii in the magnetic-axis frame; the field is
    the polar field of the undisplaced dipole (``dipole-polar``, as tabulated
    by Hardy et al. 2023) or the visible projected-area mean modulus.

    Defaults reproduce the paper prescription: a nonmagnetic DA structure,
    normalized Kurucz/Griem Balmer profiles and scalar magnetic transfer.
    ``atmosphere_structure="mean-field"`` opts into the earlier coupled
    magnetic structure experiment above 1 MG; see the DAH guide for its
    other physics switches. The equilibrium certificate always applies to
    the chosen structure physics, not to an independently relaxed surface map.

    ``mixing_length_alpha=None`` selects the automatic policy: ML2/alpha=0.7
    below 0.05 MG and a radiative atmosphere at stronger fields.
    """

    effective_temperature: float = 16_700.0
    logg: float = 8.07
    magnetic_field_megagauss: float = 0.325
    quality: Quality = "standard"
    field_geometry: Literal["uniform", "dipole"] = "uniform"
    field_angle_deg: float | None = None
    field_strength_definition: Literal["dipole-polar", "visible-mean"] = "dipole-polar"
    dipole_inclination_deg: float = 60.0
    dipole_offset_radius: tuple[float, float, float] = (0.0, 0.0, 0.0)
    disk_field_bins: int | None = 21
    atmosphere_structure: Literal["nonmagnetic", "mean-field"] = "nonmagnetic"
    balmer_profile: Literal["kurucz-griem", "unified"] = "kurucz-griem"
    normalize_balmer_strength: bool = True
    polarized_transfer: Literal["full-stokes-iquv", "scalar-stokes-i"] = (
        "scalar-stokes-i"
    )
    include_rwa_photoionization: bool = False
    # Off by default: the standard magneto-ionic treatment predicts a broad
    # cyclotron depression at 200-600 MG that J2247+1456 does not show (see
    # the DAH guide); it stays an explicit, validated-against-data option.
    include_cyclotron_absorption: bool = False
    include_magnetic_eos: bool = False
    include_centered_motion: bool = False
    mixing_length_alpha: float | None = None
    balmer_self_broadening_prescription: str | None = None


def dah_surface_cells(config: DAHConfig) -> SurfaceCells:
    """Return the visible-disk quadrature of a configuration."""

    if config.field_geometry == "uniform":
        if config.dipole_offset_radius != (0.0, 0.0, 0.0):
            raise ValueError("dipole_offset_radius applies to field_geometry='dipole'")
        return uniform_field_surface_cells(
            config.magnetic_field_megagauss, field_angle_deg=config.field_angle_deg
        )
    if config.field_geometry == "dipole":
        if config.field_angle_deg is not None:
            raise ValueError("field_angle_deg applies to field_geometry='uniform'")
        return dipole_surface_cells(
            config.magnetic_field_megagauss,
            field_strength_definition=config.field_strength_definition,
            inclination_deg=config.dipole_inclination_deg,
            offset_vector_radius=tuple(config.dipole_offset_radius),  # type: ignore[arg-type]
            n_field_bins=config.disk_field_bins,
        )
    raise ValueError("field_geometry must be 'uniform' or 'dipole'")


def structure_field_megagauss(cells: SurfaceCells) -> float:
    """Projected-area mean field modulus used for the shared structure."""

    return float(np.sum(cells.projected_weight * cells.field_strength_megagauss))


def _validate(config: DAHConfig) -> None:
    if not isinstance(config, DAHConfig):
        raise TypeError("compute_dah requires a DAHConfig")
    if not np.isfinite(config.effective_temperature) or not (
        5_000.0 <= config.effective_temperature <= 40_000.0
    ):
        raise ValueError("DAH effective_temperature must lie in 5000--40000 K")
    if not np.isfinite(config.logg) or not 7.0 <= config.logg <= 9.5:
        raise ValueError("DAH logg must lie in 7.0--9.5")
    if config.atmosphere_structure not in ("nonmagnetic", "mean-field"):
        raise ValueError("atmosphere_structure must be 'nonmagnetic' or 'mean-field'")
    if config.balmer_profile not in ("kurucz-griem", "unified"):
        raise ValueError("balmer_profile must be 'kurucz-griem' or 'unified'")
    if config.balmer_profile == "kurucz-griem" and config.balmer_self_broadening_prescription is not None:
        raise ValueError("Kurucz/Griem profiles do not include neutral self broadening")
    field = float(config.magnetic_field_megagauss)
    if not np.isfinite(field) or field < 0.0:
        raise ValueError("magnetic_field_megagauss must be finite and nonnegative")
    if config.polarized_transfer not in ("full-stokes-iquv", "scalar-stokes-i"):
        raise ValueError("polarized_transfer must be 'full-stokes-iquv' or 'scalar-stokes-i'")


def compute_dah(
    config: DAHConfig = DAHConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    iteration_callback: Callable[[int, Atmosphere, Mapping[str, object]], None]
    | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ModelResult:
    """Compute a DA structure and its disk-integrated magnetic optical spectrum.

    By default the structure is nonmagnetic, with convection suppressed above
    0.05 MG, and the magnetic line calculation uses normalized Kurucz/Griem
    kernels. Above 1 MG, H2db component strengths are normalized at each depth
    including stimulated emission, before angular weighting. No observed
    spectrum, velocity, continuum correction or fitted scale enters synthesis.

    ``atmosphere_structure="mean-field"`` instead relaxes the strong-field
    structure using the configured magnetic opacities at the projected-area
    mean field. Both modes use the shared adaptive Newton solver.
    """

    _validate(config)
    data = ModelData.default() if data is None else data
    request_fingerprint = model_request_fingerprint("DAH", config, data)
    resolution = numerical_resolution(config.quality)
    wave = (
        default_magnetic_da_wavelength_grid()
        if wavelength is None
        else np.asarray(wavelength, dtype=np.float64)
    )
    lower, upper = DAH_WAVELENGTH_RANGE_ANGSTROM
    if (
        wave.ndim != 1
        or wave.size < 2
        or np.any(~np.isfinite(wave))
        or np.any(np.diff(wave) <= 0.0)
        or wave[0] < lower
        or wave[-1] > upper
    ):
        raise ValueError(
            f"DAH synthesis requires a finite, increasing 1D wavelength grid "
            f"with at least two points in {lower:g}--{upper:g} A"
        )
    cells = dah_surface_cells(config)
    maximum_field = cells.maximum_field_megagauss
    strong = maximum_field > WEAK_FIELD_MAXIMUM_MEGAGAUSS
    structure_field = structure_field_megagauss(cells)
    mixing_length_alpha = (
        (None if structure_field >= CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS else 0.7)
        if config.mixing_length_alpha is None
        else config.mixing_length_alpha
    )
    self_broadening = _da_self_broadening_prescription(
        config.effective_temperature, config.balmer_self_broadening_prescription
    )
    broadening = BalmerBroadening(
        include_self_broadening=config.balmer_profile == "unified",
        self_broadening_prescription=self_broadening,
    )
    transitions = energies = None
    if strong:
        data.require(data.h2db_balmer_subset)
        transitions = read_h2db_transition_database(data.h2db_balmer_subset)
        energies = read_h2db_energy_database(data.h2db_balmer_subset)
    # In the normal-triplet regime the structure is the DA structure, whose
    # molecular chemistry (below 12000 K) the synthesis must also see.
    molecules = (not strong) and config.effective_temperature <= 12_000.0
    cia = None
    if molecules:
        from ..molecules import read_borysow_h2_h2_cia_table

        data.require(data.h2_h2_cia)
        cia = read_borysow_h2_h2_cia_table(data.h2_h2_cia)
    physics = MagneticPhysics(
        transitions,
        energies,
        broadening=broadening,
        balmer_profile=config.balmer_profile,
        normalize_balmer_strength=config.normalize_balmer_strength,
        include_molecular_absorption=molecules,
        h2_h2_cia_table=cia,
        include_rwa_photoionization=config.include_rwa_photoionization,
        include_cyclotron_absorption=config.include_cyclotron_absorption,
        include_magnetic_eos=config.include_magnetic_eos,
        include_centered_motion=config.include_centered_motion,
        line_regime="h2db" if strong else "linear-zeeman",
    )
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")

    structure_description: str
    if not relax_atmosphere:
        atmosphere = fixed_synthesis_atmosphere(initial_atmosphere, request_fingerprint)  # type: ignore[arg-type]
        if strong and config.include_magnetic_eos and config.atmosphere_structure == "mean-field":
            # Generic checkpoint loading rebuilds the ordinary hydrogen EOS.
            # Restore the requested mean-field chemistry before constructing
            # the shared Stark kernels, not only the cell-local populations.
            from ..magnetic_eos import atmosphere_with_magnetic_hydrogen_eos

            assert energies is not None
            atmosphere = atmosphere_with_magnetic_hydrogen_eos(
                atmosphere, structure_field, energies,
                include_centered_motion=config.include_centered_motion,
            )
        structure_description = "caller-supplied structure (not re-relaxed)"
    elif config.atmosphere_structure == "nonmagnetic" or not strong:
        base = compute_da(
            DAConfig(
                effective_temperature=config.effective_temperature,
                logg=config.logg,
                quality=config.quality,
                include_molecules=molecules,
                mixing_length_alpha=mixing_length_alpha,
                balmer_self_broadening_prescription=self_broadening,
                # DAH keeps the historical uniform DA mesh until its own
                # magnetic synthesis adopts the flux-conserving numerics.
                photospheric_depth_concentration=0.0,
            ),
            np.asarray([4_000.0, 5_000.0]),
            data=data,
            initial_atmosphere=initial_atmosphere,
            iteration_callback=iteration_callback,
        )
        atmosphere = atmosphere_with_model_request_fingerprint(
            base.atmosphere, request_fingerprint
        )
        structure_description = (
            "nonmagnetic DA structure; "
            + str(base.metadata["convection"])
        )
    else:
        assert energies is not None and transitions is not None
        from ..magnetic_eos import magnetic_hummer_mihalas_hydrogen_lte

        def magnetic_eos(temperature, pressure):
            return magnetic_hummer_mihalas_hydrogen_lte(
                temperature,
                pressure,
                structure_field,
                energies,
                include_centered_motion=config.include_centered_motion,
            )

        balmer_opacity, continuum_opacity, scattering_opacity = (
            magnetic_structure_opacity(structure_field, physics)
        )
        centers = h2db_component_wavelengths(
            structure_field, transitions, 0.8 * config.effective_temperature
        )
        centers = centers[(centers > 912.0) & (centers < 100_000.0)]
        extra = np.concatenate(
            [
                np.arange(
                    max(912.0, float(centers.min()) - 150.0),
                    min(100_000.0, float(centers.max()) + 150.0),
                    10.0,
                )
            ]
            + [center + np.arange(-4.0, 4.001, 0.2) for center in centers]
            + (
                [cyclotron_structure_wavelengths(structure_field)]
                if config.include_cyclotron_absorption
                else []
            )
        )
        extra = extra[(extra > 100.0) & (extra < 100_000.0)]
        structure_depth = 100 if config.quality == "production" else resolution.n_depth
        restart = {"initial_atmosphere": initial_atmosphere}
        if initial_atmosphere is not None and initial_atmosphere.n_depth != structure_depth:
            if initial_atmosphere.hydrogen_lte_state is None:
                raise ValueError("initial_atmosphere must be a pure-H LTE structure")
            if not np.isclose(initial_atmosphere.effective_temperature,
                              config.effective_temperature, rtol=0., atol=1e-8) or not np.isclose(
                initial_atmosphere.logg, config.logg, rtol=0., atol=1e-12
            ):
                raise ValueError("initial_atmosphere must match effective temperature and log(g)")
            # The low-level initial_atmosphere contract retains its input grid.
            # Interpolate the thermal seed onto the requested quality grid
            # instead, as the nonmagnetic DA preset does for refinement.
            restart = {
                "initial_temperature": initial_atmosphere.temperature,
                "initial_column_mass": initial_atmosphere.column_mass,
            }
        atmosphere = radiative_equilibrium_hydrogen_atmosphere(
            config.effective_temperature,
            config.logg,
            n_depth=structure_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=resolution.maximum_iterations,
            n_angle=min(resolution.n_angle, 3),
            # With the RWA the dissolved-level pseudo-continuum follows the
            # shifted edges inside the continuum hook instead.
            include_series_pseudocontinuum=not structure_uses_rwa_dissolution(
                structure_field, physics
            ),
            correlated_microfields=True,
            include_molecules=False,
            include_negative_hydrogen=False,
            unified_allard_table=_allard_lyman_profiles_for_effective_temperature(
                config.effective_temperature, data
            ),
            mixing_length_alpha=mixing_length_alpha,
            balmer_self_broadening_prescription=self_broadening,
            balmer_self_broadening_truncation_closure="stark-core",
            resolve_balmer_line_cores=False,
            **restart,
            hydrogen_eos_function=magnetic_eos
            if config.include_magnetic_eos
            else None,
            balmer_opacity_function=balmer_opacity,
            continuum_opacity_function=continuum_opacity,
            scattering_opacity_function=scattering_opacity,
            additional_structure_wavelength_angstrom=extra,
            structure_solver="adaptive-newton",
            iteration_callback=iteration_callback,
        )
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
        structure_description = (
            f"radiative structure at the {structure_field:.4g}-MG mean field: "
            + (
                "magnetic Saha/HM chemistry, "
                if config.include_magnetic_eos
                else "zero-field chemistry, "
            )
            + "H2db Halpha-H12, "
            + ("RWA bound-free, " if config.include_rwa_photoionization else "")
            + ("cyclotron, " if config.include_cyclotron_absorption else "")
            + "zero-field Lyman/Paschen/Brackett"
        )
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DAH")
    spectrum = synthesize_magnetic_hydrogen_spectrum(
        atmosphere,
        wave,
        cells,
        physics,
        polarized_transfer=config.polarized_transfer,
        n_angle=resolution.n_angle,
        progress=progress,
    )
    return ModelResult(
        "DAH",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DAH-optical-v4",
            "atmosphere_structure": structure_description,
            "structure_field_megagauss": (
                structure_field if config.atmosphere_structure == "mean-field" and strong else 0.0
            ),
            "visible_mean_field_megagauss": structure_field,
            "equilibrium_certificate_scope": (
                "shared mean-field magnetic structure"
                if config.atmosphere_structure == "mean-field" and strong
                else "nonmagnetic DA structure; magnetic spectrum is post-processed"
            ),
            "balmer_profile": config.balmer_profile,
            "normalize_balmer_strength": config.normalize_balmer_strength,
            "maximum_visible_field_megagauss": maximum_field,
            "visible_field_bounds_exact": cells.field_bounds_exact,
            "visible_field_bounds_megagauss": cells.field_bounds_megagauss,
            "maximum_synthesis_cell_field_megagauss": float(np.max(cells.field_strength_megagauss)),
            "convection": (
                "suppressed; radiative equilibrium"
                if mixing_length_alpha is None
                else f"ML2/alpha={mixing_length_alpha:g}"
            ),
            "convection_policy": (
                "automatic: radiative at B >= "
                f"{CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS:g} MG"
                if config.mixing_length_alpha is None
                else "explicit"
            ),
            "balmer_self_broadening": (
                self_broadening if config.balmer_profile == "unified" else "omitted"
            ),
            "line_physics": spectrum.metadata["magnetic_line_regime"],
            "transfer": spectrum.metadata["polarized_transfer"],
            "atmosphere_convergence_status": convergence_status,
            "model_request_fingerprint": request_fingerprint,
            "limitations": (
                "one structure shared by all surface cells; "
                "zero-field Stark profiles on H2db components; no "
                "magnetic Lyman/Paschen data; no moving-atom (decentered) "
                "states; H13+ omitted above 1 MG"
            ),
        },
    )


__all__ = [
    "CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS",
    "DAHConfig",
    "compute_dah",
    "dah_surface_cells",
    "structure_field_megagauss",
]
