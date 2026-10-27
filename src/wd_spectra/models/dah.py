"""Magnetic hydrogen-atmosphere (DAH) models on the shared structure solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..atmosphere import Atmosphere, radiative_equilibrium_hydrogen_atmosphere
from ..magnetic import (
    DAH_WAVELENGTH_RANGE_ANGSTROM,
    WEAK_FIELD_MAXIMUM_MEGAGAUSS,
    BalmerBroadening,
    balmer_component_drift_rate,
    drift_resolved_field_edges,
    field_binned_surface_cells,
    MagneticPhysics,
    SurfaceCells,
    _validate_surface_cells,
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
    H2dbTransitionDatabase,
    read_h2db_energy_database,
    read_h2db_transition_database,
)
from .common import (
    ModelData,
    atmosphere_convergence_status,
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
    _radiative_refinement_depth,
    _allard_lyman_profiles_for_effective_temperature,
    _da_self_broadening_prescription,
    compute_da,
)


# Tremblay et al. (2015) find that 1--50 kG already inhibits photospheric
# convection; Gentile Fusillo et al. (2018) need a radiative atmosphere for a
# ~43 kG DAH.
CONVECTION_SUPPRESSION_FIELD_MEGAGAUSS = 0.05
# A failed mean-field structure is re-solved on finer meshes up to this depth.
MEAN_FIELD_REFINEMENT_MAXIMUM_DEPTH = 200


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

    Defaults: a nonmagnetic DA structure, normalized Kurucz/Griem Balmer
    profiles, scalar magnetic transfer and, above 1 MG, stationary-state RWA
    photoionization (Rohrmann 2026) with the magnetic Saha/HM EOS and
    centered-motion populations evaluated on that structure in each surface
    cell. :func:`paper_dah_config` returns the 2026 paper prescription, which
    keeps the zero-field EOS and bound-free opacity.
    ``atmosphere_structure="mean-field"`` opts into the earlier coupled
    magnetic structure experiment above 1 MG; see the DAH guide for its
    other physics switches. The equilibrium certificate always applies to
    the chosen structure physics, not to an independently relaxed surface map.

    ``mixing_length_alpha=None`` selects the automatic policy: ML2/alpha=0.7
    below 0.05 MG and a radiative atmosphere at stronger fields.
    ``return_stokes=True`` requires full transfer and uncompressed dipoles;
    it returns optical-convention IQUV in a common observer sky basis.
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
    include_rwa_photoionization: bool = True
    # Off by default: the standard magneto-ionic treatment predicts a broad
    # cyclotron depression at 200-600 MG that J2247+1456 does not show (see
    # the DAH guide); it stays an explicit, validated-against-data option.
    include_cyclotron_absorption: bool = False
    include_magnetic_eos: bool = True
    include_centered_motion: bool = True
    mixing_length_alpha: float | None = None
    balmer_self_broadening_prescription: str | None = None
    # Append new fields to preserve the existing positional constructor.
    # Opt-in; takes precedence over disk_field_bins.
    disk_component_drift_angstrom: float | None = None
    # Polarimetric output uses signed, uncompressed surface geometry.
    return_stokes: bool = False
    sky_position_angle_deg: float = 0.0


def _validate_stokes_config(config: DAHConfig, *, custom_surface_cells: bool = False) -> None:
    if not np.isfinite(config.sky_position_angle_deg):
        raise ValueError("sky_position_angle_deg must be finite")
    if config.sky_position_angle_deg != 0.0 and not config.return_stokes:
        raise ValueError("sky_position_angle_deg requires return_stokes=True")
    if config.return_stokes:
        if config.polarized_transfer != "full-stokes-iquv":
            raise ValueError("return_stokes requires polarized_transfer='full-stokes-iquv'")
        if config.disk_component_drift_angstrom is not None and not custom_surface_cells:
            raise ValueError("return_stokes does not support disk_component_drift_angstrom compression")
        if (not custom_surface_cells and config.field_geometry == "dipole"
                and config.disk_field_bins is not None):
            raise ValueError("return_stokes requires disk_field_bins=None for a dipole")


# The 2026 paper comparison (Section ``sec:dah``): zero-field EOS and bound-free
# opacity on the nonmagnetic structure, i.e. the public default until 2026-10-25.
# These physics switches are fixed; the transfer mode is not part of them.
DAH_PAPER_PRESCRIPTION: Mapping[str, object] = MappingProxyType({
    "atmosphere_structure": "nonmagnetic",
    "balmer_profile": "kurucz-griem",
    "normalize_balmer_strength": True,
    "include_rwa_photoionization": False,
    "include_cyclotron_absorption": False,
    "include_magnetic_eos": False,
    "include_centered_motion": False,
})


def paper_dah_config(**parameters: object) -> DAHConfig:
    """Return a :class:`DAHConfig` with the 2026 paper prescription.

    ``parameters`` set the stellar, geometry, transfer and numerical inputs;
    they may not override the physics in :data:`DAH_PAPER_PRESCRIPTION`.
    Transfer defaults to scalar Stokes I, as in the paper figure;
    ``polarized_transfer="full-stokes-iquv"`` keeps the same physics.
    """

    fixed = sorted(set(parameters) & set(DAH_PAPER_PRESCRIPTION))
    if fixed:
        raise ValueError(f"paper_dah_config fixes {', '.join(fixed)}; use DAHConfig instead")
    return DAHConfig(**{  # type: ignore[arg-type]
        "polarized_transfer": "scalar-stokes-i", **DAH_PAPER_PRESCRIPTION, **parameters,
    })


def dah_surface_cells(
    config: DAHConfig,
    *,
    transitions: H2dbTransitionDatabase | None = None,
    wavelength_range_angstrom: tuple[float, float] = DAH_WAVELENGTH_RANGE_ANGSTROM,
) -> SurfaceCells:
    """Return the visible-disk quadrature of a configuration.

    With ``disk_component_drift_angstrom`` a dense offset-dipole grid is
    compressed using a sampled estimate of Balmer-component wavelength drift
    in ``wavelength_range_angstrom``. This is not a flux-error tolerance.
    Above 1 MG the H2db ``transitions`` (by default the bundled subset)
    supply the drift. Without a table, or below its floor, the normal-triplet
    estimate is used, matching the selected synthesis data.
    """

    _validate_stokes_config(config)
    drift = config.disk_component_drift_angstrom
    if drift is not None and (not np.isfinite(drift) or drift <= 0.0):
        raise ValueError("disk_component_drift_angstrom must be finite and positive")
    if config.field_geometry == "uniform":
        if config.dipole_offset_radius != (0.0, 0.0, 0.0):
            raise ValueError("dipole_offset_radius applies to field_geometry='dipole'")
        if config.disk_component_drift_angstrom is not None:
            raise ValueError("disk_component_drift_angstrom applies to field_geometry='dipole'")
        return uniform_field_surface_cells(
            config.magnetic_field_megagauss, field_angle_deg=config.field_angle_deg,
            signed_geometry=config.return_stokes,
            sky_position_angle_deg=config.sky_position_angle_deg,
        )
    if config.field_geometry == "dipole":
        if config.field_angle_deg is not None:
            raise ValueError("field_angle_deg applies to field_geometry='uniform'")
        if config.disk_component_drift_angstrom is not None:
            return _drift_resolved_dipole_cells(config, transitions, wavelength_range_angstrom)
        return dipole_surface_cells(
            config.magnetic_field_megagauss,
            field_strength_definition=config.field_strength_definition,
            inclination_deg=config.dipole_inclination_deg,
            offset_vector_radius=tuple(config.dipole_offset_radius),  # type: ignore[arg-type]
            n_field_bins=config.disk_field_bins,
            signed_geometry=config.return_stokes,
            sky_position_angle_deg=config.sky_position_angle_deg,
        )
    raise ValueError("field_geometry must be 'uniform' or 'dipole'")


# Dense raw quadrature for the drift-resolved disk. Distinct field bins
# require new opacity/source calculations; limb rays share those results.
DRIFT_RESOLVED_RAW_MU_NODES = 192
DRIFT_RESOLVED_RAW_AZIMUTHS = 384
DRIFT_RESOLVED_LIMB_BINS = 8


def _drift_resolved_dipole_cells(
    config: DAHConfig,
    transitions: H2dbTransitionDatabase | None,
    wavelength_range_angstrom: tuple[float, float],
) -> SurfaceCells:
    offset = np.asarray(config.dipole_offset_radius, dtype=np.float64)
    if offset.shape != (3,) or np.any(~np.isfinite(offset)) or np.linalg.norm(offset) >= 0.8:
        raise ValueError("dipole_offset_radius must be three finite values with modulus < 0.8")
    distance_scale = 1.0 - float(np.linalg.norm(offset))
    raw = dipole_surface_cells(
        config.magnetic_field_megagauss,
        field_strength_definition=config.field_strength_definition,
        inclination_deg=config.dipole_inclination_deg,
        offset_vector_radius=tuple(config.dipole_offset_radius),  # type: ignore[arg-type]
        n_mu=int(np.ceil(DRIFT_RESOLVED_RAW_MU_NODES / distance_scale)),
        n_azimuth=int(np.ceil(DRIFT_RESOLVED_RAW_AZIMUTHS / distance_scale)),
        n_field_bins=None,
    )
    if transitions is None and raw.maximum_field_megagauss > WEAK_FIELD_MAXIMUM_MEGAGAUSS:
        data = ModelData.default()
        data.require(data.h2db_balmer_subset)
        transitions = read_h2db_transition_database(data.h2db_balmer_subset)
    sampled = (
        float(np.min(raw.field_strength_megagauss)),
        float(np.max(raw.field_strength_megagauss)),
    )
    if sampled[0] == sampled[1]:
        edges = [sampled[0], np.nextafter(sampled[1], np.inf)]
        return field_binned_surface_cells(raw, edges, n_limb_bins=DRIFT_RESOLVED_LIMB_BINS)
    edges = drift_resolved_field_edges(
        sampled,
        lambda grid: balmer_component_drift_rate(grid, wavelength_range_angstrom, transitions),
        float(config.disk_component_drift_angstrom),  # type: ignore[arg-type]
    )
    return field_binned_surface_cells(raw, edges, n_limb_bins=DRIFT_RESOLVED_LIMB_BINS)


# Zhao & Stancil (2007) quantum cross sections agree with the RWA to <~0.3 dex
# through 211 MG and differ by up to 1-2 dex per transition at 235 MG
# (docs/development/history/dah-rwa-default-2026-10-25.md).
RWA_VALIDATED_FIELD_MEGAGAUSS = 200.0


# Above this, adjacent equal-weight bins can place H2db components tens of
# Angstroms apart (docs/development/history/dah-disk-quadrature-2026-10-08.md).
EQUAL_WEIGHT_BIN_COMB_FIELD_MEGAGAUSS = 100.0


def _domain_notes(config: DAHConfig, cells: SurfaceCells, strong: bool,
                  *, custom_surface_cells: bool = False) -> list[str]:
    """Documented approximations that apply to this particular request."""

    maximum = cells.maximum_field_megagauss
    notes = []
    if strong and config.include_rwa_photoionization and maximum > RWA_VALIDATED_FIELD_MEGAGAUSS:
        notes.append(
            "RWA bound-free opacity beyond its tested range: per-transition cross "
            "sections differ from quantum calculations by up to 1-2 dex at 235 MG "
            f"(validated to ~0.3 dex below {RWA_VALIDATED_FIELD_MEGAGAUSS:g} MG)"
        )
    if strong and not config.include_rwa_photoionization:
        notes.append(
            "zero-field H I bound-free and dissolved-level Balmer pseudo-continuum "
            "retained above 1 MG"
        )
    if config.disk_component_drift_angstrom is not None:
        notes.append(
            "component drift is a screened, sampled binning estimate, not a "
            "flux-error bound or validation of magnetic continuum/structure physics"
        )
    if (
        not custom_surface_cells
        and config.field_geometry == "dipole"
        and config.disk_component_drift_angstrom is None
        and config.disk_field_bins is not None
        and maximum > EQUAL_WEIGHT_BIN_COMB_FIELD_MEGAGAUSS
    ):
        notes.append(
            "equal-weight field bins can under-resolve the field spread above "
            f"{EQUAL_WEIGHT_BIN_COMB_FIELD_MEGAGAUSS:g} MG; see "
            "disk_component_drift_angstrom"
        )
    return notes


def structure_field_megagauss(cells: SurfaceCells) -> float:
    """Projected-area mean field modulus used for the shared structure."""

    return float(np.sum(cells.projected_weight * cells.field_strength_megagauss))


def _validate(config: DAHConfig, *, custom_surface_cells: bool = False) -> None:
    if not isinstance(config, DAHConfig):
        raise TypeError("compute_dah requires a DAHConfig")
    _validate_stokes_config(config, custom_surface_cells=custom_surface_cells)
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
    drift = config.disk_component_drift_angstrom
    if custom_surface_cells and drift is not None:
        raise ValueError("custom surface_cells do not support disk_component_drift_angstrom compression")
    if drift is not None and (not np.isfinite(drift) or drift <= 0.0):
        raise ValueError("disk_component_drift_angstrom must be finite and positive")
    if drift is not None and config.field_geometry != "dipole":
        raise ValueError("disk_component_drift_angstrom applies to field_geometry='dipole'")


def _snapshot_surface_cells(cells: SurfaceCells, config: DAHConfig) -> SurfaceCells:
    """Own immutable numeric inputs so callbacks cannot change a recorded map."""

    if not isinstance(cells, SurfaceCells):
        raise TypeError("surface_cells must be a SurfaceCells instance")
    arrays = {}
    for name in ("field_strength_megagauss", "field_ray_cosine", "ray_mu",
                 "projected_weight", "field_azimuth_rad"):
        value = getattr(cells, name)
        if name == "field_azimuth_rad" and value is None:
            arrays[name] = None
            continue
        if np.iscomplexobj(value):
            raise ValueError("surface cell arrays must be real")
        try:
            array = np.array(value, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("surface cell arrays must be real numeric arrays") from error
        if name == "field_azimuth_rad":
            array += np.deg2rad(config.sky_position_angle_deg % 360.)
        array.setflags(write=False)
        arrays[name] = array
    bounds = cells.field_bounds_megagauss
    if bounds is not None:
        if np.iscomplexobj(bounds):
            raise ValueError("field bounds must be real")
        try:
            array = np.asarray(bounds, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("field bounds must be two real numbers") from error
        if array.shape != (2,):
            raise ValueError("field bounds must be two real numbers")
        bounds = tuple(float(x) for x in array)
    snapshot = replace(cells, **arrays, field_bounds_megagauss=bounds,
                       field_bounds_exact=bool(cells.field_bounds_exact))
    _validate_surface_cells(snapshot, require_azimuth=config.return_stokes)
    return snapshot


def _surface_cells_sha256(cells: SurfaceCells, *, structure_only: bool = False) -> str:
    """Separate the shared structure inputs from the complete transfer map."""

    descriptor = dict(schema="dah-custom-structure-v1" if structure_only else "dah-custom-map-v1",
        cells=int(cells.projected_weight.size), bounds=cells.field_bounds_megagauss,
        bounds_exact=cells.field_bounds_exact)
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False).encode())
    names = ["field_strength_megagauss", "projected_weight"]
    if not structure_only:
        names += ["field_ray_cosine", "ray_mu", "field_azimuth_rad"]
    for name in names:
        digest.update(name.encode("ascii") + b"\0")
        value = getattr(cells, name)
        digest.update(b"none" if value is None else np.asarray(value, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _structure_request_config(config: DAHConfig, *, custom_surface_cells: bool) -> DAHConfig:
    options = dict(polarized_transfer="scalar-stokes-i", return_stokes=False, sky_position_angle_deg=0.)
    if custom_surface_cells:
        # A supplied map replaces the built-in geometry controls. The actual
        # field distribution/weights/bounds are identified separately below.
        defaults = DAHConfig()
        for name in ("magnetic_field_megagauss", "field_geometry", "field_angle_deg",
                     "field_strength_definition", "dipole_inclination_deg", "dipole_offset_radius",
                     "disk_field_bins", "disk_component_drift_angstrom"):
            options[name] = getattr(defaults, name)
    return replace(config, **options)


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
    surface_cells: SurfaceCells | None = None,
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

    ``surface_cells`` supplies a custom visible-disk quadrature in place of
    the built-in geometry controls. Its actual fields, projected weights and
    bounds determine the shared structure; its ray directions and azimuths
    determine the spectrum. ``sky_position_angle_deg`` additionally rotates
    its supplied sky azimuths. Polarized output requires those azimuths.
    Drift compression is unsupported for custom maps.
    """

    custom_geometry = surface_cells is not None
    _validate(config, custom_surface_cells=custom_geometry)
    custom_cells = _snapshot_surface_cells(surface_cells, config) if surface_cells is not None else None
    data = ModelData.default() if data is None else data
    # The structure (nonmagnetic or angle-averaged mean field) never depends on
    # the transfer mode, returned Stokes columns or observer sky axis. Retain
    # the actual opacity/EOS choices while sharing one structure claim.
    identity_options = {} if custom_cells is None else {
        "physical_data_identity": {
            "dah_custom_structure_sha256": _surface_cells_sha256(custom_cells, structure_only=True)
        }
    }
    request_fingerprint = model_request_fingerprint("DAH",
        _structure_request_config(config, custom_surface_cells=custom_geometry), data, **identity_options)
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
    # The continuous field bounds (and so the atomic regime) do not depend on
    # the compression; a drift-resolved disk is built once the data are read.
    cells = (custom_cells if custom_cells is not None
             else dah_surface_cells(replace(config, disk_component_drift_angstrom=None)))
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
    if (not custom_geometry and config.disk_component_drift_angstrom is not None
            and config.field_geometry == "dipole"):
        # Components just outside the output window still contribute wings.
        cells = dah_surface_cells(
            config,
            transitions=transitions,
            wavelength_range_angstrom=(max(0.0, wave[0] - 100.0), wave[-1] + 100.0),
        )
        structure_field = structure_field_megagauss(cells)
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
        def solve(n_depth: int, seed: dict[str, object]) -> Atmosphere:
            return radiative_equilibrium_hydrogen_atmosphere(
                config.effective_temperature,
                config.logg,
                n_depth=n_depth,
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
                **seed,
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
        atmosphere = solve(structure_depth, restart)
        refinements = 0
        depth = structure_depth
        while (
            config.quality != "quick"
            and _radiative_refinement_depth(depth) <= MEAN_FIELD_REFINEMENT_MAXIMUM_DEPTH
            and atmosphere_convergence_status(atmosphere) != "converged"
        ):
            # As for the radiative DA: a steep ionization transition can stall
            # the initial mesh. Re-solve the same equations on a finer mesh
            # (40 -> 100 -> 200), seeded by the unfinished state, which must
            # not certify itself.
            coarse = atmosphere
            refinements += 1
            depth = _radiative_refinement_depth(depth)
            atmosphere = solve(depth, {
                "initial_temperature": coarse.temperature,
                "initial_column_mass": coarse.column_mass,
            })
            atmosphere = replace(atmosphere, metadata={
                **atmosphere.metadata,
                "radiative_depth_refinement": {
                    "initial_depth_points": structure_depth,
                    "final_depth_points": atmosphere.n_depth,
                    "refinements": refinements,
                    "previous_certificate_failures": coarse.metadata.get(
                        "equilibrium_certificate", {}
                    ).get("failures"),
                    "previous_maximum_flux_residual": coarse.metadata.get(
                        "maximum_all_depth_total_flux_residual"
                    ),
                    "stellar_parameters_changed": False,
                },
            })
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
        return_stokes=config.return_stokes,
    )
    geometry_metadata = {}
    if custom_cells is not None:
        geometry_metadata = {
            "surface_geometry_source": "surface_cells",
            "surface_geometry_sha256": _surface_cells_sha256(cells),
            "structure_geometry_sha256": _surface_cells_sha256(cells, structure_only=True),
        }
        spectrum = replace(spectrum, metadata={**spectrum.metadata, **geometry_metadata,
            "surface_field_ray_cosine": cells.field_ray_cosine.tolist(),
            "surface_ray_mu": cells.ray_mu.tolist(),
            "surface_field_azimuth_rad": (None if cells.field_azimuth_rad is None
                                          else cells.field_azimuth_rad.tolist()),
            "surface_field_bounds_megagauss": cells.field_bounds_megagauss,
            "surface_field_bounds_exact": cells.field_bounds_exact,
        })
    if custom_geometry:
        disk_quadrature = f"{cells.projected_weight.size} caller-supplied surface cells"
    elif config.disk_component_drift_angstrom is not None:
        disk_quadrature = (
            f"{np.unique(cells.field_strength_megagauss).size} field bins with Balmer-component drift "
            f"target {config.disk_component_drift_angstrom:g} A (estimated); "
            f"{cells.projected_weight.size} limb rays"
        )
    elif config.field_geometry == "dipole" and config.disk_field_bins is not None:
        disk_quadrature = f"{cells.projected_weight.size} equal-weight field bins"
    else:
        disk_quadrature = f"{cells.projected_weight.size} cells"
    return ModelResult(
        "DAH",
        atmosphere,
        spectrum,
        config,
        {
            **geometry_metadata,
            "preset": "DAH-optical-stokes-v1" if config.return_stokes else "DAH-optical-v4",
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
            "disk_distinct_field_bins": int(np.unique(cells.field_strength_megagauss).size),
            "disk_limb_bins_per_field": (
                DRIFT_RESOLVED_LIMB_BINS if config.disk_component_drift_angstrom is not None else None
            ),
            "disk_quadrature": disk_quadrature,
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
            "domain_notes": _domain_notes(config, cells, strong, custom_surface_cells=custom_geometry),
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
    "DAH_PAPER_PRESCRIPTION",
    "DAHConfig",
    "RWA_VALIDATED_FIELD_MEGAGAUSS",
    "compute_dah",
    "dah_surface_cells",
    "paper_dah_config",
    "structure_field_megagauss",
]
