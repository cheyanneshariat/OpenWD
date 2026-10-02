"""High-level DA, DB, DAB, and DZ model presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike
from .._domain import solve_with_screened_boundary

from ..atmosphere import (
    Atmosphere,
    radiative_equilibrium_helium_atmosphere,
    radiative_equilibrium_hydrogen_atmosphere,
    radiative_equilibrium_hydrogen_helium_atmosphere,
)
from ..cool_da import (
    atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
)
from ..dense_eos import read_helium_reos3_table
from ..helium_ii_stark import read_helium_ii_stark_table
from ..helium_stark import read_helium_stark_table
from ..metals import (
    NIST_ASD_STRONG_ION_FILES,
    read_barklem_neutral_hydrogen_broadening,
    read_ca_i_he_profile_table,
    read_mg_he_red_wing_table,
    read_nist_asd_strong_atomic_database,
    read_stout_atomic_database,
    read_verner_photoionization_database,
    read_verner_phfit2_database,
    VernerPhotoionizationDatabase,
)
from ..molecules import read_borysow_h2_h2_cia_table
from ..quasimolecular import (
    AllardUnifiedLymanTable,
    read_allard_temperature_dependent_tables,
    read_allard_tlusty205_tables,
)
from ..spectrum import (
    synthesize_helium_spectrum,
    synthesize_hydrogen_helium_spectrum,
    synthesize_hydrogen_spectrum,
)
from .common import (
    ModelData,
    ModelResult,
    Quality,
    atmosphere_convergence_status,
    atmosphere_matches_model_request,
    atmosphere_with_model_request_fingerprint,
    fixed_synthesis_atmosphere,
    model_request_fingerprint,
    numerical_resolution,
    validate_wavelength,
    warn_if_atmosphere_not_converged,
)


@dataclass(frozen=True)
class DAConfig:
    """Best current homogeneous hydrogen-atmosphere preset."""

    effective_temperature: float = 12_000.0
    logg: float = 8.0
    quality: Quality = "standard"
    lyman_profile_source: Literal["allard", "stark"] = "allard"
    allard_minimum_effective_temperature: float = 9_000.0
    include_molecules: bool | None = None
    h3plus_partition_model: Literal[
        "neale-tennyson-1995", "none"
    ] = "neale-tennyson-1995"
    mixing_length_alpha: float | None = 0.7
    use_cool_mean_3d_temperature_differential: bool = False
    balmer_self_broadening_prescription: str | None = None
    balmer_self_broadening_truncation_closure: str = "stark-core"
    atmosphere_solver: Literal["adaptive-newton"] = (
        "adaptive-newton"
    )
    multigrid_initialization: bool = False
    # Flux-conservation numerics (see DZ): photosphere-concentrated structure
    # depths and a depth-refined final formal solution.  0 and 1 restore the
    # historical uniform depths and bare-grid formal solution.  None selects
    # 3 for the 40-layer standard mesh and 1 for 100 or more layers: with 40
    # layers the steep 10-12 kK hydrogen photosphere still left 0.5% excess
    # flux at concentration 1 (0.2-0.3% at 3), while 100 layers already
    # resolve it and the stronger setting only added solver iterations.
    photospheric_depth_concentration: float | None = None
    synthesis_transfer_depth_refinement: int = 4


@dataclass(frozen=True)
class DBConfig:
    """Best current homogeneous LTE helium-atmosphere preset."""

    effective_temperature: float = 20_000.0
    logg: float = 8.0
    quality: Quality = "standard"
    mixing_length_alpha: float = 1.25
    atmosphere_solver: Literal["adaptive-newton"] = "adaptive-newton"
    neutral_broadening: Literal["unsold", "montreal", "none"] = "unsold"
    # Flux-conservation numerics (see DZ): photosphere-concentrated structure
    # depths and a depth-refined final formal solution.  0 and 1 restore the
    # historical uniform depths and bare-grid formal solution.
    photospheric_depth_concentration: float = 1.0
    synthesis_transfer_depth_refinement: int = 4


@dataclass(frozen=True)
class DABConfig:
    """Best current homogeneous atomic H/He atmosphere preset."""

    effective_temperature: float = 20_000.0
    logg: float = 8.0
    log_hydrogen_to_helium: float = -2.0
    quality: Quality = "standard"
    mixing_length_alpha: float = 1.25
    atmosphere_solver: Literal["adaptive-newton"] = "adaptive-newton"
    lyman_profile_source: Literal["allard", "stark"] = "allard"
    allard_minimum_effective_temperature: float = 9_000.0
    balmer_self_broadening_prescription: str | None = None
    balmer_self_broadening_truncation_closure: str = "stark-core"
    neutral_broadening: Literal["unsold", "montreal", "none"] = "unsold"
    include_molecules: bool = False
    h2_he_cia_path: str | None = None
    # Flux-conservation numerics (see DZ): photosphere-concentrated structure
    # depths and a depth-refined final formal solution.  0 and 1 restore the
    # historical uniform depths and bare-grid formal solution.
    photospheric_depth_concentration: float = 1.0
    synthesis_transfer_depth_refinement: int = 4


GD40_ABUNDANCES = MappingProxyType(
    {
        "C": -7.20,
        "O": -5.61,
        "Mg": -6.24,
        "Si": -6.76,
        "Ca": -6.88,
        "Ti": -8.61,
        "Cr": -8.31,
        "Mn": -8.62,
        "Fe": -6.48,
    }
)



@dataclass(frozen=True)
class DZConfig:
    """Warm helium-dominated polluted atmosphere preset.

    The defaults reproduce the published GD 40 composition used by the
    multi-object SPY validation suite.  Abundances are log10 number ratios to
    helium.
    """

    effective_temperature: float = 15_300.0
    logg: float = 8.0
    abundances: Mapping[str, float] = field(
        default_factory=lambda: dict(GD40_ABUNDANCES)
    )
    log_hydrogen_abundance: float | None = -6.16
    quality: Quality = "standard"
    neutral_broadening: Literal["unsold", "montreal", "none"] = "unsold"
    mixing_length_alpha: float | None = 1.25
    atmosphere_solver: Literal["adaptive-newton"] = "adaptive-newton"
    maximum_metal_charge: int = 3
    # Flux conservation: the structure absorbs the synthesis line list
    # (opacity-sampled), its 40 depths are concentrated across the
    # photosphere, and the formal solution subdivides each depth interval.
    # ``structure_opacity_sampling_resolution=None`` restores the historical
    # line-centered 1000-line structure (with its own f >= 0.01 cut).
    structure_maximum_metal_lines: int | None = None
    structure_minimum_metal_oscillator_strength: float | None = None
    structure_opacity_sampling_resolution: float | None = 1000.0
    photospheric_depth_concentration: float = 1.0
    formal_maximum_metal_lines: int | None = None
    formal_minimum_metal_oscillator_strength: float = 1.0e-4
    synthesis_transfer_depth_refinement: int = 4
    lyman_profile_source: Literal["allard", "stark"] = "stark"
    allard_minimum_effective_temperature: float = 9_000.0
    balmer_self_broadening_prescription: str | None = None
    balmer_self_broadening_truncation_closure: str = "stark-core"
    ca_ii_resonance_source: Literal["chianti-reduced", "lte"] = (
        "chianti-reduced"
    )
    unified_metal_helium_profiles: Literal[
        "production", "legacy", "off"
    ] = "production"
    dense_helium_eos: Literal["reos3", "ideal"] = "ideal"
    # "nist-asd" replaces exactly matched Stout transitions (UV included)
    # with evaluated NIST values; the default keeps the paper-figure data.
    strong_line_atomic_data: Literal["stout", "nist-asd"] = "stout"
    # Opacity-Project photoionization from every tabulated level of the
    # bundled TLUSTY/SIROCCO model atoms, in place of Verner ground-state fits
    # for those ions.  Excited-level continua (Mg I 3P, Si I 1D, ...) carry
    # much of the near-UV bound-free opacity of cool DZ atmospheres.
    level_resolved_photoionization: bool = True
    # Hummer--Mihalas occupation-probability metal partition functions
    # (Q-MHD charged microfields plus neutral-He excluded volume), with the
    # same level survival applied to the line opacity.
    occupation_probability_metal_partitions: bool = True


_DA_ALI_GRIEM_TEMPERATURE_CUTOFF_K = 10_000.0
_ALLARD_MINIMUM_EFFECTIVE_TEMPERATURE_K = 9_000.0
_ALLARD_TEMPERATURE_DEPENDENT_CUTOFF_K = 13_000.0


def _metal_structure_line_budget(
    quality: Quality,
    abundances: Mapping[str, float],
    requested: int | None,
) -> tuple[int, str, float]:
    """Resolve a composition-aware structural metal-line budget.

    Ordinary trace-polluted atmospheres retain the established 1000-line
    standard budget.  Only mixtures whose summed metal number fraction makes
    line blanketing intrinsically non-trace receive a larger budget.  The
    thresholds are global composition criteria, not target names or fit
    residuals; an explicit caller value remains available for convergence
    studies.
    """

    metal_number_fraction = float(
        sum(10.0 ** float(value) for value in abundances.values())
    )
    if requested is not None:
        budget = int(requested)
        policy = "explicit override"
    elif quality == "quick":
        budget = 300
        policy = "quick fixed budget"
    elif metal_number_fraction >= 1.0e-4:
        budget = 8_000
        policy = "adaptive: extreme pollution"
    elif metal_number_fraction >= 3.0e-5:
        budget = 4_000
        policy = "adaptive: strong pollution"
    else:
        budget = 1_000
        policy = "adaptive: trace pollution"
    if budget < 1:
        raise ValueError("metal line limits must be positive")
    return budget, policy, metal_number_fraction


def _da_self_broadening_prescription(
    effective_temperature: float, requested: str | None
) -> str:
    """Resolve the public DA neutral-broadening policy.

    Observed cool-DA profiles favor Ali--Griem below 10,000 K.  The Balmer
    lines become weak at the very cool end, while the neutral-H contribution
    becomes negligible toward the warm regime.  Keep Barklem at and above the
    boundary, and retain an explicit caller override for controlled tests.
    """

    if requested is not None:
        return requested
    return (
        "ali-griem"
        if effective_temperature < _DA_ALI_GRIEM_TEMPERATURE_CUTOFF_K
        else "barklem"
    )


def _allard_lyman_profiles_for_effective_temperature(
    effective_temperature: float,
    data: ModelData,
    *,
    minimum_effective_temperature: float = (
        _ALLARD_MINIMUM_EFFECTIVE_TEMPERATURE_K
    ),
    use_fixed_low_temperature_profile: bool = True,
):
    """Resolve the validated temperature-dependent Lyman-profile policy.

    The current Ly-alpha delivery begins at 9000 K.  Clamping cooler models
    to that file and extending its final tabulated point through the optical
    suppresses the independently calibrated cool neutral-H wing and produces
    an artificial optical colour term.  Returning ``None`` below the stated
    boundary leaves the cool H--H/H2 treatment active.

    For the DA preset, between 9000 and 13,000 K use the checksum-pinned
    TLUSTY205 Allard Ly-alpha profile alone. Direct fixed-structure controls
    against the Koester DA grid show that the delivered cool Ly-alpha
    generation over-absorbs the 1300--1600 A wing, while the delivered
    Ly-beta and Ly-gamma grids do not extend through this regime. Mixed H/He
    atmospheres instead set ``use_fixed_low_temperature_profile=False``:
    their Warwick/Montreal reference grid uses the delivered Allard profiles,
    including the explicitly tabulated 9000-K Ly-alpha calculation. At and
    above 13,000 K, both presets use the temperature-dependent delivery.
    """

    if (
        not np.isfinite(minimum_effective_temperature)
        or minimum_effective_temperature <= 0.0
    ):
        raise ValueError(
            "allard_minimum_effective_temperature must be finite and positive"
        )
    if effective_temperature < minimum_effective_temperature:
        return None
    if (
        use_fixed_low_temperature_profile
        and effective_temperature < _ALLARD_TEMPERATURE_DEPENDENT_CUTOFF_K
    ):
        return _low_temperature_allard_lyman_table(data)
    data.require(data.allard_lyman)
    return read_allard_temperature_dependent_tables(data.allard_lyman)


def _low_temperature_allard_lyman_table(
    data: ModelData,
) -> AllardUnifiedLymanTable:
    """Return fixed Allard Ly-alpha with Stark-only higher Lyman members."""

    data.require(
        data.allard_tlusty205,
        fetch_command="python scripts/fetch_allard_tlusty205.py",
    )
    complete = read_allard_tlusty205_tables(data.allard_tlusty205)
    transition = (1, 2)
    return AllardUnifiedLymanTable(
        lines={transition: complete.lines[transition]},
        source_directory=complete.source_directory,
        source_sha256={transition: complete.source_sha256[transition]},
    )


def _helium_tables(data: ModelData):
    data.require(
        data.helium_i_stark,
        data.helium_ii_stark,
        fetch_command="python scripts/fetch_helium_stark.py",
    )
    return (
        read_helium_stark_table(data.helium_i_stark),
        read_helium_ii_stark_table(data.helium_ii_stark),
    )


def compute_da(
    config: DAConfig = DAConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    synthesis_transfer: Literal["formal-linear", "formal-pchip"] = "formal-pchip",
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> ModelResult:
    """Calculate a DA with cubic synthesis and the established atmosphere solve."""

    if synthesis_transfer not in ("formal-linear", "formal-pchip"):
        raise ValueError("unsupported DA synthesis_transfer")
    data = ModelData.default() if data is None else data
    request_fingerprint = model_request_fingerprint("DA", config, data)
    resolution = numerical_resolution(config.quality)
    wave = validate_wavelength(wavelength)
    allard = None
    if config.lyman_profile_source == "allard":
        allard = _allard_lyman_profiles_for_effective_temperature(
            config.effective_temperature,
            data,
            minimum_effective_temperature=(
                config.allard_minimum_effective_temperature
            ),
        )
    elif config.lyman_profile_source != "stark":
        raise ValueError("lyman_profile_source must be 'allard' or 'stark'")
    molecules = (
        config.effective_temperature <= 12_000.0
        if config.include_molecules is None
        else config.include_molecules
    )
    self_broadening_prescription = _da_self_broadening_prescription(
        config.effective_temperature, config.balmer_self_broadening_prescription
    )
    cia = None
    if molecules:
        data.require(
            data.h2_h2_cia,
            fetch_command="python scripts/fetch_molecular_data.py",
        )
        cia = read_borysow_h2_h2_cia_table(data.h2_h2_cia)
    h3plus_partition_model = (
        None
        if config.h3plus_partition_model == "none"
        else config.h3plus_partition_model
    )
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    atmosphere_initialization = (
        "caller checkpoint" if initial_atmosphere is not None else "gray"
    )
    if relax_atmosphere:
        relaxation_kwargs = dict(
            include_series_pseudocontinuum=True,
            correlated_microfields=True,
            include_molecules=molecules,
            include_negative_hydrogen=molecules,
            trihydrogen_ion_partition_model=h3plus_partition_model,
            h2_h2_cia_table=cia,
            unified_allard_table=allard,
            mixing_length_alpha=config.mixing_length_alpha,
            balmer_self_broadening_prescription=(
                self_broadening_prescription
            ),
            balmer_self_broadening_truncation_closure=(
                config.balmer_self_broadening_truncation_closure
            ),
            structure_solver=config.atmosphere_solver,
        )
        full_depth = (
            100 if config.quality == "production" else resolution.n_depth
        )
        relaxation_kwargs["depth_concentration"] = (
            config.photospheric_depth_concentration
            if config.photospheric_depth_concentration is not None
            else (1.0 if full_depth >= 100 else 3.0)
        )
        refine_radiative_grid = (
            config.quality == "production"
            and config.mixing_length_alpha is None
            and config.atmosphere_solver == "adaptive-newton"
        )
        if refine_radiative_grid and atmosphere is not None:
            # Retain a previously refined warm start. Compressing its steep
            # ionization transition back onto 100 layers can recreate failure.
            full_depth = max(full_depth, atmosphere.n_depth)
        if (
            initial_atmosphere is None
            and config.multigrid_initialization
            and config.atmosphere_solver == "adaptive-newton"
            and full_depth > 12
            and config.quality != "quick"
        ):
            coarse_depths = tuple(depth for depth in (12,) if depth < full_depth)
            for coarse_depth in coarse_depths:
                def coarse_callback(
                    iteration,
                    current_atmosphere,
                    status,
                    *,
                    _depth=coarse_depth,
                ):
                    if iteration_callback is None:
                        return
                    record = dict(status)
                    record["coarse_grid"] = True
                    record["multigrid_depth"] = _depth
                    iteration_callback(iteration, current_atmosphere, record)

                atmosphere = radiative_equilibrium_hydrogen_atmosphere(
                    config.effective_temperature,
                    config.logg,
                    n_depth=coarse_depth,
                    n_continuum_wavelength=min(
                        120 if coarse_depth == 12 else 300,
                        resolution.n_continuum,
                    ),
                    max_iterations=min(
                        60 if coarse_depth == 12 else 120,
                        resolution.maximum_iterations,
                    ),
                    n_angle=min(
                        resolution.n_angle,
                        2 if coarse_depth == 12 else 3,
                    ),
                    initial_temperature=(
                        None if atmosphere is None else atmosphere.temperature
                    ),
                    initial_column_mass=(
                        None if atmosphere is None else atmosphere.column_mass
                    ),
                    iteration_callback=coarse_callback,
                    **relaxation_kwargs,
                )
            atmosphere_initialization = (
                "automatic same-physics "
                + "/".join(str(depth) for depth in coarse_depths)
                + "-depth multigrid seed"
            )

        iteration_offset = 0
        depth_refinements = 0

        def full_callback(iteration, current_atmosphere, status):
            if iteration_callback is None:
                return
            record = dict(status)
            record["coarse_grid"] = False
            record["depth_points"] = current_atmosphere.n_depth
            record["radiative_depth_refinements"] = depth_refinements
            iteration_callback(iteration + iteration_offset, current_atmosphere, record)

        atmosphere = radiative_equilibrium_hydrogen_atmosphere(
            config.effective_temperature,
            config.logg,
            n_depth=full_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=(
                min(40, resolution.maximum_iterations)
                if refine_radiative_grid and full_depth == 100
                else resolution.maximum_iterations
            ),
            n_angle=min(resolution.n_angle, 3),
            initial_temperature=(
                None if atmosphere is None else atmosphere.temperature
            ),
            initial_column_mass=(
                None if atmosphere is None else atmosphere.column_mass
            ),
            iteration_callback=full_callback,
            **relaxation_kwargs,
        )
        if (
            refine_radiative_grid
            and full_depth == 100
            and atmosphere_convergence_status(atmosphere) != "converged"
        ):
            # A radiative hydrogen ionization transition can be too steep for
            # the initial mass mesh. Re-solve the SAME equations on 200 layers;
            # the unfinished 100-layer state is only an interpolated seed.
            # The fine solve must earn its own complete equilibrium certificate.
            coarse = atmosphere
            iteration_offset = int(coarse.metadata["radiative_equilibrium_iterations"])
            depth_refinements = 1
            atmosphere = radiative_equilibrium_hydrogen_atmosphere(
                config.effective_temperature,
                config.logg,
                n_depth=2 * full_depth,
                n_continuum_wavelength=resolution.n_continuum,
                max_iterations=resolution.maximum_iterations,
                n_angle=min(resolution.n_angle, 3),
                initial_temperature=coarse.temperature,
                initial_column_mass=coarse.column_mass,
                iteration_callback=full_callback,
                **relaxation_kwargs,
            )
            atmosphere = replace(atmosphere, metadata={
                **atmosphere.metadata,
                "radiative_depth_refinement": {
                    "initial_depth_points": coarse.n_depth,
                    "final_depth_points": atmosphere.n_depth,
                    "initial_certificate_failures": coarse.metadata["equilibrium_certificate"]["failures"],
                    "initial_maximum_flux_residual": coarse.metadata["maximum_all_depth_total_flux_residual"],
                    "initial_maximum_local_energy_residual": coarse.metadata["maximum_relative_cell_energy_balance_residual"],
                    "initial_iterations": iteration_offset,
                    "stellar_parameters_changed": False,
                },
                "radiative_equilibrium_iterations_including_depth_refinement": (
                    iteration_offset + atmosphere.metadata["radiative_equilibrium_iterations"]
                ),
            })
    assert atmosphere is not None
    if config.use_cool_mean_3d_temperature_differential:
        atmosphere = (
            atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
                atmosphere,
                neutral_radius_scale=0.5,
            )
        )
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DA")
    spectrum = synthesize_hydrogen_spectrum(
        atmosphere,
        wave,
        include_molecular_absorption=molecules,
        h2_h2_cia_table=cia,
        include_series_pseudocontinuum=True,
        unified_allard_table=allard,
        balmer_self_broadening_prescription=(
            self_broadening_prescription
        ),
        balmer_self_broadening_truncation_closure=(
            config.balmer_self_broadening_truncation_closure
        ),
        n_angle=resolution.n_angle,
        transfer_discretization=synthesis_transfer,
        transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
    )
    low_temperature_allard = (
        allard is not None
        and config.effective_temperature
        < _ALLARD_TEMPERATURE_DEPENDENT_CUTOFF_K
    )
    return ModelResult(
        "DA",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DA-production-v3",
            "microfields": "Q-MHD",
            "hydrogen_series": "Lyman through Brackett with dissolution",
            "balmer_self_broadening": (
                self_broadening_prescription
            ),
            "balmer_self_broadening_policy": (
                "automatic: Ali-Griem below 10000 K; Barklem otherwise"
                if config.balmer_self_broadening_prescription is None
                else "explicit override"
            ),
            "balmer_self_broadening_truncation_closure": (
                config.balmer_self_broadening_truncation_closure
            ),
            "lyman_profiles": (
                (
                    "TLUSTY205 Allard Lyalpha + charged-particle Stark "
                    "higher Lyman series"
                    if low_temperature_allard
                    else (
                        "temperature-dependent Allard separate-perturber "
                        "+ half Stark"
                    )
                )
                if allard is not None
                else (
                    "cool neutral-H/H2 Lyalpha wing + "
                    "Tremblay-Bergeron charged-particle Stark"
                    if config.lyman_profile_source == "allard"
                    else "Tremblay-Bergeron charged-particle Stark"
                )
            ),
            "allard_profiles_active": allard is not None,
            "allard_profile_policy": (
                "fixed TLUSTY205 Lyalpha below 13000 K; temperature-dependent "
                "Allard Lyalpha--Lygamma at and above 13000 K"
            ),
            "allard_temperature_dependent_cutoff_K": (
                _ALLARD_TEMPERATURE_DEPENDENT_CUTOFF_K
            ),
            "allard_minimum_effective_temperature_K": (
                config.allard_minimum_effective_temperature
            ),
            "molecular_equilibrium": molecules,
            "negative_hydrogen_in_charge_equilibrium": molecules,
            "h3plus_partition_model": h3plus_partition_model,
            "h3plus_spectral_opacity": "not included",
            "convection": (
                "ML2/alpha=" + f"{config.mixing_length_alpha:g}"
                if config.mixing_length_alpha is not None
                else "suppressed; radiative equilibrium"
            ),
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_solver": config.atmosphere_solver,
            "atmosphere_initialization": atmosphere_initialization,
            "atmosphere_convergence_status": convergence_status,
            "model_request_fingerprint": request_fingerprint,
            "cool_mean_3d_temperature_differential": (
                "Tremblay et al. 2013 Fig. 7 at fixed gas pressure; "
                "Montreal HM neutral radius rB=0.5"
                if config.use_cool_mean_3d_temperature_differential
                else None
            ),
        },
    )


def compute_db(
    config: DBConfig = DBConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> ModelResult:
    """Calculate one DB atmosphere and spectrum with Beauchamp25-LD lines."""

    data = ModelData.default() if data is None else data
    request_fingerprint = model_request_fingerprint("DB", config, data)
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    resolution = numerical_resolution(config.quality)
    wave = validate_wavelength(wavelength)
    he_i, he_ii = _helium_tables(data)
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        atmosphere = solve_with_screened_boundary(radiative_equilibrium_helium_atmosphere,
            config.effective_temperature,
            config.logg,
            stark_table=he_i,
            helium_ii_stark_table=he_ii,
            n_depth=resolution.n_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=resolution.maximum_iterations,
            structure_solver=config.atmosphere_solver,
            n_angle=min(resolution.n_angle, 3),
            correlated_microfields=True,
            mixing_length_alpha=config.mixing_length_alpha,
            neutral_line_broadening=config.neutral_broadening,
            depth_concentration=config.photospheric_depth_concentration,
            initial_temperature=(
                None if initial_atmosphere is None else initial_atmosphere.temperature
            ),
            initial_column_mass=(
                None if initial_atmosphere is None else initial_atmosphere.column_mass
            ),
            initial_gas_pressure=(
                initial_atmosphere.gas_pressure
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            initial_rosseland_optical_depth=(
                initial_atmosphere.rosseland_optical_depth
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            resume_supplied_structure_in_formal_flux_phase=(
                checkpoint_matches_request
            ),
            iteration_callback=iteration_callback,
        )
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DB")
    spectrum = synthesize_helium_spectrum(
        atmosphere,
        wave,
        stark_table=he_i,
        helium_ii_stark_table=he_ii,
        neutral_broadening=config.neutral_broadening,
        n_angle=resolution.n_angle,
        transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
    )
    return ModelResult(
        "DB",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DB-production-v4",
            "helium_i_profiles": "Beauchamp25-LD semi-analytical",
            "helium_ii_profiles": "Schoning/SYNSPEC table",
            "neutral_broadening": config.neutral_broadening,
            "convection": f"ML2/alpha={config.mixing_length_alpha:g}",
            "atmosphere_solver": config.atmosphere_solver,
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
        },
    )


def _mixed_molecular_data(h2_he_path: str, h2_h2_path: str):
    paths=tuple(Path(p).expanduser().resolve() for p in (h2_he_path,h2_h2_path))
    identity=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
    return _cached_mixed_molecular_data(*map(str,paths),identity)


@lru_cache(maxsize=4)
def _cached_mixed_molecular_data(h2_he_path: str, h2_h2_path: str, identity):
    from .._mixed_cia import MolecularHHePhysics, read_hitran_h2_he_cia
    return MolecularHHePhysics(read_hitran_h2_he_cia(h2_he_path),
        read_borysow_h2_h2_cia_table(h2_h2_path))


def compute_dab(
    config: DABConfig = DABConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> ModelResult:
    """Calculate a DAB/DBA, with explicit molecular H/He support.

    The molecular option requires the Abel/HITRAN H2-He table. It includes
    molecular charge/reaction thermodynamics, H2-He/H2-H2 CIA and neutral Lyalpha
    wings. It is not a dense-fluid EOS or a pressure-distorted CIA model.
    """

    data = ModelData.default() if data is None else data
    molecular = None
    if not isinstance(config.include_molecules, bool):
        raise ValueError("include_molecules must be boolean")
    if config.include_molecules:
        h2he_path = (Path(config.h2_he_cia_path).expanduser().resolve()
            if config.h2_he_cia_path is not None else data.cache / "molecular-opacity/H2-He_2011.cia")
        if not h2he_path.is_file():
            raise FileNotFoundError("Molecular DAB requires HITRAN H2-He_2011.cia; "
                "set DABConfig(h2_he_cia_path=...) or install it in ModelData.cache/molecular-opacity")
        data.require(data.h2_h2_cia)
        molecular = _mixed_molecular_data(str(h2he_path),str(data.h2_h2_cia))
    request_fingerprint = model_request_fingerprint("DAB", config, data,
        physical_data_identity=(None if molecular is None else
            {"h2_he_cia_path": str(molecular.h2_he_table.source_path),
             "h2_he_cia_sha256": molecular.h2_he_table.source_sha256,
             "molecular_closure_revision": 1}))
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    resolution = numerical_resolution(config.quality)
    wave = validate_wavelength(wavelength)
    he_i, he_ii = _helium_tables(data)
    if config.lyman_profile_source == "allard":
        allard = _allard_lyman_profiles_for_effective_temperature(
            config.effective_temperature,
            data,
            minimum_effective_temperature=(
                config.allard_minimum_effective_temperature
            ),
            use_fixed_low_temperature_profile=False,
        )
    elif config.lyman_profile_source == "stark":
        allard = None
    else:
        raise ValueError("lyman_profile_source must be 'allard' or 'stark'")
    self_broadening_prescription = _da_self_broadening_prescription(
        config.effective_temperature,
        config.balmer_self_broadening_prescription,
    )
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        atmosphere = solve_with_screened_boundary(radiative_equilibrium_hydrogen_helium_atmosphere,
            config.effective_temperature,
            config.logg,
            config.log_hydrogen_to_helium,
            molecular_h_he=molecular,
            stark_table=he_i,
            helium_ii_stark_table=he_ii,
            n_depth=resolution.n_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=resolution.maximum_iterations,
            structure_solver=config.atmosphere_solver,
            n_angle=min(resolution.n_angle, 3),
            unified_allard_table=allard,
            allard_stark_weight=1.0,
            mixing_length_alpha=config.mixing_length_alpha,
            neutral_line_broadening=config.neutral_broadening,
            depth_concentration=config.photospheric_depth_concentration,
            hydrogen_self_broadening_prescription=(
                self_broadening_prescription
            ),
            hydrogen_self_broadening_truncation_closure=(
                config.balmer_self_broadening_truncation_closure
            ),
            initial_temperature=(
                None if initial_atmosphere is None else initial_atmosphere.temperature
            ),
            initial_column_mass=(
                None if initial_atmosphere is None else initial_atmosphere.column_mass
            ),
            initial_gas_pressure=(
                initial_atmosphere.gas_pressure
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            initial_rosseland_optical_depth=(
                initial_atmosphere.rosseland_optical_depth
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            resume_supplied_structure_in_formal_flux_phase=(
                checkpoint_matches_request
            ),
            iteration_callback=iteration_callback,
        )
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DAB")
    hstate = atmosphere.hydrogen_lte_state
    has_molecules = hstate is not None and hstate.chemical_model == "molecular-h-he-hm"
    if has_molecules != config.include_molecules:
        raise ValueError("DAB synthesis chemistry does not match its atmosphere; rebuild and relax it")
    spectrum = synthesize_hydrogen_helium_spectrum(
        atmosphere,
        wave,
        molecular_h_he=molecular,
        stark_table=he_i,
        helium_ii_stark_table=he_ii,
        unified_allard_table=allard,
        allard_stark_weight=1.0,
        neutral_broadening=config.neutral_broadening,
        hydrogen_self_broadening_prescription=(
            self_broadening_prescription
        ),
        hydrogen_self_broadening_truncation_closure=(
            config.balmer_self_broadening_truncation_closure
        ),
        n_angle=resolution.n_angle,
        transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
    )
    return ModelResult(
        "DAB",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DAB-production-v5",
            "composition": ("homogeneous molecular H/He LTE" if molecular is not None
                else "homogeneous atomic H/He LTE"),
            "includes_molecular_equilibrium": molecular is not None,
            "h2_he_cia_source": None if molecular is None else str(molecular.h2_he_table.source_path),
            "h2_he_cia_sha256": None if molecular is None else molecular.h2_he_table.source_sha256,
            "microfields": "Q-MHD for H and He",
            "lyman_profiles": (
                "temperature-regime Allard + full charged Stark "
                "(Warwick convention)"
                if allard is not None
                else "charged-particle Stark"
            ),
            "allard_profiles_active": allard is not None,
            "allard_profile_policy": (
                "delivered temperature-indexed Lyalpha--Lygamma at and above "
                "the configured minimum temperature"
            ),
            "allard_minimum_effective_temperature_K": (
                config.allard_minimum_effective_temperature
            ),
            "balmer_self_broadening": self_broadening_prescription,
            "balmer_self_broadening_policy": (
                "automatic: Ali-Griem below 10000 K; Barklem otherwise"
                if config.balmer_self_broadening_prescription is None
                else "explicit override"
            ),
            "balmer_self_broadening_truncation_closure": (
                config.balmer_self_broadening_truncation_closure
            ),
            "convection": f"ML2/alpha={config.mixing_length_alpha:g}",
            "neutral_broadening": config.neutral_broadening,
            "atmosphere_solver": config.atmosphere_solver,
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
        },
    )


def _synthesis_flux_fraction(spectrum, effective_temperature: float) -> dict:
    """Return the synthesized flux integral as a fraction of sigma Teff^4.

    The structure solve conserves flux on its own opacity-sampling grid; the
    final synthesis uses a denser line list.  Their agreement is a direct
    consistency check.  The Planck-shaped flux outside the synthesis range
    is not added, so the value is reported with its wavelength limits.
    """

    from ..constants import STEFAN_BOLTZMANN
    from .._compat import trapezoid

    wavelength = np.asarray(spectrum.wavelength_angstrom, dtype=np.float64)
    flux = np.asarray(spectrum.surface_flux_lambda, dtype=np.float64)
    fraction = float(
        trapezoid(flux, wavelength) / (STEFAN_BOLTZMANN * effective_temperature**4)
    )
    return {
        "fraction": fraction,
        "wavelength_range_angstrom": [float(wavelength[0]), float(wavelength[-1])],
    }


def _polluted_helium_level_resolved_photoionization(
    data: ModelData,
    atomic,
    elements: tuple[str, ...],
    maximum_charge: int,
):
    """Assemble bundled level-resolved photoionization for DZ ions.

    Every model atom present in the release cache is used for a loaded ion
    stage.  Identified TLUSTY terms are rebased on observed Stout energies;
    the remaining ions keep their Verner ground-state fits.
    """

    from ..d6 import (
        NORAD_LEVEL_RESOLVED_FILES,
        read_identified_norad_photoionization,
        D6_TLUSTY_RAP_FILES,
        D6_TLUSTY_TOPBASE_FILES,
        read_tlusty_topbase_lte_photoionization,
        merge_topbase_photoionization_databases,
        read_sirocco_topbase_lte_photoionization,
        read_tlusty_rap_lte_photoionization,
        tlusty_observed_term_excitation_overrides,
    )

    def wanted(element: str, charge: int) -> bool:
        return element in elements and charge <= maximum_charge and (
            (element, charge) in atomic.ions
        )

    tlusty = [
        (data.tlusty_atoms / name, element, charge)
        for name, (_, _, element, charge) in D6_TLUSTY_TOPBASE_FILES.items()
        if wanted(element, charge)
    ]
    rap = [
        (data.tlusty_atoms / name, element, charge)
        for name, (_, _, element, charge) in D6_TLUSTY_RAP_FILES.items()
        if wanted(element, charge)
    ]
    sirocco = [
        (data.sirocco_atomic / f"{element.lower()}_2_levels.dat",
         data.sirocco_atomic / f"{element.lower()}_2_phot.dat", element, 1)
        for element in ("C", "O")
        if wanted(element, 1)
    ]
    norad = [
        (data.norad / name, element, charge)
        for name, (_, _, element, charge) in NORAD_LEVEL_RESOLVED_FILES.items()
        if wanted(element, charge) and (data.norad / name).is_file()
    ]
    if not (tlusty or rap or sirocco or norad):
        return None
    data.require(
        *(entry[0] for entry in tlusty),
        *(entry[0] for entry in rap),
        *(path for entry in sirocco for path in entry[:2]),
        fetch_command="python scripts/fetch_metal_data.py",
    )
    parts = []
    if tlusty:
        parts.append(read_tlusty_topbase_lte_photoionization(
            tlusty,
            atomic,
            excitation_energy_overrides_ev=(
                tlusty_observed_term_excitation_overrides(tlusty, atomic)
            ),
        ))
    if rap:
        parts.append(read_tlusty_rap_lte_photoionization(rap, atomic))
    if sirocco:
        parts.append(read_sirocco_topbase_lte_photoionization(sirocco, atomic))
    for path, element, charge in norad:
        identified = read_identified_norad_photoionization(path, element, charge, atomic)
        if identified.sections:
            parts.append(identified)
    return merge_topbase_photoionization_databases(*parts)


def compute_dz(
    config: DZConfig = DZConfig(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    synthesis_transfer: Literal["formal-linear", "formal-pchip"] = "formal-linear",
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> ModelResult:
    """Calculate one warm DZ/DBZ atmosphere with metal structural feedback."""

    if synthesis_transfer not in ("formal-linear", "formal-pchip"):
        raise ValueError("unsupported DZ synthesis_transfer")
    data = ModelData.default() if data is None else data
    request_fingerprint = model_request_fingerprint("DZ", config, data)
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    resolution = numerical_resolution(config.quality)
    wave = validate_wavelength(wavelength)
    he_i, he_ii = _helium_tables(data)
    data.require(
        data.stout,
        data.verner_photoionization,
        fetch_command="python scripts/fetch_metal_data.py",
    )
    if config.dense_helium_eos not in ("reos3", "ideal"):
        raise ValueError("dense_helium_eos must be 'reos3' or 'ideal'")
    helium_reos3 = None
    if config.dense_helium_eos == "reos3":
        data.require(
            data.helium_reos3,
            fetch_command="python scripts/fetch_metal_data.py",
        )
        helium_reos3 = read_helium_reos3_table(data.helium_reos3)
    if config.maximum_metal_charge < 1:
        raise ValueError("maximum_metal_charge must be positive")
    elements = tuple(config.abundances)
    if not elements:
        raise ValueError("DZ abundances must contain at least one element")
    default_abundances = dict(config.abundances) == dict(GD40_ABUNDANCES)
    if config.ca_ii_resonance_source not in ("chianti-reduced", "lte"):
        raise ValueError(
            "ca_ii_resonance_source must be 'chianti-reduced' or 'lte'"
        )
    use_reduced_ca_ii_source = (
        "Ca" in elements
        and config.ca_ii_resonance_source == "chianti-reduced"
    )
    if use_reduced_ca_ii_source:
        data.require(
            data.ca_ii_chianti_collisions,
            fetch_command="python scripts/fetch_metal_data.py",
        )
    atomic = read_stout_atomic_database(
        data.stout,
        elements=elements,
        maximum_charge=config.maximum_metal_charge,
    )
    if config.strong_line_atomic_data not in ("stout", "nist-asd"):
        raise ValueError(
            "strong_line_atomic_data must be 'stout' or 'nist-asd'"
        )
    if config.strong_line_atomic_data == "nist-asd":
        nist_paths = [
            data.nist_asd_strong / filename
            for filename, (_, _, element, charge) in (
                NIST_ASD_STRONG_ION_FILES.items()
            )
            if element in elements and charge <= config.maximum_metal_charge
        ]
        if nist_paths:
            data.require(
                *nist_paths,
                fetch_command="python scripts/fetch_metal_data.py",
            )
            # Replace every accuracy-graded (C or better) NIST transition the
            # synthesis can include, not only f >= 0.01: UV abundance lines
            # such as Si II 1808 (f = 0.002) are weak but important.
            atomic = read_nist_asd_strong_atomic_database(
                nist_paths, atomic, minimum_oscillator_strength=1.0e-4
            )
    photo = read_verner_photoionization_database(
        data.verner_photoionization,
        elements=elements,
        maximum_charge=config.maximum_metal_charge,
    )
    # photo.dat omits the iron-group ions other than Fe (and P, Cl, K);
    # Verner's phfit2 tables supply their ground-state fits.
    data.require(data.verner_phfit2, fetch_command="python scripts/fetch_metal_data.py")
    supplement = read_verner_phfit2_database(
        data.verner_phfit2,
        elements=elements,
        maximum_charge=config.maximum_metal_charge,
        exclude=tuple(photo.fits),
    )
    photo = VernerPhotoionizationDatabase(
        MappingProxyType({**photo.fits, **supplement}),
        source=(
            photo.source
            + ("; Verner phfit2 (Verner & Yakovlev 1995) for "
               + ", ".join(f"{e} {c}" for e, c in sorted(supplement))
               if supplement else "")
        ),
    )
    topbase_photoionization = (
        _polluted_helium_level_resolved_photoionization(
            data, atomic, elements, config.maximum_metal_charge
        )
        if config.level_resolved_photoionization
        else None
    )
    if config.unified_metal_helium_profiles not in (
        "production", "legacy", "off"
    ):
        raise ValueError(
            "unified_metal_helium_profiles must be 'production', 'legacy', or 'off'"
        )
    if config.unified_metal_helium_profiles != "off" and "Mg" in elements:
        data.require(
            data.mg_he_red_wing,
            *(
                (data.mg_i_he_density_profiles,)
                if config.unified_metal_helium_profiles == "production"
                else ()
            ),
            fetch_command="python scripts/fetch_metal_data.py",
        )
        mg_he = read_mg_he_red_wing_table(
            data.mg_he_red_wing,
            (
                data.mg_i_he_density_profiles
                if config.unified_metal_helium_profiles == "production"
                else None
            ),
        )
    else:
        mg_he = None
    if config.unified_metal_helium_profiles == "production" and "Ca" in elements:
        data.require(
            data.ca_i_he_density_profiles,
            data.ca_i_he_temperature_profiles,
            fetch_command="python scripts/fetch_metal_data.py",
        )
        ca_i_he = read_ca_i_he_profile_table(
            data.ca_i_he_density_profiles,
            data.ca_i_he_temperature_profiles,
        )
    else:
        ca_i_he = None
    structure_lines, structure_line_policy, metal_number_fraction = (
        _metal_structure_line_budget(
            config.quality,
            config.abundances,
            config.structure_maximum_metal_lines,
        )
    )
    formal_lines = (
        1000 if config.quality == "quick" else 20_000
    ) if config.formal_maximum_metal_lines is None else int(
        config.formal_maximum_metal_lines
    )
    if formal_lines < 1:
        raise ValueError("metal line limits must be positive")
    if config.structure_opacity_sampling_resolution is not None:
        # The emergent spectrum is flux conserving only if the structure
        # absorbs the same lines the synthesis does.  Opacity sampling makes
        # the structural cost nearly independent of the line count.
        if config.structure_maximum_metal_lines is None:
            structure_lines = formal_lines
            structure_line_policy = "opacity sampling: synthesis line list"
        structure_minimum_oscillator_strength = (
            config.formal_minimum_metal_oscillator_strength
            if config.structure_minimum_metal_oscillator_strength is None
            else config.structure_minimum_metal_oscillator_strength
        )
    else:
        structure_minimum_oscillator_strength = (
            0.01
            if config.structure_minimum_metal_oscillator_strength is None
            else config.structure_minimum_metal_oscillator_strength
        )
    if config.lyman_profile_source == "allard":
        allard = (
            _allard_lyman_profiles_for_effective_temperature(
                config.effective_temperature,
                data,
                minimum_effective_temperature=(
                    config.allard_minimum_effective_temperature
                ),
                use_fixed_low_temperature_profile=False,
            )
            if config.log_hydrogen_abundance is not None
            else None
        )
    elif config.lyman_profile_source == "stark":
        allard = None
    else:
        raise ValueError("lyman_profile_source must be 'allard' or 'stark'")
    self_broadening_prescription = _da_self_broadening_prescription(
        config.effective_temperature,
        config.balmer_self_broadening_prescription,
    )
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        atmosphere = solve_with_screened_boundary(radiative_equilibrium_helium_atmosphere,
            config.effective_temperature,
            config.logg,
            stark_table=he_i,
            helium_ii_stark_table=he_ii,
            n_depth=resolution.n_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=resolution.maximum_iterations,
            structure_solver=config.atmosphere_solver,
            n_angle=min(resolution.n_angle, 3),
            correlated_microfields=True,
            mixing_length_alpha=config.mixing_length_alpha,
            neutral_line_broadening=config.neutral_broadening,
            metal_database=atomic,
            metal_abundances=config.abundances,
            log_hydrogen_abundance=config.log_hydrogen_abundance,
            metal_photoionization_database=photo,
            metal_topbase_photoionization_database=topbase_photoionization,
            mg_he_red_wing_table=mg_he,
            ca_i_he_profile_table=ca_i_he,
            helium_reos3_table=helium_reos3,
            minimum_metal_oscillator_strength=(
                structure_minimum_oscillator_strength
            ),
            maximum_metal_lines=structure_lines,
            metal_line_opacity_sampling_resolution=(
                config.structure_opacity_sampling_resolution
            ),
            depth_concentration=config.photospheric_depth_concentration,
            include_dense_helium_metal_ionization=True,
            metal_occupation_probability_partitions=(
                config.occupation_probability_metal_partitions
            ),
            hydrogen_self_broadening_prescription=(
                self_broadening_prescription
            ),
            hydrogen_self_broadening_truncation_closure=(
                config.balmer_self_broadening_truncation_closure
            ),
            include_hydrogen_series_pseudocontinuum=True,
            unified_allard_table=allard,
            allard_stark_weight=1.0,
            initial_temperature=(
                None if initial_atmosphere is None else initial_atmosphere.temperature
            ),
            initial_column_mass=(
                None if initial_atmosphere is None else initial_atmosphere.column_mass
            ),
            initial_gas_pressure=(
                initial_atmosphere.gas_pressure
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            initial_rosseland_optical_depth=(
                initial_atmosphere.rosseland_optical_depth
                if checkpoint_matches_request and initial_atmosphere is not None
                else None
            ),
            resume_supplied_structure_in_formal_flux_phase=(
                checkpoint_matches_request
            ),
            iteration_callback=iteration_callback,
        )
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DZ")
    spectrum = synthesize_helium_spectrum(
        atmosphere,
        wave,
        stark_table=he_i,
        helium_ii_stark_table=he_ii,
        neutral_broadening=config.neutral_broadening,
        metal_database=atomic,
        metal_abundances=config.abundances,
        log_hydrogen_abundance=config.log_hydrogen_abundance,
        metal_photoionization_database=photo,
        metal_topbase_photoionization_database=topbase_photoionization,
        mg_he_red_wing_table=mg_he,
        ca_i_he_profile_table=ca_i_he,
        include_dense_helium_metal_ionization=True,
        metal_occupation_probability_partitions=(
            config.occupation_probability_metal_partitions
        ),
        minimum_metal_oscillator_strength=(
            config.formal_minimum_metal_oscillator_strength
        ),
        maximum_metal_lines=formal_lines,
        hydrogen_self_broadening_prescription=(
            self_broadening_prescription
        ),
        hydrogen_self_broadening_truncation_closure=(
            config.balmer_self_broadening_truncation_closure
        ),
        include_hydrogen_series_pseudocontinuum=True,
        unified_allard_table=allard,
        allard_stark_weight=1.0,
        ca_ii_resonance_collision_strengths=(
            str(data.ca_ii_chianti_collisions)
            if use_reduced_ca_ii_source
            else None
        ),
        n_angle=resolution.n_angle,
        transfer_discretization=synthesis_transfer,
        transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
    )
    return ModelResult(
        "DZ",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DZ-production-v8-paper-stout",
            "abundance_source": (
                "Klein et al. (2010), GD 40"
                if default_abundances
                else "caller-supplied"
            ),
            "atomic_lines": getattr(atomic, "source", "Stout"),
            "strong_line_atomic_data": config.strong_line_atomic_data,
            "maximum_metal_charge": config.maximum_metal_charge,
            "photoionization": (
                "level-resolved TOPbase for the bundled model-atom ions; "
                "Verner ground-term fits otherwise"
                if topbase_photoionization is not None
                else "Verner ground-term fits"
            ),
            "level_resolved_photoionization": (
                topbase_photoionization.source
                if topbase_photoionization is not None
                else "disabled"
            ),
            "metal_electron_feedback": True,
            "structure_maximum_metal_lines": structure_lines,
            "structure_metal_line_budget_policy": structure_line_policy,
            "structure_minimum_metal_oscillator_strength": (
                structure_minimum_oscillator_strength
            ),
            "structure_opacity_sampling_resolution": (
                config.structure_opacity_sampling_resolution
            ),
            "photospheric_depth_concentration": (
                config.photospheric_depth_concentration
            ),
            "synthesis_transfer_depth_refinement": (
                config.synthesis_transfer_depth_refinement
            ),
            "summed_metal_number_fraction": metal_number_fraction,
            "dense_helium_ionization": True,
            "metal_partition_functions": (
                "Hummer-Mihalas occupation probabilities: Q-MHD charged "
                "microfields and neutral-He excluded volume, iterated with "
                "charge neutrality; same survival in the line opacity"
                if config.occupation_probability_metal_partitions
                else "fixed spectroscopic cutoff 0.1 eV below each limit"
            ),
            "synthesis_flux_fraction": _synthesis_flux_fraction(
                spectrum, config.effective_temperature
            ),
            "bulk_helium_eos": (
                helium_reos3.source
                if helium_reos3 is not None
                else "ideal chemical-picture pressure closure"
            ),
            "microfields": "Q-MHD for He and trace H",
            "neutral_broadening": config.neutral_broadening,
            "unified_metal_helium_profiles": (
                {
                    "mode": config.unified_metal_helium_profiles,
                    "Mg I 2852": mg_he.source if mg_he is not None else "disabled",
                    "Ca I 4227": ca_i_he.source if ca_i_he is not None else "disabled",
                    "interpolation": (
                        "factorized published temperature and density sequences; "
                        "no fitted profile scales"
                    ),
                    "temperature_domain_policy": (
                        "use the unified profile inside its published temperature "
                        "range and the ordinary impact profile outside it"
                    ),
                }
            ),
            "balmer_self_broadening": self_broadening_prescription,
            "balmer_self_broadening_policy": (
                "automatic: Ali-Griem below 10000 K; Barklem otherwise"
                if config.balmer_self_broadening_prescription is None
                else "explicit override"
            ),
            "balmer_self_broadening_truncation_closure": (
                config.balmer_self_broadening_truncation_closure
            ),
            "hydrogen_series_pseudocontinuum": (
                config.log_hydrogen_abundance is not None
            ),
            "lyman_profiles": (
                "temperature-regime Allard + full charged Stark"
                if allard is not None
                else "charged-particle Stark"
            ),
            "allard_profiles_active": allard is not None,
            "convection": (
                f"ML2/alpha={config.mixing_length_alpha:g}"
                if config.mixing_length_alpha is not None
                else "suppressed; radiative equilibrium"
            ),
            "atmosphere_solver": config.atmosphere_solver,
            "allard_ca_ii_control": False,
            "ca_ii_resonance_source": (
                "complete-redistribution H/K source functions from CHIANTI "
                "electron rates and radiative branching, with H-K "
                "fine-structure transfer; LTE populations and extinction"
                if use_reduced_ca_ii_source
                else "LTE"
            ),
            "ca_ii_resonance_source_applied_in_structure": False,
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
        },
    )
