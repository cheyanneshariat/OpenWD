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
    TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K,
    hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
)
from ..dense_eos import read_helium_reos3_table
from ..helium_ii_stark import read_helium_ii_stark_table
from ..helium_stark import read_helium_stark_table
from ..metals import (
    NIST_ASD_STRONG_ION_FILES,
    atomic_database_with_kurucz_iron_group_positions,
    atomic_database_with_kurucz_electron_stark,
    augment_silicon_ii_kurucz_transitions,
    atomic_database_with_si_ii_he_unified_profile,
    atomic_database_with_rauch_zn_cu_transitions,
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
from ..h2_lines import read_h2_lyman_werner_data
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
    # Photospheric H2 Lyman/Werner lines with fluorescent source functions
    # (h2_lines; Xu et al. 2013 detect them below ~14,000 K).  When enabled,
    # molecules are kept in the EOS at any temperature.  Final spectrum only.
    h2_lyman_werner_lines: bool = False
    h3plus_partition_model: Literal[
        "neale-tennyson-1995", "none"
    ] = "neale-tennyson-1995"
    mixing_length_alpha: float | None = 0.7
    # Retired: True raises; kept so stored configurations still load.
    use_cool_mean_3d_temperature_differential: bool = False
    # Line-profile mode.  The Tremblay et al. (2013) <3D>-1D temperature
    # difference (log g=8 CO5BOLD sequence, 5997-12999 K) is added to the
    # converged 1D structure with hydrostatic re-integration on the Rosseland
    # scale before synthesis: all layers below 9000 K, tau_R >= 1e-2 above.
    # The spectrum is not flux-conserving (-1.5 to -2.5% bolometric, UV
    # suppressed at 6-8 kK); use it for Balmer line profiles and the default
    # 1D spectrum for SEDs and photometry.  Outside the table range the 1D
    # spectrum is returned and the metadata says why.
    mean_3d_line_profiles: bool = False
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
    # Cold solves whose lower boundary fails the absorption screen (alone or
    # with other checks) are re-run cold with a 4x deeper Rosseland bottom and
    # proportionally more layers, up to this many times.  The gray seed's
    # tau = 100 bottom leaks in hot low-gravity hydrogen.  0 disables it.
    lower_boundary_extensions: int = 4
    # Cold seeds that are convectively unstable at Rosseland depth one take the
    # Schwarzschild-limited (adiabatic) temperature below it, on layers spaced
    # in pressure down to the seed's own tau_max.  The gray law put those
    # layers in a few per cent of the column mass at up to three times the
    # convective temperature; 4250-5000 K models then stalled at ~0.75
    # sigma Teff^4.  Radiative seeds are unchanged.  False restores the gray seed.
    convective_seed: bool = True


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
    # "column-mass": mass control volumes for the structure transfer, local
    # energy balance and final spectrum together; the cold-start grid top starts
    # at Rosseland tau 1e-12; and whenever lower-boundary screening fails (also
    # together with other failed checks) the mesh is deepened (x4 bottom
    # pressure) and re-solved from the first phase (DB-production-v5). The
    # optical-depth cell widths couple each cell to its neighbours' extinction;
    # where He- free-free dominates (n_e ~ T^20) that admits growing odd-even
    # temperature modes that stall the solve. "optical-depth" is the
    # DB-production-v4 preset, which column mass cannot replace everywhere:
    # hot DBs have a steep upper-atmosphere temperature front that the mass
    # grid under-resolves. "auto" (default) uses column mass where He- free-free
    # supplies more than half of the cold seed's Rosseland-mean continuum
    # opacity somewhere in tau_R = 1e-6..1e-2, and optical depth elsewhere.
    # All use the synthesis depth refinement above for the final spectrum.
    transfer_discretization: Literal["auto", "column-mass", "optical-depth"] = "auto"


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
    # None: Allard profiles at every Teff; below the 9000-K table the local
    # temperature is clamped to it, as already happens in the cool upper
    # layers of hotter models.  A number restores a Teff cutoff below which
    # the cool neutral-H wing treatment is used instead.
    allard_minimum_effective_temperature: float | None = None
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
    # Optional trace metals, log10 N(Z)/N(H).  They join the H/He charge
    # closure and the structural (opacity-sampled) and final opacity of the
    # same homogeneous atomic mixture.  None keeps the metal-free preset.
    abundances: Mapping[str, float] | None = None
    maximum_metal_charge: int | Mapping[str, int] = 4
    metal_classical_electron_stark: bool = False
    structure_opacity_sampling_resolution: float = 1000.0
    formal_maximum_metal_lines: int | None = None
    formal_minimum_metal_oscillator_strength: float = 1.0e-4
    # Opt-in E1 lines for ions whose Stout files hold only forbidden lines:
    # "rauch-zn-cu" (Rauch et al. Zn IV-V, Cu IV-VI) and "kurucz-fe-ni"
    # (Kurucz measured-level Fe IV, VII and Ni IV-VI).  The default keeps
    # Stout alone.
    metal_line_supplements: tuple[str, ...] = ()


DAB_METAL_LINE_SUPPLEMENTS = ("rauch-zn-cu", "kurucz-fe-ni")


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
    # The dense-fluid defaults below (REOS3, Rogers-Young He- ff, partial-wave
    # He- ff, computed unified He tables) are the configuration validated on ten
    # cool DZ/DZA spectra (4000-8700 K); the alternatives reproduce older releases.
    dense_helium_eos: Literal["reos3", "ideal"] = "reos3"
    # Dense-fluid He-minus free-free correction (Iglesias, Rogers & Saumon
    # 2002 Born ratio with a Rogers-Young neutral-He S(k); no 1/|eps|^2).
    # In cool DZ photospheres (rho ~ 0.1-0.6 g/cm^3) it removes 20-70% of the
    # dominant continuum opacity.  "off" keeps the ideal John (1994) value.
    dense_helium_free_free: Literal["off", "rogers-young"] = "rogers-young"
    # He- free-free below John's (1994) 0.5063-micron edge: "automatic" keeps
    # the Carbon et al. (1969) fit to John (1968), which jumps by 16-24% at
    # the edge for 3000-5000 K; "partial-wave" uses John's edge value times
    # a first-principles e-He partial-wave wavelength shape (continuous).
    helium_minus_free_free: Literal["automatic", "partial-wave"] = "partial-wave"
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
    # Kurucz electron-impact Stark constants (log Gamma_e/n_e at 10^4 K) for
    # Stout E1 lines that have no tabulated Stark width, matched to Kurucz
    # lines once at build time (bundled table, 900-11000 A).  Measured widths
    # always take precedence; oscillator strengths and neutral broadening are
    # unchanged.  False restores the previous default, in which such lines
    # have no electron broadening.
    kurucz_electron_stark: bool = True
    # Unified (non-impact) Si+--He profile for Si II 1260/1264/1265 from
    # model-potential binary-collision calculations, replacing their
    # Lorentzian neutral-He broadening.  No stellar fitting.  False restores
    # the previous Unsold neutral-He Lorentzian for these lines.
    si_ii_he_unified_profile: bool = True
    # Unified Mg I b (5169/5174/5185 A) + He profiles computed from the
    # Allard et al. (2016) potentials (3000-12000 K, n_He 1e19-1e23; the blue
    # satellites pile up and move blueward with density).  False keeps the
    # Unsold Lorentzian; the satellite-dominated blue wing of cool DZs is then absent.
    mg_i_b_he_unified_profile: bool = True
    # Unified Mg II h&k and Ca II H&K + He profiles computed with the OpenWD
    # unified-profile engine from ab initio ion-He potentials (Mg+-He: Allard
    # et al. 2016 GG; Ca+-He: Alekseev, with R-dependent dipoles), 3000-12000 K
    # and n_He 1e19-1e23 cm^-3.  Their far wings dominate the 3000-4500 A
    # metal opacity of cool, dense DZ photospheres, where the impact
    # Lorentzian is 3-20x too weak.
    mg_ii_he_unified_profile: bool = True
    ca_ii_he_unified_profile: bool = True
    # Unified Na I D + He profiles computed from the Dell'Angelo et al. (2012)
    # potentials (Allard et al. 2023 figures), same grid and dense-fluid
    # treatment as the Mg II/Ca II tables.  False keeps the impact Lorentzian.
    na_i_he_unified_profile: bool = True
    # Legacy import switch for bare/custom atoms. Independently corroborated
    # Si II additions are already in the default reader; this switch neither
    # removes them nor appends the former unfiltered supplement to that atom.
    si_ii_kurucz_transitions: bool = True


# mean_3d_line_profiles: the Tremblay et al. (2013) differential exists only
# at log g=8 (SPY validation spans log g=7.60-8.74) and for 5997-12999 K.  Its
# upper-layer (tau_R<1e-2) overshoot cooling matches observed Balmer cores
# below about 9 kK but over-deepens them above (SPY 8615 vs 9256 K), so only
# the photospheric part is applied from this temperature upward.
_MEAN_3D_MAXIMUM_LOGG_OFFSET = 0.75
_MEAN_3D_UPPER_LAYER_MAXIMUM_TEFF_K = 9_000.0
_ALLARD_MINIMUM_EFFECTIVE_TEMPERATURE_K = 9_000.0
_ALLARD_TEMPERATURE_DEPENDENT_CUTOFF_K = 13_000.0
# DA cold-start seed span (the hydrogen solver's defaults) and the bottom
# optical-depth increase per lower-boundary extension.
_DA_SEED_ROSSELAND_DEPTH_MINIMUM = 1.0e-10
_DA_SEED_ROSSELAND_DEPTH_MAXIMUM = 1.0e2
_DA_DEEPENING_FACTOR = 4.0
# Failed radiative solves are re-solved on finer meshes up to this depth.
# 200 layers resolve the steep cool-DA ionization transition (G 76-48); a
# 100-layer attempt is the seed production uses for it.
_DA_RADIATIVE_REFINEMENT_MAXIMUM_DEPTH = 200
_DA_RADIATIVE_REFINEMENT_SEED_DEPTH = 100


def _radiative_refinement_depth(n_depth: int) -> int:
    """Next mesh for a failed radiative solve (40 -> 100 -> 200; 100 -> 200)."""

    return _DA_RADIATIVE_REFINEMENT_SEED_DEPTH if n_depth < _DA_RADIATIVE_REFINEMENT_SEED_DEPTH else 2 * n_depth


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
    """Resolve the public hydrogen neutral-broadening policy.

    Ali--Griem resonance broadening at every temperature, as in the Montreal
    DA models (Tremblay et al. 2010).  Against SPY/UVES profiles it matches
    or beats Barklem from 6.4 to 11.5 kK, and the two agree to <2e-6 in
    normalized flux at 20--33 kK.  ``effective_temperature`` is retained so
    the pre-2026-10-27 temperature-dependent policy can be restored for its
    historical controls.  An explicit caller override is retained for
    controlled tests.
    """

    return "ali-griem" if requested is None else requested


def _allard_lyman_profiles_for_effective_temperature(
    effective_temperature: float,
    data: ModelData,
    *,
    minimum_effective_temperature: float | None = (
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

    ``minimum_effective_temperature=None`` (the DAB default) applies no Teff
    cutoff: below 9000 K the delivered tables are evaluated with the local
    temperature clamped to the 9000-K table. Against the Cukanovaite et al.
    (2021) DBA grid at 8000 K, log g 8, log N(H)/N(He) = -2, this follows the
    1150--1800 A Ly-alpha wing at the same level as at 9000 K, whereas the cool
    neutral-H wing treatment over-predicts 1150--1300 A by a factor ~50.
    """

    if minimum_effective_temperature is not None:
        if (
            not np.isfinite(minimum_effective_temperature)
            or minimum_effective_temperature <= 0.0
        ):
            raise ValueError(
                "allard_minimum_effective_temperature must be None or finite and positive"
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
        (config.effective_temperature <= 12_000.0 or config.h2_lyman_werner_lines)
        if config.include_molecules is None
        else config.include_molecules
    )
    if config.h2_lyman_werner_lines and not molecules:
        raise ValueError("h2_lyman_werner_lines requires include_molecules")
    if (
        isinstance(config.lower_boundary_extensions, bool)
        or not isinstance(config.lower_boundary_extensions, int)
        or config.lower_boundary_extensions < 0
    ):
        raise ValueError("lower_boundary_extensions must be a nonnegative integer")
    if config.use_cool_mean_3d_temperature_differential:
        raise ValueError(
            "use_cool_mean_3d_temperature_differential is retired; use "
            "mean_3d_line_profiles=True"
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
            convective_seed=config.convective_seed,
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
            config.quality in ("standard", "production")
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
            record["domain_expansion"] = len(domain_segments)
            iteration_callback(iteration + iteration_offset, current_atmosphere, record)

        cold_full_solve = initial_atmosphere is None
        seed = atmosphere
        tau_max = _DA_SEED_ROSSELAND_DEPTH_MAXIMUM
        solve_depth = full_depth
        domain_segments = []
        for expansion in range(config.lower_boundary_extensions + 1):
            atmosphere = radiative_equilibrium_hydrogen_atmosphere(
                config.effective_temperature,
                config.logg,
                n_depth=solve_depth,
                tau_max=tau_max,
                n_continuum_wavelength=resolution.n_continuum,
                max_iterations=(
                    min(40, resolution.maximum_iterations)
                    if refine_radiative_grid and solve_depth == 100
                    else resolution.maximum_iterations
                ),
                n_angle=min(resolution.n_angle, 3),
                initial_temperature=(
                    None if seed is None else seed.temperature
                ),
                initial_column_mass=(
                    None if seed is None else seed.column_mass
                ),
                iteration_callback=full_callback,
                **relaxation_kwargs,
            )
            failures = atmosphere.metadata["equilibrium_certificate"]["failures"]
            iterations = int(atmosphere.metadata["radiative_equilibrium_iterations"])
            domain_segments.append(dict(
                n_depth=atmosphere.n_depth,
                seed_rosseland_depth_maximum=tau_max,
                iterations=iterations,
                absorption_escape_bound=atmosphere.metadata.get(
                    "lower_boundary_absorption_escape_bound"
                ),
                failures=list(failures),
            ))
            if (
                not cold_full_solve
                or "boundary_screening" not in failures
                or expansion == config.lower_boundary_extensions
            ):
                break
            # The bottom leaks radiation. Re-solve the same request cold on a
            # deeper seed: the warmer extrapolated cells of a supplied
            # structure would be clamped to its old bottom temperature.
            # Layers grow with the log-tau span to keep the depth spacing.
            iteration_offset += iterations
            tau_max *= _DA_DEEPENING_FACTOR
            solve_depth = int(np.ceil(full_depth * (
                np.log(tau_max / _DA_SEED_ROSSELAND_DEPTH_MINIMUM)
                / np.log(_DA_SEED_ROSSELAND_DEPTH_MAXIMUM / _DA_SEED_ROSSELAND_DEPTH_MINIMUM)
            )))
            seed = None
        domain_iterations = iteration_offset + iterations
        if len(domain_segments) > 1:
            atmosphere = replace(atmosphere, metadata={
                **atmosphere.metadata,
                "adaptive_domain_segments": domain_segments,
                "adaptive_domain_expansions": len(domain_segments) - 1,
                "radiative_equilibrium_iterations_including_domain_adaptation": (
                    domain_iterations
                ),
            })
        while (
            refine_radiative_grid
            and _radiative_refinement_depth(atmosphere.n_depth)
            <= _DA_RADIATIVE_REFINEMENT_MAXIMUM_DEPTH
            and atmosphere_convergence_status(atmosphere) != "converged"
        ):
            # A radiative hydrogen ionization transition can be too steep for
            # the initial mass mesh. Re-solve the SAME equations on a finer
            # mesh (production 100 -> 200; standard 40 -> 100 -> 200); the
            # unfinished state is only an interpolated seed. Doubling from 40
            # never yields a usable seed for G 76-48. Each fine solve must
            # earn its own complete equilibrium certificate.
            coarse = atmosphere
            iteration_offset = (
                domain_iterations
                if depth_refinements == 0
                else iteration_offset + coarse.metadata["radiative_equilibrium_iterations"]
            )
            depth_refinements += 1
            atmosphere = radiative_equilibrium_hydrogen_atmosphere(
                config.effective_temperature,
                config.logg,
                n_depth=_radiative_refinement_depth(coarse.n_depth),
                tau_max=tau_max,
                n_continuum_wavelength=resolution.n_continuum,
                max_iterations=resolution.maximum_iterations,
                n_angle=min(resolution.n_angle, 3),
                initial_temperature=coarse.temperature,
                initial_column_mass=coarse.column_mass,
                iteration_callback=full_callback,
                **relaxation_kwargs,
            )
            first = coarse.metadata.get("radiative_depth_refinement", {})
            atmosphere = replace(atmosphere, metadata={
                **atmosphere.metadata,
                "radiative_depth_refinement": {
                    "initial_depth_points": first.get("initial_depth_points", coarse.n_depth),
                    "final_depth_points": atmosphere.n_depth,
                    "refinements": depth_refinements,
                    "initial_certificate_failures": first.get(
                        "initial_certificate_failures",
                        coarse.metadata["equilibrium_certificate"]["failures"],
                    ),
                    "initial_maximum_flux_residual": first.get(
                        "initial_maximum_flux_residual",
                        coarse.metadata["maximum_all_depth_total_flux_residual"],
                    ),
                    "initial_maximum_local_energy_residual": first.get(
                        "initial_maximum_local_energy_residual",
                        coarse.metadata["maximum_relative_cell_energy_balance_residual"],
                    ),
                    "initial_iterations": iteration_offset,
                    "stellar_parameters_changed": False,
                },
                "radiative_equilibrium_iterations_including_depth_refinement": (
                    iteration_offset + atmosphere.metadata["radiative_equilibrium_iterations"]
                ),
            })
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "DA")
    # The returned atmosphere stays the 1D equilibrium structure, so it remains
    # a valid checkpoint; only the synthesis sees the <3D> differential.
    mean_3d_layers, mean_3d_note = _mean_3d_line_profile_layers(config)
    synthesis_atmosphere = (
        hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere,
            h2_h2_cia_table=cia,
            maximum_logg_offset=_MEAN_3D_MAXIMUM_LOGG_OFFSET,
            layers=mean_3d_layers,
        )
        if mean_3d_layers is not None
        else atmosphere
    )
    spectrum = synthesize_hydrogen_spectrum(
        synthesis_atmosphere,
        wave,
        h2_line_data=(
            read_h2_lyman_werner_data() if config.h2_lyman_werner_lines else None
        ),
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
                "automatic: Ali-Griem at every temperature"
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
            "h2_lyman_werner_lines": config.h2_lyman_werner_lines,
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
            "mean_3d_line_profiles": mean_3d_note,
            "flux_conserving_spectrum": mean_3d_layers is None,
        },
    )


def _mean_3d_line_profile_layers(config: DAConfig) -> tuple[str | None, str | None]:
    """Return the <3D> layers applied in line-profile mode and a provenance note."""

    if not config.mean_3d_line_profiles:
        return None, None
    teff = config.effective_temperature
    low, high = (
        float(TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[0]),
        float(TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[-1]),
    )
    if not low <= teff <= high:
        return None, (
            f"requested; not applied: Teff outside the {low:.0f}-{high:.0f} K "
            "Tremblay et al. (2013) sequence; 1D spectrum"
        )
    if abs(config.logg - 8.0) > _MEAN_3D_MAXIMUM_LOGG_OFFSET:
        return None, (
            "requested; not applied: log g more than "
            f"{_MEAN_3D_MAXIMUM_LOGG_OFFSET:g} dex from the log g=8 sequence; "
            "1D spectrum"
        )
    layers = (
        "all" if teff < _MEAN_3D_UPPER_LAYER_MAXIMUM_TEFF_K else "photospheric"
    )
    return layers, (
        "Tremblay et al. 2013 Fig. 7 <3D>-1D temperature difference, "
        f"layers={layers}, hydrostatic re-integration, synthesis only; "
        "for line profiles, not flux-conserving"
    )


_DB_COLUMN_MASS_TOP_ROSSELAND_DEPTH = 1.0e-12
_DB_DEEPENING_PRESSURE_FACTOR = 4.0
# Cool dense DZ: transparent windows reach far below the Rosseland-scaled seed
# (LP 658-2 first-phase escape 0.30, WD 2251-070 0.11-0.15).  Deepen as soon as
# the first phase shows such a leak, on any screening failure, and re-run the
# full phase sequence on the deeper mesh: resuming the extrapolated cells in the
# formal-flux phase diverged (residual 6e-5 -> 1.2e4) for LP 658-2 with a CI mix.
_DZ_DEEPENING_PRESSURE_FACTOR = 4.0
_DZ_EARLY_DEEPENING_ESCAPE_LIMIT = 0.05
_DB_HELIUM_MINUS_SCREEN_DEPTHS = (1.0e-6, 1.0e-2)
# On the optical-depth path, "auto" extends the top boundary like the lower
# one: when a cold solve fails and its worst residual sits in the top layers,
# the seed top (tau_R 1e-8) is moved up tenfold and the model re-solved cold, at
# most twice. A top layer still opaque in He II Lyman alpha (32 kK, log g 6)
# can otherwise hold a lone hot node whose temperature the equations barely fix.
_DB_OPTICAL_DEPTH_TOP_ROSSELAND_DEPTH = 1.0e-8
_DB_UPPER_BOUNDARY_EXTENSIONS = 2
_DB_UPPER_BOUNDARY_FACTOR = 10.0
_DB_UPPER_BOUNDARY_LAYERS = 3
_DB_HELIUM_MINUS_SCREEN_WAVELENGTH = np.geomspace(300.0, 100_000.0, 600)


def _db_helium_minus_share(config: DBConfig, n_depth: int) -> float:
    """Largest He- free-free share of the cold seed's Rosseland-mean continuum.

    Evaluated on the gray hydrostatic helium seed between Rosseland depths
    1e-6 and 1e-2, the outer layers where the optical-depth discretization's
    neighbour coupling becomes unstable when He- free-free dominates.
    """
    from ..atmosphere import helium_continuum_atmosphere
    from ..constants import BOLTZMANN, LIGHT_SPEED, PLANCK
    from ..helium import helium_continuum_mass_absorption_coefficient

    seed = helium_continuum_atmosphere(
        config.effective_temperature, config.logg, n_depth=n_depth,
        depth_concentration=config.photospheric_depth_concentration,
        correlated_microfields=True,
    )
    wave = _DB_HELIUM_MINUS_SCREEN_WAVELENGTH
    full = helium_continuum_mass_absorption_coefficient(seed, wave)
    # The He- factor must be positive; 1e-30 removes it to roundoff.
    without = helium_continuum_mass_absorption_coefficient(
        seed, wave, helium_minus_correction=np.full_like(full, 1.0e-30)
    )
    nu = LIGHT_SPEED / (wave * 1.0e-8)
    x = PLANCK * nu[:, None] / (BOLTZMANN * seed.temperature[None, :])
    weight = x**4 * np.exp(-x) / (-np.expm1(-x)) ** 2 * np.abs(np.gradient(nu))[:, None]
    rosseland = lambda kappa: np.sum(weight, axis=0) / np.sum(weight / kappa, axis=0)
    share = 1.0 - rosseland(without) / rosseland(full)
    lo, hi = _DB_HELIUM_MINUS_SCREEN_DEPTHS
    outer = (seed.rosseland_optical_depth >= lo) & (seed.rosseland_optical_depth <= hi)
    return float(np.max(share[outer]))


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

    if config.transfer_discretization not in ("auto", "column-mass", "optical-depth"):
        raise ValueError(
            "DB transfer_discretization must be 'auto', 'column-mass' or 'optical-depth'"
        )
    resolution = numerical_resolution(config.quality)
    helium_minus_share = None
    if config.transfer_discretization == "auto":
        helium_minus_share = _db_helium_minus_share(config, resolution.n_depth)
        column_mass = helium_minus_share > 0.5
    else:
        column_mass = config.transfer_discretization == "column-mass"
    discretization = "column-mass" if column_mass else "optical-depth"
    data = ModelData.default() if data is None else data
    request_fingerprint = model_request_fingerprint("DB", config, data)
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    wave = validate_wavelength(wavelength)
    he_i, he_ii = _helium_tables(data)
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        solver_arguments = (config.effective_temperature, config.logg)
        solver_options = dict(
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
        )
        if column_mass:
            discretization_options = dict(
                transfer_discretization="column-mass",
                tau_min=_DB_COLUMN_MASS_TOP_ROSSELAND_DEPTH,
                deepen_on_any_screen_failure=True,
                restart_after_deepening=True,
                deepening_pressure_factor=_DB_DEEPENING_PRESSURE_FACTOR,
            )
        else:
            discretization_options = {}
        atmosphere = solve_with_screened_boundary(
            radiative_equilibrium_helium_atmosphere,
            *solver_arguments,
            **solver_options,
            iteration_callback=iteration_callback,
            **discretization_options,
        )
        if (
            not column_mass
            and config.transfer_discretization == "auto"
            and initial_atmosphere is None
        ):
            extensions = []
            top = _DB_OPTICAL_DEPTH_TOP_ROSSELAND_DEPTH
            for _ in range(_DB_UPPER_BOUNDARY_EXTENSIONS):
                meta = atmosphere.metadata
                worst = meta.get("nonlinear_solver_final_worst_residual_depth_index")
                if (
                    meta.get("equilibrium_certificate", {}).get("verified")
                    or worst is None
                    or worst >= _DB_UPPER_BOUNDARY_LAYERS
                ):
                    break
                extensions.append(dict(
                    seed_top_rosseland_optical_depth=top,
                    failures=list(meta["equilibrium_certificate"]["failures"]),
                    worst_residual_depth_index=int(worst),
                ))
                top /= _DB_UPPER_BOUNDARY_FACTOR
                atmosphere = solve_with_screened_boundary(
                    radiative_equilibrium_helium_atmosphere,
                    *solver_arguments,
                    **solver_options,
                    iteration_callback=iteration_callback,
                    tau_min=top,
                )
            if extensions:
                atmosphere = replace(atmosphere, metadata={
                    **atmosphere.metadata,
                    "upper_boundary_extensions": extensions,
                    "seed_top_rosseland_optical_depth": top,
                })
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
        **(dict(transfer_discretization="column-mass") if column_mass else {}),
    )
    return ModelResult(
        "DB",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DB-production-v5" if column_mass else "DB-production-v4",
            "transfer_discretization": discretization,
            "transfer_discretization_request": config.transfer_discretization,
            "helium_minus_rosseland_share": helium_minus_share,
            "helium_i_profiles": "Beauchamp25-LD semi-analytical",
            "helium_ii_profiles": "Schoning/SYNSPEC table",
            "neutral_broadening": config.neutral_broadening,
            "convection": (
                f"ML2/alpha={config.mixing_length_alpha:g}"
                if config.mixing_length_alpha is not None
                else "suppressed; radiative equilibrium"
            ),
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
    from .._provenance import content_identity

    paths=tuple(Path(p).expanduser().resolve() for p in (h2_he_path,h2_h2_path))
    # Data replacements can preserve size and modification time. The cache
    # must follow the bytes used by the request's physical-data fingerprint.
    identity=content_identity(paths)
    return _cached_mixed_molecular_data(*map(str,paths),identity)


@lru_cache(maxsize=4)
def _cached_mixed_molecular_data(h2_he_path: str, h2_h2_path: str, identity):
    from .._mixed_cia import MolecularHHePhysics, read_hitran_h2_he_cia
    return MolecularHHePhysics(read_hitran_h2_he_cia(h2_he_path),
        read_borysow_h2_h2_cia_table(h2_h2_path))


def _dab_trace_metals(config: DABConfig, data: ModelData) -> dict | None:
    """Resolve the optional DAB trace-metal data and line budgets."""

    if config.abundances is None:
        if config.metal_line_supplements:
            raise ValueError("metal_line_supplements require trace-metal abundances")
        return None
    if config.include_molecules:
        raise ValueError("trace metals are not supported in the molecular H/He workflow")
    if not config.abundances or any(
        not np.isfinite(value) for value in config.abundances.values()
    ):
        raise ValueError("DAB abundances must contain finite log10 N(Z)/N(H) values")
    if set(config.abundances) & {"H", "He"}:
        raise ValueError("DAB abundances specify metals; set H/He with log_hydrogen_to_helium")
    if not np.isfinite(config.structure_opacity_sampling_resolution) or (
        config.structure_opacity_sampling_resolution <= 0.0
    ):
        raise ValueError("structure_opacity_sampling_resolution must be finite and positive")
    if not isinstance(config.metal_classical_electron_stark, bool):
        raise ValueError("metal_classical_electron_stark must be boolean")
    elements = tuple(config.abundances)
    data.require(
        data.stout,
        data.verner_photoionization,
        data.verner_phfit2,
        fetch_command="python scripts/fetch_metal_data.py",
    )
    supplements = config.metal_line_supplements
    if (
        not isinstance(supplements, tuple)
        or len(set(supplements)) != len(supplements)
        or not set(supplements) <= set(DAB_METAL_LINE_SUPPLEMENTS)
    ):
        raise ValueError(
            "metal_line_supplements must be a tuple of distinct names from "
            + ", ".join(DAB_METAL_LINE_SUPPLEMENTS)
        )
    atomic = read_stout_atomic_database(
        data.stout, elements=elements, maximum_charge=config.maximum_metal_charge
    )
    if "rauch-zn-cu" in supplements:
        atomic = atomic_database_with_rauch_zn_cu_transitions(atomic, data.stout)
    if "kurucz-fe-ni" in supplements:
        atomic = atomic_database_with_kurucz_iron_group_positions(atomic)
    top_charge = max(ion.charge for ion in atomic.ions.values())
    photo = read_verner_photoionization_database(
        data.verner_photoionization, elements=elements, maximum_charge=top_charge
    )
    # photo.dat omits Ti, Ni and other iron-group ions; phfit2 supplies them.
    # Neither covers Nb (Z = 41), whose bound-free opacity is therefore absent.
    supplement = read_verner_phfit2_database(
        data.verner_phfit2,
        elements=elements,
        maximum_charge=top_charge,
        exclude=tuple(photo.fits),
    )
    fits = {
        key: fit
        for key, fit in {**photo.fits, **supplement}.items()
        if key in atomic.ions
    }
    photo = VernerPhotoionizationDatabase(
        MappingProxyType(fits),
        source=photo.source + (
            "; Verner phfit2 (Verner & Yakovlev 1995) for "
            + ", ".join(f"{e} {c}" for e, c in sorted(supplement) if (e, c) in fits)
            if any(key in fits for key in supplement) else ""
        ),
    )
    formal_lines = (
        (1000 if config.quality == "quick" else 20_000)
        if config.formal_maximum_metal_lines is None
        else int(config.formal_maximum_metal_lines)
    )
    if formal_lines < 1:
        raise ValueError("metal line limits must be positive")
    return {
        "database": atomic,
        "photoionization": photo,
        # The shared charge closure references He nuclei.
        "helium_abundances": {
            element: float(value) + float(config.log_hydrogen_to_helium)
            for element, value in config.abundances.items()
        },
        "formal_lines": formal_lines,
        "ions_without_photoionization": sorted(
            f"{element} {charge}"
            for (element, charge), ion in atomic.ions.items()
            if ion.ionization_energy_ev is not None and (element, charge) not in fits
        ),
    }


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
    metals = _dab_trace_metals(config, data)
    metal_structure = {} if metals is None else dict(
        metal_database=metals["database"],
        metal_abundances=metals["helium_abundances"],
        metal_photoionization_database=metals["photoionization"],
        homogeneous_metal_host=True,
        metal_classical_electron_stark=config.metal_classical_electron_stark,
        include_dense_helium_metal_ionization=False,
        # The structure absorbs the synthesis line list, opacity-sampled.
        minimum_metal_oscillator_strength=(
            config.formal_minimum_metal_oscillator_strength
        ),
        maximum_metal_lines=metals["formal_lines"],
        metal_line_opacity_sampling_resolution=(
            config.structure_opacity_sampling_resolution
        ),
    )
    metal_synthesis = {} if metals is None else dict(
        metal_database=metals["database"],
        metal_abundances=metals["helium_abundances"],
        metal_photoionization_database=metals["photoionization"],
        metal_classical_electron_stark=config.metal_classical_electron_stark,
        include_dense_helium_metal_ionization=False,
        minimum_metal_oscillator_strength=(
            config.formal_minimum_metal_oscillator_strength
        ),
        maximum_metal_lines=metals["formal_lines"],
    )
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
        config.effective_temperature, config.balmer_self_broadening_prescription
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
            **metal_structure,
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
        **metal_synthesis,
    )
    return ModelResult(
        "DAB",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "DAB-production-v6",
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
                if config.allard_minimum_effective_temperature is not None
                else "delivered temperature-indexed Lyalpha--Lygamma at every "
                "Teff; local temperatures below 9000 K use the 9000-K table"
            ),
            "allard_minimum_effective_temperature_K": (
                config.allard_minimum_effective_temperature
            ),
            "balmer_self_broadening": self_broadening_prescription,
            "balmer_self_broadening_policy": (
                "automatic: Ali-Griem at every temperature"
                if config.balmer_self_broadening_prescription is None
                else "explicit override"
            ),
            "balmer_self_broadening_truncation_closure": (
                config.balmer_self_broadening_truncation_closure
            ),
            "convection": (
                f"ML2/alpha={config.mixing_length_alpha:g}"
                if config.mixing_length_alpha is not None
                else "suppressed; radiative equilibrium"
            ),
            "neutral_broadening": config.neutral_broadening,
            "atmosphere_solver": config.atmosphere_solver,
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
            "trace_metals": None if metals is None else {
                "abundance_reference": "hydrogen",
                "log_number_abundance_to_hydrogen": dict(config.abundances),
                "atomic_lines": "; ".join(sorted({
                    ion.source for ion in metals["database"].ions.values()
                })),
                "ion_stages": {
                    element: [ion.charge for ion in metals["database"].ion_stages(element)]
                    for element in metals["database"].elements
                },
                "photoionization": metals["photoionization"].source,
                "ions_without_photoionization": metals["ions_without_photoionization"],
                "metal_electron_feedback": "shared H/He/metal charge closure at fixed H and He nuclei",
                "metal_opacity_in_structure": True,
                "structure_opacity_sampling_resolution": config.structure_opacity_sampling_resolution,
                "formal_maximum_metal_lines": metals["formal_lines"],
                "minimum_metal_oscillator_strength": config.formal_minimum_metal_oscillator_strength,
                "metal_line_supplements": list(config.metal_line_supplements),
                "metal_line_broadening": (
                    "thermal, radiative, Unsold neutral H/He"
                    + (", SYNSPEC classical electron Stark"
                       if config.metal_classical_electron_stark else "")
                ),
                "metal_thermodynamic_derivatives": "trace-metal approximation: H/He mixture derivatives",
            },
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
        if wanted(element, charge)
    ]
    if not (tlusty or rap or sirocco or norad):
        return None
    data.require(
        *(entry[0] for entry in tlusty),
        *(entry[0] for entry in rap),
        *(path for entry in sirocco for path in entry[:2]),
        *(entry[0] for entry in norad),
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
    if config.helium_minus_free_free not in ("automatic", "partial-wave"):
        raise ValueError("helium_minus_free_free must be 'automatic' or 'partial-wave'")
    if config.dense_helium_free_free not in ("off", "rogers-young"):
        raise ValueError("dense_helium_free_free must be 'off' or 'rogers-young'")
    helium_minus_table = None
    if config.dense_helium_free_free == "rogers-young":
        from ..dense_helium_continuum import DenseHeliumCorrectionTable

        data.require(data.helium_minus_dense_free_free)
        helium_minus_table = DenseHeliumCorrectionTable(
            data.helium_minus_dense_free_free
        )
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
    if config.si_ii_kurucz_transitions:
        atomic = augment_silicon_ii_kurucz_transitions(atomic)
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
    if config.kurucz_electron_stark:
        atomic = atomic_database_with_kurucz_electron_stark(atomic)
    if config.si_ii_he_unified_profile:
        atomic = atomic_database_with_si_ii_he_unified_profile(atomic)
    if config.mg_i_b_he_unified_profile and "Mg" in elements:
        from ..metals import atomic_database_with_mg_i_b_he_unified_profile

        data.require(data.mg_i_b_he_unified_profiles)
        atomic = atomic_database_with_mg_i_b_he_unified_profile(
            atomic, data.mg_i_b_he_unified_profiles
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
    ca_ii_he = None
    mg_ii_he = None
    if config.ca_ii_he_unified_profile and "Ca" in elements:
        from ..metals import UnifiedIonicHeProfileGrid

        data.require(data.ca_ii_he_unified_profiles)
        ca_ii_he = UnifiedIonicHeProfileGrid(data.ca_ii_he_unified_profiles)
    if config.mg_ii_he_unified_profile and "Mg" in elements:
        from ..metals import UnifiedIonicHeProfileGrid

        data.require(data.mg_ii_he_unified_profiles)
        mg_ii_he = UnifiedIonicHeProfileGrid(data.mg_ii_he_unified_profiles)
    na_i_he = None
    if config.na_i_he_unified_profile and "Na" in elements:
        from ..metals import UnifiedIonicHeProfileGrid

        data.require(data.na_i_he_unified_profiles)
        na_i_he = UnifiedIonicHeProfileGrid(data.na_i_he_unified_profiles)
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
        config.effective_temperature, config.balmer_self_broadening_prescription
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
            mg_ii_he_profile_table=mg_ii_he,
            na_i_he_profile_table=na_i_he,
            ca_ii_he_profile_table=ca_ii_he,
            ca_i_he_profile_table=ca_i_he,
            helium_reos3_table=helium_reos3,
            helium_minus_correction_table=helium_minus_table,
            helium_minus_prescription=config.helium_minus_free_free,
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
            deepen_on_any_screen_failure=True,
            restart_after_deepening=True,
            deepening_pressure_factor=_DZ_DEEPENING_PRESSURE_FACTOR,
            early_deepening_escape_limit=_DZ_EARLY_DEEPENING_ESCAPE_LIMIT,
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
        mg_ii_he_profile_table=mg_ii_he,
        na_i_he_profile_table=na_i_he,
        ca_ii_he_profile_table=ca_ii_he,
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
        helium_minus_correction_table=helium_minus_table,
        helium_minus_prescription=config.helium_minus_free_free,
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
            "kurucz_electron_stark": config.kurucz_electron_stark,
            "si_ii_he_unified_profile": config.si_ii_he_unified_profile,
            "mg_i_b_he_unified_profile": config.mg_i_b_he_unified_profile,
            "mg_ii_he_unified_profile": (
                mg_ii_he.source if mg_ii_he is not None else False
            ),
            "ca_ii_he_unified_profile": (
                ca_ii_he.source if ca_ii_he is not None else False
            ),
            "na_i_he_unified_profile": (
                na_i_he.source if na_i_he is not None else False
            ),
            "si_ii_kurucz_transitions": config.si_ii_kurucz_transitions,
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
            "photoionization_population_model": spectrum.metadata.get(
                "metal_photoionization_population_model", "disabled"
            ),
            "metal_population_revision": "normalized-levels-rydberg-only-2026-10-24",
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
            "helium_minus_free_free": config.helium_minus_free_free,
            "dense_helium_free_free": (
                {"mode": config.dense_helium_free_free,
                 "table_sha256": helium_minus_table.sha256}
                if helium_minus_table is not None
                else "off"
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
                "automatic: Ali-Griem at every temperature"
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
