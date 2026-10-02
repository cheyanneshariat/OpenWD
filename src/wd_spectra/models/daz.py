"""Hydrogen-host polluted atmospheres, using the current shared DA solver.

The metal mixture participates in charge closure and structural opacity; this
is not a pure-DA atmosphere with metal lines painted onto its final spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping
import numpy as np

from ..atmosphere import radiative_equilibrium_hydrogen_atmosphere
from ..metals import (
    NIST_ASD_STRONG_ION_FILES,
    read_barklem_neutral_hydrogen_broadening,
    read_nist_asd_strong_atomic_database,
    read_stout_atomic_database,
    read_verner_photoionization_database,
)
from ..molecules import read_borysow_h2_h2_cia_table
from ..spectrum import synthesize_hydrogen_spectrum
from .common import (
    ModelData,
    ModelResult,
    Quality,
    atmosphere_matches_model_request,
    atmosphere_with_model_request_fingerprint,
    fixed_synthesis_atmosphere,
    model_request_fingerprint,
    numerical_resolution,
    validate_wavelength,
    warn_if_atmosphere_not_converged,
)
from .stellar import (
    _allard_lyman_profiles_for_effective_temperature,
    _da_self_broadening_prescription,
    _metal_structure_line_budget,
)


@dataclass(frozen=True)
class DAZConfig:
    """Polluted hydrogen atmosphere; abundances are log10 N(Z)/N(H).

    The default composition is G29-38. Stout line strengths preserve the
    paper's metal-line policy; NIST replacements are an explicit alternative.
    """

    effective_temperature: float = 11_820.0
    logg: float = 8.40
    abundances: Mapping[str, float] = field(
        default_factory=lambda: {
            "C": -6.90,
            "O": -5.00,
            "Mg": -5.77,
            "Si": -5.60,
            "Ca": -6.58,
            "Ti": -7.90,
            "Cr": -7.51,
            "Fe": -5.90,
        }
    )
    quality: Quality = "standard"
    lyman_profile_source: Literal["allard", "stark"] = "allard"
    allard_minimum_effective_temperature: float = 9_000.0
    include_molecules: bool | None = None
    h3plus_partition_model: Literal["neale-tennyson-1995", "none"] = (
        "neale-tennyson-1995"
    )
    mixing_length_alpha: float | None = 0.7
    balmer_self_broadening_prescription: str | None = None
    balmer_self_broadening_truncation_closure: str = "stark-core"
    atmosphere_solver: Literal["adaptive-newton"] = "adaptive-newton"
    maximum_metal_charge: int = 3
    # Flux conservation, as for DZ: the structure absorbs the opacity-sampled
    # synthesis line list on photosphere-concentrated depths, and the formal
    # solution subdivides each depth interval.  None/0/1 restore the
    # historical line-centered structure and bare-grid formal solution.
    structure_maximum_metal_lines: int | None = None
    structure_minimum_metal_oscillator_strength: float | None = None
    structure_opacity_sampling_resolution: float | None = 1000.0
    photospheric_depth_concentration: float = 1.0
    formal_maximum_metal_lines: int | None = None
    formal_minimum_metal_oscillator_strength: float = 1.0e-4
    synthesis_transfer_depth_refinement: int = 4
    ca_ii_resonance_source: Literal["chianti-reduced", "lte"] = "chianti-reduced"
    metal_neutral_h_broadening: Literal["barklem", "unsold"] = "unsold"
    strong_line_atomic_data: Literal["stout", "nist-asd"] = "stout"


def compute_daz(
    config=DAZConfig(),
    wavelength=None,
    *,
    data=None,
    initial_atmosphere=None,
    relax_atmosphere=True,
    iteration_callback=None,
):
    """Solve a DAZ from scratch, or explicitly synthesize a diagnostic fixed state.

    Fixed-state synthesis is warned and fingerprinted just like the other
    public presets. run_model permits only the cold-start path.
    """
    if (
        not np.isfinite(config.effective_temperature)
        or config.effective_temperature <= 0
        or not np.isfinite(config.logg)
    ):
        raise ValueError(
            "effective temperature and logg must be finite, with Teff positive"
        )
    if not config.abundances or any(
        not np.isfinite(value) for value in config.abundances.values()
    ):
        raise ValueError("DAZ abundances must contain finite log10 N(Z)/N(H) values")
    if set(config.abundances) & {"H", "He"}:
        raise ValueError("DAZ abundances specify metals relative to hydrogen, not H/He")
    for name, choices in (
        ("metal_neutral_h_broadening", ("barklem", "unsold")),
        ("ca_ii_resonance_source", ("chianti-reduced", "lte")),
        ("strong_line_atomic_data", ("stout", "nist-asd")),
        ("lyman_profile_source", ("allard", "stark")),
        ("h3plus_partition_model", ("neale-tennyson-1995", "none")),
    ):
        if getattr(config, name) not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    if config.maximum_metal_charge < 1:
        raise ValueError("maximum_metal_charge must be positive")
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    data = ModelData.default() if data is None else data
    request = model_request_fingerprint(
        "DAZ", config, data, physical_data_identity={"daz_interface_revision": 1}
    )
    resolution = numerical_resolution(config.quality)
    wave = validate_wavelength(wavelength)
    data.require(data.stout, data.verner_photoionization)
    elements = tuple(config.abundances)
    atomic = read_stout_atomic_database(
        data.stout, elements=elements, maximum_charge=config.maximum_metal_charge
    )
    if config.strong_line_atomic_data == "nist-asd":
        paths = [
            data.nist_asd_strong / filename
            for filename, (_, _, element, charge) in NIST_ASD_STRONG_ION_FILES.items()
            if element in elements and charge <= config.maximum_metal_charge
        ]
        if paths:
            data.require(*paths)
            atomic = read_nist_asd_strong_atomic_database(paths, atomic)
    if config.metal_neutral_h_broadening == "barklem":
        data.require(data.barklem_neutral_h_broadening)
        atomic = read_barklem_neutral_hydrogen_broadening(
            data.barklem_neutral_h_broadening, atomic
        )
    photo = read_verner_photoionization_database(
        data.verner_photoionization,
        elements=elements,
        maximum_charge=config.maximum_metal_charge,
    )
    reduced_ca = "Ca" in elements and config.ca_ii_resonance_source == "chianti-reduced"
    if reduced_ca:
        data.require(data.ca_ii_chianti_collisions)
    structure_lines, line_policy, metal_fraction = _metal_structure_line_budget(
        config.quality, config.abundances, config.structure_maximum_metal_lines
    )
    formal_lines = (
        (1000 if config.quality == "quick" else 20_000)
        if config.formal_maximum_metal_lines is None
        else int(config.formal_maximum_metal_lines)
    )
    if formal_lines < 1:
        raise ValueError("metal line limits must be positive")
    if config.structure_opacity_sampling_resolution is not None:
        # Flux conservation needs the structure to absorb the synthesis line
        # list; opacity sampling keeps its cost nearly independent of it.
        if config.structure_maximum_metal_lines is None:
            structure_lines = formal_lines
            line_policy = "opacity sampling: synthesis line list"
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
    allard = None
    if config.lyman_profile_source == "allard":
        allard = _allard_lyman_profiles_for_effective_temperature(
            config.effective_temperature,
            data,
            minimum_effective_temperature=config.allard_minimum_effective_temperature,
        )
    molecules = (
        config.effective_temperature <= 12_000.0
        if config.include_molecules is None
        else config.include_molecules
    )
    cia = None
    if molecules:
        data.require(data.h2_h2_cia)
        cia = read_borysow_h2_h2_cia_table(data.h2_h2_cia)
    h3 = (
        None
        if config.h3plus_partition_model == "none"
        else config.h3plus_partition_model
    )
    broadening = _da_self_broadening_prescription(
        config.effective_temperature, config.balmer_self_broadening_prescription
    )
    # Deliberately share the exact data objects and composition between the
    # atmosphere and synthesis. No helium dense-EOS/ionization prescription.
    shared = dict(
        include_series_pseudocontinuum=True,
        unified_allard_table=allard,
        h2_h2_cia_table=cia,
        balmer_self_broadening_prescription=broadening,
        balmer_self_broadening_truncation_closure=config.balmer_self_broadening_truncation_closure,
        metal_database=atomic,
        metal_abundances=config.abundances,
        metal_photoionization_database=photo,
    )
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        atmosphere = radiative_equilibrium_hydrogen_atmosphere(
            config.effective_temperature,
            config.logg,
            n_depth=100 if config.quality == "production" else resolution.n_depth,
            n_continuum_wavelength=resolution.n_continuum,
            max_iterations=resolution.maximum_iterations,
            n_angle=min(resolution.n_angle, 3),
            correlated_microfields=True,
            include_molecules=molecules,
            include_negative_hydrogen=molecules,
            trihydrogen_ion_partition_model=h3,
            mixing_length_alpha=config.mixing_length_alpha,
            structure_solver=config.atmosphere_solver,
            minimum_metal_oscillator_strength=structure_minimum_oscillator_strength,
            maximum_metal_lines=structure_lines,
            metal_line_opacity_sampling_resolution=(
                config.structure_opacity_sampling_resolution
            ),
            depth_concentration=config.photospheric_depth_concentration,
            initial_temperature=None if atmosphere is None else atmosphere.temperature,
            initial_column_mass=None if atmosphere is None else atmosphere.column_mass,
            iteration_callback=iteration_callback,
            **shared,
        )
        atmosphere = atmosphere_with_model_request_fingerprint(atmosphere, request)
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request)
    status = warn_if_atmosphere_not_converged(atmosphere, "DAZ")
    spectrum = synthesize_hydrogen_spectrum(
        atmosphere,
        wave,
        include_molecular_absorption=molecules,
        include_dense_helium_metal_ionization=False,
        minimum_metal_oscillator_strength=config.formal_minimum_metal_oscillator_strength,
        maximum_metal_lines=formal_lines,
        transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
        ca_ii_resonance_collision_strengths=(
            str(data.ca_ii_chianti_collisions) if reduced_ca else None
        ),
        n_angle=resolution.n_angle,
        **shared,
    )
    return ModelResult(
        "DAZ",
        atmosphere,
        spectrum,
        config,
        dict(
            preset="DAZ-hydrogen-metals-v1",
            abundance_reference="hydrogen",
            atomic_lines=getattr(atomic, "source", "Stout"),
            strong_line_atomic_data=config.strong_line_atomic_data,
            metal_neutral_hydrogen_broadening=config.metal_neutral_h_broadening,
            maximum_metal_charge=config.maximum_metal_charge,
            metal_electron_feedback=True,
            metal_opacity_in_structure=True,
            structure_maximum_metal_lines=structure_lines,
            structure_metal_line_budget_policy=line_policy,
            structure_minimum_metal_oscillator_strength=structure_minimum_oscillator_strength,
            structure_opacity_sampling_resolution=config.structure_opacity_sampling_resolution,
            photospheric_depth_concentration=config.photospheric_depth_concentration,
            synthesis_transfer_depth_refinement=config.synthesis_transfer_depth_refinement,
            summed_metal_number_fraction=metal_fraction,
            metal_thermodynamic_derivatives="trace-metal approximation: Q-MHD hydrogen derivatives",
            molecular_equilibrium=molecules,
            h3plus_partition_model=h3,
            balmer_self_broadening=broadening,
            dense_helium_ionization=False,
            atmosphere_solver=config.atmosphere_solver,
            ca_ii_resonance_source=config.ca_ii_resonance_source,
            ca_ii_resonance_source_applied_in_structure=False,
            atmosphere_mode=(
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            atmosphere_initialization=(
                "gray" if initial_atmosphere is None else "caller checkpoint"
            ),
            atmosphere_convergence_status=status,
            model_request_fingerprint=request,
            checkpoint_matches_model_request=atmosphere_matches_model_request(
                initial_atmosphere, request
            ),
        ),
    )
