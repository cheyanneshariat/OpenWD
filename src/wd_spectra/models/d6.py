"""Hydrogen/helium-free carbon/oxygen (D6) models with the shared solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..atmosphere import Atmosphere
from ..d6 import (
    D6_STOUT_MAXIMUM_CHARGE,
    D6_TLUSTY_RAP_FILES,
    D6_TLUSTY_TOPBASE_FILES,
    SDSS_J1637_EFFECTIVE_TEMPERATURE,
    SDSS_J1637_LOGG,
    SDSS_J1637_LOG_NUMBER_ABUNDANCE,
    d6_tlusty_excitation_energy_overrides,
    merge_topbase_photoionization_databases,
    radiative_equilibrium_d6_atmosphere,
    read_sirocco_topbase_lte_photoionization,
    read_tlusty_rap_lte_photoionization,
    read_tlusty_topbase_lte_photoionization,
    retain_d6_structure_line_stages,
    synthesize_d6_spectrum,
)
from ..metals import (
    _canonical_element,
    augment_oxygen_i_6258_6271_multiplet,
    read_stout_atomic_database,
    read_verner_photoionization_database,
)
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


@dataclass(frozen=True)
class D6Config:
    """Warm hydrogen/helium-free C/O-dominated LTE atmosphere.

    The defaults are the Hollands et al. (2025) spectroscopic solution for
    SDSS J1637+3631.  ``abundances`` are log10 number ratios relative to
    ``reference_element``, whose own entry must be zero.  Hydrogen and helium
    are absent, not trace: every supplied element contributes to the
    pressure, charge and mass closure.
    """

    effective_temperature: float = SDSS_J1637_EFFECTIVE_TEMPERATURE
    logg: float = SDSS_J1637_LOGG
    abundances: Mapping[str, float] = field(
        default_factory=lambda: dict(SDSS_J1637_LOG_NUMBER_ABUNDANCE)
    )
    reference_element: str = "C"
    quality: Quality = "standard"
    mixing_length_alpha: float | None = 1.25
    structure_maximum_metal_lines: int | None = None
    formal_maximum_metal_lines: int | None = None
    include_rydberg_dissolution: bool = True
    include_metal_series_pseudocontinuum: bool = True


# Structural line budgets.  Hollands et al. stress that the dense UV forest
# must be in the structure; 25,000 flux-ranked lines is the budget of the
# related Koester-code D6 calculations (Bhat et al. 2026).
_STRUCTURE_LINES = {"quick": 2_000, "standard": 25_000, "production": 25_000}
_FORMAL_LINES = {"quick": 5_000, "standard": 25_000, "production": 25_000}
# The D6 line-forming layers need finer depth sampling than a trace-metal
# DZ: the optical continuum and the strongest O I cores form four decades
# apart in Rosseland depth.
_N_DEPTH = {"quick": 24, "standard": 48, "production": 72}


def default_d6_wavelength_grid(quality: Quality = "standard") -> np.ndarray:
    """Optical grid matching the Hollands et al. (2025) GTC coverage."""

    step = {"quick": 0.5, "standard": 0.1, "production": 0.05}[quality]
    return np.arange(3400.0, 7700.0 + 0.5 * step, step)


def _validated_abundances(config: D6Config) -> dict[str, float]:
    for element in (*config.abundances, config.reference_element):
        if str(element).strip().capitalize() in ("H", "He"):
            raise ValueError(
                "D6 atmospheres are hydrogen/helium-free; use DZ or DQ for "
                "helium-dominated mixtures"
            )
    reference = _canonical_element(config.reference_element)
    abundances = {
        _canonical_element(element): float(value)
        for element, value in config.abundances.items()
    }
    if reference not in abundances:
        raise ValueError("the reference element must be listed with abundance 0")
    for value in abundances.values():
        if not np.isfinite(value):
            raise ValueError("abundances must be finite")
    if abundances[reference] != 0.0:
        raise ValueError("the reference element must have log abundance zero")
    return abundances


def _atomic_inputs(data: ModelData, elements: tuple[str, ...]):
    tlusty = {
        filename: (element, charge)
        for filename, (_, _, element, charge) in {
            **D6_TLUSTY_TOPBASE_FILES,
            **D6_TLUSTY_RAP_FILES,
        }.items()
        if element in elements
    }
    sirocco = [
        (data.sirocco_atomic / f"{element.lower()}_2_levels.dat",
         data.sirocco_atomic / f"{element.lower()}_2_phot.dat", element, 1)
        for element in ("C", "O")
        if element in elements
    ]
    data.require(
        data.stout,
        data.verner_photoionization,
        *(data.tlusty_atoms / name for name in tlusty),
        *(path for entry in sirocco for path in entry[:2]),
    )
    maximum_charge = {
        element: D6_STOUT_MAXIMUM_CHARGE.get(element, 2) for element in elements
    }
    database = read_stout_atomic_database(
        data.stout, elements=elements, maximum_charge=maximum_charge
    )
    database = retain_d6_structure_line_stages(database)
    if "O" in elements:
        database = augment_oxygen_i_6258_6271_multiplet(database)
    # Verner fits supply the ground-state continuum of every loaded stage
    # that has no level-resolved atom (replacement is stage-specific).
    photoionization = read_verner_photoionization_database(
        data.verner_photoionization,
        elements=elements,
        maximum_charge=max(maximum_charge.values()),
    )
    parts = []
    if sirocco:
        parts.append(read_sirocco_topbase_lte_photoionization(sirocco, database))
    bulk = [
        (data.tlusty_atoms / name, element, charge)
        for name, (element, charge) in tlusty.items()
        if (element, charge) in (("C", 0), ("O", 0))
    ]
    if bulk:
        parts.append(read_tlusty_topbase_lte_photoionization(
            bulk,
            database,
            excitation_energy_overrides_ev=d6_tlusty_excitation_energy_overrides(
                database,
                elements=tuple(element for element in ("C", "O") if element in elements),
            ),
        ))
    trace = [
        (data.tlusty_atoms / name, element, charge)
        for name, (element, charge) in tlusty.items()
        if name.endswith(".dat") and (element, charge) not in (("C", 0), ("O", 0))
    ]
    if trace:
        parts.append(read_tlusty_topbase_lte_photoionization(trace, database))
    rap = [
        (data.tlusty_atoms / name, element, charge)
        for name, (element, charge) in tlusty.items()
        if name.endswith(".rap")
    ]
    if rap:
        parts.append(read_tlusty_rap_lte_photoionization(rap, database))
    topbase = merge_topbase_photoionization_databases(*parts) if parts else None
    return database, photoionization, topbase


def compute_d6(
    config: D6Config = D6Config(),
    wavelength: ArrayLike | None = None,
    *,
    data: ModelData | None = None,
    initial_atmosphere: Atmosphere | None = None,
    relax_atmosphere: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ] | None = None,
) -> ModelResult:
    """Calculate one warm D6 atmosphere and its optical spectrum.

    The structure is solved from a cold start (or from ``initial_atmosphere``
    as a warm start) with the shared adaptive trust-region Newton/ML2 solver
    and its equilibrium certification.  ``relax_atmosphere=False`` performs
    formal synthesis on a supplied structure without claiming convergence
    for a different request.
    """

    if not isinstance(config, D6Config):
        raise TypeError("expected D6Config")
    if not np.isfinite(config.effective_temperature) or config.effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(config.logg):
        raise ValueError("logg must be finite")
    abundances = _validated_abundances(config)
    data = ModelData.default() if data is None else data
    resolution = numerical_resolution(config.quality)
    request_fingerprint = model_request_fingerprint("D6", config, data)
    checkpoint_matches_request = atmosphere_matches_model_request(
        initial_atmosphere, request_fingerprint
    )
    wave = (
        default_d6_wavelength_grid(config.quality)
        if wavelength is None
        else validate_wavelength(wavelength)
    )
    structure_lines = (
        _STRUCTURE_LINES[config.quality]
        if config.structure_maximum_metal_lines is None
        else int(config.structure_maximum_metal_lines)
    )
    formal_lines = (
        _FORMAL_LINES[config.quality]
        if config.formal_maximum_metal_lines is None
        else int(config.formal_maximum_metal_lines)
    )
    if structure_lines < 0 or formal_lines < 0:
        raise ValueError("metal line limits must be non-negative")
    elements = tuple(abundances)
    database, photoionization, topbase = _atomic_inputs(data, elements)
    if not relax_atmosphere and initial_atmosphere is None:
        raise ValueError("relax_atmosphere=False requires initial_atmosphere")
    atmosphere = initial_atmosphere
    if relax_atmosphere:
        atmosphere = radiative_equilibrium_d6_atmosphere(
            config.effective_temperature,
            config.logg,
            database,
            photoionization,
            abundances,
            reference_element=config.reference_element,
            n_depth=_N_DEPTH[config.quality],
            max_iterations=resolution.maximum_iterations,
            maximum_metal_lines=structure_lines,
            minimum_metal_oscillator_strength=1.0e-4,
            include_rydberg_dissolution=config.include_rydberg_dissolution,
            include_metal_series_pseudocontinuum=(
                config.include_metal_series_pseudocontinuum
            ),
            topbase_photoionization_database=topbase,
            n_angle=min(resolution.n_angle, 3),
            mixing_length_alpha=config.mixing_length_alpha,
            initial_temperature=(
                None if initial_atmosphere is None else initial_atmosphere.temperature
            ),
            initial_column_mass=(
                None if initial_atmosphere is None else initial_atmosphere.column_mass
            ),
            initial_rosseland_optical_depth=(
                None
                if initial_atmosphere is None
                else initial_atmosphere.rosseland_optical_depth
            ),
            resume_supplied_structure_in_formal_flux_phase=checkpoint_matches_request,
            iteration_callback=iteration_callback,
        )
    assert atmosphere is not None
    if relax_atmosphere:
        atmosphere = atmosphere_with_model_request_fingerprint(
            atmosphere, request_fingerprint
        )
    else:
        atmosphere = fixed_synthesis_atmosphere(atmosphere, request_fingerprint)
    convergence_status = warn_if_atmosphere_not_converged(atmosphere, "D6")
    spectrum = synthesize_d6_spectrum(
        atmosphere,
        wave,
        database,
        photoionization,
        abundances,
        reference_element=config.reference_element,
        minimum_metal_oscillator_strength=1.0e-6,
        maximum_metal_lines=formal_lines,
        topbase_photoionization_database=topbase,
        include_rydberg_dissolution=config.include_rydberg_dissolution,
        include_metal_series_pseudocontinuum=(
            config.include_metal_series_pseudocontinuum
        ),
        n_angle=resolution.n_angle,
    )
    return ModelResult(
        "D6",
        atmosphere,
        spectrum,
        config,
        {
            "preset": "D6-shared-solver-v1",
            "default_parameter_source": (
                "Hollands et al. (2025), SDSS J1637+3631"
            ),
            "log_number_abundance_relative_to_reference": abundances,
            "reference_element": _canonical_element(config.reference_element),
            "atomic_lines": database.source,
            "bound_free": (
                (topbase.source + "; " if topbase is not None else "")
                + "Verner ground-state fits for the remaining stages"
            ),
            "structure_maximum_metal_lines": structure_lines,
            "formal_maximum_metal_lines": formal_lines,
            "microfields": (
                "Q-MHD Rydberg survival"
                if config.include_rydberg_dissolution else "disabled"
            ),
            "metal_series_pseudocontinuum": (
                "TOPbase O I and Mg"
                if config.include_metal_series_pseudocontinuum else "disabled"
            ),
            "convection": (
                "disabled"
                if config.mixing_length_alpha is None
                else f"ML2/alpha={config.mixing_length_alpha:g}"
            ),
            "atmosphere_solver": "adaptive-newton (shared)",
            "atmosphere_mode": (
                "relaxed" if relax_atmosphere else "checkpoint formal synthesis"
            ),
            "atmosphere_convergence_status": convergence_status,
            "checkpoint_matches_model_request": checkpoint_matches_request,
            "model_request_fingerprint": request_fingerprint,
        },
    )
