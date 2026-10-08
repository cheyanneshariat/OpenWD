"""Cold-start homogeneous PG 1159 models with the shared nonlinear solver."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Mapping
import hashlib
import numpy as np
from .common import (
    ModelData,
    ModelResult,
    Quality,
    validate_wavelength,
    numerical_resolution,
    AtmosphereConvergenceWarning,
)
from ..pg1159_presets import PG1424_PUBLISHED_MASS_FRACTIONS
from .._pg1159_builder import (
    PG1159_HYDROGENIC_STARK_COMPONENTS,
    PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES,
    build_model,
    structure_wavelength,
)
from .._pg1159_structure import solve_pg1159_atmosphere, PG1159Equations
from .._pg1159_transfer import transfer_field
from ..atmosphere import helium_continuum_atmosphere
from ..spectrum import Spectrum


@dataclass(frozen=True)
class PG1159Config:
    effective_temperature: float = 110000.0
    logg: float = 7.0
    mass_fractions: Mapping[str, float] = field(
        default_factory=lambda: dict(PG1424_PUBLISHED_MASS_FRACTIONS)
    )
    target_name: str = "PG 1424+535"
    quality: Quality = "standard"
    oxygen_atom: str = "extended54-complete"
    include_radiative_acceleration: bool = False
    population_maximum_iterations: int = 120
    population_tolerance: float = 1e-2
    # Converge the upper atmosphere (tau_Ross < 1e-2) to local radiative
    # equilibrium after certification and certify the refined structure.
    refine_upper_atmosphere: bool = False


@dataclass(frozen=True)
class PG1159PopulationResult:
    """Keep structural and trace line-formation states with their own material."""

    structure: object
    line_formation: object
    line_formation_atmosphere: object

    @property
    def converged(self):
        return bool(self.structure.converged and self.line_formation.converged)


def validate_config(config):
    if not isinstance(config, PG1159Config):
        raise TypeError("expected PG1159Config")
    if (
        not np.isfinite(config.effective_temperature)
        or config.effective_temperature <= 0
        or not np.isfinite(config.logg)
    ):
        raise ValueError("Teff must be positive and finite; logg must be finite")
    numerical_resolution(config.quality)
    if config.oxygen_atom not in ("compact14", "extended54-complete"):
        raise ValueError("unknown oxygen atom")
    if not isinstance(config.include_radiative_acceleration, bool):
        raise ValueError("include_radiative_acceleration must be boolean")
    if not isinstance(config.refine_upper_atmosphere, bool):
        raise ValueError("refine_upper_atmosphere must be boolean")
    if (
        isinstance(config.population_maximum_iterations, bool)
        or not isinstance(config.population_maximum_iterations, int)
        or config.population_maximum_iterations < 1
    ):
        raise ValueError("population_maximum_iterations must be a positive integer")
    if (
        not np.isfinite(config.population_tolerance)
        or not 0 < config.population_tolerance <= 1e-2
    ):
        raise ValueError("population_tolerance must lie in (0, 1e-2]")
    if any(not np.isfinite(v) or v < 0 for v in config.mass_fractions.values()) or any(
        config.mass_fractions.get(e, 0) <= 0 for e in ("He", "C", "O")
    ):
        raise ValueError(
            "mass fractions must be finite and nonnegative with positive He, C, and O"
        )


def required_atomic_files(data, oxygen_atom):
    cache = data.cache
    paths = [
        data.ccc_hydrogen_collisions,
        data.tlusty_source,
        data.tlusty_helium_atom,
        cache / "helium-stark/Tremblay26.txt",
        data.helium_ii_stark,
        data.verner_photoionization,
    ]
    paths += [
        cache / "tlusty-atoms" / n
        for n in ("c3.dat", "c4_35+2lev.dat", "o4.dat", "o5.dat", "o6.dat")
    ]
    paths += [
        cache / "tmad-atoms" / n
        for n in (
            "C_III-V",
            "O_III-VII",
            "C_IV_syn",
            "O_III_syn",
            "O_IV_syn",
            "O_V_syn",
            "O_VI_syn",
            "O_VII_syn",
        )
    ]
    if oxygen_atom == "extended54-complete":
        paths += [
            cache / "sirocco-atomic/o_6_levels.dat",
            cache / "sirocco-atomic/o_6_phot.dat",
            cache / "chianti/o_6/o_6.scups",
        ]
    return tuple(paths)


def compute_pg1159(
    config=PG1159Config(),
    wavelength=None,
    *,
    data=None,
    iteration_callback=None,
    output=None,
):
    """Build a fresh He/C/O atmosphere; return the common ModelResult.

    Additional listed elements are included in the final line-formation
    calculation, as in the development model. Their opacity is not fed back
    into the structural solve. No previous model/checkpoint is loaded.
    """
    validate_config(config)
    wave = validate_wavelength(wavelength)
    data = ModelData.default() if data is None else data
    files = required_atomic_files(data, config.oxygen_atom)
    missing = [str(p) for p in files if not p.is_file()]
    if not (data.stout / "stout").is_dir():
        missing.append(str(data.stout / "stout"))
    if missing:
        raise FileNotFoundError(
            "PG1159 requires atomic data at ModelData/OPENWD_DATA; see docs/models/PG1159.md. Missing: "
            + ", ".join(missing)
        )
    full = {
        e: float(v) / sum(config.mass_fractions.values())
        for e, v in config.mass_fractions.items()
        if v > 0
    }
    heco = {
        e: full[e] / sum(full[k] for k in ("He", "C", "O")) for e in ("He", "C", "O")
    }
    resolution = numerical_resolution(config.quality)
    # The PG1159 validation model is converged with the same 120-point,
    # two-angle quadrature used by the development calculation.  Reusing the
    # package-wide 300/3 standard atmosphere grid increased transfer work by
    # nearly fourfold without changing the requested stellar model. Reserve
    # the denser quadrature for production-quality resolution studies.
    structure_continuum, structure_angles = {
        "quick": (120, 2),
        "standard": (120, 2),
        "production": (300, 3),
    }[config.quality]
    model, _ = build_model(
        data,
        heco,
        structure_only=True,
        population_iterations=config.population_maximum_iterations,
        oxygen_atom_preset=config.oxygen_atom,
    )
    model = replace(
        model,
        population_transfer="mass",
        use_population_ali=False,
        coupled_population_acceleration_depth=80,
        metal_population_acceleration_depth=0,
        metal_population_relative_tolerance=config.population_tolerance,
        helium_model=replace(model.helium_model, population_n_angle=structure_angles),
    )
    seed = helium_continuum_atmosphere(
        config.effective_temperature,
        config.logg,
        n_depth=resolution.n_depth,
        rosseland_frequency_points=structure_continuum,
        # The validated legacy PG1159 structures end at tau_Ross=100.  The
        # extra public decade to 1000 altered the hydrostatic column-mass grid
        # without contributing to the emergent spectrum and introduced the
        # deep, ill-conditioned residuals seen during migration.
        tau_max=100.0,
    )
    seed = model.rebuild_atmosphere(seed, seed.temperature, None)
    from .._pg1159_reference import gray_opacity_seed

    seed = gray_opacity_seed(
        model, seed, wavelength_points=structure_continuum
    )
    refinement = None
    if config.refine_upper_atmosphere:
        from .._pg1159_ali_temperature import REFINEMENT_LINE_VELOCITY_SAMPLES_KMS

        refinement_model = replace(
            model,
            metal_rate_line_velocity_samples_kms=REFINEMENT_LINE_VELOCITY_SAMPLES_KMS,
        )
        # The refinement holds the pressure structure fixed; radiation
        # pressure, when requested, enters the coupled Newton
        # re-certification that follows (a separate pressure update inside
        # the temperature refinement does not converge).
        refinement = (
            refinement_model,
            structure_wavelength(refinement_model, structure_continuum),
            {"include_radiative_acceleration": False},
        )
    result = solve_pg1159_atmosphere(
        seed,
        model,
        structure_wavelength(model, structure_continuum),
        maximum_iterations=resolution.maximum_iterations,
        cold_start=True,
        refinement=refinement,
        include_radiative_acceleration=config.include_radiative_acceleration,
        iteration_callback=(
            None
            if iteration_callback is None
            else lambda i, a, p, d: iteration_callback(i, a, d)
        ),
    )
    from .._convergence import recorded_equilibrium_status
    import warnings

    status = recorded_equilibrium_status(result.atmosphere.metadata)
    formal_model = model
    state = result.population_state
    atmosphere = result.atmosphere
    if set(full) - {"He", "C", "O"}:
        formal_model, _ = build_model(
            data,
            full,
            structure_only=False,
            population_iterations=config.population_maximum_iterations,
            oxygen_atom_preset=config.oxygen_atom,
        )
        formal_model = replace(
            formal_model,
            population_transfer="mass",
            use_population_ali=False,
            # Match the validated structural closure: the 80-state weighted
            # SVD history resolves weak population modes, followed by the
            # undamped physical map near the target. The builder's 0.25
            # damping made the expanded formal atom spend tens of costly
            # iterations applying quarter-sized updates.
            coupled_population_acceleration_depth=(
                model.coupled_population_acceleration_depth
            ),
            metal_population_acceleration_depth=0,
            metal_population_damping=1.0,
            helium_population_damping=1.0,
            # The expanded trace-element atom only affects the formal
            # spectrum.  Its fixed point develops a roughly 0.6% numerical
            # floor while the optical spectrum is already stable, so do not
            # force the formal solve below one percent.  The He/C/O structure
            # is independently certified at config.population_tolerance.
            metal_population_relative_tolerance=max(
                config.population_tolerance, 1.0e-2
            ),
            helium_model=replace(
                formal_model.helium_model, population_n_angle=structure_angles
            ),
        )
        formal_equations = PG1159Equations(
            atmosphere,
            formal_model,
            structure_wavelength(formal_model, structure_continuum),
            radiative_acceleration=False,
        )
        formal_equations.anchor = state
        atmosphere, state = formal_equations.material(formal_equations.initial_state())
    c = formal_model.transfer_coefficients(atmosphere, wave, state)
    _, radiation, closure = transfer_field(atmosphere, c, n_angle=structure_angles)
    spectrum = Spectrum(
        wave,
        radiation.interface_flux[:, 0],
        {
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "flux_convention": "surface F_lambda",
            "wavelength_medium": "vacuum",
            "composition": "PG1159",
            "line_formation": "He/C/O term-resolved NLTE with trace ion-stage NLTE",
            "source_closure_residual": closure,
        },
    )
    qualified_status = status if state.converged and closure < 1e-6 else "unconverged"
    if qualified_status != "converged":
        detail = (
            "; the spectrum passed its declared qualification profile"
            if qualified_status == "spectrum-qualified"
            else ""
        )
        warnings.warn(
            "PG1159 model has not passed the full equilibrium certificate"
            + detail,
            AtmosphereConvergenceWarning,
            stacklevel=2,
        )
    # Record only the Stout elements actually used by this requested mixture.
    stout_files = tuple(
        p
        for element in full
        if element != "He"
        for p in sorted((data.stout / "stout" / element.lower()).glob("*/*"))
        if p.suffix in (".nrg", ".tp")
    )

    def data_key(path):
        try:
            return str(path.relative_to(data.root))
        except ValueError:
            return str(path.resolve())

    identity = {
        data_key(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (*files, *stout_files)
    }
    value = ModelResult(
        "PG1159",
        result.atmosphere,
        spectrum,
        config,
        {
            "experimental": True,
            "cold_start": True,
            "stellar_parameters_fixed": True,
            "mass_fractions": full,
            "structure_mass_fractions": heco,
            "trace_opacity_in_structure": False,
            "oxygen_atom": config.oxygen_atom,
            "structure_continuum_points": structure_continuum,
            "structure_angle_points": structure_angles,
            "structure_maximum_rosseland_depth": 100.0,
            "carbon_atom_counts": dict(model.carbon_levels_per_charge),
            "oxygen_atom_counts": dict(model.oxygen_levels_per_charge),
            "helium_ii_collision_model": model.helium_model.hydrogenic_collision_model,
            "ccc_maximum_shell": model.helium_model.collision_data.maximum_level,
            "helium_ii_collision_fallback": "Mihalas above CCC shell limit",
            "atmosphere_convergence_status": qualified_status,
            "structure_convergence_status": status,
            "spectral_qualification": qualified_status == "spectrum-qualified",
            "atomic_data_sha256": identity,
            "formal_population_converged": state.converged,
            "formal_population_defect": getattr(state, "metadata", {}).get(
                "undamped_population_defect"
            ),
            "formal_electron_closure_residual": getattr(state, "metadata", {}).get(
                "electron_closure_residual"
            ),
            "population_state_layout": "structure, line_formation, line_formation_atmosphere",
            "independent_grid_validation": False,
            "full_physics_validation": False,
            "hydrogenic_linear_stark_components": PG1159_HYDROGENIC_STARK_COMPONENTS,
            "static_linear_stark_frequency_scales": dict(
                PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES
            ),
        },
        population_state=PG1159PopulationResult(
            result.population_state, state, atmosphere
        ),
    )
    if output is not None:
        from .common import save_model_result

        save_model_result(value, output)
    return value
