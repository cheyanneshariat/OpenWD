"""Packaged assembly of the accepted PG 1159 atoms; no research-script imports."""
from __future__ import annotations
from dataclasses import replace
from types import MappingProxyType
import numpy as np
from ._pg1159_helium import CoupledHeliumNLTEModel
from .pg1159 import PG1159NLTEModel
from .pg1159_oxygen import pg1159_model_with_extended_oxygen_vi
from .pg1159_presets import (
    PG1424_BEST_LINE_ATOM_COUNTS,
    PG1424_STRUCTURE_LINE_VELOCITY_SAMPLES_KMS,
    mutable_atom_counts,
)
from .helium import HELIUM_I_LINES, HELIUM_I_RESONANCE_LINES, HELIUM_II_LINES
from .helium_stark import read_helium_stark_table
from .helium_ii_stark import read_helium_ii_stark_table
from .helium_collisions import read_tlusty_helium_collision_data
from .multilevel_nlte import read_ccc_hydrogen_collision_data
from .metals import read_pg1159_atomic_database, read_verner_photoionization_database
from .light_metal_nlte import (
    read_tlusty_photoionization_threshold_data,
    atomic_database_with_tmad_formal_ions,
    atomic_database_with_tmad_profile_parameters,
    read_tmad_structure_model_atom,
    reduced_light_metal_wavelength,
)

SUPPORTED_NLTE_TRACE_ELEMENTS = (
    "N",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "S",
    "Ar",
    "Ca",
    "Fe",
)
# Formula-4 (C IV, O VI) quasi-static wings use the hydrogenic linear-Stark
# component pattern of each n_l -> n_u transition (see _hydrogenic_stark)
# instead of one Holtsmark distribution at the outermost component's scale.
# This removes the need for any series- or ion-specific wing scale: the
# pattern alone narrows the C IV 4-9 far wings to ~0.27 and the O VI 7-8
# (5290 A) wing to ~0.013 of the formula-4 value.
PG1159_HYDROGENIC_STARK_COMPONENTS = True
PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES = MappingProxyType({})


def _build_model_inputs(data):
    helium_i = read_helium_stark_table(data.cache / "helium-stark/Tremblay26.txt")
    helium_ii = read_helium_ii_stark_table(
        data.cache / "helium-stark/he2prf.dat",
        thermodynamic_interpolation="synspec-quadratic",
    )
    hydrogen_collisions = read_ccc_hydrogen_collision_data(
        data.cache / "ccc/e-H_XSEC_LS.zip", maximum_level=8
    )
    helium_i_collisions = read_tlusty_helium_collision_data(
        data.cache / "tlusty-source/tlusty200.f",
        data.cache / "tlusty-atoms/he1_14lev.dat",
    )
    helium_model = CoupledHeliumNLTEModel(
        collision_data=hydrogen_collisions,
        maximum_helium_ii_level=14,
        helium_i_stark_table=helium_i,
        helium_ii_stark_table=helium_ii,
        population_explicit_maximum_lower_level=4,
        population_n_angle=2,
        population_maximum_iterations=80,
        population_relative_tolerance=1.0e-2,
        helium_i_collision_data=helium_i_collisions,
    )
    atomic_database = read_pg1159_atomic_database(
        data.cache / "metal-opacity/atomic-line-list/stout"
    )
    photoionization = read_verner_photoionization_database(
        data.cache / "metal-opacity/verner-photoionization.dat",
        elements=("C", "O"),
        require_all_elements=True,
    )
    carbon_thresholds = {
        2: read_tlusty_photoionization_threshold_data(
            data.cache / "tlusty-atoms/c3.dat"
        ),
        3: read_tlusty_photoionization_threshold_data(
            data.cache / "tlusty-atoms/c4_35+2lev.dat"
        ),
    }
    oxygen_thresholds = {
        3: read_tlusty_photoionization_threshold_data(
            data.cache / "tlusty-atoms/o4.dat"
        ),
        4: read_tlusty_photoionization_threshold_data(
            data.cache / "tlusty-atoms/o5.dat"
        ),
        5: read_tlusty_photoionization_threshold_data(
            data.cache / "tlusty-atoms/o6.dat"
        ),
    }
    return (
        helium_model,
        atomic_database,
        photoionization,
        carbon_thresholds,
        oxygen_thresholds,
    )


def _model(
    helium_model,
    atomic_database,
    photoionization,
    carbon_thresholds,
    oxygen_thresholds,
    carbon_atom,
    oxygen_atom,
    composition,
    *,
    solve_oxygen: bool,
    iterations: int,
    damping: float,
    acceleration_depth: int,
    acceleration_damping: float,
    ion_relative_floor: float,
    level_relative_floor: float,
) -> PG1159NLTEModel:
    return PG1159NLTEModel(
        helium_model=helium_model,
        atomic_database=atomic_database,
        mass_fractions=composition,
        photoionization_database=photoionization,
        carbon_photoionization_threshold_data=carbon_thresholds,
        oxygen_photoionization_threshold_data=oxygen_thresholds,
        nlte_metal_elements=("C", "O") if solve_oxygen else ("C",),
        solve_oxygen_levels_nlte=solve_oxygen,
        carbon_levels_per_charge=carbon_atom.levels_per_charge,
        oxygen_levels_per_charge=(
            oxygen_atom.levels_per_charge if solve_oxygen else {}
        ),
        carbon_population_atomic_database=carbon_atom.atomic_database,
        carbon_formal_level_mapping=carbon_atom.formal_level_mapping,
        carbon_formal_lte_parent_mapping=(carbon_atom.formal_lte_parent_mapping),
        carbon_continuum_parent_mapping=carbon_atom.continuum_parent_mapping,
        carbon_lte_level_reservoir=carbon_atom.lte_level_reservoir,
        carbon_lte_bound_bound_couplings=(carbon_atom.lte_bound_bound_couplings),
        carbon_effective_dielectronic_couplings=(
            carbon_atom.effective_dielectronic_couplings
        ),
        carbon_collision_data=carbon_atom.collision_data,
        oxygen_population_atomic_database=(
            oxygen_atom.atomic_database if solve_oxygen else None
        ),
        oxygen_formal_level_mapping=(
            oxygen_atom.formal_level_mapping if solve_oxygen else None
        ),
        oxygen_formal_lte_parent_mapping=(
            oxygen_atom.formal_lte_parent_mapping if solve_oxygen else None
        ),
        oxygen_continuum_parent_mapping=(
            oxygen_atom.continuum_parent_mapping if solve_oxygen else None
        ),
        oxygen_lte_level_reservoir=(
            oxygen_atom.lte_level_reservoir if solve_oxygen else None
        ),
        oxygen_lte_bound_bound_couplings=(
            oxygen_atom.lte_bound_bound_couplings if solve_oxygen else None
        ),
        oxygen_effective_dielectronic_couplings=(
            oxygen_atom.effective_dielectronic_couplings if solve_oxygen else None
        ),
        oxygen_collision_data=(oxygen_atom.collision_data if solve_oxygen else None),
        ionization_scattering_iterations=3,
        metal_population_iterations=iterations,
        metal_population_minimum_iterations=min(3, iterations),
        metal_population_relative_tolerance=3.0e-3,
        metal_population_damping=damping,
        metal_population_acceleration_depth=acceleration_depth,
        metal_population_acceleration_damping=acceleration_damping,
        metal_population_relative_floor=level_relative_floor,
        metal_ion_population_relative_floor=ion_relative_floor,
        metal_level_population_relative_floor=level_relative_floor,
        helium_population_damping=0.7,
        helium_population_relative_floor=1.0e-8,
        nlte_charge_feedback=True,
    )


def _model_atom(element: str, path: Path, atomic_database, target: dict[int, int]):
    atom = read_tmad_structure_model_atom(
        path, atomic_database, target_nlte_levels_per_charge=target
    )
    if dict(atom.levels_per_charge) != target:
        raise RuntimeError(
            f"{element} atom selected {dict(atom.levels_per_charge)}, expected {target}"
        )
    return atom


def build_model(
    data,
    composition: Mapping[str, float],
    *,
    structure_only: bool,
    population_iterations: int,
    oxygen_atom_preset: str = "compact14",
):
    """Instantiate the accepted atom without target-specific profile choices."""

    CARBON_ATOM = data.cache / "tmad-atoms/C_III-V"
    OXYGEN_ATOM = data.cache / "tmad-atoms/O_III-VII"
    TMAD_FORMAL_ATOMS = {
        ("O", q): data.cache / f"tmad-atoms/O_{roman}_syn"
        for q, roman in ((2, "III"), (3, "IV"), (4, "V"), (5, "VI"), (6, "VII"))
    }
    TMAD_CIV_PROFILE_ATOMS = {("C", 3): data.cache / "tmad-atoms/C_IV_syn"}
    EXTENDED_OVI_SIROCCO_LEVELS = data.cache / "sirocco-atomic/o_6_levels.dat"
    EXTENDED_OVI_SIROCCO_PHOTOIONIZATION = data.cache / "sirocco-atomic/o_6_phot.dat"
    EXTENDED_OVI_CHIANTI_SCUPS = data.cache / "chianti/o_6/o_6.scups"
    helium, _, _, carbon_thresholds, oxygen_thresholds = _build_model_inputs(data)
    elements = tuple(element for element in composition if element != "He")
    database = read_pg1159_atomic_database(
        data.cache / "metal-opacity/atomic-line-list/stout",
        elements=("C", "O") if structure_only else elements,
    )
    database = atomic_database_with_tmad_formal_ions(database, TMAD_FORMAL_ATOMS)
    database = atomic_database_with_tmad_profile_parameters(
        database,
        TMAD_CIV_PROFILE_ATOMS,
        series_lower_principal_quantum_number=4,
        minimum_upper_principal_quantum_number=8,
    )
    photoionization = read_verner_photoionization_database(
        data.cache / "metal-opacity/verner-photoionization.dat",
        elements=("C", "O") if structure_only else elements,
    )
    counts = mutable_atom_counts(PG1424_BEST_LINE_ATOM_COUNTS)
    carbon = _model_atom("C", CARBON_ATOM, database, counts["C"])
    oxygen = _model_atom("O", OXYGEN_ATOM, database, counts["O"])
    model = _model(
        helium,
        database,
        photoionization,
        carbon_thresholds,
        oxygen_thresholds,
        carbon,
        oxygen,
        composition,
        solve_oxygen=True,
        iterations=population_iterations,
        damping=0.25,
        acceleration_depth=4,
        acceleration_damping=0.20,
        ion_relative_floor=1.0e-6,
        level_relative_floor=1.0e-5,
    )
    trace = tuple(
        element
        for element in SUPPORTED_NLTE_TRACE_ELEMENTS
        if element in composition and element in elements
    )
    model = replace(
        model,
        nlte_metal_elements=(("C", "O") if structure_only else ("C", "O", *trace)),
        trace_opacity_in_population_radiation=False,
        trace_photoionization_threshold_data=MappingProxyType({}),
        ion_stage_range_overrides=MappingProxyType({}),
        static_linear_stark_frequency_scales=(
            PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES
        ),
        hydrogenic_linear_stark_components=PG1159_HYDROGENIC_STARK_COMPONENTS,
        metal_rate_line_velocity_samples_kms=(
            PG1424_STRUCTURE_LINE_VELOCITY_SAMPLES_KMS if structure_only else None
        ),
    )
    if oxygen_atom_preset == "extended54-complete":
        model = pg1159_model_with_extended_oxygen_vi(
            model,
            OXYGEN_ATOM,
            sirocco_level_data=EXTENDED_OVI_SIROCCO_LEVELS,
            sirocco_photoionization_data=(EXTENDED_OVI_SIROCCO_PHOTOIONIZATION),
            oxygen_vi_chianti_scups=EXTENDED_OVI_CHIANTI_SCUPS,
            resolved_bound_free=True,
            include_angular_momentum_mixing=True,
            include_quadrupole_angular_momentum_mixing=True,
        )
        model = replace(model, include_semiclassical_ovi_stark_widths=True)
    elif oxygen_atom_preset != "compact14":
        raise ValueError(f"unknown oxygen atom preset {oxygen_atom_preset!r}")
    return model, database


def default_helium_nlte_structure_wavelength(
    *, n_continuum_wavelength: int = 360
) -> np.ndarray:
    """Return a structural grid resolving He I/II edges and strong lines."""

    if n_continuum_wavelength < 80:
        raise ValueError("n_continuum_wavelength must be at least 80")
    pieces: list[np.ndarray] = [
        np.geomspace(25.0, 100_000.0, n_continuum_wavelength),
        np.arange(200.0, 1000.1, 2.0),
    ]
    for line in HELIUM_I_RESONANCE_LINES:
        center = line.wavelength_vacuum_angstrom
        pieces.extend(
            (center + np.linspace(-20.0, 20.0, 61), center + np.linspace(-2.0, 2.0, 41))
        )
    for line in HELIUM_I_LINES:
        center = line.wavelength_vacuum_angstrom
        pieces.append(center + np.linspace(-80.0, 80.0, 41))
    for line in HELIUM_II_LINES:
        center = line.wavelength_vacuum_angstrom
        pieces.extend(
            (center + np.linspace(-80.0, 80.0, 41), center + np.linspace(-5.0, 5.0, 31))
        )
    wavelength = np.unique(np.concatenate(pieces))
    return np.ascontiguousarray(wavelength[wavelength > 0.0])


def structure_wavelength(model, continuum_points: int) -> np.ndarray:
    """Rebuild the He/C/O structural transfer grid from the production model."""

    pieces = [
        default_helium_nlte_structure_wavelength(
            n_continuum_wavelength=continuum_points
        )
    ]
    for element in ("C", "O"):
        if element == "C":
            database = model.carbon_population_atomic_database
            levels = model.carbon_levels_per_charge
            thresholds = model.carbon_photoionization_threshold_data
            formal_mapping = model.carbon_formal_level_mapping
            couplings = model.carbon_lte_bound_bound_couplings
        else:
            database = model.oxygen_population_atomic_database
            levels = model.oxygen_levels_per_charge
            thresholds = model.oxygen_photoionization_threshold_data
            formal_mapping = model.oxygen_formal_level_mapping
            couplings = model.oxygen_lte_bound_bound_couplings
        if database is None or model.photoionization_database is None:
            raise RuntimeError(f"production {element} atom is incomplete")
        pieces.append(
            reduced_light_metal_wavelength(
                database,
                element,
                levels,
                n_continuum_wavelength=continuum_points,
                photoionization_threshold_data=thresholds,
                formal_atomic_database=model.atomic_database,
                formal_level_mapping=formal_mapping,
                lte_bound_bound_couplings=couplings,
                effective_dielectronic_couplings=(
                    model.carbon_effective_dielectronic_couplings
                    if element == "C"
                    else model.oxygen_effective_dielectronic_couplings
                ),
                line_velocity_samples_kms=(model.metal_rate_line_velocity_samples_kms),
            )
        )
    return np.unique(np.concatenate(pieces))
