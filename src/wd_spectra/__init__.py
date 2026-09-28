"""OpenWD: open white-dwarf atmosphere and spectrum calculations."""

from .models import (
    AtmosphereConvergenceWarning,
    DAConfig,
    DAZConfig,
    DQConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    ModelData,
    ModelResult,
    ModelRun,
    PhysicsSelection,
    PhysicsSelectionPolicy,
    select_physics,
    run_model,
    atmosphere_convergence_status,
    compute_da,
    compute_daz,
    compute_dq,
    compute_dab,
    compute_db,
    compute_dz,
    load_atmosphere_checkpoint,
    save_model_result,
)
from .atmosphere import (
    Atmosphere,
    gray_helium_atmosphere,
    gray_hydrogen_atmosphere,
    gray_hydrogen_helium_atmosphere,
    helium_continuum_atmosphere,
    hydrogen_continuum_atmosphere,
    hydrogen_helium_continuum_atmosphere,
    radiative_equilibrium_helium_atmosphere,
    radiative_equilibrium_hydrogen_atmosphere,
    radiative_equilibrium_hydrogen_helium_atmosphere,
)
from .cool_da import (
    TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K,
    atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    tremblay_2013_mean_3d_temperature_difference,
)
from .gaunt import hydrogen_bound_free_gaunt_factor, hydrogen_free_free_gaunt_factor
from .helium import (
    HELIUM_I_LINES,
    HELIUM_II_LINES,
    helium_continuum_mass_absorption_coefficient,
    helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_ii_bound_free_linear_absorption_coefficient,
    helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
    helium_ii_rayleigh_scattering_cross_section,
    helium_i_line_mass_absorption_coefficient,
    helium_i_resonance_hwhm_angstrom,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_i_resonance_stark_hwhm_angstrom,
    helium_i_rydberg_bound_free_linear_absorption_coefficient,
    helium_minus_free_free_coefficient,
    helium_rayleigh_scattering_cross_section,
    helium_rayleigh_scattering_mass_coefficient,
    neutral_helium_photoionization_cross_section,
)
from .helium_ii_stark import read_helium_ii_stark_table
from .helium_molecular import (
    helium_dimer_ion_continuum_coefficient,
    helium_three_body_cia_linear_absorption_coefficient,
)
from .helium_stark import read_helium_stark_table
from .helium_neutral import deridder_van_rensbergen_helium_hwhm_angstrom
from .hydrogen_self import (
    BarklemSelfBroadeningTable,
    read_barklem_self_broadening_table,
)
from .mixture import hydrogen_helium_continuum_mass_absorption_coefficient
from .opacity import hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient
from .radiative_transfer import compiled_backend_available, emergent_flux
from .spectrum import (
    Spectrum,
    planck_lambda_angstrom,
    synthesize_balmer_spectrum,
    synthesize_gray_spectrum,
    synthesize_helium_spectrum,
    synthesize_hydrogen_helium_spectrum,
    synthesize_hydrogen_spectrum,
)
from .stark import (
    default_balmer_stark_table,
    default_brackett_stark_table,
    default_lyman_stark_table,
    default_paschen_stark_table,
)

__version__ = "0.1.4"

from .models.hot import DOConfig, DAOConfig, compute_do, compute_dao

__all__ = [
    "DOConfig", "DAOConfig", "compute_do", "compute_dao",
    "Atmosphere",
    "AtmosphereConvergenceWarning",
    "BarklemSelfBroadeningTable",
    "DAConfig",
    "DAZConfig",
    "DQConfig",
    "DABConfig",
    "DBConfig",
    "DZConfig",
    "HELIUM_I_LINES",
    "HELIUM_II_LINES",
    "ModelData",
    "ModelResult",
    "ModelRun",
    "PhysicsSelection",
    "PhysicsSelectionPolicy",
    "select_physics",
    "run_model",
    "Spectrum",
    "atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "atmosphere_convergence_status",
    "compiled_backend_available",
    "compute_da",
    "compute_daz",
    "compute_dq",
    "compute_dab",
    "compute_db",
    "compute_dz",
    "convection",
    "default_balmer_stark_table",
    "default_brackett_stark_table",
    "default_lyman_stark_table",
    "default_paschen_stark_table",
    "deridder_van_rensbergen_helium_hwhm_angstrom",
    "emergent_flux",
    "gray_helium_atmosphere",
    "gray_hydrogen_atmosphere",
    "gray_hydrogen_helium_atmosphere",
    "helium_continuum_atmosphere",
    "hydrogen_bound_free_gaunt_factor",
    "hydrogen_continuum_atmosphere",
    "hydrogen_free_free_gaunt_factor",
    "hydrogen_helium_continuum_atmosphere",
    "hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "load_atmosphere_checkpoint",
    "planck_lambda_angstrom",
    "radiative_equilibrium_helium_atmosphere",
    "radiative_equilibrium_hydrogen_atmosphere",
    "radiative_equilibrium_hydrogen_helium_atmosphere",
    "read_barklem_self_broadening_table",
    "save_model_result",
    "synthesize_balmer_spectrum",
    "synthesize_gray_spectrum",
    "synthesize_helium_spectrum",
    "synthesize_hydrogen_helium_spectrum",
    "synthesize_hydrogen_spectrum",
    "tremblay_2013_mean_3d_temperature_difference",
]

from .models.pg1159 import PG1159Config, compute_pg1159
__all__ += ["PG1159Config", "compute_pg1159"]
from .models.d6 import D6Config, compute_d6
__all__ += ["D6Config", "compute_d6"]

from .light_metal_nlte import (ChiantiScaledCollisionComponent, ChiantiTermCollisionStrength, TmadEffectiveDielectronicCoupling, TmadLTEBoundBoundCoupling, TmadStructureModelAtom, atomic_database_with_chianti_radiative_transitions, barklem_oi_electron_collision_data, fine_structure_collision_data_from_tmad, atomic_database_with_tmad_formal_ions, atomic_database_with_tmad_profile_parameters, light_metal_free_free_charge_kernel, light_metal_free_free_mass_absorption_coefficient, reduced_light_metal_wavelength, read_barklem_mgi_electron_collision_data, read_tmad_structure_model_atom, read_chianti_term_collision_strengths, read_norad_oxygen_vi_photoionization_data, read_sirocco_topbase_photoionization_data, read_tlusty_forbidden_collision_strengths, read_tlusty_photoionization_threshold_data, rydberg_angular_momentum_mixing_collision_data, rydberg_quadrupole_angular_momentum_mixing_collision_data, btm_quadrupole_l_mixing_rate_coefficient, psm20_debye_l_mixing_rate_coefficient, psm20_l_mixing_rate_coefficient, solve_reduced_light_metal_levels_nlte, solve_light_metal_ionization_nlte)
__all__ += ['ChiantiScaledCollisionComponent', 'ChiantiTermCollisionStrength', 'TmadEffectiveDielectronicCoupling', 'TmadLTEBoundBoundCoupling', 'TmadStructureModelAtom', 'atomic_database_with_chianti_radiative_transitions', 'barklem_oi_electron_collision_data', 'fine_structure_collision_data_from_tmad', 'atomic_database_with_tmad_formal_ions', 'atomic_database_with_tmad_profile_parameters', 'light_metal_free_free_charge_kernel', 'light_metal_free_free_mass_absorption_coefficient', 'reduced_light_metal_wavelength', 'read_barklem_mgi_electron_collision_data', 'read_tmad_structure_model_atom', 'read_chianti_term_collision_strengths', 'read_norad_oxygen_vi_photoionization_data', 'read_sirocco_topbase_photoionization_data', 'read_tlusty_forbidden_collision_strengths', 'read_tlusty_photoionization_threshold_data', 'rydberg_angular_momentum_mixing_collision_data', 'rydberg_quadrupole_angular_momentum_mixing_collision_data', 'btm_quadrupole_l_mixing_rate_coefficient', 'psm20_debye_l_mixing_rate_coefficient', 'psm20_l_mixing_rate_coefficient', 'solve_reduced_light_metal_levels_nlte', 'solve_light_metal_ionization_nlte']

from .pg1159_presets import (PG1424_BEST_LINE_ATOM_COUNTS, PG1424_COMPOSITION_LADDER_ATOM_COUNTS, PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS, PG1424_PUBLISHED_MASS_FRACTIONS, PG1424_RELAXED_STRUCTURE_ATOM_COUNTS, WERNER2015_LINE_ATOM_COUNTS, mutable_atom_counts)
__all__ += ['PG1424_BEST_LINE_ATOM_COUNTS', 'PG1424_COMPOSITION_LADDER_ATOM_COUNTS', 'PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS', 'PG1424_PUBLISHED_MASS_FRACTIONS', 'PG1424_RELAXED_STRUCTURE_ATOM_COUNTS', 'WERNER2015_LINE_ATOM_COUNTS', 'mutable_atom_counts']

from .models.dah import DAHConfig, compute_dah
__all__ += ["DAHConfig", "compute_dah"]
