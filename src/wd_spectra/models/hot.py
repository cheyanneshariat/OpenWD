"""Experimental DO/DAO one-shot models with restricted H/He NLTE."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import hashlib

from ..atmosphere import helium_continuum_atmosphere, hydrogen_helium_continuum_atmosphere
from ..helium import HELIUM_I_LINES, HELIUM_I_RESONANCE_LINES, HELIUM_II_LINES
from ..helium_stark import read_helium_stark_table
from ..helium_nlte import coupled_helium_ii_lines
from ..helium_ii_stark import read_helium_ii_stark_table
from ..helium_collisions import read_tlusty_helium_collision_data
from ..multilevel_nlte import read_ccc_hydrogen_collision_data
from ..hot_nlte import HotNLTEModel, solve_hot_nlte_atmosphere, transfer_field
from ..opacity import LYMAN_LINES, BALMER_LINES, PASCHEN_LINES, BRACKETT_LINES
from ..spectrum import Spectrum
from .common import (ModelData, ModelResult, Quality, numerical_resolution, validate_wavelength,
    model_request_fingerprint, atmosphere_with_model_request_fingerprint,
    warn_if_atmosphere_not_converged)


@dataclass(frozen=True)
class DOConfig:
    effective_temperature: float = 70000.
    logg: float = 8.
    quality: Quality = 'standard'
    maximum_helium_ii_level: int = 32
    population_maximum_iterations: int = 120
    population_tolerance: float = 1e-4
    hydrogenic_collision_model: str = "tlusty-mihalas"
    helium_i_profile: str = "Tremblay26.txt"
    helium_ii_interpolation: str = "series-adaptive"


@dataclass(frozen=True)
class DAOConfig:
    effective_temperature: float = 70000.
    logg: float = 8.
    log_hydrogen_to_helium: float = 2.
    quality: Quality = 'standard'
    maximum_helium_ii_level: int = 32
    maximum_hydrogen_level: int = 8
    population_maximum_iterations: int = 120
    population_tolerance: float = 1e-4
    hydrogenic_collision_model: str = "tlusty-mihalas"
    helium_i_profile: str = "Tremblay26.txt"
    helium_ii_interpolation: str = "series-adaptive"


def validate_config(config):
    if not isinstance(config, (DOConfig, DAOConfig)):
        raise TypeError('expected DOConfig or DAOConfig')
    if not np.isfinite(config.effective_temperature) or config.effective_temperature <= 0 or not np.isfinite(config.logg):
        raise ValueError('Teff must be finite and positive; logg must be finite')
    numerical_resolution(config.quality)
    if config.hydrogenic_collision_model not in ("ccc-scaled", "tlusty-mihalas"):
        raise ValueError("unknown He II collision prescription")
    if config.helium_i_profile not in ("Tremblay26.txt", "Beauchamp25_LD.txt", "Beauchamp25_NLD.txt"):
        raise ValueError("unknown He I line-profile table")
    if config.helium_ii_interpolation not in ("series-adaptive", "synspec-quadratic", "log-bilinear"):
        raise ValueError("unknown He II profile interpolation")
    for value, lower, upper, label in (
        (config.maximum_helium_ii_level, 2, 32, 'maximum_helium_ii_level'),
        (getattr(config, 'maximum_hydrogen_level', 8), 2, 20, 'maximum_hydrogen_level'),
        (config.population_maximum_iterations, 1, None, 'population_maximum_iterations')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower or (upper is not None and value > upper):
            raise ValueError(f'invalid {label}')
    if not np.isfinite(config.population_tolerance) or not 0 < config.population_tolerance <= 1e-3:
        raise ValueError('population_tolerance must lie in (0, 1e-3]')
    if isinstance(config, DAOConfig) and not np.isfinite(config.log_hydrogen_to_helium):
        raise ValueError('log_hydrogen_to_helium must be finite')


def structure_wavelength(config, n_continuum):
    pieces = [np.geomspace(25., 100000., n_continuum), np.arange(200., 1000.1, 2.)]
    lines = (*HELIUM_I_LINES, *HELIUM_I_RESONANCE_LINES, *coupled_helium_ii_lines(config.maximum_helium_ii_level))
    if isinstance(config, DAOConfig):
        lines += (*LYMAN_LINES, *BALMER_LINES, *PASCHEN_LINES, *BRACKETT_LINES)
    for line in lines:
        center = line.wavelength_vacuum_angstrom
        pieces.extend((center+np.linspace(-80., 80., 41), center+np.linspace(-5., 5., 31)))
    wave = np.unique(np.concatenate(pieces))
    return wave[wave > 0]


def _model_from_config(config, data):
    """Rebuild the declared H/He atom for matching fixed-state line formation."""
    validate_config(config)
    resolution = numerical_resolution(config.quality)
    mixed = isinstance(config, DAOConfig)
    helium_i_path = data.cache / "helium-stark" / config.helium_i_profile
    return HotNLTEModel(
        read_ccc_hydrogen_collision_data(data.ccc_hydrogen_collisions, maximum_level=8),
        read_helium_stark_table(helium_i_path),
        read_helium_ii_stark_table(data.helium_ii_stark, thermodynamic_interpolation=config.helium_ii_interpolation),
        read_tlusty_helium_collision_data(data.tlusty_source, data.tlusty_helium_atom),
        maximum_helium_ii_level=config.maximum_helium_ii_level,
        maximum_hydrogen_level=getattr(config, 'maximum_hydrogen_level', 8),
        log_hydrogen_to_helium=config.log_hydrogen_to_helium if mixed else None,
        n_angle=resolution.n_angle,
        population_maximum_iterations=config.population_maximum_iterations,
        population_tolerance=config.population_tolerance,
        hydrogenic_collision_model=config.hydrogenic_collision_model)


def _compute(config, wavelength=None, *, data=None, initial_atmosphere=None, iteration_callback=None):
    validate_config(config)
    # Cold public runs are reproducible; restarts need population/EOS provenance
    # and remain a low-level diagnostic operation in hot_nlte.
    if initial_atmosphere is not None:
        raise ValueError('DO/DAO public models require a cold start')
    wave = validate_wavelength(wavelength)
    data = ModelData.default() if data is None else data
    helium_i_path = data.cache / "helium-stark" / config.helium_i_profile
    data.require(
        data.ccc_hydrogen_collisions,
        data.tlusty_source,
        data.tlusty_helium_atom,
        helium_i_path,
        data.helium_ii_stark,
    )
    identity = {"hot_nlte_revision": "restricted-h-he-joint-newton-v7-emissivity"}
    for name in ("ccc_hydrogen_collisions", "tlusty_source", "tlusty_helium_atom",
                 "helium_i_stark", "helium_ii_stark"):
        digest = hashlib.sha256()
        path = helium_i_path if name == "helium_i_stark" else getattr(data, name)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        identity[name] = digest.hexdigest()
    resolution = numerical_resolution(config.quality)
    mixed = isinstance(config, DAOConfig)
    model = _model_from_config(config, data)
    seed_function = hydrogen_helium_continuum_atmosphere if mixed else helium_continuum_atmosphere
    args = (config.effective_temperature, config.logg)
    if mixed:
        args += (config.log_hydrogen_to_helium,)
    seed = seed_function(*args, n_depth=resolution.n_depth,
                         rosseland_frequency_points=resolution.n_continuum,
                         tau_max=1000. if mixed else 100.)
    if config.quality != 'quick':
        from ..atmosphere import radiative_equilibrium_helium_atmosphere
        seed = radiative_equilibrium_helium_atmosphere(
            config.effective_temperature,config.logg,n_depth=resolution.n_depth,
            n_continuum_wavelength=resolution.n_continuum,max_iterations=60,
            n_angle=resolution.n_angle,structure_solver='adaptive-newton',
            mixing_length_alpha=None,stark_table=model.helium_i_stark_table,
            log_hydrogen_abundance=model.log_hydrogen_to_helium,
            helium_ii_stark_table=model.helium_ii_stark_table,include_helium_ii_lines=True,
            initial_temperature=seed.temperature,initial_column_mass=seed.column_mass,
            initial_gas_pressure=seed.gas_pressure,
            initial_rosseland_optical_depth=seed.rosseland_optical_depth,
            iteration_callback=(None if iteration_callback is None else
                lambda i,a,d:iteration_callback(i,a,{**d,'solver_phase':'lte-initialization'})))
    result = solve_hot_nlte_atmosphere(
        seed, model, structure_wavelength(config, resolution.n_continuum),
        maximum_iterations=resolution.maximum_iterations, iteration_callback=iteration_callback)
    kind = 'DAO' if mixed else 'DO'
    fingerprint = model_request_fingerprint(kind, config, data, physical_data_identity=identity)
    atmosphere = atmosphere_with_model_request_fingerprint(result.atmosphere, fingerprint)
    status = warn_if_atmosphere_not_converged(atmosphere, kind)
    coefficients = model.transfer_coefficients(atmosphere, wave, result.population_state)
    _, field, closure = transfer_field(atmosphere, coefficients, n_angle=resolution.n_angle)
    collision_metadata = {
        'ccc_maximum_level': model.collision_data.maximum_level,
        'hydrogen_collision_closure': None if not mixed else
            f'CCC through n={model.collision_data.maximum_level}; TLUSTY/Mihalas above CCC',
        'helium_ii_collision_model': model.hydrogenic_collision_model,
    }
    spectrum = Spectrum(wave, field.interface_flux[:, 0], {
        'flux_unit': 'erg s^-1 cm^-2 Angstrom^-1',
        'flux_convention': 'surface F_lambda', 'wavelength_medium': 'vacuum',
        'composition': kind, 'line_formation': 'restricted H/He NLTE',
        'transfer': 'column-mass Feautrier', 'source_closure_residual': closure,
        'nlte_charge_feedback': False})
    return ModelResult(kind, atmosphere, spectrum, config, {
        **collision_metadata,
        'preset': f'{kind}-restricted-NLTE-v1', 'experimental': True,
        'cold_start': True,
        'initialization': {
            'method': 'fresh continuum' if config.quality == 'quick' else 'fresh LTE equilibrium',
            'previous_model_supplied': False,
            'stellar_parameters_fixed': True,
        },
        'atmosphere_convergence_status': status, 'model_request_fingerprint': fingerprint,
        'nlte_charge_feedback': False, 'metals': False,
        'independent_grid_validation': False, 'full_physics_validation': False,
        'nlte_populations_converged': result.population_state.converged,
    }, population_state=result.population_state)


def compute_do(config=DOConfig(), wavelength=None, **kwargs):
    """Construct a fresh, experimental pure-helium NLTE atmosphere and spectrum."""
    if not isinstance(config, DOConfig):
        raise TypeError('compute_do requires DOConfig')
    return _compute(config, wavelength, **kwargs)


def compute_dao(config=DAOConfig(), wavelength=None, **kwargs):
    """Construct a fresh, experimental mixed H/He NLTE atmosphere and spectrum."""
    if not isinstance(config, DAOConfig):
        raise TypeError('compute_dao requires DAOConfig')
    return _compute(config, wavelength, **kwargs)
