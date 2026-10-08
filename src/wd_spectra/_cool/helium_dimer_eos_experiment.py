"""Research-only He2+ charge/pressure closure; no production dispatch changes.

HM atomic partitions are retained. He2+ is added with a smooth interpolation
of the existing Stancil equilibrium constants, not an electron floor. This
is not the nonideal chemical-potential model of Kowalski et al. (2007), and
is not a validated dense-helium EOS.
"""
from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from pathlib import Path
import json

import numpy as np
from scipy.interpolate import CubicSpline
import wd_spectra.eos as eos
from wd_spectra import helium_molecular as molecular
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS


# Interpolate the internal-equilibrium factor, separating the known T^1.5
# translational factor. Natural end conditions permit C2 continuation by a
# tangent in reciprocal temperature, preserving positivity without clamping K.
_X = (1 / molecular._TEMPERATURE)[::-1]
_Y = (np.log(molecular._EQUILIBRIUM_CONSTANT)
      - 1.5*np.log(molecular._TEMPERATURE))[::-1]
_CURVE = CubicSpline(_X, _Y, bc_type="natural", extrapolate=False)


def dissociation_equilibrium(temperature):
    """Return K=n(He I)n(He II)/n(He2+) and dlnK/dlnT.

    The 4200--50400 K nodes reproduce the opacity table exactly; evaluation
    outside that range is an explicit extrapolation, not new molecular data.
    The derivative determines reaction energy from the same equilibrium law.
    """
    temperature = np.asarray(temperature, dtype=float)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    inverse = 1/temperature
    inside = np.clip(inverse, _X[0], _X[-1])
    slope = _CURVE(inside, 1)
    internal = _CURVE(inside) + (inverse-inside)*slope
    logarithm = 1.5*np.log(temperature)+internal
    if np.any(abs(logarithm) > 700):
        raise ValueError("experimental molecular equilibrium left its numerical domain")
    return np.exp(logarithm), 1.5-inverse*slope


@dataclass(frozen=True)
class DimerClosure:
    """The additional species remains explicit, separate from atomic state APIs."""
    atomic: eos.HeliumLTEState
    molecular_ion_density: np.ndarray
    specific_enthalpy: np.ndarray


def helium_dimer_lte(temperature, gas_pressure, *, maximum_level=eos.HM_MAX_BOUND_LEVEL,
                     neutral_radius_scale=eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE,
                     correlated_microfields=True, include_dimer=True):
    """Simultaneous He I/II/III, He2+, electron equilibrium at fixed P,T.

    Particle pressure includes one molecular ion, nuclear conservation two
    He nuclei, and charge neutrality one positive charge per He2+. Turning
    off the dimer is a diagnostic atomic-limit comparison, not a fallback.
    """
    temperature, pressure = np.broadcast_arrays(np.asarray(temperature, dtype=float),
                                              np.asarray(gas_pressure, dtype=float))
    if (np.any(~np.isfinite(pressure)) or np.any(pressure <= 0)
            or np.any(~np.isfinite(temperature)) or np.any(temperature <= 0)):
        raise ValueError("temperature and gas pressure must be finite and positive")
    equilibrium, dlogk = dissociation_equilibrium(temperature)
    particle = pressure/(BOLTZMANN*temperature)
    translational = 1.5*np.log(2*np.pi*eos.ELECTRON_MASS*BOLTZMANN*temperature/eos.PLANCK**2)
    lower = np.log(particle)-90
    upper = np.log(2*particle/3)

    def at_electron(ne):
        material = particle-ne
        neutral = material.copy()
        for _ in range(3):
            q0, q1, occupations, _, _ = eos._helium_partition_functions(
                temperature, ne, neutral, maximum_level=maximum_level,
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields)
            log_r1 = (np.log(2.)+translational+np.log(q1/q0)
                      -eos.HELIUM_FIRST_IONIZATION_ENERGY/(BOLTZMANN*temperature)-np.log(ne))
            log_r2 = (np.log(2.)+translational-np.log(q1)
                      -eos.HELIUM_SECOND_IONIZATION_ENERGY/(BOLTZMANN*temperature)-np.log(ne))
            terms = np.stack((np.zeros_like(ne), log_r1, log_r1+log_r2), axis=-1)
            weights = np.exp(np.clip(terms-np.max(terms, axis=-1, keepdims=True), -745., 0.))
            fractions = weights/np.sum(weights, axis=-1, keepdims=True)
            coefficient = (fractions[..., 0]*fractions[..., 1]/equilibrium
                           if include_dimer else np.zeros_like(ne))
            # Solve for total ATOMIC particles, not n(He I). The normalized
            # fractions avoid overflowing Saha ratios when He III dominates.
            atoms = 2*material/(1+np.hypot(1., 2*np.sqrt(coefficient)*np.sqrt(material)))
            neutral = atoms*fractions[..., 0]
        ion = atoms*fractions[..., 1]
        double = atoms*fractions[..., 2]
        dimer = coefficient*atoms**2
        charge = ion+2*double+dimer
        return ne-charge, neutral, ion, double, dimer, q0, q1, occupations

    for _ in range(56):
        middle = .5*(lower+upper)
        residual, *_ = at_electron(np.exp(middle))
        positive = residual > 0
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
    ne = np.exp(.5*(lower+upper))
    residual, neutral, ion, double, dimer, q0, q1, occupations = at_electron(ne)
    nuclei = neutral+ion+double+2*dimer
    if np.any(abs(residual) > 1e-10*ne):
        raise ValueError("molecular charge solve did not converge")
    low_terms = (eos.HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT*occupations
                 *np.exp(-eos.HELIUM_I_LOW_TERM_ENERGY/(BOLTZMANN*temperature[..., None])))
    state = eos.HeliumLTEState(
        mass_density=HELIUM_MASS*nuclei, helium_nuclei_density=nuclei,
        neutral_he_density=neutral, singly_ionized_he_density=ion,
        doubly_ionized_he_density=double, electron_density=ne,
        mean_ion_charge=ne/nuclei, neutral_partition_function=q0,
        singly_ionized_partition_function=q1,
        neutral_level_occupation_probability=occupations,
        neutral_level_population_density=neutral[..., None]*low_terms/q0[..., None],
        microfield_model="qmhd" if correlated_microfields else "holtsmark",
        neutral_radius_scale=neutral_radius_scale)
    _, _, _, e0, e1 = eos._helium_partition_functions(
        temperature, ne, neutral, maximum_level=maximum_level,
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields)
    # kT*dlnK/dlnT is the dissociation internal-energy difference, including
    # the extra translational particle. Derive the molecular internal energy
    # from this SAME K, rather than inserting an unrelated binding constant.
    molecular_internal = (eos.HELIUM_FIRST_IONIZATION_ENERGY+e0+e1
                          + BOLTZMANN*temperature*(1.5-dlogk))
    internal = (1.5*pressure + neutral*e0
        + ion*(eos.HELIUM_FIRST_IONIZATION_ENERGY+e1)
        + double*(eos.HELIUM_FIRST_IONIZATION_ENERGY+eos.HELIUM_SECOND_IONIZATION_ENERGY)
        + dimer*molecular_internal)
    enthalpy = (internal+pressure)/state.mass_density
    return DimerClosure(state, dimer, enthalpy)


def helium_dimer_thermodynamics(temperature, pressure, **options):
    """Differentiate enthalpy and density from the complete experimental closure."""
    t = np.asarray(temperature, dtype=float)
    eps = 2e-4
    cold = helium_dimer_lte(t*np.exp(-eps), pressure, **options)
    hot = helium_dimer_lte(t*np.exp(eps), pressure, **options)
    central = helium_dimer_lte(t, pressure, **options)
    cp = (hot.specific_enthalpy-cold.specific_enthalpy)/(2*t*np.sinh(eps))
    expansion = -np.log(hot.atomic.mass_density/cold.atomic.mass_density)/(2*eps)
    adiabatic = pressure*expansion/(central.atomic.mass_density*t*cp)
    if np.any(cp <= 0) or np.any(~np.isfinite(cp)) or np.any(expansion <= 0):
        raise ValueError("experimental EOS produced invalid thermal derivatives")
    return eos.HeliumThermodynamics(cp, expansion, adiabatic)


def helium_dimer_continuum(wavelength, temperature):
    """Existing Stancil cross sections with the chemistry's identical K(T).

    Cross sections retain the existing low-T boundary hold and wavelength
    support. Only the positive equilibrium interpolation is changed here.
    This is explicitly an extrapolation below 4200 K, not validated new data.
    """
    wave, temp = np.broadcast_arrays(np.asarray(wavelength, dtype=float),
                                    np.asarray(temperature, dtype=float))
    if np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wavelength must be finite and positive")
    k, _ = dissociation_equilibrium(temp)
    tabulated_t = np.maximum(temp, molecular._TEMPERATURE[0])
    bf = molecular._linear_interpolate_table(wave, tabulated_t,
        molecular._WAVELENGTH_BOUND_FREE, molecular._BOUND_FREE)
    ff = molecular._linear_interpolate_table(wave, tabulated_t,
        molecular._WAVELENGTH_FREE_FREE, molecular._FREE_FREE)
    bf = np.where((wave >= molecular._WAVELENGTH_BOUND_FREE[0])
                  & (wave <= molecular._WAVELENGTH_BOUND_FREE[-1]), bf, 0.)
    ff = np.where((wave >= molecular._WAVELENGTH_FREE_FREE[0])
                  & (wave <= molecular._WAVELENGTH_FREE_FREE[-1]), ff, 0.)
    return np.maximum(bf/k+ff, 0.)


@contextmanager
def helium_dimer_atmosphere_experiment(runner):
    """Scope EOS, thermal response and opacity together in one diagnostic process.

    Experimental models deliberately use a different output format: they
    cannot masquerade as a production atmosphere checkpoint when reloaded.
    """
    from wd_spectra import atmosphere, convection, helium
    from wd_spectra.models.common import _jsonable

    def lte(t, p, **options):
        return helium_dimer_lte(t, p, **options).atomic

    def thermodynamics(t, p, **options):
        if options.pop('helium_reos3_table', None) is not None:
            raise ValueError('Dimer experiment does not combine with a bulk-EOS substitution')
        return helium_dimer_thermodynamics(t, p, **options)

    def save_experimental(result, path):
        path = Path(path)
        a = result.atmosphere
        closure = helium_dimer_lte(a.temperature, a.gas_pressure)
        # Deliberately no file named atmosphere.npz and no public checkpoint
        # field named temperature: loading this as a default model must fail.
        np.savez_compressed(path/'experimental-dimer-structure.npz',
            experimental_temperature=a.temperature, experimental_pressure=a.gas_pressure,
            experimental_column_mass=a.column_mass, experimental_tau_rosseland=a.rosseland_optical_depth,
            experimental_mass_density=a.mass_density,
            experimental_electron_density=a.electron_density,
            experimental_molecular_ion_density=closure.molecular_ion_density)
        np.savetxt(path/'experimental-spectrum.txt',
                   np.column_stack((result.spectrum.wavelength_angstrom,
                                    result.spectrum.surface_flux_lambda)),
                   header='EXPERIMENTAL He2+ EOS/opacity; not a production-physics model\nwavelength_A surface_flux_lambda')
        (path/'experimental-metadata.json').write_text(json.dumps(dict(
            physics='HM atoms + ideal He2+ equilibrium; not validated dense-He chemical potentials',
            compatible_with_production_checkpoint_loader=False,
            model=_jsonable(result.metadata), atmosphere=_jsonable(a.metadata)),
            indent=2, allow_nan=False)+'\n')
        return path

    with ExitStack() as stack:
        for module in (eos, atmosphere, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_lte', lte))
        for module in (eos, convection, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_thermodynamics', thermodynamics))
        stack.enter_context(patch.object(helium, 'helium_dimer_ion_continuum_coefficient', helium_dimer_continuum))
        stack.enter_context(patch.object(runner, 'save_model_result', save_experimental))
        yield
