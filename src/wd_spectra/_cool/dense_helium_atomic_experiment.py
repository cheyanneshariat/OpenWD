"""Controlled ATOMIC dense-He component test, never a production EOS.

REOS.3 supplies bulk density and caloric quantities. Trace ionization uses
the explicitly approximate HNC pair model in dense_helium_fluid_experiment.
This is NOT Kowalski (2007): the potentials/closure differ, He2+ is absent,
and the published electron fit has an unresolved derivative discontinuity.
All out-of-domain requests fail. There is no alternate EOS or electron floor.
"""
import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import RectBivariateSpline
import wd_spectra.eos as eos
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from wd_spectra.dense_eos import read_helium_reos3_table
from .dense_helium_fluid_experiment import (
    atomic_hnc_isotherm, electron_insertion_ev, KB_EV, EV,
)
from .dense_helium_limits import DenseHeliumDomainError

PHYSICS = ('EXPERIMENTAL atomic REOS3 + HNC neutral/He+ insertion + thesis electron '
           'insertion; He2+ chemistry/bound-free ABSENT; collective He- free-free '
           'and refraction ABSENT; not validated dense-He atmosphere physics')
TABLE_VERSION = 'atomic-hnc-young1981-bruno2010-v1'


def build_table(path, maximum_density=1.6):
    """Tabulate fluid equations, not atmosphere-dependent fitted corrections."""
    path = Path(path)
    if path.exists():
        raise ValueError('refusing to overwrite an interaction table')
    if not 0 < maximum_density <= 2.:
        raise ValueError('maximum density must remain inside the electron fit support (<=2 g/cm3)')
    temperatures = np.geomspace(3000., 17000., 41)
    densities = np.geomspace(1e-12, maximum_density, 161)
    potentials = []
    structures = []
    for index, temperature in enumerate(temperatures):
        print(f'Building isotherm {index+1}/{len(temperatures)} T={temperature:g}', flush=True)
        mu, structure = atomic_hnc_isotherm(temperature, densities)
        potentials.append(mu)
        structures.append(structure)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, temperature=temperatures, density=densities,
                        mu=np.asarray(potentials), structure_zero=structures,
                        version=TABLE_VERSION, physics=PHYSICS)


class AtomicDenseEOS:
    table_version = TABLE_VERSION
    species_count = 2
    physics = PHYSICS
    dimer_description = 'free-free only; molecular bound-free absent'
    output_stem = 'atomic-dense'

    def __init__(self, bulk, interaction_path):
        self.bulk = bulk
        self.path = Path(interaction_path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as table:
            if str(table['version']) != self.table_version:
                raise ValueError('incompatible atomic HNC interaction table')
            self.temperature = table['temperature'].copy()
            self.density = table['density'].copy()
            mu = table['mu'].copy()
        if (self.temperature.ndim != 1 or self.density.ndim != 1
                or min(len(self.temperature),len(self.density)) < 4
                or np.any(~np.isfinite(self.temperature)) or np.any(~np.isfinite(self.density))
                or np.any(self.temperature <= 0) or np.any(self.density <= 0)
                or np.any(np.diff(self.temperature) <= 0) or np.any(np.diff(self.density) <= 0)
                or mu.shape != (len(self.temperature),len(self.density),self.species_count)
                or np.any(~np.isfinite(mu))):
            raise ValueError('invalid HNC interaction table contents')
        # Interpolate mu/rho: its finite second-virial limit avoids loss of
        # dilute accuracy over many orders of magnitude in density.
        self.curves = [RectBivariateSpline(np.log(self.temperature), np.log(self.density),
            mu[..., i]/self.density[None, :], s=0) for i in range(self.species_count)]

    def ionization_shift_ev(self, temperature, density):
        terms = self.excess_potentials(temperature,density)
        return electron_insertion_ev(temperature,density)+terms[1]-terms[0]

    def excess_potentials(self, temperature, density):
        t, rho = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(density, float))
        if (np.any(~np.isfinite(t)) or np.any(~np.isfinite(rho))
                or np.any(t < self.temperature[0]) or np.any(t > self.temperature[-1])
                or np.any(rho < self.density[0]) or np.any(rho > self.density[-1])):
            raise DenseHeliumDomainError(f'atomic HNC table domain violation: T={t.min():g}..{t.max():g}, '
                             f'rho={rho.min():g}..{rho.max():g}; no substitution')
        terms = [rho*curve.ev(np.log(t), np.log(rho)) for curve in self.curves]
        return terms

    def bulk_state(self, temperature, pressure):
        rho, energy, inside = self.bulk.evaluate(pressure, temperature)
        if not np.all(inside):
            raise DenseHeliumDomainError('REOS3 domain violation; no substitution')
        return rho, energy

    def lte(self, temperature, pressure, **options):
        t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
        rho, _ = self.bulk_state(t, p)
        nuclei = rho/HELIUM_MASS
        shift = self.ionization_shift_ev(t, rho)
        weights = eos.HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT*np.exp(
            -eos.HELIUM_I_LOW_TERM_ENERGY/(BOLTZMANN*t[..., None]))
        if np.any(np.sum(weights[..., 1:], axis=-1) > 1e-3):
            raise DenseHeliumDomainError('ground-state chemical approximation exceeded excited-state domain')
        partition = np.sum(weights, axis=-1)
        # Ground-state partition functions: He=1, He+=2; free electron spin=2.
        log_saha = (np.log(4.)-np.log(partition)+1.5*np.log(2*np.pi*eos.ELECTRON_MASS*BOLTZMANN*t/eos.PLANCK**2)
                    -(eos.HELIUM_FIRST_IONIZATION_ENERGY+EV*shift)/(BOLTZMANN*t))
        ion_fraction = atomic_ion_fraction(log_saha-np.log(nuclei))
        if np.any(ion_fraction > 1e-3):
            raise DenseHeliumDomainError('trace-ion HNC approximation exceeded 0.1% ionization; no substitution')
        ne = nuclei*ion_fraction
        neutral = nuclei-ne
        return eos.HeliumLTEState(
            mass_density=rho, helium_nuclei_density=nuclei,
            neutral_he_density=neutral, singly_ionized_he_density=ne,
            doubly_ionized_he_density=np.zeros_like(ne), electron_density=ne,
            mean_ion_charge=ion_fraction, neutral_partition_function=partition,
            singly_ionized_partition_function=np.full_like(ne, 2.),
            neutral_level_occupation_probability=np.ones_like(weights),
            neutral_level_population_density=neutral[..., None]*weights/partition[..., None],
            microfield_model='qmhd',
            neutral_radius_scale=options.get('neutral_radius_scale', eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE))

    def thermodynamics(self, temperature, pressure, **options):
        # As in the neutral-dominated REOS + trace-chemistry construction,
        # the tabulated bulk energy is not amended with a second ideal EOS.
        if hasattr(self.bulk, 'thermodynamics'):
            return self.bulk.thermodynamics(temperature, pressure)
        t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
        eps = 2e-4
        cold_rho, cold_u = self.bulk_state(t*np.exp(-eps), p)
        hot_rho, hot_u = self.bulk_state(t*np.exp(eps), p)
        rho, _ = self.bulk_state(t, p)
        cp = ((hot_u+p/hot_rho)-(cold_u+p/cold_rho))/(2*t*np.sinh(eps))
        expansion = -np.log(hot_rho/cold_rho)/(2*eps)
        gradient = p*expansion/(rho*t*cp)
        if (np.any(~np.isfinite(cp)) or np.any(cp <= 0) or np.any(expansion <= 0)
                or np.any(~np.isfinite(gradient))):
            raise ValueError('invalid REOS thermal derivatives')
        return eos.HeliumThermodynamics(cp, expansion, gradient)


def atomic_ion_fraction(log_saha_over_nuclei):
    """Exact x^2/(1-x)=S/n, including saturation without cancellation."""
    log_s = np.asarray(log_saha_over_nuclei, float)
    if np.any(~np.isfinite(log_s)):
        raise ValueError('nonfinite ionization equilibrium')
    # The negative exponential never overflows; logaddexp also handles
    # regimes far beyond the trace approximation for the algebra tests.
    log_x = np.log(2.)-.5*np.logaddexp(0., np.log(4.)-log_s)
    x = np.exp(log_x)/(1+np.exp(-.5*np.logaddexp(0., np.log(4.)-log_s)))
    return x


def atomic_collision_continuum(wavelength, temperature):
    """Stancil free--free only: no fictitious ideal-equilibrium He2+ donors."""
    from wd_spectra import helium_molecular as molecular
    wave, temp = np.broadcast_arrays(np.asarray(wavelength, float), np.asarray(temperature, float))
    ff = molecular._linear_interpolate_table(wave, np.maximum(temp, molecular._TEMPERATURE[0]),
                                            molecular._WAVELENGTH_FREE_FREE, molecular._FREE_FREE)
    return np.where((wave >= molecular._WAVELENGTH_FREE_FREE[0])
                    & (wave <= molecular._WAVELENGTH_FREE_FREE[-1]), ff, 0.)


def transport_surface_boundary(runner, model, teff, logg, **options):
    """Construct just the Eddington surface seed, not an unnecessary gray column.

    The old seed computes a whole gray atmosphere to use only its first point,
    probing arbitrarily low P and high T outside this EOS's domain. Here solve
    P*kappa_R/g=tau_min at the same Eddington boundary temperature. This is an
    approximate INITIAL condition, not a converged atmosphere or EOS fallback.
    """
    from scipy.optimize import brentq
    tau = 1e-8
    temperature = teff*(.75*(tau+2/3))**.25
    upper = np.searchsorted(model.bulk.temperature_grid, temperature)
    if upper == 0 or upper == len(model.bulk.temperature_grid):
        raise ValueError('surface temperature outside REOS support')
    pressure_min = max(model.bulk.pressure_by_temperature[j][0] for j in (upper-1, upper))
    pressure_min = max(pressure_min, getattr(model.bulk, 'p_min', pressure_min))

    def at_pressure(logp):
        p, t = np.array([np.exp(logp)]), np.array([temperature])
        point = runner.atmosphere_at(teff, p, t, np.array([tau]), logg=logg)
        opacity = runner.rosseland_mean_helium_continuum_opacity(point, n_frequency=160)[0]
        return np.log(p[0]*opacity/(10**logg*tau)), point

    lower = np.log(pressure_min*(1+1e-8))
    if at_pressure(lower)[0] >= 0:
        raise ValueError('requested surface depth lies below REOS support')
    upper = lower+1
    while at_pressure(upper)[0] < 0:
        upper += 1
    logp = brentq(lambda lp: at_pressure(lp)[0], lower, upper, xtol=1e-11)
    return at_pressure(logp)[1]


@contextmanager
def atomic_dense_experiment(runner, model, *, least_squares_merit=False, opacity_factory=None):
    from wd_spectra import atmosphere, convection, helium
    from wd_spectra.models.common import _jsonable
    original_compute = runner.compute_db
    original_run = runner.run
    original_solver = runner.radiative_equilibrium_helium_atmosphere

    def run(args):
        args.experimental_physics = model.physics
        if hasattr(model,'sha256'):
            args.experimental_material_certificate_version=2
            args.experimental_material_table_sha256=model.sha256
        args.experimental_exact_dense_materials = getattr(model,'exact_dense_materials',False)
        args.experimental_dense_options = getattr(model,'experiment_options',{})
        args.experimental_positive_radiative_rates = getattr(model,'positive_radiative_rates',False)
        args.experimental_full_coupled_newton = getattr(model,'full_coupled_newton',False)
        args.experimental_source_sha256 = getattr(model,'source_sha256',{})
        return original_run(args)

    def solve(*args, **kwargs):
        a = original_solver(*args, **kwargs)
        return replace(a, metadata={**a.metadata,
            'eos': model.physics, 'bulk_helium_eos': model.bulk.source,
            'helium_dimer_ion_continuum': model.dimer_description,
            'experimental_nonlinear_merit': 'RMS' if least_squares_merit else 'production RMS + 0.25 max',
            'experimental_dense_helium': model.physics, 'interaction_table_sha256': model.sha256})

    def compute(*args, **kwargs):
        a = kwargs['initial_atmosphere']
        kwargs['initial_atmosphere'] = replace(a, metadata={**a.metadata,
            'experimental_dense_helium': model.physics, 'interaction_table_sha256': model.sha256})
        return original_compute(*args, **kwargs)

    def save(result, path):
        path = Path(path)
        a = result.atmosphere
        extra = {}
        if hasattr(model,'chemistry'):
            chemistry=model.chemistry(a.temperature,a.mass_density)
            extra['experimental_molecular_ion_density']=chemistry.molecular_ion
            extra['experimental_atomic_ion_density']=chemistry.atomic_ion
        np.savez_compressed(path/f'experimental-{model.output_stem}-structure.npz',
            experimental_logg=a.logg, experimental_effective_temperature=a.effective_temperature,
            experimental_temperature=a.temperature, experimental_pressure=a.gas_pressure,
            experimental_density=a.mass_density, experimental_electron_density=a.electron_density,
            experimental_column_mass=a.column_mass, experimental_tau=a.rosseland_optical_depth,**extra)
        np.savetxt(path/'experimental-spectrum.txt', np.column_stack((
            result.spectrum.wavelength_angstrom, result.spectrum.surface_flux_lambda)), header=model.physics)
        (path/'experimental-metadata.json').write_text(json.dumps(dict(
            effective_temperature=a.effective_temperature, logg=a.logg,
            physics=model.physics, interaction_table_sha256=model.sha256,
            compatible_with_production_checkpoint_loader=False,
            atmosphere=_jsonable(a.metadata)), indent=2)+'\n')

    with ExitStack() as stack:
        # The standalone research runner installs proposal dispatchers by
        # assignment. Restore those too if this scope is used in a longer
        # lived Python process, including an exception during a run.
        from wd_spectra import adaptive_structure
        for name in ('solve_adaptive_lte_structure', 'solve_trust_region_newton'):
            stack.enter_context(patch.object(adaptive_structure, name,
                                             getattr(adaptive_structure, name)))
        if least_squares_merit:
            from wd_spectra import nonlinear
            # Match the cheap least-squares proposal's objective. All actual
            # flux/local-energy/step convergence gates are unchanged.
            stack.enter_context(patch.object(nonlinear, '_residual_merit',
                lambda r: float(np.sqrt(np.mean(np.asarray(r)**2)))))
        for module in (eos, atmosphere, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_lte', model.lte))
        for module in (eos, convection, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_thermodynamics', model.thermodynamics))
        if opacity_factory is None:
            stack.enter_context(patch.object(helium, 'helium_dimer_ion_continuum_coefficient', atomic_collision_continuum))
        else:
            absorption = opacity_factory(helium.helium_continuum_mass_absorption_coefficient)
            for module in (helium,runner):
                stack.enter_context(patch.object(module,'helium_continuum_mass_absorption_coefficient',absorption))
        stack.enter_context(patch.object(runner, 'compute_db', compute))
        stack.enter_context(patch.object(runner, 'save_model_result', save))
        stack.enter_context(patch.object(runner, 'run', run))
        stack.enter_context(patch.object(runner, 'radiative_equilibrium_helium_atmosphere', solve))
        stack.enter_context(patch.object(runner, 'helium_continuum_atmosphere',
            lambda *args, **kwargs: transport_surface_boundary(runner, model, *args, **kwargs)))
        yield


def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction-table', type=Path, required=True)
    parser.add_argument('--build-table', action='store_true')
    parser.add_argument('--maximum-density', type=float, default=1.6)
    parser.add_argument('--smooth-bulk', action='store_true')
    parser.add_argument('--least-squares-merit', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.build_table:
        if remaining:
            parser.error('table build does not accept atmosphere arguments')
        return build_table(args.interaction_table, args.maximum_density)
    from . import check_cool_db_transport_seed as runner
    from wd_spectra.models import ModelData
    forbidden = ('--reos3', '--helium-dimer', '--log-h-he', '--reuse-fresh-seed',
                 '--resume-atmosphere', '--refine-atmosphere', '--extend-atmosphere',
                 '--stable-helium-thermodynamics', '--smooth-h2', '--state-sum-h2')
    if any(x.split('=')[0] in forbidden for x in remaining):
        parser.error('atomic dense experiment requires a fresh pure-He run, no other EOS substitution')
    model = AtomicDenseEOS(read_helium_reos3_table(ModelData.default().helium_reos3), args.interaction_table)
    if args.smooth_bulk:
        from .smooth_reos3_experiment import SmoothREOS3
        model.bulk = SmoothREOS3(model.bulk,model.temperature[0],model.temperature[-1])
        print(model.bulk.source,flush=True)
    print(PHYSICS, flush=True)
    sys.argv = [sys.argv[0]]+remaining
    with atomic_dense_experiment(runner, model, least_squares_merit=args.least_squares_merit):
        runner.main()
    # The existing runner's generic summary must not misidentify this EOS.
    run_parser = argparse.ArgumentParser(add_help=False)
    run_parser.add_argument('temperature', type=int)
    run_parser.add_argument('--output-root', type=Path, required=True)
    run_args, _ = run_parser.parse_known_args(remaining)
    summary_path = run_args.output_root/str(run_args.temperature)/'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(physics=PHYSICS, validated_full_physics=False,
                   interaction_table_sha256=model.sha256)
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
