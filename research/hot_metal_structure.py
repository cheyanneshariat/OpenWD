#!/usr/bin/env python3
"""Metal-blanketed temperature structure for the hot DA/DAO trace-metal model (research).

The fixed-host trace-metal solution ignores the metals' heating and cooling.
This driver puts the metal opacity and emissivity into the energy balance with
a TMAP-style iteration (Werner & Dreizler 1999; Werner et al. 2003), reusing
the PG 1159 hybrid temperature correction:

    combined H/He + metal coefficients on a union wavelength grid
    -> formal solution -> local radiative-equilibrium step (outer layers) /
       Unsöld-Lucy step (deep) -> rebuild EOS and hydrostatics at fixed column mass
    -> (outer loop) re-solve H/He populations at fixed T with the metal
       opacity in their radiation field, then the metal NLTE populations.

Inside the temperature loop all departure coefficients (H/He and metal) are
held fixed and the LTE references follow the trial temperature. The setup is
captured from ``explore_hot_composition.py`` so the atoms, data and abundances
are exactly those of the converged trace-metal run being continued.

The ``audit`` command only evaluates the energy balance of a saved state.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wd_spectra._compat import trapezoid
from wd_spectra._hot_structure import HotEquations
from wd_spectra._mass_feautrier import mass_emissivity_energy, mass_width
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import (
    _atom_selection, fixed_electron_metal_reference, trace_metal_coefficients)
from wd_spectra.light_metal_nlte import (
    hot_metal_line_nlte_coefficients, light_metal_bound_free_nlte_coefficients)
from wd_spectra.models.common import numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra.nlte_core import NLTETransferCoefficients

LOGGER = logging.getLogger('hot_metal_structure')


class _Captured(Exception):
    pass


def capture_setup(runner_arguments):
    """Run the composition runner up to its solver call and return its inputs."""
    import explore_hot_composition as ehc
    captured = {}

    def capture(atmosphere, background, abundances, wave, **options):
        captured.update(atmosphere=atmosphere, background=background,
                        abundances=dict(abundances), wave=wave, **options)
        raise _Captured
    original, ehc.solve_hot_trace_metals = ehc.solve_hot_trace_metals, capture
    saved_argv = sys.argv
    sys.argv = ['explore_hot_composition.py'] + list(runner_arguments)
    try:
        ehc.main()
    except _Captured:
        pass
    finally:
        sys.argv = saved_argv
        ehc.solve_hot_trace_metals = original
    if not captured:
        raise RuntimeError('composition runner did not reach the trace-metal solve')
    # The runner's background closure holds the host model and Fe/Ni remainder setup.
    closure = dict(zip(captured['background'].__code__.co_freevars,
                       (cell.cell_contents for cell in captured['background'].__closure__)))
    host_closure = dict(zip(closure['host_background'].__code__.co_freevars,
                            (cell.cell_contents for cell in closure['host_background'].__closure__)))
    return SimpleNamespace(
        host=host_closure['host'], model=host_closure['model'],
        database=captured['atomic_database'], photo=captured['photoionization_database'],
        abundances=captured['abundances'], counts=captured['levels_per_charge'],
        thresholds=captured.get('photoionization_threshold_data') or {},
        collisions=captured.get('collision_data'),
        total_recombination=captured.get('total_recombination'),
        population_wavelength=captured['population_wavelength'],
        initial_populations=captured['initial_populations'],
        remainder_abundances=closure['background_abundances'],
        remainder_counts=closure['counts'], closed=closure['closed'],
        retained=closure['retained'], output=Path(runner_arguments[runner_arguments.index('--output') + 1]))


def remainder_terms(setup, atmosphere, wavelength):
    """LTE Fe/Ni opacity outside the explicit atoms, at this atmosphere."""
    absorption = np.zeros((len(wavelength), atmosphere.n_depth))
    emission = np.zeros_like(absorption)
    if not setup.remainder_abundances:
        return absorption, emission
    reference = fixed_electron_metal_reference(atmosphere, setup.database, setup.remainder_abundances)
    unity = {(e, s.charge): np.ones(atmosphere.n_depth)
             for e in setup.remainder_abundances for s in setup.database.ion_stages(e)}
    for e in setup.remainder_abundances:
        a, j = hot_metal_line_nlte_coefficients(
            atmosphere, wavelength, setup.database, reference, unity, elements=(e,),
            minimum_oscillator_strength=1e-4, maximum_lines=None,
            transition_keys=setup.retained.get(e), include_static_linear_stark=False)
        bf_counts = ({s.charge: (0 if s.charge in sorted(setup.remainder_counts[e])[:-1] else 1)
                      for s in setup.database.ion_stages(e)} if e in setup.closed else None)
        b, k = light_metal_bound_free_nlte_coefficients(
            atmosphere, wavelength, setup.database, reference, setup.photo, unity,
            elements=(e,), levels_per_charge=bf_counts, include_explicit_kramers=True)
        absorption += a + b
        emission += j + k
    return absorption, emission


def metal_terms(setup, atmosphere, wavelength, departures):
    """Explicit-atom NLTE metals (frozen departures) plus the LTE Fe/Ni remainder."""
    zero = np.zeros((len(wavelength), atmosphere.n_depth))
    reference = fixed_electron_metal_reference(atmosphere, setup.database, setup.abundances)
    selection = {e: _atom_selection(setup.database, e, setup.counts[e]) for e in setup.abundances}
    explicit = trace_metal_coefficients(
        atmosphere, wavelength, NLTETransferCoefficients(wavelength, zero, zero, zero, {}),
        setup.database, setup.photo, reference, selection, departures, setup.abundances,
        thresholds=setup.thresholds)
    a, j = remainder_terms(setup, atmosphere, wavelength)
    return explicit.true_absorption + a, explicit.thermal_emissivity + j


def structure_grid(setup):
    host = setup.host
    equations = HotEquations(host.atmosphere, setup.model, structure_wavelength(
        host.config, numerical_resolution(host.config.quality).n_continuum))
    return np.unique(np.r_[equations.wave, setup.population_wavelength])


def energy_balance(atmosphere, coefficients, n_angle):
    """Flux-constancy and local radiative-equilibrium residuals on one grid."""
    _, field, closure = transfer_field(atmosphere, coefficients, n_angle=n_angle)
    wave = coefficients.wavelength_angstrom
    target = STEFAN_BOLTZMANN * atmosphere.effective_temperature**4
    flux = trapezoid(field.interface_flux, wave, axis=0) / target - 1
    energy, emission = mass_emissivity_energy(wave, atmosphere.column_mass,
        coefficients.thermal_emissivity, field.mean_intensity, coefficients.true_absorption)
    relative = energy / np.maximum(abs(emission), 1e-30 * target)
    return field, dict(surface_flux_ratio=float(flux[0] + 1),
                       maximum_all_depth_flux_residual=float(np.max(abs(flux))),
                       maximum_relative_cell_energy_residual=float(np.max(abs(relative))),
                       relative_flux=flux, relative_cell_energy=relative,
                       source_closure_residual=float(closure))


def audit(arguments):
    setup = capture_setup(arguments.runner)
    host, model = setup.host, setup.model
    departures = {e: s.level_departure_coefficient for e, s in setup.initial_populations.items()}
    start = time.monotonic()
    wave = structure_grid(setup)
    LOGGER.info('union structure grid: %d wavelengths', len(wave))
    base = model.transfer_coefficients(host.atmosphere, wave, host.population_state)
    LOGGER.info('host coefficients %.0f s', time.monotonic() - start)
    t = time.monotonic()
    a, j = metal_terms(setup, host.atmosphere, wave, departures)
    LOGGER.info('metal coefficients %.0f s', time.monotonic() - t)
    report = {}
    for label, coefficients in (
            ('HHe', base),
            ('WithMetals', NLTETransferCoefficients(wave, base.true_absorption + a,
                                                    base.thermal_emissivity + j, base.scattering, {}))):
        t = time.monotonic()
        _, diagnostics = energy_balance(host.atmosphere, coefficients, model.n_angle)
        LOGGER.info('%s transfer %.0f s', label, time.monotonic() - t)
        report[label] = {k: v for k, v in diagnostics.items() if not isinstance(v, np.ndarray)}
        report[label]['relative_flux_by_depth'] = diagnostics['relative_flux'].tolist()
        report[label]['relative_cell_energy_by_depth'] = diagnostics['relative_cell_energy'].tolist()
    report.update(wavelength_points=len(wave), elapsed_seconds=time.monotonic() - start,
                  column_mass=host.atmosphere.column_mass.tolist(),
                  temperature=host.atmosphere.temperature.tolist())
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'energy-audit.json').write_text(json.dumps(report, indent=1) + '\n')
    for label in ('HHe', 'WithMetals'):
        r = report[label]
        print(f"{label}: surface F/sigmaT^4 = {r['surface_flux_ratio']:.5f}, max all-depth flux residual "
              f"{r['maximum_all_depth_flux_residual']:.3e}, max |cell heating/emission| "
              f"{r['maximum_relative_cell_energy_residual']:.3e}", flush=True)


class MetalBlanketedHostModel:
    """H/He model whose transfer coefficients include frozen-departure metals.

    Everything else (rates, line problems, EOS rebuild, remap) is the H/He
    model's own, so the fixed-temperature H/He solve sees the metal opacity
    and emissivity in its radiation field. Metal terms depend only on the
    atmosphere (and the frozen departures), so they are cached per structure.
    """

    def __init__(self, base, setup, departures):
        self._base, self._setup, self._departures = base, setup, departures
        self._cache = {}

    def __getattr__(self, name):
        return getattr(self._base, name)

    def transfer_coefficients(self, atmosphere, wavelength, state, **options):
        c = self._base.transfer_coefficients(atmosphere, wavelength, state, **options)
        key = (atmosphere.temperature.tobytes(), wavelength.tobytes())
        if key not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = metal_terms(self._setup, atmosphere, wavelength, self._departures)
        a, j = self._cache[key]
        return NLTETransferCoefficients(wavelength, c.true_absorption + a, c.thermal_emissivity + j,
                                        c.scattering, c.metadata)


def combined_coefficients(setup, atmosphere, host_populations, wavelength, departures):
    base = setup.model.transfer_coefficients(atmosphere, wavelength, host_populations)
    a, j = metal_terms(setup, atmosphere, wavelength, departures)
    return NLTETransferCoefficients(wavelength, base.true_absorption + a,
                                    base.thermal_emissivity + j, base.scattering, {})


def save_metal_states(path, states):
    arrays = {}
    for e, s in states.items():
        arrays[f'{e}_level_key'] = np.asarray([[str(k[0]), str(k[1]), str(k[2])] for k in s.level_key])
        arrays[f'{e}_population_density'] = s.population_density
        arrays[f'{e}_lte_population_density'] = s.lte_population_density
    np.savez_compressed(path, **arrays)


def relax(arguments):
    """Outer/inner TMAP-style iteration for a metal-blanketed structure."""
    from run_hot_public_diagnostic import save_state
    from wd_spectra._hot_structure import solve as solve_hot_structure
    from wd_spectra._pg1159_ali_temperature import temperature_correction
    from wd_spectra.hot_trace_metals import solve_hot_trace_metals

    setup = capture_setup(arguments.runner)
    host, model = setup.host, setup.model
    out = Path(arguments.output)
    out.mkdir(parents=True, exist_ok=False)
    seed_atmosphere = host.atmosphere
    atmosphere, host_populations = host.atmosphere, host.population_state
    states = dict(setup.initial_populations)
    departures = {e: s.level_departure_coefficient for e, s in states.items()}
    wave = structure_grid(setup)
    host_wave = structure_wavelength(host.config, numerical_resolution(host.config.quality).n_continuum)
    target = STEFAN_BOLTZMANN * atmosphere.effective_temperature**4
    tau = np.asarray(atmosphere.rosseland_optical_depth)
    band_first = int(np.argmax(np.r_[tau[:-1] >= arguments.local_energy_minimum_tau, True]))
    history = []
    skip_inner = skip_host = False
    if arguments.resume_host is not None:
        from wd_spectra.hot_nlte import population_arrays, with_departures
        with np.load(arguments.resume_host, allow_pickle=False) as saved:
            atmosphere = model.rebuild_atmosphere(seed_atmosphere, saved['temperature'])
            reference = model._rate_state(atmosphere)
            _, expected = population_arrays(reference)
            np.testing.assert_allclose(saved['lte_population'], expected, rtol=1e-10, atol=0)
            host_populations = with_departures(reference, saved['population'] / saved['lte_population'])
        skip_inner, skip_host = not arguments.resume_run_temperature_loop, True
        LOGGER.info('relax: resumed temperature and H/He populations from %s', arguments.resume_host)
    if arguments.resume_metals is not None:
        from wd_spectra.light_metal_nlte import ReducedLightMetalLevelState
        with np.load(arguments.resume_metals, allow_pickle=False) as saved:
            for e in list(states):
                keys = tuple((str(k[0]), int(k[1]), int(k[2])) for k in saved[f'{e}_level_key'])
                n, lte = saved[f'{e}_population_density'], saved[f'{e}_lte_population_density']
                dep = {k: n[i] / np.maximum(lte[i], 1e-300) for i, k in enumerate(keys)}
                states[e] = ReducedLightMetalLevelState(e, keys, n, lte, dep, 0., {'resumed': True})
        departures = {e: s.level_departure_coefficient for e, s in states.items()}
        LOGGER.info('relax: resumed metal populations from %s', arguments.resume_metals)
    elif arguments.resume_temperature is not None:
        with np.load(arguments.resume_temperature, allow_pickle=False) as saved:
            atmosphere = model.rebuild_atmosphere(seed_atmosphere, saved['temperature'])
        host_populations = model.remap(atmosphere, host_populations)
        skip_inner = True
        LOGGER.info('relax: resumed temperature from %s', arguments.resume_temperature)
    LOGGER.info('relax: %d structure wavelengths, band starts at cell %d', len(wave), band_first)

    def log(record):
        history.append(record)
        (out / 'history.json').write_text(json.dumps(history, indent=1) + '\n')
        LOGGER.info('relax %s', {k: v for k, v in record.items() if not isinstance(v, list)})

    converged = False
    for outer in range(arguments.outer_iterations):
        previous = None
        stationary = False
        for inner in range(0 if (outer == 0 and skip_inner) else arguments.inner_iterations):
            start = time.monotonic()
            coefficients = combined_coefficients(setup, atmosphere, host_populations, wave, departures)
            field, diagnostics = energy_balance(atmosphere, coefficients, model.n_angle)
            band = slice(band_first, atmosphere.n_depth - 1)
            local = float(np.max(abs(diagnostics['relative_cell_energy'][band])))
            flux = diagnostics['maximum_all_depth_flux_residual']
            gates = flux < arguments.flux_tolerance and local < arguments.local_energy_tolerance
            shim = SimpleNamespace(wave=wave, target=target, model=None)
            step, correction = temperature_correction(
                shim, atmosphere, host_populations,
                evaluation={'_transfer_coefficients': coefficients, '_radiation_field': field},
                previous=previous, maximum_step=arguments.maximum_step, probe_step=arguments.probe_step,
                band_first=band_first, local_maximum_rosseland_depth=arguments.local_maximum_tau)
            if previous is not None:
                # The PG 1159 correction halves reversing steps only in the
                # local-RE band; apply the same contraction to the deep
                # Unsöld-Lucy cells, which otherwise overshoot a large deficit.
                prior = previous['step']
                flipped = np.sign(step) * np.sign(prior) < 0
                step = np.where(flipped, np.clip(step, -0.5 * abs(prior), 0.5 * abs(prior)), step)
                correction['step'] = step
            largest = float(np.max(abs(step)))
            log(dict(outer=outer, inner=inner, surface_flux_ratio=diagnostics['surface_flux_ratio'],
                     maximum_flux_residual=flux, maximum_band_local_energy=local, gates=bool(gates),
                     largest_log_temperature_step=largest, seconds=round(time.monotonic() - start),
                     temperature=atmosphere.temperature.tolist(),
                     relative_cell_energy=diagnostics['relative_cell_energy'].tolist(),
                     log_temperature_step=np.asarray(step).tolist(),
                     slope_source=[str(x) for x in correction.get('slope_source', [])]))
            np.savez_compressed(out / 'state-temperature.npz', temperature=atmosphere.temperature,
                                column_mass=atmosphere.column_mass)
            if gates and largest < arguments.stationary_step:
                stationary = True
                break
            del coefficients, field
            previous = correction
            temperature = atmosphere.temperature * np.exp(step)
            atmosphere = model.rebuild_atmosphere(seed_atmosphere, temperature)
            host_populations = model.remap(atmosphere, host_populations)
        if stationary and outer > 0 and history[-1]['inner'] == 0:
            converged = True
            LOGGER.info('relax: temperature stationary after population update; converged')
            break
        # Re-solve H/He populations at fixed T with the frozen metals in their radiation field.
        start = time.monotonic()
        if not (outer == 0 and skip_host):
            blanketed = MetalBlanketedHostModel(model, setup, departures)
            result = solve_hot_structure(atmosphere, blanketed, host_wave, fixed_temperature=True,
                                         populations=host_populations,
                                         maximum_iterations=arguments.host_iterations)
            # The solver rebuilds its own atmosphere (exp(log T)); its populations
            # belong to that object, so carry it forward.
            atmosphere, host_populations = result.atmosphere, result.population_state
            save_state(out / f'host-outer-{outer:02d}.npz', atmosphere, host_populations, model)
            LOGGER.info('relax: H/He fixed-T solve %.0f s, populations converged %s',
                        time.monotonic() - start, host_populations.converged)

        # Re-solve metal NLTE populations on this structure and H/He background.
        def background(w, a=atmosphere, p=host_populations):
            base = model.transfer_coefficients(a, w, p)
            ra, rj = remainder_terms(setup, a, w)
            return NLTETransferCoefficients(w, base.true_absorption + ra, base.thermal_emissivity + rj,
                                            base.scattering, {})
        start = time.monotonic()
        metals = solve_hot_trace_metals(
            atmosphere, background, setup.abundances, np.linspace(1175., 1177., 401),
            atomic_database=setup.database, photoionization_database=setup.photo,
            levels_per_charge=setup.counts, population_wavelength=setup.population_wavelength,
            initial_populations=states, photoionization_threshold_data=setup.thresholds,
            collision_data=setup.collisions, total_recombination=setup.total_recombination,
            accelerated_lambda=True, convergence_criterion='opacity',
            tolerance=arguments.metal_tolerance, acceleration_depth=16,
            maximum_iterations=arguments.metal_iterations, require_convergence=False,
            n_angle=model.n_angle)
        states = dict(metals.populations)
        departures = {e: s.level_departure_coefficient for e, s in states.items()}
        save_metal_states(out / f'metals-outer-{outer:02d}.npz', states)
        LOGGER.info('relax: metal NLTE %.0f s, converged %s, opacity defect %s', time.monotonic() - start,
                    metals.converged, (metals.metadata.get('opacity_defect_history') or [None])[-1])
    (out / 'result.json').write_text(json.dumps(dict(converged=converged, outer_iterations=outer + 1,
        final=history[-1] if history else None), indent=1) + '\n')


def heating(arguments):
    """Split each cell's net radiative heating into H/He, explicit-NLTE metal and LTE-remainder parts."""
    from wd_spectra.hot_nlte import population_arrays, with_departures
    from wd_spectra.light_metal_nlte import ReducedLightMetalLevelState
    setup = capture_setup(arguments.runner)
    model = setup.model
    with np.load(arguments.resume_host, allow_pickle=False) as saved:
        atmosphere = model.rebuild_atmosphere(setup.host.atmosphere, saved['temperature'])
        reference = model._rate_state(atmosphere)
        host_populations = with_departures(reference, saved['population'] / saved['lte_population'])
    departures = {}
    with np.load(arguments.resume_metals, allow_pickle=False) as saved:
        for e in setup.abundances:
            keys = tuple((str(k[0]), int(k[1]), int(k[2])) for k in saved[f'{e}_level_key'])
            n, lte = saved[f'{e}_population_density'], saved[f'{e}_lte_population_density']
            departures[e] = {k: n[i] / np.maximum(lte[i], 1e-300) for i, k in enumerate(keys)}
    wave = structure_grid(setup)
    host = model.transfer_coefficients(atmosphere, wave, host_populations)
    zero = np.zeros_like(host.true_absorption)
    reference_metals = fixed_electron_metal_reference(atmosphere, setup.database, setup.abundances)
    selection = {e: _atom_selection(setup.database, e, setup.counts[e]) for e in setup.abundances}
    explicit = trace_metal_coefficients(atmosphere, wave, NLTETransferCoefficients(wave, zero, zero, zero, {}),
        setup.database, setup.photo, reference_metals, selection, departures, setup.abundances,
        thresholds=setup.thresholds)
    ra, rj = remainder_terms(setup, atmosphere, wave)
    total = NLTETransferCoefficients(wave, host.true_absorption + explicit.true_absorption + ra,
        host.thermal_emissivity + explicit.thermal_emissivity + rj, host.scattering, {})
    _, field, _ = transfer_field(atmosphere, total, n_angle=model.n_angle)
    J = field.mean_intensity
    width = 4.0 * np.pi * mass_width(atmosphere.column_mass)
    def cell(kappa, eta):
        return width * trapezoid((kappa * J - eta)[:, :-1], wave, axis=0)
    emission = width * trapezoid(total.thermal_emissivity[:, :-1], wave, axis=0)
    parts = {'H/He': cell(host.true_absorption, host.thermal_emissivity),
             'explicit NLTE metals': cell(explicit.true_absorption, explicit.thermal_emissivity),
             'LTE Fe/Ni remainder': cell(ra, rj)}
    report = dict(column_mass=atmosphere.column_mass[:-1].tolist(), temperature=atmosphere.temperature[:-1].tolist(),
                  total_emission=emission.tolist(), **{k: (v / emission).tolist() for k, v in parts.items()})
    out = Path(arguments.output); out.mkdir(parents=True, exist_ok=False)
    (out / 'heating-decomposition.json').write_text(json.dumps(report, indent=1) + '\n')
    print('cell  column_mass  T      ' + '  '.join(f'{k:>22s}' for k in parts) + '   sum (relative to total emission)')
    for d in range(atmosphere.n_depth - 1):
        values = [parts[k][d] / emission[d] for k in parts]
        print(f'{d:3d}  {atmosphere.column_mass[d]:.2e}  {atmosphere.temperature[d]:7.0f} '
              + '  '.join(f'{v:+22.3e}' for v in values) + f'   {sum(values):+.3e}', flush=True)


def synthesize(arguments):
    """Emergent spectrum on a saved structure with frozen H/He and metal departures (no new solves)."""
    from wd_spectra.hot_nlte import with_departures
    setup = capture_setup(arguments.runner)
    model, host = setup.model, setup.host
    with np.load(arguments.resume_host, allow_pickle=False) as saved:
        temperature = np.array(saved['temperature'])
        population, lte = saved['population'], saved['lte_population']
    if arguments.keep_host_temperature_above_tau is not None:
        tau = np.asarray(host.atmosphere.rosseland_optical_depth)
        outer = tau < arguments.keep_host_temperature_above_tau
        temperature[outer] = host.atmosphere.temperature[outer]
    atmosphere = model.rebuild_atmosphere(host.atmosphere, temperature)
    reference = model._rate_state(atmosphere)
    host_populations = with_departures(reference, population / lte)
    departures = {}
    with np.load(arguments.resume_metals, allow_pickle=False) as saved:
        for e in setup.abundances:
            keys = tuple((str(k[0]), int(k[1]), int(k[2])) for k in saved[f'{e}_level_key'])
            n, l = saved[f'{e}_population_density'], saved[f'{e}_lte_population_density']
            departures[e] = {k: n[i] / np.maximum(l[i], 1e-300) for i, k in enumerate(keys)}
    with np.load(arguments.wavelength_from, allow_pickle=False) as saved:
        wave = np.array(saved['wavelength'])
    start = time.monotonic()
    coefficients = combined_coefficients(setup, atmosphere, host_populations, wave, departures)
    _, field, closure = transfer_field(atmosphere, coefficients, n_angle=model.n_angle)
    out = Path(arguments.output); out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'spectrum.npz', wavelength=wave, flux=field.interface_flux[:, 0],
             temperature=atmosphere.temperature, column_mass=atmosphere.column_mass)
    LOGGER.info('synthesis on %d wavelengths %.0f s, closure %.1e', len(wave), time.monotonic() - start, closure)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(message)s')
    # Driver options come before '--'; the runner's arguments follow it.
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('audit', 'relax', 'heating', 'synthesize'))
    parser.add_argument('--output', required=True, help='new directory for this driver')
    parser.add_argument('--outer-iterations', type=int, default=8)
    parser.add_argument('--inner-iterations', type=int, default=12)
    parser.add_argument('--maximum-step', type=float, default=0.1)
    parser.add_argument('--probe-step', type=float, default=0.02)
    parser.add_argument('--local-maximum-tau', type=float, default=1e-2)
    parser.add_argument('--local-energy-minimum-tau', type=float, default=1e-5)
    parser.add_argument('--flux-tolerance', type=float, default=3e-3)
    parser.add_argument('--local-energy-tolerance', type=float, default=3e-3)
    parser.add_argument('--stationary-step', type=float, default=2e-3)
    parser.add_argument('--host-iterations', type=int, default=20)
    parser.add_argument('--metal-iterations', type=int, default=30)
    parser.add_argument('--metal-tolerance', type=float, default=1e-3)
    parser.add_argument('--resume-temperature', type=Path, default=None,
                        help='state-temperature.npz from an earlier relax run (skips its first T loop)')
    parser.add_argument('--resume-host', type=Path, default=None,
                        help='host-outer-NN.npz (temperature + H/He populations; skips first H/He solve)')
    parser.add_argument('--wavelength-from', type=Path, default=None, help='spectrum.npz whose wavelength grid to use')
    parser.add_argument('--keep-host-temperature-above-tau', type=float, default=None,
                        help='synthesize: restore the H/He-host temperature where tau_Ross is below this')
    parser.add_argument('--resume-metals', type=Path, default=None,
                        help='metals-outer-NN.npz: frozen metal populations to start from')
    parser.add_argument('--resume-run-temperature-loop', action='store_true',
                        help='with --resume-host, still run the first temperature loop')
    arguments = parser.parse_args(argv[:split])
    arguments.runner = argv[split + 1:]
    {'audit': audit, 'relax': relax, 'heating': heating, 'synthesize': synthesize}[arguments.command](arguments)


if __name__ == '__main__':
    main()
