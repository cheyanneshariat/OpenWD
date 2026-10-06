#!/usr/bin/env python3
"""G191-B2B benchmark: H/He NLTE cold start, then nine NLTE trace metals from LTE.

    python research/hot_daz_g191b2b.py --output DIR [--quality standard|production]

The H/He host is the public DAO cold start (T_eff = 52,500 K, log g = 7.53,
N(He)/N(H) = 1e-5). C, N, O, Al, Si, P, S, Fe and Ni are then solved in NLTE on
that fixed atmosphere from LTE populations, at the Preval et al. (2013)
abundances, with the bundled atomic data (wd_spectra/data/hot_daz):

* model atoms: Stout levels as below; Fe/Ni IV-VII keep all bound levels with
  the ground of Fe/Ni VIII on top, Kurucz measured-level UV and EUV lines;
* OP photoionization for C III-IV and O IV-VI, CHIANTI collisions for C and O,
  CHIANTI total recombination closing the truncated Fe/Ni atoms;
* Fe/Ni lines of stages below the explicit atoms keep LTE opacity at the host
  electron density; stages above the top carry none;
* line-weighted ALI with ion-wise Anderson acceleration, converged on the
  emergent-flux criterion (solver default tolerance 3e-3).

``standard`` (40 depths, 3 angles) is the canary configuration;
``production`` (80 depths, 4 angles) is the published model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig, save_model_result
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import (_atom_selection, fixed_electron_metal_reference,
                                         solve_hot_trace_metals)
from wd_spectra.light_metal_nlte import (hot_metal_line_nlte_coefficients,
                                         light_metal_bound_free_nlte_coefficients,
                                         read_tlusty_photoionization_threshold_data,
                                         reduced_light_metal_wavelength)
from wd_spectra.metals import EV_TO_WAVENUMBER
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config, compute_dao
from wd_spectra.nlte_core import NLTETransferCoefficients

import hot_daz_data as bundled
import wd_spectra.hot_trace_metals as trace
from hot_trace_acceleration import ElementPopulationAcceleration
from hot_trace_collisions import carbon_collisions
from hot_trace_composition import ABUNDANCES, load_composition_data
from hot_trace_oxygen_collisions import oxygen_collisions
from hot_trace_recombination import chianti_total_recombination

HOST = dict(effective_temperature=52500., logg=7.53, log_hydrogen_to_helium=5.,
            maximum_helium_ii_level=32, maximum_hydrogen_level=8)
LIGHT_COUNTS = {'C': {2: 240, 3: 243, 4: 10, 5: 1}, 'Si': {2: 50, 3: 40, 4: 30, 5: 1},
                'N': {2: 40, 3: 60, 4: 40, 5: 1}, 'O': {2: 100, 3: 150, 4: 120, 5: 60, 6: 1},
                'Al': {2: 40, 3: 20, 4: 1}, 'P': {2: 30, 3: 50, 4: 40, 5: 1},
                'S': {2: 30, 3: 60, 4: 40, 5: 30, 6: 1}}
IRON_GROUP = ('Fe', 'Ni')
TOP_CHARGE = 7
# Synthesis: 910-1990 A at 0.01 A plus 2 mA sampling around the diagnostics.
DIAGNOSTIC_LINES = (1238.821, 1242.804, 1338.615, 1343.514, 1854.716, 1862.790,
                    1117.977, 1128.008, 1062.662, 1072.974, 1409.453, 1306.624)


def host_config(quality='standard'):
    return DAOConfig(quality=quality, **HOST)


def model_atoms(database):
    counts = {e: dict(c) for e, c in LIGHT_COUNTS.items()}
    for e in IRON_GROUP:
        counts[e] = {q: int(sum(level.energy_wavenumber / EV_TO_WAVENUMBER
                                < database.ions[e, q].ionization_energy_ev - .1
                                for level in database.ions[e, q].levels))
                     for q in range(3, TOP_CHARGE)}
        counts[e][TOP_CHARGE] = 1
    return counts


def solve_metals(host, *, data=None, iterations=60, tolerance=None, profile_block_tolerance=1e-4,
                 state_callback=None):
    """Nine-element trace-metal NLTE on a converged DAO host, starting from LTE."""
    data = ModelData.default() if data is None else data
    database, photo, atomic_audit = load_composition_data(
        data, bundled.ATOMIC, iron_group=True, kurucz_positions=bundled.KURUCZ)
    counts = model_atoms(database)
    abundances = {e: float(np.log10(ABUNDANCES[e][0])) for e in counts}
    collisions, carbon_audit = carbon_collisions(bundled.CHIANTI_CARBON, database, counts['C'])
    oxygen, oxygen_audit = oxygen_collisions(bundled.CHIANTI_OXYGEN, database, counts['O'])
    collisions.update(oxygen)
    recombination = {e: chianti_total_recombination(bundled.CHIANTI_RECOMBINATION, e, sorted(counts[e])[1:])[0]
                     for e in IRON_GROUP}
    thresholds = {'C': {2: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c3.dat'),
                        3: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c4_35+2lev.dat')},
                  'O': {q: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / f'o{q + 1}.dat')
                        for q in (3, 4, 5)}}
    atmosphere = host.atmosphere
    model = _model_from_config(host.config, data)
    wave = np.unique(np.concatenate([host.spectrum.wavelength_angstrom, np.arange(910., 1990.001, .01)]
                                    + [np.arange(c - 1.5, c + 1.501, .002) for c in DIAGNOSTIC_LINES]))
    grid = np.unique(np.concatenate(
        [reduced_light_metal_wavelength(database, e, n, photoionization_threshold_data=thresholds.get(e))
         for e, n in counts.items()] + [np.arange(880., 1990., .02)]))

    # Fe/Ni opacity outside the explicit atoms: LTE lines/edges of the lower
    # stages only; stages at or above the top charge are represented by its
    # ground level and carry no remainder opacity.
    remainder = {e: abundances[e] for e in IRON_GROUP}
    reference = fixed_electron_metal_reference(atmosphere, database, remainder)
    unity = {(e, s.charge): np.ones(atmosphere.n_depth) for e in IRON_GROUP for s in database.ion_stages(e)}
    retained = {e: {(e, ion.charge, l.lower_index, l.upper_index) for ion in database.ion_stages(e)
                    if ion.charge < TOP_CHARGE for l in ion.transitions}
                - _atom_selection(database, e, counts[e])[0] for e in IRON_GROUP}

    def background(w):
        base = model.transfer_coefficients(atmosphere, w, host.population_state)
        absorption, emission = base.true_absorption.copy(), base.thermal_emissivity.copy()
        for e in IRON_GROUP:
            a, j = hot_metal_line_nlte_coefficients(
                atmosphere, w, database, reference, unity, elements=(e,), minimum_oscillator_strength=1e-4,
                maximum_lines=None, transition_keys=retained[e], include_static_linear_stark=False,
                profile_block_tolerance=0. if np.array_equal(w, wave) else profile_block_tolerance)
            bound = {s.charge: (0 if s.charge in sorted(counts[e])[:-1] or s.charge >= TOP_CHARGE else 1)
                     for s in database.ion_stages(e)}
            b, k = light_metal_bound_free_nlte_coefficients(
                atmosphere, w, database, reference, photo, unity, elements=(e,),
                levels_per_charge=bound, include_explicit_kramers=True)
            absorption += a + b
            emission += j + k
        return NLTETransferCoefficients(w, absorption, emission, base.scattering, {})

    accelerator = ElementPopulationAcceleration({f'{e}:{q}': n[q] for e, n in counts.items() for q in sorted(n)})
    ordinary = trace.population_update
    trace.population_update = accelerator
    try:
        result = solve_hot_trace_metals(
            atmosphere, background, abundances, wave, data=data, atomic_database=database,
            photoionization_database=photo, levels_per_charge=counts, population_wavelength=grid,
            photoionization_threshold_data=thresholds, collision_data=collisions,
            total_recombination=recombination, accelerated_lambda=True, convergence_criterion='flux',
            tolerance=tolerance, profile_block_tolerance=profile_block_tolerance,
            maximum_iterations=iterations, acceleration_depth=16, damping=.5, n_angle=model.n_angle,
            require_convergence=False, state_callback=state_callback)
    finally:
        trace.population_update = ordinary
    background_flux = transfer_field(atmosphere, background(wave), n_angle=model.n_angle,
                                     check_source=False)[1].interface_flux[:, 0]
    audit = dict(atomic=atomic_audit, carbon_collisions=carbon_audit, oxygen_collisions=oxygen_audit,
                 levels_per_charge={e: {str(q): n for q, n in c.items()} for e, c in counts.items()},
                 abundances=abundances, iron_group_top_charge=TOP_CHARGE,
                 acceleration=accelerator.diagnostics())
    return result, background_flux, audit


def run(output, *, quality='standard', iterations=60, tolerance=None):
    output = Path(output)
    output.mkdir(parents=True)
    data = ModelData.default()
    start = time.monotonic()
    host = compute_dao(host_config(quality), data=data)
    host_seconds = time.monotonic() - start
    save_model_result(host, output / 'host')
    if host.metadata.get('atmosphere_convergence_status') != 'converged':
        raise RuntimeError('the H/He host cold start did not converge')
    result, background_flux, audit = solve_metals(host, data=data, iterations=iterations, tolerance=tolerance)
    np.savez_compressed(output / 'spectrum.npz', wavelength=result.spectrum.wavelength_angstrom,
                        flux=result.spectrum.surface_flux_lambda, background_flux=background_flux)
    np.savez_compressed(output / 'populations.npz', **{f'{e}_{key}': getattr(s, key)
                        for e, s in result.populations.items()
                        for key in ('level_key', 'population_density', 'lte_population_density')})
    record = dict(quality=quality, host=HOST, host_seconds=host_seconds,
                  metal_seconds=time.monotonic() - start - host_seconds,
                  converged=result.converged, iterations=result.iterations,
                  flux_defect_history=result.metadata.get('flux_defect_history'),
                  metadata=result.metadata, audit=audit)
    (output / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality', choices=('standard', 'production'), default='standard')
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--tolerance', type=float, default=None)
    args = parser.parse_args()
    result, record = run(args.output, quality=args.quality, iterations=args.iterations, tolerance=args.tolerance)
    print(json.dumps({k: record[k] for k in ('converged', 'iterations', 'host_seconds', 'metal_seconds')}))


if __name__ == '__main__':
    main()
