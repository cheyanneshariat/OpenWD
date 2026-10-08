#!/usr/bin/env python3
"""Hybrid sdB model: NLTE H/He populations on a fixed LTE structure (ADS-style).

    python research/sdb_hybrid_nlte.py results/sdb/hd4539-lte-hhe-40 \
        --output results/sdb/hd4539-hybrid-hhe-40

Step 2 of the sdB build-up.  The LTE DAB structure (T, P_gas on column mass)
is kept fixed.  The restricted hot H/He NLTE atom (He I 14 levels, He II 32,
H 8) is solved by the fixed-temperature population iteration (statistical
equilibrium with formal-solution radiation fields, Anderson acceleration).
The electron density stays the LTE H/He EOS value at the fixed T, P.

Writes spectrum.npz (NLTE populations) and spectrum-lte.npz (the same
synthesis code with J = B rates, i.e. LTE populations), both in vacuum
wavelengths, so the NLTE effect is isolated from synthesis differences.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.helium_collisions import read_tlusty_helium_collision_data
from wd_spectra.helium_i_atom import read_tlusty_helium_i_atom
from wd_spectra.hot_nlte import population_arrays, transfer_field, with_departures
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import _model_from_config, structure_wavelength
from wd_spectra._hot_structure import solve

from run_hot_public_diagnostic import save_state


def synthesize(model, atmosphere, populations, wavelength, chunk=8000):
    flux = np.empty_like(wavelength)
    for start in range(0, wavelength.size, chunk):
        piece = wavelength[start:start + chunk]
        coefficients = model.transfer_coefficients(atmosphere, piece, populations)
        _, field, _ = transfer_field(atmosphere, coefficients, n_angle=model.n_angle)
        flux[start:start + chunk] = field.interface_flux[:, 0]
    return flux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lte_model', type=Path, help='directory written by sdb_lte_hhe.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--population-iterations', type=int, nargs='+', default=[300],
                        help='iteration budget of each successive population chunk; a spectrum is '
                             'written after every chunk so drift during a plateau can be measured')
    parser.add_argument('--hydrogen-levels', type=int, default=8)
    parser.add_argument('--helium-i-atom', choices=['14', '24'], default='14',
                        help='TLUSTY He I atom: 14 terms (he1_14lev.dat) or 24 terms (he1.dat)')
    parser.add_argument('--helium-conservation-row', choices=['last', 'dominant'], default='dominant',
                        help='helium equation replaced by particle conservation (dominant: most populous state)')
    parser.add_argument('--accelerated-lambda', action='store_true',
                        help='precondition the line rates with the diagonal approximate lambda operator (MALI)')
    parser.add_argument('--initial-populations', type=Path,
                        help='populations.npz from an earlier run on the same structure (skips the LTE start)')
    parser.add_argument('--newton-iterations', type=int, default=0,
                        help='after the fixed-point chunks, polish with the fixed-temperature coupled Newton solver')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)

    summary = json.loads((args.lte_model / 'run-summary.json').read_text())['config']
    saved = np.load(args.lte_model / 'atmosphere.npz', allow_pickle=False)
    config = DAOConfig(effective_temperature=summary['teff'], logg=summary['logg'],
                       log_hydrogen_to_helium=-summary['log_he_h'],
                       population_maximum_iterations=max(args.population_iterations[0], 1),
                       maximum_hydrogen_level=args.hydrogen_levels)
    data = ModelData.default(args.data_root)
    model = _model_from_config(config, data)
    model = replace(model, helium_conservation_row=args.helium_conservation_row)
    if args.helium_i_atom == '24':
        atom_file = data.tlusty_atoms / 'he1.dat'
        model = replace(model, helium_i_atom=read_tlusty_helium_i_atom(atom_file),
                        helium_i_collision_data=read_tlusty_helium_collision_data(data.tlusty_source, atom_file))
    template = gray_hydrogen_helium_atmosphere(config.effective_temperature, config.logg,
                                               config.log_hydrogen_to_helium, n_depth=len(saved['temperature']))
    template = replace(template, column_mass=saved['column_mass'], gas_pressure=saved['gas_pressure'],
                       rosseland_optical_depth=saved['rosseland_optical_depth'])
    atmosphere = model.rebuild_atmosphere(template, saved['temperature'])
    electron_ratio = atmosphere.electron_density / saved['electron_density']
    print(f'EOS check vs LTE model: n_e ratio {electron_ratio.min():.4f}..{electron_ratio.max():.4f}', flush=True)

    start = time.monotonic()
    lte_state = model._rate_state(atmosphere)             # J = B: LTE populations
    wavelength = np.arange(3000.0, 9200.0, 0.05)
    nlte_state, previous_flux, total_iterations = lte_state, None, 0
    if args.initial_populations is not None:
        with np.load(args.initial_populations) as initial:
            np.testing.assert_allclose(initial['temperature'], atmosphere.temperature, rtol=1e-12)
            _, expected = population_arrays(lte_state)
            np.testing.assert_allclose(initial['lte_population'], expected, rtol=1e-10)
            nlte_state = with_departures(lte_state, initial['population'] / initial['lte_population'])
        previous_flux = synthesize(model, atmosphere, nlte_state, wavelength)
    for chunk, budget in enumerate(args.population_iterations):
        if budget <= 0:
            continue
        nlte_state = replace(model, population_maximum_iterations=budget).solve_populations(
            atmosphere, nlte_state, accelerated_lambda=args.accelerated_lambda)
        total_iterations += nlte_state.iterations
        flux_nlte = synthesize(model, atmosphere, nlte_state, wavelength)
        np.savez(args.output / f'spectrum-chunk{chunk}.npz', wavelength_vacuum=wavelength, flux=flux_nlte)
        drift = (None if previous_flux is None else
                 float(np.max(np.abs(flux_nlte / previous_flux - 1))))
        print(f'chunk {chunk}: converged={nlte_state.converged} iterations={total_iterations} '
              f'change={nlte_state.maximum_relative_population_change:.3g} '
              f'max spectral drift since previous chunk={drift} ({time.monotonic() - start:.0f} s)', flush=True)
        previous_flux = flux_nlte
        if nlte_state.converged:
            break
    if args.newton_iterations:
        answer = solve(atmosphere, model, structure_wavelength(config, numerical_resolution(config.quality).n_continuum),
                       populations=nlte_state, fixed_temperature=True, maximum_iterations=args.newton_iterations)
        nlte_state = answer.population_state
        flux_nlte = synthesize(model, atmosphere, nlte_state, wavelength)
        drift = None if previous_flux is None else float(np.max(np.abs(flux_nlte / previous_flux - 1)))
        diagnostics = answer.atmosphere.metadata
        print(f'newton polish: populations converged={nlte_state.converged} '
              f'change={nlte_state.maximum_relative_population_change:.3g} '
              f'max spectral drift={drift} nonlinear={diagnostics.get("nonlinear_solver")} '
              f'({time.monotonic() - start:.0f} s)', flush=True)
        previous_flux = flux_nlte
    elapsed = time.monotonic() - start
    save_state(args.output / 'populations.npz', atmosphere, nlte_state, model)
    np.savez(args.output / 'spectrum.npz', wavelength_vacuum=wavelength, flux=flux_nlte)
    flux_lte = synthesize(model, atmosphere, lte_state, wavelength)
    np.savez(args.output / 'spectrum-lte.npz', wavelength_vacuum=wavelength, flux=flux_lte)

    actual, reference = population_arrays(nlte_state)
    departures = actual / reference
    (args.output / 'run-summary.json').write_text(json.dumps(dict(
        lte_model=str(args.lte_model), config=summary, populations_converged=bool(nlte_state.converged),
        population_iterations=int(total_iterations), hydrogen_levels=args.hydrogen_levels,
        accelerated_lambda=args.accelerated_lambda, helium_i_atom=args.helium_i_atom,
        helium_conservation_row=args.helium_conservation_row,
        population_change=float(nlte_state.maximum_relative_population_change),
        population_seconds=elapsed, total_seconds=time.monotonic() - start,
        departure_range_helium_i_ground=[float(departures[:, 0].min()), float(departures[:, 0].max())],
        departure_range_hydrogen_n2=[float(departures[:, 48].min()), float(departures[:, 48].max())]),
        indent=2) + '\n')
    print(f'done in {time.monotonic() - start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
