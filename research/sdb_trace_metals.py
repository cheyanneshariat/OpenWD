#!/usr/bin/env python3
"""NLTE trace metals (default: carbon, C II-IV) on a finished hybrid sdB H/He host.

    python research/sdb_trace_metals.py results/sdb/hd4539-lte-hhe-40 \
        results/sdb/hd4539-hybrid-dominant --abundance C=-4.0 \
        --output results/sdb/hd4539-carbon

The host is the LTE H/He structure plus its converged NLTE H/He populations
(sdb_lte_hhe.py + sdb_hybrid_nlte.py). It stays fixed: carbon is a trace
species solved by the hot-DAZ fixed-host NLTE machinery
(``solve_hot_trace_metals``), with these components:
- the Stout (CHIANTI-format) C II-IV levels and lines, using compact atoms
  that reach the upper levels of the main optical lines;
- Verner ground-state and Kramers excited-level photoionization, with
  TLUSTY/OP tables for C III and C IV;
- CHIANTI collision strengths for C II, C III and C IV;
- MALI line preconditioning.
Writes spectrum.npz (host+carbon and host-only), populations, run-summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import with_departures
from wd_spectra.hot_trace_metals import solve_hot_trace_metals
from wd_spectra.light_metal_nlte import read_tlusty_photoionization_threshold_data
from wd_spectra.metals import read_pg1159_atomic_database
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config

import hot_daz_data as bundled
from hot_trace_collisions import carbon_collisions

# Upper levels of C II 4267 (4f, energy rank 29), 6578/83 (3p), 3920 (4s) and
# C III 4647-51 (3p 3P), 5696 (3d 1D), 4069/4187 (4f/4d, ranks 34-42) lie well
# inside these Stout energy-ordered fine-structure counts.
CARBON_LEVELS = {1: 60, 2: 60, 3: 20, 4: 1}
CARBON_LINES = {'C II 3920': 3920.68, 'C II 4267': 4267.26, 'C II 6578': 6578.05, 'C II 6583': 6582.88,
                'C II 4074': 4074.52, 'C III 4647': 4647.42, 'C III 4650': 4650.25, 'C III 4651': 4651.47,
                'C III 5696': 5695.92, 'C III 4069': 4068.92, 'C III 4187': 4186.90}


def load_host(lte_dir, hybrid_dir, data):
    summary = json.loads((lte_dir / 'run-summary.json').read_text())['config']
    saved = np.load(lte_dir / 'atmosphere.npz', allow_pickle=False)
    stored = np.load(hybrid_dir / 'populations.npz')
    config = DAOConfig(effective_temperature=summary['teff'], logg=summary['logg'],
                       log_hydrogen_to_helium=-summary['log_he_h'],
                       maximum_hydrogen_level=int(stored['maximum_hydrogen_level']))
    model = replace(_model_from_config(config, data), helium_conservation_row='dominant')
    template = gray_hydrogen_helium_atmosphere(config.effective_temperature, config.logg,
                                               config.log_hydrogen_to_helium, n_depth=len(saved['temperature']))
    template = replace(template, column_mass=saved['column_mass'], gas_pressure=saved['gas_pressure'],
                       rosseland_optical_depth=saved['rosseland_optical_depth'])
    atmosphere = model.rebuild_atmosphere(template, saved['temperature'])
    np.testing.assert_allclose(stored['temperature'], atmosphere.temperature, rtol=1e-12)
    state = with_departures(model._rate_state(atmosphere), stored['population'] / stored['lte_population'])
    return model, atmosphere, state, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lte_model', type=Path)
    parser.add_argument('hybrid_model', type=Path)
    parser.add_argument('--abundance', action='append', default=[], help='ELEMENT=log10 N/N(H); default C=-4.0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--maximum-iterations', type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    abundances = dict((k, float(v)) for k, v in (item.split('=') for item in (args.abundance or ['C=-4.0'])))
    if set(abundances) != {'C'}:
        raise SystemExit('this driver currently configures carbon only')

    data = ModelData.default(args.data_root)
    model, atmosphere, state, summary = load_host(args.lte_model, args.hybrid_model, data)
    database = read_pg1159_atomic_database(data.stout, elements=('C',))
    counts = {'C': dict(CARBON_LEVELS)}
    collisions, audit = carbon_collisions(bundled.CHIANTI_CARBON, database, counts['C'], charges=(1, 2, 3))
    thresholds = {'C': {2: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c3.dat'),
                        3: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c4_35+2lev.dat')}}
    wavelength = np.unique(np.concatenate(
        [np.arange(3600., 7200., 0.5)] + [np.arange(c - 12., c + 12.001, 0.01) for c in CARBON_LINES.values()]))
    start = time.monotonic()
    history = []

    def progress(iteration, defect):
        history.append(dict(iteration=iteration, defect=float(defect), elapsed_seconds=time.monotonic() - start))
        print(json.dumps(history[-1]), flush=True)

    result = solve_hot_trace_metals(
        atmosphere, lambda wave: model.transfer_coefficients(atmosphere, wave, state), abundances, wavelength,
        data=data, atomic_database=database, levels_per_charge=counts, photoionization_threshold_data=thresholds,
        collision_data=collisions, accelerated_lambda=True, maximum_iterations=args.maximum_iterations,
        n_angle=model.n_angle, require_convergence=False, iteration_callback=progress,
        # Solar-like sdB carbon is ~0.14% by mass: it changes the host's mean
        # molecular weight and electron density by <0.2%, negligible for the
        # fixed H/He structure (its opacity is the separate blanketing question).
        trace_mass_limit=1e-2)
    np.savez(args.output / 'spectrum.npz', wavelength_vacuum=result.spectrum.wavelength_angstrom,
             flux=result.spectrum.surface_flux_lambda, host_flux=result.background_spectrum.surface_flux_lambda)
    np.savez(args.output / 'populations.npz', **{f'{e}_{f}': getattr(s, f) for e, s in result.populations.items()
                                                for f in ('population_density', 'lte_population_density')})
    (args.output / 'run-summary.json').write_text(json.dumps(dict(
        host=dict(lte=str(args.lte_model), hybrid=str(args.hybrid_model), **summary), abundances=abundances,
        levels_per_charge=counts, converged=bool(result.converged), iterations=int(result.iterations),
        population_defect=float(result.population_defect), elapsed_seconds=time.monotonic() - start,
        chianti_collision_pairs={k: v['selected_pairs'] for k, v in audit.items()}, history=history),
        indent=2, default=str) + '\n')
    print(f'converged={result.converged} iterations={result.iterations} defect={result.population_defect:.3g} '
          f'({time.monotonic() - start:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
