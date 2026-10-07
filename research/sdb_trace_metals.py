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
from wd_spectra.hot_trace_metals import fixed_electron_metal_reference, solve_hot_trace_metals
from wd_spectra.light_metal_nlte import hot_metal_line_nlte_coefficients, read_tlusty_photoionization_threshold_data
from wd_spectra.metals import read_pg1159_atomic_database
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config

import hot_daz_data as bundled
from hot_trace_collisions import carbon_collisions
from hot_trace_oxygen_collisions import oxygen_collisions

# Upper levels of C II 4267 (4f, energy rank 29), 6578/83 (3p), 3920 (4s) and
# C III 4647-51 (3p 3P), 5696 (3d 1D), 4069/4187 (5g) lie inside these Stout
# energy-ordered fine-structure counts; 200 C II levels include the 2s2p3p
# quartets of C II 5133-45 (absent with 60 levels).
# Other elements: compact atoms reaching the upper levels of the standard sdB
# optical lines (Stout energy ranks): N II 3s/3p/3d <= 40, N III <= 19,
# O II 3s/3p/3d <= 39 (4649, 4072/76, 4414/17, 3919), O III <= 34,
# Si III 4s/4p <= 20 and 4p'/4d (4813-29) <= 62, Si IV <= 7, S III 4s/4p <= 36.
# Iron is not in this table: its optical Fe III lines (4p, rank ~406) need a
# 400-level Fe III atom with 7e4 lines (60 s per iteration), while compact
# Fe III/IV atoms hold only forbidden 3d^n lines.  Iron (and other
# --lte-abundance elements) instead adds LTE line opacity to the fixed
# background, as in the ADS recipe (LTE metals in structure and synthesis).
ELEMENT_LEVELS = {
    'C': {1: 200, 2: 80, 3: 20, 4: 1},
    'N': {1: 60, 2: 40, 3: 10, 4: 1},
    'O': {1: 60, 2: 40, 3: 10, 4: 1},
    'Si': {1: 30, 2: 62, 3: 15, 4: 1},
    'S': {1: 30, 2: 45, 3: 10, 4: 1},
}
CARBON_LEVELS = ELEMENT_LEVELS['C']
LINES = {'C II 3920': 3920.68, 'C II 4267': 4267.26, 'C II 6578': 6578.05, 'C II 6583': 6582.88,
         'C II 4074': 4074.52, 'C III 4647': 4647.42, 'C III 4650': 4650.25, 'C III 4651': 4651.47,
         'C III 5696': 5695.92, 'C III 4069': 4068.92, 'C III 4187': 4186.90,
         'N II 3995': 3994.99, 'N II 4630': 4630.54, 'N II 5679': 5679.56, 'N III 4097': 4097.36,
         'O II 4649': 4649.13, 'O II 4072': 4072.15, 'O II 4414': 4414.90, 'O II 3919': 3919.29,
         'Si III 4553': 4552.62, 'Si IV 4089': 4088.86, 'S III 4254': 4253.59, 'Fe III 4164': 4164.73,
         'Fe III 5156': 5156.11}


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
    parser.add_argument('--maximum-iterations', type=int, default=120,
                        help='verification grid: up to 73 iterations with the overlap exclusion')
    parser.add_argument('--step', type=float, default=0.02,
                        help='uniform synthesis step (A) over 3700-7000 A; lines must not fall between samples')
    parser.add_argument('--lte-abundance', action='append', default=[],
                        help='ELEMENT=log10 N/N(H) added as LTE line opacity of all its ions (e.g. Fe=-4.8)')
    parser.add_argument('--lte-minimum-oscillator-strength', type=float, default=1e-4)
    parser.add_argument('--damping', type=float, default=0.5, help='solver mixing (solve_hot_trace_metals default 0.5)')
    parser.add_argument('--acceleration-depth', type=int, default=6, help='Anderson history (0: none)')
    parser.add_argument('--no-accelerated-lambda', action='store_true', help='plain Lambda iteration (diagnostic)')
    parser.add_argument('--mali-overlap-velocity', type=float, default=15.0,
                        help='km/s; no MALI for lines this close to another element\'s line (0: off). '
                             'Needed at 30 kK, log g 5.3 (S III 702.8 on O III 702.8)')
    parser.add_argument('--levels', action='append', default=[],
                        help='override an atom size, ELEMENT:CHARGE=COUNT (e.g. C:1=200)')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    abundances = dict((k, float(v)) for k, v in (item.split('=') for item in (args.abundance or ['C=-4.0'])))
    if set(abundances) - set(ELEMENT_LEVELS):
        raise SystemExit(f'configured elements: {sorted(ELEMENT_LEVELS)}')

    data = ModelData.default(args.data_root)
    model, atmosphere, state, summary = load_host(args.lte_model, args.hybrid_model, data)
    elements = tuple(abundances)
    database = read_pg1159_atomic_database(data.stout, elements=elements)
    counts = {e: dict(ELEMENT_LEVELS[e]) for e in elements}
    for item in args.levels:
        key, value = item.split('=')
        element, charge = key.split(':')
        counts[element][int(charge)] = int(value)
    # CHIANTI collision strengths where bundled (C II-IV, O III); the built-in
    # approximate prescription elsewhere.
    collisions, audit = {}, {}
    if 'C' in counts:
        found, found_audit = carbon_collisions(bundled.CHIANTI_CARBON, database, counts['C'], charges=(1, 2, 3))
        collisions.update(found); audit.update({f'C {k}': v for k, v in found_audit.items()})
    if 'O' in counts:
        found, found_audit = oxygen_collisions(bundled.CHIANTI_OXYGEN, database, counts['O'], charges=(2,))
        collisions.update(found); audit.update({f'O {k}': v for k, v in found_audit.items()})
    thresholds = {}
    if 'C' in counts:
        thresholds['C'] = {2: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c3.dat'),
                           3: read_tlusty_photoionization_threshold_data(data.tlusty_atoms / 'c4_35+2lev.dat')}
    # Uniform sampling: a fine grid only around a fixed line list misses any
    # other line (e.g. O II 4591) between coarse samples.
    wavelength = np.arange(3700., 7000., args.step)
    start = time.monotonic()
    history = []

    def progress(iteration, defect):
        history.append(dict(iteration=iteration, defect=float(defect), elapsed_seconds=time.monotonic() - start))
        print(json.dumps(history[-1]), flush=True)

    lte_abundances = dict((k, float(v)) for k, v in (item.split('=') for item in args.lte_abundance))
    if set(lte_abundances) & set(abundances):
        raise SystemExit('an element is either NLTE (--abundance) or LTE (--lte-abundance)')
    if lte_abundances:
        lte_database = read_pg1159_atomic_database(data.stout, elements=tuple(lte_abundances))
        # Stout lacks ionization energies above e.g. Al IV (Al IV -> V: 120 eV);
        # such a stage closes the Saha ladder (the higher ions are negligible
        # below ~50 kK).
        top = {e: min((ion.charge for ion in lte_database.ion_stages(e) if ion.ionization_energy_ev is None),
                      default=None) for e in lte_abundances}
        lte_database = replace(lte_database, ions={k: v for k, v in lte_database.ions.items()
                                                   if top[k[0]] is None or k[1] <= top[k[0]]},
                               _line_selection_cache={}, _unsold_hydrogen_coefficient_cache={})
        lte_reference = fixed_electron_metal_reference(atmosphere, lte_database, lte_abundances)
        unity = {(e, ion.charge): np.ones(atmosphere.n_depth)
                 for e in lte_abundances for ion in lte_database.ion_stages(e)}

    def background(wave):
        base = model.transfer_coefficients(atmosphere, wave, state)
        # The metal edges extend the grid to ~5 A, where the host emissivity
        # underflows and can carry a sign (seen: -2e-301 at 6.55 A).  Zero such
        # roundoff values; any negative above 1e-30 of the depth's largest
        # emissivity is a real error and is left for the solver to reject.
        emission = base.thermal_emissivity
        roundoff = (emission < 0) & (-emission < 1e-30 * np.max(np.abs(emission), axis=0))
        if roundoff.any():
            base = replace(base, thermal_emissivity=np.where(roundoff, 0.0, emission))
        if not lte_abundances:
            return base
        # Line opacity only: the bound-free edges of trace Fe lie in the EUV.
        absorption, emission = hot_metal_line_nlte_coefficients(
            atmosphere, wave, lte_database, lte_reference, unity, elements=tuple(lte_abundances),
            minimum_oscillator_strength=args.lte_minimum_oscillator_strength, maximum_lines=None,
            include_static_linear_stark=False, profile_block_tolerance=1e-4)
        return replace(base, true_absorption=base.true_absorption + absorption,
                       thermal_emissivity=base.thermal_emissivity + emission)

    result = solve_hot_trace_metals(
        atmosphere, background, abundances, wavelength,
        data=data, atomic_database=database, levels_per_charge=counts, photoionization_threshold_data=thresholds,
        collision_data=collisions, accelerated_lambda=not args.no_accelerated_lambda,
        damping=args.damping, acceleration_depth=args.acceleration_depth,
        mali_overlap_velocity=args.mali_overlap_velocity or None, maximum_iterations=args.maximum_iterations,
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
        lte_abundances=lte_abundances, lte_minimum_oscillator_strength=args.lte_minimum_oscillator_strength,
        levels_per_charge=counts, converged=bool(result.converged), iterations=int(result.iterations),
        population_defect=float(result.population_defect), elapsed_seconds=time.monotonic() - start,
        chianti_collision_pairs={k: v['selected_pairs'] for k, v in audit.items()}, history=history,
        element_population_defect_history=result.metadata.get('element_population_defect_history'),
        worst_population_defect=result.metadata.get('worst_population_defect')),
        indent=2, default=str) + '\n')
    print(f'converged={result.converged} iterations={result.iterations} defect={result.population_defect:.3g} '
          f'({time.monotonic() - start:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
