#!/usr/bin/env python3
"""Check hybrid sdB populations against one unpreconditioned statistical-equilibrium update.

    python research/sdb_fixed_point_check.py results/sdb/hd4539-lte-hhe-40 \
        results/sdb/hd4539-hybrid-mali/populations.npz [OTHER/populations.npz ...]

For each saved population state (written by sdb_hybrid_nlte.py on the given
LTE structure) computes the radiation field, solves the plain rate equations
once and reports the largest relative population change, by species block.
With two or more states it also reports their mutual differences.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.helium_collisions import read_tlusty_helium_collision_data
from wd_spectra.helium_i_atom import read_tlusty_helium_i_atom
from wd_spectra.hot_nlte import population_arrays, with_departures
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config

from test_hot_mali import _population_grid_and_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lte_model', type=Path)
    parser.add_argument('states', type=Path, nargs='+')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--helium-i-atom', choices=['14', '24'], default='14')
    parser.add_argument('--helium-conservation-row', choices=['last', 'dominant'], default='dominant')
    args = parser.parse_args()
    summary = json.loads((args.lte_model / 'run-summary.json').read_text())['config']
    saved = np.load(args.lte_model / 'atmosphere.npz', allow_pickle=False)
    first = np.load(args.states[0])
    config = DAOConfig(effective_temperature=summary['teff'], logg=summary['logg'],
                       log_hydrogen_to_helium=-summary['log_he_h'],
                       maximum_hydrogen_level=int(first['maximum_hydrogen_level']))
    data = ModelData.default(args.data_root)
    model = replace(_model_from_config(config, data), helium_conservation_row=args.helium_conservation_row)
    if args.helium_i_atom == '24':
        atom_file = data.tlusty_atoms / 'he1.dat'
        model = replace(model, helium_i_atom=read_tlusty_helium_i_atom(atom_file),
                        helium_i_collision_data=read_tlusty_helium_collision_data(data.tlusty_source, atom_file))
    template = gray_hydrogen_helium_atmosphere(config.effective_temperature, config.logg,
                                               config.log_hydrogen_to_helium, n_depth=len(saved['temperature']))
    template = replace(template, column_mass=saved['column_mass'], gas_pressure=saved['gas_pressure'],
                       rosseland_optical_depth=saved['rosseland_optical_depth'])
    atmosphere = model.rebuild_atmosphere(template, saved['temperature'])
    lte = model._rate_state(atmosphere)
    n_neutral = 24 if args.helium_i_atom == '24' else 14
    nhe = n_neutral + 1 + model.maximum_helium_ii_level
    blocks = {'He I': slice(0, n_neutral), 'He II': slice(n_neutral, nhe - 1), 'He III': slice(nhe - 1, nhe),
              'H': slice(nhe, None)}
    states = []
    for path in args.states:
        with np.load(path) as stored:
            state = with_departures(lte, stored['population'] / stored['lte_population'])
        _, wave, _, mean, fields = _population_grid_and_fields(model, atmosphere, state)
        update = model._rate_state(atmosphere, wave, mean, *fields)
        current, reference = population_arrays(state)
        following, _ = population_arrays(update)
        change = np.abs(following - current) / np.maximum(following, 1e-12 * reference.sum(axis=1)[:, None])
        print(f'{path}: one plain update changes populations by ' +
              ', '.join(f'{name} {np.max(change[:, s]):.2e}' for name, s in blocks.items()), flush=True)
        states.append(current)
    for index in range(1, len(states)):
        difference = np.abs(states[index] / states[0] - 1)
        print(f'{args.states[index]} vs {args.states[0]}: max relative difference ' +
              ', '.join(f'{name} {np.max(difference[:, s]):.2e}' for name, s in blocks.items()))


if __name__ == '__main__':
    main()
