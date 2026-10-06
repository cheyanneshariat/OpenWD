#!/usr/bin/env python3
"""He I ionization/recombination budget of a coupled H/He NLTE state.

    python research/sdb_helium_rate_budget.py STATE.npz --teff 29890 --logg 5.46 \
        --ratio 2.88 --depths 12 13 14 15 16 20

Rebuilds the coupled helium rate matrix used by the full-NLTE residual, once
with the formal-solution radiation field and once with all mean intensities
set to zero (collisions plus spontaneous emission only), and reports, per
depth, which He I levels carry the He I <-> He II exchange and how much of it
is radiative.  Diagnostic only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.helium_nlte import HELIUM_I_14_LABEL
from wd_spectra.hot_nlte import population_arrays, transfer_field
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra._hot_structure import HotEquations

from run_hot_trace_from_checkpoint import load_seed

N_NEUTRAL = 14


def zeroed(fields):
    return {key: np.zeros_like(value) for key, value in fields.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('state', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--depths', type=int, nargs='+', default=[12, 13, 14, 15, 16, 20])
    args = parser.parse_args()

    config = DAOConfig(effective_temperature=args.teff, logg=args.logg, log_hydrogen_to_helium=args.ratio)
    model, seed, populations = load_seed(args.state, config, ModelData.default(args.data_root))
    equations = HotEquations(seed, model, structure_wavelength(config, numerical_resolution(config.quality).n_continuum),
                             nlte_fraction=1.)
    x = equations.initial_state(populations)
    equations.residual(x)                      # prepares the rate and line-average caches
    cache = equations.rate_cache
    atmosphere = cache.atmosphere
    current = equations.populations(x, equations.prepare(x[:equations.nd])[1])
    coefficients = model.transfer_coefficients(atmosphere, equations.wave, current)
    _, field, _ = transfer_field(atmosphere, coefficients, n_angle=model.n_angle, check_source=False)
    mean = field.mean_intensity
    neutral, ion, _ = equations.profile_cache.fields(mean)
    full = cache.rate_matrix(mean, neutral, ion)
    quiet = cache.rate_matrix(np.zeros_like(mean), zeroed(neutral), zeroed(ion))

    actual, _ = population_arrays(current)
    helium = actual[:, :full.shape[1]]           # He I, He II levels, He III
    total_helium = helium.sum(axis=1)
    ion_start = N_NEUTRAL
    for depth in args.depths:
        n = helium[depth]
        rates, collisional = full[depth], quiet[depth]
        ionization = n[:N_NEUTRAL, None] * rates[:N_NEUTRAL, ion_start:]
        recombination = n[ion_start:, None] * rates[ion_start:, :N_NEUTRAL]
        ionization_collisional = n[:N_NEUTRAL, None] * collisional[:N_NEUTRAL, ion_start:]
        recombination_collisional = n[ion_start:, None] * collisional[ion_start:, :N_NEUTRAL]
        total_ionization = ionization.sum()
        total_recombination = recombination.sum()
        print(f'\ndepth {depth}: T={atmosphere.temperature[depth]:.0f} K, ne={atmosphere.electron_density[depth]:.2e}, '
              f'He I/He={n[:N_NEUTRAL].sum() / total_helium[depth]:.2e}')
        print(f'  He I -> He II {total_ionization:.3e} (collisional {ionization_collisional.sum() / total_ionization:.2%}); '
              f'He II -> He I {total_recombination:.3e} (collisional+spontaneous '
              f'{recombination_collisional.sum() / total_recombination:.2%}); net/total '
              f'{(total_ionization - total_recombination) / total_ionization:+.2e}')
        per_level = ionization.sum(axis=1)
        per_level_recombination = recombination.sum(axis=0)
        print('  level           pop frac   ionization share   recomb. share   ionization rate/s  (radiative share)')
        for level in np.argsort(-per_level)[:6]:
            rate = rates[level, ion_start:].sum()
            radiative = 1 - collisional[level, ion_start:].sum() / max(rate, np.finfo(float).tiny)
            print(f'  {HELIUM_I_14_LABEL[level]:14s} {n[level] / n[:N_NEUTRAL].sum():9.2e} {per_level[level] / total_ionization:14.2%} '
                  f'{per_level_recombination[level] / total_recombination:15.2%} {rate:17.3e}  ({radiative:.1%})')
        # Bound-bound exchange between He I ground and the excited He I levels.
        up = n[0] * rates[0, 1:N_NEUTRAL]
        down = n[1:N_NEUTRAL] * rates[1:N_NEUTRAL, 0]
        quiet_up = n[0] * collisional[0, 1:N_NEUTRAL]
        print(f'  ground -> excited {up.sum():.3e} (collisional {quiet_up.sum() / up.sum():.2%}); '
              f'excited -> ground {down.sum():.3e}; net/total {(up.sum() - down.sum()) / up.sum():+.2e}')
        for level in np.argsort(-up)[:3]:
            print(f'    1s2 -> {HELIUM_I_14_LABEL[level + 1]:12s} up {up[level]:.3e} down {down[level]:.3e} '
                  f'net/up {(up[level] - down[level]) / up[level]:+.2e}')


if __name__ == '__main__':
    main()
