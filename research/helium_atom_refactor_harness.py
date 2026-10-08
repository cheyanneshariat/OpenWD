#!/usr/bin/env python3
"""Bit-for-bit guard for the He I atom parameterization.

    python research/helium_atom_refactor_harness.py record  results/sdb/he-atom-reference.npz
    python research/helium_atom_refactor_harness.py compare results/sdb/he-atom-reference.npz

``record`` evaluates every helium/hot-NLTE entry point touched by the
parameterization with the default (TLUSTY 14-term) atom and stores the arrays;
``compare`` recomputes them and requires exact equality.
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from wd_spectra import DAOConfig, DOConfig
from wd_spectra import helium_nlte as he
from wd_spectra.atmosphere import gray_helium_atmosphere, gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import population_arrays
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra._hot_rates import PreparedHeliumRates


def arrays():
    data = ModelData.default()
    out = {}
    dao = _model_from_config(DAOConfig(effective_temperature=50000., logg=8., log_hydrogen_to_helium=2.,
                                       maximum_helium_ii_level=8, population_maximum_iterations=6), data)
    do = _model_from_config(DOConfig(effective_temperature=60000., logg=8., maximum_helium_ii_level=8,
                                     population_maximum_iterations=6), data)
    sdb = _model_from_config(DAOConfig(effective_temperature=30000., logg=5.5, log_hydrogen_to_helium=2.5,
                                       maximum_helium_ii_level=8, maximum_hydrogen_level=10,
                                       population_maximum_iterations=6), data)
    atmospheres = {
        'dao': dao.rebuild_atmosphere(gray_hydrogen_helium_atmosphere(50000., 8., 2., n_depth=8),
                                      gray_hydrogen_helium_atmosphere(50000., 8., 2., n_depth=8).temperature),
        'do': gray_helium_atmosphere(60000., 8., n_depth=8),
        'sdb': sdb.rebuild_atmosphere(gray_hydrogen_helium_atmosphere(30000., 5.5, 2.5, n_depth=8),
                                      gray_hydrogen_helium_atmosphere(30000., 5.5, 2.5, n_depth=8).temperature),
    }
    models = {'dao': dao, 'do': do, 'sdb': sdb}
    frequency = np.geomspace(3e13, 3e16, 400)
    for term in range(14):
        out[f'photo_{term}'] = he.neutral_helium_term_photoionization_cross_section(term, frequency)
    out['neutral_grid'] = he.default_neutral_helium_continuum_wavelength()
    for name, atmosphere in atmospheres.items():
        model = models[name]
        for index, value in enumerate(he._neutral_helium_reference_populations(atmosphere)):
            out[f'{name}_reference_{index}'] = value
        lte = model._rate_state(atmosphere)
        out[f'{name}_lte'] = population_arrays(lte)[0]
        wave = np.unique(np.concatenate([he.default_neutral_helium_continuum_wavelength(),
                                         he.default_helium_ii_continuum_wavelength(8)]))
        half_planck = 0.5 * planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
        state = he.solve_coupled_helium_statistical_equilibrium(
            atmosphere, model.collision_data, maximum_helium_ii_level=8,
            helium_i_collision_data=model.helium_i_collision_data,
            neutral_continuum_wavelength_angstrom=wave, neutral_continuum_mean_intensity_lambda=half_planck,
            helium_ii_continuum_wavelength_angstrom=wave, helium_ii_continuum_mean_intensity_lambda=half_planck)
        out[f'{name}_coupled_neutral'] = state.neutral_population_density
        out[f'{name}_coupled_ion'] = state.singly_ionized_population_density
        out[f'{name}_coupled_iii'] = state.doubly_ionized_he_density
        groups = model._line_problems(atmosphere)
        for key in sorted(groups[0]):
            out[f'{name}_line_{key[0]}_{key[1]}'] = np.sum([p.lte_line_opacity for p in groups[0][key]], axis=0)
        populations = model.solve_populations(atmosphere)
        out[f'{name}_populations'] = population_arrays(populations)[0]
        if model.log_hydrogen_to_helium is not None:
            out[f'{name}_populations_mali'] = population_arrays(
                model.solve_populations(atmosphere, accelerated_lambda=True))[0]
        synthesis = np.geomspace(300., 9000., 600)
        coefficients = model.transfer_coefficients(atmosphere, synthesis, populations)
        out[f'{name}_absorption'] = coefficients.true_absorption
        out[f'{name}_emissivity'] = coefficients.thermal_emissivity
        prepared = PreparedHeliumRates(model, atmosphere, wave, groups)
        zero = np.zeros((len(wave), atmosphere.n_depth))
        out[f'{name}_prepared'] = prepared.rate_matrix(
            half_planck, {k: np.ones(atmosphere.n_depth) * 1e-6 for k in groups[0]},
            {k: np.ones(atmosphere.n_depth) * 1e-6 for k in groups[1]})
    return out


def main():
    mode, path = sys.argv[1], sys.argv[2]
    current = arrays()
    if mode == 'record':
        np.savez_compressed(path, **current)
        print(f'recorded {len(current)} arrays to {path}')
        return
    reference = np.load(path)
    failures = [k for k in reference.files if k not in current or current[k].shape != reference[k].shape
                or not np.array_equal(current[k], reference[k], equal_nan=True)]
    extra = sorted(set(current) - set(reference.files))
    for key in failures[:20]:
        if key in current and current[key].shape == reference[key].shape:
            diff = np.nanmax(np.abs(current[key] / np.where(reference[key] == 0, 1, reference[key]) - 1))
            print(f'MISMATCH {key}: max relative {diff:.3e}')
        else:
            print(f'MISMATCH {key}: missing or shape changed')
    print(f'{len(reference.files) - len(failures)}/{len(reference.files)} arrays bit-identical; '
          f'{len(failures)} differ; extra keys: {extra[:5]}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
