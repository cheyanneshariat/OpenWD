#!/usr/bin/env python3
"""Offline test: full-NLTE H/He Newton steps in dominant-ion-referenced helium coordinates.

    python research/sdb_reference_ion_newton_test.py STATE.npz JACOBIAN.npz --teff 29890 \
        --logg 5.46 --ratio 2.88 --output results/sdb/reference-ion-test

Diagnostic only; the package equations are used unchanged.  The production
unknowns are x_i = 0.1 ln(n_i / n_HeIII) for helium levels.  Where He II
(ground) outnumbers He III, this script works in the exactly equivalent
coordinates y_i = x_i - x_g = 0.1 ln(n_i / n_g) (g = He II ground) and
y_g = -x_g = 0.1 ln(n_HeIII / n_g), transforming residuals and the Jacobian
linearly (y = T x, J_y = T J_x T^-1).  The Newton step is the same linear
algebra; what changes is step control: the He III coordinates y_g get their
own cap, so a large He III correction no longer shrinks every other component,
and the backtracking merit excludes the He III rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.hot_nlte import population_arrays
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra.nonlinear import RecoverableEvaluationError
from wd_spectra._hot_structure import HotEquations

from run_hot_public_diagnostic import save_state
from run_hot_trace_from_checkpoint import load_seed

N_HELIUM_COORDINATES = 46          # He I (14) + He II (32) levels, relative to He III
HELIUM_II_GROUND = 14
DIAGNOSTIC_KEYS = ('maximum_continuation_population_residual', 'maximum_all_depth_total_flux_residual',
                   'maximum_relative_cell_energy_balance_residual', 'nlte_maximum_relative_population_change')


def transform(nd, ni, referenced_depths):
    """Return T and T^-1 for the full unknown vector (temperatures first)."""
    size = nd + nd * ni
    forward = np.eye(size)
    for depth in referenced_depths:
        base = nd + depth * ni
        g = base + HELIUM_II_GROUND
        for i in range(base, base + N_HELIUM_COORDINATES):
            if i != g:
                forward[i, g] = -1.0
        forward[g, g] = -1.0
    return forward, np.linalg.inv(forward)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('state', type=Path)
    parser.add_argument('jacobian', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--rcond', type=float, default=1e-12, help='SVD truncation (default: full Newton)')
    parser.add_argument('--radius', type=float, default=0.12, help='max-norm cap on non-He III coordinates')
    parser.add_argument('--helium-iii-radius', type=float, default=0.3, help='separate cap on He III coordinates')
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--refresh', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)

    config = DAOConfig(effective_temperature=args.teff, logg=args.logg, log_hydrogen_to_helium=args.ratio)
    model, seed, populations = load_seed(args.state, config, ModelData.default(args.data_root))
    equations = HotEquations(seed, model, structure_wavelength(config, numerical_resolution(config.quality).n_continuum),
                             nlte_fraction=1.)
    nd = equations.nd
    x = equations.initial_state(populations)
    ni = (x.size - nd) // nd
    actual, _ = population_arrays(populations)
    referenced = [d for d in range(nd) if actual[d, HELIUM_II_GROUND] > actual[d, N_HELIUM_COORDINATES]]
    forward, inverse = transform(nd, ni, referenced)
    helium_iii_rows = np.array([nd + d * ni + HELIUM_II_GROUND for d in referenced], dtype=int)
    others = np.setdiff1d(np.arange(x.size), helium_iii_rows)
    print(f'He II-referenced depths: {referenced[0]}..{referenced[-1]} ({len(referenced)} of {nd})', flush=True)

    saved = np.load(args.jacobian)
    print(f'state vs saved Jacobian x: max |dx| = {np.max(np.abs(x - saved["x"])):.2e}', flush=True)
    jacobian_x = saved['jacobian']
    evaluation = equations.residual(x)

    def summary(residual_x):
        r = forward @ residual_x
        return r, float(r[others] @ r[others]), float(np.max(np.abs(r[others][nd:]))), float(np.max(np.abs(r[helium_iii_rows])))

    start = time.monotonic()
    for iteration in range(1, args.iterations + 1):
        if iteration > 1 and (iteration - 1) % args.refresh == 0:
            evaluation = equations.evaluate(x, True)
            jacobian_x = evaluation.jacobian
        jacobian_y = forward @ jacobian_x @ inverse
        r, merit, _, _ = summary(evaluation.residual)
        rows = np.maximum(np.max(np.abs(jacobian_y), axis=1), np.finfo(float).tiny)
        u, singular, vt = np.linalg.svd(jacobian_y / rows[:, None], full_matrices=False)
        keep = singular > args.rcond * singular[0]
        dy = vt[keep].T @ ((u[:, keep].T @ (-r / rows)) / singular[keep])
        raw_other = float(np.max(np.abs(dy[others])))
        raw_helium_iii = float(np.max(np.abs(dy[helium_iii_rows]))) if helium_iii_rows.size else 0.0
        scale = min(1.0, args.radius / max(raw_other, 1e-300))
        helium_iii_step = np.clip(dy[helium_iii_rows], -args.helium_iii_radius, args.helium_iii_radius)
        dy *= scale
        dy[helium_iii_rows] = helium_iii_step
        dx = inverse @ dy
        accepted = None
        for factor in (1.0, 0.5, 0.25, 0.125):
            try:
                trial = equations.residual(x + factor * dx)
            except RecoverableEvaluationError as exc:
                print(f'  factor {factor}: rejected ({exc})', flush=True)
                continue
            if summary(trial.residual)[1] < merit:
                accepted = factor, trial
                break
        record = dict(iteration=iteration, elapsed_seconds=time.monotonic() - start,
                      truncated_modes=int(np.sum(~keep)), raw_step_other=raw_other,
                      raw_step_helium_iii=raw_helium_iii, scale=scale, merit_before=merit,
                      accepted_factor=None if accepted is None else accepted[0])
        if accepted is None:
            record['status'] = 'no decrease'
            print(json.dumps(record), flush=True)
            (args.output / 'iterations.jsonl').open('a').write(json.dumps(record) + '\n')
            break
        factor, evaluation = accepted
        x = x + factor * dx
        _, merit_after, max_other_population, max_helium_iii = summary(evaluation.residual)
        diagnostics = evaluation.payload[2]
        record.update({k: diagnostics[k] for k in DIAGNOSTIC_KEYS}, merit_after=merit_after,
                      max_population_residual_excluding_helium_iii=max_other_population,
                      max_helium_iii_residual=max_helium_iii,
                      temperature_step=float(np.max(np.abs(factor * dx[:nd]))))
        print(json.dumps(record), flush=True)
        (args.output / 'iterations.jsonl').open('a').write(json.dumps(record) + '\n')
        save_state(args.output / 'accepted.npz', evaluation.payload[0], evaluation.payload[1], model)


if __name__ == '__main__':
    main()
