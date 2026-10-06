#!/usr/bin/env python3
"""Offline test: full-NLTE H/He Newton steps with near-singular directions truncated.

    python research/sdb_truncated_newton_test.py STATE.npz JACOBIAN.npz --teff 29890 \
        --logg 5.46 --ratio 2.88 --output results/sdb/truncated-newton-test

Diagnostic only; nothing in the package solver is changed.  Starting from a
saved full-NLTE (fraction 1) state and its saved coupled Jacobian, each step
solves the row-scaled Newton system by SVD, discards singular directions below
``rcond`` times the largest singular value, limits the step to the usual
max-norm trust radius, and backtracks on the least-squares merit.  The
Jacobian is rebuilt every ``--refresh`` iterations.  The question is whether
removing the nearly singular He I ionization modes from the step lets the
remaining residual converge, where the production driver instead scales the
whole step down to the trust radius.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra.nonlinear import RecoverableEvaluationError
from wd_spectra._hot_structure import HotEquations

from run_hot_public_diagnostic import save_state
from run_hot_trace_from_checkpoint import load_seed

DIAGNOSTIC_KEYS = ('maximum_continuation_population_residual', 'maximum_all_depth_total_flux_residual',
                   'maximum_relative_cell_energy_balance_residual', 'nlte_maximum_relative_population_change')


def truncated_step(jacobian, residual, rcond):
    rows = np.maximum(np.max(np.abs(jacobian), axis=1), np.finfo(float).tiny)
    u, singular, vt = np.linalg.svd(jacobian / rows[:, None], full_matrices=False)
    keep = singular > rcond * singular[0]
    coefficients = (u[:, keep].T @ (-residual / rows)) / singular[keep]
    full = (u.T @ (-residual / rows)) / singular
    return vt[keep].T @ coefficients, int(np.sum(~keep)), float(np.max(np.abs(vt.T @ full)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('state', type=Path)
    parser.add_argument('jacobian', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--rcond', type=float, default=1e-3)
    parser.add_argument('--radius', type=float, default=0.12)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--refresh', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)

    config = DAOConfig(effective_temperature=args.teff, logg=args.logg, log_hydrogen_to_helium=args.ratio)
    model, seed, populations = load_seed(args.state, config, ModelData.default(args.data_root))
    resolution = numerical_resolution(config.quality)
    equations = HotEquations(seed, model, structure_wavelength(config, resolution.n_continuum), nlte_fraction=1.)
    x = equations.initial_state(populations)
    saved = np.load(args.jacobian)
    print(f'state vs saved Jacobian x: max |dx| = {np.max(np.abs(x - saved["x"])):.2e}', flush=True)
    jacobian = saved['jacobian']
    evaluation = equations.residual(x)
    print(f'residual vs saved: max |dr| = {np.max(np.abs(evaluation.residual - saved["residual"])):.2e}', flush=True)

    start = time.monotonic()
    log = args.output / 'iterations.jsonl'
    for iteration in range(1, args.iterations + 1):
        if iteration > 1 and (iteration - 1) % args.refresh == 0:
            evaluation = equations.evaluate(x, True)
            jacobian = evaluation.jacobian
        residual = evaluation.residual
        merit = float(residual @ residual)
        delta, truncated, unrestricted = truncated_step(jacobian, residual, args.rcond)
        delta *= min(1.0, args.radius / max(np.max(np.abs(delta)), np.finfo(float).tiny))
        accepted = None
        for factor in (1.0, 0.5, 0.25, 0.125):
            try:
                trial = equations.residual(x + factor * delta)
            except RecoverableEvaluationError as exc:
                print(f'  factor {factor}: rejected ({exc})', flush=True)
                continue
            if float(trial.residual @ trial.residual) < merit:
                accepted = factor, trial
                break
        record = dict(iteration=iteration, elapsed_seconds=time.monotonic() - start, truncated_modes=truncated,
                      unrestricted_full_step=unrestricted, step_max=float(np.max(np.abs(delta))),
                      merit_before=merit, accepted_factor=None if accepted is None else accepted[0])
        if accepted is None:
            record['status'] = 'no decrease'
            print(json.dumps(record), flush=True)
            log.open('a').write(json.dumps(record) + '\n')
            break
        factor, evaluation = accepted
        x = x + factor * delta
        diagnostics = evaluation.payload[2]
        record.update({k: diagnostics[k] for k in DIAGNOSTIC_KEYS},
                      merit_after=float(evaluation.residual @ evaluation.residual),
                      max_abs_residual=float(np.max(np.abs(evaluation.residual))),
                      temperature_step=float(np.max(np.abs(factor * delta[:equations.nd]))))
        print(json.dumps(record), flush=True)
        log.open('a').write(json.dumps(record) + '\n')
        save_state(args.output / 'accepted.npz', evaluation.payload[0], evaluation.payload[1], model)


if __name__ == '__main__':
    main()
