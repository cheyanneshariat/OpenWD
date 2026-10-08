#!/usr/bin/env python3
"""Re-solve the full-NLTE (fraction 1) H/He stage from a public-run stage checkpoint.

    python research/sdb_restart_full_nlte.py CHECKPOINT --teff 29890 --logg 5.46 \
        --ratio 2.88 --output results/sdb/feige48-restart-f1

Research warm start for diagnosing the final continuation stage of the public
DAO cold start (run_hot_public_diagnostic.py writes stage-*.npz checkpoints).
Nothing is certified by inheritance: the full-NLTE solve applies its usual
population, flux and temperature-correction tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra._hot_structure import solve

from run_hot_public_diagnostic import save_state
from run_hot_trace_from_checkpoint import load_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True, help='log N(H)/N(He)')
    parser.add_argument('--quality', choices=['standard', 'production'], default='standard')
    parser.add_argument('--maximum-iterations', type=int, default=60)
    parser.add_argument('--equilibrate-populations', action='store_true',
                        help='solve fixed-temperature full-NLTE populations before the coupled solve')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)

    config = DAOConfig(effective_temperature=args.teff, logg=args.logg,
                       log_hydrogen_to_helium=args.ratio, quality=args.quality)
    data = ModelData.default(args.data_root)
    model, seed, populations = load_seed(args.checkpoint, config, data)
    source = Path(solve.__code__.co_filename).resolve().parent
    manifest = dict(cold_start=False, checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    config=asdict(config), status='running',
                    fixed_temperature_population_initialization=args.equilibrate_populations,
                    source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(source.rglob('*.py'))})
    (args.output / 'restart-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')

    start = time.monotonic()

    def progress(iteration, atmosphere, population_state, diagnostics):
        save_state(args.output / 'accepted.npz', atmosphere, population_state, model)
        record = dict(elapsed_seconds=time.monotonic() - start, iteration=iteration, **diagnostics)
        with (args.output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)) + '\n')

    resolution = numerical_resolution(config.quality)
    try:
        if args.equilibrate_populations:
            populations = model.solve_populations(seed, populations)
            save_state(args.output / 'population-initialization.npz', seed, populations, model)
            print(f'fixed-temperature populations: converged={populations.converged}, '
                  f'change={populations.maximum_relative_population_change:.3g}, '
                  f'{time.monotonic() - start:.0f} s', flush=True)
        answer = solve(seed, model, structure_wavelength(config, resolution.n_continuum),
                       populations=populations, nlte_fraction=1., maximum_iterations=args.maximum_iterations,
                       iteration_callback=progress)
        save_state(args.output / 'final.npz', answer.atmosphere, answer.population_state, model)
        metadata = answer.atmosphere.metadata
        manifest.update(status='finished',
                        converged=bool(metadata.get('radiative_equilibrium_solver_converged')),
                        populations_converged=bool(answer.population_state.converged))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        (args.output / 'restart-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
