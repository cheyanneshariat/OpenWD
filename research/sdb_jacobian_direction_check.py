#!/usr/bin/env python3
"""Check a saved coupled H/He Jacobian along chosen directions by central differences.

    python research/sdb_jacobian_direction_check.py STATE.npz JACOBIAN.npz --teff 29890 \
        --logg 5.46 --ratio 2.88

For the smallest singular vectors of the row-scaled Jacobian and a few random
unit directions v, compares J v with [r(x + e v) - r(x - e v)] / (2 e) for
several e.  Diagnostic only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wd_spectra import DAOConfig
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra._hot_structure import HotEquations

from run_hot_trace_from_checkpoint import load_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('state', type=Path)
    parser.add_argument('jacobian', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--singular', type=int, default=3)
    parser.add_argument('--random', type=int, default=2)
    parser.add_argument('--steps', type=float, nargs='+', default=[1e-3, 1e-4])
    args = parser.parse_args()

    config = DAOConfig(effective_temperature=args.teff, logg=args.logg, log_hydrogen_to_helium=args.ratio)
    model, seed, populations = load_seed(args.state, config, ModelData.default(args.data_root))
    equations = HotEquations(seed, model, structure_wavelength(config, numerical_resolution(config.quality).n_continuum),
                             nlte_fraction=1.)
    x = equations.initial_state(populations)
    saved = np.load(args.jacobian)
    jacobian = saved['jacobian']
    nt, nd = equations.nd, equations.nd
    rows = np.maximum(np.max(np.abs(jacobian), axis=1), np.finfo(float).tiny)
    _, singular, vt = np.linalg.svd(jacobian / rows[:, None], full_matrices=False)
    rng = np.random.default_rng(1)
    directions = [(f'singular {singular[-k]:.1e}', vt[-k]) for k in range(1, args.singular + 1)]
    directions += [(f'random {i}', v / np.linalg.norm(v)) for i, v in
                   enumerate(rng.standard_normal((args.random, x.size)))]
    for label, v in directions:
        predicted = jacobian @ v
        block = dict(thermal=slice(0, nt), populations=slice(nt, None))
        print(f'{label}: |v_T|={np.linalg.norm(v[:nt]):.3f}', flush=True)
        for step in args.steps:
            plus = equations.residual(x + step * v).residual
            minus = equations.residual(x - step * v).residual
            measured = (plus - minus) / (2 * step)
            for name, s in block.items():
                p, m = predicted[s], measured[s]
                error = np.linalg.norm(p - m) / max(np.linalg.norm(m), np.finfo(float).tiny)
                print(f'   e={step:.0e} {name:11s} |Jv|={np.linalg.norm(p):.3e} |FD|={np.linalg.norm(m):.3e} '
                      f'rel.err={error:.3f}', flush=True)
            worst = np.argmax(np.abs(predicted - measured))
            print(f'   worst row {worst} ({"thermal" if worst < nt else "pop depth %d state %d" % divmod(worst - nt, (x.size - nt) // nd)}): '
                  f'Jv={predicted[worst]:.4e} FD={measured[worst]:.4e}', flush=True)


if __name__ == '__main__':
    main()
