"""Compare native and NumPy Stark profiles without an atmosphere solve.

Run with PYTHONPATH=src and OPENBLAS_NUM_THREADS=1. The larger paired J1109
and J1637 opacity benchmarks remain necessary to establish application gains.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np

from wd_spectra import metals
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    native = metals._rt
    if not hasattr(native, 'stark_manifold_bins'):
        parser.error('build the optional C extension before benchmarking')
    db = _atomic_inputs(ModelData.default(), ('Mg',))[0]
    ion = db.ions['Mg', 1]
    def manifold(label):
        level = next(level for level in ion.levels if level.label == label)
        return metals.rydberg_stark_manifold(db, ion, level)
    upper = manifold('2p6.10g.(2G<9/2>)')
    lower = manifold('2p6.4f.(2Fo<7/2>)')
    center = 4333.17
    wave = center + np.linspace(-300., 300., 12001)
    cases = [(wave, center, .03, gamma, coupling, upper, lo, beta)
             for gamma, coupling, beta, lo in [
                 (.005, 2.e9, np.inf, None),
                 (.05, 2.e10, 7.13, lower),
                 (2., 2.e11, 1.3, lower),
             ]]
    timings = {'numpy': [], 'native': []}
    values = {}
    def evaluate():
        return [metals.manifold_quasistatic_line_profile(
            *case, correlation=.2, radiator_core_charge=2.,
            support_half_width=300.) for case in cases]
    try:
        # Exclude imports and atomic/cache initialization from both paths.
        for backend in (None, native):
            metals._rt = backend
            evaluate()
        for repeat in range(args.repeat):
            order = ('numpy', 'native') if repeat % 2 == 0 else ('native', 'numpy')
            for mode in order:
                metals._rt = native if mode == 'native' else None
                start = time.perf_counter()
                values[mode] = evaluate()
                timings[mode].append(time.perf_counter() - start)
    finally:
        metals._rt = native
    difference = max(float(np.max(np.abs(a-b))/np.max(np.abs(b)))
                     for a,b in zip(values['native'],values['numpy']))
    print(json.dumps(dict(seconds=timings,
                          median_speedup=float(np.median(timings['numpy'])/np.median(timings['native'])),
                          maximum_difference_relative_to_profile_peak=difference), indent=2))


if __name__ == '__main__':
    main()
