#!/usr/bin/env python3
"""LTE H/He sdB model with the public DAB preset at fixed stellar parameters.

    python research/sdb_lte_hhe.py --teff 23200 --logg 5.20 --log-he-h -2.27 \
        --output results/sdb/hd4539-lte-hhe-40

Step 1 of the sdB build-up (LTE H/He structure and spectrum, no metals).
Writes the ModelResult plus spectrum.npz (vacuum wavelengths, surface F_lambda).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wd_spectra import ModelData, save_model_result
from wd_spectra.models.stellar import DABConfig, compute_dab


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--log-he-h', type=float, required=True, help='log N(He)/N(H)')
    parser.add_argument('--quality', choices=['standard', 'production'], default='standard')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')

    config = DABConfig(effective_temperature=args.teff, logg=args.logg,
                       log_hydrogen_to_helium=-args.log_he_h, quality=args.quality)
    wavelength = np.arange(3000.0, 9200.0, 0.05)
    start = time.monotonic()
    progress = []

    def callback(iteration, atmosphere, diagnostics):
        progress.append(dict(iteration=iteration, elapsed_seconds=time.monotonic() - start,
                             **{k: v for k, v in diagnostics.items() if isinstance(v, (int, float, str, bool))}))
        print(json.dumps(progress[-1]), flush=True)

    result = compute_dab(config, wavelength, data=ModelData.default(args.data_root), iteration_callback=callback)
    elapsed = time.monotonic() - start
    save_model_result(result, args.output)
    np.savez(args.output / 'spectrum.npz', wavelength_vacuum=result.spectrum.wavelength_angstrom,
             flux=result.spectrum.surface_flux_lambda)
    summary = dict(config=dict(teff=args.teff, logg=args.logg, log_he_h=args.log_he_h, quality=args.quality),
                   elapsed_seconds=elapsed,
                   atmosphere_convergence_status=result.metadata.get('atmosphere_convergence_status'),
                   iterations=len(progress))
    (args.output / 'run-summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, default=str), flush=True)


if __name__ == '__main__':
    main()
