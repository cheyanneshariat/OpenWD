#!/usr/bin/env python3
"""Size of metal line blanketing in an sdB structure: LTE DA vs LTE DAZ at fixed parameters.

    python research/sdb_blanketing_check.py --teff 28200 --logg 5.61 \
        --abundance C=-4.23 --abundance N=-3.98 --abundance O=-4.35 \
        --abundance Si=-4.2 --abundance S=-4.74 --abundance Fe=-4.84 \
        --output results/sdb/feige38-blanketing

The hybrid sdB host is an H/He LTE structure without metals.  This differential
check solves the same pure-hydrogen LTE structure twice, without and with the
metal opacity (the DAZ preset: Stout lines up to charge 3, opacity-sampled in
the structure), on identical depth meshes.  The He (log He/H ~ -2.5) is left
out of both, so the difference isolates the metals.  Writes T(tau_Ross) of both
and their spectra; prints the temperature change across the line-forming layers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wd_spectra import ModelData, save_model_result
from wd_spectra.models.daz import DAZConfig, compute_daz
from wd_spectra.models.stellar import DAConfig, compute_da


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--abundance', action='append', required=True, help='ELEMENT=log10 N/N(H)')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    args.output.mkdir(parents=True)
    abundances = {k: float(v) for k, v in (item.split('=') for item in args.abundance)}
    data = ModelData.default(args.data_root)
    wavelength = np.arange(3000.0, 9200.0, 0.05)
    results, timing = {}, {}
    for label, run in (
            ('da', lambda callback: compute_da(
                DAConfig(effective_temperature=args.teff, logg=args.logg, photospheric_depth_concentration=1.0),
                wavelength, data=data, iteration_callback=callback)),
            ('daz', lambda callback: compute_daz(
                DAZConfig(effective_temperature=args.teff, logg=args.logg, abundances=abundances,
                          photospheric_depth_concentration=1.0),
                wavelength, data=data, iteration_callback=callback))):
        start = time.monotonic()

        def callback(iteration, atmosphere, diagnostics, label=label, start=start):
            print(json.dumps(dict(model=label, iteration=iteration, elapsed_seconds=time.monotonic() - start,
                                  **{k: v for k, v in diagnostics.items()
                                     if isinstance(v, (int, float, str, bool))})), flush=True)

        results[label] = run(callback)
        timing[label] = time.monotonic() - start
        save_model_result(results[label], args.output / label)
        np.savez(args.output / label / 'spectrum.npz', wavelength_vacuum=results[label].spectrum.wavelength_angstrom,
                 flux=results[label].spectrum.surface_flux_lambda)

    tau = {k: r.atmosphere.rosseland_optical_depth for k, r in results.items()}
    temperature = {k: r.atmosphere.temperature for k, r in results.items()}
    rows = []
    for t in (1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.67, 2.0):
        da = float(np.interp(np.log(t), np.log(tau['da']), temperature['da']))
        daz = float(np.interp(np.log(t), np.log(tau['daz']), temperature['daz']))
        rows.append(dict(tau_ross=t, da=da, daz=daz, change_percent=100 * (daz / da - 1)))
        print(f'tau_Ross {t:8.0e}: T(DA) {da:8.0f} K  T(DAZ) {daz:8.0f} K  {rows[-1]["change_percent"]:+6.2f}%')
    (args.output / 'run-summary.json').write_text(json.dumps(dict(
        teff=args.teff, logg=args.logg, abundances=abundances, elapsed_seconds=timing,
        convergence={k: r.metadata.get('atmosphere_convergence_status') for k, r in results.items()},
        temperature_change=rows), indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
