"""Archive the independently tested, pre-promotion J1311 spectrum unchanged.

This is a fixed-state regression artifact, never a cold-start solver input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from wd_spectra._dq.provenance import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.source/'report.json').read_text())
    assert report['status'] == 'warm_converged' and report['certificate']['verified']
    assert report['parameters'] == [5529., 8.178, -5.27]
    # Exact pre-promotion atmosphere, not a newly computed reference.
    expected = 'ecdcd41d1ae95c17528da0ff5022ebdff339af8502be2cb7621278634458e198'
    assert digest(args.source/'converged.npz') == expected
    assert digest(args.source/'spectrum.npz') == 'a7e33e91ec7e60939968d48a79005feaf272de1a3b94aae4c83ff1a98bcb736e'
    with np.load(args.source/'converged.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in
                  ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')}
    with np.load(args.source/'spectrum.npz', allow_pickle=False) as z:
        arrays.update(wavelength=z['wavelength'].copy(), flux=z['flux'].copy())
    assert arrays['wavelength'].shape == (30000,)
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output/'j1311-completed-fixed.npz'
    if fixture.exists():
        raise FileExistsError(fixture)
    np.savez_compressed(fixture, **arrays)
    manifest = dict(source_run=str(args.source), source_atmosphere_sha256=expected,
        source_spectrum_sha256=digest(args.source/'spectrum.npz'),
        fixture_sha256=digest(fixture), points=30000, n_angle=4,
        parameters=dict(effective_temperature=5529., logg=8.178, log_carbon_to_helium=-5.27),
        scope=__doc__, builder_sha256=digest(__file__))
    (args.output/'completed-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
