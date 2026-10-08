"""Extract the independently normalized historical C–A addition for packaging.

No stellar observations or fitted strength enter this conversion. The source
archive contains both the original 8states table and the historical addition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from wd_spectra._dq.provenance import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]/'src/wd_spectra/data/dq'
    with np.load(args.source, allow_pickle=False) as candidate, np.load(
            root/'c2-8states-r15000.npz', allow_pickle=False) as original:
        keys = ('wavelength_angstrom', 'temperature_K', 'partition_temperature_K',
                'partition_function')
        for key in (*keys, 'swan_cross_section', 'swan_rotational_overlap_cross_section'):
            np.testing.assert_array_equal(candidate[key], original[key])
        delta = candidate['cross_section'] - original['cross_section']
        if not np.isfinite(delta).all() or np.any(delta < 0):
            raise ValueError('Invalid added cross section')
        np.testing.assert_allclose(delta, candidate['dazambuja_diagnostic_cross_section'],
                                   atol=2e-30, rtol=1e-10)
        provenance = json.loads(str(candidate['provenance_json'].item()))['historical_C_A_diagnostic']
        provenance.update(source_archive_sha256=digest(args.source),
            extraction='positive difference from unchanged 8states; no strength scaling',
            pressure_shift='none; no validated C–A-specific dense-helium law',
            approximation='finite-bin rigid-rotor envelope, not resolved rotational lines')
        np.savez_compressed(args.output, **{k: original[k] for k in keys}, cross_section=delta,
            source='Historical C2 C–A estimate: Cooper 1979 / Nicholls 1965',
            provenance_json=json.dumps(provenance, sort_keys=True))
    print(json.dumps({'file': args.output.name, 'sha256': digest(args.output),
                      'bytes': args.output.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
