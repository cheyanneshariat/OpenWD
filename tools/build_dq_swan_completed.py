"""Preserve Hornkohl and add only ExoMol transitions outside band coverage.

No stellar inputs, calibrated mixing weights, or internal-hole filling.
Input is the audited ExoMol branch cache with v/J labels, not a spectrum.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from wd_spectra._dq.data import data_root
from wd_spectra._dq.provenance import digest


def build(exomol, output):
    if digest(exomol) != 'f1f156a35f87c1b6d97505655613816aebe149d1934b0a510c759e456f642e3f':
        raise ValueError('ExoMol branch cache differs from audited source')
    hornkohl = data_root()/'hornkohl_calibrated.npz'
    if digest(hornkohl) != 'd7a8df19e3fe81e394d2e91e389853c7d8a84111efd78f5eede2ad2062256c90':
        raise ValueError('Hornkohl input differs from audited source')
    with np.load(exomol, allow_pickle=False) as z:
        x = z['lines'].copy()
    with np.load(hornkohl, allow_pickle=False) as z:
        h = z['lines'].copy(); r = z['original_numeric_records'].copy()
    covered = np.zeros(x.shape[1], bool)
    for vu, vl in np.unique(r[2:4].T, axis=0):
        m = (r[2] == vu) & (r[3] == vl)
        covered |= ((x[4] == vu) & (x[5] == vl) &
                    (x[3]+x[6] <= r[0,m].max()) & (x[3] <= r[1,m].max()))
    candidate = np.concatenate((h, x[:3, ~covered]), axis=1)
    np.testing.assert_array_equal(candidate[:, :h.shape[1]], h)
    assert candidate.shape == (3, 1261446)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'swan-completed.npz', lines=candidate)
    report = dict(exomol_sha256=digest(exomol), hornkohl_sha256=digest(hornkohl),
                  output_sha256=digest(output/'swan-completed.npz'),
                  retained_lines=h.shape[1], added_lines=int((~covered).sum()),
                  scope=__doc__, builder_sha256=digest(__file__))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exomol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.exomol, a.output), indent=2))
