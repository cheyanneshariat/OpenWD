"""Verify a regenerated selection against the bundled numerical arrays/evidence."""
import argparse
import json
import lzma
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / 'src/wd_spectra/data/atomic/independent-kurucz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualified', type=Path)
    args = parser.parse_args()
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for key in manifest['ions']:
        with np.load(BUNDLE / (key + '.npz'), allow_pickle=False) as bundled:
            with np.load(args.qualified / (key + '.npz'), allow_pickle=False) as rebuilt:
                for array in ('records', 'levels'):
                    np.testing.assert_array_equal(bundled[array], rebuilt[array])
        expected = json.loads(lzma.decompress((BUNDLE / (key + '.evidence.json.xz')).read_bytes()))
        assert json.loads((args.qualified / (key + '.evidence.json')).read_text()) == expected
    policy = json.loads((args.qualified / 'atomic_policy.json').read_text())
    assert policy['report_by_ion'] == manifest['source_policy']['report_by_ion']
    print('All seven catalogues, every evidence row, and all 30 selection reports agree.')


if __name__ == '__main__':
    main()
