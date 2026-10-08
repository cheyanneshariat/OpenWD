"""Checksum-pinned constitutive data; never an atmosphere or a spectrum."""
from functools import lru_cache
from importlib.resources import files
import json
import os
from pathlib import Path

from .provenance import digest


def data_root():
    override = os.environ.get('OPENWD_DQ_DATA')
    return (Path(override).expanduser().resolve() if override else
            Path(files('wd_spectra').joinpath('data/dq')))


def line_root():
    return data_root()


@lru_cache(maxsize=4)
def _checked(root, identities):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = root / name
        if digest(path) != expected:
            raise ValueError(f'DQ constitutive data checksum mismatch: {path}')
    return manifest


def validate_data():
    root = data_root()
    paths = [root/name for name in ('manifest.json', 'c2-8states-r15000.npz',
             'hornkohl_calibrated.npz', 'report.json', 'correction.npz', 'c2-ca-historical.npz',
             'c2-ca-2024.npz', 'swan-completed.npz')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing DQ constitutive data: {path}; reinstall '
                                    'OpenWD or set OPENWD_DQ_DATA to a complete data directory')
    return _checked(root, tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths))
