"""Bundled data for the hot DA/DAO trace-metal module.

The files live in ``wd_spectra/data/hot_daz`` (see its README.md). Large text
tables are stored xz-compressed; :func:`data_file` returns a readable path,
decompressing once per process into a private cache directory. Checksums in
the bundled manifests refer to the uncompressed content and are verified by
the individual loaders.
"""
from __future__ import annotations

import atexit
import lzma
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

ROOT = Path(str(files('wd_spectra'))) / 'data' / 'hot_daz'
ATOMIC = ROOT / 'atomic'
KURUCZ = ROOT / 'kurucz'
CHIANTI_RECOMBINATION = ROOT / 'chianti' / 'recombination'
CHIANTI_CARBON = ROOT / 'chianti' / 'carbon'
CHIANTI_OXYGEN = ROOT / 'chianti' / 'oxygen'
G191B2B_OBSERVATIONS = ROOT / 'observations' / 'g191-b2b'

_CACHE: Path | None = None


def cache_directory() -> Path:
    """Private per-process scratch directory, removed at exit."""
    global _CACHE
    if _CACHE is None:
        _CACHE = Path(tempfile.mkdtemp(prefix='openwd-hot-daz-'))
        atexit.register(shutil.rmtree, _CACHE, True)
    return _CACHE


def data_file(directory, name) -> Path:
    """Path to ``directory/name``, or to the decompressed ``name.xz``."""
    path = Path(directory) / name
    if path.is_file():
        return path
    compressed = path.with_name(path.name + '.xz')
    if not compressed.is_file():
        raise FileNotFoundError(path)
    target = cache_directory() / f'{abs(hash(str(compressed.resolve())))}-{name}'
    if not target.is_file():
        with lzma.open(compressed, 'rb') as source, open(target, 'wb') as sink:
            shutil.copyfileobj(source, sink)
    return target
