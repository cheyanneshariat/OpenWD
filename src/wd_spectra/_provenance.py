"""Content identities for numerical code and physical inputs.

Read contents on each request: a path, file size, or modification time does
not establish that an existing atmosphere used the same physical inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def content_identity(paths: Iterable[Path]) -> str:
    """Hash an ordered set of named files, including missing-file markers."""
    result = hashlib.sha256()
    for path in sorted(set(map(Path, paths))):
        name = str(path.resolve()).encode("utf-8")
        result.update(len(name).to_bytes(8, "big"))
        result.update(name)
        if not path.is_file():
            result.update(b"missing\0")
            continue
        result.update(b"file\0")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.update(digest.digest())
    return result.hexdigest()


def numerical_code_identity() -> str:
    """Include Python equations and the installed optional native backend."""
    package = Path(__file__).resolve().parent
    paths = list(package.rglob("*.py"))
    paths.extend(package.glob("_rt*.so"))
    paths.extend(package.glob("_rt*.pyd"))
    return content_identity(paths)


def model_data_identity(data) -> str:
    """Conservatively identify bundled tables and the selected data workspace.

    Some readers use package tables directly, while others use ModelData. Hash
    both, including additions and removals, so neither route can reuse a stale
    equilibrium certificate. Caller-supplied tables outside these roots remain
    part of the adapter's explicit physical_data_identity.
    """
    package_data = Path(__file__).resolve().parent / "data"
    roots = {package_data, data.cache, data.allard_lyman}
    paths = set()
    for root in roots:
        if root.is_dir():
            paths.update(p.resolve() for p in root.rglob("*") if p.is_file())
        else:
            paths.add(root.resolve())
    return content_identity(paths)
