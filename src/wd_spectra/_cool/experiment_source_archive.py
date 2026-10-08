"""Preserve the exact Python sources used by an explicitly scoped experiment.

Archives are provenance artifacts, not another importable production tree.
Hash and archive the same bytes so later local edits cannot make a recorded
experiment unrecoverable. Runtime binaries and input data are not included.
"""
import hashlib
import io
from pathlib import Path
import tarfile


def archive_sources(paths, destination, *, base_directory=Path('.')):
    paths=sorted(Path(p) for p in paths)
    if len(set(paths))!=len(paths):
        raise ValueError('duplicate source path')
    if any(p.is_absolute() or '..' in p.parts for p in paths):
        raise ValueError('source paths must be workspace-relative')
    hashes={}
    with tarfile.open(destination,'x:gz') as archive:
        for path in paths:
            data=(Path(base_directory)/path).read_bytes()
            hashes[str(path)]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(str(path))
            info.size=len(data)
            info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    return hashes
