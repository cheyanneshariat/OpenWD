"""Content identities, independent of checkout location or old model outputs."""
import hashlib
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
