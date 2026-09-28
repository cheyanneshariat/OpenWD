"""Verify the exact external molecular data used by the qualified DAB recipe.

No downloading, silent substitution or modification of input tables.
See README.md for original sources and attribution requirements.
"""
import hashlib
from .research_paths import data_directory

SHA256 = {
    "H2-He_2011.cia": "4f0eb9cd69a1c383f53a1431495bae0c01a30a41cc1b8433c2d726763ff45431",
    "1H2__RACPPK.states.bz2": "276f5a36d094e7e1417c44f11c1d173417e92629c5f747b6a862bcce5c374a81",
}


def verify(directory):
    for name, expected in SHA256.items():
        path = directory / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{path}: differs from the qualified input; no replacement used")
        print(f"Verified {name}: {expected}", flush=True)


if __name__ == "__main__":
    verify(data_directory())
