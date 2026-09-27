"""Installed source and explicit molecular-data locations for cool workers."""
import os
from pathlib import Path
from wd_spectra.models.common import ModelData

# The archive root is the installed package parent, including in a wheel.
REPOSITORY = Path(__file__).resolve().parents[2]


def data_directory():
    return Path(os.environ.get("OPENWD_RESEARCH_DATA",
        ModelData.default().cache / "molecular-opacity")).expanduser().resolve()


def source_paths():
    return sorted(p.relative_to(REPOSITORY)
                  for p in (REPOSITORY / "wd_spectra").rglob("*.py"))
