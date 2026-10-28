"""Historical import names still refer to the package implementation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["dense_mesh_hierarchy", "dense_thermal_verification"])
def test_historical_import_alias_from_another_directory(name, tmp_path):
    wrappers = Path(__file__).resolve().parent
    source = wrappers.parents[1] / "src"
    code = (
        "import importlib; "
        f"legacy = importlib.import_module({name!r}); "
        f"canonical = importlib.import_module('wd_spectra._cool.' + {name!r}); "
        "assert legacy is canonical"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                   env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, wrappers)))))
