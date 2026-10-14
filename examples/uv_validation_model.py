#!/usr/bin/env python3
"""Generate one UV validation spectrum at its published composition.

Example: python examples/uv_validation_model.py gd40 --output results/gd40-uv
The output is intrinsic surface flux; no observational corrections are applied.
"""

import argparse
import json
import os
from pathlib import Path
import sys


# Set these before importing numerical libraries. Explicit user settings win.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENWD_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

CHECKOUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKOUT / "src"))

import numpy as np

from wd_spectra import DAZConfig, DZConfig, compiled_backend_available, run_model


CONFIGURATIONS = CHECKOUT / "docs" / "assets" / "uv-validation-configs.json"


def main(argv=None):
    targets = json.loads(CONFIGURATIONS.read_text(encoding="utf-8"))["targets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=targets, help="published validation configuration")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args(argv)
    if not compiled_backend_available():
        parser.error("Install OpenWD with its compiled C backend before generating a model")

    entry = targets[args.target]
    config_type = {"DZConfig": DZConfig, "DAZConfig": DAZConfig}[entry["config_class"]]
    config = config_type(**entry["config"])
    wavelength = np.arange(
        entry["wavelength_min_A"],
        entry["wavelength_max_A"] + 0.5 * entry["wavelength_step_A"],
        entry["wavelength_step_A"],
    )
    run = run_model(
        config,
        args.output,
        wavelength=wavelength,
        require_convergence=True,
    )
    print(f"Saved {args.target} to {run.output_directory}")


if __name__ == "__main__":
    main()
