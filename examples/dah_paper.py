#!/usr/bin/env python3
"""Cold-start predictions at the eight fixed parameter sets used in the paper.

Example: python examples/dah_paper.py j1007+1237 results/j1007
The physical surface flux is saved without velocity or calibration corrections.
"""

import argparse
from pathlib import Path

import numpy as np

from wd_spectra import DAHConfig, run_model


# Teff [K], log g [cgs], undisplaced dipole polar field [MG], inclination
# [degrees], magnetic-frame displacement [stellar radii]. See docs/models/DAH.md.
TARGETS = {
    "j1007+1237": (18687., 8.04, 6.13, 71., (0., 0., .30)),
    "j1034+0327": (15756., 8.80, 11.17, 77., (0., 0., .09)),
    "j1154+0117": (29316., 8.90, 35.53, 87., (0., 0., -.23)),
    "j2149-0728": (22642., 8.37, 45.09, 66., (0., 0., .17)),
    "j1254+5612": (12870., 8.58, 60.43, 62., (0., 0., .18)),
    "j1018+0111": (10500., 8.00, 108.12, 60., (0., .07, .10)),
    "j1351+5419": (13937., 8.43, 368.52, 34., (0., 0., .07)),
    "j2247+1456": (19000., 8.00, 437.10, 10., (0., 0., -.15)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    teff, logg, polar, inclination, offset = TARGETS[args.target]
    config = DAHConfig(effective_temperature=teff, logg=logg,
                       magnetic_field_megagauss=polar, field_geometry="dipole",
                       dipole_inclination_deg=inclination, dipole_offset_radius=offset,
                       quality="production")
    run = run_model(config, args.output, wavelength=np.arange(3600., 7000.01, 1.),
                    require_convergence=True)
    print(f"Saved {args.target} to {run.output_directory}")


if __name__ == "__main__":
    main()
