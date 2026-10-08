"""Fresh public DAZ runs; paper checkpoints are never solver inputs."""

import argparse
import logging
from wd_spectra import DAZConfig, run_model

CASES = {
    "daz-g29-38": DAZConfig(),
    "daz-g149-28": DAZConfig(
        effective_temperature=8600,
        logg=8.10,
        abundances={
            "Mg": -7.24,
            "Al": -8.17,
            "Ca": -8.04,
            "Ti": -9.48,
            "Fe": -7.41,
            "Ni": -8.33,
        },
    ),
    "daz-galex1931": DAZConfig(
        effective_temperature=20890,
        logg=7.90,
        abundances={"O": -3.62, "Mg": -4.42, "Si": -4.24, "Ca": -6.11, "Fe": -4.43},
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run = run_model(CASES[args.case], args.output, require_convergence=True)
    assert run.selection.workflow == "daz" and run.convergence_verified


if __name__ == "__main__":
    main()
