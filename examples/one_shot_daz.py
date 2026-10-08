"""Generate a polluted hydrogen atmosphere from scratch and save its spectrum."""

import argparse
from wd_spectra import DAZConfig, run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DAZConfig()
    parser.add_argument("--teff", type=float, default=defaults.effective_temperature)
    parser.add_argument("--logg", type=float, default=defaults.logg)
    parser.add_argument(
        "--quality", choices=("quick", "standard", "production"), default="standard"
    )
    parser.add_argument("--abundance", action="append", metavar="ELEMENT=LOG_Z_H")
    parser.add_argument("--output", default="results/daz-g29-38")
    parser.add_argument("--require-convergence", action="store_true")
    args = parser.parse_args()
    try:
        abundances = (
            dict(
                (item.split("=", 1)[0], float(item.split("=", 1)[1]))
                for item in args.abundance
            )
            if args.abundance
            else defaults.abundances
        )
    except (ValueError, IndexError):
        parser.error("abundances must be Element=log10(N(element)/N(H))")
    run_model(
        DAZConfig(
            effective_temperature=args.teff,
            logg=args.logg,
            quality=args.quality,
            abundances=abundances,
        ),
        args.output,
        require_convergence=args.require_convergence,
    )


if __name__ == "__main__":
    main()
