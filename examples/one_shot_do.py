#!/usr/bin/env python3
"""Cold DO/DAO runs using the atomic data installed with OpenWD."""
import argparse
import logging
from wd_spectra import DOConfig, DAOConfig, ModelData, run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teff', type=float, default=70000.)
    parser.add_argument('--logg', type=float, default=8.)
    parser.add_argument('--log-hydrogen-to-helium', type=float,
                        help='Enable the mixed DAO model; log10 N(H)/N(He)')
    parser.add_argument('--quality', choices=('quick', 'standard', 'production'), default='standard')
    parser.add_argument(
        '--data-root',
        help='optional complete custom data root; bundled data are the default',
    )
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    options = dict(effective_temperature=args.teff, logg=args.logg, quality=args.quality)
    config = (DOConfig(**options) if args.log_hydrogen_to_helium is None else
              DAOConfig(**options, log_hydrogen_to_helium=args.log_hydrogen_to_helium))
    run_model(config, args.output, data=ModelData.default(args.data_root), require_convergence=True)


if __name__ == '__main__':
    main()
