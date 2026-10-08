"""Isolated cold DQ worker: python -m wd_spectra._dq.worker --help."""
import argparse
import json
import logging
from pathlib import Path
import sys
from wd_spectra.models.common import validate_wavelength
from wd_spectra.models.dq import DQConfig, validate_config


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--teff', type=float, required=True)
    parser.add_argument('--logg', type=float, required=True)
    parser.add_argument('--log-carbon-to-helium', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=28800.)
    parser.add_argument('--read-output-grid', action='store_true')
    parser.add_argument('--no-c2-ca', action='store_true', help='Diagnostic: omit historical C2 C–A opacity')
    args = parser.parse_args(arguments)
    config = DQConfig(args.teff, args.logg, args.log_carbon_to_helium,
                      maximum_seconds=args.seconds, include_c2_ca=not args.no_c2_ca)
    validate_config(config)
    wavelength = None
    if args.read_output_grid:
        request = json.load(sys.stdin)
        if set(request) != {'output_wavelength'}:
            parser.error('Only an output wavelength grid may be supplied; no structure inputs')
        if request['output_wavelength'] is not None:
            wavelength = validate_wavelength(request['output_wavelength'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from .runtime import run_cold
    run_cold(config, args.output, wavelength=wavelength)


if __name__ == '__main__':
    main()
