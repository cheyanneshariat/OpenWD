"""Offline capture of the DAH public-default spectra after the RWA/magnetic-EOS change.

Each of the eight frozen paper atmospheres in tests/data/dah_paper is synthesized
twice with atmosphere iteration forbidden: the paper preset must first recover
the frozen paper spectrum at its original tolerance, then the new public default
is recorded. Existing controls are never overwritten, and tests/CI never run this
script. The manifest records hashes, the request identity and the measured change
from the paper spectrum. No atmosphere is reconverged or certified here.

Usage: PYTHONPATH=src python research/capture_dah_rwa_default_controls.py OUTPUT
(rerunning with the same OUTPUT resumes after the last completed star)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np

from wd_spectra import compiled_backend_available
from wd_spectra._compat import trapezoid
from wd_spectra.models import AtmosphereConvergenceWarning, DAHConfig, compute_dah, paper_dah_config
from wd_spectra.models.common import ModelData, load_atmosphere_checkpoint, model_request_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'tests/data/dah_paper'
BANDS = ((3600, 3900), (3900, 4300), (4300, 4800), (4800, 5500), (5500, 6500), (6500, 7100))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(wavelength, actual, expected):
    relative = actual / expected - 1.
    index = int(np.argmax(np.abs(relative)))
    return dict(maximum_relative_change=float(np.max(np.abs(relative))),
        maximum_change_wavelength_angstrom=float(wavelength[index]),
        integrated_flux_change=float(trapezoid(actual - expected, wavelength)
                                     / trapezoid(expected, wavelength)),
        band_fractional_changes={f'{lo}-{hi}': float(
            trapezoid((actual-expected)[take], wavelength[take])
            / trapezoid(expected[take], wavelength[take]))
            for lo, hi in BANDS if np.count_nonzero(take := (wavelength >= lo) & (wavelength < hi)) > 1})


def stellar_config(record):
    names = ('effective_temperature', 'logg', 'magnetic_field_megagauss', 'quality', 'field_geometry',
             'dipole_inclination_deg', 'dipole_offset_radius')
    values = {name: record['config'][name] for name in names}
    values['dipole_offset_radius'] = tuple(values['dipole_offset_radius'])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (args.output / 'fixed').mkdir(parents=True, exist_ok=True)
    partial = args.output / 'manifest.partial.json'
    if (args.output / 'manifest.json').exists():
        raise SystemExit(f'{args.output} is already complete')
    assert compiled_backend_available(), 'use the compiled backend for the saved controls'
    targets = json.loads((PAPER / 'manifest.json').read_text())['targets']
    # Resume: keep cases whose fixture matches the partial manifest.
    cases = json.loads(partial.read_text()) if partial.exists() else {}
    cases = {key: case for key, case in cases.items()
             if (args.output / 'fixed' / (key + '.npz')).exists()
             and digest(args.output / 'fixed' / (key + '.npz')) == case['fixture_sha256']}
    for key, record in targets.items():
        if key in cases:
            print(key, 'already captured', flush=True)
            continue
        path = PAPER / record['reference_file']
        parameters = stellar_config(record)
        with np.load(path) as saved:
            wavelength, expected = saved['wavelength'].copy(), saved['flux'].copy()
        atmosphere = load_atmosphere_checkpoint(path, parameters['effective_temperature'], parameters['logg'],
            'hydrogen', include_molecules=False, include_negative_hydrogen=False,
            trihydrogen_ion_partition_model='neale-tennyson-1995')
        outputs = {}
        for label, config in (('paper', paper_dah_config(**parameters)), ('default', DAHConfig(**parameters))):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AtmosphereConvergenceWarning)
                result = compute_dah(config, wavelength, initial_atmosphere=atmosphere, relax_atmosphere=False)
            flux = result.spectrum.surface_flux_lambda
            assert np.all(np.isfinite(flux)) and np.all(flux > 0.)
            outputs[label] = (flux.copy(), config, result.metadata)
            print(key, label, flush=True)
        np.testing.assert_allclose(outputs['paper'][0], expected, rtol=2e-8, atol=0.)
        flux, config, metadata = outputs['default']
        out = args.output / 'fixed' / (key + '.npz')
        np.savez_compressed(out, wavelength=wavelength, surface_flux=flux)
        cases[key] = dict(structure_input=str(path.relative_to(ROOT)), structure_sha256=digest(path),
            fixture_sha256=digest(out), config=json.loads(json.dumps(asdict(config))),
            request_fingerprint=model_request_fingerprint('DAH', replace(config, polarized_transfer='scalar-stokes-i'),
                                                          ModelData.default()),
            maximum_visible_field_megagauss=metadata['maximum_visible_field_megagauss'],
            domain_notes=metadata['domain_notes'],
            change_from_paper=difference(wavelength, flux, expected),
            paper_recovery=difference(wavelength, outputs['paper'][0], expected),
            atmosphere_reconverged=False, cold_start_qualification=False, tolerances_changed=False)
        (args.output / 'manifest.partial.json').write_text(json.dumps(cases, indent=2)+'\n')
    manifest = dict(description=__doc__, date='2026-10-25',
        reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'src/wd_spectra').rglob('*.py'))},
        python=platform.python_version(), numpy=np.__version__, cases=cases)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'manifest.partial.json').unlink()


if __name__ == '__main__':
    main()
