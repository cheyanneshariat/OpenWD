"""Offline capture and attribution of the post-PR38 metal population repair.

Existing controls/structures are never overwritten, and tests/CI never run
this script. Every star retains its stored configuration and wavelength grid.
The archived PR38 output must be recovered at its original tolerance before
recording the new output. The manifest reports separate component reversions;
these nonlinear differences are attribution diagnostics, not additive fluxes.
No saved atmosphere is reconverged or certified by this fixed-state audit.

Usage: PYTHONPATH=src python research/capture_metal_population_controls.py OUTPUT
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import scipy

from wd_spectra import compiled_backend_available
from wd_spectra._compat import trapezoid
from wd_spectra.models import AtmosphereConvergenceWarning, DZConfig, compute_dz, load_atmosphere_checkpoint
from wd_spectra.models.common import ModelData, model_request_fingerprint
from metal_population_control_legacy import previous_metal_opacity_populations

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests/data'
PREVIOUS = DATA / 'approved_regressions/cool_dz_release_2026_10_23/fixed'
BANDS = ((1000, 3000), (3000, 3500), (3500, 4000), (4000, 4500),
         (4500, 5500), (5500, 7000), (7000, 10000), (10000, 100000))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'fixed').mkdir()
    assert compiled_backend_available(), 'use the compiled backend for the saved controls'
    cases = {}
    for case in ('dz-pg1225', 'dz-j0738'):
        path = DATA / 'spectral_regressions' / (case + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            config = DZConfig(**json.loads(str(saved['config_json'])))
            wavelength = saved['wavelength'].copy()
        atmosphere = load_atmosphere_checkpoint(path, config.effective_temperature, config.logg, 'helium',
            include_molecules=False, include_negative_hydrogen=False,
            log_hydrogen_to_helium=getattr(config, 'log_hydrogen_to_helium', None),
            trihydrogen_ion_partition_model=None)
        with np.load(PREVIOUS / (case + '.npz'), allow_pickle=False) as prior:
            np.testing.assert_array_equal(wavelength, prior['wavelength'])
            expected = prior['surface_flux'].copy()
        outputs = {}
        for label, components in (('current', ()), ('previous', ('lines', 'verner', 'oxygen', 'topbase')),
                                   ('previous-lines', ('lines',)), ('previous-verner', ('verner',)),
                                   ('previous-oxygen', ('oxygen',)), ('previous-topbase', ('topbase',))):
            with (previous_metal_opacity_populations(components=components) if components else nullcontext()):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AtmosphereConvergenceWarning)
                    result = compute_dz(config, wavelength, initial_atmosphere=atmosphere, relax_atmosphere=False)
            assert result.spectrum.metadata['source_converged']
            assert result.spectrum.metadata['independent_radiation_scaled_source_error'] < 1e-10
            flux = result.spectrum.surface_flux_lambda
            assert np.all(np.isfinite(flux)) and np.all(flux > 0.)
            outputs[label] = flux.copy()
            print(case, label, flush=True)
        np.testing.assert_allclose(outputs['previous'], expected, rtol=2e-6, atol=1e-12*np.max(expected))
        out = args.output / 'fixed' / (case + '.npz')
        np.savez_compressed(out, wavelength=wavelength, surface_flux=outputs['current'])
        audit = args.output / (case + '-attribution.npz')
        np.savez_compressed(audit, wavelength=wavelength, **outputs)
        cases[case] = dict(structure_input=str(path.relative_to(ROOT)), structure_sha256=digest(path),
            previous_approved=str((PREVIOUS / (case + '.npz')).relative_to(ROOT)),
            previous_sha256=digest(PREVIOUS / (case + '.npz')), fixture_sha256=digest(out),
            config=json.loads(json.dumps(asdict(config))),
            request_fingerprint=model_request_fingerprint('DZ', config, ModelData.default()),
            change_from_previous=difference(wavelength, outputs['current'], expected),
            historical_recovery=difference(wavelength, outputs['previous'], expected),
            component_reversions={label: difference(wavelength, flux, outputs['current'])
                                  for label, flux in outputs.items() if label.startswith('previous-')},
            atmosphere_reconverged=False, cold_start_qualification=False, tolerances_changed=False)
        (args.output / 'manifest.partial.json').write_text(json.dumps(cases, indent=2)+'\n')
    manifest = dict(description=__doc__, date='2026-10-24',
        reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      sorted((ROOT/'src/wd_spectra').rglob('*.py'))},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, cases=cases)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'manifest.partial.json').unlink()


if __name__ == '__main__':
    main()
