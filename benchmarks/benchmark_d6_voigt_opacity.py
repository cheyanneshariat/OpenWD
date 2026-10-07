"""Compare released and native frequency-Voigt finishes in retained D6 opacity.

This is a fixed-atmosphere benchmark, with explicitly selectable retained
depths and line budget. It never constructs or iterates an atmosphere.
The released compiled-manifold function is read from a separate checkout;
both modes use the same loaded extension, atomic data, state, lines and grid.
"""
from __future__ import annotations

import argparse
import ast
import cProfile
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.suffix not in ('.pyc', '.so', '.pyd')}


def function_source(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return node, ast.get_source_segment(source, node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--reference-root', required=True, type=Path)
    parser.add_argument('--atmosphere', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--depths', type=int, default=8)
    parser.add_argument('--lines', type=int, default=500)
    parser.add_argument('--continuum', type=int, default=120)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', choices=('each', 'shared'), default='each',
                        help='warm each opacity path or share a released material warm-up')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--radiation', action='store_true')
    args = parser.parse_args()
    if min(args.depths, args.lines, args.continuum, args.repeat) < 1:
        parser.error('resolution and repeat counts must be positive')
    if args.output.exists():
        parser.error('choose a fresh output directory to preserve earlier evidence')
    args.source_root = args.source_root.resolve()
    args.reference_root = args.reference_root.resolve()
    args.atmosphere = args.atmosphere.resolve()
    args.output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root / 'src'))
    import numpy as np
    import scipy
    from wd_spectra import d6, metals
    from wd_spectra.atmosphere import Atmosphere
    from wd_spectra.constants import STEFAN_BOLTZMANN
    from wd_spectra.models.common import ModelData
    from wd_spectra.models.d6 import _atomic_inputs

    if Path(metals.__file__).resolve() != args.source_root / 'src/wd_spectra/metals.py':
        raise RuntimeError('imported another checkout')
    native = metals._rt
    for name in ('frequency_voigt_profile', 'stark_frequency_profile_finish'):
        if not hasattr(native, name):
            raise RuntimeError('rebuild the optional native extension: ' + name)
    native_path = Path(native.__file__).resolve()
    if args.source_root not in native_path.parents:
        raise RuntimeError('native extension comes from another checkout')
    reference_file = args.reference_root / 'src/wd_spectra/metals.py'
    current_file = Path(metals.__file__)
    for name in ('_voigt_profile_per_angstrom', '_humlicek_w4', '_stark_impact_profile'):
        if function_source(reference_file, name)[1] != function_source(current_file, name)[1]:
            raise RuntimeError('benchmark requires unchanged physical reference: ' + name)
    node, reference_source = function_source(reference_file, '_compiled_manifold_profile')
    reference_scope = dict(metals.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(reference_file), 'exec'),
         reference_scope)
    modes = {'released': reference_scope['_compiled_manifold_profile'],
             'native': metals._compiled_manifold_profile}
    original = metals._compiled_manifold_profile
    roots = {'source': args.source_root / 'src/wd_spectra',
             'csrc': args.source_root / 'csrc',
             'reference': args.reference_root / 'src/wd_spectra'}
    before = {name: inventory(root) for name, root in roots.items()}
    input_sha = digest(args.atmosphere)
    native_sha = digest(native_path)
    report = {'scope': 'retained-state D6 opacity including result copies; no atmosphere solve or cold start',
              'status': 'running', 'benchmark_sha256': digest(__file__),
              'source_root': str(args.source_root), 'reference_root': str(args.reference_root),
              'atmosphere': str(args.atmosphere), 'atmosphere_sha256': input_sha,
              'native_path': str(native_path), 'native_sha256': native_sha,
              'released_function_sha256': hashlib.sha256(reference_source.encode()).hexdigest(),
              'runtime': {'python': sys.version, 'executable': sys.executable,
                          'numpy': np.__version__, 'scipy': scipy.__version__,
                          'platform': platform.platform(),
                          'thread_environment': {name: os.environ.get(name) for name in (
                              'OPENWD_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS')}},
              'requested_resolution': vars(args) | {}, 'runs': [], 'warmups': [],
              'provenance_before': before}
    report['requested_resolution'] = {k: str(v) if isinstance(v, Path) else v
                                      for k, v in vars(args).items()}

    def save():
        temporary = args.output / 'report.json.writing'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'report.json')

    save()
    with np.load(args.atmosphere, allow_pickle=False) as data:
        metadata = json.loads(str(data['atmosphere_metadata_json']))
        abundance = metadata['metal_abundances']
        n_original = len(data['temperature'])
        if args.depths > n_original:
            raise ValueError('cannot retain more depths than the archived atmosphere')
        indices = np.unique(np.rint(np.linspace(0, n_original - 1, args.depths)).astype(int))
        zeros = np.zeros(len(indices))
        atmosphere = Atmosphere(float(data['effective_temperature']), float(data['logg']),
            data['rosseland_optical_depth'][indices], data['column_mass'][indices],
            data['temperature'][indices], data['gas_pressure'][indices],
            data['mass_density'][indices], zeros, zeros, data['electron_density'][indices], {})
    database, photo, topbase = _atomic_inputs(ModelData.default(), tuple(abundance))
    selected = d6._structure_metal_lines(atmosphere, database, abundance, 1e-4,
        args.lines, reference_element='C')
    keys = tuple((ion.element, ion.charge, line.lower_index, line.upper_index)
                 for ion, line in selected)
    wave = d6._structure_wavelength_grid(atmosphere, database, abundance,
        args.continuum, 1e-4, args.lines, reference_element='C', selected_lines=selected)
    opacity = d6.d6_structure_opacity_function(wave, database, photo,
        structure_line_transition_keys=keys, topbase_photoionization_database=topbase,
        minimum_metal_oscillator_strength=1e-4, include_linear_stark_quasistatic=True,
        linear_stark_profile='manifold', include_oxygen_i_series_stark=True,
        metal_series_pseudocontinuum_elements=('O', 'Mg', 'C'),
        profile_edge_optical_depth=1e-3)
    state = d6.bulk_metal_lte_state(atmosphere, database, abundance, reference_element='C')
    fixtures = {'wave': wave, 'depth_indices': indices}
    for field in fields(atmosphere):
        value = getattr(atmosphere, field.name)
        if isinstance(value, np.ndarray):
            fixtures['atmosphere_' + field.name] = value
    np.savez(args.output / 'fixture.npz', **fixtures)
    report.update(depths=atmosphere.n_depth, original_depths=n_original,
                  depth_indices=indices.tolist(), wavelengths=len(wave), selected_lines=len(keys),
                  line_transition_keys=keys, abundances=abundance,
                  fixture_sha256=digest(args.output / 'fixture.npz'))
    values = {}
    first_reference = None

    def evaluate(mode):
        metals._compiled_manifold_profile = modes[mode]
        return tuple(np.asarray(a).copy() for a in opacity(atmosphere, state))

    try:
        for mode in (modes if args.warmup == 'each' else ('released',)):
            started = time.perf_counter()
            evaluate(mode)
            report['warmups'].append({'mode': mode, 'seconds': time.perf_counter() - started})
            print('WARMUP', mode, report['warmups'][-1]['seconds'], flush=True)
            save()
        if args.warmup == 'shared':
            # The C evaluator/finish have no persistent state. Prime their
            # entrypoints on small buffers; the common material/atomic/FFT
            # caches were populated by the released full-opacity warm-up.
            probe = np.array([4000., 4333.17, 4500., 5000.])
            output = np.empty_like(probe)
            started = time.perf_counter()
            native.frequency_voigt_profile(probe, 4333.17, .03, .005, output)
            native.stark_frequency_profile_finish(probe, np.ones(7), np.empty(0),
                4333.17, .03, .005, .1, 1., .3, .2, .3, .9, output)
            if not np.all(np.isfinite(output)):
                raise AssertionError('native API warm-up failed')
            report['warmups'].append({'mode': 'native-api',
                'seconds': time.perf_counter() - started,
                'scope': 'small buffers; common material caches warmed by released opacity'})
            save()
        for repeat in range(args.repeat):
            for mode in (('released', 'native') if repeat % 2 == 0 else ('native', 'released')):
                cpu = time.process_time()
                started = time.perf_counter()
                values[mode] = evaluate(mode)
                entry = {'repeat': repeat, 'mode': mode,
                         'wall_seconds': time.perf_counter() - started,
                         'cpu_seconds': time.process_time() - cpu}
                report['runs'].append(entry)
                if first_reference is None:
                    first_reference = values['released']
                checks = []
                for base, actual in zip(first_reference, values[mode]):
                    peak = max(float(np.max(np.abs(base))), np.finfo(float).tiny)
                    checks.append({'passed': bool(np.all(np.isfinite(actual)) and
                        np.all(actual >= 0) and np.allclose(actual, base, rtol=2e-10,
                                                           atol=1e-14 * peak)),
                        'array_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
                entry['numeric_checks'] = checks
                print('RUN', entry, flush=True)
                save()
                if not all(c['passed'] for c in checks):
                    np.savez(args.output / 'failed-trial.npz',
                             absorption=values[mode][0], scattering=values[mode][1])
                    raise AssertionError('a repeated opacity evaluation changed')
        comparisons = {}
        for index, name in enumerate(('absorption', 'scattering')):
            base, actual = values['released'][index], values['native'][index]
            peak = max(float(np.max(np.abs(base))), np.finfo(float).tiny)
            passed = bool(np.all(np.isfinite(base)) and np.all(np.isfinite(actual))
                          and np.all(base >= 0) and np.all(actual >= 0)
                          and np.allclose(actual, base, rtol=2e-10, atol=1e-14 * peak))
            comparisons[name] = {'passed': passed, 'bitwise_equal': bool(np.array_equal(base, actual)),
                'maximum_difference_relative_to_peak': float(np.max(np.abs(actual - base)) / peak),
                'maximum_pointwise_relative_with_1e20_peak_floor': float(np.max(
                    np.abs(actual - base) / np.maximum(np.abs(base), 1e-20 * peak))),
                'rtol': 2e-10, 'atol_relative_to_peak': 1e-14}
        report['comparisons'] = comparisons
        np.savez(args.output / 'outputs.npz', wavelength=wave,
            released_absorption=values['released'][0], native_absorption=values['native'][0],
            released_scattering=values['released'][1], native_scattering=values['native'][1])
        report['outputs_sha256'] = digest(args.output / 'outputs.npz')
        if not all(c['passed'] for c in comparisons.values()):
            raise AssertionError('opacity equivalence failed')
        medians = {mode: {metric: statistics.median(r[metric] for r in report['runs']
                       if r['mode'] == mode) for metric in ('wall_seconds', 'cpu_seconds')}
                   for mode in modes}
        report['medians'] = medians
        report['wall_speedup'] = medians['released']['wall_seconds'] / medians['native']['wall_seconds']
        report['wall_time_reduction_percent'] = 100 * (1 - 1 / report['wall_speedup'])
        if args.radiation:
            from wd_spectra.opacity import optical_depth_from_mass_opacity
            from wd_spectra.spectrum import planck_lambda_angstrom
            from wd_spectra.radiative_transfer import coherent_scattering_feautrier_field
            from wd_spectra._compat import trapezoid
            fluxes = {}
            planck = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
            for mode, (absorption, scattering) in values.items():
                tau = optical_depth_from_mass_opacity(atmosphere.column_mass, absorption + scattering)
                _, radiation = coherent_scattering_feautrier_field(tau, planck, absorption,
                    np.broadcast_to(scattering, absorption.shape), n_angle=3)
                fluxes[mode] = trapezoid(radiation.interface_flux, wave, axis=0)
            change = float(np.max(np.abs(fluxes['native'] - fluxes['released'])) /
                           (STEFAN_BOLTZMANN * atmosphere.effective_temperature ** 4))
            report['radiation'] = {'maximum_depth_flux_change_in_stellar_flux': change,
                                   'tolerance': 1e-8, 'passed': change <= 1e-8}
            np.savez(args.output / 'radiation.npz', **fluxes)
            if change > 1e-8:
                raise AssertionError('fixed-state flux consistency failed')
        if args.profile:
            for mode in modes:
                profiler = cProfile.Profile()
                with profiler:
                    evaluate(mode)
                profiler.dump_stats(str(args.output / (mode + '.prof')))
                with (args.output / (mode + '-profile.txt')).open('w') as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(50)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        metals._compiled_manifold_profile = original
        after = {name: inventory(root) for name, root in roots.items()}
        report['provenance_after'] = after
        report['source_unchanged'] = before == after
        report['native_unchanged'] = native_sha == digest(native_path)
        report['atmosphere_unchanged'] = input_sha == digest(args.atmosphere)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == 'darwin' else 1024)
        save()
    if not all(report[k] for k in ('source_unchanged', 'native_unchanged', 'atmosphere_unchanged')):
        raise AssertionError('inputs changed during benchmark')
    report['status'] = 'completed'
    save()
    print('COMPLETE', json.dumps({'speedup': report['wall_speedup'],
          'comparisons': report['comparisons']}), flush=True)


if __name__ == '__main__':
    main()
