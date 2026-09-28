"""Cold-start refractive DQ models, isolated from other stellar solvers."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
import numpy as np
from ..atmosphere import Atmosphere
from ..spectrum import Spectrum
from .common import ModelData, ModelResult, validate_wavelength


class DQApproximationWarning(RuntimeWarning):
    """Numerical qualification does not establish agreement with observations."""


@dataclass(frozen=True)
class DQConfig:
    """Hydrogen-free, nonmagnetic classical DQ; log10 N(C nuclei)/N(He nuclei).

    Standard uses fourfold-thinned structure sampling and unchanged full-grid
    final synthesis. This speed/accuracy tradeoff is not a pointwise 1% bound.
    The wall-time budget includes convergence and independent final synthesis.
    """
    effective_temperature: float = 8000.
    logg: float = 8.
    log_carbon_to_helium: float = -5.
    quality: str = 'standard'
    maximum_seconds: float = 28800.
    include_c2_ca: bool = True


def validate_config(config):
    if not isinstance(config, DQConfig):
        raise TypeError('Expected DQConfig')
    numbers = [config.effective_temperature, config.logg,
               config.log_carbon_to_helium, config.maximum_seconds]
    if not np.all(np.isfinite(numbers)):
        raise ValueError('DQ parameters and budget must be finite')
    if config.effective_temperature <= 0 or config.maximum_seconds <= 0:
        raise ValueError('DQ temperature and time budget must be positive')
    if not -12 <= config.log_carbon_to_helium <= -3:
        raise ValueError('Trace-carbon DQ requires -12 <= log(C/He) <= -3')
    if config.quality != 'standard':
        raise ValueError('Refractive DQ currently supports only quality="standard"')
    if type(config.include_c2_ca) is not bool:
        raise ValueError('include_c2_ca must be boolean (False is a diagnostic opacity omission)')


def warn_dq_approximation():
    warnings.warn('DQ uses approximate dense-helium and molecular profiles. '
        'Convergence is not a guarantee of agreement with observations; '
        'this release is not qualified for precision abundance fitting.',
        DQApproximationWarning, stacklevel=3)


def _load_result(directory, config, data):
    """Read this worker's new output, never an atmosphere solver input."""
    from .._dq.base import DQConfig as PhysicalConfig, DQMaterial
    from ..carbon_molecular import read_c2_cross_section_table, _reos_host_state
    from ..eos import hummer_mihalas_helium_lte
    from .._dq.provenance import digest
    from .._dq.validation import qualify_spectrum

    report = json.loads((directory/'run.json').read_text())
    parameters = {key: getattr(config, key) for key in
                  ('effective_temperature', 'logg', 'log_carbon_to_helium')}
    if report.get('requested_parameters') != parameters:
        raise RuntimeError('DQ output parameters differ from the request')
    if report.get('config', {}).get('include_c2_ca') is not config.include_c2_ca:
        raise RuntimeError('DQ output C–A opacity selection differs from the request')
    if any(report['physical_config'].get(k) != v for k, v in parameters.items()):
        raise RuntimeError('DQ physical parameters differ from the request')
    if (report['status'] != 'completed' or not report.get('spectral_qualification')
            or not report.get('source_consistency_verified') or not report.get('cold_start')):
        raise RuntimeError(f'DQ worker did not produce a qualified cold result: {directory}')
    if digest(directory/'independent-spectrum.npz') != report['independent_spectrum_sha256']:
        raise RuntimeError('DQ spectrum checksum mismatch')
    meta = json.loads((directory/'model/metadata.json').read_text())
    physical = PhysicalConfig(**report['physical_config'])
    material = DQMaterial(physical, data, read_c2_cross_section_table(physical.c2_table_path))
    with np.load(directory/'model/atmosphere.npz', allow_pickle=False) as saved:
        if (float(saved['effective_temperature']) != config.effective_temperature or
                float(saved['logg']) != config.logg):
            raise RuntimeError('DQ stored atmosphere parameters differ from the request')
        # Model files store bulk state, not the helium population object.
        # Restore the host at the saved T/P before closing the He/C mixture;
        # this is output deserialization, not an atmosphere iteration.
        helium = (_reos_host_state(saved['temperature'], saved['gas_pressure'],
                                   material.helium_reos3)
                  if material.helium_reos3 is not None else
                  hummer_mihalas_helium_lte(saved['temperature'], saved['gas_pressure'],
                                          correlated_microfields=True, neutral_radius_scale=.5))
        atmosphere = Atmosphere(config.effective_temperature, config.logg,
            **{name: saved[name].copy() for name in ('rosseland_optical_depth',
               'column_mass', 'temperature', 'gas_pressure', 'mass_density', 'electron_density')},
            neutral_h_density=np.zeros_like(saved['temperature']),
            proton_density=np.zeros_like(saved['temperature']),
            helium_lte_state=helium,
            metadata=meta['atmosphere_metadata'])
    reclosed = material.chemistry(atmosphere)[0]
    for field in ('mass_density', 'electron_density'):
        np.testing.assert_allclose(getattr(reclosed, field), getattr(atmosphere, field), rtol=2e-13)
    wave, flux = np.loadtxt(directory/'model/spectrum.txt', unpack=True)
    spectrum = Spectrum(wave, flux, meta['spectrum_metadata'])
    with np.load(directory/'independent-spectrum.npz', allow_pickle=False) as fine:
        qualify_spectrum(reclosed, Spectrum(fine['wavelength'], fine['flux'], {}), config)
    return ModelResult('DQ', reclosed, spectrum, config,
                       {**meta['model_metadata'], 'run_directory': str(directory)})


def compute_dq(config=DQConfig(), wavelength=None, *, data=None,
               output_directory=None):
    """Return a qualified cold atmosphere and unscaled surface-flux spectrum.

    This can take hours. The worker prints progress and retains checkpoints
    in a new output_directory (or an announced temporary directory). A user
    wavelength grid changes only final output synthesis, never the structure
    grid or the required independent 218520-point bolometric qualification.
    Failed qualification raises; exploratory spectra are not returned.
    """
    validate_config(config)
    wave = None if wavelength is None else validate_wavelength(wavelength)
    data = ModelData.default() if data is None else data
    directory = (Path(tempfile.mkdtemp(prefix='openwd-dq-'))/'run'
                 if output_directory is None else Path(output_directory).expanduser().resolve())
    if directory.exists():
        raise FileExistsError(f'Choose a new DQ output directory: {directory}')
    from .._dq.data import validate_data
    validate_data()
    warn_dq_approximation()
    print(f'OpenWD DQ: isolated cold run; diagnostics retained in {directory}', flush=True)
    command = [sys.executable, '-m', 'wd_spectra._dq.worker',
               '--teff', str(config.effective_temperature), '--logg', str(config.logg),
               '--log-carbon-to-helium', str(config.log_carbon_to_helium),
               '--seconds', str(config.maximum_seconds), '--output', str(directory),
               '--read-output-grid']
    if not config.include_c2_ca:
        command.append('--no-c2-ca')
    environment = os.environ.copy()
    environment['OPENWD_DATA'] = str(data.root)
    # Installed package root, never a research/results directory.
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    payload = json.dumps({'output_wavelength': None if wave is None else wave.tolist()})
    try:
        subprocess.run(command, env=environment, input=payload, text=True, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'DQ cold run failed; accepted states and diagnostics retained in {directory}') from error
    return _load_result(directory, config, data)
