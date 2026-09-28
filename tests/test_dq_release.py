"""Fast release contracts; the multi-hour cold canary is separately marked."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from wd_spectra.models.dq import DQConfig, compute_dq
from wd_spectra._convergence import equilibrium_certificate
from wd_spectra._dq.validation import independent_grid, qualify_spectrum
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.spectrum import Spectrum


@pytest.mark.parametrize('kwargs', [dict(effective_temperature=0), dict(logg=np.inf),
    dict(log_carbon_to_helium=-1), dict(maximum_seconds=-1), dict(quality='quick')])
def test_invalid_requests_fail_before_worker_or_data(kwargs, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('worker started'))
    with pytest.raises(ValueError):
        compute_dq(DQConfig(**kwargs))


@pytest.mark.parametrize('kwargs', [dict(initial_atmosphere=object()), dict(relax_atmosphere=False)])
def test_previous_atmospheres_are_not_public_inputs(kwargs):
    with pytest.raises(TypeError, match='unexpected keyword'):
        compute_dq(**kwargs)


def certified_fixture():
    config = DQConfig()
    certificate = equilibrium_certificate(dict(
        radiative_equilibrium_solver_converged=True,
        maximum_all_depth_total_flux_residual=0.,
        maximum_relative_cell_energy_balance_residual=0.,
        maximum_unrestricted_log_temperature_correction=0.,
        temperature_correction_measured=True,
        electron_scattering_source_final_maximum_relative_residual=0.,
        lower_boundary_absorption_escape_bound=0.),
        flux_tolerance=.002, temperature_tolerance=.0002)
    atmosphere = SimpleNamespace(effective_temperature=config.effective_temperature,
        logg=config.logg, metadata={'carbon_abundance': config.log_carbon_to_helium,
        'dq_refractive_transfer': True, 'equilibrium_certificate': certificate})
    wave = independent_grid()
    flux = np.full_like(wave, STEFAN_BOLTZMANN*config.effective_temperature**4/(wave[-1]-wave[0]))
    return config, atmosphere, Spectrum(wave, flux, {})


def test_optional_unmeasured_diagnostics_do_not_prevent_qualification():
    config, atmosphere, spectrum = certified_fixture()
    certificate = atmosphere.metadata['equilibrium_certificate']
    for name in ('surface_flux', 'photospheric_flux'):
        assert name not in certificate['required_checks']
        assert not certificate['checks'][name]['measured']
        assert not certificate['checks'][name]['passed']
    assert qualify_spectrum(atmosphere, spectrum, config)['spectral_qualification']


def test_legacy_five_check_certificate_still_qualifies():
    config, atmosphere, spectrum = certified_fixture()
    certificate = atmosphere.metadata['equilibrium_certificate']
    required = certificate.pop('required_checks')
    certificate['checks'] = {name: certificate['checks'][name] for name in required}
    assert qualify_spectrum(atmosphere, spectrum, config)['spectral_qualification']


@pytest.mark.parametrize('name', ['all_depth_flux', 'local_energy',
    'temperature_stationarity', 'source_closure', 'boundary_screening'])
@pytest.mark.parametrize('change', ['missing', 'failed', 'unmeasured'])
def test_every_dq_check_remains_mandatory(name, change):
    config, atmosphere, spectrum = certified_fixture()
    certificate = atmosphere.metadata['equilibrium_certificate']
    # Even a certificate declaring a weaker profile cannot waive DQ's checks.
    certificate['required_checks'] = ['surface_flux']
    certificate['checks']['surface_flux'].update(
        value=0., measured=True, passed=True)
    if change == 'missing':
        certificate['checks'].pop(name)
    else:
        certificate['checks'][name]['passed' if change == 'failed' else 'measured'] = False
    with pytest.raises(ValueError, match='certificate'):
        qualify_spectrum(atmosphere, spectrum, config)


@pytest.mark.parametrize('name', ['surface_flux', 'photospheric_flux', 'missing_check'])
def test_additional_declared_required_checks_cannot_be_ignored(name):
    config, atmosphere, spectrum = certified_fixture()
    certificate = atmosphere.metadata['equilibrium_certificate']
    certificate['required_checks'] = (*certificate['required_checks'], name)
    with pytest.raises(ValueError, match='certificate'):
        qualify_spectrum(atmosphere, spectrum, config)


def test_independent_grid_and_absolute_gate():
    config, atmosphere, spectrum = certified_fixture()
    assert len(spectrum.wavelength_angstrom) == 218520
    assert qualify_spectrum(atmosphere, spectrum, config)['spectral_qualification']
    with pytest.raises(ValueError, match='bolometric'):
        qualify_spectrum(atmosphere, Spectrum(spectrum.wavelength_angstrom,
            1.003*spectrum.surface_flux_lambda, {}), config)
    with pytest.raises(ValueError, match='independent'):
        qualify_spectrum(atmosphere, Spectrum(spectrum.wavelength_angstrom[::2],
            spectrum.surface_flux_lambda[::2], {}), config)
    atmosphere.metadata['equilibrium_certificate']['checks'].pop('local_energy')
    with pytest.raises(ValueError, match='certificate'):
        qualify_spectrum(atmosphere, spectrum, config)


def test_completed_flag_is_not_spectral_qualification():
    config, atmosphere, spectrum = certified_fixture()
    atmosphere.metadata['status'] = 'completed'
    atmosphere.metadata['equilibrium_certificate']['verified'] = False
    with pytest.raises(ValueError, match='certificate'):
        qualify_spectrum(atmosphere, spectrum, config)


def test_worker_arguments_have_no_saved_state_or_structure_grid(monkeypatch, tmp_path):
    from wd_spectra.models import dq
    from wd_spectra._dq import data
    captured = {}
    monkeypatch.setattr(data, 'validate_data', lambda: {})
    monkeypatch.setattr(dq, '_load_result', lambda *a: 'qualified')
    monkeypatch.setattr(dq.subprocess, 'run', lambda command, **options:
        captured.update(command=command, options=options))
    config = DQConfig(7123., 8.07, -5.13)
    with pytest.warns(dq.DQApproximationWarning):
        assert compute_dq(config, [5000, 5100], output_directory=tmp_path/'new') == 'qualified'
    command = captured['command']
    assert command[:3] == [sys.executable, '-m', 'wd_spectra._dq.worker']
    assert command[command.index('--teff')+1] == '7123.0'
    assert not {'--iterate', '--grid', '--case', '--checkpoint'} & set(command)
    assert json.loads(captured['options']['input']) == {'output_wavelength': [5000., 5100.]}
    assert 'research' not in captured['options']['env']['PYTHONPATH']


def test_package_policy_restores_shared_hooks_on_failure(tmp_path):
    from wd_spectra import adaptive_structure as shared
    from wd_spectra._dq import runtime, automatic_conditioning as controller
    from wd_spectra._dq import dq_explicit_gradient as gradient
    from wd_spectra._dq import nonlinear_material_model as nm
    original = (shared.solve_trust_region_newton, controller.thermal_condition,
                gradient.ExplicitGradientSystem, nm.local_model)
    with pytest.raises(RuntimeError, match='deliberate'):
        with runtime.numerical_policy(tmp_path):
            assert controller.THERMAL_RELATIVE_NORM_LIMIT.get() == .8
            raise RuntimeError('deliberate interruption')
    assert original == (shared.solve_trust_region_newton, controller.thermal_condition,
                        gradient.ExplicitGradientSystem, nm.local_model)


def test_constitutive_data_are_packaged_and_checksum_pinned():
    from wd_spectra._dq.data import validate_data
    assert len(validate_data()['sha256']) == 7


def test_saved_worker_output_restores_helium_populations(tmp_path):
    """Real EOS/serialization regression, NOT evidence of cold convergence.

    The certificate and flat bolometric spectrum here are synthetic reader
    inputs. Only the separately marked canary proves a new cold calculation.
    """
    from dataclasses import asdict, replace
    from wd_spectra import gray_helium_atmosphere
    from wd_spectra.models.common import ModelData, ModelResult, save_model_result
    from wd_spectra.models.dq import _load_result
    from wd_spectra._dq.runtime import make_material
    from wd_spectra._dq.provenance import digest

    root = Path(__file__).parent/'data/dq_regressions'
    config = DQConfig(**json.loads((root/'manifest.json').read_text())['parameters'])
    data = ModelData.default()
    with np.load(root/'j1235-fixed.npz', allow_pickle=False) as saved:
        seed = gray_helium_atmosphere(config.effective_temperature, config.logg,
                                     n_depth=len(saved['temperature']))
        state = replace(seed, **{key: saved[key].copy() for key in
            ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')})
    material = make_material(config, data, tmp_path)
    atmosphere = material.chemistry(state)[0]
    _, synthetic, _ = certified_fixture()
    atmosphere = replace(atmosphere, metadata={**atmosphere.metadata,
        'dq_refractive_transfer': True,
        'carbon_abundance': config.log_carbon_to_helium,
        'equilibrium_certificate': synthetic.metadata['equilibrium_certificate']})
    wave = independent_grid()
    flux = np.full_like(wave, STEFAN_BOLTZMANN*config.effective_temperature**4/(wave[-1]-wave[0]))
    spectrum = Spectrum(wave, flux, {})
    save_model_result(ModelResult('DQ', atmosphere, spectrum, config, {}), tmp_path/'model')
    np.savez_compressed(tmp_path/'independent-spectrum.npz', wavelength=wave, flux=flux)
    report = dict(requested_parameters={key: getattr(config, key) for key in
        ('effective_temperature', 'logg', 'log_carbon_to_helium')},
        physical_config=asdict(material.config), config=asdict(config), status='completed', cold_start=True,
        spectral_qualification=True, source_consistency_verified=True,
        independent_spectrum_sha256=digest(tmp_path/'independent-spectrum.npz'))
    (tmp_path/'run.json').write_text(json.dumps(report))

    result = _load_result(tmp_path, config, data)
    assert result.atmosphere.helium_lte_state is not None
    for field in ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth',
                  'mass_density', 'electron_density'):
        np.testing.assert_array_equal(getattr(result.atmosphere, field), getattr(atmosphere, field))
    for field in ('neutral_he_density', 'electron_density', 'helium_nuclei_density'):
        np.testing.assert_array_equal(getattr(result.atmosphere.helium_lte_state, field),
                                      getattr(atmosphere.helium_lte_state, field))
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda, flux)


@pytest.mark.parametrize('strict', [False, True])
def test_run_model_dispatches_dq_without_legacy_research_worker(monkeypatch, tmp_path, strict):
    """API wiring only: the mocked atmosphere is not convergence evidence."""
    from wd_spectra import ModelResult, gray_helium_atmosphere
    from wd_spectra.models import automatic

    config = DQConfig(9347., 8.041, -4.107)
    calls = []

    def compute(request, **kwargs):
        assert request is config
        calls.append(kwargs)
        return ModelResult('DQ', gray_helium_atmosphere(9347., 8.041, n_depth=8),
            Spectrum(np.array([4000., 5000.]), np.ones(2), {}), config,
            {'atmosphere_convergence_status': 'converged', 'spectral_qualification': True})

    monkeypatch.setattr(automatic, 'compute_dq', compute)
    monkeypatch.setattr(automatic, '_cool_commands',
                        lambda *a: pytest.fail('legacy research worker selected'))
    run = automatic.run_model(config, tmp_path/'new', require_convergence=strict)
    assert run.selection.workflow == 'dq' and run.convergence_verified
    assert len(calls) == 1
    assert set(calls[0]) == {'data', 'output_directory'}
    assert calls[0]['output_directory'] == tmp_path/'new/worker'
    report = json.loads((tmp_path/'new/model-run.json').read_text())
    assert report['cold_start'] and report['initial_checkpoint'] is None
    assert report['status'] == 'completed' and report['commands'] == []


def test_run_model_retains_dq_failure_without_trying_another_workflow(monkeypatch, tmp_path):
    from wd_spectra.models import automatic

    def failed(*args, **kwargs):
        raise RuntimeError('independent spectrum did not qualify')

    monkeypatch.setattr(automatic, 'compute_dq', failed)
    monkeypatch.setattr(automatic, '_cool_commands',
                        lambda *a: pytest.fail('fallback workflow selected'))
    with pytest.raises(RuntimeError, match='did not qualify'):
        automatic.run_model(DQConfig(), tmp_path/'new')
    report = json.loads((tmp_path/'new/model-run.json').read_text())
    assert report['status'] == 'failed'
    assert not report['physics_changed_after_failure']
    assert not (tmp_path/'new/spectrum.txt').exists()


@pytest.mark.parametrize('change', ['unmeasured', 'loose', 'too_large', 'wrong_carbon', 'nonrefractive'])
def test_certificate_cannot_be_qualified_by_its_verified_flag_alone(change):
    config, atmosphere, spectrum = certified_fixture()
    check = atmosphere.metadata['equilibrium_certificate']['checks']['local_energy']
    if change == 'unmeasured':
        check['measured'] = False
    elif change == 'loose':
        check['tolerance'] = .1
    elif change == 'too_large':
        check['value'] = .003
    elif change == 'wrong_carbon':
        atmosphere.metadata['carbon_abundance'] += .1
    else:
        atmosphere.metadata['dq_refractive_transfer'] = False
    with pytest.raises(ValueError):
        qualify_spectrum(atmosphere, spectrum, config)


@pytest.mark.canary
def test_j1235_public_true_cold_and_independent_spectrum(tmp_path):
    """Hours, not seconds. The solver receives no regression fixture."""
    output = Path(os.environ.get('OPENWD_TEST_ARTIFACTS', tmp_path))/'j1235'
    result = compute_dq(DQConfig(9347., 8.041, -4.107), output_directory=output)
    assert result.atmosphere.metadata['equilibrium_certificate']['verified']
    assert result.metadata['cold_start']
    assert result.metadata['spectral_qualification']
    assert len(result.spectrum.wavelength_angstrom) == 218520
    assert abs(result.spectrum.bolometric_flux/(STEFAN_BOLTZMANN*9347.**4)-1) <= .002
