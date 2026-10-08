"""Slow public cold-start acceptance at fixed observed/control parameters.

Run explicitly with -m canary using the bundled data (or OPENWD_DATA). Each case creates a new model;
no saved atmosphere, population, Jacobian or research proposal is supplied.
These tests can take hours and are excluded from the ordinary fast suite.
"""
from dataclasses import asdict
import json

import pytest

from wd_spectra import DAOConfig, DOConfig, ModelData, run_model


CASES = [
    pytest.param(DOConfig(effective_temperature=50000.), id='do-50000-standard'),
    pytest.param(DOConfig(effective_temperature=60000.), id='do-60000-standard'),
    pytest.param(DOConfig(effective_temperature=70000.), id='do-70000-standard'),
    pytest.param(DAOConfig(effective_temperature=60000., log_hydrogen_to_helium=2.),
                 id='dao-60000-standard'),
    pytest.param(DAOConfig(effective_temperature=40204., logg=7.82,
                          log_hydrogen_to_helium=6.), id='gd153-standard'),
    pytest.param(DAOConfig(effective_temperature=40204., logg=7.82,
                          log_hydrogen_to_helium=6., quality='production',
                          maximum_helium_ii_level=8, maximum_hydrogen_level=8),
                 id='gd153-production-h8'),
    pytest.param(DAOConfig(effective_temperature=40204., logg=7.82,
                          log_hydrogen_to_helium=6., quality='production',
                          maximum_helium_ii_level=8, maximum_hydrogen_level=20),
                 id='gd153-production-h20'),
]


@pytest.fixture(scope='module')
def cold_data():
    data = ModelData.default()
    for path in (data.ccc_hydrogen_collisions, data.tlusty_source,
                 data.tlusty_helium_atom, data.helium_ii_stark,
                 data.cache/'helium-stark/Tremblay26.txt'):
        assert path.is_file(), f'Missing explicitly selected cold-canary data: {path}'
    return data


@pytest.mark.canary
@pytest.mark.parametrize('config', CASES)
def test_hot_public_model_converges_from_cold(config, cold_data, tmp_path):
    output = tmp_path/'fresh-model'
    result = run_model(config, output, data=cold_data, require_convergence=True)
    assert result.convergence_verified
    request = json.loads((output/'model-run.json').read_text())
    saved = json.loads((output/'metadata.json').read_text())
    assert request['cold_start'] is True
    assert request['initial_checkpoint'] is None
    assert request['request'] == asdict(config)
    assert saved['config'] == asdict(config)
    assert saved['model_metadata']['cold_start'] is True
    assert saved['model_metadata']['initialization']['previous_model_supplied'] is False
    atmosphere = saved['atmosphere_metadata']
    assert atmosphere['nlte_populations_converged'] is True
    assert atmosphere['nlte_maximum_relative_population_change'] < 1e-4
    certificate = atmosphere['equilibrium_certificate']
    assert certificate['verified'] is True
    tolerances = dict(all_depth_flux=3e-3, local_energy=3e-3,
                      temperature_stationarity=3e-4, source_closure=1e-6,
                      boundary_screening=3e-3)
    for name, tolerance in tolerances.items():
        check = certificate['checks'][name]
        assert check['tolerance'] == tolerance
        assert check['measured'] is True and check['passed'] is True
        assert check['value'] < tolerance
