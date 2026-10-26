"""Versioned population controls retain their exact inputs and predecessors."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import DZConfig

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'tests/data/approved_regressions/metal_populations_2026_10_24'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('case', ['dz-pg1225', 'dz-j0738'])
def test_population_controls_retain_their_input_and_approval_chain(case):
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    record = manifest['cases'][case]
    source = ROOT / record['structure_input']
    previous = ROOT / record['previous_approved']
    output = DIRECTORY / 'fixed' / (case + '.npz')
    assert digest(source) == record['structure_sha256']
    assert digest(previous) == record['previous_sha256']
    assert digest(output) == record['fixture_sha256']
    assert record['tolerances_changed'] is False
    assert record['cold_start_qualification'] is False
    assert record['atmosphere_reconverged'] is False
    assert record['historical_recovery']['maximum_relative_change'] == 0.
    assert manifest['review']['historical_reference_bytes_changed'] is False
    assert manifest['review']['J1637_reference_changed'] is False
    with np.load(source, allow_pickle=False) as saved:
        config = DZConfig(**json.loads(str(saved['config_json'])))
        assert record['config'] == json.loads(json.dumps(asdict(config)))
        wavelength = saved['wavelength']
    with np.load(output, allow_pickle=False) as current, np.load(previous, allow_pickle=False) as prior:
        np.testing.assert_array_equal(current['wavelength'], wavelength)
        np.testing.assert_array_equal(current['wavelength'], prior['wavelength'])
        assert np.all(np.isfinite(current['surface_flux'])) and np.all(current['surface_flux'] > 0.)
        maximum = float(np.max(np.abs(current['surface_flux'] / prior['surface_flux'] - 1.)))
        assert maximum == record['change_from_previous']['maximum_relative_change']
