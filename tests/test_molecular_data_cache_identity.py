"""External CIA content must identify both the parsed data and model request."""

import hashlib
import os

import numpy as np

from wd_spectra.models import DABConfig, ModelData
from wd_spectra.models.common import model_request_fingerprint
from wd_spectra.models.stellar import _mixed_molecular_data


def _write_cia(path, multiplier=1):
    path.write_text(
        "H2-He 100.0 200.0 2 1000.0\n"
        f"100.0 {multiplier * 1.0e-45:.4e}\n"
        f"200.0 {multiplier * 2.0e-45:.4e}\n"
        "H2-He 100.0 200.0 2 2000.0\n"
        f"100.0 {multiplier * 2.0e-45:.4e}\n"
        f"200.0 {multiplier * 4.0e-45:.4e}\n",
        encoding="ascii",
    )


def _request(config, data, molecular):
    return model_request_fingerprint(
        "DAB", config, data,
        physical_data_identity={
            "h2_he_cia_path": str(molecular.h2_he_table.source_path),
            "h2_he_cia_sha256": molecular.h2_he_table.source_sha256,
            "molecular_closure_revision": 1,
        },
    )


def test_same_size_and_time_cia_replacement_changes_data_and_request(tmp_path):
    path = tmp_path / "external-H2-He.cia"
    _write_cia(path)
    original_stat = path.stat()
    data = ModelData.default()
    config = DABConfig(include_molecules=True, h2_he_cia_path=str(path))
    original = _mixed_molecular_data(str(path), str(data.h2_h2_cia))
    original_request = _request(config, data, original)

    _write_cia(path, multiplier=2)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns

    replaced = _mixed_molecular_data(str(path), str(data.h2_h2_cia))
    replaced_request = _request(config, data, replaced)
    assert replaced is not original
    np.testing.assert_array_equal(
        replaced.h2_he_table.coefficient_cm5,
        2.0 * original.h2_he_table.coefficient_cm5,
    )
    assert replaced.h2_he_table.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert replaced.h2_he_table.source_sha256 != original.h2_he_table.source_sha256
    assert replaced_request["sha256"] != original_request["sha256"]


def test_unchanged_cia_content_reuses_parsed_data(tmp_path):
    path = tmp_path / "external-H2-He.cia"
    _write_cia(path)
    data = ModelData.default()
    original = _mixed_molecular_data(str(path), str(data.h2_h2_cia))
    assert _mixed_molecular_data(str(path), str(data.h2_h2_cia)) is original

    timestamp = path.stat()
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns + 1_000_000))
    assert _mixed_molecular_data(str(path), str(data.h2_h2_cia)) is original
