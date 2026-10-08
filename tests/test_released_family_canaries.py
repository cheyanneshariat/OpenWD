"""Fresh qualification of released DZ, PG 1159, and D6 models.

No stored structure is supplied. PG 1159 retains its explicitly weaker
spectrum qualification profile; this test must never relabel it converged.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import D6Config, DZConfig, PG1159Config, run_model


@pytest.mark.canary
@pytest.mark.parametrize("case", ["dz-pg1225", "pg1159-pg1424", "d6-j1637"])
def test_released_family_cold_start(case, tmp_path):
    if case == "dz-pg1225":
        # Only stellar parameters are read; atmosphere and flux arrays are
        # neither loaded nor used as a solver input.
        reference = Path(__file__).parent / "data/spectral_regressions/dz-pg1225.npz"
        with np.load(reference, allow_pickle=False) as saved:
            config = DZConfig(**json.loads(str(saved["config_json"])))
    elif case == "pg1159-pg1424":
        config = PG1159Config(
            mass_fractions={"He": 0.52, "C": 0.45, "O": 0.03},
            refine_upper_atmosphere=True,
        )
    else:
        config = D6Config()
    run = run_model(config, tmp_path / case, require_convergence=case != "pg1159-pg1424")
    record = json.loads((run.output_directory / "metadata.json").read_text())
    manifest = json.loads((run.output_directory / "model-run.json").read_text())
    assert manifest["cold_start"] is True
    assert manifest["status"] == "completed"
    certificate = record["atmosphere_metadata"]["equilibrium_certificate"]
    assert certificate["verified"] is True
    assert not certificate["failures"]
    for name in certificate["required_checks"]:
        check = certificate["checks"][name]
        assert check["measured"] and check["passed"]
        assert np.isfinite(check["value"]) and 0 <= check["value"] < check["tolerance"]
    status = record["model_metadata"]["atmosphere_convergence_status"]
    if case == "pg1159-pg1424":
        assert status == "spectrum-qualified"
        assert certificate["profile"] == "pg1159-spectrum-gate-v1"
        assert run.convergence_verified is False
        assert record["spectrum_metadata"]["source_closure_residual"] < 1e-6
    else:
        assert status == "converged" and run.convergence_verified
    flux = run.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(flux)) and np.all(flux >= 0) and np.any(flux > 0)
