"""The normal CLI uses the same physics selection as public Python runs."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra.models import cli
from wd_spectra.spectrum import Spectrum


@pytest.mark.parametrize("kind", ["DA", "DB", "DAB", "DZ", "PG1159", "D6"])
def test_cli_delegates_cold_requests_to_run_model(kind, tmp_path, monkeypatch):
    calls = []
    spectrum = Spectrum(np.array([4000., 5000.]), np.ones(2), {})

    def run(config, output, **options):
        calls.append((config, output, options))
        return SimpleNamespace(spectrum=spectrum, output_directory=output)

    monkeypatch.setattr(cli, "run_model", run)
    monkeypatch.setattr(cli, "_quicklook", lambda *args: None)
    for name in ("compute_da", "compute_db", "compute_dab", "compute_dz"):
        monkeypatch.setattr(cli, name, lambda *a, **k: pytest.fail("bypassed physics selection"))
    monkeypatch.setattr(sys, "argv", ["one-shot", "--teff", "9000", "--quality", "production",
                                    "--output", str(tmp_path / "model"), "--require-convergence",
                                    "--wavelength-min", "4000", "--wavelength-max", "5000",
                                    "--wavelength-step", "100"])
    cli.one_shot_main(kind)
    config, output, options = calls[0]
    assert len(calls) == 1
    assert config.effective_temperature == 9000 and config.quality == "production"
    assert options["require_convergence"] is True
    np.testing.assert_array_equal(options["wavelength"], np.arange(4000., 5001., 100.))


def test_strict_fixed_synthesis_is_rejected_before_reading_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["one-shot", "--synthesize-atmosphere", "old.npz",
                                    "--require-convergence", "--output", str(tmp_path / "model")])
    monkeypatch.setattr(cli, "load_atmosphere_checkpoint", lambda *a, **k: pytest.fail("read checkpoint"))
    with pytest.raises(SystemExit):
        cli.one_shot_main("DB")
