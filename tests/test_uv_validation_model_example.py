"""Validate the public UV commands without constructing any atmosphere."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import DAZConfig, DZConfig


CHECKOUT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "uv_validation_model_example", CHECKOUT / "examples" / "uv_validation_model.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
TARGETS = json.loads(EXAMPLE.CONFIGURATIONS.read_text(encoding="utf-8"))["targets"]


@pytest.mark.parametrize("target", TARGETS)
def test_one_published_model_per_command(monkeypatch, tmp_path, target):
    calls = []
    output = tmp_path / target

    def run_model(config, directory, **kwargs):
        calls.append((config, directory, kwargs))
        return SimpleNamespace(output_directory=directory)

    monkeypatch.setattr(EXAMPLE, "run_model", run_model)
    monkeypatch.setattr(EXAMPLE, "compiled_backend_available", lambda: True)
    EXAMPLE.main([target, "--output", str(output)])

    assert len(calls) == 1
    config, directory, kwargs = calls[0]
    entry = TARGETS[target]
    expected_type = {"DZConfig": DZConfig, "DAZConfig": DAZConfig}[entry["config_class"]]
    assert isinstance(config, expected_type)
    actual_config = asdict(config)
    assert {key: actual_config[key] for key in entry["config"]} == entry["config"]
    assert directory == output
    assert kwargs["require_convergence"] is True
    assert set(kwargs) == {"wavelength", "require_convergence"}
    expected_grid = np.arange(
        entry["wavelength_min_A"],
        entry["wavelength_max_A"] + 0.5 * entry["wavelength_step_A"],
        entry["wavelength_step_A"],
    )
    np.testing.assert_array_equal(kwargs["wavelength"], expected_grid)
    if isinstance(config, DAZConfig):
        assert config.h2_lyman_werner_lines
        assert config.include_molecules
        assert config.mixing_length_alpha == 0.8


def test_missing_compiled_backend_does_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(EXAMPLE, "compiled_backend_available", lambda: False)

    def forbidden_run(*args, **kwargs):
        pytest.fail("No model may run without the required C backend")

    monkeypatch.setattr(EXAMPLE, "run_model", forbidden_run)
    with pytest.raises(SystemExit) as error:
        EXAMPLE.main(["gd40", "--output", str(tmp_path / "gd40")])
    assert error.value.code == 2


def test_unknown_target_does_not_run(monkeypatch, tmp_path):
    def forbidden_run(*args, **kwargs):
        pytest.fail("Invalid target must be rejected before synthesis")

    monkeypatch.setattr(EXAMPLE, "run_model", forbidden_run)
    with pytest.raises(SystemExit) as error:
        EXAMPLE.main(["unknown", "--output", str(tmp_path / "unknown")])
    assert error.value.code == 2
