"""Cold-case dispatch assertions follow the selected production workflow."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from wd_spectra import DBConfig, DABConfig


SPEC = importlib.util.spec_from_file_location(
    "validate_cool_case", Path(__file__).parents[1] / "tools/validate_cool_case.py")
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


@pytest.mark.parametrize("case,workflow,config_type", [
    ("dense-db-5000", "dense-db", DBConfig),
    ("dense-db-8000", "dense-db", DBConfig),
    ("molecular-dab-7500", "dab", DABConfig),
    ("molecular-dab-9000", "dab", DABConfig),
    ("molecular-dab-10000", "dab", DABConfig),
])
def test_cold_cases_require_the_matching_qualified_workflow(
        monkeypatch, tmp_path, case, workflow, config_type):
    calls = []

    def run_model(config, output, **options):
        calls.append((config, output, options))
        return SimpleNamespace(
            selection=SimpleNamespace(workflow=workflow,
                                      diagnostics={"include_molecules": True}),
            convergence_verified=True)

    monkeypatch.setattr(VALIDATE, "run_model", run_model)
    monkeypatch.setattr(sys, "argv", ["validate_cool_case", case, str(tmp_path)])
    VALIDATE.main()
    assert len(calls) == 1
    config, output, options = calls[0]
    assert isinstance(config, config_type)
    assert config.effective_temperature == int(case.rsplit("-", 1)[1])
    assert config.quality == "production"
    assert output == tmp_path
    assert options == {"require_convergence": True}


@pytest.mark.parametrize("case,workflow", [
    ("dense-db-5000", "dense-db"),
    ("molecular-dab-7500", "dab"),
])
def test_unqualified_cold_case_is_rejected(monkeypatch, tmp_path, case, workflow):
    monkeypatch.setattr(VALIDATE, "run_model", lambda *args, **kwargs: SimpleNamespace(
        selection=SimpleNamespace(workflow=workflow,
                                  diagnostics={"include_molecules": True}),
        convergence_verified=False))
    monkeypatch.setattr(sys, "argv", ["validate_cool_case", case, str(tmp_path)])
    with pytest.raises(AssertionError):
        VALIDATE.main()


@pytest.mark.parametrize("workflow,diagnostics", [
    ("molecular-dab", {"include_molecules": True}),
    ("dab", {}),
    ("dab", {"include_molecules": False}),
])
def test_molecular_cold_case_rejects_wrong_route_or_atomic_physics(
        monkeypatch, tmp_path, workflow, diagnostics):
    monkeypatch.setattr(VALIDATE, "run_model", lambda *args, **kwargs: SimpleNamespace(
        selection=SimpleNamespace(workflow=workflow, diagnostics=diagnostics),
        convergence_verified=True))
    monkeypatch.setattr(sys, "argv", ["validate_cool_case", "molecular-dab-7500", str(tmp_path)])
    with pytest.raises(AssertionError):
        VALIDATE.main()
