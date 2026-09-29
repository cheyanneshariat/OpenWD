"""Exercise validation orchestration with tiny subprocesses, not atmospheres."""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("validate")
policy = module("validation_policy")


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/limitations.md",
        "tests/test_documentation.py",
        "tests/test_metals.py",
        "examples/generate_spectrum.ipynb",
    ],
)
def test_non_numerical_changes_do_not_trigger_cold_models(path):
    assert not policy.requires_full([path])


@pytest.mark.parametrize(
    "path",
    [
        "src/wd_spectra/opacity.py",
        "csrc/rt_core.c",
        "pyproject.toml",
        "tests/data/spectral_regressions/da-3000.npz",
        "tests/test_protected_model_canaries.py",
        "tests/test_dq_spectral_regression.py",
        "tests/test_dq_release.py",
        "tests/test_hot_cold_canary.py",
        "tests/test_released_family_canaries.py",
        "tools/validate.py",
        ".github/workflows/canaries.yml",
        "unknown-input.bin",
    ],
)
def test_physics_and_unknown_changes_require_full(path):
    assert policy.requires_full(["README.md", path])


def test_draft_and_label_selection_does_not_forge_a_physical_pass():
    event = {"pull_request": {"draft": False, "labels": []}}
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, False)
    event["pull_request"]["labels"] = [{"name": "full-validation"}]
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, True)
    event["pull_request"]["draft"] = True
    assert policy.plan("pull_request", event, ["src/x.py"]) == (True, False)
    assert policy.plan("schedule", {}, []) == (True, True)
    assert policy.plan("workflow_dispatch", {}, []) == (True, True)


def test_default_fast_stage_excludes_both_expensive_markers():
    tasks = runner.commands("fast")
    assert tasks["components"][-1] == "not canary and not spectral"
    assert "research/cool_models" in tasks["cool-components"]
    with pytest.raises(ValueError):
        runner.commands("cold", ["unknown"])


def test_dq_fixed_spectrum_and_true_cold_are_separate_tiers():
    spectral = runner.commands("spectra", ["dq-j1235"])["dq-j1235"]
    cold = runner.commands("cold", ["dq-j1235"])["dq-j1235"]
    assert spectral[-1].startswith("tests/test_dq_spectral_regression.py::")
    assert cold[-1] == runner.COLD_TESTS["dq-j1235"]
    assert "true_cold" in cold[-1]


def test_every_released_family_has_a_cold_case():
    assert {case.split("-")[0] for case in runner.COLD} >= {
        "da", "daz", "db", "dab", "dz", "dq", "do", "dao", "pg1159", "d6", "dah"
    }


@pytest.mark.parametrize("variable", ["OPENWD_DATA", "OPENWD_DQ_DATA"])
def test_custom_runtime_data_disables_reuse(variable, monkeypatch, tmp_path):
    monkeypatch.setenv(variable, str(tmp_path))
    assert not runner.numerical_identity(tmp_path)["reuse_eligible"]


def test_plan_never_starts_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_task", lambda *a: pytest.fail("executed"))
    output = tmp_path / "plan"
    assert runner.main(["full", "--plan", "--output", str(output)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert set(planned) == {"fast", "spectra", "cold"}
    assert set(planned["cold"]) == set(runner.COLD)
    assert not output.exists()
    with pytest.raises(SystemExit):
        runner.main(["full", "--case", "da-3000", "--plan"])


def test_failed_subprocess_logs_output_and_cannot_pass(tmp_path):
    result = runner.run_task(
        [sys.executable, "-u", "-c", "print('live failure'); raise SystemExit(3)"],
        tmp_path / "failed",
        "tiny",
        threading.Event(),
        10,
    )
    assert result["status"] == "failed" and result["returncode"] == 3
    assert "live failure" in Path(result["log"]).read_text()


def test_timeout_is_unqualified_not_a_silent_restart(tmp_path):
    result = runner.run_task(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
        tmp_path / "timeout",
        "tiny",
        threading.Event(),
        0.1,
    )
    assert result["status"] == "timed-out"
    assert result["returncode"] != 0


def test_completed_evidence_requires_same_inputs_and_kept_log(tmp_path):
    log = tmp_path / "output.log"
    log.write_text("passed")
    identity = dict(sha256="actual-inputs", reuse_eligible=True)
    previous = dict(
        identity=identity,
        inputs_unchanged_during_run=True,
        tasks={
            "cold/x": dict(
                status="passed", returncode=0, command=["test"], log=str(log)
            )
        },
    )
    assert runner.reusable(previous, identity, "cold/x", ["test"])
    assert not runner.reusable(
        previous, dict(identity, sha256="changed"), "cold/x", ["test"]
    )
    assert not runner.reusable(previous, identity, "cold/x", ["other-test"])
    assert not runner.reusable(
        dict(previous, inputs_unchanged_during_run=False), identity, "cold/x", ["test"]
    )
    previous["tasks"]["cold/x"]["status"] = "interrupted"
    assert not runner.reusable(previous, identity, "cold/x", ["test"])


def test_prose_edits_preserve_numerical_key_but_source_and_data_invalidate(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENWD_DATA", raising=False)
    for name in ("src", "docs", "csrc"):
        (tmp_path / name).mkdir()
    source = tmp_path / "src/new.py"
    source.write_text("x = 1")
    first = runner.numerical_identity(tmp_path)["sha256"]
    (tmp_path / "docs/new.md").write_text("explanation")
    assert runner.numerical_identity(tmp_path)["sha256"] == first
    source.write_text("x = 2")
    second = runner.numerical_identity(tmp_path)["sha256"]
    assert first != second
    (tmp_path / "csrc/kernel.c").write_text("changed kernel")
    assert runner.numerical_identity(tmp_path)["sha256"] != second


def test_failed_fast_stage_prevents_expensive_stages(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        runner, "numerical_identity", lambda: dict(sha256="same", reuse_eligible=True)
    )
    monkeypatch.setattr(runner, "commands", lambda tier, cases: {tier: [tier]})

    def fail(command, *args):
        called.append(command)
        return dict(status="failed", returncode=1)

    monkeypatch.setattr(runner, "run_task", fail)
    assert runner.main(["full", "--output", str(tmp_path / "run")]) == 1
    assert called == [["fast"]]
    assert not json.loads((tmp_path / "run/summary.json").read_text())[
        "full_qualification"
    ]


def test_workflow_preserves_every_cold_case_and_has_no_duplicate_push_trigger():
    workflow = (ROOT / ".github/workflows/canaries.yml").read_text()
    matrix = re.search(r"case: \[([^\]]+)\]", workflow).group(1)
    assert set(matrix.split(", ")) == set(runner.COLD)
    assert "  push:" not in workflow
    assert "cancel-in-progress:" in workflow
    assert "  qualification:" in workflow and "if: always()" in workflow
    assert 'test "$COLD_RESULT" = success' in workflow


def test_protected_node_ids_still_collect_without_running_atmospheres():
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            *runner.COLD_TESTS.values(),
        ],
        cwd=ROOT,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    for node in runner.COLD_TESTS.values():
        assert node in output
