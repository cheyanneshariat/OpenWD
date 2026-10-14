"""Exercise the actual tutorial cells without running an atmosphere in unit CI.

Mock artifacts test the notebook interface, not physical convergence. Solver
qualification is covered independently by the protected cold-start canaries.
"""

import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wd_spectra.models import DAConfig, DBConfig, DABConfig, DZConfig
from wd_spectra.models.automatic import ModelRun
from wd_spectra.models.selection import PhysicsSelection


NOTEBOOK = Path(__file__).resolve().parents[1] / "examples/generate_spectrum.ipynb"
CELLS = json.loads(NOTEBOOK.read_text())["cells"]
CODE = {c["id"]: "".join(c["source"]) for c in CELLS if c["cell_type"] == "code"}


def execute(cell, namespace):
    exec(compile(CODE[cell], f"notebook:{cell}", "exec"), namespace)


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    monkeypatch.chdir(NOTEBOOK.parent)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__main__"}
    with plt.rc_context():
        execute("setup", namespace)
        execute("controls", namespace)
        namespace["OUTPUT_ROOT"] = tmp_path
        yield namespace
    plt.close("all")


def fake_runner(workflow, qualified=True, *, empty_zoom=False):
    def run(config, directory, **kwargs):
        directory.mkdir(parents=True)
        wave = np.array([1000.0, 4000.0, 5000.0, 9000.0])
        flux = np.array(
            [1.0, 0.0 if empty_zoom else 3.0, 0.0 if empty_zoom else 2.0, 1.0]
        )
        path = directory / "spectrum.txt"
        np.savetxt(path, np.column_stack([wave, flux]))
        (directory / "model-run.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "convergence_verified": qualified,
                    "cold_start": True,
                    "initial_checkpoint": None,
                }
            )
        )
        experimental = workflow == "dense-db"
        if not experimental:
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "atmosphere_metadata": {
                            "equilibrium_certificate": {"verified": qualified}
                        }
                    }
                )
            )
        return ModelRun(
            directory,
            PhysicsSelection(workflow, "test fixture", {}, True, experimental),
            path,
            qualified,
        )

    return run


def test_notebook_is_executable_with_saved_example_outputs():
    counts = []
    for cell in CELLS:
        if cell["cell_type"] == "code":
            counts.append(cell["execution_count"])
            assert all(output["output_type"] != "error" for output in cell["outputs"])
            compile("".join(cell["source"]), cell["id"], "exec")
    assert counts == list(range(1, len(counts) + 1))
    assert "compute_" not in "\n".join(CODE.values())
    plot = next(cell for cell in CELLS if cell["id"] == "plot")
    assert any(output.get("data", {}).get("image/png") for output in plot["outputs"])
    checks = next(cell for cell in CELLS if cell["id"] == "checks")
    output = "".join("".join(item.get("text", [])) for item in checks["outputs"])
    assert "Numerical qualification: verified for declared equations" in output


@pytest.mark.parametrize(
    "model,cls",
    [("da", DAConfig), ("DB", DBConfig), ("DAB", DABConfig), ("DZ", DZConfig)],
)
def test_requests_preserve_user_parameters(notebook, model, cls):
    notebook.update(
        MODEL=model,
        TEFF=9000.0,
        LOGG=8.0,
        QUALITY="production",
        LOG_H_TO_HE=-3.0,
        DZ_ABUNDANCES={"Ca": -9.0},
    )
    execute("request", notebook)
    config = notebook["config"]
    assert isinstance(config, cls)
    assert config.effective_temperature == 9000.0
    assert config.quality == "production"
    if cls is DABConfig:
        assert config.log_hydrogen_to_helium == -3.0
    if cls is DZConfig:
        assert config.abundances == {"Ca": -9.0}


def test_calculation_uses_automatic_cold_start_and_unique_outputs(notebook):
    execute("request", notebook)
    calls = []

    def record(config, directory, **kwargs):
        calls.append((config, directory, kwargs))
        return object()

    notebook.update(
        run_model=record, REQUIRE_CONVERGENCE=True, RESEARCH_DATA=Path("pinned-data")
    )
    execute("calculate", notebook)
    execute("calculate", notebook)
    assert calls[0][1] != calls[1][1]
    assert all(config is notebook["config"] for config, _, _ in calls)
    assert calls[0][2] == {
        "research_data": Path("pinned-data"),
        "require_convergence": True,
    }


def test_failed_run_clears_previous_result_without_retry(notebook):
    execute("request", notebook)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("missing physics data")

    notebook.update(
        run_model=fail, run=object(), spectrum=object(), summary={"old": True}
    )
    with pytest.raises(ValueError, match="missing physics data"):
        execute("calculate", notebook)
    assert calls == [1]
    assert all(
        notebook[key] is None for key in ("run", "spectrum", "manifest", "summary")
    )
    with pytest.raises(RuntimeError, match="before plotting"):
        execute("plot", notebook)


@pytest.mark.parametrize(
    "workflow", ["da", "db", "dab", "dz", "dense-db"]
)
@pytest.mark.parametrize("qualified", [False, True])
def test_checks_and_plot_use_public_qualification(
    notebook, workflow, qualified, capsys
):
    execute("request", notebook)
    notebook["run_model"] = fake_runner(workflow, qualified)
    execute("calculate", notebook)
    if qualified:
        execute("checks", notebook)
    else:
        with pytest.warns(RuntimeWarning, match="UNQUALIFIED"):
            execute("checks", notebook)
    original = notebook["spectrum"].surface_flux_lambda.copy()
    execute("plot", notebook)
    assert notebook["summary"]["qualification_verified"] is qualified
    assert ("UNQUALIFIED" in notebook["fig"]._suptitle.get_text()) is not qualified
    np.testing.assert_array_equal(notebook["spectrum"].surface_flux_lambda, original)
    np.testing.assert_array_equal(
        notebook["axes"][0].lines[0].get_ydata(), notebook["wave"] * original
    )
    if workflow == "dense-db":
        assert "dedicated worker checker" in capsys.readouterr().out


@pytest.mark.parametrize("empty_zoom", [False, True])
def test_figure_saving_and_empty_zoom(notebook, empty_zoom):
    execute("request", notebook)
    notebook.update(
        run_model=fake_runner("da", empty_zoom=empty_zoom), SAVE_FIGURE=True
    )
    execute("calculate", notebook)
    execute("checks", notebook)
    if not empty_zoom:
        notebook.update(zoom_lower=10000.0, zoom_upper=11000.0)
    execute("plot", notebook)
    assert not notebook["axes"][1].lines
    for suffix in ("png", "pdf"):
        assert (
            notebook["run"].output_directory / f"openwd_da_example_spectrum.{suffix}"
        ).stat().st_size > 0


@pytest.mark.parametrize(
    "key,value",
    [("convergence_verified", False), ("cold_start", False), ("status", "failed")],
)
def test_checks_reject_inconsistent_provenance(notebook, key, value):
    execute("request", notebook)
    notebook["run_model"] = fake_runner("da")
    execute("calculate", notebook)
    path = notebook["run"].output_directory / "model-run.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        execute("checks", notebook)
