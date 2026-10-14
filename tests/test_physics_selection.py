"""Routing depends on material relevance, never a failed solver or target name."""

import json
import subprocess
import numpy as np
import pytest
from wd_spectra.models import (
    DBConfig,
    DABConfig,
    DZConfig,
    PhysicsSelection,
    PhysicsSelectionPolicy,
    select_physics,
)
from wd_spectra.models import selection, automatic


@pytest.mark.parametrize("t", [5000.0, 9000.0, 12000.0])
def test_dense_choice_uses_material_probe_not_teff(monkeypatch, t):
    monkeypatch.setattr(
        selection,
        "_helium_probe",
        lambda *a: dict(
            reos_in_domain=True, electron_fraction=1e-6, density_fractional_change=0.2
        ),
    )
    assert select_physics(DBConfig(effective_temperature=t)).workflow == "dense-db"
    monkeypatch.setattr(
        selection,
        "_helium_probe",
        lambda *a: dict(
            reos_in_domain=True, electron_fraction=1e-6, density_fractional_change=0.01
        ),
    )
    assert select_physics(DBConfig(effective_temperature=t)).workflow == "db"


@pytest.mark.parametrize("inside,ions", [(False, 0.2), (True, 0.2)])
def test_neutral_dense_model_is_not_selected_outside_its_screen(
    monkeypatch, inside, ions
):
    monkeypatch.setattr(
        selection,
        "_helium_probe",
        lambda *a: dict(
            reos_in_domain=inside, electron_fraction=ions, density_fractional_change=0.2
        ),
    )
    assert select_physics(DBConfig()).workflow == "db"


def test_out_of_domain_neutral_screen_does_not_substitute_ideal_eos(monkeypatch):
    monkeypatch.setattr(
        selection,
        "_helium_probe",
        lambda *a: dict(
            reos_in_domain=False, electron_fraction=1e-6, density_fractional_change=None
        ),
    )
    with pytest.raises(ValueError, match="outside the REOS table"):
        select_physics(DBConfig(effective_temperature=3000.0))


def test_nonfinite_molecular_screen_is_not_atomic_selection(monkeypatch):
    monkeypatch.setattr(
        selection,
        "_molecular_probe",
        lambda *a: dict(
            maximum_h2_nuclei_fraction=np.nan, maximum_electron_fractional_change=0.0
        ),
    )
    with pytest.raises(ValueError, match="Invalid molecular diagnostic"):
        select_physics(DABConfig())


def test_mixed_chemistry_never_uses_pure_helium_eos(monkeypatch):
    monkeypatch.setattr(
        selection,
        "_molecular_probe",
        lambda *a: dict(
            maximum_h2_nuclei_fraction=0.01, maximum_electron_fractional_change=0.02
        ),
    )
    chosen = select_physics(DABConfig())
    assert chosen.workflow == "dab" and chosen.diagnostics["include_molecules"]
    assert chosen.diagnostics["molecular_screen_indicated"]
    assert select_physics(DZConfig()).workflow == "dz"
    monkeypatch.setattr(
        selection,
        "_molecular_probe",
        lambda *a: dict(
            maximum_h2_nuclei_fraction=1e-12, maximum_electron_fractional_change=1e-10
        ),
    )
    # One route on both sides of the screen: no physics switch between neighbours.
    chosen = select_physics(DABConfig())
    assert chosen.workflow == "dab" and chosen.diagnostics["include_molecules"]
    assert not chosen.diagnostics["molecular_screen_indicated"]


@pytest.mark.parametrize("indicated", [False, True])
def test_metal_bearing_dab_keeps_atomic_route_or_refuses(monkeypatch, indicated):
    value = 0.1 if indicated else 1e-12
    monkeypatch.setattr(
        selection,
        "_molecular_probe",
        lambda *a: dict(
            maximum_h2_nuclei_fraction=value, maximum_electron_fractional_change=value
        ),
    )
    config = DABConfig(abundances={"Ca": -8.0})
    if indicated:
        with pytest.raises(ValueError, match="no trace metals"):
            select_physics(config)
    else:
        chosen = select_physics(config)
        assert chosen.workflow == "dab" and "include_molecules" not in chosen.diagnostics


def test_unqualified_parameter_points_are_not_called_tested(monkeypatch):
    monkeypatch.setattr(
        selection,
        "_molecular_probe",
        lambda *a: dict(
            maximum_h2_nuclei_fraction=0.1, maximum_electron_fractional_change=0.1
        ),
    )
    assert not select_physics(DABConfig(effective_temperature=5000.0)).tested_point


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_invalid_screen_tolerance(value):
    with pytest.raises(ValueError):
        PhysicsSelectionPolicy(dense_density_fraction=value)


def test_dispatch_never_retries_with_other_physics_on_failure(monkeypatch, tmp_path):
    choice = PhysicsSelection("dense-db", "test", {}, True, True)
    monkeypatch.setattr(automatic, "select_physics", lambda *a, **k: choice)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(automatic.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        automatic.run_model(
            DBConfig(effective_temperature=5000.0, quality="production"),
            tmp_path / "run",
        )
    assert len(calls) == 1
    record = json.loads((tmp_path / "run/model-run.json").read_text())
    assert record["status"] == "failed"
    assert not record["physics_changed_after_failure"]


def test_dispatch_preserves_dense_gravity_and_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(
        automatic,
        "select_physics",
        lambda *a, **k: PhysicsSelection("dense-db", "test", {}, False, True),
    )
    config = DBConfig(effective_temperature=5000.0, logg=7.5, quality="production")
    commands, _ = automatic._cool_commands(
        config, PhysicsSelection("dense-db", "test", {}, False, True), tmp_path / "new"
    )
    command = commands[0]
    assert command[command.index('--logg')+1] == '7.5'
    assert config.logg == 7.5
    assert not (tmp_path / "new").exists()
    with pytest.raises(FileExistsError):
        automatic.run_model(DBConfig(), tmp_path)


def test_only_dense_db_has_an_isolated_cool_worker(tmp_path):
    with pytest.raises(ValueError, match="no isolated cool worker"):
        automatic._cool_commands(
            DABConfig(effective_temperature=7500., logg=7.5, quality='production'),
            PhysicsSelection('dab', 'test', {}, False, False), tmp_path/'new')


def _fake_cia(tmp_path, monkeypatch, *, pinned):
    import hashlib
    from wd_spectra._cool import check_data
    path = tmp_path / "H2-He_2011.cia"
    path.write_bytes(b"stub table")
    digest = hashlib.sha256(b"stub table").hexdigest() if pinned else "0" * 64
    monkeypatch.setitem(check_data.SHA256, "H2-He_2011.cia", digest)
    return path


@pytest.mark.parametrize("logg", [7.0, 8.0, 9.5])
def test_dab_dispatch_enables_pinned_molecules_at_any_gravity(monkeypatch, tmp_path, logg):
    cia = _fake_cia(tmp_path, monkeypatch, pinned=True)
    choice = PhysicsSelection("dab", "test", {"include_molecules": True}, False, False)
    monkeypatch.setattr(automatic, "select_physics", lambda *a, **k: choice)
    seen = {}

    def compute(config, **kwargs):
        seen["config"] = config
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(automatic, "compute_dab", compute)
    request = DABConfig(effective_temperature=12_000.0, logg=logg, quality="production")
    with pytest.raises(RuntimeError, match="stop after dispatch"):
        automatic.run_model(request, tmp_path / "run", research_data=tmp_path)
    assert seen["config"].include_molecules
    assert seen["config"].h2_he_cia_path == str(cia.resolve())
    assert seen["config"].logg == logg
    record = json.loads((tmp_path / "run/model-run.json").read_text())
    assert record["request"]["include_molecules"] is False
    assert record["effective_config"]["include_molecules"] is True


def test_dab_dispatch_reads_research_data_environment(monkeypatch, tmp_path):
    cia = _fake_cia(tmp_path, monkeypatch, pinned=True)
    monkeypatch.setenv("OPENWD_RESEARCH_DATA", str(tmp_path))
    config = automatic._molecular_dab_config(DABConfig(), automatic.ModelData.default(), None)
    assert config.h2_he_cia_path == str(cia.resolve()) and config.include_molecules


def test_dab_dispatch_refuses_unpinned_cia(monkeypatch, tmp_path):
    _fake_cia(tmp_path, monkeypatch, pinned=False)
    choice = PhysicsSelection("dab", "test", {"include_molecules": True}, False, False)
    monkeypatch.setattr(automatic, "select_physics", lambda *a, **k: choice)
    with pytest.raises(ValueError, match="checksum-pinned"):
        automatic.run_model(DABConfig(quality="production"), tmp_path / "run", research_data=tmp_path)
    assert not (tmp_path / "run").exists()


def test_cool_worker_environment_cannot_replace_parent_callbacks(monkeypatch, tmp_path):
    choice = PhysicsSelection("dense-db", "test", {}, True, True)
    monkeypatch.setattr(automatic, "select_physics", lambda *a, **k: choice)
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        report = tmp_path / "run/worker/audit/5000"
        report.mkdir(parents=True)
        (report / "qualification.json").write_text(
            json.dumps(
                dict(numerically_qualified_for_declared_experimental_physics=True)
            )
        )
        output = tmp_path / "run/worker/5000"
        output.mkdir(parents=True)
        np.savetxt(output / "experimental-spectrum.txt", [[4000.0, 1.0], [5000.0, 2.0]])

    monkeypatch.setattr(automatic.subprocess, "run", execute)
    result = automatic.run_model(
        DBConfig(effective_temperature=5000.0, quality="production"), tmp_path / "run"
    )
    assert result.convergence_verified
    assert len(calls) == 1
    assert calls[0][1]["check"] is True
    assert calls[0][0][1:3] == ["-m", "wd_spectra._cool.run_cool_db"]
    assert "research" not in calls[0][1]["env"]["PYTHONPATH"]


def test_missing_worker_spectrum_is_failure_not_qualified_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        automatic,
        "select_physics",
        lambda *a, **k: PhysicsSelection("dense-db", "test", {}, True, True),
    )

    def no_spectrum(*a, **k):
        report = tmp_path / "run/worker/audit/5000"
        report.mkdir(parents=True)
        (report / "qualification.json").write_text(
            json.dumps(
                dict(numerically_qualified_for_declared_experimental_physics=True)
            )
        )

    monkeypatch.setattr(automatic.subprocess, "run", no_spectrum)
    with pytest.raises(FileNotFoundError):
        automatic.run_model(
            DBConfig(effective_temperature=5000, quality="production"), tmp_path / "run"
        )
    assert (
        json.loads((tmp_path / "run/model-run.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize("strict", [False, True])
def test_uncertified_warm_output_is_retained_and_warns_or_raises(
    monkeypatch, tmp_path, strict
):
    from wd_spectra import (
        ModelResult,
        gray_helium_atmosphere,
        AtmosphereConvergenceWarning,
    )
    from wd_spectra.spectrum import Spectrum

    config = DBConfig(effective_temperature=10000)
    result = ModelResult(
        "DB",
        gray_helium_atmosphere(10000, 8, n_depth=8),
        Spectrum(np.array([4000.0, 5000.0]), np.ones(2), {}),
        config,
        {"atmosphere_convergence_status": "unconverged"},
    )
    monkeypatch.setattr(
        automatic,
        "select_physics",
        lambda *a, **k: PhysicsSelection("db", "test", {}, True, False),
    )
    calls = []

    def compute(*a, **k):
        calls.append(k)
        return result

    monkeypatch.setattr(automatic, "compute_db", compute)
    with (
        pytest.raises(RuntimeError, match="without full equilibrium")
        if strict
        else pytest.warns(
            AtmosphereConvergenceWarning, match="without full equilibrium"
        )
    ):
        automatic.run_model(config, tmp_path / "run", require_convergence=strict)
    assert len(calls) == 1
    assert callable(calls[0]["iteration_callback"])
    assert calls[0]["initial_atmosphere"] is None
    record = json.loads((tmp_path / "run/model-run.json").read_text())
    assert record["status"] == "completed" and not record["convergence_verified"]
    assert (tmp_path / "run/spectrum.txt").is_file()
