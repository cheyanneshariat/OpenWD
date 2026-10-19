"""DB preset wiring: the He- screen ("auto"), the column-mass recipe (v5) and the
optical-depth preset (v4). Physics is stubbed; this checks the options passed to
the solver and synthesis."""

import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.models import stellar
from wd_spectra.models.stellar import DBConfig, compute_db


@pytest.fixture
def wired(monkeypatch):
    calls = {}
    atmosphere = gray_helium_atmosphere(9000.0, 7.0, n_depth=12)
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def solve(solver, *args, **kwargs):
        calls["structure"] = kwargs
        return atmosphere

    def synthesize(atm, wave, **kwargs):
        calls["synthesis"] = kwargs
        return object()

    monkeypatch.setattr(stellar, "solve_with_screened_boundary", solve)
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", synthesize)
    monkeypatch.setattr(stellar, "atmosphere_with_model_request_fingerprint", lambda a, f: a)
    monkeypatch.setattr(stellar, "warn_if_atmosphere_not_converged", lambda a, kind: "stub")
    return calls


def test_db_defaults_to_the_helium_minus_screen():
    assert DBConfig().transfer_discretization == "auto"


@pytest.mark.parametrize("share,expected", [(0.57, "column-mass"), (0.45, "optical-depth")])
def test_auto_selects_by_helium_minus_share(wired, monkeypatch, share, expected):
    monkeypatch.setattr(stellar, "_db_helium_minus_share", lambda config, n_depth: share)
    result = compute_db(DBConfig(effective_temperature=9000.0, logg=7.0))
    assert result.metadata["transfer_discretization"] == expected
    assert result.metadata["transfer_discretization_request"] == "auto"
    assert result.metadata["helium_minus_rosseland_share"] == share
    assert wired["structure"].get("transfer_discretization", "optical-depth") == expected


def test_column_mass_wires_matched_structure_and_synthesis(wired):
    result = compute_db(DBConfig(effective_temperature=9000.0, logg=7.0,
                                 transfer_discretization="column-mass"))
    structure, synthesis = wired["structure"], wired["synthesis"]
    assert structure["transfer_discretization"] == "column-mass"
    assert structure["tau_min"] == pytest.approx(1e-12)
    assert structure["deepen_on_any_screen_failure"] is True
    assert structure["restart_after_deepening"] is True
    assert structure["deepening_pressure_factor"] == pytest.approx(4.0)
    assert synthesis["transfer_discretization"] == "column-mass"
    assert synthesis["transfer_depth_refinement"] == DBConfig().synthesis_transfer_depth_refinement == 4
    assert result.metadata["preset"] == "DB-production-v5"
    assert result.metadata["transfer_discretization"] == "column-mass"


def test_optical_depth_restores_the_v4_preset(wired):
    config = DBConfig(effective_temperature=9000.0, logg=7.0, transfer_discretization="optical-depth")
    result = compute_db(config)
    structure, synthesis = wired["structure"], wired["synthesis"]
    for key in ("transfer_discretization", "tau_min", "deepen_on_any_screen_failure",
                "restart_after_deepening", "deepening_pressure_factor"):
        assert key not in structure
    assert "transfer_discretization" not in synthesis
    assert synthesis["transfer_depth_refinement"] == config.synthesis_transfer_depth_refinement
    assert result.metadata["preset"] == "DB-production-v4"


def test_unknown_db_discretization_is_rejected():
    with pytest.raises(ValueError):
        compute_db(DBConfig(transfer_discretization="mass"))


def _staged_solver(monkeypatch, outcomes):
    """Stub solve_with_screened_boundary returning prepared certificates in turn."""
    from dataclasses import replace
    calls = []
    atmosphere = gray_helium_atmosphere(32000.0, 6.0, n_depth=12)

    def solve(solver, *args, **kwargs):
        verified, worst = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(kwargs)
        return replace(atmosphere, metadata={
            "equilibrium_certificate": {"verified": verified,
                                        "failures": [] if verified else ["temperature_stationarity"]},
            "nonlinear_solver_final_worst_residual_depth_index": worst,
        })

    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))
    monkeypatch.setattr(stellar, "_db_helium_minus_share", lambda config, n_depth: 0.0)
    monkeypatch.setattr(stellar, "solve_with_screened_boundary", solve)
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda atm, wave, **k: object())
    monkeypatch.setattr(stellar, "atmosphere_with_model_request_fingerprint", lambda a, f: a)
    monkeypatch.setattr(stellar, "warn_if_atmosphere_not_converged", lambda a, kind: "stub")
    return calls


def test_auto_optical_depth_extends_the_top_when_the_top_layers_fail(monkeypatch):
    calls = _staged_solver(monkeypatch, [(False, 0), (True, 5)])
    result = compute_db(DBConfig(effective_temperature=32000.0, logg=6.0))
    assert [c.get("tau_min") for c in calls] == [None, pytest.approx(1e-9)]
    extensions = result.atmosphere.metadata["upper_boundary_extensions"]
    assert extensions[0]["worst_residual_depth_index"] == 0
    assert result.atmosphere.metadata["seed_top_rosseland_optical_depth"] == pytest.approx(1e-9)


def test_no_extension_for_deep_failures_passing_models_or_explicit_v4(monkeypatch):
    calls = _staged_solver(monkeypatch, [(False, 20)])
    compute_db(DBConfig(effective_temperature=32000.0, logg=6.0))
    assert len(calls) == 1
    calls = _staged_solver(monkeypatch, [(True, 0)])
    compute_db(DBConfig(effective_temperature=32000.0, logg=6.0))
    assert len(calls) == 1
    calls = _staged_solver(monkeypatch, [(False, 0)])
    compute_db(DBConfig(effective_temperature=32000.0, logg=6.0, transfer_discretization="optical-depth"))
    assert len(calls) == 1


def test_top_extensions_are_bounded(monkeypatch):
    calls = _staged_solver(monkeypatch, [(False, 1)])
    compute_db(DBConfig(effective_temperature=32000.0, logg=6.0))
    assert [c.get("tau_min") for c in calls] == [None, pytest.approx(1e-9), pytest.approx(1e-10)]
