"""DB preset wiring: column-mass default recipe vs the explicit optical-depth (v4) preset.
Physics is stubbed; this checks the options passed to the solver and synthesis."""

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


def test_db_defaults_to_column_mass():
    assert DBConfig().transfer_discretization == "column-mass"


def test_column_mass_default_wires_matched_structure_and_synthesis(wired):
    result = compute_db(DBConfig(effective_temperature=9000.0, logg=7.0))
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
