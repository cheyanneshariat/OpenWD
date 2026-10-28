"""compute_db refuses cool DBs whose dense neutral photosphere needs the run_model dense-helium workflow."""

import numpy as np
import pytest

from wd_spectra.models import DenseHeliumPhotosphereError, ModelData
from wd_spectra.models import automatic, selection, stellar
from wd_spectra.models.selection import PhysicsSelection
from wd_spectra.models.stellar import DBConfig


class _ReachedSolver(Exception):
    pass


def _probe(**changes):
    probe = dict(temperature=4000.0, gas_pressure=1e9, mass_density=0.5,
                 electron_fraction=1e-14, reos_in_domain=True, density_fractional_change=0.9)
    return {**probe, **changes}


def _stop_at_solver(monkeypatch):
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def solver(*args, **kwargs):
        raise _ReachedSolver
    monkeypatch.setattr(stellar, "radiative_equilibrium_helium_atmosphere", solver)


def test_dense_photosphere_is_refused_with_run_model_advice(monkeypatch, tmp_path):
    monkeypatch.setattr(selection, "_helium_probe", lambda config, data: _probe())
    with pytest.raises(DenseHeliumPhotosphereError, match="run_model"):
        stellar.compute_db(DBConfig(effective_temperature=5000.0, transfer_discretization="optical-depth"),
                           data=ModelData(tmp_path))


def test_neutral_photosphere_beyond_reos_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(selection, "_helium_probe", lambda config, data: _probe(
        reos_in_domain=False, density_fractional_change=None))
    with pytest.raises(DenseHeliumPhotosphereError, match="outside the He-REOS.3 table"):
        stellar.compute_db(DBConfig(effective_temperature=3000.0, transfer_discretization="optical-depth"),
                           data=ModelData(tmp_path))


@pytest.mark.parametrize("probe", [
    _probe(density_fractional_change=0.01),   # dense corrections negligible
    _probe(electron_fraction=0.1),            # ionized photosphere
])
def test_dilute_photosphere_reaches_the_solver(monkeypatch, tmp_path, probe):
    monkeypatch.setattr(selection, "_helium_probe", lambda config, data: probe)
    _stop_at_solver(monkeypatch)
    with pytest.raises(_ReachedSolver):
        stellar.compute_db(DBConfig(transfer_discretization="optical-depth"), data=ModelData(tmp_path))


def test_explicit_override_skips_the_screen(monkeypatch, tmp_path):
    def probe(config, data):
        raise AssertionError("screen must not run")
    monkeypatch.setattr(selection, "_helium_probe", probe)
    _stop_at_solver(monkeypatch)
    with pytest.raises(_ReachedSolver):
        stellar.compute_db(DBConfig(effective_temperature=5000.0, transfer_discretization="optical-depth"),
                           data=ModelData(tmp_path), allow_dense_photosphere=True)


def test_selection_and_guard_share_one_criterion():
    assert selection.dense_helium_indicated(_probe())
    assert not selection.dense_helium_indicated(_probe(density_fractional_change=0.049))
    assert not selection.dense_helium_indicated(_probe(electron_fraction=2e-3))


def test_run_model_passes_its_own_screen_to_compute_db(monkeypatch, tmp_path):
    monkeypatch.setattr(automatic, "select_physics",
                        lambda *a, **k: PhysicsSelection("db", "test", {}, False, False))
    calls = []

    def compute(*args, **kwargs):
        calls.append(kwargs)
        raise _ReachedSolver
    monkeypatch.setattr(automatic, "compute_db", compute)
    with pytest.raises(_ReachedSolver):
        automatic.run_model(DBConfig(), tmp_path/"run", wavelength=np.geomspace(3000., 9000., 50))
    assert calls[0]["allow_dense_photosphere"] is True
