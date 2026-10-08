from types import SimpleNamespace
import numpy as np
import pytest
import wd_spectra._dq.dq_current_energy_experiment as experiment
from wd_spectra.nonlinear import NonlinearEvaluation


def test_native_current_rows_are_preserved_and_fixed_weight_drift_is_inapplicable(monkeypatch):
    residual, jacobian = np.arange(3.), np.eye(3)
    payload = dict(thermal_cell_emission=np.array([2.]),
                   dq_augmented_auxiliary_flux=np.array([0., 3.]))
    ev = NonlinearEvaluation(residual, jacobian, payload)
    native = lambda *a:ev
    monkeypatch.setattr(experiment.ExplicitGradientSystem, 'evaluate', native)
    old_scales = experiment.phase_units.SCALES.get()
    with experiment.current_energy_phase_rows():
        got = experiment.ExplicitGradientSystem.evaluate(None, None, True)
        assert got.residual is residual
        assert got.jacobian is jacobian
        np.testing.assert_array_equal(got.payload['dq_fixed_energy_scale'], [5.])
        assert experiment.phase_units.scale_drift(got.payload) == 0.
        experiment.phase_units.refresh_energy_units(SimpleNamespace(pressure=np.array([1., 2.])), got.payload)
        assert experiment.phase_units.scale_drift(got.payload) == 0.
    assert experiment.phase_units.SCALES.get() is old_scales
    assert experiment.ExplicitGradientSystem.evaluate is native


def test_context_restores_after_error(monkeypatch):
    native = experiment.ExplicitGradientSystem.evaluate
    scales = experiment.phase_units.SCALES.get()
    with pytest.raises(RuntimeError):
        with experiment.current_energy_phase_rows():
            raise RuntimeError('stop')
    assert experiment.ExplicitGradientSystem.evaluate is native
    assert experiment.phase_units.SCALES.get() is scales




