from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra._dq.automatic_conditioning import thermal_condition




def test_thermal_handoff_is_not_an_energy_or_atmosphere_certificate():
    rows = []
    state = np.ones(3)
    ev = SimpleNamespace(payload=dict(total_flux_interface=np.array([10.,10.]),
        cell_energy_balance_relative_residual=np.array([.5])))
    system = SimpleNamespace(n=2, target=10., evaluate=lambda q,need:ev)
    returned = thermal_condition(system, state, .002, lambda row:rows.append(row))
    np.testing.assert_array_equal(returned, state)
    assert rows[-1]['phase'] == 'thermal-handoff'
    assert rows[-1]['energy'] == .5
    assert rows[-1]['atmosphere_certified'] is False


def test_exhausted_helper_returns_only_last_accepted_guess_when_requested():
    rows=[];state=np.ones(3)
    ev=SimpleNamespace(payload=dict(total_flux_interface=np.array([10.,15.]),
        cell_energy_balance_relative_residual=np.array([.5])))
    system=SimpleNamespace(n=2,target=10.,evaluate=lambda q,need:ev)
    with pytest.raises(RuntimeError,match='step budget'):
        thermal_condition(system,state,.002,rows.append,maximum_steps=0)
    result=thermal_condition(system,state,.002,rows.append,maximum_steps=0,
                             handoff_at_budget=True)
    np.testing.assert_array_equal(result,state)
    assert result is not state
    assert rows==[dict(phase='thermal-budget-handoff',iterations=0,flux=.5,
        energy=.5,thermal_target_reached=False,atmosphere_certified=False)]


def test_controller_uses_result_convergence_not_diagnostic_accounting(monkeypatch, tmp_path):
    import wd_spectra._dq.automatic_conditioning as controller
    import wd_spectra.adaptive_structure as adaptive
    class System:
        n = 2
        def __init__(self):
            self.excess_scale = np.ones(1)
        def physical(self, x, need):
            return SimpleNamespace(payload={})
        def state_from_temperature(self, x, actual):
            self.excess_scale = np.ones(1)
            return np.r_[x, 0.]
        def evaluate(self, state, need):
            return SimpleNamespace(payload={})
    system = System()
    expected = SimpleNamespace(state=np.zeros(3), converged=True,
                               diagnostics=SimpleNamespace(terminal_reason='converged'))
    monkeypatch.setattr(controller, 'thermal_condition', lambda system,state,tol,emit:state)
    monkeypatch.setattr(controller, 'refresh_energy_units', lambda *a:None)
    monkeypatch.setattr(adaptive, 'solve_trust_region_newton', lambda *a,**k:expected)
    with controller.automatic_material_trials(tmp_path):
        result = adaptive.solve_trust_region_newton(np.zeros(3), system.evaluate,
                                                    maximum_iterations=5)
    assert result is expected
    # A new pressure-domain context must retain the earlier domain's log.
    import json
    first = json.loads((tmp_path/'automatic-phases.json').read_text())
    with controller.automatic_material_trials(tmp_path):
        adaptive.solve_trust_region_newton(np.zeros(3), system.evaluate, maximum_iterations=5)
    both = json.loads((tmp_path/'automatic-phases.json').read_text())
    assert both[:len(first)] == first
    assert len(both) == 2*len(first)
    assert all(row['domain_depths'] == 2 for row in both)
    assert all(b['seconds'] >= a['seconds'] for a,b in zip(both[:-1],both[1:]))
    def conditioning_with_rejected_trial(system,state,tol,emit):
        def physical(temperature):
            return dict(atmosphere=SimpleNamespace(temperature=np.array([temperature]),
                gas_pressure=np.array([1.]),column_mass=np.array([1.]),
                rosseland_optical_depth=np.array([1.])))
        emit(dict(phase='thermal-step',accepted=True),physical(5000.))
        emit(dict(phase='thermal-step',accepted=False),rejected_payload=physical(6000.))
        return state
    monkeypatch.setattr(controller,'thermal_condition',conditioning_with_rejected_trial)
    with controller.automatic_material_trials(tmp_path):
        adaptive.solve_trust_region_newton(np.zeros(3),system.evaluate,maximum_iterations=5)
    with np.load(tmp_path/'automatic-latest.npz') as accepted:
        np.testing.assert_array_equal(accepted['temperature'],[5000.])
    with np.load(tmp_path/'automatic-last-rejected.npz') as rejected:
        np.testing.assert_array_equal(rejected['temperature'],[6000.])
        assert not rejected['accepted'] and not rejected['atmosphere_certified']


def test_verified_same_run_domain_extension_reconditions_before_steady_solve(monkeypatch,tmp_path):
    import wd_spectra._dq.automatic_conditioning as controller
    import wd_spectra.adaptive_structure as adaptive
    payload=dict(total_flux_interface=np.array([9.,11.]),
        cell_energy_balance_relative_residual=np.array([.2]))
    evaluation=SimpleNamespace(payload=payload)
    class System:
        n=2;target=10.;same_run_domain_extension=True
        def __init__(self):self.excess_scale=np.ones(1)
        def evaluate(self,*args):return evaluation
        def physical(self,*args):return evaluation
        def state_from_temperature(self,x,actual):
            self.excess_scale=np.ones(1);return np.r_[x,0.]
    system=System()
    expected=SimpleNamespace(state=np.zeros(3),converged=True)
    calls=[]
    def condition(system,state,tolerance,emit):
        calls.append((system,state.copy(),tolerance))
        return state
    monkeypatch.setattr(controller,'thermal_condition',condition)
    monkeypatch.setattr(controller,'refresh_energy_units',lambda *a:None)
    monkeypatch.setattr(adaptive,'solve_trust_region_newton',lambda *a,**k:expected)
    with controller.automatic_material_trials(tmp_path):
        assert adaptive.solve_trust_region_newton(
            np.zeros(3),system.evaluate,maximum_iterations=5
        ) is expected
    import json
    phases=json.loads((tmp_path/'automatic-phases.json').read_text())
    assert len(calls)==1
    assert phases[0]['phase']=='thermal-start-same-run-domain-extension'
    assert phases[0]['atmosphere_certified'] is False
    assert any(row['phase']=='steady-start' for row in phases)




