from types import SimpleNamespace
import numpy as np
from wd_spectra._dq.dq_thermal_corner_trial import temperature_pieces, search_thermal_pieces


def test_pieces_come_only_from_table_and_existing_box():
    table = SimpleNamespace(temperature_grid=np.array([4000., 6000., 8000.]))
    system = SimpleNamespace(n=2, eos_trial_owner=SimpleNamespace(helium_reos3=table))
    state = np.r_[np.log([5000., 6000.*np.exp(2e-4)]), 0.]
    pieces = temperature_pieces(system, state, np.zeros(3), .04)
    assert len(pieces) == 4
    assert all(i == 1 and -.04000000001 <= lo < hi <= .04000000001
               for i, lo, hi in pieces)
    np.testing.assert_allclose([p[2] for p in pieces[:-1]], [-4e-4, -2e-4, 0.], atol=1e-14)
    assert temperature_pieces(system, state, np.array([0., .001, 0.]), .04) == []


def test_smooth_eos_and_absent_eos_have_no_corner_search():
    system = SimpleNamespace(n=2, eos_trial_owner=None)
    assert temperature_pieces(system, np.zeros(3), np.zeros(3), .04) == []
    system.eos_trial_owner = SimpleNamespace(helium_reos3=SimpleNamespace(
        temperature_grid=np.array([4000., 6000., 8000.]), smooth_pressure=True))
    assert temperature_pieces(system, np.zeros(3), np.zeros(3), .04) == []


def test_piece_search_crosses_corner_without_changing_equations_or_bounds():
    anchor = np.log([5000., 6000.*np.exp(2e-4)])
    table = SimpleNamespace(temperature_grid=np.array([4000., 6000., 8000.]))
    system = SimpleNamespace(n=2, eos_trial_owner=SimpleNamespace(helium_reos3=table))
    class Model:
        n = 2
        def physical(self, logt, need):
            return SimpleNamespace(payload={'logt':logt})
        def state_from_temperature(self, logt, physical):
            return np.r_[logt, (logt[1]-logt[0])/1e-4]
        def compatible_state_jacobian(self, logt, physical):
            return np.array([[1., 0.], [0., 1.], [-1e4, 1e4]])
        def evaluate(self, state, need):
            return SimpleNamespace(payload={})
    model = Model()
    state = model.state_from_temperature(anchor, model.physical(anchor, True))
    def rows(q, ev):
        delta = q[:2]-anchor
        residual = np.array([delta[0]-.001, abs(delta[1])-.005,
                             q[2]-(q[1]-q[0])/1e-4])
        jacobian = np.array([[1., 0., 0.], [0., np.sign(delta[1]), 0.],
                             [1e4, -1e4, 1.]])
        return residual, jacobian
    original = state.copy()
    best, trials = search_thermal_pieces(system, model, state, rows, np.zeros(3), .04)
    cost, solved, direction, residual = best
    assert len(trials) == 4
    assert cost < 1e-12
    assert np.max(abs(direction[:2])) <= .04
    assert abs(direction[1]) > .0049
    np.testing.assert_allclose(residual, rows(state+direction, None)[0])
    np.testing.assert_array_equal(state, original)
