from types import SimpleNamespace

import numpy as np
import pytest

import wd_spectra._hot_recovery as recovery
from wd_spectra._hot_recovery import RejectedDirectionFallback, ThermalBranchRestart
from wd_spectra.nonlinear import RecoverableEvaluationError


def _fallback(candidate, *, allowed=True):
    proposals = SimpleNamespace(branch_candidate=candidate, correction=lambda *a: 'proposal')
    fallback = RejectedDirectionFallback(proposals, branch_restart=allowed)
    fallback.rejected(None, None)
    fallback.rejected(None, None)
    assert fallback.active
    return fallback


def test_pending_rejected_branch_switch_restarts_from_the_switched_state():
    anchor = np.array([np.log(23586.), 1., 2.])
    switched = np.array([np.log(20606.), 1.1, 2.])
    fallback = _fallback((anchor, 0, switched))
    with pytest.raises(ThermalBranchRestart) as raised:
        fallback.correction(anchor + 1e-6, None)
    assert raised.value.depth == 0
    np.testing.assert_array_equal(raised.value.state, switched)
    np.testing.assert_array_equal(raised.value.anchor, anchor)
    assert fallback.proposals.branch_candidate is None
    # The request must leave the driver instead of being treated as an
    # unavailable correction.
    assert not isinstance(raised.value, RecoverableEvaluationError)


def test_without_permission_or_candidate_the_correction_proceeds():
    candidate = (np.zeros(3), 0, np.ones(3))
    assert _fallback(candidate, allowed=False).correction(np.zeros(3), None) == 'proposal'
    assert _fallback(None).correction(np.zeros(3), None) == 'proposal'


def test_dormant_recovery_neither_corrects_nor_restarts():
    proposals = SimpleNamespace(branch_candidate=(np.zeros(3), 0, np.ones(3)))
    fallback = RejectedDirectionFallback(proposals, branch_restart=True)
    assert fallback.correction(np.zeros(3), None) is None
    assert not fallback.rejected(np.zeros(3), None)


def test_population_restoration_is_not_capped_at_twelve_chord_steps():
    # DO 55 kK/logg 8: the chord contracts by ~0.44 per step from 6e-2 and
    # needs ~15 steps to reach the 1e-7 acceptance.
    assert recovery._RESTORATION_ITERATIONS >= 20


@pytest.mark.parametrize('budget', [1, 3, 4, 5, 8, 20])
def test_branch_restarts_share_the_budget_even_when_every_newton_direction_is_rejected(
    monkeypatch, budget
):
    from wd_spectra import gray_helium_atmosphere
    import wd_spectra._hot_structure as joint
    from wd_spectra.nonlinear import NonlinearEvaluation

    attempts = []
    callbacks = []

    class Equations:
        def __init__(self, seed, model, wave, **kwargs):
            self.seed = seed
            self.wave = wave
            self.jacobian_key = None
            self.linear_solver_drivers = set()

        def initial_state(self, populations):
            return np.zeros(self.seed.n_depth)

        def evaluate(self, state, jacobian):
            if jacobian:
                self.jacobian_key = state.tobytes()
            # An uphill tangent rejects every direction, so the common
            # driver never emits an accepted-step callback.
            return NonlinearEvaluation(state + 1., -np.eye(len(state)) if jacobian else None,
                (self.seed, SimpleNamespace(converged=False), {'nlte_populations_converged': False}))

    class Proposals:
        extra_residual_evaluations = thermal_probes = neighbor_adjustments = 0

        def __init__(self, evaluate, direction, tolerance, equations):
            self.step_builder = direction
            self.branch_candidate = None

        def direction(self, *args):
            attempts.append(True)
            return self.step_builder(*args)

        def correction(self, state, evaluation):
            self.branch_candidate = (state.copy(), 0, state + .01)
            return None

        def project(self, old, trial):
            return trial

    monkeypatch.setattr(joint, 'HotEquations', Equations)
    monkeypatch.setattr(recovery, 'CoupledBracketedThermalPopulationCorrection', Proposals)
    monkeypatch.setattr(joint, 'transfer_field', lambda *a, **kw: (None, None, 0.))
    seed = gray_helium_atmosphere(55000., 8., n_depth=3)
    model = SimpleNamespace(population_tolerance=1e-4, n_angle=2,
                            transfer_coefficients=lambda *a: None)
    result = joint.solve(seed, model, np.array([100., 1000.]), maximum_iterations=budget,
                         iteration_callback=lambda *a: callbacks.append(a))

    assert not callbacks
    assert len(attempts) <= budget
    assert result.nonlinear_result.iterations <= budget
    restarts = result.atmosphere.metadata['hot_nlte_recovery']['thermal_branch_restarts']
    assert len(restarts) <= 3
    if budget > 4:
        assert restarts  # Preserve recovery when there is budget to continue.
    else:
        assert not restarts
    assert not result.atmosphere.metadata['equilibrium_certificate']['verified']
