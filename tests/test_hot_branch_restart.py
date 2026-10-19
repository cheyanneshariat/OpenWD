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
