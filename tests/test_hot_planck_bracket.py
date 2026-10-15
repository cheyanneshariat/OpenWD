"""Planck-stage thermal root bracket on a synthetic non-monotonic thin layer.

Layer 2's residual has a negative local maximum (the least-squares trap seen
in DAO 40 kK, log H/He = 0) and its only root at log T = 0. Its neighbours
are coupled to it, so moving layer 2 alone raises their residuals.
"""

from types import SimpleNamespace

import numpy as np

from wd_spectra._hot_recovery import PlanckBracketedThermalCorrection
from wd_spectra.nonlinear import NonlinearProposal


def _layer(t):
    s = t / 0.2  # root at 0, local minimum at s = 0.46, negative maximum at s = 1.21
    return -s + 1.5 * s**2 - 0.6 * s**3


def _residual(x):
    return np.array([x[0], x[1] - 0.1 * x[2], _layer(x[2]), x[3] - 0.1 * x[2], x[4]])


def _jacobian(x, step=1e-7):
    return np.column_stack([(_residual(x + step * e) - _residual(x - step * e)) / (2 * step)
                            for e in np.eye(len(x))])


def _correction():
    atmosphere = SimpleNamespace(n_depth=5, rosseland_optical_depth=np.geomspace(1e-7, 1e2, 5))
    evaluate = lambda z, jacobian: SimpleNamespace(residual=_residual(z), payload=(atmosphere, None, {}))
    equations = SimpleNamespace(planck_temperature_only=True)
    corrector = PlanckBracketedThermalCorrection(evaluate, None, 3e-4, equations)
    stuck = np.array([0., 0.1 * 0.241, 0.241, 0.1 * 0.241, 0.])  # layer 2 at the negative maximum
    e = evaluate(stuck, False)
    proposal = NonlinearProposal(np.array([0., 0., 1., 0., 0.]))
    corrector.anchor = (stuck, e, _jacobian(stuck), 0.12, proposal)
    return corrector, stuck, e


def test_brackets_the_true_root_and_relaxes_neighbours():
    corrector, stuck, e = _correction()
    assert abs(_residual(stuck)[2]) > 0.05 and abs(_residual(stuck)[1]) < 1e-12
    correction = corrector.correction(stuck, e)
    assert correction is not None
    trial = correction.trial_state(1.0)
    assert abs(trial[2]) < 1e-6
    # Neighbours follow the moved layer, so the whole residual drops.
    assert np.linalg.norm(_residual(trial)) < 1e-6 < np.linalg.norm(_residual(stuck))
    assert corrector.neighbor_adjustments > 0


def test_inactive_outside_the_planck_stage():
    corrector, stuck, e = _correction()
    corrector.equations.planck_temperature_only = False
    assert corrector.correction(stuck, e) is None


def test_inactive_for_small_or_deep_proposals():
    corrector, stuck, e = _correction()
    anchor = list(corrector.anchor)
    anchor[4] = NonlinearProposal(np.array([0., 0., 0.01, 0., 0.]))
    corrector.anchor = tuple(anchor)
    assert corrector.correction(stuck, e) is None
    corrector, stuck, e = _correction()
    e.payload[0].rosseland_optical_depth[2] = 1.0  # not optically thin
    assert corrector.correction(stuck, e) is None
