"""Release an inner least-squares solve blocked by a tabulated EOS corner.

Only the cheap proposal is restricted; the physical equations and their
acceptance tests are unchanged. Recompute the active set on every proposal.
Never certify stationarity from a constrained subspace correction.
"""
import numpy as np
from scipy.optimize import least_squares


def corner_columns(system, log_temperature):
    owner = getattr(system, 'eos_trial_owner', None)
    if owner is None:
        return np.empty(0, dtype=int)
    table = owner.helium_reos3
    # Smooth tables have no piecewise thermal-window corners.
    if hasattr(table, 'smooth_pressure'):
        return np.empty(0, dtype=int)
    knots = np.log(table.temperature_grid)
    edges = np.unique(np.r_[knots, knots-2e-4, knots+2e-4])
    distance = np.min(abs(np.asarray(log_temperature)[:, None]-edges), axis=1)
    # The bounded solve uses xtol=1e-10. Only identify numerically reached
    # corners, not broad neighborhoods or stellar-temperature regimes.
    return np.flatnonzero(distance <= 1e-10*np.maximum(1., abs(log_temperature)))


def refine_corner_subspace(system, state, solved, radius, evaluated, bounds=None):
    columns = corner_columns(system, state[:system.n]+solved.x[:system.n])
    if not len(columns):
        return solved, {}
    free = np.ones(len(state), dtype=bool)
    free[columns] = False
    if not np.any(free):
        return solved, {}
    anchor = solved.x.copy()
    def unpack(values):
        delta = anchor.copy()
        delta[free] = values
        return delta
    if bounds is None:
        lower = np.full(len(state), -radius)
        upper = np.full(len(state), radius)
    else:
        lower, upper = (np.asarray(value, dtype=float) for value in bounds)
        if lower.shape != state.shape or upper.shape != state.shape:
            raise ValueError('Corner-subspace bounds must match the trial state')
    candidate = least_squares(lambda v: evaluated(unpack(v)).residual, anchor[free],
        jac=lambda v: evaluated(unpack(v)).jacobian[:, free],
        bounds=(lower[free], upper[free]),
        method='trf', x_scale='jac', max_nfev=1000,
        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    selected = np.isfinite(candidate.cost) and candidate.cost < solved.cost
    if selected:
        candidate.x = unpack(candidate.x)
    return (candidate if selected else solved), dict(
        eos_subspace_columns=columns.tolist(), eos_subspace_selected=bool(selected),
        eos_subspace_evaluations=candidate.nfev, eos_subspace_cost=float(candidate.cost))
