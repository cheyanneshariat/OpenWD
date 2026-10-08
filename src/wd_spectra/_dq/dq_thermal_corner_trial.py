"""Bounded thermal proposals across existing, unchanged EOS table corners.

Search the adjacent smooth temperature pieces of the same cheap model. Each
start has exact model ML2 compatibility; nothing is accepted without an outer
full-physics evaluation. No temperature, opacity or EOS value is modified.
"""
import numpy as np
from scipy.optimize import least_squares
from .dq_eos_corner_subspace import corner_columns
from .dq_thermal_manifold import compatible_thermal_model


def temperature_pieces(system, state, candidate, radius):
    """Table/window intervals reachable in the existing temperature box."""
    owner = getattr(system, 'eos_trial_owner', None)
    if owner is None:
        return []
    columns = corner_columns(system, state[:system.n]+candidate[:system.n])
    grid = np.log(owner.helium_reos3.temperature_grid)
    breaks = np.unique(np.r_[grid, grid-2e-4, grid+2e-4])
    result = []
    for i in columns:
        lo = max(state[i]-radius, grid[0]+2e-4)
        hi = min(state[i]+radius, grid[-1]-2e-4)
        edges = np.r_[lo, breaks[(breaks > lo) & (breaks < hi)], hi]
        for left, right in zip(edges[:-1], edges[1:]):
            if right-left > 64*np.finfo(float).eps*max(1., abs(state[i])):
                result.append((int(i), float(left-state[i]), float(right-state[i])))
    return result


def search_thermal_pieces(system, model, state, thermal_rows, candidate, radius,
                          *, solver=least_squares):
    """Optimize temperature-only proposals separately within adjoining pieces."""
    evaluated = compatible_thermal_model(model, state, thermal_rows)
    best = None
    trials = []
    for column, left, right in temperature_pieces(system, state, candidate, radius):
        lower, upper = -np.full(system.n, radius), np.full(system.n, radius)
        lower[column], upper[column] = left, right
        start = np.clip(candidate[:system.n], lower, upper)
        start[column] = .5*(left+right)
        solved = solver(lambda d:evaluated(d)[1][:system.n], start,
            jac=lambda d:evaluated(d)[2], bounds=(lower, upper),
            method='trf', x_scale=1., max_nfev=300,
            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        compatible, residual, _ = evaluated(solved.x)
        cost = float(.5*np.sum(residual**2))
        trials.append(dict(column=column, lower=left, upper=right,
            cost=cost, nfev=int(solved.nfev), optimality=float(solved.optimality)))
        if np.isfinite(cost) and (best is None or cost < best[0]):
            best = (cost, solved, compatible-state, residual)
    return best, trials
