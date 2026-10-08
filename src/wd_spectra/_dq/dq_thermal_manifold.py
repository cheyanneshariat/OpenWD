"""Research-only pseudo-time proposal with exact algebraic ML2 elimination.

This cheap-model helper changes neither the transient equation nor its units.
It proposes temperatures on the model's compatible convection manifold; a
caller must still evaluate full physics and enforce all acceptance gates.
"""
import numpy as np
from scipy.optimize import least_squares


def compatible_thermal_model(model, state, thermal_rows):
    """Return (full state, residual, reduced Jacobian) for a temperature step."""
    n = model.n
    cache = [None, None]

    def evaluated(delta):
        if cache[0] is None or not np.array_equal(cache[0], delta):
            logt = state[:n]+delta
            physical = model.physical(logt, True)
            compatible = model.state_from_temperature(logt, physical)
            full = model.evaluate(compatible, True)
            residual, jacobian = thermal_rows(compatible, full)
            mapping = model.compatible_state_jacobian(logt, physical)
            if mapping.shape != (state.size, n):
                raise ValueError('Compatible-state tangent has the wrong shape')
            # Compatibility is enforced algebraically, not downweighted.
            if np.max(abs(residual[n:])) > 1e-9:
                raise ValueError('Compatible thermal trial violates algebraic closure')
            cache[:] = [delta.copy(), (compatible, residual, jacobian[:n]@mapping)]
        return cache[1]

    return evaluated


def solve_compatible_thermal(model, state, thermal_rows, radius, *, solver=least_squares):
    evaluated = compatible_thermal_model(model, state, thermal_rows)
    solved = solver(lambda d:evaluated(d)[1][:model.n], np.zeros(model.n),
        jac=lambda d:evaluated(d)[2],
        bounds=(-np.full(model.n, radius), np.full(model.n, radius)),
        method='trf', x_scale=1., max_nfev=300,
        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    compatible, residual, _ = evaluated(solved.x)
    return solved, compatible-state, residual
