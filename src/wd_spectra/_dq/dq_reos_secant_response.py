"""Analytic response of the unchanged, finite-window REOS thermodynamics.

Differentiate the piecewise log-P/log-T interpolant, including moving window
endpoints, rather than finite-differencing its already differenced cp and Q.
At an exact knot use the mean of the two one-sided slopes (a generalized
derivative); outside the supplied table reject, never extrapolate. Pressure
knots and isotherms with different pressure grids follow the same rule.
"""
import numpy as np
from wd_spectra.dense_eos import KJ_G_TO_ERG_G
from .dq_reos_secant_thermodynamics import reos_secant_thermodynamics


def _linear_slope(grid, values, point):
    slopes = np.diff(values) / np.diff(grid)
    lo = np.clip(np.searchsorted(grid, point, side='left')-1, 0, len(slopes)-1)
    hi = np.clip(np.searchsorted(grid, point, side='right')-1, 0, len(slopes)-1)
    # exp/log round trips can move a mathematically exact knot by a few ulps.
    nearest = np.clip(np.searchsorted(grid, point), 1, len(grid)-1)
    near = np.where(abs(point-grid[nearest-1]) < abs(point-grid[nearest]), nearest-1, nearest)
    at = abs(point-grid[near]) <= 8*np.finfo(float).eps*np.maximum(1., abs(point))
    lo = np.where(at, np.clip(near-1, 0, len(slopes)-1), lo)
    hi = np.where(at, np.clip(near, 0, len(slopes)-1), hi)
    return .5*(slopes[lo]+slopes[hi])


def table_partials(table, temperature, pressure):
    """Return d(ln rho,u)/d(ln T,ln P), with u in erg/g."""
    t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
    if not np.all(table.evaluate(p, t)[2]):
        raise ValueError('REOS response outside supplied table domain')
    x, z = np.log(t).ravel(), np.log(p).ravel()
    grid = np.log(table.temperature_grid)
    lower = np.clip(np.searchsorted(grid, x, side='right')-1, 0, len(grid)-2)
    result = np.empty((4, x.size))
    # Only the isotherms bracketing the requested temperatures are needed.
    cache = {}
    def iso(i):
        if i not in cache:
            lr, u, inside = table._at_isotherm(i, z)
            gp = np.log(table.pressure_by_temperature[i])
            rz = _linear_slope(gp, np.log(table.density_by_temperature[i]), z)
            uz = _linear_slope(gp, table.internal_energy_by_temperature[i], z)
            cache[i] = lr, u, rz, uz, inside
        return cache[i]
    for i in np.unique(lower):
        sel = lower == i
        r0, u0, rz0, uz0, in0 = iso(i)
        r1, u1, rz1, uz1, in1 = iso(i+1)
        if np.any(sel & ~(in0 & in1)):
            raise ValueError('REOS response crosses unsupported table cell')
        w = (x-grid[i])/(grid[i+1]-grid[i])
        result[:, sel] = np.asarray([(r1-r0)/(grid[i+1]-grid[i]),
            (u1-u0)/(grid[i+1]-grid[i]), (1-w)*rz0+w*rz1, (1-w)*uz0+w*uz1])[:, sel]
    for i in range(1, len(grid)-1):
        sel = abs(x-grid[i]) <= 8*np.finfo(float).eps*np.maximum(1., abs(x))
        if not np.any(sel):
            continue
        left, mid, right = iso(i-1), iso(i), iso(i+1)
        if np.any(sel & ~(left[4] & mid[4] & right[4])):
            raise ValueError('REOS knot response crosses unsupported table cell')
        for k in (0, 1):
            result[k, sel] = .5*((mid[k]-left[k])/(grid[i]-grid[i-1]) +
                                (right[k]-mid[k])/(grid[i+1]-grid[i]))[sel]
        result[2, sel] = mid[2][sel]
        result[3, sel] = mid[3][sel]
    result[[1, 3]] *= KJ_G_TO_ERG_G
    return result.reshape((4,)+t.shape)


def thermodynamic_response(table, temperature, pressure, pressure_log_response=0., *, epsilon=2e-4):
    """d(rho, cp, Q, nabla_ad)/d ln T along the supplied host-pressure path.

    The value evaluator remains reos_secant_thermodynamics, unchanged.
    pressure_log_response is d ln P_host / d ln T at fixed total pressure.
    """
    if hasattr(table,'thermodynamic_response'):
        return table.thermodynamic_response(temperature,pressure,pressure_log_response)
    t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
    th = reos_secant_thermodynamics(table, t, p, epsilon=epsilon)
    rho = table.evaluate(p, t)[0]
    rc = table.evaluate(p, t*np.exp(-epsilon))[0]
    r, u, rz, uz = table_partials(table, t, p)
    r0, u0, rz0, uz0 = table_partials(table, t*np.exp(-epsilon), p)
    r1, u1, rz1, uz1 = table_partials(table, t*np.exp(epsilon), p)
    q = np.asarray(pressure_log_response)
    log_rho_prime = r+rz*q
    log_rc_prime = r0+rz0*q
    dr = -2*epsilon*th.density_temperature_derivative
    dr_prime = r1-r0+(rz1-rz0)*q
    du_prime = u1-u0+(uz1-uz0)*q
    numerator_prime = du_prime+p/rc*((q-log_rc_prime)*np.expm1(-dr)-np.exp(-dr)*dr_prime)
    cp = th.specific_heat_constant_pressure
    cp_prime = numerator_prime/(2*t*np.sinh(epsilon))-cp
    expansion = th.density_temperature_derivative
    expansion_prime = -dr_prime/(2*epsilon)
    ad_prime = th.adiabatic_temperature_gradient*(
        q+expansion_prime/expansion-log_rho_prime-1-cp_prime/cp)
    return np.asarray([rho*log_rho_prime, cp_prime, expansion_prime, ad_prime])
