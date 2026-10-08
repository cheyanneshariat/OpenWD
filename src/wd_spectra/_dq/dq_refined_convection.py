"""Research top-down convective correction following Hubeny (2017) §5.4.3.

Hold alpha = F_rad/(T_interface**4 * gradient) fixed during the correction,
but solve the local radiative+ML2 flux balance for temperature, not a flux
cap. Update the actual EOS/opacity/ML2 coefficients between inner sweeps.
This is a material-updated implementation of the refined correction; the
approximate radiative response is ONLY a proposal. Refractive formal
transfer and every physical convergence gate remain mandatory afterwards.

No temperature switch, EOS replacement, stored atmosphere, flux clipping
or synthetic convergence. The default bracket is physical (zero flux to
positive excess flux), independent of Newton's shrinking trust radius.
Unlike TLUSTY's historical 0.999 flux cap, a radiatively stable local root
is allowed; a missing bracket or unresolved EOS discontinuity rejects it.
"""
import numpy as np
from scipy.optimize import brentq
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from wd_spectra.nonlinear import RecoverableEvaluationError

REFERENCE = 'https://doi.org/10.1093/mnras/stx758'


def established_zone(gradient, adiabatic):
    """Retain non-isolated convection and bridge interior stable holes.

    This is a temporary correction region, not a changed stability law.
    Every returned temperature is subsequently tested with actual ML2.
    """
    active = np.asarray(gradient) > np.asarray(adiabatic)
    active = active.copy(); active[0] = False
    pairs = np.flatnonzero(active[:-1] & active[1:])
    selected = np.zeros_like(active)
    if len(pairs):
        selected[pairs[0]:pairs[-1]+2] = True
    return selected


class ConvectionHistory:
    """Remember the proposal region at accepted states of this calculation.

    A temporarily stable/fragmented iterate must not erase an established
    zone. This history only selects where to try a correction; actual ML2
    flux and final stability are always recomputed without this mask.
    Pressure-coordinate overlap carries history through same-run extension.
    """
    def __init__(self):
        self.pressure = None
        self.selected = None

    def observe(self, pressure, gradient, adiabatic):
        pressure = np.asarray(pressure, float)
        selected = established_zone(gradient, adiabatic)
        if self.pressure is not None and np.any(self.selected):
            bounds = self.pressure[self.selected]
            selected |= (pressure >= bounds[0]) & (pressure <= bounds[-1])
        selected[0] = False
        self.pressure, self.selected = pressure.copy(), selected.copy()
        return selected


def refined_correction(logt, pressure, radiative_flux, materials, target,
                       radius=None, *, active=None, progress=None, maximum_sweeps=30,
                       through_bottom=False):
    x0 = np.asarray(logt, float)
    p = np.asarray(pressure, float)
    fr = np.asarray(radiative_flux, float)
    if (x0.ndim != 1 or len(x0) < 3 or p.shape != x0.shape or fr.shape != x0.shape
            or not np.all(np.isfinite(x0+p+fr)) or np.any(p <= 0)
            or np.any(np.diff(p) <= 0) or not np.isfinite(target) or target <= 0
            or (radius is not None and (not np.isfinite(radius) or radius <= 0))):
        raise ValueError('Finite ordered pressure and positive target/trust radius required')
    spacing = np.r_[1., np.diff(np.log(p))]
    gradient0 = np.r_[0., np.diff(x0)/spacing[1:]]
    values = materials(x0)
    selected = established_zone(gradient0, values[0]) if active is None else np.asarray(active, bool).copy()
    if selected.shape != x0.shape or selected[0]:
        raise ValueError('Invalid correction region')
    # §5.3 also describes solving radiative/convective equilibrium below the
    # convection zone. This optional diagnostic includes those layers but
    # never forces them convective: the same ML2 law permits F_conv=0.
    if through_bottom and np.any(selected):
        selected[np.flatnonzero(selected)[0]:] = True
    # The diffusion-shaped response has no meaningful positive coefficient
    # at a nonpositive gradient/radiative flux. Do not invent one there.
    invalid = selected & ((gradient0 <= 0) | (fr <= 0))
    if np.any(invalid):
        raise RecoverableEvaluationError('Refined radiative response undefined in selected layer')
    x = x0.copy()
    logs_mid0 = .5*(x0+np.r_[x0[0], x0[:-1]])
    if not np.any(selected):
        return x, dict(corrected_layers=[], material_calls=1, maximum_approximate_flux_error=0.)
    for sweep in range(maximum_sweeps):
        anchor = x.copy()
        mid_anchor = .5*(anchor+np.r_[anchor[0], anchor[:-1]])
        ad, loss, coefficient = values
        for i in np.flatnonzero(selected):
            def defect(value):
                mid = .5*(value+x[i-1])
                gradient = (value-x[i-1])/spacing[i]
                # Hubeny Eq. 165: hold beta=B/T^3 and F0 during the local
                # solve. The outer material sweep updates all other factors.
                trial_loss = loss[i:i+1]*np.exp(3*(mid-mid_anchor[i]))
                y = ml2_auxiliary_from_gradient(np.array([gradient]), ad[i:i+1],
                    trial_loss, coefficient[i:i+1], target)[0]
                radiation = fr[i]/target*np.exp(4*(mid-logs_mid0[i]))*gradient/gradient0[i]
                return float(radiation+max(y, 0.)**3-1)
            if radius is None:
                # At T_i=T_{i-1}, gradient and outward radiative/ML2 flux
                # vanish. Expand a positive-gradient bracket until it carries
                # the target flux; no Teff-dependent or Newton-radius bound.
                lo = x[i-1]
                width = max(abs(x0[i]-lo), spacing[i]*max(ad[i], 0.), 1e-6)
                hi = lo+width
                for _ in range(60):
                    fhi = defect(hi)
                    if not np.isfinite(fhi):
                        raise RecoverableEvaluationError(f'Nonfinite refined root bracket at layer {i}')
                    if fhi >= 0:
                        break
                    width *= 2
                    hi = lo+width
                else:
                    raise RecoverableEvaluationError(f'Refined root bracket exhausted at layer {i}')
            else:
                # Explicit bounded mode remains for old one-step diagnostics.
                lo, hi = x0[i]-radius, x0[i]+radius
            flo, fhi = defect(lo), defect(hi)
            if not np.isfinite(flo+fhi) or flo*fhi > 0:
                raise RecoverableEvaluationError(f'Refined temperature root not bracketed at layer {i}')
            x[i] = brentq(defect, lo, hi, xtol=2e-14,
                          rtol=8*np.finfo(float).eps, maxiter=100)
        # Recompute the SAME full material law and test all rows together.
        # No interpolated/stale coefficient is allowed to pass this check.
        values = materials(x)
        gradient = np.r_[0., np.diff(x)/spacing[1:]]
        y = ml2_auxiliary_from_gradient(gradient, *values, target)
        mid = .5*(x+np.r_[x[0], x[:-1]])
        residual = (fr[selected]/target*np.exp(4*(mid[selected]-logs_mid0[selected]))*
            gradient[selected]/gradient0[selected] + np.maximum(y[selected], 0.)**3-1)
        worst = float(np.max(abs(residual)))
        change = float(np.max(abs(x-anchor)))
        if progress is not None:
            progress(dict(sweep=sweep, maximum_log_temperature_update=change,
                          actual_material_local_flux_error=worst, material_calls=sweep+2))
        if worst <= 1e-6 and change <= 1e-9:
            return x, dict(corrected_layers=np.flatnonzero(selected).tolist(),
                material_calls=sweep+2, maximum_approximate_flux_error=worst)
    raise RecoverableEvaluationError('Refined material sweeps did not converge; no approximate EOS accepted')
