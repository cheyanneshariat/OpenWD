"""Research-only lower-boundary extension; no relaxation or model substitution."""
import logging
import numpy as np
from scipy.integrate import solve_ivp
from wd_spectra._compat import trapezoid
from wd_spectra.spectrum import planck_lambda_angstrom

_LOGGER = logging.getLogger(__name__)


def extend_domain(pressure, temperature, tau_rosseland, absorption_depth,
                  wavelength, gravity, target_flux, local_transport, *,
                  escape_tolerance=3e-4, maximum_pressure_factor=1e4):
    """Append a hydrostatic ML2/diffusion seed until bottom emission is screened.

    ``local_transport(P, T)`` returns (dlnT/dlnP, kappa_R, kappa_abs[lambda]).
    All existing nodes are kept exactly. Only appended temperatures are
    provisional: a full physical relaxation must follow this construction.
    The absorption-only vertical escape estimate is conservative relative to
    oblique paths, but is a screen, not proof of boundary convergence.
    """
    p, t, tau, wave, depth = [np.asarray(x, dtype=float) for x in
        (pressure, temperature, tau_rosseland, wavelength, absorption_depth)]
    if (p.ndim != 1 or p.size < 2 or t.shape != p.shape or tau.shape != p.shape
            or wave.ndim != 1 or wave.size < 2 or depth.shape != wave.shape
            or np.any(np.diff(p) <= 0) or np.any(np.diff(wave) <= 0)
            or any(np.any(~np.isfinite(x)) for x in (p, t, tau, wave, depth))
            or any(np.any(x <= 0) for x in (p, t, tau, wave)) or np.any(depth < 0)
            or not np.isfinite(gravity) or gravity <= 0
            or not np.isfinite(target_flux) or target_flux <= 0
            or not 0 < escape_tolerance < 1 or not 1 < maximum_pressure_factor < np.inf):
        raise ValueError("Domain extension requires finite physical ordered grids and bounds")

    def leakage(logt, optical):
        return float(trapezoid(np.pi*planck_lambda_angstrom(wave, np.exp(logt))
                              * np.exp(-optical), wave)/target_flux)

    initial = np.concatenate(([np.log(t[-1]), tau[-1]], depth))
    initial_escape = leakage(initial[0], initial[2:])
    _LOGGER.info("Domain seed starts at P=%.6g T=%.6g; absorption escape %.6g, requested %.6g",
                 p[-1], t[-1], initial_escape, escape_tolerance)
    if initial_escape <= escape_tolerance:
        raise ValueError("The supplied bottom already satisfies this absorption screen")
    logp0 = np.log(p[-1])
    # A gray seed can pack its deepest nodes into an almost identical mass
    # shell. Propagating that accidental compression into new domain would
    # append hundreds of redundant layers. Use the existing mesh's median
    # logarithmic spacing; the ODE error control remains independent of it.
    # Final transfer, spectrum and further domain/refinement checks remain
    # mandatory, so this is only a seed-resolution choice.
    spacing = float(np.median(np.diff(np.log(p))))
    evaluations = 0

    def rhs(logp, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1 or evaluations % 20 == 0:
            _LOGGER.info("Domain seed evaluation %d: P=%.6g T=%.6g absorption escape %.6g",
                         evaluations, np.exp(logp), np.exp(state[0]), leakage(state[0], state[2:]))
        if evaluations > 2000:
            raise RuntimeError("Domain seed exhausted 2000 transport evaluations")
        gradient, rosseland, absorption = local_transport(np.exp(logp), np.exp(state[0]))
        coefficients = np.concatenate(([gradient, rosseland], np.asarray(absorption)))
        if (coefficients.shape != state.shape or np.any(~np.isfinite(coefficients))
                or gradient < 0 or rosseland <= 0 or np.any(coefficients[2:] < 0)):
            raise ValueError("Invalid local transport during domain extension")
        return np.concatenate(([gradient], coefficients[1:]*np.exp(logp)/gravity))

    def screened(logp, state):
        return leakage(state[0], state[2:])-escape_tolerance
    screened.terminal = True
    screened.direction = -1
    solution = solve_ivp(rhs, (logp0, logp0+np.log(maximum_pressure_factor)), initial,
                         rtol=1e-5, atol=1e-9, max_step=spacing,
                         events=screened, dense_output=True)
    if not solution.success or not solution.t_events[0].size:
        raise RuntimeError("Domain extension did not reach its absorption screen within the pressure budget")
    length = solution.t[-1]-logp0
    # Even spacing avoids an arbitrarily small last cell at an event location.
    new_logp = np.linspace(logp0, solution.t[-1], max(1, int(np.ceil(length/spacing)))+1)[1:]
    new_state = solution.sol(new_logp)
    metadata = dict(initial_absorption_escape_bound=initial_escape,
                    seed_absorption_escape_bound=leakage(new_state[0, -1], new_state[2:, -1]),
                    requested_absorption_escape_bound=escape_tolerance,
                    appended_max_log_pressure_spacing=spacing,
                    pressure_factor=float(np.exp(length)), appended_nodes=int(new_logp.size),
                    seed_rhs_evaluations=int(solution.nfev))
    return (np.concatenate((p, np.exp(new_logp))),
            np.concatenate((t, np.exp(new_state[0]))),
            np.concatenate((tau, new_state[1])), metadata)
