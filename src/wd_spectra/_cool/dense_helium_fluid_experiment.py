"""Isolated dense-He interaction calculations, NOT a production EOS.

This first stage computes atomic interaction terms without atmosphere fitting.
Neutral He: Young, McMahan & Ross (1981), PRB 24, 5119, Eq. 1,
epsilon/k=10.8 K, r_m=2.9673 Angstrom, alpha=13.1 (Sec. IV).
He--He+: Bruno et al. (2010), Phys. Plasmas 17, 112315, Eqs. 3, 6,
Sec. II.A.2. Bonding and antibonding channels are kept SEPARATE.
Electron insertion: Kowalski's 2006 thesis, Eqs. 164--166, Table 2.

PY uses the compressibility/density-integration route; the separately
requested HNC closure uses its generating-functional chemical potential.
Neither is claimed to reproduce Kowalski's published chemical potentials.
He2+--He interaction data are still required for a
complete molecular closure. No public atmosphere dispatcher imports this file.
"""
from dataclasses import dataclass
from inspect import signature
import numpy as np
from scipy.fft import dst
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, root
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import logsumexp
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS

EV = 1.602176634e-12
KB_EV = BOLTZMANN / EV


def neutral_pair_ev(radius):
    """Young et al. exp-six potential, with its physical repulsive core.

The analytic exp-six expression turns down unphysically at extremely small
r. Its inner stationary point defines the infinite core, not a fitted radius.
"""
    r = np.asarray(radius, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('pair distances must be finite and positive')
    a, rm, eps = 13.1, 2.9673, 10.8 * KB_EV
    x = r / rm
    inner = brentq(lambda z: a*(1-z)+7*np.log(z), .01, .8)
    safe = np.maximum(x, inner)
    v = eps/(a-6) * (6*np.exp(a*(1-safe)) - a/safe**6)
    return np.where(x < inner, np.inf, v)


def ion_pair_ev(radius):
    """Return the two published He--He+ channels in eV (Bruno 2010).

These transport fits omit the far-asymptotic polarization tail. Their use
in dense-fluid chemistry therefore needs comparison to actual ab initio
curves, not just a converged fluid iteration.
"""
    r = np.asarray(radius, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('pair distances must be finite and positive')
    x = r/1.081-1
    e = np.exp(-2.23*x)
    bonding = 2.4730*(e**2-2*e+.2205*x**3*(1+4.3890*x)*e**2)
    antibonding = 359.*np.exp(-4.184*r+.649*r**2-.08528*r**3)
    return np.stack((bonding, antibonding))


def electron_insertion_ev(temperature, density):
    """The thesis fit as PRINTED, with no silent smoothing or extrapolation.

T in K, rho in g/cm3. The printed join at rho=.05 is continuous but not
C1; this is deliberately not patched with a guessed coefficient. This
function alone is NOT a nonideal ionization treatment.
"""
    t, rho = np.broadcast_arrays(np.asarray(temperature, float)*KB_EV,
                                np.asarray(density, float))
    if (np.any(~np.isfinite(t)) or np.any(~np.isfinite(rho))
            or np.any(t < .1) or np.any(t > 1.5)
            or np.any(rho < 0) or np.any(rho > 2)):
        raise ValueError('electron fit outside 0.1--1.5 eV, 0--2 g/cm3 research domain')
    polynomial = np.polynomial.polynomial.polyval(t,
        [1.0313, 2.99678, 10.3747, 2.99498, -1.27844, -3.10191])
    alpha = 8.60384*np.exp(-4.83886*t+1.97586*t*t)*polynomial
    rho0, cl = .05, 4.5382
    return np.where(rho < rho0, 10*alpha*rho**2+cl*rho,
                    alpha*rho+10*alpha*rho0**2+(cl-alpha)*rho0)


@dataclass
class RadialGrid:
    """Spherical Fourier transforms: k in A^-1, r in A, no r=0 division."""
    points: int = 1023
    extent: float = 24.

    def __post_init__(self):
        if self.points < 31 or self.extent <= 0:
            raise ValueError('invalid radial grid')
        self.dr = self.extent/(self.points+1)
        self.dk = np.pi/self.extent
        self.r = np.arange(1, self.points+1)*self.dr
        self.k = np.arange(1, self.points+1)*self.dk

    def forward(self, f):
        return 2*np.pi*self.dr*dst(self.r*f, type=1, axis=-1)/self.k

    def inverse(self, f):
        return self.dk/(4*np.pi**2)*dst(self.k*f, type=1, axis=-1)/self.r

    def zero(self, f):
        # Integrand vanishes at both missing endpoints for compact f.
        return 4*np.pi*self.dr*np.sum(self.r**2*f, axis=-1)


@dataclass
class FluidState:
    gamma: np.ndarray
    direct: np.ndarray
    structure: np.ndarray
    g: np.ndarray
    residual: float
    evaluations: int


def solve_neutral_py(grid, temperature, number_density, initial=None):
    """Solve neutral OZ/PY, accepting only its actual fixed-point residual.

Number density here is atoms/A^3, NOT atoms/cm^3.
"""
    if temperature <= 0 or not np.isfinite(temperature) or number_density < 0:
        raise ValueError('invalid fluid state')
    f = np.expm1(-neutral_pair_ev(grid.r)/(KB_EV*temperature))

    def fields(gamma):
        c = f*(1+gamma)
        ck = grid.forward(c)
        s = 1/(1-number_density*ck)
        return c, s, grid.inverse((s-1)*ck)

    evaluations = 0
    def residual(gamma):
        nonlocal evaluations
        evaluations += 1
        return gamma-fields(gamma)[2]

    guess = np.zeros(grid.points) if initial is None else np.asarray(initial)
    result = root(residual, guess, method='krylov',
                  options={'fatol': 2e-10, 'maxiter': 300})
    c, s, computed = fields(result.x)
    error = float(np.max(abs(result.x-computed)))
    # Compute g from closure, avoiding cancellation in the excluded core.
    g = (1+f)*(1+result.x)
    s0 = 1/(1-number_density*grid.zero(c))
    if (not np.isfinite(error) or error > 2e-8 or np.min(g) < -2e-8
            or np.min(s) <= 0 or s0 <= 0):
        raise RuntimeError(f'neutral PY failed: rho={number_density*HELIUM_MASS*1e24:g}, '
                           f'T={temperature:g}, residual={error:g}, {result.message}')
    return FluidState(result.x, c, s, g, error, evaluations)


def solve_trace_py(grid, temperature, neutral, potential):
    """Linear trace-solute OZ/PY equations for specified solute--He potential.

    For a stack of equal-weight channels, neutral solvent components are
    identical and share the indirect correlation. Average their MAYER
    functions, not their potentials or independently solved chemical
    potentials. This is the trace limit of the multicomponent OZ system.
    """
    f = np.expm1(-np.asarray(potential)/(KB_EV*temperature))
    if f.ndim == 2:
        f = f.mean(axis=0)
    def correlation(c):
        return grid.inverse((neutral.structure-1)*grid.forward(c))
    rhs = correlation(f)
    operator = LinearOperator((grid.points, grid.points),
        matvec=lambda gamma: gamma-correlation(f*gamma), dtype=float)
    # SciPy renamed the relative tolerance without changing its meaning.
    relative_tolerance = {"rtol" if "rtol" in signature(gmres).parameters else "tol": 1e-11}
    gamma, info = gmres(operator, rhs, atol=1e-11,
                        restart=100, maxiter=50, **relative_tolerance)
    c = f*(1+gamma)
    error = np.max(abs(gamma-correlation(c)))
    g = (1+f)*(1+gamma)
    if info != 0 or error > 2e-8 or np.min(g) < -2e-8:
        raise RuntimeError(f'trace PY failed: info={info}, error={error:g}, min_g={np.min(g):g}')
    return c


def atomic_isotherm(temperature, densities, grid=None):
    """Density-route excess potentials. Explicitly incomplete: no He2+ curve."""
    grid = RadialGrid() if grid is None else grid
    rho = np.asarray(densities, float)
    if rho.ndim != 1 or rho[0] != 0 or np.any(np.diff(rho) <= 0):
        raise ValueError('isotherm must start at zero density and increase')
    n = rho/(HELIUM_MASS*1e24)
    c0 = np.empty((len(n), 2))
    s0 = np.empty(len(n))
    errors = np.empty(len(n))
    previous = None
    pair = ion_pair_ev(grid.r)
    for i, ni in enumerate(n):
        state = solve_neutral_py(grid, temperature, ni, previous)
        previous = state.gamma
        c0[i, 0] = grid.zero(state.direct)
        c0[i, 1] = grid.zero(solve_trace_py(grid, temperature, state, pair))
        s0[i] = 1/(1-ni*c0[i, 0])
        errors[i] = state.residual
        if i % 10 == 0 or i == len(n)-1:
            print(f'fluid T={temperature:g} rho={rho[i]:.5g} S0={s0[i]:.5g} '
                  f'closure={errors[i]:.3g}', flush=True)
    mu = -KB_EV*temperature*cumulative_trapezoid(c0, n, axis=0, initial=0)
    return mu, s0, errors


def solve_hnc(grid, temperature, number_density, potential, *, solvent=None, initial=None):
    """Explicit alternative fluid approximation, with positive g by construction.

    HNC is a standard integral-equation closure, not a rescue for a failed
    atmosphere. Both bulk and trace calculations must request it explicitly.
    The returned chemical potential uses its generating free-energy functional.
    """
    if (not np.isfinite(temperature) or temperature <= 0
            or not np.isfinite(number_density) or number_density < 0):
        raise ValueError('invalid HNC thermodynamic state')
    potential = np.asarray(potential, float)
    if (potential.ndim not in (1, 2) or potential.shape[-1] != grid.points
            or np.any(np.isnan(potential)) or np.any(np.isneginf(potential))):
        raise ValueError('invalid HNC pair potential')
    if initial is not None and (np.shape(initial) != (grid.points,)
                                or np.any(~np.isfinite(initial))):
        raise ValueError('invalid HNC initial correlation')
    log_boltzmann = -potential/(KB_EV*temperature)
    if log_boltzmann.ndim == 2:
        log_boltzmann = logsumexp(log_boltzmann, axis=0)-np.log(len(log_boltzmann))
    evaluations = 0

    def fields(gamma):
        h = np.expm1(log_boltzmann+gamma)
        c = h-gamma
        ck = grid.forward(c)
        s = 1/(1-number_density*ck) if solvent is None else solvent.structure
        return h, c, s, grid.inverse((s-1)*ck)

    def residual(gamma):
        nonlocal evaluations
        evaluations += 1
        return gamma-fields(gamma)[-1]

    result = root(residual, np.zeros(grid.points) if initial is None else initial,
                  method='krylov', options={'fatol': 2e-10, 'maxiter': 300})
    h, c, s, computed = fields(result.x)
    error = float(np.max(abs(result.x-computed)))
    s0 = 1/(1-number_density*grid.zero(c)) if solvent is None else 1.
    if (not np.isfinite(error) or error > 2e-8 or np.any(~np.isfinite(h))
            or np.any(~np.isfinite(s)) or np.min(s) <= 0 or s0 <= 0):
        raise RuntimeError(f'HNC failed: T={temperature:g}, n={number_density:g}, '
                           f'residual={error:g}, {result.message}')
    mu = number_density*KB_EV*temperature*grid.zero(.5*h*result.x-c)
    state = FluidState(result.x, c, s, np.exp(log_boltzmann+result.x), error, evaluations)
    return state, mu


def atomic_hnc_isotherm(temperature, densities, grid=None):
    """Atomic HNC diagnostic, still not a complete He2+ chemical model."""
    grid = RadialGrid() if grid is None else grid
    rho = np.asarray(densities, float)
    if (rho.ndim != 1 or rho.size == 0 or np.any(~np.isfinite(rho))
            or rho[0] < 0 or np.any(np.diff(rho) <= 0)):
        raise ValueError('densities must be finite, nonnegative and increasing')
    n = rho/(HELIUM_MASS*1e24)
    previous, previous_ion = None, None
    mu = np.empty((len(n), 2))
    s0 = np.empty(len(n))
    for i, ni in enumerate(n):
        neutral, mu[i, 0] = solve_hnc(grid, temperature, ni,
            neutral_pair_ev(grid.r), initial=previous)
        ion, mu[i, 1] = solve_hnc(grid, temperature, ni,
            ion_pair_ev(grid.r), solvent=neutral, initial=previous_ion)
        previous, previous_ion = neutral.gamma, ion.gamma
        s0[i] = 1/(1-ni*grid.zero(neutral.direct))
        if i % 10 == 0 or i == len(n)-1:
            print(f'HNC T={temperature:g} rho={rho[i]:.5g} mu={mu[i]} '
                  f'closure={max(neutral.residual,ion.residual):.3g}', flush=True)
    return mu, s0
