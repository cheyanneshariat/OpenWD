"""Thermodynamically self-consistent neutral-He structure factor (build time).

Rogers & Young (1984), Phys. Rev. A 30, 999, doi:10.1103/PhysRevA.30.999:

    g(r) = exp[-phi/kT] {1 + (exp[f(r) gamma(r)] - 1)/f(r)},
    f(r) = 1 - exp(-alpha r).

alpha -> 0 is Percus-Yevick, alpha -> infinity is hypernetted chain. alpha
is NOT fitted to any data: at each (T, n) it is the unique value for which
the virial-route compressibility d(beta P_vir)/dn equals the fluctuation
(OZ) route 1/S(0). Percus-Yevick alone violates this by up to a factor 2.6
in 1/S(0) at 1.8 g/cm3 and underestimates S(0) against the He-REOS.3 ab
initio compressibility by 3x there; see
docs/development/history/dq-dense-he-ff-2026-10-15.md.

Pair potential: Ross & Young (1986) exp-6, as in ``solve_neutral_py``.
Lengths in Angstrom, energies in eV, number density in atoms/A^3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

from .._cool.dense_helium_fluid_experiment import (
    KB_EV, FluidState, RadialGrid, neutral_pair_ev,
)

__all__ = ["RogersYoungState", "solve_rogers_young", "solve_consistent_rogers_young",
           "RadialGrid"]

_A, _RM, _EPS = 13.1, 2.9673, 10.8*KB_EV
# Below this |S(0; PY end) - S(0; HNC end)|/S(0) the closures agree to the
# quadrature/derivative noise and alpha has no physical content.
_CLOSURE_INDEPENDENT = 1e-4


@dataclass(frozen=True)
class RogersYoungState:
    alpha: float              # 1/Angstrom
    state: FluidState
    structure_zero: float     # OZ route 1/(1 - n c(0))
    virial_structure_zero: float  # 1/(d beta P_vir / dn)


def _core_radius():
    """Inner stationary point of the exp-6 potential, in Angstrom."""
    return brentq(lambda x: _A*(1-x)+7*np.log(x), .01, .8)*_RM


def _boltzmann(r, temperature):
    """exp(-beta phi) and its r-derivative; zero inside the exp-6 core."""
    x = r/_RM
    e = np.exp(-neutral_pair_ev(r)/(KB_EV*temperature))
    dphi = _EPS/(_A-6)/_RM*(-6*_A*np.exp(_A*(1-x)) + 6*_A*x**-7)
    return e, np.where(e > 0, -dphi/(KB_EV*temperature)*e, 0.)


def solve_rogers_young(grid, temperature, number_density, alpha, initial=None):
    """Solve OZ with the RY closure at fixed alpha; return (state, S0, beta P)."""
    if (not np.isfinite(temperature) or temperature <= 0 or number_density < 0
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("invalid Rogers-Young state")
    e, de = _boltzmann(grid.r, temperature)
    f = -np.expm1(-alpha*grid.r)

    def fields(gamma):
        # exp overflow is clipped only far inside the core, where e == 0.
        y = 1 + np.expm1(np.minimum(f*gamma, 700.))/f
        c = e*y - 1 - gamma
        ck = grid.forward(c)
        s = 1/(1-number_density*ck)
        return c, s, grid.inverse((s-1)*ck), y

    guess = np.zeros(grid.points) if initial is None else np.asarray(initial)
    result = root(lambda gamma: gamma-fields(gamma)[2], guess, method="krylov",
                  options={"fatol": 2e-10, "maxiter": 400})
    c, s, computed, y = fields(result.x)
    error = float(np.max(np.abs(result.x-computed)))
    s0 = 1/(1-number_density*grid.zero(c))
    if (not np.isfinite(error) or error > 2e-8 or np.min(s) <= 0 or s0 <= 0
            or np.any(~np.isfinite(y))):
        raise RuntimeError(f"Rogers-Young OZ failed: T={temperature:g}, n={number_density:g}, "
                           f"alpha={alpha:g}, residual={error:g}")
    # beta P = n + (2 pi n^2/3) int r^3 y(r) d exp(-beta phi)/dr dr  (cavity form).
    # exp(-beta phi) jumps from 0 to its value at the exp-6 core radius sigma
    # (inner stationary point, ~4 eV): that delta function is the hard-wall
    # contact term (2 pi n^2/3) sigma^3 g(sigma+), important above ~1e4 K.
    sigma = _core_radius()
    contact = sigma**3*np.interp(sigma, grid.r, y)*np.exp(-neutral_pair_ev(sigma*(1+1e-12))/(KB_EV*temperature))
    beta_p = number_density + 2*np.pi*number_density**2/3*(np.sum(grid.r**3*y*de)*grid.dr + contact)
    return FluidState(result.x, c, s, e*y, error, 0), s0, beta_p


def _mismatch(grid, temperature, n, alpha, initial, h=1e-2):
    state, s0, _ = solve_rogers_young(grid, temperature, n, alpha, initial)
    plus = solve_rogers_young(grid, temperature, n*(1+h), alpha, state.gamma)[2]
    minus = solve_rogers_young(grid, temperature, n*(1-h), alpha, state.gamma)[2]
    dbp = (plus-minus)/(2*h*n)
    return 1/s0-dbp, state, s0, 1/dbp


def solve_consistent_rogers_young(grid, temperature, number_density, initial=None,
                                  alpha_guess=0.1, widen=8):
    """Find the self-consistent alpha by bracketing in log(alpha).

    The bracket starts at alpha_guess/1.3..*1.3 (density continuation) and is
    widened by 3x per side until the mismatch changes sign. If the PY-like and
    HNC-like ends agree in S(0) to 1e-4, the state is closure independent and
    the bracket midpoint is returned. Anything else raises: no closure switch.
    """
    lo, hi = np.log(alpha_guess/1.3), np.log(alpha_guess*1.3)
    for _ in range(widen):
        flo, _, slo, _ = _mismatch(grid, temperature, number_density, np.exp(lo), initial)
        fhi, _, shi, _ = _mismatch(grid, temperature, number_density, np.exp(hi), initial)
        if flo*fhi <= 0:
            la = brentq(lambda v: _mismatch(grid, temperature, number_density,
                                            np.exp(v), initial)[0], lo, hi, xtol=1e-4)
            break
        if abs(slo-shi) < _CLOSURE_INDEPENDENT*shi:
            la = .5*(lo+hi)
            break
        lo, hi = max(lo-np.log(3), np.log(1e-4)), min(hi+np.log(3), np.log(1e3))
    else:
        raise RuntimeError(f"no Rogers-Young consistency root: T={temperature:g}, "
                           f"n={number_density:g}, mismatch={flo:g},{fhi:g}")
    _, state, s0, s0_virial = _mismatch(grid, temperature, number_density, np.exp(la), initial)
    return RogersYoungState(float(np.exp(la)), state, float(s0), float(s0_virial))
