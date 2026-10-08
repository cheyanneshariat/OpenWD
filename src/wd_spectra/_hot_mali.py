"""Multilevel accelerated lambda iteration for the restricted hot H/He atom.

Opt-in preconditioning of the fixed-temperature population iteration
(``HotNLTEModel.solve_populations(..., accelerated_lambda=True)``).  Each
bound-bound transition is preconditioned with the local (diagonal) approximate
lambda operator in the Rybicki & Hummer (1991) form.  Writing the
profile-averaged mean intensity as J = psi S_new + (J_old - psi S_old), with
the line source function S, the net radiative rate of a line becomes linear in
the new populations:

    upward   n_l a (J - psi S_old)
    downward n_u [s + d J - psi (s + d S_old)]

for absorption a J, spontaneous s and stimulated d J rates.  At a fixed point
the old and new populations agree, so the preconditioned equations have the
same solution as the unpreconditioned ones: only the iteration path changes.
``psi`` is the profile average (the same profile and frequency weights as the
mean intensity) of the diagonal operator times the line's share of the total
extinction, so overlapping lines and continua do not claim the whole local
operator.  The clamps keep every preconditioned rate non-negative, as in the
trace-metal MALI.  Continua are not preconditioned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import helium_nlte as he
from . import multilevel_nlte as hydrogen
from ._compat import trapezoid
from ._hot_rates import PreparedHeliumRates
from .constants import BOLTZMANN, LIGHT_SPEED, PLANCK
from .hot_nlte import HotPopulationState, population_arrays
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import radiation_field

_LIGHT_SPEED_ANGSTROM = LIGHT_SPEED * 1.0e8
_MAXIMUM_OPERATOR = 0.999


def diagonal_lambda_operator(atmosphere, coefficients, n_angle):
    """Diagonal Lambda* of the current extinction (used only as a preconditioner)."""
    total = np.maximum(coefficients.true_absorption + coefficients.scattering, 1e-300)
    depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    increment = np.diff(depth, axis=-1)
    floor = 1e-14 * np.maximum(abs(depth[..., 1:]), 1e-300)
    if np.any(increment <= floor):
        depth = np.concatenate((depth[..., :1], depth[..., :1]
                                + np.cumsum(np.maximum(increment, floor), axis=-1)), axis=-1)
    source = np.ascontiguousarray(coefficients.thermal_emissivity / total)
    return radiation_field(depth, source, n_angle=n_angle,
                           calculate_diagonal_lambda=True).diagonal_lambda


@dataclass(frozen=True)
class LineCouplings:
    """Rates of one transition group, linear in the profile-averaged J_nu."""
    keys: tuple
    lower: np.ndarray        # rate-matrix index of the lower state
    upper: np.ndarray        # rate-matrix index of the upper state
    absorption: np.ndarray   # (depth, line): upward rate per unit J_nu
    spontaneous: np.ndarray  # (depth, line)
    stimulated: np.ndarray   # (depth, line): downward rate per unit J_nu
    frequency: np.ndarray    # (line,) Hz


def helium_couplings(prepared: PreparedHeliumRates):
    atom = prepared.atom
    groups = []
    for group_number, (keys, lower, upper, absorption, stimulated) in enumerate(prepared.line_updates):
        frequency = []
        for lo, hi in keys:
            if group_number == 0:
                frequency.append(atom.threshold_frequency_hz[lo - 1] - atom.threshold_frequency_hz[hi - 1])
            else:
                frequency.append(_LIGHT_SPEED_ANGSTROM / he.helium_ii_shell_transition(lo, hi).wavelength_vacuum_angstrom)
        frequency = np.asarray(frequency, dtype=np.float64)
        spontaneous = stimulated * 2.0 * PLANCK * frequency**3 / LIGHT_SPEED**2
        groups.append(LineCouplings(tuple(keys), lower, upper, absorption, spontaneous, stimulated, frequency))
    return groups


def hydrogen_couplings(atmosphere, keys, maximum_level):
    lte, _, occupation = hydrogen._reference_populations(atmosphere, maximum_level)
    one, zero = np.ones(atmosphere.n_depth), np.zeros(atmosphere.n_depth)
    absorption, spontaneous, stimulated, frequency, lower, upper = [], [], [], [], [], []
    for lo, hi in keys:
        line = (hydrogen.hydrogen_shell_transition(lo, hi) if hi <= 9
                else hydrogen._extended_hydrogen_shell_transition(lo, hi))
        args = (atmosphere.temperature, line, lte[:, lo - 1], lte[:, hi - 1],
                occupation[:, lo - 1], occupation[:, hi - 1])
        up, down = hydrogen._bound_bound_radiative_rates(*args, one)
        _, spont = hydrogen._bound_bound_radiative_rates(*args, zero)
        absorption.append(up); spontaneous.append(spont); stimulated.append(down - spont)
        frequency.append(_LIGHT_SPEED_ANGSTROM / line.wavelength_vacuum_angstrom)
        lower.append(lo - 1); upper.append(hi - 1)
    return LineCouplings(tuple(keys), np.asarray(lower), np.asarray(upper), np.array(absorption).T,
                         np.array(spontaneous).T, np.array(stimulated).T, np.asarray(frequency))


def line_operator(problems, wave, total_extinction, diagonal, departure_lower, departure_upper,
                  frequency, temperature):
    """Profile average of Lambda* times the line's share of the total extinction."""
    stimulated_factor = np.exp(-PLANCK * frequency / (BOLTZMANN * temperature))
    correction = np.maximum(departure_lower - departure_upper * stimulated_factor, 0.0) / (1.0 - stimulated_factor)
    values, weights = [], []
    for problem in problems:
        wavelength = problem.continuum.wavelength_angstrom
        index = np.searchsorted(wave, wavelength)
        profile = problem.lte_line_opacity * (_LIGHT_SPEED_ANGSTROM / wavelength[:, None] ** 2)
        share = np.clip(problem.lte_line_opacity * correction[None, :]
                        / np.maximum(total_extinction[index], 1e-300), 0.0, 1.0)
        denominator = trapezoid(profile, wavelength, axis=0)
        values.append(np.divide(trapezoid(profile * diagonal[index] * share, wavelength, axis=0), denominator,
                                out=np.zeros_like(denominator), where=denominator > 0.0))
        weights.append(problem.line.absorption_oscillator_strength)
    return np.clip(np.average(values, axis=0, weights=weights), 0.0, _MAXIMUM_OPERATOR)


def precondition(rate, couplings, fields, operators, populations):
    """Apply the Rybicki-Hummer substitution to the line rates of ``rate`` (in place)."""
    epsilon = 16.0 * np.finfo(float).eps
    for index, key in enumerate(couplings.keys):
        lo, hi = couplings.lower[index], couplings.upper[index]
        absorption = couplings.absorption[:, index]
        spontaneous = couplings.spontaneous[:, index]
        stimulated = couplings.stimulated[:, index]
        mean = np.asarray(fields[key], dtype=np.float64)
        n_lower, n_upper = populations[:, lo], populations[:, hi]
        denominator = n_lower * absorption - n_upper * stimulated
        valid = (denominator > 0.0) & (operators[key] > 0.0)
        source = np.zeros_like(mean)
        source[valid] = n_upper[valid] * spontaneous[valid] / denominator[valid]
        valid &= np.isfinite(source) & (source > 0.0)
        if not np.any(valid):
            continue
        limit = np.minimum(
            np.divide(mean, source, out=np.zeros_like(mean), where=valid),
            np.divide(spontaneous + stimulated * mean, spontaneous + stimulated * source,
                      out=np.zeros_like(mean), where=valid))
        psi = np.where(valid, np.minimum(operators[key], limit), 0.0) * (1.0 - epsilon)
        rate[:, lo, hi] -= absorption * psi * source
        rate[:, hi, lo] -= psi * (spontaneous + stimulated * source)
    return rate


class HotMALI:
    """Preconditioned rate states for one fixed atmosphere and line grid."""

    def __init__(self, model, atmosphere, wave, groups):
        self.model, self.atmosphere, self.wave, self.groups = model, atmosphere, wave, groups
        self.prepared = PreparedHeliumRates(model, atmosphere, wave, groups)
        self.helium = helium_couplings(self.prepared)
        self.hydrogen = (None if model.log_hydrogen_to_helium is None or not groups[2] else
                         hydrogen_couplings(atmosphere, tuple(groups[2]), model.maximum_hydrogen_level))
        self.helium_states = 1 + self.prepared.n_neutral + model.maximum_helium_ii_level

    def _operators(self, couplings, group, wave, total, diagonal, departure):
        operators = {}
        for index, key in enumerate(couplings.keys):
            lo, hi = couplings.lower[index], couplings.upper[index]
            operators[key] = line_operator(group[key], wave, total, diagonal, departure[:, lo], departure[:, hi],
                                           couplings.frequency[index], self.atmosphere.temperature)
        return operators

    def state(self, current, coefficients, mean, fields):
        """SE solution of the preconditioned rates for J computed from ``current``."""
        diagonal = diagonal_lambda_operator(self.atmosphere, coefficients, self.model.n_angle)
        total = coefficients.true_absorption + coefficients.scattering
        actual, reference = population_arrays(current)
        departure = actual / reference
        nhe = self.helium_states
        rate = self.prepared.rate_matrix(mean, fields[0], fields[1])
        self.last_operators = {}
        for group_number, couplings in enumerate(self.helium):
            operators = self._operators(couplings, self.groups[group_number], self.wave, total, diagonal,
                                        departure[:, :nhe])
            self.last_operators.update({(group_number, k): v for k, v in operators.items()})
            precondition(rate, couplings, fields[group_number], operators, actual[:, :nhe])
        helium = he.solve_coupled_helium_statistical_equilibrium(
            self.atmosphere, self.model.collision_data, **self.prepared.kwargs,
            neutral_line_mean_intensity_nu=fields[0], helium_ii_line_mean_intensity_nu=fields[1],
            _rate_matrix=rate)
        hydrogen_state = None
        if self.model.log_hydrogen_to_helium is not None:
            transform = None
            if self.hydrogen is not None:
                operators = self._operators(self.hydrogen, self.groups[2], self.wave, total, diagonal,
                                            departure[:, nhe:])
                self.last_operators.update({(2, k): v for k, v in operators.items()})

                def transform(matrix):
                    return precondition(matrix, self.hydrogen, fields[2], operators, actual[:, nhe:])
            hydrogen_state = hydrogen.solve_multilevel_hydrogen_statistical_equilibrium(
                self.atmosphere, self.model.collision_data, maximum_level=self.model.maximum_hydrogen_level,
                line_mean_intensity_nu=fields[2], continuum_wavelength_angstrom=self.wave,
                continuum_mean_intensity_lambda=mean, _rate_matrix_transform=transform)
        return HotPopulationState(helium, hydrogen_state)
