"""Research-only finite-material convection repair, inspired by CONREF.

Unlike the old over-flux cap, this recomputes the SAME EOS and Rosseland
opacity during the temperature correction. The proposed ML2 element velocity
is held fixed while gradient/temperature/material compatibility is solved.
Neither an energy equation nor a physical flux is replaced. The outer solver
must evaluate and accept the resulting atmosphere with its usual gates.
"""
from dataclasses import replace
import logging
import numpy as np

from wd_spectra.adaptive_structure import (
    _positive_interface_values,
    _arithmetic_interface_values,
)
from wd_spectra.convection import _ml2_local_coefficients_from_thermodynamics
from wd_spectra._ml2_auxiliary import (
    ml2_auxiliary_from_gradient,
    ml2_auxiliary_compatibility,
)
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import RecoverableEvaluationError

LOGGER = logging.getLogger(__name__)


class MaterialCoefficients:
    """Exact production material coefficients; local, fixed-pressure tangent."""

    def __init__(self, with_temperature, thermodynamics, rosseland_opacity, alpha):
        self.with_temperature = with_temperature
        self.thermodynamics = thermodynamics
        self.rosseland_opacity = rosseland_opacity
        self.alpha = alpha

    def fields(self, logt):
        atmosphere = self.with_temperature(np.exp(logt))
        thermo = self.thermodynamics(atmosphere)
        return (
            atmosphere,
            np.asarray(
                (
                    atmosphere.mass_density,
                    self.rosseland_opacity(atmosphere),
                    thermo.specific_heat_constant_pressure,
                    thermo.density_temperature_derivative,
                    thermo.adiabatic_temperature_gradient,
                )
            ),
        )

    def __call__(self, logt, derivative=False):
        atmosphere, fields = self.fields(logt)
        slopes = None
        if derivative:
            h = 2e-4
            slopes = (self.fields(logt + h)[1] - self.fields(logt - h)[1]) / (2 * h)
        return self.assemble(atmosphere, fields, slopes)

    def assemble(self, atmosphere, fields, slopes=None):
        rho, opacity, cp, expansion, adiabatic = fields
        positive, arithmetic = _positive_interface_values, _arithmetic_interface_values
        interface = replace(
            atmosphere,
            temperature=positive(atmosphere.temperature),
            gas_pressure=positive(atmosphere.gas_pressure),
            mass_density=positive(rho),
        )
        coefficients = _ml2_local_coefficients_from_thermodynamics(
            interface,
            positive(opacity),
            arithmetic(cp),
            arithmetic(expansion),
            arithmetic(adiabatic),
            self.alpha,
        )
        if slopes is None:
            return coefficients
        # All node quantities are local at fixed P, so simultaneous node
        # perturbations recover their diagonal derivatives in two EOS calls.
        n = len(atmosphere.temperature)
        average = np.eye(n) * 0.5 + np.eye(n, k=-1) * 0.5
        average[0, 0] = 1.0
        log_rho_j = average * (slopes[0] / rho)[None, :]
        log_opacity_j = average * (slopes[1] / opacity)[None, :]
        log_cp_j = average * slopes[2][None, :] / arithmetic(cp)[:, None]
        log_expansion_j = average * slopes[3][None, :] / arithmetic(expansion)[:, None]
        ad_j = average * slopes[4][None, :]
        tau = (
            positive(opacity) * self.alpha * interface.gas_pressure / interface.gravity
        )
        log_loss_j = (
            3 * average
            - log_cp_j
            - 0.5 * log_rho_j
            - 0.5 * log_expansion_j
            + (2 / (1 + 0.5 * tau ** 2) - 1)[:, None] * log_opacity_j
        )
        log_coefficient_j = log_cp_j + 0.5 * log_rho_j + average + 0.5 * log_expansion_j
        return (
            coefficients,
            (
                ad_j,
                coefficients[1][:, None] * log_loss_j,
                coefficients[2][:, None] * log_coefficient_j,
            ),
        )


class TabulatedMaterials:
    """Smooth finite-material proposal, with an independent accuracy check.

    This table lives only for one local trust region. It never supplies the
    accepted EOS, opacity, flux or convergence diagnostic. No extrapolation.
    """

    def __init__(self, exact, logt, radius):
        from numpy.polynomial import chebyshev as ch

        self.exact, self.logt, self.radius = exact, logt, radius
        self.atmosphere, base = exact.fields(logt)

        def encode(fields):
            return np.vstack((np.log(fields[:4]), fields[4]))

        self.base = encode(base)
        # Keep the existing log-linear Rosseland response: tabulated opacity
        # knots are only piecewise smooth. Only EOS quantities are represented
        # by the smooth potential and included in its accuracy requirement.
        h = 2e-4
        opacity_slope = (
            np.log(exact.fields(logt + h)[1][1]) - np.log(exact.fields(logt - h)[1][1])
        ) / (2 * h)
        for count in (9, 17):
            nodes = np.cos(np.linspace(0, np.pi, count))
            values = []
            for i, node in enumerate(nodes):
                values.append(encode(exact.fields(logt + radius * node)[1]))
                LOGGER.info("Material proposal table: %d/%d nodes", i + 1, count)
            coefficients = ch.chebfit(
                nodes, np.asarray(values).reshape(count, -1), count - 1
            )
            self.coefficients = coefficients.reshape(count, *self.base.shape)
            # Anchor the true base exactly; do not perturb the residual whose
            # tangent/proposal the outer solver requested.
            self.coefficients[0] += self.base - ch.chebval(0.0, self.coefficients)
            self.coefficients[:, 1] = 0.0
            self.coefficients[0, 1] = self.base[1]
            self.coefficients[1, 1] = radius * opacity_slope
            validation_error = 0.0
            worst = None
            for node in (-0.75, -0.25, 0.25, 0.75):
                actual = encode(exact.fields(logt + radius * node)[1])
                errors = abs(actual - ch.chebval(node, self.coefficients))
                errors[
                    1
                ] = 0.0  # Opacity remains the explicitly linearized quantity above.
                if np.max(errors) > validation_error:
                    validation_error = np.max(errors)
                    field, depth = np.unravel_index(np.argmax(errors), errors.shape)
                    worst = (
                        ("rho", "opacity", "Cp", "Q", "nabla_ad")[field],
                        depth,
                        np.exp(logt[depth] + radius * node),
                    )
            LOGGER.info(
                "Material proposal table: %d nodes, independent error %.3g at %s",
                count,
                validation_error,
                worst,
            )
            if validation_error < 1e-8:
                break
        else:
            raise RuntimeError(
                "Local material interpolation failed its accuracy check; no approximate EOS substituted"
            )
        self.derivatives = ch.chebder(self.coefficients) / radius

    def __call__(self, delta):
        from numpy.polynomial import chebyshev as ch

        if np.max(abs(delta)) > self.radius * (1 + 1e-12):
            raise ValueError("Material proposal left its interpolation interval")
        encoded = ch.chebval(delta / self.radius, self.coefficients, tensor=False)
        slopes = ch.chebval(delta / self.radius, self.derivatives, tensor=False)
        fields = np.vstack((np.exp(encoded[:4]), encoded[4]))
        slopes[:4] *= fields[:4]
        atmosphere = replace(
            self.atmosphere,
            temperature=np.exp(self.logt + delta),
            mass_density=fields[0],
        )
        return self.exact.assemble(atmosphere, fields, slopes)


def repair_trial(
    evaluation,
    proposed_delta,
    materials,
    *,
    maximum_iterations=12,
    desired_velocity=None,
    progress=None,
):
    """Map a cheap-model proposal to its actual finite-material ML2 gradient.

    The signed coordinate also keeps stable cells on their proposed stable
    branch as the EOS changes. It does not allow negative convective flux.
    Repair is exactly the identity at a zero step. Failure explicitly rejects
    the trial; no uncorrected or old atmosphere is substituted.
    """
    p = evaluation.payload
    if not np.any(proposed_delta) and desired_velocity is None:
        return proposed_delta.copy()
    logt0 = np.log(p["atmosphere"].temperature)
    operator = p["interface_gradient_operator"]
    target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
    transport = p["convection_transport"]
    ad, loss, coefficient = (
        transport[k]
        for k in ("adiabatic_gradient", "ml2_radiative_loss", "ml2_flux_coefficient")
    )
    ad_j, loss_j, coefficient_j = p["ml2_coefficient_log_temperature_responses"]
    gradient = p["temperature_gradient"] + operator @ proposed_delta
    predicted = (
        ad + ad_j @ proposed_delta,
        loss * np.exp((loss_j / loss[:, None]) @ proposed_delta),
        coefficient * np.exp((coefficient_j / coefficient[:, None]) @ proposed_delta),
    )
    y = (
        ml2_auxiliary_from_gradient(gradient, *predicted, target)
        if desired_velocity is None
        else np.asarray(desired_velocity)
    )
    active = np.ones(len(y), dtype=bool)
    active[0] = False
    logt = logt0 + proposed_delta
    identity = np.eye(len(logt))

    def actual_gradient(log_temperature):
        result = np.zeros_like(log_temperature)
        # Match the production subtraction order, rather than cancellation
        # of two separately scaled log temperatures in a dense matvec.
        result[1:] = np.diff(log_temperature) * np.diag(operator)[1:]
        return result

    worst = np.inf
    for iteration in range(maximum_iterations):
        values, responses = materials(logt, True)
        defect, tangent, _ = ml2_auxiliary_compatibility(
            actual_gradient(logt), y, values, responses, operator, target
        )
        # A flux-based closure tolerance is meaningful even when the allowed
        # superadiabatic temperature increment is extremely small.
        actual_y = ml2_auxiliary_from_gradient(actual_gradient(logt), *values, target)
        desired_flux = np.maximum(y, 0.0) ** 3
        worst = np.max(
            abs(np.maximum(actual_y[active], 0.0) ** 3 - desired_flux[active])
        )
        defect[~active] = (actual_gradient(logt) - gradient)[~active]
        tangent[~active] = operator[~active]
        defect[0] = logt[0] - logt0[0] - proposed_delta[0]
        tangent[0] = identity[0]
        update = np.linalg.solve(tangent, -defect)
        if progress is not None:
            progress(dict(iteration=iteration,flux_compatibility=float(worst),
                maximum_gradient_defect=float(np.max(abs(defect[active]))),
                maximum_log_temperature_update=float(np.max(abs(update)))))
        # Proposal accuracy, NOT an atmosphere convergence tolerance. These
        # limits are orders tighter than the outer gates, but need not chase
        # the noise of nested finite-difference thermodynamic derivatives.
        if worst <= 1e-6 and np.max(abs(update)) <= 1e-9:
            LOGGER.info(
                "Finite-material repair: %d updates, %d faces, flux compatibility %.3g, max dlnT %.3g",
                iteration,
                np.sum(active),
                worst,
                np.max(abs(logt - logt0)),
            )
            return logt - logt0
        if not np.all(np.isfinite(update)) or np.max(abs(update)) > 0.12:
            raise RecoverableEvaluationError(
                "Finite-material convection correction left its local branch"
            )
        logt += update
    raise RecoverableEvaluationError(
        f"Finite-material ML2 compatibility failed: flux defect {worst:g}; "
        f"gradient defect {np.max(abs(defect[active])):g}, correction {np.max(abs(update)):g}"
    )


def tlusty_flux_target(evaluation):
    """CONREF-like flux targets across the established convection zone.

    Only a proposal: bridge stable holes between non-isolated convective
    faces. No depth or Teff threshold, and no 0.999*F_star radiative-flux cap.
    If radiation already exceeds F_star, retain that face's present state.
    The actual energy equations decide whether the proposed repair is useful.
    """
    p = evaluation.payload
    target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
    c = p["convection_transport"]
    y0 = ml2_auxiliary_from_gradient(
        p["temperature_gradient"],
        c["adiabatic_gradient"],
        c["ml2_radiative_loss"],
        c["ml2_flux_coefficient"],
        target,
    )
    convective = y0 > 0
    convective[0] = False
    pair = np.flatnonzero(convective[:-1] & convective[1:])
    y = y0.copy()
    if len(pair):
        zone = np.arange(len(y))
        zone = (zone >= pair[0]) & (zone <= pair[-1] + 1)
        available = 1 - p["radiative_flux_interface"] / target
        selected = zone & (available > 0)
        y[selected] = np.cbrt(available[selected])
    return y0, y


def attach_repair(options, materials):
    """Attach after the research step builder, retaining all outer safeguards."""
    original_builder = options["step_builder"]
    current = {}

    def builder(state, evaluation, jacobian, radius):
        current["evaluation"] = evaluation
        return original_builder(state, evaluation, jacobian, radius)

    def projector(old, proposed):
        evaluation = current["evaluation"]
        mapping = evaluation.payload["log_temperature_from_state"]
        corrected = repair_trial(evaluation, mapping @ (proposed - old), materials)
        return old + np.linalg.solve(mapping, corrected)

    options["step_builder"] = builder
    options["trial_projector"] = projector
