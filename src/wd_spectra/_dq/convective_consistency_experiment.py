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








