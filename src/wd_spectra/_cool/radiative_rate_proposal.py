"""Consistent positive heating/cooling proposal for stiff opacity changes.

This does not replace the formal-transfer residual. Factoring two positive
rates avoids an artificial root when a linearized absorption prefactor
crosses zero, while preserving the exact base value and first derivative.
"""
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN


def radiative_rates(payload,delta):
    target=STEFAN_BOLTZMANN*payload['atmosphere'].effective_temperature**4
    e=payload['radiative_cell_energy_defect']/target
    ej=payload['radiative_cell_energy_log_temperature_jacobian']/target
    cooling=payload['thermal_cell_emission']/target
    cj=payload['thermal_cell_emission_log_temperature_jacobian']/target
    log_cj=cj/cooling[:,None]
    thermal=cooling*np.exp(log_cj@delta)
    thermal_j=thermal[:,None]*log_cj
    if not payload.get('diagnostic_positive_radiative_rates',False):
        return e+ej@delta,ej,thermal,thermal_j
    heating=cooling+e
    if np.any(heating <= 0) or np.any(~np.isfinite(heating)):
        raise ValueError('positive radiative proposal requires positive finite heating')
    log_hj=(cj+ej)/heating[:,None]
    log_ratio=np.log1p(e/cooling)+(log_hj-log_cj)@delta
    ratio_minus_one=np.expm1(log_ratio)
    exchange=thermal*ratio_minus_one
    exchange_j=(thermal_j*ratio_minus_one[:,None]
                +(thermal*np.exp(log_ratio))[:,None]*(log_hj-log_cj))
    return exchange,exchange_j,thermal,thermal_j
