"""Cached quadrature weights for linear photoionization/Milne integrals.

Only wavelength/cross-section and temperature factors are cached. Radiation
fields and populations are always current. Kernels retain the original
positive-cross-section integration intervals, including their threshold edges.
"""
from functools import lru_cache
import numpy as np
from .constants import LIGHT_SPEED, PLANCK, BOLTZMANN, PI


@lru_cache(maxsize=8)
def _kernel(wavelength_bytes, count, first_level, cross_section):
    wave = np.frombuffer(wavelength_bytes, dtype=np.float64)
    frequency = LIGHT_SPEED*1e8/wave
    order = np.argsort(frequency)
    nu = frequency[order]
    weights = np.zeros((len(wave), count))
    for index in range(count):
        sigma = cross_section(index+first_level, nu)
        selected = np.flatnonzero(sigma > 0)
        if len(selected) < 2:
            continue
        spacing = np.diff(nu[selected])
        quadrature = .5*(np.r_[0., spacing]+np.r_[spacing, 0.])
        original = order[selected]
        weights[original, index] = (4*PI*sigma[selected]/(PLANCK*nu[selected])
            *quadrature*wave[original]**2/(LIGHT_SPEED*1e8))
    spontaneous_lambda = 2*PLANCK*frequency**3/LIGHT_SPEED**2 * (LIGHT_SPEED*1e8)/wave**2
    weights.flags.writeable = False
    spontaneous_lambda.flags.writeable = False
    return weights, spontaneous_lambda


@lru_cache(maxsize=4)
def _boltzmann(wavelength_bytes, temperature_bytes):
    wave = np.frombuffer(wavelength_bytes, dtype=np.float64)
    temperature = np.frombuffer(temperature_bytes, dtype=np.float64)
    value = np.exp(-np.minimum(PLANCK*LIGHT_SPEED*1e8/wave[:,None]/(BOLTZMANN*temperature),745.))
    value.flags.writeable = False
    return value


def continuum_integrals(wavelength, temperature, mean_lambda, count, cross_section, *, first_level=1):
    """Return upward and unscaled downward radiative rates at current J."""
    wave_key = np.asarray(wavelength,dtype=np.float64).tobytes()
    weights, spontaneous = _kernel(wave_key,count,first_level,cross_section)
    boltzmann = _boltzmann(wave_key,np.asarray(temperature,dtype=np.float64).tobytes())
    upward = mean_lambda.T @ weights
    downward = ((spontaneous[:,None]+mean_lambda)*boltzmann).T @ weights
    return upward, downward
