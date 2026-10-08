"""DQ-only bounded work arrays for the unchanged helium continuum.

Every wavelength has independent continuum arithmetic. Splitting its first
axis changes allocation size, not populations, source data, or quadrature.
The shared helium implementation and other atmosphere families are unchanged.
"""
import numpy as np
from . import base

CONTINUUM_BATCH_SIZE = 1024


def helium_continuum_absorption(atmosphere, wavelength, correction=None):
    wave = np.asarray(wavelength, dtype=float)
    if wave.ndim != 1 or np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError('Finite positive 1D wavelengths required')
    if correction is not None:
        correction = np.asarray(correction, dtype=float)
        if correction.shape != (len(wave), atmosphere.n_depth):
            raise ValueError('Expected wavelength-by-depth dense-He correction')
    options = dict(include_electron_scattering=False, include_rayleigh_scattering=False)
    if len(wave) <= CONTINUUM_BATCH_SIZE:
        return base.helium_continuum_mass_absorption_coefficient(
            atmosphere, wave, helium_minus_correction=correction, **options)
    result = np.empty((len(wave), atmosphere.n_depth))
    for first in range(0, len(wave), CONTINUUM_BATCH_SIZE):
        selection = slice(first, first+CONTINUUM_BATCH_SIZE)
        dense = None if correction is None else correction[selection]
        result[selection] = base.helium_continuum_mass_absorption_coefficient(
            atmosphere, wave[selection], helium_minus_correction=dense, **options)
    return result
