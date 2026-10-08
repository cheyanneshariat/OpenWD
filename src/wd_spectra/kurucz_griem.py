"""Frequency-normalized Kurucz (1970) / Griem Stark kernels for DAH.

Transcribed from SAO Special Report 309, section 5.14 and STARK on printed
page 243 (PDF page 239). This historical opacity approximation has no
separate Doppler or unbroadened central component. It is not the unpublished
Moss/Jordan implementation. No empirical widths or strengths are fitted.
See docs/models/DAH.md for the normalization and scope of the approximation.
"""

import numpy as np
from scipy.integrate import quad

from .constants import (
    BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI, PLANCK,
)
from .opacity import BALMER_LINES, _atmosphere_level_distribution


def stark_coefficient(upper_level):
    """K_2m in Angstrom / electrostatic field unit, original KNMTAB."""
    if not 3 <= upper_level <= 22:
        raise ValueError("Kurucz/Griem Balmer prescription supports n=3..22")
    low = {3: .0125, 4: .0177, 5: .026, 6: .0348, 7: .0493}
    return low.get(upper_level, 5.5e-5 * (4 * upper_level**2)**2 / (upper_level**2 - 4))


def _exint(x):
    """Original STARK approximation to E1 for 0 < x <= 8."""
    return -np.log(x) - .57516 + x * (.97996 - x * (.21654 - x * (
        .033572 - x * (.0029222 - x * 1.05439e-4))))


def raw_profile(detuning_hz, temperature, electron_density, upper_level, rest_frequency):
    """Original normalized-variable shape times d(beta)/d(nu), units Hz^-1.

    The historical decimal constants and polynomial are retained for an
    auditable comparison. The caller supplies the rest frequency to avoid
    mixing the 1970 Rydberg wavelength convention with H2db line centers.
    Inputs broadcast. This raw approximation does not have exactly unit area.
    """
    delta, temp, ne = np.broadcast_arrays(np.abs(detuning_hz), temperature, electron_density)
    if np.any(~np.isfinite(delta)) or np.any(~np.isfinite(temp)) or np.any(~np.isfinite(ne)):
        raise ValueError("finite profile inputs required")
    if np.any(temp <= 0) or np.any(ne <= 0) or not np.isfinite(rest_frequency) or not rest_frequency > 0:
        raise ValueError("positive temperature, density and rest frequency required")
    mm = upper_level**2
    f0 = 1.25e-9 * ne**(2/3)
    dbeta = 2.997925e18 / rest_frequency**2 / f0 / stark_coefficient(upper_level)
    beta = dbeta * delta
    hkt = PLANCK / (BOLTZMANN * temp)
    y1 = mm * delta * hkt / 2
    y2 = (3.14159**2 / 2 / .0265384 / 2.997925e10) * delta**2 / ne
    q = 1.5 + .5 * (y1**2 - 1.384) / (y1**2 + 1.384)
    impact = np.zeros_like(delta)
    active = (y1 <= 8) & (y1 < y2) & (y1 > 0)
    exy2 = np.zeros_like(delta)
    at_y2 = active & (y2 <= 8)
    exy2[at_y2] = _exint(y2[at_y2])
    impact[active] = 1.438 * np.sqrt(y1[active] * (1 - 4/mm)) * (
        .4 * np.exp(-y1[active]) + _exint(y1[active]) - .5 * exy2[active])
    wing = beta > 20
    p = np.empty_like(delta)
    p[~wing] = 8 / (80 + beta[~wing]**3)
    p[wing] = 1.5 / beta[wing]**2.5
    ratio = np.array(q + impact, copy=True)
    a = 2 * mm * 3.28805e15 / delta[wing]
    correction = 6.28 * 1.48e-25 * a * ne[wing] * (
        np.sqrt(a) * (1.3 * q[wing] + .30 * impact[wing]) - 3.9 * 3.28805e15 * hkt[wing])
    ratio[wing] = q[wing] * np.minimum(1 + correction, 1.25) + impact[wing]
    profile = p * dbeta * ratio
    if np.any(profile < 0) or np.any(~np.isfinite(profile)):
        raise ValueError("Kurucz approximation left its positive finite profile domain")
    return profile


def profile_area(temperature, electron_density, upper_level, rest_frequency):
    """Integrate the symmetric frequency-detuning profile, independent of mesh.

    Split at beta=20 (the original approximation has a small discontinuity)
    and at the impact cutoffs. The wing substitution beta=1/t^2 makes the
    infinite tail a finite, smooth interval. No observed spectrum enters.
    """
    temp, ne = np.broadcast_arrays(temperature, electron_density)
    area = np.empty(temp.shape)
    for index in np.ndindex(temp.shape):
        t, e = float(temp[index]), float(ne[index])
        scale = rest_frequency**2 * 1.25e-9 * e**(2/3) * stark_coefficient(upper_level) / 2.997925e18
        a = upper_level**2 * PLANCK / (2 * BOLTZMANN * t)
        b = (3.14159**2 / 2 / .0265384 / 2.997925e10) / e
        cutoffs = np.array([a/b, 8/a, np.sqrt(8/b)]) / scale
        def integrand(beta):
            return float(raw_profile(beta * scale, t, e, upper_level, rest_frequency)) * scale
        core = quad(integrand, 0., 20., points=sorted(cutoffs[(cutoffs > 0) & (cutoffs < 20)]),
                    epsabs=2e-7, epsrel=2e-7, limit=150)[0]
        wing_points = sorted(1 / np.sqrt(cutoffs[cutoffs > 20]))
        wing = quad(lambda x: 2 * integrand(1/x**2) / x**3, 0., 1/np.sqrt(20.),
                    points=wing_points, epsabs=2e-7, epsrel=2e-7, limit=150)[0]
        area[index] = 2 * (core + wing)
    return area


def kurucz_griem_templates(atmosphere, *, broadening=None, maximum_upper_level=12,
                     include_dispersion=False):
    """Frequency-area-normalized kernels with the same LTE line populations."""
    from .magnetic import BalmerLineTemplate, frequency_hilbert_dispersion

    distribution = _atmosphere_level_distribution(atmosphere, 40)
    n2 = distribution.population_density[:, 1]
    w2 = distribution.occupation_probability[:, 1]
    templates = {}
    for line in BALMER_LINES:
        if line.upper_level > maximum_upper_level:
            continue
        center = line.wavelength_vacuum_angstrom
        nu0 = LIGHT_SPEED / (center * 1e-8)
        scale = nu0**2 * 1.25e-9 * atmosphere.electron_density**(2/3) * stark_coefficient(line.upper_level) / 2.997925e18
        # Resolve narrow outer-layer kernels as well as broad dense wings.
        delta = np.unique((scale[:, None] * np.geomspace(.01, 100., 120)).ravel())
        frequency = np.concatenate((nu0 - delta, [nu0], nu0 + delta))
        frequency = frequency[(frequency >= LIGHT_SPEED / .00025) & (frequency <= LIGHT_SPEED / 9e-6)]
        mesh = np.unique(np.concatenate((
            LIGHT_SPEED / frequency / 1e-8,
            center + np.arange(-20., 20.0001, .02),
            center - np.geomspace(20., center - 900., 300),
            center + np.geomspace(20., 25000. - center, 400),
        )))
        area = profile_area(atmosphere.temperature, atmosphere.electron_density, line.upper_level, nu0)
        shape = raw_profile(LIGHT_SPEED / (mesh[:, None]*1e-8) - nu0,
                            atmosphere.temperature, atmosphere.electron_density, line.upper_level, nu0)
        survival = np.clip(distribution.occupation_probability[:, line.upper_level-1] / w2, 0, 1)
        strength = (PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
                    * line.absorption_oscillator_strength * n2 * survival / atmosphere.mass_density
                    * -np.expm1(-PLANCK * nu0 / (BOLTZMANN * atmosphere.temperature)))
        opacity = shape * (strength / area)[None, :]
        templates[line.upper_level] = BalmerLineTemplate(
            mesh, opacity, frequency_hilbert_dispersion(mesh, opacity) if include_dispersion else None)
    return templates
