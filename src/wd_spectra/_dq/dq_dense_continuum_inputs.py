"""Published input data for the DQ dense-continuum experiment.

No stellar fit or observed flux enters these functions. The numerical
interpolation/extrapolation choices are OURS, not undocumented claims about
Blouin's private implementation. Sensitivity to them must be reported.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from wd_spectra.constants import BOHR_RADIUS, BOLTZMANN
from wd_spectra.dense_helium_continuum import HARTREE_ERG, dielectric_from_refractivity

SOURCES = {
    "electron_helium": "https://doi.org/10.1086/340689",
    "neutral_potential": "https://doi.org/10.1016/0375-9601(86)90752-8",
    "atomic_polarizability": "https://doi.org/10.1103/PhysRevA.68.012508",
    "pair_polarizability": "https://doi.org/10.1063/1.480361",
    "short_pair_polarizability": "https://doi.org/10.1021/jp9941615",
    "atmosphere_prescription": "https://doi.org/10.3847/1538-4357/aad4a9",
}

# Hattig et al. (1999) Table II: FCI/d-aug-cc-pVTZ-3321. Columns
# R [a0], Delta-alpha-ave [10^-3 a.u.], Delta-S-ave(-4) [10^-3 a.u.].
# This is the ISOTROPIC interaction polarizability, not its anisotropy.
# Unlike modern Cencek et al. data, this historical dataset has a small
# positive large-R branch. It is retained for this historical reproduction.
HATTIG = np.array([
    [3.00, -132.476, -339.090], [3.50, -71.650, -212.383],
    [4.00, -33.684, -114.433], [4.15, -26.229, -92.936],
    [4.30, -20.205, -74.762], [4.45, -15.391, -59.583],
    [4.60, -11.585, -47.045], [4.80, -7.770, -33.824],
    [5.00, -5.068, -23.881], [5.10, -4.040, -19.918],
    [5.20, -3.188, -16.523], [5.40, -1.905, -11.160],
    [5.60, -1.048, -7.313], [5.80, -.491, -4.597],
    [6.00, -.139, -2.713], [6.50, .226, -.282],
    [7.00, .267, .480], [7.50, .217, .604],
    [8.00, .156, .526], [8.50, .109, .410], [9.00, .075, .308],
    [10.00, .038, .173], [11.00, .022, .103], [12.00, .015, .066],
    [13.00, .010, .043],
])
# Maroulis (2000), Table 6, CCSD(T)/[6s4p3d1f], in a.u. (NOT 10^-3).
# Only points below Hattig's lower limit are appended; no abundance/shape fit.
MAROULIS_SHORT = np.array([[2.0, -.1307], [2.5, -.1989]])

# Masili & Starace (2003), Table VI, "This work" dynamic values, and
# Table V optimized static result including logarithmic terms. Only the
# nonresonant range omega<=0.5 is exposed here, not interpolated across poles.
MASILI = np.array([
    [0, 1.38317394], [.05, 1.387094], [.10, 1.398857], [.15, 1.418998],
    [.20, 1.448388], [.25, 1.488389], [.30, 1.541045], [.35, 1.609399],
    [.40, 1.698070], [.45, 1.814308], [.50, 1.970129],
])
_ATOM = PchipInterpolator(MASILI[:, 0]**2, MASILI[:, 1], extrapolate=False)
_R = np.r_[MAROULIS_SHORT[:, 0], HATTIG[:, 0]]
_STATIC = np.r_[MAROULIS_SHORT[:, 1], HATTIG[:, 1]*1e-3]
_ALPHA = PchipInterpolator(_R, _STATIC, extrapolate=False)
_DISPERSION = PchipInterpolator(HATTIG[:, 0], HATTIG[:, 2]*1e-3, extrapolate=False)
_CORE = brentq(lambda x: 13.1*(1-x)+7*np.log(x), .01, .8)*2.9673/(BOHR_RADIUS*1e8)


def neutral_potential_hartree(radius_bohr):
    """Ross & Young (1986), Eq. 1, with the exp-6 inner stationary core."""
    r = np.asarray(radius_bohr, float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("positive finite radius required")
    x = np.maximum(r, _CORE)*(BOHR_RADIUS*1e8)/2.9673
    v = 10.8*BOLTZMANN/HARTREE_ERG/(13.1-6)*(6*np.exp(13.1*(1-x))-13.1/x**6)
    return np.where(r < _CORE, np.inf, v)


def atomic_polarizability(photon_energy_hartree):
    w = np.asarray(photon_energy_hartree, float)
    if np.any(~np.isfinite(w)) or np.any(w < 0) or np.any(w > MASILI[-1, 0]):
        raise ValueError("Masili nonresonant interpolation requires 0<=omega<=0.5 Hartree")
    return _ATOM(w*w)


def _continued(r, x, y, interpolator, short_range):
    # Published grids end at finite R. The large-R dipole-induced-dipole
    # leading term is R^-6; its coefficient is anchored at the last point.
    # The unprovided short-range response is exposed as a sensitivity choice,
    # not silently advertised as calculated molecular data.
    inside = interpolator(np.clip(r, x[0], x[-1]))
    if short_range == "constant":
        short = y[0]
    elif short_range == "linear":
        short = y[0]+(r-x[0])*(y[1]-y[0])/(x[1]-x[0])
    elif short_range == "omit":
        short = 0.
    else:
        raise ValueError("short_range must be constant, linear, or omit")
    return np.where(r < x[0], short, np.where(r > x[-1], y[-1]*(x[-1]/r)**6, inside))


def pair_response(radius_bohr, *, short_range):
    r = np.asarray(radius_bohr, float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("positive finite radius required")
    return (_continued(r, _R, _STATIC, _ALPHA, short_range),
            _continued(r, HATTIG[:, 0], HATTIG[:, 2]*1e-3, _DISPERSION, short_range))


def pair_virial_integrals(temperature, *, short_range):
    """Return static and omega**2 coefficients of Blouin's radial integral.

    Integrate each polynomial interval separately; include the long-range
    tail to infinity. No density-dependent g(r) belongs in this SECOND virial.
    This implementation is classical as in Blouin (2018), not quantum
    path-integral refractivity at cryogenic temperature.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    thermal = BOLTZMANN*temperature/HARTREE_ERG
    def integrand(r, index):
        return pair_response(r, short_range=short_range)[index]*np.exp(
            -neutral_potential_hartree(r)/thermal)*r*r
    breaks = np.r_[_CORE, _R, np.inf]
    return np.array([sum(quad(integrand, left, right, args=(i,),
        epsabs=1e-10, epsrel=1e-8)[0] for left, right in zip(breaks[:-1], breaks[1:]))
        for i in range(2)])


def dielectric(temperature, number_density, photon_energy, *, short_range):
    coefficients = pair_virial_integrals(temperature, short_range=short_range)
    pair = coefficients[0]+np.asarray(photon_energy)**2*coefficients[1]
    return dielectric_from_refractivity(number_density, atomic_polarizability(photon_energy), pair)
