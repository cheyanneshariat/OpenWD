"""Dense-He continuum equations, with explicitly supplied fluid inputs.

Iglesias, Rogers & Saumon (2002), ApJ 569, L111, Eqs. (3.2)--(3.6),
and Blouin, Dufour & Allard (2018), ApJ 863, 184, Eqs. (2)--(9).
DOIs: 10.1086/340689 and 10.3847/1538-4357/aad4a9.

The Born approximation is used ONLY for the dense/dilute opacity ratio.
It must multiply the existing John He-minus free-free coefficient, not
replace its absolute normalization. The dielectric and finite-k structure
factor are mandatory inputs: neither S(0) nor an ideal dielectric is silently
substituted. This module does not supply a qualified Blouin fluid model.

Atomic units are used internally (r/a0, k*a0, energy/Hartree). The public
integrator accepts vacuum wavelength in Angstrom and temperature in kelvin.
No atmosphere solver or production dispatcher enables this module by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np

from .constants import (
    BOHR_RADIUS, BOLTZMANN, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PLANCK,
)

HARTREE_ERG = ELEMENTARY_CHARGE_ESU**2 / BOHR_RADIUS
_Q = 1.685
_RP = 0.5
_ALPHA_P = 5.532  # Dimensionless potential parameter, NOT atomic polarizability.


def _positive(value, name, *, zero=False):
    array = np.asarray(value, dtype=float)
    if (np.any(~np.isfinite(array))
            or np.any(array < 0 if zero else array <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return array


def electron_helium_potential(radius_bohr):
    """Iglesias (2002) Eqs. (3.5)--(3.6), in Hartree.

    The polarization denominator is (rp**2 + r**2)**2. Its large-r
    limit is -alpha_p * rp**3/r**4; its value tends to zero at r=0.
    The nuclear term retains its -2/r singularity.
    """
    r = _positive(radius_bohr, "radius_bohr")
    x = r / _RP
    coulomb = -2 / r * (1 + _Q * r) * np.exp(-2 * _Q * r)
    # expm1/log1p avoid subtracting two near-unit numbers at small r.
    polarization = -_ALPHA_P / _RP * np.where(
        x < 0.1,
        np.expm1(-2 * np.log1p(x*x)) - np.expm1(-3*x + np.log1p(2*x)),
        1/(1+x*x)**2 - np.exp(-3*x)*(1+2*x),
    )
    return coulomb + polarization


def electron_helium_fourier(wavenumber_bohr_inverse):
    """Exact 3D radial Fourier transform, in Hartree * a0**3.

    Convention: Vtilde(k)=4*pi*integral r**2 V(r) sinc(k*r) dr.
    Analytic transforms remove radial truncation error and oscillatory
    numerical quadrature from every atmosphere opacity evaluation.
    """
    k = _positive(wavenumber_bohr_inverse, "wavenumber_bohr_inverse", zero=True)
    k2 = k*k
    a = 2*_Q
    d = k2+a*a
    coulomb = -8*np.pi/d - 16*np.pi*_Q*a/d**2
    b = 3/_RP
    p = k2+b*b
    polarization = -_ALPHA_P/_RP * (
        np.pi**2*_RP**3*np.exp(-_RP*k)
        - 8*np.pi*b/p**2
        - 16*np.pi/_RP*(3*b*b-k2)/p**3
    )
    return coulomb + polarization


def dielectric_from_refractivity(number_density_cm3, polarizability_bohr3,
                                 pair_integral_bohr6):
    """Blouin (2018) Eqs. (6)--(9), for the nonabsorbing dielectric.

    ``pair_integral_bohr6`` is integral Delta-alpha(omega,r) *
    exp[-phi(r)/(kBT)] * r**2 dr in atomic units. It includes BOTH
    the static interaction polarizability and omega**2 Cauchy moment.
    It is not the bulk pair correlation g(r). Inputs broadcast normally.

    The Lorentz--Lorenz density variable is x=(epsilon-1)/(epsilon+2),
    where epsilon=n_ref**2. The opacity correction divides by epsilon**2,
    i.e. n_ref**4, not by n_ref**2. Invalid virial states raise; no clipping.
    """
    n = _positive(number_density_cm3, "number_density_cm3", zero=True)*BOHR_RADIUS**3
    alpha = _positive(polarizability_bohr3, "polarizability_bohr3")
    pair = np.asarray(pair_integral_bohr6, dtype=float)
    if np.any(~np.isfinite(pair)):
        raise ValueError("pair_integral_bohr6 must be finite")
    x = 4*np.pi/3*n*alpha + 8*np.pi**2/3*n*n*pair
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x >= 1):
        raise ValueError("refractivity virial expansion outside nonabsorbing He domain")
    return (1+2*x)/(1-x)


@lru_cache(maxsize=8)
def _legendre(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@dataclass(frozen=True)
class FreeFreeCorrection:
    factor: np.ndarray
    mean_structure_factor: np.ndarray
    quadrature_relative_change: float
    quadrature_order: int


def _weighted_structure(energy, thermal, structure_factor, order, exponent_extent):
    # I0 dk = exp[-(k/2-E/k)**2/(2*kBT)] |k**2 Vtilde/(4*pi)|**2 d(log k).
    # Bound the Maxwell exponent, not k with an arbitrary wavelength cutoff.
    limit = np.sqrt(2*thermal*exponent_extent)
    root = np.sqrt(limit*limit + 2*energy)
    lower = 2*energy/(root+limit)  # stable form of root-limit
    upper = root+limit
    midpoint = (np.log(lower)+np.log(upper))/2
    halfwidth = (np.log(upper)-np.log(lower))/2
    nodes, weights = _legendre(order)
    k = np.exp(midpoint[:, None] + halfwidth[:, None]*nodes)
    exponent = -(k/2-energy[:, None]/k)**2/(2*thermal)
    amplitude = k*k*electron_helium_fourier(k)/(4*np.pi)
    weight = np.exp(exponent)*amplitude**2*weights
    s = np.asarray(structure_factor(k), dtype=float)
    if s.shape != k.shape or np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("structure_factor must return finite positive S(k) with input shape")
    denominator = weight.sum(axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise FloatingPointError("unresolved dilute Born integral")
    return (weight*s).sum(axis=1)/denominator


def helium_minus_correction(wavelength_angstrom, temperature, *, structure_factor,
                            dielectric, rtol=2e-6, initial_order=64, max_order=1024,
                            exponent_extent=60.0):
    """Compute integral I0*S dk / integral I0 dk / |epsilon|**2.

    One thermodynamic layer per call. ``structure_factor(k)`` receives a
    2D array of k in inverse Bohr radii and must return S(k) on that array.
    ``dielectric`` is a positive real scalar or a value per wavelength.
    Each wavelength is checked by doubled-order log-k Gauss quadrature;
    failure is reported rather than replaced by a different opacity.

    ``exponent_extent`` controls neglected Maxwell tails. The default
    truncates at exp(-60); quadrature convergence and tail truncation are
    distinct, and the latter is checked by extending the bounds in tests.
    """
    wave = _positive(wavelength_angstrom, "wavelength_angstrom")
    if wave.ndim != 1 or not wave.size:
        raise ValueError("wavelength_angstrom must be a nonempty 1D array")
    t = _positive(temperature, "temperature")
    if t.ndim != 0:
        raise ValueError("temperature must describe one layer")
    eps = _positive(dielectric, "dielectric")
    if eps.shape not in ((), wave.shape):
        raise ValueError("dielectric must be scalar or match wavelength_angstrom")
    if (not np.isfinite(rtol) or not 0 < rtol < 1
            or not isinstance(initial_order, int) or initial_order < 16
            or not isinstance(max_order, int) or max_order < 2*initial_order
            or not np.isfinite(exponent_extent) or exponent_extent < 30):
        raise ValueError("invalid Born quadrature controls")
    thermal = float(t)*BOLTZMANN/HARTREE_ERG
    energy = PLANCK*LIGHT_SPEED/(wave*1e-8*HARTREE_ERG)
    order = initial_order
    previous = _weighted_structure(energy, thermal, structure_factor, order, exponent_extent)
    while 2*order <= max_order:
        order *= 2
        current = _weighted_structure(energy, thermal, structure_factor, order, exponent_extent)
        change = float(np.max(np.abs(current/previous-1)))
        if change <= rtol:
            return FreeFreeCorrection(current/eps**2, current, change, order)
        previous = current
    raise RuntimeError(f"He-minus Born quadrature failed: order={order}, relative change={change:g}")


class DenseHeliumCorrectionTable:
    """Explicit, bounded constitutive table used by the DQ material adapter.

    Cubic interpolation of log(delta) is smooth and preserves positivity.
    Density is neutral-He mass density, NOT total mixture density. No
    extrapolation, high-frequency replacement, or temperature switch occurs.
    The n=0 surface must give delta=1; the density coordinate regularizes
    interpolation at that surface and is not a physical density threshold.
    """

    def __init__(self, path):
        from scipy.ndimage import spline_filter
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as source:
            self.axes = tuple(np.array(source[k], dtype=float) for k in
                              ("log_temperature", "log1p_density", "log_wavelength"))
            factor = np.array(source["factor"], dtype=float)
            self.density_scale = float(source["density_scale_g_cm3"])
            self.metadata = json.loads(str(source["metadata_json"].item()))
        if not isinstance(self.metadata, dict) or self.metadata.get("schema") != "dense-He-ff-v1":
            raise ValueError("invalid dense-He table schema")
        for axis in self.axes:
            if (axis.ndim != 1 or len(axis) < 4 or np.any(~np.isfinite(axis))
                    or np.any(np.diff(axis) <= 0)
                    or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-10, atol=1e-14)):
                raise ValueError("dense-He table axes must be finite, increasing, uniform")
        if (not np.isfinite(self.density_scale) or self.density_scale <= 0
                or self.axes[1][0] != 0 or factor.shape != tuple(map(len, self.axes))
                or np.any(~np.isfinite(factor)) or np.any(factor <= 0)
                or not np.array_equal(factor[:, 0], np.ones_like(factor[:, 0]))):
            raise ValueError("invalid dense-He factors, shape, or zero-density limit")
        # Quadratic ghost knots avoid imposing a zero physical derivative at
        # a table boundary. They are used only to construct spline coefficients;
        # requests outside the original physical domain still raise below.
        values = np.log(factor)
        self._padding = 4
        for axis in range(3):
            v = np.moveaxis(values, axis, 0)
            left = [v[0]-j*(v[1]-v[0])+j*(j+1)/2*(v[2]-2*v[1]+v[0])
                    for j in range(self._padding, 0, -1)]
            right = [v[-1]+j*(v[-1]-v[-2])+j*(j+1)/2*(v[-3]-2*v[-2]+v[-1])
                     for j in range(1, self._padding+1)]
            values = np.moveaxis(np.concatenate((left, v, right)), 0, axis)
        self._spline = spline_filter(values, order=3, mode="mirror")

    @property
    def wavelength_bounds(self):
        return np.exp(self.axes[2][[0, -1]])

    def correction(self, wavelength, temperature, neutral_number_density):
        from scipy.ndimage import map_coordinates
        from .constants import HELIUM_MASS
        wave = _positive(wavelength, "wavelength")
        t = _positive(temperature, "temperature")
        n = _positive(neutral_number_density, "neutral_number_density", zero=True)
        if wave.ndim != 1 or t.ndim != 1 or n.shape != t.shape:
            raise ValueError("dense-He table expects wavelength and matching layer arrays")
        physical = (np.log(t), np.log1p(n*HELIUM_MASS/self.density_scale), np.log(wave))
        fractional = []
        for name, values, axis in zip(
                ("log_temperature", "log1p_neutral_density", "log_wavelength"),
                physical, self.axes):
            # Tolerance only absorbs exp(log(bound)) roundoff, not extrapolation.
            tol = 32*np.finfo(float).eps*max(1, abs(axis[0]), abs(axis[-1]))
            if np.any(values < axis[0]-tol) or np.any(values > axis[-1]+tol):
                raise ValueError(
                    "dense-He continuum requested outside supplied table domain: "
                    f"{name} requested [{values.min():.12g}, {values.max():.12g}], "
                    f"allowed [{axis[0]:.12g}, {axis[-1]:.12g}]")
            fractional.append(self._padding+(np.clip(values, axis[0], axis[-1])-axis[0])/(axis[1]-axis[0]))
        x, y, z = np.broadcast_arrays(fractional[0][None, :], fractional[1][None, :], fractional[2][:, None])
        result = np.exp(map_coordinates(self._spline, np.array([x, y, z]),
                                       order=3, mode="mirror", prefilter=False))
        result[:, n == 0] = 1.0
        return result
