"""Equation-level checks, not a claim of qualified dense-DQ atmospheres."""

import numpy as np
import pytest
from scipy.integrate import quad

from wd_spectra.constants import BOHR_RADIUS, BOLTZMANN, LIGHT_SPEED, PLANCK
from wd_spectra.dense_helium_continuum import (
    HARTREE_ERG,
    dielectric_from_refractivity,
    electron_helium_fourier,
    electron_helium_potential,
    helium_minus_correction,
)


def test_published_electron_potential_and_limits():
    r = np.array([0.01, 0.5, 1., 5., 10.])
    # Independent literal transcription of Iglesias Eqs. (3.5)--(3.6).
    expected = -2/r*(1+1.685*r)*np.exp(-3.37*r) - 5.532/0.5*(
        0.5**4/(0.5**2+r*r)**2 - np.exp(-3*r/0.5)*(1+2*r/0.5)
    )
    np.testing.assert_allclose(electron_helium_potential(r), expected, rtol=1e-13)
    assert electron_helium_potential(1e-10)*1e-10 == pytest.approx(-2, rel=1e-8)
    assert electron_helium_potential(1e4)*1e16 == pytest.approx(-5.532*0.5**3, rel=1e-8)


@pytest.mark.parametrize("k", [0.0, 0.05, 0.2, 1., 3., 10., 20.])
def test_analytic_fourier_against_independent_oscillatory_integral(k):
    if k == 0:
        numerical = 4*np.pi*quad(
            lambda r: r*r*electron_helium_potential(r), 0, np.inf,
            epsabs=1e-10, epsrel=1e-11,
        )[0]
    else:
        numerical = 4*np.pi/k*quad(
            lambda r: -2.0 if r == 0 else r*electron_helium_potential(r),
            0, np.inf, weight="sin", wvar=k, epsabs=1e-10,
        )[0]
    assert electron_helium_fourier(k) == pytest.approx(numerical, rel=2e-8, abs=1e-10)


def test_born_dilute_limit_and_dielectric_power():
    wave = np.geomspace(100, 1e6, 80)
    vacuum = helium_minus_correction(wave, 5000, structure_factor=np.ones_like, dielectric=1.)
    np.testing.assert_array_equal(vacuum.factor, np.ones_like(wave))
    correlated = helium_minus_correction(
        wave, 5000, structure_factor=lambda k: np.full_like(k, 0.3), dielectric=1.7,
    )
    np.testing.assert_allclose(correlated.factor, 0.3/1.7**2, rtol=2e-15)


def _reference_factor(wave, t, structure):
    # Independent quadrature in electron momentum difference x, not log k.
    energy = PLANCK*LIGHT_SPEED/(wave*1e-8*HARTREE_ERG)
    thermal = BOLTZMANN*t/HARTREE_ERG
    scale = np.sqrt(2*thermal)
    def integrand(y, correlated):
        x = y*scale
        root = np.sqrt(x*x+2*energy)
        k = x+root if x >= 0 else 2*energy/(root-x)
        amplitude = k*k*electron_helium_fourier(k)/(4*np.pi)
        return np.exp(-y*y)*amplitude**2/root*(structure(k) if correlated else 1)
    return (quad(integrand, -12, 12, args=(True,), epsabs=1e-12, epsrel=1e-10)[0]
            / quad(integrand, -12, 12, args=(False,), epsabs=1e-12, epsrel=1e-10)[0])


@pytest.mark.parametrize("t", [1000., 5000., 15000., 100000.])
def test_born_finite_k_weight_against_independent_quadrature(t):
    # Deliberately nonconstant analytic S, also above unity at large k;
    # verifies it is neither replaced by S(0) nor clipped to one.
    structure = lambda k: 0.1+1.4*(-np.expm1(-k*k))
    wave = np.array([100., 500., 2000., 5000., 10000., 1e6])
    value = helium_minus_correction(wave, t, structure_factor=structure, dielectric=1.)
    reference = [_reference_factor(w, t, structure) for w in wave]
    np.testing.assert_allclose(value.factor, reference, rtol=2e-6)
    extended = helium_minus_correction(
        wave, t, structure_factor=structure, dielectric=1., exponent_extent=90,
    )
    np.testing.assert_allclose(value.factor, extended.factor, rtol=2e-6)
    assert value.factor[0] > 1
    assert np.all(value.factor > 0.1)


def test_refractivity_units_and_density_orders():
    # Construct dimensionless n*a0^3, with a negative pair correction.
    density = np.array([0., 1e-4, 1e-3, .01])/BOHR_RADIUS**3
    n = density*BOHR_RADIUS**3
    x = 4*np.pi/3*n*1.383 + 8*np.pi**2/3*n*n*(-.7)
    eps = dielectric_from_refractivity(density, 1.383, -.7)
    np.testing.assert_allclose((eps-1)/(eps+2), x, rtol=1e-13)
    assert eps[0] == 1
    # In particular, density in cm^-3 is not accidentally treated as mol/cm^3.
    assert dielectric_from_refractivity(1e20, 1.383, 0) < 1.001


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_invalid_born_inputs_are_not_replaced_by_dilute_opacity(bad):
    with pytest.raises(ValueError):
        helium_minus_correction([5000], 5000, structure_factor=np.ones_like, dielectric=bad)
    with pytest.raises(ValueError):
        helium_minus_correction([5000], bad, structure_factor=np.ones_like, dielectric=1)
    with pytest.raises(ValueError):
        helium_minus_correction([5000], 5000,
                               structure_factor=lambda k: np.full_like(k, bad), dielectric=1)


def test_invalid_shapes_and_unresolved_quadrature_raise():
    with pytest.raises(ValueError, match="input shape"):
        helium_minus_correction([5000], 5000, structure_factor=lambda k: 1., dielectric=1)
    with pytest.raises(ValueError, match="dielectric must be scalar"):
        helium_minus_correction([5000], 5000, structure_factor=np.ones_like, dielectric=[1, 2])
    with pytest.raises(RuntimeError, match="quadrature failed"):
        helium_minus_correction([5000], 5000, dielectric=1,
            structure_factor=lambda k: 1+.9*np.sin(100*k), initial_order=16, max_order=32)
    with pytest.raises(ValueError, match="virial expansion"):
        dielectric_from_refractivity(1e25, 1.383, 0)
    with pytest.raises(ValueError, match="virial expansion"):
        dielectric_from_refractivity(1e23, 1.383, -1e6)


def test_optional_correction_changes_only_john_term_and_preserves_default():
    from wd_spectra import gray_helium_atmosphere
    from wd_spectra.helium import (
        helium_continuum_mass_absorption_coefficient as continuum,
        helium_minus_free_free_coefficient as john,
    )
    a = gray_helium_atmosphere(10000, 8, n_depth=6)
    wave = np.array([1000., 4339., 5063., 6200., 10000.])
    baseline = continuum(a, wave)
    np.testing.assert_array_equal(baseline, continuum(a, wave, helium_minus_correction=None))
    np.testing.assert_array_equal(baseline, continuum(a, wave, helium_minus_correction=np.ones_like(baseline)))
    corrected = continuum(a, wave, helium_minus_correction=np.full_like(baseline, .6))
    term = (john(wave[:,None]*1e-4, a.temperature[None,:])
            * a.helium_lte_state.neutral_he_density[None,:]
            * a.electron_density[None,:]*BOLTZMANN*a.temperature[None,:]
            / a.mass_density[None,:])
    np.testing.assert_allclose(corrected, baseline-.4*term, rtol=2e-14)
    with pytest.raises(ValueError, match="wavelength-by-depth"):
        continuum(a, wave, helium_minus_correction=.6)


@pytest.fixture
def synthetic_table(tmp_path):
    import json
    from wd_spectra.dense_helium_continuum import DenseHeliumCorrectionTable
    axes = (np.linspace(np.log(2000), np.log(40000), 9),
            np.linspace(0, np.log1p(1.2/.001), 11),
            np.linspace(np.log(1000), np.log(100000), 9))
    t, d, w = axes
    # A multi-linear log(delta) has nonzero derivatives at every boundary
    # except where the exact density-zero identity requires otherwise.
    logdelta = -.02*d[None,:,None]*(1+.03*t[:,None,None]+.04*w[None,None,:])
    path = tmp_path/"synthetic.npz"
    np.savez(path, log_temperature=t, log1p_density=d, log_wavelength=w,
        density_scale_g_cm3=.001, factor=np.exp(logdelta),
        metadata_json=json.dumps(dict(schema="dense-He-ff-v1", qualified=False)))
    return DenseHeliumCorrectionTable(path), axes


def test_table_boundary_values_and_slopes_are_not_flattened(synthetic_table):
    from wd_spectra.constants import HELIUM_MASS
    table, axes = synthetic_table
    tx, dx, wx = axes
    t = np.exp([tx[0], tx[0]+1e-4, tx[-1]-1e-4, tx[-1]])
    d = np.array([dx[0], dx[0]+1e-4, dx[-1]-1e-4, dx[-1]])
    w = np.exp([wx[0], wx[0]+1e-4, wx[-1]-1e-4, wx[-1]])
    actual = table.correction(w, t, .001*np.expm1(d)/HELIUM_MASS)
    expected = np.exp(-.02*d[None,:]*(1+.03*np.log(t)[None,:]+.04*np.log(w)[:,None]))
    np.testing.assert_allclose(actual, expected, rtol=3e-8)
    np.testing.assert_array_equal(actual[:, 0], 1.)
    # Probe both wavelength boundary derivatives on a nonzero-density layer.
    h = 1e-4
    for x, sign in ((wx[0], 1), (wx[-1], -1)):
        f = table.correction(np.exp([x, x+sign*h]), t[2:3],
                             .001*np.expm1(d[2:3])/HELIUM_MASS)[:,0]
        derivative = (np.log(f[1])-np.log(f[0]))/(sign*h)
        # Mirror-boundary influence decays by |sqrt(3)-2| per ghost
        # knot for the cubic B-spline filter. Four knots retain the
        # physical slope to ~0.52%, rather than forcing it to zero.
        ghost_bound = 1.01*(2-np.sqrt(3))**table._padding
        assert derivative == pytest.approx(-.02*d[2]*.04, rel=ghost_bound)


@pytest.mark.parametrize("axis", ["wavelength", "temperature", "density"])
def test_table_rejects_extrapolation_in_every_coordinate(synthetic_table, axis):
    from wd_spectra.constants import HELIUM_MASS
    table, _ = synthetic_table
    for w, t, rho in {"wavelength": [(999., 5000., .1), (100001., 5000., .1)],
                      "temperature": [(5000., 1999., .1), (5000., 40001., .1)],
                      "density": [(5000., 5000., 1.201)]}[axis]:
        coordinate = dict(wavelength="log_wavelength", temperature="log_temperature",
                          density="log1p_neutral_density")[axis]
        with pytest.raises(ValueError, match="outside supplied table.*"+coordinate+
                           " requested.*allowed"):
            table.correction(np.array([w]), np.array([t]), np.array([rho/HELIUM_MASS]))
