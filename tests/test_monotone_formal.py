"""Independent quadrature, analytic tangents, scattering and default isolation."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from wd_spectra._monotone_formal import CubicFormal, slopes, solve_cubic_source
from wd_spectra._spectrum_source import solve_spectrum_source
from wd_spectra.radiative_transfer import angular_quadrature, radiation_field, emergent_flux


@pytest.mark.parametrize("kind", ["constant", "linear", "smooth", "turnover"])
def test_cubic_flux_matches_independent_scipy_quadrature(kind):
    x = np.geomspace(1e-6, 30., 18)
    y = {"constant": np.ones_like(x), "linear": 1+x,
         "smooth": 1+x**.4, "turnover": 1+np.exp(-((np.log(x)+1)/2)**2)}[kind]
    pp = PchipInterpolator(x, y)
    expected = 0.
    for mu, weight in zip(*angular_quadrature(3)):
        intensity = y[0]*-np.expm1(-x[0]/mu)+y[-1]*np.exp(-x[-1]/mu)
        for lo, hi in zip(x[:-1], x[1:]):
            intensity += quad(lambda t: pp(t)*np.exp(-t/mu)/mu, lo, hi,
                              epsabs=1e-13, epsrel=1e-12)[0]
        expected += 2*np.pi*mu*weight*intensity
    field, flux = CubicFormal(x[None, :], 3).field(y[None, :])
    np.testing.assert_allclose(flux, expected, rtol=2e-13)
    if kind in ("constant", "linear"):
        ordinary = radiation_field(x, y[None, :], n_angle=3)
        np.testing.assert_allclose(field.mean_intensity, ordinary.mean_intensity, rtol=2e-13)


def test_pchip_slopes_and_exact_source_jacobian():
    rng = np.random.default_rng(9)
    x = np.broadcast_to(np.geomspace(1e-7, 100., 25), (3, 25)).copy()
    y = np.exp(rng.normal(size=x.shape))
    direction = rng.normal(size=x.shape)
    m, derivative = slopes(x, y, jacobian=True)
    expected = PchipInterpolator(x[0], y, axis=1).derivative()(x[0])
    np.testing.assert_allclose(m, expected, rtol=1e-11, atol=1e-8)
    formal = CubicFormal(x, 3)
    tangent = formal.field_from_slopes(np.broadcast_to(np.eye(25), derivative.shape), derivative)[0]
    eps = 1e-6
    finite = (formal.field(y+eps*direction)[0].mean_intensity
              - formal.field(y-eps*direction)[0].mean_intensity)/(2*eps)
    actual = np.einsum('ijk,ik->ij', tangent.mean_intensity, direction)
    np.testing.assert_allclose(actual, finite, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("epsilon", [1., .1, .001, 1e-6])
def test_closed_source_including_strong_scattering(epsilon):
    tau = np.broadcast_to(np.geomspace(1e-8, 100., 40), (2, 40)).copy()
    b = 1+tau**.25
    source, field, flux, record = solve_cubic_source(tau, b, epsilon*np.ones_like(b),
        (1-epsilon)*np.ones_like(b), wavelength=np.array([2000., 5000.]), n_angle=3)
    independent = CubicFormal(tau, 3).field(source)[0]
    np.testing.assert_allclose(source, epsilon*b+(1-epsilon)*independent.mean_intensity,
                               rtol=1e-8, atol=1e-10)
    assert record['source_converged']
    assert record['independent_radiation_scaled_source_error'] < 1e-10
    assert np.all(flux > 0.) and np.all(source >= 0.)


def test_chunking_does_not_change_solution():
    tau = np.broadcast_to(np.geomspace(1e-8, 100., 24), (7, 24)).copy()
    b = (1+tau**.25)*np.arange(1, 8)[:, None]
    opts = dict(wavelength=np.linspace(2000., 6000., 7), n_angle=3)
    a = solve_cubic_source(tau, b, .1*np.ones_like(b), .9*np.ones_like(b),
                          wavelength_chunk_size=2, **opts)
    c = solve_cubic_source(tau, b, .1*np.ones_like(b), .9*np.ones_like(b),
                          wavelength_chunk_size=64, **opts)
    np.testing.assert_allclose(a[0], c[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(a[2], c[2], rtol=2e-10)


def test_unclosed_source_raises_without_fallback():
    tau = np.broadcast_to(np.geomspace(1e-8, 100., 24), (2, 24)).copy()
    with pytest.raises(RuntimeError, match='did not converge'):
        solve_cubic_source(tau, 1+tau**.25, .1*np.ones_like(tau), .9*np.ones_like(tau),
                          wavelength=np.array([2000., 5000.]), max_iterations=0)


@pytest.mark.parametrize("option,value", [("max_iterations", -1), ("max_iterations", True),
                                         ("wavelength_chunk_size", 0)])
def test_bad_budgets_fail(option, value):
    tau = np.array([[.1, 1., 10.]])
    with pytest.raises(ValueError, match=option):
        solve_cubic_source(tau, np.ones_like(tau), 1., 0., wavelength=[5000.], **{option:value})


def test_explicit_linear_source_is_unchanged_and_does_not_enter_cubic(monkeypatch):
    import wd_spectra._monotone_formal as cubic
    tau = np.broadcast_to(np.geomspace(1e-8, 100., 24), (2, 24)).copy()
    opts = dict(wavelength=np.array([2000., 5000.]), n_angle=3, discretization='formal-linear')
    before = solve_spectrum_source(tau, 1+tau**.25, .1, .9, **opts)
    def forbidden(*args, **kwargs):
        raise AssertionError('cubic method entered by explicit linear selection')
    monkeypatch.setattr(cubic, 'solve_cubic_source', forbidden)
    after = solve_spectrum_source(tau, 1+tau**.25, .1, .9, **opts)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[2] == after[2]


def test_public_da_option_only_changes_synthesis(monkeypatch):
    from wd_spectra.models import stellar
    from wd_spectra import DAConfig, compute_da
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.spectrum import Spectrum
    atmosphere = gray_hydrogen_atmosphere(20000., 8., n_depth=8)
    calls = []
    def atmosphere_solve(*args, **kwargs):
        calls.append(('atmosphere', kwargs))
        return atmosphere
    def synthesis(*args, **kwargs):
        calls.append(('synthesis', kwargs))
        return Spectrum(np.array([4000., 5000.]), np.ones(2), {})
    monkeypatch.setattr(stellar, 'radiative_equilibrium_hydrogen_atmosphere', atmosphere_solve)
    monkeypatch.setattr(stellar, 'synthesize_hydrogen_spectrum', synthesis)
    monkeypatch.setattr(stellar, 'warn_if_atmosphere_not_converged', lambda *args: 'unconverged')
    config = DAConfig(effective_temperature=20000., lyman_profile_source='stark')
    compute_da(config, [4000., 5000.])
    compute_da(config, [4000., 5000.], synthesis_transfer='formal-linear')
    assert calls[1][1]['transfer_discretization'] == 'formal-pchip'
    assert calls[3][1]['transfer_discretization'] == 'formal-linear'
    assert 'synthesis_transfer' not in calls[0][1]
    for key in ('n_depth', 'n_continuum_wavelength', 'n_angle', 'max_iterations',
                'initial_temperature', 'initial_column_mass'):
        assert calls[0][1][key] == calls[2][1][key]


def test_invalid_public_option_fails_before_calculation():
    from wd_spectra import compute_da, compute_dz
    for compute in (compute_da, compute_dz):
        with pytest.raises(ValueError, match='synthesis_transfer'):
            compute(synthesis_transfer='flux-rescale')


@pytest.mark.parametrize('temperature', [5000., 13950., 22000.])
def test_helium_cubic_synthesis_closes_without_changing_legacy(temperature):
    from wd_spectra.atmosphere import gray_helium_atmosphere
    from wd_spectra.spectrum import synthesize_helium_spectrum
    atmosphere = gray_helium_atmosphere(temperature, 8., n_depth=24)
    original_temperature = atmosphere.temperature.copy()
    wave = np.geomspace(900., 300000., 100)
    options = dict(stark_table=None, include_lines=False,
                   include_uv_resonance_lines=False, include_helium_ii_lines=False)
    before = synthesize_helium_spectrum(atmosphere, wave, **options)
    cubic = synthesize_helium_spectrum(atmosphere, wave,
                                     transfer_discretization='formal-pchip', **options)
    after = synthesize_helium_spectrum(atmosphere, wave, **options)
    np.testing.assert_array_equal(before.surface_flux_lambda, after.surface_flux_lambda)
    np.testing.assert_array_equal(atmosphere.temperature, original_temperature)
    assert cubic.metadata['transfer_discretization'] == 'formal-pchip'
    assert cubic.metadata['source_converged']
    assert cubic.metadata['independent_radiation_scaled_source_error'] < 1e-10
    assert np.all(np.isfinite(cubic.surface_flux_lambda))
    assert np.all(cubic.surface_flux_lambda >= 0.)


def test_public_dz_option_only_changes_synthesis(monkeypatch):
    from types import SimpleNamespace
    from wd_spectra.models import stellar
    from wd_spectra import DZConfig, compute_dz
    from wd_spectra.atmosphere import gray_helium_atmosphere
    from wd_spectra.spectrum import Spectrum
    atmosphere = gray_helium_atmosphere(13950., 8.4, n_depth=8)
    calls = []
    def atmosphere_solve(*args, **kwargs):
        calls.append(('atmosphere', kwargs))
        return atmosphere
    def synthesis(*args, **kwargs):
        calls.append(('synthesis', kwargs))
        return Spectrum(np.array([4000., 5000.]), np.ones(2), {})
    monkeypatch.setattr(stellar, 'solve_with_screened_boundary', atmosphere_solve)
    monkeypatch.setattr(stellar, 'synthesize_helium_spectrum', synthesis)
    monkeypatch.setattr(stellar, '_helium_tables', lambda *args: (None, None))
    monkeypatch.setattr(stellar.ModelData, 'require', lambda *args, **kwargs: None)
    monkeypatch.setattr(stellar, 'read_stout_atomic_database', lambda *args, **kwargs: SimpleNamespace(source='test'))
    monkeypatch.setattr(stellar, 'read_verner_photoionization_database', lambda *args, **kwargs: SimpleNamespace(fits={}, source='test'))
    monkeypatch.setattr(stellar, 'read_verner_phfit2_database', lambda *args, **kwargs: {})
    monkeypatch.setattr(stellar, '_polluted_helium_level_resolved_photoionization', lambda *args, **kwargs: None)
    monkeypatch.setattr(stellar, 'warn_if_atmosphere_not_converged', lambda *args: 'unconverged')
    # A DZ model needs at least one metal; the trace Fe abundance is inert here.
    config = DZConfig(abundances={'Fe': -10.0}, dense_helium_eos='ideal',
                      unified_metal_helium_profiles='off', ca_ii_resonance_source='lte')
    compute_dz(config, [4000., 5000.], synthesis_transfer='formal-linear')
    compute_dz(config, [4000., 5000.], synthesis_transfer='formal-pchip')
    assert calls[1][1]['transfer_discretization'] == 'formal-linear'
    assert calls[3][1]['transfer_discretization'] == 'formal-pchip'
    assert calls[0][1] == calls[2][1]
