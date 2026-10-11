"""Regression tests for the magneto-optical dispersion (Kramers--Kronig partner) of DAH line opacity.

The default DAH line prescription (frequency-normalized Kurucz/Griem templates) produced two template
mesh nodes that round to the same frequency, a 0/0 slope and non-finite dispersion everywhere, so
full-Stokes transfer failed. These tests check the Hilbert transform against analytic partners, its
invariance to coincident nodes, its rejection of discontinuities, and full-Stokes synthesis with the
default line prescription (finiteness, positivity and the zero-field limit).
"""

import numpy as np
import pytest
from scipy.special import dawsn

from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.kurucz_griem import kurucz_griem_templates
from wd_spectra.magnetic import (
    BalmerBroadening,
    MagneticPhysics,
    PolarizedOpacity,
    frequency_hilbert_dispersion,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import read_h2db_energy_database, read_h2db_transition_database
from wd_spectra.models.common import ModelData

H2DB = ModelData.default().h2db_balmer_subset
NU0 = 0.5  # 1e15 Hz


def _mesh(frequency_1e15):
    """Wavelength mesh (A, ascending) for frequencies in units of 1e15 Hz."""
    return np.sort(LIGHT_SPEED / (np.asarray(frequency_1e15) * 1e15) / 1e-8)


def _psi(wavelength, profile):
    return frequency_hilbert_dispersion(wavelength, profile[:, None])[:, 0]


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True)


def test_hilbert_matches_the_lorentzian_partner():
    # psi = -H[L], H[L](x) = (x - x0) / (pi ((x - x0)^2 + g^2)): positive on the low-frequency side.
    g = 2e-4
    detuning = np.concatenate((-np.geomspace(g * 1e-3, 0.3, 1500), [0.0], np.geomspace(g * 1e-3, 0.3, 1500)))
    wavelength = _mesh(NU0 + detuning)
    x = LIGHT_SPEED / (wavelength * 1e-8) / 1e15 - NU0
    lorentz = (g / np.pi) / (x**2 + g**2)
    exact = -x / (np.pi * (x**2 + g**2))
    core = np.abs(x) < 50 * g
    psi = _psi(wavelength, lorentz)
    # Truncation of the wings at |x| = 0.3 changes psi by ~ (1/pi^2) ln-terms; compare relative to the peak.
    assert np.max(np.abs(psi[core] - exact[core])) < 2e-3 * np.max(np.abs(exact))


def test_hilbert_matches_the_gaussian_dawson_partner():
    # f = exp(-u^2), u = x/w: H[f](x) = (2/sqrt(pi)) D(u) (Dawson function), so psi = -(2/sqrt(pi)) D(u).
    w = 1e-3
    detuning = np.linspace(-12 * w, 12 * w, 4001)
    wavelength = _mesh(NU0 + detuning)
    u = (LIGHT_SPEED / (wavelength * 1e-8) / 1e15 - NU0) / w
    psi = _psi(wavelength, np.exp(-u**2))
    exact = -(2 / np.sqrt(np.pi)) * dawsn(u)
    assert np.max(np.abs(psi - exact)) < 1e-4


def test_coincident_frequency_nodes_do_not_change_or_poison_the_partner():
    w = 1e-3
    frequency = NU0 + np.linspace(-12 * w, 12 * w, 801)
    wavelength = _mesh(frequency)
    profile = np.exp(-((LIGHT_SPEED / (wavelength * 1e-8) / 1e15 - NU0) / w) ** 2)
    reference = _psi(wavelength, profile)
    # Insert a distinct wavelength node that rounds to (nearly) the same frequency as node 400.
    duplicate = np.nextafter(wavelength[400], np.inf)
    assert duplicate != wavelength[400]
    assert abs(LIGHT_SPEED / (duplicate * 1e-8) - LIGHT_SPEED / (wavelength[400] * 1e-8)) <= 4 * np.spacing(NU0 * 1e15)
    mesh = np.insert(wavelength, 401, duplicate)
    values = np.insert(profile, 401, profile[400])
    psi = _psi(mesh, values)
    assert np.all(np.isfinite(psi))
    np.testing.assert_allclose(np.delete(psi, 401), reference, rtol=0, atol=1e-12)
    assert psi[401] == pytest.approx(psi[400], abs=1e-12)


def test_hilbert_rejects_discontinuous_and_nonfinite_profiles():
    wavelength = _mesh(NU0 + np.linspace(-0.01, 0.01, 101))
    duplicate = np.nextafter(wavelength[50], np.inf)
    mesh = np.insert(wavelength, 51, duplicate)
    values = np.insert(np.ones(101), 51, 2.0)  # a jump at one frequency
    with pytest.raises(ValueError, match="discontinuous"):
        frequency_hilbert_dispersion(mesh, values[:, None])
    bad = np.ones(101)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        frequency_hilbert_dispersion(wavelength, bad[:, None])


def test_default_kurucz_griem_templates_have_finite_dispersion(atmosphere):
    templates = kurucz_griem_templates(atmosphere, maximum_upper_level=12, include_dispersion=True)
    rest = {3: 6564.636, 4: 4862.694, 5: 4341.692}
    for level, template in templates.items():
        assert np.all(np.isfinite(template.dispersion))
        if level in rest:
            # Far from a broad positive line psi ~ -A / (pi (nu - nu0)): positive on the red side.
            w = template.wavelength_angstrom
            red = np.argmin(abs(w - (rest[level] + 150.0)))
            blue = np.argmin(abs(w - (rest[level] - 150.0)))
            assert template.dispersion[red, 12] > 0.0 > template.dispersion[blue, 12]


def test_kurucz_griem_dispersion_is_invariant_to_added_collinear_nodes(atmosphere):
    # Adding every interval midpoint, valued by linear interpolation in frequency, leaves the piecewise-linear
    # profile unchanged, so its exact partner must be unchanged at the original nodes (Hbeta, depth 12).
    template = kurucz_griem_templates(atmosphere, maximum_upper_level=4, include_dispersion=True)[4]
    w, kappa = template.wavelength_angstrom, template.mass_absorption_coefficient[:, 12]
    frequency = LIGHT_SPEED / (w * 1e-8)
    mid_frequency = 0.5 * (frequency[1:] + frequency[:-1])
    mid_kappa = 0.5 * (kappa[1:] + kappa[:-1])
    mesh = LIGHT_SPEED / np.concatenate((frequency, mid_frequency)) / 1e-8
    order = np.argsort(mesh)
    psi = _psi(mesh[order], np.concatenate((kappa, mid_kappa))[order])
    original = np.argsort(order)[: w.size]
    scale = np.max(np.abs(template.dispersion[:, 12]))
    np.testing.assert_allclose(psi[original], template.dispersion[:, 12], rtol=0, atol=1e-7 * scale)


def _default_physics(field):
    strong = field > 1.0
    return MagneticPhysics(
        read_h2db_transition_database(H2DB) if strong else None,
        read_h2db_energy_database(H2DB) if strong else None,
        broadening=BalmerBroadening(include_self_broadening=False),
        balmer_profile="kurucz-griem", normalize_balmer_strength=True,
        line_regime="h2db" if strong else "linear-zeeman",
    )


def test_default_profile_full_stokes_synthesis_is_finite_and_positive(atmosphere):
    wavelength = np.arange(4000.0, 4200.5, 1.0)
    cells = uniform_field_surface_cells(300.0, field_angle_deg=50.0)
    spectrum = synthesize_magnetic_hydrogen_spectrum(
        atmosphere, wavelength, cells, _default_physics(300.0), polarized_transfer="full-stokes-iquv")
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))
    assert np.all(spectrum.surface_flux_lambda > 0.0)


def test_default_profile_full_stokes_reduces_to_scalar_at_zero_field(atmosphere):
    wavelength = np.arange(4800.0, 4930.5, 0.5)
    cells = uniform_field_surface_cells(0.0)
    flux = {
        transfer: synthesize_magnetic_hydrogen_spectrum(
            atmosphere, wavelength, cells, _default_physics(0.0), polarized_transfer=transfer).surface_flux_lambda
        for transfer in ("scalar-stokes-i", "full-stokes-iquv")
    }
    np.testing.assert_allclose(flux["full-stokes-iquv"], flux["scalar-stokes-i"], rtol=1e-6)


def _psi_reference(freq, values):
    """-H at every node (40-digit mpmath) for the piecewise-linear profile, zero outside, no merging."""
    import mpmath as mp

    mp.mp.dps = 40
    y = [mp.mpf(float(v)) for v in freq]
    f = [mp.mpf(float(v)) for v in values]
    out = []
    for k, x in enumerate(y):
        total = mp.mpf(0)
        for j in range(len(y) - 1):
            b = (f[j + 1] - f[j]) / (y[j + 1] - y[j])
            fx = f[j] + b * (x - y[j])
            total -= b * (y[j + 1] - y[j])
            if j == k - 1:
                total += fx * mp.log(abs(x - y[j]))
            elif j == k:
                total -= fx * mp.log(abs(x - y[j + 1]))
            else:
                total += fx * mp.log(abs((x - y[j]) / (x - y[j + 1])))
        out.append(float(-total / mp.pi))
    return np.array(out)


@pytest.mark.parametrize("ulps", [8, 63, 65, 1000])
def test_near_coincident_nodes_match_a_high_precision_reference(ulps):
    # The 8-, 63- and 65-ULP gaps here lie within 64 relative machine epsilons and are merged;
    # the 1000-ULP gap is retained. Both reproduce the exact piecewise-linear partner.
    freq = NU0 + np.linspace(-12e-3, 12e-3, 241)
    values = np.exp(-((freq - NU0) / 1e-3) ** 2)
    new = freq[90] + ulps * np.spacing(freq[90])
    freq = np.insert(freq, 91, new)
    values = np.insert(values, 91, np.interp(new, freq[[90, 92]], values[[90, 91]]))
    wavelength = LIGHT_SPEED / (freq * 1e15) / 1e-8
    psi = frequency_hilbert_dispersion(wavelength[::-1], values[::-1, None])[::-1, 0]
    reference = _psi_reference(freq, values)
    assert np.max(np.abs(psi - reference)) < 1e-10 * np.max(np.abs(reference))


@pytest.mark.parametrize("frequency", [0.497, 1.003])
@pytest.mark.parametrize("relative_epsilons, merged", [(63, True), (65, False)])
def test_discontinuous_nodes_bracket_the_relative_epsilon_merge_boundary(frequency, relative_epsilons, merged):
    # Bracket 64 relative epsilons at two mantissas, where a fixed ULP count would give different gaps.
    # Verify which side survives the wavelength conversion before testing rejection of a value jump.
    neighbour = frequency * (1.0 + relative_epsilons * np.finfo(np.float64).eps)
    freq = np.array([frequency - 0.1, frequency, neighbour, frequency + 0.1])
    wavelength = _mesh(freq)
    roundtrip = (LIGHT_SPEED / (wavelength * 1e-8) / 1e15)[::-1]
    gap = roundtrip[2] - roundtrip[1]
    assert gap > 0.0
    assert bool(gap <= 64 * np.finfo(np.float64).eps * roundtrip[2]) is merged
    profile = np.array([0.0, 1.0, 2.0, 0.0])[::-1]
    if merged:
        with pytest.raises(ValueError, match="discontinuous"):
            _psi(wavelength, profile)
    else:
        assert np.all(np.isfinite(_psi(wavelength, profile)))


@pytest.mark.parametrize("cosine, k_first, k_second", [(1.0, 10.0, 0.1), (0.0, 1.0, 5.05)])
def test_constant_coefficient_stokes_transfer_splits_into_exact_modes(cosine, k_first, k_second):
    # Linear LTE source S = a + b m. At psi = 0 the Stokes system splits into circular modes (kappa_+,
    # kappa_-); at psi = 90 deg (rho_V = 0) into pi and sigma modes. Each mode has I = a + b mu / k exactly,
    # so Stokes I is the mode mean -- not the intensity of the mean opacity used by scalar transfer.
    from wd_spectra.radiative_transfer import emergent_stokes_specific_intensity

    m = np.concatenate(([0.0], np.geomspace(1e-6, 1e4, 600)))
    a, b, mu = 1.0, 0.5, 0.7
    source = a + b * m
    manifolds = PolarizedOpacity(np.full((1, m.size), 0.1), np.full((1, m.size), 1.0), np.full((1, m.size), 10.0))
    eta_i, eta_q, eta_v = manifolds.stokes(cosine)
    emission = np.zeros((1, m.size, 4))
    emission[..., 0], emission[..., 1], emission[..., 3] = eta_i * source, eta_q * source, eta_v * source
    intensity = emergent_stokes_specific_intensity(
        m, eta_i, eta_q, eta_v, np.full_like(eta_i, 2.0 * (1 - cosine**2)), np.full_like(eta_i, 3.0 * cosine),
        source[None, :], mu, emission_stokes=emission, formal_solver="delo-linear").i[0]
    exact = 0.5 * ((a + b * mu / k_first) + (a + b * mu / k_second))
    assert intensity == pytest.approx(exact, rel=1e-10)
    assert abs(intensity - (a + b * mu / eta_i[0, 0])) > 0.05  # the scalar mean-opacity answer is far off


def _group_case(step_value):
    f = np.r_[.75, .85, .95, 1. + np.arange(64) * 8 * np.spacing(1.), 1.05, 1.15, 1.25]
    v = np.r_[0., 1., .5, .4 + np.arange(64) * step_value, .5, .2, 0.]
    wave = (LIGHT_SPEED / (f * 1e15) / 1e-8)[::-1]
    return wave, v[::-1], (LIGHT_SPEED / (wave * 1e-8) / 1e15)[::-1], v


def test_drifting_coincident_group_is_rejected_not_silently_merged():
    # Reviewer's case: adjacent changes 4e-10 (< 1e-9) but the 64-node chain drifts 2.5e-8; the drift
    # relative to the retained node exceeds the tolerance within one 64-relative-epsilon window.
    wave, values, _, _ = _group_case(4e-10)
    with pytest.raises(ValueError, match="discontinuous"):
        frequency_hilbert_dispersion(wave, values[:, None])


def test_merged_coincident_group_stays_within_the_stated_bound():
    # A slowly drifting chain (4e-11 per node) is merged in anchored groups and reproduces the 40-digit
    # unmerged reference within the stated coalescence bound (~2e-8 of the peak).
    wave, values, f_roundtrip, v = _group_case(4e-11)
    psi = frequency_hilbert_dispersion(wave, values[:, None])[::-1, 0]
    reference = _psi_reference(f_roundtrip, v)
    assert np.max(np.abs(psi - reference)) < 2e-8 * np.max(np.abs(reference))
