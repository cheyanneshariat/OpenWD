"""Ionic quasi-static Stark patterns of Rydberg n-manifolds (D6 line profiles)."""

import numpy as np
import pytest

from wd_spectra._compat import trapezoid

from wd_spectra.d6 import SDSS_J1637_LOG_NUMBER_ABUNDANCE
from wd_spectra.eos import hydrogenic_critical_microfield_beta
from wd_spectra.metals import (
    _STARK_MANIFOLD_FIELD_HZ,
    _deposit_segments,
    _voigt_profile_per_angstrom,
    manifold_quasistatic_line_profile,
    rydberg_stark_manifold,
)
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs


@pytest.fixture(scope="module")
def database():
    elements = tuple(sorted(set(SDSS_J1637_LOG_NUMBER_ABUNDANCE) | {"Mg"}))
    return _atomic_inputs(ModelData.default(), elements)[0]


def _manifold(database, key, label):
    ion = database.ions[key]
    level = next(item for item in ion.levels if item.label == label)
    return rydberg_stark_manifold(database, ion, level)


def test_segment_deposit_is_exact_at_bin_edges():
    masses = _deposit_segments(np.array([2.3]), np.array([5.8]), np.array([1.0]), 10)
    assert masses[2:6] == pytest.approx([0.7 / 3.5, 1 / 3.5, 1 / 3.5, 0.8 / 3.5])
    assert masses.sum() == pytest.approx(1.0)
    # Mass outside the window is dropped, not piled onto the edge bin.
    clipped = _deposit_segments(np.array([-2.0]), np.array([2.0]), np.array([1.0]), 10)
    assert clipped[:2] == pytest.approx([0.25, 0.25])
    assert clipped.sum() == pytest.approx(0.5)


def test_strong_field_limit_is_the_parabolic_pattern(database):
    # Mg II 7g in a field far above every quantum-defect gap: the shifts
    # approach the hydrogenic (3/2) n k e a0 F / Z with integer k, and the
    # projections of 7g sum to one at every field.
    manifold = _manifold(database, ("Mg", 1), "2p6.7g.(2G<9/2>)")
    assert manifold.l_values == (0, 1, 2, 3, 4, 5, 6)
    assert np.allclose(manifold.weights.sum(axis=1), 1.0)
    index = int(np.argmin(np.abs(np.log(_STARK_MANIFOLD_FIELD_HZ / 1.0e15))))
    k = manifold.shifts_hz[index] / (1.5 * 7 * _STARK_MANIFOLD_FIELD_HZ[index] / 2.0)
    strong = manifold.weights[index] > 1.0e-3
    assert np.max(np.abs(k[strong] - np.round(k[strong]))) < 0.05
    # The strength-weighted mean |k| is well below the outermost n - 1.
    mean_k = np.sum(manifold.weights[index] * np.abs(k))
    assert 2.0 < mean_k < 3.5


def test_first_moment_of_the_pattern_vanishes(database):
    manifold = _manifold(database, ("Mg", 1), "2p6.10g.(2G<9/2>)")
    first = np.sum(manifold.weights * manifold.shifts_hz, axis=1)
    scale = np.sqrt(manifold.mean_square_shift_hz2)
    assert np.max(np.abs(first) / np.maximum(scale, 1.0)) < 1.0e-8


def test_isolated_levels_have_no_quasistatic_pattern(database):
    # Mg II 4f is split from 4d by ~490 cm^-1 and 3d from 3p by ~36000
    # cm^-1; both stay unshifted at white dwarf microfields.  Mg II 3p is not
    # a Rydberg level (l < 2) and has no pattern.
    manifold = _manifold(database, ("Mg", 1), "2p6.4f.(2Fo<7/2>)")
    index = int(np.argmin(np.abs(np.log(_STARK_MANIFOLD_FIELD_HZ / 2.0e10))))
    unshifted = np.abs(manifold.shifts_hz[index]) < 1.0e9
    assert manifold.weights[index][unshifted].sum() > 0.999
    manifold = _manifold(database, ("Mg", 1), "2p6.3d.(2D<5/2>)")
    assert manifold.weights[index][np.abs(manifold.shifts_hz[index]) < 1.0e9].sum() > 0.999
    assert _manifold(database, ("Mg", 1), "2p6.3p.(2Po<3/2>)") is None


def test_profile_conserves_area_and_reduces_to_impact_at_weak_fields(database):
    upper = _manifold(database, ("Mg", 1), "2p6.7g.(2G<9/2>)")
    lower = _manifold(database, ("Mg", 1), "2p6.4f.(2Fo<7/2>)")
    center = 5403.06
    wavelength = np.arange(center - 60.0, center + 60.0, 0.01)
    # The impact core is the exact frequency-space Voigt profile (2026-09-30 audit).
    impact = _voigt_profile_per_angstrom(wavelength, center, 0.03, 0.05)
    weak = manifold_quasistatic_line_profile(wavelength, center, 0.03, 0.05, 1.0e5, upper, lower)
    assert trapezoid(weak, wavelength) == pytest.approx(trapezoid(impact, wavelength), rel=1.0e-3)
    assert np.max(np.abs(weak - impact)) < 1.0e-3 * np.max(impact)
    strong = manifold_quasistatic_line_profile(wavelength, center, 0.03, 0.05, 2.2e10, upper, lower)
    area = trapezoid(strong, wavelength)
    assert area == pytest.approx(trapezoid(impact, wavelength), rel=2.0e-2)
    assert abs(trapezoid(strong * (wavelength - center), wavelength) / area) < 0.05
    assert np.max(strong) < 0.2 * np.max(impact)


def test_profile_excludes_fields_that_dissolve_the_level(database):
    upper = _manifold(database, ("Mg", 1), "2p6.10g.(2G<9/2>)")
    center = 4333.17
    wavelength = np.arange(center - 60.0, center + 60.0, 0.01)
    beta_critical = float(hydrogenic_critical_microfield_beta(1.0e16, 10.0, 2.0))
    assert 5.0 < beta_critical < 8.0
    coupling = 0.481 * 1.0e16 ** (2.0 / 3.0)
    full = manifold_quasistatic_line_profile(wavelength, center, 0.03, 0.05, coupling, upper, None)
    bound = manifold_quasistatic_line_profile(
        wavelength, center, 0.03, 0.05, coupling, upper, None, beta_critical
    )
    far = np.abs(wavelength - center) > 45.0
    assert trapezoid(bound[far], wavelength[far]) < 0.8 * trapezoid(full[far], wavelength[far])
    # Both are normalized to unit area over the full tail: the dissolved
    # strength is removed separately by the Q-MHD survival factor, and the
    # profile is the conditional bound-state distribution.
    wavelength = np.arange(center - 3000.0, center + 3000.0, 0.1)
    for maximum_beta in (np.inf, beta_critical):
        profile = manifold_quasistatic_line_profile(
            wavelength, center, 0.03, 0.05, coupling, upper, None, maximum_beta,
            support_half_width=3000.0,
        )
        assert trapezoid(profile, wavelength) == pytest.approx(1.0, abs=2.0e-3)


def test_truncated_pattern_ends_at_the_largest_bound_shift(database):
    # No quasi-static strength may lie beyond the largest component shift at
    # the critical field; beyond it only the impact (Lorentz) wing remains.
    upper = _manifold(database, ("Mg", 1), "2p6.10g.(2G<9/2>)")
    center = 4333.17
    coupling = 0.481 * 1.0e16 ** (2.0 / 3.0)
    beta_critical = float(hydrogenic_critical_microfield_beta(1.0e16, 10.0, 2.0))
    edge_field = beta_critical * coupling
    edge_shift_hz = np.interp(
        np.log(edge_field), np.log(_STARK_MANIFOLD_FIELD_HZ), np.max(np.abs(upper.shifts_hz), axis=1)
    )
    edge_angstrom = center**2 * edge_shift_hz / 2.99792458e18
    wavelength = np.arange(center - 300.0, center + 300.0, 0.02)
    sigma, hwhm = 0.03, 0.05
    profile = manifold_quasistatic_line_profile(
        wavelength, center, sigma, hwhm, coupling, upper, None, beta_critical,
        support_half_width=300.0,
    )
    impact = _voigt_profile_per_angstrom(wavelength, center, sigma, hwhm)
    beyond = np.abs(wavelength - center) > 1.05 * edge_angstrom + 30.0 * hwhm
    assert np.all(profile[beyond] <= 1.0001 * impact[beyond])


def test_mg_ii_series_uses_kurucz_widths_consistently(database):
    from wd_spectra.metals import (
        mg_ii_4852_electron_stark_rate_coefficient,
        mg_ii_kurucz_electron_stark_rate_coefficient,
    )

    ion = database.ions[("Mg", 1)]
    levels = {level.label: level for level in ion.levels}
    rate = lambda lower, upper: float(  # noqa: E731
        mg_ii_kurucz_electron_stark_rate_coefficient(levels[lower], levels[upper], 10_000.0)
    )
    # 4f-8g (4852 A) reproduces the line-specific value used before.
    assert rate("2p6.4f.(2Fo<7/2>)", "2p6.8g.(2G<9/2>)") == pytest.approx(
        float(mg_ii_4852_electron_stark_rate_coefficient(4852.4, 10_000.0))
    )
    # Its series neighbours now use Kurucz too (not the capped fallback).
    assert rate("2p6.4d.(2D<5/2>)", "2p6.7f.(2Fo<7/2>)") == pytest.approx(10.0**-2.82)
    assert rate("2p6.4f.(2Fo<7/2>)", "2p6.10g.(2G<9/2>)") == pytest.approx(10.0**-2.11)
    # Widths grow along a Rydberg series.
    assert rate("2p6.4f.(2Fo<7/2>)", "2p6.10g.(2G<9/2>)") > rate("2p6.4f.(2Fo<7/2>)", "2p6.6g.(2G<9/2>)")
