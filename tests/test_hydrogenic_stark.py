"""Hydrogenic linear-Stark component patterns for formula-4 quasi-static wings."""

import numpy as np
import pytest

import wd_spectra.light_metal_nlte as light
from wd_spectra._compat import trapezoid
from wd_spectra._hydrogenic_stark import (
    COMPOSITE_LOG_BETA_MAX,
    COMPOSITE_LOG_BETA_MIN,
    COMPOSITE_POINTS,
    composite_static_profile,
    composite_static_profile_value,
    formula4_extreme_shift,
    hydrogenic_stark_pattern,
)
from wd_spectra._pg1159_builder import (
    PG1159_HYDROGENIC_STARK_COMPONENTS,
    PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES,
)


def test_lyman_alpha_pattern():
    shifts, strengths = hydrogenic_stark_pattern(1, 2)
    assert np.allclose(shifts, [0.0, 2.0])
    # 2/3 unshifted (sigma), 1/6 at each of X = +-2 (pi).
    assert np.allclose(strengths, [2.0 / 3.0, 1.0 / 3.0], atol=1e-10)


def test_balmer_alpha_pattern_matches_bethe_salpeter():
    shifts, strengths = hydrogenic_stark_pattern(2, 3)
    pattern = dict(zip(shifts.astype(int), strengths))
    # Bethe & Salpeter (1957) relative intensities of one sign of X
    # (pi: 2, 3, 4, 8; sigma: 0, 1, 5, 6); the pattern combines +X and -X.
    single_sign = {0: 10980.0, 1: 3872.0, 2: 729.0, 3: 2304.0,
                   4: 1681.0, 5: 32.0, 6: 36.0, 8: 1.0}
    reference = {x: (1.0 if x == 0 else 2.0) * v for x, v in single_sign.items()}
    scale = pattern[2] / reference[2]
    for shift, value in reference.items():
        assert pattern[shift] == pytest.approx(value * scale, rel=1e-8)


@pytest.mark.parametrize("pair", [(4, 9), (7, 8), (3, 3), (8, 17)])
def test_patterns_are_normalized_and_bounded_by_formula4(pair):
    shifts, strengths = hydrogenic_stark_pattern(*pair)
    assert strengths.sum() == pytest.approx(1.0, abs=1e-10)
    assert shifts.max() <= formula4_extreme_shift(*pair) + 1e-12


def test_composite_profile_carries_the_shifted_strength():
    beta = np.logspace(COMPOSITE_LOG_BETA_MIN, COMPOSITE_LOG_BETA_MAX, COMPOSITE_POINTS)
    for pair in ((4, 9), (7, 8)):
        shifts, strengths = hydrogenic_stark_pattern(*pair)
        area = trapezoid(composite_static_profile(*pair), beta)
        # The formula-4 support ends at beta = 30; the lost far wing is < 0.5%.
        assert area == pytest.approx(strengths[shifts > 0].sum(), rel=5e-3)
    table = composite_static_profile(7, 8)
    assert np.allclose(composite_static_profile_value(table, beta), table, rtol=1e-12)


def test_component_wings_are_much_weaker_than_formula4_for_delta_n_one():
    shifts, strengths = hydrogenic_stark_pattern(7, 8)
    extreme = formula4_extreme_shift(7, 8)
    far_wing = np.sum(strengths * (shifts / extreme) ** 1.5)
    assert far_wing < 0.02


def _profile_inputs():
    rng = np.random.default_rng(1)
    n_wave, n_depth = 4000, 5
    wave = np.linspace(5200.0, 5400.0, n_wave)
    planck = rng.uniform(1.0, 2.0, (n_wave, n_depth))
    center = np.array([5250.0, 5290.0, 5300.0, 5350.0])
    strength = np.array([1e-2, 2e-2, 1e-2, 3e-2])
    shape = (center.size, n_depth)
    field = np.full(shape, 3e11)
    field[0] = 0.0
    return (
        wave, planck, center, strength, np.full(shape, 0.05), np.full(shape, 0.02),
        np.zeros(center.size), field, np.full(shape, 2e-13), np.zeros(shape),
        np.ones(shape), np.ones(shape), np.full(shape, 0.5), np.full(shape, 0.1),
    )


def _accumulate(function, pattern):
    arguments = _profile_inputs()
    absorption = np.zeros_like(arguments[1])
    emissivity = np.zeros_like(arguments[1])
    function(*arguments, absorption, emissivity, False, static_pattern=pattern)
    return absorption, emissivity


@pytest.mark.skipif(light._rt is None, reason="compiled extension unavailable")
def test_compiled_kernel_matches_python_with_patterns():
    pattern = [None, (7, 8), None, (4, 9)]
    compiled = _accumulate(light._accumulate_metal_line_profiles, pattern)
    reference = _accumulate(light._accumulate_metal_line_profiles_python, pattern)
    for c, r in zip(compiled, reference):
        assert np.allclose(c, r, rtol=1e-12, atol=1e-12 * np.max(abs(r)))
    plain = _accumulate(light._accumulate_metal_line_profiles, None)
    assert np.max(abs(compiled[0] - plain[0])) > 1e-3 * np.max(plain[0])


def test_tabulated_stark_widths_apply_only_to_their_multiplet():
    fwhm = light._tabulated_electron_stark_fwhm_angstrom
    assert fwhm("O", 5, 1031.912, 1e17, 1e5) == pytest.approx(0.00146)
    assert fwhm("O", 5, 1033.81, 1e17, 1e5) is not None      # rate-atom centroid
    assert fwhm("O", 5, 1037.13, 1e17, 1e5) is None          # O VI 5-8 interloper
    assert fwhm("O", 4, 5616.39, 1e17, 1e5) is None          # O V 2p4s-2s6d
    assert fwhm("C", 3, 1557.3, 1e17, 1e5) is None           # C IV 5d-17f


def test_pg1159_defaults_use_components_without_series_scales():
    assert PG1159_HYDROGENIC_STARK_COMPONENTS is True
    assert dict(PG1159_STATIC_LINEAR_STARK_FREQUENCY_SCALES) == {}
