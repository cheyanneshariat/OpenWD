from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import wd_spectra.opacity as opacity_module

from wd_spectra._compat import trapezoid
from wd_spectra import (
    default_balmer_stark_table,
    default_brackett_stark_table,
    default_lyman_stark_table,
    default_paschen_stark_table,
    gray_hydrogen_atmosphere,
    gray_hydrogen_helium_atmosphere,
    hydrogen_bound_free_gaunt_factor,
    hydrogen_continuum_atmosphere,
    hydrogen_free_free_gaunt_factor,
    radiative_equilibrium_hydrogen_atmosphere,
    synthesize_balmer_spectrum,
    synthesize_gray_spectrum,
    synthesize_hydrogen_spectrum,
)
from wd_spectra.opacity import (
    BALMER_LINES,
    BRACKETT_LINES,
    LYMAN_LINES,
    PASCHEN_LINES,
    _h_h2_lyman_alpha_log_temperature_correction,
    _hminus_cross_section_per_electron_pressure,
    _lorentz_convolved_stark_profile,
    balmer_mass_absorption_coefficient,
    brackett_mass_absorption_coefficient,
    charged_particle_hydrogen_occupation_probability,
    hydrogen_continuum_mass_absorption_coefficient,
    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    hydrogen_ground_state_photoionization_cross_section,
    hydrogen_level_population,
    molecular_hydrogen_mass_absorption_coefficient,
    h2_h2_cia_mass_absorption_coefficient,
    hydrogen_rayleigh_scattering_cross_section,
    hydrogen_rayleigh_scattering_mass_coefficient,
    molecular_hydrogen_rayleigh_scattering_cross_section,
    lyman_mass_absorption_coefficient,
    lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient,
    neutral_hydrogen_self_broadening_hwhm,
    neutral_helium_balmer_broadening_hwhm,
    optical_depth_from_mass_opacity,
    paschen_mass_absorption_coefficient,
)
from wd_spectra.constants import (
    BOLTZMANN,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PLANCK,
)
from wd_spectra.convection import (
    ml2_convective_flux,
    ml2_convective_flux_for_gradient,
    ml2_convective_flux_for_gradient_from_thermodynamics,
    ml2_convective_flux_gradient_derivative_from_thermodynamics,
    ml2_temperature_gradient_for_flux,
)
from wd_spectra.atmosphere import Atmosphere, _metal_line_opacity_sampling_grid
from wd_spectra.eos import (
    hummer_mihalas_hydrogen_lte,
    hummer_mihalas_hydrogen_helium_thermodynamics,
    ideal_hydrogen_lte,
    ideal_hydrogen_thermodynamics,
)
from wd_spectra.validation import read_svo_koester_ascii


def test_expanded_metal_opacity_grid_bounds_weak_line_sampling():
    centers = np.arange(1_000.0, 3_000.0)
    grid, wing_sampled = _metal_line_opacity_sampling_grid(
        centers, maximum_wing_sampled_lines=1_000
    )
    assert wing_sampled == 1_000
    assert grid.size <= 5_000 + 3_000
    for offset in (-0.05, 0.0, 0.05):
        assert np.all(np.isin(centers[1_000:] + offset, grid))
    for offset in (-2.0, -0.5, 0.0, 0.5, 2.0):
        assert np.all(np.isin(centers[:1_000] + offset, grid))


def test_bundled_balmer_table_shape_and_normalization():
    table = default_balmer_stark_table()
    assert len(table.lines) == 20
    hbeta = table[(2, 4)]
    assert hbeta.log_profile.shape == (17, 7, hbeta.log_alpha.size)

    electron_density = 1.0e16
    field_strength = 1.25e-9 * electron_density ** (2.0 / 3.0)
    positive_offset = np.power(10.0, hbeta.log_alpha) * field_strength
    artificial_center = 1.0e7  # Keeps the full tabulated negative wing positive.
    wavelength = artificial_center + np.concatenate((-positive_offset[::-1], [0.0], positive_offset))
    profile = hbeta.wavelength_profile(wavelength, artificial_center, 20_000.0, electron_density)
    integral = trapezoid(profile, wavelength)
    np.testing.assert_allclose(integral, 1.0, rtol=0.06)


def test_van_hoof_free_free_gaunt_factor_is_physical_and_broadcasts():
    wavelength = np.array([1_000.0, 5_000.0, 20_000.0])[:, np.newaxis]
    temperature = np.array([8_000.0, 12_000.0, 20_000.0])[np.newaxis, :]
    gaunt = hydrogen_free_free_gaunt_factor(wavelength, temperature)
    assert gaunt.shape == (3, 3)
    assert np.all(np.isfinite(gaunt))
    assert np.all((gaunt > 0.8) & (gaunt < 2.0))


def test_van_hoof_free_free_gaunt_factor_reproduces_table_nodes():
    # This point lies exactly on log10(gamma^2)=0.8 and log10(u)=0.6.
    temperature = HYDROGEN_IONIZATION_ENERGY / (BOLTZMANN * 10.0**0.8)
    wavelength = (
        PLANCK
        * LIGHT_SPEED
        / (BOLTZMANN * temperature * 10.0**0.6)
        * 1.0e8
    )
    gaunt = hydrogen_free_free_gaunt_factor(wavelength, temperature)
    # Independent lookup of row 83, column 35 in the original table.
    tokens = []
    table_path = (
        Path(__file__).parents[1]
        / "src/wd_spectra/data/gaunt/gauntff.dat"
    )
    for line in table_path.read_text(encoding="ascii").splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            tokens.extend(content.split())
    table = np.asarray(tokens[6 : 6 + 146 * 81], dtype=float).reshape(146, 81)
    np.testing.assert_allclose(gaunt, table[83, 34], rtol=2.0e-13)


def test_mihalas_bound_free_gaunt_factor_threshold_values():
    lyman_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    factors = np.asarray(
        [
            hydrogen_bound_free_gaunt_factor(level, lyman_frequency / level**2)
            for level in (1, 2, 3)
        ]
    )
    np.testing.assert_allclose(
        factors,
        [0.79727048, 0.87582246, 0.90783263],
        rtol=1.0e-8,
    )


def test_mihalas_bound_free_gaunt_factor_uses_unity_above_level_ten():
    frequency = np.array([1.0e14, 1.0e15, 1.0e16])
    np.testing.assert_array_equal(
        hydrogen_bound_free_gaunt_factor(11, frequency),
        np.ones(3),
    )


def test_mihalas_bound_free_gaunt_factor_does_not_extrapolate_into_xrays():
    frequency = LIGHT_SPEED / (np.array([25.0, 10.0]) * 1.0e-8)
    np.testing.assert_array_equal(
        hydrogen_bound_free_gaunt_factor(4, frequency),
        np.ones(2),
    )


def test_stark_profile_accepts_batched_wavelength_queries():
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    wavelength = center + np.linspace(-25.0, 25.0, 123).reshape(3, 41)
    batched = line.wavelength_profile(wavelength, center, 9_000.0, 3.0e16)
    scalar_rows = np.stack(
        [
            line.wavelength_profile(row, center, 9_000.0, 3.0e16)
            for row in wavelength
        ]
    )
    np.testing.assert_array_equal(batched, scalar_rows)


def test_stark_profile_combined_corner_interpolation_matches_literal_sum():
    line = default_balmer_stark_table()[(2, 4)]
    center = 4862.694
    wavelength = center + np.array([-800.0, -23.4, -0.1, 0.0, 7.8, 900.0])
    temperature = 13_700.0
    electron_density = 4.2e16
    calculated = line.wavelength_profile(
        wavelength, center, temperature, electron_density
    )

    log_ne = np.clip(
        np.log10(electron_density),
        line.log_electron_density[0],
        line.log_electron_density[-1],
    )
    log_t = np.clip(
        np.log10(temperature), line.log_temperature[0], line.log_temperature[-1]
    )
    ne_upper = int(np.searchsorted(line.log_electron_density, log_ne, side="right"))
    ne_lower = min(max(ne_upper - 1, 0), line.log_electron_density.size - 2)
    t_upper = int(np.searchsorted(line.log_temperature, log_t, side="right"))
    t_lower = min(max(t_upper - 1, 0), line.log_temperature.size - 2)
    ne_fraction = (log_ne - line.log_electron_density[ne_lower]) / (
        line.log_electron_density[ne_lower + 1]
        - line.log_electron_density[ne_lower]
    )
    t_fraction = (log_t - line.log_temperature[t_lower]) / (
        line.log_temperature[t_lower + 1] - line.log_temperature[t_lower]
    )
    field_strength = 1.25e-9 * electron_density ** (2.0 / 3.0)
    with np.errstate(divide="ignore"):
        query = np.log10(np.abs(wavelength - center) / field_strength)
    literal_log_profile = np.zeros_like(query)
    for ne_index, ne_weight in (
        (ne_lower, 1.0 - ne_fraction),
        (ne_lower + 1, ne_fraction),
    ):
        for t_index, t_weight in (
            (t_lower, 1.0 - t_fraction),
            (t_lower + 1, t_fraction),
        ):
            corner = line.log_profile[ne_index, t_index]
            values = np.interp(
                query,
                line.log_alpha,
                corner,
                left=corner[0],
                right=corner[-1],
            )
            beyond = query > line.log_alpha[-1]
            slope = (corner[-1] - corner[-2]) / (
                line.log_alpha[-1] - line.log_alpha[-2]
            )
            values[beyond] = corner[-1] + slope * (
                query[beyond] - line.log_alpha[-1]
            )
            literal_log_profile += ne_weight * t_weight * values
    expected = 10.0**literal_log_profile / field_strength
    np.testing.assert_allclose(calculated, expected, rtol=5.0e-15)


def test_bundled_lyman_table_and_atomic_data():
    table = default_lyman_stark_table()
    assert len(table.lines) == 20
    assert len(LYMAN_LINES) == 20
    np.testing.assert_allclose(
        [LYMAN_LINES[0].wavelength_vacuum_angstrom, LYMAN_LINES[1].wavelength_vacuum_angstrom],
        [1215.67, 1025.72],
        atol=0.01,
    )


def test_bundled_infrared_tables_and_atomic_data():
    paschen_table = default_paschen_stark_table()
    brackett_table = default_brackett_stark_table()
    assert len(paschen_table.lines) == len(PASCHEN_LINES) == 19
    assert len(brackett_table.lines) == len(BRACKETT_LINES) == 10
    assert set(paschen_table.lines) == {
        (line.lower_level, line.upper_level) for line in PASCHEN_LINES
    }
    assert set(brackett_table.lines) == {
        (line.lower_level, line.upper_level) for line in BRACKETT_LINES
    }
    np.testing.assert_allclose(
        [line.wavelength_vacuum_angstrom for line in PASCHEN_LINES[:3]],
        [18756.0715, 12821.5333, 10941.0417],
        rtol=0.0,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        [line.absorption_oscillator_strength for line in PASCHEN_LINES[:3]],
        [0.84209636, 0.15058408, 0.05584025],
        rtol=5.0e-8,
    )
    np.testing.assert_allclose(
        [line.absorption_oscillator_strength for line in BRACKETT_LINES[:3]],
        [1.03773626, 0.17925237, 0.06548599],
        rtol=5.0e-8,
    )
    assert PASCHEN_LINES[-1].upper_level == 22
    assert PASCHEN_LINES[-1].absorption_oscillator_strength == pytest.approx(
        0.000528750858802903
    )
    assert BRACKETT_LINES[-1].upper_level == 14
    assert BRACKETT_LINES[-1].absorption_oscillator_strength == pytest.approx(
        0.00337542785213278
    )
    np.testing.assert_allclose(
        [LYMAN_LINES[0].absorption_oscillator_strength, LYMAN_LINES[1].absorption_oscillator_strength],
        [0.4161967, 0.0791016],
        rtol=2.0e-6,
    )


def test_excited_hydrogen_population_increases_over_da_range():
    population = hydrogen_level_population(1.0e17, [8_000.0, 12_000.0, 20_000.0], 2)
    assert np.all(np.diff(population) > 0.0)


def test_neutral_hydrogen_broadening_covers_low_and_high_balmer_members():
    atmosphere = hydrogen_continuum_atmosphere(8_000.0, 8.0, n_depth=30)
    barklem_width = neutral_hydrogen_self_broadening_hwhm(
        atmosphere, BALMER_LINES[0]
    )
    resonance_width = neutral_hydrogen_self_broadening_hwhm(
        atmosphere, BALMER_LINES[3]
    )
    assert np.all(barklem_width > 0.0)
    assert np.all(resonance_width > 0.0)
    assert barklem_width[-1] > barklem_width[0]
    assert resonance_width[-1] > resonance_width[0]
    assert np.all(np.isfinite(resonance_width))


def test_ali_griem_control_is_available_for_low_balmer_members():
    atmosphere = hydrogen_continuum_atmosphere(8_000.0, 8.0, n_depth=30)
    barklem_width = neutral_hydrogen_self_broadening_hwhm(
        atmosphere, BALMER_LINES[0]
    )
    ali_griem_width = neutral_hydrogen_self_broadening_hwhm(
        atmosphere, BALMER_LINES[0], prescription="ali-griem"
    )

    assert np.all(ali_griem_width > 0.0)
    assert np.max(np.abs(ali_griem_width - barklem_width)) > 0.0


def test_allard_2008_halpha_width_reproduces_published_temperature_table():
    atmosphere = hydrogen_continuum_atmosphere(7_000.0, 8.0, n_depth=30)
    width = neutral_hydrogen_self_broadening_hwhm(
        atmosphere, BALMER_LINES[0], prescription="allard-2008"
    )
    wavelength_cm = BALMER_LINES[0].wavelength_vacuum_angstrom * 1.0e-8
    angular_hwhm_per_perturber = (
        width
        * LIGHT_SPEED
        / wavelength_cm**2
        * 2.0
        * np.pi
        / atmosphere.neutral_h_density
        * 1.0e-8
    )
    published_temperature = np.arange(3_000.0, 12_000.1, 1_000.0)
    published_hwhm = 1.0e-8 * np.asarray(
        [4.03, 4.14, 4.21, 4.26, 4.30, 4.35, 4.40, 4.43, 4.45, 4.49]
    )
    expected = np.interp(
        np.clip(atmosphere.temperature, 3_000.0, 12_000.0),
        published_temperature,
        published_hwhm,
    )
    np.testing.assert_allclose(angular_hwhm_per_perturber, expected, rtol=2.0e-15)

    # Allard (2008) supplies Halpha only; the control deliberately retains
    # the Barklem calculation for Hbeta and Hgamma.
    np.testing.assert_array_equal(
        neutral_hydrogen_self_broadening_hwhm(
            atmosphere, BALMER_LINES[1], prescription="allard-2008"
        ),
        neutral_hydrogen_self_broadening_hwhm(
            atmosphere, BALMER_LINES[1], prescription="barklem"
        ),
    )


def test_neutral_helium_balmer_broadening_uses_mixed_helium_density():
    mixed = gray_hydrogen_helium_atmosphere(
        10_000.0, 8.0, -2.0, n_depth=12
    )
    helium_width = neutral_helium_balmer_broadening_hwhm(
        mixed, BALMER_LINES[0]
    )
    hydrogen_width = neutral_hydrogen_self_broadening_hwhm(
        mixed, BALMER_LINES[0]
    )
    assert np.all(np.isfinite(helium_width))
    assert np.all(helium_width > 0.0)
    assert np.median(helium_width / hydrogen_width) > 10.0

    pure_hydrogen = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=8)
    np.testing.assert_array_equal(
        neutral_helium_balmer_broadening_hwhm(
            pure_hydrogen, BALMER_LINES[0]
        ),
        np.zeros(pure_hydrogen.n_depth),
    )


def test_lorentz_convolution_preserves_stark_line_strength():
    line = default_balmer_stark_table()[(2, 3)]
    center = 10_000.0
    wavelength = center + np.linspace(-2_000.0, 2_000.0, 20_001)
    stark = line.wavelength_profile(wavelength, center, 8_000.0, 1.0e16)
    convolved = _lorentz_convolved_stark_profile(
        line, wavelength, center, 8_000.0, 1.0e16, 0.7
    )
    np.testing.assert_allclose(
        trapezoid(convolved, wavelength),
        trapezoid(stark, wavelength),
        rtol=3.0e-3,
    )
    assert convolved[wavelength.size // 2] < stark[wavelength.size // 2]


def test_truncated_lorentz_convolution_preserves_line_strength_and_far_wing():
    line = default_balmer_stark_table()[(2, 5)]
    center = 10_000.0
    wavelength = center + np.linspace(-2_000.0, 2_000.0, 20_001)
    stark = line.wavelength_profile(wavelength, center, 8_000.0, 1.0e16)
    full = _lorentz_convolved_stark_profile(
        line, wavelength, center, 8_000.0, 1.0e16, 0.7
    )
    truncated = _lorentz_convolved_stark_profile(
        line,
        wavelength,
        center,
        8_000.0,
        1.0e16,
        0.7,
        maximum_impact_shift_angstrom=7.7,
    )
    np.testing.assert_allclose(
        trapezoid(truncated, wavelength),
        trapezoid(stark, wavelength),
        rtol=3.0e-3,
    )
    far = np.abs(wavelength - center) > 100.0
    assert np.mean(np.abs(truncated[far] - stark[far])) < np.mean(
        np.abs(full[far] - stark[far])
    )


def test_impact_limited_convolution_has_no_translated_core_at_cutoff():
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    offset = np.arange(-35.2, 35.201, 0.005)
    profile = _lorentz_convolved_stark_profile(
        line,
        center + offset,
        center,
        8_000.0,
        1.0e16,
        0.75,
        maximum_impact_shift_angstrom=35.0,
        quadrature_order=32,
    )
    for cutoff in (-35.0, 35.0):
        index = int(np.argmin(np.abs(offset - cutoff)))
        left_slope = (profile[index] - profile[index - 1]) / 0.005
        right_slope = (profile[index + 1] - profile[index]) / 0.005
        assert abs(right_slope - left_slope) / profile[index] < 5.0e-3


def test_lorentz_convolution_is_smooth_on_irregular_halpha_sampling():
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    regular = np.linspace(-40.0, 40.0, 801)
    irregular = np.sort(
        np.unique(
            np.concatenate(
                (
                    regular[::7],
                    np.linspace(-1.0, 1.0, 101),
                    np.array([-25.7, -14.6, -6.7, 10.6, 15.3]),
                )
            )
        )
    )
    profile = _lorentz_convolved_stark_profile(
        line, center + irregular, center, 8_000.0, 1.0e16, 0.75
    )
    dense = _lorentz_convolved_stark_profile(
        line, center + regular, center, 8_000.0, 1.0e16, 0.75
    )
    np.testing.assert_allclose(
        profile,
        np.interp(irregular, regular, dense),
        rtol=0.015,
        atol=2.0e-4 * np.max(dense),
    )


def test_structure_order_lorentz_quadrature_matches_full_synthesis_order():
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    wavelength = center + np.unique(
        np.concatenate(
            (np.arange(-400.0, 400.1, 10.0), np.arange(-5.0, 5.01, 0.1))
        )
    )
    full = _lorentz_convolved_stark_profile(
        line,
        wavelength,
        center,
        8_000.0,
        1.0e16,
        0.75,
        quadrature_order=128,
    )
    structure = _lorentz_convolved_stark_profile(
        line,
        wavelength,
        center,
        8_000.0,
        1.0e16,
        0.75,
        quadrature_order=32,
    )
    assert np.max(np.abs(structure - full)) < 5.0e-4 * np.max(full)
    np.testing.assert_allclose(
        trapezoid(structure, wavelength),
        trapezoid(full, wavelength),
        rtol=8.0e-3,
    )


@pytest.mark.skipif(
    opacity_module._rt is None
    or not hasattr(
        opacity_module._rt, "hydrogen_stark_lorentz_convolution"
    ),
    reason="optional compiled hydrogen convolution is not built",
)
@pytest.mark.parametrize(
    "maximum_shift,truncation_closure",
    [(None, "renormalize"), (35.0, "renormalize"), (35.0, "stark-core")],
)
def test_compiled_hydrogen_convolution_matches_python_reference(
    monkeypatch, maximum_shift, truncation_closure
):
    line = default_balmer_stark_table()[(2, 3)]
    center = 6564.636
    wavelength = center + np.unique(
        np.concatenate(
            (
                np.linspace(-1_000.0, 1_000.0, 1001),
                np.linspace(-3.0, 3.0, 201),
            )
        )
    )
    options = dict(
        maximum_impact_shift_angstrom=maximum_shift,
        quadrature_order=32,
        truncation_closure=truncation_closure,
    )
    compiled = _lorentz_convolved_stark_profile(
        line, wavelength, center, 8732.1, 4.2e16, 0.731, **options
    )
    monkeypatch.setattr(opacity_module, "_rt", None)
    reference = _lorentz_convolved_stark_profile(
        line, wavelength, center, 8732.1, 4.2e16, 0.731, **options
    )
    np.testing.assert_allclose(
        compiled,
        reference,
        rtol=5.0e-12,
        atol=5.0e-15 * float(np.max(reference)),
    )


@pytest.mark.skipif(
    opacity_module._rt is None
    or not hasattr(
        opacity_module._rt, "hydrogen_stark_lorentz_convolution"
    ),
    reason="optional compiled hydrogen convolution is not built",
)
def test_compiled_hydrogen_nonuniform_endpoint_uses_last_profile_node():
    """An exact nonuniform-grid endpoint must not form an OOB bracket."""
    log_alpha = np.array(
        [-2.0, -1.5, -0.7, -0.2, 0.3, 0.9, 1.4, 2.0],
        dtype=np.float64,
    )
    log_profile = np.linspace(0.0, -3.0, log_alpha.size, dtype=np.float64)
    # Keep a canary immediately after each view: the endpoint path must read
    # the final node, not the one-past-the-end element.
    log_alpha_buffer = np.empty(log_alpha.size + 1, dtype=np.float64)
    log_profile_buffer = np.empty(log_profile.size + 1, dtype=np.float64)
    log_alpha_buffer[:-1] = log_alpha
    log_profile_buffer[:-1] = log_profile
    log_alpha_buffer[-1] = np.nan
    log_profile_buffer[-1] = np.nan
    field_strength = 1.0
    detuning = np.array([10.0 ** log_alpha[-1]], dtype=np.float64)
    calculated = opacity_module._rt.hydrogen_stark_lorentz_convolution(
        detuning,
        log_alpha_buffer[:-1],
        log_profile_buffer[:-1],
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        field_strength,
        0.5,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
    )
    expected = np.array([10.0 ** log_profile[-1] / field_strength])
    np.testing.assert_allclose(calculated, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    opacity_module._rt is None
    or not hasattr(
        opacity_module._rt, "hydrogen_stark_lorentz_convolution"
    ),
    reason="optional compiled hydrogen convolution is not built",
)
def test_parallel_hydrogen_profiles_match_serial_result(monkeypatch):
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=6)
    line = BALMER_LINES[0]
    wavelength = np.linspace(
        line.wavelength_vacuum_angstrom - 80.0,
        line.wavelength_vacuum_angstrom + 80.0,
        321,
    )
    options = dict(lines=(line,), self_broadening_quadrature_order=16)
    monkeypatch.setenv("OPENWD_NUM_THREADS", "1")
    serial = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, **options
    )
    monkeypatch.setenv("OPENWD_NUM_THREADS", "4")
    parallel = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, **options
    )
    np.testing.assert_array_equal(parallel, serial)


def test_balmer_structure_support_omits_only_optically_thin_wings():
    atmosphere = gray_hydrogen_helium_atmosphere(
        12_000.0, 8.0, -5.0, n_depth=10
    )
    wavelength = np.linspace(2_000.0, 11_000.0, 901)
    options = dict(
        lines=(BALMER_LINES[0],),
        self_broadening_quadrature_order=16,
    )
    full = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, **options
    )
    threshold = 1.0e-3
    supported = balmer_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        profile_edge_optical_depth=threshold,
        **options,
    )
    retained = np.any(supported > 0.0, axis=1)
    assert np.any(retained)
    assert np.any(~retained)
    np.testing.assert_allclose(
        supported[retained], full[retained], rtol=2.0e-13, atol=0.0
    )
    relevant = np.flatnonzero(atmosphere.rosseland_optical_depth <= 2.0)
    stop = int(relevant[-1]) + 1 if relevant.size else 1
    omitted_optical_depth = (
        full[~retained, 0] * atmosphere.column_mass[0]
        + np.sum(
            0.5
            * (full[~retained, 1:stop] + full[~retained, : stop - 1])
            * np.diff(atmosphere.column_mass[:stop])[np.newaxis, :],
            axis=1,
        )
    )
    assert np.max(omitted_optical_depth) < 1.1 * threshold


def test_8000k_halpha_flux_has_no_irregular_grid_spikes():
    reference = read_svo_koester_ascii(
        Path(".cache/koester/koester_t08000_g8.00.txt")
    ) if Path(".cache/koester/koester_t08000_g8.00.txt").exists() else None
    if reference is None:
        pytest.skip("cached Koester validation spectrum is unavailable")
    center = BALMER_LINES[0].wavelength_vacuum_angstrom
    selected = np.abs(reference.wavelength_angstrom - center) <= 400.0
    atmosphere = hydrogen_continuum_atmosphere(8_000.0, 8.0, n_depth=30)
    spectrum = synthesize_hydrogen_spectrum(
        atmosphere, reference.wavelength_angstrom[selected]
    )
    wavelength = spectrum.wavelength_angstrom
    flux = spectrum.surface_flux_lambda
    edge = np.abs(wavelength - center) >= 300.0
    continuum = np.polyval(np.polyfit(wavelength[edge], flux[edge], 1), wavelength)
    normalized = flux / continuum
    offset = wavelength - center
    slope = np.diff(normalized) / np.diff(offset)
    curvature = np.diff(slope) / (
        0.5 * (np.diff(offset[:-1]) + np.diff(offset[1:]))
    )
    away_from_core = np.abs(offset[1:-1]) > 2.0
    assert np.max(np.abs(curvature[away_from_core])) < 0.01


def test_balmer_bound_bound_opacity_uses_upper_level_survival():
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=16)
    line = BALMER_LINES[8]
    wavelength = np.linspace(
        line.wavelength_vacuum_angstrom - 5.0,
        line.wavelength_vacuum_angstrom + 5.0,
        101,
    )
    nonideal = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, lines=(line,), include_self_broadening=False
    )
    ideal_lower_population = hydrogen_level_population(
        atmosphere.neutral_h_density, atmosphere.temperature, 2
    )
    nonideal_lower_population = hydrogen_level_population(
        atmosphere.neutral_h_density,
        atmosphere.temperature,
        2,
        electron_density=atmosphere.electron_density,
    )
    assert np.all(nonideal_lower_population <= ideal_lower_population)
    assert np.all(nonideal >= 0.0)
    assert np.any(nonideal > 0.0)


def test_balmer_opacity_can_leave_barklem_impact_kernel_uncapped():
    atmosphere = gray_hydrogen_atmosphere(8_000.0, 8.0, n_depth=12)
    line = BALMER_LINES[0]
    wavelength = np.linspace(
        line.wavelength_vacuum_angstrom - 60.0,
        line.wavelength_vacuum_angstrom + 60.0,
        121,
    )
    tabulated_range = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, lines=(line,)
    )
    uncapped = balmer_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        lines=(line,),
        self_broadening_impact_validity_fraction=None,
    )

    assert np.all(np.isfinite(uncapped))
    assert np.all(uncapped >= 0.0)
    assert np.max(np.abs(uncapped - tabulated_range)) > 0.0


def test_balmer_impact_tail_can_close_back_into_unshifted_stark_core():
    atmosphere = gray_hydrogen_atmosphere(8_000.0, 8.0, n_depth=12)
    line = BALMER_LINES[0]
    wavelength = np.linspace(
        line.wavelength_vacuum_angstrom - 60.0,
        line.wavelength_vacuum_angstrom + 60.0,
        121,
    )
    renormalized = balmer_mass_absorption_coefficient(
        atmosphere, wavelength, lines=(line,)
    )
    conservative = balmer_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        lines=(line,),
        self_broadening_truncation_closure="stark-core",
    )
    assert np.all(np.isfinite(conservative))
    assert np.all(conservative >= 0.0)
    assert np.max(np.abs(conservative - renormalized)) > 0.0


def test_constant_mass_opacity_recovers_kappa_times_column_mass():
    mass = np.geomspace(1.0e-7, 1.0e3, 100)
    opacity = np.full((3, mass.size), 0.2)
    tau = optical_depth_from_mass_opacity(mass, opacity)
    expected = np.broadcast_to(0.2 * mass[np.newaxis, :], tau.shape)
    np.testing.assert_allclose(tau, expected, rtol=2e-15)


def test_balmer_opacity_and_spectrum_contain_four_lines():
    assert len(BALMER_LINES) == 20
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=36)
    wavelength = np.linspace(3_900.0, 6_800.0, 2_901)
    opacity = balmer_mass_absorption_coefficient(atmosphere, wavelength)
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    assert np.all(opacity >= 0.0)

    gray = synthesize_gray_spectrum(atmosphere, wavelength)
    balmer = synthesize_balmer_spectrum(atmosphere, wavelength)
    for center in (4102.899, 4341.692, 4862.691, 6564.632):
        index = int(np.argmin(np.abs(wavelength - center)))
        assert balmer.surface_flux_lambda[index] < 0.8 * gray.surface_flux_lambda[index]


def test_lyman_opacity_and_spectrum_contain_first_three_lines():
    atmosphere = gray_hydrogen_atmosphere(20_000.0, 8.0, n_depth=36)
    wavelength = np.linspace(900.0, 1250.0, 701)
    opacity = lyman_mass_absorption_coefficient(atmosphere, wavelength)
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    assert np.all(opacity >= 0.0)
    balmer_only = synthesize_balmer_spectrum(
        atmosphere, wavelength, continuum_opacity=0.1
    )
    hydrogen = synthesize_hydrogen_spectrum(
        atmosphere, wavelength, continuum_opacity=0.1
    )
    for center in (1215.67, 1025.72, 972.54):
        index = int(np.argmin(np.abs(wavelength - center)))
        assert hydrogen.surface_flux_lambda[index] < 0.98 * balmer_only.surface_flux_lambda[index]


def test_infrared_opacity_and_spectrum_contain_paschen_and_brackett_lines():
    atmosphere = gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=30)
    centers = np.asarray(
        [
            PASCHEN_LINES[0].wavelength_vacuum_angstrom,
            PASCHEN_LINES[1].wavelength_vacuum_angstrom,
            BRACKETT_LINES[0].wavelength_vacuum_angstrom,
            BRACKETT_LINES[1].wavelength_vacuum_angstrom,
        ]
    )
    wavelength = np.unique(
        np.concatenate(
            (
                np.linspace(8_000.0, 50_000.0, 180),
                *(center + np.asarray((-2.0, 0.0, 2.0)) for center in centers),
            )
        )
    )
    paschen = paschen_mass_absorption_coefficient(atmosphere, wavelength)
    brackett = brackett_mass_absorption_coefficient(atmosphere, wavelength)
    assert paschen.shape == brackett.shape == (
        wavelength.size,
        atmosphere.n_depth,
    )
    assert np.all(paschen >= 0.0)
    assert np.all(brackett >= 0.0)
    assert np.any(paschen > 0.0)
    assert np.any(brackett > 0.0)

    without_infrared = synthesize_balmer_spectrum(
        atmosphere, wavelength, continuum_opacity=0.1
    )
    with_infrared = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        continuum_opacity=0.1,
        include_paschen=True,
        include_brackett=True,
    )
    for center in centers:
        index = int(np.argmin(np.abs(wavelength - center)))
        assert (
            with_infrared.surface_flux_lambda[index]
            < 0.995 * without_infrared.surface_flux_lambda[index]
        )
    assert "Paalpha-Pa22" in with_infrared.metadata["lines"]
    assert "Bralpha-Br14" in with_infrared.metadata["lines"]


def test_infrared_stark_wing_windows_have_smooth_zero_edges():
    atmosphere = gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=8)
    wavelength = np.asarray(
        [
            7_499.0,
            7_500.0,
            7_501.0,
            24_999.0,
            25_000.0,
            25_001.0,
            13_499.0,
            13_500.0,
            13_501.0,
            49_999.0,
            50_000.0,
            50_001.0,
        ]
    )
    paschen = paschen_mass_absorption_coefficient(atmosphere, wavelength)
    brackett = brackett_mass_absorption_coefficient(atmosphere, wavelength)

    assert np.all(paschen[[0, 1, 4, 5]] == 0.0)
    assert np.all(brackett[[6, 7, 10, 11]] == 0.0)
    assert np.all(paschen[[2, 3]] >= 0.0)
    assert np.all(brackett[[8, 9]] >= 0.0)


def test_first_three_lyman_lines_follow_table_ideal_probability_convention():
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=16)
    wavelength = np.linspace(900.0, 1250.0, 351)
    opacity = lyman_mass_absorption_coefficient(
        atmosphere, wavelength, lines=LYMAN_LINES[:3]
    )

    # Replacing the cached upper-level probabilities must not change these
    # lines: the published table intentionally uses optical w=1 for Lyalpha
    # through Lygamma.  Ground populations and all other EOS quantities are
    # held fixed in this controlled copy.
    state = atmosphere.hydrogen_lte_state
    assert state is not None
    occupation = state.level_occupation_probability.copy()
    occupation[:, 1:4] *= 1.0e-4
    altered_state = type(state)(
        mass_density=state.mass_density,
        hydrogen_nuclei_density=state.hydrogen_nuclei_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        ionization_fraction=state.ionization_fraction,
        internal_partition_function=state.internal_partition_function,
        level_occupation_probability=occupation,
        level_population_density=state.level_population_density,
        microfield_model=state.microfield_model,
    )
    altered_atmosphere = Atmosphere(
        effective_temperature=atmosphere.effective_temperature,
        logg=atmosphere.logg,
        rosseland_optical_depth=atmosphere.rosseland_optical_depth,
        column_mass=atmosphere.column_mass,
        temperature=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        mass_density=atmosphere.mass_density,
        neutral_h_density=atmosphere.neutral_h_density,
        proton_density=atmosphere.proton_density,
        electron_density=atmosphere.electron_density,
        metadata=atmosphere.metadata,
        hydrogen_lte_state=altered_state,
    )
    np.testing.assert_array_equal(
        lyman_mass_absorption_coefficient(
            altered_atmosphere, wavelength, lines=LYMAN_LINES[:3]
        ),
        opacity,
    )


def test_neutral_hydrogen_lyman_alpha_wing_is_red_and_density_dependent():
    atmosphere = gray_hydrogen_atmosphere(8_000.0, 8.0, n_depth=16)
    wavelength = np.array([800.0, 900.0, 1200.0, 1300.0, 1600.0, 3000.0, 6100.0])
    opacity = lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
        atmosphere, wavelength
    )
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    np.testing.assert_array_equal(opacity[[0, 1, 2, -1]], 0.0)
    assert np.all(opacity[3:-1] > 0.0)
    assert np.all(opacity[3] > opacity[4])
    assert np.all(opacity[4] > opacity[5])


def test_molecular_hydrogen_adds_the_published_lyman_alpha_red_wing():
    atomic = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=16, include_molecules=False
    )
    molecular = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=16, include_molecules=True
    )
    wavelength = np.array([1100.0, 2000.0, 3000.0, 5000.0, 6100.0])

    atomic_opacity = (
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
            atomic, wavelength
        )
    )
    molecular_opacity = (
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
            molecular, wavelength
        )
    )

    np.testing.assert_array_equal(molecular_opacity[[0, -1]], 0.0)
    assert np.all(molecular_opacity[1:-1] > atomic_opacity[1:-1])


def test_molecular_lyman_alpha_far_wing_fades_out_of_the_line_core():
    molecular = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=12, include_molecules=True
    )
    molecular_state = molecular.hydrogen_lte_state
    assert molecular_state is not None
    assert molecular_state.molecular_hydrogen_density is not None
    atomic_control = replace(
        molecular,
        hydrogen_lte_state=replace(
            molecular_state,
            molecular_hydrogen_density=np.zeros_like(
                molecular_state.molecular_hydrogen_density
            ),
        ),
    )
    wavelength = np.array([1215.67, 1216.0, 1230.0, 1250.0])
    atomic_opacity = lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
        atomic_control, wavelength
    )
    molecular_opacity = (
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
            molecular, wavelength
        )
    )
    molecular_contribution = molecular_opacity - atomic_opacity

    np.testing.assert_array_equal(molecular_contribution[0], 0.0)
    assert np.all(molecular_contribution[1] < molecular_contribution[2])
    assert np.all(molecular_contribution[2] < molecular_contribution[3])


def test_molecular_lyman_alpha_far_wing_uses_local_temperature():
    wavelength_nm = np.array([130.0, 300.0, 600.0])
    correction = _h_h2_lyman_alpha_log_temperature_correction(
        wavelength_nm, np.array([3000.0, 5000.0, 6000.0, 8000.0])
    )

    np.testing.assert_allclose(correction[:, 2:], 0.0)
    assert np.all(correction[:, 0] < correction[:, 1])
    assert np.all(correction[:, 1] < 0.0)
    # The vector digitization gives -43.3856 at 3000 K and -41.5887
    # at 6000 K for 300 nm.  The 5000-K curve follows the expected
    # inverse-temperature interpolation (one fifth of the full correction).
    np.testing.assert_allclose(correction[1, 0], -1.7969, atol=2.0e-4)
    np.testing.assert_allclose(correction[1, 1], -0.35938, atol=2.0e-4)


def test_neutral_lyman_alpha_wing_complements_finite_allard_table(tmp_path):
    from wd_spectra.quasimolecular import (
        AllardNeutralLymanAlphaTable,
        read_allard_hydrogen_line_profile,
    )

    table_path = tmp_path / "laquasi.dat"
    table_path.write_text(
        "3 17.0 18.0 2.0D-2 3.0D-2\n"
        "1200.0 1 2 3 4 5\n"
        "1500.0 1 2 3 4 5\n"
        "1800.0 1 2 3 4 5\n",
        encoding="utf-8",
    )
    table = AllardNeutralLymanAlphaTable(
        (
            read_allard_hydrogen_line_profile(
                table_path, lower_level=1, upper_level=2
            ),
        )
    )
    atmosphere = gray_hydrogen_atmosphere(6_000.0, 8.0, n_depth=12)
    wavelength = np.array([1500.0, 2000.0])
    ordinary = lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
        atmosphere, wavelength
    )
    complementary = (
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
            atmosphere, wavelength, allard_table=table
        )
    )

    # Native SYNSPEC continues the Allard Ly-alpha table redward with a
    # |delta lambda|^-5/2 tail, so the separate approximate H-H wing remains
    # suppressed beyond the final tabulated row as well as within the table.
    assert np.all(ordinary[1] > 0.0)
    np.testing.assert_array_equal(complementary, 0.0)


def test_charged_particle_occupation_probability_dissolves_high_levels():
    level = np.array([1.0, 3.0, 6.0, 10.0])
    probability = charged_particle_hydrogen_occupation_probability(1.0e17, level)
    assert np.all(np.diff(probability) < 0.0)
    assert probability[0] > 0.999
    assert probability[-1] < 0.01
    denser = charged_particle_hydrogen_occupation_probability(1.0e18, level)
    assert np.all(denser <= probability)


def test_ground_state_photoionization_uses_stobbe_gaunt_factor():
    threshold = 13.598434599702 * 1.602176634e-12 / 6.62607015e-27
    frequency = threshold * np.array([0.99, 1.0, 1.1, 2.0])
    cross_section = hydrogen_ground_state_photoionization_cross_section(frequency)
    assert cross_section[0] == 0.0
    np.testing.assert_allclose(cross_section[1], 6.30e-18, rtol=3.0e-3)
    kramers = 6.30e-18 * np.array([1.1, 2.0]) ** -3
    assert np.all(cross_section[2:] > kramers)


def test_series_pseudocontinuum_covers_lyman_balmer_and_paschen_intervals():
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=16)
    for lower_level, wavelengths in (
        (1, [900.0, 920.0, 2000.0]),
        (2, [3600.0, 3700.0, 7000.0]),
        (3, [8000.0, 8300.0, 20_000.0]),
        (4, [14_000.0, 15_000.0, 42_000.0]),
    ):
        opacity = (
            hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                atmosphere,
                np.asarray(wavelengths),
                lower_levels=(lower_level,),
            )
        )
        np.testing.assert_array_equal(opacity[[0, 2]], 0.0)
        assert np.all(opacity[1] > 0.0)


def test_series_pseudocontinuum_keeps_only_neutral_tail_after_third_member():
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 9.0, n_depth=16)
    for lower_level, wavelengths in (
        (2, [4300.0, 4400.0, 6600.0]),
        (3, [10_900.0, 11_000.0, 19_000.0]),
        (4, [21_600.0, 21_750.0, 41_000.0]),
    ):
        opacity = (
            hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                atmosphere,
                np.asarray(wavelengths),
                lower_levels=(lower_level,),
            )
        )
        assert np.all(opacity[:2] > 0.0)
        np.testing.assert_array_equal(opacity[2], 0.0)


def test_continuum_atmosphere_has_hydrostatic_pressure_and_positive_opacity():
    atmosphere = hydrogen_continuum_atmosphere(12_000.0, 8.0, n_depth=24)
    np.testing.assert_allclose(
        atmosphere.gas_pressure,
        atmosphere.gravity * atmosphere.column_mass,
        rtol=1e-14,
    )
    rosseland = atmosphere.metadata["rosseland_opacity_cm2_g"]
    assert np.all(np.asarray(rosseland) > 0.0)
    opacity = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere, np.array([1500.0, 4000.0, 8000.0, 16_000.0])
    )
    assert opacity.shape == (4, atmosphere.n_depth)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity > 0.0)


def test_molecular_uv_opacity_is_present_only_with_molecular_eos():
    atomic = gray_hydrogen_atmosphere(5_000.0, 8.0, n_depth=12)
    molecular = gray_hydrogen_atmosphere(
        5_000.0, 8.0, n_depth=12, include_molecules=True
    )
    wavelength = np.array([950.0, 1100.0, 1500.0, 3001.0, 4000.0])
    atomic_opacity = molecular_hydrogen_mass_absorption_coefficient(
        atomic, wavelength
    )
    molecular_opacity = molecular_hydrogen_mass_absorption_coefficient(
        molecular, wavelength
    )
    np.testing.assert_array_equal(atomic_opacity, 0.0)
    assert np.all(molecular_opacity[:3] >= 0.0)
    assert np.any(molecular_opacity[:3] > 0.0)
    np.testing.assert_array_equal(molecular_opacity[3:], 0.0)


def test_h2_h2_cia_uses_binary_molecular_density_scaling(tmp_path):
    from wd_spectra.molecules import read_borysow_h2_h2_cia_table

    table_path = tmp_path / "h2-h2.dat"
    table_path.write_text(
        "@TEMPERATURES\n"
        "5000 7000\n"
        "@DATA\n"
        "5000 1e-7 2e-7\n"
        "10000 2e-7 4e-7\n",
        encoding="utf-8",
    )
    table = read_borysow_h2_h2_cia_table(table_path)
    atomic = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=16, include_molecules=False
    )
    molecular = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=16, include_molecules=True
    )
    wavelength = np.array([10_000.0, 20_000.0, 30_000.0])

    atomic_opacity = h2_h2_cia_mass_absorption_coefficient(
        atomic, wavelength, table
    )
    molecular_opacity = h2_h2_cia_mass_absorption_coefficient(
        molecular, wavelength, table
    )

    np.testing.assert_array_equal(atomic_opacity, 0.0)
    assert np.all(molecular_opacity[:2] > 0.0)
    np.testing.assert_array_equal(molecular_opacity[2], 0.0)


def test_atomic_hydrogen_rayleigh_scattering_is_uv_weighted_and_redward_only():
    wavelength = np.array([1000.0, 1500.0, 5000.0, 10_000.0])
    cross_section = hydrogen_rayleigh_scattering_cross_section(wavelength)
    assert cross_section[0] == 0.0
    assert np.all(cross_section[1:] > 0.0)
    assert cross_section[1] > cross_section[2] > cross_section[3]
    np.testing.assert_allclose(
        cross_section[2] / cross_section[3], 16.0, rtol=0.12
    )

    atmosphere = gray_hydrogen_atmosphere(8_000.0, 8.0, n_depth=12)
    opacity = hydrogen_rayleigh_scattering_mass_coefficient(
        atmosphere, wavelength
    )
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    assert np.all(opacity >= 0.0)


def test_atomic_hydrogen_rayleigh_hands_lyalpha_resonance_to_line_profile():
    center = 1215.670
    wavelength = center + np.array([0.60, 0.70, 2.0, 5.0, 20.0])
    cross_section = hydrogen_rayleigh_scattering_cross_section(wavelength)

    # The Rohrmann--Vera Rueda fit starts near +0.65 A.  Its isolated-atom
    # divergence must not appear as a discontinuous opacity spike beside the
    # separately calculated pressure-broadened Ly-alpha line.
    assert cross_section[0] == 0.0
    assert 0.0 < cross_section[1] < cross_section[2] < cross_section[3]
    assert cross_section[1] < 0.03 * cross_section[4]


def test_molecular_hydrogen_rayleigh_scattering_matches_long_wave_fit():
    wavelength = np.array([1000.0, 1216.0, 3500.0, 7000.0])
    cross_section = molecular_hydrogen_rayleigh_scattering_cross_section(
        wavelength
    )
    assert cross_section[0] == 0.0
    np.testing.assert_allclose(cross_section[1], 1.105e-24, rtol=0.002)
    np.testing.assert_allclose(cross_section[2], 6.192e-27, rtol=0.002)
    assert cross_section[2] > cross_section[3]

    atomic = gray_hydrogen_atmosphere(6_000.0, 8.0, n_depth=12)
    molecular = gray_hydrogen_atmosphere(
        6_000.0, 8.0, n_depth=12, include_molecules=True
    )
    atomic_opacity = hydrogen_rayleigh_scattering_mass_coefficient(
        atomic, wavelength
    )
    molecular_opacity = hydrogen_rayleigh_scattering_mass_coefficient(
        molecular, wavelength
    )
    assert np.any(molecular_opacity[1:] > atomic_opacity[1:])


def test_hminus_fit_matches_published_6000k_reference_values():
    wavelength_um = np.array([0.5, 0.85, 1.2, 1.65, 2.0])
    bound_free, free_free = _hminus_cross_section_per_electron_pressure(
        wavelength_um, np.full_like(wavelength_um, 6000.0)
    )
    # Convert cm^4/dyne to the customary coefficient multiplying n_H*n_e.
    scale = BOLTZMANN * 6000.0 / 1.0e-38
    np.testing.assert_allclose(
        bound_free * scale,
        [2.78, 3.60, 2.40, 0.0, 0.0],
        atol=0.015,
    )
    np.testing.assert_allclose(
        free_free * scale,
        [0.133, 0.356, 0.684, 1.262, 1.835],
        atol=0.002,
    )


def test_short_radiative_equilibrium_relaxation_preserves_hydrostatic_balance():
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=18,
        max_iterations=2,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        n_angle=2,
    )
    np.testing.assert_allclose(
        atmosphere.gas_pressure,
        atmosphere.gravity * atmosphere.column_mass,
        rtol=2e-14,
    )
    segments = atmosphere.metadata["nonlinear_solver_segments"]
    assert segments
    assert all(len(segment["iteration_history"]) <= 2 for segment in segments)
    assert atmosphere.metadata["radiative_equilibrium_converged"] == atmosphere.metadata["equilibrium_certificate"]["verified"]


def test_adaptive_newton_converges_cool_convective_flux_control():
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        6_000.0,
        8.0,
        n_depth=12,
        max_iterations=30,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=0.7,
        n_angle=2,
        structure_solver="adaptive-newton",
    )

    # This coarse flux-control test is not a local-energy/grid certificate.
    assert atmosphere.metadata["radiative_equilibrium_solver_converged"]
    assert atmosphere.metadata["radiative_equilibrium_converged"] == atmosphere.metadata["equilibrium_certificate"]["verified"]
    assert atmosphere.metadata["structure_solver"] == (
        "adaptive-trust-region-newton"
    )
    assert atmosphere.metadata["adaptive_structure_driver"] == "shared-lte"
    assert atmosphere.metadata[
        "initial_convective_gradient_projection_mode"
    ] == "unstable-node-gradient"
    assert atmosphere.metadata["maximum_total_flux_residual"] < 2.0e-3
    segments = atmosphere.metadata["nonlinear_solver_segments"]
    assert segments
    assert segments[-1]["phase"] == "local-energy-completion"
    assert atmosphere.metadata["formal_flux_completion_used"]
    assert atmosphere.metadata["maximum_relative_cell_energy_balance_residual"] < 2.0e-3
    assert atmosphere.metadata["nonlinear_solver_terminal_reason"] == (
        segments[-1]["terminal_reason"]
    )
    assert atmosphere.metadata["nonlinear_solver_residual_evaluations"] == sum(
        segment["residual_evaluations"] for segment in segments
    )
    assert atmosphere.metadata["nonlinear_solver_jacobian_evaluations"] == sum(
        segment["jacobian_evaluations"] for segment in segments
    )
    assert all("iteration_history" in segment for segment in segments)
    assert (
        atmosphere.metadata[
            "electron_scattering_source_iterations_per_evaluation"
        ]
        == 1
    )
    assert (
        atmosphere.metadata[
            "electron_scattering_source_final_maximum_relative_residual"
        ]
        >= 0.0
    )
    assert 0 <= atmosphere.metadata[
        "electron_scattering_source_final_worst_depth_index"
    ] < atmosphere.n_depth


def test_adaptive_newton_converges_warm_radiative_flux_control_with_ml2():
    """Enabling ML2 must not precondition stable radiative layers as convective."""
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        20_000.0,
        8.0,
        n_depth=12,
        max_iterations=60,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=0.7,
        n_angle=2,
        structure_solver="adaptive-newton",
    )

    assert atmosphere.metadata["radiative_equilibrium_solver_converged"]
    assert atmosphere.metadata["radiative_equilibrium_converged"] == atmosphere.metadata["equilibrium_certificate"]["verified"]
    assert atmosphere.metadata["maximum_total_flux_residual"] < 2.0e-3
    assert atmosphere.metadata["nonlinear_solver_segments"][-1]["phase"] == (
        "local-energy-completion"
    )
    assert atmosphere.metadata["maximum_relative_cell_energy_balance_residual"] < 2.0e-3


def test_adaptive_hydrogen_supplied_temperature_remains_a_warm_start():
    seed = gray_hydrogen_atmosphere(9_000.0, 8.0, n_depth=8)
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        9_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=0.7,
        n_angle=1,
        initial_temperature=seed.temperature,
        initial_column_mass=seed.column_mass,
        structure_solver="adaptive-newton",
    )

    assert atmosphere.metadata["initial_temperature_was_supplied"]
    assert not atmosphere.metadata["resumed_directly_in_formal_flux_phase"]
    assert atmosphere.metadata["initial_convective_gradient_projection"]
    assert atmosphere.metadata[
        "initial_convective_gradient_projection_mode"
    ] == "interface-transport"


def test_hydrogen_relaxation_callback_receives_updated_atmospheres():
    records = []

    def callback(iteration, atmosphere, status):
        records.append((iteration, atmosphere, status))

    result = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=8,
        max_iterations=2,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=None,
        n_angle=1,
        iteration_callback=callback,
    )

    assert [record[0] for record in records] == list(range(1, len(records) + 1))
    assert len(records) >= 2
    for _, atmosphere, status in records:
        assert atmosphere.effective_temperature == 12_000.0
        assert atmosphere.logg == 8.0
        assert np.all(np.isfinite(atmosphere.temperature))
        assert set(status) >= {
            "flux_ratio",
            "maximum_log_temperature_correction",
            "maximum_correction_depth_index",
            "converged",
        }
    np.testing.assert_allclose(
        records[-1][1].temperature,
        result.temperature,
        rtol=0.0,
        atol=0.0,
    )


def test_hydrogen_relaxation_uses_coupled_total_flux_ml2_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    from wd_spectra import adaptive_structure as convection

    original = (
        convection.ml2_temperature_gradient_for_total_flux_from_thermodynamics
    )
    calls = 0

    def tracking_solver(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        convection,
        "ml2_temperature_gradient_for_total_flux_from_thermodynamics",
        tracking_solver,
    )
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        8_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=0.7,
        n_angle=1,
    )

    assert calls > 0
    assert atmosphere.metadata["structure_solver"] == "adaptive-trust-region-newton"
    assert np.isfinite(atmosphere.metadata["maximum_total_flux_residual"])


def test_hydrogen_relaxation_preserves_custom_eos_state_and_label():
    def custom_eos(temperature, pressure):
        state = hummer_mihalas_hydrogen_lte(
            temperature,
            pressure,
            correlated_microfields=True,
        )
        return replace(state, chemical_model="test-custom-hydrogen-eos")

    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        include_paschen_lines=False,
        include_brackett_lines=False,
        mixing_length_alpha=None,
        n_angle=1,
        hydrogen_eos_function=custom_eos,
    )

    assert atmosphere.hydrogen_lte_state is not None
    assert (
        atmosphere.hydrogen_lte_state.chemical_model
        == "test-custom-hydrogen-eos"
    )
    assert atmosphere.metadata["eos"] == "test-custom-hydrogen-eos"
    assert atmosphere.metadata["radiative_equilibrium_custom_hydrogen_eos"]


def test_radiative_equilibrium_samples_hydrogen_line_cores():
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=True,
        include_lyman_lines=True,
        mixing_length_alpha=None,
        n_angle=1,
    )
    # The continuum plus broad-wing mesh alone has fewer than 1200 points at
    # this deliberately small continuum setting.
    assert atmosphere.metadata["radiative_equilibrium_wavelength_points"] > 1500
    assert atmosphere.metadata["radiative_equilibrium_resolves_balmer_line_cores"]
    assert not atmosphere.metadata["radiative_equilibrium_resolves_lyman_line_cores"]
    assert atmosphere.metadata["radiative_equilibrium_includes_paschen_lines"]
    assert atmosphere.metadata["radiative_equilibrium_includes_brackett_lines"]


def test_hot_radiative_equilibrium_automatically_samples_lyman_cores():
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        30_000.0,
        8.0,
        n_depth=8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=True,
        mixing_length_alpha=None,
        n_angle=1,
    )
    assert atmosphere.metadata["radiative_equilibrium_resolves_lyman_line_cores"]


def test_radiative_equilibrium_interpolates_checkpoint_on_extended_grid():
    initial = hydrogen_continuum_atmosphere(
        12_000.0, 8.0, n_depth=12, tau_min=1.0e-6
    )
    atmosphere = radiative_equilibrium_hydrogen_atmosphere(
        12_000.0,
        8.0,
        n_depth=10,
        tau_min=1.0e-8,
        max_iterations=1,
        n_continuum_wavelength=80,
        include_balmer_lines=False,
        include_lyman_lines=False,
        mixing_length_alpha=None,
        initial_temperature=initial.temperature,
        initial_column_mass=initial.column_mass,
    )
    assert atmosphere.n_depth == 10
    assert np.all(np.isfinite(atmosphere.temperature))
    assert np.all(np.diff(atmosphere.rosseland_optical_depth) > 0.0)


def test_ml2_flux_is_zero_for_stable_gradient_and_positive_when_unstable():
    stable = gray_hydrogen_atmosphere(14_000.0, 8.0, n_depth=24)
    stable_flux = ml2_convective_flux(stable, np.full(stable.n_depth, 0.1))
    assert np.all(stable_flux == 0.0)

    temperature = stable.temperature.copy()
    temperature[8:] *= np.geomspace(1.0, 3.0, temperature.size - 8)
    eos = ideal_hydrogen_lte(temperature, stable.gas_pressure)
    unstable = Atmosphere(
        effective_temperature=stable.effective_temperature,
        logg=stable.logg,
        rosseland_optical_depth=stable.rosseland_optical_depth,
        column_mass=stable.column_mass,
        temperature=temperature,
        gas_pressure=stable.gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={},
    )
    unstable_flux = ml2_convective_flux(
        unstable, np.full(unstable.n_depth, 0.1)
    )
    assert np.any(unstable_flux > 0.0)
    actual_gradient = np.gradient(
        np.log(unstable.temperature),
        np.log(unstable.gas_pressure),
        edge_order=2,
    )
    trial_flux = ml2_convective_flux_for_gradient(
        unstable,
        np.full(unstable.n_depth, 0.1),
        np.stack((actual_gradient, actual_gradient)),
    )
    assert trial_flux.shape == (2, unstable.n_depth)
    np.testing.assert_allclose(trial_flux[0], unstable_flux)
    np.testing.assert_allclose(trial_flux[1], unstable_flux)
    requested_flux = np.full(unstable.n_depth, 1.0e9)
    lower_gradient = ml2_temperature_gradient_for_flux(
        unstable, np.full(unstable.n_depth, 0.1), requested_flux
    )
    higher_gradient = ml2_temperature_gradient_for_flux(
        unstable, np.full(unstable.n_depth, 0.1), 2.0 * requested_flux
    )
    assert np.all(higher_gradient > lower_gradient)


def test_ml2_analytic_gradient_response_matches_centered_difference():
    atmosphere = gray_hydrogen_atmosphere(9_000.0, 8.0, n_depth=12)
    thermodynamics = ideal_hydrogen_thermodynamics(
        atmosphere.temperature, atmosphere.gas_pressure
    )
    opacity = np.full(atmosphere.n_depth, 0.1)
    gradient = np.full(atmosphere.n_depth, 0.6)
    step = 1.0e-6

    analytic = ml2_convective_flux_gradient_derivative_from_thermodynamics(
        atmosphere,
        opacity,
        gradient,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
        mixing_length_alpha=0.7,
    )
    upper = ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere,
        opacity,
        gradient + step,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
        mixing_length_alpha=0.7,
    )
    lower = ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere,
        opacity,
        gradient - step,
        thermodynamics.specific_heat_constant_pressure,
        thermodynamics.density_temperature_derivative,
        thermodynamics.adiabatic_temperature_gradient,
        mixing_length_alpha=0.7,
    )

    assert np.all(analytic > 0.0)
    np.testing.assert_allclose(
        analytic, (upper - lower) / (2.0 * step), rtol=2.0e-8
    )


def test_ml2_uses_shared_hydrogen_helium_thermodynamics_for_mixture():
    atmosphere = gray_hydrogen_helium_atmosphere(
        10_000.0, 8.0, -2.0, n_depth=12
    )
    expected = hummer_mihalas_hydrogen_helium_thermodynamics(
        atmosphere.temperature,
        atmosphere.gas_pressure,
        -2.0,
        hydrogen_neutral_radius_scale=(
            atmosphere.hydrogen_lte_state.neutral_radius_scale
        ),
        helium_neutral_radius_scale=(
            atmosphere.helium_lte_state.neutral_radius_scale
        ),
        correlated_microfields=True,
    ).adiabatic_temperature_gradient
    inferred = ml2_temperature_gradient_for_flux(
        atmosphere,
        np.full(atmosphere.n_depth, 0.1),
        np.zeros(atmosphere.n_depth),
    )
    np.testing.assert_allclose(inferred, expected, rtol=2.0e-9)
