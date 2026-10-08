from pathlib import Path

import numpy as np
from wd_spectra.models import ModelData
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.constants import (
    BOLTZMANN,
    HELIUM_FIRST_IONIZATION_ENERGY,
    HELIUM_SECOND_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PLANCK,
)

from wd_spectra import (
    HELIUM_I_LINES,
    HELIUM_II_LINES,
    deridder_van_rensbergen_helium_hwhm_angstrom,
    gray_helium_atmosphere,
    gray_hydrogen_helium_atmosphere,
    helium_continuum_mass_absorption_coefficient,
    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    hydrogen_helium_continuum_mass_absorption_coefficient,
    helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_dimer_ion_continuum_coefficient,
    helium_three_body_cia_linear_absorption_coefficient,
    helium_ii_bound_free_linear_absorption_coefficient,
    helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
    helium_ii_rayleigh_scattering_cross_section,
    helium_i_resonance_hwhm_angstrom,
    helium_i_resonance_stark_hwhm_angstrom,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_i_line_mass_absorption_coefficient,
    helium_i_rydberg_bound_free_linear_absorption_coefficient,
    helium_minus_free_free_coefficient,
    helium_rayleigh_scattering_cross_section,
    helium_rayleigh_scattering_mass_coefficient,
    neutral_helium_photoionization_cross_section,
    read_helium_stark_table,
    read_helium_ii_stark_table,
    synthesize_hydrogen_helium_spectrum,
)


def test_gray_helium_atmosphere_conserves_composition_and_charge():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=12)
    state = atmosphere.helium_lte_state
    assert state is not None
    assert atmosphere.metadata["composition"] == "pure-helium"
    assert np.allclose(
        state.neutral_he_density + state.singly_ionized_he_density + state.doubly_ionized_he_density,
        state.helium_nuclei_density,
        rtol=2e-12,
    )
    assert np.allclose(
        state.singly_ionized_he_density + 2.0 * state.doubly_ionized_he_density,
        atmosphere.electron_density,
        rtol=2e-12,
    )


def test_mixed_continuum_combines_hydrogen_and_helium_without_double_scattering():
    atmosphere = gray_hydrogen_helium_atmosphere(
        20_000.0, 8.0, -2.0, n_depth=5
    )
    wavelength = np.array([1000.0, 4000.0, 10_000.0])
    opacity = hydrogen_helium_continuum_mass_absorption_coefficient(
        atmosphere, wavelength
    )
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity > 0.0)


def test_mixed_synthesis_includes_hydrogen_pseudocontinuum_by_default():
    atmosphere = gray_hydrogen_helium_atmosphere(
        20_000.0, 8.0, -2.0, n_depth=10
    )
    wavelength = np.asarray([900.0, 915.0, 920.0, 930.0])
    opacity = hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere, wavelength
    )
    assert np.all(opacity[1:3] > 0.0)
    spectrum = synthesize_hydrogen_helium_spectrum(
        atmosphere,
        wavelength,
        stark_table=None,
        include_lines=False,
        backend="python",
    )
    assert spectrum.metadata["hydrogen_series_pseudocontinuum"] is True


def test_john_helium_minus_table_interpolation_and_long_wave_limit():
    exact = helium_minus_free_free_coefficient(15.1878, 5040.0)
    assert exact == pytest.approx(39.921e-26, rel=2e-12)
    long = helium_minus_free_free_coefficient(20.0, 5040.0)
    assert long == pytest.approx(0.173 * 20.0**2 * 1e-26, rel=2e-12)


def test_helium_minus_automatic_uses_john_1968_outside_1994_domain():
    wavelength_micron = 0.55
    temperature = 16_000.0
    frequency = LIGHT_SPEED / (wavelength_micron * 1.0e-4)
    coefficient_a = (
        3.397e-46 + (-5.216e-31 + 7.039e-15 / frequency) / frequency
    )
    coefficient_b = (
        -4.116e-42 + (1.067e-26 + 8.135e-11 / frequency) / frequency
    )
    coefficient_c = (
        5.081e-37 + (-8.724e-23 - 5.659e-8 / frequency) / frequency
    )
    expected = (
        coefficient_a * temperature
        + coefficient_b
        + coefficient_c / temperature
    ) / (BOLTZMANN * temperature)
    automatic = helium_minus_free_free_coefficient(
        wavelength_micron, temperature
    )
    old_fit = helium_minus_free_free_coefficient(
        wavelength_micron, temperature, prescription="john1968"
    )
    extrapolated_cool_table = helium_minus_free_free_coefficient(
        wavelength_micron, temperature, prescription="john1994"
    )
    assert automatic == pytest.approx(expected, rel=2e-12)
    assert old_fit == pytest.approx(expected, rel=2e-12)
    assert automatic < extrapolated_cool_table


def test_helium_minus_automatic_uses_john_1968_in_the_ultraviolet():
    automatic = helium_minus_free_free_coefficient(0.15, 8000.0)
    old_fit = helium_minus_free_free_coefficient(
        0.15, 8000.0, prescription="john1968"
    )
    extrapolated_cool_table = helium_minus_free_free_coefficient(
        0.15, 8000.0, prescription="john1994"
    )
    assert automatic == pytest.approx(old_fit, rel=2e-12)
    assert automatic > extrapolated_cool_table


def test_helium_rayleigh_has_inverse_fourth_power_limit():
    cross = helium_rayleigh_scattering_cross_section(np.asarray([4000.0, 8000.0]))
    assert cross[0] / cross[1] == pytest.approx(16.0)


def test_dense_helium_rayleigh_uses_rohrmann_structure_factor():
    atmosphere = gray_helium_atmosphere(
        4000.0, 8.0, n_depth=8, rosseland_opacity=0.01
    )
    wavelength = np.asarray([4000.0])
    dilute = helium_rayleigh_scattering_mass_coefficient(
        atmosphere, wavelength, include_fluid_correlation=False
    )[0]
    fluid = helium_rayleigh_scattering_mass_coefficient(
        atmosphere, wavelength, include_fluid_correlation=True
    )[0]
    expected = (1.0 + atmosphere.mass_density) ** (
        -46.67685 / atmosphere.temperature**0.3128
    )
    # The neutral term follows S(0) exactly; the unmodified, very small He II
    # Rayleigh contribution becomes visible only in the hottest deep point.
    np.testing.assert_allclose(fluid / dilute, expected, rtol=5e-3)


def test_helium_ii_rayleigh_has_hydrogenic_wavelength_scaling():
    from wd_spectra.opacity import hydrogen_rayleigh_scattering_cross_section

    wavelength = np.asarray([400.0, 800.0])
    assert np.allclose(
        helium_ii_rayleigh_scattering_cross_section(wavelength),
        hydrogen_rayleigh_scattering_cross_section(4.0 * wavelength),
        rtol=2e-15,
    )


def test_opacity_project_helium_threshold_cross_sections():
    expected_megabarn = [7.3361, 5.4632, 9.3672, 15.9509, 13.4126]
    threshold = [
        5.94503520e15, 1.15267210e15, 9.60145430e14,
        8.75933720e14, 8.14536220e14,
    ]
    for term, (frequency, expected) in enumerate(zip(threshold, expected_megabarn)):
        cross_section = neutral_helium_photoionization_cross_section(
            frequency, term
        )
        assert cross_section == pytest.approx(expected * 1e-18, rel=2e-4)
        assert neutral_helium_photoionization_cross_section(
            np.nextafter(frequency, 0.0), term
        ) == 0.0


def test_helium_continuum_is_positive_and_has_expected_shape():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=8)
    opacity = helium_continuum_mass_absorption_coefficient(
        atmosphere, np.asarray([300.0, 1000.0, 4000.0, 10_000.0])
    )
    assert opacity.shape == (4, 8)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity > 0.0)


def test_stancil_helium_dimer_ion_coefficient_at_table_knot():
    coefficient = helium_dimer_ion_continuum_coefficient(5000.0, 12600.0)
    expected = 0.8818e-18 / (21.097e21) + 0.0771e-39
    assert coefficient == pytest.approx(expected, rel=2e-12)


def test_stancil_helium_dimer_ion_coefficient_clips_cool_layers():
    boundary = helium_dimer_ion_continuum_coefficient(5000.0, 4200.0)
    cool = helium_dimer_ion_continuum_coefficient(5000.0, 3000.0)
    assert np.isfinite(cool)
    assert cool > 0.0
    assert cool == pytest.approx(boundary, rel=2e-15)


@pytest.mark.parametrize("wavenumber", [3000.0, 5000.0, 7000.0, 10000.0])
def test_kowalski_helium_three_body_cia_matches_analytic_fit(wavenumber):
    temperature = 5000.0
    density = 514.0 * 2.68678e19
    gamma = (
        -0.0601248 + 1.55103e-6 * temperature
    ) * temperature**-0.393053
    if wavenumber <= 4000.0:
        expected_per_amagat_cubed = (
            1.56e-19 * wavenumber**2.5 * np.exp(gamma * wavenumber)
        )
    else:
        expected_per_amagat_cubed = (
            1.56e-19
            * 4000.0**2.5
            * np.exp(gamma * 4000.0)
            * np.exp((gamma + 6.25e-4) * (wavenumber - 4000.0))
        )
    absorption = helium_three_body_cia_linear_absorption_coefficient(
        1.0e8 / wavenumber, temperature, density
    )
    assert absorption == pytest.approx(
        expected_per_amagat_cubed * 514.0**3, rel=2e-14
    )


def test_kowalski_helium_three_body_cia_has_cubic_density_and_declining_tail():
    density = 100.0 * 2.68678e19
    absorption = helium_three_body_cia_linear_absorption_coefficient(
        np.asarray([25_000.0, 10_000.0]),
        5000.0,
        density,
    )
    doubled = helium_three_body_cia_linear_absorption_coefficient(
        25_000.0, 5000.0, 2.0 * density
    )
    assert doubled == pytest.approx(8.0 * absorption[0], rel=2e-14)
    assert absorption[0] > 0.0
    assert 0.0 < absorption[1] < absorption[0]


@pytest.mark.parametrize("temperature", [1000.0, 3000.0, 5000.0, 8000.0, 10000.0])
def test_helium_cia_has_no_artificial_edge_and_a_smooth_join(temperature):
    density = 514.0 * 2.68678e19
    step = 0.01
    for center in (4000.0, 6000.0):
        wavenumber = center + np.array([-step, 0.0, step])
        coefficient = helium_three_body_cia_linear_absorption_coefficient(
            1e8 / wavenumber, temperature, density)
        log_slopes = np.diff(np.log(coefficient)) / step
        assert log_slopes[0] == pytest.approx(log_slopes[1], rel=3e-5)
        assert np.all(coefficient > 0.0)
    tail = helium_three_body_cia_linear_absorption_coefficient(
        1e8 / np.linspace(6000, 50000, 100), temperature, density)
    assert np.all(np.diff(tail) < 0.0)


def test_helium_cia_very_short_wavelength_remains_finite():
    with np.errstate(over="raise", invalid="raise"):
        coefficient = helium_three_body_cia_linear_absorption_coefficient(
            np.array([1e-100, 1.0, 1000.0]), 5000.0, 1e22)
    assert np.all(np.isfinite(coefficient))
    assert np.all(coefficient >= 0.0)


def test_helium_three_body_cia_can_be_disabled_in_continuum():
    atmosphere = gray_helium_atmosphere(5000.0, 8.0, n_depth=8)
    wavelength = np.asarray([25_000.0])
    with_cia = helium_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
    )
    without_cia = helium_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
        include_helium_three_body_cia=False,
    )
    assert np.all(with_cia >= without_cia)
    assert np.any(with_cia > without_cia)


def test_helium_rydberg_bound_free_is_finite_and_has_shell_edges():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=8)
    absorption = helium_i_rydberg_bound_free_linear_absorption_coefficient(
        atmosphere,
        np.asarray([4000.0, 8000.0, 9000.0]),
        minimum_principal_quantum_number=3,
        maximum_principal_quantum_number=3,
    )
    assert absorption.shape == (3, 8)
    assert np.all(np.isfinite(absorption))
    assert np.all(absorption[:2] > 0.0)
    assert np.all(absorption[2] == 0.0)


def test_helium_ii_bound_free_has_hydrogenic_ground_edge():
    atmosphere = gray_helium_atmosphere(40_000.0, 8.0, n_depth=8)
    absorption = helium_ii_bound_free_linear_absorption_coefficient(
        atmosphere,
        np.asarray([220.0, 240.0]),
        maximum_principal_quantum_number=1,
    )
    assert absorption.shape == (2, 8)
    assert np.all(np.isfinite(absorption))
    assert np.all(absorption[0] > 0.0)
    assert np.all(absorption[1] == 0.0)


def test_helium_ii_pseudocontinuum_is_confined_to_first_series_member():
    atmosphere = gray_helium_atmosphere(40_000.0, 8.0, n_depth=8)
    opacity = helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere,
        np.asarray([200.0, 250.0, 320.0]),
        maximum_lower_level=1,
    )
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.all(opacity[0] == 0.0)
    assert np.any(opacity[1] > 0.0)
    assert np.all(opacity[2] == 0.0)


def test_hydrogenic_helium_ii_4686_profile_is_finite_and_centered():
    atmosphere = gray_helium_atmosphere(40_000.0, 8.0, n_depth=8)
    line = next(
        item
        for item in HELIUM_II_LINES
        if (
            item.lower_principal_quantum_number,
            item.upper_principal_quantum_number,
        )
        == (3, 4)
    )
    wavelength = line.wavelength_vacuum_angstrom + np.asarray([-2.0, 0.0, 2.0])
    opacity = helium_ii_line_mass_absorption_coefficient(
        atmosphere, wavelength, lines=(line,)
    )
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.all(opacity[1] >= opacity[0])
    assert np.any(opacity[1] > opacity[0])
    assert np.allclose(opacity[0], opacity[2], rtol=1e-12)


def test_helium_ii_fuse_series_reaches_n10():
    line = next(
        item
        for item in HELIUM_II_LINES
        if (
            item.lower_principal_quantum_number,
            item.upper_principal_quantum_number,
        )
        == (2, 10)
    )
    assert line.wavelength_vacuum_angstrom == pytest.approx(949.32, abs=0.05)
    assert line.absorption_oscillator_strength == pytest.approx(0.003851)


def test_synspec_helium_ii_table_when_available():
    path = ModelData.default().helium_ii_stark
    if not path.exists():
        pytest.skip("optional SYNSPEC He II profile table not downloaded")
    table = read_helium_ii_stark_table(path)
    assert len(table.lines) == 19
    line = table[(3, 4)]
    center = next(
        item.wavelength_vacuum_angstrom
        for item in HELIUM_II_LINES
        if (
            item.lower_principal_quantum_number,
            item.upper_principal_quantum_number,
        )
        == (3, 4)
    )
    offset = line.wavelength_offset_angstrom
    profile = line.frequency_profile(
        center + offset, center, 40_000.0, 1.0e16
    )
    frequency_jacobian = 2.99792458e10 / (center * 1.0e-8) ** 2 * 1.0e-8
    area = 2.0 * trapezoid(profile, offset) * frequency_jacobian
    assert area == pytest.approx(1.0, rel=0.03)
    assert (4, 15) in table.lines
    assert any(
        line.lower_principal_quantum_number == 4
        and line.upper_principal_quantum_number == 15
        for line in HELIUM_II_LINES
    )
    bilinear = read_helium_ii_stark_table(
        path, thermodynamic_interpolation="log-bilinear"
    )
    adaptive = read_helium_ii_stark_table(
        path, thermodynamic_interpolation="series-adaptive"
    )
    query = np.asarray([center - 2.0, center, center + 2.0])
    np.testing.assert_allclose(
        adaptive[(3, 4)].frequency_profile(
            query, center, 40_000.0, 1.0e16
        ),
        line.frequency_profile(query, center, 40_000.0, 1.0e16),
    )
    high_center = next(
        item.wavelength_vacuum_angstrom
        for item in HELIUM_II_LINES
        if (
            item.lower_principal_quantum_number,
            item.upper_principal_quantum_number,
        )
        == (4, 9)
    )
    high_query = np.asarray([high_center - 2.0, high_center, high_center + 2.0])
    np.testing.assert_allclose(
        adaptive[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e16
        ),
        bilinear[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e16
        ),
    )
    # The adaptive interpolation is monotone inside the sparse high-member
    # table.  It bridges the missing log Ne=16--17 decade continuously before
    # reaching SYNSPEC's analytic continuation, rather than clamping forever
    # at Ne=1e16 cm^-3 or making SYNSPEC's discontinuous boundary jump.
    bridge_density = 10.0**16.5
    bridge = adaptive[(4, 9)].frequency_profile(
        high_query, high_center, 40_000.0, bridge_density
    )
    clamped = bilinear[(4, 9)].frequency_profile(
        high_query, high_center, 40_000.0, bridge_density
    )
    analytic = table[(4, 9)].frequency_profile(
        high_query, high_center, 40_000.0, bridge_density
    )
    assert np.all(bridge < clamped)
    assert np.all(bridge > analytic)
    np.testing.assert_allclose(
        adaptive[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e17
        ),
        table[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e17
        ),
    )
    assert not np.allclose(
        adaptive[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e17
        ),
        bilinear[(4, 9)].frequency_profile(
            high_query, high_center, 40_000.0, 1.0e17
        ),
        rtol=1.0e-3,
        atol=0.0,
    )
    # The adjacent 4->8 profile already reaches log Ne=17 and therefore needs
    # the ordinary analytic continuation, not the special missing-decade
    # bridge used by 4->9 and higher.
    adjacent = next(
        item
        for item in HELIUM_II_LINES
        if (
            item.lower_principal_quantum_number,
            item.upper_principal_quantum_number,
        )
        == (4, 8)
    )
    adjacent_query = np.asarray([
        adjacent.wavelength_vacuum_angstrom - 2.0,
        adjacent.wavelength_vacuum_angstrom,
        adjacent.wavelength_vacuum_angstrom + 2.0,
    ])
    np.testing.assert_allclose(
        adaptive[(4, 8)].frequency_profile(
            adjacent_query,
            adjacent.wavelength_vacuum_angstrom,
            40_000.0,
            1.0e18,
        ),
        table[(4, 8)].frequency_profile(
            adjacent_query,
            adjacent.wavelength_vacuum_angstrom,
            40_000.0,
            1.0e18,
        ),
    )


def test_helium_pseudocontinuum_is_finite_and_series_limited():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=8)
    wavelength = np.asarray([300.0, 550.0, 3000.0, 3500.0, 4500.0, 8000.0])
    opacity = helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere, wavelength
    )
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.all(opacity[0] == 0.0)
    assert np.all(opacity[-1] == 0.0)
    assert np.any(opacity[3:5] > 0.0)


def test_helium_pseudocontinua_are_finite_at_series_edges():
    atmosphere = gray_helium_atmosphere(40_000.0, 8.0, n_depth=8)
    he_i_edge = (
        LIGHT_SPEED * PLANCK / HELIUM_FIRST_IONIZATION_ENERGY * 1.0e8
    )
    he_ii_edge = (
        LIGHT_SPEED * PLANCK / HELIUM_SECOND_IONIZATION_ENERGY * 1.0e8
    )
    he_i_wavelength = np.asarray(
        [he_i_edge, np.nextafter(he_i_edge, np.inf)]
    )
    he_ii_wavelength = np.asarray(
        [he_ii_edge, np.nextafter(he_ii_edge, np.inf)]
    )
    he_i = helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
        atmosphere, he_i_wavelength
    )
    he_ii = (
        helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
            atmosphere, he_ii_wavelength, maximum_lower_level=1
        )
    )
    assert np.all(np.isfinite(he_i))
    assert np.all(np.isfinite(he_ii))
    assert np.all(he_i >= 0.0)
    assert np.all(he_ii >= 0.0)


def test_helium_ground_resonance_series_is_confined_to_euv_band():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=8)
    opacity = helium_i_resonance_line_mass_absorption_coefficient(
        atmosphere, np.asarray([400.0, 510.0, 584.33391989, 650.0, 800.0])
    )
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity >= 0.0)
    assert np.all(opacity[0] == 0.0)
    assert np.all(opacity[-1] == 0.0)
    assert np.all(opacity[2] > 0.0)


def test_ali_griem_resonance_width_matches_nist_6678_example():
    hwhm = helium_i_resonance_hwhm_angstrom(
        6678.15, 584.334, 0.2762, 1.0e18
    )
    # NIST quotes a 0.036-A FWHM; our convolutions use HWHM.
    assert 2.0 * hwhm == pytest.approx(0.036, rel=0.01)


@pytest.mark.parametrize(
    "lower_n,upper_n,lower_l,upper_l,alpha,beta",
    [
        (1.5, 3.0, "s", "p", 0.0538e-8, 0.3804),
        (2.0, 3.0, "p", "d", 0.0693e-8, 0.3335),
    ],
)
def test_deridder_helium_width_matches_source_table(
    lower_n, upper_n, lower_l, upper_l, alpha, beta
):
    wavelength = 5000.0
    density = 1.0e18
    temperature = 10_000.0
    width = deridder_van_rensbergen_helium_hwhm_angstrom(
        temperature,
        density,
        wavelength,
        lower_n,
        upper_n,
        lower_l,
        upper_l,
    )
    angular_hwhm = density * alpha * temperature**beta * 2.0**beta
    expected = (
        (wavelength * 1.0e-8) ** 2
        * angular_hwhm
        / (2.0 * np.pi * 2.99792458e10)
        * 1.0e8
    )
    assert width == pytest.approx(expected, rel=2e-12)


def test_helium_ground_resonance_stark_width_matches_source_table():
    hwhm = helium_i_resonance_stark_hwhm_angstrom(
        20_000.0,
        1.0e16,
        1.0e16,
        2,
    )
    # At 1e16 cm^-3 the table gives electron and He II FWHM of
    # 3.63e-4 and 9.66e-5 A, respectively.
    assert hwhm == pytest.approx(0.5 * (3.63e-4 + 9.66e-5), rel=2e-12)


def test_released_profile_table_when_available():
    path = ModelData.default().helium_i_stark
    if not path.exists():
        pytest.skip("optional Zenodo profile table not downloaded")
    table = read_helium_stark_table(path)
    assert len(HELIUM_I_LINES) == len(table.lines) == 36
    line = table[4471]
    wavelength = np.linspace(4272.756993, 4672.756993, 200_001)
    profile = line.wavelength_profile(wavelength, 4472.756993, 20_000.0, 1e16)
    assert trapezoid(profile, wavelength) == pytest.approx(1.0, rel=2e-5)

    convolved = line.wavelength_profile(
        wavelength,
        4472.756993,
        20_000.0,
        1e16,
        lorentz_hwhm_angstrom=0.5,
    )
    assert trapezoid(convolved, wavelength) == pytest.approx(1.0, rel=4e-3)
    assert np.max(convolved) < np.max(profile)


def test_compiled_helium_lorentz_kernel_matches_numpy_fallback(monkeypatch):
    import wd_spectra.helium_stark as helium_stark

    path = ModelData.default().helium_i_stark
    if path.exists() is False:
        pytest.skip("optional Zenodo profile table not downloaded")
    compiled = helium_stark._rt
    if compiled is None or not hasattr(
        compiled, "piecewise_linear_lorentz_convolution"
    ):
        pytest.skip("optional C profile kernel is not built")
    line = read_helium_stark_table(path)[4471]
    wavelength = np.linspace(4300.0, 4650.0, 701)
    accelerated = line.wavelength_profile(
        wavelength,
        4472.756993,
        13_000.0,
        3.0e16,
        lorentz_hwhm_angstrom=0.37,
    )
    monkeypatch.setattr(helium_stark, "_rt", None)
    fallback = line.wavelength_profile(
        wavelength,
        4472.756993,
        13_000.0,
        3.0e16,
        lorentz_hwhm_angstrom=0.37,
    )
    # The C and numpy convolutions sum the same terms in a different order.  In
    # the far wings (profile ~1e-5 of a 0.057 peak) the sum cancels, so rounding
    # reaches ~4e-15 of the peak.  Scale atol to the peak, not an absolute 1e-16.
    np.testing.assert_allclose(
        accelerated, fallback, rtol=2.0e-13, atol=1.0e-13 * float(np.max(fallback))
    )


def test_montreal_neutral_broadening_uses_deridder_wings_when_available():
    path = ModelData.default().helium_i_stark
    if not path.exists():
        pytest.skip("optional Zenodo profile table not downloaded")
    atmosphere = gray_helium_atmosphere(12_000.0, 8.0, n_depth=8)
    wavelength = np.linspace(5800.0, 5950.0, 1501)
    table = read_helium_stark_table(path)
    unsold = helium_i_line_mass_absorption_coefficient(
        atmosphere, wavelength, table, neutral_broadening="unsold"
    )
    montreal = helium_i_line_mass_absorption_coefficient(
        atmosphere, wavelength, table, neutral_broadening="montreal"
    )
    core = np.argmin(np.abs(wavelength - 5877.289448))
    red_wing = np.argmin(np.abs(wavelength - 5900.0))
    # In a neutral, line-forming layer the tabulated Deridder width is larger
    # than Unsold: oscillator strength moves out of the core and into wings.
    depth = 4
    assert montreal[core, depth] < unsold[core, depth]
    assert montreal[red_wing, depth] > unsold[red_wing, depth]
