"""Physical widths and numerical convolution, independent of observed stars."""

import numpy as np
from dataclasses import replace
import pytest
from scipy.fft import irfft
from wd_spectra.carbon_molecular import C2CrossSectionTable
from wd_spectra.c2_profiles import (
    C2ImpactProfiles,
    helium_impact_hwhm,
    lorentz_cell_kernel,
)


def synthetic_table():
    nu = np.geomspace(19800, 20200, 1001)
    strengths = np.zeros((len(nu), 2))
    strengths[len(nu) // 2, :] = 1e-17
    return C2CrossSectionTable(
        1e8 / nu[::-1],
        np.array([5000.0, 10000.0]),
        strengths,
        np.array([5000.0, 10000.0]),
        np.array([1000.0, 2000.0]),
    )


def test_impact_width_uses_local_state_and_synspec_units():
    expected = 1e-7 * 0.42 * 1e20 / (4 * np.pi * 2.99792458e10)
    assert helium_impact_hwhm(10000, 1e20) == pytest.approx(expected)
    assert helium_impact_hwhm(5000, 2e20) == pytest.approx(expected * 2 * 0.5**0.45)
    assert helium_impact_hwhm(5000, 0) == 0
    with pytest.raises(ValueError):
        helium_impact_hwhm(5000, -1)


def test_rotational_default_requires_data_and_preserves_explicit_legacy_profile():
    from wd_spectra._dq.base import DQConfig, DQMaterial
    from wd_spectra.models.common import ModelData
    from wd_spectra.c2_profiles import C2RotationalProfiles

    assert DQConfig().molecular_line_profile == "rotational_overlap"
    original = synthetic_table()
    with pytest.raises(ValueError, match="rebuild"):
        DQMaterial(DQConfig(), ModelData.default(), original)
    legacy = DQMaterial(
        DQConfig(molecular_line_profile="line_bin"), ModelData.default(), original
    )
    assert legacy.c2_profiles is None
    overlap = np.roll(original.cross_section, 7, axis=0) * 0.4
    both = replace(
        original,
        swan_cross_section=0.4 * original.cross_section,
        swan_rotational_overlap_cross_section=overlap,
    )
    wave = both.wavelength_angstrom
    profile = C2RotationalProfiles(both)
    np.testing.assert_allclose(
        profile.cross_section(wave, [5000], [0])[:, 0],
        0.6 * original.cross_section[:, 0] + overlap[:, 0],
        rtol=1e-14,
    )
    np.testing.assert_array_equal(both.cross_section, original.cross_section)
    with pytest.raises(ValueError):
        both.swan_rotational_overlap_cross_section[:] = 0
    with pytest.raises(ValueError, match="rotational-overlap"):
        replace(original, swan_rotational_overlap_cross_section=overlap)


def test_rotational_swan_shift_is_local_blueward_and_does_not_move_other_systems():
    from wd_spectra.c2_profiles import (
        C2RotationalProfiles,
        swan_density_shift_wavenumber,
    )

    original = synthetic_table()
    both = replace(
        original,
        swan_cross_section=0.4 * original.cross_section,
        swan_rotational_overlap_cross_section=0.4 * original.cross_section,
    )
    wave = np.linspace(4960, 5040, 4000)
    total = C2RotationalProfiles(both)
    swan = C2RotationalProfiles(both, component="swan")
    zero = swan.cross_section(wave, [7000, 7000], [1e19, 1e19])
    moved = swan.cross_section(
        wave,
        [7000, 7000],
        [1e19, 1e19],
        wavenumber_shift=swan_density_shift_wavenumber([0, 0.01]),
    )
    np.testing.assert_array_equal(zero[:, 0], moved[:, 0])
    assert wave[np.argmax(moved[:, 1])] < wave[np.argmax(zero[:, 1])]
    baseline = total.cross_section(wave, [7000, 7000], [1e19, 1e19])
    np.testing.assert_allclose(
        (baseline - zero + moved) - moved, 0.6 * baseline, atol=1e-32
    )
    with pytest.raises(ValueError):
        swan.cross_section(wave, [7000], [0], wavenumber_shift=[-1])


def test_dual_profile_table_round_trip_keeps_profiles_separate(tmp_path):
    from wd_spectra.carbon_molecular import read_c2_cross_section_table
    from wd_spectra.c2_profiles import C2RotationalProfiles

    original = synthetic_table()
    payload = {
        name: getattr(original, name)
        for name in (
            "wavelength_angstrom",
            "temperature_K",
            "cross_section",
            "partition_temperature_K",
            "partition_function",
            "source",
        )
    }
    payload["swan_cross_section"] = 0.4 * original.cross_section
    payload["swan_rotational_overlap_cross_section"] = np.roll(
        0.4 * original.cross_section, 7, axis=0
    )
    path = tmp_path / "dual-profile.npz"
    np.savez_compressed(path, **payload)
    restored = read_c2_cross_section_table(path)
    for name in payload:
        np.testing.assert_array_equal(getattr(restored, name), payload[name])
    selected = C2RotationalProfiles(restored).table
    np.testing.assert_array_equal(
        selected.cross_section,
        restored.cross_section
        - restored.swan_cross_section
        + restored.swan_rotational_overlap_cross_section,
    )


def test_exomol_definition_width_has_correct_bar_and_temperature_conversion():
    from wd_spectra.constants import BOLTZMANN

    density = 1e6 / (BOLTZMANN * 296.0)
    assert helium_impact_hwhm(296.0, density, "exomol_default") == pytest.approx(0.07)
    assert helium_impact_hwhm(296.0 * 4, density, "exomol_default") == pytest.approx(
        0.14
    )
    with pytest.raises(ValueError, match="unknown"):
        helium_impact_hwhm(7000, 1e20, "fitted")


def test_cell_kernel_integral_is_not_artificially_renormalized():
    n, step, gamma = 100, 0.3, 2.0
    kernel = lorentz_cell_kernel(n, step, gamma)
    assert np.all(kernel > 0)
    expected = 2 / np.pi * np.arctan((n - 0.5) * step / gamma)
    assert kernel[0] + 2 * np.sum(kernel[1:]) == pytest.approx(expected, rel=1e-14)
    np.testing.assert_array_equal(lorentz_cell_kernel(3, 1, 0), [1, 0, 0])


def test_fft_matches_nonperiodic_direct_convolution_and_preserves_tail_loss():
    p = C2ImpactProfiles(synthetic_table(), wavenumber_step=0.3)
    density = 1e20
    actual = p._column(5000.0, density)
    source = irfft(p._source_transform(0), p.fft_size)[: len(p.nu)]
    kernel = lorentz_cell_kernel(
        len(p.nu), p.spacing, helium_impact_hwhm(5000.0, density)
    )
    for j in (0, len(p.nu) // 2, len(p.nu) - 1):
        expected = np.sum(source * kernel[np.abs(np.arange(len(source)) - j)])
        assert actual[j] == pytest.approx(expected, rel=1e-10)
    assert 0.9 < np.sum(actual) / np.sum(source) < 1


def test_numerical_refinement_does_not_change_physical_width():
    table = synthetic_table()
    wave = np.linspace(4960, 5040, 1000)
    coarse = C2ImpactProfiles(table, wavenumber_step=0.5).cross_section(
        wave, [7000], [1e20]
    )
    fine = C2ImpactProfiles(table, wavenumber_step=0.125).cross_section(
        wave, [7000], [1e20]
    )
    np.testing.assert_allclose(coarse, fine, rtol=0.003, atol=1e-24)


def test_profile_validation_and_exact_state_cache():
    p = C2ImpactProfiles(synthetic_table(), cache_columns=2)
    with pytest.raises(ValueError, match="clamping"):
        p.cross_section([5000], [1000], [1e20])
    with pytest.raises(ValueError):
        p.cross_section([5000], [7000], [np.nan])
    a = p._column(7000, 1e20)
    assert a is p._column(7000, 1e20)
    b = p._column(7000.00001, 1e20)
    assert a is not b
    p._column(8000, 1e20)
    assert len(p.columns) == 2
    np.testing.assert_array_equal(p.cross_section([100, 100000], [7000], [1e20]), 0)


def test_swan_shift_requires_identified_transitions_and_moves_only_that_component():
    from wd_spectra.c2_profiles import swan_density_shift_wavenumber

    table = synthetic_table()
    with pytest.raises(ValueError, match="split-swan"):
        C2ImpactProfiles(table, component="swan")
    table = replace(table, swan_cross_section=0.4 * table.cross_section)
    with pytest.raises(ValueError):
        replace(table, swan_cross_section=2 * table.cross_section)
    with pytest.raises(ValueError):
        table.swan_cross_section[:] = 0
    total = C2ImpactProfiles(table)
    swan = C2ImpactProfiles(table, component="swan")
    wave = np.linspace(4960.0, 5040.0, 3000)
    unshifted = swan.cross_section(wave, [7000.0], [1e19])
    shift = swan_density_shift_wavenumber([0.01])
    assert shift[0] == pytest.approx(16.131087874)
    shifted = swan.cross_section(wave, [7000.0], [1e19], wavenumber_shift=shift)
    assert wave[np.argmax(shifted)] < wave[np.argmax(unshifted)]
    baseline = total.cross_section(wave, [7000.0], [1e19])
    combined = baseline - unshifted + shifted
    np.testing.assert_allclose(combined - shifted, baseline * 0.6, rtol=1e-11)
    with pytest.raises(ValueError):
        swan.cross_section(wave, [7000.0], [1e19], wavenumber_shift=[np.nan])
