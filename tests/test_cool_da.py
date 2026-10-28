import numpy as np
import pytest

from wd_spectra import (
    TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K,
    TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K,
    TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH,
    atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrogen_continuum_atmosphere,
    tremblay_2013_mean_3d_temperature_difference,
)


def test_tremblay_2013_temperature_difference_reproduces_digitized_7012_k_curve():
    difference = tremblay_2013_mean_3d_temperature_difference(
        7_012.0,
        10.0**TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    )
    np.testing.assert_array_equal(
        difference,
        TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K[1],
    )


def test_tremblay_2013_structure_control_preserves_pressure_and_recomputes_eos():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=30,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
        atmosphere
    )

    np.testing.assert_array_equal(corrected.gas_pressure, atmosphere.gas_pressure)
    np.testing.assert_array_equal(corrected.column_mass, atmosphere.column_mass)
    assert corrected.temperature[0] < atmosphere.temperature[0]
    assert np.max(np.abs(corrected.mass_density - atmosphere.mass_density)) > 0.0
    assert corrected.hydrogen_lte_state is not atmosphere.hydrogen_lte_state
    assert corrected.hydrogen_lte_state.microfield_model == "qmhd"
    assert corrected.hydrogen_lte_state.molecular_hydrogen_density is not None
    assert not corrected.metadata[
        "mean_3d_temperature_differential_is_equilibrium_model"
    ]


def test_tremblay_2013_structure_control_can_use_montreal_neutral_radius():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=18,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
        atmosphere,
        neutral_radius_scale=0.5,
    )

    assert corrected.hydrogen_lte_state is not None
    assert corrected.hydrogen_lte_state.neutral_radius_scale == 0.5
    assert (
        corrected.metadata[
            "mean_3d_temperature_differential_neutral_radius_scale"
        ]
        == 0.5
    )


def test_tremblay_2013_temperature_difference_rejects_extrapolation():
    with pytest.raises(ValueError, match="5997--12999"):
        tremblay_2013_mean_3d_temperature_difference(13_500.0, [1.0e-3, 1.0])
    with pytest.raises(ValueError, match="5997--12999"):
        tremblay_2013_mean_3d_temperature_difference(5_500.0, [1.0e-3, 1.0])


def test_tremblay_2013_table_covers_the_full_fig7_sequence():
    assert TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K.size == 12
    assert TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K.shape == (
        12,
        TREMBLAY_2013_LOG_ROSSELAND_DEPTH.size,
    )
    difference = tremblay_2013_mean_3d_temperature_difference(
        9_520.0,
        10.0**TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    )
    # 10**log tau round-trips through log10 only to rounding.
    np.testing.assert_allclose(
        difference, TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K[4], rtol=0.0, atol=1e-9
    )
    # Overshoot cooling of the upper layers and the warmer layers near
    # log tau_R = -1 that Tremblay et al. (2013) describe.
    assert difference[0] < -1_000.0
    assert difference[TREMBLAY_2013_LOG_ROSSELAND_DEPTH.tolist().index(-1.0)] > 200.0


def test_tremblay_2013_photospheric_layers_keep_the_1d_upper_atmosphere():
    tau = np.logspace(-6.0, 1.0, 71)
    full = tremblay_2013_mean_3d_temperature_difference(9_256.0, tau)
    photospheric = tremblay_2013_mean_3d_temperature_difference(
        9_256.0, tau, layers="photospheric"
    )
    upper = tau < 0.1 * TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH
    deep = tau > 10.0 * TREMBLAY_2013_UPPER_LAYER_ROSSELAND_DEPTH
    assert np.max(np.abs(photospheric[upper])) < 2.0e-3 * np.max(np.abs(full[upper]))
    np.testing.assert_allclose(photospheric[deep], full[deep], rtol=2.0e-3, atol=0.0)
    with pytest.raises(ValueError, match="layers"):
        tremblay_2013_mean_3d_temperature_difference(9_256.0, tau, layers="top")


def test_tremblay_2013_hydrostatic_control_reintegrates_pressure():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=10,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = (
        hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere
        )
    )

    assert np.all(np.diff(corrected.gas_pressure) > 0.0)
    np.testing.assert_allclose(
        corrected.column_mass,
        corrected.gas_pressure / corrected.gravity,
        rtol=2.0e-15,
    )
    assert np.max(np.abs(corrected.gas_pressure / atmosphere.gas_pressure - 1.0)) > 0.01
    assert corrected.metadata[
        "mean_3d_temperature_differential_preserves_pressure"
    ] is False


def test_tremblay_2013_hydrostatic_reintegration_is_consistent_without_a_differential(
    monkeypatch,
):
    import wd_spectra.cool_da as cool_da

    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=24,
        correlated_microfields=True,
        include_molecules=True,
    )
    monkeypatch.setattr(
        cool_da,
        "tremblay_2013_mean_3d_temperature_difference",
        lambda teff, tau, **kw: np.zeros_like(np.asarray(tau, dtype=float)),
    )
    unchanged = cool_da.hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
        atmosphere
    )
    np.testing.assert_allclose(
        unchanged.gas_pressure, atmosphere.gas_pressure, rtol=3.0e-3, atol=0.0
    )


def test_tremblay_2013_hydrostatic_photospheric_layers_keep_upper_temperatures():
    atmosphere = hydrogen_continuum_atmosphere(
        9_256.0,
        8.0,
        n_depth=24,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = (
        hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere, layers="photospheric"
        )
    )
    upper = atmosphere.rosseland_optical_depth < 1.0e-4
    np.testing.assert_allclose(
        corrected.temperature[upper], atmosphere.temperature[upper], rtol=1.0e-6
    )
    assert corrected.metadata["mean_3d_temperature_differential_layers"] == (
        "photospheric"
    )


def test_da_default_microphysics_uses_montreal_radius_and_ali_griem():
    from wd_spectra.eos import (
        HM_HELIUM_NEUTRAL_RADIUS_SCALE,
        HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
    )
    from wd_spectra.models.stellar import _da_self_broadening_prescription

    assert HM_NEUTRAL_HYDROGEN_RADIUS_SCALE == 0.5
    assert HM_HELIUM_NEUTRAL_RADIUS_SCALE == 0.5
    assert _da_self_broadening_prescription(5_000.0, None) == "ali-griem"
    assert _da_self_broadening_prescription(20_000.0, None) == "ali-griem"
    assert _da_self_broadening_prescription(8_000.0, "barklem") == "barklem"


def test_da_mean_3d_line_profile_mode_selects_layers_by_temperature():
    from wd_spectra import DAConfig
    from wd_spectra.models.stellar import _mean_3d_line_profile_layers

    def layers(teff, logg=8.0, enabled=True):
        return _mean_3d_line_profile_layers(
            DAConfig(
                effective_temperature=teff,
                logg=logg,
                mean_3d_line_profiles=enabled,
            )
        )

    assert layers(7_000.0, enabled=False) == (None, None)
    assert layers(7_000.0)[0] == "all"
    assert layers(8_999.0)[0] == "all"
    assert layers(9_000.0)[0] == "photospheric"
    assert layers(12_999.0)[0] == "photospheric"
    for skipped, reason in (
        (layers(5_500.0), "Teff outside"),
        (layers(13_500.0), "Teff outside"),
        (layers(8_000.0, logg=7.0), "log g"),
    ):
        assert skipped[0] is None and reason in skipped[1]


def test_da_mean_3d_line_profile_mode_changes_synthesis_but_not_the_returned_structure():
    import json
    import warnings
    from pathlib import Path

    from wd_spectra import DAConfig, compute_da
    from wd_spectra.models import (
        AtmosphereConvergenceWarning,
        load_atmosphere_checkpoint,
    )

    path = Path(__file__).parent / "data/da_cubic_regressions/da-12000-public.npz"
    with np.load(path) as saved:
        stored = json.loads(str(saved["config_json"]))
        wave = saved["wavelength"]
    config = DAConfig(**stored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        state = load_atmosphere_checkpoint(
            path,
            config.effective_temperature,
            config.logg,
            "hydrogen",
            include_molecules=True,
            include_negative_hydrogen=True,
            trihydrogen_ion_partition_model="neale-tennyson-1995",
        )
        default = compute_da(
            config, wave, initial_atmosphere=state, relax_atmosphere=False
        )
        result = compute_da(
            DAConfig(**{**stored, "mean_3d_line_profiles": True}),
            wave,
            initial_atmosphere=state,
            relax_atmosphere=False,
        )
    np.testing.assert_array_equal(result.atmosphere.temperature, state.temperature)
    assert "layers=photospheric" in result.metadata["mean_3d_line_profiles"]
    assert result.metadata["flux_conserving_spectrum"] is False
    assert default.metadata["mean_3d_line_profiles"] is None
    assert default.metadata["flux_conserving_spectrum"] is True
    relative = np.abs(
        result.spectrum.surface_flux_lambda
        / default.spectrum.surface_flux_lambda
        - 1.0
    )
    assert np.max(relative) > 5.0e-3
    with pytest.raises(ValueError, match="retired"):
        compute_da(
            DAConfig(use_cool_mean_3d_temperature_differential=True),
            wave,
            initial_atmosphere=state,
            relax_atmosphere=False,
        )
