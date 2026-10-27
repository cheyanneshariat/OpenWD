"""Caller-supplied DAH geometry, physical routing and atmosphere provenance."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from wd_spectra import DAHConfig, Spectrum, StokesSpectrum, compute_dah, paper_dah_config
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.magnetic import (
    BalmerBroadening, MagneticPhysics, SurfaceCells, synthesize_magnetic_hydrogen_spectrum,
)
from wd_spectra.models import dah
from wd_spectra.models.common import (
    AtmosphereConvergenceWarning, ModelData, numerical_resolution, save_model_result,
)


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15000., 8., n_depth=8, correlated_microfields=True)


def cells(field=.5):
    return SurfaceCells(np.full(2, field), np.array([.6, -.6]), np.array([.4, .8]),
                        np.array([.25, .75]), field_azimuth_rad=np.array([.1, 1.2]))


@pytest.mark.parametrize("factory,field", [(paper_dah_config, .5), (DAHConfig, 30.)],
                         ids=["paper-weak", "default-strong"])
def test_custom_public_geometry_matches_direct_transfer(atmosphere, tmp_path, factory, field):
    config = factory(effective_temperature=15000., logg=8.,
                     polarized_transfer="full-stokes-iquv", return_stokes=True)
    geometry = cells(field)
    wave = np.linspace(4800., 4930., 11)
    with pytest.warns(AtmosphereConvergenceWarning):
        result = compute_dah(config, wave, surface_cells=geometry,
                             initial_atmosphere=atmosphere, relax_atmosphere=False)
    data = ModelData.default()
    transitions = dah.read_h2db_transition_database(data.h2db_balmer_subset) if field > 1 else None
    energies = dah.read_h2db_energy_database(data.h2db_balmer_subset) if field > 1 else None
    physics = MagneticPhysics(transitions, energies,
        broadening=BalmerBroadening(include_self_broadening=False),
        balmer_profile=config.balmer_profile, normalize_balmer_strength=config.normalize_balmer_strength,
        include_rwa_photoionization=config.include_rwa_photoionization,
        include_magnetic_eos=config.include_magnetic_eos,
        include_centered_motion=config.include_centered_motion,
        line_regime="h2db" if field > 1 else "linear-zeeman")
    expected = synthesize_magnetic_hydrogen_spectrum(atmosphere, wave, geometry, physics,
        polarized_transfer="full-stokes-iquv", return_stokes=True,
        n_angle=numerical_resolution(config.quality).n_angle)
    for name in ("surface_flux_lambda", "surface_flux_q_lambda", "surface_flux_u_lambda", "surface_flux_v_lambda"):
        np.testing.assert_array_equal(getattr(result.spectrum, name), getattr(expected, name))
    assert result.metadata["visible_mean_field_megagauss"] == pytest.approx(field)
    assert result.metadata["maximum_visible_field_megagauss"] == field
    assert result.metadata["surface_geometry_source"] == "surface_cells"
    assert "caller-supplied" in result.metadata["disk_quadrature"]
    assert result.spectrum.metadata["magnetic_line_regime"] == ("h2db" if field > 1 else "linear-zeeman")
    save_model_result(result, tmp_path)
    saved = json.loads((tmp_path / "metadata.json").read_text())["spectrum_metadata"]
    for key, array in (("surface_field_ray_cosine", geometry.field_ray_cosine),
                       ("surface_ray_mu", geometry.ray_mu),
                       ("surface_field_azimuth_rad", geometry.field_azimuth_rad)):
        np.testing.assert_array_equal(saved[key], array)
    assert saved["surface_geometry_sha256"] == result.metadata["surface_geometry_sha256"]


@pytest.fixture
def fake_model(monkeypatch, atmosphere):
    # These unit stubs check dispatch and request identity, not equilibrium.
    seen = {}
    def host(config, *args, **kwargs):
        seen["host_config"] = config
        if "mutate_input" in seen:
            seen["mutate_input"]()
        return SimpleNamespace(atmosphere=atmosphere, metadata={"convection": "test"})
    def synthesis(atm, wave, geometry, physics, **kwargs):
        seen.update(geometry=geometry, physics=physics)
        metadata = {"magnetic_line_regime": physics.line_regime, "polarized_transfer": "test",
                    "surface_field_megagauss": geometry.field_strength_megagauss.tolist(),
                    "surface_projected_weight": geometry.projected_weight.tolist(),
                    "stokes_convention": "test"}
        if kwargs["return_stokes"]:
            return StokesSpectrum(wave, np.ones_like(wave), metadata,
                                  np.zeros_like(wave), np.zeros_like(wave), np.zeros_like(wave))
        return Spectrum(wave, np.ones_like(wave), metadata)
    monkeypatch.setattr(dah, "compute_da", host)
    monkeypatch.setattr(dah, "synthesize_magnetic_hydrogen_spectrum", synthesis)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "converged")
    return seen


def test_custom_geometry_controls_convection_and_is_snapshotted(fake_model):
    geometry = cells(.01)
    config = DAHConfig(effective_temperature=15000., logg=8., magnetic_field_megagauss=30.)
    fake_model["mutate_input"] = lambda: geometry.field_strength_megagauss.fill(300.)
    result = compute_dah(config, np.array([4800., 4900.]), surface_cells=geometry)
    assert fake_model["host_config"].mixing_length_alpha == .7
    assert result.metadata["visible_mean_field_megagauss"] == pytest.approx(.01)
    np.testing.assert_array_equal(fake_model["geometry"].field_strength_megagauss, [.01, .01])
    assert not fake_model["geometry"].field_strength_megagauss.flags.writeable
    assert geometry.field_strength_megagauss.flags.writeable
    assert fake_model["physics"].line_regime == "linear-zeeman"


def test_custom_structure_identity_tracks_field_and_weights_not_sky_basis(fake_model):
    config = DAHConfig(effective_temperature=15000., logg=8.,
                      polarized_transfer="full-stokes-iquv", return_stokes=True)
    geometry = cells()
    wave = np.array([4800., 4900.])
    first = compute_dah(config, wave, surface_cells=geometry)
    same = compute_dah(replace(config, sky_position_angle_deg=37.,
        magnetic_field_megagauss=500., field_geometry="dipole", disk_field_bins=21), wave,
        surface_cells=geometry, initial_atmosphere=first.atmosphere, relax_atmosphere=False)
    assert same.atmosphere.metadata["fixed_synthesis_request_verified"]
    assert same.metadata["surface_geometry_sha256"] != first.metadata["surface_geometry_sha256"]
    np.testing.assert_allclose(fake_model["geometry"].field_azimuth_rad,
                               geometry.field_azimuth_rad + np.deg2rad(37.))
    other_view = replace(geometry, field_ray_cosine=-geometry.field_ray_cosine,
                         ray_mu=np.array([.5, .9]))
    viewed = compute_dah(config, wave, surface_cells=other_view,
                        initial_atmosphere=first.atmosphere, relax_atmosphere=False)
    assert viewed.atmosphere.metadata["fixed_synthesis_request_verified"]
    assert viewed.metadata["surface_geometry_sha256"] != first.metadata["surface_geometry_sha256"]
    changes = [replace(geometry, field_strength_megagauss=np.array([.4, .8])),
               replace(geometry, projected_weight=np.array([.5, .5])),
               replace(geometry, field_bounds_megagauss=(.1, .9))]
    for other in changes:
        fixed = compute_dah(config, wave, surface_cells=other,
                            initial_atmosphere=first.atmosphere, relax_atmosphere=False)
        assert not fixed.atmosphere.metadata["fixed_synthesis_request_verified"]
    physical = compute_dah(replace(config, include_rwa_photoionization=False), wave,
        surface_cells=geometry, initial_atmosphere=first.atmosphere, relax_atmosphere=False)
    assert not physical.atmosphere.metadata["fixed_synthesis_request_verified"]


def test_custom_mean_field_structure_uses_actual_weighted_field(fake_model, monkeypatch, atmosphere):
    # A weak placeholder in DAHConfig must not send a strong custom map to
    # the ordinary DA solver, nor set the magnetic structure to that field.
    geometry = replace(cells(), field_strength_megagauss=np.array([10., 30.]))
    config = DAHConfig(effective_temperature=15000., logg=8., magnetic_field_megagauss=.001,
                      atmosphere_structure="mean-field", quality="quick")
    captured = {}
    opacity_factory = dah.magnetic_structure_opacity
    def opacity(field, physics):
        captured["field"] = field
        return opacity_factory(field, physics)
    def structure(*args, **kwargs):
        captured.update(kwargs)
        return atmosphere
    monkeypatch.setattr(dah, "magnetic_structure_opacity", opacity)
    monkeypatch.setattr(dah, "radiative_equilibrium_hydrogen_atmosphere", structure)
    result = compute_dah(config, np.array([4800., 4900.]), surface_cells=geometry)
    assert "host_config" not in fake_model
    assert captured["field"] == 25.
    assert captured["mixing_length_alpha"] is None
    assert captured["hydrogen_eos_function"] is not None
    assert result.metadata["structure_field_megagauss"] == 25.
    assert fake_model["physics"].line_regime == "h2db"


def test_custom_conservative_bounds_select_regime_and_domain_notes(fake_model):
    geometry = replace(cells(), field_bounds_megagauss=(.1, 240.), field_bounds_exact=False)
    result = compute_dah(DAHConfig(), np.array([4800., 4900.]), surface_cells=geometry)
    assert fake_model["physics"].line_regime == "h2db"
    assert result.metadata["maximum_visible_field_megagauss"] == 240.
    assert not result.metadata["visible_field_bounds_exact"]
    assert any("beyond its tested range" in note for note in result.metadata["domain_notes"])
    assert not any("equal-weight" in note for note in result.metadata["domain_notes"])


def test_custom_intensity_cells_can_omit_azimuth(fake_model):
    geometry = replace(cells(), field_azimuth_rad=None)
    result = compute_dah(DAHConfig(effective_temperature=15000., logg=8.),
                         np.array([4800., 4900.]), surface_cells=geometry)
    assert type(result.spectrum) is Spectrum
    assert result.spectrum.metadata["surface_field_azimuth_rad"] is None


@pytest.mark.parametrize("change", [
    {"field_strength_megagauss": np.array([-1., .5])},
    {"field_ray_cosine": np.array([1.1, 0.])}, {"ray_mu": np.array([0., .8])},
    {"projected_weight": np.array([.2, .2])}, {"field_azimuth_rad": None},
    {"field_azimuth_rad": np.array([0.])},
    {"field_strength_megagauss": np.array([complex(1., 1.), .5])},
    {"field_bounds_megagauss": (.6, .7)},
    {"field_ray_cosine": np.array([np.nan, 0.])},
])
def test_invalid_custom_cells_fail_before_atmosphere(monkeypatch, change):
    monkeypatch.setattr(dah, "compute_da", lambda *a, **k: pytest.fail("started atmosphere"))
    config = DAHConfig(return_stokes=True, polarized_transfer="full-stokes-iquv")
    with pytest.raises(ValueError):
        compute_dah(config, np.array([4800., 4900.]), surface_cells=replace(cells(), **change))


def test_custom_cells_reject_wrong_type_and_compression_before_atmosphere(monkeypatch):
    monkeypatch.setattr(dah, "compute_da", lambda *a, **k: pytest.fail("started atmosphere"))
    with pytest.raises(TypeError, match="SurfaceCells"):
        compute_dah(surface_cells=object())
    config = DAHConfig(field_geometry="dipole", disk_component_drift_angstrom=10.)
    with pytest.raises(ValueError, match="compression"):
        compute_dah(config, surface_cells=cells())
