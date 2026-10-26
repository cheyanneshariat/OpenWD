"""Signed DAH geometry, optical Stokes conventions, and saved public output."""

from dataclasses import replace
import json
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from wd_spectra import DAHConfig, Spectrum, StokesSpectrum, compute_dah, select_physics
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.magnetic import (
    BalmerBroadening, MagneticPhysics, PolarizedOpacity, SurfaceCells,
    _observer_polarization, _validate_stokes_cells, dipole_surface_cells,
    field_binned_surface_cells, synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import read_h2db_transition_database, read_h2db_energy_database
from wd_spectra.models.common import AtmosphereConvergenceWarning, ModelData, save_model_result
from wd_spectra.models.dah import dah_surface_cells
from wd_spectra.radiative_transfer import emergent_stokes_specific_intensity


def cells(cosines=(.6, -.6), azimuths=(0., 0.), field=.5):
    n = len(cosines)
    return SurfaceCells(np.full(n, field), np.array(cosines), np.full(n, .7),
                        np.full(n, 1/n), field_azimuth_rad=np.array(azimuths))


@pytest.fixture(scope="module")
def atmosphere():
    # Diagnostic fixed structure; no equilibrium or observational claim.
    return gray_hydrogen_atmosphere(15000., 8., n_depth=16, correlated_microfields=True)


def physics(field):
    data = ModelData.default()
    return MagneticPhysics(
        read_h2db_transition_database(data.h2db_balmer_subset) if field > 1 else None,
        read_h2db_energy_database(data.h2db_balmer_subset) if field > 1 else None,
        broadening=BalmerBroadening(include_self_broadening=False),
        balmer_profile="kurucz-griem", normalize_balmer_strength=True,
        include_rwa_photoionization=False, include_magnetic_eos=False,
        include_centered_motion=False,
        line_regime="h2db" if field > 1 else "linear-zeeman",
    )


def synth(atmosphere, geometry, *, output=True, wavelength=None):
    wave = np.arange(4800., 4931., 2.) if wavelength is None else wavelength
    return synthesize_magnetic_hydrogen_spectrum(
        atmosphere, wave, geometry, physics(geometry.maximum_field_megagauss),
        return_stokes=output, polarized_transfer="full-stokes-iquv",
    )


def test_fixed_uniform_angles_retain_polarity_only_when_requested():
    toward = uniform_field_surface_cells(.5, field_angle_deg=40., signed_geometry=True)
    away = uniform_field_surface_cells(.5, field_angle_deg=140., signed_geometry=True)
    np.testing.assert_allclose(toward.field_ray_cosine, -away.field_ray_cosine, atol=2e-16)
    assert np.all(away.field_ray_cosine < 0)
    assert np.all(uniform_field_surface_cells(.5, field_angle_deg=140.).field_ray_cosine > 0)


@pytest.mark.parametrize("offset", [(0., 0., 0.), (.05, -.03, .1)])
def test_signed_dipole_matches_independent_vector_geometry(offset):
    inc, bp = np.deg2rad(60.), 10.
    raw = dipole_surface_cells(bp, inclination_deg=60., offset_vector_radius=offset,
                              n_mu=4, n_azimuth=8, n_field_bins=None, signed_geometry=True)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    mu, phi = np.meshgrid((nodes+1)/2, 2*np.pi*(np.arange(8)+.5)/8, indexing="ij")
    pos = np.stack([np.sqrt(1-mu**2)*np.cos(phi), np.sqrt(1-mu**2)*np.sin(phi), mu], axis=-1)
    axis = np.array([np.sin(inc), 0., np.cos(inc)])
    center = offset[0]*np.array([np.cos(inc), 0., -np.sin(inc)]) + np.array([0., offset[1], 0.]) + offset[2]*axis
    r = pos-center
    distance = np.linalg.norm(r, axis=-1)
    b = bp/2*(3*np.einsum("...j,j->...",r,axis)[...,None]*r/distance[...,None]**5
              - axis/distance[...,None]**3)
    modulus = np.linalg.norm(b, axis=-1)
    np.testing.assert_allclose(raw.field_strength_megagauss, modulus.ravel(), rtol=1e-14)
    np.testing.assert_allclose(raw.field_ray_cosine, (b[...,2]/modulus).ravel(), atol=1e-14)
    np.testing.assert_allclose(np.exp(1j*raw.field_azimuth_rad),
                               np.exp(1j*np.arctan2(b[...,1],b[...,0]).ravel()), atol=1e-14)
    assert np.any(raw.field_ray_cosine < 0)
    rotated = dipole_surface_cells(bp, inclination_deg=60., offset_vector_radius=offset,
        n_mu=4, n_azimuth=8, n_field_bins=None, signed_geometry=True, sky_position_angle_deg=37.)
    np.testing.assert_array_equal(raw.field_strength_megagauss, rotated.field_strength_megagauss)
    np.testing.assert_allclose(rotated.field_azimuth_rad-raw.field_azimuth_rad, np.deg2rad(37.))


def test_polarized_geometry_cannot_be_silently_compressed():
    with pytest.raises(ValueError, match="n_field_bins=None"):
        dipole_surface_cells(.5, signed_geometry=True)
    with pytest.raises(ValueError, match="intensity-only"):
        field_binned_surface_cells(cells(), [.4, .6])


@pytest.mark.parametrize("change", [
    {"field_azimuth_rad":None}, {"field_ray_cosine":np.array([1.1,-.6])},
    {"ray_mu":np.array([0.,.7])}, {"projected_weight":np.array([.2,.2])},
    {"projected_weight":np.array([1.1,-.1])}, {"field_strength_megagauss":np.array([-.5,.5])},
    {"field_azimuth_rad":np.array([np.nan,0.])}, {"field_azimuth_rad":np.array([0.])},
    {"field_bounds_megagauss":(.6,.7)},
])
def test_bad_stokes_cells_fail_before_transfer(change):
    with pytest.raises(ValueError):
        _validate_stokes_cells(replace(cells(), **change))


@pytest.mark.parametrize("change", [
    {"return_stokes":True},
    {"return_stokes":True,"polarized_transfer":"full-stokes-iquv","field_geometry":"dipole"},
    {"return_stokes":True,"polarized_transfer":"full-stokes-iquv","disk_component_drift_angstrom":16.},
    {"sky_position_angle_deg":30.}, {"sky_position_angle_deg":np.inf},
])
def test_invalid_public_combinations_fail_before_atmosphere(change, monkeypatch):
    from wd_spectra.models import dah
    monkeypatch.setattr(dah, "compute_da", lambda *a, **k: pytest.fail("started atmosphere"))
    with pytest.raises(ValueError):
        compute_dah(DAHConfig(**change))


@pytest.mark.parametrize("chi,expected", [
    (0., [1.,-2.,-3.]), (np.pi/4, [2.,1.,-3.]), (np.pi/2, [-1.,2.,-3.]),
])
def test_native_conversion_and_observer_basis(chi, expected):
    actual = np.array(_observer_polarization(1.,2.,3.,chi))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert np.linalg.norm(actual) == pytest.approx(np.sqrt(14))


def test_optical_output_matches_independent_standard_matrix_including_U():
    # Reference uses red-minus-blue etaV/rhoV (Piskunov & Kochukhov Eq. 7),
    # and their propagation-matrix signs. It does not use the conversion helper.
    k = np.array([[1,.3,0,-.1],[.3,1,-.7,0],[0,.7,1,.4],[-.1,0,-.4,1.]])
    mass = np.linspace(0.,40.,12)
    mu, a, b = .63, 1., .1
    source = (a+b*mass)[None,:]
    one = np.ones_like(source)
    emission = np.stack([source,.3*source,np.zeros_like(source),.1*source],axis=-1)
    native = emergent_stokes_specific_intensity(mass,one,.3*one,.1*one,.4*one,.7*one,
        source,mu,emission_stokes=emission,formal_solver="delo-linear")
    optical = np.r_[native.i[0],np.array(_observer_polarization(native.q,native.u,native.v,0.))[:,0]]
    gradient = np.array([b,0.,0.,0.])
    correction = mu*np.linalg.solve(k,gradient)
    expected = np.array([a,0.,0.,0.])+correction-expm(-k*mass[-1]/mu)@correction
    np.testing.assert_allclose(optical,expected,atol=1e-12,rtol=1e-12)
    assert abs(expected[2]) > .001  # this test detects flipping V alone


@pytest.mark.parametrize("field,wave", [(.5,np.arange(4800.,4931.,2.)), (300.,np.arange(4000.,4201.,4.))])
def test_real_transfer_signed_cancellation_and_legacy_I(atmosphere,field,wave):
    geometry = cells(field=field)
    polarized = synth(atmosphere,geometry,wavelength=wave)
    legacy = synth(atmosphere,geometry,output=False,wavelength=wave)
    assert isinstance(polarized,StokesSpectrum) and type(legacy) is Spectrum
    np.testing.assert_array_equal(polarized.surface_flux_lambda,legacy.surface_flux_lambda)
    np.testing.assert_allclose(polarized.surface_flux_v_lambda,0.,atol=1e-12*max(polarized.surface_flux_lambda))
    folded = synth(atmosphere,replace(geometry,field_ray_cosine=abs(geometry.field_ray_cosine)),wavelength=wave)
    assert np.max(abs(folded.surface_flux_v_lambda/polarized.surface_flux_lambda)) > 1e-8
    p = np.stack([polarized.surface_flux_q_lambda,polarized.surface_flux_u_lambda,polarized.surface_flux_v_lambda])
    assert np.all(np.linalg.norm(p,axis=0) <= polarized.surface_flux_lambda)


def test_isotropic_directions_cancel_all_polarization(atmosphere):
    geometry = uniform_field_surface_cells(.5,signed_geometry=True)
    polarized = synth(atmosphere,geometry,wavelength=np.arange(4800.,4931.,5.))
    for f in (polarized.surface_flux_q_lambda,polarized.surface_flux_u_lambda,polarized.surface_flux_v_lambda):
        assert max(abs(f/polarized.surface_flux_lambda)) < 1e-12


def test_integrates_fluxes_after_rotating_each_cell(atmosphere):
    # Unequal weights, angles and limb cosines prohibit averaging fractions
    # or rotating only once after summing native Q/U.
    geometry = replace(cells((.4,-.8),(0.,np.pi/4)),ray_mu=np.array([.3,.9]),projected_weight=np.array([.2,.8]))
    wave=np.arange(4800.,4931.,5.)
    total = synth(atmosphere,geometry,wavelength=wave)
    fields=("surface_flux_lambda","surface_flux_q_lambda","surface_flux_u_lambda","surface_flux_v_lambda")
    expected=np.zeros((4,wave.size))
    for j in range(2):
        one = SurfaceCells(geometry.field_strength_megagauss[j:j+1],geometry.field_ray_cosine[j:j+1],
            geometry.ray_mu[j:j+1],np.ones(1),field_azimuth_rad=geometry.field_azimuth_rad[j:j+1])
        spec=synth(atmosphere,one,wavelength=wave)
        expected += geometry.projected_weight[j]*np.array([getattr(spec,f) for f in fields])
    np.testing.assert_allclose(np.array([getattr(total,f) for f in fields]),expected,rtol=1e-12,atol=1e-8)
    beta=.37
    rotated=synth(atmosphere,replace(geometry,field_azimuth_rad=geometry.field_azimuth_rad+beta),wavelength=wave)
    c,s=np.cos(2*beta),np.sin(2*beta)
    np.testing.assert_array_equal(rotated.surface_flux_lambda,total.surface_flux_lambda)
    np.testing.assert_allclose(rotated.surface_flux_q_lambda,c*total.surface_flux_q_lambda-s*total.surface_flux_u_lambda,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(rotated.surface_flux_u_lambda,s*total.surface_flux_q_lambda+c*total.surface_flux_u_lambda,rtol=1e-12,atol=1e-8)
    np.testing.assert_array_equal(rotated.surface_flux_v_lambda,total.surface_flux_v_lambda)


def test_pole_on_axial_dipole_has_zero_disk_linear_polarization(atmosphere):
    geometry=dipole_surface_cells(.5,inclination_deg=0.,n_mu=2,n_azimuth=4,
                                  n_field_bins=None,signed_geometry=True)
    spec=synth(atmosphere,geometry,wavelength=np.arange(4800.,4931.,5.))
    assert max(abs(spec.surface_flux_q_lambda/spec.surface_flux_lambda)) < 1e-12
    assert max(abs(spec.surface_flux_u_lambda/spec.surface_flux_lambda)) < 1e-12


@pytest.mark.parametrize("field", [0., .5])
def test_public_dispatch_and_saved_output(atmosphere,tmp_path,field):
    config=DAHConfig(effective_temperature=15000.,logg=8.,magnetic_field_megagauss=field,field_angle_deg=50.,
        polarized_transfer="full-stokes-iquv",return_stokes=True,sky_position_angle_deg=37.)
    with pytest.warns(AtmosphereConvergenceWarning):
        result=compute_dah(config,np.arange(4800.,4931.,5.),initial_atmosphere=atmosphere,relax_atmosphere=False)
    assert isinstance(result.spectrum,StokesSpectrum)
    if field == 0.:
        for name in ("q","u","v"):
            np.testing.assert_array_equal(getattr(result.spectrum,"surface_flux_"+name+"_lambda"),0.)
    else:
        assert max(abs(result.spectrum.surface_flux_v_lambda/result.spectrum.surface_flux_lambda)) > 1e-8
    save_model_result(result,tmp_path)
    disk=np.loadtxt(tmp_path/"stokes.txt")
    assert disk.shape==(27,5)
    np.testing.assert_array_equal(disk[:,1],result.spectrum.surface_flux_lambda)
    np.testing.assert_array_equal(np.loadtxt(tmp_path/"spectrum.txt")[:,1],disk[:,1])
    assert "clockwise" in json.loads((tmp_path/"metadata.json").read_text())["spectrum_metadata"]["stokes_convention"]
    assert "signed-sky-stokes-optical-v1" in result.metadata["model_request_fingerprint"]["physics_revision"]
    from wd_spectra.models.automatic import ModelRun
    run = ModelRun(tmp_path, select_physics(config), tmp_path/"spectrum.txt", False)
    loaded = run.spectrum
    assert isinstance(loaded, StokesSpectrum)
    for name in ("q","u","v"):
        attr="surface_flux_"+name+"_lambda"
        np.testing.assert_array_equal(getattr(loaded,attr),getattr(result.spectrum,attr))
    assert loaded.metadata["polarimetry_validated"] is False
    disk[0,0] += 1.
    np.savetxt(tmp_path/"stokes.txt",disk)
    with pytest.raises(ValueError,match="match"):
        run.spectrum


def test_cli_forwards_stokes_geometry(monkeypatch,tmp_path):
    from wd_spectra.models import cli
    def stop(config,*args,**kwargs):
        assert config.return_stokes and config.disk_field_bins is None
        assert config.sky_position_angle_deg==37.
        assert config.polarized_transfer=="full-stokes-iquv"
        raise RuntimeError("configuration captured")
    monkeypatch.setattr(cli,"run_model",stop)
    monkeypatch.setattr(sys,"argv",["one_shot_dah.py","--geometry","dipole","--return-stokes",
        "--transfer","full-stokes-iquv","--sky-position-angle","37","--output",str(tmp_path)])
    with pytest.raises(RuntimeError,match="configuration captured"):
        cli.one_shot_main("DAH")
