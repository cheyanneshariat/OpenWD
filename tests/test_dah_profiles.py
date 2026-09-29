"""Independent physical invariants of the released DAH profile kernels."""

import numpy as np
import pytest

from wd_spectra.kurucz_griem import profile_area, raw_profile, stark_coefficient
from wd_spectra._compat import trapezoid
from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.magnetic import _stimulated_emission_ratio
from wd_spectra.magnetic_atomic import read_h2db_transition_database
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import BALMER_LINES


def test_original_knm_balmer_column():
    # Visually checked against the printed KNMTAB, p.243 of Kurucz (1970).
    np.testing.assert_allclose([stark_coefficient(n) for n in range(3, 8)],
                               [.0125, .0177, .026, .0348, .0493], rtol=0, atol=0)


@pytest.mark.parametrize("n", [3, 4, 6, 12, 22])
def test_symmetric_positive_profile_and_analytic_limits(n):
    nu0 = 3.28805e15 * (.25 - 1/n**2)
    delta = np.r_[0., np.geomspace(1e5, 1e20, 1000)]
    shape = raw_profile(delta, 15000., 1e16, n, nu0)
    assert np.all(np.isfinite(shape)) and np.all(shape > 0)
    np.testing.assert_array_equal(shape, raw_profile(-delta, 15000., 1e16, n, nu0))
    # At zero detuning Q=1, I=0 and p(0)=0.1; core scales as ne^(-2/3).
    assert float(raw_profile(0., 15000., 8e16, n, nu0)) / shape[0] == pytest.approx(.25)
    # Holtsmark far wing, with vanishing impact/far-wing corrections.
    slope = np.log(shape[-1] / shape[-2]) / np.log(delta[-1] / delta[-2])
    assert slope == pytest.approx(-2.5, abs=2e-5)


@pytest.mark.parametrize("n,ne", [(3,1e12),(4,1e16),(6,1e18),(12,1e20),(22,1e16)])
def test_area_normalization_against_independent_dense_frequency_integral(n, ne):
    nu0 = 3.28805e15 * (.25 - 1/n**2)
    scale = nu0**2 * 1.25e-9 * ne**(2/3) * stark_coefficient(n) / 2.997925e18
    beta = np.r_[0., np.geomspace(1e-8,20.,30000), np.geomspace(20.00000001,1e10,30000)]
    delta = scale * beta
    sampled_area = 2 * trapezoid(raw_profile(delta, 15000., ne, n, nu0), delta)
    assert sampled_area / profile_area(15000., ne, n, nu0) == pytest.approx(1., abs=2e-5)


def test_template_support_and_frequency_integrated_lte_strength():
    from wd_spectra.kurucz_griem import kurucz_griem_templates as kurucz_templates
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, PLANCK
    from wd_spectra.opacity import _atmosphere_level_distribution

    atm = gray_hydrogen_atmosphere(15000., 8., n_depth=8, correlated_microfields=True)
    template = kurucz_templates(atm, maximum_upper_level=3)[3]
    wave = template.wavelength_angstrom
    assert wave[0] == pytest.approx(900.)
    assert wave[-1] == pytest.approx(25000.)
    assert np.all(np.diff(wave) > 0)
    population = _atmosphere_level_distribution(atm, 40)
    alpha = BALMER_LINES[0]
    nu0 = LIGHT_SPEED / (alpha.wavelength_vacuum_angstrom * 1e-8)
    expected = (np.pi * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
                * alpha.absorption_oscillator_strength * population.population_density[:,1]
                * np.clip(population.occupation_probability[:,2] / population.occupation_probability[:,1],0,1)
                / atm.mass_density * -np.expm1(-PLANCK*nu0/(BOLTZMANN*atm.temperature)))
    area = trapezoid(template.mass_absorption_coefficient[::-1],
                     LIGHT_SPEED/(wave[::-1]*1e-8), axis=0)
    # Optically thin layers keep effectively the whole line inside the
    # template; dense, broad far wings may legitimately leave its support.
    thin = atm.rosseland_optical_depth < .1
    np.testing.assert_allclose(area[thin], expected[thin], rtol=8e-4)


@pytest.mark.parametrize("field", [0.1, 6.13, 50., 100., 437.1])
def test_complete_component_opacity_conserves_parent_line_area(field):
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.magnetic import BalmerLineTemplate, h2db_balmer_manifolds

    atm = gray_hydrogen_atmosphere(15000., 8., n_depth=8)
    db = read_h2db_transition_database(ModelData.default().h2db_balmer_subset)
    line = BALMER_LINES[0]
    nu0 = LIGHT_SPEED / (line.wavelength_vacuum_angstrom * 1e-8)
    # A compact, unit-area kernel makes the integrated strength independent
    # of the Stark formula and avoids truncating its physical far wings.
    frequency = np.linspace(nu0 - 2e13, nu0 + 2e13, 4001)
    values = np.maximum(1 - np.abs(frequency - nu0) / 1e13, 0.) / 1e13
    template = BalmerLineTemplate(LIGHT_SPEED / frequency[::-1] / 1e-8,
                                  np.repeat(values[::-1, None], atm.n_depth, axis=1))
    components = db.balmer_components(3, line.wavelength_vacuum_angstrom, field, atm.temperature)
    centers = LIGHT_SPEED / (components.wavelength_angstrom * 1e-8)
    grid = np.linspace(centers.min() - 2e13, centers.max() + 2e13, 50001)
    wave = LIGHT_SPEED / grid[::-1] / 1e-8
    arguments = dict(templates={3: template}, maximum_upper_level=3)
    ordinary = h2db_balmer_manifolds(atm, wave, field, db, **arguments).isotropic()
    normalized = h2db_balmer_manifolds(atm, wave, field, db,
                                      normalize_line_strength=True, **arguments).isotropic()
    np.testing.assert_allclose(trapezoid(normalized[::-1], grid, axis=0), 1., rtol=2e-5)
    # An explicit option must not change subsequent calls or the database.
    again = h2db_balmer_manifolds(atm, wave, field, db, **arguments).isotropic()
    np.testing.assert_array_equal(ordinary, again)


def test_public_default_and_legacy_low_level_defaults_are_explicit():
    from wd_spectra import DAHConfig
    from wd_spectra.magnetic import MagneticPhysics

    config = DAHConfig()
    assert config.atmosphere_structure == "nonmagnetic"
    assert config.balmer_profile == "kurucz-griem"
    assert config.normalize_balmer_strength
    assert config.polarized_transfer == "scalar-stokes-i"
    assert not any((config.include_magnetic_eos, config.include_rwa_photoionization,
                    config.include_centered_motion, config.include_cyclotron_absorption))
    legacy = MagneticPhysics(None, None)
    assert legacy.balmer_profile == "unified"
    assert not legacy.normalize_balmer_strength
