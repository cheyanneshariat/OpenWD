"""The TLUSTY 24-term He I atom in the coupled helium solver (the 14-term default is unchanged)."""
from dataclasses import replace

import numpy as np
import pytest

from wd_spectra import DAOConfig, helium_nlte as he
from wd_spectra.atmosphere import gray_helium_atmosphere, gray_hydrogen_helium_atmosphere
from wd_spectra.helium import HELIUM_I_LINES
from wd_spectra.helium_collisions import read_tlusty_helium_collision_data
from wd_spectra.helium_i_atom import read_tlusty_helium_i_atom
from wd_spectra.hot_nlte import population_arrays
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
from wd_spectra._hot_mali import HotMALI
from test_hot_mali import _population_grid_and_fields


@pytest.fixture(scope="module")
def atom24():
    return read_tlusty_helium_i_atom(ModelData.default().tlusty_atoms / "he1.dat")


@pytest.fixture(scope="module")
def sdb24(atom24):
    data = ModelData.default()
    model = _model_from_config(DAOConfig(effective_temperature=30000., logg=5.5, log_hydrogen_to_helium=2.5,
                                         maximum_helium_ii_level=6, maximum_hydrogen_level=6), data)
    model = replace(model, helium_i_atom=atom24, n_angle=2,
                    helium_i_collision_data=read_tlusty_helium_collision_data(
                        data.tlusty_source, data.tlusty_atoms / "he1.dat"))
    template = gray_hydrogen_helium_atmosphere(30000., 5.5, 2.5, n_depth=8)
    return model, model.rebuild_atmosphere(template, template.temperature)


def test_atom_terms_and_collision_states(atom24):
    assert atom24.n_terms == 24 and atom24.n_fitted_collision_terms == 19
    # 5876 and 6678 have the NIST-like TLUSTY f-values of 2p-3d.
    assert atom24.oscillator_strength[(4, 9)] == pytest.approx(0.609)
    assert atom24.oscillator_strength[(5, 10)] == pytest.approx(0.711)


def test_observed_lines_map_to_resolved_terms(atom24):
    expected = {4471: (3, 14), 4922: (4, 15), 5877: (3, 8), 6678: (4, 9), 4026: (3, 19), 5016: (2, 10),
                3889: (1, 7), 4713: (3, 11), 7065: (3, 5), 7281: (4, 6), 4388: (4, 20)}
    mapped = {line.table_key_angstrom: he._helium_i_line_term_indices(line, atom24) for line in HELIUM_I_LINES}
    for key, terms in expected.items():
        assert mapped[key] == terms, key
    # The default atom keeps its superlevel mapping.
    legacy = {line.table_key_angstrom: he._helium_i_line_term_indices(line, he.HELIUM_I_14) for line in HELIUM_I_LINES}
    assert legacy[4471] == (3, 7) and legacy[6678] == (4, 6)


def test_planck_fields_recover_lte_exactly(sdb24):
    model, atmosphere = sdb24
    actual, reference = population_arrays(model._rate_state(atmosphere))
    assert actual.shape[1] == 24 + 6 + 1 + 6 + 1
    # He I and H recover LTE to roundoff; trace He II excited levels (~1e-9 of He)
    # and He III carry the linear-solve roundoff of the larger matrix.
    np.testing.assert_allclose(actual[:, :24] / reference[:, :24], 1.0, rtol=1e-10)
    np.testing.assert_allclose(actual[:, 31:] / reference[:, 31:], 1.0, rtol=1e-10)
    np.testing.assert_allclose(actual / reference, 1.0, rtol=1e-9)


def test_explicit_bound_free_is_consistent_with_lte_continuum(atom24):
    wavelength = he.default_neutral_helium_continuum_wavelength(atom24)
    for atmosphere in (gray_helium_atmosphere(60000., 8., n_depth=5),
                       gray_hydrogen_helium_atmosphere(25000., 5.2, 2.3, n_depth=5),
                       gray_hydrogen_helium_atmosphere(23200., 5.2, 2.27, n_depth=40)):
        problem = he._prepare_neutral_helium_continuum_transfer_problem(
            atmosphere, wavelength, helium_i_atom=atom24)
        assert problem.bound_free_coefficient.shape[-1] == 24


def test_mali_fixed_point_with_resolved_atom(sdb24):
    model, atmosphere = sdb24
    groups, wave, coefficients, mean, fields = _population_grid_and_fields(
        model, atmosphere, model._rate_state(atmosphere))
    solution = model._rate_state(atmosphere, wave, mean, *fields)
    preconditioned = HotMALI(model, atmosphere, wave, groups).state(solution, coefficients, mean, fields)
    np.testing.assert_allclose(population_arrays(preconditioned)[0], population_arrays(solution)[0], rtol=1e-8)


@pytest.mark.parametrize('teff', [16000., 12000.])
def test_dominant_conservation_row_recovers_lte_in_cool_layers(teff):
    # Cool sdB surface layers carry He III and high He II levels at <1e-18 of
    # the helium.  Replacing the dominant state's equation by particle
    # conservation keeps their own balance equations and recovers LTE exactly.
    data = ModelData.default()
    model = replace(_model_from_config(DAOConfig(effective_temperature=teff, logg=5.0, log_hydrogen_to_helium=3.0,
                                                 maximum_hydrogen_level=6), data),
                    helium_conservation_row="dominant")
    template = gray_hydrogen_helium_atmosphere(teff, 5.0, 3.0, n_depth=12)
    atmosphere = model.rebuild_atmosphere(template, template.temperature)
    actual, reference = population_arrays(model._rate_state(atmosphere))
    np.testing.assert_allclose(actual / reference, 1.0, rtol=1e-9)
