"""MALI preconditioning of the restricted hot H/He atom changes the path, not the fixed point."""
from dataclasses import replace

import numpy as np
import pytest

from wd_spectra import helium_nlte as he, multilevel_nlte as h
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import population_arrays, transfer_field
from wd_spectra.nlte import _profile_averaged_mean_intensity_nu
from wd_spectra._hot_mali import HotMALI
from test_hot_nlte import atom  # noqa: F401  (fixture)


def _population_grid_and_fields(model, atmosphere, state):
    """The wavelength grid and line fields exactly as solve_populations forms them."""
    groups = model._line_problems(atmosphere)
    grids = [he.default_neutral_helium_continuum_wavelength(model.helium_i_atom),
             he.default_helium_ii_continuum_wavelength(model.maximum_helium_ii_level),
             h._default_continuum_wavelength(model.maximum_hydrogen_level)]
    grids.extend(p.continuum.wavelength_angstrom for group in groups
                 for problems in group.values() for p in problems)
    wave = np.unique(np.concatenate(grids))
    coefficients = model.transfer_coefficients(atmosphere, wave, state)
    _, field, _ = transfer_field(atmosphere, coefficients, n_angle=model.n_angle, check_source=False)
    fields = [{key: np.average([
        _profile_averaged_mean_intensity_nu(
            p.continuum.wavelength_angstrom, p.lte_line_opacity,
            field.mean_intensity[np.searchsorted(wave, p.continuum.wavelength_angstrom)])
        for p in problems], axis=0, weights=[p.line.absorption_oscillator_strength for p in problems])
        for key, problems in group.items()} for group in groups]
    return groups, wave, coefficients, field.mean_intensity, fields


@pytest.fixture
def sdb(atom):  # noqa: F811
    model = replace(atom, log_hydrogen_to_helium=2.5)
    atmosphere = gray_hydrogen_helium_atmosphere(30000., 5.5, 2.5, n_depth=8)
    return model, atmosphere


def test_mali_preserves_the_statistical_equilibrium_solution(sdb):
    model, atmosphere = sdb
    groups, wave, coefficients, mean, fields = _population_grid_and_fields(
        model, atmosphere, model._rate_state(atmosphere))
    # The unpreconditioned solution for this radiation field ...
    solution = model._rate_state(atmosphere, wave, mean, *fields)
    accelerator = HotMALI(model, atmosphere, wave, groups)
    # ... must also solve the preconditioned rates built from it.
    preconditioned = accelerator.state(solution, coefficients, mean, fields)
    assert max(float(np.max(v)) for v in accelerator.last_operators.values()) > 0.1
    expected, _ = population_arrays(solution)
    actual, _ = population_arrays(preconditioned)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_mali_converges_to_a_fixed_point_of_the_unpreconditioned_update(sdb):
    # The plain iteration creeps through the optically thick transitions: after
    # 1000 iterations its per-iteration change is still above 1e-5 although it
    # is far from the solution.  MALI must converge quickly to populations that
    # one unpreconditioned statistical-equilibrium update leaves unchanged.
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=300, population_tolerance=1e-6)
    accelerated = model.solve_populations(atmosphere, accelerated_lambda=True)
    assert accelerated.converged and accelerated.iterations < 200
    groups, wave, _, mean, fields = _population_grid_and_fields(model, atmosphere, accelerated)
    update = model._rate_state(atmosphere, wave, mean, *fields)
    current, reference = population_arrays(accelerated)
    following, _ = population_arrays(update)
    change = np.abs(following - current) / np.maximum(following, 1e-12 * reference.sum(axis=1)[:, None])
    assert np.max(change) < 1e-5


def test_hydrogen_rate_matrix_identity_transform_is_a_no_op(sdb):
    model, atmosphere = sdb
    kwargs = dict(maximum_level=model.maximum_hydrogen_level)
    plain = h.solve_multilevel_hydrogen_statistical_equilibrium(atmosphere, model.collision_data, **kwargs)
    transformed = h.solve_multilevel_hydrogen_statistical_equilibrium(
        atmosphere, model.collision_data, **kwargs, _rate_matrix_transform=lambda rate: rate)
    np.testing.assert_array_equal(transformed.population_density, plain.population_density)
