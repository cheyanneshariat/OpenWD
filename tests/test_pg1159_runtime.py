"""Bounded real-atom checks, enabled when the external PG atomic data is installed."""
from dataclasses import replace
import numpy as np
import pytest
from wd_spectra.models import ModelData
from wd_spectra.models.pg1159 import required_atomic_files
from wd_spectra._pg1159_builder import build_model, structure_wavelength
from wd_spectra._pg1159_reference import planck_state
from wd_spectra.atmosphere import helium_continuum_atmosphere


def test_damped_real_atom_update_cannot_claim_rate_convergence():
    data = ModelData.default()
    if any(not path.is_file() for path in required_atomic_files(data, 'extended54-complete')):
        pytest.skip('external PG 1159 atomic data is not installed')
    model, _ = build_model(data, {'He': .52, 'C': .45, 'O': .03},
        structure_only=True, population_iterations=1,
        oxygen_atom_preset='extended54-complete')
    model = replace(model, population_transfer='mass', population_nlte_fraction=.1,
        metal_population_damping=1e-8, metal_population_minimum_damping=1e-8,
        helium_population_damping=1e-8, metal_population_acceleration_depth=0,
        coupled_population_acceleration_depth=0, use_population_ali=False,
        metal_population_relative_tolerance=1e-4)
    atmosphere = helium_continuum_atmosphere(110000., 7., n_depth=8,
        rosseland_frequency_points=120, tau_max=1000.)
    atmosphere, initial = planck_state(model, atmosphere, structure_wavelength(model, 120))
    result = model.solve_populations(atmosphere, initial)
    assert result.metadata['metal_population_relative_change'] < 1e-4
    assert result.metadata['undamped_population_relative_change'] > 1e-4
    assert not result.converged
    # A single-pass rate/tangent probe must return g(x), never the input x,
    # even when a deliberately loose test tolerance marks its map as small.
    probe = replace(model, metal_population_damping=1., helium_population_damping=1.,
        metal_population_relative_tolerance=1.).solve_populations(atmosphere, initial)
    assert not np.allclose(probe.carbon_level_state.population_density,
        initial.carbon_level_state.population_density, rtol=1e-6, atol=0.)


@pytest.mark.parametrize("handoff", [False, True])
def test_measured_state_is_returned_before_extrapolation(monkeypatch, handoff):
    """Do not let an accelerator move away from an already closed rate state."""
    import wd_spectra.pg1159 as pg
    data = ModelData.default()
    if any(not path.is_file() for path in required_atomic_files(data, 'extended54-complete')):
        pytest.skip('external PG 1159 atomic data is not installed')
    model, _ = build_model(data, {'He': .52, 'C': .45, 'O': .03},
        structure_only=True, population_iterations=3,
        oxygen_atom_preset='extended54-complete')
    model = replace(model, population_transfer='mass', population_nlte_fraction=.1 if handoff else 0.,
        metal_population_minimum_iterations=1,
        coupled_population_acceleration_depth=20, metal_population_acceleration_depth=0,
        metal_population_relative_tolerance=1e-4, use_population_ali=False)
    atmosphere = helium_continuum_atmosphere(110000., 7., n_depth=8,
        rosseland_frequency_points=120, tau_max=1000.)
    atmosphere, initial = planck_state(model, atmosphere, structure_wavelength(model, 120))
    def bad_extrapolation(x, g, old_he, new_he, history, depth, **kwargs):
        return x + .2 * (np.arange(len(x)) % 3), new_he
    monkeypatch.setattr(pg, '_coupled_population_update', bad_extrapolation)
    seen = []
    def refresh(iteration, measured):
        seen.append(measured)
        return True
    result = model.solve_populations(atmosphere, initial,
        _stop_iteration=refresh if handoff else None)
    if handoff:
        assert len(seen) == 1
        assert not result.converged
        assert result.metadata['undamped_population_relative_change'] > 1e-4
        np.testing.assert_allclose(seen[0].carbon_level_state.population_density,
            initial.carbon_level_state.population_density, rtol=1e-13)
    else:
        assert result.metadata['undamped_population_relative_change'] < 1e-4
        assert result.converged
    np.testing.assert_allclose(result.carbon_level_state.population_density,
        initial.carbon_level_state.population_density, rtol=1e-13)
    np.testing.assert_allclose(result.oxygen_level_state.population_density,
        initial.oxygen_level_state.population_density, rtol=1e-13)


def test_iteration_limit_returns_last_measured_input(monkeypatch):
    """An unfinished Anderson proposal must not become the restart state."""
    import wd_spectra.pg1159 as pg
    data = ModelData.default()
    if any(not path.is_file() for path in required_atomic_files(data, 'extended54-complete')):
        pytest.skip('external PG 1159 atomic data is not installed')
    model, _ = build_model(data, {'He': .52, 'C': .45, 'O': .03},
        structure_only=True, population_iterations=2,
        oxygen_atom_preset='extended54-complete')
    model = replace(model, population_transfer='mass', population_nlte_fraction=.1,
        metal_population_minimum_iterations=1,
        coupled_population_acceleration_depth=20, metal_population_acceleration_depth=0,
        metal_population_relative_tolerance=1e-30, use_population_ali=False)
    atmosphere = helium_continuum_atmosphere(110000., 7., n_depth=8,
        rosseland_frequency_points=120, tau_max=1000.)
    atmosphere, initial = planck_state(model, atmosphere, structure_wavelength(model, 120))
    measured_inputs = []
    measured_helium = []
    def bad_extrapolation(x, g, old_he, new_he, history, depth, **kwargs):
        measured_inputs.append(x.copy())
        measured_helium.append(old_he)
        return x + .2 * (np.arange(len(x)) % 3), new_he
    monkeypatch.setattr(pg, '_coupled_population_update', bad_extrapolation)

    result = model.solve_populations(atmosphere, initial)

    returned = pg._pack_population_vector(
        result.light_metal_state,
        result.carbon_level_state,
        result.oxygen_level_state,
        result.metal_state,
    )
    np.testing.assert_allclose(returned, measured_inputs[-1], rtol=0., atol=0.)
    np.testing.assert_allclose(
        result.helium_state.neutral_population_density,
        measured_helium[-1].neutral_population_density,
        rtol=0.,
        atol=0.,
    )
    assert result.metadata['iteration_limit_returned_measured_input']
    assert not result.converged
