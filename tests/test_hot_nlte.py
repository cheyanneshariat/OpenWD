"""Atom, common-transfer and central-solver guards; no external fit claims."""
from dataclasses import replace
import numpy as np
import pytest

from wd_spectra import DOConfig, DAOConfig, select_physics
from wd_spectra.atmosphere import gray_helium_atmosphere, gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import (HotNLTEModel, population_arrays, with_departures,
    transfer_field, solve_hot_nlte_atmosphere)
from wd_spectra.models.hot import validate_config
from wd_spectra.models.common import ModelData
from wd_spectra.helium_stark import read_helium_stark_table
from wd_spectra.helium_ii_stark import read_helium_ii_stark_table
from wd_spectra.multilevel_nlte import read_ccc_hydrogen_collision_data
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.opacity import electron_scattering_mass_coefficient
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra._compat import trapezoid
from wd_spectra._mass_feautrier import mass_energy
from test_helium_nlte import _write_small_ccc_archive


@pytest.fixture
def atom(tmp_path):
    path = tmp_path/'ccc.zip'
    _write_small_ccc_archive(path, maximum_level=3)
    data = ModelData.default()
    return HotNLTEModel(read_ccc_hydrogen_collision_data(path, maximum_level=3),
        read_helium_stark_table(data.cache/'helium-stark'/'Tremblay26.txt'),
        read_helium_ii_stark_table(data.helium_ii_stark), None,
        maximum_helium_ii_level=3, maximum_hydrogen_level=3,
        population_maximum_iterations=1, n_angle=2)


@pytest.mark.parametrize('ratio', [None, -2., 2.])
def test_lte_limit_and_element_conservation(atom, ratio):
    atom = replace(atom, log_hydrogen_to_helium=ratio)
    a = (gray_helium_atmosphere(60000., 8., n_depth=5) if ratio is None else
         gray_hydrogen_helium_atmosphere(60000., 8., ratio, n_depth=5))
    state = atom._rate_state(a)
    actual, reference = population_arrays(state)
    np.testing.assert_allclose(actual/reference, 1., rtol=1e-10)
    perturbed = with_departures(state, np.exp(np.linspace(-1, 1, actual.size).reshape(actual.shape)))
    changed, _ = population_arrays(perturbed)
    for section in (slice(0, 18), slice(18, None)):
        np.testing.assert_allclose(changed[:, section].sum(axis=1), reference[:, section].sum(axis=1), rtol=2e-15)
    wave = np.geomspace(100, 20000, 70)
    c = atom.transfer_coefficients(a, wave, state, lines=False)
    np.testing.assert_allclose(c.thermal_emissivity/c.true_absorption,
        planck_lambda_angstrom(wave[:, None], a.temperature[None, :]), rtol=1e-10)


def test_dao_counts_shared_electrons_once(atom):
    atom = replace(atom, log_hydrogen_to_helium=2.)
    a = gray_hydrogen_helium_atmosphere(70000., 8., 2., n_depth=5)
    state = atom._rate_state(a)
    wave = np.geomspace(500, 15000, 50)
    c = atom.transfer_coefficients(a, wave, state)
    from wd_spectra import helium_nlte as he, multilevel_nlte as h
    hc = h.hydrogen_nlte_transfer_coefficients(a, wave, state.hydrogen,
        maximum_level=3, include_brackett=False)
    hec = he.coupled_helium_nlte_transfer_coefficients(a, wave, state.helium,
        helium_i_stark_table=atom.helium_i_stark_table,
        helium_ii_stark_table=atom.helium_ii_stark_table)
    electron = electron_scattering_mass_coefficient(a)[None, :]
    np.testing.assert_allclose(c.scattering, hc.scattering+hec.scattering-electron, rtol=1e-14)
    assert np.all(c.scattering >= electron*(1-1e-14))


def test_nlte_transfer_conserves_cell_energy_and_closes_scattering():
    a = gray_helium_atmosphere(60000., 8., n_depth=9, tau_min=1e-6, tau_max=10)
    wave = np.geomspace(100, 20000, 100)
    b = planck_lambda_angstrom(wave[:, None], a.temperature[None, :])
    absorption = np.full_like(b, .2)
    emissivity = absorption*b*np.linspace(.7, 1.3, a.n_depth)
    scattering = np.full_like(b, .6)
    c = NLTETransferCoefficients(wave, absorption, emissivity, scattering, {})
    _, field, error = transfer_field(a, c, n_angle=3)
    assert error < 1e-12
    source_fast, fast, unchecked = transfer_field(a, c, n_angle=3, check_source=False, wavelength_chunk_size=7)
    source_checked, checked, checked_error = transfer_field(a, c, n_angle=3, wavelength_chunk_size=7)
    assert unchecked is None and checked_error < 1e-12
    np.testing.assert_array_equal(source_fast, source_checked)
    for name in ('mean_intensity', 'flux', 'interface_flux'):
        np.testing.assert_allclose(getattr(fast, name), getattr(field, name), rtol=2e-14)
        np.testing.assert_array_equal(getattr(fast, name), getattr(checked, name))
    energy, _ = mass_energy(wave, a.column_mass, emissivity/absorption, field.mean_intensity, absorption)
    # Difference each wavelength before summing: subtracting two integrated
    # fluxes discards the small outer-cell heating in platform-dependent
    # accumulation roundoff. The conservation tolerance is unchanged.
    divergence = trapezoid(np.diff(field.interface_flux, axis=1), wave, axis=0)
    np.testing.assert_allclose(divergence, energy, rtol=1e-10, atol=1e-4)


def test_public_configs_and_selection():
    for config, workflow in ((DOConfig(), 'do'), (DAOConfig(), 'dao')):
        validate_config(config)
        selected = select_physics(config)
        assert selected.workflow == workflow and selected.experimental
    for kwargs in ({'population_tolerance': np.nan}, {'maximum_helium_ii_level': True},
                   {'population_maximum_iterations': 0}, {'effective_temperature': -1}):
        with pytest.raises(ValueError):
            validate_config(DOConfig(**kwargs))
    with pytest.raises(ValueError):
        validate_config(DAOConfig(log_hydrogen_to_helium=np.nan))
    validate_config(DAOConfig(maximum_hydrogen_level=20))
    with pytest.raises(ValueError):
        validate_config(DAOConfig(maximum_hydrogen_level=21))


def test_expanded_hydrogen_atom_retains_planck_limit_and_conservation(atom):
    atom = replace(atom, maximum_hydrogen_level=20, log_hydrogen_to_helium=6.)
    a = gray_hydrogen_helium_atmosphere(40204., 7.82, 6., n_depth=5)
    state = atom._rate_state(a)
    assert state.hydrogen.metadata['ccc_maximum_level']==atom.collision_data.maximum_level
    assert state.hydrogen.metadata['merged_collision_closure'] is None
    assert 'Mihalas' in state.hydrogen.metadata['collision_closure']
    actual, reference = population_arrays(state)
    np.testing.assert_allclose(actual/reference, 1., rtol=2e-9)
    nhe = 15 + atom.maximum_helium_ii_level
    assert actual.shape == (5, nhe + 21)
    perturbed = with_departures(state, np.exp(np.linspace(-1, 1, actual.size).reshape(actual.shape)))
    changed, _ = population_arrays(perturbed)
    for section in (slice(0, nhe), slice(nhe, None)):
        np.testing.assert_allclose(changed[:, section].sum(axis=1), reference[:, section].sum(axis=1), rtol=3e-15)
    wave = np.geomspace(100., 30000., 100)
    coefficients = atom.transfer_coefficients(a, wave, state)
    np.testing.assert_allclose(coefficients.thermal_emissivity/coefficients.true_absorption,
        planck_lambda_angstrom(wave[:, None], a.temperature[None, :]), rtol=2e-9)


def test_population_exhaustion_is_not_convergence(atom):
    a = gray_helium_atmosphere(60000., 8., n_depth=4)
    state = atom.solve_populations(a)
    assert state.iterations == 1
    assert not state.converged
    assert np.isfinite(state.maximum_relative_population_change)
    # One exhausted iteration must return the checked initial state, not
    # an unmeasured last population update with the initial defect attached.
    actual, _ = population_arrays(state)
    initial, _ = population_arrays(atom._rate_state(a))
    np.testing.assert_array_equal(actual, initial)


def test_fixed_material_cache_preserves_coefficients_and_rejects_other_state(atom):
    from wd_spectra.nlte_core import _FixedTransferCache
    a = gray_helium_atmosphere(60000., 8., n_depth=4)
    state = atom._rate_state(a)
    actual, _ = population_arrays(state)
    state = with_departures(state, np.full_like(actual, 1.1))
    wave = np.geomspace(100, 15000, 80)
    cache = _FixedTransferCache(a, wave)
    direct = atom.transfer_coefficients(a, wave, state)
    cached = atom.transfer_coefficients(a, wave, state, _cache=cache)
    repeated = atom.transfer_coefficients(a, wave, state, _cache=cache)
    for name in ('true_absorption', 'thermal_emissivity', 'scattering'):
        np.testing.assert_array_equal(getattr(direct, name), getattr(cached, name))
        np.testing.assert_array_equal(getattr(direct, name), getattr(repeated, name))
    with pytest.raises(ValueError, match='different atmosphere/grid'):
        atom.transfer_coefficients(replace(a), wave, state, _cache=cache)


def test_common_opacity_covers_every_explicit_helium_rate_transition():
    from wd_spectra.helium_nlte import coupled_helium_ii_lines
    lines = coupled_helium_ii_lines(32)
    pairs = [(x.lower_principal_quantum_number, x.upper_principal_quantum_number) for x in lines]
    assert len(pairs) == len(set(pairs))
    assert {(i, j) for i in (1, 2, 3) for j in range(i+1, 33)} <= set(pairs)


@pytest.mark.parametrize('change',[.1,float('inf')])
def test_nested_populations_are_saved_without_pickle(atom, tmp_path,change):
    import json
    from wd_spectra import ModelResult, save_model_result
    from wd_spectra.hot_nlte import population_status
    from wd_spectra.spectrum import Spectrum
    a = gray_helium_atmosphere(60000., 8., n_depth=4)
    state = population_status(atom._rate_state(a), False, 1, change)
    wave = np.linspace(4000, 5000, 5)
    result = ModelResult('DO', a, Spectrum(wave, np.ones(5), {}), DOConfig(), {}, state)
    save_model_result(result, tmp_path)
    with np.load(tmp_path/'populations.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['helium.neutral_population_density'], state.helium.neutral_population_density)
        assert 'metadata_json' in saved
        metadata=json.loads(str(saved['metadata_json']))
        assert metadata['maximum_relative_population_change']==(change if np.isfinite(change) else None)


def test_incomplete_custom_data_error_lists_missing_bundled_inputs(tmp_path):
    from wd_spectra import compute_do
    with pytest.raises(FileNotFoundError, match='required model data are missing') as raised:
        compute_do(data=ModelData(tmp_path))
    message = str(raised.value)
    assert 'e-H_XSEC_LS.zip' in message
    assert 'tlusty200.f' in message
    assert 'he1_14lev.dat' in message


def test_low_level_atom_rejects_invalid_work_budget_and_cross_composition(atom):
    with pytest.raises(ValueError):
        replace(atom, population_maximum_iterations=0)
    a = gray_hydrogen_helium_atmosphere(60000., 8., 2., n_depth=4)
    with pytest.raises(ValueError, match='hydrogen-free'):
        atom.rebuild_atmosphere(a, a.temperature)
    he_atmosphere = gray_helium_atmosphere(60000., 8., n_depth=4)
    state = atom._rate_state(he_atmosphere)
    with pytest.raises(ValueError, match='compositions differ'):
        replace(atom, log_hydrogen_to_helium=2.).remap(a, state)


def test_temperature_rebuild_invalidates_seed_certificate(atom):
    a = gray_helium_atmosphere(60000., 8., n_depth=4)
    a = replace(a, metadata={'equilibrium_certificate': {'verified': True},
        'radiative_equilibrium_converged': True,
        'radiative_equilibrium_solver_converged': True,
        'model_request_fingerprint': {'old': True}})
    rebuilt = atom.rebuild_atmosphere(a, a.temperature*1.01)
    assert 'equilibrium_certificate' not in rebuilt.metadata
    assert 'model_request_fingerprint' not in rebuilt.metadata
    assert not rebuilt.metadata['radiative_equilibrium_converged']


def test_mixed_continuum_validates_the_sum_instead_of_each_species(atom):
    """A trace-He stimulated contribution can be negative in opaque hydrogen."""
    from wd_spectra import helium_nlte as he
    atom = replace(atom, log_hydrogen_to_helium=6.)
    a = gray_hydrogen_helium_atmosphere(40204., 7.82, 6., n_depth=4)
    state = atom._rate_state(a)
    actual, _ = population_arrays(state)
    departures = np.ones_like(actual)
    departures[:, :17] = .001
    state = with_departures(state, departures)
    wave = np.geomspace(100., 100000., 200)
    with pytest.raises(RuntimeError, match='non-positive total continuum'):
        he.coupled_helium_nlte_transfer_coefficients(a, wave, state.helium,
            helium_i_stark_table=atom.helium_i_stark_table,
            helium_ii_stark_table=atom.helium_ii_stark_table)
    common = atom.transfer_coefficients(a, wave, state)
    assert np.all(common.true_absorption > 0)
    _, field, _ = transfer_field(a, common, n_angle=2)
    assert np.all(np.isfinite(field.mean_intensity))
    with pytest.raises(ValueError, match='positive total extinction'):
        transfer_field(a, replace(common, true_absorption=-common.true_absorption))


def test_common_field_closes_with_signed_and_zero_net_absorption():
    a=gray_helium_atmosphere(50000.,8.,n_depth=4)
    a=replace(a,column_mass=np.array([.02,.1,.4,2.]))
    wave=np.array([1000.,4000.,9000.])
    absorption=np.broadcast_to(np.array([-.02,0.,.1,.2]),(3,4))
    eta=np.ones_like(absorption);scattering=np.full_like(absorption,.5)
    coefficients=NLTETransferCoefficients(wave,absorption,eta,scattering,{})
    source,field,error=transfer_field(a,coefficients,n_angle=3)
    assert error<1e-12
    assert np.all(source>0) and np.all(field.mean_intensity>0)


def test_optical_pickering_lines_share_the_rate_radiation_field(atom):
    atom = replace(atom, maximum_helium_ii_level=8)
    a = gray_helium_atmosphere(60000., 8., n_depth=3)
    _, ion, _ = atom._line_problems(a)
    # These optical lines already contributed opacity. A Planck rate fallback
    # for the same transitions disconnects their pumping from that opacity.
    assert {(4,5), (4,6), (4,7), (4,8)} <= set(ion)
    assert ion[(4,7)][0].line.wavelength_vacuum_angstrom == pytest.approx(5412.9,abs=.5)


def test_public_joint_driver_does_not_certify_failed_populations(atom,monkeypatch):
    import wd_spectra._hot_structure as joint
    shared=joint.solve_trust_region_newton
    calls=[]
    def spy(*args,**kwargs):
        calls.append(True)
        return shared(*args,**kwargs)
    monkeypatch.setattr(joint,'solve_trust_region_newton',spy)
    a=gray_helium_atmosphere(60000.,8.,n_depth=3)
    result=solve_hot_nlte_atmosphere(a,atom,np.geomspace(25,100000,80),maximum_iterations=1)
    assert calls==[True]
    assert not result.population_state.converged
    assert not result.atmosphere.metadata['equilibrium_certificate']['verified']
    assert not result.atmosphere.metadata['radiative_equilibrium_solver_converged']
