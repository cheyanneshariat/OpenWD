"""Public adapter integration with a compact synthetic collision fixture.

Quick cases run the real bounded nonlinear solver. Standard cases replace
the expensive LTE/NLTE relaxations while checking their adapter contracts.
All cases read the bundled public-default profiles and synthesize real flux.
These are interface checks, not physical convergence certificates.
"""
import json
import numpy as np
import pytest

from wd_spectra import DOConfig, DAOConfig, compute_do, compute_dao, save_model_result
from wd_spectra.models import hot
from wd_spectra.models.common import ModelData, NumericalResolution, AtmosphereConvergenceWarning
from wd_spectra.hot_nlte import HotAtmosphereResult
from test_helium_nlte import _write_small_ccc_archive


@pytest.fixture
def public_data(tmp_path,monkeypatch):
    data=ModelData(tmp_path)
    data.ccc_hydrogen_collisions.parent.mkdir(parents=True)
    _write_small_ccc_archive(data.ccc_hydrogen_collisions,maximum_level=8)
    for path in (data.tlusty_source,data.tlusty_helium_atom):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('synthetic external-reader fixture\n')
    bundled=ModelData.default()
    for name in ('Tremblay26.txt','he2prf.dat'):
        target=data.cache/'helium-stark'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(bundled.cache/'helium-stark'/name)
    # Keep this integration test small while retaining real CCC loading,
    # model construction, profiles, transfer and saving.
    monkeypatch.setattr(hot,'read_tlusty_helium_collision_data',lambda *args:None)
    return data


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('quality',['quick','standard'])
def test_public_hot_adapter_returns_saveable_spectrum(public_data,monkeypatch,tmp_path,mixed,quality):
    monkeypatch.setattr(hot,'numerical_resolution',lambda quality:NumericalResolution(4,40,1,1))
    config=(DAOConfig(quality=quality,maximum_helium_ii_level=3,maximum_hydrogen_level=3)
            if mixed else DOConfig(quality=quality,maximum_helium_ii_level=3))
    calls={};events=[]
    seed_name='hydrogen_helium_continuum_atmosphere' if mixed else 'helium_continuum_atmosphere'
    seed_function=getattr(hot,seed_name)
    def seed(*args,**kwargs):
        calls['seed']=seed_function(*args,**kwargs)
        calls['seed_args']=args;calls['seed_kwargs']=kwargs
        return calls['seed']
    monkeypatch.setattr(hot,seed_name,seed)
    solve=hot.solve_hot_nlte_atmosphere
    def measured_solve(a,model,wave,**kwargs):
        calls['model']=model
        if quality=='quick':return solve(a,model,wave,**kwargs)
        assert a is calls['seed']
        kwargs['iteration_callback'](1,a,{'solver_phase':'simultaneous-hot-nlte'})
        return HotAtmosphereResult(a,model._rate_state(a),None)
    monkeypatch.setattr(hot,'solve_hot_nlte_atmosphere',measured_solve)
    def lte(*args,**kwargs):
        calls['lte_kwargs']=kwargs
        kwargs['iteration_callback'](1,calls['seed'],{})
        return calls['seed']
    monkeypatch.setattr('wd_spectra.atmosphere.radiative_equilibrium_helium_atmosphere',lte)
    wave=np.linspace(4000.,7000.,31)
    with pytest.warns(AtmosphereConvergenceWarning):
        result=(compute_dao if mixed else compute_do)(config,wave,data=public_data,
            iteration_callback=lambda i,a,d:events.append(d))
    assert result.spectral_type==('DAO' if mixed else 'DO')
    assert result.metadata['atmosphere_convergence_status']!='converged'
    assert result.metadata['cold_start'] is True
    assert result.metadata['initialization']==dict(
        method='fresh continuum' if quality=='quick' else 'fresh LTE equilibrium',
        previous_model_supplied=False,stellar_parameters_fixed=True)
    assert result.metadata['ccc_maximum_level']==8
    assert result.metadata['helium_ii_collision_model']=='tlusty-mihalas'
    assert calls['model'].helium_i_stark_table.source_path.name=='Tremblay26.txt'
    assert calls['seed_kwargs']['tau_max']==(1000. if mixed else 100.)
    assert calls['seed_args']==((70000.,8.,2.) if mixed else (70000.,8.))
    if mixed:
        assert result.population_state.hydrogen.metadata['ccc_maximum_level']==8
        assert 'Mihalas' in result.metadata['hydrogen_collision_closure']
    if quality=='standard':
        assert calls['lte_kwargs']['log_hydrogen_abundance']==(2. if mixed else None)
        assert [d['solver_phase'] for d in events]==['lte-initialization','simultaneous-hot-nlte']
    else:
        assert 'lte_kwargs' not in calls
    expected=dict(flux_unit='erg s^-1 cm^-2 Angstrom^-1',
        flux_convention='surface F_lambda',wavelength_medium='vacuum')
    for key,value in expected.items():assert result.spectrum.metadata[key]==value
    np.testing.assert_array_equal(result.spectrum.wavelength_angstrom,wave)
    assert np.all(np.isfinite(result.spectrum.surface_flux_lambda))
    assert np.all(result.spectrum.surface_flux_lambda>0)
    output=tmp_path/'saved'
    save_model_result(result,output)
    saved=json.loads((output/'metadata.json').read_text())
    for key,value in expected.items():assert saved['spectrum_metadata'][key]==value
    assert saved['model_metadata']['ccc_maximum_level']==8
    assert (output/'populations.npz').is_file()


def test_missing_bundled_profile_is_not_called_external(public_data):
    (public_data.cache/'helium-stark/Tremblay26.txt').unlink()
    with pytest.raises(FileNotFoundError,match='required model data') as raised:
        compute_do(data=public_data)
    assert 'not bundled' not in str(raised.value)


def test_data_environment_and_unsupported_config(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENWD_DATA',str(tmp_path))
    assert ModelData.default().root==tmp_path
    from wd_spectra import select_physics
    with pytest.raises(TypeError,match='DAZConfig.*DQConfig.*DOConfig.*DAOConfig'):
        select_physics(object())


def test_hot_db_config_keeps_explicit_lte_selection(monkeypatch):
    from wd_spectra import DBConfig,select_physics
    monkeypatch.setattr('wd_spectra.models.selection._helium_probe',lambda *args:
        dict(electron_fraction=1.,reos_in_domain=False,density_fractional_change=None))
    assert select_physics(DBConfig(effective_temperature=70000.)).workflow=='db'
