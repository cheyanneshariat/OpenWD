"""Warm starts must reproduce the declared material state and lose certificates."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from wd_spectra import DAOConfig
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import population_arrays, with_departures
from wd_spectra.models.common import ModelData
from wd_spectra.models.hot import _model_from_config
research = Path(__file__).resolve().parents[1]/'research'
sys.path.insert(0,str(research))
spec = importlib.util.spec_from_file_location('trace_checkpoint',research/'run_hot_trace_from_checkpoint.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_checkpoint_roundtrip_and_composition_guard(tmp_path):
    config = DAOConfig(52500.,7.53,5.,maximum_hydrogen_level=4,maximum_helium_ii_level=4)
    data = ModelData.default();model = _model_from_config(config,data)
    a = gray_hydrogen_helium_atmosphere(52500.,7.53,5.,n_depth=5,tau_max=10.)
    p = model._rate_state(a)
    old,ref = population_arrays(p)
    departure = np.ones_like(old);departure[:,0]=2.;departure[:,-2]=.7
    p = with_departures(p,departure)
    file = tmp_path/'checkpoint.npz'
    module.save_state(file,a,p,model)
    _,restored,q = module.load_seed(file,config,data)
    np.testing.assert_array_equal(a.temperature,restored.temperature)
    np.testing.assert_allclose(population_arrays(q)[0],population_arrays(p)[0],rtol=1e-13)
    assert not restored.metadata['radiative_equilibrium_converged']
    assert not q.converged
    with pytest.raises(AssertionError,match='EOS/composition/atom'):
        module.load_seed(file,replace(config,log_hydrogen_to_helium=4.),data)
    with pytest.raises(ValueError,match='maximum_hydrogen_level'):
        module.load_seed(file,replace(config,maximum_hydrogen_level=5),data)


def test_frozen_host_reader_refuses_changed_code_or_unconverged_host(tmp_path,monkeypatch):
    import json
    import resynthesize_hot_trace as frozen
    from wd_spectra import ModelResult,save_model_result
    from wd_spectra.spectrum import Spectrum
    monkeypatch.setattr(frozen,'identity',lambda data:dict(numerical_code='current',physical_data='tables'))
    (tmp_path/'numerical-identity.json').write_text(json.dumps(dict(numerical_code='old',physical_data='tables')))
    with pytest.raises(ValueError,match='code/data changed'):
        frozen.load_frozen_host(tmp_path,ModelData.default())
    frozen.record_identity(tmp_path,ModelData.default())
    config=DAOConfig(52500.,7.53,5.)
    a=gray_hydrogen_helium_atmosphere(52500.,7.53,5.,n_depth=5)
    host=ModelResult('DAO',a,Spectrum(np.array([1174.,1177.]),np.ones(2),{}),config,
                     {'atmosphere_convergence_status':'unconverged'})
    save_model_result(host,tmp_path/'host')
    with pytest.raises(ValueError,match='not converged'):
        frozen.load_frozen_host(tmp_path,ModelData.default())


def test_trace_only_identity_exception_cannot_hide_host_or_table_changes(tmp_path,monkeypatch):
    import hashlib,json
    import resynthesize_hot_trace as frozen
    package=Path(__file__).resolve().parents[1]/'src/wd_spectra'
    paths=list(package.rglob('*.py'))+list(package.glob('_rt*.so'))+list(package.glob('_rt*.pyd'))
    old={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old['hot_trace_metals.py']='earlier trace version'
    record=dict(numerical_code='earlier',physical_data='same-tables')
    manifest=dict(identity=record,files=old)
    (tmp_path/'numerical-identity.json').write_text(json.dumps(record))
    (tmp_path/'validated-code-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(frozen,'identity',lambda data:dict(numerical_code='current',physical_data='same-tables'))
    with pytest.raises(ValueError,match='code/data changed'):frozen.verify_frozen_identity(tmp_path,None)
    assert 'only hot_trace_metals.py changed' in frozen.verify_frozen_identity(tmp_path,None,True)
    old['light_metal_nlte.py']='earlier metal rate equations'
    (tmp_path/'validated-code-manifest.json').write_text(json.dumps(manifest))
    assert 'light_metal_nlte.py' in frozen.verify_frozen_identity(tmp_path,None,True)
    monkeypatch.setattr(frozen,'identity',lambda data:dict(numerical_code='current',physical_data='changed-tables'))
    with pytest.raises(ValueError,match='code/data changed'):frozen.verify_frozen_identity(tmp_path,None,True)
    monkeypatch.setattr(frozen,'identity',lambda data:dict(numerical_code='current',physical_data='same-tables'))
    old['hot_nlte.py']='different host equations'
    (tmp_path/'validated-code-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='code/data changed'):frozen.verify_frozen_identity(tmp_path,None,True)
