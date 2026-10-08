"""Packaged C–A opacity, unchanged chemistry/Swan data, and diagnostic control."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.models.common import ModelData
from wd_spectra.models.dq import DQConfig, validate_config
from wd_spectra._dq.runtime import make_material
from wd_spectra._dq.data import data_root, validate_data
from wd_spectra._dq.c2_ca import load_ca_table
from wd_spectra.carbon_molecular import read_c2_cross_section_table


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    root=Path(__file__).parent/'data/dq_regressions'
    params=json.loads((root/'manifest.json').read_text())['parameters']
    data=ModelData.default();output=tmp_path_factory.mktemp('ca')
    on=make_material(DQConfig(**params),data,output)
    off=make_material(DQConfig(**params,include_c2_ca=False),data,output)
    with np.load(root/'j1235-fixed.npz',allow_pickle=False) as z:
        seed=gray_helium_atmosphere(params['effective_temperature'],params['logg'],n_depth=len(z['temperature']))
        state=replace(seed,**{k:z[k].copy() for k in
            ('temperature','gas_pressure','column_mass','rosseland_optical_depth')})
    return on,off,state


def test_packaged_data_identity_and_population_convention():
    assert 'c2-ca-2024.npz' in validate_data()['sha256']
    parent=read_c2_cross_section_table(data_root()/'c2-8states-r15000.npz')
    ca=load_ca_table(parent)
    assert ca.swan_cross_section is None
    assert ca.swan_rotational_overlap_cross_section is None
    assert np.any(ca.cross_section>0)
    assert np.isfinite(ca.cross_section).all() and np.all(ca.cross_section>=0)
    np.testing.assert_array_equal(ca.partition_function,parent.partition_function)
    with np.load(data_root()/'c2-ca-2024.npz',allow_pickle=False) as z:
        provenance=json.loads(str(z['provenance_json'].item()))
    assert len(provenance['bands'])==63
    assert provenance['bands'][0]['A']==1.711e7
    assert 1.88 < provenance['f00_new_over_old'] < 1.89
    assert provenance['parent_sha256']==validate_data()['sha256']['c2-ca-historical.npz']
    with pytest.raises(ValueError,match='partition'):
        load_ca_table(replace(parent,partition_function=parent.partition_function*2))


def test_default_and_diagnostic_switch(pair):
    on,off,state=pair
    assert DQConfig().include_c2_ca is True
    assert on.ca_table is not None and off.ca_table is None
    assert on.experiment_metadata['c2_ca_included'] is True
    assert off.experiment_metadata['c2_ca_included'] is False
    with pytest.raises(ValueError,match='boolean'):
        validate_config(DQConfig(include_c2_ca='False'))
    a,c,n=on.chemistry(state);b,d,m=off.chemistry(state)
    np.testing.assert_array_equal(n,m)
    for field in ('mass_density','electron_density','temperature'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    w=np.array([3000.,3850.,4100.,4381.,4697.,5165.,6500.])
    on.prepare_opacity_state(a,c,n);off.prepare_opacity_state(b,d,m)
    for j in (0,a.n_depth//2,a.n_depth-1):
        np.testing.assert_array_equal(on.swan_column(a,j,w),off.swan_column(b,j,w))
    original=off.absorption(state,w)
    expected=original+on.ca_table.cross_section_for_wavelength_temperature(w,a.temperature)*(n/a.mass_density)[None,:]
    actual=on.absorption(state,w)
    np.testing.assert_allclose(actual,expected,rtol=3e-15,atol=1e-30)
    assert np.any(actual>original)
    np.testing.assert_array_equal(on.absorption(state,w,include_c2=False),off.absorption(state,w,include_c2=False))
    np.testing.assert_array_equal(on.absorption(state,w,structure=True),actual)
    np.testing.assert_allclose(np.concatenate([on.absorption(state,q) for q in np.array_split(w,3)]),actual,rtol=2e-12)


def test_diagnostic_switch_reaches_worker(monkeypatch,tmp_path):
    from wd_spectra.models import dq
    from wd_spectra._dq import data
    captured={}
    monkeypatch.setattr(data,'validate_data',lambda:{})
    monkeypatch.setattr(dq,'_load_result',lambda *a:'qualified')
    monkeypatch.setattr(dq.subprocess,'run',lambda command,**kw:captured.update(command=command))
    with pytest.warns(dq.DQApproximationWarning):
        dq.compute_dq(DQConfig(include_c2_ca=False),output_directory=tmp_path/'new')
    assert '--no-c2-ca' in captured['command']


def test_historical_band_origins_strength_and_area(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools'))
    from build_dq_ca_historical import band_constants, oscillator_strength, FC
    from dq_ca_band_envelope import geometry_for_band, exact_band_fractions
    for levels,wavelength in { (0,0):3850.62, (0,2):4381.01, (0,3):4697.,
                              (1,3):4337.53, (2,4):4302.94 }.items():
        assert abs(1e8/band_constants(*levels)[0]-wavelength)<.01
    # Independent Cooper/App. A scale check against the Swan f(0,0).
    assert .024<oscillator_strength(19378.44,.73521,3.52,6)<.026
    assert FC.shape==(7,9) and np.all(FC>=0)
    origin,bu,bl=band_constants(0,0)
    edges=np.linspace(1.,100000.,10001)
    geometry=geometry_for_band(edges,origin,bu,bl)
    for t in (4000.,7000.,12000.):
        fraction,lost=exact_band_fractions(geometry,bl,t,len(edges)-1)
        assert np.all(fraction>=0)
        assert abs(fraction.sum()+lost-1)<2e-15
        assert lost<1e-8
