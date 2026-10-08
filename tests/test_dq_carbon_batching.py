"""Full-state UV support must survive depth batching and exact-layer caching."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere, metals
from wd_spectra.models.common import ModelData
from wd_spectra.models.dq import DQConfig
from wd_spectra._dq.runtime import make_material
from wd_spectra._dq.dq_exact_opacity_cache import subset, ExactLayerOpacityCache
from wd_spectra._dq.dq_hornkohl_consistent import anchored_grid
from wd_spectra._dq.dq_bounded_carbon_lines import BoundedCarbonLines


@pytest.fixture
def material(tmp_path):
    root=Path(__file__).parent/'data/dq_regressions'
    params=json.loads((root/'manifest.json').read_text())['parameters']
    mat=make_material(DQConfig(**params),ModelData.default(),tmp_path)
    with np.load(root/'j1235-fixed.npz') as z:
        seed=gray_helium_atmosphere(params['effective_temperature'],params['logg'],n_depth=len(z['temperature']))
        seed=replace(seed,**{k:z[k].copy() for k in
            ('temperature','gas_pressure','column_mass','rosseland_optical_depth')})
    return mat,mat.chemistry(seed)


def arrays(mat,a,c,w):
    return mat.bounded_carbon.evaluate(a,c,w,np.ones((len(w),a.n_depth)))[0]


def test_explicit_support_matches_unchanged_full_state_gate(material):
    mat,(a,c,c2)=material
    mat.prepare_opacity_state(a,c,c2)
    w=np.linspace(1500,1560,121)
    padded,indices=anchored_grid(w,mat.atomic_anchors)
    kwargs=dict(minimum_oscillator_strength=1e-4,maximum_lines=None,transition_keys=mat.atomic_keys)
    implicit=metals.metal_line_mass_absorption_coefficient(a,padded,mat.atomic,c,**kwargs)[indices]
    explicit=metals.metal_line_mass_absorption_coefficient(a,padded,mat.atomic,c,
        uv_resonance_support_angstrom=mat.bounded_carbon.uv_support,**kwargs)[indices]
    np.testing.assert_array_equal(explicit,implicit)
    screened=arrays(mat,a,c,w)
    assert np.max(abs(screened-explicit)/np.maximum(explicit,1.))<2e-10


def test_depth_and_wavelength_chunks_retain_uv_support(material):
    mat,(a,c,c2)=material
    mat.prepare_opacity_state(a,c,c2)
    w=np.r_[np.linspace(1500,1560,61),np.linspace(4000,4060,31)]
    full=arrays(mat,a,c,w)
    for indices in ([0],[3],[a.n_depth//2],[a.n_depth-1],list(range(1,a.n_depth,3))):
        aa=subset(a,indices,a.n_depth);cc=subset(c,indices,a.n_depth,-1)
        part=arrays(mat,aa,cc,w)
        assert np.max(abs(part-full[:,indices])/np.maximum(full[:,indices],1.))<2e-10
    chunks=np.concatenate([arrays(mat,a,c,part) for part in np.array_split(w,4)])
    np.testing.assert_allclose(chunks,full,rtol=2e-12,atol=2e-10)


def test_temperature_probe_partial_cache_miss_equals_fresh(material):
    mat,(a,c,c2)=material
    w=np.linspace(1500,1560,61)
    original=mat.absorption(a,w)
    np.testing.assert_array_equal(mat.absorption(a,w),original)
    assert mat.opacity_cache.reused_columns>=a.n_depth
    t=a.temperature.copy();t[a.n_depth//2]*=1.0002
    changed=replace(a,temperature=t)
    cached=mat.absorption(changed,w)
    mat.opacity_cache=ExactLayerOpacityCache()
    fresh=mat.absorption(changed,w)
    np.testing.assert_allclose(cached,fresh,rtol=2e-10,atol=1e-20)
    restored=mat.absorption(a,w)
    np.testing.assert_allclose(restored,original,rtol=2e-10,atol=1e-20)


def test_wavelength_dependent_screening_preserves_declared_bound(material):
    mat,(a,c,c2)=material
    mat.prepare_opacity_state(a,c,c2)
    w=np.linspace(1500.,1700.,81)
    background=np.geomspace(1e-6,1e6,len(w))[:,None]*np.ones((1,a.n_depth))
    full,info=mat.bounded_carbon.evaluate(a,c,w,background)
    chunks=[]
    for indices in np.array_split(np.arange(len(w)),5):
        part,partial_info=mat.bounded_carbon.evaluate(a,c,w[indices],background[indices])
        assert partial_info['relative_bound']<=1e-10
        chunks.append(part)
    assert info['relative_bound']<=1e-10
    assert np.max(abs(np.concatenate(chunks)-full)/np.maximum(full,background))<2.1e-10


def test_support_change_invalidates_unchanged_columns(material):
    mat,(a,c,c2)=material
    mat.prepare_opacity_state(a,c,c2)
    cache=mat.opacity_cache
    assert any(mat.bounded_carbon.uv_support.values())
    # Synthetic near-absent carbon turns off all resonance gates. The first
    # layer is unchanged, explicitly testing the nonlocal cache dependency.
    ions=c.ion_number_density['C'].copy();ions[:,1:]*=1e-100
    dilute=replace(c,ion_number_density={**c.ion_number_density,'C':ions})
    mat.prepare_opacity_state(a,dilute,c2)
    assert mat.opacity_cache is not cache
    assert not any(mat.bounded_carbon.uv_support.values())
    cache=mat.opacity_cache
    mat.prepare_opacity_state(a,dilute,c2)
    assert mat.opacity_cache is cache
    mat.prepare_opacity_state(a,c,c2)
    assert mat.opacity_cache is not cache


def test_unprepared_or_incomplete_support_is_rejected(material):
    mat,(a,c,c2)=material
    engine=BoundedCarbonLines(mat.atomic,mat.atomic_keys,mat.atomic_anchors)
    w=np.array([1500.,1510.])
    with pytest.raises(RuntimeError,match='complete atmosphere'):
        engine.evaluate(a,c,w,np.ones((2,a.n_depth)))
    with pytest.raises(ValueError,match='Missing declared'):
        metals.metal_line_mass_absorption_coefficient(a,w,mat.atomic,c,
            transition_keys=mat.atomic_keys,uv_resonance_support_angstrom={})
    with pytest.raises(ValueError,match='finite and nonnegative'):
        metals.metal_line_mass_absorption_coefficient(a,w,mat.atomic,c,
            uv_resonance_support_angstrom={('C',0,1,2):np.nan})
