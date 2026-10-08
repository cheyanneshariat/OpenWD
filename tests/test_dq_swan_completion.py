"""Default line identity and exact, bounded cache without spectral refitting."""
import numpy as np
import pytest
from wd_spectra.carbon_molecular import read_c2_cross_section_table
from wd_spectra._dq.data import data_root, validate_data
from wd_spectra._dq.runtime import make_material
from wd_spectra._dq.swan_strength_cache import CachedSwan
from wd_spectra._dq.dq_swan_depth_integral import SpatiallyIntegratedSwan
from wd_spectra.models.dq import DQConfig
from wd_spectra.models.common import ModelData


def test_default_is_checked_combination(tmp_path):
    manifest = validate_data()
    assert manifest['sha256']['swan-completed.npz'] == '89f65fedf0b936a20dea3f4cc6ef9de87edc9467f1aa882e96463012e0cee943'
    assert manifest['sha256']['c2-ca-2024.npz'] == '5594b764c7cb3bc43d0ff37c560fc0329025dd0b24553d902c6b3205fc998fd3'
    with np.load(data_root()/'swan-completed.npz') as z, np.load(data_root()/'hornkohl_calibrated.npz') as h:
        assert z['lines'].shape == (3, 1261446)
        np.testing.assert_array_equal(z['lines'][:, :29004], h['lines'])
    mat = make_material(DQConfig(), ModelData.default(), tmp_path)
    assert isinstance(mat.integrated_swan, CachedSwan)
    assert len(mat.integrated_swan.nu) == 1261446
    assert mat.integrated_swan.strength_cache_bytes == 1024**3
    assert mat.ca_table is not None


def test_structure_cache_cap_restored_on_failure(tmp_path, monkeypatch):
    mat = make_material(DQConfig(), ModelData.default(), tmp_path)
    cold = next(cls for cls in type(mat).__mro__ if cls.__name__ == 'ColdDQ')
    def fail(self, *args, **kwargs):
        assert self.integrated_swan.strength_cache_bytes == 32*1024**2
        assert self.maximum_memory_bytes == 4*1024**3
        raise RuntimeError('deliberate solve failure')
    monkeypatch.setattr(cold.__bases__[0], 'solve', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        cold.solve(mat)
    assert mat.integrated_swan.strength_cache_bytes == 1024**3


@pytest.mark.parametrize('name', ['hornkohl_calibrated.npz', 'swan-completed.npz'])
def test_exact_cache_and_eviction(name):
    parent = read_c2_cross_section_table(data_root()/'c2-8states-r15000.npz')
    base = SpatiallyIntegratedSwan(parent, branches=data_root()/name)
    budget = 2 * (2*len(base.nu)+1) * 8
    cache = CachedSwan(parent, branches=data_root()/name, strength_cache_bytes=budget)
    wave = np.linspace(3800., 6800., 101)
    for t in (4000., 6000., 8000., 6000., 6000.+1e-7):
        for left, right in ((0.,0.), (1.,1.+1e-7), (0.,100.), (100.,0.), (-100.,50.)):
            np.testing.assert_array_equal(cache.cell_average(wave,t,left,right),base.cell_average(wave,t,left,right))
        q = np.array([-100., 0., 1., 2e4, 1e6])
        np.testing.assert_array_equal(cache.cumulative(q,t),base.cumulative(q,t))
        assert cache._thermal_bytes <= budget
    assert len(cache._thermal_cache) == 2
    assert 6000. in cache._thermal_cache and 6000.+1e-7 in cache._thermal_cache
    for values in cache._thermal_cache.values():
        assert all(not a.flags.writeable for a in values)
    cache.set_strength_cache_bytes(0)
    assert cache._thermal_bytes == 0 and not cache._thermal_cache
    cache.set_strength_cache_bytes(budget)
    np.testing.assert_array_equal(cache.cell_average(wave,6000.,0.,100.),base.cell_average(wave,6000.,0.,100.))
    for bad in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            cache.set_strength_cache_bytes(bad)
    uncached = CachedSwan(parent, branches=data_root()/name, strength_cache_bytes=0)
    np.testing.assert_array_equal(uncached.cell_average(wave,6000.,0.,100.),base.cell_average(wave,6000.,0.,100.))
    assert uncached._thermal_bytes == 0 and not uncached._thermal_cache
