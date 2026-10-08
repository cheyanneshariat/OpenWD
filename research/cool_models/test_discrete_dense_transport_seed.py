"""A discrete initializer is tested on its own equations, not called converged."""
from dataclasses import dataclass,replace
from types import SimpleNamespace
import numpy as np
import pytest
from discrete_dense_transport_seed import discrete_seed
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root


@dataclass
class Column:
    temperature: np.ndarray
    gas_pressure: np.ndarray
    rosseland_optical_depth: np.ndarray
    mass_density: np.ndarray
    gravity: float=1e8
    effective_temperature: float=5000.

    @property
    def n_depth(self): return len(self.temperature)

    @property
    def logg(self): return float(np.log10(self.gravity))


@pytest.mark.parametrize('logg', [7.5, 8.0])
def test_initializer_solves_same_discrete_transport_with_convective_onset(monkeypatch, logg):
    import discrete_dense_transport_seed as module
    target=STEFAN_BOLTZMANN*5000.**4
    coefficient=target*1e9
    pressure=np.geomspace(1e6,1e13,45)
    seed=Column(np.full(45,4200.),pressure,np.geomspace(1e-8,100.,45),pressure/1e12,gravity=10**logg)
    def at(teff,p,t,tau,*,logg):
        assert logg == seed.logg
        return Column(t,p,tau,p/1e12,gravity=10**logg,effective_temperature=teff)
    class Materials:
        def __init__(self,with_temperature,*unused): self.at=with_temperature
        def fields(self,logt):
            a=self.at(np.exp(logt))
            return a,np.array([a.mass_density,np.ones(2)])
        def assemble(self,a,f): return np.full(2,.2),np.full(2,1e-4),np.full(2,coefficient)
    monkeypatch.setattr(module,'MaterialCoefficients',Materials)
    initialized,error=discrete_seed(seed,SimpleNamespace(atmosphere_at=at),
        dict(thermodynamics=None,rosseland_opacity=None,mixing_length_alpha=1.,
             with_temperature=lambda t:replace(seed,temperature=t)))
    assert initialized.temperature[0] == seed.temperature[0]
    np.testing.assert_array_equal(initialized.gas_pressure,seed.gas_pressure)
    g=np.diff(np.log(initialized.temperature))/np.diff(np.log(pressure))
    contrast,_=_ml2_contrast_and_root(np.maximum(g-.2,0.),1e-4)
    conv=coefficient*contrast**3
    rad=16*STEFAN_BOLTZMANN*seed.gravity*(initialized.temperature[:-1]*initialized.temperature[1:])**2/(3*np.sqrt(pressure[:-1]*pressure[1:]))*g
    np.testing.assert_allclose((rad+conv)/target,1.,atol=1e-6,rtol=0)
    assert error < 1e-6
    assert np.any(conv == 0) and np.any(conv > .5*target)
