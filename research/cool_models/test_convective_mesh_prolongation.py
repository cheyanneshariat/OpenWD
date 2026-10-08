from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_discrete_dense_transport_seed import Column
from wd_spectra.constants import STEFAN_BOLTZMANN
import convective_mesh_prolongation as module


@pytest.mark.parametrize('logg', [7.5, 8.0])
def test_prolongation_preserves_transport_not_just_temperature(monkeypatch, logg):
    target=STEFAN_BOLTZMANN*5000**4
    p=np.geomspace(1e9,1e10,25)
    seed=Column(4500*(p/p[0])**.21,p,np.geomspace(.1,10,len(p)),p/1e12,gravity=10**logg)
    parent=SimpleNamespace(gas_pressure=p[::2],metadata={
        'convective_flux_fraction_by_interface':np.r_[np.zeros(3),np.linspace(.01,1,10)]})
    def at(teff,p,t,tau,*,logg):
        assert logg == seed.logg
        return Column(t,p,tau,p/1e12,gravity=10**logg,effective_temperature=teff)
    class Material:
        def __init__(self,at,*unused): self.at=at
        def fields(self,lt):
            a=self.at(np.exp(lt))
            return a,None
        def assemble(self,a,f):
            # The surface radiative part has a stable gradient; interior
            # convection depends nonlinearly on the current temperature.
            ad=np.where(a.gas_pressure<1.4e9,.3,.2)
            return ad,np.full(a.n_depth,1e-4),target*1e8*(a.temperature/5000)**2
    monkeypatch.setattr(module,'MaterialCoefficients',Material)
    initialized,error=module.prolongate(seed,parent,SimpleNamespace(atmosphere_at=at),
        dict(thermodynamics=None,rosseland_opacity=None,mixing_length_alpha=1.,
             with_temperature=lambda t:replace(seed,temperature=t)))
    assert error<1e-6
    assert initialized.temperature[0]==seed.temperature[0]
    np.testing.assert_array_equal(initialized.gas_pressure,p)
    assert not np.allclose(initialized.temperature,seed.temperature,rtol=1e-4)


@pytest.mark.parametrize('logg', [7.5, 8.0])
def test_explicit_stable_projection_changes_temperatures_not_flux(monkeypatch, logg):
    target=STEFAN_BOLTZMANN*5000**4
    p=np.geomspace(1e9,1e10,25)
    seed=Column(4500*(p/p[0])**.4,p,np.geomspace(.1,10,len(p)),p/1e12,gravity=10**logg)
    def at(teff,p,t,tau,*,logg):
        assert logg == seed.logg
        return Column(t,p,tau,p/1e12,gravity=10**logg,effective_temperature=teff)
    class Material:
        def __init__(self,at,*unused):self.at=at
        def fields(self,lt):return self.at(np.exp(lt)),None
        def assemble(self,a,f):
            return np.full(a.n_depth,.2),np.full(a.n_depth,1e-4),np.full(a.n_depth,target*1e12)
    monkeypatch.setattr(module,'MaterialCoefficients',Material)
    initialized,error=module.transport_profile(seed,np.zeros(len(p)),SimpleNamespace(atmosphere_at=at),
        dict(thermodynamics=None,rosseland_opacity=None,mixing_length_alpha=1.,
            with_temperature=lambda t:replace(seed,temperature=t)),project_stable=True)
    assert error<1e-6
    np.testing.assert_allclose(initialized.temperature,4500*(p/p[0])**.2,rtol=1e-13)
