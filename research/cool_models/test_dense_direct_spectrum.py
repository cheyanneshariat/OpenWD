from dataclasses import dataclass
import numpy as np
import pytest
from wd_spectra.spectrum import Spectrum,planck_lambda_angstrom
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra._stable_feautrier import cancellation_safe_field
import dense_direct_spectrum as module


@dataclass(frozen=True)
class Atmosphere:
    column_mass:np.ndarray
    temperature:np.ndarray
    metadata:dict


@dataclass(frozen=True)
class Result:
    atmosphere:Atmosphere
    spectrum:Spectrum
    metadata:dict


def test_direct_synthesis_uses_coupled_field_without_changing_structure(monkeypatch):
    m=np.geomspace(1e-8,1e4,40);t=5000*(.75*(m+2/3))**.25
    a=Atmosphere(m,t,{'experimental_dense_helium':'test','radiative_equilibrium_converged':False})
    wave=np.geomspace(1000,1e6,100)
    absorption=np.ones((100,40))*.1;scattering=np.ones((100,40))*10.
    monkeypatch.setattr(module,'pure_helium_opacities',lambda a,w:(absorption,scattering))
    original=Result(a,Spectrum(wave,np.ones(100),{'original':True}),{})
    result=module.direct_result(original,8)
    depths=optical_depth_from_mass_opacity(m,absorption+scattering)
    _,field=cancellation_safe_field(depths,planck_lambda_angstrom(wave[:,None],t[None,:]),
        absorption,scattering,n_angle=8)
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda,field.interface_flux[:,0])
    np.testing.assert_array_equal(original.spectrum.surface_flux_lambda,np.ones(100))
    assert result.atmosphere.temperature is t
    assert not result.atmosphere.metadata['radiative_equilibrium_converged']
    assert result.atmosphere.metadata['experimental_spectrum_radiation_scale_source_error']<1e-10


def test_direct_research_synthesis_refuses_unlabeled_production_state():
    result=Result(Atmosphere(np.ones(2),np.ones(2),{}),Spectrum(np.ones(2),np.ones(2),{}),{})
    with pytest.raises(ValueError,match='explicitly experimental'):
        module.direct_result(result,8)


def test_mass_synthesis_matches_structure_and_rejects_wrong_transfer(monkeypatch):
    from mass_conservative_feautrier import mass_field
    m=np.geomspace(1e-8,100.,25);t=5000*(1+m)**.1;wave=np.geomspace(1000,1e6,100)
    atmosphere=Atmosphere(m,t,{'experimental_dense_helium':'test','experimental_mass_conservative_transfer':True})
    absorption=np.broadcast_to(np.exp(np.sin(np.arange(25))),(100,25));scattering=absorption*.2
    monkeypatch.setattr(module,'pure_helium_opacities',lambda a,w:(absorption,scattering))
    original=Result(atmosphere,Spectrum(wave,np.ones(100),{}),{})
    with pytest.raises(ValueError,match='same declared'):
        module.direct_result(original,8)
    result=module.direct_result(original,8,mass_conservative=True)
    tau=optical_depth_from_mass_opacity(m,absorption+scattering)
    _,field=mass_field(tau,planck_lambda_angstrom(wave[:,None],t[None,:]),absorption,scattering,
                      column_mass=m,n_angle=8,reconstruct_intensity=True)
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda,field.interface_flux[:,0])
    assert result.atmosphere.temperature is t
    assert result.atmosphere.metadata['experimental_positive_intensity_reconstruction'] is True
