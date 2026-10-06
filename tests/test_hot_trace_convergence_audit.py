"""A saved success flag must not override failed or missing residual checks."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from verify_hot_composition_convergence import check_record


def record():
    return dict(requested_population_tolerance=1e-4,population_defect=8e-5,
        converged=True,smoke_only=False,convergence_scope='fixed-host metal populations',
        nlte_elements=['C','Si'],element_population_defects={'C':2e-5,'Si':8e-5},
        source_closure_residual=1e-13,maximum_particle_conservation_error=1e-15)


def test_convergence_audit_requires_residuals_even_with_success_flag():
    check_record(record())
    for residual in (1e-4,.03,float('nan'),float('inf'),None):
        metadata=record();metadata['population_defect']=residual
        with pytest.raises(ValueError,match='undamped population residual'):check_record(metadata)


def test_convergence_audit_rejects_missing_element_and_loosened_tolerance():
    metadata=record();del metadata['element_population_defects']['Si']
    with pytest.raises(ValueError,match='element residual'):check_record(metadata)
    metadata=record();metadata['requested_population_tolerance']=.01
    with pytest.raises(ValueError,match='requested tolerance'):check_record(metadata)


@pytest.mark.parametrize('key',['source_closure_residual','maximum_particle_conservation_error'])
def test_small_population_defect_does_not_hide_transfer_or_conservation_failure(key):
    metadata=deepcopy(record());metadata[key]=.01
    with pytest.raises(ValueError,match='failed convergence audit'):check_record(metadata)


def test_energy_balance_audit_without_metals_recovers_unchanged_control(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import hot_trace_energy_balance as audit
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.nlte_core import NLTETransferCoefficients
    from wd_spectra.spectrum import planck_lambda_angstrom
    atmosphere=gray_hydrogen_atmosphere(50000.,7.75,n_depth=4,tau_max=3.)
    wave=np.geomspace(100.,10000.,150)
    host=SimpleNamespace(atmosphere=atmosphere,config=SimpleNamespace(quality='standard'))
    model=SimpleNamespace(n_angle=2)
    monkeypatch.setattr(audit,'numerical_resolution',lambda q:SimpleNamespace(n_continuum=150))
    monkeypatch.setattr(audit,'structure_wavelength',lambda c,n:wave)
    monkeypatch.setattr(audit,'HotEquations',lambda *args:SimpleNamespace(wave=wave))
    def background(w):
        b=planck_lambda_angstrom(w[:,None],atmosphere.temperature)
        return NLTETransferCoefficients(w,np.ones_like(b),b,np.full_like(b,.1),{})
    result=audit.audit_energy_balance(host,model,None,None,{}, {}, {}, {},wave,
                                      background,background,tmp_path)
    assert result['HHe']==result['WithMetals']
    assert result['metal_induced_surface_flux_change']==0.
    assert result['maximum_metal_induced_flux_change']==0.
    assert not result['thermal_equilibrium_with_metals_certified']


@pytest.mark.parametrize('partition',['element','ion'])
def test_grouped_acceleration_converges_to_the_same_coupled_solution(monkeypatch,partition):
    from hot_trace_acceleration import ElementPopulationAcceleration
    import wd_spectra.hot_trace_metals as trace
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.metals import read_pg1159_atomic_database,read_verner_photoionization_database
    from wd_spectra.models.common import ModelData
    from wd_spectra.nlte_core import NLTETransferCoefficients
    from wd_spectra.spectrum import planck_lambda_angstrom
    data=ModelData.default();atmosphere=gray_hydrogen_atmosphere(50000.,7.75,n_depth=5,tau_max=10.)
    database=read_pg1159_atomic_database(data.stout,elements=('C','Si'))
    photo=read_verner_photoionization_database(data.verner_photoionization,elements=('C','Si'))
    counts={'C':{2:3,3:3,4:1},'Si':{2:3,3:3,4:1}}
    def background(w):
        b=planck_lambda_angstrom(w[:,None],atmosphere.temperature)
        return NLTETransferCoefficients(w,np.ones_like(b),b,np.zeros_like(b),{})
    options=dict(atomic_database=database,photoionization_database=photo,levels_per_charge=counts,
                 maximum_iterations=120,tolerance=1e-6,n_angle=2)
    wave=np.linspace(1393.,1404.,201)
    ordinary=trace.solve_hot_trace_metals(atmosphere,background,{'C':-10.,'Si':-10.},wave,**options)
    groups=({'C':7,'Si':7} if partition=='element' else
            {f'{element}:{q}':n for element,stages in counts.items() for q,n in stages.items()})
    grouped=ElementPopulationAcceleration(groups)
    monkeypatch.setattr(trace,'population_update',grouped)
    separate=trace.solve_hot_trace_metals(atmosphere,background,{'C':-10.,'Si':-10.},wave,**options)
    assert ordinary.converged and separate.converged
    assert all(reasons.get('accepted',0)>0 for reasons in grouped.reasons.values())
    for element in counts:
        np.testing.assert_allclose(ordinary.populations[element].population_density,
            separate.populations[element].population_density,rtol=1e-5)
    np.testing.assert_allclose(ordinary.spectrum.surface_flux_lambda,separate.spectrum.surface_flux_lambda,rtol=1e-6)
    assert separate.metadata['maximum_particle_conservation_error']<1e-12
