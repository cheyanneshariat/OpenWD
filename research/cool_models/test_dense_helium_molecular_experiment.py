from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from dense_helium_molecular_experiment import MolecularDenseEOS,TABLE_VERSION,molecular_absorption
from dense_helium_atomic_experiment import atomic_dense_experiment
from test_dense_helium_atomic_experiment import AnalyticBulk
from wd_spectra.constants import HELIUM_MASS,BOLTZMANN,PLANCK,LIGHT_SPEED
from wd_spectra import helium_molecular as m


@pytest.fixture
def model(tmp_path):
    t,rho=np.geomspace(3000,17000,5),np.geomspace(1e-12,2.,8)
    mu=np.zeros((len(t),len(rho),3))
    mu[...,0],mu[...,1],mu[...,2]=rho,-2*rho,3*rho
    path=tmp_path/'unit-table.npz'
    np.savez(path,temperature=t,density=rho,mu=mu,version=TABLE_VERSION,
        bound_neighbours=np.broadcast_to(5*rho,(len(t),len(rho))),
        bound_energy_ev=[-2.347],bound_rotational_quantum_number=[1],bound_state_source='unit-test level')
    return MolecularDenseEOS(AnalyticBulk(),path)


def test_coupled_nonideal_charge_nuclei_and_opacity_populations(model):
    t,p=np.array([4000.,6000.,8000.]),np.array([1e9,1e10,1e9])
    state=model.lte(t,p)
    c=model.chemistry(t,state.mass_density)
    np.testing.assert_allclose(c.electron,c.atomic_ion+c.molecular_ion,rtol=3e-14)
    np.testing.assert_allclose(c.neutral+c.atomic_ion+2*c.molecular_ion,state.mass_density/HELIUM_MASS,rtol=3e-14)
    np.testing.assert_allclose(state.neutral_level_population_density.sum(axis=-1),c.neutral,rtol=1e-15)
    # One solute reaction He <-> He+ + e for all ions; bound pairs split by 1-exp(-N_b).
    ions=c.atomic_ion+c.molecular_ion
    nuclei=state.mass_density/HELIUM_MASS
    np.testing.assert_allclose(np.log(ions*c.electron/(nuclei-ions)),c.log_ionization_constant,atol=3e-14)
    np.testing.assert_allclose(c.molecular_ion/ions,-np.expm1(-5*state.mass_density),rtol=1e-6)
    assert np.all(c.molecular_ion > 0)


def test_explicit_dimer_donors_and_ideal_opacity_limit():
    t=m._TEMPERATURE[:4]
    n0=np.full(4,1e20); n1=np.full(4,1e10)
    nd=n0*n1/m._EQUILIBRIUM_CONSTANT[:4]
    c=SimpleNamespace(neutral=n0,atomic_ion=n1,molecular_ion=nd)
    wave=np.array([800.,2000.,5000.,10000.])
    stim=-np.expm1(-PLANCK*LIGHT_SPEED/(wave[:,None]*1e-8*BOLTZMANN*t))
    expected=m.helium_dimer_ion_continuum_coefficient(wave[:,None],t)*n0*n1*stim
    np.testing.assert_allclose(molecular_absorption(wave,t,c),expected,rtol=2e-15)
    c.molecular_ion=2*nd
    extra=molecular_absorption(wave,t,c)-expected
    bf=m._linear_interpolate_table(wave[:,None],t,m._WAVELENGTH_BOUND_FREE,m._BOUND_FREE)
    np.testing.assert_allclose(extra,bf*nd*stim,rtol=1e-13)


def test_trace_chemistry_applicability_is_local_not_a_teff_switch(model):
    from dense_helium_limits import DenseHeliumDomainError
    # The denser, hotter state can satisfy the trace-ion approximation while
    # the cooler, very dilute state cannot. These are local unit-test states,
    # not a prescription for switching the EOS at a stellar Teff threshold.
    hot = model.lte(15000., 1e10)
    assert hot.electron_density/hot.helium_nuclei_density < 1e-3
    with pytest.raises(DenseHeliumDomainError, match='trace-ion'):
        model.lte(9000., 1.)


def test_scoped_opacity_eos_restoration_after_error(model):
    from wd_spectra import eos,helium,adaptive_structure
    original=eos.hummer_mihalas_helium_lte
    opacity=helium.helium_continuum_mass_absorption_coefficient
    adaptive_solve=adaptive_structure.solve_adaptive_lte_structure
    newton_solve=adaptive_structure.solve_trust_region_newton
    runner=SimpleNamespace(hummer_mihalas_helium_lte=original,
        hummer_mihalas_helium_thermodynamics=eos.hummer_mihalas_helium_thermodynamics,
        helium_continuum_mass_absorption_coefficient=opacity,
        compute_db=lambda *a,**k:None,save_model_result=lambda *a,**k:None,
        helium_continuum_atmosphere=lambda *a,**k:None,run=lambda *a,**k:None,
        radiative_equilibrium_helium_atmosphere=lambda *a,**k:None)
    with pytest.raises(RuntimeError):
        with atomic_dense_experiment(runner,model,opacity_factory=model.opacity_factory):
            assert helium.helium_continuum_mass_absorption_coefficient is not opacity
            assert eos.hummer_mihalas_helium_lte == model.lte
            adaptive_structure.solve_adaptive_lte_structure=lambda *a,**k:None
            adaptive_structure.solve_trust_region_newton=lambda *a,**k:None
            raise RuntimeError('intentional')
    assert eos.hummer_mihalas_helium_lte is original
    assert helium.helium_continuum_mass_absorption_coefficient is opacity
    assert adaptive_structure.solve_adaptive_lte_structure is adaptive_solve
    assert adaptive_structure.solve_trust_region_newton is newton_solve


def test_dense_helium_minus_table_multiplies_free_free_and_labels_physics(tmp_path):
    from pathlib import Path
    from wd_spectra.dense_helium_continuum import (DenseHeliumCorrectionTable,
                                                    bounded_helium_minus_correction)
    from wd_spectra.models import ModelData
    from wd_spectra import helium
    table=ModelData.default().helium_minus_dense_free_free
    t,rho=np.geomspace(3000,17000,5),np.geomspace(1e-12,2.,8)
    mu=np.zeros((len(t),len(rho),3))
    path=tmp_path/'unit-table.npz'
    np.savez(path,temperature=t,density=rho,mu=mu,version=TABLE_VERSION,
        bound_neighbours=np.broadcast_to(5*rho,(len(t),len(rho))),
        bound_energy_ev=[-2.347],bound_rotational_quantum_number=[1],bound_state_source='unit-test level')
    plain=MolecularDenseEOS(AnalyticBulk(),path)
    dense=MolecularDenseEOS(AnalyticBulk(),path,helium_minus_table=table)
    assert 'ABSENT' in plain.physics and 'Rogers-Young' in dense.physics and dense.physics != plain.physics
    calls=[]
    def original(atmosphere,wavelength,**options):
        calls.append(options.get('helium_minus_correction'))
        return np.zeros((len(wavelength),atmosphere.temperature.size))
    state=SimpleNamespace(neutral_he_density=np.array([.5/HELIUM_MASS]))
    atmosphere=SimpleNamespace(temperature=np.array([7000.]),helium_lte_state=state,mass_density=np.array([.5]))
    wave=np.array([3000.,5500.,2e4])
    dense.opacity_factory(original)(atmosphere,wave,include_helium_dimer_ion=False)
    plain.opacity_factory(original)(atmosphere,wave,include_helium_dimer_ion=False)
    expected=bounded_helium_minus_correction(DenseHeliumCorrectionTable(table),wave,
        atmosphere.temperature,state.neutral_he_density)
    np.testing.assert_array_equal(calls[0],expected)
    assert np.all(expected < 1) and calls[1] is None


def test_cool_he2_photodissociation_is_continuous_with_stancil_and_falls_in_the_red():
    from types import SimpleNamespace
    c=SimpleNamespace(neutral=np.full(4,1e23),atomic_ion=np.zeros(4),molecular_ion=np.ones(4))
    wave=np.array([2000.,5000.,10000.])
    a=molecular_absorption(wave,np.array([2000.,4199.9,4200.,4200.1]),c)
    np.testing.assert_allclose(a[:,1],a[:,2],rtol=1e-3)
    np.testing.assert_allclose(a[:,3],a[:,2],rtol=1e-3)
    assert np.all(a[1:,0] < .2*a[1:,2])      # fewer excited vibrational levels at 2000 K
