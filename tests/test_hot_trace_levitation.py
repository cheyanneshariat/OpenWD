"""Momentum units and conservation, not physical validation of levitation."""
import sys
from pathlib import Path
import numpy as np
import pytest
from wd_spectra.constants import LIGHT_SPEED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from hot_trace_levitation import (radiative_acceleration, zero_drift_abundance_gradient,
    mean_element_charge, homogeneous_support_acceleration)


def test_force_units_no_extra_four_pi_or_angstrom_conversion():
    wave = np.array([100.,200.,400.])
    opacity = np.full((3,2),2.)
    flux = np.full((3,2),3e12)
    fractions = np.array([1e-6,2e-6])
    actual = radiative_acceleration(wave,opacity,flux,fractions)
    np.testing.assert_allclose(actual,2*3e12*300/(LIGHT_SPEED*fractions))


def test_thin_force_independent_of_abundance_and_momentum_adds():
    rng = np.random.default_rng(13)
    wave = np.geomspace(10.,10000.,101)
    flux = rng.normal(size=(101,5))*1e12
    a,b = rng.random((2,101,5))
    x,y = np.full(5,1e-6),np.full(5,4e-6)
    ga = radiative_acceleration(wave,a,flux,x)
    gb = radiative_acceleration(wave,b,flux,y)
    np.testing.assert_allclose(radiative_acceleration(wave,10*a,flux,10*x),ga)
    np.testing.assert_allclose(radiative_acceleration(wave,a+b,flux,np.ones(5)),x*ga+y*gb)
    np.testing.assert_allclose(radiative_acceleration(wave,-a,flux,x),-ga)


@pytest.mark.parametrize('bad', ['shape','fraction','wavelength','nan'])
def test_invalid_force_inputs_rejected(bad):
    wave = np.array([100.,200.,400.]);a=np.ones((3,2));f=a.copy();x=np.ones(2)
    if bad=='shape':f=f[:,0]
    if bad=='fraction':x[0]=0
    if bad=='wavelength':wave=wave[::-1]
    if bad=='nan':a[0,0]=np.nan
    with pytest.raises(ValueError,match='invalid force'):
        radiative_acceleration(wave,a,f,x)


def test_isothermal_ionized_hydrogen_diffusion_limit():
    from types import SimpleNamespace
    from wd_spectra.constants import BOLTZMANN
    m=np.geomspace(1e-6,1.,30);g=1e8;temperature=np.full(30,50000.)
    nh=m*g/(2*BOLTZMANN*temperature)
    host=SimpleNamespace(column_mass=m,gravity=g,temperature=temperature,
        electron_density=nh,mass_density=nh*1.66053906660e-24,
        hydrogen_lte_state=SimpleNamespace(hydrogen_nuclei_density=nh))
    a,z=12.,3.;q=np.full(30,z)
    np.testing.assert_allclose(zero_drift_abundance_gradient(host,a,q,np.zeros(30)),2*a-(z+1))
    balancing_force=np.full(30,g*(1-(z+1)/(2*a)))
    np.testing.assert_allclose(homogeneous_support_acceleration(host,a,q),balancing_force)
    np.testing.assert_allclose(zero_drift_abundance_gradient(host,a,q,balancing_force),0.,atol=1e-12)
    assert np.all(zero_drift_abundance_gradient(host,a,q,np.full(30,2*g))<0)


def test_charge_retains_omitted_lte_reservoir():
    from types import SimpleNamespace
    ref=SimpleNamespace(element_number_density={'C':np.array([10.])},
        ion_number_density={'C':np.array([[1.],[4.],[5.]])})
    # Move two explicitly represented atoms from charge 1 to 2; other six inert.
    keys=[('C',1,1),('C',2,1)]
    np.testing.assert_allclose(mean_element_charge(ref,'C',keys,
        np.array([[1.],[3.]]),np.array([[3.],[1.]])),[1.6])
