from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_discrete_dense_transport_seed import Column
from wd_spectra.constants import STEFAN_BOLTZMANN
import exact_convective_trial_projection as module


@pytest.mark.parametrize('coefficient_scale',[1e-7,1e8])
def test_projection_checks_actual_flux_and_retains_inefficient_trials(monkeypatch,coefficient_scale):
    target=STEFAN_BOLTZMANN*5000**4
    pressure=np.geomspace(1e8,1e9,20)
    seed=Column(4000*(pressure/pressure[0])**.4,pressure,np.ones(20),pressure/1e12)
    def at(teff,p,t,tau,*,logg):
        return Column(t,p,tau,p/1e12,gravity=10**logg,effective_temperature=teff)
    class Material:
        def __init__(self,at,*unused):self.at=at
        def fields(self,lt):return self.at(np.exp(lt)),None
        def assemble(self,a,f):
            return np.full(a.n_depth,.2),np.full(a.n_depth,1e-3),target*coefficient_scale*(a.temperature/5000)**2
    monkeypatch.setattr(module,'MaterialCoefficients',Material)
    result,info=module.project_trial(seed,SimpleNamespace(atmosphere_at=at),
        dict(thermodynamics=None,rosseland_opacity=None,mixing_length_alpha=1.,
             with_temperature=lambda t:replace(seed,temperature=t)))
    assert result.temperature[0]==seed.temperature[0]
    if coefficient_scale<1:
        assert result is seed
        assert info['corrected_interfaces']==0
    else:
        assert info['corrected_interfaces']==19
        assert info['maximum_actual_convective_flux_ratio']<=1+1e-6
        assert np.all(result.temperature<=seed.temperature*(1+1e-13))


@pytest.mark.parametrize('logg', [7.75, 8.0])
def test_real_helium_projection_preserves_gravity_and_evaluated_ml2_flux(logg):
    """A steep trial must respect the stellar flux at its requested gravity."""
    from wd_spectra._cool import check_cool_db_transport_seed as runner
    from wd_spectra.convection import ml2_convective_flux_for_gradient_from_thermodynamics
    from wd_spectra.adaptive_structure import (
        _positive_interface_values, _arithmetic_interface_values,
    )

    pressure = np.geomspace(1e8, 1e10, 6)
    temperature = 4000 * (pressure / pressure[0]) ** .4
    tau = np.geomspace(1e-3, 100., pressure.size)
    seed = runner.atmosphere_at(5000., pressure, temperature, tau, logg=logg)
    observed_gravities = []

    def at(teff, p, t, optical_depth, **kwargs):
        atmosphere = runner.atmosphere_at(teff, p, t, optical_depth, **kwargs)
        observed_gravities.append(atmosphere.logg)
        return atmosphere

    def thermodynamics(atmosphere):
        return runner.hummer_mihalas_helium_thermodynamics(
            atmosphere.temperature, atmosphere.gas_pressure,
            correlated_microfields=True,
        )

    def opacity(atmosphere):
        return runner.rosseland_mean_helium_continuum_opacity(
            atmosphere, n_frequency=160,
        )

    result, info = module.project_trial(seed, SimpleNamespace(atmosphere_at=at), dict(
        thermodynamics=thermodynamics, rosseland_opacity=opacity,
        mixing_length_alpha=1.25,
        with_temperature=lambda t: runner.atmosphere_at(
            5000., pressure, t, tau, logg=logg,
        ),
    ))
    assert observed_gravities and set(observed_gravities) == {logg}
    assert result.logg == logg and info['corrected_interfaces'] > 0
    assert result.temperature[0] == seed.temperature[0]
    np.testing.assert_array_equal(result.column_mass, seed.column_mass)

    # Evaluate the public ML2 flux on the corrected interfaces independently
    # of the projection's root residual and its reported flux diagnostic.
    thermo = thermodynamics(result)
    positive, arithmetic = _positive_interface_values, _arithmetic_interface_values
    interface = replace(result, temperature=positive(result.temperature),
                        gas_pressure=positive(result.gas_pressure),
                        mass_density=positive(result.mass_density))
    gradient = np.r_[0., np.diff(np.log(result.temperature)) / np.diff(np.log(pressure))]
    flux = ml2_convective_flux_for_gradient_from_thermodynamics(
        interface, positive(opacity(result)), gradient,
        arithmetic(thermo.specific_heat_constant_pressure),
        arithmetic(thermo.density_temperature_derivative),
        arithmetic(thermo.adiabatic_temperature_gradient),
        mixing_length_alpha=1.25,
    )
    assert np.all(np.isfinite(flux)) and np.all(flux >= 0)
    assert np.max(flux) / (STEFAN_BOLTZMANN * seed.effective_temperature**4) <= 1 + 1e-6
