import numpy as np
import pytest
from dense_temperature_predictor import predicted_temperature


def test_predictor_scales_at_fixed_optical_depth():
    tau=np.geomspace(1e-8,100.,80)
    temperature=5000*(.75*(tau+2/3))**.25
    np.testing.assert_allclose(predicted_temperature(tau,temperature,5000,tau,8000),
        temperature*1.6,rtol=2e-15)


def test_predictor_refuses_extrapolation():
    with pytest.raises(ValueError,match='extrapolation'):
        predicted_temperature(np.array([1.,2.,3.]),np.ones(3)*5000,5000,np.array([.9]),8000)


def test_pressure_seed_hook_is_installed_before_runner_and_restored(monkeypatch):
    import sys
    import check_cool_db_transport_seed as runner
    import dense_temperature_predictor as predictor_module
    import dense_helium_molecular_experiment as experiment
    import extended_thermal_wavelength_experiment as wrapper
    original=runner.transport_seed
    class Predictor:
        reuse_pressure=True
        heminus_join_policy='historical-hard-domain-switch'
        description='explicit test continuation, not cold start'
        def __init__(self,*args,**kwargs):pass
        def seed(self,*args,**kwargs):return 'retained-pressure-test'
    def run():
        assert runner.transport_seed(6000,80,100)=='retained-pressure-test'
    monkeypatch.setattr(predictor_module,'TemperaturePredictor',Predictor)
    monkeypatch.setattr(experiment,'main',run)
    monkeypatch.setattr(sys,'argv',['test','--temperature-profile-predictor','unused',
        '--predictor-reuse-pressure','--interaction-table','unused'])
    wrapper.main()
    assert runner.transport_seed is original


def test_explicit_depth_truncation_refines_inside_source_domain(monkeypatch):
    from dense_temperature_predictor import TemperaturePredictor
    from types import SimpleNamespace
    import check_cool_db_transport_seed as runner
    predictor=TemperaturePredictor.__new__(TemperaturePredictor)
    predictor.pressure=np.geomspace(1e5,1e12,20)
    predictor.tau=np.geomspace(1e-8,1e5,20)
    predictor.temperature=2000*(predictor.pressure/1e5)**.1
    predictor.teff=6000.
    predictor.logg=7.75
    predictor.truncate_optical_depth=100.
    predictor.refine_pressure=True
    predictor.unscaled_temperature=False
    monkeypatch.setattr(runner,'atmosphere_at',lambda teff,p,t,tau,*,logg:
        SimpleNamespace(pressure=p,temperature=t,tau=tau,logg=logg))
    seed=predictor.seed(7000,40,100,logg=7.75)
    assert seed.logg==7.75
    with pytest.raises(ValueError,match='same recorded gravity'):
        predictor.seed(7000,40,100,logg=8.)
    assert len(seed.pressure)==40
    assert np.all(np.diff(seed.pressure)>0)
    np.testing.assert_allclose(seed.tau[-1],100,rtol=1e-14)
    assert seed.pressure[-1]<predictor.pressure[-1]
    np.testing.assert_allclose(seed.temperature,
        2000*(seed.pressure/1e5)**.1*7000/6000,rtol=1e-14)
