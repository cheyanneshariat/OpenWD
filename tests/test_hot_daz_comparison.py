"""Resolution, bin integration and velocity checks independent of the atom solver."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
research=Path(__file__).resolve().parents[1]/'research'
sys.path.insert(0,str(research))
spec=importlib.util.spec_from_file_location('comparison',research/'compare_hot_daz_benchmark.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def test_flat_spectrum_preserved_by_instrument_and_bin_averaging():
    w=np.arange(1390.,1400.,.002); obs=np.arange(1392.,1395.,.013)
    flux=comparison.instrument_sample(w,np.ones(len(w)),obs,velocity=0.)
    np.testing.assert_allclose(flux,1.,rtol=1e-10)


def test_velocity_shift_and_equivalent_width_are_preserved():
    w=np.arange(1390.,1400.,.001);f=1-.5*np.exp(-.5*((w-1393.7546)/.035)**2)
    obs=np.arange(1392.,1395.,.003); factor=comparison.doppler_factor(23.8)
    result=comparison.instrument_sample(w,f,obs)
    assert obs[np.argmin(result)]==pytest.approx(1393.7546*factor,abs=.003)
    ew=trapezoid(1-result*factor,obs)
    assert ew==pytest.approx(.5*np.sqrt(2*np.pi)*.035*factor,rel=1e-3)


def test_missing_segment_is_rejected():
    w=np.r_[np.arange(1173.,1178.,.002),np.arange(1392.,1396.,.002)]
    with pytest.raises(ValueError,match='gap'):
        comparison.instrument_sample(w,np.ones(len(w)),np.linspace(1200.,1220.,100))


def test_hi_screen_has_saturated_core_and_transparent_distant_wings():
    w=np.array([1190.,1215.6701*comparison.doppler_factor(19.4),1240.])
    trans=comparison.ism_lyalpha(w)
    assert trans[1]<1e-10
    assert np.all(trans[[0,2]]>.99)


def test_comparison_cannot_mislabel_other_parameters_as_published_benchmark():
    record=dict(host_parameters=dict(teff=52500.,logg=7.53,log_h_he=5.),
                abundances=dict(C=np.log10(1.72e-7),Si=np.log10(3.68e-7)))
    comparison.verify_benchmark_parameters(record)
    record['host_parameters']['teff']=60000.
    with pytest.raises(ValueError,match='teff'):
        comparison.verify_benchmark_parameters(record)
    record['host_parameters']['teff']=52500.
    record['abundances']['C']=-6.
    with pytest.raises(ValueError,match='C/H'):
        comparison.verify_benchmark_parameters(record)


def test_local_fuse_registration_recovers_a_centroid_without_a_model():
    from register_fuse_diagnostics import fit_centroid,gaussian_bins
    wave=np.arange(1127.5,1128.7,.04)
    catalogue_center=1128.064;offset=.061
    flux=1-.4*gaussian_bins(wave-.02,wave+.02,catalogue_center+offset,.039)
    error=np.full(len(wave),.001)
    flux+=np.random.default_rng(42).normal(0,error)
    result,_=fit_centroid(wave,flux,error,catalogue_center)
    assert result['accepted']
    assert result['offset_angstrom']==pytest.approx(offset,abs=.001)


def test_local_registration_is_not_extrapolated_and_has_correct_sign():
    from compare_hot_composition import registration_offset
    registration={'regions':[{'rest_min':1121.5,'rest_max':1129.2,'offset_angstrom':.06}]}
    assert registration_offset(registration,1117.977)==0.
    offset=registration_offset(registration,1128.008)
    factor=comparison.doppler_factor(14.88)+offset/1128.008
    corrected=comparison.C_KMS*(factor**2-1)/(factor**2+1)
    assert corrected>14.88
    assert 1128.008*(comparison.doppler_factor(corrected)-comparison.doppler_factor(14.88))==pytest.approx(.06)
    registration['regions']*=2
    with pytest.raises(ValueError,match='overlapping'):
        registration_offset(registration,1128.008)
