"""Independent full-residual checks of the joint Newton research candidate."""
from dataclasses import replace
import numpy as np
import pytest
from wd_spectra.atmosphere import gray_helium_atmosphere,gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import HotNLTEModel,population_arrays
from wd_spectra.models.common import ModelData
from wd_spectra.helium_stark import read_helium_stark_table
from wd_spectra.helium_ii_stark import read_helium_ii_stark_table
from wd_spectra.multilevel_nlte import read_ccc_hydrogen_collision_data
from test_helium_nlte import _write_small_ccc_archive
from wd_spectra._hot_structure import HotEquations
from test_hot_nlte import atom
from wd_spectra._hot_rates import (PreparedHeliumRates, PreparedLineAverages,
    HeliumRadiationResponse, MixedRadiationResponse)
from wd_spectra.nlte import _profile_averaged_mean_intensity_nu
from wd_spectra import helium_nlte as he
from wd_spectra.spectrum import planck_lambda_angstrom

@pytest.mark.parametrize('ratio,fraction,hydrogen_levels',
    [(ratio,fraction,3) for ratio in (None,2.) for fraction in (0.,.3,1.)] + [(6.,1.,20)])
def test_joint_material_columns_match_full_coupled_residual(tmp_path,ratio,fraction,hydrogen_levels):
    p=tmp_path/'ccc.zip';_write_small_ccc_archive(p,maximum_level=3)
    d=ModelData.default()
    model=HotNLTEModel(read_ccc_hydrogen_collision_data(p,maximum_level=3),
        read_helium_stark_table(d.cache/'helium-stark'/'Tremblay26.txt'),read_helium_ii_stark_table(d.helium_ii_stark),None,
        maximum_helium_ii_level=3,maximum_hydrogen_level=hydrogen_levels,n_angle=2,log_hydrogen_to_helium=ratio)
    a=gray_helium_atmosphere(60000,8,n_depth=3) if ratio is None else gray_hydrogen_helium_atmosphere(60000,8,ratio,n_depth=3)
    e=HotEquations(a,model,np.geomspace(25,100000,80),nlte_fraction=fraction)
    x=e.initial_state();x[3:]+=np.linspace(-.02,.02,len(x)-3)
    if fraction==1.:
        # Simulate an abandoned trial that updated atomic caches but failed
        # before replacing the cached base residual (e.g. boundary screening).
        e.residual(x)
        trial=x.copy();trial[:3]+=.01
        ta,tr,tg,_=e.prepare(trial[:3])
        mean=planck_lambda_angstrom(e.wave[:,None],ta.temperature[None,:])
        e.rate_residual(trial,ta,tr,tg,mean,prepare_cache=True)
    evaluated=e.evaluate(x,True)
    # Full-grid temperature perturbation profiles must not survive into the
    # population columns or subsequent solves (gigabytes on production grids).
    assert list(e.prepared) == [x[:3].tobytes()]
    for k in sorted({0,2,len(x)//2,len(x)-1}|({3,18} if fraction else set())):
        # The trace-He/20-shell system has a cancellation floor in tiny
        # cross-depth rate derivatives. A step-size sweep resolves it at
        # 3e-5 without changing the derivative comparison tolerance.
        dx=np.zeros_like(x);dx[k]=3e-5 if hydrogen_levels == 20 else 1e-5
        numerical=(e.residual(x+dx).residual-e.residual(x-dx).residual)/(2*dx[k])
        np.testing.assert_allclose(evaluated.jacobian[:,k],numerical,rtol=2e-5,atol=1e-7)
    actual,reference=population_arrays(evaluated.payload[1])
    np.testing.assert_allclose(actual[:,:e.nhe].sum(axis=1),reference[:,:e.nhe].sum(axis=1),rtol=3e-15)
    if ratio is not None:
        np.testing.assert_allclose(actual[:,e.nhe:].sum(axis=1),reference[:,e.nhe:].sum(axis=1),rtol=3e-15)


@pytest.mark.parametrize('levels',[3,32])
def test_prepared_rates_match_uncached_atomic_matrix(atom,levels):
    model=replace(atom,maximum_helium_ii_level=levels)
    a=gray_helium_atmosphere(60000,8,n_depth=4)
    wave=np.geomspace(25,100000,200)
    groups=model._line_problems(a)
    cache=PreparedHeliumRates(model,a,wave,groups)
    fields=[{key:np.linspace(.002,.008,a.n_depth)*(1+.01*i)
             for i,key in enumerate(group)} for group in groups]
    for factor in (.3,1.7):
        mean=planck_lambda_angstrom(wave[:,None],a.temperature[None,:])*factor
        expected=he.solve_coupled_helium_statistical_equilibrium(a,model.collision_data,
            **cache.kwargs,neutral_line_mean_intensity_nu=fields[0],helium_ii_line_mean_intensity_nu=fields[1],
            neutral_continuum_wavelength_angstrom=wave,neutral_continuum_mean_intensity_lambda=mean,
            helium_ii_continuum_wavelength_angstrom=wave,helium_ii_continuum_mean_intensity_lambda=mean,
            _return_rate_matrix=True)
        np.testing.assert_allclose(cache.rate_matrix(mean,*fields[:2]),expected,rtol=3e-14,atol=0)
        actual,_=population_arrays(cache.state(mean,*fields))
        original,_=population_arrays(model._rate_state(a,wave,mean,*fields))
        np.testing.assert_allclose(actual,original,rtol=2e-10)

    line_wave=np.unique(np.concatenate([p.continuum.wavelength_angstrom
        for group in groups for components in group.values() for p in components]))
    mean=planck_lambda_angstrom(line_wave[:,None],a.temperature[None,:])
    averaged=PreparedLineAverages(line_wave,groups,a.n_depth).fields(mean)
    for group,actual in zip(groups,averaged):
        for key,problems in group.items():
            expected=np.average([_profile_averaged_mean_intensity_nu(
                p.continuum.wavelength_angstrom,p.lte_line_opacity,
                mean[np.searchsorted(line_wave,p.continuum.wavelength_angstrom)])
                for p in problems],axis=0,weights=[p.line.absorption_oscillator_strength for p in problems])
            np.testing.assert_allclose(actual[key],expected,rtol=3e-14,atol=0)


@pytest.mark.parametrize('levels',[3,32])
def test_analytic_helium_radiation_response_matches_full_rate_solve(atom,levels):
    model=replace(atom,maximum_helium_ii_level=levels)
    a=gray_helium_atmosphere(60000,8,n_depth=4)
    groups=model._line_problems(a)
    wave=np.unique(np.concatenate([np.geomspace(25,100000,200)]+[
        p.continuum.wavelength_angstrom for group in groups for problems in group.values() for p in problems]))
    mean=planck_lambda_angstrom(wave[:,None],a.temperature[None,:])*.7
    rates=PreparedHeliumRates(model,a,wave,groups)
    profiles=PreparedLineAverages(wave,groups,a.n_depth)
    response=HeliumRadiationResponse(rates,profiles,mean)
    rng=np.random.default_rng(31)
    direction=mean[:,:,None]*rng.uniform(-.5,.5,(len(wave),a.n_depth,3))
    actual=response.log_ratio_response(direction)
    eps=2e-5
    for i in range(3):
        values=[]
        for sign in (1,-1):
            probe=mean+sign*eps*direction[:,:,i]
            population,_=population_arrays(model._rate_state(a,wave,probe,*profiles.fields(probe)))
            values.append(np.log(population[:,:-1]/population[:,-1:]))
        np.testing.assert_allclose(actual[:,:,i],(values[0]-values[1])/(2*eps),rtol=3e-4,atol=2e-7)


def test_fixed_temperature_continuation_cannot_certify_an_atmosphere(atom,monkeypatch):
    import wd_spectra._hot_structure as joint
    a=gray_helium_atmosphere(60000,8,n_depth=3)
    a=replace(a,metadata={**a.metadata,'radiative_equilibrium_iterations':987,
                         'maximum_relative_cell_energy_balance_residual':0.,
                         'cell_energy_balance_relative_residual':[0.,0.]})
    shared=joint.solve_trust_region_newton
    calls=[]
    def spy(*args,**kwargs):
        calls.append(True)
        return shared(*args,**kwargs)
    monkeypatch.setattr(joint,'solve_trust_region_newton',spy)
    result=joint.solve(a,atom,np.geomspace(25,100000,80),fixed_temperature=True,
                       nlte_fraction=0.,maximum_iterations=2)
    assert calls==[True]
    assert result.nonlinear_result.converged
    metadata=result.atmosphere.metadata
    assert not metadata['equilibrium_certificate']['verified']
    assert not metadata['temperature_correction_measured']
    assert not metadata['radiative_equilibrium_solver_converged']
    assert not metadata['hot_nlte_recovery']['eligible']
    assert not metadata['hot_nlte_recovery']['activated']
    assert 'cell_energy_balance_relative_residual' not in metadata
    assert 'radiative_equilibrium_iterations' not in metadata
    assert metadata['initial_atmosphere_metadata']['radiative_equilibrium_iterations']==987


@pytest.mark.parametrize('fraction',[0.,.3,1.])
def test_independent_stationarity_jacobian_is_reserved_for_full_nlte(atom,monkeypatch,fraction):
    import wd_spectra._hot_structure as joint
    original=joint.HotEquations.evaluate
    requests=[]
    def evaluate(self,x,jacobian):
        requests.append(jacobian)
        return original(self,x,jacobian)
    monkeypatch.setattr(joint.HotEquations,'evaluate',evaluate)
    a=gray_helium_atmosphere(60000.,8.,n_depth=3)
    result=joint.solve(a,atom,np.geomspace(25,100000,80),nlte_fraction=fraction,maximum_iterations=1)
    assert requests[-1] == (fraction==1.)
    metadata=result.atmosphere.metadata
    assert metadata['hot_nlte_recovery']['eligible'] == (fraction == 1.)
    # A single attempted direction cannot activate the two-rejection gate.
    assert not metadata['hot_nlte_recovery']['activated']
    assert metadata['hot_nlte_recovery']['extra_jacobian_evaluations'] == 0
    assert metadata['hot_nlte_recovery']['extra_residual_evaluations'] == 0
    assert metadata['hot_nlte_recovery']['thermal_probes'] == 0
    if fraction<1.:
        assert metadata['coupled_jacobian_rank'] is None
        assert metadata['maximum_unrestricted_log_temperature_correction'] is None
        assert not metadata['temperature_correction_measured']
        assert not metadata['equilibrium_certificate']['verified']
    else:
        assert isinstance(metadata['coupled_jacobian_rank'],int)


@pytest.mark.parametrize('levels,ratio',[(3,2.),(8,6.),(20,6.)])
def test_mixed_analytic_radiation_response_matches_independent_atomic_solve(atom,levels,ratio):
    model=replace(atom,maximum_hydrogen_level=levels,log_hydrogen_to_helium=ratio)
    a=gray_hydrogen_helium_atmosphere(40204.,7.82,ratio,n_depth=4)
    groups=model._line_problems(a)
    wave=np.unique(np.concatenate([np.geomspace(25,100000,200)]+[
        p.continuum.wavelength_angstrom for group in groups for problems in group.values() for p in problems]))
    mean=planck_lambda_angstrom(wave[:,None],a.temperature[None,:])*.65
    rates=PreparedHeliumRates(model,a,wave,groups)
    profiles=PreparedLineAverages(wave,groups,a.n_depth)
    response=MixedRadiationResponse(rates,profiles,mean)
    direction=mean[:,:,None]*np.random.default_rng(51).uniform(-.5,.5,(len(wave),a.n_depth,3))
    actual=response.log_ratio_response(direction)
    nhe=15+model.maximum_helium_ii_level
    # Trace He at high density makes tiny finite differences dominated by
    # cancellation in the independent SE solve. A step-size sweep resolves
    # that floor while keeping central-difference truncation below tolerance.
    eps=3e-3
    for i in range(3):
        values=[]
        for sign in (1,-1):
            probe=mean+sign*eps*direction[:,:,i]
            population,_=population_arrays(model._rate_state(a,wave,probe,*profiles.fields(probe)))
            values.append(np.column_stack((np.log(population[:,:nhe-1]/population[:,nhe-1:nhe]),
                                           np.log(population[:,nhe:-1]/population[:,-1:]))))
        np.testing.assert_allclose(actual[:,:,i],(values[0]-values[1])/(2*eps),rtol=3e-4,atol=2e-7)


@pytest.mark.parametrize('rank_deficient',[False,True])
def test_finite_svd_failure_retries_identical_newton_system(monkeypatch,rank_deficient):
    from wd_spectra._hot_structure import _least_squares
    matrix=np.random.default_rng(16).normal(size=(12,12))
    if rank_deficient:matrix[:,-1]=matrix[:,0]+matrix[:,1]
    rhs=np.linspace(-1,1,12)
    expected,_,expected_rank,_=np.linalg.lstsq(matrix,rhs,rcond=1e-10)
    def failed(*args,**kwargs):raise np.linalg.LinAlgError('SVD did not converge')
    monkeypatch.setattr(np.linalg,'lstsq',failed)
    actual,rank,_,driver=_least_squares(matrix,rhs,rcond=1e-10)
    assert rank==expected_rank and driver=='scipy-gelss'
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-13)


def test_chunked_radiation_response_matches_independent_atomic_solve(atom, monkeypatch):
    import wd_spectra._hot_rates as rates
    monkeypatch.setattr(rates, '_RADIATION_RESPONSE_WORKING_BYTES', 8192)
    test_mixed_analytic_radiation_response_matches_independent_atomic_solve(atom, 8, 6.)


def test_nonfinite_newton_system_cannot_use_svd_recovery():
    from wd_spectra._hot_structure import _least_squares
    with pytest.raises(ValueError,match='non-finite'):
        _least_squares(np.array([[np.nan]]),np.ones(1),rcond=1e-10)


def test_completed_anchor_reuses_transfer_and_same_radiation_se_candidate(atom, monkeypatch):
    import wd_spectra._hot_structure as joint
    from wd_spectra import multilevel_nlte as hydrogen
    model = replace(atom, log_hydrogen_to_helium=2.)
    a = gray_hydrogen_helium_atmosphere(60000., 8., 2., n_depth=3)
    expected = None
    original_transfer = joint.transfer_field
    original_hydrogen = hydrogen.solve_multilevel_hydrogen_statistical_equilibrium
    counts = {"transfer": 0, "hydrogen_state": 0, "hydrogen_rate": 0}
    def transfer(*args, **kwargs):
        counts["transfer"] += 1
        return original_transfer(*args, **kwargs)
    def hydrogen_solve(*args, **kwargs):
        counts["hydrogen_rate" if kwargs.get('_return_rate_matrix') else "hydrogen_state"] += 1
        return original_hydrogen(*args, **kwargs)
    monkeypatch.setattr(joint, 'transfer_field', transfer)
    monkeypatch.setattr(hydrogen, 'solve_multilevel_hydrogen_statistical_equilibrium', hydrogen_solve)
    for reuse in (False, True):
        equations = HotEquations(a, model, np.geomspace(25., 100000., 80),
                                 fixed_temperature=True, reuse_accepted_state=reuse)
        x = equations.initial_state()
        x += np.linspace(-.015, .015, len(x))
        counts.update(transfer=0, hydrogen_state=0, hydrogen_rate=0)
        result = equations.evaluate(x, True)
        assert counts == {"transfer": 1 if reuse else 2,
                          "hydrogen_state": 1 if reuse else 2, "hydrogen_rate": 1}
        if expected is None:
            expected = result
        else:
            # Reused populations are the SE map evaluated at this radiation,
            # rather than the deliberately perturbed trial populations.
            np.testing.assert_array_equal(result.residual, expected.residual)
            np.testing.assert_array_equal(result.jacobian, expected.jacobian)
        changed = x.copy()
        changed[0] += .002
        counts.update(transfer=0, hydrogen_state=0, hydrogen_rate=0)
        refreshed = equations.evaluate(changed, True)
        assert counts["transfer"] == (1 if reuse else 2)
        assert not np.array_equal(refreshed.residual, result.residual)



@pytest.mark.parametrize('temperature_trial',[False,True])
def test_failed_trial_preserves_completed_transfer_anchor(atom, monkeypatch,temperature_trial):
    import wd_spectra._hot_structure as joint
    a = gray_helium_atmosphere(60000., 8., n_depth=3)
    equations = HotEquations(a, atom, np.geomspace(25., 100000., 80))
    x = equations.initial_state()
    equations.residual(x)
    anchor = equations.last_physics
    original = equations.rate_residual
    def fail_after_rates(*args, **kwargs):
        original(*args, **kwargs)
        raise joint.RecoverableEvaluationError('later physical check failed')
    monkeypatch.setattr(equations, 'rate_residual', fail_after_rates)
    trial = x.copy()
    if temperature_trial:
        trial[:3] += .01
    else:
        trial[3] += .01
    with pytest.raises(joint.RecoverableEvaluationError, match='later physical check failed'):
        equations.residual(trial)
    assert equations.last_physics is anchor
    monkeypatch.setattr(equations, 'rate_residual', original)
    calls = []
    original_transfer = joint.transfer_field
    def transfer(*args, **kwargs):
        calls.append(True)
        return original_transfer(*args, **kwargs)
    monkeypatch.setattr(joint, 'transfer_field', transfer)
    result = equations.evaluate(x, True)
    assert not calls
    clean = HotEquations(a, atom, equations.wave).evaluate(x, True)
    np.testing.assert_array_equal(result.residual, clean.residual)
    np.testing.assert_array_equal(result.jacobian, clean.jacobian)



@pytest.mark.parametrize('before',[False,True])
@pytest.mark.parametrize('override',['coefficients','prepared','response'])
def test_probe_override_cannot_replace_or_hit_canonical_evaluation(atom,before,override):
    a=gray_helium_atmosphere(60000.,8.,n_depth=3)
    equations=HotEquations(a,atom,np.geomspace(25.,100000.,80))
    x=equations.initial_state()
    pa,pr,pg,pc=equations.prepare(x[:3])
    if override=='coefficients':
        c=atom.transfer_coefficients(pa,equations.wave,equations.populations(x,pr),_cache=pc)
        options={'coefficients':replace(c,thermal_emissivity=c.thermal_emissivity*1.015)}
    elif override=='prepared':
        options={'prepared':equations.prepare(x[:3]+.005)}
    else:
        mean=planck_lambda_angstrom(equations.wave[:,None],pa.temperature[None,:])*.9
        options={'response':(mean,np.zeros(a.n_depth))}
    if before:
        probe=equations.residual(x,**options)
        assert equations.last_key is equations.last_evaluation is equations.last_physics is None
        assert equations.jacobian_key is equations.jacobian_evaluation is None
        canonical=equations.evaluate(x,True)
        clean=HotEquations(a,atom,equations.wave).evaluate(x,True)
        np.testing.assert_array_equal(canonical.residual,clean.residual)
        np.testing.assert_array_equal(canonical.jacobian,clean.jacobian)
    else:
        canonical=equations.evaluate(x,True)
        saved=(equations.last_key,equations.last_evaluation,equations.last_physics,
               equations.jacobian_key,equations.jacobian_evaluation)
        probe=equations.residual(x,**options)
        assert equations.last_key == saved[0]
        assert equations.last_evaluation is saved[1]
        assert equations.last_physics is saved[2]
        assert equations.jacobian_key == saved[3]
        assert equations.jacobian_evaluation is saved[4]
        assert equations.evaluate(x,True) is canonical
    assert not np.array_equal(probe.residual,canonical.residual)



def test_reference_switch_does_not_retain_unused_transfer_bundle(atom):
    a=gray_helium_atmosphere(60000.,8.,n_depth=3)
    equations=HotEquations(a,atom,np.geomspace(25.,100000.,80),reuse_accepted_state=False)
    x=equations.initial_state()
    equations.residual(x)
    assert equations.last_physics is None
