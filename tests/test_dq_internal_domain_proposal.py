from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra._domain import append_lower_domain
from wd_spectra._dq.dq_internal_domain_proposal import physical_extension_options,flux_balanced_extension,initialize_physical_extension
from wd_spectra.constants import STEFAN_BOLTZMANN


def states():
    p=np.geomspace(1e5,1e10,6)
    previous=SimpleNamespace(n_depth=6,effective_temperature=5364.,logg=7.885,
        gravity=10**7.885,gas_pressure=p,temperature=4000*(p/p[0])**.1,
        column_mass=p/10**7.885,rosseland_optical_depth=np.geomspace(1e-5,10,6),
        metadata=dict(maximum_all_depth_total_flux_residual=1e-4,
            maximum_relative_cell_energy_balance_residual=1e-4,
            maximum_unrestricted_log_temperature_correction=1e-5,
            temperature_correction_measured=True,
            electron_scattering_source_final_maximum_relative_residual=1e-9,
            lower_boundary_absorption_escape_bound=.1,radiative_equilibrium_solver_converged=True))
    extended=append_lower_domain(previous)
    seed=SimpleNamespace(n_depth=extended['n_depth'],effective_temperature=previous.effective_temperature,
        logg=previous.logg,**{f:extended['initial_'+f] for f in
            ('temperature','gas_pressure','column_mass','rosseland_optical_depth')})
    options=dict(flux_tolerance=.002,temperature_tolerance=.0002,
        enforce_local_energy_balance=True,use_convective_gradient_preconditioner=True,
        metadata=dict(protected='value'))
    return previous,seed,options


def test_exact_same_run_extension_uses_physical_rows_without_relaxing_gates():
    previous,seed,options=states()
    changed=physical_extension_options(previous,seed,options)
    assert not changed['use_convective_gradient_preconditioner']
    assert not changed['project_initial_convective_gradient']
    assert not changed['use_initial_bolometric_rescaling']
    assert changed['enforce_local_energy_balance']
    assert changed['flux_tolerance']==options['flux_tolerance']
    assert changed['temperature_tolerance']==options['temperature_tolerance']
    assert changed['metadata']['protected']=='value'
    assert options['use_convective_gradient_preconditioner']
    assert options['metadata']==dict(protected='value')


def test_exact_first_cell_of_canonical_extension_uses_physical_rows():
    previous,seed,options=states()
    count=previous.n_depth+1
    seed=SimpleNamespace(**{
        **vars(seed),
        'n_depth':count,
        **{name:getattr(seed,name)[:count].copy() for name in
           ('temperature','gas_pressure','column_mass','rosseland_optical_depth')},
    })
    changed=physical_extension_options(previous,seed,options)
    assert changed is not options
    assert not changed['use_convective_gradient_preconditioner']


@pytest.mark.parametrize('change',['absent','same_mesh','other_teff','other_logg',
    'one_ulp_temperature','pressure','mass','tau','flux','energy','stationarity','source','solver'])
def test_no_shortcut_for_unqualified_or_unrelated_states(change):
    previous,seed,options=states()
    if change=='absent':previous=None
    elif change=='same_mesh':seed=previous
    elif change=='other_teff':seed.effective_temperature+=1.
    elif change=='other_logg':seed.logg+=.01
    elif change=='one_ulp_temperature':seed.temperature[0]=np.nextafter(seed.temperature[0],np.inf)
    elif change in ('pressure','mass','tau'):
        key=dict(pressure='gas_pressure',mass='column_mass',tau='rosseland_optical_depth')[change]
        getattr(seed,key)[-1]*=1.01
    else:
        key=dict(flux='maximum_all_depth_total_flux_residual',
            energy='maximum_relative_cell_energy_balance_residual',
            stationarity='maximum_unrestricted_log_temperature_correction',
            source='electron_scattering_source_final_maximum_relative_residual',
            solver='radiative_equilibrium_solver_converged')[change]
        previous.metadata[key]=False if change=='solver' else 1.
    assert physical_extension_options(previous,seed,options) is options


def test_extension_seed_reactivates_reservoir_without_editing_old_physical_nodes():
    p=np.exp(np.arange(6)+10.)
    seed=SimpleNamespace(n_depth=6,gas_pressure=p,temperature=4000*np.exp(.4*np.arange(6)),
        effective_temperature=5000.,gravity=1e8)
    target=STEFAN_BOLTZMANN*seed.effective_temperature**4
    evaluated=[]
    class Material:
        def fields(self,logt):
            evaluated.append(logt.copy())
            fields=np.ones((5,6));fields[1]=1e8
            return SimpleNamespace(temperature=np.exp(logt)),fields
        def assemble(self,current,fields):
            # Increasing density lowers nabla_ad: simply extending the old
            # .4 gradient would grossly overload these efficient cells.
            ad=.4-.02*np.arange(6)+.002*np.log(current.temperature/4000)
            return ad,np.ones(6)*1e-3,np.ones(6)*target*1e7
    temperatures,rows=flux_balanced_extension(seed,4,Material())
    np.testing.assert_array_equal(temperatures[:3],seed.temperature[:3])
    assert np.all(temperatures[3:]<seed.temperature[3:])
    assert rows[0]['node']==3
    assert rows[0]['role']=='reactivated_terminal_reservoir'
    assert all(row['role']=='appended_node' for row in rows[1:])
    assert len(evaluated)>2
    for row in rows:
        assert row['convective_over_target']<=1.000001
        assert row['radiative_diffusion_over_target']+row['convective_over_target']==pytest.approx(1.,abs=1e-6)


def test_radiative_extension_has_no_forced_convective_flux():
    p=np.exp(np.arange(4)+10.)
    seed=SimpleNamespace(n_depth=4,gas_pressure=p,temperature=np.ones(4)*5000.,
        effective_temperature=5000.,gravity=1e8)
    class Material:
        def fields(self,logt):return None,np.ones((5,4))
        def assemble(self,current,fields):return np.ones(4)*.4,np.ones(4),np.ones(4)
    temperatures,rows=flux_balanced_extension(seed,3,Material())
    assert temperatures[2]==seed.temperature[2]
    assert rows[0]['role']=='preserved_terminal_reservoir_stable'
    assert rows[0]['convective_over_target']==0.
    assert rows[1]['role']=='appended_node'
    assert rows[1]['convective_over_target']==0.
    assert rows[1]['radiative_diffusion_over_target']==pytest.approx(1.,abs=1e-6)


def test_unrelated_or_cold_seed_never_invokes_extension_material():
    _,seed,options=states()
    def forbidden(*args):raise AssertionError('Unqualified initializer invoked')
    result,settings=initialize_physical_extension(None,seed,options,forbidden)
    assert result is seed and settings is options


def test_initializer_preserves_mesh_and_passes_real_seed_to_formal_solve(monkeypatch):
    import wd_spectra._dq.dq_internal_domain_proposal as module
    previous,seed,options=states()
    initial=seed.temperature.copy()
    wanted=initial.copy();wanted[previous.n_depth:]*=.99
    def with_temperature(t):
        return SimpleNamespace(**{**vars(seed),'temperature':np.asarray(t)})
    options.update(with_temperature=with_temperature,thermodynamics=object(),
        rosseland_opacity=object(),mixing_length_alpha=1.25)
    marker=object()
    def initialize(s,old_depth,material):
        assert s is seed and old_depth==previous.n_depth and material is marker
        return wanted,[dict(node=previous.n_depth)]
    monkeypatch.setattr(module,'flux_balanced_extension',initialize)
    result,settings=initialize_physical_extension(previous,seed,options,lambda *a:marker)
    np.testing.assert_array_equal(result.temperature,wanted)
    np.testing.assert_array_equal(seed.temperature,initial)
    for name in ('gas_pressure','column_mass','rosseland_optical_depth'):
        np.testing.assert_array_equal(getattr(result,name),getattr(seed,name))
    assert settings['metadata']['dq_domain_seed_fluxes']==[dict(node=previous.n_depth)]
    assert settings['metadata']['dq_minimum_temperature_gradient_index']==previous.n_depth
    assert settings['flux_tolerance']==options['flux_tolerance']
    assert settings['temperature_tolerance']==options['temperature_tolerance']
    assert not settings['project_initial_convective_gradient']


def test_earliest_reactivated_gradient_floor_persists_across_extensions(monkeypatch):
    import wd_spectra._dq.dq_internal_domain_proposal as module
    previous,seed,options=states()
    previous.metadata['dq_minimum_temperature_gradient_index']=3
    options.update(with_temperature=lambda t:SimpleNamespace(
            **{**vars(seed),'temperature':np.asarray(t)}),
        thermodynamics=object(),rosseland_opacity=object(),mixing_length_alpha=1.25)
    monkeypatch.setattr(module,'flux_balanced_extension',
        lambda s,old,material:(s.temperature.copy(),[dict(node=old)]))
    _,settings=initialize_physical_extension(previous,seed,options,lambda *a:object())
    assert settings['metadata']['dq_minimum_temperature_gradient_index']==3


def test_material_domain_failure_is_not_hidden_by_extrapolation():
    previous,seed,options=states()
    seed.gravity=previous.gravity
    class Material:
        def fields(self,*args):raise ValueError('unsupported EOS cell')
    with pytest.raises(ValueError,match='unsupported EOS cell'):
        flux_balanced_extension(seed,previous.n_depth,Material())
