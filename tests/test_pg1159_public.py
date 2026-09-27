from dataclasses import dataclass,replace
from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra import PG1159Config,compute_pg1159
from wd_spectra.models import select_physics,ModelData,save_model_result
from wd_spectra.models import pg1159 as public
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.spectrum import planck_lambda_angstrom


def test_selection_is_explicit_and_missing_data_is_actionable(tmp_path):
    assert select_physics(PG1159Config()).workflow=='pg1159'
    assert PG1159Config().population_tolerance == pytest.approx(1e-2)
    with pytest.raises(FileNotFoundError,match='PG1159 requires atomic data'):
        compute_pg1159(data=ModelData(tmp_path))
    with pytest.raises(TypeError,match='initial_atmosphere'):
        compute_pg1159(initial_atmosphere=object())




@pytest.mark.parametrize('changes',[
    {'effective_temperature':np.nan},{'mass_fractions':{'He':1.}},
    {'mass_fractions':{'He':.5,'C':.4,'O':-.1}}, {'oxygen_atom':'unknown'},
    {'population_maximum_iterations':0},{'population_tolerance':0.},
    {'population_tolerance':1.01e-2}])
def test_bad_input_fails_before_reading_data(changes,tmp_path):
    with pytest.raises(ValueError):
        compute_pg1159(replace(PG1159Config(),**changes),data=ModelData(tmp_path))


@pytest.mark.parametrize("trace_failure", [False, True])
def test_public_result_has_flux_metadata_and_saves(tmp_path,monkeypatch,trace_failure):
    data=ModelData(tmp_path/'data')
    for p in public.required_atomic_files(data,'extended54-complete'):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    (data.stout/'stout').mkdir(parents=True)
    @dataclass(frozen=True)
    class Helium:
        population_n_angle:int=2
        hydrogenic_collision_model:str="ccc-scaled"
        @property
        def collision_data(self):return SimpleNamespace(maximum_level=8)
    @dataclass(frozen=True)
    class Material:
        helium_model:object=Helium()
        population_transfer:str='mass'
        use_population_ali:bool=True
        coupled_population_acceleration_depth:int=0
        metal_population_acceleration_depth:int=4
        metal_population_damping:float=.25
        helium_population_damping:float=.4
        metal_population_relative_tolerance:float=1e-4
        carbon_levels_per_charge:object=None
        oxygen_levels_per_charge:object=None
        def rebuild_atmosphere(self,a,t,p):return a
        def transfer_coefficients(self,a,w,p):
            b=planck_lambda_angstrom(w[:,None],a.temperature[None,:])
            return NLTETransferCoefficients(w,np.ones_like(b),b,np.full_like(b,.1),{})
    model=Material(carbon_levels_per_charge={2:1},oxygen_levels_per_charge={5:1})
    from wd_spectra import _pg1159_reference as reference
    monkeypatch.setattr(reference,'planck_state',lambda model,a,w:(a,None))
    monkeypatch.setattr(public,'build_model',lambda *args,**kw:(model,None))
    monkeypatch.setattr(public,'structure_wavelength',lambda *args:np.geomspace(50.,10000.,60))
    seen={}
    def solve(a,m,w,**kwargs):
        seen.update(
            teff=a.effective_temperature,
            logg=a.logg,
            structure_acceleration_depth=m.coupled_population_acceleration_depth,
        )
        return SimpleNamespace(atmosphere=replace(a,metadata={'radiative_equilibrium_converged':False}),population_state=SimpleNamespace(converged=trace_failure))
    monkeypatch.setattr(public,'solve_pg1159_atmosphere',solve)
    fractions={'He':.52,'C':.45,'O':.03}
    if trace_failure:
        from wd_spectra import _convergence
        monkeypatch.setattr(_convergence,'recorded_equilibrium_status',lambda metadata:'converged')
        class FormalEquations:
            def __init__(self,a,*args,**kwargs):
                self.a=a
                seen['formal_tolerance']=args[0].metal_population_relative_tolerance
                seen['formal_acceleration_depth']=(
                    args[0].coupled_population_acceleration_depth
                )
            def initial_state(self):return np.log(self.a.temperature)
            def material(self,x):return self.a,SimpleNamespace(converged=False)
        monkeypatch.setattr(public,'PG1159Equations',FormalEquations)
        fractions['N']=.01
    config=PG1159Config(quality='quick',mass_fractions=fractions)
    with pytest.warns(RuntimeWarning,match='equilibrium certificate'):
        result=compute_pg1159(config,np.linspace(3800.,6800.,40),data=data)
    assert seen['teff'] == 110000.
    assert seen['logg'] == 7.
    assert seen['structure_acceleration_depth'] == 80
    assert result.metadata['cold_start'] is True
    assert result.metadata['ccc_maximum_shell']==8
    assert result.metadata['helium_ii_collision_model']=='ccc-scaled'
    assert result.metadata['structure_continuum_points']==120
    assert result.metadata['structure_angle_points']==2
    if trace_failure:
        assert seen['formal_tolerance'] == pytest.approx(1e-2)
        assert seen['formal_acceleration_depth'] == 80
        assert result.metadata['structure_convergence_status']=='converged'
        assert not result.population_state.line_formation.converged
        assert result.population_state.structure.converged
    assert result.metadata['atmosphere_convergence_status']=='unconverged'
    assert result.spectrum.metadata['flux_convention']=='surface F_lambda'
    assert result.spectrum.metadata['wavelength_medium']=='vacuum'
    assert result.spectrum.metadata['flux_unit']=='erg s^-1 cm^-2 Angstrom^-1'
    assert np.all(np.isfinite(result.spectrum.surface_flux_lambda))
    # Stub state is intentionally not a production dataclass; serialization
    # of the actual nested states is covered by the shared ModelResult tests.
    save_model_result(replace(result,population_state=None),tmp_path/'output')
    assert (tmp_path/'output/spectrum.txt').is_file()


def test_separate_population_material_states_serialize_without_pickle(tmp_path):
    import json
    from wd_spectra.atmosphere import gray_helium_atmosphere
    from wd_spectra.models.common import ModelResult
    from wd_spectra.spectrum import Spectrum
    from wd_spectra.models.pg1159 import PG1159PopulationResult
    a=gray_helium_atmosphere(110000.,7.,n_depth=8)
    formal=replace(a,electron_density=a.electron_density*.99)
    @dataclass(frozen=True)
    class State:
        populations:object
        metadata:object
    states=PG1159PopulationResult(
        State({'C':np.ones((3,8))},{'defect':0.}),
        State({'C':np.full((3,8),2.)},{'defect':np.inf}),formal)
    save_model_result(ModelResult('PG1159',a,Spectrum(np.array([4000.,5000.]),np.ones(2),{}),
        PG1159Config(),{},states),tmp_path)
    with np.load(tmp_path/'populations.npz',allow_pickle=False) as saved:
        np.testing.assert_equal(saved['structure.populations.C'],1.)
        np.testing.assert_equal(saved['line_formation.populations.C'],2.)
        np.testing.assert_allclose(saved['line_formation_atmosphere.electron_density'],formal.electron_density)
        metadata=json.loads(str(saved['metadata_json']))
        assert metadata['line_formation']['metadata']['defect'] is None


def test_cold_continuation_introduces_full_force_only_with_full_nlte(monkeypatch):
    from wd_spectra import _pg1159_structure as structure
    from wd_spectra.atmosphere import gray_helium_atmosphere
    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction:float=1.
        metal_population_damping:float=.25
        helium_population_damping:float=.4
        metal_population_acceleration_depth:int=4
        coupled_population_acceleration_depth:int=80
        metal_population_relative_tolerance:float=1e-4
        metal_population_iterations:int=120
        use_pg1159_response_jacobian:bool=False
    calls=[]
    seed=gray_helium_atmosphere(110000.,7.,n_depth=8)
    def stage(a,m,w,**kwargs):
        calls.append((m.population_nlte_fraction,m.coupled_population_acceleration_depth,kwargs['include_radiative_acceleration'],kwargs['radiative_acceleration_scale'],kwargs['certification_stage'],kwargs['material_tolerance_ceiling'],kwargs['stage_name']))
        return structure.PG1159AtmosphereResult(replace(a,metadata={'elapsed_seconds':1.}),
            object(),SimpleNamespace(converged=True,iterations=1))
    monkeypatch.setattr(structure,'_solve_stage',stage)
    result=structure.solve_pg1159_atmosphere(seed,Model(),np.array([100.,1000.]),include_radiative_acceleration=True,cold_start=True)
    assert calls==[
        (0.,80,False,0.,False,1e-4,'planck-initializer'),
        (.5,80,True,.5,False,1e-2,'half-nlte-population-bridge'),
        (1.,80,True,1.,False,1e-2,'full-nlte-relaxation'),
        (1.,80,True,1.,True,1e-4,'full-nlte-certification'),
    ]
    assert result.atmosphere.metadata['initialization']['previous_model_supplied'] is False


def test_caller_supplied_seed_does_not_claim_cold_provenance(monkeypatch):
    from wd_spectra import _pg1159_structure as structure
    from wd_spectra.atmosphere import gray_helium_atmosphere
    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction:float=0.
    seed=gray_helium_atmosphere(110000.,7.,n_depth=8)
    def stage(a,m,w,**kwargs):
        return structure.PG1159AtmosphereResult(replace(a,metadata={'elapsed_seconds':1.}),
            object(),SimpleNamespace(converged=True,iterations=1))
    monkeypatch.setattr(structure,'_solve_stage',stage)
    result=structure.solve_pg1159_atmosphere(seed,Model(),np.array([100.,1000.]))
    assert result.atmosphere.metadata['cold_start'] is False
    assert result.atmosphere.metadata['initialization']['previous_model_supplied'] is None
