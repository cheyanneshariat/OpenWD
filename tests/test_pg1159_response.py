import numpy as np
import pytest
from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra._compat import trapezoid
from wd_spectra._pg1159_transfer import transfer_field
from wd_spectra._mass_feautrier import mass_emissivity_energy
from wd_spectra._pg1159_response import thermal_response


def test_default_local_energy_window_excludes_numerical_surface_boundary():
    from types import SimpleNamespace
    from wd_spectra._pg1159_structure import _local_energy_mask

    atmosphere = gray_helium_atmosphere(
        140000.0, 7.0, n_depth=32, tau_min=1e-8, tau_max=100.0
    )
    mask = _local_energy_mask(atmosphere, SimpleNamespace())
    cell_depth = atmosphere.rosseland_optical_depth[:-1]
    assert np.all(cell_depth[mask] >= 1.0e-5)
    assert np.any(cell_depth[~mask] < 1.0e-5)


def test_material_response_matches_independent_depth_probes():
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    wave=np.geomspace(50.,10000.,60)
    target=STEFAN_BOLTZMANN*a.effective_temperature**4
    base=np.log(a.temperature)
    def coefficients(log_t):
        t=np.exp(log_t)
        absorption=np.broadcast_to(.4*(t/a.temperature)**1.7,(len(wave),a.n_depth)).copy()
        scattering=np.full_like(absorption,.2)
        eta=absorption*planck_lambda_angstrom(wave[:,None],t[None,:])
        return NLTETransferCoefficients(wave,absorption,eta,scattering,{})
    def residual(c):
        _,f,_=transfer_field(a,c,n_angle=2,check_source=False)
        flux=trapezoid(f.interface_flux,wave,axis=0)/target-1
        energy,emission=mass_emissivity_energy(wave,a.column_mass,c.thermal_emissivity,f.mean_intensity,c.true_absorption)
        return np.r_[np.where(abs(emission)>=target,np.diff(flux),energy/np.maximum(abs(emission),1e-30*target)),flux[0]]
    step=1e-5
    jac=thermal_response(a,wave,coefficients(base),
        [(coefficients(base+step),coefficients(base-step),step)],target,2)
    numerical=np.empty_like(jac)
    for i in range(a.n_depth):
        delta=np.zeros(a.n_depth);delta[i]=step
        numerical[:,i]=(residual(coefficients(base+delta))-residual(coefficients(base-delta)))/(2*step)
    np.testing.assert_allclose(jac,numerical,atol=2e-7,rtol=2e-5)


def test_flux_profile_response_matches_independent_depth_probes():
    a = gray_helium_atmosphere(90000.0, 7.0, n_depth=8)
    wave = np.geomspace(50.0, 10000.0, 60)
    target = STEFAN_BOLTZMANN * a.effective_temperature**4
    base = np.log(a.temperature)

    def coefficients(log_t):
        temperature = np.exp(log_t)
        absorption = np.broadcast_to(
            0.4 * (temperature / a.temperature) ** 1.7,
            (len(wave), a.n_depth),
        ).copy()
        scattering = np.full_like(absorption, 0.2)
        emissivity = absorption * planck_lambda_angstrom(
            wave[:, None], temperature[None, :]
        )
        return NLTETransferCoefficients(
            wave, absorption, emissivity, scattering, {}
        )

    def residual(log_t):
        _, field, _ = transfer_field(
            a, coefficients(log_t), n_angle=2, check_source=False
        )
        return trapezoid(field.interface_flux, wave, axis=0) / target - 1.0

    step = 1.0e-5
    jacobian = thermal_response(
        a,
        wave,
        coefficients(base),
        [(coefficients(base + step), coefficients(base - step), step)],
        target,
        2,
        flux_profile_residual=True,
    )
    numerical = np.empty_like(jacobian)
    for index in range(a.n_depth):
        delta = np.zeros(a.n_depth)
        delta[index] = step
        numerical[:, index] = (
            residual(base + delta) - residual(base - delta)
        ) / (2.0 * step)
    np.testing.assert_allclose(jacobian, numerical, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("force_scale", [0., .1, 1.])
def test_hydrostatic_response_matches_temperature_and_pressure_probes(force_scale):
    from dataclasses import replace
    from wd_spectra.constants import LIGHT_SPEED
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    wave=np.geomspace(50.,10000.,60)
    target=STEFAN_BOLTZMANN*a.effective_temperature**4
    base=np.r_[np.log(a.temperature),np.log(a.gas_pressure)]
    def coefficients(x):
        t=np.exp(x[:a.n_depth]);p=np.exp(x[a.n_depth:])
        absorption=np.broadcast_to(.4*(t/a.temperature)**1.7*(p/a.gas_pressure)**.4,(len(wave),a.n_depth)).copy()
        scattering=np.broadcast_to(.2*(t/a.temperature)**.2,(len(wave),a.n_depth)).copy()
        return NLTETransferCoefficients(wave,absorption,absorption*planck_lambda_angstrom(wave[:,None],t[None,:]),scattering,{})
    def residual(x):
        c=coefficients(x)
        _,f,_=transfer_field(a,c,n_angle=2,check_source=False)
        effective=a.gravity-force_scale*trapezoid(c.total_extinction*f.flux,wave,axis=0)/LIGHT_SPEED
        pressure=np.r_[effective[0]*a.column_mass[0],effective[0]*a.column_mass[0]+np.cumsum(.5*(effective[1:]+effective[:-1])*np.diff(a.column_mass))]
        return np.log(np.exp(x[a.n_depth:])/pressure)
    step=1e-5;probes=[]
    for offset in (0,a.n_depth):
        delta=np.zeros_like(base);delta[offset:offset+a.n_depth]=step
        probes.append((coefficients(base+delta),coefficients(base-delta),step))
    jac=thermal_response(a,wave,coefficients(base),probes,target,2,hydrostatic=True,radiative_acceleration_scale=force_scale)[a.n_depth:]
    numerical=np.empty_like(jac)
    for i in range(len(base)):
        delta=np.zeros_like(base);delta[i]=step
        numerical[:,i]=(residual(base+delta)-residual(base-delta))/(2*step)
    np.testing.assert_allclose(jac,numerical,atol=2e-7,rtol=2e-5)


def test_material_tangent_rebuilds_ion_and_level_lte_references(monkeypatch):
    """A frozen unit departure must stay LTE when T and density change."""
    from dataclasses import replace
    from types import SimpleNamespace,MappingProxyType
    from wd_spectra import _pg1159_structure as structure
    from wd_spectra.pg1159 import PG1159NLTEState
    from wd_spectra.light_metal_nlte import LightMetalNLTEState,ReducedLightMetalLevelState
    from wd_spectra.constants import PLANCK,LIGHT_SPEED,BOLTZMANN
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    keys=(('C',2,1),('C',2,2),('C',3,1))
    atom=SimpleNamespace(ions={
        ('C',2):SimpleNamespace(levels=(SimpleNamespace(index=1,energy_wavenumber=0.,statistical_weight=1.),SimpleNamespace(index=2,energy_wavenumber=100000.,statistical_weight=2.))),
        ('C',3):SimpleNamespace(levels=(SimpleNamespace(index=1,energy_wavenumber=0.,statistical_weight=1.),))})
    def material(atmosphere):
        total=atmosphere.gas_pressure/(BOLTZMANN*atmosphere.temperature)
        ratio=(atmosphere.temperature/90000.)**4
        ions=np.zeros((4,a.n_depth));ions[2]=total/(1+ratio);ions[3]=total-ions[2]
        partition=1+2*np.exp(-100000.*PLANCK*LIGHT_SPEED/(BOLTZMANN*atmosphere.temperature))
        return SimpleNamespace(element_number_density={'C':total},ion_number_density={'C':ions},partition_function={('C',2):partition,('C',3):np.ones(a.n_depth)})
    def references(atmosphere,metal):
        ground=metal.ion_number_density['C'][2]/metal.partition_function[('C',2)]
        return np.array([ground,metal.ion_number_density['C'][2]-ground,metal.ion_number_density['C'][3]])
    metal=material(a);reference=references(a,metal)
    mapping={key:np.ones(a.n_depth) for key in keys}
    levels=ReducedLightMetalLevelState('C',keys,reference,reference,mapping,0.,{},mapping)
    light=LightMetalNLTEState(metal.ion_number_density,metal.ion_number_density,
        {('C',q):np.ones(a.n_depth) for q in range(4)},{},{},{},{},0.,{})
    state=PG1159NLTEState(object(),metal,light,levels,None,{})
    class Model:
        nlte_charge_feedback=False
        atomic_database=None
        mass_fractions={'He':.5,'C':.5}
        carbon_population_atomic_database=atom
        carbon_lte_level_reservoir=None
        def rebuild_atmosphere(self,a,t,p):return replace(a,temperature=t)
    monkeypatch.setattr(structure,'pg1159_composition_atmosphere',lambda a,*args:(a,material(a)))
    monkeypatch.setattr(structure,'remap_coupled_helium_state',lambda a,s:s)
    new_a,new=structure.remap_material(Model(),a,a.temperature*1.1,a.gas_pressure*.9,state)
    expected=references(new_a,material(new_a))
    np.testing.assert_allclose(new.carbon_level_state.population_density,expected,rtol=1e-14)
    np.testing.assert_allclose(new.carbon_level_state.lte_population_density,expected,rtol=1e-14)
    np.testing.assert_allclose(new.light_metal_state.ion_number_density['C'],material(new_a).ion_number_density['C'],rtol=1e-14)


def test_trial_population_domain_failure_backtracks_but_programming_error_propagates(monkeypatch):
    from wd_spectra._pg1159_structure import PG1159Equations
    from wd_spectra.nlte_core import NonphysicalPopulationError
    from wd_spectra.nonlinear import RecoverableEvaluationError
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    equations=PG1159Equations(a,object(),np.array([100.,1000.]),radiative_acceleration=False)
    def nonphysical(x):
        raise NonphysicalPopulationError('negative trial population')
    monkeypatch.setattr(equations,'material',nonphysical)
    with pytest.raises(RecoverableEvaluationError,match='negative trial population'):
        equations.evaluate(equations.initial_state(),False)
    def broken(x):
        raise RuntimeError('missing atom mapping')
    monkeypatch.setattr(equations,'material',broken)
    with pytest.raises(RuntimeError,match='missing atom mapping'):
        equations.evaluate(equations.initial_state(),False)


def test_signed_species_continuum_is_retained_until_full_mixture_transfer():
    from types import SimpleNamespace
    from wd_spectra.multilevel_nlte import _nlte_continuum_terms
    from wd_spectra.nlte_core import NonphysicalPopulationError
    from wd_spectra._pg1159_transfer import transfer_field
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    wave=np.geomspace(100.,10000.,30)
    b=planck_lambda_angstrom(wave[:,None],a.temperature[None,:])
    problem=SimpleNamespace(bound_free_coefficient=np.ones((*b.shape,1)),
        exp_minus_photon_energy=np.full_like(b,.5),
        thermal_background_absorption=np.zeros_like(b),planck_lambda=b,
        spontaneous_intensity_lambda=b)
    bound=np.full((a.n_depth,1),.1);ion=np.ones(a.n_depth)
    with pytest.raises(NonphysicalPopulationError,match='non-positive total continuum'):
        _nlte_continuum_terms(problem,bound,ion)
    absorption,emission=_nlte_continuum_terms(problem,bound,ion,allow_signed_absorption=True)
    np.testing.assert_allclose(absorption,-.4)
    np.testing.assert_allclose(emission,.5*b)
    combined=NLTETransferCoefficients(wave,absorption+1.,emission+b,np.full_like(b,.1),{})
    _,field,_=transfer_field(a,combined,n_angle=2)
    assert np.all(np.isfinite(field.mean_intensity)) and np.all(field.mean_intensity>=0)


def test_population_transfer_rejects_nonpositive_bottom_as_trial_domain_error():
    from wd_spectra._mass_feautrier import InvalidRadiationFieldError
    from wd_spectra.nlte_core import NonphysicalPopulationError
    from wd_spectra.pg1159 import _population_transfer_extinction
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    wave=np.geomspace(100.,10000.,30)
    absorption=np.ones((len(wave),a.n_depth))
    absorption[:,-1]=-.1
    scattering=np.full_like(absorption,.2)
    emission=np.ones_like(absorption)
    coefficients=NLTETransferCoefficients(wave,absorption,emission,scattering,{})
    with pytest.raises(NonphysicalPopulationError,match='population-transfer'):
        _population_transfer_extinction(absorption,emission,scattering)
    with pytest.raises(InvalidRadiationFieldError,match='bottom absorption'):
        transfer_field(a,coefficients,n_angle=2)


def test_damped_thermal_proposal_handles_weak_modes_without_changing_root():
    from wd_spectra._pg1159_structure import damped_thermal_direction
    from wd_spectra.nonlinear import NonlinearEvaluation
    jacobian = np.diag([1e-3, 1.])
    residual = np.array([.01, .1])
    proposal = damped_thermal_direction(np.zeros(2), NonlinearEvaluation(residual, jacobian, None), jacobian, .04)
    step = proposal.direction
    assert proposal.limited
    assert np.max(abs(step)) <= .04 * (1 + 1e-14)
    radial = np.linalg.solve(jacobian, -residual)
    radial *= .04 / np.max(abs(radial))
    assert np.linalg.norm(residual + jacobian @ step) < np.linalg.norm(residual + jacobian @ radial)
    # Close to the solution this is the unrestricted Newton correction.
    near = residual * 1e-5
    exact = damped_thermal_direction(np.zeros(2), NonlinearEvaluation(near, jacobian, None), jacobian, .04)
    assert not exact.limited
    np.testing.assert_allclose(jacobian @ exact.direction, -near, rtol=1e-14)


def test_bounded_thermal_proposals_allow_trust_growth_and_require_stationarity():
    from wd_spectra._pg1159_structure import damped_thermal_direction
    from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton
    jacobian = np.diag([.01, 1.])
    root = np.array([.5, .3])
    def evaluate(x, want_jacobian):
        return NonlinearEvaluation(jacobian @ (x - root), jacobian if want_jacobian else None, None)
    result = solve_trust_region_newton(np.zeros(2), evaluate,
        step_builder=damped_thermal_direction, merit_function='least-squares',
        trust_update='legacy', broyden_updates=False, jacobian_refresh_interval=1,
        residual_tolerance=1e-10, step_tolerance=1e-8, maximum_iterations=20)
    assert result.converged
    assert max(item.trust_radius for item in result.history) > .04
    np.testing.assert_allclose(result.state, root, atol=1e-10)


@pytest.mark.parametrize("defect,closes", [(.1, False), (1e-6, True), (5e-8, True)])
def test_unclosed_eliminated_material_is_a_recoverable_trial(monkeypatch, defect, closes):
    from dataclasses import dataclass, replace
    from wd_spectra import _pg1159_structure as structure
    from wd_spectra.nonlinear import RecoverableEvaluationError
    @dataclass(frozen=True)
    class Population:
        metadata: dict
        @property
        def converged(self):
            return self.metadata.get('metal_population_converged', False)
    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction: float = 1.
        metal_population_relative_tolerance: float = 1e-4
        metal_population_iterations: int = 1
        metal_population_minimum_iterations: int = 1
        metal_population_damping: float = 1.
        metal_population_acceleration_depth: int = 0
        coupled_population_acceleration_depth: int = 0
        helium_population_damping: float = 1.
        use_population_ali: bool = False
        def rebuild_atmosphere(self, a, t, p):
            return replace(a, temperature=t)
        def solve_populations(self, a, p, **kwargs):
            return p
    atmosphere = gray_helium_atmosphere(110000., 7., n_depth=8)
    equations = structure.PG1159Equations(atmosphere, Model(), np.array([100., 1000.]), radiative_acceleration=False)
    equations.anchor = Population({})
    monkeypatch.setattr(structure, 'population_defect', lambda *args: defect)
    if closes:
        _, population = equations.material(equations.initial_state())
        assert population.converged
    else:
        with pytest.raises(RecoverableEvaluationError, match='material did not close'):
            equations.material(equations.initial_state())


def test_provisional_continuation_does_not_use_final_material_tolerance(monkeypatch):
    from dataclasses import dataclass, replace
    from wd_spectra import _pg1159_structure as structure

    @dataclass(frozen=True)
    class Population:
        metadata: dict

        @property
        def converged(self):
            return self.metadata.get("metal_population_converged", False)

    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction: float = 0.6
        metal_population_relative_tolerance: float = 1e-4
        metal_population_iterations: int = 1
        metal_population_minimum_iterations: int = 1
        metal_population_damping: float = 1.0
        metal_population_acceleration_depth: int = 0
        coupled_population_acceleration_depth: int = 0
        helium_population_damping: float = 1.0
        use_population_ali: bool = False

        def rebuild_atmosphere(self, atmosphere, temperature, population):
            return replace(atmosphere, temperature=temperature)

        def solve_populations(self, atmosphere, population, **kwargs):
            return population

    atmosphere = gray_helium_atmosphere(110000.0, 7.0, n_depth=8)
    equations = structure.PG1159Equations(
        atmosphere,
        Model(),
        np.array([100.0, 1000.0]),
        radiative_acceleration=False,
    )
    equations.anchor = Population({})
    monkeypatch.setattr(structure, "population_defect", lambda *args: 5e-8)

    _, state = equations.material(equations.initial_state())

    assert state.converged
    assert state.metadata["material_population_tolerance"] == 1e-4


def test_full_nlte_relaxation_uses_declared_material_tolerance_and_plain_near_root_map(monkeypatch):
    from dataclasses import dataclass, replace
    from wd_spectra import _pg1159_structure as structure

    population_calls = []

    @dataclass(frozen=True)
    class Population:
        metadata: dict

        @property
        def converged(self):
            return self.metadata.get("metal_population_converged", False)

    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction: float = 1.0
        metal_population_relative_tolerance: float = 1e-4
        metal_population_iterations: int = 1
        metal_population_minimum_iterations: int = 1
        metal_population_damping: float = 1.0
        metal_population_acceleration_depth: int = 0
        coupled_population_acceleration_depth: int = 10
        helium_population_damping: float = 1.0
        use_population_ali: bool = False

        def rebuild_atmosphere(self, atmosphere, temperature, population):
            return replace(atmosphere, temperature=temperature)

        def solve_populations(self, atmosphere, population, **kwargs):
            population_calls.append(kwargs)
            return population

    atmosphere = gray_helium_atmosphere(110000.0, 7.0, n_depth=8)
    equations = structure.PG1159Equations(
        atmosphere,
        Model(),
        np.array([100.0, 1000.0]),
        radiative_acceleration=False,
        certification_stage=False,
    )
    equations.anchor = Population({})
    monkeypatch.setattr(structure, "population_defect", lambda *args: 5e-8)

    _, state = equations.material(equations.initial_state())

    assert state.converged
    assert state.metadata["material_population_tolerance"] == 1e-4
    assert population_calls[0]["_coupled_population_near_root_history"] == 0
    assert population_calls[0]["_coupled_population_near_root_threshold"] == pytest.approx(3e-4)


def test_measured_response_does_not_retry_material_closure_exhaustion():
    from wd_spectra._pg1159_structure import (
        PG1159MaterialClosureError,
        measured_directional_response,
    )
    calls=[]
    def probe(x):
        calls.append(x.copy())
        raise PG1159MaterialClosureError('material did not close')
    with pytest.raises(PG1159MaterialClosureError):
        measured_directional_response(
            probe, np.zeros(2), np.zeros(2), np.array([1., .5])
        )
    assert len(calls) == 1


def test_measured_response_shrinks_for_trial_domain_failure():
    from wd_spectra._pg1159_structure import measured_directional_response
    from wd_spectra.nonlinear import RecoverableEvaluationError
    calls=[]
    def probe(x):
        calls.append(x.copy())
        if len(calls) < 3:
            raise RecoverableEvaluationError('outside domain')
        return 2*x
    derivative=measured_directional_response(
        probe, np.zeros(2), np.zeros(2), np.array([1., .5])
    )
    np.testing.assert_allclose([call[0] for call in calls], [.001,.0005,.00025])
    np.testing.assert_allclose(derivative, [2.,1.])


def test_directional_population_probe_has_a_two_cycle_material_budget(monkeypatch):
    from wd_spectra._pg1159_structure import (
        PG1159Equations,
        PG1159MaterialClosureError,
    )
    atmosphere=gray_helium_atmosphere(90000.,7.,n_depth=8)
    equations=PG1159Equations(
        atmosphere,object(),np.array([100.,1000.]),radiative_acceleration=False
    )
    budgets=[]
    def material(x, *, maximum_closure_iterations=8, maximum_population_iterations=None):
        budgets.append((maximum_closure_iterations,maximum_population_iterations))
        raise PG1159MaterialClosureError('material did not close')
    monkeypatch.setattr(equations,'material',material)
    with pytest.raises(PG1159MaterialClosureError):
        equations.directional_residual(equations.initial_state())
    assert budgets == [(2,20)]


def test_material_closure_accepts_an_already_closed_first_map(monkeypatch):
    from dataclasses import dataclass,replace
    from wd_spectra import _pg1159_structure as structure
    calls=[]
    @dataclass(frozen=True)
    class Population:
        metadata: dict
        @property
        def converged(self):
            return self.metadata.get('metal_population_converged',False)
    @dataclass(frozen=True)
    class Model:
        population_nlte_fraction: float=1.
        metal_population_relative_tolerance: float=1e-4
        metal_population_iterations: int=5
        metal_population_minimum_iterations: int=3
        metal_population_damping: float=1.
        metal_population_acceleration_depth: int=0
        coupled_population_acceleration_depth: int=0
        helium_population_damping: float=1.
        use_population_ali: bool=False
        def rebuild_atmosphere(self,a,t,p):
            return replace(a,temperature=t)
        def solve_populations(self,a,p,**kwargs):
            calls.append(self.metal_population_minimum_iterations)
            return p
    atmosphere=gray_helium_atmosphere(110000.,7.,n_depth=8)
    equations=structure.PG1159Equations(
        atmosphere,Model(),np.array([100.,1000.]),radiative_acceleration=False
    )
    equations.anchor=Population({})
    monkeypatch.setattr(structure,'population_defect',lambda *args:0.)
    _,state=equations.material(equations.initial_state())
    assert state.converged
    assert calls == [1,1]


def test_continuation_handoff_never_certifies_a_limited_or_initial_state():
    from wd_spectra.nonlinear import NonlinearEvaluation, NonlinearProposal, solve_trust_region_newton
    calls = []
    def evaluate(x, want_jacobian):
        calls.append(want_jacobian)
        return NonlinearEvaluation(x - .1, np.eye(2) if want_jacobian else None, None)
    initial = solve_trust_region_newton(np.zeros(2), evaluate,
        accepted_state_handoff=lambda x, e: True)
    assert calls == [False]
    assert not initial.converged
    assert initial.diagnostics.terminal_reason == 'accepted-state-phase-handoff'
    phase = solve_trust_region_newton(np.zeros(2), evaluate,
        step_builder=lambda *args: NonlinearProposal(np.array([.04, .04]), limited=True),
        accepted_state_handoff=lambda x, e: x[0] >= .03)
    assert not phase.converged and phase.history[-1].proposal_limited
    assert phase.diagnostics.terminal_reason == 'accepted-state-phase-handoff'
    # The normal final solver still has to measure an unrestricted correction.
    final = solve_trust_region_newton(phase.state, evaluate, allow_initial_convergence=False)
    assert final.converged
    np.testing.assert_allclose(final.state, [.1, .1], atol=1e-8)


def test_continuation_handoff_never_sees_a_rejected_trial():
    from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton
    seen = []
    def evaluate(x, want_jacobian):
        # Deliberately wrong proposal tangent: every positive trial worsens.
        return NonlinearEvaluation(1 + x, -np.eye(2) if want_jacobian else None, None)
    def handoff(x, e):
        seen.append(x.copy())
        return x[0] > 0
    result = solve_trust_region_newton(np.zeros(2), evaluate, maximum_iterations=1,
        accepted_state_handoff=handoff, finite_difference_fallback_step=None)
    assert not result.converged
    assert result.diagnostics.terminal_reason != 'accepted-state-phase-handoff'
    assert result.diagnostics.rejected_trial_evaluations > 0
    np.testing.assert_array_equal(seen, [[0., 0.]])


def test_gray_opacity_seed_preserves_request_and_mass_grid(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from wd_spectra import _pg1159_reference as reference
    atmosphere = gray_helium_atmosphere(110000., 7., n_depth=8, rosseland_opacity=1.)
    temperature_before = atmosphere.temperature.copy()
    class Model:
        def rebuild_atmosphere(self, a, temperature, population):
            return replace(a, temperature=temperature)
        def transfer_coefficients(self, a, wave, population):
            return SimpleNamespace(total_extinction=np.full((len(wave), a.n_depth), 16.))
    monkeypatch.setattr(reference, 'planck_state', lambda model, a, wave: (a, None))
    result = reference.gray_opacity_seed(Model(), atmosphere)
    gray_temperature = 110000. * (.75 * (16. * atmosphere.column_mass + 2./3.)) ** .25
    expected = np.exp((np.log(temperature_before) + 15 * np.log(gray_temperature)) / 16)
    np.testing.assert_allclose(result.temperature, expected, rtol=1e-14)
    np.testing.assert_array_equal(result.gas_pressure, atmosphere.gas_pressure)
    np.testing.assert_array_equal(result.column_mass, atmosphere.column_mass)
    np.testing.assert_array_equal(atmosphere.temperature, temperature_before)
    assert result.effective_temperature == atmosphere.effective_temperature
    assert result.logg == atmosphere.logg
    assert not result.metadata['pg1159_gray_opacity_initialization']['equilibrium_certified']


def test_coupled_response_retains_previous_directions_and_verifies_new_proposal():
    from wd_spectra._pg1159_response import refine_population_response
    from wd_spectra._pg1159_structure import damped_thermal_direction
    from wd_spectra.nonlinear import NonlinearEvaluation
    rng = np.random.default_rng(72)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    r, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    actual = q @ np.diag(np.geomspace(.03, 3., 8)) @ r.T
    approximate = np.diag(np.geomspace(.03, 3., 8))
    residual = rng.normal(size=8) * .03
    evaluation = NonlinearEvaluation(residual, None, None)
    directions = []
    def propose(matrix):
        return damped_thermal_direction(np.zeros(8), evaluation, matrix, .04).direction
    def measure(direction):
        directions.append(direction.copy())
        return actual @ direction
    corrected, metadata = refine_population_response(approximate, residual, measure, propose)
    assert 1 < metadata['population_response_probe_count'] <= 8
    assert metadata['population_response_direction_validated']
    for direction in directions[:metadata['population_response_measured_rank']]:
        np.testing.assert_allclose(corrected @ direction, actual @ direction, rtol=1e-9, atol=1e-10)
    step = propose(corrected)
    assert np.linalg.norm((corrected-actual) @ step) / np.linalg.norm(actual @ step) < .05
    assert np.linalg.norm(residual + actual @ step) < np.linalg.norm(residual)
    np.testing.assert_array_equal(approximate, np.diag(np.geomspace(.03, 3., 8)))


def test_provisional_response_can_bound_expensive_population_probes():
    from wd_spectra._pg1159_response import refine_population_response

    calls = []
    approximate = np.eye(3)
    actual = np.array([[2.0, 0.4, 0.0], [0.1, 0.5, 0.2], [0.0, 0.3, 1.5]])
    corrected, metadata = refine_population_response(
        approximate,
        np.ones(3),
        lambda direction: calls.append(direction.copy()) or actual @ direction,
        lambda matrix: np.linalg.solve(matrix, -np.ones(3)),
        maximum_probes=1,
    )

    assert len(calls) == 1
    assert metadata["population_response_probe_count"] == 1
    assert metadata["population_response_measured_rank"] == 1
    np.testing.assert_allclose(corrected @ calls[0], actual @ calls[0])


def test_population_response_probe_budget_stays_bounded_at_full_nlte():
    from wd_spectra._pg1159_structure import _population_response_probe_limit

    assert _population_response_probe_limit(0.6) == 1
    assert _population_response_probe_limit(0.8) == 1
    assert _population_response_probe_limit(0.9) == 1
    assert _population_response_probe_limit(1.0, False) == 1
    assert _population_response_probe_limit(1.0) == 1


def test_initializer_handoff_uses_physical_metrics_not_transformed_residual():
    from types import SimpleNamespace
    from wd_spectra._pg1159_structure import _initializer_ready
    from wd_spectra.nonlinear import NonlinearEvaluation

    evaluation = NonlinearEvaluation(
        np.array([0.016]),
        None,
        (
            object(),
            SimpleNamespace(converged=True),
            {
                "surface_flux_ratio": 1.0083,
                "maximum_relative_cell_energy_balance_residual": 0.0053,
                "maximum_hydrostatic_log_pressure_residual": 0.0,
            },
        ),
    )

    assert _initializer_ready(evaluation, 0.01)


def test_full_nlte_handoff_requires_the_flux_profile():
    from types import SimpleNamespace
    from wd_spectra._pg1159_structure import _initializer_ready
    from wd_spectra.nonlinear import NonlinearEvaluation

    diagnostics = {
        "surface_flux_ratio": 1.001,
        "maximum_photospheric_total_flux_residual": 0.2,
        "maximum_all_depth_total_flux_residual": 0.4,
        "maximum_relative_cell_energy_balance_residual": 0.001,
        "maximum_hydrostatic_log_pressure_residual": 0.0,
    }
    evaluation = NonlinearEvaluation(
        np.zeros(2), None, (None, SimpleNamespace(converged=True), diagnostics)
    )
    assert _initializer_ready(evaluation, 0.01)
    assert not _initializer_ready(
        evaluation, 0.01, require_flux_profile=True
    )
    diagnostics["maximum_photospheric_total_flux_residual"] = 0.009
    diagnostics["maximum_all_depth_total_flux_residual"] = 0.009
    assert _initializer_ready(
        evaluation, 0.01, require_flux_profile=True
    )


def test_initializer_handoff_applies_separate_local_energy_tolerance():
    from types import SimpleNamespace
    from wd_spectra._pg1159_structure import _initializer_ready
    from wd_spectra.nonlinear import NonlinearEvaluation

    evaluation = NonlinearEvaluation(
        np.array([0.001]),
        None,
        (
            object(),
            SimpleNamespace(converged=True),
            {
                "surface_flux_ratio": 1.008,
                "maximum_relative_cell_energy_balance_residual": 0.005,
                "maximum_hydrostatic_log_pressure_residual": 0.0,
            },
        ),
    )

    assert not _initializer_ready(evaluation, 0.01, 0.003)


def test_coupled_response_checks_good_tangent_once_without_modifying_it():
    from wd_spectra._pg1159_response import refine_population_response
    matrix = np.array([[2., .3], [-.4, 1.]])
    residual = np.array([.04, -.02])
    result, metadata = refine_population_response(matrix, residual, lambda q: matrix @ q,
        lambda j: np.linalg.solve(j, -residual))
    np.testing.assert_array_equal(result, matrix)
    assert metadata['population_response_probe_count'] == 1
    assert metadata['population_response_measured_rank'] == 0
    assert metadata['population_response_direction_validated']


def test_coupled_response_retains_jacobian_when_physical_probe_fails():
    from wd_spectra._pg1159_response import refine_population_response
    from wd_spectra.nonlinear import RecoverableEvaluationError
    def failed(q):
        raise RecoverableEvaluationError('population trial failed')
    approximate=np.eye(2)
    result,metadata=refine_population_response(
        approximate,np.ones(2),failed,lambda j: -np.ones(2))
    np.testing.assert_array_equal(result,approximate)
    assert not metadata['population_response_direction_validated']
    assert metadata['population_response_probe_failed']
    assert 'population trial failed' in metadata['population_response_probe_failure']


def test_operator_split_continuation_skips_local_material_probes(monkeypatch):
    from types import SimpleNamespace
    from wd_spectra import _pg1159_response as response
    from wd_spectra import _pg1159_structure as structure
    from wd_spectra.nlte_core import NonphysicalPopulationError
    from wd_spectra.nonlinear import NonlinearEvaluation
    atmosphere=gray_helium_atmosphere(90000.,7.,n_depth=8)
    wave=np.geomspace(100.,10000.,30)
    absorption=np.ones((len(wave),atmosphere.n_depth))
    coefficients=NLTETransferCoefficients(
        wave,absorption,absorption,0.2*absorption,{})
    model=SimpleNamespace(
        population_nlte_fraction=.9,
        helium_model=SimpleNamespace(population_n_angle=2),
    )
    equations=structure.PG1159Equations(
        atmosphere,model,wave,radiative_acceleration=False)
    state=equations.initial_state()
    population=object()
    equations.cache[state.tobytes()]=NonlinearEvaluation(
        np.zeros(atmosphere.n_depth),None,(atmosphere,population,{}))
    equations.coefficients[state.tobytes()]=coefficients
    calls=[]
    def unavailable(*args):
        calls.append(args)
        raise NonphysicalPopulationError('local trial failed')
    monkeypatch.setattr(structure,'rate_response_material',unavailable)
    monkeypatch.setattr(response,'thermal_response',
        lambda *args,**kwargs: np.eye(atmosphere.n_depth))
    result=equations.evaluate(state,True)
    np.testing.assert_array_equal(result.jacobian,np.eye(atmosphere.n_depth))
    assert not calls
    assert result.payload[2]['population_response_probe_count']==0
    assert not result.payload[2]['population_response_probe_failed']


def test_nearly_repeated_thermal_direction_does_not_amplify_measurement_noise():
    from wd_spectra._pg1159_response import refine_population_response
    actual = np.array([[2., .3], [.1, 1.5]])
    noise = 1e-5
    probes = []
    def measure(direction):
        probes.append(direction.copy())
        sign = 1 if len(probes) == 1 else -1
        return actual @ direction + sign * noise * np.array([1., -1.])
    def propose(matrix):
        return np.array([1., 0. if not probes else 1e-5])
    corrected, metadata = refine_population_response(
        np.eye(2), np.ones(2), measure, propose)
    # Errors in the measured actions must stay of their original order,
    # rather than being amplified by 1 / the tiny new-direction component.
    assert np.linalg.norm(corrected - actual) < 3 * noise
    assert len(probes) == 2
    assert metadata['population_response_direction_validated']
