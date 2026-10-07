"""Simultaneous hot H/He equations for the shared nonlinear driver.

Temperature and independent population ratios are simultaneous unknowns.
This module is a research candidate until cold and observed checks qualify it.
"""
from dataclasses import replace
import logging
import time
import numpy as np
from .hot_nlte import (population_arrays, with_departures, population_status,
    transfer_field, HotAtmosphereResult)
from .nlte_core import _FixedTransferCache, NLTETransferCoefficients, NonphysicalPopulationError
from .nlte import _profile_averaged_mean_intensity_nu
from . import helium_nlte as he, multilevel_nlte as hydrogen
from .nonlinear import NonlinearEvaluation, NonlinearProposal, RecoverableEvaluationError, solve_trust_region_newton, nonlinear_result_metadata
from ._mass_feautrier import (mass_emissivity_energy, mass_emissivity_field,
    mass_width, InvalidRadiationFieldError)
from ._hot_response import HotResponseOperator as MassResponseOperator
from .opacity import optical_depth_from_mass_opacity
from ._compat import trapezoid
from .constants import STEFAN_BOLTZMANN
from ._convergence import equilibrium_certificate
from .adaptive_structure import _thermal_boundary_absorption_escape_bound
from .spectrum import planck_lambda_angstrom
from ._hot_rates import (PreparedHeliumRates, PreparedLineAverages,
    HeliumRadiationResponse, MixedRadiationResponse)

LOGGER = logging.getLogger(__name__)


def _least_squares(matrix, rhs, *, rcond):
    """Retry a finite system with classical SVD if divide-and-conquer fails.

    Both drivers solve the same system with the same rank cutoff. Invalid
    derivatives are errors, never inputs to a numerical recovery path.
    """
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(rhs)):
        raise ValueError('non-finite coupled Jacobian or residual')
    try:
        solution,_,rank,singular=np.linalg.lstsq(matrix,rhs,rcond=rcond)
        return solution,rank,singular,'numpy-lstsq'
    except np.linalg.LinAlgError:
        from scipy.linalg import lstsq
        cutoff=np.finfo(float).eps*max(matrix.shape) if rcond is None else rcond
        LOGGER.warning('Finite Newton least-squares SVD failed; retrying identical system with LAPACK gelss')
        solution,_,rank,singular=lstsq(matrix,rhs,cond=cutoff,lapack_driver='gelss')
        return solution,rank,singular,'scipy-gelss'


class HotEquations:
    population_scale = .1

    def __init__(self, seed, model, structure_wave, *, fixed_temperature=False, nlte_fraction=1.,
                 reuse_accepted_state=True):
        self.seed, self.model = seed, model
        structure_wave=np.asarray(structure_wave,dtype=float)
        if (structure_wave.ndim != 1 or len(structure_wave)<2 or
                np.any(~np.isfinite(structure_wave)) or np.any(structure_wave<=0) or
                np.any(np.diff(structure_wave)<=0)):
            raise ValueError('structure wavelengths must be finite, positive and increasing')
        if not 0 <= nlte_fraction <= 1:
            raise ValueError('NLTE continuation fraction must be between zero and one')
        self.nlte_fraction = nlte_fraction
        self.fixed_temperature = fixed_temperature
        # The zero-radiation-continuation stage fixes populations exactly to
        # their local Planck solution. Eliminate those algebraic unknowns;
        # differentiating their actual T dependence gives the same equations
        # without building hundreds of identically constrained columns.
        self.planck_temperature_only = nlte_fraction == 0 and not fixed_temperature
        self.nd = seed.n_depth
        self.nhe = 15+model.maximum_helium_ii_level
        self.nh = 0 if model.log_hydrogen_to_helium is None else model.maximum_hydrogen_level+1
        self.indices = np.r_[np.arange(self.nhe-1), np.arange(self.nhe, self.nhe+self.nh-1)] if self.nh else np.arange(self.nhe-1)
        self.continua = np.r_[np.full(self.nhe-1, self.nhe-1), np.full(max(0,self.nh-1), self.nhe+self.nh-1)]
        groups = model._line_problems(seed)
        grids = [structure_wave, he.default_neutral_helium_continuum_wavelength(),
                 he.default_helium_ii_continuum_wavelength(model.maximum_helium_ii_level)]
        if self.nh:
            grids.append(hydrogen._default_continuum_wavelength(model.maximum_hydrogen_level))
        grids.extend(p.continuum.wavelength_angstrom for group in groups for problems in group.values() for p in problems)
        self.wave = np.unique(np.concatenate(grids))
        self.target = STEFAN_BOLTZMANN*seed.effective_temperature**4
        self.prepared = {}
        self.evaluations = 0
        self.last_key = None
        self.last_evaluation = None
        self.jacobian_key = None
        self.jacobian_evaluation = None
        self.rate_cache = None
        self.profile_cache = None
        self.reuse_accepted_state = reuse_accepted_state
        self.last_physics = None
        self.linear_solver_drivers = set()

    def prepare(self, log_t):
        key = log_t.tobytes()
        if key not in self.prepared:
            a = self.model.rebuild_atmosphere(self.seed, np.exp(log_t))
            ref = self.model._rate_state(a)
            groups = self.model._line_problems(a)
            self.prepared[key] = (a, ref, groups, _FixedTransferCache(a, self.wave))
            if len(self.prepared)>3:
                del self.prepared[next(iter(self.prepared))]
        return self.prepared[key]

    def coordinates(self, state):
        actual, reference = population_arrays(state)
        log_population = np.log(actual)
        return self.population_scale*(log_population[:,self.indices]-log_population[:,self.continua]).ravel()

    def initial_state(self, populations=None):
        t = np.log(self.seed.temperature)
        if self.planck_temperature_only:
            return t
        ref = self.prepare(t)[1]
        state = ref if populations is None else self.model.remap(self.prepare(t)[0], populations)
        return self.coordinates(state) if self.fixed_temperature else np.r_[t,self.coordinates(state)]

    def populations(self, x, reference):
        if self.planck_temperature_only:
            return reference
        nt = 0 if self.fixed_temperature else self.nd
        log_b = np.zeros((self.nd,self.nhe+self.nh))
        _, lte = population_arrays(reference)
        log_b[:,self.indices] = (x[nt:].reshape(self.nd,-1)/self.population_scale
            - np.log(lte[:,self.indices]/lte[:,self.continua]))
        return with_departures(reference, np.exp(log_b))

    def rate_residual(self,x,a,ref,groups,mean,*,prepare_cache=False):
        nt = 0 if self.fixed_temperature else self.nd
        if prepare_cache and (self.profile_cache is None or self.profile_cache.groups is not groups):
            self.profile_cache=PreparedLineAverages(self.wave,groups,self.nd)
        fields = []
        if self.profile_cache is not None and self.profile_cache.groups is groups:
            fields=self.profile_cache.fields(mean)
        else:
            for group in groups:
                fields.append({key: np.average([
                    _profile_averaged_mean_intensity_nu(p.continuum.wavelength_angstrom,p.lte_line_opacity,
                        mean[np.searchsorted(self.wave,p.continuum.wavelength_angstrom)])
                    for p in problems],axis=0,weights=[p.line.absorption_oscillator_strength for p in problems])
                    for key,problems in group.items()})
        if prepare_cache and (self.rate_cache is None or self.rate_cache.atmosphere is not a):
            self.rate_cache=PreparedHeliumRates(self.model,a,self.wave,groups)
        candidate = (self.rate_cache.state(mean,*fields)
            if self.rate_cache is not None and self.rate_cache.atmosphere is a else
            self.model._rate_state(a,self.wave,mean,*fields))
        target_coordinates = (self.nlte_fraction*self.coordinates(candidate)
            + (1-self.nlte_fraction)*self.coordinates(ref))
        pop_residual = (np.empty(0) if self.planck_temperature_only else
                        (target_coordinates-x[nt:])/self.population_scale)
        return candidate,pop_residual

    def residual(self, x, coefficients=None, prepared=None, response=None):
        # Cover both temperature/reference preparation and prepared/uncached
        # SE evaluation. The central driver backtracks only proposed trials;
        # an invalid initial/accepted state remains a fatal error.
        try:
            return self._residual(x, coefficients, prepared, response)
        except NonphysicalPopulationError as exc:
            raise RecoverableEvaluationError(str(exc)) from exc

    def _residual(self, x, coefficients=None, prepared=None, response=None):
        key = x.tobytes()
        canonical = coefficients is None and prepared is None and response is None
        if canonical and key == self.last_key:
            return self.last_evaluation
        self.evaluations += 1
        nt = 0 if self.fixed_temperature else self.nd
        t = np.log(self.seed.temperature) if self.fixed_temperature else x[:nt]
        a, ref, groups, cache = self.prepare(t) if prepared is None else prepared
        current = self.populations(x, ref)
        c = self.model.transfer_coefficients(a,self.wave,current,_cache=cache) if coefficients is None else coefficients
        if (np.any(c.true_absorption+c.scattering <= 0) or np.any(c.true_absorption[:,-1] <= 0)
                or np.any(c.thermal_emissivity < 0)
                or np.any(c.scattering < 0)):
            raise RecoverableEvaluationError('Trial leaves positive total extinction/bottom absorption/emission domain')
        if response is None:
            try:
                source, field, closure = transfer_field(a,c,n_angle=self.model.n_angle,check_source=False)
            except InvalidRadiationFieldError as exc:
                raise RecoverableEvaluationError(str(exc)) from exc
            mean = field.mean_intensity
            flux = trapezoid(field.interface_flux,self.wave,axis=0)/self.target-1
        else:
            mean, flux = response
            closure = None
        candidate,pop_residual=self.rate_residual(x,a,ref,groups,mean,prepare_cache=response is None)
        old, reference = population_arrays(current)
        new, _ = population_arrays(candidate)
        change = float(np.max(abs(new-old)/np.maximum(new,1e-12*reference.sum(axis=1)[:,None])))
        energy, emission = mass_emissivity_energy(self.wave,a.column_mass,c.thermal_emissivity,
                                      mean,c.true_absorption)
        scaled_energy = energy/np.maximum(abs(emission),1e-30*self.target)
        # Preserve sensitivity to flux differences in optically thick cells.
        # The unscaled energy equation is unchanged; only its numerical weight
        # differs from the separately reported relative-emission diagnostic.
        # Deep-cell energy is the same discrete flux difference. Use that
        # form where subtraction of large emissivity/absorption terms loses
        # precision; thin cells retain the well-scaled local heating equation.
        thermal_residual = np.where(abs(emission)>=self.target,np.diff(flux),scaled_energy)
        residual = pop_residual if self.fixed_temperature else np.r_[thermal_residual,flux[-1],pop_residual]
        b = planck_lambda_angstrom(self.wave[:,None],a.temperature[None,:])
        if np.any(c.true_absorption<0):
            # An absorption-only escape bound is not conservative in a gain
            # layer: scattering can increase its amplification path. Measure
            # the actual discrete boundary contribution with the same coupled
            # operator, zero volume emission, and the imposed bottom source.
            ext=c.true_absorption+c.scattering
            try:
                _,boundary=mass_emissivity_field(
                    optical_depth_from_mass_opacity(a.column_mass,ext),
                    np.zeros_like(c.thermal_emissivity),c.true_absorption,c.scattering,
                    bottom_source=c.thermal_emissivity[:,-1]/c.true_absorption[:,-1],
                    column_mass=a.column_mass,n_angle=self.model.n_angle,wavelength_chunk_size=512)
            except InvalidRadiationFieldError as exc:
                raise RecoverableEvaluationError(str(exc)) from exc
            boundary_bound=float(abs(trapezoid(boundary.interface_flux[:,0],self.wave))/self.target)
            boundary_method='discrete coupled boundary-source response with stimulated gain'
        else:
            boundary_bound=_thermal_boundary_absorption_escape_bound(
                self.wave,a.column_mass,c.true_absorption,b[:,-1],self.target)
            boundary_method='conservative absorption escape bound'
        diagnostics = dict(solver_phase='simultaneous-hot-nlte',
            nlte_continuation_fraction=self.nlte_fraction,
            maximum_continuation_population_residual=float(np.max(abs(pop_residual),initial=0.)),
            maximum_all_depth_total_flux_residual=float(np.max(abs(flux))),
            maximum_relative_cell_energy_balance_residual=float(np.max(abs(scaled_energy))),
            electron_scattering_source_final_maximum_relative_residual=closure,
            lower_boundary_absorption_escape_bound=boundary_bound,
            lower_boundary_screening_method=boundary_method,
            minimum_net_absorption=float(np.min(c.true_absorption)),
            nlte_populations_converged=change < self.model.population_tolerance,
            nlte_maximum_relative_population_change=change,surface_flux_ratio=float(flux[0]+1))
        current = population_status(current, diagnostics['nlte_populations_converged'], 1,change)
        evaluation = NonlinearEvaluation(residual,None,(a,current,diagnostics))
        # Commit the bundle only after all physical checks succeed. Failed
        # trials and externally supplied probe coefficients/radiation may
        # replace atomic caches without replacing the canonical anchor.
        if canonical:
            self.last_key, self.last_evaluation = key, evaluation
            self.last_physics = ((key, (a,ref,groups,cache), c, source, field, candidate,
                                  self.rate_cache.last_rate_matrix)
                                 if self.reuse_accepted_state else None)
        return evaluation

    def evaluate(self,x,jacobian):
        if jacobian and x.tobytes() == self.jacobian_key:
            return self.jacobian_evaluation
        base = self.residual(x)
        if not jacobian:
            return base
        start = time.monotonic()
        step = 1e-5
        nt = 0 if self.fixed_temperature else self.nd
        t = np.log(self.seed.temperature) if self.fixed_temperature else x[:nt]
        anchor = (self.last_physics if self.reuse_accepted_state and self.last_physics is not None
                  and self.last_physics[0] == x.tobytes() else None)
        a, reference, groups, cache = self.prepare(t) if anchor is None else anchor[1]
        self.prepared={t.tobytes():(a,reference,groups,cache)}
        if anchor is None:
            base_c = self.model.transfer_coefficients(a,self.wave,base.payload[1],_cache=cache)
            source, field, _ = transfer_field(a,base_c,n_angle=self.model.n_angle,check_source=False)
        else:
            _, _, base_c, source, field, candidate, helium_rate = anchor
        flux = trapezoid(field.interface_flux,self.wave,axis=0)/self.target-1
        ext = base_c.true_absorption+base_c.scattering
        tau = optical_depth_from_mass_opacity(a.column_mass,ext)
        j = np.empty((len(x),len(x)))
        energy,emission=mass_emissivity_energy(self.wave,a.column_mass,base_c.thermal_emissivity,
            field.mean_intensity,base_c.true_absorption)
        emission_scale=np.maximum(abs(emission),1e-30*self.target)
        widths=4*np.pi*mass_width(a.column_mass)
        spacing=np.diff(self.wave)
        quadrature=.5*(np.r_[0.,spacing]+np.r_[spacing,0.])
        # A rejected trial can prepare rates before failing a later physical
        # check. A cached base residual therefore does not establish that its
        # rate/line caches still belong to this Jacobian's atmosphere.
        if (self.rate_cache is None or self.rate_cache.atmosphere is not a or
                self.profile_cache is None or self.profile_cache.groups is not groups):
            self.rate_residual(x,a,reference,groups,field.mean_intensity,prepare_cache=True)
        response_type=MixedRadiationResponse if self.nh else HeliumRadiationResponse
        rate_response=(response_type(self.rate_cache,self.profile_cache,field.mean_intensity,
                       **({"state":candidate, "rate_matrix":helium_rate} if anchor is not None else {}))
                       if not self.planck_temperature_only else None)
        response_operator=MassResponseOperator(tau,self.wave,source,base_c.scattering/ext,
            a.column_mass,extinction=ext,n_angle=self.model.n_angle,
            wavelength_chunk_size=256,allow_stimulated_gain=True)

        def responses(plus,minus):
            da = (plus.true_absorption-minus.true_absorption)/(2*step)
            ds = (plus.scattering-minus.scattering)/(2*step)
            deta = (plus.thermal_emissivity-minus.thermal_emissivity)/(2*step)
            direct = (deta+ds*field.mean_intensity-(da+ds)*source)/ext
            db = np.zeros_like(deta)
            db[:,-1] = (plus.thermal_emissivity[:,-1]/plus.true_absorption[:,-1]-
                        minus.thermal_emissivity[:,-1]/minus.true_absorption[:,-1])/(2*step)
            dj=np.empty((len(self.wave),self.nd,self.nd))
            energy_j=np.zeros((self.nd-1,self.nd))
            def consume_mean(start,stop,response):
                dj[start:stop]=response
                energy_j[:]+=widths[:,None]*np.einsum('wdk,w->dk',
                    base_c.true_absorption[start:stop,:-1,None]*response[:,:-1,:],quadrature[start:stop])
            df,_,_ = response_operator.apply(direct,db,da+ds,
                return_auxiliary_response=False,mean_response_consumer=consume_mean)
            df/=self.target
            cells=np.arange(self.nd-1)
            energy_j[cells,cells]+=widths*trapezoid(
                da[:,:-1]*field.mean_intensity[:,:-1]-deta[:,:-1],self.wave,axis=0)
            thermal_j=energy_j/emission_scale[:,None]
            emission_j=widths*trapezoid(deta[:,:-1],self.wave,axis=0)
            thermal_j[cells,cells]-=np.where(abs(emission)>1e-30*self.target,
                energy*emission_j/emission_scale**2,0.)
            thermal_j=np.where((abs(emission)>=self.target)[:,None],np.diff(df,axis=0),thermal_j)
            return np.vstack((thermal_j,df[-1])),dj

        if nt:
            probes=[]
            material_rates=[]
            for sign in (1,-1):
                pa,pr,pg,pc=self.prepare(t+sign*step)
                probes.append(self.model.transfer_coefficients(pa,self.wave,self.populations(x,pr),_cache=pc))
                if not self.planck_temperature_only:
                    trial=x.copy();trial[:nt]+=sign*step
                    material_rates.append(self.rate_residual(trial,pa,pr,pg,field.mean_intensity)[1])
                self.prepared.pop((t+sign*step).tobytes(),None)
                del pa,pr,pg,pc
            thermal_j,dj=responses(*probes)
            j[:nt,:nt]=thermal_j
            if not self.planck_temperature_only:
                # Fixed-J material rates are local in depth. Two simultaneous
                # material probes therefore supply every diagonal block;
                # the exact SE radiation response supplies all nonlocal terms.
                pop_j=self.nlte_fraction*rate_response.log_ratio_response(dj).reshape(-1,self.nd)
                local=(material_rates[0]-material_rates[1])/(2*step)
                rows=np.arange(len(local))
                pop_j[rows,rows//len(self.indices)]+=local
                j[nt:,:nt]=pop_j
            # Each temperature probe owns a full grid of continuum and line
            # profiles. Its derivatives are now incorporated; retaining both
            # probe caches throughout the population columns wastes gigabytes
            # on production grids. Keep only this Jacobian's material anchor.
            self.prepared={t.tobytes():(a,reference,groups,cache)}
            del probes,material_rates,dj
        ni=len(self.indices)
        for level in range(0 if self.planck_temperature_only else ni):
            probes=[]
            for sign in (1,-1):
                trial=x.copy();trial[nt+level::ni]+=sign*step
                probes.append(self.model.transfer_coefficients(a,self.wave,self.populations(trial,reference),_cache=cache))
            thermal_j,dj=responses(*probes)
            pop_j=self.nlte_fraction*rate_response.log_ratio_response(dj).reshape(-1,self.nd)
            # The next column needs a fresh large radiation-response array.
            # Release this one before entering that solve, not on its return.
            del dj
            for depth in range(self.nd):
                pop_j[depth*ni+level,depth]-=1/self.population_scale
            columns=nt+np.arange(self.nd)*ni+level
            if nt:
                j[:nt,columns]=thermal_j
            j[nt:,columns]=pop_j
            if (level+1)%5==0:
                LOGGER.info('Coupled level derivative %d/%d, %.1fs',level+1,ni,time.monotonic()-start)
        # Rank/correction are measured by the actual Newton solve below.
        # Avoid an extra diagnostic-only SVD before a caller can save J.
        # The shared driver validates finiteness after the diagnostic callback.
        LOGGER.info('Coupled Jacobian assembled in %.2fs',time.monotonic()-start)
        self.last_key,self.last_evaluation=x.tobytes(),base
        evaluated=NonlinearEvaluation(base.residual,j,base.payload)
        self.jacobian_key,self.jacobian_evaluation=x.tobytes(),evaluated
        return evaluated


def solve(seed,model,wave,*,maximum_iterations=60,iteration_callback=None,populations=None,
          fixed_temperature=False,initial_jacobian=None,jacobian_callback=None,nlte_fraction=1.,
          flux_tolerance=3e-3,temperature_tolerance=3e-4,initial_trust_radius=.04):
    for tolerance in (flux_tolerance,temperature_tolerance):
        if not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('convergence tolerances must be finite and positive')
    equations = HotEquations(seed,model,wave,fixed_temperature=fixed_temperature,nlte_fraction=nlte_fraction)
    x = equations.initial_state(populations)
    proposed_temperature_correction = None
    def qualified(x,e,correction):
        d=e.payload[2]
        populations_ok = (d['nlte_populations_converged'] if nlte_fraction == 1 else
                          d['maximum_continuation_population_residual'] < model.population_tolerance)
        return (populations_ok and (fixed_temperature or
            (d['maximum_all_depth_total_flux_residual']<flux_tolerance and
             d['maximum_relative_cell_energy_balance_residual']<flux_tolerance)) and correction<temperature_tolerance)
    def callback(it,x,e):
        e.payload[2].update(newton_line_search_factor=it.line_search_factor,
            newton_unrestricted_correction=it.unrestricted_maximum_step,
            newton_unrestricted_temperature_correction=(proposed_temperature_correction
                if it.step_kind == 'newton' and not it.proposal_limited else None),
            nonlinear_step_kind=it.step_kind,nonlinear_trust_radius=it.trust_radius)
        LOGGER.info('Joint Newton iteration %d: %s',it.iteration,e.payload[2])
        if iteration_callback:
            iteration_callback(it.iteration,e.payload[0],e.payload[1],e.payload[2])
    def direction(x,e,j,radius):
        nonlocal proposed_temperature_correction
        rows=np.maximum(np.max(abs(j),axis=1),np.finfo(float).tiny)
        delta,rank,_,driver=_least_squares(j/rows[:,None],-e.residual/rows,rcond=1e-10)
        equations.linear_solver_drivers.add(driver)
        proposed_temperature_correction=(float(np.max(abs(delta[:seed.n_depth])))
            if not fixed_temperature and rank==len(x) else None)
        if rank==len(x):
            # Let the common driver scale the genuine Newton direction to
            # its trust bound. Damping the raw population-dominated system
            # can instead select strongly curved weak thermal directions.
            return NonlinearProposal(delta)
        # A damped least-squares direction avoids arbitrary box-corner
        # excursions in weak temperature/population combinations. The final
        # stationarity certificate still uses the unrestricted full Jacobian.
        from scipy.linalg import svd
        u,singular,vt=svd(j,full_matrices=False,lapack_driver='gesvd')
        rhs=u.T@(-e.residual)
        def damped(damping):
            return vt.T@(singular/(singular*singular+damping)*rhs)
        lo=0.;hi=max(float(singular[0]**2)*1e-12,1e-16)
        while np.max(abs(damped(hi)))>radius:
            hi*=10
        for _ in range(40):
            mid=.5*(lo+hi)
            if np.max(abs(damped(mid)))>radius:lo=mid
            else:hi=mid
        return NonlinearProposal(damped(hi),limited=True)
    supplied = initial_jacobian
    def evaluate(state,jacobian):
        nonlocal supplied
        if jacobian and supplied is not None:
            np.testing.assert_allclose(state,supplied['x'],rtol=0,atol=1e-12)
            actual=equations.residual(state)
            np.testing.assert_allclose(actual.residual,supplied['residual'],rtol=1e-6,atol=1e-9)
            result=NonlinearEvaluation(actual.residual,supplied['jacobian'],actual.payload)
            supplied=None
            return result
        evaluated=equations.evaluate(state,jacobian)
        if jacobian and jacobian_callback is not None:
            jacobian_callback(state,evaluated)
        return evaluated
    recovery = None
    recovery_hooks = {}
    proposal_builder = direction
    if nlte_fraction == 1. and not fixed_temperature:
        from ._hot_recovery import CoupledBracketedThermalPopulationCorrection
        from ._hot_recovery_fresh import FreshTangentFallback
        proposals = CoupledBracketedThermalPopulationCorrection(
            evaluate, direction, temperature_tolerance, equations)
        recovery = FreshTangentFallback(proposals, equations)
        proposal_builder = recovery.direction
        recovery_hooks = dict(iteration_correction=recovery.correction,
            trial_projector=recovery.project, rejected_step_handoff=recovery.rejected)
    result = solve_trust_region_newton(x,evaluate,maximum_iterations=maximum_iterations,
        residual_tolerance=model.population_tolerance,step_tolerance=temperature_tolerance,convergence_test=qualified,
        allow_initial_convergence=False,callback=callback,finite_difference_fallback_step=None,
        jacobian_refresh_interval=3,initial_trust_radius=initial_trust_radius,maximum_trust_radius=.12,
        broyden_updates=False,
        linear_regularization=0.,step_builder=proposal_builder,merit_function="least-squares",
        **recovery_hooks)
    # A continuation stage is an initialization problem, never a certified
    # atmosphere. Its accepted state and driver termination are sufficient
    # to start the next stage; an extra independent full Jacobian here costs
    # minutes without qualifying anything. Retain that independent check at
    # full NLTE, including unconverged/fixed-temperature diagnostics.
    final = equations.evaluate(result.state,nlte_fraction == 1.)
    a,p,d=final.payload
    _,_,closure = transfer_field(a,model.transfer_coefficients(a,equations.wave,p),n_angle=model.n_angle)
    d = {**d, "electron_scattering_source_final_maximum_relative_residual":closure}
    rank=None
    measured=False
    value=None
    if nlte_fraction == 1.:
        scales=np.maximum(np.max(abs(final.jacobian),axis=1),np.finfo(float).tiny)
        correction,rank,_,driver=_least_squares(final.jacobian/scales[:,None],-final.residual/scales,rcond=None)
        equations.linear_solver_drivers.add(driver)
        measured=rank==len(x) and np.all(np.isfinite(correction))
        value=float(np.max(abs(correction if fixed_temperature else correction[:seed.n_depth]))) if measured else None
    # Initialization diagnostics must not masquerade as current NLTE fields.
    # Keep them in one provenance record rather than inheriting LTE residual
    # arrays, iteration counts, and solver descriptions at the top level.
    physical={key:a.metadata[key] for key in (
        'composition','eos','bulk_helium_eos','log_hydrogen_to_helium',
        'nlte_charge_feedback','nlte_electron_closure') if key in a.metadata}
    metadata={**physical,**d,
        'initial_atmosphere_metadata':seed.metadata.get('initial_atmosphere_metadata',dict(seed.metadata)),
        'radiative_equilibrium_solver':'shared-complete-linearization-hot-nlte',
        'radiative_equilibrium_solver_converged':bool(result.converged and p.converged and not fixed_temperature and nlte_fraction == 1),
        'temperature_correction_measured':bool(measured and not fixed_temperature),
        'maximum_unrestricted_log_temperature_correction':value,
        'nonlinear_solver':nonlinear_result_metadata(result),'independent_grid_validation':False,
        'full_physics_validation':False,'coupled_jacobian_rank':int(rank) if rank is not None else None,
        'linear_solver_drivers':sorted(equations.linear_solver_drivers),
        'coupled_unknown_count':len(x),'structure_wavelength_count':len(equations.wave)}
    metadata['hot_nlte_recovery'] = dict(
        eligible=recovery is not None,
        activated=recovery.active if recovery is not None else False,
        activation_rule='two rejected Newton directions at full NLTE',
        rejected_directions_observed=recovery.rejected_directions if recovery is not None else 0,
        extra_jacobian_evaluations=recovery.extra_jacobian_evaluations if recovery is not None else 0,
        extra_residual_evaluations=proposals.extra_residual_evaluations if recovery is not None else 0,
        thermal_probes=proposals.thermal_probes if recovery is not None else 0,
        neighbor_adjustments=proposals.neighbor_adjustments if recovery is not None else 0)
    metadata['equilibrium_certificate']=equilibrium_certificate(metadata,flux_tolerance=flux_tolerance,temperature_tolerance=temperature_tolerance)
    metadata['radiative_equilibrium_converged']=metadata['equilibrium_certificate']['verified']
    return HotAtmosphereResult(replace(a,metadata=metadata),p,result)
