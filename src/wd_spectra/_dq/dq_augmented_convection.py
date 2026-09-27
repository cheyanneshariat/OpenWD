"""DQ-only coupled temperature/ML2 coordinates in the shared nonlinear engine.

Solve energy AND gradient/material compatibility simultaneously. This removes
the nonlinear material inversion inside each trial. The auxiliary flux is an
unknown, never Fstar-Frad, and is never used to certify an atmosphere: the
original temperature-gradient ML2 flux and original gates must also pass.

This uses the existing ML2 algebra, not a new convection closure. The enlarged
formulation is analogous to the explicit-gradient treatment in TLUSTY's
reference manual, Appendix C1 (https://arxiv.org/abs/1706.01935).
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from unittest.mock import patch
import numpy as np
from scipy.optimize import lsq_linear

import wd_spectra.adaptive_structure as adaptive
from . import base as dq
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import NonlinearEvaluation,NonlinearProposal,RecoverableEvaluationError
from wd_spectra._ml2_auxiliary import (ml2_auxiliary_from_gradient,
    ml2_auxiliary_compatibility,ml2_scaled_coefficients)
from .measured_proposal_guard import require_measured_proposal
from .dq_refractive_material import RefractiveDQMaterial
from .dq_refractive_material_coordinates import RefractiveMaterialCoefficients, interface_index
from .dq_reos_secant_thermodynamics import StableREOSSecantMixin
from .dq_internal_domain_proposal import initialize_physical_extension


class CoupledMaterialSystem:
    def __init__(self,first,evaluate,material):
        p=first.payload;self.n=len(p['atmosphere'].temperature)
        self.evaluate_physical=evaluate;self.material=material
        self.target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
        self.mapping=p['log_temperature_from_state']
        self.gradient_operator=p['interface_gradient_operator']
        self.pressure=p['atmosphere'].gas_pressure.copy()
        x=np.log(p['atmosphere'].temperature)
        values=tuple(p['convection_transport'][k][1:] for k in
            ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        y=ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],*values,self.target)
        self.velocity_scale=np.maximum(1.,abs(y))
        a,b=ml2_scaled_coefficients(values[1],values[2],self.target)
        self.compatibility_scale=a*self.velocity_scale+b*self.velocity_scale**2
        self.initial=np.r_[x,y/self.velocity_scale]
        self.cached=(x.copy(),first)

    def physical(self,x,need):
        point,ev=self.cached
        if not np.array_equal(point,x) or (need and ev.jacobian is None):
            ev=self.evaluate_physical(np.linalg.solve(self.mapping,x),need)
            self.cached=(x.copy(),ev)
        current_mapping=ev.payload.get('log_temperature_from_state')
        if current_mapping is not None and not np.array_equal(current_mapping,self.mapping):
            raise RuntimeError('Temperature coordinates changed inside the coupled DQ phase')
        if not np.array_equal(ev.payload['atmosphere'].gas_pressure,self.pressure):
            raise RuntimeError('Pressure grid changed inside the coupled DQ phase')
        return ev

    def state_from_temperature(self, x, physical):
        """Recompute exact auxiliary ML2 coordinates at the supplied T."""
        p = physical.payload
        values = tuple(p['convection_transport'][key][1:] for key in
            ('adiabatic_gradient', 'ml2_radiative_loss', 'ml2_flux_coefficient'))
        velocity = ml2_auxiliary_from_gradient(p['temperature_gradient'][1:],
                                             *values, self.target)
        return np.r_[x, velocity/self.velocity_scale]

    def evaluate(self,state,need):
        n=self.n;x=state[:n];y=state[n:]*self.velocity_scale
        physical=self.physical(x,need);p=physical.payload
        if not (p.get('energy_balance_is_physical_flux') and p.get('local_energy_equations_enforced')):
            raise ValueError('Coupled DQ requires physical local-energy equations')
        values=tuple(p['convection_transport'][k][1:] for k in
            ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        responses=None
        if need:
            actual,tangent=self.material(x,True)
            for value,expected in zip(actual,values):
                np.testing.assert_allclose(value[1:],expected,rtol=2e-13,atol=0)
            responses=tuple(v[1:] for v in tangent)
        compatibility,ct,cy=ml2_auxiliary_compatibility(p['temperature_gradient'][1:],
            y,values,responses,self.gradient_operator[1:],self.target)
        with np.errstate(over='ignore',invalid='ignore'):
            auxiliary=np.r_[0.,np.maximum(y,0.)**3]*self.target
            scale=p['thermal_cell_emission']+auxiliary[:-1]+auxiliary[1:]
            defect=p['radiative_cell_energy_defect']+np.diff(auxiliary)
            energy=(p['radiative_flux_interface']+auxiliary)/self.target-1.
            energy=energy.copy();energy[:-1]-=defect/scale
        residual=np.r_[energy,compatibility/self.compatibility_scale]
        if np.any(~np.isfinite(residual)) or np.any(scale<=0):
            raise RecoverableEvaluationError('Nonfinite coupled DQ trial')
        jacobian=None
        if need:
            fcj=np.zeros((n,n-1))
            fcj[1:]=np.diag(3*np.maximum(y,0.)**2*self.velocity_scale*self.target)
            et=p['radiative_flux_log_temperature_jacobian']/self.target
            et=et.copy()
            et[:-1]-=p['radiative_cell_energy_log_temperature_jacobian']/scale[:,None]
            et[:-1]+=(defect/scale**2)[:,None]*p['thermal_cell_emission_log_temperature_jacobian']
            ey=fcj/self.target
            ey=ey.copy();ey[:-1]-=np.diff(fcj,axis=0)/scale[:,None]
            ey[:-1]+=(defect/scale**2)[:,None]*(fcj[:-1]+fcj[1:])
            jacobian=np.block([[et,ey],[ct/self.compatibility_scale[:,None],
                np.diag(cy*self.velocity_scale/self.compatibility_scale)]])
        # Physical payload/fluxes remain untouched, even at incompatible trials.
        payload={**p,'dq_augmented_auxiliary_flux':auxiliary,
            'dq_augmented_scaled_compatibility':compatibility/self.compatibility_scale,
            'dq_augmented_flux_compatibility':float(np.max(abs(auxiliary-p['convective_flux_interface']))/self.target)}
        return NonlinearEvaluation(residual,jacobian,payload)


def coupled_direction(jacobian,residual,n,radius):
    """Solve directly, or minimize the same linear least-squares objective.

    Both log-temperatures and nondimensional auxiliary variables are box
    constrained. A temperature-only box permits arbitrarily large element
    velocities in nearly adiabatic stable layers, where the quadratic
    compatibility relation invalidates that nominally small linear step.
    No normal equations, additional opacity calls or material repairs.
    """
    rows=np.max(abs(jacobian),axis=1)
    if np.any(rows==0):raise RecoverableEvaluationError('Unconstrained coupled DQ equation')
    a=jacobian/rows[:,None];columns=np.max(abs(a),axis=0)
    if np.any(columns==0):raise RecoverableEvaluationError('Unconstrained coupled DQ variable')
    direction=np.linalg.solve(a/columns[None,:],-residual/rows)/columns
    if np.max(abs(direction))<=radius:return direction,True
    lower=np.full(2*n-1,-radius)
    upper=-lower
    bounded=lsq_linear(jacobian,-residual,bounds=(lower,upper),method='bvls',
        tol=1e-12,max_iter=20*len(residual))
    if not bounded.success or np.any(~np.isfinite(bounded.x)):
        raise RecoverableEvaluationError('Coupled DQ bounded subproblem did not solve')
    if np.linalg.norm(residual+jacobian@bounded.x)>np.linalg.norm(residual):
        raise RecoverableEvaluationError('Coupled DQ subproblem did not reduce its model')
    return bounded.x,False


def project_deep_monotone_temperature(old, proposed, n, first_index):
    """Project only reactivated lower-domain temperatures onto ``nabla >= 0``.

    The physical log-temperature gradient is linear in the first ``n``
    coupled coordinates.  Merely rejecting an inverted trial makes repeated
    half steps approach the active boundary without reaching it.  Projecting
    the proposed deep temperatures onto the monotone cone lets the nonlinear
    solve land on that boundary and continue along it.  Coordinates above the
    first same-run reactivated interface, including real non-grey surface
    inversions, and all auxiliary ML2 coordinates are retained exactly.
    """
    old=np.asarray(old,dtype=float)
    proposed=np.asarray(proposed,dtype=float)
    if old.shape!=proposed.shape or old.ndim!=1 or old.size!=2*n-1:
        raise ValueError('Deep-gradient projection received incompatible states')
    if not 1<=first_index<n:
        raise ValueError('Deep-gradient projection starts outside the atmosphere')
    allowance=128*np.finfo(float).eps*max(1.,float(np.max(abs(old[:n]))))
    if np.any(np.diff(old[:n])[first_index-1:] < -allowance):
        raise ValueError('Accepted state already violates the deep-gradient floor')
    result=proposed.copy()
    result[first_index:n]=np.maximum.accumulate(
        result[first_index-1:n]
    )[1:]
    return result


@contextmanager
def augmented_coordinate_solver(material_options,material_factory, *, system_class=None):
    system_class = CoupledMaterialSystem if system_class is None else system_class
    original=adaptive.solve_trust_region_newton
    def solve(initial,evaluate,**settings):
        first=evaluate(initial,False);p=first.payload
        if not (p.get('energy_balance_is_physical_flux') and p.get('local_energy_equations_enforced')
                and p.get('convection_transport') is not None):
            return original(initial,evaluate,**settings)
        first=evaluate(initial,True)
        material=material_factory(*(material_options[k] for k in
            ('with_temperature','thermodynamics','rosseland_opacity','mixing_length_alpha')))
        system=system_class(first,evaluate,material);n=system.n
        system.same_run_domain_extension = bool(
            material_options.get('metadata', {}).get(
                'dq_domain_initialization', ''
            ).startswith('same-run appended cells:')
        )
        seed_records = material_options.get('metadata', {}).get(
            'dq_domain_seed_fluxes', ()
        )
        if system.same_run_domain_extension:
            nodes = [record['node'] for record in seed_records]
            if (not nodes or any(not isinstance(node, (int, np.integer))
                                 or not 1 <= node < n for node in nodes)):
                raise ValueError('Same-run domain extension lacks valid deep nodes')
            first = material_options.get('metadata', {}).get(
                'dq_minimum_temperature_gradient_index'
            )
            if (not isinstance(first, (int, np.integer))
                    or not 1 <= first <= min(nodes)):
                raise ValueError('Same-run domain extension lacks a persistent gradient floor')
            system.minimum_temperature_gradient_index = int(first)
        gate=settings.get('convergence_test');callback=settings.get('callback')
        handoff=settings.get('rejected_step_handoff')
        def physical_state(state):return np.linalg.solve(system.mapping,state[:n])
        def converged(state,ev,size):
            tolerance=settings.get('residual_tolerance',.002)
            p=ev.payload
            return (p['dq_augmented_flux_compatibility']<tolerance and
                np.max(abs(p['total_flux_interface']/system.target-1))<tolerance and
                np.max(abs(p['cell_energy_balance_relative_residual']))<tolerance and
                (gate is None or gate(physical_state(state),system.physical(state[:n],False),size)))
        def proposal(state,ev,j,radius):
            custom = getattr(system, 'build_proposal', None)
            if custom is None:
                direction,root=coupled_direction(j,ev.residual,n,radius)
                result = NonlinearProposal(direction)
            else:
                result, root = custom(state, ev, j, radius)
                direction = result.direction
            ev.payload['diagnostic_inner_proposal_root_solved']=root
            print(f'DQ coupled proposal: dlnT={np.max(abs(direction[:n])):.6g}, '
                f'root={root}, auxiliary/ML2 flux mismatch={ev.payload["dq_augmented_flux_compatibility"]:.6g}',flush=True)
            return result
        if callback is not None:
            settings['callback']=lambda record,state,ev:callback(record,physical_state(state),ev)
        if handoff is not None:
            settings['rejected_step_handoff']=lambda state,ev:handoff(physical_state(state),ev)
        projector=None
        if system.same_run_domain_extension:
            projector=lambda old,new:project_deep_monotone_temperature(
                old,new,n,system.minimum_temperature_gradient_index
            )
        settings.update(step_builder=proposal,step_measure=lambda old,new:float(np.max(abs(new[:n]-old[:n]))),
            convergence_test=converged,trial_projector=projector,allow_initial_convergence=False,
            jacobian_refresh_interval=1,finite_difference_fallback_step=None,broyden_updates=False,
            merit_function='least-squares',trust_update='model-agreement')
        print(f'DQ coupled temperature/ML2 phase: {n} temperatures, {n-1} auxiliary variables; no nested repair',flush=True)
        result=original(system.initial,system.evaluate,**require_measured_proposal(settings))
        physical=system.physical(result.state[:n],False)
        diagnostics=replace(result.diagnostics,
            final_residual_rms=float(np.sqrt(np.mean(physical.residual**2))),
            final_residual_maximum=float(np.max(abs(physical.residual))),
            final_worst_residual_index=int(np.argmax(abs(physical.residual))))
        return replace(result,state=physical_state(result.state),evaluation=physical,diagnostics=diagnostics)
    with patch.object(adaptive,'solve_trust_region_newton',solve):yield


class AugmentedRefractiveDQMaterial(StableREOSSecantMixin,RefractiveDQMaterial):
    coupled_system_class = CoupledMaterialSystem
    finish_conditioning_in_energy_equations=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._completed_domain=None
        from .provenance import digest
        self.experiment_metadata['convection_coordinates']='coupled logT/signed ML2; physical flux independently checked'
        self.experiment_metadata['coupled_source_sha256']=digest(__file__)
        self.experiment_metadata['coupled_initialization']='cold gray/projection/rescaling, then physical energy immediately'
        self.experiment_metadata['coupled_trust_bounds']='same box on logT and nondimensional signed ML2 variables'
    def solve(self,*args,**kwargs):
        original=dq.solve_adaptive_lte_structure
        factory=partial(RefractiveMaterialCoefficients,index_callback=partial(interface_index,self))
        def adapted(seed,wave,**options):
            seed,options=initialize_physical_extension(self._completed_domain,seed,options,factory)
            # The enlarged system resolves the physical gradient constraint;
            # do not precede it with the old approximate-gradient iterations.
            options=dict(options,use_convective_gradient_preconditioner=False)
            with augmented_coordinate_solver(options,factory, system_class=self.coupled_system_class):
                return original(seed,wave,**options)
        with patch.object(dq,'solve_adaptive_lte_structure',adapted):result=super().solve(*args,**kwargs)
        self._completed_domain=result
        return result


