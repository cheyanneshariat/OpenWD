"""DQ coordinate experiment: follow the finite adiabatic gradient in trials.

Unknowns are log T and (nabla_aux - nabla_ad(T))/fixed_local_excess_scale.
The scale is the larger of the initial excess and the ML2 excess carrying
Fstar, not Teff or a prescribed adiabatic replacement for the energy row.
All physical fluxes, equations, opacities and acceptance gates are unchanged.
"""
import numpy as np
from .dq_explicit_gradient import ExplicitGradientSystem as Base
from wd_spectra._ml2_auxiliary import ml2_scaled_coefficients
from wd_spectra.nonlinear import NonlinearEvaluation


class SuperadiabaticSystem(Base):
    def inactive_gradient_motion(self,old,new):
        """Release any motion wholly inside the zero-convection half-space."""
        return (old[self.n:]<=0.) & (new[self.n:]<=0.)

    def minimum_gradient_coordinate(self, evaluation):
        """Return the deep superadiabatic coordinate representing nabla = 0."""
        ad = self.coefficients(evaluation.payload)[0]
        result = np.full(self.n-1, -np.inf, dtype=float)
        first = getattr(self, 'minimum_temperature_gradient_index', None)
        if first is not None:
            if not 1 <= first < self.n:
                raise ValueError('Deep gradient floor starts outside the atmosphere')
            result[first-1:] = -ad[first-1:]/self.excess_scale[first-1:]
        return result

    def state_from_temperature(self,x,physical):
        p=physical.payload
        ad,loss,coefficient=self.coefficients(p)
        excess=p['temperature_gradient'][1:]-ad
        if not hasattr(self,'excess_scale'):
            a,b=ml2_scaled_coefficients(loss,coefficient,self.target)
            self.excess_scale=np.maximum(abs(excess),a+b)
            if np.any(self.excess_scale<=0):raise ValueError('Positive local excess units required')
        return np.r_[x,excess/self.excess_scale]

    def compatible_state_jacobian(self,x,physical):
        p=physical.payload
        ad_j=p['ml2_coefficient_log_temperature_responses'][0][1:]
        return np.vstack([np.eye(self.n),
            (self.gradient_operator[1:]-ad_j)/self.excess_scale[:,None]])

    @property
    def trial_system_class(self):
        return type(self)

    def copy_trial_coordinates(self,model):
        model.excess_scale=self.excess_scale.copy()

    def unrestricted_stationarity_candidate(self,ev,jacobian):
        # A tiny physical temperature correction may be large in units of
        # an extremely small superadiabatic excess. Measure the unrestricted
        # Newton correction for stationarity before a native box clips it.
        # Return a proposal, never a convergence certificate: the shared
        # driver's residual, physical, boundary and caller gates still apply.
        if not hasattr(self,'requested_step_tolerance'):
            return None
        p=ev.payload;tol=self.requested_residual_tolerance
        if (np.max(abs(ev.residual))>=tol or
            np.max(abs(p['total_flux_interface']/self.target-1))>=tol or
            np.max(abs(p['cell_energy_balance_relative_residual']))>=tol or
            p['dq_augmented_flux_compatibility']>=tol):
            return None
        from .dq_augmented_convection import coupled_direction
        direction,_=coupled_direction(jacobian,ev.residual,self.n,np.inf)
        if np.max(abs(direction[:self.n]))<self.requested_step_tolerance:
            return direction
        return None

    def evaluate(self,state,need):
        n=self.n
        p=self.physical(state[:n],need).payload
        ad=self.coefficients(p)[0]
        old=np.r_[state[:n],(ad+self.excess_scale*state[n:])/self.gradient_scale]
        ev=Base.evaluate(self,old,need)
        jacobian=None
        flux_jacobian=None
        if need:
            _,responses=self.material(state[:n],True)
            ad_j=responses[0][1:]
            jacobian=ev.jacobian.copy()
            jacobian[:,:n]+=(ev.jacobian[:,n:]/self.gradient_scale[None,:])@ad_j
            jacobian[:,n:]*=(self.excess_scale/self.gradient_scale)[None,:]
            old_flux_jacobian=ev.payload['dq_augmented_auxiliary_flux_state_jacobian']
            flux_jacobian=old_flux_jacobian.copy()
            flux_jacobian[:,:n]+=(old_flux_jacobian[:,n:]/self.gradient_scale[None,:])@ad_j
            flux_jacobian[:,n:]*=(self.excess_scale/self.gradient_scale)[None,:]
        return NonlinearEvaluation(ev.residual,jacobian,{**ev.payload,
            'dq_augmented_auxiliary_flux_state_jacobian':flux_jacobian,
            'dq_superadiabatic_scale':self.excess_scale.copy(),
            'dq_superadiabatic_excess':self.excess_scale*state[n:]})
