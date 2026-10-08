"""Explicit temperature-gradient unknowns with the unchanged ML2 closure.

Research-only coordinate change, motivated by TLUSTY's explicit-gradient
formulation (reference manual, Appendix C1, arXiv:1706.01935). The raw
compatibility equation G log(T) - nabla_aux = 0 is linear. Unlike the signed
velocity chart, it has no artificial curvature in stable, zero-flux layers.
Convection, material responses, radiation and physical acceptance gates are
unchanged. The algebraic ML2 root is evaluated analytically, not iterated.
"""
import numpy as np
from wd_spectra._ml2_auxiliary import (
    ml2_auxiliary_from_gradient, ml2_auxiliary_compatibility, ml2_scaled_coefficients,
)
from wd_spectra.nonlinear import NonlinearEvaluation, RecoverableEvaluationError
from .dq_augmented_probe_reuse import ProbeReuseSystem


class ExplicitGradientSystem(ProbeReuseSystem):
    native_trust_geometry = True
    # Increasing this coordinate moves toward convection.  Moving in the
    # negative direction while both evaluated fluxes are zero is algebraic
    # stable-branch motion and may use the one-sided proposal geometry.
    one_sided_stable_gradient_coordinate = True

    def minimum_gradient_coordinate(self, evaluation):
        """Lower coordinate bound for newly physical deep interfaces.

        The absolute-gradient chart stores ``nabla/gradient_scale``.  A
        negative value at a reactivated lower reservoir is therefore a deep
        temperature inversion, not a harmless choice of coordinates.  Small
        non-grey inversions elsewhere in the atmosphere remain unconstrained.
        """
        result = np.full(self.n-1, -np.inf, dtype=float)
        first = getattr(self, 'minimum_temperature_gradient_index', None)
        if first is not None:
            if not 1 <= first < self.n:
                raise ValueError('Deep gradient floor starts outside the atmosphere')
            result[first-1:] = 0.
        return result

    def inactive_gradient_motion(self, old, new):
        """Identify auxiliary motion that stays on the flux-free branch.

        In the absolute-gradient chart the convection boundary also depends
        on the material state.  The only branch-independent statement is
        that moving the auxiliary gradient downward from an exactly stable
        anchor cannot turn convection on.
        """
        return new[self.n:] < old[self.n:]

    def trust_step_size(self, old, new):
        """Measure physical and convection-onset motion in native units.

        The cheap proposal model records which interfaces have exactly zero
        auxiliary and physical convective flux at the current anchor.  A
        negative coordinate change at those interfaces moves farther into
        stability and changes no flux, so it must not scale down the entire
        temperature step.  Every other auxiliary direction retains the
        native trust bound.
        """
        old = np.asarray(old, dtype=float)
        new = np.asarray(new, dtype=float)
        if old.shape != new.shape or old.shape != (2*self.n-1,):
            raise ValueError('Explicit-gradient trust states have the wrong shape')
        delta = new-old
        inactive = getattr(
            self, '_trust_inactive_gradient_coordinates',
            np.zeros(self.n-1, dtype=bool),
        )
        if np.asarray(inactive).shape != (self.n-1,):
            raise ValueError('Stable-gradient trust mask has the wrong shape')
        released = inactive & self.inactive_gradient_motion(old, new)
        constrained = ~released
        sizes = [float(np.max(abs(delta[:self.n])))]
        if np.any(constrained):
            sizes.append(float(np.max(abs(delta[self.n:][constrained]))))
        return max(sizes)

    def __init__(self, first, evaluate, material):
        from .dq_eos_knot_model import CURRENT_MATERIAL
        self.eos_trial_owner = CURRENT_MATERIAL.get() if material is not None else None
        self.eos_material = material
        super().__init__(first, evaluate, material)
        p = first.payload
        ad, loss, coefficient = self.coefficients(p)
        a, b = ml2_scaled_coefficients(loss, coefficient, self.target)
        # Fixed, local nondimensional units: the adiabatic gradient or the
        # ML2 excess carrying Fstar, whichever is larger. No Teff switch.
        self.gradient_scale = np.maximum(abs(ad), a+b)
        self.initial = self.state_from_temperature(self.initial[:self.n], first)

    def physical(self, x, need):
        ev=super().physical(x,need)
        if need and self.eos_trial_owner is not None:
            from .dq_eos_knot_tangent import corrected_evaluation
            ev=corrected_evaluation(self.eos_trial_owner,self.eos_material,ev)
            self.cached=(x.copy(),ev)
        return ev

    @staticmethod
    def coefficients(p):
        return tuple(p['convection_transport'][key][1:] for key in
            ('adiabatic_gradient', 'ml2_radiative_loss', 'ml2_flux_coefficient'))

    def state_from_temperature(self, x, physical):
        return np.r_[x, physical.payload['temperature_gradient'][1:]/self.gradient_scale]

    def compatible_state_jacobian(self, x, physical):
        """Derivative of the exact ML2-compatible state with respect to log T."""
        return np.vstack([
            np.eye(self.n),
            self.gradient_operator[1:]/self.gradient_scale[:, None],
        ])

    def evaluate(self, state, need):
        n = self.n
        p = self.physical(state[:n], need).payload
        if not (p.get('energy_balance_is_physical_flux') and p.get('local_energy_equations_enforced')):
            raise ValueError('Explicit-gradient DQ requires physical local-energy equations')
        values = self.coefficients(p)
        ad, loss, coefficient = values
        gradient = state[n:]*self.gradient_scale
        y = ml2_auxiliary_from_gradient(gradient, *values, self.target)
        a, b = ml2_scaled_coefficients(loss, coefficient, self.target)
        negative = np.minimum(y, 0.)
        stable = 1+negative**2
        root = np.sqrt(stable)
        norm = a*root+b*stable
        # This is the SAME current ML2 norm as the velocity formulation.
        # Evaluate the linear numerator directly to avoid cancellation.
        defect_gradient = p['temperature_gradient'][1:]-gradient
        compatibility = defect_gradient/norm
        with np.errstate(over='ignore', invalid='ignore'):
            auxiliary = np.r_[0., np.maximum(y, 0.)**3]*self.target
            scale = p['thermal_cell_emission']+auxiliary[:-1]+auxiliary[1:]
            defect = p['radiative_cell_energy_defect']+np.diff(auxiliary)
            energy = (p['radiative_flux_interface']+auxiliary)/self.target-1.
            energy = energy.copy()
            energy[:-1] -= defect/scale
        residual = np.r_[energy, compatibility]
        if np.any(~np.isfinite(residual)) or np.any(scale <= 0):
            raise RecoverableEvaluationError('Nonfinite explicit-gradient DQ trial')
        jacobian = None
        if need:
            _, tangent = self.material(state[:n], True)
            responses = tuple(v[1:] for v in tangent)
            # Differentiate the exact scalar ML2 root at fixed nabla_aux.
            _, ct, cy = ml2_auxiliary_compatibility(gradient, y, values,
                responses, np.zeros((n-1, n)), self.target)
            if np.any(cy == 0):
                raise RecoverableEvaluationError('Nondifferentiable zero-loss ML2 onset')
            yt = -ct/cy[:, None]
            yg = -self.gradient_scale/cy
            flux_y = 3*self.target*np.maximum(y, 0.)**2
            ft = np.zeros((n, n)); ft[1:] = flux_y[:, None]*yt
            fg = np.zeros((n, n-1)); fg[1:] = np.diag(flux_y*yg)
            et = (p['radiative_flux_log_temperature_jacobian']+ft)/self.target
            et[:-1] -= (p['radiative_cell_energy_log_temperature_jacobian']+
                         np.diff(ft, axis=0))/scale[:, None]
            et[:-1] += (defect/scale**2)[:, None]*(
                p['thermal_cell_emission_log_temperature_jacobian']+ft[:-1]+ft[1:])
            eg = fg/self.target
            eg[:-1] -= np.diff(fg, axis=0)/scale[:, None]
            eg[:-1] += (defect/scale**2)[:, None]*(fg[:-1]+fg[1:])
            _, loss_j, coefficient_j = responses
            velocity = np.cbrt(self.target/coefficient)
            log_vj = -coefficient_j/(3*coefficient[:, None])
            aj = loss_j*velocity[:, None]+a[:, None]*log_vj
            bj = 2*b[:, None]*log_vj
            norm_y = (a/root+2*b)*negative
            norm_t = aj*root[:, None]+bj*stable[:, None]+norm_y[:, None]*yt
            norm_g = norm_y*yg
            rt = (self.gradient_operator[1:]-compatibility[:, None]*norm_t)/norm[:, None]
            rg = (-self.gradient_scale-compatibility*norm_g)/norm
            jacobian = np.block([[et, eg], [rt, np.diag(rg)]])
        return NonlinearEvaluation(residual, jacobian, {**p,
            # Carry the derivative in exactly the coordinates of this
            # evaluation. Energy-row wrappers must not infer the chart
            # from a class name that a research launcher can replace.
            'dq_augmented_auxiliary_flux_state_jacobian':
                np.column_stack([ft, fg]) if need else None,
            'dq_augmented_auxiliary_flux': auxiliary,
            'dq_augmented_scaled_compatibility': compatibility,
            'dq_augmented_flux_compatibility': float(np.max(abs(
                auxiliary-p['convective_flux_interface']))/self.target)})


def gradient_material(wavelengths, *, system_class=ExplicitGradientSystem):
    from .dq_refined_convection_cold import refined_material
    class GradientDQ(refined_material(wavelengths, system_class)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                convection_coordinates='explicit logT / fixed-scaled temperature gradient',
                gradient_coordinate_source_sha256=digest(__file__),
                material_derivative_policy='analytic REOS thermal/window response at all T/P knots; small centered opacity probes; physical values unchanged',
                eos_corner_search=False,
                coupled_trust_bounds='same native-coordinate box for solve, clipping and radius growth')

        def solve(self, *args, **kwargs):
            # Differentiate EOS thermal fields analytically; the shared
            # radiation probes stay centered and independent of EOS corners.
            # Apply to every state/knot, never select by stellar temperature.
            from unittest.mock import patch
            import wd_spectra.adaptive_structure as adaptive
            from .dq_eos_knot_tangent import eos_probe_policy
            from .provenance import digest
            from . import dq_eos_knot_tangent as implementation
            from .dq_eos_knot_model import eos_trial_material
            from . import dq_eos_knot_model
            from . import dq_reos_secant_response
            from . import dq_eos_knot_search
            self.experiment_metadata['material_derivative_source_sha256'] = digest(implementation.__file__)
            self.experiment_metadata['eos_trial_model_source_sha256'] = digest(dq_eos_knot_model.__file__)
            self.experiment_metadata['eos_analytic_response_source_sha256'] = digest(dq_reos_secant_response.__file__)
            self.experiment_metadata['eos_corner_search_source_sha256'] = digest(dq_eos_knot_search.__file__)
            self.experiment_metadata['eos_trial_model'] = 'finite table at fixed host pressure; exact coupled origin/tangent; actual outer acceptance'
            def probes(evaluate, step, *, centered):
                return eos_probe_policy(self, evaluate, step, centered=centered)
            with patch.object(adaptive, 'temperature_response_probes', probes), eos_trial_material(self):
                return super().solve(*args, **kwargs)
    return GradientDQ


