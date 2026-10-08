"""Coupled DQ equations with a differentiated current ML2 compatibility norm.

The fixed initial stability gap is not an appropriate error scale after a
layer becomes convective. Reuse the C1 norm already tested in the cool-model
experiments: stellar-flux gradient excess on the convective branch, and that
excess plus the current stability gap on the stable branch. Differentiate
the normalization, not just the numerator. Physical equations, fluxes and
qualification gates are unchanged. This remains research-only.
"""
import numpy as np
from wd_spectra.nonlinear import NonlinearEvaluation
from .scaled_ml2_compatibility import scaled_compatibility
from .dq_finite_material_coordinates import ExactMaterialCache
from . import dq_augmented_material_scaling as scaling
from .dq_augmented_conditioned_cold import conditioned_material


class CurrentNormSystem(scaling.MaterialScaledSystem):
    def __init__(self,first,evaluate,material):
        # The parent and new compatibility block share the exact same
        # constitutive derivative request. Do not double the opacity work.
        super().__init__(first,evaluate,ExactMaterialCache(material))

    def evaluate(self,state,need):
        ev=super().evaluate(state,need);n=self.n;p=ev.payload
        y=state[n:]*self.velocity_scale
        values=tuple(p['convection_transport'][key][1:] for key in
            ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        responses=None
        if need:
            _,tangent=self.material(state[:n],True)
            responses=tuple(a[1:] for a in tangent)
        residual,ct,cy=scaled_compatibility(p['temperature_gradient'][1:],y,
            values,responses,self.gradient_operator[1:],self.target)
        complete=ev.residual.copy();complete[n:]=residual
        jacobian=None
        if need:
            jacobian=ev.jacobian.copy()
            jacobian[n:,:n]=ct
            jacobian[n:,n:]=np.diag(cy*self.velocity_scale)
        return NonlinearEvaluation(complete,jacobian,
            {**p,'dq_augmented_scaled_compatibility':residual})


def current_norm_material(wavelengths=None, *, system_class=None):
    system_class = CurrentNormSystem if system_class is None else system_class
    class CurrentNormDQ(conditioned_material(wavelengths, system_class=system_class)):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                coupled_compatibility_norm='C1 current ML2 stellar-flux excess / stable-gap; fully differentiated',
                coupled_norm_source_sha256=digest(__file__))
    return CurrentNormDQ


