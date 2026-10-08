"""DQ coupled-variable scaling based on the local adiabatic gradient.

The initial departure from adiabaticity can be arbitrarily small in a cold
ML2-projected seed. It is not a useful unit for an auxiliary stable gradient.
Use the velocity corresponding to the whole locally stable gradient range
(zero through nabla_ad), as well as the stellar-flux velocity. Scales remain
fixed throughout a nonlinear phase; no residual equation or root is changed.
This is a research experiment, not a public default.
"""
import numpy as np
from . import dq_augmented_convection as coupled
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient,ml2_scaled_coefficients


class MaterialScaledSystem(coupled.CoupledMaterialSystem):
    def __init__(self, first, evaluate, material):
        super().__init__(first,evaluate,material)
        values=tuple(first.payload['convection_transport'][k][1:] for k in
            ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        ad,loss,coefficient=values
        stable_velocity=ml2_auxiliary_from_gradient(np.zeros_like(ad),*values,self.target)
        velocity=self.initial[self.n:]*self.velocity_scale
        self.velocity_scale=np.maximum(self.velocity_scale,abs(stable_velocity))
        a,b=ml2_scaled_coefficients(loss,coefficient,self.target)
        self.compatibility_scale=a*self.velocity_scale+b*self.velocity_scale**2
        self.initial[self.n:]=velocity/self.velocity_scale


def scaled_material(wavelengths=None, *, system_class=None):
    if wavelengths is not None:
        raise ValueError('The release DQ protocol requires its full structure grid')
    base=coupled.AugmentedRefractiveDQMaterial
    selected_system = MaterialScaledSystem if system_class is None else system_class
    class MaterialScaledDQ(base):
        coupled_system_class = selected_system
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                coupled_variable_units='fixed max(stellar-flux, initial, local stable-gradient velocity)',
                coupled_scaling_source_sha256=digest(__file__))
    return MaterialScaledDQ

