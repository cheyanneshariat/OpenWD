"""Finite-material proposal coefficients with the same refractive ML2 law.

This is DQ-local. It adds derivatives of the existing grey index correction
to the proposal map; no accepted EOS, opacity or flux equation is changed.
"""
from dataclasses import replace
import numpy as np
from wd_spectra.adaptive_structure import _positive_interface_values
from .convective_consistency_experiment import MaterialCoefficients
from .dq_refractive_material import grey_ml2_loss_factor, static_index


def grey_loss_log_derivatives(tau,index):
    """dln(multiplier)/dln(tau), dln(multiplier)/dln(n)."""
    tau=np.asarray(tau);n=np.asarray(index)
    inverse=1/(1+.5*np.minimum(tau,1e150)**2)
    n4=n**4;denominator=inverse+(1-inverse)*n4
    return 2*inverse*(1-inverse)*(1-n4)/denominator,4*inverse/denominator


def interface_index(material,interface):
    if material.helium_reos3 is None:
        raise ValueError('Refractive DQ coordinates require the declared REOS host')
    saved=material.cached
    try:
        if interface.helium_lte_state is None:
            from wd_spectra.carbon_molecular import _reos_host_state
            interface=replace(interface,helium_lte_state=_reos_host_state(
                interface.temperature,interface.gas_pressure,material.helium_reos3))
        return static_index(material.chemistry(interface)[0])
    finally:
        material.cached=saved


class RefractiveMaterialCoefficients(MaterialCoefficients):
    def __init__(self,*args,index_callback):
        super().__init__(*args);self.index_callback=index_callback

    def assemble(self,atmosphere,fields,slopes=None):
        base=super().assemble(atmosphere,fields,slopes)
        values=base if slopes is None else base[0]
        pos=_positive_interface_values
        interface=replace(atmosphere,temperature=pos(atmosphere.temperature),
            gas_pressure=pos(atmosphere.gas_pressure),mass_density=pos(fields[0]))
        n=self.index_callback(interface)
        tau=pos(fields[1])*self.alpha*interface.gas_pressure/interface.gravity
        factor=grey_ml2_loss_factor(tau,n)
        corrected=values[0],values[1]*factor,values[2]
        if slopes is None:return corrected
        # Refractivity depends on interface T at fixed interface P. The
        # geometric interface T gives equal adjacent log-T derivatives.
        h=getattr(self,'index_response_step',2e-4)
        high=self.index_callback(replace(interface,temperature=interface.temperature*np.exp(h)))
        low=self.index_callback(replace(interface,temperature=interface.temperature*np.exp(-h)))
        dn=(np.log(high)-np.log(low))/(2*h)
        size=len(n);average=.5*np.eye(size)+.5*np.eye(size,k=-1);average[0,0]=1.
        opacity_response=average*(slopes[1]/fields[1])[None,:]
        dtau,dindex=grey_loss_log_derivatives(tau,n)
        log_factor_response=dtau[:,None]*opacity_response+(dindex*dn)[:,None]*average
        ad_j,loss_j,coefficient_j=base[1]
        corrected_loss=factor[:,None]*(loss_j+values[1][:,None]*log_factor_response)
        return corrected,(ad_j,corrected_loss,coefficient_j)


