"""Cancellation-resistant finite-window derivatives of the SAME He-REOS.3.

Research-only. Keep the existing log-linear density, linear-energy/log-T
interpolant and +/-2e-4 log-T secant window. Integrate its piecewise constant
slopes over that window instead of subtracting two almost equal values.
There is no new EOS, smoothing, extrapolation, or temperature switch.
"""
import numpy as np

from wd_spectra.dense_eos import KJ_G_TO_ERG_G
from wd_spectra.eos import HeliumThermodynamics


def reos_secant_thermodynamics(table, temperature, pressure, *, epsilon=2e-4):
    t,p=np.broadcast_arrays(np.asarray(temperature,dtype=float),np.asarray(pressure,dtype=float))
    if (np.any(~np.isfinite(t+p)) or np.any(t<=0) or np.any(p<=0)
            or not np.isfinite(epsilon) or epsilon<=0):
        raise ValueError('Positive finite pressure, temperature and secant interval required')
    rho,_,inside=table.evaluate(p,t)
    cold_rho,_,cold_inside=table.evaluate(p,t*np.exp(-epsilon))
    _,_,hot_inside=table.evaluate(p,t*np.exp(epsilon))
    if not np.all(inside & cold_inside & hot_inside):
        raise ValueError('DQ He-REOS.3 secant outside supplied EOS domain')
    logt=np.log(t).ravel();logp=np.log(p).ravel()
    grid=np.log(table.temperature_grid)
    drho=np.zeros_like(logt);du=np.zeros_like(logt);covered=np.zeros_like(logt)
    # Distances relative to log(T), so a window entirely within one cell
    # has width exactly 2*epsilon, not (logT+eps)-(logT-eps).
    first=np.searchsorted(grid,logt-epsilon,side='right')-1
    last=np.searchsorted(grid,logt+epsilon,side='left')-1
    first=np.clip(first,0,len(grid)-2);last=np.clip(last,0,len(grid)-2)
    for i in range(int(np.min(first)),int(np.max(last))+1):
        selected=(first<=i)&(last>=i)
        if not np.any(selected):continue
        width=np.maximum(0.,np.minimum(epsilon,grid[i+1]-logt[selected])
                            -np.maximum(-epsilon,grid[i]-logt[selected]))
        lr0,u0,in0=table._at_isotherm(i,logp[selected])
        lr1,u1,in1=table._at_isotherm(i+1,logp[selected])
        if np.any((width>0)&~(in0&in1)):
            raise ValueError('DQ He-REOS.3 secant crosses unsupported table cell')
        fraction=width/(grid[i+1]-grid[i])
        drho[selected]+=(lr1-lr0)*fraction
        du[selected]+=(u1-u0)*fraction*KJ_G_TO_ERG_G
        covered[selected]+=width
    if not np.allclose(covered,2*epsilon,rtol=2e-11,atol=0):
        raise ValueError('DQ He-REOS.3 secant window is not fully covered')
    drho=drho.reshape(t.shape);du=du.reshape(t.shape)
    # h_hot-h_cold = (u_hot-u_cold) + P/rho_cold*(exp(-dlnrho)-1).
    # expm1 and sinh retain the small increments without cancellation.
    heat=(du+p/cold_rho*np.expm1(-drho))/(2*t*np.sinh(epsilon))
    expansion=-drho/(2*epsilon)
    adiabatic=p*expansion/(rho*t*heat)
    if np.any(~np.isfinite(heat+expansion+adiabatic)) or np.any(heat<=0) or np.any(expansion<=0):
        raise ValueError('Nonphysical tabulated He thermodynamics; no value clipped')
    return HeliumThermodynamics(specific_heat_constant_pressure=heat,
        density_temperature_derivative=expansion,adiabatic_temperature_gradient=adiabatic)


class StableREOSSecantMixin:
    """Use the same trace-He pressure convention, with stable secants only."""
    def thermodynamics(self,current):
        if self.helium_reos3 is None:
            raise ValueError('Stable REOS experiment requires the declared He-REOS.3 EOS')
        current=self.chemistry(current)[0]
        pressure=np.asarray(current.metadata['dq_helium_host_pressure'])
        return reos_secant_thermodynamics(self.helium_reos3,current.temperature,pressure)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .provenance import digest
        self.experiment_metadata['thermodynamic_secants']='same REOS interpolant/window; cancellation-resistant'
        self.experiment_metadata['thermodynamic_secant_source_sha256']=digest(__file__)
