"""Preserve coarse physical ML2 transport when initializing a finer mesh.

Only an initializer: no radiative flux is manufactured, and the fine grid
must still solve its own actual formal-transfer/ML2 energy equations.
"""
import numpy as np
from scipy.optimize import brentq
from .convective_consistency_experiment import MaterialCoefficients
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root


def interface_log_pressure(pressure):
    lp=np.log(pressure)
    return np.r_[lp[0],.5*(lp[:-1]+lp[1:])]


def prolongate(seed,parent,runner,options):
    coarse_flux=np.asarray(parent.metadata['convective_flux_fraction_by_interface'])
    if np.any(~np.isfinite(coarse_flux)) or np.any(coarse_flux<0):
        raise ValueError('coarse ML2 flux must be nonnegative and finite')
    desired=np.interp(interface_log_pressure(seed.gas_pressure),
        interface_log_pressure(parent.gas_pressure),coarse_flux)
    return transport_profile(seed,desired,runner,options)


def transport_profile(seed,desired,runner,options,*,project_stable=False):
    """Initialize prescribed convective fractions through actual ML2 slopes.

    Optional stable projection enforces zero desired flux by lowering an
    unstable interpolated slope to marginal stability, with exact materials.
    It changes trial temperatures only, never the evaluated physical flux.
    """
    desired=np.asarray(desired)
    if desired.shape!=seed.temperature.shape or np.any(~np.isfinite(desired)) or np.any(desired<0):
        raise ValueError('desired ML2 flux must match the depth grid and be finite and nonnegative')
    target=STEFAN_BOLTZMANN*seed.effective_temperature**4
    lp=np.log(seed.gas_pressure)
    original=np.log(seed.temperature)
    lt=original.copy()
    for i in range(1,seed.n_depth):
        if desired[i] == 0 and not project_stable:
            # In stable regions retain the interpolated radiative slope.
            lt[i]=lt[i-1]+original[i]-original[i-1]
            continue
        pressure=seed.gas_pressure[i-1:i+1]
        tau=seed.rosseland_optical_depth[i-1:i+1]
        def at(t): return runner.atmosphere_at(seed.effective_temperature,pressure,t,tau)
        material=MaterialCoefficients(at,options['thermodynamics'],options['rosseland_opacity'],
            options['mixing_length_alpha'])
        def residual(g):
            atmosphere,fields=material.fields(np.array([lt[i-1],lt[i-1]+g*(lp[i]-lp[i-1])]))
            ad,loss,coefficient=material.assemble(atmosphere,fields)
            contrast=np.cbrt(desired[i]*target/coefficient[1])
            return g-ad[1]-loss[1]*contrast-contrast**2
        if desired[i]==0:
            slope=(original[i]-original[i-1])/(lp[i]-lp[i-1])
            if slope<=0 or residual(slope)<=0:
                lt[i]=lt[i-1]+slope*(lp[i]-lp[i-1])
                continue
        lower,upper=0.,.5
        while residual(upper)<0:
            upper*=2
            if upper>4: raise RuntimeError('could not bracket convective prolongation')
        gradient=brentq(residual,lower,upper,xtol=2e-13,rtol=1e-14)
        lt[i]=lt[i-1]+gradient*(lp[i]-lp[i-1])
        if i%40==0 or i==seed.n_depth-1:
            print(f'Convective prolongation: layer {i+1}/{seed.n_depth}, T={np.exp(lt[i]):.6g}',flush=True)
    temperatures=np.exp(lt)
    temperatures[0]=seed.temperature[0]
    initialized=options['with_temperature'](temperatures)
    material=MaterialCoefficients(options['with_temperature'],options['thermodynamics'],
        options['rosseland_opacity'],options['mixing_length_alpha'])
    atmosphere,fields=material.fields(lt)
    ad,loss,coefficient=material.assemble(atmosphere,fields)
    gradient=np.r_[0.,np.diff(lt)/np.diff(lp)]
    contrast,_=_ml2_contrast_and_root(np.maximum(gradient-ad,0.),loss)
    actual=coefficient*contrast**3/target
    error=float(np.max(abs(actual-desired)))
    if error>1e-6:
        raise RuntimeError(f'convective prolongation failed its own transport check: {error:g}')
    return initialized,error
