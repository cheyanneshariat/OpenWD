"""Finite-material ML2 trial correction; never cap the evaluated flux.

Sequentially lower trial gradients that would carry more than one stellar
flux, solving the actual nonlinear material/gradient relation. This is an
initializer/step safeguard, not a substitute for the energy equation. It
retains the surface temperature and never overwrites the physical ML2 flux.
"""
import numpy as np
from scipy.optimize import brentq
from .convective_consistency_experiment import MaterialCoefficients
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root


def project_trial(seed,runner,options):
    target=STEFAN_BOLTZMANN*seed.effective_temperature**4
    lp=np.log(seed.gas_pressure);original=np.log(seed.temperature)
    lt=original.copy();count=0
    for i in range(1,seed.n_depth):
        gradient=(original[i]-original[i-1])/(lp[i]-lp[i-1])
        if gradient>0:
            pressure=seed.gas_pressure[i-1:i+1]
            tau=seed.rosseland_optical_depth[i-1:i+1]
            def at(t):return runner.atmosphere_at(seed.effective_temperature,pressure,t,tau)
            material=MaterialCoefficients(at,options['thermodynamics'],options['rosseland_opacity'],
                options['mixing_length_alpha'])
            def residual(g):
                a,f=material.fields(np.array([lt[i-1],lt[i-1]+g*(lp[i]-lp[i-1])]))
                ad,loss,coefficient=material.assemble(a,f)
                contrast=np.cbrt(target/coefficient[1])
                return g-ad[1]-loss[1]*contrast-contrast**2
            if residual(gradient)>0:
                gradient=brentq(residual,0.,gradient,xtol=2e-13,rtol=1e-14)
                count+=1
        lt[i]=lt[i-1]+gradient*(lp[i]-lp[i-1])
    if not count:return seed,dict(corrected_interfaces=0)
    temperature=np.exp(lt);temperature[0]=seed.temperature[0]
    corrected=options['with_temperature'](temperature)
    material=MaterialCoefficients(options['with_temperature'],options['thermodynamics'],
        options['rosseland_opacity'],options['mixing_length_alpha'])
    a,f=material.fields(np.log(corrected.temperature))
    ad,loss,coefficient=material.assemble(a,f)
    gradient=np.r_[0.,np.diff(np.log(corrected.temperature))/np.diff(lp)]
    contrast,_=_ml2_contrast_and_root(np.maximum(gradient-ad,0.),loss)
    actual=coefficient*contrast**3/target
    if np.max(actual)>1+1e-6:
        raise RuntimeError('finite-material convective trial projection failed its evaluated-flux check')
    return corrected,dict(corrected_interfaces=count,maximum_actual_convective_flux_ratio=float(np.max(actual)))
