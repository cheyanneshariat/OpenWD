"""Fresh discrete diffusion/ML2 initialization using the solver's materials.

Sampling an ODE dense interpolant does not guarantee that the discrete
temperature differences carry its intended convective flux, particularly
under mesh refinement. Solve each interface's SAME discrete ML2 + diffusion
equation instead. This remains an approximate INITIAL atmosphere, never a
substitute for formal transfer or a convergence certificate.
"""
import numpy as np
from scipy.optimize import brentq
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root
from .convective_consistency_experiment import MaterialCoefficients


def discrete_seed(seed,runner,options):
    logp=np.log(seed.gas_pressure)
    logt=np.log(seed.temperature).copy()
    target=STEFAN_BOLTZMANN*seed.effective_temperature**4
    maximum_defect=0.
    for i in range(1,seed.n_depth):
        pressure=seed.gas_pressure[i-1:i+1]
        tau=seed.rosseland_optical_depth[i-1:i+1]
        def with_temperature(t):
            return runner.atmosphere_at(seed.effective_temperature,pressure,t,tau)
        material=MaterialCoefficients(with_temperature,options['thermodynamics'],
            options['rosseland_opacity'],options['mixing_length_alpha'])
        def residual(gradient):
            temperatures=np.array([logt[i-1],logt[i-1]+gradient*(logp[i]-logp[i-1])])
            atmosphere,fields=material.fields(temperatures)
            ad,loss,coefficient=material.assemble(atmosphere,fields)
            contrast,_=_ml2_contrast_and_root(np.maximum(gradient-ad[1],0.),loss[1])
            kappa=np.sqrt(fields[1,0]*fields[1,1])
            tface=np.exp(np.mean(temperatures)); pface=np.sqrt(np.prod(pressure))
            radiation=16*STEFAN_BOLTZMANN*seed.gravity*tface**4/(3*kappa*pface)*gradient
            return (radiation+coefficient[1]*contrast**3)/target-1
        upper=.5
        while residual(upper) < 0:
            upper*=2
            if upper > 4:
                raise RuntimeError('discrete transport initializer failed to bracket gradient')
        gradient=brentq(residual,0.,upper,xtol=2e-13,rtol=1e-14)
        defect=abs(residual(gradient))
        maximum_defect=max(maximum_defect,defect)
        logt[i]=logt[i-1]+gradient*(logp[i]-logp[i-1])
        if i % 20 == 0 or i == seed.n_depth-1:
            print(f'discrete seed: layer {i+1}/{seed.n_depth} T={np.exp(logt[i]):.6g} '
                  f'diffusion+ML2 defect={defect:.3g}',flush=True)
    if maximum_defect > 1e-6:
        raise RuntimeError(f'discrete initialization failed its own transport equation: {maximum_defect:g}')
    return options['with_temperature'](np.exp(logt)),maximum_defect
