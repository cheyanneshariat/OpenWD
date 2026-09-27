"""Nonlinear material compatibility repair for a coupled trial, not its flux.

Hold proposed convective transport while solving actual material slopes.
Stable auxiliary velocities are then derived from the repaired temperature
gradient. The outer solver evaluates actual radiation and ML2, with its
normal trust limit and physical convergence gates.
"""
import numpy as np
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from wd_spectra.constants import STEFAN_BOLTZMANN
from .convective_consistency_experiment import MaterialCoefficients
from .convective_mesh_prolongation import transport_profile


def project_full_trial(old,trial,*,scale,mapping,options):
    from . import check_cool_db_transport_seed as runner
    n=len(scale)+1
    candidate=options['with_temperature'](np.exp(mapping@trial[:n]))
    desired=np.r_[0.,np.maximum(trial[n:]*scale,0.)**3]
    corrected,error=transport_profile(candidate,desired,runner,options,project_stable=True)
    logt=np.log(corrected.temperature)
    material=MaterialCoefficients(options['with_temperature'],options['thermodynamics'],
        options['rosseland_opacity'],options['mixing_length_alpha'])
    coefficients=material(logt)
    gradient=np.diff(logt)/np.diff(np.log(corrected.gas_pressure))
    y=ml2_auxiliary_from_gradient(gradient,*(x[1:] for x in coefficients),
        STEFAN_BOLTZMANN*corrected.effective_temperature**4)
    projected=np.r_[np.linalg.solve(mapping,logt),y/scale]
    print(f'FULL ML2 MATERIAL COMPATIBILITY: maximum dlnT '
        f'{np.max(abs(mapping@(projected[:n]-old[:n]))):.6g}; '
        f'actual convective-fraction defect {error:.3g}',flush=True)
    return projected
