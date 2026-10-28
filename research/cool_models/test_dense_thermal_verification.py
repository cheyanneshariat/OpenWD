import numpy as np
from wd_spectra._cool.dense_thermal_verification import thermal_diagnostics
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity


def test_bolometric_tail_does_not_bound_local_cooling_tail():
    # Transparent cool skin with a free-free-like lambda^2 absorber.
    # Prescribed radiation is a diagnostic test fixture, not an atmosphere.
    wave=np.geomspace(100,1e8,8000);mass=np.geomspace(1e-8,1e-3,4)
    absorption=np.broadcast_to((wave/1e4)[:,None]**2,(len(wave),len(mass))).copy()
    tau=optical_depth_from_mass_opacity(mass,absorption)
    b=np.broadcast_to(planck_lambda_angstrom(wave,1400)[:,None],absorption.shape).copy()
    mean=b.copy();mean[wave>1e5]*=.7
    full=thermal_diagnostics(wave,tau,b,mean,np.ones_like(b),np.zeros(len(mass)))
    short=wave<=1e5
    restricted=thermal_diagnostics(wave[short],tau[short],b[short],mean[short],
        np.ones_like(b[short]),np.zeros(len(mass)))
    assert restricted['maximum_independent_cell_energy_defect']==0
    assert full['maximum_independent_cell_energy_defect']>.03
    emergent=planck_lambda_angstrom(wave,8000)
    assert trapezoid(emergent[~short],wave[~short])/trapezoid(emergent,wave)<1e-3
