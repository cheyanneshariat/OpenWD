"""Linear rate kernels versus independent direct frequency quadrature."""
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.constants import LIGHT_SPEED,PLANCK,BOLTZMANN,PI
from wd_spectra._nlte_radiative_integrals import continuum_integrals
from wd_spectra.multilevel_nlte import _photoionization_cross_section
from wd_spectra.helium_nlte import helium_ii_photoionization_cross_section,neutral_helium_term_photoionization_cross_section
from wd_spectra.spectrum import planck_lambda_angstrom


@pytest.mark.parametrize('cross_section,count,first',[
    (_photoionization_cross_section,8,1),
    (helium_ii_photoionization_cross_section,32,1),
    (neutral_helium_term_photoionization_cross_section,14,0)])
def test_cached_rates_match_direct_quadrature_and_follow_temperature(cross_section,count,first):
    wave=np.geomspace(20,100000,401)
    temperature=np.array([22000.,48000.,120000.])
    mean=planck_lambda_angstrom(wave[:,None],temperature)*np.linspace(.2,1.3,len(wave))[:,None]
    for factor in (1.,1.03):
        current=temperature*factor
        up,down=continuum_integrals(wave,current,mean,count,cross_section,first_level=first)
        nu=LIGHT_SPEED*1e8/wave[::-1]
        jnu=mean[::-1]*wave[::-1,None]**2/(LIGHT_SPEED*1e8)
        boltz=np.exp(-np.minimum(PLANCK*nu[:,None]/(BOLTZMANN*current),745))
        spontaneous=2*PLANCK*nu[:,None]**3/LIGHT_SPEED**2
        for level in range(count):
            sigma=cross_section(level+first,nu)
            valid=sigma>0
            expected_up=4*PI*trapezoid(sigma[valid,None]*jnu[valid]/(PLANCK*nu[valid,None]),nu[valid],axis=0)
            expected_down=4*PI*trapezoid(sigma[valid,None]*(spontaneous[valid]+jnu[valid])*boltz[valid]/(PLANCK*nu[valid,None]),nu[valid],axis=0)
            np.testing.assert_allclose(up[:,level],expected_up,rtol=3e-14,atol=0)
            np.testing.assert_allclose(down[:,level],expected_down,rtol=3e-14,atol=0)
