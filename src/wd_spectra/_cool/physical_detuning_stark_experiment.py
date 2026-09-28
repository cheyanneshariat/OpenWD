"""Interpolate Doppler-convolved Lyman tables at fixed physical detuning.

Resample each original density corner on the union of its physical wavelength
knots before interpolating log profile in log density/temperature. This keeps
the thermal core on a physical wavelength scale rather than stretching its
Gaussian through the plasma-field coordinates. Original table nodes and
piecewise power-law wings are retained (no renormalization). This is an
explicit interpolation experiment, not a new Stark calculation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch
import numpy as np
from wd_spectra.stark import StarkLine,_bracket


@dataclass(frozen=True)
class PhysicalDetuningLine(StarkLine):
    def _local_profile_state(self,temperature,electron_density):
        if not np.isfinite(temperature) or temperature<=0 or not np.isfinite(electron_density) or electron_density<=0:
            raise ValueError('temperature and electron density must be finite positive')
        nt,ft=_bracket(self.log_temperature,np.clip(np.log10(temperature),*self.log_temperature[[0,-1]]))
        nn,fn=_bracket(self.log_electron_density,np.clip(np.log10(electron_density),*self.log_electron_density[[0,-1]]))
        profile=((1-fn)*((1-ft)*self.log_profile[nn,nt]+ft*self.log_profile[nn,nt+1])
                 +fn*((1-ft)*self.log_profile[nn+1,nt]+ft*self.log_profile[nn+1,nt+1]))
        # log_alpha now stores log10(|Delta lambda| / Angstrom), and the
        # profile is per Angstrom. No field scaling is applied a second time.
        return 1.,profile


def physical_detuning_line(line):
    field_log=np.log10(1.25e-9)+(2/3)*line.log_electron_density
    grid=np.unique((line.log_alpha[None,:]+field_log[:,None]).ravel())
    profiles=np.empty((*line.log_profile.shape[:2],len(grid)))
    for i,field in enumerate(field_log):
        for j in range(len(line.log_temperature)):
            values=line.log_profile[i,j]-field
            x=line.log_alpha+field
            profile=np.interp(grid,x,values)
            beyond=grid>x[-1]
            slope=(values[-1]-values[-2])/(x[-1]-x[-2])
            profile[beyond]=values[-1]+slope*(grid[beyond]-x[-1])
            profiles[i,j]=profile
    return PhysicalDetuningLine(line.lower_level,line.upper_level,grid,
        line.log_electron_density,line.log_temperature,profiles,line.goodness_flag)


@contextmanager
def physical_detuning_lyman():
    original=StarkLine.wavelength_profile
    cache={}
    def profile(line,*args,**kwargs):
        if line.lower_level!=1 or isinstance(line,PhysicalDetuningLine):
            return original(line,*args,**kwargs)
        if id(line) not in cache:cache[id(line)]=physical_detuning_line(line)
        return original(cache[id(line)],*args,**kwargs)
    with patch.object(StarkLine,'wavelength_profile',profile):yield
