"""DQ-only LTE carbon screening with an explicit maximum-opacity bound.

The shared pseudo-Voigt width is at least its Gaussian FWHM. Both mixture
components have peaks below 1/(sqrt(2*pi)*sigma_thermal). Multiplying by the
unreduced LTE lower population bounds a line at every wavelength. Discard
only line/layer pairs whose summed bound is below 1e-10 of the continuum.
This is a numerical error budget, not a temperature/abundance cutoff.
"""
from types import MappingProxyType
import numpy as np

from .dq_hornkohl_consistent import anchored_grid
from wd_spectra import metals
from wd_spectra.constants import BOLTZMANN, PLANCK, LIGHT_SPEED, PI, ELEMENTARY_CHARGE_ESU, ELECTRON_MASS


class BoundedCarbonLines:
    def __init__(self,database,keys,anchors):
        self.database,self.keys,self.anchors=database,tuple(keys),anchors
        requested=set(keys)
        records={}
        resonances={}
        for ion in database.ion_stages('C'):
            levels={level.index:level for level in ion.levels}
            rates={}
            for line in ion.transitions:
                rates[line.upper_index]=rates.get(line.upper_index,0.)+line.einstein_a
            for line in ion.transitions:
                key=('C',ion.charge,line.lower_index,line.upper_index)
                if key in requested and line.transition_type=='E1':
                    if key in records:
                        raise ValueError('Ambiguous carbon transition key')
                    lower=levels[line.lower_index]
                    records[key]=(line.wavelength_vacuum_angstrom,line.absorption_oscillator_strength,
                        lower.energy_wavenumber,lower.statistical_weight,ion.atomic_mass_u)
                    support=metals.strong_uv_resonance_minimum_half_window_angstrom(ion,line,lower)
                    if support>0:
                        rate=line.radiative_damping_rate_s
                        resonances[key]=(support,rates[line.upper_index] if rate is None else rate)
        if set(records)!=requested:
            raise ValueError('Missing declared carbon transition')
        self.parameters=np.asarray([records[key] for key in self.keys]).T
        self.charges=np.asarray([key[1] for key in self.keys])
        self.resonances=resonances
        self.uv_support=None
        self._resonance_indices={key:i for i,key in enumerate(self.keys) if key in resonances}

    def prepare(self,a,carbon):
        """Decide resonance support on the complete atmosphere before slicing.

        Returns whether the support changed. Because this decision is nonlocal,
        callers must invalidate cached local opacity columns when it changes.
        The criterion intentionally matches the shared full-state LTE gate;
        it does not introduce a new physical wing prescription.
        """
        if (a.n_depth<2 or np.any(np.diff(a.column_mass)<=0)
            or np.any(~np.isfinite(a.column_mass)) or np.any(a.column_mass<=0)):
            raise ValueError('UV support requires a complete ordered atmosphere')
        support={}
        prefactor=PI*ELEMENTARY_CHARGE_ESU**2/(ELECTRON_MASS*LIGHT_SPEED)
        for key,(half_window,natural_rate) in self.resonances.items():
            i=self._resonance_indices[key]
            center,f,energy,g,mass=self.parameters[:,i]
            t=a.temperature
            population=(carbon.ion_number_density['C'][key[1]]
                /carbon.partition_function[('C',key[1])]*g
                *np.exp(-energy*PLANCK*LIGHT_SPEED/(BOLTZMANN*t)))
            sigma=center*np.sqrt(BOLTZMANN*t/(mass*1.66053906892e-24*LIGHT_SPEED**2))
            gamma=(center*1e-8)**2*natural_rate/(4*PI*LIGHT_SPEED)*1e8
            peak=np.array([metals._pseudo_voigt_profile_per_angstrom(
                np.array([center]),center,float(s),float(gamma))[0] for s in sigma])
            stimulated=-np.expm1(-PLANCK*LIGHT_SPEED/(center*1e-8*BOLTZMANN*t))
            opacity=(prefactor*f*peak*center**2*1e-8/LIGHT_SPEED
                *population*stimulated/a.mass_density)
            tau=float(opacity[0]*a.column_mass[0]
                +np.sum(.5*(opacity[1:]+opacity[:-1])*np.diff(a.column_mass)))
            support[key]=half_window if tau>=1e3 else 0.
        changed=self.uv_support is None or support!=self.uv_support
        self.uv_support=MappingProxyType(support)
        return changed

    def upper_bounds(self,a,carbon):
        center,f,energy,g,mass=self.parameters
        t=a.temperature[None,:]
        population=np.empty((len(center),a.n_depth))
        for charge in np.unique(self.charges):
            use=self.charges==charge
            population[use]=(carbon.ion_number_density['C'][charge][None,:]
                /carbon.partition_function[('C',int(charge))][None,:]
                *g[use,None]*np.exp(-energy[use,None]*PLANCK*LIGHT_SPEED/(BOLTZMANN*t)))
        sigma=center[:,None]*np.sqrt(BOLTZMANN*t/(mass[:,None]*1.66053906892e-24))/LIGHT_SPEED
        stimulated=-np.expm1(-PLANCK*LIGHT_SPEED/(center[:,None]*1e-8*BOLTZMANN*t))
        prefactor=PI*ELEMENTARY_CHARGE_ESU**2/(ELECTRON_MASS*LIGHT_SPEED)
        return (prefactor*f[:,None]*population*stimulated/a.mass_density[None,:]
            *(center[:,None]**2*1e-8/LIGHT_SPEED)/(np.sqrt(2*PI)*sigma))

    def evaluate(self,a,carbon,wavelength,background,*,relative_tolerance=1e-10):
        if self.uv_support is None:
            raise RuntimeError('Prepare UV support on the complete atmosphere before evaluating subsets')
        floor=np.min(np.asarray(background),axis=0)
        if (not 0<relative_tolerance<=1e-8 or floor.shape!=(a.n_depth,)
            or np.any(~np.isfinite(floor)) or np.any(floor<=0)):
            raise ValueError('A strictly positive continuum/error budget is required')
        bounds=self.upper_bounds(a,carbon)
        skip=bounds<=relative_tolerance*floor[None,:]/len(self.keys)
        # A union costs a few more very weak line/layer evaluations, but avoids
        # repeating atomic setup for nearly every layer. Its omitted bound is
        # no larger than the original separate-layer screening bound.
        selected=np.flatnonzero(np.any(~skip,axis=1))
        omitted=np.sum(bounds[np.all(skip,axis=1)],axis=0)
        padded,indices=anchored_grid(wavelength,self.anchors)
        result=np.zeros((len(wavelength),a.n_depth))
        if len(selected):
            keys=tuple(self.keys[i] for i in selected)
            result=metals.metal_line_mass_absorption_coefficient(a,padded,self.database,carbon,
                minimum_oscillator_strength=1e-4,maximum_lines=None,transition_keys=keys,
                uv_resonance_support_angstrom=self.uv_support)[indices]
        return result,dict(relative_bound=float(np.max(omitted/floor)),
            retained_line_layers=int(len(selected)*a.n_depth),total_line_layers=int(skip.size),
            omitted_bound=omitted)
