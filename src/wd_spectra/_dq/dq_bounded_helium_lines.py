"""DQ-only numerical screening of He I lines with a proven opacity bound.

A positive normalized profile convolved with a Lorentzian has a maximum
no larger than 1/(pi*gamma). The sum of omitted opacity bounds is required
to be smaller than the caller's numerical budget at EVERY wavelength.
This is not a temperature switch or a change to oscillator strengths.
"""
from collections import defaultdict
import numpy as np
import wd_spectra.helium as he
from .dq_exact_opacity_cache import subset


def upper_bounds(a):
    state=a.helium_lte_state
    if a.hydrogen_lte_state is not None:
        raise ValueError('This isolated optimization is restricted to hydrogen-free DQ')
    result=[]
    for line in he.HELIUM_I_LINES:
        center=line.wavelength_vacuum_angstrom*1e-8
        upper_energy=he.HELIUM_I_LOW_TERM_ENERGY[line.lower_term_index]+he.PLANCK*he.LIGHT_SPEED/center
        binding=max(he.HELIUM_FIRST_IONIZATION_ENERGY-upper_energy,np.finfo(float).tiny)
        radius=2.5*(he.HYDROGEN_IONIZATION_ENERGY/binding)**2
        neutral_rate=4.5e-9*.42*state.neutral_level_population_density[:,0]*(a.temperature/10000.)**.3*radius**.4
        radiative_rate=2.47342e-22*(he.LIGHT_SPEED/center)**2
        gamma=center**2*(neutral_rate+radiative_rate)/(4*he.PI*he.LIGHT_SPEED)*1e8
        upper=he.helium_occupation_probability(state.neutral_he_density,state.electron_density,
            a.temperature,float(line.upper_principal_quantum_number),
            neutral_h_density=a.neutral_h_density,neutral_radius_scale=state.neutral_radius_scale,
            hydrogen_neutral_radius_scale=1.,correlated_microfields=(state.microfield_model=='qmhd'))
        survival=np.minimum(upper/np.maximum(state.neutral_level_occupation_probability[:,line.lower_term_index],np.finfo(float).tiny),1.)
        prefactor=he.PI*he.ELEMENTARY_CHARGE_ESU**2/(he.ELECTRON_MASS*he.LIGHT_SPEED)
        bound=(prefactor*line.absorption_oscillator_strength
            *state.neutral_level_population_density[:,line.lower_term_index]*survival
            *(-np.expm1(-he.PLANCK*he.LIGHT_SPEED/(center*he.BOLTZMANN*a.temperature)))
            *1e8*center**2/he.LIGHT_SPEED/a.mass_density/(he.PI*gamma))
        result.append(bound)
    return np.asarray(result)


def bounded_helium_lines(a,wavelength,table,background,*,relative_tolerance=1e-10):
    w=np.asarray(wavelength,dtype=float)
    continuum=np.asarray(background,dtype=float)
    if (not 0<relative_tolerance<=1e-8 or continuum.shape!=(len(w),a.n_depth)
        or np.any(~np.isfinite(continuum)) or np.any(continuum<=0)):
        raise ValueError('A finite positive opacity floor and strict error budget are required')
    bounds=upper_bounds(a)
    floor=np.min(continuum,axis=0)
    skip=bounds<=relative_tolerance*floor[None,:]/len(he.HELIUM_I_LINES)
    omitted=np.sum(np.where(skip,bounds,0.),axis=0)
    out=np.zeros_like(continuum)
    groups=defaultdict(list)
    for j in range(a.n_depth):
        groups[tuple(np.flatnonzero(~skip[:,j]))].append(j)
    for lines,indices in groups.items():
        if lines:
            aa=subset(a,indices,a.n_depth)
            out[:,indices]=he.helium_i_line_mass_absorption_coefficient(aa,w,table,
                lines=tuple(he.HELIUM_I_LINES[i] for i in lines),neutral_broadening='unsold')
    return out,dict(omitted_bound=omitted,relative_bound=float(np.max(omitted/floor)),
        retained_line_layers=int(np.count_nonzero(~skip)),total_line_layers=int(skip.size))
