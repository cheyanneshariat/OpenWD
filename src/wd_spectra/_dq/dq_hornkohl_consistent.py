from .data import line_root
"""DQ-local consistent-opacity / native-transfer cold-start experiment.

The shared atmosphere solver is unchanged. Both structure and synthesis use
the same complete declared atomic selection, unscaled Hornkohl Swan opacity,
and coupled Feautrier surface flux. Thermal molecular widths and the existing
0.2*rho-eV shift are retained: this is not a new collisional prescription.
An independent spectrum grid must pass before claiming spectral convergence.
"""
import json
import time

import numpy as np

from . import base as dq
from .provenance import digest
from .dqsolution_sampling import SparseResolvedSwan
from .dqsolution_sampling_cold import sampling_grid
from .dq_exact_opacity_cache import ExactLayerOpacityCache
from .continuum_batches import helium_continuum_absorption
from wd_spectra.spectrum import Spectrum, solve_spectrum_source, planck_lambda_angstrom
from wd_spectra.opacity import optical_depth_from_mass_opacity

PARAMETER_KEYS=('effective_temperature','logg','log_carbon_to_helium')






def atomic_keys_and_anchors(database, threshold=1e-4):
    selected=[(ion,line) for ion in database.ion_stages('C') for line in ion.transitions
        if line.transition_type=='E1' and line.absorption_oscillator_strength>=threshold]
    if not selected:
        raise ValueError('No declared carbon lines')
    centers=np.array([line.wavelength_vacuum_angstrom for ion,line in selected])
    if np.any(~np.isfinite(centers)) or np.any(centers<=0):
        raise ValueError('Invalid atomic line center')
    return tuple(('C',ion.charge,line.lower_index,line.upper_index) for ion,line in selected),\
        (min(100.,float(centers.min())),max(100000.,float(centers.max())))


def anchored_grid(wave, anchors):
    w=np.asarray(wave,dtype=float)
    if w.ndim!=1 or len(w)<1 or np.any(~np.isfinite(w)) or np.any(w<=0) or np.any(np.diff(w)<=0):
        raise ValueError('Invalid wavelength grid')
    padded=np.unique(np.r_[anchors,w])
    return padded,np.searchsorted(padded,w)


class ConsistentDQMaterial(dq.DQMaterial):
    deadline=None

    def check_budget(self):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            raise TimeoutError('Research wall-time budget exhausted; last diagnostic state retained, no fallback')

    def __init__(self,config,data,table,*,sampling_r=30000.,sampling_phase=.5):
        super().__init__(config,data,table)
        report=json.loads((line_root()/'report.json').read_text())
        line_path=line_root()/report['artifact']['file']
        if not report.get('completed') or digest(line_path)!=report['artifact']['sha256']:
            raise ValueError('Hornkohl molecular data identity failed')
        self.line_sha=digest(line_path)
        self.hornkohl=SparseResolvedSwan(table,branches=line_path,cache_columns=192,cache_grids=192)
        self.atomic_keys,self.atomic_anchors=atomic_keys_and_anchors(self.atomic)
        self.sampling_r,self.sampling_phase=sampling_r,sampling_phase
        self.saved_structure_grid=None
        self.spectrum_grid_checks={}
        self.opacity_cache=ExactLayerOpacityCache()
        self.opacity_calls=0
        self.last_opacity_progress=time.monotonic()
        self.ca_table=None

    def absorption(self,current,wavelength,*,include_c2=True,structure=False):
        self.check_budget()
        # Deliberately no structure/synthesis branching in this opacity path.
        if len(wavelength)==1:
            value=float(wavelength[0])
            padded=np.array([value,np.nextafter(value,np.inf)])
            return self.absorption(current,padded,include_c2=include_c2,structure=structure)[:1]
        a,carbon,c2=self.chemistry(current)
        self.prepare_opacity_state(a,carbon,c2)
        self.opacity_calls+=1
        value=self.opacity_cache.evaluate(a,carbon,c2,wavelength,include_c2,self._absorption_from_chemistry)
        if len(wavelength)>10000 and time.monotonic()-self.last_opacity_progress>30:
            print(json.dumps(dict(opacity_calls=self.opacity_calls,
                evaluated_columns=self.opacity_cache.evaluated_columns,
                reused_columns=self.opacity_cache.reused_columns)),flush=True)
            self.last_opacity_progress=time.monotonic()
        return value

    def prepare_opacity_state(self,a,carbon,c2):
        pass

    def helium_line_opacity(self,a,wavelength,background):
        return dq.helium_i_line_mass_absorption_coefficient(a,wavelength,self.he_i,neutral_broadening='unsold')

    def swan_column(self,a,j,wavelength):
        shift=dq.swan_density_shift_wavenumber(a.mass_density[j])
        nu=1e8/np.asarray(wavelength)-shift
        cross=np.zeros(len(wavelength));use=nu>0
        cross[use]=self.hornkohl.swan_column(1e8/nu[use],float(a.temperature[j]))
        return cross

    def carbon_line_opacity(self,a,carbon,wavelength,background):
        padded,indices=anchored_grid(wavelength,self.atomic_anchors)
        return dq.metal_line_mass_absorption_coefficient(a,padded,self.atomic,carbon,
            minimum_oscillator_strength=1e-4,maximum_lines=None,transition_keys=self.atomic_keys)[indices]

    def _absorption_from_chemistry(self,a,carbon,c2,wavelength,*,include_c2=True):
        dense=self.dense_continuum.correction(wavelength,a.temperature,a.helium_lte_state.neutral_he_density)\
            if self.dense_continuum is not None else None
        opacity=helium_continuum_absorption(a,wavelength,dense)
        opacity+=self.helium_line_opacity(a,wavelength,opacity)
        opacity+=dq.helium_i_resonance_line_mass_absorption_coefficient(a,wavelength)
        opacity+=dq.helium_ii_line_mass_absorption_coefficient(a,wavelength,stark_table=self.he_ii)
        opacity+=dq.metal_bound_free_mass_absorption_coefficient(a,wavelength,self.atomic,carbon,self.photo)
        opacity+=self.carbon_line_opacity(a,carbon,wavelength,opacity)
        if include_c2:
            neutral=carbon.host_ion_number_density[0]
            non_swan=np.maximum(self.c2_profiles.cross_section(wavelength,a.temperature,neutral)
                -self.c2_swan_profiles.cross_section(wavelength,a.temperature,neutral),0.)
            for j,temp in enumerate(a.temperature):
                non_swan[:,j]+=self.swan_column(a,j,wavelength)
            if self.ca_table is not None:
                non_swan+=self.ca_table.cross_section_for_wavelength_temperature(wavelength,a.temperature)
            opacity+=non_swan*(c2/a.mass_density)[None,:]
        return opacity

    def structure_grid(self,seed,n_continuum):
        original=super().structure_grid(seed,n_continuum)
        optical=sampling_grid(3800.,6800.,self.sampling_r,self.sampling_phase)
        wave=np.unique(np.r_[original,optical])
        self.saved_structure_grid=wave
        print(f'DQ consistent opacity: {len(self.atomic_keys)} atomic lines, {len(wave)} wavelengths',flush=True)
        return wave

    def spectrum(self,atmosphere,wavelength,n_angle,*,include_c2=True):
        a=self.chemistry(atmosphere)[0]
        wave=np.asarray(wavelength)
        flux=np.empty(len(wave)); error=0.
        for first in range(0,len(wave),2000):
            w=wave[first:first+2000]
            query=w
            indices=np.arange(len(w))
            if len(w)==1:
                w=np.array([w[0],np.nextafter(w[0],np.inf)])
            absorption=self.absorption(a,w,include_c2=include_c2)
            scattering=self.scattering(a,w)
            tau=optical_depth_from_mass_opacity(a.column_mass,absorption+scattering)
            planck=planck_lambda_angstrom(w[:,None],a.temperature[None,:])
            source,field,meta=solve_spectrum_source(tau,planck,absorption,scattering,
                wavelength=w,n_angle=n_angle,discretization='optical-depth')
            flux[first:first+len(query)]=field.interface_flux[indices,0]
            error=max(error,meta['independent_radiation_scaled_source_error'])
            if len(wave)>30000:
                print(f'DQ consistent synthesis: {first+len(query)}/{len(wave)} wavelengths',flush=True)
        if np.any(~np.isfinite(flux)) or np.any(flux<=0):
            raise ValueError('Invalid native emergent spectrum')
        return Spectrum(wave,flux,dict(transfer_discretization='coupled-Feautrier-native',
            hornkohl_sha256=self.line_sha,atomic_line_count=len(self.atomic_keys),
            opacity_scale=1.,independent_radiation_scaled_source_error=error))


