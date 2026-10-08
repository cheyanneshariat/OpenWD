"""Explicit same-physics temperature continuation; never a cold-start claim.

Map T/Teff at fixed Rosseland depth onto the new hydrostatic pressure grid.
This is only a predictor; all materials and the full energy equations must
be recalculated at the requested Teff. No EOS or opacity is substituted.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator


def predicted_temperature(tau,temperature,old_teff,new_tau,new_teff):
    if old_teff<=0 or new_teff<=0:
        raise ValueError('effective temperatures must be positive')
    if np.min(new_tau)<tau[0]*(1-1e-12) or np.max(new_tau)>tau[-1]*(1+1e-12):
        raise ValueError('temperature predictor refuses optical-depth extrapolation')
    coordinates=np.clip(np.log(new_tau),np.log(tau[0]),np.log(tau[-1]))
    return new_teff*np.exp(PchipInterpolator(np.log(tau),np.log(temperature/old_teff),
        extrapolate=False)(coordinates))


class TemperaturePredictor:
    def __init__(self,directory,interaction_table,*,preserve_convection=False,reuse_pressure=False,
                 truncate_optical_depth=None,refine_pressure=False,unscaled_temperature=False):
        self.directory=Path(directory).resolve()
        self.metadata=json.loads((self.directory/'experimental-metadata.json').read_text())
        self.summary=json.loads((self.directory/'summary.json').read_text())
        from .dense_helium_molecular_experiment import PHYSICS
        if self.metadata['physics']!=PHYSICS:
            raise ValueError('predictor physics do not match the active dense model')
        self.preserve_convection=preserve_convection
        self.reuse_pressure=reuse_pressure
        self.truncate_optical_depth=truncate_optical_depth
        self.refine_pressure=refine_pressure
        self.unscaled_temperature=unscaled_temperature
        self.sha=hashlib.sha256(Path(interaction_table).read_bytes()).hexdigest()
        if self.metadata['interaction_table_sha256']!=self.sha or not self.summary['converged']:
            raise ValueError('predictor requires a converged atmosphere with the identical interaction table')
        source=self.metadata['atmosphere']
        self.logg=float(self.metadata.get('logg',source.get('diagnostic_logg',8.)))
        from .heminus_join_experiment import METADATA_KEY,HARD_JOIN
        self.heminus_join_policy=source.get(METADATA_KEY,HARD_JOIN)
        if not (source['maximum_all_depth_total_flux_residual']<.003
                and source['maximum_relative_cell_energy_balance_residual']<.003
                and source['radiative_equilibrium_maximum_log_temperature_correction']<.0003):
            raise ValueError('predictor source failed the physical flux, local energy or correction gates')
        self.path=self.directory/'experimental-molecular-dense-structure.npz'
        self.file_sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path) as data:
            self.temperature=data['experimental_temperature'].copy()
            self.tau=data['experimental_tau'].copy()
            self.pressure=data['experimental_pressure'].copy()
        self.teff=float(self.directory.name)
        self.used=False
        grid='retained source pressure grid' if reuse_pressure else 'fresh target hydrostatic pressure grid'
        if truncate_optical_depth is not None:
            if not self.tau[0]<truncate_optical_depth<self.tau[-1]:
                raise ValueError('optical-depth truncation must lie within the saved domain')
            grid+=f'; explicitly truncated/resampled at source Rosseland depth {truncate_optical_depth:g}'
        self.description=(f'EXPLICIT same-physics temperature predictor from {self.directory}; '
            f'{grid}; '+('unscaled source T and absolute convective transport; '
                        if unscaled_temperature else 'scaled T/Teff predictor; ')
            +'NOT a cold start or converged target')

    def seed(self,teff,n_depth,bottom_tau,table=None,mesh='pressure',*,logg=8.):
        if logg != self.logg:
            raise ValueError('retained-pressure predictor requires the same recorded gravity')
        if table is not None:raise ValueError('no alternative EOS in continuation')
        if n_depth>len(self.pressure) and not self.refine_pressure:
            raise ValueError('refining the source pressure grid requires explicit --predictor-refine-pressure')
        from . import check_cool_db_transport_seed as runner
        nodes=np.arange(len(self.pressure),dtype=float)
        if self.truncate_optical_depth is None and n_depth<=len(nodes):
            indices=np.unique(np.rint(np.linspace(0,len(nodes)-1,n_depth)).astype(int))
            if len(indices)!=n_depth:raise ValueError('pressure predictor generated duplicate nodes')
        else:
            upper=(nodes[-1] if self.truncate_optical_depth is None else
                PchipInterpolator(np.log(self.tau),nodes)(np.log(self.truncate_optical_depth)))
            indices=np.linspace(0,upper,n_depth)
        interpolate=lambda values:np.exp(PchipInterpolator(nodes,np.log(values))(indices))
        return runner.atmosphere_at(teff,interpolate(self.pressure),
            interpolate(self.temperature)*(1. if self.unscaled_temperature else teff/self.teff),
            interpolate(self.tau),logg=logg)

    def __call__(self,seed,options):
        if self.used:return seed,options
        self.used=True
        temperatures=predicted_temperature(self.tau,self.temperature,self.teff,
            seed.rosseland_optical_depth,self.teff if self.unscaled_temperature else seed.effective_temperature)
        candidate=options['with_temperature'](temperatures)
        # The predictor may alter efficient convective slopes. Correct actual
        # overcarrying trials by changing T, never by replacing the ML2 flux.
        from . import check_cool_db_transport_seed as runner
        if self.preserve_convection:
            from .convective_mesh_prolongation import transport_profile,interface_log_pressure
            fraction=np.asarray(self.metadata['atmosphere']['convective_flux_fraction_by_interface'])
            desired=np.interp(interface_log_pressure(seed.rosseland_optical_depth),
                interface_log_pressure(self.tau),fraction)
            if self.unscaled_temperature:
                desired=desired*(self.teff/seed.effective_temperature)**4
            candidate,error=transport_profile(candidate,desired,runner,options,project_stable=True)
            projection=dict(preserved_convective_fraction=True,maximum_actual_fraction_error=error)
        else:
            from .exact_convective_trial_projection import project_trial
            candidate,projection=project_trial(candidate,runner,options)
        options={**options,'metadata':{**options.get('metadata',{}),
            'experimental_external_temperature_predictor':str(self.path),
            'experimental_external_temperature_predictor_sha256':self.file_sha,
            'experimental_temperature_predictor_teff':self.teff,
            'experimental_temperature_predictor_unscaled':self.unscaled_temperature,
            'experimental_temperature_predictor_is_cold_start':False,
            'experimental_temperature_predictor_is_convergence':False,
            'experimental_temperature_predictor_projection':projection}}
        print(self.description,flush=True)
        print(f'Predictor surface T={candidate.temperature[0]:.6g}, '
              f'bottom T={candidate.temperature[-1]:.6g}; projection={projection}',flush=True)
        return candidate,options
